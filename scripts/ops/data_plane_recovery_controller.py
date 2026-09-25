#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gzip
import json
import math
import os
import re
import sys
import time
from contextlib import ExitStack
from collections import defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from scripts.ops.long_runtime_common import PROJECT_ROOT, iso_now, load_json, ordered_unique, write_payload
else:
    from .long_runtime_common import PROJECT_ROOT, iso_now, load_json, ordered_unique, write_payload

from core.storage_router import inspect_storage_path
from core.status_label_contract import evidence_label, read_label_source
from core.write_path_recovery import MAX_DOMAINS, MAX_RECORDS, digest, domain_id, recovery_pass


DEFAULT_OUT_PATH = PROJECT_ROOT / "governance" / "health" / "data_plane_recovery_controller_latest.json"
PAPER_STORAGE_PRESSURE_ADVISORY_CEILING = 0.50
PAPER_STORAGE_PRESSURE_TARGET = 0.25
WRITE_HISTORY_MAX_FILES = 64
WRITE_HISTORY_MAX_BYTES = 8 * 1024**2
WRITE_HISTORY_MAX_ROWS = 25000


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except Exception:
        return int(default)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _parse_dt(value: Any) -> datetime | None:
    if not value:
        return None
    text = str(value).strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _write_failure_history(project_root: Path, incident: dict[str, Any]) -> dict[str, Any]:
    """Count the timeline's two latest daily families before its display cap."""
    directory = project_root / "governance/events"
    route = inspect_storage_path(directory, boundary_root=project_root, allow_external=False)
    prior_path = project_root / "governance/health/data_plane_recovery_controller_latest.json"
    prior_route = inspect_storage_path(prior_path, boundary_root=project_root, allow_external=False)
    prior = load_json(Path(prior_route["resolved_path"])) if prior_route["status"] == "present" else {}
    previous = prior.get("write_failure_history")
    previous = previous if isinstance(previous, dict) else {}
    errors = []
    if prior_route["status"] not in {"present", "missing"}:
        errors.append({"path": str(prior_path), "route_status": prior_route["status"]})
    files = []
    scope_dates = []
    families = defaultdict(list)
    rows = []
    used_bytes = 0
    scanned_rows = 0
    now = datetime.now(timezone.utc)
    deadline = time.monotonic() + 8.0
    if route["status"] == "missing" and not previous and not errors:
        # Compatibility for older installations; never label this complete history.
        legacy = [row for row in (incident.get("recent_incidents") or [])
                  if isinstance(row, dict) and row.get("summary") == "write_failure"]
        count = max(len(legacy), _safe_int(prior.get("raw_write_failure_count"), 0))
        return {"source": "legacy_timeline", "complete": False, "count": count,
                "raw_event_count": len(legacy), "duplicate_count": 0,
                "errors": ([{"path": str(directory), "route_status": "missing"}]
                           if _safe_int(prior.get("raw_write_failure_count"), 0) > 0 else []),
                "coverage": "display_only_journal_unavailable"}
    if route["status"] != "present":
        errors.append({"path": str(directory), "route_status": route["status"]})
    else:
        try:
            with os.scandir(str(route["resolved_path"])) as entries:
                for index, entry in enumerate(entries):
                    if index >= 4096:
                        raise ValueError("write_history_directory_budget_exceeded")
                    match = re.fullmatch(r"write_failures_(\d{8})\.jsonl(?:\.raw-training)?(?:\.gz)?", entry.name)
                    if match:
                        day = datetime.strptime(match[1], "%Y%m%d").date()
                        if day > now.date():
                            errors.append({"path": str(directory / entry.name), "reason": "future_journal_date"})
                            continue
                        families[match[1]].append(directory / entry.name)
            scope_dates = sorted(families, reverse=True)[:2]
            files = [path for day in scope_dates for path in sorted(families[day])]
            if len(files) > WRITE_HISTORY_MAX_FILES:
                raise ValueError("write_history_file_budget_exceeded")
            for path in files:
                observation = inspect_storage_path(path, boundary_root=project_root, allow_external=False)
                if observation["status"] != "present" or observation.get("size_bytes") is None:
                    errors.append({"path": str(path), "route_status": observation["status"]})
                    continue
                with ExitStack() as stack:
                    raw = stack.enter_context(os.fdopen(os.open(str(observation["resolved_path"]), os.O_RDONLY | os.O_NOFOLLOW), "rb"))
                    before = os.fstat(raw.fileno())
                    handle = stack.enter_context(gzip.GzipFile(fileobj=raw)) if path.suffix == ".gz" else raw
                    while True:
                        if time.monotonic() >= deadline:
                            raise ValueError("write_history_time_budget_exceeded")
                        line = handle.readline(min(2 * 1024**2, WRITE_HISTORY_MAX_BYTES - used_bytes) + 1)
                        if not line:
                            break
                        used_bytes += len(line)
                        if used_bytes > WRITE_HISTORY_MAX_BYTES or len(line) > 2 * 1024**2:
                            raise ValueError("write_history_byte_budget_exceeded")
                        if not line.strip():
                            continue
                        scanned_rows += 1
                        if scanned_rows > WRITE_HISTORY_MAX_ROWS:
                            raise ValueError("write_history_row_budget_exceeded")
                        try:
                            row = json.loads(line)
                        except (ValueError, UnicodeDecodeError):
                            errors.append({"path": str(path), "reason": "invalid_jsonl_record"})
                            break
                        if not isinstance(row, dict):
                            errors.append({"path": str(path), "reason": "invalid_event_record"})
                            break
                        if (row.get("event") != "write_failure"
                                or not isinstance(row.get("timestamp_utc"), str)
                                or _parse_dt(row.get("timestamp_utc")) is None
                                or any(not isinstance(row.get(key), str) or not row[key].strip()
                                       for key in ("source", "target_path", "error"))):
                            errors.append({"path": str(path), "reason": "invalid_write_failure_schema"})
                            break
                        if _parse_dt(row["timestamp_utc"]) > now:
                            errors.append({"path": str(path), "reason": "future_write_failure"})
                            continue
                        rows.append(row)
                    after = os.fstat(raw.fileno())
                    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                        errors.append({"path": str(path), "reason": "journal_changed_during_read"})
        except (OSError, ValueError, EOFError) as exc:
            errors.append({"path": str(directory), "reason": str(exc)})

    # Collapse only exact receipts and matched inner/outer batch-error pairs.
    seen = set()
    unmatched = defaultdict(deque)
    count = 0
    recent_count = 0
    latest_failure = None
    domains = {}
    for row in sorted(rows, key=lambda item: str(item.get("timestamp_utc") or "")):
        timestamp = _parse_dt(row.get("timestamp_utc"))
        if timestamp is not None:
            latest_failure = max(latest_failure, timestamp) if latest_failure else timestamp
        encoded = json.dumps(row, sort_keys=True)
        if timestamp is not None and encoded in seen:
            continue
        seen.add(encoded)
        error = str(row.get("error") or "")
        identity = tuple(str(row.get(key) or "") for key in ("source", "run_id", "iter_id", "target_path"))
        if timestamp is not None and all(identity) and error in {"channel_batch_append_failed", "batch_write_failed"}:
            other = "batch_write_failed" if error == "channel_batch_append_failed" else "channel_batch_append_failed"
            pending = unmatched[(identity, other)]
            while pending and (timestamp - pending[0]).total_seconds() > 1:
                pending.popleft()
            if pending and 0 <= (timestamp - pending[0]).total_seconds() <= 1:
                pending.popleft()
                continue
            unmatched[(identity, error)].append(timestamp)
        count += 1
        day = timestamp.strftime("%Y%m%d") if timestamp else "unknown"
        key = domain_id(row["source"], row["target_path"], day)
        if key not in domains and len(domains) >= MAX_DOMAINS:
            errors.append({"path": str(directory), "reason": "write_history_domain_budget_exceeded"})
        else:
            domain = domains.setdefault(key, {
                "id": key, "source": row["source"], "target_path": row["target_path"],
                "day": day, "count": 0, "generation": "", "failed_records": [],
                "record_checkpoint_complete": True,
            })
            domain["count"] += 1
            domain["generation"] = digest([domain["generation"], row])
            domain["latest_failure_utc"] = timestamp.isoformat() if timestamp else None
            records = row.get("failed_records")
            valid_records = (isinstance(records, list) and bool(records)
                             and len(records) <= MAX_RECORDS
                             and all(isinstance(record, dict) and isinstance(record.get("message_id"), str)
                                     and record["message_id"] and isinstance(record.get("payload_sha256"), str)
                                     and re.fullmatch(r"[0-9a-f]{64}", record["payload_sha256"]) for record in records))
            if (row.get("record_checkpoint_complete") is not True or not valid_records
                    or len(domain["failed_records"]) + len(records) > MAX_RECORDS):
                domain["record_checkpoint_complete"] = False
            else:
                domain["failed_records"].extend(records)
        if timestamp is not None:
            recent_count += int(0 <= (now - timestamp).total_seconds() <= 900)
    observed_count = count
    previous_count = _safe_int(previous.get("count"), 0)
    if previous.get("scope_dates") != scope_dates and not errors and rows:
        previous_count = 0
    if not previous and (not rows or errors):
        previous_count = max(previous_count, _safe_int(prior.get("raw_write_failure_count"), 0))
    if count < previous_count:
        errors.append({"path": str(directory), "reason": "retained_history_count_regressed"})
        count = previous_count
    return {"source": "local_write_failure_journals", "complete": not errors,
            "count": count, "observed_unique_count": observed_count,
            "raw_event_count": len(rows), "duplicate_count": len(rows) - observed_count,
            "file_count": len(files), "bytes_read": used_bytes, "errors": errors,
            "scope_dates": scope_dates,
            "domains": list(domains.values()),
            "unmapped_prior_failure_count": (
                _safe_int(previous.get("count"), 0)
                if previous and not previous.get("domains") and previous.get("scope_dates") != scope_dates else 0
            ),
            "latest_observed_failure_utc": latest_failure.isoformat() if latest_failure else None,
            "recent_observed_failure_count": recent_count if not errors else None,
            "recent_window_seconds": 900,
            "affected_source_count": len({row["source"] for row in rows}),
            "affected_target_count": len({row["target_path"] for row in rows}),
            "coverage": "two_latest_daily_families_plain_and_gzip_with_bounded_reads",
            "historical_failures_are_not_current_sql_failure_counts": True}


def _effective_storage_backpressure(storage_control: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(storage_control, dict) or not storage_control:
        return {"authoritative": False}
    backpressure = storage_control.get("backpressure") if isinstance(storage_control.get("backpressure"), dict) else {}
    effective = backpressure.get("effective_raw_live") if isinstance(backpressure.get("effective_raw_live"), dict) else {}
    raw_live = backpressure.get("raw_live") if isinstance(backpressure.get("raw_live"), dict) else {}
    data_integrity = storage_control.get("data_integrity") if isinstance(storage_control.get("data_integrity"), dict) else {}
    source = str(backpressure.get("effective_raw_live_source") or effective.get("source") or "").strip()
    severity = str(storage_control.get("severity") or "").strip().lower()
    pressure_index = _safe_float(storage_control.get("pressure_index"), 0.0)
    storage_ready = bool(
        str(storage_control.get("overall_status") or "").strip().lower() == "ready"
        and severity == "stable"
    )
    stable_overlay_truth = bool(
        severity in {"", "stable", "ready"}
        and pressure_index < PAPER_STORAGE_PRESSURE_ADVISORY_CEILING
        and effective
    )
    stable_raw_live_truth = bool(
        storage_ready
        and effective
        and source not in {"fresh_empty_sql_ingestion_overlay", "sql_ingestion_overlay"}
    )
    overlay_clear = bool(backpressure.get("overlay_pressure_clear", False) or source == "fresh_empty_sql_ingestion_overlay")
    data_clean = bool(
        _safe_int(data_integrity.get("sql_overlay_invalid_lines"), 0) <= 0
        and _safe_int(data_integrity.get("sql_overlay_oversize_payloads"), 0) <= 0
        and _safe_int(data_integrity.get("sql_overlay_ops_write_failures"), 0) <= 0
    )
    authoritative = bool(
        data_clean
        and (
            stable_raw_live_truth
            or (
                (storage_ready or stable_overlay_truth)
                and bool(backpressure.get("overlay_adjusted", False))
                and overlay_clear
            )
        )
    )
    if not authoritative:
        return {
            "authoritative": False,
            "source": source,
            "storage_ready": storage_ready,
            "stable_overlay_truth": stable_overlay_truth,
            "stable_raw_live_truth": stable_raw_live_truth,
            "pressure_index": round(pressure_index, 3),
            "overlay_clear": overlay_clear,
            "data_clean": data_clean,
        }
    total = _safe_int(effective.get("total_pending_lines"), _safe_int(backpressure.get("total_pending_lines"), 0))
    core = _safe_int(effective.get("core_pending_lines"), _safe_int(backpressure.get("core_pending_lines"), total))
    oldest = _safe_float(effective.get("oldest_pending_age_seconds"), _safe_float(backpressure.get("oldest_pending_age_seconds"), 0.0))
    return {
        "authoritative": True,
        "source": source or "ingestion_storage_control_effective_raw_live",
        "core_pending_lines": int(core),
        "total_pending_lines": int(total),
        "oldest_pending_age_seconds": round(oldest, 3),
        "storage_ready": storage_ready,
        "stable_overlay_truth": stable_overlay_truth,
        "stable_raw_live_truth": stable_raw_live_truth,
        "pressure_index": round(pressure_index, 3),
        "overlay_clear": overlay_clear,
        "data_clean": data_clean,
        "raw_live_estimate": raw_live,
    }


def _current_storage_write_recovery(storage_control: dict[str, Any], *, pending_lines: int, writer_status: str) -> dict[str, Any]:
    steady_state = storage_control.get("steady_state") if isinstance(storage_control.get("steady_state"), dict) else {}
    target_status = steady_state.get("target_status") if isinstance(steady_state.get("target_status"), dict) else {}
    external_route = (
        storage_control.get("external_route_verification")
        if isinstance(storage_control.get("external_route_verification"), dict)
        else {}
    )
    backpressure = storage_control.get("backpressure") if isinstance(storage_control.get("backpressure"), dict) else {}
    effective_backpressure = _effective_storage_backpressure(storage_control)
    raw_live = (
        backpressure.get("effective_raw_live")
        if bool(effective_backpressure.get("authoritative", False))
        and isinstance(backpressure.get("effective_raw_live"), dict)
        else backpressure.get("raw_live")
        if isinstance(backpressure.get("raw_live"), dict)
        else backpressure
    )
    data_integrity = storage_control.get("data_integrity") if isinstance(storage_control.get("data_integrity"), dict) else {}
    route_state = str(external_route.get("verification_state") or "").strip().lower()
    storage_status = str(storage_control.get("overall_status") or "").strip().lower()
    severity = str(storage_control.get("severity") or "").strip().lower()
    pressure_index = _safe_float(storage_control.get("pressure_index"), 0.0)
    backpressure_quality_score = _safe_float(storage_control.get("backpressure_quality_score"), 0.0)
    raw_core = _safe_int(raw_live.get("core_pending_lines"), _safe_int(backpressure.get("core_pending_lines"), 0))
    raw_total = _safe_int(raw_live.get("total_pending_lines"), _safe_int(backpressure.get("total_pending_lines"), pending_lines))
    raw_oldest = _safe_float(raw_live.get("oldest_pending_age_seconds", backpressure.get("oldest_pending_age_seconds", 0.0)), 0.0)
    overlay_adjusted = bool(backpressure.get("overlay_adjusted", False))
    overlay_total = _safe_int(backpressure.get("total_pending_lines"), raw_total)
    overlay_oldest = _safe_float(backpressure.get("oldest_pending_age_seconds"), raw_oldest)
    current_sql_write_failures = _safe_int(data_integrity.get("sql_overlay_ops_write_failures"), 0)
    route_ready = route_state in {"ready", "verified", "curated_ready", "active_passthrough", "active_local_ready"}
    raw_live_clear = bool(raw_core <= 5000 and raw_total <= 15000 and raw_oldest <= 15 * 60)
    target_ready = bool(target_status.get("steady_state_ready", False))
    bounded_target_relief = bool(
        not target_ready
        and storage_status == "ready"
        and severity == "stable"
        and pressure_index < PAPER_STORAGE_PRESSURE_ADVISORY_CEILING
        and backpressure_quality_score >= 95
        and raw_live_clear
        and route_ready
    )
    overlay_only_write_relief = bool(
        overlay_adjusted
        and raw_live_clear
        and route_ready
        and current_sql_write_failures <= 0
        and pending_lines <= 5000
        and overlay_total <= 12000
    )
    writer_ready = writer_status in {"", "ok", "complete", "idle", "ready", "running", "busy"}
    current_storage_ready = bool(
        writer_ready
        and (
            (
                storage_status == "ready"
                and severity == "stable"
                and (target_ready or bounded_target_relief)
                and route_ready
                and raw_live_clear
                and current_sql_write_failures <= 0
                and pending_lines <= 5000
                and backpressure_quality_score >= 95
            )
            or overlay_only_write_relief
        )
    )
    return {
        "ready": current_storage_ready,
        "storage_status": storage_status,
        "severity": severity,
        "pressure_index": round(pressure_index, 3),
        "pressure_target": PAPER_STORAGE_PRESSURE_TARGET,
        "pressure_advisory_ceiling": PAPER_STORAGE_PRESSURE_ADVISORY_CEILING,
        "target_ready": target_ready,
        "bounded_target_relief": bounded_target_relief,
        "overlay_only_write_relief": overlay_only_write_relief,
        "overlay": {
            "overlay_adjusted": overlay_adjusted,
            "total_pending_lines": overlay_total,
            "oldest_pending_age_seconds": round(overlay_oldest, 3),
            "max_total_pending_lines": 12000,
        },
        "route_ready": route_ready,
        "route_state": route_state,
        "raw_live_clear": raw_live_clear,
        "raw_live": {
            "core_pending_lines": raw_core,
            "total_pending_lines": raw_total,
            "oldest_pending_age_seconds": round(raw_oldest, 3),
            "max_core_pending_lines": 5000,
            "max_total_pending_lines": 15000,
            "max_oldest_pending_age_seconds": 15 * 60,
        },
        "effective_backpressure": effective_backpressure,
        "current_sql_write_failures": current_sql_write_failures,
        "writer_status": writer_status,
        "policy": "historical write failures are recovered when current storage truth, raw live backlog, route verification, and SQL writer state are clean; slight stable pressure-target misses are advisory for paper",
    }


def build_payload(project_root: Path = PROJECT_ROOT, *, apply: bool = False) -> dict[str, Any]:
    health_root = project_root / "governance" / "health"
    incident = load_json(health_root / "incident_timeline_latest.json")
    backlog_drain = load_json(health_root / "external_backlog_drain_latest.json")
    queue = load_json(health_root / "ingestion_priority_queue_latest.json")
    storage = load_json(health_root / "storage_tier_policy_latest.json")
    storage_control = load_json(health_root / "ingestion_storage_control_latest.json")
    runtime = load_json(health_root / "live_runtime_separation_control_latest.json")
    writer_progress = load_json(health_root / "sql_link_service_progress_latest.json")
    writer_observation = read_label_source(project_root, "governance/health/sql_writer_observation_latest.json")
    snapshot_cache = load_json(health_root / "broker_truth_shared_snapshot_schwab_latest.json")

    recent = incident.get("recent_incidents") if isinstance(incident.get("recent_incidents"), list) else []
    write_failures = [
        row
        for row in recent
        if isinstance(row, dict) and str(row.get("summary") or "").strip().lower() == "write_failure"
    ]
    account_snapshot_failures = [
        row
        for row in recent
        if isinstance(row, dict) and str(row.get("summary") or "").strip().lower() == "get_accounts_snapshot"
    ]
    write_history = _write_failure_history(project_root, incident)
    try:
        path_recovery = recovery_pass(project_root, write_history, apply=apply)
    except (OSError, ValueError, TypeError, KeyError) as exc:
        path_recovery = {"overall_status": "blocked", "reason": f"recovery_state_unavailable:{type(exc).__name__}", "apply_requested": apply}
    write_failure_count_raw = _safe_int(write_history.get("count"), len(write_failures))
    account_snapshot_count_raw = len(account_snapshot_failures)
    raw_pending_lines = _safe_int((queue.get("lane_counts") or {}).get("core", {}).get("pending_lines", queue.get("queue_depth", 0)), 0)
    effective_backpressure = _effective_storage_backpressure(storage_control)
    pending_lines = (
        _safe_int(effective_backpressure.get("total_pending_lines"), 0)
        if bool(effective_backpressure.get("authoritative", False))
        else raw_pending_lines
    )
    drain_status = str(backlog_drain.get("overall_status") or "").strip().lower()
    runtime_clearance = str(((runtime.get("clearance_plan") or {}).get("clearance_state") or "")).strip().lower()
    hot_path_over_budget_raw = _safe_int(((storage.get("pressure") or {}).get("hot_path_over_budget_bytes", 0)), 0)
    storage_target_status = storage_control.get("steady_state", {}).get("target_status", {}) if isinstance(storage_control.get("steady_state"), dict) else {}
    external_route = storage_control.get("external_route_verification") if isinstance(storage_control.get("external_route_verification"), dict) else {}
    storage_steady_state_ready = bool(
        str(storage_control.get("overall_status") or "").strip().lower() == "ready"
        and str(storage_control.get("severity") or "").strip().lower() == "stable"
        and bool(storage_target_status.get("steady_state_ready", False))
        and _safe_int(storage_control.get("backpressure_quality_score"), 0) >= 95
        and _safe_int(storage_control.get("recovery_quality_score"), 0) >= 88
        and str(external_route.get("verification_state") or "").strip().lower()
        in {"ready", "verified", "curated_ready", "active_passthrough", "active_local_ready"}
    )
    writer_status = str(writer_progress.get("status") or "").strip().lower()
    writer_busy = writer_status in {"running", "busy"}
    current_storage_write_recovery = _current_storage_write_recovery(
        storage_control,
        pending_lines=pending_lines,
        writer_status=writer_status,
    )
    write_path_storage_ready = bool(storage_steady_state_ready or current_storage_write_recovery.get("ready", False))
    hot_path_over_budget = 0 if write_path_storage_ready else hot_path_over_budget_raw
    write_path_recovered_by_storage = bool(
        write_failure_count_raw > 0
        and write_history["source"] == "legacy_timeline"
        and not write_history["errors"]
        and write_path_storage_ready
        and pending_lines <= 5000
        and writer_status in {"", "ok", "complete", "idle", "ready", "running", "busy"}
    )
    write_failure_count = 0 if write_path_recovered_by_storage else write_failure_count_raw
    path_state = path_recovery.get("state") or {}
    reconciled = {d["id"]: d for d in path_state.get("domains", []) if d.get("historical_reconciled")}
    verified_count = sum(d["count"] for d in write_history.get("domains", [])
                         if reconciled.get(d["id"], {}).get("generation") == d["generation"])
    write_failure_count = max(0, write_failure_count - verified_count)
    # Never let the two-day journal display retire retained recovery debt.
    write_failure_count = max(write_failure_count, _safe_int(path_state.get("unreconciled_failure_count"), 0))
    snapshot_cache_ready = bool(snapshot_cache.get("fetched")) and bool(snapshot_cache.get("timestamp_utc"))
    snapshot_cache_ts = _parse_dt(snapshot_cache.get("timestamp_utc"))
    snapshot_failure_times = [
        parsed
        for parsed in (_parse_dt(row.get("timestamp_utc")) for row in account_snapshot_failures)
        if parsed is not None
    ]
    last_snapshot_failure_ts = max(snapshot_failure_times) if snapshot_failure_times else None
    snapshot_recovered_by_cache = bool(
        account_snapshot_count_raw > 0
        and snapshot_cache_ready
        and snapshot_cache_ts is not None
        and last_snapshot_failure_ts is not None
        and snapshot_cache_ts >= last_snapshot_failure_ts
    )
    account_snapshot_count = 0 if snapshot_recovered_by_cache else account_snapshot_count_raw
    drain_delta = backlog_drain.get("drain_delta") if isinstance(backlog_drain.get("drain_delta"), dict) else {}
    follow_through = backlog_drain.get("follow_through") if isinstance(backlog_drain.get("follow_through"), dict) else {}
    blocked_reasons = [str(item).strip().lower() for item in (backlog_drain.get("blocked_reasons") or []) if str(item).strip()]
    drain_progress_lines = _safe_int(drain_delta.get("total_pending_lines"), 0)
    drain_apply_requested = bool(backlog_drain.get("apply_requested", False))
    market_hours_guard = "market_hours_guard" in blocked_reasons
    follow_through_status = str(follow_through.get("status") or "").strip().lower()
    active_recovery = bool(writer_busy or drain_apply_requested or drain_progress_lines > 0 or follow_through_status in {"running", "waiting_for_writer", "polling"})
    small_steady_queue = bool(
        write_path_storage_ready
        and write_failure_count <= 0
        and account_snapshot_count <= 0
        and pending_lines <= 5000
    )

    recovery_state = "stable"
    if write_failure_count > 0 or account_snapshot_count > 0:
        recovery_state = "needs_recovery"
    if (drain_status == "blocked" and not small_steady_queue) or hot_path_over_budget > 0:
        recovery_state = "recovering_under_guard" if active_recovery else "blocked"
    elif active_recovery and recovery_state != "stable":
        recovery_state = "recovering_under_guard"
    elif pending_lines > 0 and writer_busy and not small_steady_queue:
        recovery_state = "recovering_under_guard"
    if write_history["errors"] or path_recovery.get("overall_status") == "blocked":
        recovery_state = "blocked"

    overall_status = "ready"
    if recovery_state == "blocked":
        overall_status = "blocked"
    elif recovery_state != "stable":
        overall_status = "degraded"

    recommended_actions = ordered_unique(
        [
            "drain deferred and cold backlogs before reopening write-heavy ingestion lanes" if recovery_state != "stable" else "",
            "treat repeated get_accounts_snapshot failures as a broker-side gating signal before thawing execution sleeves" if account_snapshot_count > 0 else "",
            "route write-heavy reconciliation through the single-writer service until hot-path pressure clears" if hot_path_over_budget > 0 else "",
            "use the shared broker snapshot cache as a stale fallback before reopening thaw candidates" if snapshot_cache_ready and account_snapshot_count > 0 else "",
            "keep the live lane read-only while the data plane catches up" if runtime_clearance not in {"ready", ""} and recovery_state != "stable" else "",
            "let the off-hours backlog drain fire on schedule instead of forcing deferred and cold replay work into market hours" if market_hours_guard else "",
        ]
    )

    storage_label = evidence_label(
        storage_control, scope="current_storage_write_health",
        source="governance/health/ingestion_storage_control_latest.json", max_age_seconds=300,
    )
    writer_label = evidence_label(
        writer_progress, scope="writer_progress",
        source="governance/health/sql_link_service_progress_latest.json", max_age_seconds=300,
    )
    data_integrity = storage_control.get("data_integrity")
    data_integrity = data_integrity if isinstance(data_integrity, dict) else {}
    sql_errors = data_integrity.get("sql_overlay_ops_write_failures")
    overlay = storage_control.get("sql_ingestion_pending_overlay")
    overlay = overlay if isinstance(overlay, dict) else {}
    overlay_age = overlay.get("max_source_age_seconds")
    fresh_sql_sources = _safe_int(overlay.get("fresh_source_count"), 0)
    stale_sql_sources = _safe_int(overlay.get("stale_source_count"), 0)
    current_sql_errors_known = (
        storage_label["fresh"] and isinstance(sql_errors, (int, float))
        and not isinstance(sql_errors, bool) and math.isfinite(sql_errors)
        and sql_errors >= 0 and float(sql_errors).is_integer()
        and fresh_sql_sources > 0 and isinstance(overlay_age, (int, float))
        and not isinstance(overlay_age, bool) and math.isfinite(overlay_age)
        and 0 <= overlay_age <= 300
    )
    recovery_gaps = []
    if not write_history.get("complete"):
        recovery_gaps.append("failure_history_scan_incomplete")
    if not storage_label["fresh"]:
        recovery_gaps.append("fresh_storage_evidence_required")
    if not writer_label["fresh"]:
        recovery_gaps.append("fresh_writer_progress_required")
    if not current_sql_errors_known:
        recovery_gaps.append("current_sql_failure_measurement_unavailable")
    elif sql_errors > 0:
        recovery_gaps.append("current_sql_write_failures_present")
    if not write_path_storage_ready:
        recovery_gaps.append("storage_write_recovery_checks_unmet")
    if pending_lines > 5000:
        recovery_gaps.append("pending_ingestion_above_recovery_gate")
    if writer_status not in {"ok", "complete", "idle", "ready", "running", "busy"}:
        recovery_gaps.append("writer_not_in_accepted_recovery_state")
    if write_failure_count > 0:
        recovery_gaps.append("historical_failure_reconciliation_not_verified")

    return {
        "timestamp_utc": iso_now(),
        "schema_version": 1,
        "ok": recovery_state == "stable",
        "overall_status": overall_status,
        "recovery_state": recovery_state,
        "write_failure_count": write_failure_count,
        "raw_write_failure_count": write_failure_count_raw,
        "write_failure_count_scope": "unreconciled_historical_failure_debt_not_current_sql_errors",
        "recovery_diagnostics": {
            "historical_failure_count": write_failure_count_raw,
            "unreconciled_historical_failure_count": write_failure_count,
            "recent_observed_failure_count": write_history.get("recent_observed_failure_count"),
            "recent_window_seconds": write_history.get("recent_window_seconds"),
            "latest_observed_failure_utc": write_history.get("latest_observed_failure_utc"),
            "current_sql_write_failure_count": sql_errors if current_sql_errors_known else None,
            "current_sql_measurement_available": current_sql_errors_known,
            "sql_failure_measurement_scope": "producer_reported_fresh_overlay_sources_not_all_write_paths",
            "sql_overlay_fresh_source_count": fresh_sql_sources,
            "sql_overlay_stale_source_count": stale_sql_sources,
            "sql_overlay_coverage": (
                "unavailable" if not current_sql_errors_known
                else "partial" if stale_sql_sources or _safe_int(overlay.get("fresh_pending_unknown_source_count"), 0)
                else "reported_sources_only"
            ),
            "reported_sql_overlay_failure_count": sql_errors,
            "storage_evidence": storage_label,
            "writer_evidence": writer_label,
            "writer_admission_observation": evidence_label(
                writer_observation, scope="writer_admission_not_progress",
                source="governance/health/sql_writer_observation_latest.json", max_age_seconds=180,
            ),
            "writer_deferral_reasons": writer_observation.get("blockers", []),
            "unmet_requirements": recovery_gaps,
            "authority": "diagnostic_only_no_recovery_release",
            "quiet_period_alone_proves_recovery": False,
        },
        "write_failure_history": write_history,
        "native_write_path_recovery": path_recovery,
        "checkpoint_reconciled_failure_count": verified_count,
        "write_path_recovered_by_storage": write_path_recovered_by_storage,
        "account_snapshot_failure_count": account_snapshot_count,
        "raw_account_snapshot_failure_count": account_snapshot_count_raw,
        "queue_depth": pending_lines,
        "raw_queue_depth": raw_pending_lines,
        "queue_depth_source": (
            str(effective_backpressure.get("source") or "ingestion_storage_control_effective_raw_live")
            if bool(effective_backpressure.get("authoritative", False))
            else "ingestion_priority_queue"
        ),
        "external_backlog_status": drain_status,
        "runtime_clearance_state": runtime_clearance,
        "hot_path_over_budget_bytes": hot_path_over_budget,
        "raw_hot_path_over_budget_bytes": hot_path_over_budget_raw,
        "storage_steady_state_ready": bool(storage_steady_state_ready),
        "current_storage_write_ready": bool(current_storage_write_recovery.get("ready", False)),
        "small_steady_queue": bool(small_steady_queue),
        "write_path_recovery_evidence": current_storage_write_recovery,
        "recovery_contract": {
            "backlog_drain_required": write_failure_count > 0 or pending_lines > 0,
            "writer_handoff_required": hot_path_over_budget > 0,
            "writer_service_active": writer_busy,
            "write_path_recovered_by_storage": write_path_recovered_by_storage,
            "current_storage_write_ready": bool(current_storage_write_recovery.get("ready", False)),
            "execution_lane_pause_required": account_snapshot_count > 0,
            "recommended_command": [
                "./scripts/ops/opsctl.sh",
                "external-backlog-drain",
                "--json",
            ],
            "writer_cycle_command": [
                "./scripts/ops/opsctl.sh",
                "writer-cycle-coordinator",
                "--json",
            ],
            "snapshot_probe_required": account_snapshot_count > 0,
            "snapshot_cache_ready": snapshot_cache_ready,
            "snapshot_recovered_by_cache": snapshot_recovered_by_cache,
            "snapshot_probe_command": [
                "./scripts/ops/opsctl.sh",
                "token-refresh",
                "--json",
            ],
        },
        "backlog_recovery_contract": {
            "apply_requested": drain_apply_requested,
            "market_hours_guard": market_hours_guard,
            "blocked_reasons": blocked_reasons,
            "drain_progress_lines": drain_progress_lines,
            "follow_through_status": follow_through_status,
            "progress_observed": bool(drain_progress_lines > 0 or bool(follow_through.get("progress_observed", False))),
            "off_hours_window": backlog_drain.get("off_hours_window") if isinstance(backlog_drain.get("off_hours_window"), dict) else {},
        },
        "writer_handoff_contract": {
            "service_status": str(writer_progress.get("status") or ""),
            "service_current_step": str(writer_progress.get("current_step") or ""),
            "writer_service_active": writer_busy,
            "hot_path_over_budget_bytes": hot_path_over_budget,
            "raw_hot_path_over_budget_bytes": hot_path_over_budget_raw,
            "storage_steady_state_ready": bool(storage_steady_state_ready),
            "preferred_mode": "single_writer_service" if hot_path_over_budget > 0 else "standard",
            "handoff_progress_state": ("active" if writer_busy else "idle"),
        },
        "snapshot_recovery_contract": {
            "cache_ready": snapshot_cache_ready,
            "cache_timestamp_utc": str(snapshot_cache.get("timestamp_utc") or ""),
            "last_failure_timestamp_utc": last_snapshot_failure_ts.isoformat() if last_snapshot_failure_ts else "",
            "recovered_by_fresh_cache": snapshot_recovered_by_cache,
            "stale_fallback_allowed": snapshot_cache_ready,
            "bounded_retry_count": 2 if account_snapshot_count > 0 else 0,
            "probe_required": account_snapshot_count > 0,
        },
        "recommended_actions": recommended_actions,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Publish a bounded recovery contract for repeated write-failure and account-snapshot incidents.")
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--out-file", default=str(DEFAULT_OUT_PATH))
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--apply", action="store_true", help="Advance bounded native verification and probation; never replay trades or override admission.")
    args = parser.parse_args()

    payload = build_payload(Path(args.project_root).resolve(), apply=args.apply)
    out_path = Path(args.out_file).expanduser()
    write_payload(out_path, payload)
    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        print(
            "data_plane_recovery_controller "
            f"overall_status={payload.get('overall_status', '')} "
            f"write_failure_count={int(payload.get('write_failure_count', 0) or 0)}"
        )
    return 0 if payload.get("overall_status") in {"ready", "degraded"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
