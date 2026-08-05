#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
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


DEFAULT_OUT_PATH = PROJECT_ROOT / "governance" / "health" / "data_plane_recovery_controller_latest.json"
PAPER_STORAGE_PRESSURE_ADVISORY_CEILING = 0.50
PAPER_STORAGE_PRESSURE_TARGET = 0.25


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


def build_payload(project_root: Path = PROJECT_ROOT) -> dict[str, Any]:
    health_root = project_root / "governance" / "health"
    incident = load_json(health_root / "incident_timeline_latest.json")
    backlog_drain = load_json(health_root / "external_backlog_drain_latest.json")
    queue = load_json(health_root / "ingestion_priority_queue_latest.json")
    storage = load_json(health_root / "storage_tier_policy_latest.json")
    storage_control = load_json(health_root / "ingestion_storage_control_latest.json")
    runtime = load_json(health_root / "live_runtime_separation_control_latest.json")
    writer_progress = load_json(health_root / "sql_link_service_progress_latest.json")
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
    write_failure_count_raw = len(write_failures)
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
        and write_path_storage_ready
        and pending_lines <= 5000
        and writer_status in {"", "ok", "complete", "idle", "ready", "running", "busy"}
    )
    write_failure_count = 0 if write_path_recovered_by_storage else write_failure_count_raw
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

    return {
        "timestamp_utc": iso_now(),
        "schema_version": 1,
        "ok": recovery_state == "stable",
        "overall_status": overall_status,
        "recovery_state": recovery_state,
        "write_failure_count": write_failure_count,
        "raw_write_failure_count": write_failure_count_raw,
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
    args = parser.parse_args()

    payload = build_payload(Path(args.project_root).resolve())
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
