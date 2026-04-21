#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time as time_mod
from datetime import datetime, time, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.runtime_python import resolve_runtime_python
from scripts.ops import ingestion_storage_governor as governor_src


PY = resolve_runtime_python(PROJECT_ROOT)
DEFAULT_OUT_PATH = PROJECT_ROOT / "governance" / "health" / "external_backlog_drain_latest.json"
SERVICE_REQUEST_PATH = PROJECT_ROOT / "governance" / "health" / "sql_link_service_request_latest.json"
LOCAL_TZ = ZoneInfo("America/New_York")
OFF_HOURS_START = time(16, 15)
OFF_HOURS_END = time(9, 20)
SQL_WRITER_LOCK_PATH = PROJECT_ROOT / "governance" / "locks" / "jsonl_sql_writer.lock"


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _parse_json_output(text: str) -> dict[str, Any]:
    for line in reversed([raw.strip() for raw in str(text or "").splitlines() if raw.strip()]):
        try:
            payload = json.loads(line)
        except Exception:
            continue
        if isinstance(payload, dict):
            return payload
    return {}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def _safe_int(raw: Any, default: int = 0) -> int:
    try:
        return int(float(raw))
    except Exception:
        return int(default)


def _safe_float(raw: Any, default: float = 0.0) -> float:
    try:
        return float(raw)
    except Exception:
        return float(default)


def _age_bucket(age_seconds: float) -> str:
    age = max(float(age_seconds), 0.0)
    if age < 30 * 60:
        return "fresh_lt_30m"
    if age < 2 * 60 * 60:
        return "aging_lt_2h"
    if age < 6 * 60 * 60:
        return "stale_lt_6h"
    if age < 24 * 60 * 60:
        return "stale_lt_24h"
    return "cold_gte_24h"


def _lock_owner_pid(lock_path: Path) -> int | None:
    try:
        raw = lock_path.read_text(encoding="utf-8").strip()
    except Exception:
        return None
    for token in raw.split():
        if not token.startswith("pid="):
            continue
        try:
            return int(token.split("=", 1)[1])
        except Exception:
            return None
    return None


def _off_hours_window(now_utc: datetime) -> dict[str, Any]:
    local_now = now_utc.astimezone(LOCAL_TZ)
    local_clock = local_now.timetz().replace(tzinfo=None)
    is_weekend = local_now.weekday() >= 5
    active = bool(is_weekend or local_clock >= OFF_HOURS_START or local_clock < OFF_HOURS_END)
    return {
        "active": active,
        "is_weekend": is_weekend,
        "timezone": "America/New_York",
        "local_time": local_now.isoformat(),
        "window_start_local": OFF_HOURS_START.strftime("%H:%M"),
        "window_end_local": OFF_HOURS_END.strftime("%H:%M"),
        "label": "off_hours" if active else "market_hours",
    }


def _run_json_command(
    cmd: list[str],
    *,
    cwd: Path,
    payload_path: Path | None = None,
    env_overrides: dict[str, str] | None = None,
    timeout_seconds: float | None = None,
) -> dict[str, Any]:
    env = os.environ.copy()
    if env_overrides:
        env.update({str(key): str(value) for key, value in env_overrides.items()})
    started = datetime.now(timezone.utc)
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(cwd),
            capture_output=True,
            text=True,
            check=False,
            env=env,
            timeout=timeout_seconds,
        )
        stdout = proc.stdout or ""
        stderr = proc.stderr or ""
        rc = int(proc.returncode)
        timed_out = False
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout.decode("utf-8", errors="ignore") if isinstance(exc.stdout, bytes) else str(exc.stdout or "")
        stderr = exc.stderr.decode("utf-8", errors="ignore") if isinstance(exc.stderr, bytes) else str(exc.stderr or "")
        rc = 124
        timed_out = True
    payload = _parse_json_output(stdout)
    if not payload and payload_path is not None:
        payload = _load_json(payload_path)
    if timed_out:
        payload = {**payload, "ok": False, "reason": "timeout", "timed_out": True}
    duration_ms = round((datetime.now(timezone.utc) - started).total_seconds() * 1000.0, 3)
    return {
        "cmd": list(cmd),
        "rc": rc,
        "duration_ms": duration_ms,
        "payload": payload,
        "stdout_tail": "\n".join(stdout.splitlines()[-12:]),
        "stderr_tail": "\n".join(stderr.splitlines()[-12:]),
        "timed_out": timed_out,
    }


def _step_status(result: dict[str, Any], *, nonfatal_reasons: set[str] | None = None) -> str:
    if bool(result.get("timed_out", False)):
        return "busy"
    if int(result.get("rc", 1)) != 0:
        return "error"
    payload = result.get("payload") if isinstance(result.get("payload"), dict) else {}
    reason = str(payload.get("reason") or "")
    if bool(payload.get("busy", False)) or reason in (nonfatal_reasons or set()):
        return "busy"
    if payload.get("ok") is False:
        return "error"
    return "ok"


def _step_record(result: dict[str, Any], *, nonfatal_reasons: set[str] | None = None) -> dict[str, Any]:
    return {
        "status": _step_status(result, nonfatal_reasons=nonfatal_reasons),
        "rc": int(result.get("rc", 1)),
        "duration_ms": float(result.get("duration_ms", 0.0) or 0.0),
        "timed_out": bool(result.get("timed_out", False)),
        "cmd": list(result.get("cmd") or []),
        "stdout_tail": str(result.get("stdout_tail") or ""),
        "stderr_tail": str(result.get("stderr_tail") or ""),
    }


def _drain_env(base_env: dict[str, str], *, critical: bool, off_hours_active: bool) -> tuple[str, dict[str, str]]:
    env = {str(key): str(value) for key, value in base_env.items() if str(key).strip()}
    if not off_hours_active:
        return "standard_guard", env

    env.update(
        {
            "INGEST_MAX_DEFERRED_FILES": "6" if critical else "4",
            "JSONL_SQL_MAX_COLD_LANE_FILES": "2" if critical else "1",
            "LOG_DATA_INGRESS": "0",
            "LOG_API_CALLS": "0",
            "LOG_LOOP_STATE": "0",
            "LOG_SHADOW_PNL_ATTRIBUTION": "0",
            "SQL_LINK_SERVICE_INTERVAL_SECONDS": "12" if critical else "15",
            "SQL_LINK_SERVICE_SHARDS": (
                "health_fast,trading_fast,crypto_trading_fast,runtime,crypto_runtime,"
                "aggressive_trading,trading,governance,support_watchdog,crypto_governance,data,"
                "shadow_attribution,crypto_shadow_attribution"
            ),
            "SQL_LINK_SERVICE_MERGE_MAX_SECONDS_PER_CYCLE": "25" if critical else "45",
            "SQL_LINK_SERVICE_HOT_MIN_INTERVAL_SECONDS": "30",
            "SQL_LINK_SERVICE_HOT_BATCH_SIZE": "240000" if critical else "200000",
            "SQL_LINK_SERVICE_HOT_MAX_ROWS": "2400000" if critical else "1800000",
            "SQL_LINK_SERVICE_WAL_CHECKPOINT_THRESHOLD_GB": "0.25" if critical else "0.5",
            "SQL_LINK_SERVICE_WAL_CHECKPOINT_TRIGGER_GROWTH_GB": "0.25" if critical else "0.5",
            "SQL_LINK_SERVICE_SHARD_RUNTIME_MAX_FILES": "14" if critical else "12",
            "SQL_LINK_SERVICE_SHARD_CRYPTO_RUNTIME_MAX_FILES": "10" if critical else "8",
            "SQL_LINK_SERVICE_SHARD_GOVERNANCE_MAX_FILES": "8" if critical else "6",
            "SQL_LINK_SERVICE_SHARD_AGGRESSIVE_TRADING_MAX_FILES": "14" if critical else "12",
            "SQL_LINK_SERVICE_SHARD_TRADING_MAX_FILES": "14" if critical else "12",
            "SQL_LINK_SERVICE_SHARD_RUNTIME_STATE_CHECKPOINT_LINES": "1500",
            "SQL_LINK_SERVICE_SHARD_CRYPTO_RUNTIME_STATE_CHECKPOINT_LINES": "1500",
            "SQL_LINK_SERVICE_SHARD_GOVERNANCE_STATE_CHECKPOINT_LINES": "1500",
            "SQL_LINK_SERVICE_SHARD_AGGRESSIVE_TRADING_STATE_CHECKPOINT_LINES": "1500",
            "SQL_LINK_SERVICE_SHARD_TRADING_STATE_CHECKPOINT_LINES": "1500",
            "SQL_LINK_SERVICE_SHARD_RUNTIME_MAX_LINES_PER_FILE": "16000",
            "SQL_LINK_SERVICE_SHARD_CRYPTO_RUNTIME_MAX_LINES_PER_FILE": "16000",
            "SQL_LINK_SERVICE_SHARD_GOVERNANCE_MAX_LINES_PER_FILE": "4000" if critical else "6000",
            "SQL_LINK_SERVICE_SHARD_SUPPORT_WATCHDOG_MAX_LINES_PER_FILE": "96000" if critical else "64000",
            "SQL_LINK_SERVICE_SHARD_SUPPORT_WATCHDOG_STATE_CHECKPOINT_LINES": "4000",
            "SQL_LINK_SERVICE_SHARD_AGGRESSIVE_TRADING_MAX_LINES_PER_FILE": "16000" if critical else "14000",
            "SQL_LINK_SERVICE_SHARD_TRADING_MAX_LINES_PER_FILE": "16000" if critical else "14000",
            "SQL_LINK_SERVICE_SHARD_EXPLANATIONS_MAX_FILES": "6" if critical else "5",
            "SQL_LINK_SERVICE_SHARD_CRYPTO_EXPLANATIONS_MAX_FILES": "6" if critical else "5",
            "SQL_LINK_SERVICE_SHARD_SHADOW_ATTRIBUTION_MAX_FILES": "3" if critical else "2",
            "SQL_LINK_SERVICE_SHARD_CRYPTO_SHADOW_ATTRIBUTION_MAX_FILES": "3" if critical else "2",
            "SQL_LINK_SERVICE_SHARD_EXPLANATIONS_HOT_RETENTION_BATCH_SIZE": "240000" if critical else "180000",
            "SQL_LINK_SERVICE_SHARD_CRYPTO_EXPLANATIONS_HOT_RETENTION_BATCH_SIZE": "220000" if critical else "160000",
            "SQL_LINK_SERVICE_SHARD_SHADOW_ATTRIBUTION_HOT_RETENTION_BATCH_SIZE": "260000" if critical else "220000",
            "SQL_LINK_SERVICE_SHARD_CRYPTO_SHADOW_ATTRIBUTION_HOT_RETENTION_BATCH_SIZE": "260000" if critical else "220000",
        }
    )
    return "offhours_external_backlog_drain", env


def _write_service_request(
    *,
    path: Path,
    drain_profile: str,
    drain_env: dict[str, str],
    wait_timeout_seconds: float,
    now_utc: datetime,
) -> dict[str, Any]:
    expires_utc = now_utc.timestamp() + max(float(wait_timeout_seconds), 900.0)
    payload = {
        "timestamp_utc": now_utc.isoformat(),
        "active": True,
        "request_kind": "external_backlog_drain",
        "reason": str(drain_profile or "external_backlog_drain"),
        "requested_at": now_utc.isoformat(),
        "expires_utc": datetime.fromtimestamp(expires_utc, tz=timezone.utc).isoformat(),
        "env_overrides": {str(key): str(value) for key, value in drain_env.items() if str(key).strip()},
    }
    _write_json(path, payload)
    return payload


def _backpressure_snapshot(payload: dict[str, Any]) -> dict[str, int]:
    return {
        "core_pending_lines": _safe_int(payload.get("pending_lines"), 0),
        "deferred_pending_lines": _safe_int(payload.get("pending_lines_deferred"), 0),
        "cold_pending_lines": _safe_int(payload.get("pending_lines_cold"), 0),
        "total_pending_lines": _safe_int(payload.get("pending_lines_total"), 0),
    }


def _hotspots(backpressure: dict[str, Any]) -> list[dict[str, Any]]:
    rows_by_source: dict[str, dict[str, Any]] = {}
    for lane, key in (
        ("deferred", "top_deferred_pending_files"),
        ("support", "top_support_telemetry_pending_files"),
        ("cold", "top_cold_pending_files"),
    ):
        raw_rows = backpressure.get(key)
        if not isinstance(raw_rows, list):
            continue
        for raw in raw_rows[:8]:
            if not isinstance(raw, dict):
                continue
            source_rel = str(raw.get("source_rel") or "").strip()
            if not source_rel:
                continue
            pending_lines = max(_safe_int(raw.get("pending_lines"), 0), 0)
            age_seconds = max(_safe_float(raw.get("oldest_pending_age_seconds"), 0.0), 0.0)
            if pending_lines <= 0:
                continue
            candidate_action = "drain_now"
            if source_rel.startswith("data/stale_stage/"):
                candidate_action = "reap_or_archive_stale_stage"
            elif lane == "support":
                candidate_action = "drain_support_watchdog"
            elif lane == "cold" and (age_seconds >= 6 * 60 * 60 or pending_lines >= 100000):
                candidate_action = "consider_archive_after_drain"
            elif lane == "deferred" and age_seconds >= 2 * 60 * 60:
                candidate_action = "drain_then_compact"
            current = rows_by_source.get(source_rel)
            row = {
                "lane": lane,
                "source_rel": source_rel,
                "pending_lines": pending_lines,
                "age_seconds": round(age_seconds, 3),
                "age_bucket": _age_bucket(age_seconds),
                "candidate_action": candidate_action,
            }
            if current is None:
                rows_by_source[source_rel] = row
                continue
            merged_lane = sorted({str(current.get("lane") or ""), lane})
            current["lane"] = ",".join(part for part in merged_lane if part)
            if pending_lines > int(current.get("pending_lines", 0) or 0):
                current["pending_lines"] = pending_lines
            if age_seconds > float(current.get("age_seconds", 0.0) or 0.0):
                current["age_seconds"] = round(age_seconds, 3)
                current["age_bucket"] = _age_bucket(age_seconds)
            preferred_actions = {
                "reap_or_archive_stale_stage": 3,
                "drain_support_watchdog": 2,
                "consider_archive_after_drain": 2,
                "drain_then_compact": 1,
                "drain_now": 0,
            }
            if preferred_actions.get(candidate_action, 0) > preferred_actions.get(str(current.get("candidate_action") or ""), 0):
                current["candidate_action"] = candidate_action
    rows = list(rows_by_source.values())
    rows.sort(
        key=lambda row: (
            int(row.get("pending_lines", 0)),
            float(row.get("age_seconds", 0.0) or 0.0),
        ),
        reverse=True,
    )
    return rows[:10]


def _follow_through_progress_signature(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "current_step": str(payload.get("current_step") or ""),
        "completed_shard_count": _safe_int(payload.get("completed_shard_count"), 0),
        "completed_merge_count": _safe_int(payload.get("completed_merge_count"), 0),
        "merged_rows_this_cycle": _safe_int(payload.get("merged_rows_this_cycle"), 0),
    }


def _follow_through_progressed(previous: dict[str, Any] | None, current: dict[str, Any]) -> bool:
    numeric_keys = ("completed_shard_count", "completed_merge_count", "merged_rows_this_cycle")
    if previous is None:
        return any(int(current.get(key, 0) or 0) > 0 for key in numeric_keys)
    if any(int(current.get(key, 0) or 0) > int(previous.get(key, 0) or 0) for key in numeric_keys):
        return True
    previous_step = str(previous.get("current_step") or "")
    current_step = str(current.get("current_step") or "")
    return bool(previous_step and current_step and current_step != previous_step)


def _follow_through_retry(
    *,
    project_root: Path,
    health_root: Path,
    drain_env: dict[str, str],
    poll_seconds: float,
    wait_timeout_seconds: float,
) -> dict[str, Any]:
    started = datetime.now(timezone.utc)
    deadline = started.timestamp() + max(float(wait_timeout_seconds), 0.0)
    attempts = 0
    last_result: dict[str, Any] | None = None
    observed_writer_pid = _lock_owner_pid(SQL_WRITER_LOCK_PATH)
    previous_signature: dict[str, Any] | None = None
    last_progress_signature: dict[str, Any] = {}
    progress_events = 0

    while datetime.now(timezone.utc).timestamp() <= deadline:
        attempts += 1
        result = _run_json_command(
            [str(PY), str(project_root / "scripts" / "ops" / "sql_link_shard_manager.py"), "--once", "--json"],
            cwd=project_root,
            payload_path=health_root / "sql_link_service_latest.json",
            env_overrides=drain_env,
        )
        last_result = result
        payload = result.get("payload") if isinstance(result.get("payload"), dict) else {}
        signature = _follow_through_progress_signature(payload)
        if signature and _follow_through_progressed(previous_signature, signature):
            progress_events += 1
            last_progress_signature = signature
        if signature:
            previous_signature = signature
        status = _step_status(result, nonfatal_reasons={"writer_lock_busy"})
        if status != "busy":
            break
        observed_writer_pid = observed_writer_pid or _lock_owner_pid(SQL_WRITER_LOCK_PATH)
        sleep_seconds = max(float(poll_seconds), 0.1)
        remaining = max(deadline - datetime.now(timezone.utc).timestamp(), 0.0)
        if remaining <= 0.0:
            break
        time_mod.sleep(min(sleep_seconds, remaining))

    waited_seconds = max((datetime.now(timezone.utc) - started).total_seconds(), 0.0)
    final_status = _step_status(last_result or {}, nonfatal_reasons={"writer_lock_busy"})
    completed = bool(last_result is not None and final_status != "busy")
    progress_observed = progress_events > 0
    return {
        "requested": True,
        "completed": completed,
        "timed_out": not completed,
        "attempts": attempts,
        "poll_seconds": round(float(poll_seconds), 3),
        "wait_timeout_seconds": round(float(wait_timeout_seconds), 3),
        "waited_seconds": round(waited_seconds, 3),
        "observed_writer_pid": observed_writer_pid,
        "status": "completed" if completed else "timed_out",
        "progress_observed": progress_observed,
        "progress_events": progress_events,
        "progress_state": "completed" if completed else ("progressing" if progress_observed else "stalled"),
        "last_progress_signature": last_progress_signature,
        "last_result": last_result or {},
    }


def build_payload(
    project_root: Path = PROJECT_ROOT,
    *,
    apply: bool,
    force_live_window: bool = False,
    resource_profile: str = "optional",
    follow_through: bool = False,
    poll_seconds: float = 20.0,
    wait_timeout_seconds: float = 900.0,
    now_utc: datetime | None = None,
) -> dict[str, Any]:
    now = now_utc or datetime.now(timezone.utc)
    health_root = project_root / "governance" / "health"
    backpressure_before = _load_json(health_root / "ingestion_backpressure_latest.json")
    queue_before = _load_json(health_root / "ingestion_priority_queue_latest.json")
    governor_payload = governor_src.build_payload(
        project_root,
        override_path=project_root / "config" / ".env.storage_pressure_override",
        action="status",
        changed=False,
    )
    window = _off_hours_window(now)
    mount = _load_json(health_root / "storage_mount_guard_latest.json")
    split_brain = _load_json(health_root / "storage_split_brain_reconciler_latest.json")
    unresolved_conflicts = _safe_int(((split_brain.get("summary") or {}).get("unresolved_conflicts")), 0)
    external_available = bool(mount.get("external_available", False))
    storage_mode = str(mount.get("storage_mode") or "")
    critical = str(governor_payload.get("profile") or "") == "critical_backpressure"
    drain_profile, drain_env = _drain_env(
        governor_payload.get("env_overrides") if isinstance(governor_payload.get("env_overrides"), dict) else {},
        critical=critical,
        off_hours_active=bool(window.get("active", False) or force_live_window),
    )

    blocked_reasons: list[str] = []
    if not external_available or storage_mode != "external":
        blocked_reasons.append("external_storage_unavailable")
    if unresolved_conflicts > 0:
        blocked_reasons.append("split_brain_unresolved")
    if not bool(window.get("active", False)) and not force_live_window:
        blocked_reasons.append("market_hours_guard")

    apply_executed = False
    backpressure_after = backpressure_before
    queue_after = queue_before
    steps: dict[str, Any] = {}
    writer_busy = False
    service_request_payload: dict[str, Any] = {}
    follow_through_summary = {
        "requested": bool(follow_through),
        "completed": False,
        "timed_out": False,
        "attempts": 0,
        "poll_seconds": round(float(poll_seconds), 3),
        "wait_timeout_seconds": round(float(wait_timeout_seconds), 3),
        "waited_seconds": 0.0,
        "observed_writer_pid": _lock_owner_pid(SQL_WRITER_LOCK_PATH),
        "progress_observed": False,
        "progress_events": 0,
        "progress_state": "not_requested" if not follow_through else "not_needed",
        "last_progress_signature": {},
        "status": "not_requested" if not follow_through else "not_needed",
    }

    if apply:
        steps["ingestion_backpressure_before"] = _step_record(
            _run_json_command(
                [str(PY), str(project_root / "scripts" / "ingestion_backpressure_guard.py"), "--json"],
                cwd=project_root,
                payload_path=health_root / "ingestion_backpressure_latest.json",
            )
        )
        if steps["ingestion_backpressure_before"]["status"] != "error":
            refreshed = _load_json(health_root / "ingestion_backpressure_latest.json")
            if refreshed:
                backpressure_before = refreshed

        steps["ingestion_priority_queue_before"] = _step_record(
            _run_json_command(
                [str(PY), str(project_root / "scripts" / "ops" / "ingestion_priority_queue.py"), "--json"],
                cwd=project_root,
                payload_path=health_root / "ingestion_priority_queue_latest.json",
            )
        )
        if steps["ingestion_priority_queue_before"]["status"] != "error":
            refreshed = _load_json(health_root / "ingestion_priority_queue_latest.json")
            if refreshed:
                queue_before = refreshed

        if not blocked_reasons:
            resource_guard = _run_json_command(
                [str(PY), str(project_root / "scripts" / "resource_guard.py"), "--profile", str(resource_profile or "optional"), "--json"],
                cwd=project_root,
                env_overrides=drain_env,
            )
            steps["resource_guard"] = _step_record(resource_guard)
            resource_payload = resource_guard.get("payload") if isinstance(resource_guard.get("payload"), dict) else {}
            resource_ok = bool(resource_payload.get("ok", resource_payload.get("resource_guard_ok", False)))
            if resource_ok:
                apply_executed = True
                shard_manager_initial = _run_json_command(
                    [str(PY), str(project_root / "scripts" / "ops" / "sql_link_shard_manager.py"), "--once", "--json"],
                    cwd=project_root,
                    payload_path=health_root / "sql_link_service_latest.json",
                    env_overrides=drain_env,
                )
                writer_busy = _step_status(shard_manager_initial, nonfatal_reasons={"writer_lock_busy"}) == "busy"
                shard_manager = shard_manager_initial
                steps["sql_link_shard_manager_initial"] = _step_record(shard_manager_initial, nonfatal_reasons={"writer_lock_busy"})
                if writer_busy:
                    service_request_path = health_root / "sql_link_service_request_latest.json"
                    service_request_payload = _write_service_request(
                        path=service_request_path,
                        drain_profile=drain_profile,
                        drain_env=drain_env,
                        wait_timeout_seconds=wait_timeout_seconds,
                        now_utc=now,
                    )
                    steps["sql_link_service_request"] = {
                        "status": "ok",
                        "rc": 0,
                        "duration_ms": 0.0,
                        "timed_out": False,
                        "cmd": ["write", str(service_request_path)],
                        "stdout_tail": json.dumps(service_request_payload, ensure_ascii=True),
                        "stderr_tail": "",
                    }
                    if follow_through:
                        follow_through_summary = {
                            **follow_through_summary,
                            "completed": True,
                            "timed_out": False,
                            "attempts": 1,
                            "waited_seconds": 0.0,
                            "observed_writer_pid": _lock_owner_pid(SQL_WRITER_LOCK_PATH),
                            "progress_observed": False,
                            "progress_events": 0,
                            "progress_state": "requested_live_writer",
                            "last_progress_signature": {},
                            "status": "handoff_requested",
                        }
                steps["sql_link_shard_manager"] = _step_record(shard_manager, nonfatal_reasons={"writer_lock_busy"})
                sqlite_maintenance = _run_json_command(
                    [str(PY), str(project_root / "scripts" / "sqlite_performance_maintenance.py"), "--checkpoint-only", "--json"],
                    cwd=project_root,
                    payload_path=health_root / "sqlite_maintenance_latest.json",
                    env_overrides=drain_env,
                    timeout_seconds=20.0,
                )
                steps["sqlite_maintenance"] = _step_record(sqlite_maintenance, nonfatal_reasons={"timeout"})
                stale_sweeper = _run_json_command(
                    [str(PY), str(project_root / "scripts" / "ops" / "stale_artifact_sweeper_bot.py"), "--json"],
                    cwd=project_root,
                    payload_path=health_root / "stale_artifact_sweeper_bot_latest.json",
                    env_overrides=drain_env,
                )
                steps["stale_artifact_sweeper_bot"] = _step_record(stale_sweeper, nonfatal_reasons={"already_running", "lock_busy"})
                stale_reaper = _run_json_command(
                    [str(PY), str(project_root / "scripts" / "ops" / "stale_artifact_reaper_bot.py"), "--json"],
                    cwd=project_root,
                    payload_path=health_root / "stale_artifact_reaper_bot_latest.json",
                    env_overrides=drain_env,
                )
                steps["stale_artifact_reaper_bot"] = _step_record(stale_reaper, nonfatal_reasons={"already_running", "lock_busy"})
                retention = _run_json_command(
                    [
                        str(PY),
                        str(project_root / "scripts" / "data_retention_policy.py"),
                        "--apply",
                        "--no-stale-stage",
                        "--no-stale-purge",
                        "--json",
                    ],
                    cwd=project_root,
                    payload_path=health_root / "data_retention_latest.json",
                    env_overrides=drain_env,
                )
                steps["data_retention_policy"] = _step_record(retention, nonfatal_reasons={"lock_busy"})
            else:
                blocked_reasons.append("resource_guard_blocked")

        if apply_executed:
            steps["ingestion_backpressure_after"] = _step_record(
                _run_json_command(
                    [str(PY), str(project_root / "scripts" / "ingestion_backpressure_guard.py"), "--json"],
                    cwd=project_root,
                    payload_path=health_root / "ingestion_backpressure_latest.json",
                )
            )
            refreshed = _load_json(health_root / "ingestion_backpressure_latest.json")
            if refreshed:
                backpressure_after = refreshed
            steps["ingestion_priority_queue_after"] = _step_record(
                _run_json_command(
                    [str(PY), str(project_root / "scripts" / "ops" / "ingestion_priority_queue.py"), "--json"],
                    cwd=project_root,
                    payload_path=health_root / "ingestion_priority_queue_latest.json",
                )
            )
            refreshed = _load_json(health_root / "ingestion_priority_queue_latest.json")
            if refreshed:
                queue_after = refreshed

    before_snapshot = _backpressure_snapshot(backpressure_before)
    after_snapshot = _backpressure_snapshot(backpressure_after)
    hotspots = _hotspots(backpressure_after if apply_executed else backpressure_before)
    aged_candidates = [
        row
        for row in hotspots
        if str(row.get("candidate_action") or "") in {
            "consider_archive_after_drain",
            "drain_then_compact",
            "reap_or_archive_stale_stage",
        }
    ]
    stale_stage_candidates = [
        row for row in hotspots if str(row.get("candidate_action") or "") == "reap_or_archive_stale_stage"
    ]
    support_watchdog_candidates = [
        row for row in hotspots if str(row.get("candidate_action") or "") == "drain_support_watchdog"
    ]
    top_actions: list[str] = []
    if "external_storage_unavailable" in blocked_reasons:
        top_actions.append("keep the writer on the routed local path until external BOT_LOGS storage is healthy again")
    if "split_brain_unresolved" in blocked_reasons:
        top_actions.append("resolve split-brain conflicts before draining the external backlog")
    if "market_hours_guard" in blocked_reasons:
        top_actions.append("wait for the off-hours window before raising deferred and cold drain quotas")
    if "resource_guard_blocked" in blocked_reasons:
        top_actions.append("rerun the external backlog drain after memory and disk guards return to green")
    if aged_candidates:
        top_actions.append("compact or archive the oldest deferred and cold backlog files after the active drain pass")
    if stale_stage_candidates:
        top_actions.append("reap or archive staged stale artifacts after the active drain pass so cold backlog stops recycling")
    if support_watchdog_candidates:
        top_actions.append("let the watchdog support shard drain failover and pager logs off the main governance path")
    if follow_through and follow_through_summary["status"] == "timed_out":
        if str(follow_through_summary.get("progress_state") or "") == "progressing":
            top_actions.append("the automatic follow-through timed out, but the SQL writer was still advancing shard or merge work, so let the current maintenance window run or extend the timeout next pass")
        else:
            top_actions.append("the automatic follow-through timed out without any observed shard or merge progress, so rerun during a quieter maintenance window")
    if str(follow_through_summary.get("status") or "") == "handoff_requested":
        top_actions.append("the active SQL writer accepted a live drain request and will apply the backlog-drain overrides on its next cycle")
    if writer_busy:
        if str(follow_through_summary.get("progress_state") or "") == "progressing":
            top_actions.append("let the current SQL writer finish the active drain cycle before forcing another external backlog pass")
        elif str(follow_through_summary.get("status") or "") == "handoff_requested":
            top_actions.append("let the current SQL writer roll into the requested drain cycle before judging deferred or cold progress")
        else:
            top_actions.append("rerun the external backlog drain after the current SQL writer lock holder finishes")
    if after_snapshot["deferred_pending_lines"] > 0 or after_snapshot["cold_pending_lines"] > 0:
        top_actions.append("repeat the external backlog drain during off-hours until deferred and cold queues stay below target")
    if after_snapshot["total_pending_lines"] > 0:
        top_actions.append("keep shadow attribution and channel logging throttled while the external backlog burns down")

    recommended_now = bool(window.get("active", False) and not blocked_reasons)
    payload = {
        "timestamp_utc": now.isoformat(),
        "schema_version": 1,
        "ok": not blocked_reasons,
        "overall_status": "drain_active" if apply_executed else ("ready" if not blocked_reasons else "blocked"),
        "apply_requested": bool(apply),
        "apply_executed": bool(apply_executed),
        "follow_through": follow_through_summary,
        "recommended_now": recommended_now,
        "blocked_reasons": blocked_reasons,
        "off_hours_window": window,
        "drain_profile": drain_profile,
        "governor_profile": str(governor_payload.get("profile") or ""),
        "storage_mode": storage_mode,
        "writer_busy": bool(writer_busy),
        "service_request_path": str(health_root / "sql_link_service_request_latest.json"),
        "service_request": service_request_payload,
        "backpressure_before": before_snapshot,
        "backpressure_after": after_snapshot,
        "drain_delta": {
            key: int(before_snapshot.get(key, 0) - after_snapshot.get(key, 0))
            for key in ("core_pending_lines", "deferred_pending_lines", "cold_pending_lines", "total_pending_lines")
        },
        "hotspots": hotspots,
        "aged_candidate_files": len(aged_candidates),
        "aged_candidate_pending_lines": sum(int(row.get("pending_lines", 0) or 0) for row in aged_candidates),
        "stale_stage_candidate_files": len(stale_stage_candidates),
        "stale_stage_candidate_pending_lines": sum(int(row.get("pending_lines", 0) or 0) for row in stale_stage_candidates),
        "support_watchdog_candidate_files": len(support_watchdog_candidates),
        "support_watchdog_candidate_pending_lines": sum(int(row.get("pending_lines", 0) or 0) for row in support_watchdog_candidates),
        "queue_depth_before": _safe_int(queue_before.get("queue_depth"), 0),
        "queue_depth_after": _safe_int(queue_after.get("queue_depth"), _safe_int(queue_before.get("queue_depth"), 0)),
        "drain_overrides": {
            "deferred_files_budget": _safe_int(drain_env.get("INGEST_MAX_DEFERRED_FILES"), 0),
            "cold_files_budget": _safe_int(drain_env.get("JSONL_SQL_MAX_COLD_LANE_FILES"), 0),
            "sql_interval_seconds": _safe_int(drain_env.get("SQL_LINK_SERVICE_INTERVAL_SECONDS"), 0),
            "hot_batch_size": _safe_int(drain_env.get("SQL_LINK_SERVICE_HOT_BATCH_SIZE"), 0),
        },
        "steps": steps,
        "top_actions": top_actions[:8],
    }
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Run an off-hours external backlog drain for deferred and cold ingestion lanes.")
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--out-file", default=str(DEFAULT_OUT_PATH))
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--force-live-window", action="store_true")
    parser.add_argument("--resource-profile", default="optional")
    parser.add_argument("--follow-through", action="store_true")
    parser.add_argument("--poll-seconds", type=float, default=20.0)
    parser.add_argument("--wait-timeout-seconds", type=float, default=900.0)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    payload = build_payload(
        project_root,
        apply=bool(args.apply),
        force_live_window=bool(args.force_live_window),
        resource_profile=str(args.resource_profile or "optional"),
        follow_through=bool(args.follow_through),
        poll_seconds=float(args.poll_seconds),
        wait_timeout_seconds=float(args.wait_timeout_seconds),
    )
    out_path = Path(args.out_file).expanduser()
    _write_json(out_path, payload)
    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        print(
            "external_backlog_drain "
            f"status={payload.get('overall_status', '')} "
            f"recommended_now={int(bool(payload.get('recommended_now', False)))} "
            f"apply_executed={int(bool(payload.get('apply_executed', False)))}"
        )
    return 0 if bool(payload.get("ok", False) or payload.get("apply_executed", False)) else 2


if __name__ == "__main__":
    raise SystemExit(main())
