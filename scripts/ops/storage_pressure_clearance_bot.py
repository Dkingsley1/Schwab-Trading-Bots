#!/usr/bin/env python3
from __future__ import annotations

import argparse
import fcntl
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from scripts.ops.long_runtime_common import iso_now, load_json, ordered_unique, write_payload
else:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    from .long_runtime_common import iso_now, load_json, ordered_unique, write_payload


DEFAULT_OUT_PATH = PROJECT_ROOT / "governance" / "health" / "storage_pressure_clearance_latest.json"
DEFAULT_LOCK_PATH = PROJECT_ROOT / "governance" / "locks" / "storage_pressure_clearance.lock"
PYTHON_BIN = Path(sys.executable)
RECOVERABLE_STORAGE_GATES = {
    "ingestion_backpressure_overload",
    "sql_progress_stall",
    "sql_wal_pressure",
}
ACTIVE_AUTOPILOT_STATUSES = {
    "already_running",
    "busy",
    "drain_active",
    "recovering",
    "recovering_under_guard",
    "running",
}


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


def _safe_bool(raw: Any, default: bool = False) -> bool:
    if isinstance(raw, bool):
        return raw
    text = str(raw or "").strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    return bool(default)


def _run_json(cmd: list[str], *, cwd: Path, timeout_sec: int) -> dict[str, Any]:
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(cwd),
            capture_output=True,
            text=True,
            check=False,
            timeout=max(int(timeout_sec), 1),
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
    payload: dict[str, Any] = {}
    for raw in reversed([line.strip() for line in stdout.splitlines() if line.strip()]):
        try:
            parsed = json.loads(raw)
        except Exception:
            continue
        if isinstance(parsed, dict):
            payload = parsed
            break
    return {
        "cmd": list(cmd),
        "rc": rc,
        "timed_out": timed_out,
        "stdout_tail": "\n".join(stdout.splitlines()[-10:]),
        "stderr_tail": "\n".join(stderr.splitlines()[-10:]),
        "payload": payload,
    }


def _attempt_record(name: str, result: dict[str, Any], *, accepted_rcs: set[int] | None = None) -> dict[str, Any]:
    accepted = accepted_rcs or {0}
    rc = _safe_int(result.get("rc"), 1)
    status = "ok"
    if bool(result.get("timed_out", False)):
        status = "timed_out"
    elif rc not in accepted:
        status = "error"
    return {
        "name": name,
        "status": status,
        "rc": rc,
        "timed_out": bool(result.get("timed_out", False)),
        "cmd": list(result.get("cmd") or []),
        "stdout_tail": str(result.get("stdout_tail") or ""),
        "stderr_tail": str(result.get("stderr_tail") or ""),
        "payload": result.get("payload") if isinstance(result.get("payload"), dict) else {},
    }


def _load_artifacts(project_root: Path) -> dict[str, dict[str, Any]]:
    health_root = project_root / "governance" / "health"
    candidate_roots: list[Path] = [health_root]
    external_env = os.getenv("BOT_LOGS_EXTERNAL_PROJECT_ROOT")
    external_root = Path(external_env or "/Volumes/BOT_LOGS/schwab_trading_bot") / "governance" / "health"
    fallback_root = project_root / "local_fallback_storage" / "governance" / "health"
    optional_roots: list[Path] = []
    if external_env or project_root == PROJECT_ROOT:
        optional_roots.append(external_root)
    optional_roots.append(fallback_root)
    for root in optional_roots:
        if root not in candidate_roots:
            candidate_roots.append(root)

    def load_first(name: str) -> dict[str, Any]:
        for root in candidate_roots:
            payload = load_json(root / name)
            if payload:
                payload = dict(payload)
                payload.setdefault("_source_path", str(root / name))
                return payload
        return {}

    return {
        "storage_control": load_first("ingestion_storage_control_latest.json"),
        "health_gates": load_first("health_gates_latest.json"),
        "autopilot": load_first("storage_backpressure_autopilot_latest.json"),
        "sqlite_maintenance": load_first("sqlite_maintenance_latest.json"),
        "global_killswitch": load_first("global_killswitch_latest.json"),
        "storage_mount_guard": load_first("storage_mount_guard_latest.json"),
        "storage_failback_sync": load_first("storage_failback_sync_latest.json"),
        "storage_route_status": load_first("storage_route_status_latest.json"),
    }


def _active_hard_gate_names(health_gates: dict[str, Any]) -> list[str]:
    hard_gates = health_gates.get("hard_gates") if isinstance(health_gates.get("hard_gates"), dict) else {}
    return sorted(str(name) for name, active in hard_gates.items() if bool(active))


def _autopilot_active(autopilot: dict[str, Any]) -> bool:
    status = str(autopilot.get("overall_status") or autopilot.get("status") or "").strip().lower()
    return bool(autopilot.get("busy", False)) or status in ACTIVE_AUTOPILOT_STATUSES


def _effective_storage_backpressure(storage_control: dict[str, Any]) -> dict[str, Any]:
    backpressure = storage_control.get("backpressure") if isinstance(storage_control.get("backpressure"), dict) else {}
    effective = backpressure.get("effective_raw_live") if isinstance(backpressure.get("effective_raw_live"), dict) else {}
    data_integrity = storage_control.get("data_integrity") if isinstance(storage_control.get("data_integrity"), dict) else {}
    source = str(backpressure.get("effective_raw_live_source") or effective.get("source") or "").strip()
    storage_ready = bool(
        str(storage_control.get("overall_status") or "").strip().lower() == "ready"
        and str(storage_control.get("severity") or "").strip().lower() == "stable"
    )
    overlay_clear = bool(backpressure.get("overlay_pressure_clear", False) or source == "fresh_empty_sql_ingestion_overlay")
    data_clean = bool(
        _safe_int(data_integrity.get("sql_overlay_invalid_lines"), 0) <= 0
        and _safe_int(data_integrity.get("sql_overlay_oversize_payloads"), 0) <= 0
        and _safe_int(data_integrity.get("sql_overlay_ops_write_failures"), 0) <= 0
    )
    authoritative = bool(
        storage_ready
        and bool(backpressure.get("overlay_adjusted", False))
        and overlay_clear
        and data_clean
    )
    total = _safe_int(effective.get("total_pending_lines"), _safe_int(backpressure.get("total_pending_lines"), 0))
    core = _safe_int(effective.get("core_pending_lines"), _safe_int(backpressure.get("core_pending_lines"), total))
    oldest = _safe_float(effective.get("oldest_pending_age_seconds"), _safe_float(backpressure.get("oldest_pending_age_seconds"), 0.0))
    return {
        "authoritative": authoritative,
        "source": source or "ingestion_storage_control_effective_raw_live",
        "core_pending_lines": int(core),
        "total_pending_lines": int(total),
        "oldest_pending_age_seconds": round(oldest, 3),
        "storage_ready": storage_ready,
        "overlay_clear": overlay_clear,
        "data_clean": data_clean,
    }


def _storage_metrics(artifacts: dict[str, dict[str, Any]]) -> dict[str, Any]:
    storage_control = artifacts.get("storage_control") or {}
    health_gates = artifacts.get("health_gates") or {}
    autopilot = artifacts.get("autopilot") or {}
    sqlite_maintenance = artifacts.get("sqlite_maintenance") or {}
    storage_mount_guard = artifacts.get("storage_mount_guard") or {}
    storage_failback_sync = artifacts.get("storage_failback_sync") or {}
    storage_route_status = artifacts.get("storage_route_status") or {}

    backpressure = storage_control.get("backpressure") if isinstance(storage_control.get("backpressure"), dict) else {}
    storage = storage_control.get("storage") if isinstance(storage_control.get("storage"), dict) else {}
    bounded = storage_control.get("bounded_recovery_contract") if isinstance(storage_control.get("bounded_recovery_contract"), dict) else {}
    steady_state = storage_control.get("steady_state") if isinstance(storage_control.get("steady_state"), dict) else {}
    steady_targets = steady_state.get("targets") if isinstance(steady_state.get("targets"), dict) else {}
    steady_status = steady_state.get("target_status") if isinstance(steady_state.get("target_status"), dict) else {}
    thresholds = health_gates.get("thresholds") if isinstance(health_gates.get("thresholds"), dict) else {}
    inputs = health_gates.get("inputs") if isinstance(health_gates.get("inputs"), dict) else {}

    effective_backpressure = _effective_storage_backpressure(storage_control)
    authoritative_backpressure = bool(effective_backpressure.get("authoritative", False))
    core_pending_lines = max(
        _safe_int(effective_backpressure.get("core_pending_lines"), 0)
        if authoritative_backpressure
        else _safe_int(backpressure.get("core_pending_lines"), 0),
        0,
    )
    total_pending_lines = max(
        _safe_int(effective_backpressure.get("total_pending_lines"), 0)
        if authoritative_backpressure
        else max(_safe_int(backpressure.get("total_pending_lines"), 0), _safe_int(inputs.get("backpressure_pending_lines"), 0)),
        0,
    )
    pressure_index = max(_safe_float(storage_control.get("pressure_index"), 0.0), 0.0)
    pressure_target = max(_safe_float(steady_targets.get("pressure_index"), 0.25), 0.01)
    core_target = max(_safe_int(steady_targets.get("core_pending_lines"), 5000), 0)
    pending_limit = max(_safe_int(thresholds.get("ingestion_pending_lines_limit"), 20000), 1)
    wal_limit_gb = max(
        _safe_float(
            thresholds.get("sql_wal_size_gb_limit"),
            _safe_float(os.getenv("HEALTH_GATE_SQL_WAL_SIZE_GB_LIMIT"), 24.0),
        ),
        0.1,
    )
    live_wal_values = [
        _safe_float(storage.get("sqlite_wal_size_gb"), 0.0),
        _safe_float(inputs.get("sql_wal_size_gb_live"), 0.0),
    ]
    sqlite_wal_size_gb = max(live_wal_values)
    if sqlite_wal_size_gb <= 0.0 and not storage_control and not health_gates:
        sqlite_wal_size_gb = max(
            _safe_float(sqlite_maintenance.get("wal_size_gb_after"), 0.0),
            _safe_float(sqlite_maintenance.get("wal_size_gb_before"), 0.0),
        )
    estimated_total_drain_minutes_raw = backpressure.get("estimated_total_drain_minutes")
    estimated_total_drain_minutes = (
        None
        if estimated_total_drain_minutes_raw in {None, "", "n/a"}
        else max(_safe_float(estimated_total_drain_minutes_raw), 0.0)
    )

    hard_gate_names = _active_hard_gate_names(health_gates)
    stale_hard_gate_suppressed: list[str] = []
    if authoritative_backpressure and "ingestion_backpressure_overload" in hard_gate_names:
        hard_gate_names = [name for name in hard_gate_names if name != "ingestion_backpressure_overload"]
        stale_hard_gate_suppressed.append("ingestion_backpressure_overload")
    active_reasons = ordered_unique(
        [
            "sql_wal_pressure" if sqlite_wal_size_gb > wal_limit_gb else "",
            "core_pending_above_target" if core_pending_lines > core_target else "",
            "total_pending_above_gate_limit" if total_pending_lines > pending_limit else "",
            "pressure_index_above_target" if pressure_index > pressure_target else "",
            "backpressure_overload_severe"
            if _safe_bool(inputs.get("backpressure_overload_severe"), False) and not authoritative_backpressure
            else "",
            "storage_control_blocked" if str(storage_control.get("overall_status") or "").strip().lower() == "blocked" else "",
        ]
    )
    route_verified = _safe_bool(bounded.get("route_verified"), False)
    route_verification = storage.get("route_verification") if isinstance(storage.get("route_verification"), dict) else {}
    if str(route_verification.get("verification_state") or "").strip().lower() in {"ready", "verified", "curated_ready"}:
        route_verified = True
    storage_mode_values = {
        str(storage_mount_guard.get("storage_mode") or "").strip().lower(),
        str(storage_failback_sync.get("mode") or "").strip().lower(),
        str(storage_failback_sync.get("certified_mode") or "").strip().lower(),
        str(storage_route_status.get("mode") or "").strip().lower(),
    }
    split_brain_conflicts = max(
        _safe_int(storage_failback_sync.get("split_brain_conflicts"), 0),
        _safe_int(storage_route_status.get("split_brain_conflicts"), 0),
    )
    route_blocked = bool(
        split_brain_conflicts > 0
        or any("split_brain" in value for value in storage_mode_values if value)
        or str(storage_mount_guard.get("external_unavailable_reason") or "").strip().lower()
        not in {"", "ok"}
    )
    hard_gate_set = set(hard_gate_names)
    recoverable_hard_gate_only = bool(hard_gate_set and hard_gate_set.issubset(RECOVERABLE_STORAGE_GATES))
    active_storage_pressure = bool(active_reasons)
    stale_gate_candidate = bool(recoverable_hard_gate_only and not active_storage_pressure and route_verified and not route_blocked)
    clearance_ready = bool(not hard_gate_names and not active_storage_pressure and not route_blocked and bool(steady_status.get("steady_state_ready", False)))
    if not hard_gate_names and not active_storage_pressure and not storage_control:
        clearance_ready = False

    return {
        "storage_control_present": bool(storage_control),
        "storage_control_status": str(storage_control.get("overall_status") or ""),
        "storage_control_severity": str(storage_control.get("severity") or ""),
        "recovery_state": str(storage_control.get("recovery_state") or ""),
        "hard_gate_names": hard_gate_names,
        "recoverable_hard_gate_only": recoverable_hard_gate_only,
        "active_storage_pressure": active_storage_pressure,
        "active_pressure_reasons": active_reasons,
        "effective_backpressure": effective_backpressure,
        "stale_hard_gate_suppressed": stale_hard_gate_suppressed,
        "stale_gate_candidate": stale_gate_candidate,
        "clearance_ready": clearance_ready,
        "route_verified": route_verified,
        "route_blocked": route_blocked,
        "storage_modes": sorted(value for value in storage_mode_values if value),
        "split_brain_conflicts": split_brain_conflicts,
        "autopilot_active": _autopilot_active(autopilot),
        "autopilot_status": str(autopilot.get("overall_status") or autopilot.get("status") or ""),
        "sqlite_wal_size_gb": round(sqlite_wal_size_gb, 3),
        "sqlite_wal_limit_gb": round(wal_limit_gb, 3),
        "core_pending_lines": core_pending_lines,
        "core_pending_target": core_target,
        "total_pending_lines": total_pending_lines,
        "pending_gate_limit": pending_limit,
        "pressure_index": round(pressure_index, 3),
        "pressure_index_target": round(pressure_target, 3),
        "estimated_total_drain_minutes": estimated_total_drain_minutes,
        "steady_state_ready": bool(steady_status.get("steady_state_ready", False)),
        "backlog_drain_status": str(storage.get("backlog_drain_status") or ""),
        "backlog_drain_recommended_now": bool(storage.get("backlog_drain_recommended_now", False)),
        "bounded_recovery_active": bool(bounded.get("active", False)),
        "bounded_recovery_quality_ready": bool(bounded.get("quality_ready", False)),
        "active_drain_progress": bool(bounded.get("active_drain_progress", False)),
    }


def _cmds(
    *,
    project_root: Path,
    checkpoint_mode: str,
    max_cycles: int,
    poll_seconds: float,
    wait_timeout_seconds: float,
    command_timeout_seconds: int,
) -> dict[str, list[str]]:
    ops_root = project_root / "scripts" / "ops"
    return {
        "refresh_storage_control": [str(PYTHON_BIN), str(ops_root / "ingestion_storage_control.py"), "--json"],
        "sqlite_passive_checkpoint": [
            str(PYTHON_BIN),
            str(project_root / "scripts" / "sqlite_performance_maintenance.py"),
            "--checkpoint-only",
            "--wal-checkpoint-mode",
            checkpoint_mode,
            "--json",
        ],
        "storage_backpressure_autopilot": [
            str(PYTHON_BIN),
            str(ops_root / "storage_backpressure_autopilot.py"),
            "--apply",
            "--max-cycles",
            str(max(int(max_cycles), 1)),
            "--poll-seconds",
            str(max(float(poll_seconds), 0.1)),
            "--wait-timeout-seconds",
            str(max(float(wait_timeout_seconds), 0.0)),
            "--command-timeout-seconds",
            str(max(int(command_timeout_seconds), 1)),
            "--json",
        ],
        "refresh_health_gates": [str(PYTHON_BIN), str(project_root / "scripts" / "health_gates.py")],
        "global_halt_auto_clear": [
            str(PYTHON_BIN),
            str(project_root / "scripts" / "global_risk_killswitch.py"),
            "--auto-clear",
            "--clear-blockers",
            "--exit-zero",
        ],
    }


def _planned_steps(
    metrics: dict[str, Any],
    *,
    commands: dict[str, list[str]],
    skip_checkpoint: bool,
    force_clear_stale_gate: bool,
) -> list[dict[str, Any]]:
    plan: list[dict[str, Any]] = [
        {
            "name": "refresh_storage_control",
            "reason": "make storage pressure decisions from the latest ingestion storage artifact",
            "cmd": commands["refresh_storage_control"],
            "accepted_rcs": [0],
        }
    ]
    if bool(metrics.get("active_storage_pressure", False)):
        if "sql_wal_pressure" in list(metrics.get("active_pressure_reasons") or []) and not skip_checkpoint:
            plan.append(
                {
                    "name": "sqlite_passive_checkpoint",
                    "reason": f"sqlite_wal_size_gb={metrics.get('sqlite_wal_size_gb')} limit={metrics.get('sqlite_wal_limit_gb')}",
                    "cmd": commands["sqlite_passive_checkpoint"],
                    "accepted_rcs": [0],
                }
            )
        if bool(metrics.get("autopilot_active", False)):
            plan.append(
                {
                    "name": "observe_existing_storage_autopilot",
                    "reason": f"storage_backpressure_autopilot_status={metrics.get('autopilot_status') or 'active'}",
                    "cmd": [],
                    "accepted_rcs": [0],
                }
            )
        else:
            plan.append(
                {
                    "name": "storage_backpressure_autopilot",
                    "reason": ",".join(list(metrics.get("active_pressure_reasons") or [])) or "active_storage_pressure",
                    "cmd": commands["storage_backpressure_autopilot"],
                    "accepted_rcs": [0],
                }
            )
        plan.append(
            {
                "name": "refresh_health_gates",
                "reason": "recompute hard gates after checkpoint/drain attempts",
                "cmd": commands["refresh_health_gates"],
                "accepted_rcs": [0, 2],
            }
        )
    elif bool(metrics.get("stale_gate_candidate", False)):
        plan.append(
            {
                "name": "refresh_health_gates",
                "reason": "stale recoverable storage gate candidate needs a fresh health-gate calculation",
                "cmd": commands["refresh_health_gates"],
                "accepted_rcs": [0, 2],
            }
        )
        if force_clear_stale_gate:
            plan.append(
                {
                    "name": "global_halt_auto_clear",
                    "reason": "force-clear-stale-gate requested and live storage metrics are inside the safe envelope",
                    "cmd": commands["global_halt_auto_clear"],
                    "accepted_rcs": [0],
                }
            )
    return plan


def build_payload(
    project_root: Path = PROJECT_ROOT,
    *,
    apply: bool = False,
    force_clear_stale_gate: bool = False,
    skip_checkpoint: bool = False,
    checkpoint_mode: str = "passive",
    max_cycles: int = 1,
    poll_seconds: float = 10.0,
    wait_timeout_seconds: float = 180.0,
    command_timeout_seconds: int = 900,
) -> dict[str, Any]:
    command_map = _cmds(
        project_root=project_root,
        checkpoint_mode=checkpoint_mode,
        max_cycles=max_cycles,
        poll_seconds=poll_seconds,
        wait_timeout_seconds=wait_timeout_seconds,
        command_timeout_seconds=command_timeout_seconds,
    )
    artifacts_before = _load_artifacts(project_root)
    metrics_before = _storage_metrics(artifacts_before)
    attempts: list[dict[str, Any]] = []

    metrics_for_plan = metrics_before
    if apply:
        refresh_result = _run_json(command_map["refresh_storage_control"], cwd=project_root, timeout_sec=180)
        attempts.append(_attempt_record("refresh_storage_control", refresh_result, accepted_rcs={0}))
        metrics_for_plan = _storage_metrics(_load_artifacts(project_root))

    plan = _planned_steps(
        metrics_for_plan,
        commands=command_map,
        skip_checkpoint=skip_checkpoint,
        force_clear_stale_gate=force_clear_stale_gate,
    )
    if apply:
        plan = [row for row in plan if row.get("name") != "refresh_storage_control"]
        for row in plan:
            name = str(row.get("name") or "")
            cmd = [str(part) for part in list(row.get("cmd") or []) if str(part).strip()]
            if not cmd:
                attempts.append(
                    {
                        "name": name,
                        "status": "observed",
                        "rc": 0,
                        "timed_out": False,
                        "cmd": [],
                        "stdout_tail": "",
                        "stderr_tail": "",
                        "payload": {},
                    }
                )
                continue
            timeout_sec = int(command_timeout_seconds)
            if name == "storage_backpressure_autopilot":
                timeout_sec = max(int(command_timeout_seconds), int(wait_timeout_seconds) + 120)
            elif name in {"refresh_storage_control", "refresh_health_gates"}:
                timeout_sec = 180
            elif name == "global_halt_auto_clear":
                timeout_sec = 120
            result = _run_json(cmd, cwd=project_root, timeout_sec=timeout_sec)
            attempts.append(
                _attempt_record(
                    name,
                    result,
                    accepted_rcs={_safe_int(raw) for raw in list(row.get("accepted_rcs") or [0])},
                )
            )

        post_refresh = _run_json(command_map["refresh_storage_control"], cwd=project_root, timeout_sec=180)
        attempts.append(_attempt_record("post_refresh_storage_control", post_refresh, accepted_rcs={0}))

        metrics_after_actions = _storage_metrics(_load_artifacts(project_root))
        if (
            bool(force_clear_stale_gate)
            and not bool(metrics_after_actions.get("active_storage_pressure", False))
            and not bool(metrics_after_actions.get("route_blocked", False))
        ):
            health_result = _run_json(command_map["refresh_health_gates"], cwd=project_root, timeout_sec=180)
            attempts.append(_attempt_record("post_refresh_health_gates", health_result, accepted_rcs={0, 2}))
            metrics_after_actions = _storage_metrics(_load_artifacts(project_root))
            if (
                not bool(metrics_after_actions.get("active_storage_pressure", False))
                and not bool(metrics_after_actions.get("route_blocked", False))
            ):
                halt_result = _run_json(command_map["global_halt_auto_clear"], cwd=project_root, timeout_sec=120)
                attempts.append(_attempt_record("global_halt_auto_clear", halt_result, accepted_rcs={0}))

    artifacts_after = _load_artifacts(project_root)
    metrics_after = _storage_metrics(artifacts_after)
    attempted_errors = [row for row in attempts if str(row.get("status") or "") in {"error", "timed_out"}]

    if bool(metrics_after.get("clearance_ready", False)):
        overall_status = "ready"
    elif bool(metrics_after.get("active_storage_pressure", False)) or bool(metrics_after.get("route_blocked", False)):
        overall_status = "degraded" if bool(metrics_after.get("autopilot_active", False)) or apply else "blocked"
    elif bool(metrics_after.get("stale_gate_candidate", False)):
        overall_status = "degraded"
    else:
        overall_status = "degraded" if metrics_after.get("storage_control_present") else "blocked"
    if attempted_errors:
        overall_status = "blocked"

    force_clear_allowed = bool(
        force_clear_stale_gate
        and not metrics_after.get("active_storage_pressure", False)
        and not metrics_after.get("route_blocked", False)
    )
    force_clear_refused_reason = ""
    if force_clear_stale_gate and bool(metrics_after.get("active_storage_pressure", False)):
        force_clear_refused_reason = "active_storage_pressure"
    elif force_clear_stale_gate and bool(metrics_after.get("route_blocked", False)):
        force_clear_refused_reason = "storage_route_blocked"

    operator_followups = ordered_unique(
        [
            "storage pressure is still active, so this bot refused to fake-clear the storage gate"
            if force_clear_refused_reason
            else "",
            "storage route is blocked or split-brain; resolve failback/split-brain before clearing storage pressure"
            if bool(metrics_after.get("route_blocked", False))
            else "",
            "storage backpressure autopilot is already active; do not launch duplicate drain jobs"
            if bool(metrics_after.get("autopilot_active", False))
            else "",
            "WAL pressure remains above the hard-gate limit; keep passive checkpoints and writer drain running"
            if _safe_float(metrics_after.get("sqlite_wal_size_gb"), 0.0) > _safe_float(metrics_after.get("sqlite_wal_limit_gb"), 24.0)
            else "",
            "pending backlog remains above target; keep collection protected while the storage lane drains"
            if _safe_int(metrics_after.get("core_pending_lines"), 0) > _safe_int(metrics_after.get("core_pending_target"), 5000)
            else "",
        ]
    )

    return {
        "timestamp_utc": iso_now(),
        "schema_version": 1,
        "ok": overall_status == "ready",
        "overall_status": overall_status,
        "apply_requested": bool(apply),
        "force_clear_stale_gate_requested": bool(force_clear_stale_gate),
        "force_clear_allowed": force_clear_allowed,
        "force_clear_refused_reason": force_clear_refused_reason,
        "repair_plan": plan if not apply else _planned_steps(metrics_after, commands=command_map, skip_checkpoint=skip_checkpoint, force_clear_stale_gate=force_clear_stale_gate),
        "attempts": attempts,
        "storage_pressure": {
            "before": metrics_before,
            "after": metrics_after,
        },
        "recommended_commands": [
            ["./scripts/ops/opsctl.sh", "storage-pressure-clearance", "--apply", "--force-clear-stale-gate", "--json"],
            ["./scripts/ops/opsctl.sh", "ingestion-storage-control", "--json"],
            ["./scripts/ops/opsctl.sh", "storage-backpressure-autopilot", "--apply", "--json"],
            ["./scripts/ops/opsctl.sh", "global-halt-auto-clear", "--json"],
        ],
        "operator_followups": operator_followups,
        "recommended_actions": ordered_unique(
            [
                "use storage-pressure-clearance as the parent storage recovery lane when GLOBAL_TRADING_HALT is blocked by storage pressure",
                "let the bot force-refresh and clear only stale storage gates; active WAL/backlog pressure must drain first",
                "keep the child storage backpressure autopilot on launchd so checkpoint, drain, and retention work stays coordinated",
            ]
            + operator_followups
        )[:8],
        "metrics": {
            "active_storage_pressure": bool(metrics_after.get("active_storage_pressure", False)),
            "stale_gate_candidate": bool(metrics_after.get("stale_gate_candidate", False)),
            "clearance_ready": bool(metrics_after.get("clearance_ready", False)),
            "autopilot_active": bool(metrics_after.get("autopilot_active", False)),
            "route_blocked": bool(metrics_after.get("route_blocked", False)),
            "attempt_count": len(attempts),
            "attempt_error_count": len(attempted_errors),
            "sqlite_wal_size_gb": _safe_float(metrics_after.get("sqlite_wal_size_gb"), 0.0),
            "sqlite_wal_limit_gb": _safe_float(metrics_after.get("sqlite_wal_limit_gb"), 24.0),
            "core_pending_lines": _safe_int(metrics_after.get("core_pending_lines"), 0),
            "total_pending_lines": _safe_int(metrics_after.get("total_pending_lines"), 0),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Parent storage pressure clearance bot that distinguishes active pressure from stale clearable storage gates."
    )
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--out-file", default=str(DEFAULT_OUT_PATH))
    parser.add_argument("--lock-file", default=str(DEFAULT_LOCK_PATH))
    parser.add_argument("--apply", action="store_true")
    parser.add_argument(
        "--force-clear-stale-gate",
        action="store_true",
        help="Refresh gates and attempt global auto-clear only when live storage metrics prove the gate is stale.",
    )
    parser.add_argument("--skip-checkpoint", action="store_true")
    parser.add_argument("--checkpoint-mode", choices=("passive", "restart", "truncate"), default=os.getenv("STORAGE_PRESSURE_CLEARANCE_CHECKPOINT_MODE", "passive"))
    parser.add_argument("--max-cycles", type=int, default=int(os.getenv("STORAGE_PRESSURE_CLEARANCE_MAX_CYCLES", "1")))
    parser.add_argument("--poll-seconds", type=float, default=float(os.getenv("STORAGE_PRESSURE_CLEARANCE_POLL_SECONDS", "10")))
    parser.add_argument("--wait-timeout-seconds", type=float, default=float(os.getenv("STORAGE_PRESSURE_CLEARANCE_WAIT_TIMEOUT_SECONDS", "180")))
    parser.add_argument("--command-timeout-seconds", type=int, default=int(os.getenv("STORAGE_PRESSURE_CLEARANCE_TIMEOUT_SECONDS", "900")))
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    out_file = Path(args.out_file).expanduser()
    lock_file = Path(args.lock_file).expanduser()
    lock_file.parent.mkdir(parents=True, exist_ok=True)

    with lock_file.open("a+", encoding="utf-8") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            payload = {
                "timestamp_utc": iso_now(),
                "schema_version": 1,
                "ok": True,
                "overall_status": "already_running",
                "busy": True,
            }
            write_payload(out_file, payload)
            if args.json:
                print(json.dumps(payload, ensure_ascii=True))
            else:
                print("storage_pressure_clearance overall_status=already_running")
            return 0

        payload = build_payload(
            project_root,
            apply=bool(args.apply),
            force_clear_stale_gate=bool(args.force_clear_stale_gate),
            skip_checkpoint=bool(args.skip_checkpoint),
            checkpoint_mode=str(args.checkpoint_mode),
            max_cycles=int(args.max_cycles),
            poll_seconds=float(args.poll_seconds),
            wait_timeout_seconds=float(args.wait_timeout_seconds),
            command_timeout_seconds=int(args.command_timeout_seconds),
        )
        write_payload(out_file, payload)

    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        print(
            "storage_pressure_clearance "
            f"overall_status={payload.get('overall_status', '')} "
            f"active_pressure={int(bool((payload.get('metrics') or {}).get('active_storage_pressure', False)))} "
            f"force_refused={payload.get('force_clear_refused_reason', '') or 'none'}"
        )
    return 0 if str(payload.get("overall_status") or "") in {"ready", "degraded", "already_running"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
