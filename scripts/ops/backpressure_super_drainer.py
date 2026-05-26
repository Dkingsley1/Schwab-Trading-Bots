#!/usr/bin/env python3
from __future__ import annotations

import argparse
import fcntl
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from core.runtime_python import resolve_runtime_python
    from scripts.ops import backpressure_drainer_fleet as drainer_src
    from scripts.ops import drainer_intelligence_layer as intelligence_src
    from scripts.ops import writer_cycle_coordinator as coordinator_src
    from scripts.ops.long_runtime_common import iso_now, load_json, ordered_unique, write_payload
else:
    from core.runtime_python import resolve_runtime_python
    from . import backpressure_drainer_fleet as drainer_src
    from . import drainer_intelligence_layer as intelligence_src
    from . import writer_cycle_coordinator as coordinator_src
    from .long_runtime_common import PROJECT_ROOT, iso_now, load_json, ordered_unique, write_payload


PY = resolve_runtime_python(PROJECT_ROOT)
DEFAULT_OUT_PATH = PROJECT_ROOT / "governance" / "health" / "backpressure_super_drainer_latest.json"
DEFAULT_MEMORY_PATH = PROJECT_ROOT / "governance" / "health" / "backpressure_super_drainer_memory_latest.json"
DEFAULT_LOCK_PATH = PROJECT_ROOT / "governance" / "locks" / "backpressure_super_drainer.lock"
DEFAULT_MAX_WAVES = 3
DEFAULT_TARGET_PENDING_LINES = 5_000
DEFAULT_MIN_PROGRESS_ROWS = 2_500
DEFAULT_COMMAND_TIMEOUT_SECONDS = 1_200
DEFAULT_WAIT_TIMEOUT_SECONDS = 30.0
DEFAULT_POLL_SECONDS = 2.0


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


def _parse_json_output(text: str) -> dict[str, Any]:
    for line in reversed([raw.strip() for raw in str(text or "").splitlines() if raw.strip()]):
        try:
            payload = json.loads(line)
        except Exception:
            continue
        if isinstance(payload, dict):
            return payload
    return {}


def _run_json_command(
    cmd: list[str],
    *,
    cwd: Path,
    payload_path: Path | None = None,
    timeout_sec: int,
) -> dict[str, Any]:
    started = datetime.now(timezone.utc)
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
    payload = _parse_json_output(stdout)
    if not payload and payload_path is not None:
        payload = load_json(payload_path)
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


def _step_record(result: dict[str, Any], *, status: str) -> dict[str, Any]:
    return {
        "status": status,
        "rc": _safe_int(result.get("rc"), 1),
        "duration_ms": _safe_float(result.get("duration_ms"), 0.0),
        "timed_out": bool(result.get("timed_out", False)),
        "cmd": list(result.get("cmd") or []),
        "stdout_tail": str(result.get("stdout_tail") or ""),
        "stderr_tail": str(result.get("stderr_tail") or ""),
    }


def _storage_snapshot(project_root: Path) -> dict[str, Any]:
    health = project_root / "governance" / "health"
    control = load_json(health / "ingestion_storage_control_latest.json")
    backpressure = load_json(health / "ingestion_backpressure_latest.json")
    control_bp = control.get("backpressure") if isinstance(control.get("backpressure"), dict) else {}
    queue_watermarks = control.get("queue_watermarks") if isinstance(control.get("queue_watermarks"), dict) else {}
    lanes = queue_watermarks.get("lanes") if isinstance(queue_watermarks.get("lanes"), dict) else {}

    def lane_pending(name: str) -> int:
        lane = lanes.get(name) if isinstance(lanes.get(name), dict) else {}
        return _safe_int(lane.get("pending_lines"), 0)

    core_pending = _safe_int(
        control_bp.get("core_pending_lines"),
        max(_safe_int(backpressure.get("pending_lines"), 0), lane_pending("core")),
    )
    deferred_pending = _safe_int(
        control_bp.get("deferred_pending_lines"),
        max(_safe_int(backpressure.get("pending_lines_deferred"), 0), lane_pending("deferred")),
    )
    cold_pending = _safe_int(
        control_bp.get("cold_pending_lines"),
        max(_safe_int(backpressure.get("pending_lines_cold"), 0), lane_pending("cold")),
    )
    support_pending = _safe_int(
        control_bp.get("support_pending_lines"),
        max(_safe_int(backpressure.get("pending_lines_support_telemetry"), 0), lane_pending("support_telemetry")),
    )
    stale_pending = _safe_int(
        control_bp.get("stale_stage_pending_lines"),
        max(_safe_int(backpressure.get("pending_lines_stale_stage"), 0), lane_pending("stale_stage")),
    )
    total_pending = _safe_int(
        control_bp.get("total_pending_lines"),
        max(
            _safe_int(backpressure.get("pending_lines_total"), 0),
            core_pending + deferred_pending + cold_pending + support_pending + stale_pending,
        ),
    )
    if total_pending <= 0 and backpressure:
        total_pending = max(
            _safe_int(backpressure.get("pending_lines_total"), 0),
            _safe_int(backpressure.get("pending_lines"), 0)
            + _safe_int(backpressure.get("pending_lines_deferred"), 0)
            + _safe_int(backpressure.get("pending_lines_cold"), 0)
            + _safe_int(backpressure.get("pending_lines_support_telemetry"), 0)
            + _safe_int(backpressure.get("pending_lines_stale_stage"), 0),
        )

    storage = control.get("storage") if isinstance(control.get("storage"), dict) else {}
    sql_overlay = control.get("sql_ingestion_pending_overlay") if isinstance(control.get("sql_ingestion_pending_overlay"), dict) else {}
    return {
        "timestamp_utc": str(control.get("timestamp_utc") or backpressure.get("timestamp_utc") or ""),
        "overall_status": str(control.get("overall_status") or "unknown"),
        "severity": str(control.get("severity") or "unknown"),
        "pressure_index": round(_safe_float(control.get("pressure_index"), 0.0), 6),
        "core_pending_lines": int(core_pending),
        "deferred_pending_lines": int(deferred_pending),
        "cold_pending_lines": int(cold_pending),
        "support_pending_lines": int(support_pending),
        "stale_stage_pending_lines": int(stale_pending),
        "total_pending_lines": int(total_pending),
        "estimated_total_drain_minutes": round(_safe_float(control_bp.get("estimated_total_drain_minutes"), 0.0), 3),
        "oldest_pending_age_seconds": round(
            max(
                _safe_float(control_bp.get("oldest_pending_age_seconds"), 0.0),
                _safe_float(backpressure.get("oldest_pending_age_seconds_total"), 0.0),
                _safe_float(backpressure.get("oldest_pending_age_seconds"), 0.0),
            ),
            3,
        ),
        "retention_debt_gb": round(_safe_float(storage.get("retention_debt_gb"), 0.0), 3),
        "queue_watermarks_status": str(queue_watermarks.get("overall_status") or ""),
        "queue_watermarks_source": str(control.get("queue_watermarks_source") or ""),
        "sql_ingestion_pending_overlay": sql_overlay,
    }


def _drainer_preview(project_root: Path, *, force_live_window: bool = False) -> dict[str, Any]:
    try:
        return drainer_src.build_payload(project_root, apply=False, force_live_window=bool(force_live_window))
    except TypeError:
        return drainer_src.build_payload(project_root, apply=False)
    except Exception as exc:
        return {"overall_status": "error", "error": f"{exc.__class__.__name__}: {exc}"}


def _coordinator_preview(
    project_root: Path,
    *,
    poll_seconds: float,
    wait_timeout_seconds: float,
    command_timeout_seconds: int,
    sql_manager_timeout_cap_seconds: int,
    include_maintenance: bool,
) -> dict[str, Any]:
    try:
        return coordinator_src.build_payload(
            project_root,
            apply=False,
            poll_seconds=float(poll_seconds),
            wait_timeout_seconds=float(wait_timeout_seconds),
            command_timeout_seconds=int(command_timeout_seconds),
            sql_manager_timeout_cap_seconds=int(sql_manager_timeout_cap_seconds),
            skip_maintenance=not bool(include_maintenance),
        )
    except Exception as exc:
        return {"overall_status": "error", "ok": False, "actionable": False, "error": f"{exc.__class__.__name__}: {exc}"}


def _writer_snapshot(project_root: Path) -> dict[str, Any]:
    try:
        return coordinator_src.writer_state_snapshot(project_root)
    except Exception as exc:
        return {"active": False, "error": f"{exc.__class__.__name__}: {exc}"}


def _active_drainer(preview: dict[str, Any]) -> dict[str, Any]:
    active = preview.get("active_drainer") if isinstance(preview.get("active_drainer"), dict) else {}
    return active if isinstance(active, dict) else {}


def _live_drainer_ready(preview: dict[str, Any], *, force_live_window: bool = False) -> bool:
    active = _active_drainer(preview)
    blocked = [str(item) for item in list(preview.get("blocked_reasons") or [])]
    if force_live_window:
        blocked = [item for item in blocked if item != "market_hours_guard"]
    return bool(
        str(preview.get("overall_status") or "") in {"ready", "handoff_requested"}
        and str(active.get("status") or "") == "ready"
        and (bool(active.get("live_window_safe", False)) or bool(force_live_window))
        and not blocked
    )


def _active_drainer_name(preview: dict[str, Any]) -> str:
    return str(_active_drainer(preview).get("name") or "")


def _ready_drainer_names(preview: dict[str, Any]) -> list[str]:
    rows = preview.get("candidate_drainers") if isinstance(preview.get("candidate_drainers"), list) else []
    return ordered_unique(
        [str(row.get("name") or "") for row in rows if isinstance(row, dict) and str(row.get("status") or "") == "ready"]
    )


def _planned_wave_count(total_pending: int, target_pending: int, max_waves: int) -> int:
    if total_pending <= target_pending:
        return 0
    gap = max(int(total_pending) - int(target_pending), 1)
    waves = 1 + ((gap - 1) // 50_000)
    return max(1, min(int(max_waves), int(waves)))


def _pressure_class(storage: dict[str, Any], *, target_pending_lines: int) -> str:
    total_pending = _safe_int(storage.get("total_pending_lines"), 0)
    pressure_index = _safe_float(storage.get("pressure_index"), 0.0)
    if total_pending <= target_pending_lines and pressure_index < 0.15:
        return "calm"
    if total_pending >= 250_000 or pressure_index >= 0.85:
        return "critical"
    if total_pending >= 50_000 or pressure_index >= 0.5:
        return "elevated"
    return "watch"


def _drain_strategy(
    storage: dict[str, Any],
    drainer_preview: dict[str, Any],
    coordinator_preview: dict[str, Any],
    writer_state: dict[str, Any],
    *,
    target_pending_lines: int,
    planned_waves: int,
    max_waves: int,
    include_maintenance: bool,
) -> dict[str, Any]:
    pressure_class = _pressure_class(storage, target_pending_lines=target_pending_lines)
    total_pending = _safe_int(storage.get("total_pending_lines"), 0)
    active_drainer = _active_drainer_name(drainer_preview)
    ready_names = _ready_drainer_names(drainer_preview)
    writer_active = bool(writer_state.get("active", False))
    if pressure_class == "critical":
        posture = "drain_aggressively_but_single_writer"
        recommended_target = min(target_pending_lines, 2500)
    elif pressure_class == "elevated":
        posture = "run_bounded_waves"
        recommended_target = target_pending_lines
    elif pressure_class == "watch":
        posture = "opportunistic_micro_drain"
        recommended_target = target_pending_lines
    else:
        posture = "park_and_observe"
        recommended_target = target_pending_lines
    if writer_active:
        posture = "wait_for_writer_then_re_score"
    return {
        "pressure_class": pressure_class,
        "posture": posture,
        "active_drainer": active_drainer,
        "ready_drainer_names": ready_names,
        "total_pending_lines": total_pending,
        "target_pending_lines": int(target_pending_lines),
        "recommended_target_pending_lines": int(recommended_target),
        "planned_waves": int(planned_waves),
        "max_waves": int(max_waves),
        "writer_active": writer_active,
        "coordinator_actionable": bool(coordinator_preview.get("actionable", False)),
        "live_drainer_ready": _live_drainer_ready(drainer_preview),
        "include_maintenance": bool(include_maintenance),
        "wave_ladder": [
            "score_backpressure_and_select_lane",
            "write_fleet_handoff",
            "run_writer_cycle_coordinator_single_writer",
            "refresh_storage_snapshot",
            "record_drainer_memory",
            "refresh_system_self_model",
        ],
    }


def _grandmaster_context_packet(
    *,
    strategy: dict[str, Any],
    initial_storage: dict[str, Any],
    final_storage: dict[str, Any],
    summary: dict[str, Any],
    guardrails: dict[str, Any],
) -> dict[str, Any]:
    return {
        "packet_kind": "drainer_self_intelligence_context",
        "pressure_class": str(strategy.get("pressure_class") or ""),
        "recommended_posture": str(strategy.get("posture") or ""),
        "active_drainer": str(strategy.get("active_drainer") or ""),
        "initial_pending_lines": _safe_int(initial_storage.get("total_pending_lines"), 0),
        "final_pending_lines": _safe_int(final_storage.get("total_pending_lines"), 0),
        "target_pending_lines": _safe_int(strategy.get("target_pending_lines"), 0),
        "waves_run": _safe_int(summary.get("waves_run"), 0),
        "progress_waves": _safe_int(summary.get("progress_waves"), 0),
        "stop_reason": str(summary.get("stop_reason") or ""),
        "single_writer_guard": bool(guardrails.get("single_writer_only", False)) and not bool(guardrails.get("starts_parallel_sql_writers", True)),
        "safe_next_action": "park" if _safe_int(final_storage.get("total_pending_lines"), 0) <= _safe_int(strategy.get("target_pending_lines"), 0) else "run_next_bounded_wave",
    }


def _drainer_memory_payload(payload: dict[str, Any], previous: dict[str, Any] | None = None) -> dict[str, Any]:
    previous = previous if isinstance(previous, dict) else {}
    history = previous.get("history") if isinstance(previous.get("history"), list) else []
    summary = payload.get("summary") if isinstance(payload.get("summary"), dict) else {}
    event = {
        "timestamp_utc": str(payload.get("timestamp_utc") or iso_now()),
        "overall_status": str(payload.get("overall_status") or ""),
        "active_drainer": str(payload.get("active_drainer") or summary.get("active_drainer") or ""),
        "pressure_class": str((payload.get("drainer_strategy") or {}).get("pressure_class") or "") if isinstance(payload.get("drainer_strategy"), dict) else "",
        "initial_pending_lines": _safe_int(summary.get("initial_pending_lines"), 0),
        "final_pending_lines": _safe_int(summary.get("final_pending_lines"), 0),
        "pending_lines_delta": _safe_int(summary.get("pending_lines_delta"), 0),
        "waves_run": _safe_int(summary.get("waves_run"), 0),
        "progress_waves": _safe_int(summary.get("progress_waves"), 0),
        "stop_reason": str(summary.get("stop_reason") or payload.get("stop_reason") or ""),
        "target_met": bool(payload.get("target_met_final", False)),
    }
    event["pending_lines_net_change"] = int(event["final_pending_lines"] - event["initial_pending_lines"])
    event["refill_detected"] = bool(event["waves_run"] > 0 and event["pending_lines_net_change"] > 0)
    history = [row for row in history if isinstance(row, dict)] + [event]
    history = history[-160:]
    recent = history[-30:]
    progress_events = [row for row in recent if _safe_int(row.get("pending_lines_delta"), 0) > 0 or bool(row.get("target_met", False))]
    target_events = [row for row in recent if bool(row.get("target_met", False))]
    refill_events = [
        row
        for row in recent
        if bool(row.get("refill_detected", False))
        or (_safe_int(row.get("waves_run"), 0) > 0 and _safe_int(row.get("final_pending_lines"), 0) > _safe_int(row.get("initial_pending_lines"), 0))
    ]
    return {
        "timestamp_utc": event["timestamp_utc"],
        "schema_version": 1,
        "overall_status": "ready",
        "latest_event": event,
        "history_count": len(history),
        "recent_window_count": len(recent),
        "recent_progress_event_count": len(progress_events),
        "recent_target_met_count": len(target_events),
        "recent_refill_event_count": len(refill_events),
        "recent_progress_rate": round(len(progress_events) / max(len(recent), 1), 4),
        "recent_target_met_rate": round(len(target_events) / max(len(recent), 1), 4),
        "recent_refill_rate": round(len(refill_events) / max(len(recent), 1), 4),
        "history": history,
        "memory_contract": "remember_drainer_waves_targets_progress_stalls_timeouts_and_active_lanes_for_self_model_reasoning",
    }


def write_memory(payload: dict[str, Any], memory_path: Path = DEFAULT_MEMORY_PATH) -> dict[str, Any]:
    previous = load_json(memory_path)
    memory = _drainer_memory_payload(payload, previous)
    write_payload(memory_path, memory)
    return memory


def _coordinator_command(
    project_root: Path,
    *,
    poll_seconds: float,
    wait_timeout_seconds: float,
    command_timeout_seconds: int,
    sql_manager_timeout_cap_seconds: int,
    include_maintenance: bool,
    stale_progress_minutes: float,
) -> list[str]:
    cmd = [
        str(PY),
        str(project_root / "scripts" / "ops" / "writer_cycle_coordinator.py"),
        "--apply",
        "--poll-seconds",
        str(float(poll_seconds)),
        "--wait-timeout-seconds",
        str(float(wait_timeout_seconds)),
        "--command-timeout-seconds",
        str(int(command_timeout_seconds)),
        "--stale-progress-minutes",
        str(float(stale_progress_minutes)),
    ]
    if int(sql_manager_timeout_cap_seconds) > 0:
        cmd.extend(["--sql-manager-timeout-cap-seconds", str(int(sql_manager_timeout_cap_seconds))])
    if not include_maintenance:
        cmd.append("--skip-maintenance")
    cmd.append("--json")
    return cmd


def _refresh_storage_command(project_root: Path) -> list[str]:
    return [str(PY), str(project_root / "scripts" / "ops" / "ingestion_storage_control.py"), "--json"]


def _merged_rows_from_coordinator(payload: dict[str, Any]) -> int:
    summary = payload.get("summary") if isinstance(payload.get("summary"), dict) else {}
    candidates = [_safe_int(summary.get("writer_merged_rows_delta"), 0)]
    for key in ("writer_state_after_wait", "writer_state_after_remediation", "writer_state_before"):
        state = payload.get(key) if isinstance(payload.get(key), dict) else {}
        candidates.append(_safe_int(state.get("merged_rows_this_cycle"), 0))
    return max(candidates or [0])


def _partial_progress_flag(payload: dict[str, Any]) -> bool:
    summary = payload.get("summary") if isinstance(payload.get("summary"), dict) else {}
    if bool(summary.get("partial_progress", False)):
        return True
    steps = payload.get("steps") if isinstance(payload.get("steps"), dict) else {}
    return any(isinstance(row, dict) and str(row.get("status") or "") == "partial_progress" for row in steps.values())


def _wave_status(result: dict[str, Any], payload: dict[str, Any], *, progress_observed: bool) -> str:
    if bool(result.get("timed_out", False)):
        return "timed_out_with_progress" if progress_observed else "timed_out"
    if _safe_int(result.get("rc"), 1) == 0:
        status = str(payload.get("overall_status") or "ok")
        return status if status else "ok"
    if progress_observed:
        return "partial_progress"
    return "error"


def _wave_progress(before: dict[str, Any], after: dict[str, Any], coordinator_payload: dict[str, Any], *, min_progress_rows: int) -> dict[str, Any]:
    pending_before = _safe_int(before.get("total_pending_lines"), 0)
    pending_after = _safe_int(after.get("total_pending_lines"), pending_before)
    pending_delta = max(pending_before - pending_after, 0)
    merged_rows = _merged_rows_from_coordinator(coordinator_payload)
    partial_progress = _partial_progress_flag(coordinator_payload)
    progress_observed = bool(
        pending_delta > 0
        or merged_rows >= max(int(min_progress_rows), 1)
        or partial_progress
        or str(coordinator_payload.get("overall_status") or "") in {"applied", "applied_with_followups", "progressing_waiting_for_writer"}
    )
    return {
        "pending_lines_before": int(pending_before),
        "pending_lines_after": int(pending_after),
        "pending_lines_delta": int(pending_delta),
        "merged_rows_observed": int(merged_rows),
        "partial_progress": bool(partial_progress),
        "progress_observed": bool(progress_observed),
    }


def _recommended_actions(
    *,
    actionable: bool,
    target_met: bool,
    live_drainer_ready: bool,
    active_drainer: str,
    writer_active: bool,
    waves: list[dict[str, Any]],
    include_maintenance: bool,
) -> list[str]:
    actions: list[str] = []
    if target_met:
        actions.append("keep the super drainer parked; backlog is already inside the steady-state target")
    if actionable and live_drainer_ready and active_drainer:
        actions.append(f"run bounded super-drain waves through the current active lane: {active_drainer}")
    if writer_active:
        actions.append("let the coordinator wait on the current SQL writer instead of starting a parallel writer")
    if waves and bool(waves[-1].get("progress", {}).get("progress_observed", False)):
        actions.append("schedule another bounded wave only after the storage snapshot re-scores the queue")
    if not include_maintenance:
        actions.append("keep retention maintenance out of the hot drain wave unless the backlog is stable and the writer is idle")
    actions.append("preserve single-writer SQLite discipline; widen by sequencing waves, not by adding competing writers")
    return ordered_unique(actions)[:8]


def build_payload(
    project_root: Path = PROJECT_ROOT,
    *,
    apply: bool,
    max_waves: int = DEFAULT_MAX_WAVES,
    target_pending_lines: int = DEFAULT_TARGET_PENDING_LINES,
    min_progress_rows: int = DEFAULT_MIN_PROGRESS_ROWS,
    poll_seconds: float = DEFAULT_POLL_SECONDS,
    wait_timeout_seconds: float = DEFAULT_WAIT_TIMEOUT_SECONDS,
    command_timeout_seconds: int = DEFAULT_COMMAND_TIMEOUT_SECONDS,
    sql_manager_timeout_cap_seconds: int = 0,
    stale_progress_minutes: float = coordinator_src.DEFAULT_STALE_PROGRESS_MINUTES,
    include_maintenance: bool = False,
    force_live_window: bool = False,
    cooldown_seconds: float = 0.0,
) -> dict[str, Any]:
    project_root = Path(project_root).resolve()
    max_waves = max(int(max_waves), 1)
    target_pending_lines = max(int(target_pending_lines), 0)
    min_progress_rows = max(int(min_progress_rows), 1)

    started = time.monotonic()
    initial_storage = _storage_snapshot(project_root)
    initial_drainer = _drainer_preview(project_root, force_live_window=bool(force_live_window))
    coordinator_preview = _coordinator_preview(
        project_root,
        poll_seconds=float(poll_seconds),
        wait_timeout_seconds=float(wait_timeout_seconds),
        command_timeout_seconds=int(command_timeout_seconds),
        sql_manager_timeout_cap_seconds=int(sql_manager_timeout_cap_seconds),
        include_maintenance=bool(include_maintenance),
    )
    writer_before = _writer_snapshot(project_root)
    total_pending = _safe_int(initial_storage.get("total_pending_lines"), 0)
    target_met = bool(total_pending <= target_pending_lines)
    live_ready = _live_drainer_ready(initial_drainer, force_live_window=bool(force_live_window))
    coordinator_actionable = bool(coordinator_preview.get("actionable", False))
    actionable = bool(not target_met and (live_ready or coordinator_actionable))
    active_drainer = _active_drainer_name(initial_drainer)
    planned_waves = _planned_wave_count(total_pending, target_pending_lines, max_waves)
    drainer_strategy = _drain_strategy(
        initial_storage,
        initial_drainer,
        coordinator_preview,
        writer_before,
        target_pending_lines=target_pending_lines,
        planned_waves=planned_waves,
        max_waves=max_waves,
        include_maintenance=include_maintenance,
    )

    waves: list[dict[str, Any]] = []
    stop_reason = "preview_only" if not apply else "not_started"
    final_storage = initial_storage
    refresh_steps: list[dict[str, Any]] = []

    if apply and actionable:
        for wave_number in range(1, max_waves + 1):
            wave_before_storage = _storage_snapshot(project_root)
            if _safe_int(wave_before_storage.get("total_pending_lines"), 0) <= target_pending_lines:
                stop_reason = "target_already_met"
                final_storage = wave_before_storage
                break

            wave_drainer = _drainer_preview(project_root, force_live_window=bool(force_live_window))
            wave_live_ready = _live_drainer_ready(wave_drainer, force_live_window=bool(force_live_window))
            wave_coordinator = _coordinator_preview(
                project_root,
                poll_seconds=float(poll_seconds),
                wait_timeout_seconds=float(wait_timeout_seconds),
                command_timeout_seconds=int(command_timeout_seconds),
                sql_manager_timeout_cap_seconds=int(sql_manager_timeout_cap_seconds),
                include_maintenance=bool(include_maintenance),
            )
            if not wave_live_ready and not bool(wave_coordinator.get("actionable", False)):
                stop_reason = "no_actionable_drainer"
                final_storage = wave_before_storage
                break

            cmd = _coordinator_command(
                project_root,
                poll_seconds=float(poll_seconds),
                wait_timeout_seconds=float(wait_timeout_seconds),
                command_timeout_seconds=int(command_timeout_seconds),
                sql_manager_timeout_cap_seconds=int(sql_manager_timeout_cap_seconds),
                include_maintenance=bool(include_maintenance),
                stale_progress_minutes=float(stale_progress_minutes),
            )
            coordinator_result = _run_json_command(
                cmd,
                cwd=project_root,
                payload_path=project_root / "governance" / "health" / "writer_cycle_coordinator_latest.json",
                timeout_sec=max(int(command_timeout_seconds) + 90, int(wait_timeout_seconds) + 120),
            )
            coordinator_payload = coordinator_result.get("payload") if isinstance(coordinator_result.get("payload"), dict) else {}

            refresh_result = _run_json_command(
                _refresh_storage_command(project_root),
                cwd=project_root,
                payload_path=project_root / "governance" / "health" / "ingestion_storage_control_latest.json",
                timeout_sec=180,
            )
            refresh_steps.append(_step_record(refresh_result, status="ok" if _safe_int(refresh_result.get("rc"), 1) == 0 else "error"))
            wave_after_storage = _storage_snapshot(project_root)
            progress = _wave_progress(
                wave_before_storage,
                wave_after_storage,
                coordinator_payload,
                min_progress_rows=min_progress_rows,
            )
            status = _wave_status(coordinator_result, coordinator_payload, progress_observed=bool(progress.get("progress_observed", False)))
            waves.append(
                {
                    "wave": int(wave_number),
                    "active_drainer": _active_drainer_name(wave_drainer),
                    "live_drainer_ready": bool(wave_live_ready),
                    "ready_drainer_names": _ready_drainer_names(wave_drainer),
                    "coordinator_preview_status": str(wave_coordinator.get("overall_status") or ""),
                    "coordinator_preview_actionable": bool(wave_coordinator.get("actionable", False)),
                    "coordinator_step": _step_record(coordinator_result, status=status),
                    "coordinator_overall_status": str(coordinator_payload.get("overall_status") or ""),
                    "storage_before": wave_before_storage,
                    "storage_after": wave_after_storage,
                    "progress": progress,
                }
            )
            final_storage = wave_after_storage
            if _safe_int(wave_after_storage.get("total_pending_lines"), 0) <= target_pending_lines:
                stop_reason = "target_cleared"
                break
            if not bool(progress.get("progress_observed", False)):
                stop_reason = "progress_stalled"
                break
            if wave_number >= max_waves:
                stop_reason = "max_waves_reached"
                break
            if cooldown_seconds > 0:
                time.sleep(max(float(cooldown_seconds), 0.0))
    elif apply and target_met:
        stop_reason = "target_already_met"
    elif apply and not actionable:
        stop_reason = "nothing_actionable"

    any_progress = any(bool(row.get("progress", {}).get("progress_observed", False)) for row in waves)
    any_hard_error = any(
        str(row.get("coordinator_step", {}).get("status") or "") in {"error", "timed_out"} for row in waves
    )
    final_pending = _safe_int(final_storage.get("total_pending_lines"), total_pending)
    final_target_met = bool(final_pending <= target_pending_lines)

    if not apply:
        if target_met or not actionable:
            overall_status = "idle"
            ok = True
        elif bool(writer_before.get("active", False)):
            overall_status = "waiting_for_writer"
            ok = True
        else:
            overall_status = "ready"
            ok = True
    elif not waves and final_target_met:
        overall_status = "idle"
        ok = True
    elif final_target_met:
        overall_status = "applied"
        ok = True
    elif any_progress:
        overall_status = "applied_with_followups"
        ok = True
    elif any_hard_error:
        overall_status = "apply_failed"
        ok = False
    elif stop_reason == "progress_stalled":
        overall_status = "stalled"
        ok = False
    elif not actionable:
        overall_status = "idle"
        ok = True
    else:
        overall_status = "blocked"
        ok = False

    elapsed_ms = round((time.monotonic() - started) * 1000.0, 3)
    summary = {
        "elapsed_ms": elapsed_ms,
        "initial_pending_lines": int(total_pending),
        "final_pending_lines": int(final_pending),
        "pending_lines_delta": int(max(total_pending - final_pending, 0)),
        "waves_run": int(len(waves)),
        "progress_waves": int(sum(1 for row in waves if bool(row.get("progress", {}).get("progress_observed", False)))),
        "any_progress": bool(any_progress),
        "any_hard_error": bool(any_hard_error),
        "writer_active_initial": bool(writer_before.get("active", False)),
        "active_drainer": active_drainer,
        "stop_reason": stop_reason,
    }
    latest_progress = waves[-1].get("progress", {}) if waves and isinstance(waves[-1], dict) else {}
    writer_progress_observed = any(_safe_int(row.get("progress", {}).get("merged_rows_observed"), 0) > 0 for row in waves)
    snapshot_reconciliation = {
        "pending_snapshot_changed": bool(summary["pending_lines_delta"] > 0),
        "writer_progress_observed": bool(writer_progress_observed),
        "likely_snapshot_lag": bool(writer_progress_observed and summary["pending_lines_delta"] <= 0 and len(waves) > 0),
        "latest_pending_lines_before": _safe_int(latest_progress.get("pending_lines_before"), total_pending),
        "latest_pending_lines_after": _safe_int(latest_progress.get("pending_lines_after"), final_pending),
        "latest_merged_rows_observed": _safe_int(latest_progress.get("merged_rows_observed"), 0),
    }
    guardrails = {
        "single_writer_only": True,
        "uses_writer_cycle_coordinator": True,
        "starts_parallel_sql_writers": False,
        "stops_on_progress_stall": True,
        "stops_when_target_met": True,
        "retention_maintenance_included": bool(include_maintenance),
    }
    grandmaster_packet = _grandmaster_context_packet(
        strategy=drainer_strategy,
        initial_storage=initial_storage,
        final_storage=final_storage,
        summary=summary,
        guardrails=guardrails,
    )
    payload = {
        "timestamp_utc": iso_now(),
        "schema_version": 1,
        "ok": bool(ok),
        "overall_status": overall_status,
        "apply": bool(apply),
        "mode": "backpressure_super_drainer",
        "capabilities": [
            "multi_wave_drain_coordination",
            "live_safe_drainer_lane_selection",
            "single_sql_writer_preservation",
            "timeout_aware_partial_progress_acceptance",
            "storage_snapshot_refresh_after_each_wave",
        "progress_stall_guard",
        "self_intelligence_state_vector",
        "drainer_intelligence_layer",
        "drainer_memory_feedback",
        "grandmaster_context_packet",
        ],
        "guardrails": guardrails,
        "self_intelligence_contract": {
            "included_in_system_self_model": True,
            "awareness_domain": "drainer_intelligence",
            "refreshes_self_model_after_opsctl_command": True,
            "feeds_grandmaster_context": True,
            "learns_from_wave_history": True,
            "uses_drainer_intelligence_layer": True,
            "authority_boundary": "advisory_and_queue_coordination_only_no_execution_authority",
        },
        "drainer_strategy": drainer_strategy,
        "grandmaster_context_packet": grandmaster_packet,
        "assigned_infrabots": [
            "backpressure_super_drainer",
            "drainer_intelligence_layer",
            "backpressure_drainer_fleet",
            "backpressure_slo_bot",
            "storage_backpressure_autopilot",
            "storage_pressure_clearance_bot",
            "brain_refinery_v602_system_governor_cpu_memory_backlog_pressure_bot",
        ],
        "settings": {
            "max_waves": int(max_waves),
            "planned_wave_count": int(planned_waves),
            "target_pending_lines": int(target_pending_lines),
            "min_progress_rows": int(min_progress_rows),
            "poll_seconds": float(poll_seconds),
            "wait_timeout_seconds": float(wait_timeout_seconds),
            "command_timeout_seconds": int(command_timeout_seconds),
            "sql_manager_timeout_cap_seconds": int(sql_manager_timeout_cap_seconds),
            "stale_progress_minutes": float(stale_progress_minutes),
            "include_maintenance": bool(include_maintenance),
            "force_live_window": bool(force_live_window),
            "cooldown_seconds": float(cooldown_seconds),
        },
        "actionable": bool(actionable),
        "target_met_initially": bool(target_met),
        "target_met_final": bool(final_target_met),
        "live_drainer_ready": bool(live_ready),
        "coordinator_actionable": bool(coordinator_actionable),
        "active_drainer": active_drainer,
        "initial_pending_lines": int(total_pending),
        "final_pending_lines": int(final_pending),
        "pending_lines_delta": int(summary["pending_lines_delta"]),
        "progress_waves": int(summary["progress_waves"]),
        "snapshot_reconciliation": snapshot_reconciliation,
        "ready_drainer_names": _ready_drainer_names(initial_drainer),
        "writer_state_before": writer_before,
        "initial_storage": initial_storage,
        "final_storage": final_storage,
        "drainer_preview": {
            "overall_status": str(initial_drainer.get("overall_status") or ""),
            "active_drainer": active_drainer,
            "ready_drainer_count": _safe_int(initial_drainer.get("ready_drainer_count"), 0),
            "blocked_reasons": list(initial_drainer.get("blocked_reasons") or []),
            "next_drainer_queue": list(initial_drainer.get("next_drainer_queue") or [])[:8],
        },
        "coordinator_preview": {
            "overall_status": str(coordinator_preview.get("overall_status") or ""),
            "actionable": bool(coordinator_preview.get("actionable", False)),
            "live_drainer_ready": bool(coordinator_preview.get("live_drainer_ready", False)),
            "drain_ready": bool(coordinator_preview.get("drain_ready", False)),
            "maintenance_ready": bool(coordinator_preview.get("maintenance_ready", False)),
        },
        "waves": waves,
        "refresh_steps": refresh_steps,
        "stop_reason": stop_reason,
        "recommended_actions": _recommended_actions(
            actionable=bool(actionable),
            target_met=bool(final_target_met),
            live_drainer_ready=bool(live_ready),
            active_drainer=active_drainer,
            writer_active=bool(writer_before.get("active", False)),
            waves=waves,
            include_maintenance=bool(include_maintenance),
        ),
        "summary": summary,
    }
    try:
        intelligence_layer = intelligence_src.build_intelligence_from_payloads(
            fleet=initial_drainer,
            super_drainer=payload,
            memory=load_json(project_root / "governance" / "health" / "backpressure_super_drainer_memory_latest.json"),
            storage=final_storage,
            runtime=load_json(project_root / "governance" / "health" / "runtime_throttle_control_latest.json"),
            memory_efficiency=load_json(project_root / "governance" / "health" / "memory_efficiency_control_latest.json"),
            writer=writer_before,
            target_pending_lines=int(target_pending_lines),
        )
    except Exception as exc:
        intelligence_layer = {
            "timestamp_utc": iso_now(),
            "schema_version": 1,
            "ok": False,
            "overall_status": "degraded",
            "mode": "drainer_intelligence_layer",
            "error": f"{exc.__class__.__name__}: {exc}",
            "decision_packet": {
                "action": "fall_back_to_super_drainer_strategy",
                "selected_drainer": active_drainer,
                "confidence": 0.1,
            },
        }
    payload["drainer_intelligence_layer"] = intelligence_layer
    decision = intelligence_layer.get("decision_packet") if isinstance(intelligence_layer.get("decision_packet"), dict) else {}
    if isinstance(payload.get("grandmaster_context_packet"), dict):
        payload["grandmaster_context_packet"]["intelligence_action"] = str(decision.get("action") or "")
        payload["grandmaster_context_packet"]["intelligence_confidence"] = _safe_float(decision.get("confidence"), 0.0)
        payload["grandmaster_context_packet"]["intelligence_next_ready_drainer"] = str(decision.get("next_ready_drainer") or "")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Coordinate larger bounded backpressure drain waves without starting parallel SQLite writers.")
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--out-file", default=str(DEFAULT_OUT_PATH))
    parser.add_argument("--memory-file", default=str(DEFAULT_MEMORY_PATH))
    parser.add_argument("--lock-file", default=str(DEFAULT_LOCK_PATH))
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--max-waves", type=int, default=DEFAULT_MAX_WAVES)
    parser.add_argument("--target-pending-lines", type=int, default=DEFAULT_TARGET_PENDING_LINES)
    parser.add_argument("--min-progress-rows", type=int, default=DEFAULT_MIN_PROGRESS_ROWS)
    parser.add_argument("--poll-seconds", type=float, default=DEFAULT_POLL_SECONDS)
    parser.add_argument("--wait-timeout-seconds", type=float, default=DEFAULT_WAIT_TIMEOUT_SECONDS)
    parser.add_argument("--command-timeout-seconds", type=int, default=DEFAULT_COMMAND_TIMEOUT_SECONDS)
    parser.add_argument("--sql-manager-timeout-cap-seconds", type=int, default=int(os.getenv("BACKPRESSURE_SUPER_DRAINER_SQL_MANAGER_TIMEOUT_CAP_SECONDS", "0")))
    parser.add_argument("--stale-progress-minutes", type=float, default=float(coordinator_src.DEFAULT_STALE_PROGRESS_MINUTES))
    parser.add_argument("--include-maintenance", action="store_true")
    parser.add_argument("--force-live-window", action="store_true")
    parser.add_argument("--cooldown-seconds", type=float, default=0.0)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    out_file = Path(args.out_file).expanduser()
    memory_file = Path(args.memory_file).expanduser()
    lock_file = Path(args.lock_file).expanduser()
    lock_file.parent.mkdir(parents=True, exist_ok=True)

    payload: dict[str, Any] = {
        "timestamp_utc": iso_now(),
        "schema_version": 1,
        "ok": True,
        "overall_status": "pending",
    }
    with lock_file.open("a+", encoding="utf-8") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            payload.update({"overall_status": "already_running", "busy": True})
            write_payload(out_file, payload)
            if args.json:
                print(json.dumps(payload, ensure_ascii=True))
            else:
                print("backpressure_super_drainer overall_status=already_running")
            return 0
        handle.seek(0)
        handle.truncate(0)
        handle.write(f"pid={os.getpid()} timestamp_utc={iso_now()} cmd=backpressure_super_drainer\n")
        handle.flush()

        payload = build_payload(
            project_root,
            apply=bool(args.apply),
            max_waves=int(args.max_waves),
            target_pending_lines=int(args.target_pending_lines),
            min_progress_rows=int(args.min_progress_rows),
            poll_seconds=float(args.poll_seconds),
            wait_timeout_seconds=float(args.wait_timeout_seconds),
            command_timeout_seconds=int(args.command_timeout_seconds),
            sql_manager_timeout_cap_seconds=int(args.sql_manager_timeout_cap_seconds),
            stale_progress_minutes=float(args.stale_progress_minutes),
            include_maintenance=bool(args.include_maintenance),
            force_live_window=bool(args.force_live_window),
            cooldown_seconds=float(args.cooldown_seconds),
        )
        write_payload(out_file, payload)
        payload["drainer_memory"] = write_memory(payload, memory_file)
        write_payload(out_file, payload)

    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        print(
            "backpressure_super_drainer "
            f"overall_status={payload.get('overall_status', '')} "
            f"waves_run={payload.get('summary', {}).get('waves_run', 0)} "
            f"pending={payload.get('summary', {}).get('final_pending_lines', 0)}"
        )
    return 0 if bool(payload.get("ok", False) or str(payload.get("overall_status") or "") in {"already_running", "idle", "ready", "waiting_for_writer", "applied_with_followups"}) else 2


if __name__ == "__main__":
    raise SystemExit(main())
