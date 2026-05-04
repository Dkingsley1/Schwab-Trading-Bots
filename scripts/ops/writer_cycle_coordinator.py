#!/usr/bin/env python3
from __future__ import annotations

import argparse
import fcntl
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.runtime_python import resolve_runtime_python
from scripts.ops import external_backlog_drain as drain_src
from scripts.ops import storage_maintenance_lane as maintenance_src


PY = resolve_runtime_python(PROJECT_ROOT)
DEFAULT_OUT_PATH = PROJECT_ROOT / "governance" / "health" / "writer_cycle_coordinator_latest.json"
DEFAULT_LOCK_PATH = PROJECT_ROOT / "governance" / "locks" / "writer_cycle_coordinator.lock"
DEFAULT_WAIT_TIMEOUT_SECONDS = 900.0
DEFAULT_POLL_SECONDS = 20.0
DEFAULT_COMMAND_TIMEOUT_SECONDS = 2400
WRITER_LOCK_PATH = PROJECT_ROOT / "governance" / "locks" / "jsonl_sql_writer.lock"
RECENT_PROGRESS_MINUTES = 30.0
DEFAULT_STALE_PROGRESS_MINUTES = 30.0


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def _parse_json_output(text: str) -> dict[str, Any]:
    for line in reversed([raw.strip() for raw in str(text or "").splitlines() if raw.strip()]):
        try:
            payload = json.loads(line)
        except Exception:
            continue
        if isinstance(payload, dict):
            return payload
    return {}


def _safe_float(raw: Any, default: float = 0.0) -> float:
    try:
        return float(raw)
    except Exception:
        return float(default)


def _safe_int(raw: Any, default: int = 0) -> int:
    try:
        return int(float(raw))
    except Exception:
        return int(default)


def _ordered_unique(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for raw in items:
        text = str(raw or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out


def _age_minutes_from_timestamp(raw: Any, *, now_utc: datetime | None = None) -> float | None:
    text = str(raw or "").strip().replace("Z", "+00:00")
    if not text:
        return None
    try:
        dt = datetime.fromisoformat(text)
    except Exception:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    now = now_utc or datetime.now(timezone.utc)
    return max((now - dt.astimezone(timezone.utc)).total_seconds() / 60.0, 0.0)


def _progress_cycle_age_minutes(progress: dict[str, Any], *, now_utc: datetime | None = None) -> float | None:
    return _age_minutes_from_timestamp(progress.get("cycle_started_utc"), now_utc=now_utc)


def _lock_snapshot(lock_path: Path) -> dict[str, Any]:
    try:
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        with lock_path.open("a+", encoding="utf-8") as handle:
            handle.seek(0)
            owner = handle.read().strip()
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return {"owner": owner, "held": True}
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            except Exception:
                pass
            return {"owner": owner, "held": False}
    except Exception:
        return {"owner": "", "held": False}


def _lock_owner(lock_path: Path) -> str:
    return str(_lock_snapshot(lock_path).get("owner") or "")


def _parse_lock_owner_pid(owner: str) -> int | None:
    for token in str(owner or "").split():
        if token.startswith("pid="):
            try:
                pid = int(token.split("=", 1)[1])
            except Exception:
                return None
            return pid if pid > 0 else None
    return None


def _pid_exists(pid: int) -> bool:
    try:
        os.kill(int(pid), 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except Exception:
        return False
    return True


def _pid_command(pid: int) -> str:
    try:
        proc = subprocess.run(
            ["ps", "-p", str(int(pid)), "-o", "command="],
            capture_output=True,
            text=True,
            check=False,
            timeout=2,
        )
    except Exception:
        return ""
    return (proc.stdout or "").strip()


def _stale_writer_detected(state: dict[str, Any], *, stale_progress_minutes: float) -> bool:
    if not bool(state.get("active", False)) or not bool(state.get("writer_lock_held", False)):
        return False
    current_step = str(state.get("current_step") or "")
    if current_step in {"", "complete"}:
        return False
    stale_after = max(float(stale_progress_minutes), 1.0)
    age = state.get("progress_age_minutes")
    if age is not None and float(age) >= stale_after:
        return True

    cycle_age = state.get("cycle_age_minutes")
    if cycle_age is None:
        return False
    semantic_stall_after = max(stale_after * 3.0, 90.0)
    merged_rows = _safe_int(state.get("merged_rows_this_cycle"), 0)
    merge_count = _safe_int(state.get("completed_merge_count"), 0)
    shard_count = _safe_int(state.get("completed_shard_count"), 0)
    planned_shards = _safe_int(state.get("planned_shard_count"), 0)
    timeout_count = _safe_int(state.get("timed_out_shard_count"), 0)
    if float(cycle_age) < semantic_stall_after:
        return False
    if current_step == "shard_linking":
        if merged_rows <= 0 and merge_count <= 0 and (timeout_count > 0 or shard_count < max(planned_shards, 1)):
            return True
    if current_step == "merge_primary":
        merge_stall_after = max(stale_after * 4.0, 120.0)
        if float(cycle_age) >= merge_stall_after and merged_rows <= 0 and merge_count <= 0:
            return True
    return False


def _terminate_stale_writer(
    project_root: Path,
    state: dict[str, Any],
    *,
    stale_progress_minutes: float,
    grace_seconds: float = 5.0,
) -> dict[str, Any]:
    owner = str(state.get("writer_lock_owner") or "")
    pid = _parse_lock_owner_pid(owner)
    payload: dict[str, Any] = {
        "attempted": False,
        "needed": _stale_writer_detected(state, stale_progress_minutes=stale_progress_minutes),
        "pid": pid,
        "owner": owner,
        "terminated": False,
        "lock_released": False,
        "reason": "",
    }
    if not bool(payload["needed"]):
        payload["reason"] = "writer_not_stale"
        return payload
    if pid is None:
        payload["reason"] = "missing_writer_pid"
        return payload
    command = _pid_command(pid)
    payload["command"] = command
    if "sql_link_shard_manager.py" not in command and "sql_link_shard_manager" not in owner:
        payload["reason"] = "pid_not_sql_link_shard_manager"
        return payload
    try:
        os.kill(pid, signal.SIGTERM)
        payload["attempted"] = True
    except ProcessLookupError:
        payload["attempted"] = True
        payload["terminated"] = True
    except Exception as exc:
        payload["reason"] = f"terminate_failed:{exc.__class__.__name__}"
        return payload

    deadline = time.monotonic() + max(float(grace_seconds), 0.0)
    while _pid_exists(pid) and time.monotonic() < deadline:
        time.sleep(0.2)
    payload["terminated"] = not _pid_exists(pid)
    lock_state = _lock_snapshot(project_root / "governance" / "locks" / "jsonl_sql_writer.lock")
    payload["lock_released"] = not bool(lock_state.get("held", False))
    payload["reason"] = "terminated" if bool(payload["terminated"]) else "sigterm_sent_pid_still_running"
    return payload


def writer_state_snapshot(project_root: Path = PROJECT_ROOT, *, now_utc: datetime | None = None) -> dict[str, Any]:
    now = now_utc or datetime.now(timezone.utc)
    health_root = project_root / "governance" / "health"
    progress = _load_json(health_root / "sql_link_service_progress_latest.json")
    writer_lock = project_root / "governance" / "locks" / "jsonl_sql_writer.lock"
    lock_state = _lock_snapshot(writer_lock)
    owner = str(lock_state.get("owner") or "")
    lock_held = bool(lock_state.get("held", False))
    status = str(progress.get("status") or "").strip()
    running = bool(progress.get("running", False) or status == "running")
    progress_age_minutes = _age_minutes_from_timestamp(progress.get("timestamp_utc"), now_utc=now)
    cycle_age_minutes = _progress_cycle_age_minutes(progress, now_utc=now)
    recent_progress = progress_age_minutes is None or progress_age_minutes <= RECENT_PROGRESS_MINUTES
    active = bool(lock_held) or bool(running and recent_progress)
    shards = progress.get("shards") if isinstance(progress.get("shards"), list) else []
    timed_out_shard_count = sum(1 for row in shards if isinstance(row, dict) and bool(row.get("timed_out", False)))
    return {
        "timestamp_utc": now.isoformat(),
        "active": active,
        "running": running,
        "status": status,
        "current_step": str(progress.get("current_step") or ""),
        "cycle_started_utc": str(progress.get("cycle_started_utc") or ""),
        "progress_age_minutes": round(float(progress_age_minutes), 3) if progress_age_minutes is not None else None,
        "cycle_age_minutes": round(float(cycle_age_minutes), 3) if cycle_age_minutes is not None else None,
        "writer_lock_owner": owner,
        "writer_lock_held": lock_held,
        "completed_shard_count": _safe_int(progress.get("completed_shard_count"), 0),
        "completed_merge_count": _safe_int(progress.get("completed_merge_count"), 0),
        "merged_rows_this_cycle": _safe_int(progress.get("merged_rows_this_cycle"), 0),
        "planned_shard_count": len(list(progress.get("planned_shards") or [])) if isinstance(progress.get("planned_shards"), list) else len(shards),
        "timed_out_shard_count": int(timed_out_shard_count),
    }


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
        payload = _load_json(payload_path)
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
        return "timed_out"
    if int(result.get("rc", 1)) != 0:
        return "error"
    payload = result.get("payload") if isinstance(result.get("payload"), dict) else {}
    reason = str(payload.get("reason") or "")
    accepted = nonfatal_reasons or set()
    if bool(payload.get("busy", False)) or reason in accepted:
        return "busy"
    if bool(payload.get("skipped", False)):
        return "skipped"
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


def _refresh_surface_artifacts(project_root: Path) -> dict[str, Any]:
    refresh_steps: dict[str, Any] = {}
    for name, script_name in (
        ("ingestion_storage_control", "ingestion_storage_control.py"),
        ("runtime_gate_dashboard", "runtime_gate_dashboard.py"),
        ("operator_cockpit", "operator_cockpit.py"),
    ):
        refresh = _run_json_command(
            [str(PY), str(project_root / "scripts" / "ops" / script_name), "--json"],
            cwd=project_root,
            payload_path=project_root / "governance" / "health" / f"{name}_latest.json",
            timeout_sec=120,
        )
        refresh_steps[name] = _step_record(refresh)
    return refresh_steps


def _wait_for_writer_idle(
    project_root: Path,
    *,
    poll_seconds: float,
    wait_timeout_seconds: float,
) -> dict[str, Any]:
    started = time.monotonic()
    attempts = 0
    last_state = writer_state_snapshot(project_root)
    while True:
        attempts += 1
        last_state = writer_state_snapshot(project_root)
        if not bool(last_state.get("active", False)):
            return {
                "requested": True,
                "completed": True,
                "timed_out": False,
                "attempts": attempts,
                "waited_seconds": round(max(time.monotonic() - started, 0.0), 3),
                "final_state": last_state,
            }
        waited = max(time.monotonic() - started, 0.0)
        if waited >= max(float(wait_timeout_seconds), 0.0):
            return {
                "requested": True,
                "completed": False,
                "timed_out": True,
                "attempts": attempts,
                "waited_seconds": round(waited, 3),
                "final_state": last_state,
            }
        time.sleep(max(float(poll_seconds), 0.1))


def _writer_progress_summary(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    merged_rows_delta = max(
        _safe_int(after.get("merged_rows_this_cycle"), 0) - _safe_int(before.get("merged_rows_this_cycle"), 0),
        0,
    )
    merge_count_delta = max(
        _safe_int(after.get("completed_merge_count"), 0) - _safe_int(before.get("completed_merge_count"), 0),
        0,
    )
    shard_count_delta = max(
        _safe_int(after.get("completed_shard_count"), 0) - _safe_int(before.get("completed_shard_count"), 0),
        0,
    )
    progress_observed = bool(merged_rows_delta > 0 or merge_count_delta > 0 or shard_count_delta > 0)
    return {
        "progress_observed": progress_observed,
        "merged_rows_delta": int(merged_rows_delta),
        "merge_count_delta": int(merge_count_delta),
        "shard_count_delta": int(shard_count_delta),
    }


def build_payload(
    project_root: Path = PROJECT_ROOT,
    *,
    apply: bool,
    poll_seconds: float = DEFAULT_POLL_SECONDS,
    wait_timeout_seconds: float = DEFAULT_WAIT_TIMEOUT_SECONDS,
    command_timeout_seconds: int = DEFAULT_COMMAND_TIMEOUT_SECONDS,
    stale_progress_minutes: float = DEFAULT_STALE_PROGRESS_MINUTES,
    skip_drain: bool = False,
    skip_maintenance: bool = False,
    maintenance_force: bool = False,
    maintenance_vacuum: bool = False,
) -> dict[str, Any]:
    drain_preview = drain_src.build_payload(project_root, apply=False)
    maintenance_focus = maintenance_src._priority_retention_focus(project_root, {})
    drain_ready = bool(
        not skip_drain
        and bool(drain_preview.get("recommended_now", False))
        and not list(drain_preview.get("blocked_reasons") or [])
    )
    maintenance_ready = bool(not skip_maintenance and maintenance_focus.get("enabled", False))
    actionable = bool(drain_ready or maintenance_ready)
    writer_before = writer_state_snapshot(project_root)
    stale_writer_remediation = {
        "attempted": False,
        "needed": _stale_writer_detected(writer_before, stale_progress_minutes=stale_progress_minutes),
        "reason": "preview_only" if not apply else "writer_not_stale",
    }
    writer_after_remediation = writer_before
    if apply and bool(stale_writer_remediation.get("needed", False)):
        stale_writer_remediation = _terminate_stale_writer(
            project_root,
            writer_before,
            stale_progress_minutes=float(stale_progress_minutes),
        )
        writer_after_remediation = writer_state_snapshot(project_root)
    wait_result = {
        "requested": False,
        "completed": not bool(writer_after_remediation.get("active", False)),
        "timed_out": False,
        "attempts": 0,
        "waited_seconds": 0.0,
        "final_state": writer_after_remediation,
    }
    writer_after = writer_after_remediation
    if apply and bool(writer_after_remediation.get("active", False)) and not drain_ready:
        wait_result = _wait_for_writer_idle(
            project_root,
            poll_seconds=float(poll_seconds),
            wait_timeout_seconds=float(wait_timeout_seconds),
        )
        writer_after = wait_result.get("final_state") if isinstance(wait_result.get("final_state"), dict) else writer_after_remediation
    writer_progress = _writer_progress_summary(writer_after_remediation, writer_after)

    steps: dict[str, Any] = {}
    refresh_steps: dict[str, Any] = {}
    drain_payload: dict[str, Any] = {}
    maintenance_payload: dict[str, Any] = {}
    drain_applied = False
    maintenance_applied = False
    maintenance_followup_needed = False

    if apply and actionable:
        if drain_ready:
            drain = _run_json_command(
                [
                    str(PY),
                    str(project_root / "scripts" / "ops" / "external_backlog_drain.py"),
                    "--apply",
                    "--follow-through",
                    "--poll-seconds",
                    str(float(poll_seconds)),
                    "--wait-timeout-seconds",
                    str(float(wait_timeout_seconds)),
                    "--json",
                ],
                cwd=project_root,
                payload_path=project_root / "governance" / "health" / "external_backlog_drain_latest.json",
                timeout_sec=max(int(command_timeout_seconds), int(wait_timeout_seconds) + 120),
            )
            steps["external_backlog_drain"] = _step_record(drain)
            drain_payload = drain.get("payload") if isinstance(drain.get("payload"), dict) else {}
            drain_applied = int(drain.get("rc", 1)) == 0

        maintenance_can_run = not bool(writer_after.get("active", False))
        if drain_ready:
            follow_through = drain_payload.get("follow_through") if isinstance(drain_payload.get("follow_through"), dict) else {}
            maintenance_can_run = (
                maintenance_can_run
                and not bool(drain_payload.get("writer_busy", False))
                and str(follow_through.get("status") or "") not in {"timed_out", "handoff_requested"}
            )
            if not maintenance_can_run and maintenance_ready:
                maintenance_followup_needed = True
        if maintenance_ready and maintenance_can_run:
            maintenance_cmd = [str(PY), str(project_root / "scripts" / "ops" / "storage_maintenance_lane.py")]
            if maintenance_force:
                maintenance_cmd.append("--force")
            if maintenance_vacuum:
                maintenance_cmd.append("--vacuum")
            maintenance_cmd.append("--json")
            maintenance = _run_json_command(
                maintenance_cmd,
                cwd=project_root,
                payload_path=project_root / "governance" / "health" / "storage_maintenance_latest.json",
                timeout_sec=max(int(command_timeout_seconds), 900),
            )
            steps["storage_maintenance_lane"] = _step_record(maintenance, nonfatal_reasons={"already_running"})
            maintenance_payload = maintenance.get("payload") if isinstance(maintenance.get("payload"), dict) else {}
            maintenance_applied = int(maintenance.get("rc", 1)) == 0 and str(steps["storage_maintenance_lane"].get("status") or "") == "ok"
            if str(steps["storage_maintenance_lane"].get("status") or "") == "busy":
                maintenance_followup_needed = True

        if steps:
            refresh_steps = _refresh_surface_artifacts(project_root)

    step_statuses = [str((row or {}).get("status") or "") for row in steps.values() if isinstance(row, dict)]
    has_error = any(status == "error" or status == "timed_out" for status in step_statuses)
    drain_follow_through_status = str((((drain_payload.get("follow_through") or {}).get("status")) or ""))
    applied_with_followups = bool(
        apply
        and steps
        and not has_error
        and (
            (drain_ready and bool(drain_payload.get("writer_busy", False)))
            or (drain_ready and drain_follow_through_status == "timed_out")
            or (maintenance_ready and not maintenance_applied)
            or maintenance_followup_needed
        )
    )

    if not actionable:
        overall_status = "idle"
        ok = True
    elif bool(writer_before.get("active", False)) and not apply:
        overall_status = "waiting_for_writer"
        ok = True
    elif apply and steps and has_error:
        overall_status = "apply_failed"
        ok = False
    elif apply and steps and applied_with_followups:
        overall_status = "applied_with_followups"
        ok = False
    elif apply and steps:
        overall_status = "applied"
        ok = True
    elif bool(writer_after.get("active", False)) and bool(wait_result.get("timed_out", False)) and bool(writer_progress.get("progress_observed", False)):
        overall_status = "progressing_waiting_for_writer"
        ok = True
    elif bool(writer_after.get("active", False)) and bool(wait_result.get("timed_out", False)):
        overall_status = "timed_out_waiting_for_writer"
        ok = False
    elif bool(writer_after.get("active", False)):
        overall_status = "waiting_for_writer"
        ok = True
    elif not apply:
        overall_status = "ready"
        ok = True
    else:
        overall_status = "ready"
        ok = True

    recommended_actions = _ordered_unique(
        list(drain_preview.get("top_actions") or [])[:3]
        + list(maintenance_focus.get("top_actions") or [])[:3]
        + (
            ["stale SQL writer was restarted so the next drain handoff can be picked up cleanly"]
            if bool(stale_writer_remediation.get("terminated", False)) or bool(stale_writer_remediation.get("lock_released", False))
            else []
        )
        + (
            ["let the current SQL writer finish its active merge cycle before forcing drain or retention work; progress is still being made"]
            if bool(writer_progress.get("progress_observed", False))
            else []
        )
        + (
            ["wait for the current SQL writer cycle to finish before running heavy drain or retention work again"]
            if bool(writer_before.get("active", False))
            else []
        )
        + (
            ["give the writer cycle coordinator a quieter off-hours window if the handoff keeps timing out"]
            if bool(wait_result.get("timed_out", False))
            else []
        )
    )[:8]
    if not recommended_actions:
        recommended_actions.append("keep the coordinator idle until off-hours drain or priority retention work becomes actionable again")

    payload = {
        "timestamp_utc": _utc_now(),
        "schema_version": 1,
        "ok": ok,
        "overall_status": overall_status,
        "apply": bool(apply),
        "skip_drain": bool(skip_drain),
        "skip_maintenance": bool(skip_maintenance),
        "maintenance_force": bool(maintenance_force),
        "maintenance_vacuum": bool(maintenance_vacuum),
        "actionable": actionable,
        "drain_ready": drain_ready,
        "maintenance_ready": maintenance_ready,
        "writer_state_before": writer_before,
        "writer_state_after_remediation": writer_after_remediation,
        "stale_writer_remediation": stale_writer_remediation,
        "writer_state_after_wait": writer_after,
        "writer_progress": writer_progress,
        "wait_for_writer": wait_result,
        "drain_preview": {
            "overall_status": str(drain_preview.get("overall_status") or ""),
            "recommended_now": bool(drain_preview.get("recommended_now", False)),
            "writer_busy": bool(drain_preview.get("writer_busy", False)),
            "blocked_reasons": list(drain_preview.get("blocked_reasons") or []),
            "aged_candidate_files": _safe_int(drain_preview.get("aged_candidate_files"), 0),
            "off_hours_window": drain_preview.get("off_hours_window") if isinstance(drain_preview.get("off_hours_window"), dict) else {},
        },
        "maintenance_focus": maintenance_focus,
        "steps": steps,
        "refresh_steps": refresh_steps,
        "recommended_actions": recommended_actions,
        "summary": {
            "writer_active_initial": bool(writer_before.get("active", False)),
            "stale_writer_detected": bool(stale_writer_remediation.get("needed", False)),
            "stale_writer_restart_attempted": bool(stale_writer_remediation.get("attempted", False)),
            "stale_writer_terminated": bool(stale_writer_remediation.get("terminated", False)),
            "stale_writer_lock_released": bool(stale_writer_remediation.get("lock_released", False)),
            "writer_active_after_wait": bool(writer_after.get("active", False)),
            "writer_current_step": str(writer_after.get("current_step") or writer_before.get("current_step") or ""),
            "wait_timed_out": bool(wait_result.get("timed_out", False)),
            "wait_completed": bool(wait_result.get("completed", False)),
            "waited_seconds": round(float(wait_result.get("waited_seconds", 0.0) or 0.0), 3),
            "writer_progress_observed": bool(writer_progress.get("progress_observed", False)),
            "writer_merged_rows_delta": _safe_int(writer_progress.get("merged_rows_delta"), 0),
            "writer_merge_count_delta": _safe_int(writer_progress.get("merge_count_delta"), 0),
            "drain_applied": bool(drain_applied),
            "drain_follow_through_status": drain_follow_through_status,
            "maintenance_applied": bool(maintenance_applied),
            "priority_retention_focus_shards": list(maintenance_focus.get("focus_shards") or []),
            "priority_retention_targeted_debt_gb": round(_safe_float(maintenance_focus.get("targeted_retention_debt_gb"), 0.0), 3),
        },
    }
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Wait for the SQL writer handoff and then run drain or retention work without colliding with the active writer cycle.")
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--out-file", default=str(DEFAULT_OUT_PATH))
    parser.add_argument("--lock-file", default=str(DEFAULT_LOCK_PATH))
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--poll-seconds", type=float, default=DEFAULT_POLL_SECONDS)
    parser.add_argument("--wait-timeout-seconds", type=float, default=DEFAULT_WAIT_TIMEOUT_SECONDS)
    parser.add_argument("--command-timeout-seconds", type=int, default=DEFAULT_COMMAND_TIMEOUT_SECONDS)
    parser.add_argument("--stale-progress-minutes", type=float, default=float(os.getenv("WRITER_CYCLE_STALE_PROGRESS_MINUTES", str(DEFAULT_STALE_PROGRESS_MINUTES))))
    parser.add_argument("--skip-drain", action="store_true")
    parser.add_argument("--skip-maintenance", action="store_true")
    parser.add_argument("--maintenance-force", action="store_true")
    parser.add_argument("--maintenance-vacuum", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    out_file = Path(args.out_file).expanduser()
    lock_file = Path(args.lock_file).expanduser()
    lock_file.parent.mkdir(parents=True, exist_ok=True)

    payload: dict[str, Any] = {
        "timestamp_utc": _utc_now(),
        "schema_version": 1,
        "ok": True,
        "overall_status": "pending",
    }

    with lock_file.open("a+", encoding="utf-8") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            payload.update({"overall_status": "already_running", "busy": True})
            _write_json(out_file, payload)
            if args.json:
                print(json.dumps(payload, ensure_ascii=True))
            else:
                print("writer_cycle_coordinator overall_status=already_running")
            return 0

        payload = build_payload(
            project_root,
            apply=bool(args.apply),
            poll_seconds=float(args.poll_seconds),
            wait_timeout_seconds=float(args.wait_timeout_seconds),
            command_timeout_seconds=int(args.command_timeout_seconds),
            stale_progress_minutes=float(args.stale_progress_minutes),
            skip_drain=bool(args.skip_drain),
            skip_maintenance=bool(args.skip_maintenance),
            maintenance_force=bool(args.maintenance_force),
            maintenance_vacuum=bool(args.maintenance_vacuum),
        )
        _write_json(out_file, payload)
        if bool(args.apply):
            payload["post_write_refresh_steps"] = _refresh_surface_artifacts(project_root)
            _write_json(out_file, payload)

    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        print(
            "writer_cycle_coordinator "
            f"overall_status={payload.get('overall_status', '')} "
            f"actionable={int(bool(payload.get('actionable', False)))}"
        )
    return 0 if bool(payload.get("ok", False) or str(payload.get("overall_status") or "") in {"already_running", "waiting_for_writer", "progressing_waiting_for_writer", "idle"}) else 2


if __name__ == "__main__":
    raise SystemExit(main())
