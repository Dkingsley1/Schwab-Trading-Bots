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
from scripts.ops import backpressure_drainer_fleet as drainer_src
from scripts.ops import external_backlog_drain as drain_src
from scripts.ops import storage_maintenance_lane as maintenance_src
from scripts.ops import writer_process_intelligence as writer_intelligence_src


PY = resolve_runtime_python(PROJECT_ROOT)
DEFAULT_OUT_PATH = PROJECT_ROOT / "governance" / "health" / "writer_cycle_coordinator_latest.json"
DEFAULT_LOCK_PATH = PROJECT_ROOT / "governance" / "locks" / "writer_cycle_coordinator.lock"
DEFAULT_WAIT_TIMEOUT_SECONDS = 900.0
DEFAULT_POLL_SECONDS = 20.0
DEFAULT_COMMAND_TIMEOUT_SECONDS = 2400
WRITER_LOCK_PATH = PROJECT_ROOT / "governance" / "locks" / "jsonl_sql_writer.lock"
RECENT_PROGRESS_MINUTES = 30.0
DEFAULT_STALE_PROGRESS_MINUTES = 30.0
UNOWNED_PROGRESS_GRACE_MINUTES = 2.0


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


def _child_writer_processes(pid: int | None) -> list[dict[str, Any]]:
    if pid is None:
        return []
    try:
        proc = subprocess.run(
            ["pgrep", "-P", str(int(pid))],
            capture_output=True,
            text=True,
            check=False,
            timeout=2,
        )
    except Exception:
        return []
    children: list[dict[str, Any]] = []
    for raw in (proc.stdout or "").splitlines():
        try:
            child_pid = int(raw.strip())
        except Exception:
            continue
        command = _pid_command(child_pid)
        if "link_jsonl_to_sql.py" not in command and "sql_link_shard_manager.py" not in command:
            continue
        children.append({"pid": child_pid, "command": command[:260]})
    return children


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


def _completed_writer_lock_handoff_needed(state: dict[str, Any]) -> bool:
    current_step = str(state.get("current_step") or "").strip().lower()
    status = str(state.get("status") or "").strip().lower()
    planned = _safe_int(state.get("planned_shard_count"), 0)
    completed = _safe_int(state.get("completed_shard_count"), 0)
    complete_enough = planned <= 0 or completed >= planned
    return bool(
        state.get("writer_lock_held", False)
        and current_step == "complete"
        and status in {"ok", "complete", "idle"}
        and complete_enough
        and not bool(state.get("child_writer_active", False))
    )


def _release_completed_writer_lock(
    project_root: Path,
    state: dict[str, Any],
    *,
    grace_seconds: float = 3.0,
) -> dict[str, Any]:
    owner = str(state.get("writer_lock_owner") or "")
    pid = _parse_lock_owner_pid(owner)
    payload: dict[str, Any] = {
        "attempted": False,
        "needed": _completed_writer_lock_handoff_needed(state),
        "pid": pid,
        "owner": owner,
        "terminated": False,
        "lock_released": False,
        "reason": "",
    }
    if not bool(payload["needed"]):
        payload["reason"] = "writer_not_complete_or_lock_not_held"
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
    payload["reason"] = "completed_writer_handoff_released" if bool(payload["lock_released"]) else "sigterm_sent_pid_still_holding_lock"
    return payload


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
    owner_pid = _parse_lock_owner_pid(owner)
    owner_pid_live = bool(owner_pid is not None and _pid_exists(owner_pid))
    active_children = _child_writer_processes(owner_pid)
    child_writer_active = bool(active_children)
    owner_without_pid = bool(owner and owner_pid is None)
    unowned_running_progress = bool(running and recent_progress and not lock_held and not owner_pid_live and not child_writer_active)
    unowned_progress_grace = progress_age_minutes is None or progress_age_minutes <= UNOWNED_PROGRESS_GRACE_MINUTES
    orphaned_progress = bool(unowned_running_progress and not unowned_progress_grace)
    current_step = str(progress.get("current_step") or "")
    complete_lock_handoff_needed = bool(
        lock_held
        and current_step == "complete"
        and status in {"ok", "complete", "idle"}
        and not child_writer_active
        and (
            _safe_int(progress.get("planned_shard_count"), 0) <= 0
            or _safe_int(progress.get("completed_shard_count"), 0) >= _safe_int(progress.get("planned_shard_count"), 0)
        )
    )
    active = bool(lock_held) or bool(
        (running or child_writer_active)
        and recent_progress
        and (
            lock_held
            or owner_pid_live
            or child_writer_active
            or bool(owner_without_pid and unowned_progress_grace)
            or unowned_progress_grace
        )
    )
    if orphaned_progress:
        active = False
    if lock_held:
        active_source = "completed_lock_handoff_needed" if complete_lock_handoff_needed else "writer_lock"
    elif orphaned_progress:
        active_source = "orphaned_progress"
    elif active:
        active_source = "recent_progress"
    else:
        active_source = "idle"
    shards = progress.get("shards") if isinstance(progress.get("shards"), list) else []
    timed_out_shard_count = sum(1 for row in shards if isinstance(row, dict) and bool(row.get("timed_out", False)))
    return {
        "timestamp_utc": now.isoformat(),
        "active": active,
        "active_source": active_source,
        "progress_orphaned": orphaned_progress,
        "complete_lock_handoff_needed": complete_lock_handoff_needed,
        "unowned_progress_grace_minutes": float(UNOWNED_PROGRESS_GRACE_MINUTES),
        "writer_owner_pid_live": owner_pid_live,
        "running": bool(running or child_writer_active),
        "status": status,
        "current_step": str(progress.get("current_step") or ""),
        "effective_current_step": "shard_worker_active_after_reported_complete"
        if child_writer_active and str(progress.get("current_step") or "") == "complete"
        else str(progress.get("current_step") or ""),
        "child_writer_active": child_writer_active,
        "active_child_writer_count": len(active_children),
        "active_child_writer_pids": [row["pid"] for row in active_children],
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


def _already_running_payload(project_root: Path, *, previous_path: Path) -> dict[str, Any]:
    writer_state = writer_state_snapshot(project_root)
    previous = _load_json(previous_path)
    previous_summary = previous.get("summary") if isinstance(previous.get("summary"), dict) else {}
    payload: dict[str, Any] = {
        "timestamp_utc": _utc_now(),
        "schema_version": 1,
        "ok": True,
        "overall_status": "already_running",
        "busy": True,
        "actionable": False,
        "drain_ready": None,
        "live_drainer_ready": None,
        "maintenance_ready": None,
        "writer_state_before": writer_state,
        "writer_state_after_wait": writer_state,
        "previous_coordinator": {
            "timestamp_utc": str(previous.get("timestamp_utc") or ""),
            "overall_status": str(previous.get("overall_status") or ""),
            "summary": previous_summary,
        },
        "recommended_actions": [
            "another writer-cycle coordinator is already running; use this cached writer snapshot instead of starting a second coordinator",
            "wait for the active coordinator or SQL writer to finish before launching drain or maintenance work",
        ],
        "summary": {
            "writer_active_initial": bool(writer_state.get("active", False)),
            "writer_active_after_wait": bool(writer_state.get("active", False)),
            "writer_current_step": str(writer_state.get("effective_current_step") or writer_state.get("current_step") or ""),
            "writer_progress_observed": False,
            "writer_merged_rows_delta": 0,
            "writer_merge_count_delta": 0,
            "wait_completed": not bool(writer_state.get("active", False)),
            "wait_timed_out": False,
            "waited_seconds": 0.0,
            "drain_applied": False,
            "maintenance_applied": False,
            "active_drainer": str(previous_summary.get("active_drainer") or ""),
        },
    }
    writer_intelligence = writer_intelligence_src.build_payload(project_root, writer_cycle=payload)
    payload["writer_process_intelligence"] = writer_intelligence
    writer_decision = writer_intelligence.get("decision_packet") if isinstance(writer_intelligence.get("decision_packet"), dict) else {}
    payload["summary"]["writer_process_action"] = str(writer_decision.get("action") or "")
    payload["summary"]["expanded_writer_lane_count"] = _safe_int(writer_decision.get("expanded_writer_lane_count"), 0)
    return payload


def _live_safe_drainer_ready(preview: dict[str, Any]) -> bool:
    active = preview.get("active_drainer") if isinstance(preview.get("active_drainer"), dict) else {}
    return bool(
        str(preview.get("overall_status") or "") in {"ready", "handoff_requested"}
        and str(active.get("status") or "") == "ready"
        and bool(active.get("live_window_safe", False))
        and not list(preview.get("blocked_reasons") or [])
    )


def _drainer_active_name(preview: dict[str, Any]) -> str:
    active = preview.get("active_drainer") if isinstance(preview.get("active_drainer"), dict) else {}
    return str(active.get("name") or "")


def _drainer_service_request_ok(payload: dict[str, Any]) -> bool:
    request = payload.get("service_request") if isinstance(payload.get("service_request"), dict) else {}
    overrides = request.get("env_overrides") if isinstance(request.get("env_overrides"), dict) else {}
    return bool(
        str(payload.get("overall_status") or "") in {"handoff_requested", "ready"}
        and bool(request.get("active", False))
        and str(overrides.get("SQL_LINK_SERVICE_SHARDS") or "").strip()
    )


def _sql_link_manager_timeout_seconds(
    *,
    command_timeout_seconds: int,
    wait_timeout_seconds: float,
    drainer_payload: dict[str, Any],
    timeout_cap_seconds: int = 0,
) -> int:
    request = drainer_payload.get("service_request") if isinstance(drainer_payload.get("service_request"), dict) else {}
    overrides = request.get("env_overrides") if isinstance(request.get("env_overrides"), dict) else {}
    shard_names = _ordered_unique(str(overrides.get("SQL_LINK_SERVICE_SHARDS") or "").split(","))
    shard_count = max(len(shard_names), 1)
    shard_timeout = _safe_int(overrides.get("SQL_LINK_SERVICE_SHARD_LINK_TIMEOUT_SECONDS"), 0)
    merge_budget = _safe_float(overrides.get("SQL_LINK_SERVICE_MERGE_MAX_SECONDS_PER_CYCLE"), 0.0)
    request_budget = 0
    if shard_timeout > 0:
        request_budget = int((shard_timeout * shard_count) + max(float(merge_budget), 0.0) + 120)
    timeout_seconds = max(int(command_timeout_seconds), int(wait_timeout_seconds) + 120, request_budget, 1)
    cap = int(timeout_cap_seconds)
    if cap > 0:
        timeout_seconds = min(timeout_seconds, max(cap, 1))
    return timeout_seconds


def _service_request_env(drainer_payload: dict[str, Any]) -> dict[str, str]:
    request = drainer_payload.get("service_request") if isinstance(drainer_payload.get("service_request"), dict) else {}
    overrides = request.get("env_overrides") if isinstance(request.get("env_overrides"), dict) else {}
    return {str(key): str(value) for key, value in overrides.items() if str(key).strip()}


def _catch_up_wave_limit(drainer_payload: dict[str, Any]) -> int:
    overrides = _service_request_env(drainer_payload)
    raw = overrides.get("WRITER_CYCLE_MAX_CATCH_UP_WAVES") or os.getenv("WRITER_CYCLE_MAX_CATCH_UP_WAVES")
    return max(1, min(_safe_int(raw, 1), 5))


def _storage_catch_up_wave_limit(project_root: Path) -> int:
    storage = _load_json(project_root / "governance" / "health" / "ingestion_storage_control_latest.json")
    contract = storage.get("backlog_relief_contract") if isinstance(storage.get("backlog_relief_contract"), dict) else {}
    p_core = contract.get("p_core_backlog_allocation_contract") if isinstance(contract.get("p_core_backlog_allocation_contract"), dict) else {}
    wave = p_core.get("catch_up_wave_controller") if isinstance(p_core.get("catch_up_wave_controller"), dict) else {}
    if not bool(contract.get("active", False)) and not bool(wave.get("enabled", False)):
        return 1
    return max(1, min(_safe_int(wave.get("max_waves"), 1), 5))


def _storage_followup_issues(project_root: Path) -> set[str]:
    storage = _load_json(project_root / "governance" / "health" / "ingestion_storage_control_latest.json")
    contract = storage.get("backlog_relief_contract") if isinstance(storage.get("backlog_relief_contract"), dict) else {}
    return {str(item) for item in (contract.get("active_issue_ids") or []) if str(item)}


def _should_run_next_catch_up_wave(project_root: Path, payload: dict[str, Any], *, wave_index: int, wave_limit: int) -> bool:
    if int(wave_index) >= int(wave_limit):
        return False
    merge_followup = payload.get("merge_followup") if isinstance(payload.get("merge_followup"), dict) else {}
    if bool(merge_followup.get("followup_needed", False)):
        return True
    if _safe_int(payload.get("merged_rows_this_cycle"), 0) <= 0:
        return False
    active_issues = _storage_followup_issues(project_root)
    return bool(active_issues & {"single_writer_merge_speed", "stale_old_pending_work", "sparse_huge_jsonl_files"})


def _writer_wave_record(wave_index: int, result: dict[str, Any], payload: dict[str, Any], step: dict[str, Any]) -> dict[str, Any]:
    merge_followup = payload.get("merge_followup") if isinstance(payload.get("merge_followup"), dict) else {}
    return {
        "wave": int(wave_index),
        "status": str(step.get("status") or ""),
        "rc": _safe_int(result.get("rc"), 1),
        "merged_rows_this_cycle": _safe_int(payload.get("merged_rows_this_cycle"), 0),
        "followup_needed": bool(merge_followup.get("followup_needed", False)),
        "followup_reasons": list(merge_followup.get("followup_reasons") or []),
    }


def _drain_effectiveness_score(before: dict[str, Any], after: dict[str, Any], *, merged_rows: int, waves_run: int) -> dict[str, Any]:
    before_bp = before.get("backpressure") if isinstance(before.get("backpressure"), dict) else {}
    after_bp = after.get("backpressure") if isinstance(after.get("backpressure"), dict) else {}
    before_overlay = before.get("sql_ingestion_pending_overlay") if isinstance(before.get("sql_ingestion_pending_overlay"), dict) else {}
    after_overlay = after.get("sql_ingestion_pending_overlay") if isinstance(after.get("sql_ingestion_pending_overlay"), dict) else {}
    pending_delta = _safe_int(before_bp.get("total_pending_lines"), 0) - _safe_int(after_bp.get("total_pending_lines"), 0)
    core_delta = _safe_int(before_bp.get("core_pending_lines"), 0) - _safe_int(after_bp.get("core_pending_lines"), 0)
    age_delta = _safe_float(before_bp.get("oldest_pending_age_seconds"), 0.0) - _safe_float(after_bp.get("oldest_pending_age_seconds"), 0.0)
    sparse_before = _safe_int(((before_bp.get("raw_live") or {}).get("line_estimation") or {}).get("sparse_large_line_pending_bytes"), 0)
    sparse_after = _safe_int(((after_bp.get("raw_live") or {}).get("line_estimation") or {}).get("sparse_large_line_pending_bytes"), 0)
    overlay_delta = _safe_int(before_overlay.get("total_pending_lines"), 0) - _safe_int(after_overlay.get("total_pending_lines"), 0)
    after_total_pending = _safe_int(after_bp.get("total_pending_lines"), 0)
    after_core_pending = _safe_int(after_bp.get("core_pending_lines"), 0)
    after_oldest_age = _safe_float(after_bp.get("oldest_pending_age_seconds"), 0.0)
    pending_threshold = max(_safe_int(after_bp.get("pending_lines_threshold"), 15000), 1)
    oldest_age_threshold = max(_safe_float(after_bp.get("oldest_age_threshold_seconds"), 240.0), 1.0)
    pressure_index = _safe_float(after.get("pressure_index"), 0.0)
    score = 0.0
    if int(merged_rows) > 0:
        score += 30.0
    if pending_delta > 0:
        score += 25.0
    if core_delta > 0:
        score += 20.0
    if age_delta > 0:
        score += 15.0
    if sparse_before > sparse_after:
        score += 5.0
    if int(waves_run) > 1:
        score += 5.0
    calm_backpressure = bool(
        after_total_pending <= max(1000, int(pending_threshold * 0.10))
        and after_core_pending <= max(500, int(pending_threshold * 0.05))
        and after_oldest_age <= oldest_age_threshold
        and pressure_index < 0.35
    )
    no_progress_false_alarm = bool(int(merged_rows) <= 0 and score <= 0.0 and calm_backpressure)
    if no_progress_false_alarm:
        status = "settled_no_action_needed"
        score = 70.0
        next_action = "do not escalate the writer; backlog is already inside target and no stale pending work is present"
    else:
        status = "strong_progress" if score >= 75.0 else "progress" if score >= 45.0 else "weak_progress" if int(merged_rows) > 0 else "no_progress"
        next_action = (
            "keep chaining bounded catch-up waves while deltas stay positive"
            if score >= 45.0
            else "switch to stale-source locator or collector intake audit because writer cycles are not reducing pressure"
        )
    return {
        "status": status,
        "score": round(min(score, 100.0), 2),
        "merged_rows": int(merged_rows),
        "waves_run": int(waves_run),
        "total_pending_delta": int(pending_delta),
        "core_pending_delta": int(core_delta),
        "oldest_age_delta_seconds": round(float(age_delta), 3),
        "overlay_pending_delta": int(overlay_delta),
        "sparse_pending_bytes_delta": int(max(sparse_before - sparse_after, 0)),
        "pending_after": int(after_total_pending),
        "core_pending_after": int(after_core_pending),
        "oldest_pending_age_after_seconds": round(float(after_oldest_age), 3),
        "false_alarm_guard": {
            "active": True,
            "suppressed_no_progress_alarm": no_progress_false_alarm,
            "calm_backpressure": calm_backpressure,
            "pending_threshold": int(pending_threshold),
            "oldest_age_threshold_seconds": round(float(oldest_age_threshold), 3),
            "pressure_index": round(float(pressure_index), 6),
            "policy": "only call no_progress actionable when backlog debt, oldest age, or pressure remains outside target",
        },
        "next_action": next_action,
    }


def build_payload(
    project_root: Path = PROJECT_ROOT,
    *,
    apply: bool,
    poll_seconds: float = DEFAULT_POLL_SECONDS,
    wait_timeout_seconds: float = DEFAULT_WAIT_TIMEOUT_SECONDS,
    command_timeout_seconds: int = DEFAULT_COMMAND_TIMEOUT_SECONDS,
    sql_manager_timeout_cap_seconds: int = 0,
    stale_progress_minutes: float = DEFAULT_STALE_PROGRESS_MINUTES,
    skip_drain: bool = False,
    skip_maintenance: bool = False,
    maintenance_force: bool = False,
    maintenance_vacuum: bool = False,
) -> dict[str, Any]:
    storage_before = _load_json(project_root / "governance" / "health" / "ingestion_storage_control_latest.json")
    drain_preview = drain_src.build_payload(project_root, apply=False)
    drainer_preview = drainer_src.build_payload(project_root, apply=False)
    maintenance_focus = maintenance_src._priority_retention_focus(project_root, {})
    live_drainer_ready = bool(not skip_drain and _live_safe_drainer_ready(drainer_preview))
    drain_ready = bool(
        not live_drainer_ready
        and not skip_drain
        and bool(drain_preview.get("recommended_now", False))
        and not list(drain_preview.get("blocked_reasons") or [])
    )
    maintenance_ready = bool(not skip_maintenance and maintenance_focus.get("enabled", False))
    actionable = bool(live_drainer_ready or drain_ready or maintenance_ready)
    writer_before = writer_state_snapshot(project_root)
    stale_writer_remediation = {
        "attempted": False,
        "needed": _stale_writer_detected(writer_before, stale_progress_minutes=stale_progress_minutes),
        "reason": "preview_only" if not apply else "writer_not_stale",
    }
    completed_lock_handoff = {
        "attempted": False,
        "needed": _completed_writer_lock_handoff_needed(writer_before),
        "reason": "preview_only" if not apply else "writer_not_complete_or_lock_not_held",
    }
    writer_after_remediation = writer_before
    if apply and bool(stale_writer_remediation.get("needed", False)):
        stale_writer_remediation = _terminate_stale_writer(
            project_root,
            writer_before,
            stale_progress_minutes=float(stale_progress_minutes),
        )
        writer_after_remediation = writer_state_snapshot(project_root)
        completed_lock_handoff = {
            "attempted": False,
            "needed": _completed_writer_lock_handoff_needed(writer_after_remediation),
            "reason": "writer_not_complete_or_lock_not_held",
        }
    if apply and bool(completed_lock_handoff.get("needed", False)):
        completed_lock_handoff = _release_completed_writer_lock(project_root, writer_after_remediation)
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
    catch_up_followup_needed = False
    catch_up_wave_limit = _storage_catch_up_wave_limit(project_root)
    catch_up_wave_records: list[dict[str, Any]] = []

    if apply and actionable:
        if live_drainer_ready:
            drainer_apply = _run_json_command(
                [
                    str(PY),
                    str(project_root / "scripts" / "ops" / "backpressure_drainer_fleet.py"),
                    "--apply",
                    "--ttl-seconds",
                    str(max(int(wait_timeout_seconds), 900)),
                    "--json",
                ],
                cwd=project_root,
                payload_path=project_root / "governance" / "health" / "backpressure_drainer_fleet_latest.json",
                timeout_sec=120,
            )
            steps["backpressure_drainer_fleet"] = _step_record(drainer_apply)
            drainer_payload = drainer_apply.get("payload") if isinstance(drainer_apply.get("payload"), dict) else {}
            drainer_handoff_ok = int(drainer_apply.get("rc", 1)) == 0 and _drainer_service_request_ok(drainer_payload)
            if drainer_handoff_ok and not bool(writer_after.get("active", False)):
                catch_up_wave_limit = max(_catch_up_wave_limit(drainer_payload), _storage_catch_up_wave_limit(project_root))

                def _run_shard_manager_wave(wave_index: int) -> dict[str, Any]:
                    return _run_json_command(
                        [
                            str(PY),
                            str(project_root / "scripts" / "ops" / "sql_link_shard_manager.py"),
                            "--once",
                            "--json",
                        ],
                        cwd=project_root,
                        payload_path=project_root / "governance" / "health" / "sql_link_service_latest.json",
                        timeout_sec=_sql_link_manager_timeout_seconds(
                            command_timeout_seconds=int(command_timeout_seconds),
                            wait_timeout_seconds=float(wait_timeout_seconds),
                            drainer_payload=drainer_payload,
                            timeout_cap_seconds=int(sql_manager_timeout_cap_seconds),
                        ),
                    )

                shard_manager = _run_shard_manager_wave(1)
                steps["sql_link_shard_manager"] = _step_record(shard_manager, nonfatal_reasons={"writer_lock_busy"})
                drain_payload = shard_manager.get("payload") if isinstance(shard_manager.get("payload"), dict) else {}
                if str(steps["sql_link_shard_manager"].get("status") or "") == "error" and _safe_int(drain_payload.get("merged_rows_this_cycle"), 0) > 0:
                    steps["sql_link_shard_manager"]["status"] = "partial_progress"
                drain_applied = str(steps["sql_link_shard_manager"].get("status") or "") in {"ok", "partial_progress"}
                merge_followup = drain_payload.get("merge_followup") if isinstance(drain_payload.get("merge_followup"), dict) else {}
                catch_up_followup_needed = bool(merge_followup.get("followup_needed", False))
                catch_up_wave_records.append(_writer_wave_record(1, shard_manager, drain_payload, steps["sql_link_shard_manager"]))
                for wave_index in range(2, int(catch_up_wave_limit) + 1):
                    if not _should_run_next_catch_up_wave(project_root, drain_payload, wave_index=wave_index - 1, wave_limit=catch_up_wave_limit):
                        break
                    if bool(writer_state_snapshot(project_root).get("active", False)):
                        break
                    wave_result = _run_shard_manager_wave(wave_index)
                    step_key = f"sql_link_shard_manager_wave_{wave_index}"
                    steps[step_key] = _step_record(wave_result, nonfatal_reasons={"writer_lock_busy"})
                    wave_payload = wave_result.get("payload") if isinstance(wave_result.get("payload"), dict) else {}
                    if str(steps[step_key].get("status") or "") == "error" and _safe_int(wave_payload.get("merged_rows_this_cycle"), 0) > 0:
                        steps[step_key]["status"] = "partial_progress"
                    catch_up_wave_records.append(_writer_wave_record(wave_index, wave_result, wave_payload, steps[step_key]))
                    if str(steps[step_key].get("status") or "") not in {"ok", "partial_progress"}:
                        catch_up_followup_needed = True
                        break
                    drain_payload = wave_payload
                    drain_applied = True
                    merge_followup = drain_payload.get("merge_followup") if isinstance(drain_payload.get("merge_followup"), dict) else {}
                    catch_up_followup_needed = bool(merge_followup.get("followup_needed", False))
                catch_up_followup_needed = _should_run_next_catch_up_wave(
                    project_root,
                    drain_payload,
                    wave_index=len(catch_up_wave_records),
                    wave_limit=catch_up_wave_limit,
                )
                writer_after = writer_state_snapshot(project_root)
            else:
                drain_payload = drainer_payload
                drain_applied = drainer_handoff_ok
        elif drain_ready:
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
        if live_drainer_ready:
            maintenance_can_run = maintenance_can_run and drain_applied
            if not maintenance_can_run and maintenance_ready:
                maintenance_followup_needed = True
        elif drain_ready:
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
    partial_progress = any(status == "partial_progress" for status in step_statuses)
    drain_follow_through_status = str((((drain_payload.get("follow_through") or {}).get("status")) or ""))
    applied_with_followups = bool(
        apply
        and steps
        and not has_error
        and (
            partial_progress
            or (live_drainer_ready and not drain_applied)
            or (drain_ready and bool(drain_payload.get("writer_busy", False)))
            or (drain_ready and drain_follow_through_status == "timed_out")
            or (maintenance_ready and not maintenance_applied)
            or maintenance_followup_needed
            or catch_up_followup_needed
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
        + list(drainer_preview.get("recommended_actions") or [])[:3]
        + list(maintenance_focus.get("top_actions") or [])[:3]
        + ([f"run live-safe drainer handoff now: {_drainer_active_name(drainer_preview)}"] if live_drainer_ready else [])
        + (
            ["stale SQL writer was restarted so the next drain handoff can be picked up cleanly"]
            if bool(stale_writer_remediation.get("terminated", False)) or bool(stale_writer_remediation.get("lock_released", False))
            else []
        )
        + (
            ["completed SQL writer lock handoff was reconciled so the next writer/drainer cycle can start"]
            if bool(completed_lock_handoff.get("terminated", False)) or bool(completed_lock_handoff.get("lock_released", False))
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
        + (
            ["run another focused catch-up wave; the last writer cycle reported merge caps, budget exhaustion, or partial timeout shards"]
            if catch_up_followup_needed
            else []
        )
    )[:8]
    if not recommended_actions:
        recommended_actions.append("keep the coordinator idle until off-hours drain or priority retention work becomes actionable again")

    storage_after = _load_json(project_root / "governance" / "health" / "ingestion_storage_control_latest.json")
    total_merged_rows = sum(_safe_int(row.get("merged_rows_this_cycle"), 0) for row in catch_up_wave_records)
    if total_merged_rows <= 0:
        total_merged_rows = _safe_int(drain_payload.get("merged_rows_this_cycle"), 0)
    drain_effectiveness = _drain_effectiveness_score(
        storage_before,
        storage_after,
        merged_rows=total_merged_rows,
        waves_run=len(catch_up_wave_records),
    )
    follow_through_contract = {
        "active": bool(live_drainer_ready or drain_ready),
        "writer_idle": not bool(writer_after.get("active", False)),
        "bounded_wave_limit": int(catch_up_wave_limit),
        "waves_run": len(catch_up_wave_records),
        "followup_remaining": bool(catch_up_followup_needed),
        "policy": "auto_chain_bounded_writer_waves_when_storage_contract_still_has_merge_or_stale_debt",
    }

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
        "settings": {
            "poll_seconds": float(poll_seconds),
            "wait_timeout_seconds": float(wait_timeout_seconds),
            "command_timeout_seconds": int(command_timeout_seconds),
            "sql_manager_timeout_cap_seconds": int(sql_manager_timeout_cap_seconds),
            "stale_progress_minutes": float(stale_progress_minutes),
        },
        "actionable": actionable,
        "drain_ready": drain_ready,
        "live_drainer_ready": live_drainer_ready,
        "maintenance_ready": maintenance_ready,
        "writer_state_before": writer_before,
        "writer_state_after_remediation": writer_after_remediation,
        "stale_writer_remediation": stale_writer_remediation,
        "completed_writer_lock_handoff": completed_lock_handoff,
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
        "drainer_preview": {
            "overall_status": str(drainer_preview.get("overall_status") or ""),
            "ready_drainer_count": _safe_int(drainer_preview.get("ready_drainer_count"), 0),
            "active_drainer": _drainer_active_name(drainer_preview),
            "live_window_safe": bool((drainer_preview.get("active_drainer") if isinstance(drainer_preview.get("active_drainer"), dict) else {}).get("live_window_safe", False)),
            "blocked_reasons": list(drainer_preview.get("blocked_reasons") or []),
        },
        "maintenance_focus": maintenance_focus,
        "steps": steps,
        "refresh_steps": refresh_steps,
        "catch_up_wave": (
            drain_payload.get("merge_followup")
            if isinstance(drain_payload.get("merge_followup"), dict)
            else {"followup_needed": bool(catch_up_followup_needed)}
        ),
        "catch_up_wave_controller": {
            "enabled": int(catch_up_wave_limit) > 1,
            "wave_limit": int(catch_up_wave_limit),
            "waves_run": len(catch_up_wave_records),
            "followup_remaining": bool(catch_up_followup_needed),
            "policy": "bounded_sequential_single_writer",
            "records": catch_up_wave_records,
        },
        "writer_follow_through_contract": follow_through_contract,
        "drain_effectiveness": drain_effectiveness,
        "recommended_actions": recommended_actions,
        "summary": {
            "writer_active_initial": bool(writer_before.get("active", False)),
            "stale_writer_detected": bool(stale_writer_remediation.get("needed", False)),
            "stale_writer_restart_attempted": bool(stale_writer_remediation.get("attempted", False)),
            "stale_writer_terminated": bool(stale_writer_remediation.get("terminated", False)),
            "stale_writer_lock_released": bool(stale_writer_remediation.get("lock_released", False)),
            "completed_writer_lock_handoff_needed": bool(completed_lock_handoff.get("needed", False)),
            "completed_writer_lock_handoff_attempted": bool(completed_lock_handoff.get("attempted", False)),
            "completed_writer_lock_handoff_released": bool(completed_lock_handoff.get("lock_released", False)),
            "writer_active_after_wait": bool(writer_after.get("active", False)),
            "writer_current_step": str(
                writer_after.get("effective_current_step")
                or writer_after.get("current_step")
                or writer_before.get("effective_current_step")
                or writer_before.get("current_step")
                or ""
            ),
            "wait_timed_out": bool(wait_result.get("timed_out", False)),
            "wait_completed": bool(wait_result.get("completed", False)),
            "waited_seconds": round(float(wait_result.get("waited_seconds", 0.0) or 0.0), 3),
            "writer_progress_observed": bool(writer_progress.get("progress_observed", False)),
            "writer_merged_rows_delta": _safe_int(writer_progress.get("merged_rows_delta"), 0),
            "writer_merge_count_delta": _safe_int(writer_progress.get("merge_count_delta"), 0),
            "drain_applied": bool(drain_applied),
            "partial_progress": bool(partial_progress),
            "live_drainer_applied": bool(live_drainer_ready and drain_applied),
            "active_drainer": _drainer_active_name(drainer_preview),
            "drain_follow_through_status": drain_follow_through_status,
            "maintenance_applied": bool(maintenance_applied),
            "catch_up_followup_needed": bool(catch_up_followup_needed),
            "catch_up_waves_run": len(catch_up_wave_records),
            "priority_retention_focus_shards": list(maintenance_focus.get("focus_shards") or []),
            "priority_retention_targeted_debt_gb": round(_safe_float(maintenance_focus.get("targeted_retention_debt_gb"), 0.0), 3),
        },
    }
    writer_intelligence = writer_intelligence_src.build_payload(project_root, writer_cycle=payload)
    payload["writer_process_intelligence"] = writer_intelligence
    writer_decision = writer_intelligence.get("decision_packet") if isinstance(writer_intelligence.get("decision_packet"), dict) else {}
    payload["summary"]["writer_process_action"] = str(writer_decision.get("action") or "")
    payload["summary"]["expanded_writer_lane_count"] = _safe_int(writer_decision.get("expanded_writer_lane_count"), 0)
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
    parser.add_argument("--sql-manager-timeout-cap-seconds", type=int, default=int(os.getenv("WRITER_CYCLE_SQL_MANAGER_TIMEOUT_CAP_SECONDS", "0")))
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
            payload = _already_running_payload(project_root, previous_path=out_file)
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
            sql_manager_timeout_cap_seconds=int(args.sql_manager_timeout_cap_seconds),
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
