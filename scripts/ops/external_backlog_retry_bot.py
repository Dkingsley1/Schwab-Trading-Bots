#!/usr/bin/env python3
from __future__ import annotations

import argparse
import fcntl
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.runtime_python import resolve_runtime_python
from scripts.ops import backlog_quarantine_bot as quarantine_src
from scripts.ops import external_backlog_drain as drain_src


PY = resolve_runtime_python(PROJECT_ROOT)
DEFAULT_OUT_PATH = PROJECT_ROOT / "governance" / "health" / "external_backlog_retry_bot_latest.json"
DEFAULT_LOCK_PATH = PROJECT_ROOT / "governance" / "locks" / "external_backlog_retry_bot.lock"
DEFAULT_WAIT_TIMEOUT_SECONDS = 900.0
DEFAULT_POLL_SECONDS = 20.0
DEFAULT_COMMAND_TIMEOUT_SECONDS = 1800


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


def _safe_int(raw: Any, default: int = 0) -> int:
    try:
        return int(float(raw))
    except Exception:
        return int(default)


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


def _step_status(result: dict[str, Any]) -> str:
    if bool(result.get("timed_out", False)):
        return "timed_out"
    if int(result.get("rc", 1)) != 0:
        return "error"
    payload = result.get("payload") if isinstance(result.get("payload"), dict) else {}
    if payload.get("ok") is False:
        return "error"
    return "ok"


def _step_record(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": _step_status(result),
        "rc": int(result.get("rc", 1)),
        "duration_ms": float(result.get("duration_ms", 0.0) or 0.0),
        "timed_out": bool(result.get("timed_out", False)),
        "cmd": list(result.get("cmd") or []),
        "stdout_tail": str(result.get("stdout_tail") or ""),
        "stderr_tail": str(result.get("stderr_tail") or ""),
    }


def _needs_retry(preview: dict[str, Any]) -> bool:
    backpressure = preview.get("backpressure_before") if isinstance(preview.get("backpressure_before"), dict) else {}
    return bool(
        _safe_int(backpressure.get("deferred_pending_lines"), 0) > 0
        or _safe_int(backpressure.get("cold_pending_lines"), 0) > 0
        or _safe_int(preview.get("aged_candidate_files"), 0) > 0
    )


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


def build_payload(
    project_root: Path = PROJECT_ROOT,
    *,
    apply: bool,
    poll_seconds: float = DEFAULT_POLL_SECONDS,
    wait_timeout_seconds: float = DEFAULT_WAIT_TIMEOUT_SECONDS,
    command_timeout_seconds: int = DEFAULT_COMMAND_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    preview = drain_src.build_payload(project_root, apply=False)
    quarantine_preview = quarantine_src.build_payload(project_root, apply=False)
    preview_status = str(preview.get("overall_status") or "")
    blocked_reasons = list(preview.get("blocked_reasons") or [])
    off_hours = preview.get("off_hours_window") if isinstance(preview.get("off_hours_window"), dict) else {}
    quarantine_candidate_files = _safe_int(quarantine_preview.get("candidate_files"), 0)
    backlog_needed = bool(_needs_retry(preview) or quarantine_candidate_files > 0)
    market_hours_only_block = bool(blocked_reasons and set(blocked_reasons).issubset({"market_hours_guard"}))
    actionable = bool(backlog_needed and not blocked_reasons and bool(off_hours.get("active", False)))
    quarantine_actionable = bool(quarantine_candidate_files > 0 and market_hours_only_block)

    steps: dict[str, Any] = {}
    refresh_steps: dict[str, Any] = {}
    drain_payload: dict[str, Any] = {}
    quarantine_payload: dict[str, Any] = {}
    drain_follow_through_status = ""
    drain_follow_through_progress_state = ""
    applied = False
    quarantine_applied = False

    if apply and actionable:
        result = _run_json_command(
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
        steps["external_backlog_drain"] = _step_record(result)
        drain_payload = result.get("payload") if isinstance(result.get("payload"), dict) else {}
        follow_through = drain_payload.get("follow_through") if isinstance(drain_payload.get("follow_through"), dict) else {}
        drain_follow_through_status = str(follow_through.get("status") or "")
        drain_follow_through_progress_state = str(follow_through.get("progress_state") or "")
        applied = int(result.get("rc", 1)) == 0
        refresh_steps = _refresh_surface_artifacts(project_root)
    elif apply and quarantine_actionable:
        result = _run_json_command(
            [
                str(PY),
                str(project_root / "scripts" / "ops" / "backlog_quarantine_bot.py"),
                "--apply",
                "--json",
            ],
            cwd=project_root,
            payload_path=project_root / "governance" / "health" / "backlog_quarantine_bot_latest.json",
            timeout_sec=120,
        )
        steps["backlog_quarantine_bot"] = _step_record(result)
        quarantine_payload = result.get("payload") if isinstance(result.get("payload"), dict) else {}
        quarantine_applied = int(result.get("rc", 1)) == 0 and _safe_int(quarantine_payload.get("moved_files"), 0) >= 0
        refresh_steps = _refresh_surface_artifacts(project_root)

    if not backlog_needed:
        overall_status = "idle"
        ok = True
    elif quarantine_actionable and not apply:
        overall_status = "quarantine_ready"
        ok = True
    elif quarantine_actionable and quarantine_applied:
        overall_status = "quarantine_applied_waiting_for_off_hours"
        ok = True
    elif quarantine_actionable and apply:
        overall_status = "quarantine_failed"
        ok = False
    elif "market_hours_guard" in blocked_reasons:
        overall_status = "waiting_for_off_hours"
        ok = True
    elif blocked_reasons:
        overall_status = "blocked"
        ok = False
    elif not apply:
        overall_status = "ready"
        ok = True
    elif drain_follow_through_status == "completed" and not bool(drain_payload.get("writer_busy", False)):
        overall_status = "applied"
        ok = True
    elif applied and drain_follow_through_status == "timed_out" and drain_follow_through_progress_state == "progressing":
        overall_status = "applied_progressing"
        ok = True
    elif applied:
        overall_status = "applied_with_followups"
        ok = False
    else:
        overall_status = "apply_failed"
        ok = False

    recommended_actions: list[str] = []
    if overall_status == "waiting_for_off_hours":
        recommended_actions.append("keep the retry bot installed so it automatically resumes during the next off-hours window")
    if overall_status == "quarantine_ready":
        recommended_actions.append("apply backlog quarantine now so stale prior-day attribution and explanation debt stops consuming the live backlog budget")
    if overall_status == "quarantine_applied_waiting_for_off_hours":
        recommended_actions.append("the retry bot staged stale prior-day backlog during market hours and will resume the heavier drain flow automatically after the off-hours window opens")
    if overall_status == "quarantine_failed":
        recommended_actions.append("inspect backlog quarantine move errors before the next retry cycle so stale cold debt can still be removed during market hours")
    if overall_status == "blocked":
        recommended_actions.extend(list(preview.get("top_actions") or [])[:3])
    if overall_status == "applied_progressing":
        recommended_actions.append("the retry bot handed work to a busy SQL writer that kept making progress, so let the current off-hours cycle finish before forcing another pass")
    if overall_status == "applied_with_followups":
        recommended_actions.append("the retry bot ran, but the SQL writer stayed busy long enough that another pass is still needed")
    if drain_follow_through_status == "timed_out":
        if drain_follow_through_progress_state == "progressing":
            recommended_actions.append("raise the follow-through timeout only if you want the bot to wait longer for a handoff that is already progressing")
        else:
            recommended_actions.append("raise the follow-through timeout or give the SQL writer a quieter maintenance window if you want the bot to catch the handoff more often")
    if actionable and not apply:
        recommended_actions.append("run the retry bot in apply mode or keep the launchd job enabled so it can execute automatically")
    if not recommended_actions:
        recommended_actions.append("keep the retry bot enabled so deferred and cold backlog drain attempts stay automatic during off-hours")

    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "schema_version": 1,
        "ok": ok,
        "overall_status": overall_status,
        "apply": bool(apply),
        "actionable": actionable,
        "quarantine_actionable": quarantine_actionable,
        "backlog_needed": backlog_needed,
        "preview_status": preview_status,
        "blocked_reasons": blocked_reasons,
        "off_hours_window": off_hours,
        "drain_preview": {
            "recommended_now": bool(preview.get("recommended_now", False)),
            "storage_mode": str(preview.get("storage_mode") or ""),
            "writer_busy": bool(preview.get("writer_busy", False)),
            "aged_candidate_files": _safe_int(preview.get("aged_candidate_files"), 0),
        },
        "backpressure": preview.get("backpressure_before") if isinstance(preview.get("backpressure_before"), dict) else {},
        "quarantine_preview": {
            "candidate_files": quarantine_candidate_files,
            "candidate_pending_lines": _safe_int(quarantine_preview.get("candidate_pending_lines"), 0),
            "overall_status": str(quarantine_preview.get("overall_status") or ""),
        },
        "drain_result": {
            "applied": applied,
            "overall_status": str(drain_payload.get("overall_status") or ""),
            "writer_busy": bool(drain_payload.get("writer_busy", False)),
            "follow_through_status": drain_follow_through_status,
            "follow_through_progress_state": drain_follow_through_progress_state,
            "follow_through_progress_observed": bool(((drain_payload.get("follow_through") or {}).get("progress_observed"))),
            "follow_through_progress_events": _safe_int(((drain_payload.get("follow_through") or {}).get("progress_events")), 0),
            "follow_through_attempts": _safe_int(((drain_payload.get("follow_through") or {}).get("attempts")), 0),
            "waited_seconds": float(((drain_payload.get("follow_through") or {}).get("waited_seconds", 0.0)) or 0.0),
        },
        "quarantine_result": {
            "applied": quarantine_applied,
            "overall_status": str(quarantine_payload.get("overall_status") or ""),
            "moved_files": _safe_int(quarantine_payload.get("moved_files"), 0),
            "moved_pending_lines": _safe_int(quarantine_payload.get("moved_pending_lines"), 0),
        },
        "steps": steps,
        "refresh_steps": refresh_steps,
        "recommended_actions": recommended_actions[:6],
    }
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Background infrastructure bot that retries the external backlog drain automatically during off-hours.")
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--out-file", default=str(DEFAULT_OUT_PATH))
    parser.add_argument("--lock-file", default=str(DEFAULT_LOCK_PATH))
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--poll-seconds", type=float, default=DEFAULT_POLL_SECONDS)
    parser.add_argument("--wait-timeout-seconds", type=float, default=DEFAULT_WAIT_TIMEOUT_SECONDS)
    parser.add_argument("--command-timeout-seconds", type=int, default=DEFAULT_COMMAND_TIMEOUT_SECONDS)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    out_file = Path(args.out_file).expanduser()
    lock_file = Path(args.lock_file).expanduser()
    lock_file.parent.mkdir(parents=True, exist_ok=True)

    payload: dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
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
                print("external_backlog_retry_bot overall_status=already_running")
            return 0

        payload = build_payload(
            project_root,
            apply=bool(args.apply),
            poll_seconds=float(args.poll_seconds),
            wait_timeout_seconds=float(args.wait_timeout_seconds),
            command_timeout_seconds=int(args.command_timeout_seconds),
        )
        _write_json(out_file, payload)
        if bool(args.apply):
            payload["post_write_refresh_steps"] = _refresh_surface_artifacts(project_root)
            _write_json(out_file, payload)

    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        print(
            "external_backlog_retry_bot "
            f"overall_status={payload.get('overall_status', '')} "
            f"actionable={int(bool(payload.get('actionable', False)))}"
        )
    return 0 if bool(payload.get("ok", False) or str(payload.get("overall_status") or "") in {"already_running", "waiting_for_off_hours"}) else 2


if __name__ == "__main__":
    raise SystemExit(main())
