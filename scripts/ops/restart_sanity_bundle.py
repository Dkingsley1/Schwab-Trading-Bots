#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OPSCTL = PROJECT_ROOT / "scripts" / "ops" / "opsctl.sh"
MACRO_AUTO_STATUS = PROJECT_ROOT / "governance" / "health" / "macro_auto_watch_status.json"
HEALTH_ROOT = PROJECT_ROOT / "governance" / "health"


def _parse_ts(raw: Any) -> datetime | None:
    text = str(raw or "").strip().replace("Z", "+00:00")
    if not text:
        return None
    try:
        dt = datetime.fromisoformat(text)
    except Exception:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _parse_json_from_stdout(stdout_text: str) -> dict[str, Any]:
    text = str(stdout_text or "").strip()
    if not text:
        return {}
    try:
        payload = json.loads(text)
        return payload if isinstance(payload, dict) else {}
    except Exception:
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        for line in reversed(lines):
            try:
                payload = json.loads(line)
                return payload if isinstance(payload, dict) else {}
            except Exception:
                continue
    return {}


def _run_json(cmd: list[str], *, timeout_seconds: float | None = None) -> dict[str, Any]:
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout_seconds,
        )
        payload = _parse_json_from_stdout(proc.stdout or "")
        return {
            "ok": proc.returncode == 0,
            "rc": int(proc.returncode),
            "timed_out": False,
            "command": [str(item) for item in cmd],
            "payload": payload,
            "stdout_tail": "\n".join((proc.stdout or "").splitlines()[-8:]),
            "stderr_tail": "\n".join((proc.stderr or "").splitlines()[-8:]),
        }
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout if isinstance(exc.stdout, str) else ""
        stderr = exc.stderr if isinstance(exc.stderr, str) else ""
        return {
            "ok": False,
            "rc": 124,
            "timed_out": True,
            "command": [str(item) for item in cmd],
            "payload": _parse_json_from_stdout(stdout),
            "stdout_tail": "\n".join(stdout.splitlines()[-8:]),
            "stderr_tail": "\n".join(stderr.splitlines()[-8:]),
        }


def _age_minutes(raw_ts: Any) -> float | None:
    ts = _parse_ts(raw_ts)
    if ts is None:
        return None
    return max((datetime.now(timezone.utc) - ts).total_seconds(), 0.0) / 60.0


def _safe_int(raw: Any, default: int = 0) -> int:
    try:
        return int(raw)
    except Exception:
        return default


def _artifact_status(path: Path, *, max_age_minutes: float) -> dict[str, Any]:
    payload = _load_json(path)
    age_minutes = _age_minutes(payload.get("timestamp_utc"))
    fresh = age_minutes is not None and age_minutes <= max(max_age_minutes, 0.0)
    completed = _safe_int(payload.get("completed_shard_count"))
    planned = _safe_int(payload.get("planned_shard_count"))
    running = bool(payload.get("running", False))
    status = str(payload.get("overall_status") or payload.get("status") or payload.get("current_step") or "").strip()
    current_step = str(payload.get("current_step") or "").strip()
    ok = payload.get("ok")
    ok_flag = ok is not False
    complete = bool(planned > 0 and completed >= planned)
    healthy = bool(fresh and ok_flag and (running or complete or status in {"ready", "ok", "complete", "idle"} or current_step == "complete"))
    return {
        "path": str(path),
        "exists": path.exists(),
        "fresh": bool(fresh),
        "age_minutes": round(age_minutes, 4) if age_minutes is not None else None,
        "healthy": bool(healthy),
        "running": bool(running),
        "complete": bool(complete),
        "ok": ok,
        "status": status,
        "current_step": current_step,
        "completed_shard_count": completed,
        "planned_shard_count": planned,
        "timestamp_utc": str(payload.get("timestamp_utc") or ""),
    }


def _writer_cycle_status(path: Path, *, max_age_minutes: float) -> dict[str, Any]:
    payload = _load_json(path)
    age_minutes = _age_minutes(payload.get("timestamp_utc"))
    fresh = age_minutes is not None and age_minutes <= max(max_age_minutes, 0.0)
    writer_state = payload.get("writer_state_after_remediation")
    if not isinstance(writer_state, dict):
        writer_state = payload.get("writer_state_after_wait")
    if not isinstance(writer_state, dict):
        writer_state = payload.get("writer_state_before")
    if not isinstance(writer_state, dict):
        writer_state = {}
    current_step = str(writer_state.get("effective_current_step") or writer_state.get("current_step") or "").strip()
    status = str(payload.get("overall_status") or writer_state.get("status") or "").strip()
    completed = _safe_int(writer_state.get("completed_shard_count"))
    planned = _safe_int(writer_state.get("planned_shard_count"))
    active = bool(writer_state.get("active", False))
    complete = bool(planned > 0 and completed >= planned)
    healthy = bool(fresh and payload.get("ok") is not False and (active or complete or current_step == "complete" or status in {"ready", "ok", "idle"}))
    return {
        "path": str(path),
        "exists": path.exists(),
        "fresh": bool(fresh),
        "age_minutes": round(age_minutes, 4) if age_minutes is not None else None,
        "healthy": bool(healthy),
        "active": bool(active),
        "complete": bool(complete),
        "ok": payload.get("ok"),
        "status": status,
        "current_step": current_step,
        "completed_shard_count": completed,
        "planned_shard_count": planned,
        "timestamp_utc": str(payload.get("timestamp_utc") or ""),
    }


def _sql_sync_health_snapshot(*, max_age_minutes: float) -> dict[str, Any]:
    progress = _artifact_status(HEALTH_ROOT / "sql_link_service_progress_latest.json", max_age_minutes=max_age_minutes)
    service = _artifact_status(HEALTH_ROOT / "sql_link_service_latest.json", max_age_minutes=max_age_minutes)
    coordinator = _writer_cycle_status(HEALTH_ROOT / "writer_cycle_coordinator_latest.json", max_age_minutes=max_age_minutes)
    sources = [progress, service, coordinator]
    healthy_sources = [source for source in sources if bool(source.get("healthy", False))]
    active_sources = [source for source in healthy_sources if bool(source.get("running", False) or source.get("active", False))]
    complete_sources = [source for source in healthy_sources if bool(source.get("complete", False))]
    ready = bool(active_sources or complete_sources)
    if active_sources:
        state = "active_progressing"
        reason = "fresh_sql_writer_progress_artifact"
    elif complete_sources:
        state = "recent_complete"
        reason = "recent_sql_writer_completion_artifact"
    else:
        state = "stale_or_missing"
        reason = "no_fresh_sql_writer_artifact"
    return {
        "ready": bool(ready),
        "state": state,
        "reason": reason,
        "max_age_minutes": max_age_minutes,
        "progress": progress,
        "service": service,
        "coordinator": coordinator,
    }


def _run_sql_sync_with_guard(
    *,
    timeout_seconds: float,
    max_age_minutes: float,
    runner: Callable[[list[str], float | None], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    cmd = [str(OPSCTL), "sql-sync", "--json"]
    before = _sql_sync_health_snapshot(max_age_minutes=max_age_minutes)
    if bool(before.get("ready", False)):
        return {
            "ok": True,
            "rc": 0,
            "timed_out": False,
            "skipped": True,
            "reason": before.get("reason"),
            "command": cmd,
            "payload": {
                "ok": True,
                "overall_status": "ready",
                "status": before.get("state"),
                "sanity_source": "artifact_snapshot",
            },
            "artifact_health": before,
            "stdout_tail": "",
            "stderr_tail": "",
        }

    call_runner = runner or (lambda run_cmd, run_timeout: _run_json(run_cmd, timeout_seconds=run_timeout))
    result = call_runner(cmd, timeout_seconds)
    after = _sql_sync_health_snapshot(max_age_minutes=max_age_minutes)
    result["artifact_health"] = after
    if bool(after.get("ready", False)):
        result["ok"] = True
        result["sanity_observed_ok"] = True
        result["reason"] = after.get("reason")
        result["payload"] = {
            **(result.get("payload") if isinstance(result.get("payload"), dict) else {}),
            "ok": True,
            "overall_status": "ready",
            "status": after.get("state"),
            "sanity_source": "artifact_snapshot_after_probe",
        }
    return result


def _watcher_summary(path: Path, *, max_age_hours: float) -> dict[str, Any]:
    payload = _load_json(path)
    ts = _parse_ts(payload.get("timestamp_utc"))
    age_hours = None
    fresh = False
    if ts is not None:
        age_hours = max((datetime.now(timezone.utc) - ts).total_seconds(), 0.0) / 3600.0
        fresh = age_hours <= max(max_age_hours, 0.0)
    return {
        "path": str(path),
        "exists": path.exists(),
        "fresh": bool(fresh),
        "age_hours": round(float(age_hours), 6) if age_hours is not None else None,
        "stream_state": str(payload.get("stream_state") or ""),
        "resolved_video_url": str(payload.get("resolved_video_url") or ""),
        "youtube_channel_url": str(payload.get("youtube_channel_url") or ""),
        "media_ingest_triggered": bool(payload.get("media_ingest_triggered", False)),
        "timestamp_utc": str(payload.get("timestamp_utc") or ""),
    }


def build_restart_sanity_bundle(
    *,
    start_after: bool,
    start_mode: str,
    force_restart: bool,
    max_watcher_age_hours: float,
    probe_timeout_seconds: float = 45.0,
    sql_sync_timeout_seconds: float = 20.0,
    sql_artifact_max_age_minutes: float = 30.0,
) -> dict[str, Any]:
    sql_sync = _run_sql_sync_with_guard(
        timeout_seconds=sql_sync_timeout_seconds,
        max_age_minutes=sql_artifact_max_age_minutes,
    )
    token_guard = _run_json(
        [sys.executable, str(PROJECT_ROOT / "scripts" / "ops" / "premarket_token_guard.py"), "--no-always-auth", "--json"],
        timeout_seconds=probe_timeout_seconds,
    )
    live_readiness = _run_json(
        [sys.executable, str(PROJECT_ROOT / "scripts" / "live_readiness_smoke.py"), "--json"],
        timeout_seconds=probe_timeout_seconds,
    )
    control_plane = _run_json(
        [sys.executable, str(PROJECT_ROOT / "scripts" / "platform_control_plane_report.py"), "--json"],
        timeout_seconds=probe_timeout_seconds,
    )
    watcher_status = _watcher_summary(MACRO_AUTO_STATUS, max_age_hours=max_watcher_age_hours)

    start_result: dict[str, Any] = {
        "attempted": False,
        "ok": False,
        "command": [],
        "stdout_tail": "",
        "stderr_tail": "",
    }
    if start_after:
        start_cmd = [str(OPSCTL), start_mode]
        if force_restart:
            start_cmd.append("--force-restart")
        proc = subprocess.run(
            start_cmd,
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            check=False,
        )
        start_result = {
            "attempted": True,
            "ok": proc.returncode == 0,
            "rc": int(proc.returncode),
            "command": start_cmd,
            "stdout_tail": "\n".join((proc.stdout or "").splitlines()[-8:]),
            "stderr_tail": "\n".join((proc.stderr or "").splitlines()[-8:]),
        }

    ok = bool(sql_sync.get("ok")) and bool(token_guard.get("ok")) and bool(live_readiness.get("ok")) and bool(control_plane.get("ok"))
    if start_after:
        ok = ok and bool(start_result.get("ok"))

    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "ok": bool(ok),
        "sql_sync": sql_sync,
        "token_guard": token_guard,
        "live_readiness": live_readiness,
        "control_plane": control_plane,
        "watcher_status": watcher_status,
        "start_result": start_result,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a pre-restart sanity bundle for SQL, broker readiness, control-plane health, and watcher freshness.")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--start-after", action="store_true", help="Start the configured stack after the sanity pass.")
    parser.add_argument("--start-mode", choices=["start", "start-sim", "start-live"], default="start")
    parser.add_argument("--force-restart", action="store_true")
    parser.add_argument("--max-watcher-age-hours", type=float, default=6.0)
    parser.add_argument(
        "--probe-timeout-seconds",
        type=float,
        default=float(os.getenv("RESTART_SANITY_PROBE_TIMEOUT_SECONDS", "45")),
        help="Bound non-SQL probes so restart settlement cannot hang indefinitely.",
    )
    parser.add_argument(
        "--sql-sync-timeout-seconds",
        type=float,
        default=float(os.getenv("RESTART_SANITY_SQL_SYNC_TIMEOUT_SECONDS", "20")),
        help="Bound any forced SQL sync probe; fresh SQL writer artifacts are preferred.",
    )
    parser.add_argument(
        "--sql-artifact-max-age-minutes",
        type=float,
        default=float(os.getenv("RESTART_SANITY_SQL_ARTIFACT_MAX_AGE_MINUTES", "30")),
        help="Freshness window for accepting SQL writer progress/completion artifacts.",
    )
    args = parser.parse_args()

    payload = build_restart_sanity_bundle(
        start_after=bool(args.start_after),
        start_mode=str(args.start_mode),
        force_restart=bool(args.force_restart),
        max_watcher_age_hours=float(args.max_watcher_age_hours),
        probe_timeout_seconds=max(float(args.probe_timeout_seconds), 1.0),
        sql_sync_timeout_seconds=max(float(args.sql_sync_timeout_seconds), 1.0),
        sql_artifact_max_age_minutes=max(float(args.sql_artifact_max_age_minutes), 1.0),
    )

    if args.json:
        print(json.dumps(payload, ensure_ascii=True, indent=2))
    else:
        print(
            "restart_sanity "
            f"ok={int(bool(payload.get('ok')))} "
            f"sql_ok={int(bool((payload.get('sql_sync') or {}).get('ok')))} "
            f"token_ok={int(bool((payload.get('token_guard') or {}).get('ok')))} "
            f"live_ok={int(bool((payload.get('live_readiness') or {}).get('ok')))} "
            f"control_ok={int(bool((payload.get('control_plane') or {}).get('ok')))} "
            f"watcher_fresh={int(bool((payload.get('watcher_status') or {}).get('fresh')))}"
        )
    return 0 if bool(payload.get("ok")) else 1


if __name__ == "__main__":
    raise SystemExit(main())
