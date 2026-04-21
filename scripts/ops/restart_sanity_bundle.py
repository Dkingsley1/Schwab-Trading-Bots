#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OPSCTL = PROJECT_ROOT / "scripts" / "ops" / "opsctl.sh"
MACRO_AUTO_STATUS = PROJECT_ROOT / "governance" / "health" / "macro_auto_watch_status.json"


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


def _run_json(cmd: list[str]) -> dict[str, Any]:
    proc = subprocess.run(
        cmd,
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    payload = _parse_json_from_stdout(proc.stdout or "")
    return {
        "ok": proc.returncode == 0,
        "rc": int(proc.returncode),
        "command": [str(item) for item in cmd],
        "payload": payload,
        "stdout_tail": "\n".join((proc.stdout or "").splitlines()[-8:]),
        "stderr_tail": "\n".join((proc.stderr or "").splitlines()[-8:]),
    }


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


def build_restart_sanity_bundle(*, start_after: bool, start_mode: str, force_restart: bool, max_watcher_age_hours: float) -> dict[str, Any]:
    sql_sync = _run_json([str(OPSCTL), "sql-sync", "--json"])
    token_guard = _run_json([sys.executable, str(PROJECT_ROOT / "scripts" / "ops" / "premarket_token_guard.py"), "--no-always-auth", "--json"])
    live_readiness = _run_json([sys.executable, str(PROJECT_ROOT / "scripts" / "live_readiness_smoke.py"), "--json"])
    control_plane = _run_json([sys.executable, str(PROJECT_ROOT / "scripts" / "platform_control_plane_report.py"), "--json"])
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
    args = parser.parse_args()

    payload = build_restart_sanity_bundle(
        start_after=bool(args.start_after),
        start_mode=str(args.start_mode),
        force_restart=bool(args.force_restart),
        max_watcher_age_hours=float(args.max_watcher_age_hours),
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
