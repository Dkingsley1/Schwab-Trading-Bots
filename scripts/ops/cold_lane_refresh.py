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


PY = resolve_runtime_python(PROJECT_ROOT)
DEFAULT_OUT_PATH = PROJECT_ROOT / "governance" / "health" / "cold_lane_refresh_latest.json"
DEFAULT_STRATEGY_PATH = PROJECT_ROOT / "governance" / "health" / "strategy_research_latest.json"
DEFAULT_LOCK_PATH = PROJECT_ROOT / "governance" / "health" / "cold_lane_refresh.lock"


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


def _parse_payload_ts(payload: dict[str, Any], path: Path) -> datetime | None:
    for key in ("timestamp_utc", "updated_at_utc", "generated_utc", "created_at"):
        raw = str(payload.get(key) or "").strip()
        if not raw:
            continue
        try:
            dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        except Exception:
            continue
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    try:
        return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
    except Exception:
        return None


def _artifact_age_minutes(path: Path) -> float | None:
    payload = _load_json(path)
    if not payload and not path.exists():
        return None
    ts = _parse_payload_ts(payload, path)
    if ts is None:
        return None
    return max((datetime.now(timezone.utc) - ts).total_seconds(), 0.0) / 60.0


def _run_resource_guard(profile: str) -> tuple[bool, dict[str, Any]]:
    guard_script = PROJECT_ROOT / "scripts" / "resource_guard.py"
    if not guard_script.exists():
        return True, {"ok": True, "reason": "resource_guard_missing"}
    proc = subprocess.run(
        [str(PY), str(guard_script), "--profile", str(profile or "optional")],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    detail = {
        "ok": proc.returncode == 0,
        "rc": int(proc.returncode),
        "stdout_tail": "\n".join((proc.stdout or "").splitlines()[-8:]),
        "stderr_tail": "\n".join((proc.stderr or "").splitlines()[-8:]),
    }
    return proc.returncode == 0, detail


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the heavier cold-lane strategy research refresh only when stale and resources allow.")
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--day", default=datetime.now(timezone.utc).strftime("%Y%m%d"))
    parser.add_argument("--max-rows", type=int, default=4000)
    parser.add_argument("--strategy-max-age-minutes", type=float, default=180.0)
    parser.add_argument("--sandbox-max-age-minutes", type=float, default=720.0)
    parser.add_argument("--resource-profile", default="optional")
    parser.add_argument("--strategy-out-file", default=str(DEFAULT_STRATEGY_PATH))
    parser.add_argument("--out-file", default=str(DEFAULT_OUT_PATH))
    parser.add_argument("--lock-file", default=str(DEFAULT_LOCK_PATH))
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    strategy_out_file = Path(args.strategy_out_file).expanduser()
    out_file = Path(args.out_file).expanduser()
    lock_file = Path(args.lock_file).expanduser()
    lock_file.parent.mkdir(parents=True, exist_ok=True)

    payload: dict[str, Any] = {
        "timestamp_utc": _utc_now(),
        "project_root": str(project_root),
        "day": str(args.day),
        "ok": True,
        "ran": False,
        "skipped": False,
        "reason": "pending",
        "strategy_out_file": str(strategy_out_file),
    }

    with lock_file.open("a+", encoding="utf-8") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            payload.update({"skipped": True, "reason": "already_running"})
            _write_json(out_file, payload)
            if args.json:
                print(json.dumps(payload, ensure_ascii=True))
            else:
                print("cold_lane_refresh skipped=1 reason=already_running")
            return 0

        age_minutes = _artifact_age_minutes(strategy_out_file)
        existing_strategy_payload = _load_json(strategy_out_file)
        payload["strategy_age_minutes_before"] = None if age_minutes is None else round(float(age_minutes), 3)
        if (
            (not args.force)
            and age_minutes is not None
            and age_minutes <= max(float(args.strategy_max_age_minutes), 0.0)
            and bool(existing_strategy_payload.get("ok", False))
        ):
            payload.update({"skipped": True, "reason": "fresh_strategy_research_reused"})
            _write_json(out_file, payload)
            if args.json:
                print(json.dumps(payload, ensure_ascii=True))
            else:
                print("cold_lane_refresh skipped=1 reason=fresh_strategy_research_reused")
            return 0

        guard_ok, guard_detail = _run_resource_guard(str(args.resource_profile))
        payload["resource_guard"] = guard_detail
        if not guard_ok:
            payload.update({"skipped": True, "reason": "resource_guard_blocked"})
            _write_json(out_file, payload)
            if args.json:
                print(json.dumps(payload, ensure_ascii=True))
            else:
                print("cold_lane_refresh skipped=1 reason=resource_guard_blocked")
            return 0

        cmd = [
            str(PY),
            str(project_root / "scripts" / "strategy_research_lane.py"),
            "--day",
            str(args.day),
            "--max-rows",
            str(max(int(args.max_rows), 1)),
            "--max-age-minutes",
            str(max(float(args.strategy_max_age_minutes), 0.0)),
            "--sandbox-max-age-minutes",
            str(max(float(args.sandbox_max_age_minutes), 0.0)),
            "--json",
        ]
        started = datetime.now(timezone.utc)
        proc = subprocess.run(
            cmd,
            cwd=str(project_root),
            capture_output=True,
            text=True,
            check=False,
        )
        duration_ms = round((datetime.now(timezone.utc) - started).total_seconds() * 1000.0, 3)
        strategy_payload = _load_json(strategy_out_file)
        if not strategy_payload:
            stdout = (proc.stdout or "").strip()
            if stdout.startswith("{") and stdout.endswith("}"):
                try:
                    strategy_payload = json.loads(stdout)
                except Exception:
                    strategy_payload = {}

        strategy_ok = proc.returncode == 0 and bool(strategy_payload.get("ok", False))
        if proc.returncode != 0:
            reason = f"strategy_research_exit_{proc.returncode}"
        elif not strategy_ok:
            reason = "strategy_research_not_ok"
        else:
            reason = "ok"
        payload.update(
            {
                "ran": True,
                "ok": strategy_ok,
                "reason": reason,
                "step": {
                    "name": "strategy_research_full_refresh",
                    "cmd": cmd,
                    "rc": int(proc.returncode),
                    "duration_ms": duration_ms,
                    "stdout_tail": "\n".join((proc.stdout or "").splitlines()[-12:]),
                    "stderr_tail": "\n".join((proc.stderr or "").splitlines()[-12:]),
                },
            }
        )
        age_after = _artifact_age_minutes(strategy_out_file)
        payload["strategy_age_minutes_after"] = None if age_after is None else round(float(age_after), 3)
        payload["strategy_summary"] = {
            "promotable": bool(strategy_payload.get("promotable", False)),
            "recommended_action": str(((strategy_payload.get("summary") or {}).get("recommended_action")) or ""),
            "research_sandbox_ok": bool(strategy_payload.get("research_sandbox_ok", False)),
        }
        _write_json(out_file, payload)

    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        print(
            "cold_lane_refresh "
            f"ok={int(bool(payload.get('ok', False)))} "
            f"ran={int(bool(payload.get('ran', False)))} "
            f"reason={payload.get('reason', '')}"
        )
    return 0 if payload.get("ok", False) or payload.get("skipped", False) else 1


if __name__ == "__main__":
    raise SystemExit(main())
