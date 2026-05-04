#!/usr/bin/env python3
from __future__ import annotations

import argparse
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
    from scripts.ops.long_runtime_common import iso_now, status_rank, write_payload
else:
    from .long_runtime_common import PROJECT_ROOT, iso_now, status_rank, write_payload


DEFAULT_OUT_PATH = PROJECT_ROOT / "governance" / "health" / "post_restart_settlement_latest.json"
OPSCTL = PROJECT_ROOT / "scripts" / "ops" / "opsctl.sh"
PROCESS_WATCHDOG = PROJECT_ROOT / "scripts" / "ops" / "process_watchdog.py"


def _extract_json(text: str) -> dict[str, Any]:
    raw = (text or "").strip()
    if not raw:
        return {}
    try:
        payload = json.loads(raw)
        return payload if isinstance(payload, dict) else {}
    except Exception:
        pass
    start = raw.find("{")
    end = raw.rfind("}")
    if start < 0 or end <= start:
        return {}
    try:
        payload = json.loads(raw[start : end + 1])
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _run_step(name: str, cmd: list[str], *, timeout: int, env: dict[str, str] | None = None) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            cmd,
            cwd=str(PROJECT_ROOT),
            env=env,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        stdout = completed.stdout or ""
        stderr = completed.stderr or ""
        payload = _extract_json(stdout)
        return {
            "name": name,
            "cmd": cmd,
            "rc": int(completed.returncode),
            "ok": completed.returncode == 0,
            "timed_out": False,
            "payload": payload,
            "stdout_tail": stdout.strip()[-1200:],
            "stderr_tail": stderr.strip()[-1200:],
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "name": name,
            "cmd": cmd,
            "rc": 124,
            "ok": False,
            "timed_out": True,
            "payload": _extract_json(exc.stdout if isinstance(exc.stdout, str) else ""),
            "stdout_tail": str(exc.stdout or "")[-1200:],
            "stderr_tail": str(exc.stderr or "")[-1200:],
        }
    except Exception as exc:
        return {
            "name": name,
            "cmd": cmd,
            "rc": None,
            "ok": False,
            "timed_out": False,
            "payload": {},
            "stdout_tail": "",
            "stderr_tail": str(exc)[:1200],
        }


def _step_status(step: dict[str, Any]) -> str:
    payload = step.get("payload") if isinstance(step.get("payload"), dict) else {}
    status = str(payload.get("overall_status") or payload.get("status") or "").strip().lower()
    if status:
        return status
    return "ready" if bool(step.get("ok", False)) else "blocked"


def _list_len(payload: dict[str, Any], key: str) -> int:
    value = payload.get(key)
    return len(value) if isinstance(value, list) else 0


def build_settlement(*, apply: bool, max_renice_processes: int, timeout_sec: int, out_path: Path) -> dict[str, Any]:
    py = sys.executable
    env = os.environ.copy()
    env.setdefault("BOT_RUNTIME_PROFILE", "live")
    env.setdefault("MARKET_DATA_ONLY", "1")
    env.setdefault("ALLOW_ORDER_EXECUTION", "0")
    env.setdefault("OPS_WATCHDOG_REFRESH_REPORTS", "0")
    env.setdefault("OPS_WATCHDOG_REQUIRE_COINBASE_FUTURES", "1")

    steps: list[dict[str, Any]] = []
    step_specs: list[tuple[str, list[str], int, dict[str, str] | None]] = [
        ("rolling_restart", [str(OPSCTL), "rolling-restart", "--json"], timeout_sec, env),
        ("restart_sanity", [str(OPSCTL), "restart-sanity", "--json"], timeout_sec, env),
        ("auth_lease", [str(OPSCTL), "auth-lease", "--json"], timeout_sec, env),
        ("schwab_auth_supervisor", [str(OPSCTL), "schwab-auth-supervisor", "--apply", "--json"], timeout_sec, env),
        ("global_halt_refresh", [str(OPSCTL), "global-halt-refresh", "--json"], timeout_sec, env),
        ("collector_contracts", [str(OPSCTL), "collector-contracts", "--json"], timeout_sec, env),
    ]
    if apply:
        step_specs.append(
            (
                "runtime_throttle",
                [str(OPSCTL), "runtime-throttle", "--apply", "--max-renice-processes", str(max_renice_processes), "--json"],
                timeout_sec,
                env,
            )
        )
    else:
        step_specs.append(("runtime_throttle", [str(OPSCTL), "runtime-throttle", "--json"], timeout_sec, env))
    step_specs.append(
        (
            "process_watchdog",
            [py, str(PROCESS_WATCHDOG), "--require-coinbase-futures", "--json"],
            timeout_sec,
            env,
        )
    )

    for name, cmd, timeout, step_env in step_specs:
        steps.append(_run_step(name, cmd, timeout=timeout, env=step_env))

    by_name = {str(step["name"]): step for step in steps}
    halt = by_name.get("global_halt_refresh", {}).get("payload", {})
    watchdog = by_name.get("process_watchdog", {}).get("payload", {})
    throttle = by_name.get("runtime_throttle", {}).get("payload", {})
    auth = by_name.get("auth_lease", {}).get("payload", {})

    statuses = {name: _step_status(step) for name, step in by_name.items()}
    worst = max((status_rank(status) for status in statuses.values()), default=0)
    overall_status = "ready" if worst <= 1 else "degraded" if worst == 2 else "blocked"

    restart_storms = _list_len(watchdog, "restart_storms")
    alerts = _list_len(watchdog, "alerts")
    halt_active = bool(halt.get("halt") or halt.get("global_halt_active") or halt.get("active"))
    clear_blockers = halt.get("clear_blockers") if isinstance(halt.get("clear_blockers"), list) else []
    degraded_blockers = halt.get("degraded_clear_blockers") if isinstance(halt.get("degraded_clear_blockers"), list) else []
    throttle_ready = str(throttle.get("overall_status") or throttle.get("status") or "").lower() in {"ready", "ok"}
    auth_ready = str(auth.get("overall_status") or auth.get("lease_state") or "").lower() in {"ready", "healthy", "ok"}

    recommended_actions: list[str] = []
    if halt_active or clear_blockers:
        recommended_actions.append("run global-halt-refresh, then global-halt-auto-clear once clear_blockers is empty")
    if degraded_blockers:
        recommended_actions.append("treat degraded halt blockers as watch items and rerun settlement after coverage/auth refresh")
    if restart_storms or alerts:
        recommended_actions.append("inspect process_watchdog_latest.json before expanding paper/runtime load")
    if not throttle_ready:
        recommended_actions.append("keep runtime-throttle applied until support processes calm")
    if not auth_ready:
        recommended_actions.append("refresh Schwab auth lease before expecting Schwab live loops to thaw")

    payload = {
        "timestamp_utc": iso_now(),
        "schema_version": 1,
        "ok": overall_status in {"ready", "degraded"} and not halt_active and restart_storms == 0,
        "overall_status": overall_status,
        "apply_requested": apply,
        "steps": {str(step["name"]): step for step in steps},
        "step_statuses": statuses,
        "summary": {
            "halt_active": halt_active,
            "halt_state": str(halt.get("halt_state") or halt.get("state") or ""),
            "clear_blockers": clear_blockers,
            "degraded_clear_blockers": degraded_blockers,
            "process_restart_storms": restart_storms,
            "process_alerts": alerts,
            "runtime_throttle_status": str(throttle.get("overall_status") or throttle.get("status") or ""),
            "ready_for_700_bot_paper": bool(
                (throttle.get("paper_capacity_contract") or {}).get("ready_for_700_bot_paper", False)
            )
            if isinstance(throttle.get("paper_capacity_contract"), dict)
            else False,
            "auth_lease_state": str(auth.get("lease_state") or auth.get("overall_status") or ""),
        },
        "recommended_actions": recommended_actions,
        "artifacts": {
            "out_file": str(out_path),
            "global_killswitch": str(PROJECT_ROOT / "governance" / "health" / "global_killswitch_latest.json"),
            "process_watchdog": str(PROJECT_ROOT / "governance" / "health" / "process_watchdog_latest.json"),
            "runtime_throttle": str(PROJECT_ROOT / "governance" / "health" / "runtime_throttle_control_latest.json"),
        },
    }
    write_payload(out_path, payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a compact post-restart settlement pass across restart, halt, auth, throttle, and process guards.")
    parser.add_argument("--apply", action="store_true", default=False, help="Apply runtime throttle as part of settlement.")
    parser.add_argument("--no-apply", action="store_true", default=False, help="Do not apply runtime throttle.")
    parser.add_argument("--max-renice-processes", type=int, default=8)
    parser.add_argument("--timeout-sec", type=int, default=90)
    parser.add_argument("--out-file", default=str(DEFAULT_OUT_PATH))
    parser.add_argument("--json", action="store_true", default=False)
    args = parser.parse_args()

    apply = bool(args.apply and not args.no_apply)
    payload = build_settlement(
        apply=apply,
        max_renice_processes=max(int(args.max_renice_processes), 0),
        timeout_sec=max(int(args.timeout_sec), 10),
        out_path=Path(args.out_file),
    )
    if args.json:
        print(json.dumps(payload, ensure_ascii=True, indent=2))
    else:
        summary = payload.get("summary", {})
        print(
            "post_restart_settlement "
            f"status={payload.get('overall_status')} "
            f"halt_active={summary.get('halt_active')} "
            f"restart_storms={summary.get('process_restart_storms')} "
            f"out={payload.get('artifacts', {}).get('out_file')}"
        )
    return 0 if payload.get("overall_status") in {"ready", "degraded"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
