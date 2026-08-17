#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOCK_PATH = Path(
    os.getenv(
        "PAPER_TRADE_LOCK_PATH",
        str(PROJECT_ROOT / "governance" / "health" / "PAPER_TRADE_LOCK.flag"),
    )
)
HEALTH_PATH = PROJECT_ROOT / "governance" / "health" / "paper_trade_lock_infrabot_latest.json"


CODE_CONTRACTS = [
    {
        "name": "opsctl_paper_lock_env",
        "path": PROJECT_ROOT / "scripts" / "ops" / "opsctl.sh",
        "snippets": [
            'PAPER_TRADE_LOCK_FILE="$PROJECT_ROOT/governance/health/PAPER_TRADE_LOCK.flag"',
            "paper_trade_lock_env()",
            "PAPER_TRADE_LOCK=1",
            "RUN_ALL_SLEEVES_WITH_LIVE_EXECUTOR=0",
        ],
    },
    {
        "name": "opsctl_all_feeds_include_fx",
        "path": PROJECT_ROOT / "scripts" / "ops" / "opsctl.sh",
        "snippets": [
            '[[ "$SOURCE" == "fx" || "$SOURCE" == "all" ]]',
            'fx-start --paper --force-restart --live-data',
        ],
    },
    {
        "name": "all_sleeves_live_executor_guard",
        "path": PROJECT_ROOT / "scripts" / "run_all_sleeves.py",
        "snippets": [
            "def _paper_trade_lock_enabled()",
            "args.with_live_executor = False",
            'base_env["PAPER_TRADE_LOCK"] = "1"',
        ],
    },
    {
        "name": "execution_lane_live_guard",
        "path": PROJECT_ROOT / "scripts" / "run_execution_lane.py",
        "snippets": [
            "def _paper_trade_lock_enabled()",
            'if args.mode == "live" and _paper_trade_lock_enabled():',
            'auth_error = "paper_trade_lock_active"',
        ],
    },
    {
        "name": "start_stack_paper_lock_env",
        "path": PROJECT_ROOT / "scripts" / "ops" / "start_stack.sh",
        "snippets": [
            'PAPER_TRADE_LOCK_FILE="$HEALTH_DIR/PAPER_TRADE_LOCK.flag"',
            "paper_trade_lock_env()",
            "RUN_ALL_SLEEVES_WITH_LIVE_EXECUTOR=0",
        ],
    },
    {
        "name": "all_sleeves_launchd_paper_lock_env",
        "path": PROJECT_ROOT / "scripts" / "ops" / "run_all_sleeves_launchd.sh",
        "snippets": [
            "PAPER_TRADE_LOCK_PATH",
            "RUN_ALL_SLEEVES_WITH_LIVE_EXECUTOR",
            "live_data_paper_trade_only",
        ],
    },
    {
        "name": "shadow_watchdog_launchd_paper_lock_env",
        "path": PROJECT_ROOT / "scripts" / "ops" / "run_shadow_watchdog_launchd.sh",
        "snippets": [
            "PAPER_TRADE_LOCK_PATH",
            "RUN_ALL_SLEEVES_WITH_LIVE_EXECUTOR",
            "live_data_paper_trade_only",
        ],
    },
]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except Exception:
        return ""


def _contract_status() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for contract in CODE_CONTRACTS:
        path = Path(contract["path"])
        text = _read_text(path)
        missing = [snippet for snippet in contract["snippets"] if snippet not in text]
        rows.append(
            {
                "name": contract["name"],
                "path": str(path),
                "ok": path.exists() and not missing,
                "missing_snippets": missing,
            }
        )
    return rows


def _live_execution_processes() -> list[dict[str, Any]]:
    try:
        proc = subprocess.run(
            ["ps", "-axo", "pid,command"],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
    except Exception as exc:
        return [{"pid": None, "command": "", "scan_error": str(exc)}]

    rows: list[dict[str, Any]] = []
    for line in (proc.stdout or "").splitlines()[1:]:
        raw = line.strip()
        if not raw:
            continue
        parts = raw.split(maxsplit=1)
        if len(parts) != 2:
            continue
        pid, command = parts
        if "scripts/run_execution_lane.py" in command and "--mode live" in command:
            rows.append({"pid": int(pid), "command": command})
    return rows


def _write_lock() -> None:
    LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "enabled_at_utc": _now(),
        "policy": "live_data_paper_trade_only",
        "managed_by": "paper_trade_lock_infrabot",
        "live_execution_blocked": True,
        "paper_execution_allowed": True,
    }
    LOCK_PATH.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")


def _build_payload(*, apply: bool) -> dict[str, Any]:
    if apply:
        _write_lock()

    contracts = _contract_status()
    live_processes = _live_execution_processes()
    process_scan_errors = [row for row in live_processes if row.get("scan_error")]
    live_process_count = len([row for row in live_processes if row.get("pid") is not None])
    contract_ok = all(bool(row.get("ok")) for row in contracts)
    lock_present = LOCK_PATH.exists()

    if live_process_count:
        status = "blocked"
        ok = False
        severity = "critical"
    elif contract_ok and lock_present:
        status = "ready"
        ok = True
        severity = "info"
    elif contract_ok:
        status = "needs_lock"
        ok = False
        severity = "warning"
    else:
        status = "needs_repair"
        ok = False
        severity = "critical"

    recommended_actions: list[str] = []
    if not lock_present:
        recommended_actions.append("run paper-trade-lock-infrabot --apply before starting overnight live-data paper loops")
    if not contract_ok:
        recommended_actions.append("repair missing paper lock code contracts before trusting all-sleeves startup")
    if live_process_count:
        recommended_actions.append("stop live execution lane processes before relying on paper-only operation")
    if process_scan_errors:
        recommended_actions.append("process scan failed; re-run status from an unrestricted terminal if needed")
    recommended_actions.append("start loops through ./scripts/ops/opsctl.sh livefeed-refresh so the lock is inherited by every paper-capable sleeve")

    return {
        "timestamp_utc": _now(),
        "schema_version": 1,
        "ok": ok,
        "overall_status": status,
        "severity": severity,
        "apply_requested": bool(apply),
        "lock_path": str(LOCK_PATH),
        "lock_present": lock_present,
        "paper_execution_allowed": True,
        "live_execution_blocked": lock_present and live_process_count == 0,
        "code_contracts_ok": contract_ok,
        "code_contracts": contracts,
        "live_execution_process_count": live_process_count,
        "live_execution_processes": live_processes,
        "recommended_actions": recommended_actions,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit and enable the persistent paper-trade lock.")
    parser.add_argument("--apply", action="store_true", help="Create or refresh the persistent paper trade lock.")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    payload = _build_payload(apply=bool(args.apply))
    HEALTH_PATH.parent.mkdir(parents=True, exist_ok=True)
    HEALTH_PATH.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")

    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        print(f"paper_trade_lock_status={payload['overall_status']}")
        print(f"lock_path={payload['lock_path']}")
        print(f"live_execution_process_count={payload['live_execution_process_count']}")
    return 0 if bool(payload.get("ok")) else 2


if __name__ == "__main__":
    raise SystemExit(main())
