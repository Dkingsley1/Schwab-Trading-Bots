#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from core.runtime_python import resolve_runtime_python
    from core.accountability import safe_write_json_atomic
    from scripts.brokers.schwab.common import build_schwab_trader
    from scripts.run_shadow_training_loop import (
        _broker_truth_latest_path,
        _fetch_broker_truth_snapshot,
        _write_broker_truth_shared_snapshot,
    )
else:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    from core.runtime_python import resolve_runtime_python
    from core.accountability import safe_write_json_atomic
    from scripts.brokers.schwab.common import build_schwab_trader
    from scripts.run_shadow_training_loop import (
        _broker_truth_latest_path,
        _fetch_broker_truth_snapshot,
        _write_broker_truth_shared_snapshot,
    )


PY = resolve_runtime_python(PROJECT_ROOT)
DEFAULT_OUT_PATH = PROJECT_ROOT / "governance" / "health" / "schwab_account_snapshot_refresh_latest.json"


def _quiet_auth(trader: Any, *, quiet: bool) -> None:
    if not quiet:
        trader.authenticate()
        return
    with open(os.devnull, "w", encoding="utf-8") as devnull:
        with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
            trader.authenticate()


def _positions_len(fetched: dict[str, Any]) -> int:
    payload = fetched.get("payload") if isinstance(fetched.get("payload"), dict) else {}
    accounts = payload.get("accounts")
    if isinstance(accounts, list):
        total = 0
        for row in accounts:
            if not isinstance(row, dict):
                continue
            sec = row.get("securitiesAccount") if isinstance(row.get("securitiesAccount"), dict) else row
            positions = sec.get("positions") if isinstance(sec, dict) else []
            total += len(positions) if isinstance(positions, list) else 0
        return total
    sec = payload.get("securitiesAccount") if isinstance(payload.get("securitiesAccount"), dict) else payload
    positions = sec.get("positions") if isinstance(sec, dict) else []
    return len(positions) if isinstance(positions, list) else 0


def _run_artifact(cmd: list[str], *, timeout: int = 90) -> dict[str, Any]:
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(PROJECT_ROOT),
            text=True,
            capture_output=True,
            timeout=timeout,
            check=False,
        )
    except Exception as exc:
        return {"ok": False, "error": f"{type(exc).__name__}:{exc}", "cmd": cmd[:3]}
    return {
        "ok": proc.returncode == 0,
        "returncode": int(proc.returncode),
        "stdout_tail": (proc.stdout or "")[-2000:],
        "stderr_tail": (proc.stderr or "")[-2000:],
    }


def refresh(*, quiet_auth: bool, rebuild_derived: bool) -> dict[str, Any]:
    old_env = {
        "ALLOW_ORDER_EXECUTION": os.environ.get("ALLOW_ORDER_EXECUTION"),
        "MARKET_DATA_ONLY": os.environ.get("MARKET_DATA_ONLY"),
        "LIVE_ACCOUNTS_SNAPSHOT_AGGREGATE_CONNECTED": os.environ.get("LIVE_ACCOUNTS_SNAPSHOT_AGGREGATE_CONNECTED"),
    }
    fetched: dict[str, Any] = {}
    write_ok = False
    broker_truth_state: dict[str, Any] = {}
    os.environ["ALLOW_ORDER_EXECUTION"] = "0"
    os.environ["MARKET_DATA_ONLY"] = "1"
    os.environ["LIVE_ACCOUNTS_SNAPSHOT_AGGREGATE_CONNECTED"] = "1"
    try:
        trader = build_schwab_trader(
            PROJECT_ROOT,
            mode="shadow",
            missing_credentials_message="Schwab credentials are required for account snapshot refresh",
        )
        _quiet_auth(trader, quiet=quiet_auth)
        fetched = trader._live_fetch_accounts_payload()
        fetched = dict(fetched or {})
        fetched["_forced_account_snapshot_refresh"] = True
        fetched["_forced_account_snapshot_refreshed_at_utc"] = datetime.now(timezone.utc).isoformat()
        write_ok = False
        broker_truth_state: dict[str, Any] = {}
        if bool(fetched.get("ok", False)):
            write_ok = _write_broker_truth_shared_snapshot(
                project_root=str(PROJECT_ROOT),
                broker="schwab",
                fetched=fetched,
            )
            broker_truth_state = _fetch_broker_truth_snapshot(
                trader=trader,
                broker="schwab",
                simulate=False,
                iter_count=0,
                manual_payload={},
                manual_tolerance=1.0,
                previous_state=None,
            )
            safe_write_json_atomic(
                _broker_truth_latest_path(str(PROJECT_ROOT), "schwab"),
                broker_truth_state,
                project_root=str(PROJECT_ROOT),
                source="schwab_account_snapshot_refresh.broker_truth",
            )
    except Exception as exc:
        fetched = {
            "ok": False,
            "operation": "get_accounts_snapshot",
            "error": f"{type(exc).__name__}:{exc}",
        }
    finally:
        for key, value in old_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    derived: dict[str, Any] = {}
    if rebuild_derived and bool(fetched.get("ok", False)):
        derived["covered_call_roll_watch"] = _run_artifact(
            [str(PY), str(PROJECT_ROOT / "scripts" / "ops" / "covered_call_roll_watch.py"), "--json"]
        )
        derived["account_position_study"] = _run_artifact(
            [str(PY), str(PROJECT_ROOT / "scripts" / "ops" / "account_position_study.py"), "--json"],
            timeout=120,
        )
        derived["schwab_tax_ledger_refresh"] = _run_artifact(
            [str(PY), str(PROJECT_ROOT / "scripts" / "ops" / "schwab_tax_ledger_refresh.py"), "--json"],
            timeout=180,
        )
        derived["trading_tax_estimate"] = _run_artifact(
            [str(PY), str(PROJECT_ROOT / "scripts" / "ops" / "trading_tax_estimator.py"), "--json"],
            timeout=120,
        )
        derived["position_opportunity_watch"] = _run_artifact(
            [str(PY), str(PROJECT_ROOT / "scripts" / "ops" / "position_opportunity_watch.py"), "--json"],
            timeout=120,
        )
        derived["sleeve_allocator"] = _run_artifact(
            [str(PY), str(PROJECT_ROOT / "scripts" / "sleeve_allocator.py"), "--json"],
            timeout=120,
        )
        derived["portfolio_risk_ledger"] = _run_artifact(
            [str(PY), str(PROJECT_ROOT / "scripts" / "portfolio_risk_ledger.py"), "--json"],
            timeout=120,
        )
        derived["position_round_trip_watch"] = _run_artifact(
            [
                str(PROJECT_ROOT / "scripts" / "ops" / "opsctl.sh"),
                "position-round-trip-watch",
                "--refresh-market-data",
                "--json",
            ],
            timeout=240,
        )
        derived["portfolio_allocator_service"] = _run_artifact(
            [str(PY), str(PROJECT_ROOT / "scripts" / "portfolio_allocator_service.py"), "--json"],
            timeout=120,
        )
        derived["account_buildout_plan"] = _run_artifact(
            [str(PY), str(PROJECT_ROOT / "scripts" / "ops" / "account_buildout_planner.py"), "--json"],
            timeout=120,
        )

    payload = fetched.get("payload") if isinstance(fetched.get("payload"), dict) else {}
    broker_truth_ok = bool(broker_truth_state.get("ok", False)) if broker_truth_state else bool(fetched.get("ok", False))
    broker_truth_v2 = broker_truth_state.get("broker_truth_reconcile_v2") if isinstance(broker_truth_state.get("broker_truth_reconcile_v2"), dict) else {}
    summary_ok = bool(fetched.get("ok", False)) and broker_truth_ok
    summary = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "ok": summary_ok,
        "operation": fetched.get("operation", "get_accounts_snapshot"),
        "account_snapshot_mode": str(payload.get("account_snapshot_mode") or fetched.get("account_snapshot_mode") or ""),
        "account_count": int(payload.get("account_count", fetched.get("account_count", 0)) or 0),
        "discovered_account_count": int(
            payload.get("discovered_account_count", fetched.get("discovered_account_count", 0)) or 0
        ),
        "failed_account_count": int(payload.get("failed_account_count", fetched.get("failed_account_count", 0)) or 0),
        "account_snapshot_partial": bool(payload.get("partial", False)),
        "position_rows": _positions_len(fetched),
        "shared_snapshot_write_ok": bool(write_ok),
        "broker_truth_status": str(broker_truth_state.get("status") or ""),
        "broker_truth_position_count": int(broker_truth_state.get("position_count", 0) or 0),
        "broker_truth_mismatch_count": int(broker_truth_state.get("mismatch_count", 0) or 0),
        "broker_truth_ok": broker_truth_ok,
        "broker_truth_error": str(broker_truth_state.get("error") or ""),
        "account_snapshot_proof": broker_truth_state.get("account_snapshot_proof") if isinstance(broker_truth_state.get("account_snapshot_proof"), dict) else {},
        "broker_truth_reconcile_v2": broker_truth_v2,
        "broker_truth_v2_score": float(broker_truth_v2.get("truth_score", 0.0) or 0.0),
        "broker_truth_v2_grade": str(broker_truth_v2.get("truth_grade") or ""),
        "derived": derived,
        "error": str(fetched.get("error") or ""),
        "notes": [
            "Order execution is forced off for this refresh.",
            "Raw account numbers are not emitted in this summary.",
        ],
    }
    safe_write_json_atomic(
        str(DEFAULT_OUT_PATH),
        summary,
        project_root=str(PROJECT_ROOT),
        source="schwab_account_snapshot_refresh",
    )
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Force-refresh Schwab connected-account positions and derived studies.")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--loud-auth", action="store_true", help="Do not suppress Schwab auth chatter.")
    parser.add_argument("--skip-derived", action="store_true", help="Only refresh the broker account snapshot.")
    args = parser.parse_args(argv)

    payload = refresh(quiet_auth=not args.loud_auth, rebuild_derived=not args.skip_derived)
    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        print(
            "schwab_account_snapshot_refresh "
            f"ok={int(payload.get('ok', False))} "
            f"mode={payload.get('account_snapshot_mode') or 'legacy_single'} "
            f"accounts={payload.get('account_count', 0)} "
            f"discovered={payload.get('discovered_account_count', 0)} "
            f"positions={payload.get('position_rows', 0)}"
        )
    return 0 if bool(payload.get("ok", False)) else 1


if __name__ == "__main__":
    raise SystemExit(main())
