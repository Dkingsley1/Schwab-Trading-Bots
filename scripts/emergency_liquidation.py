import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.base_trader import BaseTrader
from core.halt_flags import write_halt_flag_atomic

OPERATOR_FLAG = PROJECT_ROOT / "governance" / "health" / "OPERATOR_STOP.flag"
GLOBAL_HALT_FLAG = PROJECT_ROOT / "governance" / "health" / "GLOBAL_TRADING_HALT.flag"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_flag(path: Path, payload: dict[str, Any]) -> None:
    ok = write_halt_flag_atomic(
        path,
        payload,
        project_root=str(PROJECT_ROOT),
        source="emergency_liquidation",
    )
    if not ok:
        raise RuntimeError(f"flag_write_failed:{path}")


def _env(name: str, default: str = "") -> str:
    return os.getenv(name, default).strip()


def main() -> int:
    parser = argparse.ArgumentParser(description="Emergency liquidation tool (rare use only).")
    parser.add_argument("--apply", action="store_true", help="Execute real cancels/flatten orders. Without this, dry-run only.")
    parser.add_argument("--confirm", default="", help="Must be EXACTLY: LIQUIDATE_NOW")
    parser.add_argument("--set-operator-stop", action="store_true", default=False)
    parser.add_argument("--set-global-halt", action="store_true", default=False)
    parser.add_argument("--max-orders-cancel", type=int, default=300)
    parser.add_argument("--max-positions", type=int, default=300)
    parser.add_argument("--reason", default="emergency_liquidation_manual")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    dry_run = not bool(args.apply)
    if bool(args.apply) and str(args.confirm).strip() != "LIQUIDATE_NOW":
        raise SystemExit("confirm_token_required: use --confirm LIQUIDATE_NOW")

    api_key = _env("SCHWAB_API_KEY", "YOUR_KEY_HERE")
    secret = _env("SCHWAB_SECRET", "YOUR_SECRET_HERE")
    redirect = _env("SCHWAB_REDIRECT", "https://127.0.0.1:8182")

    trader = BaseTrader(api_key, secret, redirect, mode="live")
    trader.token_path = str(PROJECT_ROOT / "token.json")
    trader.market_data_only = False
    trader.execution_enabled = True

    auth_ok = False
    auth_error = ""
    try:
        trader.authenticate()
        auth_ok = True
    except Exception as exc:
        auth_error = f"{type(exc).__name__}:{exc}"

    flag_payload = {
        "timestamp_utc": _now(),
        "reason": str(args.reason or "emergency_liquidation_manual"),
        "operator": os.getenv("USER", "unknown"),
        "source": "scripts/emergency_liquidation.py",
    }

    if args.set_operator_stop:
        _write_flag(OPERATOR_FLAG, flag_payload)
    if args.set_global_halt:
        _write_flag(GLOBAL_HALT_FLAG, flag_payload)

    cancel_out: dict[str, Any] = {"ok": False, "error": "auth_not_ok"}
    liq_out: dict[str, Any] = {"ok": False, "error": "auth_not_ok"}

    if auth_ok:
        cancel_out = trader.cancel_all_live_open_orders(max_orders=max(int(args.max_orders_cancel), 1))
        liq_out = trader.emergency_liquidate_all_positions(
            max_positions=max(int(args.max_positions), 1),
            dry_run=bool(dry_run),
        )

    payload = {
        "timestamp_utc": _now(),
        "apply": bool(args.apply),
        "dry_run": bool(dry_run),
        "reason": str(args.reason or "emergency_liquidation_manual"),
        "auth_ok": bool(auth_ok),
        "auth_error": auth_error,
        "operator_stop": OPERATOR_FLAG.exists(),
        "global_halt": GLOBAL_HALT_FLAG.exists(),
        "cancel_open_orders": cancel_out,
        "liquidation": liq_out,
        "ok": bool(auth_ok) and bool(cancel_out.get("ok", False)) and bool(liq_out.get("ok", False)),
    }

    out = PROJECT_ROOT / "governance" / "health" / "emergency_liquidation_latest.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")

    events = PROJECT_ROOT / "governance" / "events" / "emergency_liquidation_events.jsonl"
    events.parent.mkdir(parents=True, exist_ok=True)
    with events.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=True) + "\n")

    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        print(
            "emergency_liquidation "
            f"ok={int(payload['ok'])} apply={int(bool(args.apply))} dry_run={int(bool(dry_run))} "
            f"auth_ok={int(bool(auth_ok))} canceled={len(cancel_out.get('canceled', []))} "
            f"positions={int(liq_out.get('positions_considered', 0) or 0)}"
        )

    return 0 if payload["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
