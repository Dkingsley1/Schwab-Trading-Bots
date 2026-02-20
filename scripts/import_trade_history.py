import argparse
import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in os.sys.path:
    os.sys.path.insert(0, str(PROJECT_ROOT))

from core.base_trader import BaseTrader


def _load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def _resp_json(resp: Any) -> Any:
    if resp is None:
        return None
    if hasattr(resp, "json"):
        try:
            return resp.json()
        except Exception:
            return None
    return resp


def _account_role_from_number(acct_num: str) -> str:
    last = (acct_num or "")[-1:]
    # Deterministic fallback mapping when explicit account labels are unavailable.
    if last in {"0", "3", "6", "9"}:
        return "ROTH"
    if last in {"1", "4", "7"}:
        return "INDIVIDUAL_TRADING"
    return "INDIVIDUAL_SWING"


def _fetch_accounts(client: Any) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []

    if hasattr(client, "get_account_numbers"):
        data = _resp_json(client.get_account_numbers())
        if isinstance(data, list):
            for r in data:
                if not isinstance(r, dict):
                    continue
                rows.append(
                    {
                        "account_number": str(r.get("accountNumber") or ""),
                        "account_hash": str(r.get("hashValue") or ""),
                    }
                )

    return [r for r in rows if r.get("account_hash")]


def _fetch_transactions_for_account(client: Any, account_hash: str, start_dt: datetime, end_dt: datetime) -> List[Dict[str, Any]]:
    """Fetch transactions using rolling <=60-day windows (Schwab API constraint)."""
    if not hasattr(client, "get_transactions"):
        return []

    window_days = 59
    cursor = start_dt
    rows: List[Dict[str, Any]] = []
    seen = set()

    while cursor <= end_dt:
        window_end = min(cursor + timedelta(days=window_days), end_dt)

        attempts = [
            {
                "account_hash": account_hash,
                "start_date": cursor,
                "end_date": window_end,
            },
            {
                "account_hash": account_hash,
                "startDate": cursor.isoformat(),
                "endDate": window_end.isoformat(),
            },
            {
                "account_hash": account_hash,
                "start_datetime": cursor,
                "end_datetime": window_end,
            },
        ]

        data = None
        for kwargs in attempts:
            try:
                resp = client.get_transactions(**kwargs)
                obj = _resp_json(resp)
                if isinstance(obj, list):
                    data = obj
                    break
            except Exception:
                continue

        if data:
            for tx in data:
                if not isinstance(tx, dict):
                    continue
                tx_id = str(tx.get("transactionId") or tx.get("activityId") or "")
                tx_ts = str(tx.get("transactionDate") or tx.get("tradeDate") or tx.get("settlementDate") or "")
                key = (
                    tx_id,
                    tx_ts,
                    str(tx.get("type") or tx.get("transactionSubType") or ""),
                    str(tx.get("description") or ""),
                )
                if key in seen:
                    continue
                seen.add(key)
                rows.append(tx)

        cursor = window_end + timedelta(seconds=1)

    return rows


def _to_float(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return default


def _parse_timestamp(v: Any) -> Optional[str]:
    if not v:
        return None
    s = str(v)
    try:
        if s.endswith("Z"):
            return datetime.fromisoformat(s.replace("Z", "+00:00")).astimezone(timezone.utc).isoformat()
        return datetime.fromisoformat(s).astimezone(timezone.utc).isoformat()
    except Exception:
        return None


def _extract_symbol(tx: Dict[str, Any]) -> str:
    if tx.get("symbol"):
        return str(tx.get("symbol"))
    for item in tx.get("transferItems", []) or []:
        inst = (item or {}).get("instrument") or {}
        sym = inst.get("symbol")
        if sym:
            return str(sym)
    return ""


def _normalize_transaction(tx: Dict[str, Any], account_number: str, account_role: str) -> Dict[str, Any]:
    net_amount = _to_float(tx.get("netAmount"), 0.0)
    ttype = str(tx.get("type") or tx.get("transactionSubType") or "UNKNOWN")
    desc = str(tx.get("description") or "")

    symbol = _extract_symbol(tx)
    ts = _parse_timestamp(tx.get("transactionDate") or tx.get("tradeDate") or tx.get("settlementDate"))

    qty = 0.0
    for item in tx.get("transferItems", []) or []:
        qty += _to_float((item or {}).get("amount"), 0.0)

    return {
        "timestamp_utc": ts,
        "account_number": account_number,
        "account_role": account_role,
        "transaction_type": ttype,
        "description": desc,
        "symbol": symbol,
        "quantity": qty,
        "pnl": net_amount,
        "raw": tx,
    }


def _exclude_reason(row: Dict[str, Any], policy: Dict[str, Any]) -> Optional[str]:
    filters = policy.get("mistake_filters", {})

    ts = row.get("timestamp_utc")
    year = None
    if ts:
        try:
            year = datetime.fromisoformat(ts).year
        except Exception:
            year = None

    exclude_year = filters.get("exclude_year")
    if exclude_year is not None and year == int(exclude_year):
        if bool(filters.get("exclude_negative_pnl", True)) and _to_float(row.get("pnl"), 0.0) < 0.0:
            return f"excluded_{exclude_year}_negative_pnl"
        if bool(filters.get("exclude_zero_pnl", False)) and _to_float(row.get("pnl"), 0.0) == 0.0:
            return f"excluded_{exclude_year}_zero_pnl"

    train_filters = policy.get("training_filters", {})
    if bool(train_filters.get("require_symbol", True)) and not str(row.get("symbol") or "").strip():
        return "missing_symbol"

    if abs(_to_float(row.get("pnl"), 0.0)) < float(train_filters.get("min_abs_pnl", 0.0)):
        return "below_min_abs_pnl"

    return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Import Schwab historical transactions and build ML training trade history.")
    parser.add_argument("--years-back", type=int, default=4, help="Number of years to import (default: 4).")
    parser.add_argument("--policy", default=str(PROJECT_ROOT / "config" / "trade_learning_policy.json"))
    parser.add_argument("--out-dir", default=str(PROJECT_ROOT / "data" / "trade_history"))
    parser.add_argument("--skip-auth", action="store_true", help="Use existing normalized file only (for testing pipeline).")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    policy = _load_json(Path(args.policy), {})

    api_key = os.getenv("SCHWAB_API_KEY", "YOUR_KEY_HERE")
    secret = os.getenv("SCHWAB_SECRET", "YOUR_SECRET_HERE")
    redirect = os.getenv("SCHWAB_REDIRECT", "https://127.0.0.1:8080")

    normalized: List[Dict[str, Any]] = []

    if not args.skip_auth:
        if api_key == "YOUR_KEY_HERE" or secret == "YOUR_SECRET_HERE":
            raise RuntimeError("Real SCHWAB_API_KEY and SCHWAB_SECRET are required for live import")

        trader = BaseTrader(api_key, secret, redirect, mode="shadow")
        trader.token_path = str(PROJECT_ROOT / "token.json")
        client = trader.authenticate()

        accounts = _fetch_accounts(client)
        if not accounts:
            raise RuntimeError("No accounts returned by Schwab API")

        end_dt = datetime.now(timezone.utc)
        start_dt = end_dt - timedelta(days=365 * max(args.years_back, 1))

        for acct in accounts:
            acct_num = acct.get("account_number", "")
            acct_hash = acct.get("account_hash", "")
            acct_role = _account_role_from_number(acct_num)
            txs = _fetch_transactions_for_account(client, acct_hash, start_dt, end_dt)
            print(f"[TradeImport] account={acct_num[-4:] if acct_num else 'NONE'} role={acct_role} fetched={len(txs)}")
            for tx in txs:
                normalized.append(_normalize_transaction(tx, acct_num, acct_role))

    normalized_path = out_dir / "trades_normalized.jsonl"
    training_path = out_dir / "trades_training_filtered.jsonl"

    if normalized:
        with normalized_path.open("w", encoding="utf-8") as f:
            for row in normalized:
                f.write(json.dumps(row, ensure_ascii=True) + "\n")
    else:
        existing = []
        if normalized_path.exists():
            with normalized_path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        existing.append(json.loads(line))
        normalized = existing

    kept = []
    excluded = []
    for row in normalized:
        reason = _exclude_reason(row, policy)
        if reason:
            excluded.append({"reason": reason, "row": row})
        else:
            kept.append(row)

    with training_path.open("w", encoding="utf-8") as f:
        for row in kept:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")

    summary = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "years_back": int(args.years_back),
        "normalized_rows": len(normalized),
        "kept_for_training": len(kept),
        "excluded_rows": len(excluded),
        "excluded_reasons": {
            r: sum(1 for x in excluded if x["reason"] == r)
            for r in sorted({x["reason"] for x in excluded})
        },
        "paths": {
            "normalized": str(normalized_path),
            "training_filtered": str(training_path),
        },
    }

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    summary_path = out_dir / f"import_summary_{stamp}.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
