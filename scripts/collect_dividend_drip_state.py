#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


DIVIDEND_DRIP_FEATURE_KEYS = [
    "dividend_drip_active_norm",
    "dividend_drip_recent_reinvest_norm",
    "dividend_drip_cash_only_norm",
    "dividend_drip_share_credit_norm",
    "dividend_drip_event_recency_norm",
    "dividend_drip_confidence_norm",
]

_REINVEST_TOKENS = (
    "reinvest",
    "reinvestment",
    "reinvested",
    "dividend reinvest",
    "dividend reinvestment",
    "drip",
)
_DIVIDEND_TOKENS = (
    "cash dividend",
    "qualified dividend",
    "ordinary dividend",
    "dividend",
    "dividend or interest",
    "dividend/interest",
)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), ensure_ascii=True, indent=2) + "\n", encoding="utf-8")


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    if not math.isfinite(out):
        return default
    return out


def _clamp01(value: float) -> float:
    return max(0.0, min(float(value), 1.0))


def _safe_log_norm(value: float, scale: float) -> float:
    base = max(float(scale), 1e-6)
    return _clamp01(math.log1p(max(float(value), 0.0)) / math.log1p(base))


def _parse_timestamp(raw: Any) -> Optional[datetime]:
    if raw is None:
        return None
    text = str(raw).strip()
    if not text:
        return None
    try:
        dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except Exception:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _extract_symbol(row: Mapping[str, Any]) -> str:
    direct = str(row.get("symbol") or "").strip().upper()
    if direct:
        return direct
    for item in row.get("transferItems", []) or []:
        if not isinstance(item, Mapping):
            continue
        inst = item.get("instrument") if isinstance(item.get("instrument"), Mapping) else {}
        symbol = str(inst.get("symbol") or "").strip().upper()
        if symbol:
            return symbol
    return ""


def _transaction_text(row: Mapping[str, Any]) -> str:
    parts: list[str] = []
    for key in ("type", "transactionSubType", "description"):
        value = str(row.get(key) or "").strip()
        if value:
            parts.append(value)
    for item in row.get("transferItems", []) or []:
        if not isinstance(item, Mapping):
            continue
        desc = str(item.get("description") or "").strip()
        if desc:
            parts.append(desc)
        inst = item.get("instrument") if isinstance(item.get("instrument"), Mapping) else {}
        for key in ("assetType", "description", "symbol"):
            value = str(inst.get(key) or "").strip()
            if value:
                parts.append(value)
    return " | ".join(parts).lower()


def _share_credit(row: Mapping[str, Any]) -> float:
    total = 0.0
    for item in row.get("transferItems", []) or []:
        if not isinstance(item, Mapping):
            continue
        total += abs(_to_float(item.get("amount"), 0.0))
    return total


def _classify_dividend_transaction(row: Mapping[str, Any]) -> Optional[dict[str, Any]]:
    symbol = _extract_symbol(row)
    if not symbol:
        return None

    text = _transaction_text(row)
    contains_reinvest = any(token in text for token in _REINVEST_TOKENS)
    contains_dividend = any(token in text for token in _DIVIDEND_TOKENS)
    if not contains_dividend and not contains_reinvest:
        return None

    cash_amount = abs(_to_float(row.get("netAmount"), 0.0))
    share_credit = _share_credit(row)
    event_type = ""
    confidence = 0.0

    if contains_reinvest or (contains_dividend and share_credit > 0.0 and cash_amount <= 0.05):
        event_type = "drip_reinvest"
        confidence = 0.99 if contains_reinvest else 0.76
    elif contains_dividend:
        event_type = "cash_dividend"
        if "cash dividend" in text or "qualified dividend" in text or "ordinary dividend" in text:
            confidence = 0.96
        else:
            confidence = 0.84
    else:
        return None

    ts = _parse_timestamp(
        row.get("transactionDate")
        or row.get("settlementDate")
        or row.get("tradeDate")
    )
    return {
        "symbol": symbol,
        "event_type": event_type,
        "timestamp_utc": ts.isoformat() if ts is not None else None,
        "cash_amount": cash_amount,
        "share_credit": share_credit if event_type == "drip_reinvest" else 0.0,
        "confidence": _clamp01(confidence),
        "transaction_type": str(row.get("type") or row.get("transactionSubType") or "").strip(),
        "description": str(row.get("description") or "").strip(),
    }


def _fetch_accounts(client: Any) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    if not hasattr(client, "get_account_numbers"):
        return rows
    try:
        response = client.get_account_numbers()
        payload = response.json() if hasattr(response, "json") else response
    except Exception:
        payload = None
    if not isinstance(payload, list):
        return rows
    for item in payload:
        if not isinstance(item, Mapping):
            continue
        account_number = str(item.get("accountNumber") or "").strip()
        account_hash = str(item.get("hashValue") or "").strip()
        if account_hash:
            rows.append({"account_number": account_number, "account_hash": account_hash})
    return rows


def _fetch_transactions_for_account(client: Any, account_hash: str, start_dt: datetime, end_dt: datetime) -> list[dict[str, Any]]:
    if not hasattr(client, "get_transactions"):
        return []

    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str, str]] = set()
    cursor = start_dt
    window_days = 59

    while cursor <= end_dt:
        window_end = min(cursor + timedelta(days=window_days), end_dt)
        attempts = [
            {"account_hash": account_hash, "start_date": cursor, "end_date": window_end},
            {"account_hash": account_hash, "startDate": cursor.isoformat(), "endDate": window_end.isoformat()},
            {"account_hash": account_hash, "start_datetime": cursor, "end_datetime": window_end},
        ]
        payload = None
        for kwargs in attempts:
            try:
                response = client.get_transactions(**kwargs)
                obj = response.json() if hasattr(response, "json") else response
                if isinstance(obj, list):
                    payload = obj
                    break
            except Exception:
                continue
        if isinstance(payload, list):
            for row in payload:
                if not isinstance(row, Mapping):
                    continue
                tx_id = str(row.get("transactionId") or row.get("activityId") or "").strip()
                tx_ts = str(row.get("transactionDate") or row.get("tradeDate") or row.get("settlementDate") or "").strip()
                tx_type = str(row.get("type") or row.get("transactionSubType") or "").strip()
                desc = str(row.get("description") or "").strip()
                key = (tx_id, tx_ts, tx_type, desc)
                if key in seen:
                    continue
                seen.add(key)
                rows.append(dict(row))
        cursor = window_end + timedelta(seconds=1)
    return rows


def _load_transactions_from_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    try:
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except Exception:
                    continue
                if not isinstance(payload, Mapping):
                    continue
                raw = payload.get("raw") if isinstance(payload.get("raw"), Mapping) else payload
                if isinstance(raw, Mapping):
                    merged = dict(raw)
                    if "account_number" not in merged and payload.get("account_number"):
                        merged["account_number"] = payload.get("account_number")
                    if "account_role" not in merged and payload.get("account_role"):
                        merged["account_role"] = payload.get("account_role")
                    rows.append(merged)
    except Exception:
        return []
    return rows


def _zero_feature_row() -> dict[str, float]:
    return {key: 0.0 for key in DIVIDEND_DRIP_FEATURE_KEYS}


def _aggregate_dividend_drip(
    transactions: Iterable[Mapping[str, Any]],
    *,
    now_utc: datetime,
    recent_window_days: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    cutoff = now_utc - timedelta(days=max(int(recent_window_days), 1))
    symbol_rows: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "symbol": "",
            "event_count": 0,
            "cash_dividend_events": 0,
            "drip_reinvest_events": 0,
            "recent_cash_dividend_events": 0,
            "recent_drip_reinvest_events": 0,
            "cash_dividend_amount_total": 0.0,
            "recent_cash_dividend_amount": 0.0,
            "reinvested_shares_total": 0.0,
            "recent_reinvested_shares": 0.0,
            "confidence_sum": 0.0,
            "latest_event_dt": None,
            "latest_cash_dividend_dt": None,
            "latest_drip_reinvest_dt": None,
            "account_numbers": set(),
        }
    )
    recent_events: list[dict[str, Any]] = []
    event_count = 0

    for row in transactions:
        event = _classify_dividend_transaction(row)
        if not event:
            continue
        event_count += 1
        symbol = str(event["symbol"]).upper()
        row_state = symbol_rows[symbol]
        row_state["symbol"] = symbol
        row_state["event_count"] += 1
        row_state["confidence_sum"] += float(event["confidence"])

        account_number = str(row.get("account_number") or "").strip()
        if account_number:
            row_state["account_numbers"].add(account_number[-4:])

        event_dt = _parse_timestamp(event.get("timestamp_utc"))
        if event_dt is not None and (
            row_state["latest_event_dt"] is None or event_dt > row_state["latest_event_dt"]
        ):
            row_state["latest_event_dt"] = event_dt

        if event["event_type"] == "drip_reinvest":
            row_state["drip_reinvest_events"] += 1
            row_state["reinvested_shares_total"] += float(event["share_credit"])
            if event_dt is not None and (
                row_state["latest_drip_reinvest_dt"] is None or event_dt > row_state["latest_drip_reinvest_dt"]
            ):
                row_state["latest_drip_reinvest_dt"] = event_dt
        else:
            row_state["cash_dividend_events"] += 1
            row_state["cash_dividend_amount_total"] += float(event["cash_amount"])
            if event_dt is not None and (
                row_state["latest_cash_dividend_dt"] is None or event_dt > row_state["latest_cash_dividend_dt"]
            ):
                row_state["latest_cash_dividend_dt"] = event_dt

        if event_dt is not None and event_dt >= cutoff:
            recent_events.append(event)
            if event["event_type"] == "drip_reinvest":
                row_state["recent_drip_reinvest_events"] += 1
                row_state["recent_reinvested_shares"] += float(event["share_credit"])
            else:
                row_state["recent_cash_dividend_events"] += 1
                row_state["recent_cash_dividend_amount"] += float(event["cash_amount"])

    symbol_features: dict[str, dict[str, float]] = {}
    symbol_summary: dict[str, dict[str, Any]] = {}

    for symbol, row in symbol_rows.items():
        total_events = max(int(row["event_count"]), 0)
        if total_events <= 0:
            continue
        recent_cash = int(row["recent_cash_dividend_events"])
        recent_reinvest = int(row["recent_drip_reinvest_events"])
        recent_total = recent_cash + recent_reinvest
        historical_reinvest_ratio = float(row["drip_reinvest_events"]) / float(total_events)
        recent_reinvest_ratio = float(recent_reinvest) / float(max(recent_total, 1))
        recent_cash_ratio = float(recent_cash) / float(max(recent_total, 1))
        avg_confidence = float(row["confidence_sum"]) / float(total_events)

        latest_event_dt = row["latest_event_dt"]
        latest_reinvest_dt = row["latest_drip_reinvest_dt"]
        if isinstance(latest_reinvest_dt, datetime):
            recency_target = latest_reinvest_dt
        else:
            recency_target = latest_event_dt

        days_since_event = float(recent_window_days)
        if isinstance(recency_target, datetime):
            days_since_event = max((now_utc - recency_target).total_seconds(), 0.0) / 86400.0

        drip_active_norm = _clamp01(max(recent_reinvest_ratio, historical_reinvest_ratio))
        if recent_reinvest > 0 and recent_reinvest_ratio >= 0.5:
            drip_active_norm = max(drip_active_norm, 0.82)
        recent_reinvest_norm = _clamp01(float(recent_reinvest) / 4.0)
        cash_only_norm = _clamp01(max(recent_cash_ratio, 1.0 if recent_cash > 0 and recent_reinvest == 0 else 0.0))
        share_credit_norm = _safe_log_norm(float(row["recent_reinvested_shares"]) or float(row["reinvested_shares_total"]), 10.0)
        event_recency_norm = _clamp01(1.0 - (days_since_event / float(max(recent_window_days, 1))))
        confidence_norm = _clamp01((0.7 * avg_confidence) + (0.3 * _clamp01(float(total_events) / 6.0)))

        symbol_features[symbol] = {
            "dividend_drip_active_norm": round(drip_active_norm, 6),
            "dividend_drip_recent_reinvest_norm": round(recent_reinvest_norm, 6),
            "dividend_drip_cash_only_norm": round(cash_only_norm, 6),
            "dividend_drip_share_credit_norm": round(share_credit_norm, 6),
            "dividend_drip_event_recency_norm": round(event_recency_norm, 6),
            "dividend_drip_confidence_norm": round(confidence_norm, 6),
        }
        symbol_summary[symbol] = {
            "event_count": total_events,
            "cash_dividend_events": int(row["cash_dividend_events"]),
            "drip_reinvest_events": int(row["drip_reinvest_events"]),
            "recent_cash_dividend_events": recent_cash,
            "recent_drip_reinvest_events": recent_reinvest,
            "cash_dividend_amount_total": round(float(row["cash_dividend_amount_total"]), 6),
            "reinvested_shares_total": round(float(row["reinvested_shares_total"]), 6),
            "recent_reinvested_shares": round(float(row["recent_reinvested_shares"]), 6),
            "account_count": len(row["account_numbers"]),
            "accounts": sorted(x for x in row["account_numbers"] if x),
            "latest_event_ts": latest_event_dt.isoformat() if isinstance(latest_event_dt, datetime) else None,
            "latest_cash_dividend_ts": row["latest_cash_dividend_dt"].isoformat() if isinstance(row["latest_cash_dividend_dt"], datetime) else None,
            "latest_drip_reinvest_ts": latest_reinvest_dt.isoformat() if isinstance(latest_reinvest_dt, datetime) else None,
            "avg_confidence": round(avg_confidence, 6),
        }

    global_features = _zero_feature_row()
    total_weight = 0.0
    for symbol, features in symbol_features.items():
        weight = max(float(symbol_summary.get(symbol, {}).get("event_count", 0) or 0.0), 1.0)
        total_weight += weight
        for key in DIVIDEND_DRIP_FEATURE_KEYS:
            global_features[key] += float(features.get(key, 0.0)) * weight
    if total_weight > 0.0:
        for key in DIVIDEND_DRIP_FEATURE_KEYS:
            global_features[key] = round(global_features[key] / total_weight, 6)

    drip_detected = bool(symbol_features)
    payload = {
        "timestamp_utc": now_utc.isoformat(),
        "ok": True,
        "drip_detected": drip_detected,
        "lookback_days": 0,
        "recent_window_days": int(recent_window_days),
        "derived": {
            "global_features": global_features,
            "symbol_features": symbol_features,
            "summary": {
                "symbols_observed": len(symbol_features),
                "dividend_event_count": event_count,
                "recent_event_count": len(recent_events),
            },
            "symbol_summary": symbol_summary,
        },
    }
    health = {
        "timestamp_utc": now_utc.isoformat(),
        "ok": True,
        "drip_detected": drip_detected,
        "source": "schwab_transactions",
        "symbol_count": len(symbol_features),
        "event_count": event_count,
        "recent_event_count": len(recent_events),
        "warning_count": 0 if symbol_features else 1,
        "warnings": [] if symbol_features else ["no_dividend_drip_events_detected"],
    }
    return payload, health


def collect_dividend_drip_state(
    *,
    lookback_days: int = 400,
    recent_window_days: int = 180,
    input_jsonl: str = "",
    skip_auth: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    now_utc = datetime.now(timezone.utc)
    transactions: list[dict[str, Any]] = []
    accounts: list[dict[str, str]] = []

    if input_jsonl:
        transactions.extend(_load_transactions_from_jsonl(Path(input_jsonl).expanduser()))
    elif skip_auth:
        default_input = PROJECT_ROOT / "data" / "trade_history" / "trades_normalized.jsonl"
        transactions.extend(_load_transactions_from_jsonl(default_input))
    else:
        from core.base_trader import BaseTrader

        api_key = os.getenv("SCHWAB_API_KEY", "YOUR_KEY_HERE")
        secret = os.getenv("SCHWAB_SECRET", "YOUR_SECRET_HERE")
        redirect = os.getenv("SCHWAB_REDIRECT", "https://127.0.0.1:8182")
        if api_key == "YOUR_KEY_HERE" or secret == "YOUR_SECRET_HERE":
            raise RuntimeError("SCHWAB_API_KEY and SCHWAB_SECRET are required for DRIP sync")

        trader = BaseTrader(api_key, secret, redirect, mode="shadow")
        trader.token_path = str(PROJECT_ROOT / "token.json")
        client = trader.authenticate()
        accounts = _fetch_accounts(client)
        end_dt = now_utc
        start_dt = end_dt - timedelta(days=max(int(lookback_days), 1))
        for account in accounts:
            account_number = str(account.get("account_number") or "").strip()
            account_hash = str(account.get("account_hash") or "").strip()
            if not account_hash:
                continue
            for row in _fetch_transactions_for_account(client, account_hash, start_dt, end_dt):
                merged = dict(row)
                if account_number:
                    merged["account_number"] = account_number
                transactions.append(merged)

    payload, health = _aggregate_dividend_drip(
        transactions,
        now_utc=now_utc,
        recent_window_days=max(int(recent_window_days), 1),
    )
    payload["lookback_days"] = int(lookback_days)
    payload["recent_window_days"] = int(recent_window_days)
    payload["source_row_count"] = len(transactions)
    payload["account_count"] = len(accounts)
    health["source_row_count"] = len(transactions)
    health["account_count"] = len(accounts)
    return payload, health


def main() -> int:
    parser = argparse.ArgumentParser(description="Collect real Schwab dividend DRIP state from transaction history.")
    parser.add_argument("--lookback-days", type=int, default=int(os.getenv("DIVIDEND_DRIP_LOOKBACK_DAYS", "400") or 400))
    parser.add_argument("--recent-window-days", type=int, default=int(os.getenv("DIVIDEND_DRIP_RECENT_WINDOW_DAYS", "180") or 180))
    parser.add_argument("--input-jsonl", default="")
    parser.add_argument("--skip-auth", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    payload, health = collect_dividend_drip_state(
        lookback_days=max(int(args.lookback_days), 1),
        recent_window_days=max(int(args.recent_window_days), 1),
        input_jsonl=str(args.input_jsonl or "").strip(),
        skip_auth=bool(args.skip_auth),
    )

    external_path = PROJECT_ROOT / "exports" / "external_context" / "dividend_drip_state_latest.json"
    health_path = PROJECT_ROOT / "governance" / "health" / "dividend_drip_state_sync_latest.json"
    _write_json(external_path, payload)
    _write_json(health_path, health)

    if args.json:
        print(json.dumps(health, ensure_ascii=True))
    else:
        print(f"dividend_drip_state ok={health.get('ok')} symbols={health.get('symbol_count')}")
        print(f"dividend_drip_state_latest={external_path}")
        print(f"dividend_drip_state_sync_latest={health_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
