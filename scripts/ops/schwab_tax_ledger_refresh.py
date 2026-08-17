#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from scripts.brokers.schwab.common import build_schwab_trader, fetch_account_rows, resp_json
    from scripts.ops.long_runtime_common import load_json, write_payload
    from scripts.ops.trading_tax_estimator import _account_tax_treatments
else:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    from scripts.brokers.schwab.common import build_schwab_trader, fetch_account_rows, resp_json
    from .long_runtime_common import load_json, write_payload
    from .trading_tax_estimator import _account_tax_treatments


DEFAULT_PROFILE_PATH = PROJECT_ROOT / "config" / "trading_tax_profile.json"
DEFAULT_ACCOUNT_CONTEXT_PATH = PROJECT_ROOT / "governance" / "health" / "account_policy_context_latest.json"
DEFAULT_STATUS_PATH = PROJECT_ROOT / "governance" / "health" / "schwab_tax_ledger_refresh_latest.json"
WINDOW_DAYS = 59


def _dict(raw: Any) -> dict[str, Any]:
    return raw if isinstance(raw, dict) else {}


def _list(raw: Any) -> list[Any]:
    return raw if isinstance(raw, list) else []


def _number(raw: Any) -> float | None:
    if raw in {None, ""}:
        return None
    try:
        return float(raw)
    except Exception:
        return None


def _first_number(*values: Any) -> float | None:
    for value in values:
        parsed = _number(value)
        if parsed is not None:
            return parsed
    return None


def _timestamp(raw: Any) -> str | None:
    text = str(raw or "").strip()
    if not text:
        return None
    try:
        value = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except Exception:
        return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).isoformat()


def _quiet_auth(trader: Any) -> Any:
    with open(os.devnull, "w", encoding="utf-8") as devnull:
        with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
            return trader.authenticate()


def _fetch_transactions_window(client: Any, account_hash: str, start: datetime, end: datetime) -> tuple[list[dict[str, Any]], str]:
    attempts = [
        {"account_hash": account_hash, "start_date": start, "end_date": end},
        {"account_hash": account_hash, "startDate": start.isoformat(), "endDate": end.isoformat()},
        {"account_hash": account_hash, "start_datetime": start, "end_datetime": end},
    ]
    errors: list[str] = []
    for kwargs in attempts:
        try:
            payload = resp_json(client.get_transactions(**kwargs))
        except Exception as exc:
            errors.append(f"{type(exc).__name__}:{exc}")
            continue
        if isinstance(payload, list):
            return [_dict(row) for row in payload], ""
        errors.append(f"unexpected_payload:{type(payload).__name__}")
    return [], ";".join(errors[-3:]) or "transaction_fetch_failed"


def fetch_year_transactions(
    client: Any,
    account_hash: str,
    *,
    start: datetime,
    end: datetime,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    cursor = start
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    seen: set[str] = set()
    while cursor <= end:
        window_end = min(cursor + timedelta(days=WINDOW_DAYS), end)
        fetched, error = _fetch_transactions_window(client, account_hash, cursor, window_end)
        if error:
            failures.append({
                "start": cursor.isoformat(),
                "end": window_end.isoformat(),
                "error": error,
            })
        for row in fetched:
            identity = json.dumps(
                {
                    "id": row.get("transactionId") or row.get("activityId"),
                    "date": row.get("transactionDate") or row.get("tradeDate") or row.get("settlementDate"),
                    "type": row.get("type") or row.get("transactionSubType"),
                    "net": row.get("netAmount"),
                    "description": row.get("description"),
                },
                sort_keys=True,
                default=str,
            )
            key = hashlib.sha256(identity.encode("utf-8")).hexdigest()
            if key not in seen:
                seen.add(key)
                rows.append(row)
        cursor = window_end + timedelta(seconds=1)
    return rows, failures


def _instrument(item: dict[str, Any]) -> dict[str, Any]:
    return _dict(item.get("instrument"))


def _symbol(tx: dict[str, Any], item: dict[str, Any] | None = None) -> str:
    if item:
        symbol = str(_instrument(item).get("symbol") or item.get("symbol") or "").strip().upper()
        if symbol:
            return symbol
    direct = str(tx.get("symbol") or "").strip().upper()
    if direct:
        return direct
    for candidate in _list(tx.get("transferItems")):
        symbol = str(_instrument(_dict(candidate)).get("symbol") or "").strip().upper()
        if symbol:
            return symbol
    return ""


def _action(tx: dict[str, Any], item: dict[str, Any] | None = None) -> str:
    values = []
    if item:
        values.extend([item.get("instruction"), item.get("direction"), item.get("positionEffect")])
    values.extend([tx.get("instruction"), tx.get("transactionSubType"), tx.get("description")])
    text = " ".join(str(value or "") for value in values).upper()
    if any(token in text for token in ("SELL", "SOLD", "SALE", "CLOSING SALE")):
        return "SELL"
    if any(token in text for token in ("BUY", "BOUGHT", "PURCHASE", "REINVEST")):
        return "BUY"
    if str(tx.get("type") or "").strip().upper() == "TRADE" and item:
        amount = _first_number(item.get("amount"), item.get("quantity"))
        if amount is not None and amount > 0.0:
            return "BUY"
        if amount is not None and amount < 0.0:
            return "SELL"
    return "UNKNOWN"


def _kind(tx: dict[str, Any], *, action: str, item: dict[str, Any] | None = None) -> str:
    raw_type = str(tx.get("type") or "").upper()
    subtype = str(tx.get("transactionSubType") or "").upper()
    description = str(tx.get("description") or "").upper()
    text = f"{raw_type} {subtype} {description}"
    if raw_type == "DIVIDEND_OR_INTEREST":
        if "INTEREST" in description or description.startswith("BANK INT"):
            net_amount = _number(tx.get("netAmount"))
            return "investment_interest_expense" if net_amount is not None and net_amount < 0.0 else "interest"
        return "dividend"
    if "INTEREST" in text:
        return "interest"
    if "DIVIDEND" in text:
        return "dividend"
    if raw_type == "TRADE":
        inst = _instrument(item or {})
        asset_type = str(inst.get("assetType") or inst.get("assetTypeCode") or "").strip().upper()
        effect = str(_dict(item).get("positionEffect") or "").strip().upper()
        if asset_type == "OPTION" and effect == "OPENING":
            return "acquisition"
        if asset_type == "OPTION" and effect == "CLOSING":
            return "capital_disposition"
        if action == "BUY":
            return "acquisition"
        if action == "SELL":
            return "capital_disposition"
    if action == "BUY":
        return "acquisition"
    if action == "SELL":
        return "capital_disposition"
    if raw_type in {"JOURNAL", "TRANSFER", "DEPOSIT", "WITHDRAWAL", "WIRE", "ACH"}:
        return "transfer"
    if "FEE" in text or "COMMISSION" in text:
        return "fee"
    return "unknown"


def _recursive_number(payload: Any, keys: set[str]) -> float | None:
    if isinstance(payload, dict):
        for key, value in payload.items():
            if str(key).lower() in keys:
                parsed = _number(value)
                if parsed is not None:
                    return parsed
        for value in payload.values():
            parsed = _recursive_number(value, keys)
            if parsed is not None:
                return parsed
    if isinstance(payload, list):
        for value in payload:
            parsed = _recursive_number(value, keys)
            if parsed is not None:
                return parsed
    return None


def _event_id(tx: dict[str, Any], account_label: str, item_index: int) -> str:
    identity = "|".join(
        [
            account_label,
            str(tx.get("transactionId") or tx.get("activityId") or ""),
            str(tx.get("transactionDate") or tx.get("tradeDate") or tx.get("settlementDate") or ""),
            str(item_index),
            str(tx.get("type") or ""),
            str(tx.get("description") or ""),
        ]
    )
    return f"schwab_{hashlib.sha256(identity.encode('utf-8')).hexdigest()[:24]}"


def _safe_description(raw: Any) -> str:
    text = " ".join(str(raw or "").split())
    return text[:240]


def normalize_transaction(
    tx: dict[str, Any],
    *,
    account_label: str,
    tax_treatment: str,
) -> list[dict[str, Any]]:
    items = [_dict(row) for row in _list(tx.get("transferItems"))]
    if str(tx.get("type") or "").strip().upper() == "TRADE":
        security_items = [
            item for item in items
            if str(_instrument(item).get("assetType") or _instrument(item).get("assetTypeCode") or "").strip().upper()
            not in {"", "CURRENCY"}
        ]
        rows = security_items or items or [{}]
    else:
        rows = items or [{}]
    event_time = _timestamp(tx.get("transactionDate") or tx.get("tradeDate") or tx.get("settlementDate"))
    net_amount = _number(tx.get("netAmount"))
    events: list[dict[str, Any]] = []
    for index, item in enumerate(rows):
        action = _action(tx, item)
        kind = _kind(tx, action=action, item=item)
        inst = _instrument(item)
        quantity = _first_number(item.get("amount"), item.get("quantity"))
        price = _first_number(item.get("price"), tx.get("price"))
        event: dict[str, Any] = {
            "event_id": _event_id(tx, account_label, index),
            "environment": "actual",
            "source": "schwab_transactions_api",
            "account_label": account_label,
            "tax_treatment": tax_treatment,
            "tax_event_kind": kind,
            "action": action,
            "timestamp_utc": event_time,
            "transaction_date": event_time,
            "symbol": "" if kind in {"dividend", "interest", "investment_interest_expense"} else _symbol(tx, item),
            "asset_type": str(inst.get("assetType") or inst.get("assetTypeCode") or "").strip().upper(),
            "quantity": quantity,
            "price_usd": price,
            "description": _safe_description(tx.get("description")),
            "broker_evidence": {
                "type": str(tx.get("type") or ""),
                "transaction_subtype": str(tx.get("transactionSubType") or ""),
                "status": str(tx.get("status") or ""),
                "settlement_date": _timestamp(tx.get("settlementDate")),
                "position_effect": str(item.get("positionEffect") or ""),
                "instruction": str(item.get("instruction") or ""),
                "option_put_call": str(inst.get("putCall") or ""),
                "option_underlying": str(inst.get("underlyingSymbol") or ""),
            },
        }
        if kind in {"dividend", "interest"} and net_amount is not None:
            event["amount_usd"] = abs(net_amount)
            event["tax_amount_provisional"] = True
            event["amount_evidence"] = "broker_net_cash_pending_tax_form_reconciliation"
        elif kind == "investment_interest_expense" and net_amount is not None:
            event["amount_usd"] = abs(net_amount)
            event["tax_amount_provisional"] = True
            event["amount_evidence"] = "broker_margin_interest_pending_form_4952_review"
        elif kind == "capital_disposition":
            realized = _recursive_number(tx, {"realizedgainloss", "realizedpnl", "gainloss"})
            basis = _recursive_number(tx, {"adjustedcostbasis", "costbasis"})
            proceeds = _recursive_number(tx, {"proceeds", "saleproceeds"})
            if proceeds is None and net_amount is not None and net_amount > 0.0:
                proceeds = net_amount
                event["proceeds_evidence"] = "broker_net_cash_provisional"
            if realized is not None:
                event["realized_gain_loss_usd"] = realized
                event["gain_evidence"] = "broker_reported_realized_gain_loss"
            if basis is not None:
                event["adjusted_cost_basis_usd"] = basis
            if proceeds is not None:
                event["proceeds_usd"] = proceeds
            event["wash_sale_status"] = "unknown"
            event["section_1256_verified"] = False
        elif kind == "acquisition":
            event["realization_status"] = "not_realized"
        events.append(event)
    return events


def refresh(
    *,
    tax_year: int,
    profile_path: Path,
    account_context_path: Path,
    out_path: Path,
) -> dict[str, Any]:
    now = datetime.now(timezone.utc)
    start = datetime(int(tax_year), 1, 1, tzinfo=timezone.utc)
    end = min(now, datetime(int(tax_year), 12, 31, 23, 59, 59, tzinfo=timezone.utc))
    if start > end:
        return {
            "timestamp_utc": now.isoformat(),
            "ok": False,
            "status": "future_tax_year_not_fetchable",
            "tax_year": int(tax_year),
        }

    profile = load_json(profile_path)
    account_context = load_json(account_context_path)
    treatments = _account_tax_treatments(profile, account_context)
    old_env = {
        "ALLOW_ORDER_EXECUTION": os.environ.get("ALLOW_ORDER_EXECUTION"),
        "MARKET_DATA_ONLY": os.environ.get("MARKET_DATA_ONLY"),
        "SCHWAB_AUTH_INTERACTIVE": os.environ.get("SCHWAB_AUTH_INTERACTIVE"),
    }
    os.environ["ALLOW_ORDER_EXECUTION"] = "0"
    os.environ["MARKET_DATA_ONLY"] = "1"
    os.environ["SCHWAB_AUTH_INTERACTIVE"] = "0"
    events: list[dict[str, Any]] = []
    account_rows: list[dict[str, Any]] = []
    try:
        trader = build_schwab_trader(
            PROJECT_ROOT,
            mode="shadow",
            missing_credentials_message="Schwab credentials are required for tax-ledger refresh",
        )
        client = _quiet_auth(trader)
        accounts = fetch_account_rows(client)
        if not accounts:
            raise RuntimeError("no_schwab_accounts_returned")
        for index, account in enumerate(accounts):
            account_tail = str(account.get("account_number") or "").strip()[-4:]
            account_label = f"account_{index + 1}_{account_tail}" if account_tail else f"account_{index + 1}"
            treatment = treatments.get(account_label, "unknown")
            rows, failures = fetch_year_transactions(
                client,
                str(account.get("account_hash") or ""),
                start=start,
                end=end,
            )
            normalized: list[dict[str, Any]] = []
            for tx in rows:
                normalized.extend(normalize_transaction(tx, account_label=account_label, tax_treatment=treatment))
            events.extend(normalized)
            account_rows.append({
                "account_label": account_label,
                "tax_treatment": treatment,
                "transaction_count": len(rows),
                "event_count": len(normalized),
                "failed_windows": failures,
                "complete": not failures,
            })
    except Exception as exc:
        account_rows.append({"complete": False, "error": f"{type(exc).__name__}:{exc}"})
    finally:
        for key, value in old_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    complete = bool(account_rows) and all(bool(row.get("complete", False)) for row in account_rows)
    ledger = {
        "timestamp_utc": now.isoformat(),
        "schema_version": 1,
        "tax_year": int(tax_year),
        "coverage": {
            "coverage_start": start.isoformat(),
            "coverage_end": end.isoformat(),
            "complete_for_tax_year": complete,
            "all_relevant_accounts_included": complete,
            "source": "schwab_transactions_api",
            "account_count": sum(1 for row in account_rows if row.get("account_label")),
        },
        "events": events,
        "account_fetches": account_rows,
        "redaction_contract": {
            "full_account_numbers_stored": False,
            "redacted_account_suffix_labels_stored": True,
            "account_hashes_stored": False,
            "broker_transaction_ids_stored": False,
            "stable_hashed_event_ids_only": True,
        },
        "limitations": [
            "Schwab transaction activity is not a substitute for broker tax forms or a closed-lot gains report.",
            "Broker net cash for dividends and interest remains provisional until 1099 reconciliation.",
            "Sale proceeds are never treated as realized profit without cost-basis evidence.",
        ],
    }
    if complete:
        write_payload(out_path, ledger)
        status = "ready"
    else:
        partial_path = out_path.with_name(f"{out_path.stem}_partial_{now.strftime('%Y%m%dT%H%M%SZ')}{out_path.suffix}")
        write_payload(partial_path, ledger)
        status = "refresh_failed_previous_good_preserved"
    summary = {
        "timestamp_utc": now.isoformat(),
        "ok": complete,
        "status": status,
        "tax_year": int(tax_year),
        "ledger_path": str(out_path),
        "event_count": len(events),
        "account_count": sum(1 for row in account_rows if row.get("account_label")),
        "complete_account_count": sum(1 for row in account_rows if row.get("complete")),
        "accounts": account_rows,
        "order_execution_forced_off": True,
        "market_data_only_forced_on": True,
    }
    write_payload(DEFAULT_STATUS_PATH, summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    year = datetime.now(timezone.utc).year
    parser = argparse.ArgumentParser(description="Refresh a redacted Schwab year-to-date tax evidence ledger.")
    parser.add_argument("--tax-year", type=int, default=year)
    parser.add_argument("--profile", default=str(DEFAULT_PROFILE_PATH))
    parser.add_argument("--account-context", default=str(DEFAULT_ACCOUNT_CONTEXT_PATH))
    parser.add_argument("--out-file", default="")
    parser.add_argument("--max-age-seconds", type=float, default=21600.0)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    out_path = (
        Path(args.out_file).expanduser()
        if str(args.out_file or "").strip()
        else PROJECT_ROOT / "governance" / "tax" / f"trading_tax_ledger_{int(args.tax_year)}_latest.json"
    )
    existing = load_json(out_path)
    existing_time = _timestamp(existing.get("timestamp_utc"))
    existing_dt = datetime.fromisoformat(existing_time) if existing_time else None
    age_seconds = (
        max((datetime.now(timezone.utc) - existing_dt).total_seconds(), 0.0)
        if existing_dt is not None
        else None
    )
    coverage = _dict(existing.get("coverage"))
    if (
        not args.force
        and age_seconds is not None
        and age_seconds <= max(float(args.max_age_seconds), 0.0)
        and bool(coverage.get("complete_for_tax_year", False))
        and bool(coverage.get("all_relevant_accounts_included", False))
    ):
        payload = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "ok": True,
            "status": "cached_ready",
            "tax_year": int(args.tax_year),
            "ledger_path": str(out_path),
            "ledger_age_seconds": round(age_seconds, 3),
            "event_count": len(_list(existing.get("events"))),
            "account_count": int(_number(coverage.get("account_count")) or 0),
            "order_execution_forced_off": True,
            "market_data_only_forced_on": True,
        }
        write_payload(DEFAULT_STATUS_PATH, payload)
        if args.json:
            print(json.dumps(payload, ensure_ascii=True))
        else:
            print(
                "schwab_tax_ledger_refresh "
                f"status={payload.get('status')} "
                f"tax_year={payload.get('tax_year')} "
                f"events={payload.get('event_count', 0)}"
            )
        return 0
    payload = refresh(
        tax_year=int(args.tax_year),
        profile_path=Path(args.profile).expanduser(),
        account_context_path=Path(args.account_context).expanduser(),
        out_path=out_path,
    )
    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        print(
            "schwab_tax_ledger_refresh "
            f"status={payload.get('status')} "
            f"tax_year={payload.get('tax_year')} "
            f"events={payload.get('event_count', 0)}"
        )
    return 0 if bool(payload.get("ok", False)) else 2


if __name__ == "__main__":
    raise SystemExit(main())
