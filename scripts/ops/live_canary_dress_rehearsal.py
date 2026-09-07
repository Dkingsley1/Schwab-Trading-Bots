#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
import hashlib
import hmac
import json
import math
import os
import subprocess
import sys
from datetime import datetime, timezone
from decimal import Decimal, ROUND_CEILING
from pathlib import Path
from typing import Any, Mapping

if __package__ in {None, ""}:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
else:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]

from core.accountability import safe_write_json_atomic
from core.live_canary_preflight import evaluate_live_canary_preflight
from core.live_execution_envelope import (
    build_live_execution_envelope,
    file_sha256,
    verify_live_execution_envelope,
)
from core.order_intent import build_order_intent_evidence, canonical_payload_sha256
from scripts.brokers.schwab.common import build_schwab_trader
from scripts.ops.schwab_account_hash_keychain_sync import (
    _keychain_account,
    _read_keychain_secret,
    _service_name,
)
from scripts.ops.schwab_account_snapshot_refresh import _quiet_auth, refresh

DEFAULT_OUT_PATH = (
    PROJECT_ROOT / "governance" / "health" / "live_canary_dress_rehearsal_latest.json"
)
DEFAULT_ACCOUNT_STUDY_PATH = (
    PROJECT_ROOT / "governance" / "health" / "account_position_study_latest.json"
)
READ_ONLY_ENVIRONMENT = {
    "ALLOW_ORDER_EXECUTION": "0",
    "MARKET_DATA_ONLY": "1",
    "TOP_BOT_ENABLE_LIVE_EXECUTION": "0",
    "EXECUTION_LANE_LIVE_ENABLED": "0",
    "RUN_ALL_SLEEVES_WITH_LIVE_EXECUTOR": "0",
}
PROVIDER_TIME_FIELDS = (
    "quoteTime",
    "askTime",
    "bidTime",
    "tradeTime",
    "lastTradeTime",
    "lastTradeTimestamp",
    "regularMarketTradeTime",
)
VENUE_FIELDS = ("askMICId", "bidMICId", "lastMICId", "exchangeName")
DERIVED_ENVELOPE_BLOCKERS = frozenset(
    {
        "order_intent_risk_decision_not_approved",
        "live_canary_preflight_receipt_not_ready",
        "quote_is_stale",
    }
)


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float(default)
    return number if math.isfinite(number) else float(default)


def _parse_timestamp(value: Any) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _provider_timestamp(value: Any) -> datetime | None:
    if isinstance(value, bool) or value is None or value == "":
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return _parse_timestamp(value)
    if not math.isfinite(number) or number <= 0.0:
        return None
    if number >= 1e15:
        number /= 1_000_000.0
    elif number >= 1e12:
        number /= 1_000.0
    elif number < 1e9:
        return None
    try:
        return datetime.fromtimestamp(number, tz=timezone.utc)
    except (OverflowError, OSError, ValueError):
        return None


def _mapping_nodes(payload: Any, *, max_depth: int = 5) -> list[Mapping[str, Any]]:
    nodes: list[Mapping[str, Any]] = []

    def visit(value: Any, depth: int) -> None:
        if depth > max_depth:
            return
        if isinstance(value, Mapping):
            nodes.append(value)
            for child in value.values():
                if isinstance(child, (Mapping, list, tuple)):
                    visit(child, depth + 1)
        elif isinstance(value, (list, tuple)):
            for child in value:
                if isinstance(child, (Mapping, list, tuple)):
                    visit(child, depth + 1)

    visit(payload, 0)
    return nodes


def _first_nested(payload: Any, fields: tuple[str, ...]) -> Any:
    for node in _mapping_nodes(payload):
        for field in fields:
            value = node.get(field)
            if field in node and value is not None and value != "":
                return value
    return None


def _latest_provider_time(payload: Any) -> datetime | None:
    timestamps: list[datetime] = []
    for node in _mapping_nodes(payload):
        for field in PROVIDER_TIME_FIELDS:
            if field not in node:
                continue
            parsed = _provider_timestamp(node.get(field))
            if parsed is not None:
                timestamps.append(parsed)
    return max(timestamps) if timestamps else None


def _sha256_text(value: Any) -> str:
    return hashlib.sha256(str(value or "").encode("utf-8")).hexdigest()


def _payload_sha256(payload: Any) -> str:
    try:
        encoded = json.dumps(
            payload,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
    except Exception:
        return ""
    return hashlib.sha256(encoded).hexdigest()


def _ceil_cent(value: float) -> float:
    if value <= 0.0 or not math.isfinite(value):
        return 0.0
    return float(Decimal(str(value)).quantize(Decimal("0.01"), rounding=ROUND_CEILING))


def _ordered_unique(values: list[str]) -> list[str]:
    return list(dict.fromkeys(item for item in values if item))


def _quote_summary(
    quote_result: Mapping[str, Any], *, symbol: str, now: datetime
) -> dict[str, Any]:
    snapshot = (
        quote_result.get("quote_snapshot")
        if isinstance(quote_result.get("quote_snapshot"), Mapping)
        else {}
    )
    raw = snapshot.get("raw_payload")
    if not isinstance(raw, Mapping):
        raw = quote_result.get("payload")
    raw = raw if isinstance(raw, Mapping) else {}
    bid = _safe_float(snapshot.get("bid_price"), 0.0)
    ask = _safe_float(snapshot.get("ask_price"), 0.0)
    last = _safe_float(snapshot.get("last_price"), 0.0)
    mark = _safe_float(snapshot.get("mark_price"), 0.0)
    midpoint = (bid + ask) / 2.0 if bid > 0.0 and ask >= bid else 0.0
    spread_bps = ((ask - bid) / midpoint) * 10000.0 if midpoint > 0.0 else 0.0
    provider_time = _latest_provider_time(raw)
    age_seconds = (
        max((now - provider_time).total_seconds(), 0.0)
        if provider_time is not None
        else None
    )
    realtime_value = _first_nested(raw, ("realtime", "isRealtime"))
    realtime = realtime_value is True or str(realtime_value or "").lower() == "true"
    venue = str(_first_nested(raw, VENUE_FIELDS) or "").strip()
    provider_payload_sha256 = _payload_sha256(raw)
    compact = {
        "symbol": str(symbol or "").strip().upper(),
        "bid_price": round(bid, 6),
        "ask_price": round(ask, 6),
        "last_price": round(last, 6),
        "mark_price": round(mark, 6),
        "spread_bps": round(spread_bps, 6),
        "provider_timestamp_utc": (
            provider_time.isoformat() if provider_time is not None else ""
        ),
        "quote_age_seconds": (
            round(age_seconds, 6) if age_seconds is not None else None
        ),
        "realtime": bool(realtime),
        "source_provider": "schwab_api" if quote_result.get("ok") else "",
        "source_venue": venue,
        "provider_payload_sha256": provider_payload_sha256,
        "transport": {
            "ok": bool(quote_result.get("ok", False)),
            "operation": str(quote_result.get("operation") or "get_quote"),
            "method": str(quote_result.get("method") or ""),
            "status_code": int(_safe_float(quote_result.get("status_code"), 0.0)),
            "latency_ms": round(_safe_float(quote_result.get("latency_ms"), 0.0), 3),
            "received_at_utc": now.isoformat(),
        },
    }
    compact["snapshot_id"] = canonical_payload_sha256(compact)
    return compact


def _account_context(
    account_study: Mapping[str, Any],
    *,
    account_policy_key: str,
    symbol: str,
    canary_symbols: set[str],
) -> dict[str, Any]:
    accounts = (
        account_study.get("accounts")
        if isinstance(account_study.get("accounts"), list)
        else []
    )
    account = next(
        (
            dict(row)
            for row in accounts
            if isinstance(row, Mapping)
            and str(row.get("account_policy_key") or "").strip() == account_policy_key
        ),
        {},
    )
    positions = [
        dict(row)
        for row in (
            account_study.get("positions")
            if isinstance(account_study.get("positions"), list)
            else []
        )
        if isinstance(row, Mapping)
        and str(row.get("account_policy_key") or "").strip() == account_policy_key
    ]
    capability = (
        account.get("account_capability_truth")
        if isinstance(account.get("account_capability_truth"), Mapping)
        else {}
    )
    balance = (
        capability.get("balance_truth")
        if isinstance(capability.get("balance_truth"), Mapping)
        else {}
    )
    debit = (
        capability.get("debit_truth")
        if isinstance(capability.get("debit_truth"), Mapping)
        else {}
    )
    calls = (
        capability.get("broker_call_truth")
        if isinstance(capability.get("broker_call_truth"), Mapping)
        else {}
    )
    collateral = (
        capability.get("position_collateral_truth")
        if isinstance(capability.get("position_collateral_truth"), Mapping)
        else {}
    )
    canary = (
        account.get("canary_preflight")
        if isinstance(account.get("canary_preflight"), Mapping)
        else {}
    )
    flags = account.get("flags") if isinstance(account.get("flags"), Mapping) else {}
    operator_classification = (
        capability.get("operator_classification")
        if isinstance(capability.get("operator_classification"), Mapping)
        else {}
    )
    account_kind = (
        str(
            operator_classification.get("account_kind")
            or account.get("operator_account_kind")
            or "unknown"
        )
        .strip()
        .lower()
    )
    tax_wrapper = (
        str(
            operator_classification.get("tax_wrapper")
            or account.get("tax_wrapper")
            or "unknown"
        )
        .strip()
        .lower()
    )

    equity_by_underlying: dict[str, float] = {}
    short_contracts_by_underlying: dict[str, float] = {}
    symbol_quantity = 0.0
    canary_position_symbols: set[str] = set()
    for row in positions:
        row_symbol = str(row.get("symbol") or "").strip().upper()
        underlying = str(row.get("underlying") or row_symbol).strip().upper()
        asset_type = str(row.get("asset_type") or "").strip().upper()
        quantity = _safe_float(row.get("quantity"), 0.0)
        if asset_type == "EQUITY":
            equity_by_underlying[underlying] = (
                equity_by_underlying.get(underlying, 0.0) + quantity
            )
            if row_symbol == symbol:
                symbol_quantity += quantity
            if row_symbol in canary_symbols and quantity > 0.0:
                canary_position_symbols.add(row_symbol)
        elif asset_type == "OPTION":
            short_contracts = max(
                _safe_float(row.get("short_quantity"), 0.0), -quantity, 0.0
            )
            if short_contracts > 0.0:
                short_contracts_by_underlying[underlying] = (
                    short_contracts_by_underlying.get(underlying, 0.0) + short_contracts
                )

    collateral_rows: list[dict[str, Any]] = []
    for underlying in sorted(
        set(equity_by_underlying) | set(short_contracts_by_underlying)
    ):
        equity_quantity = equity_by_underlying.get(underlying, 0.0)
        short_contracts = short_contracts_by_underlying.get(underlying, 0.0)
        reserved_shares = short_contracts * 100.0
        collateral_rows.append(
            {
                "underlying": underlying,
                "equity_quantity": round(equity_quantity, 6),
                "covered_short_contracts": round(short_contracts, 6),
                "reserved_equity_shares": round(reserved_shares, 6),
                "unencumbered_equity_shares": round(
                    max(equity_quantity - reserved_shares, 0.0), 6
                ),
            }
        )

    cash_available = max(
        _safe_float(balance.get("cash_available_for_trading"), 0.0),
        _safe_float(balance.get("cash_balance"), account.get("cash_balance", 0.0)),
    )
    return {
        "account_found": bool(account),
        "account_policy_key": account_policy_key,
        "account_kind": account_kind,
        "tax_wrapper": tax_wrapper,
        "retirement_account": tax_wrapper
        in {"ira", "roth", "roth_ira", "traditional_ira"},
        "trading_access": str(
            operator_classification.get("trading_access")
            or account.get("operator_trading_type")
            or "unknown"
        ),
        "borrowing_allowed": bool(account.get("borrowing_allowed", False)),
        "closing_only": bool(flags.get("closing_only", False)),
        "broker_call_present": bool(calls.get("in_call", False)),
        "settled_cash_broker_visible_usd": round(cash_available, 4),
        "pending_deposits_usd": round(
            _safe_float(balance.get("pending_deposits"), 0.0), 4
        ),
        "candidate_symbol_quantity": round(symbol_quantity, 6),
        "existing_canary_position_symbols": sorted(canary_position_symbols),
        "existing_canary_position_count": len(canary_position_symbols),
        "provider_balance_diagnostic": {
            "provider_margin_balance": round(
                _safe_float(debit.get("provider_margin_balance"), 0.0), 4
            ),
            "status": str(debit.get("status") or "unknown"),
            "interest_bearing_borrowing_confirmed": bool(
                debit.get("interest_bearing_borrowing_confirmed", False)
            ),
            "included_in_settled_cash": False,
            "classified_as_debt": False,
            "requires_broker_ui_confirmation": bool(
                debit.get("requires_broker_ui_confirmation", False)
            ),
        },
        "covered_position_safety": {
            "covered_short_option_count": int(
                _safe_float(collateral.get("covered_short_option_count"), 0.0)
            ),
            "uncovered_short_option_count": int(
                _safe_float(collateral.get("uncovered_short_option_count"), 0.0)
            ),
            "candidate_order_reduces_existing_collateral": False,
            "coverage_unchanged_by_candidate_buy": True,
            "collateral_by_underlying": collateral_rows,
        },
        "account_preflight_blockers": [
            str(item or "").strip()
            for item in (
                canary.get("blockers")
                if isinstance(canary.get("blockers"), list)
                else []
            )
            if str(item or "").strip()
        ],
    }


def _candidate_symbols(plan: Mapping[str, Any]) -> set[str]:
    symbols: set[str] = set()
    for stage in plan.get("stages", []):
        if not isinstance(stage, Mapping):
            continue
        symbols.update(
            str(item or "").strip().upper()
            for item in stage.get("symbols", [])
            if str(item or "").strip()
        )
    return symbols


def build_dress_rehearsal_payload(
    *,
    now: datetime,
    plan: Mapping[str, Any],
    firewall: Mapping[str, Any],
    candidate: Mapping[str, Any],
    account_study: Mapping[str, Any],
    account_study_sha256: str,
    account_reference: str,
    expected_account_reference: str,
    quote_result: Mapping[str, Any],
    preflight_receipt: Mapping[str, Any],
    account_refresh_summary: Mapping[str, Any],
    policy_sha256: str,
    symbol: str = "SCHD",
) -> dict[str, Any]:
    current = now.astimezone(timezone.utc)
    symbol_key = str(symbol or "SCHD").strip().upper()
    account_policy_key = str(plan.get("account_policy_key") or "").strip()
    route_id = str(plan.get("execution_route_id") or "").strip()
    candidate_id = str(candidate.get("candidate_id") or "").strip()
    hard_limits = (
        plan.get("hard_limits") if isinstance(plan.get("hard_limits"), Mapping) else {}
    )
    max_quantity = max(_safe_float(hard_limits.get("max_order_quantity"), 0.0), 0.0)
    max_notional = max(_safe_float(hard_limits.get("max_order_notional_usd"), 0.0), 0.0)
    account_capital = max(_safe_float(plan.get("account_capital_usd"), 0.0), 0.0)
    account_constraints = (
        plan.get("account_constraints")
        if isinstance(plan.get("account_constraints"), Mapping)
        else {}
    )
    quantity = 1.0
    max_quote_age = max(_safe_float(firewall.get("max_quote_age_seconds"), 15.0), 0.001)
    max_account_age = max(
        _safe_float(firewall.get("max_account_snapshot_age_seconds"), 30.0),
        0.001,
    )
    max_spread = max(_safe_float(firewall.get("max_spread_bps"), 75.0), 0.001)
    max_future_skew = max(
        _safe_float(firewall.get("max_future_clock_skew_seconds"), 2.0), 0.0
    )
    quote = _quote_summary(quote_result, symbol=symbol_key, now=current)
    limit_price = _ceil_cent(_safe_float(quote.get("ask_price"), 0.0))
    order_notional = round(limit_price * quantity, 4)
    canary_symbols = _candidate_symbols(plan)
    account = _account_context(
        account_study,
        account_policy_key=account_policy_key,
        symbol=symbol_key,
        canary_symbols=canary_symbols,
    )
    account_time = _parse_timestamp(
        account_study.get("timestamp_utc") or account_study.get("generated_at_utc")
    )
    account_age = (
        max((current - account_time).total_seconds(), 0.0)
        if account_time is not None
        else None
    )
    cash_available = _safe_float(account.get("settled_cash_broker_visible_usd"), 0.0)
    pending_deposits = _safe_float(account.get("pending_deposits_usd"), 0.0)
    provider_time = _parse_timestamp(quote.get("provider_timestamp_utc"))
    quote_age = quote.get("quote_age_seconds")
    spread_bps = _safe_float(quote.get("spread_bps"), 0.0)
    account_reference_matches = bool(
        account_reference
        and expected_account_reference
        and hmac.compare_digest(account_reference, expected_account_reference)
    )

    blockers: list[str] = []
    if not bool(account_refresh_summary.get("ok", False)):
        blockers.append("connected_account_snapshot_refresh_failed")
    if bool(account_refresh_summary.get("skipped", False)):
        blockers.append("connected_account_snapshot_not_refreshed_this_run")
    if not candidate_id:
        blockers.append("production_candidate_id_missing")
    if not account_policy_key:
        blockers.append("canary_account_policy_key_missing")
    if not route_id:
        blockers.append("canary_execution_route_missing")
    if not account_reference or not expected_account_reference:
        blockers.append("designated_canary_account_hash_unavailable")
    elif not account_reference_matches:
        blockers.append("live_account_not_designated_canary_account")
    if not account.get("account_found", False):
        blockers.append("designated_canary_account_truth_missing")
    required_account_kind = (
        str(account_constraints.get("required_account_kind") or "").strip().lower()
    )
    required_tax_wrapper = (
        str(account_constraints.get("required_tax_wrapper") or "").strip().lower()
    )
    if (
        required_account_kind
        and str(account.get("account_kind") or "").lower() != required_account_kind
    ):
        blockers.append("canary_account_kind_mismatch")
    if (
        required_tax_wrapper
        and str(account.get("tax_wrapper") or "").lower() != required_tax_wrapper
    ):
        blockers.append("canary_tax_wrapper_mismatch")
    if account_age is None:
        blockers.append("account_position_study_timestamp_missing")
    elif account_age > max_account_age:
        blockers.append("account_position_study_stale")
    if not quote.get("transport", {}).get("ok", False):
        blockers.append("schwab_quote_fetch_failed")
    bid = _safe_float(quote.get("bid_price"), 0.0)
    ask = _safe_float(quote.get("ask_price"), 0.0)
    if bid <= 0.0 or ask <= 0.0 or ask < bid:
        blockers.append("schwab_quote_bid_ask_invalid")
    if provider_time is None or quote_age is None:
        blockers.append("schwab_quote_provider_timestamp_missing")
    else:
        raw_age = (current - provider_time).total_seconds()
        if raw_age < -max_future_skew:
            blockers.append("schwab_quote_timestamp_in_future")
        elif _safe_float(quote_age, 0.0) > max_quote_age:
            blockers.append("schwab_quote_stale")
    if not bool(quote.get("realtime", False)):
        blockers.append("schwab_quote_not_realtime")
    if spread_bps <= 0.0 or spread_bps > max_spread:
        blockers.append("schwab_quote_spread_not_canary_ready")
    if symbol_key not in canary_symbols:
        blockers.append("symbol_not_in_canary_stage_plan")
    if quantity <= 0.0 or quantity > max_quantity:
        blockers.append("canary_quantity_exceeds_hard_limit")
    if order_notional <= 0.0 or order_notional > max_notional:
        blockers.append("canary_order_notional_exceeds_hard_limit")
    if cash_available < account_capital:
        blockers.append("canary_settled_cash_not_broker_visible")
    if cash_available < order_notional:
        blockers.append("candidate_order_not_fully_cash_funded")
    if pending_deposits > 0.0:
        blockers.append("canary_deposit_pending_settlement")
    if bool(account.get("borrowing_allowed", False)):
        blockers.append("canary_account_borrowing_authority_present")
    if bool(account.get("closing_only", False)):
        blockers.append("canary_account_closing_only")
    if bool(account.get("broker_call_present", False)):
        blockers.append("canary_account_in_broker_call")
    covered = (
        account.get("covered_position_safety")
        if isinstance(account.get("covered_position_safety"), Mapping)
        else {}
    )
    if int(covered.get("uncovered_short_option_count", 0) or 0) > 0:
        blockers.append("canary_account_uncovered_short_options")
    if _safe_float(account.get("candidate_symbol_quantity"), 0.0) > 0.0:
        blockers.append("canary_candidate_position_already_open")
    if int(account.get("existing_canary_position_count", 0) or 0) >= int(
        _safe_float(hard_limits.get("max_concurrent_positions"), 1.0)
    ):
        blockers.append("canary_concurrent_position_cap_reached")
    blockers.extend(account.get("account_preflight_blockers", []))
    blockers.extend(
        str(item or "").strip()
        for item in (
            preflight_receipt.get("blockers")
            if isinstance(preflight_receipt.get("blockers"), list)
            else []
        )
        if str(item or "").strip()
    )
    blockers = _ordered_unique(blockers)

    order_spec = {
        "orderType": "LIMIT",
        "session": "NORMAL",
        "duration": "DAY",
        "price": f"{limit_price:.2f}",
        "orderStrategyType": "SINGLE",
        "orderLegCollection": [
            {
                "instruction": "BUY",
                "quantity": quantity,
                "instrument": {"symbol": symbol_key, "assetType": "EQUITY"},
            }
        ],
    }
    order_request = {
        "symbol": symbol_key,
        "action": "BUY",
        "quantity": quantity,
        "asset_type": "EQUITY",
        "limit_price": limit_price,
        "account_reference": account_reference,
        "order_spec": order_spec,
    }
    quote_snapshot = {
        "timestamp_utc": str(quote.get("provider_timestamp_utc") or ""),
        "last_price": _safe_float(quote.get("last_price"), 0.0),
        "bid_price": bid,
        "ask_price": ask,
        "spread_bps": spread_bps,
        "quote_age_ms": (
            _safe_float(quote_age, 0.0) * 1000.0 if quote_age is not None else 0.0
        ),
        "source_provider": str(quote.get("source_provider") or ""),
        "source_venue": str(quote.get("source_venue") or ""),
        "snapshot_id": str(quote.get("snapshot_id") or ""),
    }
    risk_ready = not blockers
    intent = build_order_intent_evidence(
        decision_id=(
            f"live-canary-dress-rehearsal:{candidate_id or 'unbound'}:"
            f"{current.isoformat()}"
        ),
        symbol=symbol_key,
        action="BUY",
        quantity=quantity,
        strategy=route_id or "unbound_canary_route",
        asset_type="EQUITY",
        limit_price=limit_price,
        quote_snapshot=quote_snapshot,
        expected_fill={
            "expected_fill_price": limit_price,
            "touch_price": ask,
            "quoted_spread_bps": spread_bps,
            "partial_fill_ratio": 1.0,
            "paper_execution_status": "connected_read_only_rehearsal",
            "quote_source_mode": "schwab_api_read_only",
            "quote_crossed_or_locked": bool(ask > 0.0 and bid >= ask),
        },
        risk_decision={
            "ok": risk_ready,
            "gate": "live_canary_connected_read_only_dress_rehearsal",
            "reason": "ready" if risk_ready else "canary_rehearsal_blocked",
            "details": {
                "position_qty": _safe_float(
                    account.get("candidate_symbol_quantity"), 0.0
                ),
                "projected_position_qty": _safe_float(
                    account.get("candidate_symbol_quantity"), 0.0
                )
                + quantity,
                "order_notional": order_notional,
                "reference_price": ask,
                "intended_price": limit_price,
            },
        },
    )
    snapshot_evidence = {
        "broker_position_snapshot_sha256": str(account_study_sha256 or ""),
        "broker_position_snapshot_captured_at_utc": (
            account_time.isoformat() if account_time is not None else ""
        ),
        "broker_position_snapshot_quantity": _safe_float(
            account.get("candidate_symbol_quantity"), 0.0
        ),
        "account_policy_key": account_policy_key,
        "live_canary_preflight_receipt": dict(preflight_receipt),
    }
    envelope = build_live_execution_envelope(
        intent_evidence=intent,
        order_request=order_request,
        candidate_id=candidate_id,
        broker="schwab",
        account_reference=account_reference,
        account_snapshot_evidence=snapshot_evidence,
        policy_sha256=policy_sha256,
        ttl_seconds=max(
            _safe_float(firewall.get("live_execution_envelope_ttl_seconds"), 15.0),
            0.001,
        ),
        created_at_utc=current,
    )
    envelope_verification = verify_live_execution_envelope(
        envelope,
        expected_candidate_id=candidate_id,
        expected_account_reference=account_reference,
        expected_policy_sha256=policy_sha256,
        now_utc=current,
        max_quote_age_seconds=max_quote_age,
        max_account_snapshot_age_seconds=max_account_age,
        max_spread_bps=max_spread,
        max_future_skew_seconds=max_future_skew,
        require_affirmative_risk_decision=True,
        require_quote_provenance=True,
        allowed_quote_providers=("schwab_api",),
        require_canary_preflight_receipt=True,
        expected_account_policy_key=account_policy_key,
    )
    envelope_root_blockers = [
        str(item or "").strip()
        for item in envelope_verification.get("blockers", [])
        if str(item or "").strip()
        and str(item or "").strip() not in DERIVED_ENVELOPE_BLOCKERS
    ]
    blockers = _ordered_unique(blockers + envelope_root_blockers)

    funding_shortfall = max(account_capital - cash_available, 0.0)
    order_shortfall = max(order_notional - cash_available, 0.0)
    projected_cash = (
        round(cash_available - order_notional, 4)
        if cash_available >= order_notional and order_notional > 0.0
        else None
    )
    funding = {
        "cash_only_budget_enforced": True,
        "settled_cash_broker_visible_usd": round(cash_available, 4),
        "canary_cap_required_usd": round(account_capital, 4),
        "canary_cap_shortfall_usd": round(funding_shortfall, 4),
        "candidate_order_notional_usd": round(order_notional, 4),
        "candidate_order_cash_shortfall_usd": round(order_shortfall, 4),
        "pending_deposits_usd": round(pending_deposits, 4),
        "projected_settled_cash_after_fill_usd": projected_cash,
        "projected_cash_available": projected_cash is not None,
        "borrowing_or_buying_power_used": False,
        "policy": (
            "a missing cash amount is a funding shortfall, not a debt balance; "
            "provider margin fields never substitute for settled cash"
        ),
    }
    post_fill = {
        "symbol": symbol_key,
        "starting_quantity": round(
            _safe_float(account.get("candidate_symbol_quantity"), 0.0), 6
        ),
        "projected_quantity": round(
            _safe_float(account.get("candidate_symbol_quantity"), 0.0) + quantity,
            6,
        ),
        "projected_settled_cash_usd": projected_cash,
        "existing_canary_position_count": int(
            account.get("existing_canary_position_count", 0) or 0
        ),
        "projected_canary_position_count": int(
            account.get("existing_canary_position_count", 0) or 0
        )
        + (
            1
            if _safe_float(account.get("candidate_symbol_quantity"), 0.0) <= 0.0
            else 0
        ),
        "covered_position_safety": covered,
        "provider_balance_diagnostic": account.get("provider_balance_diagnostic", {}),
    }
    retirement_account_safety = {
        "retirement_account": bool(account.get("retirement_account", False)),
        "tax_wrapper": str(account.get("tax_wrapper") or "unknown"),
        "new_contribution_assumed": bool(
            account_constraints.get("new_contribution_assumed", False)
        ),
        "cross_account_wash_sale_review_required": bool(
            account_constraints.get("cross_account_wash_sale_review_required", False)
        ),
        "retirement_account_loss_capacity_review_required": bool(
            account_constraints.get(
                "retirement_account_loss_capacity_review_required", False
            )
        ),
        "required_operator_confirmations": list(
            preflight_receipt.get("required_operator_confirmations", [])
        ),
        "operator_attestation_ready": bool(
            preflight_receipt.get("operator_attestation_ready", False)
        ),
        "policy": (
            "retirement-account designation adds explicit loss-capacity and "
            "cross-account tax review; it does not grant live authority"
        ),
    }
    reconciliation = {
        "submit_attempt_limit": 1,
        "blind_mutation_retry_allowed": False,
        "unfilled_cancel_deadline_seconds": int(
            _safe_float(
                (
                    (plan.get("activation_contract") or {}).get(
                        "unfilled_order_cancel_deadline_seconds", 60
                    )
                    if isinstance(plan.get("activation_contract"), Mapping)
                    else 60
                ),
                60.0,
            )
        ),
        "expected_sequence": [
            "record_intent_before_submit",
            "submit_exact_sealed_payload_once",
            "capture_broker_acknowledgement_and_order_id",
            "poll_order_state_with_read_only_calls",
            "reconcile_filled_quantity_average_price_position_and_cash",
            "verify_existing_covered_position_collateral_unchanged",
            "cancel_once_after_deadline_then_reconcile",
        ],
        "ambiguous_submit_state": "submit_unknown_requires_broker_reconciliation",
        "ambiguous_cancel_state": "cancel_unknown_requires_broker_reconciliation",
        "actual_broker_acknowledgement": "not_requested_read_only_rehearsal",
        "actual_fill": "not_requested_read_only_rehearsal",
    }
    canary_ready = bool(not blockers and envelope_verification.get("ok", False))
    connected_read_only = bool(
        account_refresh_summary.get("ok", False)
        and not account_refresh_summary.get("skipped", False)
        and quote.get("transport", {}).get("ok", False)
    )
    payload: dict[str, Any] = {
        "schema_version": 1,
        "timestamp_utc": current.isoformat(),
        "ok": connected_read_only,
        "overall_status": (
            "ready"
            if canary_ready
            else "ready_locked" if connected_read_only else "blocked"
        ),
        "mode": "broker_connected_read_only",
        "policy_id": str(plan.get("policy_id") or ""),
        "candidate_id": candidate_id,
        "account_policy_key": account_policy_key,
        "execution_route_id": route_id,
        "symbol": symbol_key,
        "canary_ready": canary_ready,
        "broker_network_read_only": connected_read_only,
        "connected_account_snapshot_this_run": bool(
            account_refresh_summary.get("ok", False)
            and not account_refresh_summary.get("skipped", False)
        ),
        "account_reference_matches_designated_policy": account_reference_matches,
        "account_reference_sha256": (
            _sha256_text(account_reference) if account_reference else ""
        ),
        "account_study_age_seconds": (
            round(account_age, 6) if account_age is not None else None
        ),
        "account_snapshot_refresh": {
            "ok": bool(account_refresh_summary.get("ok", False)),
            "skipped": bool(account_refresh_summary.get("skipped", False)),
            "account_count": int(
                _safe_float(account_refresh_summary.get("account_count"), 0.0)
            ),
            "position_rows": int(
                _safe_float(account_refresh_summary.get("position_rows"), 0.0)
            ),
            "partial": bool(
                account_refresh_summary.get("account_snapshot_partial", False)
            ),
        },
        "quote": quote,
        "account": account,
        "funding_and_settlement": funding,
        "exact_order_preview": envelope.get("broker_order_request", {}),
        "post_fill_projection": post_fill,
        "retirement_account_safety": retirement_account_safety,
        "sealed_envelope_preview": envelope,
        "sealed_envelope_verification": envelope_verification,
        "reconciliation_expectations": reconciliation,
        "blockers": blockers,
        "broker_mutation_attempted": False,
        "paper_order_attempted": False,
        "live_order_attempted": False,
        "live_execution_authority": False,
        "soak_reset_required": False,
        "policy": (
            "this control may prove read-only broker connectivity and block a canary, "
            "but it cannot submit, cancel, replace, promote, or grant live authority"
        ),
    }
    serialized = json.dumps(payload, ensure_ascii=True, sort_keys=True)
    raw_reference_emitted = bool(account_reference and account_reference in serialized)
    payload["redaction"] = {
        "raw_account_number_emitted": False,
        "raw_account_hash_emitted": raw_reference_emitted,
        "provider_quote_payload_emitted": False,
    }
    if raw_reference_emitted:
        payload["ok"] = False
        payload["overall_status"] = "blocked"
        payload["canary_ready"] = False
        payload["blockers"] = _ordered_unique(
            list(payload["blockers"]) + ["raw_account_reference_redaction_failed"]
        )
    return payload


def _read_only_environment() -> dict[str, str | None]:
    previous = {key: os.environ.get(key) for key in READ_ONLY_ENVIRONMENT}
    os.environ.update(READ_ONLY_ENVIRONMENT)
    return previous


def _restore_environment(previous: Mapping[str, str | None]) -> None:
    for key, value in previous.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


def _refresh_account_study(*, quiet_auth: bool) -> dict[str, Any]:
    snapshot = refresh(quiet_auth=quiet_auth, rebuild_derived=False)
    if not bool(snapshot.get("ok", False)):
        return snapshot
    proc = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "ops" / "account_position_study.py"),
        ],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    result = dict(snapshot)
    result["account_position_study_refresh_ok"] = proc.returncode == 0
    if proc.returncode != 0:
        result["ok"] = False
        result["account_position_study_error"] = "account_position_study_refresh_failed"
    return result


def _resolve_account_references(
    *, plan: Mapping[str, Any], registry: Mapping[str, Any]
) -> tuple[str, str, str]:
    policy_key = str(plan.get("account_policy_key") or "").strip()
    slots = (
        registry.get("account_slots")
        if isinstance(registry.get("account_slots"), list)
        else []
    )
    slot = next(
        (
            dict(row)
            for row in slots
            if isinstance(row, Mapping)
            and str(row.get("account_policy_key") or "").strip() == policy_key
        ),
        {},
    )
    env_names = [
        str(item or "").strip()
        for item in slot.get("env_names", [])
        if str(item or "").strip().endswith("_HASH")
        and str(item or "").strip() != "SCHWAB_ACCOUNT_HASH"
    ]
    expected = ""
    source = ""
    for env_name in env_names:
        value = str(os.environ.get(env_name) or "").strip()
        if value:
            expected = value
            source = "policy_bound_runtime"
            break
    if not expected:
        account = _keychain_account()
        for env_name in env_names:
            value = str(
                _read_keychain_secret(_service_name(env_name), account) or ""
            ).strip()
            if value:
                expected = value
                source = "policy_bound_keychain"
                break
    selected = str(os.environ.get("SCHWAB_ACCOUNT_HASH") or "").strip() or expected
    return selected, expected, source


def run(
    *,
    symbol: str,
    refresh_account: bool,
    quiet_auth: bool,
    out_path: Path,
) -> dict[str, Any]:
    previous = _read_only_environment()
    try:
        account_refresh = (
            _refresh_account_study(quiet_auth=quiet_auth)
            if refresh_account
            else {"ok": True, "skipped": True}
        )
        plan = _load_json(PROJECT_ROOT / "config" / "live_canary_micro_policy_v1.json")
        readiness_path = (
            PROJECT_ROOT / "config" / "production_readiness_control_v1.json"
        )
        readiness = _load_json(readiness_path)
        firewall = (
            readiness.get("live_execution_risk_firewall")
            if isinstance(readiness.get("live_execution_risk_firewall"), Mapping)
            else {}
        )
        candidate = _load_json(
            PROJECT_ROOT / "governance" / "runtime" / "production_candidate_state.json"
        )
        registry = _load_json(PROJECT_ROOT / "config" / "account_policy_registry.json")
        account_reference, expected_reference, _source = _resolve_account_references(
            plan=plan, registry=registry
        )
        trader = build_schwab_trader(
            PROJECT_ROOT,
            mode="shadow",
            missing_credentials_message=(
                "Schwab credentials are required for connected canary rehearsal"
            ),
        )
        _quiet_auth(trader, quiet=quiet_auth)
        quote_result = trader._fetch_live_quote(symbol=str(symbol or "SCHD").upper())
        now = datetime.now(timezone.utc)
        account_study = _load_json(DEFAULT_ACCOUNT_STUDY_PATH)
        preflight = evaluate_live_canary_preflight(
            PROJECT_ROOT,
            symbol=str(symbol or "SCHD").upper(),
            action="BUY",
            account_reference=account_reference,
            env=dict(os.environ),
            now=now,
        )
        payload = build_dress_rehearsal_payload(
            now=now,
            plan=plan,
            firewall=firewall,
            candidate=candidate,
            account_study=account_study,
            account_study_sha256=file_sha256(DEFAULT_ACCOUNT_STUDY_PATH),
            account_reference=account_reference,
            expected_account_reference=expected_reference,
            quote_result=quote_result,
            preflight_receipt=preflight,
            account_refresh_summary=account_refresh,
            policy_sha256=file_sha256(readiness_path),
            symbol=symbol,
        )
    except Exception as exc:
        payload = {
            "schema_version": 1,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "ok": False,
            "overall_status": "blocked",
            "mode": "broker_connected_read_only",
            "canary_ready": False,
            "blockers": [f"connected_canary_rehearsal_failed:{type(exc).__name__}"],
            "broker_mutation_attempted": False,
            "paper_order_attempted": False,
            "live_order_attempted": False,
            "live_execution_authority": False,
            "soak_reset_required": False,
        }
    finally:
        _restore_environment(previous)
    safe_write_json_atomic(
        str(out_path),
        payload,
        project_root=str(PROJECT_ROOT),
        source="live_canary_dress_rehearsal",
    )
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run a Schwab-connected read-only stage-one canary dress rehearsal. "
            "No broker mutation or live authority is possible."
        )
    )
    parser.add_argument("--symbol", default="SCHD")
    parser.add_argument("--skip-account-refresh", action="store_true")
    parser.add_argument("--show-auth", action="store_true")
    parser.add_argument("--require-canary-ready", action="store_true")
    parser.add_argument("--out", default=str(DEFAULT_OUT_PATH))
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    payload = run(
        symbol=str(args.symbol or "SCHD"),
        refresh_account=not bool(args.skip_account_refresh),
        quiet_auth=not bool(args.show_auth),
        out_path=Path(args.out).expanduser().resolve(),
    )
    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        print(
            "live_canary_dress_rehearsal "
            f"ok={int(bool(payload.get('ok', False)))} "
            f"status={payload.get('overall_status', 'blocked')} "
            f"canary_ready={int(bool(payload.get('canary_ready', False)))} "
            f"blockers={','.join(payload.get('blockers', [])) or 'none'}"
        )
    if args.require_canary_ready:
        return 0 if payload.get("canary_ready", False) else 2
    return 0 if payload.get("ok", False) else 2


if __name__ == "__main__":
    raise SystemExit(main())
