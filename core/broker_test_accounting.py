"""Pure, read-only accounting for the existing supervised test ledger."""

from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal
from typing import Any, Mapping
from zoneinfo import ZoneInfo

from core.order_intent import canonical_payload_sha256
from core.supervised_broker_test import (
    account_digest,
    fresh,
    intent_payload,
    number,
    timestamp,
)


def reconcile_test_accounting(
    *,
    plan: Mapping[str, Any],
    orders: list[Mapping[str, Any]],
    transactions: Mapping[str, Any],
    reference: str,
    position_consistent: bool,
    now: datetime,
    cash_observation: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Match broker cash postings to exact orders; never infer fees from a limit."""
    pending: list[str] = []
    rows = transactions.get("rows", [])
    complete = transactions.get("source_complete") is True
    if not complete:
        pending.append("complete_account_transaction_window_required")
    unique: dict[str, Mapping[str, Any]] = {}
    invalid = False
    for row in rows:
        key = str(row.get("activityId") or row.get("transactionId") or "")
        if not key or (key in unique and dict(unique[key]) != dict(row)):
            invalid = True
        else:
            unique[key] = row
    if invalid:
        pending.append("transaction_identity_missing_or_conflicting")
    receipts = []
    for order in orders:
        payload = intent_payload(order)
        side = payload.get("action")
        reasons: list[str] = []
        if (
            payload.get("account_reference_sha256") != account_digest(reference)
            or payload.get("account_policy_key") != plan["account_policy_key"]
            or payload.get("test_id") != plan["test_id"]
            or payload.get("symbol") != plan["symbol"]
            or side not in {"BUY", "SELL"}
        ):
            reasons.append("order_account_or_test_identity_mismatch")
        if order.get("state") not in {"filled", "canceled", "rejected", "expired"}:
            reasons.append("terminal_order_required")
        quantity = number(order.get("filled_quantity", 0))
        gross = quantity * number(order.get("average_fill_price", 0))
        matches = [
            row
            for row in unique.values()
            if str(row.get("orderId") or "") == str(order.get("broker_order_id") or "")
            and bool(order.get("broker_order_id"))
        ]
        matched_qty, matched_gross, cash = Decimal(0), Decimal(0), Decimal(0)
        settled = bool(matches)
        settlement_date_reached = bool(matches)
        hashes = []
        for row in matches:
            try:
                when = timestamp(
                    row.get("time")
                    or row.get("transactionDate")
                    or row.get("tradeDate")
                )
                if not timestamp(order["created_at_utc"]) <= when <= now:
                    raise ValueError("transaction_time_outside_order_window")
                if row.get("type") != "TRADE" or row.get("status") != "VALID":
                    raise ValueError("posted_trade_required")
                items = row.get("transferItems")
                if not isinstance(items, list) or not all(
                    isinstance(item, dict) for item in items
                ):
                    raise ValueError("transaction_legs_missing")
                securities = [
                    item
                    for item in items
                    if item.get("instrument", {}).get("assetType") != "CURRENCY"
                ]
                if len(securities) != 1:
                    raise ValueError("single_equity_transaction_required")
                item = securities[0]
                if (
                    item.get("instrument", {}).get("symbol") != plan["symbol"]
                    or item.get("instrument", {}).get("assetType") != "EQUITY"
                ):
                    raise ValueError("transaction_symbol_mismatch")
                signed_qty = number(item.get("amount"))
                price = number(item.get("price"))
                if (
                    signed_qty <= 0 if side == "BUY" else signed_qty >= 0
                ) or price <= 0:
                    raise ValueError("transaction_side_or_price_mismatch")
                net = number(row.get("netAmount"))
                if net >= 0 if side == "BUY" else net <= 0:
                    raise ValueError("transaction_cash_direction_mismatch")
                matched_qty += abs(signed_qty)
                matched_gross += abs(signed_qty) * price
                cash += net
                try:
                    raw_settlement = str(row.get("settlementDate") or "")
                    if len(raw_settlement) == 10:
                        reached = (
                            date.fromisoformat(raw_settlement)
                            <= now.astimezone(ZoneInfo("America/New_York")).date()
                        )
                    else:
                        reached = timestamp(raw_settlement) <= now
                    settlement_date_reached = settlement_date_reached and reached
                    # An elapsed scheduled date is not confirmation of settlement.
                    settled = (
                        settled and reached and row.get("settlementStatus") == "SETTLED"
                    )
                except (TypeError, ValueError):
                    settled = False
                    settlement_date_reached = False
                hashes.append(canonical_payload_sha256(row))
            except (TypeError, ValueError, KeyError):
                reasons.append("matching_transaction_invalid_or_incomplete")
        if quantity > 0 and not matches:
            reasons.append("broker_trade_posting_pending")
        if matched_qty != quantity or abs(matched_gross - gross) > Decimal("0.01"):
            reasons.append("transaction_execution_quantity_or_gross_mismatch")
        charges = (-cash - gross) if side == "BUY" else (gross - cash)
        if charges < 0:
            reasons.append("negative_effective_charges_require_review")
        cash_ok = complete and not invalid and not reasons
        receipts.append(
            {
                "action": side,
                "filled_quantity": str(quantity),
                "execution_gross_usd": str(gross),
                "net_cash_usd": str(cash) if cash_ok else None,
                "effective_charges_usd": str(charges) if cash_ok else None,
                "charge_evidence": (
                    "broker_net_cash_minus_verified_execution_gross"
                    if cash_ok
                    else "pending"
                ),
                "fee_breakdown_certified": False,
                "transaction_cash_reconciled": cash_ok,
                "settlement_observed": settled and cash_ok,
                "settlement_date_reached": settlement_date_reached and cash_ok,
                "transaction_sha256s": sorted(hashes),
                "pending": sorted(set(reasons)),
            }
        )
        pending.extend(reasons)
        if quantity > 0 and not settled:
            pending.append("broker_settlement_pending")
    if not position_consistent:
        pending.append("position_reconciliation_required")
    baseline = (
        intent_payload(orders[0]).get("baseline_cash_observation", {}) if orders else {}
    )
    current = cash_observation or {}
    account_cash_ok = False
    expected_cash = None
    if orders:
        try:
            baseline_time = timestamp(baseline.get("timestamp_utc"))
            current_time = timestamp(current.get("timestamp_utc"))
            if (
                baseline.get("state") != "observed"
                or current.get("state") != "observed"
                or not fresh(current.get("timestamp_utc"), now, 30)
                or current_time > now
                or baseline_time > timestamp(orders[0]["created_at_utc"])
                or not fresh(
                    baseline.get("timestamp_utc"),
                    timestamp(orders[0]["created_at_utc"]),
                    60,
                )
                or any(
                    item.get("account_reference_sha256") != account_digest(reference)
                    for item in (baseline, current)
                )
                or any(
                    item.get("source") != "schwab_currentBalances.cashBalance"
                    or not item.get("broker_payload_sha256")
                    for item in (baseline, current)
                )
                or timestamp(transactions.get("window_start_utc")) > baseline_time
                or timestamp(transactions.get("window_end_utc")) < current_time
                or not complete
                or invalid
            ):
                raise ValueError("comparable_cash_evidence_required")
            delta = Decimal(0)
            for row in unique.values():
                when = timestamp(
                    row.get("time")
                    or row.get("transactionDate")
                    or row.get("tradeDate")
                )
                if row.get("status") != "VALID" or not baseline_time <= when <= now:
                    raise ValueError("all_account_cash_rows_must_be_posted_and_timed")
                if when <= current_time:
                    delta += number(row.get("netAmount"))
            expected_cash = number(baseline.get("balance_usd")) + delta
            account_cash_ok = abs(
                number(current.get("balance_usd")) - expected_cash
            ) <= Decimal("0.01")
            if not account_cash_ok:
                pending.append("account_cash_delta_mismatch")
        except (TypeError, ValueError, KeyError):
            pending.append("comparable_pretrade_and_posttrade_cash_evidence_required")
    fully_reconciled = (
        bool(receipts)
        and all(
            row["transaction_cash_reconciled"] and row["settlement_observed"]
            for row in receipts
        )
        and account_cash_ok
        and position_consistent
        and not pending
    )
    return {
        "timestamp_utc": now.isoformat(),
        "state": (
            "reconciled" if fully_reconciled else "pending" if orders else "not_started"
        ),
        "source_complete": complete and not invalid,
        "orders": receipts,
        "positions_reconciled": position_consistent,
        "trade_cash_reconciled": bool(receipts)
        and all(row["transaction_cash_reconciled"] for row in receipts),
        "account_cash_reconciled": account_cash_ok,
        "expected_account_cash_usd": (
            str(expected_cash) if expected_cash is not None else None
        ),
        "current_cash_observation": dict(current),
        "settlement_observed": bool(receipts)
        and all(row["settlement_observed"] for row in receipts),
        "pending": sorted(set(pending)),
        "cash_funding_authority": False,
        "tax_characterization": "not_certified",
    }
