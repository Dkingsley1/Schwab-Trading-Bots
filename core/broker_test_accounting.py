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


def reconcile_position_observation(
    *,
    plan: Mapping[str, Any],
    orders: list[Mapping[str, Any]],
    transactions: Mapping[str, Any],
    reference: str,
    position_quantity: Any,
    account_captured_at: str,
    now: datetime,
) -> dict[str, Any]:
    """Explain additional purchases without assigning outside lots to the test."""
    pending: list[str] = []
    extra = Decimal(0)
    hashes: list[str] = []
    expected = None
    test_remaining = None
    baseline = None
    try:
        entry = next(
            row for row in orders if intent_payload(row).get("action") == "BUY"
        )
        payload = intent_payload(entry)
        start = timestamp(entry["created_at_utc"])
        captured = timestamp(account_captured_at)
        baseline = number(payload.get("baseline_position_quantity", 0))
        test_remaining = sum(
            (
                number(row.get("filled_quantity", 0))
                * (1 if intent_payload(row).get("action") == "BUY" else -1)
            )
            for row in orders
        )
        if (
            transactions.get("source_complete") is not True
            or transactions.get("account_reference_sha256") != account_digest(reference)
            or not fresh(transactions.get("timestamp_utc"), now, 30)
            or timestamp(transactions.get("timestamp_utc")) > now
            or not fresh(account_captured_at, now, 30)
            or captured > now
            or timestamp(transactions.get("window_start_utc")) > start
            or not captured <= timestamp(transactions.get("window_end_utc")) <= now
            or test_remaining < 0
        ):
            raise ValueError("fresh_complete_account_bound_position_window_required")
        known_orders = {str(row.get("broker_order_id")) for row in orders}
        rows = transactions.get("rows")
        if not isinstance(rows, list) or len(rows) >= 1000:
            raise ValueError("complete_position_transactions_required")
        unique: dict[str, Mapping[str, Any]] = {}
        for row in rows:
            if not isinstance(row, dict):
                raise ValueError("position_transaction_invalid")
            identity = str(row.get("activityId") or row.get("transactionId") or "")
            if not identity or (identity in unique and dict(unique[identity]) != row):
                raise ValueError("position_transaction_identity_missing_or_conflicting")
            items = row.get("transferItems")
            if not isinstance(items, list) or not all(
                isinstance(item, dict) and isinstance(item.get("instrument"), dict)
                for item in items
            ):
                raise ValueError("position_transaction_legs_missing")
            unique[identity] = row
        test_cash = reconcile_test_accounting(
            plan=plan,
            orders=orders,
            transactions=transactions,
            reference=reference,
            position_consistent=True,
            now=now,
        )
        if not test_cash["trade_cash_reconciled"]:
            raise ValueError("verified_test_execution_postings_required")
        for row in unique.values():
            items = row["transferItems"]
            legs = [
                item
                for item in items
                if item.get("instrument", {}).get("symbol") == plan["symbol"]
            ]
            if not legs:
                continue
            when = timestamp(
                row.get("time") or row.get("transactionDate") or row.get("tradeDate")
            )
            if when < start:
                continue
            if when > captured or row.get("status") != "VALID":
                raise ValueError("posted_position_transaction_within_snapshot_required")
            if str(row.get("orderId") or "") in known_orders:
                continue
            # Dividends with zero security quantity are not new lots. Corporate
            # actions, transfers and outside reductions need separate attribution.
            if row.get("type") == "DIVIDEND_OR_INTEREST" and all(
                number(item.get("amount")) == 0 for item in legs
            ):
                continue
            securities = [
                item
                for item in items
                if item.get("instrument", {}).get("assetType") != "CURRENCY"
            ]
            if (
                row.get("type") != "TRADE"
                or not row.get("orderId")
                or len(legs) != 1
                or len(securities) != 1
                or legs[0].get("instrument", {}).get("assetType") != "EQUITY"
                or number(legs[0].get("price")) <= 0
                or number(row.get("netAmount")) >= 0
            ):
                raise ValueError("non_test_position_activity_requires_review")
            quantity = number(legs[0].get("amount"))
            if quantity <= 0:
                raise ValueError("non_test_position_reduction_requires_lot_review")
            extra += quantity
            hashes.append(canonical_payload_sha256(row))
        expected = baseline + test_remaining + extra
        if number(position_quantity) != expected:
            pending.append("broker_position_does_not_match_verified_activity")
    except (TypeError, ValueError, KeyError, StopIteration) as exc:
        pending.append(str(exc) or "position_evidence_invalid")
    return {
        "state": "reconciled" if not pending else "pending",
        "broker_position_quantity": str(position_quantity),
        "baseline_position_quantity": str(baseline) if baseline is not None else None,
        "test_remaining_quantity": (
            str(test_remaining) if test_remaining is not None else None
        ),
        "additional_purchase_quantity": str(extra),
        "expected_account_quantity": str(expected) if expected is not None else None,
        "additional_transaction_sha256s": sorted(hashes),
        "additional_shares_attributed_to_test": False,
        "sell_authority": False,
        "pending": sorted(set(pending)),
    }


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
