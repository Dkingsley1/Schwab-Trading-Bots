from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation, ROUND_FLOOR
from typing import Any, Callable, Mapping

from core.live_order_ledger import LiveOrderLedger, TERMINAL_STATES
from core.order_intent import canonical_payload_sha256

PURPOSE = "supervised_broker_test"
AUTHORITY = {
    "live_execution_authority": False,
    "autonomous_execution": False,
    "production_promotion_credit": False,
    "strategy_profitability_proven": False,
}


def number(value: Any) -> Decimal:
    if isinstance(value, bool):
        raise ValueError("boolean is not a number")
    try:
        result = Decimal(str(value))
    except (InvalidOperation, ValueError, TypeError) as exc:
        raise ValueError("invalid number") from exc
    if not result.is_finite():
        raise ValueError("nonfinite number")
    return result


def timestamp(value: Any) -> datetime:
    result = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if result.tzinfo is None:
        raise ValueError("timestamp timezone required")
    return result.astimezone(timezone.utc)


def fresh(value: Any, now: datetime, seconds: float) -> bool:
    try:
        return -2 <= (now - timestamp(value)).total_seconds() <= seconds
    except (TypeError, ValueError):
        return False


def account_digest(reference: str) -> str:
    return hashlib.sha256(reference.encode("utf-8")).hexdigest()


def validate_policy(plan: Mapping[str, Any]) -> None:
    limits = plan.get("hard_limits", {})
    authority = plan.get("authority", {})
    evidence = plan.get("evidence_contract", {})
    activation = plan.get("activation_contract", {})
    account = plan.get("account_constraints", {})
    if (
        plan.get("schema_version") != 1
        or plan.get("purpose") != PURPOSE
        or not re.fullmatch(r"[a-z0-9_]{1,80}", str(plan.get("test_id", "")))
        or plan.get("account_policy_key") != "schwab_roth_ira_primary"
        or plan.get("symbol") != "O"
        or plan.get("investment_style") != "buy_and_hold"
        or plan.get("entry_price_policy") != "passive_bid_no_chase"
        or not 0 < number(plan.get("entry_limit_ceiling_usd")) <= Decimal("58.08")
        or number(plan.get("account_capital_usd")) != 300
        or number(limits.get("max_order_notional_usd")) != 300
        or number(limits.get("max_order_quantity")) != 5
        or number(limits.get("cost_reserve_usd")) < 1
        or limits.get("max_entry_attempts") != 1
        or limits.get("max_exit_attempts") != 1
        or limits.get("max_concurrent_positions") != 1
        or limits.get("cancel_unfilled_after_seconds") != 60
        or account.get("required_account_kind") != "roth_ira"
        or account.get("required_tax_wrapper") != "roth_ira"
        or account.get("cash_only") is not True
        or account.get("existing_positions_authority") != "observe_only"
        or account.get("new_contribution_assumed") is not False
        or not 0 < number(activation.get("max_operator_attestation_hours")) <= 1
        or activation.get("live_execution_authority") is not False
        or any(
            activation.get(key) is not True
            for key in (
                "operator_confirmation_each_order",
                "limit_orders_only",
                "normal_session_only",
                "whole_shares_only",
            )
        )
        or not 0 < number(limits.get("max_spread_bps")) <= 75
        or not 0 < number(limits.get("max_limit_distance_bps")) <= 35
        or any(
            authority.get(key) is not False
            for key in (
                "autonomous_execution",
                "automatic_reentry",
                "automatic_sell",
                "automatic_reinvestment",
                "scheduler_execution",
                "live_execution_authority",
            )
        )
        or evidence.get("production_promotion_credit") is not False
        or evidence.get("strategy_profitability_proven") is not False
        or evidence.get("technical_preflight_required") is not True
        or evidence.get("immutable_release_required") is not True
    ):
        raise ValueError("supervised test policy outside reviewed boundary")


def propose_entry(
    plan: Mapping[str, Any], quote: Mapping[str, Any], *, now: datetime
) -> dict[str, Any]:
    validate_policy(plan)
    blockers: list[str] = []
    if (
        quote.get("source_provider") != "schwab_api"
        or quote.get("realtime") is not True
    ):
        blockers.append("realtime_schwab_quote_required")
    if not fresh(quote.get("provider_timestamp_utc"), now, 15):
        blockers.append("fresh_quote_required")
    try:
        bid, ask = number(quote.get("bid_price")), number(quote.get("ask_price"))
        if bid <= 0 or ask <= bid:
            raise ValueError("invalid or locked quote")
        spread = (ask - bid) / ((ask + bid) / 2) * 10000
        if spread > number(plan["hard_limits"]["max_spread_bps"]):
            blockers.append("spread_exceeds_cap")
        price = min(bid, number(plan["entry_limit_ceiling_usd"])).quantize(
            Decimal("0.01"), rounding=ROUND_FLOOR
        )
        available = number(plan["account_capital_usd"]) - number(
            plan["hard_limits"]["cost_reserve_usd"]
        )
        qty = min(
            int(available / price), int(plan["hard_limits"]["max_order_quantity"])
        )
        request = build_request(plan, action="BUY", quantity=qty, limit_price=price)
    except (ValueError, InvalidOperation, ZeroDivisionError):
        blockers.append("quote_or_affordable_whole_share_quantity_invalid")
        request = {}
    return {
        "state": "blocked" if blockers else "proposed",
        "blockers": blockers,
        "request": request if not blockers else {},
        "price_method": "lower_of_fresh_bid_and_operator_ceiling_rounded_down_to_cent",
        "operator_ceiling_usd": float(number(plan["entry_limit_ceiling_usd"])),
        "rationale": "Passive bid-side limit; reserve $1 for costs; no market order, automatic replacement, or price chasing.",
        "valuation_assessment": "not_established_by_quote",
        "fill_guaranteed": False,
        "operator_approval_required": True,
        **AUTHORITY,
    }


def build_request(
    plan: Mapping[str, Any], *, action: str, quantity: Any, limit_price: Any
) -> dict[str, Any]:
    validate_policy(plan)
    if action not in {"BUY", "SELL"}:
        raise ValueError("only BUY and separately confirmed SELL are supported")
    qty, price = number(quantity), number(limit_price)
    limits = plan["hard_limits"]
    if (
        qty <= 0
        or qty != qty.to_integral_value()
        or qty > number(limits["max_order_quantity"])
    ):
        raise ValueError("quantity must be one to five whole shares")
    if price <= 0 or price != price.quantize(Decimal("0.01")):
        raise ValueError("positive cent-valid limit price required")
    if action == "BUY" and price > number(plan["entry_limit_ceiling_usd"]):
        raise ValueError("buy limit exceeds the operator's $58.08 ceiling")
    if action == "BUY" and qty * price + number(limits["cost_reserve_usd"]) > number(
        plan["account_capital_usd"]
    ):
        raise ValueError("order plus cost reserve exceeds the $300 test budget")
    return {
        "orderType": "LIMIT",
        "session": "NORMAL",
        "duration": "DAY",
        "price": f"{price:.2f}",
        "orderStrategyType": "SINGLE",
        "orderLegCollection": [
            {
                "instruction": action,
                "quantity": int(qty),
                "instrument": {"symbol": plan["symbol"], "assetType": "EQUITY"},
            }
        ],
    }


def request_fields(
    plan: Mapping[str, Any], request: Mapping[str, Any]
) -> tuple[str, int, Decimal]:
    try:
        legs = request["orderLegCollection"]
        if len(legs) != 1:
            raise ValueError("one order leg required")
        leg = legs[0]
        action = leg["instruction"]
        expected = build_request(
            plan, action=action, quantity=leg["quantity"], limit_price=request["price"]
        )
        if dict(request) != expected:
            raise ValueError(
                "order payload differs from the approved single-equity limit template"
            )
        return action, int(leg["quantity"]), number(request["price"])
    except (KeyError, TypeError, IndexError) as exc:
        raise ValueError("invalid test order") from exc


def intent_id(plan: Mapping[str, Any], action: str) -> str:
    if action not in {"BUY", "SELL"}:
        raise ValueError("invalid test action")
    # Stable across process restarts, attestations, and release candidates.
    return f"broker-function-test:{plan['test_id']}:{action}"


def intent_payload(row: Mapping[str, Any]) -> dict[str, Any]:
    try:
        result = json.loads(str(row.get("payload_json", "{}")))
    except (ValueError, TypeError):
        return {}
    return result if isinstance(result, dict) else {}


def lifecycle_check(
    plan: Mapping[str, Any],
    request: Mapping[str, Any],
    ledger: LiveOrderLedger,
    *,
    account_reference: str,
    position_quantity: Any,
    unencumbered_quantity: Any,
    account_captured_at: str,
) -> list[str]:
    action, qty, _ = request_fields(plan, request)
    blockers: list[str] = []
    if not ledger.verify_integrity().get("ok"):
        blockers.append("order_ledger_integrity_failed")
    if ledger.unresolved():
        blockers.append("unresolved_orders_require_reconciliation")
    if ledger.get(intent_id(plan, action)):
        blockers.append("test_order_attempt_already_consumed")
    if action == "BUY":
        if number(position_quantity) != 0:
            blockers.append("preexisting_test_symbol_position_observe_only")
    else:
        entry = ledger.get(intent_id(plan, "BUY"))
        payload = intent_payload(entry)
        if (
            entry.get("state") not in TERMINAL_STATES
            or number(entry.get("filled_quantity", 0)) <= 0
            or payload.get("purpose") != PURPOSE
            or payload.get("test_id") != plan["test_id"]
            or payload.get("account_reference_sha256")
            != account_digest(account_reference)
            or payload.get("symbol") != plan["symbol"]
        ):
            blockers.append("verified_test_entry_required_for_sell")
        else:
            filled = number(entry["filled_quantity"])
            baseline = number(payload.get("baseline_position_quantity", 0))
            if (
                number(position_quantity) != baseline + filled
                or qty > filled
                or qty > number(unencumbered_quantity)
            ):
                blockers.append(
                    "sell_must_reduce_only_verified_unencumbered_test_shares"
                )
            try:
                if timestamp(account_captured_at) <= timestamp(entry["updated_at_utc"]):
                    blockers.append("post_fill_account_snapshot_required")
            except (KeyError, ValueError, TypeError):
                blockers.append("post_fill_account_snapshot_required")
    return blockers


def approval_phrase(plan: Mapping[str, Any], request: Mapping[str, Any]) -> str:
    action, qty, price = request_fields(plan, request)
    return f"CONFIRM {action} {qty} {plan['symbol']} LIMIT {price:.2f} IN MY ROTH"


def dispatch_once(
    *,
    plan: Mapping[str, Any],
    request: Mapping[str, Any],
    ledger: LiveOrderLedger,
    assessment: Mapping[str, Any],
    approved_phrase: str,
    approved_at: datetime,
    dispatch: Callable[[dict[str, Any]], Mapping[str, Any]],
    now: datetime,
) -> dict[str, Any]:
    action, qty, _ = request_fields(plan, request)
    if (
        approved_phrase != approval_phrase(plan, request)
        or not fresh(approved_at.isoformat(), now, 60)
        or not fresh(assessment.get("timestamp_utc"), now, 15)
        or assessment.get("technical_ready") is not True
        or assessment.get("operator_attestation_ready") is not True
        or assessment.get("operator_submit_ready") is not True
        or bool(assessment.get("blockers"))
        or assessment.get("request_sha256") != canonical_payload_sha256(request)
        or assessment.get("policy_sha256") != canonical_payload_sha256(plan)
        or assessment.get("purpose") != PURPOSE
    ):
        return {
            "ok": False,
            "blockers": ["fresh_exact_order_operator_approval_and_preflight_required"],
            **AUTHORITY,
        }
    key = intent_id(plan, action)
    payload = {
        "purpose": PURPOSE,
        "test_id": plan["test_id"],
        "symbol": plan["symbol"],
        "action": action,
        "quantity": qty,
        "order_spec": dict(request),
        "account_reference_sha256": assessment["account_reference_sha256"],
        "account_policy_key": plan["account_policy_key"],
        "candidate_id": assessment["candidate_id"],
        "test_policy_sha256": canonical_payload_sha256(plan),
        "baseline_position_quantity": assessment["position_quantity"],
        "baseline_cash_usd": assessment["settled_cash_usd"],
        "approved_at_utc": approved_at.isoformat(),
        "technical_receipt_sha256": canonical_payload_sha256(assessment),
        "production_promotion_credit": False,
    }
    reservation = ledger.reserve(intent_id=key, payload=payload, requested_quantity=qty)
    if not reservation.get("reserved"):
        return {
            "ok": False,
            "blockers": ["test_order_attempt_already_consumed"],
            **AUTHORITY,
        }
    ledger.mark_submitting(key)
    try:
        result = dict(dispatch(dict(request)))
        broker_id = str(result.get("order_id") or "")
        status = int(result.get("status_code") or 0)
        acknowledged = (
            result.get("ok") is True and 200 <= status < 300 and bool(broker_id)
        )
        rejected = 400 <= status < 500 and status not in {408, 409, 425, 429}
    except Exception:
        broker_id, acknowledged, rejected = "", False, False
    state = ledger.mark_submit_result(
        intent_id=key,
        acknowledged=acknowledged,
        broker_order_id=broker_id,
        definitively_rejected=rejected,
        error="" if acknowledged else "broker_test_submit_not_acknowledged",
    )
    return {
        "ok": acknowledged,
        "intent_id": key,
        "state": state["state"],
        "broker_mutation_attempted": True,
        "reconciliation_required": state["state"] not in TERMINAL_STATES,
        "automatic_retry_allowed": False,
        **AUTHORITY,
    }


def reconcile_order(
    ledger: LiveOrderLedger, row: Mapping[str, Any], broker_order: Mapping[str, Any]
) -> dict[str, Any]:
    """Accept actual executions, never substitute the order's limit for its fill."""
    payload = intent_payload(row)
    spec = payload.get("order_spec", {})
    legs = broker_order.get("orderLegCollection", [])
    if (
        str(broker_order.get("orderId", "")) != str(row.get("broker_order_id", ""))
        or len(legs) != 1
        or legs[0].get("instruction") != payload.get("action")
        or legs[0].get("instrument", {}).get("symbol") != payload.get("symbol")
        or legs[0].get("instrument", {}).get("assetType") != "EQUITY"
        or number(legs[0].get("quantity", -1)) != number(row["requested_quantity"])
        or broker_order.get("orderType") != "LIMIT"
        or number(broker_order.get("price", -1)) != number(spec.get("price", 0))
    ):
        raise ValueError("broker order identity mismatch")
    filled = number(broker_order.get("filledQuantity", 0))
    requested = number(row["requested_quantity"])
    if not 0 <= filled <= requested:
        raise ValueError("invalid broker fill quantity")
    fills = [
        leg
        for activity in broker_order.get("orderActivityCollection", [])
        if activity.get("activityType") == "EXECUTION"
        for leg in activity.get("executionLegs", [])
    ]
    executed = sum((number(leg["quantity"]) for leg in fills), Decimal(0))
    if executed != filled or any(
        number(leg["price"]) <= 0 or number(leg["quantity"]) <= 0 for leg in fills
    ):
        raise ValueError("actual fill executions missing or incomplete")
    average = (
        sum(
            (number(leg["quantity"]) * number(leg["price"]) for leg in fills),
            Decimal(0),
        )
        / filled
        if filled
        else Decimal(0)
    )
    status = str(broker_order.get("status") or "").upper()
    if status not in {
        "FILLED",
        "CANCELED",
        "CANCELLED",
        "REJECTED",
        "EXPIRED",
        "WORKING",
        "QUEUED",
        "ACCEPTED",
        "PENDING_ACTIVATION",
        "PENDING_ACKNOWLEDGEMENT",
        "PENDING_CANCEL",
        "PARTIALLY_FILLED",
    }:
        raise ValueError("unknown broker order state requires manual reconciliation")
    if status == "FILLED" and filled != requested:
        raise ValueError("filled status without full execution proof")
    if any(
        (payload["action"] == "BUY" and number(leg["price"]) > number(spec["price"]))
        or (
            payload["action"] == "SELL" and number(leg["price"]) < number(spec["price"])
        )
        for leg in fills
    ):
        raise ValueError("broker fill violates approved limit")
    if row.get("state") in TERMINAL_STATES:
        from core.live_order_ledger import normalize_broker_state

        if (
            normalize_broker_state(status) != row["state"]
            or filled != number(row["filled_quantity"])
            or abs(average - number(row["average_fill_price"])) > Decimal("0.000001")
        ):
            raise ValueError("terminal broker state changed")
        return dict(row)
    return ledger.record_broker_update(
        broker_order_id=str(row["broker_order_id"]),
        broker_status=status,
        filled_quantity=float(filled),
        average_fill_price=float(average),
    )


def holding_observation(
    *,
    plan: Mapping[str, Any],
    ledger: LiveOrderLedger,
    account_reference: str,
    position_quantity: Any,
    account_captured_at: str,
    now: datetime,
    dividend_events: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    validate_policy(plan)
    entry, exit_row = (ledger.get(intent_id(plan, side)) for side in ("BUY", "SELL"))
    blockers: list[str] = []
    if not ledger.verify_integrity().get("ok"):
        blockers.append("order_ledger_integrity_failed")
    if not fresh(account_captured_at, now, 30):
        blockers.append("fresh_account_observation_required")
    payload = intent_payload(entry)
    if not entry:
        state = "not_started"
    elif entry.get("state") not in TERMINAL_STATES or (
        exit_row and exit_row.get("state") not in TERMINAL_STATES
    ):
        state = "reconciliation_pending"
    elif payload.get("account_reference_sha256") != account_digest(account_reference):
        state = "blocked"
        blockers.append("test_account_mismatch")
    else:
        bought = number(entry["filled_quantity"])
        sold = number(exit_row.get("filled_quantity", 0))
        expected = number(payload.get("baseline_position_quantity", 0)) + bought - sold
        if number(position_quantity) != expected or sold > bought:
            blockers.append("broker_position_does_not_match_test_fills")
        latest_fill = max(
            timestamp(row["updated_at_utc"]) for row in (entry, exit_row) if row
        )
        try:
            if timestamp(account_captured_at) <= latest_fill:
                blockers.append("post_fill_account_snapshot_required")
        except ValueError:
            blockers.append("post_fill_account_snapshot_required")
        state = (
            "holding_observed"
            if bought > sold
            else "round_trip_observed" if bought > 0 else "entry_not_filled"
        )
    if blockers:
        state = "blocked"
    # Transactions are observations only; attribution to test shares is not assumed.
    dividends = []
    seen = set()
    for event in dividend_events or []:
        event_id = str(event.get("event_id") or "")
        if (
            not event_id
            or event_id in seen
            or not entry
            or event.get("account_policy_key") != plan["account_policy_key"]
            or event.get("symbol") != plan["symbol"]
            or event.get("tax_event_kind") != "dividend"
        ):
            continue
        try:
            when = timestamp(event.get("transaction_date"))
            if not timestamp(entry["created_at_utc"]) <= when <= now:
                continue
        except (TypeError, ValueError):
            continue
        seen.add(event_id)
        dividends.append(
            {
                "event_sha256": canonical_payload_sha256(event),
                "transaction_date": when.isoformat(),
                "amount_usd": event.get("amount_usd"),
                "attributed_to_test": False,
            }
        )
    return {
        "timestamp_utc": now.isoformat(),
        "purpose": PURPOSE,
        "test_id": plan["test_id"],
        "state": state,
        "blockers": blockers,
        "entry_state": entry.get("state", "not_started"),
        "exit_state": exit_row.get("state", "not_tested"),
        "buy_fill_observed": bool(
            entry and number(entry.get("filled_quantity", 0)) > 0
        ),
        "sell_fill_observed": bool(
            exit_row and number(exit_row.get("filled_quantity", 0)) > 0
        ),
        "dividend_events": dividends,
        "dividend_observed": bool(dividends),
        "dividend_reinvestment_authorized": False,
        "automatic_order_requested": False,
        **AUTHORITY,
    }
