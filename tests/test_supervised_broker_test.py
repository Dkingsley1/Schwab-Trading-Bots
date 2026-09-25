import copy
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from core.live_order_ledger import LiveOrderLedger
from core.order_intent import canonical_payload_sha256
from core.supervised_broker_test import (
    PURPOSE,
    account_digest,
    approval_phrase,
    build_request,
    dispatch_once,
    holding_observation,
    intent_id,
    lifecycle_check,
    propose_entry,
    reconcile_order,
    request_fields,
    validate_policy,
)

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def plan():
    return json.loads((ROOT / "config/supervised_broker_test_v1.json").read_text())


@pytest.fixture
def ledger(tmp_path):
    return LiveOrderLedger(tmp_path / "orders.sqlite3")


def quote(now, bid=57.08, ask=57.09):
    return {
        "source_provider": "schwab_api",
        "realtime": True,
        "provider_timestamp_utc": now.isoformat(),
        "bid_price": bid,
        "ask_price": ask,
    }


def ready(plan, request, now):
    return {
        "purpose": PURPOSE,
        "timestamp_utc": now.isoformat(),
        "technical_ready": True,
        "operator_attestation_ready": True,
        "operator_submit_ready": True,
        "blockers": [],
        "request_sha256": canonical_payload_sha256(request),
        "policy_sha256": canonical_payload_sha256(plan),
        "candidate_id": "candidate-test",
        "account_reference_sha256": account_digest("roth-test-hash"),
        "position_quantity": 0,
        "settled_cash_usd": 850.14,
    }


def submit(
    plan,
    ledger,
    *,
    action="BUY",
    price="57.09",
    qty=5,
    assessment_changes=None,
    dispatch=None
):
    now = datetime.now(timezone.utc)
    request = build_request(plan, action=action, quantity=qty, limit_price=price)
    assessment = ready(plan, request, now)
    assessment.update(assessment_changes or {})
    return dispatch_once(
        plan=plan,
        request=request,
        ledger=ledger,
        assessment=assessment,
        approved_phrase=approval_phrase(plan, request),
        approved_at=now,
        now=now,
        dispatch=dispatch
        or (lambda spec: {"ok": True, "status_code": 201, "order_id": "broker-test-1"}),
    )


def test_switch_rejection_before_dispatch_is_not_an_unknown_broker_outcome(plan, ledger):
    result = submit(plan, ledger, dispatch=lambda spec: {
        "ok": False, "error": "live_execution_switch_blocked", "broker_mutation_attempted": False,
    })
    assert result["broker_mutation_attempted"] is False
    assert result["state"] == "rejected"
    assert result["reconciliation_required"] is False
    assert result["automatic_retry_allowed"] is False
    assert submit(plan, ledger)["blockers"] == ["test_order_attempt_already_consumed"]


def order_payload(
    request, *, status="FILLED", filled=5, fill_price="57.08", broker_id="broker-test-1"
):
    return {
        **copy.deepcopy(request),
        "orderId": broker_id,
        "status": status,
        "filledQuantity": filled,
        "orderActivityCollection": (
            [
                {
                    "activityType": "EXECUTION",
                    "executionLegs": [{"quantity": filled, "price": fill_price}],
                }
            ]
            if filled
            else []
        ),
    }


def filled_entry(plan, ledger):
    submit(plan, ledger)
    request = build_request(plan, action="BUY", quantity=5, limit_price="57.09")
    return reconcile_order(
        ledger, ledger.get(intent_id(plan, "BUY")), order_payload(request)
    )


def test_passive_price_uses_lower_bid_and_keeps_budget(plan):
    now = datetime.now(timezone.utc)
    proposal = propose_entry(plan, quote(now), now=now)
    assert proposal["state"] == "proposed"
    assert proposal["request"]["price"] == "57.08"
    assert proposal["request"]["orderLegCollection"][0]["quantity"] == 5
    assert proposal["valuation_assessment"] == "not_established_by_quote"
    assert proposal["strategy_profitability_proven"] is False
    assert (
        propose_entry(plan, quote(now, bid=57.50, ask=57.51), now=now)["request"][
            "price"
        ]
        == "57.09"
    )


@pytest.mark.parametrize(
    "changes",
    [
        {"source_provider": "mock"},
        {"realtime": False},
        {"bid_price": "NaN"},
        {"bid_price": 0},
        {"ask_price": 57},
        {"ask_price": 57.08},
        {"ask_price": 60},
        {"provider_timestamp_utc": "missing"},
    ],
)
def test_bad_quotes_do_not_propose_orders(plan, changes):
    now = datetime.now(timezone.utc)
    assert propose_entry(plan, {**quote(now), **changes}, now=now)["request"] == {}


@pytest.mark.parametrize("offset", [-16, 3])
def test_quote_freshness_and_future_rejection(plan, offset):
    now = datetime.now(timezone.utc)
    assert (
        propose_entry(plan, quote(now + timedelta(seconds=offset)), now=now)["state"]
        == "blocked"
    )


@pytest.mark.parametrize(
    "qty,price",
    [
        (6, 58),
        (0, 58),
        (1.5, 58),
        (True, 58),
        (1, "NaN"),
        (1, "Infinity"),
        (1, "57.091"),
        (5, "57.10"),
        (1, 0),
        (1, -1),
    ],
)
def test_invalid_requests_rejected(plan, qty, price):
    with pytest.raises(ValueError):
        build_request(plan, action="BUY", quantity=qty, limit_price=price)


@pytest.mark.parametrize(
    "change",
    [
        {"orderType": "MARKET"},
        {"session": "SEAMLESS"},
        {"duration": "GOOD_TILL_CANCEL"},
        {"orderStrategyType": "TRIGGER"},
        {"childOrderStrategies": []},
        {
            "orderLegCollection": [
                {
                    "instruction": "BUY",
                    "quantity": 5,
                    "instrument": {"symbol": "SCHD", "assetType": "EQUITY"},
                }
            ]
        },
    ],
)
def test_payload_cannot_widen_order_scope(plan, change):
    request = build_request(plan, action="BUY", quantity=5, limit_price="57.09")
    request.update(change)
    with pytest.raises(ValueError):
        request_fields(plan, request)


def test_policy_cannot_enable_autonomy(plan):
    plan["authority"]["automatic_sell"] = True
    with pytest.raises(ValueError):
        validate_policy(plan)


def test_lowered_ceiling_rejects_old_price_and_cannot_be_widened(plan):
    assert plan["entry_limit_ceiling_usd"] == 57.09
    assert (
        build_request(plan, action="BUY", quantity=5, limit_price="57.09")["price"]
        == "57.09"
    )
    for price in ("57.10", "58.08"):
        with pytest.raises(ValueError, match="57.09 ceiling"):
            build_request(plan, action="BUY", quantity=5, limit_price=price)
    plan["entry_limit_ceiling_usd"] = 58.08
    with pytest.raises(ValueError, match="outside reviewed boundary"):
        validate_policy(plan)


@pytest.mark.parametrize(
    "section,key,value",
    [
        ("account_constraints", "cash_only", False),
        ("account_constraints", "required_tax_wrapper", "taxable"),
        ("activation_contract", "operator_confirmation_each_order", False),
        ("activation_contract", "live_execution_authority", True),
        ("activation_contract", "max_operator_attestation_hours", 2),
    ],
)
def test_policy_cannot_relax_account_or_operator_boundary(plan, section, key, value):
    plan[section][key] = value
    with pytest.raises(ValueError):
        validate_policy(plan)


def test_each_fill_must_respect_limit_even_when_average_is_below_limit(plan, ledger):
    submit(plan, ledger)
    request = build_request(plan, action="BUY", quantity=5, limit_price="57.09")
    broker = order_payload(request)
    broker["orderActivityCollection"][0]["executionLegs"] = [
        {"quantity": 1, "price": "57.10"},
        {"quantity": 4, "price": "57.00"},
    ]
    with pytest.raises(ValueError, match="violates approved limit"):
        reconcile_order(ledger, ledger.get(intent_id(plan, "BUY")), broker)


def test_dispatch_once_survives_restart_and_candidate_changes(plan, ledger):
    calls = []
    dispatch = lambda spec: calls.append(spec) or {
        "ok": True,
        "status_code": 201,
        "order_id": "broker-test-1",
    }
    assert submit(plan, ledger, dispatch=dispatch)["ok"] is True
    reopened = LiveOrderLedger(ledger.path)
    assert (
        submit(
            plan,
            reopened,
            dispatch=dispatch,
            assessment_changes={"candidate_id": "new-candidate"},
        )["ok"]
        is False
    )
    assert len(calls) == 1
    assert reopened.verify_integrity()["ok"]


@pytest.mark.parametrize(
    "changes",
    [
        {"technical_ready": False},
        {"operator_attestation_ready": False},
        {"operator_submit_ready": False},
        {"blockers": ["unsafe"]},
        {"request_sha256": "bad"},
        {"policy_sha256": "bad"},
        {"purpose": "production_canary"},
        {"timestamp_utc": "2000-01-01T00:00:00+00:00"},
    ],
)
def test_no_dispatch_without_exact_current_review(plan, ledger, changes):
    calls = []
    result = submit(
        plan,
        ledger,
        assessment_changes=changes,
        dispatch=lambda request: calls.append(request),
    )
    assert not result["ok"] and not calls and not ledger.intents()


def test_timeout_is_uncertain_and_never_retried(plan, ledger):
    calls = []

    def timeout(request):
        calls.append(request)
        raise TimeoutError("response lost")

    result = submit(plan, ledger, dispatch=timeout)
    assert result["state"] == "submit_unknown"
    assert result["automatic_retry_allowed"] is False
    assert submit(plan, ledger, dispatch=timeout)["ok"] is False
    assert len(calls) == 1


def test_ack_without_broker_order_id_remains_unknown(plan, ledger):
    result = submit(
        plan, ledger, dispatch=lambda request: {"ok": True, "status_code": 201}
    )
    assert result["state"] == "submit_unknown"


@pytest.mark.parametrize(
    "change",
    [
        {"orderId": "wrong"},
        {"price": "57.00"},
        {"orderType": "MARKET"},
        {"filledQuantity": 6},
        {"filledQuantity": 4},
        {"orderActivityCollection": []},
        {
            "orderActivityCollection": [
                {
                    "activityType": "EXECUTION",
                    "executionLegs": [{"quantity": 5, "price": 59}],
                }
            ]
        },
    ],
)
def test_reconciliation_rejects_false_fill_proof(plan, ledger, change):
    submit(plan, ledger)
    request = build_request(plan, action="BUY", quantity=5, limit_price="57.09")
    with pytest.raises(ValueError):
        reconcile_order(
            ledger,
            ledger.get(intent_id(plan, "BUY")),
            {**order_payload(request), **change},
        )
    assert ledger.get(intent_id(plan, "BUY"))["state"] == "acknowledged"


def test_buy_hold_observation_never_sells_or_proves_profitability(plan, ledger):
    entry = filled_entry(plan, ledger)
    now = datetime.now(timezone.utc) + timedelta(seconds=1)
    result = holding_observation(
        plan=plan,
        ledger=LiveOrderLedger(ledger.path),
        account_reference="roth-test-hash",
        position_quantity=5,
        account_captured_at=now.isoformat(),
        now=now,
    )
    assert result["state"] == "holding_observed"
    assert result["buy_fill_observed"] and not result["sell_fill_observed"]
    assert result["strategy_profitability_proven"] is False
    assert result["automatic_order_requested"] is False
    assert len(ledger.intents()) == 1
    assert entry["average_fill_price"] == pytest.approx(57.08)


@pytest.mark.parametrize("reference,qty", [("wrong-account", 5), ("roth-test-hash", 6)])
def test_hold_rejects_wrong_account_and_position_drift(plan, ledger, reference, qty):
    filled_entry(plan, ledger)
    now = datetime.now(timezone.utc) + timedelta(seconds=1)
    result = holding_observation(
        plan=plan,
        ledger=ledger,
        account_reference=reference,
        position_quantity=qty,
        account_captured_at=now.isoformat(),
        now=now,
    )
    assert result["state"] == "blocked"


def test_hold_recognizes_verified_outside_buy_without_expanding_sell_scope(
    plan, ledger
):
    entry = filled_entry(plan, ledger)
    now = datetime.now(timezone.utc) + timedelta(seconds=2)
    started = datetime.fromisoformat(entry["created_at_utc"])
    trade = {
        "activityId": "test-fill",
        "orderId": entry["broker_order_id"],
        "status": "VALID",
        "type": "TRADE",
        "time": (started + timedelta(milliseconds=10)).isoformat(),
        "netAmount": "-285.40",
        "transferItems": [
            {
                "amount": 5,
                "price": "57.08",
                "instrument": {"symbol": "O", "assetType": "EQUITY"},
            }
        ],
    }
    extra = {**trade, "activityId": "extra", "orderId": "manual-purchase"}
    source = {
        "source_complete": True,
        "timestamp_utc": now.isoformat(),
        "account_reference_sha256": account_digest("roth-test-hash"),
        "window_start_utc": entry["created_at_utc"],
        "window_end_utc": now.isoformat(),
        "rows": [trade, extra],
    }
    result = holding_observation(
        plan=plan,
        ledger=ledger,
        account_reference="roth-test-hash",
        position_quantity=10,
        account_captured_at=now.isoformat(),
        now=now,
        transactions=source,
    )
    assert result["state"] == "holding_observed"
    assert result["position_reconciliation"]["additional_purchase_quantity"] == "5"
    assert result["live_execution_authority"] is False
    assert len(ledger.intents()) == 1
    sell = build_request(plan, action="SELL", quantity=5, limit_price="57.09")
    assert lifecycle_check(
        plan,
        sell,
        ledger,
        account_reference="roth-test-hash",
        position_quantity=10,
        unencumbered_quantity=10,
        account_captured_at=now.isoformat(),
    )


def test_sell_only_after_verified_own_test_fill(plan, ledger):
    request = build_request(plan, action="SELL", quantity=5, limit_price="57.09")
    kwargs = dict(
        account_reference="roth-test-hash",
        position_quantity=5,
        unencumbered_quantity=5,
        account_captured_at=(
            datetime.now(timezone.utc) + timedelta(seconds=1)
        ).isoformat(),
    )
    assert "verified_test_entry_required_for_sell" in lifecycle_check(
        plan, request, ledger, **kwargs
    )
    filled_entry(plan, ledger)
    assert lifecycle_check(plan, request, ledger, **kwargs) == []
    assert lifecycle_check(
        plan, request, ledger, **{**kwargs, "unencumbered_quantity": 4}
    )
    assert lifecycle_check(plan, request, ledger, **{**kwargs, "position_quantity": 10})


def test_buy_again_is_rejected_even_after_full_round_trip(plan, ledger):
    filled_entry(plan, ledger)
    submit(
        plan,
        ledger,
        action="SELL",
        assessment_changes={"position_quantity": 5},
        dispatch=lambda request: {
            "ok": True,
            "status_code": 201,
            "order_id": "broker-test-2",
        },
    )
    sell = build_request(plan, action="SELL", quantity=5, limit_price="57.09")
    reconcile_order(
        ledger,
        ledger.get(intent_id(plan, "SELL")),
        order_payload(sell, fill_price="57.09", broker_id="broker-test-2"),
    )
    now = datetime.now(timezone.utc) + timedelta(seconds=1)
    result = holding_observation(
        plan=plan,
        ledger=ledger,
        account_reference="roth-test-hash",
        position_quantity=0,
        account_captured_at=now.isoformat(),
        now=now,
    )
    assert result["state"] == "round_trip_observed"
    buy = build_request(plan, action="BUY", quantity=5, limit_price="57.09")
    assert "test_order_attempt_already_consumed" in lifecycle_check(
        plan,
        buy,
        ledger,
        account_reference="roth-test-hash",
        position_quantity=0,
        unencumbered_quantity=0,
        account_captured_at=now.isoformat(),
    )


def test_dividends_are_deduplicated_and_not_assumed_to_be_test_alpha(plan, ledger):
    filled_entry(plan, ledger)
    now = datetime.now(timezone.utc) + timedelta(seconds=1)
    event = {
        "event_id": "broker-event-1",
        "account_policy_key": plan["account_policy_key"],
        "symbol": "O",
        "tax_event_kind": "dividend",
        "transaction_date": now.isoformat(),
    }
    result = holding_observation(
        plan=plan,
        ledger=ledger,
        account_reference="roth-test-hash",
        position_quantity=5,
        account_captured_at=now.isoformat(),
        now=now,
        dividend_events=[
            event,
            event,
            {**event, "event_id": "wrong", "account_policy_key": "taxable"},
        ],
    )
    assert len(result["dividend_events"]) == 1
    assert result["dividend_events"][0]["attributed_to_test"] is False
    assert result["dividend_reinvestment_authorized"] is False
