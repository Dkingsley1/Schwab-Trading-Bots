import copy
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from core.order_intent import canonical_payload_sha256
from core.purchase_proposals import (
    AUTHORITY,
    evaluate,
    validate_policy,
    validation_progress,
)

ROOT = Path(__file__).resolve().parents[1]
NOW = datetime(2026, 9, 17, 18, tzinfo=timezone.utc)


@pytest.fixture
def context():
    policy = json.loads((ROOT / "config/purchase_proposal_policy_v1.json").read_text())
    test = json.loads((ROOT / "config/supervised_broker_test_v1.json").read_text())
    observation = {
        "proposal_preflight": {
            "timestamp_utc": NOW.isoformat(),
            "candidate_id": "candidate-a",
            "policy_sha256": canonical_payload_sha256(test),
            "technical_ready": True,
            "technical_blockers": [],
        },
        "timestamp_utc": NOW.isoformat(),
        "purpose": "supervised_broker_test",
        "test_id": test["test_id"],
        "state": "not_started",
        "blockers": [],
        "live_execution_authority": False,
        "autonomous_execution": False,
        "quote": {
            "symbol": "O",
            "source_provider": "schwab_api",
            "realtime": True,
            "provider_timestamp_utc": NOW.isoformat(),
            "bid_price": 56.93,
            "ask_price": 56.94,
        },
        "accounting": {"state": "not_started", "pending": []},
        "purchase_scope": {
            "test_policy_sha256": canonical_payload_sha256(test),
            "account_policy_key": test["account_policy_key"],
            "symbol": "O",
            "entry_attempts": 0,
            "entry_state": "not_started",
            "entry_gross_usd": "0",
            "position_quantity": 0,
            "funding_proxy_usd": "850.14",
        },
    }
    return {
        "policy": policy,
        "test": test,
        "observation": observation,
        "source": {"ready": True, "candidate_id": "candidate-a"},
        "session": {"ready": True},
        "revoked": False,
        "now": NOW,
    }


def test_proposal_is_passive_limit_for_review_never_order_authority(context):
    result = evaluate(**context)
    assert result["state"] == "proposed_for_review"
    assert result["proposal"]["price"] == "56.93"
    assert result["proposal"]["orderLegCollection"][0]["quantity"] == 5
    assert result["valuation_assessment"] == "not_established_by_quote"
    assert not result["settled_cash_certified"]
    assert all(result[key] is False for key in AUTHORITY)


def test_real_filled_entry_consumes_scope_even_with_unspent_budget(context):
    observation = context["observation"]
    observation["state"] = "holding_observed"
    observation["purchase_scope"].update(
        entry_attempts=1,
        entry_state="filled",
        entry_gross_usd="284.65",
        position_quantity=5,
    )
    result = evaluate(**context)
    assert result["state"] == "holding_only"
    assert not result["proposal"]
    assert result["remaining_entry_attempts"] == 0
    assert "lifetime_entry_scope_consumed" in result["reasons"]
    context["source"]["candidate_id"] = "candidate-b"
    assert not evaluate(**context)["proposal"]


@pytest.mark.parametrize(
    "state",
    [
        "rejected",
        "expired",
        "canceled",
        "submitted",
        "submit_unknown",
        "partially_filled",
    ],
)
def test_every_entry_attempt_is_consumed_across_restarts(context, state):
    context["observation"]["purchase_scope"].update(entry_attempts=1, entry_state=state)
    assert not evaluate(**context)["proposal"]


@pytest.mark.parametrize(
    "case",
    [
        "revoked",
        "expired",
        "future",
        "stale",
        "wrong_symbol",
        "wrong_account",
        "wrong_policy",
        "unaccepted",
        "closed_session",
        "missing_attempts",
        "cash",
        "position",
        "bad_quote",
    ],
)
def test_fail_closed_inputs(context, case):
    obs = context["observation"]
    if case == "revoked":
        context["revoked"] = True
    elif case == "expired":
        context["now"] = NOW + timedelta(days=31)
    elif case == "future":
        obs["timestamp_utc"] = (NOW + timedelta(seconds=1)).isoformat()
    elif case == "stale":
        obs["timestamp_utc"] = (NOW - timedelta(seconds=121)).isoformat()
    elif case == "wrong_symbol":
        obs["purchase_scope"]["symbol"] = "SCHD"
    elif case == "wrong_account":
        obs["purchase_scope"]["account_policy_key"] = "taxable"
    elif case == "wrong_policy":
        obs["purchase_scope"]["test_policy_sha256"] = "different"
    elif case == "unaccepted":
        context["source"]["ready"] = False
    elif case == "closed_session":
        context["session"]["ready"] = False
    elif case == "missing_attempts":
        del obs["purchase_scope"]["entry_attempts"]
    elif case == "cash":
        obs["purchase_scope"]["funding_proxy_usd"] = 0
    elif case == "position":
        obs["purchase_scope"]["position_quantity"] = 1
    else:
        obs["quote"]["provider_timestamp_utc"] = (
            NOW - timedelta(seconds=16)
        ).isoformat()
    result = evaluate(**context)
    assert not result["proposal"]
    assert result["reasons"]


@pytest.mark.parametrize(
    "change",
    [
        {"mode": "live"},
        {"symbols": ["O", "SCHD"]},
        {"lifetime_budget_usd": 301},
        {"max_entry_attempts": 2},
        {"max_order_quantity": 6},
        {"limit_ceiling_usd": 58},
        {"cost_reserve_usd": 0},
        {"evaluation_interval_seconds": 30},
        {"minimum_purchase_spacing_seconds": 1},
        {"account_policy_key": "taxable"},
    ],
)
def test_policy_cannot_widen_reviewed_scope(context, change):
    context["policy"].update(change)
    with pytest.raises(ValueError):
        validate_policy(context["policy"], context["test"])


@pytest.mark.parametrize("authority", list(AUTHORITY))
def test_policy_cannot_enable_execution_or_promotion(context, authority):
    context["policy"]["authority"][authority] = True
    with pytest.raises(ValueError):
        validate_policy(context["policy"], context["test"])


@pytest.mark.parametrize(
    "change",
    [
        {"technical_ready": False},
        {"technical_blockers": ["halt_flags_active"]},
        {"candidate_id": "other"},
        {"policy_sha256": "other"},
        {"timestamp_utc": (NOW - timedelta(seconds=31)).isoformat()},
    ],
)
def test_fresh_existing_technical_guards_are_required(context, change):
    context["observation"]["proposal_preflight"].update(change)
    result = evaluate(**context)
    assert result["proposal"] == {}
    assert "fresh_native_technical_preflight_required" in result["reasons"]


def test_stricter_budget_and_price_cap_are_respected(context):
    context["policy"].update(lifetime_budget_usd=100, limit_ceiling_usd=56.90)
    result = evaluate(**context)
    assert result["proposal"]["price"] == "56.90"
    assert result["proposal"]["orderLegCollection"][0]["quantity"] == 1


def test_validation_deduplicates_slots_and_requires_independent_days(context):
    observation = context["observation"]
    observation["accounting"] = {
        "account_cash_reconciled": True,
        "trade_cash_reconciled": True,
        "positions_reconciled": True,
        "settlement_observed": True,
    }
    observation["purchase_scope"].update(entry_attempts=1, entry_state="filled")
    report = evaluate(**context)
    progress, history = validation_progress(context["policy"], report, {}, now=NOW)
    assert progress["qualified_observations"] == 1
    progress, history = validation_progress(context["policy"], report, history, now=NOW)
    assert progress["qualified_observations"] == 1
    report["observation_sha256"] = "different-same-slot"
    progress, history = validation_progress(context["policy"], report, history, now=NOW)
    assert progress["qualified_observations"] == 1
    for i in range(1, 21):
        when = NOW + timedelta(hours=i * 3)
        report.update(
            observation_timestamp_utc=when.isoformat(), observation_sha256=str(i)
        )
        progress, history = validation_progress(
            context["policy"], report, history, now=when
        )
    assert progress["observation_threshold_met"]
    assert progress["execution_review_eligible"] is False
    assert progress["automatic_stage_transition"] is False
    report["candidate_id"] = "candidate-b"
    progress, history = validation_progress(
        context["policy"], report, history, now=when
    )
    assert progress["qualified_observations"] == 1


def test_pending_or_unaccepted_evidence_cannot_accrue_validation(context):
    report = evaluate(**context)
    progress, history = validation_progress(context["policy"], report, {}, now=NOW)
    assert progress["qualified_observations"] == 0
    assert progress["economic_evidence"] == "not_certified_by_purchase_or_holding"


def test_malformed_history_is_not_silently_accepted(context):
    report = evaluate(**context)
    history = {
        "binding": {
            "policy_sha256": report["policy_sha256"],
            "candidate_id": report["candidate_id"],
        },
        "observations": [
            {
                "timestamp_utc": (NOW + timedelta(days=1)).isoformat(),
                "observation_sha256": "future",
            }
        ],
    }
    with pytest.raises(ValueError):
        validation_progress(context["policy"], report, history, now=NOW)
