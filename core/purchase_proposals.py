"""Proposal-only purchase policy and bounded validation history. No broker access."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Mapping

from core.order_intent import canonical_payload_sha256
from core.supervised_broker_test import (
    fresh,
    number,
    propose_entry,
    timestamp,
    validate_policy as validate_test_policy,
)

AUTHORITY = {
    "live_execution_authority": False,
    "autonomous_execution": False,
    "automatic_sell": False,
    "automatic_reentry": False,
    "automatic_reinvestment": False,
    "production_promotion_credit": False,
}


def observed_fresh(value: Any, now: datetime, seconds: float) -> bool:
    return fresh(value, now, seconds) and timestamp(value) <= now


def validate_policy(policy: Mapping[str, Any], test: Mapping[str, Any]) -> None:
    validate_test_policy(test)
    if (
        policy.get("schema_version") != 1
        or policy.get("mode") != "proposal_only"
        or policy.get("policy_id") != "roth_o_purchase_proposals_v1"
        or policy.get("test_id") != test["test_id"]
        or policy.get("account_policy_key") != test["account_policy_key"]
        or policy.get("symbols") != [test["symbol"]]
        or policy.get("investment_style") != "buy_and_hold"
        or policy.get("authority") != AUTHORITY
        or any(value is not False for value in policy.get("authority", {}).values())
        or not 0
        < number(policy.get("lifetime_budget_usd"))
        <= number(test["account_capital_usd"])
        or type(policy.get("max_entry_attempts")) is not int
        or policy["max_entry_attempts"] != 1
        or type(policy.get("max_order_quantity")) is not int
        or not 1
        <= policy["max_order_quantity"]
        <= test["hard_limits"]["max_order_quantity"]
        or not 0
        < number(policy.get("limit_ceiling_usd"))
        <= number(test["entry_limit_ceiling_usd"])
        or number(policy["limit_ceiling_usd"])
        != number(policy["limit_ceiling_usd"]).quantize(number("0.01"))
        or number(policy.get("cost_reserve_usd"))
        < number(test["hard_limits"]["cost_reserve_usd"])
        or number(policy.get("minimum_purchase_spacing_seconds")) < 86400
        or number(policy.get("evaluation_interval_seconds")) != 900
        or not 0 < number(policy.get("max_observation_age_seconds")) <= 120
    ):
        raise ValueError("purchase_policy_outside_proposal_only_test_scope")
    start, end = timestamp(policy.get("valid_from_utc")), timestamp(
        policy.get("expires_at_utc")
    )
    validation = policy.get("validation", {})
    if (
        not timedelta(0) < end - start <= timedelta(days=31)
        or type(validation.get("minimum_independent_days")) is not int
        or validation["minimum_independent_days"] < 3
        or type(validation.get("minimum_observations")) is not int
        or not 20 <= validation["minimum_observations"] <= 256
        or validation.get("economic_evidence_required_for_execution_review") is not True
        or validation.get("operator_review_required") is not True
        or validation.get("automatic_stage_transition") is not False
    ):
        raise ValueError("purchase_policy_validation_or_expiry_invalid")


def evaluate(
    *,
    policy: Mapping[str, Any],
    test: Mapping[str, Any],
    observation: Mapping[str, Any],
    source: Mapping[str, Any],
    session: Mapping[str, Any],
    revoked: bool,
    now: datetime,
) -> dict[str, Any]:
    validate_policy(policy, test)
    reasons: list[str] = []
    scope = observation.get("purchase_scope", {})
    if revoked:
        reasons.append("purchase_policy_revoked")
    if (
        not timestamp(policy["valid_from_utc"])
        <= now
        < timestamp(policy["expires_at_utc"])
    ):
        reasons.append("purchase_policy_expired_or_not_yet_effective")
    if (
        not observed_fresh(
            observation.get("timestamp_utc"),
            now,
            float(policy["max_observation_age_seconds"]),
        )
        or observation.get("purpose") != "supervised_broker_test"
        or observation.get("test_id") != test["test_id"]
        or scope.get("test_policy_sha256") != canonical_payload_sha256(test)
        or scope.get("account_policy_key") != policy["account_policy_key"]
        or scope.get("symbol") != policy["symbols"][0]
        or observation.get("live_execution_authority") is not False
        or observation.get("autonomous_execution") is not False
    ):
        reasons.append("fresh_scope_bound_observation_required")
    reasons.extend(observation.get("blockers", []))
    attempts = scope.get("entry_attempts")
    if type(attempts) is not int or attempts not in (0, 1):
        reasons.append("durable_entry_attempt_count_unknown")
    elif attempts >= policy["max_entry_attempts"]:
        reasons.append("lifetime_entry_scope_consumed")
    if scope.get("entry_state") not in {
        "not_started",
        "filled",
        "canceled",
        "rejected",
        "expired",
    }:
        reasons.append("unresolved_order_requires_reconciliation")
    accounting = observation.get("accounting", {})
    if attempts:
        if accounting.get("account_cash_reconciled") is not True:
            reasons.append("account_cash_reconciliation_pending")
        if accounting.get("trade_cash_reconciled") is not True:
            reasons.append("trade_cash_reconciliation_pending")
        if accounting.get("settlement_observed") is not True:
            reasons.append("broker_settlement_pending")
    if source.get("ready") is not True or not source.get("candidate_id"):
        reasons.append("accepted_source_required")
    if attempts == 0:
        preflight = observation.get("proposal_preflight", {})
        if (
            preflight.get("technical_ready") is not True
            or preflight.get("technical_blockers")
            or not observed_fresh(preflight.get("timestamp_utc"), now, 30)
            or preflight.get("candidate_id") != source.get("candidate_id")
            or preflight.get("policy_sha256") != canonical_payload_sha256(test)
        ):
            reasons.append("fresh_native_technical_preflight_required")
    request: dict[str, Any] = {}
    pricing: dict[str, Any] = {}
    if not reasons:
        if session.get("ready") is not True:
            reasons.append("normal_exchange_session_required")
        if number(scope.get("position_quantity", -1)) != 0:
            reasons.append("existing_position_observe_only")
        pricing = propose_entry(test, observation.get("quote", {}), now=now)
        if observation.get("quote", {}).get("symbol") != policy["symbols"][0]:
            reasons.append("quote_symbol_mismatch")
        if not observed_fresh(
            observation.get("quote", {}).get("provider_timestamp_utc"), now, 15
        ):
            reasons.append("fresh_quote_required")
        if pricing.get("state") != "proposed":
            reasons.extend(pricing.get("blockers", ["fresh_executable_quote_required"]))
        else:
            draft = pricing["request"]
            price = min(number(draft["price"]), number(policy["limit_ceiling_usd"]))
            budget = min(
                number(policy["lifetime_budget_usd"]),
                number(scope.get("funding_proxy_usd", 0)),
            )
            quantity = min(
                policy["max_order_quantity"],
                int((budget - number(policy["cost_reserve_usd"])) // price),
            )
            if quantity < 1:
                reasons.append("insufficient_observed_cash_for_proposal")
            elif not reasons:
                import copy

                request = copy.deepcopy(draft)
                request["price"] = str(price.quantize(number("0.01")))
                request["orderLegCollection"][0]["quantity"] = quantity
    state = "proposed_for_review" if request else "abstain"
    if (
        attempts == 1
        and observation.get("state") == "holding_observed"
        and "fresh_scope_bound_observation_required" not in reasons
    ):
        state = "holding_only"
    if revoked:
        state = "revoked"
    elif "purchase_policy_expired_or_not_yet_effective" in reasons:
        state = "expired"
    return {
        "schema_version": 1,
        "timestamp_utc": now.isoformat(),
        "purpose": "purchase_proposals",
        "overall_status": (
            "needs_attention" if reasons or accounting.get("pending") else "ready"
        ),
        "state": state,
        "mode": "proposal_only",
        "policy_id": policy["policy_id"],
        "policy_sha256": canonical_payload_sha256(policy),
        "candidate_id": source.get("candidate_id"),
        "source_ready": source.get("ready") is True,
        "observation_timestamp_utc": observation.get("timestamp_utc"),
        "observation_sha256": canonical_payload_sha256(observation),
        "account_policy_key": policy["account_policy_key"],
        "symbol": policy["symbols"][0],
        "entry_attempts": attempts,
        "lifetime_budget_usd": policy["lifetime_budget_usd"],
        "entry_gross_usd": scope.get("entry_gross_usd"),
        "remaining_entry_attempts": (
            max(0, policy["max_entry_attempts"] - attempts)
            if type(attempts) is int
            else None
        ),
        "proposal": request,
        "proposal_id": (
            canonical_payload_sha256(
                {"policy": policy, "request": request, "observation": observation}
            )
            if request
            else None
        ),
        "proposal_expires_at_utc": (
            (now + timedelta(seconds=15)).isoformat() if request else None
        ),
        "reasons": sorted(set(reasons)),
        "accounting": dict(accounting),
        "dividend_tracking": observation.get("dividend_tracking", {}),
        "valuation_assessment": "not_established_by_quote",
        "settled_cash_certified": False,
        "operator_review_required": True,
        "automatic_order_requested": False,
        "strategy_profitability_proven": False,
        "execution_readiness": "not_authorized_by_proposal",
        **AUTHORITY,
    }


def validation_progress(
    policy: Mapping[str, Any],
    report: Mapping[str, Any],
    history: Mapping[str, Any],
    *,
    now: datetime,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Count independent observation slots, not repeated reads or source changes."""
    binding = {
        "policy_sha256": report["policy_sha256"],
        "candidate_id": report.get("candidate_id"),
    }
    prior_rows = history.get("observations", [])
    if not isinstance(prior_rows, list) or len(prior_rows) > 256:
        raise ValueError("validation_history_invalid")
    rows = list(prior_rows) if history.get("binding") == binding else []
    slots, hashes = set(), set()
    for row in rows:
        if (
            not isinstance(row, dict)
            or not row.get("observation_sha256")
            or timestamp(row.get("timestamp_utc")) > now
            or row.get("slot")
            != int(timestamp(row["timestamp_utc"]).timestamp()) // 900
            or row.get("slot") in slots
            or row.get("observation_sha256") in hashes
        ):
            raise ValueError("validation_history_invalid")
        slots.add(row["slot"])
        hashes.add(row["observation_sha256"])
    qualified = (
        report.get("source_ready") is True
        and observed_fresh(
            report.get("observation_timestamp_utc"),
            now,
            float(policy["max_observation_age_seconds"]),
        )
        and report.get("accounting", {}).get("account_cash_reconciled") is True
        and report.get("accounting", {}).get("trade_cash_reconciled") is True
        and report.get("accounting", {}).get("positions_reconciled") is True
        and report.get("accounting", {}).get("settlement_observed") is True
        and set(report.get("reasons", [])) <= {"lifetime_entry_scope_consumed"}
    )
    slot = (
        int(timestamp(report["observation_timestamp_utc"]).timestamp()) // 900
        if qualified
        else None
    )
    if qualified and all(
        row.get("slot") != slot
        and row.get("observation_sha256") != report["observation_sha256"]
        for row in rows
    ):
        rows.append(
            {
                "slot": slot,
                "timestamp_utc": report["observation_timestamp_utc"],
                "observation_sha256": report["observation_sha256"],
            }
        )
    rows = rows[-256:]
    days = len({timestamp(row["timestamp_utc"]).date().isoformat() for row in rows})
    enough = (
        len(rows) >= policy["validation"]["minimum_observations"]
        and days >= policy["validation"]["minimum_independent_days"]
    )
    progress = {
        "stage": "proposal_only",
        "qualified_observations": len(rows),
        "independent_utc_days": days,
        "observation_threshold_met": enough,
        "execution_review_eligible": False,
        "automatic_stage_transition": False,
        "remaining_reviews": [
            "candidate_bound_regression_evidence",
            "objective_appropriate_economic_evidence",
            "explicit_operator_execution_review",
        ],
        "economic_evidence": "not_certified_by_purchase_or_holding",
        **AUTHORITY,
    }
    return progress, {"schema_version": 1, "binding": binding, "observations": rows}
