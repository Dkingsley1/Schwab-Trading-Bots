"""Research-only historical annotations, subordinate to sleeve economic contracts."""

from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime
from copy import deepcopy
from typing import Any, Mapping

from core.sleeve_strategy_specialization import (
    load_policy,
    materialize_strategy_contracts,
)

VERSION = "historical_sleeve_labels_v2"
AUTHORITY = {
    "training_eligible": False,
    "profitability_evidence": False,
    "current_candidate_validation_credit": False,
    "training_launch_authority": False,
    "promotion_authority": False,
    "paper_execution_authority": False,
    "live_execution_authority": False,
}

# These are outcome requirements, not permission to substitute forecast features
# (yield, model scores, simulated fills, etc.) for realized economic outcomes.
REQUIREMENTS = {
    "directional_alpha": ["position_entry_exit", "realized_costs", "matched_benchmark"],
    "execution_alpha": [
        "arrival_quote",
        "observed_fill",
        "fees",
        "session_close",
        "adverse_selection_path",
    ],
    "income_total_return": [
        "position_entry_exit",
        "cash_distributions",
        "corporate_actions",
        "tax_cost_basis",
        "realized_costs",
    ],
    "event_alpha": [
        "verified_event_clock",
        "event_window",
        "matched_event_cohort",
        "realized_costs",
    ],
    "macro_carry_relative_value": [
        "position_entry_exit",
        "carry_cashflows",
        "roll_cashflows",
        "funding_and_hedging_costs",
        "matched_benchmark",
    ],
    "digital_asset_alpha": [
        "venue_instrument_identity",
        "position_entry_exit",
        "realized_fees",
        "funding_cashflows_if_derivative",
        "matched_beta_benchmark",
    ],
    "basis_relative_value": [
        "synchronized_leg_prices",
        "leg_quantities",
        "realized_basis_convergence",
        "funding_fees_borrow",
    ],
    "volatility_relative_value": [
        "contract_and_expiry",
        "leg_premiums_and_payoffs",
        "greek_exposure",
        "hedging_cashflows",
        "realized_costs",
    ],
    "market_neutral_relative_value": [
        "synchronized_leg_prices",
        "leg_quantities",
        "hedge_ratio",
        "factor_residual_return",
        "borrow_and_costs",
    ],
    "hedge_utility": [
        "portfolio_with_hedge",
        "matched_portfolio_without_hedge",
        "tail_loss_reduction",
        "carry_and_false_positive_costs",
    ],
    "capital_preservation": [
        "portfolio_path",
        "cash_or_defensive_benchmark",
        "avoided_drawdown",
        "opportunity_and_reentry_costs",
    ],
    "control_only": [
        "incident_or_action_id",
        "verified_action_result",
        "detection_and_recovery_clock",
        "false_positive_adjudication",
    ],
}


# Supplemental observed-price targets use elapsed UTC time, never row counts.
# They are explicitly NOT the sleeve's primary payoff, trade P&L, or a session
# calendar. Undefined research/expiry/event horizons remain unresolved.
def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def mapping(value: Any) -> dict:
    return dict(value) if isinstance(value, Mapping) else {}


def number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (ValueError, TypeError, OverflowError):
        return None


def epoch(value: Any) -> float | None:
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        return parsed.timestamp() if parsed.tzinfo is not None else None
    except (ValueError, TypeError, OverflowError):
        return None


def label_contracts(project_root) -> tuple[dict, dict]:
    policy = load_policy(project_root / "config/sleeve_strategy_contracts_v1.json")
    with (
        project_root / "config/historical_sleeve_research_horizons_v1.json"
    ).open() as handle:
        horizons = json.load(handle)
    if horizons.get("schema_version") != 1 or horizons.get("authority") != {
        "can_change_execution_horizon": False,
        "can_train": False,
        "can_promote": False,
        "can_submit_order": False,
    }:
        raise ValueError("invalid_research_horizon_authority")
    if horizons.get("endpoint_tolerance_seconds") != {
        "minimum": 30,
        "maximum": 300,
        "horizon_fraction": 0.01,
    }:
        raise ValueError("unsupported_endpoint_tolerance_contract")
    result = {}
    for row in materialize_strategy_contracts(
        policy=policy, project_root=project_root
    ).values():
        sleeve = row["sleeve_id"]
        if sleeve in result:
            continue
        objective = row["objective_class"]
        horizon = row["holding_horizon"]
        research = research_horizon(horizons, sleeve)
        contract = {
            "version": VERSION,
            "sleeve_id": sleeve,
            "objective_class": objective,
            "primary_metric": policy["objective_classes"][objective]["primary_metric"],
            "required_outcome_evidence": REQUIREMENTS[objective],
            "decision_horizon": row["decision_horizon"],
            "holding_horizon": horizon,
            "research_horizon": research,
            "research_horizon_policy_sha256": digest(horizons),
            "benchmark": row["benchmark"],
            "cost_model": row["cost_model"],
            "source_policy_sha256": digest(policy),
            "primary_outcome_policy": "authority_specific_receipt_required_no_price_direction_substitution",
            "supplemental_price_horizons_seconds": (
                []
                if objective == "control_only"
                else sorted(
                    set([research["primary_seconds"], *research["secondary_seconds"]])
                )
            ),
            "supplemental_horizon_clock": "elapsed_utc_not_exchange_sessions",
            "unresolved_horizon": False,
            "execution_holding_horizon_unchanged": True,
            "authority": AUTHORITY,
        }
        contract["contract_sha256"] = digest(contract)
        result[sleeve] = contract
    if set(result) != set(horizons["sleeves"]):
        raise ValueError("research_horizon_sleeve_coverage_mismatch")
    return result, dict(policy.get("profile_aliases", {}))


def research_horizon(policy: dict, sleeve: str) -> dict:
    key = policy["sleeves"][sleeve]["recipe"]
    result = deepcopy(policy["recipes"][key])
    values = [result["primary_seconds"], *result["secondary_seconds"]]
    if any(type(v) is not int or not 1 <= v <= 366 * 86400 for v in values):
        raise ValueError("invalid_research_horizon_duration")
    if (
        not result.get("endpoint")
        or not result.get("clock")
        or not result.get("rationale")
    ):
        raise ValueError("incomplete_research_horizon_definition")
    result.update(
        recipe=key,
        days_are="elapsed_calendar_days_not_trading_sessions",
        execution_authority=False,
    )
    result["horizon_sha256"] = digest(result)
    return result


def _profile(
    value: Any, aliases: dict, contracts: dict, *, crypto: bool = False
) -> str:
    raw = str(value or "").strip().lower()
    if raw in contracts:
        return raw
    if raw.startswith("shadow_"):
        raw = raw[7:]
    if raw in contracts:
        return raw
    for suffix in (
        "_equities_schwab",
        "_crypto_coinbase",
        "_crypto_schwab",
        "_equities",
        "_crypto",
    ):
        if raw.endswith(suffix):
            raw = raw[: -len(suffix)]
            break
    if (crypto and raw == "default") or raw == "crypto":
        raw = "crypto_spot"
    raw = aliases.get(raw, raw)
    return raw if raw in contracts else ""


def annotate(row: dict, contracts: dict, aliases: dict, source_rel: str = "") -> dict:
    meta = mapping(row.get("metadata"))
    spec = mapping(meta.get("strategy_specialization"))
    market = mapping(row.get("market"))
    features = mapping(row.get("features"))
    crypto = (
        str(row.get("asset_class") or meta.get("shadow_domain") or "").lower()
        == "crypto"
    )
    crypto = (
        crypto
        or row.get("broker") == "coinbase"
        or "_crypto" in str(row.get("mode", ""))
    )
    identities = {
        _profile(value, aliases, contracts, crypto=crypto)
        for value in (
            row.get("sleeve_id"),
            row.get("sleeve_profile"),
            row.get("profile"),
            row.get("sleeve"),
            meta.get("sleeve_id"),
            meta.get("sleeve_profile"),
            meta.get("sleeve"),
            spec.get("sleeve_id"),
            row.get("shadow_profile"),
            meta.get("source_profile"),
            row.get("mode"),
        )
        if value
    } - {""}
    # The exact known directory profile can recover old rows without metadata.
    # It must agree with explicit identities; there is no substring bot mapping.
    for part in source_rel.replace("\\", "/").split("/")[:-1]:
        profile = _profile(part, aliases, contracts, crypto=crypto or "_crypto" in part)
        if profile:
            identities.add(profile)
    sleeve = next(iter(identities)) if len(identities) == 1 else ""
    reasons = []
    if not sleeve:
        reasons.append(
            "conflicting_sleeve_identity" if identities else "unassigned_sleeve"
        )
    timestamp = row.get("timestamp_utc") or row.get("timestamp") or row.get("ts")
    ts = epoch(timestamp)
    if ts is None:
        reasons.append("invalid_or_naive_timestamp")
    symbol = str(row.get("symbol") or "").strip().upper()[:128]
    price = None
    for value in (
        row.get("price"),
        market.get("last_price"),
        features.get("last_price"),
    ):
        if number(value) is not None:
            price = number(value)
            break
    if price is not None and price <= 0:
        price = None
    provider = str(
        row.get("source_provider")
        or row.get("broker")
        or meta.get("source_broker")
        or ""
    )[:128]
    instrument = str(row.get("instrument_type") or row.get("asset_class") or "")[:128]
    snapshot = str(row.get("snapshot_id") or meta.get("snapshot_id") or "")[:256]
    candidate = str(
        row.get("production_candidate_id") or meta.get("production_candidate_id") or ""
    )[:256]
    contract = contracts.get(sleeve, {})
    objective = contract.get("objective_class", "unassigned")
    primary_reason = "authority_specific_outcome_receipt_not_materialized"
    if contract.get("unresolved_horizon"):
        primary_reason = "research_horizon_and_authority_specific_outcome_required"
    result = {
        "sleeve_id": sleeve,
        "objective_class": objective,
        "contract_sha256": contract.get("contract_sha256"),
        "timestamp_utc": str(timestamp or "")[:64],
        "epoch": ts,
        "symbol": symbol,
        "provider": provider,
        "instrument_type": instrument,
        "snapshot_id": snapshot,
        "source_candidate_id": candidate,
        "strategy": str(row.get("strategy") or meta.get("strategy") or "")[:256],
        "decision_id": str(row.get("decision_id") or meta.get("decision_id") or "")[
            :256
        ],
        "source_quality_label": str(row.get("source_quality_label") or "unknown")[:128],
        "quote_timestamp_utc": str(
            market.get("snapshot_ts_utc") or row.get("quote_timestamp_utc") or ""
        )[:64],
        "price": price,
        "primary_label": {"status": "pending", "value": None, "reason": primary_reason},
        "record_status": "quarantined" if reasons else "annotated_context_only",
        "reasons": reasons,
        "authority": AUTHORITY,
    }
    # Keep realized producer claims separate, with explicit non-verification.
    # They cannot become labels merely because a JSON field has a numeric value.
    reported = {}
    for key in (
        "post_cost_pnl_delta",
        "realized_pnl",
        "implementation_shortfall_bps",
        "recovery_seconds",
    ):
        value = number(row.get(key))
        if value is not None:
            reported[key] = value
    if reported:
        result["reported_outcomes"] = {
            "values": reported,
            "verification": "producer_claim_not_reverified",
        }
    return result


def price_context(
    anchor: dict, outcome: dict | None, horizon: int, contract: dict
) -> dict:
    result = {
        "horizon_seconds": horizon,
        "status": "pending",
        "value": None,
        "reason": "missing_mature_same_instrument_observation",
        "is_trade_pnl": False,
        "cost_adjustment": "not_applied",
        "training_eligible": False,
    }
    if not outcome:
        return result
    if (
        anchor.get("record_status") == "quarantined"
        or outcome.get("record_status") == "quarantined"
    ):
        result["reason"] = "quarantined_endpoint"
        return result
    if contract["objective_class"] == "control_only":
        result["reason"] = "market_label_forbidden_for_control"
        return result
    if not anchor.get("snapshot_id") or not outcome.get("snapshot_id"):
        result["reason"] = "missing_snapshot_lineage"
        return result
    for key in (
        "sleeve_id",
        "symbol",
        "provider",
        "instrument_type",
        "source_candidate_id",
    ):
        if not anchor.get(key) or anchor.get(key) != outcome.get(key):
            result["reason"] = "missing_or_mismatched_instrument_or_candidate_lineage"
            return result
    if anchor["snapshot_id"] == outcome["snapshot_id"]:
        result["reason"] = "reused_snapshot"
        return result
    start, end = number(anchor.get("epoch")), number(outcome.get("epoch"))
    if (
        start is None
        or end is None
        or not horizon <= end - start <= horizon + min(300, max(30, horizon * 0.01))
    ):
        result["reason"] = "horizon_not_observed_within_tolerance"
        return result
    for point in (anchor, outcome):
        quote_time = epoch(point.get("quote_timestamp_utc"))
        if quote_time is None or not 0 <= point["epoch"] - quote_time <= 120:
            result["reason"] = "missing_or_stale_quote_timestamp"
            return result
    p, q = number(anchor.get("price")), number(outcome.get("price"))
    value = number(q / p - 1) if p and q and p > 0 and q > 0 else None
    if value is None:
        result["reason"] = "invalid_price_return"
        return result
    result.update(
        status="observed_gross_price_context",
        value=value,
        reason="supplemental_not_primary_payoff",
        matured_at_utc=outcome["timestamp_utc"],
        outcome_snapshot_id=outcome["snapshot_id"],
        evidence_sha256=digest([anchor, outcome, horizon, contract["contract_sha256"]]),
    )
    return result
