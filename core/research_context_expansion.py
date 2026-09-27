from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import Any

COLLECTOR_DEFINITIONS: tuple[dict[str, Any], ...] = (
    {
        "collector_id": "cross_asset_breadth_context",
        "title": "Cross-Asset Breadth Context",
        "source_kind": "internal_derived",
        "cadence": "intraday",
        "resource_class": "warm",
        "max_age_minutes": 180,
        "refresh_after_minutes": 30,
        "capabilities": (
            "cross_asset_correlation",
            "cross_asset_dispersion",
            "market_breadth",
            "risk_on_off_state",
            "factor_leadership",
            "sector_rotation",
            "cross_asset_dislocation",
        ),
        "feature_keys": (
            "research_cross_asset_available_norm",
            "research_cross_asset_advance_decline_norm",
            "research_cross_asset_correlation_norm",
            "research_cross_asset_dispersion_norm",
            "research_cross_asset_dislocation_norm",
            "research_factor_leadership_norm",
            "research_risk_on_state_norm",
            "research_sector_rotation_norm",
            "research_cross_asset_relative_strength_norm",
        ),
    },
    {
        "collector_id": "tape_liquidity_context",
        "title": "Tape and Liquidity Context",
        "source_kind": "internal_derived",
        "cadence": "intraday",
        "resource_class": "warm",
        "max_age_minutes": 120,
        "refresh_after_minutes": 20,
        "capabilities": (
            "bid_ask_spread",
            "consolidated_tape_quality",
            "quote_age",
            "realized_volatility",
            "vwap_state",
            "liquidity_regime",
        ),
        "feature_keys": (
            "research_tape_available_norm",
            "research_tape_liquidity_regime_norm",
            "research_tape_quality_norm",
            "research_tape_quote_freshness_norm",
            "research_tape_realized_volatility_norm",
            "research_tape_spread_regime_norm",
            "research_tape_vwap_state_norm",
        ),
    },
    {
        "collector_id": "options_greeks_surface_context",
        "title": "Options Greeks and Surface Context",
        "source_kind": "broker_native",
        "cadence": "intraday",
        "resource_class": "warm",
        "max_age_minutes": 240,
        "refresh_after_minutes": 30,
        "capabilities": (
            "options_chain",
            "implied_volatility_surface",
            "volatility_skew",
            "volatility_term_structure",
            "option_greeks",
            "option_open_interest",
            "realized_volatility",
            "volatility_risk_premium",
        ),
        "feature_keys": (
            "research_options_available_norm",
            "research_options_greeks_norm",
            "research_options_iv_surface_norm",
            "research_options_open_interest_norm",
            "research_options_skew_norm",
            "research_options_term_structure_norm",
            "research_options_vrp_norm",
        ),
    },
    {
        "collector_id": "futures_curve_context",
        "title": "Futures Curve Context",
        "source_kind": "broker_native",
        "cadence": "intraday",
        "resource_class": "warm",
        "max_age_minutes": 240,
        "refresh_after_minutes": 30,
        "capabilities": (
            "futures_term_structure",
            "futures_basis",
            "roll_yield",
            "futures_open_interest",
            "futures_volume_migration",
            "expiry_state",
            "carry_state",
            "calendar_spreads",
            "spot_futures_dislocation",
        ),
        "feature_keys": (
            "research_futures_available_norm",
            "research_futures_basis_norm",
            "research_futures_calendar_spread_norm",
            "research_futures_expiry_norm",
            "research_futures_open_interest_norm",
            "research_futures_roll_yield_norm",
            "research_futures_term_structure_norm",
            "research_futures_volume_migration_norm",
        ),
    },
    {
        "collector_id": "earnings_event_context",
        "title": "Earnings Event Context",
        "source_kind": "official_public_mesh",
        "cadence": "daily",
        "resource_class": "warm",
        "max_age_minutes": 2880,
        "refresh_after_minutes": 360,
        "capabilities": (
            "earnings_calendar",
            "reported_earnings",
            "company_guidance",
            "estimate_revisions",
            "estimate_dispersion",
        ),
        "feature_keys": (
            "research_earnings_available_norm",
            "research_earnings_calendar_proximity_norm",
            "research_earnings_guidance_activity_norm",
            "research_earnings_report_activity_norm",
            "research_estimate_dispersion_norm",
            "research_estimate_revision_direction_norm",
        ),
    },
    {
        "collector_id": "portfolio_factor_risk_context",
        "title": "Portfolio Factor Risk Context",
        "source_kind": "broker_native",
        "cadence": "intraday",
        "resource_class": "warm",
        "max_age_minutes": 120,
        "refresh_after_minutes": 20,
        "capabilities": (
            "portfolio_exposure",
            "concentration_risk",
            "portfolio_beta",
            "factor_exposure",
            "correlation_risk",
            "portfolio_liquidity",
        ),
        "feature_keys": (
            "research_portfolio_available_norm",
            "research_portfolio_beta_norm",
            "research_portfolio_concentration_norm",
            "research_portfolio_correlation_risk_norm",
            "research_portfolio_factor_exposure_norm",
            "research_portfolio_liquidity_norm",
            "research_portfolio_net_exposure_norm",
            "research_portfolio_weight_norm",
        ),
    },
    {
        "collector_id": "fixed_income_trace_context",
        "title": "FINRA Fixed-Income TRACE Context",
        "source_kind": "official",
        "cadence": "daily",
        "resource_class": "warm",
        "max_age_minutes": 4320,
        "refresh_after_minutes": 720,
        "capabilities": (
            "market_breadth",
            "turnover",
            "liquidity_regime",
            "rates_credit_regime",
        ),
        "feature_keys": (
            "research_trace_available_norm",
            "research_trace_breadth_norm",
            "research_trace_credit_regime_norm",
            "research_trace_liquidity_norm",
            "research_trace_turnover_norm",
        ),
    },
    {
        "collector_id": "bis_global_liquidity_context",
        "title": "BIS Global Liquidity Context",
        "source_kind": "official",
        "cadence": "daily",
        "resource_class": "cold",
        "max_age_minutes": 10080,
        "refresh_after_minutes": 1440,
        "capabilities": (
            "global_liquidity_regime",
            "cross_border_capital_flows",
            "bank_credit_conditions",
        ),
        "feature_keys": (
            "research_bis_available_norm",
            "research_bis_bank_credit_conditions_norm",
            "research_bis_country_coverage_norm",
            "research_bis_cross_border_flow_norm",
            "research_bis_global_liquidity_impulse_norm",
        ),
    },
)

COLLECTOR_BY_ID = {row["collector_id"]: row for row in COLLECTOR_DEFINITIONS}
COLLECTOR_IDS = tuple(COLLECTOR_BY_ID)
RUNTIME_RESEARCH_CONTEXT_FEATURE_KEYS = frozenset(
    key for collector in COLLECTOR_DEFINITIONS for key in collector["feature_keys"]
)


def collector_definition(collector_id: str) -> dict[str, Any]:
    return COLLECTOR_BY_ID[str(collector_id)]


def _parse_timestamp(raw: Any) -> datetime | None:
    text = str(raw or "").strip()
    if not text:
        return None
    try:
        value = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def research_context_ready(
    payload: Any,
    collector_id: str,
    *,
    now: datetime | None = None,
) -> bool:
    if not isinstance(payload, dict) or not bool(payload.get("ok", False)):
        return False
    definition = COLLECTOR_BY_ID.get(str(collector_id))
    if definition is None or str(payload.get("collector_id") or "") != str(
        collector_id
    ):
        return False

    observed_now = now or datetime.now(timezone.utc)
    if observed_now.tzinfo is None:
        observed_now = observed_now.replace(tzinfo=timezone.utc)
    timestamp = _parse_timestamp(payload.get("timestamp_utc"))
    if timestamp is None:
        return False
    age_seconds = (observed_now.astimezone(timezone.utc) - timestamp).total_seconds()
    if (
        age_seconds < -300.0
        or age_seconds > float(definition["max_age_minutes"]) * 60.0
    ):
        return False

    authority = payload.get("authority_contract")
    if not isinstance(authority, dict) or authority.get("observation_only") is not True:
        return False
    if any(
        authority.get(key) is not False
        for key in (
            "paper_execution_authority",
            "live_execution_authority",
            "automatic_promotion_authority",
            "registry_mutation_authority",
        )
    ):
        return False

    evidence = payload.get("evidence_contract")
    if not isinstance(evidence, dict) or any(
        evidence.get(key) is not True
        for key in (
            "missing_dimensions_are_omitted_not_zero_filled",
            "source_timestamp_is_preserved",
            "unsupported_level2_is_never_inferred",
            "point_in_time_only",
        )
    ):
        return False

    receipt = str(payload.get("snapshot_receipt_sha256") or "").strip()
    unsigned = {
        key: value for key, value in payload.items() if key != "snapshot_receipt_sha256"
    }
    raw = json.dumps(
        unsigned, ensure_ascii=True, sort_keys=True, separators=(",", ":"), default=str
    )
    expected = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    return bool(receipt and receipt == expected)
