from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Any, Mapping


CENTRAL_BANK_LIQUIDITY_FEATURE_KEYS = (
    "central_bank_liquidity_available_norm",
    "central_bank_liquidity_source_coverage_norm",
    "fed_total_assets_level_norm",
    "fed_total_assets_impulse_norm",
    "fed_reserve_balances_level_norm",
    "fed_reserve_balances_impulse_norm",
    "fed_rrp_drain_level_norm",
    "fed_rrp_drain_impulse_norm",
    "fed_repo_injection_level_norm",
    "fed_tga_drain_level_norm",
    "fed_tga_drain_impulse_norm",
    "fed_net_liquidity_impulse_norm",
    "fed_liquidity_expansion_norm",
    "fed_liquidity_tightening_norm",
    "fed_central_bank_swap_usage_norm",
    "fed_sofr_level_norm",
    "fed_effr_level_norm",
    "fed_iorb_level_norm",
    "fed_sofr_effr_spread_norm",
    "fed_effr_iorb_spread_norm",
    "fed_policy_corridor_width_norm",
    "fed_funding_stress_norm",
    "fed_financial_conditions_tightness_norm",
    "fed_adjusted_financial_conditions_tightness_norm",
    "fed_financial_stress_norm",
)

CENTRAL_BANK_LIQUIDITY_MAX_ARTIFACT_AGE_HOURS = 24.0


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _parse_timestamp(value: Any) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _unwrap_context(snapshot: Mapping[str, Any]) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    derived = _mapping(snapshot.get("derived"))
    nested = _mapping(derived.get("central_bank_liquidity"))
    context = nested if nested else snapshot
    status = _mapping(_mapping(_mapping(snapshot.get("status")).get("sources")).get("central_bank_liquidity"))
    return context, status


def assess_central_bank_liquidity_context(
    snapshot: Mapping[str, Any],
    *,
    now_utc: datetime | None = None,
    max_age_hours: float = CENTRAL_BANK_LIQUIDITY_MAX_ARTIFACT_AGE_HOURS,
) -> dict[str, Any]:
    context, status = _unwrap_context(_mapping(snapshot))
    coverage = _mapping(context.get("coverage"))
    methodology = _mapping(context.get("methodology"))
    features = _mapping(context.get("global_features"))
    reasons: list[str] = []

    timestamp = _parse_timestamp(context.get("timestamp_utc"))
    now = now_utc or datetime.now(timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    now = now.astimezone(timezone.utc)
    age_hours = None
    if timestamp is None:
        reasons.append("timestamp_missing_or_invalid")
    else:
        age_hours = (now - timestamp).total_seconds() / 3600.0
        if age_hours < -0.1:
            reasons.append("artifact_timestamp_in_future")
        elif age_hours > max(float(max_age_hours), 1.0):
            reasons.append("artifact_stale")

    coverage_ratio = float(coverage.get("required_coverage_ratio", 0.0) or 0.0)
    if coverage_ratio < 1.0:
        reasons.append("required_series_coverage_incomplete")
    for key in ("missing_required_series", "stale_required_series", "unusable_required_series"):
        if list(coverage.get(key) or []):
            reasons.append(key)
    if bool(coverage.get("future_observation_selected", False)):
        reasons.append("future_observation_selected")
    if methodology.get("point_in_time_only") is not True:
        reasons.append("point_in_time_methodology_not_declared")

    missing_features: list[str] = []
    invalid_features: list[str] = []
    for key in CENTRAL_BANK_LIQUIDITY_FEATURE_KEYS:
        if key not in features:
            missing_features.append(key)
            continue
        try:
            value = float(features[key])
        except (TypeError, ValueError):
            invalid_features.append(key)
            continue
        if not math.isfinite(value) or value < 0.0 or value > 1.0:
            invalid_features.append(key)
    if missing_features:
        reasons.append("feature_schema_incomplete")
    if invalid_features:
        reasons.append("feature_values_invalid")
    if status and not bool(status.get("ok", False)):
        reasons.append("official_macro_source_contract_not_ok")

    return {
        "ready": not reasons,
        "reasons": list(dict.fromkeys(reasons)),
        "timestamp_utc": timestamp.isoformat() if timestamp is not None else None,
        "age_hours": round(age_hours, 6) if age_hours is not None else None,
        "maximum_age_hours": max(float(max_age_hours), 1.0),
        "required_coverage_ratio": coverage_ratio,
        "missing_feature_keys": missing_features,
        "invalid_feature_keys": invalid_features,
        "future_observations_excluded": _mapping(coverage.get("future_observations_excluded")),
        "future_observation_selected": bool(coverage.get("future_observation_selected", False)),
    }


def central_bank_liquidity_context_ready(snapshot: Mapping[str, Any]) -> bool:
    return bool(assess_central_bank_liquidity_context(snapshot).get("ready", False))
