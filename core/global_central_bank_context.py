from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REGISTRY_PATH = PROJECT_ROOT / "config" / "global_central_bank_registry_v1.json"

GLOBAL_CENTRAL_BANK_FEATURE_KEYS = (
    "global_central_bank_context_available_norm",
    "global_central_bank_important_coverage_norm",
    "global_central_bank_tier1_coverage_norm",
    "global_central_bank_policy_rate_coverage_norm",
    "global_central_bank_balance_sheet_coverage_norm",
    "global_central_bank_policy_easing_breadth_norm",
    "global_central_bank_policy_tightening_breadth_norm",
    "global_central_bank_policy_hold_breadth_norm",
    "global_central_bank_policy_impulse_30d_norm",
    "global_central_bank_policy_impulse_90d_norm",
    "global_central_bank_policy_divergence_norm",
    "global_central_bank_policy_change_intensity_30d_norm",
    "global_central_bank_policy_change_intensity_90d_norm",
    "global_central_bank_synchronized_easing_norm",
    "global_central_bank_synchronized_tightening_norm",
    "global_central_bank_balance_sheet_expansion_breadth_norm",
    "global_central_bank_balance_sheet_contraction_breadth_norm",
    "global_central_bank_balance_sheet_impulse_norm",
    "global_central_bank_usd_rate_advantage_norm",
    "global_central_bank_g5_easing_breadth_norm",
    "global_central_bank_g5_tightening_breadth_norm",
    "global_central_bank_em_easing_breadth_norm",
    "global_central_bank_em_tightening_breadth_norm",
    "global_central_bank_fx_framework_coverage_norm",
    "global_central_bank_usd_eur_policy_spread_norm",
    "global_central_bank_usd_jpy_policy_spread_norm",
    "global_central_bank_usd_gbp_policy_spread_norm",
    "global_central_bank_usd_cny_policy_spread_norm",
)

CENTRAL_BANK_CROSS_SOURCE_FEATURE_KEYS = (
    "central_bank_sync_available_norm",
    "central_bank_sync_coverage_norm",
    "central_bank_sync_point_in_time_norm",
    "central_bank_sync_lineage_coverage_norm",
    "central_bank_sync_fx_coverage_norm",
    "central_bank_sync_macro_coverage_norm",
    "central_bank_sync_liquidity_coverage_norm",
    "central_bank_sync_conflict_free_norm",
    "central_bank_sync_freshness_norm",
    "central_bank_policy_fx_confirmation_norm",
    "central_bank_policy_inflation_alignment_norm",
    "central_bank_policy_external_balance_alignment_norm",
    "central_bank_policy_liquidity_alignment_norm",
    "central_bank_policy_cross_asset_confirmation_norm",
    "central_bank_policy_divergence_signal_norm",
    "central_bank_policy_spillover_risk_norm",
)

GLOBAL_CENTRAL_BANK_MAX_ARTIFACT_AGE_HOURS = 48.0
CENTRAL_BANK_SYNC_MAX_ARTIFACT_AGE_HOURS = 24.0


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


def _feature_validation(features: Mapping[str, Any], keys: tuple[str, ...]) -> tuple[list[str], list[str]]:
    missing: list[str] = []
    invalid: list[str] = []
    for key in keys:
        if key not in features:
            missing.append(key)
            continue
        try:
            value = float(features[key])
        except (TypeError, ValueError):
            invalid.append(key)
            continue
        if not math.isfinite(value) or value < 0.0 or value > 1.0:
            invalid.append(key)
    return missing, invalid


def load_global_central_bank_registry(path: Path | None = None) -> dict[str, Any]:
    registry_path = Path(path or DEFAULT_REGISTRY_PATH)
    try:
        payload = json.loads(registry_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if not isinstance(payload, dict) or not isinstance(payload.get("banks"), list):
        return {}
    return payload


def bank_registry_by_id(registry: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for raw in registry.get("banks") if isinstance(registry.get("banks"), list) else []:
        if not isinstance(raw, Mapping):
            continue
        bank_id = str(raw.get("bank_id") or "").strip()
        if bank_id:
            out[bank_id] = dict(raw)
    return out


def _unwrap_global_context(snapshot: Mapping[str, Any]) -> Mapping[str, Any]:
    derived = _mapping(snapshot.get("derived"))
    nested = _mapping(derived.get("global_central_banks"))
    return nested if nested else snapshot


def assess_global_central_bank_context(
    snapshot: Mapping[str, Any],
    *,
    now_utc: datetime | None = None,
    max_age_hours: float = GLOBAL_CENTRAL_BANK_MAX_ARTIFACT_AGE_HOURS,
) -> dict[str, Any]:
    context = _unwrap_global_context(_mapping(snapshot))
    coverage = _mapping(context.get("coverage"))
    methodology = _mapping(context.get("methodology"))
    features = _mapping(context.get("global_features"))
    contract = _mapping(context.get("contract"))
    reasons: list[str] = []

    now = now_utc or datetime.now(timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    now = now.astimezone(timezone.utc)
    timestamp = _parse_timestamp(context.get("timestamp_utc"))
    age_hours = None
    if timestamp is None:
        reasons.append("timestamp_missing_or_invalid")
    else:
        age_hours = (now - timestamp).total_seconds() / 3600.0
        if age_hours < -0.1:
            reasons.append("artifact_timestamp_in_future")
        elif age_hours > max(float(max_age_hours), 1.0):
            reasons.append("artifact_stale")

    tier1_ratio = float(coverage.get("tier_1_coverage_ratio", 0.0) or 0.0)
    important_ratio = float(coverage.get("important_bank_coverage_ratio", 0.0) or 0.0)
    tier1_minimum = float(contract.get("tier_1_minimum_ratio", 0.8) or 0.8)
    important_minimum = float(contract.get("important_bank_minimum_ratio", 0.85) or 0.85)
    if tier1_ratio < tier1_minimum:
        reasons.append("tier_1_coverage_below_contract")
    if important_ratio < important_minimum:
        reasons.append("important_bank_coverage_below_contract")
    if bool(coverage.get("future_observation_selected", False)):
        reasons.append("future_observation_selected")
    if methodology.get("point_in_time_only") is not True:
        reasons.append("point_in_time_methodology_not_declared")
    if methodology.get("missing_values_are_not_zero_filled") is not True:
        reasons.append("missing_value_policy_not_fail_visible")
    if contract.get("live_execution_authority") is not False:
        reasons.append("live_execution_authority_not_locked")
    if contract.get("automatic_promotion_authority") is not False:
        reasons.append("automatic_promotion_authority_not_locked")
    if list(coverage.get("source_failures") or []):
        reasons.append("required_source_failure")

    missing_features, invalid_features = _feature_validation(features, GLOBAL_CENTRAL_BANK_FEATURE_KEYS)
    if missing_features:
        reasons.append("feature_schema_incomplete")
    if invalid_features:
        reasons.append("feature_values_invalid")

    return {
        "ready": not reasons,
        "reasons": list(dict.fromkeys(reasons)),
        "timestamp_utc": timestamp.isoformat() if timestamp is not None else None,
        "age_hours": round(age_hours, 6) if age_hours is not None else None,
        "maximum_age_hours": max(float(max_age_hours), 1.0),
        "tier_1_coverage_ratio": tier1_ratio,
        "tier_1_minimum_ratio": tier1_minimum,
        "important_bank_coverage_ratio": important_ratio,
        "important_bank_minimum_ratio": important_minimum,
        "missing_feature_keys": missing_features,
        "invalid_feature_keys": invalid_features,
        "future_observations_excluded": coverage.get("future_observations_excluded", {}),
        "future_observation_selected": bool(coverage.get("future_observation_selected", False)),
    }


def global_central_bank_context_ready(snapshot: Mapping[str, Any]) -> bool:
    return bool(assess_global_central_bank_context(snapshot).get("ready", False))


def _unwrap_sync_context(snapshot: Mapping[str, Any]) -> Mapping[str, Any]:
    derived = _mapping(snapshot.get("derived"))
    nested = _mapping(derived.get("central_bank_cross_source"))
    return nested if nested else snapshot


def assess_central_bank_cross_source_context(
    snapshot: Mapping[str, Any],
    *,
    now_utc: datetime | None = None,
    max_age_hours: float = CENTRAL_BANK_SYNC_MAX_ARTIFACT_AGE_HOURS,
) -> dict[str, Any]:
    context = _unwrap_sync_context(_mapping(snapshot))
    coverage = _mapping(context.get("coverage"))
    methodology = _mapping(context.get("methodology"))
    features = _mapping(context.get("global_features"))
    contract = _mapping(context.get("contract"))
    reasons: list[str] = []

    now = now_utc or datetime.now(timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    now = now.astimezone(timezone.utc)
    timestamp = _parse_timestamp(context.get("timestamp_utc"))
    age_hours = None
    if timestamp is None:
        reasons.append("timestamp_missing_or_invalid")
    else:
        age_hours = (now - timestamp).total_seconds() / 3600.0
        if age_hours < -0.1:
            reasons.append("artifact_timestamp_in_future")
        elif age_hours > max(float(max_age_hours), 1.0):
            reasons.append("artifact_stale")

    sync_ratio = float(coverage.get("synchronized_bank_coverage_ratio", 0.0) or 0.0)
    minimum_sync_ratio = float(contract.get("minimum_sync_coverage_ratio", 0.6) or 0.6)
    if sync_ratio < minimum_sync_ratio:
        reasons.append("synchronized_bank_coverage_below_contract")
    if int(coverage.get("hard_conflict_count", 0) or 0) > 0:
        reasons.append("hard_source_conflicts_present")
    if bool(coverage.get("future_observation_selected", False)):
        reasons.append("future_observation_selected")
    if methodology.get("point_in_time_join") is not True:
        reasons.append("point_in_time_join_not_declared")
    if methodology.get("distinct_cross_source_required") is not True:
        reasons.append("distinct_cross_source_requirement_not_declared")
    if methodology.get("missing_values_are_neutralized") is not False:
        reasons.append("missing_value_policy_not_fail_visible")
    if contract.get("live_execution_authority") is not False:
        reasons.append("live_execution_authority_not_locked")
    if contract.get("automatic_promotion_authority") is not False:
        reasons.append("automatic_promotion_authority_not_locked")
    synchronized_count = int(coverage.get("synchronized_ready_bank_count", 0) or 0)
    distinct_link_count = int(coverage.get("distinct_cross_source_link_count", 0) or 0)
    minimum_distinct = max(int(contract.get("minimum_distinct_cross_source_count", 1) or 1), 1)
    if synchronized_count > 0 and distinct_link_count < synchronized_count * minimum_distinct:
        reasons.append("distinct_cross_source_evidence_incomplete")

    missing_features, invalid_features = _feature_validation(features, CENTRAL_BANK_CROSS_SOURCE_FEATURE_KEYS)
    if missing_features:
        reasons.append("feature_schema_incomplete")
    if invalid_features:
        reasons.append("feature_values_invalid")

    return {
        "ready": not reasons,
        "reasons": list(dict.fromkeys(reasons)),
        "timestamp_utc": timestamp.isoformat() if timestamp is not None else None,
        "age_hours": round(age_hours, 6) if age_hours is not None else None,
        "maximum_age_hours": max(float(max_age_hours), 1.0),
        "synchronized_bank_coverage_ratio": sync_ratio,
        "minimum_sync_coverage_ratio": minimum_sync_ratio,
        "hard_conflict_count": int(coverage.get("hard_conflict_count", 0) or 0),
        "distinct_cross_source_link_count": distinct_link_count,
        "minimum_distinct_cross_source_count": minimum_distinct,
        "missing_feature_keys": missing_features,
        "invalid_feature_keys": invalid_features,
        "future_observation_selected": bool(coverage.get("future_observation_selected", False)),
    }


def central_bank_cross_source_context_ready(snapshot: Mapping[str, Any]) -> bool:
    return bool(assess_central_bank_cross_source_context(snapshot).get("ready", False))
