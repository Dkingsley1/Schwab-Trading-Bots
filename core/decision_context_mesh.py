from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "decision_context_mesh_v1.json"
DECISION_CONTEXT_MESH_MAX_ARTIFACT_AGE_HOURS = 24.0

PLANE_SIGNAL_FEATURE_KEYS = (
    "context_fiscal_liquidity_signal_norm",
    "context_funding_stress_signal_norm",
    "context_cross_border_capital_signal_norm",
    "context_positioning_crowding_signal_norm",
    "context_securities_lending_signal_norm",
    "context_credit_curve_signal_norm",
    "context_volatility_surface_signal_norm",
    "context_passive_mechanical_flow_signal_norm",
    "context_market_calendar_signal_norm",
    "context_supply_chain_inventory_signal_norm",
    "context_estimates_dispersion_signal_norm",
    "context_capacity_market_impact_signal_norm",
)

DECISION_CONTEXT_MESH_FEATURE_KEYS = PLANE_SIGNAL_FEATURE_KEYS + (
    "context_mesh_available_norm",
    "context_mesh_macro_grade_norm",
    "context_mesh_micro_grade_norm",
    "context_mesh_coverage_norm",
    "context_mesh_confidence_norm",
    "context_mesh_freshness_norm",
    "context_mesh_lineage_coverage_norm",
    "context_mesh_cross_verification_norm",
)


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


def _safe_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def percentage_grade(percent: float) -> str:
    score = max(0.0, min(float(percent), 100.0))
    thresholds = (
        (97.0, "A+"),
        (93.0, "A"),
        (90.0, "A-"),
        (87.0, "B+"),
        (83.0, "B"),
        (80.0, "B-"),
        (77.0, "C+"),
        (73.0, "C"),
        (70.0, "C-"),
        (67.0, "D+"),
        (63.0, "D"),
        (60.0, "D-"),
    )
    return next((grade for threshold, grade in thresholds if score >= threshold), "F")


def load_decision_context_mesh_config(path: Path | None = None) -> dict[str, Any]:
    config_path = Path(path or DEFAULT_CONFIG_PATH)
    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _plane_rows(snapshot: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    rows = snapshot.get("planes")
    if isinstance(rows, list):
        return [row for row in rows if isinstance(row, Mapping)]
    return []


def assess_decision_context_mesh(
    snapshot: Mapping[str, Any],
    *,
    now_utc: datetime | None = None,
    max_age_hours: float = DECISION_CONTEXT_MESH_MAX_ARTIFACT_AGE_HOURS,
) -> dict[str, Any]:
    context = _mapping(snapshot)
    contract = _mapping(context.get("contract"))
    methodology = _mapping(context.get("methodology"))
    coverage = _mapping(context.get("coverage"))
    features = _mapping((_mapping(context.get("derived"))).get("global_features"))
    planes = _plane_rows(context)
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

    required_plane_count = int(contract.get("required_plane_count", 12) or 12)
    unique_plane_ids = {str(row.get("plane_id") or "") for row in planes if str(row.get("plane_id") or "")}
    if len(unique_plane_ids) != required_plane_count:
        reasons.append("plane_schema_incomplete")

    invalid_plane_scores: list[str] = []
    below_minimum: list[str] = []
    minimum_plane_score = float(contract.get("minimum_plane_score_pct", 70.0) or 70.0)
    for row in planes:
        plane_id = str(row.get("plane_id") or "")
        score = _safe_float(row.get("score_pct"))
        if score is None or score < 0.0 or score > 100.0:
            invalid_plane_scores.append(plane_id)
        elif score < minimum_plane_score:
            below_minimum.append(plane_id)
    if invalid_plane_scores:
        reasons.append("plane_scores_invalid")
    if below_minimum:
        reasons.append("plane_scores_below_minimum")

    missing_features: list[str] = []
    invalid_features: list[str] = []
    for key in DECISION_CONTEXT_MESH_FEATURE_KEYS:
        if key not in features:
            missing_features.append(key)
            continue
        value = _safe_float(features.get(key))
        if value is None or value < 0.0 or value > 1.0:
            invalid_features.append(key)
    if missing_features:
        reasons.append("feature_schema_incomplete")
    if invalid_features:
        reasons.append("feature_values_invalid")

    if methodology.get("point_in_time_only") is not True:
        reasons.append("point_in_time_methodology_not_declared")
    if methodology.get("future_observations_rejected") is not True:
        reasons.append("future_rejection_not_declared")
    if methodology.get("missing_values_are_not_zero_filled") is not True:
        reasons.append("missing_value_policy_not_fail_visible")
    if bool(coverage.get("future_observation_selected", False)):
        reasons.append("future_observation_selected")
    for authority in ("paper_execution_authority", "live_execution_authority", "automatic_promotion_authority"):
        if contract.get(authority) is not False:
            reasons.append(f"{authority}_not_locked")

    macro_scores = [
        float(row.get("score_pct"))
        for row in planes
        if row.get("plane_class") == "macro" and _safe_float(row.get("score_pct")) is not None
    ]
    micro_scores = [
        float(row.get("score_pct"))
        for row in planes
        if row.get("plane_class") == "micro" and _safe_float(row.get("score_pct")) is not None
    ]
    macro_pct = round(sum(macro_scores) / max(len(macro_scores), 1), 3)
    micro_pct = round(sum(micro_scores) / max(len(micro_scores), 1), 3)
    if macro_pct < float(contract.get("minimum_macro_score_pct", 80.0) or 80.0):
        reasons.append("macro_score_below_contract")
    if micro_pct < float(contract.get("minimum_micro_score_pct", 80.0) or 80.0):
        reasons.append("micro_score_below_contract")

    return {
        "ready": not reasons,
        "reasons": list(dict.fromkeys(reasons)),
        "timestamp_utc": timestamp.isoformat() if timestamp is not None else None,
        "age_hours": round(age_hours, 6) if age_hours is not None else None,
        "maximum_age_hours": max(float(max_age_hours), 1.0),
        "plane_count": len(unique_plane_ids),
        "required_plane_count": required_plane_count,
        "macro_percentage": macro_pct,
        "macro_grade": percentage_grade(macro_pct),
        "micro_percentage": micro_pct,
        "micro_grade": percentage_grade(micro_pct),
        "minimum_plane_score_pct": minimum_plane_score,
        "planes_below_minimum": below_minimum,
        "invalid_plane_scores": invalid_plane_scores,
        "missing_feature_keys": missing_features,
        "invalid_feature_keys": invalid_features,
        "future_observations_excluded": coverage.get("future_observations_excluded", {}),
        "future_observation_selected": bool(coverage.get("future_observation_selected", False)),
    }


def decision_context_mesh_ready(snapshot: Mapping[str, Any]) -> bool:
    return bool(assess_decision_context_mesh(snapshot).get("ready", False))
