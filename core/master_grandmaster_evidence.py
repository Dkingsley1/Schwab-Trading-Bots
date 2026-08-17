"""Deterministic sleeve-master and grand-master evidence synthesis."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from typing import Any, Iterable, Mapping

from core.regime_taxonomy import (
    build_regime_metadata_access,
    classify_regime_profile,
    evaluate_regime_compatibility,
)


EXPECTED_SOURCE_IDS = {
    "bot_organization_health",
    "bot_hierarchy",
    "regime_context",
    "paper_truth",
    "profitability_evidence",
    "source_verification",
    "runtime_throttle",
    "account_positions",
    "execution_calibration",
    "sleeve_profitability",
}
EXPECTED_SCORE_WEIGHTS = {
    "classification_confidence",
    "regime_axis_coverage",
    "regime_compatibility",
    "paper_truth",
    "source_evidence",
    "economic_evidence",
}
FORBIDDEN_STATUS_VALUES = {"blocked", "critical", "degraded", "error", "failed", "missing"}
SAFETY_FALSE_FIELDS = (
    "direct_paper_order_authority",
    "direct_live_order_authority",
    "order_payload_creation",
    "registry_mutation_authority",
    "source_code_mutation_authority",
    "execution_flag_mutation_authority",
    "global_halt_override_authority",
    "broker_truth_override_authority",
    "automatic_allocation_authority",
    "automatic_promotion_authority",
    "profitability_guaranteed",
)


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float(default)
    return result if math.isfinite(result) else float(default)


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return int(default)


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return min(max(_safe_float(value), low), high)


def _slug(value: Any) -> str:
    text = re.sub(r"[^a-z0-9]+", "_", str(value or "").strip().lower())
    return re.sub(r"_+", "_", text).strip("_")


def _ordered_unique(values: Iterable[Any]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        item = str(value or "").strip()
        if not item or item in seen:
            continue
        seen.add(item)
        result.append(item)
    return result


def _canonical_hash(value: Any) -> str:
    raw = json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _parse_timestamp(value: Any) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = f"{text[:-1]}+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _payload_timestamp(payload: Mapping[str, Any]) -> str:
    for key in ("timestamp_utc", "generated_at_utc", "updated_utc", "generated_utc"):
        value = str(payload.get(key) or "").strip()
        if value:
            return value
    return ""


def _grade(score: float, *, structurally_ready: bool = True) -> str:
    if not structurally_ready:
        return "F"
    value = _clamp(score)
    if value >= 0.98:
        return "A+"
    if value >= 0.9:
        return "A"
    if value >= 0.8:
        return "B"
    if value >= 0.7:
        return "C"
    if value >= 0.6:
        return "D"
    return "F"


def _grade_at_least(grade: str, floor: str) -> bool:
    ranks = {"A+": 6, "A": 5, "B": 4, "C": 3, "D": 2, "F": 1}
    return ranks.get(str(grade or "").upper(), 0) >= ranks.get(str(floor or "").upper(), 0)


def validate_policy(policy: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    if _safe_int(policy.get("schema_version")) != 1:
        errors.append("master_grandmaster_policy_schema_version_invalid")
    if str(policy.get("operating_mode") or "") != "evidence_synthesis_shadow_only":
        errors.append("master_grandmaster_policy_mode_not_shadow_only")
    if not str(policy.get("required_regime_model_id") or "").strip():
        errors.append("master_grandmaster_required_regime_model_id_missing")

    sources = _as_dict(policy.get("source_contracts"))
    if set(sources) != EXPECTED_SOURCE_IDS:
        errors.append("master_grandmaster_source_contracts_invalid")
    for source_id, contract in sources.items():
        row = _as_dict(contract)
        if _safe_float(row.get("max_age_minutes"), 0.0) <= 0.0:
            errors.append(f"master_grandmaster_{source_id}_max_age_invalid")
        if not isinstance(row.get("synthesis_required"), bool):
            errors.append(f"master_grandmaster_{source_id}_synthesis_requirement_invalid")
        if not isinstance(row.get("promotion_required"), bool):
            errors.append(f"master_grandmaster_{source_id}_promotion_requirement_invalid")

    thresholds = _as_dict(policy.get("thresholds"))
    if not 1 <= _safe_int(thresholds.get("maximum_sleeve_master_packets"), 0) <= 512:
        errors.append("master_grandmaster_packet_cap_invalid")
    if not 1 <= _safe_int(thresholds.get("maximum_review_examples_per_sleeve"), 0) <= 25:
        errors.append("master_grandmaster_review_example_cap_invalid")
    for key in (
        "minimum_hierarchy_coverage_ratio",
        "minimum_regime_compatibility_score",
        "maximum_master_review_ratio_ready",
        "maximum_unknown_profile_ratio_ready",
        "minimum_regime_axis_coverage_for_promotion",
        "minimum_regime_axis_specificity_for_promotion",
    ):
        value = _safe_float(thresholds.get(key), -1.0)
        if not 0.0 <= value <= 1.0:
            errors.append(f"master_grandmaster_{key}_invalid")
    for key in (
        "minimum_paper_truth_score",
        "minimum_source_evidence_score",
        "minimum_profitability_economic_evidence_score",
    ):
        value = _safe_float(thresholds.get(key), -1.0)
        if not 0.0 <= value <= 100.0:
            errors.append(f"master_grandmaster_{key}_invalid")
    if _safe_int(thresholds.get("minimum_known_context_axes"), 0) < 1:
        errors.append("master_grandmaster_minimum_known_context_axes_invalid")
    future_skew = _safe_float(thresholds.get("maximum_future_clock_skew_minutes"), -1.0)
    if not 0.0 <= future_skew <= 60.0:
        errors.append("master_grandmaster_maximum_future_clock_skew_invalid")
    if _safe_int(thresholds.get("minimum_independent_execution_samples"), -1) < 0:
        errors.append("master_grandmaster_minimum_independent_execution_samples_invalid")
    if str(thresholds.get("minimum_human_review_master_grade") or "") not in {
        "A+",
        "A",
        "B",
        "C",
        "D",
        "F",
    }:
        errors.append("master_grandmaster_minimum_human_review_master_grade_invalid")

    weights = _as_dict(policy.get("master_score_weights"))
    if set(weights) != EXPECTED_SCORE_WEIGHTS:
        errors.append("master_grandmaster_score_weights_invalid")
    if any(_safe_float(value, -1.0) < 0.0 for value in weights.values()):
        errors.append("master_grandmaster_score_weight_negative")
    if abs(sum(_safe_float(value) for value in weights.values()) - 1.0) > 1e-9:
        errors.append("master_grandmaster_score_weights_do_not_sum_to_one")

    recommendations = _as_dict(policy.get("recommendation_contract"))
    allowed = set(_ordered_unique(_as_list(recommendations.get("allowed_recommendations"))))
    forbidden = set(_ordered_unique(_as_list(recommendations.get("forbidden_recommendations"))))
    if not allowed or not forbidden or allowed.intersection(forbidden):
        errors.append("master_grandmaster_recommendation_contract_invalid")

    safety = _as_dict(policy.get("safety_contract"))
    for key in SAFETY_FALSE_FIELDS:
        if safety.get(key) is not False:
            errors.append(f"master_grandmaster_safety_{key}_must_be_false")
    if safety.get("human_authorization_required_for_live") is not True:
        errors.append("master_grandmaster_human_live_authorization_must_be_true")
    return _ordered_unique(errors)


def build_observed_regime_context(
    regime_payload: Mapping[str, Any],
    regime_model: Mapping[str, Any],
) -> dict[str, Any]:
    raw_state = str(regime_payload.get("regime_state") or "").strip()
    explicit_axes = _as_dict(regime_payload.get("regime_axes"))
    row: dict[str, Any] = {
        "bot_id": "observed_runtime_regime",
        "bot_role": "signal_sub_bot",
        "preferred_regimes": [raw_state] if raw_state else [],
    }
    if explicit_axes:
        row["regime_axes"] = explicit_axes
    profile = classify_regime_profile(
        row=row,
        module_spec={},
        classification_text="observed runtime market regime",
        raw_role="signal_sub_bot",
        role_id="signal",
        sub_sleeve_id="regime_and_forecasting",
        horizon_id="runtime",
        model=regime_model,
    )
    axes: dict[str, list[str]] = {}
    unknown_axes: list[str] = []
    wildcard_axes: list[str] = []
    for axis_id, raw_axis in _as_dict(profile.get("axes")).items():
        axis = _as_dict(raw_axis)
        if bool(axis.get("not_applicable", False)):
            continue
        values = [str(item) for item in _as_list(axis.get("values")) if str(item).strip()]
        if not values:
            continue
        axes[axis_id] = values
        if bool(axis.get("unknown", False)):
            unknown_axes.append(axis_id)
        if bool(axis.get("wildcard", False)):
            wildcard_axes.append(axis_id)
    known_axes = [axis_id for axis_id in axes if axis_id not in unknown_axes]
    specific_axes = [axis_id for axis_id in known_axes if axis_id not in wildcard_axes]
    return {
        "schema_version": 1,
        "model_id": str(regime_model.get("model_id") or ""),
        "raw_regime_state": raw_state,
        "source_status": str(regime_payload.get("overall_status") or ""),
        "stance_label": str(regime_payload.get("stance_label") or ""),
        "stance_score": _safe_float(regime_payload.get("stance_score")),
        "axes": axes,
        "known_axes": known_axes,
        "unknown_axes": unknown_axes,
        "wildcard_axes": wildcard_axes,
        "known_axis_count": len(known_axes),
        "specific_axis_count": len(specific_axes),
        "profile_id": str(profile.get("profile_id") or ""),
        "profile_confidence": _safe_float(profile.get("profile_confidence")),
    }


def _source_receipt(payload: Mapping[str, Any]) -> str:
    for key in (
        "assignment_receipt_sha256",
        "receipt_sha256",
        "evidence_receipt_sha256",
        "source_receipt_sha256",
    ):
        value = str(payload.get(key) or "").strip()
        if value:
            return value
    evidence_epoch = _as_dict(payload.get("evidence_epoch"))
    for key in ("receipt_sha256", "id"):
        value = str(evidence_epoch.get(key) or "").strip()
        if value:
            return value
    return _canonical_hash(payload)


def assess_sources(
    inputs: Mapping[str, Mapping[str, Any]],
    policy: Mapping[str, Any],
    *,
    now: datetime,
) -> dict[str, dict[str, Any]]:
    checks: dict[str, dict[str, Any]] = {}
    maximum_future_skew = _safe_float(
        _as_dict(policy.get("thresholds")).get("maximum_future_clock_skew_minutes"),
        5.0,
    )
    for source_id, raw_contract in _as_dict(policy.get("source_contracts")).items():
        contract = _as_dict(raw_contract)
        payload = _as_dict(inputs.get(source_id))
        timestamp = _payload_timestamp(payload)
        parsed = _parse_timestamp(timestamp)
        signed_age_minutes = (now - parsed).total_seconds() / 60.0 if parsed else None
        age_minutes = max(signed_age_minutes, 0.0) if signed_age_minutes is not None else None
        future_skew_minutes = max(-signed_age_minutes, 0.0) if signed_age_minutes is not None else None
        timestamp_valid = bool(
            parsed is not None
            and future_skew_minutes is not None
            and future_skew_minutes <= maximum_future_skew
        )
        available = bool(payload)
        fresh = bool(
            available
            and timestamp_valid
            and age_minutes is not None
            and age_minutes <= _safe_float(contract.get("max_age_minutes"), 0.0)
        )
        status = str(payload.get("overall_status") or payload.get("status") or "").strip().lower()
        explicit_ok = payload.get("ok")
        semantic_ready = bool(
            available
            and explicit_ok is not False
            and status not in FORBIDDEN_STATUS_VALUES
        )
        checks[source_id] = {
            "timestamp_utc": timestamp,
            "age_minutes": round(age_minutes, 6) if age_minutes is not None else None,
            "future_skew_minutes": (
                round(future_skew_minutes, 6) if future_skew_minutes is not None else None
            ),
            "timestamp_valid": timestamp_valid,
            "max_age_minutes": _safe_float(contract.get("max_age_minutes")),
            "available": available,
            "fresh": fresh,
            "status": status or "unknown",
            "explicit_ok": explicit_ok if isinstance(explicit_ok, bool) else None,
            "semantic_ready": semantic_ready,
            "synthesis_required": bool(contract.get("synthesis_required", False)),
            "promotion_required": bool(contract.get("promotion_required", False)),
            "receipt_sha256": _source_receipt(payload) if payload else "",
        }
    return checks


def _profile_axis_counts(assignments: Iterable[Mapping[str, Any]]) -> tuple[int, int, int]:
    quality = 0
    known = 0
    specific = 0
    for row in assignments:
        profile = _as_dict(row.get("regime_profile"))
        quality += _safe_int(
            profile.get("quality_axis_slot_count"),
            len(_as_list(profile.get("quality_axes"))),
        )
        known += _safe_int(
            profile.get("known_axis_slot_count"),
            len(_as_list(profile.get("known_axes"))),
        )
        specific += _safe_int(
            profile.get("specific_axis_slot_count"),
            len(_as_list(profile.get("specific_axes"))),
        )
    return quality, known, specific


def _axis_value_counts(assignments: Iterable[Mapping[str, Any]]) -> dict[str, dict[str, int]]:
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    for row in assignments:
        axes = _as_dict(_as_dict(row.get("regime_profile")).get("axes"))
        for axis_id, raw_axis in axes.items():
            axis = _as_dict(raw_axis)
            if bool(axis.get("not_applicable", False)):
                continue
            for value in _as_list(axis.get("values")):
                item = str(value or "").strip()
                if item:
                    counts[axis_id][item] += 1
    return {
        axis_id: dict(sorted(values.items(), key=lambda item: (-item[1], item[0]))[:5])
        for axis_id, values in sorted(counts.items())
    }


def _paper_scorecards(payload: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in _as_list(payload.get("sleeve_scorecards")):
        if not isinstance(row, dict):
            continue
        profile = str(row.get("profile") or "").strip()
        if profile:
            result[profile] = dict(row)
    return result


def _profitability_scorecards(payload: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for key in ("top_sleeves", "bottom_sleeves"):
        for row in _as_list(payload.get(key)):
            if not isinstance(row, dict):
                continue
            profile = str(row.get("profile") or "").strip()
            if profile:
                result[profile] = dict(row)
    return result


def _paper_summary(row: Mapping[str, Any]) -> dict[str, Any]:
    if not row:
        return {"available": False}
    return {
        "available": True,
        "status": str(row.get("status") or ""),
        "data_status": str(row.get("data_status") or ""),
        "day_utc": str(row.get("day_utc") or ""),
        "executions": _safe_int(row.get("executions")),
        "execution_realism_score": _safe_float(row.get("execution_realism_score")),
        "ending_net_pnl_total": round(_safe_float(row.get("ending_net_pnl_total")), 6),
        "pnl_per_execution": round(_safe_float(row.get("pnl_per_execution")), 6),
        "win_rate": row.get("win_rate"),
        "reasons": _ordered_unique(_as_list(row.get("reasons"))),
    }


def _profitability_summary(row: Mapping[str, Any]) -> dict[str, Any]:
    if not row:
        return {"available": False}
    return {
        "available": True,
        "data_status": str(row.get("data_status") or ""),
        "day_utc": str(row.get("day_utc") or ""),
        "executions": _safe_int(row.get("executions")),
        "net_pnl_total": round(_safe_float(row.get("net_pnl_total")), 6),
        "raw_grade": str(row.get("raw_grade") or row.get("grade") or ""),
        "control_grade": str(row.get("control_grade") or ""),
        "control_action": str(row.get("control_action") or ""),
        "harvest_action": str(row.get("harvest_action") or ""),
    }


def _master_packet(
    sleeve_id: str,
    assignments: list[Mapping[str, Any]],
    *,
    policy: Mapping[str, Any],
    regime_model: Mapping[str, Any],
    regime_context: Mapping[str, Any],
    global_components: Mapping[str, float],
    paper_scorecard: Mapping[str, Any],
    profitability_scorecard: Mapping[str, Any],
) -> dict[str, Any]:
    thresholds = _as_dict(policy.get("thresholds"))
    weights = _as_dict(policy.get("master_score_weights"))
    active = [row for row in assignments if bool(row.get("active", False))]
    signal = [
        row
        for row in assignments
        if str(row.get("regime_scope") or "") in {"market_signal", "hybrid"}
    ]
    shadow = [row for row in assignments if bool(row.get("shadow_vote_eligible", False))]
    review = [row for row in assignments if bool(row.get("needs_review", False))]
    unknown_profiles = [
        row
        for row in assignments
        if _as_list(_as_dict(row.get("regime_profile")).get("unknown_axes"))
    ]
    quality_slots, known_slots, specific_slots = _profile_axis_counts(assignments)
    axis_coverage = known_slots / max(quality_slots, 1)
    axis_specificity = specific_slots / max(quality_slots, 1)
    classification_confidence = sum(
        _safe_float(row.get("classification_confidence")) for row in assignments
    ) / max(len(assignments), 1)
    review_ratio = len(review) / max(len(assignments), 1)
    unknown_ratio = len(unknown_profiles) / max(len(assignments), 1)
    metadata_access_ready_count = sum(
        1
        for row in assignments
        if bool(
            (
                _as_dict(row.get("regime_metadata_access"))
                or build_regime_metadata_access(
                    _as_dict(row.get("regime_profile")),
                    regime_model,
                )
            ).get("access_ready", False)
        )
    )
    metadata_access_ratio = metadata_access_ready_count / max(len(assignments), 1)

    compatibility_rows: list[dict[str, Any]] = []
    minimum_context_axes = _safe_int(thresholds.get("minimum_known_context_axes"), 2)
    context_ready = _safe_int(regime_context.get("known_axis_count"), 0) >= minimum_context_axes
    if context_ready:
        for row in signal:
            profile = _as_dict(row.get("regime_profile"))
            if not profile:
                continue
            result = evaluate_regime_compatibility(
                profile,
                {"axes": _as_dict(regime_context.get("axes"))},
                regime_model,
            )
            compatibility_rows.append(
                {
                    "bot_id": str(row.get("bot_id") or ""),
                    "compatible": bool(result.get("compatible", False)),
                    "score": _safe_float(result.get("score")),
                    "reason": str(result.get("reason") or ""),
                    "hard_mismatch_axis_ids": _as_list(result.get("hard_mismatch_axis_ids")),
                    "scenario_partition_applied": bool(
                        result.get("scenario_partition_applied", False)
                    ),
                    "selected_scenario_id": str(result.get("selected_scenario_id") or ""),
                }
            )
    compatible_count = sum(1 for row in compatibility_rows if row["compatible"])
    incompatible_count = len(compatibility_rows) - compatible_count
    mean_compatibility = (
        sum(_safe_float(row.get("score")) for row in compatibility_rows)
        / len(compatibility_rows)
        if compatibility_rows
        else 1.0 if not signal else 0.5
    )
    reason_counts = Counter(str(row.get("reason") or "unknown") for row in compatibility_rows)

    cluster_counts = Counter(
        str(row.get("correlation_cluster_id") or "")
        for row in shadow
        if str(row.get("correlation_cluster_id") or "")
    )
    largest_cluster = max(cluster_counts.values(), default=0)
    largest_cluster_share = largest_cluster / max(len(shadow), 1)
    components = {
        "classification_confidence": _clamp(classification_confidence),
        "regime_axis_coverage": _clamp(axis_coverage),
        "regime_compatibility": _clamp(mean_compatibility),
        "paper_truth": _clamp(global_components.get("paper_truth", 0.0)),
        "source_evidence": _clamp(global_components.get("source_evidence", 0.0)),
        "economic_evidence": _clamp(global_components.get("economic_evidence", 0.0)),
    }
    score = sum(components[key] * _safe_float(weights.get(key)) for key in EXPECTED_SCORE_WEIGHTS)
    grade = _grade(score)
    if not active:
        status = "blocked_no_active_members"
    elif signal and not context_ready:
        status = "context_thin"
    elif review_ratio > _safe_float(thresholds.get("maximum_master_review_ratio_ready"), 0.25):
        status = "needs_metadata_evidence"
    elif unknown_ratio > _safe_float(thresholds.get("maximum_unknown_profile_ratio_ready"), 0.35):
        status = "needs_regime_evidence"
    elif compatibility_rows and (
        mean_compatibility
        < _safe_float(thresholds.get("minimum_regime_compatibility_score"), 0.55)
    ):
        status = "regime_guarded"
    else:
        status = "ready_shadow"

    recommendations: list[str] = ["continue_paper_collection"] if active else []
    if not context_ready or unknown_profiles:
        recommendations.append("collect_missing_regime_evidence")
    if incompatible_count:
        recommendations.append("review_incompatible_profiles")
    paper_summary = _paper_summary(paper_scorecard)
    profit_summary = _profitability_summary(profitability_scorecard)
    if (
        _safe_float(paper_summary.get("ending_net_pnl_total")) < 0.0
        or _safe_float(profit_summary.get("net_pnl_total")) < 0.0
    ):
        recommendations.append("investigate_negative_post_cost_results")
    if review_ratio > 0.5 and active:
        recommendations.append("prioritize_guarded_retraining")
    allowed = set(
        _ordered_unique(
            _as_list(_as_dict(policy.get("recommendation_contract")).get("allowed_recommendations"))
        )
    )
    recommendations = [item for item in _ordered_unique(recommendations) if item in allowed]
    review_limit = _safe_int(thresholds.get("maximum_review_examples_per_sleeve"), 8)
    return {
        "schema_version": 1,
        "master_id": f"sleeve_master_v2_{_slug(sleeve_id)}",
        "sleeve_id": sleeve_id,
        "reports_to": "grand_master_evidence_v2",
        "status": status,
        "grade": grade,
        "evidence_score": round(score * 100.0, 4),
        "bot_count": len(assignments),
        "active_bot_count": len(active),
        "signal_profile_count": len(signal),
        "shadow_vote_eligible_count": len(shadow),
        "sub_sleeve_count": len({str(row.get("sub_sleeve_id") or "") for row in assignments}),
        "cohort_count": len({str(row.get("cohort_id") or "") for row in assignments}),
        "correlation_cluster_count": len(cluster_counts),
        "largest_correlation_cluster_share": round(largest_cluster_share, 6),
        "classification_confidence": round(classification_confidence, 6),
        "review_count": len(review),
        "review_ratio": round(review_ratio, 6),
        "unknown_profile_count": len(unknown_profiles),
        "unknown_profile_ratio": round(unknown_ratio, 6),
        "regime_metadata_access_ready_count": metadata_access_ready_count,
        "regime_metadata_access_ratio": round(metadata_access_ratio, 6),
        "regime_axis_coverage_ratio": round(axis_coverage, 6),
        "regime_axis_specificity_ratio": round(axis_specificity, 6),
        "regime_context_evaluated": context_ready,
        "regime_compatibility": {
            "evaluated_bot_count": len(compatibility_rows),
            "compatible_bot_count": compatible_count,
            "incompatible_bot_count": incompatible_count,
            "mean_score": round(mean_compatibility, 6),
            "reason_counts": dict(sorted(reason_counts.items())),
            "hard_mismatch_examples": [
                row
                for row in compatibility_rows
                if row.get("hard_mismatch_axis_ids")
            ][:review_limit],
        },
        "regime_axis_value_counts": _axis_value_counts(assignments),
        "scope_counts": dict(
            sorted(Counter(str(row.get("regime_scope") or "") for row in assignments).items())
        ),
        "role_counts": dict(
            sorted(Counter(str(row.get("role_id") or "") for row in assignments).items())
        ),
        "score_components": {key: round(value, 6) for key, value in sorted(components.items())},
        "paper_truth": paper_summary,
        "profitability": profit_summary,
        "review_examples": [
            {
                "bot_id": str(row.get("bot_id") or ""),
                "review_reasons": _as_list(row.get("review_reasons")),
                "regime_profile_id": str(row.get("regime_profile_id") or ""),
            }
            for row in sorted(
                review,
                key=lambda item: (
                    _safe_float(item.get("classification_confidence")),
                    str(item.get("bot_id") or ""),
                ),
            )[:review_limit]
        ],
        "recommendations": recommendations,
        "authority": {
            "advisory_only": True,
            "paper_order_authority": False,
            "live_order_authority": False,
            "order_payload_created": False,
            "registry_mutation_authority": False,
            "automatic_promotion_authority": False,
        },
    }


def synthesize_master_grandmaster_evidence(
    *,
    policy: Mapping[str, Any],
    regime_model: Mapping[str, Any],
    bot_organization_health: Mapping[str, Any],
    bot_hierarchy: Mapping[str, Any],
    regime_payload: Mapping[str, Any],
    paper_truth: Mapping[str, Any],
    profitability_evidence: Mapping[str, Any],
    source_verification: Mapping[str, Any],
    runtime_throttle: Mapping[str, Any],
    account_positions: Mapping[str, Any],
    execution_calibration: Mapping[str, Any],
    sleeve_profitability: Mapping[str, Any],
    now: datetime | None = None,
) -> dict[str, Any]:
    evaluated_at = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    inputs = {
        "bot_organization_health": bot_organization_health,
        "bot_hierarchy": bot_hierarchy,
        "regime_context": regime_payload,
        "paper_truth": paper_truth,
        "profitability_evidence": profitability_evidence,
        "source_verification": source_verification,
        "runtime_throttle": runtime_throttle,
        "account_positions": account_positions,
        "execution_calibration": execution_calibration,
        "sleeve_profitability": sleeve_profitability,
    }
    policy_errors = validate_policy(policy)
    source_checks = assess_sources(inputs, policy, now=evaluated_at)
    assignments = [
        row for row in _as_list(bot_hierarchy.get("assignments")) if isinstance(row, dict)
    ]
    expected_count = _safe_int(bot_hierarchy.get("assignment_count"), len(assignments))
    organization_count = _safe_int(bot_organization_health.get("registry_bot_count"), 0)
    hierarchy_receipt = str(bot_hierarchy.get("assignment_receipt_sha256") or "")
    organization_receipt = str(bot_organization_health.get("assignment_receipt_sha256") or "")
    required_model_id = str(policy.get("required_regime_model_id") or "")
    hierarchy_model_id = str(bot_hierarchy.get("regime_model_id") or "")
    regime_context = build_observed_regime_context(regime_payload, regime_model)

    blockers = list(policy_errors)
    if not assignments:
        blockers.append("master_grandmaster_hierarchy_assignments_missing")
    if expected_count != len(assignments) or organization_count != len(assignments):
        blockers.append("master_grandmaster_hierarchy_assignment_count_mismatch")
    if not hierarchy_receipt or hierarchy_receipt != organization_receipt:
        blockers.append("master_grandmaster_hierarchy_receipt_mismatch")
    if hierarchy_model_id != required_model_id or str(regime_model.get("model_id") or "") != required_model_id:
        blockers.append("master_grandmaster_regime_model_mismatch")
    hierarchy_coverage = _safe_float(bot_organization_health.get("organization_coverage_ratio"))
    if hierarchy_coverage < _safe_float(
        _as_dict(policy.get("thresholds")).get("minimum_hierarchy_coverage_ratio"),
        1.0,
    ):
        blockers.append("master_grandmaster_hierarchy_coverage_below_floor")
    if bot_organization_health.get("ok") is not True:
        blockers.append("master_grandmaster_bot_organization_not_ready")
    for source_id, check in source_checks.items():
        if bool(check.get("synthesis_required")) and (
            not bool(check.get("available")) or not bool(check.get("fresh"))
        ):
            blockers.append(f"master_grandmaster_required_source_not_fresh:{source_id}")
    paper_truth_score = _safe_float(paper_truth.get("score"))
    paper_truth_ready = bool(
        paper_truth.get("ok") is True
        and str(paper_truth.get("overall_status") or "") == "ready"
        and paper_truth_score
        >= _safe_float(_as_dict(policy.get("thresholds")).get("minimum_paper_truth_score"), 95.0)
    )
    throttle_ready = bool(
        runtime_throttle.get("ok") is True
        and str(runtime_throttle.get("overall_status") or "") not in FORBIDDEN_STATUS_VALUES
    )
    blockers = _ordered_unique(blockers)
    operational_holds = _ordered_unique(
        [
            *(
                ["master_grandmaster_paper_truth_not_ready"]
                if not paper_truth_ready
                else []
            ),
            *(
                ["master_grandmaster_runtime_capacity_not_ready"]
                if not throttle_ready
                else []
            ),
        ]
    )

    global_components = {
        "paper_truth": _clamp(paper_truth_score / 100.0),
        "source_evidence": _clamp(
            _safe_float(source_verification.get("source_evidence_score")) / 100.0
        ),
        "economic_evidence": _clamp(
            _safe_float(profitability_evidence.get("economic_evidence_score")) / 100.0
        ),
    }
    paper_cards = _paper_scorecards(paper_truth)
    profit_cards = _profitability_scorecards(sleeve_profitability)
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in assignments:
        grouped[str(row.get("sleeve_id") or "unassigned")].append(row)
    packet_cap = _safe_int(
        _as_dict(policy.get("thresholds")).get("maximum_sleeve_master_packets"),
        256,
    )
    if len(grouped) > packet_cap:
        blockers = _ordered_unique([*blockers, "master_grandmaster_sleeve_packet_cap_exceeded"])
    sleeve_masters = [
        _master_packet(
            sleeve_id,
            rows,
            policy=policy,
            regime_model=regime_model,
            regime_context=regime_context,
            global_components=global_components,
            paper_scorecard=paper_cards.get(sleeve_id, {}),
            profitability_scorecard=profit_cards.get(sleeve_id, {}),
        )
        for sleeve_id, rows in sorted(grouped.items())[:packet_cap]
    ]

    thresholds = _as_dict(policy.get("thresholds"))
    source_evidence_ready = bool(
        bool(_as_dict(source_checks.get("source_verification")).get("fresh"))
        and source_verification.get("ok") is True
        and _safe_float(source_verification.get("source_evidence_score"))
        >= _safe_float(thresholds.get("minimum_source_evidence_score"), 95.0)
    )
    regime_context_ready = bool(
        bool(_as_dict(source_checks.get("regime_context")).get("fresh"))
        and str(regime_payload.get("overall_status") or "") == "ready"
        and _safe_int(regime_context.get("known_axis_count"))
        >= _safe_int(thresholds.get("minimum_known_context_axes"), 2)
    )
    profitability_ready = bool(
        bool(_as_dict(source_checks.get("profitability_evidence")).get("fresh"))
        and profitability_evidence.get("live_promotion_ready") is True
        and _safe_float(profitability_evidence.get("economic_evidence_score"))
        >= _safe_float(thresholds.get("minimum_profitability_economic_evidence_score"), 90.0)
    )
    independent_samples = _safe_int(execution_calibration.get("independent_samples"), 0)
    execution_evidence_ready = bool(
        bool(_as_dict(source_checks.get("execution_calibration")).get("fresh"))
        and execution_calibration.get("independent_evidence_ready") is True
        and independent_samples
        >= _safe_int(thresholds.get("minimum_independent_execution_samples"), 30)
    )
    metadata_ready = bool(
        _safe_float(bot_organization_health.get("regime_axis_coverage_ratio"))
        >= _safe_float(thresholds.get("minimum_regime_axis_coverage_for_promotion"), 0.9)
        and _safe_float(bot_organization_health.get("regime_axis_specificity_ratio"))
        >= _safe_float(thresholds.get("minimum_regime_axis_specificity_for_promotion"), 0.8)
    )
    account_awareness_ready = bool(
        bool(_as_dict(source_checks.get("account_positions")).get("fresh"))
        and account_positions.get("ok") is True
    )
    fresh_promotion_sources = all(
        bool(check.get("available"))
        and bool(check.get("fresh"))
        and bool(check.get("semantic_ready"))
        for check in source_checks.values()
        if bool(check.get("promotion_required"))
    )
    master_grade_floor = str(thresholds.get("minimum_human_review_master_grade") or "B")
    master_quality_ready = bool(
        sleeve_masters
        and all(_grade_at_least(str(row.get("grade") or "F"), master_grade_floor) for row in sleeve_masters)
    )
    promotion_gates = {
        "fresh_promotion_sources": fresh_promotion_sources,
        "paper_truth_ready": paper_truth_ready,
        "runtime_capacity_ready": throttle_ready,
        "source_evidence_ready": source_evidence_ready,
        "multi_axis_regime_context_ready": regime_context_ready,
        "regime_metadata_mature": metadata_ready,
        "profitability_evidence_ready": profitability_ready,
        "independent_execution_evidence_ready": execution_evidence_ready,
        "account_position_awareness_ready": account_awareness_ready,
        "sleeve_master_quality_ready": master_quality_ready,
    }
    promotion_blockers = [key for key, ready in promotion_gates.items() if not ready]
    paper_coordination_ready = bool(not blockers and not operational_holds)
    human_live_review_evidence_ready = bool(
        paper_coordination_ready and not promotion_blockers
    )
    weighted_master_score = (
        sum(_safe_float(row.get("evidence_score")) * max(_safe_int(row.get("bot_count")), 1) for row in sleeve_masters)
        / sum(max(_safe_int(row.get("bot_count")), 1) for row in sleeve_masters)
        if sleeve_masters
        else 0.0
    )
    status_counts = Counter(str(row.get("status") or "unknown") for row in sleeve_masters)
    grade_counts = Counter(str(row.get("grade") or "F") for row in sleeve_masters)
    recommended_posture = (
        "abstain_until_broker_truth_recovers"
        if not paper_truth_ready
        else "abstain_until_runtime_capacity_recovers"
        if not throttle_ready
        else "hold_for_human_promotion_review"
        if human_live_review_evidence_ready
        else "continue_paper_collection"
    )
    allowed_recommendations = set(
        _ordered_unique(
            _as_list(
                _as_dict(policy.get("recommendation_contract")).get("allowed_recommendations")
            )
        )
    )
    grand_recommendations = [recommended_posture]
    if not regime_context_ready or not metadata_ready:
        grand_recommendations.append("collect_missing_regime_evidence")
    if not source_evidence_ready or not fresh_promotion_sources:
        grand_recommendations.append("refresh_stale_sources")
    if not profitability_ready:
        grand_recommendations.append("investigate_negative_post_cost_results")
    grand_recommendations = [
        item for item in _ordered_unique(grand_recommendations) if item in allowed_recommendations
    ]
    authority = {
        "mode": "advisory_shadow_only",
        "paper_order_authority": False,
        "live_order_authority": False,
        "order_payload_created": False,
        "registry_mutation_authority": False,
        "source_code_mutation_authority": False,
        "execution_flag_mutation_authority": False,
        "global_halt_override_authority": False,
        "broker_truth_override_authority": False,
        "automatic_allocation_authority": False,
        "automatic_promotion_authority": False,
        "human_authorization_required_for_live": True,
    }
    grand_status = (
        "blocked_integrity"
        if blockers
        else "operational_hold"
        if operational_holds
        else "ready_with_evidence_debt"
        if promotion_blockers
        else "ready_for_human_review"
    )
    grand_master = {
        "schema_version": 1,
        "grand_master_id": "grand_master_evidence_v2",
        "status": grand_status,
        "grade": _grade(weighted_master_score / 100.0, structurally_ready=not blockers),
        "structural_grade": "A+" if not blockers else "F",
        "evidence_score": round(weighted_master_score, 4),
        "paper_coordination_ready": paper_coordination_ready,
        "human_live_review_evidence_ready": human_live_review_evidence_ready,
        "automatic_live_promotion_allowed": False,
        "recommended_posture": recommended_posture,
        "sleeve_master_count": len(sleeve_masters),
        "sleeve_master_status_counts": dict(sorted(status_counts.items())),
        "sleeve_master_grade_counts": dict(sorted(grade_counts.items())),
        "promotion_gates": promotion_gates,
        "promotion_blockers": promotion_blockers,
        "operational_holds": operational_holds,
        "account_position_count": _safe_int(account_positions.get("position_count"), 0),
        "current_regime_context": regime_context,
        "lowest_evidence_sleeves": [
            {
                "sleeve_id": str(row.get("sleeve_id") or ""),
                "status": str(row.get("status") or ""),
                "grade": str(row.get("grade") or ""),
                "evidence_score": row.get("evidence_score"),
                "review_count": row.get("review_count"),
            }
            for row in sorted(
                sleeve_masters,
                key=lambda item: (
                    _safe_float(item.get("evidence_score")),
                    str(item.get("sleeve_id") or ""),
                ),
            )[:20]
        ],
        "recommendations": grand_recommendations,
        "authority": authority,
    }
    evidence_epoch_input = {
        "policy_id": str(policy.get("policy_id") or ""),
        "source_receipts": {
            key: str(value.get("receipt_sha256") or "")
            for key, value in sorted(source_checks.items())
        },
        "hierarchy_assignment_receipt_sha256": hierarchy_receipt,
        "regime_context_profile_id": str(regime_context.get("profile_id") or ""),
    }
    return {
        "timestamp_utc": evaluated_at.isoformat(),
        "schema_version": 1,
        "policy_id": str(policy.get("policy_id") or ""),
        "operating_mode": str(policy.get("operating_mode") or ""),
        "ok": not blockers,
        "overall_status": grand_status,
        "grade": grand_master["grade"],
        "structural_grade": grand_master["structural_grade"],
        "paper_coordination_ready": paper_coordination_ready,
        "human_live_review_evidence_ready": human_live_review_evidence_ready,
        "sleeve_master_count": len(sleeve_masters),
        "organized_bot_count": len(assignments),
        "source_checks": source_checks,
        "integrity_blockers": blockers,
        "operational_holds": operational_holds,
        "promotion_blockers": promotion_blockers,
        "grand_master": grand_master,
        "sleeve_masters": sleeve_masters,
        "authority": authority,
        "evidence_epoch": {
            "id": f"master-grandmaster-v2:{_canonical_hash(evidence_epoch_input)[:16]}",
            "receipt_sha256": _canonical_hash(evidence_epoch_input),
            **evidence_epoch_input,
        },
    }
