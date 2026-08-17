"""Shadow-only hierarchical vote aggregation with bounded correlated influence."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Any, Iterable, Mapping

from core.regime_taxonomy import evaluate_regime_compatibility


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


def _weighted_mean(rows: Iterable[Mapping[str, Any]]) -> tuple[float, float]:
    items = list(rows)
    total_weight = sum(max(_safe_float(row.get("weight")), 0.0) for row in items)
    if total_weight <= 0.0:
        return 0.0, 0.0
    score = sum(
        _safe_float(row.get("score")) * max(_safe_float(row.get("weight")), 0.0)
        for row in items
    ) / total_weight
    return score, total_weight


def _disagreement(rows: Iterable[Mapping[str, Any]], mean: float) -> float:
    items = list(rows)
    total_weight = sum(max(_safe_float(row.get("weight")), 0.0) for row in items)
    if total_weight <= 0.0:
        return 1.0
    variance = sum(
        max(_safe_float(row.get("weight")), 0.0)
        * (_safe_float(row.get("score")) - mean) ** 2
        for row in items
    ) / total_weight
    return round(min(max(math.sqrt(max(variance, 0.0)), 0.0), 1.0), 6)


def aggregate_shadow_votes(
    votes: Iterable[Mapping[str, Any]],
    assignments: Mapping[str, Mapping[str, Any]],
    policy: Mapping[str, Any],
    *,
    regime_context: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Aggregate research votes without creating paper or live execution authority."""

    ensemble = _as_dict(policy.get("ensemble_policy"))
    score_min = _safe_float(ensemble.get("score_minimum"), -1.0)
    score_max = _safe_float(ensemble.get("score_maximum"), 1.0)
    confidence_floor = _safe_float(ensemble.get("minimum_confidence"), 0.25)
    bot_cap = _safe_float(ensemble.get("max_bot_weight"), 0.15)
    cluster_cap = _safe_float(ensemble.get("max_correlation_cluster_weight"), 0.25)
    sub_sleeve_cap = _safe_float(ensemble.get("max_sub_sleeve_weight"), 0.4)
    sleeve_cap = _safe_float(ensemble.get("max_sleeve_weight"), 0.55)
    threshold = _safe_float(ensemble.get("decision_threshold"), 0.15)
    disagreement_limit = _safe_float(ensemble.get("maximum_disagreement"), 0.72)
    minimum_sub_sleeves = max(int(_safe_float(ensemble.get("minimum_distinct_sub_sleeves"), 2)), 1)
    minimum_sleeves = max(int(_safe_float(ensemble.get("minimum_distinct_sleeves"), 1)), 1)
    regime_model = _as_dict(policy.get("regime_model"))
    compatibility_policy = _as_dict(regime_model.get("compatibility_policy"))
    apply_regime_context = regime_context is not None
    weight_by_regime = bool(
        compatibility_policy.get("weight_shadow_votes_by_compatibility", True)
    )

    accepted: list[dict[str, Any]] = []
    regime_evidence: list[dict[str, Any]] = []
    excluded = Counter()
    seen_vote_ids: set[str] = set()
    for index, raw in enumerate(votes):
        vote = dict(raw)
        bot_id = str(vote.get("bot_id") or "").strip()
        vote_id = str(vote.get("vote_id") or f"{bot_id}:{index}").strip()
        if not bot_id or bot_id not in assignments:
            excluded["missing_organization_assignment"] += 1
            continue
        if vote_id in seen_vote_ids:
            excluded["duplicate_vote_id"] += 1
            continue
        seen_vote_ids.add(vote_id)
        assignment = _as_dict(assignments.get(bot_id))
        if not bool(assignment.get("shadow_vote_eligible", False)):
            excluded["bot_not_shadow_vote_eligible"] += 1
            continue
        regime_compatibility: dict[str, Any] | None = None
        if apply_regime_context:
            profile = _as_dict(assignment.get("regime_profile"))
            if not profile:
                excluded["missing_regime_profile"] += 1
                regime_evidence.append(
                    {
                        "vote_id": vote_id,
                        "bot_id": bot_id,
                        "compatible": False,
                        "reason": "missing_regime_profile",
                        "score": 0.0,
                    }
                )
                continue
            regime_compatibility = evaluate_regime_compatibility(
                profile,
                _as_dict(regime_context),
                regime_model,
            )
            regime_evidence.append(
                {
                    "vote_id": vote_id,
                    "bot_id": bot_id,
                    "profile_id": str(profile.get("profile_id") or ""),
                    "compatible": bool(regime_compatibility.get("compatible", False)),
                    "reason": str(regime_compatibility.get("reason") or ""),
                    "score": _safe_float(regime_compatibility.get("score")),
                    "compared_axis_count": int(
                        _safe_float(regime_compatibility.get("compared_axis_count"), 0.0)
                    ),
                    "scenario_partition_applied": bool(
                        regime_compatibility.get("scenario_partition_applied", False)
                    ),
                    "selected_scenario_id": str(
                        regime_compatibility.get("selected_scenario_id") or ""
                    ),
                    "scenario_count": int(
                        _safe_float(regime_compatibility.get("scenario_count"), 0.0)
                    ),
                    "metadata_access_ready": bool(
                        _as_dict(regime_compatibility.get("metadata_access")).get(
                            "access_ready", False
                        )
                    ),
                    "metadata_context_ready": bool(
                        _as_dict(regime_compatibility.get("metadata_access")).get(
                            "context_ready", False
                        )
                    ),
                    "metadata_observed_axis_ids": _as_list(
                        _as_dict(regime_compatibility.get("metadata_access")).get(
                            "observed_axis_ids"
                        )
                    ),
                    "metadata_context_receipt_sha256": str(
                        _as_dict(regime_compatibility.get("metadata_access")).get(
                            "context_receipt_sha256"
                        )
                        or ""
                    ),
                }
            )
            if not bool(regime_compatibility.get("compatible", False)):
                excluded[str(regime_compatibility.get("reason") or "regime_incompatible")] += 1
                continue
        score = _safe_float(vote.get("score"), math.nan)
        confidence = _safe_float(vote.get("confidence"), 0.0)
        raw_weight = max(_safe_float(vote.get("weight"), 1.0), 0.0)
        if not math.isfinite(score) or not score_min <= score <= score_max:
            excluded["score_out_of_range"] += 1
            continue
        if confidence < confidence_floor:
            excluded["confidence_below_floor"] += 1
            continue
        if raw_weight <= 0.0:
            excluded["nonpositive_weight"] += 1
            continue
        regime_weight = (
            _safe_float(regime_compatibility.get("score"), 0.0)
            if regime_compatibility is not None and weight_by_regime
            else 1.0
        )
        effective_weight = min(raw_weight * confidence, bot_cap) * regime_weight
        if effective_weight <= 0.0:
            excluded["nonpositive_regime_adjusted_weight"] += 1
            continue
        accepted.append(
            {
                "vote_id": vote_id,
                "bot_id": bot_id,
                "score": score,
                "weight": effective_weight,
                "regime_compatibility_score": round(regime_weight, 6),
                "sleeve_id": str(assignment.get("sleeve_id") or ""),
                "sub_sleeve_id": str(assignment.get("sub_sleeve_id") or ""),
                "cohort_id": str(assignment.get("cohort_id") or ""),
                "correlation_cluster_id": str(
                    vote.get("correlation_cluster_id")
                    or assignment.get("correlation_cluster_id")
                    or bot_id
                ),
            }
        )

    cluster_groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in accepted:
        cluster_groups[
            (row["sleeve_id"], row["sub_sleeve_id"], row["correlation_cluster_id"])
        ].append(row)
    clusters: list[dict[str, Any]] = []
    for (sleeve_id, sub_sleeve_id, cluster_id), members in sorted(cluster_groups.items()):
        score, _ = _weighted_mean(members)
        clusters.append(
            {
                "sleeve_id": sleeve_id,
                "sub_sleeve_id": sub_sleeve_id,
                "correlation_cluster_id": cluster_id,
                "member_count": len(members),
                "score": round(score, 8),
                "weight": min(max(row["weight"] for row in members), cluster_cap),
            }
        )

    sub_groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in clusters:
        sub_groups[(row["sleeve_id"], row["sub_sleeve_id"])].append(row)
    sub_sleeves: list[dict[str, Any]] = []
    for (sleeve_id, sub_sleeve_id), members in sorted(sub_groups.items()):
        score, total_weight = _weighted_mean(members)
        sub_sleeves.append(
            {
                "sleeve_id": sleeve_id,
                "sub_sleeve_id": sub_sleeve_id,
                "cluster_count": len(members),
                "score": round(score, 8),
                "weight": min(total_weight, sub_sleeve_cap),
            }
        )

    sleeve_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in sub_sleeves:
        sleeve_groups[row["sleeve_id"]].append(row)
    sleeves: list[dict[str, Any]] = []
    for sleeve_id, members in sorted(sleeve_groups.items()):
        score, total_weight = _weighted_mean(members)
        sleeves.append(
            {
                "sleeve_id": sleeve_id,
                "sub_sleeve_count": len(members),
                "score": round(score, 8),
                "weight": min(total_weight, sleeve_cap),
            }
        )

    aggregate_score, aggregate_weight = _weighted_mean(sleeves)
    disagreement = _disagreement(clusters, aggregate_score)
    diversity_ready = bool(
        len(sub_sleeves) >= minimum_sub_sleeves and len(sleeves) >= minimum_sleeves
    )
    if not accepted:
        action = "HOLD"
        reason = "no_eligible_votes"
    elif not diversity_ready:
        action = "HOLD"
        reason = "insufficient_hierarchical_diversity"
    elif disagreement > disagreement_limit:
        action = "HOLD"
        reason = "cross_cell_disagreement_above_limit"
    elif aggregate_score >= threshold:
        action = "BUY"
        reason = "shadow_consensus_above_threshold"
    elif aggregate_score <= -threshold:
        action = "SELL"
        reason = "shadow_consensus_below_threshold"
    else:
        action = "HOLD"
        reason = "shadow_consensus_inside_abstention_band"

    return {
        "schema_version": 1,
        "mode": "shadow_only",
        "ok": True,
        "action": action,
        "reason": reason,
        "score": round(aggregate_score, 8),
        "aggregate_weight": round(aggregate_weight, 8),
        "disagreement": disagreement,
        "diversity_ready": diversity_ready,
        "accepted_vote_count": len(accepted),
        "excluded_vote_count": sum(excluded.values()),
        "excluded_reasons": dict(sorted(excluded.items())),
        "correlation_cluster_count": len(clusters),
        "sub_sleeve_count": len(sub_sleeves),
        "sleeve_count": len(sleeves),
        "regime_context_applied": apply_regime_context,
        "regime_compatible_vote_count": sum(
            1 for row in regime_evidence if bool(row.get("compatible", False))
        ),
        "regime_incompatible_vote_count": sum(
            1 for row in regime_evidence if not bool(row.get("compatible", False))
        ),
        "regime_metadata_access_ready_vote_count": sum(
            1 for row in regime_evidence if bool(row.get("metadata_access_ready", False))
        ),
        "regime_metadata_context_ready_vote_count": sum(
            1 for row in regime_evidence if bool(row.get("metadata_context_ready", False))
        ),
        "regime_compatibility_evidence": regime_evidence,
        "regime_contract": {
            "model_id": str(regime_model.get("model_id") or ""),
            "mode": "shadow_only",
            "weight_by_compatibility": weight_by_regime,
            "metadata_access_version": str(
                _as_dict(regime_model.get("metadata_access_contract")).get("version") or ""
            ),
            "metadata_access_mode": str(
                _as_dict(regime_model.get("metadata_access_contract")).get("mode") or ""
            ),
            "paper_execution_authority": False,
            "live_execution_authority": False,
        },
        "clusters": clusters,
        "sub_sleeves": sub_sleeves,
        "sleeves": sleeves,
        "authority": {
            "research_recommendation_only": True,
            "paper_execution_authority": False,
            "live_execution_authority": False,
            "order_payload_created": False,
        },
    }
