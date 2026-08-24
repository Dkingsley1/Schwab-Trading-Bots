from __future__ import annotations

import math
from typing import Any, Mapping, Sequence


def _correlation(
    covariance: Mapping[str, Mapping[str, float]], left: str, right: str
) -> float:
    left_var = float(dict(covariance.get(left) or {}).get(left, 0.0))
    right_var = float(dict(covariance.get(right) or {}).get(right, 0.0))
    cross = float(dict(covariance.get(left) or {}).get(right, 0.0))
    if left_var <= 0 or right_var <= 0:
        return 1.0
    return cross / math.sqrt(left_var * right_var)


def _capped_weights(
    scores: Mapping[str, float], budget: float, cap: float
) -> dict[str, float]:
    remaining = set(scores)
    weights = {key: 0.0 for key in scores}
    remaining_budget = budget
    while remaining and remaining_budget > 1e-12:
        score_sum = sum(max(0.0, scores[key]) for key in remaining)
        if score_sum <= 0:
            break
        capped: set[str] = set()
        for key in sorted(remaining):
            proposed = remaining_budget * max(0.0, scores[key]) / score_sum
            if proposed >= cap:
                weights[key] = cap
                remaining_budget -= cap
                capped.add(key)
        if not capped:
            for key in remaining:
                weights[key] = remaining_budget * max(0.0, scores[key]) / score_sum
            remaining_budget = 0.0
        remaining -= capped
    return weights


def build_multi_period_advisory(
    *,
    candidate_id: str,
    sleeves: Sequence[Mapping[str, Any]],
    covariance: Mapping[str, Mapping[str, float]],
    current_weights: Mapping[str, float],
    horizon_steps: int = 3,
    minimum_qualified_sleeves: int = 4,
    minimum_fills: int = 30,
    cash_floor: float = 0.10,
    max_sleeve_weight: float = 0.25,
    max_step_turnover: float = 0.20,
    max_pair_correlation: float = 0.75,
) -> dict[str, Any]:
    qualified = [
        dict(row)
        for row in sleeves
        if bool(row.get("qualified"))
        and str(row.get("candidate_id") or "") == candidate_id
        and int(row.get("independent_fills") or 0) >= minimum_fills
    ]
    reasons: list[str] = []
    if len(qualified) < minimum_qualified_sleeves:
        reasons.append("insufficient_candidate_bound_qualified_sleeves")
    ids = [str(row.get("sleeve_id") or "") for row in qualified]
    if any(not value for value in ids) or len(set(ids)) != len(ids):
        reasons.append("sleeve_identity_invalid")
    high_pairs = []
    for index, left in enumerate(ids):
        for right in ids[index + 1 :]:
            correlation = _correlation(covariance, left, right)
            if abs(correlation) > max_pair_correlation:
                high_pairs.append(
                    {"left": left, "right": right, "correlation": correlation}
                )
    if high_pairs:
        reasons.append("qualified_sleeves_too_correlated")
    if reasons:
        return {
            "ok": False,
            "status": "abstain",
            "reasons": reasons,
            "high_correlation_pairs": high_pairs,
            "advisory_only": True,
            "execution_authority": False,
        }

    scores: dict[str, float] = {}
    for row in qualified:
        sleeve_id = str(row["sleeve_id"])
        variance = float(dict(covariance.get(sleeve_id) or {}).get(sleeve_id, 0.0))
        net_edge = float(row.get("expected_return_bps") or 0.0) - float(
            row.get("cost_bps") or 0.0
        )
        scores[sleeve_id] = max(0.0, net_edge) / math.sqrt(max(variance, 1e-12))
    if not any(score > 0 for score in scores.values()):
        return {
            "ok": False,
            "status": "abstain",
            "reasons": ["no_positive_post_cost_candidate_edge"],
            "advisory_only": True,
            "execution_authority": False,
        }

    target = _capped_weights(scores, 1.0 - cash_floor, max_sleeve_weight)
    current = {key: max(0.0, float(current_weights.get(key, 0.0))) for key in ids}
    steps: list[dict[str, Any]] = []
    for step in range(1, max(1, horizon_steps) + 1):
        raw_turnover = sum(abs(target[key] - current[key]) for key in ids)
        blend = min(1.0, max_step_turnover / raw_turnover) if raw_turnover > 0 else 1.0
        next_weights = {
            key: current[key] + blend * (target[key] - current[key]) for key in ids
        }
        turnover = sum(abs(next_weights[key] - current[key]) for key in ids)
        steps.append(
            {
                "step": step,
                "weights": next_weights,
                "cash_weight": max(0.0, 1.0 - sum(next_weights.values())),
                "turnover": turnover,
            }
        )
        current = next_weights
    return {
        "ok": True,
        "status": "advisory_ready",
        "candidate_id": candidate_id,
        "qualified_sleeve_count": len(qualified),
        "target_weights": target,
        "steps": steps,
        "constraints": {
            "cash_floor": cash_floor,
            "max_sleeve_weight": max_sleeve_weight,
            "max_step_turnover": max_step_turnover,
            "max_pair_correlation": max_pair_correlation,
        },
        "advisory_only": True,
        "execution_authority": False,
    }
