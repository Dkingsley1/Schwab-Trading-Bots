"""Candidate-bound sleeve ranking and bounded portfolio recommendation.

The selector is deliberately advisory. It can identify the best-supported sleeve
or a low-correlation group for a configured account and route, but it cannot
change an allocator, an allowlist, a capital limit, or an order.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import statistics
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Iterable, Mapping, Sequence

from core.operating_contracts import build_operating_contract


def _dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _list(value: Any) -> list[Any]:
    return list(value) if isinstance(value, (list, tuple)) else []


def _float(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float(default)
    return number if math.isfinite(number) else float(default)


def _int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return int(default)


def _clamp(value: Any, low: float = 0.0, high: float = 1.0) -> float:
    return min(max(_float(value), low), high)


def _parse_timestamp(value: Any) -> datetime | None:
    raw = str(value or "").strip()
    if not raw:
        return None
    try:
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def canonical_hash(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _unique(values: Iterable[Any]) -> list[str]:
    return list(dict.fromkeys(str(value) for value in values if str(value)))


def validate_policy(policy: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    evidence = _dict(policy.get("sleeve_evidence"))
    score = _dict(policy.get("score"))
    weights = _dict(score.get("weights"))
    portfolio = _dict(policy.get("portfolio_selection"))
    growth = _dict(policy.get("capital_growth"))
    account_scope = _dict(policy.get("account_scaling_scope"))
    portability = _dict(policy.get("host_portability"))
    routes = _dict(policy.get("application_routes"))
    safety = _dict(policy.get("safety_contract"))

    if _int(policy.get("schema_version")) != 1:
        errors.append("schema_version_invalid")
    if str(policy.get("operating_mode") or "") != "candidate_bound_advisory_selection":
        errors.append("operating_mode_invalid")
    expected_weights = {
        "conservative_edge",
        "persistence",
        "drawdown_control",
        "execution_capacity",
        "evidence_depth",
        "regime_fit",
        "independent_breadth",
    }
    if set(weights) != expected_weights:
        errors.append("score_dimensions_invalid")
    if abs(sum(_float(value) for value in weights.values()) - 1.0) > 1e-9:
        errors.append("score_weights_must_sum_to_one")
    for key in (
        "minimum_qualified_bots",
        "minimum_candidate_samples",
        "minimum_independent_days",
        "minimum_distinct_regimes",
        "minimum_correlation_clusters",
    ):
        if _int(evidence.get(key)) < 1:
            errors.append(f"{key}_invalid")
    if not 0.0 <= _float(evidence.get("minimum_positive_day_ratio")) <= 1.0:
        errors.append("minimum_positive_day_ratio_invalid")
    if _int(portfolio.get("maximum_candidates_considered")) < 1:
        errors.append("maximum_candidates_considered_invalid")
    if _int(portfolio.get("maximum_combinations_evaluated")) < 1:
        errors.append("maximum_combinations_evaluated_invalid")
    if not 0.0 <= _float(portfolio.get("maximum_pairwise_correlation")) <= 1.0:
        errors.append("maximum_pairwise_correlation_invalid")
    if not routes:
        errors.append("application_routes_missing")
    if str(growth.get("mode") or "") != "broker_reconciled_profit_compounding_advisory":
        errors.append("capital_growth_mode_invalid")
    seed_capital = _float(growth.get("seed_capital_usd"))
    if seed_capital <= 0.0:
        errors.append("capital_growth_seed_invalid")
    reinvestment_fraction = _float(growth.get("earned_profit_reinvestment_fraction"))
    reserve_fraction = _float(growth.get("minimum_profit_reserve_fraction"))
    if not 0.0 <= reinvestment_fraction <= 1.0:
        errors.append("capital_growth_reinvestment_fraction_invalid")
    if not 0.0 <= reserve_fraction <= 1.0:
        errors.append("capital_growth_reserve_fraction_invalid")
    if abs(reinvestment_fraction + reserve_fraction - 1.0) > 1e-9:
        errors.append("capital_growth_profit_fractions_must_sum_to_one")
    if (
        not 0.0
        <= _float(
            growth.get("maximum_incremental_reinvestment_fraction_of_active_capital")
        )
        <= 1.0
    ):
        errors.append("capital_growth_incremental_fraction_invalid")
    if not 0.0 <= _float(growth.get("maximum_growth_drawdown_fraction")) <= 1.0:
        errors.append("capital_growth_drawdown_fraction_invalid")
    for key in (
        "external_deposits_count_toward_organic_progress",
        "unrealized_pnl_counts_toward_organic_progress",
        "unattributed_income_counts_toward_organic_progress",
    ):
        if growth.get(key) is not False:
            errors.append(f"capital_growth_{key}_must_be_false")
    growth_targets = [
        row for row in _list(growth.get("targets")) if isinstance(row, Mapping)
    ]
    growth_target_ids = [str(row.get("target_id") or "") for row in growth_targets]
    growth_target_capitals = [_float(row.get("capital_usd")) for row in growth_targets]
    if (
        not growth_targets
        or any(not target_id for target_id in growth_target_ids)
        or any(value <= 0.0 for value in growth_target_capitals)
        or growth_target_capitals != sorted(set(growth_target_capitals))
        or abs(growth_target_capitals[0] - seed_capital) > 1e-9
    ):
        errors.append("capital_growth_targets_invalid")
    if len(growth_target_ids) != len(set(growth_target_ids)):
        errors.append("capital_growth_target_ids_not_unique")
    deployment_rows = [
        row
        for row in _list(growth.get("deployment_requirements"))
        if isinstance(row, Mapping)
    ]
    deployment_capitals = [
        _float(row.get("maximum_capital_usd")) for row in deployment_rows
    ]
    if (
        not deployment_rows
        or any(value <= 0.0 for value in deployment_capitals)
        or deployment_capitals != sorted(set(deployment_capitals))
        or max(deployment_capitals, default=0.0) + 1e-9
        < max(growth_target_capitals, default=float("inf"))
        or any(
            _int(row.get("minimum_independent_sleeves")) < 1 for row in deployment_rows
        )
        or any(
            _float(row.get("minimum_capacity_headroom_ratio")) < 1.0
            for row in deployment_rows
        )
    ):
        errors.append("capital_growth_deployment_requirements_invalid")
    if (
        str(account_scope.get("mode") or "")
        != "classified_account_policy_isolated_ledgers"
    ):
        errors.append("account_scaling_scope_mode_invalid")
    for key in (
        "applies_to_all_classified_accounts",
        "active_canary_account_selected_separately",
        "organic_progress_isolated_by_account_policy_key",
        "operator_review_required_per_account",
    ):
        if account_scope.get(key) is not True:
            errors.append(f"account_scaling_scope_{key}_must_be_true")
    for key in (
        "cross_account_evidence_pooling",
        "cross_account_loss_netting",
        "cross_account_capital_netting",
        "raw_account_identifiers_allowed",
        "automatic_account_activation",
    ):
        if account_scope.get(key) is not False:
            errors.append(f"account_scaling_scope_{key}_must_be_false")
    for key in ("policy_and_code_portable", "account_policy_keys_portable"):
        if portability.get(key) is not True:
            errors.append(f"host_portability_{key}_must_be_true")
    for key in (
        "raw_account_identifiers_portable",
        "credential_material_portable",
        "oauth_tokens_portable",
        "keychain_bindings_portable",
        "automatic_live_reactivation_on_new_host",
    ):
        if portability.get(key) is not False:
            errors.append(f"host_portability_{key}_must_be_false")
    if not _unique(_list(portability.get("required_new_host_checks"))):
        errors.append("host_portability_required_new_host_checks_missing")
    goals = [
        row
        for row in _list(policy.get("system_scalability_goals"))
        if isinstance(row, Mapping)
    ]
    goal_ids = [str(row.get("goal_id") or "") for row in goals]
    if not goals or any(not goal_id for goal_id in goal_ids):
        errors.append("scalability_goals_invalid")
    if len(goal_ids) != len(set(goal_ids)):
        errors.append("scalability_goal_ids_not_unique")
    required_false = (
        "changes_runtime_decisions",
        "changes_paper_allocation",
        "changes_live_allocation",
        "automatic_stage_progression",
        "automatic_capital_scaling",
        "paper_execution_authority",
        "live_execution_authority",
        "allowlist_authority",
        "order_payload_created",
        "profitability_guaranteed",
    )
    if safety.get("advisory_only") is not True:
        errors.append("safety_advisory_only_must_be_true")
    if safety.get("operator_review_required") is not True:
        errors.append("safety_operator_review_required_must_be_true")
    for key in required_false:
        if safety.get(key) is not False:
            errors.append(f"safety_{key}_must_be_false")
    return _unique(errors)


def _correlation(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
    minimum_common_days: int,
) -> tuple[float | None, int]:
    common = sorted(set(left).intersection(right))
    if len(common) < minimum_common_days:
        return None, len(common)
    left_values = [_float(left[day]) for day in common]
    right_values = [_float(right[day]) for day in common]
    left_mean = statistics.fmean(left_values)
    right_mean = statistics.fmean(right_values)
    numerator = sum(
        (left_value - left_mean) * (right_value - right_mean)
        for left_value, right_value in zip(left_values, right_values)
    )
    left_scale = math.sqrt(sum((value - left_mean) ** 2 for value in left_values))
    right_scale = math.sqrt(sum((value - right_mean) ** 2 for value in right_values))
    if left_scale <= 1e-12 or right_scale <= 1e-12:
        return None, len(common)
    return numerator / (left_scale * right_scale), len(common)


def _drawdown(daily: Mapping[str, Any]) -> float:
    cumulative = 0.0
    peak = 0.0
    maximum = 0.0
    for day in sorted(daily):
        cumulative += _float(daily[day])
        peak = max(peak, cumulative)
        maximum = max(maximum, peak - cumulative)
    return maximum


def _source_freshness(
    name: str,
    payload: Mapping[str, Any],
    policy: Mapping[str, Any],
    now: datetime,
) -> dict[str, Any]:
    timestamp = _parse_timestamp(payload.get("timestamp_utc"))
    maximum = _float(_dict(policy.get("freshness_seconds")).get(name), 0.0)
    age = max((now - timestamp).total_seconds(), 0.0) if timestamp else None
    return {
        "timestamp_utc": timestamp.isoformat() if timestamp else "",
        "age_seconds": round(age, 3) if age is not None else None,
        "maximum_age_seconds": maximum,
        "fresh": bool(
            timestamp and maximum > 0.0 and age is not None and age <= maximum
        ),
    }


def _active_tier(graduation: Mapping[str, Any]) -> dict[str, Any]:
    ladder = _dict(graduation.get("capital_ladder"))
    tier_name = str(ladder.get("active_policy_tier") or "")
    rows = [row for row in _list(ladder.get("evaluations")) if isinstance(row, Mapping)]
    row = next(
        (dict(item) for item in rows if str(item.get("tier") or "") == tier_name), {}
    )
    return {
        "tier": tier_name,
        "limits": _dict(row.get("proposed_limits")),
        "operator_review_eligible": bool(row.get("operator_review_eligible", False)),
        "limits_applied": bool(row.get("limits_applied", False)),
        "automatic_scaling": False,
    }


def _capital_tier_for_target(
    graduation: Mapping[str, Any], target_capital_usd: float
) -> dict[str, Any]:
    ladder = _dict(graduation.get("capital_ladder"))
    rows = [row for row in _list(ladder.get("evaluations")) if isinstance(row, Mapping)]
    return next(
        (
            dict(row)
            for row in rows
            if abs(
                _float(_dict(row.get("proposed_limits")).get("account_capital_usd"))
                - target_capital_usd
            )
            <= 1e-9
        ),
        {},
    )


def _deployment_requirement_for_capital(
    policy: Mapping[str, Any], target_capital_usd: float
) -> dict[str, Any]:
    rows = sorted(
        [
            dict(row)
            for row in _list(
                _dict(policy.get("capital_growth")).get("deployment_requirements")
            )
            if isinstance(row, Mapping)
        ],
        key=lambda row: _float(row.get("maximum_capital_usd")),
    )
    selected = next(
        (
            row
            for row in rows
            if _float(row.get("maximum_capital_usd")) + 1e-9
            >= max(float(target_capital_usd), 0.0)
        ),
        {},
    )
    return {
        "maximum_capital_usd": _float(selected.get("maximum_capital_usd")),
        "minimum_independent_sleeves": _int(
            selected.get("minimum_independent_sleeves"), 1
        ),
        "minimum_capacity_headroom_ratio": _float(
            selected.get("minimum_capacity_headroom_ratio"), 1.0
        ),
        "cataloged": bool(selected),
    }


def _growth_target_matches_tier(
    target: Mapping[str, Any], tier: Mapping[str, Any]
) -> bool:
    requirements = _dict(tier.get("policy_requirements"))
    return bool(
        tier
        and _int(target.get("minimum_reconciled_round_trips"))
        == _int(requirements.get("minimum_reconciled_round_trips"))
        and _int(target.get("minimum_independent_days"))
        == _int(requirements.get("minimum_independent_trading_days"))
        and _int(target.get("minimum_distinct_regimes"))
        == _int(requirements.get("minimum_distinct_regime_buckets"))
        and bool(target.get("require_positive_lcb_95", False))
        == bool(requirements.get("require_positive_normal_approx_lcb_95", False))
        and bool(target.get("require_positive_benchmark_excess", False))
        == bool(requirements.get("require_positive_benchmark_excess", False))
        and bool(target.get("require_actual_fee_evidence", False))
        == bool(requirements.get("require_actual_fee_evidence", False))
    )


def _growth_target_context(
    graduation: Mapping[str, Any],
    policy: Mapping[str, Any],
    *,
    active_capital_usd: float,
) -> dict[str, Any]:
    growth = _dict(policy.get("capital_growth"))
    targets = sorted(
        [dict(row) for row in _list(growth.get("targets")) if isinstance(row, Mapping)],
        key=lambda row: _float(row.get("capital_usd")),
    )
    next_target = next(
        (
            row
            for row in targets
            if _float(row.get("capital_usd")) > active_capital_usd + 1e-9
        ),
        {},
    )
    target_capital = _float(next_target.get("capital_usd"), active_capital_usd)
    target_tier = _capital_tier_for_target(graduation, target_capital)
    deployment = _deployment_requirement_for_capital(policy, target_capital)
    return {
        "policy": growth,
        "targets": targets,
        "next_target": next_target,
        "next_target_capital_usd": target_capital,
        "next_target_tier": target_tier,
        "next_target_tier_id": str(target_tier.get("tier") or ""),
        "next_target_deployment_requirements": deployment,
    }


def _capital_growth_plan(
    graduation: Mapping[str, Any],
    policy: Mapping[str, Any],
    *,
    active_capital_usd: float,
    target_portfolio_plan: Mapping[str, Any] | None,
    candidate_bound: bool,
    sources_fresh: bool,
    global_economic_ready: bool,
    independent_execution_ready: bool,
) -> dict[str, Any]:
    context = _growth_target_context(
        graduation, policy, active_capital_usd=active_capital_usd
    )
    growth = _dict(context.get("policy"))
    targets = [
        dict(row) for row in _list(context.get("targets")) if isinstance(row, Mapping)
    ]
    next_target = _dict(context.get("next_target"))
    next_target_capital = _float(
        context.get("next_target_capital_usd"), active_capital_usd
    )
    next_target_tier = _dict(context.get("next_target_tier"))
    deployment_requirements = _dict(context.get("next_target_deployment_requirements"))
    metrics = _dict(graduation.get("metrics"))
    seed_capital = _float(growth.get("seed_capital_usd"), 200.0)
    total_post_cost_pnl = _float(metrics.get("total_post_cost_pnl_usd"))
    organic_capital = max(seed_capital + total_post_cost_pnl, 0.0)
    daily = _dict(metrics.get("daily_post_cost_pnl_usd"))
    running_equity = seed_capital
    peak_equity = seed_capital
    computed_drawdown = 0.0
    for day in sorted(daily):
        running_equity += _float(daily[day])
        peak_equity = max(peak_equity, running_equity)
        computed_drawdown = max(computed_drawdown, peak_equity - running_equity)
    drawdown_usd = max(
        computed_drawdown,
        _float(metrics.get("maximum_cumulative_drawdown_usd")),
    )
    drawdown_fraction = drawdown_usd / max(peak_equity, seed_capital, 1e-9)
    maximum_drawdown_fraction = _float(
        growth.get("maximum_growth_drawdown_fraction"), 0.1
    )
    round_trips = _int(metrics.get("reconciled_round_trip_count"))
    independent_days = _int(metrics.get("independent_trading_day_count"))
    regimes = _int(metrics.get("distinct_regime_bucket_count"))
    actual_fee_count = _int(metrics.get("actual_fee_evidence_round_trip_count"))
    benchmark_count = _int(metrics.get("benchmark_evidence_round_trip_count"))
    benchmark_excess = _float(metrics.get("total_benchmark_excess_return_bps"))
    lcb_raw = metrics.get("normal_approx_lcb_95_post_cost_return_bps")
    lcb = _float(lcb_raw) if lcb_raw is not None else None
    graduation_clear = bool(
        graduation.get("control_ok", graduation.get("ok", False))
        and not _list(graduation.get("blockers"))
    )

    target_evaluations: list[dict[str, Any]] = []
    for target in targets:
        capital = _float(target.get("capital_usd"))
        minimum_round_trips = _int(target.get("minimum_reconciled_round_trips"))
        minimum_days = _int(target.get("minimum_independent_days"))
        minimum_regimes = _int(target.get("minimum_distinct_regimes"))
        require_lcb = bool(target.get("require_positive_lcb_95", False))
        require_benchmark = bool(target.get("require_positive_benchmark_excess", False))
        require_fees = bool(target.get("require_actual_fee_evidence", False))
        checks = {
            "organic_capital_reached": organic_capital + 1e-9 >= capital,
            "reconciled_round_trips": round_trips >= minimum_round_trips,
            "independent_days": independent_days >= minimum_days,
            "distinct_regimes": regimes >= minimum_regimes,
            "positive_lcb_95": (not require_lcb) or (lcb is not None and lcb > 0.0),
            "positive_benchmark_excess": (not require_benchmark)
            or (benchmark_count >= minimum_round_trips and benchmark_excess > 0.0),
            "actual_fee_evidence": (not require_fees)
            or actual_fee_count >= minimum_round_trips,
            "growth_drawdown_within_limit": drawdown_fraction
            <= maximum_drawdown_fraction + 1e-12,
            "graduation_control_clear": graduation_clear,
        }
        tier = _capital_tier_for_target(graduation, capital)
        ladder_aligned = _growth_target_matches_tier(target, tier)
        checks["graduation_ladder_alignment"] = ladder_aligned
        evidence_checks = {
            key: value
            for key, value in checks.items()
            if key != "organic_capital_reached"
        }
        target_evaluations.append(
            {
                "target_id": str(target.get("target_id") or ""),
                "capital_usd": round(capital, 2),
                "checks": checks,
                "organic_target_earned": all(checks.values()),
                "economic_evidence_ready": all(evidence_checks.values()),
                "execution_tier_cataloged": bool(tier),
                "execution_tier": str(tier.get("tier") or ""),
                "operating_class": str(
                    _dict(tier.get("scale_governance")).get("operating_class") or ""
                ),
                "graduation_ladder_aligned": ladder_aligned,
                "execution_tier_operator_review_eligible": bool(
                    tier.get("operator_review_eligible", False)
                ),
                "automatic_scaling": False,
            }
        )

    target_requirements = next(
        (
            row
            for row in target_evaluations
            if abs(_float(row.get("capital_usd")) - next_target_capital) <= 1e-9
        ),
        {},
    )
    target_checks = _dict(target_requirements.get("checks"))
    target_plan = _dict(target_portfolio_plan)
    minimum_target_sleeves = _int(
        deployment_requirements.get("minimum_independent_sleeves"), 1
    )
    minimum_target_headroom = _float(
        deployment_requirements.get("minimum_capacity_headroom_ratio"), 1.0
    )
    observed_target_sleeves = _int(target_plan.get("sleeve_count"))
    observed_target_headroom = _float(
        target_plan.get("observed_capacity_headroom_ratio")
    )
    target_scale_governance = _dict(next_target_tier.get("scale_governance"))
    target_checks.update(
        {
            "candidate_bound": bool(candidate_bound),
            "sources_fresh": bool(sources_fresh),
            "global_economic_evidence": bool(global_economic_ready),
            "independent_execution_calibration": bool(independent_execution_ready),
            "target_portfolio_supported": bool(target_portfolio_plan),
            "target_portfolio_sleeve_breadth": bool(
                target_portfolio_plan
                and observed_target_sleeves >= minimum_target_sleeves
            ),
            "target_capacity_headroom": bool(
                target_portfolio_plan
                and observed_target_headroom + 1e-9 >= minimum_target_headroom
            ),
            "execution_tier_cataloged": bool(next_target_tier),
            "operating_class_enabled": bool(
                target_scale_governance.get("operating_class_enabled", False)
            ),
            "required_scale_controls_evidenced": bool(
                target_scale_governance.get("required_controls_evidenced", False)
            ),
            "execution_tier_review_eligible": bool(
                next_target_tier.get("operator_review_eligible", False)
            ),
        }
    )
    target_blockers = [key for key, value in target_checks.items() if not value]
    reinvestment_checks = {
        key: value
        for key, value in target_checks.items()
        if key
        not in {
            "organic_capital_reached",
            "execution_tier_cataloged",
            "execution_tier_review_eligible",
        }
    }
    reinvestment_eligible = bool(
        total_post_cost_pnl > 0.0 and next_target and all(reinvestment_checks.values())
    )
    reinvestment_fraction = _float(
        growth.get("earned_profit_reinvestment_fraction"), 0.75
    )
    incremental_cap = active_capital_usd * _float(
        growth.get("maximum_incremental_reinvestment_fraction_of_active_capital"),
        0.25,
    )
    positive_profit = max(total_post_cost_pnl, 0.0)
    reinvestment_budget = (
        min(positive_profit * reinvestment_fraction, incremental_cap)
        if reinvestment_eligible
        else 0.0
    )
    reserve_usd = max(positive_profit - reinvestment_budget, 0.0)
    target_reached = bool(next_target and organic_capital + 1e-9 >= next_target_capital)
    operator_review_ready = bool(
        next_target and target_checks and all(target_checks.values())
    )
    if round_trips <= 0:
        action = "await_first_reconciled_canary_round_trip"
    elif total_post_cost_pnl < 0.0 or drawdown_fraction > maximum_drawdown_fraction:
        action = "defend_seed_and_reduce_risk"
    elif operator_review_ready:
        action = "operator_review_next_growth_tier"
    elif target_reached and not next_target_tier:
        action = "research_and_define_next_execution_tier"
    elif target_reached and not bool(
        target_scale_governance.get("operating_class_enabled", False)
    ):
        action = "prepare_and_independently_validate_next_operating_class"
    elif reinvestment_eligible:
        action = "retain_and_compound_earned_profit_after_review"
    elif total_post_cost_pnl > 0.0:
        action = "retain_profit_and_collect_missing_evidence"
    else:
        action = "collect_and_validate_edge"

    denominator = max(next_target_capital - active_capital_usd, 0.0)
    progress = (
        _clamp((organic_capital - active_capital_usd) / denominator)
        if next_target and denominator > 0.0
        else 1.0
    )
    highest_equity_target = max(
        (
            _float(row.get("capital_usd"))
            for row in target_evaluations
            if _dict(row.get("checks")).get("organic_capital_reached", False)
        ),
        default=0.0,
    )
    highest_earned_target = max(
        (
            _float(row.get("capital_usd"))
            for row in target_evaluations
            if row.get("organic_target_earned", False)
        ),
        default=0.0,
    )
    return {
        "mode": str(growth.get("mode") or ""),
        "profit_source": str(growth.get("realized_profit_source") or ""),
        "seed_capital_usd": round(seed_capital, 2),
        "active_policy_capital_usd": round(active_capital_usd, 2),
        "broker_reconciled_post_cost_profit_usd": round(total_post_cost_pnl, 8),
        "organic_capital_usd": round(organic_capital, 8),
        "highest_equity_target_reached_usd": round(highest_equity_target, 2),
        "highest_economically_earned_target_usd": round(highest_earned_target, 2),
        "next_target_id": str(next_target.get("target_id") or ""),
        "next_target_capital_usd": round(next_target_capital, 2),
        "long_range_target_capital_usd": round(
            max((_float(row.get("capital_usd")) for row in targets), default=0.0),
            2,
        ),
        "next_target_operating_class": str(
            target_scale_governance.get("operating_class") or ""
        ),
        "next_target_deployment_requirements": {
            "minimum_independent_sleeves": minimum_target_sleeves,
            "minimum_capacity_headroom_ratio": round(minimum_target_headroom, 8),
            "operating_class_enabled": bool(
                target_scale_governance.get("operating_class_enabled", False)
            ),
            "required_scale_controls": list(
                target_scale_governance.get("required_controls") or []
            ),
            "missing_scale_controls": list(
                target_scale_governance.get("missing_controls") or []
            ),
        },
        "gap_to_next_target_usd": round(
            max(next_target_capital - organic_capital, 0.0), 8
        ),
        "progress_to_next_target_percent": round(progress * 100.0, 2),
        "growth_drawdown_usd": round(drawdown_usd, 8),
        "growth_drawdown_fraction": round(drawdown_fraction, 8),
        "reinvestment_eligible": reinvestment_eligible,
        "advisory_reinvestment_budget_usd": round(reinvestment_budget, 8),
        "advisory_profit_reserve_usd": round(reserve_usd, 8),
        "advisory_capital_after_review_usd": round(
            min(active_capital_usd + reinvestment_budget, next_target_capital), 8
        ),
        "target_sleeve_plan": dict(target_portfolio_plan or {}),
        "target_checks": target_checks,
        "target_blockers": target_blockers,
        "operator_review_ready": operator_review_ready,
        "recommended_action": action,
        "target_evaluations": target_evaluations,
        "accounting_contract": {
            "external_deposits_count_toward_organic_progress": False,
            "unrealized_pnl_counts_toward_organic_progress": False,
            "unattributed_income_counts_toward_organic_progress": False,
            "broker_reconciled_post_cost_round_trips_only": True,
        },
        "authority_contract": {
            "advisory_only": True,
            "automatic_reinvestment": False,
            "automatic_capital_scaling": False,
            "paper_execution_authority": False,
            "live_execution_authority": False,
            "operator_review_required": True,
            "profitability_guaranteed": False,
        },
    }


def _account_context(
    account_study: Mapping[str, Any],
    account_policy_key: str,
) -> dict[str, Any]:
    accounts = [
        row for row in _list(account_study.get("accounts")) if isinstance(row, Mapping)
    ]
    row = next(
        (
            dict(item)
            for item in accounts
            if str(item.get("account_policy_key") or "") == account_policy_key
        ),
        {},
    )
    truth = _dict(row.get("account_capability_truth"))
    classification = _dict(truth.get("operator_classification"))
    preflight = _dict(truth.get("canary_preflight"))
    return {
        "found": bool(row),
        "account_policy_key": account_policy_key,
        "account_kind": str(classification.get("account_kind") or "unknown"),
        "tax_wrapper": str(classification.get("tax_wrapper") or "unknown"),
        "trading_access": str(classification.get("trading_access") or "unknown"),
        "borrowing_allowed": bool(classification.get("borrowing_allowed", False)),
        "short_stock_allowed": bool(classification.get("short_stock_allowed", False)),
        "classification_complete": bool(
            classification.get("classification_complete", False)
        ),
        "configured_canary_cap_usd": _float(classification.get("canary_cap_usd")),
        "allowed_live_routes": sorted(
            str(value)
            for value in _list(classification.get("allowed_live_routes"))
            if str(value)
        ),
        "account_preflight_ready": bool(
            preflight.get("account_preflight_ready", False)
        ),
        "preflight_blockers": _unique(_list(preflight.get("blockers"))),
        "live_execution_authority": False,
    }


def _account_scaling_scope(
    account_study: Mapping[str, Any],
    policy: Mapping[str, Any],
    *,
    active_account_policy_key: str,
) -> dict[str, Any]:
    scope_policy = _dict(policy.get("account_scaling_scope"))
    keys = sorted(
        {
            str(row.get("account_policy_key") or "").strip()
            for row in _list(account_study.get("accounts"))
            if isinstance(row, Mapping)
            and str(row.get("account_policy_key") or "").strip()
        }
    )
    rows: list[dict[str, Any]] = []
    for key in keys:
        context = _account_context(account_study, key)
        rows.append(
            {
                "account_policy_key": key,
                "account_kind": str(context.get("account_kind") or "unknown"),
                "tax_wrapper": str(context.get("tax_wrapper") or "unknown"),
                "trading_access": str(context.get("trading_access") or "unknown"),
                "classification_complete": bool(
                    context.get("classification_complete", False)
                ),
                "configured_canary_cap_usd": _float(
                    context.get("configured_canary_cap_usd")
                ),
                "allowed_live_routes": list(context.get("allowed_live_routes") or []),
                "active_canary_account": key == active_account_policy_key,
                "automatic_account_activation": False,
                "live_execution_authority": False,
            }
        )
    classified_count = sum(row["classification_complete"] for row in rows)
    return {
        "mode": str(scope_policy.get("mode") or ""),
        "applies_to_all_classified_accounts": bool(
            scope_policy.get("applies_to_all_classified_accounts", False)
        ),
        "discovered_account_policy_count": len(rows),
        "classified_account_policy_count": classified_count,
        "active_account_policy_key": active_account_policy_key,
        "accounts": rows,
        "organic_progress_isolated_by_account_policy_key": True,
        "cross_account_evidence_pooling": False,
        "cross_account_loss_netting": False,
        "cross_account_capital_netting": False,
        "raw_account_identifiers_present": False,
        "automatic_account_activation": False,
        "operator_review_required_per_account": True,
    }


def _route_compatible(
    route: Mapping[str, Any],
    account: Mapping[str, Any],
) -> tuple[bool, list[str]]:
    blockers: list[str] = []
    if not route:
        blockers.append("execution_route_not_cataloged")
        return False, blockers
    if str(account.get("account_kind") or "") not in {
        str(value) for value in _list(route.get("allowed_account_kinds"))
    }:
        blockers.append("account_kind_not_route_compatible")
    if str(account.get("trading_access") or "") not in {
        str(value) for value in _list(route.get("allowed_trading_access"))
    }:
        blockers.append("trading_access_not_route_compatible")
    if (
        bool(account.get("borrowing_allowed", False))
        and route.get("borrowing_allowed") is False
    ):
        blockers.append("borrowing_account_not_route_compatible")
    if (
        bool(account.get("short_stock_allowed", False))
        and route.get("short_stock_allowed") is False
    ):
        blockers.append("short_stock_account_not_route_compatible")
    return not blockers, blockers


def _qualified_bot(
    profile: Mapping[str, Any],
    evidence_policy: Mapping[str, Any],
) -> tuple[bool, list[str]]:
    blockers: list[str] = []
    metrics = _dict(profile.get("candidate_evidence"))
    marginal = _dict(profile.get("marginal_contribution"))
    capacity = _dict(profile.get("capacity_curve"))
    regime = _dict(profile.get("current_regime_compatibility"))
    lcb = metrics.get("post_cost_return_lcb_bps")
    if not bool(profile.get("shadow_vote_eligible", False)):
        blockers.append("shadow_vote_ineligible")
    if evidence_policy.get("require_rank_evidence", True) and not bool(
        profile.get("rank_evidence_ready", False)
    ):
        blockers.append("rank_evidence_pending")
    if evidence_policy.get("require_persistence", True) and not bool(
        profile.get("persistence_ready", False)
    ):
        blockers.append("persistence_evidence_pending")
    if evidence_policy.get("require_marginal_contribution", True):
        if not bool(marginal.get("evidence_ready", False)):
            blockers.append("marginal_contribution_pending")
        if bool(marginal.get("duplicate_cluster", False)):
            blockers.append("duplicate_correlation_cluster")
    if evidence_policy.get("require_capacity_curve", True) and not bool(
        capacity.get("evidence_ready", False)
    ):
        blockers.append("capacity_curve_pending")
    if evidence_policy.get("require_current_regime_compatibility", True):
        if not bool(regime.get("compatible", False)):
            blockers.append("current_regime_incompatible")
        if _float(regime.get("score")) < _float(
            evidence_policy.get("minimum_regime_compatibility_score"), 0.55
        ):
            blockers.append("current_regime_score_below_floor")
    if lcb is None or _float(lcb, -1e12) <= _float(
        evidence_policy.get("minimum_post_cost_lcb_bps_exclusive"), 0.0
    ):
        blockers.append("positive_post_cost_lcb_pending")
    return not blockers, _unique(blockers)


def _aggregate_sleeves(
    profiles: Sequence[Mapping[str, Any]],
    assignments: Mapping[str, Mapping[str, Any]],
    policy: Mapping[str, Any],
    *,
    target_notional_usd: float,
    candidate_bound: bool,
) -> list[dict[str, Any]]:
    evidence_policy = _dict(policy.get("sleeve_evidence"))
    score_policy = _dict(policy.get("score"))
    weights = _dict(score_policy.get("weights"))
    allowed_roles = {
        str(value) for value in _list(evidence_policy.get("eligible_role_ids"))
    }
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for profile in profiles:
        bot_id = str(profile.get("bot_id") or "")
        assignment = _dict(assignments.get(bot_id))
        if str(assignment.get("role_id") or "") not in allowed_roles:
            continue
        if not bool(assignment.get("active", True)):
            continue
        sleeve_id = str(profile.get("sleeve_id") or assignment.get("sleeve_id") or "")
        if sleeve_id:
            grouped[sleeve_id].append(profile)

    sleeves: list[dict[str, Any]] = []
    for sleeve_id, rows in sorted(grouped.items()):
        qualified: list[Mapping[str, Any]] = []
        bot_blockers: dict[str, list[str]] = {}
        daily: dict[str, float] = defaultdict(float)
        days: set[str] = set()
        regimes: set[str] = set()
        candidate_samples = 0
        effective_samples = 0
        turnover_notional = 0.0
        for row in rows:
            metrics = _dict(row.get("candidate_evidence"))
            candidate_samples = max(
                candidate_samples, _int(metrics.get("sample_count"))
            )
            effective_samples = max(
                effective_samples, _int(metrics.get("effective_sample_count"))
            )
            turnover_notional += max(_float(metrics.get("turnover_notional")), 0.0)
            for day, pnl in _dict(metrics.get("daily_post_cost_pnl")).items():
                daily[str(day)] += _float(pnl)
                days.add(str(day))
            regimes.update(
                str(value) for value in _list(metrics.get("regimes")) if str(value)
            )
            ready, blockers = _qualified_bot(row, evidence_policy)
            if ready:
                qualified.append(row)
            elif blockers:
                bot_blockers[str(row.get("bot_id") or "")] = blockers

        positive_days = sum(1 for value in daily.values() if value > 0.0)
        positive_ratio = positive_days / max(len(daily), 1)
        maximum_drawdown = _drawdown(daily)
        drawdown_ratio = maximum_drawdown / max(turnover_notional, 1e-9)
        lcbs = [
            _float(
                _dict(row.get("candidate_evidence")).get("post_cost_return_lcb_bps"),
                -1e12,
            )
            for row in qualified
        ]
        conservative_lcb = min(lcbs) if lcbs else None
        cluster_capacity: dict[str, float] = defaultdict(float)
        regime_scores: list[float] = []
        rank_scores: list[float] = []
        for row in qualified:
            cluster = str(row.get("correlation_cluster_id") or row.get("bot_id") or "")
            cluster_capacity[cluster] = max(
                cluster_capacity[cluster],
                _float(
                    _dict(row.get("capacity_curve")).get("maximum_supported_notional")
                ),
            )
            regime_scores.append(
                _float(_dict(row.get("current_regime_compatibility")).get("score"))
            )
            rank_scores.append(_float(_dict(row.get("forward_rank")).get("score")))
        supported_notional = sum(cluster_capacity.values())
        largest_cluster = max(cluster_capacity.values(), default=0.0)
        largest_cluster_share = largest_cluster / max(supported_notional, 1e-9)

        minimum_samples = _int(evidence_policy.get("minimum_candidate_samples"), 30)
        minimum_days = _int(evidence_policy.get("minimum_independent_days"), 3)
        minimum_regimes = _int(evidence_policy.get("minimum_distinct_regimes"), 2)
        minimum_bots = _int(evidence_policy.get("minimum_qualified_bots"), 1)
        minimum_clusters = _int(evidence_policy.get("minimum_correlation_clusters"), 1)
        required_notional = max(
            target_notional_usd,
            _float(evidence_policy.get("minimum_supported_notional_usd"), 0.0),
        )
        eligibility_checks = {
            "candidate_bound": bool(candidate_bound),
            "qualified_bot_count": len(qualified) >= minimum_bots,
            "candidate_samples": candidate_samples >= minimum_samples,
            "independent_days": len(days) >= minimum_days,
            "distinct_regimes": len(regimes) >= minimum_regimes,
            "positive_day_ratio": positive_ratio
            >= _float(evidence_policy.get("minimum_positive_day_ratio"), 0.55),
            "positive_post_cost_lcb": conservative_lcb is not None
            and conservative_lcb
            > _float(evidence_policy.get("minimum_post_cost_lcb_bps_exclusive"), 0.0),
            "drawdown_control": drawdown_ratio
            <= _float(evidence_policy.get("maximum_drawdown_to_turnover_ratio"), 0.1),
            "independent_cluster_breadth": len(cluster_capacity) >= minimum_clusters,
            "supported_notional": supported_notional + 1e-9 >= required_notional,
        }
        blockers = [name for name, ready in eligibility_checks.items() if not ready]
        evidence_depth = statistics.fmean(
            (
                min(candidate_samples / max(minimum_samples, 1), 1.0),
                min(len(days) / max(minimum_days, 1), 1.0),
                min(len(regimes) / max(minimum_regimes, 1), 1.0),
                min(len(qualified) / max(minimum_bots, 1), 1.0),
            )
        )
        components = {
            "conservative_edge": _clamp(
                _float(conservative_lcb)
                / max(_float(score_policy.get("target_good_lcb_bps"), 10.0), 1e-9)
                if conservative_lcb is not None
                else 0.0
            ),
            "persistence": _clamp(positive_ratio),
            "drawdown_control": _clamp(
                1.0
                - drawdown_ratio
                / max(
                    _float(
                        evidence_policy.get("maximum_drawdown_to_turnover_ratio"), 0.1
                    ),
                    1e-9,
                )
            ),
            "execution_capacity": _clamp(
                supported_notional / max(required_notional, 1e-9)
            ),
            "evidence_depth": _clamp(evidence_depth),
            "regime_fit": _clamp(
                statistics.fmean(regime_scores) if regime_scores else 0.0
            ),
            "independent_breadth": _clamp(
                len(cluster_capacity) / max(minimum_clusters + 1, 2)
            ),
        }
        score = sum(
            _float(weights.get(name)) * value for name, value in components.items()
        )
        sleeves.append(
            {
                "sleeve_id": sleeve_id,
                "bot_count": len(rows),
                "qualified_bot_count": len(qualified),
                "qualified_bot_ids": sorted(
                    str(row.get("bot_id") or "") for row in qualified
                ),
                "candidate_sample_count": candidate_samples,
                "effective_sample_count": effective_samples,
                "independent_day_count": len(days),
                "distinct_regime_count": len(regimes),
                "regimes": sorted(regimes),
                "positive_day_count": positive_days,
                "positive_day_ratio": round(positive_ratio, 8),
                "daily_post_cost_pnl": {
                    key: round(value, 8) for key, value in sorted(daily.items())
                },
                "total_post_cost_pnl": round(sum(daily.values()), 8),
                "conservative_post_cost_lcb_bps": (
                    round(conservative_lcb, 8) if conservative_lcb is not None else None
                ),
                "maximum_drawdown": round(maximum_drawdown, 8),
                "turnover_notional": round(turnover_notional, 8),
                "drawdown_to_turnover_ratio": round(drawdown_ratio, 8),
                "independent_correlation_cluster_count": len(cluster_capacity),
                "maximum_supported_notional_usd": round(supported_notional, 8),
                "largest_capacity_cluster_share": round(largest_cluster_share, 8),
                "mean_regime_compatibility_score": round(
                    statistics.fmean(regime_scores) if regime_scores else 0.0, 8
                ),
                "mean_bot_forward_rank": round(
                    statistics.fmean(rank_scores) if rank_scores else 0.0, 8
                ),
                "score": round(_clamp(score), 8),
                "score_components": {
                    name: round(value, 8) for name, value in components.items()
                },
                "eligibility_checks": eligibility_checks,
                "research_eligible": not blockers,
                "evidence_blockers": blockers,
                "bot_evidence_debt": [
                    {"bot_id": bot_id, "blockers": values}
                    for bot_id, values in sorted(bot_blockers.items())[:20]
                ],
            }
        )
    return sorted(
        sleeves,
        key=lambda row: (-_float(row.get("score")), str(row.get("sleeve_id") or "")),
    )


def _pairwise_correlations(
    sleeves: Sequence[Mapping[str, Any]],
    minimum_common_days: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for left, right in itertools.combinations(sleeves, 2):
        correlation, common_days = _correlation(
            _dict(left.get("daily_post_cost_pnl")),
            _dict(right.get("daily_post_cost_pnl")),
            minimum_common_days,
        )
        rows.append(
            {
                "left_sleeve_id": str(left.get("sleeve_id") or ""),
                "right_sleeve_id": str(right.get("sleeve_id") or ""),
                "correlation": (
                    round(correlation, 8) if correlation is not None else None
                ),
                "common_day_count": common_days,
                "evidence_ready": correlation is not None,
            }
        )
    return rows


def _correlation_index(
    rows: Sequence[Mapping[str, Any]],
) -> dict[tuple[str, str], Mapping[str, Any]]:
    index: dict[tuple[str, str], Mapping[str, Any]] = {}
    for row in rows:
        left = str(row.get("left_sleeve_id") or "")
        right = str(row.get("right_sleeve_id") or "")
        index[tuple(sorted((left, right)))] = row
    return index


def _bounded_weights(
    sleeves: Sequence[Mapping[str, Any]],
    minimum_weight: float,
    maximum_weight: float,
) -> dict[str, float]:
    count = len(sleeves)
    if count <= 1:
        return {str(sleeves[0].get("sleeve_id") or ""): 1.0} if sleeves else {}
    minimum = min(max(minimum_weight, 0.0), 1.0 / count)
    maximum = max(min(maximum_weight, 1.0), 1.0 / count)
    score_total = sum(max(_float(row.get("score")), 0.01) for row in sleeves)
    raw = {
        str(row.get("sleeve_id") or ""): 0.5 / count
        + 0.5 * max(_float(row.get("score")), 0.01) / score_total
        for row in sleeves
    }
    weights = {key: min(max(value, minimum), maximum) for key, value in raw.items()}
    for _ in range(8):
        total = sum(weights.values())
        if abs(total - 1.0) <= 1e-12:
            break
        if total < 1.0:
            room = {
                key: maximum - value
                for key, value in weights.items()
                if value < maximum - 1e-12
            }
            room_total = sum(room.values())
            if room_total <= 1e-12:
                break
            for key, value in room.items():
                weights[key] += (1.0 - total) * value / room_total
        else:
            room = {
                key: value - minimum
                for key, value in weights.items()
                if value > minimum + 1e-12
            }
            room_total = sum(room.values())
            if room_total <= 1e-12:
                break
            for key, value in room.items():
                weights[key] -= (total - 1.0) * value / room_total
    total = sum(weights.values())
    return {
        key: round(value / max(total, 1e-12), 8)
        for key, value in sorted(weights.items())
    }


def _portfolio_candidates(
    sleeves: Sequence[Mapping[str, Any]],
    correlation_rows: Sequence[Mapping[str, Any]],
    policy: Mapping[str, Any],
    *,
    capital_usd: float,
    maximum_sleeves: int,
    minimum_capacity_headroom_ratio: float = 1.0,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    portfolio_policy = _dict(policy.get("portfolio_selection"))
    maximum_candidates = _int(portfolio_policy.get("maximum_candidates_considered"), 8)
    maximum_combinations = _int(
        portfolio_policy.get("maximum_combinations_evaluated"), 256
    )
    maximum_correlation = _float(
        portfolio_policy.get("maximum_pairwise_correlation"), 0.65
    )
    unknown_fails_closed = bool(
        portfolio_policy.get("unknown_correlation_fails_closed", True)
    )
    correlation_index = _correlation_index(correlation_rows)
    source = list(sleeves)[:maximum_candidates]
    candidates: list[dict[str, Any]] = []
    evaluated = 0
    rejected_unknown = 0
    rejected_correlation = 0
    rejected_capacity = 0
    for size in range(1, min(maximum_sleeves, len(source)) + 1):
        for subset in itertools.combinations(source, size):
            if evaluated >= maximum_combinations:
                break
            evaluated += 1
            pair_rows = [
                correlation_index.get(
                    tuple(
                        sorted(
                            (str(left.get("sleeve_id")), str(right.get("sleeve_id")))
                        )
                    ),
                    {},
                )
                for left, right in itertools.combinations(subset, 2)
            ]
            unknown = [row for row in pair_rows if row.get("correlation") is None]
            if unknown and unknown_fails_closed:
                rejected_unknown += 1
                continue
            measured = [
                _float(row.get("correlation"))
                for row in pair_rows
                if row.get("correlation") is not None
            ]
            if any(abs(value) > maximum_correlation for value in measured):
                rejected_correlation += 1
                continue
            weights = _bounded_weights(
                subset,
                _float(portfolio_policy.get("minimum_sleeve_weight"), 0.15),
                _float(portfolio_policy.get("maximum_sleeve_weight"), 0.6),
            )
            allocations = []
            capacity_ready = True
            observed_headroom_ratios: list[float] = []
            for sleeve in subset:
                sleeve_id = str(sleeve.get("sleeve_id") or "")
                notional = max(capital_usd, 0.0) * weights.get(sleeve_id, 0.0)
                supported = _float(sleeve.get("maximum_supported_notional_usd"))
                required_supported = notional * max(
                    float(minimum_capacity_headroom_ratio), 1.0
                )
                if required_supported > supported + 1e-9:
                    capacity_ready = False
                observed_headroom = (
                    supported / notional if notional > 1e-12 else float("inf")
                )
                observed_headroom_ratios.append(observed_headroom)
                allocations.append(
                    {
                        "sleeve_id": sleeve_id,
                        "advisory_weight": weights.get(sleeve_id, 0.0),
                        "advisory_notional_usd": round(notional, 2),
                        "maximum_supported_notional_usd": round(supported, 2),
                        "observed_capacity_headroom_ratio": (
                            round(observed_headroom, 8)
                            if math.isfinite(observed_headroom)
                            else None
                        ),
                    }
                )
            if not capacity_ready:
                rejected_capacity += 1
                continue
            mean_score = statistics.fmean(_float(row.get("score")) for row in subset)
            mean_abs_correlation = (
                statistics.fmean(abs(value) for value in measured) if measured else 0.0
            )
            diversification = 1.0 - mean_abs_correlation if size > 1 else 0.0
            portfolio_score = (
                mean_score
                + _float(portfolio_policy.get("diversification_bonus_weight"), 0.1)
                * diversification
            )
            candidates.append(
                {
                    "plan_type": "single_sleeve" if size == 1 else "multi_sleeve",
                    "sleeve_count": size,
                    "sleeve_ids": [str(row.get("sleeve_id") or "") for row in subset],
                    "portfolio_score": round(_clamp(portfolio_score), 8),
                    "mean_absolute_pairwise_correlation": (
                        round(mean_abs_correlation, 8) if measured else None
                    ),
                    "correlation_evidence_ready": not unknown,
                    "target_capital_usd": round(max(capital_usd, 0.0), 2),
                    "minimum_capacity_headroom_ratio": round(
                        max(float(minimum_capacity_headroom_ratio), 1.0), 8
                    ),
                    "observed_capacity_headroom_ratio": round(
                        min(observed_headroom_ratios, default=0.0), 8
                    ),
                    "allocations": allocations,
                    "application_allowed": False,
                    "operator_review_required": True,
                    "live_execution_authority": False,
                }
            )
        if evaluated >= maximum_combinations:
            break
    candidates.sort(
        key=lambda row: (
            -_float(row.get("portfolio_score")),
            -_int(row.get("sleeve_count")),
            tuple(row.get("sleeve_ids") or []),
        )
    )
    return candidates, {
        "evaluated_combination_count": evaluated,
        "accepted_combination_count": len(candidates),
        "unknown_correlation_rejection_count": rejected_unknown,
        "excess_correlation_rejection_count": rejected_correlation,
        "capacity_rejection_count": rejected_capacity,
    }


def _largest_independent_subset(
    sleeves: Sequence[Mapping[str, Any]],
    correlation_rows: Sequence[Mapping[str, Any]],
    policy: Mapping[str, Any],
) -> list[Mapping[str, Any]]:
    portfolio_policy = _dict(policy.get("portfolio_selection"))
    maximum = min(
        _int(portfolio_policy.get("maximum_candidates_considered"), 8), len(sleeves)
    )
    correlation_index = _correlation_index(correlation_rows)
    threshold = _float(portfolio_policy.get("maximum_pairwise_correlation"), 0.65)
    source = list(sleeves)[:maximum]
    for size in range(len(source), 0, -1):
        for subset in itertools.combinations(source, size):
            valid = True
            for left, right in itertools.combinations(subset, 2):
                row = correlation_index.get(
                    tuple(
                        sorted(
                            (str(left.get("sleeve_id")), str(right.get("sleeve_id")))
                        )
                    ),
                    {},
                )
                if (
                    row.get("correlation") is None
                    or abs(_float(row.get("correlation"))) > threshold
                ):
                    valid = False
                    break
            if valid:
                return list(subset)
    return []


def _scalability_goals(
    eligible_sleeves: Sequence[Mapping[str, Any]],
    independent_subset: Sequence[Mapping[str, Any]],
    policy: Mapping[str, Any],
    *,
    global_economic_evidence_ready: bool,
    candidate_bound: bool,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    independent_count = len(independent_subset)
    minimum_days = min(
        (_int(row.get("independent_day_count")) for row in independent_subset),
        default=0,
    )
    minimum_regimes = min(
        (_int(row.get("distinct_regime_count")) for row in independent_subset),
        default=0,
    )
    supported_notional = sum(
        _float(row.get("maximum_supported_notional_usd")) for row in independent_subset
    )
    for goal in _list(policy.get("system_scalability_goals")):
        if not isinstance(goal, Mapping):
            continue
        required_sleeves = _int(goal.get("minimum_independent_sleeves"), 1)
        checks = {
            "candidate_bound": bool(candidate_bound),
            "qualified_sleeve_count": len(eligible_sleeves) >= required_sleeves,
            "low_correlation_sleeve_count": independent_count >= required_sleeves,
            "independent_days": minimum_days
            >= _int(goal.get("minimum_days_per_sleeve"), 0),
            "regime_breadth": minimum_regimes
            >= _int(goal.get("minimum_regimes_per_sleeve"), 0),
            "supported_notional": supported_notional
            >= _float(goal.get("minimum_supported_notional_usd"), 0.0),
            "global_economic_evidence": (
                global_economic_evidence_ready
                if bool(goal.get("require_global_economic_evidence", True))
                else True
            ),
        }
        ready_count = sum(bool(value) for value in checks.values())
        rows.append(
            {
                "goal_id": str(goal.get("goal_id") or ""),
                "status": "earned" if all(checks.values()) else "collecting",
                "progress_percent": round(100.0 * ready_count / max(len(checks), 1), 2),
                "checks": checks,
                "requirements": dict(goal),
                "observed": {
                    "qualified_sleeve_count": len(eligible_sleeves),
                    "low_correlation_sleeve_count": independent_count,
                    "minimum_independent_days": minimum_days,
                    "minimum_distinct_regimes": minimum_regimes,
                    "supported_notional_usd": round(supported_notional, 2),
                    "independent_sleeve_ids": [
                        str(row.get("sleeve_id") or "") for row in independent_subset
                    ],
                },
                "automatic_progression": False,
                "operator_review_required": True,
            }
        )
    return rows


def _sleeve_parameter_contract(
    policy: Mapping[str, Any],
    sleeves: Sequence[Mapping[str, Any]],
    global_blockers: Sequence[Any],
    *,
    target_notional_usd: float,
) -> dict[str, Any]:
    evidence_policy = _dict(policy.get("sleeve_evidence"))
    portfolio_policy = _dict(policy.get("portfolio_selection"))
    required_parameters = {
        "minimum_qualified_bots": _int(evidence_policy.get("minimum_qualified_bots")),
        "minimum_candidate_samples": _int(
            evidence_policy.get("minimum_candidate_samples")
        ),
        "minimum_independent_days": _int(
            evidence_policy.get("minimum_independent_days")
        ),
        "minimum_distinct_regimes": _int(
            evidence_policy.get("minimum_distinct_regimes")
        ),
        "minimum_positive_day_ratio": _float(
            evidence_policy.get("minimum_positive_day_ratio")
        ),
        "minimum_post_cost_lcb_bps_exclusive": _float(
            evidence_policy.get("minimum_post_cost_lcb_bps_exclusive")
        ),
        "maximum_drawdown_to_turnover_ratio": _float(
            evidence_policy.get("maximum_drawdown_to_turnover_ratio")
        ),
        "minimum_correlation_clusters": _int(
            evidence_policy.get("minimum_correlation_clusters")
        ),
        "minimum_supported_notional_usd": max(
            _float(evidence_policy.get("minimum_supported_notional_usd")),
            float(target_notional_usd),
        ),
        "maximum_pairwise_correlation": _float(
            portfolio_policy.get("maximum_pairwise_correlation")
        ),
    }
    rows: list[dict[str, Any]] = []
    for sleeve in sleeves:
        application_blockers = _unique(_list(sleeve.get("application_blockers")))
        evidence_blockers = _unique(_list(sleeve.get("evidence_blockers")))
        status = (
            "application_eligible"
            if bool(sleeve.get("application_eligible", False))
            else (
                "route_research_ready"
                if bool(sleeve.get("research_eligible", False))
                and bool(sleeve.get("route_match", False))
                else (
                    "research_ready_route_blocked"
                    if bool(sleeve.get("research_eligible", False))
                    else "collecting_evidence"
                )
            )
        )
        rows.append(
            {
                "sleeve_id": str(sleeve.get("sleeve_id") or ""),
                "status": status,
                "route_match": bool(sleeve.get("route_match", False)),
                "research_eligible": bool(sleeve.get("research_eligible", False)),
                "application_eligible": bool(sleeve.get("application_eligible", False)),
                "measured_parameters": {
                    "qualified_bot_count": _int(sleeve.get("qualified_bot_count")),
                    "candidate_sample_count": _int(
                        sleeve.get("candidate_sample_count")
                    ),
                    "effective_sample_count": _int(
                        sleeve.get("effective_sample_count")
                    ),
                    "independent_day_count": _int(sleeve.get("independent_day_count")),
                    "distinct_regime_count": _int(sleeve.get("distinct_regime_count")),
                    "positive_day_ratio": _float(sleeve.get("positive_day_ratio")),
                    "conservative_post_cost_lcb_bps": sleeve.get(
                        "conservative_post_cost_lcb_bps"
                    ),
                    "drawdown_to_turnover_ratio": _float(
                        sleeve.get("drawdown_to_turnover_ratio")
                    ),
                    "independent_correlation_cluster_count": _int(
                        sleeve.get("independent_correlation_cluster_count")
                    ),
                    "maximum_supported_notional_usd": _float(
                        sleeve.get("maximum_supported_notional_usd")
                    ),
                    "mean_regime_compatibility_score": _float(
                        sleeve.get("mean_regime_compatibility_score")
                    ),
                    "score": _float(sleeve.get("score")),
                },
                "score_components": _dict(sleeve.get("score_components")),
                "eligibility_checks": _dict(sleeve.get("eligibility_checks")),
                "blockers": _unique(application_blockers or evidence_blockers),
                "release_condition": "eligibility_checks_true_route_match_true_and_global_application_gates_clear",
            }
        )
    return {
        "contract_id": "sleeve_specific_parameter_contract_v1",
        "selection_basis": "sleeve_specific_post_cost_persistence_capacity_drawdown_regime_fit_and_correlation",
        "required_parameters": required_parameters,
        "global_blockers": _unique(global_blockers),
        "sleeves": rows,
        "application_eligible_sleeve_count": sum(
            1 for row in rows if row["application_eligible"]
        ),
        "research_eligible_sleeve_count": sum(
            1 for row in rows if row["research_eligible"]
        ),
        "route_matched_sleeve_count": sum(1 for row in rows if row["route_match"]),
        "unknown_correlation_fails_closed": bool(
            portfolio_policy.get("unknown_correlation_fails_closed", True)
        ),
        "paper_execution_authority": False,
        "live_execution_authority": False,
        "automatic_capital_scaling": False,
        "operator_review_required": True,
    }


def build_selector_payload(
    policy: Mapping[str, Any],
    profitability_manifest: Mapping[str, Any],
    hierarchy: Mapping[str, Any],
    graduation: Mapping[str, Any],
    account_study: Mapping[str, Any],
    profitability_firewall: Mapping[str, Any],
    execution_calibration: Mapping[str, Any],
    *,
    account_policy_key: str = "",
    execution_route_id: str = "",
    capital_usd: float | None = None,
    now_utc: datetime | None = None,
) -> dict[str, Any]:
    now = now_utc or datetime.now(timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    now = now.astimezone(timezone.utc)
    policy_errors = validate_policy(policy)
    identity = _dict(graduation.get("identity"))
    candidate_binding = _dict(profitability_manifest.get("candidate_binding"))
    candidate_id = str(candidate_binding.get("candidate_id") or "")
    graduation_candidate_id = str(identity.get("candidate_id") or "")
    account_key = account_policy_key or str(identity.get("account_policy_key") or "")
    route_id = execution_route_id or str(identity.get("execution_route_id") or "")
    route = _dict(_dict(policy.get("application_routes")).get(route_id))
    account = _account_context(account_study, account_key)
    account_scope = _account_scaling_scope(
        account_study,
        policy,
        active_account_policy_key=account_key,
    )
    tier = _active_tier(graduation)
    tier_limits = _dict(tier.get("limits"))
    target_capital = (
        max(float(capital_usd), 0.0)
        if capital_usd is not None
        else max(_float(tier_limits.get("account_capital_usd")), 0.0)
    )
    target_notional = max(_float(tier_limits.get("max_order_notional_usd")), 0.0)
    if target_notional <= 0.0:
        target_notional = target_capital

    sources = {
        "profitability_manifest": profitability_manifest,
        "bot_hierarchy": hierarchy,
        "live_canary_graduation": graduation,
        "account_position_study": account_study,
        "profitability_firewall": profitability_firewall,
        "paper_execution_calibration": execution_calibration,
    }
    freshness = {
        name: _source_freshness(name, payload, policy, now)
        for name, payload in sources.items()
    }
    sources_fresh = all(row.get("fresh", False) for row in freshness.values())
    route_compatible, route_blockers = _route_compatible(route, account)
    account_route_allowed = route_id in set(account.get("allowed_live_routes") or [])
    configured_cap = _float(account.get("configured_canary_cap_usd"))
    global_economic_ready = bool(
        profitability_firewall.get("promotion_evidence_ready", False)
        and profitability_firewall.get("economic_evidence_ready", False)
    )
    independent_execution_ready = bool(
        execution_calibration.get("independent_evidence_ready", False)
        and _int(execution_calibration.get("independent_samples")) > 0
    )
    candidate_identity_match = bool(
        candidate_id
        and graduation_candidate_id
        and candidate_id == graduation_candidate_id
    )
    graduation_blockers = _unique(_list(graduation.get("blockers")))
    global_checks = {
        "sources_fresh": sources_fresh,
        "candidate_bound": bool(candidate_binding.get("bound", False)),
        "candidate_identity_match": candidate_identity_match,
        "account_found": bool(account.get("found", False)),
        "account_classification_complete": bool(
            account.get("classification_complete", False)
        ),
        "account_preflight_ready": bool(account.get("account_preflight_ready", False)),
        "execution_route_cataloged": bool(route),
        "execution_route_account_allowed": account_route_allowed,
        "execution_route_account_compatible": route_compatible,
        "configured_cap_covers_target": configured_cap + 1e-9 >= target_capital > 0.0,
        "active_tier_review_eligible": bool(
            tier.get("operator_review_eligible", False)
        ),
        "profitability_firewall_ready": global_economic_ready,
        "independent_execution_calibration_ready": independent_execution_ready,
        "graduation_control_clear": not graduation_blockers,
    }
    global_blockers = [name for name, ready in global_checks.items() if not ready]
    global_blockers.extend(route_blockers)
    global_blockers.extend(
        str(value) for value in account.get("preflight_blockers") or []
    )
    global_blockers.extend(f"graduation:{value}" for value in graduation_blockers)
    global_blockers = _unique(global_blockers)

    assignments = {
        str(row.get("bot_id") or ""): row
        for row in _list(hierarchy.get("assignments"))
        if isinstance(row, Mapping) and str(row.get("bot_id") or "")
    }
    profiles = [
        row
        for row in _list(profitability_manifest.get("profiles"))
        if isinstance(row, Mapping)
    ]
    sleeves = _aggregate_sleeves(
        profiles,
        assignments,
        policy,
        target_notional_usd=target_notional,
        candidate_bound=bool(candidate_binding.get("bound", False)),
    )
    route_sleeves = {
        str(value) for value in _list(route.get("eligible_sleeve_ids")) if str(value)
    }
    for sleeve in sleeves:
        sleeve["route_match"] = str(sleeve.get("sleeve_id") or "") in route_sleeves
        sleeve["application_eligible"] = bool(
            sleeve.get("research_eligible", False)
            and sleeve.get("route_match", False)
            and all(global_checks.values())
            and not route_blockers
        )
        application_blockers = list(sleeve.get("evidence_blockers") or [])
        if not sleeve.get("route_match", False):
            application_blockers.append("sleeve_not_allowed_for_execution_route")
        application_blockers.extend(global_blockers)
        sleeve["application_blockers"] = _unique(application_blockers)

    research_eligible = [row for row in sleeves if row.get("research_eligible", False)]
    application_eligible = [
        row for row in sleeves if row.get("application_eligible", False)
    ]
    correlation_rows = _pairwise_correlations(
        research_eligible,
        _int(_dict(policy.get("portfolio_selection")).get("minimum_common_days"), 5),
    )
    tier_maximum = _int(
        _dict(
            _dict(policy.get("portfolio_selection")).get("maximum_sleeves_by_tier")
        ).get(str(tier.get("tier") or "")),
        _int(
            _dict(policy.get("portfolio_selection")).get("default_maximum_sleeves"), 1
        ),
    )
    maximum_sleeves = max(
        min(tier_maximum, _int(route.get("maximum_sleeves"), tier_maximum)), 1
    )
    portfolio_candidates, search_summary = _portfolio_candidates(
        application_eligible,
        correlation_rows,
        policy,
        capital_usd=target_capital,
        maximum_sleeves=maximum_sleeves,
    )
    selected_plan = portfolio_candidates[0] if portfolio_candidates else None
    growth_context = _growth_target_context(
        graduation,
        policy,
        active_capital_usd=target_capital,
    )
    growth_target = _dict(growth_context.get("next_target"))
    growth_target_capital = _float(
        growth_context.get("next_target_capital_usd"), target_capital
    )
    growth_tier_id = str(growth_context.get("next_target_tier_id") or "")
    growth_deployment = _dict(growth_context.get("next_target_deployment_requirements"))
    growth_tier_maximum = _int(
        _dict(
            _dict(policy.get("portfolio_selection")).get("maximum_sleeves_by_tier")
        ).get(growth_tier_id),
        _int(route.get("maximum_sleeves"), maximum_sleeves),
    )
    growth_maximum_sleeves = max(
        min(
            growth_tier_maximum,
            _int(route.get("maximum_sleeves"), growth_tier_maximum),
        ),
        1,
    )
    if growth_target:
        growth_portfolio_candidates, growth_search_summary = _portfolio_candidates(
            application_eligible,
            correlation_rows,
            policy,
            capital_usd=growth_target_capital,
            maximum_sleeves=growth_maximum_sleeves,
            minimum_capacity_headroom_ratio=_float(
                growth_deployment.get("minimum_capacity_headroom_ratio"), 1.0
            ),
        )
    else:
        growth_portfolio_candidates, growth_search_summary = [], {
            "evaluated_combination_count": 0,
            "accepted_combination_count": 0,
            "unknown_correlation_rejection_count": 0,
            "excess_correlation_rejection_count": 0,
            "capacity_rejection_count": 0,
        }
    growth_selected_plan = (
        growth_portfolio_candidates[0] if growth_portfolio_candidates else None
    )
    capital_growth_plan = _capital_growth_plan(
        graduation,
        policy,
        active_capital_usd=target_capital,
        target_portfolio_plan=growth_selected_plan,
        candidate_bound=bool(candidate_binding.get("bound", False)),
        sources_fresh=sources_fresh,
        global_economic_ready=global_economic_ready,
        independent_execution_ready=independent_execution_ready,
    )
    independent_subset = _largest_independent_subset(
        research_eligible, correlation_rows, policy
    )
    goals = _scalability_goals(
        research_eligible,
        independent_subset,
        policy,
        global_economic_evidence_ready=global_economic_ready,
        candidate_bound=bool(candidate_binding.get("bound", False)),
    )

    hard_integrity_blockers: list[str] = []
    if (
        candidate_id
        and graduation_candidate_id
        and candidate_id != graduation_candidate_id
    ):
        hard_integrity_blockers.append("candidate_identity_mismatch")
    if policy_errors:
        hard_integrity_blockers.extend(policy_errors)
    structurally_ready = not hard_integrity_blockers
    recommendation_ready = bool(selected_plan and all(global_checks.values()))
    status = (
        "degraded"
        if policy_errors
        else (
            "blocked"
            if hard_integrity_blockers
            else "ready" if recommendation_ready else "ready_with_evidence_debt"
        )
    )
    sleeve_parameter_contract = _sleeve_parameter_contract(
        policy,
        sleeves,
        global_blockers,
        target_notional_usd=target_notional,
    )
    sleeve_operating_contract = build_operating_contract(
        contract_id="sleeve_scalability_selector_operating_contract_v1",
        owner="sleeve_scalability_selector",
        domain="sleeve_selection",
        status=status,
        why=(
            "recommendation_ready"
            if recommendation_ready
            else (global_blockers[0] if global_blockers else "no_selected_plan")
        ),
        safe_authority=[
            "rank_sleeves",
            "measure_sleeve_specific_parameters",
            "publish_advisory_portfolio_candidates",
            "publish_organic_growth_readiness",
        ],
        blocked_authority=[
            "runtime_route_mutation",
            "paper_allocation_change",
            "live_allocation_change",
            "automatic_stage_progression",
            "automatic_capital_scaling",
            "order_payload_creation",
        ],
        evidence_missing=_unique(
            [*global_blockers]
            + [
                f"{row.get('sleeve_id')}:{blocker}"
                for row in sleeves
                if row.get("route_match")
                for blocker in _list(row.get("evidence_blockers"))
            ]
        ),
        release_conditions=[
            "all_global_application_gates_true",
            "at_least_one_route_matched_sleeve_application_eligible",
            "selected_plan_capacity_headroom_ready",
            "pairwise_correlation_evidence_ready_or_safely_single_sleeve",
            "operator_review_accepts_advisory_plan_before_any_application",
        ],
        next_commands=[
            ["./scripts/ops/opsctl.sh", "sleeve-scalability-selector", "--json"],
            ["./scripts/ops/opsctl.sh", "sleeve-profitability-dashboard", "--json"],
            ["./scripts/ops/opsctl.sh", "bot-profitability-scalability", "--json"],
        ],
        definition_gaps=[
            (
                "route_matched_sleeves_need_application_evidence"
                if not application_eligible
                else ""
            ),
            (
                "source_freshness_blocks_sleeve_application"
                if "sources_fresh" in global_blockers
                else ""
            ),
            (
                "candidate_binding_blocks_sleeve_application"
                if "candidate_bound" in global_blockers
                else ""
            ),
        ],
        measurement={
            "evaluated_sleeve_count": len(sleeves),
            "research_eligible_sleeve_count": len(research_eligible),
            "route_matched_sleeve_count": sum(
                bool(row.get("route_match")) for row in sleeves
            ),
            "application_eligible_sleeve_count": len(application_eligible),
            "accepted_portfolio_candidate_count": len(portfolio_candidates),
            "global_blocker_count": len(global_blockers),
        },
        hardening={
            "unknown_correlation_fails_closed": bool(
                _dict(policy.get("portfolio_selection")).get(
                    "unknown_correlation_fails_closed", True
                )
            ),
            "advisory_only": True,
            "application_allowed": False,
            "profitability_guaranteed": False,
        },
    )
    payload: dict[str, Any] = {
        "timestamp_utc": now.isoformat(),
        "schema_version": 1,
        "policy_id": str(policy.get("policy_id") or ""),
        "ok": structurally_ready,
        "overall_status": status,
        "selection_mode": "best_supported_for_current_account_route_regime_and_capital",
        "control_ready": structurally_ready,
        "recommendation_ready": recommendation_ready,
        "identity": {
            "candidate_id": candidate_id,
            "graduation_candidate_id": graduation_candidate_id,
            "account_policy_key": account_key,
            "execution_route_id": route_id,
            "active_capital_tier": str(tier.get("tier") or ""),
        },
        "target": {
            "capital_usd": round(target_capital, 2),
            "maximum_order_notional_usd": round(target_notional, 2),
            "maximum_sleeves_for_active_tier": maximum_sleeves,
            "route_eligible_sleeve_ids": sorted(route_sleeves),
        },
        "account_context": account,
        "account_scaling_scope": account_scope,
        "host_portability": {
            **_dict(policy.get("host_portability")),
            "status": "portable_policy_requires_new_host_rebind",
            "live_execution_authority": False,
        },
        "source_freshness": {
            "all_sources_fresh": sources_fresh,
            "sources": freshness,
            "stale_sources": sorted(
                name for name, row in freshness.items() if not row.get("fresh")
            ),
        },
        "global_application_gates": global_checks,
        "global_application_blockers": global_blockers,
        "hard_integrity_blockers": hard_integrity_blockers,
        "summary": {
            "catalog_profile_count": len(profiles),
            "organized_assignment_count": len(assignments),
            "evaluated_sleeve_count": len(sleeves),
            "research_eligible_sleeve_count": len(research_eligible),
            "route_matched_sleeve_count": sum(
                bool(row.get("route_match")) for row in sleeves
            ),
            "application_eligible_sleeve_count": len(application_eligible),
            "earned_scalability_goal_count": sum(
                row.get("status") == "earned" for row in goals
            ),
            "scalability_goal_count": len(goals),
            "classified_account_policy_count": _int(
                account_scope.get("classified_account_policy_count")
            ),
            "organic_capital_usd": _float(
                capital_growth_plan.get("organic_capital_usd")
            ),
            "next_organic_capital_target_usd": _float(
                capital_growth_plan.get("next_target_capital_usd")
            ),
            "organic_growth_progress_percent": _float(
                capital_growth_plan.get("progress_to_next_target_percent")
            ),
            "organic_growth_action": str(
                capital_growth_plan.get("recommended_action") or ""
            ),
            "long_range_organic_target_usd": _float(
                capital_growth_plan.get("long_range_target_capital_usd")
            ),
        },
        "best_supported_research_sleeve": (
            str(research_eligible[0].get("sleeve_id") or "")
            if research_eligible
            else ""
        ),
        "best_supported_route_sleeve": (
            str(
                next(
                    (
                        row.get("sleeve_id")
                        for row in sleeves
                        if row.get("route_match") and row.get("research_eligible")
                    ),
                    "",
                )
                or ""
            )
        ),
        "sleeve_rankings": sleeves,
        "sleeve_parameter_contract": sleeve_parameter_contract,
        "pairwise_correlations": correlation_rows,
        "portfolio_search": search_summary,
        "advisory_portfolio_candidates": portfolio_candidates[:20],
        "selected_advisory_plan": selected_plan,
        "capital_growth_plan": capital_growth_plan,
        "growth_portfolio_search": growth_search_summary,
        "growth_advisory_portfolio_candidates": growth_portfolio_candidates[:20],
        "system_scalability_goals": goals,
        "evidence_debt": _unique(
            [*global_blockers]
            + [
                f"{row.get('sleeve_id')}:{blocker}"
                for row in sleeves
                if row.get("route_match")
                for blocker in _list(row.get("evidence_blockers"))
            ]
            + [
                f"capital_growth:{blocker}"
                for blocker in _list(capital_growth_plan.get("target_blockers"))
            ]
        ),
        "authority_contract": {
            "advisory_only": True,
            "application_allowed": False,
            "changes_runtime_decisions": False,
            "changes_paper_allocation": False,
            "changes_live_allocation": False,
            "automatic_stage_progression": False,
            "automatic_capital_scaling": False,
            "paper_execution_authority": False,
            "live_execution_authority": False,
            "allowlist_authority": False,
            "order_payload_created": False,
            "operator_review_required": True,
            "profitability_guaranteed": False,
        },
        "operating_contract": sleeve_operating_contract,
        "sleeve_operating_contract": sleeve_operating_contract,
    }
    payload["selection_receipt_sha256"] = canonical_hash(
        {
            "policy_id": payload["policy_id"],
            "identity": payload["identity"],
            "target": payload["target"],
            "global_application_gates": payload["global_application_gates"],
            "sleeve_rankings": payload["sleeve_rankings"],
            "pairwise_correlations": payload["pairwise_correlations"],
            "advisory_portfolio_candidates": payload["advisory_portfolio_candidates"],
            "capital_growth_plan": payload["capital_growth_plan"],
            "growth_advisory_portfolio_candidates": payload[
                "growth_advisory_portfolio_candidates"
            ],
            "system_scalability_goals": payload["system_scalability_goals"],
            "authority_contract": payload["authority_contract"],
        }
    )
    return payload
