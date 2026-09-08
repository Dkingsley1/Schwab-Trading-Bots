from __future__ import annotations

import itertools
import math
import random
from collections.abc import Mapping, Sequence
from typing import Any


def _finite_values(values: Sequence[Any]) -> list[float]:
    out: list[float] = []
    for raw in values:
        try:
            value = float(raw)
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            out.append(value)
    return out


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _variance(values: Sequence[float], *, sample: bool = True) -> float:
    if len(values) < (2 if sample else 1):
        return 0.0
    center = _mean(values)
    divisor = len(values) - 1 if sample else len(values)
    return sum((value - center) ** 2 for value in values) / divisor


def _normal_cdf(value: float) -> float:
    return 0.5 * (1.0 + math.erf(value / math.sqrt(2.0)))


def _status(available: bool, passes: bool) -> str:
    if not available:
        return "insufficient_evidence"
    return "supported" if passes else "not_supported"


def sequential_sign_sprt(
    returns_bps: Sequence[Any],
    *,
    null_win_probability: float = 0.50,
    alternative_win_probability: float = 0.57,
    alpha: float = 0.05,
    beta: float = 0.20,
    minimum_observations: int = 20,
    hurdle_bps: float = 0.0,
) -> dict[str, Any]:
    """Fixed-alternative Bernoulli e-process over post-cost outcome signs."""

    values = _finite_values(returns_bps)
    p0 = min(max(float(null_win_probability), 1e-6), 1.0 - 1e-6)
    p1 = min(max(float(alternative_win_probability), p0 + 1e-6), 1.0 - 1e-6)
    outcomes = [1 if value > float(hurdle_bps) else 0 for value in values]
    log_e = 0.0
    max_log_e = 0.0
    for outcome in outcomes:
        if outcome:
            log_e += math.log(p1 / p0)
        else:
            log_e += math.log((1.0 - p1) / (1.0 - p0))
        max_log_e = max(max_log_e, log_e)

    upper = math.log((1.0 - beta) / alpha)
    lower = math.log(beta / (1.0 - alpha))
    available = len(outcomes) >= max(int(minimum_observations), 1)
    supports = bool(available and log_e >= upper and _mean(values) > hurdle_bps)
    rejects = bool(available and log_e <= lower)
    decision = "support_positive_edge" if supports else "reject_positive_edge" if rejects else "continue_sampling"
    return {
        "method": "fixed_alternative_bernoulli_sprt_e_process",
        "status": _status(available, supports),
        "available": available,
        "passes": supports,
        "decision": decision,
        "observation_count": len(outcomes),
        "minimum_observations": max(int(minimum_observations), 1),
        "win_count": sum(outcomes),
        "loss_or_hurdle_count": len(outcomes) - sum(outcomes),
        "win_rate": round(_mean(outcomes), 8) if outcomes else None,
        "mean_post_cost_return_bps": round(_mean(values), 8) if values else None,
        "log_e_value": round(log_e, 8),
        "e_value": round(math.exp(min(log_e, 50.0)), 8),
        "always_valid_p_value": round(min(1.0, math.exp(-max_log_e)), 8),
        "support_boundary_log": round(upper, 8),
        "rejection_boundary_log": round(lower, 8),
        "null_win_probability": p0,
        "alternative_win_probability": p1,
        "hurdle_bps": float(hurdle_bps),
    }


def _moving_block_indices(
    count: int,
    *,
    block_length: int,
    rng: random.Random,
) -> list[int]:
    indices: list[int] = []
    block = max(min(int(block_length), count), 1)
    while len(indices) < count:
        start = rng.randrange(count)
        indices.extend((start + offset) % count for offset in range(block))
    return indices[:count]


def block_bootstrap_model_selection(
    profile_returns_bps: Mapping[str, Sequence[Any]],
    *,
    replications: int = 500,
    block_length: int = 3,
    alpha: float = 0.05,
    minimum_periods: int = 12,
    seed: int = 1729,
) -> dict[str, Any]:
    """White Reality Check plus a studentized SPA-style block bootstrap."""

    series = {
        str(profile): _finite_values(values)
        for profile, values in sorted(profile_returns_bps.items())
        if str(profile).strip()
    }
    series = {profile: values for profile, values in series.items() if values}
    period_count = min((len(values) for values in series.values()), default=0)
    available = bool(len(series) >= 2 and period_count >= max(int(minimum_periods), 4))
    if not available:
        return {
            "method": "white_reality_check_and_hansen_spa_block_bootstrap",
            "status": "insufficient_evidence",
            "available": False,
            "passes": False,
            "profile_count": len(series),
            "period_count": period_count,
            "minimum_periods": max(int(minimum_periods), 4),
            "replications": max(int(replications), 100),
            "blockers": [
                *(["at_least_two_aligned_profiles_required"] if len(series) < 2 else []),
                *(["aligned_period_floor_not_met"] if period_count < max(int(minimum_periods), 4) else []),
            ],
        }

    aligned = {profile: values[-period_count:] for profile, values in series.items()}
    means = {profile: _mean(values) for profile, values in aligned.items()}
    stds = {
        profile: max(math.sqrt(_variance(values)), 1e-12)
        for profile, values in aligned.items()
    }
    sqrt_n = math.sqrt(period_count)
    observed_white = max(sqrt_n * value for value in means.values())
    observed_spa = max(
        sqrt_n * means[profile] / stds[profile] for profile in aligned
    )
    centered = {
        profile: [value - means[profile] for value in values]
        for profile, values in aligned.items()
    }
    rng = random.Random(int(seed))
    reps = max(int(replications), 100)
    white_exceed = 0
    spa_exceed = 0
    for _ in range(reps):
        indices = _moving_block_indices(
            period_count,
            block_length=block_length,
            rng=rng,
        )
        boot_means = {
            profile: _mean([values[index] for index in indices])
            for profile, values in centered.items()
        }
        white_stat = max(sqrt_n * value for value in boot_means.values())
        spa_stat = max(
            sqrt_n * boot_means[profile] / stds[profile]
            for profile in aligned
        )
        white_exceed += int(white_stat >= observed_white)
        spa_exceed += int(spa_stat >= observed_spa)

    white_p = (white_exceed + 1.0) / (reps + 1.0)
    spa_p = (spa_exceed + 1.0) / (reps + 1.0)
    passes = bool(white_p <= alpha and spa_p <= alpha and max(means.values()) > 0.0)
    best_profile = max(means, key=lambda profile: (means[profile], profile))
    return {
        "method": "white_reality_check_and_hansen_spa_block_bootstrap",
        "status": _status(True, passes),
        "available": True,
        "passes": passes,
        "profile_count": len(aligned),
        "period_count": period_count,
        "minimum_periods": max(int(minimum_periods), 4),
        "replications": reps,
        "block_length": max(int(block_length), 1),
        "seed": int(seed),
        "alpha": float(alpha),
        "best_profile": best_profile,
        "best_mean_post_cost_return_bps": round(means[best_profile], 8),
        "white_reality_check_p_value": round(white_p, 8),
        "hansen_spa_p_value": round(spa_p, 8),
        "observed_white_statistic": round(observed_white, 8),
        "observed_studentized_spa_statistic": round(observed_spa, 8),
        "profile_mean_returns_bps": {
            profile: round(value, 8) for profile, value in sorted(means.items())
        },
    }


def probabilistic_sharpe_bayesian_utility(
    returns_bps: Sequence[Any],
    *,
    annualization_periods: int = 252,
    reference_sharpe: float = 0.0,
    minimum_observations: int = 20,
    posterior_draws: int = 1000,
    posterior_probability_floor: float = 0.95,
    prior_strength: float = 5.0,
    prior_scale_bps: float = 25.0,
    risk_aversion: float = 3.0,
    seed: int = 1733,
) -> dict[str, Any]:
    values_bps = _finite_values(returns_bps)
    values = [value / 10_000.0 for value in values_bps]
    n = len(values)
    std = math.sqrt(_variance(values))
    available = bool(n >= max(int(minimum_observations), 3) and std > 0.0)
    if not available:
        return {
            "method": "probabilistic_sharpe_and_bayesian_risk_utility",
            "status": "insufficient_evidence",
            "available": False,
            "passes": False,
            "observation_count": n,
            "minimum_observations": max(int(minimum_observations), 3),
            "blockers": ["sample_floor_or_nonzero_variance_not_met"],
        }

    center = _mean(values)
    standardized = [(value - center) / std for value in values]
    skew = _mean([value**3 for value in standardized])
    kurtosis = _mean([value**4 for value in standardized])
    daily_sharpe = center / std
    reference_daily = float(reference_sharpe) / math.sqrt(max(annualization_periods, 1))
    denominator = math.sqrt(
        max(
            1.0
            - skew * daily_sharpe
            + ((kurtosis - 1.0) / 4.0) * daily_sharpe**2,
            1e-12,
        )
    )
    psr_z = (daily_sharpe - reference_daily) * math.sqrt(n - 1) / denominator
    psr = _normal_cdf(psr_z)

    kappa0 = max(float(prior_strength), 1e-6)
    mu0 = 0.0
    alpha0 = 2.0
    prior_scale = max(float(prior_scale_bps), 1e-6) / 10_000.0
    beta0 = prior_scale**2 * (alpha0 - 1.0)
    kappa_n = kappa0 + n
    mu_n = (kappa0 * mu0 + n * center) / kappa_n
    alpha_n = alpha0 + n / 2.0
    centered_ss = sum((value - center) ** 2 for value in values)
    beta_n = (
        beta0
        + 0.5 * centered_ss
        + (kappa0 * n * (center - mu0) ** 2) / (2.0 * kappa_n)
    )
    rng = random.Random(int(seed))
    draws = max(int(posterior_draws), 200)
    positive_mean = 0
    positive_utility = 0
    utility_total = 0.0
    for _ in range(draws):
        precision = rng.gammavariate(alpha_n, 1.0 / max(beta_n, 1e-18))
        sigma2 = 1.0 / max(precision, 1e-18)
        sampled_mean = rng.gauss(mu_n, math.sqrt(sigma2 / kappa_n))
        utility = sampled_mean - 0.5 * float(risk_aversion) * sigma2
        positive_mean += int(sampled_mean > 0.0)
        positive_utility += int(utility > 0.0)
        utility_total += utility
    mean_probability = positive_mean / draws
    utility_probability = positive_utility / draws
    utility_mean = utility_total / draws
    passes = bool(
        psr >= posterior_probability_floor
        and utility_probability >= posterior_probability_floor
    )
    return {
        "method": "probabilistic_sharpe_and_bayesian_risk_utility",
        "status": _status(True, passes),
        "available": True,
        "passes": passes,
        "observation_count": n,
        "minimum_observations": max(int(minimum_observations), 3),
        "annualized_sharpe": round(daily_sharpe * math.sqrt(annualization_periods), 8),
        "reference_annualized_sharpe": float(reference_sharpe),
        "probabilistic_sharpe_probability": round(psr, 8),
        "posterior_positive_mean_probability": round(mean_probability, 8),
        "posterior_positive_risk_utility_probability": round(utility_probability, 8),
        "posterior_mean_return_bps": round(mu_n * 10_000.0, 8),
        "posterior_mean_risk_utility_bps": round(utility_mean * 10_000.0, 8),
        "posterior_probability_floor": float(posterior_probability_floor),
        "posterior_draws": draws,
        "seed": int(seed),
        "sample_skew": round(skew, 8),
        "sample_kurtosis": round(kurtosis, 8),
    }


def _path_drawdown(returns: Sequence[float], fraction: float) -> tuple[float, float]:
    wealth = 1.0
    peak = 1.0
    max_drawdown = 0.0
    log_growth = 0.0
    for value in returns:
        gross = max(1.0 + fraction * value, 1e-12)
        wealth *= gross
        log_growth += math.log(gross)
        peak = max(peak, wealth)
        max_drawdown = max(max_drawdown, 1.0 - wealth / peak)
    return log_growth / max(len(returns), 1), max_drawdown


def drawdown_constrained_kelly(
    returns_bps: Sequence[Any],
    *,
    minimum_observations: int = 20,
    max_fraction: float = 0.25,
    drawdown_limit: float = 0.10,
    grid_steps: int = 50,
) -> dict[str, Any]:
    values = [value / 10_000.0 for value in _finite_values(returns_bps)]
    available = len(values) >= max(int(minimum_observations), 3)
    if not available:
        return {
            "method": "drawdown_constrained_fractional_kelly",
            "status": "insufficient_evidence",
            "available": False,
            "passes": False,
            "observation_count": len(values),
            "minimum_observations": max(int(minimum_observations), 3),
        }
    variance = _variance(values)
    unconstrained = _mean(values) / variance if variance > 0.0 else 0.0
    cap = max(min(float(max_fraction), 1.0), 0.0)
    steps = max(int(grid_steps), 2)
    candidates = [cap * index / steps for index in range(steps + 1)]
    feasible: list[tuple[float, float, float]] = []
    for fraction in candidates:
        growth, drawdown = _path_drawdown(values, fraction)
        if drawdown <= float(drawdown_limit) + 1e-12:
            feasible.append((growth, -drawdown, fraction))
    best_growth, neg_drawdown, best_fraction = max(feasible) if feasible else (0.0, 0.0, 0.0)
    passes = bool(best_fraction > 0.0 and best_growth > 0.0)
    return {
        "method": "drawdown_constrained_fractional_kelly",
        "status": _status(True, passes),
        "available": True,
        "passes": passes,
        "observation_count": len(values),
        "minimum_observations": max(int(minimum_observations), 3),
        "unconstrained_kelly_fraction": round(unconstrained, 8),
        "challenger_fraction": round(best_fraction, 8),
        "maximum_fraction": cap,
        "drawdown_limit": float(drawdown_limit),
        "challenger_max_drawdown": round(-neg_drawdown, 8),
        "expected_log_growth_per_observation": round(best_growth, 12),
        "authority": "diagnostic_only_no_sizing_authority",
    }


def entropy_pooling_downside_view(
    returns_bps: Sequence[Any],
    *,
    minimum_observations: int = 20,
    tail_quantile: float = 0.25,
    target_tail_probability: float = 0.35,
) -> dict[str, Any]:
    values = _finite_values(returns_bps)
    available = len(values) >= max(int(minimum_observations), 4)
    if not available:
        return {
            "method": "minimum_relative_entropy_downside_view",
            "status": "insufficient_evidence",
            "available": False,
            "passes": False,
            "observation_count": len(values),
            "minimum_observations": max(int(minimum_observations), 4),
        }
    quantile = min(max(float(tail_quantile), 0.05), 0.50)
    ordered = sorted(values)
    cutoff_index = max(min(math.ceil(len(values) * quantile) - 1, len(values) - 1), 0)
    cutoff = ordered[cutoff_index]
    tail = [index for index, value in enumerate(values) if value <= cutoff]
    body = [index for index in range(len(values)) if index not in set(tail)]
    base_tail = len(tail) / len(values)
    target = min(max(float(target_tail_probability), base_tail), 0.95)
    weights = [0.0] * len(values)
    for index in tail:
        weights[index] = target / len(tail)
    for index in body:
        weights[index] = (1.0 - target) / max(len(body), 1)
    base_weight = 1.0 / len(values)
    relative_entropy = sum(
        weight * math.log(weight / base_weight)
        for weight in weights
        if weight > 0.0
    )
    stressed_mean = sum(weight * value for weight, value in zip(weights, values))
    effective_scenarios = 1.0 / sum(weight**2 for weight in weights)
    passes = bool(stressed_mean > 0.0)
    return {
        "method": "minimum_relative_entropy_downside_view",
        "status": _status(True, passes),
        "available": True,
        "passes": passes,
        "observation_count": len(values),
        "tail_cutoff_bps": round(cutoff, 8),
        "base_tail_probability": round(base_tail, 8),
        "target_tail_probability": round(target, 8),
        "base_mean_return_bps": round(_mean(values), 8),
        "entropy_pooled_mean_return_bps": round(stressed_mean, 8),
        "relative_entropy": round(relative_entropy, 8),
        "effective_scenario_count": round(effective_scenarios, 8),
        "view": "increase_probability_mass_on_empirical_downside_tail",
    }


def _solve_linear_system(matrix: list[list[float]], vector: list[float]) -> list[float]:
    size = len(vector)
    augmented = [list(matrix[row]) + [vector[row]] for row in range(size)]
    for column in range(size):
        pivot = max(range(column, size), key=lambda row: abs(augmented[row][column]))
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        if abs(augmented[column][column]) < 1e-12:
            augmented[column][column] = 1e-12
        divisor = augmented[column][column]
        augmented[column] = [value / divisor for value in augmented[column]]
        for row in range(size):
            if row == column:
                continue
            factor = augmented[row][column]
            augmented[row] = [
                left - factor * right
                for left, right in zip(augmented[row], augmented[column])
            ]
    return [augmented[row][-1] for row in range(size)]


def _quadratic_fit(states: Sequence[float], targets: Sequence[float]) -> list[float]:
    basis = [[1.0, state, state * state] for state in states]
    matrix = [
        [sum(row[left] * row[right] for row in basis) for right in range(3)]
        for left in range(3)
    ]
    for index in range(3):
        matrix[index][index] += 1e-9
    vector = [
        sum(row[index] * target for row, target in zip(basis, targets))
        for index in range(3)
    ]
    return _solve_linear_system(matrix, vector)


def _poly(coefficients: Sequence[float], value: float) -> float:
    return coefficients[0] + coefficients[1] * value + coefficients[2] * value * value


def least_squares_optimal_stopping(
    returns_bps: Sequence[Any],
    *,
    horizon: int = 5,
    minimum_paths: int = 12,
    training_fraction: float = 0.70,
) -> dict[str, Any]:
    values = _finite_values(returns_bps)
    path_horizon = max(int(horizon), 2)
    paths = [
        values[start : start + path_horizon]
        for start in range(0, len(values) - path_horizon + 1, path_horizon)
    ]
    available = len(paths) >= max(int(minimum_paths), 4)
    if not available:
        return {
            "method": "least_squares_monte_carlo_optimal_stopping",
            "status": "insufficient_evidence",
            "available": False,
            "passes": False,
            "observation_count": len(values),
            "independent_path_count": len(paths),
            "minimum_paths": max(int(minimum_paths), 4),
            "horizon": path_horizon,
        }
    split = min(
        max(int(len(paths) * min(max(training_fraction, 0.5), 0.85)), 3),
        len(paths) - 2,
    )
    training = paths[:split]
    holdout = paths[split:]
    cumulative = [
        [sum(path[: index + 1]) for index in range(path_horizon)]
        for path in training
    ]
    chosen_payoff = [path[-1] for path in cumulative]
    policies: dict[int, list[float]] = {}
    for index in range(path_horizon - 2, 0, -1):
        states = [path[index] for path in cumulative]
        coefficients = _quadratic_fit(states, chosen_payoff)
        policies[index] = coefficients
        chosen_payoff = [
            state if state >= _poly(coefficients, state) else future
            for state, future in zip(states, chosen_payoff)
        ]

    challenger_payoffs: list[float] = []
    baseline_payoffs: list[float] = []
    stopping_steps: list[int] = []
    for path in holdout:
        cumulative_path = [sum(path[: index + 1]) for index in range(path_horizon)]
        payoff = cumulative_path[-1]
        stop = path_horizon
        for index in range(1, path_horizon - 1):
            coefficients = policies.get(index)
            if coefficients and cumulative_path[index] >= _poly(coefficients, cumulative_path[index]):
                payoff = cumulative_path[index]
                stop = index + 1
                break
        challenger_payoffs.append(payoff)
        baseline_payoffs.append(cumulative_path[-1])
        stopping_steps.append(stop)
    improvement = _mean(challenger_payoffs) - _mean(baseline_payoffs)
    passes = bool(improvement > 0.0 and _mean(challenger_payoffs) > 0.0)
    return {
        "method": "least_squares_monte_carlo_optimal_stopping",
        "status": _status(True, passes),
        "available": True,
        "passes": passes,
        "observation_count": len(values),
        "independent_path_count": len(paths),
        "training_path_count": len(training),
        "holdout_path_count": len(holdout),
        "horizon": path_horizon,
        "holdout_challenger_mean_bps": round(_mean(challenger_payoffs), 8),
        "holdout_fixed_horizon_mean_bps": round(_mean(baseline_payoffs), 8),
        "holdout_improvement_bps": round(improvement, 8),
        "mean_stopping_step": round(_mean(stopping_steps), 6),
        "overlapping_paths_used": False,
        "authority": "diagnostic_only_no_entry_or_exit_authority",
    }


def triple_barrier_events(
    returns_bps: Sequence[Any],
    *,
    upper_barrier_bps: float = 25.0,
    lower_barrier_bps: float = 25.0,
    horizon: int = 5,
) -> list[dict[str, Any]]:
    values = _finite_values(returns_bps)
    width = max(int(horizon), 1)
    upper = abs(float(upper_barrier_bps))
    lower = abs(float(lower_barrier_bps))
    events: list[dict[str, Any]] = []
    for start in range(len(values)):
        cumulative = 0.0
        label = 0
        end = min(start + width - 1, len(values) - 1)
        for index in range(start, min(start + width, len(values))):
            cumulative += values[index]
            end = index
            if cumulative >= upper:
                label = 1
                break
            if cumulative <= -lower:
                label = -1
                break
        events.append(
            {
                "start_index": start,
                "end_index": end,
                "label": label,
                "realized_path_return_bps": round(cumulative, 8),
            }
        )
    return events


def combinatorial_purged_splits(
    events: Sequence[Mapping[str, Any]],
    *,
    group_count: int = 6,
    test_group_count: int = 2,
    embargo_observations: int = 1,
) -> dict[str, Any]:
    rows = [dict(event) for event in events]
    n = len(rows)
    groups_n = min(max(int(group_count), 2), max(n, 2))
    test_n = min(max(int(test_group_count), 1), groups_n - 1)
    groups: list[list[int]] = [[] for _ in range(groups_n)]
    for index in range(n):
        groups[min(index * groups_n // max(n, 1), groups_n - 1)].append(index)
    summaries: list[dict[str, Any]] = []
    total_violations = 0
    for test_groups in itertools.combinations(range(groups_n), test_n):
        test_indices = sorted(index for group in test_groups for index in groups[group])
        test_spans = [
            (
                int(rows[index].get("start_index", index)),
                int(rows[index].get("end_index", index)),
            )
            for index in test_indices
        ]
        train_indices: list[int] = []
        purged = 0
        embargoed = 0
        for index, row in enumerate(rows):
            if index in test_indices:
                continue
            start = int(row.get("start_index", index))
            end = int(row.get("end_index", index))
            overlaps = any(start <= test_end and end >= test_start for test_start, test_end in test_spans)
            in_embargo = any(
                test_end < start <= test_end + max(int(embargo_observations), 0)
                for _test_start, test_end in test_spans
            )
            if overlaps:
                purged += 1
            elif in_embargo:
                embargoed += 1
            else:
                train_indices.append(index)
        violations = sum(
            1
            for index in train_indices
            if any(
                int(rows[index].get("start_index", index)) <= test_end
                and int(rows[index].get("end_index", index)) >= test_start
                for test_start, test_end in test_spans
            )
        )
        total_violations += violations
        summaries.append(
            {
                "test_groups": list(test_groups),
                "train_count": len(train_indices),
                "test_count": len(test_indices),
                "purged_count": purged,
                "embargoed_count": embargoed,
                "leakage_violation_count": violations,
            }
        )
    return {
        "split_count": len(summaries),
        "group_count": groups_n,
        "test_group_count": test_n,
        "embargo_observations": max(int(embargo_observations), 0),
        "minimum_train_count": min((row["train_count"] for row in summaries), default=0),
        "minimum_test_count": min((row["test_count"] for row in summaries), default=0),
        "leakage_violation_count": total_violations,
        "splits": summaries,
    }


def cpcv_triple_barrier_diagnostic(
    returns_bps: Sequence[Any],
    *,
    minimum_observations: int = 30,
    upper_barrier_bps: float = 25.0,
    lower_barrier_bps: float = 25.0,
    horizon: int = 5,
    group_count: int = 6,
    test_group_count: int = 2,
    embargo_observations: int = 1,
) -> dict[str, Any]:
    values = _finite_values(returns_bps)
    available = len(values) >= max(int(minimum_observations), group_count * 2)
    if not available:
        return {
            "method": "triple_barrier_and_combinatorial_purged_cross_validation",
            "status": "insufficient_evidence",
            "available": False,
            "passes": False,
            "observation_count": len(values),
            "minimum_observations": max(int(minimum_observations), group_count * 2),
        }
    events = triple_barrier_events(
        values,
        upper_barrier_bps=upper_barrier_bps,
        lower_barrier_bps=lower_barrier_bps,
        horizon=horizon,
    )
    cpcv = combinatorial_purged_splits(
        events,
        group_count=group_count,
        test_group_count=test_group_count,
        embargo_observations=embargo_observations,
    )
    labels = [int(row["label"]) for row in events]
    counts = {str(label): labels.count(label) for label in (-1, 0, 1)}
    passes = bool(
        cpcv["split_count"] > 0
        and cpcv["minimum_train_count"] > 0
        and cpcv["minimum_test_count"] > 0
        and cpcv["leakage_violation_count"] == 0
    )
    return {
        "method": "triple_barrier_and_combinatorial_purged_cross_validation",
        "status": _status(True, passes),
        "available": True,
        "passes": passes,
        "observation_count": len(values),
        "event_count": len(events),
        "label_counts": counts,
        "upper_barrier_bps": abs(float(upper_barrier_bps)),
        "lower_barrier_bps": abs(float(lower_barrier_bps)),
        "horizon": max(int(horizon), 1),
        "cpcv": cpcv,
        "label_scope": "candidate_forward_profile_post_cost_path_diagnostic",
        "training_authority": False,
    }


def _capped_simplex(weights: Sequence[float], cap: float) -> list[float]:
    count = len(weights)
    if count == 0:
        return []
    ceiling = max(min(float(cap), 1.0), 1.0 / count)
    normalized = [max(float(value), 0.0) for value in weights]
    total = sum(normalized)
    normalized = [value / total for value in normalized] if total > 0.0 else [1.0 / count] * count
    fixed = [False] * count
    for _ in range(count + 1):
        over = [index for index, value in enumerate(normalized) if not fixed[index] and value > ceiling]
        if not over:
            break
        for index in over:
            normalized[index] = ceiling
            fixed[index] = True
        remaining = 1.0 - sum(normalized[index] for index in range(count) if fixed[index])
        free = [index for index in range(count) if not fixed[index]]
        if not free:
            break
        free_total = sum(normalized[index] for index in free)
        for index in free:
            normalized[index] = remaining * (
                normalized[index] / free_total if free_total > 0.0 else 1.0 / len(free)
            )
    total = sum(normalized)
    return [value / total for value in normalized]


def cost_aware_expert_aggregation(
    profile_returns_bps: Mapping[str, Sequence[Any]],
    *,
    minimum_periods: int = 12,
    learning_rate: float = 1.0,
    return_scale_bps: float = 100.0,
    transaction_cost_bps: float = 2.0,
    maximum_weight: float = 0.35,
) -> dict[str, Any]:
    series = {
        str(profile): _finite_values(values)
        for profile, values in sorted(profile_returns_bps.items())
        if str(profile).strip()
    }
    series = {profile: values for profile, values in series.items() if values}
    period_count = min((len(values) for values in series.values()), default=0)
    available = bool(len(series) >= 2 and period_count >= max(int(minimum_periods), 4))
    if not available:
        return {
            "method": "transaction_cost_aware_online_expert_aggregation",
            "status": "insufficient_evidence",
            "available": False,
            "passes": False,
            "profile_count": len(series),
            "period_count": period_count,
            "minimum_periods": max(int(minimum_periods), 4),
        }
    names = sorted(series)
    aligned = {name: series[name][-period_count:] for name in names}
    weights = [1.0 / len(names)] * len(names)
    net_total = 0.0
    equal_total = 0.0
    turnover_total = 0.0
    max_observed_weight = max(weights)
    scale = max(float(return_scale_bps), 1e-6)
    for period in range(period_count):
        returns = [aligned[name][period] for name in names]
        gross = sum(weight * value for weight, value in zip(weights, returns))
        equal_total += _mean(returns)
        raw_next = [
            weight * math.exp(float(learning_rate) * min(max(value / scale, -5.0), 5.0))
            for weight, value in zip(weights, returns)
        ]
        next_weights = _capped_simplex(raw_next, maximum_weight)
        turnover = 0.5 * sum(abs(new - old) for new, old in zip(next_weights, weights))
        net_total += gross - float(transaction_cost_bps) * turnover
        turnover_total += turnover
        max_observed_weight = max(max_observed_weight, max(next_weights))
        weights = next_weights
    best_fixed = max(sum(aligned[name]) for name in names)
    improvement = net_total - equal_total
    passes = bool(improvement > 0.0 and net_total > 0.0)
    return {
        "method": "transaction_cost_aware_online_expert_aggregation",
        "status": _status(True, passes),
        "available": True,
        "passes": passes,
        "profile_count": len(names),
        "period_count": period_count,
        "minimum_periods": max(int(minimum_periods), 4),
        "learning_rate": float(learning_rate),
        "transaction_cost_bps": float(transaction_cost_bps),
        "maximum_weight": max(float(maximum_weight), 1.0 / len(names)),
        "observed_max_weight": round(max_observed_weight, 8),
        "cumulative_turnover": round(turnover_total, 8),
        "challenger_net_return_bps": round(net_total, 8),
        "equal_weight_return_bps": round(equal_total, 8),
        "improvement_vs_equal_weight_bps": round(improvement, 8),
        "regret_vs_best_fixed_expert_bps": round(best_fixed - net_total, 8),
        "final_weights": {
            name: round(weight, 8) for name, weight in zip(names, weights)
        },
        "authority": "paper_counterfactual_only_no_allocator_authority",
    }
