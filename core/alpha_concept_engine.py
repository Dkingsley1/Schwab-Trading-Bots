"""Deterministic, candidate-safe alpha measurement diagnostics.

The functions in this module are research measurements only. They never create
signals, alter sizes or weights, write labels, promote candidates, or submit an
order. Every estimator fails closed when its declared inputs are incomplete.
"""

from __future__ import annotations

import math
import random
import re
import statistics
from collections import defaultdict
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from typing import Any

import numpy as np

NO_AUTHORITY = {
    "changes_active_action": False,
    "changes_position_size": False,
    "changes_allocator_weights": False,
    "changes_training_labels": False,
    "submits_paper_orders": False,
    "submits_live_orders": False,
    "grants_promotion": False,
    "guarantees_profitability": False,
}


def _number(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _finite(values: Sequence[Any]) -> list[float]:
    return [parsed for value in values if (parsed := _number(value)) is not None]


def _normal_cdf(value: float) -> float:
    return 0.5 * (1.0 + math.erf(value / math.sqrt(2.0)))


def _status(available: bool, supported: bool) -> str:
    if not available:
        return "insufficient_evidence"
    return "supported" if supported else "not_supported"


def _authority() -> dict[str, bool]:
    return dict(NO_AUTHORITY)


def _rank(values: Sequence[float]) -> list[float]:
    ordered = sorted(enumerate(values), key=lambda row: (row[1], row[0]))
    ranks = [0.0] * len(values)
    index = 0
    while index < len(ordered):
        end = index + 1
        while end < len(ordered) and ordered[end][1] == ordered[index][1]:
            end += 1
        average_rank = (index + 1 + end) / 2.0
        for original, _value in ordered[index:end]:
            ranks[original] = average_rank
        index = end
    return ranks


def _correlation(left: Sequence[float], right: Sequence[float]) -> float | None:
    if len(left) != len(right) or len(left) < 2:
        return None
    left_mean = statistics.fmean(left)
    right_mean = statistics.fmean(right)
    numerator = sum((a - left_mean) * (b - right_mean) for a, b in zip(left, right))
    left_ss = sum((value - left_mean) ** 2 for value in left)
    right_ss = sum((value - right_mean) ** 2 for value in right)
    denominator = math.sqrt(left_ss * right_ss)
    return numerator / denominator if denominator > 0.0 else None


def _rank_correlation(left: Sequence[float], right: Sequence[float]) -> float | None:
    return _correlation(_rank(left), _rank(right))


def _correlation_lcb(value: float | None, count: int) -> float | None:
    if value is None or count < 4:
        return None
    standard_error = math.sqrt(max(1.0 - value * value, 0.0) / (count - 2))
    return max(value - 1.96 * standard_error, -1.0)


def _mean_lcb(values: Sequence[float]) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    mean = statistics.fmean(values)
    if len(values) < 2:
        return mean, None
    return mean, mean - 1.96 * statistics.stdev(values) / math.sqrt(len(values))


def _parse_horizon_step(value: Any, fallback: int) -> float:
    text = str(value or "")
    match = re.search(r"([0-9]+(?:\.[0-9]+)?)", text)
    return float(match.group(1)) if match else float(fallback)


def information_coefficient_term_structure(
    observations: Sequence[Mapping[str, Any]],
    *,
    minimum_observations_per_horizon: int = 20,
    minimum_periods_for_icir: int = 5,
) -> dict[str, Any]:
    """Measure Pearson/rank IC, period stability, regime IC, and horizon decay."""

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    rejected = 0
    for raw in observations:
        forecast = _number(raw.get("forecast"))
        realized = _number(raw.get("realized"))
        horizon = str(raw.get("horizon") or "").strip()
        if forecast is None or realized is None or not horizon:
            rejected += 1
            continue
        grouped[horizon].append(
            {
                "forecast": forecast,
                "realized": realized,
                "period": str(raw.get("period") or "").strip(),
                "regime": str(raw.get("regime") or "unspecified").strip(),
            }
        )

    horizon_rows: list[dict[str, Any]] = []
    floor = max(int(minimum_observations_per_horizon), 4)
    period_floor = max(int(minimum_periods_for_icir), 2)
    for order, (horizon, rows) in enumerate(sorted(grouped.items()), start=1):
        forecasts = [float(row["forecast"]) for row in rows]
        realized = [float(row["realized"]) for row in rows]
        pearson = _correlation(forecasts, realized)
        rank_ic = _rank_correlation(forecasts, realized)
        by_period: dict[str, list[dict[str, Any]]] = defaultdict(list)
        by_regime: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for index, row in enumerate(rows):
            period = str(row["period"] or f"row_{index:08d}")
            by_period[period].append(row)
            by_regime[str(row["regime"])].append(row)
        period_ics = [
            value
            for period_rows in by_period.values()
            if len(period_rows) >= 3
            and (
                value := _rank_correlation(
                    [float(row["forecast"]) for row in period_rows],
                    [float(row["realized"]) for row in period_rows],
                )
            )
            is not None
        ]
        period_ic_mean = statistics.fmean(period_ics) if period_ics else None
        period_ic_std = statistics.stdev(period_ics) if len(period_ics) >= 2 else None
        icir = (
            period_ic_mean / period_ic_std
            if period_ic_mean is not None and period_ic_std not in {None, 0.0}
            else None
        )
        regimes: dict[str, Any] = {}
        for regime, regime_rows in sorted(by_regime.items()):
            value = _rank_correlation(
                [float(row["forecast"]) for row in regime_rows],
                [float(row["realized"]) for row in regime_rows],
            )
            regimes[regime] = {
                "observation_count": len(regime_rows),
                "rank_ic": round(value, 8) if value is not None else None,
            }
        available = len(rows) >= floor and rank_ic is not None
        lcb = _correlation_lcb(rank_ic, len(rows))
        supported = bool(available and lcb is not None and lcb > 0.0)
        horizon_rows.append(
            {
                "horizon": horizon,
                "horizon_step": _parse_horizon_step(horizon, order),
                "status": _status(available, supported),
                "available": available,
                "passes": supported,
                "observation_count": len(rows),
                "pearson_ic": round(pearson, 8) if pearson is not None else None,
                "rank_ic": round(rank_ic, 8) if rank_ic is not None else None,
                "rank_ic_lcb_95": round(lcb, 8) if lcb is not None else None,
                "period_ic_count": len(period_ics),
                "period_ic_mean": (
                    round(period_ic_mean, 8) if period_ic_mean is not None else None
                ),
                "period_ic_std": (
                    round(period_ic_std, 8) if period_ic_std is not None else None
                ),
                "ic_information_ratio": round(icir, 8) if icir is not None else None,
                "icir_evidence_ready": len(period_ics) >= period_floor,
                "regime_breakdown": regimes,
            }
        )

    decay_points = [
        (float(row["horizon_step"]), abs(float(row["rank_ic"])))
        for row in horizon_rows
        if row.get("rank_ic") not in {None, 0.0}
    ]
    half_life: float | None = None
    decay_slope: float | None = None
    if len(decay_points) >= 2:
        x = np.asarray([row[0] for row in decay_points], dtype=float)
        y = np.log(
            np.asarray([max(row[1], 1e-12) for row in decay_points], dtype=float)
        )
        decay_slope = float(np.polyfit(x, y, 1)[0])
        if decay_slope < 0.0:
            half_life = math.log(2.0) / -decay_slope
    available_count = sum(bool(row["available"]) for row in horizon_rows)
    supported_count = sum(bool(row["passes"]) for row in horizon_rows)
    return {
        "method": "pearson_rank_ic_term_structure_and_period_icir",
        "status": _status(available_count > 0, supported_count > 0),
        "available": available_count > 0,
        "passes": supported_count > 0,
        "horizon_count": len(horizon_rows),
        "available_horizon_count": available_count,
        "supported_horizon_count": supported_count,
        "minimum_observations_per_horizon": floor,
        "minimum_periods_for_icir": period_floor,
        "rejected_observation_count": rejected,
        "estimated_ic_decay_log_slope": (
            round(decay_slope, 8) if decay_slope is not None else None
        ),
        "estimated_ic_half_life_horizon_units": (
            round(half_life, 8) if half_life is not None else None
        ),
        "horizons": horizon_rows,
        "authority": _authority(),
    }


def _matrix(values: Sequence[Sequence[Any]]) -> np.ndarray | None:
    try:
        matrix = np.asarray(values, dtype=float)
    except (TypeError, ValueError):
        return None
    if matrix.ndim != 2 or matrix.size == 0:
        return None
    return matrix


def effective_breadth_transfer_coefficient(
    forecast_matrix: Sequence[Sequence[Any]],
    realized_matrix: Sequence[Sequence[Any]],
    *,
    implemented_weight_matrix: Sequence[Sequence[Any]] | None = None,
    minimum_periods: int = 12,
    minimum_bets: int = 2,
) -> dict[str, Any]:
    """Estimate independent breadth and forecast implementation efficiency."""

    forecasts = _matrix(forecast_matrix)
    realized = _matrix(realized_matrix)
    weights = _matrix(
        implemented_weight_matrix if implemented_weight_matrix is not None else []
    )
    floor_periods = max(int(minimum_periods), 4)
    floor_bets = max(int(minimum_bets), 2)
    if forecasts is None or realized is None or forecasts.shape != realized.shape:
        return {
            "method": "effective_breadth_and_transfer_coefficient",
            "status": "insufficient_evidence",
            "available": False,
            "passes": False,
            "blockers": ["aligned_forecast_and_realized_matrices_required"],
            "authority": _authority(),
        }
    valid_rows = np.isfinite(forecasts).all(axis=1) & np.isfinite(realized).all(axis=1)
    forecasts = forecasts[valid_rows]
    realized = realized[valid_rows]
    if weights is not None and weights.shape == valid_rows.shape + (
        forecasts.shape[1],
    ):
        weights = weights[valid_rows]
    elif weights is not None and weights.shape != forecasts.shape:
        weights = None
    period_count, bet_count = forecasts.shape
    available = period_count >= floor_periods and bet_count >= floor_bets
    if not available:
        return {
            "method": "effective_breadth_and_transfer_coefficient",
            "status": "insufficient_evidence",
            "available": False,
            "passes": False,
            "period_count": period_count,
            "bet_count": bet_count,
            "minimum_periods": floor_periods,
            "minimum_bets": floor_bets,
            "authority": _authority(),
        }
    correlation = np.corrcoef(realized, rowvar=False)
    correlation = np.nan_to_num(correlation, nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(correlation, 1.0)
    eigenvalues = np.clip(np.linalg.eigvalsh(correlation), 0.0, None)
    denominator = float(np.sum(eigenvalues**2))
    effective_bets = (
        float(np.sum(eigenvalues) ** 2 / denominator) if denominator > 0.0 else 1.0
    )
    period_ic = [
        value
        for row_forecast, row_realized in zip(forecasts, realized)
        if (value := _rank_correlation(row_forecast.tolist(), row_realized.tolist()))
        is not None
    ]
    lag_one = (
        _correlation(period_ic[:-1], period_ic[1:]) if len(period_ic) >= 3 else None
    )
    rho = min(max(float(lag_one or 0.0), -0.95), 0.95)
    effective_periods = min(
        max(period_count * (1.0 - rho) / (1.0 + rho), 1.0), float(period_count)
    )
    total_breadth = effective_bets * effective_periods
    transfer = None
    if (
        weights is not None
        and weights.shape == forecasts.shape
        and np.isfinite(weights).all()
    ):
        transfer = _rank_correlation(
            forecasts.ravel().tolist(), weights.ravel().tolist()
        )
    mean_ic = statistics.fmean(period_ic) if period_ic else None
    potential_ir = (
        mean_ic * math.sqrt(total_breadth) * transfer
        if mean_ic is not None and transfer is not None
        else None
    )
    supported = bool(
        mean_ic is not None
        and mean_ic > 0.0
        and transfer is not None
        and transfer > 0.0
    )
    return {
        "method": "effective_breadth_and_transfer_coefficient",
        "status": _status(True, supported),
        "available": True,
        "passes": supported,
        "period_count": period_count,
        "bet_count": bet_count,
        "period_ic_count": len(period_ic),
        "mean_rank_information_coefficient": (
            round(mean_ic, 8) if mean_ic is not None else None
        ),
        "period_ic_lag_one_correlation": (
            round(lag_one, 8) if lag_one is not None else None
        ),
        "effective_bets_per_period": round(effective_bets, 8),
        "effective_independent_periods": round(effective_periods, 8),
        "effective_total_breadth": round(total_breadth, 8),
        "transfer_coefficient": round(transfer, 8) if transfer is not None else None,
        "transfer_coefficient_available": transfer is not None,
        "generalized_fundamental_law_ir_diagnostic": (
            round(potential_ir, 8) if potential_ir is not None else None
        ),
        "breadth_basis": "correlation_eigenvalue_participation_ratio_times_lag_adjusted_periods",
        "authority": _authority(),
    }


def hierarchical_bayesian_skill(
    returns_by_group: Mapping[str, Sequence[Any]],
    *,
    minimum_groups: int = 2,
    minimum_observations_per_group: int = 8,
    posterior_probability_floor: float = 0.95,
) -> dict[str, Any]:
    """Empirical-Bayes normal-normal shrinkage across sleeves or families."""

    group_floor = max(int(minimum_groups), 2)
    observation_floor = max(int(minimum_observations_per_group), 3)
    groups = {
        str(name): values
        for name, raw in sorted(returns_by_group.items())
        if str(name).strip() and len(values := _finite(raw)) >= observation_floor
    }
    if len(groups) < group_floor:
        return {
            "method": "empirical_bayes_hierarchical_normal_skill",
            "status": "insufficient_evidence",
            "available": False,
            "passes": False,
            "eligible_group_count": len(groups),
            "minimum_groups": group_floor,
            "minimum_observations_per_group": observation_floor,
            "authority": _authority(),
        }
    means = {name: statistics.fmean(values) for name, values in groups.items()}
    sampling_variance = {
        name: max(statistics.variance(values) / len(values), 1e-12)
        for name, values in groups.items()
    }
    total_count = sum(len(values) for values in groups.values())
    grand_mean = sum(means[name] * len(groups[name]) for name in groups) / total_count
    observed_between = statistics.variance(means.values()) if len(means) >= 2 else 0.0
    between_variance = max(
        observed_between - statistics.fmean(sampling_variance.values()), 1e-12
    )
    rows: list[dict[str, Any]] = []
    for name in sorted(groups):
        sample_var = sampling_variance[name]
        posterior_var = 1.0 / (1.0 / between_variance + 1.0 / sample_var)
        posterior_mean = posterior_var * (
            grand_mean / between_variance + means[name] / sample_var
        )
        posterior_std = math.sqrt(posterior_var)
        positive_probability = _normal_cdf(posterior_mean / posterior_std)
        passes = bool(
            posterior_mean > 0.0
            and positive_probability >= float(posterior_probability_floor)
        )
        rows.append(
            {
                "group": name,
                "observation_count": len(groups[name]),
                "sample_mean": round(means[name], 8),
                "shrinkage_weight_on_group": round(
                    between_variance / (between_variance + sample_var), 8
                ),
                "posterior_mean": round(posterior_mean, 8),
                "posterior_std": round(posterior_std, 8),
                "posterior_lcb_95": round(posterior_mean - 1.96 * posterior_std, 8),
                "posterior_ucb_95": round(posterior_mean + 1.96 * posterior_std, 8),
                "posterior_positive_probability": round(positive_probability, 8),
                "passes": passes,
            }
        )
    supported_count = sum(bool(row["passes"]) for row in rows)
    return {
        "method": "empirical_bayes_hierarchical_normal_skill",
        "status": _status(True, supported_count > 0),
        "available": True,
        "passes": supported_count > 0,
        "group_count": len(rows),
        "supported_group_count": supported_count,
        "grand_mean": round(grand_mean, 8),
        "between_group_variance": round(between_variance, 12),
        "posterior_probability_floor": float(posterior_probability_floor),
        "groups": rows,
        "authority": _authority(),
    }


def subsample_stability_selection(
    features: Mapping[str, Sequence[Any]],
    outcomes: Sequence[Any],
    *,
    minimum_observations: int = 30,
    replications: int = 200,
    subsample_fraction: float = 0.5,
    selected_feature_count: int | None = None,
    selection_probability_floor: float = 0.8,
    sign_consistency_floor: float = 0.8,
    seed: int = 1777,
) -> dict[str, Any]:
    """Seeded subsample stability selection using correlation screeners."""

    names = sorted(str(name) for name in features if str(name).strip())
    lengths = (
        {len(outcomes), *(len(features[name]) for name in names)} if names else set()
    )
    if not names or len(lengths) != 1:
        return {
            "method": "subsample_correlation_stability_selection",
            "status": "insufficient_evidence",
            "available": False,
            "passes": False,
            "blockers": ["aligned_feature_and_outcome_vectors_required"],
            "authority": _authority(),
        }
    rows: list[tuple[float, list[float]]] = []
    for index in range(len(outcomes)):
        outcome = _number(outcomes[index])
        values = [_number(features[name][index]) for name in names]
        if outcome is not None and all(value is not None for value in values):
            rows.append(
                (outcome, [float(value) for value in values if value is not None])
            )
    floor = max(int(minimum_observations), 8)
    if len(rows) < floor:
        return {
            "method": "subsample_correlation_stability_selection",
            "status": "insufficient_evidence",
            "available": False,
            "passes": False,
            "observation_count": len(rows),
            "minimum_observations": floor,
            "authority": _authority(),
        }
    y = [row[0] for row in rows]
    x = [[row[1][column] for row in rows] for column in range(len(names))]
    full_correlations = [_correlation(values, y) or 0.0 for values in x]
    select_count = selected_feature_count or max(1, int(math.sqrt(len(names))))
    select_count = min(max(int(select_count), 1), len(names))
    reps = max(int(replications), 20)
    sample_size = min(
        max(int(round(len(rows) * min(max(subsample_fraction, 0.2), 0.8))), 4),
        len(rows) - 1,
    )
    selected = [0] * len(names)
    sign_match = [0] * len(names)
    rng = random.Random(int(seed))
    for _ in range(reps):
        indices = rng.sample(range(len(rows)), sample_size)
        correlations = [
            _correlation(
                [values[index] for index in indices], [y[index] for index in indices]
            )
            or 0.0
            for values in x
        ]
        chosen = sorted(
            range(len(names)),
            key=lambda index: (abs(correlations[index]), names[index]),
            reverse=True,
        )[:select_count]
        for index in chosen:
            selected[index] += 1
            if correlations[index] == 0.0 or full_correlations[index] == 0.0:
                continue
            sign_match[index] += int(
                math.copysign(1.0, correlations[index])
                == math.copysign(1.0, full_correlations[index])
            )
    feature_rows: list[dict[str, Any]] = []
    for index, name in enumerate(names):
        probability = selected[index] / reps
        consistency = sign_match[index] / selected[index] if selected[index] else 0.0
        stable = bool(
            probability >= float(selection_probability_floor)
            and consistency >= float(sign_consistency_floor)
            and full_correlations[index] != 0.0
        )
        feature_rows.append(
            {
                "feature": name,
                "full_sample_correlation": round(full_correlations[index], 8),
                "selection_probability": round(probability, 8),
                "sign_consistency": round(consistency, 8),
                "stable": stable,
            }
        )
    feature_rows.sort(
        key=lambda row: (float(row["selection_probability"]), str(row["feature"])),
        reverse=True,
    )
    stable = [str(row["feature"]) for row in feature_rows if row["stable"]]
    threshold = float(selection_probability_floor)
    false_selection_bound = (
        select_count**2 / ((2.0 * threshold - 1.0) * len(names))
        if threshold > 0.5
        else None
    )
    return {
        "method": "subsample_correlation_stability_selection",
        "status": _status(True, bool(stable)),
        "available": True,
        "passes": bool(stable),
        "observation_count": len(rows),
        "feature_count": len(names),
        "selected_feature_count_per_replication": select_count,
        "replications": reps,
        "subsample_size": sample_size,
        "selection_probability_floor": threshold,
        "sign_consistency_floor": float(sign_consistency_floor),
        "stable_features": stable,
        "stable_feature_count": len(stable),
        "expected_false_selection_upper_bound_under_exchangeability": (
            round(false_selection_bound, 8)
            if false_selection_bound is not None
            else None
        ),
        "features": feature_rows,
        "model_x_knockoff_guarantee_claimed": False,
        "authority": _authority(),
    }


def _aligned_regression_inputs(
    target: Sequence[Any], factors: Mapping[str, Sequence[Any]]
) -> tuple[list[str], np.ndarray, np.ndarray, list[int]]:
    names = sorted(str(name) for name in factors if str(name).strip())
    if not names or any(len(factors[name]) != len(target) for name in names):
        return names, np.empty(0), np.empty((0, 0)), []
    indices: list[int] = []
    y_rows: list[float] = []
    x_rows: list[list[float]] = []
    for index, raw_target in enumerate(target):
        target_value = _number(raw_target)
        factor_values = [_number(factors[name][index]) for name in names]
        if target_value is None or any(value is None for value in factor_values):
            continue
        indices.append(index)
        y_rows.append(target_value)
        x_rows.append([float(value) for value in factor_values if value is not None])
    return (
        names,
        np.asarray(y_rows, dtype=float),
        np.asarray(x_rows, dtype=float),
        indices,
    )


def _ridge_coefficients(x: np.ndarray, y: np.ndarray, penalty: float) -> np.ndarray:
    design = np.column_stack([np.ones(len(x)), x])
    regularizer = np.eye(design.shape[1]) * max(float(penalty), 0.0)
    regularizer[0, 0] = 0.0
    return np.linalg.pinv(design.T @ design + regularizer) @ design.T @ y


def factor_neutral_residualization(
    target_returns: Sequence[Any],
    factors: Mapping[str, Sequence[Any]],
    *,
    minimum_observations: int = 20,
    ridge_penalty: float = 1e-8,
) -> dict[str, Any]:
    """Residualize returns against an explicit factor matrix."""

    names, y, x, _indices = _aligned_regression_inputs(target_returns, factors)
    floor = max(int(minimum_observations), len(names) + 3)
    if len(y) < floor or not names:
        return {
            "method": "ridge_multifactor_neutral_residualization",
            "status": "insufficient_evidence",
            "available": False,
            "passes": False,
            "observation_count": int(len(y)),
            "factor_count": len(names),
            "minimum_observations": floor,
            "authority": _authority(),
        }
    coefficients = _ridge_coefficients(x, y, ridge_penalty)
    fitted = np.column_stack([np.ones(len(x)), x]) @ coefficients
    residuals = y - fitted
    target_ss = float(np.sum((y - np.mean(y)) ** 2))
    residual_ss = float(np.sum(residuals**2))
    r_squared = 1.0 - residual_ss / target_ss if target_ss > 0.0 else 0.0
    mean_residual, residual_lcb = _mean_lcb(residuals.tolist())
    design = np.column_stack([np.ones(len(x)), x])
    residual_dof = max(len(y) - design.shape[1], 1)
    residual_variance = residual_ss / residual_dof
    intercept_standard_error = math.sqrt(
        max(
            residual_variance * float(np.linalg.pinv(design.T @ design)[0, 0]),
            0.0,
        )
    )
    intercept_lcb = float(coefficients[0]) - 1.96 * intercept_standard_error
    residual_exposures = {
        name: _correlation(residuals.tolist(), x[:, index].tolist())
        for index, name in enumerate(names)
    }
    supported = bool(intercept_lcb > 0.0)
    return {
        "method": "ridge_multifactor_neutral_residualization",
        "status": _status(True, supported),
        "available": True,
        "passes": supported,
        "observation_count": int(len(y)),
        "factor_count": len(names),
        "intercept_alpha": round(float(coefficients[0]), 8),
        "intercept_alpha_standard_error": round(intercept_standard_error, 8),
        "intercept_alpha_lcb_95": round(intercept_lcb, 8),
        "factor_loadings": {
            name: round(float(coefficients[index + 1]), 8)
            for index, name in enumerate(names)
        },
        "r_squared": round(r_squared, 8),
        "mean_residual_alpha": (
            round(mean_residual, 8) if mean_residual is not None else None
        ),
        "residual_alpha_lcb_95": (
            round(residual_lcb, 8) if residual_lcb is not None else None
        ),
        "residual_factor_correlations": {
            name: round(value, 8) if value is not None else None
            for name, value in residual_exposures.items()
        },
        "residual_returns": [round(float(value), 8) for value in residuals],
        "ridge_penalty": float(ridge_penalty),
        "authority": _authority(),
    }


def economic_alpha_decomposition(
    gross_active_returns: Sequence[Any],
    factors: Mapping[str, Sequence[Any]],
    execution_costs: Sequence[Any],
    *,
    dynamic_factor_exposures: Mapping[str, Sequence[Any]] | None = None,
    minimum_observations: int = 20,
    ridge_penalty: float = 1e-8,
) -> dict[str, Any]:
    """Split gross active return into factor, unexplained, and execution terms."""

    names, y_all, x_all, indices = _aligned_regression_inputs(
        gross_active_returns, factors
    )
    valid_positions = [
        position
        for position, original in enumerate(indices)
        if original < len(execution_costs)
        and _number(execution_costs[original]) is not None
    ]
    y = y_all[valid_positions] if valid_positions else np.empty(0)
    x = x_all[valid_positions] if valid_positions else np.empty((0, len(names)))
    original_indices = [indices[position] for position in valid_positions]
    costs = np.asarray(
        [float(_number(execution_costs[index]) or 0.0) for index in original_indices],
        dtype=float,
    )
    floor = max(int(minimum_observations), len(names) + 3)
    if len(y) < floor or not names:
        return {
            "method": "economic_factor_selection_timing_execution_decomposition",
            "status": "insufficient_evidence",
            "available": False,
            "passes": False,
            "observation_count": int(len(y)),
            "factor_count": len(names),
            "minimum_observations": floor,
            "authority": _authority(),
        }
    coefficients = _ridge_coefficients(x, y, ridge_penalty)
    factor_contribution = float(np.mean(x @ coefficients[1:]))
    selection_alpha = float(coefficients[0])
    gross_mean = float(np.mean(y))
    execution_drag = -float(np.mean(costs))
    net_mean = gross_mean + execution_drag
    fitted = np.column_stack([np.ones(len(x)), x]) @ coefficients
    residuals = y - fitted
    residual_std = float(np.std(residuals, ddof=max(1, len(names) + 1)))
    selection_se = residual_std / math.sqrt(len(y))
    selection_lcb = selection_alpha - 1.96 * selection_se
    timing_available = False
    timing_alpha: float | None = None
    exposures = dynamic_factor_exposures or {}
    if (
        names
        and all(name in exposures for name in names)
        and all(len(exposures[name]) == len(gross_active_returns) for name in names)
    ):
        dynamic_rows: list[list[float]] = []
        dynamic_valid = True
        for original in original_indices:
            row = [_number(exposures[name][original]) for name in names]
            if any(value is None for value in row):
                dynamic_valid = False
                break
            dynamic_rows.append([float(value) for value in row if value is not None])
        if dynamic_valid and dynamic_rows:
            dynamic = np.asarray(dynamic_rows, dtype=float)
            centered = dynamic - np.mean(dynamic, axis=0)
            timing_alpha = float(np.mean(np.sum(centered * x, axis=1)))
            timing_available = True
    supported = bool(net_mean > 0.0 and selection_lcb > 0.0)
    return {
        "method": "economic_factor_selection_timing_execution_decomposition",
        "status": _status(True, supported),
        "available": True,
        "passes": supported,
        "observation_count": int(len(y)),
        "factor_count": len(names),
        "gross_active_return_mean": round(gross_mean, 8),
        "static_factor_premia_contribution": round(factor_contribution, 8),
        "unexplained_selection_alpha": round(selection_alpha, 8),
        "unexplained_selection_alpha_lcb_95": round(selection_lcb, 8),
        "execution_cost_drag": round(execution_drag, 8),
        "net_active_return_mean": round(net_mean, 8),
        "timing_alpha_diagnostic": (
            round(timing_alpha, 8) if timing_alpha is not None else None
        ),
        "timing_alpha_available": timing_available,
        "factor_loadings": {
            name: round(float(coefficients[index + 1]), 8)
            for index, name in enumerate(names)
        },
        "additive_identity": "gross_mean=static_factor_premia+unexplained_selection_alpha; net_mean=gross_mean-execution_cost",
        "timing_is_supplementary_not_double_counted": True,
        "authority": _authority(),
    }


def _ridge_predict(
    train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray, penalty: float
) -> np.ndarray:
    coefficients = _ridge_coefficients(train_x, train_y, penalty)
    return np.column_stack([np.ones(len(test_x)), test_x]) @ coefficients


def cross_fitted_causal_transportability(
    outcomes: Sequence[Any],
    treatments: Sequence[Any],
    controls: Mapping[str, Sequence[Any]],
    environments: Sequence[Any],
    *,
    minimum_observations: int = 40,
    minimum_environments: int = 2,
    ridge_penalty: float = 0.001,
    maximum_relative_heterogeneity: float = 1.0,
) -> dict[str, Any]:
    """Leave-one-environment-out partially linear DML diagnostic."""

    names = sorted(str(name) for name in controls if str(name).strip())
    lengths = {len(outcomes), len(treatments), len(environments)}
    lengths.update(len(controls[name]) for name in names)
    if not names or len(lengths) != 1:
        return {
            "method": "leave_one_environment_out_partially_linear_dml",
            "status": "insufficient_evidence",
            "available": False,
            "passes": False,
            "blockers": ["aligned_outcome_treatment_control_environment_rows_required"],
            "authority": _authority(),
        }
    rows: list[tuple[float, float, list[float], str]] = []
    for index in range(len(outcomes)):
        outcome = _number(outcomes[index])
        treatment = _number(treatments[index])
        values = [_number(controls[name][index]) for name in names]
        environment = str(environments[index] or "").strip()
        if (
            outcome is None
            or treatment is None
            or any(value is None for value in values)
            or not environment
        ):
            continue
        rows.append(
            (
                outcome,
                treatment,
                [float(value) for value in values if value is not None],
                environment,
            )
        )
    environment_names = sorted({row[3] for row in rows})
    floor = max(int(minimum_observations), len(names) + 8)
    environment_floor = max(int(minimum_environments), 2)
    if len(rows) < floor or len(environment_names) < environment_floor:
        return {
            "method": "leave_one_environment_out_partially_linear_dml",
            "status": "insufficient_evidence",
            "available": False,
            "passes": False,
            "observation_count": len(rows),
            "environment_count": len(environment_names),
            "minimum_observations": floor,
            "minimum_environments": environment_floor,
            "authority": _authority(),
        }
    y = np.asarray([row[0] for row in rows], dtype=float)
    d = np.asarray([row[1] for row in rows], dtype=float)
    x = np.asarray([row[2] for row in rows], dtype=float)
    env = [row[3] for row in rows]
    y_residual = np.zeros(len(rows), dtype=float)
    d_residual = np.zeros(len(rows), dtype=float)
    environment_effects: list[dict[str, Any]] = []
    for environment in environment_names:
        test = np.asarray([name == environment for name in env], dtype=bool)
        train = ~test
        if int(np.sum(test)) < 3 or int(np.sum(train)) <= len(names) + 2:
            return {
                "method": "leave_one_environment_out_partially_linear_dml",
                "status": "insufficient_evidence",
                "available": False,
                "passes": False,
                "blockers": [f"environment_fold_too_thin:{environment}"],
                "authority": _authority(),
            }
        y_residual[test] = y[test] - _ridge_predict(
            x[train], y[train], x[test], ridge_penalty
        )
        d_residual[test] = d[test] - _ridge_predict(
            x[train], d[train], x[test], ridge_penalty
        )
    denominator = float(np.sum(d_residual**2))
    if denominator <= 1e-12:
        return {
            "method": "leave_one_environment_out_partially_linear_dml",
            "status": "insufficient_evidence",
            "available": False,
            "passes": False,
            "blockers": ["residualized_treatment_has_no_variation"],
            "authority": _authority(),
        }
    effect = float(np.sum(d_residual * y_residual) / denominator)
    errors = y_residual - effect * d_residual
    standard_error = math.sqrt(float(np.sum((d_residual * errors) ** 2))) / denominator
    for environment in environment_names:
        mask = np.asarray([name == environment for name in env], dtype=bool)
        env_denominator = float(np.sum(d_residual[mask] ** 2))
        env_effect = (
            float(np.sum(d_residual[mask] * y_residual[mask]) / env_denominator)
            if env_denominator > 1e-12
            else None
        )
        environment_effects.append(
            {
                "environment": environment,
                "observation_count": int(np.sum(mask)),
                "effect": round(env_effect, 8) if env_effect is not None else None,
            }
        )
    finite_effects = [
        float(row["effect"]) for row in environment_effects if row["effect"] is not None
    ]
    sign_agreement = (
        sum(
            math.copysign(1.0, value) == math.copysign(1.0, effect)
            for value in finite_effects
        )
        / len(finite_effects)
        if finite_effects and effect != 0.0
        else 0.0
    )
    heterogeneity = (
        statistics.pstdev(finite_effects) if len(finite_effects) >= 2 else 0.0
    )
    relative_heterogeneity = heterogeneity / max(abs(effect), 1e-12)
    lower = effect - 1.96 * standard_error
    upper = effect + 1.96 * standard_error
    supported = bool(
        lower > 0.0
        and sign_agreement >= 0.75
        and relative_heterogeneity <= float(maximum_relative_heterogeneity)
    )
    return {
        "method": "leave_one_environment_out_partially_linear_dml",
        "status": _status(True, supported),
        "available": True,
        "passes": supported,
        "observation_count": len(rows),
        "control_count": len(names),
        "environment_count": len(environment_names),
        "cross_fitted_effect": round(effect, 8),
        "robust_standard_error": round(standard_error, 8),
        "effect_lcb_95": round(lower, 8),
        "effect_ucb_95": round(upper, 8),
        "environment_sign_agreement": round(sign_agreement, 8),
        "effect_heterogeneity": round(heterogeneity, 8),
        "relative_effect_heterogeneity": round(relative_heterogeneity, 8),
        "maximum_relative_heterogeneity": float(maximum_relative_heterogeneity),
        "environment_effects": environment_effects,
        "cross_fit_basis": "leave_one_environment_out_nuisance_estimation",
        "causal_claim_proven": False,
        "required_assumptions": [
            "no_unmeasured_confounding",
            "overlap_and_treatment_variation",
            "correct_point_in_time_ordering",
            "stable_outcome_definition",
            "transportable_structural_relationship",
        ],
        "authority": _authority(),
    }


def capacity_impact_surface(
    *,
    expected_gross_alpha_bps: Any,
    half_spread_bps: Any,
    fees_bps: Any,
    baseline_slippage_bps: Any,
    daily_dollar_volume: Any,
    volatility_bps: Any,
    impact_coefficient: Any,
    notionals: Sequence[Any],
    impact_exponent: float = 0.5,
    minimum_surface_points: int = 3,
) -> dict[str, Any]:
    """Build an explicit square-root impact and net-alpha capacity curve."""

    named = {
        "expected_gross_alpha_bps": _number(expected_gross_alpha_bps),
        "half_spread_bps": _number(half_spread_bps),
        "fees_bps": _number(fees_bps),
        "baseline_slippage_bps": _number(baseline_slippage_bps),
        "daily_dollar_volume": _number(daily_dollar_volume),
        "volatility_bps": _number(volatility_bps),
        "impact_coefficient": _number(impact_coefficient),
    }
    missing = [key for key, value in named.items() if value is None]
    surface_notionals = sorted(
        {
            value
            for raw in notionals
            if (value := _number(raw)) is not None and value > 0.0
        }
    )
    floor = max(int(minimum_surface_points), 2)
    if (
        missing
        or len(surface_notionals) < floor
        or float(named.get("daily_dollar_volume") or 0.0) <= 0.0
    ):
        return {
            "method": "explicit_square_root_capacity_impact_surface",
            "status": "insufficient_evidence",
            "available": False,
            "passes": False,
            "missing_required_inputs": missing,
            "surface_point_count": len(surface_notionals),
            "minimum_surface_points": floor,
            "unknown_cost_defaults_used": False,
            "authority": _authority(),
        }
    exponent = min(max(float(impact_exponent), 0.1), 1.5)
    gross = float(named["expected_gross_alpha_bps"] or 0.0)
    fixed_cost = sum(
        float(named[key] or 0.0)
        for key in ("half_spread_bps", "fees_bps", "baseline_slippage_bps")
    )
    volume = float(named["daily_dollar_volume"] or 0.0)
    volatility = float(named["volatility_bps"] or 0.0)
    coefficient = float(named["impact_coefficient"] or 0.0)
    rows: list[dict[str, Any]] = []
    for notional in surface_notionals:
        participation = notional / volume
        impact = coefficient * volatility * participation**exponent
        total_cost = fixed_cost + impact
        net = gross - total_cost
        rows.append(
            {
                "notional": round(notional, 2),
                "participation_ratio": round(participation, 10),
                "market_impact_bps": round(impact, 8),
                "total_cost_bps": round(total_cost, 8),
                "net_alpha_bps": round(net, 8),
                "expected_net_alpha_dollars": round(notional * net / 10_000.0, 8),
                "positive_net_alpha": net > 0.0,
            }
        )
    positive = [float(row["notional"]) for row in rows if row["positive_net_alpha"]]
    supported = bool(positive)
    return {
        "method": "explicit_square_root_capacity_impact_surface",
        "status": _status(True, supported),
        "available": True,
        "passes": supported,
        "surface_point_count": len(rows),
        "expected_gross_alpha_bps": gross,
        "fixed_cost_bps": round(fixed_cost, 8),
        "daily_dollar_volume": volume,
        "volatility_bps": volatility,
        "impact_coefficient": coefficient,
        "impact_exponent": exponent,
        "largest_tested_positive_net_alpha_notional": (
            max(positive) if positive else None
        ),
        "first_tested_non_positive_notional": next(
            (float(row["notional"]) for row in rows if not row["positive_net_alpha"]),
            None,
        ),
        "surface": rows,
        "unknown_cost_defaults_used": False,
        "authority": _authority(),
    }


def _side(value: Any) -> int:
    text = str(value or "").strip().upper()
    if text in {"BUY", "B", "LONG", "1", "+1"}:
        return 1
    if text in {"SELL", "S", "SHORT", "-1"}:
        return -1
    return 0


def execution_alpha_attribution(
    fills: Sequence[Mapping[str, Any]], *, minimum_fills: int = 1
) -> dict[str, Any]:
    """Attribute decision, arrival, fill, fee, spread, and markout economics."""

    rows: list[dict[str, Any]] = []
    rejected = 0
    for raw in fills:
        side = _side(raw.get("side"))
        quantity = _number(raw.get("quantity"))
        decision = _number(raw.get("decision_price"))
        arrival = _number(raw.get("arrival_price"))
        fill = _number(raw.get("fill_price"))
        if (
            side == 0
            or quantity is None
            or quantity <= 0.0
            or not all(
                value is not None and value > 0.0 for value in (decision, arrival, fill)
            )
        ):
            rejected += 1
            continue
        mid = _number(raw.get("fill_mid_price"))
        markout = _number(raw.get("markout_price"))
        fee_bps = _number(raw.get("fee_bps"))
        if fee_bps is None:
            fee_amount = _number(raw.get("fee_amount")) or 0.0
            fee_bps = fee_amount / (float(fill) * quantity) * 10_000.0
        timing_cost = (
            side * (float(arrival) - float(decision)) / float(decision) * 10_000.0
        )
        route_cost = side * (float(fill) - float(arrival)) / float(arrival) * 10_000.0
        total_shortfall = (
            side * (float(fill) - float(decision)) / float(decision) * 10_000.0
            + fee_bps
        )
        spread_cost = (
            side * (float(fill) - mid) / mid * 10_000.0
            if mid is not None and mid > 0.0
            else None
        )
        markout_alpha = (
            side * (markout - float(fill)) / float(fill) * 10_000.0
            if markout is not None and markout > 0.0
            else None
        )
        execution_alpha = (
            markout_alpha - timing_cost - route_cost - fee_bps
            if markout_alpha is not None
            else None
        )
        rows.append(
            {
                "notional": float(fill) * quantity,
                "timing_cost_bps": timing_cost,
                "route_slippage_bps": route_cost,
                "fee_bps": fee_bps,
                "implementation_shortfall_bps": total_shortfall,
                "spread_cost_bps": spread_cost,
                "post_fill_markout_alpha_bps": markout_alpha,
                "execution_alpha_after_markout_bps": execution_alpha,
            }
        )
    floor = max(int(minimum_fills), 1)
    if len(rows) < floor:
        return {
            "method": "decision_arrival_fill_markout_execution_attribution",
            "status": "insufficient_evidence",
            "available": False,
            "passes": False,
            "fill_count": len(rows),
            "minimum_fills": floor,
            "rejected_fill_count": rejected,
            "authority": _authority(),
        }
    total_notional = sum(float(row["notional"]) for row in rows)

    def weighted(key: str) -> float | None:
        eligible = [row for row in rows if row.get(key) is not None]
        denominator = sum(float(row["notional"]) for row in eligible)
        return (
            sum(float(row["notional"]) * float(row[key]) for row in eligible)
            / denominator
            if denominator > 0.0
            else None
        )

    execution_alpha = weighted("execution_alpha_after_markout_bps")
    markout_coverage = sum(
        row["execution_alpha_after_markout_bps"] is not None for row in rows
    ) / len(rows)
    supported = bool(
        execution_alpha is not None
        and execution_alpha > 0.0
        and markout_coverage == 1.0
    )
    return {
        "method": "decision_arrival_fill_markout_execution_attribution",
        "status": _status(True, supported),
        "available": True,
        "passes": supported,
        "fill_count": len(rows),
        "rejected_fill_count": rejected,
        "total_notional": round(total_notional, 2),
        "markout_coverage_ratio": round(markout_coverage, 8),
        "weighted_timing_cost_bps": round(weighted("timing_cost_bps") or 0.0, 8),
        "weighted_route_slippage_bps": round(weighted("route_slippage_bps") or 0.0, 8),
        "weighted_fee_bps": round(weighted("fee_bps") or 0.0, 8),
        "weighted_implementation_shortfall_bps": round(
            weighted("implementation_shortfall_bps") or 0.0, 8
        ),
        "weighted_spread_cost_bps": (
            round(value, 8)
            if (value := weighted("spread_cost_bps")) is not None
            else None
        ),
        "weighted_post_fill_markout_alpha_bps": (
            round(value, 8)
            if (value := weighted("post_fill_markout_alpha_bps")) is not None
            else None
        ),
        "weighted_execution_alpha_after_markout_bps": (
            round(execution_alpha, 8) if execution_alpha is not None else None
        ),
        "sign_convention": "positive_cost_is_adverse; positive_markout_or_execution_alpha_is_favorable",
        "authority": _authority(),
    }


def _timestamp(value: Any) -> datetime | None:
    raw = str(value or "").strip().replace("Z", "+00:00")
    if not raw:
        return None
    try:
        parsed = datetime.fromisoformat(raw)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def point_in_time_security_master_audit(
    security_records: Sequence[Mapping[str, Any]],
    observations: Sequence[Mapping[str, Any]],
    *,
    minimum_identifier_coverage_ratio: float = 1.0,
    minimum_observation_resolution_ratio: float = 1.0,
) -> dict[str, Any]:
    """Audit effective-dated identity, symbol reuse, actions, and delistings."""

    records: list[dict[str, Any]] = []
    invalid = 0
    stable_identifier_keys = {"figi", "cusip", "isin", "perm_id", "lei"}
    for raw in security_records:
        security_id = str(raw.get("security_id") or "").strip()
        symbol = str(raw.get("symbol") or "").strip().upper()
        valid_from = _timestamp(raw.get("valid_from"))
        valid_to = _timestamp(raw.get("valid_to"))
        if (
            not security_id
            or not symbol
            or valid_from is None
            or (valid_to is not None and valid_to <= valid_from)
        ):
            invalid += 1
            continue
        identifiers = (
            raw.get("identifiers")
            if isinstance(raw.get("identifiers"), Mapping)
            else {}
        )
        has_identifier = any(
            str(key).lower() in stable_identifier_keys and str(value or "").strip()
            for key, value in identifiers.items()
        )
        status = str(raw.get("status") or "active").strip().lower()
        delisted_at = _timestamp(raw.get("delisted_at"))
        corporate_actions = (
            raw.get("corporate_actions")
            if isinstance(raw.get("corporate_actions"), list)
            else []
        )
        action_covered = all(
            isinstance(action, Mapping)
            and _timestamp(action.get("effective_at")) is not None
            and _number(action.get("adjustment_factor")) not in {None, 0.0}
            for action in corporate_actions
        )
        records.append(
            {
                "security_id": security_id,
                "symbol": symbol,
                "valid_from": valid_from,
                "valid_to": valid_to,
                "has_stable_identifier": has_identifier,
                "delisting_covered": status != "delisted" or delisted_at is not None,
                "corporate_action_covered": action_covered,
            }
        )
    overlaps: list[dict[str, Any]] = []
    by_symbol: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_symbol[str(record["symbol"])].append(record)
    far_future = datetime.max.replace(tzinfo=timezone.utc)
    for symbol, symbol_records in sorted(by_symbol.items()):
        ordered = sorted(symbol_records, key=lambda row: row["valid_from"])
        for left, right in zip(ordered, ordered[1:]):
            if (left["valid_to"] or far_future) > right["valid_from"]:
                overlaps.append(
                    {
                        "symbol": symbol,
                        "left_security_id": left["security_id"],
                        "right_security_id": right["security_id"],
                    }
                )
    resolved = 0
    ambiguous = 0
    unresolved = 0
    for observation in observations:
        timestamp = _timestamp(
            observation.get("timestamp_utc") or observation.get("timestamp")
        )
        symbol = str(observation.get("symbol") or "").strip().upper()
        security_id = str(observation.get("security_id") or "").strip()
        if timestamp is None or (not symbol and not security_id):
            unresolved += 1
            continue
        matches = [
            record
            for record in records
            if (not symbol or record["symbol"] == symbol)
            and (not security_id or record["security_id"] == security_id)
            and record["valid_from"] <= timestamp
            and (record["valid_to"] is None or timestamp < record["valid_to"])
        ]
        if len(matches) == 1:
            resolved += 1
        elif len(matches) > 1:
            ambiguous += 1
        else:
            unresolved += 1
    record_count = len(records)
    observation_count = len(observations)
    identifier_ratio = (
        sum(bool(row["has_stable_identifier"]) for row in records) / record_count
        if record_count
        else 0.0
    )
    action_ratio = (
        sum(bool(row["corporate_action_covered"]) for row in records) / record_count
        if record_count
        else 0.0
    )
    delisting_ratio = (
        sum(bool(row["delisting_covered"]) for row in records) / record_count
        if record_count
        else 0.0
    )
    resolution_ratio = resolved / observation_count if observation_count else 0.0
    available = bool(record_count and observation_count)
    supported = bool(
        available
        and not overlaps
        and invalid == 0
        and ambiguous == 0
        and identifier_ratio >= float(minimum_identifier_coverage_ratio)
        and resolution_ratio >= float(minimum_observation_resolution_ratio)
        and action_ratio == 1.0
        and delisting_ratio == 1.0
    )
    blockers: list[str] = []
    if not record_count:
        blockers.append("security_records_missing")
    if not observation_count:
        blockers.append("point_in_time_observations_missing")
    if invalid:
        blockers.append("invalid_effective_dated_records")
    if overlaps:
        blockers.append("overlapping_symbol_identity_intervals")
    if ambiguous:
        blockers.append("ambiguous_point_in_time_resolution")
    if unresolved:
        blockers.append("unresolved_point_in_time_observations")
    if identifier_ratio < float(minimum_identifier_coverage_ratio):
        blockers.append("stable_identifier_coverage_below_floor")
    if action_ratio < 1.0:
        blockers.append("corporate_action_adjustment_coverage_incomplete")
    if delisting_ratio < 1.0:
        blockers.append("delisting_coverage_incomplete")
    return {
        "method": "effective_dated_security_master_resolution_audit",
        "status": _status(available, supported),
        "available": available,
        "passes": supported,
        "record_count": record_count,
        "invalid_record_count": invalid,
        "observation_count": observation_count,
        "resolved_observation_count": resolved,
        "ambiguous_observation_count": ambiguous,
        "unresolved_observation_count": unresolved,
        "symbol_interval_overlap_count": len(overlaps),
        "symbol_interval_overlaps": overlaps,
        "stable_identifier_coverage_ratio": round(identifier_ratio, 8),
        "corporate_action_coverage_ratio": round(action_ratio, 8),
        "delisting_coverage_ratio": round(delisting_ratio, 8),
        "observation_resolution_ratio": round(resolution_ratio, 8),
        "blockers": blockers,
        "authority": _authority(),
    }


def split_conformal_residual_calibration(
    predictions: Sequence[Any],
    outcomes: Sequence[Any],
    *,
    minimum_observations: int = 30,
    calibration_fraction: float = 0.6,
    miscoverage_alpha: float = 0.1,
    coverage_tolerance: float = 0.05,
    maximum_width_to_outcome_scale: float = 2.0,
) -> dict[str, Any]:
    """Measure time-ordered split-conformal residual coverage and sharpness."""

    paired = [
        (prediction, outcome)
        for raw_prediction, raw_outcome in zip(predictions, outcomes)
        if (prediction := _number(raw_prediction)) is not None
        and (outcome := _number(raw_outcome)) is not None
    ]
    minimum = max(int(minimum_observations), 4)
    alpha = min(max(float(miscoverage_alpha), 1e-6), 0.5)
    fraction = min(max(float(calibration_fraction), 0.25), 0.8)
    if len(paired) < minimum:
        return {
            "method": "time_ordered_split_conformal_absolute_residual",
            "status": "insufficient_evidence",
            "available": False,
            "passes": False,
            "observation_count": len(paired),
            "minimum_observations": minimum,
            "authority": _authority(),
        }
    split = min(max(int(len(paired) * fraction), 2), len(paired) - 2)
    calibration = paired[:split]
    evaluation = paired[split:]
    residuals = np.asarray(
        [abs(outcome - prediction) for prediction, outcome in calibration],
        dtype=float,
    )
    rank = min(
        math.ceil((len(residuals) + 1) * (1.0 - alpha)) / len(residuals),
        1.0,
    )
    try:
        radius = float(np.quantile(residuals, rank, method="higher"))
    except TypeError:  # NumPy < 1.22 compatibility.
        radius = float(np.quantile(residuals, rank, interpolation="higher"))
    evaluation_residuals = np.asarray(
        [abs(outcome - prediction) for prediction, outcome in evaluation],
        dtype=float,
    )
    coverage = float(np.mean(evaluation_residuals <= radius))
    target = 1.0 - alpha
    evaluation_outcomes = np.asarray(
        [outcome for _, outcome in evaluation], dtype=float
    )
    outcome_scale = (
        float(np.std(evaluation_outcomes, ddof=1)) if len(evaluation) > 1 else 0.0
    )
    width_to_scale = (2.0 * radius / outcome_scale) if outcome_scale > 1e-12 else None
    coverage_floor = max(target - float(coverage_tolerance), 0.0)
    coverage_quality = max(
        0.0,
        1.0 - abs(coverage - target) / max(float(coverage_tolerance), alpha),
    )
    sharpness_ready = bool(
        width_to_scale is not None
        and width_to_scale <= max(float(maximum_width_to_outcome_scale), 0.0)
    )
    supported = bool(coverage >= coverage_floor and sharpness_ready)
    blockers: list[str] = []
    if coverage < coverage_floor:
        blockers.append("evaluation_coverage_below_floor")
    if width_to_scale is None:
        blockers.append("evaluation_outcome_scale_not_estimable")
    elif not sharpness_ready:
        blockers.append("conformal_interval_too_wide_for_outcome_scale")
    return {
        "method": "time_ordered_split_conformal_absolute_residual",
        "status": _status(True, supported),
        "available": True,
        "passes": supported,
        "observation_count": len(paired),
        "calibration_observation_count": len(calibration),
        "evaluation_observation_count": len(evaluation),
        "miscoverage_alpha": round(alpha, 8),
        "target_coverage": round(target, 8),
        "empirical_coverage": round(coverage, 8),
        "coverage_quality_norm": round(min(max(coverage_quality, 0.0), 1.0), 8),
        "interval_radius": round(radius, 8),
        "interval_width_to_outcome_scale": (
            round(width_to_scale, 8) if width_to_scale is not None else None
        ),
        "time_order_preserved": True,
        "random_split_used": False,
        "blockers": blockers,
        "authority": _authority(),
    }


def sequential_change_point_stability(
    values_by_group: Mapping[str, Sequence[Any]],
    *,
    minimum_observations_per_group: int = 30,
    minimum_groups: int = 1,
    burn_in_observations: int = 10,
    drift_delta_standard_deviations: float = 0.05,
    alarm_threshold_standard_deviations: float = 5.0,
    recent_window_observations: int = 10,
    minimum_stability_score: float = 0.7,
) -> dict[str, Any]:
    """Run a bounded two-sided Page-CUSUM stability diagnostic per group."""

    rows: list[dict[str, Any]] = []
    minimum = max(int(minimum_observations_per_group), 6)
    burn_in = max(int(burn_in_observations), 4)
    recent_window = max(int(recent_window_observations), 1)
    delta = max(float(drift_delta_standard_deviations), 0.0)
    threshold = max(float(alarm_threshold_standard_deviations), 1e-6)
    for group, raw_values in sorted(values_by_group.items()):
        values = _finite(raw_values)
        if len(values) < minimum or len(values) <= burn_in:
            continue
        baseline = values[:burn_in]
        center = statistics.fmean(baseline)
        scale = statistics.stdev(baseline) if len(baseline) > 1 else 0.0
        if scale <= 1e-12:
            scale = statistics.pstdev(values)
        if scale <= 1e-12:
            scale = 1.0
        positive = 0.0
        negative = 0.0
        alarms: list[int] = []
        for index, value in enumerate(values[burn_in:], start=burn_in):
            standardized = (value - center) / scale
            positive = max(0.0, positive + standardized - delta)
            negative = min(0.0, negative + standardized + delta)
            if positive > threshold or abs(negative) > threshold:
                alarms.append(index)
                positive = 0.0
                negative = 0.0
        recent_alarm = bool(alarms and alarms[-1] >= len(values) - recent_window)
        expected_windows = max((len(values) - burn_in) / max(recent_window, 1), 1.0)
        stability = math.exp(-len(alarms) / expected_windows)
        if recent_alarm:
            stability *= 0.5
        rows.append(
            {
                "group": str(group),
                "observation_count": len(values),
                "change_point_count": len(alarms),
                "change_point_indices": alarms[-10:],
                "last_change_point_index": alarms[-1] if alarms else None,
                "recent_change_point": recent_alarm,
                "stability_score_norm": round(min(max(stability, 0.0), 1.0), 8),
            }
        )
    floor_groups = max(int(minimum_groups), 1)
    available = len(rows) >= floor_groups
    overall_stability = (
        statistics.fmean(row["stability_score_norm"] for row in rows) if rows else 0.0
    )
    supported = bool(
        available
        and overall_stability >= float(minimum_stability_score)
        and not any(row["recent_change_point"] for row in rows)
    )
    blockers: list[str] = []
    if not available:
        blockers.append("insufficient_candidate_bound_group_history")
    if available and overall_stability < float(minimum_stability_score):
        blockers.append("aggregate_change_point_stability_below_floor")
    if any(row["recent_change_point"] for row in rows):
        blockers.append("recent_change_point_detected")
    return {
        "method": "two_sided_page_cusum_group_stability",
        "status": _status(available, supported),
        "available": available,
        "passes": supported,
        "group_count": len(rows),
        "minimum_groups": floor_groups,
        "observation_count": sum(row["observation_count"] for row in rows),
        "change_point_count": sum(row["change_point_count"] for row in rows),
        "recent_change_point_group_count": sum(
            row["recent_change_point"] for row in rows
        ),
        "stability_score_norm": round(min(max(overall_stability, 0.0), 1.0), 8),
        "groups": rows,
        "blockers": blockers,
        "authority": _authority(),
    }


def residual_redundancy_graph(
    returns_by_group: Mapping[str, Sequence[Any]],
    *,
    minimum_common_observations: int = 20,
    minimum_groups: int = 2,
    absolute_correlation_threshold: float = 0.7,
    minimum_independent_group_ratio: float = 0.5,
) -> dict[str, Any]:
    """Cluster candidate return streams that remain economically redundant."""

    minimum_common = max(int(minimum_common_observations), 3)
    groups = {
        str(group): _finite(values)
        for group, values in returns_by_group.items()
        if len(_finite(values)) >= minimum_common
    }
    names = sorted(groups)
    adjacency: dict[str, set[str]] = {name: set() for name in names}
    pairs: list[dict[str, Any]] = []
    threshold = min(max(float(absolute_correlation_threshold), 0.0), 1.0)
    for left_index, left_name in enumerate(names):
        for right_name in names[left_index + 1 :]:
            common = min(len(groups[left_name]), len(groups[right_name]))
            left = groups[left_name][-common:]
            right = groups[right_name][-common:]
            correlation = _correlation(left, right)
            if correlation is None:
                continue
            redundant = abs(correlation) >= threshold
            if redundant:
                adjacency[left_name].add(right_name)
                adjacency[right_name].add(left_name)
            pairs.append(
                {
                    "left": left_name,
                    "right": right_name,
                    "common_observations": common,
                    "correlation": round(correlation, 8),
                    "absolute_correlation": round(abs(correlation), 8),
                    "redundant": redundant,
                }
            )
    components: list[list[str]] = []
    remaining = set(names)
    while remaining:
        seed = min(remaining)
        stack = [seed]
        component: set[str] = set()
        while stack:
            current = stack.pop()
            if current in component:
                continue
            component.add(current)
            stack.extend(adjacency[current] - component)
        remaining -= component
        components.append(sorted(component))
    floor_groups = max(int(minimum_groups), 2)
    available = bool(len(names) >= floor_groups and pairs)
    independent_ratio = len(components) / len(names) if names else 0.0
    supported = bool(
        available and independent_ratio >= float(minimum_independent_group_ratio)
    )
    return {
        "method": "absolute_correlation_redundancy_components",
        "status": _status(available, supported),
        "available": available,
        "passes": supported,
        "group_count": len(names),
        "minimum_groups": floor_groups,
        "pair_count": len(pairs),
        "redundant_pair_count": sum(row["redundant"] for row in pairs),
        "independent_component_count": len(components),
        "independent_group_ratio": round(independent_ratio, 8),
        "maximum_absolute_pairwise_correlation": (
            round(max(row["absolute_correlation"] for row in pairs), 8)
            if pairs
            else None
        ),
        "components": components,
        "pairwise_correlations": sorted(
            pairs,
            key=lambda row: (
                -float(row["absolute_correlation"]),
                row["left"],
                row["right"],
            ),
        ),
        "blockers": (
            []
            if supported
            else [
                (
                    "insufficient_group_overlap"
                    if not available
                    else "independent_group_ratio_below_floor"
                )
            ]
        ),
        "authority": _authority(),
    }


def regime_conditional_robustness(
    outcomes: Sequence[Any],
    regimes: Sequence[Any],
    *,
    minimum_regimes: int = 2,
    minimum_observations_per_regime: int = 10,
    minimum_supported_regime_ratio: float = 0.67,
    minimum_mean_lcb: float = 0.0,
) -> dict[str, Any]:
    """Require positive lower-bound evidence across independently observed regimes."""

    grouped: dict[str, list[float]] = defaultdict(list)
    for raw_outcome, raw_regime in zip(outcomes, regimes):
        outcome = _number(raw_outcome)
        regime = str(raw_regime or "").strip().lower()
        if outcome is not None and regime:
            grouped[regime].append(outcome)
    minimum_per_regime = max(int(minimum_observations_per_regime), 2)
    rows: list[dict[str, Any]] = []
    for regime, values in sorted(grouped.items()):
        if len(values) < minimum_per_regime:
            continue
        mean, lcb = _mean_lcb(values)
        supported = bool(lcb is not None and lcb > float(minimum_mean_lcb))
        rows.append(
            {
                "regime": regime,
                "observation_count": len(values),
                "mean_outcome": round(float(mean or 0.0), 8),
                "mean_lcb_95": round(float(lcb or 0.0), 8) if lcb is not None else None,
                "supported": supported,
            }
        )
    floor_regimes = max(int(minimum_regimes), 2)
    available = len(rows) >= floor_regimes
    supported_ratio = sum(row["supported"] for row in rows) / len(rows) if rows else 0.0
    supported = bool(
        available and supported_ratio >= float(minimum_supported_regime_ratio)
    )
    return {
        "method": "regime_stratified_mean_lower_confidence_bounds",
        "status": _status(available, supported),
        "available": available,
        "passes": supported,
        "observation_count": sum(row["observation_count"] for row in rows),
        "regime_count": len(rows),
        "minimum_regimes": floor_regimes,
        "supported_regime_count": sum(row["supported"] for row in rows),
        "supported_regime_ratio": round(supported_ratio, 8),
        "regimes": rows,
        "blockers": (
            []
            if supported
            else [
                (
                    "independent_regime_depth_pending"
                    if not available
                    else "supported_regime_ratio_below_floor"
                )
            ]
        ),
        "authority": _authority(),
    }


def cost_stress_survival(
    expected_gross_alpha_bps: Any,
    base_cost_bps: Any,
    *,
    uncertainty_buffer_bps: Any = 0.0,
    cost_multipliers: Sequence[Any] = (1.0, 1.25, 1.5, 2.0),
    minimum_positive_scenario_ratio: float = 0.75,
) -> dict[str, Any]:
    """Stress conservative net edge across monotonic transaction-cost shocks."""

    gross = _number(expected_gross_alpha_bps)
    cost = _number(base_cost_bps)
    uncertainty = _number(uncertainty_buffer_bps)
    multipliers = sorted(
        {
            value
            for raw in cost_multipliers
            if (value := _number(raw)) is not None and value >= 1.0
        }
    )
    if gross is None or cost is None or uncertainty is None or not multipliers:
        return {
            "method": "monotonic_transaction_cost_stress_survival",
            "status": "insufficient_evidence",
            "available": False,
            "passes": False,
            "scenario_count": len(multipliers),
            "blockers": ["gross_alpha_cost_or_uncertainty_input_missing"],
            "authority": _authority(),
        }
    scenarios = [
        {
            "cost_multiplier": round(multiplier, 8),
            "stressed_cost_bps": round(cost * multiplier, 8),
            "conservative_net_edge_bps": round(
                gross - cost * multiplier - max(uncertainty, 0.0), 8
            ),
        }
        for multiplier in multipliers
    ]
    for row in scenarios:
        row["positive_conservative_net_edge"] = bool(
            row["conservative_net_edge_bps"] > 0.0
        )
    positive_ratio = sum(
        row["positive_conservative_net_edge"] for row in scenarios
    ) / len(scenarios)
    supported = positive_ratio >= float(minimum_positive_scenario_ratio)
    return {
        "method": "monotonic_transaction_cost_stress_survival",
        "status": _status(True, supported),
        "available": True,
        "passes": supported,
        "scenario_count": len(scenarios),
        "positive_scenario_count": sum(
            row["positive_conservative_net_edge"] for row in scenarios
        ),
        "positive_scenario_ratio": round(positive_ratio, 8),
        "worst_conservative_net_edge_bps": round(
            min(row["conservative_net_edge_bps"] for row in scenarios), 8
        ),
        "scenarios": scenarios,
        "blockers": [] if supported else ["cost_stress_survival_ratio_below_floor"],
        "authority": _authority(),
    }


def active_learning_value_of_information(
    collection_gaps: Sequence[Mapping[str, Any]], *, minimum_gaps: int = 1
) -> dict[str, Any]:
    """Rank explicit collection gaps by cost-aware expected information value."""

    rows: list[dict[str, Any]] = []
    rejected = 0
    for raw in collection_gaps:
        gap_id = str(raw.get("gap_id") or "").strip()
        values = {
            key: _number(raw.get(key))
            for key in (
                "uncertainty",
                "economic_relevance",
                "expected_uncertainty_reduction",
                "novelty",
                "coverage_deficit",
                "observation_cost",
                "resource_pressure",
            )
        }
        if not gap_id or any(value is None for value in values.values()):
            rejected += 1
            continue
        bounded = {
            key: min(max(float(value or 0.0), 0.0), 1.0)
            for key, value in values.items()
            if key != "observation_cost"
        }
        cost = max(float(values["observation_cost"] or 0.0), 1e-9)
        raw_score = (
            bounded["uncertainty"]
            * bounded["economic_relevance"]
            * bounded["expected_uncertainty_reduction"]
            * bounded["novelty"]
            * bounded["coverage_deficit"]
            / (cost * (1.0 + bounded["resource_pressure"]))
        )
        minimum = max(int(float(raw.get("minimum_observations") or 0)), 0)
        observed = max(int(float(raw.get("observed_observations") or 0)), 0)
        rows.append(
            {
                "gap_id": gap_id,
                "route": str(raw.get("route") or "candidate_research_collection"),
                "raw_value_of_information": raw_score,
                "minimum_observations": minimum,
                "observed_observations": observed,
                "remaining_observations": max(minimum - observed, 0),
                "inputs": {**bounded, "observation_cost": cost},
            }
        )
    floor = max(int(minimum_gaps), 1)
    if len(rows) < floor:
        return {
            "method": "cost_aware_active_learning_value_of_information",
            "status": "insufficient_evidence",
            "available": False,
            "passes": False,
            "gap_count": len(rows),
            "minimum_gaps": floor,
            "rejected_gap_count": rejected,
            "authority": _authority(),
        }
    maximum = max(float(row["raw_value_of_information"]) for row in rows)
    for row in rows:
        row["priority_score_0_100"] = round(
            (
                100.0 * float(row.pop("raw_value_of_information")) / maximum
                if maximum > 0.0
                else 0.0
            ),
            8,
        )
    rows.sort(
        key=lambda row: (float(row["priority_score_0_100"]), str(row["gap_id"])),
        reverse=True,
    )
    for index, row in enumerate(rows, start=1):
        row["priority_rank"] = index
    return {
        "method": "cost_aware_active_learning_value_of_information",
        "status": "advisory",
        "available": True,
        "passes": False,
        "gap_count": len(rows),
        "rejected_gap_count": rejected,
        "ranked_collection_gaps": rows,
        "top_gap_id": str(rows[0]["gap_id"]),
        "labels_or_samples_created": False,
        "economic_support_claimed": False,
        "authority": _authority(),
    }


MEASUREMENT_FUNCTIONS = {
    "information_coefficient_term_structure": information_coefficient_term_structure,
    "effective_breadth_transfer_coefficient": effective_breadth_transfer_coefficient,
    "hierarchical_bayesian_skill": hierarchical_bayesian_skill,
    "subsample_stability_selection": subsample_stability_selection,
    "economic_alpha_decomposition": economic_alpha_decomposition,
    "factor_neutral_residualization": factor_neutral_residualization,
    "cross_fitted_causal_transportability": cross_fitted_causal_transportability,
    "capacity_impact_surface": capacity_impact_surface,
    "execution_alpha_attribution": execution_alpha_attribution,
    "point_in_time_security_master_audit": point_in_time_security_master_audit,
    "split_conformal_residual_calibration": split_conformal_residual_calibration,
    "sequential_change_point_stability": sequential_change_point_stability,
    "residual_redundancy_graph": residual_redundancy_graph,
    "regime_conditional_robustness": regime_conditional_robustness,
    "cost_stress_survival": cost_stress_survival,
    "active_learning_value_of_information": active_learning_value_of_information,
}
