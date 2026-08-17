from __future__ import annotations

import hashlib
import json
import math
import random
from collections import defaultdict
from datetime import datetime, timezone
from itertools import combinations
from statistics import NormalDist
from typing import Any, Iterable, Mapping, Sequence


NORMAL = NormalDist()


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except Exception:
        return float(default)
    return parsed if math.isfinite(parsed) else float(default)


def _mean(values: Sequence[float]) -> float:
    return sum(values) / max(len(values), 1)


def _sample_stddev(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = _mean(values)
    return math.sqrt(max(sum((value - mean) ** 2 for value in values) / (len(values) - 1), 0.0))


def _metadata(row: Mapping[str, Any]) -> Mapping[str, Any]:
    raw = row.get("metadata")
    return raw if isinstance(raw, Mapping) else {}


def _text(row: Mapping[str, Any], *keys: str, default: str = "") -> str:
    metadata = _metadata(row)
    for key in keys:
        value = row.get(key)
        if value is None or str(value).strip() == "":
            value = metadata.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return default


def _timestamp(row: Mapping[str, Any]) -> datetime | None:
    raw = _text(row, "timestamp_utc", "timestamp")
    if not raw:
        return None
    try:
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except Exception:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _day_key(row: Mapping[str, Any], index: int) -> str:
    timestamp = _timestamp(row)
    return timestamp.date().isoformat() if timestamp is not None else f"unknown-day-{index}"


def _session_key(row: Mapping[str, Any], index: int) -> str:
    day = _day_key(row, index)
    session = _text(row, "session", "market_session", default="unknown").lower()
    return f"{day}:{session}"


def _serial_effective_sample_size(values: Sequence[float]) -> tuple[float, float]:
    count = len(values)
    if count < 3:
        return float(max(count, 1)), 0.0
    mean = _mean(values)
    denominator = sum((value - mean) ** 2 for value in values)
    if denominator <= 1e-18:
        return float(count), 0.0
    numerator = sum((values[index] - mean) * (values[index - 1] - mean) for index in range(1, count))
    rho = max(min(numerator / denominator, 0.99), -0.99)
    effective = count * (1.0 - rho) / max(1.0 + rho, 1e-9)
    return max(1.0, min(float(count), effective)), rho


def _cluster_effective_sample_size(values: Sequence[float], labels: Sequence[str]) -> tuple[float, float]:
    count = len(values)
    groups: dict[str, list[float]] = defaultdict(list)
    for value, label in zip(values, labels):
        groups[str(label)].append(float(value))
    group_count = len(groups)
    if count <= 1 or group_count <= 1:
        return 1.0, 1.0
    grand = _mean(values)
    between = sum(len(group) * (_mean(group) - grand) ** 2 for group in groups.values())
    within = sum(sum((value - _mean(group)) ** 2 for value in group) for group in groups.values())
    ms_between = between / max(group_count - 1, 1)
    ms_within = within / max(count - group_count, 1)
    mean_size = count / group_count
    denominator = ms_between + max(mean_size - 1.0, 0.0) * ms_within
    if denominator <= 1e-18:
        intracluster_correlation = 1.0
    else:
        intracluster_correlation = max(0.0, min((ms_between - ms_within) / denominator, 1.0))
    design_effect = 1.0 + max(mean_size - 1.0, 0.0) * intracluster_correlation
    return max(1.0, min(float(count), count / max(design_effect, 1.0))), intracluster_correlation


def _cluster_standard_error(values: Sequence[float], labels: Sequence[str]) -> float | None:
    count = len(values)
    groups: dict[str, list[float]] = defaultdict(list)
    for value, label in zip(values, labels):
        groups[str(label)].append(float(value))
    group_count = len(groups)
    if count < 2 or group_count < 2:
        return None
    mean = _mean(values)
    cluster_sums = [sum(value - mean for value in group) for group in groups.values()]
    variance = (group_count / (group_count - 1.0)) * sum(value * value for value in cluster_sums) / (count * count)
    return math.sqrt(max(variance, 0.0))


def _percentile(values: Sequence[float], probability: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    position = max(0.0, min(float(probability), 1.0)) * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _block_bootstrap_lcb(
    values: Sequence[float],
    labels: Sequence[str],
    *,
    iterations: int,
    seed_material: str,
) -> float | None:
    groups: dict[str, list[float]] = defaultdict(list)
    for value, label in zip(values, labels):
        groups[str(label)].append(float(value))
    blocks = list(groups.values())
    if len(blocks) < 2:
        return None
    seed = int(hashlib.sha256(seed_material.encode("utf-8")).hexdigest()[:16], 16)
    rng = random.Random(seed)
    estimates: list[float] = []
    for _ in range(max(int(iterations), 200)):
        sampled = [blocks[rng.randrange(len(blocks))] for _index in range(len(blocks))]
        flat = [value for block in sampled for value in block]
        estimates.append(_mean(flat))
    return _percentile(estimates, 0.025)


def _skew_kurtosis(values: Sequence[float]) -> tuple[float, float]:
    count = len(values)
    stddev = _sample_stddev(values)
    if count < 3 or stddev <= 1e-18:
        return 0.0, 3.0
    mean = _mean(values)
    skew = sum(((value - mean) / stddev) ** 3 for value in values) / count
    kurtosis = sum(((value - mean) / stddev) ** 4 for value in values) / count
    return skew, max(kurtosis, 1.0)


def deflated_sharpe_probability(
    values: Sequence[float],
    *,
    effective_sample_size: float,
    hypothesis_count: int,
) -> dict[str, Any]:
    count = len(values)
    stddev = _sample_stddev(values)
    if count < 3 or stddev <= 1e-18 or effective_sample_size < 3.0:
        return {"available": False, "probability": None, "sharpe": None, "benchmark_sharpe": None}
    sharpe = _mean(values) / stddev
    trials = max(int(hypothesis_count), 1)
    expected_max_z = NORMAL.inv_cdf(max(min(1.0 - (1.0 / max(trials, 2)), 1.0 - 1e-9), 1e-9)) if trials > 1 else 0.0
    benchmark = expected_max_z / math.sqrt(max(effective_sample_size - 1.0, 1.0))
    skew, kurtosis = _skew_kurtosis(values)
    denominator = math.sqrt(max(1.0 - skew * sharpe + ((kurtosis - 1.0) / 4.0) * sharpe * sharpe, 1e-9))
    test_value = (sharpe - benchmark) * math.sqrt(max(effective_sample_size - 1.0, 1.0)) / denominator
    return {
        "available": True,
        "probability": round(NORMAL.cdf(test_value), 8),
        "sharpe": round(sharpe, 8),
        "benchmark_sharpe": round(benchmark, 8),
        "skew": round(skew, 8),
        "kurtosis": round(kurtosis, 8),
        "hypothesis_count": trials,
    }


def clustered_post_cost_statistics(
    rows: Iterable[Mapping[str, Any]],
    *,
    pnl_key: str = "post_cost_pnl_delta",
    return_key: str = "post_cost_return_bps",
    minimum_samples: int = 30,
    minimum_days: int = 7,
    minimum_symbols: int = 5,
    minimum_effective_samples: float = 20.0,
    hypothesis_count: int = 1,
    bootstrap_iterations: int = 1200,
) -> dict[str, Any]:
    observations: list[tuple[float, float, Mapping[str, Any]]] = []
    for row in rows:
        pnl = _safe_float(row.get(pnl_key), float("nan"))
        returns = _safe_float(row.get(return_key), float("nan"))
        if math.isfinite(pnl) and math.isfinite(returns):
            observations.append((pnl, returns, row))
    count = len(observations)
    if not observations:
        return {
            "available": False,
            "promotion_evidence_sufficient": False,
            "positive_clustered_lower_confidence_bound_95": False,
            "blockers": ["no_post_cost_observations"],
        }

    pnl_values = [item[0] for item in observations]
    return_values = [item[1] for item in observations]
    rows_list = [item[2] for item in observations]
    day_labels = [_day_key(row, index) for index, row in enumerate(rows_list)]
    session_labels = [_session_key(row, index) for index, row in enumerate(rows_list)]
    symbol_labels = [_text(row, "symbol", default="unknown") for row in rows_list]
    strategy_labels = [_text(row, "strategy", "source_profile", "profile", default="unknown") for row in rows_list]
    regime_labels = [_text(row, "regime", "market_regime", "regime_label", default="unknown") for row in rows_list]
    dimensions = {
        "day": day_labels,
        "session": session_labels,
        "symbol": symbol_labels,
        "strategy": strategy_labels,
        "regime": regime_labels,
    }
    serial_ess, lag_one = _serial_effective_sample_size(return_values)
    cluster_effective: dict[str, float] = {}
    intracluster: dict[str, float] = {}
    standard_errors: dict[str, float | None] = {}
    for name, labels in dimensions.items():
        effective, correlation = _cluster_effective_sample_size(return_values, labels)
        cluster_effective[name] = effective
        intracluster[name] = correlation
        standard_errors[name] = _cluster_standard_error(return_values, labels)
    eligible_effective = [serial_ess, cluster_effective["day"]]
    effective_n = max(1.0, min(eligible_effective))
    iid_se_pnl = _sample_stddev(pnl_values) / math.sqrt(count) if count > 1 else 0.0
    iid_se_return = _sample_stddev(return_values) / math.sqrt(count) if count > 1 else 0.0
    pnl_cluster_errors = [
        value
        for value in (
            _cluster_standard_error(pnl_values, day_labels),
            _cluster_standard_error(pnl_values, symbol_labels) if len(set(symbol_labels)) > 1 else None,
            _cluster_standard_error(pnl_values, strategy_labels) if len(set(strategy_labels)) > 1 else None,
        )
        if value is not None
    ]
    return_cluster_errors = [value for value in standard_errors.values() if value is not None]
    conservative_pnl_se = max([iid_se_pnl, *pnl_cluster_errors])
    conservative_return_se = max([iid_se_return, *return_cluster_errors])
    mean_pnl = _mean(pnl_values)
    mean_return = _mean(return_values)
    clustered_pnl_lcb = mean_pnl - 1.96 * conservative_pnl_se
    clustered_return_lcb = mean_return - 1.96 * conservative_return_se
    seed_material = json.dumps(
        {"days": day_labels, "pnl": pnl_values, "returns": return_values},
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )
    bootstrap_pnl_lcb = _block_bootstrap_lcb(
        pnl_values,
        day_labels,
        iterations=bootstrap_iterations,
        seed_material=f"pnl:{seed_material}",
    )
    bootstrap_return_lcb = _block_bootstrap_lcb(
        return_values,
        day_labels,
        iterations=bootstrap_iterations,
        seed_material=f"return:{seed_material}",
    )
    final_pnl_lcb = min(clustered_pnl_lcb, bootstrap_pnl_lcb) if bootstrap_pnl_lcb is not None else None
    final_return_lcb = min(clustered_return_lcb, bootstrap_return_lcb) if bootstrap_return_lcb is not None else None
    unique_days = len(set(day_labels))
    unique_symbols = len({value for value in symbol_labels if value and value != "unknown"})
    unique_strategies = len({value for value in strategy_labels if value and value != "unknown"})
    unique_regimes = len({value for value in regime_labels if value and value != "unknown"})
    blockers: list[str] = []
    if count < max(int(minimum_samples), 1):
        blockers.append("minimum_trade_samples_pending")
    if unique_days < max(int(minimum_days), 1):
        blockers.append("minimum_independent_days_pending")
    if unique_symbols < max(int(minimum_symbols), 1):
        blockers.append("minimum_symbol_breadth_pending")
    if effective_n < max(float(minimum_effective_samples), 1.0):
        blockers.append("cluster_effective_sample_size_pending")
    evidence_sufficient = not blockers
    positive_lcb = bool(
        evidence_sufficient
        and final_pnl_lcb is not None
        and final_return_lcb is not None
        and final_pnl_lcb > 0.0
        and final_return_lcb > 0.0
    )
    one_sided_p = (
        1.0 - NORMAL.cdf(mean_return / conservative_return_se)
        if conservative_return_se > 1e-18
        else 0.0
        if mean_return > 0.0
        else 1.0
    )
    dsr = deflated_sharpe_probability(
        return_values,
        effective_sample_size=effective_n,
        hypothesis_count=hypothesis_count,
    )
    return {
        "available": True,
        "sample_count": count,
        "unique_day_count": unique_days,
        "unique_session_count": len(set(session_labels)),
        "unique_symbol_count": unique_symbols,
        "unique_strategy_count": unique_strategies,
        "unique_regime_count": unique_regimes,
        "effective_sample_size": round(effective_n, 6),
        "serial_effective_sample_size": round(serial_ess, 6),
        "lag_one_autocorrelation": round(lag_one, 8),
        "cluster_effective_sample_size": {key: round(value, 6) for key, value in cluster_effective.items()},
        "intracluster_correlation": {key: round(value, 8) for key, value in intracluster.items()},
        "mean_post_cost_pnl_delta": round(mean_pnl, 8),
        "mean_post_cost_return_bps": round(mean_return, 8),
        "conservative_standard_error_post_cost_pnl_delta": round(conservative_pnl_se, 8),
        "conservative_standard_error_post_cost_return_bps": round(conservative_return_se, 8),
        "clustered_lower_confidence_bound_95_post_cost_pnl_delta": round(clustered_pnl_lcb, 8),
        "clustered_lower_confidence_bound_95_post_cost_return_bps": round(clustered_return_lcb, 8),
        "block_bootstrap_lower_confidence_bound_95_post_cost_pnl_delta": round(bootstrap_pnl_lcb, 8) if bootstrap_pnl_lcb is not None else None,
        "block_bootstrap_lower_confidence_bound_95_post_cost_return_bps": round(bootstrap_return_lcb, 8) if bootstrap_return_lcb is not None else None,
        "promotion_lower_confidence_bound_95_post_cost_pnl_delta": round(final_pnl_lcb, 8) if final_pnl_lcb is not None else None,
        "promotion_lower_confidence_bound_95_post_cost_return_bps": round(final_return_lcb, 8) if final_return_lcb is not None else None,
        "one_sided_positive_expectancy_p_value": round(max(0.0, min(one_sided_p, 1.0)), 10),
        "deflated_sharpe": dsr,
        "promotion_evidence_sufficient": evidence_sufficient,
        "positive_clustered_lower_confidence_bound_95": positive_lcb,
        "blockers": blockers,
        "thresholds": {
            "minimum_samples": max(int(minimum_samples), 1),
            "minimum_days": max(int(minimum_days), 1),
            "minimum_symbols": max(int(minimum_symbols), 1),
            "minimum_effective_samples": max(float(minimum_effective_samples), 1.0),
        },
        "policy": "promotion uses the most conservative clustered and day-block-bootstrap confidence bound; raw trade count is never treated as independent evidence",
    }


def benjamini_hochberg(p_values: Mapping[str, float], *, alpha: float = 0.05) -> dict[str, Any]:
    cleaned = {
        str(key): max(0.0, min(_safe_float(value, 1.0), 1.0))
        for key, value in p_values.items()
        if str(key).strip()
    }
    ordered = sorted(cleaned.items(), key=lambda item: (item[1], item[0]))
    count = len(ordered)
    adjusted: dict[str, float] = {}
    running = 1.0
    for rank, (key, value) in reversed(list(enumerate(ordered, start=1))):
        running = min(running, value * count / rank)
        adjusted[key] = max(0.0, min(running, 1.0))
    rows = [
        {
            "hypothesis_id": key,
            "p_value": round(value, 10),
            "q_value": round(adjusted[key], 10),
            "passes_fdr": adjusted[key] <= max(float(alpha), 0.0),
        }
        for key, value in ordered
    ]
    return {
        "method": "benjamini_hochberg_fdr",
        "alpha": float(alpha),
        "hypothesis_count": count,
        "rows": rows,
        "passing_hypotheses": [row["hypothesis_id"] for row in rows if row["passes_fdr"]],
    }


def probability_of_backtest_overfitting(
    strategy_period_returns: Mapping[str, Sequence[float]],
    *,
    minimum_periods: int = 8,
    max_combinations: int = 70,
) -> dict[str, Any]:
    series = {
        str(key): [float(value) for value in values]
        for key, values in strategy_period_returns.items()
        if str(key).strip() and isinstance(values, Sequence)
    }
    if len(series) < 2:
        return {"available": False, "pbo": None, "blockers": ["minimum_strategy_count_pending"]}
    period_count = min((len(values) for values in series.values()), default=0)
    if period_count < max(int(minimum_periods), 4):
        return {"available": False, "pbo": None, "blockers": ["minimum_period_count_pending"]}
    usable_periods = period_count if period_count % 2 == 0 else period_count - 1
    half = usable_periods // 2
    splits = list(combinations(range(usable_periods), half))[: max(int(max_combinations), 1)]
    logits: list[float] = []
    names = sorted(series)
    for in_sample_indexes in splits:
        in_set = set(in_sample_indexes)
        out_indexes = [index for index in range(usable_periods) if index not in in_set]
        in_scores = {name: _mean([series[name][index] for index in in_sample_indexes]) for name in names}
        selected = max(names, key=lambda name: (in_scores[name], name))
        out_scores = sorted(((_mean([series[name][index] for index in out_indexes]), name) for name in names))
        out_rank = next(index for index, (_score, name) in enumerate(out_scores, start=1) if name == selected)
        percentile = (out_rank - 0.5) / len(names)
        logits.append(math.log(percentile / max(1.0 - percentile, 1e-12)))
    pbo = sum(1 for value in logits if value <= 0.0) / max(len(logits), 1)
    return {
        "available": True,
        "pbo": round(pbo, 8),
        "split_count": len(logits),
        "strategy_count": len(names),
        "period_count": usable_periods,
        "passes": pbo <= 0.20,
        "policy": "CSCV-style in-sample winner ranking on held-out periods; promotion ceiling is PBO <= 0.20",
    }


def risk_of_ruin_statistics(
    daily_pnl: Sequence[float],
    *,
    initial_capital: float = 10_000.0,
    ruin_equity_fraction: float = 0.50,
    drawdown_budget_fraction: float = 0.10,
    horizon_days: int = 252,
    iterations: int = 2_000,
    block_days: int = 5,
    minimum_days: int = 30,
    maximum_ruin_probability: float = 0.01,
    maximum_drawdown_breach_probability: float = 0.05,
    seed_material: str = "profitability-risk-of-ruin-v1",
) -> dict[str, Any]:
    values = [_safe_float(value, float("nan")) for value in daily_pnl]
    values = [value for value in values if math.isfinite(value)]
    required_days = max(int(minimum_days), 2)
    capital = max(_safe_float(initial_capital, 0.0), 0.0)
    if len(values) < required_days or capital <= 0.0:
        blockers = []
        if len(values) < required_days:
            blockers.append("minimum_independent_days_pending")
        if capital <= 0.0:
            blockers.append("initial_capital_not_positive")
        return {
            "available": False,
            "passes": False,
            "day_count": len(values),
            "ruin_probability": None,
            "drawdown_breach_probability": None,
            "p99_max_drawdown_fraction": None,
            "blockers": blockers,
            "thresholds": {
                "minimum_days": required_days,
                "maximum_ruin_probability": max(float(maximum_ruin_probability), 0.0),
                "maximum_drawdown_breach_probability": max(float(maximum_drawdown_breach_probability), 0.0),
            },
        }

    horizon = max(int(horizon_days), 1)
    run_count = max(int(iterations), 200)
    block = max(1, min(int(block_days), len(values)))
    ruin_floor = capital * max(0.0, min(float(ruin_equity_fraction), 1.0))
    drawdown_budget = max(0.0, min(float(drawdown_budget_fraction), 1.0))
    seed_payload = json.dumps(
        {
            "seed_material": str(seed_material),
            "daily_pnl": values,
            "horizon_days": horizon,
            "block_days": block,
        },
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )
    seed = int(hashlib.sha256(seed_payload.encode("utf-8")).hexdigest()[:16], 16)
    rng = random.Random(seed)
    max_drawdowns: list[float] = []
    ruin_count = 0
    breach_count = 0
    for _ in range(run_count):
        sampled: list[float] = []
        while len(sampled) < horizon:
            start = rng.randrange(len(values))
            sampled.extend(values[(start + offset) % len(values)] for offset in range(block))
        equity = capital
        peak = capital
        path_max_drawdown = 0.0
        ruined = False
        for pnl in sampled[:horizon]:
            equity += pnl
            peak = max(peak, equity)
            path_max_drawdown = max(path_max_drawdown, (peak - equity) / max(peak, 1e-12))
            if equity <= ruin_floor:
                ruined = True
        max_drawdowns.append(path_max_drawdown)
        ruin_count += int(ruined)
        breach_count += int(path_max_drawdown > drawdown_budget)

    ruin_probability = ruin_count / run_count
    breach_probability = breach_count / run_count
    p99_drawdown = _percentile(max_drawdowns, 0.99)
    max_ruin = max(float(maximum_ruin_probability), 0.0)
    max_breach = max(float(maximum_drawdown_breach_probability), 0.0)
    passes = bool(ruin_probability <= max_ruin and breach_probability <= max_breach)
    return {
        "available": True,
        "passes": passes,
        "day_count": len(values),
        "simulation_count": run_count,
        "horizon_days": horizon,
        "block_days": block,
        "initial_capital": round(capital, 6),
        "ruin_equity_fraction": round(max(0.0, min(float(ruin_equity_fraction), 1.0)), 6),
        "drawdown_budget_fraction": round(drawdown_budget, 6),
        "ruin_probability": round(ruin_probability, 8),
        "drawdown_breach_probability": round(breach_probability, 8),
        "p99_max_drawdown_fraction": round(p99_drawdown, 8) if p99_drawdown is not None else None,
        "blockers": [] if passes else [
            *(["ruin_probability_above_ceiling"] if ruin_probability > max_ruin else []),
            *(["drawdown_breach_probability_above_ceiling"] if breach_probability > max_breach else []),
        ],
        "thresholds": {
            "minimum_days": required_days,
            "maximum_ruin_probability": max_ruin,
            "maximum_drawdown_breach_probability": max_breach,
        },
        "policy": "deterministic moving-block bootstrap estimates capital-floor and drawdown-budget breach risk; insufficient history fails closed",
    }
