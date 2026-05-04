from __future__ import annotations

from typing import Any


def _num(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _clamp(value: Any, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, float(_num(value))))


def model_calibration_decay_score(
    *,
    calibration_error: Any,
    challenger_gap: Any,
    stress_replay_pass_rate: Any,
    leakage_risk: Any,
) -> float:
    error = _clamp(_num(calibration_error) / 0.20)
    challenger = _clamp(abs(_num(challenger_gap)) / 0.15)
    stress_fail = 1.0 - _clamp(stress_replay_pass_rate)
    leakage = _clamp(leakage_risk)
    return _clamp(0.35 * error + 0.20 * challenger + 0.25 * stress_fail + 0.20 * leakage)


def transaction_cost_pressure_score(
    *,
    spread_bps: Any,
    slippage_bps: Any,
    fill_rate: Any,
    queue_adverse_selection: Any,
) -> float:
    spread = _clamp(_num(spread_bps) / 60.0)
    slippage = _clamp(_num(slippage_bps) / 80.0)
    fill_gap = 1.0 - _clamp(fill_rate)
    adverse = _clamp(queue_adverse_selection)
    return _clamp(0.25 * spread + 0.35 * slippage + 0.20 * fill_gap + 0.20 * adverse)


def portfolio_conflict_score(
    *,
    gross_exposure: Any,
    net_exposure: Any,
    hedge_ratio_error: Any,
    sleeve_correlation: Any,
) -> float:
    gross = abs(_num(gross_exposure))
    net = abs(_num(net_exposure))
    hidden_overlap = _clamp((gross - net) / max(gross, 1.0))
    hedge_error = _clamp(abs(_num(hedge_ratio_error)))
    correlation = _clamp(abs(_num(sleeve_correlation)))
    leverage = _clamp(gross / 3.0)
    return _clamp(0.30 * hidden_overlap + 0.25 * hedge_error + 0.25 * correlation + 0.20 * leverage)


def event_risk_score(
    *,
    surprise_magnitude: Any,
    time_to_event_minutes: Any,
    source_confidence: Any,
    historical_impact: Any,
) -> float:
    surprise = _clamp(abs(_num(surprise_magnitude)))
    time_to_event = max(_num(time_to_event_minutes, 9999.0), 0.0)
    proximity = _clamp(1.0 - (time_to_event / 1440.0))
    confidence = _clamp(source_confidence)
    impact = _clamp(abs(_num(historical_impact)))
    return _clamp(0.30 * surprise + 0.25 * proximity + 0.20 * confidence + 0.25 * impact)


def feature_confidence_score(
    *,
    missing_rate: Any,
    stale_rate: Any,
    source_disagreement: Any,
    label_confidence: Any,
) -> float:
    missing_penalty = _clamp(missing_rate)
    stale_penalty = _clamp(stale_rate)
    disagreement = _clamp(source_disagreement)
    confidence = _clamp(label_confidence)
    return _clamp(confidence * (1.0 - 0.35 * missing_penalty - 0.30 * stale_penalty - 0.35 * disagreement))


def liquidity_regime_stress_score(
    *,
    spread_bps: Any,
    quote_fade_rate: Any,
    auction_imbalance: Any,
    halt_reopen_flag: Any,
) -> float:
    spread = _clamp(_num(spread_bps) / 80.0)
    fade = _clamp(quote_fade_rate)
    imbalance = _clamp(abs(_num(auction_imbalance)))
    halt = 1.0 if bool(halt_reopen_flag) else 0.0
    return _clamp(0.30 * spread + 0.30 * fade + 0.25 * imbalance + 0.15 * halt)


def governor_pressure_score(
    *,
    cpu_pressure: Any,
    memory_pressure: Any,
    backlog_ratio: Any,
    halt_pressure: Any,
    storage_pressure: Any,
) -> float:
    cpu = _clamp(cpu_pressure)
    memory = _clamp(memory_pressure)
    backlog = _clamp(backlog_ratio)
    halt = _clamp(halt_pressure)
    storage = _clamp(storage_pressure)
    return _clamp(0.22 * cpu + 0.22 * memory + 0.22 * backlog + 0.20 * halt + 0.14 * storage)


def adaptive_learning_priority_score(
    *,
    uncertainty: Any,
    drift: Any,
    observation_value: Any,
    runtime_pressure: Any,
) -> float:
    uncertainty_score = _clamp(uncertainty)
    drift_score = _clamp(drift)
    value_score = _clamp(observation_value)
    pressure_penalty = _clamp(runtime_pressure)
    return _clamp((0.35 * uncertainty_score + 0.30 * drift_score + 0.35 * value_score) * (1.0 - 0.55 * pressure_penalty))


def catastrophic_forgetting_risk_score(
    *,
    legacy_replay_drop: Any,
    new_slice_gain: Any,
    rehearsal_coverage: Any,
    regime_distance: Any,
) -> float:
    replay_drop = _clamp(abs(_num(legacy_replay_drop)) / 0.20)
    shortcut_gain = _clamp(max(_num(new_slice_gain), 0.0) / 0.25)
    coverage_gap = 1.0 - _clamp(rehearsal_coverage)
    distance = _clamp(regime_distance)
    return _clamp(0.38 * replay_drop + 0.18 * shortcut_gain + 0.24 * coverage_gap + 0.20 * distance)


def simulation_to_reality_gap_score(
    *,
    paper_live_slippage_gap: Any,
    replay_fill_error: Any,
    synthetic_stress_error: Any,
    live_context_coverage: Any,
) -> float:
    slippage_gap = _clamp(abs(_num(paper_live_slippage_gap)) / 75.0)
    fill_error = _clamp(replay_fill_error)
    stress_error = _clamp(synthetic_stress_error)
    coverage_gap = 1.0 - _clamp(live_context_coverage)
    return _clamp(0.30 * slippage_gap + 0.25 * fill_error + 0.25 * stress_error + 0.20 * coverage_gap)


def causal_representation_stability_score(
    *,
    intervention_consistency: Any,
    feature_overlap_leakage: Any,
    source_disagreement: Any,
    regime_transfer_success: Any,
) -> float:
    consistency = _clamp(intervention_consistency)
    leakage_penalty = _clamp(feature_overlap_leakage)
    disagreement_penalty = _clamp(source_disagreement)
    transfer = _clamp(regime_transfer_success)
    return _clamp((0.55 * consistency + 0.45 * transfer) * (1.0 - 0.50 * leakage_penalty - 0.35 * disagreement_penalty))
