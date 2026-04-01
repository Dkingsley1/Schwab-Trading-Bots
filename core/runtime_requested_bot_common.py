import numpy as np

from runtime_training_common import (
    future_max_drawdown,
    future_realized_vol,
    future_return,
    observation_feature,
)


def clip01(value) -> float:
    return float(np.clip(float(value), 0.0, 1.0))


def centered01(value: float) -> float:
    return float((float(value) - 0.5) * 2.0)


def quote_quality(obs) -> float:
    return clip01(
        (0.65 * observation_feature(obs, "data_quality_quote_agreement_norm", 1.0))
        + (0.35 * (1.0 - observation_feature(obs, "data_quality_quote_deviation_norm", 0.0)))
    )


def base_runtime_gate(
    obs,
    *,
    min_quote_agreement: float = 0.76,
    max_quote_deviation: float = 0.30,
    max_spread_bps: float = 35.0,
    min_queue_depth: float = 0.0,
    max_latency_ms: float | None = 3500.0,
) -> bool:
    if observation_feature(obs, "data_quality_quote_agreement_norm", 1.0) < min_quote_agreement:
        return False
    if observation_feature(obs, "data_quality_quote_deviation_norm", 0.0) > max_quote_deviation:
        return False
    if abs(observation_feature(obs, "spread_bps", 0.0)) > max_spread_bps:
        return False
    if observation_feature(obs, "queue_depth", 0.0) < min_queue_depth:
        return False
    if max_latency_ms is not None and observation_feature(obs, "market_data_latency_ms", 0.0) > max_latency_ms:
        return False
    return True


def directional_outcome_label(
    sequence,
    idx: int,
    horizon: int,
    *,
    bias: float,
    support: float,
    min_support: float,
    min_abs_bias: float,
    move_base: float,
    move_scale: float = 0.00045,
    move_floor: float = 0.00035,
    realized_vol_gate: float = 0.018,
    drawdown_gate: float = 0.012,
    support_bonus: float = 0.0008,
    quote_bonus_weight: float = 0.00025,
    realized_vol_penalty: float = 0.16,
    drawdown_penalty: float = 0.26,
    success_floor: float = 0.00035,
    failure_floor: float = 0.00050,
):
    if support < min_support or abs(bias) < min_abs_bias:
        return None

    obs = sequence[idx]
    expected_up = bias >= 0.0
    fwd_ret = future_return(sequence, idx, horizon)
    realized = future_realized_vol(sequence, idx, horizon)
    drawdown = abs(future_max_drawdown(sequence, idx, horizon))
    signed_ret = fwd_ret if expected_up else -fwd_ret
    move_threshold = max(float(move_floor), float(move_base) - (float(move_scale) * float(support)))
    if abs(fwd_ret) < move_threshold and realized < realized_vol_gate and drawdown < drawdown_gate:
        return None

    success_score = (
        signed_ret
        + (float(support_bonus) * float(support))
        + (float(quote_bonus_weight) * quote_quality(obs))
        - (float(realized_vol_penalty) * realized)
        - (float(drawdown_penalty) * drawdown)
    )
    failure_score = (
        (-signed_ret)
        + ((float(realized_vol_penalty) * 0.72) * realized)
        + ((float(drawdown_penalty) * 0.72) * drawdown)
    )
    if success_score >= float(success_floor):
        return 1.0 if expected_up else 0.0
    if failure_score >= float(failure_floor):
        return 0.0 if expected_up else 1.0
    return None
