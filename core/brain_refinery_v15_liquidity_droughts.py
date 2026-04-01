import numpy as np

from indicator_bot_common import train_price_indicator_bot, train_runtime_indicator_bot
from runtime_training_common import (
    feature_ema,
    feature_std,
    future_max_drawdown,
    future_realized_vol,
    future_return,
    observation_feature,
    price_change,
)


def ema(x, span):
    alpha = 2 / (span + 1)
    out = np.zeros_like(x)
    out[0] = x[0]
    for i in range(1, len(x)):
        out[i] = alpha * x[i] + (1 - alpha) * out[i - 1]
    return out


def rsi(prices, period=14):
    deltas = np.diff(prices, prepend=prices[0])
    gains = np.maximum(deltas, 0)
    losses = np.maximum(-deltas, 0)
    avg_gain = ema(gains, period)
    avg_loss = ema(losses, period) + 1e-8
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def rolling_std(x, window):
    out = np.zeros_like(x)
    for i in range(len(x)):
        start = max(0, i - window + 1)
        out[i] = np.std(x[start : i + 1])
    return out


def simulate_liquidity_droughts(n=5000):
    prices = np.zeros(n)
    prices[0] = 100.0
    drift = 0.0001
    vol = 0.005
    for i in range(1, n):
        if i % 900 == 0:
            jump = np.random.uniform(-0.1, 0.1)
            prices[i] = max(0.1, prices[i - 1] * (1 + jump))
            continue
        ret = drift + vol * np.random.randn()
        prices[i] = prices[i - 1] * np.exp(ret)
    return prices


def build_features(prices):
    returns = np.log(prices[1:] / prices[:-1])
    returns = np.concatenate([[0.0], returns])
    sma = np.convolve(prices, np.ones(10) / 10, mode="same")
    ema10 = ema(prices, 10)
    rsi14 = rsi(prices, 14)
    vol10 = rolling_std(returns, 10)
    return np.stack([returns, sma, ema10, rsi14, vol10], axis=1)


def _clip01(value):
    return float(np.clip(value, 0.0, 1.0))


def _liquidity_drought_signal(obs):
    spread = _clip01(abs(observation_feature(obs, "spread_bps")) / 35.0)
    depth = 1.0 - _clip01(observation_feature(obs, "queue_depth") / 8.0)
    fill_risk = _clip01(abs(observation_feature(obs, "lag_expected_fill_delta_bps")) / 18.0)
    slip_risk = _clip01(abs(observation_feature(obs, "lag_slippage_bps")) / 20.0)
    rel_vol_drought = 1.0 - _clip01(observation_feature(obs, "market_micro_relative_volume_norm"))
    latency = _clip01(observation_feature(obs, "data_quality_market_data_latency_norm"))
    return _clip01(
        (0.22 * spread)
        + (0.18 * depth)
        + (0.18 * fill_risk)
        + (0.16 * slip_risk)
        + (0.16 * rel_vol_drought)
        + (0.10 * latency)
    )


def _runtime_feature_vector(sequence, idx):
    obs = sequence[idx]
    return np.asarray(
        [
            observation_feature(obs, "pct_from_close"),
            observation_feature(obs, "mom_5m"),
            observation_feature(obs, "vol_30m"),
            observation_feature(obs, "range_pos"),
            observation_feature(obs, "spread_bps"),
            observation_feature(obs, "queue_depth"),
            observation_feature(obs, "market_data_latency_ms"),
            observation_feature(obs, "data_quality_quote_agreement_norm"),
            observation_feature(obs, "data_quality_quote_deviation_norm"),
            observation_feature(obs, "data_quality_market_data_latency_norm"),
            observation_feature(obs, "lag_expected_fill_delta_bps"),
            observation_feature(obs, "lag_slippage_bps"),
            observation_feature(obs, "lag_latency_ms"),
            observation_feature(obs, "market_micro_relative_volume_norm"),
            observation_feature(obs, "market_micro_order_flow_imbalance_norm"),
            observation_feature(obs, "market_micro_short_pressure_norm"),
            observation_feature(obs, "market_micro_block_trade_norm"),
            observation_feature(obs, "market_micro_gap_continuation_norm"),
            observation_feature(obs, "market_micro_reversal_risk_norm"),
            observation_feature(obs, "market_micro_range_expansion_norm"),
            observation_feature(obs, "options_unusual_flow_norm"),
            observation_feature(obs, "options_specialist_vote"),
            observation_feature(obs, "futures_specialist_vote"),
            observation_feature(obs, "behavior_prior"),
            price_change(sequence, idx, 1),
            price_change(sequence, idx, 3),
            feature_std(sequence, idx, "pct_from_close", 4),
            feature_ema(sequence, idx, "spread_bps", 4),
            feature_ema(sequence, idx, "market_micro_relative_volume_norm", 4),
        ],
        dtype=np.float32,
    )


def _runtime_sample_filter(sequence, idx, horizon):
    obs = sequence[idx]
    return (
        observation_feature(obs, "data_quality_quote_agreement_norm", 1.0) >= 0.72
        and observation_feature(obs, "data_quality_quote_deviation_norm", 0.0) <= 0.42
        and max(
            _liquidity_drought_signal(obs),
            observation_feature(obs, "market_micro_reversal_risk_norm"),
            observation_feature(obs, "market_micro_gap_continuation_norm"),
        )
        >= 0.14
    )


def _runtime_confidence(sequence, idx, horizon):
    obs = sequence[idx]
    quote_quality = _clip01(
        0.60 * observation_feature(obs, "data_quality_quote_agreement_norm", 1.0)
        + 0.25 * (1.0 - observation_feature(obs, "data_quality_quote_deviation_norm", 0.0))
        + 0.15 * (1.0 - observation_feature(obs, "data_quality_market_data_latency_norm", 0.0))
    )
    return (
        (0.34 * _liquidity_drought_signal(obs))
        + (0.18 * observation_feature(obs, "market_micro_reversal_risk_norm"))
        + (0.14 * observation_feature(obs, "market_micro_gap_continuation_norm"))
        + (0.12 * observation_feature(obs, "market_micro_range_expansion_norm"))
        + (0.10 * observation_feature(obs, "options_unusual_flow_norm"))
        + (0.12 * quote_quality)
    )


def _drought_direction(obs):
    range_bias = (observation_feature(obs, "range_pos") - 0.5) * 2.0
    flow_bias = (observation_feature(obs, "market_micro_order_flow_imbalance_norm") - 0.5) * 2.0
    continuation_bias = observation_feature(obs, "market_micro_gap_continuation_norm") - observation_feature(obs, "market_micro_reversal_risk_norm")
    specialist_bias = (
        0.55 * observation_feature(obs, "futures_specialist_vote")
        + 0.45 * observation_feature(obs, "options_specialist_vote")
    )
    return float(
        (0.28 * observation_feature(obs, "behavior_prior"))
        + (0.24 * continuation_bias)
        + (0.22 * flow_bias)
        + (0.16 * specialist_bias)
        + (0.10 * range_bias)
    )


def _runtime_liquidity_label(sequence, idx, horizon):
    obs = sequence[idx]
    drought_signal = _liquidity_drought_signal(obs)
    direction = _drought_direction(obs)
    if drought_signal < 0.16 or abs(direction) < 0.08:
        return None

    expected_up = direction >= 0.0
    fwd_ret = future_return(sequence, idx, horizon)
    realized_vol = future_realized_vol(sequence, idx, horizon)
    drawdown = abs(future_max_drawdown(sequence, idx, horizon))
    signed_ret = fwd_ret if expected_up else -fwd_ret
    move_threshold = max(0.00035, 0.00095 - (0.00035 * drought_signal))
    if abs(fwd_ret) < move_threshold and realized_vol < 0.018:
        return None

    quote_quality = _clip01(
        0.55 * observation_feature(obs, "data_quality_quote_agreement_norm", 1.0)
        + 0.45 * (1.0 - observation_feature(obs, "data_quality_quote_deviation_norm", 0.0))
    )
    success_score = (
        signed_ret
        + (0.0007 * drought_signal)
        + (0.0002 * quote_quality)
        - (0.10 * realized_vol)
        - (0.40 * drawdown)
    )
    failure_score = ((-signed_ret) + (0.08 * realized_vol) + (0.22 * drawdown))
    if success_score >= 0.00018:
        return 1.0 if expected_up else 0.0
    if failure_score >= 0.00040:
        return 0.0 if expected_up else 1.0
    return None


def _train_synthetic():
    return train_price_indicator_bot(
        run_tag="brain_refinery_v15_liquidity_droughts",
        feature_names=["returns", "sma10", "ema10", "rsi14", "vol10"],
        feature_builder=build_features,
        price_simulator=simulate_liquidity_droughts,
    )


def train_brain():
    return train_runtime_indicator_bot(
        run_tag="brain_refinery_v15_liquidity_droughts",
        feature_names=[
            "pct_from_close",
            "mom_5m",
            "vol_30m",
            "range_pos",
            "spread_bps",
            "queue_depth",
            "market_data_latency_ms",
            "data_quality_quote_agreement_norm",
            "data_quality_quote_deviation_norm",
            "data_quality_market_data_latency_norm",
            "lag_expected_fill_delta_bps",
            "lag_slippage_bps",
            "lag_latency_ms",
            "market_micro_relative_volume_norm",
            "market_micro_order_flow_imbalance_norm",
            "market_micro_short_pressure_norm",
            "market_micro_block_trade_norm",
            "market_micro_gap_continuation_norm",
            "market_micro_reversal_risk_norm",
            "market_micro_range_expansion_norm",
            "options_unusual_flow_norm",
            "options_specialist_vote",
            "futures_specialist_vote",
            "behavior_prior",
            "ret_1",
            "ret_3",
            "pct_from_close_std_4",
            "spread_bps_ema_4",
            "market_micro_relative_volume_ema_4",
        ],
        runtime_feature_builder=_runtime_feature_vector,
        runtime_label_builder=_runtime_liquidity_label,
        sample_filter=_runtime_sample_filter,
        confidence_builder=_runtime_confidence,
        min_confidence=0.32,
        lookback_days=30,
        window=20,
        horizon=4,
        min_samples=96,
        min_sequences=6,
        acted_prob_threshold=0.60,
        fallback_trainer=_train_synthetic,
        allow_fallback_on_insufficient_data=False,
        max_best_val_loss=0.6925,
        max_final_val_loss=0.7050,
        min_long_precision=0.52,
        min_short_precision=0.48,
        require_both_sides_precision=True,
        min_acted_accuracy=0.53,
        min_accuracy_lift_over_majority=0.02,
        min_precision_balance_score=0.35,
    )


if __name__ == "__main__":
    train_brain()
