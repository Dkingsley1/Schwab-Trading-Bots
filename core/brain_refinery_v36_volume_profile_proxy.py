import numpy as np

from indicator_bot_common import rolling_std, train_indicator_bot, train_runtime_indicator_bot
from runtime_training_common import (
    feature_ema,
    feature_std,
    future_max_drawdown,
    future_realized_vol,
    future_return,
    observation_feature,
    price_change,
)


def profile_proxy(close, volume, window=60, bins=24):
    poc = np.zeros_like(close)
    va_width = np.zeros_like(close)

    for i in range(len(close)):
        start = max(0, i - window + 1)
        p = close[start : i + 1]
        v = volume[start : i + 1]
        pmin, pmax = np.min(p), np.max(p)
        if pmax <= pmin:
            poc[i] = close[i]
            va_width[i] = 0.0
            continue
        edges = np.linspace(pmin, pmax, bins + 1)
        idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
        vol_by_bin = np.zeros(bins)
        for j, b in enumerate(idx):
            vol_by_bin[b] += v[j]
        k = int(np.argmax(vol_by_bin))
        poc[i] = 0.5 * (edges[k] + edges[k + 1])
        va_width[i] = (edges[min(k + 1, bins)] - edges[max(k - 1, 0)]) / max(close[i], 1e-8)

    return poc, va_width


def build_features(panel):
    c = panel["close"]
    v = panel["volume"]
    r = panel["ret"]

    poc, va_width = profile_proxy(c, v, window=60, bins=24)
    dist_poc = (c - poc) / (poc + 1e-8)
    pull = np.abs(dist_poc) / (rolling_std(r, 20) + 1e-8)
    vol_z = (v - np.mean(v)) / (np.std(v) + 1e-8)

    return np.stack([r, dist_poc, va_width, pull, vol_z], axis=1)


def _clip01(value):
    return float(np.clip(value, 0.0, 1.0))


def _volume_profile_signal(obs):
    return _clip01(
        max(
            observation_feature(obs, "market_micro_relative_volume_norm"),
            observation_feature(obs, "market_micro_order_flow_imbalance_norm"),
            observation_feature(obs, "market_micro_range_expansion_norm"),
            observation_feature(obs, "market_micro_trend_persistence_norm"),
            observation_feature(obs, "market_micro_block_trade_norm"),
            abs(observation_feature(obs, "range_pos") - 0.5) * 2.0,
        )
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
            observation_feature(obs, "market_micro_opening_auction_norm"),
            observation_feature(obs, "market_micro_closing_auction_norm"),
            observation_feature(obs, "market_micro_relative_volume_norm"),
            observation_feature(obs, "market_micro_order_flow_imbalance_norm"),
            observation_feature(obs, "market_micro_gap_continuation_norm"),
            observation_feature(obs, "market_micro_reversal_risk_norm"),
            observation_feature(obs, "market_micro_trend_persistence_norm"),
            observation_feature(obs, "market_micro_range_expansion_norm"),
            observation_feature(obs, "market_micro_block_trade_norm"),
            observation_feature(obs, "options_unusual_flow_norm"),
            observation_feature(obs, "options_specialist_vote"),
            observation_feature(obs, "futures_specialist_vote"),
            observation_feature(obs, "data_quality_quote_agreement_norm"),
            observation_feature(obs, "data_quality_quote_deviation_norm"),
            price_change(sequence, idx, 1),
            price_change(sequence, idx, 3),
            price_change(sequence, idx, 6),
            feature_std(sequence, idx, "pct_from_close", 6),
            feature_ema(sequence, idx, "market_micro_relative_volume_norm", 4),
            feature_ema(sequence, idx, "market_micro_order_flow_imbalance_norm", 4),
        ],
        dtype=np.float32,
    )


def _runtime_sample_filter(sequence, idx, horizon):
    obs = sequence[idx]
    return (
        observation_feature(obs, "data_quality_quote_agreement_norm", 1.0) >= 0.76
        and observation_feature(obs, "data_quality_quote_deviation_norm", 0.0) <= 0.35
        and _volume_profile_signal(obs) >= 0.20
    )


def _runtime_confidence(sequence, idx, horizon):
    obs = sequence[idx]
    quote_quality = _clip01(
        0.60 * observation_feature(obs, "data_quality_quote_agreement_norm", 1.0)
        + 0.40 * (1.0 - observation_feature(obs, "data_quality_quote_deviation_norm", 0.0))
    )
    return (
        (0.30 * _volume_profile_signal(obs))
        + (0.18 * observation_feature(obs, "market_micro_gap_continuation_norm"))
        + (0.16 * observation_feature(obs, "market_micro_reversal_risk_norm"))
        + (0.14 * observation_feature(obs, "market_micro_trend_persistence_norm"))
        + (0.10 * observation_feature(obs, "market_micro_block_trade_norm"))
        + (0.12 * quote_quality)
    )


def _profile_direction(obs):
    range_bias = (observation_feature(obs, "range_pos") - 0.5) * 2.0
    flow_bias = (observation_feature(obs, "market_micro_order_flow_imbalance_norm") - 0.5) * 2.0
    auction_bias = observation_feature(obs, "market_micro_gap_continuation_norm") - observation_feature(obs, "market_micro_reversal_risk_norm")
    specialist_bias = (
        0.60 * observation_feature(obs, "futures_specialist_vote")
        + 0.40 * observation_feature(obs, "options_specialist_vote")
    )
    persistence_bias = (observation_feature(obs, "market_micro_trend_persistence_norm") - 0.5) * 2.0
    return float(
        (0.28 * range_bias)
        + (0.28 * flow_bias)
        + (0.18 * auction_bias)
        + (0.16 * specialist_bias)
        + (0.10 * persistence_bias)
    )


def _runtime_profile_label(sequence, idx, horizon):
    obs = sequence[idx]
    profile_signal = _volume_profile_signal(obs)
    direction = _profile_direction(obs)
    if profile_signal < 0.18 or abs(direction) < 0.10:
        return None

    expected_up = direction >= 0.0
    fwd_ret = future_return(sequence, idx, horizon)
    realized_vol = future_realized_vol(sequence, idx, horizon)
    drawdown = abs(future_max_drawdown(sequence, idx, horizon))
    signed_ret = fwd_ret if expected_up else -fwd_ret
    move_threshold = max(0.00035, 0.00085 - (0.00030 * profile_signal))
    if abs(fwd_ret) < move_threshold and realized_vol < 0.016:
        return None

    success_score = (
        signed_ret
        + (0.0008 * profile_signal)
        + (0.0003 * observation_feature(obs, "market_micro_trend_persistence_norm"))
        - (0.10 * realized_vol)
        - (0.45 * drawdown)
    )
    failure_score = ((-signed_ret) + (0.08 * realized_vol) + (0.25 * drawdown))
    if success_score >= 0.00015:
        return 1.0 if expected_up else 0.0
    if failure_score >= 0.00035:
        return 0.0 if expected_up else 1.0
    return None


def _train_synthetic():
    return train_indicator_bot(
        run_tag="brain_refinery_v36_volume_profile_proxy",
        feature_names=["ret", "dist_poc", "value_area_width", "poc_pull", "volume_z"],
        feature_builder=build_features,
    )


def train_brain():
    return train_runtime_indicator_bot(
        run_tag="brain_refinery_v36_volume_profile_proxy",
        feature_names=[
            "pct_from_close",
            "mom_5m",
            "vol_30m",
            "range_pos",
            "spread_bps",
            "queue_depth",
            "market_micro_opening_auction_norm",
            "market_micro_closing_auction_norm",
            "market_micro_relative_volume_norm",
            "market_micro_order_flow_imbalance_norm",
            "market_micro_gap_continuation_norm",
            "market_micro_reversal_risk_norm",
            "market_micro_trend_persistence_norm",
            "market_micro_range_expansion_norm",
            "market_micro_block_trade_norm",
            "options_unusual_flow_norm",
            "options_specialist_vote",
            "futures_specialist_vote",
            "data_quality_quote_agreement_norm",
            "data_quality_quote_deviation_norm",
            "ret_1",
            "ret_3",
            "ret_6",
            "pct_from_close_std_6",
            "market_micro_relative_volume_ema_4",
            "market_micro_order_flow_imbalance_ema_4",
        ],
        runtime_feature_builder=_runtime_feature_vector,
        runtime_label_builder=_runtime_profile_label,
        sample_filter=_runtime_sample_filter,
        confidence_builder=_runtime_confidence,
        min_confidence=0.36,
        lookback_days=30,
        window=24,
        horizon=6,
        min_samples=224,
        min_sequences=6,
        acted_prob_threshold=0.60,
        fallback_trainer=_train_synthetic,
        allow_fallback_on_insufficient_data=False,
        max_best_val_loss=0.6925,
        max_final_val_loss=0.7050,
        min_long_precision=0.52,
        min_short_precision=0.52,
        require_both_sides_precision=True,
        min_acted_accuracy=0.53,
        min_accuracy_lift_over_majority=0.02,
        min_precision_balance_score=0.35,
    )


if __name__ == "__main__":
    train_brain()
