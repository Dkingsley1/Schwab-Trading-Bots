import numpy as np

from indicator_bot_common import bollinger, ema, rolling_std, train_indicator_bot, train_runtime_indicator_bot, vwap
from runtime_training_common import (
    feature_ema,
    feature_std,
    future_max_drawdown,
    future_realized_vol,
    future_return,
    observation_feature,
    price_change,
)

_SCALP_RUNTIME_MODES = [
    "shadow_aggressive_equities",
    "shadow_intraday_aggressive_equities",
]
_SCALP_SYMBOLS = [
    "SPY",
    "QQQ",
    "DIA",
    "IWM",
    "NVDA",
    "AAPL",
    "MSFT",
    "AMZN",
    "META",
    "TSLA",
    "AMD",
    "AVGO",
    "GOOGL",
    "SMH",
]


def build_features(panel):
    c = panel["close"]
    v = panel["volume"]
    r = panel["ret"]

    ret1 = r
    ret5 = ema(r, 5)
    ret15 = ema(r, 15)
    _, mid, _ = bollinger(c, window=20, k=2.0)
    dist_mid = (c - mid) / (mid + 1e-8)
    vw = vwap(c, v, session=45)
    vdev = (c - vw) / (vw + 1e-8)
    noise = rolling_std(ret1, 20)

    return np.stack([ret1, ret5, ret15, dist_mid, vdev, noise], axis=1)


def _clip01(value):
    return float(np.clip(value, 0.0, 1.0))


def _centered01(value):
    return float((float(value) - 0.5) * 2.0)


def _quote_quality(obs):
    return _clip01(
        (0.65 * observation_feature(obs, "data_quality_quote_agreement_norm", 1.0))
        + (0.35 * (1.0 - observation_feature(obs, "data_quality_quote_deviation_norm", 0.0)))
    )


def _scalp_support(obs):
    opening_impulse = max(
        observation_feature(obs, "day_opening_auction_signal_norm"),
        observation_feature(obs, "market_micro_opening_auction_norm"),
        observation_feature(obs, "day_session_open_norm"),
        observation_feature(obs, "day_session_power_hour_norm"),
    )
    order_flow = _clip01(abs(_centered01(observation_feature(obs, "market_micro_order_flow_imbalance_norm", 0.5))))
    exec_ok = _clip01(1.0 - observation_feature(obs, "day_execution_cost_risk_norm"))
    return _clip01(
        (0.22 * opening_impulse)
        + (0.18 * observation_feature(obs, "market_micro_relative_volume_norm"))
        + (0.16 * observation_feature(obs, "market_micro_trend_persistence_norm"))
        + (0.16 * order_flow)
        + (0.14 * exec_ok)
        + (0.14 * _quote_quality(obs))
    )


def _scalp_bias(obs):
    return float(
        np.clip(
            (0.22 * observation_feature(obs, "behavior_prior"))
            + (0.18 * _centered01(observation_feature(obs, "market_micro_order_flow_imbalance_norm", 0.5)))
            + (0.16 * _centered01(observation_feature(obs, "futures_specialist_vote", 0.5)))
            + (0.16 * observation_feature(obs, "mom_5m") * 120.0)
            + (0.12 * observation_feature(obs, "pct_from_close") * 140.0)
            + (0.10 * _centered01(observation_feature(obs, "range_pos", 0.5)))
            + (0.06 * max(observation_feature(obs, "day_session_open_norm"), observation_feature(obs, "day_session_power_hour_norm"))),
            -1.0,
            1.0,
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
            observation_feature(obs, "market_data_latency_ms"),
            observation_feature(obs, "day_execution_cost_risk_norm"),
            observation_feature(obs, "day_opening_auction_signal_norm"),
            observation_feature(obs, "market_micro_opening_auction_norm"),
            observation_feature(obs, "day_session_open_norm"),
            observation_feature(obs, "day_session_midday_norm"),
            observation_feature(obs, "day_session_power_hour_norm"),
            observation_feature(obs, "market_micro_relative_volume_norm"),
            observation_feature(obs, "market_micro_trend_persistence_norm"),
            observation_feature(obs, "market_micro_order_flow_imbalance_norm"),
            observation_feature(obs, "data_quality_quote_agreement_norm", 1.0),
            observation_feature(obs, "data_quality_quote_deviation_norm"),
            observation_feature(obs, "behavior_prior"),
            observation_feature(obs, "futures_specialist_vote"),
            price_change(sequence, idx, 1),
            price_change(sequence, idx, 3),
            feature_std(sequence, idx, "pct_from_close", 4),
            feature_ema(sequence, idx, "pct_from_close", 3),
            feature_ema(sequence, idx, "spread_bps", 3),
            feature_std(sequence, idx, "spread_bps", 4),
            feature_ema(sequence, idx, "market_micro_order_flow_imbalance_norm", 3),
            feature_ema(sequence, idx, "market_micro_relative_volume_norm", 3),
        ],
        dtype=np.float32,
    )


def _runtime_sample_filter(sequence, idx, horizon):
    obs = sequence[idx]
    return (
        observation_feature(obs, "data_quality_quote_agreement_norm", 1.0) >= 0.84
        and observation_feature(obs, "data_quality_quote_deviation_norm", 0.0) <= 0.22
        and abs(observation_feature(obs, "spread_bps", 0.0)) <= 20.0
        and observation_feature(obs, "queue_depth", 0.0) >= 1.0
        and observation_feature(obs, "market_data_latency_ms", 0.0) <= 1900.0
        and _scalp_support(obs) >= 0.30
        and abs(_scalp_bias(obs)) >= 0.14
    )


def _runtime_confidence(sequence, idx, horizon):
    obs = sequence[idx]
    return (
        (0.30 * _scalp_support(obs))
        + (0.24 * _clip01(abs(_scalp_bias(obs)) / 0.9))
        + (0.14 * _clip01(1.0 - (abs(observation_feature(obs, "spread_bps", 0.0)) / 18.0)))
        + (0.12 * _clip01(1.0 - (observation_feature(obs, "market_data_latency_ms", 0.0) / 2200.0)))
        + (0.10 * observation_feature(obs, "market_micro_relative_volume_norm"))
        + (0.10 * _quote_quality(obs))
    )


def _runtime_scalp_label(sequence, idx, horizon):
    obs = sequence[idx]
    support = _scalp_support(obs)
    bias = _scalp_bias(obs)
    if support < 0.30 or abs(bias) < 0.14:
        return None

    expected_up = bias >= 0.0
    fwd_ret = future_return(sequence, idx, horizon)
    realized = future_realized_vol(sequence, idx, horizon)
    drawdown = abs(future_max_drawdown(sequence, idx, horizon))
    signed_ret = fwd_ret if expected_up else -fwd_ret
    move_threshold = max(0.00044, 0.00096 - (0.00025 * support))
    if abs(fwd_ret) < move_threshold and realized < 0.018 and drawdown < 0.0090:
        return None

    success_score = (
        signed_ret
        + (0.00080 * support)
        + (0.00020 * _quote_quality(obs))
        - (0.18 * realized)
        - (0.24 * drawdown)
    )
    failure_score = (
        (-signed_ret)
        + (0.19 * realized)
        + (0.24 * drawdown)
    )
    if success_score >= 0.00046:
        return 1.0 if expected_up else 0.0
    if failure_score >= 0.00064:
        return 0.0 if expected_up else 1.0
    return None


def _train_synthetic():
    return train_indicator_bot(
        run_tag="brain_refinery_v44_intraday_scalp_1m_5m",
        feature_names=["ret1", "ret5", "ret15", "dist_mid", "vwap_dev", "noise"],
        feature_builder=build_features,
        window=36,
        horizon=2,
    )


def train_brain():
    return train_runtime_indicator_bot(
        run_tag="brain_refinery_v44_intraday_scalp_1m_5m",
        feature_names=[
            "pct_from_close",
            "mom_5m",
            "vol_30m",
            "range_pos",
            "spread_bps",
            "queue_depth",
            "market_data_latency_ms",
            "day_execution_cost_risk_norm",
            "day_opening_auction_signal_norm",
            "market_micro_opening_auction_norm",
            "day_session_open_norm",
            "day_session_midday_norm",
            "day_session_power_hour_norm",
            "market_micro_relative_volume_norm",
            "market_micro_trend_persistence_norm",
            "market_micro_order_flow_imbalance_norm",
            "data_quality_quote_agreement_norm",
            "data_quality_quote_deviation_norm",
            "behavior_prior",
            "futures_specialist_vote",
            "ret_1",
            "ret_3",
            "pct_from_close_std_4",
            "pct_from_close_ema_3",
            "spread_bps_ema_3",
            "spread_bps_std_4",
            "market_micro_order_flow_imbalance_norm_ema_3",
            "market_micro_relative_volume_norm_ema_3",
        ],
        runtime_feature_builder=_runtime_feature_vector,
        runtime_label_builder=_runtime_scalp_label,
        mode_allowlist=_SCALP_RUNTIME_MODES,
        symbol_allowlist=_SCALP_SYMBOLS,
        sample_filter=_runtime_sample_filter,
        confidence_builder=_runtime_confidence,
        min_confidence=0.46,
        lookback_days=45,
        window=16,
        horizon=3,
        min_samples=224,
        min_sequences=4,
        min_positive_samples=40,
        min_negative_samples=40,
        acted_prob_threshold=0.67,
        fallback_trainer=_train_synthetic,
        allow_fallback_on_insufficient_data=False,
        max_best_val_loss=0.6925,
        max_final_val_loss=0.7050,
        min_long_precision=0.52,
        min_short_precision=0.52,
        require_both_sides_precision=True,
        min_acted_accuracy=0.53,
        min_long_acted_count=6,
        min_short_acted_count=6,
        min_accuracy_lift_over_majority=0.015,
        min_precision_balance_score=0.25,
    )


if __name__ == "__main__":
    train_brain()
