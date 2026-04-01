import numpy as np

from indicator_bot_common import ema, rolling_std, train_indicator_bot, train_runtime_indicator_bot
from runtime_training_common import (
    feature_ema,
    feature_std,
    future_return,
    observation_feature,
    price_change,
)


def build_features(panel):
    r = panel["ret"]
    h = panel["high"]
    l = panel["low"]
    c = panel["close"]
    v = panel["volume"]

    edge_proxy = np.abs(ema(r, 8))
    spread_proxy = (h - l) / (c + 1e-8)
    slippage_proxy = spread_proxy / (v / (ema(v, 20) + 1e-8) + 1e-8)
    tx_cost = spread_proxy + slippage_proxy

    net_edge = edge_proxy - tx_cost
    net_edge_z = net_edge / (rolling_std(net_edge, 20) + 1e-8)
    pass_filter = (net_edge > 0.0).astype(float)

    return np.stack([r, edge_proxy, spread_proxy, slippage_proxy, tx_cost, net_edge, net_edge_z, pass_filter], axis=1)


def _clip01(value):
    return float(np.clip(value, 0.0, 1.0))


def _cost_pressure(obs):
    spread = _clip01(abs(observation_feature(obs, "spread_bps")) / 25.0)
    fill = _clip01(abs(observation_feature(obs, "lag_expected_fill_delta_bps")) / 20.0)
    slip = _clip01(abs(observation_feature(obs, "lag_slippage_bps")) / 20.0)
    latency = _clip01(abs(observation_feature(obs, "lag_latency_ms")) / 4000.0)
    return _clip01((0.30 * spread) + (0.28 * fill) + (0.28 * slip) + (0.14 * latency))


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
            observation_feature(obs, "lag_expected_fill_delta_bps"),
            observation_feature(obs, "lag_slippage_bps"),
            observation_feature(obs, "lag_latency_ms"),
            observation_feature(obs, "market_micro_relative_volume_norm"),
            observation_feature(obs, "market_micro_order_flow_imbalance_norm"),
            observation_feature(obs, "market_micro_block_trade_norm"),
            observation_feature(obs, "options_unusual_flow_norm"),
            observation_feature(obs, "options_specialist_vote"),
            observation_feature(obs, "futures_specialist_vote"),
            observation_feature(obs, "behavior_prior"),
            price_change(sequence, idx, 1),
            price_change(sequence, idx, 3),
            feature_std(sequence, idx, "pct_from_close", 4),
            feature_ema(sequence, idx, "spread_bps", 4),
            feature_ema(sequence, idx, "lag_slippage_bps", 4),
        ],
        dtype=np.float32,
    )


def _runtime_sample_filter(sequence, idx, horizon):
    obs = sequence[idx]
    return (
        observation_feature(obs, "data_quality_quote_agreement_norm", 1.0) >= 0.76
        and max(
            observation_feature(obs, "market_micro_order_flow_imbalance_norm"),
            observation_feature(obs, "market_micro_block_trade_norm"),
            abs(observation_feature(obs, "behavior_prior")),
        )
        >= 0.10
    )


def _runtime_confidence(sequence, idx, horizon):
    obs = sequence[idx]
    quote_quality = _clip01(
        0.60 * observation_feature(obs, "data_quality_quote_agreement_norm", 1.0)
        + 0.40 * (1.0 - observation_feature(obs, "data_quality_quote_deviation_norm", 0.0))
    )
    return (
        (0.26 * (1.0 - _cost_pressure(obs)))
        + (0.18 * observation_feature(obs, "market_micro_order_flow_imbalance_norm"))
        + (0.16 * observation_feature(obs, "market_micro_relative_volume_norm"))
        + (0.12 * observation_feature(obs, "market_micro_block_trade_norm"))
        + (0.12 * max(abs(observation_feature(obs, "options_specialist_vote")), abs(observation_feature(obs, "futures_specialist_vote"))))
        + (0.16 * quote_quality)
    )


def _runtime_cost_label(sequence, idx, horizon):
    obs = sequence[idx]
    future_ret = future_return(sequence, idx, horizon)
    cost_pressure = _cost_pressure(obs)
    realized_edge = abs(future_ret)
    net_after_cost = realized_edge - (0.0018 * cost_pressure)
    if net_after_cost < 0.00035:
        return 0.0
    return 1.0 if future_ret > 0.0 else 0.0


def _train_synthetic():
    return train_indicator_bot(
        run_tag="brain_refinery_v69_cost_aware_execution_filter",
        feature_names=["ret", "edge_proxy", "spread_proxy", "slippage_proxy", "tx_cost", "net_edge", "net_edge_z", "pass_filter"],
        feature_builder=build_features,
        window=44,
        horizon=3,
    )


if __name__ == "__main__":
    train_runtime_indicator_bot(
        run_tag="brain_refinery_v69_cost_aware_execution_filter",
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
            "lag_expected_fill_delta_bps",
            "lag_slippage_bps",
            "lag_latency_ms",
            "market_micro_relative_volume_norm",
            "market_micro_order_flow_imbalance_norm",
            "market_micro_block_trade_norm",
            "options_unusual_flow_norm",
            "options_specialist_vote",
            "futures_specialist_vote",
            "behavior_prior",
            "ret_1",
            "ret_3",
            "pct_from_close_std_4",
            "spread_bps_ema_4",
            "lag_slippage_bps_ema_4",
        ],
        runtime_feature_builder=_runtime_feature_vector,
        runtime_label_builder=_runtime_cost_label,
        sample_filter=_runtime_sample_filter,
        confidence_builder=_runtime_confidence,
        min_confidence=0.36,
        lookback_days=30,
        window=18,
        horizon=4,
        min_samples=224,
        min_sequences=8,
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
