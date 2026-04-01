import numpy as np

from indicator_bot_common import train_runtime_indicator_bot
from runtime_requested_bot_common import base_runtime_gate, clip01, quote_quality
from runtime_training_common import (
    feature_ema,
    observation_feature,
    price_change,
    risk_support_label_builder,
)

_MODES = [
    "shadow_equities",
    "shadow_aggressive_equities",
    "shadow_intraday_aggressive_equities",
    "shadow_conservative_equities",
    "shadow_dividend_equities",
    "shadow_crypto",
    "shadow_crypto_futures_crypto",
]


def _support_score(obs) -> float:
    spread_ok = clip01(1.0 - (abs(observation_feature(obs, "spread_bps", 0.0)) / 24.0))
    latency_ok = clip01(1.0 - (observation_feature(obs, "market_data_latency_ms", 0.0) / 2800.0))
    queue_ok = clip01(observation_feature(obs, "queue_depth", 0.0) / 4.0)
    return clip01(
        (0.28 * quote_quality(obs))
        + (0.18 * observation_feature(obs, "crypto_cross_provider_price_agreement_norm", 1.0))
        + (0.16 * spread_ok)
        + (0.16 * latency_ok)
        + (0.12 * queue_ok)
        + (0.10 * observation_feature(obs, "market_micro_relative_volume_norm"))
    )


def _runtime_feature_vector(sequence, idx):
    obs = sequence[idx]
    return np.asarray(
        [
            observation_feature(obs, "spread_bps"),
            observation_feature(obs, "queue_depth"),
            observation_feature(obs, "market_data_latency_ms"),
            observation_feature(obs, "lag_latency_ms"),
            observation_feature(obs, "lag_expected_fill_delta_bps"),
            observation_feature(obs, "lag_slippage_bps"),
            observation_feature(obs, "market_micro_relative_volume_norm"),
            observation_feature(obs, "data_quality_quote_agreement_norm", 1.0),
            observation_feature(obs, "data_quality_quote_deviation_norm"),
            observation_feature(obs, "data_quality_market_data_latency_norm"),
            observation_feature(obs, "crypto_cross_provider_price_agreement_norm", 1.0),
            observation_feature(obs, "infra_risk_throttle_norm"),
            observation_feature(obs, "behavior_prior"),
            price_change(sequence, idx, 1),
            price_change(sequence, idx, 3),
            feature_ema(sequence, idx, "data_quality_quote_agreement_norm", 4),
            feature_ema(sequence, idx, "infra_risk_throttle_norm", 4),
        ],
        dtype=np.float32,
    )


def _runtime_sample_filter(sequence, idx, horizon):
    obs = sequence[idx]
    return (
        base_runtime_gate(obs, min_quote_agreement=0.82, max_quote_deviation=0.22, max_spread_bps=24.0, min_queue_depth=0.5, max_latency_ms=2800.0)
        and _support_score(obs) >= 0.24
    )


def _runtime_confidence(sequence, idx, horizon):
    obs = sequence[idx]
    return (
        (0.48 * _support_score(obs))
        + (0.24 * quote_quality(obs))
        + (0.14 * clip01(1.0 - observation_feature(obs, "infra_risk_throttle_norm")))
        + (0.14 * observation_feature(obs, "market_micro_relative_volume_norm"))
    )


def train_brain():
    return train_runtime_indicator_bot(
        run_tag="brain_refinery_v105_feed_consensus_execution_guard",
        feature_names=[
            "spread_bps",
            "queue_depth",
            "market_data_latency_ms",
            "lag_latency_ms",
            "lag_expected_fill_delta_bps",
            "lag_slippage_bps",
            "market_micro_relative_volume_norm",
            "data_quality_quote_agreement_norm",
            "data_quality_quote_deviation_norm",
            "data_quality_market_data_latency_norm",
            "crypto_cross_provider_price_agreement_norm",
            "infra_risk_throttle_norm",
            "behavior_prior",
            "ret_1",
            "ret_3",
            "quote_agreement_ema_4",
            "throttle_ema_4",
        ],
        runtime_feature_builder=_runtime_feature_vector,
        runtime_label_builder=risk_support_label_builder(
            min_return=-0.0012,
            max_drawdown=0.018,
            max_realized_vol=0.026,
            vol_multiplier=3.2,
        ),
        sample_filter=_runtime_sample_filter,
        confidence_builder=_runtime_confidence,
        min_confidence=0.28,
        lookback_days=45,
        mode_allowlist=_MODES,
        window=18,
        horizon=6,
        min_samples=320,
        min_sequences=4,
        min_positive_samples=90,
        max_best_val_loss=0.694,
        max_final_val_loss=0.706,
        min_acted_accuracy=0.56,
        min_accuracy_lift_over_majority=0.015,
        allow_fallback_on_insufficient_data=False,
        require_both_sides_precision=False,
    )


if __name__ == "__main__":
    train_brain()
