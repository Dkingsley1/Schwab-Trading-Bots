import numpy as np

from indicator_bot_common import train_runtime_indicator_bot
from runtime_requested_bot_common import base_runtime_gate, centered01, clip01, quote_quality
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
    risk_on = observation_feature(obs, "flow_risk_on_norm")
    risk_off = observation_feature(obs, "breadth_risk_off_norm")
    coherence = clip01(1.0 - abs(risk_on - risk_off))
    basis_stability = clip01(
        1.0
        - max(
            abs(centered01(observation_feature(obs, "futures_basis_divergence_norm", 0.5))),
            abs(centered01(observation_feature(obs, "crypto_hyperliquid_basis_norm", 0.5))),
        )
    )
    return clip01(
        (0.24 * quote_quality(obs))
        + (0.20 * coherence)
        + (0.16 * basis_stability)
        + (0.14 * clip01(1.0 - observation_feature(obs, "capital_flow_outflow_norm")))
        + (0.14 * clip01(1.0 - observation_feature(obs, "options_negative_bias_norm")))
        + (0.12 * clip01(1.0 - observation_feature(obs, "bond_credit_risk_off_norm")))
    )


def _runtime_feature_vector(sequence, idx):
    obs = sequence[idx]
    return np.asarray(
        [
            observation_feature(obs, "pct_from_close"),
            observation_feature(obs, "mom_5m"),
            observation_feature(obs, "vol_30m"),
            observation_feature(obs, "flow_risk_on_norm"),
            observation_feature(obs, "breadth_risk_off_norm"),
            observation_feature(obs, "capital_flow_outflow_norm"),
            observation_feature(obs, "options_negative_bias_norm"),
            observation_feature(obs, "bond_credit_risk_off_norm"),
            observation_feature(obs, "futures_basis_divergence_norm"),
            observation_feature(obs, "crypto_hyperliquid_basis_norm"),
            observation_feature(obs, "crypto_cross_provider_price_agreement_norm", 1.0),
            observation_feature(obs, "market_micro_relative_volume_norm"),
            observation_feature(obs, "data_quality_quote_agreement_norm", 1.0),
            observation_feature(obs, "data_quality_quote_deviation_norm"),
            observation_feature(obs, "behavior_prior"),
            price_change(sequence, idx, 3),
            price_change(sequence, idx, 6),
            feature_ema(sequence, idx, "flow_risk_on_norm", 4),
            feature_ema(sequence, idx, "breadth_risk_off_norm", 4),
            _support_score(obs),
        ],
        dtype=np.float32,
    )


def _runtime_sample_filter(sequence, idx, horizon):
    obs = sequence[idx]
    return (
        base_runtime_gate(obs, min_quote_agreement=0.76, max_quote_deviation=0.30, max_spread_bps=34.0, min_queue_depth=0.0, max_latency_ms=3400.0)
        and _support_score(obs) >= 0.22
    )


def _runtime_confidence(sequence, idx, horizon):
    obs = sequence[idx]
    return (
        (0.52 * _support_score(obs))
        + (0.20 * quote_quality(obs))
        + (0.14 * observation_feature(obs, "market_micro_relative_volume_norm"))
        + (0.14 * clip01(1.0 - observation_feature(obs, "capital_flow_outflow_norm")))
    )


def train_brain():
    return train_runtime_indicator_bot(
        run_tag="brain_refinery_v106_cross_asset_regime_stability_guard",
        feature_names=[
            "pct_from_close",
            "mom_5m",
            "vol_30m",
            "flow_risk_on_norm",
            "breadth_risk_off_norm",
            "capital_flow_outflow_norm",
            "options_negative_bias_norm",
            "bond_credit_risk_off_norm",
            "futures_basis_divergence_norm",
            "crypto_hyperliquid_basis_norm",
            "crypto_cross_provider_price_agreement_norm",
            "market_micro_relative_volume_norm",
            "data_quality_quote_agreement_norm",
            "data_quality_quote_deviation_norm",
            "behavior_prior",
            "ret_3",
            "ret_6",
            "flow_risk_on_ema_4",
            "breadth_risk_off_ema_4",
            "cross_asset_support_score",
        ],
        runtime_feature_builder=_runtime_feature_vector,
        runtime_label_builder=risk_support_label_builder(
            min_return=0.0001,
            max_drawdown=0.015,
            max_realized_vol=0.022,
            vol_multiplier=2.8,
        ),
        sample_filter=_runtime_sample_filter,
        confidence_builder=_runtime_confidence,
        min_confidence=0.32,
        lookback_days=75,
        mode_allowlist=_MODES,
        window=18,
        horizon=6,
        min_samples=320,
        min_sequences=4,
        min_positive_samples=60,
        min_negative_samples=40,
        max_best_val_loss=0.694,
        max_final_val_loss=0.706,
        min_acted_accuracy=0.60,
        min_accuracy_lift_over_majority=0.02,
        acted_prob_threshold=0.78,
        min_acted_coverage=0.04,
        max_acted_coverage=0.24,
        allow_fallback_on_insufficient_data=False,
        require_both_sides_precision=False,
    )


if __name__ == "__main__":
    train_brain()
