import numpy as np

from indicator_bot_common import train_runtime_indicator_bot
from runtime_requested_bot_common import (
    base_runtime_gate,
    centered01,
    clip01,
    directional_outcome_label,
    quote_quality,
)
from runtime_training_common import feature_ema, observation_feature, price_change

_MODES = [
    "shadow_crypto",
    "shadow_crypto_futures_crypto",
]


def _setup_signal(obs) -> float:
    return clip01(
        max(
            observation_feature(obs, "crypto_deribit_mark_iv_norm"),
            abs(centered01(observation_feature(obs, "crypto_hyperliquid_basis_norm", 0.5))),
            observation_feature(obs, "crypto_hyperliquid_open_interest_norm"),
            abs(centered01(observation_feature(obs, "crypto_hyperliquid_funding_norm", 0.5))),
            observation_feature(obs, "crypto_defillama_dex_volume_growth_norm"),
        )
    )


def _relief_signal(obs) -> float:
    spread_ok = clip01(1.0 - (abs(observation_feature(obs, "spread_bps", 0.0)) / 24.0))
    queue_ok = clip01(observation_feature(obs, "queue_depth", 0.0) / 4.0)
    return clip01(
        (0.34 * clip01(1.0 - observation_feature(obs, "infra_risk_throttle_norm")))
        + (0.18 * observation_feature(obs, "crypto_cross_provider_price_agreement_norm"))
        + (0.16 * spread_ok)
        + (0.16 * queue_ok)
        + (0.16 * quote_quality(obs))
    )


def _bias(obs) -> float:
    return float(
        np.clip(
            (0.28 * observation_feature(obs, "behavior_prior"))
            + (0.20 * centered01(observation_feature(obs, "crypto_coingecko_momentum_norm", 0.5)))
            + (0.16 * centered01(observation_feature(obs, "crypto_hyperliquid_basis_norm", 0.5)))
            + (0.12 * centered01(observation_feature(obs, "crypto_hyperliquid_funding_norm", 0.5)))
            + (0.12 * observation_feature(obs, "mom_5m") * 150.0)
            + (0.12 * observation_feature(obs, "pct_from_close") * 110.0)
            - (0.12 * observation_feature(obs, "infra_risk_throttle_norm")),
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
            observation_feature(obs, "spread_bps"),
            observation_feature(obs, "queue_depth"),
            observation_feature(obs, "market_data_latency_ms"),
            observation_feature(obs, "lag_latency_ms"),
            observation_feature(obs, "lag_slippage_bps"),
            observation_feature(obs, "behavior_prior"),
            observation_feature(obs, "infra_risk_throttle_norm"),
            observation_feature(obs, "data_quality_quote_agreement_norm", 1.0),
            observation_feature(obs, "crypto_cross_provider_price_agreement_norm"),
            observation_feature(obs, "crypto_deribit_mark_iv_norm"),
            observation_feature(obs, "crypto_hyperliquid_open_interest_norm"),
            observation_feature(obs, "crypto_hyperliquid_funding_norm"),
            observation_feature(obs, "crypto_hyperliquid_basis_norm"),
            observation_feature(obs, "crypto_coingecko_momentum_norm"),
            observation_feature(obs, "crypto_defillama_dex_volume_growth_norm"),
            price_change(sequence, idx, 1),
            price_change(sequence, idx, 3),
            feature_ema(sequence, idx, "infra_risk_throttle_norm", 4),
            feature_ema(sequence, idx, "crypto_hyperliquid_open_interest_norm", 4),
        ],
        dtype=np.float32,
    )


def _runtime_sample_filter(sequence, idx, horizon):
    obs = sequence[idx]
    return (
        base_runtime_gate(obs, min_quote_agreement=0.80, max_quote_deviation=0.24, max_spread_bps=28.0, min_queue_depth=0.8, max_latency_ms=2600.0)
        and observation_feature(obs, "crypto_cross_provider_price_agreement_norm", 0.0) >= 0.72
        and _setup_signal(obs) >= 0.20
        and _relief_signal(obs) >= 0.18
        and abs(_bias(obs)) >= 0.08
    )


def _runtime_confidence(sequence, idx, horizon):
    obs = sequence[idx]
    return (
        (0.26 * _setup_signal(obs))
        + (0.24 * _relief_signal(obs))
        + (0.20 * clip01(abs(_bias(obs)) / 0.9))
        + (0.16 * quote_quality(obs))
        + (0.14 * observation_feature(obs, "crypto_hyperliquid_open_interest_norm"))
    )


def _runtime_crypto_relief_label(sequence, idx, horizon):
    obs = sequence[idx]
    return directional_outcome_label(
        sequence,
        idx,
        horizon,
        bias=_bias(obs),
        support=max(_setup_signal(obs), _relief_signal(obs)),
        min_support=0.20,
        min_abs_bias=0.08,
        move_base=0.00095,
        move_scale=0.00042,
        move_floor=0.00030,
        success_floor=0.00025,
        failure_floor=0.00048,
    )


def train_brain():
    return train_runtime_indicator_bot(
        run_tag="brain_refinery_v103_crypto_throttle_relief_momentum",
        feature_names=[
            "pct_from_close",
            "mom_5m",
            "vol_30m",
            "spread_bps",
            "queue_depth",
            "market_data_latency_ms",
            "lag_latency_ms",
            "lag_slippage_bps",
            "behavior_prior",
            "infra_risk_throttle_norm",
            "data_quality_quote_agreement_norm",
            "crypto_cross_provider_price_agreement_norm",
            "crypto_deribit_mark_iv_norm",
            "crypto_hyperliquid_open_interest_norm",
            "crypto_hyperliquid_funding_norm",
            "crypto_hyperliquid_basis_norm",
            "crypto_coingecko_momentum_norm",
            "crypto_defillama_dex_volume_growth_norm",
            "ret_1",
            "ret_3",
            "infra_risk_throttle_ema_4",
            "open_interest_ema_4",
        ],
        runtime_feature_builder=_runtime_feature_vector,
        runtime_label_builder=_runtime_crypto_relief_label,
        sample_filter=_runtime_sample_filter,
        confidence_builder=_runtime_confidence,
        min_confidence=0.34,
        lookback_days=45,
        mode_allowlist=_MODES,
        window=20,
        horizon=4,
        min_samples=280,
        min_sequences=3,
        min_positive_samples=70,
        min_negative_samples=70,
        max_best_val_loss=0.694,
        max_final_val_loss=0.706,
        min_long_precision=0.52,
        min_short_precision=0.52,
        require_both_sides_precision=True,
        min_acted_accuracy=0.55,
        min_long_acted_count=5,
        min_short_acted_count=5,
        min_accuracy_lift_over_majority=0.015,
        min_precision_balance_score=0.35,
        allow_fallback_on_insufficient_data=False,
    )


if __name__ == "__main__":
    train_brain()
