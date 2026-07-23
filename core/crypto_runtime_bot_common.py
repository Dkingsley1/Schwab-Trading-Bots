"""Shared runtime training helpers for collection-first crypto bot modules."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

from indicator_bot_common import train_runtime_indicator_bot
from runtime_requested_bot_common import (
    base_runtime_gate,
    centered01,
    clip01,
    directional_outcome_label,
    quote_quality,
)
from runtime_training_common import feature_ema, feature_std, observation_feature, price_change

CRYPTO_MODES = ("shadow_crypto", "shadow_crypto_futures_crypto")


@dataclass(frozen=True)
class CryptoRuntimeSpec:
    bot_id: str
    feature_names: Sequence[str]
    feature_fields: Sequence[str]
    signal_builder: Callable[[dict], float]
    bias_builder: Callable[[dict], float]
    mode_allowlist: Sequence[str] = CRYPTO_MODES
    min_signal: float = 0.20
    min_abs_bias: float = 0.08
    min_confidence: float = 0.34
    window: int = 24
    horizon: int = 4
    lookback_days: int = 45
    min_samples: int = 280
    min_sequences: int = 3
    min_positive_samples: int = 70
    min_negative_samples: int = 70
    batch_size: int = 128
    defer_on_quality_failure: bool = False


def crypto_quality(obs: dict) -> float:
    return clip01(
        (0.46 * quote_quality(obs))
        + (0.28 * observation_feature(obs, "crypto_cross_provider_price_agreement_norm", 1.0))
        + (0.16 * (1.0 - observation_feature(obs, "data_quality_market_data_latency_norm", 0.0)))
        + (0.10 * clip01(observation_feature(obs, "queue_depth", 0.0) / 4.0))
    )


def funding_pressure(obs: dict) -> float:
    return clip01(
        (0.28 * abs(centered01(observation_feature(obs, "crypto_hyperliquid_funding_norm", 0.5))))
        + (0.24 * abs(centered01(observation_feature(obs, "crypto_hyperliquid_basis_norm", 0.5))))
        + (0.22 * observation_feature(obs, "crypto_hyperliquid_open_interest_norm"))
        + (0.16 * observation_feature(obs, "crypto_deribit_mark_iv_norm"))
        + (0.10 * observation_feature(obs, "crypto_liquidation_pressure_norm"))
    )


def liquidity_impulse(obs: dict) -> float:
    return clip01(
        (0.30 * observation_feature(obs, "crypto_defillama_stablecoin_growth_norm"))
        + (0.26 * observation_feature(obs, "crypto_defillama_dex_volume_growth_norm"))
        + (0.16 * observation_feature(obs, "crypto_exchange_liquidity_norm"))
        + (0.14 * observation_feature(obs, "flow_risk_on_norm"))
        + (0.14 * crypto_quality(obs))
    )


def risk_off_pressure(obs: dict) -> float:
    return clip01(
        (0.24 * observation_feature(obs, "breadth_risk_off_norm"))
        + (0.20 * observation_feature(obs, "infra_risk_throttle_norm"))
        + (0.16 * observation_feature(obs, "ctx_VIX_X_pct_from_close"))
        + (0.14 * abs(centered01(observation_feature(obs, "market_crypto_spy_corr_norm", 0.5))))
        + (0.14 * funding_pressure(obs))
        + (0.12 * observation_feature(obs, "fx_dollar_funding_stress_norm"))
    )


def _runtime_feature_vector(spec: CryptoRuntimeSpec, sequence, idx):
    obs = sequence[idx]
    base = [observation_feature(obs, field) for field in spec.feature_fields]
    derived = [
        price_change(sequence, idx, 1),
        price_change(sequence, idx, 3),
        feature_std(sequence, idx, "pct_from_close", 4),
        feature_ema(sequence, idx, "crypto_hyperliquid_open_interest_norm", 4),
        feature_ema(sequence, idx, "crypto_cross_provider_price_agreement_norm", 4),
    ]
    return np.asarray([*base, *derived], dtype=np.float32)


def _runtime_sample_filter(spec: CryptoRuntimeSpec, sequence, idx, horizon):
    obs = sequence[idx]
    return (
        base_runtime_gate(
            obs,
            min_quote_agreement=0.78,
            max_quote_deviation=0.30,
            max_spread_bps=34.0,
            min_queue_depth=0.6,
            max_latency_ms=3200.0,
        )
        and crypto_quality(obs) >= 0.60
        and spec.signal_builder(obs) >= spec.min_signal
        and abs(spec.bias_builder(obs)) >= spec.min_abs_bias
    )


def _runtime_confidence(spec: CryptoRuntimeSpec, sequence, idx, horizon):
    obs = sequence[idx]
    return (
        (0.30 * spec.signal_builder(obs))
        + (0.24 * clip01(abs(spec.bias_builder(obs)) / 0.9))
        + (0.20 * crypto_quality(obs))
        + (0.14 * observation_feature(obs, "crypto_hyperliquid_open_interest_norm"))
        + (0.12 * clip01(1.0 - observation_feature(obs, "infra_risk_throttle_norm")))
    )


def _runtime_label(spec: CryptoRuntimeSpec, sequence, idx, horizon):
    obs = sequence[idx]
    return directional_outcome_label(
        sequence,
        idx,
        horizon,
        bias=spec.bias_builder(obs),
        support=spec.signal_builder(obs),
        min_support=spec.min_signal,
        min_abs_bias=spec.min_abs_bias,
        move_base=0.00105,
        move_scale=0.00036,
        move_floor=0.00035,
        success_floor=0.00030,
        failure_floor=0.00055,
    )


def train_crypto_runtime_bot(spec: CryptoRuntimeSpec):
    return train_runtime_indicator_bot(
        run_tag=spec.bot_id,
        feature_names=list(spec.feature_names),
        runtime_feature_builder=lambda sequence, idx: _runtime_feature_vector(spec, sequence, idx),
        runtime_label_builder=lambda sequence, idx, horizon: _runtime_label(spec, sequence, idx, horizon),
        sample_filter=lambda sequence, idx, horizon: _runtime_sample_filter(spec, sequence, idx, horizon),
        confidence_builder=lambda sequence, idx, horizon: _runtime_confidence(spec, sequence, idx, horizon),
        min_confidence=spec.min_confidence,
        lookback_days=spec.lookback_days,
        mode_allowlist=list(spec.mode_allowlist),
        window=spec.window,
        horizon=spec.horizon,
        min_samples=spec.min_samples,
        min_sequences=spec.min_sequences,
        min_positive_samples=spec.min_positive_samples,
        min_negative_samples=spec.min_negative_samples,
        batch_size=spec.batch_size,
        max_best_val_loss=0.694,
        max_final_val_loss=0.706,
        min_long_precision=0.52,
        min_short_precision=0.52,
        require_both_sides_precision=True,
        min_acted_accuracy=0.60,
        min_long_acted_count=5,
        min_short_acted_count=5,
        min_accuracy_lift_over_majority=0.03,
        min_precision_balance_score=0.35,
        allow_fallback_on_insufficient_data=False,
        defer_on_quality_failure=spec.defer_on_quality_failure,
    )
