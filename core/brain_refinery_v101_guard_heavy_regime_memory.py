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
    "shadow_equities",
    "shadow_aggressive_equities",
    "shadow_intraday_aggressive_equities",
    "shadow_conservative_equities",
    "shadow_dividend_equities",
    "shadow_crypto",
    "shadow_crypto_futures_crypto",
]


def _guard_pressure(obs) -> float:
    return clip01(
        (0.28 * observation_feature(obs, "breadth_risk_off_norm"))
        + (0.22 * observation_feature(obs, "bond_credit_risk_off_norm"))
        + (0.18 * observation_feature(obs, "options_negative_bias_norm"))
        + (0.16 * observation_feature(obs, "capital_flow_outflow_norm"))
        + (0.16 * observation_feature(obs, "infra_risk_throttle_norm"))
    )


def _risk_on(obs) -> float:
    return clip01(
        (0.36 * observation_feature(obs, "flow_risk_on_norm"))
        + (0.22 * observation_feature(obs, "market_micro_relative_volume_norm"))
        + (0.18 * max(observation_feature(obs, "mom_5m"), 0.0) * 120.0)
        + (0.14 * max(observation_feature(obs, "pct_from_close"), 0.0) * 90.0)
        + (0.10 * quote_quality(obs))
    )


def _bias(obs) -> float:
    return float(
        np.clip(
            (0.52 * (_risk_on(obs) - _guard_pressure(obs)))
            + (0.24 * observation_feature(obs, "behavior_prior"))
            + (0.14 * observation_feature(obs, "mom_5m") * 140.0)
            + (0.10 * centered01(observation_feature(obs, "range_pos", 0.5))),
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
            observation_feature(obs, "behavior_prior"),
            observation_feature(obs, "flow_risk_on_norm"),
            observation_feature(obs, "capital_flow_outflow_norm"),
            observation_feature(obs, "breadth_risk_off_norm"),
            observation_feature(obs, "bond_credit_risk_off_norm"),
            observation_feature(obs, "options_negative_bias_norm"),
            observation_feature(obs, "infra_risk_throttle_norm"),
            observation_feature(obs, "market_micro_relative_volume_norm"),
            observation_feature(obs, "data_quality_quote_agreement_norm", 1.0),
            observation_feature(obs, "data_quality_quote_deviation_norm"),
            price_change(sequence, idx, 3),
            price_change(sequence, idx, 6),
            feature_ema(sequence, idx, "flow_risk_on_norm", 4),
            feature_ema(sequence, idx, "breadth_risk_off_norm", 4),
        ],
        dtype=np.float32,
    )


def _runtime_sample_filter(sequence, idx, horizon):
    obs = sequence[idx]
    return (
        base_runtime_gate(obs, min_quote_agreement=0.80, max_quote_deviation=0.24, max_spread_bps=28.0, min_queue_depth=0.0)
        and max(_guard_pressure(obs), _risk_on(obs)) >= 0.18
        and abs(_bias(obs)) >= 0.07
    )


def _runtime_confidence(sequence, idx, horizon):
    obs = sequence[idx]
    return (
        (0.28 * max(_guard_pressure(obs), _risk_on(obs)))
        + (0.24 * clip01(abs(_bias(obs)) / 0.9))
        + (0.18 * quote_quality(obs))
        + (0.16 * observation_feature(obs, "market_micro_relative_volume_norm"))
        + (0.14 * (1.0 - observation_feature(obs, "infra_risk_throttle_norm")))
    )


def _runtime_guard_memory_label(sequence, idx, horizon):
    obs = sequence[idx]
    return directional_outcome_label(
        sequence,
        idx,
        horizon,
        bias=_bias(obs),
        support=max(_guard_pressure(obs), _risk_on(obs)),
        min_support=0.18,
        min_abs_bias=0.07,
        move_base=0.00115,
        move_scale=0.00055,
        success_floor=0.00032,
        failure_floor=0.00052,
    )


def train_brain():
    return train_runtime_indicator_bot(
        run_tag="brain_refinery_v101_guard_heavy_regime_memory",
        feature_names=[
            "pct_from_close",
            "mom_5m",
            "vol_30m",
            "range_pos",
            "behavior_prior",
            "flow_risk_on_norm",
            "capital_flow_outflow_norm",
            "breadth_risk_off_norm",
            "bond_credit_risk_off_norm",
            "options_negative_bias_norm",
            "infra_risk_throttle_norm",
            "market_micro_relative_volume_norm",
            "data_quality_quote_agreement_norm",
            "data_quality_quote_deviation_norm",
            "ret_3",
            "ret_6",
            "flow_risk_on_ema_4",
            "breadth_risk_off_ema_4",
        ],
        runtime_feature_builder=_runtime_feature_vector,
        runtime_label_builder=_runtime_guard_memory_label,
        sample_filter=_runtime_sample_filter,
        confidence_builder=_runtime_confidence,
        min_confidence=0.34,
        lookback_days=45,
        mode_allowlist=_MODES,
        window=24,
        horizon=6,
        min_samples=320,
        min_sequences=4,
        min_positive_samples=80,
        min_negative_samples=80,
        max_best_val_loss=0.695,
        max_final_val_loss=0.705,
        min_long_precision=0.52,
        min_short_precision=0.52,
        require_both_sides_precision=True,
        min_acted_accuracy=0.55,
        min_long_acted_count=6,
        min_short_acted_count=6,
        min_accuracy_lift_over_majority=0.015,
        min_precision_balance_score=0.35,
        allow_fallback_on_insufficient_data=False,
    )


if __name__ == "__main__":
    train_brain()
