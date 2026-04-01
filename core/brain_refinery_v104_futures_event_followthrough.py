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
    "shadow_aggressive_equities",
    "shadow_intraday_aggressive_equities",
    "shadow_swing_aggressive_equities",
]


def _event_signal(obs) -> float:
    return clip01(
        max(
            observation_feature(obs, "calendar_event_proximity_norm"),
            observation_feature(obs, "calendar_high_impact_24h_norm"),
            observation_feature(obs, "calendar_macro_event_norm"),
            observation_feature(obs, "calendar_fomc_event_norm"),
            observation_feature(obs, "calendar_cpi_event_norm"),
            observation_feature(obs, "calendar_labor_event_norm"),
        )
    )


def _microstructure_signal(obs) -> float:
    return clip01(
        (0.24 * abs(centered01(observation_feature(obs, "futures_order_book_imbalance_norm", 0.5))))
        + (0.18 * abs(centered01(observation_feature(obs, "futures_taker_imbalance_norm", 0.5))))
        + (0.18 * observation_feature(obs, "futures_session_volume_profile_norm"))
        + (0.14 * observation_feature(obs, "market_micro_relative_volume_norm"))
        + (0.12 * observation_feature(obs, "market_micro_order_flow_imbalance_norm"))
        + (0.14 * quote_quality(obs))
    )


def _bias(obs) -> float:
    return float(
        np.clip(
            (0.26 * observation_feature(obs, "behavior_prior"))
            + (0.18 * centered01(observation_feature(obs, "futures_specialist_vote", 0.5)))
            + (0.16 * centered01(observation_feature(obs, "futures_order_book_imbalance_norm", 0.5)))
            + (0.12 * centered01(observation_feature(obs, "futures_taker_imbalance_norm", 0.5)))
            + (0.14 * observation_feature(obs, "mom_5m") * 140.0)
            + (0.14 * observation_feature(obs, "pct_from_close") * 120.0),
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
            observation_feature(obs, "behavior_prior"),
            observation_feature(obs, "futures_specialist_vote"),
            observation_feature(obs, "futures_order_book_imbalance_norm"),
            observation_feature(obs, "futures_taker_imbalance_norm"),
            observation_feature(obs, "futures_basis_divergence_norm"),
            observation_feature(obs, "futures_session_volume_profile_norm"),
            observation_feature(obs, "market_micro_relative_volume_norm"),
            observation_feature(obs, "market_micro_order_flow_imbalance_norm"),
            observation_feature(obs, "calendar_event_proximity_norm"),
            observation_feature(obs, "calendar_high_impact_24h_norm"),
            observation_feature(obs, "calendar_macro_event_norm"),
            observation_feature(obs, "data_quality_quote_agreement_norm", 1.0),
            observation_feature(obs, "data_quality_quote_deviation_norm"),
            price_change(sequence, idx, 3),
            price_change(sequence, idx, 6),
            feature_ema(sequence, idx, "futures_order_book_imbalance_norm", 4),
            feature_ema(sequence, idx, "market_micro_order_flow_imbalance_norm", 4),
        ],
        dtype=np.float32,
    )


def _runtime_sample_filter(sequence, idx, horizon):
    obs = sequence[idx]
    return (
        base_runtime_gate(obs, min_quote_agreement=0.82, max_quote_deviation=0.24, max_spread_bps=22.0, min_queue_depth=1.0, max_latency_ms=2000.0)
        and max(_event_signal(obs), _microstructure_signal(obs)) >= 0.22
        and abs(_bias(obs)) >= 0.10
    )


def _runtime_confidence(sequence, idx, horizon):
    obs = sequence[idx]
    return (
        (0.28 * max(_event_signal(obs), _microstructure_signal(obs)))
        + (0.24 * clip01(abs(_bias(obs)) / 0.9))
        + (0.18 * quote_quality(obs))
        + (0.16 * observation_feature(obs, "market_micro_order_flow_imbalance_norm"))
        + (0.14 * observation_feature(obs, "futures_session_volume_profile_norm"))
    )


def _runtime_event_followthrough_label(sequence, idx, horizon):
    obs = sequence[idx]
    return directional_outcome_label(
        sequence,
        idx,
        horizon,
        bias=_bias(obs),
        support=max(_event_signal(obs), _microstructure_signal(obs)),
        min_support=0.22,
        min_abs_bias=0.10,
        move_base=0.00100,
        move_scale=0.00048,
        success_floor=0.00030,
        failure_floor=0.00052,
    )


def train_brain():
    return train_runtime_indicator_bot(
        run_tag="brain_refinery_v104_futures_event_followthrough",
        feature_names=[
            "pct_from_close",
            "mom_5m",
            "vol_30m",
            "range_pos",
            "spread_bps",
            "queue_depth",
            "market_data_latency_ms",
            "behavior_prior",
            "futures_specialist_vote",
            "futures_order_book_imbalance_norm",
            "futures_taker_imbalance_norm",
            "futures_basis_divergence_norm",
            "futures_session_volume_profile_norm",
            "market_micro_relative_volume_norm",
            "market_micro_order_flow_imbalance_norm",
            "calendar_event_proximity_norm",
            "calendar_high_impact_24h_norm",
            "calendar_macro_event_norm",
            "data_quality_quote_agreement_norm",
            "data_quality_quote_deviation_norm",
            "ret_3",
            "ret_6",
            "order_book_ema_4",
            "order_flow_ema_4",
        ],
        runtime_feature_builder=_runtime_feature_vector,
        runtime_label_builder=_runtime_event_followthrough_label,
        sample_filter=_runtime_sample_filter,
        confidence_builder=_runtime_confidence,
        min_confidence=0.35,
        lookback_days=45,
        mode_allowlist=_MODES,
        window=20,
        horizon=4,
        min_samples=300,
        min_sequences=4,
        min_positive_samples=70,
        min_negative_samples=70,
        max_best_val_loss=0.693,
        max_final_val_loss=0.705,
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
