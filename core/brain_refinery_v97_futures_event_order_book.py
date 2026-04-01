import numpy as np

from indicator_bot_common import ema, rolling_std, train_indicator_bot, train_runtime_indicator_bot
from runtime_training_common import (
    feature_ema,
    future_max_drawdown,
    future_realized_vol,
    future_return,
    observation_feature,
    price_change,
)

_FUTURES_EVENT_MODES = [
    "shadow_aggressive_equities",
    "shadow_intraday_aggressive_equities",
    "shadow_swing_aggressive_equities",
]
_FUTURES_EVENT_SYMBOLS = [
    "SPY",
    "QQQ",
    "DIA",
    "IWM",
    "SMH",
    "NVDA",
    "AAPL",
    "MSFT",
]


def build_features(panel):
    c = panel["close"]
    r = panel["ret"]
    v = panel["volume"]

    ret_fast = ema(r, 5)
    ret_slow = ema(r, 15)
    vol_fast = rolling_std(r, 12)
    vol_slow = rolling_std(r, 36)
    rel_volume = v / (ema(v, 20) + 1e-8)
    impulse = np.abs(np.diff(c, prepend=c[0]) / (c + 1e-8))

    return np.stack([r, ret_fast, ret_slow, vol_fast, vol_slow, rel_volume, impulse], axis=1)


def _clip01(value):
    return float(np.clip(value, 0.0, 1.0))


def _centered01(value):
    return float((float(value) - 0.5) * 2.0)


def _quote_quality(obs):
    return _clip01(
        (0.65 * observation_feature(obs, "data_quality_quote_agreement_norm", 1.0))
        + (0.35 * (1.0 - observation_feature(obs, "data_quality_quote_deviation_norm", 0.0)))
    )


def _event_signal(obs):
    return _clip01(
        max(
            observation_feature(obs, "calendar_event_proximity_norm"),
            observation_feature(obs, "calendar_high_impact_24h_norm"),
            observation_feature(obs, "calendar_macro_event_norm"),
            observation_feature(obs, "calendar_fomc_event_norm"),
            observation_feature(obs, "calendar_cpi_event_norm"),
            observation_feature(obs, "calendar_labor_event_norm"),
        )
    )


def _order_book_signal(obs):
    return _clip01(
        (0.28 * abs(_centered01(observation_feature(obs, "futures_order_book_imbalance_norm", 0.5))))
        + (0.18 * abs(_centered01(observation_feature(obs, "futures_taker_imbalance_norm", 0.5))))
        + (0.16 * observation_feature(obs, "futures_session_volume_profile_norm"))
        + (0.14 * observation_feature(obs, "futures_basis_divergence_norm"))
        + (0.10 * abs(_centered01(observation_feature(obs, "futures_basis_bps_norm", 0.5))))
        + (0.08 * abs(_centered01(observation_feature(obs, "futures_negative_bias_norm", 0.5))))
        + (0.06 * _quote_quality(obs))
    )


def _directional_bias(obs):
    return float(
        np.clip(
            (0.24 * observation_feature(obs, "behavior_prior"))
            + (0.18 * _centered01(observation_feature(obs, "futures_specialist_vote", 0.5)))
            + (0.18 * _centered01(observation_feature(obs, "futures_order_book_imbalance_norm", 0.5)))
            + (0.12 * _centered01(observation_feature(obs, "futures_taker_imbalance_norm", 0.5)))
            + (0.12 * observation_feature(obs, "mom_5m") * 140.0)
            + (0.10 * observation_feature(obs, "pct_from_close") * 120.0)
            + (0.06 * _centered01(observation_feature(obs, "futures_basis_bps_norm", 0.5))),
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
            observation_feature(obs, "lag_expected_fill_delta_bps"),
            observation_feature(obs, "lag_latency_ms"),
            observation_feature(obs, "lag_slippage_bps"),
            observation_feature(obs, "market_micro_relative_volume_norm"),
            observation_feature(obs, "data_quality_quote_agreement_norm", 1.0),
            observation_feature(obs, "data_quality_quote_deviation_norm"),
            observation_feature(obs, "behavior_prior"),
            observation_feature(obs, "futures_specialist_vote"),
            observation_feature(obs, "futures_order_book_imbalance_norm"),
            observation_feature(obs, "futures_taker_imbalance_norm"),
            observation_feature(obs, "futures_basis_bps_norm"),
            observation_feature(obs, "futures_basis_divergence_norm"),
            observation_feature(obs, "futures_term_structure_norm"),
            observation_feature(obs, "futures_session_volume_profile_norm"),
            observation_feature(obs, "futures_negative_bias_norm"),
            observation_feature(obs, "futures_liquidation_risk_norm"),
            observation_feature(obs, "calendar_event_proximity_norm"),
            observation_feature(obs, "calendar_high_impact_24h_norm"),
            observation_feature(obs, "calendar_macro_event_norm"),
            price_change(sequence, idx, 3),
            price_change(sequence, idx, 6),
            feature_ema(sequence, idx, "futures_order_book_imbalance_norm", 4),
            feature_ema(sequence, idx, "futures_session_volume_profile_norm", 4),
        ],
        dtype=np.float32,
    )


def _runtime_sample_filter(sequence, idx, horizon):
    obs = sequence[idx]
    return (
        observation_feature(obs, "data_quality_quote_agreement_norm", 1.0) >= 0.82
        and observation_feature(obs, "data_quality_quote_deviation_norm", 0.0) <= 0.24
        and observation_feature(obs, "queue_depth", 0.0) >= 1.0
        and observation_feature(obs, "market_data_latency_ms", 0.0) <= 1800.0
        and max(_event_signal(obs), _order_book_signal(obs)) >= 0.28
        and abs(_directional_bias(obs)) >= 0.14
    )


def _runtime_confidence(sequence, idx, horizon):
    obs = sequence[idx]
    return (
        (0.28 * max(_event_signal(obs), _order_book_signal(obs)))
        + (0.24 * _clip01(abs(_directional_bias(obs)) / 0.9))
        + (0.16 * _quote_quality(obs))
        + (0.16 * observation_feature(obs, "market_micro_relative_volume_norm"))
        + (0.16 * observation_feature(obs, "futures_session_volume_profile_norm"))
    )


def _runtime_futures_event_label(sequence, idx, horizon):
    obs = sequence[idx]
    event_signal = _event_signal(obs)
    order_signal = _order_book_signal(obs)
    bias = _directional_bias(obs)
    if max(event_signal, order_signal) < 0.28 or abs(bias) < 0.14:
        return None

    expected_up = bias >= 0.0
    fwd_ret = future_return(sequence, idx, horizon)
    realized = future_realized_vol(sequence, idx, horizon)
    drawdown = abs(future_max_drawdown(sequence, idx, horizon))
    signed_ret = fwd_ret if expected_up else -fwd_ret
    move_threshold = max(0.00055, 0.00120 - (0.00040 * max(event_signal, order_signal)))
    if abs(fwd_ret) < move_threshold and realized < 0.020 and drawdown < 0.012:
        return None

    success_score = (
        signed_ret
        + (0.00095 * order_signal)
        + (0.00030 * event_signal)
        + (0.00020 * _quote_quality(obs))
        - (0.20 * realized)
        - (0.24 * drawdown)
    )
    failure_score = (
        (-signed_ret)
        + (0.00080 * event_signal)
        + (0.18 * realized)
        + (0.22 * drawdown)
    )
    if success_score >= 0.00050:
        return 1.0 if expected_up else 0.0
    if failure_score >= 0.00060:
        return 0.0 if expected_up else 1.0
    return None


def _train_synthetic():
    return train_indicator_bot(
        run_tag="brain_refinery_v97_futures_event_order_book",
        feature_names=["ret", "ret_fast", "ret_slow", "vol_fast", "vol_slow", "rel_volume", "impulse"],
        feature_builder=build_features,
        window=42,
        horizon=3,
    )


def train_brain():
    return train_runtime_indicator_bot(
        run_tag="brain_refinery_v97_futures_event_order_book",
        feature_names=[
            "pct_from_close",
            "mom_5m",
            "vol_30m",
            "range_pos",
            "spread_bps",
            "queue_depth",
            "market_data_latency_ms",
            "lag_expected_fill_delta_bps",
            "lag_latency_ms",
            "lag_slippage_bps",
            "market_micro_relative_volume_norm",
            "data_quality_quote_agreement_norm",
            "data_quality_quote_deviation_norm",
            "behavior_prior",
            "futures_specialist_vote",
            "futures_order_book_imbalance_norm",
            "futures_taker_imbalance_norm",
            "futures_basis_bps_norm",
            "futures_basis_divergence_norm",
            "futures_term_structure_norm",
            "futures_session_volume_profile_norm",
            "futures_negative_bias_norm",
            "futures_liquidation_risk_norm",
            "calendar_event_proximity_norm",
            "calendar_high_impact_24h_norm",
            "calendar_macro_event_norm",
            "ret_3",
            "ret_6",
            "futures_order_book_imbalance_ema_4",
            "futures_session_volume_profile_ema_4",
        ],
        runtime_feature_builder=_runtime_feature_vector,
        runtime_label_builder=_runtime_futures_event_label,
        mode_allowlist=_FUTURES_EVENT_MODES,
        symbol_allowlist=_FUTURES_EVENT_SYMBOLS,
        sample_filter=_runtime_sample_filter,
        confidence_builder=_runtime_confidence,
        min_confidence=0.44,
        lookback_days=45,
        window=18,
        horizon=4,
        min_samples=224,
        min_sequences=5,
        min_positive_samples=32,
        min_negative_samples=32,
        acted_prob_threshold=0.67,
        fallback_trainer=_train_synthetic,
        allow_fallback_on_insufficient_data=False,
        max_best_val_loss=0.6925,
        max_final_val_loss=0.7050,
        min_long_precision=0.52,
        min_short_precision=0.52,
        require_both_sides_precision=True,
        min_acted_accuracy=0.53,
        min_long_acted_count=4,
        min_short_acted_count=4,
        min_accuracy_lift_over_majority=0.02,
        min_precision_balance_score=0.30,
    )


if __name__ == "__main__":
    train_brain()
