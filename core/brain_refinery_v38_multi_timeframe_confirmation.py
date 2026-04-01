import numpy as np

from indicator_bot_common import ema, rolling_std, train_indicator_bot, train_runtime_indicator_bot
from runtime_training_common import (
    feature_ema,
    feature_std,
    future_max_drawdown,
    future_realized_vol,
    future_return,
    observation_feature,
    price_change,
)

_MTF_RUNTIME_MODES = [
    "shadow_equities",
    "shadow_aggressive_equities",
    "shadow_intraday_aggressive_equities",
    "shadow_swing_aggressive_equities",
    "shadow_dividend_equities",
]
_MTF_SYMBOLS = [
    "SPY",
    "QQQ",
    "IWM",
    "DIA",
    "XLK",
    "XLF",
    "XLE",
    "XLI",
    "XLV",
    "XLY",
    "XLP",
    "XLB",
    "XLU",
    "XLRE",
    "XLC",
    "SMH",
]


def downsample(x, step):
    y = np.zeros_like(x)
    for i in range(len(x)):
        j = (i // step) * step
        y[i] = x[j]
    return y


def build_features(panel):
    r = panel["ret"]

    ret_5 = r
    ret_30 = downsample(r, 6)
    ret_60 = downsample(r, 12)

    trend_5 = ema(ret_5, 8)
    trend_30 = ema(ret_30, 8)
    trend_60 = ema(ret_60, 8)

    confirm_30 = np.sign(trend_5) * np.sign(trend_30)
    confirm_60 = np.sign(trend_5) * np.sign(trend_60)
    disagreement = (confirm_30 < 0).astype(float) + (confirm_60 < 0).astype(float)
    noise = rolling_std(ret_5, 20)

    return np.stack([ret_5, trend_5, trend_30, trend_60, confirm_30, confirm_60, disagreement, noise], axis=1)


def _clip01(value):
    return float(np.clip(value, 0.0, 1.0))


def _centered01(value):
    return float((float(value) - 0.5) * 2.0)


def _quote_quality(obs):
    return _clip01(
        (0.65 * observation_feature(obs, "data_quality_quote_agreement_norm", 1.0))
        + (0.35 * (1.0 - observation_feature(obs, "data_quality_quote_deviation_norm", 0.0)))
    )


def _confirmation_support(obs):
    return _clip01(
        (0.18 * observation_feature(obs, "day_regime_trend_norm"))
        + (0.16 * observation_feature(obs, "day_regime_alignment_norm"))
        + (0.18 * observation_feature(obs, "swing_weekly_trend_confirm_norm"))
        + (0.16 * observation_feature(obs, "swing_regime_trend_norm"))
        + (0.12 * observation_feature(obs, "swing_regime_alignment_norm"))
        + (0.10 * observation_feature(obs, "lead_lag_alignment_norm"))
        + (0.06 * observation_feature(obs, "market_micro_trend_persistence_norm"))
        + (0.04 * _quote_quality(obs))
    )


def _confirmation_bias(obs):
    breadth_balance = observation_feature(obs, "breadth_advance_decline_norm") - observation_feature(
        obs, "breadth_risk_off_norm"
    )
    session_focus = max(
        observation_feature(obs, "day_session_open_norm"),
        observation_feature(obs, "day_session_power_hour_norm"),
    )
    return float(
        np.clip(
            (0.20 * observation_feature(obs, "behavior_prior"))
            + (0.16 * observation_feature(obs, "mom_15m") * 90.0)
            + (0.12 * observation_feature(obs, "mom_5m") * 110.0)
            + (0.12 * observation_feature(obs, "pct_from_close") * 110.0)
            + (0.12 * _centered01(observation_feature(obs, "market_micro_order_flow_imbalance_norm", 0.5)))
            + (0.10 * breadth_balance)
            + (0.10 * observation_feature(obs, "lead_lag_alignment_norm"))
            + (0.08 * session_focus),
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
            observation_feature(obs, "mom_15m"),
            observation_feature(obs, "vol_30m"),
            observation_feature(obs, "range_pos"),
            observation_feature(obs, "spread_bps"),
            observation_feature(obs, "queue_depth"),
            observation_feature(obs, "day_regime_trend_norm"),
            observation_feature(obs, "day_regime_alignment_norm"),
            observation_feature(obs, "swing_weekly_trend_confirm_norm"),
            observation_feature(obs, "swing_regime_trend_norm"),
            observation_feature(obs, "swing_regime_alignment_norm"),
            observation_feature(obs, "lead_lag_alignment_norm"),
            observation_feature(obs, "breadth_advance_decline_norm"),
            observation_feature(obs, "breadth_sector_dispersion_norm"),
            observation_feature(obs, "breadth_risk_off_norm"),
            observation_feature(obs, "market_micro_relative_volume_norm"),
            observation_feature(obs, "market_micro_trend_persistence_norm"),
            observation_feature(obs, "market_micro_order_flow_imbalance_norm"),
            observation_feature(obs, "options_vol_expectation_norm"),
            observation_feature(obs, "data_quality_quote_agreement_norm", 1.0),
            observation_feature(obs, "data_quality_quote_deviation_norm"),
            observation_feature(obs, "behavior_prior"),
            observation_feature(obs, "day_session_open_norm"),
            observation_feature(obs, "day_session_midday_norm"),
            observation_feature(obs, "day_session_power_hour_norm"),
            price_change(sequence, idx, 3),
            price_change(sequence, idx, 6),
            feature_std(sequence, idx, "pct_from_close", 6),
            feature_ema(sequence, idx, "behavior_prior", 4),
            feature_ema(sequence, idx, "day_regime_alignment_norm", 4),
        ],
        dtype=np.float32,
    )


def _runtime_sample_filter(sequence, idx, horizon):
    obs = sequence[idx]
    return (
        observation_feature(obs, "data_quality_quote_agreement_norm", 1.0) >= 0.78
        and observation_feature(obs, "data_quality_quote_deviation_norm", 0.0) <= 0.30
        and abs(observation_feature(obs, "spread_bps", 0.0)) <= 34.0
        and observation_feature(obs, "queue_depth", 0.0) >= 1.0
        and _confirmation_support(obs) >= 0.24
        and abs(_confirmation_bias(obs)) >= 0.10
    )


def _runtime_confidence(sequence, idx, horizon):
    obs = sequence[idx]
    support = _confirmation_support(obs)
    bias = _clip01(abs(_confirmation_bias(obs)) / 0.9)
    return (
        (0.34 * support)
        + (0.24 * bias)
        + (0.14 * _quote_quality(obs))
        + (0.14 * observation_feature(obs, "market_micro_relative_volume_norm"))
        + (0.14 * max(observation_feature(obs, "day_session_open_norm"), observation_feature(obs, "day_session_power_hour_norm")))
    )


def _runtime_confirmation_label(sequence, idx, horizon):
    obs = sequence[idx]
    support = _confirmation_support(obs)
    bias = _confirmation_bias(obs)
    if support < 0.24 or abs(bias) < 0.10:
        return None

    expected_up = bias >= 0.0
    fwd_ret = future_return(sequence, idx, horizon)
    realized = future_realized_vol(sequence, idx, horizon)
    drawdown = abs(future_max_drawdown(sequence, idx, horizon))
    signed_ret = fwd_ret if expected_up else -fwd_ret
    move_threshold = max(0.00045, 0.00110 - (0.00050 * support))
    if abs(fwd_ret) < move_threshold and realized < 0.018 and drawdown < 0.012:
        return None

    support_bonus = (
        (0.00095 * support)
        + (0.00030 * observation_feature(obs, "lead_lag_alignment_norm"))
        + (0.00020 * _quote_quality(obs))
    )
    penalty = (
        (0.26 * drawdown)
        + (0.18 * realized)
        + (0.00020 * observation_feature(obs, "breadth_sector_dispersion_norm"))
        + (0.00020 * observation_feature(obs, "breadth_risk_off_norm"))
    )
    success_score = signed_ret + support_bonus - penalty
    failure_score = (
        (-signed_ret)
        + (0.18 * realized)
        + (0.24 * drawdown)
        + (0.00020 * observation_feature(obs, "breadth_sector_dispersion_norm"))
    )
    if success_score >= 0.00045:
        return 1.0 if expected_up else 0.0
    if failure_score >= 0.00062:
        return 0.0 if expected_up else 1.0
    return None


def _train_synthetic():
    return train_indicator_bot(
        run_tag="brain_refinery_v38_multi_timeframe_confirmation",
        feature_names=["ret_5", "trend_5", "trend_30", "trend_60", "confirm_30", "confirm_60", "disagreement", "noise"],
        feature_builder=build_features,
    )


def train_brain():
    return train_runtime_indicator_bot(
        run_tag="brain_refinery_v38_multi_timeframe_confirmation",
        feature_names=[
            "pct_from_close",
            "mom_5m",
            "mom_15m",
            "vol_30m",
            "range_pos",
            "spread_bps",
            "queue_depth",
            "day_regime_trend_norm",
            "day_regime_alignment_norm",
            "swing_weekly_trend_confirm_norm",
            "swing_regime_trend_norm",
            "swing_regime_alignment_norm",
            "lead_lag_alignment_norm",
            "breadth_advance_decline_norm",
            "breadth_sector_dispersion_norm",
            "breadth_risk_off_norm",
            "market_micro_relative_volume_norm",
            "market_micro_trend_persistence_norm",
            "market_micro_order_flow_imbalance_norm",
            "options_vol_expectation_norm",
            "data_quality_quote_agreement_norm",
            "data_quality_quote_deviation_norm",
            "behavior_prior",
            "day_session_open_norm",
            "day_session_midday_norm",
            "day_session_power_hour_norm",
            "ret_3",
            "ret_6",
            "pct_from_close_std_6",
            "behavior_prior_ema_4",
            "day_regime_alignment_norm_ema_4",
        ],
        runtime_feature_builder=_runtime_feature_vector,
        runtime_label_builder=_runtime_confirmation_label,
        mode_allowlist=_MTF_RUNTIME_MODES,
        symbol_allowlist=_MTF_SYMBOLS,
        sample_filter=_runtime_sample_filter,
        confidence_builder=_runtime_confidence,
        min_confidence=0.35,
        lookback_days=30,
        window=24,
        horizon=6,
        min_samples=320,
        min_sequences=8,
        acted_prob_threshold=0.67,
        fallback_trainer=_train_synthetic,
        allow_fallback_on_insufficient_data=False,
        max_best_val_loss=0.6900,
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
