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

_LONG_TREND_RUNTIME_MODES = [
    "shadow_equities",
    "shadow_conservative_equities",
    "shadow_dividend_equities",
    "shadow_swing_aggressive_equities",
]
_LONG_TREND_SYMBOLS = [
    "SPY",
    "QQQ",
    "DIA",
    "IWM",
    "SCHD",
    "VIG",
    "DGRO",
    "XLK",
    "XLF",
    "XLI",
    "XLV",
    "XLP",
    "XLU",
    "XLRE",
    "XLE",
]


def hold_sample(x, step):
    out = np.zeros_like(x)
    for i in range(len(x)):
        j = (i // step) * step
        out[i] = x[j]
    return out


def build_features(panel):
    c = panel["close"]
    r = panel["ret"]

    r_15 = hold_sample(r, 3)
    r_30 = hold_sample(r, 6)
    r_60 = hold_sample(r, 12)

    t_15 = ema(r_15, 10)
    t_30 = ema(r_30, 10)
    t_60 = ema(r_60, 10)

    long_drift = ema(np.diff(c, prepend=c[0]) / (c + 1e-8), 30)
    regime_strength = (np.abs(t_30) + np.abs(t_60)) / (rolling_std(r, 30) + 1e-8)

    align_15_60 = np.sign(t_15) * np.sign(t_60)
    align_30_60 = np.sign(t_30) * np.sign(t_60)

    return np.stack(
        [
            r,
            t_15,
            t_30,
            t_60,
            long_drift,
            regime_strength,
            align_15_60,
            align_30_60,
        ],
        axis=1,
    )


def _clip01(value):
    return float(np.clip(value, 0.0, 1.0))


def _centered01(value):
    return float((float(value) - 0.5) * 2.0)


def _quote_quality(obs):
    return _clip01(
        (0.65 * observation_feature(obs, "data_quality_quote_agreement_norm", 1.0))
        + (0.35 * (1.0 - observation_feature(obs, "data_quality_quote_deviation_norm", 0.0)))
    )


def _long_trend_support(obs):
    return _clip01(
        (0.18 * observation_feature(obs, "day_regime_trend_norm"))
        + (0.12 * observation_feature(obs, "day_regime_alignment_norm"))
        + (0.18 * observation_feature(obs, "swing_sector_relative_strength_norm"))
        + (0.18 * observation_feature(obs, "swing_weekly_trend_confirm_norm"))
        + (0.16 * observation_feature(obs, "swing_regime_trend_norm"))
        + (0.10 * observation_feature(obs, "swing_regime_alignment_norm"))
        + (0.04 * observation_feature(obs, "lead_lag_alignment_norm"))
        + (0.04 * _quote_quality(obs))
    )


def _long_trend_bias(obs):
    breadth_balance = observation_feature(obs, "breadth_advance_decline_norm") - observation_feature(
        obs, "breadth_risk_off_norm"
    )
    return float(
        np.clip(
            (0.20 * observation_feature(obs, "behavior_prior"))
            + (0.16 * observation_feature(obs, "capital_flow_signed_scaled"))
            + (0.14 * observation_feature(obs, "flow_direction_signed"))
            + (0.14 * observation_feature(obs, "mom_15m") * 85.0)
            + (0.10 * observation_feature(obs, "pct_from_close") * 90.0)
            + (0.10 * breadth_balance)
            + (0.08 * _centered01(observation_feature(obs, "swing_sector_relative_strength_norm", 0.5)))
            + (0.08 * observation_feature(obs, "lead_lag_alignment_norm")),
            -1.0,
            1.0,
        )
    )


def _runtime_feature_vector(sequence, idx):
    obs = sequence[idx]
    return np.asarray(
        [
            observation_feature(obs, "pct_from_close"),
            observation_feature(obs, "mom_15m"),
            observation_feature(obs, "vol_30m"),
            observation_feature(obs, "range_pos"),
            observation_feature(obs, "spread_bps"),
            observation_feature(obs, "queue_depth"),
            observation_feature(obs, "market_micro_relative_volume_norm"),
            observation_feature(obs, "market_micro_trend_persistence_norm"),
            observation_feature(obs, "day_regime_trend_norm"),
            observation_feature(obs, "day_regime_alignment_norm"),
            observation_feature(obs, "swing_sector_relative_strength_norm"),
            observation_feature(obs, "swing_weekly_trend_confirm_norm"),
            observation_feature(obs, "swing_regime_trend_norm"),
            observation_feature(obs, "swing_regime_alignment_norm"),
            observation_feature(obs, "lead_lag_alignment_norm"),
            observation_feature(obs, "breadth_advance_decline_norm"),
            observation_feature(obs, "breadth_risk_off_norm"),
            observation_feature(obs, "options_vol_expectation_norm"),
            observation_feature(obs, "data_quality_quote_agreement_norm", 1.0),
            observation_feature(obs, "data_quality_quote_deviation_norm"),
            observation_feature(obs, "behavior_prior"),
            observation_feature(obs, "capital_flow_signed_scaled"),
            observation_feature(obs, "flow_direction_signed"),
            observation_feature(obs, "flow_risk_on_norm"),
            price_change(sequence, idx, 3),
            price_change(sequence, idx, 6),
            feature_std(sequence, idx, "pct_from_close", 8),
            feature_ema(sequence, idx, "behavior_prior", 4),
            feature_ema(sequence, idx, "swing_regime_trend_norm", 5),
        ],
        dtype=np.float32,
    )


def _runtime_sample_filter(sequence, idx, horizon):
    obs = sequence[idx]
    return (
        observation_feature(obs, "data_quality_quote_agreement_norm", 1.0) >= 0.80
        and observation_feature(obs, "data_quality_quote_deviation_norm", 0.0) <= 0.26
        and abs(observation_feature(obs, "spread_bps", 0.0)) <= 28.0
        and observation_feature(obs, "queue_depth", 0.0) >= 1.0
        and _long_trend_support(obs) >= 0.24
        and abs(_long_trend_bias(obs)) >= 0.12
    )


def _runtime_confidence(sequence, idx, horizon):
    obs = sequence[idx]
    support = _long_trend_support(obs)
    bias = _clip01(abs(_long_trend_bias(obs)) / 0.9)
    return (
        (0.34 * support)
        + (0.22 * bias)
        + (0.16 * _quote_quality(obs))
        + (0.16 * observation_feature(obs, "flow_risk_on_norm"))
        + (0.12 * observation_feature(obs, "market_micro_trend_persistence_norm"))
    )


def _runtime_long_trend_label(sequence, idx, horizon):
    obs = sequence[idx]
    support = _long_trend_support(obs)
    bias = _long_trend_bias(obs)
    if support < 0.24 or abs(bias) < 0.12:
        return None

    expected_up = bias >= 0.0
    fwd_ret = future_return(sequence, idx, horizon)
    realized = future_realized_vol(sequence, idx, horizon)
    drawdown = abs(future_max_drawdown(sequence, idx, horizon))
    signed_ret = fwd_ret if expected_up else -fwd_ret
    move_threshold = max(0.00055, 0.00130 - (0.00055 * support))
    if abs(fwd_ret) < move_threshold and realized < 0.018 and drawdown < 0.012:
        return None

    support_bonus = (
        (0.0010 * support)
        + (0.00030 * _quote_quality(obs))
        + (0.00025 * observation_feature(obs, "flow_risk_on_norm"))
    )
    penalty = (
        (0.26 * drawdown)
        + (0.20 * realized)
        + (0.00020 * observation_feature(obs, "breadth_risk_off_norm"))
        + (0.00015 * observation_feature(obs, "options_vol_expectation_norm"))
    )
    success_score = signed_ret + support_bonus - penalty
    failure_score = (
        (-signed_ret)
        + (0.18 * realized)
        + (0.24 * drawdown)
        + (0.00020 * observation_feature(obs, "breadth_risk_off_norm"))
    )
    if success_score >= 0.00050:
        return 1.0 if expected_up else 0.0
    if failure_score >= 0.00072:
        return 0.0 if expected_up else 1.0
    return None


def _train_synthetic():
    return train_indicator_bot(
        run_tag="brain_refinery_v41_long_interval_trend",
        feature_names=[
            "ret",
            "trend_15",
            "trend_30",
            "trend_60",
            "long_drift",
            "regime_strength",
            "align_15_60",
            "align_30_60",
        ],
        feature_builder=build_features,
        window=36,
        horizon=4,
    )


def train_brain():
    return train_runtime_indicator_bot(
        run_tag="brain_refinery_v41_long_interval_trend",
        feature_names=[
            "pct_from_close",
            "mom_15m",
            "vol_30m",
            "range_pos",
            "spread_bps",
            "queue_depth",
            "market_micro_relative_volume_norm",
            "market_micro_trend_persistence_norm",
            "day_regime_trend_norm",
            "day_regime_alignment_norm",
            "swing_sector_relative_strength_norm",
            "swing_weekly_trend_confirm_norm",
            "swing_regime_trend_norm",
            "swing_regime_alignment_norm",
            "lead_lag_alignment_norm",
            "breadth_advance_decline_norm",
            "breadth_risk_off_norm",
            "options_vol_expectation_norm",
            "data_quality_quote_agreement_norm",
            "data_quality_quote_deviation_norm",
            "behavior_prior",
            "capital_flow_signed_scaled",
            "flow_direction_signed",
            "flow_risk_on_norm",
            "ret_3",
            "ret_6",
            "pct_from_close_std_8",
            "behavior_prior_ema_4",
            "swing_regime_trend_norm_ema_5",
        ],
        runtime_feature_builder=_runtime_feature_vector,
        runtime_label_builder=_runtime_long_trend_label,
        mode_allowlist=_LONG_TREND_RUNTIME_MODES,
        symbol_allowlist=_LONG_TREND_SYMBOLS,
        sample_filter=_runtime_sample_filter,
        confidence_builder=_runtime_confidence,
        min_confidence=0.35,
        lookback_days=30,
        window=24,
        horizon=8,
        min_samples=288,
        min_sequences=6,
        acted_prob_threshold=0.68,
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
