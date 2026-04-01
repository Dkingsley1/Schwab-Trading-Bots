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

_SWING_RUNTIME_MODES = [
    "shadow_equities",
    "shadow_swing_aggressive_equities",
    "shadow_conservative_equities",
    "shadow_dividend_equities",
]
_SWING_SYMBOLS = [
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
        out[i] = x[(i // step) * step]
    return out


def build_features(panel):
    r = panel["ret"]
    c = panel["close"]

    d2 = hold_sample(r, 16)
    d5 = hold_sample(r, 40)
    m2 = ema(d2, 8)
    m5 = ema(d5, 8)
    drift = ema(np.diff(c, prepend=c[0]) / (c + 1e-8), 30)
    vol = rolling_std(r, 30)
    align = np.sign(m2) * np.sign(m5)

    return np.stack([r, m2, m5, drift, vol, align], axis=1)


def _clip01(value):
    return float(np.clip(value, 0.0, 1.0))


def _centered01(value):
    return float((float(value) - 0.5) * 2.0)


def _quote_quality(obs):
    return _clip01(
        (0.65 * observation_feature(obs, "data_quality_quote_agreement_norm", 1.0))
        + (0.35 * (1.0 - observation_feature(obs, "data_quality_quote_deviation_norm", 0.0)))
    )


def _swing_support(obs):
    return _clip01(
        (0.18 * observation_feature(obs, "day_regime_trend_norm"))
        + (0.14 * observation_feature(obs, "day_regime_alignment_norm"))
        + (0.18 * observation_feature(obs, "swing_sector_relative_strength_norm"))
        + (0.18 * observation_feature(obs, "swing_weekly_trend_confirm_norm"))
        + (0.16 * observation_feature(obs, "swing_regime_trend_norm"))
        + (0.08 * observation_feature(obs, "swing_regime_alignment_norm"))
        + (0.04 * observation_feature(obs, "lead_lag_alignment_norm"))
        + (0.04 * _quote_quality(obs))
    )


def _swing_bias(obs):
    breadth_balance = observation_feature(obs, "breadth_advance_decline_norm") - observation_feature(
        obs, "breadth_risk_off_norm"
    )
    return float(
        np.clip(
            (0.20 * observation_feature(obs, "behavior_prior"))
            + (0.18 * observation_feature(obs, "capital_flow_signed_scaled"))
            + (0.14 * observation_feature(obs, "flow_direction_signed"))
            + (0.12 * observation_feature(obs, "flow_risk_on_norm"))
            + (0.14 * observation_feature(obs, "mom_15m") * 85.0)
            + (0.10 * observation_feature(obs, "pct_from_close") * 90.0)
            + (0.06 * breadth_balance)
            + (0.06 * _centered01(observation_feature(obs, "swing_sector_relative_strength_norm", 0.5))),
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
        observation_feature(obs, "data_quality_quote_agreement_norm", 1.0) >= 0.82
        and observation_feature(obs, "data_quality_quote_deviation_norm", 0.0) <= 0.22
        and abs(observation_feature(obs, "spread_bps", 0.0)) <= 24.0
        and observation_feature(obs, "queue_depth", 0.0) >= 1.0
        and observation_feature(obs, "swing_weekly_trend_confirm_norm") >= 0.60
        and _swing_support(obs) >= 0.30
        and abs(_swing_bias(obs)) >= 0.15
    )


def _runtime_confidence(sequence, idx, horizon):
    obs = sequence[idx]
    return (
        (0.34 * _swing_support(obs))
        + (0.22 * _clip01(abs(_swing_bias(obs)) / 0.9))
        + (0.16 * _quote_quality(obs))
        + (0.16 * observation_feature(obs, "flow_risk_on_norm"))
        + (0.12 * observation_feature(obs, "market_micro_trend_persistence_norm"))
    )


def _runtime_swing_label(sequence, idx, horizon):
    obs = sequence[idx]
    support = _swing_support(obs)
    bias = _swing_bias(obs)
    if support < 0.30 or abs(bias) < 0.15:
        return None

    expected_up = bias >= 0.0
    fwd_ret = future_return(sequence, idx, horizon)
    realized = future_realized_vol(sequence, idx, horizon)
    drawdown = abs(future_max_drawdown(sequence, idx, horizon))
    signed_ret = fwd_ret if expected_up else -fwd_ret
    move_threshold = max(0.00075, 0.00145 - (0.00050 * support))
    if abs(fwd_ret) < move_threshold and realized < 0.016 and drawdown < 0.010:
        return None

    success_score = (
        signed_ret
        + (0.0010 * support)
        + (0.00025 * _quote_quality(obs))
        + (0.00020 * observation_feature(obs, "flow_risk_on_norm"))
        - (0.26 * drawdown)
        - (0.18 * realized)
    )
    failure_score = (
        (-signed_ret)
        + (0.18 * realized)
        + (0.24 * drawdown)
        + (0.00020 * observation_feature(obs, "breadth_risk_off_norm"))
    )
    if success_score >= 0.00070:
        return 1.0 if expected_up else 0.0
    if failure_score >= 0.00090:
        return 0.0 if expected_up else 1.0
    return None


def _train_synthetic():
    return train_indicator_bot(
        run_tag="brain_refinery_v46_swing_2d_5d",
        feature_names=["ret", "mom_2d", "mom_5d", "drift", "vol", "align"],
        feature_builder=build_features,
        window=48,
        horizon=6,
    )


def train_brain():
    return train_runtime_indicator_bot(
        run_tag="brain_refinery_v46_swing_2d_5d",
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
        runtime_label_builder=_runtime_swing_label,
        mode_allowlist=_SWING_RUNTIME_MODES,
        symbol_allowlist=_SWING_SYMBOLS,
        sample_filter=_runtime_sample_filter,
        confidence_builder=_runtime_confidence,
        min_confidence=0.40,
        lookback_days=45,
        window=24,
        horizon=8,
        min_samples=288,
        min_sequences=6,
        min_positive_samples=24,
        min_negative_samples=24,
        acted_prob_threshold=0.68,
        fallback_trainer=_train_synthetic,
        allow_fallback_on_insufficient_data=False,
        max_best_val_loss=0.6900,
        max_final_val_loss=0.7050,
        min_long_precision=0.52,
        min_short_precision=0.52,
        require_both_sides_precision=True,
        min_acted_accuracy=0.53,
        min_long_acted_count=4,
        min_short_acted_count=4,
        min_accuracy_lift_over_majority=0.02,
        min_precision_balance_score=0.35,
    )


if __name__ == "__main__":
    train_brain()
