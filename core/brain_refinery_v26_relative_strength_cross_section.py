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

_RS_RUNTIME_MODES = [
    "shadow_equities",
    "shadow_aggressive_equities",
    "shadow_conservative_equities",
    "shadow_dividend_equities",
    "shadow_swing_aggressive_equities",
]
_RS_SYMBOLS = [
    "SPY",
    "QQQ",
    "DIA",
    "IWM",
    "XLK",
    "XLF",
    "XLI",
    "XLE",
    "XLV",
    "XLP",
    "XLU",
    "XLRE",
    "SMH",
    "SOXX",
    "SCHD",
    "VIG",
    "DGRO",
]


def build_features(panel):
    c = panel["close"]
    b = panel["bench_close"]
    r = panel["ret"]
    rb = panel["bench_ret"]

    rs = c / (b + 1e-8)
    rs_fast = ema(rs, 8)
    rs_slow = ema(rs, 34)
    rs_spread = (rs_fast - rs_slow) / (np.abs(rs_slow) + 1e-8)
    rel_impulse = np.diff(rs_spread, prepend=rs_spread[0])
    alpha = r - rb
    alpha_ema = ema(alpha, 12)
    alpha_vol = rolling_std(alpha, 20)
    bench_trend = ema(rb, 20)

    return np.stack([r, rb, alpha, alpha_ema, alpha_vol, rs_spread, rel_impulse, bench_trend], axis=1)


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
            observation_feature(obs, "ctx_VIX_X_pct_from_close"),
            observation_feature(obs, "ctx_UUP_pct_from_close"),
            observation_feature(obs, "breadth_advance_decline_norm"),
            observation_feature(obs, "breadth_risk_off_norm"),
            observation_feature(obs, "market_micro_relative_volume_norm"),
            observation_feature(obs, "market_micro_trend_persistence_norm"),
            observation_feature(obs, "market_micro_order_flow_imbalance_norm"),
            observation_feature(obs, "data_quality_quote_agreement_norm", 1.0),
            observation_feature(obs, "data_quality_quote_deviation_norm"),
            observation_feature(obs, "behavior_prior"),
            observation_feature(obs, "day_regime_trend_norm"),
            observation_feature(obs, "day_regime_alignment_norm"),
            observation_feature(obs, "swing_sector_relative_strength_norm"),
            observation_feature(obs, "swing_weekly_trend_confirm_norm"),
            observation_feature(obs, "swing_regime_trend_norm"),
            observation_feature(obs, "swing_regime_alignment_norm"),
            observation_feature(obs, "capital_flow_signed_scaled"),
            observation_feature(obs, "flow_direction_signed"),
            observation_feature(obs, "flow_risk_on_norm"),
            observation_feature(obs, "lead_lag_alignment_norm"),
            observation_feature(obs, "options_vol_expectation_norm"),
            observation_feature(obs, "options_negative_bias_norm"),
            price_change(sequence, idx, 3),
            price_change(sequence, idx, 6),
            feature_std(sequence, idx, "pct_from_close", 6),
            feature_std(sequence, idx, "vol_30m", 8),
            feature_ema(sequence, idx, "behavior_prior", 4),
            feature_ema(sequence, idx, "swing_sector_relative_strength_norm", 4),
        ],
        dtype=np.float32,
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


def _relative_strength_support(obs):
    return _clip01(
        (0.18 * observation_feature(obs, "day_regime_trend_norm"))
        + (0.10 * observation_feature(obs, "day_regime_alignment_norm"))
        + (0.17 * observation_feature(obs, "swing_sector_relative_strength_norm"))
        + (0.14 * observation_feature(obs, "swing_weekly_trend_confirm_norm"))
        + (0.11 * observation_feature(obs, "swing_regime_trend_norm"))
        + (0.08 * observation_feature(obs, "swing_regime_alignment_norm"))
        + (0.08 * observation_feature(obs, "market_micro_trend_persistence_norm"))
        + (0.05 * observation_feature(obs, "flow_risk_on_norm"))
        + (0.05 * observation_feature(obs, "market_micro_relative_volume_norm"))
        + (0.04 * _quote_quality(obs))
    )


def _direction_bias(obs):
    return float(
        (0.22 * observation_feature(obs, "behavior_prior"))
        + (0.17 * observation_feature(obs, "flow_direction_signed"))
        + (0.14 * observation_feature(obs, "capital_flow_signed_scaled"))
        + (0.14 * observation_feature(obs, "breadth_advance_decline_norm"))
        + (0.11 * observation_feature(obs, "mom_15m") * 90.0)
        + (0.08 * observation_feature(obs, "pct_from_close") * 110.0)
        + (0.08 * _centered01(observation_feature(obs, "range_pos")))
        + (0.06 * _centered01(observation_feature(obs, "swing_sector_relative_strength_norm")))
    )


def _runtime_sample_filter(sequence, idx, horizon):
    obs = sequence[idx]
    return (
        observation_feature(obs, "data_quality_quote_agreement_norm", 1.0) >= 0.80
        and observation_feature(obs, "data_quality_quote_deviation_norm", 0.0) <= 0.24
        and abs(observation_feature(obs, "spread_bps")) <= 30.0
        and observation_feature(obs, "queue_depth", 0.0) >= 1.0
        and _relative_strength_support(obs) >= 0.26
        and abs(_direction_bias(obs)) >= 0.15
    )


def _runtime_confidence(sequence, idx, horizon):
    obs = sequence[idx]
    support = _relative_strength_support(obs)
    bias = _clip01(abs(_direction_bias(obs)) / 0.9)
    quote = _quote_quality(obs)
    flow = _clip01(
        (0.55 * observation_feature(obs, "market_micro_relative_volume_norm"))
        + (0.45 * observation_feature(obs, "market_micro_trend_persistence_norm"))
    )
    return (
        (0.34 * support)
        + (0.24 * bias)
        + (0.16 * flow)
        + (0.12 * _clip01(observation_feature(obs, "lead_lag_alignment_norm")))
        + (0.08 * quote)
        + (0.06 * _clip01(1.0 - observation_feature(obs, "breadth_risk_off_norm")))
    )


def _runtime_relative_strength_label(sequence, idx, horizon):
    obs = sequence[idx]
    support = _relative_strength_support(obs)
    bias = _direction_bias(obs)
    if support < 0.26 or abs(bias) < 0.15:
        return None

    expected_up = bias >= 0.0
    fwd_ret = future_return(sequence, idx, horizon)
    realized = future_realized_vol(sequence, idx, horizon)
    drawdown = abs(future_max_drawdown(sequence, idx, horizon))
    signed_ret = fwd_ret if expected_up else -fwd_ret
    move_threshold = max(0.00046, 0.00102 - (0.00035 * support))
    if abs(fwd_ret) < move_threshold and realized < 0.020 and drawdown < 0.011:
        return None

    support_bonus = (
        (0.00095 * support)
        + (0.00030 * _clip01(abs(bias) / 0.9))
        + (0.00015 * observation_feature(obs, "flow_risk_on_norm"))
    )
    penalty = (
        (0.34 * drawdown)
        + (0.22 * realized)
        + (0.00025 * observation_feature(obs, "breadth_risk_off_norm"))
        + (0.00015 * observation_feature(obs, "options_negative_bias_norm"))
    )
    success_score = signed_ret + support_bonus - penalty
    failure_score = (-signed_ret) + (0.20 * realized) + (0.30 * drawdown)
    failure_score += (
        (0.00020 * observation_feature(obs, "breadth_risk_off_norm"))
        + (0.00015 * observation_feature(obs, "options_negative_bias_norm"))
    )

    if success_score >= 0.00046:
        return 1.0 if expected_up else 0.0
    if failure_score >= 0.00060:
        return 0.0 if expected_up else 1.0
    return None


def _train_synthetic():
    return train_indicator_bot(
        run_tag="brain_refinery_v26_relative_strength_cross_section",
        feature_names=["ret", "bench_ret", "alpha", "alpha_ema12", "alpha_vol20", "rs_spread", "rel_impulse", "bench_trend"],
        feature_builder=build_features,
        window=48,
        horizon=5,
    )


def train_brain():
    return train_runtime_indicator_bot(
        run_tag="brain_refinery_v26_relative_strength_cross_section",
        feature_names=[
            "pct_from_close",
            "mom_5m",
            "mom_15m",
            "vol_30m",
            "range_pos",
            "spread_bps",
            "queue_depth",
            "ctx_VIX_X_pct_from_close",
            "ctx_UUP_pct_from_close",
            "breadth_advance_decline_norm",
            "breadth_risk_off_norm",
            "market_micro_relative_volume_norm",
            "market_micro_trend_persistence_norm",
            "market_micro_order_flow_imbalance_norm",
            "data_quality_quote_agreement_norm",
            "data_quality_quote_deviation_norm",
            "behavior_prior",
            "day_regime_trend_norm",
            "day_regime_alignment_norm",
            "swing_sector_relative_strength_norm",
            "swing_weekly_trend_confirm_norm",
            "swing_regime_trend_norm",
            "swing_regime_alignment_norm",
            "capital_flow_signed_scaled",
            "flow_direction_signed",
            "flow_risk_on_norm",
            "lead_lag_alignment_norm",
            "options_vol_expectation_norm",
            "options_negative_bias_norm",
            "ret_3",
            "ret_6",
            "pct_from_close_std_6",
            "vol_30m_std_8",
            "behavior_prior_ema_4",
            "swing_sector_relative_strength_norm_ema_4",
        ],
        runtime_feature_builder=_runtime_feature_vector,
        runtime_label_builder=_runtime_relative_strength_label,
        mode_allowlist=_RS_RUNTIME_MODES,
        symbol_allowlist=_RS_SYMBOLS,
        sample_filter=_runtime_sample_filter,
        confidence_builder=_runtime_confidence,
        min_confidence=0.36,
        lookback_days=45,
        window=24,
        horizon=6,
        min_samples=320,
        min_sequences=8,
        min_positive_samples=48,
        min_negative_samples=48,
        acted_prob_threshold=0.68,
        fallback_trainer=_train_synthetic,
        allow_fallback_on_insufficient_data=False,
        max_best_val_loss=0.6900,
        max_final_val_loss=0.7040,
        min_long_precision=0.52,
        min_short_precision=0.52,
        require_both_sides_precision=True,
        min_acted_accuracy=0.52,
        min_long_acted_count=6,
        min_short_acted_count=6,
        min_accuracy_lift_over_majority=0.01,
        min_precision_balance_score=0.40,
    )


if __name__ == "__main__":
    train_brain()
