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
    symbol_role_features,
)

_SECTOR_ROLE_MAP = {
    "growth": ["QQQ", "XLK", "XLC", "SMH", "SOXX"],
    "cyclical": ["SPY", "DIA", "IWM", "XLF", "XLI", "XLB", "XLE", "XLY", "KRE"],
    "defensive": ["XLV", "XLP", "XLU", "XLRE", "SCHD", "VIG", "DGRO"],
}
_EXPANDED_SECTOR_SYMBOLS = sorted({sym for values in _SECTOR_ROLE_MAP.values() for sym in values})
_SECTOR_ROTATION_MODES = [
    "shadow_equities",
    "shadow_aggressive_equities",
    "shadow_swing_aggressive_equities",
    "shadow_conservative_equities",
    "shadow_dividend_equities",
]


def hold_sample(x, step):
    out = np.zeros_like(x)
    for i in range(len(x)):
        out[i] = x[(i // step) * step]
    return out


def build_features(panel):
    r = panel["ret"]
    rb = panel["bench_ret"]

    alpha = r - rb
    rs_fast = ema(alpha, 8)
    rs_slow = ema(alpha, 30)
    rs_spread = rs_fast - rs_slow

    week_alpha = hold_sample(alpha, 78)
    month_alpha = hold_sample(alpha, 390)
    rot_pressure = ema(week_alpha + month_alpha, 10)
    rs_noise = rolling_std(alpha, 30)

    return np.stack([r, rb, alpha, rs_fast, rs_slow, rs_spread, rot_pressure, rs_noise], axis=1)


def _clip01(value):
    return float(np.clip(value, 0.0, 1.0))


def _centered01(value):
    return float((float(value) - 0.5) * 2.0)


def _quote_quality(obs):
    return _clip01(
        (0.65 * observation_feature(obs, "data_quality_quote_agreement_norm", 1.0))
        + (0.35 * (1.0 - observation_feature(obs, "data_quality_quote_deviation_norm", 0.0)))
    )


def _risk_on_signal(obs):
    breadth_strength = _clip01((observation_feature(obs, "breadth_advance_decline_norm") + 1.0) / 2.0)
    return _clip01(
        (0.20 * observation_feature(obs, "swing_sector_relative_strength_norm"))
        + (0.18 * observation_feature(obs, "swing_weekly_trend_confirm_norm"))
        + (0.12 * observation_feature(obs, "swing_regime_trend_norm"))
        + (0.10 * observation_feature(obs, "day_regime_alignment_norm"))
        + (0.12 * observation_feature(obs, "flow_risk_on_norm"))
        + (0.12 * _clip01(max(observation_feature(obs, "capital_flow_signed_scaled"), 0.0)))
        + (0.08 * breadth_strength)
        + (0.08 * observation_feature(obs, "market_micro_trend_persistence_norm"))
    )


def _risk_off_signal(obs):
    return _clip01(
        (0.26 * observation_feature(obs, "breadth_risk_off_norm"))
        + (0.22 * observation_feature(obs, "options_negative_bias_norm"))
        + (0.18 * observation_feature(obs, "options_vol_expectation_norm"))
        + (0.14 * _clip01(max(-observation_feature(obs, "capital_flow_signed_scaled"), 0.0)))
        + (0.10 * observation_feature(obs, "breadth_sector_dispersion_norm"))
        + (0.10 * _clip01(1.0 - observation_feature(obs, "market_micro_trend_persistence_norm")))
    )


def _role_supports(obs, roles):
    risk_on = _risk_on_signal(obs)
    risk_off = _risk_off_signal(obs)
    quality = _quote_quality(obs)
    trend = _clip01(
        (0.55 * observation_feature(obs, "swing_regime_trend_norm"))
        + (0.45 * observation_feature(obs, "day_regime_trend_norm"))
    )
    role_weight = max(sum(float(value) for value in roles.values()), 1.0)
    support = (
        (roles["role_growth"] * _clip01((0.55 * risk_on) + (0.25 * trend) + (0.20 * quality)))
        + (roles["role_cyclical"] * _clip01((0.48 * risk_on) + (0.22 * trend) + (0.15 * quality) + (0.15 * observation_feature(obs, "flow_risk_on_norm"))))
        + (roles["role_defensive"] * _clip01((0.50 * risk_off) + (0.20 * quality) + (0.15 * trend) + (0.15 * observation_feature(obs, "breadth_risk_off_norm"))))
    ) / role_weight
    headwind = (
        (roles["role_growth"] * _clip01((0.55 * risk_off) + (0.25 * (1.0 - trend)) + (0.20 * (1.0 - quality))))
        + (roles["role_cyclical"] * _clip01((0.48 * risk_off) + (0.22 * (1.0 - trend)) + (0.15 * (1.0 - quality)) + (0.15 * observation_feature(obs, "breadth_risk_off_norm"))))
        + (roles["role_defensive"] * _clip01((0.50 * risk_on) + (0.20 * (1.0 - quality)) + (0.15 * (1.0 - trend)) + (0.15 * observation_feature(obs, "flow_risk_on_norm"))))
    ) / role_weight
    return _clip01(support), _clip01(headwind)


def _rotation_bias(obs):
    breadth_balance = observation_feature(obs, "breadth_advance_decline_norm") - observation_feature(
        obs, "breadth_risk_off_norm"
    )
    return float(
        np.clip(
            (0.20 * observation_feature(obs, "behavior_prior"))
            + (0.18 * observation_feature(obs, "flow_direction_signed"))
            + (0.14 * observation_feature(obs, "capital_flow_signed_scaled"))
            + (0.14 * _centered01(observation_feature(obs, "swing_sector_relative_strength_norm", 0.5)))
            + (0.12 * breadth_balance)
            + (0.10 * observation_feature(obs, "mom_15m") * 85.0)
            + (0.12 * observation_feature(obs, "pct_from_close") * 85.0),
            -1.0,
            1.0,
        )
    )


def _runtime_feature_vector(sequence, idx):
    obs = sequence[idx]
    roles = symbol_role_features(str(obs.get("symbol") or ""), _SECTOR_ROLE_MAP)
    return np.asarray(
        [
            observation_feature(obs, "pct_from_close"),
            observation_feature(obs, "mom_15m"),
            observation_feature(obs, "vol_30m"),
            observation_feature(obs, "range_pos"),
            observation_feature(obs, "spread_bps"),
            observation_feature(obs, "queue_depth"),
            observation_feature(obs, "day_regime_trend_norm"),
            observation_feature(obs, "day_regime_alignment_norm"),
            observation_feature(obs, "swing_sector_relative_strength_norm"),
            observation_feature(obs, "swing_weekly_trend_confirm_norm"),
            observation_feature(obs, "swing_regime_trend_norm"),
            observation_feature(obs, "swing_regime_alignment_norm"),
            observation_feature(obs, "breadth_sector_dispersion_norm"),
            observation_feature(obs, "breadth_advance_decline_norm"),
            observation_feature(obs, "breadth_risk_off_norm"),
            observation_feature(obs, "market_micro_relative_volume_norm"),
            observation_feature(obs, "market_micro_trend_persistence_norm"),
            observation_feature(obs, "options_vol_expectation_norm"),
            observation_feature(obs, "options_negative_bias_norm"),
            observation_feature(obs, "capital_flow_signed_scaled"),
            observation_feature(obs, "flow_direction_signed"),
            observation_feature(obs, "flow_risk_on_norm"),
            observation_feature(obs, "data_quality_quote_agreement_norm", 1.0),
            observation_feature(obs, "data_quality_quote_deviation_norm"),
            observation_feature(obs, "behavior_prior"),
            price_change(sequence, idx, 3),
            price_change(sequence, idx, 6),
            feature_std(sequence, idx, "pct_from_close", 8),
            feature_ema(sequence, idx, "swing_sector_relative_strength_norm", 5),
            feature_ema(sequence, idx, "flow_risk_on_norm", 4),
            roles["role_growth"],
            roles["role_cyclical"],
            roles["role_defensive"],
        ],
        dtype=np.float32,
    )


def _runtime_sample_filter(sequence, idx, horizon):
    obs = sequence[idx]
    roles = symbol_role_features(str(obs.get("symbol") or ""), _SECTOR_ROLE_MAP)
    support, headwind = _role_supports(obs, roles)
    return (
        sum(float(value) for value in roles.values()) > 0.0
        and observation_feature(obs, "data_quality_quote_agreement_norm", 1.0) >= 0.78
        and observation_feature(obs, "data_quality_quote_deviation_norm", 0.0) <= 0.28
        and abs(observation_feature(obs, "spread_bps", 0.0)) <= 35.0
        and observation_feature(obs, "queue_depth", 0.0) >= 1.0
        and max(support, headwind) >= 0.18
        and abs(_rotation_bias(obs)) >= 0.08
    )


def _runtime_confidence(sequence, idx, horizon):
    obs = sequence[idx]
    roles = symbol_role_features(str(obs.get("symbol") or ""), _SECTOR_ROLE_MAP)
    support, headwind = _role_supports(obs, roles)
    return (
        (0.28 * max(support, headwind))
        + (0.20 * _clip01(abs(_rotation_bias(obs)) / 0.9))
        + (0.18 * _quote_quality(obs))
        + (0.18 * observation_feature(obs, "swing_weekly_trend_confirm_norm"))
        + (0.16 * observation_feature(obs, "market_micro_relative_volume_norm"))
    )


def _runtime_sector_rotation_label(sequence, idx, horizon):
    obs = sequence[idx]
    roles = symbol_role_features(str(obs.get("symbol") or ""), _SECTOR_ROLE_MAP)
    if sum(float(value) for value in roles.values()) <= 0.0:
        return None

    support, headwind = _role_supports(obs, roles)
    bias = _rotation_bias(obs)
    if max(support, headwind) < 0.18 or abs(bias) < 0.08:
        return None

    expected_up = support >= headwind
    fwd_ret = future_return(sequence, idx, horizon)
    realized = future_realized_vol(sequence, idx, horizon)
    drawdown = abs(future_max_drawdown(sequence, idx, horizon))
    signed_ret = fwd_ret if expected_up else -fwd_ret
    move_threshold = max(0.00065, 0.00140 - (0.00060 * max(support, headwind)))
    if abs(fwd_ret) < move_threshold and realized < 0.018 and drawdown < 0.012:
        return None

    support_score = (
        signed_ret
        + (0.0010 * support)
        - (0.00055 * headwind)
        + (0.00020 * _quote_quality(obs))
        - (0.26 * drawdown)
        - (0.18 * realized)
    )
    failure_score = (
        (-signed_ret)
        + (0.00095 * headwind)
        - (0.00040 * support)
        + (0.20 * realized)
        + (0.24 * drawdown)
    )
    if support_score >= 0.00055 and support >= headwind:
        return 1.0
    if failure_score >= 0.00080 and headwind > support:
        return 0.0
    return None


def _train_synthetic():
    return train_indicator_bot(
        run_tag="brain_refinery_v52_sector_rotation_rs",
        feature_names=["ret", "bench_ret", "alpha", "rs_fast", "rs_slow", "rs_spread", "rot_pressure", "rs_noise"],
        feature_builder=build_features,
        window=48,
        horizon=5,
    )


def train_brain():
    return train_runtime_indicator_bot(
        run_tag="brain_refinery_v52_sector_rotation_rs",
        feature_names=[
            "pct_from_close",
            "mom_15m",
            "vol_30m",
            "range_pos",
            "spread_bps",
            "queue_depth",
            "day_regime_trend_norm",
            "day_regime_alignment_norm",
            "swing_sector_relative_strength_norm",
            "swing_weekly_trend_confirm_norm",
            "swing_regime_trend_norm",
            "swing_regime_alignment_norm",
            "breadth_sector_dispersion_norm",
            "breadth_advance_decline_norm",
            "breadth_risk_off_norm",
            "market_micro_relative_volume_norm",
            "market_micro_trend_persistence_norm",
            "options_vol_expectation_norm",
            "options_negative_bias_norm",
            "capital_flow_signed_scaled",
            "flow_direction_signed",
            "flow_risk_on_norm",
            "data_quality_quote_agreement_norm",
            "data_quality_quote_deviation_norm",
            "behavior_prior",
            "ret_3",
            "ret_6",
            "pct_from_close_std_8",
            "swing_sector_relative_strength_norm_ema_5",
            "flow_risk_on_norm_ema_4",
            "role_growth",
            "role_cyclical",
            "role_defensive",
        ],
        runtime_feature_builder=_runtime_feature_vector,
        runtime_label_builder=_runtime_sector_rotation_label,
        mode_allowlist=_SECTOR_ROTATION_MODES,
        symbol_allowlist=_EXPANDED_SECTOR_SYMBOLS,
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
