import numpy as np

from indicator_bot_common import atr, ema, rolling_std, train_indicator_bot, train_runtime_indicator_bot
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

_BOND_ROLE_MAP = {
    "long_duration": ["TLT", "IEF", "TLH", "VGIT", "VGLT", "EDV", "ZROZ"],
    "short_duration": ["SHY", "FLOT", "VGSH", "SCHO"],
    "inflation": ["TIP", "VTIP", "SCHP"],
    "credit": ["LQD", "IGIB", "HYG", "JNK", "USHY"],
}
_EXPANDED_BOND_SYMBOLS = sorted({sym for values in _BOND_ROLE_MAP.values() for sym in values} | {"AGG", "BND", "MUB"})
_BOND_RUNTIME_MODES = [
    "shadow_bond_equities",
    "shadow_conservative_equities",
    "shadow_default_equities",
    "shadow_dividend_equities",
]


def hold_sample(x, step):
    out = np.zeros_like(x)
    for i in range(len(x)):
        out[i] = x[(i // step) * step]
    return out


def build_features(panel):
    c = panel["close"]
    h = panel["high"]
    l = panel["low"]
    r = panel["ret"]
    b = panel["bench_ret"]
    vix = panel.get("vix", np.zeros_like(c))

    q_ret = hold_sample(r, 1170)
    q_bench = hold_sample(b, 1170)
    q_alpha = q_ret - q_bench

    trend_fast = ema(q_ret, 6)
    trend_slow = ema(q_ret, 16)
    trend_spread = trend_fast - trend_slow

    atr_l = atr(h, l, c, period=28) / (c + 1e-8)
    vol = rolling_std(r, 120)

    vix_z = (vix - np.mean(vix)) / (np.std(vix) + 1e-8)
    rates_regime = trend_spread - 0.35 * vix_z - 0.25 * vol

    return np.stack([r, q_alpha, trend_fast, trend_slow, trend_spread, atr_l, vol, vix_z, rates_regime], axis=1)


def _runtime_feature_vector(sequence, idx):
    obs = sequence[idx]
    roles = symbol_role_features(str(obs.get("symbol") or ""), _BOND_ROLE_MAP)
    return np.asarray(
        [
            observation_feature(obs, "pct_from_close"),
            observation_feature(obs, "mom_5m"),
            observation_feature(obs, "vol_30m"),
            observation_feature(obs, "range_pos"),
            observation_feature(obs, "ctx_UUP_pct_from_close"),
            observation_feature(obs, "ctx_VIX_X_pct_from_close"),
            observation_feature(obs, "fx_usd_strength_norm"),
            observation_feature(obs, "fx_eurusd_momentum_norm"),
            observation_feature(obs, "fx_usdjpy_momentum_norm"),
            observation_feature(obs, "fx_proxy_agreement_norm"),
            observation_feature(obs, "fx_risk_on_alignment_norm"),
            observation_feature(obs, "fx_corr_confidence_norm"),
            observation_feature(obs, "bond_duration_regime_norm"),
            observation_feature(obs, "bond_curve_flattener_norm"),
            observation_feature(obs, "bond_curve_steepener_norm"),
            observation_feature(obs, "bond_carry_roll_norm"),
            observation_feature(obs, "bond_credit_risk_on_norm"),
            observation_feature(obs, "bond_credit_risk_off_norm"),
            observation_feature(obs, "bond_inflation_breakeven_norm"),
            observation_feature(obs, "bond_yield_2y_norm"),
            observation_feature(obs, "bond_yield_5y_norm"),
            observation_feature(obs, "bond_curve_2s10s_norm"),
            observation_feature(obs, "bond_curve_5s30s_norm"),
            observation_feature(obs, "bond_real_yield_10y_norm"),
            observation_feature(obs, "bond_yield_10y_norm"),
            observation_feature(obs, "bond_yield_30y_norm"),
            observation_feature(obs, "bond_duration_years_norm"),
            observation_feature(obs, "bond_nav_discount_norm"),
            observation_feature(obs, "bond_auction_window_norm"),
            observation_feature(obs, "bond_auction_tail_norm"),
            observation_feature(obs, "bond_credit_spread_level_norm"),
            observation_feature(obs, "bond_credit_spread_change_norm"),
            observation_feature(obs, "bond_hy_ig_flow_norm"),
            observation_feature(obs, "bond_nav_stress_norm"),
            observation_feature(obs, "calendar_macro_event_norm"),
            observation_feature(obs, "calendar_macro_abs_surprise_norm"),
            observation_feature(obs, "calendar_treasury_auction_norm"),
            observation_feature(obs, "options_iv_skew_norm"),
            observation_feature(obs, "options_vol_expectation_norm"),
            observation_feature(obs, "options_gamma_exposure_norm"),
            observation_feature(obs, "options_unusual_flow_norm"),
            observation_feature(obs, "breadth_risk_off_norm"),
            observation_feature(obs, "data_quality_quote_agreement_norm", 1.0),
            observation_feature(obs, "data_quality_quote_deviation_norm"),
            observation_feature(obs, "data_quality_market_data_latency_norm"),
            observation_feature(obs, "market_micro_opening_auction_norm"),
            observation_feature(obs, "market_micro_relative_volume_norm"),
            observation_feature(obs, "market_micro_options_flow_norm"),
            observation_feature(obs, "market_micro_short_pressure_norm"),
            observation_feature(obs, "market_micro_credit_flow_norm"),
            price_change(sequence, idx, 6),
            feature_ema(sequence, idx, "bond_duration_regime_norm", 6),
            feature_ema(sequence, idx, "bond_carry_roll_norm", 6),
            feature_ema(sequence, idx, "bond_credit_spread_change_norm", 6),
            feature_std(sequence, idx, "pct_from_close", 8),
            roles["role_long_duration"],
            roles["role_short_duration"],
            roles["role_inflation"],
            roles["role_credit"],
        ],
        dtype=np.float32,
    )


def _clip01(value):
    return float(np.clip(value, 0.0, 1.0))


def _long_duration_support(obs):
    return _clip01(
        (0.24 * _clip01(observation_feature(obs, "bond_duration_regime_norm")))
        + (0.18 * _clip01(observation_feature(obs, "bond_curve_flattener_norm")))
        + (0.10 * _clip01(observation_feature(obs, "bond_carry_roll_norm")))
        + (0.14 * _clip01(observation_feature(obs, "bond_credit_risk_off_norm")))
        + (0.12 * _clip01(observation_feature(obs, "breadth_risk_off_norm")))
        + (0.08 * _clip01(observation_feature(obs, "calendar_macro_abs_surprise_norm")))
        + (0.08 * _clip01(observation_feature(obs, "calendar_treasury_auction_norm")))
        + (0.06 * _clip01(observation_feature(obs, "market_micro_short_pressure_norm")))
    )


def _short_duration_support(obs):
    rising_rates = _clip01(max(-observation_feature(obs, "bond_duration_regime_norm"), 0.0))
    auction_stress = _clip01(abs(observation_feature(obs, "bond_auction_tail_norm") - 0.5) * 2.0)
    return _clip01(
        (0.24 * rising_rates)
        + (0.24 * _clip01(observation_feature(obs, "bond_curve_steepener_norm")))
        + (0.16 * _clip01(observation_feature(obs, "bond_carry_roll_norm")))
        + (0.14 * _clip01(observation_feature(obs, "bond_credit_risk_on_norm")))
        + (0.12 * _clip01(1.0 - observation_feature(obs, "breadth_risk_off_norm")))
        + (0.10 * _clip01(1.0 - auction_stress))
    )


def _inflation_support(obs):
    real_yield_pressure = _clip01(max(-observation_feature(obs, "bond_real_yield_10y_norm"), 0.0))
    return _clip01(
        (0.34 * _clip01(observation_feature(obs, "bond_inflation_breakeven_norm")))
        + (0.22 * _clip01(observation_feature(obs, "bond_curve_steepener_norm")))
        + (0.16 * real_yield_pressure)
        + (0.10 * _clip01(observation_feature(obs, "bond_carry_roll_norm")))
        + (0.10 * _clip01(observation_feature(obs, "calendar_macro_event_norm")))
        + (0.08 * _clip01(observation_feature(obs, "market_micro_relative_volume_norm")))
    )


def _credit_support(obs):
    spread_stress = _clip01(observation_feature(obs, "bond_credit_spread_level_norm"))
    nav_stress = _clip01(observation_feature(obs, "bond_nav_stress_norm"))
    return _clip01(
        (0.24 * _clip01(observation_feature(obs, "bond_credit_risk_on_norm")))
        + (0.16 * _clip01(observation_feature(obs, "bond_carry_roll_norm")))
        + (0.16 * _clip01(1.0 - spread_stress))
        + (0.14 * _clip01(1.0 - nav_stress))
        + (0.12 * _clip01(observation_feature(obs, "bond_hy_ig_flow_norm")))
        + (0.10 * _clip01(1.0 - observation_feature(obs, "breadth_risk_off_norm")))
        + (0.08 * _clip01(observation_feature(obs, "market_micro_credit_flow_norm")))
    )


def _role_supports(obs, roles):
    long_duration = _long_duration_support(obs)
    short_duration = _short_duration_support(obs)
    inflation = _inflation_support(obs)
    credit = _credit_support(obs)
    role_weight = max(sum(float(value) for value in roles.values()), 1.0)
    support = (
        (roles["role_long_duration"] * long_duration)
        + (roles["role_short_duration"] * short_duration)
        + (roles["role_inflation"] * inflation)
        + (roles["role_credit"] * credit)
    ) / role_weight
    headwind = (
        (roles["role_long_duration"] * max(short_duration, credit))
        + (roles["role_short_duration"] * long_duration)
        + (roles["role_inflation"] * max(long_duration, credit))
        + (roles["role_credit"] * max(long_duration, short_duration))
    ) / role_weight
    return _clip01(support), _clip01(headwind)


def _runtime_sample_filter(sequence, idx, horizon):
    obs = sequence[idx]
    roles = symbol_role_features(str(obs.get("symbol") or ""), _BOND_ROLE_MAP)
    role_support, role_headwind = _role_supports(obs, roles)
    quote_agreement = observation_feature(obs, "data_quality_quote_agreement_norm", 1.0)
    quote_deviation = observation_feature(obs, "data_quality_quote_deviation_norm", 0.0)
    latency = observation_feature(obs, "data_quality_market_data_latency_norm", 0.0)
    duration_signal = max(
        abs(observation_feature(obs, "bond_duration_regime_norm")),
        abs(observation_feature(obs, "bond_curve_flattener_norm")),
        abs(observation_feature(obs, "bond_curve_steepener_norm")),
        abs(observation_feature(obs, "bond_curve_2s10s_norm")),
        abs(observation_feature(obs, "bond_curve_5s30s_norm")),
    )
    macro_signal = max(
        abs(observation_feature(obs, "calendar_treasury_auction_norm")),
        abs(observation_feature(obs, "bond_auction_window_norm")),
        abs(observation_feature(obs, "bond_auction_tail_norm") - 0.5) * 2.0,
        abs(observation_feature(obs, "calendar_macro_event_norm")),
        abs(observation_feature(obs, "calendar_macro_abs_surprise_norm")),
    )
    return (
        quote_agreement >= 0.74
        and quote_deviation <= 0.35
        and latency <= 0.92
        and max(role_support, role_headwind) >= 0.18
        and max(duration_signal, macro_signal) >= 0.10
    )


def _runtime_confidence(sequence, idx, horizon):
    obs = sequence[idx]
    roles = symbol_role_features(str(obs.get("symbol") or ""), _BOND_ROLE_MAP)
    role_support, role_headwind = _role_supports(obs, roles)
    duration_signal = _clip01(
        max(
            abs(observation_feature(obs, "bond_duration_regime_norm")),
            abs(observation_feature(obs, "bond_curve_flattener_norm")),
            abs(observation_feature(obs, "bond_curve_steepener_norm")),
            abs(observation_feature(obs, "bond_curve_2s10s_norm")),
            abs(observation_feature(obs, "bond_curve_5s30s_norm")),
        )
    )
    credit_signal = _clip01(
        max(
            abs(observation_feature(obs, "bond_credit_risk_on_norm")),
            abs(observation_feature(obs, "bond_credit_risk_off_norm")),
            abs(observation_feature(obs, "bond_credit_spread_level_norm")),
            abs(observation_feature(obs, "bond_credit_spread_change_norm") - 0.5) * 2.0,
            abs(observation_feature(obs, "bond_nav_stress_norm")),
        )
    )
    macro_signal = _clip01(
        max(
            abs(observation_feature(obs, "calendar_treasury_auction_norm")),
            abs(observation_feature(obs, "bond_auction_window_norm")),
            abs(observation_feature(obs, "bond_auction_tail_norm") - 0.5) * 2.0,
            abs(observation_feature(obs, "calendar_macro_event_norm")),
            abs(observation_feature(obs, "calendar_macro_abs_surprise_norm")),
        )
    )
    flow_signal = _clip01(
        max(
            abs(observation_feature(obs, "options_gamma_exposure_norm")),
            abs(observation_feature(obs, "options_unusual_flow_norm")),
            abs(observation_feature(obs, "bond_hy_ig_flow_norm") - 0.5) * 2.0,
            abs(observation_feature(obs, "market_micro_credit_flow_norm")),
        )
    )
    quote_signal = _clip01(
        0.60 * observation_feature(obs, "data_quality_quote_agreement_norm", 1.0)
        + 0.25 * (1.0 - observation_feature(obs, "data_quality_quote_deviation_norm", 0.0))
        + 0.15 * (1.0 - observation_feature(obs, "data_quality_market_data_latency_norm", 0.0))
    )
    role_conviction = _clip01(max(role_support, role_headwind))
    return (
        (0.22 * role_conviction)
        + (0.18 * duration_signal)
        + (0.18 * credit_signal)
        + (0.16 * macro_signal)
        + (0.14 * flow_signal)
        + (0.12 * quote_signal)
    )


def _runtime_rates_label(sequence, idx, horizon):
    obs = sequence[idx]
    roles = symbol_role_features(str(obs.get("symbol") or ""), _BOND_ROLE_MAP)
    if sum(float(value) for value in roles.values()) <= 0.0:
        return None

    role_support, role_headwind = _role_supports(obs, roles)
    if max(role_support, role_headwind) < 0.18 or abs(role_support - role_headwind) < 0.04:
        return None

    fwd_ret = future_return(sequence, idx, horizon)
    dd = abs(future_max_drawdown(sequence, idx, horizon))
    realized = future_realized_vol(sequence, idx, horizon)
    carry_roll = _clip01(observation_feature(obs, "bond_carry_roll_norm"))
    move_threshold = max(0.0008, 0.0019 - (0.0010 * max(role_support, role_headwind)))
    if abs(fwd_ret) < move_threshold and dd < 0.012 and realized < 0.020:
        return None

    support_score = (
        fwd_ret
        + (0.0012 * role_support)
        + (0.0004 * carry_roll)
        - (0.0006 * role_headwind)
        - (0.72 * dd)
        - (0.22 * realized)
    )
    failure_score = (
        (-fwd_ret)
        + (0.0011 * role_headwind)
        - (0.0004 * role_support)
        + (0.64 * dd)
        + (0.18 * realized)
    )
    if support_score >= 0.0007 and role_support > role_headwind:
        return 1.0
    if failure_score >= 0.0010 and role_headwind >= role_support:
        return 0.0
    return None


def _train_synthetic():
    return train_indicator_bot(
        run_tag="brain_refinery_v95_rates_regime_bond_bot",
        feature_names=[
            "ret",
            "q_alpha",
            "trend_fast",
            "trend_slow",
            "trend_spread",
            "atr_long",
            "vol",
            "vix_z",
            "rates_regime",
        ],
        feature_builder=build_features,
        window=84,
        horizon=24,
    )


if __name__ == "__main__":
    train_runtime_indicator_bot(
        run_tag="brain_refinery_v95_rates_regime_bond_bot",
        feature_names=[
            "pct_from_close",
            "mom_5m",
            "vol_30m",
            "range_pos",
            "ctx_UUP_pct_from_close",
            "ctx_VIX_X_pct_from_close",
            "fx_usd_strength_norm",
            "fx_eurusd_momentum_norm",
            "fx_usdjpy_momentum_norm",
            "fx_proxy_agreement_norm",
            "fx_risk_on_alignment_norm",
            "fx_corr_confidence_norm",
            "bond_duration_regime_norm",
            "bond_curve_flattener_norm",
            "bond_curve_steepener_norm",
            "bond_carry_roll_norm",
            "bond_credit_risk_on_norm",
            "bond_credit_risk_off_norm",
            "bond_inflation_breakeven_norm",
            "bond_yield_2y_norm",
            "bond_yield_5y_norm",
            "bond_curve_2s10s_norm",
            "bond_curve_5s30s_norm",
            "bond_real_yield_10y_norm",
            "bond_yield_10y_norm",
            "bond_yield_30y_norm",
            "bond_duration_years_norm",
            "bond_nav_discount_norm",
            "bond_auction_window_norm",
            "bond_auction_tail_norm",
            "bond_credit_spread_level_norm",
            "bond_credit_spread_change_norm",
            "bond_hy_ig_flow_norm",
            "bond_nav_stress_norm",
            "calendar_macro_event_norm",
            "calendar_macro_abs_surprise_norm",
            "calendar_treasury_auction_norm",
            "options_iv_skew_norm",
            "options_vol_expectation_norm",
            "options_gamma_exposure_norm",
            "options_unusual_flow_norm",
            "breadth_risk_off_norm",
            "data_quality_quote_agreement_norm",
            "data_quality_quote_deviation_norm",
            "data_quality_market_data_latency_norm",
            "market_micro_opening_auction_norm",
            "market_micro_relative_volume_norm",
            "market_micro_options_flow_norm",
            "market_micro_short_pressure_norm",
            "market_micro_credit_flow_norm",
            "ret_6",
            "bond_duration_regime_ema_6",
            "bond_carry_roll_ema_6",
            "bond_credit_spread_change_ema_6",
            "pct_from_close_std_8",
            "role_long_duration",
            "role_short_duration",
            "role_inflation",
            "role_credit",
        ],
        runtime_feature_builder=_runtime_feature_vector,
        runtime_label_builder=_runtime_rates_label,
        lookback_days=60,
        mode_allowlist=_BOND_RUNTIME_MODES,
        symbol_allowlist=_EXPANDED_BOND_SYMBOLS,
        sample_filter=_runtime_sample_filter,
        confidence_builder=_runtime_confidence,
        min_confidence=0.32,
        window=24,
        horizon=12,
        min_samples=192,
        min_sequences=4,
        fallback_trainer=_train_synthetic,
        allow_fallback_on_insufficient_data=False,
        max_best_val_loss=0.6928,
        max_final_val_loss=0.7050,
        min_long_precision=0.52,
        min_short_precision=0.52,
        require_both_sides_precision=True,
        min_acted_accuracy=0.53,
        min_accuracy_lift_over_majority=0.02,
        min_label_balance_score=0.25,
        min_precision_balance_score=0.35,
    )
