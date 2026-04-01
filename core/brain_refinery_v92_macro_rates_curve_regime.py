import numpy as np

from indicator_bot_common import ema, rolling_mean, rolling_std, train_indicator_bot, train_runtime_indicator_bot
from runtime_training_common import (
    feature_ema,
    feature_std,
    multi_horizon_direction_label_builder,
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


def build_features(panel):
    r = panel["ret"]
    bench = panel["bench_ret"]
    vix = panel["vix"]
    vix3m = panel["vix3m"]

    short_rate_proxy = 2.0 + 180.0 * ema(rolling_std(r, 20), 12)
    long_rate_proxy = 2.2 + 140.0 * ema(rolling_std(bench, 45), 20) + 90.0 * np.maximum(ema(bench, 70), -0.01)

    curve_slope_proxy = long_rate_proxy - short_rate_proxy
    real_rate_proxy = short_rate_proxy - (2.0 + 40.0 * ema(np.abs(r), 35))

    term_vol = (vix3m - vix) / (vix + 1e-8)
    curve_inversion_risk = np.tanh(-curve_slope_proxy / 1.8)
    liquidity_fracture = np.tanh(1.3 * term_vol + 0.9 * (rolling_std(r, 30) * 130.0))

    return np.stack(
        [
            r,
            short_rate_proxy / 10.0,
            long_rate_proxy / 10.0,
            curve_slope_proxy / 10.0,
            real_rate_proxy / 10.0,
            term_vol,
            curve_inversion_risk,
            liquidity_fracture,
        ],
        axis=1,
    )


def _runtime_feature_vector(sequence, idx):
    obs = sequence[idx]
    roles = symbol_role_features(str(obs.get("symbol") or ""), _BOND_ROLE_MAP)
    return np.asarray(
        [
            observation_feature(obs, "pct_from_close"),
            observation_feature(obs, "mom_5m"),
            observation_feature(obs, "vol_30m"),
            observation_feature(obs, "range_pos"),
            observation_feature(obs, "spread_bps"),
            observation_feature(obs, "ctx_UUP_pct_from_close"),
            observation_feature(obs, "options_iv_atm_norm"),
            observation_feature(obs, "options_iv_skew_norm"),
            observation_feature(obs, "options_vol_expectation_norm"),
            observation_feature(obs, "bond_duration_regime_norm"),
            observation_feature(obs, "bond_curve_steepener_norm"),
            observation_feature(obs, "bond_curve_flattener_norm"),
            observation_feature(obs, "bond_carry_roll_norm"),
            observation_feature(obs, "bond_inflation_breakeven_norm"),
            observation_feature(obs, "bond_yield_2y_norm"),
            observation_feature(obs, "bond_yield_5y_norm"),
            observation_feature(obs, "bond_curve_2s10s_norm"),
            observation_feature(obs, "bond_curve_5s30s_norm"),
            observation_feature(obs, "bond_real_yield_10y_norm"),
            observation_feature(obs, "bond_yield_10y_norm"),
            observation_feature(obs, "bond_yield_30y_norm"),
            observation_feature(obs, "bond_auction_window_norm"),
            observation_feature(obs, "bond_auction_tail_norm"),
            observation_feature(obs, "bond_credit_spread_level_norm"),
            observation_feature(obs, "bond_credit_spread_change_norm"),
            observation_feature(obs, "bond_nav_stress_norm"),
            observation_feature(obs, "calendar_macro_surprise_norm"),
            observation_feature(obs, "calendar_macro_abs_surprise_norm"),
            observation_feature(obs, "calendar_treasury_auction_norm"),
            observation_feature(obs, "breadth_risk_off_norm"),
            observation_feature(obs, "ctx_VIX_X_pct_from_close"),
            observation_feature(obs, "data_quality_quote_agreement_norm", 1.0),
            observation_feature(obs, "data_quality_quote_deviation_norm"),
            observation_feature(obs, "data_quality_market_data_latency_norm"),
            observation_feature(obs, "market_micro_opening_auction_norm"),
            observation_feature(obs, "market_micro_relative_volume_norm"),
            observation_feature(obs, "market_micro_options_flow_norm"),
            observation_feature(obs, "market_micro_short_pressure_norm"),
            observation_feature(obs, "market_micro_credit_flow_norm"),
            price_change(sequence, idx, 3),
            price_change(sequence, idx, 6),
            feature_ema(sequence, idx, "bond_curve_steepener_norm", 4),
            feature_ema(sequence, idx, "bond_curve_flattener_norm", 4),
            feature_ema(sequence, idx, "bond_credit_spread_change_norm", 4),
            feature_std(sequence, idx, "pct_from_close", 6),
            roles["role_long_duration"],
            roles["role_short_duration"],
            roles["role_inflation"],
            roles["role_credit"],
        ],
        dtype=np.float32,
    )


def _clip01(value):
    return float(np.clip(value, 0.0, 1.0))


def _runtime_sample_filter(sequence, idx, horizon):
    obs = sequence[idx]
    spread = abs(observation_feature(obs, "spread_bps"))
    quote_agreement = observation_feature(obs, "data_quality_quote_agreement_norm", 1.0)
    curve_signal = max(
        abs(observation_feature(obs, "bond_curve_steepener_norm")),
        abs(observation_feature(obs, "bond_curve_flattener_norm")),
        abs(observation_feature(obs, "bond_curve_2s10s_norm")),
        abs(observation_feature(obs, "bond_curve_5s30s_norm")),
    )
    yield_signal = max(
        abs(observation_feature(obs, "bond_yield_2y_norm") - 0.5),
        abs(observation_feature(obs, "bond_yield_5y_norm") - 0.5),
        abs(observation_feature(obs, "bond_yield_10y_norm")),
        abs(observation_feature(obs, "bond_yield_30y_norm")),
        abs(observation_feature(obs, "bond_real_yield_10y_norm")),
    )
    auction_signal = max(
        abs(observation_feature(obs, "calendar_treasury_auction_norm")),
        abs(observation_feature(obs, "bond_auction_window_norm")),
        abs(observation_feature(obs, "bond_auction_tail_norm") - 0.5) * 2.0,
    )
    credit_signal = max(
        abs(observation_feature(obs, "bond_credit_spread_level_norm")),
        abs(observation_feature(obs, "bond_credit_spread_change_norm") - 0.5) * 2.0,
        abs(observation_feature(obs, "bond_nav_stress_norm")),
    )
    return (
        spread <= 30.0
        and quote_agreement >= 0.85
        and (curve_signal > 0.015 or yield_signal > 0.015 or auction_signal > 0.08 or credit_signal > 0.08)
    )


def _runtime_confidence(sequence, idx, horizon):
    obs = sequence[idx]
    curve_signal = _clip01(
        max(
            abs(observation_feature(obs, "bond_curve_steepener_norm")),
            abs(observation_feature(obs, "bond_curve_flattener_norm")),
            abs(observation_feature(obs, "bond_curve_2s10s_norm")),
            abs(observation_feature(obs, "bond_curve_5s30s_norm")),
        )
    )
    macro_signal = _clip01(
        max(
            abs(observation_feature(obs, "calendar_macro_surprise_norm")),
            abs(observation_feature(obs, "calendar_macro_abs_surprise_norm")),
            abs(observation_feature(obs, "calendar_treasury_auction_norm")),
            abs(observation_feature(obs, "bond_auction_window_norm")),
            abs(observation_feature(obs, "bond_auction_tail_norm") - 0.5) * 2.0,
        )
    )
    credit_signal = _clip01(
        max(
            abs(observation_feature(obs, "bond_credit_spread_level_norm")),
            abs(observation_feature(obs, "bond_credit_spread_change_norm") - 0.5) * 2.0,
            abs(observation_feature(obs, "bond_nav_stress_norm")),
        )
    )
    duration_signal = _clip01(abs(observation_feature(obs, "bond_duration_regime_norm")))
    quote_ok = _clip01(observation_feature(obs, "data_quality_quote_agreement_norm", 1.0))
    spread_ok = _clip01(1.0 - (abs(observation_feature(obs, "spread_bps")) / 30.0))
    return (0.24 * curve_signal) + (0.20 * macro_signal) + (0.18 * duration_signal) + (0.16 * credit_signal) + (0.12 * quote_ok) + (0.10 * spread_ok)


def _train_synthetic():
    return train_indicator_bot(
        run_tag="brain_refinery_v92_macro_rates_curve_regime",
        feature_names=[
            "ret",
            "short_rate_proxy",
            "long_rate_proxy",
            "curve_slope_proxy",
            "real_rate_proxy",
            "term_vol",
            "curve_inversion_risk",
            "liquidity_fracture",
        ],
        feature_builder=build_features,
    )


if __name__ == "__main__":
    train_runtime_indicator_bot(
        run_tag="brain_refinery_v92_macro_rates_curve_regime",
        feature_names=[
            "pct_from_close",
            "mom_5m",
            "vol_30m",
            "range_pos",
            "spread_bps",
            "ctx_UUP_pct_from_close",
            "options_iv_atm_norm",
            "options_iv_skew_norm",
            "options_vol_expectation_norm",
            "bond_duration_regime_norm",
            "bond_curve_steepener_norm",
            "bond_curve_flattener_norm",
            "bond_carry_roll_norm",
            "bond_inflation_breakeven_norm",
            "bond_yield_2y_norm",
            "bond_yield_5y_norm",
            "bond_curve_2s10s_norm",
            "bond_curve_5s30s_norm",
            "bond_real_yield_10y_norm",
            "bond_yield_10y_norm",
            "bond_yield_30y_norm",
            "bond_auction_window_norm",
            "bond_auction_tail_norm",
            "bond_credit_spread_level_norm",
            "bond_credit_spread_change_norm",
            "bond_nav_stress_norm",
            "calendar_macro_surprise_norm",
            "calendar_macro_abs_surprise_norm",
            "calendar_treasury_auction_norm",
            "breadth_risk_off_norm",
            "ctx_VIX_X_pct_from_close",
            "data_quality_quote_agreement_norm",
            "data_quality_quote_deviation_norm",
            "data_quality_market_data_latency_norm",
            "market_micro_opening_auction_norm",
            "market_micro_relative_volume_norm",
            "market_micro_options_flow_norm",
            "market_micro_short_pressure_norm",
            "market_micro_credit_flow_norm",
            "ret_3",
            "ret_6",
            "bond_curve_steepener_ema_4",
            "bond_curve_flattener_ema_4",
            "bond_credit_spread_change_ema_4",
            "pct_from_close_std_6",
            "role_long_duration",
            "role_short_duration",
            "role_inflation",
            "role_credit",
        ],
        runtime_feature_builder=_runtime_feature_vector,
        runtime_label_builder=multi_horizon_direction_label_builder(horizons=[6, 12], min_return=0.0005),
        lookback_days=45,
        mode_allowlist=["shadow_bond_equities"],
        symbol_allowlist=_EXPANDED_BOND_SYMBOLS,
        sample_filter=_runtime_sample_filter,
        confidence_builder=_runtime_confidence,
        min_confidence=0.34,
        window=24,
        horizon=12,
        min_samples=1200,
        min_sequences=12,
        acted_prob_threshold=0.68,
        fallback_trainer=_train_synthetic,
        allow_fallback_on_insufficient_data=False,
        max_best_val_loss=0.6928,
        max_final_val_loss=0.7600,
        min_long_precision=0.10,
        min_short_precision=0.10,
    )
