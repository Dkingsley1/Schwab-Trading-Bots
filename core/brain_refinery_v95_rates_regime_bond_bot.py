import numpy as np

from indicator_bot_common import atr, ema, rolling_std, train_indicator_bot, train_runtime_indicator_bot
from runtime_training_common import (
    direction_label_builder,
    feature_ema,
    feature_std,
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

    # Proxy for rates risk regime pressure.
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
            observation_feature(obs, "bond_duration_regime_norm"),
            observation_feature(obs, "bond_curve_flattener_norm"),
            observation_feature(obs, "bond_curve_steepener_norm"),
            observation_feature(obs, "bond_carry_roll_norm"),
            observation_feature(obs, "bond_credit_risk_on_norm"),
            observation_feature(obs, "bond_credit_risk_off_norm"),
            observation_feature(obs, "bond_inflation_breakeven_norm"),
            observation_feature(obs, "bond_curve_2s10s_norm"),
            observation_feature(obs, "bond_curve_5s30s_norm"),
            observation_feature(obs, "bond_real_yield_10y_norm"),
            observation_feature(obs, "bond_yield_10y_norm"),
            observation_feature(obs, "bond_duration_years_norm"),
            observation_feature(obs, "bond_nav_discount_norm"),
            observation_feature(obs, "calendar_treasury_auction_norm"),
            observation_feature(obs, "options_iv_skew_norm"),
            observation_feature(obs, "options_vol_expectation_norm"),
            observation_feature(obs, "options_gamma_exposure_norm"),
            observation_feature(obs, "options_unusual_flow_norm"),
            observation_feature(obs, "breadth_risk_off_norm"),
            price_change(sequence, idx, 6),
            feature_ema(sequence, idx, "bond_duration_regime_norm", 6),
            feature_ema(sequence, idx, "bond_carry_roll_norm", 6),
            feature_std(sequence, idx, "pct_from_close", 8),
            roles["role_long_duration"],
            roles["role_short_duration"],
            roles["role_inflation"],
            roles["role_credit"],
        ],
        dtype=np.float32,
    )


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
            "bond_duration_regime_norm",
            "bond_curve_flattener_norm",
            "bond_curve_steepener_norm",
            "bond_carry_roll_norm",
            "bond_credit_risk_on_norm",
            "bond_credit_risk_off_norm",
            "bond_inflation_breakeven_norm",
            "bond_curve_2s10s_norm",
            "bond_curve_5s30s_norm",
            "bond_real_yield_10y_norm",
            "bond_yield_10y_norm",
            "bond_duration_years_norm",
            "bond_nav_discount_norm",
            "calendar_treasury_auction_norm",
            "options_iv_skew_norm",
            "options_vol_expectation_norm",
            "options_gamma_exposure_norm",
            "options_unusual_flow_norm",
            "breadth_risk_off_norm",
            "ret_6",
            "bond_duration_regime_ema_6",
            "bond_carry_roll_ema_6",
            "pct_from_close_std_8",
            "role_long_duration",
            "role_short_duration",
            "role_inflation",
            "role_credit",
        ],
        runtime_feature_builder=_runtime_feature_vector,
        runtime_label_builder=direction_label_builder(min_return=0.0),
        lookback_days=30,
        mode_allowlist=["shadow_bond_equities"],
        window=28,
        horizon=18,
        min_samples=192,
        min_sequences=2,
        fallback_trainer=_train_synthetic,
    )
