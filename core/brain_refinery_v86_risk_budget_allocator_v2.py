import numpy as np

from indicator_bot_common import ema, rolling_mean, rolling_std, train_indicator_bot, train_runtime_indicator_bot
from runtime_training_common import (
    feature_ema,
    feature_std,
    observation_feature,
    price_change,
    risk_support_label_builder,
    rolling_drawdown as runtime_rolling_drawdown,
)


def rolling_drawdown(close, window=180):
    out = np.zeros_like(close)
    for i in range(len(close)):
        s = max(0, i - window + 1)
        peak = np.max(close[s : i + 1])
        out[i] = (close[i] - peak) / (peak + 1e-8)
    return out


def build_features(panel):
    c = panel["close"]
    r = panel["ret"]
    rb = panel["bench_ret"]
    vix = panel["vix"]
    vix9d = panel["vix9d"]
    vix3m = panel["vix3m"]

    vol_fast = rolling_std(r, 10)
    vol_slow = rolling_std(r, 40)
    vol_ratio = vol_fast / (vol_slow + 1e-8)

    dd = rolling_drawdown(c, 220)
    dd_pressure = np.abs(ema(dd, 10))

    corr_proxy = ema(r * rb, 20) / (np.sqrt(ema(r * r, 20) * ema(rb * rb, 20)) + 1e-8)
    corr_stress = np.maximum(np.abs(corr_proxy) - 0.65, 0.0)

    term_slope = (vix9d - vix3m) / (vix + 1e-8)
    vol_of_vol = rolling_std(vix, 20) / (rolling_mean(vix, 20) + 1e-8)

    budget_risk = ema(vol_ratio + dd_pressure + corr_stress + np.maximum(term_slope, 0.0) + vol_of_vol, 8)
    risk_budget_score = 1.0 - np.clip(0.35 * vol_ratio + 0.30 * dd_pressure + 0.20 * corr_stress + 0.15 * np.maximum(term_slope, 0.0), 0.0, 1.0)

    return np.stack(
        [r, rb, vol_fast, vol_slow, vol_ratio, dd, dd_pressure, corr_proxy, corr_stress, term_slope, vol_of_vol, budget_risk, risk_budget_score],
        axis=1,
    )


def _runtime_feature_vector(sequence, idx):
    return np.asarray(
        [
            observation_feature(sequence[idx], "pct_from_close"),
            observation_feature(sequence[idx], "mom_5m"),
            observation_feature(sequence[idx], "vol_30m"),
            observation_feature(sequence[idx], "range_pos"),
            observation_feature(sequence[idx], "spread_bps"),
            observation_feature(sequence[idx], "queue_depth"),
            observation_feature(sequence[idx], "active_sub_bots"),
            observation_feature(sequence[idx], "active_options_sub_bots"),
            observation_feature(sequence[idx], "options_specialist_vote"),
            observation_feature(sequence[idx], "behavior_prior"),
            observation_feature(sequence[idx], "ctx_VIX_X_pct_from_close"),
            observation_feature(sequence[idx], "ctx_UUP_pct_from_close"),
            observation_feature(sequence[idx], "options_iv_atm_norm"),
            observation_feature(sequence[idx], "options_iv_skew_norm"),
            observation_feature(sequence[idx], "options_put_call_oi_ratio_norm"),
            observation_feature(sequence[idx], "options_negative_bias_norm"),
            observation_feature(sequence[idx], "options_roll_yield_norm"),
            observation_feature(sequence[idx], "bond_duration_regime_norm"),
            observation_feature(sequence[idx], "bond_credit_risk_on_norm"),
            observation_feature(sequence[idx], "bond_credit_spread_level_norm"),
            observation_feature(sequence[idx], "dividend_quality_score_norm"),
            observation_feature(sequence[idx], "capital_flow_signed_scaled"),
            observation_feature(sequence[idx], "breadth_risk_off_norm"),
            observation_feature(sequence[idx], "calendar_macro_abs_surprise_norm"),
            observation_feature(sequence[idx], "data_quality_quote_agreement_norm"),
            observation_feature(sequence[idx], "lag_adjusted_return_1m"),
            observation_feature(sequence[idx], "lag_expected_fill_delta_bps"),
            observation_feature(sequence[idx], "lag_fee_bps"),
            price_change(sequence, idx, 6),
            feature_std(sequence, idx, "pct_from_close", 8),
            feature_std(sequence, idx, "vol_30m", 8),
            runtime_rolling_drawdown(sequence, idx, 14),
            feature_ema(sequence, idx, "behavior_prior", 4),
        ],
        dtype=np.float32,
    )


def _train_synthetic():
    return train_indicator_bot(
        run_tag="brain_refinery_v86_risk_budget_allocator_v2",
        feature_names=[
            "ret",
            "bench_ret",
            "vol_fast",
            "vol_slow",
            "vol_ratio",
            "drawdown",
            "dd_pressure",
            "corr_proxy",
            "corr_stress",
            "term_slope",
            "vol_of_vol",
            "budget_risk",
            "risk_budget_score",
        ],
        feature_builder=build_features,
        window=52,
        horizon=4,
    )


if __name__ == "__main__":
    train_runtime_indicator_bot(
        run_tag="brain_refinery_v86_risk_budget_allocator_v2",
        feature_names=[
            "pct_from_close",
            "mom_5m",
            "vol_30m",
            "range_pos",
            "spread_bps",
            "queue_depth",
            "active_sub_bots",
            "active_options_sub_bots",
            "options_specialist_vote",
            "behavior_prior",
            "ctx_VIX_X_pct_from_close",
            "ctx_UUP_pct_from_close",
            "options_iv_atm_norm",
            "options_iv_skew_norm",
            "options_put_call_oi_ratio_norm",
            "options_negative_bias_norm",
            "options_roll_yield_norm",
            "bond_duration_regime_norm",
            "bond_credit_risk_on_norm",
            "bond_credit_spread_level_norm",
            "dividend_quality_score_norm",
            "capital_flow_signed_scaled",
            "breadth_risk_off_norm",
            "calendar_macro_abs_surprise_norm",
            "data_quality_quote_agreement_norm",
            "lag_adjusted_return_1m",
            "lag_expected_fill_delta_bps",
            "lag_fee_bps",
            "ret_6",
            "pct_from_close_std_8",
            "vol_30m_std_8",
            "drawdown_14",
            "behavior_prior_ema_4",
        ],
        runtime_feature_builder=_runtime_feature_vector,
        runtime_label_builder=risk_support_label_builder(
            min_return=0.0,
            max_drawdown=0.015,
            max_realized_vol=0.02,
            vol_multiplier=3.0,
        ),
        lookback_days=21,
        window=28,
        horizon=10,
        min_samples=224,
        min_sequences=3,
        fallback_trainer=_train_synthetic,
    )
