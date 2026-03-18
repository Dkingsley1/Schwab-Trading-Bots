import numpy as np

from indicator_bot_common import ema, rolling_std, train_indicator_bot, train_runtime_indicator_bot
from runtime_training_common import (
    direction_label_builder,
    feature_ema,
    feature_std,
    observation_feature,
    price_change,
)


def build_features(panel):
    r = panel["ret"]
    rb = panel["bench_ret"]
    c = panel["close"]
    b = panel["bench_close"]
    vix = panel["vix"]

    rs = c / (b + 1e-8)
    rs_fast = ema(rs, 8)
    rs_slow = ema(rs, 34)
    rs_spread = (rs_fast - rs_slow) / (rs_slow + 1e-8)

    alpha = r - rb
    alpha_smooth = ema(alpha, 10)
    alpha_vol = rolling_std(alpha, 20)

    vix_chg = np.diff(vix, prepend=vix[0]) / (vix + 1e-8)

    return np.stack([r, rb, alpha, alpha_smooth, alpha_vol, rs_spread, vix_chg], axis=1)


def _runtime_feature_vector(sequence, idx):
    return np.asarray(
        [
            observation_feature(sequence[idx], "pct_from_close"),
            observation_feature(sequence[idx], "mom_5m"),
            observation_feature(sequence[idx], "vol_30m"),
            observation_feature(sequence[idx], "range_pos"),
            observation_feature(sequence[idx], "ctx_VIX_X_pct_from_close"),
            observation_feature(sequence[idx], "ctx_UUP_pct_from_close"),
            observation_feature(sequence[idx], "bond_duration_regime_norm"),
            observation_feature(sequence[idx], "bond_credit_risk_off_norm"),
            observation_feature(sequence[idx], "bond_credit_spread_level_norm"),
            observation_feature(sequence[idx], "bond_curve_flattener_norm"),
            observation_feature(sequence[idx], "breadth_risk_off_norm"),
            observation_feature(sequence[idx], "calendar_macro_abs_surprise_norm"),
            observation_feature(sequence[idx], "dividend_quality_score_norm"),
            observation_feature(sequence[idx], "options_negative_bias_norm"),
            observation_feature(sequence[idx], "capital_flow_outflow_norm"),
            observation_feature(sequence[idx], "data_quality_quote_agreement_norm"),
            price_change(sequence, idx, 3),
            price_change(sequence, idx, 6),
            feature_std(sequence, idx, "pct_from_close", 8),
            feature_ema(sequence, idx, "breadth_risk_off_norm", 4),
        ],
        dtype=np.float32,
    )


def _train_synthetic():
    return train_indicator_bot(
        run_tag="brain_refinery_v31_defensive_rotation",
        feature_names=["ret", "bench_ret", "alpha", "alpha_ema10", "alpha_vol20", "rs_spread", "vix_chg"],
        feature_builder=build_features,
    )


if __name__ == "__main__":
    train_runtime_indicator_bot(
        run_tag="brain_refinery_v31_defensive_rotation",
        feature_names=[
            "pct_from_close",
            "mom_5m",
            "vol_30m",
            "range_pos",
            "ctx_VIX_X_pct_from_close",
            "ctx_UUP_pct_from_close",
            "bond_duration_regime_norm",
            "bond_credit_risk_off_norm",
            "bond_credit_spread_level_norm",
            "bond_curve_flattener_norm",
            "breadth_risk_off_norm",
            "calendar_macro_abs_surprise_norm",
            "dividend_quality_score_norm",
            "options_negative_bias_norm",
            "capital_flow_outflow_norm",
            "data_quality_quote_agreement_norm",
            "ret_3",
            "ret_6",
            "pct_from_close_std_8",
            "breadth_risk_off_ema_4",
        ],
        runtime_feature_builder=_runtime_feature_vector,
        runtime_label_builder=direction_label_builder(min_return=0.0004),
        lookback_days=21,
        window=24,
        horizon=6,
        min_samples=224,
        min_sequences=3,
        fallback_trainer=_train_synthetic,
    )
