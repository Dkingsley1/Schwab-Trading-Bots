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
    obs = sequence[idx]
    return np.asarray(
        [
            observation_feature(obs, "pct_from_close"),
            observation_feature(obs, "mom_5m"),
            observation_feature(obs, "vol_30m"),
            observation_feature(obs, "range_pos"),
            observation_feature(obs, "ctx_VIX_X_pct_from_close"),
            observation_feature(obs, "ctx_UUP_pct_from_close"),
            observation_feature(obs, "bond_duration_regime_norm"),
            observation_feature(obs, "bond_credit_risk_off_norm"),
            observation_feature(obs, "bond_credit_spread_level_norm"),
            observation_feature(obs, "bond_curve_flattener_norm"),
            observation_feature(obs, "breadth_risk_off_norm"),
            observation_feature(obs, "calendar_macro_abs_surprise_norm"),
            observation_feature(obs, "calendar_macro_revision_norm"),
            observation_feature(obs, "dividend_quality_score_norm"),
            observation_feature(obs, "dividend_safety_composite_norm"),
            observation_feature(obs, "options_negative_bias_norm"),
            observation_feature(obs, "capital_flow_outflow_norm"),
            observation_feature(obs, "market_micro_credit_flow_norm"),
            observation_feature(obs, "data_quality_quote_agreement_norm"),
            observation_feature(obs, "data_quality_quote_deviation_norm"),
            price_change(sequence, idx, 3),
            price_change(sequence, idx, 6),
            feature_std(sequence, idx, "pct_from_close", 8),
            feature_ema(sequence, idx, "breadth_risk_off_norm", 4),
        ],
        dtype=np.float32,
    )


def _clip01(value):
    return float(np.clip(value, 0.0, 1.0))


def _risk_off_signal(obs):
    vix = _clip01(abs(observation_feature(obs, "ctx_VIX_X_pct_from_close")) / 0.03)
    usd = _clip01(max(observation_feature(obs, "ctx_UUP_pct_from_close"), 0.0) / 0.015)
    duration = _clip01(observation_feature(obs, "bond_duration_regime_norm"))
    credit = _clip01(observation_feature(obs, "bond_credit_risk_off_norm"))
    breadth = _clip01(observation_feature(obs, "breadth_risk_off_norm"))
    options = _clip01(observation_feature(obs, "options_negative_bias_norm"))
    flow = _clip01(observation_feature(obs, "capital_flow_outflow_norm"))
    macro = _clip01(
        max(
            observation_feature(obs, "calendar_macro_abs_surprise_norm"),
            observation_feature(obs, "calendar_macro_revision_norm"),
        )
    )
    credit_micro = _clip01(observation_feature(obs, "market_micro_credit_flow_norm"))
    return _clip01(
        (0.22 * vix)
        + (0.08 * usd)
        + (0.14 * duration)
        + (0.18 * credit)
        + (0.16 * breadth)
        + (0.10 * options)
        + (0.07 * flow)
        + (0.03 * macro)
        + (0.02 * credit_micro)
    )


def _runtime_sample_filter(sequence, idx, horizon):
    obs = sequence[idx]
    return (
        observation_feature(obs, "data_quality_quote_agreement_norm", 1.0) >= 0.76
        and observation_feature(obs, "data_quality_quote_deviation_norm", 0.0) <= 0.30
        and _risk_off_signal(obs) >= 0.24
        and max(
            observation_feature(obs, "bond_duration_regime_norm"),
            observation_feature(obs, "bond_credit_risk_off_norm"),
            observation_feature(obs, "breadth_risk_off_norm"),
            observation_feature(obs, "options_negative_bias_norm"),
        )
        >= 0.18
    )


def _runtime_confidence(sequence, idx, horizon):
    obs = sequence[idx]
    quote_quality = _clip01(
        0.60 * observation_feature(obs, "data_quality_quote_agreement_norm", 1.0)
        + 0.40 * (1.0 - observation_feature(obs, "data_quality_quote_deviation_norm", 0.0))
    )
    defensive_quality = _clip01(
        0.55 * observation_feature(obs, "dividend_quality_score_norm")
        + 0.45 * observation_feature(obs, "dividend_safety_composite_norm")
    )
    rate_signal = _clip01(
        0.55 * observation_feature(obs, "bond_duration_regime_norm")
        + 0.45 * observation_feature(obs, "bond_credit_risk_off_norm")
    )
    return (
        0.30 * _risk_off_signal(obs)
        + 0.20 * rate_signal
        + 0.16 * defensive_quality
        + 0.14 * _clip01(observation_feature(obs, "options_negative_bias_norm"))
        + 0.10 * _clip01(observation_feature(obs, "market_micro_credit_flow_norm"))
        + 0.10 * quote_quality
    )


def _runtime_defensive_label(sequence, idx, horizon):
    obs = sequence[idx]
    risk_off = _risk_off_signal(obs)
    if risk_off < 0.24:
        return None

    fwd_ret = future_return(sequence, idx, horizon)
    dd = abs(future_max_drawdown(sequence, idx, horizon))
    realized = future_realized_vol(sequence, idx, horizon)
    defensive_bias = _clip01(
        0.30 * observation_feature(obs, "bond_duration_regime_norm")
        + 0.25 * observation_feature(obs, "bond_credit_risk_off_norm")
        + 0.20 * observation_feature(obs, "dividend_quality_score_norm")
        + 0.15 * observation_feature(obs, "dividend_safety_composite_norm")
        + 0.10 * observation_feature(obs, "options_negative_bias_norm")
    )
    move_threshold = max(0.0007, 0.0016 - (0.0007 * risk_off))
    if abs(fwd_ret) < move_threshold and dd < 0.010:
        return None

    support_score = (
        fwd_ret
        + (0.0012 * defensive_bias)
        - (0.75 * dd)
        - (0.28 * realized)
        - (0.0006 * observation_feature(obs, "capital_flow_outflow_norm"))
    )
    failure_score = (
        (-fwd_ret)
        + (0.65 * dd)
        + (0.18 * realized)
        + (0.0005 * observation_feature(obs, "capital_flow_outflow_norm"))
        - (0.0005 * defensive_bias)
    )
    if support_score >= 0.0006:
        return 1.0
    if failure_score >= 0.0010:
        return 0.0
    return None


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
            "calendar_macro_revision_norm",
            "dividend_quality_score_norm",
            "dividend_safety_composite_norm",
            "options_negative_bias_norm",
            "capital_flow_outflow_norm",
            "market_micro_credit_flow_norm",
            "data_quality_quote_agreement_norm",
            "data_quality_quote_deviation_norm",
            "ret_3",
            "ret_6",
            "pct_from_close_std_8",
            "breadth_risk_off_ema_4",
        ],
        runtime_feature_builder=_runtime_feature_vector,
        runtime_label_builder=_runtime_defensive_label,
        sample_filter=_runtime_sample_filter,
        confidence_builder=_runtime_confidence,
        min_confidence=0.34,
        lookback_days=30,
        window=24,
        horizon=8,
        min_samples=256,
        min_sequences=6,
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
