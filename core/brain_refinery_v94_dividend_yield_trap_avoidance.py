import numpy as np

from indicator_bot_common import ema, rolling_std, train_indicator_bot, train_runtime_indicator_bot
from runtime_training_common import (
    feature_ema,
    future_max_drawdown,
    future_realized_vol,
    future_return,
    observation_feature,
    price_change,
)


def hold_sample(x, step):
    out = np.zeros_like(x)
    for i in range(len(x)):
        out[i] = x[(i // step) * step]
    return out


def rolling_drawdown(close, window=220):
    out = np.zeros_like(close)
    for i in range(len(close)):
        start = max(0, i - window + 1)
        peak = np.max(close[start : i + 1])
        out[i] = (close[i] - peak) / (peak + 1e-8)
    return out


def downside_vol(x, window=120):
    out = np.zeros_like(x)
    for i in range(len(x)):
        start = max(0, i - window + 1)
        seg = x[start : i + 1]
        neg = np.minimum(seg, 0.0)
        out[i] = np.sqrt(np.mean(neg * neg))
    return out


def build_features(panel):
    c = panel["close"]
    r = panel["ret"]
    b = panel["bench_ret"]

    dd = rolling_drawdown(c, window=260)
    dd_fast = ema(dd, 8)
    dd_slow = ema(dd, 21)

    vol = rolling_std(r, 120)
    dvol = downside_vol(r, window=120)

    q_ret = hold_sample(r, 1170)
    q_bench = hold_sample(b, 1170)
    q_alpha = q_ret - q_bench

    # Trap proxy: high downside stress + weak relative trend.
    trap_score = np.maximum(-dd_fast, 0.0) + np.maximum(-q_alpha, 0.0) + np.maximum(dvol - vol, 0.0)
    recovery = ema(r, 13)

    return np.stack([r, dd, dd_fast, dd_slow, q_alpha, vol, dvol, trap_score, recovery], axis=1)


def _clip01(value):
    return float(np.clip(value, 0.0, 1.0))


def _yield_trap_signal(obs):
    quality = observation_feature(obs, "dividend_quality_score_norm")
    safety = observation_feature(obs, "dividend_safety_composite_norm")
    return _clip01(
        0.22 * observation_feature(obs, "dividend_yield_norm")
        + 0.18 * observation_feature(obs, "dividend_payout_ratio_norm")
        + 0.12 * (1.0 - quality)
        + 0.12 * (1.0 - safety)
        + 0.12 * observation_feature(obs, "dividend_compound_drawdown_norm")
        + 0.10 * observation_feature(obs, "dividend_ex_slippage_risk_norm")
        + 0.08 * observation_feature(obs, "dividend_drip_cash_only_norm")
        + 0.06 * (1.0 - observation_feature(obs, "dividend_drip_active_norm"))
        - 0.04 * observation_feature(obs, "dividend_drip_confidence_norm")
        + 0.08 * observation_feature(obs, "capital_flow_outflow_norm")
        + 0.06 * observation_feature(obs, "options_negative_bias_norm")
    )


def _runtime_feature_vector(sequence, idx):
    obs = sequence[idx]
    return np.asarray(
        [
            observation_feature(obs, "pct_from_close"),
            observation_feature(obs, "mom_5m"),
            observation_feature(obs, "vol_30m"),
            observation_feature(obs, "range_pos"),
            observation_feature(obs, "ctx_VIX_X_pct_from_close"),
            observation_feature(obs, "bond_credit_risk_off_norm"),
            observation_feature(obs, "dividend_yield_norm"),
            observation_feature(obs, "dividend_payout_ratio_norm"),
            observation_feature(obs, "dividend_ex_date_proximity_norm"),
            observation_feature(obs, "dividend_pay_date_proximity_norm"),
            observation_feature(obs, "dividend_quality_score_norm"),
            observation_feature(obs, "dividend_capture_exit_signal_norm"),
            observation_feature(obs, "dividend_compound_drawdown_norm"),
            observation_feature(obs, "dividend_drip_cash_only_norm"),
            observation_feature(obs, "dividend_drip_active_norm"),
            observation_feature(obs, "dividend_drip_confidence_norm"),
            observation_feature(obs, "dividend_safety_composite_norm"),
            observation_feature(obs, "dividend_ex_slippage_risk_norm"),
            observation_feature(obs, "dividend_tax_qualified_hold_norm"),
            observation_feature(obs, "dividend_growth_momentum_norm"),
            observation_feature(obs, "dividend_rebalance_due_norm"),
            observation_feature(obs, "capital_flow_outflow_norm"),
            observation_feature(obs, "options_negative_bias_norm"),
            observation_feature(obs, "data_quality_quote_agreement_norm"),
            price_change(sequence, idx, 6),
            price_change(sequence, idx, 12),
            feature_ema(sequence, idx, "dividend_quality_score_norm", 4),
            feature_ema(sequence, idx, "dividend_payout_ratio_norm", 4),
        ],
        dtype=np.float32,
    )


def _runtime_sample_filter(sequence, idx, horizon):
    obs = sequence[idx]
    return (
        observation_feature(obs, "data_quality_quote_agreement_norm", 1.0) >= 0.74
        and max(_yield_trap_signal(obs), observation_feature(obs, "dividend_yield_norm")) >= 0.24
    )


def _runtime_confidence(sequence, idx, horizon):
    obs = sequence[idx]
    quote_quality = _clip01(observation_feature(obs, "data_quality_quote_agreement_norm", 1.0))
    trap_signal = _yield_trap_signal(obs)
    macro_stress = _clip01(
        0.55 * _clip01(abs(observation_feature(obs, "ctx_VIX_X_pct_from_close")) / 0.03)
        + 0.45 * observation_feature(obs, "bond_credit_risk_off_norm")
    )
    quality_gap = _clip01(
        0.50 * (1.0 - observation_feature(obs, "dividend_quality_score_norm"))
        + 0.50 * (1.0 - observation_feature(obs, "dividend_safety_composite_norm"))
    )
    return (0.34 * trap_signal) + (0.24 * macro_stress) + (0.20 * quality_gap) + (0.12 * quote_quality) + (0.10 * observation_feature(obs, "dividend_ex_slippage_risk_norm"))


def _runtime_yield_trap_label(sequence, idx, horizon):
    obs = sequence[idx]
    trap_signal = _yield_trap_signal(obs)
    if trap_signal < 0.24:
        return None

    fwd_ret = future_return(sequence, idx, horizon)
    dd = abs(future_max_drawdown(sequence, idx, horizon))
    realized = future_realized_vol(sequence, idx, horizon)
    safe_recovery = _clip01(
        0.55 * observation_feature(obs, "dividend_quality_score_norm")
        + 0.45 * observation_feature(obs, "dividend_safety_composite_norm")
        - 0.35 * trap_signal
    )
    downside_score = (
        (-fwd_ret)
        + (0.78 * dd)
        + (0.22 * realized)
        + (0.0008 * trap_signal)
    )
    upside_score = (
        fwd_ret
        + (0.0010 * safe_recovery)
        - (0.65 * dd)
        - (0.20 * realized)
        - (0.0005 * trap_signal)
    )
    move_threshold = max(0.0007, 0.0015 - (0.0005 * trap_signal))
    if abs(fwd_ret) < move_threshold and dd < 0.012:
        return None
    if downside_score >= 0.0011 and trap_signal >= 0.28:
        return 0.0
    if upside_score >= 0.0007 and safe_recovery >= 0.15:
        return 1.0
    return None


def _train_synthetic():
    return train_indicator_bot(
        run_tag="brain_refinery_v94_dividend_yield_trap_avoidance",
        feature_names=[
            "ret",
            "drawdown",
            "dd_fast",
            "dd_slow",
            "q_alpha",
            "vol",
            "downside_vol",
            "trap_score",
            "recovery",
        ],
        feature_builder=build_features,
        window=84,
        horizon=24,
    )


if __name__ == "__main__":
    train_runtime_indicator_bot(
        run_tag="brain_refinery_v94_dividend_yield_trap_avoidance",
        feature_names=[
            "pct_from_close",
            "mom_5m",
            "vol_30m",
            "range_pos",
            "ctx_VIX_X_pct_from_close",
            "bond_credit_risk_off_norm",
            "dividend_yield_norm",
            "dividend_payout_ratio_norm",
            "dividend_ex_date_proximity_norm",
            "dividend_pay_date_proximity_norm",
            "dividend_quality_score_norm",
            "dividend_capture_exit_signal_norm",
            "dividend_compound_drawdown_norm",
            "dividend_drip_cash_only_norm",
            "dividend_drip_active_norm",
            "dividend_drip_confidence_norm",
            "dividend_safety_composite_norm",
            "dividend_ex_slippage_risk_norm",
            "dividend_tax_qualified_hold_norm",
            "dividend_growth_momentum_norm",
            "dividend_rebalance_due_norm",
            "capital_flow_outflow_norm",
            "options_negative_bias_norm",
            "data_quality_quote_agreement_norm",
            "ret_6",
            "ret_12",
            "dividend_quality_score_ema_4",
            "dividend_payout_ratio_ema_4",
        ],
        runtime_feature_builder=_runtime_feature_vector,
        runtime_label_builder=_runtime_yield_trap_label,
        sample_filter=_runtime_sample_filter,
        confidence_builder=_runtime_confidence,
        min_confidence=0.38,
        lookback_days=45,
        window=30,
        horizon=12,
        min_samples=224,
        min_sequences=8,
        fallback_trainer=_train_synthetic,
        allow_fallback_on_insufficient_data=False,
        max_best_val_loss=0.6925,
        max_final_val_loss=0.7050,
        min_long_precision=0.52,
        min_short_precision=0.52,
        require_both_sides_precision=True,
        min_acted_accuracy=0.53,
        min_accuracy_lift_over_majority=0.02,
        min_precision_balance_score=0.35,
    )
