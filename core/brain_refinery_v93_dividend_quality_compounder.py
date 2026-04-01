import numpy as np

from indicator_bot_common import adx, atr, ema, rolling_std, train_indicator_bot, train_runtime_indicator_bot
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
    h = panel["high"]
    l = panel["low"]
    r = panel["ret"]
    b = panel["bench_ret"]

    # Slow-sampled momentum proxies for quality compounding behavior.
    q_ret = hold_sample(r, 1170)
    y_ret = hold_sample(r, 4680)
    q_mom = ema(q_ret, 8)
    y_mom = ema(y_ret, 8)

    rel_alpha = q_ret - hold_sample(b, 1170)
    adx_l = adx(h, l, c, period=28)
    atr_l = atr(h, l, c, period=28) / (c + 1e-8)

    dvol = downside_vol(r, window=150)
    vol = rolling_std(r, 150)

    # Proxy "dividend quality": persistent relative strength with controlled downside.
    quality_compound = (np.maximum(q_mom + y_mom + 0.5 * rel_alpha, 0.0)) / (dvol + 0.5 * vol + 1e-8)

    return np.stack([r, q_mom, y_mom, rel_alpha, adx_l, atr_l, dvol, quality_compound], axis=1)


def _clip01(value):
    return float(np.clip(value, 0.0, 1.0))


def _quality_compound_signal(obs):
    return _clip01(
        0.22 * observation_feature(obs, "dividend_quality_score_norm")
        + 0.18 * observation_feature(obs, "dividend_safety_composite_norm")
        + 0.14 * observation_feature(obs, "dividend_compound_bias_norm")
        + 0.14 * observation_feature(obs, "dividend_compound_growth_norm")
        + 0.10 * observation_feature(obs, "dividend_growth_momentum_norm")
        + 0.10 * observation_feature(obs, "long_term_quality_dividend_norm")
        + 0.08 * observation_feature(obs, "dividend_capture_entry_signal_norm")
        + 0.08 * observation_feature(obs, "dividend_drip_active_norm")
        + 0.06 * observation_feature(obs, "dividend_drip_recent_reinvest_norm")
        + 0.04 * observation_feature(obs, "dividend_drip_confidence_norm")
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
            observation_feature(obs, "bond_duration_regime_norm"),
            observation_feature(obs, "bond_credit_risk_off_norm"),
            observation_feature(obs, "dividend_yield_norm"),
            observation_feature(obs, "dividend_payout_ratio_norm"),
            observation_feature(obs, "dividend_quality_score_norm"),
            observation_feature(obs, "dividend_capture_entry_signal_norm"),
            observation_feature(obs, "dividend_compound_bias_norm"),
            observation_feature(obs, "dividend_compound_growth_norm"),
            observation_feature(obs, "dividend_compound_drawdown_norm"),
            observation_feature(obs, "dividend_drip_active_norm"),
            observation_feature(obs, "dividend_drip_recent_reinvest_norm"),
            observation_feature(obs, "dividend_drip_share_credit_norm"),
            observation_feature(obs, "dividend_drip_confidence_norm"),
            observation_feature(obs, "dividend_safety_composite_norm"),
            observation_feature(obs, "dividend_growth_momentum_norm"),
            observation_feature(obs, "long_term_quality_dividend_norm"),
            observation_feature(obs, "capital_flow_outflow_norm"),
            observation_feature(obs, "options_negative_bias_norm"),
            observation_feature(obs, "data_quality_quote_agreement_norm"),
            price_change(sequence, idx, 6),
            price_change(sequence, idx, 12),
            feature_ema(sequence, idx, "dividend_quality_score_norm", 4),
            feature_ema(sequence, idx, "dividend_compound_growth_norm", 4),
        ],
        dtype=np.float32,
    )


def _runtime_sample_filter(sequence, idx, horizon):
    obs = sequence[idx]
    return (
        observation_feature(obs, "data_quality_quote_agreement_norm", 1.0) >= 0.74
        and observation_feature(obs, "dividend_payout_ratio_norm", 0.0) <= 0.95
        and _quality_compound_signal(obs) >= 0.24
    )


def _runtime_confidence(sequence, idx, horizon):
    obs = sequence[idx]
    quote_quality = _clip01(observation_feature(obs, "data_quality_quote_agreement_norm", 1.0))
    growth_signal = _clip01(
        0.45 * observation_feature(obs, "dividend_compound_growth_norm")
        + 0.35 * observation_feature(obs, "dividend_growth_momentum_norm")
        + 0.10 * observation_feature(obs, "dividend_capture_entry_signal_norm")
        + 0.10 * observation_feature(obs, "dividend_drip_recent_reinvest_norm")
    )
    safety_signal = _clip01(
        0.55 * observation_feature(obs, "dividend_safety_composite_norm")
        + 0.45 * observation_feature(obs, "dividend_quality_score_norm")
    )
    stress_penalty = _clip01(
        0.45 * observation_feature(obs, "capital_flow_outflow_norm")
        + 0.30 * observation_feature(obs, "options_negative_bias_norm")
        + 0.25 * observation_feature(obs, "bond_credit_risk_off_norm")
    )
    return (0.34 * _quality_compound_signal(obs)) + (0.24 * growth_signal) + (0.22 * safety_signal) + (0.12 * quote_quality) + (0.08 * (1.0 - stress_penalty))


def _runtime_quality_compound_label(sequence, idx, horizon):
    obs = sequence[idx]
    quality_signal = _quality_compound_signal(obs)
    if quality_signal < 0.24:
        return None

    fwd_ret = future_return(sequence, idx, horizon)
    dd = abs(future_max_drawdown(sequence, idx, horizon))
    realized = future_realized_vol(sequence, idx, horizon)
    growth_signal = observation_feature(obs, "dividend_growth_momentum_norm")
    stress = max(
        observation_feature(obs, "capital_flow_outflow_norm"),
        observation_feature(obs, "options_negative_bias_norm"),
        observation_feature(obs, "bond_credit_risk_off_norm"),
    )
    move_threshold = max(0.0007, 0.0015 - (0.0006 * quality_signal))
    if abs(fwd_ret) < move_threshold and dd < 0.012:
        return None

    support_score = (
        fwd_ret
        + (0.0012 * quality_signal)
        + (0.0006 * growth_signal)
        - (0.80 * dd)
        - (0.30 * realized)
        - (0.0008 * stress)
    )
    failure_score = (
        (-fwd_ret)
        + (0.70 * dd)
        + (0.20 * realized)
        + (0.0006 * stress)
        - (0.0006 * quality_signal)
    )
    if support_score >= 0.0007:
        return 1.0
    if failure_score >= 0.0010:
        return 0.0
    return None


def _train_synthetic():
    return train_indicator_bot(
        run_tag="brain_refinery_v93_dividend_quality_compounder",
        feature_names=[
            "ret",
            "q_mom",
            "y_mom",
            "rel_alpha",
            "adx_long",
            "atr_long",
            "downside_vol",
            "quality_compound",
        ],
        feature_builder=build_features,
        window=84,
        horizon=24,
    )


if __name__ == "__main__":
    train_runtime_indicator_bot(
        run_tag="brain_refinery_v93_dividend_quality_compounder",
        feature_names=[
            "pct_from_close",
            "mom_5m",
            "vol_30m",
            "range_pos",
            "ctx_VIX_X_pct_from_close",
            "bond_duration_regime_norm",
            "bond_credit_risk_off_norm",
            "dividend_yield_norm",
            "dividend_payout_ratio_norm",
            "dividend_quality_score_norm",
            "dividend_capture_entry_signal_norm",
            "dividend_compound_bias_norm",
            "dividend_compound_growth_norm",
            "dividend_compound_drawdown_norm",
            "dividend_drip_active_norm",
            "dividend_drip_recent_reinvest_norm",
            "dividend_drip_share_credit_norm",
            "dividend_drip_confidence_norm",
            "dividend_safety_composite_norm",
            "dividend_growth_momentum_norm",
            "long_term_quality_dividend_norm",
            "capital_flow_outflow_norm",
            "options_negative_bias_norm",
            "data_quality_quote_agreement_norm",
            "ret_6",
            "ret_12",
            "dividend_quality_score_ema_4",
            "dividend_compound_growth_ema_4",
        ],
        runtime_feature_builder=_runtime_feature_vector,
        runtime_label_builder=_runtime_quality_compound_label,
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
