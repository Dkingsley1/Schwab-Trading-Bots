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
    rolling_drawdown as runtime_rolling_drawdown,
    symbol_role_features,
)

_CREDIT_ROLE_MAP = {
    "hy_credit": ["HYG", "JNK", "USHY"],
    "ig_credit": ["LQD", "IGIB"],
    "treasury": ["TLT", "IEF", "SHY", "TIP"],
}
_EXPANDED_CREDIT_SYMBOLS = sorted({sym for values in _CREDIT_ROLE_MAP.values() for sym in values})
_CREDIT_RUNTIME_MODES = [
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


def rolling_drawdown(close, window=220):
    out = np.zeros_like(close)
    for i in range(len(close)):
        start = max(0, i - window + 1)
        peak = np.max(close[start : i + 1])
        out[i] = (close[i] - peak) / (peak + 1e-8)
    return out


def build_features(panel):
    c = panel["close"]
    r = panel["ret"]
    b = panel["bench_ret"]

    q_ret = hold_sample(r, 1170)
    q_bench = hold_sample(b, 1170)
    rel = q_ret - q_bench

    rel_fast = ema(rel, 6)
    rel_slow = ema(rel, 16)
    spread_mom = rel_fast - rel_slow

    dd = rolling_drawdown(c, window=260)
    dd_fast = ema(dd, 8)
    dd_slow = ema(dd, 21)

    vol = rolling_std(r, 120)
    tail_risk = np.maximum(-dd_fast, 0.0) + np.maximum(vol - ema(vol, 20), 0.0)
    credit_rotation = spread_mom - 0.45 * tail_risk

    return np.stack([r, rel, rel_fast, rel_slow, spread_mom, dd, dd_fast, dd_slow, vol, tail_risk, credit_rotation], axis=1)


def _runtime_feature_vector(sequence, idx):
    obs = sequence[idx]
    roles = symbol_role_features(str(obs.get("symbol") or ""), _CREDIT_ROLE_MAP)
    return np.asarray(
        [
            observation_feature(obs, "pct_from_close"),
            observation_feature(obs, "mom_5m"),
            observation_feature(obs, "vol_30m"),
            observation_feature(obs, "range_pos"),
            observation_feature(obs, "spread_bps"),
            observation_feature(obs, "ctx_UUP_pct_from_close"),
            observation_feature(obs, "fx_usd_strength_norm"),
            observation_feature(obs, "fx_eurusd_momentum_norm"),
            observation_feature(obs, "fx_usdjpy_momentum_norm"),
            observation_feature(obs, "fx_proxy_agreement_norm"),
            observation_feature(obs, "fx_risk_on_alignment_norm"),
            observation_feature(obs, "fx_crypto_alignment_norm"),
            observation_feature(obs, "fx_corr_confidence_norm"),
            observation_feature(obs, "bond_credit_risk_on_norm"),
            observation_feature(obs, "bond_credit_risk_off_norm"),
            observation_feature(obs, "bond_carry_roll_norm"),
            observation_feature(obs, "bond_credit_spread_level_norm"),
            observation_feature(obs, "bond_credit_spread_change_norm"),
            observation_feature(obs, "bond_hy_ig_flow_norm"),
            observation_feature(obs, "bond_nav_stress_norm"),
            observation_feature(obs, "bond_nav_discount_norm"),
            observation_feature(obs, "calendar_macro_surprise_norm"),
            observation_feature(obs, "options_negative_bias_norm"),
            observation_feature(obs, "options_roll_yield_norm"),
            observation_feature(obs, "options_vol_expectation_norm"),
            observation_feature(obs, "options_put_call_oi_ratio_norm"),
            observation_feature(obs, "options_gamma_exposure_norm"),
            observation_feature(obs, "options_unusual_flow_norm"),
            observation_feature(obs, "breadth_risk_off_norm"),
            observation_feature(obs, "data_quality_quote_agreement_norm", 1.0),
            observation_feature(obs, "data_quality_quote_deviation_norm"),
            observation_feature(obs, "market_micro_credit_flow_norm"),
            observation_feature(obs, "market_micro_relative_volume_norm"),
            observation_feature(obs, "behavior_prior"),
            price_change(sequence, idx, 3),
            price_change(sequence, idx, 6),
            feature_ema(sequence, idx, "bond_credit_risk_on_norm", 5),
            feature_ema(sequence, idx, "bond_credit_risk_off_norm", 5),
            feature_std(sequence, idx, "pct_from_close", 8),
            runtime_rolling_drawdown(sequence, idx, 12),
            roles["role_hy_credit"],
            roles["role_ig_credit"],
            roles["role_treasury"],
        ],
        dtype=np.float32,
    )


def _clip01(value):
    return float(np.clip(value, 0.0, 1.0))


def _credit_risk_on_signal(obs):
    spread_level = _clip01(observation_feature(obs, "bond_credit_spread_level_norm"))
    nav_stress = _clip01(observation_feature(obs, "bond_nav_stress_norm"))
    breadth_risk_off = _clip01(observation_feature(obs, "breadth_risk_off_norm"))
    return _clip01(
        (0.24 * _clip01(observation_feature(obs, "bond_credit_risk_on_norm")))
        + (0.18 * _clip01(observation_feature(obs, "bond_carry_roll_norm")))
        + (0.16 * _clip01(1.0 - spread_level))
        + (0.12 * _clip01(1.0 - nav_stress))
        + (0.12 * _clip01(observation_feature(obs, "bond_hy_ig_flow_norm")))
        + (0.10 * _clip01(1.0 - breadth_risk_off))
        + (0.08 * _clip01(observation_feature(obs, "market_micro_credit_flow_norm")))
    )


def _credit_risk_off_signal(obs):
    return _clip01(
        (0.28 * _clip01(observation_feature(obs, "bond_credit_risk_off_norm")))
        + (0.20 * _clip01(observation_feature(obs, "bond_nav_stress_norm")))
        + (0.16 * _clip01(observation_feature(obs, "breadth_risk_off_norm")))
        + (0.14 * _clip01(observation_feature(obs, "options_negative_bias_norm")))
        + (0.12 * _clip01(observation_feature(obs, "options_gamma_exposure_norm")))
        + (0.10 * _clip01(observation_feature(obs, "calendar_macro_surprise_norm")))
    )


def _credit_quality_signal(obs):
    quote_quality = _clip01(
        0.65 * observation_feature(obs, "data_quality_quote_agreement_norm", 1.0)
        + 0.35 * (1.0 - observation_feature(obs, "data_quality_quote_deviation_norm", 0.0))
    )
    return _clip01(
        (0.26 * _clip01(observation_feature(obs, "bond_carry_roll_norm")))
        + (0.20 * _clip01(1.0 - observation_feature(obs, "bond_nav_discount_norm")))
        + (0.18 * _clip01(1.0 - observation_feature(obs, "options_vol_expectation_norm")))
        + (0.16 * _clip01(1.0 - observation_feature(obs, "spread_bps")))
        + (0.10 * _clip01(observation_feature(obs, "behavior_prior")))
        + (0.10 * quote_quality)
    )


def _role_supports(obs, roles):
    risk_on = _credit_risk_on_signal(obs)
    risk_off = _credit_risk_off_signal(obs)
    quality = _credit_quality_signal(obs)
    role_weight = max(sum(float(value) for value in roles.values()), 1.0)
    support = (
        (roles["role_hy_credit"] * risk_on)
        + (roles["role_ig_credit"] * _clip01((0.55 * quality) + (0.25 * risk_on) + (0.20 * risk_off)))
        + (roles["role_treasury"] * risk_off)
    ) / role_weight
    headwind = (
        (roles["role_hy_credit"] * risk_off)
        + (roles["role_ig_credit"] * _clip01((0.45 * risk_off) + (0.35 * (1.0 - quality)) + (0.20 * risk_on)))
        + (roles["role_treasury"] * risk_on)
    ) / role_weight
    return _clip01(support), _clip01(headwind)


def _runtime_sample_filter(sequence, idx, horizon):
    obs = sequence[idx]
    roles = symbol_role_features(str(obs.get("symbol") or ""), _CREDIT_ROLE_MAP)
    role_support, role_headwind = _role_supports(obs, roles)
    return (
        observation_feature(obs, "data_quality_quote_agreement_norm", 1.0) >= 0.74
        and observation_feature(obs, "data_quality_quote_deviation_norm", 0.0) <= 0.34
        and max(_credit_risk_on_signal(obs), _credit_risk_off_signal(obs), _credit_quality_signal(obs)) >= 0.18
        and max(role_support, role_headwind) >= 0.18
    )


def _runtime_confidence(sequence, idx, horizon):
    obs = sequence[idx]
    roles = symbol_role_features(str(obs.get("symbol") or ""), _CREDIT_ROLE_MAP)
    role_support, role_headwind = _role_supports(obs, roles)
    quote_quality = _clip01(
        0.65 * observation_feature(obs, "data_quality_quote_agreement_norm", 1.0)
        + 0.35 * (1.0 - observation_feature(obs, "data_quality_quote_deviation_norm", 0.0))
    )
    return (
        (0.24 * max(role_support, role_headwind))
        + (0.22 * _credit_quality_signal(obs))
        + (0.20 * _credit_risk_on_signal(obs))
        + (0.18 * _credit_risk_off_signal(obs))
        + (0.16 * quote_quality)
    )


def _runtime_credit_rotation_label(sequence, idx, horizon):
    obs = sequence[idx]
    roles = symbol_role_features(str(obs.get("symbol") or ""), _CREDIT_ROLE_MAP)
    if sum(float(value) for value in roles.values()) <= 0.0:
        return None

    role_support, role_headwind = _role_supports(obs, roles)
    if max(role_support, role_headwind) < 0.18 or abs(role_support - role_headwind) < 0.04:
        return None

    fwd_ret = future_return(sequence, idx, horizon)
    dd = abs(future_max_drawdown(sequence, idx, horizon))
    realized = future_realized_vol(sequence, idx, horizon)
    spread_bps = _clip01(observation_feature(obs, "spread_bps"))
    move_threshold = max(0.0008, 0.0018 - (0.0009 * max(role_support, role_headwind)))
    if abs(fwd_ret) < move_threshold and dd < 0.012 and realized < 0.020:
        return None

    support_score = (
        fwd_ret
        + (0.0012 * role_support)
        - (0.0006 * role_headwind)
        - (0.0003 * spread_bps)
        - (0.70 * dd)
        - (0.24 * realized)
    )
    failure_score = (
        (-fwd_ret)
        + (0.0010 * role_headwind)
        - (0.0004 * role_support)
        + (0.55 * dd)
        + (0.18 * realized)
        + (0.0002 * spread_bps)
    )
    if support_score >= 0.0007 and role_support > role_headwind:
        return 1.0
    if failure_score >= 0.0009 and role_headwind >= role_support:
        return 0.0
    return None


def _train_synthetic():
    return train_indicator_bot(
        run_tag="brain_refinery_v96_credit_spread_rotation_bot",
        feature_names=[
            "ret",
            "rel",
            "rel_fast",
            "rel_slow",
            "spread_mom",
            "drawdown",
            "dd_fast",
            "dd_slow",
            "vol",
            "tail_risk",
            "credit_rotation",
        ],
        feature_builder=build_features,
        window=84,
        horizon=24,
    )


if __name__ == "__main__":
    train_runtime_indicator_bot(
        run_tag="brain_refinery_v96_credit_spread_rotation_bot",
        feature_names=[
            "pct_from_close",
            "mom_5m",
            "vol_30m",
            "range_pos",
            "spread_bps",
            "ctx_UUP_pct_from_close",
            "fx_usd_strength_norm",
            "fx_eurusd_momentum_norm",
            "fx_usdjpy_momentum_norm",
            "fx_proxy_agreement_norm",
            "fx_risk_on_alignment_norm",
            "fx_crypto_alignment_norm",
            "fx_corr_confidence_norm",
            "bond_credit_risk_on_norm",
            "bond_credit_risk_off_norm",
            "bond_carry_roll_norm",
            "bond_credit_spread_level_norm",
            "bond_credit_spread_change_norm",
            "bond_hy_ig_flow_norm",
            "bond_nav_stress_norm",
            "bond_nav_discount_norm",
            "calendar_macro_surprise_norm",
            "options_negative_bias_norm",
            "options_roll_yield_norm",
            "options_vol_expectation_norm",
            "options_put_call_oi_ratio_norm",
            "options_gamma_exposure_norm",
            "options_unusual_flow_norm",
            "breadth_risk_off_norm",
            "data_quality_quote_agreement_norm",
            "data_quality_quote_deviation_norm",
            "market_micro_credit_flow_norm",
            "market_micro_relative_volume_norm",
            "behavior_prior",
            "ret_3",
            "ret_6",
            "bond_credit_risk_on_ema_5",
            "bond_credit_risk_off_ema_5",
            "pct_from_close_std_8",
            "drawdown_12",
            "role_hy_credit",
            "role_ig_credit",
            "role_treasury",
        ],
        runtime_feature_builder=_runtime_feature_vector,
        runtime_label_builder=_runtime_credit_rotation_label,
        lookback_days=45,
        mode_allowlist=_CREDIT_RUNTIME_MODES,
        symbol_allowlist=_EXPANDED_CREDIT_SYMBOLS,
        sample_filter=_runtime_sample_filter,
        confidence_builder=_runtime_confidence,
        min_confidence=0.32,
        window=24,
        horizon=12,
        min_samples=192,
        min_sequences=4,
        fallback_trainer=_train_synthetic,
        allow_fallback_on_insufficient_data=False,
        max_best_val_loss=0.6925,
        max_final_val_loss=0.7050,
        min_long_precision=0.52,
        min_short_precision=0.52,
        require_both_sides_precision=True,
        min_acted_accuracy=0.53,
        min_accuracy_lift_over_majority=0.02,
        min_label_balance_score=0.25,
        min_precision_balance_score=0.35,
    )
