import numpy as np

from indicator_bot_common import ema, rolling_std, train_indicator_bot, train_runtime_indicator_bot
from runtime_training_common import (
    feature_ema,
    future_max_drawdown,
    future_realized_vol,
    future_return,
    observation_feature,
    price_change,
    symbol_role_features,
)

_DEF_DIV_ROLE_MAP = {
    "defensive_core": ["PG", "KO", "PEP", "JNJ", "XLP", "XLV", "XLU", "SCHD", "VIG", "DGRO"],
    "energy_income": ["XOM", "CVX", "XLE", "HDV", "VYM"],
    "real_asset_income": ["O", "VNQ", "XLRE", "WPC"],
}
_DEF_DIV_SYMBOLS = sorted({sym for values in _DEF_DIV_ROLE_MAP.values() for sym in values})
_DEF_DIV_MODES = [
    "shadow_bond_equities",
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
    b = panel["bench_ret"]
    q_ret = hold_sample(r, 390)
    q_alpha = q_ret - hold_sample(b, 390)
    q_fast = ema(q_alpha, 6)
    q_slow = ema(q_alpha, 18)
    rel_quality = q_fast - q_slow
    draw_stress = rolling_std(np.minimum(r, 0.0), 60)
    carry = ema(q_ret, 10)

    return np.stack([r, q_alpha, q_fast, q_slow, rel_quality, draw_stress, carry], axis=1)


def _clip01(value):
    return float(np.clip(value, 0.0, 1.0))


def _quote_quality(obs):
    return _clip01(
        (0.65 * observation_feature(obs, "data_quality_quote_agreement_norm", 1.0))
        + (0.35 * (1.0 - observation_feature(obs, "data_quality_quote_deviation_norm", 0.0)))
    )


def _risk_off_signal(obs):
    return _clip01(
        (0.24 * observation_feature(obs, "breadth_risk_off_norm"))
        + (0.22 * observation_feature(obs, "bond_credit_risk_off_norm"))
        + (0.14 * observation_feature(obs, "bond_duration_regime_norm"))
        + (0.14 * _clip01(abs(observation_feature(obs, "ctx_VIX_X_pct_from_close")) / 0.03))
        + (0.14 * observation_feature(obs, "options_negative_bias_norm"))
        + (0.12 * observation_feature(obs, "capital_flow_outflow_norm"))
    )


def _quality_signal(obs):
    return _clip01(
        (0.24 * observation_feature(obs, "dividend_quality_score_norm"))
        + (0.20 * observation_feature(obs, "dividend_safety_composite_norm"))
        + (0.14 * observation_feature(obs, "dividend_growth_momentum_norm"))
        + (0.12 * observation_feature(obs, "dividend_capture_entry_signal_norm"))
        + (0.10 * observation_feature(obs, "dividend_drip_active_norm"))
        + (0.10 * observation_feature(obs, "long_term_quality_dividend_norm"))
        + (0.10 * _quote_quality(obs))
    )


def _role_supports(obs, roles):
    risk_off = _risk_off_signal(obs)
    quality = _quality_signal(obs)
    risk_on = _clip01(
        (0.50 * observation_feature(obs, "flow_risk_on_norm"))
        + (0.30 * observation_feature(obs, "market_micro_relative_volume_norm"))
        + (0.20 * max(observation_feature(obs, "pct_from_close"), 0.0) * 60.0)
    )
    role_weight = max(sum(float(value) for value in roles.values()), 1.0)
    support = (
        (roles["role_defensive_core"] * _clip01((0.55 * risk_off) + (0.35 * quality) + (0.10 * (1.0 - risk_on))))
        + (roles["role_energy_income"] * _clip01((0.35 * quality) + (0.30 * risk_off) + (0.20 * risk_on) + (0.15 * observation_feature(obs, "market_micro_relative_volume_norm"))))
        + (roles["role_real_asset_income"] * _clip01((0.45 * risk_off) + (0.35 * quality) + (0.20 * observation_feature(obs, "bond_duration_regime_norm"))))
    ) / role_weight
    headwind = (
        (roles["role_defensive_core"] * _clip01((0.55 * risk_on) + (0.25 * (1.0 - quality)) + (0.20 * (1.0 - risk_off))))
        + (roles["role_energy_income"] * _clip01((0.45 * (1.0 - quality)) + (0.30 * observation_feature(obs, "capital_flow_outflow_norm")) + (0.25 * (1.0 - risk_on))))
        + (roles["role_real_asset_income"] * _clip01((0.45 * (1.0 - quality)) + (0.35 * observation_feature(obs, "capital_flow_outflow_norm")) + (0.20 * (1.0 - risk_off))))
    ) / role_weight
    return _clip01(support), _clip01(headwind)


def _concentration_bias(obs):
    return float(
        np.clip(
            (0.22 * observation_feature(obs, "behavior_prior"))
            + (0.18 * _quality_signal(obs))
            + (0.16 * _risk_off_signal(obs))
            + (0.12 * observation_feature(obs, "dividend_growth_momentum_norm"))
            + (0.10 * observation_feature(obs, "pct_from_close") * 80.0)
            - (0.12 * observation_feature(obs, "capital_flow_outflow_norm"))
            - (0.10 * observation_feature(obs, "flow_risk_on_norm")),
            -1.0,
            1.0,
        )
    )


def _runtime_feature_vector(sequence, idx):
    obs = sequence[idx]
    roles = symbol_role_features(str(obs.get("symbol") or ""), _DEF_DIV_ROLE_MAP)
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
            observation_feature(obs, "dividend_safety_composite_norm"),
            observation_feature(obs, "dividend_growth_momentum_norm"),
            observation_feature(obs, "dividend_capture_entry_signal_norm"),
            observation_feature(obs, "dividend_drip_active_norm"),
            observation_feature(obs, "long_term_quality_dividend_norm"),
            observation_feature(obs, "capital_flow_outflow_norm"),
            observation_feature(obs, "flow_risk_on_norm"),
            observation_feature(obs, "breadth_risk_off_norm"),
            observation_feature(obs, "options_negative_bias_norm"),
            observation_feature(obs, "market_micro_relative_volume_norm"),
            observation_feature(obs, "data_quality_quote_agreement_norm", 1.0),
            observation_feature(obs, "data_quality_quote_deviation_norm"),
            observation_feature(obs, "behavior_prior"),
            price_change(sequence, idx, 6),
            price_change(sequence, idx, 12),
            feature_ema(sequence, idx, "dividend_quality_score_norm", 4),
            feature_ema(sequence, idx, "breadth_risk_off_norm", 4),
            roles["role_defensive_core"],
            roles["role_energy_income"],
            roles["role_real_asset_income"],
        ],
        dtype=np.float32,
    )


def _runtime_sample_filter(sequence, idx, horizon):
    obs = sequence[idx]
    roles = symbol_role_features(str(obs.get("symbol") or ""), _DEF_DIV_ROLE_MAP)
    support, headwind = _role_supports(obs, roles)
    return (
        sum(float(value) for value in roles.values()) > 0.0
        and observation_feature(obs, "data_quality_quote_agreement_norm", 1.0) >= 0.78
        and observation_feature(obs, "data_quality_quote_deviation_norm", 0.0) <= 0.30
        and max(support, headwind) >= 0.18
        and abs(_concentration_bias(obs)) >= 0.08
    )


def _runtime_confidence(sequence, idx, horizon):
    obs = sequence[idx]
    roles = symbol_role_features(str(obs.get("symbol") or ""), _DEF_DIV_ROLE_MAP)
    support, headwind = _role_supports(obs, roles)
    return (
        (0.30 * max(support, headwind))
        + (0.22 * _clip01(abs(_concentration_bias(obs)) / 0.9))
        + (0.18 * _quality_signal(obs))
        + (0.16 * _risk_off_signal(obs))
        + (0.14 * _quote_quality(obs))
    )


def _runtime_defensive_dividend_label(sequence, idx, horizon):
    obs = sequence[idx]
    roles = symbol_role_features(str(obs.get("symbol") or ""), _DEF_DIV_ROLE_MAP)
    if sum(float(value) for value in roles.values()) <= 0.0:
        return None

    support, headwind = _role_supports(obs, roles)
    bias = _concentration_bias(obs)
    if max(support, headwind) < 0.18 or abs(bias) < 0.08:
        return None

    expected_up = support >= headwind
    fwd_ret = future_return(sequence, idx, horizon)
    drawdown = abs(future_max_drawdown(sequence, idx, horizon))
    realized = future_realized_vol(sequence, idx, horizon)
    signed_ret = fwd_ret if expected_up else -fwd_ret
    move_threshold = max(0.00065, 0.00145 - (0.00060 * max(support, headwind)))
    if abs(fwd_ret) < move_threshold and drawdown < 0.012 and realized < 0.020:
        return None

    success_score = (
        signed_ret
        + (0.00110 * support)
        - (0.00055 * headwind)
        - (0.80 * drawdown)
        - (0.26 * realized)
    )
    failure_score = (
        (-signed_ret)
        + (0.00095 * headwind)
        + (0.65 * drawdown)
        + (0.18 * realized)
    )
    if success_score >= 0.00065 and expected_up:
        return 1.0
    if failure_score >= 0.00095 and not expected_up:
        return 0.0
    return None


def _train_synthetic():
    return train_indicator_bot(
        run_tag="brain_refinery_v99_defensive_dividend_concentration",
        feature_names=["ret", "q_alpha", "q_fast", "q_slow", "rel_quality", "draw_stress", "carry"],
        feature_builder=build_features,
        window=84,
        horizon=12,
    )


def train_brain():
    return train_runtime_indicator_bot(
        run_tag="brain_refinery_v99_defensive_dividend_concentration",
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
            "dividend_safety_composite_norm",
            "dividend_growth_momentum_norm",
            "dividend_capture_entry_signal_norm",
            "dividend_drip_active_norm",
            "long_term_quality_dividend_norm",
            "capital_flow_outflow_norm",
            "flow_risk_on_norm",
            "breadth_risk_off_norm",
            "options_negative_bias_norm",
            "market_micro_relative_volume_norm",
            "data_quality_quote_agreement_norm",
            "data_quality_quote_deviation_norm",
            "behavior_prior",
            "ret_6",
            "ret_12",
            "dividend_quality_score_ema_4",
            "breadth_risk_off_ema_4",
            "role_defensive_core",
            "role_energy_income",
            "role_real_asset_income",
        ],
        runtime_feature_builder=_runtime_feature_vector,
        runtime_label_builder=_runtime_defensive_dividend_label,
        mode_allowlist=_DEF_DIV_MODES,
        symbol_allowlist=_DEF_DIV_SYMBOLS,
        sample_filter=_runtime_sample_filter,
        confidence_builder=_runtime_confidence,
        min_confidence=0.40,
        lookback_days=45,
        window=30,
        horizon=12,
        min_samples=224,
        min_sequences=6,
        acted_prob_threshold=0.66,
        fallback_trainer=_train_synthetic,
        allow_fallback_on_insufficient_data=False,
        max_best_val_loss=0.6925,
        max_final_val_loss=0.7050,
        min_long_precision=0.52,
        min_short_precision=0.52,
        require_both_sides_precision=True,
        min_acted_accuracy=0.53,
        min_accuracy_lift_over_majority=0.02,
        min_precision_balance_score=0.30,
    )


if __name__ == "__main__":
    train_brain()
