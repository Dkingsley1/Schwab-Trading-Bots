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

_VOL_RUNTIME_MODES = [
    "shadow_equities",
    "shadow_aggressive_equities",
    "shadow_swing_aggressive_equities",
    "shadow_conservative_equities",
]
_VOL_SYMBOLS = [
    "SPY",
    "QQQ",
    "IWM",
    "DIA",
    "TLT",
    "XLF",
    "XLK",
    "XLE",
    "XLV",
    "SMH",
]


def build_features(panel):
    r = panel["ret"]
    vix = panel["vix"]

    vol_fast = rolling_std(r, 10)
    vol_slow = rolling_std(r, 40)
    vol_ratio = vol_fast / (vol_slow + 1e-8)

    vol_accel = np.diff(vol_fast, prepend=vol_fast[0])
    vix_chg = np.diff(vix, prepend=vix[0]) / (vix + 1e-8)
    regime = ema(vol_ratio + np.maximum(vix_chg, 0.0), 8)

    return np.stack([r, vol_fast, vol_slow, vol_ratio, vol_accel, vix_chg, regime], axis=1)


def _clip01(value):
    return float(np.clip(value, 0.0, 1.0))


def _centered_feature(obs, name):
    return abs(observation_feature(obs, name, 0.5) - 0.5) * 2.0


def _quote_quality(obs):
    return _clip01(
        (0.65 * observation_feature(obs, "data_quality_quote_agreement_norm", 1.0))
        + (0.35 * (1.0 - observation_feature(obs, "data_quality_quote_deviation_norm", 0.0)))
    )


def _vol_easing_signal(obs):
    vix_drag = _clip01(1.0 - min(abs(observation_feature(obs, "ctx_VIX_X_pct_from_close")) / 0.03, 1.0))
    return _clip01(
        (0.22 * vix_drag)
        + (0.20 * _clip01(1.0 - observation_feature(obs, "options_negative_bias_norm")))
        + (0.18 * _clip01(1.0 - observation_feature(obs, "breadth_risk_off_norm")))
        + (0.14 * _clip01(1.0 - _centered_feature(obs, "options_iv_term_structure_norm")))
        + (0.12 * _clip01(1.0 - _centered_feature(obs, "options_iv_skew_norm")))
        + (0.08 * observation_feature(obs, "day_regime_alignment_norm"))
        + (0.06 * _quote_quality(obs))
    )


def _vol_stress_signal(obs):
    return _clip01(
        (0.24 * _clip01(min(abs(observation_feature(obs, "ctx_VIX_X_pct_from_close")) / 0.03, 1.0)))
        + (0.20 * observation_feature(obs, "options_negative_bias_norm"))
        + (0.18 * observation_feature(obs, "options_vol_expectation_norm"))
        + (0.12 * _centered_feature(obs, "options_iv_term_structure_norm"))
        + (0.10 * _centered_feature(obs, "options_iv_skew_norm"))
        + (0.10 * observation_feature(obs, "breadth_risk_off_norm"))
        + (0.06 * observation_feature(obs, "options_unusual_flow_norm"))
    )


def _vol_direction_bias(obs):
    return float(
        np.clip(
            (0.22 * observation_feature(obs, "behavior_prior"))
            + (0.18 * observation_feature(obs, "mom_5m") * 110.0)
            + (0.12 * observation_feature(obs, "pct_from_close") * 90.0)
            + (0.14 * (observation_feature(obs, "breadth_advance_decline_norm") - observation_feature(obs, "breadth_risk_off_norm")))
            - (0.20 * observation_feature(obs, "options_negative_bias_norm"))
            - (0.14 * observation_feature(obs, "options_vol_expectation_norm")),
            -1.0,
            1.0,
        )
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
            observation_feature(obs, "options_chain_available"),
            observation_feature(obs, "options_iv_atm_norm"),
            observation_feature(obs, "options_iv_skew_norm"),
            observation_feature(obs, "options_iv_term_structure_norm"),
            observation_feature(obs, "options_iv_realized_spread_norm"),
            observation_feature(obs, "options_vol_expectation_norm"),
            observation_feature(obs, "options_negative_bias_norm"),
            observation_feature(obs, "options_unusual_flow_norm"),
            observation_feature(obs, "options_gamma_expiry_skew_norm"),
            observation_feature(obs, "breadth_advance_decline_norm"),
            observation_feature(obs, "breadth_risk_off_norm"),
            observation_feature(obs, "market_micro_relative_volume_norm"),
            observation_feature(obs, "market_micro_order_flow_imbalance_norm"),
            observation_feature(obs, "data_quality_quote_agreement_norm", 1.0),
            observation_feature(obs, "data_quality_quote_deviation_norm"),
            observation_feature(obs, "behavior_prior"),
            price_change(sequence, idx, 3),
            price_change(sequence, idx, 6),
            feature_std(sequence, idx, "pct_from_close", 6),
            feature_ema(sequence, idx, "options_iv_skew_norm", 4),
            feature_ema(sequence, idx, "options_vol_expectation_norm", 4),
        ],
        dtype=np.float32,
    )


def _runtime_sample_filter(sequence, idx, horizon):
    obs = sequence[idx]
    directional_edge = (_vol_easing_signal(obs) - _vol_stress_signal(obs)) + (0.35 * _vol_direction_bias(obs))
    return (
        observation_feature(obs, "options_chain_available") >= 0.5
        and observation_feature(obs, "data_quality_quote_agreement_norm", 1.0) >= 0.78
        and observation_feature(obs, "data_quality_quote_deviation_norm", 0.0) <= 0.30
        and max(_vol_easing_signal(obs), _vol_stress_signal(obs)) >= 0.18
        and abs(directional_edge) >= 0.06
    )


def _runtime_confidence(sequence, idx, horizon):
    obs = sequence[idx]
    directional_edge = abs((_vol_easing_signal(obs) - _vol_stress_signal(obs)) + (0.35 * _vol_direction_bias(obs)))
    return (
        (0.30 * max(_vol_easing_signal(obs), _vol_stress_signal(obs)))
        + (0.12 * _clip01(directional_edge / 0.35))
        + (0.22 * _quote_quality(obs))
        + (0.08 * observation_feature(obs, "options_vol_expectation_norm"))
        + (0.14 * observation_feature(obs, "market_micro_relative_volume_norm"))
        + (0.14 * observation_feature(obs, "options_unusual_flow_norm"))
    )


def _runtime_vol_transition_label(sequence, idx, horizon):
    obs = sequence[idx]
    easing = _vol_easing_signal(obs)
    stress = _vol_stress_signal(obs)
    directional_edge = (easing - stress) + (0.35 * _vol_direction_bias(obs))
    if max(easing, stress) < 0.18 or abs(directional_edge) < 0.05:
        return None

    expected_up = directional_edge >= 0.0
    fwd_ret = future_return(sequence, idx, horizon)
    realized = future_realized_vol(sequence, idx, horizon)
    drawdown = abs(future_max_drawdown(sequence, idx, horizon))
    signed_ret = fwd_ret if expected_up else -fwd_ret
    move_threshold = max(0.00045, 0.00105 - (0.00040 * max(easing, stress)))
    if abs(fwd_ret) < move_threshold and realized < 0.018 and drawdown < 0.012:
        return None

    success_score = (
        signed_ret
        + (0.00085 * easing)
        - (0.00075 * stress)
        + (0.00020 * max(_vol_direction_bias(obs), 0.0))
        + (0.00020 * _quote_quality(obs))
        - (0.22 * realized)
        - (0.26 * drawdown)
    )
    failure_score = (
        (-signed_ret)
        + (0.00090 * stress)
        - (0.00040 * easing)
        + (0.00020 * max(-_vol_direction_bias(obs), 0.0))
        + (0.18 * realized)
        + (0.22 * drawdown)
    )
    if success_score >= 0.00035:
        return 1.0 if expected_up else 0.0
    if failure_score >= 0.00050:
        return 0.0 if expected_up else 1.0
    return None


def _train_synthetic():
    return train_indicator_bot(
        run_tag="brain_refinery_v51_vol_regime_transition",
        feature_names=["ret", "vol_fast", "vol_slow", "vol_ratio", "vol_accel", "vix_chg", "regime"],
        feature_builder=build_features,
        window=42,
        horizon=3,
    )


def train_brain():
    return train_runtime_indicator_bot(
        run_tag="brain_refinery_v51_vol_regime_transition",
        feature_names=[
            "pct_from_close",
            "mom_5m",
            "vol_30m",
            "range_pos",
            "ctx_VIX_X_pct_from_close",
            "options_chain_available",
            "options_iv_atm_norm",
            "options_iv_skew_norm",
            "options_iv_term_structure_norm",
            "options_iv_realized_spread_norm",
            "options_vol_expectation_norm",
            "options_negative_bias_norm",
            "options_unusual_flow_norm",
            "options_gamma_expiry_skew_norm",
            "breadth_advance_decline_norm",
            "breadth_risk_off_norm",
            "market_micro_relative_volume_norm",
            "market_micro_order_flow_imbalance_norm",
            "data_quality_quote_agreement_norm",
            "data_quality_quote_deviation_norm",
            "behavior_prior",
            "ret_3",
            "ret_6",
            "pct_from_close_std_6",
            "options_iv_skew_norm_ema_4",
            "options_vol_expectation_norm_ema_4",
        ],
        runtime_feature_builder=_runtime_feature_vector,
        runtime_label_builder=_runtime_vol_transition_label,
        mode_allowlist=_VOL_RUNTIME_MODES,
        symbol_allowlist=_VOL_SYMBOLS,
        sample_filter=_runtime_sample_filter,
        confidence_builder=_runtime_confidence,
        min_confidence=0.34,
        lookback_days=45,
        window=24,
        horizon=6,
        min_samples=192,
        min_sequences=6,
        min_positive_samples=32,
        min_negative_samples=32,
        acted_prob_threshold=0.67,
        fallback_trainer=_train_synthetic,
        allow_fallback_on_insufficient_data=False,
        max_best_val_loss=0.6925,
        max_final_val_loss=0.7050,
        min_long_precision=0.52,
        min_short_precision=0.52,
        require_both_sides_precision=True,
        min_acted_accuracy=0.53,
        min_long_acted_count=4,
        min_short_acted_count=4,
        min_accuracy_lift_over_majority=0.02,
        min_precision_balance_score=0.35,
    )


if __name__ == "__main__":
    train_brain()
