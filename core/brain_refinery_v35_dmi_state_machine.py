import numpy as np

from indicator_bot_common import adx, rolling_std, train_indicator_bot, train_runtime_indicator_bot, true_range
from runtime_training_common import (
    feature_ema,
    feature_std,
    future_max_drawdown,
    future_realized_vol,
    future_return,
    observation_feature,
    price_change,
)

_TREND_RUNTIME_MODES = [
    "shadow_equities",
    "shadow_aggressive_equities",
    "shadow_conservative_equities",
    "shadow_intraday_aggressive_equities",
    "shadow_swing_aggressive_equities",
]
_LIQUID_TREND_SYMBOLS = [
    "SPY",
    "QQQ",
    "IWM",
    "DIA",
    "XLK",
    "XLF",
    "XLE",
    "XLI",
    "XLV",
    "SMH",
    "SOXX",
    "AAPL",
    "MSFT",
    "NVDA",
    "AMZN",
    "META",
    "TLT",
    "IEF",
    "LQD",
    "HYG",
]


def dmi(high, low, close, period=14):
    up_move = np.diff(high, prepend=high[0])
    down_move = -np.diff(low, prepend=low[0])
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    tr = true_range(high, low, close) + 1e-8

    plus_di = 100.0 * np.convolve(plus_dm, np.ones(period) / period, mode="same") / (
        np.convolve(tr, np.ones(period) / period, mode="same") + 1e-8
    )
    minus_di = 100.0 * np.convolve(minus_dm, np.ones(period) / period, mode="same") / (
        np.convolve(tr, np.ones(period) / period, mode="same") + 1e-8
    )
    return plus_di, minus_di


def build_features(panel):
    c = panel["close"]
    h = panel["high"]
    l = panel["low"]
    r = panel["ret"]

    plus_di, minus_di = dmi(h, l, c, period=14)
    adx14 = adx(h, l, c, period=14)

    trend_state = np.where(adx14 > 25.0, 1.0, 0.0)
    dir_state = np.sign(plus_di - minus_di)
    state_flip = np.abs(np.diff(dir_state, prepend=dir_state[0]))
    chop = rolling_std(r, 14)

    return np.stack([r, plus_di, minus_di, adx14, trend_state, dir_state, state_flip, chop], axis=1)


def _clip01(value):
    return float(np.clip(value, 0.0, 1.0))


def _centered01(value):
    return float((float(value) - 0.5) * 2.0)


def _is_crypto_symbol(obs) -> bool:
    return "-USD" in str(obs.get("symbol") or "")


def _quote_quality(obs):
    return _clip01(
        (0.65 * observation_feature(obs, "data_quality_quote_agreement_norm", 1.0))
        + (0.35 * (1.0 - observation_feature(obs, "data_quality_quote_deviation_norm", 0.0)))
    )


def _trend_state_support(obs):
    breadth_strength = _clip01((observation_feature(obs, "breadth_advance_decline_norm") + 1.0) / 2.0)
    crypto_support = _clip01(
        (0.32 * observation_feature(obs, "market_micro_trend_persistence_norm"))
        + (0.20 * observation_feature(obs, "market_micro_relative_volume_norm"))
        + (0.16 * observation_feature(obs, "market_crypto_current_alignment_norm", 0.5))
        + (0.14 * observation_feature(obs, "fx_crypto_alignment_norm", 0.5))
        + (0.10 * observation_feature(obs, "crypto_coingecko_momentum_norm", 0.5))
        + (0.08 * (1.0 - observation_feature(obs, "market_crypto_divergence_norm", 0.0)))
    )
    return _clip01(
        (0.18 * observation_feature(obs, "day_regime_trend_norm"))
        + (0.20 * observation_feature(obs, "day_regime_alignment_norm"))
        + (0.18 * observation_feature(obs, "market_micro_trend_persistence_norm"))
        + (0.12 * observation_feature(obs, "market_micro_relative_volume_norm"))
        + (0.08 * max(observation_feature(obs, "day_session_open_norm"), observation_feature(obs, "day_session_power_hour_norm")))
        + (0.10 * breadth_strength)
        + (0.08 * _quote_quality(obs))
        + (0.06 * crypto_support)
    )


def _trend_headwind_support(obs):
    breadth_weakness = _clip01((1.0 - observation_feature(obs, "breadth_advance_decline_norm")) / 2.0)
    session_pressure = _clip01(
        max(
            observation_feature(obs, "day_session_open_norm"),
            observation_feature(obs, "day_session_power_hour_norm"),
        )
    )
    crypto_headwind = _clip01(
        (0.28 * observation_feature(obs, "market_micro_trend_persistence_norm"))
        + (0.24 * observation_feature(obs, "breadth_risk_off_norm"))
        + (0.16 * observation_feature(obs, "market_crypto_divergence_norm", 0.0))
        + (0.14 * (1.0 - observation_feature(obs, "market_crypto_current_alignment_norm", 0.5)))
        + (0.10 * (1.0 - observation_feature(obs, "fx_crypto_alignment_norm", 0.5)))
        + (0.08 * (1.0 - observation_feature(obs, "crypto_coingecko_momentum_norm", 0.5)))
    )
    return _clip01(
        (0.16 * observation_feature(obs, "day_regime_trend_norm"))
        + (0.10 * (1.0 - observation_feature(obs, "day_regime_alignment_norm")))
        + (0.16 * observation_feature(obs, "market_micro_trend_persistence_norm"))
        + (0.16 * observation_feature(obs, "breadth_risk_off_norm"))
        + (0.14 * breadth_weakness)
        + (0.08 * observation_feature(obs, "breadth_sector_dispersion_norm"))
        + (0.08 * session_pressure)
        + (0.06 * _quote_quality(obs))
        + (0.06 * crypto_headwind)
    )


def _direction_bias(obs):
    bias = (
        (0.22 * observation_feature(obs, "behavior_prior"))
        + (0.18 * _centered01(observation_feature(obs, "market_micro_order_flow_imbalance_norm", 0.5)))
        + (0.16 * _centered01(observation_feature(obs, "futures_specialist_vote", 0.5)))
        + (0.14 * observation_feature(obs, "mom_15m") * 90.0)
        + (0.10 * observation_feature(obs, "mom_5m") * 120.0)
        + (0.08 * observation_feature(obs, "pct_from_close") * 120.0)
        + (0.06 * _centered01(observation_feature(obs, "range_pos", 0.5)))
        + (0.06 * observation_feature(obs, "breadth_advance_decline_norm"))
        + (0.06 * _centered01(observation_feature(obs, "market_crypto_current_alignment_norm", 0.5)))
        + (0.04 * _centered01(observation_feature(obs, "crypto_coingecko_momentum_norm", 0.5)))
        + (0.04 * _centered01(observation_feature(obs, "fx_crypto_alignment_norm", 0.5)))
    )
    return float(
        np.clip(bias, -1.0, 1.0)
    )


def _trend_supports(obs):
    return _trend_state_support(obs), _trend_headwind_support(obs)


def _directional_trend_support(obs, bias=None):
    directional_bias = _direction_bias(obs) if bias is None else float(bias)
    support, headwind = _trend_supports(obs)
    directional_support = support if directional_bias >= 0.0 else headwind
    opposing_support = headwind if directional_bias >= 0.0 else support
    return directional_support, opposing_support


def _trend_directional_agreement(obs, bias=None):
    directional_bias = _direction_bias(obs) if bias is None else float(bias)
    target = 1.0 if directional_bias >= 0.0 else -1.0
    signed_components = [
        0.24 * observation_feature(obs, "behavior_prior"),
        0.18 * _centered01(observation_feature(obs, "market_micro_order_flow_imbalance_norm", 0.5)),
        0.16 * _centered01(observation_feature(obs, "futures_specialist_vote", 0.5)),
        0.12 * observation_feature(obs, "mom_15m") * 90.0,
        0.10 * observation_feature(obs, "mom_5m") * 120.0,
        0.08 * observation_feature(obs, "pct_from_close") * 120.0,
        0.05 * _centered01(observation_feature(obs, "range_pos", 0.5)),
        0.04 * observation_feature(obs, "breadth_advance_decline_norm"),
        0.02 * _centered01(observation_feature(obs, "market_crypto_current_alignment_norm", 0.5)),
        0.01 * _centered01(observation_feature(obs, "crypto_coingecko_momentum_norm", 0.5)),
    ]
    aligned = sum(max(target * component, 0.0) for component in signed_components)
    conflicting = sum(max(-(target * component), 0.0) for component in signed_components)
    total = aligned + conflicting
    if total <= 1e-8:
        return 0.0
    return _clip01((aligned - (0.45 * conflicting)) / total)


def _trend_instability(obs):
    spread_drag = _clip01(abs(observation_feature(obs, "spread_bps", 0.0)) / 32.0)
    return _clip01(
        (0.24 * observation_feature(obs, "breadth_sector_dispersion_norm"))
        + (0.20 * observation_feature(obs, "breadth_risk_off_norm"))
        + (0.18 * observation_feature(obs, "infra_risk_throttle_norm"))
        + (0.14 * observation_feature(obs, "day_session_midday_norm"))
        + (0.14 * observation_feature(obs, "data_quality_quote_deviation_norm"))
        + (0.10 * spread_drag)
    )


def _runtime_feature_vector(sequence, idx):
    obs = sequence[idx]
    return np.asarray(
        [
            observation_feature(obs, "pct_from_close"),
            observation_feature(obs, "mom_5m"),
            observation_feature(obs, "mom_15m"),
            observation_feature(obs, "vol_30m"),
            observation_feature(obs, "range_pos"),
            observation_feature(obs, "spread_bps"),
            observation_feature(obs, "queue_depth"),
            observation_feature(obs, "market_data_latency_ms"),
            observation_feature(obs, "market_micro_relative_volume_norm"),
            observation_feature(obs, "market_micro_trend_persistence_norm"),
            observation_feature(obs, "market_micro_order_flow_imbalance_norm"),
            observation_feature(obs, "day_regime_trend_norm"),
            observation_feature(obs, "day_regime_alignment_norm"),
            observation_feature(obs, "breadth_advance_decline_norm"),
            observation_feature(obs, "breadth_sector_dispersion_norm"),
            observation_feature(obs, "breadth_risk_off_norm"),
            observation_feature(obs, "data_quality_quote_agreement_norm", 1.0),
            observation_feature(obs, "data_quality_quote_deviation_norm"),
            observation_feature(obs, "behavior_prior"),
            observation_feature(obs, "futures_specialist_vote"),
            observation_feature(obs, "day_session_open_norm"),
            observation_feature(obs, "day_session_midday_norm"),
            observation_feature(obs, "day_session_power_hour_norm"),
            price_change(sequence, idx, 3),
            price_change(sequence, idx, 6),
            feature_std(sequence, idx, "pct_from_close", 6),
            feature_ema(sequence, idx, "behavior_prior", 4),
        ],
        dtype=np.float32,
    )


def _runtime_sample_filter(sequence, idx, horizon):
    obs = sequence[idx]
    is_crypto = _is_crypto_symbol(obs)
    bias = _direction_bias(obs)
    directional_support, opposing_support = _directional_trend_support(obs, bias)
    min_quote_agreement = 0.74 if is_crypto else 0.78
    max_quote_deviation = 0.34 if is_crypto else 0.28
    max_spread_bps = 46.0 if is_crypto else 34.0
    min_queue_depth = 0.0
    min_directional_support = 0.14 if is_crypto else 0.16
    min_bias = 0.05 if is_crypto else 0.07
    min_support_gap = -0.02 if is_crypto else 0.00
    min_agreement = 0.40 if is_crypto else 0.44
    max_instability = 0.84 if is_crypto else 0.78
    agreement = _trend_directional_agreement(obs, bias)
    if not is_crypto and abs(bias) >= 0.18 and agreement < 0.56:
        return False
    return (
        observation_feature(obs, "data_quality_quote_agreement_norm", 1.0) >= min_quote_agreement
        and observation_feature(obs, "data_quality_quote_deviation_norm", 0.0) <= max_quote_deviation
        and abs(observation_feature(obs, "spread_bps", 0.0)) <= max_spread_bps
        and observation_feature(obs, "queue_depth", 0.0) >= min_queue_depth
        and directional_support >= min_directional_support
        and (directional_support - opposing_support) >= min_support_gap
        and abs(bias) >= min_bias
        and agreement >= min_agreement
        and _trend_instability(obs) <= max_instability
    )


def _runtime_confidence(sequence, idx, horizon):
    obs = sequence[idx]
    bias_raw = _direction_bias(obs)
    bias = _clip01(abs(bias_raw) / 0.9)
    directional_support, opposing_support = _directional_trend_support(obs, bias_raw)
    quote = _quote_quality(obs)
    agreement = _trend_directional_agreement(obs, bias_raw)
    instability = _trend_instability(obs)
    session_focus = _clip01(
        max(
            observation_feature(obs, "day_session_open_norm"),
            observation_feature(obs, "day_session_power_hour_norm"),
        )
    )
    return _clip01(
        (0.30 * directional_support)
        + (0.20 * bias)
        + (0.18 * quote)
        + (0.10 * observation_feature(obs, "market_micro_trend_persistence_norm"))
        + (0.08 * session_focus)
        + (0.10 * agreement)
        - (0.08 * instability)
        - (0.06 * opposing_support)
    )


def _runtime_trend_label(sequence, idx, horizon):
    obs = sequence[idx]
    bias = _direction_bias(obs)
    is_crypto = _is_crypto_symbol(obs)
    directional_support, opposing_support = _directional_trend_support(obs, bias)
    min_directional_support = 0.14 if is_crypto else 0.16
    min_bias = 0.05 if is_crypto else 0.07
    min_support_gap = -0.02 if is_crypto else 0.00
    if directional_support < min_directional_support or abs(bias) < min_bias:
        return None
    if (directional_support - opposing_support) < min_support_gap:
        return None

    agreement = _trend_directional_agreement(obs, bias)
    instability = _trend_instability(obs)
    min_agreement = 0.40 if is_crypto else 0.44
    max_instability = 0.84 if is_crypto else 0.78
    if not is_crypto and instability > 0.66:
        return None
    if agreement < min_agreement or instability > max_instability:
        return None

    expected_up = bias >= 0.0
    fwd_ret = future_return(sequence, idx, horizon)
    realized = future_realized_vol(sequence, idx, horizon)
    drawdown = abs(future_max_drawdown(sequence, idx, horizon))
    signed_ret = fwd_ret if expected_up else -fwd_ret
    move_threshold = (
        max(0.00028, 0.00080 - (0.00058 * directional_support))
        if is_crypto
        else max(0.00040, 0.00100 - (0.00060 * directional_support))
    )
    move_threshold += (
        (0.00010 * max(0.0, 0.56 - agreement))
        + (0.00008 * instability)
        + (0.00008 * opposing_support)
    )
    realized_floor = 0.020 if is_crypto else 0.015
    drawdown_floor = 0.0130 if is_crypto else 0.0100
    if abs(fwd_ret) < move_threshold and realized < realized_floor and drawdown < drawdown_floor:
        return None

    support_bonus = (
        (0.0010 * directional_support)
        + (0.00025 * _quote_quality(obs))
        + (0.00015 * observation_feature(obs, "market_micro_relative_volume_norm"))
        + (0.00018 * agreement)
        + (0.00018 * observation_feature(obs, "market_crypto_current_alignment_norm", 0.5))
        + (0.00012 * observation_feature(obs, "crypto_coingecko_momentum_norm", 0.5))
        - (0.00016 * opposing_support)
    )
    penalty = (
        (0.28 * drawdown)
        + (0.20 * realized)
        + (0.00025 * observation_feature(obs, "breadth_risk_off_norm"))
        + (0.00015 * observation_feature(obs, "breadth_sector_dispersion_norm"))
        + (0.00010 * observation_feature(obs, "infra_risk_throttle_norm"))
        + (0.00014 * instability)
    )
    success_score = signed_ret + support_bonus - penalty
    failure_score = (
        (-signed_ret)
        + (0.20 * realized)
        + (0.26 * drawdown)
        + (0.00020 * observation_feature(obs, "breadth_sector_dispersion_norm"))
        + (0.00012 * observation_feature(obs, "infra_risk_throttle_norm"))
        + (0.00016 * instability)
        + (0.00018 * max(opposing_support - directional_support, 0.0))
    )
    if is_crypto:
        success_gate = 0.00040
        failure_gate = 0.00054
    else:
        success_gate = 0.00048
        failure_gate = 0.00062
    success_gate += (0.00010 * max(0.0, 0.56 - agreement)) + (0.00008 * instability)
    failure_gate += (0.00008 * max(0.0, 0.54 - agreement)) + (0.00008 * instability)
    if success_score >= success_gate:
        return 1.0 if expected_up else 0.0
    if failure_score >= failure_gate:
        return 0.0 if expected_up else 1.0
    return None


def _train_synthetic():
    return train_indicator_bot(
        run_tag="brain_refinery_v35_dmi_state_machine",
        feature_names=["ret", "plus_di", "minus_di", "adx14", "trend_state", "dir_state", "state_flip", "chop"],
        feature_builder=build_features,
    )


def train_brain():
    return train_runtime_indicator_bot(
        run_tag="brain_refinery_v35_dmi_state_machine",
        feature_names=[
            "pct_from_close",
            "mom_5m",
            "mom_15m",
            "vol_30m",
            "range_pos",
            "spread_bps",
            "queue_depth",
            "market_data_latency_ms",
            "market_micro_relative_volume_norm",
            "market_micro_trend_persistence_norm",
            "market_micro_order_flow_imbalance_norm",
            "day_regime_trend_norm",
            "day_regime_alignment_norm",
            "breadth_advance_decline_norm",
            "breadth_sector_dispersion_norm",
            "breadth_risk_off_norm",
            "data_quality_quote_agreement_norm",
            "data_quality_quote_deviation_norm",
            "behavior_prior",
            "futures_specialist_vote",
            "day_session_open_norm",
            "day_session_midday_norm",
            "day_session_power_hour_norm",
            "ret_3",
            "ret_6",
            "pct_from_close_std_6",
            "behavior_prior_ema_4",
        ],
        runtime_feature_builder=_runtime_feature_vector,
        runtime_label_builder=_runtime_trend_label,
        mode_allowlist=_TREND_RUNTIME_MODES,
        symbol_allowlist=_LIQUID_TREND_SYMBOLS,
        sample_filter=_runtime_sample_filter,
        confidence_builder=_runtime_confidence,
        min_confidence=0.46,
        sample_stride=4,
        lookback_days=60,
        window=18,
        horizon=6,
        batch_size=16,
        min_samples=80,
        min_sequences=4,
        min_positive_samples=20,
        min_negative_samples=20,
        acted_prob_threshold=0.70,
        fallback_trainer=_train_synthetic,
        allow_fallback_on_insufficient_data=False,
        max_best_val_loss=0.6900,
        max_final_val_loss=0.7050,
        min_long_precision=0.52,
        min_short_precision=0.52,
        require_both_sides_precision=True,
        min_acted_accuracy=0.60,
        min_long_acted_count=4,
        min_short_acted_count=4,
        min_accuracy_lift_over_majority=0.02,
        min_precision_balance_score=0.30,
        max_acted_coverage=0.30,
    )


if __name__ == "__main__":
    train_brain()
