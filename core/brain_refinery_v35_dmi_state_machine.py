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
    "shadow_crypto",
    "shadow_crypto_futures_crypto",
    "shadow_dividend_equities",
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
    "XLY",
    "XLP",
    "XLB",
    "XLU",
    "XLRE",
    "XLC",
    "SMH",
    "SOXX",
    "AAPL",
    "MSFT",
    "NVDA",
    "BTC-USD",
    "ETH-USD",
    "SOL-USD",
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
    min_quote_agreement = 0.78 if is_crypto else 0.82
    max_quote_deviation = 0.30 if is_crypto else 0.22
    max_spread_bps = 42.0 if is_crypto else 28.0
    min_queue_depth = 0.0 if is_crypto else 1.0
    min_support = 0.22 if is_crypto else 0.26
    min_bias = 0.11 if is_crypto else 0.15
    return (
        observation_feature(obs, "data_quality_quote_agreement_norm", 1.0) >= min_quote_agreement
        and observation_feature(obs, "data_quality_quote_deviation_norm", 0.0) <= max_quote_deviation
        and abs(observation_feature(obs, "spread_bps", 0.0)) <= max_spread_bps
        and observation_feature(obs, "queue_depth", 0.0) >= min_queue_depth
        and _trend_state_support(obs) >= min_support
        and abs(_direction_bias(obs)) >= min_bias
    )


def _runtime_confidence(sequence, idx, horizon):
    obs = sequence[idx]
    support = _trend_state_support(obs)
    bias = _clip01(abs(_direction_bias(obs)) / 0.9)
    quote = _quote_quality(obs)
    session_focus = _clip01(
        max(
            observation_feature(obs, "day_session_open_norm"),
            observation_feature(obs, "day_session_power_hour_norm"),
        )
    )
    return (
        (0.34 * support)
        + (0.24 * bias)
        + (0.18 * quote)
        + (0.12 * observation_feature(obs, "market_micro_trend_persistence_norm"))
        + (0.12 * session_focus)
    )


def _runtime_trend_label(sequence, idx, horizon):
    obs = sequence[idx]
    support = _trend_state_support(obs)
    bias = _direction_bias(obs)
    is_crypto = _is_crypto_symbol(obs)
    min_support = 0.22 if is_crypto else 0.26
    min_bias = 0.11 if is_crypto else 0.15
    if support < min_support or abs(bias) < min_bias:
        return None

    expected_up = bias >= 0.0
    fwd_ret = future_return(sequence, idx, horizon)
    realized = future_realized_vol(sequence, idx, horizon)
    drawdown = abs(future_max_drawdown(sequence, idx, horizon))
    signed_ret = fwd_ret if expected_up else -fwd_ret
    move_threshold = (
        max(0.00040, 0.00100 - (0.00056 * support))
        if is_crypto
        else max(0.00058, 0.00128 - (0.00060 * support))
    )
    realized_floor = 0.024 if is_crypto else 0.018
    drawdown_floor = 0.0145 if is_crypto else 0.0115
    if abs(fwd_ret) < move_threshold and realized < realized_floor and drawdown < drawdown_floor:
        return None

    support_bonus = (
        (0.0010 * support)
        + (0.00025 * _quote_quality(obs))
        + (0.00015 * observation_feature(obs, "market_micro_relative_volume_norm"))
        + (0.00018 * observation_feature(obs, "market_crypto_current_alignment_norm", 0.5))
        + (0.00012 * observation_feature(obs, "crypto_coingecko_momentum_norm", 0.5))
    )
    penalty = (
        (0.28 * drawdown)
        + (0.20 * realized)
        + (0.00025 * observation_feature(obs, "breadth_risk_off_norm"))
        + (0.00015 * observation_feature(obs, "breadth_sector_dispersion_norm"))
        + (0.00010 * observation_feature(obs, "infra_risk_throttle_norm"))
    )
    success_score = signed_ret + support_bonus - penalty
    failure_score = (
        (-signed_ret)
        + (0.20 * realized)
        + (0.26 * drawdown)
        + (0.00020 * observation_feature(obs, "breadth_sector_dispersion_norm"))
        + (0.00012 * observation_feature(obs, "infra_risk_throttle_norm"))
    )
    success_gate = 0.00036 if is_crypto else 0.00048
    failure_gate = 0.00050 if is_crypto else 0.00062
    if is_crypto:
        success_gate = 0.00042
        failure_gate = 0.00058
    else:
        success_gate = 0.00058
        failure_gate = 0.00074
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
        min_confidence=0.44,
        sample_stride=8,
        lookback_days=45,
        window=24,
        horizon=6,
        min_samples=224,
        min_sequences=6,
        min_positive_samples=40,
        min_negative_samples=40,
        acted_prob_threshold=0.65,
        fallback_trainer=_train_synthetic,
        allow_fallback_on_insufficient_data=False,
        max_best_val_loss=0.6900,
        max_final_val_loss=0.7050,
        min_long_precision=0.52,
        min_short_precision=0.52,
        require_both_sides_precision=True,
        min_acted_accuracy=0.53,
        min_long_acted_count=6,
        min_short_acted_count=6,
        min_accuracy_lift_over_majority=0.02,
        min_precision_balance_score=0.35,
    )


if __name__ == "__main__":
    train_brain()
