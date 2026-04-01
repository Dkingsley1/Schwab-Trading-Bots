import numpy as np

from indicator_bot_common import adx, atr, rolling_std, train_indicator_bot, train_runtime_indicator_bot
from runtime_training_common import (
    feature_ema,
    feature_std,
    future_max_drawdown,
    future_realized_vol,
    future_return,
    observation_feature,
    price_change,
)

_ATR_RUNTIME_MODES = [
    "shadow_equities",
    "shadow_aggressive_equities",
    "shadow_intraday_aggressive_equities",
    "shadow_swing_aggressive_equities",
]
_ATR_SYMBOLS = [
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
]


def build_features(panel):
    c = panel["close"]
    h = panel["high"]
    l = panel["low"]
    r = panel["ret"]

    atr14 = atr(h, l, c, period=14)
    atr_pct = atr14 / (c + 1e-8)
    adx14 = adx(h, l, c, period=14)
    vol20 = rolling_std(r, 20)
    trend_proxy = np.diff(c, prepend=c[0]) / (c + 1e-8)

    return np.stack([r, atr_pct, adx14, vol20, trend_proxy], axis=1)


def _clip01(value):
    return float(np.clip(value, 0.0, 1.0))


def _centered01(value):
    return float((float(value) - 0.5) * 2.0)


def _quote_quality(obs):
    return _clip01(
        (0.65 * observation_feature(obs, "data_quality_quote_agreement_norm", 1.0))
        + (0.35 * (1.0 - observation_feature(obs, "data_quality_quote_deviation_norm", 0.0)))
    )


def _atr_regime_support(obs):
    realized_vol = _clip01(abs(observation_feature(obs, "vol_30m")) / 0.02)
    return _clip01(
        (0.22 * observation_feature(obs, "day_regime_trend_norm"))
        + (0.18 * observation_feature(obs, "day_regime_alignment_norm"))
        + (0.16 * observation_feature(obs, "market_micro_trend_persistence_norm"))
        + (0.12 * observation_feature(obs, "market_micro_relative_volume_norm"))
        + (0.10 * observation_feature(obs, "lead_lag_alignment_norm"))
        + (0.08 * realized_vol)
        + (0.08 * observation_feature(obs, "options_vol_expectation_norm"))
        + (0.06 * _quote_quality(obs))
    )


def _atr_direction_bias(obs):
    return float(
        np.clip(
            (0.22 * observation_feature(obs, "behavior_prior"))
            + (0.18 * _centered01(observation_feature(obs, "market_micro_order_flow_imbalance_norm", 0.5)))
            + (0.16 * _centered01(observation_feature(obs, "futures_specialist_vote", 0.5)))
            + (0.14 * observation_feature(obs, "mom_15m") * 90.0)
            + (0.10 * observation_feature(obs, "mom_5m") * 120.0)
            + (0.08 * observation_feature(obs, "pct_from_close") * 110.0)
            + (0.06 * _centered01(observation_feature(obs, "range_pos", 0.5)))
            + (0.06 * observation_feature(obs, "breadth_advance_decline_norm")),
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
            observation_feature(obs, "mom_15m"),
            observation_feature(obs, "vol_30m"),
            observation_feature(obs, "range_pos"),
            observation_feature(obs, "spread_bps"),
            observation_feature(obs, "queue_depth"),
            observation_feature(obs, "market_data_latency_ms"),
            observation_feature(obs, "ctx_VIX_X_pct_from_close"),
            observation_feature(obs, "breadth_advance_decline_norm"),
            observation_feature(obs, "breadth_risk_off_norm"),
            observation_feature(obs, "market_micro_relative_volume_norm"),
            observation_feature(obs, "market_micro_trend_persistence_norm"),
            observation_feature(obs, "market_micro_order_flow_imbalance_norm"),
            observation_feature(obs, "day_regime_trend_norm"),
            observation_feature(obs, "day_regime_alignment_norm"),
            observation_feature(obs, "lead_lag_alignment_norm"),
            observation_feature(obs, "options_vol_expectation_norm"),
            observation_feature(obs, "data_quality_quote_agreement_norm", 1.0),
            observation_feature(obs, "data_quality_quote_deviation_norm"),
            observation_feature(obs, "behavior_prior"),
            observation_feature(obs, "futures_specialist_vote"),
            price_change(sequence, idx, 3),
            price_change(sequence, idx, 6),
            feature_std(sequence, idx, "pct_from_close", 6),
            feature_std(sequence, idx, "vol_30m", 8),
            feature_ema(sequence, idx, "behavior_prior", 4),
        ],
        dtype=np.float32,
    )


def _runtime_sample_filter(sequence, idx, horizon):
    obs = sequence[idx]
    return (
        observation_feature(obs, "data_quality_quote_agreement_norm", 1.0) >= 0.78
        and observation_feature(obs, "data_quality_quote_deviation_norm", 0.0) <= 0.28
        and abs(observation_feature(obs, "spread_bps", 0.0)) <= 34.0
        and observation_feature(obs, "queue_depth", 0.0) >= 1.0
        and _atr_regime_support(obs) >= 0.22
        and abs(_atr_direction_bias(obs)) >= 0.10
    )


def _runtime_confidence(sequence, idx, horizon):
    obs = sequence[idx]
    support = _atr_regime_support(obs)
    bias = _clip01(abs(_atr_direction_bias(obs)) / 0.9)
    return (
        (0.34 * support)
        + (0.24 * bias)
        + (0.16 * _quote_quality(obs))
        + (0.14 * observation_feature(obs, "market_micro_relative_volume_norm"))
        + (0.12 * observation_feature(obs, "lead_lag_alignment_norm"))
    )


def _runtime_atr_regime_label(sequence, idx, horizon):
    obs = sequence[idx]
    support = _atr_regime_support(obs)
    bias = _atr_direction_bias(obs)
    if support < 0.22 or abs(bias) < 0.10:
        return None

    expected_up = bias >= 0.0
    fwd_ret = future_return(sequence, idx, horizon)
    realized = future_realized_vol(sequence, idx, horizon)
    drawdown = abs(future_max_drawdown(sequence, idx, horizon))
    signed_ret = fwd_ret if expected_up else -fwd_ret
    move_threshold = max(0.00045, 0.00115 - (0.00050 * support))
    if abs(fwd_ret) < move_threshold and realized < 0.018 and drawdown < 0.012:
        return None

    success_score = (
        signed_ret
        + (0.00095 * support)
        + (0.00025 * _quote_quality(obs))
        - (0.28 * drawdown)
        - (0.18 * realized)
    )
    failure_score = (
        (-signed_ret)
        + (0.20 * realized)
        + (0.24 * drawdown)
    )
    if success_score >= 0.00045:
        return 1.0 if expected_up else 0.0
    if failure_score >= 0.00062:
        return 0.0 if expected_up else 1.0
    return None


def _train_synthetic():
    return train_indicator_bot(
        run_tag="brain_refinery_v23_atr_adx_regime",
        feature_names=["ret", "atr_pct", "adx14", "vol20", "trend_proxy"],
        feature_builder=build_features,
    )


def train_brain():
    return train_runtime_indicator_bot(
        run_tag="brain_refinery_v23_atr_adx_regime",
        feature_names=[
            "pct_from_close",
            "mom_5m",
            "mom_15m",
            "vol_30m",
            "range_pos",
            "spread_bps",
            "queue_depth",
            "market_data_latency_ms",
            "ctx_VIX_X_pct_from_close",
            "breadth_advance_decline_norm",
            "breadth_risk_off_norm",
            "market_micro_relative_volume_norm",
            "market_micro_trend_persistence_norm",
            "market_micro_order_flow_imbalance_norm",
            "day_regime_trend_norm",
            "day_regime_alignment_norm",
            "lead_lag_alignment_norm",
            "options_vol_expectation_norm",
            "data_quality_quote_agreement_norm",
            "data_quality_quote_deviation_norm",
            "behavior_prior",
            "futures_specialist_vote",
            "ret_3",
            "ret_6",
            "pct_from_close_std_6",
            "vol_30m_std_8",
            "behavior_prior_ema_4",
        ],
        runtime_feature_builder=_runtime_feature_vector,
        runtime_label_builder=_runtime_atr_regime_label,
        mode_allowlist=_ATR_RUNTIME_MODES,
        symbol_allowlist=_ATR_SYMBOLS,
        sample_filter=_runtime_sample_filter,
        confidence_builder=_runtime_confidence,
        min_confidence=0.34,
        lookback_days=30,
        window=24,
        horizon=6,
        min_samples=288,
        min_sequences=8,
        acted_prob_threshold=0.67,
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


if __name__ == "__main__":
    train_brain()
