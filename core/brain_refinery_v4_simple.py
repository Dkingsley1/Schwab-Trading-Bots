import numpy as np

from indicator_bot_common import train_price_indicator_bot, train_runtime_indicator_bot
from runtime_training_common import (
    feature_ema,
    feature_std,
    future_max_drawdown,
    future_realized_vol,
    future_return,
    observation_feature,
    price_change,
)

_SIMPLE_RUNTIME_MODES = [
    "shadow_equities",
    "shadow_conservative_equities",
    "shadow_dividend_equities",
    "shadow_swing_aggressive_equities",
]
_SIMPLE_SYMBOLS = [
    "SPY",
    "QQQ",
    "DIA",
    "IWM",
    "SCHD",
    "VIG",
    "DGRO",
    "XLK",
    "XLF",
    "XLI",
    "XLV",
    "XLP",
    "XLU",
    "XLRE",
    "XLE",
    "AAPL",
    "MSFT",
    "NVDA",
    "AMZN",
    "META",
]


def ema(x, span):
    alpha = 2 / (span + 1)
    out = np.zeros_like(x)
    out[0] = x[0]
    for i in range(1, len(x)):
        out[i] = alpha * x[i] + (1 - alpha) * out[i - 1]
    return out


def rsi(prices, period=14):
    deltas = np.diff(prices, prepend=prices[0])
    gains = np.maximum(deltas, 0)
    losses = np.maximum(-deltas, 0)
    avg_gain = ema(gains, period)
    avg_loss = ema(losses, period) + 1e-8
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def rolling_std(x, window):
    out = np.zeros_like(x)
    for i in range(len(x)):
        start = max(0, i - window + 1)
        out[i] = np.std(x[start : i + 1])
    return out


def simulate_simple(n=5000):
    t = np.linspace(0, 200, n)
    prices = np.sin(t) + np.random.normal(0, 0.05, n)
    prices = (prices - prices.min()) + 1.0
    return prices * 100.0


def build_features(prices):
    returns = np.log(prices[1:] / prices[:-1])
    returns = np.concatenate([[0.0], returns])
    sma = np.convolve(prices, np.ones(10) / 10, mode="same")
    ema10 = ema(prices, 10)
    rsi14 = rsi(prices, 14)
    vol10 = rolling_std(returns, 10)
    return np.stack([returns, sma, ema10, rsi14, vol10], axis=1)


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
            observation_feature(obs, "ctx_VIX_X_pct_from_close"),
            observation_feature(obs, "ctx_UUP_pct_from_close"),
            observation_feature(obs, "breadth_advance_decline_norm"),
            observation_feature(obs, "breadth_risk_off_norm"),
            observation_feature(obs, "fx_usd_strength_norm"),
            observation_feature(obs, "fx_eurusd_momentum_norm"),
            observation_feature(obs, "fx_proxy_agreement_norm"),
            observation_feature(obs, "fx_risk_on_alignment_norm"),
            observation_feature(obs, "fx_corr_confidence_norm"),
            observation_feature(obs, "options_vol_expectation_norm"),
            observation_feature(obs, "options_negative_bias_norm"),
            observation_feature(obs, "data_quality_quote_agreement_norm", 1.0),
            observation_feature(obs, "data_quality_quote_deviation_norm"),
            observation_feature(obs, "behavior_prior"),
            observation_feature(obs, "day_regime_trend_norm"),
            observation_feature(obs, "day_regime_alignment_norm"),
            observation_feature(obs, "swing_regime_trend_norm"),
            observation_feature(obs, "swing_regime_alignment_norm"),
            price_change(sequence, idx, 3),
            price_change(sequence, idx, 6),
            feature_std(sequence, idx, "pct_from_close", 6),
            feature_std(sequence, idx, "vol_30m", 8),
            feature_ema(sequence, idx, "behavior_prior", 4),
            feature_ema(sequence, idx, "pct_from_close", 4),
        ],
        dtype=np.float32,
    )


def _clip01(value):
    return float(np.clip(value, 0.0, 1.0))


def _direction_bias(obs):
    return float(
        (0.30 * observation_feature(obs, "behavior_prior"))
        + (0.24 * observation_feature(obs, "mom_15m"))
        + (0.18 * observation_feature(obs, "pct_from_close"))
        + (0.14 * observation_feature(obs, "breadth_advance_decline_norm"))
        + (0.14 * ((observation_feature(obs, "range_pos") - 0.5) * 2.0))
    )


def _trend_support(obs):
    return _clip01(
        (0.24 * observation_feature(obs, "day_regime_trend_norm"))
        + (0.16 * observation_feature(obs, "day_regime_alignment_norm"))
        + (0.18 * observation_feature(obs, "swing_regime_trend_norm"))
        + (0.12 * observation_feature(obs, "swing_regime_alignment_norm"))
        + (0.12 * _clip01(abs(observation_feature(obs, "behavior_prior")) * 2.0))
        + (0.10 * _clip01(abs(observation_feature(obs, "mom_15m")) * 90.0))
        + (0.08 * observation_feature(obs, "data_quality_quote_agreement_norm", 1.0))
    )


def _runtime_sample_filter(sequence, idx, horizon):
    obs = sequence[idx]
    trend_support = _trend_support(obs)
    bias = abs(_direction_bias(obs))
    return (
        observation_feature(obs, "data_quality_quote_agreement_norm", 1.0) >= 0.82
        and observation_feature(obs, "data_quality_quote_deviation_norm", 0.0) <= 0.24
        and abs(observation_feature(obs, "spread_bps")) <= 28.0
        and observation_feature(obs, "queue_depth", 0.0) >= 1.0
        and trend_support >= 0.24
        and bias >= 0.0010
    )


def _runtime_confidence(sequence, idx, horizon):
    obs = sequence[idx]
    quote_ok = _clip01(
        0.70 * observation_feature(obs, "data_quality_quote_agreement_norm", 1.0)
        + 0.30 * (1.0 - observation_feature(obs, "data_quality_quote_deviation_norm", 0.0))
    )
    flow = _clip01(abs(observation_feature(obs, "mom_15m")) * 90.0)
    breadth = _clip01(abs(observation_feature(obs, "breadth_advance_decline_norm")))
    return (
        (0.28 * _trend_support(obs))
        + (0.24 * _clip01(abs(_direction_bias(obs)) * 120.0))
        + (0.16 * flow)
        + (0.12 * breadth)
        + (0.10 * quote_ok)
        + (0.10 * _clip01(1.0 - observation_feature(obs, "breadth_risk_off_norm")))
    )


def _runtime_trend_label(sequence, idx, horizon):
    obs = sequence[idx]
    support = _trend_support(obs)
    bias = _direction_bias(obs)
    if support < 0.24 or abs(bias) < 0.0010:
        return None

    expected_up = bias >= 0.0
    fwd_ret = future_return(sequence, idx, horizon)
    realized = future_realized_vol(sequence, idx, horizon)
    drawdown = abs(future_max_drawdown(sequence, idx, horizon))
    signed_ret = fwd_ret if expected_up else -fwd_ret
    move_threshold = max(0.00085, 0.00195 - (0.00085 * support))
    if abs(fwd_ret) < move_threshold and realized < 0.018 and drawdown < 0.0105:
        return None

    quote_quality = _clip01(
        0.70 * observation_feature(obs, "data_quality_quote_agreement_norm", 1.0)
        + 0.30 * (1.0 - observation_feature(obs, "data_quality_quote_deviation_norm", 0.0))
    )
    success_score = (
        signed_ret
        + (0.0012 * support)
        + (0.00025 * quote_quality)
        - (0.18 * realized)
        - (0.24 * drawdown)
    )
    failure_score = (-signed_ret) + (0.15 * realized) + (0.22 * drawdown)
    if success_score >= 0.00078:
        return 1.0 if expected_up else 0.0
    if failure_score >= 0.00105:
        return 0.0 if expected_up else 1.0
    return None


def _train_synthetic():
    return train_price_indicator_bot(
        run_tag="brain_refinery_v4_simple",
        feature_names=["returns", "sma10", "ema10", "rsi14", "vol10"],
        feature_builder=build_features,
        price_simulator=simulate_simple,
    )


def train_brain():
    return train_runtime_indicator_bot(
        run_tag="brain_refinery_v4_simple",
        feature_names=[
            "pct_from_close",
            "mom_5m",
            "mom_15m",
            "vol_30m",
            "range_pos",
            "spread_bps",
            "queue_depth",
            "ctx_VIX_X_pct_from_close",
            "ctx_UUP_pct_from_close",
            "breadth_advance_decline_norm",
            "breadth_risk_off_norm",
            "fx_usd_strength_norm",
            "fx_eurusd_momentum_norm",
            "fx_proxy_agreement_norm",
            "fx_risk_on_alignment_norm",
            "fx_corr_confidence_norm",
            "options_vol_expectation_norm",
            "options_negative_bias_norm",
            "data_quality_quote_agreement_norm",
            "data_quality_quote_deviation_norm",
            "behavior_prior",
            "day_regime_trend_norm",
            "day_regime_alignment_norm",
            "swing_regime_trend_norm",
            "swing_regime_alignment_norm",
            "ret_3",
            "ret_6",
            "pct_from_close_std_6",
            "vol_30m_std_8",
            "behavior_prior_ema_4",
            "pct_from_close_ema_4",
        ],
        runtime_feature_builder=_runtime_feature_vector,
        runtime_label_builder=_runtime_trend_label,
        mode_allowlist=_SIMPLE_RUNTIME_MODES,
        symbol_allowlist=_SIMPLE_SYMBOLS,
        sample_filter=_runtime_sample_filter,
        confidence_builder=_runtime_confidence,
        min_confidence=0.44,
        sample_stride=4,
        lookback_days=30,
        window=20,
        horizon=6,
        min_samples=1200,
        min_sequences=12,
        acted_prob_threshold=0.66,
        fallback_trainer=_train_synthetic,
        allow_fallback_on_insufficient_data=False,
        max_best_val_loss=0.6925,
        max_final_val_loss=0.7050,
        min_long_precision=0.54,
        min_short_precision=0.54,
        require_both_sides_precision=True,
        min_acted_accuracy=0.57,
        min_accuracy_lift_over_majority=0.03,
        min_label_balance_score=0.20,
        min_precision_balance_score=0.45,
    )


if __name__ == "__main__":
    train_brain()
