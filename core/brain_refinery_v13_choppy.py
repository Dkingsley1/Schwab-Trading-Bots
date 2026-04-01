import numpy as np

from indicator_bot_common import train_price_indicator_bot, train_runtime_indicator_bot
from runtime_training_common import (
    feature_ema,
    feature_std,
    future_realized_vol,
    future_return,
    observation_feature,
    price_change,
)

_CHOPPY_RUNTIME_MODES = [
    "shadow_equities",
    "shadow_aggressive_equities",
    "shadow_conservative_equities",
    "shadow_dividend_equities",
    "shadow_intraday_aggressive_equities",
]
_CHOPPY_SYMBOLS = [
    "SPY",
    "QQQ",
    "DIA",
    "IWM",
    "XLK",
    "XLF",
    "XLI",
    "XLV",
    "XLP",
    "XLU",
    "XLRE",
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


def simulate_choppy(n=5000):
    prices = np.zeros(n)
    prices[0] = 100.0
    mu = 100.0
    theta = 0.2
    vol = 1.0
    for i in range(1, n):
        noise = vol * np.random.randn()
        mean_pull = theta * (mu - prices[i - 1])
        prices[i] = max(0.1, prices[i - 1] + mean_pull + noise)
    return prices


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
            observation_feature(obs, "vol_30m"),
            observation_feature(obs, "range_pos"),
            observation_feature(obs, "spread_bps"),
            observation_feature(obs, "ctx_VIX_X_pct_from_close"),
            observation_feature(obs, "breadth_advance_decline_norm"),
            observation_feature(obs, "breadth_risk_off_norm"),
            observation_feature(obs, "options_negative_bias_norm"),
            observation_feature(obs, "options_vol_expectation_norm"),
            observation_feature(obs, "data_quality_quote_agreement_norm", 1.0),
            observation_feature(obs, "data_quality_quote_deviation_norm"),
            observation_feature(obs, "behavior_prior"),
            observation_feature(obs, "day_regime_chop_norm"),
            observation_feature(obs, "day_regime_trend_norm"),
            observation_feature(obs, "day_regime_alignment_norm"),
            observation_feature(obs, "market_micro_relative_volume_norm"),
            observation_feature(obs, "market_micro_order_flow_imbalance_norm"),
            price_change(sequence, idx, 1),
            price_change(sequence, idx, 3),
            feature_std(sequence, idx, "pct_from_close", 6),
            feature_std(sequence, idx, "vol_30m", 6),
            feature_ema(sequence, idx, "pct_from_close", 4),
            feature_ema(sequence, idx, "behavior_prior", 4),
        ],
        dtype=np.float32,
    )


def _clip01(value):
    return float(np.clip(value, 0.0, 1.0))


def _stretch_direction(obs):
    range_component = (observation_feature(obs, "range_pos") - 0.5) * 2.0
    price_component = observation_feature(obs, "pct_from_close") * 140.0
    momentum_component = observation_feature(obs, "mom_5m") * 110.0
    return float((0.45 * range_component) + (0.30 * price_component) + (0.25 * momentum_component))


def _chop_support(obs):
    return _clip01(
        (0.30 * observation_feature(obs, "day_regime_chop_norm"))
        + (0.18 * _clip01(1.0 - observation_feature(obs, "day_regime_alignment_norm")))
        + (0.14 * _clip01(1.0 - abs(observation_feature(obs, "behavior_prior")) * 2.0))
        + (0.12 * _clip01(abs(_stretch_direction(obs)) / 1.2))
        + (0.10 * observation_feature(obs, "data_quality_quote_agreement_norm", 1.0))
        + (0.08 * observation_feature(obs, "options_vol_expectation_norm"))
        + (0.08 * _clip01(abs(observation_feature(obs, "market_micro_order_flow_imbalance_norm"))))
    )


def _runtime_sample_filter(sequence, idx, horizon):
    obs = sequence[idx]
    stretch = abs(_stretch_direction(obs))
    return (
        observation_feature(obs, "data_quality_quote_agreement_norm", 1.0) >= 0.84
        and observation_feature(obs, "data_quality_quote_deviation_norm", 0.0) <= 0.22
        and abs(observation_feature(obs, "spread_bps")) <= 24.0
        and observation_feature(obs, "day_regime_chop_norm") >= 0.28
        and observation_feature(obs, "day_regime_trend_norm") <= 0.58
        and observation_feature(obs, "day_regime_alignment_norm") <= 0.54
        and observation_feature(obs, "market_micro_relative_volume_norm") >= 0.18
        and abs(observation_feature(obs, "behavior_prior")) <= 0.36
        and stretch >= 0.22
    )


def _runtime_confidence(sequence, idx, horizon):
    obs = sequence[idx]
    edge_stretch = _clip01(abs(_stretch_direction(obs)) / 1.2)
    quote_ok = _clip01(
        0.70 * observation_feature(obs, "data_quality_quote_agreement_norm", 1.0)
        + 0.30 * (1.0 - observation_feature(obs, "data_quality_quote_deviation_norm", 0.0))
    )
    anti_trend = _clip01(1.0 - observation_feature(obs, "day_regime_trend_norm"))
    anti_alignment = _clip01(1.0 - observation_feature(obs, "day_regime_alignment_norm"))
    return (
        (0.30 * _chop_support(obs))
        + (0.22 * edge_stretch)
        + (0.16 * anti_trend)
        + (0.12 * anti_alignment)
        + (0.10 * _clip01(1.0 - abs(observation_feature(obs, "behavior_prior")) * 2.0))
        + (0.10 * quote_ok)
    )


def _runtime_choppy_label(sequence, idx, horizon):
    obs = sequence[idx]
    chop = _chop_support(obs)
    stretch = _stretch_direction(obs)
    anti_trend = _clip01(1.0 - observation_feature(obs, "day_regime_trend_norm"))
    if chop < 0.30 or anti_trend < 0.42 or abs(stretch) < 0.24:
        return None

    expected_up = stretch < 0.0
    fwd_ret = future_return(sequence, idx, horizon)
    realized = future_realized_vol(sequence, idx, horizon)
    signed_ret = fwd_ret if expected_up else -fwd_ret
    move_threshold = max(0.00035, 0.00095 - (0.00040 * chop))
    if abs(fwd_ret) < move_threshold and realized < 0.017:
        return None

    success_score = (
        signed_ret
        + (0.0011 * chop)
        + (0.0005 * anti_trend)
        - (0.17 * realized)
        - (0.00025 * observation_feature(obs, "breadth_risk_off_norm"))
    )
    failure_score = (
        (-signed_ret)
        + (0.14 * realized)
        + (0.00010 * observation_feature(obs, "breadth_risk_off_norm"))
    )
    if success_score >= 0.00035:
        return 1.0 if expected_up else 0.0
    if failure_score >= 0.00050:
        return 0.0 if expected_up else 1.0
    return None


def _train_synthetic():
    return train_price_indicator_bot(
        run_tag="brain_refinery_v13_choppy",
        feature_names=["returns", "sma10", "ema10", "rsi14", "vol10"],
        feature_builder=build_features,
        price_simulator=simulate_choppy,
    )


def train_brain():
    return train_runtime_indicator_bot(
        run_tag="brain_refinery_v13_choppy",
        feature_names=[
            "pct_from_close",
            "mom_5m",
            "vol_30m",
            "range_pos",
            "spread_bps",
            "ctx_VIX_X_pct_from_close",
            "breadth_advance_decline_norm",
            "breadth_risk_off_norm",
            "options_negative_bias_norm",
            "options_vol_expectation_norm",
            "data_quality_quote_agreement_norm",
            "data_quality_quote_deviation_norm",
            "behavior_prior",
            "day_regime_chop_norm",
            "day_regime_trend_norm",
            "day_regime_alignment_norm",
            "market_micro_relative_volume_norm",
            "market_micro_order_flow_imbalance_norm",
            "ret_1",
            "ret_3",
            "pct_from_close_std_6",
            "vol_30m_std_6",
            "pct_from_close_ema_4",
            "behavior_prior_ema_4",
        ],
        runtime_feature_builder=_runtime_feature_vector,
        runtime_label_builder=_runtime_choppy_label,
        mode_allowlist=_CHOPPY_RUNTIME_MODES,
        symbol_allowlist=_CHOPPY_SYMBOLS,
        sample_filter=_runtime_sample_filter,
        confidence_builder=_runtime_confidence,
        min_confidence=0.42,
        sample_stride=3,
        lookback_days=45,
        window=18,
        horizon=2,
        min_samples=128,
        min_sequences=6,
        min_positive_samples=24,
        min_negative_samples=24,
        acted_prob_threshold=0.60,
        fallback_trainer=_train_synthetic,
        allow_fallback_on_insufficient_data=False,
        max_best_val_loss=0.6925,
        max_final_val_loss=0.7050,
        min_long_precision=0.53,
        min_short_precision=0.53,
        require_both_sides_precision=True,
        min_acted_accuracy=0.55,
        min_long_acted_count=4,
        min_short_acted_count=4,
        min_accuracy_lift_over_majority=0.02,
        min_label_balance_score=0.20,
        min_precision_balance_score=0.40,
    )


if __name__ == "__main__":
    train_brain()
