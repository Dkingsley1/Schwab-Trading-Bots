import numpy as np
from indicator_bot_common import train_price_indicator_bot, train_runtime_indicator_bot
from runtime_training_common import (
    feature_ema,
    feature_std,
    observation_feature,
    price_change,
    selective_direction_label_builder,
)

_TOP_CRYPTO_SYMBOLS = ["BTC-USD", "ETH-USD", "SOL-USD"]


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
        out[i] = np.std(x[start:i + 1])
    return out


def simulate_mean_revert(n=5000):
    prices = np.zeros(n)
    prices[0] = 100.0
    mu = 100.0
    theta = 0.05
    vol = 0.5
    for i in range(1, n):
        dx = theta * (mu - prices[i - 1]) + vol * np.random.randn()
        prices[i] = max(0.1, prices[i - 1] + dx)
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
    return np.asarray(
        [
            observation_feature(sequence[idx], "pct_from_close"),
            observation_feature(sequence[idx], "mom_5m"),
            observation_feature(sequence[idx], "vol_30m"),
            observation_feature(sequence[idx], "range_pos"),
            observation_feature(sequence[idx], "spread_bps"),
            observation_feature(sequence[idx], "ctx_VIX_X_pct_from_close"),
            observation_feature(sequence[idx], "ctx_UUP_pct_from_close"),
            observation_feature(sequence[idx], "breadth_advance_decline_norm"),
            observation_feature(sequence[idx], "breadth_risk_off_norm"),
            observation_feature(sequence[idx], "options_iv_atm_norm"),
            observation_feature(sequence[idx], "options_iv_skew_norm"),
            observation_feature(sequence[idx], "options_vol_expectation_norm"),
            observation_feature(sequence[idx], "data_quality_quote_agreement_norm"),
            observation_feature(sequence[idx], "behavior_prior"),
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


def _runtime_sample_filter(sequence, idx, horizon):
    obs = sequence[idx]
    range_pos = observation_feature(obs, "range_pos")
    stretch = max(
        abs(observation_feature(obs, "pct_from_close")),
        abs(observation_feature(obs, "mom_5m")),
    )
    trend_bias = abs(observation_feature(obs, "behavior_prior"))
    quote_agreement = observation_feature(obs, "data_quality_quote_agreement_norm", 1.0)
    spread = abs(observation_feature(obs, "spread_bps"))
    return (
        (range_pos <= 0.22 or range_pos >= 0.78)
        and stretch >= 0.001
        and trend_bias <= 0.45
        and quote_agreement >= 0.85
        and spread <= 30.0
    )


def _runtime_confidence(sequence, idx, horizon):
    obs = sequence[idx]
    range_pos = observation_feature(obs, "range_pos")
    edge_stretch = _clip01(abs(range_pos - 0.5) * 2.0)
    stretch = _clip01(
        max(
            abs(observation_feature(obs, "pct_from_close")) * 150.0,
            abs(observation_feature(obs, "mom_5m")) * 120.0,
        )
    )
    low_trend = _clip01(1.0 - min(abs(observation_feature(obs, "behavior_prior")) * 2.0, 1.0))
    quote_ok = _clip01(observation_feature(obs, "data_quality_quote_agreement_norm", 1.0))
    vol_context = _clip01(abs(observation_feature(obs, "options_vol_expectation_norm")))
    return (0.28 * edge_stretch) + (0.24 * stretch) + (0.18 * low_trend) + (0.15 * quote_ok) + (0.15 * vol_context)


def _train_synthetic():
    return train_price_indicator_bot(
        run_tag="brain_refinery_v8_mean_revert",
        feature_names=["returns", "sma10", "ema10", "rsi14", "vol10"],
        feature_builder=build_features,
        price_simulator=simulate_mean_revert,
    )


if __name__ == "__main__":
    train_runtime_indicator_bot(
        run_tag="brain_refinery_v8_mean_revert",
        feature_names=[
            "pct_from_close",
            "mom_5m",
            "vol_30m",
            "range_pos",
            "spread_bps",
            "ctx_VIX_X_pct_from_close",
            "ctx_UUP_pct_from_close",
            "breadth_advance_decline_norm",
            "breadth_risk_off_norm",
            "options_iv_atm_norm",
            "options_iv_skew_norm",
            "options_vol_expectation_norm",
            "data_quality_quote_agreement_norm",
            "behavior_prior",
            "ret_1",
            "ret_3",
            "pct_from_close_std_6",
            "vol_30m_std_6",
            "pct_from_close_ema_4",
            "behavior_prior_ema_4",
        ],
        runtime_feature_builder=_runtime_feature_vector,
        runtime_label_builder=selective_direction_label_builder(min_abs_return=0.00045),
        mode_allowlist=["shadow_crypto"],
        symbol_allowlist=_TOP_CRYPTO_SYMBOLS,
        sample_filter=_runtime_sample_filter,
        confidence_builder=_runtime_confidence,
        min_confidence=0.46,
        lookback_days=21,
        window=24,
        horizon=4,
        min_samples=224,
        min_sequences=3,
        acted_prob_threshold=0.68,
        fallback_trainer=_train_synthetic,
    )
