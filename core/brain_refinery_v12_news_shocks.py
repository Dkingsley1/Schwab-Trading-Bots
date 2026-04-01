import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
import json
from datetime import datetime
import os

from indicator_bot_common import train_price_indicator_bot, train_runtime_indicator_bot
from runtime_training_common import (
    feature_ema,
    feature_std,
    future_max_drawdown,
    future_realized_vol,
    future_return,
    observation_feature,
    price_change,
    symbol_role_features,
)

# -----------------------------
# Feature engineering helpers
# -----------------------------
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
        out[i] = np.std(x[start:i+1])
    return out

# -----------------------------
# Model
# -----------------------------
class TradingBrain(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.layer1 = nn.Linear(input_dim, 128)
        self.layer2 = nn.Linear(128, 64)
        self.layer3 = nn.Linear(64, 32)
        self.out = nn.Linear(32, 1)

    def __call__(self, x):
        x = nn.relu(self.layer1(x))
        x = nn.relu(self.layer2(x))
        x = nn.relu(self.layer3(x))
        return self.out(x)

def loss_fn(model, x, y):
    logits = model(x)
    probs = mx.sigmoid(logits)
    return nn.losses.binary_cross_entropy(probs, y)

# -----------------------------
# Data pipeline
# -----------------------------
def make_dataset(prices, window=30):
    returns = np.log(prices[1:] / prices[:-1])
    returns = np.concatenate([[0.0], returns])

    sma = np.convolve(prices, np.ones(10)/10, mode="same")
    ema10 = ema(prices, 10)
    rsi14 = rsi(prices, 14)
    vol10 = rolling_std(returns, 10)

    features = np.stack([returns, sma, ema10, rsi14, vol10], axis=1)

    mean = features.mean(axis=0, keepdims=True)
    std = features.std(axis=0, keepdims=True) + 1e-8
    features = (features - mean) / std

    X = []
    y = []

    for i in range(len(features) - window - 1):
        window_feats = features[i:i+window].reshape(-1)
        X.append(window_feats)

        next_ret = returns[i + window + 1]
        y.append(1.0 if next_ret > 0 else 0.0)

    X = mx.array(np.array(X), dtype=mx.float32)
    y = mx.array(np.array(y).reshape(-1, 1), dtype=mx.float32)
    return X, y

def split_data(X, y, train_ratio=0.7, val_ratio=0.15):
    n = X.shape[0]
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    X_train, y_train = X[:n_train], y[:n_train]
    X_val, y_val = X[n_train:n_train+n_val], y[n_train:n_train+n_val]
    X_test, y_test = X[n_train+n_val:], y[n_train+n_val:]
    return X_train, y_train, X_val, y_val, X_test, y_test

# -----------------------------
# Saving artifacts
# -----------------------------
def save_artifacts(model, config, metrics, run_tag):
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    models_dir = os.path.join(base_dir, "models")
    logs_dir = os.path.join(base_dir, "logs")
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = f"{run_tag}_{ts}"

    params = model.parameters()
    state = {f"p{i}": p for i, p in enumerate(params)}
    model_path = os.path.join(models_dir, f"{base_name}.npz")
    np.savez(model_path, **{k: np.array(v) for k, v in state.items()})

    log_path = os.path.join(logs_dir, f"{base_name}.json")
    payload = {
        "timestamp": ts,
        "model_path": model_path,
        "config": config,
        "metrics": metrics,
    }
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"Saved model: {model_path}")
    print(f"Saved log: {log_path}")

# -----------------------------
# Loading artifacts
# -----------------------------
def load_model(model, npz_path):
    data = np.load(npz_path)
    params = model.parameters()
    for i, p in enumerate(params):
        key = f"p{i}"
        if key in data:
            p[:] = mx.array(data[key])
    return model

# -----------------------------
# Predict demo
# -----------------------------
def predict_demo(model, sample_input):
    x = mx.array(sample_input, dtype=mx.float32).reshape(1, -1)
    y = model(x)
    mx.eval(y)
    print(f"Prediction: {float(y.squeeze())}")

# -----------------------------
# Simulation generators
# -----------------------------
# News shocks: rare large jumps

def simulate_news_shocks(n=5000):
    prices = np.zeros(n)
    prices[0] = 100.0
    drift = 0.0001
    vol = 0.01
    shock_prob = 0.005
    shock_scale = 0.08
    for i in range(1, n):
        shock = 0.0
        if np.random.rand() < shock_prob:
            shock = np.random.choice([-1, 1]) * np.random.exponential(shock_scale)
        ret = drift + vol * np.random.randn() + shock
        prices[i] = max(0.1, prices[i-1] * np.exp(ret))
    return prices

# -----------------------------
# Training
# -----------------------------
FEATURE_SOURCE = "prices"

_NEWS_ROLE_MAP = {
    "shock": ["UVXY", "VIXY", "SOXL", "SOXS", "MSTR", "SMCI", "COIN", "TSLA"],
    "bond": ["TLT", "IEF", "SHY", "TIP", "LQD", "HYG"],
    "dividend": ["SCHD", "VIG", "DGRO", "JNJ", "PG", "KO", "PEP"],
}
_NEWS_RUNTIME_MODES = [
    "shadow_equities",
    "shadow_aggressive_equities",
    "shadow_intraday_aggressive_equities",
    "shadow_swing_aggressive_equities",
]
_NEWS_RUNTIME_SYMBOLS = sorted(
    {
        "SPY",
        "QQQ",
        "DIA",
        "IWM",
        "SMH",
        "XLK",
        "XLF",
        "GLD",
        "TLT",
        "AAPL",
        "MSFT",
        "NVDA",
        *(_NEWS_ROLE_MAP["shock"]),
        *(_NEWS_ROLE_MAP["bond"]),
    }
)


def _clip01(value):
    return float(np.clip(value, 0.0, 1.0))


def _quote_quality(obs):
    return _clip01(
        (0.58 * observation_feature(obs, "data_quality_quote_agreement_norm", 1.0))
        + (0.22 * (1.0 - observation_feature(obs, "data_quality_quote_deviation_norm", 0.0)))
        + (0.20 * (1.0 - observation_feature(obs, "data_quality_market_data_latency_norm", 0.0)))
    )


def _shock_regime_signal(obs):
    event_signal = max(
        observation_feature(obs, "news_shock_rate"),
        observation_feature(obs, "news_recent_impact"),
        observation_feature(obs, "calendar_macro_event_norm"),
        observation_feature(obs, "calendar_macro_abs_surprise_norm"),
        observation_feature(obs, "calendar_macro_revision_norm"),
        observation_feature(obs, "calendar_fomc_event_norm"),
        observation_feature(obs, "calendar_cpi_event_norm"),
        observation_feature(obs, "calendar_labor_event_norm"),
        observation_feature(obs, "calendar_treasury_auction_norm"),
    )
    topical_signal = max(
        observation_feature(obs, "news_topic_earnings_norm"),
        observation_feature(obs, "news_topic_guidance_norm"),
        observation_feature(obs, "news_topic_mna_norm"),
        observation_feature(obs, "news_topic_regulatory_norm"),
    )
    freshness_signal = max(
        observation_feature(obs, "news_items_30m"),
        observation_feature(obs, "news_items_2h") * 0.9,
        observation_feature(obs, "news_items_24h") * 0.55,
    )
    micro_signal = max(
        observation_feature(obs, "market_micro_options_flow_norm"),
        observation_feature(obs, "market_micro_short_pressure_norm"),
        observation_feature(obs, "market_micro_credit_flow_norm"),
        observation_feature(obs, "market_micro_block_trade_norm"),
        abs(observation_feature(obs, "market_micro_order_flow_imbalance_norm", 0.5) - 0.5) * 2.0,
    )
    return _clip01(max(event_signal, topical_signal, freshness_signal, micro_signal))


def _directional_hint(obs):
    sentiment = observation_feature(obs, "news_sentiment")
    polarity = observation_feature(obs, "news_positive_share") - observation_feature(obs, "news_negative_share")
    specialist = (
        0.55 * observation_feature(obs, "options_specialist_vote")
        + 0.35 * observation_feature(obs, "futures_specialist_vote")
    )
    order_flow = (observation_feature(obs, "market_micro_order_flow_imbalance_norm", 0.5) - 0.5) * 2.0
    behavior = observation_feature(obs, "behavior_prior")
    short_drag = observation_feature(obs, "market_micro_short_pressure_norm")
    credit_drag = observation_feature(obs, "market_micro_credit_flow_norm")
    risk_off_drag = max(observation_feature(obs, "breadth_risk_off_norm") - 0.5, 0.0) * 2.0
    return float(
        np.clip(
            (0.34 * sentiment)
            + (0.24 * polarity)
            + (0.14 * specialist)
            + (0.12 * order_flow)
            + (0.10 * behavior)
            - (0.05 * short_drag)
            - (0.05 * credit_drag)
            - (0.02 * risk_off_drag),
            -1.0,
            1.0,
        )
    )


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
    roles = symbol_role_features(str(obs.get("symbol") or ""), _NEWS_ROLE_MAP)
    return np.asarray(
        [
            observation_feature(obs, "pct_from_close"),
            observation_feature(obs, "mom_5m"),
            observation_feature(obs, "vol_30m"),
            observation_feature(obs, "range_pos"),
            observation_feature(obs, "spread_bps"),
            observation_feature(obs, "market_data_latency_ms"),
            observation_feature(obs, "news_available"),
            observation_feature(obs, "news_items_30m"),
            observation_feature(obs, "news_items_2h"),
            observation_feature(obs, "news_items_24h"),
            observation_feature(obs, "news_sentiment"),
            observation_feature(obs, "news_negative_share"),
            observation_feature(obs, "news_positive_share"),
            observation_feature(obs, "news_shock_rate"),
            observation_feature(obs, "news_recent_impact"),
            observation_feature(obs, "news_source_quality_norm"),
            observation_feature(obs, "news_entity_relevance_norm"),
            observation_feature(obs, "news_topic_earnings_norm"),
            observation_feature(obs, "news_topic_guidance_norm"),
            observation_feature(obs, "news_topic_mna_norm"),
            observation_feature(obs, "news_topic_regulatory_norm"),
            observation_feature(obs, "news_novelty_norm"),
            observation_feature(obs, "news_duplicate_cluster_norm"),
            observation_feature(obs, "news_premarket_norm"),
            observation_feature(obs, "news_intraday_norm"),
            observation_feature(obs, "news_after_hours_norm"),
            observation_feature(obs, "calendar_event_proximity_norm"),
            observation_feature(obs, "calendar_high_impact_24h_norm"),
            observation_feature(obs, "calendar_macro_event_norm"),
            observation_feature(obs, "calendar_macro_surprise_norm"),
            observation_feature(obs, "calendar_macro_abs_surprise_norm"),
            observation_feature(obs, "calendar_macro_revision_norm"),
            observation_feature(obs, "calendar_fomc_event_norm"),
            observation_feature(obs, "calendar_cpi_event_norm"),
            observation_feature(obs, "calendar_labor_event_norm"),
            observation_feature(obs, "calendar_treasury_auction_norm"),
            observation_feature(obs, "ctx_VIX_X_pct_from_close"),
            observation_feature(obs, "ctx_UUP_pct_from_close"),
            observation_feature(obs, "breadth_advance_decline_norm"),
            observation_feature(obs, "breadth_risk_off_norm"),
            observation_feature(obs, "options_iv_atm_norm"),
            observation_feature(obs, "options_iv_skew_norm"),
            observation_feature(obs, "options_vol_expectation_norm"),
            observation_feature(obs, "options_unusual_flow_norm"),
            observation_feature(obs, "data_quality_quote_agreement_norm"),
            observation_feature(obs, "data_quality_quote_deviation_norm"),
            observation_feature(obs, "data_quality_stale_streak_norm"),
            observation_feature(obs, "data_quality_market_data_latency_norm"),
            observation_feature(obs, "market_micro_opening_auction_norm"),
            observation_feature(obs, "market_micro_closing_auction_norm"),
            observation_feature(obs, "market_micro_relative_volume_norm"),
            observation_feature(obs, "market_micro_order_flow_imbalance_norm"),
            observation_feature(obs, "market_micro_options_flow_norm"),
            observation_feature(obs, "market_micro_short_pressure_norm"),
            observation_feature(obs, "market_micro_credit_flow_norm"),
            observation_feature(obs, "market_micro_block_trade_norm"),
            observation_feature(obs, "options_specialist_vote"),
            observation_feature(obs, "futures_specialist_vote"),
            observation_feature(obs, "behavior_prior"),
            observation_feature(obs, "active_sub_bots"),
            observation_feature(obs, "active_futures_sub_bots"),
            price_change(sequence, idx, 3),
            feature_std(sequence, idx, "pct_from_close", 6),
            feature_ema(sequence, idx, "news_sentiment", 4),
            feature_ema(sequence, idx, "behavior_prior", 4),
            roles["role_shock"],
            roles["role_bond"],
            roles["role_dividend"],
        ],
        dtype=np.float32,
    )


def _runtime_sample_filter(sequence, idx, horizon):
    obs = sequence[idx]
    quote_agreement = observation_feature(obs, "data_quality_quote_agreement_norm", 1.0)
    quote_deviation = observation_feature(obs, "data_quality_quote_deviation_norm", 0.0)
    stale_streak = observation_feature(obs, "data_quality_stale_streak_norm", 0.0)
    latency = observation_feature(obs, "data_quality_market_data_latency_norm", 0.0)
    quote_quality = _quote_quality(obs)
    event_signal = _shock_regime_signal(obs)
    freshness_signal = max(
        observation_feature(obs, "news_items_30m"),
        observation_feature(obs, "news_items_2h"),
        observation_feature(obs, "news_items_24h") * 0.55,
    )
    directional_signal = abs(_directional_hint(obs))
    source_signal = max(
        observation_feature(obs, "news_source_quality_norm"),
        observation_feature(obs, "news_entity_relevance_norm"),
        observation_feature(obs, "news_novelty_norm"),
    )
    duplicate_cluster = observation_feature(obs, "news_duplicate_cluster_norm", 0.0)
    return (
        quote_agreement >= 0.80
        and quote_deviation <= 0.24
        and stale_streak <= 0.55
        and latency <= 0.80
        and quote_quality >= 0.78
        and event_signal >= 0.22
        and freshness_signal >= 0.08
        and source_signal >= 0.20
        and (directional_signal >= 0.10 or event_signal >= 0.55)
        and (duplicate_cluster <= 0.82 or event_signal >= 0.52 or observation_feature(obs, "news_novelty_norm") >= 0.55)
    )


def _runtime_confidence(sequence, idx, horizon):
    obs = sequence[idx]
    event_signal = _clip01(max(_shock_regime_signal(obs), abs(observation_feature(obs, "news_sentiment"))))
    topical_signal = _clip01(
        max(
            observation_feature(obs, "news_topic_earnings_norm"),
            observation_feature(obs, "news_topic_guidance_norm"),
            observation_feature(obs, "news_topic_mna_norm"),
            observation_feature(obs, "news_topic_regulatory_norm"),
        )
    )
    directional_signal = _clip01(abs(_directional_hint(obs)))
    source_signal = _clip01(
        max(
            observation_feature(obs, "news_source_quality_norm"),
            observation_feature(obs, "news_entity_relevance_norm"),
            observation_feature(obs, "news_novelty_norm"),
        )
    )
    freshness_signal = _clip01(
        max(
            observation_feature(obs, "news_items_30m"),
            observation_feature(obs, "news_items_2h"),
            observation_feature(obs, "news_items_24h"),
        )
    )
    quote_signal = _quote_quality(obs)
    return (0.28 * event_signal) + (0.16 * topical_signal) + (0.22 * directional_signal) + (0.16 * source_signal) + (0.08 * freshness_signal) + (0.10 * quote_signal)


def _runtime_shock_label(sequence, idx, horizon):
    obs = sequence[idx]
    shock_signal = _shock_regime_signal(obs)
    directional_hint = _directional_hint(obs)
    directional_signal = abs(directional_hint)
    if shock_signal < 0.22 or directional_signal < 0.10:
        return None

    fwd_ret = future_return(sequence, idx, horizon)
    realized = future_realized_vol(sequence, idx, horizon)
    drawdown = abs(future_max_drawdown(sequence, idx, horizon))
    quote_quality = _quote_quality(obs)
    source_signal = max(
        observation_feature(obs, "news_source_quality_norm"),
        observation_feature(obs, "news_entity_relevance_norm"),
        observation_feature(obs, "news_novelty_norm"),
    )
    expected_up = directional_hint >= 0.0
    signed_ret = fwd_ret if expected_up else -fwd_ret
    vol_gate = abs(observation_feature(obs, "vol_30m", 0.0)) * 1.05
    move_threshold = max(0.00065, 0.00145 - (0.00060 * shock_signal), vol_gate)
    if abs(fwd_ret) < move_threshold and realized < 0.024 and drawdown < 0.014:
        return None

    success_score = (
        signed_ret
        + (0.00100 * shock_signal)
        + (0.00040 * directional_signal)
        + (0.00020 * quote_quality)
        + (0.00020 * source_signal)
        - (0.18 * realized)
        - (0.20 * drawdown)
        - (0.00015 * observation_feature(obs, "breadth_risk_off_norm"))
    )
    failure_score = (
        (-signed_ret)
        + (0.00085 * shock_signal)
        + (0.00025 * directional_signal)
        + (0.16 * realized)
        + (0.18 * drawdown)
    )
    if success_score >= 0.00055:
        return 1.0 if expected_up else 0.0
    if failure_score >= 0.00080:
        return 0.0 if expected_up else 1.0

    return None


def _train_synthetic():
    return train_price_indicator_bot(
        run_tag="brain_refinery_v12_news_shocks",
        feature_names=["returns", "sma10", "ema10", "rsi14", "vol10"],
        feature_builder=build_features,
        price_simulator=simulate_news_shocks,
    )


def train_brain():
    return train_runtime_indicator_bot(
        run_tag="brain_refinery_v12_news_shocks",
        feature_names=[
            "pct_from_close",
            "mom_5m",
            "vol_30m",
            "range_pos",
            "spread_bps",
            "market_data_latency_ms",
            "news_available",
            "news_items_30m",
            "news_items_2h",
            "news_items_24h",
            "news_sentiment",
            "news_negative_share",
            "news_positive_share",
            "news_shock_rate",
            "news_recent_impact",
            "news_source_quality_norm",
            "news_entity_relevance_norm",
            "news_topic_earnings_norm",
            "news_topic_guidance_norm",
            "news_topic_mna_norm",
            "news_topic_regulatory_norm",
            "news_novelty_norm",
            "news_duplicate_cluster_norm",
            "news_premarket_norm",
            "news_intraday_norm",
            "news_after_hours_norm",
            "calendar_event_proximity_norm",
            "calendar_high_impact_24h_norm",
            "calendar_macro_event_norm",
            "calendar_macro_surprise_norm",
            "calendar_macro_abs_surprise_norm",
            "calendar_macro_revision_norm",
            "calendar_fomc_event_norm",
            "calendar_cpi_event_norm",
            "calendar_labor_event_norm",
            "calendar_treasury_auction_norm",
            "ctx_VIX_X_pct_from_close",
            "ctx_UUP_pct_from_close",
            "breadth_advance_decline_norm",
            "breadth_risk_off_norm",
            "options_iv_atm_norm",
            "options_iv_skew_norm",
            "options_vol_expectation_norm",
            "options_unusual_flow_norm",
            "data_quality_quote_agreement_norm",
            "data_quality_quote_deviation_norm",
            "data_quality_stale_streak_norm",
            "data_quality_market_data_latency_norm",
            "market_micro_opening_auction_norm",
            "market_micro_closing_auction_norm",
            "market_micro_relative_volume_norm",
            "market_micro_order_flow_imbalance_norm",
            "market_micro_options_flow_norm",
            "market_micro_short_pressure_norm",
            "market_micro_credit_flow_norm",
            "market_micro_block_trade_norm",
            "options_specialist_vote",
            "futures_specialist_vote",
            "behavior_prior",
            "active_sub_bots",
            "active_futures_sub_bots",
            "ret_3",
            "pct_from_close_std_6",
            "news_sentiment_ema_4",
            "behavior_prior_ema_4",
            "role_shock",
            "role_bond",
            "role_dividend",
        ],
        runtime_feature_builder=_runtime_feature_vector,
        runtime_label_builder=_runtime_shock_label,
        mode_allowlist=_NEWS_RUNTIME_MODES,
        symbol_allowlist=_NEWS_RUNTIME_SYMBOLS,
        sample_filter=_runtime_sample_filter,
        confidence_builder=_runtime_confidence,
        min_confidence=0.38,
        lookback_days=30,
        window=18,
        horizon=6,
        min_samples=192,
        min_sequences=6,
        min_positive_samples=32,
        min_negative_samples=32,
        acted_prob_threshold=0.66,
        fallback_trainer=_train_synthetic,
        allow_fallback_on_insufficient_data=False,
        max_best_val_loss=0.6929,
        max_final_val_loss=0.7000,
        min_long_precision=0.05,
        min_short_precision=0.05,
        require_both_sides_precision=True,
        min_acted_accuracy=0.53,
        min_long_acted_count=4,
        min_short_acted_count=4,
        min_accuracy_lift_over_majority=0.01,
        min_precision_balance_score=0.25,
    )

if __name__ == "__main__":
    train_brain()
