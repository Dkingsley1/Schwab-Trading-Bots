import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
import json
from datetime import datetime
import os

from indicator_bot_common import train_price_indicator_bot, train_runtime_indicator_bot
from runtime_training_common import (
    direction_label_builder,
    feature_ema,
    feature_std,
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
            observation_feature(obs, "news_sentiment"),
            observation_feature(obs, "news_negative_share"),
            observation_feature(obs, "news_positive_share"),
            observation_feature(obs, "news_shock_rate"),
            observation_feature(obs, "news_recent_impact"),
            observation_feature(obs, "news_source_quality_norm"),
            observation_feature(obs, "news_entity_relevance_norm"),
            observation_feature(obs, "news_topic_earnings_norm"),
            observation_feature(obs, "news_topic_guidance_norm"),
            observation_feature(obs, "news_topic_regulatory_norm"),
            observation_feature(obs, "news_novelty_norm"),
            observation_feature(obs, "news_premarket_norm"),
            observation_feature(obs, "news_after_hours_norm"),
            observation_feature(obs, "calendar_event_proximity_norm"),
            observation_feature(obs, "calendar_high_impact_24h_norm"),
            observation_feature(obs, "calendar_macro_surprise_norm"),
            observation_feature(obs, "calendar_macro_abs_surprise_norm"),
            observation_feature(obs, "ctx_VIX_X_pct_from_close"),
            observation_feature(obs, "ctx_UUP_pct_from_close"),
            observation_feature(obs, "breadth_advance_decline_norm"),
            observation_feature(obs, "breadth_risk_off_norm"),
            observation_feature(obs, "options_iv_atm_norm"),
            observation_feature(obs, "options_iv_skew_norm"),
            observation_feature(obs, "options_vol_expectation_norm"),
            observation_feature(obs, "data_quality_quote_agreement_norm"),
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
            "news_sentiment",
            "news_negative_share",
            "news_positive_share",
            "news_shock_rate",
            "news_recent_impact",
            "news_source_quality_norm",
            "news_entity_relevance_norm",
            "news_topic_earnings_norm",
            "news_topic_guidance_norm",
            "news_topic_regulatory_norm",
            "news_novelty_norm",
            "news_premarket_norm",
            "news_after_hours_norm",
            "calendar_event_proximity_norm",
            "calendar_high_impact_24h_norm",
            "calendar_macro_surprise_norm",
            "calendar_macro_abs_surprise_norm",
            "ctx_VIX_X_pct_from_close",
            "ctx_UUP_pct_from_close",
            "breadth_advance_decline_norm",
            "breadth_risk_off_norm",
            "options_iv_atm_norm",
            "options_iv_skew_norm",
            "options_vol_expectation_norm",
            "data_quality_quote_agreement_norm",
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
        runtime_label_builder=direction_label_builder(min_return=0.001),
        lookback_days=21,
        window=18,
        horizon=6,
        min_samples=192,
        min_sequences=3,
        fallback_trainer=_train_synthetic,
    )

if __name__ == "__main__":
    train_brain()
