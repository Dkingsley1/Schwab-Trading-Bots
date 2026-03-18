import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
import json
from datetime import datetime
import os

from indicator_bot_common import train_price_indicator_bot

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
# Choppy: high noise, frequent reversals around a flat mean

def simulate_choppy(n=5000):
    prices = np.zeros(n)
    prices[0] = 100.0
    mu = 100.0
    theta = 0.2
    vol = 1.0
    for i in range(1, n):
        noise = vol * np.random.randn()
        mean_pull = theta * (mu - prices[i-1])
        prices[i] = max(0.1, prices[i-1] + mean_pull + noise)
    return prices

# -----------------------------
# Training
# -----------------------------
FEATURE_SOURCE = "prices"


def build_features(prices):
    returns = np.log(prices[1:] / prices[:-1])
    returns = np.concatenate([[0.0], returns])
    sma = np.convolve(prices, np.ones(10) / 10, mode="same")
    ema10 = ema(prices, 10)
    rsi14 = rsi(prices, 14)
    vol10 = rolling_std(returns, 10)
    return np.stack([returns, sma, ema10, rsi14, vol10], axis=1)


def train_brain():
    return train_price_indicator_bot(
        run_tag="brain_refinery_v13_choppy",
        feature_names=["returns", "sma10", "ema10", "rsi14", "vol10"],
        feature_builder=build_features,
        price_simulator=simulate_choppy,
    )

if __name__ == "__main__":
    train_brain()
