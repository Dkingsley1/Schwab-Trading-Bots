import json
import os
from datetime import datetime

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np


def ema(x, span):
    alpha = 2.0 / (span + 1.0)
    out = np.zeros_like(x)
    out[0] = x[0]
    for i in range(1, len(x)):
        out[i] = alpha * x[i] + (1.0 - alpha) * out[i - 1]
    return out


def rolling_std(x, window):
    out = np.zeros_like(x)
    for i in range(len(x)):
        start = max(0, i - window + 1)
        out[i] = np.std(x[start : i + 1])
    return out


def macd(prices, fast=12, slow=26, signal=9):
    fast_ema = ema(prices, fast)
    slow_ema = ema(prices, slow)
    macd_line = fast_ema - slow_ema
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


def tsi(prices, fast=13, slow=25):
    mom = np.diff(prices, prepend=prices[0])
    abs_mom = np.abs(mom)
    mom_smoothed = ema(ema(mom, fast), slow)
    abs_smoothed = ema(ema(abs_mom, fast), slow) + 1e-8
    return 100.0 * (mom_smoothed / abs_smoothed)


def stochastic_momentum_index(close, high, low, period=14, smooth=3):
    hh = np.zeros_like(close)
    ll = np.zeros_like(close)
    for i in range(len(close)):
        start = max(0, i - period + 1)
        hh[i] = np.max(high[start : i + 1])
        ll[i] = np.min(low[start : i + 1])

    mid = (hh + ll) * 0.5
    half_range = (hh - ll) * 0.5 + 1e-8
    rel = close - mid
    smi_num = ema(ema(rel, smooth), smooth)
    smi_den = ema(ema(half_range, smooth), smooth) + 1e-8
    return 100.0 * (smi_num / smi_den)


def on_balance_volume(close, volume):
    out = np.zeros_like(close)
    for i in range(1, len(close)):
        if close[i] > close[i - 1]:
            out[i] = out[i - 1] + volume[i]
        elif close[i] < close[i - 1]:
            out[i] = out[i - 1] - volume[i]
        else:
            out[i] = out[i - 1]
    return out


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


def make_dataset(close, high, low, volume, window=30):
    rets = np.log(close[1:] / close[:-1])
    rets = np.concatenate([[0.0], rets])

    macd_line, macd_signal, macd_hist = macd(close)
    tsi_line = tsi(close)
    smi_line = stochastic_momentum_index(close, high, low)

    vol_z = (volume - np.mean(volume)) / (np.std(volume) + 1e-8)
    obv = on_balance_volume(close, volume)
    obv_delta = np.diff(obv, prepend=obv[0])
    ret_vol = rolling_std(rets, 10)

    features = np.stack(
        [
            rets,
            macd_line,
            macd_signal,
            macd_hist,
            tsi_line,
            smi_line,
            vol_z,
            obv_delta,
            ret_vol,
        ],
        axis=1,
    )

    mean = features.mean(axis=0, keepdims=True)
    std = features.std(axis=0, keepdims=True) + 1e-8
    features = (features - mean) / std

    X = []
    y = []
    horizon = 3
    for i in range(len(features) - window - horizon):
        X.append(features[i : i + window].reshape(-1))
        fwd_ret = (close[i + window + horizon] - close[i + window]) / max(close[i + window], 1e-8)
        y.append(1.0 if fwd_ret > 0 else 0.0)

    X = mx.array(np.array(X), dtype=mx.float32)
    y = mx.array(np.array(y).reshape(-1, 1), dtype=mx.float32)
    return X, y


def split_data(X, y, train_ratio=0.7, val_ratio=0.15):
    n = X.shape[0]
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    X_train, y_train = X[:n_train], y[:n_train]
    X_val, y_val = X[n_train : n_train + n_val], y[n_train : n_train + n_val]
    X_test, y_test = X[n_train + n_val :], y[n_train + n_val :]
    return X_train, y_train, X_val, y_val, X_test, y_test


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


def simulate_ohlcv(n=6000):
    close = np.zeros(n, dtype=np.float64)
    high = np.zeros(n, dtype=np.float64)
    low = np.zeros(n, dtype=np.float64)
    volume = np.zeros(n, dtype=np.float64)

    close[0] = 100.0
    regime = 1.0
    for i in range(1, n):
        if i % 1200 == 0:
            regime *= -1.0

        drift = 0.00025 * regime
        shock = np.random.normal(0.0, 0.0095)
        r = drift + shock
        close[i] = max(1.0, close[i - 1] * np.exp(r))

        day_range = abs(np.random.normal(0.0025, 0.0015)) + abs(r) * 0.6
        high[i] = close[i] * (1.0 + day_range)
        low[i] = close[i] * max(1e-6, 1.0 - day_range)

        vol_base = 1_000_000.0
        vol_spike = 4_000_000.0 * min(abs(r) * 25.0, 1.0)
        volume[i] = vol_base + vol_spike + np.random.uniform(0.0, 250_000.0)

    high[0] = close[0] * 1.002
    low[0] = close[0] * 0.998
    volume[0] = 1_000_000.0

    return close, high, low, volume


def train_brain():
    np.random.seed(42)

    close, high, low, volume = simulate_ohlcv(n=6000)

    window = 30
    X, y = make_dataset(close, high, low, volume, window=window)
    X_train, y_train, X_val, y_val, X_test, y_test = split_data(X, y)

    input_dim = X.shape[1]
    brain = TradingBrain(input_dim)
    mx.eval(brain.parameters())

    optimizer = optim.Adam(learning_rate=0.0008)
    loss_and_grad_fn = nn.value_and_grad(brain, loss_fn)

    epochs = 220
    batch_size = 128
    patience = 18
    best_val = float("inf")
    patience_left = patience

    print("Training...")

    for epoch in range(epochs):
        idx = np.random.permutation(X_train.shape[0])

        total_loss = 0.0
        num_batches = 0
        for start in range(0, X_train.shape[0], batch_size):
            batch_idx = mx.array(idx[start : start + batch_size])
            xb = mx.take(X_train, batch_idx, axis=0)
            yb = mx.take(y_train, batch_idx, axis=0)

            loss, grads = loss_and_grad_fn(brain, xb, yb)
            optimizer.update(brain, grads)
            mx.eval(brain.parameters(), optimizer.state)

            total_loss += float(loss)
            num_batches += 1

        val_loss = float(loss_fn(brain, X_val, y_val))
        if epoch % 10 == 0:
            print(f"Epoch {epoch} | Train {total_loss / num_batches:.6f} | Val {val_loss:.6f}")

        if val_loss < best_val:
            best_val = val_loss
            patience_left = patience
        else:
            patience_left -= 1
            if patience_left == 0:
                print("Early stopping.")
                break

    preds = mx.sigmoid(brain(X_test))
    pred_labels = (preds > 0.5).astype(mx.float32)
    acc = float(mx.mean((pred_labels == y_test).astype(mx.float32)))

    print(f"Test accuracy: {acc:.4f}")

    config = {
        "window": window,
        "learning_rate": 0.0008,
        "epochs": epochs,
        "batch_size": batch_size,
        "patience": patience,
        "input_dim": int(input_dim),
        "num_points": int(len(close)),
        "features": [
            "returns",
            "macd_line",
            "macd_signal",
            "macd_hist",
            "tsi",
            "smi",
            "volume_zscore",
            "obv_delta",
            "ret_vol_10",
        ],
        "target_horizon_steps": 3,
    }
    metrics = {
        "best_val_loss": float(best_val),
        "final_val_loss": float(val_loss),
        "test_accuracy": float(acc),
    }
    save_artifacts(brain, config, metrics, run_tag="brain_refinery_v22_macd_tsi_smi")

    return brain


if __name__ == "__main__":
    trained_model = train_brain()
