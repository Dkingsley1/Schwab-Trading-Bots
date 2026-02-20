import json
import os
from datetime import datetime
from typing import Callable, Dict, List

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np


ArrayMap = Dict[str, np.ndarray]


def ema(x: np.ndarray, span: int) -> np.ndarray:
    alpha = 2.0 / (span + 1.0)
    out = np.zeros_like(x)
    out[0] = x[0]
    for i in range(1, len(x)):
        out[i] = alpha * x[i] + (1.0 - alpha) * out[i - 1]
    return out


def rolling_mean(x: np.ndarray, window: int) -> np.ndarray:
    out = np.zeros_like(x)
    for i in range(len(x)):
        start = max(0, i - window + 1)
        out[i] = np.mean(x[start : i + 1])
    return out


def rolling_std(x: np.ndarray, window: int) -> np.ndarray:
    out = np.zeros_like(x)
    for i in range(len(x)):
        start = max(0, i - window + 1)
        out[i] = np.std(x[start : i + 1])
    return out


def macd_line(prices: np.ndarray, fast: int = 12, slow: int = 26) -> np.ndarray:
    return ema(prices, fast) - ema(prices, slow)


def true_range(high: np.ndarray, low: np.ndarray, close: np.ndarray) -> np.ndarray:
    prev_close = np.concatenate([[close[0]], close[:-1]])
    tr1 = high - low
    tr2 = np.abs(high - prev_close)
    tr3 = np.abs(low - prev_close)
    return np.maximum(tr1, np.maximum(tr2, tr3))


def atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
    return ema(true_range(high, low, close), period)


def adx(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
    up_move = np.diff(high, prepend=high[0])
    down_move = -np.diff(low, prepend=low[0])
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    tr = true_range(high, low, close) + 1e-8

    plus_di = 100.0 * ema(plus_dm, period) / (ema(tr, period) + 1e-8)
    minus_di = 100.0 * ema(minus_dm, period) / (ema(tr, period) + 1e-8)
    dx = 100.0 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-8)
    return ema(dx, period)


def vwap(close: np.ndarray, volume: np.ndarray, session: int = 60) -> np.ndarray:
    out = np.zeros_like(close)
    for i in range(len(close)):
        start = max(0, i - session + 1)
        w = volume[start : i + 1]
        p = close[start : i + 1]
        out[i] = np.sum(p * w) / (np.sum(w) + 1e-8)
    return out


def bollinger(close: np.ndarray, window: int = 20, k: float = 2.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mid = rolling_mean(close, window)
    sig = rolling_std(close, window)
    up = mid + k * sig
    dn = mid - k * sig
    return dn, mid, up


def tsi(prices: np.ndarray, fast: int = 13, slow: int = 25) -> np.ndarray:
    mom = np.diff(prices, prepend=prices[0])
    abs_mom = np.abs(mom)
    mom_smoothed = ema(ema(mom, fast), slow)
    abs_smoothed = ema(ema(abs_mom, fast), slow) + 1e-8
    return 100.0 * (mom_smoothed / abs_smoothed)


def stochastic_momentum_index(close: np.ndarray, high: np.ndarray, low: np.ndarray, period: int = 14, smooth: int = 3) -> np.ndarray:
    hh = np.zeros_like(close)
    ll = np.zeros_like(close)
    for i in range(len(close)):
        start = max(0, i - period + 1)
        hh[i] = np.max(high[start : i + 1])
        ll[i] = np.min(low[start : i + 1])
    mid = 0.5 * (hh + ll)
    half_range = 0.5 * (hh - ll) + 1e-8
    rel = close - mid
    return 100.0 * (ema(ema(rel, smooth), smooth) / (ema(ema(half_range, smooth), smooth) + 1e-8))


def zscore(x: np.ndarray) -> np.ndarray:
    return (x - np.mean(x)) / (np.std(x) + 1e-8)


def simulate_market_panel(n: int = 6000) -> ArrayMap:
    close = np.zeros(n, dtype=np.float64)
    high = np.zeros(n, dtype=np.float64)
    low = np.zeros(n, dtype=np.float64)
    volume = np.zeros(n, dtype=np.float64)

    bench = np.zeros(n, dtype=np.float64)
    close[0] = 100.0
    bench[0] = 100.0

    regime = 1.0
    for i in range(1, n):
        if i % 1100 == 0:
            regime *= -1.0

        common = np.random.normal(0.0, 0.006)
        idio = np.random.normal(0.0, 0.007)
        r = 0.0002 * regime + 0.6 * common + 0.4 * idio
        rb = 0.00015 * regime + common

        close[i] = max(1.0, close[i - 1] * np.exp(r))
        bench[i] = max(1.0, bench[i - 1] * np.exp(rb))

        intrarange = abs(np.random.normal(0.0024, 0.0012)) + 0.6 * abs(r)
        high[i] = close[i] * (1.0 + intrarange)
        low[i] = close[i] * max(1e-6, 1.0 - intrarange)

        volume[i] = 1_000_000.0 + 4_000_000.0 * min(abs(r) * 22.0, 1.0) + np.random.uniform(0, 250_000)

    high[0] = close[0] * 1.002
    low[0] = close[0] * 0.998
    volume[0] = 1_000_000.0

    ret = np.diff(close, prepend=close[0]) / np.maximum(np.concatenate([[close[0]], close[:-1]]), 1e-8)
    bench_ret = np.diff(bench, prepend=bench[0]) / np.maximum(np.concatenate([[bench[0]], bench[:-1]]), 1e-8)

    vix_base = 18.0 + 220.0 * rolling_std(ret, 20)
    vix = np.maximum(vix_base + np.random.normal(0.0, 0.6, n), 9.0)
    vix9d = np.maximum(vix + np.random.normal(0.0, 0.8, n), 8.5)
    vix3m = np.maximum(vix + np.random.normal(0.0, 0.7, n), 9.0)

    breadth_bias = np.tanh(6.0 * ret)
    adv = np.maximum(1200 + 700 * breadth_bias + np.random.normal(0.0, 120.0, n), 50.0)
    dec = np.maximum(1200 - 700 * breadth_bias + np.random.normal(0.0, 120.0, n), 50.0)

    up_vol = np.maximum(2.0e8 + 8.0e7 * breadth_bias + np.random.normal(0.0, 2.5e7, n), 1.0e6)
    down_vol = np.maximum(2.0e8 - 8.0e7 * breadth_bias + np.random.normal(0.0, 2.5e7, n), 1.0e6)

    open_price = np.concatenate([[close[0]], close[:-1] * (1.0 + np.random.normal(0.0, 0.0025, n - 1))])
    gap = (open_price - np.concatenate([[open_price[0]], close[:-1]])) / np.maximum(np.concatenate([[open_price[0]], close[:-1]]), 1e-8)

    return {
        "close": close,
        "high": high,
        "low": low,
        "volume": volume,
        "bench_close": bench,
        "ret": ret,
        "bench_ret": bench_ret,
        "vix": vix,
        "vix9d": vix9d,
        "vix3m": vix3m,
        "adv": adv,
        "dec": dec,
        "up_vol": up_vol,
        "down_vol": down_vol,
        "open": open_price,
        "gap": gap,
    }


class TradingBrain(nn.Module):
    def __init__(self, input_dim: int):
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
    probs = mx.sigmoid(model(x))
    return nn.losses.binary_cross_entropy(probs, y)


def split_data(X, y, train_ratio=0.7, val_ratio=0.15):
    n = X.shape[0]
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    X_train, y_train = X[:n_train], y[:n_train]
    X_val, y_val = X[n_train : n_train + n_val], y[n_train : n_train + n_val]
    X_test, y_test = X[n_train + n_val :], y[n_train + n_val :]
    return X_train, y_train, X_val, y_val, X_test, y_test


def make_windowed_dataset(features: np.ndarray, close: np.ndarray, window: int, horizon: int):
    feat_mean = features.mean(axis=0, keepdims=True)
    feat_std = features.std(axis=0, keepdims=True) + 1e-8
    feats = (features - feat_mean) / feat_std

    X = []
    y = []
    for i in range(len(feats) - window - horizon):
        X.append(feats[i : i + window].reshape(-1))
        fwd = (close[i + window + horizon] - close[i + window]) / max(close[i + window], 1e-8)
        y.append(1.0 if fwd > 0 else 0.0)

    return mx.array(np.array(X), dtype=mx.float32), mx.array(np.array(y).reshape(-1, 1), dtype=mx.float32)


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
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump({"timestamp": ts, "model_path": model_path, "config": config, "metrics": metrics}, f, indent=2)

    print(f"Saved model: {model_path}")
    print(f"Saved log: {log_path}")


def train_indicator_bot(
    *,
    run_tag: str,
    feature_names: List[str],
    feature_builder: Callable[[ArrayMap], np.ndarray],
    num_points: int = 6000,
    window: int = 30,
    horizon: int = 3,
    learning_rate: float = 0.0008,
    epochs: int = 220,
    batch_size: int = 128,
    patience: int = 18,
) -> TradingBrain:
    np.random.seed(42)

    panel = simulate_market_panel(n=num_points)
    features = feature_builder(panel)
    close = panel["close"]

    X, y = make_windowed_dataset(features, close, window=window, horizon=horizon)
    X_train, y_train, X_val, y_val, X_test, y_test = split_data(X, y)

    brain = TradingBrain(int(X.shape[1]))
    mx.eval(brain.parameters())

    optimizer = optim.Adam(learning_rate=learning_rate)
    loss_and_grad_fn = nn.value_and_grad(brain, loss_fn)

    best_val = float("inf")
    patience_left = patience

    print("Training...")
    for epoch in range(epochs):
        idx = np.random.permutation(X_train.shape[0])
        total_loss = 0.0
        batches = 0

        for start in range(0, X_train.shape[0], batch_size):
            bidx = mx.array(idx[start : start + batch_size])
            xb = mx.take(X_train, bidx, axis=0)
            yb = mx.take(y_train, bidx, axis=0)

            loss, grads = loss_and_grad_fn(brain, xb, yb)
            optimizer.update(brain, grads)
            mx.eval(brain.parameters(), optimizer.state)

            total_loss += float(loss)
            batches += 1

        val_loss = float(loss_fn(brain, X_val, y_val))
        if epoch % 10 == 0:
            print(f"Epoch {epoch} | Train {total_loss / max(batches,1):.6f} | Val {val_loss:.6f}")

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
        "horizon": horizon,
        "learning_rate": learning_rate,
        "epochs": epochs,
        "batch_size": batch_size,
        "patience": patience,
        "input_dim": int(X.shape[1]),
        "num_points": num_points,
        "features": feature_names,
    }
    metrics = {
        "best_val_loss": float(best_val),
        "final_val_loss": float(val_loss),
        "test_accuracy": float(acc),
    }
    save_artifacts(brain, config, metrics, run_tag=run_tag)
    return brain
