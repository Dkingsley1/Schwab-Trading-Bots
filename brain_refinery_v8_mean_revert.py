import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
import json
from datetime import datetime
import os

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
# Mean-reversion: pull back to midline

def simulate_mean_revert(n=5000):
    prices = np.zeros(n)
    prices[0] = 100.0
    mu = 100.0
    theta = 0.05
    vol = 0.5
    for i in range(1, n):
        dx = theta * (mu - prices[i-1]) + vol * np.random.randn()
        prices[i] = max(0.1, prices[i-1] + dx)
    return prices

# -----------------------------
# Training
# -----------------------------
def train_brain():
    np.random.seed(42)

    prices = simulate_mean_revert(n=5000)

    window = 30
    X, y = make_dataset(prices, window=window)
    X_train, y_train, X_val, y_val, X_test, y_test = split_data(X, y)

    input_dim = X.shape[1]
    brain = TradingBrain(input_dim)
    mx.eval(brain.parameters())

    optimizer = optim.Adam(learning_rate=0.001)
    loss_and_grad_fn = nn.value_and_grad(brain, loss_fn)

    epochs = 200
    batch_size = 128
    patience = 15
    best_val = float("inf")
    patience_left = patience

    print("Training...")

    for epoch in range(epochs):
        idx = np.random.permutation(X_train.shape[0])

        total_loss = 0.0
        num_batches = 0

        for start in range(0, X_train.shape[0], batch_size):
            batch_idx = mx.array(idx[start:start+batch_size])
            xb = mx.take(X_train, batch_idx, axis=0)
            yb = mx.take(y_train, batch_idx, axis=0)

            loss, grads = loss_and_grad_fn(brain, xb, yb)
            optimizer.update(brain, grads)
            mx.eval(brain.parameters(), optimizer.state)

            total_loss += float(loss)
            num_batches += 1

        val_loss = float(loss_fn(brain, X_val, y_val))
        if epoch % 10 == 0:
            print(f"Epoch {epoch} | Train {total_loss/num_batches:.6f} | Val {val_loss:.6f}")

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
        "learning_rate": 0.001,
        "epochs": epochs,
        "batch_size": batch_size,
        "patience": patience,
        "input_dim": int(input_dim),
        "num_points": int(len(prices)),
    }
    metrics = {
        "best_val_loss": float(best_val),
        "final_val_loss": float(val_loss),
        "test_accuracy": float(acc),
    }
    save_artifacts(brain, config, metrics, run_tag="brain_refinery_v8_mean_revert")

    return brain

if __name__ == "__main__":
    trained_model = train_brain()
