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
        out[i] = np.std(x[start:i + 1])
    return out


def rolling_mean(x, window):
    out = np.zeros_like(x)
    for i in range(len(x)):
        start = max(0, i - window + 1)
        out[i] = np.mean(x[start:i + 1])
    return out


def rolling_max(x, window):
    out = np.zeros_like(x)
    for i in range(len(x)):
        start = max(0, i - window + 1)
        out[i] = np.max(x[start:i + 1])
    return out


def rolling_skew(x, window):
    out = np.zeros_like(x)
    for i in range(len(x)):
        start = max(0, i - window + 1)
        w = x[start:i + 1]
        m = np.mean(w)
        s = np.std(w) + 1e-8
        out[i] = np.mean(((w - m) / s) ** 3)
    return out


# -----------------------------
# Model
# -----------------------------
class TradingBrain(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.layer1 = nn.Linear(input_dim, 96)
        self.layer2 = nn.Linear(96, 48)
        self.layer3 = nn.Linear(48, 24)
        self.out = nn.Linear(24, 1)

    def __call__(self, x):
        x = nn.relu(self.layer1(x))
        x = nn.relu(self.layer2(x))
        x = nn.relu(self.layer3(x))
        return self.out(x)


def weighted_bce_loss(model, x_arr, y_arr, pos_weight, reg_lambda):
    logits = model(x_arr)
    probs = mx.sigmoid(logits)
    eps = 1e-8
    bce = -(pos_weight * y_arr * mx.log(probs + eps) + (1.0 - y_arr) * mx.log(1.0 - probs + eps))
    bce = mx.mean(bce)

    # Keep signature compatible; reg_lambda is currently unused.
    return bce


# -----------------------------
# Data pipeline
# -----------------------------
def make_dataset(prices, window=40, horizon=20, drawdown_threshold=-0.05):
    returns = np.log(prices[1:] / prices[:-1])
    returns = np.concatenate([[0.0], returns])

    sma10 = np.convolve(prices, np.ones(10) / 10, mode="same")
    ema10 = ema(prices, 10)
    rsi14 = rsi(prices, 14)
    vol10 = rolling_std(returns, 10)

    recent_high = np.maximum.accumulate(prices)
    drawdown = prices / (recent_high + 1e-8) - 1.0
    downside_vol10 = rolling_std(np.minimum(returns, 0.0), 10)
    skew5 = rolling_skew(returns, 5)

    crash_flag_raw = (returns < -0.03).astype(np.float32)
    crash_flag = rolling_max(crash_flag_raw, 6)
    rebound_proxy = rolling_mean(np.maximum(returns, 0.0), 5)

    features = np.stack(
        [
            returns,
            sma10,
            ema10,
            rsi14,
            vol10,
            drawdown,
            downside_vol10,
            skew5,
            crash_flag,
            rebound_proxy,
        ],
        axis=1,
    )

    mean = features.mean(axis=0, keepdims=True)
    std = features.std(axis=0, keepdims=True) + 1e-8
    features = (features - mean) / std

    x_rows = []
    y_rows = []

    max_i = len(features) - window - horizon
    for i in range(max_i):
        x_rows.append(features[i:i + window].reshape(-1))

        anchor_idx = i + window - 1
        current_price = prices[anchor_idx]
        future_prices = prices[anchor_idx + 1:anchor_idx + 1 + horizon]
        future_min_dd = np.min(future_prices) / (current_price + 1e-8) - 1.0

        crash_label = 1.0 if future_min_dd <= drawdown_threshold else 0.0
        y_rows.append(crash_label)

    x_arr = mx.array(np.array(x_rows), dtype=mx.float32)
    y_arr = mx.array(np.array(y_rows).reshape(-1, 1), dtype=mx.float32)
    return x_arr, y_arr


def split_data(x_arr, y_arr, train_ratio=0.7, val_ratio=0.15):
    n = x_arr.shape[0]
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    x_train, y_train = x_arr[:n_train], y_arr[:n_train]
    x_val, y_val = x_arr[n_train:n_train + n_val], y_arr[n_train:n_train + n_val]
    x_test, y_test = x_arr[n_train + n_val:], y_arr[n_train + n_val:]
    return x_train, y_train, x_val, y_val, x_test, y_test


# -----------------------------
# Metric helpers
# -----------------------------
def metrics_for_threshold(probs_np, y_np, threshold):
    preds = (probs_np >= threshold).astype(np.float32)
    y_true = y_np.astype(np.float32)

    tp = float(np.sum((preds == 1.0) & (y_true == 1.0)))
    fp = float(np.sum((preds == 1.0) & (y_true == 0.0)))
    fn = float(np.sum((preds == 0.0) & (y_true == 1.0)))
    tn = float(np.sum((preds == 0.0) & (y_true == 0.0)))

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2.0 * precision * recall / (precision + recall + 1e-8)
    acc = (tp + tn) / (tp + tn + fp + fn + 1e-8)

    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def find_best_threshold(probs_np, y_np):
    best_t = 0.5
    best_metrics = metrics_for_threshold(probs_np, y_np, best_t)

    for t in np.linspace(0.10, 0.90, 33):
        m = metrics_for_threshold(probs_np, y_np, float(t))
        if (m["f1"] > best_metrics["f1"]) or (
            abs(m["f1"] - best_metrics["f1"]) < 1e-8 and m["recall"] > best_metrics["recall"]
        ):
            best_t = float(t)
            best_metrics = m

    return best_t, best_metrics


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
    x_arr = mx.array(sample_input, dtype=mx.float32).reshape(1, -1)
    y_arr = model(x_arr)
    mx.eval(y_arr)
    print(f"Prediction logit: {float(y_arr.squeeze())}")
    print(f"Crash probability: {float(mx.sigmoid(y_arr).squeeze())}")


# -----------------------------
# Simulation generator
# -----------------------------
def simulate_flash_crash(n=5000):
    prices = np.zeros(n)
    prices[0] = 100.0

    drift = 0.00015
    base_vol = 0.009

    crash_start = np.random.randint(1400, 2600)
    crash_len = np.random.randint(8, 20)
    total_drop = np.random.uniform(0.20, 0.40)

    leg_drop = 1.0 - (1.0 - total_drop) ** (1.0 / crash_len)

    rebound_len = np.random.randint(60, 140)
    rebound_strength = np.random.uniform(0.35, 0.85)
    target_recovery = total_drop * rebound_strength

    for i in range(1, n):
        if crash_start <= i < crash_start + crash_len:
            ret = -leg_drop + 0.01 * np.random.randn()
            prices[i] = max(0.1, prices[i - 1] * (1.0 + ret))
            continue

        if crash_start + crash_len <= i < crash_start + crash_len + rebound_len:
            progress = (i - (crash_start + crash_len) + 1) / rebound_len
            mean_rebound = (target_recovery / rebound_len) * (1.25 - progress)
            ret = mean_rebound + 0.02 * np.random.randn()
            prices[i] = max(0.1, prices[i - 1] * (1.0 + ret))
            continue

        ret = drift + base_vol * np.random.randn()
        prices[i] = max(0.1, prices[i - 1] * np.exp(ret))

    return prices


# -----------------------------
# Training
# -----------------------------
def train_brain():
    np.random.seed(42)

    prices = simulate_flash_crash(n=5000)

    window = 40
    horizon = 20
    drawdown_threshold = -0.05

    x_arr, y_arr = make_dataset(
        prices,
        window=window,
        horizon=horizon,
        drawdown_threshold=drawdown_threshold,
    )
    x_train, y_train, x_val, y_val, x_test, y_test = split_data(x_arr, y_arr)

    y_train_np = np.array(y_train).reshape(-1)
    pos_count = float(np.sum(y_train_np))
    neg_count = float(len(y_train_np) - pos_count)
    pos_weight_value = neg_count / max(pos_count, 1.0)
    pos_weight_value = float(np.clip(pos_weight_value, 1.0, 4.0))
    pos_weight = mx.array(pos_weight_value, dtype=mx.float32)
    reg_lambda = mx.array(1e-5, dtype=mx.float32)

    input_dim = x_arr.shape[1]
    brain = TradingBrain(input_dim)
    mx.eval(brain.parameters())

    optimizer = optim.Adam(learning_rate=0.0006)
    loss_and_grad_fn = nn.value_and_grad(brain, weighted_bce_loss)

    epochs = 220
    batch_size = 128
    patience = 25
    best_val_f1 = -1.0
    patience_left = patience
    best_threshold = 0.5

    print("Training...")
    print(f"Crash label positive rate (train): {pos_count / max(len(y_train_np), 1):.4f}")
    print(f"Positive class weight: {pos_weight_value:.3f}")

    for epoch in range(epochs):
        idx = np.random.permutation(x_train.shape[0])

        total_loss = 0.0
        num_batches = 0

        for start in range(0, x_train.shape[0], batch_size):
            batch_idx = mx.array(idx[start:start + batch_size])
            xb = mx.take(x_train, batch_idx, axis=0)
            yb = mx.take(y_train, batch_idx, axis=0)

            loss, grads = loss_and_grad_fn(brain, xb, yb, pos_weight, reg_lambda)
            optimizer.update(brain, grads)
            mx.eval(brain.parameters(), optimizer.state)

            total_loss += float(loss)
            num_batches += 1

        val_loss = float(weighted_bce_loss(brain, x_val, y_val, pos_weight, reg_lambda))

        val_probs_np = np.array(mx.sigmoid(brain(x_val))).reshape(-1)
        y_val_np = np.array(y_val).reshape(-1)
        current_threshold, val_metrics = find_best_threshold(val_probs_np, y_val_np)

        if epoch % 10 == 0:
            print(
                "Epoch "
                f"{epoch} | Train {total_loss / num_batches:.6f} | ValLoss {val_loss:.6f} | "
                f"ValF1 {val_metrics['f1']:.4f} | ValRec {val_metrics['recall']:.4f} | T {current_threshold:.2f}"
            )

        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            best_threshold = current_threshold
            patience_left = patience
        else:
            patience_left -= 1
            if patience_left == 0:
                print("Early stopping.")
                break
    test_probs_np = np.array(mx.sigmoid(brain(x_test))).reshape(-1)
    y_test_np = np.array(y_test).reshape(-1)
    test_metrics = metrics_for_threshold(test_probs_np, y_test_np, best_threshold)

    print(f"Test accuracy: {test_metrics['accuracy']:.4f}")
    print(f"Test precision: {test_metrics['precision']:.4f}")
    print(f"Test recall: {test_metrics['recall']:.4f}")
    print(f"Test F1: {test_metrics['f1']:.4f}")
    print(f"Decision threshold: {best_threshold:.2f}")

    config = {
        "window": window,
        "horizon": horizon,
        "drawdown_threshold": drawdown_threshold,
        "learning_rate": 0.0006,
        "epochs": epochs,
        "batch_size": batch_size,
        "patience": patience,
        "input_dim": int(input_dim),
        "num_points": int(len(prices)),
        "feature_count": 10,
        "pos_weight": pos_weight_value,
        "reg_lambda": float(reg_lambda),
        "target": "future drawdown <= threshold",
    }
    metrics = {
        "best_val_f1": float(best_val_f1),
        "final_val_loss": float(val_loss),
        "test_accuracy": float(test_metrics["accuracy"]),
        "test_precision": float(test_metrics["precision"]),
        "test_recall": float(test_metrics["recall"]),
        "test_f1": float(test_metrics["f1"]),
        "decision_threshold": float(best_threshold),
    }
    save_artifacts(brain, config, metrics, run_tag="brain_refinery_v21_flash_crash")

    return brain


if __name__ == "__main__":
    trained_model = train_brain()
