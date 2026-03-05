import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
import json
from datetime import datetime
import os

class TradingBrain(nn.Module):
    """
    Takes the last 5 price points and predicts the next price.
    """
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(5, 32)
        self.layer2 = nn.Linear(32, 16)
        self.output = nn.Linear(16, 1)

    def __call__(self, x):
        x = nn.relu(self.layer1(x))
        x = nn.relu(self.layer2(x))
        return self.output(x)

def loss_fn(model, x, y):
    return nn.losses.mse_loss(model(x), y)

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

def train_brain():
    np.random.seed(42)

    brain = TradingBrain()
    mx.eval(brain.parameters())

    optimizer = optim.Adam(learning_rate=0.01)

    t = np.linspace(0, 50, 500)
    prices = np.sin(t) + np.random.normal(0, 0.1, 500)
    prices = mx.array(prices, dtype=mx.float32)

    print("Training on simulated price action...")

    num_windows = len(prices) - 5
    loss_and_grad_fn = nn.value_and_grad(brain, loss_fn)

    epochs = 100

    for epoch in range(epochs):
        total_loss = 0.0
        for i in range(num_windows):
            x = prices[i:i+5].reshape(1, -1)
            y = prices[i+5:i+6].reshape(1, -1)

            loss, grads = loss_and_grad_fn(brain, x, y)
            optimizer.update(brain, grads)
            mx.eval(brain.parameters(), optimizer.state)

            total_loss += float(loss)

        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Error Level = {total_loss / num_windows:.6f}")

    print("Model training complete.")

    config = {
        "window": 5,
        "learning_rate": 0.01,
        "epochs": epochs,
        "num_points": 500,
    }
    metrics = {
        "final_loss": float(total_loss / num_windows),
    }
    save_artifacts(brain, config, metrics, run_tag="brain_refinery_v1")

    return brain

if __name__ == "__main__":
    trained_model = train_brain()
