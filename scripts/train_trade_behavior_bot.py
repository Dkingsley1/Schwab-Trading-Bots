import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = PROJECT_ROOT / "data" / "trade_history" / "trade_learning_dataset.json"
MODELS_DIR = PROJECT_ROOT / "models"
LOGS_DIR = PROJECT_ROOT / "logs"


LABEL_TO_ID = {"negative": 0, "neutral": 1, "positive": 2}
ID_TO_LABEL = {v: k for k, v in LABEL_TO_ID.items()}


def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - np.max(z, axis=1, keepdims=True)
    ez = np.exp(z)
    return ez / np.clip(np.sum(ez, axis=1, keepdims=True), 1e-8, None)


def _one_hot(y: np.ndarray, n_classes: int) -> np.ndarray:
    oh = np.zeros((y.shape[0], n_classes), dtype=np.float64)
    oh[np.arange(y.shape[0]), y] = 1.0
    return oh


def _weighted_ce(probs: np.ndarray, y: np.ndarray, weights: np.ndarray) -> float:
    p = np.clip(probs[np.arange(len(y)), y], 1e-8, 1.0)
    loss = -np.log(p)
    return float(np.sum(loss * weights) / np.clip(np.sum(weights), 1e-8, None))


def _confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> List[List[int]]:
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    return cm.tolist()


def _metrics(probs: np.ndarray, y: np.ndarray) -> Dict[str, object]:
    pred = np.argmax(probs, axis=1)
    acc = float(np.mean(pred == y)) if len(y) else 0.0

    out: Dict[str, object] = {"accuracy": acc}
    f1s: List[float] = []
    recalls: List[float] = []

    for cid, name in ID_TO_LABEL.items():
        tp = int(np.sum((pred == cid) & (y == cid)))
        fp = int(np.sum((pred == cid) & (y != cid)))
        fn = int(np.sum((pred != cid) & (y == cid)))
        prec = float(tp / max(tp + fp, 1))
        rec = float(tp / max(tp + fn, 1))
        f1 = float((2 * prec * rec) / max(prec + rec, 1e-8))
        out[f"{name}_precision"] = prec
        out[f"{name}_recall"] = rec
        out[f"{name}_f1"] = f1
        f1s.append(f1)
        recalls.append(rec)

    out["macro_f1"] = float(sum(f1s) / max(len(f1s), 1))
    out["balanced_accuracy"] = float(sum(recalls) / max(len(recalls), 1))
    out["confusion_matrix"] = _confusion_matrix(y, pred, len(LABEL_TO_ID))
    out["labels"] = [ID_TO_LABEL[i] for i in range(len(LABEL_TO_ID))]
    return out


def _load_dataset(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, object]]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    rows = obj.get("data", [])

    x_rows: List[List[float]] = []
    y_rows: List[int] = []
    w_rows: List[float] = []

    for r in rows:
        label = str(r.get("label") or "neutral").lower()
        if label not in LABEL_TO_ID:
            continue
        feats = r.get("features") or []
        if not isinstance(feats, list) or not feats:
            continue

        x_rows.append([float(v) for v in feats])
        y_rows.append(LABEL_TO_ID[label])
        w_rows.append(float(r.get("sample_weight", 1.0)))

    X = np.asarray(x_rows, dtype=np.float64)
    y = np.asarray(y_rows, dtype=np.int64)
    w = np.asarray(w_rows, dtype=np.float64)
    return X, y, w, obj


def _standardize_train_test(X: np.ndarray, train_idx: np.ndarray, test_idx: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x_tr = X[train_idx]
    x_te = X[test_idx]

    mu = np.mean(x_tr, axis=0, keepdims=True)
    sigma = np.std(x_tr, axis=0, keepdims=True)
    sigma = np.where(sigma < 1e-8, 1.0, sigma)

    x_tr = (x_tr - mu) / sigma
    x_te = (x_te - mu) / sigma
    return x_tr, x_te, mu.squeeze(0), sigma.squeeze(0)


def _rebalance_weights(y: np.ndarray, w: np.ndarray, cap: float = 2.0) -> np.ndarray:
    out = w.astype(np.float64).copy()
    counts = np.bincount(y, minlength=len(LABEL_TO_ID)).astype(np.float64)
    total = float(np.sum(counts))
    n_classes = float(len(LABEL_TO_ID))

    factors = np.ones_like(counts)
    for cid in range(len(counts)):
        if counts[cid] > 0:
            # Inverse-frequency factor, capped to avoid unstable overshoot.
            factors[cid] = min((total / (n_classes * counts[cid])), cap)

    for cid in range(len(counts)):
        out[y == cid] *= factors[cid]

    # Keep average weight around 1.0 for stable learning rates.
    mean_w = float(np.mean(out)) if len(out) else 1.0
    if mean_w > 1e-8:
        out /= mean_w
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Train a weighted behavior policy model on past trades.")
    parser.add_argument("--dataset", default=str(DATASET_PATH))
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--l2", type=float, default=1e-4)
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--class-balance-cap", type=float, default=2.0)
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        print(f"Missing dataset: {dataset_path}")
        return 2

    X, y, w, ds = _load_dataset(dataset_path)
    if len(y) < 20:
        print(f"Not enough rows to train behavior model: {len(y)}")
        return 2

    rng = np.random.default_rng(args.seed)
    idx = np.arange(len(y))
    rng.shuffle(idx)

    test_n = max(1, int(len(y) * max(min(args.test_ratio, 0.5), 0.05)))
    test_idx = idx[:test_n]
    train_idx = idx[test_n:]
    if len(train_idx) < 5:
        print("Not enough training rows after split")
        return 2

    x_tr, x_te, mu, sigma = _standardize_train_test(X, train_idx, test_idx)
    y_tr, y_te = y[train_idx], y[test_idx]
    w_tr, w_te = w[train_idx], w[test_idx]

    # Rebalance training emphasis to prevent one class from dominating metrics.
    w_tr = _rebalance_weights(y_tr, w_tr, cap=max(args.class_balance_cap, 1.0))

    _, d = x_tr.shape
    c = len(LABEL_TO_ID)

    W = np.zeros((d, c), dtype=np.float64)
    b = np.zeros((1, c), dtype=np.float64)

    y_tr_oh = _one_hot(y_tr, c)
    best = {"epoch": -1, "loss": math.inf, "W": None, "b": None}

    for epoch in range(args.epochs):
        logits = x_tr @ W + b
        probs = _softmax(logits)

        err = (probs - y_tr_oh) * w_tr[:, None]
        grad_W = (x_tr.T @ err) / np.clip(np.sum(w_tr), 1e-8, None) + (args.l2 * W)
        grad_b = np.sum(err, axis=0, keepdims=True) / np.clip(np.sum(w_tr), 1e-8, None)

        W -= args.lr * grad_W
        b -= args.lr * grad_b

        tr_loss = _weighted_ce(probs, y_tr, w_tr)
        if tr_loss < best["loss"]:
            best = {"epoch": epoch, "loss": tr_loss, "W": W.copy(), "b": b.copy()}

        if epoch % 50 == 0 or epoch == args.epochs - 1:
            te_probs = _softmax(x_te @ W + b)
            te_loss = _weighted_ce(te_probs, y_te, w_te)
            te_acc = float(np.mean(np.argmax(te_probs, axis=1) == y_te))
            print(f"Epoch {epoch:03d} | TrainLoss {tr_loss:.4f} | TestLoss {te_loss:.4f} | TestAcc {te_acc:.4f}")

    W = best["W"]
    b = best["b"]

    tr_probs = _softmax(x_tr @ W + b)
    te_probs = _softmax(x_te @ W + b)

    tr_metrics = _metrics(tr_probs, y_tr)
    te_metrics = _metrics(te_probs, y_te)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    model_path = MODELS_DIR / f"trade_behavior_policy_{timestamp}.npz"
    np.savez_compressed(
        model_path,
        W=W.astype(np.float32),
        b=b.astype(np.float32),
        mu=mu.astype(np.float32),
        sigma=sigma.astype(np.float32),
        labels=np.asarray(["negative", "neutral", "positive"]),
        feature_dim=np.asarray([X.shape[1]], dtype=np.int32),
    )

    label_counts = {name: int(np.sum(y == cid)) for name, cid in LABEL_TO_ID.items()}

    log_payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": str(dataset_path),
        "rows": int(len(y)),
        "train_rows": int(len(y_tr)),
        "test_rows": int(len(y_te)),
        "label_counts": label_counts,
        "best_epoch": int(best["epoch"]),
        "best_train_loss": float(best["loss"]),
        "train_metrics": tr_metrics,
        "test_metrics": te_metrics,
        "model_path": str(model_path),
        "outcome_learning": ds.get("outcome_learning", {}),
    }

    log_path = LOGS_DIR / f"trade_behavior_policy_{timestamp}.json"
    log_path.write_text(json.dumps(log_payload, indent=2), encoding="utf-8")

    print("Saved model:", model_path)
    print("Saved log:", log_path)
    print(json.dumps({"label_counts": label_counts, "test_metrics": te_metrics}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
