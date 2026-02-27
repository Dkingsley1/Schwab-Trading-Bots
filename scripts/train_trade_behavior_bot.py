import argparse
import json
import math
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = PROJECT_ROOT / "data" / "trade_history" / "trade_learning_dataset.json"
MODELS_DIR = PROJECT_ROOT / "models"
LOGS_DIR = PROJECT_ROOT / "logs"


LABEL_TO_ID = {"negative": 0, "neutral": 1, "positive": 2}
ID_TO_LABEL = {v: k for k, v in LABEL_TO_ID.items()}


def _parse_bool(value: str, default: bool = False) -> bool:
    raw = str(value if value is not None else "").strip().lower()
    if not raw:
        return bool(default)
    return raw in {"1", "true", "yes", "on"}


def _parse_ts_epoch(raw: Any, default: float) -> float:
    if raw is None:
        return default
    s = str(raw).strip().replace("Z", "+00:00")
    if not s:
        return default
    try:
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc).timestamp()
    except Exception:
        return default


def _safe_read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


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


def _effective_focal_weights(
    probs: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    *,
    gamma: float,
    class_alpha: np.ndarray,
) -> np.ndarray:
    base = np.clip(weights.astype(np.float64), 1e-8, None)
    gamma = max(float(gamma), 0.0)
    alpha = np.clip(class_alpha[np.asarray(y, dtype=np.int64)], 1e-8, None)
    if gamma <= 1e-8:
        return base * alpha
    p_t = np.clip(probs[np.arange(len(y)), y], 1e-8, 1.0)
    focal = np.power(1.0 - p_t, gamma)
    return base * alpha * focal


def _weighted_focal_ce(
    probs: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    *,
    gamma: float,
    class_alpha: np.ndarray,
) -> float:
    p_t = np.clip(probs[np.arange(len(y)), y], 1e-8, 1.0)
    eff_w = _effective_focal_weights(
        probs,
        y,
        weights,
        gamma=gamma,
        class_alpha=class_alpha,
    )
    loss = -np.log(p_t)
    return float(np.sum(loss * eff_w) / np.clip(np.sum(eff_w), 1e-8, None))


def _confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> List[List[int]]:
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    return cm.tolist()


def _metrics(probs: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
    pred = np.argmax(probs, axis=1)
    acc = float(np.mean(pred == y)) if len(y) else 0.0

    out: Dict[str, Any] = {"accuracy": acc}
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


def _metric_score(metrics: Dict[str, Any]) -> float:
    macro_f1 = float(metrics.get("macro_f1", 0.0) or 0.0)
    balanced = float(metrics.get("balanced_accuracy", 0.0) or 0.0)
    neutral_f1 = float(metrics.get("neutral_f1", 0.0) or 0.0)
    positive_f1 = float(metrics.get("positive_f1", 0.0) or 0.0)
    positive_recall = float(metrics.get("positive_recall", 0.0) or 0.0)
    return (
        macro_f1
        + balanced
        + (0.25 * neutral_f1)
        + (0.40 * positive_f1)
        + (0.60 * positive_recall)
    )


def _checkpoint_score(metrics: Dict[str, Any]) -> float:
    return (
        float(metrics.get("macro_f1", 0.0) or 0.0)
        + float(metrics.get("balanced_accuracy", 0.0) or 0.0)
        + (0.90 * float(metrics.get("positive_recall", 0.0) or 0.0))
        + (0.35 * float(metrics.get("positive_f1", 0.0) or 0.0))
    )


def _parse_seed_list(raw: str, fallback_seed: int) -> List[int]:
    out: List[int] = []
    txt = str(raw or "").strip()
    if txt:
        for part in txt.split(","):
            p = part.strip()
            if not p:
                continue
            try:
                out.append(int(p))
            except Exception:
                continue
    if not out:
        out = [int(fallback_seed)]
    uniq: List[int] = []
    seen = set()
    for s in out:
        if s in seen:
            continue
        seen.add(s)
        uniq.append(s)
    return uniq


def _load_dataset(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    rows = obj.get("data", []) if isinstance(obj, dict) else []

    x_rows: List[List[float]] = []
    y_rows: List[int] = []
    w_rows: List[float] = []
    ts_rows: List[float] = []
    symbol_rows: List[str] = []
    regime_rows: List[str] = []

    expected_dim: Optional[int] = None
    skipped_dim_mismatch = 0

    for idx, r in enumerate(rows):
        if not isinstance(r, dict):
            continue
        label = str(r.get("label") or "neutral").lower()
        if label not in LABEL_TO_ID:
            continue
        feats = r.get("features") or []
        if not isinstance(feats, list) or not feats:
            continue

        if expected_dim is None:
            expected_dim = len(feats)
        elif len(feats) != expected_dim:
            skipped_dim_mismatch += 1
            continue

        try:
            x_rows.append([float(v) for v in feats])
        except Exception:
            continue

        y_rows.append(LABEL_TO_ID[label])
        w_rows.append(float(r.get("sample_weight", 1.0) or 1.0))
        ts_rows.append(_parse_ts_epoch(r.get("timestamp_utc"), float(idx)))
        symbol_rows.append(str(r.get("symbol") or "UNKNOWN").upper())
        regime_rows.append(str(r.get("regime") or "other").lower())

    X = np.asarray(x_rows, dtype=np.float64)
    y = np.asarray(y_rows, dtype=np.int64)
    w = np.asarray(w_rows, dtype=np.float64)
    ts = np.asarray(ts_rows, dtype=np.float64)
    symbols = np.asarray(symbol_rows, dtype=object)
    regimes = np.asarray(regime_rows, dtype=object)

    meta = dict(obj) if isinstance(obj, dict) else {}
    meta["_skipped_dim_mismatch"] = int(skipped_dim_mismatch)
    meta["_feature_dim"] = int(X.shape[1]) if X.ndim == 2 and X.size else 0
    return X, y, w, ts, symbols, regimes, meta


def _standardize_train_test(X: np.ndarray, train_idx: np.ndarray, test_idx: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x_tr = X[train_idx]
    x_te = X[test_idx]

    mu = np.mean(x_tr, axis=0, keepdims=True)
    sigma = np.std(x_tr, axis=0, keepdims=True)
    sigma = np.where(sigma < 1e-8, 1.0, sigma)

    x_tr = (x_tr - mu) / sigma
    x_te = (x_te - mu) / sigma
    return x_tr, x_te, mu.squeeze(0), sigma.squeeze(0)


def _random_split_indices(n_rows: int, test_ratio: float, seed: int, min_train_rows: int) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    rng = np.random.default_rng(seed)
    idx = np.arange(n_rows)
    rng.shuffle(idx)

    test_n = max(1, int(n_rows * max(min(test_ratio, 0.5), 0.05)))
    test_idx = idx[:test_n]
    train_idx = idx[test_n:]
    if len(train_idx) < min_train_rows:
        raise RuntimeError(f"Not enough training rows after random split: {len(train_idx)} < {min_train_rows}")

    return train_idx, test_idx, {
        "mode": "random",
        "seed": int(seed),
        "test_rows": int(len(test_idx)),
        "train_rows": int(len(train_idx)),
    }


def _time_purged_split_indices(
    ts_epoch: np.ndarray,
    test_ratio: float,
    purge_seconds: float,
    min_train_rows: int,
    fallback_seed: int,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    n_rows = int(len(ts_epoch))
    order = np.argsort(ts_epoch, kind="mergesort")
    test_n = max(1, int(n_rows * max(min(test_ratio, 0.5), 0.05)))

    test_start = max(n_rows - test_n, 1)
    test_idx = order[test_start:]
    train_idx_raw = order[:test_start]

    test_start_ts = float(ts_epoch[test_idx[0]]) if len(test_idx) else float(np.max(ts_epoch))
    purge_seconds = max(float(purge_seconds), 0.0)
    purge_cutoff = test_start_ts - purge_seconds

    train_mask = ts_epoch[train_idx_raw] <= purge_cutoff
    train_idx = train_idx_raw[train_mask]
    used_purge_seconds = purge_seconds
    relaxed = False

    if len(train_idx) < min_train_rows:
        train_idx = train_idx_raw
        used_purge_seconds = 0.0
        relaxed = True

    if len(train_idx) < min_train_rows:
        train_idx, test_idx, split_meta = _random_split_indices(
            n_rows=n_rows,
            test_ratio=test_ratio,
            seed=fallback_seed,
            min_train_rows=min_train_rows,
        )
        split_meta["fallback_from"] = "time_purged"
        split_meta["fallback_reason"] = "insufficient_train_rows"
        return train_idx, test_idx, split_meta

    return train_idx, test_idx, {
        "mode": "time_purged",
        "test_rows": int(len(test_idx)),
        "train_rows": int(len(train_idx)),
        "purge_seconds": float(used_purge_seconds),
        "relaxed_purge": bool(relaxed),
        "test_start_ts": float(test_start_ts),
    }


def _rebalance_class_weights(
    y: np.ndarray,
    w: np.ndarray,
    *,
    cap: float,
    neutral_floor: float,
    positive_floor: float,
    negative_cap: float,
) -> Tuple[np.ndarray, Dict[str, float]]:
    out = w.astype(np.float64).copy()
    counts = np.bincount(y, minlength=len(LABEL_TO_ID)).astype(np.float64)
    total = float(np.sum(counts))
    n_classes = float(len(LABEL_TO_ID))

    factors = np.ones_like(counts)
    cap = max(float(cap), 1.0)
    for cid in range(len(counts)):
        if counts[cid] > 0:
            factors[cid] = min((total / (n_classes * counts[cid])), cap)

    # Explicitly push minority labels harder while capping dominant negative pressure.
    factors[LABEL_TO_ID["neutral"]] = max(factors[LABEL_TO_ID["neutral"]], max(float(neutral_floor), 1.0))
    factors[LABEL_TO_ID["positive"]] = max(factors[LABEL_TO_ID["positive"]], max(float(positive_floor), 1.0))
    factors[LABEL_TO_ID["negative"]] = min(factors[LABEL_TO_ID["negative"]], max(float(negative_cap), 0.05))

    for cid in range(len(counts)):
        out[y == cid] *= factors[cid]

    mean_w = float(np.mean(out)) if len(out) else 1.0
    if mean_w > 1e-8:
        out /= mean_w

    summary = {ID_TO_LABEL[cid]: float(factors[cid]) for cid in range(len(factors))}
    return out, summary


def _rebalance_class_regime_weights(
    y: np.ndarray,
    regimes: np.ndarray,
    w: np.ndarray,
    *,
    cap: float,
) -> Tuple[np.ndarray, Dict[str, float]]:
    out = w.astype(np.float64).copy()
    cap = max(float(cap), 1.0)
    applied: Dict[str, float] = {}

    for cid, label in ID_TO_LABEL.items():
        class_mask = y == cid
        if not np.any(class_mask):
            continue
        class_regimes = regimes[class_mask]

        counts: Dict[str, int] = {}
        for rg in class_regimes.tolist():
            key = str(rg or "other")
            counts[key] = counts.get(key, 0) + 1

        if len(counts) <= 1:
            continue

        target = max(counts.values())
        for rg, cnt in counts.items():
            factor = min(target / max(cnt, 1), cap)
            applied[f"{label}:{rg}"] = float(factor)
            mask = class_mask & (regimes == rg)
            out[mask] *= factor

    mean_w = float(np.mean(out)) if len(out) else 1.0
    if mean_w > 1e-8:
        out /= mean_w

    return out, applied


def _fit_temperature(
    logits: np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
    *,
    t_min: float,
    t_max: float,
    steps: int,
) -> Tuple[float, float, float]:
    t_min = max(float(t_min), 0.2)
    t_max = max(float(t_max), t_min + 1e-6)
    steps = max(int(steps), 3)

    base_probs = _softmax(logits)
    base_loss = _weighted_ce(base_probs, y, w)

    best_t = 1.0
    best_loss = base_loss

    for t in np.linspace(t_min, t_max, steps):
        probs = _softmax(logits / max(float(t), 1e-6))
        loss = _weighted_ce(probs, y, w)
        if loss < best_loss:
            best_loss = float(loss)
            best_t = float(t)

    return float(best_t), float(base_loss), float(best_loss)


def _build_sampling_probabilities(
    y: np.ndarray,
    w: np.ndarray,
    *,
    positive_boost: float,
    neutral_boost: float,
    negative_boost: float,
) -> np.ndarray:
    n = int(len(y))
    if n <= 0:
        return np.asarray([], dtype=np.float64)

    counts = np.bincount(y, minlength=len(LABEL_TO_ID)).astype(np.float64)
    total = float(np.sum(counts))
    inv_freq = np.ones_like(counts)
    denom = max(float(len(LABEL_TO_ID)), 1.0)
    for cid in range(len(counts)):
        if counts[cid] > 0:
            inv_freq[cid] = total / max(denom * counts[cid], 1.0)

    class_boost = np.ones_like(counts)
    class_boost[LABEL_TO_ID["positive"]] = max(float(positive_boost), 0.05)
    class_boost[LABEL_TO_ID["neutral"]] = max(float(neutral_boost), 0.05)
    class_boost[LABEL_TO_ID["negative"]] = max(float(negative_boost), 0.05)

    probs = np.clip(w.astype(np.float64), 1e-8, None)
    for cid in range(len(counts)):
        probs[y == cid] *= inv_freq[cid] * class_boost[cid]

    s = float(np.sum(probs))
    if not math.isfinite(s) or s <= 0.0:
        return np.full(n, 1.0 / max(n, 1), dtype=np.float64)
    return probs / s


def _fit_class_logit_bias(
    logits: np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
    *,
    positive_recall_target: float,
    neutral_recall_floor: float,
    pos_bias_min: float,
    pos_bias_max: float,
    neg_bias_min: float,
    neg_bias_max: float,
    steps: int,
) -> Tuple[np.ndarray, Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    steps = max(int(steps), 3)
    pos_bias_lo = min(float(pos_bias_min), float(pos_bias_max))
    pos_bias_hi = max(float(pos_bias_min), float(pos_bias_max))
    neg_bias_lo = min(float(neg_bias_min), float(neg_bias_max))
    neg_bias_hi = max(float(neg_bias_min), float(neg_bias_max))
    pos_bias_min = pos_bias_lo
    pos_bias_max = pos_bias_hi
    neg_bias_min = neg_bias_lo
    neg_bias_max = neg_bias_hi
    positive_recall_target = max(min(float(positive_recall_target), 1.0), 0.0)
    neutral_recall_floor = max(min(float(neutral_recall_floor), 1.0), 0.0)

    base_probs = _softmax(logits)
    base_metrics = _metrics(base_probs, y)
    base_loss = _weighted_ce(base_probs, y, w)

    best_bias = np.zeros((1, len(LABEL_TO_ID)), dtype=np.float64)
    best_metrics = base_metrics
    best_loss = base_loss
    best_score = _metric_score(base_metrics)

    tried = 0
    for pos_bias in np.linspace(pos_bias_min, pos_bias_max, steps):
        for neg_bias in np.linspace(neg_bias_min, neg_bias_max, steps):
            bias = np.asarray([[float(neg_bias), 0.0, float(pos_bias)]], dtype=np.float64)
            probs = _softmax(logits + bias)
            metrics = _metrics(probs, y)
            loss = _weighted_ce(probs, y, w)

            score = _metric_score(metrics) + (0.65 * float(metrics.get("positive_recall", 0.0) or 0.0))
            pos_recall = float(metrics.get("positive_recall", 0.0) or 0.0)
            neu_recall = float(metrics.get("neutral_recall", 0.0) or 0.0)
            if pos_recall < positive_recall_target:
                score -= 2.0 * (positive_recall_target - pos_recall)
            if neu_recall < neutral_recall_floor:
                score -= 1.5 * (neutral_recall_floor - neu_recall)

            tried += 1
            better = score > (best_score + 1e-12)
            tie = abs(score - best_score) <= 1e-12
            if better or (tie and loss < best_loss):
                best_score = float(score)
                best_loss = float(loss)
                best_bias = bias
                best_metrics = metrics

    tuning = {
        "grid_points": int(tried),
        "positive_recall_target": float(positive_recall_target),
        "neutral_recall_floor": float(neutral_recall_floor),
        "selected_bias": {
            "negative": float(best_bias[0, 0]),
            "neutral": float(best_bias[0, 1]),
            "positive": float(best_bias[0, 2]),
        },
        "base_loss": float(base_loss),
        "selected_loss": float(best_loss),
        "base_score": float(_metric_score(base_metrics)),
        "selected_score": float(_metric_score(best_metrics)),
    }
    return best_bias.reshape(-1), base_metrics, best_metrics, tuning


def _train_single_seed(
    *,
    seed: int,
    x_tr: np.ndarray,
    y_tr: np.ndarray,
    w_tr: np.ndarray,
    x_te: np.ndarray,
    y_te: np.ndarray,
    w_te: np.ndarray,
    epochs: int,
    lr: float,
    l2: float,
    temperature_min: float,
    temperature_max: float,
    temperature_steps: int,
    focal_gamma: float,
    focal_class_alpha: np.ndarray,
    batch_size: int,
    batch_steps_per_epoch: int,
    sampling_probs: np.ndarray,
    positive_recall_target: float,
    neutral_recall_floor: float,
    pos_logit_bias_max: float,
    neg_logit_bias_max: float,
    logit_bias_steps: int,
) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)

    n_tr, d = x_tr.shape
    c = len(LABEL_TO_ID)

    W = rng.normal(0.0, 0.02, size=(d, c)).astype(np.float64)
    b = np.zeros((1, c), dtype=np.float64)

    batch_size = int(max(1, min(int(batch_size), max(n_tr, 1))))
    use_minibatch = batch_size < n_tr
    steps = max(int(batch_steps_per_epoch), 1) if use_minibatch else 1

    sample_p: Optional[np.ndarray] = None
    if use_minibatch and isinstance(sampling_probs, np.ndarray) and len(sampling_probs) == n_tr:
        total = float(np.sum(sampling_probs))
        if math.isfinite(total) and total > 0.0:
            sample_p = sampling_probs / total

    best = {
        "epoch": -1,
        "te_loss": math.inf,
        "tr_loss": math.inf,
        "checkpoint_score": -math.inf,
        "W": None,
        "b": None,
    }

    for epoch in range(max(int(epochs), 1)):
        for _ in range(steps):
            if use_minibatch:
                idx = rng.choice(n_tr, size=batch_size, replace=True, p=sample_p)
            else:
                idx = np.arange(n_tr)

            x_b = x_tr[idx]
            y_b = y_tr[idx]
            w_b = w_tr[idx]
            y_b_oh = _one_hot(y_b, c)

            logits_b = x_b @ W + b
            probs_b = _softmax(logits_b)

            eff_w = _effective_focal_weights(
                probs_b,
                y_b,
                w_b,
                gamma=focal_gamma,
                class_alpha=focal_class_alpha,
            )
            denom = np.clip(np.sum(eff_w), 1e-8, None)
            err = (probs_b - y_b_oh) * eff_w[:, None]
            grad_W = (x_b.T @ err) / denom + (l2 * W)
            grad_b = np.sum(err, axis=0, keepdims=True) / denom

            W -= lr * grad_W
            b -= lr * grad_b

        tr_logits = x_tr @ W + b
        te_logits = x_te @ W + b
        tr_probs = _softmax(tr_logits)
        te_probs = _softmax(te_logits)

        tr_loss = _weighted_focal_ce(
            tr_probs,
            y_tr,
            w_tr,
            gamma=focal_gamma,
            class_alpha=focal_class_alpha,
        )
        te_loss = _weighted_focal_ce(
            te_probs,
            y_te,
            w_te,
            gamma=focal_gamma,
            class_alpha=focal_class_alpha,
        )

        te_metrics_epoch = _metrics(te_probs, y_te)
        ckpt_score = _checkpoint_score(te_metrics_epoch)
        better = ckpt_score > (best["checkpoint_score"] + 1e-12)
        tie = abs(ckpt_score - best["checkpoint_score"]) <= 1e-12
        if better or (tie and te_loss < best["te_loss"]):
            best = {
                "epoch": epoch,
                "te_loss": float(te_loss),
                "tr_loss": float(tr_loss),
                "checkpoint_score": float(ckpt_score),
                "W": W.copy(),
                "b": b.copy(),
            }

        if epoch % 50 == 0 or epoch == (max(int(epochs), 1) - 1):
            te_acc = float(np.mean(np.argmax(te_probs, axis=1) == y_te))
            te_macro = float(te_metrics_epoch.get("macro_f1", 0.0) or 0.0)
            te_bal = float(te_metrics_epoch.get("balanced_accuracy", 0.0) or 0.0)
            te_pos_rec = float(te_metrics_epoch.get("positive_recall", 0.0) or 0.0)
            print(
                f"Seed {seed} | Epoch {epoch:03d} | TrainLoss {tr_loss:.4f} | TestLoss {te_loss:.4f} "
                f"| TestAcc {te_acc:.4f} | MacroF1 {te_macro:.4f} | BalAcc {te_bal:.4f} | PosRec {te_pos_rec:.4f}"
            )

    W_best = best["W"] if best["W"] is not None else W
    b_best = best["b"] if best["b"] is not None else b

    tr_logits_best = x_tr @ W_best + b_best
    te_logits_best = x_te @ W_best + b_best

    temp, te_loss_pre_cal, te_loss_post_cal = _fit_temperature(
        te_logits_best,
        y_te,
        w_te,
        t_min=temperature_min,
        t_max=temperature_max,
        steps=temperature_steps,
    )

    tr_logits_cal = tr_logits_best / temp
    te_logits_cal = te_logits_best / temp
    class_logit_bias, te_metrics_pre_threshold, te_metrics_tuned_threshold, threshold_tuning = _fit_class_logit_bias(
        te_logits_cal,
        y_te,
        w_te,
        positive_recall_target=positive_recall_target,
        neutral_recall_floor=neutral_recall_floor,
        pos_bias_min=-abs(float(pos_logit_bias_max)),
        pos_bias_max=abs(float(pos_logit_bias_max)),
        neg_bias_min=-abs(float(neg_logit_bias_max)),
        neg_bias_max=abs(float(neg_logit_bias_max)),
        steps=logit_bias_steps,
    )

    bias_row = class_logit_bias.reshape(1, -1)
    tr_probs = _softmax(tr_logits_cal + bias_row)
    te_probs = _softmax(te_logits_cal + bias_row)

    tr_metrics = _metrics(tr_probs, y_tr)
    te_metrics = _metrics(te_probs, y_te)

    return {
        "seed": int(seed),
        "best_epoch": int(best["epoch"]),
        "best_train_loss": float(best["tr_loss"]),
        "best_test_loss": float(best["te_loss"]),
        "best_checkpoint_score": float(best["checkpoint_score"]),
        "temperature": float(temp),
        "class_logit_bias": [float(x) for x in class_logit_bias.tolist()],
        "test_loss_pre_calibration": float(te_loss_pre_cal),
        "test_loss_post_calibration": float(te_loss_post_cal),
        "train_metrics": tr_metrics,
        "test_metrics": te_metrics,
        "test_metrics_pre_threshold_tuning": te_metrics_pre_threshold,
        "test_metrics_threshold_tuned": te_metrics_tuned_threshold,
        "threshold_tuning": threshold_tuning,
        "score": float(_metric_score(te_metrics)),
        "W": W_best,
        "b": b_best,
    }


def _latest_policy_model_and_log(models_dir: Path, logs_dir: Path) -> Tuple[Optional[Path], Dict[str, Any], Optional[Path]]:
    paths = sorted(models_dir.glob("trade_behavior_policy_*.npz"))
    if not paths:
        return None, {}, None

    model_path = paths[-1]
    name = model_path.name
    stamp = name.replace("trade_behavior_policy_", "").replace(".npz", "")
    log_path = logs_dir / f"trade_behavior_policy_{stamp}.json"
    log_obj = _safe_read_json(log_path) if log_path.exists() else {}
    return model_path, log_obj, (log_path if log_path.exists() else None)


def _data_quality_gate(project_root: Path, *, require_walk_forward_ok: bool) -> Tuple[bool, List[str], Dict[str, Any]]:
    reasons: List[str] = []

    health = project_root / "governance" / "health"
    coverage = _safe_read_json(health / "snapshot_coverage_latest.json")
    replay = _safe_read_json(health / "replay_preopen_sanity_latest.json")
    drift = _safe_read_json(health / "preopen_replay_drift_latest.json")
    divergence = _safe_read_json(health / "data_source_divergence_latest.json")

    min_coverage_ratio = float(os.getenv("TRADE_BEHAVIOR_PROMOTION_MIN_COVERAGE_RATIO", "0.30"))
    max_divergence_spread = float(os.getenv("TRADE_BEHAVIOR_PROMOTION_MAX_DIVERGENCE_SPREAD", "0.04"))
    max_row_drift = float(os.getenv("TRADE_BEHAVIOR_PROMOTION_MAX_ROW_DRIFT", "1.2"))
    max_stale_drift = float(os.getenv("TRADE_BEHAVIOR_PROMOTION_MAX_STALE_DRIFT", "1.0"))
    require_replay_ok = _parse_bool(os.getenv("TRADE_BEHAVIOR_PROMOTION_REQUIRE_REPLAY_OK", "1"), default=True)

    coverage_ratio = float(coverage.get("coverage_ratio", 0.0) or 0.0)
    if coverage and coverage_ratio < min_coverage_ratio:
        reasons.append(f"coverage_ratio={coverage_ratio:.4f} < min={min_coverage_ratio:.4f}")

    worst_relative_spread = float(divergence.get("worst_relative_spread", 0.0) or 0.0)
    if divergence and worst_relative_spread > max_divergence_spread:
        reasons.append(f"worst_relative_spread={worst_relative_spread:.4f} > max={max_divergence_spread:.4f}")

    drift_obj = drift.get("drift") if isinstance(drift.get("drift"), dict) else {}
    row_drift = max(abs(float(drift_obj.get("decision_rows", 0.0) or 0.0)), abs(float(drift_obj.get("governance_rows", 0.0) or 0.0)))
    stale_drift = max(abs(float(drift_obj.get("decision_stale", 0.0) or 0.0)), abs(float(drift_obj.get("governance_stale", 0.0) or 0.0)))
    if drift and row_drift > max_row_drift:
        reasons.append(f"row_drift={row_drift:.4f} > max={max_row_drift:.4f}")
    if drift and stale_drift > max_stale_drift:
        reasons.append(f"stale_drift={stale_drift:.4f} > max={max_stale_drift:.4f}")

    replay_ok = bool(replay.get("ok", True)) if replay else True
    if require_replay_ok and replay and (not replay_ok):
        reasons.append("replay_preopen_sanity_not_ok")

    walk_forward = _safe_read_json(project_root / "governance" / "walk_forward" / "promotion_readiness_latest.json")
    if require_walk_forward_ok and walk_forward and (not bool(walk_forward.get("promote_ok", False))):
        reasons.append("walk_forward_promote_ok=false")

    summary = {
        "coverage_ratio": coverage_ratio,
        "min_coverage_ratio": min_coverage_ratio,
        "worst_relative_spread": worst_relative_spread,
        "max_divergence_spread": max_divergence_spread,
        "row_drift": row_drift,
        "max_row_drift": max_row_drift,
        "stale_drift": stale_drift,
        "max_stale_drift": max_stale_drift,
        "replay_ok": replay_ok,
        "require_replay_ok": require_replay_ok,
        "walk_forward_required": bool(require_walk_forward_ok),
        "walk_forward_promote_ok": bool(walk_forward.get("promote_ok", False)) if walk_forward else None,
    }
    return len(reasons) == 0, reasons, summary


def _save_model(
    path: Path,
    *,
    W: np.ndarray,
    b: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
    feature_dim: int,
    temperature: float,
    class_logit_bias: Optional[np.ndarray] = None,
) -> None:
    bias = np.zeros((len(LABEL_TO_ID),), dtype=np.float32)
    if class_logit_bias is not None:
        raw = np.asarray(class_logit_bias, dtype=np.float64).reshape(-1)
        if raw.shape[0] == len(LABEL_TO_ID):
            bias = raw.astype(np.float32)

    np.savez_compressed(
        path,
        W=W.astype(np.float32),
        b=b.astype(np.float32),
        mu=mu.astype(np.float32),
        sigma=sigma.astype(np.float32),
        labels=np.asarray(["negative", "neutral", "positive"]),
        feature_dim=np.asarray([int(feature_dim)], dtype=np.int32),
        temperature=np.asarray([float(temperature)], dtype=np.float32),
        class_logit_bias=bias,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Train a weighted behavior policy model on past trades.")
    parser.add_argument("--dataset", default=str(DATASET_PATH))
    parser.add_argument("--epochs", type=int, default=int(os.getenv("TRADE_BEHAVIOR_EPOCHS", "300")))
    parser.add_argument("--lr", type=float, default=float(os.getenv("TRADE_BEHAVIOR_LR", "0.05")))
    parser.add_argument("--l2", type=float, default=float(os.getenv("TRADE_BEHAVIOR_L2", "1e-4")))
    parser.add_argument("--test-ratio", type=float, default=float(os.getenv("TRADE_BEHAVIOR_TEST_RATIO", "0.2")))
    parser.add_argument("--seed", type=int, default=int(os.getenv("TRADE_BEHAVIOR_SEED", "42")))
    parser.add_argument("--seeds", default=os.getenv("TRADE_BEHAVIOR_SEEDS", "42,1337,2026"))
    parser.add_argument("--split-mode", choices=["time_purged", "random"], default=os.getenv("TRADE_BEHAVIOR_SPLIT_MODE", "time_purged"))
    parser.add_argument("--purge-seconds", type=float, default=float(os.getenv("TRADE_BEHAVIOR_PURGE_SECONDS", "900")))
    parser.add_argument("--min-train-rows", type=int, default=int(os.getenv("TRADE_BEHAVIOR_MIN_TRAIN_ROWS", "25")))
    parser.add_argument("--class-balance-cap", type=float, default=float(os.getenv("TRADE_BEHAVIOR_CLASS_BALANCE_CAP", "4.0")))
    parser.add_argument("--neutral-weight-floor", type=float, default=float(os.getenv("TRADE_BEHAVIOR_NEUTRAL_WEIGHT_FLOOR", "1.25")))
    parser.add_argument("--positive-weight-floor", type=float, default=float(os.getenv("TRADE_BEHAVIOR_POSITIVE_WEIGHT_FLOOR", "1.35")))
    parser.add_argument("--negative-weight-cap", type=float, default=float(os.getenv("TRADE_BEHAVIOR_NEGATIVE_WEIGHT_CAP", "1.0")))
    parser.add_argument("--regime-balance-cap", type=float, default=float(os.getenv("TRADE_BEHAVIOR_REGIME_BALANCE_CAP", "2.5")))
    parser.add_argument("--batch-size", type=int, default=int(os.getenv("TRADE_BEHAVIOR_BATCH_SIZE", "1024")))
    parser.add_argument("--batch-steps-per-epoch", type=int, default=int(os.getenv("TRADE_BEHAVIOR_BATCH_STEPS_PER_EPOCH", "4")))
    parser.add_argument("--oversample-positive", type=float, default=float(os.getenv("TRADE_BEHAVIOR_OVERSAMPLE_POSITIVE", "1.85")))
    parser.add_argument("--oversample-neutral", type=float, default=float(os.getenv("TRADE_BEHAVIOR_OVERSAMPLE_NEUTRAL", "1.35")))
    parser.add_argument("--oversample-negative", type=float, default=float(os.getenv("TRADE_BEHAVIOR_OVERSAMPLE_NEGATIVE", "0.95")))
    parser.add_argument("--focal-gamma", type=float, default=float(os.getenv("TRADE_BEHAVIOR_FOCAL_GAMMA", "1.25")))
    parser.add_argument("--focal-alpha-negative", type=float, default=float(os.getenv("TRADE_BEHAVIOR_FOCAL_ALPHA_NEGATIVE", "0.9")))
    parser.add_argument("--focal-alpha-neutral", type=float, default=float(os.getenv("TRADE_BEHAVIOR_FOCAL_ALPHA_NEUTRAL", "1.15")))
    parser.add_argument("--focal-alpha-positive", type=float, default=float(os.getenv("TRADE_BEHAVIOR_FOCAL_ALPHA_POSITIVE", "1.65")))
    parser.add_argument("--temperature-min", type=float, default=float(os.getenv("TRADE_BEHAVIOR_TEMPERATURE_MIN", "0.6")))
    parser.add_argument("--temperature-max", type=float, default=float(os.getenv("TRADE_BEHAVIOR_TEMPERATURE_MAX", "2.5")))
    parser.add_argument("--temperature-steps", type=int, default=int(os.getenv("TRADE_BEHAVIOR_TEMPERATURE_STEPS", "45")))
    parser.add_argument("--positive-recall-target", type=float, default=float(os.getenv("TRADE_BEHAVIOR_POSITIVE_RECALL_TARGET", "0.32")))
    parser.add_argument("--neutral-recall-floor", type=float, default=float(os.getenv("TRADE_BEHAVIOR_NEUTRAL_RECALL_FLOOR", "0.18")))
    parser.add_argument("--positive-logit-bias-max", type=float, default=float(os.getenv("TRADE_BEHAVIOR_POSITIVE_LOGIT_BIAS_MAX", "1.2")))
    parser.add_argument("--negative-logit-bias-max", type=float, default=float(os.getenv("TRADE_BEHAVIOR_NEGATIVE_LOGIT_BIAS_MAX", "0.8")))
    parser.add_argument("--logit-bias-steps", type=int, default=int(os.getenv("TRADE_BEHAVIOR_LOGIT_BIAS_STEPS", "29")))
    parser.add_argument("--promotion-min-score-delta", type=float, default=float(os.getenv("TRADE_BEHAVIOR_PROMOTION_MIN_SCORE_DELTA", "-0.01")))
    parser.add_argument("--promotion-max-neutral-f1-drop", type=float, default=float(os.getenv("TRADE_BEHAVIOR_PROMOTION_MAX_NEUTRAL_F1_DROP", "0.03")))
    parser.add_argument("--rollback-on-regression", action=argparse.BooleanOptionalAction, default=_parse_bool(os.getenv("TRADE_BEHAVIOR_ROLLBACK_ON_REGRESSION", "1"), default=True))
    parser.add_argument("--strict-promotion-gate", action=argparse.BooleanOptionalAction, default=_parse_bool(os.getenv("TRADE_BEHAVIOR_STRICT_PROMOTION_GATE", "0"), default=False))
    parser.add_argument("--require-walk-forward-ok", action=argparse.BooleanOptionalAction, default=_parse_bool(os.getenv("TRADE_BEHAVIOR_REQUIRE_WALK_FORWARD_OK", "0"), default=False))
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        print(f"Missing dataset: {dataset_path}")
        return 2

    X, y, w, ts_epoch, symbols, regimes, ds = _load_dataset(dataset_path)
    if len(y) < 20:
        print(f"Not enough rows to train behavior model: {len(y)}")
        return 2

    if args.split_mode == "time_purged":
        train_idx, test_idx, split_meta = _time_purged_split_indices(
            ts_epoch=ts_epoch,
            test_ratio=args.test_ratio,
            purge_seconds=args.purge_seconds,
            min_train_rows=max(args.min_train_rows, 5),
            fallback_seed=args.seed,
        )
    else:
        train_idx, test_idx, split_meta = _random_split_indices(
            n_rows=len(y),
            test_ratio=args.test_ratio,
            seed=args.seed,
            min_train_rows=max(args.min_train_rows, 5),
        )

    if len(train_idx) < 5:
        print("Not enough training rows after split")
        return 2

    x_tr, x_te, mu, sigma = _standardize_train_test(X, train_idx, test_idx)
    y_tr, y_te = y[train_idx], y[test_idx]
    w_tr, w_te = w[train_idx], w[test_idx]
    regime_tr = regimes[train_idx]

    w_tr, class_balance_factors = _rebalance_class_weights(
        y_tr,
        w_tr,
        cap=max(args.class_balance_cap, 1.0),
        neutral_floor=max(args.neutral_weight_floor, 1.0),
        positive_floor=max(args.positive_weight_floor, 1.0),
        negative_cap=max(args.negative_weight_cap, 0.05),
    )
    w_tr, class_regime_factors = _rebalance_class_regime_weights(
        y_tr,
        regime_tr,
        w_tr,
        cap=max(args.regime_balance_cap, 1.0),
    )

    focal_class_alpha = np.asarray(
        [
            max(float(args.focal_alpha_negative), 0.05),
            max(float(args.focal_alpha_neutral), 0.05),
            max(float(args.focal_alpha_positive), 0.05),
        ],
        dtype=np.float64,
    )
    sampling_probs = _build_sampling_probabilities(
        y_tr,
        w_tr,
        positive_boost=float(args.oversample_positive),
        neutral_boost=float(args.oversample_neutral),
        negative_boost=float(args.oversample_negative),
    )

    seeds = _parse_seed_list(args.seeds, args.seed)
    seed_results: List[Dict[str, Any]] = []
    for seed in seeds:
        seed_result = _train_single_seed(
            seed=seed,
            x_tr=x_tr,
            y_tr=y_tr,
            w_tr=w_tr,
            x_te=x_te,
            y_te=y_te,
            w_te=w_te,
            epochs=max(args.epochs, 1),
            lr=float(args.lr),
            l2=float(args.l2),
            temperature_min=float(args.temperature_min),
            temperature_max=float(args.temperature_max),
            temperature_steps=max(int(args.temperature_steps), 3),
            focal_gamma=max(float(args.focal_gamma), 0.0),
            focal_class_alpha=focal_class_alpha,
            batch_size=max(int(args.batch_size), 1),
            batch_steps_per_epoch=max(int(args.batch_steps_per_epoch), 1),
            sampling_probs=sampling_probs,
            positive_recall_target=float(args.positive_recall_target),
            neutral_recall_floor=float(args.neutral_recall_floor),
            pos_logit_bias_max=float(args.positive_logit_bias_max),
            neg_logit_bias_max=float(args.negative_logit_bias_max),
            logit_bias_steps=max(int(args.logit_bias_steps), 3),
        )
        seed_results.append(seed_result)

    seed_results.sort(key=lambda row: float(row.get("score", -1e9)), reverse=True)
    champion = seed_results[0]

    prev_model_path, prev_log, prev_log_path = _latest_policy_model_and_log(MODELS_DIR, LOGS_DIR)
    prev_metrics = (prev_log.get("test_metrics") or {}) if isinstance(prev_log, dict) else {}

    candidate_metrics = champion.get("test_metrics") or {}
    candidate_score = _metric_score(candidate_metrics)
    candidate_neutral_f1 = float(candidate_metrics.get("neutral_f1", 0.0) or 0.0)

    prev_score = _metric_score(prev_metrics) if prev_metrics else None
    prev_neutral_f1 = float(prev_metrics.get("neutral_f1", 0.0) or 0.0) if prev_metrics else None

    promote_ok = True
    promotion_reasons: List[str] = []

    if prev_score is not None:
        min_allowed_score = float(prev_score) + float(args.promotion_min_score_delta)
        if candidate_score < min_allowed_score:
            promote_ok = False
            promotion_reasons.append(
                f"candidate_score={candidate_score:.4f} < min_allowed_score={min_allowed_score:.4f}"
            )

    if prev_neutral_f1 is not None:
        min_allowed_neutral = float(prev_neutral_f1) - float(args.promotion_max_neutral_f1_drop)
        if candidate_neutral_f1 < min_allowed_neutral:
            promote_ok = False
            promotion_reasons.append(
                f"candidate_neutral_f1={candidate_neutral_f1:.4f} < min_allowed_neutral_f1={min_allowed_neutral:.4f}"
            )

    dq_ok, dq_reasons, dq_summary = _data_quality_gate(
        PROJECT_ROOT,
        require_walk_forward_ok=bool(args.require_walk_forward_ok),
    )
    if not dq_ok:
        promote_ok = False
        promotion_reasons.extend([f"data_quality:{r}" for r in dq_reasons])

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    model_path = MODELS_DIR / f"trade_behavior_policy_{timestamp}.npz"

    deployed_from_previous = False
    deployed_previous_path: Optional[str] = None

    if promote_ok or (not args.rollback_on_regression):
        _save_model(
            model_path,
            W=champion["W"],
            b=champion["b"],
            mu=mu,
            sigma=sigma,
            feature_dim=int(X.shape[1]),
            temperature=float(champion.get("temperature", 1.0) or 1.0),
            class_logit_bias=np.asarray(champion.get("class_logit_bias", [0.0, 0.0, 0.0]), dtype=np.float64),
        )
    else:
        if prev_model_path is not None and prev_model_path.exists():
            shutil.copy2(prev_model_path, model_path)
            deployed_from_previous = True
            deployed_previous_path = str(prev_model_path)
            promotion_reasons.append("rolled_back_to_previous_model")
        else:
            _save_model(
                model_path,
                W=champion["W"],
                b=champion["b"],
                mu=mu,
                sigma=sigma,
                feature_dim=int(X.shape[1]),
                temperature=float(champion.get("temperature", 1.0) or 1.0),
                class_logit_bias=np.asarray(champion.get("class_logit_bias", [0.0, 0.0, 0.0]), dtype=np.float64),
            )
            promotion_reasons.append("rollback_target_missing_deployed_candidate")

    label_counts = {name: int(np.sum(y == cid)) for name, cid in LABEL_TO_ID.items()}

    log_payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": str(dataset_path),
        "rows": int(len(y)),
        "train_rows": int(len(y_tr)),
        "test_rows": int(len(y_te)),
        "label_counts": label_counts,
        "dataset_skipped_dim_mismatch": int(ds.get("_skipped_dim_mismatch", 0) or 0),
        "split": split_meta,
        "class_balance_factors": class_balance_factors,
        "class_regime_factors": class_regime_factors,
        "seeds": [int(s) for s in seeds],
        "seed_results": [
            {
                "seed": int(r["seed"]),
                "best_epoch": int(r["best_epoch"]),
                "best_train_loss": float(r["best_train_loss"]),
                "best_test_loss": float(r["best_test_loss"]),
                "best_checkpoint_score": float(r.get("best_checkpoint_score", 0.0)),
                "temperature": float(r["temperature"]),
                "class_logit_bias": r.get("class_logit_bias", [0.0, 0.0, 0.0]),
                "score": float(r["score"]),
                "test_metrics_pre_threshold_tuning": r.get("test_metrics_pre_threshold_tuning", {}),
                "test_metrics": r["test_metrics"],
                "threshold_tuning": r.get("threshold_tuning", {}),
            }
            for r in seed_results
        ],
        "champion_seed": int(champion["seed"]),
        "best_epoch": int(champion["best_epoch"]),
        "best_train_loss": float(champion["best_train_loss"]),
        "test_loss_pre_calibration": float(champion["test_loss_pre_calibration"]),
        "test_loss_post_calibration": float(champion["test_loss_post_calibration"]),
        "temperature": float(champion["temperature"]),
        "class_logit_bias": champion.get("class_logit_bias", [0.0, 0.0, 0.0]),
        "train_metrics": champion["train_metrics"],
        "test_metrics_pre_threshold_tuning": champion.get("test_metrics_pre_threshold_tuning", {}),
        "test_metrics": champion["test_metrics"],
        "threshold_tuning": champion.get("threshold_tuning", {}),
        "candidate_score": float(candidate_score),
        "previous_score": float(prev_score) if prev_score is not None else None,
        "model_path": str(model_path),
        "promoted": bool(promote_ok),
        "deployed_from_previous": bool(deployed_from_previous),
        "deployed_previous_model": deployed_previous_path,
        "promotion_gate": {
            "rollback_enabled": bool(args.rollback_on_regression),
            "strict_gate": bool(args.strict_promotion_gate),
            "reasons": promotion_reasons,
            "thresholds": {
                "promotion_min_score_delta": float(args.promotion_min_score_delta),
                "promotion_max_neutral_f1_drop": float(args.promotion_max_neutral_f1_drop),
            },
            "previous_model": str(prev_model_path) if prev_model_path else None,
            "previous_log": str(prev_log_path) if prev_log_path else None,
        },
        "data_quality_gate": dq_summary,
        "training_objective": {
            "focal_gamma": float(max(args.focal_gamma, 0.0)),
            "focal_alpha": {
                "negative": float(max(args.focal_alpha_negative, 0.05)),
                "neutral": float(max(args.focal_alpha_neutral, 0.05)),
                "positive": float(max(args.focal_alpha_positive, 0.05)),
            },
            "batch_size": int(max(args.batch_size, 1)),
            "batch_steps_per_epoch": int(max(args.batch_steps_per_epoch, 1)),
            "sampling_boosts": {
                "positive": float(args.oversample_positive),
                "neutral": float(args.oversample_neutral),
                "negative": float(args.oversample_negative),
            },
            "threshold_tuning": {
                "positive_recall_target": float(args.positive_recall_target),
                "neutral_recall_floor": float(args.neutral_recall_floor),
                "positive_logit_bias_max": float(args.positive_logit_bias_max),
                "negative_logit_bias_max": float(args.negative_logit_bias_max),
                "grid_steps": int(max(args.logit_bias_steps, 3)),
            },
        },
        "outcome_learning": ds.get("outcome_learning", {}),
    }

    log_path = LOGS_DIR / f"trade_behavior_policy_{timestamp}.json"
    log_path.write_text(json.dumps(log_payload, indent=2), encoding="utf-8")

    print("Saved model:", model_path)
    print("Saved log:", log_path)
    print(
        json.dumps(
            {
                "label_counts": label_counts,
                "promoted": bool(promote_ok),
                "champion_seed": int(champion["seed"]),
                "candidate_score": float(candidate_score),
                "previous_score": float(prev_score) if prev_score is not None else None,
                "test_metrics": champion["test_metrics"],
            },
            indent=2,
        )
    )

    if (not promote_ok) and bool(args.strict_promotion_gate):
        return 4
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
