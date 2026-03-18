import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
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


def _sha256_file(path: Path) -> str:
    if (not path) or (not path.exists()):
        return ""
    h = hashlib.sha256()
    try:
        with open(path, "rb") as f:
            while True:
                chunk = f.read(1024 * 1024)
                if not chunk:
                    break
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return ""


def _sha256_json_obj(obj: Any) -> str:
    try:
        encoded = json.dumps(obj, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()
    except Exception:
        return ""


def _git_commit(project_root: Path) -> str:
    try:
        proc = subprocess.run(
            ["git", "-C", str(project_root), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode == 0:
            return str(proc.stdout or "").strip()
        return ""
    except Exception:
        return ""


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
    macro = float(metrics.get("macro_f1", 0.0) or 0.0)
    balanced = float(metrics.get("balanced_accuracy", 0.0) or 0.0)
    pos_recall = float(metrics.get("positive_recall", 0.0) or 0.0)
    pos_precision = float(metrics.get("positive_precision", 0.0) or 0.0)
    pos_f1 = float(metrics.get("positive_f1", 0.0) or 0.0)
    gap = max(pos_recall - pos_precision, 0.0)
    return (
        macro
        + balanced
        + (0.35 * pos_f1)
        + (0.45 * pos_precision)
        + (0.20 * pos_recall)
        - (0.75 * max(gap - 0.22, 0.0))
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


def _curated_dataset_guard(ds: Dict[str, Any]) -> Tuple[bool, str, Dict[str, Any]]:
    source = ds.get("source") if isinstance(ds.get("source"), dict) else {}
    dataset_kind = str(ds.get("dataset_kind") or "").strip().lower()
    decision_sources = int(source.get("decision_files", 0) or 0) + int(source.get("decision_sql_files", 0) or 0)
    governance_sources = int(source.get("governance_files", 0) or 0) + int(source.get("governance_sql_files", 0) or 0)
    pnl_sources = int(source.get("pnl_attribution_files", 0) or 0) + int(source.get("pnl_sql_files", 0) or 0)

    summary = {
        "dataset_kind": dataset_kind,
        "decision_sources": int(decision_sources),
        "governance_sources": int(governance_sources),
        "pnl_sources": int(pnl_sources),
    }
    if dataset_kind != "curated_decision_governance":
        return False, "dataset_kind_not_curated", summary
    if decision_sources <= 0:
        return False, "decision_sources_missing", summary
    if governance_sources <= 0:
        return False, "governance_sources_missing", summary
    return True, "ok", summary


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


def _derive_validation_split(
    ts_epoch: np.ndarray,
    train_idx: np.ndarray,
    *,
    val_ratio: float,
    mode: str,
    purge_seconds: float,
    seed: int,
    min_train_rows: int,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    train_idx = np.asarray(train_idx, dtype=np.int64)
    n_rows = int(len(train_idx))
    min_train_rows = max(int(min_train_rows), 5)
    if n_rows < (min_train_rows + 2):
        raise RuntimeError(f"Not enough rows for validation split: {n_rows}")

    val_ratio = max(min(float(val_ratio), 0.40), 0.05)
    val_n = max(1, int(n_rows * val_ratio))
    if n_rows - val_n < min_train_rows:
        val_n = max(1, n_rows - min_train_rows)

    if mode == "random":
        rng = np.random.default_rng(seed)
        order = train_idx.copy()
        rng.shuffle(order)
        val_idx = order[:val_n]
        tr_idx = order[val_n:]
        return tr_idx, val_idx, {
            "mode": "random",
            "seed": int(seed),
            "validation_rows": int(len(val_idx)),
            "train_rows": int(len(tr_idx)),
        }

    order = train_idx[np.argsort(ts_epoch[train_idx], kind="mergesort")]
    val_idx = order[-val_n:]
    tr_raw = order[:-val_n]

    val_start_ts = float(ts_epoch[val_idx[0]]) if len(val_idx) else float(np.max(ts_epoch[order]))
    purge_seconds = max(float(purge_seconds), 0.0)
    purge_cutoff = val_start_ts - purge_seconds
    tr_idx = tr_raw[ts_epoch[tr_raw] <= purge_cutoff]
    used_purge_seconds = purge_seconds
    relaxed = False

    if len(tr_idx) < min_train_rows:
        tr_idx = tr_raw
        used_purge_seconds = 0.0
        relaxed = True

    if len(tr_idx) < min_train_rows:
        rng = np.random.default_rng(seed)
        fallback = train_idx.copy()
        rng.shuffle(fallback)
        val_idx = fallback[:val_n]
        tr_idx = fallback[val_n:]
        return tr_idx, val_idx, {
            "mode": "random",
            "seed": int(seed),
            "validation_rows": int(len(val_idx)),
            "train_rows": int(len(tr_idx)),
            "fallback_from": "time_purged",
            "fallback_reason": "insufficient_train_rows",
        }

    return tr_idx, val_idx, {
        "mode": "time_purged",
        "validation_rows": int(len(val_idx)),
        "train_rows": int(len(tr_idx)),
        "purge_seconds": float(used_purge_seconds),
        "relaxed_purge": bool(relaxed),
        "validation_start_ts": float(val_start_ts),
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
    positive_precision_floor: float,
    max_pos_recall_precision_gap: float,
    score_weight_positive_precision: float,
    score_gap_penalty: float,
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
    positive_precision_floor = max(min(float(positive_precision_floor), 1.0), 0.0)
    max_pos_recall_precision_gap = max(float(max_pos_recall_precision_gap), 0.0)
    score_weight_positive_precision = max(float(score_weight_positive_precision), 0.0)
    score_gap_penalty = max(float(score_gap_penalty), 0.0)

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

            pos_recall = float(metrics.get("positive_recall", 0.0) or 0.0)
            pos_precision = float(metrics.get("positive_precision", 0.0) or 0.0)
            neu_recall = float(metrics.get("neutral_recall", 0.0) or 0.0)
            pos_gap = max(pos_recall - pos_precision, 0.0)

            score = _metric_score(metrics)
            score += score_weight_positive_precision * pos_precision
            if pos_recall < positive_recall_target:
                score -= 2.0 * (positive_recall_target - pos_recall)
            if neu_recall < neutral_recall_floor:
                score -= 1.5 * (neutral_recall_floor - neu_recall)
            if pos_precision < positive_precision_floor:
                score -= 2.5 * (positive_precision_floor - pos_precision)
            if pos_gap > max_pos_recall_precision_gap:
                score -= score_gap_penalty * (pos_gap - max_pos_recall_precision_gap)

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
        "positive_precision_floor": float(positive_precision_floor),
        "max_pos_recall_precision_gap": float(max_pos_recall_precision_gap),
        "score_weight_positive_precision": float(score_weight_positive_precision),
        "score_gap_penalty": float(score_gap_penalty),
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
    x_val: np.ndarray,
    y_val: np.ndarray,
    w_val: np.ndarray,
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
    positive_precision_floor: float,
    max_pos_recall_precision_gap: float,
    threshold_score_weight_positive_precision: float,
    threshold_score_gap_penalty: float,
    pos_logit_bias_max: float,
    neg_logit_bias_max: float,
    logit_bias_steps: int,
    early_stop_patience: int,
    early_stop_min_delta: float,
    early_stop_min_epochs: int,
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

    early_stop_patience = max(int(early_stop_patience), 1)
    early_stop_min_epochs = max(int(early_stop_min_epochs), 1)
    early_stop_min_delta = max(float(early_stop_min_delta), 0.0)

    best = {
        "epoch": -1,
        "val_loss": math.inf,
        "tr_loss": math.inf,
        "checkpoint_score": -math.inf,
        "W": None,
        "b": None,
    }
    no_improve = 0
    ran_epochs = 0
    early_stopped = False

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
        val_logits = x_val @ W + b
        tr_probs = _softmax(tr_logits)
        val_probs = _softmax(val_logits)

        tr_loss = _weighted_focal_ce(
            tr_probs,
            y_tr,
            w_tr,
            gamma=focal_gamma,
            class_alpha=focal_class_alpha,
        )
        val_loss = _weighted_focal_ce(
            val_probs,
            y_val,
            w_val,
            gamma=focal_gamma,
            class_alpha=focal_class_alpha,
        )

        val_metrics_epoch = _metrics(val_probs, y_val)
        ckpt_score = _checkpoint_score(val_metrics_epoch)
        improved = ckpt_score > (best["checkpoint_score"] + early_stop_min_delta)
        tie = abs(ckpt_score - best["checkpoint_score"]) <= early_stop_min_delta
        if (not improved) and tie and val_loss < (best["val_loss"] - 1e-6):
            improved = True

        if improved:
            best = {
                "epoch": epoch,
                "val_loss": float(val_loss),
                "tr_loss": float(tr_loss),
                "checkpoint_score": float(ckpt_score),
                "W": W.copy(),
                "b": b.copy(),
            }
            no_improve = 0
        else:
            no_improve += 1

        ran_epochs = epoch + 1
        if epoch % 50 == 0 or epoch == (max(int(epochs), 1) - 1):
            val_acc = float(np.mean(np.argmax(val_probs, axis=1) == y_val))
            val_macro = float(val_metrics_epoch.get("macro_f1", 0.0) or 0.0)
            val_bal = float(val_metrics_epoch.get("balanced_accuracy", 0.0) or 0.0)
            val_pos_rec = float(val_metrics_epoch.get("positive_recall", 0.0) or 0.0)
            val_pos_prec = float(val_metrics_epoch.get("positive_precision", 0.0) or 0.0)
            print(
                f"Seed {seed} | Epoch {epoch:03d} | TrainLoss {tr_loss:.4f} | ValLoss {val_loss:.4f} "
                f"| ValAcc {val_acc:.4f} | MacroF1 {val_macro:.4f} | BalAcc {val_bal:.4f} "
                f"| PosRec {val_pos_rec:.4f} | PosPrec {val_pos_prec:.4f}"
            )

        if ran_epochs >= early_stop_min_epochs and no_improve >= early_stop_patience:
            early_stopped = True
            break

    W_best = best["W"] if best["W"] is not None else W
    b_best = best["b"] if best["b"] is not None else b

    tr_logits_best = x_tr @ W_best + b_best
    val_logits_best = x_val @ W_best + b_best
    te_logits_best = x_te @ W_best + b_best

    temp, val_loss_pre_cal, val_loss_post_cal = _fit_temperature(
        val_logits_best,
        y_val,
        w_val,
        t_min=temperature_min,
        t_max=temperature_max,
        steps=temperature_steps,
    )

    tr_logits_cal = tr_logits_best / temp
    val_logits_cal = val_logits_best / temp
    te_logits_cal = te_logits_best / temp

    class_logit_bias, val_metrics_pre_threshold, val_metrics_tuned_threshold, threshold_tuning = _fit_class_logit_bias(
        val_logits_cal,
        y_val,
        w_val,
        positive_recall_target=positive_recall_target,
        neutral_recall_floor=neutral_recall_floor,
        positive_precision_floor=positive_precision_floor,
        max_pos_recall_precision_gap=max_pos_recall_precision_gap,
        score_weight_positive_precision=threshold_score_weight_positive_precision,
        score_gap_penalty=threshold_score_gap_penalty,
        pos_bias_min=-abs(float(pos_logit_bias_max)),
        pos_bias_max=abs(float(pos_logit_bias_max)),
        neg_bias_min=-abs(float(neg_logit_bias_max)),
        neg_bias_max=abs(float(neg_logit_bias_max)),
        steps=logit_bias_steps,
    )

    bias_row = class_logit_bias.reshape(1, -1)
    tr_probs = _softmax(tr_logits_cal + bias_row)
    val_probs = _softmax(val_logits_cal + bias_row)
    te_probs = _softmax(te_logits_cal + bias_row)
    te_probs_pre_threshold = _softmax(te_logits_cal)

    tr_metrics = _metrics(tr_probs, y_tr)
    val_metrics = _metrics(val_probs, y_val)
    te_metrics_pre_threshold = _metrics(te_probs_pre_threshold, y_te)
    te_metrics = _metrics(te_probs, y_te)

    return {
        "seed": int(seed),
        "best_epoch": int(best["epoch"]),
        "epochs_ran": int(ran_epochs),
        "early_stopped": bool(early_stopped),
        "best_train_loss": float(best["tr_loss"]),
        "best_validation_loss": float(best["val_loss"]),
        "best_checkpoint_score": float(best["checkpoint_score"]),
        "temperature": float(temp),
        "class_logit_bias": [float(x) for x in class_logit_bias.tolist()],
        "validation_loss_pre_calibration": float(val_loss_pre_cal),
        "validation_loss_post_calibration": float(val_loss_post_cal),
        "train_metrics": tr_metrics,
        "validation_metrics_pre_threshold_tuning": val_metrics_pre_threshold,
        "validation_metrics_threshold_tuned": val_metrics_tuned_threshold,
        "validation_metrics": val_metrics,
        "test_metrics_pre_threshold_tuning": te_metrics_pre_threshold,
        "test_metrics": te_metrics,
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


def _model_feature_names_from_npz(arr: Any) -> List[str]:
    try:
        files = set(arr.files)
    except Exception:
        files = set()
    if "feature_names" not in files:
        return []

    raw = np.asarray(arr["feature_names"]).reshape(-1)
    out: List[str] = []
    for value in raw.tolist():
        if isinstance(value, bytes):
            name = value.decode("utf-8", errors="ignore").strip()
        else:
            name = str(value).strip()
        if name:
            out.append(name)
    return out


def _model_schema_summary(path: Optional[Path]) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "path": str(path) if path else "",
        "exists": bool(path and path.exists()),
        "load_ok": False,
        "feature_dim_field": 0,
        "effective_dim": 0,
        "mu_dim": 0,
        "sigma_dim": 0,
        "feature_names_count": 0,
        "has_feature_names": False,
        "feature_names_sha256": "",
        "error": "",
    }
    if (path is None) or (not path.exists()):
        return out

    try:
        arr = np.load(path, allow_pickle=False)
        w = np.asarray(arr["W"])
        mu = np.asarray(arr["mu"]).reshape(-1)
        sigma = np.asarray(arr["sigma"]).reshape(-1)

        model_dim = int(w.shape[0]) if w.ndim == 2 else 0
        mu_dim = int(mu.shape[0])
        sigma_dim = int(sigma.shape[0])
        effective_dim = min(model_dim, mu_dim, sigma_dim)

        feature_dim_field = 0
        if "feature_dim" in arr.files:
            try:
                feature_dim_field = int(np.asarray(arr["feature_dim"]).reshape(-1)[0])
            except Exception:
                feature_dim_field = 0

        feature_names = _model_feature_names_from_npz(arr)
        feature_names_sha = _sha256_json_obj(feature_names) if feature_names else ""

        out.update(
            {
                "load_ok": True,
                "feature_dim_field": int(feature_dim_field),
                "effective_dim": int(effective_dim),
                "mu_dim": int(mu_dim),
                "sigma_dim": int(sigma_dim),
                "feature_names_count": int(len(feature_names)),
                "has_feature_names": bool(feature_names),
                "feature_names_sha256": feature_names_sha,
                "feature_names": feature_names,
            }
        )
    except Exception as exc:
        out["error"] = str(exc)
    return out


def _rollback_schema_compatible(
    prev_schema: Dict[str, Any],
    *,
    dataset_feature_dim: int,
    dataset_feature_names: List[str],
    require_feature_names: bool,
) -> Tuple[bool, str]:
    if not bool(prev_schema.get("load_ok", False)):
        return False, "previous_model_schema_unreadable"

    prev_dim = int(prev_schema.get("effective_dim", 0) or 0)
    if prev_dim <= 0:
        return False, "previous_model_dim_invalid"
    dataset_dim = int(dataset_feature_dim)
    if prev_dim > dataset_dim:
        return False, f"feature_dim_mismatch prev={prev_dim} dataset={int(dataset_feature_dim)}"

    if not require_feature_names:
        return True, "ok"

    prev_names = [str(x) for x in (prev_schema.get("feature_names") or []) if str(x)]
    if len(prev_names) != prev_dim:
        return False, "previous_model_missing_feature_names"
    if dataset_feature_names:
        expected_prefix = dataset_feature_names[:prev_dim]
        if prev_names != expected_prefix:
            return False, "feature_name_order_mismatch"
    if prev_dim == dataset_dim:
        return True, "ok"
    return True, f"prefix_compatible prev={prev_dim} dataset={dataset_dim}"


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


def _snapshot_training_coverage_summary(
    *,
    dataset_path: Path,
    dataset_obj: Dict[str, Any],
) -> Dict[str, Any]:
    snapshot_context = dataset_obj.get("snapshot_context") if isinstance(dataset_obj.get("snapshot_context"), dict) else {}
    snapshot_features = snapshot_context.get("features") if isinstance(snapshot_context.get("features"), dict) else {}
    snapshot_meta = snapshot_context.get("meta") if isinstance(snapshot_context.get("meta"), dict) else {}

    feature_names_raw = dataset_obj.get("feature_names") if isinstance(dataset_obj.get("feature_names"), list) else []
    feature_names = [str(name) for name in feature_names_raw if str(name)]

    snapshot_feature_names_from_context = sorted(
        key for key in snapshot_features.keys() if str(key).startswith("snapshot_")
    )
    snapshot_feature_names_in_model = sorted(
        {name for name in feature_names if name.startswith("snapshot_")}
    )
    missing_snapshot_features = [
        key for key in snapshot_feature_names_from_context if key not in snapshot_feature_names_in_model
    ]

    total_snapshot_features = len(snapshot_feature_names_from_context)
    present_snapshot_features = total_snapshot_features - len(missing_snapshot_features)
    feature_coverage_ratio = (
        float(present_snapshot_features) / float(total_snapshot_features)
        if total_snapshot_features > 0
        else 1.0
    )

    raw_debug_context = (
        snapshot_meta.get("raw_debug_context")
        if isinstance(snapshot_meta.get("raw_debug_context"), dict)
        else {}
    )

    def _ratio(value: Any, default: float = 0.0) -> float:
        try:
            out = float(value)
        except Exception:
            out = float(default)
        if not math.isfinite(out):
            out = float(default)
        return max(0.0, min(out, 1.0))

    required_ratio = _ratio(os.getenv("SNAPSHOT_TRAINING_REQUIRED_RATIO", "1.0"), 1.0)
    ratio_epsilon = max(min(float(os.getenv("SNAPSHOT_TRAINING_RATIO_EPSILON", "1e-6")), 0.01), 0.0)

    snapshot_raw_sql_ingest_ratio = _ratio(
        snapshot_features.get(
            "snapshot_raw_sql_ingest_ratio",
            raw_debug_context.get("ingest_coverage_ratio", 0.0),
        ),
        0.0,
    )
    snapshot_cov_fill_ratio = _ratio(snapshot_features.get("snapshot_cov_fill_ratio", 0.0), 0.0)
    snapshot_replay_ok = _ratio(snapshot_features.get("snapshot_replay_ok", 0.0), 0.0)

    meets_raw_ratio = snapshot_raw_sql_ingest_ratio + ratio_epsilon >= required_ratio
    meets_fill_ratio = snapshot_cov_fill_ratio + ratio_epsilon >= required_ratio
    meets_feature_coverage = feature_coverage_ratio + ratio_epsilon >= required_ratio

    rows = int(dataset_obj.get("rows", 0) or 0)
    all_snapshot_data_incorporated = bool(
        rows > 0
        and meets_raw_ratio
        and meets_fill_ratio
        and meets_feature_coverage
        and (len(missing_snapshot_features) == 0)
    )

    if rows <= 0:
        reason = "dataset_rows_zero"
    elif len(missing_snapshot_features) > 0:
        reason = "snapshot_feature_names_missing"
    elif not meets_raw_ratio:
        reason = "snapshot_raw_sql_ingest_ratio_below_required"
    elif not meets_fill_ratio:
        reason = "snapshot_cov_fill_ratio_below_required"
    elif not meets_feature_coverage:
        reason = "snapshot_feature_coverage_ratio_below_required"
    else:
        reason = "ok"

    return {
        "dataset_path": str(dataset_path),
        "dataset_timestamp_utc": dataset_obj.get("timestamp_utc"),
        "dataset_rows": rows,
        "feature_dim": int(dataset_obj.get("_feature_dim", 0) or 0),
        "feature_names_count": len(feature_names),
        "snapshot_context_feature_count": total_snapshot_features,
        "snapshot_feature_names_in_model_count": len(snapshot_feature_names_in_model),
        "snapshot_feature_coverage_ratio": float(feature_coverage_ratio),
        "missing_snapshot_feature_names": missing_snapshot_features,
        "snapshot_raw_sql_ingest_ratio": float(snapshot_raw_sql_ingest_ratio),
        "snapshot_cov_fill_ratio": float(snapshot_cov_fill_ratio),
        "snapshot_replay_ok": float(snapshot_replay_ok),
        "required_ratio": float(required_ratio),
        "ratio_epsilon": float(ratio_epsilon),
        "all_snapshot_data_incorporated": bool(all_snapshot_data_incorporated),
        "reason": reason,
    }


def _write_snapshot_training_coverage_artifact(
    *,
    project_root: Path,
    dataset_path: Path,
    dataset_obj: Dict[str, Any],
    model_path: Path,
    log_path: Path,
) -> Dict[str, Any]:
    payload = _snapshot_training_coverage_summary(dataset_path=dataset_path, dataset_obj=dataset_obj)
    payload.update(
        {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "model_path": str(model_path),
            "training_log": str(log_path),
        }
    )

    out_path = project_root / "governance" / "health" / "snapshot_training_coverage_latest.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")

    payload["artifact_path"] = str(out_path)
    return payload


def _cap_snapshot_feature_weights(
    W: np.ndarray,
    feature_names: List[str],
    *,
    max_abs_weight: float,
) -> Dict[str, Any]:
    cap = max(float(max_abs_weight), 0.0)
    if cap <= 0.0:
        return {
            "enabled": False,
            "max_abs_weight": 0.0,
            "snapshot_feature_count": 0,
            "weights_capped": 0,
        }

    if W.ndim != 2 or W.shape[0] <= 0:
        return {
            "enabled": True,
            "max_abs_weight": cap,
            "snapshot_feature_count": 0,
            "weights_capped": 0,
        }

    capped = 0
    snapshot_idx: List[int] = []
    for i, name in enumerate(feature_names[: W.shape[0]]):
        if str(name).startswith("snapshot_"):
            snapshot_idx.append(i)

    for idx in snapshot_idx:
        row = W[idx, :]
        before = row.copy()
        np.clip(row, -cap, cap, out=row)
        capped += int(np.sum(before != row))

    return {
        "enabled": True,
        "max_abs_weight": cap,
        "snapshot_feature_count": int(len(snapshot_idx)),
        "weights_capped": int(capped),
    }


def _save_model(
    path: Path,
    *,
    W: np.ndarray,
    b: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
    feature_dim: int,
    feature_names: Optional[List[str]] = None,
    temperature: float,
    class_logit_bias: Optional[np.ndarray] = None,
) -> None:
    bias = np.zeros((len(LABEL_TO_ID),), dtype=np.float32)
    if class_logit_bias is not None:
        raw = np.asarray(class_logit_bias, dtype=np.float64).reshape(-1)
        if raw.shape[0] == len(LABEL_TO_ID):
            bias = raw.astype(np.float32)

    payload = {
        "W": W.astype(np.float32),
        "b": b.astype(np.float32),
        "mu": mu.astype(np.float32),
        "sigma": sigma.astype(np.float32),
        "labels": np.asarray(["negative", "neutral", "positive"]),
        "feature_dim": np.asarray([int(feature_dim)], dtype=np.int32),
        "temperature": np.asarray([float(temperature)], dtype=np.float32),
        "class_logit_bias": bias,
    }

    names = [str(name) for name in (feature_names or []) if str(name)]
    if names:
        payload["feature_names"] = np.asarray(names[: int(feature_dim)], dtype="<U128")

    np.savez_compressed(path, **payload)


def main() -> int:
    parser = argparse.ArgumentParser(description="Train a weighted behavior policy model on past trades.")
    parser.add_argument("--dataset", default=str(DATASET_PATH))
    parser.add_argument("--epochs", type=int, default=int(os.getenv("TRADE_BEHAVIOR_EPOCHS", "300")))
    parser.add_argument("--lr", type=float, default=float(os.getenv("TRADE_BEHAVIOR_LR", "0.02")))
    parser.add_argument("--l2", type=float, default=float(os.getenv("TRADE_BEHAVIOR_L2", "1e-4")))
    parser.add_argument("--test-ratio", type=float, default=float(os.getenv("TRADE_BEHAVIOR_TEST_RATIO", "0.2")))
    parser.add_argument("--val-ratio", type=float, default=float(os.getenv("TRADE_BEHAVIOR_VAL_RATIO", "0.15")))
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
    parser.add_argument("--batch-size", type=int, default=int(os.getenv("TRADE_BEHAVIOR_BATCH_SIZE", "1536")))
    parser.add_argument("--batch-steps-per-epoch", type=int, default=int(os.getenv("TRADE_BEHAVIOR_BATCH_STEPS_PER_EPOCH", "2")))
    parser.add_argument("--oversample-positive", type=float, default=float(os.getenv("TRADE_BEHAVIOR_OVERSAMPLE_POSITIVE", "1.35")))
    parser.add_argument("--oversample-neutral", type=float, default=float(os.getenv("TRADE_BEHAVIOR_OVERSAMPLE_NEUTRAL", "1.15")))
    parser.add_argument("--oversample-negative", type=float, default=float(os.getenv("TRADE_BEHAVIOR_OVERSAMPLE_NEGATIVE", "1.00")))
    parser.add_argument("--focal-gamma", type=float, default=float(os.getenv("TRADE_BEHAVIOR_FOCAL_GAMMA", "0.55")))
    parser.add_argument("--focal-alpha-negative", type=float, default=float(os.getenv("TRADE_BEHAVIOR_FOCAL_ALPHA_NEGATIVE", "1.0")))
    parser.add_argument("--focal-alpha-neutral", type=float, default=float(os.getenv("TRADE_BEHAVIOR_FOCAL_ALPHA_NEUTRAL", "1.05")))
    parser.add_argument("--focal-alpha-positive", type=float, default=float(os.getenv("TRADE_BEHAVIOR_FOCAL_ALPHA_POSITIVE", "1.25")))
    parser.add_argument("--temperature-min", type=float, default=float(os.getenv("TRADE_BEHAVIOR_TEMPERATURE_MIN", "0.6")))
    parser.add_argument("--temperature-max", type=float, default=float(os.getenv("TRADE_BEHAVIOR_TEMPERATURE_MAX", "2.5")))
    parser.add_argument("--temperature-steps", type=int, default=int(os.getenv("TRADE_BEHAVIOR_TEMPERATURE_STEPS", "45")))
    parser.add_argument("--positive-recall-target", type=float, default=float(os.getenv("TRADE_BEHAVIOR_POSITIVE_RECALL_TARGET", "0.24")))
    parser.add_argument("--neutral-recall-floor", type=float, default=float(os.getenv("TRADE_BEHAVIOR_NEUTRAL_RECALL_FLOOR", "0.24")))
    parser.add_argument("--positive-logit-bias-max", type=float, default=float(os.getenv("TRADE_BEHAVIOR_POSITIVE_LOGIT_BIAS_MAX", "0.55")))
    parser.add_argument("--negative-logit-bias-max", type=float, default=float(os.getenv("TRADE_BEHAVIOR_NEGATIVE_LOGIT_BIAS_MAX", "0.35")))
    parser.add_argument("--logit-bias-steps", type=int, default=int(os.getenv("TRADE_BEHAVIOR_LOGIT_BIAS_STEPS", "21")))
    parser.add_argument("--threshold-positive-precision-floor", type=float, default=float(os.getenv("TRADE_BEHAVIOR_THRESHOLD_POS_PREC_FLOOR", "0.19")))
    parser.add_argument("--threshold-max-pos-recall-precision-gap", type=float, default=float(os.getenv("TRADE_BEHAVIOR_THRESHOLD_MAX_POS_REC_PREC_GAP", "0.25")))
    parser.add_argument("--threshold-score-weight-positive-precision", type=float, default=float(os.getenv("TRADE_BEHAVIOR_THRESHOLD_SCORE_WEIGHT_POS_PREC", "0.80")))
    parser.add_argument("--threshold-score-gap-penalty", type=float, default=float(os.getenv("TRADE_BEHAVIOR_THRESHOLD_SCORE_GAP_PENALTY", "0.90")))
    parser.add_argument("--early-stop-patience", type=int, default=int(os.getenv("TRADE_BEHAVIOR_EARLY_STOP_PATIENCE", "35")))
    parser.add_argument("--early-stop-min-delta", type=float, default=float(os.getenv("TRADE_BEHAVIOR_EARLY_STOP_MIN_DELTA", "0.0005")))
    parser.add_argument("--early-stop-min-epochs", type=int, default=int(os.getenv("TRADE_BEHAVIOR_EARLY_STOP_MIN_EPOCHS", "40")))
    parser.add_argument("--promotion-min-score-delta", type=float, default=float(os.getenv("TRADE_BEHAVIOR_PROMOTION_MIN_SCORE_DELTA", "-0.01")))
    parser.add_argument("--promotion-max-neutral-f1-drop", type=float, default=float(os.getenv("TRADE_BEHAVIOR_PROMOTION_MAX_NEUTRAL_F1_DROP", "0.03")))
    parser.add_argument("--promotion-min-accuracy", type=float, default=float(os.getenv("TRADE_BEHAVIOR_PROMOTION_MIN_ACCURACY", "0.38")))
    parser.add_argument("--promotion-min-macro-f1", type=float, default=float(os.getenv("TRADE_BEHAVIOR_PROMOTION_MIN_MACRO_F1", "0.33")))
    parser.add_argument("--promotion-min-balanced-accuracy", type=float, default=float(os.getenv("TRADE_BEHAVIOR_PROMOTION_MIN_BALANCED_ACCURACY", "0.36")))
    parser.add_argument("--promotion-min-positive-precision", type=float, default=float(os.getenv("TRADE_BEHAVIOR_PROMOTION_MIN_POSITIVE_PRECISION", "0.175")))
    parser.add_argument("--promotion-max-pos-recall-precision-gap", type=float, default=float(os.getenv("TRADE_BEHAVIOR_PROMOTION_MAX_POS_RECALL_PREC_GAP", "0.28")))
    parser.add_argument("--rollback-on-regression", action=argparse.BooleanOptionalAction, default=_parse_bool(os.getenv("TRADE_BEHAVIOR_ROLLBACK_ON_REGRESSION", "1"), default=True))
    parser.add_argument(
        "--rollback-require-schema-match",
        action=argparse.BooleanOptionalAction,
        default=_parse_bool(os.getenv("TRADE_BEHAVIOR_ROLLBACK_REQUIRE_SCHEMA_MATCH", "1"), default=True),
    )
    parser.add_argument(
        "--require-feature-names",
        action=argparse.BooleanOptionalAction,
        default=_parse_bool(os.getenv("TRADE_BEHAVIOR_REQUIRE_FEATURE_NAMES", "1"), default=True),
    )
    parser.add_argument(
        "--require-curated-dataset",
        action=argparse.BooleanOptionalAction,
        default=_parse_bool(os.getenv("TRADE_BEHAVIOR_REQUIRE_CURATED_DATASET", "1"), default=True),
    )
    parser.add_argument("--strict-promotion-gate", action=argparse.BooleanOptionalAction, default=_parse_bool(os.getenv("TRADE_BEHAVIOR_STRICT_PROMOTION_GATE", "0"), default=False))
    parser.add_argument("--require-walk-forward-ok", action=argparse.BooleanOptionalAction, default=_parse_bool(os.getenv("TRADE_BEHAVIOR_REQUIRE_WALK_FORWARD_OK", "0"), default=False))
    parser.add_argument("--max-abs-snapshot-weight", type=float, default=float(os.getenv("TRADE_BEHAVIOR_MAX_ABS_SNAPSHOT_WEIGHT", "1.25")))
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        print(f"Missing dataset: {dataset_path}")
        return 2

    X, y, w, ts_epoch, symbols, regimes, ds = _load_dataset(dataset_path)
    if len(y) < 20:
        print(f"Not enough rows to train behavior model: {len(y)}")
        return 2

    curated_ok, curated_reason, curated_summary = _curated_dataset_guard(ds)
    if args.require_curated_dataset and not curated_ok:
        print(f"Curated dataset guard failed: {curated_reason}")
        return 2

    if args.split_mode == "time_purged":
        train_outer_idx, test_idx, outer_split_meta = _time_purged_split_indices(
            ts_epoch=ts_epoch,
            test_ratio=args.test_ratio,
            purge_seconds=args.purge_seconds,
            min_train_rows=max(args.min_train_rows, 5),
            fallback_seed=args.seed,
        )
    else:
        train_outer_idx, test_idx, outer_split_meta = _random_split_indices(
            n_rows=len(y),
            test_ratio=args.test_ratio,
            seed=args.seed,
            min_train_rows=max(args.min_train_rows, 5),
        )

    if len(train_outer_idx) < 8:
        print("Not enough rows after outer split")
        return 2

    train_idx, val_idx, val_split_meta = _derive_validation_split(
        ts_epoch=ts_epoch,
        train_idx=train_outer_idx,
        val_ratio=args.val_ratio,
        mode=args.split_mode,
        purge_seconds=args.purge_seconds,
        seed=int(args.seed) + 17,
        min_train_rows=max(args.min_train_rows, 5),
    )

    if len(train_idx) < 5 or len(val_idx) < 1:
        print("Not enough rows after validation split")
        return 2

    x_train_raw = X[train_idx]
    mu = np.mean(x_train_raw, axis=0, keepdims=True)
    sigma = np.std(x_train_raw, axis=0, keepdims=True)
    sigma = np.where(sigma < 1e-8, 1.0, sigma)

    x_tr = (x_train_raw - mu) / sigma
    x_val = (X[val_idx] - mu) / sigma
    x_te = (X[test_idx] - mu) / sigma
    mu = mu.squeeze(0)
    sigma = sigma.squeeze(0)

    y_tr, y_val, y_te = y[train_idx], y[val_idx], y[test_idx]
    w_tr, w_val, w_te = w[train_idx], w[val_idx], w[test_idx]
    regime_tr = regimes[train_idx]

    split_meta = dict(outer_split_meta)
    split_meta["outer_train_rows"] = int(len(train_outer_idx))
    split_meta["outer_test_rows"] = int(len(test_idx))
    split_meta["validation"] = val_split_meta

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
            x_val=x_val,
            y_val=y_val,
            w_val=w_val,
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
            positive_precision_floor=float(args.threshold_positive_precision_floor),
            max_pos_recall_precision_gap=float(args.threshold_max_pos_recall_precision_gap),
            threshold_score_weight_positive_precision=float(args.threshold_score_weight_positive_precision),
            threshold_score_gap_penalty=float(args.threshold_score_gap_penalty),
            pos_logit_bias_max=float(args.positive_logit_bias_max),
            neg_logit_bias_max=float(args.negative_logit_bias_max),
            logit_bias_steps=max(int(args.logit_bias_steps), 3),
            early_stop_patience=max(int(args.early_stop_patience), 1),
            early_stop_min_delta=max(float(args.early_stop_min_delta), 0.0),
            early_stop_min_epochs=max(int(args.early_stop_min_epochs), 1),
        )
        seed_results.append(seed_result)

    seed_results.sort(key=lambda row: float(row.get("score", -1e9)), reverse=True)
    champion = seed_results[0]

    feature_names_raw = ds.get("feature_names") if isinstance(ds.get("feature_names"), list) else []
    feature_names = [str(name) for name in feature_names_raw if str(name)]
    dataset_feature_dim = int(X.shape[1]) if X.ndim == 2 else 0
    if dataset_feature_dim <= 0:
        print("Invalid dataset feature_dim<=0")
        return 2
    if args.require_feature_names and (not feature_names):
        print("Dataset missing feature_names while require_feature_names=true")
        return 2
    if feature_names and len(feature_names) != dataset_feature_dim:
        print(
            f"Dataset feature_names length mismatch: names={len(feature_names)} "
            f"feature_dim={dataset_feature_dim}"
        )
        return 2
    if not feature_names:
        feature_names = [f"feature_{i}" for i in range(dataset_feature_dim)]
    snapshot_weight_cap = _cap_snapshot_feature_weights(
        champion["W"],
        feature_names,
        max_abs_weight=float(args.max_abs_snapshot_weight),
    )

    prev_model_path, prev_log, prev_log_path = _latest_policy_model_and_log(MODELS_DIR, LOGS_DIR)
    prev_model_schema = _model_schema_summary(prev_model_path) if prev_model_path else {}
    rollback_schema_ok = True
    rollback_schema_reason = "disabled"
    if bool(args.rollback_require_schema_match):
        rollback_schema_ok, rollback_schema_reason = _rollback_schema_compatible(
            prev_model_schema,
            dataset_feature_dim=dataset_feature_dim,
            dataset_feature_names=feature_names,
            require_feature_names=bool(args.require_feature_names),
        )
    prev_metrics = (prev_log.get("test_metrics") or {}) if isinstance(prev_log, dict) else {}

    candidate_metrics = champion.get("test_metrics") or {}
    candidate_score = _metric_score(candidate_metrics)
    candidate_accuracy = float(candidate_metrics.get("accuracy", 0.0) or 0.0)
    candidate_macro_f1 = float(candidate_metrics.get("macro_f1", 0.0) or 0.0)
    candidate_balanced_accuracy = float(candidate_metrics.get("balanced_accuracy", 0.0) or 0.0)
    candidate_positive_precision = float(candidate_metrics.get("positive_precision", 0.0) or 0.0)
    candidate_positive_recall = float(candidate_metrics.get("positive_recall", 0.0) or 0.0)
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

    if candidate_accuracy < float(args.promotion_min_accuracy):
        promote_ok = False
        promotion_reasons.append(
            f"candidate_accuracy={candidate_accuracy:.4f} < min_accuracy={float(args.promotion_min_accuracy):.4f}"
        )
    if candidate_macro_f1 < float(args.promotion_min_macro_f1):
        promote_ok = False
        promotion_reasons.append(
            f"candidate_macro_f1={candidate_macro_f1:.4f} < min_macro_f1={float(args.promotion_min_macro_f1):.4f}"
        )
    if candidate_balanced_accuracy < float(args.promotion_min_balanced_accuracy):
        promote_ok = False
        promotion_reasons.append(
            f"candidate_balanced_accuracy={candidate_balanced_accuracy:.4f} < min_balanced_accuracy={float(args.promotion_min_balanced_accuracy):.4f}"
        )
    if candidate_positive_precision < float(args.promotion_min_positive_precision):
        promote_ok = False
        promotion_reasons.append(
            f"candidate_positive_precision={candidate_positive_precision:.4f} < min_positive_precision={float(args.promotion_min_positive_precision):.4f}"
        )

    pos_recall_precision_gap = candidate_positive_recall - candidate_positive_precision
    if pos_recall_precision_gap > float(args.promotion_max_pos_recall_precision_gap):
        promote_ok = False
        promotion_reasons.append(
            f"pos_recall_precision_gap={pos_recall_precision_gap:.4f} > max_gap={float(args.promotion_max_pos_recall_precision_gap):.4f}"
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
            feature_dim=dataset_feature_dim,
            feature_names=feature_names,
            temperature=float(champion.get("temperature", 1.0) or 1.0),
            class_logit_bias=np.asarray(champion.get("class_logit_bias", [0.0, 0.0, 0.0]), dtype=np.float64),
        )
    else:
        if prev_model_path is not None and prev_model_path.exists() and (rollback_schema_ok or (not bool(args.rollback_require_schema_match))):
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
                feature_dim=dataset_feature_dim,
                feature_names=feature_names,
                temperature=float(champion.get("temperature", 1.0) or 1.0),
                class_logit_bias=np.asarray(champion.get("class_logit_bias", [0.0, 0.0, 0.0]), dtype=np.float64),
            )
            if prev_model_path is not None and prev_model_path.exists():
                promotion_reasons.append(f"rollback_blocked_schema_guard:{rollback_schema_reason}")
            else:
                promotion_reasons.append("rollback_target_missing_deployed_candidate")

    label_counts = {name: int(np.sum(y == cid)) for name, cid in LABEL_TO_ID.items()}

    dataset_lineage = ds.get("lineage", {}) if isinstance(ds.get("lineage"), dict) else {}
    feature_schema_version = str(
        dataset_lineage.get("feature_schema_version")
        or ds.get("feature_schema_version")
        or "trade_behavior_features_v2"
    )
    trainer_script_path = Path(__file__).resolve()
    model_sha256 = _sha256_file(model_path)
    dataset_sha256 = _sha256_file(dataset_path)
    deployed_previous_model_sha256 = _sha256_file(Path(deployed_previous_path)) if deployed_previous_path else ""

    log_payload = {
        "log_schema_version": max(int(os.getenv("LOG_SCHEMA_VERSION", "2")), 1),
        "run_id": str(os.getenv("CORRELATION_RUN_ID", "") or "").strip(),
        "iter_id": str(os.getenv("CORRELATION_ITER_ID", "") or "").strip(),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": str(dataset_path),
        "rows": int(len(y)),
        "feature_dim": int(dataset_feature_dim),
        "feature_names": feature_names,
        "train_rows": int(len(y_tr)),
        "validation_rows": int(len(y_val)),
        "test_rows": int(len(y_te)),
        "label_counts": label_counts,
        "dataset_skipped_dim_mismatch": int(ds.get("_skipped_dim_mismatch", 0) or 0),
        "dataset_curated_guard": {
            "enabled": bool(args.require_curated_dataset),
            "ok": bool(curated_ok),
            "reason": str(curated_reason),
            "summary": curated_summary,
        },
        "split": split_meta,
        "class_balance_factors": class_balance_factors,
        "class_regime_factors": class_regime_factors,
        "seeds": [int(s) for s in seeds],
        "seed_results": [
            {
                "seed": int(r["seed"]),
                "best_epoch": int(r["best_epoch"]),
                "epochs_ran": int(r.get("epochs_ran", 0)),
                "early_stopped": bool(r.get("early_stopped", False)),
                "best_train_loss": float(r["best_train_loss"]),
                "best_validation_loss": float(r.get("best_validation_loss", 0.0)),
                "best_checkpoint_score": float(r.get("best_checkpoint_score", 0.0)),
                "temperature": float(r["temperature"]),
                "class_logit_bias": r.get("class_logit_bias", [0.0, 0.0, 0.0]),
                "score": float(r["score"]),
                "validation_metrics_pre_threshold_tuning": r.get("validation_metrics_pre_threshold_tuning", {}),
                "validation_metrics_threshold_tuned": r.get("validation_metrics_threshold_tuned", {}),
                "validation_metrics": r.get("validation_metrics", {}),
                "test_metrics_pre_threshold_tuning": r.get("test_metrics_pre_threshold_tuning", {}),
                "test_metrics": r["test_metrics"],
                "threshold_tuning": r.get("threshold_tuning", {}),
            }
            for r in seed_results
        ],
        "champion_seed": int(champion["seed"]),
        "best_epoch": int(champion["best_epoch"]),
        "epochs_ran": int(champion.get("epochs_ran", 0)),
        "early_stopped": bool(champion.get("early_stopped", False)),
        "best_train_loss": float(champion["best_train_loss"]),
        "best_validation_loss": float(champion.get("best_validation_loss", 0.0)),
        "validation_loss_pre_calibration": float(champion.get("validation_loss_pre_calibration", 0.0)),
        "validation_loss_post_calibration": float(champion.get("validation_loss_post_calibration", 0.0)),
        "temperature": float(champion["temperature"]),
        "class_logit_bias": champion.get("class_logit_bias", [0.0, 0.0, 0.0]),
        "train_metrics": champion["train_metrics"],
        "validation_metrics_pre_threshold_tuning": champion.get("validation_metrics_pre_threshold_tuning", {}),
        "validation_metrics_threshold_tuned": champion.get("validation_metrics_threshold_tuned", {}),
        "validation_metrics": champion.get("validation_metrics", {}),
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
            "rollback_require_schema_match": bool(args.rollback_require_schema_match),
            "rollback_schema_guard": {
                "enabled": bool(args.rollback_require_schema_match),
                "ok": bool(rollback_schema_ok),
                "reason": str(rollback_schema_reason),
                "dataset_feature_dim": int(dataset_feature_dim),
                "dataset_feature_names_count": int(len(feature_names)),
                "dataset_feature_names_sha256": _sha256_json_obj(feature_names),
                "previous_model": {
                    "path": str(prev_model_schema.get("path", "")),
                    "exists": bool(prev_model_schema.get("exists", False)),
                    "load_ok": bool(prev_model_schema.get("load_ok", False)),
                    "feature_dim_field": int(prev_model_schema.get("feature_dim_field", 0) or 0),
                    "effective_dim": int(prev_model_schema.get("effective_dim", 0) or 0),
                    "feature_names_count": int(prev_model_schema.get("feature_names_count", 0) or 0),
                    "has_feature_names": bool(prev_model_schema.get("has_feature_names", False)),
                    "feature_names_sha256": str(prev_model_schema.get("feature_names_sha256", "")),
                    "error": str(prev_model_schema.get("error", "")),
                },
            },
            "strict_gate": bool(args.strict_promotion_gate),
            "reasons": promotion_reasons,
            "thresholds": {
                "promotion_min_score_delta": float(args.promotion_min_score_delta),
                "promotion_max_neutral_f1_drop": float(args.promotion_max_neutral_f1_drop),
                "promotion_min_accuracy": float(args.promotion_min_accuracy),
                "promotion_min_macro_f1": float(args.promotion_min_macro_f1),
                "promotion_min_balanced_accuracy": float(args.promotion_min_balanced_accuracy),
                "promotion_min_positive_precision": float(args.promotion_min_positive_precision),
                "promotion_max_pos_recall_precision_gap": float(args.promotion_max_pos_recall_precision_gap),
            },
            "previous_model": str(prev_model_path) if prev_model_path else None,
            "previous_log": str(prev_log_path) if prev_log_path else None,
        },
        "data_quality_gate": dq_summary,
        "training_objective": {
            "snapshot_weight_cap": snapshot_weight_cap,
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
                "positive_precision_floor": float(args.threshold_positive_precision_floor),
                "max_pos_recall_precision_gap": float(args.threshold_max_pos_recall_precision_gap),
                "score_weight_positive_precision": float(args.threshold_score_weight_positive_precision),
                "score_gap_penalty": float(args.threshold_score_gap_penalty),
                "positive_logit_bias_max": float(args.positive_logit_bias_max),
                "negative_logit_bias_max": float(args.negative_logit_bias_max),
                "grid_steps": int(max(args.logit_bias_steps, 3)),
            },
            "early_stopping": {
                "patience": int(max(args.early_stop_patience, 1)),
                "min_delta": float(max(args.early_stop_min_delta, 0.0)),
                "min_epochs": int(max(args.early_stop_min_epochs, 1)),
            },
            "split": {
                "outer_mode": str(args.split_mode),
                "test_ratio": float(args.test_ratio),
                "val_ratio": float(args.val_ratio),
                "purge_seconds": float(args.purge_seconds),
            },
        },
        "lineage": {
            "feature_schema_version": feature_schema_version,
            "dataset_path": str(dataset_path),
            "dataset_sha256": dataset_sha256,
            "dataset_payload_sha256": str(dataset_lineage.get("output_payload_sha256") or ""),
            "dataset_builder_script": str(dataset_lineage.get("builder_script") or ""),
            "dataset_builder_script_sha256": str(dataset_lineage.get("builder_script_sha256") or ""),
            "dataset_builder_git_commit": str(dataset_lineage.get("git_commit") or ""),
            "trainer_script": str(trainer_script_path),
            "trainer_script_sha256": _sha256_file(trainer_script_path),
            "git_commit": _git_commit(PROJECT_ROOT),
            "model_sha256": model_sha256,
            "deployed_previous_model_sha256": deployed_previous_model_sha256,
            "candidate_payload_sha256": _sha256_json_obj(
                {
                    "test_metrics": champion["test_metrics"],
                    "candidate_score": float(candidate_score),
                    "champion_seed": int(champion["seed"]),
                }
            ),
        },
        "outcome_learning": ds.get("outcome_learning", {}),
    }

    log_path = LOGS_DIR / f"trade_behavior_policy_{timestamp}.json"
    snapshot_training_coverage = _write_snapshot_training_coverage_artifact(
        project_root=PROJECT_ROOT,
        dataset_path=dataset_path,
        dataset_obj=ds,
        model_path=model_path,
        log_path=log_path,
    )
    log_payload["snapshot_training_coverage"] = snapshot_training_coverage
    log_path.write_text(json.dumps(log_payload, indent=2), encoding="utf-8")

    print("Saved model:", model_path)
    print("Saved log:", log_path)
    print(
        "Snapshot training coverage:",
        snapshot_training_coverage.get("artifact_path", ""),
        "all_snapshot_data_incorporated=",
        int(bool(snapshot_training_coverage.get("all_snapshot_data_incorporated", False))),
    )
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
