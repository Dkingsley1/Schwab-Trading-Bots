import json
import importlib
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

from runtime_training_common import (
    RuntimeConfidenceBuilder,
    RuntimeFeatureBuilder,
    RuntimeLabelBuilder,
    RuntimeSampleFilter,
    load_runtime_observation_sequences,
    make_runtime_windowed_dataset,
)


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


def weighted_loss_fn(
    model,
    x,
    y,
    *,
    sample_weight=None,
    pos_weight: float = 1.0,
    neg_weight: float = 1.0,
):
    probs = mx.sigmoid(model(x))
    losses = nn.losses.binary_cross_entropy(probs, y)
    class_weight = (y * float(pos_weight)) + ((1.0 - y) * float(neg_weight))
    if sample_weight is not None:
        class_weight = class_weight * sample_weight
    return mx.sum(losses * class_weight) / (mx.sum(class_weight) + 1e-6)


def split_data(X, y, train_ratio=0.7, val_ratio=0.15):
    n = X.shape[0]
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    X_train, y_train = X[:n_train], y[:n_train]
    X_val, y_val = X[n_train : n_train + n_val], y[n_train : n_train + n_val]
    X_test, y_test = X[n_train + n_val :], y[n_train + n_val :]
    return X_train, y_train, X_val, y_val, X_test, y_test


def make_windowed_dataset(
    features: np.ndarray,
    close: np.ndarray,
    window: int,
    horizon: int,
    *,
    return_anchor_index: bool = False,
):
    feat_mean = features.mean(axis=0, keepdims=True)
    feat_std = features.std(axis=0, keepdims=True) + 1e-8
    feats = (features - feat_mean) / feat_std

    X = []
    y = []
    anchor_idx = []
    for i in range(len(feats) - window - horizon):
        X.append(feats[i : i + window].reshape(-1))
        fwd = (close[i + window + horizon] - close[i + window]) / max(close[i + window], 1e-8)
        y.append(1.0 if fwd > 0 else 0.0)
        anchor_idx.append(i + window)

    x_out = mx.array(np.array(X), dtype=mx.float32)
    y_out = mx.array(np.array(y).reshape(-1, 1), dtype=mx.float32)
    if return_anchor_index:
        return x_out, y_out, np.asarray(anchor_idx, dtype=np.int64)
    return x_out, y_out


def _flatten_param_tree(tree, prefix: str = "") -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    if isinstance(tree, dict):
        for key, value in tree.items():
            name = f"{prefix}.{key}" if prefix else str(key)
            out.update(_flatten_param_tree(value, name))
        return out
    out[prefix] = np.asarray(tree)
    return out


def _assign_param_tree(target, flat: Dict[str, np.ndarray], prefix: str = "") -> None:
    if isinstance(target, dict):
        for key, value in target.items():
            name = f"{prefix}.{key}" if prefix else str(key)
            _assign_param_tree(value, flat, name)
        return
    if prefix in flat:
        target[:] = mx.array(flat[prefix])


def _snapshot_model_params(model: nn.Module) -> Dict[str, np.ndarray]:
    mx.eval(model.parameters())
    flat = _flatten_param_tree(model.parameters())
    return {key: np.array(value, copy=True) for key, value in flat.items()}


def _restore_model_params(model: nn.Module, flat: Dict[str, np.ndarray]) -> None:
    _assign_param_tree(model.parameters(), flat)
    mx.eval(model.parameters())


def load_model(model, npz_path):
    data = np.load(npz_path, allow_pickle=True)
    params = model.parameters()
    flat_keys = [str(k) for k in data.files if "." in str(k)]
    if flat_keys:
        _assign_param_tree(params, {str(k): data[k] for k in flat_keys})
        return model

    # Backward compatibility for older sequential saves.
    if isinstance(params, dict):
        flat_params = _flatten_param_tree(params)
        for i, key in enumerate(flat_params.keys()):
            legacy = f"p{i}"
            if legacy in data:
                flat_params[key][...] = data[legacy]
        _assign_param_tree(params, flat_params)
        return model

    for i, p in enumerate(params):
        key = f"p{i}"
        if key in data:
            p[:] = mx.array(data[key])
    return model


def _teacher_registry_row(project_root: Path, bot_id: str) -> Dict[str, object]:
    registry_path = project_root / "master_bot_registry.json"
    if not registry_path.exists():
        return {}
    try:
        registry = json.loads(registry_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    for row in registry.get("sub_bots", []):
        if str(row.get("bot_id") or "").strip() == bot_id:
            return dict(row)
    return {}


def _latest_matching_file(base_dir: Path, pattern: str) -> Optional[Path]:
    matches = sorted(base_dir.glob(pattern), key=lambda p: p.stat().st_mtime if p.exists() else 0.0, reverse=True)
    return matches[0] if matches else None


def _resolve_teacher_artifacts(project_root: Path, bot_id: str) -> Tuple[Optional[Path], Optional[Path]]:
    row = _teacher_registry_row(project_root, bot_id)
    model_path = Path(str(row.get("model_path") or "")).expanduser() if row.get("model_path") else None
    log_path = Path(str(row.get("log_file") or "")).expanduser() if row.get("log_file") else None
    latest_model = _latest_matching_file(project_root / "models", f"{bot_id}_*.npz")
    latest_log = _latest_matching_file(project_root / "logs", f"{bot_id}_*.json")

    if latest_model is not None and (model_path is None or (not model_path.exists()) or latest_model.stat().st_mtime >= model_path.stat().st_mtime):
        model_path = latest_model
    if latest_log is not None and (log_path is None or (not log_path.exists()) or latest_log.stat().st_mtime >= log_path.stat().st_mtime):
        log_path = latest_log
    return model_path, log_path


def _load_teacher_spec(project_root: Path, bot_id: str) -> Optional[Dict[str, object]]:
    model_path, log_path = _resolve_teacher_artifacts(project_root, bot_id)
    if model_path is None or log_path is None or (not model_path.exists()) or (not log_path.exists()):
        return None
    try:
        payload = json.loads(log_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    config = dict(payload.get("config") or {})
    core_dir = project_root / "core"
    if str(core_dir) not in sys.path:
        sys.path.insert(0, str(core_dir))
    try:
        module = importlib.import_module(f"core.{bot_id}")
    except Exception:
        try:
            module = importlib.import_module(bot_id)
        except Exception:
            return None
    feature_builder = getattr(module, "build_features", None)
    source_kind = str(getattr(module, "FEATURE_SOURCE", "panel") or "panel").strip().lower()
    if not callable(feature_builder):
        return None
    return {
        "bot_id": bot_id,
        "model_path": model_path,
        "log_path": log_path,
        "config": config,
        "feature_builder": feature_builder,
        "source_kind": source_kind,
    }


def _panel_from_prices(prices: np.ndarray) -> ArrayMap:
    prices = np.asarray(prices, dtype=np.float64)
    prev = np.concatenate([[prices[0]], prices[:-1]])
    ret = np.diff(prices, prepend=prices[0]) / np.maximum(prev, 1e-8)
    vol = rolling_std(ret, 20)
    high = prices * (1.0 + np.maximum(np.abs(vol), 0.0015))
    low = prices * np.maximum(1e-6, 1.0 - np.maximum(np.abs(vol), 0.0015))
    volume = np.maximum(1_000_000.0 * (1.0 + 25.0 * np.abs(ret)), 100_000.0)
    bench_close = ema(prices, 20)
    bench_prev = np.concatenate([[bench_close[0]], bench_close[:-1]])
    bench_ret = np.diff(bench_close, prepend=bench_close[0]) / np.maximum(bench_prev, 1e-8)
    open_price = prev
    gap = (open_price - prev) / np.maximum(prev, 1e-8)
    breadth_bias = np.tanh(6.0 * ret)
    adv = np.maximum(1200 + 700 * breadth_bias, 50.0)
    dec = np.maximum(1200 - 700 * breadth_bias, 50.0)
    up_vol = np.maximum(2.0e8 + 8.0e7 * breadth_bias, 1.0e6)
    down_vol = np.maximum(2.0e8 - 8.0e7 * breadth_bias, 1.0e6)
    vix = np.maximum(18.0 + 220.0 * rolling_std(ret, 20), 9.0)
    return {
        "close": prices,
        "high": high,
        "low": low,
        "volume": volume,
        "bench_close": bench_close,
        "ret": ret,
        "bench_ret": bench_ret,
        "vix": vix,
        "vix9d": np.maximum(vix - 0.5, 8.5),
        "vix3m": np.maximum(vix + 0.5, 9.0),
        "adv": adv,
        "dec": dec,
        "up_vol": up_vol,
        "down_vol": down_vol,
        "open": open_price,
        "gap": gap,
    }


def _teacher_soft_targets(
    *,
    project_root: Path,
    teacher_ids: List[str],
    panel: Optional[ArrayMap],
    prices: Optional[np.ndarray],
    student_anchor_idx: np.ndarray,
) -> Tuple[Optional[np.ndarray], List[str]]:
    aggregates: List[np.ndarray] = []
    used_ids: List[str] = []
    panel_obj = panel
    prices_obj = np.asarray(prices, dtype=np.float64) if prices is not None else None
    if panel_obj is None and prices_obj is not None:
        panel_obj = _panel_from_prices(prices_obj)
    if prices_obj is None and panel_obj is not None:
        prices_obj = np.asarray(panel_obj["close"], dtype=np.float64)

    for bot_id in teacher_ids:
        spec = _load_teacher_spec(project_root, bot_id)
        if not spec:
            continue
        config = dict(spec.get("config") or {})
        try:
            source_kind = str(spec.get("source_kind") or "panel").strip().lower()
            if source_kind == "prices":
                if prices_obj is None:
                    continue
                features = spec["feature_builder"](prices_obj)
                x_teacher, _, teacher_anchor_idx = make_windowed_dataset(
                    features,
                    prices_obj,
                    window=int(config.get("window", 30) or 30),
                    horizon=int(config.get("horizon", 1) or 1),
                    return_anchor_index=True,
                )
            else:
                if panel_obj is None:
                    continue
                features = spec["feature_builder"](panel_obj)
                x_teacher, _, teacher_anchor_idx = make_windowed_dataset(
                    features,
                    panel_obj["close"],
                    window=int(config.get("window", 30) or 30),
                    horizon=int(config.get("horizon", 3) or 3),
                    return_anchor_index=True,
                )
            input_dim = int(config.get("input_dim") or int(x_teacher.shape[1]))
            model = TradingBrain(input_dim)
            load_model(model, str(spec["model_path"]))
            probs = mx.sigmoid(model(x_teacher))
            mx.eval(probs)
            teacher_probs = np.asarray(probs).reshape(-1)
        except Exception:
            continue

        index_to_prob = {int(idx): float(prob) for idx, prob in zip(teacher_anchor_idx.tolist(), teacher_probs.tolist())}
        aligned = []
        coverage = 0
        for idx in student_anchor_idx.tolist():
            prob = index_to_prob.get(int(idx))
            if prob is None:
                aligned.append(np.nan)
            else:
                coverage += 1
                aligned.append(prob)
        if coverage == 0:
            continue
        aggregates.append(np.asarray(aligned, dtype=np.float64))
        used_ids.append(bot_id)

    if not aggregates:
        return None, []

    stacked = np.vstack(aggregates)
    valid = np.isfinite(stacked)
    counts = np.sum(valid, axis=0)
    sums = np.sum(np.where(valid, stacked, 0.0), axis=0)
    blended = np.where(counts > 0, sums / np.maximum(counts, 1), np.nan)
    return blended.astype(np.float32), used_ids


def _distillation_config(project_root: Path) -> Tuple[bool, List[str], float]:
    enabled = str(os.getenv("DISTILLATION_ENABLED", "0")).strip() == "1"
    is_student = str(os.getenv("DISTILLATION_STUDENT", "0")).strip() == "1"
    teacher_ids = [tok.strip() for tok in str(os.getenv("DISTILLATION_TEACHERS", "")).split(",") if tok.strip()]
    try:
        teacher_weight = float(os.getenv("DISTILLATION_TEACHER_WEIGHT", "0.30") or 0.30)
    except ValueError:
        teacher_weight = 0.30
    teacher_weight = min(max(teacher_weight, 0.0), 0.90)
    return enabled and is_student and bool(teacher_ids), teacher_ids, teacher_weight


def save_artifacts(model, config, metrics, run_tag):
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    models_dir = os.path.join(base_dir, "models")
    logs_dir = os.path.join(base_dir, "logs")
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = f"{run_tag}_{ts}"

    params = model.parameters()
    state = _flatten_param_tree(params)
    model_path = os.path.join(models_dir, f"{base_name}.npz")
    np.savez(model_path, **{k: np.asarray(v) for k, v in state.items()})

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
    project_root = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

    panel = simulate_market_panel(n=num_points)
    features = feature_builder(panel)
    close = panel["close"]

    X, y, anchor_idx = make_windowed_dataset(features, close, window=window, horizon=horizon, return_anchor_index=True)
    X_train, y_train, X_val, y_val, X_test, y_test = split_data(X, y)
    n = X.shape[0]
    n_train = int(n * 0.7)
    n_val = int(n * 0.15)
    anchor_train = anchor_idx[:n_train]
    anchor_val = anchor_idx[n_train : n_train + n_val]
    anchor_test = anchor_idx[n_train + n_val :]

    distillation_enabled, teacher_ids, teacher_weight = _distillation_config(project_root)
    teacher_soft_train = None
    teacher_soft_val = None
    used_teacher_ids: List[str] = []
    if distillation_enabled:
        teacher_soft_all, used_teacher_ids = _teacher_soft_targets(
            project_root=project_root,
            teacher_ids=teacher_ids,
            panel=panel,
            prices=panel["close"],
            student_anchor_idx=anchor_idx,
        )
        if teacher_soft_all is not None and used_teacher_ids:
            teacher_soft_train = teacher_soft_all[:n_train]
            teacher_soft_val = teacher_soft_all[n_train : n_train + n_val]
        else:
            distillation_enabled = False

    brain = TradingBrain(int(X.shape[1]))
    mx.eval(brain.parameters())

    optimizer = optim.Adam(learning_rate=learning_rate)
    loss_and_grad_fn = nn.value_and_grad(brain, loss_fn)

    best_val = float("inf")
    best_epoch = -1
    best_params = _snapshot_model_params(brain)
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
            if distillation_enabled and teacher_soft_train is not None:
                soft_np = teacher_soft_train[np.asarray(bidx)]
                soft_np = np.where(np.isfinite(soft_np), soft_np, np.asarray(yb).reshape(-1))
                hard_np = np.asarray(yb).reshape(-1)
                target_np = ((1.0 - teacher_weight) * hard_np) + (teacher_weight * soft_np)
                yb = mx.array(target_np.reshape(-1, 1), dtype=mx.float32)

            loss, grads = loss_and_grad_fn(brain, xb, yb)
            optimizer.update(brain, grads)
            mx.eval(brain.parameters(), optimizer.state)

            total_loss += float(loss)
            batches += 1

        if distillation_enabled and teacher_soft_val is not None:
            val_soft = np.where(np.isfinite(teacher_soft_val), teacher_soft_val, np.asarray(y_val).reshape(-1))
            val_hard = np.asarray(y_val).reshape(-1)
            y_val_effective = mx.array((((1.0 - teacher_weight) * val_hard) + (teacher_weight * val_soft)).reshape(-1, 1), dtype=mx.float32)
        else:
            y_val_effective = y_val
        val_loss = float(loss_fn(brain, X_val, y_val_effective))
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
    pred_probs_np = np.asarray(preds).reshape(-1)
    y_test_np = np.asarray(y_test).reshape(-1)
    y_all_np = np.asarray(y).reshape(-1)
    dataset_positive_rate = float(np.mean(y_all_np)) if y_all_np.size else 0.0
    acted_threshold = 0.65
    quality_metrics = _classification_quality_metrics(
        pred_probs_np,
        y_test_np,
        acted_threshold=acted_threshold,
        positive_rate=dataset_positive_rate,
    )
    acc = float(quality_metrics["test_accuracy"])
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
        **quality_metrics,
    }
    if distillation_enabled:
        metrics["distillation_active"] = True
        metrics["distillation_teacher_count"] = len(used_teacher_ids)
    config["distillation"] = {
        "enabled": bool(distillation_enabled),
        "teacher_ids": used_teacher_ids,
        "teacher_weight": float(teacher_weight if distillation_enabled else 0.0),
    }
    save_artifacts(brain, config, metrics, run_tag=run_tag)
    return brain


def train_price_indicator_bot(
    *,
    run_tag: str,
    feature_names: List[str],
    feature_builder: Callable[[np.ndarray], np.ndarray],
    price_simulator: Callable[[int], np.ndarray],
    dataset_builder: Optional[Callable[[np.ndarray], Tuple[object, ...]]] = None,
    num_points: int = 5000,
    window: int = 30,
    horizon: int = 1,
    learning_rate: float = 0.001,
    epochs: int = 200,
    batch_size: int = 128,
    patience: int = 15,
) -> TradingBrain:
    np.random.seed(42)
    project_root = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

    prices = np.asarray(price_simulator(num_points), dtype=np.float64)
    if dataset_builder is not None:
        dataset_out = dataset_builder(prices)
        if len(dataset_out) == 3:
            X, y, anchor_idx = dataset_out
        else:
            X, y = dataset_out[:2]
            anchor_idx = np.arange(int(X.shape[0]), dtype=np.int64) + int(window)
    else:
        features = feature_builder(prices)
        X, y, anchor_idx = make_windowed_dataset(features, prices, window=window, horizon=horizon, return_anchor_index=True)
    X_train, y_train, X_val, y_val, X_test, y_test = split_data(X, y)
    n = X.shape[0]
    n_train = int(n * 0.7)
    n_val = int(n * 0.15)

    distillation_enabled, teacher_ids, teacher_weight = _distillation_config(project_root)
    teacher_soft_train = None
    teacher_soft_val = None
    used_teacher_ids: List[str] = []
    if distillation_enabled:
        teacher_soft_all, used_teacher_ids = _teacher_soft_targets(
            project_root=project_root,
            teacher_ids=teacher_ids,
            panel=_panel_from_prices(prices),
            prices=prices,
            student_anchor_idx=anchor_idx,
        )
        if teacher_soft_all is not None and used_teacher_ids:
            teacher_soft_train = teacher_soft_all[:n_train]
            teacher_soft_val = teacher_soft_all[n_train : n_train + n_val]
        else:
            distillation_enabled = False

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
            if distillation_enabled and teacher_soft_train is not None:
                soft_np = teacher_soft_train[np.asarray(bidx)]
                soft_np = np.where(np.isfinite(soft_np), soft_np, np.asarray(yb).reshape(-1))
                hard_np = np.asarray(yb).reshape(-1)
                target_np = ((1.0 - teacher_weight) * hard_np) + (teacher_weight * soft_np)
                yb = mx.array(target_np.reshape(-1, 1), dtype=mx.float32)

            loss, grads = loss_and_grad_fn(brain, xb, yb)
            optimizer.update(brain, grads)
            mx.eval(brain.parameters(), optimizer.state)
            total_loss += float(loss)
            batches += 1

        if distillation_enabled and teacher_soft_val is not None:
            val_soft = np.where(np.isfinite(teacher_soft_val), teacher_soft_val, np.asarray(y_val).reshape(-1))
            val_hard = np.asarray(y_val).reshape(-1)
            y_val_effective = mx.array((((1.0 - teacher_weight) * val_hard) + (teacher_weight * val_soft)).reshape(-1, 1), dtype=mx.float32)
        else:
            y_val_effective = y_val
        val_loss = float(loss_fn(brain, X_val, y_val_effective))
        if epoch % 10 == 0:
            print(f"Epoch {epoch} | Train {total_loss / max(batches, 1):.6f} | Val {val_loss:.6f}")
        if val_loss < best_val:
            best_val = val_loss
            patience_left = patience
        else:
            patience_left -= 1
            if patience_left == 0:
                print("Early stopping.")
                break

    preds = mx.sigmoid(brain(X_test))
    pred_probs_np = np.asarray(preds).reshape(-1)
    y_test_np = np.asarray(y_test).reshape(-1)
    y_all_np = np.asarray(y).reshape(-1)
    dataset_positive_rate = float(np.mean(y_all_np)) if y_all_np.size else 0.0
    acted_threshold = 0.65
    quality_metrics = _classification_quality_metrics(
        pred_probs_np,
        y_test_np,
        acted_threshold=acted_threshold,
        positive_rate=dataset_positive_rate,
    )
    acc = float(quality_metrics["test_accuracy"])
    print(f"Test accuracy: {acc:.4f}")

    config = {
        "window": window,
        "horizon": horizon,
        "learning_rate": learning_rate,
        "epochs": epochs,
        "batch_size": batch_size,
        "patience": patience,
        "input_dim": int(X.shape[1]),
        "num_points": int(len(prices)),
        "features": feature_names,
        "distillation": {
            "enabled": bool(distillation_enabled),
            "teacher_ids": used_teacher_ids,
            "teacher_weight": float(teacher_weight if distillation_enabled else 0.0),
        },
    }
    metrics = {
        "best_val_loss": float(best_val),
        "final_val_loss": float(val_loss),
        "test_accuracy": float(acc),
        **quality_metrics,
    }
    if distillation_enabled:
        metrics["distillation_active"] = True
        metrics["distillation_teacher_count"] = len(used_teacher_ids)
    save_artifacts(brain, config, metrics, run_tag=run_tag)
    return brain


def train_runtime_indicator_bot(
    *,
    run_tag: str,
    feature_names: List[str],
    runtime_feature_builder: RuntimeFeatureBuilder,
    runtime_label_builder: RuntimeLabelBuilder,
    lookback_days: int = 14,
    mode_allowlist: Optional[List[str]] = None,
    symbol_allowlist: Optional[List[str]] = None,
    sample_filter: Optional[RuntimeSampleFilter] = None,
    confidence_builder: Optional[RuntimeConfidenceBuilder] = None,
    min_confidence: float = 0.0,
    sample_stride: int = 1,
    window: int = 30,
    horizon: int = 3,
    learning_rate: float = 0.0008,
    epochs: int = 220,
    batch_size: int = 128,
    patience: int = 18,
    min_samples: int = 256,
    min_sequences: int = 2,
    min_positive_samples: int = 0,
    min_negative_samples: int = 0,
    acted_prob_threshold: float = 0.65,
    fallback_trainer: Optional[Callable[[], TradingBrain]] = None,
    allow_fallback_on_insufficient_data: bool = True,
    max_best_val_loss: Optional[float] = None,
    max_final_val_loss: Optional[float] = None,
    min_long_precision: float = 0.0,
    min_short_precision: float = 0.0,
    require_both_sides_precision: bool = False,
    min_acted_accuracy: float = 0.0,
    min_long_acted_count: int = 0,
    min_short_acted_count: int = 0,
    min_accuracy_lift_over_majority: Optional[float] = None,
    min_label_balance_score: Optional[float] = None,
    min_precision_balance_score: float = 0.0,
) -> TradingBrain:
    np.random.seed(42)
    project_root = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    try:
        env_lookback_override = int(float(os.getenv("RUNTIME_TRAIN_LOOKBACK_DAYS_OVERRIDE", "").strip() or 0))
    except ValueError:
        env_lookback_override = 0
    effective_lookback_days = max(int(lookback_days), int(env_lookback_override or 0))

    print(
        "[RuntimeTraining] loading_sequences "
        f"run_tag={run_tag} lookback_days={effective_lookback_days}",
        flush=True,
    )
    sequences = load_runtime_observation_sequences(
        project_root,
        lookback_days=effective_lookback_days,
        mode_allowlist=mode_allowlist,
        symbol_allowlist=symbol_allowlist,
    )
    sequence_count = len(sequences)
    observation_count = sum(len(rows) for rows in sequences.values())
    print(
        "[RuntimeTraining] sequences_loaded "
        f"run_tag={run_tag} sequences={sequence_count} observations={observation_count}",
        flush=True,
    )
    print(
        "[RuntimeTraining] building_dataset "
        f"run_tag={run_tag} window={window} horizon={horizon}",
        flush=True,
    )
    X_np, y_np, runtime_meta = make_runtime_windowed_dataset(
        sequences=sequences,
        feature_builder=runtime_feature_builder,
        label_builder=runtime_label_builder,
        sample_filter=sample_filter,
        confidence_builder=confidence_builder,
        min_confidence=min_confidence,
        sample_stride=sample_stride,
        window=window,
        horizon=horizon,
    )

    sample_count = int(X_np.shape[0]) if X_np.ndim == 2 else 0
    print(
        "[RuntimeTraining] dataset_ready "
        f"run_tag={run_tag} samples={sample_count} "
        f"eligible_sequences={runtime_meta.get('eligible_sequences', 0)} "
        f"positive_rate={float(runtime_meta.get('positive_rate', 0.0) or 0.0):.4f}",
        flush=True,
    )
    sample_confidence = np.asarray(
        runtime_meta.pop("_sample_confidence", np.ones((sample_count,), dtype=np.float32)),
        dtype=np.float32,
    ).reshape(-1)
    labels_np = np.asarray(y_np).reshape(-1)
    positive_rate = float(runtime_meta.get("positive_rate", 0.0) or 0.0)
    positive_samples = int(np.sum(labels_np >= 0.5))
    negative_samples = int(sample_count - positive_samples)
    if (
        sample_count < max(int(min_samples), batch_size * 2)
        or int(runtime_meta.get("eligible_sequences", 0) or 0) < max(int(min_sequences), 1)
        or positive_rate <= 0.02
        or positive_rate >= 0.98
    ):
        if fallback_trainer is not None and allow_fallback_on_insufficient_data:
            print(
                "[RuntimeTraining] fallback "
                f"run_tag={run_tag} samples={sample_count} "
                f"eligible_sequences={runtime_meta.get('eligible_sequences', 0)} "
                f"positive_rate={positive_rate:.4f}",
                flush=True,
            )
            return fallback_trainer()
        raise RuntimeError(
            f"insufficient_runtime_training_data run_tag={run_tag} "
            f"samples={sample_count} eligible_sequences={runtime_meta.get('eligible_sequences', 0)} "
            f"positive_rate={positive_rate:.4f}"
        )
    if positive_samples < max(int(min_positive_samples), 0) or negative_samples < max(int(min_negative_samples), 0):
        if fallback_trainer is not None and allow_fallback_on_insufficient_data:
            return fallback_trainer()
        raise RuntimeError(
            f"insufficient_runtime_training_side_samples run_tag={run_tag} "
            f"positive_samples={positive_samples} negative_samples={negative_samples} "
            f"min_positive_samples={int(min_positive_samples)} min_negative_samples={int(min_negative_samples)}"
        )

    feat_mean = X_np.mean(axis=0, keepdims=True)
    feat_std = X_np.std(axis=0, keepdims=True) + 1e-8
    X_np = (X_np - feat_mean) / feat_std

    X = mx.array(X_np, dtype=mx.float32)
    y = mx.array(y_np, dtype=mx.float32)
    X_train, y_train, X_val, y_val, X_test, y_test = split_data(X, y)
    n_train = int(sample_count * 0.7)
    n_val = int(sample_count * 0.15)
    sample_confidence_train = sample_confidence[:n_train]
    sample_confidence_val = sample_confidence[n_train : n_train + n_val]
    sample_confidence_test = sample_confidence[n_train + n_val :]

    brain = TradingBrain(int(X.shape[1]))
    mx.eval(brain.parameters())

    optimizer = optim.Adam(learning_rate=learning_rate)
    train_positive_rate = float(np.mean(np.asarray(y_train).reshape(-1))) if X_train.shape[0] else positive_rate
    train_positive_samples = int(np.sum(np.asarray(y_train).reshape(-1) >= 0.5)) if X_train.shape[0] else 0
    train_negative_samples = int(X_train.shape[0] - train_positive_samples)
    val_positive_samples = int(np.sum(np.asarray(y_val).reshape(-1) >= 0.5)) if X_val.shape[0] else 0
    val_negative_samples = int(X_val.shape[0] - val_positive_samples)
    test_positive_samples = int(np.sum(np.asarray(y_test).reshape(-1) >= 0.5)) if X_test.shape[0] else 0
    test_negative_samples = int(X_test.shape[0] - test_positive_samples)
    class_pos_weight = float(np.clip(0.5 / max(train_positive_rate, 1e-6), 0.5, 4.0))
    class_neg_weight = float(np.clip(0.5 / max(1.0 - train_positive_rate, 1e-6), 0.5, 4.0))
    train_sample_weights = np.clip(0.25 + (0.75 * sample_confidence_train.reshape(-1)), 0.25, 1.0).astype(np.float32)
    val_sample_weights = np.clip(0.25 + (0.75 * sample_confidence_val.reshape(-1)), 0.25, 1.0).astype(np.float32)

    def runtime_loss(model, x, y, sample_weight):
        return weighted_loss_fn(
            model,
            x,
            y,
            sample_weight=sample_weight,
            pos_weight=class_pos_weight,
            neg_weight=class_neg_weight,
        )

    loss_and_grad_fn = nn.value_and_grad(brain, runtime_loss)

    best_val = float("inf")
    patience_left = patience

    print("Training...", flush=True)
    for epoch in range(epochs):
        idx = np.random.permutation(X_train.shape[0])
        total_loss = 0.0
        batches = 0

        for start in range(0, X_train.shape[0], batch_size):
            bidx = mx.array(idx[start : start + batch_size])
            xb = mx.take(X_train, bidx, axis=0)
            yb = mx.take(y_train, bidx, axis=0)
            wb = mx.array(train_sample_weights[idx[start : start + batch_size]].reshape(-1, 1), dtype=mx.float32)

            loss, grads = loss_and_grad_fn(brain, xb, yb, wb)
            optimizer.update(brain, grads)
            mx.eval(brain.parameters(), optimizer.state)

            total_loss += float(loss)
            batches += 1

        val_weight = mx.array(val_sample_weights.reshape(-1, 1), dtype=mx.float32)
        val_loss = float(runtime_loss(brain, X_val, y_val, val_weight))
        if epoch % 10 == 0:
            print(
                f"Epoch {epoch} | Train {total_loss / max(batches, 1):.6f} | Val {val_loss:.6f}",
                flush=True,
            )

        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
            best_params = _snapshot_model_params(brain)
            patience_left = patience
        else:
            patience_left -= 1
            if patience_left == 0:
                print("Early stopping.", flush=True)
                break

    if best_params:
        _restore_model_params(brain, best_params)

    configured_acted_threshold = float(min(max(acted_prob_threshold, 0.5), 0.95))
    val_weight = mx.array(val_sample_weights.reshape(-1, 1), dtype=mx.float32)
    val_loss = float(runtime_loss(brain, X_val, y_val, val_weight))
    val_pred_probs_np = np.asarray(mx.sigmoid(brain(X_val))).reshape(-1) if X_val.shape[0] else np.zeros((0,), dtype=np.float32)
    y_val_np = np.asarray(y_val).reshape(-1)
    desired_long_actions = min(max(int(min_long_acted_count), 2), 6) if int(min_long_acted_count) > 0 else 2
    desired_short_actions = min(max(int(min_short_acted_count), 2), 6) if int(min_short_acted_count) > 0 else 2
    long_acted_threshold, short_acted_threshold, threshold_meta = _select_calibrated_action_thresholds(
        val_pred_probs_np,
        y_val_np,
        default_threshold=configured_acted_threshold,
        sample_confidence=sample_confidence_val,
        min_long_acted_count=desired_long_actions,
        min_short_acted_count=desired_short_actions,
    )
    preds = mx.sigmoid(brain(X_test))
    pred_probs_np = np.asarray(preds).reshape(-1)
    y_test_np = np.asarray(y_test).reshape(-1)
    quality_metrics = _classification_quality_metrics(
        pred_probs_np,
        y_test_np,
        long_acted_threshold=long_acted_threshold,
        short_acted_threshold=short_acted_threshold,
        sample_confidence=sample_confidence_test,
        positive_rate=positive_rate,
    )
    acc = float(quality_metrics["test_accuracy"])
    acted_accuracy = float(quality_metrics["acted_accuracy"])
    long_precision = float(quality_metrics["long_precision"])
    short_precision = float(quality_metrics["short_precision"])
    accuracy_lift_over_majority = float(quality_metrics["accuracy_lift_over_majority"])
    label_balance_score = float(quality_metrics["label_balance_score"])
    precision_balance_score = float(quality_metrics["precision_balance_score"])
    print(f"Test accuracy: {acc:.4f}")

    config = {
        "window": window,
        "horizon": horizon,
        "learning_rate": learning_rate,
        "epochs": epochs,
        "batch_size": batch_size,
        "patience": patience,
        "input_dim": int(X.shape[1]),
        "num_points": int(sample_count),
        "features": feature_names,
        "data_source": "live_runtime",
        "runtime": {
            "lookback_days": int(effective_lookback_days),
            "mode_allowlist": list(mode_allowlist or []),
            "symbol_allowlist": list(symbol_allowlist or []),
            "sample_filter_active": bool(sample_filter is not None),
            "confidence_builder_active": bool(confidence_builder is not None),
            "min_confidence": float(min_confidence),
            "sample_stride": int(max(sample_stride, 1)),
            "acted_prob_threshold": float(long_acted_threshold),
            "short_acted_prob_threshold": float(short_acted_threshold),
            "configured_acted_prob_threshold": float(configured_acted_threshold),
            "acted_threshold_calibration": threshold_meta,
            "positive_samples": int(positive_samples),
            "negative_samples": int(negative_samples),
            "min_positive_samples": int(min_positive_samples),
            "min_negative_samples": int(min_negative_samples),
            "train_positive_rate": float(train_positive_rate),
            "train_positive_samples": int(train_positive_samples),
            "train_negative_samples": int(train_negative_samples),
            "val_positive_samples": int(val_positive_samples),
            "val_negative_samples": int(val_negative_samples),
            "test_positive_samples": int(test_positive_samples),
            "test_negative_samples": int(test_negative_samples),
            "class_pos_weight": float(class_pos_weight),
            "class_neg_weight": float(class_neg_weight),
            "train_sample_weight_mean": float(np.mean(train_sample_weights)) if train_sample_weights.size else 0.0,
            "val_sample_weight_mean": float(np.mean(val_sample_weights)) if val_sample_weights.size else 0.0,
            "best_epoch": int(best_epoch),
            **runtime_meta,
        },
        "distillation": {
            "enabled": False,
            "teacher_ids": [],
            "teacher_weight": 0.0,
        },
    }
    metrics = {
        "best_val_loss": float(best_val),
        "final_val_loss": float(val_loss),
        **quality_metrics,
    }
    quality_failures: list[str] = []
    if max_best_val_loss is not None and float(best_val) > float(max_best_val_loss):
        quality_failures.append(
            f"best_val_loss={float(best_val):.6f} > max_best_val_loss={float(max_best_val_loss):.6f}"
        )
    if max_final_val_loss is not None and float(val_loss) > float(max_final_val_loss):
        quality_failures.append(
            f"final_val_loss={float(val_loss):.6f} > max_final_val_loss={float(max_final_val_loss):.6f}"
        )
    if float(long_precision) < float(min_long_precision):
        quality_failures.append(
            f"long_precision={float(long_precision):.4f} < min_long_precision={float(min_long_precision):.4f}"
        )
    if float(short_precision) < float(min_short_precision):
        quality_failures.append(
            f"short_precision={float(short_precision):.4f} < min_short_precision={float(min_short_precision):.4f}"
        )
    if require_both_sides_precision and (float(long_precision) <= 0.0 or float(short_precision) <= 0.0):
        quality_failures.append(
            f"require_both_sides_precision long_precision={float(long_precision):.4f} short_precision={float(short_precision):.4f}"
        )
    if float(acted_accuracy) < float(min_acted_accuracy):
        quality_failures.append(
            f"acted_accuracy={float(acted_accuracy):.4f} < min_acted_accuracy={float(min_acted_accuracy):.4f}"
        )
    if int(quality_metrics["long_acted_count"]) < int(min_long_acted_count):
        quality_failures.append(
            f"long_acted_count={int(quality_metrics['long_acted_count'])} < min_long_acted_count={int(min_long_acted_count)}"
        )
    if int(quality_metrics["short_acted_count"]) < int(min_short_acted_count):
        quality_failures.append(
            f"short_acted_count={int(quality_metrics['short_acted_count'])} < min_short_acted_count={int(min_short_acted_count)}"
        )
    if min_accuracy_lift_over_majority is not None and float(accuracy_lift_over_majority) < float(min_accuracy_lift_over_majority):
        quality_failures.append(
            "accuracy_lift_over_majority="
            f"{float(accuracy_lift_over_majority):.4f} < min_accuracy_lift_over_majority={float(min_accuracy_lift_over_majority):.4f}"
        )
    if min_label_balance_score is not None and float(label_balance_score) < float(min_label_balance_score):
        quality_failures.append(
            f"label_balance_score={float(label_balance_score):.4f} < min_label_balance_score={float(min_label_balance_score):.4f}"
        )
    if float(precision_balance_score) < float(min_precision_balance_score):
        quality_failures.append(
            "precision_balance_score="
            f"{float(precision_balance_score):.4f} < min_precision_balance_score={float(min_precision_balance_score):.4f}"
        )
    if quality_failures:
        raise RuntimeError(
            f"runtime_training_quality_guard_failed run_tag={run_tag} "
            + "; ".join(quality_failures)
        )
    save_artifacts(brain, config, metrics, run_tag=run_tag)
    return brain


def _classification_quality_metrics(
    pred_probs_np: np.ndarray,
    y_true_np: np.ndarray,
    *,
    acted_threshold: float = 0.65,
    long_acted_threshold: Optional[float] = None,
    short_acted_threshold: Optional[float] = None,
    sample_confidence: Optional[np.ndarray] = None,
    positive_rate: Optional[float] = None,
) -> Dict[str, float]:
    pred_probs = np.asarray(pred_probs_np, dtype=np.float32).reshape(-1)
    y_true = np.asarray(y_true_np, dtype=np.float32).reshape(-1)
    pred_labels = (pred_probs > 0.5).astype(np.float32)
    symmetric_threshold = float(min(max(float(acted_threshold), 0.5), 0.95))
    long_threshold = float(
        min(
            max(float(long_acted_threshold if long_acted_threshold is not None else symmetric_threshold), 0.5),
            0.95,
        )
    )
    short_threshold = float(
        max(
            min(float(short_acted_threshold if short_acted_threshold is not None else (1.0 - symmetric_threshold)), 0.5),
            0.05,
        )
    )
    if short_threshold > long_threshold:
        long_threshold = symmetric_threshold
        short_threshold = 1.0 - symmetric_threshold

    test_accuracy = float(np.mean((pred_labels == y_true).astype(np.float32))) if y_true.size else 0.0
    used_positive_rate = float(np.mean(y_true)) if positive_rate is None and y_true.size else float(positive_rate or 0.0)
    majority_class_accuracy = max(used_positive_rate, 1.0 - used_positive_rate)
    accuracy_lift_over_majority = test_accuracy - majority_class_accuracy
    label_balance_score = float(np.clip(1.0 - (2.0 * abs(used_positive_rate - 0.5)), 0.0, 1.0))

    long_mask = pred_probs >= long_threshold
    short_mask = pred_probs <= short_threshold
    acted_mask = long_mask | short_mask
    acted_coverage = float(np.mean(acted_mask.astype(np.float32))) if acted_mask.size else 0.0
    acted_pred = np.zeros_like(pred_probs, dtype=np.float32)
    acted_pred[long_mask] = 1.0
    acted_accuracy = (
        float(np.mean((acted_pred[acted_mask] == y_true[acted_mask]).astype(np.float32)))
        if np.any(acted_mask)
        else 0.0
    )
    long_acted_count = int(np.sum(long_mask))
    short_acted_count = int(np.sum(short_mask))
    long_precision = float(np.mean(y_true[long_mask])) if np.any(long_mask) else 0.0
    short_precision = float(np.mean(1.0 - y_true[short_mask])) if np.any(short_mask) else 0.0
    precision_high = max(long_precision, short_precision)
    precision_low = min(long_precision, short_precision)
    precision_balance_score = float(precision_low / precision_high) if precision_high > 0.0 else 0.0
    pred_confidence = np.abs(pred_probs - 0.5) * 2.0
    sample_conf = np.asarray(sample_confidence, dtype=np.float32).reshape(-1) if sample_confidence is not None else np.zeros((0,), dtype=np.float32)

    return {
        "test_accuracy": float(test_accuracy),
        "positive_rate": float(used_positive_rate),
        "majority_class_accuracy": float(majority_class_accuracy),
        "accuracy_lift_over_majority": float(accuracy_lift_over_majority),
        "label_balance_score": float(label_balance_score),
        "acted_prob_threshold": float(long_threshold),
        "short_acted_prob_threshold": float(short_threshold),
        "acted_coverage": float(acted_coverage),
        "acted_count": int(np.sum(acted_mask)),
        "acted_accuracy": float(acted_accuracy),
        "long_acted_count": int(long_acted_count),
        "short_acted_count": int(short_acted_count),
        "long_precision": float(long_precision),
        "short_precision": float(short_precision),
        "precision_balance_score": float(precision_balance_score),
        "pred_confidence_mean": float(np.mean(pred_confidence)) if pred_confidence.size else 0.0,
        "pred_confidence_max": float(np.max(pred_confidence)) if pred_confidence.size else 0.0,
        "input_confidence_mean": float(np.mean(sample_conf)) if sample_conf.size else 0.0,
    }


def _select_calibrated_action_thresholds(
    pred_probs_np: np.ndarray,
    y_true_np: np.ndarray,
    *,
    default_threshold: float,
    sample_confidence: Optional[np.ndarray] = None,
    min_long_acted_count: int = 2,
    min_short_acted_count: int = 2,
) -> tuple[float, float, Dict[str, Any]]:
    default_threshold = float(min(max(default_threshold, 0.5), 0.95))
    pred_probs = np.asarray(pred_probs_np, dtype=np.float32).reshape(-1)
    y_true = np.asarray(y_true_np, dtype=np.float32).reshape(-1)
    if pred_probs.size == 0 or y_true.size == 0:
        return default_threshold, 1.0 - default_threshold, {
            "calibrated": False,
            "reason": "empty_validation_split",
            "selected_threshold": float(default_threshold),
            "selected_long_threshold": float(default_threshold),
            "selected_short_threshold": float(1.0 - default_threshold),
            "default_threshold": float(default_threshold),
            "candidate_count": 0,
        }

    long_candidates = sorted(
        {
            float(default_threshold),
            0.50,
            0.51,
            0.52,
            0.53,
            0.54,
            0.55,
            0.56,
            0.58,
            0.60,
            0.62,
            0.64,
            0.66,
            0.68,
            0.70,
            0.72,
        }
    )
    short_candidates = sorted(
        {
            float(1.0 - default_threshold),
            0.28,
            0.30,
            0.32,
            0.34,
            0.36,
            0.38,
            0.40,
            0.42,
            0.44,
            0.45,
            0.46,
            0.47,
            0.48,
            0.49,
            0.50,
        }
    )
    best_long_threshold = float(default_threshold)
    best_short_threshold = float(1.0 - default_threshold)
    best_metrics = _classification_quality_metrics(
        pred_probs,
        y_true,
        long_acted_threshold=best_long_threshold,
        short_acted_threshold=best_short_threshold,
        sample_confidence=sample_confidence,
    )
    best_key = (
        1
        if (
            int(best_metrics["long_acted_count"]) >= int(min_long_acted_count)
            and int(best_metrics["short_acted_count"]) >= int(min_short_acted_count)
        )
        else 0,
        1 if int(best_metrics["acted_count"]) >= max(int(min_long_acted_count) + int(min_short_acted_count), 6) else 0,
        min(int(best_metrics["long_acted_count"]), int(best_metrics["short_acted_count"])),
        float(best_metrics["precision_balance_score"]),
        float(best_metrics["acted_accuracy"]),
        float(best_metrics["accuracy_lift_over_majority"]),
        -(
            abs(float(best_long_threshold) - float(default_threshold))
            + abs(float(best_short_threshold) - float(1.0 - default_threshold))
        ),
    )

    for long_threshold in long_candidates:
        for short_threshold in short_candidates:
            if float(short_threshold) > float(long_threshold):
                continue
            metrics = _classification_quality_metrics(
                pred_probs,
                y_true,
                long_acted_threshold=float(long_threshold),
                short_acted_threshold=float(short_threshold),
                sample_confidence=sample_confidence,
            )
            key = (
                1
                if (
                    int(metrics["long_acted_count"]) >= int(min_long_acted_count)
                    and int(metrics["short_acted_count"]) >= int(min_short_acted_count)
                )
                else 0,
                1 if int(metrics["acted_count"]) >= max(int(min_long_acted_count) + int(min_short_acted_count), 6) else 0,
                min(int(metrics["long_acted_count"]), int(metrics["short_acted_count"])),
                float(metrics["precision_balance_score"]),
                float(metrics["acted_accuracy"]),
                float(metrics["accuracy_lift_over_majority"]),
                -(
                    abs(float(long_threshold) - float(default_threshold))
                    + abs(float(short_threshold) - float(1.0 - default_threshold))
                ),
            )
            if key > best_key:
                best_long_threshold = float(long_threshold)
                best_short_threshold = float(short_threshold)
                best_metrics = metrics
                best_key = key

    return best_long_threshold, best_short_threshold, {
        "calibrated": bool(
            (abs(best_long_threshold - default_threshold) > 1e-9)
            or (abs(best_short_threshold - (1.0 - default_threshold)) > 1e-9)
        ),
        "reason": "validation_grid_search",
        "selected_threshold": float(best_long_threshold),
        "selected_long_threshold": float(best_long_threshold),
        "selected_short_threshold": float(best_short_threshold),
        "default_threshold": float(default_threshold),
        "candidate_count": int(len(long_candidates) * len(short_candidates)),
        "validation_metrics": best_metrics,
    }


def _select_calibrated_acted_threshold(
    pred_probs_np: np.ndarray,
    y_true_np: np.ndarray,
    *,
    default_threshold: float,
    sample_confidence: Optional[np.ndarray] = None,
) -> tuple[float, Dict[str, Any]]:
    long_threshold, short_threshold, meta = _select_calibrated_action_thresholds(
        pred_probs_np,
        y_true_np,
        default_threshold=default_threshold,
        sample_confidence=sample_confidence,
    )
    meta = dict(meta)
    meta["selected_threshold"] = float(max(long_threshold, 1.0 - short_threshold))
    return float(meta["selected_threshold"]), meta
