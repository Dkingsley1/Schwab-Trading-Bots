import json
import importlib
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

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
    window: int = 30,
    horizon: int = 3,
    learning_rate: float = 0.0008,
    epochs: int = 220,
    batch_size: int = 128,
    patience: int = 18,
    min_samples: int = 256,
    min_sequences: int = 2,
    acted_prob_threshold: float = 0.65,
    fallback_trainer: Optional[Callable[[], TradingBrain]] = None,
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
    positive_rate = float(runtime_meta.get("positive_rate", 0.0) or 0.0)
    if (
        sample_count < max(int(min_samples), batch_size * 2)
        or int(runtime_meta.get("eligible_sequences", 0) or 0) < max(int(min_sequences), 1)
        or positive_rate <= 0.02
        or positive_rate >= 0.98
    ):
        if fallback_trainer is not None:
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

    feat_mean = X_np.mean(axis=0, keepdims=True)
    feat_std = X_np.std(axis=0, keepdims=True) + 1e-8
    X_np = (X_np - feat_mean) / feat_std

    X = mx.array(X_np, dtype=mx.float32)
    y = mx.array(y_np, dtype=mx.float32)
    X_train, y_train, X_val, y_val, X_test, y_test = split_data(X, y)
    n_train = int(sample_count * 0.7)
    n_val = int(sample_count * 0.15)
    sample_confidence_test = sample_confidence[n_train + n_val :]

    brain = TradingBrain(int(X.shape[1]))
    mx.eval(brain.parameters())

    optimizer = optim.Adam(learning_rate=learning_rate)
    loss_and_grad_fn = nn.value_and_grad(brain, loss_fn)

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

            loss, grads = loss_and_grad_fn(brain, xb, yb)
            optimizer.update(brain, grads)
            mx.eval(brain.parameters(), optimizer.state)

            total_loss += float(loss)
            batches += 1

        val_loss = float(loss_fn(brain, X_val, y_val))
        if epoch % 10 == 0:
            print(
                f"Epoch {epoch} | Train {total_loss / max(batches, 1):.6f} | Val {val_loss:.6f}",
                flush=True,
            )

        if val_loss < best_val:
            best_val = val_loss
            patience_left = patience
        else:
            patience_left -= 1
            if patience_left == 0:
                print("Early stopping.", flush=True)
                break

    preds = mx.sigmoid(brain(X_test))
    pred_labels = (preds > 0.5).astype(mx.float32)
    acc = float(mx.mean((pred_labels == y_test).astype(mx.float32)))
    print(f"Test accuracy: {acc:.4f}")
    pred_probs_np = np.asarray(preds).reshape(-1)
    pred_labels_np = (pred_probs_np > 0.5).astype(np.float32)
    y_test_np = np.asarray(y_test).reshape(-1)
    acted_threshold = float(min(max(acted_prob_threshold, 0.5), 0.95))
    acted_mask = (pred_probs_np >= acted_threshold) | (pred_probs_np <= (1.0 - acted_threshold))
    acted_coverage = float(np.mean(acted_mask.astype(np.float32))) if acted_mask.size else 0.0
    acted_accuracy = (
        float(np.mean((pred_labels_np[acted_mask] == y_test_np[acted_mask]).astype(np.float32)))
        if np.any(acted_mask)
        else 0.0
    )
    long_mask = pred_probs_np >= acted_threshold
    short_mask = pred_probs_np <= (1.0 - acted_threshold)
    long_precision = float(np.mean(y_test_np[long_mask])) if np.any(long_mask) else 0.0
    short_precision = float(np.mean(1.0 - y_test_np[short_mask])) if np.any(short_mask) else 0.0
    pred_confidence = np.abs(pred_probs_np - 0.5) * 2.0

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
            "acted_prob_threshold": float(acted_threshold),
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
        "test_accuracy": float(acc),
        "positive_rate": float(positive_rate),
        "acted_prob_threshold": float(acted_threshold),
        "acted_coverage": float(acted_coverage),
        "acted_accuracy": float(acted_accuracy),
        "long_precision": float(long_precision),
        "short_precision": float(short_precision),
        "pred_confidence_mean": float(np.mean(pred_confidence)) if pred_confidence.size else 0.0,
        "pred_confidence_max": float(np.max(pred_confidence)) if pred_confidence.size else 0.0,
        "input_confidence_mean": float(np.mean(sample_confidence_test)) if sample_confidence_test.size else 0.0,
    }
    save_artifacts(brain, config, metrics, run_tag=run_tag)
    return brain
