import json
import importlib
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from core.ml_backend_contract import detect_installed_backends, resolve_backend_contract

_MLX_IMPORT_ERROR: Optional[Exception] = None
_MLX_AVAILABLE = False
try:
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim

    _MLX_AVAILABLE = True
except Exception as exc:
    mx = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    optim = None  # type: ignore[assignment]
    _MLX_IMPORT_ERROR = exc

from runtime_training_common import (
    RuntimeConfidenceBuilder,
    RuntimeFeatureBuilder,
    RuntimeLabelBuilder,
    RuntimeSampleFilter,
    load_runtime_observation_sequences,
    make_runtime_windowed_dataset,
)


ArrayMap = Dict[str, np.ndarray]


def _ml_runtime_optional_mode() -> bool:
    generic = str(os.getenv("BOT_ML_RUNTIME_OPTIONAL", "")).strip().lower()
    if generic:
        return generic in {"1", "true", "yes", "on"}
    return str(os.getenv("BOT_MLX_OPTIONAL", "0")).strip().lower() in {"1", "true", "yes", "on"}


def _configured_ml_backend() -> str:
    backend = str(os.getenv("BOT_ML_BACKEND", "")).strip().lower().replace("-", "_")
    return backend or "native_default"


def _mlx_optional_mode() -> bool:
    return _ml_runtime_optional_mode()


def _current_backend_contract() -> Dict[str, Any]:
    installed = detect_installed_backends()
    installed["mlx"] = bool(_MLX_AVAILABLE)
    return resolve_backend_contract(
        _configured_ml_backend(),
        mode=str(os.getenv("BOT_RUNTIME_ACCESS_MODE", "native")),
        installed=installed,
    )


def _require_mlx_runtime(context: str = "this operation") -> None:
    if _MLX_AVAILABLE:
        return
    detail = f"MLX is required for {context}"
    ml_backend = _configured_ml_backend()
    backend_contract = _current_backend_contract()
    if _mlx_optional_mode():
        runtime_mode = str(os.getenv("BOT_RUNTIME_ACCESS_MODE", "native")).strip().lower()
        if runtime_mode == "portable":
            detail += (
                f" but portable mode intentionally leaves ML runtimes optional "
                f"(backend={ml_backend}). Install MLX on the target machine or use non-MLX workflows there."
            )
        else:
            detail += f" and the configured ML runtime is optional (backend={ml_backend})."
    supported_roles = list(backend_contract.get("roles_supported") or [])
    if supported_roles and not bool(backend_contract.get("live_trading_supported")):
        detail += (
            f" Backend contract roles={','.join(str(role) for role in supported_roles)}"
            f" are currently observation-only; the live TradingBrain path remains MLX-only."
        )
    if _MLX_IMPORT_ERROR is not None:
        detail += f" Original import error: {_MLX_IMPORT_ERROR}"
    raise RuntimeError(detail) from _MLX_IMPORT_ERROR


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


if _MLX_AVAILABLE:
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
else:
    class TradingBrain:
        def __init__(self, input_dim: int):
            _require_mlx_runtime("TradingBrain initialization")


def loss_fn(model, x, y):
    _require_mlx_runtime("MLX loss evaluation")
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
    _require_mlx_runtime("MLX weighted loss evaluation")
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
    _require_mlx_runtime("windowed dataset tensor conversion")
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
    _require_mlx_runtime("MLX parameter assignment")
    if isinstance(target, dict):
        for key, value in target.items():
            name = f"{prefix}.{key}" if prefix else str(key)
            _assign_param_tree(value, flat, name)
        return
    if prefix in flat:
        target[:] = mx.array(flat[prefix])


def _snapshot_model_params(model: Any) -> Dict[str, np.ndarray]:
    _require_mlx_runtime("MLX parameter snapshot")
    mx.eval(model.parameters())
    flat = _flatten_param_tree(model.parameters())
    return {key: np.array(value, copy=True) for key, value in flat.items()}


def _restore_model_params(model: Any, flat: Dict[str, np.ndarray]) -> None:
    _require_mlx_runtime("MLX parameter restore")
    _assign_param_tree(model.parameters(), flat)
    mx.eval(model.parameters())


def load_model(model, npz_path):
    _require_mlx_runtime("MLX model loading")
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
    _require_mlx_runtime("teacher distillation soft targets")
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


_TRAINING_GUARD_PRESETS: Dict[str, Dict[str, float]] = {
    "intraday": {
        "min_label_balance_score": 0.22,
        "min_acted_coverage": 0.03,
        "max_acted_coverage": 0.48,
    },
    "futures": {
        "min_label_balance_score": 0.18,
        "min_acted_coverage": 0.04,
        "max_acted_coverage": 0.58,
    },
    "options": {
        "min_label_balance_score": 0.16,
        "min_acted_coverage": 0.03,
        "max_acted_coverage": 0.42,
    },
    "dividend": {
        "min_label_balance_score": 0.14,
        "min_acted_coverage": 0.02,
        "max_acted_coverage": 0.30,
    },
    "long_term": {
        "min_label_balance_score": 0.12,
        "min_acted_coverage": 0.01,
        "max_acted_coverage": 0.24,
    },
    "bond": {
        "min_label_balance_score": 0.15,
        "min_acted_coverage": 0.02,
        "max_acted_coverage": 0.32,
    },
    "core": {
        "min_label_balance_score": 0.16,
        "min_acted_coverage": 0.02,
        "max_acted_coverage": 0.42,
    },
}

_TRAINING_PATH_PRESETS: Dict[str, Dict[str, float]] = {
    "intraday": {
        "lookback_days_floor": 60,
        "sample_stride_cap": 2,
        "min_confidence_cap": 0.40,
        "min_samples_cap": 256,
        "min_sequences_cap": 4,
        "min_side_samples_cap": 40,
        "batch_size_cap": 96,
        "patience_floor": 20,
        "epochs_floor": 220,
        "autofix_max_lookback_days": 120,
        "autofix_min_confidence_floor": 0.18,
    },
    "futures": {
        "lookback_days_floor": 75,
        "sample_stride_cap": 2,
        "min_confidence_cap": 0.38,
        "min_samples_cap": 192,
        "min_sequences_cap": 4,
        "min_side_samples_cap": 28,
        "batch_size_cap": 96,
        "patience_floor": 20,
        "epochs_floor": 220,
        "autofix_max_lookback_days": 150,
        "autofix_min_confidence_floor": 0.14,
    },
    "options": {
        "lookback_days_floor": 60,
        "sample_stride_cap": 2,
        "min_confidence_cap": 0.38,
        "min_samples_cap": 192,
        "min_sequences_cap": 4,
        "min_side_samples_cap": 28,
        "batch_size_cap": 96,
        "patience_floor": 20,
        "epochs_floor": 220,
        "autofix_max_lookback_days": 135,
        "autofix_min_confidence_floor": 0.16,
    },
    "dividend": {
        "lookback_days_floor": 90,
        "sample_stride_cap": 1,
        "min_confidence_cap": 0.32,
        "min_samples_cap": 160,
        "min_sequences_cap": 3,
        "min_side_samples_cap": 20,
        "batch_size_cap": 96,
        "patience_floor": 20,
        "epochs_floor": 220,
        "autofix_max_lookback_days": 150,
        "autofix_min_confidence_floor": 0.10,
    },
    "long_term": {
        "lookback_days_floor": 120,
        "sample_stride_cap": 1,
        "min_confidence_cap": 0.30,
        "min_samples_cap": 144,
        "min_sequences_cap": 3,
        "min_side_samples_cap": 18,
        "batch_size_cap": 96,
        "patience_floor": 22,
        "epochs_floor": 240,
        "autofix_max_lookback_days": 180,
        "autofix_min_confidence_floor": 0.08,
    },
    "bond": {
        "lookback_days_floor": 90,
        "sample_stride_cap": 1,
        "min_confidence_cap": 0.32,
        "min_samples_cap": 160,
        "min_sequences_cap": 3,
        "min_side_samples_cap": 20,
        "batch_size_cap": 96,
        "patience_floor": 20,
        "epochs_floor": 220,
        "autofix_max_lookback_days": 150,
        "autofix_min_confidence_floor": 0.10,
    },
    "core": {
        "lookback_days_floor": 75,
        "sample_stride_cap": 2,
        "min_confidence_cap": 0.36,
        "min_samples_cap": 192,
        "min_sequences_cap": 4,
        "min_side_samples_cap": 28,
        "batch_size_cap": 96,
        "patience_floor": 20,
        "epochs_floor": 220,
        "autofix_max_lookback_days": 135,
        "autofix_min_confidence_floor": 0.12,
    },
}

_TRAINING_ROLE_PATH_OVERRIDES: Dict[str, Dict[str, float]] = {
    "signal_sub_bot": {
        "lookback_days_bonus": 0,
        "min_confidence_delta": 0.0,
        "min_samples_multiplier": 1.0,
        "min_side_samples_multiplier": 1.0,
    },
    "infrastructure_sub_bot": {
        "lookback_days_bonus": 14,
        "min_confidence_delta": -0.03,
        "min_samples_multiplier": 0.75,
        "min_side_samples_multiplier": 0.5,
    },
    "options_sub_bot": {
        "lookback_days_bonus": 7,
        "min_confidence_delta": -0.02,
        "min_samples_multiplier": 0.9,
        "min_side_samples_multiplier": 0.85,
    },
    "futures_sub_bot": {
        "lookback_days_bonus": 7,
        "min_confidence_delta": -0.02,
        "min_samples_multiplier": 0.9,
        "min_side_samples_multiplier": 0.85,
    },
}


def _safe_json_load(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _env_bool(name: str, default: bool) -> bool:
    raw = str(os.getenv(name, "")).strip().lower()
    if not raw:
        return bool(default)
    return raw not in {"0", "false", "no", "off"}


def _env_int(name: str, default: int) -> int:
    raw = str(os.getenv(name, "")).strip()
    if not raw:
        return int(default)
    try:
        return int(float(raw))
    except Exception:
        return int(default)


def _env_float(name: str, default: float) -> float:
    raw = str(os.getenv(name, "")).strip()
    if not raw:
        return float(default)
    try:
        return float(raw)
    except Exception:
        return float(default)


def _strategy_bot_id(raw: str) -> str:
    text = str(raw or "").strip()
    if not text:
        return ""
    if "::" in text:
        text = text.split("::", 1)[1]
    return text.strip().lower()


def _load_registry_bot_context(project_root: Path, run_tag: str) -> Dict[str, Any]:
    registry_path = project_root / "master_bot_registry.json"
    payload = _safe_json_load(registry_path)
    rows = payload.get("sub_bots") if isinstance(payload.get("sub_bots"), list) else []
    run_tag_norm = _strategy_bot_id(run_tag)
    for row in rows:
        if not isinstance(row, dict):
            continue
        if _strategy_bot_id(str(row.get("bot_id") or "")) != run_tag_norm:
            continue
        return {
            "bot_id": str(row.get("bot_id") or ""),
            "bot_role": str(row.get("bot_role") or ""),
            "active": bool(row.get("active", False)),
            "lifecycle_state": str(row.get("lifecycle_state") or ""),
            "reason": str(row.get("reason") or ""),
            "promotion_reason": str(row.get("promotion_reason") or ""),
        }
    return {
        "bot_id": run_tag,
        "bot_role": "",
        "active": False,
        "lifecycle_state": "",
        "reason": "",
        "promotion_reason": "",
    }


def _infer_training_family(run_tag: str) -> str:
    tag = str(run_tag or "").strip().lower()
    if not tag:
        return "core"
    if any(tok in tag for tok in ("dividend", "yield_trap", "compounder")):
        return "dividend"
    if any(tok in tag for tok in ("long_interval", "long_term", "core_etf", "quality_compound")):
        return "long_term"
    if any(tok in tag for tok in ("bond", "rates", "treasury", "duration")):
        return "bond"
    if any(tok in tag for tok in ("futures", "order_book", "followthrough", "curve", "basis")):
        return "futures"
    if any(tok in tag for tok in ("options", "iv_", "put_call", "vol_surface", "gamma")):
        return "options"
    if any(tok in tag for tok in ("intraday", "ultrafast", "proxy", "simple", "dmi", "choppy", "news_shocks", "flash")):
        return "intraday"
    return "core"


def _training_path_recent_diagnostic(project_root: Path, run_tag: str) -> Dict[str, Any]:
    diagnostics_path = project_root / "governance" / "training_diagnostics" / f"{run_tag}_latest.json"
    payload = _safe_json_load(diagnostics_path)
    if not payload:
        return {
            "status": "",
            "sample_count": 0,
            "eligible_sequences": 0,
            "sequence_count": 0,
            "observation_count": 0,
            "positive_rate": 0.0,
            "skipped_filtered": 0,
            "skipped_low_confidence": 0,
            "skipped_labels": 0,
            "adaptation_strength": 0.0,
            "diagnostics_path": "",
        }
    status = str(payload.get("status") or "").strip().lower()
    sample_count = int(payload.get("sample_count", 0) or 0)
    eligible_sequences = int(payload.get("eligible_sequences", 0) or 0)
    sequence_count = int(payload.get("sequence_count", 0) or 0)
    observation_count = int(payload.get("observation_count", 0) or 0)
    positive_rate = float(payload.get("positive_rate", 0.0) or 0.0)
    skipped_filtered = int(payload.get("skipped_filtered", 0) or 0)
    skipped_low_confidence = int(payload.get("skipped_low_confidence", 0) or 0)
    skipped_labels = int(payload.get("skipped_labels", 0) or 0)
    starvation_score = 0.0
    if status == "deferred_sample_starved":
        starvation_score += 0.55
    if sample_count <= 0:
        starvation_score += 0.20
    if eligible_sequences <= 0 or sequence_count <= 0:
        starvation_score += 0.15
    if observation_count <= 0:
        starvation_score += 0.10
    if positive_rate <= 0.02 or positive_rate >= 0.98:
        starvation_score += 0.10
    if skipped_low_confidence > max(sample_count, 0):
        starvation_score += 0.10
    if skipped_labels > max(sample_count, 0):
        starvation_score += 0.10
    if skipped_filtered > max(sample_count, 0):
        starvation_score += 0.05
    starvation_score = float(np.clip(starvation_score, 0.0, 1.0))
    return {
        "status": status,
        "sample_count": int(sample_count),
        "eligible_sequences": int(eligible_sequences),
        "sequence_count": int(sequence_count),
        "observation_count": int(observation_count),
        "positive_rate": float(positive_rate),
        "skipped_filtered": int(skipped_filtered),
        "skipped_low_confidence": int(skipped_low_confidence),
        "skipped_labels": int(skipped_labels),
        "adaptation_strength": round(starvation_score, 6),
        "diagnostics_path": str(diagnostics_path) if payload else "",
    }


def _resolve_runtime_training_path_profile(
    run_tag: str,
    *,
    lookback_days: int,
    sample_stride: int,
    min_confidence: float,
    batch_size: int,
    patience: int,
    epochs: int,
    min_samples: int,
    min_sequences: int,
    min_positive_samples: int,
    min_negative_samples: int,
    project_root: Optional[Path] = None,
) -> Dict[str, Any]:
    root = project_root or Path(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    registry_context = _load_registry_bot_context(root, run_tag)
    family = _infer_training_family(run_tag)
    bot_role = str(registry_context.get("bot_role") or "").strip().lower()
    if family == "core" and bot_role == "options_sub_bot":
        family = "options"
    elif family == "core" and bot_role == "futures_sub_bot":
        family = "futures"
    preset = dict(_TRAINING_PATH_PRESETS.get(family, _TRAINING_PATH_PRESETS["core"]))
    role_overlay = dict(_TRAINING_ROLE_PATH_OVERRIDES.get(bot_role, _TRAINING_ROLE_PATH_OVERRIDES["signal_sub_bot"]))
    diagnostic = _training_path_recent_diagnostic(root, run_tag)
    adaptation_strength = float(diagnostic.get("adaptation_strength", 0.0) or 0.0)

    requested_lookback_days = max(int(lookback_days), 1)
    lookback_floor = max(
        int(preset["lookback_days_floor"]) + int(role_overlay.get("lookback_days_bonus", 0) or 0),
        requested_lookback_days,
    )
    if adaptation_strength > 0.0:
        lookback_floor = max(lookback_floor, int(math.ceil(float(lookback_days) + (21.0 * adaptation_strength))))
    lookback_cap = max(_env_int("RUNTIME_TRAIN_LOOKBACK_DAYS_CAP", 0), 0)
    if lookback_cap > 0:
        lookback_floor = max(requested_lookback_days, min(int(lookback_floor), int(lookback_cap)))

    min_conf_cap = max(
        0.0,
        float(preset["min_confidence_cap"]) + float(role_overlay.get("min_confidence_delta", 0.0) or 0.0) - (0.08 * adaptation_strength),
    )
    sample_stride_cap = max(1, int(preset["sample_stride_cap"]))
    min_samples_cap = max(
        64,
        int(
            round(
                float(preset["min_samples_cap"])
                * float(role_overlay.get("min_samples_multiplier", 1.0) or 1.0)
                * (1.0 - (0.25 * adaptation_strength))
            )
        ),
    )
    min_sequences_cap = max(2, int(round(float(preset["min_sequences_cap"]) - (1.0 if adaptation_strength >= 0.50 else 0.0))))
    min_side_samples_cap = max(
        0,
        int(
            round(
                float(preset["min_side_samples_cap"])
                * float(role_overlay.get("min_side_samples_multiplier", 1.0) or 1.0)
                * (1.0 - (0.35 * adaptation_strength))
            )
        ),
    )
    sample_stride_floor = min(
        sample_stride_cap,
        max(1, _env_int("RUNTIME_TRAIN_SAMPLE_STRIDE_FLOOR", 1)),
    )
    retrain_profile = str(os.getenv("RETRAIN_PROFILE", "") or "").strip().lower()
    explicit_stride_override = max(_env_int("RUNTIME_TRAIN_SAMPLE_STRIDE_OVERRIDE", 0), 0)
    if retrain_profile == "coverage_canary":
        canary_stride_floor = max(
            1,
            _env_int(
                "RETRAIN_COVERAGE_CANARY_SAMPLE_STRIDE",
                explicit_stride_override or 1,
            ),
        )
        sample_stride_floor = min(sample_stride_floor, canary_stride_floor)
        if explicit_stride_override > 0:
            sample_stride_floor = min(sample_stride_floor, explicit_stride_override)
    batch_size_cap = max(32, int(preset["batch_size_cap"]))
    batch_size_cap = max(32, min(batch_size_cap, _env_int("RUNTIME_TRAIN_BATCH_SIZE_CAP", batch_size_cap)))
    max_samples_cap = max(0, _env_int("RUNTIME_TRAIN_MAX_SAMPLES", 0))
    patience_floor = max(int(preset["patience_floor"]), int(patience))
    epochs_floor = max(int(preset["epochs_floor"]), int(epochs))
    autofix_max_lookback_days = max(
        int(preset["autofix_max_lookback_days"]),
        int(lookback_floor + 21),
    )
    autofix_min_confidence_floor = max(
        0.0,
        min(float(preset["autofix_min_confidence_floor"]), float(min_conf_cap)),
    )

    applied = {
        "family": family,
        "bot_role": bot_role,
        "registry_context": registry_context,
        "diagnostic_adaptation": diagnostic,
        "lookback_days": int(lookback_floor),
        "sample_stride": int(min(max(int(sample_stride), sample_stride_floor), sample_stride_cap)),
        "min_confidence": float(min(max(float(min_confidence), 0.0), min_conf_cap)),
        "batch_size": int(min(max(int(batch_size), 1), batch_size_cap)),
        "max_samples": int(max_samples_cap),
        "patience": int(patience_floor),
        "epochs": int(epochs_floor),
        "min_samples": int(min(max(int(min_samples), 1), min_samples_cap)),
        "min_sequences": int(min(max(int(min_sequences), 1), min_sequences_cap)),
        "min_positive_samples": int(min(max(int(min_positive_samples), 0), min_side_samples_cap)) if int(min_positive_samples) > 0 else int(min_positive_samples),
        "min_negative_samples": int(min(max(int(min_negative_samples), 0), min_side_samples_cap)) if int(min_negative_samples) > 0 else int(min_negative_samples),
        "autofix_max_lookback_days": int(autofix_max_lookback_days),
        "autofix_min_confidence_floor": float(autofix_min_confidence_floor),
        "explicit_adjustments": {
            "lookback_days_increased": int(lookback_floor) > int(lookback_days),
            "lookback_days_capped": int(lookback_cap) > 0 and int(lookback_floor) < int(max(int(preset["lookback_days_floor"]) + int(role_overlay.get("lookback_days_bonus", 0) or 0), requested_lookback_days)),
            "sample_stride_increased_for_memory": int(min(max(int(sample_stride), sample_stride_floor), sample_stride_cap)) > int(max(int(sample_stride), 1)),
            "sample_stride_reduced": int(min(max(int(sample_stride), sample_stride_floor), sample_stride_cap)) < int(max(int(sample_stride), 1)),
            "min_confidence_reduced": float(min(max(float(min_confidence), 0.0), min_conf_cap)) < float(max(float(min_confidence), 0.0)),
            "batch_size_reduced": int(min(max(int(batch_size), 1), batch_size_cap)) < int(max(int(batch_size), 1)),
            "min_samples_reduced": int(min(max(int(min_samples), 1), min_samples_cap)) < int(max(int(min_samples), 1)),
            "min_sequences_reduced": int(min(max(int(min_sequences), 1), min_sequences_cap)) < int(max(int(min_sequences), 1)),
            "min_positive_samples_reduced": int(min_positive_samples) > 0 and int(min(max(int(min_positive_samples), 0), min_side_samples_cap)) < int(max(int(min_positive_samples), 0)),
            "min_negative_samples_reduced": int(min_negative_samples) > 0 and int(min(max(int(min_negative_samples), 0), min_side_samples_cap)) < int(max(int(min_negative_samples), 0)),
        },
        "memory_efficiency": {
            "profile": str(os.getenv("BOT_MEMORY_EFFICIENCY_PROFILE", "") or ""),
            "lookback_cap": int(lookback_cap),
            "sample_stride_floor": int(sample_stride_floor),
            "batch_size_cap": int(batch_size_cap),
            "max_samples": int(max_samples_cap),
        },
    }
    return applied


def _family_profiles(family: str) -> List[str]:
    fam = str(family or "").strip().lower()
    mapping = {
        "intraday": ["intraday_aggressive", "aggressive", "swing_aggressive"],
        "futures": ["schwab_futures", "crypto_futures", "aggressive"],
        "options": ["intraday_aggressive", "aggressive", "swing_aggressive"],
        "dividend": ["dividend", "conservative"],
        "long_term": ["default", "conservative"],
        "bond": ["bond", "conservative"],
        "core": ["default", "aggressive", "conservative"],
    }
    return list(mapping.get(fam, mapping["core"]))


def _paper_guard_adaptation(project_root: Path, run_tag: str, family: str) -> Dict[str, Any]:
    payload = _safe_json_load(project_root / "governance" / "health" / "paper_performance_latest.json")
    sleeve_rows = payload.get("sleeve_latest") if isinstance(payload.get("sleeve_latest"), list) else []
    relevant_profiles = set(_family_profiles(family))
    matched_profiles: List[str] = []
    weak_profiles: List[str] = []
    weak_profile_score = 0.0
    for sleeve in sleeve_rows:
        if not isinstance(sleeve, dict):
            continue
        profile = str(sleeve.get("profile") or "").strip().lower()
        if not profile or profile not in relevant_profiles:
            continue
        matched_profiles.append(profile)
        ending_net = float(sleeve.get("ending_net_pnl_total", 0.0) or 0.0)
        win_rate = sleeve.get("win_rate")
        losing_count = int(sleeve.get("losing_strategy_count", 0) or 0)
        winning_count = int(sleeve.get("winning_strategy_count", 0) or 0)
        is_weak = ending_net < 0.0 or (win_rate is not None and float(win_rate) < 0.45) or losing_count > winning_count
        if is_weak:
            weak_profiles.append(profile)
            weak_profile_score += abs(min(ending_net, 0.0)) / 20.0
            if win_rate is not None:
                weak_profile_score += max(0.0, 0.55 - float(win_rate))
    hard_negative = _paper_loss_hard_negative_context(project_root, run_tag)
    bot_loss_score = float(hard_negative.get("loss_score", 0.0) or 0.0)
    adaptation_strength = float(
        np.clip(
            (0.08 * len(set(weak_profiles)))
            + min(0.12, weak_profile_score)
            + min(0.10, bot_loss_score / 20.0),
            0.0,
            0.24,
        )
    )
    return {
        "relevant_profiles": sorted(relevant_profiles),
        "matched_profiles": sorted(set(matched_profiles)),
        "weak_profiles": sorted(set(weak_profiles)),
        "weak_profile_count": int(len(set(weak_profiles))),
        "bot_loss_score": float(bot_loss_score),
        "adaptation_strength": round(float(adaptation_strength), 6),
    }


def _load_calibration_abstention_overrides(project_root: Path) -> Dict[str, Any]:
    return _safe_json_load(project_root / "governance" / "health" / "calibration_abstention_overrides_latest.json")


def _resolve_learned_acted_threshold(
    project_root: Path,
    *,
    run_tag: str,
    family: str,
    base_threshold: float,
) -> tuple[float, Dict[str, Any]]:
    payload = _load_calibration_abstention_overrides(project_root)
    bot_overrides = payload.get("bot_overrides") if isinstance(payload.get("bot_overrides"), dict) else {}
    family_overrides = payload.get("family_overrides") if isinstance(payload.get("family_overrides"), dict) else {}
    normalized_bot_id = str(run_tag or "").strip().lower()
    normalized_family = str(family or "").strip().lower()
    adjusted = float(min(max(base_threshold, 0.5), 0.95))
    applied_sources: List[Dict[str, Any]] = []

    family_row = family_overrides.get(normalized_family)
    if isinstance(family_row, dict):
        uplift = float(family_row.get("acted_prob_threshold_uplift", 0.0) or 0.0)
        adjusted = float(min(max(adjusted + uplift, 0.5), 0.95))
        applied_sources.append(
            {
                "scope": "family",
                "id": normalized_family,
                "mode": str(family_row.get("mode") or ""),
                "acted_prob_threshold_uplift": round(uplift, 6),
            }
        )

    bot_row = bot_overrides.get(normalized_bot_id)
    if isinstance(bot_row, dict):
        uplift = float(bot_row.get("acted_prob_threshold_uplift", 0.0) or 0.0)
        adjusted = float(min(max(adjusted + uplift, 0.5), 0.95))
        applied_sources.append(
            {
                "scope": "bot",
                "id": normalized_bot_id,
                "mode": str(bot_row.get("mode") or ""),
                "acted_prob_threshold_uplift": round(uplift, 6),
            }
        )

    meta = {
        "override_file": str(project_root / "governance" / "health" / "calibration_abstention_overrides_latest.json"),
        "base_threshold": round(float(base_threshold), 6),
        "adjusted_threshold": round(float(adjusted), 6),
        "applied_sources": applied_sources,
    }
    return adjusted, meta


def _runtime_training_autofix_plan(
    *,
    lookback_days: int,
    symbol_allowlist: Optional[List[str]],
    min_confidence: float,
    sample_stride: int,
    max_lookback_days: Optional[int] = None,
    min_confidence_floor: Optional[float] = None,
) -> List[Dict[str, Any]]:
    base_lookback = max(int(lookback_days), 1)
    base_stride = max(int(sample_stride), 1)
    configured_confidence_floor = _env_float(
        "RUNTIME_TRAIN_AUTOFIX_MIN_CONFIDENCE_FLOOR",
        float(min_confidence_floor if min_confidence_floor is not None else 0.0),
    )
    confidence_floor = max(0.0, min(configured_confidence_floor, float(min_confidence)))
    max_lookback = max(
        base_lookback,
        _env_int(
            "RUNTIME_TRAIN_AUTOFIX_MAX_LOOKBACK_DAYS",
            int(max_lookback_days if max_lookback_days is not None else max(base_lookback * 3, base_lookback + 14)),
        ),
    )
    allow_symbol_broaden = _env_bool("RUNTIME_TRAIN_AUTOFIX_ALLOW_SYMBOL_SCOPE_BROADEN", True)
    enabled = _env_bool("RUNTIME_TRAIN_AUTOFIX_INSUFFICIENT_DATA", True)

    plan: List[Dict[str, Any]] = [
        {
            "reason": "base",
            "lookback_days": int(base_lookback),
            "symbol_allowlist": list(symbol_allowlist or []),
            "min_confidence": float(min_confidence),
            "sample_stride": int(base_stride),
        }
    ]
    if not enabled:
        return plan

    widened_lookback = min(max_lookback, max(base_lookback + 7, base_lookback * 2))
    if widened_lookback > base_lookback:
        plan.append(
            {
                "reason": "widen_lookback",
                "lookback_days": int(widened_lookback),
                "symbol_allowlist": list(symbol_allowlist or []),
                "min_confidence": float(min_confidence),
                "sample_stride": int(base_stride),
            }
        )
    if base_stride > 1:
        plan.append(
            {
                "reason": "lower_stride",
                "lookback_days": int(widened_lookback),
                "symbol_allowlist": list(symbol_allowlist or []),
                "min_confidence": float(min_confidence),
                "sample_stride": 1,
            }
        )
    if symbol_allowlist and allow_symbol_broaden:
        plan.append(
            {
                "reason": "broaden_symbol_scope",
                "lookback_days": int(widened_lookback),
                "symbol_allowlist": [],
                "min_confidence": float(min_confidence),
                "sample_stride": 1,
            }
        )
    final_lookback = min(max_lookback, max(widened_lookback, base_lookback + 21))
    plan.append(
        {
            "reason": "full_recovery",
            "lookback_days": int(final_lookback),
            "symbol_allowlist": [],
            "min_confidence": float(confidence_floor),
            "sample_stride": 1,
        }
    )

    deduped: List[Dict[str, Any]] = []
    seen: set[tuple[int, tuple[str, ...], float, int]] = set()
    for row in plan:
        symbol_scope = tuple(sorted(str(item).strip().upper() for item in (row.get("symbol_allowlist") or []) if str(item).strip()))
        key = (
            int(row.get("lookback_days", base_lookback)),
            symbol_scope,
            round(float(row.get("min_confidence", min_confidence)), 6),
            int(row.get("sample_stride", base_stride)),
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(row)
    return deduped


def _runtime_data_insufficiency_reason(
    *,
    sample_count: int,
    eligible_sequences: int,
    positive_rate: float,
    batch_size: int,
    min_samples: int,
    min_sequences: int,
) -> str:
    if sample_count < max(int(min_samples), int(batch_size) * 2):
        return "sample_count"
    if int(eligible_sequences) < max(int(min_sequences), 1):
        return "eligible_sequences"
    if positive_rate <= 0.02:
        return "positive_rate_low"
    if positive_rate >= 0.98:
        return "positive_rate_high"
    return ""


def _resolve_training_guard_profile(
    run_tag: str,
    *,
    min_label_balance_score: Optional[float],
    min_acted_coverage: Optional[float],
    max_acted_coverage: Optional[float],
    project_root: Optional[Path] = None,
) -> Dict[str, Any]:
    family = _infer_training_family(run_tag)
    preset = dict(_TRAINING_GUARD_PRESETS.get(family, _TRAINING_GUARD_PRESETS["core"]))
    root = project_root or Path(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    adaptation = _paper_guard_adaptation(root, run_tag, family)
    if adaptation["adaptation_strength"] > 0.0:
        tighten = float(adaptation["adaptation_strength"])
        preset["min_label_balance_score"] = min(0.45, float(preset["min_label_balance_score"]) + (0.16 * tighten))
        preset["min_acted_coverage"] = min(0.12, float(preset["min_acted_coverage"]) + (0.08 * tighten))
        preset["max_acted_coverage"] = max(
            float(preset["min_acted_coverage"]) + 0.08,
            float(preset["max_acted_coverage"]) - (0.18 * tighten),
        )
        if family == "intraday":
            preset["max_acted_coverage"] = max(float(preset["min_acted_coverage"]) + 0.06, float(preset["max_acted_coverage"]) - (0.06 * tighten))
        elif family == "bond":
            preset["min_label_balance_score"] = min(0.38, float(preset["min_label_balance_score"]) + (0.05 * tighten))
            preset["min_acted_coverage"] = min(0.10, float(preset["min_acted_coverage"]) + (0.03 * tighten))
        elif family in {"dividend", "long_term"}:
            preset["max_acted_coverage"] = max(float(preset["min_acted_coverage"]) + 0.05, float(preset["max_acted_coverage"]) - (0.04 * tighten))
    applied = {
        "family": family,
        "preset": family if family in _TRAINING_GUARD_PRESETS else "core",
        "min_label_balance_score": float(
            min_label_balance_score if min_label_balance_score is not None else preset["min_label_balance_score"]
        ),
        "min_acted_coverage": float(
            min_acted_coverage if min_acted_coverage is not None else preset["min_acted_coverage"]
        ),
        "max_acted_coverage": float(
            max_acted_coverage if max_acted_coverage is not None else preset["max_acted_coverage"]
        ),
        "explicit_overrides": {
            "min_label_balance_score": min_label_balance_score is not None,
            "min_acted_coverage": min_acted_coverage is not None,
            "max_acted_coverage": max_acted_coverage is not None,
        },
        "adaptive_from_live_behavior": adaptation,
    }
    return applied


def _deferred_sample_starved_reason(
    *,
    run_tag: str,
    project_root: Path,
    sample_count: int,
    eligible_sequences: int,
    positive_rate: float,
    autofix_attempts: List[Dict[str, Any]],
    extra_diagnostics: Optional[Dict[str, Any]] = None,
) -> str:
    recommended_retry = autofix_attempts[-1] if autofix_attempts else {}
    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "status": "deferred_sample_starved",
        "family": _infer_training_family(run_tag),
        "sample_count": int(sample_count),
        "eligible_sequences": int(eligible_sequences),
        "positive_rate": round(float(positive_rate), 6),
        "autofix_attempts": list(autofix_attempts),
        "recommended_retry": dict(recommended_retry),
        "failure_categories": ["sample_starved", "defer_until_more_data"],
    }
    if isinstance(extra_diagnostics, dict):
        payload.update({str(k): v for k, v in extra_diagnostics.items()})
    diagnostics_path = _write_runtime_training_diagnostics(
        project_root,
        run_tag,
        payload,
    )
    return (
        f"defer_runtime_training_until_more_data run_tag={run_tag} "
        f"samples={sample_count} eligible_sequences={eligible_sequences} "
        f"positive_rate={positive_rate:.4f} diagnostics_path={diagnostics_path} "
        f"recommended_retry={json.dumps(recommended_retry, ensure_ascii=True)}"
    )


def _paper_loss_hard_negative_context(project_root: Path, run_tag: str) -> Dict[str, Any]:
    paper_path = project_root / "governance" / "health" / "paper_performance_latest.json"
    payload = _safe_json_load(paper_path)
    hard_pack_path = project_root / "governance" / "training_diagnostics" / "paper_hard_examples_latest.json"
    hard_pack = _safe_json_load(hard_pack_path)
    sleeve_rows = payload.get("sleeve_latest") if isinstance(payload.get("sleeve_latest"), list) else []
    run_tag_norm = str(run_tag or "").strip().lower()
    matched_profiles: List[str] = []
    top_loss_pnl = 0.0
    weak_sleeve_count = 0
    matched_trade_count = 0
    for sleeve in sleeve_rows:
        if not isinstance(sleeve, dict):
            continue
        profile = str(sleeve.get("profile") or "").strip().lower()
        ending_net = float(sleeve.get("ending_net_pnl_total", 0.0) or 0.0)
        win_rate = sleeve.get("win_rate")
        is_weak = ending_net < 0.0 or (win_rate is not None and float(win_rate) < 0.45)
        if is_weak:
            weak_sleeve_count += 1
        losing_rows = sleeve.get("top_losing_strategies") if isinstance(sleeve.get("top_losing_strategies"), list) else []
        for row in losing_rows:
            if not isinstance(row, dict):
                continue
            bot_id = _strategy_bot_id(str(row.get("strategy") or ""))
            if bot_id != run_tag_norm:
                continue
            matched_profiles.append(profile)
            top_loss_pnl = min(top_loss_pnl, float(row.get("ending_net_pnl_total", 0.0) or 0.0))
            matched_trade_count += 1
    for row in hard_pack.get("strategies") or []:
        if not isinstance(row, dict):
            continue
        bot_id = _strategy_bot_id(str(row.get("strategy") or ""))
        if bot_id != run_tag_norm:
            continue
        profile = str(row.get("profile") or "").strip().lower()
        if profile:
            matched_profiles.append(profile)
        top_loss_pnl = min(top_loss_pnl, float(row.get("ending_net_pnl_total", 0.0) or 0.0))
        matched_trade_count += int(row.get("trade_count", 0) or 0)
    loss_score = abs(float(top_loss_pnl))
    multiplier = float(
        np.clip(
            1.0
            + min(loss_score / 10.0, 0.65)
            + (0.05 * len(set(matched_profiles)))
            + min(float(matched_trade_count) / 50.0, 0.15),
            1.0,
            1.95,
        )
    )
    return {
        "enabled": bool(loss_score > 0.0 or matched_profiles),
        "matched_profiles": sorted(set(matched_profiles)),
        "loss_score": float(loss_score),
        "weak_sleeve_count": int(weak_sleeve_count),
        "matched_trade_count": int(matched_trade_count),
        "weight_multiplier": float(multiplier),
        "hard_example_pack_path": str(hard_pack_path) if hard_pack else "",
    }


def _write_runtime_training_diagnostics(project_root: Path, run_tag: str, payload: Dict[str, Any]) -> str:
    out_dir = project_root / "governance" / "training_diagnostics"
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    timestamped = out_dir / f"{run_tag}_{ts}.json"
    latest = out_dir / f"{run_tag}_latest.json"
    text = json.dumps(payload, ensure_ascii=True, indent=2)
    timestamped.write_text(text, encoding="utf-8")
    latest.write_text(text, encoding="utf-8")
    return str(latest)


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
    _require_mlx_runtime(f"training {run_tag}")
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
    _require_mlx_runtime(f"training {run_tag}")
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
    min_acted_coverage: Optional[float] = None,
    max_acted_coverage: Optional[float] = None,
    walk_forward_folds: int = 3,
    hard_negative_mining: bool = True,
) -> TradingBrain:
    _require_mlx_runtime(f"runtime training {run_tag}")
    np.random.seed(42)
    project_root = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    training_path = _resolve_runtime_training_path_profile(
        run_tag,
        lookback_days=lookback_days,
        sample_stride=sample_stride,
        min_confidence=min_confidence,
        batch_size=batch_size,
        patience=patience,
        epochs=epochs,
        min_samples=min_samples,
        min_sequences=min_sequences,
        min_positive_samples=min_positive_samples,
        min_negative_samples=min_negative_samples,
        project_root=project_root,
    )
    lookback_days = int(training_path["lookback_days"])
    sample_stride = int(training_path["sample_stride"])
    min_confidence = float(training_path["min_confidence"])
    batch_size = int(training_path["batch_size"])
    max_samples = int(training_path.get("max_samples", 0) or 0)
    patience = int(training_path["patience"])
    epochs = int(training_path["epochs"])
    min_samples = int(training_path["min_samples"])
    min_sequences = int(training_path["min_sequences"])
    min_positive_samples = int(training_path["min_positive_samples"])
    min_negative_samples = int(training_path["min_negative_samples"])
    guard_profile = _resolve_training_guard_profile(
        run_tag,
        min_label_balance_score=min_label_balance_score,
        min_acted_coverage=min_acted_coverage,
        max_acted_coverage=max_acted_coverage,
        project_root=project_root,
    )
    min_label_balance_score = float(guard_profile["min_label_balance_score"])
    min_acted_coverage = float(guard_profile["min_acted_coverage"])
    max_acted_coverage = float(guard_profile["max_acted_coverage"])
    hard_negative_context = _paper_loss_hard_negative_context(project_root, run_tag) if hard_negative_mining else {
        "enabled": False,
        "matched_profiles": [],
        "loss_score": 0.0,
        "weak_sleeve_count": 0,
        "weight_multiplier": 1.0,
    }
    try:
        env_lookback_override = int(float(os.getenv("RUNTIME_TRAIN_LOOKBACK_DAYS_OVERRIDE", "").strip() or 0))
    except ValueError:
        env_lookback_override = 0
    env_stride_override = max(_env_int("RUNTIME_TRAIN_SAMPLE_STRIDE_OVERRIDE", int(sample_stride)), 1)
    env_min_confidence_override = _env_float("RUNTIME_TRAIN_MIN_CONFIDENCE_OVERRIDE", float(min_confidence))
    fast_fail_zero_sample_attempts = max(_env_int("RUNTIME_TRAIN_FAST_FAIL_ZERO_SAMPLE_ATTEMPTS", 0), 0)
    autofix_attempts: List[Dict[str, Any]] = []
    autofix_plan = _runtime_training_autofix_plan(
        lookback_days=max(int(lookback_days), int(env_lookback_override or 0)),
        symbol_allowlist=symbol_allowlist,
        min_confidence=min(max(float(min_confidence), 0.0), max(float(env_min_confidence_override), 0.0)),
        sample_stride=env_stride_override,
        max_lookback_days=int(training_path["autofix_max_lookback_days"]),
        min_confidence_floor=float(training_path["autofix_min_confidence_floor"]),
    )
    X_np = np.zeros((0, 0), dtype=np.float32)
    y_np = np.zeros((0, 1), dtype=np.float32)
    runtime_meta: Dict[str, Any] = {}
    effective_lookback_days = max(int(lookback_days), int(env_lookback_override or 0))
    effective_symbol_allowlist = list(symbol_allowlist or [])
    effective_min_confidence = float(min_confidence)
    effective_sample_stride = int(env_stride_override)
    insufficiency_reason = ""
    last_sequence_count = 0
    last_observation_count = 0

    for attempt_index, attempt in enumerate(autofix_plan):
        effective_lookback_days = max(int(attempt.get("lookback_days", lookback_days) or lookback_days), int(env_lookback_override or 0))
        effective_symbol_allowlist = list(attempt.get("symbol_allowlist") or [])
        effective_min_confidence = float(attempt.get("min_confidence", min_confidence) or 0.0)
        effective_sample_stride = max(int(attempt.get("sample_stride", sample_stride) or sample_stride), 1)
        print(
            "[RuntimeTraining] loading_sequences "
            f"run_tag={run_tag} lookback_days={effective_lookback_days} "
            f"autofix_attempt={attempt_index} reason={attempt.get('reason', 'base')}",
            flush=True,
        )
        sequences = load_runtime_observation_sequences(
            project_root,
            lookback_days=effective_lookback_days,
            mode_allowlist=mode_allowlist,
            symbol_allowlist=effective_symbol_allowlist or None,
        )
        sequence_count = len(sequences)
        observation_count = sum(len(rows) for rows in sequences.values())
        last_sequence_count = int(sequence_count)
        last_observation_count = int(observation_count)
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
            min_confidence=effective_min_confidence,
            sample_stride=effective_sample_stride,
            max_samples=max_samples,
            window=window,
            horizon=horizon,
        )

        sample_count = int(X_np.shape[0]) if X_np.ndim == 2 else 0
        positive_rate = float(runtime_meta.get("positive_rate", 0.0) or 0.0)
        insufficiency_reason = _runtime_data_insufficiency_reason(
            sample_count=sample_count,
            eligible_sequences=int(runtime_meta.get("eligible_sequences", 0) or 0),
            positive_rate=positive_rate,
            batch_size=batch_size,
            min_samples=min_samples,
            min_sequences=min_sequences,
        )
        autofix_attempts.append(
            {
                "attempt_index": int(attempt_index),
                "reason": str(attempt.get("reason") or "base"),
                "lookback_days": int(effective_lookback_days),
                "symbol_allowlist": list(effective_symbol_allowlist or []),
                "min_confidence": float(effective_min_confidence),
                "sample_stride": int(effective_sample_stride),
                "sequence_count": int(sequence_count),
                "observation_count": int(observation_count),
                "samples": int(sample_count),
                "eligible_sequences": int(runtime_meta.get("eligible_sequences", 0) or 0),
                "positive_rate": round(float(positive_rate), 6),
                "skipped_filtered": int(runtime_meta.get("skipped_filtered", 0) or 0),
                "skipped_low_confidence": int(runtime_meta.get("skipped_low_confidence", 0) or 0),
                "skipped_labels": int(runtime_meta.get("skipped_labels", 0) or 0),
                "insufficiency_reason": insufficiency_reason,
            }
        )
        print(
            "[RuntimeTraining] dataset_ready "
            f"run_tag={run_tag} samples={sample_count} "
            f"eligible_sequences={runtime_meta.get('eligible_sequences', 0)} "
            f"positive_rate={positive_rate:.4f} "
            f"insufficiency_reason={insufficiency_reason or 'ok'}",
            flush=True,
        )
        if (
            insufficiency_reason
            and fast_fail_zero_sample_attempts > 0
            and sample_count == 0
            and int(runtime_meta.get("eligible_sequences", 0) or 0) == 0
            and (attempt_index + 1) >= fast_fail_zero_sample_attempts
        ):
            insufficiency_reason = f"{insufficiency_reason}_fast_fail_zero_sample"
            print(
                "[RuntimeTraining] fast_fail_zero_sample "
                f"run_tag={run_tag} attempts={attempt_index + 1} "
                f"reason={attempt.get('reason', 'base')}",
                flush=True,
            )
            break
        if not insufficiency_reason:
            break

    sample_count = int(X_np.shape[0]) if X_np.ndim == 2 else 0
    sample_confidence = np.asarray(
        runtime_meta.pop("_sample_confidence", np.ones((sample_count,), dtype=np.float32)),
        dtype=np.float32,
    ).reshape(-1)
    labels_np = np.asarray(y_np).reshape(-1)
    positive_rate = float(runtime_meta.get("positive_rate", 0.0) or 0.0)
    positive_samples = int(np.sum(labels_np >= 0.5))
    negative_samples = int(sample_count - positive_samples)
    if insufficiency_reason:
        if fallback_trainer is not None and allow_fallback_on_insufficient_data:
            print(
                "[RuntimeTraining] fallback "
                f"run_tag={run_tag} samples={sample_count} "
                f"eligible_sequences={runtime_meta.get('eligible_sequences', 0)} "
                f"positive_rate={positive_rate:.4f}",
                flush=True,
            )
            return fallback_trainer()
        if _env_bool("RUNTIME_TRAIN_DEFER_SAMPLE_STARVED", True):
            raise RuntimeError(
                _deferred_sample_starved_reason(
                    run_tag=run_tag,
                    project_root=project_root,
                    sample_count=sample_count,
                    eligible_sequences=int(runtime_meta.get("eligible_sequences", 0) or 0),
                    positive_rate=positive_rate,
                    autofix_attempts=autofix_attempts,
                    extra_diagnostics={
                        "training_path": training_path,
                        "sequence_count": int(runtime_meta.get("sequence_count", last_sequence_count) or last_sequence_count),
                        "observation_count": int(last_observation_count),
                        "mode_allowlist": list(mode_allowlist or []),
                        "symbol_allowlist": list(effective_symbol_allowlist or []),
                        "lookback_days": int(effective_lookback_days),
                        "min_confidence": float(effective_min_confidence),
                        "sample_stride": int(effective_sample_stride),
                        "skipped_filtered": int(runtime_meta.get("skipped_filtered", 0) or 0),
                        "skipped_low_confidence": int(runtime_meta.get("skipped_low_confidence", 0) or 0),
                        "skipped_labels": int(runtime_meta.get("skipped_labels", 0) or 0),
                    },
                )
            )
        raise RuntimeError(
            f"insufficient_runtime_training_data run_tag={run_tag} "
            f"samples={sample_count} eligible_sequences={runtime_meta.get('eligible_sequences', 0)} "
            f"positive_rate={positive_rate:.4f} "
            f"autofix_attempts={json.dumps(autofix_attempts, ensure_ascii=True)}"
        )
    if positive_samples < max(int(min_positive_samples), 0) or negative_samples < max(int(min_negative_samples), 0):
        if fallback_trainer is not None and allow_fallback_on_insufficient_data:
            return fallback_trainer()
        if _env_bool("RUNTIME_TRAIN_DEFER_SAMPLE_STARVED", True):
            raise RuntimeError(
                _deferred_sample_starved_reason(
                    run_tag=run_tag,
                    project_root=project_root,
                    sample_count=sample_count,
                    eligible_sequences=int(runtime_meta.get("eligible_sequences", 0) or 0),
                    positive_rate=positive_rate,
                    autofix_attempts=autofix_attempts,
                    extra_diagnostics={
                        "training_path": training_path,
                        "sequence_count": int(runtime_meta.get("sequence_count", last_sequence_count) or last_sequence_count),
                        "observation_count": int(last_observation_count),
                        "mode_allowlist": list(mode_allowlist or []),
                        "symbol_allowlist": list(effective_symbol_allowlist or []),
                        "lookback_days": int(effective_lookback_days),
                        "min_confidence": float(effective_min_confidence),
                        "sample_stride": int(effective_sample_stride),
                        "skipped_filtered": int(runtime_meta.get("skipped_filtered", 0) or 0),
                        "skipped_low_confidence": int(runtime_meta.get("skipped_low_confidence", 0) or 0),
                        "skipped_labels": int(runtime_meta.get("skipped_labels", 0) or 0),
                    },
                )
            )
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
    if hard_negative_context.get("enabled"):
        neg_train_mask = (np.asarray(y_train).reshape(-1) < 0.5)
        neg_val_mask = (np.asarray(y_val).reshape(-1) < 0.5)
        negative_multiplier = float(hard_negative_context.get("weight_multiplier", 1.0) or 1.0)
        train_sample_weights = np.where(neg_train_mask, train_sample_weights * negative_multiplier, train_sample_weights)
        val_sample_weights = np.where(neg_val_mask, val_sample_weights * negative_multiplier, val_sample_weights)
        train_sample_weights = np.clip(train_sample_weights, 0.25, 2.0).astype(np.float32)
        val_sample_weights = np.clip(val_sample_weights, 0.25, 2.0).astype(np.float32)

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
    configured_acted_threshold, threshold_override_meta = _resolve_learned_acted_threshold(
        project_root,
        run_tag=run_tag,
        family=guard_profile["family"],
        base_threshold=configured_acted_threshold,
    )
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
        min_long_precision=min_long_precision,
        min_short_precision=min_short_precision,
        require_both_sides_precision=require_both_sides_precision,
        min_acted_accuracy=min_acted_accuracy,
        min_accuracy_lift_over_majority=min_accuracy_lift_over_majority,
        min_precision_balance_score=min_precision_balance_score,
        min_acted_coverage=min_acted_coverage,
        max_acted_coverage=max_acted_coverage,
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
    acted_coverage = float(quality_metrics["acted_coverage"])
    long_precision = float(quality_metrics["long_precision"])
    short_precision = float(quality_metrics["short_precision"])
    accuracy_lift_over_majority = float(quality_metrics["accuracy_lift_over_majority"])
    label_balance_score = float(quality_metrics["label_balance_score"])
    precision_balance_score = float(quality_metrics["precision_balance_score"])
    walk_forward_summary = _walk_forward_multi_split_summary(
        pred_probs_np,
        y_test_np,
        long_acted_threshold=long_acted_threshold,
        short_acted_threshold=short_acted_threshold,
        sample_confidence=sample_confidence_test,
        positive_rate=positive_rate,
        requested_folds=walk_forward_folds,
    )
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
            "symbol_allowlist": list(effective_symbol_allowlist or []),
            "sample_filter_active": bool(sample_filter is not None),
            "confidence_builder_active": bool(confidence_builder is not None),
            "min_confidence": float(effective_min_confidence),
            "sample_stride": int(effective_sample_stride),
            "acted_prob_threshold": float(long_acted_threshold),
            "short_acted_prob_threshold": float(short_acted_threshold),
            "configured_acted_prob_threshold": float(configured_acted_threshold),
            "acted_threshold_calibration": threshold_meta,
            "learned_threshold_override": threshold_override_meta,
            "guard_profile": guard_profile,
            "hard_negative_mining": hard_negative_context,
            "training_path": training_path,
            "autofix_attempts": autofix_attempts,
            "autofix_selected": (autofix_attempts[-1] if autofix_attempts else {}),
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
        "walk_forward_multi_split": walk_forward_summary,
        **quality_metrics,
    }
    threshold_rows = {
        "best_val_loss": max_best_val_loss,
        "final_val_loss": max_final_val_loss,
        "long_precision": min_long_precision,
        "short_precision": min_short_precision,
        "acted_accuracy": min_acted_accuracy,
        "accuracy_lift_over_majority": min_accuracy_lift_over_majority,
        "label_balance_score": min_label_balance_score,
        "precision_balance_score": min_precision_balance_score,
        "acted_coverage": min_acted_coverage,
        "acted_coverage_max": max_acted_coverage,
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
    if min_acted_coverage is not None and float(acted_coverage) < float(min_acted_coverage):
        quality_failures.append(
            f"acted_coverage={float(acted_coverage):.4f} < min_acted_coverage={float(min_acted_coverage):.4f}"
        )
    if max_acted_coverage is not None and float(acted_coverage) > float(max_acted_coverage):
        quality_failures.append(
            f"acted_coverage={float(acted_coverage):.4f} > max_acted_coverage={float(max_acted_coverage):.4f}"
        )
    if float(precision_balance_score) < float(min_precision_balance_score):
        quality_failures.append(
            "precision_balance_score="
            f"{float(precision_balance_score):.4f} < min_precision_balance_score={float(min_precision_balance_score):.4f}"
        )
    failure_rows = _build_failure_rows(
        {
            "best_val_loss": float(best_val),
            "final_val_loss": float(val_loss),
            **quality_metrics,
        },
        threshold_rows,
    )
    diagnostics_payload = {
        "timestamp_utc": datetime.utcnow().isoformat() + "Z",
        "run_tag": run_tag,
        "status": "failed" if quality_failures else "trained",
        "family": guard_profile["family"],
        "guard_profile": guard_profile,
        "quality_failures": quality_failures,
        "failure_rows": failure_rows,
        "failure_categories": _quality_failure_categories(failure_rows),
        "hard_negative_mining": hard_negative_context,
        "runtime_meta": config["runtime"],
        "metrics": metrics,
    }
    diagnostics_path = _write_runtime_training_diagnostics(project_root, run_tag, diagnostics_payload)
    metrics["training_diagnostics_path"] = diagnostics_path
    metrics["training_failure_categories"] = diagnostics_payload["failure_categories"]
    if quality_failures:
        raise RuntimeError(
            f"runtime_training_quality_guard_failed run_tag={run_tag} "
            + "; ".join(quality_failures)
            + f"; diagnostics_path={diagnostics_path}"
        )
    save_artifacts(brain, config, metrics, run_tag=run_tag)
    return brain


def _walk_forward_multi_split_summary(
    pred_probs_np: np.ndarray,
    y_true_np: np.ndarray,
    *,
    long_acted_threshold: float,
    short_acted_threshold: float,
    sample_confidence: Optional[np.ndarray] = None,
    positive_rate: Optional[float] = None,
    requested_folds: int = 3,
) -> Dict[str, Any]:
    pred_probs = np.asarray(pred_probs_np, dtype=np.float32).reshape(-1)
    y_true = np.asarray(y_true_np, dtype=np.float32).reshape(-1)
    sample_conf = np.asarray(sample_confidence, dtype=np.float32).reshape(-1) if sample_confidence is not None else np.zeros((0,), dtype=np.float32)
    fold_count = min(max(int(requested_folds), 1), 5)
    if pred_probs.size < max(fold_count * 8, 24):
        return {
            "enabled": False,
            "requested_folds": int(fold_count),
            "reason": "insufficient_samples",
            "sample_count": int(pred_probs.size),
            "folds": [],
        }

    boundaries = np.linspace(0, pred_probs.size, num=fold_count + 1, dtype=np.int64)
    folds: List[Dict[str, Any]] = []
    test_accuracy_values: List[float] = []
    acted_accuracy_values: List[float] = []
    for idx in range(fold_count):
        start = int(boundaries[idx])
        end = int(boundaries[idx + 1])
        if end - start <= 0:
            continue
        fold_metrics = _classification_quality_metrics(
            pred_probs[start:end],
            y_true[start:end],
            long_acted_threshold=long_acted_threshold,
            short_acted_threshold=short_acted_threshold,
            sample_confidence=sample_conf[start:end] if sample_conf.size == pred_probs.size else None,
            positive_rate=positive_rate,
        )
        folds.append(
            {
                "fold_index": idx,
                "start_offset": start,
                "end_offset": end,
                "sample_count": int(end - start),
                "test_accuracy": round(float(fold_metrics["test_accuracy"]), 6),
                "acted_accuracy": round(float(fold_metrics["acted_accuracy"]), 6),
                "acted_coverage": round(float(fold_metrics["acted_coverage"]), 6),
                "accuracy_lift_over_majority": round(float(fold_metrics["accuracy_lift_over_majority"]), 6),
            }
        )
        test_accuracy_values.append(float(fold_metrics["test_accuracy"]))
        acted_accuracy_values.append(float(fold_metrics["acted_accuracy"]))

    return {
        "enabled": bool(folds),
        "requested_folds": int(fold_count),
        "fold_count": int(len(folds)),
        "sample_count": int(pred_probs.size),
        "test_accuracy_mean": round(float(np.mean(test_accuracy_values)) if test_accuracy_values else 0.0, 6),
        "test_accuracy_std": round(float(np.std(test_accuracy_values)) if test_accuracy_values else 0.0, 6),
        "acted_accuracy_mean": round(float(np.mean(acted_accuracy_values)) if acted_accuracy_values else 0.0, 6),
        "acted_accuracy_std": round(float(np.std(acted_accuracy_values)) if acted_accuracy_values else 0.0, 6),
        "folds": folds,
        "evaluation_source": "chronological_test_tail",
    }


def _quality_failure_categories(failure_rows: List[Dict[str, Any]]) -> List[str]:
    categories: List[str] = []
    metrics = {str(row.get("metric") or ""): row for row in failure_rows if isinstance(row, dict)}
    if "label_balance_score" in metrics:
        categories.append("label_cleanup")
    if any(metric in metrics for metric in ("acted_accuracy", "long_precision", "short_precision", "precision_balance_score")):
        categories.append("threshold_calibration")
    if any(metric in metrics for metric in ("best_val_loss", "final_val_loss")):
        categories.append("symbol_narrowing")
    if "acted_coverage" in metrics:
        categories.append("acted_coverage_tuning")
    if any(metric in metrics for metric in ("accuracy_lift_over_majority",)):
        categories.append("family_guard_review")
    return categories


def _build_failure_rows(
    quality_metrics: Dict[str, Any],
    thresholds: Dict[str, Optional[float]],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for metric, threshold in thresholds.items():
        if threshold is None:
            continue
        actual_key = "acted_coverage" if metric == "acted_coverage_max" else metric
        actual = quality_metrics.get(actual_key)
        try:
            actual_float = float(actual)
            threshold_float = float(threshold)
        except Exception:
            continue
        row: Dict[str, Any] = {
            "metric": metric,
            "actual": round(actual_float, 6),
            "threshold": round(threshold_float, 6),
            "slack": round(actual_float - threshold_float, 6),
        }
        if metric == "acted_coverage_max":
            row["metric"] = "acted_coverage"
            row["direction"] = "max"
            row["slack"] = round(threshold_float - float(quality_metrics.get("acted_coverage", 0.0) or 0.0), 6)
        else:
            row["direction"] = "min"
        if row["direction"] == "min" and actual_float < threshold_float:
            rows.append(row)
        elif row["direction"] == "max" and float(quality_metrics.get("acted_coverage", 0.0) or 0.0) > threshold_float:
            rows.append(row)
    return rows


def _classification_quality_metrics(
    pred_probs_np: np.ndarray,
    y_true_np: np.ndarray,
    *,
    acted_threshold: float = 0.65,
    long_acted_threshold: Optional[float] = None,
    short_acted_threshold: Optional[float] = None,
    sample_confidence: Optional[np.ndarray] = None,
    positive_rate: Optional[float] = None,
) -> Dict[str, Any]:
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
    clipped_probs = np.clip(pred_probs.astype(np.float64), 1e-6, 1.0 - 1e-6)
    brier_score = float(np.mean((clipped_probs - y_true.astype(np.float64)) ** 2)) if y_true.size else 0.0
    expected_calibration_error = 0.0
    calibration_bins: List[Dict[str, Any]] = []
    confidence_deciles: List[Dict[str, Any]] = []
    if clipped_probs.size:
        bin_edges = np.linspace(0.0, 1.0, num=11, dtype=np.float64)
        for bin_idx in range(10):
            if bin_idx == 9:
                mask = (clipped_probs >= bin_edges[bin_idx]) & (clipped_probs <= bin_edges[bin_idx + 1])
            else:
                mask = (clipped_probs >= bin_edges[bin_idx]) & (clipped_probs < bin_edges[bin_idx + 1])
            count = int(np.sum(mask))
            if count <= 0:
                continue
            avg_prob = float(np.mean(clipped_probs[mask]))
            avg_label = float(np.mean(y_true[mask]))
            accuracy = float(np.mean(((clipped_probs[mask] > 0.5).astype(np.float32) == y_true[mask]).astype(np.float32)))
            bin_weight = count / max(int(clipped_probs.size), 1)
            expected_calibration_error += abs(avg_prob - avg_label) * bin_weight
            row = {
                "bin_start": round(float(bin_edges[bin_idx]), 4),
                "bin_end": round(float(bin_edges[bin_idx + 1]), 4),
                "count": count,
                "avg_prob": round(avg_prob, 6),
                "avg_label": round(avg_label, 6),
                "accuracy": round(accuracy, 6),
            }
            calibration_bins.append(row)
            confidence_deciles.append(row)

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
        "brier_score": float(brier_score),
        "expected_calibration_error": float(expected_calibration_error),
        "calibration_bins": calibration_bins,
        "confidence_deciles": confidence_deciles,
    }


def _select_calibrated_action_thresholds(
    pred_probs_np: np.ndarray,
    y_true_np: np.ndarray,
    *,
    default_threshold: float,
    sample_confidence: Optional[np.ndarray] = None,
    min_long_acted_count: int = 2,
    min_short_acted_count: int = 2,
    min_long_precision: float = 0.0,
    min_short_precision: float = 0.0,
    require_both_sides_precision: bool = False,
    min_acted_accuracy: float = 0.0,
    min_accuracy_lift_over_majority: Optional[float] = None,
    min_precision_balance_score: float = 0.0,
    min_acted_coverage: Optional[float] = None,
    max_acted_coverage: Optional[float] = None,
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
            0.74,
            0.76,
            0.78,
            0.80,
            0.82,
            0.84,
            0.86,
            0.88,
            0.90,
            0.92,
            0.94,
        }
    )
    short_candidates = sorted(
        {
            float(1.0 - default_threshold),
            0.06,
            0.08,
            0.10,
            0.12,
            0.15,
            0.18,
            0.20,
            0.22,
            0.24,
            0.26,
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

    def _candidate_key(metrics: Dict[str, Any], long_threshold: float, short_threshold: float) -> tuple:
        long_count = int(metrics["long_acted_count"])
        short_count = int(metrics["short_acted_count"])
        acted_coverage = float(metrics["acted_coverage"])
        counts_ok = long_count >= int(min_long_acted_count) and short_count >= int(min_short_acted_count)
        long_precision_ok = float(metrics["long_precision"]) >= float(min_long_precision)
        short_precision_ok = float(metrics["short_precision"]) >= float(min_short_precision)
        both_sides_ok = (not require_both_sides_precision) or (long_count > 0 and short_count > 0)
        precision_ok = long_precision_ok and short_precision_ok and both_sides_ok
        acted_accuracy_ok = float(metrics["acted_accuracy"]) >= float(min_acted_accuracy)
        lift_ok = (
            True
            if min_accuracy_lift_over_majority is None
            else float(metrics["accuracy_lift_over_majority"]) >= float(min_accuracy_lift_over_majority)
        )
        balance_ok = float(metrics["precision_balance_score"]) >= float(min_precision_balance_score)
        coverage_ok = True
        if min_acted_coverage is not None and acted_coverage < float(min_acted_coverage):
            coverage_ok = False
        if max_acted_coverage is not None and acted_coverage > float(max_acted_coverage):
            coverage_ok = False
        if min_acted_coverage is not None and max_acted_coverage is not None:
            coverage_target = 0.5 * (float(min_acted_coverage) + float(max_acted_coverage))
        elif max_acted_coverage is not None:
            coverage_target = min(float(max_acted_coverage), 0.18)
        elif min_acted_coverage is not None:
            coverage_target = max(float(min_acted_coverage), 0.10)
        else:
            coverage_target = 0.18
        coverage_distance = abs(acted_coverage - coverage_target)
        dominant_precision = min(
            float(metrics["long_precision"]) if require_both_sides_precision else 1.0,
            float(metrics["short_precision"]),
        )
        return (
            1 if (counts_ok and precision_ok and acted_accuracy_ok and lift_ok and balance_ok and coverage_ok) else 0,
            1 if counts_ok and coverage_ok else 0,
            1 if precision_ok and coverage_ok else 0,
            1 if counts_ok else 0,
            1 if precision_ok else 0,
            1 if coverage_ok else 0,
            1 if acted_accuracy_ok else 0,
            1 if lift_ok else 0,
            1 if balance_ok else 0,
            -coverage_distance,
            -acted_coverage,
            float(metrics["acted_accuracy"]),
            float(metrics["accuracy_lift_over_majority"]),
            dominant_precision,
            float(metrics["precision_balance_score"]),
            min(long_count, short_count),
            -(
                abs(float(long_threshold) - float(default_threshold))
                + abs(float(short_threshold) - float(1.0 - default_threshold))
            ),
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
    best_key = _candidate_key(best_metrics, best_long_threshold, best_short_threshold)

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
            key = _candidate_key(metrics, float(long_threshold), float(short_threshold))
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
    **kwargs: Any,
) -> tuple[float, Dict[str, Any]]:
    long_threshold, short_threshold, meta = _select_calibrated_action_thresholds(
        pred_probs_np,
        y_true_np,
        default_threshold=default_threshold,
        sample_confidence=sample_confidence,
        **kwargs,
    )
    meta = dict(meta)
    meta["selected_threshold"] = float(max(long_threshold, 1.0 - short_threshold))
    return float(meta["selected_threshold"]), meta
