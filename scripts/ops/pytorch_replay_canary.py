#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gzip
import json
import math
import os
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = PROJECT_ROOT / "governance" / "health" / "pytorch_replay_canary_latest.json"
DEFAULT_HISTORY_OUT = PROJECT_ROOT / "governance" / "health" / "pytorch_replay_canary_history.jsonl"
DEFAULT_ENABLED = os.getenv("PYTORCH_REPLAY_CANARY_ENABLED", "0").strip().lower() in {"1", "true", "yes", "on"}

FEATURE_NAMES = [
    "model_score",
    "threshold",
    "score_gap",
    "abs_score_gap",
    "intent_score",
    "quantity_log1p",
    "queue_depth_log1p",
    "lane_budget_mult",
    "master_weight_trend",
    "master_weight_mean_revert",
    "master_weight_shock",
    "allow_live_promotion",
    "guard_blocked_intent",
    "bot_weight",
    "test_accuracy",
    "bot_promoted",
    "action_buy",
    "action_sell",
    "lane_futures",
    "lane_day",
    "lane_swing",
    "lane_options",
    "lane_long_term",
    "layer_grand_master",
    "layer_sub_bot_paper_mirror",
    "layer_options_sub_bot_paper_mirror",
    "layer_futures_sub_bot_paper_mirror",
    "shadow_domain_crypto",
    "shadow_domain_equities",
    "bot_role_signal_sub_bot",
    "bot_role_options_sub_bot",
    "bot_role_infrastructure_sub_bot",
]


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_float(raw: Any, default: float = 0.0) -> float:
    try:
        return float(raw)
    except Exception:
        return float(default)


def _text_key(raw: Any, default: str = "unknown") -> str:
    text = str(raw or "").strip().lower()
    return text or default


def _glob_source_paths(project_root: Path, *, max_files: int) -> list[Path]:
    root = project_root / "exports" / "paper_broker_bridge" / "paper"
    paths = sorted(list(root.glob("paper_bridge_orders_*.jsonl")) + list(root.glob("paper_bridge_orders_*.jsonl.gz")))
    return [path for path in paths if ".local_fallback" not in path.name][-max(int(max_files), 1) :]


def _iter_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    opener = gzip.open if path.suffix == ".gz" else open
    try:
        with opener(path, "rt", encoding="utf-8", errors="ignore") as handle:
            for raw in handle:
                line = raw.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except Exception:
                    continue
                if isinstance(payload, dict):
                    rows.append(payload)
    except Exception:
        return []
    return rows


def _lane_flags(value: str) -> dict[str, float]:
    lane = _text_key(value)
    return {
        "lane_futures": 1.0 if "futures" in lane else 0.0,
        "lane_day": 1.0 if "day" in lane else 0.0,
        "lane_swing": 1.0 if "swing" in lane else 0.0,
        "lane_options": 1.0 if "option" in lane else 0.0,
        "lane_long_term": 1.0 if ("long_term" in lane or "long-term" in lane) else 0.0,
    }


def _layer_flags(value: str) -> dict[str, float]:
    layer = _text_key(value)
    return {
        "layer_grand_master": 1.0 if layer == "grand_master" else 0.0,
        "layer_sub_bot_paper_mirror": 1.0 if layer == "sub_bot_paper_mirror" else 0.0,
        "layer_options_sub_bot_paper_mirror": 1.0 if layer == "options_sub_bot_paper_mirror" else 0.0,
        "layer_futures_sub_bot_paper_mirror": 1.0 if layer == "futures_sub_bot_paper_mirror" else 0.0,
    }


def _domain_flags(value: str) -> dict[str, float]:
    domain = _text_key(value)
    return {
        "shadow_domain_crypto": 1.0 if domain == "crypto" else 0.0,
        "shadow_domain_equities": 1.0 if domain == "equities" else 0.0,
    }


def _role_flags(value: str) -> dict[str, float]:
    role = _text_key(value)
    return {
        "bot_role_signal_sub_bot": 1.0 if role == "signal_sub_bot" else 0.0,
        "bot_role_options_sub_bot": 1.0 if role == "options_sub_bot" else 0.0,
        "bot_role_infrastructure_sub_bot": 1.0 if role == "infrastructure_sub_bot" else 0.0,
    }


def _master_weight_flags(meta: dict[str, Any]) -> dict[str, float]:
    raw = meta.get("master_weights") if isinstance(meta.get("master_weights"), dict) else {}
    return {
        "master_weight_trend": _safe_float(raw.get("trend"), 0.0),
        "master_weight_mean_revert": _safe_float(raw.get("mean_revert"), 0.0),
        "master_weight_shock": _safe_float(raw.get("shock"), 0.0),
    }


def _sample_quality(low_rows: bool, low_class_balance: bool) -> str:
    if low_rows and low_class_balance:
        return "degraded_low_rows_and_class_balance"
    if low_rows:
        return "degraded_low_rows"
    if low_class_balance:
        return "degraded_class_balance"
    return "full"


def _pnl_target_from_row(row: dict[str, Any], *, deadzone: float) -> tuple[float | None, str]:
    event_pnl = _safe_float(row.get("realized_pnl"), 0.0) + _safe_float(row.get("unrealized_pnl"), 0.0)
    total_pnl = _safe_float(row.get("realized_pnl_total"), 0.0) + _safe_float(row.get("unrealized_pnl_total"), 0.0)
    if abs(event_pnl) > float(deadzone):
        return float(event_pnl), "event"
    if abs(total_pnl) > float(deadzone):
        return float(total_pnl), "total_fallback"
    return None, "flat"


def _example_from_row(row: dict[str, Any], *, deadzone: float = 1e-4) -> dict[str, Any] | None:
    meta = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    model_score = _safe_float(row.get("model_score"), math.nan)
    threshold = _safe_float(row.get("threshold"), math.nan)
    if not math.isfinite(model_score) or not math.isfinite(threshold):
        return None

    pnl_target, label_source = _pnl_target_from_row(row, deadzone=deadzone)
    if pnl_target is None:
        return None

    action = str(row.get("action") or "").strip().upper()
    lane = _text_key(meta.get("runtime_lane") or meta.get("source_profile") or row.get("strategy"))
    layer = _text_key(meta.get("layer"))
    source_profile = _text_key(meta.get("source_profile"))
    shadow_domain = _text_key(meta.get("shadow_domain"))
    bot_role = _text_key(meta.get("bot_role"))
    score_gap = model_score - threshold
    feature_map = {
        "model_score": model_score,
        "threshold": threshold,
        "score_gap": score_gap,
        "abs_score_gap": abs(score_gap),
        "intent_score": _safe_float(meta.get("intent_score"), model_score),
        "quantity_log1p": math.log1p(max(_safe_float(row.get("quantity"), 0.0), 0.0)),
        "queue_depth_log1p": math.log1p(max(_safe_float(meta.get("queue_depth"), 0.0), 0.0)),
        "lane_budget_mult": _safe_float(meta.get("lane_budget_mult"), 1.0),
        "allow_live_promotion": 1.0 if bool(meta.get("allow_live_promotion")) else 0.0,
        "guard_blocked_intent": 1.0 if bool(meta.get("guard_blocked_intent")) else 0.0,
        "bot_weight": _safe_float(meta.get("bot_weight"), 0.0),
        "test_accuracy": _safe_float(meta.get("test_accuracy"), 0.0),
        "bot_promoted": 1.0 if bool(meta.get("bot_promoted")) else 0.0,
        "action_buy": 1.0 if action == "BUY" else 0.0,
        "action_sell": 1.0 if action == "SELL" else 0.0,
        **_master_weight_flags(meta),
        **_lane_flags(lane),
        **_layer_flags(layer),
        **_domain_flags(shadow_domain),
        **_role_flags(bot_role),
    }
    return {
        "timestamp_utc": str(row.get("timestamp_utc") or ""),
        "pnl_target": float(pnl_target),
        "label": 1.0 if pnl_target > 0.0 else 0.0,
        "label_source": label_source,
        "baseline_score": float(score_gap),
        "features": np.asarray([feature_map[name] for name in FEATURE_NAMES], dtype=np.float32),
        "layer": layer,
        "runtime_lane": lane,
        "source_profile": source_profile,
        "shadow_domain": shadow_domain,
    }


def _load_examples(project_root: Path, *, max_files: int, max_rows: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    source_paths = _glob_source_paths(project_root, max_files=max_files)
    examples: list[dict[str, Any]] = []
    rows_scanned = 0
    label_sources: Counter[str] = Counter()
    for path in source_paths:
        for row in _iter_rows(path):
            rows_scanned += 1
            example = _example_from_row(row)
            if example is not None:
                label_sources[str(example.get("label_source") or "unknown")] += 1
                examples.append(example)
    examples.sort(key=lambda row: str(row.get("timestamp_utc") or ""))
    if len(examples) > int(max_rows):
        examples = examples[-int(max_rows) :]
    meta = {
        "source_files": [str(path) for path in source_paths],
        "file_count": len(source_paths),
        "rows_scanned": int(rows_scanned),
        "rows_used": int(len(examples)),
        "label_source_counts": dict(label_sources),
    }
    return examples, meta


def _prepare_arrays(examples: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    timestamps = np.asarray([str(row.get("timestamp_utc") or "") for row in examples], dtype=object)
    features = np.asarray([row["features"] for row in examples], dtype=np.float32)
    labels = np.asarray([float(row.get("label", 0.0)) for row in examples], dtype=np.float32)
    pnl = np.asarray([float(row.get("pnl_target", 0.0)) for row in examples], dtype=np.float32)
    segments = {
        "layer": np.asarray([str(row.get("layer") or "unknown") for row in examples], dtype=object),
        "runtime_lane": np.asarray([str(row.get("runtime_lane") or "unknown") for row in examples], dtype=object),
        "source_profile": np.asarray([str(row.get("source_profile") or "unknown") for row in examples], dtype=object),
        "shadow_domain": np.asarray([str(row.get("shadow_domain") or "unknown") for row in examples], dtype=object),
    }
    return timestamps, features, labels, pnl, segments


def _top_bucket_metrics(scores: np.ndarray, pnl: np.ndarray, labels: np.ndarray, *, fraction: float) -> dict[str, float]:
    if len(scores) == 0:
        return {"count": 0.0, "mean_net_pnl_total": 0.0, "hit_rate": 0.0}
    bucket = max(1, int(len(scores) * max(min(float(fraction), 0.5), 0.01)))
    order = np.argsort(scores)
    idx = order[-bucket:]
    return {
        "count": float(bucket),
        "mean_net_pnl_total": float(np.mean(pnl[idx])) if len(idx) else 0.0,
        "hit_rate": float(np.mean(labels[idx])) if len(idx) else 0.0,
    }


def _binary_metrics(scores: np.ndarray, labels: np.ndarray, pnl: np.ndarray, *, threshold: float, top_fraction: float) -> dict[str, Any]:
    if len(scores) == 0:
        return {
            "accuracy": 0.0,
            "selected_rate": 0.0,
            "selected_count": 0,
            "selected_mean_net_pnl_total": 0.0,
            "selected_hit_rate": 0.0,
            "top_bucket": _top_bucket_metrics(scores, pnl, labels, fraction=top_fraction),
        }
    preds = (scores >= float(threshold)).astype(np.float32)
    selected = preds > 0.0
    return {
        "accuracy": float(np.mean(preds == labels)),
        "selected_rate": float(np.mean(selected)),
        "selected_count": int(np.sum(selected)),
        "selected_mean_net_pnl_total": float(np.mean(pnl[selected])) if np.any(selected) else 0.0,
        "selected_hit_rate": float(np.mean(labels[selected])) if np.any(selected) else 0.0,
        "top_bucket": _top_bucket_metrics(scores, pnl, labels, fraction=top_fraction),
    }


def _selection_metrics(selected: np.ndarray, labels: np.ndarray, pnl: np.ndarray, scores: np.ndarray, *, top_fraction: float) -> dict[str, Any]:
    if len(selected) == 0:
        return {
            "accuracy": 0.0,
            "selected_rate": 0.0,
            "selected_count": 0,
            "selected_mean_net_pnl_total": 0.0,
            "selected_hit_rate": 0.0,
            "top_bucket": _top_bucket_metrics(scores, pnl, labels, fraction=top_fraction),
        }
    picked = selected.astype(bool)
    preds = picked.astype(np.float32)
    return {
        "accuracy": float(np.mean(preds == labels)),
        "selected_rate": float(np.mean(picked)),
        "selected_count": int(np.sum(picked)),
        "selected_mean_net_pnl_total": float(np.mean(pnl[picked])) if np.any(picked) else 0.0,
        "selected_hit_rate": float(np.mean(labels[picked])) if np.any(picked) else 0.0,
        "top_bucket": _top_bucket_metrics(scores, pnl, labels, fraction=top_fraction),
    }


def _effective_min_rows(requested_min_rows: int) -> int:
    requested = max(int(requested_min_rows), 10)
    return max(300, min(requested, 1000))


def _effective_min_class_rows(requested_min_class_rows: int, train_rows: int, val_rows: int) -> int:
    requested = max(int(requested_min_class_rows), 1)
    bounded = min(requested, max(train_rows // 10, 12), max(val_rows // 4, 12))
    return max(int(bounded), 12)


def _segment_deltas(
    pytorch_scores: np.ndarray,
    baseline_scores: np.ndarray,
    labels: np.ndarray,
    pnl: np.ndarray,
    segment_values: np.ndarray,
    *,
    top_fraction: float,
    min_segment_rows: int,
    limit: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    unique_values, counts = np.unique(segment_values, return_counts=True)
    for raw_value, raw_count in zip(unique_values.tolist(), counts.tolist()):
        value = str(raw_value or "unknown")
        count = int(raw_count)
        if value == "unknown" or count < int(max(min_segment_rows, 1)):
            continue
        mask = segment_values == raw_value
        pytorch_metrics = _binary_metrics(
            pytorch_scores[mask],
            labels[mask],
            pnl[mask],
            threshold=0.5,
            top_fraction=top_fraction,
        )
        baseline_metrics = _binary_metrics(
            baseline_scores[mask],
            labels[mask],
            pnl[mask],
            threshold=0.0,
            top_fraction=top_fraction,
        )
        rows.append(
            {
                "segment": value,
                "rows": count,
                "accuracy_vs_baseline": round(
                    float(pytorch_metrics["accuracy"]) - float(baseline_metrics["accuracy"]),
                    6,
                ),
                "selected_mean_net_pnl_total_vs_baseline": round(
                    float(pytorch_metrics["selected_mean_net_pnl_total"])
                    - float(baseline_metrics["selected_mean_net_pnl_total"]),
                    6,
                ),
                "top_bucket_mean_net_pnl_total_vs_baseline": round(
                    float(pytorch_metrics["top_bucket"]["mean_net_pnl_total"])
                    - float(baseline_metrics["top_bucket"]["mean_net_pnl_total"]),
                    6,
                ),
            }
        )
    rows.sort(
        key=lambda row: (
            abs(float(row["top_bucket_mean_net_pnl_total_vs_baseline"])),
            int(row["rows"]),
        ),
        reverse=True,
    )
    return rows[: max(int(limit), 1)]


def _threshold_candidates(scores: np.ndarray, default_threshold: float) -> list[float]:
    if len(scores) == 0:
        return [float(default_threshold)]
    quantiles = np.linspace(0.1, 0.9, 17)
    raw = [float(default_threshold), 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65]
    raw.extend(float(np.quantile(scores, q)) for q in quantiles.tolist())
    candidates = sorted({round(min(max(value, 0.01), 0.99), 6) for value in raw})
    return candidates or [float(default_threshold)]


def _calibrate_threshold(
    scores: np.ndarray,
    labels: np.ndarray,
    pnl: np.ndarray,
    *,
    default_threshold: float,
    top_fraction: float,
) -> dict[str, Any]:
    if len(scores) == 0:
        metrics = _binary_metrics(scores, labels, pnl, threshold=default_threshold, top_fraction=top_fraction)
        return {
            "threshold": float(default_threshold),
            "selected_count": int(metrics["selected_count"]),
            "selected_rate": float(metrics["selected_rate"]),
            "selected_mean_net_pnl_total": float(metrics["selected_mean_net_pnl_total"]),
            "selected_hit_rate": float(metrics["selected_hit_rate"]),
            "accuracy": float(metrics["accuracy"]),
        }

    min_selected = max(8, int(len(scores) * min(max(float(top_fraction), 0.05), 0.15)))
    max_selected = max(min_selected, int(len(scores) * 0.8))
    best_row: dict[str, Any] | None = None
    for threshold in _threshold_candidates(scores, default_threshold):
        selected = scores >= float(threshold)
        selected_count = int(np.sum(selected))
        if selected_count < min_selected or selected_count > max_selected:
            continue
        metrics = _selection_metrics(selected, labels, pnl, scores, top_fraction=top_fraction)
        row = {
            "threshold": float(threshold),
            "selected_count": int(metrics["selected_count"]),
            "selected_rate": float(metrics["selected_rate"]),
            "selected_mean_net_pnl_total": float(metrics["selected_mean_net_pnl_total"]),
            "selected_hit_rate": float(metrics["selected_hit_rate"]),
            "accuracy": float(metrics["accuracy"]),
        }
        if best_row is None:
            best_row = row
            continue
        best_key = (
            float(best_row["selected_mean_net_pnl_total"]),
            float(best_row["selected_hit_rate"]),
            float(best_row["accuracy"]),
            int(best_row["selected_count"]),
        )
        row_key = (
            float(row["selected_mean_net_pnl_total"]),
            float(row["selected_hit_rate"]),
            float(row["accuracy"]),
            int(row["selected_count"]),
        )
        if row_key > best_key:
            best_row = row
    if best_row is not None:
        return best_row
    metrics = _binary_metrics(scores, labels, pnl, threshold=default_threshold, top_fraction=top_fraction)
    return {
        "threshold": float(default_threshold),
        "selected_count": int(metrics["selected_count"]),
        "selected_rate": float(metrics["selected_rate"]),
        "selected_mean_net_pnl_total": float(metrics["selected_mean_net_pnl_total"]),
        "selected_hit_rate": float(metrics["selected_hit_rate"]),
        "accuracy": float(metrics["accuracy"]),
    }


def _calibrate_segment_thresholds(
    scores: np.ndarray,
    labels: np.ndarray,
    pnl: np.ndarray,
    segments: np.ndarray,
    *,
    default_threshold: float,
    top_fraction: float,
    min_segment_rows: int,
) -> dict[str, Any]:
    global_row = _calibrate_threshold(
        scores,
        labels,
        pnl,
        default_threshold=default_threshold,
        top_fraction=top_fraction,
    )
    threshold_by_segment: dict[str, float] = {}
    selected_rate_by_segment: dict[str, float] = {}
    rows: list[dict[str, Any]] = []
    unique_values, counts = np.unique(segments, return_counts=True)
    for raw_value, raw_count in zip(unique_values.tolist(), counts.tolist()):
        segment = str(raw_value or "unknown")
        count = int(raw_count)
        if segment == "unknown" or count < int(max(min_segment_rows, 1)):
            continue
        mask = segments == raw_value
        segment_row = _calibrate_threshold(
            scores[mask],
            labels[mask],
            pnl[mask],
            default_threshold=float(global_row["threshold"]),
            top_fraction=top_fraction,
        )
        threshold_by_segment[segment] = float(segment_row["threshold"])
        selected_rate_by_segment[segment] = float(segment_row["selected_rate"])
        rows.append(
            {
                "segment": segment,
                "rows": count,
                **segment_row,
            }
        )
    rows.sort(key=lambda row: int(row["rows"]), reverse=True)
    return {
        "context": "source_profile",
        "default_threshold": float(default_threshold),
        "global_threshold": float(global_row["threshold"]),
        "global_selected_rate": float(global_row["selected_rate"]),
        "min_segment_rows": int(max(min_segment_rows, 1)),
        "segment_count": len(rows),
        "threshold_by_segment": threshold_by_segment,
        "selected_rate_by_segment": selected_rate_by_segment,
        "segments": rows[:10],
    }


def _threshold_for_selected_rate(scores: np.ndarray, selected_rate: float, fallback_threshold: float) -> float:
    if len(scores) == 0:
        return float(fallback_threshold)
    rate = min(max(float(selected_rate), 0.02), 0.8)
    quantile = float(np.quantile(scores, max(0.0, min(1.0, 1.0 - rate))))
    return float(min(max(quantile, 0.01), 0.99))


def _calibrated_thresholds_for_segments(
    scores: np.ndarray,
    segments: np.ndarray,
    calibration: dict[str, Any],
) -> np.ndarray:
    thresholds = np.full(len(scores), float(calibration.get("global_threshold", 0.5)), dtype=np.float32)
    if len(scores) == 0:
        return thresholds
    global_rate = float(calibration.get("global_selected_rate", 0.25))
    unique_segments = np.unique(segments)
    segment_rates = calibration.get("selected_rate_by_segment", {})
    for raw_segment in unique_segments.tolist():
        segment = str(raw_segment or "unknown")
        mask = segments == raw_segment
        target_rate = float(segment_rates.get(segment, global_rate))
        thresholds[mask] = _threshold_for_selected_rate(
            scores[mask],
            target_rate,
            float(calibration.get("global_threshold", 0.5)),
        )
    return thresholds


def _resolve_device(requested: str) -> str:
    import torch

    raw = str(requested or "auto").strip().lower()
    if raw in {"mps", "cpu"}:
        return raw
    mps_backend = getattr(torch.backends, "mps", None)
    if mps_backend is not None and getattr(mps_backend, "is_available", lambda: False)():
        return "mps"
    return "cpu"


def _feature_importance(weights: np.ndarray, *, limit: int = 8) -> list[dict[str, float]]:
    rows = [{"feature": name, "weight": float(weight)} for name, weight in zip(FEATURE_NAMES, weights)]
    rows.sort(key=lambda row: abs(float(row["weight"])), reverse=True)
    return rows[: max(int(limit), 1)]


def _split_index(total_rows: int, *, validation_fraction: float, min_train_rows: int, min_val_rows: int) -> int | None:
    if total_rows < max(min_train_rows + min_val_rows, 2):
        return None
    split_idx = int(total_rows * (1.0 - max(min(float(validation_fraction), 0.4), 0.05)))
    split_idx = min(
        max(split_idx, min(max(total_rows // 2, min_train_rows), total_rows - min_val_rows)),
        total_rows - min_val_rows,
    )
    if split_idx <= 0 or split_idx >= total_rows:
        return None
    return int(split_idx)


def _train_probe_metrics(
    X_train_raw: np.ndarray,
    y_train: np.ndarray,
    pnl_train: np.ndarray,
    train_segments: dict[str, np.ndarray],
    X_val_raw: np.ndarray,
    y_val: np.ndarray,
    pnl_val: np.ndarray,
    val_segments: dict[str, np.ndarray],
    *,
    device: str,
    epochs: int,
    lr: float,
    top_fraction: float,
) -> dict[str, Any]:
    mean = X_train_raw.mean(axis=0, keepdims=True)
    std = X_train_raw.std(axis=0, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    X_train = ((X_train_raw - mean) / std).astype(np.float32)
    X_val = ((X_val_raw - mean) / std).astype(np.float32)

    import torch

    torch.manual_seed(7)
    np.random.seed(7)
    resolved_device = _resolve_device(device)
    model = torch.nn.Linear(X_train.shape[1], 1).to(resolved_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=1e-4)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    X_train_t = torch.from_numpy(X_train).to(resolved_device)
    y_train_t = torch.from_numpy(y_train.reshape(-1, 1)).to(resolved_device)
    X_val_t = torch.from_numpy(X_val).to(resolved_device)

    train_losses: list[float] = []
    for _ in range(int(max(epochs, 1))):
        optimizer.zero_grad(set_to_none=True)
        logits = model(X_train_t)
        loss = loss_fn(logits, y_train_t)
        loss.backward()
        optimizer.step()
        train_losses.append(float(loss.detach().cpu().item()))

    with torch.no_grad():
        train_logits = model(X_train_t).squeeze(-1)
        train_probs = torch.sigmoid(train_logits).detach().cpu().numpy()
        val_logits = model(X_val_t).squeeze(-1)
        val_probs = torch.sigmoid(val_logits).detach().cpu().numpy()
        weights = model.weight.detach().cpu().numpy()[0]
        bias = float(model.bias.detach().cpu().numpy()[0])

    baseline_train = X_train_raw[:, FEATURE_NAMES.index("score_gap")]
    baseline_val = X_val_raw[:, FEATURE_NAMES.index("score_gap")]
    baseline_metrics = _binary_metrics(baseline_val, y_val, pnl_val, threshold=0.0, top_fraction=top_fraction)
    pytorch_metrics = _binary_metrics(val_probs, y_val, pnl_val, threshold=0.5, top_fraction=top_fraction)
    threshold_calibration = _calibrate_segment_thresholds(
        train_probs,
        y_train,
        pnl_train,
        train_segments["source_profile"],
        default_threshold=0.5,
        top_fraction=top_fraction,
        min_segment_rows=max(24, len(X_train_raw) // 20),
    )
    calibrated_thresholds = _calibrated_thresholds_for_segments(
        val_probs,
        val_segments["source_profile"],
        threshold_calibration,
    )
    pytorch_calibrated_metrics = _selection_metrics(
        val_probs >= calibrated_thresholds,
        y_val,
        pnl_val,
        val_probs,
        top_fraction=top_fraction,
    )
    min_segment_rows = max(12, min(30, len(X_val_raw) // 8 if len(X_val_raw) else 12))
    segment_deltas = {
        name: _segment_deltas(
            val_probs,
            baseline_val,
            y_val,
            pnl_val,
            values,
            top_fraction=top_fraction,
            min_segment_rows=min_segment_rows,
            limit=5,
        )
        for name, values in val_segments.items()
    }
    return {
        "device": resolved_device,
        "training": {
            "epochs": int(max(epochs, 1)),
            "learning_rate": float(lr),
            "final_train_loss": float(train_losses[-1]) if train_losses else 0.0,
            "train_losses_tail": [round(x, 6) for x in train_losses[-5:]],
            "bias": bias,
        },
        "baseline": baseline_metrics,
        "pytorch": pytorch_metrics,
        "pytorch_calibrated": pytorch_calibrated_metrics,
        "threshold_calibration": threshold_calibration,
        "feature_importance": _feature_importance(weights, limit=8),
        "segment_deltas": segment_deltas,
        "deltas": {
            "accuracy_vs_baseline": round(
                float(pytorch_metrics["accuracy"]) - float(baseline_metrics["accuracy"]),
                6,
            ),
            "selected_mean_net_pnl_total_vs_baseline": round(
                float(pytorch_metrics["selected_mean_net_pnl_total"]) - float(baseline_metrics["selected_mean_net_pnl_total"]),
                6,
            ),
            "top_bucket_mean_net_pnl_total_vs_baseline": round(
                float(pytorch_metrics["top_bucket"]["mean_net_pnl_total"])
                - float(baseline_metrics["top_bucket"]["mean_net_pnl_total"]),
                6,
            ),
        },
        "calibrated_deltas": {
            "accuracy_vs_baseline": round(
                float(pytorch_calibrated_metrics["accuracy"]) - float(baseline_metrics["accuracy"]),
                6,
            ),
            "selected_mean_net_pnl_total_vs_baseline": round(
                float(pytorch_calibrated_metrics["selected_mean_net_pnl_total"])
                - float(baseline_metrics["selected_mean_net_pnl_total"]),
                6,
            ),
            "selected_hit_rate_vs_baseline": round(
                float(pytorch_calibrated_metrics["selected_hit_rate"]) - float(baseline_metrics["selected_hit_rate"]),
                6,
            ),
        },
    }


def _walk_forward_slices(total_rows: int, *, folds: int, min_train_rows: int, min_val_rows: int) -> list[tuple[int, int]]:
    if total_rows < max(min_train_rows + min_val_rows, 2):
        return []
    fold_count = max(int(folds), 1)
    val_size = max(int(min_val_rows), total_rows // (fold_count + 2))
    first_train_end = max(int(min_train_rows), total_rows - (fold_count * val_size))
    first_train_end = min(first_train_end, total_rows - min_val_rows)
    if first_train_end <= 0:
        return []
    slices: list[tuple[int, int]] = []
    train_end = first_train_end
    while train_end + min_val_rows <= total_rows and len(slices) < fold_count:
        val_end = min(train_end + val_size, total_rows)
        if val_end - train_end < min_val_rows:
            break
        slices.append((train_end, val_end))
        train_end += val_size
    return slices


def _aggregate_segment_rows(fold_segment_rows: list[list[dict[str, Any]]]) -> list[dict[str, Any]]:
    buckets: dict[str, dict[str, Any]] = {}
    for rows in fold_segment_rows:
        for row in rows:
            segment = str(row.get("segment") or "unknown")
            bucket = buckets.setdefault(
                segment,
                {
                    "segment": segment,
                    "rows_seen": 0,
                    "folds_seen": 0,
                    "selected_mean_values": [],
                    "top_bucket_values": [],
                    "accuracy_values": [],
                },
            )
            bucket["rows_seen"] += int(row.get("rows", 0) or 0)
            bucket["folds_seen"] += 1
            bucket["selected_mean_values"].append(float(row.get("selected_mean_net_pnl_total_vs_baseline", 0.0) or 0.0))
            bucket["top_bucket_values"].append(float(row.get("top_bucket_mean_net_pnl_total_vs_baseline", 0.0) or 0.0))
            bucket["accuracy_values"].append(float(row.get("accuracy_vs_baseline", 0.0) or 0.0))
    out: list[dict[str, Any]] = []
    for bucket in buckets.values():
        selected_values = bucket["selected_mean_values"]
        top_bucket_values = bucket["top_bucket_values"]
        accuracy_values = bucket["accuracy_values"]
        out.append(
            {
                "segment": bucket["segment"],
                "rows_seen": int(bucket["rows_seen"]),
                "folds_seen": int(bucket["folds_seen"]),
                "mean_selected_mean_net_pnl_total_vs_baseline": round(float(np.mean(selected_values)), 6) if selected_values else 0.0,
                "mean_top_bucket_mean_net_pnl_total_vs_baseline": round(float(np.mean(top_bucket_values)), 6) if top_bucket_values else 0.0,
                "mean_accuracy_vs_baseline": round(float(np.mean(accuracy_values)), 6) if accuracy_values else 0.0,
                "positive_selected_mean_folds": int(sum(1 for value in selected_values if value > 0.0)),
                "positive_top_bucket_folds": int(sum(1 for value in top_bucket_values if value > 0.0)),
            }
        )
    out.sort(
        key=lambda row: (
            float(row["mean_selected_mean_net_pnl_total_vs_baseline"]),
            float(row["mean_top_bucket_mean_net_pnl_total_vs_baseline"]),
            int(row["folds_seen"]),
        ),
        reverse=True,
    )
    return out


def _walk_forward_probe(
    timestamps: np.ndarray,
    features: np.ndarray,
    labels: np.ndarray,
    pnl: np.ndarray,
    segments: dict[str, np.ndarray],
    *,
    folds: int,
    validation_fraction: float,
    device: str,
    epochs: int,
    lr: float,
    top_fraction: float,
) -> dict[str, Any]:
    min_val_rows = max(48, int(len(features) * max(min(float(validation_fraction), 0.25), 0.08)))
    min_train_rows = max(128, min(len(features) - min_val_rows, len(features) // 2))
    slices = _walk_forward_slices(
        len(features),
        folds=folds,
        min_train_rows=min_train_rows,
        min_val_rows=min_val_rows,
    )
    if not slices:
        return {
            "fold_count": 0,
            "folds": [],
            "summary": {
                "raw_mean_selected_mean_net_pnl_total_vs_baseline": 0.0,
                "raw_mean_top_bucket_mean_net_pnl_total_vs_baseline": 0.0,
                "calibrated_mean_selected_mean_net_pnl_total_vs_baseline": 0.0,
                "positive_raw_top_bucket_folds": 0,
                "positive_calibrated_selected_mean_folds": 0,
            },
            "source_profile_summary": [],
        }

    fold_rows: list[dict[str, Any]] = []
    source_profile_fold_rows: list[list[dict[str, Any]]] = []
    for idx, (train_end, val_end) in enumerate(slices, start=1):
        train_segments = {name: values[:train_end] for name, values in segments.items()}
        val_segments = {name: values[train_end:val_end] for name, values in segments.items()}
        probe = _train_probe_metrics(
            features[:train_end],
            labels[:train_end],
            pnl[:train_end],
            train_segments,
            features[train_end:val_end],
            labels[train_end:val_end],
            pnl[train_end:val_end],
            val_segments,
            device=device,
            epochs=epochs,
            lr=lr,
            top_fraction=top_fraction,
        )
        source_profile_fold_rows.append(probe["segment_deltas"].get("source_profile", []))
        fold_rows.append(
            {
                "fold_index": idx,
                "train_rows": int(train_end),
                "val_rows": int(val_end - train_end),
                "val_start_timestamp_utc": str(timestamps[train_end]) if train_end < len(timestamps) else "",
                "val_end_timestamp_utc": str(timestamps[val_end - 1]) if val_end - 1 < len(timestamps) and val_end > train_end else "",
                "deltas": probe["deltas"],
                "calibrated_deltas": probe["calibrated_deltas"],
            }
        )

    raw_selected_mean = [float(row["deltas"]["selected_mean_net_pnl_total_vs_baseline"]) for row in fold_rows]
    raw_top_bucket = [float(row["deltas"]["top_bucket_mean_net_pnl_total_vs_baseline"]) for row in fold_rows]
    calibrated_selected_mean = [float(row["calibrated_deltas"]["selected_mean_net_pnl_total_vs_baseline"]) for row in fold_rows]
    return {
        "fold_count": len(fold_rows),
        "folds": fold_rows,
        "summary": {
            "raw_mean_selected_mean_net_pnl_total_vs_baseline": round(float(np.mean(raw_selected_mean)), 6) if raw_selected_mean else 0.0,
            "raw_mean_top_bucket_mean_net_pnl_total_vs_baseline": round(float(np.mean(raw_top_bucket)), 6) if raw_top_bucket else 0.0,
            "calibrated_mean_selected_mean_net_pnl_total_vs_baseline": round(float(np.mean(calibrated_selected_mean)), 6) if calibrated_selected_mean else 0.0,
            "positive_raw_top_bucket_folds": int(sum(1 for value in raw_top_bucket if value > 0.0)),
            "positive_calibrated_selected_mean_folds": int(sum(1 for value in calibrated_selected_mean if value > 0.0)),
        },
        "source_profile_summary": _aggregate_segment_rows(source_profile_fold_rows),
    }


def _micro_model_probe(
    features: np.ndarray,
    labels: np.ndarray,
    pnl: np.ndarray,
    segments: dict[str, np.ndarray],
    *,
    validation_fraction: float,
    device: str,
    epochs: int,
    lr: float,
    top_fraction: float,
    max_profiles: int = 6,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    unique_profiles, counts = np.unique(segments["source_profile"], return_counts=True)
    for raw_profile, raw_count in zip(unique_profiles.tolist(), counts.tolist()):
        profile = str(raw_profile or "unknown")
        count = int(raw_count)
        if profile == "unknown" or count < 140:
            continue
        mask = segments["source_profile"] == raw_profile
        profile_features = features[mask]
        profile_labels = labels[mask]
        profile_pnl = pnl[mask]
        profile_segments = {name: values[mask] for name, values in segments.items()}
        split_idx = _split_index(
            len(profile_features),
            validation_fraction=validation_fraction,
            min_train_rows=max(96, count // 2),
            min_val_rows=max(32, count // 5),
        )
        if split_idx is None:
            continue
        train_pos = int(np.sum(profile_labels[:split_idx] > 0.5))
        train_neg = int(split_idx - train_pos)
        val_pos = int(np.sum(profile_labels[split_idx:] > 0.5))
        val_neg = int(len(profile_labels[split_idx:]) - val_pos)
        if min(train_pos, train_neg, val_pos, val_neg) < 12:
            continue
        probe = _train_probe_metrics(
            profile_features[:split_idx],
            profile_labels[:split_idx],
            profile_pnl[:split_idx],
            {name: values[:split_idx] for name, values in profile_segments.items()},
            profile_features[split_idx:],
            profile_labels[split_idx:],
            profile_pnl[split_idx:],
            {name: values[split_idx:] for name, values in profile_segments.items()},
            device=device,
            epochs=epochs,
            lr=lr,
            top_fraction=top_fraction,
        )
        rows.append(
            {
                "source_profile": profile,
                "rows_total": count,
                "rows_train": int(split_idx),
                "rows_val": int(len(profile_features) - split_idx),
                "deltas": probe["deltas"],
                "calibrated_deltas": probe["calibrated_deltas"],
                "threshold_calibration": probe["threshold_calibration"],
                "feature_importance": probe["feature_importance"][:5],
            }
        )
    rows.sort(
        key=lambda row: (
            float(row["calibrated_deltas"]["selected_mean_net_pnl_total_vs_baseline"]),
            float(row["deltas"]["top_bucket_mean_net_pnl_total_vs_baseline"]),
            int(row["rows_total"]),
        ),
        reverse=True,
    )
    return rows[: max(int(max_profiles), 1)]


def _mlx_shadow_assist(
    walk_forward: dict[str, Any],
    micro_models: list[dict[str, Any]],
) -> dict[str, Any]:
    micro_by_profile = {str(row.get("source_profile") or ""): row for row in micro_models}
    fold_count = int(walk_forward.get("fold_count", 0) or 0)
    min_positive_folds = max(1, fold_count // 2)
    eligible: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for row in walk_forward.get("source_profile_summary", []):
        profile = str(row.get("segment") or "")
        micro = micro_by_profile.get(profile)
        if not micro:
            continue
        walk_selected = float(row.get("mean_selected_mean_net_pnl_total_vs_baseline", 0.0) or 0.0)
        walk_top = float(row.get("mean_top_bucket_mean_net_pnl_total_vs_baseline", 0.0) or 0.0)
        micro_selected = float(micro["calibrated_deltas"]["selected_mean_net_pnl_total_vs_baseline"])
        micro_top = float(micro["deltas"]["top_bucket_mean_net_pnl_total_vs_baseline"])
        candidate = {
            "source_profile": profile,
            "walk_forward_mean_selected_mean_net_pnl_total_vs_baseline": round(walk_selected, 6),
            "walk_forward_mean_top_bucket_mean_net_pnl_total_vs_baseline": round(walk_top, 6),
            "walk_forward_positive_top_bucket_folds": int(row.get("positive_top_bucket_folds", 0) or 0),
            "micro_model_selected_mean_net_pnl_total_vs_baseline": round(micro_selected, 6),
            "micro_model_top_bucket_mean_net_pnl_total_vs_baseline": round(micro_top, 6),
            "micro_model_threshold": round(float(micro["threshold_calibration"].get("global_threshold", 0.5)), 6),
        }
        if (
            walk_selected > 0.0
            and walk_top > 0.0
            and int(row.get("positive_top_bucket_folds", 0) or 0) >= min_positive_folds
            and micro_selected > 0.0
        ):
            eligible.append(candidate)
        else:
            rejected.append(candidate)
    eligible.sort(
        key=lambda row: (
            float(row["walk_forward_mean_selected_mean_net_pnl_total_vs_baseline"]),
            float(row["micro_model_selected_mean_net_pnl_total_vs_baseline"]),
        ),
        reverse=True,
    )
    rejected.sort(
        key=lambda row: (
            float(row["walk_forward_mean_selected_mean_net_pnl_total_vs_baseline"]),
            float(row["micro_model_selected_mean_net_pnl_total_vs_baseline"]),
        ),
        reverse=True,
    )
    return {
        "mode": "source_profile_shadow_assist",
        "status": "active_candidates" if eligible else "observation_only",
        "eligible_source_profiles": eligible,
        "rejected_source_profiles": rejected[:5],
    }


def _history_entry(payload: dict[str, Any]) -> dict[str, Any]:
    walk_forward_summary = payload.get("walk_forward", {}).get("summary", {}) if isinstance(payload.get("walk_forward"), dict) else {}
    assist = payload.get("mlx_shadow_assist", {}) if isinstance(payload.get("mlx_shadow_assist"), dict) else {}
    return {
        "timestamp_utc": str(payload.get("timestamp_utc") or _now_utc()),
        "ok": bool(payload.get("ok")),
        "rows_total": int(((payload.get("dataset") or {}) if isinstance(payload.get("dataset"), dict) else {}).get("rows_total", 0) or 0),
        "raw_selected_mean_net_pnl_total_vs_baseline": float(((payload.get("deltas") or {}) if isinstance(payload.get("deltas"), dict) else {}).get("selected_mean_net_pnl_total_vs_baseline", 0.0) or 0.0),
        "raw_top_bucket_mean_net_pnl_total_vs_baseline": float(((payload.get("deltas") or {}) if isinstance(payload.get("deltas"), dict) else {}).get("top_bucket_mean_net_pnl_total_vs_baseline", 0.0) or 0.0),
        "calibrated_selected_mean_net_pnl_total_vs_baseline": float(((payload.get("calibrated_deltas") or {}) if isinstance(payload.get("calibrated_deltas"), dict) else {}).get("selected_mean_net_pnl_total_vs_baseline", 0.0) or 0.0),
        "walk_forward_calibrated_mean_selected_mean_net_pnl_total_vs_baseline": float(walk_forward_summary.get("calibrated_mean_selected_mean_net_pnl_total_vs_baseline", 0.0) or 0.0),
        "assist_candidate_count": int(len(assist.get("eligible_source_profiles", []))) if isinstance(assist.get("eligible_source_profiles"), list) else 0,
        "recommendations": list(payload.get("recommendations") or []),
    }


def _append_history(path: Path, entry: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, ensure_ascii=True) + "\n")


def _load_history_rows(path: Path, *, limit: int) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as handle:
            for raw in handle:
                line = raw.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except Exception:
                    continue
                if isinstance(payload, dict):
                    rows.append(payload)
    except Exception:
        return []
    return rows[-max(int(limit), 1) :]


def _history_scoreboard(path: Path, *, limit: int = 30) -> dict[str, Any]:
    rows = _load_history_rows(path, limit=limit)
    if not rows:
        return {
            "history_path": str(path),
            "runs_tracked": 0,
            "ok_runs": 0,
            "positive_calibrated_runs": 0,
            "recent_mean_calibrated_selected_mean_net_pnl_total_vs_baseline": 0.0,
            "recent_mean_raw_top_bucket_mean_net_pnl_total_vs_baseline": 0.0,
            "active_assist_candidate_runs": 0,
        }
    calibrated_values = [float(row.get("calibrated_selected_mean_net_pnl_total_vs_baseline", 0.0) or 0.0) for row in rows]
    raw_top_bucket_values = [float(row.get("raw_top_bucket_mean_net_pnl_total_vs_baseline", 0.0) or 0.0) for row in rows]
    return {
        "history_path": str(path),
        "runs_tracked": int(len(rows)),
        "ok_runs": int(sum(1 for row in rows if bool(row.get("ok")))),
        "positive_calibrated_runs": int(sum(1 for value in calibrated_values if value > 0.0)),
        "recent_mean_calibrated_selected_mean_net_pnl_total_vs_baseline": round(float(np.mean(calibrated_values)), 6) if calibrated_values else 0.0,
        "recent_mean_raw_top_bucket_mean_net_pnl_total_vs_baseline": round(float(np.mean(raw_top_bucket_values)), 6) if raw_top_bucket_values else 0.0,
        "active_assist_candidate_runs": int(sum(1 for row in rows if int(row.get("assist_candidate_count", 0) or 0) > 0)),
    }


def disabled_pytorch_replay_canary_payload(project_root: Path, history_path: Path) -> dict[str, Any]:
    return {
        "timestamp_utc": _now_utc(),
        "ok": True,
        "disabled": True,
        "mode": "disabled_mlx_primary",
        "project_root": str(project_root),
        "load": {"source_files": [], "file_count": 0, "rows_scanned": 0, "rows_used": 0},
        "feature_names": FEATURE_NAMES,
        "notes": [
            "PyTorch replay canary is disabled by default so it does not compete with MLX on unified memory during live collection.",
            "MLX remains the default runtime for live, paper, and research feature collection.",
            "Use --force or PYTORCH_REPLAY_CANARY_ENABLED=1 only for an intentional offline sidecar replay window.",
        ],
        "mlx_shadow_assist": {
            "mode": "disabled",
            "status": "disabled",
            "eligible_source_profiles": [],
            "rejected_source_profiles": [],
        },
        "recommendations": [
            "keep_mlx_live_default_backend",
            "keep_pytorch_replay_canary_disabled_during_live_collection",
        ],
        "scoreboard": _history_scoreboard(history_path, limit=30),
    }


def build_pytorch_replay_canary(
    project_root: Path,
    *,
    max_files: int,
    max_rows: int,
    validation_fraction: float,
    min_rows: int,
    min_class_rows: int,
    epochs: int,
    lr: float,
    device: str,
    top_fraction: float,
    walk_forward_folds: int = 3,
) -> dict[str, Any]:
    examples, load_meta = _load_examples(project_root, max_files=max_files, max_rows=max_rows)
    requested_min_rows = max(int(min_rows), 10)
    effective_min_rows = _effective_min_rows(requested_min_rows)
    requested_min_class_rows = max(int(min_class_rows), 1)
    payload: dict[str, Any] = {
        "timestamp_utc": _now_utc(),
        "ok": False,
        "mode": "offline_shadow_replay",
        "load": load_meta,
        "feature_names": FEATURE_NAMES,
        "requested_dataset_requirements": {
            "min_rows": requested_min_rows,
            "min_class_rows": requested_min_class_rows,
        },
        "effective_dataset_requirements": {
            "min_rows": effective_min_rows,
        },
        "notes": [
            "This canary is replay-only and does not alter the live MLX trading path.",
            "It trains a tiny PyTorch classifier on recent paper bridge outcomes and compares it with the score-gap baseline.",
            "The replay probe now prefers event-level PnL labels and falls back to total PnL only when event labels are flat.",
            "Walk-forward replay, source-profile micro-models, and the MLX shadow-assist map stay sidecar-only until they consistently beat baseline.",
        ],
    }
    if len(examples) < int(effective_min_rows):
        payload["reason"] = f"insufficient_rows:{len(examples)}"
        return payload

    timestamps, features, labels, pnl, segments = _prepare_arrays(examples)
    split_idx = _split_index(
        len(features),
        validation_fraction=validation_fraction,
        min_train_rows=max(128, effective_min_rows // 2),
        min_val_rows=max(48, effective_min_rows // 4),
    )
    if split_idx is None:
        payload["reason"] = "invalid_split"
        return payload

    train_pos = int(np.sum(labels[:split_idx] > 0.5))
    train_neg = int(split_idx - train_pos)
    val_pos = int(np.sum(labels[split_idx:] > 0.5))
    val_neg = int(len(labels[split_idx:]) - val_pos)
    effective_min_class_rows = _effective_min_class_rows(requested_min_class_rows, split_idx, len(features) - split_idx)
    payload["effective_dataset_requirements"]["min_class_rows"] = effective_min_class_rows
    if min(train_pos, train_neg, val_pos, val_neg) < int(effective_min_class_rows):
        payload["reason"] = (
            f"insufficient_class_balance:train_pos={train_pos}:train_neg={train_neg}:"
            f"val_pos={val_pos}:val_neg={val_neg}"
        )
        payload["dataset"] = {
            "rows_total": int(len(features)),
            "rows_train": int(split_idx),
            "rows_val": int(len(features) - split_idx),
            "train_positive_rows": train_pos,
            "train_negative_rows": train_neg,
            "val_positive_rows": val_pos,
            "val_negative_rows": val_neg,
        }
        return payload

    low_rows = len(examples) < requested_min_rows
    low_class_balance = min(train_pos, train_neg, val_pos, val_neg) < requested_min_class_rows
    warnings: list[str] = []
    if low_rows:
        warnings.append(
            f"rows_below_requested_floor:{len(examples)}<{requested_min_rows}; running in degraded sample mode"
        )
    if low_class_balance:
        warnings.append(
            "class_balance_below_requested_floor:"
            f"{min(train_pos, train_neg, val_pos, val_neg)}<{requested_min_class_rows}; "
            "running in degraded class-balance mode"
        )

    latest_probe = _train_probe_metrics(
        features[:split_idx],
        labels[:split_idx],
        pnl[:split_idx],
        {name: values[:split_idx] for name, values in segments.items()},
        features[split_idx:],
        labels[split_idx:],
        pnl[split_idx:],
        {name: values[split_idx:] for name, values in segments.items()},
        device=device,
        epochs=epochs,
        lr=lr,
        top_fraction=top_fraction,
    )
    walk_forward = _walk_forward_probe(
        timestamps,
        features,
        labels,
        pnl,
        segments,
        folds=max(int(walk_forward_folds), 1),
        validation_fraction=validation_fraction,
        device=device,
        epochs=epochs,
        lr=lr,
        top_fraction=top_fraction,
    )
    micro_models = _micro_model_probe(
        features,
        labels,
        pnl,
        segments,
        validation_fraction=validation_fraction,
        device=device,
        epochs=epochs,
        lr=lr,
        top_fraction=top_fraction,
    )
    mlx_shadow_assist = _mlx_shadow_assist(walk_forward, micro_models)

    payload.update(
        {
            "ok": True,
            "device": latest_probe["device"],
            "sample_quality": _sample_quality(low_rows, low_class_balance),
            "warnings": warnings,
            "dataset": {
                "rows_total": int(len(features)),
                "rows_train": int(split_idx),
                "rows_val": int(len(features) - split_idx),
                "train_positive_rows": train_pos,
                "train_negative_rows": train_neg,
                "val_positive_rows": val_pos,
                "val_negative_rows": val_neg,
                "val_start_timestamp_utc": str(timestamps[split_idx]) if split_idx < len(timestamps) else "",
                "val_end_timestamp_utc": str(timestamps[-1]) if len(timestamps) else "",
                "train_mean_net_pnl_total": float(np.mean(pnl[:split_idx])) if split_idx else 0.0,
                "val_mean_net_pnl_total": float(np.mean(pnl[split_idx:])) if len(features) > split_idx else 0.0,
                "label_source_counts": load_meta.get("label_source_counts", {}),
            },
            "training": latest_probe["training"],
            "baseline": latest_probe["baseline"],
            "pytorch": latest_probe["pytorch"],
            "pytorch_calibrated": latest_probe["pytorch_calibrated"],
            "threshold_calibration": latest_probe["threshold_calibration"],
            "feature_importance": latest_probe["feature_importance"],
            "segment_deltas": latest_probe["segment_deltas"],
            "deltas": latest_probe["deltas"],
            "calibrated_deltas": latest_probe["calibrated_deltas"],
            "walk_forward": walk_forward,
            "micro_models": micro_models,
            "mlx_shadow_assist": mlx_shadow_assist,
        }
    )

    walk_summary = walk_forward.get("summary", {}) if isinstance(walk_forward, dict) else {}
    recommendations = ["keep_mlx_live_default_backend"]
    if str(payload.get("device") or "") == "mps":
        recommendations.append("candidate_pytorch_replay_shadow_compare_on_mps")
    else:
        recommendations.append("keep_pytorch_replay_canary_cpu_only")
    if low_rows or low_class_balance:
        recommendations.append("expand_paper_history_for_stronger_pytorch_replay_confidence")
    if int(len(mlx_shadow_assist.get("eligible_source_profiles", []))) > 0:
        recommendations.append("candidate_segment_calibrated_pytorch_shadow_filter")
    if float(walk_summary.get("calibrated_mean_selected_mean_net_pnl_total_vs_baseline", 0.0) or 0.0) > 0.0:
        recommendations.append("candidate_walk_forward_pytorch_research_lane")
    if float(payload["deltas"]["top_bucket_mean_net_pnl_total_vs_baseline"]) > 0.0 and not (low_rows or low_class_balance):
        recommendations.append("promote_pytorch_replay_probe_to_recurring_research_check")
    if "candidate_segment_calibrated_pytorch_shadow_filter" not in recommendations and "candidate_walk_forward_pytorch_research_lane" not in recommendations:
        recommendations.append("keep_pytorch_replay_canary_observation_only")
    payload["recommendations"] = recommendations
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Offline PyTorch shadow replay canary over recent paper bridge orders.")
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--history-out", default=str(DEFAULT_HISTORY_OUT))
    parser.add_argument("--max-files", type=int, default=20)
    parser.add_argument("--max-rows", type=int, default=40000)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--min-rows", type=int, default=2000)
    parser.add_argument("--min-class-rows", type=int, default=200)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--device", default="auto", choices=("auto", "mps", "cpu"))
    parser.add_argument("--top-fraction", type=float, default=0.1)
    parser.add_argument("--walk-forward-folds", type=int, default=3)
    parser.add_argument("--force", action="store_true", help="Run the PyTorch canary despite the MLX-primary default.")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    history_path = Path(args.history_out).expanduser().resolve()
    if DEFAULT_ENABLED or bool(args.force):
        payload = build_pytorch_replay_canary(
            project_root,
            max_files=int(args.max_files),
            max_rows=int(args.max_rows),
            validation_fraction=float(args.validation_fraction),
            min_rows=int(args.min_rows),
            min_class_rows=int(args.min_class_rows),
            epochs=int(args.epochs),
            lr=float(args.learning_rate),
            device=str(args.device),
            top_fraction=float(args.top_fraction),
            walk_forward_folds=int(args.walk_forward_folds),
        )
        _append_history(history_path, _history_entry(payload))
        payload["scoreboard"] = _history_scoreboard(history_path, limit=30)
    else:
        payload = disabled_pytorch_replay_canary_payload(project_root, history_path)

    out_path = Path(args.out).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        print(
            "pytorch_replay_canary "
            f"ok={str(bool(payload.get('ok'))).lower()} "
            f"mode={payload.get('mode', 'unknown')} "
            f"device={payload.get('device', 'none')} "
            f"rows={int(((payload.get('dataset') or {}) if isinstance(payload.get('dataset'), dict) else {}).get('rows_total', 0) or 0)}"
        )
    return 0 if bool(payload.get("ok")) else 2


if __name__ == "__main__":
    raise SystemExit(main())
