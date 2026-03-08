import argparse
import glob
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = PROJECT_ROOT / "data" / "trade_history" / "trade_learning_dataset.json"
MODELS_DIR = PROJECT_ROOT / "models"
OUT_DIR = PROJECT_ROOT / "exports" / "sql_reports"
LATEST_HEALTH = PROJECT_ROOT / "governance" / "health" / "replay_feature_ablation_latest.json"

LABEL_TO_ID = {"negative": 0, "neutral": 1, "positive": 2}


def _load_json(path: Path) -> dict[str, Any]:
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def _latest_model() -> Path | None:
    rows = sorted(glob.glob(str(MODELS_DIR / "trade_behavior_policy_*.npz")))
    return Path(rows[-1]) if rows else None


def _macro_f1(cm: np.ndarray) -> float:
    f1s = []
    for i in range(cm.shape[0]):
        tp = float(cm[i, i])
        fp = float(np.sum(cm[:, i]) - tp)
        fn = float(np.sum(cm[i, :]) - tp)
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0
        f1s.append(f1)
    return float(np.mean(f1s)) if f1s else 0.0


def _balanced_acc(cm: np.ndarray) -> float:
    recalls = []
    for i in range(cm.shape[0]):
        denom = float(np.sum(cm[i, :]))
        recalls.append((float(cm[i, i]) / denom) if denom > 0 else 0.0)
    return float(np.mean(recalls)) if recalls else 0.0


def _eval(X: np.ndarray, y: np.ndarray, W: np.ndarray, b: np.ndarray, temp: float, bias: np.ndarray) -> dict[str, float]:
    logits = (X @ W + b) / max(float(temp), 1e-6)
    logits = logits + bias.reshape(1, -1)
    pred = np.argmax(logits, axis=1)
    n = len(y)
    cm = np.zeros((3, 3), dtype=np.int64)
    for t, p in zip(y, pred):
        if 0 <= int(t) < 3 and 0 <= int(p) < 3:
            cm[int(t), int(p)] += 1
    acc = float(np.mean(pred == y)) if n > 0 else 0.0
    return {
        "accuracy": acc,
        "macro_f1": _macro_f1(cm),
        "balanced_accuracy": _balanced_acc(cm),
    }


def _parse_model_feature_names(arr: Any) -> list[str]:
    try:
        files = set(arr.files)
    except Exception:
        files = set()
    if "feature_names" not in files:
        return []

    raw = np.asarray(arr["feature_names"]).reshape(-1)
    out: list[str] = []
    for value in raw.tolist():
        if isinstance(value, bytes):
            name = value.decode("utf-8", errors="ignore").strip()
        else:
            name = str(value).strip()
        if name:
            out.append(name)
    return out


def _align_features(
    *,
    X: np.ndarray,
    feature_names: list[str],
    W: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
    model_feature_names: list[str],
) -> tuple[np.ndarray, list[str], np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    dataset_dim = int(X.shape[1]) if X.ndim == 2 else 0
    model_dim = int(W.shape[0]) if W.ndim == 2 else 0
    mu_dim = int(mu.shape[0]) if mu.ndim == 1 else int(np.asarray(mu).reshape(-1).shape[0])
    sigma_dim = int(sigma.shape[0]) if sigma.ndim == 1 else int(np.asarray(sigma).reshape(-1).shape[0])

    effective_dim = min(model_dim, mu_dim, sigma_dim)
    if effective_dim <= 0:
        raise ValueError(
            f"invalid_model_dims model_dim={model_dim} mu_dim={mu_dim} sigma_dim={sigma_dim}"
        )

    index_strategy = "exact"
    aligned_idx: list[int] = []

    if model_feature_names and feature_names:
        dataset_index_by_name = {str(name): i for i, name in enumerate(feature_names)}
        seen: set[int] = set()
        mapped: list[int] = []
        for name in model_feature_names:
            idx = dataset_index_by_name.get(str(name))
            if idx is None or idx in seen:
                continue
            seen.add(idx)
            mapped.append(int(idx))
            if len(mapped) >= effective_dim:
                break
        if len(mapped) >= effective_dim:
            aligned_idx = mapped[:effective_dim]
            index_strategy = "name_map"

    if not aligned_idx:
        if dataset_dim < effective_dim:
            raise ValueError(
                f"dataset_dim={dataset_dim} is smaller than required_model_dim={effective_dim}"
            )
        aligned_idx = list(range(effective_dim))
        if dataset_dim > effective_dim:
            index_strategy = "prefix_truncate"

    X_aligned = X[:, aligned_idx]
    names_aligned = [feature_names[i] if i < len(feature_names) else f"feature_{i}" for i in aligned_idx]

    W_aligned = W[:effective_dim, :]
    mu_aligned = np.asarray(mu).reshape(-1)[:effective_dim]
    sigma_aligned = np.asarray(sigma).reshape(-1)[:effective_dim]

    alignment = {
        "strategy": index_strategy,
        "dataset_dim": dataset_dim,
        "model_dim": model_dim,
        "mu_dim": mu_dim,
        "sigma_dim": sigma_dim,
        "effective_dim": int(effective_dim),
        "mapped_feature_count": int(len(aligned_idx)),
        "model_feature_names_available": bool(model_feature_names),
    }

    return X_aligned, names_aligned, W_aligned, mu_aligned, sigma_aligned, alignment


def main() -> int:
    ap = argparse.ArgumentParser(description="Ablation report for replay-health features in behavior model.")
    ap.add_argument("--dataset", default=str(DATASET_PATH))
    ap.add_argument("--model", default="")
    ap.add_argument("--out-file", default="")
    ap.add_argument(
        "--require-model-feature-names",
        action=argparse.BooleanOptionalAction,
        default=str(os.getenv("REPLAY_ABLATION_REQUIRE_MODEL_FEATURE_NAMES", "1")).strip() == "1",
    )
    ap.add_argument(
        "--require-full-dim-match",
        action=argparse.BooleanOptionalAction,
        default=str(os.getenv("REPLAY_ABLATION_REQUIRE_FULL_DIM_MATCH", "1")).strip() == "1",
    )
    ap.add_argument(
        "--require-name-map",
        action=argparse.BooleanOptionalAction,
        default=str(os.getenv("REPLAY_ABLATION_REQUIRE_NAME_MAP", "0")).strip() == "1",
    )
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    ds = _load_json(Path(args.dataset))
    rows = ds.get("data") if isinstance(ds.get("data"), list) else []
    if not rows:
        rows = ds.get("examples") if isinstance(ds.get("examples"), list) else []
    if not rows:
        print("missing_examples")
        return 2

    feature_names = [str(x) for x in (ds.get("feature_names") or []) if str(x)]
    if not feature_names:
        print("missing_feature_names")
        return 2

    X = []
    y = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        f = row.get("features") if isinstance(row.get("features"), list) else []
        if len(f) != len(feature_names):
            continue
        label = str(row.get("label", "")).strip().lower()
        if label not in LABEL_TO_ID:
            continue
        X.append([float(v) for v in f])
        y.append(LABEL_TO_ID[label])

    if not X:
        print("no_valid_rows")
        return 2

    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.int64)

    model_path = Path(args.model) if args.model else _latest_model()
    if model_path is None or (not model_path.exists()):
        print("missing_model")
        return 2

    arr = np.load(model_path, allow_pickle=False)
    W = np.asarray(arr["W"], dtype=np.float64)
    b = np.asarray(arr["b"], dtype=np.float64)
    mu = np.asarray(arr["mu"], dtype=np.float64)
    sigma = np.asarray(arr["sigma"], dtype=np.float64)
    temp = float(np.asarray(arr.get("temperature", np.asarray([1.0]))).reshape(-1)[0])
    bias = np.asarray(arr.get("class_logit_bias", np.zeros((3,), dtype=np.float32)), dtype=np.float64).reshape(-1)
    if bias.shape[0] != 3:
        bias = np.zeros((3,), dtype=np.float64)

    model_feature_names = _parse_model_feature_names(arr)

    try:
        X_aligned, feature_names_aligned, W_aligned, mu_aligned, sigma_aligned, alignment = _align_features(
            X=X,
            feature_names=feature_names,
            W=W,
            mu=mu,
            sigma=sigma,
            model_feature_names=model_feature_names,
        )
    except Exception as exc:
        print(f"feature_alignment_failed:{exc}")
        return 2

    Xn = (X_aligned - mu_aligned.reshape(1, -1)) / np.where(
        np.abs(sigma_aligned) < 1e-8,
        1.0,
        sigma_aligned,
    ).reshape(1, -1)

    idx_e2e = [
        i for i, n in enumerate(feature_names_aligned)
        if n in {"snapshot_e2e_replay_ok", "snapshot_e2e_hash_match"}
    ]
    idx_paper = [
        i for i, n in enumerate(feature_names_aligned)
        if n in {"snapshot_paper_replay_ok", "snapshot_paper_replay_hash_match"}
    ]

    base = _eval(Xn, y, W_aligned, b, temp, bias)

    Xe = Xn.copy()
    if idx_e2e:
        Xe[:, idx_e2e] = 0.0
    no_e2e = _eval(Xe, y, W_aligned, b, temp, bias)

    Xp = Xn.copy()
    if idx_paper:
        Xp[:, idx_paper] = 0.0
    no_paper = _eval(Xp, y, W_aligned, b, temp, bias)

    strict_failed: list[str] = []
    if args.require_model_feature_names and (not bool(alignment.get("model_feature_names_available", False))):
        strict_failed.append("model_feature_names_missing")
    if args.require_full_dim_match and (int(alignment.get("effective_dim", 0) or 0) != int(alignment.get("dataset_dim", 0) or 0)):
        strict_failed.append("feature_dim_not_fully_aligned")
    if args.require_name_map and str(alignment.get("strategy", "")).strip().lower() not in {"name_map", "exact"}:
        strict_failed.append("name_map_alignment_required")

    out = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "ok": len(strict_failed) == 0,
        "model_path": str(model_path),
        "dataset_path": str(Path(args.dataset)),
        "rows": int(Xn.shape[0]),
        "feature_dim": int(Xn.shape[1]),
        "dimension_alignment": alignment,
        "ablation": {
            "baseline": base,
            "without_e2e_replay": no_e2e,
            "without_paper_replay": no_paper,
            "e2e_feature_count": int(len(idx_e2e)),
            "paper_feature_count": int(len(idx_paper)),
        },
        "delta": {
            "macro_f1_no_e2e_minus_base": float(no_e2e["macro_f1"] - base["macro_f1"]),
            "macro_f1_no_paper_minus_base": float(no_paper["macro_f1"] - base["macro_f1"]),
            "balanced_accuracy_no_e2e_minus_base": float(no_e2e["balanced_accuracy"] - base["balanced_accuracy"]),
            "balanced_accuracy_no_paper_minus_base": float(no_paper["balanced_accuracy"] - base["balanced_accuracy"]),
        },
        "failed_checks": strict_failed,
        "strict_checks": {
            "require_model_feature_names": bool(args.require_model_feature_names),
            "require_full_dim_match": bool(args.require_full_dim_match),
            "require_name_map": bool(args.require_name_map),
        },
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_path = Path(args.out_file) if args.out_file else (OUT_DIR / f"replay_feature_ablation_{ts}.json")
    out_path.write_text(json.dumps(out, ensure_ascii=True, indent=2), encoding="utf-8")
    LATEST_HEALTH.parent.mkdir(parents=True, exist_ok=True)
    LATEST_HEALTH.write_text(json.dumps(out, ensure_ascii=True, indent=2), encoding="utf-8")

    if args.json:
        print(json.dumps(out, ensure_ascii=True))
    else:
        print(f"replay_feature_ablation_ok={int(len(strict_failed) == 0)} out_file={out_path}")
    return 0 if len(strict_failed) == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
