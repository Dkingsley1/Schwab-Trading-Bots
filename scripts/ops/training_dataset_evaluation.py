#!/usr/bin/env python3
"""Evaluate prepared market-path datasets without publishing runtime models."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import subprocess
import warnings
import zipfile

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.training_diagnostic_contract import diagnostic_age_hours
from scripts.ops.long_runtime_common import load_json, write_payload
from scripts.ops.training_dataset_preflight import purged_split_evidence

AUTHORITY = {
    "diagnostic_model_fit": True,
    "runtime_model_write": False,
    "registry_write": False,
    "promotion_authority": False,
    "paper_execution_authority": False,
    "live_execution_authority": False,
    "profitability_evidence": False,
}

REGULARIZATION_CANDIDATES = (0.001, 0.01, 0.1, 1.0)


def load_dataset(root: Path, result: dict) -> tuple:
    path = Path(result["dataset"]["path"]).resolve()
    if not path.is_relative_to((root / "exports/training/preflight").resolve()):
        raise ValueError("dataset_outside_preflight_directory")
    if path.stat().st_size > 32 * 1024 * 1024:
        raise ValueError("dataset_byte_budget_exceeded")
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != result["dataset"]["sha256"]:
        raise ValueError("dataset_digest_mismatch")
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        if sum(item.file_size for item in archive.infolist()) > 64 * 1024 * 1024:
            raise ValueError("dataset_expansion_budget_exceeded")
    with np.load(io.BytesIO(data), allow_pickle=False) as stored:
        X = stored["features"]
        y = stored["labels"].reshape(-1)
        split = stored["split"]
        eligible = stored["eligible_mask"]
        evidence = json.loads(str(stored["evidence_json"]))
        receipt = json.loads(str(stored["receipt_json"]))
    if receipt != result["receipt"]:
        raise ValueError("dataset_receipt_mismatch")
    for relative, digest in receipt["source_sha256"].items():
        source = (root / relative).resolve()
        if not source.is_relative_to(root.resolve()) or hashlib.sha256(source.read_bytes()).hexdigest() != digest:
            raise ValueError("dataset_source_changed")
    if X.ndim != 2 or not 1 <= len(y) <= 2000 or not 1 <= X.shape[1] <= 2048:
        raise ValueError("dataset_shape_budget_exceeded")
    if not (len(X) == len(y) == len(split) == len(eligible) == len(evidence)):
        raise ValueError("dataset_lineage_alignment_failed")
    if not np.isfinite(X).all() or not np.isfinite(y).all() or not np.isin(y, [0, 1]).all():
        raise ValueError("dataset_values_invalid")
    embargo = int(receipt["preparation_settings"]["split_policy"].get("embargo_minutes", 390))
    verified_split = np.asarray([row["train_validation_test_split"] for row in purged_split_evidence(evidence, embargo)])
    if not np.array_equal(split, verified_split) or not np.array_equal(eligible, np.isin(split, ["train", "validation", "test"])):
        raise ValueError("dataset_partition_validation_failed")
    if not np.array_equal(y, np.asarray([row["label_value"] for row in evidence])):
        raise ValueError("dataset_labels_disagree_with_lineage")
    policy = receipt["preparation_settings"]["split_policy"]
    for name, minimum in (("train", 144), ("validation", 48), ("test", 48)):
        labels = y[split == name]
        if len(labels) < int(policy.get(f"minimum_{name}_samples", minimum)) or len(np.unique(labels)) != 2:
            raise ValueError(f"{name}_evidence_insufficient")
    return X, y, split, evidence, embargo


def _model(regularization: float):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    return make_pipeline(StandardScaler(), LogisticRegression(
        solver="liblinear", C=regularization, max_iter=200, random_state=0,
    ))


def select_regularization(X: np.ndarray, y: np.ndarray, evidence: list[dict], embargo: int) -> dict:
    """Selection is nested entirely inside the outer training partition."""
    from sklearn.metrics import brier_score_loss

    result = {"status": "insufficient_inner_evidence", "selected_c": 1.0,
              "scope": "outer_training_partition_only", "candidates": []}
    if not evidence:
        return result
    inner = np.asarray([row["train_validation_test_split"]
                        for row in purged_split_evidence(evidence, embargo)])
    result["split_counts"] = {name: int((inner == name).sum())
                              for name in ("train", "validation", "test", "purged")}
    training_labels = y[inner == "train"]
    if len(training_labels) < 64 or len(np.unique(training_labels)) != 2:
        return result
    scoring_windows = [name for name in ("validation", "test")
                       if int((inner == name).sum()) >= 24 and len(np.unique(y[inner == name])) == 2]
    if not scoring_windows:
        return result
    result["inner_scoring_windows"] = scoring_windows
    for value in REGULARIZATION_CANDIDATES:
        model = _model(value)
        model.fit(X[inner == "train"], y[inner == "train"])
        scores = [float(brier_score_loss(y[inner == name],
                  model.predict_proba(X[inner == name])[:, 1]))
                  for name in scoring_windows]
        result["candidates"].append({"c": value, "mean_inner_brier_score": float(np.mean(scores))})
    chosen = min(result["candidates"], key=lambda row: (row["mean_inner_brier_score"], row["c"]))
    result.update(status="selected", selected_c=chosen["c"])
    return result


def evaluate_dataset(X: np.ndarray, y: np.ndarray, split: np.ndarray,
                     evidence: list[dict] | None = None, embargo: int = 390) -> dict:
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, brier_score_loss
    from threadpoolctl import threadpool_limits

    training = split == "train"
    baseline_probability = float(np.mean(y[training]))
    with threadpool_limits(limits=1), warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        selection = select_regularization(X[training], y[training],
            [row for row, keep in zip(evidence, training) if keep] if evidence else [], embargo)
        model = _model(selection["selected_c"])
        model.fit(X[training], y[training])
        metrics = {}
        for name in ("validation", "test"):
            mask = split == name
            actual = y[mask]
            probability = model.predict_proba(X[mask])[:, 1]
            predicted = probability >= 0.5
            baseline = np.full(len(actual), baseline_probability >= 0.5)
            accuracy = float(accuracy_score(actual, predicted))
            baseline_accuracy = float(accuracy_score(actual, baseline))
            metrics[name] = {
                "sample_count": len(actual),
                "positive_label_rate": float(np.mean(actual)),
                "accuracy": accuracy,
                "balanced_accuracy": float(balanced_accuracy_score(actual, predicted)),
                "brier_score": float(brier_score_loss(actual, probability)),
                "baseline_accuracy": baseline_accuracy,
                "baseline_brier_score": float(brier_score_loss(actual, np.full(len(actual), baseline_probability))),
                "accuracy_lift_vs_training_majority": accuracy - baseline_accuracy,
            }
    return {
        "status": "evaluated", "model": "standard_scaler_logistic_regression_nested_purged_v2",
        "training_selection": selection,
        "train_sample_count": int(training.sum()), "purged_sample_count": int((split == "purged").sum()),
        "metrics": metrics,
        "diagnostic_quality_passed": all(
            row["accuracy_lift_vs_training_majority"] > 0
            and row["balanced_accuracy"] > 0.5
            and row["brier_score"] < row["baseline_brier_score"]
            for row in metrics.values()
        ),
        "positive_accuracy_lift_on_both_holdouts": all(row["accuracy_lift_vs_training_majority"] > 0 for row in metrics.values()),
        "interpretation": "diagnostic market-path prediction; correlated samples and repeated holdout reuse do not establish independent trading edge",
    }


def build_payload(root: Path = PROJECT_ROOT, *, preflight: dict | None = None) -> dict:
    now = datetime.now(timezone.utc)
    if preflight is None:
        preflight = load_json(root / "governance/health/training_dataset_preflight_latest.json")
    payload = {
        "timestamp_utc": now.isoformat(), "overall_status": "blocked", "ok": False,
        "authority_contract": dict(AUTHORITY), "results": [], "blockers": [],
        "preflight_timestamp_utc": preflight.get("timestamp_utc"),
        "evaluation_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    age = diagnostic_age_hours(preflight, now)
    snapshot_age = diagnostic_age_hours({"timestamp_utc": preflight.get("snapshot_timestamp_utc")}, now)
    content_age = diagnostic_age_hours({"timestamp_utc": preflight.get("snapshot_latest_row_timestamp_utc")}, now)
    if age is None or age > 24 or snapshot_age is None or snapshot_age > 24 or content_age is None or content_age > 24:
        payload["blockers"] = ["preflight_or_snapshot_stale"]
        return payload
    candidates = [row for row in preflight.get("results", []) if row.get("data_checks_passed") and row.get("dataset")]
    for result in candidates[:10]:
        output = {"bot_id": result["bot_id"], "dataset_sha256": result["dataset"]["sha256"]}
        try:
            output.update(evaluate_dataset(*load_dataset(root, result)))
        except Exception as exc:
            output.update(status="blocked", reason=f"{type(exc).__name__}:{exc}")
        payload["results"].append(output)
    evaluated = [row for row in payload["results"] if row["status"] == "evaluated"]
    payload.update(
        overall_status="evaluated" if evaluated else "blocked",
        ok=bool(evaluated) and len(evaluated) == len(candidates),
        evaluated_bot_count=len(evaluated),
        positive_holdout_lift_count=sum(row["positive_accuracy_lift_on_both_holdouts"] for row in evaluated),
        diagnostic_quality_passed_count=sum(row["diagnostic_quality_passed"] for row in evaluated),
    )
    if not candidates:
        payload["blockers"].append("no_data_ready_materialized_datasets")
    return payload


def prepare_current_datasets(root: Path) -> None:
    env = {**os.environ, "BOT_MLX_DISABLE": "1", "OPENBLAS_NUM_THREADS": "1",
           "VECLIB_MAXIMUM_THREADS": "1", "OMP_NUM_THREADS": "1"}
    steps = (
        ("scripts/resource_guard.py", ["--profile", "refresh"], 20),
        ("scripts/build_runtime_training_snapshot.py", ["--reuse-if-fresh-minutes", "15",
          "--max-runtime-seconds", "150", "--incremental-max-runtime-seconds", "30",
          "--incremental-max-candidate-rows", "25000"], 180),
        ("scripts/ops/training_dataset_preflight.py", ["--materialize"], 120),
    )
    for script, arguments, timeout in steps:
        subprocess.run([sys.executable, str(root / script), *arguments],
                       check=True, capture_output=True, text=True, timeout=timeout, env=env)
        if script.endswith("build_runtime_training_snapshot.py"):
            snapshot = load_json(root / "governance/health/runtime_training_snapshot_latest.json")
            age = diagnostic_age_hours(snapshot, datetime.now(timezone.utc))
            if age is None or age > 15 / 60:
                raise ValueError("snapshot_refresh_did_not_publish_fresh_evidence")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--prepare", action="store_true", help="Resource-guarded snapshot refresh (150s), then dataset preparation (120s).")
    args = parser.parse_args()
    if args.prepare:
        try:
            prepare_current_datasets(PROJECT_ROOT)
        except (subprocess.SubprocessError, OSError, ValueError) as exc:
            payload = {"timestamp_utc": datetime.now(timezone.utc).isoformat(), "overall_status": "blocked", "ok": False,
                       "authority_contract": dict(AUTHORITY), "blockers": [f"preparation_failed:{type(exc).__name__}"], "results": []}
            write_payload(PROJECT_ROOT / "governance/health/training_dataset_evaluation_latest.json", payload)
            print(json.dumps(payload))
            return 2
    payload = build_payload()
    write_payload(PROJECT_ROOT / "governance/health/training_dataset_evaluation_latest.json", payload)
    print(json.dumps(payload, indent=2) if args.json else f"Dataset evaluation: {payload['overall_status']}; evaluated={payload.get('evaluated_bot_count', 0)}")
    return 0 if payload["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
