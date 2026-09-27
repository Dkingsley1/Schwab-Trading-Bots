import json
from datetime import datetime, timedelta, timezone
import subprocess

import numpy as np
import pytest

from scripts.ops import training_dataset_evaluation as evaluation


def test_stale_or_missing_preflight_does_not_fit_models(tmp_path, monkeypatch):
    monkeypatch.setattr(evaluation, "evaluate_dataset", lambda *args: pytest.fail("unexpected model fit"))
    payload = evaluation.build_payload(tmp_path)
    assert not payload["ok"]
    assert payload["blockers"] == ["preflight_or_snapshot_stale"]
    assert not payload["authority_contract"]["runtime_model_write"]


def test_explicit_preflight_does_not_consume_shared_latest(tmp_path, monkeypatch):
    monkeypatch.setattr(evaluation, "load_json", lambda *a: pytest.fail("shared latest read"))
    result = evaluation.build_payload(tmp_path, preflight={})
    assert result["blockers"] == ["preflight_or_snapshot_stale"]


def test_diagnostic_fit_uses_training_partition_and_keeps_holdouts_separate():
    rng = np.random.default_rng(14)
    X = rng.normal(size=(500, 4))
    y = (X[:, 0] > 0).astype(float)
    split = np.asarray(["train"] * 280 + ["purged"] * 20 + ["validation"] * 100 + ["test"] * 100)
    first = evaluation.evaluate_dataset(X, y, split)
    changed = y.copy()
    changed[split == "test"] = 1 - changed[split == "test"]
    changed[split == "purged"] = 1 - changed[split == "purged"]
    second = evaluation.evaluate_dataset(X, changed, split)
    assert first["metrics"]["validation"] == second["metrics"]["validation"]
    assert first["metrics"]["test"]["accuracy"] > 0.9
    assert second["metrics"]["test"]["accuracy"] < 0.1
    assert first["purged_sample_count"] == 20
    assert not evaluation.AUTHORITY["profitability_evidence"]


def test_dataset_outside_preflight_directory_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="outside_preflight"):
        evaluation.load_dataset(tmp_path, {"dataset": {"path": str(tmp_path / "untrusted.npz")}})


def test_dataset_digest_mismatch_is_rejected_before_numpy_load(tmp_path, monkeypatch):
    path = tmp_path / "exports/training/preflight/example.npz"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"wrong bytes")
    monkeypatch.setattr(evaluation.np, "load", lambda *a, **k: pytest.fail("unverified dataset read"))
    with pytest.raises(ValueError, match="digest_mismatch"):
        evaluation.load_dataset(tmp_path, {"dataset": {"path": str(path), "sha256": "wrong"}})


def test_nested_selection_never_uses_outer_holdout_features_or_labels():
    rng = np.random.default_rng(42)
    X = rng.normal(size=(800, 4))
    y = (X[:, 0] + rng.normal(size=800) * 0.4 > 0).astype(float)
    split = np.asarray(["train"] * 500 + ["purged"] * 100 + ["validation"] * 100 + ["test"] * 100)
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    evidence = [{
        "feature_window_started_at_utc": (start + timedelta(minutes=i - 5)).isoformat(),
        "feature_timestamp_utc": (start + timedelta(minutes=i)).isoformat(),
        "label_matured_at_utc": (start + timedelta(minutes=i + 5)).isoformat(),
    } for i in range(800)]
    first = evaluation.evaluate_dataset(X, y, split, evidence, 10)
    X[split != "train"] *= -100
    y[split != "train"] = 1 - y[split != "train"]
    second = evaluation.evaluate_dataset(X, y, split, evidence, 10)
    assert first["training_selection"] == second["training_selection"]
    selection = first["training_selection"]
    assert selection["status"] == "selected"
    assert selection["split_counts"]["purged"] > 0
    assert len(selection["candidates"]) == 4
    assert sum(selection["split_counts"].values()) == 500
    # A long embargo may remove the middle inner window; the later inner window
    # is still training-only evidence and must not be replaced by outer holdouts.
    one_window = evaluation.evaluate_dataset(X, y, split, evidence, 100)["training_selection"]
    assert one_window["status"] == "selected"
    assert one_window["inner_scoring_windows"] == ["test"]


def test_inner_selection_does_not_relax_evidence_floors():
    assert evaluation.select_regularization(np.ones((10, 1)), np.zeros(10), [], 390) == {
        "status": "insufficient_inner_evidence", "selected_c": 1.0,
        "scope": "outer_training_partition_only", "candidates": [],
    }


def test_fresh_preflight_cannot_refresh_old_observations(tmp_path, monkeypatch):
    path = tmp_path / "governance/health/training_dataset_preflight_latest.json"
    path.parent.mkdir(parents=True)
    now = datetime.now(timezone.utc)
    path.write_text(json.dumps({"timestamp_utc": now.isoformat(),
        "snapshot_timestamp_utc": now.isoformat(),
        "snapshot_latest_row_timestamp_utc": (now - timedelta(hours=25)).isoformat()}))
    monkeypatch.setattr(evaluation, "evaluate_dataset", lambda *a: pytest.fail("stale fit"))
    assert evaluation.build_payload(tmp_path)["blockers"] == ["preflight_or_snapshot_stale"]


@pytest.mark.parametrize("failure", [None, 0, 1, 2])
def test_prepare_refreshes_snapshot_before_dataset_and_stops_on_failure(tmp_path, monkeypatch, failure):
    calls = []
    path = tmp_path / "governance/health/runtime_training_snapshot_latest.json"
    path.parent.mkdir(parents=True)

    def run(command, **kwargs):
        calls.append((command, kwargs))
        if len(calls) - 1 == failure:
            raise subprocess.CalledProcessError(2, command)
        if len(calls) == 2:
            path.write_text(json.dumps({"timestamp_utc": datetime.now(timezone.utc).isoformat()}))

    monkeypatch.setattr(evaluation.subprocess, "run", run)
    if failure is None:
        evaluation.prepare_current_datasets(tmp_path)
        assert len(calls) == 3
        assert calls[1][0][1].endswith("build_runtime_training_snapshot.py")
        assert calls[2][0][-1] == "--materialize"
    else:
        with pytest.raises(subprocess.CalledProcessError):
            evaluation.prepare_current_datasets(tmp_path)
        assert len(calls) == failure + 1
    assert all(row[1]["env"]["OMP_NUM_THREADS"] == "1" for row in calls)
    assert all(row[1]["timeout"] <= 180 for row in calls)


def test_snapshot_busy_or_no_fresh_publication_does_not_reuse_old_dataset(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(evaluation.subprocess, "run", lambda *a, **kw: calls.append(a))
    with pytest.raises(ValueError, match="did_not_publish_fresh"):
        evaluation.prepare_current_datasets(tmp_path)
    assert len(calls) == 2
