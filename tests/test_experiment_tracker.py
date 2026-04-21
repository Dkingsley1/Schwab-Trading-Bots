import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import scripts.experiment_tracker as tracker


def test_build_experiment_row_tracks_replayability_bundle(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    (project_root / "governance" / "feature_versions").mkdir(parents=True, exist_ok=True)
    (project_root / "governance" / "walk_forward").mkdir(parents=True, exist_ok=True)
    (project_root / "exports" / "one_numbers").mkdir(parents=True, exist_ok=True)
    (project_root / "datasets").mkdir(parents=True, exist_ok=True)
    (project_root / "models").mkdir(parents=True, exist_ok=True)
    (project_root / "governance" / "health").mkdir(parents=True, exist_ok=True)

    (project_root / "governance" / "feature_versions" / "latest.json").write_text(
        json.dumps({"env_hash": "env-123", "file_hashes": {"a": "b"}}),
        encoding="utf-8",
    )
    (project_root / "governance" / "walk_forward" / "promotion_gate_latest.json").write_text("{}", encoding="utf-8")
    (project_root / "exports" / "one_numbers" / "one_numbers_summary.json").write_text("{}", encoding="utf-8")
    (project_root / "governance" / "content_store").mkdir(parents=True, exist_ok=True)
    (project_root / "datasets" / "train.parquet").write_text("dataset", encoding="utf-8")
    (project_root / "models" / "model.npz").write_text("model", encoding="utf-8")
    (project_root / "governance" / "health" / "replay_end_to_end_latest.json").write_text(
        json.dumps({"replay_hash": "replay-abc"}),
        encoding="utf-8",
    )
    (project_root / "governance" / "content_store" / "latest.json").write_text(
        json.dumps({"manifest_hash": "content-store-hash"}),
        encoding="utf-8",
    )

    row = tracker.build_experiment_row(
        project_root,
        name="shadow_canary",
        status="completed",
        notes="institutional test",
        dataset_file="datasets/train.parquet",
        model_file="models/model.npz",
        replay_file="governance/health/replay_end_to_end_latest.json",
        tags=["canary", "pytorch"],
    )

    assert row["name"] == "shadow_canary"
    assert row["status"] == "completed"
    assert row["tags"] == ["canary", "pytorch"]
    assert row["artifact_hashes"]["datasets/train.parquet"]
    assert row["artifact_hashes"]["models/model.npz"]
    assert row["replayability"]["dataset_hash"]
    assert row["replayability"]["model_hash"]
    assert row["replayability"]["replay_hash"] == "replay-abc"
    assert row["replayability"]["feature_env_hash"] == "env-123"
    assert row["replayability"]["content_store_manifest_hash"] == "content-store-hash"
    assert row["replayability"]["exact_replay_ready"] is True
    assert len(row["replayability"]["bundle_hash"]) == 64
    assert row["experiment_id"].startswith("exp_")
