import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import scripts.experiment_tracker as tracker


def test_signing_key_is_created_and_repaired_with_owner_only_permissions(tmp_path: Path) -> None:
    key_path = tmp_path / "immutable_ledger_signing_key.txt"
    secret = tracker._ensure_signing_key(key_path)

    assert len(secret) == 64
    assert key_path.stat().st_mode & 0o077 == 0

    key_path.chmod(0o644)
    assert tracker._ensure_signing_key(key_path) == secret
    assert key_path.stat().st_mode & 0o077 == 0


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
    signing_key = project_root / "governance" / "experiments" / "immutable_ledger_signing_key.txt"
    signing_key.parent.mkdir(parents=True, exist_ok=True)
    signing_key.write_text("ledger-secret", encoding="utf-8")
    approval = project_root / "governance" / "approvals" / "committee.json"
    rollback = project_root / "governance" / "releases" / "rollback.json"
    deploy = project_root / "governance" / "releases" / "deploy.json"
    approval.parent.mkdir(parents=True, exist_ok=True)
    rollback.parent.mkdir(parents=True, exist_ok=True)
    approval.write_text(json.dumps({"approved": True}), encoding="utf-8")
    rollback.write_text(json.dumps({"rollback": "bundle-a"}), encoding="utf-8")
    deploy.write_text(json.dumps({"deploy": "bundle-a"}), encoding="utf-8")

    row = tracker.build_experiment_row(
        project_root,
        name="shadow_canary",
        status="completed",
        notes="institutional test",
        dataset_file="datasets/train.parquet",
        model_file="models/model.npz",
        replay_file="governance/health/replay_end_to_end_latest.json",
        tags=["canary", "pytorch"],
        event_type="promotion_candidate",
        approval_file=str(approval),
        rollback_file=str(rollback),
        deploy_file=str(deploy),
        signing_secret="ledger-secret",
        signing_key_id="immutable_ledger_signing_key.txt",
    )

    assert row["name"] == "shadow_canary"
    assert row["status"] == "completed"
    assert row["event_type"] == "promotion_candidate"
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
    assert row["attestations"]["attestation_ready"] is True
    assert row["ledger_contract"]["signature_ready"] is True
    assert row["experiment_id"].startswith("exp_")

    summary = tracker.write_experiment_artifacts(project_root, row)
    assert summary["ledger_row_count"] == 1
    assert summary["overall_status"] == "ready"
    assert summary["latest_exact_replay_ready"] is True
    assert summary["latest_signature_ready"] is True
    assert summary["latest_attestation_ready"] is True
