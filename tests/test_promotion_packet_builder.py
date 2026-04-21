import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import scripts.promotion_packet_builder as src


def test_promotion_packet_builder_captures_dataset_code_model_and_rollback_bundle(tmp_path: Path) -> None:
    model_path = tmp_path / "models" / "brain_refinery_v43_intraday_ultrafast_proxy.npz"
    log_path = tmp_path / "logs" / "brain_refinery_v43_intraday_ultrafast_proxy.json"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    model_path.write_bytes(b"model-bytes")
    log_path.write_text("{}", encoding="utf-8")

    payload = src.build_payload(
        retrain_scorecard={
            "target_count": 1,
            "failure_count": 0,
            "master_update_status": "updated",
            "target_outcomes": [{"bot_id": "brain_refinery_v43_intraday_ultrafast_proxy", "status": "trained"}],
            "lineage": {
                "git_commit": "abc123",
                "weekly_retrain_script_sha256": "f" * 64,
                "registry_backup_before_retrain": "/tmp/registry_backup_before_retrain.json",
            },
        },
        training_success={"confirmed_training_success": True},
        feature_store_manifest={
            "strict_ok": True,
            "dataset_contract": {"rows_path": "/tmp/runtime.jsonl", "rows_sha256": "rows-hash"},
            "point_in_time_contract": {"dataset_join_keys": ["snapshot_id"], "event_join_keys": ["join_key"]},
            "feature_contract": {"env_hash": "env-hash"},
            "label_contract": {"feature_schema_version": "trade_behavior_features_v4", "horizons": {"primary_seconds": 300}},
            "contract_hashes": {"dataset_manifest_sha256": "a" * 64, "point_in_time_contract_sha256": "b" * 64},
        },
        replay_hash_registry_guard={
            "ok": True,
            "details": {"paper": {"current_hash": "paper-hash"}, "e2e": {"current_hash": "e2e-hash"}},
        },
        bot_support_owner_guard={"ok": True},
        new_bot_admission_guard={"ok": True},
        schema_compatibility_guard={"ok": True},
        golden_replay_regression_guard={"ok": True},
        cohort_drift_baseline_guard={"ok": True},
        probation_guard={"ok": True},
        champion_registry={"champion": {"name": "alpha", "rollback_candidate": "beta"}},
        content_store={"manifest_hash": "c" * 64},
        master_registry={
            "sub_bots": [
                {
                    "bot_id": "brain_refinery_v43_intraday_ultrafast_proxy",
                    "active": True,
                    "model_path": str(model_path),
                    "log_file": str(log_path),
                    "test_accuracy": 0.77,
                }
            ]
        },
        signing_key="test-signing-key",
        signing_source="unit-test",
    )

    assert payload["ok"] is True
    assert payload["dataset"]["dataset_manifest_sha256"] == "a" * 64
    assert payload["code"]["git_commit"] == "abc123"
    assert payload["rollback_bundle"]["content_store_manifest_hash"] == "c" * 64
    assert payload["rollback_bundle"]["rollback_candidate"] == "beta"
    assert payload["model_artifacts"][0]["model_sha256"]
    assert len(payload["packet_sha256"]) == 64


def test_promotion_packet_builder_keeps_model_artifacts_empty_without_trained_targets() -> None:
    payload = src.build_payload(
        retrain_scorecard={
            "target_count": 1,
            "failure_count": 1,
            "master_update_status": "skipped_by_flag",
            "target_outcomes": [{"bot_id": "brain_refinery_v43_intraday_ultrafast_proxy", "status": "failed"}],
            "lineage": {"git_commit": "abc123", "weekly_retrain_script_sha256": "f" * 64},
        },
        training_success={"confirmed_training_success": False},
        feature_store_manifest={
            "strict_ok": True,
            "dataset_contract": {"rows_path": "/tmp/runtime.jsonl", "rows_sha256": "rows-hash"},
            "point_in_time_contract": {"dataset_join_keys": ["snapshot_id"], "event_join_keys": ["join_key"]},
            "feature_contract": {"env_hash": "env-hash"},
            "label_contract": {"feature_schema_version": "trade_behavior_features_v4", "horizons": {"primary_seconds": 300}},
            "contract_hashes": {"dataset_manifest_sha256": "a" * 64, "point_in_time_contract_sha256": "b" * 64},
        },
        replay_hash_registry_guard={
            "ok": True,
            "details": {"paper": {"current_hash": "paper-hash"}, "e2e": {"current_hash": "e2e-hash"}},
        },
        bot_support_owner_guard={"ok": True},
        new_bot_admission_guard={"ok": True},
        schema_compatibility_guard={"ok": True},
        golden_replay_regression_guard={"ok": True},
        cohort_drift_baseline_guard={"ok": True},
        probation_guard={"ok": True},
        champion_registry={"champion": {"name": "alpha", "rollback_candidate": "beta"}},
        content_store={"manifest_hash": "c" * 64},
        master_registry={
            "sub_bots": [
                {
                    "bot_id": "brain_refinery_v43_intraday_ultrafast_proxy",
                    "active": True,
                    "model_path": "/tmp/missing.npz",
                }
            ]
        },
        signing_key="test-signing-key",
        signing_source="unit-test",
    )

    assert payload["promotion_scope"]["trained_bot_ids"] == []
    assert payload["model_artifacts"] == []
    assert src._idle_promotion_scope(payload) is True
