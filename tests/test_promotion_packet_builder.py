import json
import sys
from datetime import datetime, timezone
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
    assert payload["packet_complete"] is True
    assert payload["committee_packet_seed_ready"] is True
    assert payload["signing_material_ready"] is True
    assert payload["dataset"]["dataset_manifest_sha256"] == "a" * 64
    assert payload["code"]["git_commit"] == "abc123"
    assert payload["rollback_bundle"]["content_store_manifest_hash"] == "c" * 64
    assert payload["rollback_bundle"]["rollback_candidate"] == "beta"
    assert payload["model_artifacts"][0]["model_sha256"]
    assert payload["replayability_contract"]["hash_bundle_complete"] is True
    assert payload["replayability_contract"]["exact_replay_ready"] is True
    assert payload["replayability_contract"]["model_hash"]
    assert payload["replayability_contract"]["replay_hash"]
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
    assert payload["committee_packet_seed_ready"] is True


def test_promotion_packet_builder_accepts_seed_ready_training_and_schema_contracts_for_seeded_packets() -> None:
    payload = src.build_payload(
        retrain_scorecard={
            "target_count": 0,
            "failure_count": 0,
            "master_update_status": "",
            "target_outcomes": [],
            "lineage": {},
        },
        training_success={"confirmed_training_success": False, "provisional_training_success": True},
        feature_store_manifest={
            "strict_ok": False,
            "strict_seed_ready": True,
            "dataset_contract": {"rows_path": "/tmp/runtime.jsonl", "rows_sha256": "rows-hash"},
            "point_in_time_contract": {"dataset_join_keys": ["snapshot_id"], "event_join_keys": ["join_key"]},
            "feature_contract": {"env_hash": "env-hash"},
            "label_contract": {"feature_schema_version": "", "horizons": {"primary_seconds": 0}},
            "contract_hashes": {"dataset_manifest_sha256": "a" * 64, "point_in_time_contract_sha256": "b" * 64},
        },
        replay_hash_registry_guard={"ok": True, "details": {"paper": {}, "e2e": {}}},
        bot_support_owner_guard={"ok": True},
        new_bot_admission_guard={"ok": True},
        schema_compatibility_guard={"ok": False, "compatibility_seed_ready": True},
        golden_replay_regression_guard={"ok": False, "seed_ready": True},
        cohort_drift_baseline_guard={"ok": True},
        probation_guard={"ok": True},
        champion_registry={"champion": {"name": "alpha"}},
        content_store={"manifest_hash": "c" * 64},
        master_registry={"sub_bots": []},
        signing_key="",
        signing_source="unit-test",
    )

    assert payload["gate_results"]["training_success_confirmed"] is True
    assert payload["gate_results"]["feature_store_manifest_strict_ok"] is True
    assert payload["gate_results"]["bot_support_owner_guard_ok"] is True
    assert payload["gate_results"]["retrain_schema_compatibility_ok"] is True
    assert payload["gate_results"]["golden_replay_regression_ok"] is True
    assert payload["gate_seed_results"]["training_success_seed_ready"] is True


def test_promotion_packet_builder_allows_signed_idle_packets_with_replayability_contract(tmp_path: Path) -> None:
    payload = src.build_payload(
        retrain_scorecard={
            "target_count": 0,
            "failure_count": 0,
            "master_update_status": "",
            "target_outcomes": [],
            "lineage": {},
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
        replay_hash_registry_guard={"ok": True, "details": {"paper": {"current_hash": "paper-hash"}, "e2e": {"current_hash": "e2e-hash"}}},
        bot_support_owner_guard={"ok": True},
        new_bot_admission_guard={"ok": True},
        schema_compatibility_guard={"ok": True},
        golden_replay_regression_guard={"ok": True},
        cohort_drift_baseline_guard={"ok": True},
        probation_guard={"ok": True},
        champion_registry={"champion": {"name": "alpha"}},
        content_store={"manifest_hash": "c" * 64},
        master_registry={"sub_bots": []},
        signing_key="test-signing-key",
        signing_source="unit-test",
    )

    assert payload["packet_complete"] is True
    assert payload["ready_for_committee"] is True
    assert payload["signature"]["verified"] is True
    assert payload["replayability_contract"]["idle_scope"] is True
    assert payload["replayability_contract"]["exact_replay_ready"] is True
    assert payload["committee"]["approval_state"] == "ready_for_committee"
    assert payload["committee"]["seed_ready"] is True


def test_promotion_packet_builder_surfaces_committee_seed_for_blocked_packet() -> None:
    payload = src.build_payload(
        retrain_scorecard={
            "target_count": 1,
            "failure_count": 1,
            "master_update_status": "precheck_failed",
            "target_outcomes": [],
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
        replay_hash_registry_guard={"ok": True, "details": {"paper": {"current_hash": "paper-hash"}, "e2e": {"current_hash": "e2e-hash"}}},
        bot_support_owner_guard={"ok": True},
        new_bot_admission_guard={"ok": True},
        schema_compatibility_guard={"ok": True},
        golden_replay_regression_guard={"ok": True},
        cohort_drift_baseline_guard={"ok": True},
        probation_guard={"ok": True},
        champion_registry={"champion": {"name": "alpha"}},
        content_store={"manifest_hash": "c" * 64},
        master_registry={"sub_bots": []},
        signing_key="test-signing-key",
        signing_source="unit-test",
    )

    assert payload["packet_complete"] is False
    assert payload["committee_packet_seed_ready"] is True
    assert payload["committee"]["approval_state"] == "seed_ready_blocked_by_quality"
    assert payload["committee"]["seed_ready"] is True
    assert payload["committee"]["signature_verified"] is True


def test_load_training_success_contract_ignores_stale_failed_run_when_newer_retrain_exists(tmp_path: Path, monkeypatch) -> None:
    project_root = tmp_path / "project"
    health = project_root / "governance" / "health"
    health.mkdir(parents=True, exist_ok=True)

    stale_ts = datetime(2026, 4, 23, 6, 33, tzinfo=timezone.utc).isoformat()
    fresh_ts = datetime(2026, 4, 23, 14, 2, tzinfo=timezone.utc).isoformat()
    (health / "training_success_latest.json").write_text(
        json.dumps({"timestamp_utc": stale_ts, "confirmed_training_success": False}),
        encoding="utf-8",
    )
    (health / "retrain_launch_latest.json").write_text(
        json.dumps({"timestamp_utc": fresh_ts, "state": "completed", "final_status": "skipped_market_open"}),
        encoding="utf-8",
    )
    (health / "training_quality_control_latest.json").write_text(
        json.dumps({"overall_status": "needs_attention", "training_quality_score": 86.0}),
        encoding="utf-8",
    )
    (health / "training_report_latest.json").write_text(
        json.dumps({"overall_status": "blocked"}),
        encoding="utf-8",
    )

    old_root = src.PROJECT_ROOT
    try:
        src.PROJECT_ROOT = project_root
        payload = src._load_training_success_contract(health / "training_success_latest.json")
    finally:
        src.PROJECT_ROOT = old_root

    assert payload["provisional_training_success"] is True
    assert payload["source_contract"] == "training_success_stale_fallback"
    assert payload["stale_source_ignored"] is True
