import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.ops import training_lineage_manifest as src


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def test_training_lineage_manifest_reports_ready_when_bundle_is_complete(tmp_path: Path) -> None:
    health_root = tmp_path / "governance" / "health"
    experiments_root = tmp_path / "governance" / "experiments"
    feature_store_root = tmp_path / "governance" / "feature_store"

    _write_json(
        feature_store_root / "latest.json",
        {
            "ok": True,
            "lineage_schema_version": 2,
            "dataset_contract": {"rows_sha256": "rows-hash"},
            "point_in_time_contract": {"dataset_join_keys": ["snapshot_id", "symbol"]},
        },
    )
    _write_json(health_root / "replay_hash_registry_guard_latest.json", {"ok": True})
    _write_json(health_root / "paper_replay_drill_latest.json", {"ok": True})
    _write_json(health_root / "replay_end_to_end_latest.json", {"ok": True})
    _write_json(health_root / "promotion_quality_gate_latest.json", {"ok": True})
    _write_json(
        tmp_path / "governance" / "champion_challenger" / "promotion_autopilot_packet_latest.json",
        {"packet_complete": True, "signature_verified": True},
    )
    _write_json(tmp_path / "governance" / "research" / "multiple_testing_guard_latest.json", {"ok": True})
    _write_json(tmp_path / "governance" / "research" / "decay_monitor_latest.json", {"overall_status": "ready"})
    _write_json(health_root / "training_report_latest.json", {"summary": {"confirmed_training_success": True}})
    _write_json(health_root / "snapshot_coverage_latest.json", {"ok": True})
    experiments_root.mkdir(parents=True, exist_ok=True)
    (experiments_root / "experiment_registry.jsonl").write_text(
        json.dumps(
            {
                "experiment_id": "exp_ready",
                "replayability": {
                    "bundle_hash": "bundle-hash",
                    "dataset_hash": "dataset-hash",
                    "model_hash": "model-hash",
                    "replay_hash": "replay-hash",
                    "exact_replay_ready": True,
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    payload = src.build_payload(tmp_path)

    assert payload["overall_status"] == "ready"
    assert payload["lineage_contract_ready"] is True
    assert payload["promotion_bundle_ready"] is True
    assert payload["hash_bundle_complete"] is True
    assert payload["promotion_packet_ready"] is True
    assert payload["missing_contracts"] == []


def test_training_lineage_manifest_blocks_when_hashes_and_guards_are_missing(tmp_path: Path) -> None:
    health_root = tmp_path / "governance" / "health"
    experiments_root = tmp_path / "governance" / "experiments"
    feature_store_root = tmp_path / "governance" / "feature_store"

    _write_json(
        feature_store_root / "latest.json",
        {
            "ok": False,
            "lineage_schema_version": 1,
            "dataset_contract": {"rows_sha256": ""},
            "point_in_time_contract": {"dataset_join_keys": []},
        },
    )
    _write_json(health_root / "replay_hash_registry_guard_latest.json", {"ok": False})
    _write_json(health_root / "paper_replay_drill_latest.json", {"ok": False})
    _write_json(health_root / "replay_end_to_end_latest.json", {"ok": True})
    _write_json(health_root / "promotion_quality_gate_latest.json", {"ok": False})
    _write_json(health_root / "training_report_latest.json", {"summary": {"confirmed_training_success": False}})
    experiments_root.mkdir(parents=True, exist_ok=True)
    (experiments_root / "experiment_registry.jsonl").write_text(
        json.dumps({"experiment_id": "exp_blocked", "replayability": {"exact_replay_ready": False}}) + "\n",
        encoding="utf-8",
    )

    payload = src.build_payload(tmp_path)

    assert payload["overall_status"] == "blocked"
    assert payload["lineage_contract_ready"] is False
    assert "bundle_hashes" in payload["missing_contracts"]
    assert "replay_hash_registry_guard" in payload["missing_contracts"]


def test_training_lineage_manifest_recognizes_contract_backed_research_and_packet_seed(tmp_path: Path) -> None:
    health_root = tmp_path / "governance" / "health"
    experiments_root = tmp_path / "governance" / "experiments"
    feature_store_root = tmp_path / "governance" / "feature_store"

    _write_json(
        feature_store_root / "latest.json",
        {
            "ok": True,
            "lineage_schema_version": 2,
            "dataset_contract": {"rows_sha256": "rows-hash"},
            "point_in_time_contract": {"dataset_join_keys": ["snapshot_id", "symbol"]},
        },
    )
    _write_json(health_root / "replay_hash_registry_guard_latest.json", {"ok": False})
    _write_json(health_root / "paper_replay_drill_latest.json", {"ok": False})
    _write_json(health_root / "replay_end_to_end_latest.json", {"ok": True})
    _write_json(health_root / "promotion_quality_gate_latest.json", {"ok": False})
    _write_json(
        tmp_path / "governance" / "champion_challenger" / "promotion_autopilot_packet_latest.json",
        {
            "packet_complete": False,
            "signature_verified": False,
            "signed_bundle_contract": {"packet_sha256": "seeded-packet"},
            "evidence_bundle": {"source_count": 4},
        },
    )
    _write_json(
        tmp_path / "governance" / "research" / "multiple_testing_guard_latest.json",
        {
            "ok": False,
            "family_size": 270,
            "correction_method": "benjamini_hochberg_fdr",
            "failed_checks": [],
        },
    )
    _write_json(tmp_path / "governance" / "research" / "decay_monitor_latest.json", {"overall_status": "ready"})
    _write_json(health_root / "training_report_latest.json", {"summary": {"confirmed_training_success": False}})
    _write_json(health_root / "snapshot_coverage_latest.json", {"ok": True})
    experiments_root.mkdir(parents=True, exist_ok=True)
    (experiments_root / "experiment_registry.jsonl").write_text(
        json.dumps({"experiment_id": "exp_partial", "replayability": {"exact_replay_ready": False}}) + "\n",
        encoding="utf-8",
    )

    payload = src.build_payload(tmp_path)

    assert payload["overall_status"] == "needs_attention"
    assert payload["multiple_testing_ready"] is True
    assert payload["multiple_testing_contract_present"] is True
    assert payload["promotion_packet_seed_ready"] is True
    assert payload["promotion_packet_ready"] is False
    assert payload["repairable_lineage_contract"]["lineage_recovery_ready"] is False


def test_training_lineage_manifest_degrades_when_recovery_contract_is_seeded(tmp_path: Path) -> None:
    health_root = tmp_path / "governance" / "health"
    experiments_root = tmp_path / "governance" / "experiments"
    feature_store_root = tmp_path / "governance" / "feature_store"

    _write_json(
        feature_store_root / "latest.json",
        {
            "ok": True,
            "lineage_schema_version": 2,
            "dataset_contract": {"rows_sha256": "rows-hash"},
            "point_in_time_contract": {"dataset_join_keys": ["snapshot_id", "symbol"]},
        },
    )
    _write_json(health_root / "replay_hash_registry_guard_latest.json", {"ok": False})
    _write_json(health_root / "paper_replay_drill_latest.json", {"ok": True})
    _write_json(health_root / "replay_end_to_end_latest.json", {"ok": True})
    _write_json(health_root / "promotion_quality_gate_latest.json", {"ok": False})
    _write_json(
        tmp_path / "governance" / "champion_challenger" / "promotion_autopilot_packet_latest.json",
        {
            "packet_complete": False,
            "signature_verified": False,
            "signed_bundle_contract": {"packet_sha256": "seeded-packet"},
            "evidence_bundle": {"source_count": 4},
        },
    )
    _write_json(
        tmp_path / "governance" / "research" / "multiple_testing_guard_latest.json",
        {
            "ok": False,
            "family_size": 120,
            "correction_method": "benjamini_hochberg_fdr",
            "failed_checks": [],
        },
    )
    _write_json(tmp_path / "governance" / "research" / "decay_monitor_latest.json", {"overall_status": "ready"})
    _write_json(health_root / "training_report_latest.json", {"summary": {"confirmed_training_success": False}})
    _write_json(health_root / "snapshot_coverage_latest.json", {"ok": True})
    experiments_root.mkdir(parents=True, exist_ok=True)
    (experiments_root / "experiment_registry.jsonl").write_text(
        json.dumps(
            {
                "experiment_id": "exp_recovering",
                "replayability": {
                    "dataset_hash": "dataset-hash",
                    "model_hash": "model-hash",
                    "replay_hash": "replay-hash",
                    "exact_replay_ready": False,
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    payload = src.build_payload(tmp_path)

    assert payload["overall_status"] == "degraded"
    assert payload["repairable_lineage_contract"]["lineage_recovery_ready"] is True


def test_training_lineage_manifest_credits_stronger_provisional_packet_lineage(tmp_path: Path) -> None:
    health_root = tmp_path / "governance" / "health"
    feature_store_root = tmp_path / "governance" / "feature_store"
    champion_root = tmp_path / "governance" / "champion_challenger"

    _write_json(
        feature_store_root / "latest.json",
        {
            "ok": True,
            "lineage_schema_version": 2,
            "dataset_contract": {"rows_sha256": "rows-hash"},
            "point_in_time_contract": {"dataset_join_keys": ["snapshot_id", "symbol"]},
        },
    )
    _write_json(health_root / "replay_hash_registry_guard_latest.json", {"ok": True})
    _write_json(health_root / "paper_replay_drill_latest.json", {"ok": False})
    _write_json(health_root / "replay_end_to_end_latest.json", {"ok": False})
    _write_json(health_root / "promotion_quality_gate_latest.json", {"ok": False})
    _write_json(health_root / "training_report_latest.json", {"summary": {"confirmed_training_success": False}})
    _write_json(health_root / "snapshot_coverage_latest.json", {"ok": True})
    _write_json(tmp_path / "governance" / "research" / "multiple_testing_guard_latest.json", {"ok": False, "family_size": 454, "correction_method": "benjamini_hochberg_fdr", "failed_checks": ["no_valid_rows"]})
    _write_json(tmp_path / "governance" / "research" / "decay_monitor_latest.json", {"overall_status": "needs_work"})
    _write_json(
        champion_root / "promotion_packet_latest.json",
        {
            "committee_packet_seed_ready": True,
            "packet_complete": False,
            "signature": {"verified": False},
            "replayability_contract": {
                "dataset_hash": "dataset-hash",
                "model_hash": "model-hash",
                "replay_hash": "replay-hash",
                "bundle_hash": "bundle-hash",
                "hash_bundle_complete": True,
                "exact_replay_ready": False,
            },
        },
    )

    payload = src.build_payload(tmp_path)

    assert payload["stronger_provisional_lineage_ready"] is True
    assert payload["lineage_score"] >= 82.5
    assert payload["promotion_packet_ready"] is False
    assert "exact_replay_ready" in payload["missing_contracts"]


def test_training_lineage_manifest_uses_signed_packet_replayability_fallback(tmp_path: Path) -> None:
    health_root = tmp_path / "governance" / "health"
    experiments_root = tmp_path / "governance" / "experiments"
    feature_store_root = tmp_path / "governance" / "feature_store"
    champion_root = tmp_path / "governance" / "champion_challenger"

    _write_json(
        feature_store_root / "latest.json",
        {
            "ok": True,
            "lineage_schema_version": 2,
            "dataset_contract": {"rows_sha256": "rows-hash"},
            "point_in_time_contract": {"dataset_join_keys": ["snapshot_id", "symbol"]},
        },
    )
    _write_json(health_root / "replay_hash_registry_guard_latest.json", {"ok": True})
    _write_json(health_root / "paper_replay_drill_latest.json", {"ok": True})
    _write_json(health_root / "replay_end_to_end_latest.json", {"ok": True})
    _write_json(health_root / "promotion_quality_gate_latest.json", {"ok": True})
    _write_json(health_root / "training_report_latest.json", {"summary": {"confirmed_training_success": False}})
    _write_json(health_root / "snapshot_coverage_latest.json", {"ok": True})
    _write_json(
        champion_root / "promotion_packet_latest.json",
        {
            "committee_packet_seed_ready": True,
            "packet_complete": True,
            "signature": {"verified": True},
            "replayability_contract": {
                "idle_scope": True,
                "dataset_hash": "dataset-hash",
                "model_hash": "model-hash",
                "replay_hash": "replay-hash",
                "bundle_hash": "bundle-hash",
                "hash_bundle_complete": True,
                "exact_replay_ready": True,
            },
        },
    )
    _write_json(
        champion_root / "promotion_autopilot_packet_latest.json",
        {
            "packet_complete": True,
            "signature_verified": True,
            "signed_bundle_contract": {"packet_sha256": "seeded-packet"},
            "evidence_bundle": {"source_count": 4},
        },
    )
    experiments_root.mkdir(parents=True, exist_ok=True)
    (experiments_root / "experiment_registry.jsonl").write_text(
        json.dumps({"experiment_id": "exp_packet_fallback", "replayability": {"bundle_hash": "", "exact_replay_ready": False}}) + "\n",
        encoding="utf-8",
    )

    payload = src.build_payload(tmp_path)

    assert payload["bundle_hashes"]["model_hash"] == "model-hash"
    assert payload["bundle_hashes"]["replay_hash"] == "replay-hash"
    assert payload["hash_bundle_complete"] is True
    assert payload["exact_replay_ready"] is True
    assert payload["strong_signed_packet_replay_ready"] is True
    assert payload["training_confirmed"] is True
