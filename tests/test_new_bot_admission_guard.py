import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import scripts.new_bot_admission_guard as src


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def test_new_bot_admission_guard_passes_probation_candidate_with_complete_contracts(tmp_path: Path) -> None:
    diagnostics_root = tmp_path / "governance" / "training_diagnostics"
    _write_json(
        diagnostics_root / "brain_refinery_v43_intraday_ultrafast_proxy_latest.json",
        {"status": "ok", "sample_count": 55, "eligible_sequences": 8, "sequence_count": 8},
    )

    payload = src.build_payload(
        registry={
            "sub_bots": [
                {
                    "bot_id": "brain_refinery_v43_intraday_ultrafast_proxy",
                    "active": True,
                    "lifecycle_state": "probation",
                }
            ]
        },
        walk_forward={
            "bots": {
                "brain_refinery_v43_intraday_ultrafast_proxy": {
                    "runs": 12,
                    "forward_mean": 0.61,
                    "delta": 0.02,
                    "status": "pass",
                }
            }
        },
        feature_store_manifest={
            "ok": True,
            "strict_ok": True,
            "point_in_time_contract": {"complete": True},
            "contract_hashes": {"dataset_manifest_sha256": "a" * 64},
        },
        replay_hash_registry_guard={
            "ok": True,
            "details": {
                "paper": {"current_hash": "paper-hash"},
                "e2e": {"current_hash": "e2e-hash"},
            },
        },
        ownership_payload={
            "owners_by_bot_id": {"brain_refinery_v43_intraday_ultrafast_proxy": "desk-intraday"},
        },
        diagnostics_root=diagnostics_root,
        min_training_sample_count=40,
        min_eligible_sequences=4,
        min_walk_forward_runs=12,
    )

    assert payload["ok"] is True
    assert payload["candidate_bot_count"] == 1
    assert payload["blocking_candidate_count"] == 0
    assert payload["candidates"][0]["support_owner"] == "desk-intraday"


def test_new_bot_admission_guard_blocks_missing_owner_and_incomplete_global_prereqs(tmp_path: Path) -> None:
    diagnostics_root = tmp_path / "governance" / "training_diagnostics"
    _write_json(
        diagnostics_root / "brain_refinery_v99_new_lane_latest.json",
        {"status": "deferred_sample_starved", "sample_count": 5, "eligible_sequences": 1, "sequence_count": 1},
    )

    payload = src.build_payload(
        registry={"sub_bots": [{"bot_id": "brain_refinery_v99_new_lane", "active": True}]},
        walk_forward={"bots": {"brain_refinery_v99_new_lane": {"runs": 3, "status": "fail"}}},
        feature_store_manifest={"ok": False, "point_in_time_contract": {"complete": False}, "contract_hashes": {}},
        replay_hash_registry_guard={"ok": False, "details": {}},
        ownership_payload={},
        diagnostics_root=diagnostics_root,
        min_training_sample_count=40,
        min_eligible_sequences=4,
        min_walk_forward_runs=12,
    )

    assert payload["ok"] is False
    assert payload["candidate_bot_count"] == 1
    assert payload["global_prerequisites"]["global_failed_checks"] == [
        "feature_store_manifest_not_strict_ready",
        "replay_hash_registry_not_ready",
    ]
    assert "support_owner_missing" in payload["blocking_candidates"][0]["failed_contracts"]
    assert "training_diagnostics_deferred_sample_starved" in payload["blocking_candidates"][0]["failed_contracts"]


def test_new_bot_admission_guard_ignores_inactive_probation_backlog_without_active_rollout_scope(tmp_path: Path) -> None:
    diagnostics_root = tmp_path / "governance" / "training_diagnostics"
    _write_json(
        diagnostics_root / "brain_refinery_v99_backlog_latest.json",
        {"status": "deferred_sample_starved", "sample_count": 5, "eligible_sequences": 1, "sequence_count": 1},
    )

    payload = src.build_payload(
        registry={
            "sub_bots": [
                {
                    "bot_id": "brain_refinery_v99_backlog",
                    "active": False,
                    "lifecycle_state": "probation",
                }
            ]
        },
        walk_forward={"bots": {"brain_refinery_v99_backlog": {"runs": 2, "status": "insufficient_runs"}}},
        feature_store_manifest={
            "ok": True,
            "strict_ok": True,
            "point_in_time_contract": {"complete": True},
            "contract_hashes": {"dataset_manifest_sha256": "a" * 64},
        },
        replay_hash_registry_guard={
            "ok": True,
            "details": {
                "paper": {"current_hash": "paper-hash"},
                "e2e": {"current_hash": "e2e-hash"},
            },
        },
        ownership_payload={"default_owner": "ml-governance"},
        diagnostics_root=diagnostics_root,
        min_training_sample_count=40,
        min_eligible_sequences=4,
        min_walk_forward_runs=12,
    )

    assert payload["ok"] is True
    assert payload["candidate_bot_count"] == 0
    assert payload["blocking_candidate_count"] == 0


def test_new_bot_admission_guard_ignores_plain_data_collection_only_rows(tmp_path: Path) -> None:
    diagnostics_root = tmp_path / "governance" / "training_diagnostics"
    _write_json(
        diagnostics_root / "brain_refinery_v120_collector_latest.json",
        {"status": "deferred_sample_starved", "sample_count": 0, "eligible_sequences": 0, "sequence_count": 0},
    )

    payload = src.build_payload(
        registry={
            "sub_bots": [
                {
                    "bot_id": "brain_refinery_v120_collector",
                    "active": True,
                    "lifecycle_state": "data_collection_only",
                }
            ]
        },
        walk_forward={"bots": {"brain_refinery_v120_collector": {"runs": 0, "status": "insufficient_runs"}}},
        feature_store_manifest={"ok": False, "point_in_time_contract": {"complete": False}, "contract_hashes": {}},
        replay_hash_registry_guard={"ok": False, "details": {}},
        ownership_payload={},
        diagnostics_root=diagnostics_root,
        min_training_sample_count=40,
        min_eligible_sequences=4,
        min_walk_forward_runs=12,
    )

    assert payload["ok"] is True
    assert payload["candidate_bot_count"] == 0
    assert payload["blocking_candidate_count"] == 0
    assert payload["global_prerequisites"]["global_failed_checks"] == []


def test_new_bot_admission_guard_ignores_paper_live_data_without_explicit_promotion(tmp_path: Path) -> None:
    diagnostics_root = tmp_path / "governance" / "training_diagnostics"
    _write_json(
        diagnostics_root / "brain_refinery_v35_paper_observer_latest.json",
        {"status": "deferred_sample_starved", "sample_count": 0, "eligible_sequences": 0, "sequence_count": 0},
    )

    payload = src.build_payload(
        registry={
            "sub_bots": [
                {
                    "bot_id": "brain_refinery_v35_paper_observer",
                    "active": True,
                    "lifecycle_state": "paper_live_data",
                }
            ]
        },
        walk_forward={"bots": {"brain_refinery_v35_paper_observer": {"runs": 0, "status": "insufficient_runs"}}},
        feature_store_manifest={"ok": False, "point_in_time_contract": {"complete": False}, "contract_hashes": {}},
        replay_hash_registry_guard={"ok": False, "details": {}},
        ownership_payload={},
        diagnostics_root=diagnostics_root,
        min_training_sample_count=40,
        min_eligible_sequences=4,
        min_walk_forward_runs=12,
    )

    assert payload["ok"] is True
    assert payload["candidate_bot_count"] == 0
    assert payload["blocking_candidate_count"] == 0


def test_new_bot_admission_guard_ignores_guarded_paper_soak_rows_without_explicit_promotion(tmp_path: Path) -> None:
    diagnostics_root = tmp_path / "governance" / "training_diagnostics"
    _write_json(
        diagnostics_root / "brain_refinery_v36_guarded_paper_latest.json",
        {"status": "deferred_sample_starved", "sample_count": 0, "eligible_sequences": 0, "sequence_count": 0},
    )

    payload = src.build_payload(
        registry={
            "sub_bots": [
                {
                    "bot_id": "brain_refinery_v36_guarded_paper",
                    "active": True,
                    "lifecycle_state": "active",
                    "paper_live_data_enabled": True,
                    "paper_trading_enabled": True,
                    "paper_trade_lock_policy": "market_data_and_paper_only_until_explicit_graduation",
                    "direct_execution_allowed": False,
                    "trading_enabled": False,
                    "live_trading_enabled": False,
                    "execution_enabled": False,
                    "allocation_enabled": False,
                }
            ]
        },
        walk_forward={"bots": {"brain_refinery_v36_guarded_paper": {"runs": 0, "status": "insufficient_runs"}}},
        feature_store_manifest={"ok": False, "point_in_time_contract": {"complete": False}, "contract_hashes": {}},
        replay_hash_registry_guard={"ok": False, "details": {}},
        ownership_payload={},
        diagnostics_root=diagnostics_root,
        min_training_sample_count=40,
        min_eligible_sequences=4,
        min_walk_forward_runs=12,
    )

    assert payload["ok"] is True
    assert payload["candidate_bot_count"] == 0
    assert payload["blocking_candidate_count"] == 0


def test_new_bot_admission_guard_targeted_advisory_scope_does_not_block_coverage_repair(tmp_path: Path) -> None:
    diagnostics_root = tmp_path / "governance" / "training_diagnostics"
    _write_json(
        diagnostics_root / "brain_refinery_v4_simple_latest.json",
        {"status": "deferred_sample_starved", "sample_count": 5, "eligible_sequences": 1, "sequence_count": 1},
    )

    payload = src.build_payload(
        registry={
            "sub_bots": [
                {"bot_id": "brain_refinery_v4_simple", "active": True},
                {"bot_id": "brain_refinery_v900_unrelated_new_bot", "active": True},
            ]
        },
        walk_forward={
            "bots": {
                "brain_refinery_v4_simple": {"runs": 0, "status": "insufficient_runs"},
                "brain_refinery_v900_unrelated_new_bot": {"runs": 0, "status": "insufficient_runs"},
            }
        },
        feature_store_manifest={"ok": False, "point_in_time_contract": {"complete": False}, "contract_hashes": {}},
        replay_hash_registry_guard={"ok": False, "details": {}},
        ownership_payload={"default_owner": "coverage-repair"},
        diagnostics_root=diagnostics_root,
        min_training_sample_count=40,
        min_eligible_sequences=4,
        min_walk_forward_runs=12,
        include_bot_ids={"brain_refinery_v4_simple"},
        advisory_only=True,
    )

    assert payload["ok"] is True
    assert payload["contract_ok"] is False
    assert payload["advisory_only"] is True
    assert payload["scope"]["target_scoped"] is True
    assert payload["candidate_bot_count"] == 1
    assert payload["blocking_candidate_count"] == 1
    assert payload["blocking_candidates"][0]["bot_id"] == "brain_refinery_v4_simple"
