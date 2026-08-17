from __future__ import annotations

import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.ops import roster_expansion_slots as slots_src


STABILITY_CAPACITY_BOT_IDS = {
    "brain_refinery_v714_runtime_lane_load_smoother_bot",
    "brain_refinery_v715_collector_fanout_budget_allocator_bot",
    "brain_refinery_v716_overnight_heavy_view_cotenant_guard_bot",
    "brain_refinery_v717_mlx_batch_size_runtime_governor_bot",
    "brain_refinery_v718_runtime_capacity_regression_guard_bot",
    "brain_refinery_v719_shard_writer_queue_balancer_bot",
    "brain_refinery_v720_explanation_backlog_chunker_bot",
    "brain_refinery_v721_ops_data_plane_integrity_sentinel_bot",
    "brain_refinery_v722_wal_queue_checkpoint_scheduler_bot",
    "brain_refinery_v723_data_plane_backpressure_regression_guard_bot",
    "brain_refinery_v724_global_halt_clearance_verifier_bot",
    "brain_refinery_v725_sleeve_thaw_sequence_coordinator_bot",
    "brain_refinery_v726_tripwire_root_cause_cluster_bot",
    "brain_refinery_v727_paper_trade_lock_recovery_guard_bot",
    "brain_refinery_v728_halt_recovery_regression_guard_bot",
    "brain_refinery_v729_bot_admission_capacity_score_bot",
    "brain_refinery_v730_label_taxonomy_drift_guard_bot",
    "brain_refinery_v731_training_readiness_sample_debt_bot",
    "brain_refinery_v732_report_surface_freshness_contract_bot",
    "brain_refinery_v733_expansion_quality_regression_guard_bot",
}


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def test_stability_capacity_slots_are_planned() -> None:
    specs = {
        str(row.get("bot_id") or ""): row
        for row in slots_src.DEFAULT_SLOT_SPECS
        if str(row.get("bot_id") or "") in STABILITY_CAPACITY_BOT_IDS
    }

    assert set(specs) == STABILITY_CAPACITY_BOT_IDS
    assert {row["sleeve_profile"] for row in specs.values()} == {
        "runtime_capacity_governance",
        "data_plane_backpressure_resilience",
        "halt_recovery_stability",
        "expansion_quality_governance",
    }
    assert all(row["sleeve_family"] == "quant_models" for row in specs.values())
    assert all(row["bot_role"] == "infrastructure_sub_bot" for row in specs.values())
    assert "runtime_capacity_pressure_snapshot" in specs["brain_refinery_v714_runtime_lane_load_smoother_bot"]["data_intake_collections"]
    assert "sql_writer_queue_balance_state" in specs["brain_refinery_v719_shard_writer_queue_balancer_bot"]["data_intake_collections"]
    assert "global_halt_clearance_verification" in specs["brain_refinery_v724_global_halt_clearance_verifier_bot"]["data_intake_collections"]
    assert "bot_admission_capacity_score" in specs["brain_refinery_v729_bot_admission_capacity_score_bot"]["data_intake_collections"]


def test_stability_capacity_apply_keeps_collection_only_contract(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    registry_path = project_root / "master_bot_registry.json"
    _write_json(registry_path, {"summary": {}, "sub_bots": []})

    apply_result = slots_src.apply_registry(project_root, registry_path=registry_path)
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    rows = {
        str(row.get("bot_id") or ""): row
        for row in registry.get("sub_bots", [])
        if str(row.get("bot_id") or "") in STABILITY_CAPACITY_BOT_IDS
    }

    assert set(rows) == STABILITY_CAPACITY_BOT_IDS
    assert apply_result["added_slots"] >= len(STABILITY_CAPACITY_BOT_IDS)
    for row in rows.values():
        assert row["active"] is True
        assert row["bot_role"] == "infrastructure_sub_bot"
        assert row["lifecycle_state"] == "data_collection_only"
        assert row["data_collection_active"] is True
        assert row["training_excluded"] is True
        assert row["exclude_from_training"] is True
        assert row["training_candidate_after_threshold"] is True
        assert row["allocation_enabled"] is False
        assert row["paper_trading_enabled"] is False
        assert row["live_trading_enabled"] is False
        assert row["minimum_training_observations"] >= 3000
        assert "research_only" in row["labeling_tags"]
        assert row["direct_execution_allowed"] is False


def test_stability_capacity_provider_and_storage_contracts(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    registry_path = project_root / "master_bot_registry.json"
    _write_json(registry_path, {"summary": {}, "sub_bots": []})

    slots_src.apply_registry(project_root, registry_path=registry_path)
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    rows = {
        str(row.get("sleeve_profile") or ""): row
        for row in registry.get("sub_bots", [])
        if str(row.get("bot_id") or "") in {
            "brain_refinery_v714_runtime_lane_load_smoother_bot",
            "brain_refinery_v719_shard_writer_queue_balancer_bot",
            "brain_refinery_v724_global_halt_clearance_verifier_bot",
            "brain_refinery_v729_bot_admission_capacity_score_bot",
        }
    }

    assert rows["runtime_capacity_governance"]["provider_capability_profile"] == "research_only_runtime_capacity_guard"
    assert rows["data_plane_backpressure_resilience"]["provider_capability_profile"] == "research_only_data_plane_backpressure_guard"
    assert rows["halt_recovery_stability"]["provider_capability_profile"] == "research_only_halt_recovery_guard"
    assert rows["expansion_quality_governance"]["provider_capability_profile"] == "research_only_expansion_quality_guard"
    assert "governance/resource" in rows["runtime_capacity_governance"]["storage_targets"]
    assert "governance/storage" in rows["data_plane_backpressure_resilience"]["storage_targets"]
    assert "governance/halts" in rows["halt_recovery_stability"]["storage_targets"]
    assert "governance/reports" in rows["expansion_quality_governance"]["storage_targets"]


def test_sleeve_strategy_manifest_includes_stability_capacity_wave() -> None:
    manifest = json.loads((PROJECT_ROOT / "config" / "sleeve_strategy_expansion.json").read_text(encoding="utf-8"))
    sleeves = {str(row.get("name") or ""): row for row in manifest["sleeves"]}

    assert set(sleeves) >= {
        "runtime_capacity_governance",
        "data_plane_backpressure_resilience",
        "halt_recovery_stability",
        "expansion_quality_governance",
    }
    assert len(sleeves["runtime_capacity_governance"]["strategies"]) == 5
    assert len(sleeves["data_plane_backpressure_resilience"]["strategies"]) == 5
    assert len(sleeves["halt_recovery_stability"]["strategies"]) == 5
    assert len(sleeves["expansion_quality_governance"]["strategies"]) == 5
