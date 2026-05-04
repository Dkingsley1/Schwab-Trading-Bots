from __future__ import annotations

import importlib.util
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = PROJECT_ROOT / "scripts" / "ops" / "intelligence_layer_advancement_expansion.py"
spec = importlib.util.spec_from_file_location("intelligence_layer_advancement_expansion", MODULE_PATH)
intelligence_layer = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(intelligence_layer)


def _write_registry(root: Path) -> None:
    (root / "master_bot_registry.json").write_text(
        json.dumps(
            {
                "updated_at_utc": "2026-05-03T00:00:00+00:00",
                "summary": {"total_bots": 1, "active_bots": 1},
                "sub_bots": [
                    {
                        "bot_id": "brain_refinery_v933_seed_alpha_cleanup_guard",
                        "bot_role": "infrastructure_sub_bot",
                        "active": True,
                        "data_collection_active": False,
                    }
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )


def test_dry_run_plans_intelligence_layer_advancement_pack(tmp_path: Path) -> None:
    _write_registry(tmp_path)

    payload = intelligence_layer.build_payload(tmp_path)

    assert payload["system_count"] == 10
    assert payload["bot_count"] == 30
    assert payload["planned_bot_count"] == 30
    assert payload["pack"]["bot_ids"][0].startswith("brain_refinery_v934_")
    assert payload["pack"]["bot_ids"][-1].startswith("brain_refinery_v963_")
    assert payload["pack"]["storage_retention_rule"]["sample_rate"] == 0.14
    assert payload["pack"]["paper_only_floor"]["live_trading_enabled"] is False
    assert payload["pack"]["paper_only_floor"]["graduation_requires_minimum_observations"] == 24000
    assert payload["pack"]["runtime_capacity_floor"] == 925
    assert "metacognitive_routing_v2" in payload["pack"]["intelligence_advancements"]
    assert "safety_invariant_verification_v2" in payload["pack"]["intelligence_advancements"]
    assert {system["slug"] for system in payload["pack"]["systems"]} == {
        "metacognitive_routing_v2",
        "world_model_counterfactual_lab",
        "alpha_evaluation_benchmark_suite",
        "memory_compression_retrieval_v2",
        "multi_agent_debate_critic_board",
        "active_learning_experiment_design_v2",
        "ensemble_governance_uncertainty",
        "library_tool_intelligence_router",
        "safety_invariant_verification_v2",
        "self_improvement_backlog_planner",
    }


def test_apply_adds_collect_only_intelligence_layer_bots(tmp_path: Path) -> None:
    _write_registry(tmp_path)

    payload = intelligence_layer.apply_registry(tmp_path)

    assert payload["added_bot_count"] == 30
    assert payload["new_total_bots"] == 31
    registry = json.loads((tmp_path / "master_bot_registry.json").read_text(encoding="utf-8"))
    added = [
        row
        for row in registry["sub_bots"]
        if row.get("intelligence_layer_advancement_version") == intelligence_layer.PACK_VERSION
    ]
    assert len(added) == 30
    assert registry["summary"]["intelligence_layer_advancement_bot_count"] == 30
    assert registry["summary"]["data_collection_active_bots"] == 30
    for row in added:
        assert row["active"] is True
        assert row["lifecycle_state"] == "data_collection_only"
        assert row["training_excluded"] is True
        assert row["exclude_from_training"] is True
        assert row["training_candidate_after_threshold"] is True
        assert row["paper_trading_enabled"] is False
        assert row["live_trading_enabled"] is False
        assert row["execution_enabled"] is False
        assert row["allocation_enabled"] is False
        assert row["paper_trade_lock_required"] is True
        assert row["minimum_training_observations"] == 24000
        assert row["minimum_data_collection_days"] == 90
        assert row["capability_pack_contract"]["bot_pack_size_rule"] == "10_systems_3_bots_each_30_total_max"
        assert row["advanced_intelligence_layer_contract"]["contract_version"] == "intelligence_layer_advancement_layers_v1"
        assert row["training_threshold_policy"]["requires_safety_invariant_clearance"] is True
        assert row["training_threshold_policy"]["requires_memory_quality_clearance"] is True
    assert (tmp_path / "config" / "intelligence_layer_advancement_v1.json").exists()
    assert (tmp_path / "governance" / "health" / "intelligence_layer_advancement_latest.json").exists()
    assert (tmp_path / "governance" / "intelligence_layer" / "metacognition").is_dir()
    assert "governance/intelligence_layer/metacognition" in payload["storage_targets_ready"]


def test_apply_is_idempotent_by_slot_kind(tmp_path: Path) -> None:
    _write_registry(tmp_path)

    first = intelligence_layer.apply_registry(tmp_path)
    second = intelligence_layer.apply_registry(tmp_path)
    registry = json.loads((tmp_path / "master_bot_registry.json").read_text(encoding="utf-8"))
    added = [
        row
        for row in registry["sub_bots"]
        if row.get("intelligence_layer_advancement_version") == intelligence_layer.PACK_VERSION
    ]

    assert first["added_bot_count"] == 30
    assert second["added_bot_count"] == 0
    assert len(added) == 30
    assert second["pack"]["bot_ids"][0] == first["added_bot_ids"][0]
