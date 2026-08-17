from __future__ import annotations

import importlib.util
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = PROJECT_ROOT / "scripts" / "ops" / "apex_self_awareness_intelligence_expansion.py"
spec = importlib.util.spec_from_file_location("apex_self_awareness_intelligence_expansion", MODULE_PATH)
apex = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(apex)


def _write_registry(root: Path) -> None:
    (root / "master_bot_registry.json").write_text(
        json.dumps(
            {
                "updated_at_utc": "2026-05-04T00:00:00+00:00",
                "summary": {"total_bots": 1, "active_bots": 1},
                "sub_bots": [
                    {
                        "bot_id": "brain_refinery_v963_seed_intelligence_backlog_master",
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


def test_dry_run_plans_apex_pack_to_reach_thousand_bot_platform(tmp_path: Path) -> None:
    _write_registry(tmp_path)

    payload = apex.build_payload(tmp_path)

    assert payload["system_count"] == 10
    assert payload["bot_count"] == 46
    assert payload["planned_bot_count"] == 46
    assert payload["target_platform_total_bots"] == 1000
    assert payload["pack"]["bot_ids"][0].startswith("brain_refinery_v964_")
    assert payload["pack"]["bot_ids"][-1].startswith("brain_refinery_v1009_")
    assert payload["pack"]["storage_retention_rule"]["sample_rate"] == 0.12
    assert payload["pack"]["paper_only_floor"]["live_trading_enabled"] is False
    assert payload["pack"]["paper_only_floor"]["graduation_requires_minimum_observations"] == 30000
    assert payload["pack"]["paper_only_floor"]["graduation_requires_collection_days"] == 120
    assert payload["pack"]["runtime_capacity_floor"] == 1000
    assert "deep_self_state_vectors" in payload["pack"]["self_awareness_advancements"]
    assert "autonomous_upgrade_foundry" in payload["pack"]["intelligence_advancements"]
    assert {system["slug"] for system in payload["pack"]["systems"]} == {
        "self_model_deep_introspection",
        "meta_reasoning_policy_engine",
        "experience_memory_os",
        "world_model_scenario_oracle",
        "autonomous_upgrade_foundry",
        "alpha_safety_causal_judge",
        "resource_autonomy_governor",
        "operator_copilot_narrative",
        "grandmaster_collective_intelligence",
        "adaptive_research_frontier",
    }


def test_apply_adds_collect_only_apex_bots(tmp_path: Path) -> None:
    _write_registry(tmp_path)

    payload = apex.apply_registry(tmp_path)

    assert payload["added_bot_count"] == 46
    assert payload["new_total_bots"] == 47
    registry = json.loads((tmp_path / "master_bot_registry.json").read_text(encoding="utf-8"))
    added = [
        row
        for row in registry["sub_bots"]
        if row.get("apex_self_awareness_intelligence_version") == apex.PACK_VERSION
    ]
    assert len(added) == 46
    assert registry["summary"]["apex_self_awareness_intelligence_bot_count"] == 46
    assert registry["summary"]["data_collection_active_bots"] == 46
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
        assert row["minimum_training_observations"] == 30000
        assert row["minimum_data_collection_days"] == 120
        assert row["data_collection_capture_mode"] == "sampled"
        assert row["data_collection_compute_guard_mode"] == "sustain"
        assert row["capability_pack_contract"]["target_platform_total_bots"] == 1000
        assert row["apex_self_awareness_intelligence_contract"]["contract_version"] == "apex_self_awareness_intelligence_layers_v1"
        assert row["training_threshold_policy"]["requires_assumption_inventory_clearance"] is True
        assert row["training_threshold_policy"]["requires_runtime_pressure_clear"] is True
    assert (tmp_path / "config" / "apex_self_awareness_intelligence_v1.json").exists()
    assert (tmp_path / "governance" / "health" / "apex_self_awareness_intelligence_latest.json").exists()
    assert (tmp_path / "governance" / "apex_intelligence" / "self_model").is_dir()
    assert "governance/apex_intelligence/self_model" in payload["storage_targets_ready"]


def test_apply_is_idempotent_by_slot_kind(tmp_path: Path) -> None:
    _write_registry(tmp_path)

    first = apex.apply_registry(tmp_path)
    second = apex.apply_registry(tmp_path)
    registry = json.loads((tmp_path / "master_bot_registry.json").read_text(encoding="utf-8"))
    added = [
        row
        for row in registry["sub_bots"]
        if row.get("apex_self_awareness_intelligence_version") == apex.PACK_VERSION
    ]

    assert first["added_bot_count"] == 46
    assert second["added_bot_count"] == 0
    assert len(added) == 46
    assert second["pack"]["bot_ids"][0] == first["added_bot_ids"][0]
