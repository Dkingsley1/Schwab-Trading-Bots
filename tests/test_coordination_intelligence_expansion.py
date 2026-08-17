from __future__ import annotations

import importlib.util
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = PROJECT_ROOT / "scripts" / "ops" / "coordination_intelligence_expansion.py"
spec = importlib.util.spec_from_file_location("coordination_intelligence_expansion", MODULE_PATH)
coordination = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(coordination)


def _write_registry(root: Path) -> None:
    (root / "master_bot_registry.json").write_text(
        json.dumps(
            {
                "updated_at_utc": "2026-05-02T00:00:00+00:00",
                "summary": {"total_bots": 1, "active_bots": 1},
                "sub_bots": [
                    {
                        "bot_id": "brain_refinery_v848_seed_bot",
                        "bot_role": "signal_sub_bot",
                        "active": True,
                        "data_collection_active": False,
                    }
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )


def test_dry_run_plans_coordination_intelligence_pack(tmp_path: Path) -> None:
    _write_registry(tmp_path)

    payload = coordination.build_payload(tmp_path)

    assert payload["layer_count"] == 10
    assert payload["bot_count"] == 30
    assert payload["planned_bot_count"] == 30
    assert payload["coordination"]["bot_ids"][0].startswith("brain_refinery_v849_")
    assert payload["coordination"]["bot_ids"][-1].startswith("brain_refinery_v878_")
    assert payload["coordination"]["storage_retention_rule"]["dedupe_required"] is True
    assert payload["coordination"]["storage_retention_rule"]["sample_rate"] == 0.18
    assert payload["coordination"]["paper_only_floor"]["live_trading_enabled"] is False
    assert payload["coordination"]["paper_only_floor"]["graduation_requires_minimum_observations"] == 15000
    assert payload["coordination"]["capacity_check"]["heavy_reasoning_mode"] == "digest_only_until_host_pressure_clear"
    assert {layer["slug"] for layer in payload["coordination"]["layers"]} == {
        "bot_genome_lineage_map_v2",
        "strategy_conflict_resolver",
        "capital_allocation_simulator",
        "market_regime_memory",
        "research_to_bot_pipeline",
        "feature_store_quality_layer",
        "adversarial_paper_trading_lab",
        "sleeve_master_upgrade_pack",
        "bot_admission_committee",
        "system_explainability_dashboard",
    }


def test_apply_adds_collection_only_coordination_bots(tmp_path: Path) -> None:
    _write_registry(tmp_path)

    payload = coordination.apply_registry(tmp_path)

    assert payload["added_bot_count"] == 30
    assert payload["new_total_bots"] == 31
    registry = json.loads((tmp_path / "master_bot_registry.json").read_text(encoding="utf-8"))
    added = [
        row
        for row in registry["sub_bots"]
        if row.get("coordination_intelligence_version") == coordination.COORDINATION_VERSION
    ]
    assert len(added) == 30
    assert registry["summary"]["coordination_intelligence_bot_count"] == 30
    assert registry["summary"]["structured_capability_pack_bot_count"] == 30
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
        assert row["minimum_training_observations"] == 15000
        assert row["minimum_data_collection_days"] == 60
        assert row["capability_pack_contract"]["bot_pack_size_rule"] == "10_layers_3_bots_each_30_total_max"
        assert row["advanced_intelligence_layer_contract"]["contract_version"] == "coordination_intelligence_layers_v1"
    assert (tmp_path / "config" / "coordination_intelligence_pack_v1.json").exists()
    assert (tmp_path / "governance" / "health" / "coordination_intelligence_latest.json").exists()


def test_apply_is_idempotent_by_slot_kind(tmp_path: Path) -> None:
    _write_registry(tmp_path)

    first = coordination.apply_registry(tmp_path)
    second = coordination.apply_registry(tmp_path)
    registry = json.loads((tmp_path / "master_bot_registry.json").read_text(encoding="utf-8"))
    added = [
        row
        for row in registry["sub_bots"]
        if row.get("coordination_intelligence_version") == coordination.COORDINATION_VERSION
    ]

    assert first["added_bot_count"] == 30
    assert second["added_bot_count"] == 0
    assert len(added) == 30
    assert second["coordination"]["bot_ids"][0] == first["added_bot_ids"][0]
