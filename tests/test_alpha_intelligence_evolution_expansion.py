from __future__ import annotations

import importlib.util
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = PROJECT_ROOT / "scripts" / "ops" / "alpha_intelligence_evolution_expansion.py"
spec = importlib.util.spec_from_file_location("alpha_intelligence_evolution_expansion", MODULE_PATH)
alpha_evolution = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(alpha_evolution)


def _write_registry(root: Path) -> None:
    (root / "master_bot_registry.json").write_text(
        json.dumps(
            {
                "updated_at_utc": "2026-05-03T00:00:00+00:00",
                "summary": {"total_bots": 1, "active_bots": 1},
                "sub_bots": [
                    {
                        "bot_id": "brain_refinery_v903_seed_self_awareness_bridge",
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


def test_dry_run_plans_alpha_intelligence_evolution_pack(tmp_path: Path) -> None:
    _write_registry(tmp_path)

    payload = alpha_evolution.build_payload(tmp_path)

    assert payload["system_count"] == 10
    assert payload["bot_count"] == 30
    assert payload["planned_bot_count"] == 30
    assert payload["pack"]["bot_ids"][0].startswith("brain_refinery_v904_")
    assert payload["pack"]["bot_ids"][-1].startswith("brain_refinery_v933_")
    assert payload["pack"]["storage_retention_rule"]["sample_rate"] == 0.16
    assert payload["pack"]["paper_only_floor"]["live_trading_enabled"] is False
    assert payload["pack"]["paper_only_floor"]["graduation_requires_minimum_observations"] == 20000
    assert payload["pack"]["alpha_admission_guard_bot_id"].startswith("brain_refinery_v")
    assert "training_readiness_gating" in payload["pack"]["intelligence_upgrades"]
    assert "self_model_feeds_alpha_readiness" in payload["pack"]["self_awareness_upgrades"]
    assert {system["slug"] for system in payload["pack"]["systems"]} == {
        "training_readiness_brain",
        "execution_reality_lab",
        "portfolio_exposure_brain",
        "data_source_confidence_engine",
        "research_intake_pipeline",
        "duplicate_alpha_novelty_engine",
        "regime_playbook_memory_v2",
        "professional_dashboard_v2",
        "broker_data_adapter_mesh",
        "autonomous_cleanup_governor",
    }


def test_apply_adds_collect_only_alpha_intelligence_bots(tmp_path: Path) -> None:
    _write_registry(tmp_path)

    payload = alpha_evolution.apply_registry(tmp_path)

    assert payload["added_bot_count"] == 30
    assert payload["new_total_bots"] == 31
    registry = json.loads((tmp_path / "master_bot_registry.json").read_text(encoding="utf-8"))
    added = [
        row
        for row in registry["sub_bots"]
        if row.get("alpha_intelligence_evolution_version") == alpha_evolution.PACK_VERSION
    ]
    assert len(added) == 30
    assert registry["summary"]["alpha_intelligence_evolution_bot_count"] == 30
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
        assert row["minimum_training_observations"] == 20000
        assert row["minimum_data_collection_days"] == 75
        assert row["capability_pack_contract"]["bot_pack_size_rule"] == "10_systems_3_bots_each_30_total_max"
        assert row["advanced_intelligence_layer_contract"]["contract_version"] == "alpha_intelligence_evolution_layers_v1"
        assert row["training_threshold_policy"]["requires_duplicate_alpha_clearance"] is True
    assert (tmp_path / "config" / "alpha_intelligence_evolution_v1.json").exists()
    assert (tmp_path / "governance" / "health" / "alpha_intelligence_evolution_latest.json").exists()
    assert (tmp_path / "governance" / "alpha_intelligence" / "training_readiness").is_dir()
    assert "governance/alpha_intelligence/training_readiness" in payload["storage_targets_ready"]


def test_apply_is_idempotent_by_slot_kind(tmp_path: Path) -> None:
    _write_registry(tmp_path)

    first = alpha_evolution.apply_registry(tmp_path)
    second = alpha_evolution.apply_registry(tmp_path)
    registry = json.loads((tmp_path / "master_bot_registry.json").read_text(encoding="utf-8"))
    added = [
        row
        for row in registry["sub_bots"]
        if row.get("alpha_intelligence_evolution_version") == alpha_evolution.PACK_VERSION
    ]

    assert first["added_bot_count"] == 30
    assert second["added_bot_count"] == 0
    assert len(added) == 30
    assert second["pack"]["bot_ids"][0] == first["added_bot_ids"][0]
