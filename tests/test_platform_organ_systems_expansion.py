from __future__ import annotations

import importlib.util
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = PROJECT_ROOT / "scripts" / "ops" / "platform_organ_systems_expansion.py"
spec = importlib.util.spec_from_file_location("platform_organ_systems_expansion", MODULE_PATH)
platform_organs = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(platform_organs)


def _write_registry(root: Path) -> None:
    (root / "master_bot_registry.json").write_text(
        json.dumps(
            {
                "updated_at_utc": "2026-05-04T00:00:00+00:00",
                "summary": {"total_bots": 1316, "active_bots": 1261, "max_bot_version": 1325},
                "sub_bots": [
                    {
                        "bot_id": "brain_refinery_v1325_strategy_gap_sector_pair_rotation_spread_arb_master_bridge_bot",
                        "bot_role": "infrastructure_sub_bot",
                        "active": True,
                        "data_collection_active": True,
                        "lifecycle_state": "data_collection_only",
                        "slot_label": "Existing Quant Strategy Gap Bot",
                        "slot_kind": "quant_strategy_gap_sector_pair_rotation_spread_arb_master_bridge",
                    }
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )


def test_dry_run_plans_fourteen_organ_system_pack(tmp_path: Path) -> None:
    _write_registry(tmp_path)

    payload = platform_organs.build_payload(tmp_path)

    assert payload["organ_count"] == 14
    assert payload["bot_count"] == 70
    assert payload["planned_bot_count"] == 70
    assert payload["target_platform_total_bots"] == 1386
    assert payload["pack"]["bot_ids"][0].startswith("brain_refinery_v1326_")
    assert payload["pack"]["bot_ids"][-1].startswith("brain_refinery_v1395_")
    assert payload["pack"]["storage_retention_rule"]["sample_rate"] == 0.02
    assert payload["pack"]["storage_retention_rule"]["max_daily_mb_per_bot"] == 3
    assert payload["pack"]["paper_only_floor"]["live_trading_enabled"] is False
    assert payload["pack"]["paper_only_floor"]["graduation_requires_minimum_observations"] == 60000
    assert payload["pack"]["paper_only_floor"]["graduation_requires_collection_days"] == 150
    assert "data_quality_v2" in payload["pack"]["organ_systems"]
    assert "audit_immune_system" in payload["pack"]["organ_systems"]


def test_apply_adds_collect_only_organ_bots(tmp_path: Path) -> None:
    _write_registry(tmp_path)

    payload = platform_organs.apply_registry(tmp_path)

    assert payload["added_bot_count"] == 70
    assert payload["new_total_bots"] == 71
    registry = json.loads((tmp_path / "master_bot_registry.json").read_text(encoding="utf-8"))
    added = [row for row in registry["sub_bots"] if row.get("platform_organ_systems_version") == platform_organs.PACK_VERSION]
    assert len(added) == 70
    assert registry["summary"]["platform_organ_systems_bot_count"] == 70
    assert registry["summary"]["data_collection_active_bots"] == 71
    assert registry["summary"]["max_bot_version"] == 1395
    organs = {row["organ_system"] for row in added}
    assert "operator_cockpit_v2" in organs
    assert "backpressure_circulatory_system" in organs
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
        assert row["minimum_training_observations"] == 60000
        assert row["minimum_data_collection_days"] == 150
        assert row["data_collection_capture_mode"] == "thin_sampled"
        assert row["data_collection_sample_rate"] == 0.02
        assert row["data_collection_compute_guard_mode"] == "thin_digest"
        assert row["capability_pack_contract"]["target_platform_total_bots"] == 1386
        assert row["platform_organ_systems_contract"]["contract_version"] == "platform_organ_systems_layers_v1"
        assert row["training_threshold_policy"]["requires_runtime_pressure_clearance"] is True
        assert row["label_contract"]["required_join_mode"] == "point_in_time_only"
    assert (tmp_path / "config" / "platform_organ_systems_v1.json").exists()
    assert (tmp_path / "governance" / "health" / "platform_organ_systems_latest.json").exists()
    assert (tmp_path / "governance" / "platform_organ_systems" / "data_quality_v2").is_dir()
    assert "governance/platform_organ_systems/data_quality_v2" in payload["storage_targets_ready"]


def test_apply_is_idempotent_by_slot_kind(tmp_path: Path) -> None:
    _write_registry(tmp_path)

    first = platform_organs.apply_registry(tmp_path)
    second = platform_organs.apply_registry(tmp_path)
    registry = json.loads((tmp_path / "master_bot_registry.json").read_text(encoding="utf-8"))
    added = [row for row in registry["sub_bots"] if row.get("platform_organ_systems_version") == platform_organs.PACK_VERSION]

    assert first["added_bot_count"] == 70
    assert second["added_bot_count"] == 0
    assert len(added) == 70
    assert second["pack"]["bot_ids"][0] == first["added_bot_ids"][0]
