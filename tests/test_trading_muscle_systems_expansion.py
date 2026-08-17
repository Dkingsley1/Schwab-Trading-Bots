from __future__ import annotations

import importlib.util
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = PROJECT_ROOT / "scripts" / "ops" / "trading_muscle_systems_expansion.py"
spec = importlib.util.spec_from_file_location("trading_muscle_systems_expansion", MODULE_PATH)
trading_muscles = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(trading_muscles)


def _write_registry(root: Path) -> None:
    (root / "master_bot_registry.json").write_text(
        json.dumps(
            {
                "updated_at_utc": "2026-05-04T00:00:00+00:00",
                "summary": {"total_bots": 1386, "active_bots": 1331, "max_bot_version": 1395},
                "sub_bots": [
                    {
                        "bot_id": "brain_refinery_v1395_platform_organ_audit_immune_system_master_bridge_bot",
                        "bot_role": "infrastructure_sub_bot",
                        "active": True,
                        "data_collection_active": True,
                        "lifecycle_state": "data_collection_only",
                        "slot_label": "Existing Platform Organ Bot",
                        "slot_kind": "platform_organ_systems_audit_immune_system_master_bridge",
                    }
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )


def test_dry_run_plans_fourteen_trading_muscle_pack(tmp_path: Path) -> None:
    _write_registry(tmp_path)

    payload = trading_muscles.build_payload(tmp_path)

    assert payload["muscle_count"] == 14
    assert payload["bot_count"] == 70
    assert payload["planned_bot_count"] == 70
    assert payload["target_platform_total_bots"] == 1456
    assert payload["pack"]["bot_ids"][0].startswith("brain_refinery_v1396_")
    assert payload["pack"]["bot_ids"][-1].startswith("brain_refinery_v1465_")
    assert payload["pack"]["storage_retention_rule"]["sample_rate"] == 0.025
    assert payload["pack"]["storage_retention_rule"]["max_daily_mb_per_bot"] == 5
    assert payload["pack"]["paper_only_floor"]["paper_trading_enabled"] is False
    assert payload["pack"]["paper_only_floor"]["live_trading_enabled"] is False
    assert payload["pack"]["paper_only_floor"]["graduation_requires_minimum_observations"] == 75000
    assert payload["pack"]["paper_only_floor"]["graduation_requires_collection_days"] == 180
    assert "intraday_momentum_muscle" in payload["pack"]["muscle_systems"]
    assert "exit_rebalance_muscle" in payload["pack"]["muscle_systems"]


def test_apply_adds_collect_only_trading_muscle_bots(tmp_path: Path) -> None:
    _write_registry(tmp_path)

    payload = trading_muscles.apply_registry(tmp_path)

    assert payload["added_bot_count"] == 70
    assert payload["new_total_bots"] == 71
    registry = json.loads((tmp_path / "master_bot_registry.json").read_text(encoding="utf-8"))
    added = [row for row in registry["sub_bots"] if row.get("trading_muscle_systems_version") == trading_muscles.PACK_VERSION]
    assert len(added) == 70
    assert registry["summary"]["trading_muscle_systems_bot_count"] == 70
    assert registry["summary"]["data_collection_active_bots"] == 71
    assert registry["summary"]["max_bot_version"] == 1465
    muscles = {row["trading_muscle"] for row in added}
    assert "options_convexity_muscle" in muscles
    assert "position_sizing_muscle" in muscles
    for row in added:
        assert row["active"] is True
        assert row["lifecycle_state"] == "data_collection_only"
        assert row["training_excluded"] is True
        assert row["exclude_from_training"] is True
        assert row["training_candidate_after_threshold"] is True
        assert row["trade_candidate_collection_active"] is True
        assert row["paper_trade_readiness_gated"] is True
        assert row["paper_trading_enabled"] is False
        assert row["live_trading_enabled"] is False
        assert row["execution_enabled"] is False
        assert row["allocation_enabled"] is False
        assert row["paper_trade_lock_required"] is True
        assert row["minimum_training_observations"] == 75000
        assert row["minimum_data_collection_days"] == 180
        assert row["data_collection_capture_mode"] == "thin_sampled"
        assert row["data_collection_sample_rate"] == 0.025
        assert row["data_collection_compute_guard_mode"] == "thin_digest"
        assert row["capability_pack_contract"]["target_platform_total_bots"] == 1456
        assert row["trading_muscle_systems_contract"]["contract_version"] == "trading_muscle_systems_layers_v1"
        assert row["training_threshold_policy"]["requires_execution_realism_clearance"] is True
        assert row["label_contract"]["required_join_mode"] == "point_in_time_only"
    assert (tmp_path / "config" / "trading_muscle_systems_v1.json").exists()
    assert (tmp_path / "governance" / "health" / "trading_muscle_systems_latest.json").exists()
    assert (tmp_path / "governance" / "trading_muscle_systems" / "intraday_momentum_muscle").is_dir()
    assert "governance/trading_muscle_systems/intraday_momentum_muscle" in payload["storage_targets_ready"]


def test_apply_is_idempotent_by_slot_kind(tmp_path: Path) -> None:
    _write_registry(tmp_path)

    first = trading_muscles.apply_registry(tmp_path)
    second = trading_muscles.apply_registry(tmp_path)
    registry = json.loads((tmp_path / "master_bot_registry.json").read_text(encoding="utf-8"))
    added = [row for row in registry["sub_bots"] if row.get("trading_muscle_systems_version") == trading_muscles.PACK_VERSION]

    assert first["added_bot_count"] == 70
    assert second["added_bot_count"] == 0
    assert len(added) == 70
    assert second["pack"]["bot_ids"][0] == first["added_bot_ids"][0]
