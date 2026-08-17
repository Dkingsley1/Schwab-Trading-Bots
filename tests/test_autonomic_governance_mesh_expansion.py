from __future__ import annotations

import importlib.util
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = PROJECT_ROOT / "scripts" / "ops" / "autonomic_governance_mesh_expansion.py"
spec = importlib.util.spec_from_file_location("autonomic_governance_mesh_expansion", MODULE_PATH)
mesh = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(mesh)


def _write_registry(root: Path) -> None:
    (root / "master_bot_registry.json").write_text(
        json.dumps(
            {
                "updated_at_utc": "2026-05-06T00:00:00+00:00",
                "summary": {"total_bots": 1557, "active_bots": 1502, "max_bot_version": 1557},
                "sub_bots": [
                    {
                        "bot_id": "brain_refinery_v1557_quant_operational_operator_decision_packet_builder_master_bridge_bot",
                        "bot_role": "infrastructure_sub_bot",
                        "active": True,
                        "data_collection_active": True,
                        "lifecycle_state": "data_collection_only",
                        "slot_label": "Existing Quant Operational Bot",
                        "slot_kind": "quant_operational_intelligence_operator_decision_packet_builder_master_bridge",
                    }
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )


def test_dry_run_plans_autonomic_governance_mesh(tmp_path: Path) -> None:
    _write_registry(tmp_path)

    payload = mesh.build_payload(tmp_path)

    assert payload["system_count"] == 14
    assert payload["bot_count"] == 56
    assert payload["planned_bot_count"] == 56
    assert payload["target_platform_total_bots"] == 1604
    assert payload["pack"]["bot_ids"][0].startswith("brain_refinery_v1558_")
    assert payload["pack"]["bot_ids"][-1].startswith("brain_refinery_v1613_")
    assert payload["pack"]["storage_retention_rule"]["sample_rate"] == 0.012
    assert payload["pack"]["storage_retention_rule"]["max_daily_mb_per_bot"] == 2
    assert payload["pack"]["paper_only_floor"]["live_trading_enabled"] is False


def test_apply_adds_collect_only_governance_bots(tmp_path: Path) -> None:
    _write_registry(tmp_path)

    payload = mesh.apply_registry(tmp_path)

    assert payload["added_bot_count"] == 56
    registry = json.loads((tmp_path / "master_bot_registry.json").read_text(encoding="utf-8"))
    added = [row for row in registry["sub_bots"] if row.get("autonomic_governance_mesh_version") == mesh.PACK_VERSION]
    assert len(added) == 56
    assert registry["summary"]["autonomic_governance_mesh_bot_count"] == 56
    assert registry["summary"]["max_bot_version"] == 1613
    assert registry["summary"]["target_platform_total_bots"] == 1604
    for row in added:
        assert row["active"] is True
        assert row["lifecycle_state"] == "data_collection_only"
        assert row["training_excluded"] is True
        assert row["paper_trading_enabled"] is False
        assert row["live_trading_enabled"] is False
        assert row["execution_enabled"] is False
        assert row["allocation_enabled"] is False
        assert row["data_collection_capture_mode"] == "thin_digest_with_heartbeat_fallback"
        assert row["data_collection_sample_rate"] == 0.012
        assert row["self_accommodating_policy"]["high_pressure_fallback"] == "heartbeat"
        assert row["label_contract"]["required_join_mode"] == "point_in_time_only"
        assert row["autonomic_governance_mesh_contract"]["authority_boundary"] == "collection_only_advisory_no_execution_no_allocation_no_halt_clearance"
    assert (tmp_path / "config" / "autonomic_governance_mesh_v1.json").exists()
    assert (tmp_path / "governance" / "health" / "autonomic_governance_mesh_latest.json").exists()
    assert (tmp_path / "governance" / "autonomic_governance_mesh" / "system_governor_council").is_dir()


def test_apply_is_idempotent_by_slot_kind(tmp_path: Path) -> None:
    _write_registry(tmp_path)

    first = mesh.apply_registry(tmp_path)
    second = mesh.apply_registry(tmp_path)
    registry = json.loads((tmp_path / "master_bot_registry.json").read_text(encoding="utf-8"))
    added = [row for row in registry["sub_bots"] if row.get("autonomic_governance_mesh_version") == mesh.PACK_VERSION]

    assert first["added_bot_count"] == 56
    assert second["added_bot_count"] == 0
    assert len(added) == 56
    assert second["pack"]["bot_ids"][0] == first["added_bot_ids"][0]
