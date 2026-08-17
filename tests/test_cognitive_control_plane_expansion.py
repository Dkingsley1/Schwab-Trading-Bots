from __future__ import annotations

import importlib.util
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = PROJECT_ROOT / "scripts" / "ops" / "cognitive_control_plane_expansion.py"
spec = importlib.util.spec_from_file_location("cognitive_control_plane_expansion", MODULE_PATH)
control = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(control)


def _write_registry(root: Path) -> None:
    (root / "master_bot_registry.json").write_text(
        json.dumps(
            {
                "updated_at_utc": "2026-05-02T00:00:00+00:00",
                "summary": {"total_bots": 1, "active_bots": 1},
                "sub_bots": [
                    {
                        "bot_id": "brain_refinery_v818_seed_bot",
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


def test_dry_run_plans_cognitive_control_plane(tmp_path: Path) -> None:
    _write_registry(tmp_path)

    payload = control.build_payload(tmp_path)

    assert payload["pack_count"] == 1
    assert payload["bot_count"] == 15
    assert payload["planned_bot_count"] == 15
    assert payload["control_plane"]["bot_ids"][0].startswith("brain_refinery_v819_")
    assert payload["control_plane"]["bot_ids"][-1].startswith("brain_refinery_v833_")
    assert payload["control_plane"]["storage_retention_rule"]["dedupe_required"] is True
    assert payload["control_plane"]["storage_retention_rule"]["sample_rate"] == 0.25
    assert payload["control_plane"]["paper_only_floor"]["live_trading_enabled"] is False
    assert payload["control_plane"]["paper_only_floor"]["graduation_requires_minimum_observations"] == 8000
    assert payload["control_plane"]["capacity_check"]["heavy_reasoning_mode"] == "cold_lane_only_when_host_pressure_is_clear"
    assert "hierarchical_planning" in payload["control_plane"]["cognitive_depth"]
    assert "grandmaster_cognition_bridge" in payload["control_plane"]["cognitive_depth"]


def test_apply_adds_collection_only_control_plane_bots(tmp_path: Path) -> None:
    _write_registry(tmp_path)

    payload = control.apply_registry(tmp_path)

    assert payload["added_bot_count"] == 15
    assert payload["new_total_bots"] == 16
    registry = json.loads((tmp_path / "master_bot_registry.json").read_text(encoding="utf-8"))
    added = [row for row in registry["sub_bots"] if row.get("cognitive_control_version") == control.CONTROL_VERSION]
    assert len(added) == 15
    assert registry["summary"]["cognitive_control_plane_bot_count"] == 15
    assert registry["summary"]["structured_capability_pack_bot_count"] == 15
    assert registry["summary"]["data_collection_active_bots"] == 15
    for row in added:
        assert row["active"] is True
        assert row["lifecycle_state"] == "data_collection_only"
        assert row["training_excluded"] is True
        assert row["exclude_from_training"] is True
        assert row["training_candidate_after_threshold"] is True
        assert row["paper_trading_enabled"] is False
        assert row["live_trading_enabled"] is False
        assert row["execution_enabled"] is False
        assert row["paper_trade_lock_required"] is True
        assert row["minimum_training_observations"] == 8000
        assert row["minimum_data_collection_days"] == 30
        assert row["capability_pack_contract"]["bot_pack_size_rule"] == "5_to_15_bots_max"
        assert row["advanced_intelligence_layer_contract"]["contract_version"] == "advanced_intelligence_layers_v4"
    assert (tmp_path / "config" / "cognitive_control_plane_v1.json").exists()
    assert (tmp_path / "governance" / "health" / "cognitive_control_plane_latest.json").exists()


def test_apply_is_idempotent_by_slot_kind(tmp_path: Path) -> None:
    _write_registry(tmp_path)

    first = control.apply_registry(tmp_path)
    second = control.apply_registry(tmp_path)
    registry = json.loads((tmp_path / "master_bot_registry.json").read_text(encoding="utf-8"))
    added = [row for row in registry["sub_bots"] if row.get("cognitive_control_version") == control.CONTROL_VERSION]

    assert first["added_bot_count"] == 15
    assert second["added_bot_count"] == 0
    assert len(added) == 15
    assert second["control_plane"]["bot_ids"][0] == first["added_bot_ids"][0]
