from __future__ import annotations

import importlib.util
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = PROJECT_ROOT / "scripts" / "ops" / "intelligence_capability_expansion.py"
spec = importlib.util.spec_from_file_location("intelligence_capability_expansion", MODULE_PATH)
ice = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(ice)


def _write_registry(root: Path) -> None:
    (root / "master_bot_registry.json").write_text(
        json.dumps(
            {
                "updated_at_utc": "2026-05-02T00:00:00+00:00",
                "master_policy": {"min_active_bots": 150},
                "summary": {"total_bots": 1, "active_bots": 1},
                "sub_bots": [
                    {
                        "bot_id": "brain_refinery_v763_seed_bot",
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


def test_dry_run_plans_eight_guarded_capability_packs(tmp_path: Path) -> None:
    _write_registry(tmp_path)

    payload = ice.build_payload(tmp_path)

    assert payload["pack_count"] == 8
    assert payload["bot_count_per_pack"] == 5
    assert payload["planned_bot_count"] == 40
    assert payload["advanced_intelligence_layer_contract"]["critic_loop"] == "observer_critic_loop_trace"
    for pack in payload["packs"]:
        assert pack["bot_count"] == 5
        assert len(pack["bot_ids"]) == 5
        assert pack["dedicated_data_intake"]
        assert pack["storage_retention_rule"]["dedupe_required"] is True
        assert pack["paper_only_floor"]["live_trading_enabled"] is False
        assert pack["sleeve_master_bot_id"]
        assert pack["regression_guard_bot_id"]
        assert pack["capacity_check"]["active_bot_floor"] == 700


def test_apply_adds_collection_only_pack_bots_and_updates_summary(tmp_path: Path) -> None:
    _write_registry(tmp_path)

    payload = ice.apply_registry(tmp_path)

    assert payload["added_bot_count"] == 40
    assert payload["new_total_bots"] == 41
    assert payload["added_bot_ids"][0].startswith("brain_refinery_v764_")
    assert payload["added_bot_ids"][-1].startswith("brain_refinery_v803_")
    registry = json.loads((tmp_path / "master_bot_registry.json").read_text(encoding="utf-8"))
    rows = registry["sub_bots"]
    added = [row for row in rows if row.get("capability_pack_version") == ice.CAPABILITY_PACK_VERSION]
    assert len(added) == 40
    assert registry["summary"]["total_bots"] == 41
    assert registry["summary"]["structured_capability_pack_bot_count"] == 40
    assert registry["summary"]["data_collection_active_bots"] == 40
    for row in added:
        assert row["active"] is True
        assert row["data_collection_active"] is True
        assert row["lifecycle_state"] == "data_collection_only"
        assert row["training_excluded"] is True
        assert row["exclude_from_training"] is True
        assert row["training_candidate_after_threshold"] is True
        assert row["paper_trading_enabled"] is False
        assert row["live_trading_enabled"] is False
        assert row["execution_enabled"] is False
        assert row["paper_trade_lock_required"] is True
        assert row["paper_runtime_capacity_floor"] == 700
        assert row["minimum_training_observations"] == 3000
        assert row["minimum_data_collection_days"] == 14
        assert row["capability_pack_contract"]["bot_pack_size_rule"] == "5_to_15_bots_max"
        assert row["advanced_intelligence_layer_contract"]["resource_budget"] == "compute_capital_allocation_trace"
    assert (tmp_path / "config" / "intelligence_capability_packs_v1.json").exists()
    assert (tmp_path / "governance" / "health" / "intelligence_capability_expansion_latest.json").exists()


def test_apply_is_idempotent_by_slot_kind(tmp_path: Path) -> None:
    _write_registry(tmp_path)

    first = ice.apply_registry(tmp_path)
    second = ice.apply_registry(tmp_path)
    registry = json.loads((tmp_path / "master_bot_registry.json").read_text(encoding="utf-8"))
    added = [row for row in registry["sub_bots"] if row.get("capability_pack_version") == ice.CAPABILITY_PACK_VERSION]

    assert first["added_bot_count"] == 40
    assert second["added_bot_count"] == 0
    assert len(added) == 40
    assert second["packs"][0]["bot_ids"][0] == first["added_bot_ids"][0]
