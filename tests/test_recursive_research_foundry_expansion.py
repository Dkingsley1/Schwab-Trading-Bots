from __future__ import annotations

import importlib.util
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = PROJECT_ROOT / "scripts" / "ops" / "recursive_research_foundry_expansion.py"
spec = importlib.util.spec_from_file_location("recursive_research_foundry_expansion", MODULE_PATH)
foundry = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(foundry)


def _write_registry(root: Path) -> None:
    (root / "master_bot_registry.json").write_text(
        json.dumps(
            {
                "updated_at_utc": "2026-05-02T00:00:00+00:00",
                "summary": {"total_bots": 1, "active_bots": 1},
                "sub_bots": [
                    {
                        "bot_id": "brain_refinery_v833_seed_bot",
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


def test_dry_run_plans_recursive_research_foundry(tmp_path: Path) -> None:
    _write_registry(tmp_path)

    payload = foundry.build_payload(tmp_path)

    assert payload["pack_count"] == 1
    assert payload["bot_count"] == 15
    assert payload["planned_bot_count"] == 15
    assert payload["foundry"]["bot_ids"][0].startswith("brain_refinery_v834_")
    assert payload["foundry"]["bot_ids"][-1].startswith("brain_refinery_v848_")
    assert payload["foundry"]["storage_retention_rule"]["dedupe_required"] is True
    assert payload["foundry"]["storage_retention_rule"]["sample_rate"] == 0.2
    assert payload["foundry"]["paper_only_floor"]["live_trading_enabled"] is False
    assert payload["foundry"]["paper_only_floor"]["graduation_requires_minimum_observations"] == 12000
    assert payload["foundry"]["capacity_check"]["heavy_reasoning_mode"] == "cold_lane_digest_only_until_host_pressure_clear"
    assert "proof_obligation_generation" in payload["foundry"]["research_depth"]
    assert "grandmaster_foundry_bridge" in payload["foundry"]["research_depth"]


def test_apply_adds_collection_only_foundry_bots(tmp_path: Path) -> None:
    _write_registry(tmp_path)

    payload = foundry.apply_registry(tmp_path)

    assert payload["added_bot_count"] == 15
    assert payload["new_total_bots"] == 16
    registry = json.loads((tmp_path / "master_bot_registry.json").read_text(encoding="utf-8"))
    added = [row for row in registry["sub_bots"] if row.get("recursive_foundry_version") == foundry.FOUNDRY_VERSION]
    assert len(added) == 15
    assert registry["summary"]["recursive_research_foundry_bot_count"] == 15
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
        assert row["minimum_training_observations"] == 12000
        assert row["minimum_data_collection_days"] == 45
        assert row["capability_pack_contract"]["bot_pack_size_rule"] == "5_to_15_bots_max"
        assert row["advanced_intelligence_layer_contract"]["contract_version"] == "advanced_intelligence_layers_v5"
    assert (tmp_path / "config" / "recursive_research_foundry_v1.json").exists()
    assert (tmp_path / "governance" / "health" / "recursive_research_foundry_latest.json").exists()


def test_apply_is_idempotent_by_slot_kind(tmp_path: Path) -> None:
    _write_registry(tmp_path)

    first = foundry.apply_registry(tmp_path)
    second = foundry.apply_registry(tmp_path)
    registry = json.loads((tmp_path / "master_bot_registry.json").read_text(encoding="utf-8"))
    added = [row for row in registry["sub_bots"] if row.get("recursive_foundry_version") == foundry.FOUNDRY_VERSION]

    assert first["added_bot_count"] == 15
    assert second["added_bot_count"] == 0
    assert len(added) == 15
    assert second["foundry"]["bot_ids"][0] == first["added_bot_ids"][0]
