from __future__ import annotations

import importlib.util
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = PROJECT_ROOT / "scripts" / "ops" / "quant_strategy_gap_expansion.py"
spec = importlib.util.spec_from_file_location("quant_strategy_gap_expansion", MODULE_PATH)
quant_gap = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(quant_gap)


def _write_registry(root: Path) -> None:
    (root / "master_bot_registry.json").write_text(
        json.dumps(
            {
                "updated_at_utc": "2026-05-04T00:00:00+00:00",
                "summary": {"total_bots": 1196, "active_bots": 1141},
                "sub_bots": [
                    {
                        "bot_id": "brain_refinery_v1205_institutional_institutional_reporting_evidence_pack_master_bridge_bot",
                        "bot_role": "infrastructure_sub_bot",
                        "active": True,
                        "data_collection_active": True,
                        "lifecycle_state": "data_collection_only",
                        "slot_label": "Existing Institutional Bot",
                        "slot_kind": "institutional_alpha_validation_institutional_reporting_evidence_pack_master_bridge",
                    }
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )


def test_dry_run_plans_twenty_four_strategy_gap_pack(tmp_path: Path) -> None:
    _write_registry(tmp_path)

    payload = quant_gap.build_payload(tmp_path)

    assert payload["strategy_count"] == 24
    assert payload["bot_count"] == 120
    assert payload["planned_bot_count"] == 120
    assert payload["target_platform_total_bots"] == 1316
    assert payload["pack"]["bot_ids"][0].startswith("brain_refinery_v1206_")
    assert payload["pack"]["bot_ids"][-1].startswith("brain_refinery_v1325_")
    assert payload["pack"]["storage_retention_rule"]["sample_rate"] == 0.03
    assert payload["pack"]["storage_retention_rule"]["max_daily_mb_per_bot"] == 4
    assert payload["pack"]["paper_only_floor"]["live_trading_enabled"] is False
    assert payload["pack"]["paper_only_floor"]["graduation_requires_minimum_observations"] == 45000
    assert payload["pack"]["paper_only_floor"]["graduation_requires_collection_days"] == 120
    assert "convertible_bond_arbitrage" in payload["pack"]["strategy_sleeves"]
    assert "sector_pair_rotation_spread_arb" in payload["pack"]["strategy_sleeves"]


def test_apply_adds_collect_only_strategy_gap_bots(tmp_path: Path) -> None:
    _write_registry(tmp_path)

    payload = quant_gap.apply_registry(tmp_path)

    assert payload["added_bot_count"] == 120
    assert payload["new_total_bots"] == 121
    registry = json.loads((tmp_path / "master_bot_registry.json").read_text(encoding="utf-8"))
    added = [row for row in registry["sub_bots"] if row.get("quant_strategy_gap_version") == quant_gap.PACK_VERSION]
    assert len(added) == 120
    assert registry["summary"]["quant_strategy_gap_bot_count"] == 120
    assert registry["summary"]["data_collection_active_bots"] == 121
    assert registry["summary"]["max_bot_version"] == 1325
    sleeves = {row["sleeve_profile"] for row in added}
    assert "merger_event_arbitrage" in sleeves
    assert "dealer_opex_pinning_v2" in sleeves
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
        assert row["minimum_training_observations"] == 45000
        assert row["minimum_data_collection_days"] == 120
        assert row["data_collection_capture_mode"] == "thin_sampled"
        assert row["data_collection_sample_rate"] == 0.03
        assert row["data_collection_compute_guard_mode"] == "thin_digest"
        assert row["capability_pack_contract"]["target_platform_total_bots"] == 1316
        assert row["quant_strategy_gap_contract"]["contract_version"] == "quant_strategy_gap_layers_v1"
        assert row["training_threshold_policy"]["requires_duplicate_alpha_clearance"] is True
        assert row["label_contract"]["required_join_mode"] == "point_in_time_only"
    assert (tmp_path / "config" / "quant_strategy_gap_v1.json").exists()
    assert (tmp_path / "governance" / "health" / "quant_strategy_gap_latest.json").exists()
    assert (tmp_path / "governance" / "quant_strategy_gap" / "convertible_bond_arbitrage").is_dir()
    assert "governance/quant_strategy_gap/convertible_bond_arbitrage" in payload["storage_targets_ready"]


def test_apply_is_idempotent_by_slot_kind(tmp_path: Path) -> None:
    _write_registry(tmp_path)

    first = quant_gap.apply_registry(tmp_path)
    second = quant_gap.apply_registry(tmp_path)
    registry = json.loads((tmp_path / "master_bot_registry.json").read_text(encoding="utf-8"))
    added = [row for row in registry["sub_bots"] if row.get("quant_strategy_gap_version") == quant_gap.PACK_VERSION]

    assert first["added_bot_count"] == 120
    assert second["added_bot_count"] == 0
    assert len(added) == 120
    assert second["pack"]["bot_ids"][0] == first["added_bot_ids"][0]
