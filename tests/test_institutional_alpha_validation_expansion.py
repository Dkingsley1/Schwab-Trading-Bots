from __future__ import annotations

import importlib.util
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = PROJECT_ROOT / "scripts" / "ops" / "institutional_alpha_validation_expansion.py"
spec = importlib.util.spec_from_file_location("institutional_alpha_validation_expansion", MODULE_PATH)
institutional = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(institutional)


def _write_registry(root: Path) -> None:
    (root / "master_bot_registry.json").write_text(
        json.dumps(
            {
                "updated_at_utc": "2026-05-04T00:00:00+00:00",
                "summary": {"total_bots": 1086, "active_bots": 1021},
                "sub_bots": [
                    {
                        "bot_id": "brain_refinery_v1085_frontier_operator_copilot_intent_bridge_evidence_writer_bot",
                        "bot_role": "infrastructure_sub_bot",
                        "active": True,
                        "data_collection_active": True,
                        "lifecycle_state": "data_collection_only",
                        "slot_label": "Existing Frontier Bot",
                        "slot_kind": "frontier_intelligence_operator_copilot_intent_bridge_evidence_writer",
                    }
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )


def test_dry_run_plans_twenty_four_system_institutional_pack(tmp_path: Path) -> None:
    _write_registry(tmp_path)

    payload = institutional.build_payload(tmp_path)

    assert payload["system_count"] == 24
    assert payload["bot_count"] == 120
    assert payload["planned_bot_count"] == 120
    assert payload["target_platform_total_bots"] == 1196
    assert payload["pack"]["bot_ids"][0].startswith("brain_refinery_v1086_")
    assert payload["pack"]["bot_ids"][-1].startswith("brain_refinery_v1205_")
    assert payload["pack"]["storage_retention_rule"]["sample_rate"] == 0.04
    assert payload["pack"]["storage_retention_rule"]["max_daily_mb_per_bot"] == 6
    assert payload["pack"]["paper_only_floor"]["live_trading_enabled"] is False
    assert payload["pack"]["paper_only_floor"]["graduation_requires_minimum_observations"] == 60000
    assert payload["pack"]["paper_only_floor"]["graduation_requires_collection_days"] == 210
    assert "alpha_evidence_court" in payload["pack"]["institutional_systems"]
    assert "institutional_reporting_evidence_pack" in payload["pack"]["institutional_systems"]


def test_apply_adds_collect_only_institutional_validation_bots(tmp_path: Path) -> None:
    _write_registry(tmp_path)

    payload = institutional.apply_registry(tmp_path)

    assert payload["added_bot_count"] == 120
    assert payload["new_total_bots"] == 121
    registry = json.loads((tmp_path / "master_bot_registry.json").read_text(encoding="utf-8"))
    added = [
        row
        for row in registry["sub_bots"]
        if row.get("institutional_alpha_validation_version") == institutional.PACK_VERSION
    ]
    assert len(added) == 120
    assert registry["summary"]["institutional_alpha_validation_bot_count"] == 120
    assert registry["summary"]["data_collection_active_bots"] == 121
    assert registry["summary"]["max_bot_version"] == 1205
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
        assert row["minimum_data_collection_days"] == 210
        assert row["data_collection_capture_mode"] == "thin_sampled"
        assert row["data_collection_sample_rate"] == 0.04
        assert row["data_collection_compute_guard_mode"] == "thin_digest"
        assert row["capability_pack_contract"]["target_platform_total_bots"] == 1196
        assert row["institutional_alpha_validation_contract"]["contract_version"] == "institutional_alpha_validation_layers_v1"
        assert row["training_threshold_policy"]["requires_alpha_evidence_court_clearance"] is True
        assert row["training_threshold_policy"]["requires_model_governance_clearance"] is True
        assert row["label_contract"]["required_join_mode"] == "point_in_time_only"
    assert (tmp_path / "config" / "institutional_alpha_validation_v1.json").exists()
    assert (tmp_path / "governance" / "health" / "institutional_alpha_validation_latest.json").exists()
    assert (tmp_path / "governance" / "institutional_alpha_validation" / "alpha_evidence_court").is_dir()
    assert "governance/institutional_alpha_validation/alpha_evidence_court" in payload["storage_targets_ready"]


def test_apply_is_idempotent_by_slot_kind(tmp_path: Path) -> None:
    _write_registry(tmp_path)

    first = institutional.apply_registry(tmp_path)
    second = institutional.apply_registry(tmp_path)
    registry = json.loads((tmp_path / "master_bot_registry.json").read_text(encoding="utf-8"))
    added = [
        row
        for row in registry["sub_bots"]
        if row.get("institutional_alpha_validation_version") == institutional.PACK_VERSION
    ]

    assert first["added_bot_count"] == 120
    assert second["added_bot_count"] == 0
    assert len(added) == 120
    assert second["pack"]["bot_ids"][0] == first["added_bot_ids"][0]
