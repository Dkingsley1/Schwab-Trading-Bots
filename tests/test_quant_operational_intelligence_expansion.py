from __future__ import annotations

import importlib.util
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = PROJECT_ROOT / "scripts" / "ops" / "quant_operational_intelligence_expansion.py"
spec = importlib.util.spec_from_file_location("quant_operational_intelligence_expansion", MODULE_PATH)
quant_ops = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(quant_ops)


def _write_registry(root: Path) -> None:
    (root / "master_bot_registry.json").write_text(
        json.dumps(
            {
                "updated_at_utc": "2026-05-06T00:00:00+00:00",
                "summary": {"total_bots": 1477, "active_bots": 1422, "max_bot_version": 1477},
                "sub_bots": [
                    {
                        "bot_id": "brain_refinery_v1477_exotic_gap_option_jump_risk_bot",
                        "bot_role": "signal_sub_bot",
                        "active": True,
                        "data_collection_active": True,
                        "lifecycle_state": "data_collection_only",
                        "slot_label": "Existing Exotic Bot",
                        "slot_kind": "exotic_gap_option_jump_risk",
                    }
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )


def test_dry_run_plans_quant_and_operational_intelligence_pack(tmp_path: Path) -> None:
    _write_registry(tmp_path)

    payload = quant_ops.build_payload(tmp_path)

    assert payload["system_count"] == 20
    assert payload["quant_system_count"] == 10
    assert payload["operational_system_count"] == 10
    assert payload["bot_count"] == 80
    assert payload["planned_bot_count"] == 80
    assert payload["target_platform_total_bots"] == 1548
    assert payload["pack"]["bot_ids"][0].startswith("brain_refinery_v1478_")
    assert payload["pack"]["bot_ids"][-1].startswith("brain_refinery_v1557_")
    assert payload["pack"]["storage_retention_rule"]["sample_rate"] == 0.018
    assert payload["pack"]["storage_retention_rule"]["max_daily_mb_per_bot"] == 3
    assert payload["pack"]["paper_only_floor"]["live_trading_enabled"] is False
    assert payload["pack"]["paper_only_floor"]["graduation_requires_minimum_observations"] == 65000
    assert payload["pack"]["paper_only_floor"]["graduation_requires_collection_days"] == 160
    systems = {row["slug"] for row in payload["pack"]["intelligence_systems"]}
    assert "alpha_factor_court" in systems
    assert "backlog_outcome_verifier" in systems


def test_apply_adds_collect_only_quant_operational_bots(tmp_path: Path) -> None:
    _write_registry(tmp_path)

    payload = quant_ops.apply_registry(tmp_path)

    assert payload["added_bot_count"] == 80
    assert payload["new_total_bots"] == 81
    registry = json.loads((tmp_path / "master_bot_registry.json").read_text(encoding="utf-8"))
    added = [
        row
        for row in registry["sub_bots"]
        if row.get("quant_operational_intelligence_version") == quant_ops.PACK_VERSION
    ]
    assert len(added) == 80
    assert registry["summary"]["quant_operational_intelligence_bot_count"] == 80
    assert registry["summary"]["data_collection_active_bots"] == 81
    assert registry["summary"]["max_bot_version"] == 1557
    systems = {row["intelligence_system"] for row in added}
    assert "model_uncertainty_calibration" in systems
    assert "safe_command_router" in systems
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
        assert row["minimum_training_observations"] == 65000
        assert row["minimum_data_collection_days"] == 160
        assert row["data_collection_capture_mode"] == "thin_sampled"
        assert row["data_collection_sample_rate"] == 0.018
        assert row["data_collection_compute_guard_mode"] == "thin_digest"
        assert row["capability_pack_contract"]["target_platform_total_bots"] == 1548
        assert row["quant_operational_intelligence_contract"]["contract_version"] == "quant_operational_intelligence_layers_v1"
        assert row["quant_operational_intelligence_contract"]["authority_boundary"] == "collection_only_advisory_no_execution_no_allocation_no_halt_clearance"
        assert row["training_threshold_policy"]["requires_runtime_pressure_clearance"] is True
        assert row["training_threshold_policy"]["requires_paper_live_separation_clearance"] is True
        assert row["label_contract"]["required_join_mode"] == "point_in_time_only"
    assert (tmp_path / "config" / "quant_operational_intelligence_v1.json").exists()
    assert (tmp_path / "governance" / "health" / "quant_operational_intelligence_latest.json").exists()
    assert (tmp_path / "governance" / "quant_operational_intelligence" / "alpha_factor_court").is_dir()
    assert "governance/quant_operational_intelligence/alpha_factor_court" in payload["storage_targets_ready"]


def test_apply_is_idempotent_by_slot_kind(tmp_path: Path) -> None:
    _write_registry(tmp_path)

    first = quant_ops.apply_registry(tmp_path)
    second = quant_ops.apply_registry(tmp_path)
    registry = json.loads((tmp_path / "master_bot_registry.json").read_text(encoding="utf-8"))
    added = [
        row
        for row in registry["sub_bots"]
        if row.get("quant_operational_intelligence_version") == quant_ops.PACK_VERSION
    ]

    assert first["added_bot_count"] == 80
    assert second["added_bot_count"] == 0
    assert len(added) == 80
    assert second["pack"]["bot_ids"][0] == first["added_bot_ids"][0]
