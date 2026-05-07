from __future__ import annotations

import importlib.util
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = PROJECT_ROOT / "scripts" / "ops" / "training_labeling_intelligence.py"
spec = importlib.util.spec_from_file_location("training_labeling_intelligence", MODULE_PATH)
tli = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(tli)


def _write_registry(root: Path) -> None:
    rows = [
        {
            "bot_id": "brain_refinery_v1613_autonomic_governance_expansion_stability_stress_oracle_governor_bridge_bot",
            "bot_role": "infrastructure_sub_bot",
            "active": True,
            "data_collection_active": True,
            "training_excluded": True,
            "lifecycle_state": "data_collection_only",
            "slot_kind": "autonomic_governance_mesh_expansion_stability_stress_oracle_governor_bridge",
        },
        {
            "bot_id": "brain_refinery_v43_intraday_ultrafast_proxy",
            "bot_role": "signal_sub_bot",
            "active": True,
            "quality_score": 0.6,
            "label_contract": {
                "label_family": "intraday_fast",
                "primary_horizon": "5m_30m_forward_return",
                "required_context": ["one_minute_bars"],
            },
            "data_label_contract_version": "existing_v1",
        },
    ]
    (root / "master_bot_registry.json").write_text(
        json.dumps({"summary": {"total_bots": len(rows), "active_bots": len(rows), "max_bot_version": 1613}, "sub_bots": rows}) + "\n",
        encoding="utf-8",
    )


def test_dry_run_reports_missing_labels_and_plans_intelligence_layer(tmp_path: Path) -> None:
    _write_registry(tmp_path)

    payload = tli.build_payload(tmp_path)

    assert payload["system_count"] == 6
    assert payload["bot_count"] == 24
    assert payload["planned_bot_count"] == 24
    assert payload["target_platform_total_bots"] == 1628
    assert payload["missing_label_contract_count"] == 1
    assert payload["pack"]["bot_ids"][0].startswith("brain_refinery_v1614_")
    assert payload["pack"]["bot_ids"][-1].startswith("brain_refinery_v1637_")
    assert payload["universal_label_contract_version"] == tli.UNIVERSAL_LABEL_CONTRACT_VERSION


def test_apply_adds_bots_and_normalizes_all_label_contracts(tmp_path: Path) -> None:
    _write_registry(tmp_path)

    payload = tli.apply_registry(tmp_path)

    assert payload["added_bot_count"] == 24
    assert payload["label_contract_summary"]["missing_contracts_before"] == 1
    assert payload["label_contract_summary"]["updated_label_contract_bot_count"] == 1
    registry = json.loads((tmp_path / "master_bot_registry.json").read_text(encoding="utf-8"))
    rows = registry["sub_bots"]
    added = [row for row in rows if row.get("capability_pack_slug") == tli.PACK_SLUG]
    assert len(added) == 24
    assert registry["summary"]["training_label_contract_bot_count"] == len(rows)
    assert registry["summary"]["universal_label_contract_bot_count"] == len(rows)
    assert registry["summary"]["training_labeling_intelligence_bot_count"] == 24
    missing_fixed = rows[0]
    preserved = rows[1]
    assert missing_fixed["data_label_contract_version"] == tli.UNIVERSAL_LABEL_CONTRACT_VERSION
    assert missing_fixed["label_contract"]["required_join_mode"] == "point_in_time_only"
    assert preserved["label_contract"]["primary_horizon"] == "5m_30m_forward_return"
    assert preserved["universal_label_contract"]["label_family"] == "intraday_fast"
    for row in added:
        assert row["paper_trading_enabled"] is False
        assert row["live_trading_enabled"] is False
        assert row["execution_enabled"] is False
        assert row["allocation_enabled"] is False
        assert row["data_collection_sample_rate"] == 0.01
        assert row["data_collection_max_daily_storage_mb"] == 1
        assert row["label_contract"]["required_join_mode"] == "point_in_time_only"
    assert (tmp_path / "config" / "training_labeling_intelligence_v1.json").exists()
    assert (tmp_path / "governance" / "health" / "training_labeling_intelligence_latest.json").exists()
    assert (tmp_path / "governance" / "training_labeling_intelligence" / "label_coverage_latest.json").exists()


def test_apply_is_idempotent_by_slot_kind(tmp_path: Path) -> None:
    _write_registry(tmp_path)

    first = tli.apply_registry(tmp_path)
    second = tli.apply_registry(tmp_path)
    registry = json.loads((tmp_path / "master_bot_registry.json").read_text(encoding="utf-8"))
    added = [row for row in registry["sub_bots"] if row.get("capability_pack_slug") == tli.PACK_SLUG]

    assert first["added_bot_count"] == 24
    assert second["added_bot_count"] == 0
    assert len(added) == 24
    assert second["pack"]["bot_ids"][0] == first["added_bot_ids"][0]


def test_training_process_intelligence_reads_walk_forward_coverage_artifacts(tmp_path: Path) -> None:
    _write_registry(tmp_path)
    health = tmp_path / "governance" / "health"
    walk_forward = tmp_path / "governance" / "walk_forward"
    health.mkdir(parents=True)
    walk_forward.mkdir(parents=True)
    (health / "training_quality_control_latest.json").write_text(
        json.dumps(
            {
                "overall_status": "needs_attention",
                "targeted_actions": {"targeted_retrain_bot_ids": ["brain_refinery_v17_mixed_regime"]},
            }
        ),
        encoding="utf-8",
    )
    (health / "training_runtime_control_latest.json").write_text(json.dumps({"snapshot_ready": True}), encoding="utf-8")
    (walk_forward / "coverage_gap_closer_latest.json").write_text(
        json.dumps(
            {
                "active_stage_candidates": [
                    {"bot_id": "brain_refinery_v35_dmi_state_machine"},
                    {"bot_id": "brain_refinery_v4_simple"},
                ],
                "autopilot_contract": {"launch_contract": {"coverage_repair_ready": True}},
            }
        ),
        encoding="utf-8",
    )

    intelligence = tli._training_process_intelligence(tmp_path)

    assert intelligence["coverage_repair_bot_ids"] == [
        "brain_refinery_v35_dmi_state_machine",
        "brain_refinery_v4_simple",
    ]
    assert intelligence["selected_target_source"] == "coverage_repair"
    assert intelligence["recommended_retrain_profile"] == "coverage_canary"
