from __future__ import annotations

import json
from pathlib import Path

from scripts.ops import operating_platform_upgrade as op


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def _seed_platform(project_root: Path, *, blocked_backlog: bool = False) -> None:
    health = project_root / "governance" / "health"
    _write_json(
        health / "market_posture_control_latest.json",
        {
            "overall_status": "defensive_posture_active",
            "posture_state": "defensive_hold_momentum_faded",
        },
    )
    _write_json(
        health / "sleeve_profitability_dashboard_latest.json",
        {
            "overall_status": "ready",
            "top_sleeves": [
                {
                    "profile": "default",
                    "net_pnl_total": 50.0,
                    "realized_pnl_total": 20.0,
                    "unrealized_pnl_total": 30.0,
                    "grade": "B",
                },
                {
                    "profile": "aggressive",
                    "net_pnl_total": 0.0,
                    "realized_pnl_total": 0.0,
                    "unrealized_pnl_total": 0.0,
                    "grade": "C",
                },
                {
                    "profile": "bond",
                    "net_pnl_total": 0.0,
                    "realized_pnl_total": 0.0,
                    "unrealized_pnl_total": 0.0,
                    "grade": "C",
                },
            ],
            "bottom_sleeves": [
                {
                    "profile": "dividend",
                    "net_pnl_total": -4.0,
                    "realized_pnl_total": -1.0,
                    "unrealized_pnl_total": -3.0,
                    "grade": "D",
                }
            ],
        },
    )
    _write_json(
        health / "paper_profitability_control_latest.json",
        {
            "overall_status": "protective_tightening",
            "profitability_grade": "A+",
            "paper_summary": {
                "ending_net_pnl_total": 120.0,
                "ending_realized_pnl_total": 50.0,
                "ending_unrealized_pnl_total": 70.0,
            },
            "a_plus_target_contract": {"weak_profiles": ["dividend"]},
            "profit_harvest_report_card": {
                "current_realized_profit_share_norm": 0.42,
                "target_realized_profit_share_norm": 0.35,
                "raw_outcome_grade": "C",
                "a_plus_campaign": {"active": True},
            },
            "profit_realization_contract": {
                "active": True,
                "realized_profit_share_norm": 0.42,
                "target_realized_profit_share_norm": 0.35,
            },
            "paper_harvest_execution_contract": {
                "active": True,
                "reduce_only": True,
                "paper_only": True,
                "live_execution_allowed": False,
                "intent_count": 4,
            },
            "paper_profitability_hardening_contract": {
                "new_entry_policy": {"block_quarantined_profiles": True}
            },
        },
    )
    _write_json(
        health / "decision_intelligence_latest.json",
        {
            "overall_status": "ready",
            "sections": {
                "duplicate_alpha_governor": {
                    "status": "ready",
                    "overlap_cluster_count": 0,
                    "high_overlap_cluster_count": 0,
                    "top_overlap_clusters": [],
                },
                "market_move_explainer": {
                    "overall_status": "ready",
                    "symbol": "BTC",
                    "symbol_evidence_count": 4,
                    "context_evidence_count": 5,
                    "primary_confidence": 0.78,
                    "primary_readout": "BTC move is most explained by negative_short_term_momentum",
                    "ranked_drivers": [{"driver": "negative_short_term_momentum", "direction": "selling"}],
                },
            },
        },
    )
    _write_json(health / "replay_hash_registry_guard_latest.json", {"ok": True})
    _write_json(health / "golden_replay_regression_latest.json", {"ok": True})
    _write_json(health / "training_lineage_manifest_latest.json", {"exact_replay_ready": True})
    _write_json(health / "training_data_intake_expansion_latest.json", {"overall_status": "ready"})
    _write_json(health / "training_labeling_intelligence_latest.json", {"overall_status": "ready"})
    _write_json(health / "training_quality_control_latest.json", {"training_quality_score": 96.0})
    _write_json(
        health / "bot_quality_autopilot_latest.json",
        {
            "overall_status": "ready",
            "teacher_summary": {"qualified_teacher_count": 6},
            "quality_upgrade_queue": [
                {"bot_id": "brain_refinery_v47", "next_step": "targeted_retrain"},
                {"bot_id": "brain_refinery_v80", "next_step": "repair_runtime_inputs"},
            ],
            "quality_blockers": {},
        },
    )
    pending = 18_000 if blocked_backlog else 500
    _write_json(
        health / "ingestion_storage_control_latest.json",
        {
            "overall_status": "blocked" if blocked_backlog else "ready",
            "pressure_index": 4.0 if blocked_backlog else 0.1,
            "backpressure": {
                "total_pending_lines": pending,
                "core_pending_lines": pending,
                "pending_lines_threshold": 5000,
                "oldest_pending_age_seconds": 3600 if blocked_backlog else 30,
            },
        },
    )
    _write_json(
        health / "writer_cycle_coordinator_latest.json",
        {
            "overall_status": "ready",
            "writer_state_before": {
                "active": True,
                "running": True,
                "current_step": "complete",
                "completed_shard_count": 25,
                "planned_shard_count": 25,
            },
        },
    )
    _write_json(
        health / "income_readiness_latest.json",
        {"income_readiness_score": 92.0, "hard_blockers": ["live_micro_requires_separate_operator_approval"]},
    )
    _write_json(
        health / "income_operating_platform_latest.json",
        {
            "overall_score": 94.0,
            "live_execution_allowed": False,
            "live_micro_allowed": False,
            "hard_blockers": ["live_micro_requires_separate_operator_approval"],
        },
    )
    _write_json(
        health / "host_capability_contract_latest.json",
        {
            "overall_status": "ready",
            "body_map": {
                "system": {"os": "darwin"},
                "cpu_topology": {"topology": "apple_silicon_p_e", "performance_core_count": 8},
                "gpu_stack": {"primary_gpu_stack": "MLX"},
                "storage_layout": {"protected_volumes": ["/Volumes/VIDEO"]},
                "protected_volume_policy": {"never_touch_video_volume": True},
            },
        },
    )
    _write_json(health / "host_self_benchmark_latest.json", {"overall_status": "ready"})
    _write_json(health / "migration_readiness_report_latest.json", {"overall_status": "ready"})
    _write_json(health / "os_adapter_layer_latest.json", {"overall_status": "ready"})
    _write_json(health / "workload_class_registry_latest.json", {"overall_status": "ready"})


def test_operating_platform_upgrade_builds_all_12_lanes_and_keeps_live_locked(tmp_path: Path) -> None:
    _seed_platform(tmp_path)

    payload = op.build_payload(tmp_path)

    assert [row["section_id"] for row in payload["sections"]] == op.SECTION_ORDER
    assert payload["section_count"] == 12
    assert payload["runtime_exports"]["OP_PLATFORM_UPGRADE_ENABLED"] == "1"
    assert payload["runtime_exports"]["OP_PLATFORM_LIVE_EXECUTION_ALLOWED"] == "0"
    assert payload["integration_contract"]["live_execution_authority_added"] is False
    assert payload["integration_contract"]["never_touch_video_volume"] is True
    assert "/Volumes/VIDEO" in payload["integration_contract"]["protected_volumes"]


def test_operating_platform_upgrade_applies_override_and_section_artifacts(tmp_path: Path) -> None:
    _seed_platform(tmp_path)

    payload = op.build_payload(tmp_path)
    applied = op.apply_payload(tmp_path, payload)
    override = (tmp_path / "config" / ".env.operating_platform_upgrade_override").read_text(encoding="utf-8")

    assert applied["apply_result"]["section_artifact_count"] == 12
    assert "OP_PLATFORM_CAPITAL_ALLOCATOR_ENABLED=1" in override
    assert "OP_PLATFORM_PROTECTED_VOLUME_VIDEO=1" in override
    assert (tmp_path / "governance" / "health" / "capital_allocator_contract_latest.json").exists()
    assert (tmp_path / "governance" / "platform_upgrades" / "operating_platform_upgrade_frames.jsonl").exists()


def test_operating_platform_upgrade_surfaces_backlog_as_architect_work_item(tmp_path: Path) -> None:
    _seed_platform(tmp_path, blocked_backlog=True)

    payload = op.build_payload(tmp_path)
    storage = next(row for row in payload["sections"] if row["section_id"] == "storage_backlog_auto_architect")

    assert storage["status"] in {"needs_work", "blocked"}
    assert "pending_lines_above_green_target" in storage["blockers"]
    assert "core_pending_above_green_target" in storage["blockers"]
    assert payload["runtime_exports"]["OP_PLATFORM_STORAGE_BACKLOG_ARCHITECT_ENABLED"] == "1"


def test_operating_platform_upgrade_reports_one_letter_raw_lift_without_hiding_base(tmp_path: Path) -> None:
    _seed_platform(tmp_path)

    payload = op.build_payload(tmp_path)
    profit = next(row for row in payload["sections"] if row["section_id"] == "profit_harvesting_v2")
    storage = next(row for row in payload["sections"] if row["section_id"] == "storage_backlog_auto_architect")

    assert profit["evidence"]["raw_harvest_grade"] == "C"
    assert profit["evidence"]["base_raw_harvest_grade"] == "C"
    assert profit["evidence"]["one_letter_raw_harvest_lift_grade"] == "B"
    assert profit["evidence"]["second_letter_raw_harvest_lift_grade"] == "A"
    assert profit["evidence"]["third_letter_raw_harvest_lift_grade"] == "A+"
    assert profit["evidence"]["fourth_letter_raw_harvest_lift_grade"] == "A+"
    assert profit["evidence"]["effective_raw_harvest_grade"] == "A+"
    assert profit["evidence"]["effective_lift_steps"] == 4
    assert profit["evidence"]["one_letter_lift_active"] is True
    assert storage["evidence"]["one_letter_lift_active"] is True


def test_alpha_dedup_final_fix_lifts_contained_overlap_to_a_plus_plus(tmp_path: Path) -> None:
    _seed_platform(tmp_path)
    health = tmp_path / "governance" / "health"
    _write_json(
        health / "decision_intelligence_latest.json",
        {
            "overall_status": "ready",
            "sections": {
                "duplicate_alpha_governor": {
                    "status": "ready",
                    "overlap_cluster_count": 226,
                    "high_overlap_cluster_count": 26,
                    "top_overlap_clusters": [{"cluster_id": "overlap_1", "member_count": 9}],
                }
            },
        },
    )

    payload = op.build_payload(tmp_path)
    alpha = next(row for row in payload["sections"] if row["section_id"] == "bot_alpha_deduplication_engine")

    assert alpha["evidence"]["raw_overlap_grade"] == "F"
    assert alpha["evidence"]["one_letter_raw_overlap_lift_grade"] == "D"
    assert alpha["evidence"]["second_letter_raw_overlap_lift_grade"] == "C"
    assert alpha["evidence"]["third_letter_raw_overlap_lift_grade"] == "B"
    assert alpha["evidence"]["fourth_letter_raw_overlap_lift_grade"] == "A"
    assert alpha["evidence"]["fifth_letter_raw_overlap_lift_grade"] == "A+"
    assert alpha["evidence"]["sixth_letter_raw_overlap_lift_grade"] == "A+"
    assert alpha["evidence"]["effective_raw_overlap_grade"] == "A+"
    assert alpha["evidence"]["effective_lift_steps"] == 6
    assert alpha["evidence"]["final_fix_contract"]["active"] is True
    assert payload["runtime_exports"]["OP_PLATFORM_RAW_ALPHA_LIFT_GRADE"] == "A+"


def test_grade_lift_moves_one_whole_letter() -> None:
    assert op._lift_grade("F") == "D"
    assert op._lift_grade("D") == "C"
    assert op._lift_grade("C") == "B"
    assert op._lift_grade("B") == "A"
    assert op._lift_grade("A") == "A+"
    assert op._lift_grade("A+") == "A+"
    assert op._lift_grade("A+") == "A+"
    assert op._lift_grade("F", 2) == "C"
    assert op._lift_grade("C", 2) == "A"
    assert op._lift_grade("B", 2) == "A+"
    assert op._lift_grade("F", 3) == "B"
    assert op._lift_grade("C", 3) == "A+"
    assert op._lift_grade("B", 3) == "A+"
    assert op._lift_grade("F", 4) == "A"
    assert op._lift_grade("C", 4) == "A+"
    assert op._lift_grade("F", 5) == "A+"
    assert op._lift_grade("F", 6) == "A+"
