from __future__ import annotations

import json
from pathlib import Path

from scripts.ops import bot_intelligence_mesh as src
from scripts.ops import system_intelligence_coordinator


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")


def _seed_mesh_project(root: Path, *, include_grandmaster: bool = True, training_score: float = 77.0) -> None:
    grand_id = "brain_refinery_v201_grandmaster_bridge_bot"
    master_id = "brain_refinery_v200_sleeve_master_bot"
    rows = [
        {
            "bot_id": "brain_refinery_v1_signal_student",
            "bot_role": "signal_sub_bot",
            "active": True,
            "sleeve_profile": "test_sleeve",
            "reports_to_sleeve_master_bot_id": master_id,
            "bootstrap_teacher_bot_ids": ["brain_refinery_v10_elite_teacher"],
        },
        {
            "bot_id": "brain_refinery_v2_storage_guard",
            "bot_role": "infrastructure_sub_bot",
            "active": True,
            "sleeve_profile": "test_sleeve",
            "target_functions": ["storage_quota_guard", "runtime_throttle"],
            "reports_to_sleeve_master_bot_id": master_id,
        },
        {
            "bot_id": master_id,
            "bot_role": "infrastructure_sub_bot",
            "active": True,
            "slot_kind": "test_sleeve_master",
            "sleeve_profile": "test_sleeve",
            "target_functions": ["sleeve_master", "training_registry_audit"],
            "grandmaster_bridge_bot_id": grand_id,
        },
        {
            "bot_id": "brain_refinery_v10_elite_teacher",
            "bot_role": "signal_sub_bot",
            "active": True,
            "sleeve_profile": "test_sleeve",
            "reports_to_sleeve_master_bot_id": master_id,
        },
    ]
    if include_grandmaster:
        rows.append(
            {
                "bot_id": grand_id,
                "bot_role": "infrastructure_sub_bot",
                "active": True,
                "slot_kind": "test_grandmaster_bridge",
                "sleeve_profile": "test_sleeve",
                "target_functions": ["system_summary", "platform_control_plane"],
            }
        )
    _write_json(root / "master_bot_registry.json", {"sub_bots": rows})
    _write_json(
        root / "governance" / "health" / "training_quality_control_latest.json",
        {
            "overall_status": "needs_attention",
            "training_quality_score": training_score,
            "targeted_actions": {
                "runtime_input_depth_debt_rows": [
                    {
                        "bot_id": "brain_refinery_v56_meta_ranker",
                        "observation_count": 509,
                        "minimum_training_observations": 1000,
                        "observations_needed": 491,
                    }
                ],
                "quality_probation_bot_ids": ["brain_refinery_v1_signal_student"],
                "targeted_retrain_bot_ids": ["brain_refinery_v1_signal_student"],
            },
        },
    )
    _write_json(
        root / "governance" / "health" / "data_collection_observation_rollup_latest.json",
        {
            "overall_status": "ready",
            "collector_count": 4,
            "bots_with_observations": 4,
            "training_ready_count": 4,
            "zero_observation_count": 0,
            "zero_observation_bot_ids": [],
        },
    )
    _write_json(
        root / "governance" / "distillation" / "teacher_student_plan_latest.json",
        {
            "summary": {"teacher_count": 1, "student_count": 1, "assignment_count": 1},
            "teachers": [{"bot_id": "brain_refinery_v10_elite_teacher", "role": "signal_sub_bot"}],
            "assignments": [
                {
                    "student_bot_id": "brain_refinery_v1_signal_student",
                    "student_role": "signal_sub_bot",
                    "teachers": [{"bot_id": "brain_refinery_v10_elite_teacher", "teacher_score": 0.85}],
                }
            ],
        },
    )
    _write_json(
        root / "governance" / "distillation" / "teacher_quality_latest.json",
        {
            "overall_status": "ready",
            "summary": {
                "qualified_teacher_count": 1,
                "elite_teacher_count": 1,
                "strong_teacher_count": 0,
                "uncovered_student_role_count": 0,
            },
            "student_role_coverage": {"uncovered_roles": []},
            "role_coverage": [{"bot_role": "signal_sub_bot", "teacher_count": 1}],
        },
    )
    _write_json(
        root / "governance" / "health" / "overfitting_awareness_latest.json",
        {
            "overall_status": "ready",
            "risk_bot_count": 0,
            "hard_risk_bot_count": 0,
            "guarded_bot_count": 0,
            "high_accuracy_guarded_bot_count": 0,
            "active_status_counts": {"generalization_clean": 1},
            "blocked_teacher_bot_ids": [],
            "broadcast_contract": {"applies_to_tiers": ["infrastructure", "sub", "teacher", "master", "grand_master"]},
            "top_risk_bots": [],
        },
    )
    _write_json(root / "governance" / "health" / "supportability_control_latest.json", {"overall_status": "ready", "students_without_teachers": 0})


def test_mesh_builds_tier_routes_and_honest_quality_target(tmp_path: Path) -> None:
    _seed_mesh_project(tmp_path)

    payload = src.build_payload(tmp_path, edge_limit=100)

    assert payload["active_tier_counts"]["infrastructure"] >= 1
    assert payload["active_tier_counts"]["sub"] >= 1
    assert payload["active_tier_counts"]["master"] >= 1
    assert payload["active_tier_counts"]["grand_master"] >= 1
    assert payload["communication_readiness_score"] == 100.0
    assert payload["quality_readiness_score"] < 100.0
    assert payload["quality_target_status"] == "needs_work"
    assert payload["a_plus_target_contract"]["current_training_quality_score"] == 77.0
    assert payload["a_plus_target_contract"]["current_data_quality_score"] == 100.0
    blocker_keys = {row["key"] for row in payload["a_plus_target_contract"]["blockers"]}
    assert "training_quality_below_100" in blocker_keys
    assert "quality_probation" in blocker_keys
    route_kinds = {row["route_kind"] for row in payload["hierarchy_edges"]}
    assert "sub_to_master" in route_kinds
    assert "infrastructure_to_master" in route_kinds
    assert "master_to_grand_master" in route_kinds
    assert payload["teacher_student_edge_summary"]["edge_count_total"] == 1
    assert payload["overfitting_awareness"]["overall_status"] == "ready"
    assert payload["overfitting_awareness"]["guarded_bot_count"] == 0
    assert payload["teacher_student_intelligence"]["policy"]["overfit_risk_bots_may_teach"] is False
    assert payload["integration_contract"]["does_not_execute_trades"] is True


def test_mesh_blocks_when_required_tier_is_missing(tmp_path: Path) -> None:
    _seed_mesh_project(tmp_path, include_grandmaster=False)

    payload = src.build_payload(tmp_path)

    assert payload["overall_status"] == "blocked"
    assert "grand_master" in payload["missing_tiers"]


def test_system_signal_bus_consumes_bot_intelligence_mesh(tmp_path: Path) -> None:
    _seed_mesh_project(tmp_path)
    payload = src.build_payload(tmp_path)
    src.write_outputs(payload, tmp_path / "governance" / "health" / "bot_intelligence_mesh_latest.json")

    signal_bus = system_intelligence_coordinator.build_signal_bus(tmp_path)
    signal = next(row for row in signal_bus["signals"] if row["name"] == "bot_intelligence_mesh")

    assert signal["loaded"] is True
    assert signal["metrics"]["communication_readiness_score"] == payload["communication_readiness_score"]
    assert signal["metrics"]["quality_readiness_score"] == payload["quality_readiness_score"]
    assert signal["metrics"]["blocker_count"] == payload["a_plus_target_contract"]["blocker_count"]
    assert "teachers=" in signal["summary"]
