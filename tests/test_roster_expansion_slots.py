import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.ops import roster_expansion_slots as slots_src
from scripts.ops import roster_resilience_planner as roster_src


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def test_roster_expansion_slots_build_and_apply_registry(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    _write_json(
        project_root / "master_bot_registry.json",
        {
            "summary": {"total_bots": 1, "active_bots": 0, "inactive_bots": 1, "active_signal_sub_bots": 0, "active_infrastructure_sub_bots": 0, "inactive_signal_sub_bots": 1, "inactive_infrastructure_sub_bots": 0},
            "sub_bots": [
                {
                    "bot_id": "brain_refinery_v10_seasonal",
                    "bot_role": "signal_sub_bot",
                    "active": False,
                    "reason": "inactive",
                    "promotion_reason": "quality_gate_hold_prev_plus_0.005",
                    "weight": 0.0,
                    "preference_score": 0.0,
                    "quality_score": 0.9,
                    "test_accuracy": 0.93,
                    "candidate_test_accuracy": 0.93,
                    "candidate_quality_score": 0.99,
                    "previous_best_accuracy": 0.93,
                    "no_improvement_streak": 0,
                    "deleted_from_rotation": False,
                    "delete_reason": "",
                    "promoted": False,
                    "model_path": "",
                    "log_file": "",
                    "candidate_log_file": "",
                    "lifecycle_state": "inactive",
                }
            ],
        },
    )
    _write_json(
        project_root / "governance" / "distillation" / "teacher_quality_latest.json",
        {
            "qualified_teachers": [
                {"bot_id": "brain_refinery_v10_seasonal", "bot_role": "signal_sub_bot"},
                {"bot_id": "brain_refinery_v99_defensive_dividend_concentration", "bot_role": "options_sub_bot"},
            ]
        },
    )
    _write_json(
        project_root / "governance" / "health" / "supportability_control_latest.json",
        {
            "teacher_student": {
                "teacher_gap_by_role": [
                    {"student_role": "signal_sub_bot", "missing_assignments": 8},
                    {"student_role": "infrastructure_sub_bot", "missing_assignments": 4},
                    {"student_role": "options_sub_bot", "missing_assignments": 2},
                ]
            }
        },
    )
    _write_json(
        project_root / "governance" / "walk_forward" / "coverage_seed_latest.json",
        {
            "seed_queue": [
                {"bot_id": "brain_refinery_v68_risk_budget_layer", "bot_role": "infrastructure_sub_bot"},
                {"bot_id": "brain_refinery_v42_tick_to_swing_alignment", "bot_role": "signal_sub_bot"},
            ]
        },
    )
    _write_json(project_root / "governance" / "health" / "regime_control_plane_latest.json", {"regime_state": "risk_off_shock"})

    payload = slots_src.build_payload(project_root)

    assert payload["summary"]["planned_slot_count"] == len(slots_src.DEFAULT_SLOT_SPECS)
    assert payload["summary"]["missing_slot_count"] == len(slots_src.DEFAULT_SLOT_SPECS)
    assert payload["summary"]["role_counts"]["infrastructure_sub_bot"] >= 28
    assert payload["summary"]["role_counts"]["signal_sub_bot"] >= 136
    assert payload["summary"]["role_counts"]["options_sub_bot"] >= 29
    assert payload["summary"]["role_counts"]["futures_sub_bot"] == 17
    assert payload["roster_slots"][0]["priority"] == "critical"
    assert payload["summary"]["live_regime"] == "risk_off_shock"
    assert payload["summary"]["regime_fit_slot_count"] >= 1
    regime_slot_ids = [row["bot_id"] for row in payload["current_regime_priority_slots"]]
    assert payload["current_regime_priority_slots"][0]["bot_id"] == "brain_refinery_v109_defensive_options_risk_off_teacher"
    assert "brain_refinery_v109_defensive_options_risk_off_teacher" in regime_slot_ids

    apply_result = slots_src.apply_registry(project_root, registry_path=project_root / "master_bot_registry.json")
    registry = json.loads((project_root / "master_bot_registry.json").read_text(encoding="utf-8"))
    bot_ids = {str(row.get("bot_id") or "") for row in registry.get("sub_bots") or []}

    assert apply_result["added_slots"] == len(slots_src.DEFAULT_SLOT_SPECS)
    assert "brain_refinery_v267_infra_teacher_execution_quality_champion" in bot_ids
    assert "brain_refinery_v116_drawdown_circuit_allocator" in bot_ids
    assert "brain_refinery_v119_put_call_stress_reversal_overlay" in bot_ids
    assert "brain_refinery_v120_energy_shock_inflation_pass_through" in bot_ids
    assert "brain_refinery_v129_liquidity_void_air_pocket_guard" in bot_ids
    assert "brain_refinery_v136_news_sentiment_crowding_reversal" in bot_ids
    assert "brain_refinery_v257_crypto_spot_momentum_regime_bot" in bot_ids
    assert "brain_refinery_v266_crypto_weekend_gap_liquidity_bot" in bot_ids
    assert "brain_refinery_v313_master_roster_load_balancer" in bot_ids
    assert "brain_refinery_v317_collection_observation_value_ranker" in bot_ids
    assert registry["summary"]["total_bots"] == len(slots_src.DEFAULT_SLOT_SPECS) + 1
    assert registry["summary"]["active_bots"] == len(slots_src.DEFAULT_SLOT_SPECS)
    assert registry["summary"]["inactive_infrastructure_sub_bots"] == 0
    assert registry["summary"]["inactive_signal_sub_bots"] == 1


def test_roster_resilience_planner_surfaces_expansion_plan(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    _write_json(
        project_root / "governance" / "health" / "supportability_control_latest.json",
        {
            "supportability": {"active_bots": 1, "active_supportable_bots": 0, "active_supportability_score": 0.0},
            "teacher_student": {"teacher_count": 0, "students_without_teachers": 14},
            "teacher_quality": {"elite_teacher_count": 0},
        },
    )
    _write_json(project_root / "governance" / "health" / "training_requalification_latest.json", {"reactivation_ready_count": 0})
    _write_json(
        project_root / "governance" / "health" / "training_requalification_latest.json",
        {
            "reactivation_ready_count": 0,
            "top_candidates": [
                {
                    "bot_id": "brain_refinery_v35_dmi_state_machine",
                    "bot_role": "signal_sub_bot",
                    "priority": 94.771,
                    "walk_forward_status": "insufficient_runs",
                    "actions": ["recover_training_log", "repair_runtime_inputs"],
                }
            ],
        },
    )
    _write_json(project_root / "governance" / "walk_forward" / "coverage_seed_latest.json", {"coverage_shortfall_bots": 4, "standing_queue": {"seed_queue_size": 8}})
    _write_json(project_root / "governance" / "walk_forward" / "new_bot_graduation_latest.json", {"maturity": {"mature_bots": 0}})
    _write_json(
        project_root / "governance" / "health" / "roster_expansion_slots_latest.json",
        {
            "overall_status": "degraded",
            "summary": {
                "planned_slot_count": 13,
                "registered_slot_count": 6,
                "missing_slot_count": 7,
                "critical_slots_missing": ["brain_refinery_v267_infra_teacher_execution_quality_champion"],
                "live_regime": "risk_off_shock",
            },
            "current_regime_priority_slots": [
                {
                    "bot_id": "brain_refinery_v109_defensive_options_risk_off_teacher",
                    "bot_role": "options_sub_bot",
                    "priority": "critical",
                    "slot_label": "Defensive Options Teacher",
                    "registered": False,
                    "regime_fit_score": 3,
                }
            ],
            "recommended_actions": ["sync the planned roster expansion slots into the master registry so the bench roadmap is explicit"],
        },
    )
    _write_json(
        project_root / "master_bot_registry.json",
        {
            "sub_bots": [
                {
                    "bot_id": "brain_refinery_v101_guard_heavy_regime_memory",
                    "bot_role": "signal_sub_bot",
                    "active": False,
                    "reason": "new_runtime_candidate",
                    "promotion_reason": "new_runtime_candidate",
                    "lifecycle_state": "inactive",
                },
                {
                    "bot_id": "brain_refinery_v105_feed_consensus_execution_guard",
                    "bot_role": "infrastructure_sub_bot",
                    "active": False,
                    "reason": "new_runtime_candidate",
                    "promotion_reason": "new_runtime_candidate",
                    "lifecycle_state": "inactive",
                },
            ]
        },
    )
    _write_json(project_root / "governance" / "health" / "regime_control_plane_latest.json", {"regime_state": "risk_off_shock"})

    payload = roster_src.build_payload(project_root)

    assert payload["overall_status"] == "blocked"
    assert payload["replacement_shortlist"][0]["bot_id"] == "brain_refinery_v35_dmi_state_machine"
    assert payload["a_plus_contract"]["a_plus_ready"] is False
    assert payload["roster_expansion"]["planned_slot_count"] == 13
    assert payload["roster_expansion"]["missing_slot_count"] == 7
    assert payload["current_regime"]["live_regime"] == "risk_off_shock"
    assert payload["current_regime"]["regime_fit_replacements"][0]["bot_id"] == "brain_refinery_v101_guard_heavy_regime_memory"
    assert payload["current_regime"]["regime_priority_slots"][0]["bot_id"] == "brain_refinery_v109_defensive_options_risk_off_teacher"
    assert any("roster expansion slots" in action for action in payload["recommended_actions"])
