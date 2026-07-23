import json
from pathlib import Path

from scripts.ops import live_money_readiness_contract as src


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def _write_ready_sources(project_root: Path) -> None:
    health = project_root / "governance" / "health"
    _write_json(
        health / "paper_execution_truth_layer_latest.json",
        {
            "ok": True,
            "overall_status": "ready",
            "grade": "A",
            "score": 94.0,
            "failed_checks": [],
            "gates": {
                "decision_replay_harness": {"ok": True, "score": 94.0, "reasons": [], "best_candidate": {"win_rate": 0.62}},
                "paper_broker_truth_reconciliation": {
                    "ok": True,
                    "score": 100.0,
                    "broker_truth_v2_grade": "A+",
                    "broker_truth_v2_score": 1.0,
                    "mismatch_count": 0,
                    "source_verification_ok": True,
                },
                "data_ingestion_quality_gate": {"ok": True, "score": 96.0, "total_pending_lines": 20, "oldest_pending_age_seconds": 1.0},
            },
        },
    )
    _write_json(
        health / "promotion_quality_gate_latest.json",
        {
            "ok": True,
            "failed_checks": [],
            "details": {
                "promotion_packet_ok": True,
                "paper_execution_truth_layer_ok": True,
                "daily_verify_ok": True,
                "cohort_drift_baseline_ok": True,
            },
        },
    )
    _write_json(health / "paper_profitability_control_latest.json", {"overall_status": "ready", "profitability_grade": "A+"})
    _write_json(
        health / "source_verification_latest.json",
        {"ok": True, "overall_status": "ready", "overall": {"mean_source_confidence_score": 0.94}, "unverified_sources": []},
    )
    _write_json(health / "health_gates_latest.json", {"hard_gate_triggered": False, "hard_gates": {}})
    _write_json(
        health / "ingestion_storage_control_latest.json",
        {
            "overall_status": "ready",
            "severity": "stable",
            "continuous_run_soak_contract": {
                "ready": True,
                "status": "ready",
                "grade": "A",
                "score": 94.0,
                "horizon_days": 28.0,
                "min_pressure_days": 35.0,
                "blockers": [],
            },
        },
    )
    _write_json(
        health / "training_runtime_control_latest.json",
        {"overall_status": "ready", "snapshot_ready": True, "training_launch_contract": {"launch_allowed": True, "recommended_batch_size": 1}},
    )
    _write_json(
        health / "live_runtime_separation_control_latest.json",
        {
            "overall_status": "ready",
            "clearance_plan": {"clearance_state": "ready"},
            "release_contract": {"live_lane_should_be_read_only": False},
        },
    )
    _write_json(health / "live_readiness_smoke_latest.json", {"ok": True, "overall_status": "ready", "readiness_score": 94.0})
    _write_json(
        health / "global_killswitch_latest.json",
        {
            "timestamp_utc": "2026-08-26T11:00:00+00:00",
            "halt": False,
            "halt_latched": False,
            "halt_required": False,
            "clear_ready": True,
            "operating_mode": "normal",
            "clear_blockers": [],
            "critical_hard_gate_names": [],
            "degraded_clear_blockers": [],
        },
    )
    risk = project_root / "governance" / "risk"
    _write_json(
        risk / "risk_service_boundary_latest.json",
        {
            "timestamp_utc": "2026-08-26T11:00:00+00:00",
            "ok": True,
            "overall_status": "ready",
            "independent_service_boundary": {
                "service_isolation_ready": True,
                "service_count": 5,
                "policy_hash_count": 3,
            },
            "services": {
                "pre_trade_service": {"endpoint_slug": "risk.pre_trade.approval"},
                "kill_switch_service": {"endpoint_slug": "risk.kill_switch"},
            },
        },
    )
    _write_json(
        risk / "portfolio_risk_latest.json",
        {
            "timestamp_utc": "2026-08-26T11:00:00+00:00",
            "risk_level": "low",
            "risk_score": 20.0,
            "limits": {
                "gross_exposure_cap": 0.15,
                "max_single_symbol_share": 0.10,
                "max_intraday_turnover": 0.50,
            },
        },
    )
    _write_json(
        risk / "execution_budget_latest.json",
        {
            "timestamp_utc": "2026-08-26T11:00:00+00:00",
            "risk_level": "low",
            "global": {
                "max_total_actions_per_hour": 5,
                "max_total_open_orders": 2,
            },
            "sleeves": {"core": {"max_actions_per_hour": 5}},
        },
    )


def test_live_money_contract_blocks_before_target_even_when_all_sections_are_a_grade(tmp_path: Path) -> None:
    _write_ready_sources(tmp_path)

    payload = src.build_payload(tmp_path, as_of_date="2026-08-25")

    assert payload["faithful_live_money_ready"] is False
    assert payload["live_money_locked"] is True
    assert payload["days_remaining"] == 1
    assert "target_window_not_complete" in payload["blocking_reasons"]
    assert payload["grade_summary"]["below_floor_sections"] == []


def test_live_money_contract_clears_after_target_only_when_all_sections_are_a_or_better(tmp_path: Path) -> None:
    _write_ready_sources(tmp_path)

    payload = src.build_payload(tmp_path, as_of_date="2026-08-26")

    assert payload["faithful_live_money_ready"] is True
    assert payload["overall_status"] == "ready"
    assert payload["live_money_locked"] is False
    assert payload["blocking_reasons"] == []


def test_live_money_contract_blocks_current_replay_and_soak_debt(tmp_path: Path) -> None:
    _write_ready_sources(tmp_path)
    health = tmp_path / "governance" / "health"
    paper_truth = json.loads((health / "paper_execution_truth_layer_latest.json").read_text(encoding="utf-8"))
    paper_truth["ok"] = False
    paper_truth["grade"] = "C"
    paper_truth["gates"]["decision_replay_harness"] = {
        "ok": False,
        "score": 55.0,
        "reasons": ["counterfactual_win_rate_below_floor"],
        "best_candidate": {"win_rate": 0.0, "aggregate_net_pnl_total": -2292.52},
    }
    _write_json(health / "paper_execution_truth_layer_latest.json", paper_truth)
    storage = json.loads((health / "ingestion_storage_control_latest.json").read_text(encoding="utf-8"))
    storage["continuous_run_soak_contract"] = {"ready": False, "status": "blocked", "grade": "D", "blockers": ["storage_growth_forecast_not_28_day_ready"]}
    _write_json(health / "ingestion_storage_control_latest.json", storage)

    payload = src.build_payload(tmp_path, as_of_date="2026-08-26")

    assert payload["faithful_live_money_ready"] is False
    assert "paper_execution_truth_below_A" in payload["blocking_reasons"]
    assert "decision_replay_harness_below_A" in payload["blocking_reasons"]
    assert "continuous_soak_below_A" in payload["blocking_reasons"]


def test_live_money_contract_grades_managed_safety_controls_without_unlocking_live_money(tmp_path: Path) -> None:
    _write_ready_sources(tmp_path)
    health = tmp_path / "governance" / "health"
    risk = tmp_path / "governance" / "risk"

    paper_truth = json.loads((health / "paper_execution_truth_layer_latest.json").read_text(encoding="utf-8"))
    paper_truth["gates"]["decision_replay_harness"] = {
        "ok": False,
        "status": "warn",
        "score": 82.0,
        "reasons": ["counterfactual_low_sample_outcome_attribution_pending"],
        "best_candidate": {"kept_count": 1, "aggregate_net_pnl_total": 0.0},
        "advisory_only": True,
        "grade_blocking": False,
        "paper_replay_ok": True,
    }
    _write_json(health / "paper_execution_truth_layer_latest.json", paper_truth)
    _write_json(
        health / "ingestion_storage_control_latest.json",
        {
            "overall_status": "ready",
            "severity": "stable",
            "continuous_run_soak_contract": {
                "ready": False,
                "soak_ready": True,
                "status": "watch",
                "grade": "A",
                "score": 94.0,
                "blockers": [],
                "warnings": ["bounded_drain_time_backlog_allowed_for_soak"],
            },
        },
    )
    _write_json(
        health / "training_runtime_control_latest.json",
        {
            "overall_status": "constrained",
            "snapshot_ready": True,
            "training_launch_contract": {
                "mode": "prep_only",
                "launch_allowed": False,
                "prep_allowed": True,
                "launch_blockers": ["autonomic_training_budget_closed"],
                "recommended_batch_size": 0,
                "training_quality_score": 98.0,
            },
        },
    )
    _write_json(
        health / "live_runtime_separation_control_latest.json",
        {
            "overall_status": "ready",
            "clearance_plan": {"clearance_state": "managed_cold_lane_deferred"},
            "release_contract": {"live_lane_should_be_read_only": True},
        },
    )
    _write_json(
        health / "global_killswitch_latest.json",
        {
            "timestamp_utc": "2026-08-26T11:00:00+00:00",
            "halt": False,
            "halt_latched": False,
            "halt_required": False,
            "clear_ready": True,
            "operating_mode": "degraded_collection",
            "clear_blockers": [],
            "critical_hard_gate_names": [],
            "degraded_clear_blockers": ["runtime_clearance=managed_cold_lane_deferred"],
        },
    )
    _write_json(
        risk / "risk_service_boundary_latest.json",
        {
            "timestamp_utc": "2026-08-26T11:00:00+00:00",
            "ok": True,
            "overall_status": "ready",
            "independent_service_boundary": {
                "service_isolation_ready": True,
                "service_count": 5,
                "policy_hash_count": 3,
            },
            "services": {
                "pre_trade_service": {"endpoint_slug": "risk.pre_trade.approval"},
                "kill_switch_service": {"endpoint_slug": "risk.kill_switch"},
            },
        },
    )

    before_target = src.build_payload(tmp_path, as_of_date="2026-08-25")

    assert before_target["faithful_live_money_ready"] is False
    assert before_target["blocking_reasons"] == ["target_window_not_complete"]
    assert before_target["grade_summary"]["below_floor_sections"] == []
    assert before_target["grade_summary"]["not_ready_sections"] == []
    sections = {row["section_id"]: row for row in before_target["sections"]}
    assert sections["decision_replay_harness"]["grade"] == "A+"
    assert sections["decision_replay_harness"]["ready"] is True
    assert sections["training_runtime"]["grade"] == "A+"
    assert sections["live_runtime_release"]["grade"] == "A+"
    assert sections["risk_controls"]["grade"] == "A+"

    at_target = src.build_payload(tmp_path, as_of_date="2026-08-26")

    assert at_target["faithful_live_money_ready"] is False
    assert at_target["live_money_locked"] is True
    assert at_target["operator_execution_release_required"] is True
    assert at_target["blocking_reasons"] == ["live_execution_operator_release_required"]


def test_live_money_contract_treats_replay_rows_low_collecting_as_managed_ready(tmp_path: Path) -> None:
    _write_ready_sources(tmp_path)
    health = tmp_path / "governance" / "health"

    paper_truth = json.loads((health / "paper_execution_truth_layer_latest.json").read_text(encoding="utf-8"))
    paper_truth["gates"]["decision_replay_harness"] = {
        "ok": False,
        "status": "warn",
        "score": 82.0,
        "reasons": [
            "counterfactual_low_sample_outcome_attribution_pending",
            "paper_replay_rows_low_collecting",
        ],
        "best_candidate": {"kept_count": 1, "aggregate_net_pnl_total": 0.0},
        "low_sample": True,
        "advisory_only": True,
        "grade_blocking": False,
        "paper_replay_ok": False,
    }
    _write_json(health / "paper_execution_truth_layer_latest.json", paper_truth)

    payload = src.build_payload(tmp_path, as_of_date="2026-08-25")

    assert payload["blocking_reasons"] == ["target_window_not_complete"]
    assert payload["grade_summary"]["below_floor_sections"] == []
    assert payload["grade_summary"]["not_ready_sections"] == []
    sections = {row["section_id"]: row for row in payload["sections"]}
    assert sections["decision_replay_harness"]["grade"] == "A+"
    assert sections["decision_replay_harness"]["ready"] is True
    assert sections["decision_replay_harness"]["evidence"]["managed_collection_ready"] is True


def test_live_money_contract_grades_idle_seed_ready_promotion_packet_without_unlocking_live_money(
    tmp_path: Path,
) -> None:
    _write_ready_sources(tmp_path)
    health = tmp_path / "governance" / "health"
    promotion_quality = json.loads((health / "promotion_quality_gate_latest.json").read_text(encoding="utf-8"))
    promotion_quality["details"]["promotion_packet_ok"] = False
    promotion_quality["details"]["promotion"] = {
        "promotion_scope_active": False,
        "promote_ok": False,
        "considered_bots": 0,
    }
    _write_json(health / "promotion_quality_gate_latest.json", promotion_quality)
    _write_json(
        tmp_path / "governance" / "champion_challenger" / "promotion_packet_latest.json",
        {
            "ok": False,
            "packet_complete": False,
            "ready_for_committee": False,
            "committee_packet_seed_ready": True,
            "signing_material_ready": False,
            "trained_models_complete": False,
            "dataset": {"rows_sha256": "a" * 64},
            "replayability_contract": {
                "hash_bundle_complete": True,
                "exact_replay_ready": True,
            },
            "gate_results": {
                "training_success_confirmed": True,
                "feature_store_manifest_strict_ok": True,
                "bot_support_owner_guard_ok": True,
                "new_bot_admission_ok": True,
                "retrain_schema_compatibility_ok": True,
                "golden_replay_regression_ok": True,
                "cohort_drift_baseline_ok": True,
                "champion_challenger_probation_ok": True,
                "replay_hash_registry_ok": True,
                "content_store_manifest_present": True,
            },
            "committee": {"seed_ready": True},
            "packet_sha256": "b" * 64,
        },
    )

    before_target = src.build_payload(tmp_path, as_of_date="2026-08-25")

    assert before_target["blocking_reasons"] == ["target_window_not_complete"]
    assert before_target["grade_summary"]["below_floor_sections"] == []
    assert before_target["grade_summary"]["not_ready_sections"] == []
    sections = {row["section_id"]: row for row in before_target["sections"]}
    assert sections["promotion_packet"]["grade"] == "A+"
    assert sections["promotion_packet"]["ready"] is True
    assert sections["promotion_packet"]["evidence"]["managed_idle_preclearance"] is True

    at_target = src.build_payload(tmp_path, as_of_date="2026-08-26")

    assert at_target["faithful_live_money_ready"] is False
    assert at_target["live_money_locked"] is True
    assert at_target["operator_execution_release_required"] is True
    assert at_target["blocking_reasons"] == ["live_execution_operator_release_required"]
