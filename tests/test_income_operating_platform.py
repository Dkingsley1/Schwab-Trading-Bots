import importlib.util
import json
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "ops" / "income_operating_platform.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("income_operating_platform", SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("failed to load income_operating_platform")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def _seed_health(project_root: Path) -> None:
    health = project_root / "governance" / "health"
    _write_json(
        health / "income_readiness_latest.json",
        {
            "income_readiness_score": 91.0,
            "income_readiness_grade": "A+",
            "hard_blockers": ["live_micro_requires_separate_operator_approval"],
        },
    )
    _write_json(
        health / "paper_profitability_control_latest.json",
        {
            "overall_status": "protective_tightening",
            "paper_summary": {
                "ending_net_pnl_total": 2500.0,
                "ending_realized_pnl_total": 900.0,
                "ending_unrealized_pnl_total": 1600.0,
            },
            "profit_harvest_report_card": {
                "grade": "A+",
                "raw_outcome_grade": "B",
                "current_realized_profit_share_norm": 0.36,
                "current_unrealized_profit_share_norm": 0.64,
                "target_realized_profit_share_norm": 0.35,
            },
            "profit_realization_contract": {
                "active": True,
                "realized_profit_share_norm": 0.36,
                "unrealized_profit_share_norm": 0.64,
                "target_realized_profit_share_norm": 0.35,
            },
            "paper_harvest_execution_contract": {
                "active": True,
                "reduce_only": True,
                "paper_only": True,
                "live_execution_allowed": False,
                "intent_count": 8,
            },
            "a_plus_target_contract": {
                "weak_profiles": ["swing_aggressive"],
            },
        },
    )
    _write_json(
        health / "paper_performance_latest.json",
        {
            "sleeve_latest": [
                {
                    "profile": "default",
                    "executions": 120,
                    "ending_net_pnl_total": 700.0,
                    "ending_realized_pnl_total": 300.0,
                    "ending_unrealized_pnl_total": 400.0,
                    "mean_slippage_gap_bps": 0.5,
                    "poor_or_fair_fill_count": 0,
                },
                {
                    "profile": "swing_aggressive",
                    "executions": 60,
                    "ending_net_pnl_total": -120.0,
                    "ending_realized_pnl_total": -30.0,
                    "ending_unrealized_pnl_total": -90.0,
                    "mean_slippage_gap_bps": 0.0,
                    "poor_or_fair_fill_count": 0,
                },
            ],
            "history_daily_series": [
                {"day_utc": f"202605{day:02d}", "ending_net_pnl_total": 1000.0 + day * 10.0, "change_vs_previous_day": 10.0}
                for day in range(1, 31)
            ],
        },
    )
    _write_json(health / "paper_execution_calibration_latest.json", {"timestamp_utc": "2999-01-01T00:00:00+00:00", "ok": True})
    _write_json(health / "training_runtime_control_latest.json", {"overall_status": "ready"})
    _write_json(health / "training_quality_control_latest.json", {"training_quality_score": 95.0})
    _write_json(health / "promotion_quality_gate_latest.json", {"ok": True, "failed_checks": []})
    _write_json(
        health / "account_policy_context_latest.json",
        {
            "timestamp_utc": "2999-01-01T00:00:00+00:00",
            "account_policy_context": {
                "configured_account_slots": [
                    {
                        "account_policy_key": "paper_only",
                        "auto_order_enabled": False,
                        "requires_operator_confirmation": True,
                        "env_bindings": [{"name": "X", "present": False}],
                    }
                ]
            },
        },
    )
    _write_json(
        health / "chaos_drill_coordinator_latest.json",
        {
            "overall_status": "ready",
            "drills": [{"drill": "snapshot_restore", "overdue": False}],
            "overdue_drills": [],
            "drill_program": {"program_score": 100.0},
            "restore_discipline": {"restore_proof_ready": True},
            "schedule_contract": {"discipline_ready": True},
        },
    )
    _write_json(health / "process_watchdog_latest.json", {"overall_status": "ready"})
    _write_json(health / "runtime_gate_dashboard_latest.json", {"overall": {"status": "ready"}})
    _write_json(health / "storage_quota_guard_latest.json", {"ok": True})
    _write_json(health / "ingestion_backpressure_latest.json", {"pending_lines_total": 0})


def test_income_operating_platform_builds_all_ten_lanes_and_keeps_live_locked(tmp_path: Path) -> None:
    module = _load_module()
    _seed_health(tmp_path)

    payload = module.build_payload(tmp_path)

    assert [row["section_id"] for row in payload["sections"]] == module.SECTION_ORDER
    assert payload["live_execution_allowed"] is False
    assert payload["live_micro_allowed"] is False
    assert payload["paper_only"] is True
    assert payload["requires_separate_live_micro_approval"] is True
    assert "live_micro_requires_separate_operator_approval" in payload["hard_blockers"]
    assert "/Volumes/VIDEO" in payload["runtime_contract"]["protected_volumes"]


def test_income_operating_platform_ranks_sleeves_and_contains_weak_profile(tmp_path: Path) -> None:
    module = _load_module()
    _seed_health(tmp_path)

    payload = module.build_payload(tmp_path)
    ranking = next(row for row in payload["sections"] if row["section_id"] == "sleeve_profitability_ranking")
    ranked = ranking["evidence"]["ranked_sleeves"]

    assert ranked[0]["profile"] == "default"
    assert ranked[0]["profitability_tier"] == "scale_candidate"
    weak = next(row for row in ranked if row["profile"] == "swing_aggressive")
    assert weak["profitability_tier"] == "contained_weak_sleeve"


def test_income_operating_platform_lifts_controlled_drawdown_without_hiding_raw_debt(tmp_path: Path) -> None:
    module = _load_module()
    _seed_health(tmp_path)
    health = tmp_path / "governance" / "health"
    paper_profitability = json.loads((health / "paper_profitability_control_latest.json").read_text(encoding="utf-8"))
    paper_profitability.update(
        {
            "low_grade_control_report_card": {
                "control_posture_grade": "A+",
                "a_plus_control_ready": True,
                "active_blocker_count": 0,
            },
            "raw_operational_containment_filter": {
                "contained_grade": "A+",
                "contained_weak_profile_count": 1,
                "contained_strategy_control_count": 1,
            },
            "paper_profitability_hardening_contract": {
                "new_entry_policy": {"block_quarantined_profiles": True},
            },
        }
    )
    paper_profitability["a_plus_target_contract"] = {
        "current": {
            "weak_profile_count": 1,
            "unprotected_weak_profile_count": 0,
            "unprotected_strategy_control_count": 0,
        }
    }
    _write_json(health / "paper_profitability_control_latest.json", paper_profitability)
    _write_json(
        health / "paper_performance_latest.json",
        {
            "sleeve_latest": [],
            "history_daily_series": [
                {"day_utc": "20260501", "ending_net_pnl_total": 100000.0, "change_vs_previous_day": 100000.0},
                {"day_utc": "20260502", "ending_net_pnl_total": 100.0, "change_vs_previous_day": -99900.0},
            ]
            + [
                {"day_utc": f"202605{day:02d}", "ending_net_pnl_total": 100.0 + day, "change_vs_previous_day": 1.0}
                for day in range(3, 31)
            ],
        },
    )

    payload = module.build_payload(tmp_path)
    drawdown = next(row for row in payload["sections"] if row["section_id"] == "drawdown_governor")

    assert drawdown["grade"] == "A++"
    assert drawdown["evidence"]["raw_drawdown_grade"] == "F"
    assert drawdown["evidence"]["drawdown_control_ready"] is True
    assert "raw_drawdown_evidence_needs_clean_refreshes" in drawdown["blockers"]
    assert "drawdown_ratio_above_income_limit" not in payload["hard_blockers"]


def test_income_operating_platform_runtime_control_and_dashboard_are_paper_only(tmp_path: Path) -> None:
    module = _load_module()
    _seed_health(tmp_path)

    payload = module.build_payload(tmp_path)
    control = module.build_runtime_control_payload(payload)
    dashboard = module.build_dashboard_payload(payload)

    assert control["paper_only"] is True
    assert control["live_execution_allowed"] is False
    assert control["live_micro_allowed"] is False
    assert dashboard["headline"]["paper_only"] is True
    assert dashboard["headline"]["live_execution_allowed"] is False


def test_income_operating_platform_can_reach_controlled_100_without_unlocking_live(tmp_path: Path) -> None:
    module = _load_module()
    _seed_health(tmp_path)
    health = tmp_path / "governance" / "health"
    _write_json(
        health / "income_readiness_latest.json",
        {
            "income_readiness_score": 89.5,
            "income_readiness_grade": "A",
            "hard_blockers": ["live_micro_requires_separate_operator_approval"],
        },
    )
    _write_json(
        health / "training_runtime_control_latest.json",
        {
            "overall_status": "constrained",
            "training_launch_contract": {"prep_allowed": True},
            "pretraining_drain_buffer": {"safe_to_launch_now": True},
        },
    )
    _write_json(health / "training_quality_control_latest.json", {"training_quality_score": 100.0})
    _write_json(
        health / "promotion_quality_gate_latest.json",
        {
            "ok": True,
            "failed_checks": [],
            "details": {
                "feature_store_manifest_ready": True,
                "retrain_schema_compatibility_ok": True,
                "golden_replay_regression_ok": True,
                "cohort_drift_baseline_ok": True,
                "leak_overfit_ok": True,
                "replay_ok": True,
                "replay_hash_registry_ok": True,
                "champion_challenger_probation_ok": True,
                "reconciliation_slo_ok": True,
                "snapshot_coverage_ok": True,
                "data_source_divergence_ok": True,
                "artifact_freshness_ok": True,
            },
        },
    )
    paper_profitability = json.loads((health / "paper_profitability_control_latest.json").read_text(encoding="utf-8"))
    paper_profitability.update(
        {
            "low_grade_control_report_card": {
                "control_posture_grade": "A+",
                "a_plus_control_ready": True,
                "active_blocker_count": 0,
            },
            "raw_operational_containment_filter": {
                "contained_grade": "A+",
                "contained_weak_profile_count": 1,
                "contained_strategy_control_count": 1,
            },
            "paper_profitability_hardening_contract": {
                "new_entry_policy": {"block_quarantined_profiles": True},
            },
            "a_plus_target_contract": {
                "weak_profiles": ["swing_aggressive"],
                "current": {
                    "weak_profile_count": 1,
                    "unprotected_weak_profile_count": 0,
                    "unprotected_strategy_control_count": 0,
                },
            },
        }
    )
    _write_json(health / "paper_profitability_control_latest.json", paper_profitability)
    _write_json(
        health / "paper_performance_latest.json",
        {
            "sleeve_latest": [
                {
                    "profile": f"sleeve_{idx}",
                    "executions": 100,
                    "ending_net_pnl_total": 100.0 + idx,
                    "ending_realized_pnl_total": 25.0,
                    "ending_unrealized_pnl_total": 75.0,
                    "mean_slippage_gap_bps": 0.5,
                    "poor_or_fair_fill_count": 0,
                }
                for idx in range(6)
            ]
            + [
                {
                    "profile": "swing_aggressive",
                    "executions": 10,
                    "ending_net_pnl_total": -10.0,
                    "ending_realized_pnl_total": 0.0,
                    "ending_unrealized_pnl_total": -10.0,
                    "mean_slippage_gap_bps": 0.5,
                    "poor_or_fair_fill_count": 0,
                }
            ]
            + [
                {
                    "profile": f"probation_{idx}",
                    "executions": 0,
                    "ending_net_pnl_total": 0.0,
                    "ending_realized_pnl_total": 0.0,
                    "ending_unrealized_pnl_total": 0.0,
                    "mean_slippage_gap_bps": 0.0,
                    "poor_or_fair_fill_count": 0,
                }
                for idx in range(3)
            ],
            "history_daily_series": [
                {"day_utc": f"202605{day:02d}", "ending_net_pnl_total": 1000.0 + day * 10.0, "change_vs_previous_day": 10.0}
                for day in range(1, 31)
            ],
        },
    )

    payload = module.build_payload(tmp_path)
    by_id = {row["section_id"]: row for row in payload["sections"]}

    assert payload["income_operating_score"] == 100.0
    assert payload["income_operating_grade"] == "A++"
    assert payload["live_execution_allowed"] is False
    assert payload["live_micro_allowed"] is False
    assert payload["non_live_hard_blockers"] == []
    assert by_id["income_promotion_gate"]["evidence"]["raw_income_promotion_score"] < 100.0
    assert by_id["income_promotion_gate"]["evidence"]["controlled_money_promotion_ready"] is True
    assert by_id["live_micro_lane"]["evidence"]["controlled_micro_safety_ready"] is True
    assert by_id["withdrawal_simulator"]["evidence"]["controlled_withdrawal_ready"] is True
    assert by_id["account_rules_layer"]["evidence"]["controlled_account_ready"] is True
    assert by_id["sleeve_profitability_ranking"]["evidence"]["controlled_sleeve_rotation_ready"] is True
