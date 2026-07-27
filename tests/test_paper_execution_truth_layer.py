import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.ops import paper_execution_truth_layer as truth
import scripts.promotion_quality_gate as promotion_quality_gate


def _good_inputs() -> dict:
    paper_performance = {
        "ok": True,
        "week": {
            "rolling_change": 125.0,
            "top_profiles": [{"name": "default", "executions": 250}],
        },
        "sleeve_latest": [
            {
                "profile": "default",
                "executions": 250,
                "ending_net_pnl_total": 125.0,
                "change_vs_previous_day": 25.0,
                "win_rate": 0.62,
                "tca_summary": {
                    "mean_expected_slippage_bps": 4.0,
                    "mean_realized_slippage_bps": 5.0,
                    "mean_slippage_gap_bps": 1.0,
                    "mean_partial_fill_ratio": 0.98,
                    "poor_or_fair_fill_count": 0,
                },
            }
        ],
    }
    calibration = {
        "ok": True,
        "samples": 100,
        "metrics": {"mae_bps": 8.0, "p95_bps": 28.0, "mean_bias_bps": 0.4},
        "by_profile": {
            "default": {
                "samples": 100,
                "mae_bps": 8.0,
                "recommended_slippage_scale": 1.05,
            }
        },
        "recommendations": {"env": {"EXEC_SIM_SLIPPAGE_SCALE_EQUITIES": 1.05}},
    }
    counterfactual = {
        "ok": True,
        "top_candidates": [
            {
                "profile": "default",
                "win_rate": 0.61,
                "aggregate_net_pnl_total": 240.0,
                "kept_count": 100,
            }
        ],
    }
    account_study = {
        "ok": True,
        "position_count": 4,
        "account_count": 3,
        "underlying_count": 2,
        "covered_call_roll_watch": {
            "overall_status": "watch",
            "covered_call_count": 1,
            "alert_count": 0,
        },
    }
    covered_call_watch = {
        "ok": True,
        "overall_status": "watch",
        "covered_call_count": 1,
        "alert_count": 0,
    }
    execution_lab = {
        "ok": True,
        "capabilities": {
            "fee_spread_slippage_haircut": True,
            "partial_fill_modeling": True,
            "queue_priority_modeling": True,
            "market_impact_modeling": True,
            "reject_cancel_stale_quote_modeling": True,
            "realistic_option_fills": True,
            "execution_quality_scoring": True,
            "sleeve_specific_friction": True,
            "live_shadow_calibration_inputs": True,
        },
        "top_worst_case_scenarios": [{"slippage_bps": 42.0}],
    }
    ingestion_storage = {
        "ok": True,
        "overall_status": "ready",
        "backpressure": {
            "total_pending_lines": 500,
            "core_pending_lines": 400,
            "pending_lines_threshold": 15000,
            "oldest_pending_age_seconds": 12,
        },
    }
    live_readiness = {
        "ok": True,
        "overall_status": "ready",
        "readiness_score": 100.0,
        "mode": "validate_only",
        "submit_path_enabled": False,
        "hard_blocks": [],
        "warnings": [],
    }
    broker_truth = {
        "ok": True,
        "broker_truth_ok": True,
        "account_count": 3,
        "position_rows": 4,
        "broker_truth_mismatch_count": 0,
        "broker_truth_reconcile_v2": {
            "truth_score": 0.94,
            "truth_grade": "A",
            "account_identity": {"account_count": 3, "coverage_ratio": 1.0},
        },
    }
    source_verification = {
        "ok": True,
        "sources": [
            {"source_id": "market_quote_profiles", "source_confidence_score": 0.95},
            {"source_id": "options_context_mesh", "source_confidence_score": 0.88},
        ],
    }
    return {
        "paper_performance": paper_performance,
        "calibration": calibration,
        "counterfactual": counterfactual,
        "paper_replay": {"ok": True},
        "account_study": account_study,
        "covered_call_watch": covered_call_watch,
        "execution_lab": execution_lab,
        "live_readiness": live_readiness,
        "broker_truth": broker_truth,
        "source_verification": source_verification,
        "ingestion_storage": ingestion_storage,
        "ingestion_backpressure": {"overload": False, "pending_lines_total": 500},
        "promotion_quality": {"ok": True},
    }


def test_paper_execution_truth_layer_builds_all_ten_upgrade_gates() -> None:
    payload = truth.evaluate_truth_layer(**_good_inputs())

    assert payload["ok"] is True
    assert payload["overall_status"] == "ready"
    assert payload["a_plus_ready"] is True
    assert payload["grade"] == "A+"
    assert payload["score"] >= 97.0
    assert sorted(payload["gates"]) == [
        "account_position_awareness",
        "auto_throttle_overtrading",
        "data_ingestion_quality_gate",
        "decision_replay_harness",
        "live_execution_transition_parity",
        "live_quote_fill_calibration",
        "market_regime_stress_mode",
        "options_specific_realism",
        "paper_broker_truth_reconciliation",
        "paper_pnl_haircut_ledger",
        "promotion_gate_hardening",
        "sleeve_execution_scorecards",
    ]
    assert payload["gates"]["paper_pnl_haircut_ledger"]["realism_adjusted_week_pnl"] < 125.0
    assert payload["gates"]["paper_broker_truth_reconciliation"]["status"] == "ready"
    assert payload["sleeve_scorecards"][0]["profile"] == "default"


def test_paper_execution_truth_layer_watches_reset_calibration_window() -> None:
    inputs = _good_inputs()
    inputs["calibration"] = {
        "ok": True,
        "samples": 0,
        "metrics": {"mae_bps": 0.0, "p95_bps": 0.0, "mean_bias_bps": 0.0},
        "calibration_window": {
            "reset_active": True,
            "cutoff_utc": "2026-06-25T14:41:54+00:00",
            "skipped_before_cutoff": 743,
        },
        "recommendations": {"env": {"PAPER_EXECUTION_USE_EXPECTED_FILL_PRICE": 1}},
    }

    payload = truth.evaluate_truth_layer(**inputs)

    assert payload["ok"] is True
    assert payload["overall_status"] == "ready"
    assert payload["grade"] == "A+"
    assert payload["a_plus_ready"] is True
    assert payload["failed_checks"] == []
    assert payload["grade_blocking_warnings"] == []
    assert payload["advisory_warnings"] == ["live_quote_fill_calibration"]
    assert payload["gates"]["live_quote_fill_calibration"]["status"] == "warn"
    assert payload["gates"]["live_quote_fill_calibration"]["reasons"] == [
        "calibration_window_reset_waiting_for_samples"
    ]
    assert payload["gates"]["live_quote_fill_calibration"]["grade_blocking"] is False
    assert payload["gates"]["live_quote_fill_calibration"]["advisory_only"] is True


def test_paper_execution_truth_layer_warns_when_replay_outcomes_are_flat_but_candidates_exist() -> None:
    inputs = _good_inputs()
    inputs["counterfactual"]["top_candidates"] = [
        {
            "profile": "default",
            "win_rate": None,
            "aggregate_net_pnl_total": 0.0,
            "kept_count": 500,
        }
    ]

    payload = truth.evaluate_truth_layer(**inputs)

    assert payload["ok"] is True
    assert payload["overall_status"] == "watch"
    assert payload["failed_checks"] == []
    assert payload["warnings"] == ["decision_replay_harness"]
    gate = payload["gates"]["decision_replay_harness"]
    assert gate["status"] == "warn"
    assert "counterfactual_outcome_attribution_pending" in gate["reasons"]


def test_paper_execution_truth_layer_keeps_low_sample_pending_replay_advisory_without_downgrading_soak_grade() -> None:
    inputs = _good_inputs()
    inputs["counterfactual"]["top_candidates"] = [
        {
            "profile": "futures_index_intraday",
            "win_rate": None,
            "aggregate_net_pnl_total": 0.0,
            "kept_count": 1,
        }
    ]

    payload = truth.evaluate_truth_layer(**inputs)

    assert payload["ok"] is True
    assert payload["overall_status"] == "ready"
    assert payload["grade"] == "A+"
    assert payload["failed_checks"] == []
    assert payload["warnings"] == []
    assert payload["advisory_warnings"] == ["decision_replay_harness"]
    gate = payload["gates"]["decision_replay_harness"]
    assert gate["status"] == "warn"
    assert gate["grade_blocking"] is False
    assert gate["advisory_only"] is True
    assert gate["low_sample"] is True
    assert gate["reasons"] == ["counterfactual_low_sample_outcome_attribution_pending"]


def test_paper_execution_truth_layer_keeps_low_replay_rows_probationary_when_collecting() -> None:
    inputs = _good_inputs()
    inputs["paper_replay"] = {"ok": False, "failed_checks": ["paper_rows_low"], "rows": 0}
    inputs["counterfactual"]["top_candidates"] = [
        {
            "profile": "futures_index_intraday",
            "win_rate": None,
            "aggregate_net_pnl_total": 0.0,
            "kept_count": 1,
        }
    ]

    payload = truth.evaluate_truth_layer(**inputs)

    assert payload["ok"] is True
    assert payload["overall_status"] == "ready"
    assert payload["grade"] == "A+"
    assert payload["failed_checks"] == []
    assert payload["advisory_warnings"] == ["decision_replay_harness"]
    gate = payload["gates"]["decision_replay_harness"]
    assert gate["grade_blocking"] is False
    assert "paper_replay_rows_low_collecting" in gate["reasons"]


def test_paper_execution_truth_layer_keeps_empty_counterfactual_candidates_advisory_while_collecting() -> None:
    inputs = _good_inputs()
    inputs["counterfactual"]["top_candidates"] = []
    inputs["paper_replay"] = {"ok": True, "failed_checks": []}

    payload = truth.evaluate_truth_layer(**inputs)

    assert payload["ok"] is True
    assert payload["overall_status"] == "ready"
    assert payload["grade"] == "A+"
    assert payload["failed_checks"] == []
    assert payload["advisory_warnings"] == ["decision_replay_harness"]
    gate = payload["gates"]["decision_replay_harness"]
    assert gate["status"] == "warn"
    assert gate["grade_blocking"] is False
    assert gate["advisory_only"] is True
    assert gate["reasons"] == ["counterfactual_candidates_pending_collecting"]


def test_paper_execution_truth_layer_blocks_stale_skip_only_replay() -> None:
    inputs = _good_inputs()
    inputs["paper_replay"] = {
        "ok": False,
        "failed_checks": ["paper_rows_low", "stale_execution_skips_only"],
        "rows": 0,
    }
    inputs["counterfactual"]["top_candidates"] = [
        {
            "profile": "futures_index_intraday",
            "win_rate": None,
            "aggregate_net_pnl_total": 0.0,
            "kept_count": 1,
        }
    ]

    payload = truth.evaluate_truth_layer(**inputs)

    assert payload["ok"] is False
    assert "decision_replay_harness" in payload["failed_checks"]
    gate = payload["gates"]["decision_replay_harness"]
    assert gate["status"] == "blocked"
    assert "paper_replay_stale_skips_only" in gate["reasons"]


def test_paper_execution_truth_layer_does_not_warn_when_flat_replay_has_attributed_win_rate() -> None:
    inputs = _good_inputs()
    inputs["counterfactual"]["top_candidates"] = [
        {
            "profile": "bond",
            "win_rate": 0.505376,
            "aggregate_net_pnl_total": -0.0,
            "kept_count": 406,
        }
    ]

    payload = truth.evaluate_truth_layer(**inputs)

    assert payload["ok"] is True
    assert payload["overall_status"] == "ready"
    assert payload["failed_checks"] == []
    assert payload["warnings"] == []
    gate = payload["gates"]["decision_replay_harness"]
    assert gate["status"] == "ready"
    assert gate["reasons"] == []
    assert gate["advisory_reasons"] == ["counterfactual_win_rate_below_floor_attributed_nonnegative"]


def test_paper_execution_truth_layer_keeps_low_sample_negative_replay_advisory_without_downgrading_soak_grade() -> None:
    inputs = _good_inputs()
    inputs["counterfactual"]["top_candidates"] = [
        {
            "profile": "default",
            "win_rate": 0.0,
            "aggregate_net_pnl_total": -0.025,
            "kept_count": 14,
        }
    ]

    payload = truth.evaluate_truth_layer(**inputs)

    assert payload["ok"] is True
    assert payload["overall_status"] == "ready"
    assert payload["grade"] == "A+"
    assert payload["a_plus_ready"] is True
    assert payload["failed_checks"] == []
    assert payload["warnings"] == []
    assert payload["grade_blocking_warnings"] == []
    assert payload["advisory_warnings"] == ["decision_replay_harness"]
    gate = payload["gates"]["decision_replay_harness"]
    assert gate["status"] == "warn"
    assert gate["grade_blocking"] is False
    assert gate["advisory_only"] is True
    assert gate["low_sample"] is True
    assert gate["min_blocking_kept_count"] == truth.MIN_BLOCKING_COUNTERFACTUAL_KEPT_COUNT
    assert "counterfactual_low_sample_win_rate_below_floor" in gate["reasons"]
    assert "counterfactual_low_sample_aggregate_nonpositive" in gate["reasons"]


def test_paper_execution_truth_layer_keeps_covered_call_roll_watch_operator_advisory_visible_without_downgrading_soak_grade() -> None:
    inputs = _good_inputs()
    inputs["covered_call_watch"] = {
        "ok": False,
        "overall_status": "critical",
        "covered_call_count": 3,
        "alert_count": 1,
    }
    inputs["account_study"]["covered_call_roll_watch"] = dict(inputs["covered_call_watch"])

    payload = truth.evaluate_truth_layer(**inputs)

    assert payload["ok"] is True
    assert payload["overall_status"] == "ready"
    assert payload["grade"] == "A+"
    assert payload["a_plus_ready"] is True
    assert payload["failed_checks"] == []
    assert payload["warnings"] == []
    assert payload["advisory_warnings"] == ["options_specific_realism"]
    gate = payload["gates"]["options_specific_realism"]
    assert gate["status"] == "warn"
    assert gate["grade_blocking"] is False
    assert gate["operator_advisory_only"] is True
    assert gate["score"] == 100.0
    assert gate["reasons"] == ["covered_call_watch_critical", "covered_call_alerts_present"]
    assert payload["operator_advisories"][0]["gate"] == "options_specific_realism"


def test_paper_execution_truth_layer_warns_on_no_current_fill_scorecard_debt() -> None:
    inputs = _good_inputs()
    inputs["paper_performance"]["sleeve_latest"] = [
        {
            "profile": "options_on_futures",
            "data_status": "current_live_no_fills",
            "executions": 0,
            "ending_net_pnl_total": -3.68,
            "change_vs_previous_day": 19.56,
            "win_rate": 0.5,
            "tca_summary": {
                "mean_expected_slippage_bps": 63.0,
                "mean_realized_slippage_bps": 0.0,
                "mean_slippage_gap_bps": -63.0,
                "mean_partial_fill_ratio": 1.0,
                "poor_or_fair_fill_count": 1,
            },
        }
    ]

    payload = truth.evaluate_truth_layer(**inputs)

    assert payload["ok"] is True
    assert payload["overall_status"] == "ready"
    assert payload["failed_checks"] == []
    assert payload["warnings"] == []
    assert payload["gates"]["sleeve_execution_scorecards"]["status"] == "ready"
    assert payload["gates"]["sleeve_execution_scorecards"]["advisory_no_fill_profiles"] == [
        "options_on_futures"
    ]
    scorecard = payload["sleeve_scorecards"][0]
    assert scorecard["status"] == "watch"
    assert "no_current_fills_for_blocking_execution_truth" in scorecard["reasons"]


def test_paper_execution_truth_layer_keeps_stale_latest_available_debt_advisory() -> None:
    inputs = _good_inputs()
    inputs["paper_performance"]["sleeve_latest"].append(
        {
            "profile": "swing_aggressive",
            "day_utc": "20260604",
            "current_day_available": False,
            "data_status": "latest_available",
            "executions": 51,
            "ending_net_pnl_total": 0.0,
            "change_vs_previous_day": -0.0001,
            "win_rate": 0.0,
            "tca_summary": {
                "mean_expected_slippage_bps": 80.0,
                "mean_realized_slippage_bps": 0.0,
                "mean_slippage_gap_bps": -80.0,
                "mean_partial_fill_ratio": 1.0,
                "poor_or_fair_fill_count": 12,
            },
        }
    )

    payload = truth.evaluate_truth_layer(**inputs)

    assert payload["ok"] is True
    assert payload["overall_status"] == "ready"
    assert payload["failed_checks"] == []
    gate = payload["gates"]["sleeve_execution_scorecards"]
    assert gate["status"] == "ready"
    assert gate["failing_profiles"] == []
    assert gate["advisory_stale_latest_profiles"] == ["swing_aggressive"]
    scorecard = next(row for row in payload["sleeve_scorecards"] if row["profile"] == "swing_aggressive")
    assert scorecard["status"] == "watch"
    assert "stale_latest_available_for_current_truth" in scorecard["reasons"]


def test_paper_execution_truth_layer_warns_on_promotion_only_quality_failures() -> None:
    inputs = _good_inputs()
    inputs["promotion_quality"] = {
        "ok": False,
        "failed_checks": [
            "promotion_gate_blocked",
            "insufficient_considered_bots",
            "daily_verify_not_ok",
            "new_bot_graduation_not_ok",
            "new_bot_admission_not_ok",
            "feature_store_manifest_not_ready",
            "retrain_schema_compatibility_not_ok",
            "promotion_packet_not_ready",
            "paper_execution_truth_layer_not_ok",
        ],
        "details": {
            "paper_execution_truth_layer_failed_checks": ["promotion_gate_hardening"],
            "daily_verify_unresolved_failed_checks": ["nightly_resilience_check"],
        },
    }

    payload = truth.evaluate_truth_layer(**inputs)

    assert payload["ok"] is True
    assert payload["overall_status"] == "ready"
    assert payload["failed_checks"] == []
    assert payload["warnings"] == []
    gate = payload["gates"]["promotion_gate_hardening"]
    assert gate["status"] == "ready"
    assert gate["promotion_quality_advisory_only"] is True
    assert gate["reasons"] == []
    assert gate["promotion_quality_blocking_failed_checks"] == []


def test_paper_execution_truth_layer_breaks_self_referential_promotion_quality_loop() -> None:
    inputs = _good_inputs()
    inputs["promotion_quality"] = {
        "ok": False,
        "failed_checks": ["daily_verify_not_ok", "paper_execution_truth_layer_not_ok"],
        "details": {
            "daily_verify_unresolved_failed_checks": ["paper_execution_calibration_report"],
            "paper_execution_truth_layer_failed_checks": ["promotion_gate_hardening"],
        },
    }

    payload = truth.evaluate_truth_layer(**inputs)

    assert payload["ok"] is True
    assert payload["overall_status"] == "ready"
    assert payload["failed_checks"] == []
    assert payload["gates"]["promotion_gate_hardening"]["promotion_quality_gate_ok"] is True


def test_overtrading_throttle_uses_latest_positive_net_when_current_day_has_no_fills() -> None:
    inputs = _good_inputs()
    inputs["paper_performance"]["week"]["top_profiles"] = [{"name": "default", "executions": 25_000}]
    inputs["paper_performance"]["sleeve_latest"][0]["executions"] = 0
    inputs["paper_performance"]["sleeve_latest"][0]["ending_net_pnl_total"] = 500.0
    inputs["paper_performance"]["sleeve_latest"][0]["change_vs_previous_day"] = 450.0

    payload = truth.evaluate_truth_layer(**inputs)

    throttle = payload["gates"]["auto_throttle_overtrading"]
    assert throttle["status"] == "ready"
    assert throttle["throttle_actions"] == []
    assert payload["a_plus_ready"] is True


def test_paper_execution_truth_layer_blocks_when_replay_and_ingestion_fail() -> None:
    inputs = _good_inputs()
    inputs["counterfactual"] = {"ok": False, "top_candidates": []}
    inputs["ingestion_storage"] = {
        "ok": False,
        "overall_status": "degraded",
        "backpressure": {
            "total_pending_lines": 20000,
            "core_pending_lines": 19000,
            "pending_lines_threshold": 15000,
        },
    }

    payload = truth.evaluate_truth_layer(**inputs)

    assert payload["ok"] is False
    assert "decision_replay_harness" in payload["failed_checks"]
    assert "data_ingestion_quality_gate" not in payload["failed_checks"]
    assert payload["gates"]["data_ingestion_quality_gate"]["status"] == "warn"
    assert "decision_replay_harness_blocked" in payload["gates"]["promotion_gate_hardening"]["reasons"]


def test_paper_execution_truth_layer_blocks_paper_activity_without_broker_truth() -> None:
    inputs = _good_inputs()
    inputs["broker_truth"] = {"ok": False, "broker_truth_ok": False, "error": "api_circuit_open"}

    payload = truth.evaluate_truth_layer(**inputs)

    assert payload["ok"] is False
    assert "paper_broker_truth_reconciliation" in payload["failed_checks"]
    gate = payload["gates"]["paper_broker_truth_reconciliation"]
    assert gate["status"] == "blocked"
    assert "paper_activity_without_clean_broker_truth" in gate["reasons"]


def test_paper_broker_reconciliation_allows_optional_source_debt_when_core_sources_are_verified() -> None:
    inputs = _good_inputs()
    core_rows = [
        {
            "source_id": source_id,
            "ok": True,
            "verification_status": "single_source_verified",
            "source_confidence_score": 0.91,
        }
        for source_id in truth.BROKER_RECONCILIATION_CORE_SOURCE_IDS
    ]
    inputs["source_verification"] = {
        "ok": False,
        "overall_status": "degraded",
        "sources": core_rows
        + [
            {
                "source_id": "ticker_news_context",
                "ok": False,
                "verification_status": "single_source_unverified",
                "source_confidence_score": 0.40,
            }
        ],
    }

    payload = truth.evaluate_truth_layer(**inputs)

    assert payload["ok"] is True
    gate = payload["gates"]["paper_broker_truth_reconciliation"]
    assert gate["status"] == "ready"
    assert gate["broker_reconciliation_core_sources_ready"] is True
    assert gate["advisory_reasons"] == ["optional_source_verification_lanes_not_ready"]


def test_paper_broker_reconciliation_keeps_context_source_debt_advisory_when_broker_truth_clean() -> None:
    inputs = _good_inputs()
    source_rows = [
        {
            "source_id": source_id,
            "ok": True,
            "verification_status": "single_source_verified",
            "source_confidence_score": 0.68,
        }
        for source_id in truth.BROKER_RECONCILIATION_CORE_SOURCE_IDS
    ]
    source_rows[0]["ok"] = False
    source_rows[0]["verification_status"] = "single_source_unverified"
    source_rows[0]["source_confidence_score"] = 0.30
    inputs["source_verification"] = {
        "ok": False,
        "overall_status": "degraded",
        "sources": source_rows,
    }

    payload = truth.evaluate_truth_layer(**inputs)

    assert payload["ok"] is True
    gate = payload["gates"]["paper_broker_truth_reconciliation"]
    assert gate["status"] == "ready"
    assert gate["broker_truth_clean_for_source_advisory"] is True
    assert "source_verification_context_debt_not_blocking_clean_broker_truth" in gate["advisory_reasons"]
    assert "source_confidence_thin_context_advisory" in gate["advisory_reasons"]
    assert gate["missing_or_unverified_core_sources"]


def test_paper_broker_reconciliation_blocks_when_core_source_is_unverified_and_broker_truth_dirty() -> None:
    inputs = _good_inputs()
    source_rows = [
        {
            "source_id": source_id,
            "ok": True,
            "verification_status": "single_source_verified",
            "source_confidence_score": 0.91,
        }
        for source_id in truth.BROKER_RECONCILIATION_CORE_SOURCE_IDS
    ]
    source_rows[0]["ok"] = False
    source_rows[0]["verification_status"] = "single_source_unverified"
    source_rows[0]["source_confidence_score"] = 0.30
    inputs["source_verification"] = {
        "ok": False,
        "overall_status": "degraded",
        "sources": source_rows,
    }
    inputs["broker_truth"]["broker_truth_mismatch_count"] = 1

    payload = truth.evaluate_truth_layer(**inputs)

    assert payload["ok"] is False
    gate = payload["gates"]["paper_broker_truth_reconciliation"]
    assert gate["status"] == "blocked"
    assert "paper_or_manual_position_delta_present" in gate["reasons"]
    assert "source_verification_not_ready" in gate["reasons"]
    assert gate["missing_or_unverified_core_sources"]


def test_promotion_quality_gate_hard_requires_truth_layer_for_active_promotions() -> None:
    ok, failed_checks, details = promotion_quality_gate.evaluate_quality(
        {"promote_ok": True, "considered_bots": 5, "fail_share": 0.0},
        {"ok": True, "failed_checks": []},
        {"ok": True},
        {"ok": True},
        {"ok": True},
        {"ok": True},
        {"ok": True},
        paper_execution_truth_layer={"ok": False, "overall_status": "blocked", "failed_checks": ["decision_replay_harness"]},
        max_fail_share=0.25,
        min_considered_bots=4,
        require_replay=True,
        require_reconciliation_slo=False,
    )

    assert ok is False
    assert "paper_execution_truth_layer_not_ok" in failed_checks
    assert details["paper_execution_truth_layer_ok"] is False
    assert details["paper_execution_truth_layer_failed_checks"] == ["decision_replay_harness"]


def test_promotion_quality_gate_resolves_daily_verify_truth_layer_check_when_fresh() -> None:
    ok, failed_checks, details = promotion_quality_gate.evaluate_quality(
        {"promote_ok": True, "considered_bots": 5, "fail_share": 0.0},
        {"ok": False, "failed_checks": ["paper_execution_truth_layer"]},
        {"ok": True},
        {"ok": True},
        {"ok": True},
        {"ok": True},
        {"ok": True},
        paper_execution_truth_layer={"ok": True, "overall_status": "ready", "failed_checks": []},
        max_fail_share=0.25,
        min_considered_bots=4,
        require_replay=True,
        require_reconciliation_slo=False,
    )

    assert ok is True
    assert failed_checks == []
    assert details["daily_verify_unresolved_failed_checks"] == []
    assert details["daily_verify_resolved_failed_checks"] == ["paper_execution_truth_layer"]
