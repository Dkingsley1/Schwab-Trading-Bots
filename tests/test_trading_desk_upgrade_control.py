import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.ops import trading_desk_upgrade_control as control


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def _write_sources(root: Path, overrides: dict[str, dict] | None = None) -> None:
    base = {
        "paper_execution_truth_layer": {
            "ok": True,
            "overall_status": "ready",
            "score": 97.0,
            "raw_metric_score": 88.0,
            "grade": "A+",
            "a_plus_ready": True,
            "failed_checks": [],
            "warnings": [],
            "gates": {
                "live_execution_transition_parity": {"status": "ready", "score": 100.0},
                "auto_throttle_overtrading": {"status": "ready", "throttle_actions": []},
                "decision_replay_harness": {"status": "ready", "score": 94.0},
                "market_regime_stress_mode": {"status": "ready", "score": 91.0, "worst_slippage_bps": 42.0},
            },
            "sleeve_scorecards": [
                {
                    "profile": "default",
                    "status": "ready",
                    "execution_realism_score": 100.0,
                    "ending_net_pnl_total": 500.0,
                    "change_vs_previous_day": 75.0,
                    "executions": 1000,
                },
                {
                    "profile": "dividend",
                    "status": "watch",
                    "execution_realism_score": 78.0,
                    "ending_net_pnl_total": -25.0,
                    "change_vs_previous_day": -5.0,
                    "executions": 300,
                },
                {
                    "profile": "schwab_futures",
                    "status": "ready",
                    "execution_realism_score": 92.0,
                    "ending_net_pnl_total": 90.0,
                    "change_vs_previous_day": 30.0,
                    "executions": 400,
                },
            ],
            "paper_pnl_haircut_ledger": {
                "raw_week_pnl": 40.0,
                "realism_adjusted_week_pnl": 34.0,
            },
        },
        "promotion_quality": {"ok": True, "failed_checks": []},
        "live_readiness": {"ok": True, "overall_status": "ready", "hard_blocks": [], "submit_path_enabled": False},
        "paper_live_data_standard": {"ok": True, "overall_status": "ready"},
        "live_canary": {
            "overall_status": "ready",
            "recommended_mode": "supervised_canary",
            "supervised_canary_ready": True,
            "staged_preclearance_ready": False,
            "preapproved_supervised_ready": False,
            "blocking_reasons": [],
            "preclearance_score": 100.0,
            "target_canary_weight": 0.04,
            "applied_canary_weight": 0.04,
            "canary_weight_ok": True,
        },
        "paper_performance": {"ok": True, "week": {"top_profiles": [{"name": "default", "executions": 1000}]}},
        "account_position": {"ok": True, "account_count": 3, "position_count": 9, "covered_call_roll_watch": {"covered_call_count": 3, "alert_count": 0, "overall_status": "watch"}},
        "covered_call_roll_watch": {"ok": True, "overall_status": "watch", "covered_call_count": 3, "alert_count": 0},
        "execution_lab": {"ok": True, "overall_status": "ready", "scenario_count": 8},
        "counterfactual_replay": {"ok": True, "top_candidates": [{"profile": "default"}]},
        "paper_replay_drill": {"ok": True, "overall_status": "ready"},
        "strategy_attribution": {"ok": True, "aggregates": {"row_count": 100, "total_pnl_proxy": 12.5}},
        "operator_cockpit": {"ok": True, "overall_status": "ready"},
        "a_plus_operating_packet": {"ok": True, "overall_grade": "A+", "a_plus_ready": True},
        "ingestion_storage": {"ok": True, "overall_status": "ready", "backpressure": {"total_pending_lines": 500, "pending_lines_threshold": 15000, "oldest_pending_age_seconds": 10.0}},
        "storage_retention": {"ok": True, "overall_status": "ready"},
        "bot_quality_autopilot": {"ok": True, "overall_status": "ready"},
        "infrastructure_autofix": {"ok": True, "overall_status": "ready"},
        "system_self_model": {"ok": True, "overall_status": "ready"},
        "whole_system_intelligence": {"ok": True, "overall_status": "ready"},
    }
    if overrides:
        base.update(overrides)
    for name, payload in base.items():
        _write_json(root / control.SOURCE_FILES[name], payload)


def test_trading_desk_upgrade_control_builds_all_ten_lanes(tmp_path: Path) -> None:
    _write_sources(tmp_path)

    payload = control.build_payload(tmp_path)

    assert payload["ok"] is True
    assert payload["lane_count"] == 10
    assert {row["id"] for row in payload["lanes"]} == set(control.LANE_LABELS)
    assert payload["authority_boundary"]["live_trading_enabled_by_this_artifact"] is False
    assert payload["authority_boundary"]["real_capital_allocation_enabled_by_this_artifact"] is False
    assert payload["paper_to_live_acceptance"]["ready"] is True


def test_acceptance_harness_blocks_when_truth_layer_is_not_ready(tmp_path: Path) -> None:
    _write_sources(
        tmp_path,
        {
            "paper_execution_truth_layer": {
                "ok": False,
                "overall_status": "blocked",
                "score": 40.0,
                "grade": "F",
                "a_plus_ready": False,
                "failed_checks": ["decision_replay_harness"],
                "warnings": [],
                "gates": {},
                "sleeve_scorecards": [],
            }
        },
    )

    payload = control.build_payload(tmp_path)
    acceptance = next(row for row in payload["lanes"] if row["id"] == "paper_live_acceptance_harness")

    assert payload["ok"] is False
    assert acceptance["status"] == "blocked"
    assert "truth_layer_not_a_plus_ready" in acceptance["blockers"]


def test_advisory_capital_allocator_is_paper_only_and_caps_weight(tmp_path: Path) -> None:
    _write_sources(tmp_path)

    payload = control.build_payload(tmp_path)
    plan = payload["advisory_capital_plan"]
    top = plan["ranked_sleeves"][0]

    assert plan["enabled_for_live"] is False
    assert plan["advisory_only"] is True
    assert top["profile"] == "default"
    assert top["advisory_paper_weight"] <= plan["max_sleeve_weight"]


def test_storage_pressure_blocks_ingestion_discipline_lane(tmp_path: Path) -> None:
    _write_sources(
        tmp_path,
        {
            "ingestion_storage": {
                "ok": False,
                "overall_status": "degraded",
                "backpressure": {
                    "total_pending_lines": 20000,
                    "pending_lines_threshold": 15000,
                    "oldest_pending_age_seconds": 300.0,
                },
            }
        },
    )

    payload = control.build_payload(tmp_path)
    storage = next(row for row in payload["lanes"] if row["id"] == "storage_ingestion_discipline")

    assert storage["status"] == "blocked"
    assert "pending_lines_above_threshold" in storage["blockers"]


def test_weekend_no_attribution_rows_is_not_degradation_warning(tmp_path: Path) -> None:
    _write_sources(
        tmp_path,
        {
            "strategy_attribution": {
                "ok": False,
                "day": "20260613",
                "row_count": 0,
                "file_count": 0,
                "total_pnl_proxy": 0.0,
            }
        },
    )

    payload = control.build_payload(tmp_path)
    attribution = next(row for row in payload["lanes"] if row["id"] == "decision_quality_attribution")

    assert attribution["status"] == "ready"
    assert attribution["warnings"] == []
    assert attribution["evidence"]["attribution_coverage_state"] == "market_closed_no_rows"
    assert "decision_quality_attribution:strategy_attribution_has_no_rows" not in payload["warnings"]


def test_degraded_broad_cockpit_is_evidence_when_a_plus_packet_is_ready(tmp_path: Path) -> None:
    _write_sources(
        tmp_path,
        {
            "operator_cockpit": {"ok": False, "overall_status": "degraded"},
            "a_plus_operating_packet": {"ok": True, "overall_grade": "A+", "a_plus_ready": True},
        },
    )

    payload = control.build_payload(tmp_path)
    cockpit = next(row for row in payload["lanes"] if row["id"] == "operator_cockpit")

    assert cockpit["warnings"] == []
    assert cockpit["evidence"]["operator_cockpit_status"] == "degraded"
    assert "operator_cockpit:operator_cockpit_not_ready" not in payload["warnings"]
