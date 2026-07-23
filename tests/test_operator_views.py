import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.ops import sleeve_profitability_dashboard as sleeve_src
from scripts.ops import system_done_for_today as done_src


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def test_sleeve_profitability_dashboard_summarizes_totals_and_harvest_attention(tmp_path: Path) -> None:
    health = tmp_path / "governance" / "health"
    _write_json(
        health / "paper_performance_latest.json",
        {
            "ok": True,
            "day": {"day_utc": "20260526"},
            "sleeve_latest": [
                {
                    "profile": "bond",
                    "data_status": "current",
                    "executions": 50,
                    "ending_realized_pnl_total": 10.0,
                    "ending_unrealized_pnl_total": 100.0,
                    "ending_net_pnl_total": 110.0,
                    "winning_strategy_count": 2,
                    "losing_strategy_count": 0,
                },
                {
                    "profile": "aggressive",
                    "data_status": "current",
                    "executions": 20,
                    "ending_realized_pnl_total": 5.0,
                    "ending_unrealized_pnl_total": -50.0,
                    "ending_net_pnl_total": -45.0,
                    "winning_strategy_count": 0,
                    "losing_strategy_count": 2,
                },
            ],
        },
    )
    _write_json(
        health / "paper_profitability_control_latest.json",
        {
            "overall_status": "protective_tightening",
            "profitability_grade": "B",
            "active_profile_controls": {
                "aggressive": {
                    "action": "quarantine_new_entries",
                    "control_posture_grade": "A+",
                    "a_plus_plus_strengthening": {"active": True, "control_grade": "A+"},
                }
            },
        },
    )

    payload = sleeve_src.build_payload(tmp_path)

    assert payload["overall_status"] == "ready"
    assert payload["totals"]["net_pnl_total"] == 65.0
    assert payload["top_sleeves"][0]["profile"] == "bond"
    assert payload["harvest_attention"][0]["profile"] == "bond"
    assert payload["weak_sleeve_count"] == 1
    assert payload["weak_sleeve_control_a_plus_plus_count"] == 1
    assert payload["bottom_sleeves"][0]["raw_grade"] == "C-"
    assert payload["bottom_sleeves"][0]["control_grade"] == "A+"
    assert payload["bottom_sleeves"][0]["display_grade"] == "A+"


def test_system_done_for_today_reports_stop_chasing_when_core_green(tmp_path: Path) -> None:
    health = tmp_path / "governance" / "health"
    _write_json(
        health / "health_fast_latest.json",
        {
            "storage": {
                "backpressure": {
                    "total_pending_lines": 100,
                    "pending_lines_threshold": 15000,
                    "oldest_pending_age_seconds": 0.0,
                    "oldest_age_threshold_seconds": 240.0,
                }
            },
            "runtime_pressure": {"overall_status": "advisory", "host_saturation_score": 42.0},
            "memory": {"overall_status": "ready"},
        },
    )
    _write_json(
        health / "writer_cycle_coordinator_latest.json",
        {"writer_state_before": {"active": False, "current_step": "complete", "completed_shard_count": 26, "planned_shard_count": 26}},
    )
    _write_json(health / "runtime_throttle_control_latest.json", {"overall_status": "advisory", "host_saturation_score": 42.0})
    _write_json(health / "paper_performance_latest.json", {"ok": True, "day": {"day_utc": "20260526", "available": True, "ending_net_pnl_total": 10.0}})
    _write_json(health / "paper_profitability_control_latest.json", {"overall_status": "ready", "profitability_grade": "A"})
    _write_json(health / "watchdog_intelligence_latest.json", {"overall_status": "ready", "grade": "A", "score": 99.0})
    _write_json(
        health / "bot_needs_intelligence_latest.json",
        {
            "training_candidate_selector": {"selected_count": 0},
            "zero_observation_repair_contract": {"active": False, "zero_observation_count": 0},
        },
    )

    payload = done_src.build_payload(tmp_path)

    assert payload["overall_status"] == "done_for_today"
    assert payload["can_stop_chasing"] is True
    assert payload["blockers"] == []
