import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.ops import data_plane_recovery_controller as data_plane_src
from scripts.ops import lane_thaw_controller as thaw_src


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def test_lane_thaw_controller_marks_low_error_lane_as_candidate(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    health = project_root / "governance" / "health"
    _write_json(
        health / "data_ingress_latest_intraday_aggressive_equities_schwab_defensive.json",
        {
            "loop_state": "paused_anomaly_killswitch",
            "pause_gate": "anomaly_killswitch",
            "timestamp_utc": "2026-04-21T12:00:00+00:00",
            "iter_error_rate": 0.0,
            "iter_error_count": 0,
            "total_counts": {"api_error": 25},
        },
    )
    _write_json(health / "auth_lease_manager_latest.json", {"lease_state": "healthy"})
    _write_json(
        health / "live_runtime_separation_control_latest.json",
        {"clearance_plan": {"clearance_state": "coverage_cycles_ready"}},
    )
    _write_json(project_root / "governance" / "walk_forward" / "coverage_gap_closer_latest.json", {"autopilot_contract": {"can_launch_now": True}})
    _write_json(health / "process_watchdog_latest.json", {"restart_storms": []})

    payload = thaw_src.build_payload(project_root)

    assert payload["overall_status"] == "degraded"
    assert payload["candidate_count"] == 1
    assert payload["lanes"][0]["thaw_state"] == "candidate"
    assert payload["lanes"][0]["cooldown_contract"]["state"] == "elapsed"


def test_data_plane_recovery_controller_flags_write_failures_and_snapshot_failures(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    health = project_root / "governance" / "health"
    _write_json(
        health / "incident_timeline_latest.json",
        {
            "recent_incidents": [
                {"summary": "write_failure"},
                {"summary": "write_failure"},
                {"summary": "get_accounts_snapshot"},
            ]
        },
    )
    _write_json(health / "external_backlog_drain_latest.json", {"overall_status": "degraded"})
    _write_json(health / "ingestion_priority_queue_latest.json", {"queue_depth": 120})
    _write_json(health / "storage_tier_policy_latest.json", {"pressure": {"hot_path_over_budget_bytes": 0}})
    _write_json(health / "live_runtime_separation_control_latest.json", {"clearance_plan": {"clearance_state": "protect_live"}})

    payload = data_plane_src.build_payload(project_root)

    assert payload["overall_status"] == "degraded"
    assert payload["write_failure_count"] == 2
    assert payload["account_snapshot_failure_count"] == 1
    assert payload["recovery_contract"]["backlog_drain_required"] is True
    assert payload["recovery_contract"]["snapshot_probe_required"] is True


def test_data_plane_recovery_controller_marks_guarded_recovery_when_writer_handoff_is_active(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    health = project_root / "governance" / "health"
    _write_json(
        health / "incident_timeline_latest.json",
        {
            "recent_incidents": [
                {"summary": "write_failure"},
                {"summary": "get_accounts_snapshot"},
            ]
        },
    )
    _write_json(
        health / "external_backlog_drain_latest.json",
        {
            "overall_status": "blocked",
            "apply_requested": True,
            "blocked_reasons": ["market_hours_guard"],
            "drain_delta": {"total_pending_lines": 120},
            "follow_through": {"status": "not_needed"},
            "off_hours_window": {"active": False},
        },
    )
    _write_json(health / "ingestion_priority_queue_latest.json", {"queue_depth": 450})
    _write_json(health / "storage_tier_policy_latest.json", {"pressure": {"hot_path_over_budget_bytes": 100}})
    _write_json(health / "live_runtime_separation_control_latest.json", {"clearance_plan": {"clearance_state": "protect_live"}})
    _write_json(health / "sql_link_service_progress_latest.json", {"status": "running", "current_step": "shard_linking"})
    _write_json(health / "broker_truth_shared_snapshot_schwab_latest.json", {"fetched": True, "timestamp_utc": "2026-04-21T18:57:22+00:00"})

    payload = data_plane_src.build_payload(project_root)

    assert payload["overall_status"] == "degraded"
    assert payload["recovery_state"] == "recovering_under_guard"
    assert payload["backlog_recovery_contract"]["market_hours_guard"] is True
    assert payload["writer_handoff_contract"]["writer_service_active"] is True


def test_lane_thaw_controller_holds_lane_during_active_cooldown(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    health = project_root / "governance" / "health"
    _write_json(
        health / "data_ingress_latest_swing_aggressive_equities_schwab_core_volatile.json",
        {
            "loop_state": "paused_anomaly_killswitch",
            "pause_gate": "anomaly_killswitch",
            "timestamp_utc": "2099-04-21T12:00:00+00:00",
            "iter_error_rate": 0.0,
            "iter_error_count": 0,
            "total_counts": {"api_error": 25},
        },
    )
    _write_json(health / "auth_lease_manager_latest.json", {"lease_state": "healthy"})
    _write_json(
        health / "live_runtime_separation_control_latest.json",
        {"clearance_plan": {"clearance_state": "coverage_cycles_ready"}},
    )
    _write_json(project_root / "governance" / "walk_forward" / "coverage_gap_closer_latest.json", {"autopilot_contract": {"can_launch_now": True}})
    _write_json(health / "process_watchdog_latest.json", {"restart_storms": []})

    payload = thaw_src.build_payload(project_root)

    assert payload["lanes"][0]["thaw_state"] == "hold"
    assert "cooldown_active" in payload["lanes"][0]["reasons"]
    assert payload["lanes"][0]["cooldown_contract"]["state"] == "active"


def test_lane_thaw_controller_tracks_repeat_trip_history_and_escalates_chronic_lane(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    health = project_root / "governance" / "health"
    now = datetime.now(timezone.utc)
    trip_a = (now - timedelta(hours=18)).isoformat()
    trip_b = (now - timedelta(hours=4)).isoformat()
    current_trip = (now - timedelta(minutes=20)).isoformat()

    _write_json(
        health / "lane_thaw_controller_latest.json",
        {
            "cooldown_history": {
                "history_window_hours": 72,
                "watch_trip_threshold": 2,
                "chronic_trip_threshold": 3,
                "lane_history": {
                    "swing_aggressive_equities_schwab_core_volatile": {
                        "active_trip": False,
                        "trip_count_total": 2,
                        "recent_trip_starts_utc": [trip_a, trip_b],
                        "last_trip_started_utc": trip_b,
                        "last_trip_recovered_utc": (now - timedelta(hours=2)).isoformat(),
                        "last_seen_state": "clear",
                    }
                },
            }
        },
    )
    _write_json(
        health / "data_ingress_latest_swing_aggressive_equities_schwab_core_volatile.json",
        {
            "loop_state": "paused_anomaly_killswitch",
            "pause_gate": "anomaly_killswitch",
            "timestamp_utc": current_trip,
            "iter_error_rate": 0.0,
            "iter_error_count": 0,
            "total_counts": {"api_error": 25},
        },
    )
    _write_json(health / "auth_lease_manager_latest.json", {"lease_state": "healthy"})
    _write_json(
        health / "live_runtime_separation_control_latest.json",
        {"clearance_plan": {"clearance_state": "coverage_cycles_ready"}},
    )
    _write_json(project_root / "governance" / "walk_forward" / "coverage_gap_closer_latest.json", {"autopilot_contract": {"can_launch_now": True}})
    _write_json(health / "process_watchdog_latest.json", {"restart_storms": []})

    payload = thaw_src.build_payload(project_root)

    assert payload["cooldown_history"]["chronic_offender_count"] == 1
    assert payload["cooldown_history"]["new_trip_count"] == 1
    lane = payload["lanes"][0]
    assert lane["trip_history"]["trip_count_total"] == 3
    assert lane["trip_history"]["trip_count_window"] == 3
    assert lane["trip_history"]["chronic_offender"] is True
    assert lane["escalation_contract"]["level"] == "chronic"
    assert lane["escalation_contract"]["operator_review_required"] is True
    assert "chronic_offender_review_required" in lane["reasons"]
    assert "chronic_trip_cooldown" in lane["cooldown_contract"]["reasons"]
