import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.ops import incident_closeout_autopilot as src


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def test_incident_closeout_autopilot_blocks_when_review_and_runtime_clearance_are_open(tmp_path: Path) -> None:
    health = tmp_path / "governance" / "health"
    _write_json(health / "incident_timeline_latest.json", {"open_incident_count": 1})
    _write_json(health / "incident_review_packet_latest.json", {"review_required": True})
    _write_json(health / "live_runtime_separation_control_latest.json", {"clearance_plan": {"clearance_state": "awaiting_coverage_cycles"}})
    _write_json(health / "auth_lease_manager_latest.json", {"lease_state": "warning"})
    _write_json(health / "remote_alert_control_latest.json", {"critical_backlog": {"unacked_count": 1, "unsent_count": 0}})
    _write_json(health / "lane_thaw_controller_latest.json", {"paused_lane_count": 1, "blocked_count": 0})
    _write_json(health / "data_plane_recovery_controller_latest.json", {"write_failure_count": 2, "account_snapshot_failure_count": 0})
    _write_json(health / "process_watchdog_latest.json", {"restart_storms": [{"name": "ops"}], "alerts": []})

    payload = src.build_payload(tmp_path)

    assert payload["overall_status"] == "blocked"
    assert payload["closeout_ready"] is False
    assert "incident_review_packet_latest.json" in payload["required_artifacts"]
    assert any(row["surface"] == "runtime_clearance" for row in payload["blocking_surfaces"])


def test_incident_closeout_autopilot_is_ready_when_all_surfaces_are_clear(tmp_path: Path) -> None:
    health = tmp_path / "governance" / "health"
    _write_json(health / "incident_timeline_latest.json", {"open_incident_count": 0, "auto_close_contract": {"closure_ready": True}})
    _write_json(health / "incident_review_packet_latest.json", {"review_required": False, "closure_contract": {"closure_ready": True}})
    _write_json(health / "live_runtime_separation_control_latest.json", {"clearance_plan": {"clearance_state": "cleared"}})
    _write_json(health / "auth_lease_manager_latest.json", {"lease_state": "healthy"})
    _write_json(health / "remote_alert_control_latest.json", {"critical_backlog": {"unacked_count": 0, "unsent_count": 0}})
    _write_json(health / "lane_thaw_controller_latest.json", {"paused_lane_count": 0, "blocked_count": 0})
    _write_json(health / "data_plane_recovery_controller_latest.json", {"write_failure_count": 0, "account_snapshot_failure_count": 0})
    _write_json(health / "process_watchdog_latest.json", {"restart_storms": [], "alerts": []})

    payload = src.build_payload(tmp_path)

    assert payload["overall_status"] == "ready"
    assert payload["closeout_ready"] is True
    assert payload["blocking_surfaces"] == []


def test_incident_closeout_autopilot_softens_data_plane_blocker_when_recovery_is_bounded(tmp_path: Path) -> None:
    health = tmp_path / "governance" / "health"
    _write_json(health / "incident_timeline_latest.json", {"open_incident_count": 0})
    _write_json(health / "incident_review_packet_latest.json", {"review_required": False})
    _write_json(health / "live_runtime_separation_control_latest.json", {"clearance_plan": {"clearance_state": "cleared"}})
    _write_json(health / "auth_lease_manager_latest.json", {"lease_state": "healthy"})
    _write_json(health / "remote_alert_control_latest.json", {"critical_backlog": {"unacked_count": 0, "unsent_count": 0}})
    _write_json(health / "lane_thaw_controller_latest.json", {"paused_lane_count": 0, "blocked_count": 0})
    _write_json(
        health / "data_plane_recovery_controller_latest.json",
        {
            "overall_status": "degraded",
            "recovery_state": "recovering_under_guard",
            "write_failure_count": 2,
            "account_snapshot_failure_count": 1,
            "external_backlog_status": "drain_active",
            "writer_handoff_contract": {"writer_service_active": True},
            "backlog_recovery_contract": {"follow_through_status": "handoff_requested", "drain_progress_lines": 25},
        },
    )
    _write_json(health / "process_watchdog_latest.json", {"restart_storms": [], "alerts": []})

    payload = src.build_payload(tmp_path)

    assert payload["overall_status"] == "ready"
    assert payload["bounded_data_plane_recovery"] is True
    assert payload["bounded_warning_closeout_ready"] is True
    assert payload["closeout_ready"] is True
    assert any(row["surface"] == "data_plane_recovery" and row["severity"] == "warning" for row in payload["blocking_surfaces"])


def test_incident_closeout_autopilot_degrades_when_runtime_and_review_are_recoverable(tmp_path: Path) -> None:
    health = tmp_path / "governance" / "health"
    _write_json(health / "incident_timeline_latest.json", {"open_incident_count": 2})
    _write_json(
        health / "incident_review_packet_latest.json",
        {
            "overall_status": "degraded",
            "review_required": True,
            "closure_contract": {"closure_ready": False, "closure_reason": "open_surfaces_present_or_no_recent_incidents"},
        },
    )
    _write_json(
        health / "live_runtime_separation_control_latest.json",
        {"overall_status": "degraded", "clearance_plan": {"clearance_state": "awaiting_cold_lane"}},
    )
    _write_json(health / "auth_lease_manager_latest.json", {"lease_state": "warning"})
    _write_json(health / "remote_alert_control_latest.json", {"critical_backlog": {"unacked_count": 1, "unsent_count": 0}})
    _write_json(health / "lane_thaw_controller_latest.json", {"paused_lane_count": 0, "blocked_count": 0})
    _write_json(
        health / "data_plane_recovery_controller_latest.json",
        {
            "overall_status": "degraded",
            "recovery_state": "recovering_under_guard",
            "write_failure_count": 1,
            "account_snapshot_failure_count": 0,
            "external_backlog_status": "drain_active",
            "writer_handoff_contract": {"writer_service_active": True},
            "backlog_recovery_contract": {"follow_through_status": "handoff_requested", "drain_progress_lines": 10},
        },
    )
    _write_json(health / "process_watchdog_latest.json", {"restart_storms": [], "alerts": []})

    payload = src.build_payload(tmp_path)

    assert payload["overall_status"] == "degraded"
    assert payload["recoverable_runtime_clearance"] is True
    assert payload["recoverable_review_gate"] is True
    assert payload["bounded_incident_backlog"] is True
    assert payload["bounded_closeout_path_ready"] is True
    assert any(row["surface"] == "runtime_clearance" and row["severity"] == "warning" for row in payload["blocking_surfaces"])
    assert any(row["surface"] == "incident_review" and row["severity"] == "warning" for row in payload["blocking_surfaces"])


def test_incident_closeout_autopilot_bounds_blocked_review_when_no_open_surfaces_remain(tmp_path: Path) -> None:
    health = tmp_path / "governance" / "health"
    _write_json(
        health / "incident_timeline_latest.json",
        {
            "overall_status": "blocked",
            "open_incident_count": 2,
            "auto_close_contract": {
                "closure_ready": False,
                "candidate_count": 0,
                "review_required": True,
                "closure_reason": "open_surfaces_present",
            },
            "watch_surfaces": [],
        },
    )
    _write_json(
        health / "incident_review_packet_latest.json",
        {
            "overall_status": "blocked",
            "review_required": True,
            "open_incident_count": 2,
            "open_surfaces": [],
            "closure_contract": {
                "closure_ready": False,
                "candidate_count": 0,
                "review_required": True,
                "closure_reason": "open_surfaces_present",
            },
        },
    )
    _write_json(health / "live_runtime_separation_control_latest.json", {"overall_status": "ready", "clearance_plan": {"clearance_state": "ready"}})
    _write_json(health / "auth_lease_manager_latest.json", {"lease_state": "healthy"})
    _write_json(health / "remote_alert_control_latest.json", {"critical_backlog": {"unacked_count": 0, "unsent_count": 0}})
    _write_json(health / "lane_thaw_controller_latest.json", {"paused_lane_count": 0, "blocked_count": 0})
    _write_json(health / "data_plane_recovery_controller_latest.json", {"write_failure_count": 0, "account_snapshot_failure_count": 0})
    _write_json(health / "process_watchdog_latest.json", {"restart_storms": [], "alerts": []})

    payload = src.build_payload(tmp_path)

    assert payload["overall_status"] == "degraded"
    assert payload["recoverable_review_gate"] is True
    assert payload["bounded_incident_backlog"] is True
    assert payload["bounded_closeout_path_ready"] is True
    assert payload["closeout_ready"] is False
    assert any(row["surface"] == "incident_review" and row["severity"] == "warning" for row in payload["blocking_surfaces"])


def test_incident_closeout_autopilot_accepts_coverage_cycles_ready_as_bounded_runtime(tmp_path: Path) -> None:
    health = tmp_path / "governance" / "health"
    _write_json(health / "incident_timeline_latest.json", {"open_incident_count": 0, "auto_close_contract": {"closure_ready": True}})
    _write_json(health / "incident_review_packet_latest.json", {"review_required": False, "closure_contract": {"closure_ready": True}})
    _write_json(
        health / "live_runtime_separation_control_latest.json",
        {"overall_status": "degraded", "clearance_plan": {"clearance_state": "coverage_cycles_ready"}},
    )
    _write_json(
        health / "auth_lease_manager_latest.json",
        {
            "lease_state": "warning",
            "broker_state": {"broker_ready": True, "auth_ok": True, "configured_for_refresh": True},
            "lease_budget": {"expires_in_seconds": 900, "critical_lease_seconds": 300},
        },
    )
    _write_json(health / "remote_alert_control_latest.json", {"critical_backlog": {"unacked_count": 0, "unsent_count": 0}})
    _write_json(health / "lane_thaw_controller_latest.json", {"paused_lane_count": 0, "blocked_count": 0})
    _write_json(
        health / "data_plane_recovery_controller_latest.json",
        {
            "overall_status": "degraded",
            "recovery_state": "recovering_under_guard",
            "write_failure_count": 0,
            "account_snapshot_failure_count": 1,
            "external_backlog_status": "drain_active",
            "writer_handoff_contract": {"writer_service_active": True},
            "backlog_recovery_contract": {"follow_through_status": "handoff_requested", "drain_progress_lines": 12},
        },
    )
    _write_json(health / "process_watchdog_latest.json", {"restart_storms": [], "alerts": []})

    payload = src.build_payload(tmp_path)

    assert payload["overall_status"] == "ready"
    assert payload["recoverable_runtime_clearance"] is True
    assert payload["bounded_auth_lease"] is True
    assert payload["bounded_warning_closeout_ready"] is True
    assert payload["closeout_ready"] is True
    assert payload["closeout_score"] >= 90.0


def test_incident_closeout_autopilot_accepts_guarded_read_only_for_paper_soak(tmp_path: Path) -> None:
    health = tmp_path / "governance" / "health"
    _write_json(health / "incident_timeline_latest.json", {"open_incident_count": 0, "auto_close_contract": {"closure_ready": True}})
    _write_json(health / "incident_review_packet_latest.json", {"review_required": False, "closure_contract": {"closure_ready": True}})
    _write_json(
        health / "live_runtime_separation_control_latest.json",
        {"overall_status": "ready", "clearance_plan": {"clearance_state": "guarded_live_read_only"}},
    )
    _write_json(health / "auth_lease_manager_latest.json", {"lease_state": "healthy"})
    _write_json(health / "remote_alert_control_latest.json", {"critical_backlog": {"unacked_count": 0, "unsent_count": 0}})
    _write_json(health / "lane_thaw_controller_latest.json", {"paused_lane_count": 0, "blocked_count": 0})
    _write_json(health / "data_plane_recovery_controller_latest.json", {"write_failure_count": 0, "account_snapshot_failure_count": 0})
    _write_json(health / "process_watchdog_latest.json", {"restart_storms": [], "alerts": []})

    payload = src.build_payload(tmp_path)

    assert payload["overall_status"] == "ready"
    assert payload["closeout_ready"] is True
    assert payload["guarded_read_only_runtime"] is True
    assert payload["recoverable_runtime_clearance"] is True
    assert any(row["surface"] == "runtime_clearance" and row["severity"] == "warning" for row in payload["blocking_surfaces"])


def test_incident_closeout_autopilot_accepts_managed_coverage_stage_deferred_for_paper_soak(tmp_path: Path) -> None:
    health = tmp_path / "governance" / "health"
    _write_json(health / "incident_timeline_latest.json", {"open_incident_count": 0, "auto_close_contract": {"closure_ready": True}})
    _write_json(health / "incident_review_packet_latest.json", {"review_required": False, "closure_contract": {"closure_ready": True}})
    _write_json(
        health / "live_runtime_separation_control_latest.json",
        {
            "overall_status": "ready",
            "clearance_plan": {"clearance_state": "managed_coverage_stage_deferred"},
            "live_plane": {"ready": True, "live_lane_running": True},
        },
    )
    _write_json(health / "auth_lease_manager_latest.json", {"lease_state": "healthy"})
    _write_json(health / "remote_alert_control_latest.json", {"critical_backlog": {"unacked_count": 0, "unsent_count": 0}})
    _write_json(health / "lane_thaw_controller_latest.json", {"paused_lane_count": 0, "blocked_count": 0})
    _write_json(health / "data_plane_recovery_controller_latest.json", {"write_failure_count": 0, "account_snapshot_failure_count": 0})
    _write_json(health / "process_watchdog_latest.json", {"restart_storms": [], "alerts": []})

    payload = src.build_payload(tmp_path)

    assert payload["overall_status"] == "ready"
    assert payload["closeout_ready"] is True
    assert payload["managed_coverage_stage_deferred_runtime"] is True
    assert payload["recoverable_runtime_clearance"] is True
    assert any(
        row["surface"] == "runtime_clearance"
        and row["severity"] == "warning"
        and "coverage repair is deferred" in row["summary"]
        for row in payload["blocking_surfaces"]
    )


def test_incident_closeout_autopilot_softens_process_watchdog_when_timeline_marks_storage_backpressure_watch(tmp_path: Path) -> None:
    health = tmp_path / "governance" / "health"
    _write_json(
        health / "incident_timeline_latest.json",
        {
            "open_incident_count": 0,
            "auto_close_contract": {"closure_ready": True},
            "watch_surfaces": [
                {
                    "surface": "process_watchdog",
                    "watch_reason": "derived_storage_backpressure",
                }
            ],
        },
    )
    _write_json(health / "incident_review_packet_latest.json", {"review_required": False, "closure_contract": {"closure_ready": True}})
    _write_json(
        health / "live_runtime_separation_control_latest.json",
        {"overall_status": "degraded", "clearance_plan": {"clearance_state": "coverage_cycles_ready"}},
    )
    _write_json(
        health / "auth_lease_manager_latest.json",
        {
            "lease_state": "warning",
            "broker_state": {"broker_ready": True, "auth_ok": True, "configured_for_refresh": True},
            "lease_budget": {"expires_in_seconds": 900, "critical_lease_seconds": 300},
        },
    )
    _write_json(health / "remote_alert_control_latest.json", {"critical_backlog": {"unacked_count": 0, "unsent_count": 0}})
    _write_json(health / "lane_thaw_controller_latest.json", {"paused_lane_count": 0, "blocked_count": 0})
    _write_json(
        health / "data_plane_recovery_controller_latest.json",
        {
            "overall_status": "degraded",
            "recovery_state": "recovering_under_guard",
            "write_failure_count": 0,
            "account_snapshot_failure_count": 1,
            "external_backlog_status": "drain_active",
            "writer_handoff_contract": {"writer_service_active": True},
            "backlog_recovery_contract": {"follow_through_status": "handoff_requested", "drain_progress_lines": 12},
        },
    )
    _write_json(health / "process_watchdog_latest.json", {"restart_storms": [{"name": "execution_lane_paper"}], "alerts": [{"name": "execution_lane_paper"}]})

    payload = src.build_payload(tmp_path)

    assert payload["overall_status"] == "ready"
    assert any(row["surface"] == "process_watchdog" and row["severity"] == "warning" for row in payload["blocking_surfaces"])
    assert payload["closeout_score"] >= 90.0


def test_incident_closeout_autopilot_softens_isolated_read_only_watchdog_debt(tmp_path: Path) -> None:
    health = tmp_path / "governance" / "health"
    _write_json(health / "incident_timeline_latest.json", {"open_incident_count": 0, "auto_close_contract": {"closure_ready": True}})
    _write_json(health / "incident_review_packet_latest.json", {"review_required": False, "closure_contract": {"closure_ready": True}})
    _write_json(health / "live_runtime_separation_control_latest.json", {"overall_status": "ready", "clearance_plan": {"clearance_state": "cleared"}})
    _write_json(
        health / "auth_lease_manager_latest.json",
        {
            "lease_state": "warning",
            "broker_state": {"broker_ready": True, "auth_ok": True, "configured_for_refresh": True},
            "lease_budget": {"expires_in_seconds": 900, "critical_lease_seconds": 300},
        },
    )
    _write_json(health / "remote_alert_control_latest.json", {"critical_backlog": {"unacked_count": 0, "unsent_count": 0}})
    _write_json(health / "lane_thaw_controller_latest.json", {"paused_lane_count": 0, "blocked_count": 0})
    _write_json(health / "data_plane_recovery_controller_latest.json", {"write_failure_count": 0, "account_snapshot_failure_count": 0})
    _write_json(
        health / "process_watchdog_latest.json",
        {
            "restart_storms": [
                {
                    "name": "coinbase_loop",
                    "impact": "read_only_collection",
                    "quarantinable": True,
                    "blocks_execution_clear": False,
                }
            ],
            "restart_storm_isolation": {
                "isolated_count": 1,
                "execution_blocking_count": 0,
                "all_active_storms_isolated": True,
            },
            "alerts": [{"name": "coinbase_loop", "type": "restart_storm"}],
        },
    )

    payload = src.build_payload(tmp_path)
    watchdog = next(row for row in payload["blocking_surfaces"] if row["surface"] == "process_watchdog")

    assert payload["overall_status"] == "ready"
    assert payload["isolated_read_only_watchdog"] is True
    assert watchdog["severity"] == "warning"
    assert "read-only collector restart debt is isolated" in watchdog["summary"]
