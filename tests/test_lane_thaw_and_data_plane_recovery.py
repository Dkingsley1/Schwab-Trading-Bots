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


def test_lane_thaw_controller_blocks_lane_when_systemic_guardrails_are_hot(tmp_path: Path) -> None:
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
    _write_json(health / "data_plane_recovery_controller_latest.json", {"write_failure_count": 1, "account_snapshot_failure_count": 1, "queue_depth": 2500})
    _write_json(health / "global_killswitch_latest.json", {"halt": True})
    _write_json(health / "incident_timeline_latest.json", {"summary": {"risk_halt_events": 1}})

    payload = thaw_src.build_payload(project_root)

    assert payload["overall_status"] == "blocked"
    lane = payload["lanes"][0]
    assert lane["thaw_state"] == "blocked"
    assert "global_killswitch_active" in lane["reasons"]
    assert "incident_risk_halt_active" in lane["reasons"]
    assert "account_snapshot_recovery_pending" in lane["reasons"]
    assert lane["guardrail_quorum"]["hard_blocked"] is True


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
    assert payload["recovery_contract"]["snapshot_probe_command"] == [
        "./scripts/ops/opsctl.sh",
        "token-refresh",
        "--json",
    ]


def test_data_plane_recovery_controller_treats_old_write_failures_as_recovered_after_steady_storage(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    health = project_root / "governance" / "health"
    _write_json(
        health / "incident_timeline_latest.json",
        {
            "recent_incidents": [
                {"summary": "write_failure", "timestamp_utc": "2026-06-05T13:47:22+00:00"},
                {"summary": "write_failure", "timestamp_utc": "2026-06-05T13:47:23+00:00"},
            ]
        },
    )
    _write_json(health / "external_backlog_drain_latest.json", {"overall_status": "blocked", "blocked_reasons": ["market_hours_guard"]})
    _write_json(health / "ingestion_priority_queue_latest.json", {"lane_counts": {"core": {"pending_lines": 374}}})
    _write_json(health / "storage_tier_policy_latest.json", {"pressure": {"hot_path_over_budget_bytes": 0}})
    _write_json(
        health / "ingestion_storage_control_latest.json",
        {
            "overall_status": "ready",
            "severity": "stable",
            "backpressure_quality_score": 100,
            "recovery_quality_score": 96,
            "steady_state": {"target_status": {"steady_state_ready": True}},
            "external_route_verification": {"verification_state": "active_local_ready"},
        },
    )
    _write_json(health / "sql_link_service_progress_latest.json", {"status": "ok", "current_step": "complete"})
    _write_json(health / "live_runtime_separation_control_latest.json", {"clearance_plan": {"clearance_state": "ready"}})

    payload = data_plane_src.build_payload(project_root)

    assert payload["overall_status"] == "ready"
    assert payload["recovery_state"] == "stable"
    assert payload["raw_write_failure_count"] == 2
    assert payload["write_failure_count"] == 0
    assert payload["write_path_recovered_by_storage"] is True
    assert payload["storage_steady_state_ready"] is True


def test_data_plane_recovery_controller_recovers_write_failures_from_current_storage_truth(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    health = project_root / "governance" / "health"
    _write_json(
        health / "incident_timeline_latest.json",
        {
            "recent_incidents": [
                {"summary": "write_failure", "timestamp_utc": f"2026-06-05T13:47:2{i}+00:00"}
                for i in range(6)
            ]
        },
    )
    _write_json(health / "external_backlog_drain_latest.json", {"overall_status": "blocked", "blocked_reasons": ["external_storage_unavailable"]})
    _write_json(health / "ingestion_priority_queue_latest.json", {"lane_counts": {"core": {"pending_lines": 2172}}})
    _write_json(health / "storage_tier_policy_latest.json", {"pressure": {"hot_path_over_budget_bytes": 319843803301}})
    _write_json(
        health / "ingestion_storage_control_latest.json",
        {
            "overall_status": "ready",
            "severity": "stable",
            "backpressure_quality_score": 100,
            "recovery_quality_score": 82,
            "steady_state": {"target_status": {"steady_state_ready": True}},
            "external_route_verification": {"verification_state": "active_local_ready"},
            "backpressure": {
                "raw_live": {
                    "core_pending_lines": 2059,
                    "total_pending_lines": 2172,
                    "oldest_pending_age_seconds": 23.5,
                }
            },
            "data_integrity": {"sql_overlay_ops_write_failures": 0},
        },
    )
    _write_json(health / "sql_link_service_progress_latest.json", {"status": "running", "current_step": "shard_linking"})
    _write_json(health / "live_runtime_separation_control_latest.json", {"clearance_plan": {"clearance_state": "awaiting_coverage_cycles"}})

    payload = data_plane_src.build_payload(project_root)

    assert payload["overall_status"] == "ready"
    assert payload["recovery_state"] == "stable"
    assert payload["raw_write_failure_count"] == 6
    assert payload["write_failure_count"] == 0
    assert payload["storage_steady_state_ready"] is False
    assert payload["current_storage_write_ready"] is True
    assert payload["write_path_recovered_by_storage"] is True
    assert payload["hot_path_over_budget_bytes"] == 0
    assert payload["raw_hot_path_over_budget_bytes"] == 319843803301


def test_data_plane_recovery_controller_uses_bounded_storage_pressure_relief(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    health = project_root / "governance" / "health"
    _write_json(
        health / "incident_timeline_latest.json",
        {
            "recent_incidents": [
                {"summary": "write_failure", "timestamp_utc": f"2026-06-05T13:48:2{i}+00:00"}
                for i in range(6)
            ]
        },
    )
    _write_json(health / "external_backlog_drain_latest.json", {"overall_status": "blocked", "blocked_reasons": ["external_storage_unavailable"]})
    _write_json(health / "ingestion_priority_queue_latest.json", {"lane_counts": {"core": {"pending_lines": 1938}}})
    _write_json(health / "storage_tier_policy_latest.json", {"pressure": {"hot_path_over_budget_bytes": 319843803301}})
    _write_json(
        health / "ingestion_storage_control_latest.json",
        {
            "overall_status": "ready",
            "severity": "stable",
            "pressure_index": 0.334,
            "backpressure_quality_score": 97,
            "recovery_quality_score": 82,
            "steady_state": {"target_status": {"steady_state_ready": False, "target_breaches": ["pressure_index"]}},
            "external_route_verification": {"verification_state": "active_local_ready"},
            "backpressure": {
                "raw_live": {
                    "core_pending_lines": 2059,
                    "total_pending_lines": 2172,
                    "oldest_pending_age_seconds": 23.5,
                }
            },
            "data_integrity": {"sql_overlay_ops_write_failures": 0},
        },
    )
    _write_json(health / "sql_link_service_progress_latest.json", {"status": "ok", "current_step": "complete"})
    _write_json(health / "live_runtime_separation_control_latest.json", {"clearance_plan": {"clearance_state": "awaiting_coverage_cycles"}})

    payload = data_plane_src.build_payload(project_root)

    assert payload["overall_status"] == "ready"
    assert payload["raw_write_failure_count"] == 6
    assert payload["write_failure_count"] == 0
    assert payload["storage_steady_state_ready"] is False
    assert payload["current_storage_write_ready"] is True
    assert payload["write_path_recovery_evidence"]["bounded_target_relief"] is True
    assert payload["write_path_recovered_by_storage"] is True
    assert payload["hot_path_over_budget_bytes"] == 0


def test_data_plane_recovery_controller_recovers_during_overlay_only_storage_cleanup(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    health = project_root / "governance" / "health"
    _write_json(
        health / "incident_timeline_latest.json",
        {"recent_incidents": [{"summary": "write_failure", "timestamp_utc": "2026-06-05T13:48:20+00:00"} for _ in range(6)]},
    )
    _write_json(health / "external_backlog_drain_latest.json", {"overall_status": "blocked", "blocked_reasons": ["external_storage_unavailable"]})
    _write_json(health / "ingestion_priority_queue_latest.json", {"lane_counts": {"core": {"pending_lines": 1938}}})
    _write_json(health / "storage_tier_policy_latest.json", {"pressure": {"hot_path_over_budget_bytes": 319843803301}})
    _write_json(
        health / "ingestion_storage_control_latest.json",
        {
            "overall_status": "blocked",
            "severity": "critical",
            "pressure_index": 22.88,
            "backpressure_quality_score": 40,
            "steady_state": {"target_status": {"steady_state_ready": False, "target_breaches": ["pressure_index"]}},
            "external_route_verification": {"verification_state": "active_local_ready"},
            "backpressure": {
                "core_pending_lines": 2577,
                "total_pending_lines": 2717,
                "overlay_adjusted": True,
                "oldest_pending_age_seconds": 5491.09,
                "raw_live": {
                    "core_pending_lines": 2059,
                    "total_pending_lines": 2172,
                    "oldest_pending_age_seconds": 23.5,
                },
            },
            "data_integrity": {"sql_overlay_ops_write_failures": 0},
        },
    )
    _write_json(health / "sql_link_service_progress_latest.json", {"status": "ok", "current_step": "complete"})
    _write_json(health / "live_runtime_separation_control_latest.json", {"clearance_plan": {"clearance_state": "awaiting_coverage_cycles"}})

    payload = data_plane_src.build_payload(project_root)

    assert payload["overall_status"] == "ready"
    assert payload["write_failure_count"] == 0
    assert payload["current_storage_write_ready"] is True
    assert payload["write_path_recovery_evidence"]["overlay_only_write_relief"] is True
    assert payload["write_path_recovered_by_storage"] is True
    assert payload["hot_path_over_budget_bytes"] == 0


def test_data_plane_recovery_controller_uses_effective_raw_live_when_fresh_empty_overlay_clears_stale_raw_live(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    health = project_root / "governance" / "health"
    _write_json(
        health / "incident_timeline_latest.json",
        {"recent_incidents": [{"summary": "write_failure", "timestamp_utc": "2026-06-24T14:48:20+00:00"} for _ in range(3)]},
    )
    _write_json(health / "external_backlog_drain_latest.json", {"overall_status": "blocked", "blocked_reasons": ["external_storage_unavailable", "market_hours_guard"]})
    _write_json(health / "ingestion_priority_queue_latest.json", {"lane_counts": {"core": {"pending_lines": 954}}})
    _write_json(health / "storage_tier_policy_latest.json", {"pressure": {"hot_path_over_budget_bytes": 319843803301}})
    _write_json(
        health / "ingestion_storage_control_latest.json",
        {
            "overall_status": "needs_work",
            "severity": "stable",
            "pressure_index": 0.0,
            "backpressure_quality_score": 100,
            "recovery_quality_score": 82,
            "steady_state": {"target_status": {"steady_state_ready": False}},
            "external_route_verification": {"verification_state": "active_local_ready"},
            "backpressure": {
                "core_pending_lines": 0,
                "total_pending_lines": 0,
                "overlay_adjusted": True,
                "overlay_pressure_clear": True,
                "oldest_pending_age_seconds": 0.0,
                "raw_live": {
                    "core_pending_lines": 954,
                    "total_pending_lines": 15195,
                    "oldest_pending_age_seconds": 20197.0,
                },
                "effective_raw_live": {
                    "core_pending_lines": 0,
                    "total_pending_lines": 0,
                    "oldest_pending_age_seconds": 0.0,
                    "source": "fresh_empty_sql_ingestion_overlay",
                    "reconciled_from_raw_live": True,
                    "raw_live_estimate": {
                        "core_pending_lines": 954,
                        "total_pending_lines": 15195,
                        "oldest_pending_age_seconds": 20197.0,
                    },
                },
                "effective_raw_live_source": "fresh_empty_sql_ingestion_overlay",
            },
            "data_integrity": {
                "sql_overlay_invalid_lines": 0,
                "sql_overlay_oversize_payloads": 0,
                "sql_overlay_ops_write_failures": 0,
            },
        },
    )
    _write_json(health / "sql_link_service_progress_latest.json", {"status": "running", "current_step": "shard_linking"})
    _write_json(health / "live_runtime_separation_control_latest.json", {"clearance_plan": {"clearance_state": "awaiting_coverage_cycles"}})

    payload = data_plane_src.build_payload(project_root)

    assert payload["overall_status"] == "ready"
    assert payload["queue_depth"] == 0
    assert payload["queue_depth_source"] == "fresh_empty_sql_ingestion_overlay"
    assert payload["current_storage_write_ready"] is True
    assert payload["write_path_recovery_evidence"]["raw_live_clear"] is True
    assert payload["write_path_recovery_evidence"]["raw_live"]["total_pending_lines"] == 0
    assert payload["write_path_recovery_evidence"]["effective_backpressure"]["raw_live_estimate"]["total_pending_lines"] == 15195
    assert payload["write_path_recovery_evidence"]["overlay_only_write_relief"] is True
    assert payload["write_path_recovered_by_storage"] is True
    assert payload["hot_path_over_budget_bytes"] == 0


def test_data_plane_recovery_controller_uses_stable_raw_live_truth_without_overlay_adjustment(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    health = project_root / "governance" / "health"
    _write_json(health / "incident_timeline_latest.json", {"recent_incidents": []})
    _write_json(health / "external_backlog_drain_latest.json", {"overall_status": "drain_active"})
    _write_json(health / "ingestion_priority_queue_latest.json", {"queue_depth": 6175})
    _write_json(health / "storage_tier_policy_latest.json", {"pressure": {"hot_path_over_budget_bytes": 0}})
    _write_json(
        health / "ingestion_storage_control_latest.json",
        {
            "overall_status": "ready",
            "severity": "stable",
            "pressure_index": 0.031,
            "backpressure_quality_score": 100,
            "steady_state": {"target_status": {"steady_state_ready": True}},
            "external_route_verification": {"verification_state": "active_local_ready"},
            "backpressure": {
                "overlay_adjusted": False,
                "effective_raw_live_source": "raw_live_backpressure",
                "effective_raw_live": {
                    "core_pending_lines": 471,
                    "total_pending_lines": 1575,
                    "oldest_pending_age_seconds": 7.5,
                },
            },
            "data_integrity": {
                "sql_overlay_invalid_lines": 0,
                "sql_overlay_oversize_payloads": 0,
                "sql_overlay_ops_write_failures": 0,
            },
        },
    )
    _write_json(health / "sql_link_service_progress_latest.json", {"status": "running", "current_step": "shard_linking"})
    _write_json(health / "live_runtime_separation_control_latest.json", {"clearance_plan": {"clearance_state": "ready"}})

    payload = data_plane_src.build_payload(project_root)

    assert payload["overall_status"] == "ready"
    assert payload["recovery_state"] == "stable"
    assert payload["queue_depth"] == 1575
    assert payload["queue_depth_source"] == "raw_live_backpressure"
    assert payload["write_path_recovery_evidence"]["effective_backpressure"]["stable_raw_live_truth"] is True


def test_data_plane_recovery_controller_clears_snapshot_failures_after_fresh_cache(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    health = project_root / "governance" / "health"
    _write_json(
        health / "incident_timeline_latest.json",
        {
            "recent_incidents": [
                {"summary": "get_accounts_snapshot", "timestamp_utc": "2026-05-03T20:18:08+00:00"},
            ]
        },
    )
    _write_json(health / "external_backlog_drain_latest.json", {"overall_status": "ready"})
    _write_json(health / "ingestion_priority_queue_latest.json", {"queue_depth": 66})
    _write_json(health / "storage_tier_policy_latest.json", {"pressure": {"hot_path_over_budget_bytes": 0}})
    _write_json(health / "live_runtime_separation_control_latest.json", {"clearance_plan": {"clearance_state": "ready"}})
    _write_json(
        health / "broker_truth_shared_snapshot_schwab_latest.json",
        {"fetched": {"ok": True}, "timestamp_utc": "2026-05-03T23:05:40+00:00"},
    )

    payload = data_plane_src.build_payload(project_root)

    assert payload["overall_status"] == "ready"
    assert payload["account_snapshot_failure_count"] == 0
    assert payload["raw_account_snapshot_failure_count"] == 1
    assert payload["recovery_contract"]["snapshot_recovered_by_cache"] is True
    assert payload["snapshot_recovery_contract"]["recovered_by_fresh_cache"] is True


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


def test_data_plane_recovery_controller_trusts_storage_control_steady_state_over_stale_hot_path(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    health = project_root / "governance" / "health"
    _write_json(health / "incident_timeline_latest.json", {"recent_incidents": []})
    _write_json(health / "external_backlog_drain_latest.json", {"overall_status": "ready"})
    _write_json(health / "ingestion_priority_queue_latest.json", {"queue_depth": 66})
    _write_json(health / "storage_tier_policy_latest.json", {"pressure": {"hot_path_over_budget_bytes": 100}})
    _write_json(
        health / "ingestion_storage_control_latest.json",
        {
            "overall_status": "ready",
            "severity": "stable",
            "backpressure_quality_score": 100.0,
            "recovery_quality_score": 96.0,
            "steady_state": {"target_status": {"steady_state_ready": True}},
            "external_route_verification": {"verification_state": "ready"},
        },
    )
    _write_json(health / "live_runtime_separation_control_latest.json", {"clearance_plan": {"clearance_state": "ready"}})
    _write_json(health / "sql_link_service_progress_latest.json", {"status": "running", "current_step": "shard_linking"})

    payload = data_plane_src.build_payload(project_root)

    assert payload["overall_status"] == "ready"
    assert payload["recovery_state"] == "stable"
    assert payload["queue_depth"] == 66
    assert payload["small_steady_queue"] is True
    assert payload["hot_path_over_budget_bytes"] == 0
    assert payload["raw_hot_path_over_budget_bytes"] == 100
    assert payload["storage_steady_state_ready"] is True


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
