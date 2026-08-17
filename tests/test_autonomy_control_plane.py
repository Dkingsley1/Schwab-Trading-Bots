import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.ops import autonomy_control_plane


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def test_autonomy_control_plane_rolls_up_split_recovery_coverage_auth_and_promotion(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    health_root = project_root / "governance" / "health"
    walk_root = project_root / "governance" / "walk_forward"
    champion_root = project_root / "governance" / "champion_challenger"

    _write_json(
        health_root / "live_runtime_separation_control_latest.json",
        {
            "overall_status": "blocked",
            "shared_host_pressure": {"contention_score": 3},
            "release_contract": {
                "live_lane_should_be_read_only": True,
                "promotions_should_wait_for_cold_lane": True,
                "shared_host_training_resume_allowed": False,
            },
            "clearance_plan": {"clearance_state": "awaiting_coverage_cycles"},
            "recommended_actions": ["keep live runtime on frozen release bundles while training and coverage jobs run in a cold lane"],
        },
    )
    _write_json(health_root / "live_readiness_smoke_latest.json", {"live_lane_running": True})
    _write_json(
        health_root / "auth_lease_manager_latest.json",
        {
            "overall_status": "degraded",
            "lease_state": "warning",
            "lease_budget": {"expires_in_seconds": 900},
            "recommended_actions": ["./scripts/ops/opsctl.sh token-refresh --json"],
            "fallback_ladder": ["silent_refresh", "interactive_token_refresh"],
        },
    )
    _write_json(
        walk_root / "coverage_seed_latest.json",
        {
            "overall_status": "needs_coverage",
            "coverage_shortfall_bots": 2,
            "seed_queue": [{"bot_id": "brain_refinery_v43_intraday_ultrafast_proxy"}],
            "recommended_actions": ["keep a standing walk-forward seed queue so promotion coverage is built continuously instead of only during retrain windows"],
        },
    )
    _write_json(
        walk_root / "coverage_gap_closer_latest.json",
        {
            "active_stage": [{"bot_id": "brain_refinery_v43_intraday_ultrafast_proxy"}],
            "autopilot_contract": {
                "overall_status": "degraded",
                "launch_state": "stage_only_off_hours",
                "stage_candidate_count": 1,
                "can_launch_now": False,
                "off_hours_preferred": True,
                "next_action": "launch the staged coverage pass after the cold lane is clear",
            },
        },
    )
    _write_json(health_root / "training_requalification_latest.json", {"reactivation_ready_count": 4})
    _write_json(
        champion_root / "promotion_autopilot_packet_latest.json",
        {
            "overall_status": "ready",
            "autopilot_state": "awaiting_approval",
            "promotion_ready": True,
            "blockers": [],
            "approval_record": {"approval_state": "awaiting_operator_signoff"},
            "recommended_actions": ["record operator approval against the signed packet sha and rollback reference before promotion"],
        },
    )
    _write_json(
        health_root / "live_canary_control_latest.json",
        {
            "overall_status": "degraded",
            "recommended_mode": "preapproved_supervised",
            "supervised_canary_ready": False,
            "staged_preclearance_ready": True,
            "preapproved_supervised_ready": True,
            "preclearance_score": 85.0,
            "bounded_blocker_count": 3,
            "blocking_reasons": ["live_lane_not_running", "runtime_clearance_not_ready", "promotion_packet_preclearance_only"],
        },
    )
    _write_json(
        health_root / "incident_timeline_latest.json",
        {
            "overall_status": "degraded",
            "open_incident_count": 1,
            "recommended_actions": ["use the incident timeline as the single review surface for watchdog, auth, and failover interventions"],
        },
    )
    _write_json(
        health_root / "incident_review_packet_latest.json",
        {"overall_status": "degraded", "review_required": True, "open_incident_count": 1},
    )
    _write_json(
        health_root / "notification_escalation_ladder_latest.json",
        {"overall_status": "degraded", "attended_runtime_ready": True, "unattended_runtime_ready": False, "critical_backlog": {"grouped_unsent_count": 0}},
    )
    _write_json(
        health_root / "lane_thaw_controller_latest.json",
        {"overall_status": "degraded", "paused_lane_count": 1, "candidate_count": 0, "blocked_count": 1},
    )
    _write_json(
        health_root / "data_plane_recovery_controller_latest.json",
        {
            "overall_status": "degraded",
            "recovery_state": "recovering_under_guard",
            "write_failure_count": 2,
            "account_snapshot_failure_count": 1,
            "queue_depth": 50,
            "writer_handoff_contract": {"writer_service_active": True},
            "backlog_recovery_contract": {"drain_progress_lines": 25},
        },
    )
    _write_json(
        health_root / "runtime_throttle_control_latest.json",
        {
            "overall_status": "degraded",
            "throttle_profile": "sustain",
            "host_saturation_score": 71.5,
            "compute_pressure_level": "high",
            "memory_pressure_level": "elevated",
            "upgrade_track": {"upgradeable": True},
            "recommended_actions": ["shift retention, timeline, report, and SQL maintenance jobs into off-hours throttle windows before touching the live lanes"],
        },
    )
    _write_json(
        health_root / "data_ingress_latest_intraday_aggressive_equities_schwab.json",
        {"loop_state": "paused_anomaly_killswitch", "iter_error_rate": 0.0, "iter_error_count": 0, "total_counts": {"api_error": 0}},
    )
    _write_json(
        health_root / "data_ingress_latest_default_crypto_coinbase.json",
        {"loop_state": "degraded_market_data", "iter_error_rate": 0.2, "iter_error_count": 2, "total_counts": {"api_error": 5}},
    )

    payload = autonomy_control_plane.build_payload(project_root)

    assert payload["overall_status"] == "blocked"
    assert payload["live_research_split"]["contention_score"] == 3
    assert payload["live_research_split"]["clearance_state"] == "awaiting_coverage_cycles"
    assert payload["coverage_autopilot"]["defer_retrains"] is True
    assert payload["coverage_autopilot"]["launch_state"] == "stage_only_off_hours"
    assert payload["auth_lease_workflow"]["prestage_refresh_required"] is True
    assert payload["promotion_autopilot"]["autopilot_state"] == "awaiting_approval"
    assert payload["live_canary_control"]["preapproved_supervised_ready"] is True
    assert payload["notification_escalation"]["attended_runtime_ready"] is True
    assert payload["incident_timeline"]["open_incident_count"] == 1
    assert payload["incident_review"]["review_required"] is True
    assert payload["lane_thaw_controller"]["paused_lane_count"] == 1
    assert payload["data_plane_recovery"]["write_failure_count"] == 2
    assert payload["runtime_throttle_control"]["throttle_profile"] == "sustain"
    assert payload["component_statuses"]["runtime_throttle_control"] == "degraded"
    assert payload["component_statuses"]["live_canary_control"] == "needs_work"
    assert payload["component_statuses"]["incident_closure_loop"] == "blocked"
    assert "incident_review_packet_latest.json" in payload["incident_closure_loop"]["required_artifacts"]
    assert payload["autonomous_repair_path_count"] >= 2
    assert payload["lane_recovery_playbooks"]["triggered_playbook_count"] >= 3


def test_autonomy_control_plane_prefers_incident_closeout_artifact_when_present(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    health_root = project_root / "governance" / "health"
    walk_root = project_root / "governance" / "walk_forward"
    champion_root = project_root / "governance" / "champion_challenger"

    _write_json(health_root / "live_runtime_separation_control_latest.json", {"overall_status": "ready", "clearance_plan": {"clearance_state": "cleared"}})
    _write_json(health_root / "live_readiness_smoke_latest.json", {"live_lane_running": True})
    _write_json(health_root / "auth_lease_manager_latest.json", {"overall_status": "ready", "lease_state": "healthy", "lease_budget": {"expires_in_seconds": 3600}})
    _write_json(walk_root / "coverage_seed_latest.json", {"coverage_shortfall_bots": 0, "seed_queue": []})
    _write_json(walk_root / "coverage_gap_closer_latest.json", {"autopilot_contract": {"overall_status": "ready"}})
    _write_json(health_root / "training_requalification_latest.json", {"reactivation_ready_count": 0})
    _write_json(champion_root / "promotion_autopilot_packet_latest.json", {"overall_status": "ready", "autopilot_state": "awaiting_approval", "promotion_ready": True, "blockers": [], "approval_record": {"approval_state": "awaiting_operator_signoff"}})
    _write_json(health_root / "live_canary_control_latest.json", {"overall_status": "ready", "recommended_mode": "supervised_canary", "supervised_canary_ready": True, "staged_preclearance_ready": True, "preapproved_supervised_ready": True})
    _write_json(health_root / "incident_timeline_latest.json", {"overall_status": "ready", "open_incident_count": 0, "recent_incident_count": 0})
    _write_json(health_root / "incident_review_packet_latest.json", {"overall_status": "ready", "review_required": False, "open_incident_count": 0})
    _write_json(health_root / "incident_closeout_autopilot_latest.json", {"overall_status": "ready", "closeout_ready": True, "required_artifacts": [], "blocking_surfaces": []})
    _write_json(health_root / "notification_escalation_ladder_latest.json", {"overall_status": "ready", "attended_runtime_ready": True, "unattended_runtime_ready": True, "critical_backlog": {"grouped_unsent_count": 0}})
    _write_json(health_root / "lane_thaw_controller_latest.json", {"overall_status": "ready", "paused_lane_count": 0, "candidate_count": 0, "blocked_count": 0})
    _write_json(health_root / "data_plane_recovery_controller_latest.json", {"overall_status": "ready", "recovery_state": "ready", "write_failure_count": 0, "account_snapshot_failure_count": 0, "queue_depth": 0, "writer_handoff_contract": {"writer_service_active": True}, "backlog_recovery_contract": {"drain_progress_lines": 0}})
    _write_json(health_root / "runtime_throttle_control_latest.json", {"overall_status": "ready", "throttle_profile": "protect_live", "host_saturation_score": 10.0, "compute_pressure_level": "low", "memory_pressure_level": "low", "upgrade_track": {"upgradeable": True}})

    payload = autonomy_control_plane.build_payload(project_root)

    assert payload["incident_closure_loop"]["closeout_ready"] is True
    assert payload["component_statuses"]["incident_closure_loop"] == "ready"


def test_autonomy_control_plane_degrades_missing_runtime_throttle_when_script_exists(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    health_root = project_root / "governance" / "health"
    walk_root = project_root / "governance" / "walk_forward"
    champion_root = project_root / "governance" / "champion_challenger"
    (project_root / "scripts" / "ops").mkdir(parents=True, exist_ok=True)
    (project_root / "scripts" / "ops" / "runtime_throttle_control.py").write_text("#!/usr/bin/env python3\n", encoding="utf-8")

    _write_json(health_root / "live_runtime_separation_control_latest.json", {"overall_status": "ready", "clearance_plan": {"clearance_state": "cleared"}})
    _write_json(health_root / "live_readiness_smoke_latest.json", {"live_lane_running": True})
    _write_json(health_root / "auth_lease_manager_latest.json", {"overall_status": "ready", "lease_state": "healthy", "lease_budget": {"expires_in_seconds": 3600}})
    _write_json(walk_root / "coverage_seed_latest.json", {"coverage_shortfall_bots": 0, "seed_queue": []})
    _write_json(walk_root / "coverage_gap_closer_latest.json", {"autopilot_contract": {"overall_status": "ready"}})
    _write_json(health_root / "training_requalification_latest.json", {"reactivation_ready_count": 0})
    _write_json(champion_root / "promotion_autopilot_packet_latest.json", {"overall_status": "ready", "autopilot_state": "awaiting_approval", "promotion_ready": True, "blockers": [], "approval_record": {"approval_state": "awaiting_operator_signoff"}})
    _write_json(health_root / "live_canary_control_latest.json", {"overall_status": "ready", "recommended_mode": "supervised_canary", "supervised_canary_ready": True, "staged_preclearance_ready": True, "preapproved_supervised_ready": True})
    _write_json(health_root / "incident_timeline_latest.json", {"overall_status": "ready", "open_incident_count": 0, "recent_incident_count": 0})
    _write_json(health_root / "incident_review_packet_latest.json", {"overall_status": "ready", "review_required": False, "open_incident_count": 0})
    _write_json(health_root / "incident_closeout_autopilot_latest.json", {"overall_status": "ready", "closeout_ready": True, "required_artifacts": [], "blocking_surfaces": []})
    _write_json(health_root / "notification_escalation_ladder_latest.json", {"overall_status": "ready", "attended_runtime_ready": True, "unattended_runtime_ready": True, "critical_backlog": {"grouped_unsent_count": 0}})
    _write_json(health_root / "lane_thaw_controller_latest.json", {"overall_status": "ready", "paused_lane_count": 0, "candidate_count": 0, "blocked_count": 0})
    _write_json(health_root / "data_plane_recovery_controller_latest.json", {"overall_status": "ready", "recovery_state": "ready", "write_failure_count": 0, "account_snapshot_failure_count": 0, "queue_depth": 0, "writer_handoff_contract": {"writer_service_active": True}, "backlog_recovery_contract": {"drain_progress_lines": 0}})

    payload = autonomy_control_plane.build_payload(project_root)

    assert payload["runtime_throttle_control"]["automation_script_present"] is True
    assert payload["component_statuses"]["runtime_throttle_control"] == "degraded"


def test_autonomy_control_plane_normalizes_protect_live_throttle_to_needs_work_component_status(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    health_root = project_root / "governance" / "health"
    walk_root = project_root / "governance" / "walk_forward"
    champion_root = project_root / "governance" / "champion_challenger"

    _write_json(health_root / "live_runtime_separation_control_latest.json", {"overall_status": "ready", "clearance_plan": {"clearance_state": "cleared"}})
    _write_json(health_root / "live_readiness_smoke_latest.json", {"live_lane_running": True})
    _write_json(health_root / "auth_lease_manager_latest.json", {"overall_status": "ready", "lease_state": "healthy", "lease_budget": {"expires_in_seconds": 3600}})
    _write_json(walk_root / "coverage_seed_latest.json", {"coverage_shortfall_bots": 0, "seed_queue": []})
    _write_json(walk_root / "coverage_gap_closer_latest.json", {"autopilot_contract": {"overall_status": "ready"}})
    _write_json(health_root / "training_requalification_latest.json", {"reactivation_ready_count": 0})
    _write_json(champion_root / "promotion_autopilot_packet_latest.json", {"overall_status": "ready", "autopilot_state": "awaiting_approval", "promotion_ready": True, "blockers": [], "approval_record": {"approval_state": "awaiting_operator_signoff"}})
    _write_json(health_root / "live_canary_control_latest.json", {"overall_status": "degraded", "recommended_mode": "preapproved_supervised", "supervised_canary_ready": False, "staged_preclearance_ready": True, "preapproved_supervised_ready": True})
    _write_json(health_root / "incident_timeline_latest.json", {"overall_status": "ready", "open_incident_count": 0, "recent_incident_count": 0})
    _write_json(health_root / "incident_review_packet_latest.json", {"overall_status": "ready", "review_required": False, "open_incident_count": 0})
    _write_json(health_root / "incident_closeout_autopilot_latest.json", {"overall_status": "ready", "closeout_ready": True, "required_artifacts": [], "blocking_surfaces": []})
    _write_json(health_root / "notification_escalation_ladder_latest.json", {"overall_status": "ready", "attended_runtime_ready": True, "unattended_runtime_ready": True, "critical_backlog": {"grouped_unsent_count": 0}})
    _write_json(health_root / "lane_thaw_controller_latest.json", {"overall_status": "ready", "paused_lane_count": 0, "candidate_count": 0, "blocked_count": 0})
    _write_json(health_root / "data_plane_recovery_controller_latest.json", {"overall_status": "ready", "recovery_state": "ready", "write_failure_count": 0, "account_snapshot_failure_count": 0, "queue_depth": 0, "writer_handoff_contract": {"writer_service_active": True}, "backlog_recovery_contract": {"drain_progress_lines": 0}})
    _write_json(
        health_root / "runtime_throttle_control_latest.json",
        {
            "overall_status": "blocked",
            "throttle_profile": "protect_live",
            "host_saturation_score": 100.0,
            "compute_pressure_level": "high",
            "memory_pressure_level": "high",
            "upgrade_track": {"upgradeable": True},
        },
    )

    payload = autonomy_control_plane.build_payload(project_root)

    assert payload["runtime_throttle_control"]["raw_overall_status"] == "blocked"
    assert payload["runtime_throttle_control"]["overall_status"] == "needs_work"
    assert payload["component_statuses"]["runtime_throttle_control"] == "needs_work"


def test_autonomy_control_plane_normalizes_bounded_watch_states_to_needs_work(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    health_root = project_root / "governance" / "health"
    walk_root = project_root / "governance" / "walk_forward"
    champion_root = project_root / "governance" / "champion_challenger"
    (project_root / "scripts" / "ops").mkdir(parents=True, exist_ok=True)
    (project_root / "scripts" / "ops" / "runtime_throttle_control.py").write_text("#!/usr/bin/env python3\n", encoding="utf-8")
    (project_root / "scripts" / "ops" / "chrome_headless_guard.py").write_text("#!/usr/bin/env python3\n", encoding="utf-8")

    _write_json(
        health_root / "live_runtime_separation_control_latest.json",
        {
            "overall_status": "ready",
            "release_contract": {
                "live_lane_should_be_read_only": True,
                "promotions_should_wait_for_cold_lane": True,
                "shared_host_training_resume_allowed": False,
            },
            "shared_host_pressure": {"contention_score": 1},
            "clearance_plan": {"clearance_state": "managed_coverage_stage_deferred"},
        },
    )
    _write_json(health_root / "live_readiness_smoke_latest.json", {"live_lane_running": False})
    _write_json(
        health_root / "auth_lease_manager_latest.json",
        {
            "overall_status": "degraded",
            "lease_state": "warning",
            "lease_budget": {"expires_in_seconds": 900},
            "fallback_ladder": ["silent_refresh", "interactive_token_refresh"],
        },
    )
    _write_json(walk_root / "coverage_seed_latest.json", {"coverage_shortfall_bots": 4, "seed_queue": [{"bot_id": "a"}]})
    _write_json(
        walk_root / "coverage_gap_closer_latest.json",
        {
            "autopilot_contract": {
                "overall_status": "degraded",
                "stage_candidate_count": 4,
                "launch_state": "stage_only_training_blocked",
                "next_action": "wait for retrain to finish",
            }
        },
    )
    _write_json(health_root / "training_requalification_latest.json", {"reactivation_ready_count": 0})
    _write_json(
        champion_root / "promotion_autopilot_packet_latest.json",
        {
            "overall_status": "degraded",
            "autopilot_state": "seed_ready",
            "promotion_ready": False,
            "blockers": ["coverage"],
            "readiness_repair_contract": {"repairable_gate_count": 1, "critical_repair_gate_count": 0},
            "signability_contract": {"committee_packet_seed_ready": True},
        },
    )
    _write_json(
        health_root / "live_canary_control_latest.json",
        {
            "overall_status": "blocked",
            "recommended_mode": "validate_only",
            "supervised_canary_ready": False,
            "staged_preclearance_ready": False,
            "preapproved_supervised_ready": False,
            "preclearance_score": 85.0,
            "blocking_reasons": [
                "faithful_live_money_contract_not_ready",
                "runtime_clearance_not_ready",
                "live_lane_read_only",
                "promotion_packet_preclearance_only",
            ],
        },
    )
    _write_json(
        health_root / "incident_timeline_latest.json",
        {"overall_status": "degraded", "open_incident_count": 0, "recent_incident_count": 3, "watch_surface_count": 3},
    )
    _write_json(health_root / "incident_review_packet_latest.json", {"overall_status": "ready", "review_required": False, "open_incident_count": 0})
    _write_json(
        health_root / "incident_closeout_autopilot_latest.json",
        {
            "overall_status": "ready",
            "closeout_ready": True,
            "bounded_closeout_path_ready": False,
            "required_artifacts": [],
            "blocking_surfaces": [],
        },
    )
    _write_json(health_root / "notification_escalation_ladder_latest.json", {"overall_status": "ready", "attended_runtime_ready": True, "unattended_runtime_ready": True, "critical_backlog": {"grouped_unsent_count": 0}})
    _write_json(health_root / "lane_thaw_controller_latest.json", {"overall_status": "ready", "paused_lane_count": 0, "candidate_count": 0, "blocked_count": 0})
    _write_json(
        health_root / "data_plane_recovery_controller_latest.json",
        {
            "overall_status": "degraded",
            "recovery_state": "recovering_under_guard",
            "write_failure_count": 0,
            "account_snapshot_failure_count": 7,
            "queue_depth": 54473,
            "writer_handoff_contract": {"writer_service_active": True},
            "backlog_recovery_contract": {"drain_progress_lines": -184},
        },
    )
    _write_json(
        health_root / "runtime_throttle_control_latest.json",
        {
            "overall_status": "blocked",
            "throttle_profile": "protect_live",
            "host_saturation_score": 88.0,
            "compute_pressure_level": "high",
            "memory_pressure_level": "high",
            "upgrade_track": {"upgradeable": True},
        },
    )
    _write_json(
        health_root / "chrome_headless_guard_latest.json",
        {
            "overall_status": "degraded",
            "timeline_pdf_policy": "headless_only",
            "interactive_protection_active": True,
            "timeline_autorender_suppressed": False,
            "headless_process_count": 1,
            "stale_headless_count": 0,
            "orphan_headless_count": 0,
            "upgrade_track": {"upgradeable": True},
        },
    )

    payload = autonomy_control_plane.build_payload(project_root)

    assert payload["overall_status"] == "ready"
    assert payload["component_statuses"]["live_research_split"] == "ready"
    assert payload["component_statuses"]["coverage_autopilot"] == "needs_work"
    assert payload["component_statuses"]["auth_lease_workflow"] == "needs_work"
    assert payload["component_statuses"]["data_plane_recovery"] == "needs_work"
    assert payload["component_statuses"]["live_canary_control"] == "needs_work"
    assert payload["component_statuses"]["incident_timeline"] == "needs_work"
    assert payload["component_statuses"]["runtime_throttle_control"] == "needs_work"
    assert payload["component_statuses"]["chrome_headless_guard"] == "needs_work"


def test_autonomy_control_plane_rewards_bounded_release_and_protection_contracts(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    health_root = project_root / "governance" / "health"
    walk_root = project_root / "governance" / "walk_forward"
    champion_root = project_root / "governance" / "champion_challenger"
    (project_root / "scripts" / "ops").mkdir(parents=True, exist_ok=True)
    (project_root / "scripts" / "ops" / "runtime_throttle_control.py").write_text("#!/usr/bin/env python3\n", encoding="utf-8")
    (project_root / "scripts" / "ops" / "chrome_headless_guard.py").write_text("#!/usr/bin/env python3\n", encoding="utf-8")

    _write_json(
        health_root / "live_runtime_separation_control_latest.json",
        {
            "overall_status": "degraded",
            "release_contract": {
                "live_lane_should_be_read_only": True,
                "promotions_should_wait_for_cold_lane": True,
                "shared_host_training_resume_allowed": False,
            },
            "shared_host_pressure": {"contention_score": 2},
            "clearance_plan": {"clearance_state": "awaiting_coverage_cycles"},
        },
    )
    _write_json(health_root / "live_readiness_smoke_latest.json", {"live_lane_running": False})
    _write_json(
        health_root / "auth_lease_manager_latest.json",
        {
            "overall_status": "degraded",
            "lease_state": "warning",
            "lease_budget": {"expires_in_seconds": 900},
            "fallback_ladder": ["silent_refresh", "interactive_token_refresh"],
        },
    )
    _write_json(walk_root / "coverage_seed_latest.json", {"coverage_shortfall_bots": 4, "seed_queue": [{"bot_id": "a"}]})
    _write_json(
        walk_root / "coverage_gap_closer_latest.json",
        {
            "autopilot_contract": {
                "overall_status": "degraded",
                "stage_candidate_count": 4,
                "launch_state": "waiting_for_idle",
                "next_action": "wait for retrain to finish",
            }
        },
    )
    _write_json(health_root / "training_requalification_latest.json", {"reactivation_ready_count": 0})
    _write_json(
        champion_root / "promotion_autopilot_packet_latest.json",
        {
            "overall_status": "degraded",
            "autopilot_state": "repairing_readiness",
            "promotion_ready": False,
            "blockers": ["training_success_confirmed", "promotion_packet_incomplete", "promotion_quality_gate_failed", "promotion_pipeline_failed"],
            "committee_packet_seed_ready": True,
            "readiness_repair_contract": {"repairable_gate_count": 3, "critical_repair_gate_count": 1},
        },
    )
    _write_json(
        health_root / "live_canary_control_latest.json",
        {
            "overall_status": "degraded",
            "recommended_mode": "preapproved_supervised",
            "supervised_canary_ready": False,
            "staged_preclearance_ready": True,
            "preapproved_supervised_ready": True,
            "preclearance_score": 95.0,
        },
    )
    _write_json(
        health_root / "incident_timeline_latest.json",
        {"overall_status": "degraded", "open_incident_count": 0, "recent_incident_count": 5, "watch_surface_count": 5},
    )
    _write_json(health_root / "incident_review_packet_latest.json", {"overall_status": "ready", "review_required": False, "open_incident_count": 0})
    _write_json(
        health_root / "incident_closeout_autopilot_latest.json",
        {
            "overall_status": "ready",
            "closeout_ready": True,
            "open_incident_count": 0,
            "closeout_score": 92.0,
            "bounded_closeout_path_ready": False,
            "required_artifacts": [],
            "blocking_surfaces": [],
        },
    )
    _write_json(health_root / "notification_escalation_ladder_latest.json", {"overall_status": "ready", "attended_runtime_ready": True, "unattended_runtime_ready": True, "critical_backlog": {"grouped_unsent_count": 0}})
    _write_json(health_root / "lane_thaw_controller_latest.json", {"overall_status": "ready", "paused_lane_count": 0, "candidate_count": 0, "blocked_count": 0})
    _write_json(
        health_root / "data_plane_recovery_controller_latest.json",
        {
            "overall_status": "degraded",
            "recovery_state": "recovering_under_guard",
            "write_failure_count": 0,
            "account_snapshot_failure_count": 7,
            "queue_depth": 54473,
            "writer_handoff_contract": {"writer_service_active": True},
            "backlog_recovery_contract": {"drain_progress_lines": -184},
        },
    )
    _write_json(
        health_root / "runtime_throttle_control_latest.json",
        {
            "overall_status": "blocked",
            "throttle_profile": "protect_live",
            "host_saturation_score": 88.0,
            "compute_pressure_level": "high",
            "memory_pressure_level": "high",
            "upgrade_track": {"upgradeable": True},
        },
    )
    _write_json(
        health_root / "chrome_headless_guard_latest.json",
        {
            "overall_status": "degraded",
            "timeline_pdf_policy": "headless_only",
            "interactive_protection_active": True,
            "timeline_autorender_suppressed": False,
            "headless_process_count": 0,
            "stale_headless_count": 0,
            "orphan_headless_count": 0,
            "runaway_detected": False,
            "runaway_without_lock": False,
            "upgrade_track": {"upgradeable": True},
        },
    )

    payload = autonomy_control_plane.build_payload(project_root)

    assert payload["overall_status"] == "ready"
    assert payload["component_statuses"]["chrome_headless_guard"] == "ready"
    assert payload["component_statuses"]["coverage_autopilot"] == "needs_work"
    assert payload["component_statuses"]["promotion_autopilot"] == "needs_work"
    assert payload["autonomy_score"] >= 95.0
