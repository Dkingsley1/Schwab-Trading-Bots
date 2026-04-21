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
    assert payload["notification_escalation"]["attended_runtime_ready"] is True
    assert payload["incident_timeline"]["open_incident_count"] == 1
    assert payload["incident_review"]["review_required"] is True
    assert payload["lane_thaw_controller"]["paused_lane_count"] == 1
    assert payload["data_plane_recovery"]["write_failure_count"] == 2
    assert payload["autonomous_repair_path_count"] >= 2
    assert payload["lane_recovery_playbooks"]["triggered_playbook_count"] >= 3
