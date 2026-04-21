import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.ops import ops_coordinator


def test_build_ops_coordinator_payload_runs_core_steps(tmp_path, monkeypatch) -> None:
    project_root = tmp_path / "project"
    (project_root / "governance" / "health").mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(ops_coordinator, "PY", Path("/usr/bin/python3"))

    calls: list[list[str]] = []

    def _fake_run(cmd: list[str], *, cwd: Path, payload_path: Path | None = None) -> dict:
        calls.append(cmd)
        joined = " ".join(cmd)
        if "resource_guard.py" in joined:
            payload = {"ok": True, "memory_pressure_kind": "none"}
        elif "process_watchdog.py" in joined:
            payload = {"ok": True, "storage_mode": "external", "storage_ok": True, "network_ok": True, "restart_storms": []}
        elif "live_readiness_smoke.py" in joined:
            payload = {
                "ok": True,
                "overall_status": "ready",
                "readiness_score": 96.5,
                "broker_ready": True,
                "session_ready": True,
                "paper_lane_fresh": True,
                "live_lane_running": True,
                "process_watchdog": {"healthy": True},
            }
        elif "live_runtime_separation_control.py" in joined:
            payload = {
                "ok": True,
                "overall_status": "degraded",
                "shared_host_pressure": {"contention_score": 2},
                "release_contract": {
                    "live_lane_should_be_read_only": True,
                    "promotions_should_wait_for_cold_lane": True,
                    "shared_host_training_resume_allowed": False,
                },
                "clearance_plan": {
                    "clearance_state": "awaiting_coverage_cycles",
                    "coverage_gap_closer": {"launch_state": "stage_only_off_hours"},
                },
            }
        elif "incident_timeline.py" in joined:
            payload = {"ok": True, "overall_status": "degraded", "recent_incident_count": 4, "open_incident_count": 1}
        elif "derived_state_snapshot.py" in joined:
            payload = {"ok": True, "risk_level": "medium", "gross_risk_budget": 0.61, "max_total_actions_per_hour": 28}
        elif "strategy_research_lane.py" in joined:
            payload = {"ok": True, "promotable": False, "research_sandbox_ok": True, "summary": {"recommended_action": "hold"}}
        elif "training_registry_audit.py" in joined:
            payload = {
                "ok": True,
                "active_sample_starved": [{"bot_id": "brain_refinery_v4_simple"}],
                "active_quality_failed": [],
                "active_stale_diagnostics": [{"bot_id": "brain_refinery_v13_choppy"}],
                "tier_counts": {"active_repair": 1, "active_stale": 1},
            }
        elif "training_label_audit.py" in joined:
            payload = {
                "ok": True,
                "top_actions": ["fix_shared_runtime_input"],
                "recommendation_counts": {"fix_shared_runtime_input": 1},
            }
        elif "provider_mesh_control.py" in joined:
            payload = {
                "ok": True,
                "overall_status": "ready",
                "summary": {"required_failure_count": 0, "soft_failure_count": 1},
            }
        elif "platform_control_plane_report.py" in joined:
            payload = {
                "ok": True,
                "storage_sql_backlog_shaping": {"pending_lines": 120, "pending_lines_cold": 45, "cold_lane_recommendation": "offload_shadow_pnl_attribution"},
                "model_registry_and_rollout": {"promotion_status": "held_out", "training_reason": "sample_starved"},
                "institutional_readiness": {"overall_status": "advancing", "overall_score": 81.25},
            }
        elif "service_control_plane.py" in joined:
            payload = {
                "ok": True,
                "overall_status": "degraded",
                "summary": {"completion_score": 71.4},
            }
        elif "promotion_autopilot_packet.py" in joined:
            payload = {
                "ok": True,
                "overall_status": "ready",
                "autopilot_state": "awaiting_approval",
                "promotion_ready": True,
                "blockers": [],
            }
        elif "notification_escalation_ladder.py" in joined:
            payload = {
                "ok": False,
                "overall_status": "degraded",
                "attended_runtime_ready": True,
                "unattended_runtime_ready": False,
                "critical_backlog": {"grouped_unsent_count": 0},
            }
        elif "incident_review_packet.py" in joined:
            payload = {
                "ok": False,
                "overall_status": "degraded",
                "review_required": True,
                "open_incident_count": 1,
            }
        elif "lane_thaw_controller.py" in joined:
            payload = {
                "ok": True,
                "overall_status": "degraded",
                "paused_lane_count": 2,
                "candidate_count": 1,
                "cooldown_history": {
                    "chronic_offender_count": 1,
                    "watchlist_count": 2,
                },
            }
        elif "data_plane_recovery_controller.py" in joined:
            payload = {
                "ok": True,
                "overall_status": "degraded",
                "recovery_state": "recovering_under_guard",
                "write_failure_count": 3,
                "account_snapshot_failure_count": 1,
                "writer_handoff_contract": {"writer_service_active": True},
                "backlog_recovery_contract": {"drain_progress_lines": 40},
            }
        elif "sql_access_runtime_audit.py" in joined:
            payload = {
                "ok": True,
                "critical_packages_ok": True,
                "profile_files_present": {"live": True, "research": True, "media": True, "ops": True},
                "recommendations": ["sql_access_runtime_ready"],
                "data_library_roles": {"sqlite": "ingestion", "duckdb": "analytics"},
            }
        elif "runtime_dependency_profiles.py" in joined:
            payload = {
                "ok": True,
                "profile_counts": {"live": 10, "research": 20, "media": 5, "ops": 8},
                "overlap_package_count": 4,
            }
        elif "sql_analytics_mirror.py" in joined:
            payload = {
                "ok": True,
                "summary_refresh_ok": True,
                "materialized_summaries": {"source_record_count": 2500},
                "duckdb_mirror": {"mirror_ready": True},
            }
        elif "macro_event_intelligence.py" in joined:
            payload = {
                "ok": True,
                "overall_status": "ready",
                "market_relevance": "high",
                "transcript_quality": "full_replay",
            }
        elif "autonomy_control_plane.py" in joined:
            payload = {
                "ok": True,
                "overall_status": "degraded",
                "autonomy_score": 76.5,
                "lane_recovery_playbooks": {"triggered_playbook_count": 2},
            }
        else:
            raise AssertionError(f"unexpected command: {cmd}")
        return {"cmd": cmd, "rc": 0, "duration_ms": 5.0, "payload": payload, "stdout_tail": "", "stderr_tail": ""}

    monkeypatch.setattr(ops_coordinator, "_run_json_command", _fake_run)

    payload = ops_coordinator.build_ops_coordinator_payload(
        project_root,
        day="20260404",
        max_rows=4000,
        strategy_max_age_minutes=90.0,
        sandbox_max_age_minutes=720.0,
        watchdog_refresh_max_age_seconds=7200,
        resource_profile="refresh",
    )

    assert payload["ok"] is True
    assert payload["overall_status"] == "degraded"
    assert payload["summary"]["storage_mode"] == "external"
    assert payload["summary"]["live_readiness_status"] == "ready"
    assert payload["summary"]["runtime_separation_status"] == "degraded"
    assert payload["summary"]["runtime_clearance_state"] == "awaiting_coverage_cycles"
    assert payload["summary"]["shared_host_training_resume_allowed"] is False
    assert payload["summary"]["institutional_readiness_score"] == 81.25
    assert payload["summary"]["risk_level"] == "medium"
    assert payload["summary"]["recommended_action"] == "hold"
    assert payload["summary"]["pending_lines"] == 120
    assert payload["summary"]["active_sample_starved"] == 1
    assert payload["summary"]["active_stale_diagnostics"] == 1
    assert payload["summary"]["incident_timeline_status"] == "degraded"
    assert payload["summary"]["promotion_autopilot_state"] == "awaiting_approval"
    assert payload["summary"]["notification_ladder_status"] == "degraded"
    assert payload["summary"]["incident_review_status"] == "degraded"
    assert payload["summary"]["lane_thaw_candidates"] == 1
    assert payload["summary"]["lane_thaw_chronic_offenders"] == 1
    assert payload["summary"]["data_plane_write_failures"] == 3
    assert payload["summary"]["sql_access_runtime_ready"] is True
    assert payload["summary"]["runtime_dependency_profiles_ready"] is True
    assert payload["summary"]["analytics_mirror_ready"] is True
    assert payload["summary"]["macro_event_relevance"] == "high"
    assert payload["summary"]["autonomy_status"] == "degraded"
    assert payload["summary"]["autonomy_score"] == 76.5
    assert payload["live_readiness"]["watchdog_healthy"] is True
    assert payload["runtime_separation"]["contention_score"] == 2
    assert payload["runtime_separation"]["clearance_state"] == "awaiting_coverage_cycles"
    assert payload["runtime_separation"]["coverage_gap_launch_state"] == "stage_only_off_hours"
    assert payload["incident_timeline"]["open_incident_count"] == 1
    assert payload["training_label_quality"]["top_actions"] == ["fix_shared_runtime_input"]
    assert payload["provider_mesh"]["overall_status"] == "ready"
    assert payload["service_control_plane"]["overall_status"] == "degraded"
    assert payload["promotion_autopilot"]["autopilot_state"] == "awaiting_approval"
    assert payload["notification_escalation_ladder"]["attended_runtime_ready"] is True
    assert payload["incident_review_packet"]["review_required"] is True
    assert payload["lane_thaw_controller"]["candidate_count"] == 1
    assert payload["lane_thaw_controller"]["chronic_offender_count"] == 1
    assert payload["lane_thaw_controller"]["watchlist_count"] == 2
    assert payload["data_plane_recovery"]["write_failure_count"] == 3
    assert payload["sql_access_runtime_audit"]["critical_packages_ok"] is True
    assert payload["runtime_dependency_profiles"]["overlap_package_count"] == 4
    assert payload["sql_analytics_mirror"]["duckdb_mirror_ready"] is True
    assert payload["macro_event_intelligence"]["market_relevance"] == "high"
    assert payload["autonomy_control_plane"]["autonomy_score"] == 76.5
    assert payload["steps"]["live_readiness_smoke"]["status"] == "ok"
    assert payload["steps"]["live_runtime_separation_control"]["status"] == "ok"
    assert payload["steps"]["incident_timeline"]["status"] == "ok"
    assert payload["steps"]["strategy_research_fast"]["status"] == "ok"
    assert payload["steps"]["training_registry_audit"]["status"] == "ok"
    assert payload["steps"]["training_label_audit"]["status"] == "ok"
    assert payload["steps"]["provider_mesh"]["status"] == "ok"
    assert payload["steps"]["service_control_plane"]["status"] == "ok"
    assert payload["steps"]["promotion_autopilot_packet"]["status"] == "ok"
    assert payload["steps"]["notification_escalation_ladder"]["status"] == "degraded"
    assert payload["steps"]["incident_review_packet"]["status"] == "degraded"
    assert payload["steps"]["lane_thaw_controller"]["status"] == "ok"
    assert payload["steps"]["data_plane_recovery_controller"]["status"] == "ok"
    assert payload["steps"]["sql_access_runtime_audit"]["status"] == "ok"
    assert payload["steps"]["runtime_dependency_profiles"]["status"] == "ok"
    assert payload["steps"]["sql_analytics_mirror"]["status"] == "ok"
    assert payload["steps"]["macro_event_intelligence"]["status"] == "ok"
    assert payload["steps"]["autonomy_control_plane"]["status"] == "ok"
    assert any("process_watchdog.py" in " ".join(cmd) for cmd in calls)


def test_build_ops_coordinator_payload_fails_when_watchdog_reports_not_ok(tmp_path, monkeypatch) -> None:
    project_root = tmp_path / "project"
    (project_root / "governance" / "health").mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(ops_coordinator, "PY", Path("/usr/bin/python3"))

    def _fake_run(cmd: list[str], *, cwd: Path, payload_path: Path | None = None) -> dict:
        joined = " ".join(cmd)
        if "resource_guard.py" in joined:
            payload = {"ok": True}
        elif "process_watchdog.py" in joined:
            payload = {"ok": False, "reason": "restart_storm"}
        elif "live_readiness_smoke.py" in joined:
            payload = {"ok": True, "overall_status": "ready"}
        elif "live_runtime_separation_control.py" in joined:
            payload = {"ok": True, "overall_status": "ready"}
        elif "incident_timeline.py" in joined:
            payload = {"ok": True, "overall_status": "ready"}
        elif "promotion_autopilot_packet.py" in joined:
            payload = {"ok": True, "overall_status": "ready"}
        elif "notification_escalation_ladder.py" in joined:
            payload = {"ok": True, "overall_status": "ready", "critical_backlog": {"grouped_unsent_count": 0}}
        elif "incident_review_packet.py" in joined:
            payload = {"ok": True, "overall_status": "ready", "review_required": False}
        elif "lane_thaw_controller.py" in joined:
            payload = {"ok": True, "overall_status": "ready"}
        elif "data_plane_recovery_controller.py" in joined:
            payload = {"ok": True, "overall_status": "ready"}
        elif "sql_access_runtime_audit.py" in joined:
            payload = {"ok": True, "critical_packages_ok": True}
        elif "runtime_dependency_profiles.py" in joined:
            payload = {"ok": True, "profile_counts": {"live": 1, "research": 1, "media": 1, "ops": 1}}
        elif "sql_analytics_mirror.py" in joined:
            payload = {"ok": True, "duckdb_mirror": {"mirror_ready": True}}
        elif "macro_event_intelligence.py" in joined:
            payload = {"ok": True, "overall_status": "ready"}
        elif "autonomy_control_plane.py" in joined:
            payload = {"ok": True, "overall_status": "ready"}
        else:
            payload = {"ok": True}
        return {"cmd": cmd, "rc": 0, "duration_ms": 5.0, "payload": payload, "stdout_tail": "", "stderr_tail": ""}

    monkeypatch.setattr(ops_coordinator, "_run_json_command", _fake_run)

    payload = ops_coordinator.build_ops_coordinator_payload(
        project_root,
        day="20260404",
        max_rows=4000,
        strategy_max_age_minutes=90.0,
        sandbox_max_age_minutes=720.0,
        watchdog_refresh_max_age_seconds=7200,
        resource_profile="refresh",
    )

    assert payload["ok"] is False
    assert payload["steps"]["process_watchdog"]["status"] == "error"
