import json
from pathlib import Path

from scripts.ops import infrabot_adaptive_governor


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True), encoding="utf-8")


def _base_ready_health(project_root: Path) -> Path:
    health = project_root / "governance" / "health"
    _write_json(health / "pressure_relief_control_latest.json", {"overall_status": "ready", "host_saturation_score": 20, "compute_pressure_level": "normal", "memory_pressure_level": "normal"})
    _write_json(health / "runtime_throttle_control_latest.json", {"overall_status": "ready", "host_saturation_score": 20})
    _write_json(health / "memory_pressure_intelligence_latest.json", {"overall_status": "ready", "host_saturation_score": 20, "memory_pressure_level": "normal"})
    _write_json(health / "schwab_auth_supervisor_latest.json", {"overall_status": "ready", "token": {"expires_in_seconds": 1800}})
    _write_json(health / "auth_lease_manager_latest.json", {"overall_status": "ready", "lease_state": "healthy", "expires_in_seconds": 1800})
    _write_json(health / "premarket_token_guard_latest.json", {"ok": True, "network_ok": True})
    _write_json(health / "broker_readiness_latest.json", {"ready_for_open": True, "auth_ok": True, "network_ok": True, "token_expires_in_seconds": 1800})
    _write_json(health / "global_killswitch_latest.json", {"overall_status": "ready", "clear_state": {"blockers": []}})
    _write_json(health / "paper_400_ramp_latest.json", {"overall_status": "ready", "stage": "armed", "blockers": []})
    _write_json(health / "ingestion_storage_control_latest.json", {"overall_status": "ready", "backpressure": {"total_pending_lines": 0}, "storage": {"retention_debt_gb": 0}})
    _write_json(health / "storage_backpressure_autopilot_latest.json", {"overall_status": "ready"})
    _write_json(health / "writer_cycle_coordinator_latest.json", {"overall_status": "ready", "writer_state_before": {"child_writer_active": False, "complete_lock_handoff_needed": False}})
    _write_json(
        health / "training_runtime_control_latest.json",
        {
            "overall_status": "ready",
            "training_launch_contract": {"launch_allowed": True, "launch_blockers": [], "prep_blockers": [], "recommended_retrain_command": ["./scripts/ops/opsctl.sh", "retrain-orchestrate", "--json"]},
        },
    )
    _write_json(health / "livefeed_local_latest.json", {"status": "running", "alive": True, "idle_heartbeat_seconds": 10, "skipped_files": 0})
    _write_json(health / "live_feed_heavy_view_latest.json", {"mode": "active"})
    _write_json(health / "commands_hygiene_latest.json", {"overall_status": "ready", "commands_changed": False, "issues": []})
    _write_json(
        health / "command_validity_latest.json",
        {
            "ok": True,
            "overall_status": "degraded",
            "issues": [],
            "metrics": {
                "blocked_entry_count": 0,
                "degraded_entry_count": 0,
                "smoke_failure_count": 0,
                "runtime_smoke_failure_count": 0,
                "base_runtime_smoke_failure_count": 0,
                "contract_dispatch_smoke_failure_count": 0,
                "commands_hygiene_failure_count": 0,
                "contract_hash_mismatch_count": 0,
            },
        },
    )
    _write_json(health / "infrastructure_autofix_bot_latest.json", {"overall_status": "ready", "repair_plan": [], "advisory_repair_plan": []})
    _write_json(health / "infrabot_gap_roster_latest.json", {"overall_status": "ready", "active_count": 0})
    _write_json(health / "master_infrastructure_supervisor_latest.json", {"overall_status": "ready"})
    _write_json(health / "bot_quality_autopilot_latest.json", {"overall_status": "ready"})
    _write_json(health / "paper_profitability_control_latest.json", {"overall_status": "ready", "profitability_grade": "A"})
    _write_json(health / "runtime_paper_regression_guard_latest.json", {"overall_status": "ready"})
    _write_json(health / "paper_execution_backlog_relief_latest.json", {"ok": True})
    _write_json(health / "source_verification_latest.json", {"overall_status": "ready", "overall": {"unverified_sources": [], "stale_sources": []}})
    _write_json(health / "provider_mesh_latest.json", {"overall_status": "ready"})
    _write_json(health / "system_needs_intelligence_latest.json", {"low_grade_layer_audit": {"active_blocker_count": 0, "control_posture_grade": "A+"}})
    return health


def test_infrabot_adaptive_governor_routes_pressure_and_blocks_broad_fanout(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    health = _base_ready_health(project_root)
    _write_json(health / "pressure_relief_control_latest.json", {"overall_status": "degraded", "host_saturation_score": 82, "compute_pressure_level": "high", "memory_pressure_level": "normal"})
    _write_json(health / "runtime_throttle_control_latest.json", {"overall_status": "degraded", "host_saturation_score": 78})
    _write_json(health / "ingestion_storage_control_latest.json", {"overall_status": "degraded", "backpressure": {"total_pending_lines": 15000}, "storage": {"retention_debt_gb": 2.5}})
    _write_json(health / "writer_cycle_coordinator_latest.json", {"overall_status": "waiting_for_writer", "writer_state_before": {"child_writer_active": True, "complete_lock_handoff_needed": True}})
    _write_json(
        health / "training_runtime_control_latest.json",
        {
            "overall_status": "constrained",
            "training_launch_contract": {"launch_allowed": False, "launch_blockers": ["host_saturation"], "prep_blockers": []},
        },
    )
    _write_json(health / "infrastructure_autofix_bot_latest.json", {"overall_status": "degraded", "repair_plan": [{"name": "commands_hygiene"}], "advisory_repair_plan": []})

    payload = infrabot_adaptive_governor.build_payload(project_root, max_actions=4)

    assert payload["overall_status"] == "guarded"
    assert payload["system_needs_contract"]["need_count"] >= 4
    assert {need["id"] for need in payload["system_needs_contract"]["needs"]} >= {"host_pressure", "storage_backpressure", "writer_handoff", "training_gate", "infrastructure_repair_plan"}
    assert payload["capability_registry"]["capability_count"] >= 10
    assert payload["safety_guard"]["live_execution_authority"] is False
    assert payload["safety_guard"]["host_pressure_block_active"] is True
    assert payload["safety_guard"]["training_launch_allowed"] is False

    routes = {row["capability_id"]: row for row in payload["adaptive_policy_router"]["routes"]}
    assert routes["pressure_relief_control"]["action"] == "run_now"
    assert routes["runtime_throttle_control"]["action"] == "run_now"
    assert routes["memory_pressure_intelligence"]["action"] == "run_now"
    assert routes["writer_cycle_coordinator"]["action"] == "run_now"
    assert routes["training_runtime_control"]["action"] == "advisory_only"
    assert routes["infrastructure_autofix"]["action"] == "blocked_by_safety"
    assert routes["storage_backpressure_autopilot"]["action"] == "blocked_by_safety"
    assert routes["storage_backpressure_autopilot"]["blocked_by"] == ["safety_guard"]
    assert routes["pressure_relief_control"]["command"] == ["./scripts/ops/opsctl.sh", "pressure-relief", "--apply", "--json"]


def test_infrabot_adaptive_governor_surfaces_live_canary_readiness_bar(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    health = _base_ready_health(project_root)
    _write_json(
        health / "live_canary_readiness_contract_latest.json",
        {
            "overall_status": "blocked",
            "live_canary_money_ready": False,
            "ready_gate_count": 4,
            "gate_count": 7,
            "blockers": ["raw_profitability_posture_blocked", "sustained_window_not_met"],
            "infrastructure_message": "Before live canary money: no raw D-grade posture, no unexplained sleeve paper-trading dropouts, no auth/token surprises, no source mutation from runtime, clean CI, clean storage pressure, and clean promotion/paper gate freshness for a sustained window.",
        },
    )

    payload = infrabot_adaptive_governor.build_payload(project_root, max_actions=4)

    needs = {need["id"]: need for need in payload["system_needs_contract"]["needs"]}
    assert "live_canary_readiness_bar" in needs
    assert needs["live_canary_readiness_bar"]["severity"] == "critical"
    assert "live_canary_readiness_contract" in needs["live_canary_readiness_bar"]["target_capabilities"]

    routes = {row["capability_id"]: row for row in payload["adaptive_policy_router"]["routes"]}
    assert routes["live_canary_readiness_contract"]["action"] == "advisory_only"
    assert routes["live_canary_readiness_contract"]["command"] == ["./scripts/ops/opsctl.sh", "live-canary-readiness", "--apply", "--json"]


def test_infrabot_adaptive_governor_apply_writes_contracts_and_feedback(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    health = _base_ready_health(project_root)

    payload = infrabot_adaptive_governor.build_payload(project_root, apply=True)

    assert payload["apply_result"]["applied"] is True
    assert payload["apply_result"]["executed_commands"] == []
    assert payload["adaptive_policy_router"]["integration_contract"]["live_execution_authority"] is False
    assert "command_surface_hygiene" not in {need["id"] for need in payload["system_needs_contract"]["needs"]}
    for filename in [
        "infrabot_adaptive_governor_latest.json",
        "infrabot_system_needs_contract_latest.json",
        "infrabot_capability_registry_latest.json",
        "infrabot_adaptive_policy_latest.json",
        "infrabot_safety_guard_latest.json",
    ]:
        assert (health / filename).exists()
    feedback_rows = (health / "infrabot_adaptive_feedback.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(feedback_rows) == 1
    assert json.loads(feedback_rows[0])["event"] == "adaptive_governor_publish"


def test_infrabot_adaptive_governor_routes_livefeed_refresh_guard(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    health = _base_ready_health(project_root)
    _write_json(health / "livefeed_local_latest.json", {"status": "paused_runtime_pressure", "alive": True, "idle_heartbeat_seconds": 125, "skipped_files": 2})
    _write_json(health / "live_feed_heavy_view_latest.json", {"mode": "expired_or_closed"})

    payload = infrabot_adaptive_governor.build_payload(project_root)

    needs = {need["id"] for need in payload["system_needs_contract"]["needs"]}
    routes = {row["capability_id"]: row for row in payload["adaptive_policy_router"]["routes"]}
    assert "livefeed_continuity" in needs
    assert routes["livefeed_refresh_guard"]["action"] == "run_now"
    assert routes["livefeed_refresh_guard"]["command"] == ["./scripts/ops/opsctl.sh", "livefeed-refresh-guard", "--apply", "--json"]


def test_infrabot_adaptive_governor_routes_broker_auth_self_heal_before_paper_ramp(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    health = _base_ready_health(project_root)
    _write_json(health / "schwab_auth_supervisor_latest.json", {"overall_status": "blocked", "token": {"expires_in_seconds": 88}})
    _write_json(health / "auth_lease_manager_latest.json", {"overall_status": "blocked", "lease_state": "critical", "expires_in_seconds": 88})
    _write_json(health / "premarket_token_guard_latest.json", {"ok": False, "network_ok": True})
    _write_json(health / "broker_readiness_latest.json", {"ready_for_open": False, "auth_ok": False, "network_ok": True, "token_expires_in_seconds": 88})
    _write_json(health / "global_killswitch_latest.json", {"overall_status": "degraded", "clear_state": {"blockers": ["auth_lease_critical"]}})
    _write_json(health / "paper_400_ramp_latest.json", {"overall_status": "blocked", "stage": "blocked", "blockers": ["global_clear_blocker=auth_lease_critical"]})

    payload = infrabot_adaptive_governor.build_payload(project_root, max_actions=6)

    needs = {need["id"] for need in payload["system_needs_contract"]["needs"]}
    routes = {row["capability_id"]: row for row in payload["adaptive_policy_router"]["routes"]}
    assert "broker_auth_continuity" in needs
    assert routes["broker_auth_supervisor"]["action"] == "run_now"
    assert routes["broker_auth_supervisor"]["command"] == ["./scripts/ops/opsctl.sh", "schwab-auth-supervisor", "--apply", "--json"]
    assert routes["global_halt_refresh"]["action"] == "run_now"
    assert routes["paper_ramp_guard"]["action"] == "run_now"
    assert routes["runtime_paper_regression_guard"]["action"] == "advisory_only"


def test_infrabot_adaptive_governor_routes_cleanup_handoff_specialists(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    health = _base_ready_health(project_root)
    _write_json(
        health / "ingestion_storage_control_latest.json",
        {
            "overall_status": "ready",
            "pressure_index": 0.62,
            "backpressure": {"total_pending_lines": 5363},
            "storage": {"retention_debt_gb": 0, "backlog_drain_recommended_now": True},
            "storage_plane_contract": {"allowed_work": {"raw_training_compaction_apply": True}},
        },
    )
    _write_json(
        health / "external_backlog_drain_latest.json",
        {
            "overall_status": "drain_active",
            "recommended_now": True,
            "material_drain_recommended": True,
            "follow_through": {"status": "handoff_requested"},
        },
    )
    _write_json(
        health / "raw_training_compaction_intelligence_latest.json",
        {
            "overall_status": "needs_work",
            "raw_summary": {
                "training_candidate_count": 157,
                "training_candidate_gb": 4.812941,
                "compression_candidate_count": 12,
                "compression_candidate_gb": 1.25,
            },
        },
    )
    _write_json(
        health / "storage_retention_unison_latest.json",
        {
            "overall_status": "needs_work",
            "next_action": "clear hard blockers, then rerun storage-retention-unison --apply",
        },
    )

    payload = infrabot_adaptive_governor.build_payload(project_root, max_actions=8)

    needs = {need["id"] for need in payload["system_needs_contract"]["needs"]}
    routes = {row["capability_id"]: row for row in payload["adaptive_policy_router"]["routes"]}
    assert "cleanup_handoff_ingestion" in needs
    assert routes["external_backlog_drain_handoff"]["action"] == "run_now"
    assert routes["external_backlog_drain_handoff"]["command"] == [
        "./scripts/ops/opsctl.sh",
        "external-backlog-drain",
        "--apply",
        "--follow-through",
        "--poll-seconds",
        "5",
        "--wait-timeout-seconds",
        "45",
        "--json",
    ]
    assert routes["raw_training_cleanup_handoff"]["action"] == "run_now"
    assert routes["storage_retention_unison_handoff"]["action"] == "run_now"


def test_infrabot_adaptive_governor_does_not_route_storage_backpressure_when_overlay_clear(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    health = _base_ready_health(project_root)
    _write_json(
        health / "health_fast_latest.json",
        {
            "overall_status": "ready",
            "storage": {
                "severity": "stable",
                "pressure_index": 0.006,
                "backpressure": {
                    "core_pending_lines": 0,
                    "total_pending_lines": 1885,
                    "oldest_pending_age_seconds": 0.0,
                    "overlay_pressure_clear": True,
                },
            },
        },
    )
    _write_json(
        health / "ingestion_storage_control_latest.json",
        {
            "overall_status": "ready",
            "severity": "stable",
            "pressure_index": 0.006,
            "backpressure": {
                "total_pending_lines": 1885,
                "core_pending_lines": 0,
                "pending_lines_threshold": 15000,
                "oldest_pending_age_seconds": 0.0,
                "oldest_age_threshold_seconds": 240.0,
                "overlay_pressure_clear": True,
            },
            "storage": {"retention_debt_gb": 0.0},
        },
    )
    _write_json(health / "storage_backpressure_autopilot_latest.json", {"overall_status": "applied_with_followups"})

    payload = infrabot_adaptive_governor.build_payload(project_root)

    needs = {need["id"] for need in payload["system_needs_contract"]["needs"]}
    assert "storage_backpressure" not in needs


def test_infrabot_adaptive_governor_routes_paper_truth_watch_without_feedback_blocker(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    health = _base_ready_health(project_root)
    _write_json(
        health / "paper_execution_truth_layer_latest.json",
        {
            "ok": True,
            "overall_status": "watch",
            "failed_checks": [],
            "warnings": ["decision_replay_harness", "promotion_gate_hardening"],
        },
    )

    payload = infrabot_adaptive_governor.build_payload(project_root)

    needs = {need["id"] for need in payload["system_needs_contract"]["needs"]}
    routes = {row["capability_id"]: row for row in payload["adaptive_policy_router"]["routes"]}
    assert "paper_truth_watch_reconciliation" in needs
    assert "paper_feedback_quality" not in needs
    assert routes["paper_execution_truth_layer"]["action"] == "advisory_only"
    assert routes["paper_execution_truth_layer"]["needs"] == ["paper_truth_watch_reconciliation"]


def test_safe_repair_classifies_ok_protective_tightening_as_success() -> None:
    classification = infrabot_adaptive_governor._classify_command_outcome(
        0,
        {"ok": True, "overall_status": "protective_tightening"},
    )

    assert classification["outcome"] == "success"
    assert classification["success_like"] is True
    assert classification["retryable"] is False
