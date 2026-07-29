import json
from pathlib import Path

from scripts.ops import infrabot_adaptive_governor


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True), encoding="utf-8")


def _raw_profitability_fixture() -> dict:
    return {
        "raw_profitability_grade": "D",
        "controlled_profitability_grade": "A+",
        "financial_profitability_grade": "D",
        "a_plus_target_contract": {
            "combined_a_plus_ready": False,
            "current": {
                "net_pnl": -17908.060398,
                "realized_pnl": -4853.579587,
                "unrealized_pnl": -13054.480811,
                "change_vs_previous_day": 335.377531,
                "executions": 596,
                "weak_profile_count": 25,
                "strategy_control_count": 24,
                "unprotected_weak_profile_count": 0,
                "unprotected_strategy_control_count": 0,
            },
            "thresholds": {"min_net_pnl": 50000.0},
        },
        "raw_profitability_improvement_contract": {
            "requirements": [
                {"id": "1_weak_sleeves_zero_new_entries", "ready": True},
                {"id": "2_strict_clean_sleeve_admission", "ready": True},
                {"id": "3_position_harvest_evidence_layer", "ready": True},
                {"id": "4_position_level_paper_telemetry", "ready": True},
                {"id": "5_loss_cause_training_feedback", "ready": True},
                {"id": "6_losing_strategy_pair_quarantine", "ready": True},
                {"id": "7_raw_recovery_burn_down_guard", "ready": True},
            ],
            "loss_cause_training_feedback_contract": {
                "top_loss_causes": [
                    {"cause": "conflict:low", "count": 25},
                    {"cause": "event_proximity:low", "count": 25},
                    {"cause": "fill_quality:unknown", "count": 25},
                    {"cause": "source_quality:low", "count": 25},
                ],
            },
            "burn_down_contract": {
                "required_average_daily_net_improvement": 596.935347,
                "top_drag_profiles": [
                    {"profile": "bond", "net_pnl_total": -7413.988011},
                    {"profile": "aggressive", "net_pnl_total": -7118.301148},
                ],
            },
            "runtime_enforcement": {
                "block_new_entries_on_weak_profiles": True,
                "keep_sells_and_reduce_only_paths_open": True,
                "feed_loss_causes_to_training": True,
                "paper_only": True,
                "live_execution_allowed": False,
            },
        },
    }


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


def _write_self_healing_playbook_fixture(health: Path) -> None:
    lanes = {
        "storage_writer": ("storage_writer.self_heal", 1, ["governance/health/ingestion_storage_control_latest.json"]),
        "runtime_memory": ("runtime_memory.self_heal", 2, ["governance/health/runtime_throttle_control_latest.json"]),
        "raw_profitability_recovery": ("raw_profitability_recovery.self_heal", 2, ["governance/health/paper_profitability_control_latest.json"]),
        "source_truth": ("source_truth.self_heal", 2, ["governance/health/source_verification_latest.json"]),
        "governance_regression": ("governance_regression.self_heal", 2, ["governance/health/grade_regression_guard_latest.json"]),
        "auth_live_lock": ("auth_live_lock.self_heal", 1, ["governance/health/production_readiness_control_latest.json"]),
    }
    playbooks = []
    for lane, (playbook_id, max_attempts, proof_artifacts) in lanes.items():
        playbooks.append(
            {
                "playbook_id": playbook_id,
                "lane": lane,
                "primary_capability": lane,
                "owner_command": ["./scripts/ops/opsctl.sh", "system-needs", "--json"],
                "verify_command": ["./scripts/ops/opsctl.sh", "system-needs", "--json"],
                "proof_artifacts": proof_artifacts,
                "max_attempts_per_incident": max_attempts,
                "cooldown_seconds": 300,
                "hold_condition": "hold visible and escalate when retry budget is exhausted",
                "authority_boundary": "safe_repair_or_read_only_no_live_execution_no_dependency_mutation",
                "live_execution_authority": False,
                "dependency_mutation_authority": False,
                "complete": True,
            }
        )
    _write_json(
        health / "infrabot_library_self_awareness_control_latest.json",
        {
            "overall_status": "ready",
            "ok": True,
            "self_healing_playbooks": {
                "enabled": True,
                "grade": "A+",
                "playbook_count": len(playbooks),
                "complete_playbook_count": len(playbooks),
                "all_playbooks_complete": True,
                "all_lanes_have_playbooks": True,
                "all_needs_have_playbooks": True,
                "authority_safe": True,
                "live_execution_authority": False,
                "dependency_mutation_authority": False,
                "playbooks": playbooks,
            },
        },
    )


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
    assert "production_quality_control" in needs["live_canary_readiness_bar"]["target_capabilities"]
    assert "production_quality_slo_guard" in needs["live_canary_readiness_bar"]["target_capabilities"]
    assert "paper_performance_refresh" in needs["live_canary_readiness_bar"]["target_capabilities"]
    assert "daily_verify_auto_remediation" in needs["live_canary_readiness_bar"]["target_capabilities"]

    routes = {row["capability_id"]: row for row in payload["adaptive_policy_router"]["routes"]}
    assert routes["live_canary_readiness_contract"]["action"] == "advisory_only"
    assert routes["live_canary_readiness_contract"]["command"] == ["./scripts/ops/opsctl.sh", "live-canary-readiness", "--apply", "--json"]
    assert routes["production_quality_control"]["action"] == "run_now"
    assert routes["production_quality_control"]["command"] == ["./scripts/ops/opsctl.sh", "production-quality", "--apply", "--refresh-contract", "--json"]
    assert routes["production_quality_slo_guard"]["action"] == "run_now"
    assert routes["production_quality_slo_guard"]["command"] == ["./scripts/ops/opsctl.sh", "production-quality-slo", "--apply", "--refresh-quality", "--json"]


def test_infrabot_adaptive_governor_routes_raw_profitability_burn_down(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    health = _base_ready_health(project_root)
    raw_fixture = _raw_profitability_fixture()
    _write_json(
        health / "paper_profitability_control_latest.json",
        {
            "overall_status": "ready",
            "profitability_grade": "A+",
            "raw_profitability_grade": "D",
            "controlled_profitability_grade": "A+",
            "financial_profitability_grade": "D",
            "low_grade_layer_summary": {"control_posture_grade": "A+", "active_blocker_count": 0},
        },
    )
    _write_json(health / "paper_runtime_profitability_controls_latest.json", raw_fixture)
    _write_json(
        health / "live_canary_readiness_contract_latest.json",
        {
            "overall_status": "blocked",
            "live_canary_money_ready": False,
            "blockers": ["raw_profitability_posture_blocked"],
        },
    )

    payload = infrabot_adaptive_governor.build_payload(project_root, max_actions=8)

    needs = {need["id"]: need for need in payload["system_needs_contract"]["needs"]}
    routes = {row["capability_id"]: row for row in payload["adaptive_policy_router"]["routes"]}
    raw_need = needs["raw_profitability_burn_down"]
    assert raw_need["severity"] == "critical"
    assert "master_grandmaster_profitability_trainer" in raw_need["target_capabilities"]
    assert "training_data_intake_labeling" in raw_need["target_capabilities"]
    assert "raw_net_pnl=-17908.060398" in raw_need["evidence"]
    assert "top_loss_causes=conflict:low,event_proximity:low,fill_quality:unknown,source_quality:low" in raw_need["evidence"]
    assert routes["paper_profitability_control"]["action"] == "run_now"
    assert routes["master_grandmaster_profitability_trainer"]["action"] == "run_now"
    assert routes["master_grandmaster_profitability_trainer"]["command"] == [
        "./scripts/ops/opsctl.sh",
        "master-grandmaster-train",
        "--apply",
        "--json",
    ]


def test_infrabot_adaptive_governor_routes_source_quality_to_health_gate_recheck(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    health = _base_ready_health(project_root)
    _write_json(
        health / "source_verification_latest.json",
        {
            "overall_status": "degraded",
            "overall": {
                "unverified_sources": ["ticker_news_context"],
                "stale_sources": ["ticker_news_context"],
            },
        },
    )

    payload = infrabot_adaptive_governor.build_payload(project_root, max_actions=8)

    needs = {need["id"]: need for need in payload["system_needs_contract"]["needs"]}
    assert "source_quality" in needs
    assert "health_gates_recheck" in needs["source_quality"]["target_capabilities"]
    assert "health_gates_recheck" in {
        row["id"] for row in payload["capability_registry"]["capabilities"]
    }


def test_infrabot_adaptive_governor_surfaces_production_quality_slo_breach(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    health = _base_ready_health(project_root)
    _write_json(
        health / "production_quality_slo_guard_latest.json",
        {
            "overall_status": "blocked",
            "breach_count": 1,
            "warning_count": 0,
            "breached_lanes": [{"lane_id": "storage_pressure_clean"}],
            "warning_lanes": [],
        },
    )

    payload = infrabot_adaptive_governor.build_payload(project_root, max_actions=4)

    needs = {need["id"]: need for need in payload["system_needs_contract"]["needs"]}
    routes = {row["capability_id"]: row for row in payload["adaptive_policy_router"]["routes"]}
    assert "production_quality_slo_breach" in needs
    assert needs["production_quality_slo_breach"]["severity"] == "critical"
    assert "production_quality_slo_guard" in needs["production_quality_slo_breach"]["target_capabilities"]
    assert routes["production_quality_slo_guard"]["action"] == "run_now"


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


def test_infrabot_adaptive_governor_routes_include_self_healing_playbooks(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    health = _base_ready_health(project_root)
    _write_self_healing_playbook_fixture(health)
    _write_json(
        health / "ingestion_storage_control_latest.json",
        {"overall_status": "degraded", "backpressure": {"total_pending_lines": 15000}, "storage": {"retention_debt_gb": 1.0}},
    )
    _write_json(
        health / "live_canary_readiness_contract_latest.json",
        {
            "overall_status": "blocked",
            "live_canary_money_ready": False,
            "blockers": ["paper_ramp_not_ready"],
        },
    )

    payload = infrabot_adaptive_governor.build_payload(project_root, max_actions=8)

    safety_contract = payload["safety_guard"]["self_healing_playbook_contract"]
    routes = {row["capability_id"]: row for row in payload["adaptive_policy_router"]["routes"]}
    assert safety_contract["enabled"] is True
    assert safety_contract["grade"] == "A+"
    assert safety_contract["authority_safe"] is True
    assert payload["adaptive_policy_router"]["integration_contract"]["uses_self_healing_playbooks"] is True

    storage_route = routes["storage_backpressure_autopilot"]
    assert storage_route["self_healing"]["lane"] == "storage_writer"
    assert storage_route["self_healing"]["playbook_id"] == "storage_writer.self_heal"
    assert storage_route["self_healing"]["max_attempts_per_incident"] == 1
    assert storage_route["self_healing"]["cooldown_seconds"] == 300
    assert storage_route["self_healing"]["proof_artifacts"] == ["governance/health/ingestion_storage_control_latest.json"]
    assert storage_route["self_healing"]["hold_condition"]
    assert storage_route["self_healing"]["contract_ready"] is True
    assert routes["paper_ramp_guard"]["self_healing"]["lane"] == "auth_live_lock"
    assert routes["paper_ramp_guard"]["self_healing"]["playbook_id"] == "auth_live_lock.self_heal"
    assert routes["live_canary_readiness_contract"]["self_healing"]["lane"] == "auth_live_lock"


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
    _write_json(
        health / "storage_quota_guard_latest.json",
        {
            "overall_status": "degraded",
            "quota_summary": {
                "hard_breaches": 0,
                "soft_breaches": 1,
                "degraded_families": ["sql_link_shards"],
            },
        },
    )

    payload = infrabot_adaptive_governor.build_payload(project_root, max_actions=10)

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
    assert routes["deep_cold_second_cold_handoff"]["action"] == "run_now"
    assert routes["storage_retention_unison_handoff"]["action"] == "run_now"
    assert routes["stateful_sql_quota_relief"]["action"] == "run_now"


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


def test_self_healing_retry_budget_uses_playbook_and_ignores_blocked_no_apply(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    state: dict = {"capabilities": {}}
    healing_context = {
        "lane": "storage_writer",
        "playbook_id": "storage_writer.self_heal",
        "max_attempts_per_incident": 1,
        "cooldown_seconds": 300,
        "hold_condition": "hold visible and escalate when retry budget is exhausted",
    }
    command = ["./scripts/ops/opsctl.sh", "storage-backpressure-autopilot", "--apply", "--quick-bounded", "--json"]

    blocked = infrabot_adaptive_governor._classify_command_outcome(0, {"overall_status": "blocked"})
    blocked_state = infrabot_adaptive_governor._update_self_healing_state(
        project_root,
        state,
        cap_id="storage_backpressure_autopilot",
        command=command,
        classification=blocked,
        returncode=0,
        healing_context=healing_context,
    )
    assert blocked_state["failure_count"] == 0
    assert blocked_state["retry_budget_exhausted"] is False

    failed = infrabot_adaptive_governor._classify_command_outcome(1, {})
    failed_state = infrabot_adaptive_governor._update_self_healing_state(
        project_root,
        state,
        cap_id="storage_backpressure_autopilot",
        command=command,
        classification=failed,
        returncode=1,
        healing_context=healing_context,
    )
    assert failed_state["failure_count"] == 1
    assert failed_state["retry_budget_exhausted"] is True
    state["capabilities"]["storage_backpressure_autopilot"]["cooldown_until_utc"] = ""

    gate = infrabot_adaptive_governor._self_healing_execution_gate(
        state,
        "storage_backpressure_autopilot",
        infrabot_adaptive_governor._utc_now(),
        healing_context,
    )
    assert gate["active"] is True
    assert gate["gate"] == "retry_budget"
    assert gate["reason"] == "self_healing_retry_budget_exhausted"
