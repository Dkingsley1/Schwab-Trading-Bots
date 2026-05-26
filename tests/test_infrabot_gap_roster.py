import json
from pathlib import Path

from scripts.ops import infrabot_gap_roster
from scripts.ops import system_cleanliness_infrabot


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True), encoding="utf-8")


def test_infrabot_gap_roster_assigns_gap_bots(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    health = project_root / "governance" / "health"

    _write_json(
        health / "writer_cycle_coordinator_latest.json",
        {
            "overall_status": "waiting_for_writer",
            "writer_state_before": {
                "current_step": "complete",
                "complete_lock_handoff_needed": True,
                "child_writer_active": False,
            },
            "summary": {"completed_writer_lock_handoff_needed": True},
        },
    )
    _write_json(
        health / "operator_cockpit_latest.json",
        {
            "overall_status": "degraded",
            "recommended_actions": ["health_gates_stale"],
            "hardening_scorecard": {"self_auditing_bots_current": False},
            "surfaces": {
                "rolling_restart_controller": {"status": "blocked"},
                "blackstart_recovery": {"status": "blocked"},
                "chaos_drill_coordinator": {"status": "degraded"},
            },
        },
    )
    _write_json(health / "artifact_freshness_slo_latest.json", {"summary": {"stale_count": 2}})
    _write_json(health / "runtime_artifact_refresh_latest.json", {"overall_status": "degraded"})
    _write_json(health / "provider_mesh_latest.json", {"overall_status": "degraded", "soft_failures": ["sec_edgar_context"]})
    _write_json(health / "source_verification_latest.json", {"overall_status": "ready", "overall": {"unverified_sources": ["market_quote_profiles"]}})
    _write_json(health / "paper_profitability_control_latest.json", {"overall_status": "protective_tightening", "profitability_grade": "D"})
    _write_json(health / "paper_performance_latest.json", {"overall_status": "ready"})
    _write_json(health / "system_cleanliness_infrabot_latest.json", {"blocked_layers": ["promotion_replay"]})
    _write_json(health / "promotion_quality_gate_latest.json", {"overall_status": "blocked", "ok": False})
    _write_json(
        health / "bot_needs_intelligence_latest.json",
        {"need_counts": {"collect_more_data": 7, "targeted_quality_retrain": 2, "top_off_walk_forward_runs": 1}},
    )
    _write_json(health / "training_data_intake_expansion_latest.json", {"collect_first_count": 5})
    _write_json(health / "all_sleeves_launcher_latest.json", {"overall_status": "ready"})
    _write_json(health / "system_self_model_latest.json", {"overall_status": "degraded"})
    _write_json(health / "master_infrastructure_supervisor_latest.json", {"overall_status": "blocked"})
    _write_json(
        health / "pressure_relief_control_latest.json",
        {"host_saturation_score": 62.0, "compute_pressure_level": "normal", "memory_pressure_level": "normal"},
    )
    _write_json(health / "memory_efficiency_control_latest.json", {"overall_status": "ready"})
    _write_json(health / "host_capability_contract_latest.json", {"body_map": {"storage_layout": {"protected_volumes": []}}, "adapters": {"protected_storage": {"denylist": []}}})
    _write_json(health / "codex_project_guard_latest.json", {"workspace_boundary": {"blocked_volume": ""}})
    _write_json(
        health / "training_runtime_control_latest.json",
        {
            "overall_status": "constrained",
            "training_launch_contract": {"launch_allowed": False, "launch_blockers": ["runtime_snapshot_not_fresh"]},
            "host_training_headroom_gate": {"batch_cap": 1, "batch10_training_safe": False, "batch20_training_safe": False},
        },
    )
    _write_json(health / "paper_execution_backlog_relief_latest.json", {"ok": False, "pending_rows_after": 12})
    _write_json(health / "paper_reconciliation_slo_latest.json", {"ok": False, "overall_status": "degraded"})
    _write_json(health / "bot_quality_autopilot_latest.json", {"overall_status": "needs_work", "recommended_actions": ["compress duplicates"]})
    _write_json(health / "training_quality_control_latest.json", {"targeted_actions": {"quality_probation_bot_ids": ["brain_refinery_v42"]}})
    _write_json(health / "livefeed_local_latest.json", {"status": "paused_runtime_pressure", "alive": True, "idle_heartbeat_seconds": 25})
    _write_json(health / "live_feed_heavy_view_latest.json", {"mode": "expired_or_closed"})
    _write_json(
        health / "auth_lease_manager_latest.json",
        {"overall_status": "ready", "lease_budget": {"expires_in_seconds": 500, "min_lease_seconds": 1200}},
    )
    _write_json(health / "schwab_auth_supervisor_latest.json", {"overall_status": "ready", "token": {"refresh_needed": False, "expires_in_seconds": 500}})
    _write_json(
        health / "market_move_explainer_latest.json",
        {"overall_status": "thin", "symbol": "BTC", "primary_confidence": 0.62, "symbol_evidence_count": 0, "source_coverage": {"market_micro": True}, "unknowns": ["thin"]},
    )

    payload = infrabot_gap_roster.build_payload(project_root)

    assert payload["bot_count"] == 16
    assert payload["active_count"] == 16
    assert payload["overall_status"] == "blocked"
    names = payload["assigned_infrabots"]
    assert "writer_lock_handoff_infrabot" in names
    assert "health_truth_reconciler_infrabot" in names
    assert "provider_cross_verification_infrabot" in names
    assert "paper_feedback_repair_infrabot" in names
    assert "promotion_replay_gate_infrabot" in names
    assert "bot_data_labeling_targeter_infrabot" in names
    assert "recovery_drill_infrabot" in names
    assert "self_audit_freshness_infrabot" in names
    assert "cotenant_headroom_guard_infrabot" in names
    assert "protected_volume_boundary_infrabot" in names
    assert "training_batch_readiness_infrabot" in names
    assert "paper_execution_queue_reconciler_infrabot" in names
    assert "duplicate_alpha_compression_infrabot" in names
    assert "livefeed_mirror_continuity_infrabot" in names
    assert "auth_lease_preflight_infrabot" in names
    assert "market_explanation_evidence_infrabot" in names
    assert payload["integration_contract"]["live_execution_authority"] is False
    assert "/Volumes/VIDEO" in payload["integration_contract"]["protected_volume_denylist"]


def test_system_cleanliness_infrabot_delegates_gap_roster(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    health = project_root / "governance" / "health"
    champion = project_root / "governance" / "champion_challenger"

    _write_json(health / "ingestion_storage_control_latest.json", {"overall_status": "ready", "pressure_index": 0.0, "storage": {"retention_debt_gb": 0.0}})
    _write_json(health / "collector_contracts_latest.json", {"required_failures": [], "soft_failures": []})
    _write_json(health / "source_verification_latest.json", {"overall_status": "ready", "overall": {"unverified_sources": []}})
    _write_json(health / "training_quality_control_latest.json", {"overall_status": "ready", "rollout": {"considered_bots": 4, "min_considered_bots": 4}})
    _write_json(health / "replay_hash_registry_guard_latest.json", {"ok": True})
    _write_json(health / "golden_replay_regression_latest.json", {"ok": True})
    _write_json(champion / "promotion_autopilot_packet_latest.json", {"promotion_ready": True})
    _write_json(health / "writer_cycle_coordinator_latest.json", {"overall_status": "ready", "writer_state_before": {"complete_lock_handoff_needed": False}})
    _write_json(health / "operator_cockpit_latest.json", {"overall_status": "ready", "hardening_scorecard": {"self_auditing_bots_current": True}, "surfaces": {}})
    _write_json(health / "provider_mesh_latest.json", {"overall_status": "ready"})
    _write_json(health / "paper_profitability_control_latest.json", {"overall_status": "ready", "profitability_grade": "B"})
    _write_json(health / "paper_performance_latest.json", {"overall_status": "ready"})
    _write_json(health / "promotion_quality_gate_latest.json", {"overall_status": "ready", "ok": True})
    _write_json(health / "bot_needs_intelligence_latest.json", {"need_counts": {}})
    _write_json(health / "training_data_intake_expansion_latest.json", {"collect_first_count": 0})
    _write_json(health / "all_sleeves_launcher_latest.json", {"overall_status": "ready"})
    _write_json(health / "system_self_model_latest.json", {"overall_status": "ready"})
    _write_json(health / "master_infrastructure_supervisor_latest.json", {"overall_status": "ready"})
    _write_json(health / "pressure_relief_control_latest.json", {"host_saturation_score": 20.0, "compute_pressure_level": "normal", "memory_pressure_level": "normal"})
    _write_json(health / "memory_efficiency_control_latest.json", {"overall_status": "ready"})
    _write_json(
        health / "host_capability_contract_latest.json",
        {
            "body_map": {"storage_layout": {"protected_volumes": ["/Volumes/VIDEO"]}},
            "adapters": {"protected_storage": {"denylist": ["/Volumes/VIDEO"]}},
        },
    )
    _write_json(
        health / "training_runtime_control_latest.json",
        {
            "overall_status": "ready",
            "training_launch_contract": {"launch_allowed": True, "launch_blockers": [], "prep_blockers": []},
            "host_training_headroom_gate": {"batch_cap": 20, "batch10_training_safe": True, "batch20_training_safe": True},
        },
    )
    _write_json(health / "paper_execution_backlog_relief_latest.json", {"ok": True, "pending_rows_after": 0})
    _write_json(health / "paper_reconciliation_slo_latest.json", {"ok": True, "overall_status": "ready"})
    _write_json(health / "bot_quality_autopilot_latest.json", {"overall_status": "ready", "recommended_actions": []})
    _write_json(health / "livefeed_local_latest.json", {"status": "running", "alive": True, "idle_heartbeat_seconds": 15})
    _write_json(health / "live_feed_heavy_view_latest.json", {"mode": "active"})
    _write_json(
        health / "auth_lease_manager_latest.json",
        {"overall_status": "ready", "lease_budget": {"expires_in_seconds": 3600, "min_lease_seconds": 1200}},
    )
    _write_json(health / "schwab_auth_supervisor_latest.json", {"overall_status": "ready", "token": {"refresh_needed": False, "expires_in_seconds": 3600}})
    _write_json(
        health / "market_move_explainer_latest.json",
        {
            "overall_status": "ready",
            "symbol": "BTC",
            "primary_confidence": 0.81,
            "symbol_evidence_count": 2,
            "source_coverage": {"market_micro": True, "crypto_market": True},
            "unknowns": [],
        },
    )

    payload = system_cleanliness_infrabot.build_payload(project_root)

    assert payload["overall_status"] == "ready"
    assert payload["gap_roster_active_count"] == 0
    assert "infrabot_gap_roster" in payload["assigned_scope"]
    assert "protected_volume_boundary_infrabot" in payload["supervision_contract"]["delegated_infrabots"]
    assert "training_batch_readiness_infrabot" in payload["supervision_contract"]["delegated_infrabots"]
