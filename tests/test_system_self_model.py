from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from scripts.ops import system_self_model as src


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def test_system_self_model_builds_awareness_domains_and_optimizations(tmp_path: Path) -> None:
    rows = [
        {
            "bot_id": f"brain_refinery_v{i}",
            "active": True,
            "data_collection_active": i >= 2,
            "training_excluded": i >= 2,
            "lifecycle_state": "data_collection_only" if i >= 2 else "active",
            "sleeve_profile": f"sleeve_{i % 3}",
            "capability_pack_slug": "test_pack" if i >= 2 else "",
        }
        for i in range(1, 8)
    ]
    _write_json(tmp_path / "master_bot_registry.json", {"summary": {}, "sub_bots": rows})
    health = tmp_path / "governance" / "health"
    _write_json(
        health / "memory_efficiency_control_latest.json",
        {
            "overall_status": "ready",
            "recommended_profile": "pro_balanced",
            "cotenant_awareness": {
                "mode": "managed_cotenant",
                "open_apps": ["PyCharm", "Chrome"],
                "memory_pressure_clear": True,
                "storage_pressure_clear": True,
            },
            "memory_snapshot": {"memory_pressure_state": "green", "memory_pressure_kind": "none", "swap_used_gb": 0.2},
            "storage_snapshot": {"pressure_index": 0.01},
            "expansion_session": {"pressure_level": "normal", "sleeve_profile_count": 3},
        },
    )
    _write_json(
        health / "runtime_throttle_control_latest.json",
        {
            "overall_status": "ready",
            "memory_pressure_level": "normal",
            "cpu_pressure_level": "watch",
            "host_saturation_score": 52.0,
            "throttle_profile": "observe",
        },
    )
    _write_json(health / "ingestion_storage_control_latest.json", {"overall_status": "ready", "severity": "stable", "pressure_index": 0.01})
    _write_json(
        health / "backpressure_drainer_fleet_latest.json",
        {
            "overall_status": "ready",
            "ready_drainer_count": 1,
            "active_drainer": {"name": "core_decision_drainer", "status": "ready", "live_window_safe": True},
        },
    )
    _write_json(
        health / "backpressure_super_drainer_latest.json",
        {
            "overall_status": "idle",
            "target_met_final": True,
            "active_drainer": "core_decision_drainer",
            "ready_drainer_names": ["core_decision_drainer"],
            "settings": {"target_pending_lines": 5000, "planned_wave_count": 0},
            "summary": {"final_pending_lines": 100, "waves_run": 0, "progress_waves": 0, "stop_reason": "target_already_met"},
            "guardrails": {"single_writer_only": True, "starts_parallel_sql_writers": False},
            "assigned_infrabots": ["backpressure_super_drainer"],
            "grandmaster_context_packet": {"active_drainer": "core_decision_drainer"},
        },
    )
    _write_json(health / "writer_cycle_coordinator_latest.json", {"overall_status": "ready", "summary": {"writer_active_after_wait": False}})
    _write_json(
        health / "writer_process_intelligence_latest.json",
        {
            "overall_status": "ready",
            "decision_packet": {
                "action": "park_writer_and_observe",
                "confidence": 0.78,
                "expanded_writer_lane_count": 25,
                "hot_lane_count": 6,
                "warm_lane_count": 9,
                "cold_lane_count": 10,
                "risk_flags": [],
            },
            "writer_health": {"state": "idle", "active": False, "progress_age_minutes": 0.0},
            "process_topology": {"duplicate_sql_writer_processes": False},
            "safety_envelope": {
                "single_writer_only": True,
                "starts_parallel_sql_writers": False,
                "max_parallel_sql_writers": 1,
            },
            "process_playbook": [],
        },
    )
    _write_json(
        health / "whole_system_intelligence_latest.json",
        {
            "overall_status": "ready",
            "system_signal_bus": {
                "overall_status": "ready",
                "summary": {"signal_count": 22, "loaded_signal_count": 18, "top_risk": "none"},
            },
            "system_brain": {
                "overall_status": "ready",
                "decision_packet": {
                    "action": "observe_and_expand_cautiously",
                    "operating_mode": "steady_state",
                    "confidence": 0.78,
                    "top_risk": "none",
                    "safe_next_command": ["./scripts/ops/opsctl.sh", "system-intelligence", "--json"],
                    "risk_flags": [],
                    "do_not_do": ["do_not_start_parallel_sql_writers"],
                },
            },
            "system_process_contracts": {"overall_status": "ready", "contract_count": 7, "blocked_contract_count": 0},
            "system_self_intelligence": {
                "overall_status": "ready",
                "trend": {"trajectory": "flat", "pending_lines_delta": 0, "pressure_index_delta": 0.0},
                "uncertainty": {"level": "low", "score": 5, "missing_signals": [], "stale_signals": [], "conflicting_signals": [], "contract_violations": []},
                "learning_memory": {"same_action_repeat_count": 1},
                "action_effectiveness": {"verdict": "effective", "same_action_run_length": 2},
                "causal_diagnosis": {"primary_root_cause": "stable_or_observing", "confidence": 0.82},
                "integration_routing": {"route_mode": "observe_and_refresh", "primary_owner": "system_brain"},
                "capability_gaps": [],
                "reflex": {"action": "follow_system_brain", "blocks_brain_action_until_refreshed": False},
                "self_questions": ["No blocking self-question; continue monitoring outcome after the next safe action."],
            },
            "codex_handoff": {
                "overall_status": "ready",
                "attention_packet": {"needs_codex": ["observe_current_state_and_continue_safe_expansion"]},
                "communication_contract": {"proactive_delivery_to_codex": False},
            },
        },
    )
    _write_json(
        health / "system_self_intelligence_latest.json",
        {
            "overall_status": "ready",
            "trend": {"trajectory": "flat", "pending_lines_delta": 0, "pressure_index_delta": 0.0},
            "uncertainty": {"level": "low", "score": 5, "missing_signals": [], "stale_signals": [], "conflicting_signals": [], "contract_violations": []},
            "learning_memory": {"same_action_repeat_count": 1},
            "action_effectiveness": {"verdict": "effective", "same_action_run_length": 2},
            "causal_diagnosis": {"primary_root_cause": "stable_or_observing", "confidence": 0.82},
            "integration_routing": {"route_mode": "observe_and_refresh", "primary_owner": "system_brain"},
            "capability_gaps": [],
            "reflex": {"action": "follow_system_brain", "blocks_brain_action_until_refreshed": False},
            "self_questions": ["No blocking self-question; continue monitoring outcome after the next safe action."],
        },
    )
    _write_json(
        health / "codex_operator_bridge_latest.json",
        {
            "overall_status": "advisory",
            "attention_packet": {
                "needs_codex": ["review_trade_digest"],
                "active_blockers": ["host_training_headroom_not_clear"],
                "safe_next_commands": [["./scripts/ops/opsctl.sh", "training-runtime-control", "--json"]],
                "communication_contract": {"delivery_channel": "artifact_handoff", "proactive_delivery_to_codex": True},
            },
            "sections": {
                "paper_trading": {"day": {"day_utc": "20260614", "executions": 120, "ending_net_pnl_total": 12.5, "change_vs_previous_day": 8.0}},
                "training": {"launch_allowed": False, "recommended_batch_size": 0, "launch_blockers": ["host_training_headroom_not_clear"]},
                "writer": {"active": True, "completed_shard_count": 2, "planned_shard_count": 4},
                "memory": {"classification": "soft_guard", "safe_for_training": False},
                "livefeed": {"alive": True},
            },
        },
    )
    _write_json(health / "storage_backpressure_autopilot_latest.json", {"overall_status": "ready", "metrics": {"backpressure_actionable": False}})
    _write_json(
        health / "mlx_intelligence_router_latest.json",
        {
            "overall_status": "ready",
            "library_coverage": {"coverage_ratio": 1.0, "missing_count": 0},
            "route_coverage": {"route_coverage_ratio": 1.0, "blocked_lane_count": 0},
            "library_utilization_matrix": {"mapped_library_ratio": 1.0},
            "runtime_caps": {
                "profile": "foreground_safe",
                "max_concurrent_mlx_jobs": 2,
                "compile_mode": "direct_stable",
                "heavy_vlm_enabled": True,
                "cpu_pressure_level": "watch",
                "memory_pressure_level": "normal",
                "host_saturation_score": 52.0,
                "host_pressure_state": "foreground_safe",
            },
            "control_contract": {"safe_utilization_goal": "100_percent_library_coverage_with_cpu_memory_aware_caps"},
        },
    )
    _write_json(
        health / "library_utilization_router_latest.json",
        {
            "overall_status": "ready",
            "coverage": {
                "managed_non_mlx_package_count": 80,
                "locked_non_mlx_package_count": 75,
                "coverage_ratio": 1.0,
                "locked_runtime_ok_ratio": 1.0,
                "missing_runtime_count": 0,
                "version_mismatch_count": 0,
            },
            "library_utilization_matrix": {"mapped_package_ratio": 1.0},
            "runtime_caps": {"profile": "foreground_safe"},
            "control_contract": {
                "safe_utilization_goal": "100_percent_non_mlx_library_lane_coverage_with_runtime_caps",
                "default_ml_backend": "mlx",
                "portable_ml_policy": "pytorch_onnx_transformers_stay_canary_or_off_hours_when_live_collection_is_active",
            },
        },
    )
    _write_json(health / "global_killswitch_latest.json", {"halt": False, "action": "none", "reasons": [], "clear_ready": True, "clear_blockers": []})
    _write_json(health / "process_watchdog_latest.json", {"overall_status": "ready", "status": []})
    _write_json(health / "auth_lease_manager_latest.json", {"overall_status": "ready", "lease_state": "healthy", "lease_budget": {"expires_in_seconds": 2400, "min_lease_seconds": 1200}, "broker_state": {"auth_ok": True}})
    _write_json(health / "data_plane_recovery_controller_latest.json", {"overall_status": "ready", "recovery_state": "ready", "queue_depth": 0})
    _write_json(health / "live_runtime_separation_control_latest.json", {"overall_status": "ready", "live_plane": {"live_lane_running": True}})
    _write_json(
        health / "operator_cockpit_latest.json",
        {
            "overall_status": "ready",
            "adaptive_posture": {"hard_blockers": [], "pressure_level": "normal"},
            "hardening_scorecard": {"process_ownership_canonical": True},
            "recommended_actions": ["watch queue"],
        },
    )
    _write_json(health / "core_bot_materialization_guard_latest.json", {"overall_status": "ready", "summary": {"missing_core_module_count": 0, "duplicate_core_version_count": 0}})
    _write_json(tmp_path / "governance" / "alerts" / "incident_auto_halt_latest.json", {"overall_status": "ready", "event": "none"})
    scripts_ops = tmp_path / "scripts" / "ops"
    scripts_ops.mkdir(parents=True, exist_ok=True)
    (scripts_ops / "mlx_intelligence_router.py").write_text(
        "LANE_SPECS = []\nlibrary_utilization_matrix = {}\nrecommended_runtime_env = {}\ncpu_pressure_level = 'watch'\nhost_pressure_state = 'foreground_safe'\nhost_saturation_score = 52.0\n100_percent_library_coverage_with_cpu_memory_aware_caps = True\n",
        encoding="utf-8",
    )
    (scripts_ops / "library_utilization_router.py").write_text(
        "LANE_SPECS = []\nlibrary_utilization_matrix = {}\nLIBRARY_DEFAULT_ML_BACKEND = 'mlx'\n",
        encoding="utf-8",
    )
    (scripts_ops / "system_self_model.py").write_text(
        "_host_pressure_intelligence\ncpu_pressure_level\nhost_pressure_state\nhost_saturation_score\n",
        encoding="utf-8",
    )
    (scripts_ops / "backpressure_super_drainer.py").write_text(
        "self_intelligence_contract = {}\ndrainer_strategy = {}\ngrandmaster_context_packet = {}\nwriter_cycle_coordinator.py\n",
        encoding="utf-8",
    )
    (scripts_ops / "writer_process_intelligence.py").write_text(
        "writer_expansion_contract = {}\nwriter_process_intelligence\nsingle_writer_only\n",
        encoding="utf-8",
    )
    (scripts_ops / "writer_cycle_coordinator.py").write_text("writer_process_intelligence\n", encoding="utf-8")
    (scripts_ops / "sql_link_shard_manager.py").write_text("writer_progress\nadmission_evidence\n", encoding="utf-8")
    (scripts_ops / "system_intelligence_coordinator.py").write_text(
        "system_signal_bus\nsystem_brain\nsystem_process_contracts\nsystem_self_intelligence\nself_intelligence_contract\ncausal_diagnosis\naction_effectiveness\nintegration_routing\ncodex_handoff\nglobal_safety_contract\n",
        encoding="utf-8",
    )
    (scripts_ops / "opsctl.sh").write_text(
        "mlx-intelligence-router\nlibrary-utilization-router\nbackpressure-super-drainer\nwriter-process-intelligence\nsystem-intelligence\nrun_then_refresh_self_model\n",
        encoding="utf-8",
    )

    payload = src.build_payload(tmp_path)

    assert payload["overall_status"] == "ready"
    assert payload["identity"]["total_bots"] == 7
    assert payload["identity"]["data_collection_active_bots"] == 6
    assert payload["awareness_domains"]["resource_awareness"]["status"] == "advisory"
    assert payload["awareness_domains"]["host_pressure_intelligence"]["status"] == "advisory"
    assert payload["awareness_domains"]["host_pressure_intelligence"]["cpu_pressure_level"] == "watch"
    assert payload["awareness_domains"]["host_pressure_intelligence"]["recommended_intelligence_posture"] == "foreground_safe"
    assert payload["awareness_domains"]["mlx_intelligence_awareness"]["status"] == "ready"
    assert payload["awareness_domains"]["mlx_intelligence_awareness"]["library_coverage_ratio"] == 1.0
    assert payload["awareness_domains"]["library_utilization_awareness"]["status"] == "ready"
    assert payload["awareness_domains"]["library_utilization_awareness"]["mapped_package_ratio"] == 1.0
    assert payload["awareness_domains"]["library_utilization_awareness"]["default_ml_backend"] == "mlx"
    assert payload["awareness_domains"]["drainer_intelligence"]["status"] == "ready"
    assert payload["awareness_domains"]["drainer_intelligence"]["active_drainer"] == "core_decision_drainer"
    assert payload["awareness_domains"]["drainer_intelligence"]["single_writer_guard"] is True
    assert payload["awareness_domains"]["writer_process_intelligence"]["status"] == "ready"
    assert payload["awareness_domains"]["writer_process_intelligence"]["expanded_writer_lane_count"] == 25
    assert payload["awareness_domains"]["writer_process_intelligence"]["single_writer_guard"] is True
    assert payload["awareness_domains"]["whole_system_intelligence"]["status"] == "ready"
    assert payload["awareness_domains"]["whole_system_intelligence"]["signal_count"] == 22
    assert payload["awareness_domains"]["whole_system_intelligence"]["action"] == "observe_and_expand_cautiously"
    assert payload["awareness_domains"]["whole_system_intelligence"]["contract_count"] == 7
    assert payload["awareness_domains"]["whole_system_intelligence"]["self_reflex_action"] == "follow_system_brain"
    assert payload["awareness_domains"]["whole_system_intelligence"]["self_uncertainty_level"] == "low"
    assert payload["awareness_domains"]["whole_system_intelligence"]["self_causal_root"] == "stable_or_observing"
    assert payload["awareness_domains"]["whole_system_intelligence"]["self_action_effect_verdict"] == "effective"
    assert payload["awareness_domains"]["whole_system_intelligence"]["self_integration_route"] == "observe_and_refresh"
    assert payload["awareness_domains"]["whole_system_intelligence"]["codex_handoff_channel"] == "artifact_handoff"
    assert payload["awareness_domains"]["system_self_intelligence"]["status"] == "ready"
    assert payload["awareness_domains"]["system_self_intelligence"]["trajectory"] == "flat"
    assert payload["awareness_domains"]["system_self_intelligence"]["reflex_action"] == "follow_system_brain"
    assert payload["awareness_domains"]["system_self_intelligence"]["uncertainty_level"] == "low"
    assert payload["awareness_domains"]["system_self_intelligence"]["causal_root"] == "stable_or_observing"
    assert payload["awareness_domains"]["system_self_intelligence"]["action_effect_verdict"] == "effective"
    assert payload["awareness_domains"]["system_self_intelligence"]["integration_route_mode"] == "observe_and_refresh"
    assert payload["awareness_domains"]["codex_operator_bridge"]["status"] == "advisory"
    assert payload["awareness_domains"]["codex_operator_bridge"]["needs_codex_count"] == 1
    assert payload["awareness_domains"]["codex_operator_bridge"]["paper_day_executions"] == 120
    assert payload["awareness_domains"]["codex_operator_bridge"]["training_launch_blockers"] == ["host_training_headroom_not_clear"]
    assert payload["awareness_domains"]["bot_awareness"]["status"] == "ready"
    assert payload["awareness_domains"]["failure_memory"]["status"] == "ready"
    assert payload["awareness_domains"]["halt_recovery_intelligence"]["status"] == "ready"
    assert payload["awareness_domains"]["halt_recovery_intelligence"]["live_lane_running"] is True
    assert payload["awareness_domains"]["halt_recovery_intelligence"]["next_safe_command"][-2:] == ["health-fast", "--json"]
    assert payload["surface_matrix"]["process_watchdog"]["status"] == "ready"
    ranks = [row["rank"] for row in payload["upgrades_and_optimizations"]]
    assert ranks == sorted(set(ranks))
    assert payload["awareness_domains"]["dependency_awareness"]["edge_count"] >= 5
    assert payload["dependency_memory"]["edge_count"] >= 5
    assert payload["failure_memory_index"]["current_event_count"] >= 1
    assert payload["registry_diff_memory"]["diff_status"] == "baseline"
    assert payload["upgrade_optimizer"]["next_generation_backlog"]
    assert "mlx_compute_brain" in payload["upgrade_optimizer"]["implemented_lanes"]
    assert "library_utilization_brain" in payload["upgrade_optimizer"]["implemented_lanes"]
    assert "host_pressure_intelligence" in payload["upgrade_optimizer"]["implemented_lanes"]
    assert "drainer_intelligence" in payload["upgrade_optimizer"]["implemented_lanes"]
    assert "writer_process_intelligence" in payload["upgrade_optimizer"]["implemented_lanes"]
    assert "whole_system_intelligence" in payload["upgrade_optimizer"]["implemented_lanes"]
    assert "system_self_intelligence" in payload["upgrade_optimizer"]["implemented_lanes"]
    assert payload["control_contract"]["platform_brain_mode"] == "big_platform_brain_operational_control_plane"
    assert len(payload["upgrades_and_optimizations"]) >= 6
    assert payload["control_contract"]["consciousness_claim"] == "none_operational_self_model_only"


def test_system_self_model_writes_json_and_markdown(tmp_path: Path) -> None:
    _write_json(tmp_path / "master_bot_registry.json", {"summary": {"total_bots": 1, "active_bots": 1}, "sub_bots": []})
    _write_json(tmp_path / "governance" / "health" / "memory_efficiency_control_latest.json", {"overall_status": "ready", "memory_snapshot": {"memory_pressure_state": "green", "memory_pressure_kind": "none"}})
    _write_json(tmp_path / "governance" / "health" / "ingestion_storage_control_latest.json", {"overall_status": "ready", "severity": "stable", "pressure_index": 0.0})

    payload = src.build_payload(tmp_path)
    out_path = tmp_path / "governance" / "health" / "system_self_model_latest.json"
    md_path = tmp_path / "exports" / "reports" / "operator" / "system_self_model_latest.md"
    brief_path = tmp_path / "exports" / "reports" / "operator" / "system_self_brief_latest.md"
    dependency_path = tmp_path / "governance" / "health" / "system_dependency_memory_latest.json"
    failure_path = tmp_path / "governance" / "health" / "system_failure_memory_latest.json"
    registry_diff_path = tmp_path / "governance" / "health" / "system_registry_diff_latest.json"
    upgrade_path = tmp_path / "governance" / "health" / "system_upgrade_optimizer_latest.json"
    src.write_outputs(
        payload,
        out_path=out_path,
        markdown_path=md_path,
        brief_path=brief_path,
        dependency_memory_path=dependency_path,
        failure_memory_path=failure_path,
        registry_diff_path=registry_diff_path,
        upgrade_plan_path=upgrade_path,
    )

    assert out_path.exists()
    assert md_path.exists()
    assert brief_path.exists()
    assert dependency_path.exists()
    assert failure_path.exists()
    assert registry_diff_path.exists()
    assert upgrade_path.exists()
    assert "# System Self Model" in md_path.read_text(encoding="utf-8")
    assert "# System Self Brief" in brief_path.read_text(encoding="utf-8")


def test_system_self_model_builds_active_halt_recovery_plan(tmp_path: Path) -> None:
    _write_json(tmp_path / "master_bot_registry.json", {"summary": {"total_bots": 1, "active_bots": 1}, "sub_bots": []})
    health = tmp_path / "governance" / "health"
    _write_json(health / "memory_efficiency_control_latest.json", {"overall_status": "ready", "memory_snapshot": {"memory_pressure_state": "green", "memory_pressure_kind": "none"}})
    _write_json(health / "ingestion_storage_control_latest.json", {"overall_status": "ready", "severity": "stable", "pressure_index": 0.0, "total_pending_lines": 12})
    _write_json(
        health / "global_killswitch_latest.json",
        {
            "halt": True,
            "clear_ready": True,
            "clear_blockers": [],
            "global_halt_payload": {
                "reason": "softguard_api_circuit_open",
                "source": "base_trader.softguard",
                "details": {"operation": "get_accounts_snapshot"},
            },
        },
    )
    _write_json(
        health / "process_watchdog_latest.json",
        {
            "overall_status": "degraded",
            "status": [
                {"name": "all_sleeves", "process_live": False, "global_halt_active": True},
                {"name": "coinbase_loop", "process_live": False, "global_halt_active": True},
            ],
        },
    )
    _write_json(
        health / "auth_lease_manager_latest.json",
        {
            "overall_status": "degraded",
            "lease_state": "warning",
            "lease_budget": {"expires_in_seconds": 900, "min_lease_seconds": 1200},
            "broker_state": {"auth_ok": True},
        },
    )
    _write_json(
        health / "data_plane_recovery_controller_latest.json",
        {"overall_status": "degraded", "recovery_state": "recovering_under_guard", "runtime_clearance_state": "awaiting_coverage_cycles", "queue_depth": 12},
    )
    _write_json(health / "live_runtime_separation_control_latest.json", {"overall_status": "degraded", "live_plane": {"live_lane_running": False}})
    _write_json(tmp_path / "governance" / "alerts" / "incident_auto_halt_latest.json", {"overall_status": "ready"})

    payload = src.build_payload(tmp_path)
    halt = payload["awareness_domains"]["halt_recovery_intelligence"]

    assert halt["status"] == "blocked"
    assert halt["halt_active"] is True
    assert halt["clear_ready"] is True
    assert halt["auth_refresh_needed"] is True
    assert halt["next_safe_command"] == ["./scripts/ops/opsctl.sh", "token-refresh", "--json"]
    assert ["./scripts/ops/opsctl.sh", "global-halt-auto-clear", "--json"] in halt["recovery_sequence"]
    assert ["./scripts/ops/opsctl.sh", "livefeed-refresh"] in halt["recovery_sequence"]
    assert "refresh_or_confirm_broker_auth_lease" in halt["needs"]


def test_system_self_model_escalates_auth_fallback_when_token_does_not_extend(tmp_path: Path) -> None:
    _write_json(tmp_path / "master_bot_registry.json", {"summary": {"total_bots": 1, "active_bots": 1}, "sub_bots": []})
    health = tmp_path / "governance" / "health"
    _write_json(health / "memory_efficiency_control_latest.json", {"overall_status": "ready", "memory_snapshot": {"memory_pressure_state": "green", "memory_pressure_kind": "none"}})
    _write_json(health / "ingestion_storage_control_latest.json", {"overall_status": "ready", "severity": "stable", "pressure_index": 0.0, "total_pending_lines": 0})
    _write_json(health / "global_killswitch_latest.json", {"halt": True, "clear_ready": True, "clear_blockers": [], "global_halt_payload": {"reason": "softguard_api_circuit_open"}})
    _write_json(health / "process_watchdog_latest.json", {"overall_status": "ready", "status": [{"name": "all_sleeves", "process_live": True}]})
    _write_json(
        health / "auth_lease_manager_latest.json",
        {
            "overall_status": "blocked",
            "lease_state": "critical",
            "lease_budget": {"expires_in_seconds": 420, "min_lease_seconds": 1200, "critical_lease_seconds": 600},
            "broker_state": {
                "auth_ok": False,
                "broker_operable": True,
                "auth_reason": "auth_succeeded_but_token_not_ready:token_expiring_soon:420.0",
            },
            "fallback_ladder": ["silent_refresh", "interactive_token_refresh", "browser_auth_fallback"],
        },
    )
    _write_json(health / "data_plane_recovery_controller_latest.json", {"overall_status": "ready", "queue_depth": 0})
    _write_json(health / "live_runtime_separation_control_latest.json", {"overall_status": "ready", "live_plane": {"live_lane_running": True}})

    halt = src.build_payload(tmp_path)["awareness_domains"]["halt_recovery_intelligence"]

    assert halt["status"] == "blocked"
    assert halt["operator_auth_required"] is True
    assert halt["next_safe_command"] == ["./scripts/ops/opsctl.sh", "token-refresh-interactive", "--force", "--json"]
    assert "operator_interactive_schwab_auth_refresh" in halt["needs"]
    assert ["./scripts/ops/opsctl.sh", "token-refresh-interactive", "--force", "--json"] in halt["recovery_sequence"]


def test_system_self_model_softens_guarded_paper_managed_surfaces(tmp_path: Path) -> None:
    now = datetime(2026, 7, 27, 16, 15, tzinfo=timezone.utc)
    health = tmp_path / "governance" / "health"
    _write_json(
        health / "runtime_gate_dashboard_latest.json",
        {
            "timestamp_utc": now.isoformat(),
            "overall": {
                "status": "ok",
                "ok": True,
                "raw_attention": ["training_quality_control_blocked", "bot_quality_autopilot_blocked"],
                "soak_management_context": {
                    "soak_ready": True,
                    "soak_status": "ready",
                    "soak_grade": "A+",
                    "paper_guard_clean": True,
                    "paper_stage": "armed",
                    "health_fast_status": "ready",
                },
            },
        },
    )
    _write_json(health / "training_quality_control_latest.json", {"timestamp_utc": now.isoformat(), "overall_status": "blocked"})
    _write_json(health / "bot_quality_autopilot_latest.json", {"timestamp_utc": now.isoformat(), "overall_status": "blocked"})
    _write_json(
        health / "data_plane_recovery_controller_latest.json",
        {
            "timestamp_utc": now.isoformat(),
            "overall_status": "degraded",
            "recovery_state": "recovering_under_guard",
            "runtime_clearance_state": "managed_coverage_stage_deferred",
        },
    )
    _write_json(
        health / "master_infrastructure_supervisor_latest.json",
        {
            "timestamp_utc": now.isoformat(),
            "overall_status": "degraded",
            "metrics": {"blocked_check_count": 0, "hard_failed_attempt_count": 0},
            "checks": [
                {"name": "governance_artifact_freshness", "status": "degraded"},
                {"name": "operator_cockpit_readiness", "status": "degraded"},
                {"name": "self_auditing_infra_bots", "status": "degraded"},
            ],
        },
    )

    matrix = src._surface_matrix(health, tmp_path, now=now)

    assert matrix["runtime_gate_dashboard"]["status"] == "ready"
    assert matrix["runtime_gate_dashboard"]["guarded_paper_context_enabled"] is True
    for name in ("training_quality", "bot_quality", "data_plane_recovery", "master_infra"):
        assert matrix[name]["status"] == "advisory"
        assert matrix[name]["guarded_paper_advisory_only"] is True
        assert matrix[name]["raw_status"] in {"blocked", "degraded"}
    for name in ("codex_operator_bridge", "quant_model_control", "capital_growth_intelligence", "capital_growth_awareness", "capital_rotation_control"):
        assert matrix[name]["status"] == "advisory"
        assert matrix[name]["raw_status"] == "missing"


def test_system_self_model_keeps_optional_support_staleness_advisory_during_guarded_paper_soak(tmp_path: Path) -> None:
    now = datetime(2026, 7, 27, 16, 15, tzinfo=timezone.utc)
    old = now - timedelta(hours=8)
    health = tmp_path / "governance" / "health"
    _write_json(
        health / "runtime_gate_dashboard_latest.json",
        {
            "timestamp_utc": now.isoformat(),
            "overall": {
                "status": "ok",
                "ok": True,
                "soak_management_context": {
                    "soak_ready": True,
                    "soak_status": "ready",
                    "soak_grade": "A+",
                    "paper_guard_clean": True,
                    "paper_stage": "armed",
                    "health_fast_status": "ready",
                },
            },
        },
    )
    for artifact, status in {
        "backpressure_super_drainer_latest.json": "applied_with_followups",
        "backpressure_super_drainer_memory_latest.json": "ready",
        "mlx_runtime_audit_latest.json": "ready",
        "mlx_library_upgrade_latest.json": "ready",
        "mlx_intelligence_router_latest.json": "advisory",
        "library_utilization_router_latest.json": "advisory",
    }.items():
        _write_json(health / artifact, {"timestamp_utc": old.isoformat(), "overall_status": status})

    matrix = src._surface_matrix(health, tmp_path, now=now)
    memory = src._dependency_memory(matrix, {}, now=now)

    assert memory["stale_source_count"] == 0
    assert memory["managed_stale_source_count"] == 6
    assert {row["surface"] for row in memory["managed_stale_sources"]} == {
        "backpressure_super_drainer",
        "backpressure_super_drainer_memory",
        "mlx_runtime",
        "mlx_library",
        "mlx_intelligence_router",
        "library_utilization_router",
    }
