from __future__ import annotations

import json
import os
from pathlib import Path

from scripts.ops import system_intelligence_coordinator as src


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def _seed_pressure_project(project_root: Path) -> None:
    health = project_root / "governance" / "health"
    _write_json(
        project_root / "master_bot_registry.json",
        {
            "sub_bots": [
                {"bot_id": "brain_refinery_v1", "active": True, "data_collection_active": True, "sleeve_profile": "core"},
                {"bot_id": "brain_refinery_v2", "active": True, "data_collection_active": True, "sleeve_profile": "macro"},
            ]
        },
    )
    _write_json(health / "operator_cockpit_latest.json", {"overall_status": "ready", "adaptive_posture": {"hard_blockers": []}})
    _write_json(
        health / "ingestion_storage_control_latest.json",
        {
            "overall_status": "blocked",
            "severity": "critical",
            "pressure_index": 7.121,
            "backpressure": {"total_pending_lines": 17267, "core_pending_lines": 16717, "pending_lines_threshold": 15000},
        },
    )
    _write_json(
        health / "memory_efficiency_control_latest.json",
        {
            "overall_status": "blocked",
            "memory_snapshot": {
                "memory_pressure_state": "yellow",
                "memory_pressure_kind": "compressor",
                "swap_used_gb": 1.2,
                "compressed_store_gb": 13.2,
            },
        },
    )
    _write_json(
        health / "bot_logs_cleanup_intelligence_latest.json",
        {
            "overall_status": "ready",
            "cleanup_needed": False,
            "target_free_gb": 64.0,
            "remaining_to_target_gb": 0.0,
            "disk_after": {"capacity_pct": 85.0, "free_gb": 130.0},
            "candidate_summary": {"eligible_gb": 4.0, "eligible_count": 12},
            "intelligence_layer": {"decision": "observe", "pressure_level": "normal"},
        },
    )
    _write_json(
        health / "storage_quota_guard_latest.json",
        {
            "overall_status": "ready",
            "quota_summary": {"hard_breaches": 0, "soft_breaches": 0, "tracked_lane_count": 4},
            "lanes": [],
            "recommended_actions": [],
        },
    )
    _write_json(
        health / "runtime_throttle_control_latest.json",
        {
            "overall_status": "ready",
            "memory_pressure_level": "normal",
            "cpu_pressure_level": "watch",
            "host_saturation_score": 52.0,
        },
    )
    _write_json(
        health / "writer_process_intelligence_latest.json",
        {
            "overall_status": "ready",
            "decision_packet": {"action": "run_focused_writer_cycle", "writer_state": "idle", "expanded_writer_lane_count": 25, "risk_flags": []},
            "writer_health": {"state": "idle", "active": False},
            "safety_envelope": {"single_writer_only": True, "starts_parallel_sql_writers": False, "writer_recovery_required": False},
        },
    )
    _write_json(
        health / "drainer_intelligence_layer_latest.json",
        {
            "overall_status": "ready",
            "decision_packet": {
                "action": "run_micro_drain_after_pressure_relief",
                "selected_drainer": "core_decision_drainer",
                "total_pending_lines": 17267,
                "target_pending_lines": 10000,
                "risk_flags": ["storage_critical", "memory_pressure_high"],
            },
        },
    )
    _write_json(health / "backpressure_drainer_fleet_latest.json", {"overall_status": "ready", "ready_drainer_count": 2})
    _write_json(health / "backpressure_super_drainer_latest.json", {"overall_status": "ready", "active_drainer": "core_decision_drainer"})
    _write_json(health / "writer_cycle_coordinator_latest.json", {"overall_status": "ready", "summary": {"writer_active_after_wait": False}})
    _write_json(health / "process_watchdog_latest.json", {"overall_status": "ready", "status": []})
    _write_json(health / "process_fanout_guard_latest.json", {"overall_status": "ready", "summary": {"triggered": False}})
    _write_json(
        health / "guard_intelligence_latest.json",
        {
            "overall_status": "ready",
            "policy_mode": "full_schwab_observe",
            "pressure_score": 0.42,
            "signals": {
                "fanout": {
                    "source": "test",
                    "process_count": 44,
                    "max_count": 180,
                    "target_count": 140,
                    "total_rss_mb": 2200.0,
                    "max_rss_mb": 12288.0,
                    "target_rss_mb": 8192.0,
                    "triggered": False,
                },
                "resource_pressure": {"score": 0.2},
                "storage_pressure": {"score": 0.0},
                "guard_status_counts": {"blockers": [], "warnings": [], "stale_core_artifacts": []},
            },
            "recommended_env_overrides": {
                "PROCESS_FANOUT_GUARD_ACTIVE": "0",
                "TRAINING_RUNTIME_PAUSED_FOR_FANOUT": "0",
                "SHADOW_RESEARCH_PAUSED_FOR_FANOUT": "0",
                "RUN_ALL_SLEEVES_WITH_SPECIALIZED_SLEEVES": "1",
            },
        },
    )
    _write_json(health / "global_killswitch_latest.json", {"overall_status": "ready", "halt": False, "clear_ready": True, "clear_blockers": []})
    _write_json(health / "auth_lease_manager_latest.json", {"overall_status": "ready", "lease_state": "healthy", "broker_state": {"auth_ok": True}})
    _write_json(health / "data_plane_recovery_controller_latest.json", {"overall_status": "ready", "queue_depth": 0})
    _write_json(health / "live_runtime_separation_control_latest.json", {"overall_status": "ready", "live_plane": {"live_lane_running": True}})
    _write_json(
        health / "paper_live_data_standard_latest.json",
        {
            "overall_status": "ready",
            "counts_after": {
                "paper_live_data_enabled_bots": 35,
                "legacy_bootstrap_paper_bots": 22,
                "standard_promoted_paper_bots": 0,
                "collection_until_standard_bots": 1567,
                "data_collection_active_bots": 1580,
                "direct_execution_allowed_bots": 0,
                "live_trading_enabled_bots": 0,
            },
            "paper_lane_target": {"minimum": 30, "target": 40, "maximum": 50, "within_target_band": True},
        },
    )
    _write_json(
        health / "sleeve_ticker_universe_latest.json",
        {
            "overall_status": "ready",
            "symbol_counts": {
                "SHADOW_SYMBOLS_CORE": 96,
                "SHADOW_SYMBOLS_VOLATILE": 31,
                "SHADOW_SYMBOLS_DEFENSIVE": 65,
                "COINBASE_WATCH_SYMBOLS": 17,
                "BOND_SYMBOLS": 25,
                "FX_SYMBOLS": 12,
            },
            "sleeve_groups": {"equity_core": [], "cross_asset": [], "income_rates": [], "crypto": [], "long_term_sector": []},
            "env_overrides": {"SLEEVE_TICKER_UNIVERSE_ENABLED": "1"},
        },
    )
    _write_json(health / "mlx_intelligence_router_latest.json", {"overall_status": "ready"})
    _write_json(health / "library_utilization_router_latest.json", {"overall_status": "ready"})
    _write_json(health / "training_quality_control_latest.json", {"overall_status": "ready"})
    _write_json(health / "bot_quality_autopilot_latest.json", {"overall_status": "ready"})
    _write_json(health / "core_bot_materialization_guard_latest.json", {"overall_status": "ready", "summary": {}})
    _write_json(health / "system_self_model_latest.json", {"overall_status": "ready", "identity": {"active_bots": 2, "data_collection_active_bots": 2}})
    _write_json(health / "platform_brain_v6_latest.json", {"overall_status": "ready", "section_count": 15, "gate_blockers": []})
    _write_json(
        health / "pycharm_active_bot_highlights_latest.json",
        {
            "overall_status": "ready",
            "file_color": "Blue",
            "scope_strategy": "brain_refinery_family_with_inactive_exclusions",
            "scope_pattern_bytes": 512,
            "project_view_style": "scope_background_color",
            "foreground_blue_source": "pycharm_vcs_modified_file_status",
            "foreground_blue_supported_without_dirtying_files": False,
            "active_core_bot_file_count": 2,
            "inactive_core_bot_file_count": 0,
            "file_colors_path": str(project_root / ".idea" / "fileColors.xml"),
            "workspace_path": str(project_root / ".idea" / "workspace.xml"),
        },
    )


def test_whole_system_intelligence_builds_signal_bus_brain_contracts_and_handoff(tmp_path: Path) -> None:
    _seed_pressure_project(tmp_path)

    payload = src.build_payload(tmp_path)
    signal_bus = payload["system_signal_bus"]
    brain = payload["system_brain"]
    contracts = payload["system_process_contracts"]
    self_layer = payload["system_self_intelligence"]
    super_layer = payload["system_super_intelligence"]
    outcome_layer = payload["super_intelligence_outcome_learning"]
    recursive_layer = payload["system_recursive_intelligence"]
    handoff = payload["codex_handoff"]

    assert payload["overall_status"] == "degraded"
    assert signal_bus["summary"]["signal_count"] >= 22
    assert signal_bus["summary"]["top_risk"] == "ingestion_storage"
    assert signal_bus["summary"]["storage_critical"] is True
    assert signal_bus["summary"]["memory_pressure_high"] is True
    assert brain["decision_packet"]["action"] == "relieve_pressure_then_micro_drain"
    assert brain["decision_packet"]["safe_next_command"] == ["./scripts/ops/opsctl.sh", "pressure-relief", "--apply", "--json"]
    assert "do_not_start_parallel_sql_writers" in brain["decision_packet"]["do_not_do"]
    assert contracts["global_safety_contract"]["parallel_sql_writers_allowed"] is False
    assert contracts["global_safety_contract"]["live_trade_authority_added"] is False
    assert self_layer["trend"]["trajectory"] == "baseline"
    assert self_layer["uncertainty"]["level"] == "low"
    assert self_layer["reflex"]["action"] == "follow_system_brain"
    assert self_layer["learning_memory"]["memory_event_count"] == 0
    assert self_layer["action_effectiveness"]["verdict"] == "insufficient_history"
    assert self_layer["causal_diagnosis"]["primary_root_cause"] == "storage_backpressure_primary"
    assert self_layer["integration_routing"]["route_mode"] == "storage_first_recovery"
    assert self_layer["integration_routing"]["primary_owner"] == "backpressure_storage_brain_v2"
    awareness = self_layer["awareness_state_vector"]
    assert awareness["grade"] in {"A", "B", "C", "D", "F"}
    assert awareness["level"] in {"high", "medium", "low"}
    assert awareness["known_now"]["causal_root"] == "storage_backpressure_primary"
    assert awareness["body_map"]["storage"]["total_pending_lines"] == 17267
    assert awareness["senses"]["signal_count"] >= 22
    assert awareness["identity"]["active_bots"] == 2
    assert awareness["boundaries"]["trade_authority"] == "none"
    assert awareness["boundaries"]["protected_volume_denylist"] == ["/Volumes/VIDEO"]
    assert awareness["boundaries"]["protected_volume_policy"] == "never_touch_or_clean_VIDEO_without_explicit_user_request"
    assert awareness["blind_spots"]
    assert awareness["next_probe_plan"]
    assert awareness["confidence_calibration"]["confidence_level"] in {"high", "medium", "low"}
    assert awareness["confidence_calibration"]["claim_style"] in {"direct", "qualified", "ask_or_measure_first"}
    assert awareness["confidence_calibration"]["overconfidence_guard"]["active"] is True
    assert awareness["degradation_forecast"]["horizon_minutes"] == 30
    assert awareness["degradation_forecast"]["risks"]
    assert awareness["autonomy_posture"]["mode"] in {
        "ask_operator_or_observe_only",
        "measure_before_apply",
        "bounded_infrastructure_only",
        "bounded_apply_allowed",
    }
    assert "live_trade_authority" in awareness["autonomy_posture"]["blocked_actions"]
    assert "parallel_sql_writers" in awareness["autonomy_posture"]["blocked_actions"]
    assert awareness["consistency_checks"]["overall_status"] in {"ready", "advisory"}
    assert awareness["evidence_after_action"][0]["command"] == ["./scripts/ops/opsctl.sh", "system-intelligence", "--json"]
    assert super_layer["decision_packet"]["executive_mode"] == "drain"
    assert super_layer["decision_packet"]["owner"] == "backpressure_storage_brain_v2"
    assert super_layer["adaptive_policy"]["guard_policy_mode"] == "full_schwab_observe"
    assert super_layer["adaptive_policy"]["sleeve_posture"] == "protect_collection_and_drain"
    assert super_layer["regime_drift_audit"]["current_operational_regime"] == "storage_backpressure"
    assert super_layer["regime_drift_audit"]["regime_policy"]["expansion_allowed"] is False
    assert super_layer["objective_guardrail_layer"]["overall_status"] == "ready"
    assert super_layer["objective_guardrail_layer"]["invariants"]["trade_authority"] == "none"
    assert super_layer["objective_guardrail_layer"]["invariants"]["parallel_sql_writers_allowed"] is False
    assert super_layer["adversarial_simulation_layer"]["top_scenario"] == "storage_refill_after_cleanup"
    assert super_layer["decision_quality_layer"]["quality_grade"] == "high"
    assert super_layer["paper_lane_governor_layer"]["paper_live_data_enabled_bots"] == 35
    assert super_layer["paper_lane_governor_layer"]["paper_lane_posture"] == "standard_30_50_active"
    assert super_layer["symbol_universe_intelligence_layer"]["core_symbol_count"] == 96
    assert super_layer["cognitive_twin_counterfactual_layer"]["worlds"]
    assert "storage_backpressure_primary" in super_layer["semantic_synthesis_layer"]["thesis_statement"]
    assert outcome_layer["intervention_outcome"]["verdict"] == "baseline"
    assert outcome_layer["confidence_recovery_engine"]["state"] == "monitoring"
    assert recursive_layer["policy_hypothesis_lab"]["experiments"]
    assert recursive_layer["next_more_advanced_layer"]["name"] == "cognitive_twin_counterfactual_simulator"
    assert handoff["attention_packet"]["top_risk"] == "ingestion_storage"
    assert handoff["attention_packet"]["super_mode"] == "drain"
    assert handoff["attention_packet"]["super_regime"] == "storage_backpressure"
    assert handoff["attention_packet"]["super_guardrail_status"] == "ready"
    assert handoff["attention_packet"]["super_decision_quality"] == "high"
    assert "storage_backpressure_primary" in handoff["attention_packet"]["super_thesis"]
    assert handoff["attention_packet"]["adaptive_policy"]["expansion_posture"] == "catalog_only"
    assert handoff["attention_packet"]["uncertainty_level"] == "low"
    assert handoff["attention_packet"]["self_awareness_grade"] == awareness["grade"]
    assert handoff["attention_packet"]["self_awareness_level"] == awareness["level"]
    assert handoff["attention_packet"]["operator_boundaries"]["protected_volume_denylist"] == ["/Volumes/VIDEO"]
    assert handoff["attention_packet"]["self_awareness_blind_spots"]
    assert handoff["attention_packet"]["self_awareness_confidence"]["claim_style"] == awareness["confidence_calibration"]["claim_style"]
    assert handoff["attention_packet"]["self_awareness_autonomy"]["mode"] == awareness["autonomy_posture"]["mode"]
    assert handoff["attention_packet"]["self_awareness_forecast"]["posture"] == awareness["degradation_forecast"]["posture"]
    assert handoff["attention_packet"]["self_awareness_consistency"]["overall_status"] == awareness["consistency_checks"]["overall_status"]
    assert handoff["attention_packet"]["self_awareness_evidence_after_action"]
    assert handoff["attention_packet"]["causal_root"] == "storage_backpressure_primary"
    assert handoff["attention_packet"]["action_effectiveness"] == "insufficient_history"
    assert handoff["attention_packet"]["integration_route"] == "storage_first_recovery"
    assert handoff["attention_packet"]["outcome_verdict"] == "baseline"
    assert handoff["attention_packet"]["recursive_status"] in {"ready", "advisory", "degraded"}
    assert handoff["attention_packet"]["next_more_advanced_layer"] == "cognitive_twin_counterfactual_simulator"
    assert handoff["attention_packet"]["upgrade_integration"]["plan_count"] >= 1
    assert handoff["attention_packet"]["upgrade_integration"]["contract"]["requires_proof_metric"] is True
    assert "integrate_pending_upgrades_with_guardrails" in handoff["attention_packet"]["needs_codex"]
    assert "docs/pycharm/intelligence_layers_latest.md" in handoff["attention_packet"]["pycharm_index_path"]
    assert "apply_pressure_relief_before_heavy_work" in handoff["attention_packet"]["needs_codex"]
    assert handoff["communication_contract"]["proactive_delivery_to_codex"] is False


def test_whole_system_intelligence_writes_artifacts_and_self_memory(tmp_path: Path) -> None:
    _seed_pressure_project(tmp_path)
    payload = src.build_payload(tmp_path)
    out_path = tmp_path / "governance" / "health" / "whole_system_intelligence_latest.json"
    signal_path = tmp_path / "governance" / "health" / "system_signal_bus_latest.json"
    brain_path = tmp_path / "governance" / "health" / "system_brain_latest.json"
    contracts_path = tmp_path / "governance" / "health" / "system_process_contracts_latest.json"
    self_path = tmp_path / "governance" / "health" / "system_self_intelligence_latest.json"
    super_path = tmp_path / "governance" / "health" / "system_super_intelligence_latest.json"
    outcome_path = tmp_path / "governance" / "health" / "super_intelligence_outcome_learning_latest.json"
    storage_causal_replay_path = tmp_path / "governance" / "health" / "storage_causal_replay_memory_latest.json"
    recursive_path = tmp_path / "governance" / "health" / "system_recursive_intelligence_latest.json"
    documentation_reporting_path = tmp_path / "governance" / "health" / "documentation_reporting_intelligence_latest.json"
    handoff_path = tmp_path / "governance" / "health" / "codex_handoff_latest.json"
    handoff_md_path = tmp_path / "exports" / "reports" / "operator" / "codex_handoff_latest.md"
    memory_path = tmp_path / "governance" / "system_intelligence" / "self_intelligence_memory.jsonl"
    super_memory_path = tmp_path / "governance" / "system_intelligence" / "super_intelligence_memory.jsonl"
    outcome_memory_path = tmp_path / "governance" / "system_intelligence" / "intervention_outcomes.jsonl"
    storage_causal_replay_memory_path = tmp_path / "governance" / "system_intelligence" / "storage_causal_replay_memory.jsonl"
    recursive_memory_path = tmp_path / "governance" / "system_intelligence" / "recursive_intelligence_memory.jsonl"
    super_override_path = tmp_path / "config" / ".env.super_intelligence_override"
    pycharm_index_path = tmp_path / "docs" / "pycharm" / "intelligence_layers_latest.md"
    pycharm_index_json_path = tmp_path / "governance" / "health" / "intelligence_layers_pycharm_index_latest.json"
    context_path = tmp_path / "governance" / "health" / "whole_system_intelligence_context_latest.json"

    src.write_outputs(
        payload,
        out_path=out_path,
        signal_bus_path=signal_path,
        brain_path=brain_path,
        contracts_path=contracts_path,
        self_intelligence_path=self_path,
        super_intelligence_path=super_path,
        outcome_learning_path=outcome_path,
        storage_causal_replay_path=storage_causal_replay_path,
        recursive_intelligence_path=recursive_path,
        documentation_reporting_path=documentation_reporting_path,
        handoff_path=handoff_path,
        handoff_markdown_path=handoff_md_path,
        memory_path=memory_path,
        super_memory_path=super_memory_path,
        outcome_memory_path=outcome_memory_path,
        storage_causal_replay_memory_path=storage_causal_replay_memory_path,
        recursive_memory_path=recursive_memory_path,
        super_override_path=super_override_path,
        pycharm_index_path=pycharm_index_path,
        pycharm_index_json_path=pycharm_index_json_path,
        context_path=context_path,
    )

    assert out_path.exists()
    assert signal_path.exists()
    assert brain_path.exists()
    assert contracts_path.exists()
    assert self_path.exists()
    assert super_path.exists()
    assert outcome_path.exists()
    assert storage_causal_replay_path.exists()
    assert recursive_path.exists()
    assert documentation_reporting_path.exists()
    assert handoff_path.exists()
    assert memory_path.exists()
    assert super_memory_path.exists()
    assert outcome_memory_path.exists()
    assert storage_causal_replay_memory_path.exists()
    assert recursive_memory_path.exists()
    assert super_override_path.exists()
    assert pycharm_index_path.exists()
    assert pycharm_index_json_path.exists()
    assert context_path.exists()
    assert "# Codex Handoff" in handoff_md_path.read_text(encoding="utf-8")
    assert "Causal Root" in handoff_md_path.read_text(encoding="utf-8")
    assert "Super Intelligence" in handoff_md_path.read_text(encoding="utf-8")
    assert "Outcome Learning" in handoff_md_path.read_text(encoding="utf-8")
    assert "Storage Causal Replay" in handoff_md_path.read_text(encoding="utf-8")
    assert "Recursive Intelligence" in handoff_md_path.read_text(encoding="utf-8")
    assert "Super Regime" in handoff_md_path.read_text(encoding="utf-8")
    assert "Thesis" in handoff_md_path.read_text(encoding="utf-8")
    assert "Paper Lane" in handoff_md_path.read_text(encoding="utf-8")
    assert "Intelligence Layers PyCharm Index" in pycharm_index_path.read_text(encoding="utf-8")
    assert "PyCharm Note" in pycharm_index_path.read_text(encoding="utf-8")
    assert "Active Bot Rows" in pycharm_index_path.read_text(encoding="utf-8")
    assert "PyCharm File Color Status" in pycharm_index_path.read_text(encoding="utf-8")
    assert "PyCharm Project View Style" in pycharm_index_path.read_text(encoding="utf-8")
    assert "pycharm_vcs_modified_file_status" in pycharm_index_path.read_text(encoding="utf-8")
    assert "documentation_reporting_intelligence" in pycharm_index_path.read_text(encoding="utf-8")
    assert "cognitive_twin_counterfactual_simulator" in pycharm_index_path.read_text(encoding="utf-8")
    assert "relieve_pressure_then_micro_drain" in memory_path.read_text(encoding="utf-8")
    assert "relieve_pressure_then_micro_drain" in outcome_memory_path.read_text(encoding="utf-8")
    assert "storage_backpressure_primary" in storage_causal_replay_memory_path.read_text(encoding="utf-8")
    assert "recursive_score" in recursive_memory_path.read_text(encoding="utf-8")
    assert "SUPER_INTELLIGENCE_EXECUTIVE_MODE=drain" in super_override_path.read_text(encoding="utf-8")
    assert "SUPER_INTELLIGENCE_OPERATIONAL_REGIME=storage_backpressure" in super_override_path.read_text(encoding="utf-8")
    assert "SUPER_INTELLIGENCE_OBJECTIVE_GUARDRAIL_STATUS=ready" in super_override_path.read_text(encoding="utf-8")


def test_system_intelligence_routes_training_quality_to_guarded_recovery_batch(tmp_path: Path) -> None:
    _seed_pressure_project(tmp_path)
    health = tmp_path / "governance" / "health"
    training_command = [
        "./scripts/ops/opsctl.sh",
        "retrain-force-targeted",
        "--include-bot-ids",
        "brain_refinery_v10,brain_refinery_v17",
        "--retrain-profile",
        "coverage_batch20_canary",
        "--skip-master-update",
    ]
    _write_json(
        health / "ingestion_storage_control_latest.json",
        {
            "overall_status": "ready",
            "severity": "stable",
            "pressure_index": 0.01,
            "backpressure": {
                "total_pending_lines": 250,
                "core_pending_lines": 250,
                "pending_lines_threshold": 15000,
                "oldest_pending_age_seconds": 30,
            },
        },
    )
    _write_json(
        health / "memory_efficiency_control_latest.json",
        {
            "overall_status": "ready",
            "memory_snapshot": {
                "memory_pressure_state": "green",
                "memory_pressure_kind": "none",
                "swap_used_gb": 1.0,
                "compressed_store_gb": 5.0,
            },
            "cotenant_awareness": {"memory_pressure_clear": True},
        },
    )
    _write_json(
        health / "runtime_throttle_control_latest.json",
        {
            "overall_status": "degraded",
            "memory_pressure_level": "normal",
            "cpu_pressure_level": "watch",
            "host_saturation_score": 42.0,
        },
    )
    _write_json(
        health / "training_quality_control_latest.json",
        {
            "overall_status": "blocked",
            "training_quality_score": 56.5,
            "top_priorities": ["runtime_input_coverage", "active_probation_isolation"],
        },
    )
    _write_json(
        health / "training_runtime_control_latest.json",
        {
            "overall_status": "degraded",
            "training_launch_contract": {
                "mode": "canary_training_allowed",
                "launch_allowed": True,
                "recommended_batch_size": 20,
                "requested_batch_size": 20,
                "available_canary_pool_size": 23,
                "training_quality_recovery_canary": True,
                "launch_blockers": [],
                "recommended_retrain_command": training_command,
                "host_training_headroom_gate": {
                    "selected_training_profile": "coverage_batch20_canary",
                    "batch20_execution_mode": "sequential_memory_guarded_waves",
                    "batch20_wave_size": 4,
                },
            },
        },
    )
    _write_json(health / "guard_intelligence_latest.json", {"overall_status": "ready", "policy_mode": "full_schwab_observe", "signals": {"guard_status_counts": {"blockers": []}}})
    _write_json(health / "writer_process_intelligence_latest.json", {"overall_status": "ready", "decision_packet": {"action": "observe", "risk_flags": []}, "writer_health": {"state": "idle", "active": False}})

    payload = src.build_payload(tmp_path)
    system_brain = payload["system_brain"]
    brain = system_brain["decision_packet"]
    super_decision = payload["system_super_intelligence"]["decision_packet"]
    handoff = payload["codex_handoff"]["attention_packet"]

    assert payload["system_signal_bus"]["summary"]["training_runtime_launch_allowed"] is True
    assert brain["action"] == "run_guarded_training_recovery_canary"
    assert brain["safe_next_command"] == training_command
    assert brain["training_recovery_batch_size"] == 20
    assert "do_not_promote_recovery_canary_to_master_during_quality_recovery" in brain["do_not_do"]
    assert super_decision["executive_mode"] == "train"
    assert super_decision["owner"] == "training_runtime_control"
    assert handoff["safe_next_command"] == training_command
    assert "run_guarded_training_recovery_canary_and_refresh_quality" in handoff["needs_codex"]
    assert handoff["integration_route"] == "training_recovery_first"


def test_signal_bus_normalizes_controlled_training_debt_during_guarded_paper_soak(tmp_path: Path) -> None:
    _seed_pressure_project(tmp_path)
    health = tmp_path / "governance" / "health"
    _write_json(
        health / "ingestion_storage_control_latest.json",
        {
            "overall_status": "ready",
            "severity": "stable",
            "pressure_index": 0.0,
            "backpressure": {"total_pending_lines": 200, "core_pending_lines": 200, "pending_lines_threshold": 15000},
        },
    )
    _write_json(
        health / "memory_efficiency_control_latest.json",
        {
            "overall_status": "ready",
            "memory_snapshot": {"memory_pressure_state": "green", "memory_pressure_kind": "none", "swap_used_gb": 0.0, "compressed_store_gb": 1.0},
            "cotenant_awareness": {"memory_pressure_clear": True},
        },
    )
    _write_json(
        health / "runtime_throttle_control_latest.json",
        {"overall_status": "ready", "memory_pressure_level": "normal", "cpu_pressure_level": "watch", "host_saturation_score": 34.0},
    )
    _write_json(
        health / "health_fast_latest.json",
        {
            "ok": True,
            "overall_status": "ready",
            "strict_all_clear": True,
            "operational_readiness": {
                "guarded_paper": {"ok": True, "status": "ready", "blockers": []},
                "live_execution": {"ok": False, "status": "blocked_read_only", "blockers": ["operator_required"]},
            },
        },
    )
    _write_json(health / "runtime_paper_regression_guard_latest.json", {"ok": True, "overall_status": "ready"})
    _write_json(
        health / "training_quality_control_latest.json",
        {
            "overall_status": "blocked",
            "training_quality_score": 93.5,
            "training_quality_index": 93.5,
            "improvement_status_counts": {"blocked": 1, "needs_work": 2, "ready": 23, "recoverable_blocked": 0, "effective_blocked": 1},
            "recoverable_blocked_keys": [],
            "top_priorities": ["active_probation_isolation", "multiple_testing_control", "promotion_coverage"],
            "a_plus_contract": {"quality_score": 93.5, "promotion_confidence_ready": False, "bench_depth": 1591, "roster_a_plus_ready": False},
            "control_contract": {
                "raw_evidence_preserved": True,
                "controlled_raw_need_count": 7,
                "controlled_raw_need_keys": ["multiple_testing_control", "paper_loss_feedback"],
                "training_process_ready": True,
                "paper_feedback_control_ready": True,
                "label_contract_ready": True,
                "lane_training_control_ready": True,
                "calibration_control_ready": True,
            },
        },
    )
    _write_json(
        health / "bot_quality_autopilot_latest.json",
        {
            "overall_status": "blocked",
            "quality_blockers": {
                "quality_probation_bot_ids": ["bot_a"],
                "targeted_retrain_bot_ids": ["bot_a"],
                "repair_runtime_input_bot_ids": [],
                "students_without_teachers": 0,
                "coverage_shortfall_bots": 4,
                "infrastructure_helper_count": 0,
            },
            "teacher_summary": {"qualified_teacher_count": 3, "elite_teacher_count": 1},
            "quality_upgrade_queue": [{"bot_id": "bot_a"}],
            "attempts": [{"cmd": ["python", "scripts/ops/training_quality_control.py", "--json"], "rc": 2, "timed_out": False}],
        },
    )

    signal_bus = src.build_signal_bus(tmp_path)
    signals = {str(row["name"]): row for row in signal_bus["signals"]}

    assert signal_bus["overall_status"] == "ready"
    assert signal_bus["summary"]["blocked_signal_count"] == 0
    assert signal_bus["summary"]["guarded_paper_advisory_signals"] == ["training_quality", "bot_quality"]
    assert signals["training_quality"]["status"] == "ready"
    assert signals["training_quality"]["source_status"] == "blocked"
    assert signals["training_quality"]["severity_score"] == 20
    assert signals["training_quality"]["raw_severity_score"] == 90
    assert signals["training_quality"]["metrics"]["does_not_block_guarded_paper_soak"] is True
    assert signals["bot_quality"]["status"] == "ready"
    assert signals["bot_quality"]["metrics"]["controlled_training_quality_exit_count"] == 1


def test_signal_bus_keeps_bot_mesh_quality_target_debt_advisory_during_guarded_paper_soak(tmp_path: Path) -> None:
    _seed_pressure_project(tmp_path)
    health = tmp_path / "governance" / "health"
    _write_json(
        health / "ingestion_storage_control_latest.json",
        {
            "overall_status": "ready",
            "severity": "stable",
            "pressure_index": 0.0,
            "backpressure": {"total_pending_lines": 200, "core_pending_lines": 200, "pending_lines_threshold": 15000},
        },
    )
    _write_json(
        health / "memory_efficiency_control_latest.json",
        {
            "overall_status": "ready",
            "memory_snapshot": {"memory_pressure_state": "green", "memory_pressure_kind": "none", "swap_used_gb": 0.0, "compressed_store_gb": 1.0},
            "cotenant_awareness": {"memory_pressure_clear": True},
        },
    )
    _write_json(
        health / "runtime_throttle_control_latest.json",
        {"overall_status": "ready", "memory_pressure_level": "normal", "cpu_pressure_level": "watch", "host_saturation_score": 34.0},
    )
    _write_json(
        health / "health_fast_latest.json",
        {
            "ok": True,
            "overall_status": "ready",
            "strict_all_clear": True,
            "operational_readiness": {
                "guarded_paper": {"ok": True, "status": "ready", "blockers": []},
                "live_execution": {"ok": False, "status": "blocked_read_only", "blockers": ["operator_required"]},
            },
        },
    )
    _write_json(health / "runtime_paper_regression_guard_latest.json", {"ok": True, "overall_status": "ready"})
    _write_json(
        health / "bot_intelligence_mesh_latest.json",
        {
            "overall_status": "ready",
            "communication_readiness_score": 100.0,
            "quality_readiness_score": 57.725,
            "bot_count": 1584,
            "active_bot_count": 1584,
            "missing_tiers": [],
            "a_plus_target_contract": {
                "blocker_count": 6,
                "current_training_quality_score": 93.5,
                "current_data_quality_score": 0.0,
                "current_collection_coverage_score": 16.0,
                "current_training_readiness_score": 0.0,
            },
            "teacher_student_intelligence": {"summary": {"teacher_count": 10, "student_count": 1574, "elite_teacher_count": 5}},
            "hierarchy_edge_summary": {
                "edge_count_total": 42,
                "active_sub_or_infra_route_ratio": 1.0,
                "active_master_route_ratio": 1.0,
            },
            "what_the_system_needs": ["keep training quality debt visible without blocking guarded paper soak"],
        },
    )

    signal_bus = src.build_signal_bus(tmp_path)
    mesh = next(row for row in signal_bus["signals"] if row["name"] == "bot_intelligence_mesh")

    assert mesh["status"] == "ready"
    assert mesh["source_status"] == "ready"
    assert mesh["raw_severity_score"] == 65
    assert mesh["severity_score"] == 20
    assert mesh["metrics"]["does_not_block_guarded_paper_soak"] is True
    assert mesh["metrics"]["normalization_reason"] == "guarded_paper_soak_green_and_bot_mesh_quality_target_debt_visible"


def test_signal_bus_treats_full_eligible_paper_cohort_as_ready(tmp_path: Path) -> None:
    _seed_pressure_project(tmp_path)
    health = tmp_path / "governance" / "health"
    _write_json(
        health / "paper_live_data_standard_latest.json",
        {
            "overall_status": "ready",
            "counts_after": {
                "paper_live_data_enabled_bots": 1584,
                "collection_until_standard_bots": 148,
                "data_collection_active_bots": 1732,
                "direct_execution_allowed_bots": 0,
                "live_trading_enabled_bots": 0,
            },
            "paper_lane_target": {"minimum": 30, "target": 40, "maximum": 50, "within_target_band": False},
            "safety_contract": {"paper_trade_lock": "1", "market_data_only": "1", "live_execution_allowed": False},
        },
    )

    signal_bus = src.build_signal_bus(tmp_path)
    paper = next(row for row in signal_bus["signals"] if row["name"] == "paper_live_data_standard")

    assert paper["status"] == "ready"
    assert paper["severity_score"] == 0
    assert paper["metrics"]["full_eligible_paper_soak"] is True


def test_super_paper_lane_accepts_full_eligible_paper_soak(tmp_path: Path) -> None:
    _seed_pressure_project(tmp_path)
    health = tmp_path / "governance" / "health"
    _write_json(
        health / "ingestion_storage_control_latest.json",
        {
            "overall_status": "ready",
            "severity": "stable",
            "pressure_index": 0.0,
            "backpressure": {"total_pending_lines": 0, "core_pending_lines": 0, "pending_lines_threshold": 15000},
        },
    )
    _write_json(
        health / "memory_efficiency_control_latest.json",
        {
            "overall_status": "ready",
            "memory_snapshot": {"memory_pressure_state": "green", "memory_pressure_kind": "none", "swap_used_gb": 0.0, "compressed_store_gb": 1.0},
            "cotenant_awareness": {"memory_pressure_clear": True},
        },
    )
    _write_json(
        health / "runtime_throttle_control_latest.json",
        {"overall_status": "ready", "memory_pressure_level": "normal", "cpu_pressure_level": "normal", "host_saturation_score": 22.0},
    )
    _write_json(
        health / "paper_live_data_standard_latest.json",
        {
            "overall_status": "ready",
            "counts_after": {
                "paper_live_data_enabled_bots": 120,
                "collection_until_standard_bots": 20,
                "data_collection_active_bots": 140,
                "direct_execution_allowed_bots": 0,
                "live_trading_enabled_bots": 0,
            },
            "paper_lane_target": {"minimum": 30, "target": 40, "maximum": 50, "within_target_band": False},
            "safety_contract": {"paper_trade_lock": "1", "market_data_only": "1", "live_execution_allowed": False},
        },
    )

    payload = src.build_payload(tmp_path)
    super_layer = payload["system_super_intelligence"]
    paper_lane = super_layer["paper_lane_governor_layer"]

    assert paper_lane["overall_status"] == "ready"
    assert paper_lane["paper_lane_posture"] == "full_eligible_paper_soak_active"
    assert paper_lane["full_eligible_paper_soak"] is True
    assert paper_lane["hard_blocks"] == []
    assert super_layer["overall_status"] != "blocked"


def test_signal_bus_defers_training_runtime_and_data_plane_under_guarded_paper_soak(tmp_path: Path) -> None:
    _seed_pressure_project(tmp_path)
    health = tmp_path / "governance" / "health"
    _write_json(
        health / "health_fast_latest.json",
        {
            "ok": True,
            "overall_status": "ready",
            "strict_all_clear": True,
            "operational_readiness": {
                "guarded_paper": {"ok": True, "status": "ready", "blockers": []},
                "live_execution": {"ok": False, "status": "blocked_read_only", "blockers": ["operator_required"]},
            },
        },
    )
    _write_json(health / "runtime_paper_regression_guard_latest.json", {"ok": True, "overall_status": "ready"})
    _write_json(
        health / "live_runtime_separation_control_latest.json",
        {"overall_status": "ready", "ok": True, "live_plane": {"ready": True, "live_lane_running": True}},
    )
    _write_json(
        health / "training_quality_control_latest.json",
        {
            "overall_status": "blocked",
            "training_quality_score": 93.5,
            "recoverable_blocked_keys": [],
            "control_contract": {
                "raw_evidence_preserved": True,
                "training_process_ready": True,
                "paper_feedback_control_ready": True,
                "label_contract_ready": True,
                "lane_training_control_ready": True,
                "calibration_control_ready": True,
            },
        },
    )
    _write_json(
        health / "training_runtime_control_latest.json",
        {
            "overall_status": "degraded",
            "training_launch_contract": {
                "mode": "prep_only",
                "launch_allowed": False,
                "prep_allowed": True,
                "launch_blockers": ["autonomic_training_budget_closed", "training_quality_blocked"],
            },
        },
    )
    _write_json(
        health / "data_plane_recovery_controller_latest.json",
        {
            "overall_status": "degraded",
            "recovery_state": "recovering_under_guard",
            "queue_depth": 2268,
            "write_failure_count": 0,
            "account_snapshot_failure_count": 0,
            "write_path_recovery_evidence": {
                "raw_live_clear": True,
                "route_ready": True,
                "storage_status": "ready",
                "severity": "stable",
                "pressure_index": 0.154,
                "current_sql_write_failures": 0,
                "writer_status": "ok",
                "raw_live": {"core_pending_lines": 2315, "total_pending_lines": 9168, "oldest_pending_age_seconds": 0.0},
            },
        },
    )

    signal_bus = src.build_signal_bus(tmp_path)
    signals = {str(row["name"]): row for row in signal_bus["signals"]}

    assert signals["training_runtime"]["status"] == "ready"
    assert signals["training_runtime"]["source_status"] == "degraded"
    assert signals["training_runtime"]["metrics"]["does_not_block_guarded_paper_soak"] is True
    assert signals["data_plane_recovery"]["status"] == "ready"
    assert signals["data_plane_recovery"]["source_status"] == "degraded"
    assert signals["data_plane_recovery"]["metrics"]["normalization_reason"] == "guarded_paper_soak_green_and_data_plane_recovering_under_guard"


def test_training_runtime_idle_without_candidates_is_not_scored_as_degraded() -> None:
    severity = src._severity_for_signal(
        "training_runtime",
        "ready",
        {
            "launch_allowed": False,
            "prep_allowed": True,
            "launch_blockers": ["no_bot_needs_training_candidates"],
            "recommended_batch_size": 0,
        },
        True,
    )

    assert severity == 0


def test_signal_bus_normalizes_bounded_training_canary_and_platform_brain_under_guarded_paper_soak(tmp_path: Path) -> None:
    _seed_pressure_project(tmp_path)
    health = tmp_path / "governance" / "health"
    _write_json(
        health / "health_fast_latest.json",
        {
            "ok": True,
            "overall_status": "ready",
            "strict_all_clear": True,
            "operational_readiness": {
                "guarded_paper": {"ok": True, "status": "ready", "blockers": []},
                "live_execution": {"ok": False, "status": "blocked_read_only", "blockers": ["operator_required"]},
            },
        },
    )
    _write_json(health / "runtime_paper_regression_guard_latest.json", {"ok": True, "overall_status": "ready"})
    _write_json(
        health / "training_runtime_control_latest.json",
        {
            "overall_status": "degraded",
            "training_launch_contract": {
                "mode": "canary_training_allowed",
                "launch_allowed": True,
                "prep_allowed": True,
                "launch_blockers": [],
                "recommended_batch_size": 2,
                "available_canary_pool_size": 12,
                "requested_batch_size": 4,
                "training_quality_recovery_canary": True,
                "host_training_headroom_gate": {"selected_training_profile": "coverage_batch30_canary"},
                "recommended_retrain_command": ["./scripts/ops/opsctl.sh", "retrain-force-targeted", "--skip-master-update"],
            },
        },
    )
    _write_json(
        health / "platform_brain_v6_latest.json",
        {"overall_status": "needs_work", "section_count": 15, "gate_blockers": []},
    )

    signal_bus = src.build_signal_bus(tmp_path)
    signals = {str(row["name"]): row for row in signal_bus["signals"]}

    assert signals["training_runtime"]["status"] == "ready"
    assert signals["training_runtime"]["source_status"] == "degraded"
    assert signals["training_runtime"]["metrics"]["normalization_reason"] == "guarded_paper_soak_green_and_training_runtime_deferred"
    assert signals["platform_brain_v6"]["status"] == "ready"
    assert signals["platform_brain_v6"]["source_status"] == "needs_work"
    assert signals["platform_brain_v6"]["metrics"]["normalization_reason"] == "guarded_paper_soak_green_and_platform_brain_has_no_gate_blockers"


def test_signal_bus_keeps_optional_support_staleness_managed_under_guarded_paper_soak(tmp_path: Path) -> None:
    _seed_pressure_project(tmp_path)
    health = tmp_path / "governance" / "health"
    old_timestamp = "2026-01-01T00:00:00+00:00"
    _write_json(
        health / "health_fast_latest.json",
        {
            "ok": True,
            "overall_status": "ready",
            "strict_all_clear": True,
            "operational_readiness": {
                "guarded_paper": {"ok": True, "status": "ready", "blockers": []},
                "live_execution": {"ok": False, "status": "blocked_read_only", "blockers": ["operator_required"]},
            },
        },
    )
    _write_json(health / "runtime_paper_regression_guard_latest.json", {"ok": True, "overall_status": "ready"})
    _write_json(health / "backpressure_super_drainer_latest.json", {"timestamp_utc": old_timestamp, "overall_status": "applied_with_followups"})
    _write_json(health / "mlx_intelligence_router_latest.json", {"timestamp_utc": old_timestamp, "overall_status": "advisory"})
    _write_json(health / "library_utilization_router_latest.json", {"timestamp_utc": old_timestamp, "overall_status": "advisory"})

    signal_bus = src.build_signal_bus(tmp_path)
    signals = {str(row["name"]): row for row in signal_bus["signals"]}

    assert signal_bus["summary"]["stale_signal_count"] == 0
    assert signal_bus["summary"]["managed_stale_signals"] == [
        "backpressure_super_drainer",
        "mlx_intelligence_router",
        "library_utilization_router",
    ]
    for name in signal_bus["summary"]["managed_stale_signals"]:
        assert signals[name]["stale"] is False
        assert signals[name]["raw_stale"] is True
        assert signals[name]["managed_stale"] is True

    payload = src.build_payload(tmp_path)
    assert payload["system_self_intelligence"]["uncertainty"]["stale_signals"] == []
    blind_spot_names = {
        str(row.get("name") or "")
        for row in payload["system_self_intelligence"]["awareness_state_vector"]["blind_spots"]
    }
    assert not any(name.startswith("stale_signal:") for name in blind_spot_names)


def test_signal_bus_normalizes_managed_soak_advisories_when_runtime_is_green(tmp_path: Path) -> None:
    _seed_pressure_project(tmp_path)
    health = tmp_path / "governance" / "health"
    _write_json(
        health / "health_fast_latest.json",
        {
            "ok": True,
            "overall_status": "ready",
            "strict_all_clear": True,
            "operational_readiness": {
                "guarded_paper": {"ok": True, "status": "ready", "blockers": []},
                "live_execution": {"ok": False, "status": "blocked_read_only", "blockers": ["operator_required"]},
            },
        },
    )
    _write_json(health / "runtime_paper_regression_guard_latest.json", {"ok": True, "overall_status": "ready"})
    _write_json(
        health / "ingestion_storage_control_latest.json",
        {
            "overall_status": "ready",
            "severity": "stable",
            "pressure_index": 0.01,
            "backpressure": {"total_pending_lines": 844, "core_pending_lines": 227, "pending_lines_threshold": 15000},
        },
    )
    _write_json(
        health / "memory_efficiency_control_latest.json",
        {
            "overall_status": "ready",
            "memory_snapshot": {
                "memory_pressure_state": "green",
                "memory_pressure_kind": "none",
                "swap_used_gb": 1.3,
                "compressed_store_gb": 5.4,
            },
            "cotenant_awareness": {"memory_pressure_clear": True},
        },
    )
    _write_json(
        health / "runtime_throttle_control_latest.json",
        {"overall_status": "advisory", "memory_pressure_level": "normal", "cpu_pressure_level": "normal", "host_saturation_score": 36.0},
    )
    _write_json(
        health / "macro_event_intelligence_latest.json",
        {
            "overall_status": "ready",
            "market_relevance": "high",
            "source": "Federal Reserve",
            "transcript_quality": "live_excerpt",
            "live_detected": False,
            "replay_contract": {"replay_pending": False, "replay_completed": False, "full_video_required": False},
            "calendar_verification": {"status": "not_requested", "ok": False, "reason": "disabled"},
        },
    )
    _write_json(
        health / "training_quality_control_latest.json",
        {"overall_status": "ready", "training_quality_score": 100.0},
    )
    _write_json(
        health / "bot_quality_autopilot_latest.json",
        {
            "overall_status": "needs_work",
            "quality_blockers": {
                "refresh_diagnostics_bot_ids": ["brain_refinery_v265_crypto_risk_off_contagion_shock_guard"],
                "repair_runtime_input_bot_ids": [],
                "quality_probation_bot_ids": [],
                "targeted_retrain_bot_ids": [],
                "students_without_teachers": 0,
                "coverage_shortfall_bots": 0,
                "infrastructure_helper_count": 0,
                "planned_queue_count": 14,
            },
            "teacher_summary": {"qualified_teacher_count": 7, "elite_teacher_count": 2},
            "quality_upgrade_queue": [{"bot_id": "brain_refinery_v10_seasonal"}],
            "attempts": [{"cmd": ["python", "scripts/ops/teacher_quality_guard.py", "--json"], "rc": 0, "timed_out": False}],
        },
    )
    _write_json(
        health / "writer_cycle_coordinator_latest.json",
        {
            "overall_status": "waiting_for_writer",
            "ok": True,
            "summary": {
                "handoff_only": True,
                "writer_active_initial": True,
                "writer_active_after_wait": True,
                "writer_current_step": "shard_linking",
                "completed_writer_lock_handoff_needed": False,
                "wait_timed_out": False,
            },
            "writer_state_before": {
                "active": True,
                "running": True,
                "writer_owner_pid_live": True,
                "writer_lock_held": True,
                "child_writer_active": True,
                "active_child_writer_count": 1,
                "progress_orphaned": False,
                "complete_lock_handoff_needed": False,
                "progress_age_minutes": 0.1,
                "completed_shard_count": 5,
                "planned_shard_count": 18,
                "timed_out_shard_count": 0,
            },
        },
    )
    _write_json(
        health / "training_data_intake_expansion_latest.json",
        {
            "overall_status": "ready",
            "collector_count": 1742,
            "weak_record_count": 1742,
            "trainable_candidate_count": 6,
            "collect_first_count": 1603,
            "summaries": {"weakness_counts": {"sample_starved": 1731, "sequence_starved": 1741}},
        },
    )
    _write_json(
        health / "operating_platform_upgrade_latest.json",
        {
            "overall_status": "applied_with_work_items",
            "ok": True,
            "sections": [
                {"status": "needs_work", "blockers": ["market_posture_control_missing"], "evidence": {"live_execution_allowed": False}},
                {"status": "ready", "blockers": [], "evidence": {"live_execution_allowed": False}},
            ],
        },
    )
    _write_json(
        health / "deeper_intelligence_layers_latest.json",
        {
            "overall_status": "advisory",
            "layer_count": 10,
            "ready_count": 9,
            "advisory_count": 1,
            "degraded_count": 0,
            "blocked_count": 0,
            "missing_surfaces": ["belief_ledger_confidence"],
            "surface_snapshot": {"storage": {"pending_ratio": 0.04}, "runtime": {"pressure_high": False}},
        },
    )
    _write_json(
        health / "backpressure_super_drainer_latest.json",
        {
            "overall_status": "waiting_for_writer",
            "decision_packet": {"selected_drainer": "alert_notification_drainer", "total_pending_lines": 844, "target_pending_lines": 5000},
            "settings": {"target_pending_lines": 5000},
            "summary": {"final_pending_lines": 844, "stop_reason": "target_already_met"},
        },
    )
    _write_json(
        health / "mlx_intelligence_router_latest.json",
        {
            "overall_status": "advisory",
            "ok": True,
            "library_coverage": {"coverage_ratio": 1.0, "missing_count": 0, "compatibility_excluded_count": 3},
            "route_coverage": {"route_coverage_ratio": 1.0, "blocked_lane_count": 0, "excluded_lane_count": 2},
            "runtime_caps": {
                "profile": "foreground_safe",
                "memory_pressure_level": "normal",
                "cpu_pressure_level": "normal",
                "memory_free_pct": 90.0,
                "swap_used_gb": 1.3,
                "host_saturation_score": 36.0,
                "max_concurrent_mlx_jobs": 1,
                "compile_smoke_ok": True,
                "metal_available": True,
            },
            "lane_optimization_summary": {"allowed_lane_count": 3},
            "readiness_repair_plan": {"status": "ready"},
        },
    )
    cell_payload = {
        "overall_status": "advisory",
        "score": 100.0,
        "grade": "A+",
        "operational_health": {
            "status": "ready",
            "grade": "A",
            "raw_status": "blocked",
            "raw_grade": "F",
            "managed_raw_need_count": 15,
            "guarded_paper_soak_health": {"ready": True, "status": "ready"},
        },
        "cells": [{"cell_id": "storage_writer_cell", "overall_status": "blocked"}],
        "top_needs": [{"cell_id": "storage_writer_cell"}],
    }
    _write_json(health / "distributed_cell_architecture_latest.json", cell_payload)
    _write_json(
        health / "cell_federation_intelligence_latest.json",
        {**cell_payload, "intelligence_score": 100.0, "intelligence_grade": "A+"},
    )

    signal_bus = src.build_signal_bus(tmp_path)
    signals = {str(row["name"]): row for row in signal_bus["signals"]}
    normalized = {
        "macro_event_intelligence",
        "runtime_throttle",
        "bot_quality",
        "writer_cycle_coordinator",
        "training_data_intake",
        "operating_platform_upgrade",
        "deeper_intelligence_layers",
        "backpressure_super_drainer",
        "mlx_intelligence_router",
        "distributed_cell_architecture",
        "cell_federation_intelligence",
    }

    assert signal_bus["overall_status"] == "ready"
    assert signal_bus["summary"]["top_risk_score"] <= 20
    assert set(signal_bus["summary"]["guarded_paper_advisory_signals"]).issuperset(normalized)
    for name in normalized:
        assert signals[name]["status"] == "ready"
        assert signals[name]["severity_score"] == 20
        assert signals[name]["metrics"]["does_not_block_guarded_paper_soak"] is True


def test_signal_bus_normalizes_complete_writer_and_empty_drainer_handoffs_under_guarded_paper_soak(tmp_path: Path) -> None:
    _seed_pressure_project(tmp_path)
    health = tmp_path / "governance" / "health"
    _write_json(
        health / "health_fast_latest.json",
        {
            "ok": True,
            "overall_status": "ready",
            "strict_all_clear": True,
            "operational_readiness": {
                "guarded_paper": {"ok": True, "status": "ready", "blockers": []},
                "live_execution": {"ok": False, "status": "blocked_read_only", "blockers": ["operator_required"]},
            },
        },
    )
    _write_json(health / "runtime_paper_regression_guard_latest.json", {"ok": True, "overall_status": "ready"})
    _write_json(
        health / "ingestion_storage_control_latest.json",
        {
            "overall_status": "ready",
            "severity": "stable",
            "pressure_index": 0.0,
            "backpressure": {"total_pending_lines": 0, "core_pending_lines": 0, "pending_lines_threshold": 15000},
        },
    )
    _write_json(
        health / "memory_efficiency_control_latest.json",
        {
            "overall_status": "ready",
            "memory_snapshot": {"memory_pressure_state": "green", "memory_pressure_kind": "none", "swap_used_gb": 1.0, "compressed_store_gb": 3.0},
            "cotenant_awareness": {"memory_pressure_clear": True},
        },
    )
    _write_json(
        health / "runtime_throttle_control_latest.json",
        {"overall_status": "ready", "memory_pressure_level": "normal", "cpu_pressure_level": "normal", "host_saturation_score": 24.0},
    )
    _write_json(
        health / "writer_cycle_coordinator_latest.json",
        {
            "overall_status": "handoff_released",
            "ok": True,
            "summary": {
                "handoff_only": True,
                "writer_active_initial": True,
                "writer_active_after_wait": False,
                "writer_current_step": "complete",
                "completed_writer_lock_handoff_needed": True,
                "wait_timed_out": False,
                "wait_completed": True,
            },
            "writer_state_before": {
                "active": True,
                "running": False,
                "writer_owner_pid_live": True,
                "writer_lock_held": True,
                "child_writer_active": False,
                "active_child_writer_count": 0,
                "progress_orphaned": False,
                "complete_lock_handoff_needed": True,
                "progress_age_minutes": 0.5,
                "timed_out_shard_count": 0,
            },
            "safety_envelope": {"single_writer_only": True},
        },
    )
    _write_json(
        health / "backpressure_drainer_fleet_latest.json",
        {
            "overall_status": "handoff_requested",
            "ok": True,
            "decision_packet": {
                "action": "park_and_observe",
                "selected_drainer": "core_decision_drainer",
                "total_pending_lines": 0,
                "target_pending_lines": 5000,
                "risk_flags": [],
            },
            "summary": {
                "initial_pending_lines": 0,
                "final_pending_lines": 0,
                "pending_lines_delta": 0,
                "waves_run": 0,
                "progress_waves": 0,
                "any_progress": False,
            },
        },
    )

    signal_bus = src.build_signal_bus(tmp_path)
    signals = {str(row["name"]): row for row in signal_bus["signals"]}

    assert signal_bus["overall_status"] == "ready"
    assert signals["writer_cycle_coordinator"]["status"] == "ready"
    assert signals["writer_cycle_coordinator"]["source_status"] == "handoff_released"
    assert signals["writer_cycle_coordinator"]["severity_score"] == 20
    assert signals["writer_cycle_coordinator"]["metrics"]["normalization_reason"] == "guarded_paper_soak_green_and_writer_handoff_released_complete"
    assert signals["backpressure_drainer_fleet"]["status"] == "ready"
    assert signals["backpressure_drainer_fleet"]["source_status"] == "handoff_requested"
    assert signals["backpressure_drainer_fleet"]["severity_score"] == 20
    assert signals["backpressure_drainer_fleet"]["metrics"]["normalization_reason"] == "guarded_paper_soak_green_and_drainer_handoff_has_no_pending_backlog"


def test_signal_bus_marks_guarded_paper_advisory_staleness_as_managed(tmp_path: Path) -> None:
    _seed_pressure_project(tmp_path)
    health = tmp_path / "governance" / "health"
    old_timestamp = "2026-01-01T00:00:00+00:00"
    _write_json(
        health / "health_fast_latest.json",
        {
            "ok": True,
            "overall_status": "ready",
            "strict_all_clear": True,
            "operational_readiness": {
                "guarded_paper": {"ok": True, "status": "ready", "blockers": []},
                "live_execution": {"ok": False, "status": "blocked_read_only", "blockers": ["operator_required"]},
            },
        },
    )
    _write_json(health / "runtime_paper_regression_guard_latest.json", {"ok": True, "overall_status": "ready"})
    _write_json(
        health / "ingestion_storage_control_latest.json",
        {
            "overall_status": "ready",
            "severity": "stable",
            "pressure_index": 0.0,
            "backpressure": {"total_pending_lines": 0, "core_pending_lines": 0, "pending_lines_threshold": 15000},
        },
    )
    _write_json(
        health / "memory_efficiency_control_latest.json",
        {
            "overall_status": "ready",
            "memory_snapshot": {"memory_pressure_state": "green", "memory_pressure_kind": "none", "swap_used_gb": 1.0, "compressed_store_gb": 3.0},
            "cotenant_awareness": {"memory_pressure_clear": True},
        },
    )
    _write_json(
        health / "runtime_throttle_control_latest.json",
        {"overall_status": "ready", "memory_pressure_level": "normal", "cpu_pressure_level": "normal", "host_saturation_score": 24.0},
    )
    _write_json(
        health / "training_data_intake_expansion_latest.json",
        {
            "timestamp_utc": old_timestamp,
            "overall_status": "ready",
            "collector_count": 1742,
            "trainable_candidate_count": 6,
            "collect_first_count": 1603,
            "summaries": {"weakness_counts": {"sample_starved": 1731}},
        },
    )
    _write_json(
        health / "macro_event_intelligence_latest.json",
        {
            "timestamp_utc": old_timestamp,
            "overall_status": "ready",
            "market_relevance": "high",
            "source": "Federal Reserve",
            "transcript_quality": "live_excerpt",
            "live_detected": False,
            "replay_contract": {"replay_pending": False, "replay_completed": False, "full_video_required": False},
            "calendar_verification": {"status": "not_requested", "ok": False, "reason": "disabled"},
        },
    )

    signal_bus = src.build_signal_bus(tmp_path)
    signals = {str(row["name"]): row for row in signal_bus["signals"]}

    assert signal_bus["overall_status"] == "ready"
    for name in {"training_data_intake", "macro_event_intelligence"}:
        assert signals[name]["status"] == "ready"
        assert signals[name]["stale"] is False
        assert signals[name]["raw_stale"] is True
        assert signals[name]["managed_stale"] is True
        assert signals[name]["metrics"]["source_stale"] is True
        assert signals[name]["metrics"]["managed_by"] == "system_signal_bus_guarded_paper_advisory"


def test_signal_bus_manages_auth_warning_above_paper_readiness_floor(tmp_path: Path) -> None:
    _seed_pressure_project(tmp_path)
    health = tmp_path / "governance" / "health"
    _write_json(
        health / "health_fast_latest.json",
        {
            "ok": True,
            "overall_status": "ready",
            "strict_all_clear": True,
            "operational_readiness": {
                "guarded_paper": {"ok": True, "status": "ready", "blockers": []},
                "live_execution": {"ok": False, "status": "blocked_read_only", "blockers": ["operator_required"]},
            },
        },
    )
    _write_json(health / "runtime_paper_regression_guard_latest.json", {"ok": True, "overall_status": "ready"})
    _write_json(
        health / "ingestion_storage_control_latest.json",
        {
            "overall_status": "ready",
            "severity": "stable",
            "pressure_index": 0.0,
            "backpressure": {"total_pending_lines": 0, "core_pending_lines": 0, "pending_lines_threshold": 15000},
        },
    )
    _write_json(
        health / "memory_efficiency_control_latest.json",
        {
            "overall_status": "ready",
            "memory_snapshot": {"memory_pressure_state": "green", "memory_pressure_kind": "none", "swap_used_gb": 1.0, "compressed_store_gb": 3.0},
            "cotenant_awareness": {"memory_pressure_clear": True},
        },
    )
    _write_json(
        health / "runtime_throttle_control_latest.json",
        {"overall_status": "ready", "memory_pressure_level": "normal", "cpu_pressure_level": "normal", "host_saturation_score": 24.0},
    )
    _write_json(
        health / "auth_lease_manager_latest.json",
        {
            "overall_status": "degraded",
            "lease_state": "warning",
            "lease_budget": {"expires_in_seconds": 1120, "critical_lease_seconds": 600, "token_lease_grace": True},
            "broker_state": {
                "broker_ready": True,
                "broker_operable": True,
                "network_ok": True,
                "auth_ok": False,
                "auth_probe_ok": False,
                "configured_for_refresh": True,
                "auth_reason": "account_probe_failed:403",
            },
        },
    )

    signal_bus = src.build_signal_bus(tmp_path)
    auth = next(row for row in signal_bus["signals"] if row["name"] == "auth_lease_manager")

    assert signal_bus["overall_status"] == "ready"
    assert auth["status"] == "ready"
    assert auth["source_status"] == "degraded"
    assert auth["raw_severity_score"] == 90
    assert auth["severity_score"] == 20
    assert auth["metrics"]["normalization_reason"] == "guarded_paper_soak_green_and_auth_warning_above_paper_readiness_floor"


def test_system_intelligence_routes_storage_quota_top_risk_to_quota_remediation(tmp_path: Path) -> None:
    _seed_pressure_project(tmp_path)
    health = tmp_path / "governance" / "health"
    _write_json(
        health / "ingestion_storage_control_latest.json",
        {
            "overall_status": "ready",
            "severity": "stable",
            "pressure_index": 0.01,
            "backpressure": {"total_pending_lines": 250, "core_pending_lines": 250, "pending_lines_threshold": 15000},
        },
    )
    _write_json(
        health / "storage_quota_guard_latest.json",
        {
            "overall_status": "blocked",
            "quota_summary": {
                "hard_breaches": 1,
                "soft_breaches": 0,
                "tracked_lane_count": 5,
                "blocked_families": ["decisions"],
                "worst_over_hard_gb": 9.431,
                "worst_hard_ratio": 1.262,
            },
            "lanes": [
                {
                    "family": "decisions",
                    "used_gb": 45.431,
                    "hard_quota_gb": 36.0,
                    "over_hard_gb": 9.431,
                    "hard_ratio": 1.262,
                    "status": "blocked",
                }
            ],
        },
    )
    _write_json(
        health / "memory_efficiency_control_latest.json",
        {
            "overall_status": "ready",
            "memory_snapshot": {"memory_pressure_state": "green", "memory_pressure_kind": "none"},
            "cotenant_awareness": {"memory_pressure_clear": True},
        },
    )
    _write_json(
        health / "runtime_throttle_control_latest.json",
        {"overall_status": "degraded", "memory_pressure_level": "normal", "cpu_pressure_level": "watch", "host_saturation_score": 42.0},
    )

    payload = src.build_payload(tmp_path)
    system_brain = payload["system_brain"]
    brain = system_brain["decision_packet"]
    super_decision = payload["system_super_intelligence"]["decision_packet"]

    assert payload["system_signal_bus"]["summary"]["top_risk"] == "storage_quota_guard"
    assert brain["action"] == "refresh_storage_quota_then_drain_decisions"
    assert brain["operating_mode"] == "storage_quota_remediation"
    assert brain["safe_next_command"] == ["./scripts/ops/opsctl.sh", "storage-quota-guard", "--json"]
    assert any(step["command"][1] == "governance-telemetry-compactor" for step in system_brain["playbook"])
    assert super_decision["executive_mode"] == "quota"
    assert super_decision["owner"] == "storage_quota_guard"


def test_codex_handoff_surfaces_storage_quota_pressure_actions(tmp_path: Path) -> None:
    _seed_pressure_project(tmp_path)
    health = tmp_path / "governance" / "health"
    _write_json(
        health / "storage_quota_guard_latest.json",
        {
            "overall_status": "blocked",
            "quota_summary": {
                "hard_breaches": 2,
                "soft_breaches": 0,
                "tracked_lane_count": 5,
                "blocked_families": ["decisions", "governance_telemetry"],
                "worst_over_hard_gb": 180.79,
                "worst_hard_ratio": 16.066,
            },
            "lanes": [
                {
                    "family": "governance_telemetry",
                    "used_gb": 192.79,
                    "hard_quota_gb": 12.0,
                    "over_hard_gb": 180.79,
                    "hard_ratio": 16.066,
                    "status": "blocked",
                },
                {
                    "family": "decisions",
                    "used_gb": 63.775,
                    "hard_quota_gb": 36.0,
                    "over_hard_gb": 27.775,
                    "hard_ratio": 1.772,
                    "status": "blocked",
                },
            ],
            "recommended_actions": [
                "shed verbose governance telemetry and compact jsonl ingest journals before trusting the support telemetry quota",
                "prioritize ingestion-storage-control and the core decision drainer before widening decision log producers",
            ],
        },
    )

    payload = src.build_payload(tmp_path)
    handoff = payload["codex_handoff"]["attention_packet"]
    quota = handoff["storage_quota_pressure"]

    assert quota["status"] == "blocked"
    assert quota["blocked_lanes"] == ["governance_telemetry", "decisions"]
    assert quota["worst_over_hard_gb"] == 180.79
    assert quota["top_quota_lanes"][0]["family"] == "governance_telemetry"
    assert "follow_storage_quota_remediation_before_growth" in handoff["needs_codex"]
    assert any(item == "quota_blocked_lanes=governance_telemetry,decisions" for item in handoff["why"])
    assert any(item == "quota_worst_over_hard_gb=180.79" for item in handoff["why"])


def test_documentation_reporting_treats_ok_report_bundle_entries_as_ready(tmp_path: Path) -> None:
    health = tmp_path / "governance" / "health"
    (tmp_path / "README.md").write_text(
        "Auto-Refreshed Highlights\nCOMMANDS.md\ndocs/showcase/generated/highlights_latest.md\n",
        encoding="utf-8",
    )
    (tmp_path / "COMMANDS.md").write_text(
        "Live Feed Views\nReports And PDFs\ndocs-reporting-intelligence\n",
        encoding="utf-8",
    )
    _write_json(health / "commands_hygiene_latest.json", {"overall_status": "ready", "ok": True, "issues": []})
    _write_json(health / "commands_contract_latest.json", {"entry_count": 151, "contract_hash": "abc"})
    _write_json(health / "report_quality_guard_latest.json", {"overall_status": "ready", "ok": True})
    _write_json(
        health / "report_pdf_bundle_latest.json",
        {
            "overall_status": "ready",
            "ok": True,
            "index_ok": True,
            "index_html_ok": True,
            "entries": [{"slug": "daily_runtime_summary", "ok": True, "detail": "report_ready"}],
        },
    )
    _write_json(
        health / "pycharm_active_bot_highlights_latest.json",
        {
            "overall_status": "ready",
            "file_color": "Blue",
            "scope_strategy": "brain_refinery_family_with_inactive_exclusions",
            "scope_pattern_bytes": 512,
            "project_view_style": "scope_background_color",
            "foreground_blue_source": "pycharm_vcs_modified_file_status",
            "foreground_blue_supported_without_dirtying_files": False,
            "active_core_bot_file_count": 2,
            "inactive_core_bot_file_count": 0,
        },
    )

    payload = src.build_documentation_reporting_intelligence(
        tmp_path,
        {"summary": {"active_bots": 2, "collection_bots": 2, "paper_live_data_bots": 1}},
    )

    assert payload["overall_status"] == "ready"
    assert payload["decision_packet"]["advisories"] == []
    assert payload["reporting_layer"]["bundle_error_count"] == 0
    assert payload["pycharm_visibility_layer"]["project_view_style"] == "scope_background_color"


def test_super_intelligence_routes_guard_throttle_before_expansion(tmp_path: Path) -> None:
    _seed_pressure_project(tmp_path)
    health = tmp_path / "governance" / "health"
    _write_json(
        health / "ingestion_storage_control_latest.json",
        {
            "overall_status": "ready",
            "severity": "normal",
            "pressure_index": 0.0,
            "backpressure": {"total_pending_lines": 0, "core_pending_lines": 0, "pending_lines_threshold": 15000},
        },
    )
    _write_json(
        health / "memory_efficiency_control_latest.json",
        {
            "overall_status": "ready",
            "memory_snapshot": {
                "memory_pressure_state": "green",
                "memory_pressure_kind": "none",
                "swap_used_gb": 1.2,
                "compressed_store_gb": 6.0,
            },
            "cotenant_awareness": {"memory_pressure_clear": True},
        },
    )
    _write_json(
        health / "guard_intelligence_latest.json",
        {
            "overall_status": "active",
            "policy_mode": "protective_throttle",
            "pressure_score": 1.08,
            "signals": {
                "fanout": {
                    "source": "test",
                    "process_count": 188,
                    "max_count": 180,
                    "target_count": 140,
                    "total_rss_mb": 13000.0,
                    "max_rss_mb": 12288.0,
                    "target_rss_mb": 8192.0,
                    "triggered": True,
                },
                "resource_pressure": {"score": 1.08},
                "storage_pressure": {"score": 0.0},
                "guard_status_counts": {"blockers": ["process_fanout"], "warnings": [], "stale_core_artifacts": []},
            },
            "recommended_env_overrides": {
                "PROCESS_FANOUT_GUARD_ACTIVE": "1",
                "TRAINING_RUNTIME_PAUSED_FOR_FANOUT": "1",
                "SHADOW_RESEARCH_PAUSED_FOR_FANOUT": "1",
                "RUN_ALL_SLEEVES_WITH_SPECIALIZED_SLEEVES": "0",
            },
        },
    )

    payload = src.build_payload(tmp_path)
    signal_bus = payload["system_signal_bus"]
    contracts = payload["system_process_contracts"]
    super_layer = payload["system_super_intelligence"]

    assert signal_bus["summary"]["guard_policy_mode"] == "protective_throttle"
    assert "guard_intelligence_throttle_active" in payload["system_brain"]["decision_packet"]["risk_flags"]
    assert super_layer["decision_packet"]["executive_mode"] == "stabilize"
    assert super_layer["decision_packet"]["safe_next_command"] == ["./scripts/ops/opsctl.sh", "guard-intelligence", "--apply", "--json"]
    assert super_layer["adaptive_policy"]["expansion_posture"] == "closed"
    assert super_layer["regime_drift_audit"]["current_operational_regime"] == "guard_throttle"
    assert super_layer["objective_guardrail_layer"]["invariants"]["live_trading_enabled"] is False
    sleeve_contract = next(row for row in contracts["contracts"] if row["name"] == "sleeves")
    assert "guard_intelligence_protective_throttle" in sleeve_contract["active_risks"]


def test_small_stable_backlog_with_memory_pressure_routes_as_resource_pressure(tmp_path: Path) -> None:
    _seed_pressure_project(tmp_path)
    health = tmp_path / "governance" / "health"
    _write_json(
        health / "ingestion_storage_control_latest.json",
        {
            "overall_status": "ready",
            "severity": "normal",
            "pressure_index": 0.0,
            "backpressure": {"total_pending_lines": 1200, "core_pending_lines": 1200, "pending_lines_threshold": 15000},
        },
    )

    payload = src.build_payload(tmp_path)
    brain = payload["system_brain"]
    super_layer = payload["system_super_intelligence"]

    assert brain["decision_packet"]["action"] == "relieve_pressure_then_observe_backlog"
    assert brain["decision_packet"]["storage_evidence"]["pending_ratio"] < 1.0
    assert super_layer["regime_drift_audit"]["current_operational_regime"] == "resource_pressure"
    assert super_layer["regime_drift_audit"]["material_storage_backlog"] is False
    assert super_layer["decision_packet"]["executive_mode"] == "stabilize"
    assert super_layer["decision_packet"]["owner"] == "runtime_throttle_control"


def test_self_intelligence_prechecks_conflicting_drainer_writer_state(tmp_path: Path) -> None:
    _seed_pressure_project(tmp_path)
    _write_json(
        tmp_path / "governance" / "health" / "drainer_intelligence_layer_latest.json",
        {
            "overall_status": "ready",
            "decision_packet": {
                "action": "verify_writer_progress_then_re_score",
                "selected_drainer": "core_decision_drainer",
                "total_pending_lines": 17267,
                "target_pending_lines": 10000,
                "risk_flags": ["writer_progress_stale", "storage_critical"],
            },
        },
    )

    payload = src.build_payload(tmp_path)
    self_layer = payload["system_self_intelligence"]
    handoff = payload["codex_handoff"]["attention_packet"]

    assert self_layer["overall_status"] == "degraded"
    assert "drainer_waits_on_writer_after_writer_idle" in self_layer["uncertainty"]["conflicting_signals"]
    assert self_layer["reflex"]["action"] == "refresh_drainer_intelligence_before_apply"
    assert handoff["safe_next_command"] == ["./scripts/ops/opsctl.sh", "drainer-intelligence-layer", "--apply", "--json"]
    assert "run_self_intelligence_precheck_before_brain_action" in handoff["needs_codex"]


def test_drainer_signal_metrics_surface_backlog_needs_packet() -> None:
    metrics = src._drainer_metrics(
        {
            "overall_status": "ready",
            "decision_packet": {
                "action": "tighten_intake_then_re_score",
                "selected_drainer": "core_decision_drainer",
                "total_pending_lines": 30344,
                "target_pending_lines": 10000,
                "backlog_grade": "C",
                "backlog_score": 73.7,
                "risk_flags": ["recent_refill_after_drain"],
            },
            "backlog_needs_packet": {
                "current_grade": "C",
                "next_grade": "B",
                "top_need_section": "core_decision",
                "top_need": "drain core decision pending lines",
                "needs": [{"section_id": "core_decision", "what_it_needs": "drain core decision pending lines"}],
                "accelerator_contract": {
                    "latest_needs_artifact": "governance/health/backlog_drain_needs_latest.json",
                    "fix_ledger_artifact": "governance/system_intelligence/backlog_drain_fix_ledger.jsonl",
                },
            },
        }
    )

    assert metrics["backlog_grade"] == "C"
    assert metrics["backlog_score"] == 73.7
    assert metrics["needs_count"] == 1
    assert metrics["top_need_section"] == "core_decision"
    assert metrics["needs_artifact"] == "governance/health/backlog_drain_needs_latest.json"
    assert "need=core_decision" in src._signal_summary("drainer_intelligence", metrics)


def test_self_intelligence_refreshes_drainer_storage_alignment_when_pending_totals_drift(tmp_path: Path) -> None:
    _seed_pressure_project(tmp_path)
    health = tmp_path / "governance" / "health"
    _write_json(
        health / "ingestion_storage_control_latest.json",
        {
            "overall_status": "blocked",
            "severity": "critical",
            "pressure_index": 2.716,
            "backpressure": {"total_pending_lines": 40742, "core_pending_lines": 39726, "pending_lines_threshold": 15000},
        },
    )
    _write_json(
        health / "drainer_intelligence_layer_latest.json",
        {
            "overall_status": "ready",
            "decision_packet": {
                "action": "wait_for_writer_then_re_score",
                "selected_drainer": "core_decision_drainer",
                "total_pending_lines": 628010,
                "target_pending_lines": 5000,
                "risk_flags": ["storage_critical", "writer_active"],
            },
        },
    )
    _write_json(
        health / "backpressure_super_drainer_latest.json",
        {
            "overall_status": "applied_with_followups",
            "active_drainer": "core_decision_drainer",
            "summary": {"final_pending_lines": 493811, "total_pending_lines": 493811},
        },
    )

    payload = src.build_payload(tmp_path)
    self_layer = payload["system_self_intelligence"]
    handoff = payload["codex_handoff"]["attention_packet"]
    conflicts = self_layer["uncertainty"]["conflicting_signals"]

    assert "drainer_pending_total_drift_from_storage" in conflicts
    assert "super_drainer_pending_total_drift_from_storage" in conflicts
    assert self_layer["reflex"]["action"] == "refresh_drainer_storage_alignment_before_apply"
    assert self_layer["reflex"]["command"] == ["./scripts/ops/opsctl.sh", "ingestion-storage-control", "--json"]
    assert self_layer["reflex"]["followup_command"] == ["./scripts/ops/opsctl.sh", "drainer-intelligence-layer", "--apply", "--json"]
    assert self_layer["reflex"]["verification_command"] == ["./scripts/ops/opsctl.sh", "system-intelligence", "--json"]
    assert self_layer["reflex"]["blocks_brain_action_until_refreshed"] is True
    assert handoff["safe_next_command"] == ["./scripts/ops/opsctl.sh", "ingestion-storage-control", "--json"]
    assert handoff["super_mode"] == "precheck"


def test_self_intelligence_allows_small_pending_total_drift(tmp_path: Path) -> None:
    _seed_pressure_project(tmp_path)
    health = tmp_path / "governance" / "health"
    _write_json(
        health / "ingestion_storage_control_latest.json",
        {
            "overall_status": "blocked",
            "severity": "critical",
            "pressure_index": 2.716,
            "backpressure": {"total_pending_lines": 40742, "core_pending_lines": 39726, "pending_lines_threshold": 15000},
        },
    )
    _write_json(
        health / "drainer_intelligence_layer_latest.json",
        {
            "overall_status": "ready",
            "decision_packet": {
                "action": "run_micro_drain_after_pressure_relief",
                "selected_drainer": "core_decision_drainer",
                "total_pending_lines": 48000,
                "target_pending_lines": 10000,
                "risk_flags": ["storage_critical"],
            },
        },
    )
    _write_json(
        health / "backpressure_super_drainer_latest.json",
        {
            "overall_status": "ready",
            "active_drainer": "core_decision_drainer",
            "summary": {"final_pending_lines": 48500, "total_pending_lines": 48500},
        },
    )

    payload = src.build_payload(tmp_path)
    self_layer = payload["system_self_intelligence"]
    conflicts = self_layer["uncertainty"]["conflicting_signals"]

    assert "drainer_pending_total_drift_from_storage" not in conflicts
    assert "super_drainer_pending_total_drift_from_storage" not in conflicts
    assert self_layer["reflex"]["action"] == "follow_system_brain"


def test_self_intelligence_refreshes_drainer_storage_alignment_when_drainer_underreports_pending(tmp_path: Path) -> None:
    _seed_pressure_project(tmp_path)
    health = tmp_path / "governance" / "health"
    _write_json(
        health / "ingestion_storage_control_latest.json",
        {
            "overall_status": "blocked",
            "severity": "critical",
            "pressure_index": 8.351,
            "backpressure": {"total_pending_lines": 125266, "core_pending_lines": 124250, "pending_lines_threshold": 15000},
        },
    )
    _write_json(
        health / "drainer_intelligence_layer_latest.json",
        {
            "overall_status": "ready",
            "decision_packet": {
                "action": "wait_for_writer_then_re_score",
                "selected_drainer": "core_decision_drainer",
                "total_pending_lines": 40742,
                "target_pending_lines": 5000,
                "risk_flags": ["storage_critical", "writer_active"],
            },
        },
    )

    payload = src.build_payload(tmp_path)
    self_layer = payload["system_self_intelligence"]

    assert "drainer_pending_total_drift_from_storage" in self_layer["uncertainty"]["conflicting_signals"]
    assert self_layer["reflex"]["action"] == "refresh_drainer_storage_alignment_before_apply"
    assert self_layer["reflex"]["command"] == ["./scripts/ops/opsctl.sh", "ingestion-storage-control", "--json"]
    assert self_layer["reflex"]["followup_command"] == ["./scripts/ops/opsctl.sh", "drainer-intelligence-layer", "--apply", "--json"]


def test_self_intelligence_refreshes_stale_pressure_surfaces_with_valid_command(tmp_path: Path) -> None:
    _seed_pressure_project(tmp_path)
    health = tmp_path / "governance" / "health"
    old_epoch = 1
    os.utime(health / "memory_efficiency_control_latest.json", (old_epoch, old_epoch))
    os.utime(health / "runtime_throttle_control_latest.json", (old_epoch, old_epoch))

    payload = src.build_payload(tmp_path)
    self_layer = payload["system_self_intelligence"]
    handoff = payload["codex_handoff"]["attention_packet"]

    assert self_layer["reflex"]["action"] == "refresh_stale_pressure_surfaces"
    assert self_layer["reflex"]["command"] == ["./scripts/ops/opsctl.sh", "memory-efficiency", "status", "--json"]
    assert self_layer["reflex"]["followup_command"] == ["./scripts/ops/opsctl.sh", "runtime-throttle", "--json"]
    assert handoff["safe_next_command"] == ["./scripts/ops/opsctl.sh", "memory-efficiency", "status", "--json"]


def test_self_intelligence_refreshes_stale_storage_decision_surfaces(tmp_path: Path) -> None:
    _seed_pressure_project(tmp_path)
    health = tmp_path / "governance" / "health"
    _write_json(
        health / "ingestion_storage_control_latest.json",
        {
            "overall_status": "ready",
            "severity": "stable",
            "pressure_index": 0.2,
            "backpressure": {"total_pending_lines": 1200, "core_pending_lines": 400, "pending_lines_threshold": 15000},
        },
    )
    _write_json(
        health / "memory_efficiency_control_latest.json",
        {
            "overall_status": "ready",
            "memory_snapshot": {
                "memory_pressure_state": "green",
                "memory_pressure_kind": "none",
                "memory_free_pct": 83.0,
                "swap_used_gb": 0.2,
                "compressed_store_gb": 2.0,
            },
            "cotenant_awareness": {"memory_pressure_clear": True},
        },
    )
    _write_json(
        health / "runtime_throttle_control_latest.json",
        {
            "overall_status": "ready",
            "memory_pressure_level": "normal",
            "cpu_pressure_level": "watch",
            "host_saturation_score": 35.0,
        },
    )
    _write_json(
        health / "storage_quota_guard_latest.json",
        {
            "overall_status": "blocked",
            "quota_summary": {"hard_breaches": 2, "soft_breaches": 0, "tracked_lane_count": 4},
            "lanes": [{"family": "decisions", "status": "blocked"}],
            "recommended_actions": ["refresh quota lanes before treating this as current pressure"],
        },
    )
    old_epoch = 1
    os.utime(health / "storage_quota_guard_latest.json", (old_epoch, old_epoch))

    payload = src.build_payload(tmp_path)
    signal_bus = payload["system_signal_bus"]
    self_layer = payload["system_self_intelligence"]
    handoff = payload["codex_handoff"]["attention_packet"]
    quota_signal = next(row for row in signal_bus["signals"] if row["name"] == "storage_quota_guard")

    assert quota_signal["stale"] is True
    assert quota_signal["raw_severity_score"] >= 90
    assert quota_signal["severity_score"] < quota_signal["raw_severity_score"]
    assert signal_bus["summary"]["stale_top_signal"] == "storage_quota_guard"
    assert self_layer["reflex"]["action"] == "refresh_stale_decision_surfaces"
    assert self_layer["reflex"]["command"] == ["./scripts/ops/opsctl.sh", "storage-quota-guard", "--json"]
    assert self_layer["reflex"]["refresh_plan"][0]["signal"] == "storage_quota_guard"
    assert handoff["safe_next_command"] == ["./scripts/ops/opsctl.sh", "storage-quota-guard", "--json"]


def test_deeper_intelligence_refresh_command_persists_artifact(tmp_path: Path) -> None:
    _seed_pressure_project(tmp_path)
    health = tmp_path / "governance" / "health"
    _write_json(
        health / "deeper_intelligence_layers_latest.json",
        {
            "overall_status": "advisory",
            "layer_count": 10,
            "blocked_count": 0,
            "degraded_count": 0,
        },
    )
    old_epoch = 1
    os.utime(health / "deeper_intelligence_layers_latest.json", (old_epoch, old_epoch))

    signal_bus = src.build_signal_bus(tmp_path)
    deeper_signal = next(row for row in signal_bus["signals"] if row["name"] == "deeper_intelligence_layers")

    assert deeper_signal["stale"] is True
    assert deeper_signal["refresh_command"] == ["./scripts/ops/opsctl.sh", "deeper-intelligence-layers", "--apply", "--json"]


def test_signal_bus_does_not_report_memory_high_when_memory_controller_is_storage_blocked(tmp_path: Path) -> None:
    _seed_pressure_project(tmp_path)
    health = tmp_path / "governance" / "health"
    _write_json(
        health / "memory_efficiency_control_latest.json",
        {
            "overall_status": "blocked",
            "reasons": ["storage_pressure_critical", "co_running_heavy_competition"],
            "memory_snapshot": {
                "memory_pressure_state": "green",
                "memory_pressure_kind": "none",
                "swap_used_gb": 1.5,
                "compressed_store_gb": 10.0,
            },
            "cotenant_awareness": {"memory_pressure_clear": True, "storage_pressure_clear": False},
            "reasons": ["memory_headroom_ok"],
        },
    )
    _write_json(
        health / "process_fanout_guard_latest.json",
        {
            "overall_status": "degraded",
            "triggered": True,
            "fanout": {"targetable_count": 0, "total_rss_mb": 4091.0},
            "startup_policy": {"core_sleeve_restart_allowed": True},
        },
    )

    payload = src.build_payload(tmp_path)
    signal_bus = payload["system_signal_bus"]
    self_layer = payload["system_self_intelligence"]
    memory_signal = next(row for row in signal_bus["signals"] if row["name"] == "memory_efficiency")

    assert signal_bus["summary"]["memory_pressure_high"] is False
    assert memory_signal["severity_score"] < 65
    assert "fanout_guard_holding_without_targetable_processes" not in self_layer["uncertainty"]["conflicting_signals"]


def test_process_watchdog_restart_in_progress_is_not_reported_as_down(tmp_path: Path) -> None:
    _seed_pressure_project(tmp_path)
    health = tmp_path / "governance" / "health"
    _write_json(
        health / "process_watchdog_latest.json",
        {
            "overall_status": "degraded",
            "status": [
                {"name": "all_sleeves", "running": 0, "process_live": False, "restarted_pid": 1234},
                {"name": "coinbase_loop", "running": 1, "process_live": True},
            ],
            "alerts": [],
        },
    )

    payload = src.build_payload(tmp_path)
    watchdog = next(row for row in payload["system_signal_bus"]["signals"] if row["name"] == "process_watchdog")

    assert watchdog["metrics"]["down_processes"] == []
    assert watchdog["metrics"]["restarted_count"] == 1
    assert watchdog["status"] == "advisory"
    assert watchdog["source_status"] == "degraded"
    assert watchdog["severity_score"] < watchdog["raw_severity_score"]
    assert watchdog["metrics"]["normalization_reason"] == "watchdog_has_no_down_processes_or_alerts"


def test_process_fanout_clear_hold_is_not_reported_as_degraded(tmp_path: Path) -> None:
    _seed_pressure_project(tmp_path)
    health = tmp_path / "governance" / "health"
    _write_json(
        health / "process_fanout_guard_latest.json",
        {
            "overall_status": "active",
            "triggered": False,
            "fanout": {"targetable_count": 24, "total_rss_mb": 4953.0},
            "override": {"hold_active": True},
            "startup_policy": {"core_sleeve_restart_allowed": False},
        },
    )

    payload = src.build_payload(tmp_path)
    fanout = next(row for row in payload["system_signal_bus"]["signals"] if row["name"] == "process_fanout_guard")

    assert fanout["status"] == "advisory"
    assert fanout["source_status"] == "active"
    assert fanout["metrics"]["resolved_fanout_state"] is True
    assert fanout["severity_score"] < 65
    assert "triggered=False" in fanout["summary"]


def test_super_intelligence_does_not_treat_conservative_fanout_hold_as_guard_conflict(tmp_path: Path) -> None:
    _seed_pressure_project(tmp_path)
    health = tmp_path / "governance" / "health"
    _write_json(
        health / "process_fanout_guard_latest.json",
        {
            "overall_status": "active",
            "triggered": True,
            "fanout": {"targetable_count": 12, "total_rss_mb": 9300.0},
            "startup_policy": {"core_sleeve_restart_allowed": False},
        },
    )
    _write_json(
        health / "guard_intelligence_latest.json",
        {
            "overall_status": "ready",
            "policy_mode": "full_schwab_observe",
            "pressure_score": 0.49,
            "signals": {
                "fanout": {
                    "source": "test",
                    "process_count": 86,
                    "max_count": 180,
                    "target_count": 100,
                    "total_rss_mb": 6100.0,
                    "max_rss_mb": 12288.0,
                    "target_rss_mb": 6144.0,
                    "triggered": False,
                },
                "resource_pressure": {"score": 0.2},
                "storage_pressure": {"score": 0.0},
                "guard_status_counts": {"blockers": [], "warnings": ["process_fanout"], "stale_core_artifacts": []},
            },
            "recommended_env_overrides": {"PROCESS_FANOUT_GUARD_ACTIVE": "0"},
        },
    )

    payload = src.build_payload(tmp_path)
    conflicts = payload["system_self_intelligence"]["uncertainty"]["conflicting_signals"]

    assert "guard_full_observe_conflicts_with_active_fanout_trigger" not in conflicts
    assert payload["system_super_intelligence"]["decision_packet"]["executive_mode"] != "precheck"


def test_self_intelligence_suppresses_stale_auth_halt_blocker_when_auth_lease_is_healthy(tmp_path: Path) -> None:
    _seed_pressure_project(tmp_path)
    health = tmp_path / "governance" / "health"
    _write_json(
        health / "global_killswitch_latest.json",
        {
            "overall_status": "ready",
            "halt": False,
            "clear_ready": False,
            "clear_blockers": ["auth_lease_critical"],
        },
    )
    _write_json(
        health / "auth_lease_manager_latest.json",
        {
            "overall_status": "ready",
            "lease_state": "healthy",
            "broker_state": {"auth_ok": True, "broker_operable": True},
        },
    )

    payload = src.build_payload(tmp_path)
    conflicts = payload["system_self_intelligence"]["uncertainty"]["conflicting_signals"]
    self_questions = payload["codex_handoff"]["attention_packet"]["self_questions"]

    assert "halt_clear_blockers_present_without_active_halt" not in conflicts
    assert all("halt_clear_blockers_present_without_active_halt" not in question for question in self_questions)


def test_self_intelligence_keeps_halt_blocker_conflict_when_auth_lease_is_not_clear(tmp_path: Path) -> None:
    _seed_pressure_project(tmp_path)
    health = tmp_path / "governance" / "health"
    _write_json(
        health / "global_killswitch_latest.json",
        {
            "overall_status": "ready",
            "halt": False,
            "clear_ready": False,
            "clear_blockers": ["auth_lease_critical"],
        },
    )
    _write_json(
        health / "auth_lease_manager_latest.json",
        {
            "overall_status": "blocked",
            "lease_state": "critical",
            "broker_state": {"auth_ok": False, "broker_operable": False},
        },
    )

    payload = src.build_payload(tmp_path)
    conflicts = payload["system_self_intelligence"]["uncertainty"]["conflicting_signals"]

    assert "halt_clear_blockers_present_without_active_halt" in conflicts


def test_self_intelligence_scores_repeated_action_effectiveness(tmp_path: Path) -> None:
    _seed_pressure_project(tmp_path)
    memory_path = tmp_path / "governance" / "system_intelligence" / "self_intelligence_memory.jsonl"
    memory_path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "timestamp_utc": f"2026-05-06T12:0{i}:00Z",
            "status": "degraded",
            "action": "relieve_pressure_then_micro_drain",
            "top_risk": "ingestion_storage",
            "pending_lines": 17267,
            "trajectory": "flat",
            "uncertainty_level": "low",
            "reflex_action": "follow_system_brain",
        }
        for i in range(3)
    ]
    memory_path.write_text("\n".join(json.dumps(row, ensure_ascii=True) for row in rows) + "\n", encoding="utf-8")

    payload = src.build_payload(tmp_path)
    self_layer = payload["system_self_intelligence"]
    super_layer = payload["system_super_intelligence"]
    handoff = payload["codex_handoff"]["attention_packet"]

    assert self_layer["learning_memory"]["same_action_repeat_count"] == 3
    assert self_layer["action_effectiveness"]["same_action_run_length"] == 4
    assert self_layer["action_effectiveness"]["verdict"] == "ineffective_so_far"
    assert "pressure_playbook_not_reducing_backlog" in self_layer["causal_diagnosis"]["root_causes"]
    assert "add_drain_outcome_verifier" in [row["gap"] for row in self_layer["capability_gaps"]]
    assert self_layer["reflex"]["action"] == "escalate_repeated_action_not_clearing_pressure"
    assert self_layer["reflex"]["command"] == [
        "./scripts/ops/opsctl.sh",
        "backpressure-super-drainer",
        "--apply",
        "--max-waves",
        "1",
        "--target-pending-lines",
        "5000",
        "--json",
    ]
    assert self_layer["reflex"]["followup_command"] == ["./scripts/ops/opsctl.sh", "ingestion-storage-control", "--json"]
    assert self_layer["reflex"]["verification_command"] == ["./scripts/ops/opsctl.sh", "system-intelligence", "--json"]
    assert self_layer["reflex"]["evidence_window"]["requires_single_sql_writer"] is True
    assert super_layer["overall_status"] == "advisory"
    assert super_layer["decision_packet"]["action"] == "run_outcome_verified_micro_drain"
    assert super_layer["decision_packet"]["owner"] == "backpressure_super_drainer"
    assert handoff["safe_next_command"] == self_layer["reflex"]["command"]
    assert "run_outcome_verified_micro_drain_then_measure" in handoff["needs_codex"]
    assert handoff["upgrade_integration"]["overall_status"] == "active"
    assert handoff["upgrade_integration"]["next_upgrade"] == "escalate_repeated_action_not_clearing_pressure"
    assert handoff["upgrade_integration"]["next_safe_command"] == self_layer["reflex"]["command"]
    assert handoff["upgrade_integration"]["plan"][0]["proof_metric"] == "pending_lines_delta<=-250"


def test_outcome_learning_credits_recent_verified_super_drainer_progress() -> None:
    signal_bus = {
        "summary": {"total_pending_lines": 493811},
        "signals": [
            {
                "name": "backpressure_super_drainer",
                "age_minutes": 2.0,
                "metrics": {
                    "initial_pending_lines": 759207,
                    "final_pending_lines": 628010,
                    "total_pending_lines": 628010,
                    "pending_lines_delta": 131197,
                    "waves_run": 1,
                    "progress_waves": 1,
                    "any_progress": True,
                    "stop_reason": "max_waves_reached",
                },
            }
        ],
    }
    super_intelligence = {
        "overall_status": "degraded",
        "decision_packet": {
            "action": "run_outcome_verified_micro_drain",
            "executive_mode": "rethink",
            "owner": "backpressure_super_drainer",
            "top_attention": "ingestion_storage",
        },
        "decision_quality_layer": {"quality_score": 47.1},
        "adversarial_simulation_layer": {
            "resilience_score": 57,
            "top_scenario": "storage_refill_after_cleanup",
        },
        "regime_drift_audit": {
            "overall_status": "degraded",
            "current_operational_regime": "storage_backpressure",
        },
        "adaptive_policy": {"guard_policy_mode": "full_schwab_observe"},
        "semantic_synthesis_layer": {"invalidators": ["decision_quality_low"]},
    }
    self_intelligence = {
        "action_effectiveness": {"verdict": "worsening"},
        "causal_diagnosis": {"primary_root_cause": "storage_backpressure_primary", "confidence": 0.76},
    }
    outcome_events = [
        {
            "timestamp_utc": "2026-05-19T19:00:00Z",
            "status": "degraded",
            "action": "run_outcome_verified_micro_drain",
            "pending_lines": 628010,
            "decision_quality_score": 67.1,
            "resilience_score": 57,
        }
    ]

    payload = src.build_outcome_learning(
        signal_bus=signal_bus,
        system_brain={},
        self_intelligence=self_intelligence,
        super_intelligence=super_intelligence,
        outcome_events=outcome_events,
    )

    assert payload["intervention_outcome"]["verdict"] == "effective"
    assert payload["overall_status"] == "ready"
    assert payload["drain_outcome_verifier"]["state"] == "verified_recent_progress"
    assert payload["drain_outcome_verifier"]["current_below_verified_final"] is True
    assert payload["drain_outcome_verifier"]["pending_lines_delta"] == 131197
    assert "recent_drain_progress_verified" in payload["causal_replay_scorer"]["replay_findings"]
    credit = payload["policy_credit_assignment"]["run_outcome_verified_micro_drain"]
    assert credit["credit_score"] > 50
    assert "verified_drain_delta=131197" in credit["evidence"]


def test_outcome_learning_marks_ineffective_so_far_as_advisory_proof_debt() -> None:
    signal_bus = {"summary": {"total_pending_lines": 9168}, "signals": []}
    super_intelligence = {
        "overall_status": "advisory",
        "decision_packet": {
            "action": "run_outcome_verified_micro_drain",
            "executive_mode": "rethink",
            "owner": "backpressure_super_drainer",
            "top_attention": "macro_event_intelligence",
        },
        "decision_quality_layer": {"quality_score": 59.3},
        "adversarial_simulation_layer": {"resilience_score": 57, "top_scenario": "stale_signal_false_clear"},
        "regime_drift_audit": {"overall_status": "advisory", "current_operational_regime": "expansion_rehearsal_ready"},
        "adaptive_policy": {"guard_policy_mode": "full_schwab_observe"},
        "semantic_synthesis_layer": {"invalidators": []},
    }
    self_intelligence = {
        "action_effectiveness": {"verdict": "ineffective_so_far"},
        "causal_diagnosis": {"primary_root_cause": "pressure_playbook_not_reducing_backlog", "confidence": 0.6},
    }
    outcome_events = [
        {
            "timestamp_utc": "2026-07-27T15:00:00Z",
            "status": "advisory",
            "action": "run_outcome_verified_micro_drain",
            "pending_lines": 9168,
            "decision_quality_score": 59.3,
            "resilience_score": 57,
        }
    ]

    payload = src.build_outcome_learning(
        signal_bus=signal_bus,
        system_brain={},
        self_intelligence=self_intelligence,
        super_intelligence=super_intelligence,
        outcome_events=outcome_events,
    )

    assert payload["intervention_outcome"]["verdict"] == "ineffective_so_far"
    assert payload["overall_status"] == "advisory"
    assert payload["playbook_mutation_guard"]["mutation_allowed"] is True


def test_outcome_learning_treats_read_only_replan_quality_drop_as_advisory_debt() -> None:
    signal_bus = {
        "summary": {
            "total_pending_lines": 4098,
            "blocked_signal_count": 0,
            "severe_signal_count": 0,
            "storage_critical": False,
            "memory_pressure_high": False,
            "runtime_pressure_high": False,
            "writer_recovery_required": False,
        },
        "signals": [
            {
                "name": "paper_live_data_standard",
                "status": "ready",
                "metrics": {
                    "paper_live_data_enabled_bots": 1584,
                    "collection_until_standard_bots": 148,
                    "data_collection_active_bots": 1732,
                    "direct_execution_allowed_bots": 0,
                    "live_trading_enabled_bots": 0,
                    "covered_by_paper_or_collection": True,
                    "full_eligible_paper_soak": True,
                },
            }
        ],
    }
    super_intelligence = {
        "overall_status": "advisory",
        "decision_packet": {
            "action": "reroute_stalled_playbook",
            "executive_mode": "rethink",
            "owner": "drainer_intelligence_layer",
            "top_attention": "pressure_playbook_not_reducing_backlog",
        },
        "decision_quality_layer": {"quality_score": 59.3},
        "adversarial_simulation_layer": {"resilience_score": 57, "top_scenario": "stale_signal_false_clear"},
        "regime_drift_audit": {"overall_status": "ready", "current_operational_regime": "expansion_rehearsal_ready"},
        "adaptive_policy": {"guard_policy_mode": "full_schwab_observe"},
        "semantic_synthesis_layer": {"invalidators": []},
    }
    self_intelligence = {
        "action_effectiveness": {"verdict": "worsening"},
        "causal_diagnosis": {"primary_root_cause": "pressure_playbook_not_reducing_backlog", "confidence": 0.6},
    }
    outcome_events = [
        {
            "timestamp_utc": "2026-07-27T16:20:00Z",
            "status": "degraded",
            "action": "cautious_expansion_rehearsal",
            "pending_lines": 4098,
            "decision_quality_score": 72.55,
            "resilience_score": 57,
        }
    ]

    payload = src.build_outcome_learning(
        signal_bus=signal_bus,
        system_brain={},
        self_intelligence=self_intelligence,
        super_intelligence=super_intelligence,
        outcome_events=outcome_events,
    )

    assert payload["intervention_outcome"]["verdict"] == "ineffective_so_far"
    assert payload["overall_status"] == "advisory"
    assert "read_only_replan_quality_debt" in payload["causal_replay_scorer"]["replay_findings"]
    credit = payload["policy_credit_assignment"]["reroute_stalled_playbook"]
    assert "quality_drop_is_read_only_replan_debt" in credit["evidence"]


def test_self_intelligence_uses_verified_drain_progress_before_escalating_repeated_action(tmp_path: Path) -> None:
    _seed_pressure_project(tmp_path)
    health = tmp_path / "governance" / "health"
    _write_json(
        health / "ingestion_storage_control_latest.json",
        {
            "overall_status": "blocked",
            "severity": "critical",
            "pressure_index": 1488.423,
            "backpressure": {"total_pending_lines": 493811, "core_pending_lines": 46505, "pending_lines_threshold": 15000},
        },
    )
    _write_json(
        health / "backpressure_super_drainer_latest.json",
        {
            "overall_status": "applied_with_followups",
            "active_drainer": "core_decision_drainer",
            "summary": {
                "initial_pending_lines": 759207,
                "final_pending_lines": 628010,
                "pending_lines_delta": 131197,
                "waves_run": 1,
                "progress_waves": 1,
                "any_progress": True,
                "stop_reason": "max_waves_reached",
            },
        },
    )
    _write_json(
        health / "drainer_intelligence_layer_latest.json",
        {
            "overall_status": "ready",
            "decision_packet": {
                "action": "wait_for_writer_then_re_score",
                "selected_drainer": "core_decision_drainer",
                "total_pending_lines": 493811,
                "target_pending_lines": 10000,
                "risk_flags": ["storage_critical", "writer_active"],
            },
        },
    )
    memory_path = tmp_path / "governance" / "system_intelligence" / "self_intelligence_memory.jsonl"
    memory_path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "timestamp_utc": f"2026-05-19T19:0{i}:00Z",
            "status": "degraded",
            "action": "relieve_pressure_then_micro_drain",
            "top_risk": "ingestion_storage",
            "pending_lines": 6806,
            "trajectory": "flat",
            "uncertainty_level": "low",
            "reflex_action": "follow_system_brain",
        }
        for i in range(3)
    ]
    memory_path.write_text("\n".join(json.dumps(row, ensure_ascii=True) for row in rows) + "\n", encoding="utf-8")

    payload = src.build_payload(tmp_path)
    self_layer = payload["system_self_intelligence"]
    storage_replay = payload["storage_causal_replay_memory"]
    handoff = payload["codex_handoff"]["attention_packet"]

    assert self_layer["drain_outcome_verifier"]["state"] == "verified_recent_progress"
    assert self_layer["action_effectiveness"]["verdict"] == "effective"
    assert self_layer["action_effectiveness"]["verified_drain_delta"] == 131197
    assert self_layer["action_effectiveness"]["measurement_rebased_by_verified_drain"] is True
    assert storage_replay["overall_status"] == "ready"
    assert storage_replay["memory_status"]["replay_ready"] is True
    assert storage_replay["memory_status"]["latest_verified_drain_delta"] == 131197
    assert handoff["storage_causal_replay"]["replay_ready"] is True
    assert handoff["storage_causal_replay"]["latest_verified_drain_delta"] == 131197
    assert self_layer["reflex"]["action"] == "follow_system_brain"
    assert "super_drainer_pending_total_drift_from_storage" not in self_layer["uncertainty"]["conflicting_signals"]
    assert "add_drain_outcome_verifier" not in [row["gap"] for row in self_layer["capability_gaps"]]
    assert "persist_storage_causal_replay_memory" not in [row["gap"] for row in self_layer["capability_gaps"]]
