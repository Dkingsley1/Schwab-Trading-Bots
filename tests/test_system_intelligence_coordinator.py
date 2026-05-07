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
    assert handoff["attention_packet"]["causal_root"] == "storage_backpressure_primary"
    assert handoff["attention_packet"]["action_effectiveness"] == "insufficient_history"
    assert handoff["attention_packet"]["integration_route"] == "storage_first_recovery"
    assert handoff["attention_packet"]["outcome_verdict"] == "baseline"
    assert handoff["attention_packet"]["recursive_status"] in {"ready", "advisory", "degraded"}
    assert handoff["attention_packet"]["next_more_advanced_layer"] == "cognitive_twin_counterfactual_simulator"
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
    recursive_path = tmp_path / "governance" / "health" / "system_recursive_intelligence_latest.json"
    documentation_reporting_path = tmp_path / "governance" / "health" / "documentation_reporting_intelligence_latest.json"
    handoff_path = tmp_path / "governance" / "health" / "codex_handoff_latest.json"
    handoff_md_path = tmp_path / "exports" / "reports" / "operator" / "codex_handoff_latest.md"
    memory_path = tmp_path / "governance" / "system_intelligence" / "self_intelligence_memory.jsonl"
    super_memory_path = tmp_path / "governance" / "system_intelligence" / "super_intelligence_memory.jsonl"
    outcome_memory_path = tmp_path / "governance" / "system_intelligence" / "intervention_outcomes.jsonl"
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
        recursive_intelligence_path=recursive_path,
        documentation_reporting_path=documentation_reporting_path,
        handoff_path=handoff_path,
        handoff_markdown_path=handoff_md_path,
        memory_path=memory_path,
        super_memory_path=super_memory_path,
        outcome_memory_path=outcome_memory_path,
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
    assert recursive_path.exists()
    assert documentation_reporting_path.exists()
    assert handoff_path.exists()
    assert memory_path.exists()
    assert super_memory_path.exists()
    assert outcome_memory_path.exists()
    assert recursive_memory_path.exists()
    assert super_override_path.exists()
    assert pycharm_index_path.exists()
    assert pycharm_index_json_path.exists()
    assert context_path.exists()
    assert "# Codex Handoff" in handoff_md_path.read_text(encoding="utf-8")
    assert "Causal Root" in handoff_md_path.read_text(encoding="utf-8")
    assert "Super Intelligence" in handoff_md_path.read_text(encoding="utf-8")
    assert "Outcome Learning" in handoff_md_path.read_text(encoding="utf-8")
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
    assert "recursive_score" in recursive_memory_path.read_text(encoding="utf-8")
    assert "SUPER_INTELLIGENCE_EXECUTIVE_MODE=drain" in super_override_path.read_text(encoding="utf-8")
    assert "SUPER_INTELLIGENCE_OPERATIONAL_REGIME=storage_backpressure" in super_override_path.read_text(encoding="utf-8")
    assert "SUPER_INTELLIGENCE_OBJECTIVE_GUARDRAIL_STATUS=ready" in super_override_path.read_text(encoding="utf-8")


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
            "overall_status": "ready",
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
    assert watchdog["severity_score"] < 70


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

    assert self_layer["learning_memory"]["same_action_repeat_count"] == 3
    assert self_layer["action_effectiveness"]["same_action_run_length"] == 4
    assert self_layer["action_effectiveness"]["verdict"] == "ineffective_so_far"
    assert "pressure_playbook_not_reducing_backlog" in self_layer["causal_diagnosis"]["root_causes"]
    assert "add_drain_outcome_verifier" in [row["gap"] for row in self_layer["capability_gaps"]]
    assert self_layer["reflex"]["action"] == "escalate_repeated_action_not_clearing_pressure"
