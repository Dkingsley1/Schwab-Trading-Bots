from __future__ import annotations

import json
from pathlib import Path

from scripts.ops import deeper_intelligence_layers as src
from scripts.ops import system_intelligence_coordinator


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")


def _seed_project(root: Path, *, pressure: bool = False, unsafe_trade_authority: bool = False) -> None:
    rows = [
        {
            "bot_id": "brain_refinery_v1_test_core_bot",
            "active": True,
            "data_collection_active": True,
            "paper_live_data_enabled": True,
            "sleeve_profile": "core",
            "direct_execution_allowed": unsafe_trade_authority,
            "live_trading_enabled": unsafe_trade_authority,
        },
        {
            "bot_id": "brain_refinery_v2_test_macro_bot",
            "active": True,
            "data_collection_active": True,
            "paper_live_data_enabled": False,
            "sleeve_profile": "macro",
            "direct_execution_allowed": False,
            "live_trading_enabled": False,
        },
    ]
    _write_json(root / "master_bot_registry.json", {"sub_bots": rows})

    health = root / "governance" / "health"
    pending = 22000 if pressure else 0
    _write_json(
        health / "ingestion_storage_control_latest.json",
        {
            "overall_status": "degraded" if pressure else "ready",
            "severity": "critical" if pressure else "normal",
            "pressure_index": 4.2 if pressure else 0.0,
            "backpressure": {"total_pending_lines": pending, "pending_lines_threshold": 15000},
        },
    )
    _write_json(
        health / "runtime_throttle_control_latest.json",
        {
            "overall_status": "degraded" if pressure else "ready",
            "host_saturation_score": 77.0 if pressure else 25.0,
            "memory_pressure_level": "high" if pressure else "normal",
            "cpu_pressure_level": "watch",
        },
    )
    _write_json(
        health / "memory_efficiency_control_latest.json",
        {
            "overall_status": "degraded" if pressure else "ready",
            "memory_snapshot": {
                "memory_pressure_state": "yellow" if pressure else "green",
                "memory_pressure_kind": "compressor" if pressure else "none",
            },
        },
    )
    _write_json(
        health / "paper_live_data_standard_latest.json",
        {
            "overall_status": "ready",
            "counts_after": {
                "paper_live_data_enabled_bots": 35,
                "direct_execution_allowed_bots": 1 if unsafe_trade_authority else 0,
                "live_trading_enabled_bots": 1 if unsafe_trade_authority else 0,
            },
            "paper_lane_target": {"minimum": 30, "maximum": 50, "within_target_band": True},
        },
    )
    _write_json(health / "system_signal_bus_latest.json", {"overall_status": "ready", "summary": {"top_risk": "none"}})
    _write_json(health / "system_brain_latest.json", {"overall_status": "ready"})
    _write_json(health / "system_self_intelligence_latest.json", {"overall_status": "ready"})
    _write_json(health / "system_super_intelligence_latest.json", {"overall_status": "ready"})
    _write_json(health / "system_recursive_intelligence_latest.json", {"overall_status": "ready"})
    _write_json(health / "whole_system_governor_latest.json", {"overall_status": "ready"})
    _write_json(health / "platform_brain_v6_latest.json", {"overall_status": "ready", "section_count": 15})
    _write_json(health / "guard_intelligence_latest.json", {"overall_status": "ready", "policy_mode": "full_schwab_observe"})
    _write_json(health / "process_watchdog_latest.json", {"overall_status": "ready", "status": []})
    _write_json(health / "process_fanout_guard_latest.json", {"overall_status": "ready", "summary": {"triggered": False}})
    _write_json(health / "live_runtime_separation_control_latest.json", {"overall_status": "ready", "live_plane": {"live_lane_running": True}})
    _write_json(health / "global_killswitch_latest.json", {"overall_status": "ready", "halt": False})
    _write_json(health / "auth_lease_manager_latest.json", {"overall_status": "ready", "lease_state": "healthy"})
    _write_json(health / "training_quality_control_latest.json", {"overall_status": "ready"})
    _write_json(health / "commands_contract_latest.json", {"entry_count": 200})
    _write_json(health / "documentation_reporting_intelligence_latest.json", {"overall_status": "ready"})
    _write_json(health / "codex_handoff_latest.json", {"overall_status": "ready"})
    _write_json(health / "artifact_freshness_slo_latest.json", {"overall_status": "ready"})


def test_build_payload_installs_all_ten_self_awareness_layers(tmp_path: Path) -> None:
    _seed_project(tmp_path)

    payload = src.build_payload(tmp_path)
    layer_ids = {row["layer_id"] for row in payload["layers"]}

    assert payload["layer_count"] == 10
    assert layer_ids == {row["layer_id"] for row in src.LAYER_DEFINITIONS}
    assert payload["contract"]["models_may_override_global_halt"] is False
    assert payload["layer_map"]["causal_world_model"]["decision"] == "rank_root_causes_before_restart_retrain_or_expansion"
    assert payload["layer_map"]["living_ontology_memory_graph"]["evidence"] == ["registered_bots:2", "sleeve_profiles:2"]
    assert payload["layer_map"]["operator_dialogue"]["does_not_execute_trades"] is True
    assert payload["operator_dialogue_packet"]["safe_next_command"]


def test_pressure_and_trade_authority_close_promotion_and_constitution(tmp_path: Path) -> None:
    _seed_project(tmp_path, pressure=True, unsafe_trade_authority=True)

    payload = src.build_payload(tmp_path)

    assert payload["overall_status"] == "blocked"
    assert payload["layer_map"]["resource_economist"]["overall_status"] in {"advisory", "degraded"}
    assert "resource_budget_protective_mode" in payload["layer_map"]["resource_economist"]["blockers"]
    assert payload["layer_map"]["promotion_court"]["overall_status"] == "blocked"
    assert "unexpected_trade_authority_detected" in payload["layer_map"]["promotion_court"]["blockers"]
    assert payload["layer_map"]["constitutional_risk"]["overall_status"] == "blocked"
    assert "trade_authority_invariant_violation" in payload["layer_map"]["constitutional_risk"]["blockers"]


def test_apply_writes_health_config_contract_and_pycharm_docs(tmp_path: Path) -> None:
    _seed_project(tmp_path)
    payload = src.build_payload(tmp_path)
    health_path = tmp_path / "governance" / "health" / "deeper_intelligence_layers_latest.json"
    config_path = tmp_path / "config" / "deeper_intelligence_layers_v1.json"
    contract_path = tmp_path / "governance" / "system_intelligence" / "deeper_intelligence_layers_contract.json"
    markdown_path = tmp_path / "exports" / "reports" / "operator" / "deeper_intelligence_layers_latest.md"
    pycharm_path = tmp_path / "docs" / "pycharm" / "deeper_intelligence_layers_latest.md"

    src.write_outputs(
        payload,
        health_path=health_path,
        config_path=config_path,
        contract_path=contract_path,
        markdown_path=markdown_path,
        pycharm_path=pycharm_path,
    )

    assert json.loads(health_path.read_text(encoding="utf-8"))["layer_count"] == 10
    assert json.loads(config_path.read_text(encoding="utf-8"))["hard_invariants"]["collect_only_until_promotion_court"] is True
    assert "constitutional_risk" in contract_path.read_text(encoding="utf-8")
    assert "# Deeper Intelligence Layers" in markdown_path.read_text(encoding="utf-8")
    assert "Causal World Model Layer" in pycharm_path.read_text(encoding="utf-8")


def test_system_signal_bus_consumes_deeper_layer_artifact(tmp_path: Path) -> None:
    _seed_project(tmp_path)
    payload = src.build_payload(tmp_path)
    src.write_outputs(
        payload,
        health_path=tmp_path / "governance" / "health" / "deeper_intelligence_layers_latest.json",
        config_path=tmp_path / "config" / "deeper_intelligence_layers_v1.json",
        contract_path=tmp_path / "governance" / "system_intelligence" / "deeper_intelligence_layers_contract.json",
        markdown_path=tmp_path / "exports" / "reports" / "operator" / "deeper_intelligence_layers_latest.md",
        pycharm_path=tmp_path / "docs" / "pycharm" / "deeper_intelligence_layers_latest.md",
    )

    signal_bus = system_intelligence_coordinator.build_signal_bus(tmp_path)
    signal = next(row for row in signal_bus["signals"] if row["name"] == "deeper_intelligence_layers")

    assert signal["loaded"] is True
    assert signal["metrics"]["layer_count"] == 10
    assert signal["metrics"]["authority_boundary"] == "advisory_control_plane_with_constitutional_lockout_attestation"
