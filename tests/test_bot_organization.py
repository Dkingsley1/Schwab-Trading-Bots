from __future__ import annotations

import copy
import json
from pathlib import Path

from core import bot_organization
from scripts.ops import artifact_freshness_slo
from scripts.ops import bot_organization_control
from scripts.ops import runtime_artifact_refresh
from scripts.ops import runtime_gate_dashboard
from scripts.ops import source_mutation_guard

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _policy() -> dict:
    return json.loads(
        (PROJECT_ROOT / "config" / "bot_organization_v1.json").read_text(
            encoding="utf-8"
        )
    )


def _row(bot_id: str, **overrides: object) -> dict:
    row = {
        "bot_id": bot_id,
        "bot_role": "signal_sub_bot",
        "active": True,
        "lifecycle_state": "paper_live_data",
        "paper_trading_enabled": True,
        "sleeve_profile": "equity_core",
        "sleeve_family": "equity",
        "strategy_family": "trend_and_momentum",
        "horizon_id": "daily_to_multiday",
        "preferred_regimes": ["risk_on_trend"],
        "regime_axes": {
            "market_direction": ["bull_trend"],
            "volatility_state": ["normal"],
            "liquidity_state": ["normal"],
            "macro_state": ["growth_expansion"],
            "rates_credit_state": ["neutral"],
            "correlation_state": ["stable"],
            "event_phase": ["continuous"],
            "market_session": ["continuous"],
        },
    }
    row.update(overrides)
    return row


def test_policy_is_shadow_only_and_fail_closed() -> None:
    policy = _policy()

    assert bot_organization.validate_policy(policy) == []
    assert policy["bot_setup_contract"]["contract_id"] == "bot_setup_contract_v1"

    unsafe = copy.deepcopy(policy)
    unsafe["safety_contract"]["live_execution_authority"] = True
    assert (
        "organization_safety_live_execution_authority_must_be_false"
        in bot_organization.validate_policy(unsafe)
    )

    unsafe_setup = copy.deepcopy(policy)
    unsafe_setup["bot_setup_contract"]["authority"]["can_submit_live_order"] = True
    assert (
        "organization_bot_setup_authority_can_submit_live_order_must_be_false"
        in bot_organization.validate_policy(unsafe_setup)
    )

    unsafe_tripwire = copy.deepcopy(policy)
    unsafe_tripwire["tripwire_contract"]["authority"]["can_submit_live_order"] = True
    assert (
        "organization_tripwire_authority_can_submit_live_order_must_be_false"
        in bot_organization.validate_policy(unsafe_tripwire)
    )


def test_registry_receives_one_provenance_backed_assignment_per_bot() -> None:
    registry = {
        "sub_bots": [
            _row("alpha_trend"),
            _row(
                "alpha_mean_reversion",
                sleeve_profile="intraday_aggressive",
                strategy_family="mean_reversion",
                horizon_id="intraday",
                preferred_regimes=["rangebound"],
            ),
        ]
    }

    result = bot_organization.organize_registry(registry, _policy())

    assert result["ok"] is True
    assert result["organization_coverage_ratio"] == 1.0
    assert result["unique_assignment_ratio"] == 1.0
    assert result["high_confidence_ratio"] == 1.0
    assert result["regime_axis_coverage_ratio"] == 1.0
    assert result["regime_axis_specificity_ratio"] == 1.0
    assert result["regime_quality_grade"] == "A+"
    assert result["grade"] == "A+"
    assert all(row["regime_profile_id"] for row in result["assignments"])
    assert all(
        row["authority"]["organization_layer_execution_authority"] is False
        for row in result["assignments"]
    )
    assert result["bot_setup_summary"]["hardening"]["overall_status"] == "ready"
    assert result["bot_setup_summary"]["setup_coverage_ratio"] == 1.0
    assert result["bot_setup_summary"]["tier_counts"]["sub"] == 2
    assert result["tripwire_summary"]["hardening"]["overall_status"] == "ready"
    assert result["tripwire_summary"]["tripwire_count"] >= 10
    assert result["blocking_tripwire_count"] == 0
    assert result["tripwire_contract"]["authority"]["metadata_only"] is True
    contract = result["operating_contract"]
    assert contract["complete"] is True
    assert contract["domain"] == "bot_organization"
    assert "automatic_registry_mutation" in contract["blocked_authority"]
    assert contract["hardening"]["metadata_authority_only"] is True
    assert all(row["setup_tier"] == "sub" for row in result["assignments"])
    assert all(row["setup_role_group"] == "signal" for row in result["assignments"])
    assert all(
        row["setup_lifecycle_state"] == "paper_live_data"
        for row in result["assignments"]
    )
    assert result["regime_metadata_access_ratio"] == 1.0
    assert result["regime_metadata_access_grade"] == "A+"
    assert all(
        row["regime_metadata_access"]["access_ready"] is True
        for row in result["assignments"]
    )


def test_duplicate_bot_identity_blocks_catalog() -> None:
    registry = {"sub_bots": [_row("duplicate"), _row("duplicate")]}

    result = bot_organization.organize_registry(registry, _policy())

    assert result["ok"] is False
    assert result["duplicate_bot_ids"] == ["duplicate"]
    assert "duplicate_registry_bot_ids" in result["blockers"]
    assert "tripwire:duplicate_bot_identity" in result["blockers"]
    assert result["blocking_tripwire_count"] >= 1
    assert any(
        row["tripwire_id"] == "duplicate_bot_identity"
        for row in result["blocking_tripwires"]
    )


def test_setup_contract_blocks_unknown_lifecycle_state() -> None:
    registry = {"sub_bots": [_row("bad_lifecycle", lifecycle_state="needs_decision")]}

    result = bot_organization.organize_registry(registry, _policy())

    assert result["ok"] is False
    assert "bot_setup_contract_hardening_failed" in result["blockers"]
    assert (
        "assignment_lifecycle_states_are_known"
        in result["setup_hardening_failed_checks"]
    )
    assert result["bot_setup_summary"]["hardening"]["overall_status"] == "blocked"


def test_module_spec_parser_never_imports_or_executes_module(tmp_path: Path) -> None:
    marker = tmp_path / "executed"
    module = tmp_path / "bot.py"
    module.write_text(
        "BOT_SPEC = {'bot_id': 'literal', 'sleeve_profile': 'safe'}\n"
        f"Path({str(marker)!r}).write_text('bad')\n",
        encoding="utf-8",
    )

    spec, error = bot_organization.load_literal_bot_spec(module)

    assert error == ""
    assert spec["sleeve_profile"] == "safe"
    assert not marker.exists()


def test_control_build_is_path_isolated_and_does_not_write(tmp_path: Path) -> None:
    config = tmp_path / "config" / "bot_organization_v1.json"
    registry = tmp_path / "master_bot_registry.json"
    catalog = tmp_path / "core" / "bot_catalog.json"
    hierarchy_out = tmp_path / "governance" / "bot_organization" / "hierarchy.json"
    config.parent.mkdir(parents=True)
    catalog.parent.mkdir(parents=True)
    config.write_text(json.dumps(_policy()), encoding="utf-8")
    registry.write_text(json.dumps({"sub_bots": [_row("alpha")]}), encoding="utf-8")
    catalog.write_text(
        json.dumps({"bots": [{"bot_id": "alpha", "category": "general_signal"}]}),
        encoding="utf-8",
    )

    health, hierarchy = bot_organization_control.build_payload(
        tmp_path,
        config_path=config,
        registry_path=registry,
        catalog_input_path=catalog,
        hierarchy_out_path=hierarchy_out,
    )

    assert health["ok"] is True
    assert health["hierarchy_catalog"]["path"] == str(hierarchy_out)
    assert health["hierarchy_contract"]["bot_setup_profile_recorded"] is True
    assert health["hierarchy_contract"]["bot_setup_hardening_ready"] is True
    assert hierarchy["bot_setup_summary"]["hardening"]["overall_status"] == "ready"
    assert hierarchy["assignment_count"] == 1
    assert not hierarchy_out.exists()


def test_repository_registry_is_fully_organized() -> None:
    health, hierarchy = bot_organization_control.build_payload(PROJECT_ROOT)

    assert health["ok"] is True
    assert health["structural_grade"] == "A+"
    assert health["organization_coverage_ratio"] == 1.0
    assert health["unique_assignment_ratio"] == 1.0
    assert health["setup_coverage_ratio"] == 1.0
    assert health["setup_hardening_status"] == "ready"
    assert health["bot_setup_summary"]["hardening"]["overall_status"] == "ready"
    assert health["tripwire_summary"]["hardening"]["overall_status"] == "ready"
    assert health["blocking_tripwire_count"] == 0
    assert health["hierarchy_contract"]["tripwire_contract_recorded"] is True
    assert health["hierarchy_contract"]["tripwire_hardening_ready"] is True
    assert health["bot_setup_summary"]["active_tier_counts"]["master"] >= 1
    assert health["bot_setup_summary"]["active_tier_counts"]["grand_master"] >= 1
    assert (
        health["high_confidence_ratio"]
        >= _policy()["hierarchy"]["minimum_high_confidence_ratio"]
    )
    assert health["regime_model_contract"]["axis_ids"] == [
        "market_direction",
        "volatility_state",
        "liquidity_state",
        "macro_state",
        "rates_credit_state",
        "correlation_state",
        "event_phase",
        "market_session",
        "operational_state",
    ]
    assert health["regime_axis_coverage_ratio"] > 0.0
    assert health["hard_limit_shadow_cells"] == []
    assert hierarchy["assignment_count"] == health["registry_bot_count"]
    assert hierarchy["regime_model_id"] == "multi_axis_regime_taxonomy_v1"
    assert hierarchy["authority_contract"]["live_execution_authority"] is False
    assert health["invalid_regime_scenario_profile_count"] == 0
    assert health["overbroad_regime_profile_count"] == 0
    assert health["regime_scenario_profile_count"] >= 1
    assert health["regime_scenario_count"] >= 7
    assert health["regime_metadata_access_ready_count"] == health["registry_bot_count"]
    assert health["regime_metadata_access_ratio"] == 1.0
    assert health["regime_metadata_access_grade"] == "A+"
    assert health["regime_metadata_access_error_count"] == 0
    assert hierarchy["regime_model_contract"]["metadata_access_version"] == (
        "regime_metadata_access_v1"
    )
    assert hierarchy["authority_contract"]["metadata_only"] is True
    assert hierarchy["bot_setup_contract"]["authority"]["metadata_only"] is True
    assert hierarchy["tripwire_contract"]["authority"]["metadata_only"] is True
    assert hierarchy["tripwire_summary"]["hardening"]["overall_status"] == "ready"
    target = next(
        row
        for row in hierarchy["assignments"]
        if row["bot_id"]
        == "brain_refinery_v1358_platform_organ_regime_router_optimization_modeler_bot"
    )
    assert target["regime_scope"] == "operational_control"
    assert target["regime_scenario_partitioned"] is True
    assert target["regime_scenario_count"] == 7
    assert target["review_reasons"] == []
    assert target["authority"]["organization_layer_execution_authority"] is False


def test_repository_wiring_keeps_organization_evidence_required_and_protected() -> None:
    refresh_steps = {
        row["name"]: row for row in runtime_artifact_refresh._step_specs(PROJECT_ROOT)
    }
    freshness = artifact_freshness_slo._artifact_contract(PROJECT_ROOT)
    dashboard = runtime_gate_dashboard._artifact_config(PROJECT_ROOT)
    ownership = json.loads(
        (PROJECT_ROOT / "config" / "control_surface_ownership_v1.json").read_text(
            encoding="utf-8"
        )
    )
    owned_resources = {
        str(row.get("resource_path") or "") for row in ownership.get("controls", [])
    }

    assert refresh_steps["bot_organization_control"]["payload_path"] == (
        PROJECT_ROOT / "governance" / "health" / "bot_organization_latest.json"
    )
    assert freshness["bot_organization_control"]["required"] is True
    assert dashboard["bot_organization_control"]["required"] is True
    assert "core/bot_organization.py" in source_mutation_guard.DEFAULT_PROTECTED_PATHS
    assert (
        "core/hierarchical_ensemble.py" in source_mutation_guard.DEFAULT_PROTECTED_PATHS
    )
    assert "core/regime_taxonomy.py" in source_mutation_guard.DEFAULT_PROTECTED_PATHS
    assert "governance/health/bot_organization_latest.json" in owned_resources
    assert "governance/bot_organization/bot_hierarchy_latest.json" in owned_resources

    ci_text = (PROJECT_ROOT / ".github" / "workflows" / "ci_guardrails.yml").read_text(
        encoding="utf-8"
    )
    opsctl_text = (PROJECT_ROOT / "scripts" / "ops" / "opsctl.sh").read_text(
        encoding="utf-8"
    )
    assert "bot_organization_control.py" in ci_text
    assert (
        "bot-organization|bot-hierarchy|sleeve-subsections|hierarchical-bots"
        in opsctl_text
    )


def test_dashboard_summary_preserves_multi_axis_regime_quality() -> None:
    payload = {
        "overall_status": "ready_with_review_debt",
        "grade": "B",
        "structural_grade": "A+",
        "registry_bot_count": 10,
        "organized_bot_count": 10,
        "organization_coverage_ratio": 1.0,
        "high_confidence_ratio": 0.9,
        "regime_quality_grade": "C",
        "regime_axis_coverage_ratio": 0.75,
        "regime_axis_specificity_ratio": 0.6,
        "regime_review_count": 3,
        "regime_scenario_profile_count": 2,
        "regime_scenario_count": 7,
        "regime_scenario_review_count": 0,
        "invalid_regime_scenario_profile_count": 0,
        "overbroad_regime_profile_count": 0,
        "unknown_regime_profile_count": 2,
        "regime_metadata_access_grade": "A+",
        "regime_metadata_access_ready_count": 10,
        "regime_metadata_access_ratio": 1.0,
        "regime_metadata_context_required_count": 2,
        "regime_metadata_access_error_count": 0,
        "bot_setup_summary": {
            "contract_id": "bot_setup_contract_v1",
            "setup_coverage_ratio": 1.0,
            "tier_counts": {"sub": 8, "master": 1, "grand_master": 1},
            "role_group_counts": {"signal": 7, "risk": 2, "coordination": 1},
            "lifecycle_state_counts": {"paper_live_data": 10},
            "missing_setup_metadata_count": 0,
            "hardening": {"overall_status": "ready"},
        },
        "tripwire_summary": {
            "overall_status": "active_advisory",
            "active_tripwire_count": 2,
            "blocking_tripwire_count": 0,
            "severity_counts": {"advisory": 2},
            "category_counts": {"classification_quality": 2},
            "hardening": {"overall_status": "ready"},
        },
        "review_queue_count": 3,
        "hard_limit_shadow_cells": [],
    }

    summary = runtime_gate_dashboard._artifact_summary(
        "bot_organization_control", payload
    )

    assert summary["regime_quality_grade"] == "C"
    assert summary["regime_axis_coverage_ratio"] == 0.75
    assert summary["regime_axis_specificity_ratio"] == 0.6
    assert summary["regime_review_count"] == 3
    assert summary["regime_scenario_profile_count"] == 2
    assert summary["regime_scenario_count"] == 7
    assert summary["regime_scenario_review_count"] == 0
    assert summary["invalid_regime_scenario_profile_count"] == 0
    assert summary["overbroad_regime_profile_count"] == 0
    assert summary["unknown_regime_profile_count"] == 2
    assert summary["regime_metadata_access_grade"] == "A+"
    assert summary["regime_metadata_access_ready_count"] == 10
    assert summary["regime_metadata_access_ratio"] == 1.0
    assert summary["regime_metadata_access_error_count"] == 0
    assert summary["bot_setup_contract_id"] == "bot_setup_contract_v1"
    assert summary["setup_hardening_status"] == "ready"
    assert summary["setup_coverage_ratio"] == 1.0
    assert summary["setup_tier_counts"] == {"sub": 8, "master": 1, "grand_master": 1}
    assert summary["setup_role_group_counts"] == {
        "signal": 7,
        "risk": 2,
        "coordination": 1,
    }
    assert summary["setup_lifecycle_state_counts"] == {"paper_live_data": 10}
    assert summary["missing_setup_metadata_count"] == 0
    assert summary["tripwire_status"] == "active_advisory"
    assert summary["tripwire_hardening_status"] == "ready"
    assert summary["active_tripwire_count"] == 2
    assert summary["blocking_tripwire_count"] == 0
    assert summary["tripwire_severity_counts"] == {"advisory": 2}
