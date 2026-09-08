from __future__ import annotations

import json
from pathlib import Path

import pytest

from core.system_role_contracts import (
    RoleAuthorityError,
    build_contract_report,
    component_action_guard,
    evaluate_component_action,
    validate_contract,
)
from scripts.ops import artifact_freshness_slo, runtime_artifact_refresh, runtime_gate_dashboard


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _copy_contract(project_root: Path) -> dict:
    contract = json.loads(
        (PROJECT_ROOT / "config" / "system_role_contracts_v1.json").read_text(encoding="utf-8")
    )
    path = project_root / "config" / "system_role_contracts_v1.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(contract), encoding="utf-8")
    return contract


def test_repository_role_contract_is_complete_and_conflict_free() -> None:
    report = build_contract_report(PROJECT_ROOT)

    assert report["ok"] is True
    assert report["grade"] == "A+"
    assert report["summary"]["role_count"] >= 15
    assert report["summary"]["component_count"] >= 23
    assert report["summary"]["state_domain_count"] >= 23
    assert report["summary"]["registry_role_coverage_ratio"] == 1.0
    assert report["summary"]["authority_conflict_count"] == 0
    assert report["blockers"] == []


def test_repository_wires_role_contract_into_freshness_refresh_dashboard_and_live_bar() -> None:
    freshness = artifact_freshness_slo._artifact_contract(PROJECT_ROOT)
    dashboard = runtime_gate_dashboard._artifact_config(PROJECT_ROOT)
    refresh = {row["name"]: row for row in runtime_artifact_refresh._step_specs(PROJECT_ROOT)}
    readiness = json.loads(
        (PROJECT_ROOT / "config" / "production_readiness_control_v1.json").read_text(encoding="utf-8")
    )
    capabilities = {
        row["capability_id"]: row
        for row in readiness["live_money_production_bar"]["required_capabilities"]
    }

    assert freshness["system_role_contract"]["required"] is True
    assert dashboard["system_role_contract"]["required"] is True
    assert refresh["system_role_contract"]["depends_on"] == [
        "bot_organization_control",
        "control_surface_ownership",
    ]
    assert "system_role_contract" in refresh["artifact_freshness_slo_post_master"]["depends_on"]
    assert capabilities["system_role_contract"]["required"] is True


def test_runtime_authority_allows_only_the_declared_execution_owner(tmp_path: Path) -> None:
    _copy_contract(tmp_path)

    allowed = evaluate_component_action(
        tmp_path,
        component_id="live_execution_gateway",
        action="live_submit",
        state_domain="live_order_submission",
    )
    denied = evaluate_component_action(
        tmp_path,
        component_id="strategy_fleet",
        action="live_submit",
        state_domain="live_order_submission",
    )

    assert allowed["ok"] is True
    assert denied["ok"] is False
    assert "component_action_not_allowed" in denied["blockers"]
    assert "exclusive_action_owned_by_other_component" in denied["blockers"]
    assert "state_domain_owned_by_other_component" in denied["blockers"]


def test_unknown_or_missing_role_contract_fails_closed(tmp_path: Path) -> None:
    decision = evaluate_component_action(
        tmp_path,
        component_id="paper_execution_gateway",
        action="paper_submit",
        state_domain="paper_order_submission",
    )

    assert decision["ok"] is False
    assert "system_role_contract_invalid" in decision["blockers"]


def test_duplicate_state_resource_writer_is_rejected(tmp_path: Path) -> None:
    contract = _copy_contract(tmp_path)
    duplicate = dict(contract["state_domains"][0])
    duplicate["domain_id"] = "duplicate_writer_claim"
    duplicate["writer_component_id"] = "storage_lifecycle_controller"
    contract["state_domains"].append(duplicate)
    contract["components"] = [dict(row) for row in contract["components"]]
    for component in contract["components"]:
        if component.get("component_id") == "storage_lifecycle_controller":
            component["state_domains"] = list(component["state_domains"]) + ["duplicate_writer_claim"]
            component["allowed_actions"] = list(component["allowed_actions"]) + [duplicate["required_action"]]
            break

    result = validate_contract(contract, check_sources=False)

    assert result["ok"] is False
    assert any(item.startswith("state_resource_writer_conflict:") for item in result["blockers"])


def test_sensitive_action_lease_is_single_flight(tmp_path: Path) -> None:
    _copy_contract(tmp_path)

    with component_action_guard(
        tmp_path,
        component_id="paper_execution_gateway",
        action="paper_submit",
        state_domain="paper_order_submission",
    ):
        with pytest.raises(RoleAuthorityError, match="sensitive_action_lease_busy"):
            with component_action_guard(
                tmp_path,
                component_id="paper_execution_gateway",
                action="paper_submit",
                state_domain="paper_order_submission",
            ):
                pass


def test_contract_cannot_weaken_required_fields_or_create_escalation_cycle(tmp_path: Path) -> None:
    contract = _copy_contract(tmp_path)
    contract["required_role_fields"].remove("purpose")
    roles = {row["role_id"]: row for row in contract["roles"]}
    roles["data_collector"]["escalation_owner"] = "context_processor"
    roles["context_processor"]["escalation_owner"] = "data_collector"

    result = validate_contract(contract, check_sources=False)

    assert result["ok"] is False
    assert "required_role_field_policy_missing:purpose" in result["blockers"]
    assert any(item.startswith("escalation_cycle:") for item in result["blockers"])


def test_exclusive_action_rejects_a_second_component_claimant(tmp_path: Path) -> None:
    contract = _copy_contract(tmp_path)
    for component in contract["components"]:
        if component.get("component_id") == "strategy_fleet":
            component["allowed_actions"].append("live_submit")
            break

    result = validate_contract(contract, check_sources=False)

    assert result["ok"] is False
    assert any(
        item.startswith("exclusive_action:live_submit:component_claimants_invalid")
        for item in result["blockers"]
    )
