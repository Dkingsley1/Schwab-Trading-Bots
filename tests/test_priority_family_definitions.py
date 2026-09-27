from copy import deepcopy
import json
from pathlib import Path

import pytest

from core.institutional_decision_flow import (
    build_research_priority_catalog,
    load_policy,
    resolve_sleeve_policy,
)
from core.system_role_contracts import (
    _infrastructure_definition_report,
    evaluate_component_action,
    validate_contract,
)
from scripts.ops.sleeve_alpha_toolbox_control import build_payload, render_markdown

ROOT = Path(__file__).resolve().parents[1]


def role_contract():
    return json.loads((ROOT / "config/system_role_contracts_v1.json").read_text())


def test_five_priority_families_and_ten_family_catalog_are_explicit():
    policy = load_policy()
    before = deepcopy(policy)
    catalog = build_research_priority_catalog(policy)
    assert [row["family_id"] for row in catalog["families"][:5]] == [
        "swing_directional",
        "intraday_momentum",
        "relative_value",
        "macro_rates_fx",
        "long_horizon_income",
    ]
    assert len(catalog["deferred_family_ids"]) == 5
    assert len(catalog["families"]) == len(policy["sleeve_policy_families"])
    assert catalog["recommended_concurrent_family_reviews"] == 3
    assert catalog["workers_started"] is False
    assert catalog["profitability_verified"] is False
    assert catalog["bot_mandate"]["one_primary_family"] is True
    infra = next(
        row
        for row in catalog["families"]
        if row["family_id"] == "infrastructure_control"
    )
    assert infra["research_status"] == "non_trading"
    assert infra["research_rank"] is None
    catalog["families"][0]["research_definition"]["hypothesis"] = "edited"
    assert policy == before


def test_research_metadata_does_not_change_any_registered_execution_receipt():
    policy = load_policy()
    old = deepcopy(policy)
    old.pop("research_priority_contract")
    changed = deepcopy(policy)
    changed["research_priority_contract"]["priority_families"][0][
        "hypothesis"
    ] = "Different research wording"
    for profile in policy["profile_policy_map"]:
        assert (
            resolve_sleeve_policy(profile, policy)[1]
            == resolve_sleeve_policy(profile, old)[1]
        )
        assert (
            resolve_sleeve_policy(profile, policy)[1]
            == resolve_sleeve_policy(profile, changed)[1]
        )
    assert (
        build_research_priority_catalog(policy)["catalog_sha256"]
        != build_research_priority_catalog(changed)["catalog_sha256"]
    )


@pytest.mark.parametrize(
    "case",
    [
        "missing",
        "authority",
        "string_authority",
        "rank",
        "bool_rank",
        "unknown",
        "duplicate",
        "infra",
        "missing_inputs",
        "bad_benchmarks",
        "duplicate_deferred",
        "unknown_deferred",
        "review_over_cap",
        "bool_review",
        "empty_evidence",
        "multi_family",
    ],
)
def test_invalid_or_authority_bearing_research_catalog_is_rejected(case):
    policy = load_policy()
    contract = policy["research_priority_contract"]
    first = contract["priority_families"][0]
    if case == "missing":
        policy.pop("research_priority_contract")
    elif case in {"authority", "string_authority"}:
        contract["authority"]["allocates_capital"] = (
            True if case == "authority" else "false"
        )
    elif case == "rank":
        first["rank"] = 2
    elif case == "bool_rank":
        first["rank"] = True
    elif case == "unknown":
        first["family_id"] = "invented"
    elif case == "duplicate":
        first["family_id"] = contract["priority_families"][1]["family_id"]
    elif case == "infra":
        first["family_id"] = "infrastructure_control"
    elif case == "missing_inputs":
        first.pop("inputs_required")
    elif case == "bad_benchmarks":
        first["benchmarks"] = [False]
    elif case == "duplicate_deferred":
        contract["deferred_family_ids"][0] = first["family_id"]
    elif case == "unknown_deferred":
        contract["deferred_family_ids"][0] = "missing"
    elif case == "review_over_cap":
        contract["recommended_concurrent_family_reviews"] = 6
    elif case == "bool_review":
        contract["recommended_concurrent_family_reviews"] = True
    elif case == "empty_evidence":
        contract["required_evidence"] = []
    elif case == "multi_family":
        contract["bot_mandate"]["one_primary_family"] = False
    with pytest.raises(ValueError):
        build_research_priority_catalog(policy)


def test_native_toolbox_publishes_priorities_without_earning_evidence():
    payload = build_payload(ROOT)
    catalog = payload["research_priority_catalog"]
    assert not any(catalog["authority"].values())
    assert all(
        row["research_priority"]["definition_only"] for row in payload["sleeve_routes"]
    )
    assert all(
        row["economic_evidence_status"] == "not_assessed_by_definition"
        for row in catalog["families"]
    )
    rendered = render_markdown(payload)
    assert "## Priority Research Families" in rendered
    assert "### 1. Swing Trend" in rendered
    assert "Tail hedges are judged on portfolio protection" in rendered


def test_infrastructure_definitions_inherit_real_owners_not_new_permissions():
    contract = role_contract()
    before = deepcopy(contract)
    report = _infrastructure_definition_report(contract)
    assert report["definition_complete"] is True
    assert len(report["domains"]) == 10
    assert not any(report["authority"].values())
    assert report["operational_health_verified"] is False
    assert report["new_workers_started"] is False
    for row in report["domains"]:
        assert row["owner_sources"]
        assert row["inherited_freshness_slo"]
        assert row["inherited_resource_budget"]
        assert row["inherited_failure_behavior"]
        assert row["escalation_owner"]
        assert row["operational_outcome"] == "not_assessed_by_definition"
    assert contract == before


@pytest.mark.parametrize(
    "case",
    [
        "missing",
        "authority",
        "mode",
        "owner",
        "domain",
        "duplicate",
        "actions",
        "metrics",
        "completion",
        "cadence",
        "malformed_row",
        "trade_owner",
        "missing_domain",
        "promotion_owner",
    ],
)
def test_infrastructure_definition_errors_fail_closed(case):
    contract = role_contract()
    catalog = contract["infrastructure_responsibility_contract"]
    first = catalog["domains"][0]
    if case == "missing":
        contract.pop("infrastructure_responsibility_contract")
    elif case == "authority":
        catalog["authority"]["starts_workers"] = True
    elif case == "mode":
        catalog["mode"] = "automatic_execution"
    elif case == "owner":
        first["owner_component_id"] = "invented"
    elif case == "domain":
        first["state_domain_id"] = "process_lifecycle"
    elif case == "duplicate":
        first["domain_id"] = catalog["domains"][1]["domain_id"]
    elif case == "actions":
        first["requested_actions"] = ["manage_storage"]
    elif case == "metrics":
        first["success_metrics"] = []
    elif case == "completion":
        first["completion_evidence"] = "exit_zero"
    elif case == "cadence":
        catalog["cadence_policy"] = ""
    elif case == "malformed_row":
        catalog["domains"][0] = []
    elif case == "trade_owner":
        first.update(
            owner_component_id="live_execution_gateway",
            state_domain_id="live_order_submission",
            requested_actions=["live_submit"],
        )
    elif case == "missing_domain":
        catalog["domains"].pop()
    elif case == "promotion_owner":
        first.update(
            owner_component_id="production_candidate_controller",
            state_domain_id="production_candidate_state",
            requested_actions=["write_candidate_state"],
        )
    result = validate_contract(contract, check_sources=False)
    assert result["ok"] is False
    assert any(
        reason.startswith("infrastructure_definition:") for reason in result["blockers"]
    )


def test_infrastructure_catalog_never_authorizes_a_trade_or_state_takeover(tmp_path):
    config = tmp_path / "config/system_role_contracts_v1.json"
    config.parent.mkdir()
    config.write_text(json.dumps(role_contract()))
    for component, action, domain in [
        ("storage_lifecycle_controller", "live_submit", "live_order_submission"),
        ("storage_lifecycle_controller", "write_state", "risk_decision_state"),
        ("observability_reporter", "repair_infrastructure", "storage_lifecycle"),
    ]:
        assert (
            evaluate_component_action(
                tmp_path, component_id=component, action=action, state_domain=domain
            )["ok"]
            is False
        )
    assert (
        evaluate_component_action(
            tmp_path,
            component_id="storage_lifecycle_controller",
            action="manage_storage",
            state_domain="storage_lifecycle",
        )["ok"]
        is True
    )
