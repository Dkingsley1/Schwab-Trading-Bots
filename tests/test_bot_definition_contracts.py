from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

from core.bot_definition_contracts import (
    AREAS,
    FALSE_AUTHORITY,
    FALSE_COMPLETION,
    SourceInventory,
    audit_definitions,
    render_audit_markdown,
    safe_project_file,
    validate_definition_policy,
)

ROOT = Path(__file__).resolve().parents[1]


def policy():
    return json.loads((ROOT / "config/bot_organization_v1.json").read_text())[
        "definition_audit_contract"
    ]


@pytest.fixture
def fleet(tmp_path):
    source = tmp_path / "core/alpha.py"
    source.parent.mkdir()
    source.write_text("def signal(features):\n    return features['momentum'] > 0\n")
    reference = {
        "path": "core/alpha.py",
        "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "symbol": "signal",
    }
    contract = policy()
    values = {
        "purpose": {
            "role": "signal_sub_bot",
            "hypothesis": "Positive momentum predicts next-window excess return after costs",
            "primary_family": "swing_directional",
            "objective": "directional_alpha",
        },
        "scope": {
            "universe": ["SPY"],
            "venue": "approved_broker",
            "session": "XNYS_regular",
            "decision_interval_seconds": 60,
            "holding_horizon_seconds": 3600,
            "inputs": ["momentum", "timestamp_utc"],
        },
        "decision_rules": {
            "output_rule": {
                "formula": "features['momentum'] > 0",
                "output_schema": "boolean signal",
            },
            "parameters": {"values": {"lookback": 20}, "units": "bars"},
            "position_management": {
                key: "Delegated to the version-bound existing position owner"
                for key in ("entry", "add", "trim", "exit", "time_stop")
            },
            "test_cases": [
                {
                    "case_id": "positive",
                    "kind": "normal",
                    "inputs": {"momentum": 1},
                    "expected": 1,
                },
                {
                    "case_id": "abstain",
                    "kind": "abstention",
                    "inputs": {"momentum": 0},
                    "expected": 0,
                },
            ],
        },
        "abstention": {
            "conditions": ["momentum <= 0", "timestamp outside freshness contract"],
            "maximum_input_age_seconds": 30,
            "supported_regimes": ["measured_trend"],
        },
        "boundaries": {
            "owner": "existing_sleeve_controller",
            "allowed_outputs": ["shadow_vote"],
            "forbidden_actions": ["submit_order", "change_risk_limits"],
            "risk_owner": "existing_risk_owner",
            "dependencies": ["point_in_time_features"],
        },
        "training": {
            "target": "post_cost_forward_return",
            "label_horizon_seconds": 3600,
            "join_policy": "exact_symbol_time_candidate",
            "split_policy": "purged_chronological",
            "experiment_contract": {
                key: "required exact immutable binding"
                for key in (
                    "candidate_binding",
                    "trial_accounting",
                    "dataset_version",
                    "untouched_holdout",
                    "embargo",
                )
            },
        },
        "accountability": {
            "metric": "post_cost_excess_return_lcb",
            "benchmark": "same_universe_passive",
            "invalidation": "negative mature candidate-forward LCB",
            "trace_fields": contract["trace_required_fields"],
            "lifecycle": {
                key: "existing lifecycle owner review required"
                for key in ("collect", "evaluate", "invalidate", "retire")
            },
        },
    }
    contract["bot_definitions"] = {
        "alpha": {
            "revision": "v1",
            "areas": {
                area: {
                    field: {"value": value, "references": [reference]}
                    for field, value in fields.items()
                }
                for area, fields in values.items()
            },
        }
    }
    registry = {
        "sub_bots": [
            {
                "bot_id": "alpha",
                "bot_role": "signal_sub_bot",
                "active": True,
                "test_accuracy": 1.0,
                "quality_score": 1.0,
            }
        ]
    }
    catalog = {"bots": [{"bot_id": "alpha", "core_file": "core/alpha.py"}]}
    assignments = [{"bot_id": "alpha", "sub_sleeve_id": "trend_and_momentum"}]
    return registry, catalog, assignments, contract, tmp_path


def test_complete_definition_never_grants_evidence_or_runtime_conformance(fleet):
    before = deepcopy(fleet[:-1])
    result = audit_definitions(*fleet)
    assert result["definition_complete"] is True
    assert result["definition_complete_count"] == 1
    assert result["audit_coverage_ratio"] == 1.0
    assert all(area["complete_count"] == 1 for area in result["area_summary"].values())
    assert result["economic_evidence"]["status"] == "not_assessed_by_definition_audit"
    assert result["economic_evidence"]["definition_grants_evidence"] is False
    assert (
        result["implementation_verification"]["runtime_conformance_verified"] is False
    )
    assert not any(result["authority"].values())
    assert fleet[:-1] == before
    result["records"][0]["areas"]["purpose"]["fields"]["hypothesis"][
        "value"
    ] = "changed"
    assert fleet[:-1] == before


@pytest.mark.parametrize("area", list(AREAS))
def test_every_area_is_required(fleet, area):
    fleet[3]["bot_definitions"]["alpha"]["areas"].pop(area)
    result = audit_definitions(*fleet)
    assert result["definition_complete"] is False
    assert result["records"][0]["areas"][area]["complete"] is False


@pytest.mark.parametrize("key", FALSE_AUTHORITY)
def test_no_new_authority(key):
    contract = policy()
    contract["authority"][key] = True
    assert "definition_audit_authority_invalid" in validate_definition_policy(contract)


@pytest.mark.parametrize("key", FALSE_COMPLETION)
def test_completion_semantics_cannot_be_relaxed(key):
    contract = policy()
    contract["completion_policy"][key] = True
    assert "definition_completion_policy_invalid" in validate_definition_policy(
        contract
    )


@pytest.mark.parametrize("value", [True, 0, -1, "60", float("nan")])
def test_timing_requires_positive_finite_numbers(fleet, value):
    fleet[3]["bot_definitions"]["alpha"]["areas"]["scope"]["decision_interval_seconds"][
        "value"
    ] = value
    result = audit_definitions(*fleet)
    assert result["definition_complete"] is False


@pytest.mark.parametrize(
    "field,value",
    [
        ("primary_family", ["trend", "reversal"]),
        ("hypothesis", "tbd"),
        ("hypothesis", "sleeve_specific"),
    ],
)
def test_placeholders_and_multiple_primary_families_fail(fleet, field, value):
    fleet[3]["bot_definitions"]["alpha"]["areas"]["purpose"][field]["value"] = value
    assert audit_definitions(*fleet)["definition_complete"] is False


def test_inferred_family_and_labels_remain_incomplete(fleet):
    fleet[3]["bot_definitions"] = {}
    fleet[0]["sub_bots"][0].update(
        label_contract={"source": "inferred_from_registry_identity"},
        training_label_materialization_contract={
            "required_outputs": ["return_bucket"],
            "minimum_label_maturity_seconds": 86400,
            "required_join_mode": "point_in_time",
            "evaluation_split_policy": "purged_chronological",
        },
    )
    result = audit_definitions(*fleet)
    assert result["audit_coverage_ratio"] == 1.0
    assert result["definition_complete_count"] == 0
    assert (
        result["records"][0]["areas"]["purpose"]["fields"]["primary_family"]["inferred"]
        is True
    )
    assert result["records"][0]["areas"]["training"]["defined_fields"] == 0


def test_task_description_is_not_credited_as_economic_hypothesis(fleet):
    fleet[3]["bot_definitions"] = {}
    fleet[0]["sub_bots"][0]["slot_objective"] = "Score candidate observations"
    record = audit_definitions(*fleet)["records"][0]
    assert record["declared_task_description"] == "Score candidate observations"
    assert "hypothesis" not in record["areas"]["purpose"]["fields"]
    assert "objective" not in record["areas"]["purpose"]["fields"]


def test_source_changes_invalidate_pinned_definition(fleet):
    before = audit_definitions(*fleet)
    (fleet[4] / "core/alpha.py").write_text("def signal(features):\n    return False\n")
    after = audit_definitions(*fleet)
    assert after["definition_complete"] is False
    assert (
        before["records"][0]["definition_sha256"]
        != after["records"][0]["definition_sha256"]
    )
    assert any("source_hash_mismatch" in key for key in after["gap_counts"])
    assert before["audit_sha256"] != after["audit_sha256"]


def test_collection_wrapper_cannot_be_declared_implemented_strategy(fleet):
    path = fleet[4] / "core/alpha.py"
    path.write_text(
        "def signal(features):\n    return describe_registry_backed_bot(features)\n"
    )
    new_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    for fields in fleet[3]["bot_definitions"]["alpha"]["areas"].values():
        for record in fields.values():
            record["references"][0]["sha256"] = new_hash
    result = audit_definitions(*fleet)
    assert result["definition_complete"] is False
    assert (
        "collection_wrapper_not_implemented_strategy" in result["records"][0]["issues"]
    )


def test_no_market_not_applicable_escape(fleet):
    fleet[3]["bot_definitions"]["alpha"]["areas"]["training"]["target"]["value"] = {
        "not_applicable": True,
        "reason": "I am calling myself infrastructure",
    }
    assert audit_definitions(*fleet)["definition_complete"] is False


def test_operational_objective_is_not_fabricated_market_return(fleet):
    fleet[0]["sub_bots"][0]["bot_role"] = "infrastructure_sub_bot"
    fleet[3]["bot_definitions"]["alpha"]["areas"]["purpose"]["role"][
        "value"
    ] = "infrastructure_sub_bot"
    for field in AREAS["training"]:
        fleet[3]["bot_definitions"]["alpha"]["areas"]["training"][field]["value"] = {
            "not_applicable": True,
            "reason": "Deterministic service with no trainable model",
        }
    result = audit_definitions(*fleet)
    assert result["records"][0]["areas"]["training"]["complete"] is True
    assert (
        result["records"][0]["economic_evidence"]["status"]
        == "not_applicable_operational_role"
    )


def test_empty_and_malformed_registry_not_vacuously_complete(fleet):
    fleet[0]["sub_bots"] = []
    assert audit_definitions(*fleet)["definition_complete"] is False
    fleet[0]["sub_bots"] = [None, {}, {"bot_id": "alpha"}, {"bot_id": "alpha"}]
    result = audit_definitions(*fleet)
    assert result["audited_record_count"] == 4
    assert result["definition_complete_count"] == 0
    assert result["issue_counts"]["duplicate_registry_identity"] == 2


def test_inactive_missing_source_is_not_skipped(fleet):
    fleet[0]["sub_bots"].append({"bot_id": "cold", "active": False})
    result = audit_definitions(*fleet)
    assert result["audited_record_count"] == 2
    assert result["definition_complete_count"] == 1
    assert result["definition_incomplete_count"] == 1


def test_missing_definitions_not_labelled_duplicate(fleet):
    fleet[3]["bot_definitions"] = {}
    assert audit_definitions(*fleet)["identical_complete_mandates"] == []


def test_definition_identity_ignores_accuracy_and_profitability_labels(fleet):
    before = audit_definitions(*fleet)["records"][0]["definition_sha256"]
    fleet[0]["sub_bots"][0].update(
        test_accuracy=0.0, quality_score=0.0, profitability_status="approved"
    )
    after = audit_definitions(*fleet)["records"][0]["definition_sha256"]
    assert before == after


def test_fixed_parameters_and_boolean_parameters_are_explicit(fleet):
    record = fleet[3]["bot_definitions"]["alpha"]["areas"]["decision_rules"][
        "parameters"
    ]
    record["value"] = {
        "values": {},
        "units": "dimensionless",
        "no_tunable_parameters": True,
    }
    assert audit_definitions(*fleet)["definition_complete"] is True
    record["value"] = {"values": {"use_filter": False}, "units": "boolean"}
    assert audit_definitions(*fleet)["definition_complete"] is True


def test_market_role_cannot_be_reassigned_by_definition(fleet):
    fleet[3]["bot_definitions"]["alpha"]["areas"]["purpose"]["role"][
        "value"
    ] = "infrastructure_sub_bot"
    result = audit_definitions(*fleet)
    assert result["definition_complete"] is False
    assert "definition_registry_role_conflict" in result["records"][0]["issues"]


@pytest.mark.parametrize("role", [None, [], {}, "made_up_role"])
def test_invalid_registry_role_still_receives_an_audit_record(fleet, role):
    fleet[0]["sub_bots"][0]["bot_role"] = role
    result = audit_definitions(*fleet)
    assert result["audit_coverage_ratio"] == 1.0
    assert result["definition_complete"] is False
    assert "registry_role_unknown_or_invalid" in result["records"][0]["issues"]


@pytest.mark.parametrize(
    "strict,complete,expected", [(False, False, 0), (True, False, 2), (True, True, 0)]
)
def test_native_cli_strict_exit_is_separate_from_runtime_gates(
    tmp_path, monkeypatch, strict, complete, expected
):
    import sys
    from scripts.ops import bot_organization_control as control

    audit = {
        "definition_complete": complete,
        "definition_complete_count": int(complete),
        "definition_incomplete_count": int(not complete),
        "registry_record_count": 1,
        "audited_record_count": 1,
    }
    monkeypatch.setattr(
        control,
        "build_payload",
        lambda *args, **kwargs: (
            {"ok": True, "definition_audit": audit},
            {"definition_audit": audit},
        ),
    )
    argv = ["bot-organization", "--project-root", str(tmp_path), "--json"]
    if strict:
        argv.append("--require-definition-complete")
    monkeypatch.setattr(sys, "argv", argv)
    assert control.main() == expected
    assert (
        tmp_path / "exports/reports/operator/bot_definition_audit_latest.md"
    ).is_file()
    assert not (tmp_path / "master_bot_registry.json").exists()


def test_declared_abstention_case_can_have_empty_input_and_false_output(fleet):
    cases = fleet[3]["bot_definitions"]["alpha"]["areas"]["decision_rules"][
        "test_cases"
    ]["value"]
    cases[1].update(inputs={}, expected=False)
    assert audit_definitions(*fleet)["definition_complete"] is True


def test_native_control_separates_org_grade_and_definition_status(tmp_path):
    from scripts.ops.bot_organization_control import build_payload

    config = tmp_path / "config"
    core = tmp_path / "core"
    config.mkdir()
    core.mkdir()
    (config / "bot_organization_v1.json").write_bytes(
        (ROOT / "config/bot_organization_v1.json").read_bytes()
    )
    (tmp_path / "master_bot_registry.json").write_text(
        json.dumps({"sub_bots": [{"bot_id": "alpha", "bot_role": "signal_sub_bot"}]})
    )
    (core / "bot_catalog.json").write_text(json.dumps({"bots": [{"bot_id": "alpha"}]}))
    health, hierarchy = build_payload(tmp_path)
    assert health["definition_audit"]["audit_coverage_ratio"] == 1.0
    assert health["definition_audit"]["definition_complete"] is False
    assert "records" not in health["definition_audit"]
    assert len(hierarchy["definition_audit"]["records"]) == 1
    assert "not_definition_completeness_or_economic_evidence" in health["grade_scope"]
    assert not (tmp_path / "governance").exists()


def test_reference_symbols_and_json_pointers_are_verified(fleet):
    inventory = SourceInventory(fleet[4], policy()["source_budget"])
    ref = {
        "path": "core/alpha.py",
        "sha256": hashlib.sha256((fleet[4] / "core/alpha.py").read_bytes()).hexdigest(),
        "symbol": "missing",
    }
    assert inventory.reference_error(ref) == "source_symbol_missing"
    path = fleet[4] / "rules.json"
    path.write_text('{"a/b": {"x~y": [4]}}')
    ref = {
        "path": "rules.json",
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "symbol": "/a~1b/x~0y/0",
    }
    assert inventory.reference_error(ref) == ""
    ref["symbol"] = "/absent"
    assert inventory.reference_error(ref) == "source_symbol_missing"


def test_inventory_never_imports_source(fleet):
    path = fleet[4] / "core/explosive.py"
    marker = fleet[4] / "executed"
    path.write_text(
        f"open({str(marker)!r}, 'w').write('bad')\nraise RuntimeError('never execute')\nBOT_SPEC = {{'bot_id': 'safe'}}\n"
    )
    result = SourceInventory(fleet[4], policy()["source_budget"]).read(
        "core/explosive.py"
    )
    assert result["status"] == "read"
    assert result["literals"]["BOT_SPEC"]["bot_id"] == "safe"
    assert not marker.exists()


@pytest.mark.parametrize(
    "raw", ["/Volumes/VIDEO/forbidden.py", "../outside.py", "/etc/passwd", ""]
)
def test_external_paths_are_rejected_before_any_probe(fleet, monkeypatch, raw):
    def forbidden(*args, **kwargs):
        raise AssertionError("external path probed")

    monkeypatch.setattr(Path, "is_symlink", forbidden)
    assert safe_project_file(fleet[4], raw) is None


def test_symlink_target_is_not_followed(fleet):
    (fleet[4] / "core/escape.py").symlink_to("/Volumes/VIDEO/forbidden.py")
    assert safe_project_file(fleet[4], "core/escape.py") is None


def test_source_budget_is_bounded_and_visible(fleet):
    fleet[3]["source_budget"]["maximum_total_bytes"] = 1
    result = audit_definitions(*fleet)
    assert result["definition_complete"] is False
    assert result["source_inventory"]["bytes_read"] == 0
    assert result["issue_counts"]["source_size_limit"] == 1


def test_markdown_separates_all_four_statuses(fleet):
    text = render_audit_markdown(audit_definitions(*fleet))
    assert "Audited 1/1" in text
    assert "Definition complete: 1" in text
    assert "do not verify runtime conformance or economic edge" in text
    assert "Economic evidence: not assessed" in text
