from copy import deepcopy
import ast
import json
from pathlib import Path

import pytest

from core.bot_definition_contracts import AREAS, FALSE_AUTHORITY, render_audit_markdown
from core.bot_operating_definitions import (
    POLICY_PATH,
    compile_catalog,
    expand_definition,
    inspect_program,
    registry_projection,
    validate_catalog,
    validate_policy,
)

ROOT = Path(__file__).resolve().parents[1]
PROCESS_POLICY_PATH = "config/bot_process_definition_contract_v1.json"


@pytest.fixture
def fleet(tmp_path):
    policy = json.loads((ROOT / POLICY_PATH).read_text())
    process_path = tmp_path / PROCESS_POLICY_PATH
    process_path.parent.mkdir(parents=True, exist_ok=True)
    process_path.write_bytes((ROOT / PROCESS_POLICY_PATH).read_bytes())
    for source in policy["shared_sources"]:
        path = tmp_path / source
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("def shared_owner():\n    return None\n")
    programs = {
        "wrapper": "BOT_SPEC = {'bot_id': 'wrapper', 'bot_role': 'signal_sub_bot', 'slot_objective': 'Collect observations'}\ndef describe_bot():\n    return describe_registry_backed_bot(BOT_SPEC)\ndef train_brain():\n    return train_registry_backed_bot(BOT_SPEC)\n",
        "runtime": "from indicator_bot_common import train_runtime_indicator_bot\nSYMBOLS = ['SPY']\ndef features(rows, index):\n    return rows[index]['close']\ndef train_brain():\n    return train_runtime_indicator_bot(horizon=5, window=14, symbol_allowlist=SYMBOLS)\n",
        "synthetic": "from indicator_bot_common import train_indicator_bot\ndef train_brain():\n    return train_indicator_bot(window=24, horizon=3)\n",
    }
    for name, code in programs.items():
        (tmp_path / f"core/{name}.py").write_text(code)
    registry = {
        "sub_bots": [
            {
                "bot_id": name,
                "bot_role": "signal_sub_bot",
                "active": True,
                "execution_enabled": False,
                "label_contract": {
                    "label_family": "generic_directional",
                    "source": "inferred_from_registry_identity",
                },
            }
            for name in (*programs, "slot")
        ]
    }
    catalog = {
        "bots": [
            {
                "bot_id": row["bot_id"],
                "core_file": (
                    f"core/{row['bot_id']}.py" if row["bot_id"] in programs else ""
                ),
                "runner": "",
            }
            for row in registry["sub_bots"]
        ]
    }
    legacy = {
        "definition_complete_count": 0,
        "definition_complete": False,
        "records": [
            {
                "registry_index": i,
                "bot_id": row["bot_id"],
                "definition_complete": False,
                "issues": ["unresolved_trading_mandate"],
            }
            for i, row in enumerate(registry["sub_bots"])
        ],
        "economic_evidence": {
            "status": "not_assessed_by_definition_audit",
            "definition_grants_evidence": False,
        },
    }
    manifest = compile_catalog(registry, catalog, policy, tmp_path)
    return registry, catalog, legacy, policy, manifest, tmp_path


def test_complete_operating_definitions_preserve_trading_and_economic_gaps(fleet):
    before = deepcopy(fleet[:-1])
    result = validate_catalog(*fleet)
    assert result["definition_complete"] is True
    assert result["definition_complete_count"] == 4
    assert result["process_summary"]["defined_bot_count"] == 4
    assert result["process_summary"]["runtime_verified_bot_count"] == 0
    assert all(row["complete_count"] == 4 for row in result["area_summary"].values())
    assert (
        result["standalone_trading_mandate_summary"]["definition_complete_count"] == 0
    )
    assert result["economic_evidence"]["definition_grants_evidence"] is False
    assert not any(result["authority"].values())
    assert {row["profile_id"] for row in result["records"]} == {
        "collection_wrapper",
        "registry_declared_slot",
        "runtime_model_program",
        "synthetic_research_program",
    }
    for record in result["records"]:
        assert record["standalone_trading_mandate"]["issues"] == [
            "unresolved_trading_mandate"
        ]
        assert (
            record["implementation_verification"]["runtime_conformance_verified"]
            is False
        )
        expanded = expand_definition(
            {"binding": record["binding"], "binding_sha256": record["binding_sha256"]},
            fleet[3],
        )
        assert set(expanded) == set(AREAS)
        assert all(set(expanded[area]) == set(fields) for area, fields in AREAS.items())
    assert fleet[:-1] == before


def test_process_policy_drift_needs_explicit_review_not_silent_repin(fleet):
    path = fleet[5] / PROCESS_POLICY_PATH
    policy = json.loads(path.read_text())
    policy["stages"]["registry_binding"]["title"] = "Reviewed registry identity"
    path.write_text(json.dumps(policy))
    before = deepcopy(fleet[4])
    result = validate_catalog(*fleet)
    assert result["definition_complete_count"] == 0
    assert "operating_process_policy_changed" in result["errors"]
    assert fleet[4] == before
    fleet[4].update(compile_catalog(fleet[0], fleet[1], fleet[3], fleet[5]))
    assert validate_catalog(*fleet)["definition_complete_count"] == 4


def test_process_policy_symlink_is_rejected_without_reading_target(fleet):
    path = fleet[5] / PROCESS_POLICY_PATH
    path.unlink()
    path.symlink_to("/Volumes/VIDEO/no_probe.json")
    result = validate_catalog(*fleet)
    assert result["definition_complete_count"] == 0
    assert "process_policy_missing_or_disallowed" in result["errors"]


def test_exact_model_arguments_not_guessed_holding_seconds(fleet):
    result = validate_catalog(*fleet)
    record = next(row for row in result["records"] if row["bot_id"] == "runtime")
    program = record["binding"]["source"]["program"]
    call = next(
        call
        for call in program["training_calls"]
        if call["call"] == "train_runtime_indicator_bot"
    )
    assert call["keywords"] == {
        "horizon": "5",
        "window": "14",
        "symbol_allowlist": "SYMBOLS",
    }
    assert program["configuration_expressions"]["SYMBOLS"] == "['SPY']"
    expanded = expand_definition(
        {"binding": record["binding"], "binding_sha256": record["binding_sha256"]},
        fleet[3],
    )
    assert (
        expanded["scope"]["decision_interval_seconds"]["value"]["mode"]
        == "event_driven"
    )
    assert (
        expanded["training"]["label_horizon_seconds"]["value"][
            "seconds_inferred_from_samples"
        ]
        is False
    )


@pytest.mark.parametrize("area", list(AREAS))
def test_every_area_remains_required(fleet, area):
    fleet[3]["areas"].pop(area)
    assert not validate_catalog(*fleet)["definition_complete"]
    with pytest.raises(ValueError):
        compile_catalog(fleet[0], fleet[1], fleet[3], fleet[5])


@pytest.mark.parametrize("flag", FALSE_AUTHORITY)
def test_policy_cannot_grant_any_authority(fleet, flag):
    fleet[3]["authority"][flag] = True
    assert "operating_authority_invalid" in validate_policy(fleet[3])
    assert not validate_catalog(*fleet)["definition_complete"]


def test_source_change_invalidates_only_affected_bot_and_never_repins(fleet):
    before = deepcopy(fleet[4])
    path = fleet[5] / "core/runtime.py"
    path.write_text(path.read_text().replace("horizon=5", "horizon=6"))
    result = validate_catalog(*fleet)
    assert result["definition_complete_count"] == 3
    affected = next(row for row in result["records"] if row["bot_id"] == "runtime")
    assert "operating_binding_changed" in affected["issues"]
    assert fleet[4] == before
    assert not (fleet[5] / "config/bot_operating_definitions_v1.json").exists()


def test_shared_owner_change_is_not_silently_accepted(fleet):
    (fleet[5] / "core/runtime_training_common.py").write_text(
        "def changed_owner():\n    pass\n"
    )
    result = validate_catalog(*fleet)
    assert result["definition_complete_count"] == 0
    assert "operating_shared_sources_changed" in result["errors"]


def test_collection_and_registry_slots_cannot_become_implemented_strategies(fleet):
    result = validate_catalog(*fleet)
    for row in result["records"]:
        if row["bot_id"] in {"slot", "wrapper"}:
            assert row["definition_complete"]
            assert not row["standalone_trading_mandate"]["definition_complete"]
    fleet[4]["entries"]["runtime"]["profile_id"] = "registry_declared_slot"
    result = validate_catalog(*fleet)
    assert result["definition_complete_count"] == 3


def test_missing_model_source_does_not_fall_back_to_registry_slot(fleet):
    (fleet[5] / "core/runtime.py").unlink()
    result = validate_catalog(*fleet)
    assert result["definition_complete_count"] == 3
    fleet[1]["bots"][1]["core_file"] = ""
    result = validate_catalog(*fleet)
    assert result["definition_complete_count"] == 3
    assert not next(row for row in result["records"] if row["bot_id"] == "runtime")[
        "definition_complete"
    ]


def test_dedicated_implementation_added_to_slot_requires_review(fleet):
    (fleet[5] / "core/slot.py").write_text("def train_brain():\n    return 0\n")
    assert validate_catalog(*fleet)["definition_complete_count"] == 3


@pytest.mark.parametrize(
    "field", ["execution_enabled", "direct_execution_allowed", "live_trading_enabled"]
)
def test_execution_authority_never_earns_operating_clearance(fleet, field):
    fleet[0]["sub_bots"][0][field] = True
    result = validate_catalog(*fleet)
    assert not result["records"][0]["definition_complete"]
    with pytest.raises(ValueError):
        compile_catalog(fleet[0], fleet[1], fleet[3], fleet[5])


def test_runtime_metrics_do_not_redefine_the_bot(fleet):
    before = validate_catalog(*fleet)
    for row in fleet[0]["sub_bots"]:
        row.update(
            test_accuracy=0.001,
            quality_score=99,
            profitability_status="ready",
            active=False,
        )
    after = validate_catalog(*fleet)
    assert after["definition_complete_count"] == 4
    assert [row["definition_sha256"] for row in before["records"]] == [
        row["definition_sha256"] for row in after["records"]
    ]


def test_declared_training_target_change_requires_new_binding(fleet):
    fleet[0]["sub_bots"][0]["label_contract"]["label_family"] = "changed_objective"
    assert validate_catalog(*fleet)["definition_complete_count"] == 3


def test_new_removed_and_duplicate_registry_identities_fail(fleet):
    fleet[0]["sub_bots"].append({"bot_id": "new", "bot_role": "signal_sub_bot"})
    assert not validate_catalog(*fleet)["definition_complete"]
    fleet[0]["sub_bots"].pop()
    fleet[0]["sub_bots"].pop()
    assert "operating_orphan_definitions" in validate_catalog(*fleet)["errors"]
    fleet[0]["sub_bots"].append(deepcopy(fleet[0]["sub_bots"][0]))
    assert not validate_catalog(*fleet)["definition_complete"]


def test_empty_registry_not_complete(fleet):
    fleet[0]["sub_bots"] = []
    assert not validate_catalog(*fleet)["definition_complete"]


def test_no_source_import_or_training_on_authoring(fleet):
    path = fleet[5] / "core/runtime.py"
    marker = fleet[5] / "trained"
    path.write_text(
        path.read_text()
        + f"\nopen({str(marker)!r}, 'w').write('bad')\nraise RuntimeError('must not import')\n"
    )
    manifest = compile_catalog(fleet[0], fleet[1], fleet[3], fleet[5])
    assert manifest["entries"]["runtime"]
    assert not marker.exists()


def test_order_call_in_program_requires_trading_review(fleet):
    path = fleet[5] / "core/runtime.py"
    path.write_text(path.read_text() + "\ndef dangerous():\n    broker.place_order()\n")
    with pytest.raises(ValueError, match="operating_program_has_order_calls"):
        compile_catalog(fleet[0], fleet[1], fleet[3], fleet[5])


def test_external_and_symlink_sources_never_followed(fleet):
    fleet[1]["bots"][3]["core_file"] = "/Volumes/VIDEO/forbidden.py"
    with pytest.raises(ValueError, match="source_unreadable"):
        compile_catalog(fleet[0], fleet[1], fleet[3], fleet[5])
    fleet[1]["bots"][3]["core_file"] = "core/escape.py"
    (fleet[5] / "core/escape.py").symlink_to("/Volumes/VIDEO/forbidden.py")
    assert not validate_catalog(*fleet)["definition_complete"]


def test_budget_exhaustion_stays_incomplete(fleet):
    fleet[3]["source_budget"]["maximum_total_bytes"] = 1
    assert not validate_catalog(*fleet)["definition_complete"]
    with pytest.raises(ValueError):
        compile_catalog(fleet[0], fleet[1], fleet[3], fleet[5])


def test_manifest_contains_receipts_not_duplicated_registry_or_source(fleet):
    assert all(
        "binding" not in entry and "binding_sha256" in entry
        for entry in fleet[4]["entries"].values()
    )


def test_compact_snapshot_preserves_grade_normalization(tmp_path):
    from scripts.ops.long_runtime_common import write_payload

    pretty, compact = tmp_path / "pretty.json", tmp_path / "compact.json"
    payload = {"grade": "A++", "records": [{"a": 1, "b": [2, 3]}]}
    write_payload(pretty, payload)
    write_payload(compact, payload, compact=True)
    assert json.loads(pretty.read_text()) == json.loads(compact.read_text())
    assert compact.stat().st_size < pretty.stat().st_size


def test_required_owner_sources_cannot_be_removed(fleet):
    fleet[3]["shared_sources"].pop()
    assert "operating_required_shared_sources_missing_or_duplicate" in validate_policy(
        fleet[3]
    )
    assert not validate_catalog(*fleet)["definition_complete"]


def test_program_inspection_resolves_aliases_and_qualified_defaults():
    program = inspect_program(
        ast.parse(
            "from helpers import train_runtime_indicator_bot as train\nWINDOW: int\nclass Spec:\n    horizon: int = 4\n    def f(self, window=24):\n        pass\ndef f(window=14):\n    return train(horizon=5)\n"
        )
    )
    assert program["runtime_training_path_present"]
    assert program["training_calls"][0]["call"] == "train_runtime_indicator_bot"
    assert program["functions"]["Spec.f"]["defaults"] == {"window": "24"}
    assert program["functions"]["f"]["defaults"] == {"window": "14"}
    assert program["class_field_defaults"] == {"Spec": {"horizon": "4"}}


def test_unknown_program_requires_review_not_synthetic_default(fleet):
    (fleet[5] / "core/synthetic.py").write_text("def train_brain():\n    return None\n")
    with pytest.raises(ValueError, match="operating_program_kind_requires_review"):
        compile_catalog(fleet[0], fleet[1], fleet[3], fleet[5])


@pytest.mark.parametrize("value", [0, 1, "false", [], {}])
def test_malformed_authority_is_not_treated_as_disabled(fleet, value):
    fleet[0]["sub_bots"][0]["execution_enabled"] = value
    assert not validate_catalog(*fleet)["records"][0]["definition_complete"]


def test_shared_defaults_published_once(fleet):
    (fleet[5] / "core/crypto_runtime_bot_common.py").write_text(
        "class CryptoRuntimeSpec:\n    horizon: int = 4\n"
    )
    fleet[4].update(compile_catalog(fleet[0], fleet[1], fleet[3], fleet[5]))
    result = validate_catalog(*fleet)
    assert result["definition_complete"]
    assert (
        result["shared_source_contracts"]["core/crypto_runtime_bot_common.py"][
            "program"
        ]["class_field_defaults"]["CryptoRuntimeSpec"]["horizon"]
        == "4"
    )


def test_definition_resolver_change_invalidates_pinned_definitions(fleet):
    (fleet[5] / "core/bot_operating_definitions.py").write_text(
        "def new_interpretation():\n    pass\n"
    )
    assert "operating_shared_sources_changed" in validate_catalog(*fleet)["errors"]


@pytest.mark.parametrize(
    "flag, expected",
    [
        (None, 0),
        ("--require-definition-complete", 0),
        ("--require-trading-mandate-complete", 2),
    ],
)
def test_native_cli_keeps_strict_scopes_separate_without_auto_authoring(
    fleet, monkeypatch, flag, expected
):
    import sys
    from scripts.ops import bot_organization_control as control

    audit = validate_catalog(*fleet)
    monkeypatch.setattr(
        control,
        "build_payload",
        lambda *args, **kwargs: (
            {"ok": True, "definition_audit": audit},
            {"definition_audit": audit},
        ),
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("routine audit cannot author new definitions")

    monkeypatch.setattr(control, "compile_operating_catalog", forbidden)
    argv = ["bot-organization", "--project-root", str(fleet[5]), "--json"]
    if flag:
        argv.append(flag)
    monkeypatch.setattr(sys, "argv", argv)
    assert control.main() == expected
    assert not (fleet[5] / "config/bot_operating_definitions_v1.json").exists()


def test_ownership_repair_never_materializes_definitions():
    ownership = json.loads(
        (ROOT / "config/control_surface_ownership_v1.json").read_text()
    )
    owned = next(
        row
        for row in ownership["controls"]
        if row["control_id"] == "bot_operating_definition_catalog"
    )
    assert "--materialize-operating-definitions" not in owned["owner_command"]


def test_all_32_subsections_are_numbered_and_reported(fleet):
    audit = validate_catalog(*fleet)
    record = audit["records"][0]
    expanded = expand_definition(
        {"binding": record["binding"], "binding_sha256": record["binding_sha256"]},
        fleet[3],
    )
    report = render_audit_markdown(audit)
    numbers = []
    for area_index, (area, fields) in enumerate(AREAS.items(), 1):
        assert set(audit["area_summary"][area]["subsections"]) == set(fields)
        assert f"### {area_index}. {area.replace('_', ' ').title()}" in report
        for field_index, field in enumerate(fields, 1):
            expected = f"{area_index}.{field_index}"
            subsection = expanded[area][field]["subsection"]
            numbers.append(subsection["number"])
            assert subsection["number"] == expected
            assert subsection["title"] == fleet[3]["subsections"][area][field]
            assert f"{expected} {subsection['title']}" in report
            assert record["areas"][area]["subsections"][field]["complete"]
            assert (
                audit["area_summary"][area]["subsections"][field]["complete_count"] == 4
            )
    assert len(set(numbers)) == 32
    assert "Standalone trading mandates complete: 0/4" in report


@pytest.mark.parametrize("area", list(AREAS))
def test_missing_subsection_cannot_waive_a_required_field(fleet, area):
    fleet[3]["subsections"][area].pop(AREAS[area][0])
    audit = validate_catalog(*fleet)
    assert not audit["definition_complete"]
    assert f"operating_subsections_invalid:{area}" in audit["errors"]
    with pytest.raises(ValueError, match="operating_subsections_invalid"):
        compile_catalog(fleet[0], fleet[1], fleet[3], fleet[5])


@pytest.mark.parametrize(
    "value", [None, [], {}, "", "   ", "invalid|table", "invalid\nheading", "x" * 101]
)
def test_subsection_titles_reject_malformed_values(fleet, value):
    fleet[3]["subsections"]["purpose"]["role"] = value
    assert "operating_subsections_invalid:purpose" in validate_policy(fleet[3])
    assert not validate_catalog(*fleet)["definition_complete"]


@pytest.mark.parametrize("value", [None, [], {}, {"unknown": {}}])
def test_subsection_area_contract_must_cover_all_seven_areas(fleet, value):
    fleet[3]["subsections"] = value
    assert "operating_subsection_areas_required" in validate_policy(fleet[3])
    assert not validate_catalog(*fleet)["definition_complete"]


def test_subsection_changes_require_review_and_cannot_change_evidence(fleet):
    baseline = validate_catalog(*fleet)
    manifest = deepcopy(fleet[4])
    fleet[3]["subsections"]["purpose"]["role"] = "Registered Role"
    changed = validate_catalog(*fleet)
    assert not changed["definition_complete"]
    assert "operating_catalog_policy_mismatch" in changed["errors"]
    assert all(
        not leaf["complete"]
        for record in changed["records"]
        for area in record["areas"].values()
        for leaf in area["subsections"].values()
    )
    assert fleet[4] == manifest
    fleet[4].update(compile_catalog(fleet[0], fleet[1], fleet[3], fleet[5]))
    rebound = validate_catalog(*fleet)
    assert rebound["definition_complete"]
    assert (
        baseline["records"][0]["definition_sha256"]
        != rebound["records"][0]["definition_sha256"]
    )
    assert (
        baseline["standalone_trading_mandate_summary"]
        == rebound["standalone_trading_mandate_summary"]
    )
    assert baseline["economic_evidence"] == rebound["economic_evidence"]
