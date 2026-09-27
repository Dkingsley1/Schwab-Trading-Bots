import ast
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

from core.bot_definition_contracts import FALSE_AUTHORITY, digest
from core.bot_operating_definitions import (
    POLICY_PATH as OPERATING_POLICY_PATH,
    compile_catalog,
    inspect_program,
    validate_catalog,
)
from core.bot_process_definitions import POLICY_PATH, expand_definition, validate_policy

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def policy():
    return json.loads((ROOT / POLICY_PATH).read_text())


@pytest.fixture
def bindings(tmp_path):
    """Use the same four-profile observation boundary as the operating tests."""
    operating_policy = json.loads((ROOT / OPERATING_POLICY_PATH).read_text())
    process_path = tmp_path / POLICY_PATH
    process_path.parent.mkdir(parents=True, exist_ok=True)
    process_path.write_text((ROOT / POLICY_PATH).read_text())
    for source in operating_policy["shared_sources"]:
        path = tmp_path / source
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("def shared_owner():\n    return None\n")
    programs = {
        "wrapper": (
            "BOT_SPEC = {'bot_id': 'wrapper', 'bot_role': 'signal_sub_bot', "
            "'slot_objective': 'Collect observations', 'execution_enabled': False}\n"
            "def describe_bot():\n    return describe_registry_backed_bot(BOT_SPEC)\n"
            "def train_brain():\n    return train_registry_backed_bot(BOT_SPEC)\n"
        ),
        "runtime": (
            "from indicator_bot_common import train_runtime_indicator_bot as train\n"
            "from runtime_training_common import direction_label_builder\n"
            "SYMBOLS = ['SPY']\n"
            "def vector(rows, index):\n    return rows[index]['close']\n"
            "def _runtime_sample_filter(rows, index, horizon):\n    return True\n"
            "def train_brain(window=14):\n"
            "    return train(runtime_feature_builder=vector, "
            "runtime_label_builder=direction_label_builder(min_return=0.0007), "
            "sample_filter=_runtime_sample_filter, horizon=5, window=window, "
            "symbol_allowlist=SYMBOLS, allow_fallback_on_insufficient_data=False)\n"
        ),
        "synthetic": (
            "from indicator_bot_common import train_indicator_bot\n"
            "def build_features(rows):\n    return rows['close']\n"
            "def train_brain():\n    return train_indicator_bot("
            "feature_builder=build_features, window=24, horizon=3)\n"
        ),
    }
    for name, code in programs.items():
        (tmp_path / f"core/{name}.py").write_text(code)
    registry = {
        "sub_bots": [
            {
                "bot_id": name,
                "bot_role": "signal_sub_bot",
                "execution_enabled": False,
                "target_functions": ["declared_future_rule"],
                "data_intake_collections": ["declared_observations"],
                "storage_targets": ["declared_output"],
                "label_contract": {"label_family": "generic_directional"},
            }
            for name in (*programs, "slot")
        ]
    }
    catalog = {
        "bots": [
            {
                "bot_id": name,
                "core_file": f"core/{name}.py" if name in programs else "",
                "runner": "",
            }
            for name in (*programs, "slot")
        ]
    }
    manifest = compile_catalog(registry, catalog, operating_policy, tmp_path)
    audit = validate_catalog(
        registry, catalog, {}, operating_policy, manifest, tmp_path
    )
    assert audit["definition_complete"]
    return {row["bot_id"]: row["binding"] for row in audit["records"]}


def stage(result, stage_id):
    return next(row for row in result["stages"] if row["id"] == stage_id)


def test_four_profiles_preserve_observed_binding_and_separate_evidence(
    bindings, policy
):
    before = deepcopy((bindings, policy))
    assert validate_policy(policy) == []
    for binding in bindings.values():
        result = expand_definition(binding, policy)
        assert result["profile_id"] == binding["profile_id"]
        assert result["bot_role"] == binding["registry_definition"]["bot_role"]
        assert result["observed_binding"] == binding
        assert result["observed_binding"] is not binding
        assert result["binding_sha256"] == digest(binding)
        assert result["definition_valid"] is True
        assert result["runtime_verified"] is False
        assert result["runnable_strategy_verified"] is False
        assert not any(result["authority"].values())
        assert result["economic_evidence"] == {
            "status": "not_assessed",
            "definition_grants_evidence": False,
        }
        assert [row["order"] for row in result["stages"]] == list(
            range(1, len(result["stages"]) + 1)
        )
        for row in result["stages"]:
            assert binding["profile_id"] in row["applicable_profiles"]
            assert row["runtime_verified"] is False
            assert row["completion_evidence"]["runtime_completion_verified"] is False
            assert row["completion_evidence"]["runtime_receipts"] == []
    assert (bindings, policy) == before


def test_callbacks_bind_exact_local_function_and_external_expression(bindings, policy):
    result = expand_definition(bindings["runtime"], policy)
    feature = stage(result, "feature_definition")
    reference = next(
        ref
        for ref in feature["source_references"]
        if ref["kind"] == "callback_expression"
    )
    assert reference["expression"] == "vector"
    assert reference["local_function"]["symbol"] == "vector"
    assert reference["local_function"]["signature"] == "rows, index"
    assert reference["name_resolution_verified"] is False
    assert reference["path"] == "core/runtime.py"
    assert reference["sha256"] == bindings["runtime"]["source"]["sha256"]
    label = stage(result, "label_definition")
    reference = label["source_references"][0]
    assert reference["expression"] == "direction_label_builder(min_return=0.0007)"
    assert reference["local_function"] is None
    assert reference["verification_scope"] == "callback_expression_presence_only"
    assert label["status"] == "implemented_by_source_reference"
    filters = stage(result, "filter_definition")
    assert any(
        ref.get("symbol") == "_runtime_sample_filter"
        for ref in filters["source_references"]
    )
    call = next(
        ref
        for ref in stage(result, "program_entrypoints")["source_references"]
        if ref["kind"] == "trainer_call_expression"
    )
    assert call["call"] == "train_runtime_indicator_bot"
    assert call["keywords"]["horizon"] == "5"
    assert call["keywords"]["window"] == "window"
    assert call["keywords"]["allow_fallback_on_insufficient_data"] == "False"
    assert result["observed_binding"]["source"]["program"][
        "configuration_expressions"
    ] == {"SYMBOLS": "['SPY']"}


def test_missing_callbacks_are_declared_gaps_and_not_model_steps(bindings, policy):
    result = expand_definition(bindings["synthetic"], policy)
    for stage_id in ("label_definition", "filter_definition"):
        row = stage(result, stage_id)
        assert row["status"] == "declared"
        assert row["source_references"] == []
        assert (
            row["reference_gap"] == "no_matching_local_function_or_callback_expression"
        )
    assert (
        result["dependency_semantics"]
        == "authored_definition_order_not_observed_execution_order"
    )


def test_slot_and_wrapper_never_gain_model_or_strategy_implementation(bindings, policy):
    slot = expand_definition(bindings["slot"], policy)
    wrapper = expand_definition(bindings["wrapper"], policy)
    assert slot["implementation_kind"] == "no_dedicated_implementation"
    assert [row["id"] for row in slot["stages"]] == [
        "registry_binding",
        "completion_evidence",
    ]
    assert all(
        row["status"] == "declared" and row["source_references"] == []
        for row in slot["stages"]
    )
    assert stage(slot, "registry_binding")["declaration_references"][0]["selector"] == {
        "bot_id": "slot"
    }
    assert slot["observed_binding"]["registry_definition"]["target_functions"] == [
        "declared_future_rule"
    ]
    assert wrapper["implementation_kind"] == "metadata_only_wrapper"
    for stage_id, function in (
        ("metadata_description", "describe_bot"),
        ("blocked_training_metadata", "train_brain"),
    ):
        row = stage(wrapper, stage_id)
        assert row["status"] == "implemented_by_source_reference"
        assert [ref["symbol"] for ref in row["source_references"]] == [function]
    assert all(
        row["id"]
        not in {
            "program_entrypoints",
            "input_validation",
            "feature_definition",
            "label_definition",
            "filter_definition",
            "evaluation",
            "output_publication",
        }
        for row in wrapper["stages"]
    )


@pytest.mark.parametrize("flag", FALSE_AUTHORITY)
@pytest.mark.parametrize("value", [True, 0, 1, "false", None, [], {}])
def test_authority_must_be_literal_false(policy, flag, value):
    policy["authority"][flag] = value
    assert "process_authority_invalid" in validate_policy(policy)


@pytest.mark.parametrize("field", ["authority", "profiles", "stages"])
@pytest.mark.parametrize("value", [None, [], {}, "bad"])
def test_malformed_top_level_shapes_do_not_crash(policy, field, value):
    policy[field] = value
    assert validate_policy(policy)


@pytest.mark.parametrize(
    "mutation, error",
    [
        ("self_cycle", "dependency_cycle"),
        ("cycle", "dependency_cycle"),
        ("unknown_dependency", "dependency_id_invalid"),
        ("forward_dependency", "dependency_order_invalid"),
        ("disconnected", "dependency_coverage_invalid"),
        ("incomplete_evidence", "dependency_coverage_invalid"),
        ("duplicate_id", "stage_ids_invalid"),
        ("missing_stage", "stage_ids_invalid"),
        ("invalid_id", "stage_shape_invalid"),
        ("duplicate_dependency", "stage_shape_invalid"),
        ("nested_dependency", "stage_shape_invalid"),
        ("nonlist_dependencies", "stage_shape_invalid"),
        ("unknown_stage_field", "stage_shape_invalid"),
    ],
)
def test_graph_rejection(policy, bindings, mutation, error):
    nodes = policy["profiles"]["runtime_model_program"]["stages"]
    if mutation == "self_cycle":
        nodes[1]["depends_on"] = [nodes[1]["id"]]
    elif mutation == "cycle":
        nodes[0]["depends_on"] = [nodes[-1]["id"]]
    elif mutation == "unknown_dependency":
        nodes[1]["depends_on"] = ["missing"]
    elif mutation == "forward_dependency":
        nodes[2], nodes[1] = nodes[1], nodes[2]
    elif mutation == "disconnected":
        nodes[2]["depends_on"] = []
    elif mutation == "incomplete_evidence":
        nodes[-1]["depends_on"] = ["registry_binding"]
    elif mutation == "duplicate_id":
        nodes.append(deepcopy(nodes[1]))
    elif mutation == "missing_stage":
        nodes.pop(2)
    elif mutation == "invalid_id":
        nodes[1]["id"] = "../execute"
    elif mutation == "duplicate_dependency":
        nodes[1]["depends_on"] *= 2
    elif mutation == "nested_dependency":
        nodes[1]["depends_on"] = [{}]
    elif mutation == "nonlist_dependencies":
        nodes[1]["depends_on"] = "registry_binding"
    else:
        nodes[1]["command"] = "execute"
    assert f"process_{error}:runtime_model_program" in validate_policy(policy)
    with pytest.raises(ValueError, match=error):
        expand_definition(bindings["runtime"], policy)


def test_valid_independent_stage_reordering_is_authored_and_hash_bound(
    bindings, policy
):
    before = expand_definition(bindings["runtime"], policy)
    nodes = policy["profiles"]["runtime_model_program"]["stages"]
    feature_index = next(
        i for i, node in enumerate(nodes) if node["id"] == "feature_definition"
    )
    label_index = next(
        i for i, node in enumerate(nodes) if node["id"] == "label_definition"
    )
    nodes[feature_index], nodes[label_index] = nodes[label_index], nodes[feature_index]
    assert validate_policy(policy) == []
    after = expand_definition(bindings["runtime"], policy)
    assert after["definition_sha256"] != before["definition_sha256"]
    assert after["binding_sha256"] == before["binding_sha256"]
    assert after["economic_evidence"] == before["economic_evidence"]


@pytest.mark.parametrize(
    "field, value",
    [
        ("owner", "new_worker"),
        ("owner", {}),
        ("retry_policy", {"maximum_attempts": 3}),
        ("retry_policy", "automatic_retry"),
        ("failure_policy", "ignore_and_continue"),
        ("completion_evidence", ["profitability_verified"]),
        ("completion_evidence", ["tests_passed"]),
        ("applicable_profiles", ["registry_declared_slot"]),
        ("inputs", []),
        ("outputs", [""]),
        ("title", None),
        ("runtime_verified", True),
        ("maximum_attempts", 1),
    ],
)
@pytest.mark.parametrize(
    "stage_id",
    ["feature_definition", "input_validation", "evaluation", "output_publication"],
)
def test_stage_semantics_are_strict(policy, field, value, stage_id):
    policy["stages"][stage_id][field] = value
    assert f"process_stage_contract_invalid:{stage_id}" in validate_policy(policy)


@pytest.mark.parametrize(
    "profile",
    ["collection_wrapper", "registry_declared_slot", "synthetic_research_program"],
)
def test_runtime_binding_cannot_be_reclassified(bindings, policy, profile):
    bindings["runtime"]["profile_id"] = profile
    with pytest.raises(
        ValueError, match="process_(profile_source_mismatch|slot_has_implementation)"
    ):
        expand_definition(bindings["runtime"], policy)


@pytest.mark.parametrize(
    "key",
    [
        "execution_enabled",
        "direct_execution_allowed",
        "live_trading_enabled",
        "starts_workers",
    ],
)
@pytest.mark.parametrize("container", ["registry_definition", "module_spec"])
def test_observed_declarations_cannot_grant_authority(bindings, policy, key, container):
    bindings["wrapper"][container][key] = True
    with pytest.raises(ValueError, match="process_binding_authority_invalid"):
        expand_definition(bindings["wrapper"], policy)


@pytest.mark.parametrize(
    "mutation",
    [
        "bot_id",
        "bot_role",
        "spec_identity",
        "source_hash",
        "source_path",
        "dependency_hash",
        "dependency_path",
        "order_call",
        "kind_unknown",
        "kind_not_boolean",
        "function_lines",
        "function_defaults",
        "call_keywords",
        "call_line",
        "program_shape",
        "nonfinite_declaration",
        "extra_authority",
    ],
)
def test_invalid_observed_bindings_rejected(bindings, policy, mutation):
    binding = bindings["runtime"]
    program = binding["source"]["program"]
    if mutation == "bot_id":
        binding["bot_id"] = "other"
    elif mutation == "bot_role":
        binding["registry_definition"]["bot_role"] = []
    elif mutation == "spec_identity":
        binding["module_spec"]["bot_id"] = "other"
    elif mutation == "source_hash":
        binding["source"]["sha256"] = "stale"
    elif mutation == "source_path":
        binding["source"]["path"] = "core/../outside.py"
    elif mutation == "dependency_hash":
        binding["dependency_hashes"]["core/helper.py"] = None
    elif mutation == "dependency_path":
        binding["dependency_hashes"]["/external/helper.py"] = "a" * 64
    elif mutation == "order_call":
        program["order_calls"] = ["broker.place_order"]
    elif mutation == "kind_unknown":
        program["runtime_training_path_present"] = False
    elif mutation == "kind_not_boolean":
        program["runtime_training_path_present"] = 1
    elif mutation == "function_lines":
        program["functions"]["vector"]["end_line"] = 0
    elif mutation == "function_defaults":
        program["functions"]["vector"]["defaults"] = []
    elif mutation == "call_keywords":
        program["training_calls"][0]["keywords"] = None
    elif mutation == "call_line":
        program["training_calls"][0]["line"] = True
    elif mutation == "program_shape":
        program.pop("functions")
    elif mutation == "nonfinite_declaration":
        binding["registry_definition"]["minimum_training_observations"] = float("nan")
    else:
        binding["authority"] = {"starts_workers": True}
    with pytest.raises(ValueError, match="process_"):
        expand_definition(binding, policy)


def test_exact_callback_expressions_and_qualified_function_names(bindings, policy):
    code = (
        "from helpers import train_runtime_indicator_bot\n"
        "class Spec:\n"
        "    horizon: int = 4\n"
        "    def features(self, window=24):\n        return window\n"
        "def features(window=14):\n    return window\n"
        "def train_brain():\n"
        "    return train_runtime_indicator_bot(runtime_feature_builder=lambda rows, idx: rows[idx], "
        "runtime_label_builder=None, sample_filter=Spec().features)\n"
    )
    binding = bindings["runtime"]
    binding["source"]["program"] = inspect_program(ast.parse(code))
    binding["source"]["sha256"] = hashlib.sha256(code.encode()).hexdigest()
    result = expand_definition(binding, policy)
    refs = stage(result, "feature_definition")["source_references"]
    functions = {
        ref["symbol"]: ref for ref in refs if ref["kind"] == "function_definition"
    }
    assert functions["Spec.features"]["defaults"] == {"window": "24"}
    assert functions["features"]["defaults"] == {"window": "14"}
    assert (
        next(ref for ref in refs if ref["kind"] == "callback_expression")[
            "local_function"
        ]
        is None
    )
    assert stage(result, "label_definition")["status"] == "declared"
    assert (
        stage(result, "filter_definition")["source_references"][0]["expression"]
        == "Spec().features"
    )


def test_real_runtime_source_keeps_synthetic_fallback_and_exact_callbacks(
    bindings, policy
):
    relative = "core/brain_refinery_v13_choppy.py"
    code = (ROOT / relative).read_text()
    binding = bindings["runtime"]
    binding["source"] = {
        "path": relative,
        "sha256": hashlib.sha256(code.encode()).hexdigest(),
        "program": inspect_program(ast.parse(code)),
    }
    binding["bot_id"] = "brain_refinery_v13_choppy"
    binding["registry_definition"]["bot_id"] = binding["bot_id"]
    binding["catalog_binding"]["core_file"] = relative
    result = expand_definition(binding, policy)
    program = result["observed_binding"]["source"]["program"]
    assert program["runtime_training_path_present"]
    assert program["synthetic_training_path_present"]
    assert result["profile_id"] == "runtime_model_program"
    for stage_id, expression in (
        ("feature_definition", "_runtime_feature_vector"),
        ("label_definition", "_runtime_choppy_label"),
        ("filter_definition", "_runtime_sample_filter"),
    ):
        refs = stage(result, stage_id)["source_references"]
        ref = next(ref for ref in refs if ref.get("expression") == expression)
        assert ref["local_function"]["symbol"] == expression
        assert ref["local_function"]["line"] == program["functions"][expression]["line"]


@pytest.mark.parametrize(
    "relative",
    [
        "core/brain_refinery_v109_defensive_options_risk_off_teacher.py",
        "core/brain_refinery_v634_dividend_free_cash_flow_yield_quality_bot.py",
    ],
)
def test_real_wrapper_source_stays_metadata_only(bindings, policy, relative):
    code = (ROOT / relative).read_text()
    tree = ast.parse(code)
    spec = next(
        ast.literal_eval(node.value)
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "BOT_SPEC"
            for target in node.targets
        )
    )
    binding = bindings["wrapper"]
    binding.update(bot_id=spec["bot_id"], module_spec=spec)
    binding["registry_definition"].update(
        bot_id=spec["bot_id"], bot_role=spec["bot_role"]
    )
    binding["source"] = {
        "path": relative,
        "sha256": hashlib.sha256(code.encode()).hexdigest(),
        "program": inspect_program(tree),
    }
    binding["catalog_binding"]["core_file"] = relative
    result = expand_definition(binding, policy)
    assert result["implementation_kind"] == "metadata_only_wrapper"
    assert result["observed_binding"]["module_spec"] == spec
    assert (
        stage(result, "blocked_training_metadata")["source_references"][0]["symbol"]
        == "train_brain"
    )
    assert result["runnable_strategy_verified"] is False


def test_expansion_is_pure_and_never_probes_sources(bindings, policy, monkeypatch):
    import builtins
    import subprocess

    def forbidden(*args, **kwargs):
        raise AssertionError(
            "expansion may not read, stat, import a program, or start work"
        )

    with monkeypatch.context() as patch:
        patch.setattr(builtins, "open", forbidden)
        patch.setattr(Path, "stat", forbidden)
        patch.setattr(Path, "lstat", forbidden)
        patch.setattr(Path, "read_text", forbidden)
        patch.setattr(subprocess, "Popen", forbidden)
        patch.setattr(builtins, "__import__", forbidden)
        expand_definition(bindings["runtime"], policy)
        expand_definition(bindings["slot"], policy)
        bindings["runtime"]["source"]["path"] = "/external/forbidden.py"
        try:
            expand_definition(bindings["runtime"], policy)
        except ValueError as exc:
            assert str(exc) == "process_program_facts_invalid"
        else:
            raise AssertionError("external source was accepted")


def test_returned_data_has_no_mutable_aliases_to_inputs(bindings, policy):
    before = deepcopy((bindings, policy))
    result = expand_definition(bindings["runtime"], policy)
    result["observed_binding"]["registry_definition"]["target_functions"].append(
        "changed"
    )
    stage(result, "feature_definition")["inputs"].append("changed")
    stage(result, "program_entrypoints")["source_references"][0]["defaults"][
        "new"
    ] = "value"
    assert (bindings, policy) == before


def test_project_local_dependency_receipts_are_preserved(bindings, policy):
    binding = bindings["runtime"]
    binding["dependency_hashes"].update(
        {"scripts/observed_helper.py": "a" * 64, "observed_helper.py": "b" * 64}
    )
    result = expand_definition(binding, policy)
    assert (
        result["observed_binding"]["dependency_hashes"] == binding["dependency_hashes"]
    )


@pytest.mark.parametrize("bot_id", ["runtime", "synthetic"])
@pytest.mark.parametrize(
    "stage_id", ["input_validation", "evaluation", "output_publication"]
)
def test_new_model_stages_are_required_explicit_gaps(
    bindings, policy, bot_id, stage_id
):
    binding = bindings[bot_id]
    result = expand_definition(binding, policy)
    assert len(result["stages"]) == 9
    row = stage(result, stage_id)
    assert row["status"] == "declared"
    assert row["source_references"] == []
    assert row["reference_gap"] == "no_matching_local_function_or_callback_expression"
    assert row["owner"]["path"] == binding["source"]["path"]
    assert row["failure_policy"] == "source_failure_semantics_not_verified"
    assert row["retry_policy"] == "existing_owner_policy_not_assessed"
    assert row["runtime_verified"] is False
    assert row["completion_evidence"]["runtime_completion_verified"] is False
    nodes = policy["profiles"][binding["profile_id"]]["stages"]
    nodes[:] = [node for node in nodes if node["id"] != stage_id]
    with pytest.raises(ValueError, match="process_stage_ids_invalid"):
        expand_definition(binding, policy)


@pytest.mark.parametrize(
    "binding_id, trainer",
    [
        ("runtime", "train_runtime_indicator_bot"),
        ("synthetic", "train_indicator_bot"),
    ],
)
def test_new_model_stages_bind_named_functions_and_exact_callbacks(
    bindings, policy, binding_id, trainer
):
    code = (
        f"from helpers import {trainer}\n"
        "def _validate_input_schema(rows):\n    return rows\n"
        "class Research:\n"
        "    def evaluate(self, predictions, labels):\n        return {}\n"
        "def save_artifacts(result, path):\n    return None\n"
        "def accept(rows):\n    return rows\n"
        "def measure(predictions, labels):\n    return {}\n"
        "def persist(result):\n    return None\n"
        "def train_brain():\n"
        f"    return {trainer}(input_validator=accept, evaluator=measure, output_writer=persist)\n"
    )
    binding = bindings[binding_id]
    program = inspect_program(ast.parse(code))
    binding["source"]["program"] = program
    binding["source"]["sha256"] = hashlib.sha256(code.encode()).hexdigest()
    result = expand_definition(binding, policy)
    for stage_id, symbol, callback in (
        ("input_validation", "_validate_input_schema", "accept"),
        ("evaluation", "Research.evaluate", "measure"),
        ("output_publication", "save_artifacts", "persist"),
    ):
        row = stage(result, stage_id)
        assert row["status"] == "implemented_by_source_reference"
        definition = next(
            ref
            for ref in row["source_references"]
            if ref["kind"] == "function_definition"
        )
        assert definition["symbol"] == symbol
        assert definition["line"] == program["functions"][symbol]["line"]
        assert (
            definition["selection_basis"] == "function_name_only_not_proven_call_path"
        )
        expression = next(
            ref
            for ref in row["source_references"]
            if ref["kind"] == "callback_expression"
        )
        assert expression["expression"] == callback
        assert expression["local_function"]["symbol"] == callback
        assert expression["name_resolution_verified"] is False
        assert row["runtime_verified"] is False
        assert row["completion_evidence"]["runtime_receipts"] == []


def test_new_stages_do_not_infer_behavior_from_unrelated_names(bindings, policy):
    code = (
        "from helpers import train_runtime_indicator_bot\n"
        "def validate_model(model):\n    return model\n"
        "def save_inputs(rows):\n    return rows\n"
        "def train_brain():\n"
        "    return train_runtime_indicator_bot(input_validator=None, output_writer=None)\n"
    )
    binding = bindings["runtime"]
    binding["source"]["program"] = inspect_program(ast.parse(code))
    binding["source"]["sha256"] = hashlib.sha256(code.encode()).hexdigest()
    result = expand_definition(binding, policy)
    for stage_id in ("input_validation", "output_publication"):
        assert stage(result, stage_id)["status"] == "declared"
        assert stage(result, stage_id)["source_references"] == []


@pytest.mark.parametrize("container", ["registry_definition", "module_spec"])
def test_unspecified_legacy_direct_execution_is_preserved(bindings, policy, container):
    binding = bindings["wrapper"]
    binding[container]["direct_execution_allowed"] = None
    before = deepcopy(binding)
    result = expand_definition(binding, policy)
    assert result["observed_binding"] == before
    assert result["observed_binding"][container]["direct_execution_allowed"] is None
    assert result["implementation_kind"] == "metadata_only_wrapper"
    assert not any(result["authority"].values())
    assert result["runnable_strategy_verified"] is False
    assert binding == before


@pytest.mark.parametrize("value", [True, 0, 1, "false", "null", [], {}])
def test_malformed_legacy_direct_execution_still_rejected(bindings, policy, value):
    bindings["wrapper"]["module_spec"]["direct_execution_allowed"] = value
    with pytest.raises(ValueError, match="process_binding_authority_invalid"):
        expand_definition(bindings["wrapper"], policy)


@pytest.mark.parametrize(
    "key",
    [
        "starts_workers",
        "submits_orders",
        "claims_economic_evidence",
        "execution_enabled",
        "live_trading_enabled",
    ],
)
def test_other_null_authority_fields_remain_invalid(bindings, policy, key):
    bindings["wrapper"]["module_spec"][key] = None
    with pytest.raises(ValueError, match="process_binding_authority_invalid"):
        expand_definition(bindings["wrapper"], policy)
