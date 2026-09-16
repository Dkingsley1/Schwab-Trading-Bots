"""Pure process-definition expansion from an already observed operating binding.

The caller must first validate the pinned operating catalog. This module performs
no source reads, imports of bot programs, execution, materialization or repinning.
Its graph orders authored definitions, not an inferred runtime workflow.
"""

from __future__ import annotations

import ast
from copy import deepcopy
import re
from typing import Any, Mapping

from core.bot_definition_contracts import FALSE_AUTHORITY, KNOWN_ROLES, digest

POLICY_PATH = "config/bot_process_definition_contract_v1.json"
SCOPE = "authored_process_definition_not_runtime_or_economic_completion"
_KINDS = {
    "collection_wrapper": "metadata_only_wrapper",
    "registry_declared_slot": "no_dedicated_implementation",
    "runtime_model_program": "source_defined_runtime_model",
    "synthetic_research_program": "source_defined_synthetic_or_offline_research",
}
_MODELS = {"runtime_model_program", "synthetic_research_program"}
_APPLICABILITY = {
    "registry_binding": set(_KINDS),
    "metadata_description": {"collection_wrapper"},
    "blocked_training_metadata": {"collection_wrapper"},
    "program_entrypoints": _MODELS,
    "input_validation": _MODELS,
    "feature_definition": _MODELS,
    "label_definition": _MODELS,
    "filter_definition": _MODELS,
    "evaluation": _MODELS,
    "output_publication": _MODELS,
    "completion_evidence": set(_KINDS),
}
_CALLBACKS = {
    "input_validation": {
        "input_validator",
        "input_validation_callback",
        "data_validator",
    },
    "feature_definition": {"feature_builder", "runtime_feature_builder"},
    "label_definition": {"label_builder", "runtime_label_builder"},
    "filter_definition": {"sample_filter", "confidence_builder", "observation_filter"},
    "evaluation": {"evaluator", "evaluation_callback", "evaluation_fn"},
    "output_publication": {
        "output_writer",
        "output_publisher",
        "publication_callback",
        "artifact_writer",
        "artifact_publisher",
    },
}
_TOKENS = {
    "feature_definition": {"feature", "features"},
    "label_definition": {"label", "labels"},
    "filter_definition": {"filter", "confidence"},
    "evaluation": {"evaluate", "evaluation", "evaluator"},
}
_TOKEN_GROUPS = {
    "input_validation": (
        {"validate", "check", "verify"},
        {
            "input",
            "inputs",
            "data",
            "dataset",
            "samples",
            "sequences",
            "observations",
            "schema",
        },
    ),
    "output_publication": (
        {"publish", "save", "write", "export"},
        {
            "output",
            "outputs",
            "artifact",
            "artifacts",
            "model",
            "weights",
            "metrics",
            "results",
            "report",
            "checkpoint",
        },
    ),
}
_TRAINERS = {
    "train_indicator_bot",
    "train_price_indicator_bot",
    "train_runtime_indicator_bot",
    "train_crypto_runtime_bot",
}
_AUTHORITY_FIELDS = set(FALSE_AUTHORITY) | {
    "execution_enabled",
    "direct_execution_allowed",
    "live_trading_enabled",
    "paper_trading_enabled",
    "trading_enabled",
    "allocation_enabled",
}


def _keys(value: Any, keys: set[str]) -> bool:
    return isinstance(value, dict) and set(value) == keys


def _text(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _strings(value: Any, *, nonempty: bool = False) -> bool:
    return (
        isinstance(value, list)
        and (bool(value) or not nonempty)
        and all(_text(item) for item in value)
        and len(value) == len(set(value))
    )


def _graph_errors(nodes: Any, expected: set[str]) -> list[str]:
    if not isinstance(nodes, list) or not nodes:
        return ["stages_invalid"]
    if any(
        not _keys(node, {"id", "depends_on"})
        or not isinstance(node["id"], str)
        or not re.fullmatch(r"[a-z][a-z0-9_]*", node["id"])
        or not _strings(node["depends_on"])
        for node in nodes
    ):
        return ["stage_shape_invalid"]
    ids = [node["id"] for node in nodes]
    if len(set(ids)) != len(ids) or set(ids) != expected:
        return ["stage_ids_invalid"]
    dependencies = {node["id"]: set(node["depends_on"]) for node in nodes}
    if any(not deps <= expected for deps in dependencies.values()):
        return ["dependency_id_invalid"]
    errors = []
    remaining = dict(dependencies)
    resolved: set[str] = set()
    while remaining:
        ready = {key for key, deps in remaining.items() if deps <= resolved}
        if not ready:
            return ["dependency_cycle"]
        resolved.update(ready)
        remaining = {key: deps for key, deps in remaining.items() if key not in ready}
    seen: set[str] = set()
    ancestors: dict[str, set[str]] = {}
    for key in ids:
        if not dependencies[key] <= seen:
            errors.append("dependency_order_invalid")
            break
        ancestors[key] = set(dependencies[key])
        for dependency in dependencies[key]:
            ancestors[key].update(ancestors[dependency])
        seen.add(key)
    if not errors and (
        ids[0] != "registry_binding"
        or ids[-1] != "completion_evidence"
        or any("registry_binding" not in ancestors[key] for key in ids[1:])
        or ancestors["completion_evidence"] != expected - {"completion_evidence"}
    ):
        errors.append("dependency_coverage_invalid")
    return errors


def validate_policy(policy: Mapping[str, Any]) -> list[str]:
    """Return deterministic contract/graph errors; reject extra authority fields."""
    if not _keys(
        policy,
        {
            "schema_version",
            "contract_id",
            "mode",
            "completion_scope",
            "dependency_semantics",
            "authority",
            "profiles",
            "stages",
        },
    ):
        return ["process_policy_fields_invalid"]
    errors = []
    if (
        type(policy["schema_version"]) is not int
        or policy["schema_version"] != 1
        or policy["contract_id"] != "bot_process_definitions_v1"
        or policy["mode"] != "source_bound_definition_only"
        or policy["completion_scope"] != SCOPE
        or policy["dependency_semantics"]
        != "authored_definition_order_not_observed_execution_order"
    ):
        errors.append("process_scope_invalid")
    if not _keys(policy["authority"], set(FALSE_AUTHORITY)) or any(
        value is not False for value in policy["authority"].values()
    ):
        errors.append("process_authority_invalid")
    if not _keys(policy["stages"], set(_APPLICABILITY)):
        errors.append("process_stage_templates_invalid")
    else:
        for key, applicable in _APPLICABILITY.items():
            stage = policy["stages"][key]
            definition_only = key in {"registry_binding", "completion_evidence"}
            owner = (
                "registry_definition_owner"
                if key == "registry_binding"
                else (
                    "bot_organization_controller"
                    if key == "completion_evidence"
                    else "bound_source_owner"
                )
            )
            evidence = (
                ["matching_registry_projection"]
                if key == "registry_binding"
                else (
                    [
                        "valid_authored_graph_and_binding",
                        "separate_runtime_and_economic_evidence",
                    ]
                    if key == "completion_evidence"
                    else [
                        "source_reference_or_explicit_gap",
                        "independent_runtime_receipt_required",
                    ]
                )
            )
            if (
                not _keys(
                    stage,
                    {
                        "title",
                        "applicable_profiles",
                        "owner",
                        "inputs",
                        "outputs",
                        "failure_policy",
                        "retry_policy",
                        "completion_evidence",
                    },
                )
                or not _text(stage["title"])
                or not _strings(stage["applicable_profiles"], nonempty=True)
                or set(stage["applicable_profiles"]) != applicable
                or stage["owner"] != owner
                or not _strings(stage["inputs"], nonempty=True)
                or not _strings(stage["outputs"], nonempty=True)
                or stage["failure_policy"]
                != (
                    "reject_invalid_binding"
                    if definition_only
                    else "source_failure_semantics_not_verified"
                )
                or stage["retry_policy"]
                != (
                    "no_retry_definition_only"
                    if definition_only
                    else "existing_owner_policy_not_assessed"
                )
                or stage["completion_evidence"] != evidence
            ):
                errors.append(f"process_stage_contract_invalid:{key}")
    if not _keys(policy["profiles"], set(_KINDS)):
        errors.append("process_profiles_invalid")
    else:
        for key, kind in _KINDS.items():
            profile = policy["profiles"][key]
            if (
                not _keys(profile, {"implementation_kind", "purpose", "stages"})
                or profile["implementation_kind"] != kind
                or not _text(profile["purpose"])
            ):
                errors.append(f"process_profile_invalid:{key}")
                continue
            expected = {
                stage for stage, profiles in _APPLICABILITY.items() if key in profiles
            }
            errors.extend(
                f"process_{error}:{key}"
                for error in _graph_errors(profile["stages"], expected)
            )
    return errors


def _source_path(value: Any) -> bool:
    # Lexical validation only: even invalid paths must never trigger a stat/read.
    return isinstance(value, str) and bool(
        re.fullmatch(r"(?:[A-Za-z0-9_]+/)*[A-Za-z0-9_]+\.py", value)
    )


def _sha(value: Any) -> bool:
    return isinstance(value, str) and bool(re.fullmatch(r"[0-9a-f]{64}", value))


def _program_valid(program: Any) -> bool:
    if not _keys(
        program,
        {
            "functions",
            "class_field_defaults",
            "configuration_expressions",
            "training_calls",
            "literal_input_keys",
            "imports",
            "order_calls",
            "runtime_training_path_present",
            "synthetic_training_path_present",
            "collection_helper_present",
        },
    ):
        return False
    if any(
        type(program[key]) is not bool
        for key in (
            "runtime_training_path_present",
            "synthetic_training_path_present",
            "collection_helper_present",
        )
    ) or any(
        not _strings(program[key])
        for key in ("literal_input_keys", "imports", "order_calls")
    ):
        return False
    for key in ("functions", "class_field_defaults", "configuration_expressions"):
        if not isinstance(program[key], dict):
            return False
    for name, function in program["functions"].items():
        if (
            not isinstance(name, str)
            or not re.fullmatch(
                r"[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*", name
            )
            or not _keys(function, {"line", "end_line", "signature", "defaults"})
            or type(function["line"]) is not int
            or function["line"] < 1
            or type(function["end_line"]) is not int
            or function["end_line"] < function["line"]
            or not isinstance(function["signature"], str)
            or not isinstance(function["defaults"], dict)
            or any(
                not _text(k) or not _text(v) for k, v in function["defaults"].items()
            )
        ):
            return False
    for key in ("class_field_defaults", "configuration_expressions"):
        for name, value in program[key].items():
            if not _text(name):
                return False
            if key == "class_field_defaults":
                if not isinstance(value, dict) or any(
                    not _text(k) or not _text(v) for k, v in value.items()
                ):
                    return False
            elif not _text(value):
                return False
    calls = program["training_calls"]
    return isinstance(calls, list) and all(
        _keys(call, {"call", "line", "arguments", "keywords"})
        and _text(call["call"])
        and type(call["line"]) is int
        and call["line"] > 0
        and isinstance(call["arguments"], list)
        and all(_text(arg) for arg in call["arguments"])
        and isinstance(call["keywords"], dict)
        and all(_text(k) and _text(v) for k, v in call["keywords"].items())
        for call in calls
    )


def _validate_binding(binding: Any) -> None:
    if not _keys(
        binding,
        {
            "bot_id",
            "profile_id",
            "registry_definition",
            "catalog_binding",
            "source",
            "module_spec",
            "dependency_hashes",
        },
    ):
        raise ValueError("process_binding_fields_invalid")
    profile = binding["profile_id"]
    registry, spec, source = (
        binding["registry_definition"],
        binding["module_spec"],
        binding["source"],
    )
    if (
        not _text(binding["bot_id"])
        or not isinstance(profile, str)
        or profile not in _KINDS
        or not isinstance(registry, dict)
        or registry.get("bot_id") != binding["bot_id"]
        or not isinstance(registry.get("bot_role"), str)
        or registry["bot_role"] not in KNOWN_ROLES
        or not isinstance(spec, dict)
        or spec.get("bot_id", binding["bot_id"]) != binding["bot_id"]
        or spec.get("bot_role", registry["bot_role"]) != registry["bot_role"]
    ):
        raise ValueError("process_binding_identity_invalid")
    for declaration in (registry, spec):
        # Legacy direct-execution metadata may be unspecified. Preserve null;
        # it is not permission, and process authority remains literal false.
        if any(
            key in declaration
            and declaration[key] is not False
            and not (key == "direct_execution_allowed" and declaration[key] is None)
            for key in _AUTHORITY_FIELDS
        ):
            raise ValueError("process_binding_authority_invalid")
    catalog = binding["catalog_binding"]
    dependencies = binding["dependency_hashes"]
    if (
        not _keys(source, {"path", "sha256", "program"})
        or not _keys(catalog, {"core_file", "runner"})
        or not isinstance(catalog["runner"], str)
        or (catalog["core_file"] != "" and not _source_path(catalog["core_file"]))
        or not isinstance(dependencies, dict)
        or any(
            not _source_path(path) or not _sha(sha)
            for path, sha in dependencies.items()
        )
    ):
        raise ValueError("process_source_binding_invalid")
    if profile == "registry_declared_slot":
        if (
            source != {"path": "", "sha256": None, "program": {}}
            or spec
            or dependencies
            or catalog["core_file"]
        ):
            raise ValueError("process_slot_has_implementation")
    else:
        program = source["program"]
        if (
            not _source_path(source["path"])
            or not _sha(source["sha256"])
            or not _program_valid(program)
        ):
            raise ValueError("process_program_facts_invalid")
        if program["order_calls"]:
            raise ValueError("process_program_has_order_calls")
        observed_profile = (
            "collection_wrapper"
            if program["collection_helper_present"]
            else (
                "runtime_model_program"
                if program["runtime_training_path_present"]
                else (
                    "synthetic_research_program"
                    if program["synthetic_training_path_present"]
                    else None
                )
            )
        )
        if observed_profile != profile:
            raise ValueError("process_profile_source_mismatch")
    try:
        digest(binding)
    except (TypeError, ValueError, RecursionError) as exc:
        raise ValueError("process_binding_not_json") from exc


def _references(stage_id: str, source: Mapping[str, Any]) -> list[dict[str, Any]]:
    program = source["program"]
    functions = program.get("functions", {})
    base = {"path": source["path"], "sha256": source["sha256"]}
    references = []
    exact = {
        "metadata_description": {"describe_bot"},
        "blocked_training_metadata": {"train_brain"},
        "program_entrypoints": {"train_brain", "main"},
    }.get(stage_id, set())
    for name, facts in functions.items():
        tokens = set(name.rsplit(".", 1)[-1].split("_"))
        groups = _TOKEN_GROUPS.get(stage_id, ())
        if (
            name in exact
            or tokens & _TOKENS.get(stage_id, set())
            or (groups and all(tokens & group for group in groups))
        ):
            references.append(
                {
                    **base,
                    "kind": "function_definition",
                    "symbol": name,
                    **deepcopy(facts),
                    "selection_basis": (
                        "named_entrypoint"
                        if name in exact
                        else "function_name_only_not_proven_call_path"
                    ),
                    "verification_scope": "function_definition_presence_only",
                }
            )
    for call in program.get("training_calls", []):
        if stage_id == "program_entrypoints" and call["call"] in _TRAINERS:
            references.append(
                {
                    **base,
                    "kind": "trainer_call_expression",
                    **deepcopy(call),
                    "verification_scope": "callsite_presence_not_invocation_or_helper_implementation",
                }
            )
        for keyword, expression in call["keywords"].items():
            if keyword not in _CALLBACKS.get(stage_id, set()):
                continue
            try:
                node = ast.parse(expression, mode="eval").body
            except (SyntaxError, ValueError, RecursionError) as exc:
                raise ValueError("process_callback_expression_invalid") from exc
            if isinstance(node, ast.Constant) and node.value is None:
                continue
            # Attach observed same-name definitions, without claiming runtime
            # name resolution. Factories and imports remain callsite expressions.
            local = functions.get(node.id) if isinstance(node, ast.Name) else None
            references.append(
                {
                    **base,
                    "kind": "callback_expression",
                    "call": call["call"],
                    "line": call["line"],
                    "keyword": keyword,
                    "expression": expression,
                    "local_function": (
                        {"symbol": node.id, **deepcopy(local)} if local else None
                    ),
                    "name_resolution_verified": False,
                    "verification_scope": "callback_expression_presence_only",
                }
            )
    return references


def expand_definition(
    observed_binding: Mapping[str, Any], policy: Mapping[str, Any]
) -> dict[str, Any]:
    """Expand a validated operating record's ``binding`` with an authored policy.

    Shape/classification validation cannot certify source freshness. The parent
    audit retains that responsibility and must pin this policy and resolver.
    Missing local callbacks remain explicit gaps, never invented implementations.
    """
    errors = validate_policy(policy)
    if errors:
        raise ValueError(";".join(errors))
    _validate_binding(observed_binding)
    binding = deepcopy(observed_binding)
    profile_id = binding["profile_id"]
    profile = policy["profiles"][profile_id]
    registry_ref = {
        "kind": "registry_projection",
        "path": "master_bot_registry.json",
        "selector": {"bot_id": binding["bot_id"]},
        "projection_sha256": digest(binding["registry_definition"]),
    }
    stages = []
    for order, node in enumerate(profile["stages"], 1):
        stage_id = node["id"]
        template = deepcopy(policy["stages"][stage_id])
        references = _references(stage_id, binding["source"])
        owner = {"id": template.pop("owner")}
        if owner["id"] == "bound_source_owner":
            owner.update(
                path=binding["source"]["path"], sha256=binding["source"]["sha256"]
            )
        elif owner["id"] == "registry_definition_owner":
            owner["reference"] = deepcopy(registry_ref)
        stages.append(
            {
                **template,
                **deepcopy(node),
                "order": order,
                "owner": owner,
                "status": (
                    "implemented_by_source_reference" if references else "declared"
                ),
                "source_references": references,
                "declaration_references": (
                    [deepcopy(registry_ref)] if stage_id == "registry_binding" else []
                ),
                "reference_gap": (
                    "no_matching_local_function_or_callback_expression"
                    if not references and owner["id"] == "bound_source_owner"
                    else None
                ),
                "runtime_verified": False,
                "completion_evidence": {
                    "required": template["completion_evidence"],
                    "runtime_receipts": [],
                    "runtime_completion_verified": False,
                },
            }
        )
    binding_sha = digest(binding)
    policy_sha = digest(policy)
    return {
        "contract_id": policy["contract_id"],
        "completion_scope": SCOPE,
        "bot_id": binding["bot_id"],
        "bot_role": binding["registry_definition"]["bot_role"],
        "profile_id": profile_id,
        "implementation_kind": profile["implementation_kind"],
        "purpose": profile["purpose"],
        "dependency_semantics": policy["dependency_semantics"],
        "binding_sha256": binding_sha,
        "policy_sha256": policy_sha,
        "definition_sha256": digest({"binding": binding_sha, "policy": policy_sha}),
        "observed_binding": binding,
        "stages": stages,
        "definition_valid": True,
        "runtime_verified": False,
        "runnable_strategy_verified": False,
        "authority": {key: False for key in FALSE_AUTHORITY},
        "economic_evidence": {
            "status": "not_assessed",
            "definition_grants_evidence": False,
        },
    }
