"""Pinned operating definitions, separate from executable trading mandates."""

from __future__ import annotations

import ast
import json
from collections import Counter
from copy import deepcopy
from pathlib import Path
import re
from typing import Any, Mapping

from core.bot_definition_contracts import (
    AREAS,
    FALSE_AUTHORITY,
    KNOWN_ROLES,
    SourceInventory,
    digest,
    safe_project_file,
)
from core.bot_process_definitions import (
    POLICY_PATH as PROCESS_POLICY_PATH,
    expand_definition as expand_process_definition,
    validate_policy as validate_process_policy,
)

POLICY_PATH = "config/bot_operating_definition_contract_v1.json"
CATALOG_PATH = "config/bot_operating_definitions_v1.json"
SCOPE = "registered_bot_operating_definitions_not_standalone_trading_mandates"
PROFILE_IDS = {
    "collection_wrapper",
    "registry_declared_slot",
    "runtime_model_program",
    "synthetic_research_program",
}
REGISTRY_FIELDS = (
    "bot_id",
    "bot_role",
    "strategy_family",
    "slot_kind",
    "slot_objective",
    "sleeve_profile",
    "sleeve_family",
    "target_functions",
    "data_intake_collections",
    "storage_targets",
    "label_contract",
    "training_label_materialization_contract",
    "minimum_training_observations",
    "minimum_training_sequences",
    "minimum_data_collection_days",
    "retention_profile",
)
ORDER_CALLS = {
    "place_order",
    "submit_order",
    "place_market_order",
    "submit_live_order",
    "submit_paper_order",
}
AUTHORITY_FIELDS = (
    "execution_enabled",
    "direct_execution_allowed",
    "live_trading_enabled",
)
TRAIN_CALLS = {
    "train_indicator_bot",
    "train_price_indicator_bot",
    "train_runtime_indicator_bot",
    "train_crypto_runtime_bot",
    "CryptoRuntimeSpec",
    "dict",
}
REQUIRED_SHARED_SOURCES = {
    "core/registry_backed_collection_bot.py",
    "core/indicator_bot_common.py",
    "core/runtime_training_common.py",
    "core/crypto_runtime_bot_common.py",
    "core/runtime_requested_bot_common.py",
    "core/institutional_decision_flow.py",
    "core/execution_lane_pipeline.py",
    "core/bot_operating_definitions.py",
    "core/bot_definition_contracts.py",
    "core/bot_process_definitions.py",
}


def load_process_policy(root: Path) -> dict[str, Any]:
    path = safe_project_file(root, PROCESS_POLICY_PATH)
    if path is None:
        raise ValueError("process_policy_missing_or_disallowed")
    try:
        before = path.stat()
        if before.st_size > 64_000:
            raise ValueError("process_policy_size_limit")
        with path.open("rb") as handle:
            raw = handle.read(64_001)
        after = path.stat()
        if len(raw) > 64_000 or (before.st_ino, before.st_size, before.st_mtime_ns) != (
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        ):
            raise ValueError("process_policy_changed_during_read")
        policy = json.loads(raw)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("process_policy_unreadable") from exc
    errors = validate_process_policy(policy)
    if errors:
        raise ValueError(";".join(errors))
    return policy


def _name(node: ast.AST) -> str:
    return ast.unparse(node)


def inspect_program(tree: ast.Module) -> dict[str, Any]:
    functions = {}
    class_defaults = {}
    configurations = {}
    imports = set()
    calls = []
    input_fields = set()
    order_calls = set()
    call_names = set()
    parents = {
        child: parent
        for parent in ast.walk(tree)
        for child in ast.iter_child_nodes(parent)
    }
    aliases = {
        alias.asname or alias.name: alias.name
        for node in tree.body
        if isinstance(node, ast.ImportFrom)
        for alias in node.names
    }
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            class_defaults[node.name] = {
                child.target.id: _name(child.value)
                for child in node.body
                if isinstance(child, ast.AnnAssign)
                and isinstance(child.target, ast.Name)
                and child.value is not None
            }
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            defaults = {}
            positional = node.args.posonlyargs + node.args.args
            for argument, default in zip(
                positional[-len(node.args.defaults) :], node.args.defaults
            ):
                defaults[argument.arg] = _name(default)
            for argument, default in zip(node.args.kwonlyargs, node.args.kw_defaults):
                if default is not None:
                    defaults[argument.arg] = _name(default)
            names = [node.name]
            parent = parents.get(node)
            while parent is not None:
                if isinstance(
                    parent, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
                ):
                    names.append(parent.name)
                parent = parents.get(parent)
            functions[".".join(reversed(names))] = {
                "line": node.lineno,
                "end_line": node.end_lineno,
                "signature": _name(node.args),
                "defaults": defaults,
            }
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module)
        elif isinstance(node, ast.Import):
            imports.update(alias.name for alias in node.names)
        elif isinstance(node, ast.Call):
            name = _name(node.func)
            name = aliases.get(name, name)
            call_names.add(name)
            if name.rsplit(".", 1)[-1] in ORDER_CALLS:
                order_calls.add(name)
            if name in TRAIN_CALLS or name.endswith("_label_builder"):
                calls.append(
                    {
                        "call": name,
                        "line": node.lineno,
                        "arguments": [_name(arg) for arg in node.args],
                        "keywords": {
                            key.arg or "**": _name(key.value) for key in node.keywords
                        },
                    }
                )
            if (
                name.endswith("observation_feature")
                and len(node.args) > 1
                and isinstance(node.args[1], ast.Constant)
                and isinstance(node.args[1].value, str)
            ):
                input_fields.add(node.args[1].value)
        elif (
            isinstance(node, ast.Subscript)
            and isinstance(node.slice, ast.Constant)
            and isinstance(node.slice.value, str)
        ):
            input_fields.add(node.slice.value)
    for node in tree.body:
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        for target in targets:
            if (
                isinstance(target, ast.Name)
                and node.value is not None
                and target.id != "BOT_SPEC"
                and any(
                    token in target.id
                    for token in (
                        "SPEC",
                        "FEATURE",
                        "MODE",
                        "SYMBOL",
                        "HORIZON",
                        "WINDOW",
                    )
                )
            ):
                configurations[target.id] = _name(node.value)
    return {
        "functions": functions,
        "class_field_defaults": class_defaults,
        "configuration_expressions": configurations,
        "training_calls": calls,
        "literal_input_keys": sorted(input_fields),
        "imports": sorted(imports),
        "order_calls": sorted(order_calls),
        "runtime_training_path_present": bool(
            call_names
            & {
                "train_runtime_indicator_bot",
                "train_crypto_runtime_bot",
                "CryptoRuntimeSpec",
            }
        ),
        "synthetic_training_path_present": bool(
            call_names
            & {
                "train_indicator_bot",
                "train_price_indicator_bot",
                "np.random.normal",
                "np.random.lognormal",
            }
        )
        or any(name.startswith("simulate_") for name in call_names),
        "collection_helper_present": bool(
            call_names & {"describe_registry_backed_bot", "train_registry_backed_bot"}
        ),
    }


def validate_policy(policy: Mapping[str, Any]) -> list[str]:
    errors = []
    if (
        policy.get("schema_version") != 1
        or policy.get("contract_id") != "bot_operating_definitions_v1"
    ):
        errors.append("operating_contract_version_invalid")
    if (
        policy.get("completion_scope") != SCOPE
        or policy.get("mode") != "source_bound_definition_only"
    ):
        errors.append("operating_completion_scope_invalid")
    authority = policy.get("authority")
    if (
        not isinstance(authority, dict)
        or set(authority) != set(FALSE_AUTHORITY)
        or any(value is not False for value in authority.values())
    ):
        errors.append("operating_authority_invalid")
    if (
        policy.get("rebind_policy")
        != "explicit_materialization_only_never_scheduled_auto_repin"
    ):
        errors.append("operating_rebind_policy_invalid")
    areas = policy.get("areas")
    if not isinstance(areas, dict) or set(areas) != set(AREAS):
        errors.append("operating_seven_areas_required")
    else:
        for area, fields in AREAS.items():
            values = areas.get(area)
            if (
                not isinstance(values, dict)
                or set(values) != set(fields)
                or any(
                    not isinstance(value, str) or not value.strip()
                    for value in values.values()
                )
            ):
                errors.append(f"operating_area_invalid:{area}")
    subsections = policy.get("subsections")
    if not isinstance(subsections, dict) or set(subsections) != set(AREAS):
        errors.append("operating_subsection_areas_required")
    else:
        for area, fields in AREAS.items():
            titles = subsections[area]
            if (
                not isinstance(titles, dict)
                or set(titles) != set(fields)
                or any(
                    not isinstance(title, str)
                    or not title.strip()
                    or len(title) > 100
                    or any(char in title for char in "\n\r|")
                    for title in titles.values()
                )
            ):
                errors.append(f"operating_subsections_invalid:{area}")
    profiles = policy.get("profiles")
    if not isinstance(profiles, dict) or set(profiles) != PROFILE_IDS:
        errors.append("operating_profiles_invalid")
    else:
        for key, profile in profiles.items():
            if (
                not isinstance(profile, dict)
                or set(profile)
                != {"purpose", "training_status", "hypothesis", "data_origin"}
                or any(
                    not isinstance(value, str) or not value.strip()
                    for value in profile.values()
                )
            ):
                errors.append(f"operating_profile_invalid:{key}")
    budget = policy.get("source_budget", {})
    for field, maximum in (
        ("maximum_module_bytes", 2_000_000),
        ("maximum_total_bytes", 64_000_000),
        ("maximum_files", 2500),
    ):
        value = budget.get(field) if isinstance(budget, dict) else None
        if type(value) is not int or not 1 <= value <= maximum:
            errors.append(f"operating_source_budget_invalid:{field}")
    paths = policy.get("shared_sources")
    if (
        not isinstance(paths, list)
        or not paths
        or not all(
            isinstance(path, str)
            and path.startswith("core/")
            and ".." not in Path(path).parts
            for path in paths
        )
    ):
        errors.append("operating_shared_sources_invalid")
    elif not REQUIRED_SHARED_SOURCES.issubset(paths) or len(paths) != len(set(paths)):
        errors.append("operating_required_shared_sources_missing_or_duplicate")
    return errors


def registry_projection(row: Mapping[str, Any]) -> dict[str, Any]:
    return {key: deepcopy(row[key]) for key in REGISTRY_FIELDS if key in row}


def _source_path(root: Path, bot_id: str, catalog_row: Mapping[str, Any]) -> str:
    exact = f"core/{bot_id}.py"
    if re.fullmatch(r"[A-Za-z0-9_]+", bot_id) and safe_project_file(root, exact):
        return exact
    raw = catalog_row.get("core_file")
    return raw if isinstance(raw, str) else ""


def _dependency_paths(root: Path, program: Mapping[str, Any]) -> list[str]:
    paths = []
    for name in program.get("imports", []):
        if not re.fullmatch(r"[A-Za-z0-9_.]+", name):
            continue
        raw = name.replace(".", "/") + ".py"
        for candidate in (raw, "core/" + raw):
            if safe_project_file(root, candidate):
                paths.append(candidate)
                break
    return sorted(set(paths))


def _observe(
    row: Mapping[str, Any], catalog_row: Mapping[str, Any], inventory: SourceInventory
) -> tuple[dict[str, Any], list[str]]:
    issues = []
    bot_id = row.get("bot_id")
    if not isinstance(bot_id, str) or not bot_id.strip():
        return {}, ["operating_bot_identity_invalid"]
    if not isinstance(row.get("bot_role"), str) or row["bot_role"] not in KNOWN_ROLES:
        issues.append("operating_registry_role_invalid")
    for key in ("label_contract", "training_label_materialization_contract"):
        if key in row and not isinstance(row[key], dict):
            issues.append(f"operating_registry_contract_invalid:{key}")
    if any(
        row.get(field) is not None and row.get(field) is not False
        for field in AUTHORITY_FIELDS
    ):
        issues.append("execution_authority_requires_separate_trading_mandate")
    path = _source_path(inventory.root, bot_id, catalog_row)
    source = inventory.read(path) if path else {}
    if path and source.get("status") != "read":
        issues.append(f"operating_source_unreadable:{path}")
    spec = source.get("literals", {}).get("BOT_SPEC", {})
    if not isinstance(spec, dict):
        spec = {}
        issues.append("operating_module_spec_invalid")
    if spec.get("bot_id") not in (None, bot_id):
        issues.append("operating_module_identity_conflict")
    if spec.get("bot_role") not in (None, row.get("bot_role")):
        issues.append("operating_module_role_conflict")
    program = source.get("program", {})
    if not path:
        profile = "registry_declared_slot"
    elif source.get("collection_wrapper"):
        profile = "collection_wrapper"
    elif program.get("runtime_training_path_present"):
        profile = "runtime_model_program"
    else:
        profile = "synthetic_research_program"
    if path and not source.get("functions"):
        issues.append("operating_no_program_entrypoints")
    if (
        path
        and profile == "synthetic_research_program"
        and not program.get("synthetic_training_path_present")
    ):
        issues.append("operating_program_kind_requires_review")
    if program.get("order_calls"):
        issues.append("operating_program_has_order_calls")
    dependencies = {}
    for dependency in _dependency_paths(inventory.root, program):
        observed = inventory.read(dependency)
        if observed.get("status") != "read":
            issues.append(f"operating_dependency_unreadable:{dependency}")
        else:
            dependencies[dependency] = observed["sha256"]
    return {
        "bot_id": bot_id,
        "profile_id": profile,
        "registry_definition": registry_projection(row),
        "catalog_binding": {
            "core_file": catalog_row.get("core_file", ""),
            "runner": catalog_row.get("runner", ""),
        },
        "source": {
            "path": path,
            "sha256": source.get("sha256"),
            "program": deepcopy(program),
        },
        "module_spec": deepcopy(spec),
        "dependency_hashes": dependencies,
    }, issues


def _shared_sources(
    policy: Mapping[str, Any], inventory: SourceInventory
) -> tuple[dict[str, str], list[str]]:
    hashes = {}
    errors = []
    for path in policy["shared_sources"]:
        source = inventory.read(path)
        if source.get("status") != "read":
            errors.append(f"operating_shared_source_unreadable:{path}")
        else:
            hashes[path] = source["sha256"]
    return hashes, errors


def _catalog_rows(catalog: Mapping[str, Any]) -> dict[str, list[dict[str, Any]]]:
    rows = {}
    for row in catalog.get("bots", []) if isinstance(catalog.get("bots"), list) else []:
        if isinstance(row, dict):
            rows.setdefault(str(row.get("bot_id", "")), []).append(row)
    return rows


def _entry(binding: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "revision": "operating-v1",
        "profile_id": binding["profile_id"],
        "source_path": binding["source"]["path"],
        "source_sha256": binding["source"]["sha256"],
        "registry_definition_sha256": digest(binding["registry_definition"]),
        "dependency_hashes": deepcopy(binding["dependency_hashes"]),
        "binding_sha256": digest(binding),
    }


def compile_catalog(
    registry: Mapping[str, Any],
    catalog: Mapping[str, Any],
    policy: Mapping[str, Any],
    root: Path,
) -> dict[str, Any]:
    """Explicit authoring only. Normal refresh calls validate_catalog, never this."""
    errors = validate_policy(policy)
    if errors:
        raise ValueError(";".join(errors))
    process_policy = load_process_policy(root)
    inventory = SourceInventory(
        root, policy["source_budget"], inspector=inspect_program
    )
    shared, errors = _shared_sources(policy, inventory)
    catalog_rows = _catalog_rows(catalog)
    entries = {}
    registry_rows = registry.get("sub_bots", [])
    if not isinstance(registry_rows, list):
        raise ValueError("operating_registry_rows_invalid")
    for row in registry_rows:
        if not isinstance(row, dict) or not isinstance(row.get("bot_id"), str):
            errors.append("operating_registry_row_invalid")
            continue
        bot_id = row["bot_id"]
        if bot_id in entries or len(catalog_rows.get(bot_id, [])) != 1:
            errors.append(f"operating_identity_ambiguous:{bot_id}")
            continue
        binding, issues = _observe(row, catalog_rows[bot_id][0], inventory)
        errors.extend(f"{bot_id}:{issue}" for issue in issues)
        if not binding:
            continue
        if not issues:
            expand_process_definition(binding, process_policy)
        entries[bot_id] = _entry(binding)
    if not entries:
        errors.append("operating_registry_empty")
    if errors:
        raise ValueError(";".join(errors))
    return {
        "schema_version": 1,
        "contract_id": policy["contract_id"],
        "completion_scope": SCOPE,
        "policy_sha256": digest(policy),
        "process_policy_sha256": digest(process_policy),
        "shared_source_hashes": shared,
        "entries": entries,
        "publication_mode": "explicit_source_bound_authoring_not_runtime_activation",
    }


def expand_definition(
    entry: Mapping[str, Any], policy: Mapping[str, Any]
) -> dict[str, Any]:
    binding = entry["binding"]
    profile = policy["profiles"][binding["profile_id"]]
    registry = binding["registry_definition"]
    source = binding["source"]
    program = source.get("program", {})
    spec = binding["module_spec"]
    metadata_job = binding["profile_id"] in {
        "collection_wrapper",
        "registry_declared_slot",
    }
    source_ref = (
        {
            "path": source["path"],
            "sha256": source["sha256"],
            "functions": list(program.get("functions", {})),
        }
        if source["path"]
        else {
            "path": "master_bot_registry.json",
            "selector": {"bot_id": binding["bot_id"]},
            "projection_sha256": digest(registry),
        }
    )
    event_clock = {
        "mode": "event_driven",
        "trigger": (
            "existing_registry_review"
            if metadata_job
            else "existing_trainer_invocation"
        ),
        "new_schedule_created": False,
    }
    position_owner = {
        action: "not_owned_by_this_component; existing institutional/execution-lane controllers retain ownership"
        for action in ("entry", "add", "trim", "exit", "time_stop")
    }
    parameters = {
        "training_calls": program.get("training_calls", []),
        "configuration_expressions": program.get("configuration_expressions", {}),
        "function_defaults": {
            name: value["defaults"]
            for name, value in program.get("functions", {}).items()
            if value["defaults"]
        },
        "interpretation": "exact source expressions; no execution or conversion of observations to seconds",
    }
    labels = registry.get("training_label_materialization_contract", {})
    values = {
        "purpose": {
            "role": registry["bot_role"],
            "hypothesis": profile["hypothesis"],
            "primary_family": binding["profile_id"],
            "objective": {
                "operating_job": profile["purpose"],
                "bot_id": binding["bot_id"],
                "declared_research_objective": spec.get("slot_objective")
                or registry.get("slot_objective"),
                "declared_label_family": registry.get("label_contract", {}).get(
                    "label_family"
                ),
            },
        },
        "scope": {
            "universe": {
                "registry_input_declarations": registry.get(
                    "data_intake_collections", spec.get("data_intake_collections", [])
                ),
                "source_selection": parameters,
                "scope": (
                    "metadata_record" if metadata_job else "source_defined_dataset"
                ),
            },
            "venue": {"owns_order_venue": False, "data_origin": profile["data_origin"]},
            "session": event_clock,
            "decision_interval_seconds": event_clock,
            "holding_horizon_seconds": {
                "position_holding": "not_owned",
                "prediction_horizon": parameters,
            },
            "inputs": {
                "literal_source_keys": program.get("literal_input_keys", []),
                "registry_input_contract": registry.get("label_contract", {}),
                "source": source_ref,
            },
        },
        "decision_rules": {
            "output_rule": {
                "source": source_ref,
                "training_status": profile["training_status"],
                "metadata_projection": (
                    registry if metadata_job else "not_the_model_rule"
                ),
            },
            "parameters": (
                parameters
                if not metadata_job
                else {"registry_definition": registry, "module_spec": spec}
            ),
            "position_management": position_owner,
            "test_cases": [
                {
                    "kind": "normal",
                    "input": "matching registry/source/policy binding",
                    "expected": "operating definition complete; no execution or economic credit",
                },
                {
                    "kind": "abstention",
                    "input": "changed identity/source/policy or execution authority",
                    "expected": "operating definition incomplete; retain separate evidence gaps",
                },
            ],
        },
        "abstention": {
            "conditions": {
                "definition_rejections": [
                    "missing_binding",
                    "source_changed",
                    "registry_definition_changed",
                    "profile_changed",
                    "policy_changed",
                    "identity_ambiguous",
                    "execution_authority_present",
                ],
                "source_filter_functions": [
                    name
                    for name in program.get("functions", {})
                    if "filter" in name or "confidence" in name
                ],
            },
            "maximum_input_age_seconds": {
                "definition_check": "each native audit checks exact hashes",
                "declared_freshness_slo_seconds": spec.get("freshness_slo_seconds"),
                "source_timing_and_filters": source_ref,
            },
            "supported_regimes": {
                "declarations": spec.get("preferred_regimes", []),
                "empirical_support_verified": False,
                "implemented_filters": source_ref,
            },
        },
        "boundaries": {
            "owner": {
                "definition_owner": "bot_organization_controller",
                "program_owner": source_ref,
            },
            "allowed_outputs": ["definition_and_audit_evidence"]
            + (
                ["metadata_description"]
                if metadata_job
                else ["source_defined_training_artifacts_when_separately_admitted"]
            ),
            "forbidden_actions": list(FALSE_AUTHORITY),
            "risk_owner": [
                "core/institutional_decision_flow.py",
                "core/execution_lane_pipeline.py",
            ],
            "dependencies": binding["dependency_hashes"],
        },
        "training": {
            "target": {
                "status": profile["training_status"],
                "source": source_ref,
                "declared_future_labels": labels,
            },
            "label_horizon_seconds": {
                "declaration": (
                    labels.get("label_horizon_policy", {})
                    if isinstance(labels, dict)
                    else {}
                ),
                "source_horizon_arguments": parameters,
                "seconds_inferred_from_samples": False,
            },
            "join_policy": {
                "registry_declaration": (
                    labels.get("required_join_mode")
                    if isinstance(labels, dict)
                    else None
                ),
                "source": source_ref,
                "correctness_verified": False,
            },
            "split_policy": {
                "source": source_ref,
                "declared_evaluation_policy": (
                    labels.get("evaluation_split_policy")
                    if isinstance(labels, dict)
                    else None
                ),
                "data_origin": profile["data_origin"],
            },
            "experiment_contract": {
                "bot_id": binding["bot_id"],
                "binding_sha256": entry["binding_sha256"],
                "candidate_dataset_model_and_holdout_evidence": "required_separately_by_existing_evidence_owners",
                "training_started": False,
            },
        },
        "accountability": {
            "metric": {
                "operating_metric": "all_32_fields_bound_and_source_identity_matches",
                "model_measurements": source_ref,
                "economic_measurements": "not_assessed",
            },
            "benchmark": "exact declared source/registry projection; not a trading-return benchmark",
            "invalidation": "changed code/registry definition/dependency/policy/role/authority requires explicit re-review",
            "trace_fields": [
                "bot_id",
                "profile_id",
                "definition_sha256",
                "binding_sha256",
                "source",
                "registry_definition",
                "all_seven_area_results",
                "issues",
                "economic_evidence",
                "standalone_trading_mandate",
            ],
            "lifecycle": {
                "draft": "unbound or new registry entry",
                "bound": "explicit materialization plus validation",
                "changed": "incomplete until reviewed and explicitly rebound",
                "retired": "keep registry lifecycle unchanged; removed registry identities become orphan definitions",
            },
        },
    }
    return {
        area: {
            field: {
                "subsection": {
                    "number": f"{area_index}.{field_index}",
                    "title": policy["subsections"][area][field],
                },
                "contract": policy["areas"][area][field],
                "value": values[area][field],
            }
            for field_index, field in enumerate(fields, 1)
        }
        for area_index, (area, fields) in enumerate(AREAS.items(), 1)
    }


def validate_catalog(
    registry: Mapping[str, Any],
    catalog: Mapping[str, Any],
    legacy: Mapping[str, Any],
    policy: Mapping[str, Any],
    manifest: Mapping[str, Any],
    root: Path,
) -> dict[str, Any]:
    policy_errors = validate_policy(policy)
    process_policy = {}
    try:
        process_policy = load_process_policy(root)
    except ValueError as exc:
        policy_errors.append(str(exc))
    if process_policy and manifest.get("process_policy_sha256") != digest(
        process_policy
    ):
        policy_errors.append("operating_process_policy_changed")
    if not manifest:
        policy_errors.append("operating_definitions_not_materialized")
    elif (
        manifest.get("policy_sha256") != digest(policy)
        or manifest.get("completion_scope") != SCOPE
        or manifest.get("schema_version") != 1
        or manifest.get("contract_id") != "bot_operating_definitions_v1"
        or manifest.get("publication_mode")
        != "explicit_source_bound_authoring_not_runtime_activation"
    ):
        policy_errors.append("operating_catalog_policy_mismatch")
    entries = manifest.get("entries", {})
    if not isinstance(entries, dict):
        entries = {}
        policy_errors.append("operating_entries_invalid")
    inventory = (
        None
        if validate_policy(policy)
        else SourceInventory(root, policy["source_budget"], inspector=inspect_program)
    )
    shared_contracts = {}
    if inventory is not None:
        shared, errors = _shared_sources(policy, inventory)
        policy_errors.extend(errors)
        if manifest and shared != manifest.get("shared_source_hashes"):
            policy_errors.append("operating_shared_sources_changed")
        shared_contracts = {
            path: {"sha256": sha, "program": inventory.cache[path].get("program", {})}
            for path, sha in shared.items()
        }
    rows = registry.get("sub_bots", [])
    rows = rows if isinstance(rows, list) else []
    ids = [str(row.get("bot_id", "")) for row in rows if isinstance(row, dict)]
    duplicates = {bot_id for bot_id, count in Counter(ids).items() if count > 1}
    orphan_ids = sorted(set(entries) - set(ids))
    if orphan_ids:
        policy_errors.append("operating_orphan_definitions")
    catalog_rows = _catalog_rows(catalog)
    old_by_index = {
        record["registry_index"]: record for record in legacy.get("records", [])
    }
    records = []
    for index, raw in enumerate(rows):
        row = raw if isinstance(raw, dict) else {}
        bot_id = str(row.get("bot_id", ""))
        issues = list(policy_errors)
        if not bot_id or bot_id in duplicates or len(catalog_rows.get(bot_id, [])) != 1:
            issues.append("operating_identity_missing_or_ambiguous")
        entry = entries.get(bot_id, {})
        if not isinstance(entry, dict) or not entry:
            entry = {}
            issues.append("operating_bot_definition_missing")
        observed = {}
        if inventory is not None and len(catalog_rows.get(bot_id, [])) == 1:
            observed, observed_issues = _observe(
                row, catalog_rows[bot_id][0], inventory
            )
            issues.extend(observed_issues)
        if entry:
            if not observed or entry != _entry(observed):
                issues.append("operating_binding_changed")
        areas = {}
        processes = {}
        if entry and not issues:
            areas = expand_definition(
                {"binding": observed, "binding_sha256": entry["binding_sha256"]}, policy
            )
            if set(areas) != set(AREAS) or any(
                set(areas[area]) != set(fields) for area, fields in AREAS.items()
            ):
                issues.append("operating_area_coverage_invalid")
            try:
                processes = expand_process_definition(observed, process_policy)
            except ValueError as exc:
                issues.append(str(exc))
        complete = bool(entry and not issues and areas)
        records.append(
            {
                "registry_index": index,
                "bot_id": bot_id,
                "definition_complete": complete,
                "definition_status": "complete" if complete else "incomplete",
                "completion_scope": SCOPE,
                "profile_id": observed.get("profile_id"),
                "definition_sha256": (
                    digest(
                        {
                            "policy": manifest.get("policy_sha256"),
                            "process_policy": manifest.get("process_policy_sha256"),
                            "entry": entry,
                            "shared_sources": manifest.get("shared_source_hashes"),
                        }
                    )
                    if entry
                    else None
                ),
                "binding_sha256": entry.get("binding_sha256"),
                "binding": observed,
                "process_definition": {
                    "definition_valid": bool(complete and processes),
                    "definition_sha256": processes.get("definition_sha256"),
                    "implementation_kind": processes.get("implementation_kind"),
                    "stage_count": len(processes.get("stages", [])),
                    "source_referenced_stage_count": sum(
                        bool(stage["source_references"])
                        for stage in processes.get("stages", [])
                    ),
                    "source_reference_gaps": [
                        stage["id"]
                        for stage in processes.get("stages", [])
                        if stage["reference_gap"]
                    ],
                    "runtime_verified": False,
                },
                "issues": sorted(set(issues)),
                "areas": {
                    area: {
                        "complete": complete,
                        "required_fields": len(fields),
                        "defined_fields": len(fields) if complete else 0,
                        "subsections": {
                            field: {"complete": complete} for field in fields
                        },
                    }
                    for area, fields in AREAS.items()
                },
                "standalone_trading_mandate": old_by_index.get(index, {}),
                "economic_evidence": {
                    "status": "not_assessed_by_definition_audit",
                    "definition_grants_evidence": False,
                },
                "implementation_verification": {
                    "runtime_conformance_verified": False,
                    "status": "not_verified_by_definition_completeness",
                },
            }
        )
    complete_count = sum(row["definition_complete"] for row in records)
    total = len(rows)
    subsection_titles = policy["subsections"] if inventory is not None else {}
    return {
        "contract_id": "bot_operating_definitions_v1",
        "completion_scope": SCOPE,
        "definition_complete": bool(
            total and complete_count == total and not policy_errors
        ),
        "status": (
            "complete"
            if total and complete_count == total and not policy_errors
            else "incomplete"
        ),
        "registry_record_count": total,
        "audited_record_count": len(records),
        "audit_coverage_ratio": len(records) / total if total else 0.0,
        "definition_complete_count": complete_count,
        "definition_incomplete_count": total - complete_count,
        "definition_completeness_ratio": complete_count / total if total else 0.0,
        "errors": sorted(set(policy_errors)),
        "orphan_definition_ids": orphan_ids,
        "area_summary": {
            area: {
                "complete_count": complete_count,
                "defined_field_count": complete_count * len(fields),
                "required_field_count": total * len(fields),
                "subsections": {
                    field: {
                        "number": f"{area_index}.{field_index}",
                        "title": subsection_titles.get(area, {}).get(field),
                        "complete_count": complete_count,
                        "required_count": total,
                    }
                    for field_index, field in enumerate(fields, 1)
                },
            }
            for area_index, (area, fields) in enumerate(AREAS.items(), 1)
        },
        "kind_counts": dict(
            Counter(row["profile_id"] for row in records if row["profile_id"])
        ),
        "process_contract": process_policy,
        "process_summary": {
            "defined_bot_count": sum(
                row["process_definition"]["definition_valid"] for row in records
            ),
            "stage_count": sum(
                row["process_definition"]["stage_count"] for row in records
            ),
            "source_referenced_stage_count": sum(
                row["process_definition"]["source_referenced_stage_count"]
                for row in records
            ),
            "source_reference_gap_count": sum(
                len(row["process_definition"]["source_reference_gaps"])
                for row in records
            ),
            "runtime_verified_bot_count": 0,
            "runtime_completion_claimed": False,
        },
        "issue_counts": dict(
            Counter(issue for row in records for issue in row["issues"])
        ),
        "gap_counts": {},
        "role_contracts": deepcopy(policy),
        "shared_source_contracts": shared_contracts,
        "standalone_trading_mandate_summary": {
            key: deepcopy(value) for key, value in legacy.items() if key != "records"
        },
        "economic_evidence": deepcopy(
            legacy.get(
                "economic_evidence", {"status": "not_assessed_by_definition_audit"}
            )
        ),
        "authority": {key: False for key in FALSE_AUTHORITY},
        "source_inventory": {
            "files_read": inventory.files_read if inventory else 0,
            "bytes_read": inventory.bytes_read if inventory else 0,
        },
        "audit_sha256": digest(
            {
                "manifest": manifest,
                "record_states": [
                    (row["bot_id"], row["definition_sha256"], row["issues"])
                    for row in records
                ],
                "errors": policy_errors,
            }
        ),
        "records": records,
    }
