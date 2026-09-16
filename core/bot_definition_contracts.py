"""Bounded, non-executing inventory and seven-area bot definition audit."""

from __future__ import annotations

import ast
from collections import Counter, defaultdict
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Mapping

AREAS = {
    "purpose": ("role", "hypothesis", "primary_family", "objective"),
    "scope": (
        "universe",
        "venue",
        "session",
        "decision_interval_seconds",
        "holding_horizon_seconds",
        "inputs",
    ),
    "decision_rules": (
        "output_rule",
        "parameters",
        "position_management",
        "test_cases",
    ),
    "abstention": ("conditions", "maximum_input_age_seconds", "supported_regimes"),
    "boundaries": (
        "owner",
        "allowed_outputs",
        "forbidden_actions",
        "risk_owner",
        "dependencies",
    ),
    "training": (
        "target",
        "label_horizon_seconds",
        "join_policy",
        "split_policy",
        "experiment_contract",
    ),
    "accountability": (
        "metric",
        "benchmark",
        "invalidation",
        "trace_fields",
        "lifecycle",
    ),
}
FALSE_AUTHORITY = (
    "changes_decisions",
    "changes_registry",
    "changes_risk_limits",
    "starts_workers",
    "submits_orders",
    "promotes_candidates",
    "claims_economic_evidence",
)
FALSE_COMPLETION = (
    "inferred_classification_is_explicit_definition",
    "generic_placeholder_is_complete",
    "collection_wrapper_is_strategy_implementation",
    "missing_source_is_complete",
    "definition_implies_runtime_conformance",
    "definition_implies_economic_evidence",
    "unknown_economics_is_failure",
)
PLACEHOLDERS = {
    "unknown",
    "none",
    "null",
    "tbd",
    "todo",
    "sleeve_specific",
    "strategy_horizon_specific",
    "contract_supported",
    "sleeve_manifest_universe",
    "generic_directional",
}
NON_MARKET_ROLES = {"infrastructure_sub_bot", "infrastructure_bot"}
KNOWN_ROLES = NON_MARKET_ROLES | {
    "signal_sub_bot",
    "options_sub_bot",
    "futures_sub_bot",
    "macro_sub_bot",
    "crypto_sub_bot",
}
TEXT_FIELDS = {
    "role",
    "hypothesis",
    "primary_family",
    "objective",
    "venue",
    "session",
    "owner",
    "risk_owner",
    "join_policy",
    "split_policy",
    "metric",
    "benchmark",
    "invalidation",
}
NUMERIC_FIELDS = {
    "decision_interval_seconds",
    "holding_horizon_seconds",
    "maximum_input_age_seconds",
    "label_horizon_seconds",
}
LIST_FIELDS = {
    "universe",
    "inputs",
    "conditions",
    "supported_regimes",
    "allowed_outputs",
    "forbidden_actions",
    "dependencies",
    "trace_fields",
}


def digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def _finite_json(value: Any) -> bool:
    try:
        json.dumps(value, allow_nan=False, sort_keys=True)
    except (ValueError, TypeError):
        return False
    return True


def _map(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _meaningful(value: Any) -> bool:
    if isinstance(value, str):
        return bool(value.strip()) and value.strip().lower() not in PLACEHOLDERS
    if isinstance(value, bool) or value is None:
        return False
    if isinstance(value, (int, float)):
        return math.isfinite(value)
    if isinstance(value, (dict, list)):
        return bool(value) and all(
            _meaningful(item)
            for item in (value.values() if isinstance(value, dict) else value)
        )
    return False


def validate_definition_policy(contract: Mapping[str, Any]) -> list[str]:
    errors = []
    if contract.get("contract_id") != "bot_definition_seven_areas_v1":
        errors.append("definition_contract_id_invalid")
    if contract.get("mode") != "definition_audit_only":
        errors.append("definition_audit_mode_invalid")
    authority = _map(contract.get("authority"))
    if set(authority) != set(FALSE_AUTHORITY) or any(
        authority.get(key) is not False for key in FALSE_AUTHORITY
    ):
        errors.append("definition_audit_authority_invalid")
    completion = _map(contract.get("completion_policy"))
    if completion.get("all_seven_areas_required") is not True or any(
        completion.get(key) is not False for key in FALSE_COMPLETION
    ):
        errors.append("definition_completion_policy_invalid")
    if contract.get("areas") != list(AREAS):
        errors.append("definition_seven_areas_required")
    budget = _map(contract.get("source_budget"))
    for key, ceiling in (
        ("maximum_module_bytes", 2_000_000),
        ("maximum_total_bytes", 64_000_000),
        ("maximum_files", 2500),
    ):
        if type(budget.get(key)) is not int or not 1 <= budget[key] <= ceiling:
            errors.append(f"definition_{key}_invalid")
    if not isinstance(contract.get("bot_definitions"), dict):
        errors.append("definition_bot_definitions_invalid")
    for key in (
        "owner",
        "source_resolution",
        "decision_rules",
        "training",
        "economic_acceptance",
        "duplicates",
        "refresh",
    ):
        if not _meaningful(_map(contract.get("review_policy")).get(key)):
            errors.append(f"definition_review_{key}_missing")
    if _map(contract.get("review_policy")).get("owner") != "bot_organization_control":
        errors.append("definition_review_owner_invalid")
    for key in ("economics", "candidate_evidence", "training_lineage"):
        if not _meaningful(_map(contract.get("evidence_owners")).get(key)):
            errors.append(f"definition_evidence_owner_{key}_missing")
    required_trace = {
        "bot_id",
        "definition_sha256",
        "candidate_id",
        "experiment_trial_id",
        "feature_snapshot_id",
        "input_timestamp_utc",
        "decision_timestamp_utc",
        "rule_results",
        "abstention_reason",
        "output",
        "outcome_join_id",
    }
    trace = contract.get("trace_required_fields")
    if (
        not isinstance(trace, list)
        or not all(isinstance(item, str) for item in trace)
        or set(trace) != required_trace
    ):
        errors.append("definition_trace_contract_invalid")
    return errors


def safe_project_file(root: Path, raw: Any) -> Path | None:
    """Reject external/symlink routes before probing their targets, including VIDEO."""
    if (
        not isinstance(raw, str)
        or not raw
        or Path(raw).is_absolute()
        or ".." in Path(raw).parts
    ):
        return None
    path = root
    try:
        for part in Path(raw).parts:
            path = path / part
            if path.is_symlink():
                return None
        return path if path.is_file() else None
    except OSError:
        return None


class SourceInventory:
    def __init__(self, root: Path, budget: Mapping[str, Any], *, inspector=None):
        self.root = root
        self.budget = budget
        self.inspector = inspector
        self.cache: dict[str, dict[str, Any]] = {}
        self.bytes_read = 0
        self.files_read = 0

    def read(self, raw: Any) -> dict[str, Any]:
        if not isinstance(raw, str):
            return {"status": "source_path_invalid"}
        if raw in self.cache:
            return self.cache[raw]
        result: dict[str, Any] = {"path": raw, "status": "source_missing_or_disallowed"}
        self.cache[raw] = result
        path = safe_project_file(self.root, raw)
        if path is None:
            return result
        if self.files_read >= self.budget["maximum_files"]:
            result["status"] = "source_budget_exhausted"
            return result
        remaining = self.budget["maximum_total_bytes"] - self.bytes_read
        limit = min(self.budget["maximum_module_bytes"], remaining)
        if limit <= 0:
            result["status"] = "source_budget_exhausted"
            return result
        try:
            before = path.stat()
            if before.st_size > limit:
                result["status"] = "source_size_limit"
                return result
            with path.open("rb") as handle:
                raw_bytes = handle.read(limit + 1)
            self.files_read += 1
            self.bytes_read += len(raw_bytes)
            after = path.stat()
            if len(raw_bytes) > limit or (
                before.st_ino,
                before.st_size,
                before.st_mtime_ns,
            ) != (after.st_ino, after.st_size, after.st_mtime_ns):
                result["status"] = "source_changed_during_read"
                return result
            result["sha256"] = hashlib.sha256(raw_bytes).hexdigest()
            if path.suffix == ".json":
                result.update(status="read", document=json.loads(raw_bytes))
                return result
            if path.suffix != ".py":
                result["status"] = "source_type_unsupported"
                return result
            tree = ast.parse(raw_bytes.decode("utf-8"), filename=raw)
        except (OSError, ValueError, SyntaxError, UnicodeError, RecursionError):
            result["status"] = "source_parse_failed"
            return result
        functions: dict[str, Any] = {}
        literals: dict[str, Any] = {}
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                functions[node.name] = {
                    "line": node.lineno,
                    "end_line": node.end_lineno,
                }
                if isinstance(node, ast.ClassDef):
                    for child in node.body:
                        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                            functions[f"{node.name}.{child.name}"] = {
                                "line": child.lineno,
                                "end_line": child.end_lineno,
                            }
            if isinstance(node, (ast.Assign, ast.AnnAssign)):
                targets = (
                    node.targets if isinstance(node, ast.Assign) else [node.target]
                )
                for target in targets:
                    if isinstance(target, ast.Name):
                        try:
                            literals[target.id] = ast.literal_eval(node.value)
                        except (ValueError, TypeError, SyntaxError, RecursionError):
                            pass
        calls = {
            node.func.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        }
        wrapper = (
            "train_registry_backed_bot" in calls
            or "describe_registry_backed_bot" in calls
        )
        result.update(
            status="read",
            functions=functions,
            literals=literals,
            collection_wrapper=wrapper,
        )
        if self.inspector is not None:
            result["program"] = self.inspector(tree)
        return result

    def reference_error(self, reference: Any) -> str:
        if not isinstance(reference, dict) or set(reference) != {
            "path",
            "sha256",
            "symbol",
        }:
            return "source_reference_invalid"
        source = self.read(reference.get("path"))
        if source.get("status") != "read":
            return str(source.get("status"))
        if reference.get("sha256") != source.get("sha256"):
            return "source_hash_mismatch"
        symbol = reference.get("symbol")
        if not isinstance(symbol, str) or not symbol:
            return "source_symbol_missing"
        if "document" in source:
            if not symbol.startswith("/"):
                return "source_json_pointer_invalid"
            value = source["document"]
            try:
                for part in symbol[1:].split("/"):
                    key = part.replace("~1", "/").replace("~0", "~")
                    if isinstance(value, list):
                        if not re.fullmatch(r"0|[1-9][0-9]*", key):
                            return "source_json_pointer_invalid"
                        value = value[int(key)]
                    else:
                        value = value[key]
            except (KeyError, IndexError, TypeError, ValueError):
                return "source_symbol_missing"
        elif symbol not in source.get("functions", {}) and symbol not in source.get(
            "literals", {}
        ):
            return "source_symbol_missing"
        return ""


def _field(value: Any, source: str, *, inferred: bool = False) -> dict[str, Any]:
    if not _finite_json(value):
        return {
            "value": None,
            "source": source,
            "inferred": inferred,
            "source_errors": ["field_value_not_finite_json"],
        }
    return {"value": deepcopy(value), "source": source, "inferred": inferred}


def _recover_fields(
    row: Mapping[str, Any], spec: Mapping[str, Any], assignment: Mapping[str, Any]
) -> dict[str, dict[str, Any]]:
    areas: dict[str, dict[str, Any]] = {key: {} for key in AREAS}

    def take(area: str, field: str, names: tuple[str, ...]) -> None:
        for name in names:
            for source, data in (("registry", row), ("module.BOT_SPEC", spec)):
                if _meaningful(data.get(name)):
                    areas[area][field] = _field(data[name], f"{source}.{name}")
                    return

    take("purpose", "role", ("bot_role",))
    take("purpose", "hypothesis", ("hypothesis", "economic_thesis"))
    take("purpose", "primary_family", ("strategy_family",))
    take("purpose", "objective", ("objective_class",))
    for field, names in {
        "universe": ("symbols", "universe"),
        "venue": ("venue", "broker"),
        "session": ("trading_session",),
        "decision_interval_seconds": ("decision_interval_seconds",),
        "holding_horizon_seconds": ("holding_horizon_seconds",),
        "inputs": ("data_intake_collections", "required_inputs"),
    }.items():
        take("scope", field, names)
    take("abstention", "maximum_input_age_seconds", ("freshness_slo_seconds",))
    take("abstention", "supported_regimes", ("preferred_regimes",))
    take("abstention", "conditions", ("abstention_conditions",))
    for field, names in {
        "owner": ("owner",),
        "allowed_outputs": ("storage_targets", "allowed_outputs"),
        "forbidden_actions": ("forbidden_actions",),
        "risk_owner": ("risk_owner",),
        "dependencies": ("target_functions",),
    }.items():
        take("boundaries", field, names)
    label = _map(row.get("training_label_materialization_contract"))
    declared_label = _map(row.get("label_contract"))
    for field, key in (
        ("target", "required_outputs"),
        ("label_horizon_seconds", "minimum_label_maturity_seconds"),
        ("join_policy", "required_join_mode"),
        ("split_policy", "evaluation_split_policy"),
    ):
        if _meaningful(label.get(key)):
            areas["training"][field] = _field(
                label[key],
                f"registry.training_label_materialization_contract.{key}",
                inferred="inferred" in str(declared_label.get("source", "")),
            )
    for field in AREAS["accountability"]:
        take("accountability", field, (field,))
    # Classifications are useful repair hints but are not authored bot mandates.
    if "primary_family" not in areas["purpose"] and assignment.get("sub_sleeve_id"):
        areas["purpose"]["primary_family"] = _field(
            assignment["sub_sleeve_id"], "organization.classification", inferred=True
        )
    return areas


def _value_error(field: str, value: Any, non_market: bool, trace: list[str]) -> str:
    if isinstance(value, dict) and "not_applicable" in value:
        if (
            non_market
            and field
            in {
                "primary_family",
                "universe",
                "venue",
                "session",
                "holding_horizon_seconds",
                "position_management",
                "risk_owner",
                "benchmark",
                "target",
                "label_horizon_seconds",
                "join_policy",
                "split_policy",
                "experiment_contract",
            }
            and set(value) == {"not_applicable", "reason"}
            and value["not_applicable"] is True
            and _meaningful(value.get("reason"))
        ):
            return ""
        return "not_applicable_not_permitted"
    if field == "parameters":
        if (
            not isinstance(value, dict)
            or not isinstance(value.get("values"), dict)
            or not _meaningful(value.get("units"))
            or not _finite_json(value)
        ):
            return "parameter_values_and_units_required"
        if not value["values"] and value.get("no_tunable_parameters") is not True:
            return "explicit_fixed_parameter_contract_required"
        return ""
    if field == "test_cases":
        if (
            not isinstance(value, list)
            or not _finite_json(value)
            or not all(
                isinstance(item, dict)
                and isinstance(item.get("inputs"), dict)
                and "expected" in item
                and isinstance(item.get("case_id"), str)
                and _meaningful(item.get("case_id"))
                for item in value
            )
        ):
            return "golden_normal_and_abstention_cases_required"
        if not {"normal", "abstention"}.issubset(
            {str(item.get("kind")) for item in value}
        ):
            return "golden_normal_and_abstention_cases_required"
        if len({item["case_id"] for item in value}) != len(value):
            return "duplicate_test_case_identity"
        return ""
    if not _meaningful(value):
        return "missing_or_generic"
    if field in TEXT_FIELDS and not isinstance(value, str):
        return "explicit_text_required"
    if field in NUMERIC_FIELDS and (
        isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0
    ):
        return "positive_seconds_required"
    if field in LIST_FIELDS and (
        not isinstance(value, list)
        or not all(isinstance(item, str) and _meaningful(item) for item in value)
    ):
        return "explicit_list_required"
    if field == "primary_family" and not isinstance(value, str):
        return "one_primary_family_required"
    if field == "output_rule" and (
        not isinstance(value, dict)
        or not _meaningful(value.get("formula"))
        or not _meaningful(value.get("output_schema"))
    ):
        return "formula_and_output_schema_required"
    if field == "position_management" and (
        not isinstance(value, dict)
        or any(
            not _meaningful(value.get(key))
            for key in ("entry", "add", "trim", "exit", "time_stop")
        )
    ):
        return "position_rules_or_exact_delegation_required"
    if field == "trace_fields" and not set(trace).issubset(set(value)):
        return "required_trace_fields_missing"
    if field == "experiment_contract" and (
        not isinstance(value, dict)
        or any(
            not _meaningful(value.get(key))
            for key in (
                "candidate_binding",
                "trial_accounting",
                "dataset_version",
                "untouched_holdout",
                "embargo",
            )
        )
    ):
        return "experiment_lineage_and_holdout_required"
    if field == "lifecycle" and (
        not isinstance(value, dict)
        or any(
            not _meaningful(value.get(key))
            for key in ("collect", "evaluate", "invalidate", "retire")
        )
    ):
        return "explicit_lifecycle_transitions_required"
    return ""


def _overlay_fields(
    areas: dict[str, Any], definition: Mapping[str, Any], inventory: SourceInventory
) -> list[str]:
    errors = []
    for area, fields in _map(definition.get("areas")).items():
        if area not in AREAS or not isinstance(fields, dict):
            errors.append(f"invalid_area:{area}")
            continue
        for field, record in fields.items():
            if (
                field not in AREAS[area]
                or not isinstance(record, dict)
                or set(record) != {"value", "references"}
            ):
                errors.append(f"invalid_field:{area}.{field}")
                continue
            references = record.get("references")
            reference_errors = []
            if not isinstance(references, list) or not references:
                reference_errors.append("source_reference_required")
            else:
                reference_errors = [
                    error
                    for ref in references
                    if (error := inventory.reference_error(ref))
                ]
            value = record.get("value")
            try:
                json.dumps(value, allow_nan=False, sort_keys=True)
            except (ValueError, TypeError):
                value = None
                reference_errors.append("definition_value_not_finite_json")
            areas[area][field] = {
                "value": deepcopy(value),
                "source": "reviewed_definition",
                "inferred": False,
                "references": deepcopy(references),
                "source_errors": reference_errors,
            }
    return errors


def audit_definitions(
    registry: Mapping[str, Any],
    catalog: Mapping[str, Any],
    assignments: list[dict[str, Any]],
    contract: Mapping[str, Any],
    project_root: Path,
) -> dict[str, Any]:
    errors = validate_definition_policy(contract)
    if errors:
        return {
            "status": "invalid_contract",
            "errors": errors,
            "audit_coverage_ratio": 0.0,
            "definition_complete": False,
            "records": [],
            "economic_evidence": {"status": "not_assessed_by_definition_audit"},
        }
    inventory = SourceInventory(project_root, contract["source_budget"])
    raw_rows = registry.get("sub_bots")
    rows = raw_rows if isinstance(raw_rows, list) else []
    if not isinstance(raw_rows, list) or not rows:
        errors.append("registry_empty_or_invalid")
    by_id = defaultdict(list)
    for item in (
        catalog.get("bots", []) if isinstance(catalog.get("bots"), list) else []
    ):
        if isinstance(item, dict):
            by_id[str(item.get("bot_id", ""))].append(item)
    assignment_by_id = {item["bot_id"]: item for item in assignments}
    counts = Counter(
        str(row.get("bot_id", "")) for row in rows if isinstance(row, dict)
    )
    definitions = contract["bot_definitions"]
    contract_hash = (
        digest(contract)
        if all(_finite_json(value) for value in definitions.values())
        else "invalid_nonfinite_definition"
    )
    for bot_id in definitions:
        if bot_id not in counts:
            errors.append(f"orphan_definition:{bot_id}")
    records = []
    missing_counts: Counter[str] = Counter()
    duplicate_mandates: dict[str, list[str]] = defaultdict(list)
    source_reuse: dict[str, list[str]] = defaultdict(list)
    for index, raw_row in enumerate(rows):
        row = _map(raw_row)
        bot_id = str(row.get("bot_id") or "")
        issues = []
        if not isinstance(raw_row, dict) or not bot_id:
            issues.append("registry_row_or_identity_invalid")
        if counts.get(bot_id, 0) > 1:
            issues.append("duplicate_registry_identity")
        catalog_rows = by_id[bot_id]
        if len(catalog_rows) != 1:
            issues.append("catalog_binding_missing_or_ambiguous")
        catalog_row = catalog_rows[0] if len(catalog_rows) == 1 else {}
        assignment = assignment_by_id.get(bot_id, {})
        if not assignment:
            issues.append("organization_assignment_missing")
        safe_id = bool(re.fullmatch(r"[A-Za-z0-9_]+", bot_id))
        exact = f"core/{bot_id}.py" if safe_id else ""
        raw_path = (
            exact
            if exact and safe_project_file(project_root, exact)
            else catalog_row.get("core_file", "")
        )
        source = inventory.read(raw_path)
        runner_sources = []
        # The existing catalog stores multiple runner paths in a comma-separated field.
        for runner in str(catalog_row.get("runner") or "").split(","):
            if runner.strip():
                runner_source = inventory.read(runner.strip())
                runner_sources.append(
                    {
                        key: runner_source[key]
                        for key in ("path", "status", "sha256")
                        if key in runner_source
                    }
                )
        if source.get("status") != "read":
            issues.append(str(source.get("status")))
        spec = _map(source.get("literals", {}).get("BOT_SPEC"))
        if spec.get("bot_id") and spec["bot_id"] != bot_id:
            issues.append("module_identity_conflict")
        if (
            spec.get("bot_role")
            and row.get("bot_role")
            and spec["bot_role"] != row["bot_role"]
        ):
            issues.append("module_role_conflict")
        registry_role = row.get("bot_role")
        if not isinstance(registry_role, str) or registry_role not in KNOWN_ROLES:
            issues.append("registry_role_unknown_or_invalid")
        non_market = (
            isinstance(registry_role, str) and registry_role in NON_MARKET_ROLES
        )
        kind = "non_market_service" if non_market else "market_candidate"
        if source.get("collection_wrapper"):
            kind = "collection_wrapper"
        elif source.get("status") != "read":
            kind = "registry_only_or_unresolved"
        areas = _recover_fields(row, spec, assignment)
        definition = _map(definitions.get(bot_id))
        if bot_id in definitions and not definition:
            issues.append("definition_shape_invalid")
        if definition:
            if (
                set(definition) != {"revision", "areas"}
                or not isinstance(definition.get("revision"), str)
                or not _meaningful(definition.get("revision"))
            ):
                issues.append("definition_revision_or_shape_invalid")
            issues.extend(_overlay_fields(areas, definition, inventory))
        role = _map(areas["purpose"].get("role")).get("value")
        if role != row.get("bot_role"):
            issues.append("definition_registry_role_conflict")
        results = {}
        for area, fields in AREAS.items():
            gaps = []
            for field in fields:
                record = _map(areas[area].get(field))
                problem = _value_error(
                    field,
                    record.get("value"),
                    non_market,
                    contract["trace_required_fields"],
                )
                if record.get("inferred"):
                    problem = "inferred_not_authored"
                if record.get("source_errors"):
                    problem = ",".join(record["source_errors"])
                # Precise runtime rules and experiment provenance require reviewed source bindings.
                if (
                    not problem
                    and (
                        area == "decision_rules"
                        or field
                        in {
                            "experiment_contract",
                            "conditions",
                            "invalidation",
                            "lifecycle",
                        }
                    )
                    and record.get("source") != "reviewed_definition"
                ):
                    problem = "reviewed_source_binding_required"
                if problem:
                    gaps.append({"field": field, "reason": problem})
                    missing_counts[f"{area}.{field}:{problem}"] += 1
            results[area] = {
                "complete": not gaps,
                "defined_fields": len(fields) - len(gaps),
                "required_fields": len(fields),
                "gaps": gaps,
                "fields": areas[area],
            }
        if source.get("collection_wrapper") and not non_market:
            issues.append("collection_wrapper_not_implemented_strategy")
        complete = not issues and all(result["complete"] for result in results.values())
        definition_material = {
            "contract_id": contract["contract_id"],
            "contract_sha256": contract_hash,
            "bot_id": bot_id,
            "revision": definition.get("revision"),
            "source_sha256": source.get("sha256"),
            "areas": areas,
        }
        record = {
            "registry_index": index,
            "bot_id": bot_id,
            "active": row.get("active") is True,
            "kind": kind,
            "definition_complete": complete,
            "definition_status": "complete" if complete else "incomplete",
            "definition_sha256": digest(definition_material),
            "source": {
                key: source[key]
                for key in (
                    "path",
                    "status",
                    "sha256",
                    "functions",
                    "collection_wrapper",
                )
                if key in source
            },
            "runner_sources": runner_sources,
            "runner_binding_verified": False,
            "declared_task_description": row.get("slot_objective")
            or spec.get("slot_objective")
            or "",
            "classification": {
                key: assignment.get(key)
                for key in (
                    "sleeve_id",
                    "sub_sleeve_id",
                    "horizon_id",
                    "role_id",
                    "provenance",
                )
            },
            "areas": results,
            "issues": sorted(set(issues)),
            "implementation_verification": {
                "status": "not_verified_by_static_definition_audit",
                "runtime_conformance_verified": False,
            },
            "economic_evidence": {
                "status": (
                    "not_applicable_operational_role"
                    if non_market
                    else "not_assessed_by_definition_audit"
                ),
                "definition_grants_evidence": False,
            },
            "repair": {
                "owner": contract["review_policy"]["owner"],
                "priority": 1 if row.get("active") is True else 2,
                "source_path": source.get("path", ""),
                "areas_remaining": [
                    area for area in AREAS if not results[area]["complete"]
                ],
                "requires_strategy_implementation": "collection_wrapper_not_implemented_strategy"
                in issues,
            },
        }
        if source.get("sha256"):
            source_reuse[source["sha256"]].append(bot_id)
        if complete:
            normalized = {
                area: {field: item["value"] for field, item in values.items()}
                for area, values in areas.items()
            }
            duplicate_mandates[digest(normalized)].append(bot_id)
        records.append(record)
    total = len(rows)
    completed = sum(record["definition_complete"] for record in records)
    audit_hash = digest(
        {
            "contract_sha256": contract_hash,
            "errors": errors,
            "records": [
                {
                    "registry_index": record["registry_index"],
                    "definition_sha256": record["definition_sha256"],
                    "issues": record["issues"],
                    "runner_sources": record["runner_sources"],
                    "gaps": {
                        area: row["gaps"] for area, row in record["areas"].items()
                    },
                }
                for record in records
            ],
        }
    )
    return {
        "contract_id": contract["contract_id"],
        "audit_sha256": audit_hash,
        "contract_sha256": contract_hash,
        "status": (
            "complete" if total and completed == total and not errors else "incomplete"
        ),
        "errors": errors,
        "registry_record_count": total,
        "audited_record_count": len(records),
        "audit_coverage_ratio": len(records) / total if total else 0.0,
        "definition_complete": bool(total and completed == total and not errors),
        "definition_complete_count": completed,
        "definition_incomplete_count": total - completed,
        "definition_completeness_ratio": completed / total if total else 0.0,
        "required_fields_by_area": {
            area: list(fields) for area, fields in AREAS.items()
        },
        "active_definition_incomplete_count": sum(
            record["active"] and not record["definition_complete"] for record in records
        ),
        "area_summary": {
            area: {
                "complete_count": sum(
                    record["areas"][area]["complete"] for record in records
                ),
                "defined_field_count": sum(
                    record["areas"][area]["defined_fields"] for record in records
                ),
                "required_field_count": total * len(fields),
            }
            for area, fields in AREAS.items()
        },
        "kind_counts": dict(Counter(record["kind"] for record in records)),
        "issue_counts": dict(
            Counter(issue for record in records for issue in record["issues"])
        ),
        "gap_counts": dict(sorted(missing_counts.items())),
        "identical_complete_mandates": [
            ids for _, ids in sorted(duplicate_mandates.items()) if len(ids) > 1
        ],
        "shared_source_groups": [
            ids for _, ids in sorted(source_reuse.items()) if len(ids) > 1
        ],
        "duplicate_checks": {
            "source_reuse_is_economic_correlation": False,
            "missing_mandates_are_duplicates": False,
            "automatic_merge_or_retirement": False,
        },
        "source_inventory": {
            "files_read": inventory.files_read,
            "bytes_read": inventory.bytes_read,
            "status_counts": dict(
                Counter(source["status"] for source in inventory.cache.values())
            ),
        },
        "economic_evidence": {
            "status": "not_assessed_by_definition_audit",
            "evidence_owners": deepcopy(contract["evidence_owners"]),
            "definition_grants_evidence": False,
            "accuracy_is_profitability": False,
        },
        "implementation_verification": {
            "status": "not_verified_by_static_definition_audit",
            "runtime_conformance_verified": False,
        },
        "authority": deepcopy(contract["authority"]),
        "records": records,
    }


def render_audit_markdown(audit: Mapping[str, Any]) -> str:
    lines = [
        "# Bot Definition Audit",
        "",
        f"Audited {audit.get('audited_record_count', 0)}/{audit.get('registry_record_count', 0)} registry records.",
        f"Definition complete: {audit.get('definition_complete_count', 0)}. Definition incomplete: {audit.get('definition_incomplete_count', 0)}.",
        "",
        "Audit coverage is not definition completeness. Static definitions do not verify runtime conformance or economic edge.",
        "Economic evidence: not assessed here; candidate-bound evidence remains with the existing strategy and profitability owners.",
        "",
        "## Seven Areas",
        "",
        "| Area | Complete Bots | Defined / Required Fields |",
        "| --- | ---: | ---: |",
    ]
    if audit.get("completion_scope"):
        lines[5:5] = [
            "Completion scope: source-bound operating jobs, not standalone trading strategies.",
            f"Standalone trading mandates complete: {audit.get('standalone_trading_mandate_summary', {}).get('definition_complete_count', 0)}/{audit.get('registry_record_count', 0)}. Their original requirements and gaps remain separate and unchanged.",
            "",
        ]
    for area, row in audit.get("area_summary", {}).items():
        lines.append(
            f"| {area} | {row['complete_count']} | {row['defined_field_count']} / {row['required_field_count']} |"
        )
    for area_index, (area, row) in enumerate(audit.get("area_summary", {}).items(), 1):
        if not row.get("subsections"):
            continue
        lines.extend(
            [
                "",
                f"### {area_index}. {area.replace('_', ' ').title()}",
                "",
                "| Subsection | Field | Complete / Required Bots |",
                "| --- | --- | ---: |",
            ]
        )
        for field, subsection in row["subsections"].items():
            title = subsection.get("title") or "Invalid Subsection Definition"
            lines.append(
                f"| {subsection['number']} {title} | `{field}` | {subsection['complete_count']} / {subsection['required_count']} |"
            )
    if audit.get("process_summary"):
        summary = audit["process_summary"]
        lines.extend(
            [
                "",
                "## Bot Processes",
                "",
                f"Defined bots: {summary['defined_bot_count']}. Authored stages: {summary['stage_count']}. Source-referenced stages: {summary['source_referenced_stage_count']}. Explicit source-reference gaps: {summary['source_reference_gap_count']}.",
                "Runtime-verified bot count: 0. The dependency graph orders definitions; it does not certify execution order, successful work or profitability.",
                "",
                "| Profile | Ordered Process Stages |",
                "| --- | --- |",
            ]
        )
        contract = audit.get("process_contract", {})
        for profile, definition in contract.get("profiles", {}).items():
            titles = [
                contract["stages"][stage["id"]]["title"]
                for stage in definition["stages"]
            ]
            lines.append(f"| {profile} | {'; '.join(titles)} |")
        lines.extend(
            [
                "",
                "Every stage declares inputs, outputs, owner, dependency IDs, failure/retry semantics and required completion evidence. Per-bot expansion shows exact source references or explicit gaps. Collection wrappers remain metadata-only and registry slots do not acquire implementations.",
            ]
        )
    lines.extend(["", "## Inventory", ""])
    lines.extend(
        f"- {kind}: {count}" for kind, count in audit.get("kind_counts", {}).items()
    )
    lines.extend(["", "## Highest-Frequency Gaps", ""])
    if not audit.get("gap_counts"):
        lines.append("None in this completion scope.")
    for gap, count in sorted(
        audit.get("gap_counts", {}).items(), key=lambda item: (-item[1], item[0])
    )[:25]:
        lines.append(f"- {gap}: {count}")
    lines.extend(["", "## Issues", ""])
    if not audit.get("issue_counts"):
        lines.append("None in this completion scope.")
    lines.extend(
        f"- {issue}: {count}" for issue, count in audit.get("issue_counts", {}).items()
    )
    lines.extend(
        [
            "",
            "## Resolution",
            "",
            "Every registry row retains an operating definition and its separate standalone trading-mandate audit under definition_audit.records in the native hierarchy JSON.",
            "",
            "Inspect one complete definition with bot-organization --bot-definition BOT_ID --json. Operating role contracts are owned by config/bot_operating_definition_contract_v1.json; pinned per-bot receipts are in config/bot_operating_definitions_v1.json. A source or registry-definition change requires explicit review and --materialize-operating-definitions; scheduled audits never repin changed sources.",
            "",
            "Use --require-definition-complete for all operating definitions, and --require-trading-mandate-complete for the original standalone trading requirements. Existing strategy evidence, runtime conformance and promotion gates are unchanged. No collection wrapper or registry-only slot has been turned into a trading implementation.",
            "",
            "No registry mutation, model training, order, promotion, worker start, or economic clearance was performed.",
            "",
        ]
    )
    return "\n".join(lines)
