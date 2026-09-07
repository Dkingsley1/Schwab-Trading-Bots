"""Deterministic, execution-free organization for the registered bot fleet."""

from __future__ import annotations

import ast
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping

from core.operating_contracts import build_operating_contract
from core.regime_taxonomy import (
    build_regime_metadata_access,
    classify_regime_profile,
    validate_regime_model,
)

REQUIRED_LEVELS = ("sleeve_id", "sub_sleeve_id", "cohort_id", "role_id")
SOURCE_CONFIDENCE = {
    "registry_explicit": 1.0,
    "registry_tag": 0.9,
    "module_literal": 0.85,
    "policy_rule": 0.78,
    "catalog_category": 0.72,
    "role_fallback": 0.72,
    "policy_fallback": 0.6,
}
SIGNAL_ROLES = {
    "signal_sub_bot",
    "options_sub_bot",
    "futures_sub_bot",
    "macro_sub_bot",
    "crypto_sub_bot",
}
SETUP_TIER_IDS = ("infrastructure", "sub", "master", "grand_master")
GRAND_SETUP_MARKERS = (
    "grandmaster",
    "grand_master",
    "grand master",
)
MASTER_SETUP_MARKERS = (
    "sleeve_master",
    "master_bot",
    "master_coordination",
    "per_sleeve_master_bots",
)
INFRASTRUCTURE_SETUP_MARKERS = (
    "infrastructure",
    "infra",
    "guard",
    "watchdog",
    "supervisor",
    "validator",
)
TRIPWIRE_SEVERITIES = ("advisory", "watch", "degraded", "critical")
TRIPWIRE_BLOCKING_SEVERITIES = {"degraded", "critical"}
TRIPWIRE_OPERATORS = {"gt", "gte", "lt", "lte", "eq", "ne"}
TRIPWIRE_AUTHORITY_FALSE_FIELDS = (
    "can_change_runtime_decisions",
    "can_mutate_registry",
    "can_change_source_code",
    "can_submit_paper_order",
    "can_submit_live_order",
    "can_allocate_capital",
    "can_promote_candidate",
    "can_claim_profitability",
)


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return int(default)


def slug(value: Any) -> str:
    text = re.sub(r"[^a-z0-9]+", "_", str(value or "").strip().lower())
    return re.sub(r"_+", "_", text).strip("_")


def canonical_hash(value: Any) -> str:
    raw = json.dumps(
        value, ensure_ascii=True, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _ordered_unique(values: Iterable[Any]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        item = str(value or "").strip()
        if not item or item in seen:
            continue
        seen.add(item)
        result.append(item)
    return result


def _tag_value(tags: Iterable[Any], prefix: str) -> str:
    expected = f"{prefix}:"
    for raw in tags:
        item = str(raw or "").strip()
        if item.startswith(expected):
            return slug(item[len(expected) :])
    return ""


def _valid_identifier(value: Any, invalid: set[str]) -> str:
    item = slug(value)
    return "" if item in invalid else item


def _field_present(value: Any) -> bool:
    if isinstance(value, bool):
        return True
    if isinstance(value, list):
        return bool(_ordered_unique(value))
    if isinstance(value, dict):
        return bool(value)
    return bool(str(value or "").strip())


def _rule_match(
    text: str, rules: Iterable[Mapping[str, Any]], id_key: str
) -> tuple[str, str]:
    for rule in rules:
        identifier = slug(rule.get(id_key))
        tokens = [
            str(token or "").strip().lower() for token in _as_list(rule.get("tokens"))
        ]
        if identifier and any(token and token in text for token in tokens):
            return identifier, "policy_rule"
    return "", ""


def _validate_tripwire_contract(policy: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    contract = _as_dict(policy.get("tripwire_contract"))
    if not contract:
        return ["organization_tripwire_contract_missing"]
    if str(contract.get("contract_id") or "") != "bot_tripwire_contract_v1":
        errors.append("organization_tripwire_contract_id_invalid")
    authority = _as_dict(contract.get("authority"))
    invariants = _as_dict(contract.get("hardening_invariants"))
    if authority.get("metadata_only") is not True:
        errors.append("organization_tripwire_authority_metadata_only_required")
    authority_false_fields = _ordered_unique(
        _as_list(invariants.get("authority_false_fields"))
    ) or list(TRIPWIRE_AUTHORITY_FALSE_FIELDS)
    for key in authority_false_fields:
        if authority.get(key) is not False:
            errors.append(f"organization_tripwire_authority_{key}_must_be_false")

    severity_levels = set(_ordered_unique(_as_list(contract.get("severity_levels"))))
    if not set(TRIPWIRE_SEVERITIES).issubset(severity_levels):
        errors.append("organization_tripwire_severity_levels_incomplete")
    rows = [row for row in _as_list(contract.get("tripwires")) if isinstance(row, dict)]
    tripwire_ids = [slug(row.get("tripwire_id")) for row in rows]
    if not rows:
        errors.append("organization_tripwire_rows_missing")
    if any(not item for item in tripwire_ids):
        errors.append("organization_tripwire_id_missing")
    if len(tripwire_ids) != len(set(tripwire_ids)):
        errors.append("organization_tripwire_duplicate_ids")
    required_ids = set(
        _ordered_unique(_as_list(invariants.get("required_tripwire_ids")))
    )
    if required_ids and not required_ids.issubset(set(tripwire_ids)):
        errors.append("organization_tripwire_required_ids_missing")
    required_fields = _ordered_unique(
        _as_list(invariants.get("required_tripwire_fields"))
    ) or [
        "tripwire_id",
        "category",
        "severity",
        "metric",
        "operator",
        "threshold",
        "owner",
        "action",
        "evidence_required",
        "applies_to_tiers",
    ]
    for row in rows:
        for field in required_fields:
            if field == "threshold":
                missing = row.get(field) is None
            else:
                missing = not _field_present(row.get(field))
            if missing:
                errors.append(
                    f"organization_tripwire_{slug(row.get('tripwire_id')) or 'unknown'}_{field}_missing"
                )
        severity = slug(row.get("severity"))
        operator = slug(row.get("operator"))
        if severity not in TRIPWIRE_SEVERITIES:
            errors.append(
                f"organization_tripwire_{slug(row.get('tripwire_id')) or 'unknown'}_severity_invalid"
            )
        if operator not in TRIPWIRE_OPERATORS:
            errors.append(
                f"organization_tripwire_{slug(row.get('tripwire_id')) or 'unknown'}_operator_invalid"
            )
        for key in TRIPWIRE_AUTHORITY_FALSE_FIELDS:
            if row.get(key) is True:
                errors.append(
                    f"organization_tripwire_{slug(row.get('tripwire_id')) or 'unknown'}_{key}_must_not_be_true"
                )
    return _ordered_unique(errors)


def validate_policy(policy: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    hierarchy = _as_dict(policy.get("hierarchy"))
    classification = _as_dict(policy.get("classification"))
    resources = _as_dict(policy.get("resource_budgets"))
    ensemble = _as_dict(policy.get("ensemble_policy"))
    admission = _as_dict(policy.get("admission_policy"))
    safety = _as_dict(policy.get("safety_contract"))

    errors.extend(validate_regime_model(_as_dict(policy.get("regime_model"))))

    if _safe_int(policy.get("schema_version")) != 1:
        errors.append("organization_policy_schema_version_invalid")
    if str(policy.get("operating_mode") or "") != "metadata_and_shadow_only":
        errors.append("organization_operating_mode_not_shadow_only")
    if tuple(hierarchy.get("levels") or ()) != REQUIRED_LEVELS:
        errors.append("organization_hierarchy_levels_invalid")
    for key in (
        "required_registry_coverage_ratio",
        "required_unique_assignment_ratio",
        "minimum_high_confidence_ratio",
        "review_confidence_floor",
    ):
        value = _safe_float(hierarchy.get(key), -1.0)
        if not 0.0 <= value <= 1.0:
            errors.append(f"organization_{key}_invalid")

    for list_key, id_key in (
        ("sleeve_rules", "sleeve_id"),
        ("strategy_family_rules", "family_id"),
        ("horizon_rules", "horizon_id"),
        ("regime_rules", "regime_id"),
        ("role_rules", "role_id"),
    ):
        rows = [
            row
            for row in _as_list(classification.get(list_key))
            if isinstance(row, dict)
        ]
        identifiers = [slug(row.get(id_key)) for row in rows]
        if not rows or any(not item for item in identifiers):
            errors.append(f"organization_{list_key}_invalid")
        if len(identifiers) != len(set(identifiers)):
            errors.append(f"organization_{list_key}_duplicate_ids")
        if any(not _as_list(row.get("tokens")) for row in rows):
            errors.append(f"organization_{list_key}_missing_tokens")

    setup = _as_dict(policy.get("bot_setup_contract"))
    if not setup:
        errors.append("organization_bot_setup_contract_missing")
    else:
        setup_authority = _as_dict(setup.get("authority"))
        setup_invariants = _as_dict(setup.get("hardening_invariants"))
        if setup_authority.get("metadata_only") is not True:
            errors.append("organization_bot_setup_authority_metadata_only_required")
        for key in _ordered_unique(
            _as_list(setup_invariants.get("authority_false_fields"))
        ) or [
            "can_change_runtime_decisions",
            "can_mutate_registry",
            "can_change_source_code",
            "can_submit_paper_order",
            "can_submit_live_order",
            "can_allocate_capital",
            "can_promote_candidate",
            "can_claim_profitability",
        ]:
            if setup_authority.get(key) is not False:
                errors.append(f"organization_bot_setup_authority_{key}_must_be_false")

        tier_rows = [
            row
            for row in _as_list(setup.get("tier_definitions"))
            if isinstance(row, dict)
        ]
        tier_ids = [slug(row.get("tier_id")) for row in tier_rows]
        required_tiers = set(
            _ordered_unique(_as_list(setup_invariants.get("required_tiers")))
        ) or set(SETUP_TIER_IDS)
        if not required_tiers.issubset(set(tier_ids)):
            errors.append("organization_bot_setup_required_tiers_missing")
        if len(tier_ids) != len(set(tier_ids)):
            errors.append("organization_bot_setup_tier_duplicate_ids")
        required_tier_fields = _ordered_unique(
            _as_list(setup_invariants.get("required_tier_fields"))
        ) or [
            "tier_id",
            "display_name",
            "purpose",
            "reports_to_tier",
            "owns",
            "consumes",
            "publishes",
            "setup_requires",
            "forbidden_actions",
        ]
        if any(
            not _field_present(row.get(field))
            for row in tier_rows
            for field in required_tier_fields
        ):
            errors.append("organization_bot_setup_tier_fields_missing")

        role_rows = [
            row for row in _as_list(setup.get("role_groups")) if isinstance(row, dict)
        ]
        role_ids = [slug(row.get("role_id")) for row in role_rows]
        required_roles = set(
            _ordered_unique(_as_list(setup_invariants.get("required_role_groups")))
        )
        if not required_roles:
            required_roles = {
                slug(row.get("role_id"))
                for row in _as_list(classification.get("role_rules"))
                if isinstance(row, dict)
            }
            required_roles.update(
                slug(value)
                for value in _as_dict(classification.get("role_fallbacks")).values()
            )
            required_roles.discard("")
        if not required_roles.issubset(set(role_ids)):
            errors.append("organization_bot_setup_required_role_groups_missing")
        if len(role_ids) != len(set(role_ids)):
            errors.append("organization_bot_setup_role_group_duplicate_ids")
        required_role_fields = _ordered_unique(
            _as_list(setup_invariants.get("required_role_fields"))
        ) or ["role_id", "display_name", "purpose", "primary_outputs", "success_signal"]
        if any(
            not _field_present(row.get(field))
            for row in role_rows
            for field in required_role_fields
        ):
            errors.append("organization_bot_setup_role_group_fields_missing")

        lifecycle_rows = [
            row
            for row in _as_list(setup.get("lifecycle_states"))
            if isinstance(row, dict)
        ]
        lifecycle_ids = [slug(row.get("state")) for row in lifecycle_rows]
        required_lifecycle = set(
            _ordered_unique(_as_list(setup_invariants.get("allowed_lifecycle_states")))
        )
        if not required_lifecycle.issubset(set(lifecycle_ids)):
            errors.append("organization_bot_setup_lifecycle_states_missing")
        if len(lifecycle_ids) != len(set(lifecycle_ids)):
            errors.append("organization_bot_setup_lifecycle_duplicate_ids")
        required_lifecycle_fields = _ordered_unique(
            _as_list(setup_invariants.get("required_lifecycle_fields"))
        ) or ["state", "meaning", "paper_vote_allowed", "live_vote_allowed"]
        if any(
            not _field_present(row.get(field))
            for row in lifecycle_rows
            for field in required_lifecycle_fields
        ):
            errors.append("organization_bot_setup_lifecycle_fields_missing")

    errors.extend(_validate_tripwire_contract(policy))

    soft = _safe_int(resources.get("max_shadow_voters_per_cell_soft"))
    hard = _safe_int(resources.get("max_shadow_voters_per_cell_hard"))
    total = _safe_int(resources.get("max_total_shadow_voters"))
    if not 0 < soft <= hard <= total:
        errors.append("organization_shadow_voter_budgets_invalid")
    if _safe_int(resources.get("max_parallel_training_jobs_per_sub_sleeve")) != 1:
        errors.append("organization_sub_sleeve_training_not_single_flight")
    if _safe_int(resources.get("max_parallel_training_jobs_global")) != 1:
        errors.append("organization_global_training_not_single_flight")

    if str(ensemble.get("mode") or "") != "shadow_only":
        errors.append("organization_ensemble_not_shadow_only")
    for key in (
        "max_bot_weight",
        "max_correlation_cluster_weight",
        "max_sub_sleeve_weight",
        "max_sleeve_weight",
    ):
        value = _safe_float(ensemble.get(key), -1.0)
        if not 0.0 < value <= 1.0:
            errors.append(f"organization_ensemble_{key}_invalid")
    if _safe_float(ensemble.get("score_minimum"), 0.0) >= _safe_float(
        ensemble.get("score_maximum"), 0.0
    ):
        errors.append("organization_ensemble_score_range_invalid")
    if _safe_int(ensemble.get("minimum_distinct_sub_sleeves")) < 1:
        errors.append("organization_ensemble_diversity_floor_invalid")

    for key in (
        "require_named_sleeve_and_sub_sleeve",
        "require_documented_capability_gap",
        "require_incremental_out_of_sample_value",
        "require_positive_stressed_post_cost_expectancy",
        "require_locked_holdout",
        "require_multiple_testing_adjustment",
        "require_maximum_parent_or_peer_correlation",
        "require_resource_budget_clearance",
        "require_human_registry_admission",
    ):
        if admission.get(key) is not True:
            errors.append(f"organization_admission_{key}_disabled")
    if (
        not 0.0
        < _safe_float(admission.get("maximum_parent_or_peer_correlation"), 0.0)
        < 1.0
    ):
        errors.append("organization_admission_correlation_limit_invalid")
    if not 1 <= _safe_int(admission.get("max_new_bots_per_release"), 0) <= 10:
        errors.append("organization_admission_release_limit_invalid")

    for key in (
        "changes_runtime_decisions",
        "automatic_registry_mutation",
        "automatic_source_code_changes",
        "paper_execution_authority",
        "live_execution_authority",
        "automatic_live_promotion",
        "profitability_guaranteed",
    ):
        if safety.get(key) is not False:
            errors.append(f"organization_safety_{key}_must_be_false")
    return _ordered_unique(errors)


def load_literal_bot_spec(
    path: Path, *, maximum_bytes: int = 2_000_000
) -> tuple[dict[str, Any], str]:
    try:
        if path.stat().st_size > maximum_bytes:
            return {}, "module_too_large"
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (OSError, SyntaxError, UnicodeError):
        return {}, "module_parse_failed"
    for node in tree.body:
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        if not any(
            isinstance(target, ast.Name) and target.id == "BOT_SPEC"
            for target in targets
        ):
            continue
        try:
            value = ast.literal_eval(node.value)
        except (ValueError, TypeError, SyntaxError):
            return {}, "bot_spec_not_literal"
        return (_as_dict(value), "")
    return {}, "bot_spec_missing"


def _module_path(
    project_root: Path, bot_id: str, catalog_row: Mapping[str, Any]
) -> Path | None:
    exact = project_root / "core" / f"{bot_id}.py"
    if exact.is_file():
        return exact
    raw = str(catalog_row.get("core_file") or "").strip()
    candidate = project_root / raw
    return candidate if raw and candidate.is_file() else None


def _field_value(
    row: Mapping[str, Any],
    module_spec: Mapping[str, Any],
    *,
    field: str,
    tag_prefix: str = "",
    invalid: set[str],
) -> tuple[str, str]:
    direct = _valid_identifier(row.get(field), invalid)
    if direct:
        return direct, "registry_explicit"
    if tag_prefix:
        tagged = _valid_identifier(
            _tag_value(_as_list(row.get("labeling_tags")), tag_prefix), invalid
        )
        if tagged:
            return tagged, "registry_tag"
    module_direct = _valid_identifier(module_spec.get(field), invalid)
    if module_direct:
        return module_direct, "module_literal"
    if tag_prefix:
        module_tagged = _valid_identifier(
            _tag_value(_as_list(module_spec.get("labeling_tags")), tag_prefix), invalid
        )
        if module_tagged:
            return module_tagged, "module_literal"
    return "", ""


def _classification_text(
    row: Mapping[str, Any], module_spec: Mapping[str, Any], category: str
) -> str:
    fields = (
        "bot_id",
        "bot_role",
        "slot_kind",
        "slot_label",
        "slot_objective",
        "sleeve_family",
        "sleeve_profile",
        "capability_pack_slug",
    )
    values = [str(category)]
    for source in (row, module_spec):
        values.extend(str(source.get(field) or "") for field in fields)
    return " ".join(values).lower()


def _target_functions(
    row: Mapping[str, Any], module_spec: Mapping[str, Any]
) -> set[str]:
    functions: set[str] = set()
    for item in _as_list(row.get("target_functions")) or _as_list(
        module_spec.get("target_functions")
    ):
        text = str(item or "").strip().lower()
        if text:
            functions.add(text)
    return functions


def _setup_tier(row: Mapping[str, Any], module_spec: Mapping[str, Any]) -> str:
    identity_source = row
    if not any(
        str(row.get(field) or "").strip()
        for field in ("bot_id", "slot_kind", "bot_intelligence_layer")
    ):
        identity_source = module_spec
    identity_text = " ".join(
        [
            str(identity_source.get("bot_id") or ""),
            str(identity_source.get("slot_kind") or ""),
            str(identity_source.get("bot_intelligence_layer") or ""),
        ]
    ).lower()
    functions = _target_functions(row, module_spec)
    raw_role = (
        str(row.get("bot_role") or module_spec.get("bot_role") or "").strip().lower()
    )
    if (
        any(marker in identity_text for marker in GRAND_SETUP_MARKERS)
        or "grand_master" in functions
    ):
        return "grand_master"
    if (
        any(marker in identity_text for marker in MASTER_SETUP_MARKERS)
        or "sleeve_master" in functions
        or "master_bot" in functions
        or "sleeve_masters" in functions
    ):
        return "master"
    if raw_role == "infrastructure_sub_bot" or any(
        marker in identity_text for marker in INFRASTRUCTURE_SETUP_MARKERS
    ):
        return "infrastructure"
    return "sub"


def _setup_lifecycle_state(row: Mapping[str, Any]) -> str:
    explicit = slug(row.get("lifecycle_state"))
    if explicit:
        return explicit
    if bool(row.get("deleted_from_rotation", False)):
        return "deleted"
    return "active" if bool(row.get("active", False)) else "retired"


def _setup_contract_maps(
    policy: Mapping[str, Any],
) -> tuple[
    dict[str, Any],
    dict[str, dict[str, Any]],
    dict[str, dict[str, Any]],
    dict[str, dict[str, Any]],
]:
    setup = _as_dict(policy.get("bot_setup_contract"))
    tiers = {
        slug(row.get("tier_id")): row
        for row in _as_list(setup.get("tier_definitions"))
        if isinstance(row, dict) and slug(row.get("tier_id"))
    }
    roles = {
        slug(row.get("role_id")): row
        for row in _as_list(setup.get("role_groups"))
        if isinstance(row, dict) and slug(row.get("role_id"))
    }
    lifecycle = {
        slug(row.get("state")): row
        for row in _as_list(setup.get("lifecycle_states"))
        if isinstance(row, dict) and slug(row.get("state"))
    }
    return setup, tiers, roles, lifecycle


def _setup_hardening(
    *,
    setup: Mapping[str, Any],
    tiers: Mapping[str, Mapping[str, Any]],
    roles: Mapping[str, Mapping[str, Any]],
    lifecycle: Mapping[str, Mapping[str, Any]],
    assignments: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    rows = list(assignments)
    invariants = _as_dict(setup.get("hardening_invariants"))
    required_fields = _ordered_unique(
        _as_list(invariants.get("assignment_required_fields"))
    ) or [
        "bot_id",
        "setup_tier",
        "setup_role_group",
        "setup_lifecycle_state",
        "sleeve_id",
        "sub_sleeve_id",
        "cohort_id",
        "role_id",
        "regime_profile_id",
        "correlation_cluster_id",
    ]
    authority = _as_dict(setup.get("authority"))
    authority_false_fields = _ordered_unique(
        _as_list(invariants.get("authority_false_fields"))
    ) or [
        "can_change_runtime_decisions",
        "can_mutate_registry",
        "can_change_source_code",
        "can_submit_paper_order",
        "can_submit_live_order",
        "can_allocate_capital",
        "can_promote_candidate",
        "can_claim_profitability",
    ]
    checks = {
        "setup_contract_present": bool(setup),
        "tier_definitions_complete": set(SETUP_TIER_IDS).issubset(set(tiers)),
        "assignment_roles_have_setup_groups": all(
            slug(row.get("setup_role_group")) in roles for row in rows
        ),
        "assignment_lifecycle_states_are_known": all(
            slug(row.get("setup_lifecycle_state")) in lifecycle for row in rows
        ),
        "assignment_tiers_are_known": all(
            slug(row.get("setup_tier")) in tiers for row in rows
        ),
        "assignments_have_required_setup_fields": all(
            _field_present(row.get(field)) for row in rows for field in required_fields
        ),
        "metadata_authority_remains_true": authority.get("metadata_only") is True,
        "authority_false_fields_remain_false": all(
            authority.get(field) is False for field in authority_false_fields
        ),
        "lifecycle_states_do_not_grant_live_votes": all(
            _as_dict(row).get("live_vote_allowed") is False
            for row in lifecycle.values()
        ),
    }
    failed = [key for key, value in checks.items() if not value]
    return {
        "overall_status": "ready" if not failed else "blocked",
        "check_count": len(checks),
        "failed_check_count": len(failed),
        "failed_checks": failed,
        "checks": checks,
        "invariants": invariants,
    }


def _bot_setup_summary(
    *,
    setup: Mapping[str, Any],
    tiers: Mapping[str, Mapping[str, Any]],
    roles: Mapping[str, Mapping[str, Any]],
    lifecycle: Mapping[str, Mapping[str, Any]],
    assignments: Iterable[Mapping[str, Any]],
    hardening: Mapping[str, Any],
) -> dict[str, Any]:
    rows = list(assignments)
    missing = [
        str(row.get("bot_id") or "")
        for row in rows
        if not row.get("setup_tier")
        or not row.get("setup_role_group")
        or not row.get("setup_lifecycle_state")
    ]
    return {
        "contract_id": str(setup.get("contract_id") or ""),
        "assignment_count": len(rows),
        "setup_coverage_ratio": round(
            (len(rows) - len(missing)) / max(len(rows), 1), 6
        ),
        "missing_setup_metadata_count": len(missing),
        "missing_setup_metadata_examples": missing[:25],
        "tier_definition_ids": sorted(tiers),
        "role_group_ids": sorted(roles),
        "lifecycle_state_ids": sorted(lifecycle),
        "tier_counts": dict(
            sorted(Counter(row.get("setup_tier", "") for row in rows).items())
        ),
        "active_tier_counts": dict(
            sorted(
                Counter(
                    row.get("setup_tier", "")
                    for row in rows
                    if bool(row.get("active", False))
                ).items()
            )
        ),
        "role_group_counts": dict(
            sorted(Counter(row.get("setup_role_group", "") for row in rows).items())
        ),
        "lifecycle_state_counts": dict(
            sorted(
                Counter(row.get("setup_lifecycle_state", "") for row in rows).items()
            )
        ),
        "routing_invariants": _as_dict(setup.get("routing_invariants")),
        "operator_output_requirements": _as_list(
            setup.get("operator_output_requirements")
        ),
        "hardening": dict(hardening),
    }


def _metric_value(metrics: Mapping[str, Any], path: str) -> Any:
    current: Any = metrics
    for part in str(path or "").split("."):
        if not part:
            continue
        if isinstance(current, Mapping):
            current = current.get(part)
        else:
            return None
    return current


def _compare_tripwire(value: Any, operator: str, threshold: Any) -> bool:
    op = slug(operator)
    if op in {"eq", "ne"}:
        matched = value == threshold
        return matched if op == "eq" else not matched
    try:
        left = float(value)
        right = float(threshold)
    except (TypeError, ValueError):
        return False
    if op == "gt":
        return left > right
    if op == "gte":
        return left >= right
    if op == "lt":
        return left < right
    if op == "lte":
        return left <= right
    return False


def _tripwire_hardening(contract: Mapping[str, Any]) -> dict[str, Any]:
    errors = _validate_tripwire_contract({"tripwire_contract": contract})
    checks = {
        "contract_present": bool(contract),
        "contract_valid": not errors,
        "metadata_authority_remains_true": _as_dict(contract.get("authority")).get(
            "metadata_only"
        )
        is True,
        "tripwires_present": bool(_as_list(contract.get("tripwires"))),
        "no_tripwire_grants_execution_or_profitability": not any(
            row.get(field) is True
            for row in _as_list(contract.get("tripwires"))
            if isinstance(row, dict)
            for field in TRIPWIRE_AUTHORITY_FALSE_FIELDS
        ),
    }
    failed = [key for key, value in checks.items() if not value]
    return {
        "overall_status": "ready" if not failed else "blocked",
        "check_count": len(checks),
        "failed_check_count": len(failed),
        "failed_checks": failed,
        "validation_errors": errors,
        "checks": checks,
    }


def _evaluate_tripwires(
    policy: Mapping[str, Any], metrics: Mapping[str, Any]
) -> dict[str, Any]:
    contract = _as_dict(policy.get("tripwire_contract"))
    hardening = _tripwire_hardening(contract)
    evaluated: list[dict[str, Any]] = []
    for row in _as_list(contract.get("tripwires")):
        if not isinstance(row, dict):
            continue
        metric = str(row.get("metric") or "")
        value = _metric_value(metrics, metric)
        active = _compare_tripwire(
            value, str(row.get("operator") or ""), row.get("threshold")
        )
        severity = slug(row.get("severity"))
        evaluated.append(
            {
                "tripwire_id": slug(row.get("tripwire_id")),
                "category": slug(row.get("category")),
                "severity": severity,
                "metric": metric,
                "operator": slug(row.get("operator")),
                "threshold": row.get("threshold"),
                "current_value": value,
                "active": active,
                "blocking": bool(active and severity in TRIPWIRE_BLOCKING_SEVERITIES),
                "owner": str(row.get("owner") or ""),
                "action": str(row.get("action") or ""),
                "evidence_required": _as_list(row.get("evidence_required")),
                "applies_to_tiers": _as_list(row.get("applies_to_tiers")),
            }
        )
    active_rows = [row for row in evaluated if row["active"]]
    blocking_rows = [row for row in active_rows if row["blocking"]]
    severity_counts = dict(
        sorted(Counter(row["severity"] for row in active_rows).items())
    )
    category_counts = dict(
        sorted(Counter(row["category"] for row in active_rows).items())
    )
    return {
        "contract_id": str(contract.get("contract_id") or ""),
        "overall_status": (
            "blocked"
            if blocking_rows or hardening["overall_status"] != "ready"
            else "active_advisory" if active_rows else "ready"
        ),
        "active_tripwire_count": len(active_rows),
        "blocking_tripwire_count": len(blocking_rows),
        "tripwire_count": len(evaluated),
        "severity_counts": severity_counts,
        "category_counts": category_counts,
        "active_tripwires": active_rows,
        "blocking_tripwires": blocking_rows,
        "evaluated_tripwires": evaluated,
        "metrics": dict(metrics),
        "authority": _as_dict(contract.get("authority")),
        "hardening": hardening,
        "operator_output_requirements": _as_list(
            contract.get("operator_output_requirements")
        ),
    }


def _role_assignment(
    row: Mapping[str, Any],
    module_spec: Mapping[str, Any],
    text: str,
    classification: Mapping[str, Any],
) -> tuple[str, str]:
    role, source = _rule_match(
        text, _as_list(classification.get("role_rules")), "role_id"
    )
    if role:
        return role, source
    raw_role = str(row.get("bot_role") or module_spec.get("bot_role") or "").strip()
    fallback = slug(_as_dict(classification.get("role_fallbacks")).get(raw_role))
    if fallback:
        return fallback, "role_fallback"
    return "shared_service", "policy_fallback"


def _assignment_confidence(sources: Iterable[str]) -> float:
    values = [SOURCE_CONFIDENCE.get(str(source), 0.5) for source in sources]
    return round(sum(values) / max(len(values), 1), 4)


def _quality_grade(ratio: float, *, structurally_ready: bool) -> str:
    if not structurally_ready:
        return "F"
    if ratio >= 0.98:
        return "A+"
    if ratio >= 0.9:
        return "A"
    if ratio >= 0.8:
        return "B"
    if ratio >= 0.7:
        return "C"
    if ratio >= 0.6:
        return "D"
    return "F"


def organize_registry(
    registry: Mapping[str, Any],
    policy: Mapping[str, Any],
    *,
    catalog: Mapping[str, Any] | None = None,
    project_root: Path | None = None,
) -> dict[str, Any]:
    hierarchy = _as_dict(policy.get("hierarchy"))
    classification = _as_dict(policy.get("classification"))
    resources = _as_dict(policy.get("resource_budgets"))
    invalid = {slug(item) for item in _as_list(hierarchy.get("invalid_identifiers"))}
    invalid.add("")
    rows = [row for row in _as_list(registry.get("sub_bots")) if isinstance(row, dict)]
    catalog_rows = [
        row for row in _as_list(_as_dict(catalog).get("bots")) if isinstance(row, dict)
    ]
    catalog_by_id = {str(row.get("bot_id") or "").strip(): row for row in catalog_rows}
    setup_contract, setup_tiers, setup_roles, setup_lifecycle = _setup_contract_maps(
        policy
    )
    bot_ids = [str(row.get("bot_id") or "").strip() for row in rows]
    duplicate_bot_ids = sorted(
        {item for item in bot_ids if item and bot_ids.count(item) > 1}
    )
    assignments: list[dict[str, Any]] = []
    module_stats = Counter()

    for row in sorted(rows, key=lambda item: str(item.get("bot_id") or "")):
        bot_id = str(row.get("bot_id") or "").strip()
        catalog_row = catalog_by_id.get(bot_id, {})
        module_spec: dict[str, Any] = {}
        if project_root is not None:
            path = _module_path(project_root, bot_id, catalog_row)
            if path is not None:
                module_spec, module_error = load_literal_bot_spec(path)
                module_stats["parsed" if module_spec else module_error or "empty"] += 1
            else:
                module_stats["module_missing"] += 1

        text = _classification_text(
            row, module_spec, str(catalog_row.get("category") or "")
        )
        sleeve, sleeve_source = _field_value(
            row,
            module_spec,
            field="sleeve_profile",
            tag_prefix="sleeve_profile",
            invalid=invalid,
        )
        if not sleeve:
            sleeve, sleeve_source = _field_value(
                row,
                module_spec,
                field="sleeve_family",
                tag_prefix="sleeve_family",
                invalid=invalid,
            )
        if not sleeve:
            sleeve, sleeve_source = _rule_match(
                text,
                _as_list(classification.get("sleeve_rules")),
                "sleeve_id",
            )
        if not sleeve:
            category = slug(catalog_row.get("category"))
            sleeve = _valid_identifier(
                _as_dict(classification.get("category_to_sleeve")).get(category),
                invalid,
            )
            sleeve_source = "catalog_category" if sleeve else ""
        raw_role = str(row.get("bot_role") or module_spec.get("bot_role") or "").strip()
        if not sleeve:
            default_key = (
                "default_signal_sleeve"
                if raw_role in SIGNAL_ROLES
                else "default_infrastructure_sleeve"
            )
            sleeve = _valid_identifier(classification.get(default_key), invalid)
            sleeve_source = "policy_fallback"

        family, family_source = _field_value(
            row,
            module_spec,
            field="strategy_family",
            tag_prefix="strategy_family",
            invalid=invalid,
        )
        if not family:
            family, family_source = _rule_match(
                text,
                _as_list(classification.get("strategy_family_rules")),
                "family_id",
            )
        if not family:
            fallback_key = (
                "default_signal_family"
                if raw_role in SIGNAL_ROLES
                else "default_infrastructure_family"
            )
            family = _valid_identifier(classification.get(fallback_key), invalid)
            family_source = "policy_fallback"

        horizon, horizon_source = _field_value(
            row,
            module_spec,
            field="horizon_id",
            tag_prefix="horizon",
            invalid=invalid,
        )
        if not horizon:
            horizon, horizon_source = _rule_match(
                text,
                _as_list(classification.get("horizon_rules")),
                "horizon_id",
            )
        if not horizon:
            fallback_key = (
                "default_signal_horizon"
                if raw_role in SIGNAL_ROLES
                else "default_infrastructure_horizon"
            )
            horizon = _valid_identifier(classification.get(fallback_key), invalid)
            horizon_source = "policy_fallback"

        role, role_source = _role_assignment(row, module_spec, text, classification)
        preferred_regimes = _ordered_unique(
            slug(item)
            for source in (row, module_spec)
            for item in _as_list(source.get("preferred_regimes"))
            if _valid_identifier(item, invalid)
        )
        if preferred_regimes:
            regimes = _ordered_unique(
                _rule_match(
                    regime,
                    _as_list(classification.get("regime_rules")),
                    "regime_id",
                )[0]
                or "specialized_regime"
                for regime in preferred_regimes
            )[:4]
            regime_source = (
                "registry_explicit"
                if _as_list(row.get("preferred_regimes"))
                else "module_literal"
            )
        else:
            regime, regime_source = _rule_match(
                text,
                _as_list(classification.get("regime_rules")),
                "regime_id",
            )
            regimes = [
                regime
                or _valid_identifier(classification.get("default_regime"), invalid)
            ]
            if not regime:
                regime_source = "policy_fallback"

        regime_profile = classify_regime_profile(
            row=row,
            module_spec=module_spec,
            classification_text=text,
            raw_role=raw_role,
            role_id=role,
            sub_sleeve_id=family,
            horizon_id=horizon,
            model=_as_dict(policy.get("regime_model")),
        )
        regime_metadata_access = build_regime_metadata_access(
            regime_profile,
            _as_dict(policy.get("regime_model")),
        )
        cohort = str(
            regime_profile.get("cohort_id") or slug(f"{horizon}__{regimes[0]}")
        )
        cell_id = "/".join((sleeve, family, cohort, role))
        correlation_cluster = "/".join((sleeve, family, horizon))
        base_confidence = _assignment_confidence(
            (sleeve_source, family_source, horizon_source, role_source)
        )
        confidence = round(
            (
                base_confidence * 4.0
                + _safe_float(regime_profile.get("profile_confidence"), 0.0)
            )
            / 5.0,
            4,
        )
        review_floor = _safe_float(hierarchy.get("review_confidence_floor"), 0.7)
        review_reasons = _ordered_unique(
            [
                (
                    "classification_confidence_below_floor"
                    if confidence < review_floor
                    else ""
                ),
                *_as_list(regime_profile.get("review_reasons")),
            ]
        )
        authority = {
            "paper_trading_enabled_in_registry": bool(
                row.get("paper_trading_enabled", row.get("paper_trade_enabled", False))
            ),
            "allocation_enabled_in_registry": bool(
                row.get("allocation_enabled", False)
            ),
            "execution_enabled_in_registry": bool(row.get("execution_enabled", False)),
            "live_trading_enabled_in_registry": bool(
                row.get("live_trading_enabled", False)
            ),
            "organization_layer_execution_authority": False,
        }
        setup_tier = _setup_tier(row, module_spec)
        setup_role_group = role
        setup_lifecycle_state = _setup_lifecycle_state(row)
        setup_tier_contract = _as_dict(setup_tiers.get(setup_tier))
        shadow_vote_eligible = bool(
            row.get("active", False)
            and raw_role in SIGNAL_ROLES
            and authority["paper_trading_enabled_in_registry"]
            and not row.get("deleted_from_rotation", False)
        )
        assignments.append(
            {
                "bot_id": bot_id,
                "active": bool(row.get("active", False)),
                "lifecycle_state": str(row.get("lifecycle_state") or ""),
                "setup_contract_id": str(setup_contract.get("contract_id") or ""),
                "setup_tier": setup_tier,
                "setup_role_group": setup_role_group,
                "setup_lifecycle_state": setup_lifecycle_state,
                "setup_reports_to_tier": str(
                    setup_tier_contract.get("reports_to_tier") or ""
                ),
                "sleeve_id": sleeve,
                "sub_sleeve_id": family,
                "horizon_id": horizon,
                "regime_ids": regimes,
                "preferred_regimes": preferred_regimes[:8],
                "regime_scope": str(regime_profile.get("scope") or ""),
                "regime_profile_id": str(regime_profile.get("profile_id") or ""),
                "regime_profile": regime_profile,
                "regime_metadata_access": regime_metadata_access,
                "regime_scenario_partitioned": bool(
                    regime_profile.get("scenario_partitioned", False)
                ),
                "regime_scenario_count": _safe_int(
                    regime_profile.get("scenario_count")
                ),
                "regime_axis_coverage_ratio": _safe_float(
                    regime_profile.get("axis_coverage_ratio")
                ),
                "regime_axis_specificity_ratio": _safe_float(
                    regime_profile.get("axis_specificity_ratio")
                ),
                "cohort_id": cohort,
                "role_id": role,
                "cell_id": cell_id,
                "correlation_cluster_id": correlation_cluster,
                "shadow_vote_eligible": shadow_vote_eligible,
                "resource_class": (
                    "latency_sensitive"
                    if horizon in {"subminute", "intraday"}
                    else "standard"
                ),
                "classification_confidence": confidence,
                "needs_review": bool(review_reasons),
                "review_reasons": review_reasons,
                "provenance": {
                    "sleeve": sleeve_source,
                    "sub_sleeve": family_source,
                    "horizon": horizon_source,
                    "regime": regime_source,
                    "regime_axes": {
                        axis_id: str(_as_dict(axis).get("source") or "")
                        for axis_id, axis in _as_dict(
                            regime_profile.get("axes")
                        ).items()
                    },
                    "regime_scope": str(regime_profile.get("scope_source") or ""),
                    "regime_scenarios": str(
                        regime_profile.get("scenario_partition_source") or ""
                    ),
                    "regime_metadata_access": str(
                        regime_metadata_access.get("contract_version") or ""
                    ),
                    "role": role_source,
                },
                "authority": authority,
            }
        )

    required_fields = REQUIRED_LEVELS
    invalid_assignments = [
        row["bot_id"]
        for row in assignments
        if any(slug(row.get(field)) in invalid for field in required_fields)
    ]
    assignment_ids = [row["bot_id"] for row in assignments if row["bot_id"]]
    organized_count = len(assignments) - len(invalid_assignments)
    registry_count = len(rows)
    coverage_ratio = organized_count / max(registry_count, 1)
    unique_ratio = len(set(assignment_ids)) / max(registry_count, 1)
    high_confidence_count = sum(
        1
        for row in assignments
        if _safe_float(row.get("classification_confidence"))
        >= _safe_float(hierarchy.get("review_confidence_floor"), 0.7)
    )
    high_confidence_ratio = high_confidence_count / max(registry_count, 1)
    regime_quality_axis_slots = sum(
        _safe_int(
            _as_dict(row.get("regime_profile")).get("quality_axis_slot_count"),
            len(_as_list(_as_dict(row.get("regime_profile")).get("quality_axes"))),
        )
        for row in assignments
    )
    regime_known_axis_slots = sum(
        _safe_int(
            _as_dict(row.get("regime_profile")).get("known_axis_slot_count"),
            len(_as_list(_as_dict(row.get("regime_profile")).get("known_axes"))),
        )
        for row in assignments
    )
    regime_specific_axis_slots = sum(
        _safe_int(
            _as_dict(row.get("regime_profile")).get("specific_axis_slot_count"),
            len(_as_list(_as_dict(row.get("regime_profile")).get("specific_axes"))),
        )
        for row in assignments
    )
    regime_axis_coverage_ratio = regime_known_axis_slots / max(
        regime_quality_axis_slots, 1
    )
    regime_axis_specificity_ratio = regime_specific_axis_slots / max(
        regime_quality_axis_slots, 1
    )
    regime_profile_confidence = sum(
        _safe_float(_as_dict(row.get("regime_profile")).get("profile_confidence"))
        for row in assignments
    ) / max(registry_count, 1)
    regime_review_count = sum(
        1
        for row in assignments
        if bool(_as_dict(row.get("regime_profile")).get("requires_review"))
    )
    regime_scenario_profile_count = sum(
        1
        for row in assignments
        if bool(_as_dict(row.get("regime_profile")).get("scenario_partitioned", False))
    )
    regime_scenario_count = sum(
        _safe_int(_as_dict(row.get("regime_profile")).get("scenario_count"))
        for row in assignments
    )
    regime_scenario_review_count = sum(
        1
        for row in assignments
        if _as_list(_as_dict(row.get("regime_profile")).get("scenario_review_reasons"))
    )
    invalid_regime_scenario_profile_count = sum(
        1
        for row in assignments
        if _as_list(_as_dict(row.get("regime_profile")).get("scenario_contract_errors"))
    )
    overbroad_regime_profile_count = sum(
        1
        for row in assignments
        if any(
            reason
            in {"regime_profile_overbroad", "regime_profile_multi_axis_breadth_high"}
            for reason in _as_list(
                _as_dict(row.get("regime_profile")).get("review_reasons")
            )
        )
    )
    wildcard_regime_profile_count = sum(
        1 if _as_list(_as_dict(row.get("regime_profile")).get("wildcard_axes")) else 0
        for row in assignments
    )
    unknown_regime_profile_count = sum(
        1 if _as_list(_as_dict(row.get("regime_profile")).get("unknown_axes")) else 0
        for row in assignments
    )
    unmapped_regime_label_counts = Counter(
        label
        for row in assignments
        for label in _as_list(
            _as_dict(row.get("regime_profile")).get("unmapped_raw_regime_terms")
        )
    )
    unmapped_regime_profile_count = sum(
        1
        for row in assignments
        if _as_list(
            _as_dict(row.get("regime_profile")).get("unmapped_raw_regime_terms")
        )
    )
    regime_metadata_access_ready_count = sum(
        1
        for row in assignments
        if bool(_as_dict(row.get("regime_metadata_access")).get("access_ready", False))
    )
    regime_metadata_access_ratio = regime_metadata_access_ready_count / max(
        registry_count, 1
    )
    regime_metadata_context_required_count = sum(
        1
        for row in assignments
        if _as_list(
            _as_dict(row.get("regime_metadata_access")).get(
                "runtime_context_required_axis_ids"
            )
        )
    )
    regime_metadata_access_error_count = sum(
        1
        for row in assignments
        if _as_list(_as_dict(row.get("regime_metadata_access")).get("errors"))
    )
    explicit_sleeve_count = sum(
        1
        for row in assignments
        if _as_dict(row.get("provenance")).get("sleeve")
        in {"registry_explicit", "registry_tag", "module_literal"}
    )
    shadow_rows = [row for row in assignments if row["shadow_vote_eligible"]]
    cell_shadow_counts = Counter(row["cell_id"] for row in shadow_rows)
    soft_limit = _safe_int(resources.get("max_shadow_voters_per_cell_soft"), 24)
    hard_limit = _safe_int(resources.get("max_shadow_voters_per_cell_hard"), 96)
    soft_cells = [
        {"cell_id": key, "shadow_voter_count": value, "limit": soft_limit}
        for key, value in sorted(
            cell_shadow_counts.items(), key=lambda item: (-item[1], item[0])
        )
        if value > soft_limit
    ]
    hard_cells = [row for row in soft_cells if row["shadow_voter_count"] > hard_limit]
    setup_hardening = _setup_hardening(
        setup=setup_contract,
        tiers=setup_tiers,
        roles=setup_roles,
        lifecycle=setup_lifecycle,
        assignments=assignments,
    )
    bot_setup_summary = _bot_setup_summary(
        setup=setup_contract,
        tiers=setup_tiers,
        roles=setup_roles,
        lifecycle=setup_lifecycle,
        assignments=assignments,
        hardening=setup_hardening,
    )
    review_count = sum(1 for row in assignments if row["needs_review"])
    explicit_sleeve_ratio = explicit_sleeve_count / max(registry_count, 1)
    organization_layer_execution_authority_count = sum(
        1
        for row in assignments
        if _as_dict(row.get("authority")).get("organization_layer_execution_authority")
        is True
    )
    live_registry_flag_count = sum(
        1
        for row in assignments
        if _as_dict(row.get("authority")).get("live_trading_enabled_in_registry")
        is True
    )
    paper_registry_flag_count = sum(
        1
        for row in assignments
        if _as_dict(row.get("authority")).get("paper_trading_enabled_in_registry")
        is True
    )
    tripwire_metrics = {
        "registry_bot_count": registry_count,
        "organized_bot_count": organized_count,
        "organization_coverage_ratio": round(coverage_ratio, 6),
        "unique_assignment_ratio": round(unique_ratio, 6),
        "duplicate_bot_id_count": len(duplicate_bot_ids),
        "invalid_assignment_count": len(invalid_assignments),
        "high_confidence_ratio": round(high_confidence_ratio, 6),
        "review_queue_count": review_count,
        "explicit_sleeve_ratio": round(explicit_sleeve_ratio, 6),
        "soft_shadow_cell_count": len(soft_cells),
        "hard_shadow_cell_count": len(hard_cells),
        "shadow_voter_count": len(shadow_rows),
        "setup_failed_check_count": len(_as_list(setup_hardening.get("failed_checks"))),
        "regime_metadata_access_error_count": regime_metadata_access_error_count,
        "unknown_regime_profile_count": unknown_regime_profile_count,
        "overbroad_regime_profile_count": overbroad_regime_profile_count,
        "invalid_regime_scenario_profile_count": invalid_regime_scenario_profile_count,
        "organization_layer_execution_authority_count": organization_layer_execution_authority_count,
        "live_registry_flag_count": live_registry_flag_count,
        "paper_registry_flag_count": paper_registry_flag_count,
    }
    tripwire_summary = _evaluate_tripwires(policy, tripwire_metrics)

    policy_errors = validate_policy(policy)
    blockers = list(policy_errors)
    if not rows:
        blockers.append("bot_registry_empty_or_invalid")
    if duplicate_bot_ids:
        blockers.append("duplicate_registry_bot_ids")
    if coverage_ratio < _safe_float(
        hierarchy.get("required_registry_coverage_ratio"), 1.0
    ):
        blockers.append("registry_organization_coverage_below_floor")
    if unique_ratio < _safe_float(
        hierarchy.get("required_unique_assignment_ratio"), 1.0
    ):
        blockers.append("registry_unique_assignment_ratio_below_floor")
    if high_confidence_ratio < _safe_float(
        hierarchy.get("minimum_high_confidence_ratio"), 0.6
    ):
        blockers.append("registry_high_confidence_ratio_below_floor")
    if hard_cells:
        blockers.append("shadow_voter_cell_hard_limit_exceeded")
    if len(shadow_rows) > _safe_int(resources.get("max_total_shadow_voters"), 2000):
        blockers.append("total_shadow_voter_limit_exceeded")
    if invalid_regime_scenario_profile_count:
        blockers.append("invalid_regime_scenario_contracts")
    minimum_metadata_access_ratio = _safe_float(
        _as_dict(
            _as_dict(policy.get("regime_model")).get("metadata_access_contract")
        ).get("minimum_registry_access_ratio"),
        1.0,
    )
    if regime_metadata_access_ratio < minimum_metadata_access_ratio:
        blockers.append("regime_metadata_access_coverage_below_floor")
    if regime_metadata_access_error_count:
        blockers.append("regime_metadata_access_contract_errors")
    if _as_list(setup_hardening.get("failed_checks")):
        blockers.append("bot_setup_contract_hardening_failed")
    if _as_dict(tripwire_summary.get("hardening")).get("overall_status") != "ready":
        blockers.append("tripwire_contract_hardening_failed")
    blockers.extend(
        f"tripwire:{row.get('tripwire_id')}"
        for row in _as_list(tripwire_summary.get("blocking_tripwires"))
        if isinstance(row, dict) and row.get("tripwire_id")
    )
    blockers = _ordered_unique(blockers)
    active_advisory_tripwires = [
        row
        for row in _as_list(tripwire_summary.get("active_tripwires"))
        if isinstance(row, dict)
        and str(row.get("severity") or "") not in TRIPWIRE_BLOCKING_SEVERITIES
    ]
    advisories = _ordered_unique(
        [
            (
                "review_low_confidence_assignments"
                if high_confidence_count < registry_count
                else ""
            ),
            "review_incomplete_regime_profiles" if regime_review_count else "",
            (
                "replace_unknown_regime_axes_with_evidence_backed_metadata"
                if unknown_regime_profile_count
                else ""
            ),
            (
                "map_or_retire_unrecognized_preferred_regime_labels"
                if unmapped_regime_label_counts
                else ""
            ),
            (
                "repair_invalid_regime_scenario_contracts"
                if invalid_regime_scenario_profile_count
                else ""
            ),
            "rank_and_park_oversubscribed_shadow_cells" if soft_cells else "",
            (
                "increase_explicit_sleeve_metadata_coverage"
                if explicit_sleeve_count < registry_count
                else ""
            ),
            (
                "repair_regime_metadata_access"
                if regime_metadata_access_error_count
                else ""
            ),
            (
                "review_bot_setup_contract_metadata"
                if _as_list(setup_hardening.get("failed_checks"))
                else ""
            ),
            *[
                f"tripwire:{row.get('tripwire_id')}"
                for row in active_advisory_tripwires
                if row.get("tripwire_id")
            ],
        ]
    )

    counts = {
        "sleeves": dict(
            sorted(Counter(row["sleeve_id"] for row in assignments).items())
        ),
        "sub_sleeves": dict(
            sorted(Counter(row["sub_sleeve_id"] for row in assignments).items())
        ),
        "horizons": dict(
            sorted(Counter(row["horizon_id"] for row in assignments).items())
        ),
        "roles": dict(sorted(Counter(row["role_id"] for row in assignments).items())),
        "setup_tiers": dict(
            sorted(Counter(row["setup_tier"] for row in assignments).items())
        ),
        "setup_role_groups": dict(
            sorted(Counter(row["setup_role_group"] for row in assignments).items())
        ),
        "setup_lifecycle_states": dict(
            sorted(Counter(row["setup_lifecycle_state"] for row in assignments).items())
        ),
        "cohorts": dict(
            sorted(Counter(row["cohort_id"] for row in assignments).items())
        ),
        "regime_scopes": dict(
            sorted(Counter(row["regime_scope"] for row in assignments).items())
        ),
        "regime_scenario_ids": dict(
            sorted(
                Counter(
                    str(scenario.get("scenario_id") or "")
                    for row in assignments
                    for scenario in _as_list(
                        _as_dict(row.get("regime_profile")).get("regime_scenarios")
                    )
                    if isinstance(scenario, dict)
                    and str(scenario.get("scenario_id") or "")
                ).items()
            )
        ),
        "regime_profiles": dict(
            sorted(Counter(row["regime_profile_id"] for row in assignments).items())
        ),
        "regime_axes": {
            axis_id: dict(sorted(axis_counts.items()))
            for axis_id, axis_counts in sorted(
                {
                    axis_id: Counter(
                        value
                        for row in assignments
                        for value in _as_list(
                            _as_dict(_as_dict(row.get("regime_profile")).get("axes"))
                            .get(axis_id, {})
                            .get("values")
                        )
                    )
                    for axis_id in (
                        "market_direction",
                        "volatility_state",
                        "liquidity_state",
                        "macro_state",
                        "rates_credit_state",
                        "correlation_state",
                        "event_phase",
                        "market_session",
                        "operational_state",
                    )
                }.items()
            )
        },
    }
    review_limit = _safe_int(resources.get("max_review_queue_rows"), 250)
    review_candidates = sorted(
        (row for row in assignments if row["needs_review"]),
        key=lambda row: (row["classification_confidence"], row["bot_id"]),
    )[:review_limit]
    review_queue = [
        {
            "bot_id": row["bot_id"],
            "sleeve_id": row["sleeve_id"],
            "sub_sleeve_id": row["sub_sleeve_id"],
            "horizon_id": row["horizon_id"],
            "regime_scope": row["regime_scope"],
            "regime_profile_id": row["regime_profile_id"],
            "classification_confidence": row["classification_confidence"],
            "regime_axis_coverage_ratio": row["regime_axis_coverage_ratio"],
            "regime_axis_specificity_ratio": row["regime_axis_specificity_ratio"],
            "review_reasons": row["review_reasons"],
            "preferred_regimes": row["preferred_regimes"],
            "unknown_axes": _as_list(
                _as_dict(row.get("regime_profile")).get("unknown_axes")
            ),
            "wildcard_axes": _as_list(
                _as_dict(row.get("regime_profile")).get("wildcard_axes")
            ),
            "critical_unknown_axes": _as_list(
                _as_dict(row.get("regime_profile")).get("critical_unknown_axes")
            ),
            "unmapped_raw_regime_terms": _as_list(
                _as_dict(row.get("regime_profile")).get("unmapped_raw_regime_terms")
            ),
            "regime_scenario_partitioned": bool(
                row.get("regime_scenario_partitioned", False)
            ),
            "regime_scenario_count": _safe_int(row.get("regime_scenario_count")),
            "scenario_contract_errors": _as_list(
                _as_dict(row.get("regime_profile")).get("scenario_contract_errors")
            ),
            "scenario_review_reasons": _as_list(
                _as_dict(row.get("regime_profile")).get("scenario_review_reasons")
            ),
            "regime_metadata_access_ready": bool(
                _as_dict(row.get("regime_metadata_access")).get("access_ready", False)
            ),
            "regime_metadata_access_errors": _as_list(
                _as_dict(row.get("regime_metadata_access")).get("errors")
            ),
        }
        for row in review_candidates
    ]
    assignment_receipt = canonical_hash(assignments)
    structurally_ready = not blockers
    classification_quality_score = (
        high_confidence_ratio * 0.7
        + regime_axis_coverage_ratio * 0.2
        + regime_axis_specificity_ratio * 0.1
    )
    classification_grade = _quality_grade(
        classification_quality_score,
        structurally_ready=structurally_ready,
    )
    regime_quality_score = (
        regime_axis_coverage_ratio * 0.65 + regime_axis_specificity_ratio * 0.35
    )
    regime_quality_grade = _quality_grade(
        regime_quality_score,
        structurally_ready=structurally_ready,
    )
    organization_status = (
        "ready_with_review_debt"
        if structurally_ready and advisories
        else "ready" if structurally_ready else "blocked"
    )
    organization_operating_contract = build_operating_contract(
        contract_id="bot_organization_operating_contract_v1",
        owner="bot_organization_control",
        domain="bot_organization",
        status=organization_status,
        why=blockers[0] if blockers else (advisories[0] if advisories else "ready"),
        safe_authority=[
            "classify_bot_metadata",
            "publish_roster_assignments",
            "evaluate_tripwires",
            "surface_review_queue",
        ],
        blocked_authority=[
            "runtime_decision_change",
            "paper_order_submission",
            "live_order_submission",
            "capital_allocation",
            "automatic_registry_mutation",
            "automatic_bot_promotion",
        ],
        evidence_missing=[
            *blockers,
            *advisories,
            *[
                f"tripwire:{row.get('tripwire_id')}"
                for row in _as_list(tripwire_summary.get("active_tripwires"))
                if isinstance(row, dict) and row.get("tripwire_id")
            ],
        ],
        release_conditions=[
            "registry_organization_coverage_ratio_at_floor",
            "unique_assignment_ratio_at_floor",
            "high_confidence_ratio_at_floor",
            "blocking_tripwire_count_zero",
            "setup_contract_hardening_ready",
            "regime_metadata_access_errors_zero",
            "review_queue_assigned_or_retired",
        ],
        next_commands=[
            ["./scripts/ops/opsctl.sh", "bot-organization", "--json"],
            ["./scripts/ops/opsctl.sh", "infrabot-gap-roster", "--json"],
            ["./scripts/ops/opsctl.sh", "roster-resilience", "--json"],
        ],
        definition_gaps=[
            "unknown_regime_axes_need_metadata" if unknown_regime_profile_count else "",
            (
                "explicit_sleeve_metadata_not_complete"
                if explicit_sleeve_count < registry_count
                else ""
            ),
            "review_queue_exceeds_limit" if review_count > review_limit else "",
        ],
        measurement={
            "registry_bot_count": registry_count,
            "organized_bot_count": organized_count,
            "organization_coverage_ratio": round(coverage_ratio, 6),
            "explicit_sleeve_ratio": round(explicit_sleeve_ratio, 6),
            "unknown_regime_profile_count": unknown_regime_profile_count,
            "review_queue_count": review_count,
            "active_tripwire_count": tripwire_summary["active_tripwire_count"],
            "blocking_tripwire_count": tripwire_summary["blocking_tripwire_count"],
        },
        hardening={
            "tripwire_contract_status": tripwire_summary["overall_status"],
            "setup_hardening_status": setup_hardening["overall_status"],
            "metadata_authority_only": True,
            "organization_layer_execution_authority_count": organization_layer_execution_authority_count,
        },
    )
    return {
        "ok": structurally_ready,
        "overall_status": organization_status,
        "grade": classification_grade,
        "structural_grade": "A+" if structurally_ready else "F",
        "classification_quality_grade": classification_grade,
        "classification_quality_score": round(classification_quality_score, 6),
        "regime_quality_grade": regime_quality_grade,
        "regime_quality_score": round(regime_quality_score, 6),
        "regime_model_id": str(
            _as_dict(policy.get("regime_model")).get("model_id") or ""
        ),
        "policy_id": str(policy.get("policy_id") or ""),
        "registry_bot_count": registry_count,
        "organized_bot_count": organized_count,
        "unique_assignment_count": len(set(assignment_ids)),
        "organization_coverage_ratio": round(coverage_ratio, 6),
        "unique_assignment_ratio": round(unique_ratio, 6),
        "high_confidence_assignment_count": high_confidence_count,
        "high_confidence_ratio": round(high_confidence_ratio, 6),
        "regime_quality_axis_slots": regime_quality_axis_slots,
        "regime_known_axis_slots": regime_known_axis_slots,
        "regime_specific_axis_slots": regime_specific_axis_slots,
        "regime_axis_coverage_ratio": round(regime_axis_coverage_ratio, 6),
        "regime_axis_specificity_ratio": round(regime_axis_specificity_ratio, 6),
        "mean_regime_profile_confidence": round(regime_profile_confidence, 6),
        "regime_review_count": regime_review_count,
        "regime_scenario_profile_count": regime_scenario_profile_count,
        "regime_scenario_count": regime_scenario_count,
        "regime_scenario_review_count": regime_scenario_review_count,
        "invalid_regime_scenario_profile_count": invalid_regime_scenario_profile_count,
        "overbroad_regime_profile_count": overbroad_regime_profile_count,
        "wildcard_regime_profile_count": wildcard_regime_profile_count,
        "unknown_regime_profile_count": unknown_regime_profile_count,
        "regime_metadata_access_grade": _quality_grade(
            regime_metadata_access_ratio,
            structurally_ready=not regime_metadata_access_error_count,
        ),
        "regime_metadata_access_ready_count": regime_metadata_access_ready_count,
        "regime_metadata_access_ratio": round(regime_metadata_access_ratio, 6),
        "regime_metadata_context_required_count": regime_metadata_context_required_count,
        "regime_metadata_access_error_count": regime_metadata_access_error_count,
        "unmapped_regime_profile_count": unmapped_regime_profile_count,
        "unmapped_regime_label_counts": dict(
            sorted(
                unmapped_regime_label_counts.items(),
                key=lambda item: (-item[1], item[0]),
            )
        ),
        "bot_setup_contract": {
            "contract_id": str(setup_contract.get("contract_id") or ""),
            "authority": _as_dict(setup_contract.get("authority")),
            "tier_definitions": _as_list(setup_contract.get("tier_definitions")),
            "role_groups": _as_list(setup_contract.get("role_groups")),
            "lifecycle_states": _as_list(setup_contract.get("lifecycle_states")),
            "routing_invariants": _as_dict(setup_contract.get("routing_invariants")),
            "operator_output_requirements": _as_list(
                setup_contract.get("operator_output_requirements")
            ),
        },
        "bot_setup_summary": bot_setup_summary,
        "setup_coverage_ratio": bot_setup_summary["setup_coverage_ratio"],
        "setup_hardening_status": setup_hardening["overall_status"],
        "setup_hardening_failed_checks": setup_hardening["failed_checks"],
        "tripwire_contract": {
            "contract_id": str(
                _as_dict(policy.get("tripwire_contract")).get("contract_id") or ""
            ),
            "authority": _as_dict(
                _as_dict(policy.get("tripwire_contract")).get("authority")
            ),
            "severity_levels": _as_list(
                _as_dict(policy.get("tripwire_contract")).get("severity_levels")
            ),
            "hardening_invariants": _as_dict(
                _as_dict(policy.get("tripwire_contract")).get("hardening_invariants")
            ),
            "operator_output_requirements": _as_list(
                _as_dict(policy.get("tripwire_contract")).get(
                    "operator_output_requirements"
                )
            ),
        },
        "tripwire_summary": tripwire_summary,
        "tripwire_metrics": tripwire_metrics,
        "operating_contract": organization_operating_contract,
        "bot_organization_operating_contract": organization_operating_contract,
        "active_tripwire_count": tripwire_summary["active_tripwire_count"],
        "blocking_tripwire_count": tripwire_summary["blocking_tripwire_count"],
        "tripwire_severity_counts": tripwire_summary["severity_counts"],
        "tripwire_category_counts": tripwire_summary["category_counts"],
        "active_tripwires": tripwire_summary["active_tripwires"],
        "blocking_tripwires": tripwire_summary["blocking_tripwires"],
        "explicit_sleeve_assignment_count": explicit_sleeve_count,
        "explicit_sleeve_ratio": round(explicit_sleeve_ratio, 6),
        "review_queue_count": review_count,
        "review_queue_limit": review_limit,
        "review_queue_truncated": review_count > review_limit,
        "review_queue": review_queue,
        "duplicate_bot_ids": duplicate_bot_ids,
        "invalid_assignment_bot_ids": invalid_assignments[:review_limit],
        "shadow_voter_count": len(shadow_rows),
        "oversubscribed_shadow_cells": soft_cells,
        "hard_limit_shadow_cells": hard_cells,
        "module_literal_stats": dict(sorted(module_stats.items())),
        "counts": counts,
        "blockers": blockers,
        "advisories": advisories,
        "assignment_receipt_sha256": assignment_receipt,
        "assignments": assignments,
    }
