"""Deterministic, execution-free organization for the registered bot fleet."""

from __future__ import annotations

import ast
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping

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
    raw = json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8")
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


def _rule_match(text: str, rules: Iterable[Mapping[str, Any]], id_key: str) -> tuple[str, str]:
    for rule in rules:
        identifier = slug(rule.get(id_key))
        tokens = [str(token or "").strip().lower() for token in _as_list(rule.get("tokens"))]
        if identifier and any(token and token in text for token in tokens):
            return identifier, "policy_rule"
    return "", ""


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
        rows = [row for row in _as_list(classification.get(list_key)) if isinstance(row, dict)]
        identifiers = [slug(row.get(id_key)) for row in rows]
        if not rows or any(not item for item in identifiers):
            errors.append(f"organization_{list_key}_invalid")
        if len(identifiers) != len(set(identifiers)):
            errors.append(f"organization_{list_key}_duplicate_ids")
        if any(not _as_list(row.get("tokens")) for row in rows):
            errors.append(f"organization_{list_key}_missing_tokens")

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
    if _safe_float(ensemble.get("score_minimum"), 0.0) >= _safe_float(ensemble.get("score_maximum"), 0.0):
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
    if not 0.0 < _safe_float(admission.get("maximum_parent_or_peer_correlation"), 0.0) < 1.0:
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


def load_literal_bot_spec(path: Path, *, maximum_bytes: int = 2_000_000) -> tuple[dict[str, Any], str]:
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
        if not any(isinstance(target, ast.Name) and target.id == "BOT_SPEC" for target in targets):
            continue
        try:
            value = ast.literal_eval(node.value)
        except (ValueError, TypeError, SyntaxError):
            return {}, "bot_spec_not_literal"
        return (_as_dict(value), "")
    return {}, "bot_spec_missing"


def _module_path(project_root: Path, bot_id: str, catalog_row: Mapping[str, Any]) -> Path | None:
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
        tagged = _valid_identifier(_tag_value(_as_list(row.get("labeling_tags")), tag_prefix), invalid)
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


def _classification_text(row: Mapping[str, Any], module_spec: Mapping[str, Any], category: str) -> str:
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


def _role_assignment(
    row: Mapping[str, Any],
    module_spec: Mapping[str, Any],
    text: str,
    classification: Mapping[str, Any],
) -> tuple[str, str]:
    role, source = _rule_match(text, _as_list(classification.get("role_rules")), "role_id")
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
    catalog_rows = [row for row in _as_list(_as_dict(catalog).get("bots")) if isinstance(row, dict)]
    catalog_by_id = {str(row.get("bot_id") or "").strip(): row for row in catalog_rows}
    bot_ids = [str(row.get("bot_id") or "").strip() for row in rows]
    duplicate_bot_ids = sorted({item for item in bot_ids if item and bot_ids.count(item) > 1})
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

        text = _classification_text(row, module_spec, str(catalog_row.get("category") or ""))
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
            sleeve = _valid_identifier(_as_dict(classification.get("category_to_sleeve")).get(category), invalid)
            sleeve_source = "catalog_category" if sleeve else ""
        raw_role = str(row.get("bot_role") or module_spec.get("bot_role") or "").strip()
        if not sleeve:
            default_key = "default_signal_sleeve" if raw_role in SIGNAL_ROLES else "default_infrastructure_sleeve"
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
            fallback_key = "default_signal_family" if raw_role in SIGNAL_ROLES else "default_infrastructure_family"
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
            fallback_key = "default_signal_horizon" if raw_role in SIGNAL_ROLES else "default_infrastructure_horizon"
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
            regime_source = "registry_explicit" if _as_list(row.get("preferred_regimes")) else "module_literal"
        else:
            regime, regime_source = _rule_match(
                text,
                _as_list(classification.get("regime_rules")),
                "regime_id",
            )
            regimes = [regime or _valid_identifier(classification.get("default_regime"), invalid)]
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
        cohort = str(regime_profile.get("cohort_id") or slug(f"{horizon}__{regimes[0]}"))
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
                "classification_confidence_below_floor" if confidence < review_floor else "",
                *_as_list(regime_profile.get("review_reasons")),
            ]
        )
        authority = {
            "paper_trading_enabled_in_registry": bool(
                row.get("paper_trading_enabled", row.get("paper_trade_enabled", False))
            ),
            "allocation_enabled_in_registry": bool(row.get("allocation_enabled", False)),
            "execution_enabled_in_registry": bool(row.get("execution_enabled", False)),
            "live_trading_enabled_in_registry": bool(row.get("live_trading_enabled", False)),
            "organization_layer_execution_authority": False,
        }
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
                "regime_scenario_count": _safe_int(regime_profile.get("scenario_count")),
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
                "resource_class": "latency_sensitive" if horizon in {"subminute", "intraday"} else "standard",
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
                        for axis_id, axis in _as_dict(regime_profile.get("axes")).items()
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
    regime_axis_coverage_ratio = regime_known_axis_slots / max(regime_quality_axis_slots, 1)
    regime_axis_specificity_ratio = regime_specific_axis_slots / max(regime_quality_axis_slots, 1)
    regime_profile_confidence = sum(
        _safe_float(_as_dict(row.get("regime_profile")).get("profile_confidence"))
        for row in assignments
    ) / max(registry_count, 1)
    regime_review_count = sum(
        1 for row in assignments if bool(_as_dict(row.get("regime_profile")).get("requires_review"))
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
            reason in {"regime_profile_overbroad", "regime_profile_multi_axis_breadth_high"}
            for reason in _as_list(_as_dict(row.get("regime_profile")).get("review_reasons"))
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
        if _as_list(_as_dict(row.get("regime_profile")).get("unmapped_raw_regime_terms"))
    )
    regime_metadata_access_ready_count = sum(
        1
        for row in assignments
        if bool(_as_dict(row.get("regime_metadata_access")).get("access_ready", False))
    )
    regime_metadata_access_ratio = regime_metadata_access_ready_count / max(registry_count, 1)
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
        for key, value in sorted(cell_shadow_counts.items(), key=lambda item: (-item[1], item[0]))
        if value > soft_limit
    ]
    hard_cells = [row for row in soft_cells if row["shadow_voter_count"] > hard_limit]

    policy_errors = validate_policy(policy)
    blockers = list(policy_errors)
    if not rows:
        blockers.append("bot_registry_empty_or_invalid")
    if duplicate_bot_ids:
        blockers.append("duplicate_registry_bot_ids")
    if coverage_ratio < _safe_float(hierarchy.get("required_registry_coverage_ratio"), 1.0):
        blockers.append("registry_organization_coverage_below_floor")
    if unique_ratio < _safe_float(hierarchy.get("required_unique_assignment_ratio"), 1.0):
        blockers.append("registry_unique_assignment_ratio_below_floor")
    if high_confidence_ratio < _safe_float(hierarchy.get("minimum_high_confidence_ratio"), 0.6):
        blockers.append("registry_high_confidence_ratio_below_floor")
    if hard_cells:
        blockers.append("shadow_voter_cell_hard_limit_exceeded")
    if len(shadow_rows) > _safe_int(resources.get("max_total_shadow_voters"), 2000):
        blockers.append("total_shadow_voter_limit_exceeded")
    if invalid_regime_scenario_profile_count:
        blockers.append("invalid_regime_scenario_contracts")
    minimum_metadata_access_ratio = _safe_float(
        _as_dict(_as_dict(policy.get("regime_model")).get("metadata_access_contract")).get(
            "minimum_registry_access_ratio"
        ),
        1.0,
    )
    if regime_metadata_access_ratio < minimum_metadata_access_ratio:
        blockers.append("regime_metadata_access_coverage_below_floor")
    if regime_metadata_access_error_count:
        blockers.append("regime_metadata_access_contract_errors")
    blockers = _ordered_unique(blockers)
    advisories = _ordered_unique(
        [
            "review_low_confidence_assignments" if high_confidence_count < registry_count else "",
            "review_incomplete_regime_profiles" if regime_review_count else "",
            "replace_unknown_regime_axes_with_evidence_backed_metadata"
            if unknown_regime_profile_count
            else "",
            "map_or_retire_unrecognized_preferred_regime_labels"
            if unmapped_regime_label_counts
            else "",
            "repair_invalid_regime_scenario_contracts"
            if invalid_regime_scenario_profile_count
            else "",
            "rank_and_park_oversubscribed_shadow_cells" if soft_cells else "",
            "increase_explicit_sleeve_metadata_coverage" if explicit_sleeve_count < registry_count else "",
            "repair_regime_metadata_access" if regime_metadata_access_error_count else "",
        ]
    )

    counts = {
        "sleeves": dict(sorted(Counter(row["sleeve_id"] for row in assignments).items())),
        "sub_sleeves": dict(sorted(Counter(row["sub_sleeve_id"] for row in assignments).items())),
        "horizons": dict(sorted(Counter(row["horizon_id"] for row in assignments).items())),
        "roles": dict(sorted(Counter(row["role_id"] for row in assignments).items())),
        "cohorts": dict(sorted(Counter(row["cohort_id"] for row in assignments).items())),
        "regime_scopes": dict(sorted(Counter(row["regime_scope"] for row in assignments).items())),
        "regime_scenario_ids": dict(
            sorted(
                Counter(
                    str(scenario.get("scenario_id") or "")
                    for row in assignments
                    for scenario in _as_list(
                        _as_dict(row.get("regime_profile")).get("regime_scenarios")
                    )
                    if isinstance(scenario, dict) and str(scenario.get("scenario_id") or "")
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
                            _as_dict(
                                _as_dict(row.get("regime_profile")).get("axes")
                            ).get(axis_id, {})
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
            "regime_scenario_partitioned": bool(row.get("regime_scenario_partitioned", False)),
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
    regime_quality_score = regime_axis_coverage_ratio * 0.65 + regime_axis_specificity_ratio * 0.35
    regime_quality_grade = _quality_grade(
        regime_quality_score,
        structurally_ready=structurally_ready,
    )
    return {
        "ok": structurally_ready,
        "overall_status": (
            "ready_with_review_debt" if structurally_ready and advisories else "ready" if structurally_ready else "blocked"
        ),
        "grade": classification_grade,
        "structural_grade": "A+" if structurally_ready else "F",
        "classification_quality_grade": classification_grade,
        "classification_quality_score": round(classification_quality_score, 6),
        "regime_quality_grade": regime_quality_grade,
        "regime_quality_score": round(regime_quality_score, 6),
        "regime_model_id": str(_as_dict(policy.get("regime_model")).get("model_id") or ""),
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
            sorted(unmapped_regime_label_counts.items(), key=lambda item: (-item[1], item[0]))
        ),
        "explicit_sleeve_assignment_count": explicit_sleeve_count,
        "explicit_sleeve_ratio": round(explicit_sleeve_count / max(registry_count, 1), 6),
        "review_queue_count": sum(1 for row in assignments if row["needs_review"]),
        "review_queue_limit": review_limit,
        "review_queue_truncated": sum(1 for row in assignments if row["needs_review"])
        > review_limit,
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
