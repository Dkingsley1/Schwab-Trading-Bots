"""Versioned multi-axis regime metadata and shadow compatibility scoring."""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Iterable, Mapping


REQUIRED_REGIME_AXES = (
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
VALID_SCOPES = {"market_signal", "hybrid", "operational_control"}
SOURCE_CONFIDENCE = {
    "registry_explicit": 1.0,
    "module_literal": 0.85,
    "policy_rule": 0.78,
    "policy_wildcard": 0.72,
    "policy_fallback": 0.4,
    "not_applicable": 1.0,
}


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if value is None:
        return []
    return [value]


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


def _ordered_unique(values: Iterable[Any]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        item = slug(value)
        if not item or item in seen:
            continue
        seen.add(item)
        result.append(item)
    return result


def _canonical_hash(value: Any) -> str:
    raw = json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _token_matches(text: str, token: Any) -> bool:
    expected = slug(token)
    return bool(expected and expected in slug(text))


def _rule_values(terms: Iterable[str], rules: Iterable[Mapping[str, Any]]) -> tuple[list[str], dict[str, list[str]]]:
    values: list[str] = []
    evidence: dict[str, list[str]] = {}
    normalized_terms = _ordered_unique(terms)
    for rule in rules:
        value_id = slug(rule.get("value_id"))
        tokens = _as_list(rule.get("tokens"))
        matched = [term for term in normalized_terms if any(_token_matches(term, token) for token in tokens)]
        if value_id and matched:
            values.append(value_id)
            evidence[value_id] = matched
    return _ordered_unique(values), evidence


def _axis_map(model: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        slug(row.get("axis_id")): dict(row)
        for row in _as_list(model.get("axes"))
        if isinstance(row, dict) and slug(row.get("axis_id"))
    }


def _metadata_access_errors(
    model: Mapping[str, Any],
    axes: Mapping[str, Mapping[str, Any]],
) -> list[str]:
    contract = _as_dict(model.get("metadata_access_contract"))
    errors: list[str] = []
    if str(contract.get("version") or "") != "regime_metadata_access_v1":
        errors.append("regime_metadata_access_version_invalid")
    if str(contract.get("mode") or "") != "read_only_runtime_context":
        errors.append("regime_metadata_access_mode_invalid")
    if not str(contract.get("catalog_uri") or "").strip():
        errors.append("regime_metadata_access_catalog_uri_missing")

    configured_by_scope = _as_dict(contract.get("readable_axes_by_scope"))
    for scope in VALID_SCOPES:
        configured = _ordered_unique(_as_list(configured_by_scope.get(scope)))
        required = [
            axis_id
            for axis_id, axis in axes.items()
            if scope in {slug(item) for item in _as_list(axis.get("required_for_scopes"))}
        ]
        if configured != required:
            errors.append(f"regime_metadata_access_{scope}_readable_axes_invalid")

    precedence = _ordered_unique(contract.get("resolver_precedence") or [])
    required_precedence = [
        "registry_explicit",
        "module_literal",
        "policy_rule",
        "runtime_observed_context",
    ]
    if precedence != required_precedence:
        errors.append("regime_metadata_access_resolver_precedence_invalid")
    for key in (
        "require_context_provenance",
        "fail_closed_on_invalid_context",
        "expose_unknown_profile_axes",
        "allow_training_observation",
    ):
        if contract.get(key) is not True:
            errors.append(f"regime_metadata_access_{key}_disabled")
    for key in (
        "infer_missing_profile_preferences",
        "changes_runtime_decisions",
        "paper_execution_authority",
        "live_execution_authority",
    ):
        if contract.get(key) is not False:
            errors.append(f"regime_metadata_access_{key}_must_be_false")
    minimum_ratio = _safe_float(contract.get("minimum_registry_access_ratio"), -1.0)
    if not 0.0 < minimum_ratio <= 1.0:
        errors.append("regime_metadata_access_registry_ratio_invalid")
    return _ordered_unique(errors)


def validate_regime_model(model: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    if _safe_int(model.get("schema_version")) != 1:
        errors.append("regime_model_schema_version_invalid")
    if str(model.get("mode") or "") != "multi_axis_shadow_only":
        errors.append("regime_model_mode_not_shadow_only")

    axes = _axis_map(model)
    if tuple(axes) != REQUIRED_REGIME_AXES:
        errors.append("regime_model_required_axes_invalid")
    for axis_id, axis in axes.items():
        values = _ordered_unique(axis.get("values") or [])
        required_values = {
            slug(axis.get("unknown_value")),
            slug(axis.get("wildcard_value")),
            slug(axis.get("not_applicable_value")),
        }
        if not values or "" in required_values or not required_values.issubset(set(values)):
            errors.append(f"regime_model_{axis_id}_values_invalid")
        scopes = {slug(item) for item in _as_list(axis.get("applicable_scopes"))}
        required_scopes = {slug(item) for item in _as_list(axis.get("required_for_scopes"))}
        if not scopes or not scopes.issubset(VALID_SCOPES) or not required_scopes.issubset(scopes):
            errors.append(f"regime_model_{axis_id}_scopes_invalid")
        rules = [row for row in _as_list(axis.get("rules")) if isinstance(row, dict)]
        rule_values = [slug(row.get("value_id")) for row in rules]
        if not rules or any(not item or item not in values for item in rule_values):
            errors.append(f"regime_model_{axis_id}_rules_invalid")
        if len(rule_values) != len(set(rule_values)):
            errors.append(f"regime_model_{axis_id}_duplicate_rule_values")
        if any(not _as_list(row.get("tokens")) for row in rules):
            errors.append(f"regime_model_{axis_id}_rule_tokens_missing")
        weight = _safe_float(axis.get("compatibility_weight"), -1.0)
        if not 0.0 < weight <= 5.0:
            errors.append(f"regime_model_{axis_id}_compatibility_weight_invalid")

    cohort_axes = _as_dict(model.get("cohort_axes_by_scope"))
    for scope in VALID_SCOPES:
        configured = [slug(item) for item in _as_list(cohort_axes.get(scope))]
        if not configured or any(item not in axes for item in configured):
            errors.append(f"regime_model_{scope}_cohort_axes_invalid")

    minimum_coverage = _as_dict(model.get("minimum_axis_coverage_by_scope"))
    for scope in VALID_SCOPES:
        value = _safe_float(minimum_coverage.get(scope), -1.0)
        if not 0.0 <= value <= 1.0:
            errors.append(f"regime_model_{scope}_coverage_floor_invalid")

    compatibility = _as_dict(model.get("compatibility_policy"))
    if str(compatibility.get("mode") or "") != "shadow_filter":
        errors.append("regime_model_compatibility_mode_invalid")
    for key in (
        "minimum_compatibility_score",
        "unknown_profile_axis_score",
        "wildcard_profile_axis_score",
        "unknown_context_axis_score",
    ):
        value = _safe_float(compatibility.get(key), -1.0)
        if not 0.0 <= value <= 1.0:
            errors.append(f"regime_model_{key}_invalid")
    if _safe_int(compatibility.get("minimum_context_axes"), 0) < 1:
        errors.append("regime_model_minimum_context_axes_invalid")
    minimum_context_by_scope = _as_dict(compatibility.get("minimum_context_axes_by_scope"))
    for scope in VALID_SCOPES:
        if _safe_int(minimum_context_by_scope.get(scope), 0) < 1:
            errors.append(f"regime_model_{scope}_minimum_context_axes_invalid")
    hard_mismatch = _as_dict(compatibility.get("hard_mismatch_axes_by_scope"))
    for scope in VALID_SCOPES:
        configured = [slug(item) for item in _as_list(hard_mismatch.get(scope))]
        if not configured or any(item not in axes for item in configured):
            errors.append(f"regime_model_{scope}_hard_mismatch_axes_invalid")

    safety = _as_dict(model.get("safety_contract"))
    for key in (
        "changes_runtime_decisions",
        "paper_execution_authority",
        "live_execution_authority",
        "automatic_regime_promotion",
    ):
        if safety.get(key) is not False:
            errors.append(f"regime_model_safety_{key}_must_be_false")

    scenarios = _as_dict(model.get("scenario_partition_contract"))
    if str(scenarios.get("version") or "") != "regime_scenario_partition_v1":
        errors.append("regime_scenario_partition_version_invalid")
    if str(scenarios.get("mode") or "") != "bounded_best_match_shadow_only":
        errors.append("regime_scenario_partition_mode_invalid")
    minimum_scenarios = _safe_int(scenarios.get("minimum_scenarios_per_profile"), 0)
    maximum_scenarios = _safe_int(scenarios.get("maximum_scenarios_per_profile"), 0)
    maximum_axis_values = _safe_int(scenarios.get("maximum_values_per_axis_per_scenario"), 0)
    if not 2 <= minimum_scenarios <= maximum_scenarios <= 32:
        errors.append("regime_scenario_partition_count_bounds_invalid")
    if not 1 <= maximum_axis_values <= max(_safe_int(model.get("maximum_values_per_axis"), 0), 1):
        errors.append("regime_scenario_partition_axis_value_limit_invalid")
    for key in (
        "require_unique_scenario_ids",
        "require_explicit_scope",
        "require_explicit_axes",
        "require_same_scope",
    ):
        if scenarios.get(key) is not True:
            errors.append(f"regime_scenario_partition_{key}_disabled")
    if str(scenarios.get("selection_policy") or "") != "highest_compatible_score_then_scenario_id":
        errors.append("regime_scenario_partition_selection_policy_invalid")
    if str(scenarios.get("unmatched_action") or "") != "exclude_and_report":
        errors.append("regime_scenario_partition_unmatched_action_invalid")
    for key in ("paper_execution_authority", "live_execution_authority"):
        if scenarios.get(key) is not False:
            errors.append(f"regime_scenario_partition_{key}_must_be_false")
    errors.extend(_metadata_access_errors(model, axes))
    return _ordered_unique(errors)


def build_regime_metadata_access(
    profile: Mapping[str, Any],
    model: Mapping[str, Any],
) -> dict[str, Any]:
    """Describe read-only catalog access without inventing regime preferences."""

    axes = _axis_map(model)
    contract = _as_dict(model.get("metadata_access_contract"))
    scope = slug(profile.get("scope"))
    errors = _metadata_access_errors(model, axes)
    if scope not in VALID_SCOPES:
        errors.append("regime_metadata_access_profile_scope_invalid")
    readable_axis_ids = _ordered_unique(
        _as_list(_as_dict(contract.get("readable_axes_by_scope")).get(scope))
    )
    missing_catalog_axis_ids = [axis_id for axis_id in readable_axis_ids if axis_id not in axes]
    if missing_catalog_axis_ids:
        errors.append("regime_metadata_access_catalog_axes_missing")
    profile_axes = _as_dict(profile.get("axes"))
    profile_unknown_axis_ids = [
        axis_id
        for axis_id in readable_axis_ids
        if bool(_as_dict(profile_axes.get(axis_id)).get("unknown", False))
    ]
    profile_specific_axis_ids = [
        axis_id
        for axis_id in readable_axis_ids
        if axis_id in set(_as_list(profile.get("specific_axes")))
    ]
    catalog_signature = {
        "model_id": str(model.get("model_id") or ""),
        "scope": scope,
        "axis_definitions": {
            axis_id: {
                "values": _ordered_unique(_as_dict(axes.get(axis_id)).get("values") or []),
                "unknown_value": slug(_as_dict(axes.get(axis_id)).get("unknown_value")),
                "wildcard_value": slug(_as_dict(axes.get(axis_id)).get("wildcard_value")),
                "not_applicable_value": slug(
                    _as_dict(axes.get(axis_id)).get("not_applicable_value")
                ),
            }
            for axis_id in readable_axis_ids
        },
    }
    errors = _ordered_unique(errors)
    return {
        "schema_version": 1,
        "contract_version": str(contract.get("version") or ""),
        "mode": str(contract.get("mode") or ""),
        "scope": scope,
        "access_ready": bool(not errors and readable_axis_ids),
        "catalog_uri": str(contract.get("catalog_uri") or ""),
        "catalog_receipt_sha256": _canonical_hash(catalog_signature),
        "readable_axis_ids": readable_axis_ids,
        "profile_specific_axis_ids": profile_specific_axis_ids,
        "profile_unknown_axis_ids": profile_unknown_axis_ids,
        "runtime_context_required_axis_ids": profile_unknown_axis_ids,
        "profile_preferences_complete": not profile_unknown_axis_ids,
        "resolver_precedence": _ordered_unique(contract.get("resolver_precedence") or []),
        "preference_inference_allowed": False,
        "errors": errors,
        "authority": {
            "read_only": True,
            "training_observation_allowed": bool(
                contract.get("allow_training_observation", False)
            ),
            "changes_runtime_decisions": False,
            "paper_execution_authority": False,
            "live_execution_authority": False,
            "order_payload_created": False,
        },
    }


def build_regime_metadata_view(
    profile: Mapping[str, Any],
    context: Mapping[str, Any],
    model: Mapping[str, Any],
) -> dict[str, Any]:
    """Materialize a validated context view for a bot's readable regime axes."""

    access = build_regime_metadata_access(profile, model)
    axes = _axis_map(model)
    context_axes = _as_dict(context.get("axes")) or dict(context)
    observed: dict[str, list[str]] = {}
    invalid: dict[str, list[str]] = {}
    known_axis_ids: list[str] = []
    for raw_axis_id in _as_list(access.get("readable_axis_ids")):
        axis_id = str(raw_axis_id)
        axis = _as_dict(axes.get(axis_id))
        allowed = set(_ordered_unique(axis.get("values") or []))
        values = _ordered_unique(_as_list(context_axes.get(axis_id)))
        invalid_values = [value for value in values if value not in allowed]
        if invalid_values:
            invalid[axis_id] = invalid_values
        values = [value for value in values if value in allowed]
        if not values:
            continue
        observed[axis_id] = values
        if slug(axis.get("unknown_value")) not in values:
            known_axis_ids.append(axis_id)

    compatibility = _as_dict(model.get("compatibility_policy"))
    minimum_axes = max(
        _safe_int(
            _as_dict(compatibility.get("minimum_context_axes_by_scope")).get(
                str(access.get("scope") or "")
            ),
            _safe_int(compatibility.get("minimum_context_axes"), 2),
        ),
        1,
    )
    access_ready = bool(access.get("access_ready", False))
    context_ready = bool(access_ready and not invalid and len(known_axis_ids) >= minimum_axes)
    status = (
        "ready"
        if context_ready
        else "invalid_context"
        if invalid
        else "insufficient_context"
        if access_ready
        else "blocked_contract"
    )
    context_signature = {
        "catalog_receipt_sha256": str(access.get("catalog_receipt_sha256") or ""),
        "profile_id": str(profile.get("profile_id") or ""),
        "axes": observed,
    }
    return {
        **access,
        "status": status,
        "context_ready": context_ready,
        "minimum_context_axes": minimum_axes,
        "observed_axis_ids": list(observed),
        "known_context_axis_ids": known_axis_ids,
        "missing_context_axis_ids": [
            axis_id
            for axis_id in _as_list(access.get("readable_axis_ids"))
            if axis_id not in observed
        ],
        "invalid_context_values": invalid,
        "context_axes": observed,
        "context_receipt_sha256": _canonical_hash(context_signature),
    }


def infer_regime_scope(
    *,
    raw_role: str,
    role_id: str,
    sub_sleeve_id: str,
    model: Mapping[str, Any],
) -> str:
    scope = _as_dict(model.get("scope_policy"))
    operational_sub_sleeves = {slug(item) for item in _as_list(scope.get("operational_sub_sleeves"))}
    hybrid_sub_sleeves = {slug(item) for item in _as_list(scope.get("hybrid_sub_sleeves"))}
    signal_roles = {str(item or "").strip() for item in _as_list(scope.get("signal_registry_roles"))}
    family = slug(sub_sleeve_id)
    if family in operational_sub_sleeves:
        return "operational_control"
    if str(raw_role or "").strip() in signal_roles or slug(role_id) == "signal":
        return "market_signal"
    if family in hybrid_sub_sleeves:
        return "hybrid"
    return "operational_control"


def _explicit_axis_values(
    source: Mapping[str, Any],
    axis_id: str,
    allowed: set[str],
) -> tuple[list[str], list[str]]:
    raw = _as_dict(source.get("regime_axes")).get(axis_id)
    values = _ordered_unique(_as_list(raw))
    return [item for item in values if item in allowed], [item for item in values if item not in allowed]


def _classify_regime_profile_single(
    *,
    row: Mapping[str, Any],
    module_spec: Mapping[str, Any],
    classification_text: str,
    raw_role: str,
    role_id: str,
    sub_sleeve_id: str,
    horizon_id: str,
    model: Mapping[str, Any],
) -> dict[str, Any]:
    axes = _axis_map(model)
    row_scope = slug(row.get("regime_scope"))
    module_scope = slug(module_spec.get("regime_scope"))
    invalid_explicit_scopes = _ordered_unique(
        item for item in (row_scope, module_scope) if item and item not in VALID_SCOPES
    )
    if row_scope in VALID_SCOPES:
        scope = row_scope
        scope_source = "registry_explicit"
    elif module_scope in VALID_SCOPES:
        scope = module_scope
        scope_source = "module_literal"
    else:
        scope = infer_regime_scope(
            raw_role=raw_role,
            role_id=role_id,
            sub_sleeve_id=sub_sleeve_id,
            model=model,
        )
        scope_source = "policy_rule"
    row_terms = _ordered_unique(_as_list(row.get("preferred_regimes")))
    module_terms = _ordered_unique(_as_list(module_spec.get("preferred_regimes")))
    raw_terms = _ordered_unique([*row_terms, *module_terms])
    maximum_values = max(_safe_int(model.get("maximum_values_per_axis"), 3), 1)
    global_wildcards = _ordered_unique(model.get("global_wildcard_tokens") or [])
    axis_results: dict[str, dict[str, Any]] = {}
    invalid_explicit_values: dict[str, list[str]] = {}

    for axis_id, axis in axes.items():
        values = _ordered_unique(axis.get("values") or [])
        allowed = set(values)
        unknown = slug(axis.get("unknown_value"))
        wildcard = slug(axis.get("wildcard_value"))
        not_applicable = slug(axis.get("not_applicable_value"))
        applicable = scope in {slug(item) for item in _as_list(axis.get("applicable_scopes"))}
        matched_evidence: dict[str, list[str]] = {}
        invalid_values: list[str] = []

        if not applicable:
            selected = [not_applicable]
            source = "not_applicable"
        else:
            selected, invalid_values = _explicit_axis_values(row, axis_id, allowed)
            source = "registry_explicit" if selected else ""
            if not selected:
                selected, module_invalid = _explicit_axis_values(module_spec, axis_id, allowed)
                invalid_values.extend(module_invalid)
                source = "module_literal" if selected else ""
            if not selected:
                selected, matched_evidence = _rule_values(row_terms, _as_list(axis.get("rules")))
                source = "registry_explicit" if selected else ""
            if not selected:
                selected, matched_evidence = _rule_values(module_terms, _as_list(axis.get("rules")))
                source = "module_literal" if selected else ""
            wildcard_tokens = _ordered_unique([*global_wildcards, *_as_list(axis.get("wildcard_tokens"))])
            wildcard_matches = [
                term
                for term in raw_terms
                if any(_token_matches(term, token) for token in wildcard_tokens)
            ]
            if not selected and wildcard_matches:
                selected = [wildcard]
                source = "registry_explicit" if row_terms else "module_literal"
                matched_evidence = {wildcard: wildcard_matches}
            if not selected:
                selected, matched_evidence = _rule_values([classification_text], _as_list(axis.get("rules")))
                source = "policy_rule" if selected else ""
            if not selected and any(_token_matches(classification_text, token) for token in wildcard_tokens):
                selected = [wildcard]
                source = "policy_wildcard"
            if not selected:
                selected = [unknown]
                source = "policy_fallback"

        selected = _ordered_unique(selected)[:maximum_values]
        if invalid_values:
            invalid_explicit_values[axis_id] = _ordered_unique(invalid_values)
        primary = selected[0]
        axis_results[axis_id] = {
            "values": selected,
            "primary_value": primary,
            "source": source,
            "confidence": round(SOURCE_CONFIDENCE.get(source, 0.4), 4),
            "matched_evidence": matched_evidence,
            "wildcard": wildcard in selected,
            "unknown": unknown in selected,
            "not_applicable": not_applicable in selected,
            "multi_value": len(selected) > 1,
        }

    quality_axes = [
        axis_id
        for axis_id, axis in axes.items()
        if scope in {slug(item) for item in _as_list(axis.get("required_for_scopes"))}
    ]
    known_axes = [
        axis_id
        for axis_id in quality_axes
        if not axis_results[axis_id]["unknown"] and not axis_results[axis_id]["not_applicable"]
    ]
    specific_axes = [axis_id for axis_id in known_axes if not axis_results[axis_id]["wildcard"]]
    wildcard_axes = [axis_id for axis_id in quality_axes if axis_results[axis_id]["wildcard"]]
    unknown_axes = [axis_id for axis_id in quality_axes if axis_results[axis_id]["unknown"]]
    multi_value_axes = [axis_id for axis_id in quality_axes if axis_results[axis_id]["multi_value"]]
    denominator = max(len(quality_axes), 1)
    coverage_ratio = len(known_axes) / denominator
    specificity_ratio = len(specific_axes) / denominator
    confidence = sum(axis_results[axis_id]["confidence"] for axis_id in quality_axes) / denominator
    selected_evidence_terms = _ordered_unique(
        term
        for axis in axis_results.values()
        for terms in _as_dict(axis.get("matched_evidence")).values()
        for term in _as_list(terms)
    )
    recognized_raw_terms = _ordered_unique(
        term
        for term in raw_terms
        if any(_token_matches(term, token) for token in global_wildcards)
        or any(
            _token_matches(term, token)
            for axis in axes.values()
            for rule in _as_list(axis.get("rules"))
            if isinstance(rule, dict)
            for token in _as_list(rule.get("tokens"))
        )
    )
    unmapped_raw_terms = [term for term in raw_terms if term not in set(recognized_raw_terms)]
    critical = {
        slug(item)
        for item in _as_list(_as_dict(model.get("critical_axes_by_scope")).get(scope))
    }
    critical_unknown_axes = sorted(critical.intersection(unknown_axes))
    minimum_coverage = _safe_float(
        _as_dict(model.get("minimum_axis_coverage_by_scope")).get(scope),
        0.0,
    )
    review_reasons: list[str] = []
    if coverage_ratio < minimum_coverage:
        review_reasons.append("regime_axis_coverage_below_scope_floor")
    if critical_unknown_axes:
        review_reasons.append("critical_regime_axes_unknown")
    if len(wildcard_axes) > _safe_int(model.get("maximum_wildcard_axes_before_review"), 2):
        review_reasons.append("regime_profile_overbroad")
    if len(multi_value_axes) > _safe_int(model.get("maximum_multi_value_axes_before_review"), 3):
        review_reasons.append("regime_profile_multi_axis_breadth_high")
    if invalid_explicit_values:
        review_reasons.append("invalid_explicit_regime_axis_values")
    if invalid_explicit_scopes:
        review_reasons.append("invalid_explicit_regime_scope")
    explicit_axes_present = bool(_as_dict(row.get("regime_axes")) or _as_dict(module_spec.get("regime_axes")))
    if unmapped_raw_terms and not explicit_axes_present:
        review_reasons.append("unmapped_preferred_regime_labels")

    cohort_axis_ids = [
        slug(item)
        for item in _as_list(_as_dict(model.get("cohort_axes_by_scope")).get(scope))
        if slug(item) in axis_results
    ]
    axis_short_names = _as_dict(model.get("axis_short_names"))
    cohort_components = [
        f"{slug(axis_short_names.get(axis_id) or axis_id)}_{axis_results[axis_id]['primary_value']}"
        for axis_id in cohort_axis_ids
    ]
    cohort_id = slug("__".join([horizon_id, *cohort_components]))
    signature = {
        "scope": scope,
        "scope_source": scope_source,
        "axes": {axis_id: axis_results[axis_id]["values"] for axis_id in REQUIRED_REGIME_AXES},
    }
    profile_id = f"regime_profile_{_canonical_hash(signature)[:16]}"
    return {
        "schema_version": 1,
        "model_id": str(model.get("model_id") or ""),
        "scope": scope,
        "scope_source": scope_source,
        "profile_id": profile_id,
        "raw_preferred_regimes": raw_terms[:16],
        "recognized_raw_regime_terms": recognized_raw_terms[:16],
        "selected_evidence_terms": selected_evidence_terms[:16],
        "unmapped_raw_regime_terms": unmapped_raw_terms[:16],
        "axes": axis_results,
        "quality_axes": quality_axes,
        "known_axes": known_axes,
        "specific_axes": specific_axes,
        "wildcard_axes": wildcard_axes,
        "unknown_axes": unknown_axes,
        "critical_unknown_axes": critical_unknown_axes,
        "multi_value_axes": multi_value_axes,
        "axis_coverage_ratio": round(coverage_ratio, 6),
        "axis_specificity_ratio": round(specificity_ratio, 6),
        "profile_confidence": round(confidence, 4),
        "requires_review": bool(review_reasons),
        "review_reasons": review_reasons,
        "invalid_explicit_values": invalid_explicit_values,
        "invalid_explicit_scopes": invalid_explicit_scopes,
        "cohort_axes": cohort_axis_ids,
        "cohort_components": cohort_components,
        "cohort_id": cohort_id,
        "signature_sha256": _canonical_hash(signature),
        "quality_axis_slot_count": len(quality_axes),
        "known_axis_slot_count": len(known_axes),
        "specific_axis_slot_count": len(specific_axes),
    }


def _declared_regime_scenarios(
    row: Mapping[str, Any],
    module_spec: Mapping[str, Any],
) -> tuple[list[Any], str, list[str]]:
    for source, source_name in ((row, "registry_explicit"), (module_spec, "module_literal")):
        if "regime_scenarios" not in source:
            continue
        raw = source.get("regime_scenarios")
        if not isinstance(raw, list):
            return [], source_name, ["regime_scenarios_not_a_list"]
        return list(raw), source_name, []
    return [], "", []


def _scenario_axis_summary(
    profiles: Iterable[Mapping[str, Any]],
    model: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    profile_rows = list(profiles)
    result: dict[str, dict[str, Any]] = {}
    for axis_id, axis in _axis_map(model).items():
        rows = [_as_dict(_as_dict(profile.get("axes")).get(axis_id)) for profile in profile_rows]
        values = _ordered_unique(
            value for row in rows for value in _as_list(row.get("values"))
        )
        result[axis_id] = {
            "values": values,
            "primary_value": values[0] if values else slug(axis.get("unknown_value")),
            "source": "scenario_partition",
            "confidence": round(
                sum(_safe_float(row.get("confidence"), 0.0) for row in rows)
                / max(len(rows), 1),
                4,
            ),
            "matched_evidence": {
                str(index): _as_dict(row.get("matched_evidence"))
                for index, row in enumerate(rows)
                if _as_dict(row.get("matched_evidence"))
            },
            "wildcard": any(bool(row.get("wildcard", False)) for row in rows),
            "unknown": any(bool(row.get("unknown", False)) for row in rows),
            "not_applicable": bool(rows) and all(
                bool(row.get("not_applicable", False)) for row in rows
            ),
            "multi_value": any(bool(row.get("multi_value", False)) for row in rows),
            "partitioned_value_count": len(values),
        }
    return result


def classify_regime_profile(
    *,
    row: Mapping[str, Any],
    module_spec: Mapping[str, Any],
    classification_text: str,
    raw_role: str,
    role_id: str,
    sub_sleeve_id: str,
    horizon_id: str,
    model: Mapping[str, Any],
) -> dict[str, Any]:
    base = _classify_regime_profile_single(
        row=row,
        module_spec=module_spec,
        classification_text=classification_text,
        raw_role=raw_role,
        role_id=role_id,
        sub_sleeve_id=sub_sleeve_id,
        horizon_id=horizon_id,
        model=model,
    )
    raw_scenarios, scenario_source, contract_errors = _declared_regime_scenarios(
        row,
        module_spec,
    )
    if not scenario_source:
        return base

    contract = _as_dict(model.get("scenario_partition_contract"))
    minimum_scenarios = max(_safe_int(contract.get("minimum_scenarios_per_profile"), 2), 1)
    maximum_scenarios = max(
        _safe_int(contract.get("maximum_scenarios_per_profile"), 12),
        minimum_scenarios,
    )
    maximum_axis_values = max(
        _safe_int(contract.get("maximum_values_per_axis_per_scenario"), 1),
        1,
    )
    if not minimum_scenarios <= len(raw_scenarios) <= maximum_scenarios:
        contract_errors.append("regime_scenario_count_out_of_bounds")
    if bool(contract.get("require_explicit_scope", True)) and str(
        base.get("scope_source") or ""
    ) not in {"registry_explicit", "module_literal"}:
        contract_errors.append("regime_scenario_parent_scope_not_explicit")

    scenario_rows: list[dict[str, Any]] = []
    scenario_ids: list[str] = []
    scenario_review_reasons: list[str] = []
    axis_ids = set(_axis_map(model))
    for index, raw_scenario in enumerate(raw_scenarios):
        if not isinstance(raw_scenario, dict):
            contract_errors.append(f"regime_scenario_{index}_not_an_object")
            continue
        scenario_id = slug(raw_scenario.get("scenario_id"))
        if not scenario_id:
            contract_errors.append(f"regime_scenario_{index}_id_missing")
            continue
        if scenario_id in scenario_ids:
            contract_errors.append(f"regime_scenario_{scenario_id}_id_duplicate")
            continue
        scenario_ids.append(scenario_id)
        scenario_scope = slug(raw_scenario.get("regime_scope") or base.get("scope"))
        if scenario_scope not in VALID_SCOPES:
            contract_errors.append(f"regime_scenario_{scenario_id}_scope_invalid")
            continue
        if bool(contract.get("require_same_scope", True)) and scenario_scope != base.get("scope"):
            contract_errors.append(f"regime_scenario_{scenario_id}_scope_mismatch")

        raw_axes = raw_scenario.get("regime_axes")
        scenario_axes = _as_dict(raw_axes)
        if bool(contract.get("require_explicit_axes", True)) and not scenario_axes:
            contract_errors.append(f"regime_scenario_{scenario_id}_axes_missing")
        unknown_axis_ids = sorted(slug(key) for key in scenario_axes if slug(key) not in axis_ids)
        if unknown_axis_ids:
            contract_errors.append(f"regime_scenario_{scenario_id}_axis_unknown")
        for axis_id, raw_values in scenario_axes.items():
            if len(_ordered_unique(_as_list(raw_values))) > maximum_axis_values:
                contract_errors.append(
                    f"regime_scenario_{scenario_id}_{slug(axis_id)}_values_over_limit"
                )

        preferred = _ordered_unique(
            _as_list(raw_scenario.get("preferred_regimes")) or [scenario_id]
        )
        scenario_profile = _classify_regime_profile_single(
            row={
                "bot_id": str(row.get("bot_id") or module_spec.get("bot_id") or ""),
                "regime_scope": scenario_scope,
                "regime_axes": scenario_axes,
                "preferred_regimes": preferred,
            },
            module_spec={},
            classification_text=" ".join(
                [scenario_id, str(raw_scenario.get("description") or ""), *preferred]
            ),
            raw_role=raw_role,
            role_id=role_id,
            sub_sleeve_id=sub_sleeve_id,
            horizon_id=horizon_id,
            model=model,
        )
        if scenario_profile.get("requires_review", False):
            scenario_review_reasons.extend(
                f"{scenario_id}:{reason}"
                for reason in _as_list(scenario_profile.get("review_reasons"))
            )
        scenario_rows.append(
            {
                "scenario_id": scenario_id,
                "description": str(raw_scenario.get("description") or "").strip(),
                "preferred_regimes": preferred,
                "profile": scenario_profile,
            }
        )

    if not scenario_rows:
        contract_errors.append("regime_scenario_partition_has_no_valid_scenarios")
    contract_errors = _ordered_unique(contract_errors)
    scenario_review_reasons = _ordered_unique(scenario_review_reasons)
    profiles = [_as_dict(item.get("profile")) for item in scenario_rows]
    quality_slot_count = sum(_safe_int(item.get("quality_axis_slot_count")) for item in profiles)
    known_slot_count = sum(_safe_int(item.get("known_axis_slot_count")) for item in profiles)
    specific_slot_count = sum(_safe_int(item.get("specific_axis_slot_count")) for item in profiles)
    denominator = max(quality_slot_count, 1)
    quality_axes = _ordered_unique(
        axis_id for item in profiles for axis_id in _as_list(item.get("quality_axes"))
    )
    known_axes = _ordered_unique(
        axis_id for item in profiles for axis_id in _as_list(item.get("known_axes"))
    )
    specific_axes = _ordered_unique(
        axis_id for item in profiles for axis_id in _as_list(item.get("specific_axes"))
    )
    wildcard_axes = _ordered_unique(
        axis_id for item in profiles for axis_id in _as_list(item.get("wildcard_axes"))
    )
    unknown_axes = _ordered_unique(
        axis_id for item in profiles for axis_id in _as_list(item.get("unknown_axes"))
    )
    critical_unknown_axes = _ordered_unique(
        axis_id for item in profiles for axis_id in _as_list(item.get("critical_unknown_axes"))
    )
    multi_value_axes = _ordered_unique(
        axis_id for item in profiles for axis_id in _as_list(item.get("multi_value_axes"))
    )
    review_reasons = _ordered_unique(
        [
            "invalid_regime_scenario_contract" if contract_errors else "",
            "regime_scenario_requires_review" if scenario_review_reasons else "",
        ]
    )
    signature = {
        "scope": base.get("scope"),
        "scenario_contract_version": str(contract.get("version") or ""),
        "scenarios": [
            {
                "scenario_id": item["scenario_id"],
                "profile_signature_sha256": _as_dict(item.get("profile")).get(
                    "signature_sha256"
                ),
            }
            for item in scenario_rows
        ],
    }
    signature_sha256 = _canonical_hash(signature)
    profile_id = f"regime_profile_{signature_sha256[:16]}"
    return {
        **base,
        "profile_id": profile_id,
        "axes": _scenario_axis_summary(profiles, model),
        "quality_axes": quality_axes,
        "known_axes": known_axes,
        "specific_axes": specific_axes,
        "wildcard_axes": wildcard_axes,
        "unknown_axes": unknown_axes,
        "critical_unknown_axes": critical_unknown_axes,
        "multi_value_axes": multi_value_axes,
        "axis_coverage_ratio": round(known_slot_count / denominator, 6),
        "axis_specificity_ratio": round(specific_slot_count / denominator, 6),
        "profile_confidence": round(
            sum(_safe_float(item.get("profile_confidence")) for item in profiles)
            / max(len(profiles), 1),
            4,
        ),
        "requires_review": bool(review_reasons),
        "review_reasons": review_reasons,
        "cohort_id": slug(f"{horizon_id}__scenario_set_{signature_sha256[:8]}"),
        "signature_sha256": signature_sha256,
        "quality_axis_slot_count": quality_slot_count,
        "known_axis_slot_count": known_slot_count,
        "specific_axis_slot_count": specific_slot_count,
        "scenario_partitioned": True,
        "scenario_partition_contract_version": str(contract.get("version") or ""),
        "scenario_partition_source": scenario_source,
        "scenario_count": len(scenario_rows),
        "regime_scenarios": scenario_rows,
        "scenario_contract_errors": contract_errors,
        "scenario_review_reasons": scenario_review_reasons,
    }


def _evaluate_single_regime_compatibility(
    profile: Mapping[str, Any],
    context: Mapping[str, Any],
    model: Mapping[str, Any],
) -> dict[str, Any]:
    axes = _axis_map(model)
    compatibility = _as_dict(model.get("compatibility_policy"))
    profile_axes = _as_dict(profile.get("axes"))
    context_axes = _as_dict(context.get("axes")) or dict(context)
    wildcard_score = _safe_float(compatibility.get("wildcard_profile_axis_score"), 0.65)
    unknown_profile_score = _safe_float(compatibility.get("unknown_profile_axis_score"), 0.25)
    unknown_context_score = _safe_float(compatibility.get("unknown_context_axis_score"), 0.5)
    profile_scope = slug(profile.get("scope"))
    minimum_axes = max(
        _safe_int(
            _as_dict(compatibility.get("minimum_context_axes_by_scope")).get(profile_scope),
            _safe_int(compatibility.get("minimum_context_axes"), 2),
        ),
        1,
    )
    minimum_score = _safe_float(compatibility.get("minimum_compatibility_score"), 0.55)
    hard_mismatch_axes = {
        slug(item)
        for item in _as_list(
            _as_dict(compatibility.get("hard_mismatch_axes_by_scope")).get(profile_scope)
        )
    }
    comparisons: list[dict[str, Any]] = []
    invalid_context: dict[str, list[str]] = {}

    for axis_id, axis in axes.items():
        if axis_id not in context_axes:
            continue
        profile_axis = _as_dict(profile_axes.get(axis_id))
        if not profile_axis or bool(profile_axis.get("not_applicable", False)):
            continue
        allowed = set(_ordered_unique(axis.get("values") or []))
        context_values = _ordered_unique(_as_list(context_axes.get(axis_id)))
        invalid = [item for item in context_values if item not in allowed]
        if invalid:
            invalid_context[axis_id] = invalid
        context_values = [item for item in context_values if item in allowed]
        if not context_values:
            continue
        profile_values = set(_ordered_unique(profile_axis.get("values") or []))
        wildcard = slug(axis.get("wildcard_value"))
        unknown = slug(axis.get("unknown_value"))
        context_unknown = unknown in context_values
        if unknown in profile_values:
            axis_score = unknown_profile_score
            reason = "profile_axis_unknown"
        elif wildcard in profile_values:
            axis_score = wildcard_score
            reason = "profile_axis_wildcard"
        elif context_unknown:
            axis_score = unknown_context_score
            reason = "context_axis_unknown"
        elif profile_values.intersection(context_values):
            axis_score = 1.0
            reason = "axis_match"
        else:
            axis_score = 0.0
            reason = "axis_mismatch"
        comparisons.append(
            {
                "axis_id": axis_id,
                "profile_values": sorted(profile_values),
                "context_values": context_values,
                "score": round(axis_score, 6),
                "weight": _safe_float(axis.get("compatibility_weight"), 1.0),
                "reason": reason,
            }
        )

    total_weight = sum(row["weight"] for row in comparisons)
    score = (
        sum(row["score"] * row["weight"] for row in comparisons) / total_weight
        if total_weight > 0.0
        else 0.0
    )
    enough_axes = len(comparisons) >= minimum_axes
    hard_mismatch_axis_ids = [
        str(row.get("axis_id") or "")
        for row in comparisons
        if row.get("reason") == "axis_mismatch" and row.get("axis_id") in hard_mismatch_axes
    ]
    compatible = bool(
        enough_axes
        and not invalid_context
        and not hard_mismatch_axis_ids
        and score >= minimum_score
    )
    if invalid_context:
        reason = "invalid_regime_context"
    elif not enough_axes:
        reason = "insufficient_regime_context"
    elif hard_mismatch_axis_ids:
        reason = "critical_regime_axis_mismatch"
    elif score < minimum_score:
        reason = "regime_compatibility_below_floor"
    else:
        reason = "regime_compatible"
    return {
        "schema_version": 1,
        "mode": "shadow_only",
        "compatible": compatible,
        "reason": reason,
        "score": round(score, 6),
        "compared_axis_count": len(comparisons),
        "minimum_context_axes": minimum_axes,
        "minimum_compatibility_score": minimum_score,
        "invalid_context_values": invalid_context,
        "hard_mismatch_axis_ids": hard_mismatch_axis_ids,
        "comparisons": comparisons,
        "authority": {
            "paper_execution_authority": False,
            "live_execution_authority": False,
            "order_payload_created": False,
        },
    }


def evaluate_regime_compatibility(
    profile: Mapping[str, Any],
    context: Mapping[str, Any],
    model: Mapping[str, Any],
) -> dict[str, Any]:
    """Select the best bounded scenario without granting execution authority."""

    metadata_view = build_regime_metadata_view(profile, context, model)
    if not bool(profile.get("scenario_partitioned", False)):
        return {
            **_evaluate_single_regime_compatibility(profile, context, model),
            "metadata_access": metadata_view,
        }

    scenarios = [
        row for row in _as_list(profile.get("regime_scenarios")) if isinstance(row, dict)
    ]
    contract_errors = _ordered_unique(profile.get("scenario_contract_errors") or [])
    if contract_errors or not scenarios:
        return {
            "schema_version": 1,
            "mode": "shadow_only",
            "compatible": False,
            "reason": "invalid_regime_scenario_contract",
            "score": 0.0,
            "compared_axis_count": 0,
            "minimum_context_axes": 0,
            "minimum_compatibility_score": _safe_float(
                _as_dict(model.get("compatibility_policy")).get(
                    "minimum_compatibility_score"
                ),
                0.55,
            ),
            "invalid_context_values": {},
            "hard_mismatch_axis_ids": [],
            "comparisons": [],
            "scenario_partition_applied": True,
            "scenario_count": len(scenarios),
            "evaluated_scenario_count": 0,
            "selected_scenario_id": "",
            "scenario_contract_errors": contract_errors
            or ["regime_scenario_partition_has_no_valid_scenarios"],
            "scenario_results": [],
            "metadata_access": metadata_view,
            "authority": {
                "paper_execution_authority": False,
                "live_execution_authority": False,
                "order_payload_created": False,
            },
        }

    evaluations: list[dict[str, Any]] = []
    for scenario in scenarios:
        scenario_id = slug(scenario.get("scenario_id"))
        scenario_profile = _as_dict(scenario.get("profile"))
        result = _evaluate_single_regime_compatibility(scenario_profile, context, model)
        evaluations.append({"scenario_id": scenario_id, "result": result})
    ordered = sorted(
        evaluations,
        key=lambda item: (
            not bool(_as_dict(item.get("result")).get("compatible", False)),
            -_safe_float(_as_dict(item.get("result")).get("score")),
            str(item.get("scenario_id") or ""),
        ),
    )
    selected = ordered[0]
    selected_result = _as_dict(selected.get("result"))
    compatible = bool(selected_result.get("compatible", False))
    reasons = {
        str(_as_dict(item.get("result")).get("reason") or "") for item in evaluations
    }
    if compatible:
        reason = "regime_scenario_compatible"
    elif reasons == {"invalid_regime_context"}:
        reason = "invalid_regime_context"
    elif reasons == {"insufficient_regime_context"}:
        reason = "insufficient_regime_context"
    else:
        reason = "no_regime_scenario_compatible"
    return {
        **selected_result,
        "compatible": compatible,
        "reason": reason,
        "scenario_partition_applied": True,
        "scenario_count": len(scenarios),
        "evaluated_scenario_count": len(evaluations),
        "selected_scenario_id": str(selected.get("scenario_id") or ""),
        "selected_scenario_reason": str(selected_result.get("reason") or ""),
        "scenario_contract_errors": [],
        "scenario_results": [
            {
                "scenario_id": str(item.get("scenario_id") or ""),
                "compatible": bool(_as_dict(item.get("result")).get("compatible", False)),
                "reason": str(_as_dict(item.get("result")).get("reason") or ""),
                "score": _safe_float(_as_dict(item.get("result")).get("score")),
                "compared_axis_count": _safe_int(
                    _as_dict(item.get("result")).get("compared_axis_count")
                ),
                "hard_mismatch_axis_ids": _as_list(
                    _as_dict(item.get("result")).get("hard_mismatch_axis_ids")
                ),
            }
            for item in sorted(evaluations, key=lambda item: str(item.get("scenario_id") or ""))
        ],
        "metadata_access": metadata_view,
        "authority": {
            "paper_execution_authority": False,
            "live_execution_authority": False,
            "order_payload_created": False,
        },
    }
