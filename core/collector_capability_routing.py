"""Deterministic, execution-free routing for collector capabilities."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping


EXPECTED_SAFETY_FLAGS = (
    "changes_runtime_decisions",
    "launches_collectors",
    "fetches_external_data",
    "mutates_bot_registry",
    "rewrites_historical_outcomes",
    "paper_execution_authority",
    "live_execution_authority",
    "automatic_promotion_authority",
    "profitability_guaranteed",
)
REQUIRED_ADMISSION_FLAGS = (
    "require_named_capability_gap",
    "require_verified_source_provenance",
    "require_point_in_time_timestamp_contract",
    "require_shared_cache_plan",
    "require_freshness_and_fallback_contract",
    "require_resource_budget_clearance",
    "require_failure_isolation_boundary",
    "require_incremental_out_of_sample_value_for_promotion",
    "require_human_approval",
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


def canonical_hash(value: Any) -> str:
    raw = json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _parse_timestamp(value: Any) -> datetime | None:
    text = str(value or "").strip().replace("Z", "+00:00")
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text)
    except (TypeError, ValueError):
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _payload_timestamp(payload: Mapping[str, Any], path: Path) -> datetime | None:
    for key in ("timestamp_utc", "updated_at_utc", "created_at_utc", "ended_utc", "started_utc"):
        parsed = _parse_timestamp(payload.get(key))
        if parsed is not None:
            return parsed
    try:
        return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
    except OSError:
        return None


_MISSING = object()


def _values_at_path(payload: Any, dotted_path: str) -> list[Any]:
    parts = [part for part in str(dotted_path or "").split(".") if part]

    def walk(value: Any, index: int) -> list[Any]:
        if index >= len(parts):
            return [value]
        part = parts[index]
        if part == "*":
            if isinstance(value, Mapping):
                children = value.values()
            elif isinstance(value, list):
                children = value
            else:
                children = []
            return [item for child in children for item in walk(child, index + 1)]
        if isinstance(value, Mapping) and part in value:
            return walk(value[part], index + 1)
        if isinstance(value, list) and part.isdigit():
            position = int(part)
            if 0 <= position < len(value):
                return walk(value[position], index + 1)
        return []

    return walk(payload, 0)


def _value_at_path(payload: Any, dotted_path: str) -> Any:
    values = _values_at_path(payload, dotted_path)
    return values[0] if values else _MISSING


def _evaluate_capability_proofs(
    producer: Mapping[str, Any],
    payload: Any,
    *,
    producer_usable: bool,
) -> tuple[list[str], dict[str, dict[str, Any]]]:
    capabilities = _ordered_unique(_as_list(producer.get("capabilities")))
    evidence_contract = _as_dict(producer.get("capability_evidence_contract"))
    proof_specs = _as_dict(producer.get("capability_proofs"))
    usable: list[str] = []
    proof_rows: dict[str, dict[str, Any]] = {}

    evidence_mode = str(evidence_contract.get("mode") or "")
    materialized_rows: dict[str, dict[str, Any]] = {}
    if evidence_mode == "capability_rows":
        rows = _value_at_path(payload, str(evidence_contract.get("path") or "capabilities"))
        id_field = str(evidence_contract.get("id_field") or "capability_id")
        if isinstance(rows, list):
            materialized_rows = {
                str(row.get(id_field) or ""): row
                for row in rows
                if isinstance(row, dict) and str(row.get(id_field) or "")
            }

    for capability_id in capabilities:
        mode = "producer_health_inherited"
        passed = bool(producer_usable)
        details: dict[str, Any] = {}
        if evidence_mode == "capability_rows":
            mode = "materialized_capability_receipt"
            evidence = _as_dict(materialized_rows.get(capability_id))
            usable_field = str(evidence_contract.get("usable_field") or "usable")
            receipt_field = str(evidence_contract.get("receipt_field") or "proof_receipt_sha256")
            receipt = str(evidence.get(receipt_field) or "").strip()
            required_equals = _as_dict(evidence_contract.get("required_equals"))
            contract_passed = all(
                _value_at_path(payload, path) == expected
                for path, expected in required_equals.items()
            )
            passed = bool(
                producer_usable
                and evidence
                and evidence.get(usable_field) is True
                and (receipt or not bool(evidence_contract.get("require_receipt", True)))
                and contract_passed
            )
            details = {
                "evidence_row_present": bool(evidence),
                "evidence_row_usable": evidence.get(usable_field) is True,
                "receipt_present": bool(receipt),
                "contract_equality_check_count": len(required_equals),
                "contract_equality_passed": contract_passed,
            }
        elif capability_id in proof_specs:
            mode = "field_level_payload_proof"
            spec = _as_dict(proof_specs.get(capability_id))
            paths = _ordered_unique(_as_list(spec.get("paths")))
            matches = [bool(_values_at_path(payload, path)) for path in paths]
            proof_mode = str(spec.get("mode") or "all").lower()
            path_passed = bool(matches and (any(matches) if proof_mode == "any" else all(matches)))
            equals = _as_dict(spec.get("equals"))
            equality_passed = all(_value_at_path(payload, path) == expected for path, expected in equals.items())
            passed = bool(producer_usable and path_passed and equality_passed)
            details = {
                "path_mode": proof_mode,
                "required_paths": paths,
                "present_path_count": sum(matches),
                "required_path_count": len(paths),
                "equality_check_count": len(equals),
                "proof_family": str(spec.get("proof_family") or "payload_field"),
            }
        if passed:
            usable.append(capability_id)
        proof_rows[capability_id] = {
            "mode": mode,
            "passed": passed,
            **details,
        }
    return usable, proof_rows


def _mapping_age_minutes(payload: Mapping[str, Any], *, now: datetime) -> float | None:
    for key in ("timestamp_utc", "updated_at_utc", "created_at_utc", "ended_utc", "started_utc"):
        parsed = _parse_timestamp(payload.get(key))
        if parsed is not None:
            return max((now - parsed).total_seconds() / 60.0, 0.0)
    return None


def flatten_capabilities(catalog: Mapping[str, Any]) -> tuple[list[str], dict[str, dict[str, Any]]]:
    capability_ids: list[str] = []
    capability_planes: dict[str, dict[str, Any]] = {}
    for raw_plane in _as_list(catalog.get("planes")):
        plane = _as_dict(raw_plane)
        for raw_capability in _as_list(plane.get("capabilities")):
            capability_id = str(raw_capability or "").strip()
            if capability_id:
                capability_ids.append(capability_id)
                capability_planes.setdefault(capability_id, plane)
    return capability_ids, capability_planes


def validate_catalog(catalog: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    contract = _as_dict(catalog.get("catalog_contract"))
    axes = _as_dict(catalog.get("capability_axes"))
    policy = _as_dict(catalog.get("routing_policy"))
    safety = _as_dict(catalog.get("safety_contract"))
    admission = _as_dict(catalog.get("physical_producer_admission"))
    planes = [_as_dict(row) for row in _as_list(catalog.get("planes"))]
    producers = [_as_dict(row) for row in _as_list(catalog.get("producers"))]

    if _safe_int(catalog.get("schema_version")) != 1:
        errors.append("capability_catalog_schema_version_invalid")
    if str(catalog.get("catalog_id") or "") != "collector_capability_catalog_v1":
        errors.append("capability_catalog_id_invalid")
    if str(catalog.get("operating_mode") or "") != "metadata_subscription_shadow_only":
        errors.append("capability_catalog_not_shadow_only")

    required_plane_count = _safe_int(contract.get("required_plane_count"), 25)
    minimum_capability_count = _safe_int(contract.get("minimum_capability_count"), 250)
    if len(planes) != required_plane_count:
        errors.append("capability_catalog_plane_count_invalid")
    plane_ids = [str(row.get("plane_id") or "").strip() for row in planes]
    if any(not item for item in plane_ids) or len(plane_ids) != len(set(plane_ids)):
        errors.append("capability_catalog_plane_ids_invalid")
    capability_ids, _ = flatten_capabilities(catalog)
    capability_set = set(capability_ids)
    if len(capability_ids) < minimum_capability_count:
        errors.append("capability_catalog_capability_count_below_floor")
    if len(capability_ids) != len(capability_set):
        errors.append("capability_catalog_duplicate_capability_ids")

    allowed_source_kinds = set(_ordered_unique(_as_list(axes.get("source_kinds"))))
    allowed_cadences = set(_ordered_unique(_as_list(axes.get("cadences"))))
    allowed_resources = set(_ordered_unique(_as_list(axes.get("resource_classes"))))
    producer_ids = [str(row.get("producer_id") or "").strip() for row in producers]
    if any(not item for item in producer_ids) or len(producer_ids) != len(set(producer_ids)):
        errors.append("capability_catalog_producer_ids_invalid")
    collector_names: list[str] = []
    for producer in producers:
        producer_id = str(producer.get("producer_id") or "").strip()
        kind = str(producer.get("producer_kind") or "").strip()
        if kind not in {"collector", "artifact"}:
            errors.append(f"capability_producer_kind_invalid:{producer_id}")
        if str(producer.get("source_kind") or "") not in allowed_source_kinds:
            errors.append(f"capability_producer_source_kind_invalid:{producer_id}")
        if str(producer.get("cadence") or "") not in allowed_cadences:
            errors.append(f"capability_producer_cadence_invalid:{producer_id}")
        if str(producer.get("resource_class") or "") not in allowed_resources:
            errors.append(f"capability_producer_resource_class_invalid:{producer_id}")
        if _safe_float(producer.get("max_age_minutes"), 0.0) <= 0.0:
            errors.append(f"capability_producer_freshness_invalid:{producer_id}")
        unknown = sorted(set(_ordered_unique(_as_list(producer.get("capabilities")))) - capability_set)
        if unknown:
            errors.append(f"capability_producer_unknown_capabilities:{producer_id}")
        declared = set(_ordered_unique(_as_list(producer.get("capabilities"))))
        proof_capabilities = set(_as_dict(producer.get("capability_proofs")))
        if not proof_capabilities.issubset(declared):
            errors.append(f"capability_producer_proof_not_declared:{producer_id}")
        evidence_contract = _as_dict(producer.get("capability_evidence_contract"))
        if evidence_contract and str(evidence_contract.get("mode") or "") != "capability_rows":
            errors.append(f"capability_producer_evidence_contract_invalid:{producer_id}")
        cache = _as_dict(producer.get("cache_contract"))
        if str(cache.get("mode") or "") != "shared_snapshot" or not _as_list(cache.get("deduplicate_by")):
            errors.append(f"capability_producer_cache_contract_invalid:{producer_id}")
        if not str(producer.get("fallback_policy") or "").strip():
            errors.append(f"capability_producer_fallback_policy_missing:{producer_id}")
        if kind == "collector":
            name = str(producer.get("collector_name") or "").strip()
            if not name:
                errors.append(f"capability_producer_collector_name_missing:{producer_id}")
            collector_names.append(name)
        elif not str(producer.get("artifact_path") or "").strip():
            errors.append(f"capability_producer_artifact_path_missing:{producer_id}")
    if len(collector_names) != len(set(collector_names)):
        errors.append("capability_catalog_duplicate_collector_mappings")

    for mapping_key in ("scope_required_capabilities", "role_required_capabilities", "regime_axis_capability_map"):
        for key, raw_values in _as_dict(catalog.get(mapping_key)).items():
            unknown = sorted(set(_ordered_unique(_as_list(raw_values))) - capability_set)
            if unknown:
                errors.append(f"capability_catalog_unknown_{mapping_key}:{key}")
    for plane in planes:
        plane_id = str(plane.get("plane_id") or "")
        routing = _as_dict(plane.get("routing"))
        plane_caps = set(_ordered_unique(_as_list(plane.get("capabilities"))))
        if not plane_caps:
            errors.append(f"capability_plane_empty:{plane_id}")
        if not set(_ordered_unique(_as_list(routing.get("required_when_matched")))).issubset(plane_caps):
            errors.append(f"capability_plane_required_not_in_plane:{plane_id}")

    for key in EXPECTED_SAFETY_FLAGS:
        if safety.get(key) is not False:
            errors.append(f"capability_safety_{key}_must_be_false")
    for key in REQUIRED_ADMISSION_FLAGS:
        if admission.get(key) is not True:
            errors.append(f"capability_admission_{key}_disabled")
    if _safe_int(admission.get("minimum_subscribed_bot_count"), 0) < 1:
        errors.append("capability_admission_minimum_subscribed_bot_count_invalid")
    if str(policy.get("profile_mode") or "") != "content_addressed_shared_subscription_profiles":
        errors.append("capability_profile_mode_invalid")
    if policy.get("external_fetch_launch_authority") is not False:
        errors.append("capability_router_external_fetch_authority_enabled")
    for key in (
        "fail_closed_on_invalid_catalog",
        "fail_closed_to_collect_only_on_missing_required_capability",
        "cache_reuse_required",
    ):
        if policy.get(key) is not True:
            errors.append(f"capability_routing_policy_{key}_disabled")
    return _ordered_unique(errors)


def _producer_rows(
    project_root: Path,
    catalog: Mapping[str, Any],
    collector_contracts: Mapping[str, Any],
    *,
    now: datetime,
) -> list[dict[str, Any]]:
    collector_rows = {
        str(row.get("name") or ""): row
        for row in _as_list(collector_contracts.get("rows"))
        if isinstance(row, dict) and str(row.get("name") or "")
    }
    rows: list[dict[str, Any]] = []
    for raw_producer in _as_list(catalog.get("producers")):
        producer = _as_dict(raw_producer)
        producer_id = str(producer.get("producer_id") or "")
        kind = str(producer.get("producer_kind") or "")
        capabilities = _ordered_unique(_as_list(producer.get("capabilities")))
        row: dict[str, Any] = {
            "producer_id": producer_id,
            "producer_kind": kind,
            "source_kind": str(producer.get("source_kind") or ""),
            "cadence": str(producer.get("cadence") or ""),
            "resource_class": str(producer.get("resource_class") or ""),
            "capability_count": len(capabilities),
            "capabilities": capabilities,
            "usable_capabilities": [],
            "capability_proofs": {},
            "available": False,
            "fresh": False,
            "usable": False,
            "published_ok": None,
            "published_status": "missing",
            "age_minutes": None,
        }
        if kind == "collector":
            name = str(producer.get("collector_name") or "")
            contract_row = _as_dict(collector_rows.get(name))
            contract_usable = bool(contract_row.get("contract_ok", False))
            payload: Any = {}
            if producer.get("capability_proofs") or producer.get("capability_evidence_contract"):
                payload_path = Path(str(contract_row.get("payload_path") or ""))
                try:
                    payload = json.loads(payload_path.read_text(encoding="utf-8"))
                except (OSError, ValueError, TypeError):
                    payload = {}
            usable_capabilities, capability_proofs = _evaluate_capability_proofs(
                producer,
                payload,
                producer_usable=contract_usable,
            )
            row.update(
                {
                    "collector_name": name,
                    "available": bool(contract_row),
                    "fresh": bool(contract_row.get("fresh", False)),
                    "usable": contract_usable,
                    "usable_capabilities": usable_capabilities,
                    "capability_proofs": capability_proofs,
                    "published_ok": bool(contract_row.get("ok", False)) if contract_row else None,
                    "published_status": "ready" if contract_usable else "degraded",
                    "age_minutes": round(_safe_float(contract_row.get("age_seconds")) / 60.0, 3)
                    if contract_row.get("age_seconds") is not None
                    else None,
                    "required_collector": bool(contract_row.get("required", False)),
                    "collector_contract_ok": bool(contract_row.get("contract_ok", False)),
                }
            )
        elif kind == "artifact":
            relative_path = str(producer.get("artifact_path") or "")
            path = project_root / relative_path
            payload: Any = {}
            try:
                loaded = json.loads(path.read_text(encoding="utf-8"))
                payload = loaded
            except (OSError, ValueError, TypeError):
                payload = {}
            payload_mapping = payload if isinstance(payload, Mapping) else {}
            timestamp = _payload_timestamp(payload_mapping, path) if payload else None
            age = max((now - timestamp).total_seconds() / 60.0, 0.0) if timestamp is not None else None
            fresh = bool(age is not None and age <= _safe_float(producer.get("max_age_minutes"), 0.0))
            artifact_usable = bool(payload and fresh)
            usable_capabilities, capability_proofs = _evaluate_capability_proofs(
                producer,
                payload,
                producer_usable=artifact_usable,
            )
            published_status = str(
                payload_mapping.get("overall_status") or payload_mapping.get("status") or ""
            )
            row.update(
                {
                    "artifact_path": relative_path,
                    "available": bool(payload),
                    "fresh": fresh,
                    "usable": artifact_usable,
                    "usable_capabilities": usable_capabilities,
                    "capability_proofs": capability_proofs,
                    "published_ok": payload_mapping.get("ok") if "ok" in payload_mapping else None,
                    "published_status": published_status or ("ready" if payload else "missing"),
                    "age_minutes": round(age, 3) if age is not None else None,
                }
            )
        rows.append(row)
    return rows


def _assignment_text(assignment: Mapping[str, Any]) -> str:
    values: list[str] = []
    for key in (
        "bot_id",
        "sleeve_id",
        "sub_sleeve_id",
        "horizon_id",
        "role_id",
        "cell_id",
        "correlation_cluster_id",
    ):
        values.append(str(assignment.get(key) or ""))
    values.extend(str(value or "") for value in _as_list(assignment.get("regime_ids")))
    values.extend(str(value or "") for value in _as_list(assignment.get("preferred_regimes")))
    return " ".join(values).lower().replace("-", "_")


def _matched_plane_score(plane: Mapping[str, Any], assignment: Mapping[str, Any], text: str) -> int:
    routing = _as_dict(plane.get("routing"))
    scope = str(assignment.get("regime_scope") or "market_signal")
    role = str(assignment.get("role_id") or "signal")
    scopes = set(_ordered_unique(_as_list(routing.get("target_scopes"))))
    roles = set(_ordered_unique(_as_list(routing.get("target_roles"))))
    tokens = _ordered_unique(_as_list(routing.get("target_tokens")))
    scope_match = scope in scopes
    role_match = role in roles
    token_match = any(token.lower().replace("-", "_") in text for token in tokens)
    if not (scope_match or role_match or token_match):
        return 0
    return (4 if token_match else 0) + (2 if role_match else 0) + (1 if scope_match else 0)


def _unknown_regime_axes(assignment: Mapping[str, Any]) -> list[str]:
    profile = _as_dict(assignment.get("regime_profile"))
    access = _as_dict(assignment.get("regime_metadata_access"))
    explicit = _ordered_unique(_as_list(access.get("runtime_context_required_axis_ids")))
    if explicit:
        return explicit
    result: list[str] = []
    for axis_id, raw_axis in _as_dict(profile.get("axes")).items():
        axis = _as_dict(raw_axis)
        if bool(axis.get("unknown", False)):
            result.append(str(axis_id))
    return _ordered_unique(result)


def build_capability_routing(
    project_root: Path,
    catalog: Mapping[str, Any],
    collector_contracts: Mapping[str, Any],
    hierarchy: Mapping[str, Any],
    *,
    now: datetime | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    current = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    errors = validate_catalog(catalog)
    policy = _as_dict(catalog.get("routing_policy"))
    all_capabilities, capability_planes = flatten_capabilities(catalog)
    capability_set = set(all_capabilities)
    producers = _producer_rows(project_root, catalog, collector_contracts, now=current)
    producer_by_capability: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for producer in producers:
        for capability_id in _as_list(producer.get("capabilities")):
            producer_by_capability[str(capability_id)].append(producer)

    current_collectors = {
        str(row.get("name") or "")
        for row in _as_list(collector_contracts.get("rows"))
        if isinstance(row, dict) and str(row.get("name") or "")
    }
    mapped_collectors = {
        str(row.get("collector_name") or "")
        for row in producers
        if str(row.get("producer_kind") or "") == "collector"
    }
    unmapped_collectors = sorted(current_collectors - mapped_collectors)
    missing_catalog_collectors = sorted(mapped_collectors - current_collectors)
    collector_contracts_age = _mapping_age_minutes(collector_contracts, now=current)
    hierarchy_age = _mapping_age_minutes(hierarchy, now=current)
    collector_contracts_fresh = bool(
        collector_contracts_age is not None and collector_contracts_age <= 30.0
    )
    hierarchy_fresh = bool(hierarchy_age is not None and hierarchy_age <= 24.0 * 60.0)

    assignments = [row for row in _as_list(hierarchy.get("assignments")) if isinstance(row, dict)]
    expected_assignment_count = _safe_int(hierarchy.get("assignment_count"), len(assignments))
    scope_requirements = _as_dict(catalog.get("scope_required_capabilities"))
    role_requirements = _as_dict(catalog.get("role_required_capabilities"))
    regime_map = _as_dict(catalog.get("regime_axis_capability_map"))
    max_required = max(_safe_int(policy.get("max_required_capabilities_per_profile"), 48), 1)
    max_optional = max(_safe_int(policy.get("max_optional_capabilities_per_profile"), 160), 1)
    profiles: dict[str, dict[str, Any]] = {}
    context_profiles: dict[str, dict[str, Any]] = {}
    bindings: list[dict[str, Any]] = []
    required_subscription_count: Counter[str] = Counter()
    optional_subscription_count: Counter[str] = Counter()

    for assignment in assignments:
        bot_id = str(assignment.get("bot_id") or "").strip()
        if not bot_id:
            continue
        scope = str(assignment.get("regime_scope") or "market_signal")
        role = str(assignment.get("role_id") or "signal")
        text = _assignment_text(assignment)
        ranked_planes: list[tuple[int, int, dict[str, Any]]] = []
        for index, raw_plane in enumerate(_as_list(catalog.get("planes"))):
            plane = _as_dict(raw_plane)
            score = _matched_plane_score(plane, assignment, text)
            if score > 0:
                ranked_planes.append((score, -index, plane))
        ranked_planes.sort(key=lambda item: (item[0], item[1]), reverse=True)

        required = _ordered_unique(
            list(_as_list(scope_requirements.get(scope))) + list(_as_list(role_requirements.get(role)))
        )
        optional: list[str] = []
        matched_plane_ids: list[str] = []
        for _, _, plane in ranked_planes:
            matched_plane_ids.append(str(plane.get("plane_id") or ""))
            routing = _as_dict(plane.get("routing"))
            required.extend(_ordered_unique(_as_list(routing.get("required_when_matched"))))
            optional.extend(_ordered_unique(_as_list(plane.get("capabilities"))))
        required = [item for item in _ordered_unique(required) if item in capability_set][:max_required]
        optional = [item for item in _ordered_unique(optional) if item in capability_set and item not in required][
            :max_optional
        ]

        profile_signature = {
            "scope": scope,
            "role": role,
            "required_capability_ids": required,
            "optional_capability_ids": optional,
        }
        profile_id = f"cap_profile_{canonical_hash(profile_signature)[:16]}"
        if profile_id not in profiles:
            profiles[profile_id] = {
                "profile_id": profile_id,
                **profile_signature,
                "matched_plane_ids": _ordered_unique(matched_plane_ids),
                "bot_count": 0,
            }
        profiles[profile_id]["bot_count"] += 1

        unknown_axes = _unknown_regime_axes(assignment)
        context_caps = _ordered_unique(
            capability
            for axis_id in unknown_axes
            for capability in _as_list(regime_map.get(axis_id))
            if str(capability) in capability_set
        )
        context_signature = {"axis_ids": unknown_axes, "capability_ids": context_caps}
        context_id = f"runtime_context_{canonical_hash(context_signature)[:16]}"
        if context_id not in context_profiles:
            context_profiles[context_id] = {
                "runtime_context_profile_id": context_id,
                **context_signature,
                "bot_count": 0,
            }
        context_profiles[context_id]["bot_count"] += 1

        for capability_id in required:
            required_subscription_count[capability_id] += 1
        for capability_id in optional:
            optional_subscription_count[capability_id] += 1
        bindings.append(
            {
                "bot_id": bot_id,
                "cell_id": str(assignment.get("cell_id") or ""),
                "profile_id": profile_id,
                "runtime_context_profile_id": context_id,
            }
        )

    subscribed_capabilities = set(required_subscription_count) | set(optional_subscription_count)
    required_capabilities = set(required_subscription_count)
    producer_supported = {capability for capability, rows in producer_by_capability.items() if rows}
    producer_usable = {
        capability
        for capability, rows in producer_by_capability.items()
        if any(capability in set(_as_list(row.get("usable_capabilities"))) for row in rows)
    }
    source_priority = _ordered_unique(
        _as_list(policy.get("source_kind_priority"))
        or [
            "official",
            "broker_native",
            "official_derived",
            "official_public_mesh",
            "internal_control",
            "internal_derived",
            "internal_replay",
            "public_licensed_mesh",
            "public_mesh",
            "public",
        ]
    )
    source_rank = {source_kind: index for index, source_kind in enumerate(source_priority)}
    capability_resolutions: list[dict[str, Any]] = []
    for capability_id in sorted(subscribed_capabilities):
        configured_rows = producer_by_capability.get(capability_id, [])
        usable_rows = [
            row
            for row in configured_rows
            if capability_id in set(_as_list(row.get("usable_capabilities")))
        ]
        usable_rows.sort(
            key=lambda row: (
                source_rank.get(str(row.get("source_kind") or ""), len(source_rank)),
                _safe_float(row.get("age_minutes"), 1.0e12),
                str(row.get("producer_id") or ""),
            )
        )
        selected = usable_rows[0] if usable_rows else {}
        capability_resolutions.append(
            {
                "capability_id": capability_id,
                "required": capability_id in required_capabilities,
                "configured_producer_ids": [
                    str(row.get("producer_id") or "") for row in configured_rows
                ],
                "usable_producer_ids": [str(row.get("producer_id") or "") for row in usable_rows],
                "selected_producer_id": str(selected.get("producer_id") or ""),
                "selected_source_kind": str(selected.get("source_kind") or ""),
                "selected_age_minutes": selected.get("age_minutes"),
                "selected_proof": _as_dict(selected.get("capability_proofs")).get(capability_id, {}),
                "failover_producer_ids": [
                    str(row.get("producer_id") or "") for row in usable_rows[1:]
                ],
                "usable_producer_count": len(usable_rows),
                "redundant": len(usable_rows) >= 2,
            }
        )
    unsupported_required = sorted(required_capabilities - producer_supported)
    unavailable_required = sorted(
        capability for capability in required_capabilities & producer_supported if capability not in producer_usable
    )
    unsupported_optional = sorted((subscribed_capabilities - required_capabilities) - producer_supported)
    unavailable_optional = sorted(
        capability
        for capability in (subscribed_capabilities - required_capabilities) & producer_supported
        if capability not in producer_usable
    )
    max_gap_rows = max(_safe_int(policy.get("max_gap_rows"), 100), 1)

    gap_rows: list[dict[str, Any]] = []
    for capability_id, producer_status in (
        [(item, "unsupported") for item in unsupported_required + unsupported_optional]
        + [(item, "unavailable") for item in unavailable_required + unavailable_optional]
    ):
        required_count = required_subscription_count[capability_id]
        optional_count = optional_subscription_count[capability_id]
        plane = _as_dict(capability_planes.get(capability_id))
        configured_producers = producer_by_capability.get(capability_id, [])
        gap_rows.append(
            {
                "capability_id": capability_id,
                "plane_id": str(plane.get("plane_id") or ""),
                "required_subscription_count": required_count,
                "optional_subscription_count": optional_count,
                "subscribed_bot_count": required_count + optional_count,
                "severity": "live_promotion_blocker" if required_count else "advisory",
                "candidate_blocking": bool(required_count),
                "paper_soak_blocker": False,
                "producer_status": producer_status,
                "configured_producer_ids": [
                    str(row.get("producer_id") or "") for row in configured_producers
                ],
                "recovery_mode": (
                    "admit_new_source_backed_producer"
                    if producer_status == "unsupported"
                    else "refresh_or_repair_existing_producer_proof"
                ),
                "proof_requirement": (
                    "source_provenance_point_in_time_freshness_cache_failure_isolation_and_human_approval"
                    if producer_status == "unsupported"
                    else "fresh_capability_specific_proof"
                ),
            }
        )
    gap_rows.sort(
        key=lambda row: (
            row["required_subscription_count"] > 0,
            row["subscribed_bot_count"],
            row["capability_id"],
        ),
        reverse=True,
    )
    plane_gap_rollups: list[dict[str, Any]] = []
    grouped_gaps: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in gap_rows:
        grouped_gaps[str(row.get("plane_id") or "unknown")].append(row)
    for plane_id, rows in grouped_gaps.items():
        plane_gap_rollups.append(
            {
                "plane_id": plane_id,
                "gap_count": len(rows),
                "candidate_blocking_gap_count": sum(
                    1 for row in rows if bool(row.get("candidate_blocking"))
                ),
                "unsupported_gap_count": sum(
                    1 for row in rows if row.get("producer_status") == "unsupported"
                ),
                "unavailable_gap_count": sum(
                    1 for row in rows if row.get("producer_status") == "unavailable"
                ),
                "subscribed_bot_count": sum(
                    _safe_int(row.get("subscribed_bot_count")) for row in rows
                ),
                "top_capability_ids": [
                    str(row.get("capability_id") or "")
                    for row in sorted(
                        rows,
                        key=lambda item: (
                            _safe_int(item.get("subscribed_bot_count")),
                            str(item.get("capability_id") or ""),
                        ),
                        reverse=True,
                    )[:5]
                ],
            }
        )
    plane_gap_rollups.sort(
        key=lambda row: (
            row["candidate_blocking_gap_count"],
            row["subscribed_bot_count"],
            row["plane_id"],
        ),
        reverse=True,
    )
    admission = _as_dict(catalog.get("physical_producer_admission"))
    minimum_subscribers = max(_safe_int(admission.get("minimum_subscribed_bot_count"), 1), 1)
    next_admission_candidates = [
        {
            "capability_id": row["capability_id"],
            "plane_id": row["plane_id"],
            "subscribed_bot_count": row["subscribed_bot_count"],
            "candidate_blocking": row["candidate_blocking"],
            "admission_state": "human_review_required",
        }
        for row in gap_rows
        if row["producer_status"] == "unsupported"
        and _safe_int(row.get("subscribed_bot_count")) >= minimum_subscribers
    ][:12]
    recovery_candidates = [
        {
            "capability_id": row["capability_id"],
            "plane_id": row["plane_id"],
            "configured_producer_ids": row["configured_producer_ids"],
            "candidate_blocking": row["candidate_blocking"],
        }
        for row in gap_rows
        if row["producer_status"] == "unavailable"
    ][:12]

    structural_blockers = list(errors)
    if not assignments or expected_assignment_count <= 0:
        structural_blockers.append("capability_router_bot_hierarchy_missing")
    if expected_assignment_count != len(assignments):
        structural_blockers.append("capability_router_hierarchy_assignment_count_mismatch")
    if unmapped_collectors:
        structural_blockers.append("capability_router_current_collectors_unmapped")
    if not current_collectors:
        structural_blockers.append("capability_router_collector_contracts_missing")
    elif missing_catalog_collectors:
        structural_blockers.append("capability_router_catalog_collectors_missing_from_contracts")
    if len(bindings) != len(assignments):
        structural_blockers.append("capability_router_bot_binding_incomplete")
    authority = {key: False for key in EXPECTED_SAFETY_FLAGS}
    if any(authority.values()):
        structural_blockers.append("capability_router_authority_contract_unsafe")
    structural_blockers = _ordered_unique(structural_blockers)

    required_collector_failures = _ordered_unique(collector_contracts.get("required_failures") or [])
    input_freshness_blockers = _ordered_unique(
        [
            "collector_contracts_input_stale" if not collector_contracts_fresh else "",
            "bot_hierarchy_input_stale" if not hierarchy_fresh else "",
        ]
    )
    structural_ok = not structural_blockers
    paper_soak_ready = bool(
        structural_ok and not required_collector_failures and not input_freshness_blockers
    )
    live_promotion_ready = bool(
        paper_soak_ready
        and not unsupported_required
        and not unavailable_required
    )
    weighted_required = sum(required_subscription_count.values())
    weighted_optional = sum(optional_subscription_count.values())
    naive_fetches = weighted_required + weighted_optional
    active_producers = [row for row in producers if set(_as_list(row.get("capabilities"))) & subscribed_capabilities]
    shared_fetches = len(active_producers)
    binding_ratio = len(bindings) / len(assignments) if assignments else 0.0
    subscription_coverage = len(subscribed_capabilities & producer_supported) / len(subscribed_capabilities) if subscribed_capabilities else 0.0
    required_usable_ratio = len(required_capabilities & producer_usable) / len(required_capabilities) if required_capabilities else 0.0
    producer_coverage = len(producer_supported) / len(capability_set) if capability_set else 0.0
    full_catalog_coverage_ready = bool(len(producer_supported) == len(capability_set))
    required_redundant_count = sum(
        1 for row in capability_resolutions if row["required"] and row["redundant"]
    )
    required_redundancy_ratio = (
        required_redundant_count / len(required_capabilities) if required_capabilities else 0.0
    )
    required_single_source = [
        str(row["capability_id"])
        for row in capability_resolutions
        if row["required"] and row["usable_producer_count"] == 1
    ]

    routing_payload = {
        "timestamp_utc": current.isoformat(),
        "schema_version": 1,
        "catalog_id": str(catalog.get("catalog_id") or ""),
        "operating_mode": "metadata_subscription_shadow_only",
        "catalog_receipt_sha256": canonical_hash(catalog),
        "hierarchy_receipt_sha256": str(hierarchy.get("assignment_receipt_sha256") or ""),
        "subscription_profiles": sorted(profiles.values(), key=lambda row: row["profile_id"]),
        "runtime_context_profiles": sorted(
            context_profiles.values(), key=lambda row: row["runtime_context_profile_id"]
        ),
        "bot_bindings": bindings,
        "capability_resolutions": capability_resolutions,
        "authority_contract": authority,
        "cache_contract": {
            "mode": "content_addressed_shared_subscription_profiles",
            "one_physical_fetch_may_publish_many_capabilities": True,
            "many_bots_share_one_snapshot": True,
            "router_launches_physical_collectors": False,
        },
    }
    routing_payload["routing_receipt_sha256"] = canonical_hash(
        {
            "catalog_receipt_sha256": routing_payload["catalog_receipt_sha256"],
            "hierarchy_receipt_sha256": routing_payload["hierarchy_receipt_sha256"],
            "subscription_profiles": routing_payload["subscription_profiles"],
            "runtime_context_profiles": routing_payload["runtime_context_profiles"],
            "bot_bindings": bindings,
            "capability_resolutions": capability_resolutions,
        }
    )

    health_payload = {
        "timestamp_utc": current.isoformat(),
        "schema_version": 1,
        "ok": structural_ok,
        "status": "ready" if structural_ok and not gap_rows else ("ready_with_coverage_debt" if structural_ok else "blocked"),
        "overall_status": "ready" if structural_ok and not gap_rows else ("ready_with_coverage_debt" if structural_ok else "blocked"),
        "paper_soak_ready": paper_soak_ready,
        "live_promotion_ready": live_promotion_ready,
        "structural_blockers": structural_blockers,
        "paper_soak_blockers": [
            f"required_collector_failure:{name}" for name in required_collector_failures
        ]
        + input_freshness_blockers,
        "live_promotion_blockers": _ordered_unique(
            [f"unsupported_required_capability:{item}" for item in unsupported_required]
            + [f"unusable_required_capability:{item}" for item in unavailable_required]
        ),
        "summary": {
            "plane_count": len(_as_list(catalog.get("planes"))),
            "capability_count": len(all_capabilities),
            "producer_count": len(producers),
            "collector_producer_count": sum(1 for row in producers if row["producer_kind"] == "collector"),
            "artifact_producer_count": sum(1 for row in producers if row["producer_kind"] == "artifact"),
            "current_collector_count": len(current_collectors),
            "mapped_current_collector_count": len(current_collectors & mapped_collectors),
            "unmapped_current_collector_count": len(unmapped_collectors),
            "missing_catalog_collector_count": len(missing_catalog_collectors),
            "assignment_count": len(assignments),
            "bot_binding_count": len(bindings),
            "bot_binding_coverage_ratio": round(binding_ratio, 6),
            "subscription_profile_count": len(profiles),
            "runtime_context_profile_count": len(context_profiles),
            "weighted_required_subscription_count": weighted_required,
            "weighted_optional_subscription_count": weighted_optional,
            "subscribed_capability_count": len(subscribed_capabilities),
            "supported_subscribed_capability_count": len(subscribed_capabilities & producer_supported),
            "unsupported_required_capability_count": len(unsupported_required),
            "unavailable_required_capability_count": len(unavailable_required),
            "unsupported_optional_capability_count": len(unsupported_optional),
            "unavailable_optional_capability_count": len(unavailable_optional),
            "producer_supported_capability_count": len(producer_supported),
            "producer_usable_capability_count": len(producer_usable),
            "catalog_producer_coverage_ratio": round(producer_coverage, 6),
            "full_catalog_coverage_ready": full_catalog_coverage_ready,
            "subscription_producer_coverage_ratio": round(subscription_coverage, 6),
            "required_capability_usable_ratio": round(required_usable_ratio, 6),
            "required_capability_redundant_count": required_redundant_count,
            "required_capability_redundancy_ratio": round(required_redundancy_ratio, 6),
            "naive_per_bot_fetch_count": naive_fetches,
            "shared_physical_producer_count": shared_fetches,
            "estimated_fetch_avoidance_ratio": round(1.0 - (shared_fetches / naive_fetches), 6)
            if naive_fetches
            else 0.0,
        },
        "current_collector_mapping": {
            "complete": bool(current_collectors and not unmapped_collectors and not missing_catalog_collectors),
            "unmapped_collectors": unmapped_collectors,
            "missing_catalog_collectors": missing_catalog_collectors,
        },
        "input_freshness": {
            "collector_contracts": {
                "age_minutes": round(collector_contracts_age, 3)
                if collector_contracts_age is not None
                else None,
                "max_age_minutes": 30.0,
                "fresh": collector_contracts_fresh,
            },
            "bot_hierarchy": {
                "age_minutes": round(hierarchy_age, 3) if hierarchy_age is not None else None,
                "max_age_minutes": 24.0 * 60.0,
                "fresh": hierarchy_fresh,
            },
        },
        "coverage_debt": {
            "managed": True,
            "blocks_guarded_paper_soak": False,
            "blocks_live_promotion_when_required": bool(unsupported_required or unavailable_required),
            "candidate_blocking_gap_count": len(unsupported_required) + len(unavailable_required),
            "optional_gap_count": len(unsupported_optional) + len(unavailable_optional),
            "unsupported_gap_count": len(unsupported_required) + len(unsupported_optional),
            "unavailable_gap_count": len(unavailable_required) + len(unavailable_optional),
            "gap_count": len(gap_rows),
            "rows": gap_rows[:max_gap_rows],
            "plane_rollups": plane_gap_rollups,
            "next_admission_candidates": next_admission_candidates,
            "recovery_candidates": recovery_candidates,
            "admission_contract": {
                "minimum_subscribed_bot_count": minimum_subscribers,
                "human_approval_required": admission.get("require_human_approval") is True,
                "automatic_producer_creation": False,
                "automatic_live_promotion": False,
            },
            "gap_receipt_sha256": canonical_hash(gap_rows),
        },
        "producer_health": producers,
        "provider_resilience": {
            "required_capability_count": len(required_capabilities),
            "redundant_required_capability_count": required_redundant_count,
            "required_redundancy_ratio": round(required_redundancy_ratio, 6),
            "single_source_required_capability_count": len(required_single_source),
            "single_source_required_capability_ids": required_single_source,
            "no_source_required_capability_ids": sorted(
                required_capabilities - producer_usable
            ),
            "redundancy_is_advisory_unless_declared_by_candidate": True,
            "provider_selection_published": True,
            "failover_selection_published": True,
        },
        "resource_distribution": dict(Counter(row["resource_class"] for row in producers)),
        "cadence_distribution": dict(Counter(row["cadence"] for row in producers)),
        "authority_contract": authority,
        "routing_receipt_sha256": routing_payload["routing_receipt_sha256"],
        "routing_artifact": "governance/collector_capabilities/bot_subscriptions_latest.json",
        "policy": {
            "unsupported_capabilities_remain_explicit": True,
            "current_required_collector_failure_blocks_guarded_paper_soak": True,
            "unsupported_required_capability_blocks_live_promotion_not_guarded_paper_soak": True,
            "live_promotion_is_candidate_specific": True,
            "optional_catalog_gaps_do_not_veto_candidate_promotion": True,
            "capability_claims_require_producer_or_field_level_proof": True,
            "provider_selection_and_failover_are_published": True,
            "full_catalog_coverage_is_reported_not_required": True,
            "logical_capabilities_do_not_create_physical_processes": True,
        },
    }
    return health_payload, routing_payload
