"""Deterministic, execution-free routing for collector capabilities."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INGESTION_ROUTING_POLICY_PATH = (
    PROJECT_ROOT / "config" / "sleeve_ingestion_routing_v2.json"
)
DEFAULT_DECISION_POLICY_PATH = (
    PROJECT_ROOT / "config" / "institutional_decision_flow_v1.json"
)

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
EXPECTED_INGESTION_SAFETY_FLAGS = (
    "changes_strategy_signal",
    "launches_collectors",
    "fetches_external_data",
    "mutates_bot_registry",
    "paper_execution_authority",
    "live_execution_authority",
    "automatic_promotion_authority",
    "profitability_guaranteed",
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


def _load_mapping(path: Path) -> dict[str, Any]:
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError):
        return {}
    return loaded if isinstance(loaded, dict) else {}


def load_ingestion_routing_policy(
    path: Path | str = DEFAULT_INGESTION_ROUTING_POLICY_PATH,
) -> dict[str, Any]:
    """Load the route policy without granting it runtime or execution authority."""

    return _load_mapping(Path(path))


def load_decision_alignment_policy(
    path: Path | str = DEFAULT_DECISION_POLICY_PATH,
) -> dict[str, Any]:
    return _load_mapping(Path(path))


def _normalized(value: Any) -> str:
    return str(value or "").strip().lower().replace("-", "_").replace(" ", "_")


def _decision_family_id(
    assignment: Mapping[str, Any],
    decision_policy: Mapping[str, Any],
) -> tuple[str, str]:
    """Resolve the same family vocabulary used by the institutional decision flow."""

    scope = _normalized(assignment.get("regime_scope"))
    role = _normalized(assignment.get("role_id"))
    if scope == "operational_control" or role in {"control", "operations"}:
        return "infrastructure_control", "operational_scope"

    profile_values = [
        _normalized(assignment.get("sleeve_id")),
        _normalized(assignment.get("sub_sleeve_id")),
        _normalized(assignment.get("bot_id")),
    ]
    exact_map = {
        _normalized(key): str(value or "")
        for key, value in _as_dict(decision_policy.get("profile_policy_map")).items()
    }
    for value in profile_values:
        if value and exact_map.get(value):
            return exact_map[value], "exact_profile"

    profile_text = " ".join(value for value in profile_values if value)
    for index, raw_rule in enumerate(_as_list(decision_policy.get("profile_policy_rules"))):
        rule = _as_dict(raw_rule)
        tokens = [_normalized(value) for value in _as_list(rule.get("profile_tokens_any"))]
        if any(token and token in profile_text for token in tokens):
            return str(rule.get("policy_family_id") or "balanced_directional"), f"profile_rule:{index}"
    return "balanced_directional", "default_fallback"


def validate_ingestion_routing_policy(
    policy: Mapping[str, Any],
    *,
    catalog: Mapping[str, Any],
    decision_policy: Mapping[str, Any],
) -> list[str]:
    errors: list[str] = []
    if _safe_int(policy.get("schema_version")) != 2:
        errors.append("ingestion_routing_schema_version_invalid")
    if str(policy.get("policy_id") or "") != "sleeve_ingestion_routing_v2":
        errors.append("ingestion_routing_policy_id_invalid")
    if str(policy.get("operating_mode") or "") != "decision_aligned_observation_routing":
        errors.append("ingestion_routing_operating_mode_invalid")

    capability_ids, _ = flatten_capabilities(catalog)
    capability_set = set(capability_ids)
    plane_ids = {
        str(_as_dict(row).get("plane_id") or "")
        for row in _as_list(catalog.get("planes"))
    }
    lanes = _as_dict(policy.get("lane_contracts"))
    if set(lanes) != {"core", "deferred", "cold"}:
        errors.append("ingestion_routing_lane_contract_invalid")

    quality = _as_dict(policy.get("route_quality_contract"))
    weights = _as_dict(quality.get("weights"))
    if not weights or abs(sum(_safe_float(value) for value in weights.values()) - 1.0) > 1e-6:
        errors.append("ingestion_routing_quality_weights_invalid")
    for key in (
        "paper_route_score_floor",
        "live_route_score_floor",
        "required_capability_coverage_floor",
    ):
        value = _safe_float(quality.get(key), -1.0)
        if not 0.0 <= value <= 1.0:
            errors.append(f"ingestion_routing_{key}_invalid")

    base_profiles = _as_dict(policy.get("base_profiles"))
    for profile_id, raw_profile in base_profiles.items():
        unknown = sorted(
            set(_ordered_unique(_as_list(_as_dict(raw_profile).get("required_capability_ids"))))
            - capability_set
        )
        if unknown:
            errors.append(f"ingestion_routing_base_profile_unknown_capability:{profile_id}")

    family_routes = _as_dict(policy.get("family_routes"))
    decision_families = set(_as_dict(decision_policy.get("sleeve_policy_families")))
    if set(family_routes) != decision_families:
        errors.append("ingestion_routing_decision_family_coverage_invalid")
    for family_id, raw_route in family_routes.items():
        route = _as_dict(raw_route)
        if str(route.get("lane") or "") not in lanes:
            errors.append(f"ingestion_routing_family_lane_invalid:{family_id}")
        if any(str(value) not in base_profiles for value in _as_list(route.get("base_profiles"))):
            errors.append(f"ingestion_routing_family_base_profile_invalid:{family_id}")
        unknown_caps = sorted(
            set(_ordered_unique(_as_list(route.get("required_capability_ids"))))
            - capability_set
        )
        if unknown_caps:
            errors.append(f"ingestion_routing_family_unknown_capability:{family_id}")
        paper_caps = set(
            _ordered_unique(_as_list(route.get("paper_required_capability_ids")))
        )
        if not paper_caps.issubset(
            set(_ordered_unique(_as_list(route.get("required_capability_ids"))))
        ):
            errors.append(
                f"ingestion_routing_family_paper_capability_not_live_required:{family_id}"
            )
        unknown_planes = sorted(
            set(_ordered_unique(_as_list(route.get("optional_plane_ids")))) - plane_ids
        )
        if unknown_planes:
            errors.append(f"ingestion_routing_family_unknown_plane:{family_id}")
        if _safe_int(route.get("max_required_capabilities")) <= 0:
            errors.append(f"ingestion_routing_family_required_cap_invalid:{family_id}")
        if _safe_int(route.get("max_optional_capabilities")) <= 0:
            errors.append(f"ingestion_routing_family_optional_cap_invalid:{family_id}")

    transport = _as_dict(policy.get("transport_contract"))
    if not transport or any(value is not True for value in transport.values()):
        errors.append("ingestion_routing_transport_contract_incomplete")
    safety = _as_dict(policy.get("safety_contract"))
    for key in EXPECTED_INGESTION_SAFETY_FLAGS:
        if safety.get(key) is not False:
            errors.append(f"ingestion_routing_safety_{key}_must_be_false")
    alignment = _as_dict(policy.get("decision_alignment"))
    if str(alignment.get("decision_stage") or "") != "02_data_qualification":
        errors.append("ingestion_routing_decision_stage_alignment_invalid")
    return _ordered_unique(errors)


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
            "max_age_minutes": _safe_float(producer.get("max_age_minutes"), 0.0),
            "collector_quality_score": 0.0,
            "source_coverage_ratio": 0.0,
            "error_budget_remaining": 0.0,
            "payload_integrity_ready": False,
            "failure_domain": str(
                producer.get("failure_domain")
                or producer.get("collector_name")
                or producer.get("artifact_path")
                or producer_id
            ),
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
                    "collector_quality_score": _safe_float(
                        contract_row.get("quality_score"),
                        1.0 if contract_usable else 0.0,
                    ),
                    "source_coverage_ratio": _safe_float(
                        _as_dict(contract_row.get("source_status")).get(
                            "coverage_ratio"
                        ),
                        1.0 if contract_usable else 0.0,
                    ),
                    "error_budget_remaining": _safe_float(
                        _as_dict(contract_row.get("error_budget")).get(
                            "error_budget_remaining"
                        ),
                        1.0 if contract_usable else 0.0,
                    ),
                    "payload_integrity_ready": bool(
                        contract_usable
                        and (
                            str(contract_row.get("payload_sha256") or "")
                            or contract_row.get("payload_present") is not False
                        )
                    ),
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
                    "collector_quality_score": 1.0 if artifact_usable else 0.0,
                    "source_coverage_ratio": 1.0 if artifact_usable else 0.0,
                    "error_budget_remaining": 1.0 if artifact_usable else 0.0,
                    "payload_integrity_ready": bool(artifact_usable),
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


def _profile_spec(
    assignment: Mapping[str, Any],
    *,
    catalog: Mapping[str, Any],
    ingestion_policy: Mapping[str, Any],
    decision_policy: Mapping[str, Any],
    capability_set: set[str],
) -> dict[str, Any]:
    family_id, family_match_source = _decision_family_id(assignment, decision_policy)
    family_routes = _as_dict(ingestion_policy.get("family_routes"))
    route = deepcopy(
        _as_dict(
            family_routes.get(family_id)
            or family_routes.get("balanced_directional")
        )
    )
    base_profiles = _as_dict(ingestion_policy.get("base_profiles"))
    scope = str(assignment.get("regime_scope") or "market_signal")
    role = str(assignment.get("role_id") or "signal")
    text = _assignment_text(assignment)

    required: list[str] = []
    paper_required: list[str] = []
    for base_profile_id in _as_list(route.get("base_profiles")):
        base_required = _ordered_unique(
            _as_list(
                _as_dict(base_profiles.get(str(base_profile_id))).get(
                    "required_capability_ids"
                )
            )
        )
        required.extend(base_required)
        paper_required.extend(base_required)
    structural_required = _ordered_unique(
        list(
            _as_list(
                _as_dict(catalog.get("scope_required_capabilities")).get(scope)
            )
        )
        + list(
            _as_list(
                _as_dict(catalog.get("role_required_capabilities")).get(role)
            )
        )
    )
    required.extend(structural_required)
    paper_required.extend(structural_required)
    required.extend(_ordered_unique(_as_list(route.get("required_capability_ids"))))
    paper_required.extend(
        _ordered_unique(_as_list(route.get("paper_required_capability_ids")))
    )

    planes_by_id = {
        str(_as_dict(raw_plane).get("plane_id") or ""): _as_dict(raw_plane)
        for raw_plane in _as_list(catalog.get("planes"))
    }
    matched_plane_ids = _ordered_unique(_as_list(route.get("optional_plane_ids")))
    token_matched_plane_ids: list[str] = []
    for raw_plane in _as_list(catalog.get("planes")):
        plane = _as_dict(raw_plane)
        routing = _as_dict(plane.get("routing"))
        tokens = _ordered_unique(_as_list(routing.get("target_tokens")))
        if tokens and any(_normalized(token) in _normalized(text) for token in tokens):
            plane_id = str(plane.get("plane_id") or "")
            token_matched_plane_ids.append(plane_id)
            token_required = _ordered_unique(
                _as_list(routing.get("required_when_matched"))
            )
            required.extend(token_required)
            paper_required.extend(token_required)
    matched_plane_ids = _ordered_unique(matched_plane_ids + token_matched_plane_ids)

    optional: list[str] = []
    for plane_id in matched_plane_ids:
        optional.extend(_ordered_unique(_as_list(_as_dict(planes_by_id.get(plane_id)).get("capabilities"))))

    max_required = max(
        _safe_int(route.get("max_required_capabilities"), 32),
        1,
    )
    max_optional = max(
        _safe_int(route.get("max_optional_capabilities"), 72),
        1,
    )
    required = [
        item for item in _ordered_unique(required) if item in capability_set
    ][:max_required]
    paper_required = [
        item
        for item in _ordered_unique(paper_required)
        if item in capability_set and item in required
    ]
    optional = [
        item
        for item in _ordered_unique(optional)
        if item in capability_set and item not in required
    ][:max_optional]
    return {
        "scope": scope,
        "role": role,
        "decision_policy_family_id": family_id,
        "family_match_source": family_match_source,
        "ingestion_lane": str(route.get("lane") or "core"),
        "cadence": str(route.get("cadence") or "intraday"),
        "degradation_policy": str(route.get("degradation_policy") or "collect_only"),
        "live_independent_failover_required": bool(
            route.get("live_independent_failover_required", False)
        ),
        "paper_required_capability_ids": paper_required,
        "required_capability_ids": required,
        "optional_capability_ids": optional,
        "matched_plane_ids": matched_plane_ids,
        "token_matched_plane_ids": _ordered_unique(token_matched_plane_ids),
    }


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


def _producer_route_score(
    producer: Mapping[str, Any],
    capability_id: str,
    ingestion_policy: Mapping[str, Any],
) -> tuple[float, dict[str, float]]:
    quality_contract = _as_dict(ingestion_policy.get("route_quality_contract"))
    weights = _as_dict(quality_contract.get("weights"))
    authority_scores = _as_dict(quality_contract.get("source_authority_scores"))
    max_age = max(_safe_float(producer.get("max_age_minutes"), 0.0), 1e-9)
    age = _safe_float(producer.get("age_minutes"), max_age * 2.0)
    proof = _as_dict(_as_dict(producer.get("capability_proofs")).get(capability_id))
    components = {
        "source_authority": _safe_float(
            authority_scores.get(str(producer.get("source_kind") or "")),
            0.5,
        ),
        "collector_quality": _safe_float(
            producer.get("collector_quality_score"),
            1.0 if producer.get("usable") else 0.0,
        ),
        "freshness_margin": max(0.0, min(1.0, 1.0 - (age / max_age))),
        "capability_proof": 1.0
        if proof.get("passed") is True
        else (0.85 if producer.get("usable") and not proof else 0.0),
        "source_coverage": _safe_float(
            producer.get("source_coverage_ratio"),
            1.0 if producer.get("usable") else 0.0,
        ),
        "error_budget_remaining": _safe_float(
            producer.get("error_budget_remaining"),
            1.0 if producer.get("usable") else 0.0,
        ),
        "payload_integrity": 1.0
        if producer.get("payload_integrity_ready")
        else 0.0,
    }
    components = {
        key: round(max(0.0, min(1.0, value)), 6)
        for key, value in components.items()
    }
    score = sum(
        _safe_float(weights.get(key), 0.0) * value
        for key, value in components.items()
    )
    return round(max(0.0, min(1.0, score)), 6), components


def _profile_delivery_routes(
    profiles: Mapping[str, Mapping[str, Any]],
    capability_resolutions: Iterable[Mapping[str, Any]],
    ingestion_policy: Mapping[str, Any],
) -> list[dict[str, Any]]:
    resolution_by_capability = {
        str(row.get("capability_id") or ""): row
        for row in capability_resolutions
        if str(row.get("capability_id") or "")
    }
    quality = _as_dict(ingestion_policy.get("route_quality_contract"))
    paper_floor = _safe_float(quality.get("paper_route_score_floor"), 0.70)
    live_floor = _safe_float(quality.get("live_route_score_floor"), 0.86)
    coverage_floor = _safe_float(
        quality.get("required_capability_coverage_floor"), 1.0
    )
    minimum_failovers = max(
        _safe_int(quality.get("minimum_independent_live_failovers"), 1),
        0,
    )
    rows: list[dict[str, Any]] = []
    for profile_id, raw_profile in sorted(profiles.items()):
        profile = _as_dict(raw_profile)
        required_ids = _ordered_unique(
            _as_list(profile.get("required_capability_ids"))
        )
        paper_required_ids = _ordered_unique(
            _as_list(profile.get("paper_required_capability_ids"))
        )
        delivery_rows: list[dict[str, Any]] = []
        for capability_id in required_ids:
            resolution = _as_dict(resolution_by_capability.get(capability_id))
            delivery_rows.append(
                {
                    "capability_id": capability_id,
                    "selected_producer_id": str(
                        resolution.get("selected_producer_id") or ""
                    ),
                    "selected_source_kind": str(
                        resolution.get("selected_source_kind") or ""
                    ),
                    "route_score": _safe_float(
                        resolution.get("selected_route_score"), 0.0
                    ),
                    "selected_failure_domain": str(
                        resolution.get("selected_failure_domain") or ""
                    ),
                    "independent_failover_producer_ids": list(
                        resolution.get("independent_failover_producer_ids") or []
                    ),
                    "ready": bool(resolution.get("selected_producer_id")),
                }
            )
        usable_rows = [row for row in delivery_rows if row["ready"]]
        paper_rows = [
            row for row in delivery_rows if row["capability_id"] in paper_required_ids
        ]
        usable_paper_rows = [row for row in paper_rows if row["ready"]]
        live_route_scores = [float(row["route_score"]) for row in usable_rows]
        paper_route_scores = [
            float(row["route_score"]) for row in usable_paper_rows
        ]
        coverage_ratio = (
            len(usable_rows) / len(required_ids) if required_ids else 1.0
        )
        paper_coverage_ratio = (
            len(usable_paper_rows) / len(paper_required_ids)
            if paper_required_ids
            else 1.0
        )
        live_minimum_route_score = (
            min(live_route_scores) if live_route_scores else 0.0
        )
        paper_minimum_route_score = (
            min(paper_route_scores) if paper_route_scores else 0.0
        )
        live_average_route_score = (
            sum(live_route_scores) / len(live_route_scores)
            if live_route_scores
            else 0.0
        )
        paper_average_route_score = (
            sum(paper_route_scores) / len(paper_route_scores)
            if paper_route_scores
            else 0.0
        )
        independent_count = sum(
            1 for row in usable_rows if row["independent_failover_producer_ids"]
        )
        independent_ratio = (
            independent_count / len(required_ids) if required_ids else 1.0
        )
        paper_ready = bool(
            paper_coverage_ratio >= coverage_floor
            and paper_minimum_route_score >= paper_floor
        )
        independent_required = bool(
            profile.get("live_independent_failover_required", False)
        )
        live_ready = bool(
            paper_ready
            and coverage_ratio >= coverage_floor
            and live_minimum_route_score >= live_floor
            and (
                not independent_required
                or independent_count >= minimum_failovers
            )
        )
        missing_ids = [row["capability_id"] for row in delivery_rows if not row["ready"]]
        paper_missing_ids = [
            row["capability_id"] for row in paper_rows if not row["ready"]
        ]
        below_paper_ids = [
            row["capability_id"]
            for row in paper_rows
            if row["ready"] and float(row["route_score"]) < paper_floor
        ]
        route_material = {
            "profile_id": profile_id,
            "decision_policy_family_id": str(
                profile.get("decision_policy_family_id") or ""
            ),
            "ingestion_lane": str(profile.get("ingestion_lane") or "core"),
            "cadence": str(profile.get("cadence") or "intraday"),
            "paper_required_capability_ids": paper_required_ids,
            "required_capability_ids": required_ids,
            "delivery_routes": delivery_rows,
        }
        rows.append(
            {
                **route_material,
                "degradation_policy": str(
                    profile.get("degradation_policy") or "collect_only"
                ),
                "required_capability_count": len(required_ids),
                "usable_required_capability_count": len(usable_rows),
                "required_capability_coverage_ratio": round(coverage_ratio, 6),
                "paper_required_capability_count": len(paper_required_ids),
                "usable_paper_required_capability_count": len(
                    usable_paper_rows
                ),
                "paper_required_capability_coverage_ratio": round(
                    paper_coverage_ratio, 6
                ),
                "minimum_route_score": round(paper_minimum_route_score, 6),
                "average_route_score": round(paper_average_route_score, 6),
                "paper_minimum_route_score": round(
                    paper_minimum_route_score, 6
                ),
                "paper_average_route_score": round(
                    paper_average_route_score, 6
                ),
                "live_minimum_route_score": round(
                    live_minimum_route_score, 6
                ),
                "live_average_route_score": round(
                    live_average_route_score, 6
                ),
                "independent_failover_coverage_ratio": round(
                    independent_ratio, 6
                ),
                "paper_decision_data_ready": paper_ready,
                "live_decision_data_ready": live_ready,
                "missing_required_capability_ids": missing_ids,
                "missing_paper_required_capability_ids": paper_missing_ids,
                "below_paper_score_capability_ids": below_paper_ids,
                "route_state": "ready" if paper_ready else "collect_only",
                "route_receipt_sha256": canonical_hash(route_material),
            }
        )
    return rows


def build_capability_routing(
    project_root: Path,
    catalog: Mapping[str, Any],
    collector_contracts: Mapping[str, Any],
    hierarchy: Mapping[str, Any],
    *,
    now: datetime | None = None,
    ingestion_policy: Mapping[str, Any] | None = None,
    decision_policy: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    current = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    active_ingestion_policy = dict(
        ingestion_policy or load_ingestion_routing_policy()
    )
    active_decision_policy = dict(
        decision_policy or load_decision_alignment_policy()
    )
    errors = validate_catalog(catalog)
    errors.extend(
        validate_ingestion_routing_policy(
            active_ingestion_policy,
            catalog=catalog,
            decision_policy=active_decision_policy,
        )
    )
    errors = _ordered_unique(errors)
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
    regime_map = _as_dict(catalog.get("regime_axis_capability_map"))
    profiles: dict[str, dict[str, Any]] = {}
    context_profiles: dict[str, dict[str, Any]] = {}
    bindings: list[dict[str, Any]] = []
    runtime_sleeve_routes: list[dict[str, Any]] = []
    required_subscription_count: Counter[str] = Counter()
    optional_subscription_count: Counter[str] = Counter()

    def register_profile(
        assignment: Mapping[str, Any],
        *,
        bot_binding: bool,
    ) -> tuple[str, str, dict[str, Any]]:
        spec = _profile_spec(
            assignment,
            catalog=catalog,
            ingestion_policy=active_ingestion_policy,
            decision_policy=active_decision_policy,
            capability_set=capability_set,
        )
        profile_signature = {
            "route_policy_id": str(active_ingestion_policy.get("policy_id") or ""),
            "scope": spec["scope"],
            "role": spec["role"],
            "decision_policy_family_id": spec["decision_policy_family_id"],
            "ingestion_lane": spec["ingestion_lane"],
            "cadence": spec["cadence"],
            "degradation_policy": spec["degradation_policy"],
            "live_independent_failover_required": spec[
                "live_independent_failover_required"
            ],
            "paper_required_capability_ids": spec[
                "paper_required_capability_ids"
            ],
            "required_capability_ids": spec["required_capability_ids"],
            "optional_capability_ids": spec["optional_capability_ids"],
        }
        profile_id = f"cap_profile_{canonical_hash(profile_signature)[:16]}"
        if profile_id not in profiles:
            profiles[profile_id] = {
                "profile_id": profile_id,
                **profile_signature,
                "family_match_source": spec["family_match_source"],
                "matched_plane_ids": spec["matched_plane_ids"],
                "token_matched_plane_ids": spec["token_matched_plane_ids"],
                "bot_count": 0,
                "runtime_profile_count": 0,
                "profile_receipt_sha256": canonical_hash(profile_signature),
            }
        counter_key = "bot_count" if bot_binding else "runtime_profile_count"
        profiles[profile_id][counter_key] += 1

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
        if bot_binding:
            context_profiles[context_id]["bot_count"] += 1
        return profile_id, context_id, spec

    for assignment in assignments:
        bot_id = str(assignment.get("bot_id") or "").strip()
        if not bot_id:
            continue
        profile_id, context_id, spec = register_profile(
            assignment,
            bot_binding=True,
        )

        for capability_id in spec["required_capability_ids"]:
            required_subscription_count[capability_id] += 1
        for capability_id in spec["optional_capability_ids"]:
            optional_subscription_count[capability_id] += 1
        binding_material = {
            "bot_id": bot_id,
            "cell_id": str(assignment.get("cell_id") or ""),
            "sleeve_id": str(assignment.get("sleeve_id") or ""),
            "sub_sleeve_id": str(assignment.get("sub_sleeve_id") or ""),
            "horizon_id": str(assignment.get("horizon_id") or ""),
            "role_id": str(assignment.get("role_id") or ""),
            "decision_policy_family_id": spec["decision_policy_family_id"],
            "ingestion_lane": spec["ingestion_lane"],
            "profile_id": profile_id,
            "runtime_context_profile_id": context_id,
        }
        bindings.append(
            {
                **binding_material,
                "binding_receipt_sha256": canonical_hash(binding_material),
            }
        )

    for runtime_profile in _ordered_unique(
        _as_list(active_ingestion_policy.get("runtime_profiles"))
    ):
        synthetic_assignment = {
            "bot_id": f"runtime::{runtime_profile}",
            "cell_id": f"runtime/{runtime_profile}",
            "sleeve_id": runtime_profile,
            "sub_sleeve_id": "runtime_decision_lane",
            "horizon_id": "runtime",
            "regime_scope": "market_signal",
            "role_id": "signal",
            "regime_profile": {"axes": {}},
            "regime_metadata_access": {
                "runtime_context_required_axis_ids": []
            },
        }
        profile_id, context_id, spec = register_profile(
            synthetic_assignment,
            bot_binding=False,
        )
        material = {
            "runtime_profile": runtime_profile,
            "decision_policy_family_id": spec["decision_policy_family_id"],
            "ingestion_lane": spec["ingestion_lane"],
            "cadence": spec["cadence"],
            "profile_id": profile_id,
            "runtime_context_profile_id": context_id,
        }
        runtime_sleeve_routes.append(
            {
                **material,
                "route_binding_receipt_sha256": canonical_hash(material),
            }
        )

    runtime_required_capabilities = {
        capability_id
        for route in runtime_sleeve_routes
        for capability_id in _as_list(
            _as_dict(profiles.get(str(route.get("profile_id") or ""))).get(
                "required_capability_ids"
            )
        )
    }
    runtime_optional_capabilities = {
        capability_id
        for route in runtime_sleeve_routes
        for capability_id in _as_list(
            _as_dict(profiles.get(str(route.get("profile_id") or ""))).get(
                "optional_capability_ids"
            )
        )
    }
    subscribed_capabilities = (
        set(required_subscription_count)
        | set(optional_subscription_count)
        | runtime_required_capabilities
        | runtime_optional_capabilities
    )
    required_capabilities = set(required_subscription_count)
    bot_profile_ids = {
        str(row.get("profile_id") or "") for row in bindings
    }
    runtime_profile_ids = {
        str(row.get("profile_id") or "") for row in runtime_sleeve_routes
    }
    bot_subscription_profiles = [
        row
        for profile_id, row in profiles.items()
        if profile_id in bot_profile_ids
    ]
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
        usable_rows = []
        for row in configured_rows:
            if capability_id not in set(_as_list(row.get("usable_capabilities"))):
                continue
            route_score, route_components = _producer_route_score(
                row,
                capability_id,
                active_ingestion_policy,
            )
            usable_rows.append(
                {
                    **row,
                    "route_score": route_score,
                    "route_score_components": route_components,
                }
            )
        usable_rows.sort(
            key=lambda row: (
                -_safe_float(row.get("route_score"), 0.0),
                source_rank.get(str(row.get("source_kind") or ""), len(source_rank)),
                _safe_float(row.get("age_minutes"), 1.0e12),
                str(row.get("producer_id") or ""),
            )
        )
        selected = usable_rows[0] if usable_rows else {}
        selected_failure_domain = str(selected.get("failure_domain") or "")
        independent_failovers = [
            row
            for row in usable_rows[1:]
            if str(row.get("failure_domain") or "") != selected_failure_domain
        ]
        resolution_material = {
            "capability_id": capability_id,
            "selected_producer_id": str(selected.get("producer_id") or ""),
            "selected_failure_domain": selected_failure_domain,
            "selected_route_score": _safe_float(selected.get("route_score"), 0.0),
            "failover_producer_ids": [
                str(row.get("producer_id") or "") for row in usable_rows[1:]
            ],
            "independent_failover_producer_ids": [
                str(row.get("producer_id") or "") for row in independent_failovers
            ],
        }
        capability_resolutions.append(
            {
                "capability_id": capability_id,
                "required": capability_id in required_capabilities,
                "configured_producer_ids": [
                    str(row.get("producer_id") or "") for row in configured_rows
                ],
                "usable_producer_ids": [
                    str(row.get("producer_id") or "") for row in usable_rows
                ],
                "selected_producer_id": resolution_material[
                    "selected_producer_id"
                ],
                "selected_source_kind": str(selected.get("source_kind") or ""),
                "selected_age_minutes": selected.get("age_minutes"),
                "selected_failure_domain": selected_failure_domain,
                "selected_route_score": round(
                    _safe_float(selected.get("route_score"), 0.0), 6
                ),
                "selected_route_score_components": _as_dict(
                    selected.get("route_score_components")
                ),
                "selected_proof": _as_dict(selected.get("capability_proofs")).get(
                    capability_id, {}
                ),
                "failover_producer_ids": resolution_material[
                    "failover_producer_ids"
                ],
                "independent_failover_producer_ids": resolution_material[
                    "independent_failover_producer_ids"
                ],
                "usable_producer_count": len(usable_rows),
                "independent_failure_domain_count": len(
                    {
                        str(row.get("failure_domain") or "")
                        for row in usable_rows
                        if str(row.get("failure_domain") or "")
                    }
                ),
                "redundant": len(usable_rows) >= 2,
                "independently_redundant": bool(independent_failovers),
                "route_receipt_sha256": canonical_hash(resolution_material),
            }
        )
    profile_delivery_routes = _profile_delivery_routes(
        profiles,
        capability_resolutions,
        active_ingestion_policy,
    )
    profile_delivery_by_id = {
        str(row.get("profile_id") or ""): row
        for row in profile_delivery_routes
    }
    for runtime_route in runtime_sleeve_routes:
        delivery = _as_dict(
            profile_delivery_by_id.get(str(runtime_route.get("profile_id") or ""))
        )
        runtime_route.update(
            {
                "route_state": str(delivery.get("route_state") or "missing"),
                "required_capability_coverage_ratio": _safe_float(
                    delivery.get("required_capability_coverage_ratio"), 0.0
                ),
                "paper_required_capability_coverage_ratio": _safe_float(
                    delivery.get("paper_required_capability_coverage_ratio"),
                    0.0,
                ),
                "minimum_route_score": _safe_float(
                    delivery.get("minimum_route_score"), 0.0
                ),
                "average_route_score": _safe_float(
                    delivery.get("average_route_score"), 0.0
                ),
                "paper_decision_data_ready": bool(
                    delivery.get("paper_decision_data_ready", False)
                ),
                "live_decision_data_ready": bool(
                    delivery.get("live_decision_data_ready", False)
                ),
                "route_receipt_sha256": str(
                    delivery.get("route_receipt_sha256") or ""
                ),
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
    expected_runtime_profiles = len(
        _ordered_unique(_as_list(active_ingestion_policy.get("runtime_profiles")))
    )
    if len(runtime_sleeve_routes) != expected_runtime_profiles:
        structural_blockers.append("ingestion_router_runtime_profile_binding_incomplete")
    if any(not str(row.get("binding_receipt_sha256") or "") for row in bindings):
        structural_blockers.append("ingestion_router_bot_binding_receipt_missing")
    if any(
        not str(row.get("route_binding_receipt_sha256") or "")
        for row in runtime_sleeve_routes
    ):
        structural_blockers.append("ingestion_router_runtime_binding_receipt_missing")
    authority = {key: False for key in EXPECTED_SAFETY_FLAGS}
    if any(authority.values()):
        structural_blockers.append("capability_router_authority_contract_unsafe")
    ingestion_authority = {
        key: bool(_as_dict(active_ingestion_policy.get("safety_contract")).get(key))
        for key in EXPECTED_INGESTION_SAFETY_FLAGS
    }
    if any(ingestion_authority.values()):
        structural_blockers.append("ingestion_router_authority_contract_unsafe")
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
    required_independent_redundant_count = sum(
        1
        for row in capability_resolutions
        if row["required"] and row["independently_redundant"]
    )
    required_independent_redundancy_ratio = (
        required_independent_redundant_count / len(required_capabilities)
        if required_capabilities
        else 0.0
    )
    required_single_source = [
        str(row["capability_id"])
        for row in capability_resolutions
        if row["required"] and row["usable_producer_count"] == 1
    ]
    paper_ready_profile_count = sum(
        1 for row in profile_delivery_routes if row["paper_decision_data_ready"]
    )
    live_ready_profile_count = sum(
        1 for row in profile_delivery_routes if row["live_decision_data_ready"]
    )
    runtime_paper_ready_count = sum(
        1 for row in runtime_sleeve_routes if row["paper_decision_data_ready"]
    )
    runtime_live_ready_count = sum(
        1 for row in runtime_sleeve_routes if row["live_decision_data_ready"]
    )
    route_quality_values = [
        _safe_float(row.get("average_route_score"), 0.0)
        for row in profile_delivery_routes
        if _safe_int(row.get("required_capability_count"), 0) > 0
    ]
    average_profile_route_quality = (
        sum(route_quality_values) / len(route_quality_values)
        if route_quality_values
        else 0.0
    )

    routing_payload = {
        "timestamp_utc": current.isoformat(),
        "schema_version": 2,
        "catalog_id": str(catalog.get("catalog_id") or ""),
        "operating_mode": "metadata_subscription_shadow_only",
        "catalog_receipt_sha256": canonical_hash(catalog),
        "ingestion_routing_policy_id": str(
            active_ingestion_policy.get("policy_id") or ""
        ),
        "ingestion_routing_policy_receipt_sha256": canonical_hash(
            active_ingestion_policy
        ),
        "decision_policy_id": str(active_decision_policy.get("policy_id") or ""),
        "decision_policy_receipt_sha256": canonical_hash(active_decision_policy),
        "hierarchy_receipt_sha256": str(hierarchy.get("assignment_receipt_sha256") or ""),
        "subscription_profiles": sorted(
            bot_subscription_profiles, key=lambda row: row["profile_id"]
        ),
        "ingestion_route_profiles": sorted(
            profiles.values(), key=lambda row: row["profile_id"]
        ),
        "profile_delivery_routes": profile_delivery_routes,
        "runtime_sleeve_routes": runtime_sleeve_routes,
        "runtime_context_profiles": sorted(
            context_profiles.values(), key=lambda row: row["runtime_context_profile_id"]
        ),
        "bot_bindings": bindings,
        "capability_resolutions": capability_resolutions,
        "authority_contract": authority,
        "ingestion_authority_contract": ingestion_authority,
        "decision_alignment_contract": {
            "aligned": not bool(errors),
            "decision_stage": str(
                _as_dict(active_ingestion_policy.get("decision_alignment")).get(
                    "decision_stage"
                )
                or ""
            ),
            "family_count": len(
                {
                    str(row.get("decision_policy_family_id") or "")
                    for row in profiles.values()
                }
            ),
            "bot_binding_count": len(bindings),
            "runtime_profile_binding_count": len(runtime_sleeve_routes),
            "paper_and_live_share_route_definition": True,
        },
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
            "ingestion_routing_policy_receipt_sha256": routing_payload[
                "ingestion_routing_policy_receipt_sha256"
            ],
            "decision_policy_receipt_sha256": routing_payload[
                "decision_policy_receipt_sha256"
            ],
            "hierarchy_receipt_sha256": routing_payload["hierarchy_receipt_sha256"],
            "subscription_profiles": routing_payload["subscription_profiles"],
            "ingestion_route_profiles": routing_payload[
                "ingestion_route_profiles"
            ],
            "profile_delivery_routes": profile_delivery_routes,
            "runtime_sleeve_routes": runtime_sleeve_routes,
            "runtime_context_profiles": routing_payload["runtime_context_profiles"],
            "bot_bindings": bindings,
            "capability_resolutions": capability_resolutions,
        }
    )

    health_payload = {
        "timestamp_utc": current.isoformat(),
        "schema_version": 2,
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
            "subscription_profile_count": len(bot_profile_ids),
            "ingestion_route_profile_count": len(profiles),
            "runtime_route_profile_count": len(runtime_profile_ids),
            "runtime_context_profile_count": len(context_profiles),
            "decision_family_count": len(
                {
                    str(row.get("decision_policy_family_id") or "")
                    for row in profiles.values()
                }
            ),
            "profile_delivery_route_count": len(profile_delivery_routes),
            "paper_ready_profile_route_count": paper_ready_profile_count,
            "live_ready_profile_route_count": live_ready_profile_count,
            "runtime_sleeve_route_count": len(runtime_sleeve_routes),
            "runtime_paper_ready_route_count": runtime_paper_ready_count,
            "runtime_live_ready_route_count": runtime_live_ready_count,
            "average_profile_route_quality": round(
                average_profile_route_quality, 6
            ),
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
            "required_capability_independently_redundant_count": required_independent_redundant_count,
            "required_capability_independent_redundancy_ratio": round(
                required_independent_redundancy_ratio, 6
            ),
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
        "ingestion_routing_contract": {
            "policy_id": str(active_ingestion_policy.get("policy_id") or ""),
            "policy_receipt_sha256": canonical_hash(active_ingestion_policy),
            "decision_policy_id": str(active_decision_policy.get("policy_id") or ""),
            "decision_policy_receipt_sha256": canonical_hash(
                active_decision_policy
            ),
            "decision_stage": str(
                _as_dict(active_ingestion_policy.get("decision_alignment")).get(
                    "decision_stage"
                )
                or ""
            ),
            "decision_family_count": len(
                _as_dict(active_ingestion_policy.get("family_routes"))
            ),
            "profile_route_count": len(profile_delivery_routes),
            "paper_ready_profile_route_count": paper_ready_profile_count,
            "live_ready_profile_route_count": live_ready_profile_count,
            "runtime_route_count": len(runtime_sleeve_routes),
            "runtime_paper_ready_route_count": runtime_paper_ready_count,
            "runtime_live_ready_route_count": runtime_live_ready_count,
            "average_profile_route_quality": round(
                average_profile_route_quality, 6
            ),
            "paper_data_debt_blocks_global_collection": False,
            "live_data_debt_blocks_candidate_promotion": True,
            "transport_contract": deepcopy(
                _as_dict(active_ingestion_policy.get("transport_contract"))
            ),
            "authority_contract": ingestion_authority,
            "routing_artifact_receipt_sha256": routing_payload[
                "routing_receipt_sha256"
            ],
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
            "independently_redundant_required_capability_count": required_independent_redundant_count,
            "independent_required_redundancy_ratio": round(
                required_independent_redundancy_ratio, 6
            ),
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
        "ingestion_authority_contract": ingestion_authority,
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


def resolve_runtime_ingestion_route(
    routing_payload: Mapping[str, Any],
    profile: str,
    *,
    now: datetime | None = None,
    max_age_minutes: float = 30.0,
) -> dict[str, Any]:
    """Return a bounded, receipt-checked route summary for one runtime sleeve."""

    profile_name = _normalized(profile) or "default"
    routes = [
        _as_dict(row)
        for row in _as_list(routing_payload.get("runtime_sleeve_routes"))
    ]
    route = next(
        (
            row
            for row in routes
            if _normalized(row.get("runtime_profile")) == profile_name
        ),
        {},
    )
    if not route and profile_name != "default":
        route = next(
            (
                row
                for row in routes
                if _normalized(row.get("runtime_profile")) == "default"
            ),
            {},
        )
    if not route:
        return {
            "status": "missing",
            "route_state": "missing",
            "runtime_profile": profile_name,
            "paper_decision_data_ready": False,
            "live_decision_data_ready": False,
            "receipt_valid": False,
            "cause": "runtime_ingestion_route_missing",
            "authority_contract": {
                "paper_execution_authority": False,
                "live_execution_authority": False,
                "automatic_promotion_authority": False,
            },
        }

    profile_id = str(route.get("profile_id") or "")
    delivery = next(
        (
            _as_dict(row)
            for row in _as_list(routing_payload.get("profile_delivery_routes"))
            if str(_as_dict(row).get("profile_id") or "") == profile_id
        ),
        {},
    )
    binding_material = {
        "runtime_profile": str(route.get("runtime_profile") or ""),
        "decision_policy_family_id": str(
            route.get("decision_policy_family_id") or ""
        ),
        "ingestion_lane": str(route.get("ingestion_lane") or ""),
        "cadence": str(route.get("cadence") or ""),
        "profile_id": profile_id,
        "runtime_context_profile_id": str(
            route.get("runtime_context_profile_id") or ""
        ),
    }
    delivery_material = {
        "profile_id": profile_id,
        "decision_policy_family_id": str(
            delivery.get("decision_policy_family_id") or ""
        ),
        "ingestion_lane": str(delivery.get("ingestion_lane") or "core"),
        "cadence": str(delivery.get("cadence") or "intraday"),
        "paper_required_capability_ids": list(
            delivery.get("paper_required_capability_ids") or []
        ),
        "required_capability_ids": list(
            delivery.get("required_capability_ids") or []
        ),
        "delivery_routes": list(delivery.get("delivery_routes") or []),
    }
    binding_receipt_valid = bool(
        route.get("route_binding_receipt_sha256")
        and str(route.get("route_binding_receipt_sha256"))
        == canonical_hash(binding_material)
    )
    delivery_receipt_valid = bool(
        delivery.get("route_receipt_sha256")
        and str(delivery.get("route_receipt_sha256"))
        == canonical_hash(delivery_material)
    )
    timestamp = _parse_timestamp(routing_payload.get("timestamp_utc"))
    current = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    age_minutes = (
        max((current - timestamp).total_seconds() / 60.0, 0.0)
        if timestamp is not None
        else None
    )
    fresh = bool(
        age_minutes is not None
        and age_minutes <= max(float(max_age_minutes), 0.0)
    )
    receipt_valid = bool(binding_receipt_valid and delivery_receipt_valid)
    base_state = str(delivery.get("route_state") or route.get("route_state") or "missing")
    if not receipt_valid:
        status = "invalid_receipt"
        cause = "runtime_ingestion_route_receipt_invalid"
    elif not fresh:
        status = "stale"
        cause = "runtime_ingestion_route_stale"
    else:
        status = "ready" if base_state == "ready" else "collect_only"
        cause = "none" if status == "ready" else "required_route_evidence_incomplete"
    selected_producers = {
        str(_as_dict(row).get("selected_producer_id") or "")
        for row in _as_list(delivery.get("delivery_routes"))
        if str(_as_dict(row).get("selected_producer_id") or "")
    }
    summary_material = {
        "routing_receipt_sha256": str(
            routing_payload.get("routing_receipt_sha256") or ""
        ),
        "runtime_profile": str(route.get("runtime_profile") or profile_name),
        "profile_id": profile_id,
        "route_receipt_sha256": str(delivery.get("route_receipt_sha256") or ""),
        "route_state": base_state,
        "required_capability_coverage_ratio": _safe_float(
            delivery.get("required_capability_coverage_ratio"), 0.0
        ),
        "paper_required_capability_coverage_ratio": _safe_float(
            delivery.get("paper_required_capability_coverage_ratio"), 0.0
        ),
        "live_required_capability_coverage_ratio": _safe_float(
            delivery.get("required_capability_coverage_ratio"), 0.0
        ),
        "minimum_route_score": _safe_float(
            delivery.get("minimum_route_score"), 0.0
        ),
    }
    return {
        "status": status,
        "route_state": base_state,
        "cause": cause,
        "runtime_profile": str(route.get("runtime_profile") or profile_name),
        "requested_runtime_profile": profile_name,
        "fallback_profile_used": _normalized(route.get("runtime_profile"))
        != profile_name,
        "decision_policy_family_id": str(
            route.get("decision_policy_family_id") or ""
        ),
        "ingestion_lane": str(route.get("ingestion_lane") or ""),
        "cadence": str(route.get("cadence") or ""),
        "profile_id": profile_id,
        "required_capability_count": _safe_int(
            delivery.get("required_capability_count"), 0
        ),
        "usable_required_capability_count": _safe_int(
            delivery.get("usable_required_capability_count"), 0
        ),
        "required_capability_coverage_ratio": _safe_float(
            delivery.get("required_capability_coverage_ratio"), 0.0
        ),
        "paper_required_capability_coverage_ratio": _safe_float(
            delivery.get("paper_required_capability_coverage_ratio"), 0.0
        ),
        "live_required_capability_coverage_ratio": _safe_float(
            delivery.get("required_capability_coverage_ratio"), 0.0
        ),
        "minimum_route_score": _safe_float(
            delivery.get("minimum_route_score"), 0.0
        ),
        "average_route_score": _safe_float(
            delivery.get("average_route_score"), 0.0
        ),
        "paper_average_route_score": _safe_float(
            delivery.get("paper_average_route_score"), 0.0
        ),
        "live_average_route_score": _safe_float(
            delivery.get("live_average_route_score"), 0.0
        ),
        "independent_failover_coverage_ratio": _safe_float(
            delivery.get("independent_failover_coverage_ratio"), 0.0
        ),
        "selected_producer_count": len(selected_producers),
        "paper_decision_data_ready": bool(
            delivery.get("paper_decision_data_ready", False)
            and fresh
            and receipt_valid
        ),
        "live_decision_data_ready": bool(
            delivery.get("live_decision_data_ready", False)
            and fresh
            and receipt_valid
        ),
        "missing_required_capability_ids": list(
            delivery.get("missing_required_capability_ids") or []
        ),
        "missing_paper_required_capability_ids": list(
            delivery.get("missing_paper_required_capability_ids") or []
        ),
        "below_paper_score_capability_ids": list(
            delivery.get("below_paper_score_capability_ids") or []
        ),
        "degradation_policy": str(
            delivery.get("degradation_policy") or "collect_only"
        ),
        "artifact_age_minutes": round(age_minutes, 3)
        if age_minutes is not None
        else None,
        "artifact_fresh": fresh,
        "receipt_valid": receipt_valid,
        "binding_receipt_valid": binding_receipt_valid,
        "delivery_receipt_valid": delivery_receipt_valid,
        "routing_receipt_sha256": str(
            routing_payload.get("routing_receipt_sha256") or ""
        ),
        "route_receipt_sha256": str(delivery.get("route_receipt_sha256") or ""),
        "route_summary_receipt_sha256": canonical_hash(summary_material),
        "authority_contract": {
            "paper_execution_authority": False,
            "live_execution_authority": False,
            "automatic_promotion_authority": False,
        },
    }
