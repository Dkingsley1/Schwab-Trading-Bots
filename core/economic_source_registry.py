from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REGISTRY_PATH = PROJECT_ROOT / "config" / "economic_source_registry_v1.json"


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _canonical_hash(payload: Any) -> str:
    raw = json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _is_https(value: Any) -> bool:
    return str(value or "").strip().lower().startswith("https://")


def _nested_value(payload: Any, path: str) -> Any:
    current = payload
    for token in str(path or "").split("."):
        if not token or not isinstance(current, Mapping):
            return None
        current = current.get(token)
    return current


def load_economic_source_registry(path: Path | None = None) -> dict[str, Any]:
    return _read_json(path or DEFAULT_REGISTRY_PATH)


def validate_economic_source_registry(
    registry: Mapping[str, Any],
    *,
    project_root: Path = PROJECT_ROOT,
) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    if int(registry.get("schema_version", 0) or 0) != 1:
        errors.append("schema_version must be 1")
    if str(registry.get("registry_id") or "") != "economic_source_registry_v1":
        errors.append("registry_id must be economic_source_registry_v1")
    sources = registry.get("sources") if isinstance(registry.get("sources"), list) else []
    groups = registry.get("source_groups") if isinstance(registry.get("source_groups"), list) else []
    catalog = _read_json(project_root / "config" / "collector_capability_catalog_v1.json")
    sleeve_routing = _read_json(project_root / "config" / "sleeve_ingestion_routing_v2.json")
    mesh = _read_json(project_root / "config" / "decision_context_mesh_v1.json")
    producer_rows = catalog.get("producers") if isinstance(catalog.get("producers"), list) else []
    producers = {
        str(row.get("producer_id") or ""): row
        for row in producer_rows
        if isinstance(row, Mapping) and str(row.get("producer_id") or "")
    }
    decision_families = set((sleeve_routing.get("family_routes") or {}).keys())
    decision_planes = {
        str(row.get("plane_id") or "")
        for row in (mesh.get("planes") or [])
        if isinstance(row, Mapping) and str(row.get("plane_id") or "")
    }
    source_ids: list[str] = []
    for index, row in enumerate(sources):
        if not isinstance(row, Mapping):
            errors.append(f"sources[{index}] is not an object")
            continue
        source_id = str(row.get("source_id") or "").strip()
        source_ids.append(source_id)
        if not source_id:
            errors.append(f"sources[{index}] missing source_id")
            continue
        if not str(row.get("publisher") or "").strip():
            errors.append(f"{source_id}: publisher must not be empty")
        if not _is_https(row.get("official_url")):
            errors.append(f"{source_id}: official_url must use https")
        if not str(row.get("cadence") or "").strip():
            errors.append(f"{source_id}: cadence must not be empty")
        if not str(row.get("auth_mode") or "").strip():
            errors.append(f"{source_id}: auth_mode must not be empty")
        producer_id = str(row.get("producer_id") or "")
        producer = producers.get(producer_id)
        if producer is None:
            errors.append(f"{source_id}: unknown producer_id={producer_id}")
            continue
        producer_capabilities = set(producer.get("capabilities") or [])
        capability_ids = {str(value) for value in row.get("capability_ids", []) if str(value)}
        if not capability_ids:
            errors.append(f"{source_id}: capability_ids must not be empty")
        unknown_capabilities = sorted(capability_ids - producer_capabilities)
        if unknown_capabilities:
            errors.append(f"{source_id}: capabilities not declared by {producer_id}: {','.join(unknown_capabilities)}")
        granularities = {str(value) for value in row.get("granularity", []) if str(value)}
        if not granularities or not granularities <= {"macro", "micro"}:
            errors.append(f"{source_id}: granularity must contain only macro and/or micro")
        unknown_families = sorted(
            {str(value) for value in row.get("decision_family_ids", []) if str(value)} - decision_families
        )
        if unknown_families:
            errors.append(f"{source_id}: unknown decision families: {','.join(unknown_families)}")
        unknown_planes = sorted(
            {str(value) for value in row.get("decision_plane_ids", []) if str(value)} - decision_planes
        )
        if unknown_planes:
            errors.append(f"{source_id}: unknown decision planes: {','.join(unknown_planes)}")
        if not row.get("decision_family_ids"):
            errors.append(f"{source_id}: decision_family_ids must not be empty")
        if not row.get("decision_plane_ids"):
            errors.append(f"{source_id}: decision_plane_ids must not be empty")
        if not row.get("evidence_domains"):
            errors.append(f"{source_id}: evidence_domains must not be empty")
        if row.get("required_for_collection_or_paper") is not False:
            errors.append(f"{source_id}: public economic evidence must remain optional for collection and paper")
    duplicate_source_ids = sorted(source_id for source_id, count in Counter(source_ids).items() if source_id and count > 1)
    if duplicate_source_ids:
        errors.append(f"duplicate source_ids: {','.join(duplicate_source_ids)}")

    group_ids: list[str] = []
    expanded_group_member_count = 0
    for index, group in enumerate(groups):
        if not isinstance(group, Mapping):
            errors.append(f"source_groups[{index}] is not an object")
            continue
        group_id = str(group.get("group_id") or "").strip()
        group_ids.append(group_id)
        producer_id = str(group.get("producer_id") or "")
        producer = producers.get(producer_id)
        if not group_id:
            errors.append(f"source_groups[{index}] missing group_id")
            continue
        if not str(group.get("publisher") or "").strip():
            errors.append(f"{group_id}: publisher must not be empty")
        if producer is None:
            errors.append(f"{group_id}: unknown producer_id={producer_id}")
            continue
        group_capabilities = {str(value) for value in group.get("capability_ids", []) if str(value)}
        if not group_capabilities:
            errors.append(f"{group_id}: capability_ids must not be empty")
        unknown_capabilities = sorted(group_capabilities - set(producer.get("capabilities") or []))
        if unknown_capabilities:
            errors.append(f"{group_id}: capabilities not declared by {producer_id}: {','.join(unknown_capabilities)}")
        relative_path = Path(str(group.get("registry_path") or ""))
        target = (project_root / relative_path).resolve()
        try:
            target.relative_to(project_root.resolve())
        except ValueError:
            errors.append(f"{group_id}: registry_path escapes project root")
            continue
        if not target.exists():
            errors.append(f"{group_id}: registry_path missing: {relative_path}")
            continue
        nested = _read_json(target)
        members = _nested_value(nested, str(group.get("member_path") or ""))
        if not isinstance(members, list):
            errors.append(f"{group_id}: member_path does not resolve to a list")
            continue
        expanded_group_member_count += len(members)
        expected = int(group.get("expected_member_count", 0) or 0)
        if expected and len(members) != expected:
            errors.append(f"{group_id}: expected {expected} members, found {len(members)}")
        member_ids: list[str] = []
        for member in members:
            if not isinstance(member, Mapping):
                errors.append(f"{group_id}: non-object member")
                continue
            member_id = str(member.get(str(group.get("member_id_field") or "")) or "").strip()
            member_ids.append(member_id)
            if not member_id:
                errors.append(f"{group_id}: member missing id")
            if not _is_https(member.get(str(group.get("member_url_field") or ""))):
                errors.append(f"{group_id}:{member_id}: official URL must use https")
        duplicates = sorted(value for value, count in Counter(member_ids).items() if value and count > 1)
        if duplicates:
            errors.append(f"{group_id}: duplicate member ids: {','.join(duplicates)}")
        unknown_families = sorted(
            {str(value) for value in group.get("decision_family_ids", []) if str(value)} - decision_families
        )
        unknown_planes = sorted(
            {str(value) for value in group.get("decision_plane_ids", []) if str(value)} - decision_planes
        )
        if unknown_families:
            errors.append(f"{group_id}: unknown decision families: {','.join(unknown_families)}")
        if unknown_planes:
            errors.append(f"{group_id}: unknown decision planes: {','.join(unknown_planes)}")
        if not group.get("decision_family_ids"):
            errors.append(f"{group_id}: decision_family_ids must not be empty")
        if not group.get("decision_plane_ids"):
            errors.append(f"{group_id}: decision_plane_ids must not be empty")
        granularities = {str(value) for value in group.get("granularity", []) if str(value)}
        if not granularities or not granularities <= {"macro", "micro"}:
            errors.append(f"{group_id}: granularity must contain only macro and/or micro")
        if group.get("required_for_collection_or_paper") is not False:
            errors.append(f"{group_id}: grouped evidence must remain optional for collection and paper")
    duplicate_group_ids = sorted(group_id for group_id, count in Counter(group_ids).items() if group_id and count > 1)
    if duplicate_group_ids:
        errors.append(f"duplicate group_ids: {','.join(duplicate_group_ids)}")

    contract = registry.get("contract") if isinstance(registry.get("contract"), Mapping) else {}
    required_true = (
        "official_primary_sources_preferred",
        "shared_snapshot_required",
        "point_in_time_lineage_required",
        "future_observations_rejected",
        "missing_dimensions_omitted_not_zero_filled",
        "source_failures_isolated",
        "source_count_is_not_alpha_or_readiness",
    )
    required_false = (
        "per_bot_network_fanout_allowed",
        "paper_execution_authority",
        "live_execution_authority",
        "automatic_promotion_authority",
        "profitability_guaranteed",
    )
    for key in required_true:
        if contract.get(key) is not True:
            errors.append(f"contract.{key} must be true")
    for key in required_false:
        if contract.get(key) is not False:
            errors.append(f"contract.{key} must be false")
    if not any(bool(row.get("added_by_current_change")) for row in sources if isinstance(row, Mapping)):
        warnings.append("registry does not identify any newly added direct sources")
    return {
        "ok": not errors,
        "registry_id": str(registry.get("registry_id") or ""),
        "schema_version": registry.get("schema_version"),
        "direct_source_count": len(sources),
        "source_group_count": len(groups),
        "expanded_group_member_count": expanded_group_member_count,
        "total_routed_source_count": len(sources) + expanded_group_member_count,
        "errors": errors,
        "warnings": warnings,
        "registry_sha256": _canonical_hash(registry),
    }


def build_economic_source_inventory(
    registry: Mapping[str, Any],
    *,
    project_root: Path = PROJECT_ROOT,
) -> dict[str, Any]:
    validation = validate_economic_source_registry(registry, project_root=project_root)
    rows: list[dict[str, Any]] = [dict(row) for row in registry.get("sources", []) if isinstance(row, Mapping)]
    for group in registry.get("source_groups", []) if isinstance(registry.get("source_groups"), list) else []:
        if not isinstance(group, Mapping):
            continue
        nested = _read_json(project_root / str(group.get("registry_path") or ""))
        members = _nested_value(nested, str(group.get("member_path") or ""))
        for member in members if isinstance(members, list) else []:
            if not isinstance(member, Mapping):
                continue
            member_id = str(member.get(str(group.get("member_id_field") or "")) or "")
            rows.append(
                {
                    "source_id": f"{group.get('group_id')}:{member_id}",
                    "publisher": str(member.get(str(group.get("member_name_field") or "")) or member_id),
                    "official_url": str(member.get(str(group.get("member_url_field") or "")) or ""),
                    "producer_id": str(group.get("producer_id") or ""),
                    "granularity": list(group.get("granularity") or []),
                    "capability_ids": list(group.get("capability_ids") or []),
                    "decision_plane_ids": list(group.get("decision_plane_ids") or []),
                    "decision_family_ids": list(group.get("decision_family_ids") or []),
                    "evidence_domains": ["central_bank_policy"],
                    "source_group_id": str(group.get("group_id") or ""),
                    "required_for_collection_or_paper": False,
                }
            )
    granularity_counts = Counter(
        "both" if set(row.get("granularity") or []) == {"macro", "micro"}
        else "macro" if "macro" in (row.get("granularity") or [])
        else "micro"
        for row in rows
    )
    by_family: dict[str, list[str]] = defaultdict(list)
    by_plane: dict[str, list[str]] = defaultdict(list)
    by_producer: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        source_id = str(row.get("source_id") or "")
        by_producer[str(row.get("producer_id") or "")].append(source_id)
        for family_id in row.get("decision_family_ids", []) or []:
            by_family[str(family_id)].append(source_id)
        for plane_id in row.get("decision_plane_ids", []) or []:
            by_plane[str(plane_id)].append(source_id)
    return {
        "registry_id": str(registry.get("registry_id") or ""),
        "validation": validation,
        "summary": {
            "direct_source_count": validation["direct_source_count"],
            "grouped_source_count": validation["expanded_group_member_count"],
            "total_routed_source_count": len(rows),
            "official_primary_source_count": sum(1 for row in rows if row.get("source_tier") != "secondary_crosscheck_only"),
            "secondary_crosscheck_source_count": sum(1 for row in rows if row.get("source_tier") == "secondary_crosscheck_only"),
            "new_direct_source_ids": sorted(
                str(row.get("source_id") or "") for row in rows if row.get("added_by_current_change") is True
            ),
            "granularity_counts": dict(sorted(granularity_counts.items())),
            "producer_count": len([key for key in by_producer if key]),
            "decision_family_count": len([key for key in by_family if key]),
            "decision_plane_count": len([key for key in by_plane if key]),
            "source_count_is_not_alpha_or_readiness": True,
        },
        "sources": sorted(rows, key=lambda row: str(row.get("source_id") or "")),
        "routes_by_producer": {key: sorted(value) for key, value in sorted(by_producer.items()) if key},
        "routes_by_decision_family": {key: sorted(value) for key, value in sorted(by_family.items()) if key},
        "routes_by_decision_plane": {key: sorted(value) for key, value in sorted(by_plane.items()) if key},
        "authority": {
            "inventory_and_context_only": True,
            "paper_execution_authority": False,
            "live_execution_authority": False,
            "automatic_promotion_authority": False,
        },
    }
