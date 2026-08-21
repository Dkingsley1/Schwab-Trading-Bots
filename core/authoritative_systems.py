from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import urlparse


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "authoritative_systems_v1.json"
EXPECTED_REFERENCE_COUNT = 20


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def load_registry(path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("authoritative systems registry must be an object")
    return payload


def validate_registry(
    registry: Mapping[str, Any], *, project_root: str | Path = PROJECT_ROOT
) -> dict[str, Any]:
    errors: list[str] = []
    controls = _mapping(registry.get("controls"))
    required = [str(value) for value in registry.get("required_control_ids") or []]
    references = [
        row for row in registry.get("references") or [] if isinstance(row, Mapping)
    ]
    authority = _mapping(registry.get("authority"))

    if int(registry.get("schema_version") or 0) != 1:
        errors.append("schema_version_must_be_1")
    if len(references) != EXPECTED_REFERENCE_COUNT:
        errors.append(f"reference_count_must_equal_{EXPECTED_REFERENCE_COUNT}")
    if len(set(required)) != 8 or len(required) != 8:
        errors.append("required_control_count_must_equal_8")
    if set(required) != set(controls):
        errors.append("control_registry_does_not_match_required_control_ids")
    if authority.get("influence_only") is not True:
        errors.append("reference_authority_must_be_influence_only")
    for key, value in authority.items():
        if str(key).startswith("can_") and bool(value):
            errors.append(f"forbidden_authority_enabled:{key}")

    root = Path(project_root)
    for control_id in required:
        row = _mapping(controls.get(control_id))
        owner = str(row.get("owner") or "").strip()
        test = str(row.get("test") or "").strip()
        if not owner or not (root / owner).is_file():
            errors.append(f"control_owner_missing:{control_id}:{owner or 'unset'}")
        if not test or not (root / test).is_file():
            errors.append(f"control_test_missing:{control_id}:{test or 'unset'}")

    seen_ids: set[str] = set()
    seen_urls: set[str] = set()
    control_adoption = {control_id: 0 for control_id in required}
    for row in references:
        reference_id = str(row.get("id") or "").strip()
        url = str(row.get("official_url") or "").strip()
        parsed = urlparse(url)
        if not reference_id:
            errors.append("reference_id_missing")
        elif reference_id in seen_ids:
            errors.append(f"duplicate_reference_id:{reference_id}")
        seen_ids.add(reference_id)
        if parsed.scheme != "https" or not parsed.netloc:
            errors.append(f"reference_url_not_https:{reference_id or 'unknown'}")
        elif url in seen_urls:
            errors.append(f"duplicate_reference_url:{reference_id or 'unknown'}")
        seen_urls.add(url)
        if row.get("primary_source") is not True:
            errors.append(f"reference_not_primary_source:{reference_id or 'unknown'}")
        if row.get("influence_only") is not True:
            errors.append(f"reference_not_influence_only:{reference_id or 'unknown'}")
        adopted = [str(value) for value in row.get("adopted_controls") or []]
        unknown = sorted(set(adopted) - set(required))
        if unknown:
            errors.append(
                f"reference_unknown_controls:{reference_id or 'unknown'}:{','.join(unknown)}"
            )
        for control_id in set(adopted) & set(required):
            control_adoption[control_id] += 1
        if not row.get("principles"):
            errors.append(f"reference_principles_missing:{reference_id or 'unknown'}")

    for control_id, count in control_adoption.items():
        if count <= 0:
            errors.append(f"control_has_no_authoritative_reference:{control_id}")

    semantics = _mapping(registry.get("readiness_semantics"))
    required_semantics = (
        "reference_count_is_not_evidence",
        "implementation_ready_is_not_live_ready",
        "paper_live_equivalence_requires_observed_pairs",
        "profitability_requires_candidate_bound_forward_post_cost_evidence",
        "live_execution_remains_independently_gated",
    )
    for key in required_semantics:
        if semantics.get(key) is not True:
            errors.append(f"readiness_semantic_missing:{key}")

    return {
        "ok": not errors,
        "errors": errors,
        "reference_count": len(references),
        "control_count": len(controls),
        "control_adoption": control_adoption,
        "authority_safe": not any(
            error.startswith("forbidden_authority") for error in errors
        ),
    }
