#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from scripts.ops.long_runtime_common import (
        iso_now,
        payload_age_minutes,
        write_payload,
    )
else:
    from .long_runtime_common import (
        PROJECT_ROOT,
        iso_now,
        payload_age_minutes,
        write_payload,
    )


DEFAULT_OUT_PATH = (
    PROJECT_ROOT
    / "governance"
    / "health"
    / "canonical_representation_audit_latest.json"
)
SOURCE_OF_TRUTH_PATH = PROJECT_ROOT / "docs" / "architecture" / "SOURCE_OF_TRUTH.md"

SEVERITY_RANK = {"ready": 0, "advisory": 1, "watch": 2, "degraded": 3, "critical": 4}
FAIL_SEVERITIES = {"degraded", "critical"}
READY_STATUSES = {"ok", "ready", "guarded_ready", "advisory"}
READY_WITH_DEBT_STATUSES = {
    "ready_with_evidence_debt",
    "ready_with_review_debt",
}
FORBIDDEN_STRATEGY_AUTHORITY_FIELDS = (
    "can_create_intent",
    "can_reverse_intent",
    "can_increase_quantity",
    "can_allocate_capital",
    "can_change_labels",
    "can_grant_promotion",
    "can_submit_live_order",
)
PAPER_COUNT_FIELDS_STRICT = (
    "total_bots",
    "active_bots",
    "data_collection_active_bots",
    "paper_live_data_enabled_bots",
)
PAPER_COUNT_FIELDS_ADVISORY = (
    "legacy_bootstrap_paper_bots",
    "collection_until_standard_bots",
    "standard_promoted_paper_bots",
)


def _resolve(project_root: Path, raw: Any) -> Path:
    path = Path(str(raw or ""))
    return path if path.is_absolute() else project_root / path


def _as_dict(raw: Any) -> dict[str, Any]:
    return raw if isinstance(raw, dict) else {}


def _as_list(raw: Any) -> list[Any]:
    return raw if isinstance(raw, list) else []


def _as_bool(raw: Any, default: bool = False) -> bool:
    if raw is None:
        return default
    if isinstance(raw, bool):
        return raw
    return str(raw).strip().lower() in {"1", "true", "yes", "on", "enabled"}


def _as_int(raw: Any, default: int = 0) -> int:
    try:
        return int(raw)
    except Exception:
        return default


def _as_float(raw: Any, default: float = 0.0) -> float:
    try:
        return float(raw)
    except Exception:
        return default


def _collection_len(raw: Any) -> int:
    if isinstance(raw, (list, tuple, dict, set)):
        return len(raw)
    return 0


def _sha256(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return ""


def _read_json(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    record: dict[str, Any] = {
        "path": str(path),
        "exists": path.exists(),
        "json_valid": False,
        "sha256": "",
        "size_bytes": 0,
        "age_minutes": None,
    }
    if not path.exists():
        return {}, record
    try:
        stat = path.stat()
        record["size_bytes"] = stat.st_size
        record["sha256"] = _sha256(path)
    except OSError:
        pass
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        record["parse_error"] = str(exc)[:500]
        return {}, record
    if not isinstance(payload, dict):
        record["parse_error"] = "json_root_not_object"
        return {}, record
    record["json_valid"] = True
    record["age_minutes"] = payload_age_minutes(payload, path)
    return payload, record


def _metric(payload: dict[str, Any], key: str, default: Any = None) -> Any:
    if key in payload and payload.get(key) is not None:
        return payload.get(key)
    summary = _as_dict(payload.get("summary"))
    if key in summary and summary.get(key) is not None:
        return summary.get(key)
    return default


def _status(payload: dict[str, Any]) -> str:
    return str(payload.get("overall_status") or payload.get("status") or "").strip()


def _status_ready_or_debt(payload: dict[str, Any]) -> bool:
    status = _status(payload).lower()
    return (
        _as_bool(payload.get("ok"), False)
        or status in READY_STATUSES | READY_WITH_DEBT_STATUSES
    )


def _sleeve_names(raw: Any) -> set[str]:
    names: set[str] = set()
    if isinstance(raw, dict):
        names.update(str(key).strip() for key in raw if str(key).strip())
        return names
    if not isinstance(raw, list):
        return names
    for item in raw:
        if isinstance(item, str) and item.strip():
            names.add(item.strip())
        elif isinstance(item, dict):
            for key in ("name", "sleeve_id", "id"):
                value = str(item.get(key) or "").strip()
                if value:
                    names.add(value)
                    break
    return names


def _contract_sleeve_names(contracts: dict[str, Any]) -> set[str]:
    names: set[str] = set()
    for key, raw_contract in contracts.items():
        contract = _as_dict(raw_contract)
        sleeve_id = str(contract.get("sleeve_id") or "").strip()
        if not sleeve_id and str(key).startswith("sleeve::"):
            parts = str(key).split("::")
            if len(parts) >= 2:
                sleeve_id = parts[1]
        if sleeve_id:
            names.add(sleeve_id)
    return names


def _sorted_sample(values: set[str] | list[str], limit: int = 12) -> list[str]:
    return sorted(str(value) for value in values if str(value))[:limit]


def _add_finding(
    findings: list[dict[str, Any]],
    *,
    check_id: str,
    severity: str,
    finding_id: str,
    message: str,
    canonical_source: str,
    represented_artifact: str,
    details: dict[str, Any] | None = None,
    remediation: str = "",
) -> None:
    findings.append(
        {
            "finding_id": finding_id,
            "check_id": check_id,
            "severity": severity,
            "message": message,
            "canonical_source": canonical_source,
            "represented_artifact": represented_artifact,
            "details": details or {},
            "remediation": remediation,
        }
    )


def _check_record(
    checks: dict[str, dict[str, Any]],
    check_id: str,
    *,
    canonical_sources: list[str],
    represented_artifacts: list[str],
) -> dict[str, Any]:
    record = {
        "check_id": check_id,
        "status": "ready",
        "canonical_sources": canonical_sources,
        "represented_artifacts": represented_artifacts,
        "metrics": {},
    }
    checks[check_id] = record
    return record


def _artifact_missing_finding(
    findings: list[dict[str, Any]],
    *,
    check_id: str,
    severity: str,
    finding_id: str,
    canonical_source: str,
    represented_artifact: str,
) -> None:
    _add_finding(
        findings,
        check_id=check_id,
        severity=severity,
        finding_id=finding_id,
        message="The represented runtime artifact is missing or is not valid JSON.",
        canonical_source=canonical_source,
        represented_artifact=represented_artifact,
        remediation="Refresh the owning control and re-run this audit.",
    )


def _audit_system_role_contract(
    project_root: Path,
    checks: dict[str, dict[str, Any]],
    findings: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    config_path = project_root / "config" / "system_role_contracts_v1.json"
    live_path = (
        project_root / "governance" / "health" / "system_role_contract_latest.json"
    )
    config, config_record = _read_json(config_path)
    live, live_record = _read_json(live_path)
    records = {
        "system_role_contract_config": config_record,
        "system_role_contract_latest": live_record,
    }
    check = _check_record(
        checks,
        "system_role_contract_alignment",
        canonical_sources=[str(config_path), str(SOURCE_OF_TRUTH_PATH)],
        represented_artifacts=[str(live_path)],
    )
    if not config:
        _artifact_missing_finding(
            findings,
            check_id=check["check_id"],
            severity="critical",
            finding_id="system_role_contract_config_missing",
            canonical_source=str(config_path),
            represented_artifact=str(live_path),
        )
        return records
    if not live:
        _artifact_missing_finding(
            findings,
            check_id=check["check_id"],
            severity="degraded",
            finding_id="system_role_contract_report_missing",
            canonical_source=str(config_path),
            represented_artifact=str(live_path),
        )
        return records

    expected_counts = {
        "role_count": _collection_len(config.get("roles")),
        "component_count": _collection_len(config.get("components")),
        "state_domain_count": _collection_len(config.get("state_domains")),
        "control_surface_binding_count": _collection_len(
            config.get("control_surface_bindings")
        ),
        "exclusive_action_count": _collection_len(
            config.get("exclusive_action_owners") or config.get("exclusive_actions")
        ),
    }
    check["metrics"]["expected_counts"] = expected_counts
    check["metrics"]["represented_counts"] = {
        key: _metric(live, key) for key in expected_counts
    }
    for key, expected in expected_counts.items():
        represented = _metric(live, key)
        if represented != expected:
            _add_finding(
                findings,
                check_id=check["check_id"],
                severity="degraded",
                finding_id=f"system_role_{key}_mismatch",
                message=f"System role contract count mismatch for {key}.",
                canonical_source=str(config_path),
                represented_artifact=str(live_path),
                details={"expected": expected, "represented": represented},
                remediation="./scripts/ops/opsctl.sh system-role-contract --json",
            )
    config_policy = str(config.get("policy_id") or "")
    live_policy = str(live.get("policy_id") or "")
    check["metrics"]["policy_id"] = {
        "canonical": config_policy,
        "represented": live_policy,
    }
    if config_policy and live_policy and config_policy != live_policy:
        _add_finding(
            findings,
            check_id=check["check_id"],
            severity="degraded",
            finding_id="system_role_policy_id_mismatch",
            message="System role policy id differs between config and live report.",
            canonical_source=str(config_path),
            represented_artifact=str(live_path),
            details={"expected": config_policy, "represented": live_policy},
            remediation="./scripts/ops/opsctl.sh system-role-contract --json",
        )
    coverage = _as_float(_metric(live, "registry_role_coverage_ratio"), 0.0)
    conflict_count = _as_int(_metric(live, "authority_conflict_count"), 0)
    check["metrics"]["registry_role_coverage_ratio"] = coverage
    check["metrics"]["authority_conflict_count"] = conflict_count
    if coverage < 1.0:
        _add_finding(
            findings,
            check_id=check["check_id"],
            severity="degraded",
            finding_id="system_role_registry_coverage_incomplete",
            message="Not every registry role resolves to a canonical system role.",
            canonical_source=str(config_path),
            represented_artifact=str(live_path),
            details={"registry_role_coverage_ratio": coverage},
            remediation="./scripts/ops/opsctl.sh system-role-contract --json",
        )
    if conflict_count > 0:
        _add_finding(
            findings,
            check_id=check["check_id"],
            severity="critical",
            finding_id="system_role_authority_conflicts_present",
            message="The role contract reports authority conflicts.",
            canonical_source=str(config_path),
            represented_artifact=str(live_path),
            details={"authority_conflict_count": conflict_count},
            remediation="Resolve the conflicting role ownership before widening execution.",
        )
    if _as_list(live.get("blockers")):
        _add_finding(
            findings,
            check_id=check["check_id"],
            severity="degraded",
            finding_id="system_role_contract_blockers_present",
            message="The role contract report has blockers.",
            canonical_source=str(config_path),
            represented_artifact=str(live_path),
            details={"blockers": _as_list(live.get("blockers"))},
            remediation="./scripts/ops/opsctl.sh system-role-contract --json",
        )
    if _as_list(live.get("warnings")):
        _add_finding(
            findings,
            check_id=check["check_id"],
            severity="watch",
            finding_id="system_role_contract_warnings_present",
            message="The role contract report has warnings.",
            canonical_source=str(config_path),
            represented_artifact=str(live_path),
            details={"warnings": _as_list(live.get("warnings"))},
            remediation="./scripts/ops/opsctl.sh system-role-contract --json",
        )
    return records


def _audit_bot_organization(
    project_root: Path,
    checks: dict[str, dict[str, Any]],
    findings: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    registry_path = project_root / "master_bot_registry.json"
    org_path = project_root / "governance" / "health" / "bot_organization_latest.json"
    registry, registry_record = _read_json(registry_path)
    org, org_record = _read_json(org_path)
    records = {
        "master_bot_registry": registry_record,
        "bot_organization_latest": org_record,
    }
    check = _check_record(
        checks,
        "bot_registry_organization_alignment",
        canonical_sources=[str(registry_path), str(SOURCE_OF_TRUTH_PATH)],
        represented_artifacts=[str(org_path)],
    )
    if not registry:
        _artifact_missing_finding(
            findings,
            check_id=check["check_id"],
            severity="critical",
            finding_id="master_bot_registry_missing",
            canonical_source=str(registry_path),
            represented_artifact=str(org_path),
        )
        return records
    if not org:
        _artifact_missing_finding(
            findings,
            check_id=check["check_id"],
            severity="degraded",
            finding_id="bot_organization_report_missing",
            canonical_source=str(registry_path),
            represented_artifact=str(org_path),
        )
        return records

    registry_summary = _as_dict(registry.get("summary"))
    registry_count = _as_int(registry_summary.get("total_bots"), 0)
    active_count = _as_int(registry_summary.get("active_bots"), 0)
    represented_registry_count = _as_int(_metric(org, "registry_bot_count"), 0)
    organized_count = _as_int(_metric(org, "organized_bot_count"), 0)
    coverage = _as_float(_metric(org, "organization_coverage_ratio"), 0.0)
    explicit_ratio = _as_float(_metric(org, "explicit_sleeve_ratio"), 0.0)
    review_count = _as_int(_metric(org, "review_queue_count"), 0)
    setup_summary = _as_dict(org.get("bot_setup_summary"))
    setup_hardening = _as_dict(setup_summary.get("hardening"))
    tripwire_summary = _as_dict(org.get("tripwire_summary"))
    tripwire_hardening = _as_dict(tripwire_summary.get("hardening"))
    active_tripwires = _as_list(tripwire_summary.get("active_tripwires"))
    blocking_tripwires = _as_list(tripwire_summary.get("blocking_tripwires"))
    check["metrics"] = {
        "registry_total_bots": registry_count,
        "registry_active_bots": active_count,
        "represented_registry_bot_count": represented_registry_count,
        "organized_bot_count": organized_count,
        "organization_coverage_ratio": coverage,
        "explicit_sleeve_ratio": explicit_ratio,
        "review_queue_count": review_count,
        "structural_grade": _metric(org, "structural_grade", ""),
        "assignment_grade": _metric(org, "grade", ""),
        "setup_hardening_status": setup_hardening.get("overall_status", ""),
        "tripwire_status": tripwire_summary.get("overall_status", ""),
        "tripwire_hardening_status": tripwire_hardening.get("overall_status", ""),
        "active_tripwire_count": _as_int(
            tripwire_summary.get("active_tripwire_count"), 0
        ),
        "blocking_tripwire_count": _as_int(
            tripwire_summary.get("blocking_tripwire_count"), 0
        ),
        "tripwire_severity_counts": _as_dict(tripwire_summary.get("severity_counts")),
        "tripwire_category_counts": _as_dict(tripwire_summary.get("category_counts")),
    }
    if registry_count and represented_registry_count != registry_count:
        _add_finding(
            findings,
            check_id=check["check_id"],
            severity="degraded",
            finding_id="bot_organization_registry_count_mismatch",
            message="Bot organization report does not represent the registry bot count.",
            canonical_source=str(registry_path),
            represented_artifact=str(org_path),
            details={
                "registry_total_bots": registry_count,
                "represented_registry_bot_count": represented_registry_count,
            },
            remediation="./scripts/ops/opsctl.sh bot-organization --json",
        )
    if registry_count and organized_count != registry_count:
        _add_finding(
            findings,
            check_id=check["check_id"],
            severity="degraded",
            finding_id="bot_organization_organized_count_mismatch",
            message="Organized bot count does not match the registry bot count.",
            canonical_source=str(registry_path),
            represented_artifact=str(org_path),
            details={
                "registry_total_bots": registry_count,
                "organized_bot_count": organized_count,
            },
            remediation="./scripts/ops/opsctl.sh bot-organization --json",
        )
    if coverage < 1.0:
        _add_finding(
            findings,
            check_id=check["check_id"],
            severity="degraded",
            finding_id="bot_organization_coverage_incomplete",
            message="Not every registered bot has an organization assignment.",
            canonical_source=str(registry_path),
            represented_artifact=str(org_path),
            details={"organization_coverage_ratio": coverage},
            remediation="./scripts/ops/opsctl.sh bot-organization --json",
        )
    invalid_ids = _as_list(org.get("invalid_assignment_bot_ids"))
    duplicate_ids = _as_list(org.get("duplicate_bot_ids"))
    if invalid_ids or duplicate_ids:
        _add_finding(
            findings,
            check_id=check["check_id"],
            severity="degraded",
            finding_id="bot_organization_assignment_integrity_failed",
            message="The bot organization report has invalid or duplicate assignments.",
            canonical_source=str(registry_path),
            represented_artifact=str(org_path),
            details={
                "invalid_assignment_bot_ids": invalid_ids[:25],
                "duplicate_bot_ids": duplicate_ids[:25],
            },
            remediation="./scripts/ops/opsctl.sh bot-organization --json",
        )
    if setup_hardening and str(setup_hardening.get("overall_status") or "") != "ready":
        _add_finding(
            findings,
            check_id=check["check_id"],
            severity="degraded",
            finding_id="bot_setup_hardening_not_ready",
            message="Bot setup metadata hardening is not ready.",
            canonical_source=str(registry_path),
            represented_artifact=str(org_path),
            details={"setup_hardening": setup_hardening},
            remediation="./scripts/ops/opsctl.sh bot-organization --json",
        )
    if (
        tripwire_hardening
        and str(tripwire_hardening.get("overall_status") or "") != "ready"
    ):
        _add_finding(
            findings,
            check_id=check["check_id"],
            severity="degraded",
            finding_id="bot_tripwire_hardening_not_ready",
            message="Bot tripwire contract hardening is not ready.",
            canonical_source=str(registry_path),
            represented_artifact=str(org_path),
            details={"tripwire_hardening": tripwire_hardening},
            remediation="./scripts/ops/opsctl.sh bot-organization --json",
        )
    if blocking_tripwires:
        _add_finding(
            findings,
            check_id=check["check_id"],
            severity="degraded",
            finding_id="bot_organization_blocking_tripwires_present",
            message="Bot organization has degraded or critical tripwires active.",
            canonical_source=str(registry_path),
            represented_artifact=str(org_path),
            details={"blocking_tripwires": blocking_tripwires[:25]},
            remediation="Repair blocking tripwires by owner/action before widening runtime routing.",
        )
    elif active_tripwires:
        _add_finding(
            findings,
            check_id=check["check_id"],
            severity="advisory",
            finding_id="bot_organization_active_tripwires_present",
            message="Bot organization has active non-blocking tripwires with owners, actions, and required evidence.",
            canonical_source=str(registry_path),
            represented_artifact=str(org_path),
            details={
                "active_tripwire_count": len(active_tripwires),
                "severity_counts": _as_dict(tripwire_summary.get("severity_counts")),
                "category_counts": _as_dict(tripwire_summary.get("category_counts")),
                "active_tripwires": active_tripwires[:25],
            },
            remediation="Work the active tripwire evidence queue; do not treat advisory tripwires as execution authority.",
        )
    elif not tripwire_summary and (review_count > 0 or explicit_ratio < 0.95):
        _add_finding(
            findings,
            check_id=check["check_id"],
            severity="advisory",
            finding_id="bot_organization_review_debt_present",
            message="Bot organization is structurally complete, but review debt remains.",
            canonical_source=str(registry_path),
            represented_artifact=str(org_path),
            details={
                "review_queue_count": review_count,
                "explicit_sleeve_ratio": explicit_ratio,
                "structural_grade": _metric(org, "structural_grade", ""),
                "assignment_grade": _metric(org, "grade", ""),
            },
            remediation="Continue classifying review-queue bots; do not treat review debt as execution authority.",
        )
    if registry_summary.get("deletion_guard_ok") is False:
        _add_finding(
            findings,
            check_id=check["check_id"],
            severity="advisory",
            finding_id="registry_deletion_guard_holding",
            message="The registry deletion guard is holding because training success is not confirmed.",
            canonical_source=str(registry_path),
            represented_artifact=str(org_path),
            details={
                "deletion_guard_reason": registry_summary.get(
                    "deletion_guard_reason", ""
                )
            },
            remediation="Keep deletion guarded until trained targets are confirmed.",
        )
    return records


def _audit_sleeve_strategy_contracts(
    project_root: Path,
    checks: dict[str, dict[str, Any]],
    findings: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    config_path = project_root / "config" / "sleeve_strategy_contracts_v1.json"
    contracts_path = (
        project_root
        / "governance"
        / "research"
        / "sleeve_strategy_contracts_latest.json"
    )
    org_path = project_root / "governance" / "health" / "bot_organization_latest.json"
    config, config_record = _read_json(config_path)
    contracts_payload, contracts_record = _read_json(contracts_path)
    org, org_record = _read_json(org_path)
    records = {
        "sleeve_strategy_contracts_config": config_record,
        "sleeve_strategy_contracts_latest": contracts_record,
        "bot_organization_latest_for_sleeves": org_record,
    }
    check = _check_record(
        checks,
        "sleeve_strategy_contract_representation",
        canonical_sources=[str(config_path), str(SOURCE_OF_TRUTH_PATH)],
        represented_artifacts=[str(contracts_path), str(org_path)],
    )
    if not config:
        _artifact_missing_finding(
            findings,
            check_id=check["check_id"],
            severity="critical",
            finding_id="sleeve_strategy_config_missing",
            canonical_source=str(config_path),
            represented_artifact=str(contracts_path),
        )
        return records
    if not contracts_payload:
        _artifact_missing_finding(
            findings,
            check_id=check["check_id"],
            severity="degraded",
            finding_id="sleeve_strategy_contract_report_missing",
            canonical_source=str(config_path),
            represented_artifact=str(contracts_path),
        )
        return records

    source_manifest_path = _resolve(project_root, config.get("source_manifest"))
    manifest, manifest_record = _read_json(source_manifest_path)
    records["sleeve_strategy_source_manifest"] = manifest_record
    core_sleeves = _sleeve_names(config.get("sleeves"))
    included_sleeves = _sleeve_names(config.get("included_sleeves"))
    expansion_sleeves = _sleeve_names(manifest.get("sleeves"))
    canonical_universe = core_sleeves | included_sleeves | expansion_sleeves
    contracts = _as_dict(contracts_payload.get("contracts"))
    contract_sleeves = _contract_sleeve_names(contracts)
    bot_org_sleeves = _sleeve_names(
        _as_dict(_as_dict(org.get("counts")).get("sleeves"))
    )
    check["metrics"] = {
        "core_sleeve_count": len(core_sleeves),
        "included_sleeve_count": len(included_sleeves),
        "expansion_manifest_sleeve_count": len(expansion_sleeves),
        "canonical_universe_sleeve_count": len(canonical_universe),
        "represented_contract_sleeve_count": len(contract_sleeves),
        "represented_bot_organization_sleeve_count": len(bot_org_sleeves),
        "reported_contract_count": contracts_payload.get("contract_count"),
        "actual_contract_count": len(contracts),
        "derived_sleeve_policy_present": bool(config.get("derived_sleeve_policy")),
        "source_manifest_path": str(source_manifest_path),
        "source_manifest_present": bool(manifest),
    }
    if config.get("source_manifest") and not manifest:
        _add_finding(
            findings,
            check_id=check["check_id"],
            severity="degraded",
            finding_id="sleeve_strategy_source_manifest_missing",
            message="Sleeve strategy config references a source manifest that is missing or invalid.",
            canonical_source=str(config_path),
            represented_artifact=str(source_manifest_path),
            details={"source_manifest": config.get("source_manifest")},
            remediation="Restore the sleeve strategy source manifest before regenerating contracts.",
        )
    reported_count = _as_int(contracts_payload.get("contract_count"), -1)
    if reported_count >= 0 and reported_count != len(contracts):
        _add_finding(
            findings,
            check_id=check["check_id"],
            severity="degraded",
            finding_id="sleeve_strategy_contract_count_mismatch",
            message="Reported contract count does not match the represented contract map.",
            canonical_source=str(config_path),
            represented_artifact=str(contracts_path),
            details={
                "reported_contract_count": reported_count,
                "actual_contract_count": len(contracts),
            },
            remediation="./scripts/ops/opsctl.sh sleeve-strategy-specialization --json",
        )
    missing_core = core_sleeves - contract_sleeves
    if missing_core:
        _add_finding(
            findings,
            check_id=check["check_id"],
            severity="degraded",
            finding_id="core_sleeves_missing_from_strategy_contracts",
            message="One or more explicit core sleeves are missing from generated strategy contracts.",
            canonical_source=str(config_path),
            represented_artifact=str(contracts_path),
            details={"missing_core_sleeves": _sorted_sample(missing_core, 50)},
            remediation="./scripts/ops/opsctl.sh sleeve-strategy-specialization --json",
        )
    missing_expansion = expansion_sleeves - contract_sleeves
    if missing_expansion:
        _add_finding(
            findings,
            check_id=check["check_id"],
            severity="degraded",
            finding_id="expansion_sleeves_missing_from_strategy_contracts",
            message="One or more manifest sleeves are missing from generated strategy contracts.",
            canonical_source=str(source_manifest_path),
            represented_artifact=str(contracts_path),
            details={
                "missing_expansion_sleeves": _sorted_sample(missing_expansion, 50)
            },
            remediation="./scripts/ops/opsctl.sh sleeve-strategy-specialization --json",
        )
    unexplained_contract_sleeves = contract_sleeves - canonical_universe
    if unexplained_contract_sleeves:
        _add_finding(
            findings,
            check_id=check["check_id"],
            severity="degraded",
            finding_id="strategy_contract_sleeves_not_backed_by_canonical_universe",
            message="Generated strategy contracts include sleeves not backed by config or manifest.",
            canonical_source=str(config_path),
            represented_artifact=str(contracts_path),
            details={
                "unexplained_contract_sleeves": _sorted_sample(
                    unexplained_contract_sleeves, 50
                )
            },
            remediation="Add the sleeves to the source manifest or remove the generated contracts.",
        )
    if expansion_sleeves - core_sleeves and not config.get("derived_sleeve_policy"):
        _add_finding(
            findings,
            check_id=check["check_id"],
            severity="degraded",
            finding_id="derived_sleeves_without_policy",
            message="Expanded sleeves exist without a derived-sleeve policy.",
            canonical_source=str(config_path),
            represented_artifact=str(contracts_path),
            details={"expanded_sleeve_count": len(expansion_sleeves - core_sleeves)},
            remediation="Add a derived_sleeve_policy before admitting expanded sleeve contracts.",
        )
    bot_sleeves_outside_strategy_contracts = bot_org_sleeves - contract_sleeves
    check["metrics"]["bot_sleeves_outside_strategy_contract_count"] = len(
        bot_sleeves_outside_strategy_contracts
    )
    check["metrics"]["bot_sleeves_outside_strategy_contract_examples"] = _sorted_sample(
        bot_sleeves_outside_strategy_contracts
    )

    authority_violations: list[dict[str, Any]] = []
    for strategy_id, raw_contract in contracts.items():
        contract = _as_dict(raw_contract)
        authority = _as_dict(contract.get("authority"))
        measurement = _as_dict(contract.get("measurement_parameters"))
        for field in FORBIDDEN_STRATEGY_AUTHORITY_FIELDS:
            if authority.get(field) is True:
                authority_violations.append(
                    {"strategy_id": strategy_id, "field": field}
                )
        for field in ("automatic_live_promotion_allowed", "may_allocate_capital"):
            if measurement.get(field) is True:
                authority_violations.append(
                    {"strategy_id": strategy_id, "field": field}
                )
    top_authority = _as_dict(contracts_payload.get("authority_contract"))
    for field in FORBIDDEN_STRATEGY_AUTHORITY_FIELDS:
        if top_authority.get(field) is True:
            authority_violations.append(
                {"strategy_id": "authority_contract", "field": field}
            )
    if authority_violations:
        _add_finding(
            findings,
            check_id=check["check_id"],
            severity="critical",
            finding_id="strategy_contract_authority_leak",
            message="A strategy contract claims forbidden order, capital, label, or promotion authority.",
            canonical_source=str(config_path),
            represented_artifact=str(contracts_path),
            details={"violations": authority_violations[:50]},
            remediation="Remove execution/capital/promotion authority from strategy contracts before any runtime use.",
        )
    return records


def _audit_paper_execution_authority(
    project_root: Path,
    checks: dict[str, dict[str, Any]],
    findings: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    registry_path = project_root / "master_bot_registry.json"
    paper_path = (
        project_root / "governance" / "health" / "paper_live_data_standard_latest.json"
    )
    dashboard_path = (
        project_root / "governance" / "health" / "runtime_gate_dashboard_latest.json"
    )
    registry, registry_record = _read_json(registry_path)
    paper, paper_record = _read_json(paper_path)
    dashboard, dashboard_record = _read_json(dashboard_path)
    records = {
        "master_bot_registry_for_paper": registry_record,
        "paper_live_data_standard_latest": paper_record,
        "runtime_gate_dashboard_latest_for_paper": dashboard_record,
    }
    check = _check_record(
        checks,
        "paper_authority_representation",
        canonical_sources=[
            str(paper_path),
            str(project_root / "core" / "profitability_hardening.py"),
            str(project_root / "core" / "base_trader.py"),
            str(SOURCE_OF_TRUTH_PATH),
        ],
        represented_artifacts=[str(registry_path), str(dashboard_path)],
    )
    if not paper:
        _artifact_missing_finding(
            findings,
            check_id=check["check_id"],
            severity="degraded",
            finding_id="paper_live_data_standard_missing",
            canonical_source=str(paper_path),
            represented_artifact=str(dashboard_path),
        )
        return records
    registry_summary = _as_dict(registry.get("summary"))
    paper_counts = _as_dict(paper.get("counts_after"))
    check["metrics"] = {
        "paper_standard_status": _status(paper),
        "paper_standard_ok": paper.get("ok"),
        "registry_counts": {
            key: registry_summary.get(key) for key in PAPER_COUNT_FIELDS_STRICT
        },
        "paper_standard_counts": {
            key: paper_counts.get(key) for key in PAPER_COUNT_FIELDS_STRICT
        },
        "direct_execution_allowed_bots": paper_counts.get(
            "direct_execution_allowed_bots"
        ),
        "paper_execution_authority_bots": paper_counts.get(
            "paper_execution_authority_bots"
        ),
        "paper_probation_authority_bots": paper_counts.get(
            "paper_probation_authority_bots"
        ),
        "live_trading_enabled_bots": paper_counts.get("live_trading_enabled_bots"),
    }
    if not _status_ready_or_debt(paper):
        _add_finding(
            findings,
            check_id=check["check_id"],
            severity="degraded",
            finding_id="paper_live_data_standard_not_ready",
            message="Paper live-data standard is not ready.",
            canonical_source=str(paper_path),
            represented_artifact=str(dashboard_path),
            details={"overall_status": _status(paper), "ok": paper.get("ok")},
            remediation="./scripts/ops/opsctl.sh paper-live-data-standard --json",
        )
    for key in PAPER_COUNT_FIELDS_STRICT:
        registry_value = registry_summary.get(key)
        paper_value = paper_counts.get(key)
        if (
            registry_value is not None
            and paper_value is not None
            and registry_value != paper_value
        ):
            _add_finding(
                findings,
                check_id=check["check_id"],
                severity="degraded",
                finding_id=f"paper_standard_{key}_count_mismatch",
                message=f"Paper standard and registry disagree on {key}.",
                canonical_source=str(paper_path),
                represented_artifact=str(registry_path),
                details={"registry": registry_value, "paper_standard": paper_value},
                remediation="./scripts/ops/opsctl.sh paper-live-data-standard --apply --json",
            )
    advisory_mismatches = {}
    for key in PAPER_COUNT_FIELDS_ADVISORY:
        registry_value = registry_summary.get(key)
        paper_value = paper_counts.get(key)
        if (
            registry_value is not None
            and paper_value is not None
            and registry_value != paper_value
        ):
            advisory_mismatches[key] = {
                "registry": registry_value,
                "paper_standard": paper_value,
            }
    if advisory_mismatches:
        _add_finding(
            findings,
            check_id=check["check_id"],
            severity="watch",
            finding_id="paper_standard_legacy_count_representation_mismatch",
            message="Registry legacy paper counters differ from the paper standard counters.",
            canonical_source=str(paper_path),
            represented_artifact=str(registry_path),
            details=advisory_mismatches,
            remediation="Trust paper_live_data_standard for paper authority; reconcile registry summary counters on the next registry refresh.",
        )
    live_enabled = _as_int(paper_counts.get("live_trading_enabled_bots"), 0)
    if live_enabled > 0:
        _add_finding(
            findings,
            check_id=check["check_id"],
            severity="critical",
            finding_id="paper_standard_claims_live_enabled_bots",
            message="Paper live-data standard claims live trading is enabled.",
            canonical_source=str(paper_path),
            represented_artifact=str(paper_path),
            details={"live_trading_enabled_bots": live_enabled},
            remediation="Remove live authority from the paper standard output immediately.",
        )
    attention = _as_list(_as_dict(dashboard.get("overall")).get("attention"))
    launcher_summary = _as_dict(
        _as_dict(_as_dict(dashboard.get("artifacts")).get("all_sleeves_launcher")).get(
            "summary"
        )
    )
    paper_execution_ready = launcher_summary.get("paper_execution_ready")
    check["metrics"]["dashboard_paper_execution_ready"] = paper_execution_ready
    check["metrics"]["dashboard_attention_contains_paper_guard"] = (
        "paper_execution_safety_guard_active" in attention
    )
    check["metrics"]["paper_guard_is_expected_when_execution_ready_false"] = bool(
        paper.get("ok") is True
        and paper_execution_ready is False
        and "paper_execution_safety_guard_active" in attention
    )
    if (
        paper_execution_ready is False
        and "paper_execution_safety_guard_active" not in attention
    ):
        _add_finding(
            findings,
            check_id=check["check_id"],
            severity="watch",
            finding_id="dashboard_missing_paper_execution_guard_attention",
            message="Launcher says paper execution is not ready, but dashboard attention omits the paper guard.",
            canonical_source=str(dashboard_path),
            represented_artifact=str(dashboard_path),
            details={"launcher_paper_execution_ready": paper_execution_ready},
            remediation="./scripts/ops/opsctl.sh runtime-gate-dashboard --json",
        )
    return records


def _audit_profitability_truth(
    project_root: Path,
    checks: dict[str, dict[str, Any]],
    findings: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    bot_profit_path = (
        project_root
        / "governance"
        / "health"
        / "bot_profitability_scalability_latest.json"
    )
    paper_profit_path = (
        project_root
        / "governance"
        / "health"
        / "paper_profitability_control_latest.json"
    )
    firewall_path = (
        project_root
        / "governance"
        / "health"
        / "profitability_evidence_firewall_latest.json"
    )
    sleeve_selector_path = (
        project_root
        / "governance"
        / "health"
        / "sleeve_scalability_selector_latest.json"
    )
    bot_profit, bot_profit_record = _read_json(bot_profit_path)
    paper_profit, paper_profit_record = _read_json(paper_profit_path)
    firewall, firewall_record = _read_json(firewall_path)
    sleeve_selector, sleeve_selector_record = _read_json(sleeve_selector_path)
    records = {
        "bot_profitability_scalability_latest": bot_profit_record,
        "paper_profitability_control_latest": paper_profit_record,
        "profitability_evidence_firewall_latest": firewall_record,
        "sleeve_scalability_selector_latest": sleeve_selector_record,
    }
    check = _check_record(
        checks,
        "profitability_truth_representation",
        canonical_sources=[
            str(project_root / "config" / "bot_profitability_scalability_v1.json"),
            str(paper_profit_path),
            str(SOURCE_OF_TRUTH_PATH),
        ],
        represented_artifacts=[
            str(bot_profit_path),
            str(firewall_path),
            str(sleeve_selector_path),
        ],
    )
    control_grade = str(bot_profit.get("control_grade") or "")
    economic_grade = str(bot_profit.get("economic_and_scale_evidence_grade") or "")
    evidence_debt = _as_list(bot_profit.get("evidence_debt"))
    candidate_binding = _as_dict(bot_profit.get("candidate_binding"))
    selector_summary = _as_dict(sleeve_selector.get("summary"))
    application_eligible = _as_int(
        selector_summary.get("application_eligible_sleeve_count"), 0
    )
    earned_scalability = _as_int(
        selector_summary.get("earned_scalability_goal_count"), 0
    )
    claim_ready = any(
        _as_bool(payload.get("profitability_claim_ready"), False)
        for payload in (bot_profit, paper_profit, firewall)
    )
    promotion_ready = any(
        _as_bool(payload.get("promotion_evidence_ready"), False)
        or _as_bool(payload.get("raw_promotion_evidence_ready"), False)
        for payload in (bot_profit, paper_profit, firewall)
    )
    check["metrics"] = {
        "control_grade": control_grade,
        "economic_and_scale_evidence_grade": economic_grade,
        "evidence_debt": evidence_debt,
        "candidate_binding": candidate_binding,
        "paper_profitability_status": _status(paper_profit),
        "profitability_firewall_status": _status(firewall),
        "application_eligible_sleeve_count": application_eligible,
        "earned_scalability_goal_count": earned_scalability,
        "profitability_claim_ready": claim_ready,
        "promotion_evidence_ready": promotion_ready,
    }
    weak_evidence_grade = economic_grade.upper() in {
        "",
        "F",
        "D",
        "N/A",
        "NONE",
        "NULL",
    }
    if claim_ready and weak_evidence_grade:
        _add_finding(
            findings,
            check_id=check["check_id"],
            severity="critical",
            finding_id="profitability_claim_with_weak_evidence",
            message="A profitability-ready claim is present while economic evidence is weak or missing.",
            canonical_source=str(bot_profit_path),
            represented_artifact=str(firewall_path),
            details={
                "economic_and_scale_evidence_grade": economic_grade,
                "profitability_claim_ready": claim_ready,
            },
            remediation="Remove profitability claim readiness until post-cost evidence clears the firewall.",
        )
    elif control_grade.upper() in {"A", "A+"} and weak_evidence_grade:
        _add_finding(
            findings,
            check_id=check["check_id"],
            severity="advisory",
            finding_id="profitability_control_grade_separated_from_economic_evidence",
            message="Profitability controls are strong, but economic-and-scale evidence remains weak.",
            canonical_source=str(
                project_root / "config" / "bot_profitability_scalability_v1.json"
            ),
            represented_artifact=str(bot_profit_path),
            details={
                "control_grade": control_grade,
                "economic_and_scale_evidence_grade": economic_grade,
                "evidence_debt": evidence_debt,
                "candidate_binding_bound": candidate_binding.get("bound"),
            },
            remediation="Keep reporting control grade and economic evidence grade separately.",
        )
    if paper_profit and not _status_ready_or_debt(paper_profit):
        _add_finding(
            findings,
            check_id=check["check_id"],
            severity="watch",
            finding_id="paper_profitability_missing_evidence",
            message="Paper profitability control is blocked by missing evidence.",
            canonical_source=str(paper_profit_path),
            represented_artifact=str(paper_profit_path),
            details={
                "overall_status": _status(paper_profit),
                "paper_summary": _as_dict(paper_profit.get("paper_summary")),
            },
            remediation="./scripts/ops/opsctl.sh paper-profitability-control --apply --json",
        )
    if firewall and not _status_ready_or_debt(firewall):
        _add_finding(
            findings,
            check_id=check["check_id"],
            severity="watch",
            finding_id="profitability_firewall_blocked",
            message="The profitability evidence firewall is blocking claims or promotion.",
            canonical_source=str(firewall_path),
            represented_artifact=str(firewall_path),
            details={"overall_status": _status(firewall), "ok": firewall.get("ok")},
            remediation="./scripts/ops/opsctl.sh profitability-evidence-firewall --json",
        )
    if sleeve_selector and application_eligible <= 0:
        _add_finding(
            findings,
            check_id=check["check_id"],
            severity="advisory",
            finding_id="no_application_eligible_sleeves",
            message="Sleeve selector has no application-eligible sleeves yet.",
            canonical_source=str(sleeve_selector_path),
            represented_artifact=str(sleeve_selector_path),
            details={
                "application_eligible_sleeve_count": application_eligible,
                "earned_scalability_goal_count": earned_scalability,
                "evidence_debt": _as_list(sleeve_selector.get("evidence_debt")),
            },
            remediation="Collect candidate-bound post-cost, persistence, execution, benchmark, and account-route evidence before scaling.",
        )
    return records


def _audit_broker_account_truth(
    project_root: Path,
    checks: dict[str, dict[str, Any]],
    findings: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    snapshot_path = (
        project_root
        / "governance"
        / "health"
        / "schwab_account_snapshot_refresh_latest.json"
    )
    shared_path = (
        project_root
        / "governance"
        / "health"
        / "broker_truth_shared_snapshot_schwab_latest.json"
    )
    boundary_path = (
        project_root
        / "governance"
        / "health"
        / "schwab_broker_boundary_control_latest.json"
    )
    capability_path = (
        project_root
        / "governance"
        / "health"
        / "schwab_account_capability_truth_latest.json"
    )
    position_study_path = (
        project_root / "governance" / "health" / "account_position_study_latest.json"
    )
    capability_script = (
        project_root / "scripts" / "ops" / "schwab_account_capability_truth.py"
    )
    snapshot, snapshot_record = _read_json(snapshot_path)
    shared, shared_record = _read_json(shared_path)
    boundary, boundary_record = _read_json(boundary_path)
    capability, capability_record = _read_json(capability_path)
    position_study, position_study_record = _read_json(position_study_path)
    records = {
        "schwab_account_snapshot_refresh_latest": snapshot_record,
        "broker_truth_shared_snapshot_schwab_latest": shared_record,
        "schwab_broker_boundary_control_latest": boundary_record,
        "schwab_account_capability_truth_latest": capability_record,
        "account_position_study_latest": position_study_record,
    }
    check = _check_record(
        checks,
        "broker_account_truth_representation",
        canonical_sources=[
            str(snapshot_path),
            str(boundary_path),
            str(capability_script),
            str(SOURCE_OF_TRUTH_PATH),
        ],
        represented_artifacts=[
            str(shared_path),
            str(capability_path),
            str(position_study_path),
        ],
    )
    position_accounts = [
        row for row in _as_list(position_study.get("accounts")) if isinstance(row, dict)
    ]
    embedded_capabilities = [
        _as_dict(row.get("account_capability_truth"))
        for row in position_accounts
        if _as_dict(row.get("account_capability_truth"))
    ]
    embedded_capability_live_authority_count = sum(
        1
        for item in embedded_capabilities
        if _as_dict(item.get("operator_classification")).get("live_execution_authority")
        is True
        or _as_dict(item.get("canary_preflight")).get("live_execution_authority")
        is True
        or item.get("live_execution_authority") is True
    )
    check["metrics"] = {
        "account_snapshot_status": _status(snapshot),
        "account_snapshot_ok": snapshot.get("ok"),
        "broker_truth_ok": snapshot.get("broker_truth_ok"),
        "broker_truth_status": snapshot.get("broker_truth_status"),
        "broker_truth_v2_grade": snapshot.get("broker_truth_v2_grade"),
        "account_count": snapshot.get("account_count"),
        "failed_account_count": snapshot.get("failed_account_count"),
        "published_as_canonical": snapshot.get("published_as_canonical"),
        "shared_snapshot_present": bool(shared),
        "boundary_status": _status(boundary),
        "boundary_ok": boundary.get("ok"),
        "capability_script_present": capability_script.exists(),
        "capability_artifact_present": bool(capability),
        "account_position_study_present": bool(position_study),
        "embedded_account_capability_count": len(embedded_capabilities),
        "embedded_account_capability_covers_accounts": bool(position_accounts)
        and len(embedded_capabilities) == len(position_accounts),
        "embedded_capability_live_authority_count": embedded_capability_live_authority_count,
    }
    if not snapshot:
        _artifact_missing_finding(
            findings,
            check_id=check["check_id"],
            severity="degraded",
            finding_id="schwab_account_snapshot_missing",
            canonical_source=str(snapshot_path),
            represented_artifact=str(shared_path),
        )
    elif (
        not _status_ready_or_debt(snapshot) or snapshot.get("broker_truth_ok") is False
    ):
        _add_finding(
            findings,
            check_id=check["check_id"],
            severity="degraded",
            finding_id="schwab_account_snapshot_not_ready",
            message="Schwab account snapshot refresh is not ready.",
            canonical_source=str(snapshot_path),
            represented_artifact=str(shared_path),
            details={
                "overall_status": _status(snapshot),
                "broker_truth_ok": snapshot.get("broker_truth_ok"),
                "broker_truth_status": snapshot.get("broker_truth_status"),
                "provider_failure": snapshot.get("provider_failure"),
            },
            remediation="./scripts/ops/opsctl.sh schwab-account-snapshot-refresh --json",
        )
    if snapshot and snapshot.get("broker_truth_mismatch_count") not in {None, 0}:
        _add_finding(
            findings,
            check_id=check["check_id"],
            severity="degraded",
            finding_id="broker_truth_mismatch_present",
            message="Broker truth reconciliation reports mismatches.",
            canonical_source=str(snapshot_path),
            represented_artifact=str(shared_path),
            details={
                "broker_truth_mismatch_count": snapshot.get(
                    "broker_truth_mismatch_count"
                )
            },
            remediation="./scripts/ops/opsctl.sh schwab-account-snapshot-refresh --json",
        )
    if not shared:
        _artifact_missing_finding(
            findings,
            check_id=check["check_id"],
            severity="degraded",
            finding_id="broker_truth_shared_snapshot_missing",
            canonical_source=str(snapshot_path),
            represented_artifact=str(shared_path),
        )
    if not boundary:
        _artifact_missing_finding(
            findings,
            check_id=check["check_id"],
            severity="degraded",
            finding_id="schwab_broker_boundary_missing",
            canonical_source=str(boundary_path),
            represented_artifact=str(boundary_path),
        )
    elif not _status_ready_or_debt(boundary):
        _add_finding(
            findings,
            check_id=check["check_id"],
            severity="degraded",
            finding_id="schwab_broker_boundary_not_ready",
            message="Schwab broker boundary control is not ready.",
            canonical_source=str(boundary_path),
            represented_artifact=str(boundary_path),
            details={"overall_status": _status(boundary), "ok": boundary.get("ok")},
            remediation="./scripts/ops/opsctl.sh schwab-broker-boundary --json",
        )
    if embedded_capability_live_authority_count:
        _add_finding(
            findings,
            check_id=check["check_id"],
            severity="critical",
            finding_id="embedded_account_capability_claims_live_authority",
            message="Embedded account capability truth claims live execution authority.",
            canonical_source=str(position_study_path),
            represented_artifact=str(position_study_path),
            details={
                "embedded_capability_live_authority_count": embedded_capability_live_authority_count
            },
            remediation="Remove live_execution_authority from embedded account capability truth before widening execution.",
        )
    if capability_script.exists() and not capability and not embedded_capabilities:
        _add_finding(
            findings,
            check_id=check["check_id"],
            severity="advisory",
            finding_id="schwab_account_capability_truth_not_published",
            message="Account capability truth builder exists, but no latest runtime artifact is published.",
            canonical_source=str(capability_script),
            represented_artifact=str(capability_path),
            details={"capability_artifact_path": str(capability_path)},
            remediation="Publish an account-capability latest artifact or explicitly document that account capability is embedded in the account-position study.",
        )
    elif capability and not _status_ready_or_debt(capability):
        _add_finding(
            findings,
            check_id=check["check_id"],
            severity="watch",
            finding_id="schwab_account_capability_truth_not_ready",
            message="Published account capability truth is not ready.",
            canonical_source=str(capability_script),
            represented_artifact=str(capability_path),
            details={"overall_status": _status(capability), "ok": capability.get("ok")},
            remediation="Refresh account capability truth and preserve live_execution_authority=false unless explicitly approved.",
        )
    return records


def _audit_dashboard_attention(
    project_root: Path,
    checks: dict[str, dict[str, Any]],
    findings: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    dashboard_path = (
        project_root / "governance" / "health" / "runtime_gate_dashboard_latest.json"
    )
    storage_path = (
        project_root / "governance" / "health" / "ingestion_storage_control_latest.json"
    )
    external_path = (
        project_root / "governance" / "health" / "external_backlog_drain_latest.json"
    )
    dashboard, dashboard_record = _read_json(dashboard_path)
    storage, storage_record = _read_json(storage_path)
    external, external_record = _read_json(external_path)
    records = {
        "runtime_gate_dashboard_latest": dashboard_record,
        "ingestion_storage_control_latest": storage_record,
        "external_backlog_drain_latest": external_record,
    }
    check = _check_record(
        checks,
        "dashboard_attention_representation",
        canonical_sources=[
            str(dashboard_path),
            str(storage_path),
            str(SOURCE_OF_TRUTH_PATH),
        ],
        represented_artifacts=[str(dashboard_path), str(external_path)],
    )
    if not dashboard:
        _artifact_missing_finding(
            findings,
            check_id=check["check_id"],
            severity="degraded",
            finding_id="runtime_gate_dashboard_missing",
            canonical_source=str(dashboard_path),
            represented_artifact=str(dashboard_path),
        )
        return records
    overall = _as_dict(dashboard.get("overall"))
    tiers = _as_dict(overall.get("attention_tiers"))
    critical = _as_list(tiers.get("critical"))
    degraded = _as_list(tiers.get("degraded"))
    watch = _as_list(tiers.get("watch"))
    advisory = _as_list(tiers.get("advisory"))
    storage_summary = _as_dict(
        _as_dict(
            _as_dict(dashboard.get("artifacts")).get("ingestion_storage_control")
        ).get("summary")
    )
    external_summary = _as_dict(
        _as_dict(
            _as_dict(dashboard.get("artifacts")).get("external_backlog_drain")
        ).get("summary")
    )
    check["metrics"] = {
        "dashboard_status": overall.get("status") or dashboard.get("overall_status"),
        "dashboard_ok": overall.get("ok") if "ok" in overall else dashboard.get("ok"),
        "critical_attention_count": len(critical),
        "degraded_attention_count": len(degraded),
        "watch_attention_count": len(watch),
        "advisory_attention_count": len(advisory),
        "watch_attention": watch,
        "advisory_attention": advisory,
        "storage_overall_status": storage_summary.get("overall_status")
        or _status(storage),
        "storage_pressure_index": storage_summary.get("pressure_index")
        or storage.get("pressure_index"),
        "external_backlog_status": external_summary.get("overall_status")
        or _status(external),
        "external_writer_busy": (
            external_summary.get("writer_busy")
            if external_summary
            else external.get("writer_busy")
        ),
        "external_aged_candidate_files": (
            external_summary.get("aged_candidate_files")
            if external_summary
            else external.get("aged_candidate_files")
        ),
    }
    if critical or degraded:
        _add_finding(
            findings,
            check_id=check["check_id"],
            severity="degraded",
            finding_id="dashboard_reports_degraded_or_critical_attention",
            message="Runtime dashboard has degraded or critical attention items.",
            canonical_source=str(dashboard_path),
            represented_artifact=str(dashboard_path),
            details={"critical": critical, "degraded": degraded},
            remediation="Resolve degraded/critical dashboard items before widening runtime scope.",
        )
    if storage and _status(storage) == "ready" and external:
        estimated_total = _as_float(
            _as_dict(storage.get("bounded_recovery_contract")).get(
                "estimated_total_drain_minutes"
            ),
            _as_float(storage_summary.get("estimated_total_drain_minutes"), 0.0),
        )
        writer_busy = _as_bool(
            (
                external_summary.get("writer_busy")
                if external_summary
                else external.get("writer_busy")
            ),
            False,
        )
        aged_files = _as_int(
            (
                external_summary.get("aged_candidate_files")
                if external_summary
                else external.get("aged_candidate_files")
            ),
            0,
        )
        visible_external_attention = any(
            item
            in {
                "external_backlog_drain_recommended",
                "external_backlog_drain_writer_busy",
            }
            for item in [*critical, *degraded, *watch, *advisory]
        )
        if (
            visible_external_attention
            and writer_busy
            and estimated_total <= 1.0
            and aged_files <= 3
        ):
            _add_finding(
                findings,
                check_id=check["check_id"],
                severity="advisory",
                finding_id="tiny_external_backlog_tail_visible",
                message="Storage is ready, but a tiny external backlog tail is still visible as watch attention.",
                canonical_source=str(storage_path),
                represented_artifact=str(dashboard_path),
                details={
                    "estimated_total_drain_minutes": estimated_total,
                    "aged_candidate_files": aged_files,
                    "writer_busy": writer_busy,
                },
                remediation="Let the external writer finish or downgrade tiny-tail backlog attention when storage remains stable.",
            )
    return records


def _finalize_checks(
    checks: dict[str, dict[str, Any]],
    findings: list[dict[str, Any]],
) -> None:
    highest_by_check: dict[str, str] = {}
    for finding in findings:
        check_id = str(finding.get("check_id") or "")
        severity = str(finding.get("severity") or "ready")
        previous = highest_by_check.get(check_id, "ready")
        if SEVERITY_RANK.get(severity, 0) > SEVERITY_RANK.get(previous, 0):
            highest_by_check[check_id] = severity
    for check_id, check in checks.items():
        check["status"] = highest_by_check.get(check_id, "ready")


def _overall_status(findings: list[dict[str, Any]]) -> str:
    worst = "ready"
    for finding in findings:
        severity = str(finding.get("severity") or "ready")
        if SEVERITY_RANK.get(severity, 0) > SEVERITY_RANK.get(worst, 0):
            worst = severity
    return worst


def build_payload(project_root: Path = PROJECT_ROOT) -> dict[str, Any]:
    project_root = project_root.resolve()
    checks: dict[str, dict[str, Any]] = {}
    findings: list[dict[str, Any]] = []
    source_records: dict[str, dict[str, Any]] = {}
    for audit in (
        _audit_system_role_contract,
        _audit_bot_organization,
        _audit_sleeve_strategy_contracts,
        _audit_paper_execution_authority,
        _audit_profitability_truth,
        _audit_broker_account_truth,
        _audit_dashboard_attention,
    ):
        source_records.update(audit(project_root, checks, findings))
    _finalize_checks(checks, findings)
    severity_counts = {
        severity: sum(1 for finding in findings if finding.get("severity") == severity)
        for severity in ("critical", "degraded", "watch", "advisory")
    }
    status = _overall_status(findings)
    ok = not any(finding.get("severity") in FAIL_SEVERITIES for finding in findings)
    return {
        "schema_version": 1,
        "policy_id": "canonical_representation_audit_v1",
        "timestamp_utc": iso_now(),
        "ok": ok,
        "overall_status": status,
        "summary": {
            "check_count": len(checks),
            "finding_count": len(findings),
            **{
                f"{severity}_count": count
                for severity, count in severity_counts.items()
            },
            "canonical_source_count": sum(
                len(check.get("canonical_sources", [])) for check in checks.values()
            ),
            "represented_artifact_count": sum(
                len(check.get("represented_artifacts", [])) for check in checks.values()
            ),
            "failed_check_count": sum(
                1 for check in checks.values() if check.get("status") in FAIL_SEVERITIES
            ),
        },
        "authority_contract": {
            "read_only_audit": True,
            "can_mutate_registry": False,
            "can_change_source_code": False,
            "can_submit_paper_order": False,
            "can_submit_live_order": False,
            "can_allocate_capital": False,
            "can_promote_candidate": False,
            "can_claim_profitability": False,
            "purpose": "Compare canonical source-of-truth definitions to represented runtime artifacts.",
        },
        "checks": checks,
        "findings": findings,
        "source_files": source_records,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Audit canonical source-of-truth files against represented runtime artifacts."
    )
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--out-file", type=Path)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    project_root = args.project_root.resolve()
    out_path = args.out_file or (
        project_root / "governance" / "health" / DEFAULT_OUT_PATH.name
    )
    if not out_path.is_absolute():
        out_path = project_root / out_path
    payload = build_payload(project_root)
    write_payload(out_path, payload)
    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        summary = payload["summary"]
        print(
            "canonical_representation_audit "
            f"status={payload['overall_status']} ok={str(payload['ok']).lower()} "
            f"checks={summary['check_count']} findings={summary['finding_count']} "
            f"degraded={summary['degraded_count']} critical={summary['critical_count']}"
        )
    return 0 if payload["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
