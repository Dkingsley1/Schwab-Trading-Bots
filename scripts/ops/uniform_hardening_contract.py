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
    from scripts.ops.long_runtime_common import load_json, ordered_unique, payload_age_minutes, write_payload
else:
    from .long_runtime_common import PROJECT_ROOT, load_json, ordered_unique, payload_age_minutes, write_payload


DEFAULT_CONFIG = PROJECT_ROOT / "config" / "production_uniform_hardening_v1.json"
DEFAULT_OUT = PROJECT_ROOT / "governance" / "health" / "uniform_hardening_contract_latest.json"
SCHEMA_VERSION = 1
GRADE_RANK = {"F": 0, "D": 1, "C": 2, "B": 3, "A": 4, "A+": 5}
COMMON_CONTROL_IDS = (
    "artifact_contract",
    "freshness_slo",
    "owner_command",
    "bounded_recovery",
    "regression_tests",
    "fail_closed_policy",
    "bounded_automation",
    "atomic_publication",
    "live_authority_boundary",
    "candidate_scope_and_evidence_separation",
)


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    return str(value or "").strip().lower() in {"1", "true", "yes", "on", "ready", "ok"}


def _project_path(project_root: Path, value: Any) -> Path:
    path = Path(str(value or ""))
    return path if path.is_absolute() else project_root / path


def _value_at(payload: dict[str, Any], dotted_path: str) -> Any:
    current: Any = payload
    for part in str(dotted_path or "").split("."):
        if not part or not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def _status(payload: dict[str, Any]) -> str:
    for key in ("overall_status", "status", "state"):
        value = str(payload.get(key) or "").strip().lower()
        if value:
            return value
    if "ok" in payload:
        return "ready" if _truthy(payload.get("ok")) else "blocked"
    return "present" if payload else "missing"


def _grade(value: Any) -> str:
    return str(value or "").strip().upper()


def _grade_at_least(value: Any, floor: Any) -> bool:
    actual = _grade(value)
    required = _grade(floor)
    return bool(actual and required and GRADE_RANK.get(actual, -1) >= GRADE_RANK.get(required, 999))


def _score_grade(score: float) -> str:
    if score >= 100.0:
        return "A+"
    if score >= 90.0:
        return "A"
    if score >= 80.0:
        return "B"
    if score >= 70.0:
        return "C"
    if score >= 60.0:
        return "D"
    return "F"


def _command_routable(project_root: Path, command: list[Any]) -> tuple[bool, str]:
    parts = [str(item).strip() for item in command if str(item).strip()]
    if not parts:
        return False, "command_missing"
    executable = _project_path(project_root, parts[0])
    if not executable.is_file():
        return False, f"command_executable_missing:{parts[0]}"
    if executable.name != "opsctl.sh":
        return True, "direct_script_present"
    if len(parts) < 2:
        return False, "opsctl_route_missing"
    try:
        text = executable.read_text(encoding="utf-8")
    except OSError:
        return False, "opsctl_unreadable"
    route = parts[1]
    return (route in text, "opsctl_route_present" if route in text else f"opsctl_route_missing:{route}")


def _structural_controls(project_root: Path, domain: dict[str, Any]) -> list[dict[str, Any]]:
    artifacts = [row for row in _as_list(domain.get("artifacts")) if isinstance(row, dict)]
    owner_ok, owner_detail = _command_routable(project_root, _as_list(domain.get("owner_command")))
    recovery_ok, recovery_detail = _command_routable(project_root, _as_list(domain.get("recovery_command")))
    tests = [str(item).strip() for item in _as_list(domain.get("regression_tests")) if str(item).strip()]
    missing_tests = [test for test in tests if not _project_path(project_root, test).is_file()]
    controls = {
        "artifact_contract": (
            bool(artifacts)
            and all(str(row.get("artifact_id") or "").strip() and str(row.get("path") or "").strip() for row in artifacts),
            f"declared_artifacts={len(artifacts)}",
        ),
        "freshness_slo": (
            bool(artifacts) and all(_safe_float(row.get("max_age_minutes"), 0.0) > 0.0 for row in artifacts),
            "every_artifact_has_positive_max_age",
        ),
        "owner_command": (owner_ok, owner_detail),
        "bounded_recovery": (recovery_ok, recovery_detail),
        "regression_tests": (bool(tests) and not missing_tests, f"missing={','.join(missing_tests)}" if missing_tests else f"present={len(tests)}"),
        "fail_closed_policy": (str(domain.get("failure_policy") or "").strip() == "fail_closed", str(domain.get("failure_policy") or "")),
        "bounded_automation": (bool(domain.get("bounded_automation", False)), "bounded_automation_declared"),
        "atomic_publication": (bool(domain.get("atomic_publication_required", False)), "atomic_publication_required"),
        "live_authority_boundary": (
            str(domain.get("live_execution_authority") or "").strip() in {"none", "operator_gated"},
            f"authority={str(domain.get('live_execution_authority') or 'missing')}",
        ),
        "candidate_scope_and_evidence_separation": (
            bool(_as_list(domain.get("candidate_scopes"))) and bool(domain.get("evidence_grade_separate", False)),
            f"scopes={','.join(str(item) for item in _as_list(domain.get('candidate_scopes')))}",
        ),
    }
    return [
        {"control_id": control_id, "ready": bool(controls[control_id][0]), "detail": controls[control_id][1]}
        for control_id in COMMON_CONTROL_IDS
    ]


def _artifact_row(project_root: Path, spec: dict[str, Any], *, now: datetime) -> dict[str, Any]:
    artifact_id = str(spec.get("artifact_id") or "unnamed").strip()
    path = _project_path(project_root, spec.get("path"))
    payload = load_json(path)
    exists = path.is_file() and bool(payload)
    status_path = str(spec.get("status_path") or "").strip()
    status = (
        str(_value_at(payload, status_path) or "").strip().lower()
        if status_path
        else _status(payload)
    )
    max_age_minutes = _safe_float(spec.get("max_age_minutes"), 0.0)
    age_minutes = payload_age_minutes(payload, path, now=now) if payload else None
    fresh = bool(age_minutes is not None and age_minutes <= max_age_minutes) if max_age_minutes > 0.0 else exists
    ready_statuses = {str(item).strip().lower() for item in _as_list(spec.get("ready_statuses")) if str(item).strip()}
    status_ready = bool(not ready_statuses or status in ready_statuses)
    truthy_paths = [str(item) for item in _as_list(spec.get("truthy_paths"))]
    falsey_paths = [str(item) for item in _as_list(spec.get("falsey_paths"))]
    zero_paths = [str(item) for item in _as_list(spec.get("zero_paths"))]
    truthy_ready = all(_truthy(_value_at(payload, key)) for key in truthy_paths)
    falsey_ready = all(not _truthy(_value_at(payload, key)) for key in falsey_paths)
    zero_ready = all(_safe_float(_value_at(payload, key), 1.0) == 0.0 for key in zero_paths)
    grade_requirements = _as_dict(spec.get("grade_requirements"))
    grade_rows = [
        {
            "path": key,
            "actual": _grade(_value_at(payload, key)),
            "floor": _grade(floor),
            "ready": _grade_at_least(_value_at(payload, key), floor),
        }
        for key, floor in grade_requirements.items()
    ]
    grades_ready = all(row["ready"] for row in grade_rows)
    ready = bool(exists and fresh and status_ready and truthy_ready and falsey_ready and zero_ready and grades_ready)
    blockers = ordered_unique(
        [
            f"{artifact_id}_missing" if not exists else "",
            f"{artifact_id}_stale" if exists and not fresh else "",
            f"{artifact_id}_status_not_ready" if exists and not status_ready else "",
            f"{artifact_id}_truthy_contract_failed" if exists and not truthy_ready else "",
            f"{artifact_id}_falsey_contract_failed" if exists and not falsey_ready else "",
            f"{artifact_id}_zero_contract_failed" if exists and not zero_ready else "",
            f"{artifact_id}_grade_floor_failed" if exists and not grades_ready else "",
        ]
    )
    return {
        "artifact_id": artifact_id,
        "path": str(path),
        "required": bool(spec.get("required", True)),
        "exists": exists,
        "status": status,
        "status_path": status_path,
        "ready": ready,
        "blockers": blockers,
        "age_minutes": round(float(age_minutes), 4) if age_minutes is not None else None,
        "max_age_minutes": max_age_minutes,
        "fresh": fresh,
        "truthy_paths": truthy_paths,
        "falsey_paths": falsey_paths,
        "zero_paths": zero_paths,
        "grade_requirements": grade_rows,
    }


def _domain_row(
    project_root: Path,
    domain: dict[str, Any],
    *,
    now: datetime,
    critical_domains: set[str],
    evidence_domains: set[str],
) -> dict[str, Any]:
    domain_id = str(domain.get("domain_id") or "unnamed").strip()
    structural_controls = _structural_controls(project_root, domain)
    structural_ready_count = sum(1 for row in structural_controls if row["ready"])
    structural_score = round(100.0 * structural_ready_count / max(len(structural_controls), 1), 2)
    structural_ready = structural_ready_count == len(structural_controls)
    artifacts = [
        _artifact_row(project_root, spec, now=now)
        for spec in _as_list(domain.get("artifacts"))
        if isinstance(spec, dict)
    ]
    required_artifacts = [row for row in artifacts if row["required"]]
    supporting_artifacts = [row for row in artifacts if not row["required"]]
    required_ready_count = sum(1 for row in required_artifacts if row["ready"])
    evidence_score = round(100.0 * required_ready_count / max(len(required_artifacts), 1), 2)
    evidence_ready = bool(required_artifacts and required_ready_count == len(required_artifacts))
    critical = domain_id in critical_domains
    evidence_domain = domain_id in evidence_domains
    if not structural_ready:
        status = "blocked"
    elif evidence_ready:
        status = "ready"
    elif evidence_domain:
        status = "evidence_pending"
    else:
        status = "blocked"
    structural_blockers = [
        f"{domain_id}:{row['control_id']}" for row in structural_controls if not row["ready"]
    ]
    evidence_blockers = [
        f"{domain_id}:{blocker}"
        for row in required_artifacts
        if not row["ready"]
        for blocker in row["blockers"]
    ]
    supporting_warnings = [
        f"{domain_id}:{blocker}"
        for row in supporting_artifacts
        if not row["ready"]
        for blocker in row["blockers"]
    ]
    return {
        "domain_id": domain_id,
        "title": str(domain.get("title") or domain_id),
        "status": status,
        "ok": bool(structural_ready and (evidence_ready or evidence_domain)),
        "critical_runtime_domain": critical,
        "evidence_domain": evidence_domain,
        "structural_ready": structural_ready,
        "structural_score": structural_score,
        "structural_grade": _score_grade(structural_score),
        "structural_ready_control_count": structural_ready_count,
        "structural_control_count": len(structural_controls),
        "structural_controls": structural_controls,
        "evidence_ready": evidence_ready,
        "evidence_score": evidence_score,
        "evidence_grade": _score_grade(evidence_score),
        "required_artifact_count": len(required_artifacts),
        "required_ready_artifact_count": required_ready_count,
        "artifacts": artifacts,
        "structural_blockers": structural_blockers,
        "evidence_blockers": evidence_blockers,
        "supporting_warnings": supporting_warnings,
        "owner_command": _as_list(domain.get("owner_command")),
        "recovery_command": _as_list(domain.get("recovery_command")),
        "candidate_scopes": _as_list(domain.get("candidate_scopes")),
        "live_execution_authority": str(domain.get("live_execution_authority") or "none"),
    }


def build_payload(
    project_root: Path = PROJECT_ROOT,
    *,
    config_path: Path = DEFAULT_CONFIG,
    now: datetime | None = None,
) -> dict[str, Any]:
    current = now or datetime.now(timezone.utc)
    effective_config = config_path if config_path.is_absolute() else project_root / config_path
    config = load_json(effective_config)
    domains = [row for row in _as_list(config.get("domains")) if isinstance(row, dict)]
    domain_ids = [str(row.get("domain_id") or "").strip() for row in domains]
    critical_domains = {str(item) for item in _as_list(config.get("critical_runtime_domains"))}
    evidence_domains = {str(item) for item in _as_list(config.get("evidence_domains"))}
    manifest_blockers = ordered_unique(
        [
            "uniform_hardening_config_missing" if not config else "",
            "uniform_hardening_domains_missing" if not domains else "",
            "uniform_hardening_domain_ids_missing" if any(not item for item in domain_ids) else "",
            "uniform_hardening_domain_ids_duplicated" if len(set(domain_ids)) != len(domain_ids) else "",
            "uniform_hardening_domain_classes_overlap" if critical_domains & evidence_domains else "",
            "uniform_hardening_domain_classification_incomplete"
            if set(domain_ids) != critical_domains | evidence_domains
            else "",
            "uniform_hardening_common_controls_mismatch"
            if tuple(config.get("required_common_controls") or ()) != COMMON_CONTROL_IDS
            else "",
        ]
    )
    rows = [
        _domain_row(
            project_root,
            domain,
            now=current,
            critical_domains=critical_domains,
            evidence_domains=evidence_domains,
        )
        for domain in domains
    ]
    structural_blockers = ordered_unique(
        [*manifest_blockers, *[blocker for row in rows for blocker in row["structural_blockers"]]]
    )
    critical_blockers = ordered_unique(
        [
            blocker
            for row in rows
            if row["critical_runtime_domain"] and not row["evidence_ready"]
            for blocker in row["evidence_blockers"]
        ]
    )
    evidence_debt_domains = [
        row["domain_id"] for row in rows if row["evidence_domain"] and not row["evidence_ready"]
    ]
    uniform_floor_ready = not structural_blockers and bool(rows)
    critical_runtime_ready = not critical_blockers and all(
        row["evidence_ready"] for row in rows if row["critical_runtime_domain"]
    )
    all_domain_evidence_ready = bool(rows) and all(row["evidence_ready"] for row in rows)
    structural_score = round(
        sum(_safe_float(row["structural_score"]) for row in rows) / max(len(rows), 1), 2
    )
    critical_rows = [row for row in rows if row["critical_runtime_domain"]]
    critical_score = round(
        100.0 * sum(1 for row in critical_rows if row["evidence_ready"]) / max(len(critical_rows), 1), 2
    )
    evidence_score = round(
        100.0 * sum(1 for row in rows if row["evidence_ready"]) / max(len(rows), 1), 2
    )
    if not uniform_floor_ready or not critical_runtime_ready:
        overall_status = "blocked"
    elif evidence_debt_domains:
        overall_status = "ready_with_evidence_debt"
    else:
        overall_status = "ready"
    recovery_commands = []
    seen_commands: set[tuple[str, ...]] = set()
    for row in rows:
        if row["status"] == "ready":
            continue
        command = tuple(str(item) for item in row["recovery_command"])
        if command and command not in seen_commands:
            seen_commands.add(command)
            recovery_commands.append(list(command))
    canonical = json.dumps(config, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {
        "timestamp_utc": current.isoformat(),
        "schema_version": SCHEMA_VERSION,
        "policy_id": str(config.get("policy_id") or "production_uniform_hardening_v1"),
        "overall_status": overall_status,
        "ok": bool(uniform_floor_ready and critical_runtime_ready),
        "uniform_floor_ready": uniform_floor_ready,
        "critical_runtime_ready": critical_runtime_ready,
        "all_domain_evidence_ready": all_domain_evidence_ready,
        "uniform_structural_score": structural_score,
        "uniform_structural_grade": _score_grade(structural_score),
        "critical_runtime_score": critical_score,
        "critical_runtime_grade": _score_grade(critical_score),
        "all_domain_evidence_score": evidence_score,
        "all_domain_evidence_grade": _score_grade(evidence_score),
        "domain_count": len(rows),
        "critical_runtime_domain_count": len(critical_rows),
        "evidence_domain_count": sum(1 for row in rows if row["evidence_domain"]),
        "structurally_ready_domain_count": sum(1 for row in rows if row["structural_ready"]),
        "evidence_ready_domain_count": sum(1 for row in rows if row["evidence_ready"]),
        "domain_statuses": {row["domain_id"]: row["status"] for row in rows},
        "domains": rows,
        "structural_blockers": structural_blockers,
        "critical_runtime_blockers": critical_blockers,
        "evidence_debt_domains": evidence_debt_domains,
        "supporting_warnings": ordered_unique(
            [warning for row in rows for warning in row["supporting_warnings"]]
        ),
        "recommended_recovery_commands": recovery_commands,
        "manifest": {
            "path": str(effective_config),
            "sha256": hashlib.sha256(canonical).hexdigest(),
            "required_common_controls": list(COMMON_CONTROL_IDS),
            "minimum_structural_grade": str(config.get("minimum_structural_grade") or "A+"),
        },
        "live_execution_authority": False,
        "live_orders_must_remain_disabled": True,
        "control_contract": {
            **_as_dict(config.get("control_contract")),
            "every_domain_uses_the_same_structural_floor": True,
            "structural_grade_never_overrides_runtime_or_economic_evidence": True,
            "supporting_artifact_degradation_is_visible_but_not_misclassified_as_core_failure": True,
            "only_explicit_operator_gates_may_authorize_live_execution": True,
        },
    }


def evaluation_exit_code(payload: dict[str, Any], *, structural_only: bool = False) -> int:
    ready = bool(payload.get("uniform_floor_ready", False)) if structural_only else bool(payload.get("ok", False))
    return 0 if ready else 2


def main() -> int:
    parser = argparse.ArgumentParser(description="Enforce one structural production-hardening floor across every critical system domain.")
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--out-file", default=str(DEFAULT_OUT))
    parser.add_argument(
        "--structural-only",
        action="store_true",
        help="Use the uniform structural floor as the process exit condition while preserving runtime evidence failures in the payload.",
    )
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    project_root = Path(args.project_root).expanduser().resolve()
    payload = build_payload(project_root, config_path=Path(args.config).expanduser())
    payload["evaluation_mode"] = "structural_only" if args.structural_only else "full_runtime"
    write_payload(Path(args.out_file).expanduser(), payload)
    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        print(
            "uniform_hardening_contract "
            f"status={payload['overall_status']} floor={payload['uniform_structural_grade']} "
            f"critical={payload['critical_runtime_grade']} evidence={payload['all_domain_evidence_grade']}"
        )
    return evaluation_exit_code(payload, structural_only=args.structural_only)


if __name__ == "__main__":
    raise SystemExit(main())
