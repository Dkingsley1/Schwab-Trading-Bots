#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

if __package__ in {None, ""}:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from scripts.ops.long_runtime_common import (
        PROJECT_ROOT,
        evidence_freshness,
        iso_now,
        load_json,
        ordered_unique,
        parse_iso_utc,
        write_payload,
    )
    from scripts.ops.ops_scheduled_job_catalog import (
        ACTIVE_INSTALL_POLICY,
        DEFAULT_JOB_SPECS,
        REMOVED_INSTALL_POLICY,
        JobSpec,
    )
else:
    from .long_runtime_common import (
        PROJECT_ROOT,
        evidence_freshness,
        iso_now,
        load_json,
        ordered_unique,
        parse_iso_utc,
        write_payload,
    )
    from .ops_scheduled_job_catalog import (
        ACTIVE_INSTALL_POLICY,
        DEFAULT_JOB_SPECS,
        REMOVED_INSTALL_POLICY,
        JobSpec,
    )


DEFAULT_OUT_PATH = (
    PROJECT_ROOT / "governance" / "health" / "ops_scheduled_job_lifecycle_latest.json"
)
DEFAULT_INSTALLER_PATH = (
    PROJECT_ROOT / "scripts" / "ops" / "install_ops_automation_launchd.sh"
)
SCHEMA_VERSION = 1
REQUIRED_LIFECYCLE_FIELDS = (
    "run_id",
    "scheduled",
    "eligible",
    "deferred",
    "started",
    "completed",
    "failed",
    "next_eligible_utc",
)
HARD_ISSUES = {
    "runner_missing",
    "installation_missing",
    "unexpected_legacy_installed",
    "evidence_missing",
    "evidence_timestamp_missing",
    "evidence_timestamp_invalid",
    "evidence_future_timestamp",
    "evidence_stale",
    "job_failed",
}
LIFECYCLE_RUNNER_NAME = "run_scheduled_lifecycle_job.py"
INSTALLER_COMMAND = "./scripts/ops/install_ops_automation_launchd.sh"
KICKSTART_COMMAND_RE = re.compile(
    r"^launchctl kickstart -k gui/\$\(id -u\)/(com\.dankingsley\.ops\.[A-Za-z0-9_.-]+)$"
)


def _resolve(project_root: Path, raw: str) -> Path:
    path = Path(str(raw or ""))
    return path if path.is_absolute() else project_root / path


def _launch_agents_dir(raw: str | None = None) -> Path:
    if raw:
        return Path(raw).expanduser()
    return Path.home() / "Library" / "LaunchAgents"


def _installer_labels(path: Path) -> dict[str, Any]:
    active: list[str] = []
    removed: list[str] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except Exception:
        return {
            "path": str(path),
            "present": False,
            "active_labels": [],
            "removed_labels": [],
        }
    pattern = re.compile(r'^(install_job|remove_job)\s+"([^"]+)"')
    for line in lines:
        match = pattern.search(line.strip())
        if not match:
            continue
        if match.group(1) == "install_job":
            active.append(match.group(2))
        else:
            removed.append(match.group(2))
    return {
        "path": str(path),
        "present": True,
        "active_labels": ordered_unique(active),
        "removed_labels": ordered_unique(removed),
    }


def _catalog_alignment(project_root: Path, jobs: Iterable[JobSpec]) -> dict[str, Any]:
    installer = _installer_labels(
        DEFAULT_INSTALLER_PATH
        if project_root == PROJECT_ROOT
        else project_root / "scripts" / "ops" / "install_ops_automation_launchd.sh"
    )
    active_labels = set(installer["active_labels"])
    removed_labels = set(installer["removed_labels"])
    catalog_active = {
        job.label for job in jobs if job.install_policy == ACTIVE_INSTALL_POLICY
    }
    catalog_removed = {
        job.label for job in jobs if job.install_policy == REMOVED_INSTALL_POLICY
    }
    missing_active = sorted(active_labels - catalog_active)
    missing_removed = sorted(removed_labels - catalog_removed)
    active_not_installer = sorted(catalog_active - active_labels)
    removed_not_installer = sorted(catalog_removed - removed_labels)
    ok = bool(
        installer["present"]
        and not missing_active
        and not missing_removed
        and not active_not_installer
        and not removed_not_installer
    )
    return {
        "ok": ok,
        "installer": installer,
        "catalog_active_count": len(catalog_active),
        "catalog_removed_count": len(catalog_removed),
        "missing_active_labels": missing_active,
        "missing_removed_labels": missing_removed,
        "active_catalog_labels_not_in_installer": active_not_installer,
        "removed_catalog_labels_not_in_installer": removed_not_installer,
    }


def _lifecycle_state(payload: dict[str, Any], *, now: datetime) -> dict[str, Any]:
    lifecycle = payload.get("job_lifecycle")
    has_receipt = isinstance(lifecycle, dict) and bool(lifecycle)
    raw = lifecycle if has_receipt else payload
    raw = raw if isinstance(raw, dict) else {}
    missing_fields = [
        field
        for field in REQUIRED_LIFECYCLE_FIELDS
        if not has_receipt or field not in raw
    ]
    next_eligible = parse_iso_utc(raw.get("next_eligible_utc"))
    if next_eligible is None:
        next_state = "unknown"
    elif next_eligible <= now:
        next_state = "eligible_now"
    else:
        next_state = "not_yet_eligible"
    failed = bool(raw.get("failed", False))
    deferred = bool(raw.get("deferred", False))
    completed = (
        bool(raw.get("completed", False))
        or str(raw.get("run_state") or "") == "completed"
    )
    started = bool(raw.get("started", False)) or bool(raw.get("started_utc"))
    if failed:
        transition_state = "failed"
    elif deferred:
        transition_state = "deferred"
    elif completed:
        transition_state = "completed"
    elif started:
        transition_state = "started"
    elif bool(raw.get("scheduled", False)):
        transition_state = "scheduled"
    else:
        transition_state = "unknown"
    return {
        "source": "job_lifecycle" if has_receipt else "derived_top_level",
        "receipt_present": has_receipt,
        "missing_fields": missing_fields,
        "run_id": str(raw.get("run_id") or ""),
        "scheduled": bool(raw.get("scheduled", False)),
        "eligible": bool(raw.get("eligible", False)),
        "deferred": deferred,
        "deferred_reason": str(raw.get("deferred_reason") or ""),
        "started": started,
        "started_utc": str(raw.get("started_utc") or payload.get("started_utc") or ""),
        "completed": completed,
        "completed_utc": str(
            raw.get("completed_utc") or payload.get("completed_utc") or ""
        ),
        "failed": failed,
        "failure_reason": str(raw.get("failure_reason") or ""),
        "terminal_status": str(
            raw.get("terminal_status")
            or payload.get("overall_status")
            or payload.get("status")
            or ""
        ),
        "next_eligible_utc": str(raw.get("next_eligible_utc") or ""),
        "next_eligibility_state": next_state,
    }


def _evidence_issue(freshness: dict[str, Any]) -> str:
    status = str(freshness.get("status") or "")
    if status == "timestamp_invalid":
        return "evidence_timestamp_invalid"
    if status == "future_timestamp":
        return "evidence_future_timestamp"
    if status == "stale":
        return "evidence_stale"
    return f"evidence_{status}" if status else "evidence_missing"


def _lifecycle_receipt_freshness(
    lifecycle: dict[str, Any], spec: JobSpec, *, now: datetime
) -> dict[str, Any]:
    completed = parse_iso_utc(lifecycle.get("completed_utc"))
    max_age_seconds = min(max(float(spec.cadence_seconds) * 2.0, 300.0), 1800.0)
    if completed is None:
        return {
            "status": "timestamp_missing",
            "fresh": False,
            "completed_utc": str(lifecycle.get("completed_utc") or ""),
            "age_seconds": None,
            "max_age_seconds": max_age_seconds,
        }
    age_seconds = (now - completed).total_seconds()
    if age_seconds < -60.0:
        status = "future_timestamp"
        fresh = False
    else:
        age_seconds = max(age_seconds, 0.0)
        fresh = age_seconds <= max_age_seconds
        status = "fresh" if fresh else "stale"
    return {
        "status": status,
        "fresh": fresh,
        "completed_utc": completed.astimezone(timezone.utc).isoformat(),
        "age_seconds": round(age_seconds, 3),
        "max_age_seconds": max_age_seconds,
    }


def _stale_evidence_managed_by_fresh_deferral(
    freshness: dict[str, Any],
    lifecycle: dict[str, Any],
    lifecycle_freshness: dict[str, Any],
) -> bool:
    if freshness.get("status") != "stale":
        return False
    if not bool(lifecycle.get("receipt_present", False)):
        return False
    if bool(lifecycle.get("failed", False)):
        return False
    if not bool(lifecycle.get("deferred", False)):
        return False
    if not str(lifecycle.get("deferred_reason") or "").strip():
        return False
    return bool(lifecycle_freshness.get("fresh", False))


def _plist_path(launch_agents_dir: Path, label: str) -> Path:
    return launch_agents_dir / f"{label}.plist"


def _plist_uses_lifecycle_runner(path: Path) -> bool:
    try:
        return LIFECYCLE_RUNNER_NAME in path.read_text(encoding="utf-8")
    except Exception:
        return False


def _job_row(
    project_root: Path, launch_agents_dir: Path, spec: JobSpec, *, now: datetime
) -> dict[str, Any]:
    runner_path = _resolve(project_root, spec.runner)
    artifact_path = _resolve(project_root, spec.artifact)
    plist_path = _plist_path(launch_agents_dir, spec.label)
    runner_present = runner_path.exists()
    plist_present = plist_path.exists()
    lifecycle_wrapper_required = (
        spec.install_policy == ACTIVE_INSTALL_POLICY
        and (spec.resource_class != "single_writer" or spec.deadline_seconds > 0)
    )
    lifecycle_wrapper_present = (
        _plist_uses_lifecycle_runner(plist_path) if plist_present else False
    )
    artifact_present = artifact_path.exists()
    payload = load_json(artifact_path) if artifact_present else {}
    freshness = evidence_freshness(
        payload,
        max_age_minutes=max(spec.freshness_slo_seconds / 60.0, 1.0),
        now=now,
    )
    lifecycle = (
        _lifecycle_state(payload, now=now)
        if payload
        else {
            "source": "missing",
            "receipt_present": False,
            "missing_fields": list(REQUIRED_LIFECYCLE_FIELDS),
            "run_id": "",
            "scheduled": False,
            "eligible": False,
            "deferred": False,
            "deferred_reason": "",
            "started": False,
            "started_utc": "",
            "completed": False,
            "completed_utc": "",
            "failed": False,
            "failure_reason": "",
            "terminal_status": "",
            "next_eligible_utc": "",
            "next_eligibility_state": "unknown",
        }
    )
    lifecycle_freshness = _lifecycle_receipt_freshness(lifecycle, spec, now=now)
    stale_evidence_managed_by_deferral = _stale_evidence_managed_by_fresh_deferral(
        freshness,
        lifecycle,
        lifecycle_freshness,
    )
    single_writer_fresh_without_receipt = bool(
        spec.resource_class == "single_writer"
        and not lifecycle_wrapper_required
        and artifact_present
        and freshness.get("fresh", False)
        and not lifecycle.get("receipt_present", False)
    )
    issues: list[str] = []
    if spec.install_policy == ACTIVE_INSTALL_POLICY and not runner_present:
        issues.append("runner_missing")
    if spec.install_policy == ACTIVE_INSTALL_POLICY and not plist_present:
        issues.append("installation_missing")
    if spec.install_policy == REMOVED_INSTALL_POLICY and plist_present:
        issues.append("unexpected_legacy_installed")
    if spec.install_policy == ACTIVE_INSTALL_POLICY:
        if not artifact_present:
            issues.append("evidence_missing")
        elif not bool(freshness.get("fresh", False)):
            if stale_evidence_managed_by_deferral:
                issues.append("evidence_stale_under_fresh_deferral")
            else:
                issues.append(_evidence_issue(freshness))
        if artifact_present and not lifecycle["receipt_present"]:
            if single_writer_fresh_without_receipt:
                issues.append("single_writer_fresh_artifact_liveness")
            else:
                issues.append("lifecycle_receipt_missing")
        elif artifact_present and lifecycle["missing_fields"]:
            issues.append("lifecycle_fields_missing")
        if (
            plist_present
            and lifecycle_wrapper_required
            and not lifecycle_wrapper_present
        ):
            issues.append("lifecycle_wrapper_missing")
        if lifecycle["failed"]:
            issues.append("job_failed")
    else:
        if artifact_present and lifecycle["receipt_present"]:
            issues.append("legacy_lifecycle_receipt_retained")

    hard_issues = [issue for issue in issues if issue in HARD_ISSUES]
    lifecycle_debt = any(
        issue
        in {
            "lifecycle_receipt_missing",
            "lifecycle_fields_missing",
            "lifecycle_wrapper_missing",
        }
        for issue in issues
    )
    if spec.install_policy == REMOVED_INSTALL_POLICY and not hard_issues:
        operational_status = "retired"
    elif hard_issues:
        operational_status = hard_issues[0]
    elif lifecycle_debt:
        operational_status = "lifecycle_debt"
    elif lifecycle["deferred"]:
        operational_status = "deferred"
    else:
        operational_status = "ready"
    return {
        "job_id": spec.job_id,
        "domain": spec.domain,
        "label": spec.label,
        "runner": spec.runner,
        "runner_present": runner_present,
        "artifact": spec.artifact,
        "artifact_present": artifact_present,
        "artifact_freshness": freshness,
        "plist": str(plist_path),
        "plist_present": plist_present,
        "lifecycle_wrapper_present": lifecycle_wrapper_present,
        "lifecycle_wrapper_required": lifecycle_wrapper_required,
        "install_policy": spec.install_policy,
        "installation_state": (
            "expected_installed"
            if spec.install_policy == ACTIVE_INSTALL_POLICY
            else "expected_removed"
        ),
        "cadence_seconds": spec.cadence_seconds,
        "freshness_slo_seconds": spec.freshness_slo_seconds,
        "resource_class": spec.resource_class,
        "deadline_seconds": spec.deadline_seconds,
        "owner": spec.owner,
        "authority_boundary": spec.authority_boundary,
        "live_execution_authority": False,
        "dependencies": list(spec.dependencies),
        "lifecycle": lifecycle,
        "lifecycle_receipt_freshness": lifecycle_freshness,
        "stale_evidence_managed_by_fresh_deferral": stale_evidence_managed_by_deferral,
        "single_writer_fresh_without_receipt": single_writer_fresh_without_receipt,
        "issues": ordered_unique(issues),
        "hard_issues": hard_issues,
        "operational_status": operational_status,
    }


def _count_by(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        value = str(row.get(key) or "unknown")
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))


def _known_labels(rows: list[dict[str, Any]]) -> set[str]:
    return {str(row.get("label") or "") for row in rows if row.get("label")}


def _action_row(action: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    job_id = str(action.get("job_id") or "")
    if not job_id:
        return {}
    for row in rows:
        if row.get("job_id") == job_id:
            return row
    return {}


def _action_preflight(
    action: dict[str, Any],
    *,
    rows: list[dict[str, Any]],
    known_labels: set[str],
) -> dict[str, Any]:
    command = str(action.get("command") or "")
    row = _action_row(action, rows)
    auto_execute_requested = bool(action.get("auto_execute", False))
    blocked_by: list[str] = []
    command_family = "manual_review"
    command_allowlisted = False
    mutation_scope = "none"
    operator_confirmation_required = bool(command)

    if not command:
        blocked_by.append("manual_source_alignment_required")
    elif command == INSTALLER_COMMAND:
        command_family = "ops_launchd_installer"
        command_allowlisted = True
        mutation_scope = "local_launchagent_plists"
    else:
        kickstart = KICKSTART_COMMAND_RE.match(command)
        if kickstart:
            label = kickstart.group(1)
            command_family = "launchctl_kickstart"
            command_allowlisted = label in known_labels
            mutation_scope = "local_launchd_job_kickstart"
            if not command_allowlisted:
                blocked_by.append("label_not_in_catalog")
            if not row:
                blocked_by.append("job_not_in_catalog")
            elif row.get("install_policy") != ACTIVE_INSTALL_POLICY:
                blocked_by.append("job_not_active")
            elif not bool(row.get("plist_present", False)):
                blocked_by.append("plist_missing")
        else:
            command_family = "unknown"
            blocked_by.append("command_not_allowlisted")

    if auto_execute_requested:
        blocked_by.append("auto_execute_not_permitted")

    if (
        action.get("category") == "lifecycle_receipt"
        and row
        and bool(row.get("lifecycle_wrapper_required", False))
        and not bool(row.get("lifecycle_wrapper_present", False))
    ):
        blocked_by.append("lifecycle_wrapper_not_installed")

    blocked_by = ordered_unique(blocked_by)
    if blocked_by:
        status = "blocked_by_prerequisite"
    elif operator_confirmation_required:
        status = "ready_for_operator_confirmation"
    else:
        status = "manual_review_required"

    return {
        "status": status,
        "ok_to_execute_after_operator_confirmation": status
        == "ready_for_operator_confirmation",
        "command_family": command_family,
        "command_allowlisted": command_allowlisted,
        "operator_confirmation_required": operator_confirmation_required,
        "auto_execute_requested": auto_execute_requested,
        "auto_execute_permitted": False,
        "execution_authority": False,
        "mutation_scope": mutation_scope,
        "blocked_by": blocked_by,
    }


def _attach_action_preflights(
    actions: list[dict[str, Any]], rows: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    labels = _known_labels(rows)
    hardened: list[dict[str, Any]] = []
    for action in actions:
        item = dict(action)
        item["preflight"] = _action_preflight(item, rows=rows, known_labels=labels)
        hardened.append(item)
    return hardened


def _action_queue_preflight_summary(actions: list[dict[str, Any]]) -> dict[str, Any]:
    statuses = [
        str((action.get("preflight") or {}).get("status") or "unknown")
        for action in actions
    ]
    blocked = [
        action
        for action in actions
        if str((action.get("preflight") or {}).get("status") or "")
        == "blocked_by_prerequisite"
    ]
    rejected = [
        action
        for action in actions
        if str((action.get("preflight") or {}).get("command_family") or "") == "unknown"
    ]
    auto_execute = [
        action
        for action in actions
        if bool(action.get("auto_execute", False))
        or bool((action.get("preflight") or {}).get("auto_execute_requested", False))
    ]
    ready = [
        action
        for action in actions
        if bool(
            (action.get("preflight") or {}).get(
                "ok_to_execute_after_operator_confirmation", False
            )
        )
    ]
    command_families = [
        str((action.get("preflight") or {}).get("command_family") or "unknown")
        for action in actions
    ]
    return {
        "schema_version": 1,
        "safe_to_review": not rejected and not auto_execute,
        "execution_authority": False,
        "auto_execute_action_count": len(auto_execute),
        "ready_for_operator_confirmation_count": len(ready),
        "blocked_action_count": len(blocked),
        "rejected_command_count": len(rejected),
        "status_counts": _value_counts(statuses),
        "command_family_counts": _value_counts(command_families),
        "required_operator_sequence": [
            "resolve P0 installation/catalog blockers first",
            "reinstall lifecycle-wrapped LaunchAgents before closing wrapper-dependent receipt debt",
            "kick stale or missing evidence producers only after checking resource and market-hours context",
            "rerun ops-scheduled-jobs after each operator-approved action batch",
        ],
    }


def _value_counts(values: list[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        key = value or "unknown"
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def _remediation_action_queue(
    rows: list[dict[str, Any]], alignment: dict[str, Any]
) -> list[dict[str, Any]]:
    actions: list[dict[str, Any]] = []
    if not alignment.get("ok", False):
        actions.append(
            {
                "priority": "P0",
                "action_id": "align_installer_catalog",
                "category": "catalog_alignment",
                "job_id": "",
                "issue": "installer_catalog_mismatch",
                "command": "",
                "auto_execute": False,
                "rationale": "The checked-in catalog and install_ops_automation_launchd.sh disagree; fix source alignment before trusting job-level remediation.",
            }
        )

    wrapper_missing = [
        row
        for row in rows
        if row["install_policy"] == ACTIVE_INSTALL_POLICY
        and "lifecycle_wrapper_missing" in row["issues"]
    ]
    if wrapper_missing:
        actions.append(
            {
                "priority": "P1",
                "action_id": "reinstall_lifecycle_wrapped_launchd",
                "category": "lifecycle_adoption",
                "job_id": "",
                "issue": "lifecycle_wrapper_missing",
                "affected_job_count": len(wrapper_missing),
                "command": INSTALLER_COMMAND,
                "auto_execute": False,
                "rationale": "Regenerate local LaunchAgent plists so scheduled jobs run through run_scheduled_lifecycle_job.py and stamp receipts after each producer run.",
            }
        )

    for row in rows:
        if row["install_policy"] != ACTIVE_INSTALL_POLICY:
            if "unexpected_legacy_installed" in row["issues"]:
                actions.append(
                    {
                        "priority": "P0",
                        "action_id": f"remove_legacy_{row['job_id']}",
                        "category": "legacy_retirement",
                        "job_id": row["job_id"],
                        "issue": "unexpected_legacy_installed",
                        "command": INSTALLER_COMMAND,
                        "auto_execute": False,
                        "rationale": "The installer retires this legacy direct scheduler; remove the local plist before claiming clean scheduling authority.",
                    }
                )
            continue
        if "installation_missing" in row["issues"]:
            actions.append(
                {
                    "priority": "P0",
                    "action_id": f"restore_plist_{row['job_id']}",
                    "category": "installation",
                    "job_id": row["job_id"],
                    "issue": "installation_missing",
                    "command": INSTALLER_COMMAND,
                    "auto_execute": False,
                    "rationale": "The job is cataloged as active but its local LaunchAgent plist is absent.",
                }
            )
        if "evidence_missing" in row["issues"]:
            actions.append(
                {
                    "priority": "P1",
                    "action_id": f"refresh_missing_artifact_{row['job_id']}",
                    "category": "evidence_refresh",
                    "job_id": row["job_id"],
                    "issue": "evidence_missing",
                    "command": f"launchctl kickstart -k gui/$(id -u)/{row['label']}",
                    "auto_execute": False,
                    "rationale": "Kick the installed scheduled job or run its listed runner once, then verify the expected latest artifact exists.",
                }
            )
        if "evidence_stale" in row["issues"]:
            actions.append(
                {
                    "priority": "P1",
                    "action_id": f"refresh_stale_artifact_{row['job_id']}",
                    "category": "evidence_refresh",
                    "job_id": row["job_id"],
                    "issue": "evidence_stale",
                    "command": f"launchctl kickstart -k gui/$(id -u)/{row['label']}",
                    "auto_execute": False,
                    "rationale": "The latest producer timestamp is past its freshness SLO; force a scheduled run only after checking resource and market-hours guard context.",
                }
            )
        if any(
            issue in row["issues"]
            for issue in ("lifecycle_receipt_missing", "lifecycle_fields_missing")
        ):
            wrapper_required = bool(row.get("lifecycle_wrapper_required", True))
            actions.append(
                {
                    "priority": "P2",
                    "action_id": f"close_lifecycle_receipt_{row['job_id']}",
                    "category": "lifecycle_receipt",
                    "job_id": row["job_id"],
                    "issue": "lifecycle_receipt_missing",
                    "command": f"launchctl kickstart -k gui/$(id -u)/{row['label']}",
                    "auto_execute": False,
                    "rationale": (
                        "After lifecycle-wrapped plists are installed, one completed scheduled run should stamp this artifact with job_lifecycle."
                        if wrapper_required
                        else "The persistent single-writer service stamps lifecycle internally on the next writer cycle or explicit service kick."
                    ),
                }
            )
    priority_order = {"P0": 0, "P1": 1, "P2": 2, "P3": 3}
    sorted_actions = sorted(
        actions,
        key=lambda row: (
            priority_order.get(str(row.get("priority") or "P3"), 99),
            str(row.get("category") or ""),
            str(row.get("job_id") or ""),
            str(row.get("action_id") or ""),
        ),
    )
    return _attach_action_preflights(sorted_actions, rows)


def build_payload(
    project_root: Path = PROJECT_ROOT,
    *,
    launch_agents_dir: Path | None = None,
    jobs: Iterable[JobSpec] = DEFAULT_JOB_SPECS,
    now: datetime | None = None,
) -> dict[str, Any]:
    current = now or datetime.now(timezone.utc)
    launch_dir = launch_agents_dir or _launch_agents_dir()
    job_list = list(jobs)
    rows = [_job_row(project_root, launch_dir, spec, now=current) for spec in job_list]
    alignment = _catalog_alignment(project_root, job_list)
    active_rows = [
        row for row in rows if row["install_policy"] == ACTIVE_INSTALL_POLICY
    ]
    removed_rows = [
        row for row in rows if row["install_policy"] == REMOVED_INSTALL_POLICY
    ]
    hard_issue_rows = [row for row in rows if row["hard_issues"]]
    lifecycle_debt_rows = [
        row for row in active_rows if row["operational_status"] == "lifecycle_debt"
    ]
    lifecycle_wrapper_missing = [
        row for row in active_rows if "lifecycle_wrapper_missing" in row["issues"]
    ]
    deferred_rows = [
        row for row in active_rows if row["operational_status"] == "deferred"
    ]
    missing_installation = [
        row for row in active_rows if "installation_missing" in row["issues"]
    ]
    missing_evidence = [
        row for row in active_rows if "evidence_missing" in row["issues"]
    ]
    stale_evidence = [row for row in active_rows if "evidence_stale" in row["issues"]]
    unexpected_legacy = [
        row for row in removed_rows if "unexpected_legacy_installed" in row["issues"]
    ]
    if hard_issue_rows or not alignment["ok"]:
        overall_status = "blocked"
    elif lifecycle_debt_rows:
        overall_status = "ready_with_lifecycle_debt"
    elif deferred_rows:
        overall_status = "ready_with_deferrals"
    else:
        overall_status = "ready"
    recommended_actions = []
    if missing_installation:
        recommended_actions.append(
            "reinstall or intentionally retire missing LaunchAgent plists before claiming unattended coverage"
        )
    if missing_evidence:
        recommended_actions.append(
            "run missing scheduled producers once under their guarded wrapper and verify they publish current artifacts"
        )
    if stale_evidence:
        recommended_actions.append(
            "inspect stale scheduled jobs for quiet-window, lock, resource, or dependency deferrals"
        )
    managed_stale_deferrals = [
        row
        for row in active_rows
        if bool(row.get("stale_evidence_managed_by_fresh_deferral", False))
    ]
    if managed_stale_deferrals:
        recommended_actions.append(
            "keep quiet-window and skip deferrals visible while treating fresh lifecycle receipts as scheduler liveness evidence"
        )
    if lifecycle_debt_rows:
        recommended_actions.append(
            "add job_lifecycle receipts to scheduled producers until C09 transition coverage is complete"
        )
    if unexpected_legacy:
        recommended_actions.append(
            "remove legacy direct scheduler plists that the installer now retires"
        )
    if lifecycle_wrapper_missing:
        recommended_actions.append(
            "reinstall ops automations so LaunchAgents use the shared scheduled lifecycle runner"
        )
    if not alignment["ok"]:
        recommended_actions.append(
            "align DEFAULT_JOB_SPECS with install_ops_automation_launchd.sh before extending the scheduler catalog"
        )
    if not recommended_actions:
        recommended_actions.append(
            "keep scheduled jobs under catalog review and preserve explicit deferral receipts"
        )
    action_queue = _remediation_action_queue(rows, alignment)
    action_queue_preflight = _action_queue_preflight_summary(action_queue)
    return {
        "schema_version": SCHEMA_VERSION,
        "timestamp_utc": iso_now(),
        "source": "ops_scheduled_job_lifecycle",
        "overall_status": overall_status,
        "ok": overall_status
        in {"ready", "ready_with_deferrals", "ready_with_lifecycle_debt"},
        "project_root": str(project_root),
        "launch_agents_dir": str(launch_dir),
        "job_count": len(rows),
        "active_job_count": len(active_rows),
        "removed_legacy_job_count": len(removed_rows),
        "ready_job_count": sum(
            1 for row in rows if row["operational_status"] == "ready"
        ),
        "deferred_job_count": len(deferred_rows),
        "lifecycle_debt_count": len(lifecycle_debt_rows),
        "hard_issue_count": len(hard_issue_rows),
        "installation_missing_count": len(missing_installation),
        "evidence_missing_count": len(missing_evidence),
        "evidence_stale_count": len(stale_evidence),
        "managed_stale_deferral_count": len(managed_stale_deferrals),
        "unexpected_legacy_installed_count": len(unexpected_legacy),
        "lifecycle_wrapper_missing_count": len(lifecycle_wrapper_missing),
        "action_queue_count": len(action_queue),
        "action_queue_preflight": action_queue_preflight,
        "status_counts": _count_by(rows, "operational_status"),
        "domain_counts": _count_by(rows, "domain"),
        "installer_alignment": alignment,
        "jobs": rows,
        "action_queue": action_queue,
        "recommended_actions": ordered_unique(recommended_actions),
        "control_contract": {
            "read_only_by_default": True,
            "apply_only_writes_catalog_artifact": True,
            "installer_is_not_installation_evidence": True,
            "plist_presence_is_local_installation_evidence": True,
            "missing_transition_receipts_are_lifecycle_debt": True,
            "lifecycle_debt_is_advisory_when_hard_issues_are_clear": True,
            "deferred_jobs_are_visible_without_failure_credit": True,
            "action_queue_auto_execute": False,
            "action_queue_preflight_required": True,
            "generic_runner_preserves_producer_timestamp": True,
            "persistent_single_writer_uses_internal_receipts": True,
            "bounded_single_writer_requires_lifecycle_wrapper": True,
            "fresh_single_writer_artifact_satisfies_liveness": True,
            "queue_execution_authority": False,
            "operator_confirmation_required_for_queue_actions": True,
            "launchctl_mutation_authority": False,
            "live_execution_authority": False,
            "training_launch_authority": False,
            "storage_delete_authority": False,
            "credential_mutation_authority": False,
        },
    }


def _queue_only_payload(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": payload.get("schema_version", SCHEMA_VERSION),
        "timestamp_utc": payload.get("timestamp_utc", ""),
        "source": payload.get("source", "ops_scheduled_job_lifecycle"),
        "overall_status": payload.get("overall_status", ""),
        "ok": payload.get("ok", False),
        "job_count": payload.get("job_count", 0),
        "active_job_count": payload.get("active_job_count", 0),
        "hard_issue_count": payload.get("hard_issue_count", 0),
        "lifecycle_debt_count": payload.get("lifecycle_debt_count", 0),
        "lifecycle_wrapper_missing_count": payload.get(
            "lifecycle_wrapper_missing_count", 0
        ),
        "evidence_missing_count": payload.get("evidence_missing_count", 0),
        "evidence_stale_count": payload.get("evidence_stale_count", 0),
        "action_queue_count": payload.get("action_queue_count", 0),
        "action_queue_preflight": payload.get("action_queue_preflight", {}),
        "action_queue": payload.get("action_queue", []),
        "recommended_actions": payload.get("recommended_actions", []),
    }


def _preflight_only_payload(payload: dict[str, Any]) -> dict[str, Any]:
    actions = payload.get("action_queue", [])
    if not isinstance(actions, list):
        actions = []
    return {
        "schema_version": payload.get("schema_version", SCHEMA_VERSION),
        "timestamp_utc": payload.get("timestamp_utc", ""),
        "source": payload.get("source", "ops_scheduled_job_lifecycle"),
        "overall_status": payload.get("overall_status", ""),
        "ok": payload.get("ok", False),
        "job_count": payload.get("job_count", 0),
        "active_job_count": payload.get("active_job_count", 0),
        "hard_issue_count": payload.get("hard_issue_count", 0),
        "lifecycle_debt_count": payload.get("lifecycle_debt_count", 0),
        "action_queue_count": payload.get("action_queue_count", len(actions)),
        "action_queue_preflight": payload.get("action_queue_preflight", {}),
        "actions": [
            {
                "priority": action.get("priority", ""),
                "action_id": action.get("action_id", ""),
                "category": action.get("category", ""),
                "job_id": action.get("job_id", ""),
                "issue": action.get("issue", ""),
                "command": action.get("command", ""),
                "preflight": action.get("preflight", {}),
            }
            for action in actions
            if isinstance(action, dict)
        ],
        "recommended_actions": payload.get("recommended_actions", []),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Inventory launchd-backed ops jobs and their scheduled lifecycle evidence."
    )
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--launch-agents-dir", type=Path, default=None)
    parser.add_argument("--out-file", type=Path, default=DEFAULT_OUT_PATH)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--queue-only", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    project_root = args.project_root.expanduser().resolve()
    launch_agents_dir = (
        args.launch_agents_dir.expanduser() if args.launch_agents_dir else None
    )
    payload = build_payload(project_root, launch_agents_dir=launch_agents_dir)
    if args.apply:
        out_file = args.out_file.expanduser()
        if not out_file.is_absolute():
            out_file = project_root / out_file
        write_payload(out_file, payload)

    if args.preflight_only:
        output_payload = _preflight_only_payload(payload)
    elif args.queue_only:
        output_payload = _queue_only_payload(payload)
    else:
        output_payload = payload
    if args.json:
        print(json.dumps(output_payload, ensure_ascii=True, indent=2))
    else:
        print(
            "ops_scheduled_job_lifecycle "
            f"status={payload['overall_status']} "
            f"jobs={payload['job_count']} "
            f"hard_issues={payload['hard_issue_count']} "
            f"lifecycle_debt={payload['lifecycle_debt_count']}"
        )
    return 0 if payload["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
