#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from scripts.ops.long_runtime_common import PROJECT_ROOT, iso_now, load_json, ordered_unique, payload_age_minutes, write_payload
    from scripts.ops.system_drift_registry import surface_specs
else:
    from .long_runtime_common import PROJECT_ROOT, iso_now, load_json, ordered_unique, payload_age_minutes, write_payload
    from .system_drift_registry import surface_specs


DEFAULT_OUT_PATH = PROJECT_ROOT / "governance" / "health" / "system_drift_guard_latest.json"
BLOCKED_STATUSES = {"blocked", "critical", "missing"}
DEGRADED_STATUSES = {"degraded", "warn", "warning", "needs_work", "inactive", "thin"}
READY_STATUSES = {"ready", "ok", "stable", "watch", "advisory", "guarded", "applied", "applied_with_followups", "cleared"}
RECOVERY_STATES = {"active", "already_running", "busy", "drain_active", "recovering", "recovering_under_guard", "stabilized_recovery"}
PROTECTED_ARCHITECTURE_BLOCKED_NODES = {
    "adaptive_regression_guard",
    "all_sleeves_launcher",
    "paper_ramp",
    "runtime_throttle",
    "storage_control",
    "system_drift_guard",
    "system_self_model",
}
SOFT_ARCHITECTURE_SCOREBOARD_BLOCKERS = {
    "autonomous_drill_program",
    "notification_escalation_ladder",
}


def _safe_int(raw: Any, default: int = 0) -> int:
    try:
        return int(float(raw))
    except Exception:
        return int(default)


def _status_from_bool(ok: Any) -> str:
    if ok is True:
        return "ready"
    if ok is False:
        return "blocked"
    return "missing"


def _normalize_status(raw_status: Any, ok: Any) -> str:
    text = str(raw_status or "").strip().lower()
    if ok is True:
        return "blocked" if text in BLOCKED_STATUSES else "ready"
    if ok is False and not text:
        return "blocked"
    if not text:
        return "missing"
    if text in READY_STATUSES:
        return "ready"
    if text in DEGRADED_STATUSES:
        return "degraded"
    if text in BLOCKED_STATUSES:
        return "blocked"
    return text


def _safe_bool(raw: Any) -> bool:
    if isinstance(raw, bool):
        return raw
    return str(raw or "").strip().lower() in {"1", "true", "yes", "on"}


def _safe_list(raw: Any) -> list[Any]:
    return raw if isinstance(raw, list) else []


def _as_dict(raw: Any) -> dict[str, Any]:
    return raw if isinstance(raw, dict) else {}


def _guarded_paper_strict_clear_for_spec(spec: dict[str, Any]) -> bool:
    artifact_path = Path(spec.get("artifact_path") or "")
    health_fast = load_json(artifact_path.parent / "health_fast_latest.json")
    operational = _as_dict(health_fast.get("operational_readiness"))
    guarded_paper = _as_dict(operational.get("guarded_paper"))
    live_execution = _as_dict(operational.get("live_execution"))
    guarded_ready = bool(guarded_paper.get("ok", False)) and str(guarded_paper.get("status") or "").strip().lower() in {
        "ready",
        "armed",
        "guarded_ready",
    }
    live_locked = str(live_execution.get("status") or "").strip().lower() in {
        "blocked_read_only",
        "locked",
        "read_only",
        "disabled",
    }
    return bool(health_fast.get("strict_all_clear", False) and guarded_ready and live_locked)


def _recovery_deferred_reason(spec: dict[str, Any], payload: dict[str, Any], status: str) -> str:
    if status != "blocked" or not payload:
        return ""

    guarded_paper_strict_clear = _guarded_paper_strict_clear_for_spec(spec)
    surface_name = str(spec.get("name") or "").strip()
    if guarded_paper_strict_clear and surface_name == "architecture_upgrade_scoreboard":
        blocked_slugs = {
            str(row.get("slug") or "").strip()
            for row in _safe_list(payload.get("rows"))
            if isinstance(row, dict) and str(row.get("status") or "").strip().lower() in {"blocked", "critical"}
        }
        if blocked_slugs and blocked_slugs <= {"self_healing_ops_plane", "immutable_incident_review"}:
            return "guarded_paper_architecture_recovery_debt"

    if guarded_paper_strict_clear and surface_name == "master_infrastructure_supervisor":
        blocked_checks = {
            str(row.get("name") or "").strip()
            for row in _safe_list(payload.get("checks"))
            if isinstance(row, dict) and str(row.get("status") or "").strip().lower() in {"blocked", "critical"}
        }
        if blocked_checks and blocked_checks <= {
            "external_drive_route_health",
            "governance_artifact_freshness",
            "self_auditing_infra_bots",
        }:
            return "guarded_paper_infrastructure_recovery_debt"

    recovery_state = str(payload.get("recovery_state") or "").strip().lower()
    if recovery_state in RECOVERY_STATES:
        return f"recovery_state={recovery_state}"

    storage = payload.get("storage") if isinstance(payload.get("storage"), dict) else {}
    backlog_drain_status = str(storage.get("backlog_drain_status") or "").strip().lower()
    if backlog_drain_status in RECOVERY_STATES:
        return f"backlog_drain_status={backlog_drain_status}"

    pressure_deferred = _safe_int(payload.get("pressure_deferred_count"), 0)
    critical_regressions = _safe_int(payload.get("critical_regression_count"), 0)
    if pressure_deferred > 0 and critical_regressions <= pressure_deferred:
        return f"pressure_deferred_count={pressure_deferred}"

    if (
        _safe_bool(payload.get("execute_safe_repairs")) is False
        and _safe_int(payload.get("safe_repair_step_count"), 0) > 0
        and _safe_int(payload.get("attempt_count"), 0) == 0
    ):
        return "safe_repairs_planned_not_executed"

    if _safe_int(payload.get("authority_violation_count"), 0) == 0:
        blocked_nodes = {str(item or "").strip() for item in _safe_list(payload.get("blocked_nodes")) if str(item or "").strip()}
        if blocked_nodes and blocked_nodes <= PROTECTED_ARCHITECTURE_BLOCKED_NODES:
            return "protected_architecture_dependencies_blocked"

    rows = _safe_list(payload.get("rows"))
    non_ready_slugs = {
        str(row.get("slug") or "").strip()
        for row in rows
        if isinstance(row, dict) and str(row.get("status") or "").strip().lower() in {"blocked", "missing"}
    }
    if non_ready_slugs and non_ready_slugs <= SOFT_ARCHITECTURE_SCOREBOARD_BLOCKERS:
        return "soft_architecture_scoreboard_blockers"

    return ""


def _artifact_candidates(path: Path) -> list[Path]:
    candidates = [path]
    try:
        rel_path = path.relative_to(PROJECT_ROOT)
    except Exception:
        rel_path = None
    if rel_path is not None and rel_path.parts and rel_path.parts[0] in {"data", "decisions", "decision_explanations", "exports", "governance", "logs", "models"}:
        candidates.append(PROJECT_ROOT / "local_fallback_storage" / rel_path)
        external_root = Path(os.getenv("BOT_LOGS_EXTERNAL_PROJECT_ROOT", "/Volumes/BOT_LOGS/schwab_trading_bot")).expanduser()
        candidates.append(external_root / rel_path)
    out: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        out.append(candidate)
    return out


def _load_freshest_artifact(path: Path) -> tuple[Path, dict[str, Any]]:
    best_path: Path | None = None
    best_payload: dict[str, Any] = {}
    best_mtime = -1.0
    for candidate in _artifact_candidates(path):
        payload = load_json(candidate)
        if not payload:
            continue
        try:
            mtime = candidate.stat().st_mtime
        except Exception:
            mtime = 0.0
        if mtime >= best_mtime:
            best_path = candidate
            best_payload = payload
            best_mtime = mtime
    return best_path or path, best_payload


def _command_validity_row(spec: dict[str, Any]) -> dict[str, Any]:
    path, payload = _load_freshest_artifact(Path(spec["artifact_path"]))
    age_minutes = payload_age_minutes(payload, path)
    if not payload:
        status = "missing"
        detail = "artifact_missing"
    else:
        metrics = payload.get("metrics") if isinstance(payload.get("metrics"), dict) else {}
        blocked_entries = _safe_int(metrics.get("blocked_entry_count"), 0)
        smoke_failures = _safe_int(metrics.get("smoke_failure_count"), 0)
        runtime_smoke_failures = _safe_int(metrics.get("runtime_smoke_failure_count"), 0)
        operator_gated = _safe_int(metrics.get("operator_gated_entry_count"), 0)
        if blocked_entries > 0 or smoke_failures > 0 or runtime_smoke_failures > 0:
            status = "blocked"
        else:
            status = "ready"
        detail = (
            f"blocked_entries={blocked_entries} smoke_failures={smoke_failures} "
            f"runtime_smoke_failures={runtime_smoke_failures} operator_gated={operator_gated}"
        )
    stale = False
    max_age_minutes = spec.get("max_age_minutes")
    if payload and isinstance(max_age_minutes, (int, float)) and isinstance(age_minutes, (int, float)) and age_minutes > float(max_age_minutes):
        if status == "ready":
            status = "degraded"
        stale = True
        detail = f"{detail} stale_minutes={age_minutes:.2f}"
    return {
        "name": spec["name"],
        "family": spec["family"],
        "path": str(path),
        "exists": path.exists(),
        "status": status,
        "ok": status == "ready",
        "age_minutes": age_minutes,
        "stale": stale,
        "detail": detail,
        "repair_commands": list(spec.get("repair_commands") or []),
        "notes": list(spec.get("notes") or []),
        "assigned_bot": str(spec.get("assigned_bot") or ""),
        "owner_bot": str(spec.get("owner_bot") or ""),
    }


def _commands_hygiene_row(spec: dict[str, Any]) -> dict[str, Any]:
    path, payload = _load_freshest_artifact(Path(spec["artifact_path"]))
    age_minutes = payload_age_minutes(payload, path)
    if not payload:
        status = "missing"
        detail = "artifact_missing"
    else:
        commands_changed = bool(payload.get("commands_changed", False))
        runbook_changed = bool(payload.get("runbook_changed", False))
        apply_results = payload.get("apply_results") if isinstance(payload.get("apply_results"), dict) else {}
        commands_written = bool(apply_results.get("commands_md_written", False))
        runbook_written = bool(apply_results.get("runbook_written", False))
        if commands_changed and not commands_written:
            status = "degraded"
        elif runbook_changed and not runbook_written:
            status = "degraded"
        else:
            status = "ready"
        detail = (
            f"commands_changed={str(commands_changed).lower()} runbook_changed={str(runbook_changed).lower()} "
            f"commands_written={str(commands_written).lower()} runbook_written={str(runbook_written).lower()}"
        )
    stale = False
    max_age_minutes = spec.get("max_age_minutes")
    if payload and isinstance(max_age_minutes, (int, float)) and isinstance(age_minutes, (int, float)) and age_minutes > float(max_age_minutes):
        if status == "ready":
            status = "degraded"
        stale = True
        detail = f"{detail} stale_minutes={age_minutes:.2f}"
    return {
        "name": spec["name"],
        "family": spec["family"],
        "path": str(path),
        "exists": path.exists(),
        "status": status,
        "ok": status == "ready",
        "age_minutes": age_minutes,
        "stale": stale,
        "detail": detail,
        "repair_commands": list(spec.get("repair_commands") or []),
        "notes": list(spec.get("notes") or []),
    }


def _watchdog_row(spec: dict[str, Any]) -> dict[str, Any]:
    path, payload = _load_freshest_artifact(Path(spec["artifact_path"]))
    age_minutes = payload_age_minutes(payload, path)
    if not payload:
        status = "missing"
        detail = "artifact_missing"
    else:
        restart_storms = payload.get("restart_storms") if isinstance(payload.get("restart_storms"), list) else []
        recent = payload.get("recent_restart_storms") if isinstance(payload.get("recent_restart_storms"), list) else []
        unresolved_recent = [row for row in recent if isinstance(row, dict) and not bool(row.get("resolved", False))]
        alerts = payload.get("alerts") if isinstance(payload.get("alerts"), list) else []
        safety_pause = payload.get("safety_pause") if isinstance(payload.get("safety_pause"), dict) else {}
        if restart_storms or unresolved_recent:
            status = "blocked"
        elif alerts:
            status = "degraded"
        else:
            status = "ready"
        detail = (
            f"restart_storms={len(restart_storms)} unresolved_recent={len(unresolved_recent)} "
            f"alerts={len(alerts)} safety_pause={str(safety_pause.get('reason') or '').strip() or 'none'}"
        )
    stale = False
    max_age_minutes = spec.get("max_age_minutes")
    if payload and isinstance(max_age_minutes, (int, float)) and isinstance(age_minutes, (int, float)) and age_minutes > float(max_age_minutes):
        if status == "ready":
            status = "degraded"
        stale = True
        detail = f"{detail} stale_minutes={age_minutes:.2f}"
    return {
        "name": spec["name"],
        "family": spec["family"],
        "path": str(path),
        "exists": path.exists(),
        "status": status,
        "ok": status == "ready",
        "age_minutes": age_minutes,
        "stale": stale,
        "detail": detail,
        "repair_commands": list(spec.get("repair_commands") or []),
        "notes": list(spec.get("notes") or []),
        "assigned_bot": str(spec.get("assigned_bot") or ""),
        "owner_bot": str(spec.get("owner_bot") or ""),
    }


def _generic_row(spec: dict[str, Any]) -> dict[str, Any]:
    path, payload = _load_freshest_artifact(Path(spec["artifact_path"]))
    age_minutes = payload_age_minutes(payload, path)
    recovery_deferred_reason = ""
    if not payload:
        status = "missing"
        detail = "artifact_missing"
    else:
        status_key = str(spec.get("status_key") or "").strip()
        ok_key = str(spec.get("ok_key") or "").strip()
        raw_status = str(payload.get(status_key) or "").strip().lower() if status_key else ""
        if not raw_status and ok_key:
            raw_status = _status_from_bool(payload.get(ok_key))
        status = _normalize_status(raw_status or "ready", payload.get(ok_key) if ok_key else None)
        recovery_deferred_reason = _recovery_deferred_reason(spec, payload, status)
        if recovery_deferred_reason:
            status = "degraded"
        detail = status
        if recovery_deferred_reason:
            detail = f"{detail} recovery_deferred={recovery_deferred_reason}"
    stale = False
    max_age_minutes = spec.get("max_age_minutes")
    if payload and isinstance(max_age_minutes, (int, float)) and isinstance(age_minutes, (int, float)) and age_minutes > float(max_age_minutes):
        if status == "ready":
            status = "degraded"
        stale = True
        detail = f"{detail} stale_minutes={age_minutes:.2f}"
    return {
        "name": spec["name"],
        "family": spec["family"],
        "path": str(path),
        "exists": path.exists(),
        "status": status,
        "ok": status == "ready",
        "age_minutes": age_minutes,
        "stale": stale,
        "detail": detail,
        "recovery_deferred": bool(recovery_deferred_reason),
        "recovery_deferred_reason": recovery_deferred_reason,
        "repair_commands": list(spec.get("repair_commands") or []),
        "notes": list(spec.get("notes") or []),
        "assigned_bot": str(spec.get("assigned_bot") or ""),
        "owner_bot": str(spec.get("owner_bot") or ""),
    }


def _surface_row(spec: dict[str, Any]) -> dict[str, Any]:
    kind = str(spec.get("kind") or "").strip()
    if kind == "commands_hygiene":
        return _commands_hygiene_row(spec)
    if kind == "command_validity":
        return _command_validity_row(spec)
    if kind == "watchdog":
        return _watchdog_row(spec)
    return _generic_row(spec)


def build_payload(project_root: Path = PROJECT_ROOT) -> dict[str, Any]:
    rows = [_surface_row(spec) for spec in surface_specs(project_root)]
    blocked = [row for row in rows if str(row.get("status") or "") in BLOCKED_STATUSES]
    degraded = [row for row in rows if str(row.get("status") or "") in DEGRADED_STATUSES]
    stale = [row for row in rows if bool(row.get("stale", False))]
    missing = [row for row in rows if str(row.get("status") or "") == "missing"]
    overall_status = "ready"
    if blocked:
        overall_status = "blocked"
    elif degraded:
        overall_status = "degraded"

    family_metrics: dict[str, dict[str, int]] = {}
    for row in rows:
        family = str(row.get("family") or "other")
        metrics = family_metrics.setdefault(family, {"surface_count": 0, "blocked_count": 0, "degraded_count": 0})
        metrics["surface_count"] += 1
        if str(row.get("status") or "") in BLOCKED_STATUSES:
            metrics["blocked_count"] += 1
        elif str(row.get("status") or "") in DEGRADED_STATUSES:
            metrics["degraded_count"] += 1

    recommended_actions = ordered_unique(
        [
            "run `./scripts/ops/opsctl.sh system-drift-autopilot --apply --json` to repair safe drift surfaces in one pass"
            if blocked or degraded
            else "",
        ]
        + [
            f"repair {row['name']} with `{ ' '.join(row.get('repair_commands', [])[0]) }`"
            for row in blocked[:6]
            if row.get("repair_commands")
        ]
        + [str(note) for row in blocked[:6] for note in (row.get("notes") or [])]
    )

    return {
        "timestamp_utc": iso_now(),
        "schema_version": 2,
        "ok": overall_status == "ready",
        "overall_status": overall_status,
        "surfaces": rows,
        "families": family_metrics,
        "metrics": {
            "surface_count": len(rows),
            "blocked_surface_count": len(blocked),
            "degraded_surface_count": len(degraded),
            "stale_surface_count": len(stale),
            "missing_surface_count": len(missing),
            "repairable_surface_count": sum(1 for row in rows if row.get("repair_commands")),
        },
        "recommended_actions": recommended_actions,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Aggregate system-wide drift surfaces into a registry-backed guard artifact.")
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--out-file")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    payload = build_payload(project_root)
    out_file = Path(args.out_file).expanduser() if args.out_file else project_root / "governance" / "health" / "system_drift_guard_latest.json"
    write_payload(out_file, payload)

    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        print(
            "system_drift_guard "
            f"overall_status={payload.get('overall_status', '')} "
            f"blocked={int((payload.get('metrics') or {}).get('blocked_surface_count', 0) or 0)} "
            f"degraded={int((payload.get('metrics') or {}).get('degraded_surface_count', 0) or 0)}"
        )
    return 0 if payload.get("overall_status") in {"ready", "degraded"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
