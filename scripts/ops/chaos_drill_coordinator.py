#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from scripts.ops.long_runtime_common import PROJECT_ROOT, iso_now, load_json, ordered_unique, parse_iso_utc, utc_now, write_payload
    from scripts.ops.production_recovery_drill_harness import build_payload as build_isolated_drill_payload
else:
    from .long_runtime_common import PROJECT_ROOT, iso_now, load_json, ordered_unique, parse_iso_utc, utc_now, write_payload
    from .production_recovery_drill_harness import build_payload as build_isolated_drill_payload


DEFAULT_OUT_PATH = PROJECT_ROOT / "governance" / "health" / "chaos_drill_coordinator_latest.json"
DEFAULT_STATE_PATH = PROJECT_ROOT / "governance" / "runtime" / "chaos_drill_state.json"
DEFAULT_ISOLATED_HARNESS_PATH = PROJECT_ROOT / "governance" / "health" / "production_recovery_drill_harness_latest.json"
REQUIRED_PRODUCTION_DRILLS = (
    "auth_expiry",
    "broker_network_outage",
    "managed_process_crash",
    "reboot_blackstart",
    "disk_capacity_exhaustion",
    "external_storage_loss",
    "memory_pressure",
    "database_corruption_or_lock",
    "market_data_delay_or_malformed_payload",
    "order_reject_partial_fill_cancel_replace",
)


def _load_state(path: Path) -> dict[str, Any]:
    payload = load_json(path)
    return payload if isinstance(payload.get("drills"), dict) else {"drills": {}}


def _save_state(path: Path, payload: dict[str, Any]) -> None:
    write_payload(path, payload)


def _record_isolated_harness(
    project_root: Path,
    *,
    state_path: Path,
    harness_path: Path,
) -> dict[str, Any]:
    payload = build_isolated_drill_payload(project_root)
    write_payload(harness_path, payload)
    if not bool(payload.get("production_recovery_evidence", False)):
        return payload
    state = _load_state(state_path)
    drills = state.get("drills") if isinstance(state.get("drills"), dict) else {}
    for row in payload.get("drills") if isinstance(payload.get("drills"), list) else []:
        if not isinstance(row, dict):
            continue
        drill_name = str(row.get("drill") or "").strip()
        if drill_name not in REQUIRED_PRODUCTION_DRILLS:
            continue
        drills[drill_name] = {
            "completed_at_utc": str(payload.get("timestamp_utc") or iso_now()),
            "result": str(row.get("result") or "fail"),
            "recovery_seconds": max(float(row.get("recovery_seconds", 0.0) or 0.0), 0.0),
            "containment_verified": bool(row.get("containment_verified", False)),
            "no_duplicate_orders": bool(row.get("no_duplicate_orders", False)),
            "evidence": f"{harness_path}#{drill_name}:{str(row.get('evidence_sha256') or '')}",
            "evidence_sha256": str(row.get("evidence_sha256") or ""),
            "evidence_class": str(payload.get("evidence_class") or ""),
            "harness_run_sha256": str(payload.get("run_sha256") or ""),
            "real_outage_evidence": bool(payload.get("real_outage_evidence", False)),
            "note": "isolated non-destructive production recovery drill",
        }
    state["drills"] = drills
    state["last_isolated_harness"] = {
        "timestamp_utc": payload.get("timestamp_utc"),
        "run_sha256": payload.get("run_sha256"),
        "path": str(harness_path),
        "ok": payload.get("ok"),
    }
    _save_state(state_path, state)
    return payload


def build_payload(
    project_root: Path = PROJECT_ROOT,
    *,
    state_path: Path = DEFAULT_STATE_PATH,
    overdue_days: float = 7.0,
) -> dict[str, Any]:
    health_root = project_root / "governance" / "health"
    state = _load_state(state_path)
    snapshot_drill_path = project_root / "exports" / "state_snapshot_drills" / "latest.json"
    snapshot_drill = load_json(snapshot_drill_path)
    reboot_resilience_path = health_root / "reboot_resilience_latest.json"
    reboot_resilience = load_json(reboot_resilience_path)
    storage_resilience_path = health_root / "storage_resilience_control_latest.json"
    storage_resilience = load_json(storage_resilience_path)
    token_guard_path = health_root / "premarket_token_guard_latest.json"
    token_guard = load_json(token_guard_path)
    process_watchdog_path = health_root / "process_watchdog_latest.json"
    process_watchdog = load_json(process_watchdog_path)
    backup_restore_events = sorted(project_root.glob("governance/watchdog/backup_restore_events.jsonl*"))
    weekly_drill_installer = project_root / "scripts" / "install_weekly_dr_drill_launchd.sh"
    snapshot_drill_script = project_root / "scripts" / "daily_state_snapshot_drill.py"
    backup_restore_script = project_root / "scripts" / "backup_restore_verify.py"

    observed_sources = {
        "snapshot_restore": snapshot_drill.get("timestamp_utc"),
        "reboot_blackstart": reboot_resilience.get("timestamp_utc"),
        "external_storage_loss": storage_resilience.get("timestamp_utc"),
        "auth_expiry": token_guard.get("timestamp_utc"),
        "managed_process_crash": process_watchdog.get("timestamp_utc"),
    }

    overdue: list[dict[str, Any]] = []
    drills: list[dict[str, Any]] = []
    cutoff_days = float(overdue_days)
    failed_drills: list[dict[str, Any]] = []
    unverified_drills: list[dict[str, Any]] = []
    for drill_name in REQUIRED_PRODUCTION_DRILLS:
        source_ts = observed_sources.get(drill_name)
        recorded = ((state.get("drills") or {}).get(drill_name)) if isinstance(state.get("drills"), dict) else None
        ts = parse_iso_utc((recorded or {}).get("completed_at_utc")) if isinstance(recorded, dict) else None
        observed_ts = parse_iso_utc(source_ts)
        age_days = None
        if ts is not None:
            age_days = max((utc_now() - ts).total_seconds() / 86400.0, 0.0)
        recorded_drill = bool(isinstance(recorded, dict) and ts is not None)
        result = str((recorded or {}).get("result") or "").strip().lower() if isinstance(recorded, dict) else ""
        containment_verified = bool((recorded or {}).get("containment_verified", False)) if isinstance(recorded, dict) else False
        no_duplicate_orders = bool((recorded or {}).get("no_duplicate_orders", False)) if isinstance(recorded, dict) else False
        recovery_seconds = None
        if isinstance(recorded, dict) and recorded.get("recovery_seconds") is not None:
            try:
                recovery_seconds = max(float(recorded.get("recovery_seconds")), 0.0)
            except Exception:
                recovery_seconds = None
        verified = bool(recorded_drill and result == "pass" and containment_verified and no_duplicate_orders and recovery_seconds is not None)
        is_overdue = not verified or age_days is None or float(age_days) > cutoff_days
        row = {
            "drill": drill_name,
            "completed_at_utc": ts.isoformat() if ts is not None else "",
            "observed_runtime_evidence_at_utc": observed_ts.isoformat() if observed_ts is not None else "",
            "age_days": round(float(age_days), 4) if age_days is not None else None,
            "overdue": bool(is_overdue),
            "recorded_drill": recorded_drill,
            "result": result or "pending",
            "containment_verified": containment_verified,
            "no_duplicate_orders": no_duplicate_orders,
            "recovery_seconds": recovery_seconds,
            "evidence": str((recorded or {}).get("evidence") or "") if isinstance(recorded, dict) else "",
            "note": str((recorded or {}).get("note") or "") if isinstance(recorded, dict) else "",
            "evidence_sha256": str((recorded or {}).get("evidence_sha256") or "") if isinstance(recorded, dict) else "",
            "evidence_class": str((recorded or {}).get("evidence_class") or "") if isinstance(recorded, dict) else "",
            "harness_run_sha256": str((recorded or {}).get("harness_run_sha256") or "") if isinstance(recorded, dict) else "",
            "real_outage_evidence": bool((recorded or {}).get("real_outage_evidence", False)) if isinstance(recorded, dict) else False,
            "verified": verified,
        }
        drills.append(row)
        if is_overdue:
            overdue.append(row)
        if result == "fail":
            failed_drills.append(row)
        elif not verified:
            unverified_drills.append(row)

    restore_discipline = {
        "snapshot_restore_present": bool(snapshot_drill),
        "storage_resilience_present": bool(storage_resilience),
        "backup_restore_event_log_count": len(backup_restore_events),
        "restore_proof_ready": bool(snapshot_drill) and bool(storage_resilience) and bool(backup_restore_events),
    }
    schedule_contract = {
        "weekly_drill_installer_present": weekly_drill_installer.exists(),
        "snapshot_drill_script_present": snapshot_drill_script.exists(),
        "backup_restore_script_present": backup_restore_script.exists(),
        "discipline_ready": weekly_drill_installer.exists() and snapshot_drill_script.exists() and backup_restore_script.exists(),
    }
    overall_status = "ready"
    if failed_drills or (overdue and (not schedule_contract["discipline_ready"] or not restore_discipline["restore_proof_ready"])):
        overall_status = "blocked"
    elif unverified_drills or overdue:
        overall_status = "evidence_pending"

    recommended_actions = ordered_unique(
        [
            "record every required production drill with pass/fail, containment, recovery time, and duplicate-order proof" if overdue else "",
            "exercise auth, broker network, process, storage, memory, database, market-data, and order-lifecycle faults weekly" if len(overdue) >= 1 else "",
            "keep the weekly restore drill installer and snapshot/restore scripts present so resilience stays a scheduled discipline"
            if not weekly_drill_installer.exists()
            else "",
        ]
    )
    program_score = max(0.0, round(100.0 - (18.0 * len(overdue)), 2))
    if restore_discipline["restore_proof_ready"]:
        program_score = min(program_score + 8.0, 100.0)
    if schedule_contract["discipline_ready"]:
        program_score = min(program_score + 6.0, 100.0)
    next_priority_drill = str((overdue[0] or {}).get("drill") or "") if overdue else ""

    return {
        "timestamp_utc": iso_now(),
        "schema_version": 1,
        "ok": overall_status == "ready",
        "overall_status": overall_status,
        "overdue_days_threshold": cutoff_days,
        "drills": drills,
        "overdue_drills": overdue,
        "unverified_drills": unverified_drills,
        "failed_drills": failed_drills,
        "required_drills": list(REQUIRED_PRODUCTION_DRILLS),
        "verified_drill_count": sum(1 for row in drills if row.get("verified", False)),
        "required_drill_count": len(REQUIRED_PRODUCTION_DRILLS),
        "drill_program": {
            "program_score": program_score,
            "next_priority_drill": next_priority_drill,
            "weekly_cadence_target_days": cutoff_days,
            "automation_ready": True,
        },
        "restore_discipline": restore_discipline,
        "schedule_contract": schedule_contract,
        "state_path": str(state_path),
        "infra_bots": ["chaos_drill_coordinator", "daily_state_snapshot_drill", "reboot_resilience_guard", "process_watchdog"],
        "evidence_scope": {
            "isolated_non_destructive_drills_accepted": True,
            "real_outage_evidence_required": False,
            "live_execution_authority": False,
            "policy": "weekly isolated recovery proof is required; real destructive outages are never triggered by automation",
        },
        "recommended_actions": recommended_actions,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Coordinate weekly chaos drills and record drill completions.")
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--state-path", default=str(DEFAULT_STATE_PATH))
    parser.add_argument("--overdue-days", type=float, default=7.0)
    parser.add_argument("--record-drill", default="")
    parser.add_argument("--note", default="")
    parser.add_argument("--result", choices=("pass", "fail"), default="pass")
    parser.add_argument("--recovery-seconds", type=float)
    parser.add_argument("--containment-verified", action="store_true")
    parser.add_argument("--no-duplicate-orders", action="store_true")
    parser.add_argument("--evidence", default="")
    parser.add_argument("--run-isolated", action="store_true")
    parser.add_argument("--isolated-harness-file", default=str(DEFAULT_ISOLATED_HARNESS_PATH))
    parser.add_argument("--out-file", default=str(DEFAULT_OUT_PATH))
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    state_path = Path(args.state_path).expanduser()
    if args.run_isolated:
        harness_path = Path(args.isolated_harness_file).expanduser()
        if not harness_path.is_absolute():
            harness_path = project_root / harness_path
        _record_isolated_harness(
            project_root,
            state_path=state_path,
            harness_path=harness_path,
        )
    if str(args.record_drill or "").strip():
        state = _load_state(state_path)
        drills = state.get("drills") if isinstance(state.get("drills"), dict) else {}
        drill_name = str(args.record_drill).strip()
        if drill_name not in REQUIRED_PRODUCTION_DRILLS:
            parser.error(f"unknown drill {drill_name!r}; choose one of: {', '.join(REQUIRED_PRODUCTION_DRILLS)}")
        if args.recovery_seconds is None:
            parser.error("--record-drill requires --recovery-seconds")
        drills[drill_name] = {
            "completed_at_utc": iso_now(),
            "result": str(args.result),
            "recovery_seconds": max(float(args.recovery_seconds), 0.0),
            "containment_verified": bool(args.containment_verified),
            "no_duplicate_orders": bool(args.no_duplicate_orders),
            "evidence": str(args.evidence or ""),
            "note": str(args.note or ""),
        }
        state["drills"] = drills
        _save_state(state_path, state)

    payload = build_payload(project_root, state_path=state_path, overdue_days=float(args.overdue_days))
    out_path = Path(args.out_file).expanduser()
    write_payload(out_path, payload)
    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        print(
            "chaos_drill_coordinator "
            f"overall_status={payload.get('overall_status', '')} "
            f"overdue={len(payload.get('overdue_drills') or [])}"
        )
    return 0 if payload.get("overall_status") in {"ready", "evidence_pending"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
