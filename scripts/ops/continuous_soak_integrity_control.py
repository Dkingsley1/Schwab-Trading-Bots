#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gzip
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from scripts.ops.long_runtime_common import iso_now, load_json, parse_iso_utc, write_payload
else:
    from .long_runtime_common import PROJECT_ROOT, iso_now, load_json, parse_iso_utc, write_payload


DEFAULT_OUT_PATH = PROJECT_ROOT / "governance" / "health" / "continuous_soak_integrity_control_latest.json"
DEFAULT_MAINTENANCE_DIR = PROJECT_ROOT / "governance" / "maintenance_events"


def _as_dict(raw: Any) -> dict[str, Any]:
    return raw if isinstance(raw, dict) else {}


def _as_list(raw: Any) -> list[Any]:
    return raw if isinstance(raw, list) else []


def _as_float(raw: Any, default: float = 0.0) -> float:
    try:
        return float(raw)
    except (TypeError, ValueError):
        return default


def _status(raw: Any) -> str:
    return str(raw or "").strip().lower()


def _grade(score: float, *, complete: bool = False) -> str:
    if complete and score >= 100.0:
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


def _artifact_ready(payload: dict[str, Any], *, grades: set[str] | None = None) -> bool:
    if not payload:
        return False
    status_ready = _status(payload.get("overall_status") or payload.get("status")) in {
        "ready",
        "stable",
        "healthy",
        "ready_locked",
        "advisory",
    }
    if grades is None:
        return bool(payload.get("ok", status_ready) and status_ready)
    grade = str(payload.get("overall_grade") or payload.get("grade") or "").strip().upper().replace("A++", "A+")
    return bool(status_ready and grade in grades)


def _read_candidate_events(path: Path) -> list[dict[str, Any]]:
    source = path
    if not source.is_file():
        compressed = path.with_suffix(path.suffix + ".gz")
        if compressed.is_file():
            source = compressed
        else:
            return []
    try:
        if source.suffix == ".gz":
            with gzip.open(source, "rt", encoding="utf-8") as handle:
                text = handle.read()
        else:
            text = source.read_text(encoding="utf-8")
    except (OSError, UnicodeError):
        return []
    rows: list[dict[str, Any]] = []
    for line in text.splitlines():
        try:
            row = json.loads(line)
        except (TypeError, ValueError):
            continue
        if isinstance(row, dict) and parse_iso_utc(row.get("timestamp_utc")) is not None:
            rows.append(row)
    return sorted(rows, key=lambda row: parse_iso_utc(row.get("timestamp_utc")) or datetime.min.replace(tzinfo=timezone.utc))


def _planned_maintenance_windows(
    project_root: Path,
    *,
    current_time: datetime,
) -> list[dict[str, Any]]:
    directory = project_root / "governance" / "maintenance_events"
    if not directory.is_dir():
        return []
    windows: dict[str, dict[str, Any]] = {}
    for path in sorted(directory.glob("*.json")):
        if path.name.startswith("latest_"):
            continue
        payload = load_json(path)
        classification = _status(payload.get("classification"))
        accounting = _as_dict(payload.get("soak_accounting"))
        if not classification.startswith("planned_"):
            continue
        if _status(payload.get("status")) != "completed":
            continue
        if bool(accounting.get("counts_as_system_degradation", payload.get("counts_as_system_degradation", True))):
            continue
        if bool(accounting.get("counts_as_trading_system_failure", payload.get("counts_as_trading_system_failure", True))):
            continue
        offline = _as_dict(payload.get("actual_offline_window"))
        start = parse_iso_utc(
            offline.get("offline_start_utc")
            or offline.get("start_utc")
        )
        end = parse_iso_utc(
            offline.get("offline_end_utc")
            or offline.get("end_utc")
        )
        if start is None or end is None or end <= start or start >= current_time:
            continue
        end = min(end, current_time)
        event_id = str(payload.get("event_id") or path.stem)
        windows[event_id] = {
            "event_id": event_id,
            "classification": classification,
            "title": str(payload.get("title") or event_id),
            "offline_start_utc": start.isoformat(),
            "offline_end_utc": end.isoformat(),
            "duration_hours": round((end - start).total_seconds() / 3600.0, 6),
            "source_path": str(path),
            "counts_as_system_degradation": False,
            "counts_as_trading_system_failure": False,
            "resets_candidate_clock": False,
            "earns_active_runtime_credit": False,
        }
    return sorted(windows.values(), key=lambda row: str(row.get("offline_start_utc") or ""))


def _maintenance_overlap_hours(
    windows: list[dict[str, Any]],
    *,
    window_start: datetime | None,
    window_end: datetime,
) -> float:
    if window_start is None or window_end <= window_start:
        return 0.0
    total_seconds = 0.0
    for row in windows:
        start = parse_iso_utc(row.get("offline_start_utc"))
        end = parse_iso_utc(row.get("offline_end_utc"))
        if start is None or end is None:
            continue
        overlap_start = max(start, window_start)
        overlap_end = min(end, window_end)
        if overlap_end > overlap_start:
            total_seconds += (overlap_end - overlap_start).total_seconds()
    return total_seconds / 3600.0


def _historical_soak_evidence(
    *,
    candidate_state: dict[str, Any],
    event_path: Path,
    current_time: datetime,
    scope_starts: dict[str, datetime],
    clean_start: datetime | None,
) -> dict[str, Any]:
    events = _read_candidate_events(event_path)
    initialized = parse_iso_utc(candidate_state.get("initialized_at_utc"))
    if initialized is None and events:
        initialized = parse_iso_utc(events[0].get("timestamp_utc"))
    if initialized is None and scope_starts:
        initialized = min(scope_starts.values())
    if initialized is None:
        return {
            "available": False,
            "source_path": str(event_path),
            "policy": "historical candidate time remains visible but never substitutes for the current clean 720-hour window",
        }

    valid_events = []
    for row in events:
        timestamp = parse_iso_utc(row.get("timestamp_utc"))
        if timestamp is not None and initialized <= timestamp <= current_time:
            valid_events.append((timestamp, row))

    segments: list[dict[str, Any]] = []
    cursor = initialized
    active_candidate_id = "pre_event_history"
    active_generation: int | None = None
    for timestamp, event in valid_events:
        if timestamp > cursor:
            segments.append(
                {
                    "candidate_id": active_candidate_id,
                    "generation": active_generation,
                    "started_utc": cursor.isoformat(),
                    "ended_utc": timestamp.isoformat(),
                    "elapsed_hours": round(max((timestamp - cursor).total_seconds() / 3600.0, 0.0), 6),
                    "ended_by_event_type": str(event.get("event_type") or ""),
                    "ended_by_change_reason": str(event.get("change_reason") or ""),
                    "ended_by_changed_scopes": [str(item) for item in _as_list(event.get("changed_scopes"))],
                }
            )
        cursor = max(cursor, timestamp)
        active_candidate_id = str(event.get("candidate_id") or "unknown")
        try:
            active_generation = int(event.get("generation"))
        except (TypeError, ValueError):
            active_generation = None
    segments.append(
        {
            "candidate_id": active_candidate_id,
            "generation": active_generation,
            "started_utc": cursor.isoformat(),
            "ended_utc": current_time.isoformat(),
            "elapsed_hours": round(max((current_time - cursor).total_seconds() / 3600.0, 0.0), 6),
            "ended_by_event_type": "",
            "ended_by_change_reason": "",
            "ended_by_changed_scopes": [],
        }
    )
    historical_hours = max((current_time - initialized).total_seconds() / 3600.0, 0.0)
    pre_latest_reset_hours = (
        max((clean_start - initialized).total_seconds() / 3600.0, 0.0)
        if clean_start is not None
        else historical_hours
    )
    scope_elapsed_hours = {
        scope: round(max((current_time - started).total_seconds() / 3600.0, 0.0), 6)
        for scope, started in sorted(scope_starts.items())
    }
    return {
        "available": True,
        "initial_candidate_started_utc": initialized.isoformat(),
        "historical_segmented_wall_clock_hours": round(historical_hours, 6),
        "historical_segmented_wall_clock_days": round(historical_hours / 24.0, 6),
        "wall_clock_hours_before_latest_full_system_window": round(pre_latest_reset_hours, 6),
        "candidate_event_count": len(valid_events),
        "accepted_change_count": sum(1 for _, row in valid_events if row.get("event_type") == "candidate_change_accepted"),
        "chain_recovery_count": sum(1 for _, row in valid_events if row.get("event_type") == "candidate_chain_recovery_anchor"),
        "segment_count": len(segments),
        "longest_segment_hours": round(max((float(row.get("elapsed_hours", 0.0)) for row in segments), default=0.0), 6),
        "scope_window_elapsed_hours": scope_elapsed_hours,
        "recent_segments": segments[-12:],
        "older_segment_count": max(len(segments) - 12, 0),
        "source_path": str(event_path),
        "historical_time_preserved": True,
        "counts_toward_current_clean_720_hours": False,
        "candidate_event_timeline_is_runtime_heartbeat_proof": False,
        "policy": "preserve pre-reset time as segmented historical wall-clock evidence; only the current unchanged candidate window can satisfy the clean 720-hour promotion contract",
    }


def build_payload(project_root: Path = PROJECT_ROOT, *, now: datetime | None = None) -> dict[str, Any]:
    current_time = now or datetime.now(timezone.utc)
    health = project_root / "governance" / "health"
    unattended = load_json(health / "unattended_soak_readiness_latest.json")
    storage = load_json(health / "ingestion_storage_control_latest.json")
    memory = load_json(health / "memory_efficiency_control_latest.json")
    process = load_json(health / "process_watchdog_latest.json")
    throttle = load_json(health / "runtime_throttle_control_latest.json")
    source = load_json(health / "source_verification_latest.json")
    production = load_json(health / "production_excellence_control_latest.json")
    paper_regression = load_json(health / "runtime_paper_regression_guard_latest.json")
    paper_truth = load_json(health / "paper_execution_truth_layer_latest.json")
    candidate_state = load_json(project_root / "governance" / "runtime" / "production_candidate_state.json")
    candidate = _as_dict(production.get("candidate"))
    chain = _as_dict(candidate.get("event_chain"))
    storage_soak = _as_dict(storage.get("continuous_run_soak_contract"))
    production_source = (
        (project_root / "scripts" / "ops" / "production_excellence_control.py").read_text(encoding="utf-8")
        if (project_root / "scripts" / "ops" / "production_excellence_control.py").is_file()
        else ""
    )
    source_refresh_source = (
        (project_root / "scripts" / "ops" / "source_verification_autorefresh.py").read_text(encoding="utf-8")
        if (project_root / "scripts" / "ops" / "source_verification_autorefresh.py").is_file()
        else ""
    )
    controls = [
        ("01_candidate_hash_chain", "Candidate fingerprints and events are hash-chained", "verify_candidate_event_chain" in production_source),
        ("02_explicit_chain_recovery", "Event-log recovery is explicit and resets every window", "candidate_chain_recovery_anchor" in production_source and "all_evidence_windows_reset" in production_source),
        ("03_drift_resets_scope_clock", "Accepted drift resets every affected evidence scope", "scope_windows_started_utc" in production_source and "changed_scopes" in production_source),
        ("04_full_720_hour_contract", "A clean completion requires the full 720 hours", "required_hours" in production_source and "thirty_day_window" in production_source),
        ("05_unattended_runtime_gate", "Runtime health gates paper soak independently of elapsed time", (project_root / "scripts" / "ops" / "unattended_soak_readiness.py").is_file()),
        ("06_storage_memory_pressure_self_heal", "Storage and memory pressure retain explicit self-healing gates", (project_root / "scripts" / "ops" / "storage_backpressure_autopilot.py").is_file() and (project_root / "scripts" / "ops" / "memory_pressure_intelligence.py").is_file()),
        ("07_source_retry_survives_restart", "Source refresh retry, quarantine, and fairness survive restarts", "source_verification_retry_state.json" in source_refresh_source and "starvation_override" in source_refresh_source),
        ("08_regression_and_incident_evidence", "Regression and incident evidence remain separate from soak credit", (project_root / "scripts" / "ops" / "grade_regression_guard.py").is_file() and (project_root / "scripts" / "ops" / "incident_timeline.py").is_file()),
    ]
    control_rows = [
        {"control_id": control_id, "title": title, "implemented": implemented, "status": "ready" if implemented else "blocked"}
        for control_id, title, implemented in controls
    ]
    storage_ready = bool(storage_soak.get("soak_ready", storage_soak.get("ready", False)))
    candidate_ready = bool(
        candidate.get("candidate_ready", False)
        and not candidate.get("candidate_drift", True)
        and chain.get("ok", False)
        and int(chain.get("event_count", 0) or 0) >= 1
    )
    runtime_checks = {
        "candidate_chain_current": candidate_ready,
        "unattended_runtime_A_plus": _artifact_ready(unattended, grades={"A+"}) and bool(unattended.get("safe_to_leave_unattended", False)),
        "storage_30_day_capacity": storage_ready and float(storage_soak.get("horizon_days", 0.0) or 0.0) >= 30.0,
        "memory_control_ready": _artifact_ready(memory),
        "process_restart_storm_clear": not _as_list(process.get("restart_storms")) and _status(process.get("overall_status")) in {"ready", "stable", "healthy"},
        "runtime_pressure_managed": _status(throttle.get("overall_status")) in {"ready", "advisory"} and _status(throttle.get("memory_pressure_level")) in {"normal", "low", ""},
        "source_hardening_A_plus": str(source.get("source_control_grade") or "").strip().upper() in {"A+", "A++"},
        "paper_runtime_regression_clear": bool(
            paper_regression.get("ok", False)
            and _status(paper_regression.get("overall_status")) == "ready"
            and not _as_list(paper_regression.get("failed_guards"))
        ),
        "paper_truth_reconciled_A_plus": bool(
            paper_truth.get("ok", False)
            and paper_truth.get("a_plus_ready", False)
            and str(paper_truth.get("grade") or paper_truth.get("overall_grade") or "").strip().upper() in {"A+", "A++"}
            and not paper_truth.get("blocked_gates")
        ),
    }
    windows = _as_dict(candidate.get("scope_windows_started_utc"))
    parsed_scope_starts = {
        str(scope): parsed
        for scope, value in windows.items()
        if (parsed := parse_iso_utc(value)) is not None
    }
    parsed_starts = list(parsed_scope_starts.values())
    clean_start = max(parsed_starts) if parsed_starts else None
    observed_window_elapsed_hours = (
        max((current_time - clean_start).total_seconds() / 3600.0, 0.0) if clean_start else 0.0
    )
    planned_maintenance_windows = _planned_maintenance_windows(
        project_root,
        current_time=current_time,
    )
    clean_maintenance_excluded_hours = _maintenance_overlap_hours(
        planned_maintenance_windows,
        window_start=clean_start,
        window_end=current_time,
    )
    active_clean_window_elapsed_hours = max(
        observed_window_elapsed_hours - clean_maintenance_excluded_hours,
        0.0,
    )
    credited_clean_window_elapsed_hours = active_clean_window_elapsed_hours if candidate_ready else 0.0
    elapsed_complete = bool(candidate_ready and credited_clean_window_elapsed_hours >= 720.0)
    implemented_count = sum(1 for row in control_rows if row["implemented"])
    runtime_ready_count = sum(1 for value in runtime_checks.values() if value)
    control_ready = implemented_count == len(control_rows)
    capacity_ready = bool(control_ready and all(runtime_checks.values()))
    control_score = 100.0 * implemented_count / max(len(control_rows), 1)
    runtime_score = 100.0 * runtime_ready_count / max(len(runtime_checks), 1)
    elapsed_score = min(100.0 * credited_clean_window_elapsed_hours / 720.0, 100.0)
    raw_event_path = str(chain.get("path") or "governance/evidence/production_candidate_events.jsonl")
    event_path = Path(raw_event_path)
    if not event_path.is_absolute():
        event_path = project_root / event_path
    historical_soak = _historical_soak_evidence(
        candidate_state=candidate_state,
        event_path=event_path,
        current_time=current_time,
        scope_starts=parsed_scope_starts,
        clean_start=clean_start,
    )
    main_soak_elapsed_hours = _as_float(
        historical_soak.get("historical_segmented_wall_clock_hours"),
        observed_window_elapsed_hours,
    )
    main_soak_elapsed_days = main_soak_elapsed_hours / 24.0
    main_soak_progress_percent = min(100.0 * main_soak_elapsed_hours / 720.0, 100.0)
    initial_soak_start = parse_iso_utc(
        historical_soak.get("initial_candidate_started_utc")
    )
    main_maintenance_excluded_hours = _maintenance_overlap_hours(
        planned_maintenance_windows,
        window_start=initial_soak_start,
        window_end=current_time,
    )
    main_soak_active_runtime_evidence_hours = max(
        main_soak_elapsed_hours - main_maintenance_excluded_hours,
        0.0,
    )
    return {
        "timestamp_utc": iso_now(),
        "schema_version": 1,
        "ok": control_ready,
        "overall_status": "ready" if capacity_ready else "needs_attention",
        "control_grade": _grade(control_score, complete=control_ready),
        "control_score": round(control_score, 3),
        "operational_capacity_grade": _grade(runtime_score, complete=capacity_ready),
        "operational_capacity_score": round(runtime_score, 3),
        "operational_capacity_ready": capacity_ready,
        "safe_for_unattended_paper_soak": capacity_ready,
        "main_soak_elapsed_hours": round(main_soak_elapsed_hours, 6),
        "main_soak_elapsed_days": round(main_soak_elapsed_days, 6),
        "main_soak_progress_percent": round(main_soak_progress_percent, 3),
        "main_soak_counting_mode": "cumulative_segmented_candidate_wall_clock",
        "main_soak_includes_pre_reset_time": True,
        "main_soak_count_is_promotion_credit": False,
        "main_soak_active_runtime_evidence_hours": round(main_soak_active_runtime_evidence_hours, 6),
        "main_soak_planned_maintenance_excluded_hours": round(main_maintenance_excluded_hours, 6),
        "elapsed_evidence_grade": _grade(elapsed_score, complete=elapsed_complete),
        "elapsed_evidence_score": round(elapsed_score, 3),
        "clean_window_started_utc": clean_start.isoformat() if clean_start else "",
        "clean_window_elapsed_hours": round(credited_clean_window_elapsed_hours, 6),
        "observed_window_elapsed_hours": round(observed_window_elapsed_hours, 6),
        "clean_window_planned_maintenance_excluded_hours": round(clean_maintenance_excluded_hours, 6),
        "planned_maintenance": {
            "event_count": len(planned_maintenance_windows),
            "events": planned_maintenance_windows,
            "counts_as_system_degradation": False,
            "counts_as_trading_system_failure": False,
            "current_candidate_reset_count": 0,
            "pre_event_soak_credit_preserved": True,
            "offline_time_earns_active_runtime_credit": False,
            "policy": "planned host maintenance preserves prior soak history and candidate continuity but does not earn active-runtime evidence while the system is offline",
        },
        "historical_soak_evidence": historical_soak,
        "candidate_drift_invalidates_elapsed_credit": bool(clean_start and not candidate_ready),
        "clean_720_hours_complete": elapsed_complete,
        "controls": control_rows,
        "runtime_checks": runtime_checks,
        "blockers": [key for key, value in runtime_checks.items() if not value] + [row["control_id"] for row in control_rows if not row["implemented"]],
        "grading_contract": {
            "control_A_plus_is_hardening_only": True,
            "operational_A_plus_means_capacity_to_run_unattended": True,
            "elapsed_A_plus_requires_720_clean_hours": True,
            "restart_or_accepted_change_resets_credit": True,
            "planned_host_maintenance_is_an_explicit_restart_exception": True,
            "planned_maintenance_preserves_pre_event_credit": True,
            "planned_maintenance_offline_time_earns_credit": False,
            "planned_maintenance_counts_as_failure": False,
            "unaccepted_candidate_drift_receives_zero_elapsed_credit": True,
            "pre_reset_time_is_preserved_as_segmented_history": True,
            "pre_reset_time_is_included_in_main_soak_count": True,
            "main_soak_count_and_clean_promotion_clock_are_separate": True,
            "historical_segmented_time_does_not_replace_clean_candidate_credit": True,
            "no_grade_authorizes_live_money": True,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate continuous-soak hardening, capacity, and elapsed evidence separately.")
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--out-file", type=Path, default=DEFAULT_OUT_PATH)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    project_root = args.project_root.resolve()
    out_path = args.out_file if args.out_file.is_absolute() else project_root / args.out_file
    payload = build_payload(project_root)
    write_payload(out_path, payload)
    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        print(
            "continuous_soak_integrity "
            f"control_grade={payload['control_grade']} capacity_grade={payload['operational_capacity_grade']} "
            f"elapsed_grade={payload['elapsed_evidence_grade']}"
        )
    return 0 if payload.get("ok", False) else 2


if __name__ == "__main__":
    raise SystemExit(main())
