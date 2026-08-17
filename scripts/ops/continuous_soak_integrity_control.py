#!/usr/bin/env python3
from __future__ import annotations

import argparse
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


def _as_dict(raw: Any) -> dict[str, Any]:
    return raw if isinstance(raw, dict) else {}


def _as_list(raw: Any) -> list[Any]:
    return raw if isinstance(raw, list) else []


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
    starts = [parse_iso_utc(value) for value in windows.values()]
    parsed_starts = [value for value in starts if value is not None]
    clean_start = max(parsed_starts) if parsed_starts else None
    observed_window_elapsed_hours = (
        max((current_time - clean_start).total_seconds() / 3600.0, 0.0) if clean_start else 0.0
    )
    credited_clean_window_elapsed_hours = observed_window_elapsed_hours if candidate_ready else 0.0
    elapsed_complete = bool(candidate_ready and credited_clean_window_elapsed_hours >= 720.0)
    implemented_count = sum(1 for row in control_rows if row["implemented"])
    runtime_ready_count = sum(1 for value in runtime_checks.values() if value)
    control_ready = implemented_count == len(control_rows)
    capacity_ready = bool(control_ready and all(runtime_checks.values()))
    control_score = 100.0 * implemented_count / max(len(control_rows), 1)
    runtime_score = 100.0 * runtime_ready_count / max(len(runtime_checks), 1)
    elapsed_score = min(100.0 * credited_clean_window_elapsed_hours / 720.0, 100.0)
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
        "elapsed_evidence_grade": _grade(elapsed_score, complete=elapsed_complete),
        "elapsed_evidence_score": round(elapsed_score, 3),
        "clean_window_started_utc": clean_start.isoformat() if clean_start else "",
        "clean_window_elapsed_hours": round(credited_clean_window_elapsed_hours, 6),
        "observed_window_elapsed_hours": round(observed_window_elapsed_hours, 6),
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
            "unaccepted_candidate_drift_receives_zero_elapsed_credit": True,
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
