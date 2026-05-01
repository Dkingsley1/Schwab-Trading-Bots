#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from scripts.ops.long_runtime_common import (
        PROJECT_ROOT,
        iso_now,
        load_json,
        load_recent_jsonl,
        ordered_unique,
        payload_age_minutes,
        payload_timestamp,
        write_payload,
    )
else:
    from .long_runtime_common import (
        PROJECT_ROOT,
        iso_now,
        load_json,
        load_recent_jsonl,
        ordered_unique,
        payload_age_minutes,
        payload_timestamp,
        write_payload,
    )


DEFAULT_OUT_PATH = PROJECT_ROOT / "governance" / "health" / "incident_timeline_latest.json"


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except Exception:
        return int(default)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _discover_event_files(project_root: Path, *, limit_per_pattern: int) -> list[Path]:
    candidates: list[Path] = []
    patterns = (
        "governance/events/auth_events_*.jsonl",
        "governance/events/premarket_token_guard_*.jsonl",
        "governance/events/live_softguard_*.jsonl",
        "governance/events/live_execution_guard_*.jsonl",
        "governance/events/live_macro_events_*.jsonl",
        "governance/events/nightly_resilience_*.jsonl",
        "governance/events/write_failures_*.jsonl",
        "governance/watchdog/*events*.jsonl",
    )
    for pattern in patterns:
        matches = sorted(project_root.glob(pattern))
        candidates.extend(matches[-max(int(limit_per_pattern), 1) :])
    unique = sorted({path.resolve() for path in candidates}, reverse=True)
    return [Path(path) for path in unique]


def _category_for_path(path: Path) -> str:
    name = path.name.lower()
    if "auth" in name or "token" in name:
        return "auth_lease"
    if "failover" in name or "recovery" in name or "resilience" in name:
        return "recovery"
    if "softguard" in name or "killswitch" in name or "halt" in name:
        return "risk_halt"
    if "write_failures" in name or "schema" in name:
        return "data_plane"
    if "macro" in name:
        return "macro"
    if "execution" in name:
        return "execution"
    return "operations"


def _severity_for_event(category: str, row: dict[str, Any]) -> str:
    explicit = str(row.get("severity") or row.get("level") or "").strip().lower()
    if explicit in {"info", "warning", "warn", "critical"}:
        return "warning" if explicit == "warn" else explicit

    state_bits = " ".join(
        str(row.get(key) or "")
        for key in ("status", "overall_status", "state", "reason", "event", "message", "summary")
    ).lower()
    if category == "risk_halt":
        return "critical"
    if any(token in state_bits for token in ("critical", "killswitch", "halt", "blocked", "liquidation")):
        return "critical"
    if bool(row.get("ok")) is False or any(token in state_bits for token in ("warning", "degraded", "retry", "fail")):
        return "warning"
    return "info"


def _summary_for_event(path: Path, row: dict[str, Any]) -> str:
    for key in ("summary", "message", "reason", "event", "status", "overall_status", "state"):
        text = str(row.get(key) or "").strip()
        if text:
            return text
    return path.stem


def _source_rel(project_root: Path, path: Path) -> str:
    try:
        return str(path.resolve().relative_to(project_root.resolve()))
    except Exception:
        return str(path.resolve())


def _recent_incidents(project_root: Path, *, files_per_pattern: int, rows_per_file: int) -> list[dict[str, Any]]:
    incidents: list[dict[str, Any]] = []
    for path in _discover_event_files(project_root, limit_per_pattern=files_per_pattern):
        rows = load_recent_jsonl(path, limit=max(int(rows_per_file), 1))
        category = _category_for_path(path)
        for row in rows:
            ts = payload_timestamp(row, path)
            if ts is None:
                continue
            incidents.append(
                {
                    "timestamp_utc": ts.isoformat(),
                    "category": category,
                    "severity": _severity_for_event(category, row),
                    "summary": _summary_for_event(path, row),
                    "source_rel": _source_rel(project_root, path),
                }
            )
    incidents.sort(key=lambda item: str(item.get("timestamp_utc") or ""), reverse=True)
    return incidents


def _open_surface_incidents(project_root: Path) -> dict[str, list[dict[str, Any]]]:
    health_root = project_root / "governance" / "health"
    walk_root = project_root / "governance" / "walk_forward"
    live_readiness_path = health_root / "live_readiness_smoke_latest.json"
    runtime_path = health_root / "live_runtime_separation_control_latest.json"
    auth_path = health_root / "auth_lease_manager_latest.json"
    coverage_path = walk_root / "coverage_seed_latest.json"
    coverage_gap_closer_path = walk_root / "coverage_gap_closer_latest.json"
    watchdog_path = health_root / "process_watchdog_latest.json"
    storage_path = health_root / "ingestion_storage_control_latest.json"

    live_readiness = load_json(live_readiness_path)
    runtime = load_json(runtime_path)
    auth = load_json(auth_path)
    coverage = load_json(coverage_path)
    coverage_gap_closer = load_json(coverage_gap_closer_path)
    watchdog = load_json(watchdog_path)
    storage = load_json(storage_path)

    incidents: list[dict[str, Any]] = []
    watch_surfaces: list[dict[str, Any]] = []
    surfaces = (
        ("live_readiness", live_readiness, live_readiness_path, "execution"),
        ("runtime_separation", runtime, runtime_path, "operations"),
        ("auth_lease", auth, auth_path, "auth_lease"),
        ("coverage_seed", coverage, coverage_path, "coverage"),
    )
    for surface, payload, path, category in surfaces:
        status = str(payload.get("overall_status") or "").strip().lower()
        if status not in {"blocked", "critical", "degraded", "needs_coverage", "warning"}:
            continue
        row = {
            "surface": surface,
            "status": status,
            "category": category,
            "severity": "critical" if status in {"blocked", "critical"} else "warning",
            "summary": str(payload.get("reason") or status or surface),
            "age_minutes": round(float(payload_age_minutes(payload, path) or 0.0), 3),
            "thread_id": surface,
        }
        watch_reason = ""
        if surface == "runtime_separation":
            clearance_state = str(((payload.get("clearance_plan") or {}).get("clearance_state") or "")).strip().lower()
            if status in {"degraded", "warning", "needs_attention"} and clearance_state in {
                "awaiting_cold_lane",
                "awaiting_coverage_cycles",
                "staged_preclearance",
                "coverage_cycles_ready",
                "off_hours_cold_lane_launch_ready",
                "scheduled_off_hours_launch",
            }:
                row["summary"] = f"runtime preclearance: {clearance_state}"
                row["thread_id"] = "runtime_preclearance"
                watch_reason = "planned_runtime_release_window"
        elif surface == "coverage_seed":
            shortfall = _safe_int(payload.get("coverage_shortfall_bots"), 0)
            seed_queue_size = len(payload.get("seed_queue") if isinstance(payload.get("seed_queue"), list) else [])
            gap_status = str(coverage_gap_closer.get("overall_status") or "").strip().lower()
            if status == "needs_coverage" and shortfall > 0 and (seed_queue_size > 0 or gap_status in {"needs_cycles", "ready", "degraded"}):
                row["summary"] = f"coverage preclearance: shortfall_bots={shortfall}"
                row["thread_id"] = "coverage_preclearance"
                watch_reason = "planned_walk_forward_seed_gap"
        elif surface == "auth_lease":
            lease_state = str(payload.get("lease_state") or "").strip().lower()
            lease_budget = payload.get("lease_budget") if isinstance(payload.get("lease_budget"), dict) else {}
            expires_in_seconds = _safe_float(lease_budget.get("expires_in_seconds"), 0.0)
            critical_lease_seconds = _safe_float(lease_budget.get("critical_lease_seconds"), 0.0)
            if (
                status in {"degraded", "warning"}
                and lease_state == "warning"
                and expires_in_seconds > critical_lease_seconds > 0.0
            ):
                row["summary"] = f"auth refresh watch: expires_in_seconds={round(expires_in_seconds, 3)}"
                row["thread_id"] = "auth_refresh_watch"
                watch_reason = "bounded_auth_refresh_window"
        if watch_reason:
            row["watch_reason"] = watch_reason
            watch_surfaces.append(row)
        elif surface == "live_readiness" and bool(payload.get("canary_control", {}).get("bounded_runtime_preclearance", False)):
            row["summary"] = "live release window is precleared but still waiting on the bounded canary/runtime handoff"
            row["thread_id"] = "live_release_window"
            row["watch_reason"] = "bounded_release_window"
            watch_surfaces.append(row)
        elif surface == "live_readiness" and bool(payload.get("process_watchdog", {}).get("bounded_paper_lane_watchdog", False)):
            row["summary"] = "live readiness is waiting on bounded paper-lane watchdog pressure under active storage recovery"
            row["thread_id"] = "live_release_window"
            row["watch_reason"] = "bounded_release_window"
            watch_surfaces.append(row)
        else:
            incidents.append(row)

    restart_storms = watchdog.get("restart_storms") if isinstance(watchdog.get("restart_storms"), list) else []
    alerts = watchdog.get("alerts") if isinstance(watchdog.get("alerts"), list) else []
    storage_status = str(storage.get("overall_status") or "").strip().lower()
    storage_recovery_state = str(storage.get("recovery_state") or "").strip().lower()
    storage_pressure_index = _safe_float(storage.get("pressure_index"), 0.0)
    storage_bounded_recovery = (
        storage.get("bounded_recovery_contract")
        if isinstance(storage.get("bounded_recovery_contract"), dict)
        else {}
    )
    storage_drain_follow_status = str(storage_bounded_recovery.get("drain_follow_through_status") or "").strip().lower()
    storage_drain_contract_active = bool(
        storage_bounded_recovery.get("active", False)
        or storage_bounded_recovery.get("quality_ready", False)
        or storage_bounded_recovery.get("active_drain_progress", False)
        or storage_drain_follow_status
        in {"handoff_requested", "drain_active", "writer_handoff_active", "requested_live_writer"}
    )
    derived_paper_lane_watchdog = bool(
        (restart_storms or alerts)
        and storage_status in {"blocked", "degraded"}
        and storage_recovery_state in {"blocked_backpressure", "recovering_under_guard", "stabilized_recovery"}
        and (storage_pressure_index <= 6.5 or storage_drain_contract_active)
        and all(str((row or {}).get("name") or "").strip().lower() == "execution_lane_paper" for row in restart_storms if isinstance(row, dict))
        and all(str((row or {}).get("name") or "").strip().lower() == "execution_lane_paper" for row in alerts if isinstance(row, dict))
    )
    if restart_storms or alerts:
        row = {
            "surface": "process_watchdog",
            "status": "degraded",
            "category": "operations",
            "severity": "warning",
            "summary": f"restart_storms={len(restart_storms)} alerts={len(alerts)}",
            "age_minutes": round(float(payload_age_minutes(watchdog, watchdog_path) or 0.0), 3),
            "thread_id": "process_watchdog",
        }
        if derived_paper_lane_watchdog:
            row["summary"] = "paper execution lane watchdog pressure is being absorbed inside bounded storage recovery"
            row["watch_reason"] = "derived_storage_backpressure"
            watch_surfaces.append(row)
        else:
            incidents.append(row)
    return {
        "open_surfaces": incidents,
        "watch_surfaces": watch_surfaces,
    }


def _stitched_threads(recent_incidents: list[dict[str, Any]], open_surfaces: list[dict[str, Any]], watch_surfaces: list[dict[str, Any]]) -> list[dict[str, Any]]:
    buckets: dict[str, dict[str, Any]] = {}

    def touch(thread_id: str, *, category: str, severity: str, recent: int = 0, open_count: int = 0, watch_count: int = 0) -> None:
        row = buckets.setdefault(
            thread_id,
            {
                "thread_id": thread_id,
                "category": category,
                "severity": severity,
                "recent_event_count": 0,
                "open_surface_count": 0,
                "watch_surface_count": 0,
            },
        )
        row["recent_event_count"] += recent
        row["open_surface_count"] += open_count
        row["watch_surface_count"] += watch_count
        if severity == "critical" or row["severity"] != "critical":
            row["severity"] = severity

    for row in recent_incidents:
        category = str(row.get("category") or "operations")
        touch(f"recent:{category}", category=category, severity=str(row.get("severity") or "info"), recent=1)
    for row in open_surfaces:
        thread_id = str(row.get("thread_id") or row.get("surface") or "operations")
        touch(thread_id, category=str(row.get("category") or "operations"), severity=str(row.get("severity") or "warning"), open_count=1)
    for row in watch_surfaces:
        thread_id = str(row.get("thread_id") or row.get("surface") or "operations")
        touch(thread_id, category=str(row.get("category") or "operations"), severity=str(row.get("severity") or "warning"), watch_count=1)

    stitched = sorted(
        buckets.values(),
        key=lambda row: (
            0 if str(row.get("severity") or "") == "critical" else 1,
            -(int(row.get("open_surface_count") or 0)),
            -(int(row.get("watch_surface_count") or 0)),
            str(row.get("thread_id") or ""),
        ),
    )
    return stitched


def build_payload(
    project_root: Path = PROJECT_ROOT,
    *,
    files_per_pattern: int = 2,
    rows_per_file: int = 25,
    recent_limit: int = 20,
) -> dict[str, Any]:
    recent_incidents = _recent_incidents(
        project_root,
        files_per_pattern=max(int(files_per_pattern), 1),
        rows_per_file=max(int(rows_per_file), 1),
    )[: max(int(recent_limit), 1)]
    surface_rollup = _open_surface_incidents(project_root)
    open_surfaces = list(surface_rollup.get("open_surfaces") or [])
    watch_surfaces = list(surface_rollup.get("watch_surfaces") or [])
    stitched_threads = _stitched_threads(recent_incidents, open_surfaces, watch_surfaces)

    severity_counts = Counter(str(row.get("severity") or "info") for row in recent_incidents)
    category_counts = Counter(str(row.get("category") or "operations") for row in recent_incidents)
    open_severity_counts = Counter(str(row.get("severity") or "warning") for row in open_surfaces)

    overall_status = "ready"
    if open_severity_counts.get("critical", 0) > 0:
        overall_status = "blocked"
    elif open_surfaces or watch_surfaces or severity_counts.get("warning", 0) > 0 or severity_counts.get("critical", 0) > 0:
        overall_status = "degraded"

    review_required = len(open_surfaces) > 0
    closure_ready = len(open_surfaces) == 0
    auto_close_contract = {
        "closure_ready": closure_ready,
        "candidate_count": (1 if closure_ready else 0),
        "review_required": review_required,
        "closure_reason": ("watch_only_or_open_surfaces_cleared" if closure_ready else "open_surfaces_present"),
    }

    recommended_actions = ordered_unique(
        [
            "pause risky or write-heavy lanes until the auth lease and runtime separation surfaces clear" if any(str(row.get("category") or "") == "auth_lease" for row in open_surfaces) else "",
            "treat live softguard or killswitch incidents as freeze events first, then repair feeds or coverage debt before resuming" if any(str(row.get("category") or "") == "risk_halt" for row in recent_incidents) else "",
            "use the incident timeline as the single review surface for watchdog, auth, and failover interventions" if recent_incidents else "",
            "treat bounded runtime, auth, and coverage preclearance states as watch items instead of active incidents" if watch_surfaces else "",
            "auto-close this incident packet once the open surfaces stay clear and the review hash is archived" if closure_ready else "",
        ]
    )

    return {
        "timestamp_utc": iso_now(),
        "schema_version": 1,
        "ok": overall_status == "ready",
        "overall_status": overall_status,
        "recent_incident_count": len(recent_incidents),
        "open_incident_count": len(open_surfaces),
        "watch_surface_count": len(watch_surfaces),
        "review_required": review_required,
        "incident_counts": {
            "by_severity": dict(severity_counts),
            "by_category": dict(category_counts),
            "open_by_severity": dict(open_severity_counts),
        },
        "intervention_counts": {
            "recovery_events": int(category_counts.get("recovery", 0)),
            "risk_halt_events": int(category_counts.get("risk_halt", 0)),
            "auth_events": int(category_counts.get("auth_lease", 0)),
            "macro_events": int(category_counts.get("macro", 0)),
        },
        "auto_close_contract": auto_close_contract,
        "open_surfaces": open_surfaces,
        "watch_surfaces": watch_surfaces,
        "stitched_threads": stitched_threads,
        "recent_incidents": recent_incidents,
        "recommended_actions": recommended_actions,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Publish a machine-readable incident timeline across events and watchdog surfaces.")
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--files-per-pattern", type=int, default=2)
    parser.add_argument("--rows-per-file", type=int, default=25)
    parser.add_argument("--recent-limit", type=int, default=20)
    parser.add_argument("--out-file", default=str(DEFAULT_OUT_PATH))
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    payload = build_payload(
        Path(args.project_root).resolve(),
        files_per_pattern=int(args.files_per_pattern),
        rows_per_file=int(args.rows_per_file),
        recent_limit=int(args.recent_limit),
    )
    out_path = Path(args.out_file).expanduser()
    write_payload(out_path, payload)
    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        print(
            "incident_timeline "
            f"overall_status={payload.get('overall_status', '')} "
            f"recent_incidents={int(payload.get('recent_incident_count', 0) or 0)} "
            f"open_surfaces={int(payload.get('open_incident_count', 0) or 0)}"
        )
    return 0 if payload.get("overall_status") in {"ready", "degraded"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
