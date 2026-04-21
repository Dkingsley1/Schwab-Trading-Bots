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


def _open_surface_incidents(project_root: Path) -> list[dict[str, Any]]:
    health_root = project_root / "governance" / "health"
    live_readiness_path = health_root / "live_readiness_smoke_latest.json"
    runtime_path = health_root / "live_runtime_separation_control_latest.json"
    auth_path = health_root / "auth_lease_manager_latest.json"
    coverage_path = project_root / "governance" / "walk_forward" / "coverage_seed_latest.json"
    watchdog_path = health_root / "process_watchdog_latest.json"

    live_readiness = load_json(live_readiness_path)
    runtime = load_json(runtime_path)
    auth = load_json(auth_path)
    coverage = load_json(coverage_path)
    watchdog = load_json(watchdog_path)

    incidents: list[dict[str, Any]] = []
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
        incidents.append(
            {
                "surface": surface,
                "status": status,
                "category": category,
                "severity": "critical" if status in {"blocked", "critical"} else "warning",
                "summary": str(payload.get("reason") or status or surface),
                "age_minutes": round(float(payload_age_minutes(payload, path) or 0.0), 3),
            }
        )

    restart_storms = watchdog.get("restart_storms") if isinstance(watchdog.get("restart_storms"), list) else []
    alerts = watchdog.get("alerts") if isinstance(watchdog.get("alerts"), list) else []
    if restart_storms or alerts:
        incidents.append(
            {
                "surface": "process_watchdog",
                "status": "degraded",
                "category": "operations",
                "severity": "warning",
                "summary": f"restart_storms={len(restart_storms)} alerts={len(alerts)}",
                "age_minutes": round(float(payload_age_minutes(watchdog, watchdog_path) or 0.0), 3),
            }
        )
    return incidents


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
    open_surfaces = _open_surface_incidents(project_root)

    severity_counts = Counter(str(row.get("severity") or "info") for row in recent_incidents)
    category_counts = Counter(str(row.get("category") or "operations") for row in recent_incidents)
    open_severity_counts = Counter(str(row.get("severity") or "warning") for row in open_surfaces)

    overall_status = "ready"
    if open_severity_counts.get("critical", 0) > 0:
        overall_status = "blocked"
    elif open_surfaces or severity_counts.get("warning", 0) > 0 or severity_counts.get("critical", 0) > 0:
        overall_status = "degraded"

    closure_ready = len(open_surfaces) == 0 and len(recent_incidents) > 0 and overall_status != "blocked"
    auto_close_contract = {
        "closure_ready": closure_ready,
        "candidate_count": (1 if closure_ready else 0),
        "review_required": len(open_surfaces) > 0,
        "closure_reason": ("all_open_surfaces_cleared" if closure_ready else "open_surfaces_present_or_no_recent_incidents"),
    }

    recommended_actions = ordered_unique(
        [
            "pause risky or write-heavy lanes until the auth lease and runtime separation surfaces clear" if any(str(row.get("category") or "") == "auth_lease" for row in open_surfaces) else "",
            "treat live softguard or killswitch incidents as freeze events first, then repair feeds or coverage debt before resuming" if any(str(row.get("category") or "") == "risk_halt" for row in recent_incidents) else "",
            "use the incident timeline as the single review surface for watchdog, auth, and failover interventions" if recent_incidents else "",
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
