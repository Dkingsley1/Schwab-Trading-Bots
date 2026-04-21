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
else:
    from .long_runtime_common import PROJECT_ROOT, iso_now, load_json, ordered_unique, parse_iso_utc, utc_now, write_payload


DEFAULT_OUT_PATH = PROJECT_ROOT / "governance" / "health" / "chaos_drill_coordinator_latest.json"
DEFAULT_STATE_PATH = PROJECT_ROOT / "governance" / "runtime" / "chaos_drill_state.json"


def _load_state(path: Path) -> dict[str, Any]:
    payload = load_json(path)
    return payload if isinstance(payload.get("drills"), dict) else {"drills": {}}


def _save_state(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


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

    default_sources = {
        "snapshot_restore": snapshot_drill.get("timestamp_utc"),
        "reboot_blackstart": reboot_resilience.get("timestamp_utc"),
        "storage_failover": storage_resilience.get("timestamp_utc"),
        "auth_expiry": token_guard.get("timestamp_utc"),
        "queue_backlog_surge": process_watchdog.get("timestamp_utc"),
        "sql_writer_stall": process_watchdog.get("timestamp_utc"),
    }

    overdue: list[dict[str, Any]] = []
    drills: list[dict[str, Any]] = []
    cutoff_days = float(overdue_days)
    for drill_name, source_ts in default_sources.items():
        recorded = ((state.get("drills") or {}).get(drill_name)) if isinstance(state.get("drills"), dict) else None
        ts = parse_iso_utc((recorded or {}).get("completed_at_utc")) if isinstance(recorded, dict) else None
        if ts is None:
            ts = parse_iso_utc(source_ts)
        age_days = None
        if ts is not None:
            age_days = max((utc_now() - ts).total_seconds() / 86400.0, 0.0)
        is_overdue = age_days is None or float(age_days) > cutoff_days
        row = {
            "drill": drill_name,
            "completed_at_utc": ts.isoformat() if ts is not None else "",
            "age_days": round(float(age_days), 4) if age_days is not None else None,
            "overdue": bool(is_overdue),
        }
        drills.append(row)
        if is_overdue:
            overdue.append(row)

    overall_status = "ready"
    if len(overdue) >= 2:
        overall_status = "blocked"
    elif overdue:
        overall_status = "degraded"

    recommended_actions = ordered_unique(
        [
            "record the next storage failover and black-start rehearsal in the chaos drill state file" if overdue else "",
            "exercise auth-expiry, SQL writer stall, and backlog surge scenarios weekly during the long-run window" if len(overdue) >= 1 else "",
        ]
    )
    program_score = max(0.0, round(100.0 - (18.0 * len(overdue)), 2))
    next_priority_drill = str((overdue[0] or {}).get("drill") or "") if overdue else ""

    return {
        "timestamp_utc": iso_now(),
        "schema_version": 1,
        "ok": overall_status == "ready",
        "overall_status": overall_status,
        "overdue_days_threshold": cutoff_days,
        "drills": drills,
        "overdue_drills": overdue,
        "drill_program": {
            "program_score": program_score,
            "next_priority_drill": next_priority_drill,
            "weekly_cadence_target_days": cutoff_days,
            "automation_ready": True,
        },
        "state_path": str(state_path),
        "infra_bots": ["chaos_drill_coordinator", "daily_state_snapshot_drill", "reboot_resilience_guard", "process_watchdog"],
        "recommended_actions": recommended_actions,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Coordinate weekly chaos drills and record drill completions.")
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--state-path", default=str(DEFAULT_STATE_PATH))
    parser.add_argument("--overdue-days", type=float, default=7.0)
    parser.add_argument("--record-drill", default="")
    parser.add_argument("--note", default="")
    parser.add_argument("--out-file", default=str(DEFAULT_OUT_PATH))
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    state_path = Path(args.state_path).expanduser()
    if str(args.record_drill or "").strip():
        state = _load_state(state_path)
        drills = state.get("drills") if isinstance(state.get("drills"), dict) else {}
        drills[str(args.record_drill).strip()] = {"completed_at_utc": iso_now(), "note": str(args.note or "")}
        state["drills"] = drills
        _save_state(state_path, state)

    payload = build_payload(Path(args.project_root).resolve(), state_path=state_path, overdue_days=float(args.overdue_days))
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
    return 0 if payload.get("overall_status") in {"ready", "degraded"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
