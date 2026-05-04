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
    from scripts.ops.long_runtime_common import PROJECT_ROOT, iso_now, load_json, ordered_unique, payload_age_minutes, write_payload
else:
    from .long_runtime_common import PROJECT_ROOT, iso_now, load_json, ordered_unique, payload_age_minutes, write_payload


DEFAULT_OUT_PATH = PROJECT_ROOT / "governance" / "health" / "rolling_restart_controller_latest.json"


def _latest_shadow_age_minutes(project_root: Path) -> float | None:
    latest_age: float | None = None
    health_root = project_root / "governance" / "health"
    for path in health_root.glob("shadow_loop_*.json"):
        payload = load_json(path)
        age = payload_age_minutes(payload, path)
        if age is None:
            continue
        if latest_age is None or age < latest_age:
            latest_age = age
    return latest_age


def build_payload(
    project_root: Path = PROJECT_ROOT,
    *,
    max_session_age_minutes: float = 30.0,
    swap_restart_gb: float = 20.0,
    max_shadow_age_minutes: float = 120.0,
) -> dict[str, Any]:
    health_root = project_root / "governance" / "health"
    process_watchdog = load_json(health_root / "process_watchdog_latest.json")
    resource_guard = load_json(health_root / "resource_guard_latest.json")
    session_ready_path = health_root / "session_ready_latest.json"
    session_ready = load_json(session_ready_path)
    snapshot_drill_path = project_root / "exports" / "state_snapshot_drills" / "latest.json"
    snapshot_drill = load_json(snapshot_drill_path)

    session_age_minutes = payload_age_minutes(session_ready, session_ready_path)
    shadow_age_minutes = _latest_shadow_age_minutes(project_root)
    snapshot_age_minutes = payload_age_minutes(snapshot_drill, snapshot_drill_path)
    swap_used_gb = float(resource_guard.get("swap_used_gb", 0.0) or 0.0)
    memory_state = str(resource_guard.get("memory_pressure_state") or "")
    restart_storms = len(process_watchdog.get("restart_storms") or [])
    checkpoint_fresh = snapshot_age_minutes is not None and float(snapshot_age_minutes) <= 24.0 * 60.0

    due_signals = {
        "session_stale": session_age_minutes is not None and float(session_age_minutes) > float(max_session_age_minutes),
        "shadow_heartbeat_stale": shadow_age_minutes is not None and float(shadow_age_minutes) > float(max_shadow_age_minutes),
        "swap_pressure_high": swap_used_gb >= float(swap_restart_gb),
        "restart_storm_present": restart_storms > 0,
        "checkpoint_missing_or_stale": not checkpoint_fresh,
    }
    due = any(due_signals.values())
    recommended_scope = "none"
    if due_signals["restart_storm_present"] or due_signals["session_stale"]:
        recommended_scope = "full_stack"
    elif due_signals["swap_pressure_high"] or due_signals["shadow_heartbeat_stale"]:
        recommended_scope = "worker_only"

    overall_status = "ready"
    if due and not checkpoint_fresh:
        overall_status = "blocked"
    elif due:
        overall_status = "degraded"

    recommended_actions = ordered_unique(
        [
            "run a checkpoint bundle before the next controlled restart window" if not checkpoint_fresh else "",
            "recycle worker-side processes in a rolling window to relieve swap buildup" if due_signals["swap_pressure_high"] else "",
            "schedule a full-stack restart after sanity checks if runtime heartbeats stay stale" if due_signals["session_stale"] or due_signals["restart_storm_present"] else "",
            "keep restart windows off-hours and per-sleeve so the runtime never hard-stops all lanes at once" if due else "",
        ]
    )

    return {
        "timestamp_utc": iso_now(),
        "schema_version": 1,
        "ok": overall_status == "ready",
        "overall_status": overall_status,
        "restart_due": bool(due),
        "recommended_scope": recommended_scope,
        "runtime_signals": {
            "session_ready_age_minutes": round(float(session_age_minutes), 4) if session_age_minutes is not None else None,
            "latest_shadow_age_minutes": round(float(shadow_age_minutes), 4) if shadow_age_minutes is not None else None,
            "swap_used_gb": round(float(swap_used_gb), 3),
            "memory_pressure_state": memory_state,
            "restart_storms": restart_storms,
            "checkpoint_age_minutes": round(float(snapshot_age_minutes), 4) if snapshot_age_minutes is not None else None,
        },
        "checkpoint_resume": {
            "checkpoint_fresh": bool(checkpoint_fresh),
            "state_snapshot_files_checked": int(snapshot_drill.get("files_checked", 0) or 0),
            "missing_files": snapshot_drill.get("missing_files") if isinstance(snapshot_drill.get("missing_files"), list) else [],
        },
        "restart_windows": {
            "preferred_window_local": "03:00-05:00 America/New_York",
            "restart_scope_ladder": ["worker_only", "sleeve_group", "full_stack"],
            "infra_bots": ["rolling_restart_controller", "restart_sanity_bundle", "daily_state_snapshot_drill"],
        },
        "due_signals": due_signals,
        "recommended_actions": recommended_actions,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Plan rolling restart windows with checkpoint and resume prerequisites.")
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--max-session-age-minutes", type=float, default=30.0)
    parser.add_argument("--swap-restart-gb", type=float, default=20.0)
    parser.add_argument("--max-shadow-age-minutes", type=float, default=120.0)
    parser.add_argument("--out-file", default=str(DEFAULT_OUT_PATH))
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    payload = build_payload(
        Path(args.project_root).resolve(),
        max_session_age_minutes=float(args.max_session_age_minutes),
        swap_restart_gb=float(args.swap_restart_gb),
        max_shadow_age_minutes=float(args.max_shadow_age_minutes),
    )
    out_path = Path(args.out_file).expanduser()
    write_payload(out_path, payload)
    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        print(
            "rolling_restart_controller "
            f"overall_status={payload.get('overall_status', '')} "
            f"recommended_scope={payload.get('recommended_scope', '')}"
        )
    return 0 if payload.get("overall_status") in {"ready", "degraded"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
