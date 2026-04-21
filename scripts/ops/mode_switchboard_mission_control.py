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
    from scripts.ops.long_runtime_common import PROJECT_ROOT, iso_now, load_json, ordered_unique, write_payload
else:
    from .long_runtime_common import PROJECT_ROOT, iso_now, load_json, ordered_unique, write_payload


DEFAULT_OUT_PATH = PROJECT_ROOT / "governance" / "health" / "mode_switchboard_mission_control_latest.json"


def _process_names(process_watchdog: dict[str, Any]) -> list[str]:
    rows = process_watchdog.get("status") if isinstance(process_watchdog.get("status"), list) else []
    return [str((row or {}).get("name") or "").strip().lower() for row in rows if isinstance(row, dict)]


def _mode_row(name: str, *, active: bool, reason: str, ready: bool) -> dict[str, Any]:
    state = "active" if active else ("ready" if ready else "idle")
    return {
        "mode": name,
        "state": state,
        "active": bool(active),
        "ready": bool(ready),
        "reason": reason,
    }


def build_payload(project_root: Path = PROJECT_ROOT) -> dict[str, Any]:
    health_root = project_root / "governance" / "health"
    live_readiness = load_json(health_root / "live_readiness_smoke_latest.json")
    runtime = load_json(health_root / "live_runtime_separation_control_latest.json")
    portable = load_json(health_root / "portable_brain_contract_latest.json")
    access_mode = load_json(health_root / "runtime_access_mode_latest.json")
    process_watchdog = load_json(health_root / "process_watchdog_latest.json")

    names = _process_names(process_watchdog)
    shadow_active = any(token in name for name in names for token in ("shadow", "watchdog", "all_sleeves"))
    paper_active = bool(live_readiness.get("paper_lane_fresh", False)) or any("paper" in name for name in names)
    live_active = bool(live_readiness.get("live_lane_running", False))

    shadow_ready = shadow_active or bool(names)
    paper_ready = paper_active or bool(live_readiness.get("paper_lane_fresh", False))
    live_ready = bool(live_readiness.get("broker_ready", False)) and bool(live_readiness.get("session_ready", False))

    modes = [
        _mode_row("shadow", active=shadow_active, ready=shadow_ready, reason="process_watchdog_shadow_lane" if shadow_active else "shadow_lane_not_detected"),
        _mode_row("paper", active=paper_active, ready=paper_ready, reason="paper_lane_fresh_or_running" if paper_active else "paper_lane_not_fresh"),
        _mode_row("live", active=live_active, ready=live_ready, reason="broker_and_session_ready" if live_ready else "live_lane_gated"),
    ]

    overall_status = "ready"
    if not shadow_ready or not paper_ready:
        overall_status = "degraded"
    if str(runtime.get("overall_status") or "").strip().lower() == "blocked":
        overall_status = "blocked"

    live_contract = runtime.get("release_contract") if isinstance(runtime.get("release_contract"), dict) else {}
    host_contract = portable.get("host_contract") if isinstance(portable.get("host_contract"), dict) else {}
    recommended_actions = ordered_unique(
        [
            "keep live in read-only posture while the switchboard still sees runtime separation contention" if bool(live_contract.get("live_lane_should_be_read_only", False)) else "",
            "refresh the paper lane before claiming three-mode continuity" if not paper_ready else "",
            "bring the shadow watchdog back up before treating the switchboard as fully available" if not shadow_ready else "",
        ]
    )

    return {
        "timestamp_utc": iso_now(),
        "schema_version": 1,
        "ok": overall_status == "ready",
        "overall_status": overall_status,
        "modes": modes,
        "mode_counts": {
            "active": sum(1 for row in modes if bool(row.get("active"))),
            "ready": sum(1 for row in modes if bool(row.get("ready"))),
        },
        "control_surface": {
            "runtime_access_mode": str(access_mode.get("mode") or ""),
            "host_profile": str(host_contract.get("host_profile") or ""),
            "clearance_state": str(((runtime.get("clearance_plan") or {}).get("clearance_state") or "")),
            "live_lane_should_be_read_only": bool(live_contract.get("live_lane_should_be_read_only", False)),
            "shared_host_training_resume_allowed": bool(live_contract.get("shared_host_training_resume_allowed", False)),
        },
        "recommended_actions": recommended_actions,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Publish the three-mode switchboard mission-control snapshot.")
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--out-file", default=str(DEFAULT_OUT_PATH))
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    payload = build_payload(Path(args.project_root).resolve())
    out_path = Path(args.out_file).expanduser()
    write_payload(out_path, payload)
    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        print(
            "mode_switchboard_mission_control "
            f"overall_status={payload.get('overall_status', '')} "
            f"active_modes={int(((payload.get('mode_counts') or {}).get('active', 0) or 0))}"
        )
    return 0 if payload.get("overall_status") in {"ready", "degraded"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
