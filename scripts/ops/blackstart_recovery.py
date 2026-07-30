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


DEFAULT_OUT_PATH = PROJECT_ROOT / "governance" / "health" / "blackstart_recovery_latest.json"


def _latest_shadow_loop_age_minutes(health_root: Path) -> float | None:
    latest_age: float | None = None
    for path in health_root.glob("shadow_loop_*.json"):
        payload = load_json(path)
        if not payload:
            continue
        age = payload_age_minutes(payload, path)
        if age is None:
            continue
        latest_age = float(age) if latest_age is None else min(latest_age, float(age))
    launcher_path = health_root / "all_sleeves_launcher_latest.json"
    launcher = load_json(launcher_path)
    if launcher:
        launcher_ready = str(launcher.get("overall_status") or launcher.get("status") or "").strip().lower() in {
            "ready",
            "ok",
            "running",
        }
        launcher_age = payload_age_minutes(launcher, launcher_path)
        if launcher_ready and launcher_age is not None:
            latest_age = float(launcher_age) if latest_age is None else min(latest_age, float(launcher_age))
    return latest_age


def build_payload(project_root: Path = PROJECT_ROOT, *, max_session_age_minutes: float = 30.0) -> dict[str, Any]:
    health_root = project_root / "governance" / "health"
    reboot_resilience = load_json(health_root / "reboot_resilience_latest.json")
    session_ready_path = health_root / "session_ready_latest.json"
    session_ready = load_json(session_ready_path)
    live_readiness_path = health_root / "live_readiness_smoke_latest.json"
    live_readiness = load_json(live_readiness_path)
    storage_resilience = load_json(health_root / "storage_resilience_control_latest.json")
    storage_route = load_json(health_root / "storage_route_status_latest.json")
    auth_lease = load_json(health_root / "auth_lease_manager_latest.json")
    process_watchdog = load_json(health_root / "process_watchdog_latest.json")

    session_age_minutes = payload_age_minutes(session_ready, session_ready_path)
    live_age_minutes = payload_age_minutes(live_readiness, live_readiness_path)
    launchd_ok = bool(reboot_resilience.get("ok", False))
    storage_ok = bool(storage_route.get("ok", storage_resilience.get("ok", False)))
    lease_state = str(auth_lease.get("lease_state") or "").strip().lower()
    broker_state = auth_lease.get("broker_state") if isinstance(auth_lease.get("broker_state"), dict) else {}
    auth_ok = True
    if auth_lease:
        auth_ok = bool(auth_lease.get("ok", False))
        if not auth_ok and lease_state == "warning":
            auth_ok = bool(
                broker_state.get("broker_ready", False)
                and broker_state.get("network_ok", False)
                and broker_state.get("auth_ok", False)
            )
    restart_storm_count = len(process_watchdog.get("restart_storms") or []) if isinstance(process_watchdog.get("restart_storms"), list) else 0
    restart_ok = restart_storm_count <= 0
    session_payload_ok = bool(session_ready.get("ok", session_ready.get("ready", False)))
    shadow_loop_age_minutes = _latest_shadow_loop_age_minutes(health_root)
    session_freshness_inferred_from_shadow_loop = bool(
        session_payload_ok
        and shadow_loop_age_minutes is not None
        and float(shadow_loop_age_minutes) <= float(max_session_age_minutes)
    )
    session_ok = session_payload_ok and (
        session_age_minutes is None
        or float(session_age_minutes) <= float(max_session_age_minutes)
        or session_freshness_inferred_from_shadow_loop
    )
    live_ok = bool(live_readiness.get("ok", False)) and (
        live_age_minutes is None or float(live_age_minutes) <= 24.0 * 60.0
    )

    stages = [
        {"name": "launchd_recovery", "ok": launchd_ok, "command": "./scripts/ops/opsctl.sh restart-sanity --json", "auto_recoverable": True},
        {"name": "storage_mount", "ok": storage_ok, "command": "./scripts/ops/opsctl.sh storage-resilience --json", "auto_recoverable": True},
        {"name": "auth_lease", "ok": auth_ok, "command": "./scripts/ops/opsctl.sh token-refresh --json", "auto_recoverable": True},
        {
            "name": "session_ready",
            "ok": session_ok,
            "command": "./scripts/session_ready_check.py",
            "auto_recoverable": False,
            "freshness_inferred_from_shadow_loop": session_freshness_inferred_from_shadow_loop,
        },
        {"name": "live_readiness", "ok": live_ok, "command": "./scripts/ops/opsctl.sh start --force-restart", "auto_recoverable": False},
        {"name": "restart_sanity", "ok": restart_ok, "command": "./scripts/ops/opsctl.sh status --json", "auto_recoverable": True},
    ]

    overall_status = "ready"
    if not launchd_ok or not storage_ok or not auth_ok or not restart_ok:
        overall_status = "blocked"
    elif not session_ok or not live_ok:
        overall_status = "degraded"
    blocked_stage_count = sum(1 for row in stages if not bool(row.get("ok", False)))
    auto_recoverable_stage_count = sum(1 for row in stages if (not bool(row.get("ok", False))) and bool(row.get("auto_recoverable", False)))
    reboot_resilience_score = 100.0 - (blocked_stage_count * 14.0)
    if restart_storm_count > 0:
        reboot_resilience_score -= 8.0
    reboot_resilience_score = max(round(reboot_resilience_score, 2), 0.0)
    production_grade_ready = all(bool(row.get("ok", False)) for row in stages)

    recommended_actions = ordered_unique(
        [
            "recover launchd labels before starting sleeves after a reboot" if not launchd_ok else "",
            "remount or fail back storage before reopening the runtime lane" if not storage_ok else "",
            "refresh the broker auth lease before resuming live loops" if not auth_ok else "",
            "clear process restart storms before trusting a post-reboot runtime handoff" if not restart_ok else "",
            "run restart sanity and live readiness after black-start sequencing completes" if overall_status != "ready" else "",
        ]
    )

    return {
        "timestamp_utc": iso_now(),
        "schema_version": 1,
        "ok": overall_status == "ready",
        "overall_status": overall_status,
        "startup_order": [row["name"] for row in stages],
        "stages": stages,
        "blocked_stage_count": blocked_stage_count,
        "auto_recoverable_stage_count": auto_recoverable_stage_count,
        "reboot_resilience_score": reboot_resilience_score,
        "production_grade_ready": production_grade_ready,
        "evidence": {
            "session_ready_age_minutes": round(float(session_age_minutes), 4) if session_age_minutes is not None else None,
            "latest_shadow_loop_age_minutes": (
                round(float(shadow_loop_age_minutes), 4) if shadow_loop_age_minutes is not None else None
            ),
            "session_freshness_inferred_from_shadow_loop": session_freshness_inferred_from_shadow_loop,
            "live_readiness_age_minutes": round(float(live_age_minutes), 4) if live_age_minutes is not None else None,
            "storage_mode": str(storage_route.get("mode") or ""),
            "recovered_labels": len(reboot_resilience.get("recovered") or []),
            "restart_storm_count": restart_storm_count,
            "lease_state": str(auth_lease.get("lease_state") or ""),
        },
        "recovery_contract": {
            "launchd_recovery_ready": launchd_ok,
            "storage_ready": storage_ok,
            "auth_ready": auth_ok,
            "session_ready": session_ok,
            "session_freshness_inferred_from_shadow_loop": session_freshness_inferred_from_shadow_loop,
            "restart_sanity_ready": restart_ok,
            "recommended_stage_commands": [str(row.get("command") or "") for row in stages if not bool(row.get("ok", False))][:4],
        },
        "infra_bots": ["blackstart_recovery", "reboot_resilience_guard", "restart_sanity_bundle", "live_readiness_smoke"],
        "recommended_actions": recommended_actions,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Coordinate black-start recovery ordering after reboot or cold restart.")
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--max-session-age-minutes", type=float, default=30.0)
    parser.add_argument("--out-file", default=str(DEFAULT_OUT_PATH))
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    payload = build_payload(Path(args.project_root).resolve(), max_session_age_minutes=float(args.max_session_age_minutes))
    out_path = Path(args.out_file).expanduser()
    write_payload(out_path, payload)
    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        print(
            "blackstart_recovery "
            f"overall_status={payload.get('overall_status', '')} "
            f"stages={len(payload.get('stages') or [])}"
        )
    return 0 if payload.get("overall_status") in {"ready", "degraded"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
