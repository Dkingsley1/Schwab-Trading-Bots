#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _payload_age_seconds(payload: dict[str, Any]) -> float | None:
    text = str(payload.get("timestamp_utc") or "").strip()
    if not text:
        return None
    try:
        dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except Exception:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return max((datetime.now(timezone.utc) - dt.astimezone(timezone.utc)).total_seconds(), 0.0)


def _ordered_unique(values: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        text = str(value or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Live-readiness smoke test for broker/auth/execution prerequisites.")
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--allow-live-broker-submit", action="store_true", help="Only when explicitly enabled should a real broker submit path be attempted.")
    parser.add_argument("--out-file", default=str(PROJECT_ROOT / "governance" / "health" / "live_readiness_smoke_latest.json"))
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    root = Path(args.project_root).resolve()
    broker = _load(root / "governance" / "health" / "broker_readiness_latest.json")
    token_guard = _load(root / "governance" / "health" / "premarket_token_guard_latest.json")
    session = _load(root / "governance" / "health" / "session_ready_latest.json")
    paper_lane = _load(root / "governance" / "health" / "execution_lane_paper_latest.json")
    live_lane = _load(root / "governance" / "health" / "execution_lane_live_latest.json")
    storage = _load(root / "governance" / "health" / "storage_route_status_latest.json")
    resource_guard = _load(root / "governance" / "health" / "resource_guard_latest.json")
    watchdog = _load(root / "governance" / "health" / "process_watchdog_latest.json")

    paper_lane_fresh = not bool(paper_lane.get("stale", False)) if paper_lane else False
    live_lane_running = bool(live_lane) and not bool(live_lane.get("stale", False))
    broker_ready = bool(broker.get("ready_for_open", False))
    session_ready = bool(session.get("ready", session.get("ok", False)))
    storage_ok = bool(storage.get("ok", True))
    network_ok = bool(broker.get("network_ok", token_guard.get("network", {}).get("ok", False)))
    auth_ok = bool(broker.get("auth_ok", token_guard.get("auth", {}).get("ok", False)))
    token_warning_level = str(broker.get("token_warning_level") or "")
    swap_only_pressure = str(resource_guard.get("memory_pressure_kind") or "") == "swap_only"
    token_age_seconds = broker.get("token_age_seconds", token_guard.get("token_after", {}).get("age_seconds"))
    watchdog_targets = watchdog.get("status") if isinstance(watchdog.get("status"), list) else []
    watchdog_restart_storms = len(watchdog.get("restart_storms", []) or [])
    watchdog_alerts = len(watchdog.get("alerts", []) or [])
    watchdog_age_seconds = _payload_age_seconds(watchdog)
    watchdog_unhealthy_targets = 0
    watchdog_running_targets = 0
    for row in watchdog_targets:
        if not isinstance(row, dict):
            continue
        running_count = int(row.get("running", 0) or 0) + int(row.get("alt_running", 0) or 0)
        heartbeat_ok = bool(row.get("heartbeat_ok", False))
        if running_count > 0 and heartbeat_ok:
            watchdog_running_targets += 1
        else:
            watchdog_unhealthy_targets += 1
    watchdog_healthy = (
        bool(watchdog_targets)
        and watchdog_restart_storms == 0
        and watchdog_alerts == 0
        and watchdog_unhealthy_targets == 0
        and (watchdog_age_seconds is None or watchdog_age_seconds <= 900.0)
    )
    preopen_dashboard = {
        "broker_ready": broker_ready,
        "session_ready": session_ready,
        "network_ok": network_ok,
        "auth_ok": auth_ok,
        "token_warning_level": token_warning_level,
        "token_age_seconds": token_age_seconds,
        "account_probe_status_code": broker.get("account_probe_status_code"),
        "paper_lane_fresh": paper_lane_fresh,
        "live_lane_running": live_lane_running,
        "watchdog_restart_storms": watchdog_restart_storms,
    }

    hard_blocks = _ordered_unique(
        [
            "broker_not_ready" if not broker_ready else "",
            "network_not_ready" if not network_ok else "",
            "auth_not_ready" if not auth_ok else "",
            "session_not_ready" if not session_ready else "",
            "storage_not_ready" if not storage_ok else "",
            "watchdog_restart_storm" if watchdog_restart_storms > 0 else "",
            "watchdog_targets_missing" if watchdog_targets and watchdog_unhealthy_targets > 0 else "",
            "paper_lane_stale" if not paper_lane_fresh else "",
        ]
    )
    warnings = _ordered_unique(
        [
            "token_watch_window" if token_warning_level in {"watch", "warning"} else "",
            "token_stale" if token_warning_level in {"stale", "expired"} else "",
            "live_lane_not_running" if (args.allow_live_broker_submit and not live_lane_running) else "",
            "watchdog_alerts_present" if watchdog_alerts > 0 else "",
            "watchdog_payload_missing" if not watchdog_targets else "",
            "watchdog_payload_stale" if watchdog_age_seconds is not None and watchdog_age_seconds > 900.0 else "",
            "swap_only_pressure" if swap_only_pressure else "",
            "resource_guard_not_ok" if resource_guard and not bool(resource_guard.get("resource_guard_ok", False)) else "",
        ]
    )

    readiness_score = 100.0
    if not broker_ready:
        readiness_score -= 35.0
    if not network_ok:
        readiness_score -= 20.0
    if not auth_ok:
        readiness_score -= 20.0
    if not session_ready:
        readiness_score -= 15.0
    if not storage_ok:
        readiness_score -= 15.0
    if not paper_lane_fresh:
        readiness_score -= 15.0
    if watchdog_restart_storms > 0:
        readiness_score -= 20.0
    if watchdog_alerts > 0:
        readiness_score -= 8.0
    if watchdog_targets and watchdog_unhealthy_targets > 0:
        readiness_score -= min(12.0, float(watchdog_unhealthy_targets) * 4.0)
    if watchdog_age_seconds is not None and watchdog_age_seconds > 900.0:
        readiness_score -= 8.0
    if token_warning_level in {"watch", "warning"}:
        readiness_score -= 4.0
    elif token_warning_level in {"stale", "expired"}:
        readiness_score -= 10.0
    if swap_only_pressure:
        readiness_score -= 4.0
    if resource_guard and not bool(resource_guard.get("resource_guard_ok", False)):
        readiness_score -= 8.0
    if args.allow_live_broker_submit and not live_lane_running:
        readiness_score -= 6.0
    readiness_score = max(min(readiness_score, 100.0), 0.0)

    overall_status = "ready"
    if hard_blocks or readiness_score < 70.0:
        overall_status = "blocked"
    elif warnings or readiness_score < 90.0:
        overall_status = "degraded"

    recommended_actions = _ordered_unique(
        [
            "refresh_broker_auth_before_open" if token_warning_level in {"watch", "warning", "stale", "expired"} or not auth_ok else "",
            "stabilize_network_and_account_probe" if not network_ok else "",
            "restore_session_gate_prerequisites" if not session_ready else "",
            "refresh_or_restart_paper_execution_lane" if not paper_lane_fresh else "",
            "investigate_watchdog_alerts_and_missing_heartbeats" if (watchdog_alerts > 0 or watchdog_unhealthy_targets > 0) else "",
            "clear_restart_storm_before_live_submit" if watchdog_restart_storms > 0 else "",
            "recycle_noncritical_workers_to_relieve_swap_pressure" if swap_only_pressure else "",
            "keep_live_submit_disabled_until_live_lane_is_running" if (args.allow_live_broker_submit and not live_lane_running) else "",
        ]
    )

    payload = {
        "timestamp_utc": _iso_now(),
        "schema_version": 2,
        "ok": overall_status in {"ready", "degraded"},
        "overall_status": overall_status,
        "readiness_score": round(float(readiness_score), 2),
        "mode": "validate_only" if not args.allow_live_broker_submit else "broker_submit_enabled",
        "broker_ready": broker_ready,
        "session_ready": session_ready,
        "paper_lane_fresh": paper_lane_fresh,
        "live_lane_running": live_lane_running,
        "storage_mode": str(storage.get("mode") or ""),
        "storage_ok": storage_ok,
        "submit_path_enabled": bool(args.allow_live_broker_submit),
        "submit_guard_reason": ("" if args.allow_live_broker_submit else "explicit_flag_required"),
        "hard_blocks": hard_blocks,
        "warnings": warnings,
        "recommended_actions": recommended_actions,
        "preopen_dashboard": preopen_dashboard,
        "process_watchdog": {
            "healthy": watchdog_healthy,
            "payload_age_seconds": round(float(watchdog_age_seconds), 3) if watchdog_age_seconds is not None else None,
            "target_count": len(watchdog_targets),
            "healthy_target_count": int(watchdog_running_targets),
            "unhealthy_target_count": int(watchdog_unhealthy_targets),
            "restart_storm_count": int(watchdog_restart_storms),
            "alert_count": int(watchdog_alerts),
        },
        "memory_hygiene": {
            "resource_guard_ok": bool(resource_guard.get("resource_guard_ok", False)),
            "memory_pressure_state": str(resource_guard.get("memory_pressure_state") or ""),
            "memory_pressure_kind": str(resource_guard.get("memory_pressure_kind") or ""),
            "swap_used_gb": resource_guard.get("swap_used_gb"),
            "swap_only_headroom": bool(swap_only_pressure and broker_ready),
            "recommended_actions": [
                item
                for item in [
                    ("schedule_worker_recycle" if swap_only_pressure else ""),
                    ("prefer_sql_maintenance_off_hours" if swap_only_pressure else ""),
                    ("trim_noncritical_refresh_workers" if swap_only_pressure else ""),
                ]
                if item
            ],
        },
    }
    out_path = Path(args.out_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        print(
            "live_readiness_smoke "
            f"ok={int(bool(payload['ok']))} "
            f"overall_status={payload.get('overall_status', '')} "
            f"readiness_score={payload.get('readiness_score', 0.0)} "
            f"submit_enabled={int(bool(args.allow_live_broker_submit))}"
        )
    return 0 if payload["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
