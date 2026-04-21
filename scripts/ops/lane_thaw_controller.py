#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from scripts.ops.long_runtime_common import PROJECT_ROOT, iso_now, load_json, ordered_unique, write_payload
else:
    from .long_runtime_common import PROJECT_ROOT, iso_now, load_json, ordered_unique, write_payload


DEFAULT_OUT_PATH = PROJECT_ROOT / "governance" / "health" / "lane_thaw_controller_latest.json"
DEFAULT_HISTORY_WINDOW_HOURS = 72
DEFAULT_CHRONIC_TRIP_THRESHOLD = 3
DEFAULT_WATCH_TRIP_THRESHOLD = 2


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


def _lane_rows(project_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted((project_root / "governance" / "health").glob("data_ingress_latest_*.json")):
        payload = load_json(path)
        if not payload:
            continue
        lane = path.stem.replace("data_ingress_latest_", "")
        rows.append(
            {
                "lane": lane,
                "loop_state": str(payload.get("loop_state") or "").strip().lower(),
                "pause_gate": str(payload.get("pause_gate") or payload.get("pause_reason") or "").strip().lower(),
                "iter_error_rate": round(_safe_float(payload.get("iter_error_rate"), 0.0), 6),
                "iter_error_count": _safe_int(payload.get("iter_error_count"), 0),
                "api_error_total": _safe_int(((payload.get("total_counts") or {}).get("api_error", 0)), 0),
                "updated_at_utc": str(payload.get("timestamp_utc") or payload.get("updated_at_utc") or ""),
                "_source_file": str(path),
            }
        )
    return rows


def _parse_iso(raw: Any) -> datetime | None:
    text = str(raw or "").strip().replace("Z", "+00:00")
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text)
    except Exception:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _history_path_for(project_root: Path, history_path: Path | None) -> Path:
    if history_path is not None:
        return history_path
    return project_root / "governance" / "health" / "lane_thaw_controller_latest.json"


def _lane_history_from_payload(history_payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    history_root = history_payload.get("cooldown_history")
    if not isinstance(history_root, dict):
        return {}
    lane_history = history_root.get("lane_history")
    if not isinstance(lane_history, dict):
        return {}
    normalized: dict[str, dict[str, Any]] = {}
    for lane, row in lane_history.items():
        lane_key = str(lane or "").strip()
        if lane_key:
            normalized[lane_key] = row if isinstance(row, dict) else {}
    return normalized


def _recent_trip_starts(raw_values: Any, *, now: datetime, history_window_hours: int) -> list[str]:
    cutoff = now - timedelta(hours=max(int(history_window_hours), 1))
    normalized: list[str] = []
    for raw in raw_values if isinstance(raw_values, list) else []:
        parsed = _parse_iso(raw)
        if parsed is None:
            continue
        if cutoff <= parsed <= now:
            normalized.append(parsed.isoformat())
    return sorted(set(normalized))


def _trip_history_for_lane(
    lane: str,
    *,
    paused: bool,
    lane_updated_at: datetime | None,
    previous: dict[str, Any],
    now: datetime,
    history_window_hours: int,
    chronic_trip_threshold: int,
    watch_trip_threshold: int,
) -> dict[str, Any]:
    previous_active = bool(previous.get("active_trip", False))
    last_seen_state = str(previous.get("last_seen_state") or "clear")
    recent_starts = _recent_trip_starts(previous.get("recent_trip_starts_utc"), now=now, history_window_hours=history_window_hours)
    trip_count_total = _safe_int(previous.get("trip_count_total"), 0)
    new_trip_started = False
    recovered_this_run = False

    event_time = lane_updated_at or now
    if paused and not previous_active:
        new_trip_started = True
        trip_count_total += 1
        recent_starts.append(event_time.isoformat())
    elif paused and previous_active:
        event_time = _parse_iso(previous.get("last_trip_started_utc")) or event_time
    elif previous_active and not paused:
        recovered_this_run = True

    recent_starts = _recent_trip_starts(recent_starts, now=now, history_window_hours=history_window_hours)
    trip_count_window = len(recent_starts)
    escalation_level = "normal"
    if trip_count_window >= max(int(chronic_trip_threshold), 1):
        escalation_level = "chronic"
    elif trip_count_window >= max(int(watch_trip_threshold), 1):
        escalation_level = "watch"

    escalation_actions: list[str] = []
    if escalation_level == "watch":
        escalation_actions = [
            "increase_observation",
            "capture_trip_context",
        ]
    elif escalation_level == "chronic":
        escalation_actions = [
            "require_operator_review",
            "extend_cooldown_window",
            "open_incident_review_packet",
        ]

    last_trip_started_utc = str(previous.get("last_trip_started_utc") or "")
    if paused and new_trip_started:
        last_trip_started_utc = event_time.isoformat()

    last_trip_recovered_utc = str(previous.get("last_trip_recovered_utc") or "")
    if recovered_this_run:
        last_trip_recovered_utc = now.isoformat()

    return {
        "lane": lane,
        "active_trip": paused,
        "new_trip_started": new_trip_started,
        "recovered_this_run": recovered_this_run,
        "trip_count_total": trip_count_total,
        "trip_count_window": trip_count_window,
        "recent_trip_starts_utc": recent_starts,
        "last_trip_started_utc": last_trip_started_utc,
        "last_trip_recovered_utc": last_trip_recovered_utc,
        "last_seen_state": ("paused_anomaly_killswitch" if paused else "clear"),
        "previous_seen_state": last_seen_state,
        "history_window_hours": max(int(history_window_hours), 1),
        "watch_trip_threshold": max(int(watch_trip_threshold), 1),
        "chronic_trip_threshold": max(int(chronic_trip_threshold), 1),
        "watch_candidate": escalation_level == "watch",
        "chronic_offender": escalation_level == "chronic",
        "escalation_level": escalation_level,
        "escalation_actions": escalation_actions,
    }


def _adaptive_cooldown(
    row: dict[str, Any],
    *,
    auth_state: str,
    clearance_state: str,
    trip_history: dict[str, Any],
) -> dict[str, Any]:
    now = datetime.now(timezone.utc)
    updated_at = _parse_iso(row.get("updated_at_utc"))
    source_file = Path(str(row.get("_source_file") or ""))
    if updated_at is None and source_file.exists():
        try:
            updated_at = datetime.fromtimestamp(source_file.stat().st_mtime, tz=timezone.utc)
        except Exception:
            updated_at = None

    cooldown_seconds = 300
    reasons: list[str] = []
    api_error_total = _safe_int(row.get("api_error_total"), 0)
    iter_error_rate = _safe_float(row.get("iter_error_rate"), 0.0)
    if api_error_total >= 250:
        cooldown_seconds += 900
        reasons.append("api_error_cooldown")
    elif api_error_total >= 100:
        cooldown_seconds += 420
        reasons.append("api_error_cooldown")
    if iter_error_rate > 0.02:
        cooldown_seconds += 600
        reasons.append("iter_error_cooldown")
    if auth_state == "warning":
        cooldown_seconds += 300
        reasons.append("auth_warning_cooldown")
    if clearance_state not in {"coverage_cycles_ready", "off_hours_cold_lane_launch_ready", "ready", ""}:
        cooldown_seconds += 300
        reasons.append("runtime_clearance_cooldown")
    escalation_level = str(trip_history.get("escalation_level") or "normal")
    if escalation_level == "watch":
        cooldown_seconds += 300
        reasons.append("repeat_trip_cooldown")
    elif escalation_level == "chronic":
        cooldown_seconds += 900
        reasons.append("chronic_trip_cooldown")

    elapsed_seconds = None
    remaining_seconds = None
    thaw_after_utc = ""
    cooldown_state = "unknown"
    if updated_at is not None:
        elapsed_seconds = max((now - updated_at).total_seconds(), 0.0)
        remaining_seconds = max(float(cooldown_seconds) - elapsed_seconds, 0.0)
        thaw_after_utc = (updated_at + timedelta(seconds=int(cooldown_seconds))).isoformat()
        cooldown_state = "elapsed" if remaining_seconds <= 0.0 else "active"

    return {
        "base_seconds": 300,
        "adaptive_seconds": int(cooldown_seconds),
        "elapsed_seconds": (round(elapsed_seconds, 3) if elapsed_seconds is not None else None),
        "remaining_seconds": (round(remaining_seconds, 3) if remaining_seconds is not None else None),
        "thaw_after_utc": thaw_after_utc,
        "state": cooldown_state,
        "reasons": reasons,
    }
def build_payload(
    project_root: Path = PROJECT_ROOT,
    *,
    history_path: Path | None = None,
    history_window_hours: int = DEFAULT_HISTORY_WINDOW_HOURS,
    chronic_trip_threshold: int = DEFAULT_CHRONIC_TRIP_THRESHOLD,
    watch_trip_threshold: int = DEFAULT_WATCH_TRIP_THRESHOLD,
) -> dict[str, Any]:
    health_root = project_root / "governance" / "health"
    runtime = load_json(health_root / "live_runtime_separation_control_latest.json")
    auth = load_json(health_root / "auth_lease_manager_latest.json")
    coverage = load_json(project_root / "governance" / "walk_forward" / "coverage_gap_closer_latest.json")
    watchdog = load_json(health_root / "process_watchdog_latest.json")
    prior_payload = load_json(_history_path_for(project_root, history_path))
    prior_lane_history = _lane_history_from_payload(prior_payload)

    lanes = _lane_rows(project_root)
    paused = [row for row in lanes if str(row.get("loop_state") or "") == "paused_anomaly_killswitch"]
    paused_by_lane = {str(row.get("lane") or ""): row for row in paused}
    auth_state = str(auth.get("lease_state") or "").strip().lower()
    clearance_state = str(((runtime.get("clearance_plan") or {}).get("clearance_state") or "")).strip().lower()
    restart_storms = len(watchdog.get("restart_storms") or [])
    autopilot_contract = (coverage.get("autopilot_contract") or {}) if isinstance(coverage.get("autopilot_contract"), dict) else {}
    coverage_ready = bool(
        autopilot_contract.get("can_launch_now", False)
        or autopilot_contract.get("can_auto_launch_off_hours", False)
    )
    now = datetime.now(timezone.utc)

    lane_history: dict[str, dict[str, Any]] = {}
    for lane_name in sorted(set(prior_lane_history) | set(paused_by_lane)):
        row = paused_by_lane.get(lane_name)
        updated_at = _parse_iso((row or {}).get("updated_at_utc")) if row else None
        lane_history[lane_name] = _trip_history_for_lane(
            lane_name,
            paused=row is not None,
            lane_updated_at=updated_at,
            previous=prior_lane_history.get(lane_name, {}),
            now=now,
            history_window_hours=history_window_hours,
            chronic_trip_threshold=chronic_trip_threshold,
            watch_trip_threshold=watch_trip_threshold,
        )

    thaw_rows: list[dict[str, Any]] = []
    for row in paused:
        reasons: list[str] = []
        state = "candidate"
        trip_history = lane_history.get(str(row.get("lane") or ""), {})
        cooldown_contract = _adaptive_cooldown(
            row,
            auth_state=auth_state,
            clearance_state=clearance_state,
            trip_history=trip_history,
        )
        if auth_state == "critical":
            state = "blocked"
            reasons.append("auth_critical")
        if restart_storms > 0:
            state = "blocked"
            reasons.append("restart_storm_present")
        if clearance_state not in {"coverage_cycles_ready", "off_hours_cold_lane_launch_ready", "ready", ""} and not coverage_ready:
            state = "hold"
            reasons.append(f"runtime_clearance={clearance_state or 'protect_live'}")
        if _safe_float(row.get("iter_error_rate"), 0.0) > 0.02:
            state = "hold"
            reasons.append("iter_error_rate_elevated")
        if _safe_int(row.get("api_error_total"), 0) >= 250:
            state = "hold"
            reasons.append("api_error_total_hot")
        if str(cooldown_contract.get("state") or "") == "active":
            state = "hold"
            reasons.append("cooldown_active")
        escalation_level = str(trip_history.get("escalation_level") or "normal")
        if escalation_level == "watch":
            reasons.append("repeat_trip_watch")
        if escalation_level == "chronic":
            state = "hold"
            reasons.append("chronic_offender_review_required")
        if not reasons and auth_state == "warning":
            reasons.append("auth_warning_but_observable")
        thaw_rows.append(
            {
                **row,
                "thaw_state": state,
                "reasons": reasons,
                "cooldown_contract": cooldown_contract,
                "trip_history": trip_history,
                "escalation_contract": {
                    "level": escalation_level,
                    "auto_actions": list(trip_history.get("escalation_actions") or []),
                    "operator_review_required": escalation_level == "chronic",
                    "incident_key": (
                        f"lane_chronic_offender:{row.get('lane')}"
                        if escalation_level == "chronic"
                        else f"lane_repeat_trip:{row.get('lane')}"
                        if escalation_level == "watch"
                        else ""
                    ),
                },
                "recovery_contract": {
                    "refresh_accounts_snapshot_first": True,
                    "rerun_market_data_probe": True,
                    "micro_replay_required": True,
                    "operator_resume_required": state != "candidate" or escalation_level == "chronic",
                },
            }
        )

    candidate_count = sum(1 for row in thaw_rows if str(row.get("thaw_state") or "") == "candidate")
    hold_count = sum(1 for row in thaw_rows if str(row.get("thaw_state") or "") == "hold")
    blocked_count = sum(1 for row in thaw_rows if str(row.get("thaw_state") or "") == "blocked")
    new_trip_count = sum(1 for row in lane_history.values() if bool(row.get("new_trip_started", False)))
    recovered_trip_count = sum(1 for row in lane_history.values() if bool(row.get("recovered_this_run", False)))
    watchlist_count = sum(1 for row in lane_history.values() if bool(row.get("watch_candidate", False)))
    chronic_offender_count = sum(1 for row in lane_history.values() if bool(row.get("chronic_offender", False)))

    overall_status = "ready"
    if blocked_count > 0:
        overall_status = "blocked"
    elif paused:
        overall_status = "degraded"

    recommended_actions = ordered_unique(
        [
            "refresh account snapshots and a small market-data probe before thawing anomaly-paused sleeves" if paused else "",
            "only thaw candidate lanes after a micro replay confirms they stay below the anomaly gate" if candidate_count > 0 else "",
            "keep high-api-error lanes frozen until the broker/auth surfaces cool off" if hold_count > 0 or blocked_count > 0 else "",
            "respect adaptive cooldown windows so anomaly-paused lanes do not bounce between resume and rehalt under the same broker or data stress" if paused else "",
            "escalate chronic anomaly lanes into operator review and incident review packets before allowing another thaw attempt" if chronic_offender_count > 0 else "",
            "watch repeat-trip lanes closely because they are trending toward chronic guardrail offender status" if watchlist_count > 0 else "",
        ]
    )

    return {
        "timestamp_utc": iso_now(),
        "schema_version": 2,
        "ok": overall_status == "ready",
        "overall_status": overall_status,
        "paused_lane_count": len(paused),
        "candidate_count": candidate_count,
        "hold_count": hold_count,
        "blocked_count": blocked_count,
        "clearance_state": clearance_state,
        "auth_state": auth_state,
        "coverage_launch_ready": coverage_ready,
        "lanes": thaw_rows,
        "cooldown_history": {
            "history_window_hours": max(int(history_window_hours), 1),
            "watch_trip_threshold": max(int(watch_trip_threshold), 1),
            "chronic_trip_threshold": max(int(chronic_trip_threshold), 1),
            "new_trip_count": new_trip_count,
            "recovered_trip_count": recovered_trip_count,
            "watchlist_count": watchlist_count,
            "chronic_offender_count": chronic_offender_count,
            "lane_history": lane_history,
        },
        "recommended_actions": recommended_actions,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Identify anomaly-paused sleeves that are safe to thaw after bounded recovery checks.")
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--out-file", default=str(DEFAULT_OUT_PATH))
    parser.add_argument("--history-window-hours", type=int, default=DEFAULT_HISTORY_WINDOW_HOURS)
    parser.add_argument("--watch-trip-threshold", type=int, default=DEFAULT_WATCH_TRIP_THRESHOLD)
    parser.add_argument("--chronic-trip-threshold", type=int, default=DEFAULT_CHRONIC_TRIP_THRESHOLD)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    out_path = Path(args.out_file).expanduser()
    payload = build_payload(
        project_root,
        history_path=out_path,
        history_window_hours=max(int(args.history_window_hours), 1),
        watch_trip_threshold=max(int(args.watch_trip_threshold), 1),
        chronic_trip_threshold=max(int(args.chronic_trip_threshold), 1),
    )
    write_payload(out_path, payload)
    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        print(
            "lane_thaw_controller "
            f"overall_status={payload.get('overall_status', '')} "
            f"paused_lane_count={int(payload.get('paused_lane_count', 0) or 0)}"
        )
    return 0 if payload.get("overall_status") in {"ready", "degraded"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
