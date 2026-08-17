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
    from scripts.ops.long_runtime_common import PROJECT_ROOT, iso_now, load_json, ordered_unique, parse_iso_utc, write_payload
    from scripts.ops import production_quality_control
else:
    from .long_runtime_common import PROJECT_ROOT, iso_now, load_json, ordered_unique, parse_iso_utc, write_payload
    from . import production_quality_control


DEFAULT_OUT_PATH = PROJECT_ROOT / "governance" / "health" / "production_quality_slo_guard_latest.json"
DEFAULT_STATE_PATH = PROJECT_ROOT / "governance" / "health" / "production_quality_slo_guard_state.json"
DEFAULT_HISTORY_PATH = PROJECT_ROOT / "governance" / "health" / "production_quality_slo_events.jsonl"
SCHEMA_VERSION = 1
CRITICAL_WARN_MINUTES = 30.0
CRITICAL_BREACH_MINUTES = 120.0
HIGH_WARN_MINUTES = 120.0
HIGH_BREACH_MINUTES = 360.0
LANE_SCOPE_BY_ID = {
    "raw_profitability_recovery": "economic_evidence",
    "paper_trading_continuity": "runtime_operation",
    "auth_token_continuity": "runtime_operation",
    "storage_pressure_clean": "runtime_operation",
    "promotion_paper_freshness": "promotion_evidence",
    "source_and_ci_integrity": "release_integrity",
}


def _as_dict(raw: Any) -> dict[str, Any]:
    return raw if isinstance(raw, dict) else {}


def _as_list(raw: Any) -> list[Any]:
    return raw if isinstance(raw, list) else []


def _safe_float(raw: Any, default: float = 0.0) -> float:
    try:
        return float(raw)
    except Exception:
        return float(default)


def _iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat()


def _load_or_build_quality(project_root: Path, *, refresh_quality: bool, apply: bool) -> dict[str, Any]:
    path = project_root / "governance" / "health" / "production_quality_control_latest.json"
    if refresh_quality:
        return production_quality_control.build_payload(project_root, refresh_contract=True, apply=apply)
    payload = load_json(path)
    if payload:
        return payload
    return production_quality_control.build_payload(project_root, refresh_contract=True, apply=apply)


def _lane_thresholds(severity: str) -> tuple[float, float]:
    if str(severity or "").strip().lower() == "critical":
        return CRITICAL_WARN_MINUTES, CRITICAL_BREACH_MINUTES
    return HIGH_WARN_MINUTES, HIGH_BREACH_MINUTES


def _lane_duration_minutes(existing: dict[str, Any], now: datetime) -> tuple[str, float]:
    first_seen = str(existing.get("first_seen_utc") or "").strip()
    first_seen_dt = parse_iso_utc(first_seen)
    if first_seen_dt is None:
        first_seen_dt = now
        first_seen = _iso(now)
    duration = max((now - first_seen_dt).total_seconds() / 60.0, 0.0)
    return first_seen, duration


def _active_lane_rows(quality: dict[str, Any], previous_state: dict[str, Any], now: datetime) -> list[dict[str, Any]]:
    previous_lanes = _as_dict(previous_state.get("lanes"))
    rows: list[dict[str, Any]] = []
    for lane in _as_list(quality.get("active_lanes")):
        if not isinstance(lane, dict):
            continue
        lane_id = str(lane.get("lane_id") or "").strip()
        if not lane_id:
            continue
        existing = _as_dict(previous_lanes.get(lane_id))
        first_seen, duration_minutes = _lane_duration_minutes(existing, now)
        hit_count = int(_safe_float(existing.get("hit_count"), 0.0)) + 1
        severity = str(lane.get("severity") or "high").strip().lower()
        warn_minutes, breach_minutes = _lane_thresholds(severity)
        status = "breach" if duration_minutes >= breach_minutes else "warning" if duration_minutes >= warn_minutes else "watch"
        rows.append(
            {
                "lane_id": lane_id,
                "title": lane.get("title"),
                "severity": severity,
                "scope": str(lane.get("scope") or LANE_SCOPE_BY_ID.get(lane_id) or "runtime_operation"),
                "status": status,
                "first_seen_utc": first_seen,
                "last_seen_utc": _iso(now),
                "active_minutes": round(duration_minutes, 4),
                "warn_after_minutes": warn_minutes,
                "breach_after_minutes": breach_minutes,
                "hit_count": hit_count,
                "blocking_reasons": _as_list(lane.get("blocking_reasons")),
                "owner_capabilities": _as_list(lane.get("owner_capabilities")),
                "stop_when": lane.get("stop_when"),
                "expected_impact": lane.get("expected_impact"),
                "commands": _as_list(lane.get("commands")),
            }
        )
    return rows


def _next_state(active_lanes: list[dict[str, Any]], previous_state: dict[str, Any], now: datetime) -> dict[str, Any]:
    previous_lanes = _as_dict(previous_state.get("lanes"))
    active_ids = {str(row.get("lane_id") or "") for row in active_lanes}
    lanes = {str(row.get("lane_id")): row for row in active_lanes if row.get("lane_id")}
    resolved: list[dict[str, Any]] = []
    for lane_id, previous in previous_lanes.items():
        if lane_id in active_ids:
            continue
        resolved.append(
            {
                "lane_id": lane_id,
                "resolved_at_utc": _iso(now),
                "first_seen_utc": previous.get("first_seen_utc"),
                "last_seen_utc": previous.get("last_seen_utc"),
                "previous_status": previous.get("status"),
                "hit_count": previous.get("hit_count", 0),
            }
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "timestamp_utc": iso_now(),
        "lanes": lanes,
        "last_resolved_lanes": resolved[-20:],
        "policy": {
            "critical_warn_minutes": CRITICAL_WARN_MINUTES,
            "critical_breach_minutes": CRITICAL_BREACH_MINUTES,
            "high_warn_minutes": HIGH_WARN_MINUTES,
            "high_breach_minutes": HIGH_BREACH_MINUTES,
            "live_execution_authority": False,
        },
    }


def _append_history(history_path: Path, payload: dict[str, Any]) -> None:
    history_path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "timestamp_utc": payload.get("timestamp_utc"),
        "overall_status": payload.get("overall_status"),
        "active_lane_count": payload.get("active_lane_count"),
        "breach_count": payload.get("breach_count"),
        "warning_count": payload.get("warning_count"),
        "operational_status": payload.get("operational_status"),
        "operational_active_lane_count": payload.get("operational_active_lane_count"),
        "operational_breach_count": payload.get("operational_breach_count"),
        "breached_lane_ids": [row.get("lane_id") for row in _as_list(payload.get("breached_lanes"))],
        "warning_lane_ids": [row.get("lane_id") for row in _as_list(payload.get("warning_lanes"))],
        "live_execution_authority": False,
    }
    with history_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def build_payload(
    project_root: Path = PROJECT_ROOT,
    *,
    refresh_quality: bool = False,
    apply: bool = False,
    state_path: Path | None = None,
    out_path: Path | None = None,
    history_path: Path | None = None,
) -> dict[str, Any]:
    now = datetime.now(timezone.utc)
    effective_state = state_path or project_root / "governance" / "health" / DEFAULT_STATE_PATH.name
    effective_out = out_path or project_root / "governance" / "health" / DEFAULT_OUT_PATH.name
    effective_history = history_path or project_root / "governance" / "health" / DEFAULT_HISTORY_PATH.name
    previous_state = load_json(effective_state)
    quality = _load_or_build_quality(project_root, refresh_quality=refresh_quality, apply=apply)
    active_lanes = _active_lane_rows(quality, previous_state, now)
    warning_lanes = [row for row in active_lanes if row.get("status") == "warning"]
    breached_lanes = [row for row in active_lanes if row.get("status") == "breach"]
    critical_active = [row for row in active_lanes if row.get("severity") == "critical"]
    operational_lanes = [row for row in active_lanes if row.get("scope") == "runtime_operation"]
    operational_warning_lanes = [row for row in operational_lanes if row.get("status") == "warning"]
    operational_breached_lanes = [row for row in operational_lanes if row.get("status") == "breach"]
    operational_critical_active = [row for row in operational_lanes if row.get("severity") == "critical"]
    non_operational_lanes = [row for row in active_lanes if row.get("scope") != "runtime_operation"]
    next_state = _next_state(active_lanes, previous_state, now)
    quality_status = str(quality.get("overall_status") or "").strip().lower()
    if breached_lanes:
        overall_status = "blocked"
    elif warning_lanes:
        overall_status = "degraded"
    elif active_lanes:
        overall_status = "watch"
    elif quality_status == "ready":
        overall_status = "ready"
    else:
        overall_status = "waiting_for_quality_signal"

    if operational_breached_lanes:
        operational_status = "blocked"
    elif operational_warning_lanes:
        operational_status = "degraded"
    elif operational_lanes:
        operational_status = "watch"
    else:
        operational_status = "ready"

    if active_lanes:
        recommended_actions = [
            "keep live orders disabled while production quality SLO is not ready",
            "use production-quality ordered lanes to repair active blockers",
            "use infrabot-adaptive-governor exact allowlist for any safe execution",
            "escalate breached lanes instead of repeating unbounded repair loops" if breached_lanes else "",
            "rerun production-quality-slo after each production-quality refresh",
        ]
    else:
        recommended_actions = [
            "production quality SLO is clear; keep monitoring resolved lanes for recurrence",
            "keep live money governed by the live-canary readiness contract",
            "rerun production-quality-slo after each production-quality refresh",
        ]

    payload = {
        "schema_version": SCHEMA_VERSION,
        "timestamp_utc": iso_now(),
        "overall_status": overall_status,
        "ok": overall_status == "ready",
        "operational_ok": operational_status == "ready",
        "operational_status": operational_status,
        "source": "production_quality_slo_guard",
        "live_execution_authority": False,
        "safe_apply_only": True,
        "production_quality_status": quality.get("overall_status"),
        "live_canary_money_ready": bool(_as_dict(quality.get("live_canary_readiness")).get("live_canary_money_ready", False)),
        "active_lane_count": len(active_lanes),
        "critical_active_lane_count": len(critical_active),
        "warning_count": len(warning_lanes),
        "breach_count": len(breached_lanes),
        "operational_active_lane_count": len(operational_lanes),
        "operational_critical_active_lane_count": len(operational_critical_active),
        "operational_warning_count": len(operational_warning_lanes),
        "operational_breach_count": len(operational_breached_lanes),
        "operational_lanes": operational_lanes,
        "operational_warning_lanes": operational_warning_lanes,
        "operational_breached_lanes": operational_breached_lanes,
        "non_operational_lane_count": len(non_operational_lanes),
        "non_operational_lanes": non_operational_lanes,
        "active_lanes": active_lanes,
        "warning_lanes": warning_lanes,
        "breached_lanes": breached_lanes,
        "state_path": str(effective_state),
        "history_path": str(effective_history),
        "quality_artifact_path": str(project_root / "governance" / "health" / "production_quality_control_latest.json"),
        "governor_safe_execution_command": _as_list(quality.get("governor_safe_execution_command")),
        "control_contract": {
            "tracks_recurring_degradation": True,
            "critical_lane_warn_minutes": CRITICAL_WARN_MINUTES,
            "critical_lane_breach_minutes": CRITICAL_BREACH_MINUTES,
            "high_lane_warn_minutes": HIGH_WARN_MINUTES,
            "high_lane_breach_minutes": HIGH_BREACH_MINUTES,
            "live_orders_remain_disabled_while_active_or_breached": bool(active_lanes or breached_lanes),
            "runtime_source_mutation_allowed": False,
            "operational_health_uses_runtime_operation_lanes_only": True,
            "economic_release_and_promotion_debt_remain_visible_and_live_blocking": True,
        },
        "next_state": next_state,
        "recommended_actions": ordered_unique(recommended_actions),
    }
    if apply:
        write_payload(effective_state, next_state)
        write_payload(effective_out, payload)
        _append_history(effective_history, payload)
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Track recurring production-quality lane degradation against SLOs.")
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--apply", action="store_true", help="Persist state/history and write latest SLO artifact.")
    parser.add_argument("--refresh-quality", action="store_true", help="Refresh production_quality_control_latest.json first.")
    parser.add_argument("--state-path", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--history-path", type=Path, default=None)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    payload = build_payload(
        args.project_root.resolve(),
        refresh_quality=args.refresh_quality,
        apply=args.apply,
        state_path=args.state_path,
        out_path=args.out,
        history_path=args.history_path,
    )
    if args.json:
        print(json.dumps(payload, ensure_ascii=True, indent=2))
    else:
        print(
            "production_quality_slo_guard "
            f"status={payload['overall_status']} "
            f"active_lanes={payload['active_lane_count']} "
            f"breaches={payload['breach_count']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
