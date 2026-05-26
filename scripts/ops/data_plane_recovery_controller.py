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
    from scripts.ops.long_runtime_common import PROJECT_ROOT, iso_now, load_json, ordered_unique, write_payload
else:
    from .long_runtime_common import PROJECT_ROOT, iso_now, load_json, ordered_unique, write_payload


DEFAULT_OUT_PATH = PROJECT_ROOT / "governance" / "health" / "data_plane_recovery_controller_latest.json"


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except Exception:
        return int(default)


def _parse_dt(value: Any) -> datetime | None:
    if not value:
        return None
    text = str(value).strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def build_payload(project_root: Path = PROJECT_ROOT) -> dict[str, Any]:
    health_root = project_root / "governance" / "health"
    incident = load_json(health_root / "incident_timeline_latest.json")
    backlog_drain = load_json(health_root / "external_backlog_drain_latest.json")
    queue = load_json(health_root / "ingestion_priority_queue_latest.json")
    storage = load_json(health_root / "storage_tier_policy_latest.json")
    storage_control = load_json(health_root / "ingestion_storage_control_latest.json")
    runtime = load_json(health_root / "live_runtime_separation_control_latest.json")
    writer_progress = load_json(health_root / "sql_link_service_progress_latest.json")
    snapshot_cache = load_json(health_root / "broker_truth_shared_snapshot_schwab_latest.json")

    recent = incident.get("recent_incidents") if isinstance(incident.get("recent_incidents"), list) else []
    write_failures = [
        row
        for row in recent
        if isinstance(row, dict) and str(row.get("summary") or "").strip().lower() == "write_failure"
    ]
    account_snapshot_failures = [
        row
        for row in recent
        if isinstance(row, dict) and str(row.get("summary") or "").strip().lower() == "get_accounts_snapshot"
    ]
    write_failure_count = len(write_failures)
    account_snapshot_count_raw = len(account_snapshot_failures)
    pending_lines = _safe_int((queue.get("lane_counts") or {}).get("core", {}).get("pending_lines", queue.get("queue_depth", 0)), 0)
    drain_status = str(backlog_drain.get("overall_status") or "").strip().lower()
    runtime_clearance = str(((runtime.get("clearance_plan") or {}).get("clearance_state") or "")).strip().lower()
    hot_path_over_budget_raw = _safe_int(((storage.get("pressure") or {}).get("hot_path_over_budget_bytes", 0)), 0)
    storage_target_status = storage_control.get("steady_state", {}).get("target_status", {}) if isinstance(storage_control.get("steady_state"), dict) else {}
    external_route = storage_control.get("external_route_verification") if isinstance(storage_control.get("external_route_verification"), dict) else {}
    storage_steady_state_ready = bool(
        str(storage_control.get("overall_status") or "").strip().lower() == "ready"
        and str(storage_control.get("severity") or "").strip().lower() == "stable"
        and bool(storage_target_status.get("steady_state_ready", False))
        and _safe_int(storage_control.get("backpressure_quality_score"), 0) >= 95
        and _safe_int(storage_control.get("recovery_quality_score"), 0) >= 88
        and str(external_route.get("verification_state") or "").strip().lower() in {"ready", "verified", "curated_ready", "active_passthrough"}
    )
    hot_path_over_budget = 0 if storage_steady_state_ready else hot_path_over_budget_raw
    writer_busy = str(writer_progress.get("status") or "").strip().lower() in {"running", "busy"}
    snapshot_cache_ready = bool(snapshot_cache.get("fetched")) and bool(snapshot_cache.get("timestamp_utc"))
    snapshot_cache_ts = _parse_dt(snapshot_cache.get("timestamp_utc"))
    snapshot_failure_times = [
        parsed
        for parsed in (_parse_dt(row.get("timestamp_utc")) for row in account_snapshot_failures)
        if parsed is not None
    ]
    last_snapshot_failure_ts = max(snapshot_failure_times) if snapshot_failure_times else None
    snapshot_recovered_by_cache = bool(
        account_snapshot_count_raw > 0
        and snapshot_cache_ready
        and snapshot_cache_ts is not None
        and last_snapshot_failure_ts is not None
        and snapshot_cache_ts >= last_snapshot_failure_ts
    )
    account_snapshot_count = 0 if snapshot_recovered_by_cache else account_snapshot_count_raw
    drain_delta = backlog_drain.get("drain_delta") if isinstance(backlog_drain.get("drain_delta"), dict) else {}
    follow_through = backlog_drain.get("follow_through") if isinstance(backlog_drain.get("follow_through"), dict) else {}
    blocked_reasons = [str(item).strip().lower() for item in (backlog_drain.get("blocked_reasons") or []) if str(item).strip()]
    drain_progress_lines = _safe_int(drain_delta.get("total_pending_lines"), 0)
    drain_apply_requested = bool(backlog_drain.get("apply_requested", False))
    market_hours_guard = "market_hours_guard" in blocked_reasons
    follow_through_status = str(follow_through.get("status") or "").strip().lower()
    active_recovery = bool(writer_busy or drain_apply_requested or drain_progress_lines > 0 or follow_through_status in {"running", "waiting_for_writer", "polling"})
    small_steady_queue = bool(
        storage_steady_state_ready
        and write_failure_count <= 0
        and account_snapshot_count <= 0
        and pending_lines <= 5000
    )

    recovery_state = "stable"
    if write_failure_count > 0 or account_snapshot_count > 0:
        recovery_state = "needs_recovery"
    if (drain_status == "blocked" and not small_steady_queue) or hot_path_over_budget > 0:
        recovery_state = "recovering_under_guard" if active_recovery else "blocked"
    elif active_recovery and recovery_state != "stable":
        recovery_state = "recovering_under_guard"
    elif pending_lines > 0 and writer_busy and not small_steady_queue:
        recovery_state = "recovering_under_guard"

    overall_status = "ready"
    if recovery_state == "blocked":
        overall_status = "blocked"
    elif recovery_state != "stable":
        overall_status = "degraded"

    recommended_actions = ordered_unique(
        [
            "drain deferred and cold backlogs before reopening write-heavy ingestion lanes" if recovery_state != "stable" else "",
            "treat repeated get_accounts_snapshot failures as a broker-side gating signal before thawing execution sleeves" if account_snapshot_count > 0 else "",
            "route write-heavy reconciliation through the single-writer service until hot-path pressure clears" if hot_path_over_budget > 0 else "",
            "use the shared broker snapshot cache as a stale fallback before reopening thaw candidates" if snapshot_cache_ready and account_snapshot_count > 0 else "",
            "keep the live lane read-only while the data plane catches up" if runtime_clearance not in {"ready", ""} and recovery_state != "stable" else "",
            "let the off-hours backlog drain fire on schedule instead of forcing deferred and cold replay work into market hours" if market_hours_guard else "",
        ]
    )

    return {
        "timestamp_utc": iso_now(),
        "schema_version": 1,
        "ok": recovery_state == "stable",
        "overall_status": overall_status,
        "recovery_state": recovery_state,
        "write_failure_count": write_failure_count,
        "account_snapshot_failure_count": account_snapshot_count,
        "raw_account_snapshot_failure_count": account_snapshot_count_raw,
        "queue_depth": pending_lines,
        "external_backlog_status": drain_status,
        "runtime_clearance_state": runtime_clearance,
        "hot_path_over_budget_bytes": hot_path_over_budget,
        "raw_hot_path_over_budget_bytes": hot_path_over_budget_raw,
        "storage_steady_state_ready": bool(storage_steady_state_ready),
        "small_steady_queue": bool(small_steady_queue),
        "recovery_contract": {
            "backlog_drain_required": write_failure_count > 0 or pending_lines > 0,
            "writer_handoff_required": hot_path_over_budget > 0,
            "writer_service_active": writer_busy,
            "execution_lane_pause_required": account_snapshot_count > 0,
            "recommended_command": [
                "./scripts/ops/opsctl.sh",
                "external-backlog-drain",
                "--json",
            ],
            "writer_cycle_command": [
                "./scripts/ops/opsctl.sh",
                "writer-cycle-coordinator",
                "--json",
            ],
            "snapshot_probe_required": account_snapshot_count > 0,
            "snapshot_cache_ready": snapshot_cache_ready,
            "snapshot_recovered_by_cache": snapshot_recovered_by_cache,
            "snapshot_probe_command": [
                "./scripts/ops/opsctl.sh",
                "token-refresh",
                "--json",
            ],
        },
        "backlog_recovery_contract": {
            "apply_requested": drain_apply_requested,
            "market_hours_guard": market_hours_guard,
            "blocked_reasons": blocked_reasons,
            "drain_progress_lines": drain_progress_lines,
            "follow_through_status": follow_through_status,
            "progress_observed": bool(drain_progress_lines > 0 or bool(follow_through.get("progress_observed", False))),
            "off_hours_window": backlog_drain.get("off_hours_window") if isinstance(backlog_drain.get("off_hours_window"), dict) else {},
        },
        "writer_handoff_contract": {
            "service_status": str(writer_progress.get("status") or ""),
            "service_current_step": str(writer_progress.get("current_step") or ""),
            "writer_service_active": writer_busy,
            "hot_path_over_budget_bytes": hot_path_over_budget,
            "raw_hot_path_over_budget_bytes": hot_path_over_budget_raw,
            "storage_steady_state_ready": bool(storage_steady_state_ready),
            "preferred_mode": "single_writer_service" if hot_path_over_budget > 0 else "standard",
            "handoff_progress_state": ("active" if writer_busy else "idle"),
        },
        "snapshot_recovery_contract": {
            "cache_ready": snapshot_cache_ready,
            "cache_timestamp_utc": str(snapshot_cache.get("timestamp_utc") or ""),
            "last_failure_timestamp_utc": last_snapshot_failure_ts.isoformat() if last_snapshot_failure_ts else "",
            "recovered_by_fresh_cache": snapshot_recovered_by_cache,
            "stale_fallback_allowed": snapshot_cache_ready,
            "bounded_retry_count": 2 if account_snapshot_count > 0 else 0,
            "probe_required": account_snapshot_count > 0,
        },
        "recommended_actions": recommended_actions,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Publish a bounded recovery contract for repeated write-failure and account-snapshot incidents.")
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
            "data_plane_recovery_controller "
            f"overall_status={payload.get('overall_status', '')} "
            f"write_failure_count={int(payload.get('write_failure_count', 0) or 0)}"
        )
    return 0 if payload.get("overall_status") in {"ready", "degraded"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
