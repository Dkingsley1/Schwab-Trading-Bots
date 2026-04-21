#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, time, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_PATH = PROJECT_ROOT / "governance" / "health" / "ingestion_storage_control_latest.json"
LOCAL_TZ = ZoneInfo("America/New_York")
OFF_HOURS_START = time(16, 15)
OFF_HOURS_END = time(9, 20)
DEFAULT_TARGET_PRESSURE_INDEX = 0.25
DEFAULT_TARGET_CORE_PENDING_LINES = 5000
DEFAULT_TARGET_TOTAL_DRAIN_MINUTES = 15.0
DEFAULT_TARGET_STALE_STAGE_PENDING_LINES = 0
DEFAULT_TARGET_RETENTION_DEBT_GB = 0.25


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _parse_iso_utc(raw: Any) -> datetime | None:
    text = str(raw or "").strip()
    if not text:
        return None
    text = text.replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(text)
    except Exception:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _safe_float(raw: Any, default: float = 0.0) -> float:
    try:
        return float(raw)
    except Exception:
        return float(default)


def _safe_int(raw: Any, default: int = 0) -> int:
    try:
        return int(float(raw))
    except Exception:
        return int(default)


def _steady_state_targets() -> dict[str, Any]:
    return {
        "pressure_index": max(_safe_float(os.getenv("BACKPRESSURE_TARGET_PRESSURE_INDEX"), DEFAULT_TARGET_PRESSURE_INDEX), 0.01),
        "core_pending_lines": max(_safe_int(os.getenv("BACKPRESSURE_TARGET_CORE_PENDING_LINES"), DEFAULT_TARGET_CORE_PENDING_LINES), 0),
        "estimated_total_drain_minutes": max(
            _safe_float(os.getenv("BACKPRESSURE_TARGET_TOTAL_DRAIN_MINUTES"), DEFAULT_TARGET_TOTAL_DRAIN_MINUTES),
            0.0,
        ),
        "stale_stage_pending_lines": max(
            _safe_int(os.getenv("BACKPRESSURE_TARGET_STALE_STAGE_PENDING_LINES"), DEFAULT_TARGET_STALE_STAGE_PENDING_LINES),
            0,
        ),
        "retention_debt_gb": max(
            _safe_float(os.getenv("BACKPRESSURE_TARGET_RETENTION_DEBT_GB"), DEFAULT_TARGET_RETENTION_DEBT_GB),
            0.0,
        ),
        "support_watchdog_shard_required": True,
    }


def _target_ratio(actual: float, target: float) -> float:
    actual_value = max(float(actual), 0.0)
    target_value = max(float(target), 0.0)
    if target_value <= 0.0:
        return 0.0 if actual_value <= 0.0 else 1.0
    return actual_value / target_value


def _backpressure_scorecard(
    *,
    pressure_index: float,
    core_pending_lines: int,
    total_pending_lines: int,
    drain_minutes_total: float | None,
    stale_stage_pending_lines: int,
    retention_debt_gb: float,
    overall_status: str,
    severity: str,
) -> dict[str, Any]:
    targets = _steady_state_targets()
    pressure_ratio = _target_ratio(pressure_index, float(targets["pressure_index"]))
    core_ratio = _target_ratio(core_pending_lines, float(targets["core_pending_lines"]))
    retention_ratio = _target_ratio(retention_debt_gb, float(targets["retention_debt_gb"]))
    stale_stage_ratio = _target_ratio(stale_stage_pending_lines, max(_safe_int(targets["stale_stage_pending_lines"]), 0))
    if drain_minutes_total is None:
        total_drain_ratio = 0.0 if int(total_pending_lines) <= 0 else 2.0
    else:
        total_drain_ratio = _target_ratio(drain_minutes_total, float(targets["estimated_total_drain_minutes"]))

    penalties = {
        "pressure_index": min(max(pressure_ratio - 1.0, 0.0) * 35.0, 35.0),
        "core_pending_lines": min(max(core_ratio - 1.0, 0.0) * 25.0, 25.0),
        "estimated_total_drain_minutes": min(max(total_drain_ratio - 1.0, 0.0) * 25.0, 25.0),
        "stale_stage_pending_lines": 15.0 if stale_stage_ratio > 0.0 else 0.0,
        "retention_debt_gb": min(max(retention_ratio - 1.0, 0.0) * 10.0, 10.0),
    }
    quality_score = max(0.0, 100.0 - sum(penalties.values()))

    target_flags = {
        "pressure_index_ok": pressure_ratio <= 1.0,
        "core_pending_lines_ok": core_ratio <= 1.0,
        "estimated_total_drain_minutes_ok": total_drain_ratio <= 1.0,
        "stale_stage_pending_lines_ok": stale_stage_ratio <= 0.0,
        "retention_debt_gb_ok": retention_ratio <= 1.0,
    }
    target_breaches = [
        name
        for name, ok in (
            ("pressure_index", target_flags["pressure_index_ok"]),
            ("core_pending_lines", target_flags["core_pending_lines_ok"]),
            ("estimated_total_drain_minutes", target_flags["estimated_total_drain_minutes_ok"]),
            ("stale_stage_pending_lines", target_flags["stale_stage_pending_lines_ok"]),
            ("retention_debt_gb", target_flags["retention_debt_gb_ok"]),
        )
        if not ok
    ]
    if quality_score >= 95.0:
        quality_label = "excellent"
    elif quality_score >= 85.0:
        quality_label = "good"
    elif quality_score >= 70.0:
        quality_label = "watch"
    elif quality_score >= 50.0:
        quality_label = "stressed"
    else:
        quality_label = "degraded"

    return {
        "targets": targets,
        "target_status": {
            **target_flags,
            "steady_state_ready": not target_breaches and str(overall_status or "") == "ready" and str(severity or "") == "stable",
            "target_breach_count": len(target_breaches),
            "target_breaches": target_breaches,
        },
        "quality_score": round(quality_score, 2),
        "quality_label": quality_label,
        "penalties": {key: round(float(value), 3) for key, value in penalties.items()},
        "ratios": {
            "pressure_index": round(pressure_ratio, 3),
            "core_pending_lines": round(core_ratio, 3),
            "estimated_total_drain_minutes": round(total_drain_ratio, 3),
            "stale_stage_pending_lines": round(stale_stage_ratio, 3),
            "retention_debt_gb": round(retention_ratio, 3),
        },
    }


def _freshest_non_empty_json(paths: list[Path]) -> tuple[dict[str, Any], str]:
    best_payload: dict[str, Any] = {}
    best_source = ""
    best_ts = 0.0
    for path in paths:
        payload = _load_json(path)
        if not payload:
            continue
        ts = _parse_iso_utc(payload.get("timestamp_utc"))
        score = ts.timestamp() if ts is not None else 0.0
        if score >= best_ts:
            best_payload = payload
            best_source = str(path)
            best_ts = score
    return best_payload, best_source


def _off_hours_active(now_utc: datetime) -> bool:
    local_now = now_utc.astimezone(LOCAL_TZ)
    if local_now.weekday() >= 5:
        return True
    local_clock = local_now.timetz().replace(tzinfo=None)
    return bool(local_clock >= OFF_HOURS_START or local_clock < OFF_HOURS_END)


def build_payload(project_root: Path = PROJECT_ROOT, *, now_utc: datetime | None = None) -> dict[str, Any]:
    now = now_utc or datetime.now(timezone.utc)
    health_root = project_root / "governance" / "health"

    backpressure = _load_json(health_root / "ingestion_backpressure_latest.json")
    sql_progress = _load_json(health_root / "sql_link_service_progress_latest.json")
    sql_service = _load_json(health_root / "sql_link_service_latest.json")
    governor = _load_json(health_root / "ingestion_storage_governor_latest.json")
    backlog_drain = _load_json(health_root / "external_backlog_drain_latest.json")
    backlog_quarantine = _load_json(health_root / "backlog_quarantine_bot_latest.json")
    health_gates = _load_json(health_root / "health_gates_latest.json")
    storage_maintenance = _load_json(health_root / "storage_maintenance_latest.json")
    stale_sweeper = _load_json(health_root / "stale_artifact_sweeper_bot_latest.json")
    stale_reaper = _load_json(health_root / "stale_artifact_reaper_bot_latest.json")
    retention = _load_json(health_root / "data_retention_latest.json")
    sql_ingestion, sql_ingestion_source = _freshest_non_empty_json(
        [
            health_root / "jsonl_sql_ingestion_health_trading_latest.json",
            health_root / "jsonl_sql_ingestion_health_latest.json",
            health_root / "jsonl_sql_ingestion_health_data_latest.json",
            health_root / "jsonl_sql_ingestion_health_governance_latest.json",
        ]
    )

    cycle_started = _parse_iso_utc(sql_progress.get("cycle_started_utc"))
    cycle_elapsed_seconds = max((now - cycle_started).total_seconds(), 1.0) if cycle_started is not None else 0.0
    merged_rows_this_cycle = _safe_int(sql_progress.get("merged_rows_this_cycle"), 0)
    throughput_rows_per_second = (
        round(merged_rows_this_cycle / max(cycle_elapsed_seconds, 1.0), 6) if merged_rows_this_cycle > 0 and cycle_elapsed_seconds > 0 else 0.0
    )

    core_pending_lines = _safe_int(backpressure.get("pending_lines"), 0)
    total_pending_lines = _safe_int(backpressure.get("pending_lines_total"), 0)
    deferred_pending_lines = _safe_int(backpressure.get("pending_lines_deferred"), 0)
    cold_pending_lines = _safe_int(backpressure.get("pending_lines_cold"), 0)
    support_pending_lines = _safe_int(backpressure.get("pending_lines_support_telemetry"), 0)
    stale_stage_pending_lines = _safe_int(backpressure.get("pending_lines_stale_stage"), 0)
    oldest_age_seconds = _safe_float(backpressure.get("oldest_pending_age_seconds"), 0.0)
    pending_threshold = max(_safe_int(backpressure.get("pending_lines_threshold"), 15000), 1)
    age_threshold = max(_safe_float(backpressure.get("oldest_age_threshold_seconds"), 240.0), 1.0)
    retention_debt_gb = _safe_float(health_gates.get("storage_pressure", {}).get("retention_debt_gb"), _safe_float(health_gates.get("retention_debt_gb"), 0.0))
    severe_backpressure = bool(health_gates.get("storage_pressure", {}).get("severe_backpressure_overload", False) or health_gates.get("ingestion_pressure", {}).get("severe_backpressure_overload", False))
    hard_gate_flags = health_gates.get("hard_gates") if isinstance(health_gates.get("hard_gates"), dict) else {}
    storage_hard_gate = any(
        bool(hard_gate_flags.get(key, False))
        for key in (
            "ingestion_pending_lines",
            "ingestion_oldest_age",
            "ingestion_invalid_lines",
            "ingestion_backpressure_overload",
            "priority_shard_storage",
            "sql_progress_stall",
            "sql_wal_pressure",
        )
    )
    hard_gate = storage_hard_gate if hard_gate_flags else bool(health_gates.get("hard_gate_triggered", False))
    governor_profile = str(governor.get("profile") or "")
    governor_sql = governor.get("sql_primary_db") if isinstance(governor.get("sql_primary_db"), dict) else {}
    governor_throttles = governor.get("throttle_controls") if isinstance(governor.get("throttle_controls"), dict) else {}
    route_drift = bool(governor_sql.get("route_drift", False))
    drain_window = backlog_drain.get("off_hours_window") if isinstance(backlog_drain.get("off_hours_window"), dict) else {}
    drain_overrides = backlog_drain.get("drain_overrides") if isinstance(backlog_drain.get("drain_overrides"), dict) else {}
    backlog_drain_status = str(backlog_drain.get("overall_status") or "")
    backlog_drain_recommended = bool(backlog_drain.get("recommended_now", False))
    aged_candidate_files = _safe_int(backlog_drain.get("aged_candidate_files"), 0)
    off_hours_active = _off_hours_active(now)
    backlog_quarantine_status = str(backlog_quarantine.get("overall_status") or "")
    backlog_quarantine_candidate_files = _safe_int(backlog_quarantine.get("candidate_files"), 0)
    backlog_quarantine_moved_files = _safe_int(backlog_quarantine.get("moved_files"), 0)
    backlog_quarantine_moved_pending_lines = _safe_int(backlog_quarantine.get("moved_pending_lines"), 0)

    core_ratio = core_pending_lines / pending_threshold
    total_ratio = total_pending_lines / max(pending_threshold * 20, 1)
    age_ratio = oldest_age_seconds / age_threshold
    retention_ratio = retention_debt_gb / 2.0 if retention_debt_gb > 0.0 else 0.0
    drain_minutes_core = round((core_pending_lines / max(throughput_rows_per_second, 1e-9)) / 60.0, 3) if throughput_rows_per_second > 0.0 else None
    drain_minutes_total = round((total_pending_lines / max(throughput_rows_per_second, 1e-9)) / 60.0, 3) if throughput_rows_per_second > 0.0 else None
    pressure_index = round(max(core_ratio, age_ratio, total_ratio, retention_ratio, 0.0), 3)

    if hard_gate or severe_backpressure or pressure_index >= 3.0:
        severity = "critical"
    elif bool(backpressure.get("overload", False)) or pressure_index >= 1.5:
        severity = "high"
    elif pressure_index >= 0.75:
        severity = "elevated"
    else:
        severity = "stable"

    top_actions: list[str] = []
    if route_drift:
        top_actions.append("normalize the SQL linker back onto the routed repo DB path before allowing failback debt to grow")
    if backlog_drain_recommended:
        top_actions.append("run the external backlog drain during the current off-hours window to burn down deferred and cold backlog")
    if backlog_quarantine_candidate_files > 0:
        top_actions.append("stage stale prior-day shadow attribution and explanation backlog during market hours so the overnight drain can focus on live deferred lanes")
    if cold_pending_lines >= max(pending_threshold, 1000):
        top_actions.append("offload shadow attribution and other cold lanes first")
    if support_pending_lines >= max(pending_threshold, 1000):
        top_actions.append("keep watchdog failover and pager telemetry on the support shard so support logs stop inflating core ingestion pressure")
    if stale_stage_pending_lines >= max(pending_threshold, 1000):
        top_actions.append("reap or archive stale-stage artifacts so cold archive debt stops inflating total drain time")
    if retention_debt_gb > 0.0:
        top_actions.append("force retention and compaction on priority shards before broad retrains")
    if drain_minutes_core is not None and drain_minutes_core > 30.0:
        top_actions.append("reduce writer fan-out or increase merge throughput until core drain time is below 30 minutes")
    if drain_minutes_total is not None and drain_minutes_total > 180.0:
        top_actions.append("split deferred and explanation shards to keep total drain time below three hours")
    if _safe_int(((stale_sweeper.get("summary") or {}).get("candidate_files")), 0) > 0:
        top_actions.append("continue stale-stage sweeps so stale debug and report artifacts stop competing with hot ingestion")
    if _safe_int(sql_ingestion.get("sqlite", {}).get("invalid"), 0) > 0:
        top_actions.append("quarantine invalid ingestion rows before they amplify backlog and replay drift")
    if governor_profile == "critical_backpressure":
        top_actions.append("keep the storage-pressure governor in critical mode until deferred and cold lanes drain under target")
    if aged_candidate_files > 0:
        top_actions.append("retire or compact the oldest deferred and cold backlog files once the active drain pass completes")

    recommended_mode = str(health_gates.get("recommended_operating_mode") or "")
    if not recommended_mode:
        recommended_mode = "maintenance_only" if severity in {"critical", "high"} else "normal"
    if backlog_drain_recommended and severity in {"critical", "high"} and off_hours_active:
        recommended_mode = "maintenance_drain_window"
    elif severity in {"critical", "high"} and (backlog_drain_recommended or backlog_quarantine_candidate_files > 0):
        recommended_mode = "market_hours_backlog_protection"

    ok = severity in {"stable", "elevated"} and not hard_gate
    overall_status = "ready" if ok else "needs_work"
    if severity == "critical" or hard_gate:
        overall_status = "blocked"

    steady_state = _backpressure_scorecard(
        pressure_index=pressure_index,
        core_pending_lines=core_pending_lines,
        total_pending_lines=total_pending_lines,
        drain_minutes_total=drain_minutes_total,
        stale_stage_pending_lines=stale_stage_pending_lines,
        retention_debt_gb=retention_debt_gb,
        overall_status=overall_status,
        severity=severity,
    )

    payload = {
        "timestamp_utc": now.isoformat(),
        "schema_version": 1,
        "ok": ok,
        "overall_status": overall_status,
        "severity": severity,
        "recommended_operating_mode": recommended_mode,
        "pressure_index": pressure_index,
        "backpressure_quality_score": float(steady_state.get("quality_score", 0.0) or 0.0),
        "steady_state": steady_state,
        "throughput": {
            "merged_rows_this_cycle": merged_rows_this_cycle,
            "cycle_elapsed_seconds": round(cycle_elapsed_seconds, 3) if cycle_elapsed_seconds else 0.0,
            "throughput_rows_per_second": throughput_rows_per_second,
        },
        "backpressure": {
            "core_pending_lines": core_pending_lines,
            "deferred_pending_lines": deferred_pending_lines,
            "cold_pending_lines": cold_pending_lines,
            "support_pending_lines": support_pending_lines,
            "stale_stage_pending_lines": stale_stage_pending_lines,
            "total_pending_lines": total_pending_lines,
            "oldest_pending_age_seconds": round(oldest_age_seconds, 3),
            "pending_lines_threshold": pending_threshold,
            "oldest_age_threshold_seconds": round(age_threshold, 3),
            "estimated_core_drain_minutes": drain_minutes_core,
            "estimated_total_drain_minutes": drain_minutes_total,
        },
        "storage": {
            "retention_debt_gb": round(retention_debt_gb, 3),
            "sqlite_wal_size_gb": round(_safe_float(sql_service.get("sqlite_wal_size_gb"), 0.0), 3),
            "storage_maintenance_reason": str(storage_maintenance.get("reason") or ""),
            "governor_profile": governor_profile,
            "sql_primary_route_drift": route_drift,
            "backlog_drain_status": backlog_drain_status,
            "backlog_drain_recommended_now": backlog_drain_recommended,
            "backlog_drain_off_hours": off_hours_active,
            "aged_backlog_candidate_files": aged_candidate_files,
            "backlog_quarantine_status": backlog_quarantine_status,
            "backlog_quarantine_candidate_files": backlog_quarantine_candidate_files,
            "backlog_quarantine_moved_files": backlog_quarantine_moved_files,
            "backlog_quarantine_moved_pending_lines": backlog_quarantine_moved_pending_lines,
            "stale_stage_candidate_files": _safe_int(((stale_sweeper.get("summary") or {}).get("candidate_files")), 0),
            "stale_stage_staged_files": _safe_int(((stale_sweeper.get("summary") or {}).get("staged_files")), 0),
            "stale_stage_deleted_files": _safe_int(((stale_reaper.get("summary") or {}).get("deleted_files")), 0),
            "retention_deleted": _safe_int(retention.get("deleted"), 0),
        },
        "throttling": {
            "deferred_files_budget": _safe_int(governor_throttles.get("deferred_files_budget"), 0),
            "cold_files_budget": _safe_int(governor_throttles.get("cold_files_budget"), 0),
            "backlog_drain_deferred_budget": _safe_int(drain_overrides.get("deferred_files_budget"), 0),
            "backlog_drain_cold_budget": _safe_int(drain_overrides.get("cold_files_budget"), 0),
            "log_api_calls": str(governor_throttles.get("log_api_calls") or ""),
            "log_loop_state": str(governor_throttles.get("log_loop_state") or ""),
            "log_data_ingress": str(governor_throttles.get("log_data_ingress") or ""),
            "log_shadow_pnl_attribution": str(governor_throttles.get("log_shadow_pnl_attribution") or ""),
        },
        "data_integrity": {
            "sql_ingestion_source": sql_ingestion_source,
            "sql_invalid_lines": _safe_int(sql_ingestion.get("sqlite", {}).get("invalid"), 0),
            "sql_files_discovered": _safe_int(sql_ingestion.get("files_discovered"), 0),
        },
        "top_actions": top_actions[:8],
        "source_files": {
            "ingestion_backpressure": str(health_root / "ingestion_backpressure_latest.json"),
            "sql_link_service_progress": str(health_root / "sql_link_service_progress_latest.json"),
            "sql_link_service": str(health_root / "sql_link_service_latest.json"),
            "health_gates": str(health_root / "health_gates_latest.json"),
            "storage_maintenance": str(health_root / "storage_maintenance_latest.json"),
            "ingestion_storage_governor": str(health_root / "ingestion_storage_governor_latest.json"),
            "external_backlog_drain": str(health_root / "external_backlog_drain_latest.json"),
        },
    }
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Build an ingestion and storage control-plane artifact with drain-time estimates.")
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--out-file", default=str(DEFAULT_OUT_PATH))
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    payload = build_payload(Path(args.project_root).resolve())
    out_path = Path(args.out_file).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        print(
            "ingestion_storage_control "
            f"status={payload['overall_status']} "
            f"severity={payload.get('severity', '')} "
            f"pressure_index={float(payload.get('pressure_index', 0.0) or 0.0):.2f}"
        )
    return 0 if bool(payload.get("ok", False)) else 2


if __name__ == "__main__":
    raise SystemExit(main())
