#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from scripts.ops.long_runtime_common import iso_now, load_json, ordered_unique, write_payload
else:
    from .long_runtime_common import PROJECT_ROOT, iso_now, load_json, ordered_unique, write_payload


DEFAULT_OUT_PATH = PROJECT_ROOT / "governance" / "health" / "drainer_intelligence_layer_latest.json"
DEFAULT_CONTEXT_PATH = PROJECT_ROOT / "governance" / "health" / "drainer_intelligence_context_latest.json"
DEFAULT_NEEDS_PATH = PROJECT_ROOT / "governance" / "health" / "backlog_drain_needs_latest.json"
DEFAULT_FIX_LEDGER_PATH = PROJECT_ROOT / "governance" / "system_intelligence" / "backlog_drain_fix_ledger.jsonl"
DEFAULT_TARGET_PENDING_LINES = 10_000
GRADE_ORDER = ("F", "D", "C", "B", "A", "A+", "A++")
GRADE_TARGETS = {"F": 45.0, "D": 60.0, "C": 75.0, "B": 90.0, "A": 97.0, "A+": 99.0, "A++": 100.0}


def _safe_int(raw: Any, default: int = 0) -> int:
    try:
        return int(float(raw))
    except Exception:
        return int(default)


def _safe_float(raw: Any, default: float = 0.0) -> float:
    try:
        return float(raw)
    except Exception:
        return float(default)


def _status(payload: dict[str, Any], default: str = "missing") -> str:
    text = str(payload.get("overall_status") or payload.get("status") or "").strip().lower()
    return text or default


def _nested(payload: dict[str, Any], key: str) -> dict[str, Any]:
    value = payload.get(key)
    return value if isinstance(value, dict) else {}


def _candidate_drainers(fleet: dict[str, Any]) -> list[dict[str, Any]]:
    rows = fleet.get("candidate_drainers") if isinstance(fleet.get("candidate_drainers"), list) else []
    return [row for row in rows if isinstance(row, dict)]


def _active_drainer(fleet: dict[str, Any], super_drainer: dict[str, Any]) -> dict[str, Any]:
    active = fleet.get("active_drainer") if isinstance(fleet.get("active_drainer"), dict) else {}
    active_name = str(super_drainer.get("active_drainer") or active.get("name") or "").strip()
    if active and str(active.get("name") or "") == active_name:
        return active
    for row in _candidate_drainers(fleet):
        if str(row.get("name") or "") == active_name:
            return row
    return active if active else {"name": active_name} if active_name else {}


def _total_pending_lines(fleet: dict[str, Any], super_drainer: dict[str, Any], storage: dict[str, Any]) -> int:
    storage_bp = storage.get("backpressure") if isinstance(storage.get("backpressure"), dict) else {}
    storage_overlay = storage.get("sql_ingestion_pending_overlay") if isinstance(storage.get("sql_ingestion_pending_overlay"), dict) else {}
    super_summary = super_drainer.get("summary") if isinstance(super_drainer.get("summary"), dict) else {}
    metrics = fleet.get("metrics") if isinstance(fleet.get("metrics"), dict) else {}
    storage_current_total = max(
        _safe_int(storage_bp.get("total_pending_lines"), 0),
        _safe_int(storage.get("pending_lines_total"), 0),
        _safe_int(storage_overlay.get("total_pending_lines"), 0),
    )
    if storage_current_total > 0:
        return storage_current_total
    current_total = max(
        _safe_int(metrics.get("total_pending_lines"), 0),
        _safe_int(super_summary.get("final_pending_lines"), 0),
    )
    if current_total > 0:
        return current_total
    return _safe_int(super_summary.get("initial_pending_lines"), 0)


def _writer_state_from_payload(writer: dict[str, Any]) -> dict[str, Any]:
    for key in ("writer_state_after_wait", "writer_state_after_remediation", "writer_state_before"):
        nested = writer.get(key) if isinstance(writer.get(key), dict) else {}
        if nested:
            return nested
    return writer


def _writer_active(fleet: dict[str, Any], super_drainer: dict[str, Any], writer: dict[str, Any]) -> bool:
    super_writer = super_drainer.get("writer_state_before") if isinstance(super_drainer.get("writer_state_before"), dict) else {}
    writer_state = _writer_state_from_payload(writer)
    current_writer_authoritative = bool(
        "active" in writer_state
        or "progress_orphaned" in writer_state
        or "writer_lock_held" in writer_state
        or "writer_owner_pid_live" in writer_state
    )
    if current_writer_authoritative:
        return bool(
            fleet.get("writer_lock_held", False)
            or fleet.get("writer_active", False)
            or writer_state.get("active", False)
        )
    return bool(
        fleet.get("writer_lock_held", False)
        or fleet.get("writer_active", False)
        or super_writer.get("active", False)
        or writer_state.get("active", False)
    )


def _memory_pressure_high(memory_efficiency: dict[str, Any], runtime: dict[str, Any]) -> bool:
    memory_snapshot = memory_efficiency.get("memory_snapshot") if isinstance(memory_efficiency.get("memory_snapshot"), dict) else {}
    state = str(memory_snapshot.get("memory_pressure_state") or "").strip().lower()
    kind = str(memory_snapshot.get("memory_pressure_kind") or "").strip().lower()
    runtime_level = str(runtime.get("memory_pressure_level") or "").strip().lower()
    memory_calm = state in {"", "green", "normal"} and kind in {"", "none", "green", "normal"} and runtime_level in {"", "normal", "green"}
    if memory_calm:
        return False
    return bool(
        _status(memory_efficiency) in {"blocked", "critical", "degraded"}
        or state in {"yellow", "red", "warning", "critical"}
        or kind not in {"", "none", "green", "normal"}
        or runtime_level in {"high", "critical"}
    )


def _runtime_pressure_high(runtime: dict[str, Any]) -> bool:
    compute_level = str(runtime.get("compute_pressure_level") or "").strip().lower()
    host_saturation = _safe_float(runtime.get("host_saturation_score"), 0.0)
    runtime_status = _status(runtime)
    memory_level = str(runtime.get("memory_pressure_level") or "").strip().lower()
    soft_degraded_but_backlog_safe = bool(
        runtime_status == "degraded"
        and compute_level in {"", "normal", "low", "elevated"}
        and memory_level in {"", "normal", "green"}
        and 0.0 < host_saturation <= 50.0
    )
    if soft_degraded_but_backlog_safe:
        return False
    return bool(
        runtime_status in {"blocked", "critical", "degraded"}
        or compute_level in {"high", "critical"}
        or host_saturation >= 80.0
    )


def _recent_memory(memory: dict[str, Any]) -> dict[str, Any]:
    latest_event = memory.get("latest_event") if isinstance(memory.get("latest_event"), dict) else {}
    latest_net_change = _safe_int(
        latest_event.get("pending_lines_net_change"),
        _safe_int(latest_event.get("final_pending_lines"), 0) - _safe_int(latest_event.get("initial_pending_lines"), 0),
    )
    latest_waves = _safe_int(latest_event.get("waves_run"), 0)
    latest_progress_waves = _safe_int(latest_event.get("progress_waves"), 0)
    latest_refill = bool(latest_event.get("refill_detected", False)) or bool(
        latest_waves > 0
        and _safe_int(latest_event.get("final_pending_lines"), 0) > _safe_int(latest_event.get("initial_pending_lines"), 0)
    )
    latest_no_visible_progress = bool(
        latest_waves > 0
        and latest_progress_waves > 0
        and latest_net_change == 0
        and not bool(latest_event.get("target_met", False))
    )
    return {
        "history_count": _safe_int(memory.get("history_count"), 0),
        "recent_progress_rate": _safe_float(memory.get("recent_progress_rate"), 0.0),
        "recent_target_met_rate": _safe_float(memory.get("recent_target_met_rate"), 0.0),
        "recent_refill_rate": _safe_float(memory.get("recent_refill_rate"), 0.0),
        "latest_refill_detected": latest_refill,
        "latest_pending_lines_net_change": latest_net_change,
        "latest_progress_waves": latest_progress_waves,
        "latest_no_visible_pending_progress": latest_no_visible_progress,
        "latest_event": latest_event,
    }


def _writer_health(fleet: dict[str, Any], super_drainer: dict[str, Any], writer: dict[str, Any]) -> dict[str, Any]:
    super_writer = super_drainer.get("writer_state_before") if isinstance(super_drainer.get("writer_state_before"), dict) else {}
    writer_state = _writer_state_from_payload(writer)
    writer_summary = writer.get("summary") if isinstance(writer.get("summary"), dict) else {}
    active = _writer_active(fleet, super_drainer, writer)
    progress_orphaned = bool(writer_state.get("progress_orphaned", False) or super_writer.get("progress_orphaned", False))
    progress_age = max(
        _safe_float(writer_state.get("progress_age_minutes"), 0.0),
        _safe_float(super_writer.get("progress_age_minutes"), 0.0),
    )
    cycle_age = max(
        _safe_float(writer_state.get("cycle_age_minutes"), 0.0),
        _safe_float(super_writer.get("cycle_age_minutes"), 0.0),
    )
    merged_rows = max(
        _safe_int(writer_state.get("merged_rows_this_cycle"), 0),
        _safe_int(super_writer.get("merged_rows_this_cycle"), 0),
    )
    timed_out_shards = max(
        _safe_int(writer_state.get("timed_out_shard_count"), 0),
        _safe_int(super_writer.get("timed_out_shard_count"), 0),
    )
    completed_merges = max(
        _safe_int(writer_state.get("completed_merge_count"), 0),
        _safe_int(super_writer.get("completed_merge_count"), 0),
    )
    completed_shards = max(
        _safe_int(writer_state.get("completed_shard_count"), 0),
        _safe_int(super_writer.get("completed_shard_count"), 0),
    )
    planned_shards = max(
        _safe_int(writer_state.get("planned_shard_count"), 0),
        _safe_int(super_writer.get("planned_shard_count"), 0),
    )
    if progress_orphaned:
        state = "orphaned_progress"
    elif not active:
        state = "idle"
    elif progress_age >= 75.0 and merged_rows <= 0:
        state = "stalled"
    elif progress_age >= 45.0:
        state = "stale_progress"
    else:
        state = "active_progressing"
    return {
        "state": state,
        "active": bool(active),
        "progress_age_minutes": round(progress_age, 3),
        "cycle_age_minutes": round(cycle_age, 3),
        "merged_rows_this_cycle": int(merged_rows),
        "completed_merge_count": int(completed_merges),
        "completed_shard_count": int(completed_shards),
        "planned_shard_count": int(planned_shards),
        "timed_out_shard_count": int(timed_out_shards),
        "progress_orphaned": bool(progress_orphaned),
        "active_source": str(writer_state.get("active_source") or super_writer.get("active_source") or ""),
        "writer_owner_pid_live": bool(writer_state.get("writer_owner_pid_live", False) or super_writer.get("writer_owner_pid_live", False)),
        "current_step": str(writer_state.get("current_step") or super_writer.get("current_step") or ""),
        "writer_lock_owner": str(fleet.get("writer_lock_owner") or super_writer.get("writer_lock_owner") or writer_state.get("writer_lock_owner") or ""),
        "process_action": str(writer_summary.get("writer_process_action") or ""),
    }


def _lane_family(name: str, pressure_lane: str = "") -> str:
    text = f"{name} {pressure_lane}".lower()
    families = (
        ("core_decision", ("core_decision", "decision_channel", "trading")),
        ("governance_telemetry", ("governance_execution", "governance_telemetry", "governance_journal")),
        ("derivatives", ("derivatives", "options", "futures", "greeks")),
        ("market_data", ("market_data", "provider", "quote", "source_verification")),
        ("macro_event", ("macro", "earnings", "sentiment", "stress_scenario")),
        ("predictive_stability", ("predictive", "stability", "forecast", "trajectory")),
        ("self_healing", ("self_healing", "recovery", "blackstart", "autofix")),
        ("collector_utility", ("collector_utility", "collector_budget", "collection_value")),
        ("hot_path_storage", ("hot_path", "storage_budget", "watermark", "write_budget")),
        ("admission_evidence", ("admission", "sample_depth", "walk_forward", "teacher_lineage")),
        ("writer_progress", ("writer_progress", "writer_cycle", "sql_link", "jsonl_sql_writer")),
        ("model_research", ("model_research", "retrain", "champion", "training")),
        ("runtime_memory", ("memory_runtime", "runtime_channel", "runtime_artifact")),
        ("data_quality", ("data_quality", "schema", "contract", "entitlement")),
        ("settlement", ("settlement", "reconciliation", "portfolio_ledger")),
        ("support_alerts", ("support", "alert", "watchdog", "pager", "incident")),
        ("reports", ("report", "cockpit", "showcase", "presentation")),
        ("cold_stage", ("cold_stage", "explanation", "stale_stage")),
    )
    for family, needles in families:
        if any(needle in text for needle in needles):
            return family
    return "other"


def _overlay_family_for_pending(row: dict[str, Any]) -> str:
    source_rel = str(row.get("source_rel") or "").strip().lower()
    shard = str(row.get("shard") or "").strip().lower()
    pressure_lane = str(row.get("pressure_lane") or "").strip().lower()
    stream = str(row.get("stream") or "").strip().lower()
    if pressure_lane == "core" or source_rel.startswith("decisions/") or "governance/channels/decision/" in source_rel:
        return "core_decision"
    if (
        pressure_lane == "support"
        or shard == "governance"
        or source_rel.startswith("governance/health/")
        or source_rel.startswith("governance/events/")
        or "jsonl_ingest_batch_journal" in source_rel
    ):
        if "support_watchdog" in source_rel or stream == "governance_watchdog":
            return "support_alerts"
        return "governance_telemetry"
    if pressure_lane == "deferred":
        return "data_quality"
    if pressure_lane == "cold":
        return "cold_stage"
    return "hot_path_storage"


def _storage_overlay_family_context(storage: dict[str, Any]) -> dict[str, Any]:
    overlay = storage.get("sql_ingestion_pending_overlay") if isinstance(storage.get("sql_ingestion_pending_overlay"), dict) else {}
    if not overlay or not bool(overlay.get("active", False)):
        return {"active": False, "used_for_pressure": False, "family_pending_lines": {}, "top_pending_files": []}

    family_pending: dict[str, int] = {}
    lane_top_totals: dict[str, int] = {}
    top_files = overlay.get("top_pending_files") if isinstance(overlay.get("top_pending_files"), list) else []
    for row in top_files:
        if not isinstance(row, dict):
            continue
        pending = _safe_int(row.get("pending_lines"), 0)
        if pending <= 0:
            continue
        family = _overlay_family_for_pending(row)
        pressure_lane = str(row.get("pressure_lane") or "").strip().lower()
        family_pending[family] = family_pending.get(family, 0) + pending
        lane_top_totals[pressure_lane] = lane_top_totals.get(pressure_lane, 0) + pending

    lane_to_family = {
        "core": "core_decision",
        "support": "governance_telemetry",
        "deferred": "data_quality",
        "cold": "cold_stage",
    }
    lane_keys = {
        "core": "core_pending_lines",
        "support": "support_pending_lines",
        "deferred": "deferred_pending_lines",
        "cold": "cold_pending_lines",
    }
    for lane, key in lane_keys.items():
        lane_total = _safe_int(overlay.get(key), 0)
        if lane_total <= 0:
            continue
        remainder = max(lane_total - lane_top_totals.get(lane, 0), 0)
        if remainder > 0:
            family = lane_to_family[lane]
            family_pending[family] = family_pending.get(family, 0) + remainder

    return {
        "active": bool(family_pending),
        "used_for_pressure": bool(overlay.get("used_for_pressure", False)),
        "family_pending_lines": family_pending,
        "total_pending_lines": _safe_int(overlay.get("total_pending_lines"), 0),
        "core_pending_lines": _safe_int(overlay.get("core_pending_lines"), 0),
        "support_pending_lines": _safe_int(overlay.get("support_pending_lines"), 0),
        "deferred_pending_lines": _safe_int(overlay.get("deferred_pending_lines"), 0),
        "cold_pending_lines": _safe_int(overlay.get("cold_pending_lines"), 0),
        "invalid_lines": _safe_int(overlay.get("invalid_lines"), 0),
        "oversize_payloads": _safe_int(overlay.get("oversize_payloads"), 0),
        "ops_write_failures": _safe_int(overlay.get("ops_write_failures"), 0),
        "top_pending_files": top_files[:8],
    }


def _augment_candidates_with_storage_overlay(candidates: list[dict[str, Any]], overlay_context: dict[str, Any]) -> list[dict[str, Any]]:
    family_pending = overlay_context.get("family_pending_lines") if isinstance(overlay_context.get("family_pending_lines"), dict) else {}
    if not family_pending:
        return [dict(row) for row in candidates]

    best_name_by_family: dict[str, str] = {}
    best_score_by_family: dict[str, tuple[int, int]] = {}
    for row in candidates:
        family = _lane_family(str(row.get("name") or ""), str(row.get("assigned_pressure_lane") or ""))
        if _safe_int(family_pending.get(family), 0) <= 0:
            continue
        score = (_safe_int(row.get("priority_score"), 0), _safe_int(row.get("pending_lines"), 0))
        if family not in best_score_by_family or score > best_score_by_family[family]:
            best_score_by_family[family] = score
            best_name_by_family[family] = str(row.get("name") or "")

    augmented: list[dict[str, Any]] = []
    for row in candidates:
        candidate = dict(row)
        family = _lane_family(str(candidate.get("name") or ""), str(candidate.get("assigned_pressure_lane") or ""))
        overlay_pending = _safe_int(family_pending.get(family), 0) if str(candidate.get("name") or "") == best_name_by_family.get(family, "") else 0
        if overlay_pending > 0:
            raw_pending = _safe_int(candidate.get("pending_lines"), 0)
            candidate["raw_pending_lines"] = raw_pending
            candidate["pending_lines"] = max(raw_pending, overlay_pending)
            candidate["storage_overlay_pending_lines"] = overlay_pending
            candidate["storage_overlay_active"] = True
            if str(candidate.get("status") or "").strip().lower() in {"", "idle", "watch"}:
                candidate["status"] = "ready"
                candidate["readiness_reason"] = "storage_overlay_pending"
        augmented.append(candidate)
    return augmented


def _lane_family_summary(scores: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_family: dict[str, dict[str, Any]] = {}
    for row in scores:
        family = str(row.get("family") or _lane_family(str(row.get("name") or ""), str(row.get("assigned_pressure_lane") or "")))
        current = by_family.setdefault(
            family,
            {
                "family": family,
                "ready_count": 0,
                "lane_count": 0,
                "pending_lines": 0,
                "utility_score": 0.0,
                "top_lane": "",
            },
        )
        current["lane_count"] = _safe_int(current.get("lane_count"), 0) + 1
        if str(row.get("status") or "") == "ready":
            current["ready_count"] = _safe_int(current.get("ready_count"), 0) + 1
        current["pending_lines"] = _safe_int(current.get("pending_lines"), 0) + _safe_int(row.get("pending_lines"), 0)
        current["utility_score"] = round(_safe_float(current.get("utility_score"), 0.0) + _safe_float(row.get("utility_score"), 0.0), 3)
        if not current.get("top_lane") or _safe_float(row.get("utility_score"), 0.0) > _safe_float(current.get("top_lane_score"), -1.0):
            current["top_lane"] = str(row.get("name") or "")
            current["top_lane_score"] = _safe_float(row.get("utility_score"), 0.0)
    rows = list(by_family.values())
    for row in rows:
        row.pop("top_lane_score", None)
    return sorted(rows, key=lambda row: (_safe_float(row.get("utility_score"), 0.0), _safe_int(row.get("pending_lines"), 0)), reverse=True)


def _pressure_forecast(memory: dict[str, Any], total_pending_lines: int, target_pending_lines: int, writer_health: dict[str, Any]) -> dict[str, Any]:
    history = memory.get("history") if isinstance(memory.get("history"), list) else []
    recent = [row for row in history[-8:] if isinstance(row, dict)]
    deltas = [_safe_int(row.get("pending_lines_delta"), 0) for row in recent]
    positive = [delta for delta in deltas if delta > 0]
    median_like = sorted(positive)[len(positive) // 2] if positive else 0
    remaining = max(int(total_pending_lines) - int(target_pending_lines), 0)
    waves_to_target = 0 if remaining <= 0 else (1 + ((remaining - 1) // max(int(median_like), 1))) if median_like > 0 else None
    if remaining <= 0:
        trajectory = "target_met"
    elif str(writer_health.get("state") or "") in {"stale_progress", "stalled"}:
        trajectory = "blocked_by_writer_progress"
    elif not positive:
        trajectory = "flat_or_unknown"
    else:
        trajectory = "clearing"
    return {
        "trajectory": trajectory,
        "remaining_pending_lines": int(remaining),
        "typical_progress_rows": int(median_like),
        "estimated_waves_to_target": waves_to_target,
        "history_points": len(recent),
    }


def _risk_flags(
    *,
    fleet: dict[str, Any],
    super_drainer: dict[str, Any],
    storage: dict[str, Any],
    runtime: dict[str, Any],
    memory_efficiency: dict[str, Any],
    writer: dict[str, Any],
    memory: dict[str, Any],
    total_pending_lines: int,
    target_pending_lines: int,
    writer_health: dict[str, Any],
) -> list[str]:
    risks: list[str] = []
    recent = _recent_memory(memory)
    if _writer_active(fleet, super_drainer, writer):
        risks.append("writer_active")
    if str(writer_health.get("state") or "") == "orphaned_progress":
        risks.append("writer_progress_orphaned")
    if str(writer_health.get("state") or "") == "stale_progress":
        risks.append("writer_progress_stale")
    if str(writer_health.get("state") or "") == "stalled":
        risks.append("writer_progress_stalled")
    if total_pending_lines > target_pending_lines:
        risks.append("target_not_met")
    if _status(storage) in {"blocked", "critical"} or str(storage.get("severity") or "").strip().lower() == "critical":
        risks.append("storage_critical")
    if _runtime_pressure_high(runtime):
        risks.append("runtime_pressure_high")
    if _memory_pressure_high(memory_efficiency, runtime):
        risks.append("memory_pressure_high")
    if _safe_int(fleet.get("ready_drainer_count"), 0) <= 0 and total_pending_lines > target_pending_lines:
        risks.append("no_ready_drainers")
    if _safe_int(recent.get("history_count"), 0) >= 3 and _safe_float(recent.get("recent_progress_rate"), 0.0) < 0.25:
        risks.append("recent_progress_rate_low")
    if bool(recent.get("latest_refill_detected", False)) and total_pending_lines > target_pending_lines:
        risks.append("recent_refill_after_drain")
    if bool(recent.get("latest_no_visible_pending_progress", False)) and total_pending_lines > target_pending_lines:
        risks.append("visible_pending_progress_missing")
    if str(super_drainer.get("stop_reason") or _nested(super_drainer, "summary").get("stop_reason") or "") == "progress_stalled":
        risks.append("progress_stalled")
    if "market_hours_guard" in list(fleet.get("blocked_reasons") or []):
        risks.append("market_hours_guard")
    return ordered_unique(risks)


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(float(lo), min(float(hi), float(value)))


def _confidence(*, risks: list[str], fleet: dict[str, Any], super_drainer: dict[str, Any], memory: dict[str, Any]) -> float:
    score = 0.45
    if _candidate_drainers(fleet):
        score += 0.18
    if _safe_int(fleet.get("ready_drainer_count"), 0) > 0:
        score += 0.12
    if super_drainer:
        score += 0.08
    if _safe_int(memory.get("history_count"), 0) > 0:
        score += 0.08
    if "recent_progress_rate_low" in risks:
        score -= 0.12
    if "writer_progress_stale" in risks:
        score -= 0.06
    if "writer_progress_stalled" in risks:
        score -= 0.16
    if "no_ready_drainers" in risks:
        score -= 0.2
    if "memory_pressure_high" in risks or "runtime_pressure_high" in risks:
        score -= 0.08
    if "recent_refill_after_drain" in risks:
        score -= 0.08
    return round(_clamp(score, 0.1, 0.95), 3)


def _metric_score(value: float, *, green: float, warning: float, critical: float) -> float:
    value = max(float(value), 0.0)
    green = max(float(green), 0.0)
    warning = max(float(warning), green + 1.0)
    critical = max(float(critical), warning + 1.0)
    if value <= green:
        return 100.0
    if value <= warning:
        return 100.0 - ((value - green) / (warning - green)) * 20.0
    if value <= critical:
        return 80.0 - ((value - warning) / (critical - warning)) * 35.0
    return 45.0 - _clamp((value - critical) / max(critical, 1.0), 0.0, 1.0) * 35.0


def _grade_from_score(score: float) -> str:
    if score >= 99.0:
        return "A++"
    if score >= 97.0:
        return "A+"
    if score >= 90.0:
        return "A"
    if score >= 75.0:
        return "B"
    if score >= 60.0:
        return "C"
    if score >= 45.0:
        return "D"
    return "F"


def _severity_from_score(score: float) -> str:
    if score >= 90.0:
        return "clear"
    if score >= 75.0:
        return "stable"
    if score >= 60.0:
        return "strained"
    if score >= 45.0:
        return "degraded"
    return "critical"


def _next_grade(grade: str) -> str:
    normalized = str(grade or "F").strip().upper()
    try:
        index = GRADE_ORDER.index(normalized)
    except ValueError:
        index = 0
    if index >= len(GRADE_ORDER) - 1:
        return "A++"
    return GRADE_ORDER[index + 1]


def _score_target_for_grade(grade: str) -> float:
    return float(GRADE_TARGETS.get(str(grade or "").strip().upper(), 45.0))


def _command_for_steps(playbook: list[dict[str, Any]], preferred_steps: set[str]) -> list[str]:
    for row in playbook:
        if not isinstance(row, dict):
            continue
        if str(row.get("step") or "") in preferred_steps and isinstance(row.get("command"), list):
            return [str(item) for item in row.get("command", [])]
    for row in playbook:
        if isinstance(row, dict) and isinstance(row.get("command"), list):
            return [str(item) for item in row.get("command", [])]
    return ["./scripts/ops/opsctl.sh", "drainer-intelligence-layer", "--json"]


def _a_grade_exit_criteria(section_id: str, fallback: list[str]) -> list[str]:
    criteria = {
        "core_decision": [
            "core_pending_lines <= 5000",
            "total_pending_lines <= 10000",
            "oldest_pending_age_seconds <= 3600",
            "writer_state is idle or active_progressing without stale progress",
        ],
        "crypto_sparse_decision": [
            "sparse_pending_lines <= 250",
            "sparse_estimated_pending_bytes <= 536870912",
            "oldest_pending_age_seconds <= 3600",
            "latest refill check is false after the capped wave",
        ],
        "runtime_capacity": [
            "host_saturation_score <= 50",
            "compute_pressure_level is normal or low",
            "memory_pressure_level is normal",
            "foreground computer-task governor remains active while user apps are open",
        ],
        "writer_merge_health": [
            "writer_state is idle or active_progressing",
            "progress_age_minutes < 15 when active",
            "timed_out_shard_count == 0",
        ],
        "deferred_data_quality": [
            "deferred_pending_lines <= 10000",
            "core_pending_lines <= 5000",
        ],
        "support_watchdog": [
            "support_pending_lines <= 1000",
            "support telemetry remains shard-isolated",
        ],
        "provider_market_data": [
            "provider_pending_lines <= 250",
            "provider_oldest_pending_age_seconds <= 300",
            "source verification is refreshed separately",
        ],
    }
    return criteria.get(str(section_id or ""), [*fallback, "section_score >= 90.0", "section_grade == A"])


def _section_need_profile(section: dict[str, Any], playbook: list[dict[str, Any]], risks: list[str]) -> dict[str, Any]:
    section_id = str(section.get("section_id") or "")
    grade = str(section.get("grade") or "F")
    next_grade = _next_grade(grade)
    target_score = _score_target_for_grade(next_grade)
    score = _safe_float(section.get("score"), 0.0)
    score_gap = round(max(target_score - score, 0.0), 1)
    pending = _safe_int(section.get("pending_lines"), 0)
    target_pending = _safe_int(section.get("target_pending_lines"), 0)
    oldest_age = _safe_float(section.get("oldest_pending_age_seconds"), 0.0)
    evidence = [str(item) for item in section.get("evidence", []) if str(item).strip()] if isinstance(section.get("evidence"), list) else []

    command = _command_for_steps(playbook, {"re_score_before_next_wave", "re_score_drainers", "run_selected_lane", "micro_drain"})
    specific_need = "raise score above the next grade threshold"
    exit_criteria: list[str] = [f"section_score >= {target_score:.1f}", f"section_grade >= {next_grade}"]
    measurements: list[str] = ["pending_lines", "oldest_pending_age_seconds", "section_score"]
    owner = "backpressure_storage_brain_v2"
    accelerator = "drainer_intelligence_layer"

    if section_id == "core_decision":
        specific_need = "drain core decision pending lines and reduce the oldest pending age before widening or training"
        exit_criteria = [
            "core_pending_lines <= 15000 for a C grade",
            "core_pending_lines <= 5000 plus oldest_pending_age_seconds <= 3600 for a B/A posture",
            "writer_state is idle or active_progressing without stale progress",
        ]
        measurements = ["core_pending_lines", "total_pending_lines", "oldest_pending_age_seconds", "writer_state"]
        command = _command_for_steps(playbook, {"run_selected_lane", "micro_drain", "wait_for_single_writer", "inspect_writer"})
        accelerator = "core_decision_drainer"
    elif section_id == "crypto_sparse_decision":
        specific_need = "finish the sparse-large crypto decision tail with small capped writer waves"
        exit_criteria = [
            "sparse_pending_lines <= 700 for a B attempt and <= 250 for green",
            "sparse_estimated_pending_bytes <= 536870912",
            "no refill is detected after the capped wave",
        ]
        measurements = ["sparse_pending_lines", "sparse_estimated_pending_bytes", "top_file", "latest_refill_detected"]
        command = _command_for_steps(playbook, {"micro_drain", "run_selected_lane", "re_score_before_next_wave"})
        accelerator = "sparse_large_line_accelerator"
    elif section_id == "runtime_capacity":
        specific_need = "cool runtime before asking the drainer or training lanes to do heavier work"
        exit_criteria = [
            "compute_pressure_level is not high",
            "host_saturation_score <= 60, ideally <= 50",
            "memory pressure remains green/normal",
        ]
        measurements = ["host_saturation_score", "compute_pressure_level", "memory_pressure_level", "memory_pressure_state"]
        command = _command_for_steps(playbook, {"pressure_relief", "runtime_throttle"})
        owner = "adaptive_resource_governor"
        accelerator = "pressure_relief_control"
    elif section_id == "writer_merge_health":
        specific_need = "keep the single writer moving without stale progress or orphaned locks"
        exit_criteria = [
            "writer_state in idle or active_progressing",
            "progress_age_minutes < 45 when active",
            "timed_out_shard_count is stable or falling",
        ]
        measurements = ["writer_state", "progress_age_minutes", "completed_merge_count", "timed_out_shard_count"]
        command = ["./scripts/ops/opsctl.sh", "writer-cycle-coordinator", "--json"]
        owner = "writer_cycle_coordinator"
        accelerator = "single_writer_coordinator"
    elif section_id == "deferred_data_quality":
        specific_need = "leave deferred work throttled until core pressure is no longer the bottleneck"
        exit_criteria = ["deferred_pending_lines <= 10000", "core_decision grade >= C"]
        measurements = ["deferred_pending_lines", "core_pending_lines"]
        accelerator = "deferred_lane_accelerator"
    elif section_id == "support_watchdog":
        specific_need = "keep support telemetry shard-isolated so it cannot crowd the core writer"
        exit_criteria = ["support_pending_lines <= 1000", "writer_shedding keeps support telemetry active"]
        measurements = ["support_pending_lines", "writer_shedding.level"]
        accelerator = "support_telemetry_shedder"
    elif section_id == "provider_market_data":
        specific_need = "queue provider spillover only after core and sparse pressure cool"
        exit_criteria = ["provider_pending_lines <= 250", "source verification is refreshed separately"]
        measurements = ["provider_pending_lines", "provider_oldest_pending_age_seconds", "provider_top_lane"]
        owner = "provider_adapter_verification"
        accelerator = "provider_lane_drainer"

    if "recent_refill_after_drain" in risks and section_id in {"core_decision", "crypto_sparse_decision"}:
        specific_need += "; current refill evidence says intake must be tightened before another larger wave"
        exit_criteria.append("latest refill check is false after pressure relief and re-score")
    if "visible_pending_progress_missing" in risks and section_id in {"core_decision", "crypto_sparse_decision"}:
        specific_need += "; the last wave made writer progress but pending did not visibly drop, so measurement refresh is required"
        exit_criteria.append("next storage refresh shows pending_lines_delta < 0 or the drainer fleet identifies the unchanged source lane")
        measurements = ordered_unique([*measurements, "pending_lines_delta", "progress_waves", "source_lane_pending_lines"])

    return {
        "section_id": section_id,
        "label": str(section.get("label") or ""),
        "current_grade": grade,
        "current_score": round(score, 1),
        "next_grade": next_grade,
        "target_score": round(target_score, 1),
        "score_gap": score_gap,
        "a_grade_target_score": 90.0,
        "a_grade_score_gap": round(max(90.0 - score, 0.0), 1),
        "pending_lines": pending,
        "target_pending_lines": target_pending,
        "oldest_pending_age_seconds": round(oldest_age, 3),
        "what_it_needs": specific_need,
        "measurements_to_check": measurements,
        "exit_criteria": exit_criteria,
        "a_grade_exit_criteria": _a_grade_exit_criteria(section_id, exit_criteria),
        "recommended_command": command,
        "system_owner": owner,
        "accelerator": accelerator,
        "priority_weight": round(_safe_float(section.get("weight"), 0.0), 3),
        "evidence": evidence,
        "frame_of_reference": {
            "source": str(section.get("source") or ""),
            "primary_issue": str(section.get("primary_issue") or ""),
            "recommended_next_action": str(section.get("recommended_next_action") or ""),
            "observed_grade": grade,
            "observed_score": round(score, 1),
            "target_grade": next_grade,
            "a_grade_target_score": 90.0,
        },
    }


def _backlog_needs_packet(
    *,
    timestamp_utc: str,
    scorecard: dict[str, Any],
    decision_packet: dict[str, Any],
    playbook: list[dict[str, Any]],
    risks: list[str],
    recent: dict[str, Any],
    writer_health: dict[str, Any],
) -> dict[str, Any]:
    sections = [row for row in scorecard.get("sections", []) if isinstance(row, dict)] if isinstance(scorecard.get("sections"), list) else []
    needs = [
        _section_need_profile(row, playbook, risks)
        for row in sections
        if str(row.get("grade") or "A") not in {"A", "A+", "A++"}
    ]
    needs = sorted(
        needs,
        key=lambda row: (
            GRADE_ORDER.index(str(row.get("current_grade") or "F")) if str(row.get("current_grade") or "F") in GRADE_ORDER else 0,
            -_safe_float(row.get("priority_weight"), 0.0),
            -_safe_float(row.get("score_gap"), 0.0),
        ),
    )
    overall_grade = str(scorecard.get("overall_grade") or "F")
    next_overall_grade = _next_grade(overall_grade)
    target_score = _score_target_for_grade(next_overall_grade)
    overall_score = _safe_float(scorecard.get("overall_score"), 0.0)
    next_commands = []
    for row in playbook:
        if isinstance(row, dict) and isinstance(row.get("command"), list):
            next_commands.append(
                {
                    "step": str(row.get("step") or ""),
                    "command": [str(item) for item in row.get("command", [])],
                }
            )

    top_need = needs[0] if needs else {}
    status = "clear" if not needs and overall_grade in {"A", "A+", "A++"} else "needs_attention"
    a_grade_needs = [
        {
            "section_id": str(row.get("section_id") or ""),
            "current_grade": str(row.get("current_grade") or ""),
            "current_score": _safe_float(row.get("current_score"), 0.0),
            "a_grade_score_gap": _safe_float(row.get("a_grade_score_gap"), 0.0),
            "a_grade_exit_criteria": list(row.get("a_grade_exit_criteria") or []),
            "recommended_command": list(row.get("recommended_command") or []),
        }
        for row in needs
    ]
    return {
        "timestamp_utc": timestamp_utc,
        "schema_version": 1,
        "overall_status": status,
        "current_grade": overall_grade,
        "current_score": round(overall_score, 1),
        "next_grade": next_overall_grade,
        "target_score": round(target_score, 1),
        "score_gap": round(max(target_score - overall_score, 0.0), 1),
        "a_grade_target_score": 90.0,
        "a_grade_score_gap": round(max(90.0 - overall_score, 0.0), 1),
        "top_need_section": str(top_need.get("section_id") or ""),
        "top_need": str(top_need.get("what_it_needs") or ""),
        "blocking_sections": [str(row.get("section_id") or "") for row in needs[:5]],
        "needs": needs,
        "a_grade_lift_contract": {
            "target_grade": "A",
            "target_score": 90.0,
            "score_gap": round(max(90.0 - overall_score, 0.0), 1),
            "blocking_sections": [str(row.get("section_id") or "") for row in a_grade_needs],
            "needs": a_grade_needs,
        },
        "next_command_sequence": next_commands,
        "troubleshooting_summary": {
            "action": str(decision_packet.get("action") or ""),
            "selected_drainer": str(decision_packet.get("selected_drainer") or ""),
            "total_pending_lines": _safe_int(decision_packet.get("total_pending_lines"), 0),
            "target_pending_lines": _safe_int(decision_packet.get("target_pending_lines"), 0),
            "adaptive_target_pending_lines": _safe_int(decision_packet.get("adaptive_target_pending_lines"), 0),
            "risk_flags": risks,
            "writer_state": str(writer_health.get("state") or ""),
            "latest_refill_detected": bool(recent.get("latest_refill_detected", False)),
            "latest_pending_lines_net_change": _safe_int(recent.get("latest_pending_lines_net_change"), 0),
            "latest_progress_waves": _safe_int(recent.get("latest_progress_waves"), 0),
            "latest_no_visible_pending_progress": bool(recent.get("latest_no_visible_pending_progress", False)),
        },
        "accelerator_contract": {
            "exact_measurements_required_each_cycle": ordered_unique(
                [
                    "core_pending_lines",
                    "total_pending_lines",
                    "oldest_pending_age_seconds",
                    "sparse_large_line_pending_lines",
                    "sparse_large_line_pending_bytes",
                    "writer_state",
                    "host_saturation_score",
                    "compute_pressure_level",
                    "latest_refill_detected",
                ]
            ),
            "log_fix_frames": True,
            "latest_needs_artifact": str(DEFAULT_NEEDS_PATH.relative_to(PROJECT_ROOT)),
            "fix_ledger_artifact": str(DEFAULT_FIX_LEDGER_PATH.relative_to(PROJECT_ROOT)),
            "verification_command": ["./scripts/ops/opsctl.sh", "drainer-intelligence-layer", "--json"],
        },
        "fix_reference_frame": {
            "frame_type": "backlog_drain_needs_snapshot",
            "before_grade": overall_grade,
            "target_grade": next_overall_grade,
            "a_grade_target_score": 90.0,
            "a_grade_score_gap": round(max(90.0 - overall_score, 0.0), 1),
            "before_score": round(overall_score, 1),
            "target_score": round(target_score, 1),
            "primary_blocker": str(top_need.get("section_id") or ""),
            "expected_evidence_after_fix": list(top_need.get("exit_criteria") or []) if top_need else [],
            "source_artifacts": [
                "governance/health/drainer_intelligence_layer_latest.json",
                "governance/health/ingestion_storage_control_latest.json",
                "governance/health/backpressure_drainer_fleet_latest.json",
                "governance/health/writer_cycle_coordinator_latest.json",
            ],
        },
    }


def _needs_fingerprint(packet: dict[str, Any]) -> str:
    stable = {
        "grade": packet.get("current_grade"),
        "score": packet.get("current_score"),
        "action": _nested(packet, "troubleshooting_summary").get("action"),
        "top_need_section": packet.get("top_need_section"),
        "blocking_sections": packet.get("blocking_sections"),
        "needs": [
            {
                "section_id": row.get("section_id"),
                "current_grade": row.get("current_grade"),
                "current_score": row.get("current_score"),
                "pending_lines": row.get("pending_lines"),
                "score_gap": row.get("score_gap"),
            }
            for row in packet.get("needs", [])
            if isinstance(row, dict)
        ],
    }
    encoded = json.dumps(stable, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:20]


def _last_jsonl_entry(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            lines = [line.strip() for line in handle if line.strip()]
    except Exception:
        return {}
    for raw in reversed(lines[-20:]):
        try:
            payload = json.loads(raw)
        except Exception:
            continue
        if isinstance(payload, dict):
            return payload
    return {}


def _append_fix_reference_if_changed(path: Path, payload: dict[str, Any]) -> None:
    packet = payload.get("backlog_needs_packet") if isinstance(payload.get("backlog_needs_packet"), dict) else {}
    if not packet:
        return
    fingerprint = _needs_fingerprint(packet)
    last = _last_jsonl_entry(path)
    if str(last.get("fingerprint") or "") == fingerprint:
        return
    entry = {
        "timestamp_utc": payload.get("timestamp_utc"),
        "schema_version": 1,
        "event_type": "backlog_drain_fix_reference_frame",
        "fingerprint": fingerprint,
        "current_grade": packet.get("current_grade"),
        "current_score": packet.get("current_score"),
        "target_grade": packet.get("next_grade"),
        "top_need_section": packet.get("top_need_section"),
        "top_need": packet.get("top_need"),
        "blocking_sections": packet.get("blocking_sections"),
        "risk_flags": _nested(packet, "troubleshooting_summary").get("risk_flags", []),
        "next_command_sequence": packet.get("next_command_sequence", [])[:4],
        "fix_reference_frame": packet.get("fix_reference_frame", {}),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, ensure_ascii=True, sort_keys=True) + "\n")


def _section_row(
    *,
    section_id: str,
    label: str,
    score: float,
    pending_lines: int = 0,
    target_pending_lines: int = 0,
    oldest_pending_age_seconds: float = 0.0,
    primary_issue: str,
    recommended_next_action: str,
    evidence: list[str] | None = None,
    weight: float,
    source: str,
) -> dict[str, Any]:
    clean_score = round(_clamp(score, 0.0, 100.0), 1)
    return {
        "section_id": section_id,
        "label": label,
        "grade": _grade_from_score(clean_score),
        "score": clean_score,
        "severity": _severity_from_score(clean_score),
        "pending_lines": int(max(pending_lines, 0)),
        "target_pending_lines": int(max(target_pending_lines, 0)),
        "oldest_pending_age_seconds": round(max(float(oldest_pending_age_seconds), 0.0), 3),
        "primary_issue": primary_issue,
        "recommended_next_action": recommended_next_action,
        "evidence": ordered_unique(evidence or []),
        "weight": round(float(weight), 3),
        "source": source,
    }


def _backpressure_value(storage: dict[str, Any], fleet: dict[str, Any], key: str) -> int:
    bp = storage.get("backpressure") if isinstance(storage.get("backpressure"), dict) else {}
    raw_live = bp.get("raw_live") if isinstance(bp.get("raw_live"), dict) else {}
    metrics = fleet.get("metrics") if isinstance(fleet.get("metrics"), dict) else {}
    storage_candidates = [_safe_int(bp.get(key), 0), _safe_int(raw_live.get(key), 0)]
    storage_value = max(storage_candidates)
    if key in bp or key in raw_live or storage_value > 0:
        return storage_value
    return _safe_int(metrics.get(key), 0)


def _backpressure_oldest_age(storage: dict[str, Any], fleet: dict[str, Any]) -> float:
    bp = storage.get("backpressure") if isinstance(storage.get("backpressure"), dict) else {}
    raw_live = bp.get("raw_live") if isinstance(bp.get("raw_live"), dict) else {}
    metrics = fleet.get("metrics") if isinstance(fleet.get("metrics"), dict) else {}
    return max(
        _safe_float(bp.get("oldest_pending_age_seconds"), 0.0),
        _safe_float(raw_live.get("oldest_pending_age_seconds"), 0.0),
        _safe_float(metrics.get("oldest_pending_age_seconds"), 0.0),
    )


def _sparse_large_context(storage: dict[str, Any], active: dict[str, Any], candidates: list[dict[str, Any]]) -> dict[str, Any]:
    bp = storage.get("backpressure") if isinstance(storage.get("backpressure"), dict) else {}
    raw_live = bp.get("raw_live") if isinstance(bp.get("raw_live"), dict) else {}
    line_estimation = raw_live.get("line_estimation") if isinstance(raw_live.get("line_estimation"), dict) else {}
    if not line_estimation:
        line_estimation = bp.get("line_estimation") if isinstance(bp.get("line_estimation"), dict) else {}
    storage_line_estimation_present = bool(line_estimation)

    pending = _safe_int(line_estimation.get("sparse_large_line_pending_lines"), 0)
    file_count = _safe_int(line_estimation.get("sparse_large_line_files"), 0)
    bytes_total = _safe_int(line_estimation.get("sparse_large_line_bytes"), 0)
    pending_bytes = _safe_int(line_estimation.get("sparse_large_line_pending_bytes"), 0)
    active_flag = bool(line_estimation.get("sparse_large_line_active", False))
    top_files: list[dict[str, Any]] = []

    for row in [active, *candidates]:
        pressure = row.get("sparse_large_line_pressure") if isinstance(row.get("sparse_large_line_pressure"), dict) else {}
        if not pressure:
            continue
        if not storage_line_estimation_present:
            pending = max(pending, _safe_int(pressure.get("pending_lines"), 0))
            file_count = max(file_count, _safe_int(pressure.get("file_count"), 0))
            bytes_total = max(bytes_total, _safe_int(pressure.get("file_size_bytes"), 0))
            pending_bytes = max(pending_bytes, _safe_int(pressure.get("estimated_pending_bytes"), 0))
        active_flag = active_flag or bool(pressure.get("active", False))
        if not top_files and isinstance(pressure.get("top_files"), list):
            top_files = [item for item in pressure.get("top_files", []) if isinstance(item, dict)][:4]
            if pending_bytes <= 0:
                pending_bytes = sum(_safe_int(item.get("estimated_pending_bytes"), 0) for item in top_files if isinstance(item, dict))

    return {
        "active": bool(active_flag or pending > 0 or bytes_total > 0),
        "pending_lines": int(pending),
        "file_count": int(file_count),
        "file_size_bytes": int(bytes_total),
        "estimated_pending_bytes": int(pending_bytes),
        "top_files": top_files,
        "policy": str(line_estimation.get("sparse_large_line_policy") or ""),
    }


def _candidate_family_context(candidates: list[dict[str, Any]], family: str) -> dict[str, Any]:
    pending = 0
    oldest_age = 0.0
    ready_count = 0
    top_lane = ""
    top_lane_pending = -1
    for row in candidates:
        row_family = _lane_family(str(row.get("name") or ""), str(row.get("assigned_pressure_lane") or ""))
        if row_family != family:
            continue
        row_pending = _safe_int(row.get("pending_lines"), 0)
        pending += row_pending
        oldest_age = max(oldest_age, _safe_float(row.get("oldest_pending_age_seconds"), 0.0))
        if str(row.get("status") or "") == "ready":
            ready_count += 1
        if row_pending > top_lane_pending:
            top_lane_pending = row_pending
            top_lane = str(row.get("name") or "")
    return {
        "pending_lines": int(pending),
        "oldest_pending_age_seconds": round(oldest_age, 3),
        "ready_count": int(ready_count),
        "top_lane": top_lane,
    }


def _runtime_capacity_score(runtime: dict[str, Any], memory_efficiency: dict[str, Any]) -> float:
    score = 100.0
    runtime_status = _status(runtime, "ready")
    memory_status = _status(memory_efficiency, "ready")
    host_saturation = _safe_float(runtime.get("host_saturation_score"), 0.0)
    compute_level = str(runtime.get("compute_pressure_level") or "").strip().lower()
    memory_level = str(runtime.get("memory_pressure_level") or "").strip().lower()
    memory_high = _memory_pressure_high(memory_efficiency, runtime)
    protect_live_soft_block = bool(
        str(runtime.get("throttle_profile") or "").strip().lower() == "protect_live"
        and host_saturation < 75.0
        and not memory_high
    )
    backlog_safe_soft_degraded = bool(
        runtime_status == "degraded"
        and compute_level in {"", "normal", "low", "elevated"}
        and memory_level in {"", "normal", "green"}
        and 0.0 < host_saturation <= 50.0
        and not memory_high
    )
    bounded_soft_degraded = bool(
        runtime_status == "degraded"
        and compute_level in {"", "normal", "low", "elevated"}
        and memory_level in {"", "normal", "green"}
        and 0.0 < host_saturation <= 70.0
        and not memory_high
    )
    if runtime_status in {"blocked", "critical"}:
        score = min(score, 62.0 if protect_live_soft_block else 45.0)
    elif runtime_status == "degraded" and bounded_soft_degraded and not backlog_safe_soft_degraded:
        score = min(score, 82.0)
    elif runtime_status == "degraded" and not backlog_safe_soft_degraded:
        score = min(score, 68.0)
    if memory_status in {"blocked", "critical"}:
        score = min(score, 62.0 if protect_live_soft_block else 45.0)
    elif memory_status == "degraded":
        score = min(score, 68.0)
    if host_saturation > 0:
        score = min(score, _metric_score(host_saturation, green=50.0, warning=75.0, critical=90.0))
    if compute_level in {"high", "critical"}:
        score = min(score, 62.0 if protect_live_soft_block else 55.0)
    if memory_high:
        score = min(score, 55.0)
    return score


def _backlog_section_scorecard(
    *,
    fleet: dict[str, Any],
    storage: dict[str, Any],
    runtime: dict[str, Any],
    memory_efficiency: dict[str, Any],
    writer_health: dict[str, Any],
    active: dict[str, Any],
    candidates: list[dict[str, Any]],
    total_pending_lines: int,
    target_pending_lines: int,
) -> dict[str, Any]:
    oldest_age = _backpressure_oldest_age(storage, fleet)
    core_pending = _backpressure_value(storage, fleet, "core_pending_lines")
    deferred_pending = _backpressure_value(storage, fleet, "deferred_pending_lines")
    support_pending = _backpressure_value(storage, fleet, "support_pending_lines")
    cold_pending = _backpressure_value(storage, fleet, "cold_pending_lines")
    stale_stage_pending = _backpressure_value(storage, fleet, "stale_stage_pending_lines")
    sparse = _sparse_large_context(storage, active, candidates)
    provider_context = _candidate_family_context(candidates, "market_data")

    sections: list[dict[str, Any]] = []
    core_target = 5_000
    core_score = 0.72 * _metric_score(core_pending, green=core_target, warning=15_000, critical=40_000)
    if core_pending > 0:
        core_score += 0.28 * _metric_score(oldest_age, green=240.0, warning=3_600.0, critical=86_400.0)
    else:
        core_score += 28.0
    sections.append(
        _section_row(
            section_id="core_decision",
            label="Core decision backlog",
            score=core_score,
            pending_lines=core_pending,
            target_pending_lines=core_target,
            oldest_pending_age_seconds=oldest_age if core_pending > 0 else 0.0,
            primary_issue=(
                "core decision work is above the green target and the oldest pending item is stale"
                if core_pending > core_target
                else "core decision backlog is inside the green target"
            ),
            recommended_next_action=(
                "keep prioritizing the core_decision_drainer and re-score after each writer wave"
                if core_pending > core_target
                else "keep core on watch and avoid widening until the total queue stays green"
            ),
            evidence=[f"core_pending_lines={core_pending}", f"oldest_pending_age_seconds={round(oldest_age, 3)}"],
            weight=0.28,
            source="ingestion_storage_control",
        )
    )

    sparse_pending = _safe_int(sparse.get("pending_lines"), 0)
    sparse_bytes = _safe_int(sparse.get("file_size_bytes"), 0)
    sparse_pending_bytes = _safe_int(sparse.get("estimated_pending_bytes"), 0)
    sparse_work_bytes = sparse_pending_bytes if sparse_pending_bytes > 0 else sparse_bytes
    sparse_files = _safe_int(sparse.get("file_count"), 0)
    sparse_top = sparse.get("top_files") if isinstance(sparse.get("top_files"), list) else []
    sparse_score = 100.0
    if bool(sparse.get("active", False)):
        sparse_score = (
            0.42 * _metric_score(sparse_pending, green=250, warning=1_000, critical=3_000)
            + 0.38 * _metric_score(sparse_work_bytes, green=512 * 1024 * 1024, warning=2 * 1024 * 1024 * 1024, critical=8 * 1024 * 1024 * 1024)
            + 0.20 * _metric_score(oldest_age, green=900.0, warning=7_200.0, critical=86_400.0)
        )
    sections.append(
        _section_row(
            section_id="crypto_sparse_decision",
            label="Sparse crypto decision files",
            score=sparse_score,
            pending_lines=sparse_pending,
            target_pending_lines=250,
            oldest_pending_age_seconds=oldest_age if sparse_pending > 0 else 0.0,
            primary_issue=(
                "a few huge crypto decision files dominate drain time"
                if bool(sparse.get("active", False))
                else "no sparse-large decision file pressure detected"
            ),
            recommended_next_action=(
                "keep crypto merge caps tiny and let focused writer cycles chew through sparse files before widening"
                if bool(sparse.get("active", False))
                else "leave sparse-large controls in monitor mode"
            ),
            evidence=ordered_unique(
                [
                    f"sparse_pending_lines={sparse_pending}",
                    f"sparse_file_count={sparse_files}",
                    f"sparse_file_size_bytes={sparse_bytes}",
                    f"sparse_estimated_pending_bytes={sparse_pending_bytes}" if sparse_pending_bytes > 0 else "",
                    f"top_file={str(sparse_top[0].get('source_rel') or '')}" if sparse_top else "",
                ]
            ),
            weight=0.24,
            source="backpressure_drainer_fleet",
        )
    )

    deferred_target = 10_000
    sections.append(
        _section_row(
            section_id="deferred_data_quality",
            label="Deferred data quality",
            score=_metric_score(deferred_pending, green=deferred_target, warning=25_000, critical=75_000),
            pending_lines=deferred_pending,
            target_pending_lines=deferred_target,
            primary_issue=(
                "deferred data quality lanes still need drain time"
                if deferred_pending > deferred_target
                else "deferred data quality lanes are below their pressure target"
            ),
            recommended_next_action=(
                "run deferred lanes after core and sparse decision pressure cools"
                if deferred_pending > deferred_target
                else "keep deferred lanes throttled while core is the bottleneck"
            ),
            evidence=[f"deferred_pending_lines={deferred_pending}"],
            weight=0.10,
            source="ingestion_storage_control",
        )
    )

    support_target = 1_000
    sections.append(
        _section_row(
            section_id="support_watchdog",
            label="Support and watchdog telemetry",
            score=_metric_score(support_pending, green=support_target, warning=5_000, critical=25_000),
            pending_lines=support_pending,
            target_pending_lines=support_target,
            primary_issue=(
                "support telemetry is crowding the writer"
                if support_pending > support_target
                else "support telemetry is contained"
            ),
            recommended_next_action=(
                "keep support telemetry shed and shard-isolated until core clears"
                if support_pending > support_target
                else "keep current shedding posture; no extra action needed"
            ),
            evidence=[f"support_pending_lines={support_pending}"],
            weight=0.06,
            source="ingestion_storage_control",
        )
    )

    cold_total = cold_pending + stale_stage_pending
    sections.append(
        _section_row(
            section_id="cold_stage",
            label="Cold and stale stage",
            score=_metric_score(cold_total, green=500, warning=2_000, critical=10_000),
            pending_lines=cold_total,
            target_pending_lines=500,
            primary_issue=(
                "cold/stale stage work is waiting behind hot lanes"
                if cold_total > 500
                else "cold/stale stage lanes are clean"
            ),
            recommended_next_action=(
                "leave cold lanes frozen until the core queue is below target"
                if cold_total > 500
                else "keep cold lanes parked while hot backlog clears"
            ),
            evidence=[f"cold_pending_lines={cold_pending}", f"stale_stage_pending_lines={stale_stage_pending}"],
            weight=0.04,
            source="ingestion_storage_control",
        )
    )

    writer_state = str(writer_health.get("state") or "idle")
    writer_score_by_state = {
        "idle": 98.0 if total_pending_lines <= target_pending_lines else 92.0,
        "active_progressing": 82.0,
        "orphaned_progress": 64.0,
        "stale_progress": 50.0,
        "stalled": 25.0,
    }
    writer_score = writer_score_by_state.get(writer_state, 70.0)
    writer_progress_age = _safe_float(writer_health.get("progress_age_minutes"), 0.0)
    writer_timed_out_shards = _safe_int(writer_health.get("timed_out_shard_count"), 0)
    writer_completed_merges = _safe_int(writer_health.get("completed_merge_count"), 0)
    if writer_state == "active_progressing" and writer_progress_age <= 15.0 and writer_timed_out_shards <= 0 and writer_completed_merges > 0:
        writer_score = max(writer_score, 92.0)
    writer_score -= min(writer_timed_out_shards * 8.0, 20.0)
    if writer_state in {"active_progressing", "stale_progress", "stalled"}:
        writer_score = min(
            writer_score,
            _metric_score(writer_progress_age, green=15.0, warning=45.0, critical=90.0),
        )
    sections.append(
        _section_row(
            section_id="writer_merge_health",
            label="Writer merge health",
            score=writer_score,
            primary_issue=(
                "writer progress is orphaned and should not block a fresh focused cycle"
                if writer_state == "orphaned_progress"
                else "writer is making progress"
                if writer_state == "active_progressing"
                else "writer is idle and available for the next bounded wave"
                if writer_state == "idle"
                else "writer progress needs recovery attention"
            ),
            recommended_next_action=(
                "let writer-cycle-coordinator launch the next focused writer and ignore orphaned progress as inactive"
                if writer_state == "orphaned_progress"
                else "wait for the active writer to report progress, then re-score"
                if writer_state == "active_progressing"
                else "start one bounded focused writer cycle"
                if writer_state == "idle"
                else "run writer-cycle-coordinator recovery before more drain waves"
            ),
            evidence=[
                f"writer_state={writer_state}",
                f"progress_age_minutes={writer_health.get('progress_age_minutes', 0.0)}",
                f"timed_out_shard_count={writer_health.get('timed_out_shard_count', 0)}",
                f"completed_merge_count={writer_health.get('completed_merge_count', 0)}",
            ],
            weight=0.15,
            source="writer_cycle_coordinator",
        )
    )

    runtime_score = _runtime_capacity_score(runtime, memory_efficiency)
    sections.append(
        _section_row(
            section_id="runtime_capacity",
            label="Runtime capacity envelope",
            score=runtime_score,
            primary_issue=(
                "runtime or memory pressure is constraining drain size"
                if runtime_score < 75.0
                else "runtime capacity is clear for bounded drain work"
            ),
            recommended_next_action=(
                "run pressure relief before heavier drain or training work"
                if runtime_score < 75.0
                else "keep drain waves bounded; do not expand training while backlog is blocked"
            ),
            evidence=[
                f"runtime_status={_status(runtime)}",
                f"memory_efficiency_status={_status(memory_efficiency)}",
                f"host_saturation_score={runtime.get('host_saturation_score', '')}",
                f"compute_pressure_level={runtime.get('compute_pressure_level', '')}",
            ],
            weight=0.08,
            source="runtime_throttle_control",
        )
    )

    provider_pending = _safe_int(provider_context.get("pending_lines"), 0)
    provider_oldest = _safe_float(provider_context.get("oldest_pending_age_seconds"), 0.0)
    provider_score = 0.72 * _metric_score(provider_pending, green=250, warning=1_000, critical=5_000)
    provider_score += 0.28 * _metric_score(provider_oldest if provider_pending > 0 else 0.0, green=300.0, warning=1_800.0, critical=14_400.0)
    sections.append(
        _section_row(
            section_id="provider_market_data",
            label="Provider and market-data spillover",
            score=provider_score,
            pending_lines=provider_pending,
            target_pending_lines=250,
            oldest_pending_age_seconds=provider_oldest,
            primary_issue=(
                "provider/market-data tail work is present but not the main blocker"
                if provider_pending > 0
                else "provider/market-data backlog has no visible pending tail"
            ),
            recommended_next_action=(
                "queue the provider drainer after core sparse pressure cools"
                if provider_pending > 0
                else "monitor source verification separately from backlog pressure"
            ),
            evidence=[
                f"provider_pending_lines={provider_pending}",
                f"provider_oldest_pending_age_seconds={provider_oldest}",
                f"provider_top_lane={provider_context.get('top_lane', '')}",
            ],
            weight=0.05,
            source="backpressure_drainer_fleet",
        )
    )

    weighted_total = sum(_safe_float(row.get("score"), 0.0) * _safe_float(row.get("weight"), 0.0) for row in sections)
    weight_sum = sum(_safe_float(row.get("weight"), 0.0) for row in sections) or 1.0
    overall_score = weighted_total / weight_sum
    core_score_clean = next((_safe_float(row.get("score"), 100.0) for row in sections if row.get("section_id") == "core_decision"), 100.0)
    sparse_score_clean = next((_safe_float(row.get("score"), 100.0) for row in sections if row.get("section_id") == "crypto_sparse_decision"), 100.0)
    if core_score_clean < 45.0 and sparse_score_clean < 45.0:
        overall_score = min(overall_score, 44.0)
    elif core_score_clean < 45.0 or sparse_score_clean < 45.0:
        overall_score = min(overall_score, 49.0)

    ranked_sections = sorted(sections, key=lambda row: (_safe_float(row.get("score"), 0.0), -_safe_float(row.get("weight"), 0.0)))
    return {
        "overall_score": round(_clamp(overall_score, 0.0, 100.0), 1),
        "overall_grade": _grade_from_score(overall_score),
        "overall_severity": _severity_from_score(overall_score),
        "total_pending_lines": int(total_pending_lines),
        "target_pending_lines": int(target_pending_lines),
        "grade_scale": {
            "A++": "pristine, tiny backlog, and no active constraint on acceleration",
            "A+": "clear, green, and not constraining backlog acceleration",
            "A": "green or comfortably contained",
            "B": "stable but watch it",
            "C": "strained and needs scheduled drain time",
            "D": "degraded; prioritize before expansion",
            "F": "critical blocker for widening/training",
        },
        "sections": sections,
        "worst_sections": ranked_sections[:3],
        "operator_next_focus": [
            {
                "section_id": str(row.get("section_id") or ""),
                "grade": str(row.get("grade") or ""),
                "recommended_next_action": str(row.get("recommended_next_action") or ""),
            }
            for row in ranked_sections[:3]
        ],
    }


def _lane_scores(
    *,
    candidates: list[dict[str, Any]],
    writer_active: bool,
    memory_or_runtime_high: bool,
    market_hours_guarded: bool,
) -> list[dict[str, Any]]:
    scores: list[dict[str, Any]] = []
    for row in candidates:
        name = str(row.get("name") or "")
        pending = _safe_int(row.get("pending_lines"), 0)
        priority = _safe_int(row.get("priority_score"), 0)
        live_safe = bool(row.get("live_window_safe", False))
        ready = str(row.get("status") or "") == "ready"
        stale_tail = str(row.get("readiness_reason") or "") == "stale_tail"
        family = _lane_family(name, str(row.get("assigned_pressure_lane") or ""))
        concentration = row.get("concentration") if isinstance(row.get("concentration"), dict) else {}
        overlay_pending = _safe_int(row.get("storage_overlay_pending_lines"), 0)
        utility = (priority / 1000.0) + (pending / 250.0)
        if overlay_pending > 0:
            utility += 10.0
        if stale_tail:
            utility += 6.0
        if bool(concentration.get("concentrated", False)):
            utility += 8.0
        if ready:
            utility += 12.0
        risk = 0.08
        if writer_active:
            risk += 0.15
        if memory_or_runtime_high:
            risk += 0.12
        if market_hours_guarded and not live_safe:
            risk += 0.32
        if not ready:
            risk += 0.2
        confidence = _clamp(0.72 - risk + (0.08 if ready else 0.0), 0.1, 0.95)
        if writer_active:
            mode = "wait_then_re_score"
        elif market_hours_guarded and not live_safe:
            mode = "park_until_protected_window"
        elif ready and memory_or_runtime_high:
            mode = "micro_drain_with_cooldown"
        elif ready:
            mode = "bounded_handoff"
        else:
            mode = "observe"
        scores.append(
            {
                "name": name,
                "status": str(row.get("status") or ""),
                "assigned_pressure_lane": str(row.get("assigned_pressure_lane") or ""),
                "family": family,
                "pending_lines": int(pending),
                "raw_pending_lines": _safe_int(row.get("raw_pending_lines"), pending),
                "storage_overlay_pending_lines": int(overlay_pending),
                "storage_overlay_active": bool(row.get("storage_overlay_active", False)),
                "priority_score": int(priority),
                "utility_score": round(utility, 3),
                "risk_score": round(risk, 3),
                "confidence": round(confidence, 3),
                "recommended_mode": mode,
                "live_window_safe": live_safe,
                "reason_codes": ordered_unique(
                    [
                        "ready" if ready else "idle",
                        "stale_tail" if stale_tail else "",
                        "storage_overlay" if overlay_pending > 0 else "",
                        "concentrated" if bool(concentration.get("concentrated", False)) else "",
                        "writer_wait" if writer_active else "",
                        "memory_runtime_guard" if memory_or_runtime_high else "",
                    ]
                ),
            }
        )
    return sorted(scores, key=lambda row: (_safe_float(row.get("utility_score"), 0.0), _safe_int(row.get("pending_lines"), 0)), reverse=True)


def _decision_action(
    *,
    risks: list[str],
    total_pending_lines: int,
    target_pending_lines: int,
    active_drainer: str,
) -> str:
    if total_pending_lines <= target_pending_lines:
        return "park_and_observe"
    if "no_ready_drainers" in risks:
        return "refresh_backpressure_and_rebuild_lane_scores"
    if "writer_progress_stalled" in risks:
        return "run_writer_recovery_check_then_re_score"
    if "writer_progress_stale" in risks:
        return "verify_writer_progress_then_re_score"
    if "writer_active" in risks:
        return "wait_for_writer_then_re_score"
    if "recent_refill_after_drain" in risks:
        return "tighten_intake_then_re_score"
    if "visible_pending_progress_missing" in risks:
        return "verify_drain_measurement_then_re_score"
    if "market_hours_guard" in risks and not active_drainer:
        return "park_until_protected_window"
    if "memory_pressure_high" in risks or "runtime_pressure_high" in risks:
        return "run_micro_drain_after_pressure_relief"
    if "recent_progress_rate_low" in risks:
        return "run_one_diagnostic_wave_then_compare_progress"
    return "run_bounded_wave"


def _drain_playbook(action: str, *, selected_drainer: str, next_ready_drainer: str, target_pending_lines: int) -> list[dict[str, Any]]:
    if action in {"verify_writer_progress_then_re_score", "run_writer_recovery_check_then_re_score"}:
        return [
            {"step": "inspect_writer", "command": ["./scripts/ops/opsctl.sh", "writer-cycle-coordinator", "--json"]},
            {"step": "refresh_storage", "command": ["./scripts/ops/opsctl.sh", "ingestion-storage-control", "--json"]},
            {"step": "re_score_drainers", "command": ["./scripts/ops/opsctl.sh", "drainer-intelligence-layer", "--json"]},
        ]
    if action == "wait_for_writer_then_re_score":
        return [
            {"step": "wait_for_single_writer", "command": ["./scripts/ops/opsctl.sh", "writer-cycle-coordinator", "--json"]},
            {"step": "re_score_active_lane", "command": ["./scripts/ops/opsctl.sh", "drainer-intelligence-layer", "--json"]},
        ]
    if action == "run_micro_drain_after_pressure_relief":
        return [
            {"step": "pressure_relief", "command": ["./scripts/ops/opsctl.sh", "pressure-relief", "--apply", "--json"]},
            {"step": "micro_drain", "command": ["./scripts/ops/opsctl.sh", "backpressure-super-drainer", "--apply", "--max-waves", "1", "--target-pending-lines", str(target_pending_lines), "--poll-seconds", "2", "--wait-timeout-seconds", "20", "--command-timeout-seconds", "540", "--sql-manager-timeout-cap-seconds", "420", "--json"]},
        ]
    if action == "tighten_intake_then_re_score":
        return [
            {"step": "pressure_relief", "command": ["./scripts/ops/opsctl.sh", "pressure-relief", "--apply", "--json"]},
            {"step": "runtime_throttle", "command": ["./scripts/ops/opsctl.sh", "runtime-throttle", "--apply", "--json"]},
            {"step": "refresh_storage", "command": ["./scripts/ops/opsctl.sh", "ingestion-storage-control", "--json"]},
            {"step": "re_score_before_next_wave", "command": ["./scripts/ops/opsctl.sh", "drainer-intelligence-layer", "--json"]},
        ]
    if action == "verify_drain_measurement_then_re_score":
        return [
            {"step": "refresh_storage", "command": ["./scripts/ops/opsctl.sh", "ingestion-storage-control", "--json"]},
            {"step": "refresh_drainer_fleet", "command": ["./scripts/ops/opsctl.sh", "backpressure-drainer-fleet", "--json"]},
            {"step": "inspect_writer", "command": ["./scripts/ops/opsctl.sh", "writer-cycle-coordinator", "--json"]},
            {"step": "re_score_before_next_wave", "command": ["./scripts/ops/opsctl.sh", "drainer-intelligence-layer", "--json"]},
        ]
    if action == "run_bounded_wave":
        return [
            {"step": "run_selected_lane", "selected_drainer": selected_drainer, "command": ["./scripts/ops/opsctl.sh", "backpressure-super-drainer", "--apply", "--max-waves", "1", "--target-pending-lines", str(target_pending_lines), "--poll-seconds", "2", "--wait-timeout-seconds", "30", "--command-timeout-seconds", "900", "--sql-manager-timeout-cap-seconds", "900", "--json"]},
            {"step": "queue_next_lane", "selected_drainer": next_ready_drainer, "command": ["./scripts/ops/opsctl.sh", "drainer-intelligence-layer", "--json"]},
        ]
    return [{"step": "observe", "command": ["./scripts/ops/opsctl.sh", "drainer-intelligence-layer", "--json"]}]


def build_intelligence_from_payloads(
    *,
    fleet: dict[str, Any],
    super_drainer: dict[str, Any],
    memory: dict[str, Any],
    storage: dict[str, Any],
    runtime: dict[str, Any],
    memory_efficiency: dict[str, Any],
    writer: dict[str, Any],
    target_pending_lines: int = DEFAULT_TARGET_PENDING_LINES,
) -> dict[str, Any]:
    target_pending_lines = max(int(target_pending_lines), 0)
    overlay_context = _storage_overlay_family_context(storage)
    candidates = _augment_candidates_with_storage_overlay(_candidate_drainers(fleet), overlay_context)
    active = _active_drainer(fleet, super_drainer)
    active_name = str(active.get("name") or "").strip()
    total_pending = _total_pending_lines(fleet, super_drainer, storage)
    writer_health = _writer_health(fleet, super_drainer, writer)
    risks = _risk_flags(
        fleet=fleet,
        super_drainer=super_drainer,
        storage=storage,
        runtime=runtime,
        memory_efficiency=memory_efficiency,
        writer=writer,
        memory=memory,
        total_pending_lines=total_pending,
        target_pending_lines=target_pending_lines,
        writer_health=writer_health,
    )
    writer_is_active = "writer_active" in risks
    pressure_guarded = bool("memory_pressure_high" in risks or "runtime_pressure_high" in risks)
    market_guarded = bool("market_hours_guard" in risks)
    lane_scores = _lane_scores(
        candidates=candidates,
        writer_active=writer_is_active,
        memory_or_runtime_high=pressure_guarded,
        market_hours_guarded=market_guarded,
    )
    selected = next((row for row in lane_scores if row["name"] == active_name), lane_scores[0] if lane_scores else {})
    selected_name = active_name or str(selected.get("name") or "")
    next_lane = next((row for row in lane_scores if row.get("name") != selected_name and row.get("status") == "ready"), {})
    action = _decision_action(
        risks=risks,
        total_pending_lines=total_pending,
        target_pending_lines=target_pending_lines,
        active_drainer=selected_name,
    )
    confidence = _confidence(risks=risks, fleet=fleet, super_drainer=super_drainer, memory=memory)
    recent = _recent_memory(memory)
    family_summary = _lane_family_summary(lane_scores)
    pressure_forecast = _pressure_forecast(memory, total_pending, target_pending_lines, writer_health)
    section_scorecard = _backlog_section_scorecard(
        fleet=fleet,
        storage=storage,
        runtime=runtime,
        memory_efficiency=memory_efficiency,
        writer_health=writer_health,
        active=active,
        candidates=candidates,
        total_pending_lines=total_pending,
        target_pending_lines=target_pending_lines,
    )
    adaptive_target = target_pending_lines
    if "storage_critical" in risks:
        adaptive_target = min(adaptive_target, 2500)
    if pressure_guarded:
        adaptive_target = max(adaptive_target, 5000)

    status = "ready"
    if "no_ready_drainers" in risks and total_pending > target_pending_lines:
        status = "degraded"
    if "progress_stalled" in risks:
        status = "degraded"
    if "writer_progress_stalled" in risks:
        status = "degraded"
    if confidence < 0.35:
        status = "degraded"
    playbook = _drain_playbook(
        action,
        selected_drainer=selected_name,
        next_ready_drainer=str(next_lane.get("name") or ""),
        target_pending_lines=int(adaptive_target),
    )
    timestamp_utc = iso_now()
    decision_packet = {
        "action": action,
        "selected_drainer": selected_name,
        "selected_pressure_lane": str(active.get("assigned_pressure_lane") or selected.get("assigned_pressure_lane") or ""),
        "next_ready_drainer": str(next_lane.get("name") or ""),
        "confidence": confidence,
        "total_pending_lines": int(total_pending),
        "storage_overlay_used": bool(overlay_context.get("active", False)),
        "target_pending_lines": int(target_pending_lines),
        "adaptive_target_pending_lines": int(adaptive_target),
        "recommended_max_waves": 1 if writer_is_active or pressure_guarded else 2,
        "recommended_cooldown_seconds": 90 if pressure_guarded else 45,
        "writer_health": writer_health,
        "pressure_forecast": pressure_forecast,
        "backlog_grade": section_scorecard["overall_grade"],
        "backlog_score": section_scorecard["overall_score"],
        "worst_backlog_sections": [row["section_id"] for row in section_scorecard.get("worst_sections", [])[:3] if isinstance(row, dict)],
        "risk_flags": risks,
        "reason_codes": ordered_unique(
            [
                "single_writer_guard" if writer_is_active else "writer_idle",
                "storage_critical" if "storage_critical" in risks else "",
                "pressure_guarded" if pressure_guarded else "",
                "recent_memory_low_progress" if "recent_progress_rate_low" in risks else "",
                "ready_lane_available" if _safe_int(fleet.get("ready_drainer_count"), 0) > 0 else "",
            ]
        ),
    }
    needs_packet = _backlog_needs_packet(
        timestamp_utc=timestamp_utc,
        scorecard=section_scorecard,
        decision_packet=decision_packet,
        playbook=playbook,
        risks=risks,
        recent=recent,
        writer_health=writer_health,
    )

    payload = {
        "timestamp_utc": timestamp_utc,
        "schema_version": 1,
        "ok": status == "ready",
        "overall_status": status,
        "mode": "drainer_intelligence_layer",
        "decision_packet": decision_packet,
        "storage_overlay_context": overlay_context,
        "backlog_section_scorecard": section_scorecard,
        "backlog_needs_packet": needs_packet,
        "lane_intelligence": lane_scores[:10],
        "lane_family_summary": family_summary[:10],
        "drain_playbook": playbook,
        "safety_envelope": {
            "single_writer_only": True,
            "starts_parallel_sql_writers": False,
            "max_apply_waves_now": 0 if writer_is_active else (1 if pressure_guarded else 2),
            "collector_expansion_allowed": False,
            "writer_recovery_required": bool("writer_progress_stale" in risks or "writer_progress_stalled" in risks),
            "protected_families": ["core_decision", "runtime_memory", "support_alerts"],
            "degrade_first_families": ["reports", "cold_stage", "model_research"],
        },
        "learning_summary": {
            "history_count": recent["history_count"],
            "recent_progress_rate": recent["recent_progress_rate"],
            "recent_target_met_rate": recent["recent_target_met_rate"],
            "recent_refill_rate": recent["recent_refill_rate"],
            "latest_refill_detected": bool(recent["latest_refill_detected"]),
            "latest_pending_lines_net_change": int(recent["latest_pending_lines_net_change"]),
            "latest_progress_waves": int(recent["latest_progress_waves"]),
            "latest_no_visible_pending_progress": bool(recent["latest_no_visible_pending_progress"]),
            "latest_active_drainer": str(recent["latest_event"].get("active_drainer") or ""),
            "latest_stop_reason": str(recent["latest_event"].get("stop_reason") or ""),
        },
        "control_contract": {
            "authority_boundary": "advisory_only_no_trade_authority_no_writer_start_authority",
            "single_writer_only": True,
            "starts_parallel_sql_writers": False,
            "feeds": ["backpressure_drainer_fleet", "backpressure_super_drainer", "backlog_drain_needs", "system_self_model"],
            "decision_loop": [
                "read_fleet_scores",
                "read_super_drainer_memory",
                "classify_pressure_and_writer_state",
                "rank_next_lane",
                "emit_exact_needs_packet",
                "append_fix_reference_frame",
                "emit_context_packet",
            ],
        },
        "source_status": {
            "fleet_status": _status(fleet),
            "super_drainer_status": _status(super_drainer),
            "storage_status": _status(storage),
            "runtime_status": _status(runtime),
            "memory_efficiency_status": _status(memory_efficiency),
        },
    }
    return payload


def build_payload(project_root: Path = PROJECT_ROOT, *, target_pending_lines: int = DEFAULT_TARGET_PENDING_LINES) -> dict[str, Any]:
    health = Path(project_root) / "governance" / "health"
    return build_intelligence_from_payloads(
        fleet=load_json(health / "backpressure_drainer_fleet_latest.json"),
        super_drainer=load_json(health / "backpressure_super_drainer_latest.json"),
        memory=load_json(health / "backpressure_super_drainer_memory_latest.json"),
        storage=load_json(health / "ingestion_storage_control_latest.json"),
        runtime=load_json(health / "runtime_throttle_control_latest.json"),
        memory_efficiency=load_json(health / "memory_efficiency_control_latest.json"),
        writer=load_json(health / "writer_cycle_coordinator_latest.json"),
        target_pending_lines=int(target_pending_lines),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the advisory intelligence layer for backpressure drainers.")
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--out-file", default=str(DEFAULT_OUT_PATH))
    parser.add_argument("--context-file", default=str(DEFAULT_CONTEXT_PATH))
    parser.add_argument("--needs-file", default=str(DEFAULT_NEEDS_PATH))
    parser.add_argument("--fix-ledger-file", default=str(DEFAULT_FIX_LEDGER_PATH))
    parser.add_argument("--target-pending-lines", type=int, default=DEFAULT_TARGET_PENDING_LINES)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    project_root = Path(args.project_root).expanduser().resolve()
    payload = build_payload(project_root, target_pending_lines=int(args.target_pending_lines))
    out_file = Path(args.out_file).expanduser()
    if not out_file.is_absolute():
        out_file = project_root / out_file
    write_payload(out_file, payload)
    needs_file = Path(args.needs_file).expanduser()
    if not needs_file.is_absolute():
        needs_file = project_root / needs_file
    write_payload(needs_file, _nested(payload, "backlog_needs_packet"))
    fix_ledger_file = Path(args.fix_ledger_file).expanduser()
    if not fix_ledger_file.is_absolute():
        fix_ledger_file = project_root / fix_ledger_file
    _append_fix_reference_if_changed(fix_ledger_file, payload)
    if args.apply:
        context_file = Path(args.context_file).expanduser()
        if not context_file.is_absolute():
            context_file = project_root / context_file
        write_payload(context_file, payload)

    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        decision = payload.get("decision_packet") if isinstance(payload.get("decision_packet"), dict) else {}
        scorecard = payload.get("backlog_section_scorecard") if isinstance(payload.get("backlog_section_scorecard"), dict) else {}
        needs = payload.get("backlog_needs_packet") if isinstance(payload.get("backlog_needs_packet"), dict) else {}
        print(
            "drainer_intelligence_layer "
            f"status={payload.get('overall_status', '')} "
            f"action={decision.get('action', '')} "
            f"selected={decision.get('selected_drainer', '')} "
            f"backlog_grade={scorecard.get('overall_grade', '')} "
            f"backlog_score={scorecard.get('overall_score', '')} "
            f"top_need={needs.get('top_need_section', '')}"
        )
    return 0 if str(payload.get("overall_status") or "") in {"ready", "advisory", "degraded"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
