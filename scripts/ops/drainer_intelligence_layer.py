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
    from scripts.ops.long_runtime_common import iso_now, load_json, ordered_unique, write_payload
else:
    from .long_runtime_common import PROJECT_ROOT, iso_now, load_json, ordered_unique, write_payload


DEFAULT_OUT_PATH = PROJECT_ROOT / "governance" / "health" / "drainer_intelligence_layer_latest.json"
DEFAULT_CONTEXT_PATH = PROJECT_ROOT / "governance" / "health" / "drainer_intelligence_context_latest.json"
DEFAULT_TARGET_PENDING_LINES = 10_000


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
    super_summary = super_drainer.get("summary") if isinstance(super_drainer.get("summary"), dict) else {}
    metrics = fleet.get("metrics") if isinstance(fleet.get("metrics"), dict) else {}
    return max(
        _safe_int(storage_bp.get("total_pending_lines"), 0),
        _safe_int(storage.get("pending_lines_total"), 0),
        _safe_int(metrics.get("total_pending_lines"), 0),
        _safe_int(super_summary.get("final_pending_lines"), 0),
        _safe_int(super_summary.get("initial_pending_lines"), 0),
    )


def _writer_active(fleet: dict[str, Any], super_drainer: dict[str, Any], writer: dict[str, Any]) -> bool:
    super_writer = super_drainer.get("writer_state_before") if isinstance(super_drainer.get("writer_state_before"), dict) else {}
    return bool(
        fleet.get("writer_lock_held", False)
        or fleet.get("writer_active", False)
        or super_writer.get("active", False)
        or writer.get("active", False)
    )


def _memory_pressure_high(memory_efficiency: dict[str, Any], runtime: dict[str, Any]) -> bool:
    memory_snapshot = memory_efficiency.get("memory_snapshot") if isinstance(memory_efficiency.get("memory_snapshot"), dict) else {}
    state = str(memory_snapshot.get("memory_pressure_state") or "").strip().lower()
    kind = str(memory_snapshot.get("memory_pressure_kind") or "").strip().lower()
    runtime_level = str(runtime.get("memory_pressure_level") or "").strip().lower()
    return bool(
        _status(memory_efficiency) in {"blocked", "critical", "degraded"}
        or state in {"yellow", "red", "warning", "critical"}
        or kind not in {"", "none", "green", "normal"}
        or runtime_level in {"high", "critical"}
    )


def _runtime_pressure_high(runtime: dict[str, Any]) -> bool:
    return bool(
        _status(runtime) in {"blocked", "critical", "degraded"}
        or str(runtime.get("compute_pressure_level") or "").strip().lower() in {"high", "critical"}
        or _safe_float(runtime.get("host_saturation_score"), 0.0) >= 80.0
    )


def _recent_memory(memory: dict[str, Any]) -> dict[str, Any]:
    return {
        "history_count": _safe_int(memory.get("history_count"), 0),
        "recent_progress_rate": _safe_float(memory.get("recent_progress_rate"), 0.0),
        "recent_target_met_rate": _safe_float(memory.get("recent_target_met_rate"), 0.0),
        "latest_event": memory.get("latest_event") if isinstance(memory.get("latest_event"), dict) else {},
    }


def _writer_health(fleet: dict[str, Any], super_drainer: dict[str, Any], writer: dict[str, Any]) -> dict[str, Any]:
    super_writer = super_drainer.get("writer_state_before") if isinstance(super_drainer.get("writer_state_before"), dict) else {}
    active = _writer_active(fleet, super_drainer, writer)
    progress_age = max(
        _safe_float(writer.get("progress_age_minutes"), 0.0),
        _safe_float(super_writer.get("progress_age_minutes"), 0.0),
    )
    cycle_age = max(
        _safe_float(writer.get("cycle_age_minutes"), 0.0),
        _safe_float(super_writer.get("cycle_age_minutes"), 0.0),
    )
    merged_rows = max(
        _safe_int(writer.get("merged_rows_this_cycle"), 0),
        _safe_int(super_writer.get("merged_rows_this_cycle"), 0),
    )
    if not active:
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
        "current_step": str(writer.get("current_step") or super_writer.get("current_step") or ""),
        "writer_lock_owner": str(fleet.get("writer_lock_owner") or super_writer.get("writer_lock_owner") or writer.get("writer_lock_owner") or ""),
    }


def _lane_family(name: str, pressure_lane: str = "") -> str:
    text = f"{name} {pressure_lane}".lower()
    families = (
        ("core_decision", ("core_decision", "decision_channel", "trading")),
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
    return round(_clamp(score, 0.1, 0.95), 3)


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
        utility = (priority / 1000.0) + (pending / 250.0)
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
            {"step": "micro_drain", "command": ["./scripts/ops/opsctl.sh", "backpressure-super-drainer", "--apply", "--max-waves", "1", "--target-pending-lines", str(target_pending_lines), "--json"]},
        ]
    if action == "run_bounded_wave":
        return [
            {"step": "run_selected_lane", "selected_drainer": selected_drainer, "command": ["./scripts/ops/opsctl.sh", "backpressure-super-drainer", "--apply", "--max-waves", "1", "--target-pending-lines", str(target_pending_lines), "--json"]},
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
    candidates = _candidate_drainers(fleet)
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
    next_lane = next((row for row in lane_scores if row.get("name") != active_name and row.get("status") == "ready"), {})
    action = _decision_action(
        risks=risks,
        total_pending_lines=total_pending,
        target_pending_lines=target_pending_lines,
        active_drainer=active_name,
    )
    confidence = _confidence(risks=risks, fleet=fleet, super_drainer=super_drainer, memory=memory)
    recent = _recent_memory(memory)
    family_summary = _lane_family_summary(lane_scores)
    pressure_forecast = _pressure_forecast(memory, total_pending, target_pending_lines, writer_health)
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
        selected_drainer=active_name,
        next_ready_drainer=str(next_lane.get("name") or ""),
        target_pending_lines=int(adaptive_target),
    )

    payload = {
        "timestamp_utc": iso_now(),
        "schema_version": 1,
        "ok": status == "ready",
        "overall_status": status,
        "mode": "drainer_intelligence_layer",
        "decision_packet": {
            "action": action,
            "selected_drainer": active_name,
            "selected_pressure_lane": str(active.get("assigned_pressure_lane") or selected.get("assigned_pressure_lane") or ""),
            "next_ready_drainer": str(next_lane.get("name") or ""),
            "confidence": confidence,
            "total_pending_lines": int(total_pending),
            "target_pending_lines": int(target_pending_lines),
            "adaptive_target_pending_lines": int(adaptive_target),
            "recommended_max_waves": 1 if writer_is_active or pressure_guarded else 2,
            "recommended_cooldown_seconds": 90 if pressure_guarded else 45,
            "writer_health": writer_health,
            "pressure_forecast": pressure_forecast,
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
        },
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
            "latest_active_drainer": str(recent["latest_event"].get("active_drainer") or ""),
            "latest_stop_reason": str(recent["latest_event"].get("stop_reason") or ""),
        },
        "control_contract": {
            "authority_boundary": "advisory_only_no_trade_authority_no_writer_start_authority",
            "single_writer_only": True,
            "starts_parallel_sql_writers": False,
            "feeds": ["backpressure_drainer_fleet", "backpressure_super_drainer", "system_self_model"],
            "decision_loop": [
                "read_fleet_scores",
                "read_super_drainer_memory",
                "classify_pressure_and_writer_state",
                "rank_next_lane",
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
    if args.apply:
        context_file = Path(args.context_file).expanduser()
        if not context_file.is_absolute():
            context_file = project_root / context_file
        write_payload(context_file, payload)

    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        decision = payload.get("decision_packet") if isinstance(payload.get("decision_packet"), dict) else {}
        print(
            "drainer_intelligence_layer "
            f"status={payload.get('overall_status', '')} "
            f"action={decision.get('action', '')} "
            f"selected={decision.get('selected_drainer', '')}"
        )
    return 0 if str(payload.get("overall_status") or "") in {"ready", "advisory", "degraded"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
