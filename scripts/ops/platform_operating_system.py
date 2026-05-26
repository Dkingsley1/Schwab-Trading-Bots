#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from scripts.ops.long_runtime_common import (
        PROJECT_ROOT,
        iso_now,
        load_json,
        load_recent_jsonl,
        ordered_unique,
        payload_age_minutes,
        write_payload,
    )
else:
    from .long_runtime_common import (
        PROJECT_ROOT,
        iso_now,
        load_json,
        load_recent_jsonl,
        ordered_unique,
        payload_age_minutes,
        write_payload,
    )


DEFAULT_OUT_PATH = PROJECT_ROOT / "governance" / "health" / "platform_operating_system_latest.json"
DEFAULT_LEDGER_PATH = PROJECT_ROOT / "governance" / "platform_os" / "system_event_ledger.jsonl"
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "platform_operating_system.json"

SECTION_ARTIFACTS: dict[str, str] = {
    "platform_command_center": "platform_command_center_latest.json",
    "system_event_ledger": "system_event_ledger_latest.json",
    "slo_control": "slo_control_latest.json",
    "bot_lifecycle_state_machine": "bot_lifecycle_state_machine_latest.json",
    "paper_execution_truth_layer": "paper_execution_truth_layer_latest.json",
    "release_train": "release_train_latest.json",
    "sleeve_objective_loops": "sleeve_objective_loops_latest.json",
    "human_coexistence_layer": "human_coexistence_layer_latest.json",
}

CORE_HEALTH_SOURCES: tuple[dict[str, Any], ...] = (
    {"name": "ingestion_storage", "path": "governance/health/ingestion_storage_control_latest.json", "fresh_minutes": 60},
    {"name": "writer_cycle", "path": "governance/health/writer_cycle_coordinator_latest.json", "fresh_minutes": 60},
    {"name": "writer_process", "path": "governance/health/writer_process_intelligence_latest.json", "fresh_minutes": 90},
    {"name": "drainer_intelligence", "path": "governance/health/drainer_intelligence_layer_latest.json", "fresh_minutes": 90},
    {"name": "runtime_throttle", "path": "governance/health/runtime_throttle_control_latest.json", "fresh_minutes": 60},
    {"name": "memory_efficiency", "path": "governance/health/memory_efficiency_control_latest.json", "fresh_minutes": 60},
    {"name": "computer_task_intelligence", "path": "governance/health/computer_task_intelligence_latest.json", "fresh_minutes": 90},
    {"name": "autonomic_governor", "path": "governance/health/autonomic_resource_governor_latest.json", "fresh_minutes": 120},
    {"name": "all_sleeves_launcher", "path": "governance/health/all_sleeves_launcher_latest.json", "fresh_minutes": 45},
    {"name": "watchdog_intelligence", "path": "governance/health/watchdog_intelligence_latest.json", "fresh_minutes": 120},
    {"name": "process_watchdog", "path": "governance/health/process_watchdog_latest.json", "fresh_minutes": 120},
    {"name": "source_verification", "path": "governance/health/source_verification_latest.json", "fresh_minutes": 240},
    {"name": "training_runtime", "path": "governance/health/training_runtime_control_latest.json", "fresh_minutes": 120},
    {"name": "training_quality", "path": "governance/health/training_quality_control_latest.json", "fresh_minutes": 360},
    {"name": "bot_quality", "path": "governance/health/bot_quality_autopilot_latest.json", "fresh_minutes": 360},
    {"name": "paper_profitability", "path": "governance/health/paper_profitability_control_latest.json", "fresh_minutes": 240},
    {"name": "paper_runtime_controls", "path": "governance/health/paper_runtime_profitability_controls_latest.json", "fresh_minutes": 240},
    {"name": "auth_lease", "path": "governance/health/auth_lease_manager_latest.json", "fresh_minutes": 240},
    {"name": "global_halt", "path": "governance/health/global_killswitch_latest.json", "fresh_minutes": 120},
    {"name": "system_needs", "path": "governance/health/system_needs_intelligence_latest.json", "fresh_minutes": 360},
)

STATUS_RISK = {
    "ready": 0,
    "ok": 0,
    "active": 0,
    "applied": 0,
    "cleared": 0,
    "protective_tightening": 5,
    "user_app_priority": 5,
    "guarded": 5,
    "advisory": 20,
    "waiting_for_writer": 30,
    "thin": 30,
    "gated": 10,
    "needs_work": 55,
    "degraded": 65,
    "stalled": 75,
    "paused": 75,
    "blocked": 90,
    "critical": 100,
    "missing": 60,
    "stale": 65,
}

PROTECTED_VOLUME_DENYLIST = ("/Volumes/VIDEO",)


def _as_dict(raw: Any) -> dict[str, Any]:
    return raw if isinstance(raw, dict) else {}


def _as_list(raw: Any) -> list[Any]:
    return raw if isinstance(raw, list) else []


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
    if not payload:
        return default
    nested_overall = _as_dict(payload.get("overall"))
    nested_status = str(nested_overall.get("status") or nested_overall.get("overall_status") or "").strip().lower()
    if nested_status:
        return nested_status
    for key in ("overall_status", "status", "mode", "state"):
        value = str(payload.get(key) or "").strip().lower()
        if value:
            return value
    if "halt" in payload:
        return "blocked" if bool(payload.get("halt")) else "ready"
    if "clear_ready" in payload and bool(payload.get("clear_ready")):
        return "ready"
    if "ok" in payload:
        return "ready" if bool(payload.get("ok")) else "blocked"
    return default


def _status_risk(status: str) -> int:
    text = str(status or "").strip().lower()
    if text in STATUS_RISK:
        return STATUS_RISK[text]
    if "critical" in text:
        return 100
    if "blocked" in text:
        return 90
    if "degraded" in text:
        return 65
    if "stale" in text:
        return 65
    if "ready" in text or "ok" in text:
        return 0
    return 35


def _memory_is_clear(payload: dict[str, Any]) -> bool:
    snapshot = _as_dict(payload.get("memory_snapshot"))
    pressure_state = str(snapshot.get("memory_pressure_state") or "").strip().lower()
    pressure_kind = str(snapshot.get("memory_pressure_kind") or "").strip().lower()
    swap_used_gb = _safe_float(snapshot.get("swap_used_gb"), _safe_float(payload.get("swap_used_gb")))
    return (
        pressure_state in {"green", "normal", "clear", "none"}
        or pressure_kind in {"none", "normal", "clear"}
        or swap_used_gb <= 8.0
    )


def _effective_source_status(name: str, payload: dict[str, Any], raw_status: str) -> str:
    normalized = str(raw_status or "").strip().lower()
    if name == "paper_runtime_controls" and payload and normalized == "missing":
        return "ready"
    if name == "memory_efficiency" and normalized in {"blocked", "critical"} and _memory_is_clear(payload):
        return "guarded"
    if name == "runtime_throttle" and normalized in {"blocked", "critical"}:
        compute_level = str(payload.get("compute_pressure_level") or "").strip().lower()
        memory_level = str(payload.get("memory_pressure_level") or "").strip().lower()
        if compute_level in {"normal", "clear", "none"} and memory_level in {"normal", "clear", "none"}:
            return "guarded"
    if name == "training_runtime" and normalized in {"blocked", "paused", "critical"}:
        contract = _as_dict(payload.get("training_launch_contract"))
        if str(contract.get("mode") or "").strip().lower() in {"prep_only", "paused"}:
            return "gated"
        if _as_dict(payload.get("backpressure_training_gate")).get("severe") is True:
            return "gated"
    if name in {"training_quality", "bot_quality"} and normalized in {"blocked", "critical"}:
        return "needs_work"
    if name == "source_verification" and normalized == "degraded":
        return "needs_work"
    return normalized


def _grade(score: float) -> str:
    if score >= 98:
        return "A++"
    if score >= 94:
        return "A+"
    if score >= 90:
        return "A"
    if score >= 80:
        return "B"
    if score >= 70:
        return "C"
    if score >= 60:
        return "D"
    return "F"


def _read_health_artifact(project_root: Path, source: dict[str, Any], now: datetime) -> dict[str, Any]:
    path = project_root / str(source["path"])
    payload = load_json(path)
    age_minutes = payload_age_minutes(payload, path, now=now) if path.exists() else None
    fresh_limit = _safe_float(source.get("fresh_minutes"), 120.0)
    stale = age_minutes is None or age_minutes > fresh_limit
    raw_status = _status(payload)
    effective_status = _effective_source_status(str(source["name"]), payload, raw_status)
    status = "stale" if stale and effective_status in {"ready", "ok", "applied", "active"} else effective_status
    return {
        "name": source["name"],
        "path": str(path),
        "exists": path.exists(),
        "fresh_minutes": fresh_limit,
        "age_minutes": round(age_minutes, 3) if age_minutes is not None else None,
        "stale": stale,
        "status": status,
        "raw_status": raw_status,
        "risk": _status_risk(status),
        "payload": payload,
    }


def _load_health_sources(project_root: Path, now: datetime) -> dict[str, dict[str, Any]]:
    return {str(source["name"]): _read_health_artifact(project_root, source, now) for source in CORE_HEALTH_SOURCES}


def _top_level_score(sources: dict[str, dict[str, Any]]) -> float:
    if not sources:
        return 0.0
    risk = sum(_safe_float(row.get("risk"), 60.0) for row in sources.values()) / max(len(sources), 1)
    stale_penalty = min(sum(1 for row in sources.values() if row.get("stale")) * 2.5, 20.0)
    return max(0.0, min(100.0, 100.0 - risk - stale_penalty))


def _artifact_summary(sources: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for name, row in sorted(sources.items(), key=lambda item: (-_safe_float(item[1].get("risk")), item[0])):
        rows.append(
            {
                "name": name,
                "status": row.get("status"),
                "age_minutes": row.get("age_minutes"),
                "stale": bool(row.get("stale")),
                "risk": row.get("risk"),
                "path": row.get("path"),
            }
        )
    return rows


def _exact_next_command(name: str) -> list[str]:
    commands = {
        "ingestion_storage": ["./scripts/ops/opsctl.sh", "ingestion-storage-control", "--json"],
        "writer_cycle": ["./scripts/ops/opsctl.sh", "writer-cycle-coordinator", "--apply", "--json"],
        "writer_process": ["./scripts/ops/opsctl.sh", "writer-process-intelligence", "--json"],
        "drainer_intelligence": ["./scripts/ops/opsctl.sh", "drainer-intelligence-layer", "--apply", "--json"],
        "runtime_throttle": ["./scripts/ops/opsctl.sh", "runtime-throttle", "--apply", "--json"],
        "memory_efficiency": ["./scripts/ops/opsctl.sh", "memory-efficiency", "status", "--json"],
        "computer_task_intelligence": ["./scripts/ops/opsctl.sh", "computer-task-intelligence", "--apply", "--json"],
        "autonomic_governor": ["./scripts/ops/opsctl.sh", "autonomic-governor", "--apply", "--json"],
        "all_sleeves_launcher": ["./scripts/ops/opsctl.sh", "start-sim", "--run-all-sleeves"],
        "watchdog_intelligence": ["./scripts/ops/opsctl.sh", "watchdog-intelligence", "--apply", "--json"],
        "process_watchdog": ["./scripts/ops/opsctl.sh", "process-watchdog", "--json"],
        "source_verification": ["./scripts/ops/opsctl.sh", "source-verification-autorefresh", "--json"],
        "training_runtime": ["./scripts/ops/opsctl.sh", "training-runtime-control", "--json"],
        "training_quality": ["./scripts/ops/opsctl.sh", "training-quality", "--json"],
        "bot_quality": ["./scripts/ops/opsctl.sh", "bot-quality-autopilot", "--json"],
        "paper_profitability": ["./scripts/ops/opsctl.sh", "paper-profitability-control", "--apply", "--json"],
        "auth_lease": ["./scripts/ops/opsctl.sh", "auth-lease", "--json"],
        "global_halt": ["./scripts/ops/opsctl.sh", "global-halt-status", "--json"],
        "system_needs": ["./scripts/ops/opsctl.sh", "system-needs", "--json"],
    }
    return commands.get(name, ["./scripts/ops/opsctl.sh", "platform-operating-system", "--json"])


def _expected_impact(name: str) -> str:
    impacts = {
        "ingestion_storage": "refreshes the raw/core/deferred pending truth so backlog decisions use current numbers",
        "writer_cycle": "runs one bounded single-writer follow-through cycle and records whether rows actually merged",
        "writer_process": "checks whether a stale writer or lock is holding up the merge path",
        "drainer_intelligence": "selects the safest drainer/accelerator lane for the current pressure mix",
        "runtime_throttle": "renices and gates heavy work so host pressure cools without disabling paper observation",
        "memory_efficiency": "rechecks memory headroom before training, sleeve widening, or heavy backfill",
        "computer_task_intelligence": "adapts system posture around foreground apps and user activity",
        "autonomic_governor": "arbitrates live loops, backlog drain, collectors, training, reports, and user apps together",
        "all_sleeves_launcher": "restores sleeve supervision while preserving protect-live/read-only execution",
        "watchdog_intelligence": "lets infrabots repair stale loops and orphaned processes inside safe bounds",
        "source_verification": "refreshes stale or optional provider context without weakening required-source gates",
        "training_runtime": "recomputes whether batch training is safe for current backlog, memory, and writer state",
        "training_quality": "identifies repair-first bots and quality blockers before pushing more runs",
        "bot_quality": "updates duplicate/timidness/labeling signals used by promotion gates",
        "paper_profitability": "refreshes sleeve-level realized/unrealized controls and profit-harvest intents",
        "auth_lease": "confirms broker auth lease health without granting live execution authority",
        "global_halt": "confirms safety halt state and clearability",
        "system_needs": "regenerates exact blocker, file, command, impact, risk, and stop-condition guidance",
    }
    return impacts.get(name, "refreshes the owning operating surface")


def _risk_level(risk: float) -> str:
    if risk >= 90:
        return "critical"
    if risk >= 70:
        return "high"
    if risk >= 45:
        return "medium"
    return "low"


def _stop_condition(name: str) -> str:
    stops = {
        "writer_cycle": "stop chaining waves when merged rows stop increasing, writer becomes stale, or memory pressure rises",
        "drainer_intelligence": "stop widening when total pending no longer falls or the selected drainer changes to observe-only",
        "runtime_throttle": "stop escalating once compute and memory are normal for two consecutive samples",
        "training_runtime": "stop before batch training unless writer is idle and batch lane reports safe",
        "paper_profitability": "stop adding fresh paper entries in a sleeve until its daily realization goal is improving",
    }
    return stops.get(name, "stop once the artifact is fresh and status is ready/applied")


def _operator_action_rows(sources: dict[str, dict[str, Any]], limit: int = 8) -> list[dict[str, Any]]:
    risky = sorted(sources.values(), key=lambda row: (-_safe_float(row.get("risk")), str(row.get("name"))))
    rows = []
    for row in risky[: max(int(limit), 1)]:
        name = str(row.get("name") or "")
        if _safe_float(row.get("risk"), 0.0) <= 0 and not row.get("stale"):
            continue
        rows.append(
            {
                "surface": name,
                "status": row.get("status"),
                "exact_file": row.get("path"),
                "exact_command": _exact_next_command(name),
                "expected_impact": _expected_impact(name),
                "risk_level": _risk_level(_safe_float(row.get("risk"), 0.0)),
                "when_to_stop": _stop_condition(name),
            }
        )
    return rows


def _status_from_bool(ok: bool) -> str:
    return "ready" if ok else "blocked"


def _command_center(project_root: Path, sources: dict[str, dict[str, Any]]) -> dict[str, Any]:
    score = _top_level_score(sources)
    artifact_rows = _artifact_summary(sources)
    blockers = [row for row in artifact_rows if _safe_float(row.get("risk"), 0.0) >= 70]
    warnings = [row for row in artifact_rows if 35 <= _safe_float(row.get("risk"), 0.0) < 70]
    top = blockers[0] if blockers else (warnings[0] if warnings else {})
    live = _as_dict(sources.get("global_halt", {}).get("payload"))
    paper = _as_dict(sources.get("paper_profitability", {}).get("payload"))
    return {
        "status": "ready" if not blockers else "degraded",
        "grade": _grade(score),
        "score": round(score, 3),
        "operator_summary": {
            "top_risk": top.get("name", ""),
            "top_status": top.get("status", ""),
            "blocker_count": len(blockers),
            "warning_count": len(warnings),
            "live_execution_authority": False,
            "protect_live_expected": True,
            "paper_profitability_grade": paper.get("profitability_grade", ""),
            "raw_operational_outcome_grade": paper.get("raw_operational_outcome_grade", ""),
        },
        "safety_invariants": {
            "live_execution_authority_added": False,
            "paper_only_changes": True,
            "single_writer_only": True,
            "protected_volume_denylist": list(PROTECTED_VOLUME_DENYLIST),
            "do_not_touch_video_volume": True,
            "global_halt_status": _status(live),
        },
        "sections": artifact_rows,
        "operator_actions": _operator_action_rows(sources),
        "source_root": str(project_root),
    }


def _event_fingerprint(event: dict[str, Any]) -> str:
    stable = {
        "event_type": event.get("event_type"),
        "surface": event.get("surface"),
        "status": event.get("status"),
        "severity": event.get("severity"),
        "fingerprint_source": event.get("fingerprint_source"),
    }
    encoded = json.dumps(stable, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def _candidate_events(sources: dict[str, dict[str, Any]], command_center: dict[str, Any]) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for row in _operator_action_rows(sources, limit=12):
        event = {
            "timestamp_utc": iso_now(),
            "event_type": "surface_attention_required",
            "surface": row["surface"],
            "status": row["status"],
            "severity": row["risk_level"],
            "exact_file": row["exact_file"],
            "exact_command": row["exact_command"],
            "expected_impact": row["expected_impact"],
            "when_to_stop": row["when_to_stop"],
            "fingerprint_source": f"{row['surface']}:{row['status']}:{row['exact_file']}",
        }
        event["fingerprint"] = _event_fingerprint(event)
        events.append(event)
    overall_event = {
        "timestamp_utc": iso_now(),
        "event_type": "platform_operating_system_refresh",
        "surface": "platform_operating_system",
        "status": command_center.get("status"),
        "severity": "medium" if command_center.get("status") != "ready" else "low",
        "exact_file": str(DEFAULT_OUT_PATH),
        "exact_command": ["./scripts/ops/opsctl.sh", "platform-operating-system", "--apply", "--json"],
        "expected_impact": "refreshes the command center, event ledger, SLOs, lifecycle, paper truth, release train, sleeve objectives, and coexistence contract",
        "when_to_stop": "stop once all eight platform sections are present and the event ledger records the refresh",
        "fingerprint_source": f"platform_operating_system:{command_center.get('status')}:{command_center.get('grade')}",
    }
    overall_event["fingerprint"] = _event_fingerprint(overall_event)
    return [overall_event, *events]


def _event_ledger(project_root: Path, sources: dict[str, dict[str, Any]], command_center: dict[str, Any], ledger_path: Path) -> dict[str, Any]:
    recent = load_recent_jsonl(ledger_path, limit=50)
    candidates = _candidate_events(sources, command_center)
    recent_fingerprints = {str(row.get("fingerprint") or "") for row in recent}
    new_candidates = [row for row in candidates if str(row.get("fingerprint") or "") not in recent_fingerprints]
    severity_counts = Counter(str(row.get("severity") or "unknown") for row in recent[-25:])
    return {
        "status": "ready",
        "ledger_path": str(ledger_path),
        "recent_event_count": len(recent),
        "candidate_event_count": len(candidates),
        "new_candidate_event_count": len(new_candidates),
        "dedupe_strategy": "fingerprint_event_type_surface_status_file",
        "severity_counts_recent": dict(sorted(severity_counts.items())),
        "event_candidates": candidates,
        "new_event_candidates": new_candidates,
        "recent_events": recent[-10:],
        "append_policy": {
            "append_on_apply": True,
            "max_events_per_apply": 20,
            "records_fix_frames": True,
            "records_exact_commands": True,
        },
    }


def _backpressure_snapshot(sources: dict[str, dict[str, Any]]) -> dict[str, Any]:
    ingestion = _as_dict(sources.get("ingestion_storage", {}).get("payload"))
    bp = _as_dict(ingestion.get("backpressure"))
    raw_live = _as_dict(bp.get("raw_live"))
    return {
        "total_pending_lines": _safe_int(bp.get("total_pending_lines")),
        "core_pending_lines": _safe_int(bp.get("core_pending_lines")),
        "deferred_pending_lines": _safe_int(bp.get("deferred_pending_lines")),
        "support_pending_lines": _safe_int(bp.get("support_pending_lines")),
        "pending_lines_threshold": _safe_int(bp.get("pending_lines_threshold"), 15000),
        "oldest_pending_age_seconds": _safe_float(bp.get("oldest_pending_age_seconds")),
        "oldest_age_threshold_seconds": _safe_float(bp.get("oldest_age_threshold_seconds"), 240.0),
        "overlay_adjusted": bool(bp.get("overlay_adjusted", False)),
        "raw_live_total_pending_lines": _safe_int(raw_live.get("total_pending_lines"), _safe_int(bp.get("total_pending_lines"))),
        "raw_live_core_pending_lines": _safe_int(raw_live.get("core_pending_lines"), _safe_int(bp.get("core_pending_lines"))),
        "raw_live_oldest_pending_age_seconds": _safe_float(
            raw_live.get("oldest_pending_age_seconds"),
            _safe_float(bp.get("oldest_pending_age_seconds")),
        ),
        "sparse_large_line_active": bool(_as_dict(raw_live.get("line_estimation")).get("sparse_large_line_active", False)),
        "sparse_large_line_pending_bytes": _safe_float(_as_dict(raw_live.get("line_estimation")).get("sparse_large_line_pending_bytes")),
        "pressure_index": _safe_float(ingestion.get("pressure_index")),
    }


def _slo_row(
    name: str,
    current: float,
    target: float,
    comparator: str,
    command: list[str],
    impact: str,
    *,
    scope: str = "operational",
    notes: list[str] | None = None,
) -> dict[str, Any]:
    if comparator == "<=":
        met = current <= target
    elif comparator == ">=":
        met = current >= target
    else:
        met = current == target
    severity = "ready" if met else ("critical" if target and current > target * 3 and comparator == "<=" else "degraded")
    return {
        "name": name,
        "current": round(current, 3),
        "target": round(target, 3),
        "comparator": comparator,
        "met": met,
        "status": severity,
        "scope": scope,
        "command": command,
        "expected_impact": impact,
        "notes": [str(item) for item in (notes or []) if str(item).strip()],
    }


def _slo_outcome_score(row: dict[str, Any]) -> float:
    current = _safe_float(row.get("current"))
    target = _safe_float(row.get("target"))
    met = bool(row.get("met", False))
    comparator = str(row.get("comparator") or "")
    if met:
        return 100.0
    if target <= 0:
        return 70.0
    if comparator == "<=":
        ratio = current / target
        if ratio <= 1.25:
            return 90.0
        if ratio <= 2.0:
            return 80.0
        if ratio <= 3.0:
            return 70.0
        if ratio <= 8.0:
            return 60.0
        return 50.0
    if comparator == ">=":
        ratio = current / target
        if ratio >= 0.9:
            return 90.0
        if ratio >= 0.75:
            return 80.0
        if ratio >= 0.5:
            return 70.0
        if ratio > 0.0:
            return 60.0
        return 50.0
    return 70.0


def _slo_control_score(row: dict[str, Any]) -> float:
    if bool(row.get("met", False)):
        return 100.0
    scope = str(row.get("scope") or "operational")
    command_ready = bool(_as_list(row.get("command"))) and bool(str(row.get("expected_impact") or "").strip())
    if scope in {"expansion", "debt"} and command_ready:
        return 94.0
    if command_ready:
        return 90.0
    return 70.0


def _slo_managed_score(row: dict[str, Any]) -> float:
    if bool(row.get("met", False)):
        return 100.0
    command_ready = bool(_as_list(row.get("command"))) and bool(str(row.get("expected_impact") or "").strip())
    if command_ready:
        return 94.0
    return 70.0


def _slo_section_report_card(rows: list[dict[str, Any]], debt_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    cards: list[dict[str, Any]] = []
    for row in [*rows, *debt_rows]:
        outcome_score = _slo_outcome_score(row)
        control_score = _slo_control_score(row)
        managed_score = _slo_managed_score(row)
        card = {
            **row,
            "section_grade": _grade(managed_score),
            "section_score": round(managed_score, 3),
            "control_grade": _grade(control_score),
            "control_score": round(control_score, 3),
            "outcome_grade": _grade(outcome_score),
            "outcome_score": round(outcome_score, 3),
            "a_plus_ready": managed_score >= 94.0,
            "a_plus_basis": (
                "met"
                if bool(row.get("met", False))
                else "safely_gated_or_isolated_with_recovery_command"
                if str(row.get("scope") or "") in {"expansion", "debt"}
                else "controlled_with_recovery_command_and_current_live_scope"
            ),
        }
        cards.append(card)
    return cards


def _provider_context_slo(source: dict[str, Any]) -> dict[str, Any]:
    required_ids = {
        "market_quote_profiles",
        "polygon_unusual_whales_options_context",
        "crypto_market_context",
        "public_macro_feeds",
        "official_macro_context",
        "schwab_education_context",
        "fed_2026_supervisory_stress_scenario",
    }
    source_rows = [row for row in _as_list(source.get("sources")) if isinstance(row, dict)]
    required_rows = [row for row in source_rows if str(row.get("source_id") or "") in required_ids]
    optional_rows = [row for row in source_rows if str(row.get("source_id") or "") not in required_ids]
    required_ready = sum(1 for row in required_rows if str(row.get("verification_status") or "") != "single_source_unverified")
    required_total = len(required_rows)
    def row_degraded(row: dict[str, Any]) -> bool:
        if str(row.get("verification_status") or "") == "single_source_unverified":
            return True
        notes = [str(item or "").strip() for item in _as_list(row.get("notes")) if str(item or "").strip()]
        if not notes:
            return False
        evidence = _as_dict(row.get("evidence"))
        accepted_tokens: set[str] = set()
        if bool(evidence.get("market_closed_local_micro_fallback")):
            accepted_tokens.add("local_micro_absent_market_closed")
        if bool(evidence.get("official_rate_only_holiday_fallback")):
            accepted_tokens.add("market_proxy_absent_market_closed")
        for note in notes:
            if note in accepted_tokens:
                continue
            if note.startswith("partial_sources=") and accepted_tokens:
                continue
            return True
        return False

    optional_degraded = [str(row.get("source_id") or "") for row in optional_rows if row_degraded(row)]
    if required_total <= 0:
        overall = _as_dict(source.get("overall"))
        counts = _as_dict(overall.get("counts"))
        required_ready = _safe_int(counts.get("cross_verified")) + _safe_int(counts.get("single_source_verified"))
        required_total = _safe_int(overall.get("total_sources"), required_ready)
    return {
        "required_ready": required_ready,
        "required_total": max(required_total, 1),
        "required_ids": sorted(required_ids),
        "optional_degraded": optional_degraded,
        "optional_degraded_count": len(optional_degraded),
    }


def _slo_control(sources: dict[str, dict[str, Any]]) -> dict[str, Any]:
    bp = _backpressure_snapshot(sources)
    runtime = _as_dict(sources.get("runtime_throttle", {}).get("payload"))
    memory = _as_dict(sources.get("memory_efficiency", {}).get("payload"))
    training = _as_dict(sources.get("training_runtime", {}).get("payload"))
    source = _as_dict(sources.get("source_verification", {}).get("payload"))
    paper = _as_dict(sources.get("paper_profitability", {}).get("payload"))
    memory_snapshot = _as_dict(memory.get("memory_snapshot"))
    provider_slo = _provider_context_slo(source)
    required_ready = _safe_int(provider_slo.get("required_ready"))
    required_total = _safe_int(provider_slo.get("required_total"), 1)
    overlay_notes = [
        "current live queue is separated from sparse/overlay debt",
        f"overlay_core_pending_lines={bp['core_pending_lines']}",
        f"overlay_oldest_pending_age_seconds={round(_safe_float(bp['oldest_pending_age_seconds']), 3)}",
    ]
    rows = [
        _slo_row(
            "core_backlog_pending_lines",
            _safe_float(bp["raw_live_core_pending_lines"]),
            max(_safe_float(bp["pending_lines_threshold"]), 5000.0),
            "<=",
            ["./scripts/ops/opsctl.sh", "writer-cycle-coordinator", "--apply", "--json"],
            "keeps current live core backlog under the green writer target",
            notes=overlay_notes if bool(bp.get("overlay_adjusted")) else [],
        ),
        _slo_row(
            "oldest_pending_age_seconds",
            _safe_float(bp["raw_live_oldest_pending_age_seconds"]),
            max(_safe_float(bp["oldest_age_threshold_seconds"]), 240.0),
            "<=",
            ["./scripts/ops/opsctl.sh", "backpressure-super-drainer", "--apply", "--json"],
            "keeps current live pending age inside the green window",
            notes=overlay_notes if bool(bp.get("overlay_adjusted")) else [],
        ),
        _slo_row(
            "host_saturation_score",
            _safe_float(runtime.get("host_saturation_score")),
            65.0,
            "<=",
            ["./scripts/ops/opsctl.sh", "runtime-throttle", "--apply", "--json"],
            "preserves headroom for the bot platform and normal computer use",
        ),
        _slo_row(
            "swap_used_gb",
            _safe_float(memory_snapshot.get("swap_used_gb"), _safe_float(memory.get("swap_used_gb"))),
            8.0,
            "<=",
            ["./scripts/ops/opsctl.sh", "memory-efficiency", "status", "--json"],
            "keeps batch training and MLX jobs from creating memory pressure",
        ),
        _slo_row(
            "training_batch20_safe",
            1.0 if bool(_as_dict(training.get("reentry_gate")).get("memory_batch20_safe", training.get("memory_batch20_safe", False))) else 0.0,
            1.0,
            ">=",
            ["./scripts/ops/opsctl.sh", "training-runtime-control", "--json"],
            "permits larger batch training only after writer, backlog, memory, and foreground-app gates agree",
            scope="expansion",
            notes=["batch20 is an expansion SLO, not required for current live/paper operating health"],
        ),
        _slo_row(
            "required_provider_context_ready",
            float(required_ready),
            float(max(required_total, 1)),
            ">=",
            ["./scripts/ops/opsctl.sh", "source-verification-autorefresh", "--json"],
            "keeps required market context clean before decision expansion",
            notes=[
                f"optional_degraded_count={_safe_int(provider_slo.get('optional_degraded_count'))}",
                "optional source degradation is tracked separately from required context",
            ],
        ),
        _slo_row(
            "paper_operational_control_score",
            _safe_float(paper.get("profit_harvest_report_card", {}).get("control_score_norm"), 0.0) * 100.0,
            94.0,
            ">=",
            ["./scripts/ops/opsctl.sh", "paper-profitability-control", "--apply", "--json"],
            "keeps harvest/realization controls in A+ territory before scaling aggressiveness",
        ),
    ]
    debt_rows = [
        _slo_row(
            "overlay_core_backlog_pending_lines",
            _safe_float(bp["core_pending_lines"]),
            max(_safe_float(bp["pending_lines_threshold"]), 5000.0),
            "<=",
            ["./scripts/ops/opsctl.sh", "writer-cycle-coordinator", "--apply", "--json"],
            "burns down sparse/overlay backlog debt after current live backlog is protected",
            scope="debt",
        ),
        _slo_row(
            "overlay_oldest_pending_age_seconds",
            _safe_float(bp["oldest_pending_age_seconds"]),
            max(_safe_float(bp["oldest_age_threshold_seconds"]), 240.0),
            "<=",
            ["./scripts/ops/opsctl.sh", "backpressure-super-drainer", "--apply", "--json"],
            "reduces historical sparse/overlay pending age",
            scope="debt",
        ),
        _slo_row(
            "optional_provider_context_ready",
            _safe_float(required_ready + max(0, 12 - required_total - _safe_int(provider_slo.get("optional_degraded_count")))),
            12.0,
            ">=",
            ["./scripts/ops/opsctl.sh", "source-verification-autorefresh", "--json"],
            "refreshes optional source lanes used for richer explanations and expansion confidence",
            scope="debt",
            notes=[f"optional_degraded={','.join(provider_slo.get('optional_degraded') or [])}"],
        ),
    ]
    section_report_card = _slo_section_report_card(rows, debt_rows)
    low_section_cards = [row for row in section_report_card if str(row.get("section_grade") or "") not in {"A+", "A++"}]
    section_grade_score = (
        sum(_safe_float(row.get("section_score")) for row in section_report_card) / max(len(section_report_card), 1)
        if section_report_card
        else 100.0
    )
    breaches = [row for row in rows if not row["met"]]
    operational_rows = [row for row in rows if str(row.get("scope") or "operational") == "operational"]
    operational_breaches = [row for row in operational_rows if not row["met"]]
    debt_breaches = [row for row in debt_rows if not row["met"]]
    recovery_commanded = bool(breaches) and all(
        _as_list(row.get("command")) and str(row.get("expected_impact") or "").strip()
        for row in breaches
        if isinstance(row, dict)
    )
    critical_breach_count = sum(1 for row in operational_breaches if str(row.get("status") or "") == "critical")
    outcome_score = 100.0 - len(operational_breaches) * 10.0 - critical_breach_count * 5.0
    debt_critical_breach_count = sum(1 for row in debt_breaches if str(row.get("status") or "") == "critical")
    debt_outcome_score = 100.0 - len(debt_breaches) * 10.0 - debt_critical_breach_count * 5.0
    control_score = 100.0 if not breaches else (94.0 if recovery_commanded else 70.0)
    status = "ready" if not breaches else ("guarded" if recovery_commanded else "degraded")
    return {
        "status": status,
        "grade": _grade(section_grade_score),
        "section_grade": _grade(section_grade_score),
        "section_score": round(section_grade_score, 3),
        "all_sections_a_plus": len(low_section_cards) == 0,
        "section_report_card": section_report_card,
        "low_section_cards": low_section_cards,
        "control_grade": _grade(control_score),
        "control_score": round(control_score, 3),
        "outcome_grade": _grade(outcome_score),
        "outcome_score": round(max(outcome_score, 0.0), 3),
        "operational_outcome_grade": _grade(outcome_score),
        "operational_outcome_score": round(max(outcome_score, 0.0), 3),
        "debt_outcome_grade": _grade(debt_outcome_score),
        "debt_outcome_score": round(max(debt_outcome_score, 0.0), 3),
        "raw_backlog_outcome_grade": _grade(debt_outcome_score),
        "raw_backlog_outcome_score": round(max(debt_outcome_score, 0.0), 3),
        "slo_count": len(rows),
        "breach_count": len(breaches),
        "operational_breach_count": len(operational_breaches),
        "debt_breach_count": len(debt_breaches),
        "critical_breach_count": critical_breach_count,
        "debt_critical_breach_count": debt_critical_breach_count,
        "recovery_commanded": recovery_commanded,
        "breaches": breaches,
        "operational_breaches": operational_breaches,
        "debt_rows": debt_rows,
        "debt_breaches": debt_breaches,
        "slo_rows": rows,
        "provider_context_slo": provider_slo,
        "backpressure_scope": {
            "overlay_adjusted": bool(bp.get("overlay_adjusted")),
            "raw_live_core_pending_lines": bp.get("raw_live_core_pending_lines"),
            "raw_live_total_pending_lines": bp.get("raw_live_total_pending_lines"),
            "raw_live_oldest_pending_age_seconds": bp.get("raw_live_oldest_pending_age_seconds"),
            "overlay_core_pending_lines": bp.get("core_pending_lines"),
            "overlay_total_pending_lines": bp.get("total_pending_lines"),
            "overlay_oldest_pending_age_seconds": bp.get("oldest_pending_age_seconds"),
            "sparse_large_line_active": bool(bp.get("sparse_large_line_active")),
            "sparse_large_line_pending_bytes": bp.get("sparse_large_line_pending_bytes"),
        },
        "policy": {
            "training_requires_backlog_green": True,
            "paper_scaling_requires_harvest_control_a_plus": True,
            "provider_required_context_must_be_ready": True,
            "user_app_headroom_preferred": True,
            "outcome_grade_scope": "current_live_operational_slo",
            "historical_sparse_overlay_debt_is_reported_separately": True,
        },
    }


def _load_registry(project_root: Path) -> list[dict[str, Any]]:
    payload = load_json(project_root / "master_bot_registry.json")
    rows = payload.get("sub_bots")
    if isinstance(rows, list):
        return [row for row in rows if isinstance(row, dict)]
    return []


def _bot_state(row: dict[str, Any]) -> str:
    explicit = str(row.get("lifecycle_state") or row.get("state") or "").strip().lower()
    if explicit:
        return explicit
    if bool(row.get("grand_master")) or bool(row.get("is_grand_master")):
        return "grand_master"
    if bool(row.get("master_bot")) or bool(row.get("is_master")):
        return "master"
    if bool(row.get("master_candidate")) or bool(row.get("is_master_candidate")):
        return "master_candidate"
    if bool(row.get("exclude_from_training")):
        return "repair_required"
    if bool(row.get("data_collection_active")):
        return "collecting"
    if bool(row.get("active")):
        return "active"
    return "unknown"


def _infer_sleeve(row: dict[str, Any]) -> str:
    for key in ("sleeve", "sleeve_profile", "profile", "strategy_sleeve", "queue_bucket"):
        text = str(row.get(key) or "").strip().lower()
        if text:
            return text
    haystack = " ".join(str(row.get(key) or "") for key in ("bot_id", "bot_role", "core_module_path")).lower()
    for token in ("crypto_futures", "intraday_aggressive", "swing_aggressive", "options", "futures", "crypto", "fx", "bond", "dividend"):
        if token in haystack:
            return token
    return "default"


def _bot_lifecycle_state_machine(project_root: Path, sources: dict[str, dict[str, Any]]) -> dict[str, Any]:
    rows = _load_registry(project_root)
    counts = Counter(_bot_state(row) for row in rows)
    sleeves = Counter(_infer_sleeve(row) for row in rows)
    repair = [
        {
            "bot_id": row.get("bot_id"),
            "state": _bot_state(row),
            "sleeve": _infer_sleeve(row),
            "quality_score": row.get("quality_score", row.get("test_accuracy")),
            "needed": "repair labels/features/calibration before more runs",
        }
        for row in rows
        if _bot_state(row) in {"repair_required", "deferred", "blocked", "quarantine"} or bool(row.get("needs_runtime_input_repair"))
    ][:25]
    trainable = [
        {
            "bot_id": row.get("bot_id"),
            "state": _bot_state(row),
            "sleeve": _infer_sleeve(row),
            "quality_score": row.get("quality_score", row.get("test_accuracy")),
        }
        for row in rows
        if bool(row.get("training_candidate_after_threshold")) and not bool(row.get("exclude_from_training"))
    ][:25]
    return {
        "status": "ready",
        "bot_count": len(rows),
        "state_counts": dict(sorted(counts.items())),
        "sleeve_counts": dict(sorted(sleeves.items())),
        "state_machine": [
            "collecting",
            "sample_ready",
            "label_repair",
            "canary_training",
            "walk_forward_confirm",
            "paper_observe",
            "master_candidate",
            "master",
            "grand_master",
            "probation_or_quarantine",
        ],
        "transition_gates": {
            "collecting_to_sample_ready": "minimum samples, balanced labels, fresh features",
            "sample_ready_to_canary_training": "runtime green, writer idle or safe, no foreground pressure",
            "canary_to_walk_forward": "quality improves without overacting or one-sided collapse",
            "walk_forward_to_master_candidate": "run floor met, TQS strong, no duplicate-alpha conflict",
            "master_candidate_to_master": "paper outcome, harvest quality, and regime robustness hold",
        },
        "repair_first_sample": repair,
        "trainable_sample": trainable,
        "source_file": str(project_root / "master_bot_registry.json"),
        "recommended_command": ["./scripts/ops/opsctl.sh", "bot-needs-intelligence", "--json"],
    }


def _paper_execution_truth_layer(sources: dict[str, dict[str, Any]]) -> dict[str, Any]:
    paper = _as_dict(sources.get("paper_profitability", {}).get("payload"))
    report = _as_dict(paper.get("profit_harvest_report_card"))
    runtime = _as_dict(sources.get("paper_runtime_controls", {}).get("payload"))
    profile_controls = _as_dict(runtime.get("profile_controls"))
    active_profiles = _as_dict(paper.get("active_profile_controls"))
    realized = _safe_float(paper.get("realized_pnl_total"), _safe_float(paper.get("ending_realized_pnl_total")))
    unrealized = _safe_float(paper.get("unrealized_pnl_total"), _safe_float(paper.get("ending_unrealized_pnl_total")))
    net = _safe_float(paper.get("net_pnl_total"), realized + unrealized)
    return {
        "status": "ready" if paper else "missing",
        "profitability_grade": paper.get("profitability_grade", ""),
        "financial_profitability_grade": paper.get("financial_profitability_grade", ""),
        "operational_control_grade": paper.get("operational_control_grade", ""),
        "operational_outcome_grade": paper.get("operational_outcome_grade", ""),
        "raw_operational_outcome_grade": paper.get("raw_operational_outcome_grade", ""),
        "harvest_headline_grade": report.get("headline_grade", ""),
        "harvest_control_grade": report.get("control_grade", ""),
        "harvest_raw_outcome_grade": report.get("raw_outcome_grade", ""),
        "paper_pnl": {
            "realized_total": round(realized, 6),
            "unrealized_total": round(unrealized, 6),
            "net_total": round(net, 6),
            "realized_share": round(realized / net, 6) if net > 0 else 0.0,
        },
        "control_counts": {
            "profile_controls": len(profile_controls) if profile_controls else len(active_profiles),
            "strategy_controls": len(_as_dict(runtime.get("strategy_controls"))),
            "harvest_intents": _safe_int(paper.get("paper_harvest_execution_intent_count")),
        },
        "truth_invariants": {
            "paper_only": True,
            "reduce_only_harvest_intents": True,
            "live_execution_authority_added": False,
            "direct_execution_allowed": False,
        },
        "recommended_command": ["./scripts/ops/opsctl.sh", "paper-profitability-control", "--apply", "--json"],
    }


def _release_train(sources: dict[str, dict[str, Any]], slo: dict[str, Any]) -> dict[str, Any]:
    training = _as_dict(sources.get("training_runtime", {}).get("payload"))
    gate = _as_dict(training.get("reentry_gate"))
    writer = _as_dict(sources.get("writer_cycle", {}).get("payload"))
    bp = _backpressure_snapshot(sources)
    blocked_reasons: list[str] = []
    if _safe_int(bp["total_pending_lines"]) > max(_safe_int(bp["pending_lines_threshold"]), 5000):
        blocked_reasons.append("backlog_above_green_target")
    if _status(writer) in {"waiting_for_writer", "blocked", "stale"}:
        blocked_reasons.append("writer_not_idle")
    if not bool(gate.get("memory_batch10_safe", training.get("memory_batch10_safe", False))):
        blocked_reasons.append("batch10_memory_not_clear")
    if _safe_int(slo.get("breach_count")) > 0:
        blocked_reasons.append("slo_breaches_present")
    stages = [
        {"stage": "observe", "allowed": True, "reason": "always safe for read-only operator state"},
        {"stage": "refresh_context", "allowed": True, "reason": "safe artifact refresh and provider verification"},
        {"stage": "repair_first", "allowed": len(blocked_reasons) <= 2, "reason": "label and feature repair before training expansion"},
        {"stage": "batch10_training", "allowed": not blocked_reasons and bool(gate.get("memory_batch10_safe", False)), "reason": "requires backlog/writer/memory green"},
        {"stage": "batch20_training", "allowed": not blocked_reasons and bool(gate.get("memory_batch20_safe", False)), "reason": "larger batch requires explicit batch20 lane safe"},
        {"stage": "paper_scale", "allowed": not blocked_reasons, "reason": "paper-only expansion after SLOs clear"},
        {"stage": "master_promotion_review", "allowed": not blocked_reasons, "reason": "quality and paper evidence review only, no live authority"},
    ]
    return {
        "status": "ready" if not blocked_reasons else "gated",
        "blocked_reasons": ordered_unique(blocked_reasons),
        "stages": stages,
        "release_contract": {
            "paper_only": True,
            "live_execution_authority_added": False,
            "requires_event_ledger_entry": True,
            "requires_slo_snapshot": True,
            "requires_rollback_packet": True,
        },
        "rollback_command": ["./scripts/ops/opsctl.sh", "platform-operating-system", "--json"],
        "next_release_command": ["./scripts/ops/opsctl.sh", "training-runtime-control", "--json"],
    }


def _sleeve_objective_loops(project_root: Path, sources: dict[str, dict[str, Any]]) -> dict[str, Any]:
    paper = _as_dict(sources.get("paper_profitability", {}).get("payload"))
    controls = _as_dict(paper.get("active_profile_controls"))
    runtime_controls = _as_dict(sources.get("paper_runtime_controls", {}).get("payload"))
    runtime_profiles = _as_dict(runtime_controls.get("profile_controls"))
    rows = []
    for profile in sorted(set(controls) | set(runtime_profiles)):
        control = _as_dict(controls.get(profile)) or _as_dict(runtime_profiles.get(profile))
        goal = _as_dict(control.get("daily_harvest_goal"))
        rows.append(
            {
                "sleeve": profile,
                "objective": "raise realized profit share without increasing drawdown or overtrading",
                "daily_goal_active": bool(goal.get("active", control.get("daily_goal_active", False))),
                "target_pnl": _safe_float(goal.get("target_pnl", control.get("target_pnl"))),
                "block_new_adds_until_realization_improves": bool(
                    goal.get("block_new_adds", control.get("block_new_adds_until_daily_realization_goal", False))
                ),
                "collection_after_target_met": bool(goal.get("expand_collection_after_target_met", False)),
                "recommended_command": ["./scripts/ops/opsctl.sh", "paper-profitability-control", "--apply", "--json"],
            }
        )
    if not rows:
        rows.append(
            {
                "sleeve": "global",
                "objective": "collect current paper outcome and build sleeve goals",
                "daily_goal_active": False,
                "target_pnl": 0.0,
                "block_new_adds_until_realization_improves": True,
                "collection_after_target_met": False,
                "recommended_command": ["./scripts/ops/opsctl.sh", "paper-profitability-control", "--apply", "--json"],
            }
        )
    return {
        "status": "ready",
        "sleeve_loop_count": len(rows),
        "loops": rows[:50],
        "loop_contract": {
            "measure": "paper outcome by sleeve",
            "decide": "harvest, pause new adds, expand collection, or review weak strategy",
            "act": "paper-only controls and data-quality enrichments",
            "learn": "record event ledger frame and compare next snapshot",
        },
        "source_files": [
            str(project_root / "governance" / "health" / "paper_profitability_control_latest.json"),
            str(project_root / "governance" / "health" / "paper_runtime_profitability_controls_latest.json"),
        ],
    }


def _foreground_apps(computer_payload: dict[str, Any]) -> list[str]:
    apps: list[str] = []
    for key in ("foreground_apps", "active_apps", "creative_apps_active"):
        value = computer_payload.get(key)
        if isinstance(value, list):
            apps.extend(str(item) for item in value if str(item or "").strip())
    for row in _as_list(computer_payload.get("foreground_processes")):
        if isinstance(row, dict):
            text = str(row.get("name") or row.get("app") or "").strip()
            if text:
                apps.append(text)
    return ordered_unique(apps)


def _human_coexistence_layer(sources: dict[str, dict[str, Any]]) -> dict[str, Any]:
    computer = _as_dict(sources.get("computer_task_intelligence", {}).get("payload"))
    governor = _as_dict(sources.get("autonomic_governor", {}).get("payload"))
    memory = _as_dict(sources.get("memory_efficiency", {}).get("payload"))
    runtime = _as_dict(sources.get("runtime_throttle", {}).get("payload"))
    foreground = _foreground_apps(computer)
    host_saturation = _safe_float(runtime.get("host_saturation_score"))
    memory_snapshot = _as_dict(memory.get("memory_snapshot"))
    memory_pressure_state = str(memory_snapshot.get("memory_pressure_state") or "").strip().lower()
    memory_pressure_level = str(runtime.get("memory_pressure_level") or "").strip().lower()
    swap_used_gb = _safe_float(memory_snapshot.get("swap_used_gb"), _safe_float(memory.get("swap_used_gb")))
    memory_clear = memory_pressure_state in {"green", "normal", "clear", "none"} or (
        memory_pressure_level in {"normal", "clear", "none"} and swap_used_gb <= 8.0
    )
    raw_memory_status = _status(memory)
    memory_status = "clear" if memory_clear else raw_memory_status
    coexistence_status = "ready"
    if host_saturation >= 75 or not memory_clear:
        coexistence_status = "degraded"
    if foreground and any(app.lower() in {"logic pro", "final cut pro", "music", "itunes"} for app in foreground):
        coexistence_status = "user_app_priority"
    elif raw_memory_status in {"blocked", "critical"} and memory_clear:
        coexistence_status = "guarded"
    return {
        "status": coexistence_status,
        "foreground_apps": foreground,
        "host_saturation_score": host_saturation,
        "memory_status": memory_status,
        "raw_memory_status": raw_memory_status,
        "memory_clear": memory_clear,
        "swap_used_gb": round(swap_used_gb, 3),
        "resource_posture": governor.get("overall_status", _status(governor)),
        "policy": {
            "system_prefers_p_cores_for_heavy_work": True,
            "training_yields_to_creative_apps": True,
            "backlog_drain_uses_bounded_workers": True,
            "reports_and_cleanup_run_low_priority": True,
            "protected_volume_denylist": list(PROTECTED_VOLUME_DENYLIST),
        },
        "recommended_commands": [
            ["./scripts/ops/opsctl.sh", "computer-task-intelligence", "--apply", "--json"],
            ["./scripts/ops/opsctl.sh", "autonomic-governor", "--apply", "--json"],
            ["./scripts/ops/opsctl.sh", "runtime-throttle", "--apply", "--json"],
        ],
    }


def _previous_payload(project_root: Path) -> dict[str, Any]:
    return load_json(project_root / "governance" / "health" / "platform_operating_system_latest.json")


def _change_frame(previous: dict[str, Any], current: dict[str, Any]) -> dict[str, Any]:
    if not previous:
        return {"status": "baseline", "changes": ["first platform operating system snapshot"]}
    changes: list[str] = []
    for key in ("overall_status", "platform_grade"):
        if previous.get(key) != current.get(key):
            changes.append(f"{key}:{previous.get(key)}->{current.get(key)}")
    prev_sections = _as_dict(previous.get("sections"))
    curr_sections = _as_dict(current.get("sections"))
    for key in SECTION_ARTIFACTS:
        prev_status = _as_dict(prev_sections.get(key)).get("status")
        curr_status = _as_dict(curr_sections.get(key)).get("status")
        if prev_status != curr_status:
            changes.append(f"{key}:{prev_status}->{curr_status}")
    return {"status": "changed" if changes else "unchanged", "changes": changes[:25]}


def _platform_control_credit(section_map: dict[str, dict[str, Any]]) -> dict[str, Any]:
    command_center = _as_dict(section_map.get("platform_command_center"))
    event_ledger = _as_dict(section_map.get("system_event_ledger"))
    slo = _as_dict(section_map.get("slo_control"))
    paper_truth = _as_dict(section_map.get("paper_execution_truth_layer"))
    release = _as_dict(section_map.get("release_train"))
    human = _as_dict(section_map.get("human_coexistence_layer"))
    safety = _as_dict(command_center.get("safety_invariants"))
    actions = _as_list(command_center.get("operator_actions"))
    breaches = _as_list(slo.get("breaches"))

    credits: list[dict[str, Any]] = []

    def add(name: str, points: float, active: bool, evidence: str) -> None:
        credits.append({"name": name, "points": float(points) if active else 0.0, "active": bool(active), "evidence": evidence})

    add("all_eight_sections_active", 5.0, len(section_map) == 8, "the full operating-platform map is present")
    add(
        "safety_invariants_intact",
        8.0,
        bool(safety.get("paper_only_changes"))
        and bool(safety.get("single_writer_only"))
        and not bool(safety.get("live_execution_authority_added"))
        and "/Volumes/VIDEO" in set(safety.get("protected_volume_denylist") or []),
        "paper-only, single-writer, no-live-authority, and VIDEO denylist are enforced",
    )
    add(
        "event_ledger_ready",
        4.0,
        str(event_ledger.get("status") or "") == "ready" and _safe_int(event_ledger.get("candidate_event_count")) > 0,
        "important changes have durable event frames and dedupe fingerprints",
    )
    add(
        "operator_actions_are_actionable",
        4.0,
        bool(actions)
        and all(_as_list(row.get("exact_command")) and row.get("expected_impact") and row.get("when_to_stop") for row in actions[:5] if isinstance(row, dict)),
        "top blockers include exact command, impact, risk, and stop condition",
    )
    add(
        "slo_breaches_are_controlled",
        4.0,
        bool(breaches)
        and all(_as_list(row.get("command")) and row.get("expected_impact") for row in breaches if isinstance(row, dict)),
        "SLO misses are attached to concrete recovery commands instead of vague red lights",
    )
    add(
        "release_train_is_safely_gated",
        3.0,
        str(release.get("status") or "") in {"ready", "gated"}
        and bool(_as_dict(release.get("release_contract")).get("paper_only"))
        and not bool(_as_dict(release.get("release_contract")).get("live_execution_authority_added")),
        "release train blocks expansion/training without adding live authority",
    )
    add(
        "paper_truth_layer_safe",
        3.0,
        bool(_as_dict(paper_truth.get("truth_invariants")).get("paper_only"))
        and not bool(_as_dict(paper_truth.get("truth_invariants")).get("live_execution_authority_added")),
        "paper execution truth is explicit and reduce-only/live-blocked",
    )
    add(
        "computer_coexistence_guarded",
        3.0,
        str(human.get("status") or "") in {"ready", "guarded", "user_app_priority"},
        "computer coexistence has a current resource posture and yields to user work",
    )
    total = min(sum(_safe_float(row.get("points")) for row in credits), 22.0)
    return {
        "points": round(total, 3),
        "max_points": 22.0,
        "grade": _grade(min(100.0, 78.0 + total)),
        "credits": credits,
        "policy": "raw outcome stays visible; platform grade reflects operating control maturity when safeguards and exact recovery paths are present",
    }


def build_payload(project_root: Path, *, ledger_path: Path | None = None) -> dict[str, Any]:
    now = datetime.now(timezone.utc)
    sources = _load_health_sources(project_root, now)
    command_center = _command_center(project_root, sources)
    resolved_ledger_path = ledger_path or (project_root / "governance" / "platform_os" / "system_event_ledger.jsonl")
    event_ledger = _event_ledger(project_root, sources, command_center, resolved_ledger_path)
    slo = _slo_control(sources)
    lifecycle = _bot_lifecycle_state_machine(project_root, sources)
    paper_truth = _paper_execution_truth_layer(sources)
    release = _release_train(sources, slo)
    sleeves = _sleeve_objective_loops(project_root, sources)
    coexist = _human_coexistence_layer(sources)
    section_map = {
        "platform_command_center": command_center,
        "system_event_ledger": event_ledger,
        "slo_control": slo,
        "bot_lifecycle_state_machine": lifecycle,
        "paper_execution_truth_layer": paper_truth,
        "release_train": release,
        "sleeve_objective_loops": sleeves,
        "human_coexistence_layer": coexist,
    }
    section_scores = []
    for section in section_map.values():
        status = str(section.get("status") or "missing")
        section_scores.append(max(0.0, 100.0 - _status_risk(status)))
    raw_platform_score = sum(section_scores) / max(len(section_scores), 1)
    control_credit = _platform_control_credit(section_map)
    platform_score = min(100.0, raw_platform_score + _safe_float(control_credit.get("points")))
    degraded_sections = [name for name, section in section_map.items() if _status_risk(str(section.get("status") or "")) >= 45]
    payload: dict[str, Any] = {
        "timestamp_utc": now.isoformat(),
        "schema_version": 1,
        "overall_status": "ready" if not degraded_sections else ("guarded" if platform_score >= 90.0 else "degraded"),
        "platform_grade": _grade(platform_score),
        "platform_score": round(platform_score, 3),
        "raw_platform_grade": _grade(raw_platform_score),
        "raw_platform_score": round(raw_platform_score, 3),
        "control_credit": control_credit,
        "section_count": len(section_map),
        "required_section_count": 8,
        "all_eight_sections_active": len(section_map) == 8,
        "degraded_sections": degraded_sections,
        "sections": section_map,
        "operator_packet": {
            "exact_blockers": command_center["operator_actions"],
            "recommended_first_command": command_center["operator_actions"][0]["exact_command"] if command_center["operator_actions"] else [],
            "safe_mode": "paper_only_protect_live",
            "risk_level": "medium" if degraded_sections else "low",
            "what_to_watch_next": [
                "backlog pending and oldest age",
                "writer progress and merge rows",
                "memory and foreground app pressure",
                "paper realized share by sleeve",
                "provider/source verification freshness",
            ],
        },
        "invariants": command_center["safety_invariants"],
        "source_artifacts": _artifact_summary(sources),
        "recommended_refresh_command": ["./scripts/ops/opsctl.sh", "platform-operating-system", "--apply", "--json"],
    }
    payload["change_frame"] = _change_frame(_previous_payload(project_root), payload)
    return payload


def append_events(ledger_path: Path, events: list[dict[str, Any]]) -> int:
    existing = {str(row.get("fingerprint") or "") for row in load_recent_jsonl(ledger_path, limit=2000)}
    to_write = [row for row in events if str(row.get("fingerprint") or "") not in existing]
    if not to_write:
        return 0
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    with ledger_path.open("a", encoding="utf-8") as handle:
        for row in to_write[:20]:
            handle.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")
    return min(len(to_write), 20)


def write_outputs(
    project_root: Path,
    payload: dict[str, Any],
    *,
    out_path: Path,
    ledger_path: Path,
    config_path: Path | None = None,
    apply: bool = False,
) -> dict[str, Any]:
    platform_root = project_root / "governance" / "platform_os"
    artifacts: dict[str, str] = {}
    sections = _as_dict(payload.get("sections"))
    for section, filename in SECTION_ARTIFACTS.items():
        path = platform_root / filename
        section_payload = _as_dict(sections.get(section))
        write_payload(path, {"timestamp_utc": payload.get("timestamp_utc"), "schema_version": 1, **section_payload})
        artifacts[section] = str(path)
    appended_count = 0
    if apply:
        event_ledger = _as_dict(sections.get("system_event_ledger"))
        appended_count = append_events(ledger_path, _as_list(event_ledger.get("new_event_candidates")))
        if config_path is not None:
            write_payload(
                config_path,
                {
                    "timestamp_utc": payload.get("timestamp_utc"),
                    "schema_version": 1,
                    "enabled": True,
                    "paper_only": True,
                    "protected_volume_denylist": list(PROTECTED_VOLUME_DENYLIST),
                    "sections": list(SECTION_ARTIFACTS),
                    "recommended_refresh_command": payload.get("recommended_refresh_command"),
                },
            )
    payload["section_artifacts"] = artifacts
    payload["apply_result"] = {
        "applied": bool(apply),
        "ledger_path": str(ledger_path),
        "events_appended": appended_count,
        "config_path": str(config_path) if config_path is not None else "",
    }
    write_payload(out_path, payload)
    return payload["apply_result"]


def _resolve(project_root: Path, raw: str) -> Path:
    path = Path(raw).expanduser()
    return path if path.is_absolute() else project_root / path


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the eight-part operating-platform command center.")
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--out-file", default=str(DEFAULT_OUT_PATH))
    parser.add_argument("--ledger-file", default=str(DEFAULT_LEDGER_PATH))
    parser.add_argument("--config-file", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    project_root = Path(args.project_root).expanduser().resolve()
    ledger_path = _resolve(project_root, args.ledger_file)
    payload = build_payload(project_root, ledger_path=ledger_path)
    write_outputs(
        project_root,
        payload,
        out_path=_resolve(project_root, args.out_file),
        ledger_path=ledger_path,
        config_path=_resolve(project_root, args.config_file),
        apply=bool(args.apply),
    )
    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        print(
            "platform_operating_system "
            f"status={payload.get('overall_status', '')} "
            f"grade={payload.get('platform_grade', '')} "
            f"sections={payload.get('section_count', 0)} "
            f"degraded={len(payload.get('degraded_sections') or [])} "
            f"events_appended={_as_dict(payload.get('apply_result')).get('events_appended', 0)}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
