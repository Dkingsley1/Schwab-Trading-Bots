#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable

if __package__ in {None, ""}:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from scripts.ops.long_runtime_common import iso_now, load_json, ordered_unique, payload_age_minutes, write_payload
else:
    from .long_runtime_common import PROJECT_ROOT, iso_now, load_json, ordered_unique, payload_age_minutes, write_payload


DEFAULT_OUT_PATH = PROJECT_ROOT / "governance" / "health" / "a_plus_operating_packet_latest.json"
DEFAULT_MD_OUT_PATH = PROJECT_ROOT / "governance" / "health" / "a_plus_operating_packet_latest.md"
PYTHON_BIN = Path(sys.executable)

LANE_WEIGHTS = {
    "health_scorecard": 1.35,
    "anti_degradation_guardrails": 1.25,
    "paper_performance_attribution": 1.05,
    "account_position_intelligence": 1.0,
    "livefeed_quality": 0.9,
    "event_mode": 0.75,
    "notification_reliability": 0.9,
    "promotion_discipline": 1.0,
    "platform_dependency_freeze": 0.95,
    "disaster_recovery": 0.95,
}

LANE_LABELS = {
    "health_scorecard": "A+ Health Scorecard",
    "anti_degradation_guardrails": "Anti-Degradation Guardrails",
    "paper_performance_attribution": "Paper Performance Attribution",
    "account_position_intelligence": "Account + Position Intelligence",
    "livefeed_quality": "Livefeed Quality",
    "event_mode": "Event Mode",
    "notification_reliability": "Notification Reliability",
    "promotion_discipline": "Model Promotion Discipline",
    "platform_dependency_freeze": "Dependency Freeze + Benchmarking",
    "disaster_recovery": "Disaster Recovery",
}

SOURCE_FILES = {
    "health_fast": "governance/health/health_fast_latest.json",
    "process_watchdog": "governance/health/process_watchdog_latest.json",
    "rolling_restart": "governance/health/rolling_restart_controller_latest.json",
    "ingestion_storage": "governance/health/ingestion_storage_control_latest.json",
    "runtime_throttle": "governance/health/runtime_throttle_control_latest.json",
    "process_fanout": "governance/health/process_fanout_guard_latest.json",
    "command_validity": "governance/health/command_validity_latest.json",
    "writer_cycle": "governance/health/writer_cycle_coordinator_latest.json",
    "paper_performance": "governance/health/paper_performance_latest.json",
    "paper_profitability": "governance/health/paper_profitability_control_latest.json",
    "sleeve_profitability": "governance/health/sleeve_profitability_dashboard_latest.json",
    "paper_ramp": "governance/health/paper_400_ramp_latest.json",
    "runtime_paper_guard": "governance/health/runtime_paper_regression_guard_latest.json",
    "account_position": "governance/health/account_position_study_latest.json",
    "account_policy": "governance/health/account_policy_context_latest.json",
    "account_snapshot": "governance/health/schwab_account_snapshot_refresh_latest.json",
    "livefeed_local": "governance/health/livefeed_local_latest.json",
    "livefeed_refresh_guard": "governance/health/livefeed_refresh_guard_latest.json",
    "spacex_ipo_watch": "governance/health/spacex_ipo_downside_watch_latest.json",
    "macro_event": "governance/health/macro_event_intelligence_latest.json",
    "event_store": "governance/health/point_in_time_event_store_latest.json",
    "notification_watch": "governance/health/mac_notification_watch_state.json",
    "notification_ladder": "governance/health/notification_escalation_ladder_latest.json",
    "promotion_quality": "governance/health/promotion_quality_gate_latest.json",
    "promotion_packet": "governance/champion_challenger/promotion_autopilot_packet_latest.json",
    "promotion_pipeline": "governance/walk_forward/promotion_pipeline_latest.json",
    "adaptive_regression_guard": "governance/health/adaptive_regression_guard_latest.json",
    "system_architecture_contract_graph": "governance/health/system_architecture_contract_graph_latest.json",
    "system_architecture_autopilot": "governance/health/system_architecture_autopilot_latest.json",
    "release_freeze": "governance/health/release_freeze_guard_latest.json",
    "runtime_dependency": "governance/health/runtime_dependency_profiles_latest.json",
    "library_router": "governance/health/library_utilization_router_latest.json",
    "mlx_router": "governance/health/mlx_intelligence_router_latest.json",
    "mlx_upgrade": "governance/health/mlx_library_upgrade_latest.json",
    "storage_dr": "governance/health/storage_disaster_recovery_latest.json",
    "post_restart": "governance/health/post_restart_settlement_latest.json",
    "paper_replay_drill": "governance/health/paper_replay_drill_latest.json",
}

SAFE_REFRESH_COMMANDS = [
    ["health-fast", "--json"],
    ["ingestion-storage-control", "--json"],
    ["runtime-throttle", "--apply", "--json"],
    ["process-watchdog", "--json"],
    ["rolling-restart", "--json"],
    ["paper-400-ramp", "--json"],
    ["paper-profitability-control", "--json"],
    ["account-position-study", "--json"],
    ["covered-call-roll-watch", "--json"],
    ["spacex-ipo-watch", "--json"],
    ["storage-disaster-recovery", "--json"],
    ["command-validity", "--json"],
]


def _as_dict(raw: Any) -> dict[str, Any]:
    return raw if isinstance(raw, dict) else {}


def _as_list(raw: Any) -> list[Any]:
    return raw if isinstance(raw, list) else []


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


def _status(raw: Any) -> str:
    return str(raw or "").strip().lower()


def _payload_status(payload: dict[str, Any]) -> str:
    status = _status(payload.get("overall_status") or payload.get("status") or payload.get("state"))
    if status:
        return status
    if bool(payload.get("ok", False)):
        return "ready"
    return "missing" if not payload else "unknown"


def _managed_storage_dr_advisory(payload: dict[str, Any]) -> bool:
    if not bool(payload.get("ok", False)):
        return False
    if _status(payload.get("overall_status")) != "degraded":
        return False
    mode = _status(payload.get("current_storage_mode"))
    if mode not in {"local_fallback", "external_available_unverified"}:
        return False
    probe = _as_dict(payload.get("storage_probe"))
    initial_probe = _as_dict(payload.get("initial_storage_probe"))
    external_available = bool(probe.get("external_available", False) or initial_probe.get("external_available", False))
    if not external_available:
        return False
    curated_restore = _as_dict(payload.get("curated_restore"))
    restore_external = _as_dict(payload.get("restore_external"))
    deferred_reason = str(curated_restore.get("skipped_reason") or restore_external.get("skipped_reason") or "").strip()
    return deferred_reason in {"", "not_required", "apply_disabled", "writer_not_quiet"}


def _managed_rolling_restart_advisory(restart: dict[str, Any], process: dict[str, Any]) -> bool:
    if not bool(restart.get("restart_due", False)):
        return False
    due_signals = _as_dict(restart.get("due_signals"))
    active_due = {str(key) for key, value in due_signals.items() if bool(value)}
    if active_due and active_due - {"session_stale"}:
        return False
    checkpoint = _as_dict(restart.get("checkpoint_resume"))
    if not bool(checkpoint.get("checkpoint_fresh", False)):
        return False
    if _safe_int(_as_dict(restart.get("runtime_signals")).get("restart_storms"), 0) > 0:
        return False
    creative_pause = _as_dict(process.get("creative_cotenant_pause"))
    if bool(creative_pause.get("active", False)):
        return True
    for row in _as_list(process.get("status")):
        if isinstance(row, dict) and bool(row.get("paused_by_creative_cotenant_guard", False)):
            return True
    return False


def _runtime_guarded_ready(payload: dict[str, Any]) -> bool:
    if _payload_status(payload) != "ready":
        return False
    relief = _as_dict(payload.get("soft_cap_advisory_reclassification"))
    measurements = _as_dict(relief.get("measurements"))
    return bool(
        relief.get("active", False)
        and _status(relief.get("to_status")) == "ready"
        and bool(measurements.get("runtime_ready_guarded", False))
    )


def _source_path(project_root: Path, name: str) -> Path:
    return project_root / SOURCE_FILES[name]


def _load_sources(project_root: Path) -> dict[str, dict[str, Any]]:
    return {name: load_json(_source_path(project_root, name)) for name in SOURCE_FILES}


def _source_age(project_root: Path, sources: dict[str, dict[str, Any]], name: str) -> float | None:
    path = _source_path(project_root, name)
    payload = sources.get(name, {})
    age = payload_age_minutes(payload, path)
    if age is not None:
        return round(float(age), 3)
    return None


def _fresh_enough(project_root: Path, sources: dict[str, dict[str, Any]], name: str, max_age_minutes: float) -> bool:
    age = _source_age(project_root, sources, name)
    return bool(age is not None and age <= max_age_minutes)


def _grade(score: float) -> str:
    score = _safe_float(score)
    if score >= 97.0:
        return "A+"
    if score >= 92.0:
        return "A+"
    if score >= 85.0:
        return "A"
    if score >= 75.0:
        return "B"
    if score >= 65.0:
        return "C"
    if score >= 50.0:
        return "D"
    return "F"


def _score_from_status(status: str, *, ok: bool | None = None) -> float:
    normalized = _status(status)
    if ok is True and normalized in {"", "unknown", "missing"}:
        return 100.0
    if normalized in {"ready", "ok", "running", "clear_ready", "idle", "applied"}:
        return 100.0
    if normalized in {"advisory", "watch", "guarded_ready", "protective_tightening", "waiting_for_first_quote"}:
        return 88.0
    if normalized in {"degraded", "needs_work", "repairing_readiness", "waiting_for_writer"}:
        return 68.0
    if normalized in {"blocked", "critical", "halted", "error", "failed"}:
        return 30.0
    if ok is True:
        return 92.0
    if normalized in {"missing", ""}:
        return 35.0
    return 75.0


def _cap_score(score: float) -> float:
    return round(max(0.0, min(100.0, float(score))), 2)


def _evidence_source(project_root: Path, sources: dict[str, dict[str, Any]], name: str) -> dict[str, Any]:
    payload = sources.get(name, {})
    return {
        "name": name,
        "path": str(_source_path(project_root, name)),
        "present": bool(payload),
        "status": _payload_status(payload),
        "age_minutes": _source_age(project_root, sources, name),
    }


def _pid_alive(pid_path: Path) -> bool:
    try:
        pid = int(pid_path.read_text(encoding="utf-8").strip())
    except Exception:
        return False
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except Exception:
        return False


def _lane(
    lane_id: str,
    *,
    score: float,
    status: str,
    sources: list[str],
    project_root: Path,
    source_payloads: dict[str, dict[str, Any]],
    blockers: list[str] | None = None,
    warnings: list[str] | None = None,
    evidence: dict[str, Any] | None = None,
    next_commands: list[list[str]] | None = None,
) -> dict[str, Any]:
    score = _cap_score(score)
    blockers = ordered_unique(blockers or [])
    warnings = ordered_unique(warnings or [])
    if blockers and score > 74.0:
        score = 74.0
    if _status(status) in {"ready", "ok"} and blockers:
        status = "blocked"
    return {
        "id": lane_id,
        "label": LANE_LABELS[lane_id],
        "status": status,
        "score": score,
        "grade": _grade(score),
        "a_plus": score >= 92.0 and not blockers,
        "blockers": blockers,
        "warnings": warnings,
        "evidence": evidence or {},
        "sources": [_evidence_source(project_root, source_payloads, name) for name in sources],
        "next_commands": next_commands or [],
    }


def _count_alerts(health_fast: dict[str, Any], process_watchdog: dict[str, Any]) -> int:
    summary = _as_dict(_as_dict(health_fast.get("process_watchdog")).get("alert_summary"))
    if summary:
        return _safe_int(summary.get("total_count"), 0)
    alerts = process_watchdog.get("alerts")
    return len(alerts) if isinstance(alerts, list) else 0


def _health_scorecard(project_root: Path, sources: dict[str, dict[str, Any]]) -> dict[str, Any]:
    health = sources["health_fast"]
    process = sources["process_watchdog"]
    restart = sources["rolling_restart"]
    guarded_paper = _as_dict(_as_dict(health.get("operational_readiness")).get("guarded_paper"))
    runtime = _as_dict(health.get("runtime_pressure"))
    storage = _as_dict(health.get("storage"))
    alert_count = _count_alerts(health, process)
    restart_due = bool(restart.get("restart_due", False))
    managed_restart_due = _managed_rolling_restart_advisory(restart, process)
    score = 100.0
    blockers: list[str] = []
    warnings: list[str] = []
    if not bool(health.get("ok", False)):
        score -= 30.0
        blockers.append("health_fast_not_ok")
    if not bool(health.get("strict_all_clear", False)):
        score -= 15.0
        warnings.append("strict_all_clear_false")
    if not bool(guarded_paper.get("ok", False)):
        score -= 20.0
        blockers.extend(str(item) for item in _as_list(guarded_paper.get("blockers")))
    if alert_count > 0:
        score -= min(25.0, 10.0 + alert_count * 3.0)
        blockers.append("process_alerts_active")
    if _status(runtime.get("overall_status")) in {"degraded", "blocked", "critical"}:
        score -= 15.0
        warnings.append(f"runtime_status={runtime.get('overall_status')}")
    if _status(storage.get("severity")) in {"high", "critical", "blocked"}:
        score -= 20.0
        blockers.append(f"storage_severity={storage.get('severity')}")
    if restart_due and managed_restart_due:
        warnings.append("rolling_restart_managed_creative_hold")
    elif restart_due:
        score -= 10.0
        warnings.append("rolling_restart_due")
    status = "ready" if score >= 92.0 and not blockers else ("advisory" if score >= 75.0 else "degraded")
    return _lane(
        "health_scorecard",
        score=score,
        status=status,
        sources=["health_fast", "process_watchdog", "rolling_restart"],
        project_root=project_root,
        source_payloads=sources,
        blockers=blockers,
        warnings=warnings,
        evidence={
            "health_fast_ok": bool(health.get("ok", False)),
            "strict_all_clear": bool(health.get("strict_all_clear", False)),
            "guarded_paper_ready": bool(guarded_paper.get("ok", False)),
            "process_alert_count": alert_count,
            "restart_due": restart_due,
            "rolling_restart_managed_advisory": managed_restart_due,
        },
        next_commands=[["./scripts/ops/opsctl.sh", "health-fast", "--json"], ["./scripts/session_ready_check.py"]],
    )


def _anti_degradation(project_root: Path, sources: dict[str, dict[str, Any]]) -> dict[str, Any]:
    storage = sources["ingestion_storage"]
    runtime = sources["runtime_throttle"]
    process_fanout = sources["process_fanout"]
    command = sources["command_validity"]
    writer = sources["writer_cycle"]
    backpressure = _as_dict(storage.get("backpressure"))
    pressure = _safe_float(storage.get("pressure_index"), _safe_float(_as_dict(_as_dict(runtime.get("runtime_snapshot")).get("storage_pressure")).get("pressure_index"), 0.0))
    pending = _safe_int(backpressure.get("total_pending_lines"), 0)
    oldest = _safe_float(backpressure.get("oldest_pending_age_seconds"), 0.0)
    host = _safe_float(runtime.get("host_saturation_score"), 0.0)
    runtime_guarded_ready = _runtime_guarded_ready(runtime)
    smoke_failures = len(_as_list(command.get("smoke_failures")))
    score = 100.0
    blockers: list[str] = []
    warnings: list[str] = []
    if _status(storage.get("overall_status")) not in {"ready", "ok", "advisory", ""}:
        score -= 20.0
        blockers.append(f"storage_status={storage.get('overall_status')}")
    if _status(storage.get("severity")) in {"high", "critical", "blocked"}:
        score -= 25.0
        blockers.append(f"storage_severity={storage.get('severity')}")
    if pressure >= 0.5:
        score -= min(25.0, pressure * 10.0)
        warnings.append("storage_pressure_index_above_target")
    if oldest > 240.0:
        score -= 10.0
        warnings.append("oldest_pending_age_above_240s")
    if pending >= 15000:
        score -= 20.0
        blockers.append("pending_lines_above_threshold")
    if _status(runtime.get("overall_status")) in {"degraded", "blocked", "critical"}:
        score -= 18.0
        warnings.append(f"runtime_status={runtime.get('overall_status')}")
    if host >= 60.0:
        if not runtime_guarded_ready:
            score -= 15.0
        warnings.append("host_saturation_guarded_or_hot")
    if _status(process_fanout.get("overall_status")) in {"degraded", "blocked", "critical"}:
        score -= 10.0
        warnings.append("process_fanout_not_ready")
    if smoke_failures > 0:
        score -= min(30.0, smoke_failures * 10.0)
        blockers.append("command_smoke_failures")
    if _status(writer.get("overall_status")) in {"degraded", "blocked", "critical"}:
        score -= 8.0
        warnings.append("writer_cycle_not_clear")
    status = "ready" if score >= 92.0 and not blockers else ("advisory" if score >= 75.0 else "degraded")
    return _lane(
        "anti_degradation_guardrails",
        score=score,
        status=status,
        sources=["ingestion_storage", "runtime_throttle", "process_fanout", "command_validity", "writer_cycle"],
        project_root=project_root,
        source_payloads=sources,
        blockers=blockers,
        warnings=warnings,
        evidence={
            "storage_pressure_index": round(pressure, 3),
            "total_pending_lines": pending,
            "oldest_pending_age_seconds": round(oldest, 3),
            "host_saturation_score": round(host, 3),
            "runtime_guarded_ready": runtime_guarded_ready,
            "command_smoke_failures": smoke_failures,
        },
        next_commands=[
            ["./scripts/ops/opsctl.sh", "runtime-throttle", "--apply", "--json"],
            ["./scripts/ops/opsctl.sh", "storage-backpressure-autopilot", "--apply", "--quick-bounded", "--json"],
            ["./scripts/ops/opsctl.sh", "writer-cycle-coordinator", "--apply", "--json"],
        ],
    )


def _paper_performance(project_root: Path, sources: dict[str, dict[str, Any]]) -> dict[str, Any]:
    perf = sources["paper_performance"]
    profit = sources["paper_profitability"]
    ramp = sources["paper_ramp"]
    guard = sources["runtime_paper_guard"]
    summary = _as_dict(profit.get("paper_summary"))
    sleeve_latest = _as_list(perf.get("sleeve_latest"))
    executions = _safe_int(summary.get("executions"), 0)
    net_pnl = _safe_float(summary.get("ending_net_pnl_total"), 0.0)
    score = 55.0
    blockers: list[str] = []
    warnings: list[str] = []
    if bool(profit.get("ok", False)):
        score += 15.0
    if _fresh_enough(project_root, sources, "paper_profitability", 480.0):
        score += 10.0
    else:
        warnings.append("paper_profitability_stale")
    if _fresh_enough(project_root, sources, "paper_performance", 720.0):
        score += 10.0
    else:
        warnings.append("paper_performance_stale")
    if executions > 0:
        score += 10.0
    else:
        warnings.append("no_current_paper_executions")
    if sleeve_latest:
        score += 5.0
    if bool(ramp.get("armed", False)) or _status(ramp.get("stage")) in {"armed", "ready"}:
        score += 10.0
    else:
        blockers.append("paper_ramp_not_armed")
    if _status(guard.get("overall_status")) in {"blocked", "critical"}:
        blockers.append("runtime_paper_guard_blocked")
        score -= 20.0
    if net_pnl < 0:
        warnings.append("paper_net_pnl_negative_monitor_attribution")
    status = "ready" if score >= 92.0 and not blockers else ("advisory" if score >= 75.0 else "degraded")
    return _lane(
        "paper_performance_attribution",
        score=score,
        status=status,
        sources=["paper_performance", "paper_profitability", "sleeve_profitability", "paper_ramp", "runtime_paper_guard"],
        project_root=project_root,
        source_payloads=sources,
        blockers=blockers,
        warnings=warnings,
        evidence={
            "executions": executions,
            "ending_net_pnl_total": round(net_pnl, 4),
            "sleeve_latest_count": len(sleeve_latest),
            "paper_ramp_stage": ramp.get("stage"),
            "paper_ramp_armed": bool(ramp.get("armed", False)),
        },
        next_commands=[
            ["./scripts/ops/opsctl.sh", "paper-profitability-control", "--apply", "--json"],
            ["./scripts/ops/opsctl.sh", "paper-performance", "--json"],
            ["./scripts/ops/opsctl.sh", "sleeve-profitability-dashboard", "--json"],
        ],
    )


def _account_positions(project_root: Path, sources: dict[str, dict[str, Any]]) -> dict[str, Any]:
    study = sources["account_position"]
    policy = sources["account_policy"]
    snapshot = sources["account_snapshot"]
    roll_watch = _as_dict(study.get("covered_call_roll_watch"))
    account_count = _safe_int(study.get("account_count"), 0)
    position_count = _safe_int(study.get("position_count"), 0)
    covered_calls = _safe_int(roll_watch.get("covered_call_count"), 0)
    alert_count = _safe_int(roll_watch.get("alert_count"), 0)
    score = 45.0
    blockers: list[str] = []
    warnings: list[str] = []
    if bool(study.get("ok", False)):
        score += 15.0
    if account_count > 0:
        score += 15.0
    else:
        blockers.append("no_visible_accounts")
    if position_count > 0:
        score += 15.0
    else:
        warnings.append("no_visible_positions")
    if covered_calls > 0:
        score += 10.0
    else:
        warnings.append("no_covered_calls_detected")
    if _fresh_enough(project_root, sources, "account_position", 720.0):
        score += 5.0
    else:
        warnings.append("account_position_study_stale")
    if bool(policy):
        score += 5.0
    if alert_count > 0:
        score -= min(20.0, alert_count * 5.0)
        warnings.append("covered_call_roll_alerts_active")
    if _status(snapshot.get("overall_status")) in {"blocked", "critical"}:
        blockers.append("account_snapshot_refresh_blocked")
    status = "ready" if score >= 92.0 and not blockers else ("advisory" if score >= 75.0 else "degraded")
    return _lane(
        "account_position_intelligence",
        score=score,
        status=status,
        sources=["account_position", "account_policy", "account_snapshot"],
        project_root=project_root,
        source_payloads=sources,
        blockers=blockers,
        warnings=warnings,
        evidence={
            "account_count": account_count,
            "position_count": position_count,
            "covered_call_count": covered_calls,
            "covered_call_roll_alert_count": alert_count,
            "roll_watch_status": roll_watch.get("overall_status"),
        },
        next_commands=[
            ["./scripts/ops/opsctl.sh", "account-position-study", "--json"],
            ["./scripts/ops/opsctl.sh", "covered-call-roll-watch", "--json"],
            ["./scripts/ops/opsctl.sh", "account-policy-context", "--json"],
        ],
    )


def _livefeed(project_root: Path, sources: dict[str, dict[str, Any]]) -> dict[str, Any]:
    feed = sources["livefeed_local"]
    refresh_guard = sources.get("livefeed_refresh_guard", {})
    alive = bool(feed.get("alive", False))
    running = _status(feed.get("status")) == "running"
    age = _source_age(project_root, sources, "livefeed_local")
    guard_age = _source_age(project_root, sources, "livefeed_refresh_guard")
    score = 100.0
    blockers: list[str] = []
    warnings: list[str] = []
    if not alive or not running:
        score -= 45.0
        blockers.append("livefeed_not_running")
    if age is None or age > 10.0:
        score -= 20.0
        warnings.append("livefeed_health_stale")
    if _safe_int(feed.get("skipped_unreadable_count"), 0) > 0:
        score -= 8.0
        warnings.append("livefeed_skipped_unreadable_files")
    if _safe_int(feed.get("stale_count"), 0) > 0:
        score -= 8.0
        warnings.append("livefeed_stale_sources")
    if refresh_guard and not bool(refresh_guard.get("ok", False)):
        score -= 12.0
        warnings.append("livefeed_refresh_guard_not_ready")
    if guard_age is None or guard_age > 60.0:
        score -= 5.0
        warnings.append("livefeed_refresh_guard_stale")
    status = "ready" if score >= 92.0 and not blockers else ("advisory" if score >= 75.0 else "degraded")
    return _lane(
        "livefeed_quality",
        score=score,
        status=status,
        sources=["livefeed_local", "livefeed_refresh_guard"],
        project_root=project_root,
        source_payloads=sources,
        blockers=blockers,
        warnings=warnings,
        evidence={
            "status": feed.get("status"),
            "alive": alive,
            "source": feed.get("source"),
            "heavy": feed.get("heavy"),
            "age_minutes": age,
            "refresh_guard_status": refresh_guard.get("overall_status"),
            "refresh_guard_routes": f"{refresh_guard.get('route_ok_count', 0)}/{refresh_guard.get('route_count', 0)}",
            "refresh_guard_age_minutes": guard_age,
        },
        next_commands=[
            ["./scripts/ops/opsctl.sh", "livefeed-refresh-guard", "--apply", "--json"],
            ["./scripts/ops/opsctl.sh", "livefeed-refresh"],
            ["./scripts/ops/opsctl.sh", "feed", "--source", "all", "--heavy"],
        ],
    )


def _event_mode(project_root: Path, sources: dict[str, dict[str, Any]]) -> dict[str, Any]:
    ipo = sources["spacex_ipo_watch"]
    macro = sources["macro_event"]
    event_store = sources["event_store"]
    ipo_status = _payload_status(ipo)
    score = _score_from_status(ipo_status, ok=ipo.get("ok") if isinstance(ipo.get("ok"), bool) else None)
    blockers: list[str] = []
    warnings: list[str] = []
    quote = _as_dict(ipo.get("quote"))
    if _status(ipo.get("policy")) != "monitoring_only_no_order_instruction":
        score -= 10.0
        warnings.append("event_policy_not_explicitly_monitoring_only")
    if ipo_status == "waiting_for_first_quote":
        warnings.append("ipo_watch_waiting_for_first_quote")
    if quote and not bool(quote.get("ok", False)):
        warnings.append(str(quote.get("error") or "quote_not_ready"))
    if not _fresh_enough(project_root, sources, "spacex_ipo_watch", 30.0):
        score -= 15.0
        warnings.append("ipo_watch_stale")
    if not macro:
        score -= 8.0
        warnings.append("macro_event_context_missing")
    if not event_store:
        score -= 8.0
        warnings.append("point_in_time_event_store_missing")
    status = "ready" if score >= 92.0 and not blockers else ("advisory" if score >= 75.0 else "degraded")
    return _lane(
        "event_mode",
        score=score,
        status=status,
        sources=["spacex_ipo_watch", "macro_event", "event_store"],
        project_root=project_root,
        source_payloads=sources,
        blockers=blockers,
        warnings=warnings,
        evidence={
            "symbol": ipo.get("symbol"),
            "ipo_watch_status": ipo_status,
            "quote_ok": bool(quote.get("ok", False)) if quote else False,
            "alert_triggered": bool(_as_dict(ipo.get("alert")).get("triggered", False)),
            "proxy_symbols": _as_list(ipo.get("proxy_symbols"))[:12],
        },
        next_commands=[["./scripts/ops/opsctl.sh", "spacex-ipo-watch", "--json"], ["./scripts/ops/opsctl.sh", "macro-crosscheck", "--json"]],
    )


def _notifications(project_root: Path, sources: dict[str, dict[str, Any]]) -> dict[str, Any]:
    watch = sources["notification_watch"]
    ladder = sources["notification_ladder"]
    pid_alive = _pid_alive(project_root / "governance" / "health" / "mac_notification_watch.pid")
    last_delivery = _as_dict(watch.get("last_delivery"))
    imessage = _as_dict(last_delivery.get("imessage"))
    mac = _as_dict(last_delivery.get("mac"))
    imessage_ok = bool(imessage) and _safe_int(imessage.get("returncode"), 1) == 0
    mac_ok = bool(mac) and _safe_int(mac.get("returncode"), 1) == 0
    running = _status(watch.get("status")) == "running" or pid_alive
    age = _source_age(project_root, sources, "notification_watch")
    score = 100.0
    blockers: list[str] = []
    warnings: list[str] = []
    if not running:
        score -= 35.0
        blockers.append("notification_watch_not_running")
    if not imessage_ok:
        score -= 25.0
        blockers.append("imessage_last_delivery_not_confirmed")
    if not mac_ok:
        score -= 10.0
        warnings.append("mac_notification_last_delivery_not_confirmed")
    if age is None or age > 60.0:
        score -= 12.0
        warnings.append("notification_state_stale")
    if not ladder:
        score -= 8.0
        warnings.append("notification_escalation_ladder_missing")
    status = "ready" if score >= 92.0 and not blockers else ("advisory" if score >= 75.0 else "degraded")
    return _lane(
        "notification_reliability",
        score=score,
        status=status,
        sources=["notification_watch", "notification_ladder"],
        project_root=project_root,
        source_payloads=sources,
        blockers=blockers,
        warnings=warnings,
        evidence={
            "watch_status": watch.get("status"),
            "watch_pid_alive": pid_alive,
            "imessage_last_delivery_ok": imessage_ok,
            "mac_last_delivery_ok": mac_ok,
            "state_age_minutes": age,
            "imessage_attempted": bool(last_delivery.get("imessage_attempted", False)),
        },
        next_commands=[["./scripts/ops/opsctl.sh", "notify-watch", "--enable-imessage"], ["./scripts/ops/opsctl.sh", "notify-test", "--enable-imessage"]],
    )


def _promotion(project_root: Path, sources: dict[str, dict[str, Any]]) -> dict[str, Any]:
    quality = sources["promotion_quality"]
    packet = sources["promotion_packet"]
    pipeline = sources["promotion_pipeline"]
    repair = _as_dict(packet.get("readiness_repair_contract"))
    critical_repairs = _safe_int(repair.get("critical_repair_gate_count"), 0)
    warning_repairs = _safe_int(repair.get("warning_repair_gate_count"), 0)
    promotion_ready = bool(packet.get("promotion_ready", False) or packet.get("canary_packet_ready", False))
    score = 100.0
    blockers: list[str] = []
    warnings: list[str] = []
    if not bool(quality.get("ok", False)):
        score -= 25.0
        blockers.append("promotion_quality_gate_not_ok")
    if _status(quality.get("overall_status")) in {"degraded", "blocked", "critical"}:
        score -= 12.0
        warnings.append(f"promotion_quality_status={quality.get('overall_status')}")
    if not promotion_ready:
        score -= 15.0
        warnings.append("promotion_packet_not_ready")
    if critical_repairs > 0:
        score -= 25.0
        blockers.append("critical_promotion_repairs_active")
    if warning_repairs > 0:
        score -= 8.0
        warnings.append("promotion_warning_repairs_active")
    if _status(pipeline.get("overall_status")) in {"degraded", "blocked", "critical"}:
        score -= 10.0
        warnings.append("promotion_pipeline_not_ready")
    status = "ready" if score >= 92.0 and not blockers else ("advisory" if score >= 75.0 else "degraded")
    return _lane(
        "promotion_discipline",
        score=score,
        status=status,
        sources=["promotion_quality", "promotion_packet", "promotion_pipeline"],
        project_root=project_root,
        source_payloads=sources,
        blockers=blockers,
        warnings=warnings,
        evidence={
            "promotion_quality_ok": bool(quality.get("ok", False)),
            "promotion_ready": promotion_ready,
            "critical_repair_gate_count": critical_repairs,
            "warning_repair_gate_count": warning_repairs,
            "autopilot_state": packet.get("autopilot_state"),
        },
        next_commands=[["./scripts/ops/opsctl.sh", "promotion-quality-gate", "--json"], ["./scripts/ops/opsctl.sh", "promotion-autopilot", "--json"]],
    )


def _dependency_freeze(project_root: Path, sources: dict[str, dict[str, Any]]) -> dict[str, Any]:
    release = sources["release_freeze"]
    runtime_dep = sources["runtime_dependency"]
    library = sources["library_router"]
    mlx = sources["mlx_router"]
    mlx_upgrade = sources["mlx_upgrade"]
    lock_path = project_root / "config" / "requirements.lock.txt"
    score = 100.0
    blockers: list[str] = []
    warnings: list[str] = []
    for name, payload in (
        ("release_freeze", release),
        ("runtime_dependency", runtime_dep),
        ("library_router", library),
        ("mlx_router", mlx),
        ("mlx_upgrade", mlx_upgrade),
    ):
        status = _payload_status(payload)
        if status in {"blocked", "critical", "failed"}:
            score -= 25.0
            blockers.append(f"{name}_blocked")
        elif status in {"degraded", "needs_work"}:
            score -= 12.0
            warnings.append(f"{name}_degraded")
        elif not payload:
            score -= 8.0
            warnings.append(f"{name}_missing")
    if not lock_path.exists():
        score -= 20.0
        blockers.append("requirements_lock_missing")
    if not _fresh_enough(project_root, sources, "runtime_dependency", 7 * 24 * 60):
        score -= 5.0
        warnings.append("runtime_dependency_profile_stale")
    status = "ready" if score >= 92.0 and not blockers else ("advisory" if score >= 75.0 else "degraded")
    return _lane(
        "platform_dependency_freeze",
        score=score,
        status=status,
        sources=["release_freeze", "runtime_dependency", "library_router", "mlx_router", "mlx_upgrade"],
        project_root=project_root,
        source_payloads=sources,
        blockers=blockers,
        warnings=warnings,
        evidence={
            "requirements_lock_present": lock_path.exists(),
            "requirements_lock_path": str(lock_path),
            "release_freeze_status": _payload_status(release),
            "runtime_dependency_status": _payload_status(runtime_dep),
            "library_router_status": _payload_status(library),
            "mlx_router_status": _payload_status(mlx),
        },
        next_commands=[
            ["./scripts/freeze_env_snapshot.sh"],
            ["./scripts/ops/opsctl.sh", "runtime-dependency-profiles", "--json"],
            ["./scripts/ops/opsctl.sh", "mlx-intelligence-router", "--apply", "--json"],
            ["./scripts/ops/opsctl.sh", "library-utilization-router", "--apply", "--json"],
        ],
    )


def _disaster_recovery(project_root: Path, sources: dict[str, dict[str, Any]]) -> dict[str, Any]:
    dr = sources["storage_dr"]
    restart = sources["rolling_restart"]
    process = sources["process_watchdog"]
    settlement = sources["post_restart"]
    replay = sources["paper_replay_drill"]
    managed_restart_due = _managed_rolling_restart_advisory(restart, process)
    score = 100.0
    blockers: list[str] = []
    warnings: list[str] = []
    if not bool(dr.get("ok", False)):
        score -= 30.0
        blockers.append("storage_disaster_recovery_not_ok")
    if _status(dr.get("overall_status")) in {"degraded", "blocked", "critical"}:
        if _managed_storage_dr_advisory(dr):
            warnings.append("storage_dr_managed_local_fallback")
        else:
            score -= 15.0
            warnings.append(f"storage_dr_status={dr.get('overall_status')}")
    if bool(restart.get("restart_due", False)) and managed_restart_due:
        warnings.append("rolling_restart_managed_creative_hold")
    elif bool(restart.get("restart_due", False)):
        score -= 12.0
        warnings.append("rolling_restart_due")
    if _status(restart.get("overall_status")) in {"degraded", "blocked", "critical"} and not managed_restart_due:
        score -= 10.0
        warnings.append("rolling_restart_not_ready")
    if settlement and not bool(settlement.get("ok", False)):
        score -= 8.0
        warnings.append("post_restart_settlement_not_ok")
    if replay and _status(replay.get("overall_status")) in {"degraded", "blocked", "critical"}:
        score -= 8.0
        warnings.append("paper_replay_drill_not_ready")
    status = "ready" if score >= 92.0 and not blockers else ("advisory" if score >= 75.0 else "degraded")
    return _lane(
        "disaster_recovery",
        score=score,
        status=status,
        sources=["storage_dr", "rolling_restart", "post_restart", "paper_replay_drill"],
        project_root=project_root,
        source_payloads=sources,
        blockers=blockers,
        warnings=warnings,
        evidence={
            "storage_dr_ok": bool(dr.get("ok", False)),
            "storage_dr_managed_advisory": _managed_storage_dr_advisory(dr),
            "rolling_restart_due": bool(restart.get("restart_due", False)),
            "rolling_restart_managed_advisory": managed_restart_due,
            "restart_scope": restart.get("recommended_scope"),
            "post_restart_ok": bool(settlement.get("ok", False)) if settlement else None,
        },
        next_commands=[["./scripts/ops/opsctl.sh", "storage-disaster-recovery", "--apply", "--json"], ["./scripts/ops/opsctl.sh", "rolling-restart", "--json"]],
    )


LANE_BUILDERS: list[Callable[[Path, dict[str, dict[str, Any]]], dict[str, Any]]] = [
    _health_scorecard,
    _anti_degradation,
    _paper_performance,
    _account_positions,
    _livefeed,
    _event_mode,
    _notifications,
    _promotion,
    _dependency_freeze,
    _disaster_recovery,
]


def _run_json(cmd: list[str], *, cwd: Path, timeout_sec: int) -> dict[str, Any]:
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(cwd),
            capture_output=True,
            text=True,
            check=False,
            timeout=max(int(timeout_sec), 1),
        )
        stdout = proc.stdout or ""
        stderr = proc.stderr or ""
        rc = int(proc.returncode)
        timed_out = False
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout.decode("utf-8", errors="ignore") if isinstance(exc.stdout, bytes) else str(exc.stdout or "")
        stderr = exc.stderr.decode("utf-8", errors="ignore") if isinstance(exc.stderr, bytes) else str(exc.stderr or "")
        rc = 124
        timed_out = True
    payload: dict[str, Any] = {}
    for line in reversed([row.strip() for row in stdout.splitlines() if row.strip()]):
        try:
            parsed = json.loads(line)
        except Exception:
            continue
        if isinstance(parsed, dict):
            payload = parsed
            break
    return {
        "cmd": cmd,
        "rc": rc,
        "ok": rc == 0,
        "timed_out": timed_out,
        "payload": payload,
        "stdout_tail": "\n".join(stdout.splitlines()[-8:]),
        "stderr_tail": "\n".join(stderr.splitlines()[-8:]),
    }


def _refresh_sources(project_root: Path, *, timeout_sec: int) -> list[dict[str, Any]]:
    opsctl = str(project_root / "scripts" / "ops" / "opsctl.sh")
    results: list[dict[str, Any]] = []
    for args in SAFE_REFRESH_COMMANDS:
        result = _run_json([opsctl, *args], cwd=project_root, timeout_sec=timeout_sec)
        results.append(result)
    return results


def _markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# A+ Operating Packet",
        "",
        f"- Timestamp: `{payload.get('timestamp_utc', '')}`",
        f"- Overall grade: `{payload.get('overall_grade', '')}`",
        f"- Overall score: `{payload.get('overall_score', '')}`",
        f"- A+ ready: `{payload.get('a_plus_ready', False)}`",
        "",
        "| Lane | Grade | Score | Status | Blockers |",
        "| --- | ---: | ---: | --- | --- |",
    ]
    for lane in _as_list(payload.get("lanes")):
        if not isinstance(lane, dict):
            continue
        blockers = ", ".join(str(item) for item in _as_list(lane.get("blockers"))) or "-"
        lines.append(
            f"| {lane.get('label', lane.get('id', ''))} | `{lane.get('grade', '')}` | "
            f"{lane.get('score', '')} | `{lane.get('status', '')}` | {blockers} |"
        )
    commands = _as_list(payload.get("next_safe_commands"))
    if commands:
        lines.extend(["", "## Next Safe Commands", ""])
        for cmd in commands:
            if isinstance(cmd, list):
                lines.append(f"- `{' '.join(str(part) for part in cmd)}`")
            else:
                lines.append(f"- `{cmd}`")
    return "\n".join(lines) + "\n"


def build_payload(
    project_root: Path = PROJECT_ROOT,
    *,
    apply: bool = False,
    timeout_sec: int = 180,
) -> dict[str, Any]:
    refresh_results = _refresh_sources(project_root, timeout_sec=timeout_sec) if apply else []
    sources = _load_sources(project_root)
    lanes = [builder(project_root, sources) for builder in LANE_BUILDERS]
    weighted_total = sum(float(lane["score"]) * LANE_WEIGHTS.get(str(lane["id"]), 1.0) for lane in lanes)
    weight_total = sum(LANE_WEIGHTS.get(str(lane["id"]), 1.0) for lane in lanes) or 1.0
    overall_score = _cap_score(weighted_total / weight_total)
    blocker_count = sum(len(_as_list(lane.get("blockers"))) for lane in lanes)
    warning_count = sum(len(_as_list(lane.get("warnings"))) for lane in lanes)
    a_plus_lanes = [lane for lane in lanes if bool(lane.get("a_plus", False))]
    non_a_plus_lanes = [lane for lane in lanes if not bool(lane.get("a_plus", False))]
    a_plus_ready = bool(overall_score >= 92.0 and len(non_a_plus_lanes) == 0 and blocker_count == 0)
    next_commands: list[list[str]] = []
    for lane in non_a_plus_lanes:
        next_commands.extend(cmd for cmd in _as_list(lane.get("next_commands")) if isinstance(cmd, list))
    next_commands = next_commands[:12]
    return {
        "timestamp_utc": iso_now(),
        "schema_version": 1,
        "ok": a_plus_ready,
        "overall_status": "ready" if a_plus_ready else ("advisory" if overall_score >= 85.0 and blocker_count == 0 else "needs_work"),
        "overall_score": overall_score,
        "overall_grade": _grade(overall_score),
        "a_plus_ready": a_plus_ready,
        "lane_count": len(lanes),
        "a_plus_lane_count": len(a_plus_lanes),
        "non_a_plus_lane_count": len(non_a_plus_lanes),
        "blocker_count": blocker_count,
        "warning_count": warning_count,
        "lanes": lanes,
        "next_safe_commands": next_commands,
        "apply_requested": bool(apply),
        "refresh_results": refresh_results,
        "policy": {
            "live_execution_enabled": False,
            "paper_live_data_allowed": True,
            "orders_never_enabled_by_this_packet": True,
            "a_plus_definition": "overall_score>=92 and every lane grade is A+ or better with no blockers",
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the A+ operating packet across health, guardrails, paper, accounts, livefeed, events, notifications, promotion, dependencies, and DR.")
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--out-file", default=str(DEFAULT_OUT_PATH))
    parser.add_argument("--md-out", default=str(DEFAULT_MD_OUT_PATH))
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--timeout-sec", type=int, default=180)
    parser.add_argument("--no-md", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    payload = build_payload(project_root, apply=bool(args.apply), timeout_sec=int(args.timeout_sec))
    out_file = Path(args.out_file).expanduser()
    write_payload(out_file, payload)
    if not args.no_md:
        Path(args.md_out).expanduser().parent.mkdir(parents=True, exist_ok=True)
        Path(args.md_out).expanduser().write_text(_markdown(payload), encoding="utf-8")
    if args.json:
        print(json.dumps(payload, ensure_ascii=True, indent=2))
    else:
        print(
            "a_plus_operating_packet "
            f"overall_status={payload['overall_status']} "
            f"grade={payload['overall_grade']} "
            f"score={payload['overall_score']} "
            f"a_plus_ready={payload['a_plus_ready']} "
            f"a_plus_lanes={payload['a_plus_lane_count']}/{payload['lane_count']} "
            f"blockers={payload['blocker_count']}"
        )
    return 0 if payload.get("overall_status") in {"ready", "advisory", "needs_work"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
