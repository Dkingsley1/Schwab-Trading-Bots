#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_PATH = PROJECT_ROOT / "governance" / "health" / "operator_cockpit_latest.json"
DEFAULT_MARKDOWN_PATH = PROJECT_ROOT / "exports" / "reports" / "operator" / "operator_cockpit_latest.md"


def _load_json(path: Path) -> dict[str, Any]:
    candidates = [path]
    try:
        rel_path = path.relative_to(PROJECT_ROOT)
    except Exception:
        rel_path = None
    if rel_path is not None and rel_path.parts and rel_path.parts[0] in {"data", "decisions", "decision_explanations", "exports", "governance", "logs", "models"}:
        candidates.append(PROJECT_ROOT / "local_fallback_storage" / rel_path)
        external_root = Path(os.getenv("BOT_LOGS_EXTERNAL_PROJECT_ROOT", "/Volumes/BOT_LOGS/schwab_trading_bot")).expanduser()
        candidates.append(external_root / rel_path)

    best_payload: dict[str, Any] = {}
    best_mtime = -1.0
    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        try:
            payload = json.loads(candidate.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(payload, dict):
            continue
        try:
            mtime = candidate.stat().st_mtime
        except Exception:
            mtime = 0.0
        if mtime >= best_mtime:
            best_payload = payload
            best_mtime = mtime
    return best_payload


def _ordered_unique(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for raw in items:
        text = str(raw or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _payload_status(payload: dict[str, Any], default: str = "missing") -> str:
    if not payload:
        return default
    return str(payload.get("overall_status") or payload.get("status") or default).strip() or default


def _is_storage_steady(storage: dict[str, Any], backlog_drain: dict[str, Any]) -> bool:
    status = _payload_status(storage, "")
    severity = str(storage.get("severity") or "").strip().lower()
    pressure_index = _safe_float(storage.get("pressure_index"), 1.0)
    backpressure = storage.get("backpressure") if isinstance(storage.get("backpressure"), dict) else {}
    storage_details = storage.get("storage") if isinstance(storage.get("storage"), dict) else {}
    writer_shedding = storage.get("writer_shedding") if isinstance(storage.get("writer_shedding"), dict) else {}
    steady_state = storage.get("steady_state") if isinstance(storage.get("steady_state"), dict) else {}
    target_status = steady_state.get("target_status") if isinstance(steady_state.get("target_status"), dict) else {}
    queue = storage.get("queue_watermarks") if isinstance(storage.get("queue_watermarks"), dict) else {}
    breaches = queue.get("breaches") if isinstance(queue.get("breaches"), dict) else {}
    hard_breaches = breaches.get("hard") if isinstance(breaches.get("hard"), list) else []
    elevated_breaches = breaches.get("elevated") if isinstance(breaches.get("elevated"), list) else []
    total_pending = _safe_int(backpressure.get("total_pending_lines"), 0)
    core_pending = _safe_int(backpressure.get("core_pending_lines"), 0)
    drain_minutes = _safe_float(backpressure.get("estimated_total_drain_minutes"), 0.0)
    backlog_recommended = bool(
        storage_details.get("backlog_drain_recommended_now", False)
        or backlog_drain.get("recommended_now", False)
        or backlog_drain.get("material_drain_recommended", False)
    )
    effective_raw_live = backpressure.get("effective_raw_live") if isinstance(backpressure.get("effective_raw_live"), dict) else {}
    effective_source = str(backpressure.get("effective_raw_live_source") or effective_raw_live.get("source") or "").strip()
    overlay_managed_clear = bool(
        backpressure.get("overlay_adjusted", False)
        and backpressure.get("overlay_pressure_clear", False)
        and effective_source in {"fresh_empty_sql_ingestion_overlay", "sql_ingestion_overlay_pressure"}
        and total_pending <= 15000
        and core_pending <= 5000
    )
    if backlog_recommended and overlay_managed_clear:
        backlog_recommended = False
    steady_ready = bool(target_status.get("steady_state_ready", False)) if target_status else True
    return (
        status == "ready"
        and severity in {"", "stable", "low", "ready"}
        and pressure_index <= 0.25
        and total_pending <= 15000
        and core_pending <= 5000
        and (drain_minutes <= 15.0 or drain_minutes == 0.0)
        and not backlog_recommended
        and not hard_breaches
        and not elevated_breaches
        and not bool(writer_shedding.get("active", False))
        and steady_ready
    )


def _is_memory_healthy(memory: dict[str, Any]) -> bool:
    if not memory:
        return False
    snapshot = memory.get("memory_snapshot") if isinstance(memory.get("memory_snapshot"), dict) else {}
    status = _payload_status(memory, "")
    state = str(snapshot.get("memory_pressure_state") or "").strip().lower()
    kind = str(snapshot.get("memory_pressure_kind") or "").strip().lower()
    swap_used_gb = _safe_float(snapshot.get("swap_used_gb"), 99.0)
    return (
        status != "blocked"
        and state in {"green", "normal", "ok", "clear"}
        and kind in {"", "none", "green", "normal", "ok", "clear"}
        and swap_used_gb <= 8.0
    )


def _lease_status(auth_lease: dict[str, Any]) -> str:
    raw = _payload_status(auth_lease)
    if not auth_lease:
        return raw
    budget = auth_lease.get("lease_budget") if isinstance(auth_lease.get("lease_budget"), dict) else {}
    lease_state = str(auth_lease.get("lease_state") or "").strip().lower()
    expires = _safe_float(budget.get("expires_in_seconds"), 0.0)
    critical = _safe_float(budget.get("critical_lease_seconds"), 600.0)
    if raw == "ready" or lease_state == "healthy":
        return "ready"
    if raw == "degraded" and lease_state in {"warning", "renewing"} and expires > critical:
        return "advisory"
    return raw


def _runtime_separation_status(live_runtime_separation: dict[str, Any], *, storage_steady: bool, memory_healthy: bool) -> str:
    raw = _payload_status(live_runtime_separation)
    if not live_runtime_separation:
        return raw
    clearance = live_runtime_separation.get("clearance_plan") if isinstance(live_runtime_separation.get("clearance_plan"), dict) else {}
    pressure = live_runtime_separation.get("shared_host_pressure") if isinstance(live_runtime_separation.get("shared_host_pressure"), dict) else {}
    signals = pressure.get("signals") if isinstance(pressure.get("signals"), dict) else {}
    clearance_state = str(clearance.get("clearance_state") or "").strip()
    contention_score = _safe_int(pressure.get("contention_score"), 0)
    restart_storm = bool(signals.get("restart_storm_present", False))
    swap_elevated = bool(signals.get("swap_pressure_elevated", False))
    if (
        raw == "degraded"
        and storage_steady
        and memory_healthy
        and clearance_state in {"awaiting_coverage_cycles", "awaiting_cold_lane", "ready", "cleared"}
        and contention_score <= 3
        and not restart_storm
        and not swap_elevated
    ):
        return "advisory"
    return raw


def _snapshot_cache_status(snapshot_cache: dict[str, Any]) -> str:
    raw = _payload_status(snapshot_cache)
    cache = snapshot_cache.get("cache_health") if isinstance(snapshot_cache.get("cache_health"), dict) else {}
    if raw == "degraded" and bool(cache.get("snapshot_ready", False)):
        return "advisory"
    return raw


def _rolling_restart_status(rolling_restart: dict[str, Any], *, storage_steady: bool, memory_healthy: bool) -> str:
    raw = _payload_status(rolling_restart)
    if not rolling_restart:
        return raw
    signals = rolling_restart.get("due_signals") if isinstance(rolling_restart.get("due_signals"), dict) else {}
    scope = str(rolling_restart.get("recommended_scope") or "").strip().lower()
    checkpoint_only = bool(signals.get("checkpoint_missing_or_stale", False)) and not any(
        bool(signals.get(key, False))
        for key in (
            "session_stale",
            "shadow_heartbeat_stale",
            "swap_pressure_high",
            "restart_storm_present",
        )
    )
    if raw in {"blocked", "degraded"} and storage_steady and memory_healthy and checkpoint_only and scope in {"", "none"}:
        return "advisory"
    return raw


def _artifact_freshness_status(artifact_freshness: dict[str, Any], *, storage_steady: bool, process_ready: bool) -> str:
    raw = _payload_status(artifact_freshness)
    if not artifact_freshness:
        return raw
    summary = artifact_freshness.get("sla_summary") if isinstance(artifact_freshness.get("sla_summary"), dict) else {}
    artifacts = artifact_freshness.get("artifacts") if isinstance(artifact_freshness.get("artifacts"), list) else []
    stale_required = _safe_int(summary.get("stale_required"), 0)
    stale_required_names = [
        str(row.get("name") or "")
        for row in artifacts
        if isinstance(row, dict) and bool(row.get("required", False)) and bool(row.get("stale", False))
    ]
    if raw in {"blocked", "degraded"} and storage_steady and process_ready and stale_required <= 2:
        if not stale_required_names or all(name in {"process_watchdog", "live_readiness_smoke"} for name in stale_required_names):
            return "advisory"
    return raw


def _storage_tier_status(storage_tier: dict[str, Any], *, storage_steady: bool) -> str:
    raw = _payload_status(storage_tier)
    pressure = storage_tier.get("pressure") if isinstance(storage_tier.get("pressure"), dict) else {}
    plan = storage_tier.get("upgrade_plan") if isinstance(storage_tier.get("upgrade_plan"), dict) else {}
    top_families = plan.get("top_hot_path_families") if isinstance(plan.get("top_hot_path_families"), list) else []
    sql_link_bytes = 0
    for row in top_families:
        if isinstance(row, dict) and str(row.get("family") or "") == "sql_link_shards":
            sql_link_bytes += _safe_int(row.get("bytes"), 0)
    over_budget = _safe_int(pressure.get("hot_path_over_budget_bytes"), 0)
    if raw in {"blocked", "degraded"} and storage_steady and over_budget > 0 and sql_link_bytes > 0:
        return "advisory"
    return raw


def _backlog_retry_status(backlog_retry_bot: dict[str, Any], *, storage_steady: bool, backlog_drain: dict[str, Any]) -> str:
    raw = _payload_status(backlog_retry_bot)
    drain_recommended = bool(backlog_drain.get("recommended_now", False) or backlog_drain.get("material_drain_recommended", False))
    if raw == "applied_with_followups" and storage_steady and not drain_recommended:
        return "advisory"
    return raw


def _master_infra_status(master_infra: dict[str, Any], *, storage_steady: bool, process_ready: bool) -> str:
    raw = _payload_status(master_infra)
    hardening = master_infra.get("hardening_scorecard") if isinstance(master_infra.get("hardening_scorecard"), dict) else {}
    operator_followups = master_infra.get("operator_followups") if isinstance(master_infra.get("operator_followups"), list) else []
    checks = master_infra.get("checks") if isinstance(master_infra.get("checks"), list) else []
    blocked_check_names = {
        str(row.get("name") or "")
        for row in checks
        if isinstance(row, dict) and str(row.get("status") or "").strip().lower() == "blocked"
    }
    production_proof_debt_only = bool(
        blocked_check_names
        and blocked_check_names.issubset({"autonomous_recovery_drills", "governance_artifact_freshness"})
    )
    core_hardening_ready = all(
        bool(hardening.get(key, False))
        for key in (
            "truth_layer_ready",
            "storage_route_certified",
            "process_ownership_canonical",
            "command_surface_clean",
            "launchd_jobs_installed",
        )
    )
    if (
        raw in {"blocked", "degraded"}
        and storage_steady
        and process_ready
        and core_hardening_ready
        and not operator_followups
        and (raw != "blocked" or production_proof_debt_only or not blocked_check_names)
    ):
        return "advisory"
    return raw


def _is_global_halt_clear(global_killswitch: dict[str, Any]) -> bool:
    if not global_killswitch:
        return True
    return not bool(global_killswitch.get("halt", False))


def _paper_soak_contract_ready(soak: dict[str, Any], paper_guard: dict[str, Any]) -> bool:
    return bool(
        _payload_status(soak, "") == "ready"
        and bool(soak.get("ok", False))
        and bool(soak.get("safe_to_leave_unattended", False))
        and _payload_status(paper_guard, "") == "ready"
        and bool(paper_guard.get("ok", False))
    )


def _paper_soak_managed_status(status: str, *, paper_soak_ready: bool) -> str:
    clean = str(status or "").strip()
    if not paper_soak_ready:
        return clean
    if clean in {
        "applied_with_followups",
        "blocked",
        "critical",
        "degraded",
        "missing",
        "needs_attention",
        "needs_coverage",
        "needs_cycles",
        "needs_review",
        "needs_work",
        "thin",
    }:
        return "managed_paper_soak"
    return clean


def _registry_counts(registry: dict[str, Any]) -> dict[str, int]:
    summary = registry.get("summary") if isinstance(registry.get("summary"), dict) else {}
    rows = registry.get("sub_bots") if isinstance(registry.get("sub_bots"), list) else []
    if not rows and isinstance(registry.get("bots"), list):
        rows = registry.get("bots") or []

    total_bots = _safe_int(summary.get("total_bots"), 0)
    active_bots = _safe_int(summary.get("active_bots"), 0)
    data_collection_bots = _safe_int(summary.get("data_collection_active_bots"), 0)
    sleeve_profile_count = _safe_int(summary.get("sleeve_profile_count"), 0)

    if rows:
        total_bots = len(rows)
        active_bots = sum(1 for row in rows if isinstance(row, dict) and bool(row.get("active", False)))
        data_collection_bots = sum(
            1 for row in rows if isinstance(row, dict) and bool(row.get("data_collection_active", False))
        )
        sleeve_profiles = {
            str(row.get("sleeve_profile") or "").strip()
            for row in rows
            if isinstance(row, dict) and str(row.get("sleeve_profile") or "").strip()
        }
        if sleeve_profiles:
            sleeve_profile_count = len(sleeve_profiles)

    return {
        "total_bots": total_bots,
        "active_bots": active_bots,
        "data_collection_active_bots": data_collection_bots,
        "sleeve_profile_count": sleeve_profile_count,
    }


def _render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Operator Cockpit",
        "",
        f"- Timestamp UTC: `{payload.get('timestamp_utc', '')}`",
        f"- Overall Status: `{payload.get('overall_status', '')}`",
        "",
        "## Immediate Focus",
        "",
    ]
    for item in payload.get("recommended_actions") or []:
        lines.append(f"- {item}")
    posture = payload.get("adaptive_posture") if isinstance(payload.get("adaptive_posture"), dict) else {}
    if posture:
        lines.extend(["", "## Adaptive Posture", ""])
        for key in (
            "overall_status",
            "live_collection_ready",
            "storage_steady",
            "memory_healthy",
            "global_halt_clear",
            "total_bots",
            "active_bots",
            "sleeve_profile_count",
        ):
            if key in posture:
                lines.append(f"- `{key}`: `{posture.get(key)}`")
    domains = payload.get("readiness_domains") if isinstance(payload.get("readiness_domains"), dict) else {}
    if domains:
        lines.extend(["", "## Readiness Domains", ""])
        for key, row in domains.items():
            if not isinstance(row, dict):
                continue
            lines.append(
                f"- `{key}`: `{row.get('status', '')}`"
                + (f" ({row.get('summary', '')})" if str(row.get("summary") or "").strip() else "")
            )
    scores = payload.get("maturity_scores") if isinstance(payload.get("maturity_scores"), dict) else {}
    if scores:
        lines.extend(["", "## Maturity Scores", ""])
        for key in (
            "feature_sophistication",
            "data_collection_breadth",
            "infrastructure_control_plane",
            "operational_cleanliness",
            "unattended_autonomy",
        ):
            if key in scores:
                lines.append(f"- `{key}`: `{scores.get(key)}`")
    hardening = payload.get("hardening_scorecard") if isinstance(payload.get("hardening_scorecard"), dict) else {}
    if hardening:
        lines.extend(["", "## Hardening Scorecard", ""])
        for key, value in hardening.items():
            lines.append(f"- `{key}`: `{int(bool(value))}`")
    lines.extend(["", "## Upgrade Lanes", ""])
    for key, row in (payload.get("upgrade_lanes") or {}).items():
        if not isinstance(row, dict):
            continue
        lines.append(
            f"- `{key}`: `{row.get('status', '')}`"
            + (f" ({row.get('summary', '')})" if str(row.get("summary") or "").strip() else "")
        )
    lines.extend(["", "## Long-Run Lanes", ""])
    for key, row in (payload.get("long_run_lanes") or {}).items():
        if not isinstance(row, dict):
            continue
        lines.append(
            f"- `{key}`: `{row.get('status', '')}`"
            + (f" ({row.get('summary', '')})" if str(row.get("summary") or "").strip() else "")
        )
    lines.extend(["", "## Key Surfaces", ""])
    for key, row in (payload.get("surfaces") or {}).items():
        if not isinstance(row, dict):
            continue
        lines.append(f"- `{key}`: `{row.get('status', '')}`")
    return "\n".join(lines) + "\n"


def build_payload(project_root: Path = PROJECT_ROOT) -> dict[str, Any]:
    health_root = project_root / "governance" / "health"
    runtime = _load_json(health_root / "runtime_gate_dashboard_latest.json")
    platform = _load_json(health_root / "platform_control_plane_latest.json")
    training = _load_json(health_root / "training_report_latest.json")
    training_quality = _load_json(health_root / "training_quality_control_latest.json")
    storage = _load_json(health_root / "ingestion_storage_control_latest.json")
    governor = _load_json(health_root / "ingestion_storage_governor_latest.json")
    backlog_drain = _load_json(health_root / "external_backlog_drain_latest.json")
    backlog_retry_bot = _load_json(health_root / "external_backlog_retry_bot_latest.json")
    queue = _load_json(health_root / "ingestion_priority_queue_latest.json")
    resilience = _load_json(health_root / "storage_resilience_control_latest.json")
    split_brain = _load_json(health_root / "storage_split_brain_reconciler_latest.json")
    requalification = _load_json(health_root / "training_requalification_latest.json")
    coverage_seed = _load_json(project_root / "governance" / "walk_forward" / "coverage_seed_latest.json")
    calibration = _load_json(health_root / "calibration_abstention_control_latest.json")
    paper_calibration = _load_json(health_root / "paper_execution_calibration_latest.json")
    remediation = _load_json(health_root / "daily_verify_auto_remediation_bot_latest.json")
    storage_tier = _load_json(health_root / "storage_tier_policy_latest.json")
    training_runtime = _load_json(health_root / "training_runtime_control_latest.json")
    regime_control = _load_json(health_root / "regime_control_plane_latest.json")
    supportability_control = _load_json(health_root / "supportability_control_latest.json")
    provider_mesh = _load_json(health_root / "provider_mesh_latest.json")
    service_control_plane = _load_json(health_root / "service_control_plane_latest.json")
    teacher_quality = _load_json(project_root / "governance" / "distillation" / "teacher_quality_latest.json")
    bot_quality_autopilot = _load_json(health_root / "bot_quality_autopilot_latest.json")
    infrastructure_autofix = _load_json(health_root / "infrastructure_autofix_bot_latest.json")
    live_runtime_separation = _load_json(health_root / "live_runtime_separation_control_latest.json")
    rolling_restart = _load_json(health_root / "rolling_restart_controller_latest.json")
    auth_lease = _load_json(health_root / "auth_lease_manager_latest.json")
    blackstart_recovery = _load_json(health_root / "blackstart_recovery_latest.json")
    sleeve_isolation = _load_json(health_root / "sleeve_isolation_guard_latest.json")
    artifact_freshness = _load_json(health_root / "artifact_freshness_slo_latest.json")
    snapshot_cache = _load_json(health_root / "runtime_snapshot_cache_control_latest.json")
    remote_alert = _load_json(health_root / "remote_alert_control_latest.json")
    storage_quota = _load_json(health_root / "storage_quota_guard_latest.json")
    release_freeze = _load_json(health_root / "release_freeze_guard_latest.json")
    roster_expansion = _load_json(health_root / "roster_expansion_slots_latest.json")
    roster_resilience = _load_json(health_root / "roster_resilience_planner_latest.json")
    chaos_drills = _load_json(health_root / "chaos_drill_coordinator_latest.json")
    master_infra = _load_json(health_root / "master_infrastructure_supervisor_latest.json")
    memory_efficiency = _load_json(health_root / "memory_efficiency_control_latest.json")
    system_self_model = _load_json(health_root / "system_self_model_latest.json")
    global_killswitch = _load_json(health_root / "global_killswitch_latest.json")
    process_watchdog = _load_json(health_root / "process_watchdog_latest.json")
    soak_readiness = _load_json(health_root / "unattended_soak_readiness_latest.json")
    paper_regression_guard = _load_json(health_root / "runtime_paper_regression_guard_latest.json")
    production_readiness = _load_json(health_root / "production_readiness_control_latest.json")
    production_soak = _load_json(health_root / "production_soak_enhancement_latest.json")
    registry = _load_json(project_root / "master_bot_registry.json")

    attention = runtime.get("overall", {}).get("attention") if isinstance(runtime.get("overall"), dict) else []
    process_lane_status = next(
        (
            str(row.get("status") or "")
            for row in (master_infra.get("checks") or [])
            if isinstance(row, dict) and row.get("name") == "process_lane_ownership"
        ),
        "missing",
    )
    paper_soak_ready = _paper_soak_contract_ready(soak_readiness, paper_regression_guard)
    process_watchdog_ready = _payload_status(process_watchdog, "") == "ready"
    process_ready = process_lane_status == "ready" or (paper_soak_ready and process_watchdog_ready)
    storage_steady = _is_storage_steady(storage, backlog_drain)
    if paper_soak_ready and _payload_status(storage, "") == "ready":
        storage_steady = True
    memory_healthy = _is_memory_healthy(memory_efficiency)
    global_halt_clear = _is_global_halt_clear(global_killswitch)
    storage_tier_lane_status = _storage_tier_status(storage_tier, storage_steady=storage_steady)
    runtime_separation_lane_status = _runtime_separation_status(
        live_runtime_separation,
        storage_steady=storage_steady,
        memory_healthy=memory_healthy,
    )
    snapshot_cache_lane_status = _snapshot_cache_status(snapshot_cache)
    auth_lease_lane_status = _lease_status(auth_lease)
    backlog_retry_lane_status = _backlog_retry_status(backlog_retry_bot, storage_steady=storage_steady, backlog_drain=backlog_drain)
    rolling_restart_lane_status = _rolling_restart_status(rolling_restart, storage_steady=storage_steady, memory_healthy=memory_healthy)
    artifact_freshness_lane_status = _artifact_freshness_status(artifact_freshness, storage_steady=storage_steady, process_ready=process_ready)
    master_infra_lane_status = _master_infra_status(master_infra, storage_steady=storage_steady, process_ready=process_ready)
    if paper_soak_ready and master_infra_lane_status in {"blocked", "degraded"} and process_ready:
        master_infra_lane_status = "advisory"

    expansion_session = memory_efficiency.get("expansion_session") if isinstance(memory_efficiency.get("expansion_session"), dict) else {}
    registry_counts = _registry_counts(registry)
    total_bots = registry_counts["total_bots"] or _safe_int(expansion_session.get("total_bots"), 0)
    active_bots = registry_counts["active_bots"] or _safe_int(expansion_session.get("active_bots"), 0)
    data_collection_bots = registry_counts["data_collection_active_bots"] or _safe_int(
        expansion_session.get("data_collection_active_bots"),
        0,
    )
    sleeve_profile_count = max(
        registry_counts["sleeve_profile_count"],
        _safe_int(expansion_session.get("sleeve_profile_count"), 0),
    )
    pressure_level = "massive" if active_bots >= 500 else str(expansion_session.get("pressure_level") or "standard")
    global_halt_summary = {
        "halt": bool(global_killswitch.get("halt", False)),
        "action": str(global_killswitch.get("action") or "none"),
        "reason_count": len(global_killswitch.get("reasons") or []) if isinstance(global_killswitch.get("reasons"), list) else 0,
    }

    raw_recommended_actions = _ordered_unique(
        list(attention or [])
        + list((storage_tier.get("upgrade_plan") or {}).get("recommended_actions") or [])
        + list((training_runtime.get("recommended_actions") or [])[:3])
        + list((provider_mesh.get("recommended_actions") or [])[:3])
        + list((service_control_plane.get("recommended_actions") or [])[:3])
        + list((regime_control.get("recommended_actions") or [])[:3])
        + list((roster_expansion.get("recommended_actions") or [])[:2])
        + list((supportability_control.get("recommended_actions") or [])[:3])
        + list((teacher_quality.get("recommended_actions") or [])[:2])
        + list((bot_quality_autopilot.get("recommended_actions") or [])[:2])
        + list((infrastructure_autofix.get("recommended_actions") or [])[:2])
        + list(storage.get("top_actions") or [])
        + list((governor.get("top_actions") or [])[:2])
        + list((backlog_drain.get("top_actions") or [])[:2])
        + list((backlog_retry_bot.get("recommended_actions") or [])[:2])
        + list((queue.get("top_actions") or [])[:2])
        + list((resilience.get("top_actions") or [])[:2])
        + list((requalification.get("recommended_actions") or [])[:2])
        + list((coverage_seed.get("recommended_actions") or [])[:2])
        + list((calibration.get("top_actions") or [])[:2])
        + list((paper_calibration.get("top_actions") or [])[:2])
        + list((remediation.get("recommended_actions") or [])[:2])
        + list((live_runtime_separation.get("recommended_actions") or [])[:2])
        + list((rolling_restart.get("recommended_actions") or [])[:2])
        + list((auth_lease.get("recommended_actions") or [])[:2])
        + list((blackstart_recovery.get("recommended_actions") or [])[:2])
        + list((sleeve_isolation.get("recommended_actions") or [])[:2])
        + list((artifact_freshness.get("recommended_actions") or [])[:2])
        + list((snapshot_cache.get("recommended_actions") or [])[:2])
        + list((remote_alert.get("recommended_actions") or [])[:2])
        + list((storage_quota.get("recommended_actions") or [])[:2])
        + list((release_freeze.get("recommended_actions") or [])[:2])
        + list((roster_resilience.get("recommended_actions") or [])[:2])
        + list((chaos_drills.get("recommended_actions") or [])[:2])
        + list((master_infra.get("operator_followups") or [])[:3])
    )
    suppressed_actions = {
        "external_backlog_drain_recommended",
        "external_backlog_retry_bot_followups",
    } if storage_steady else set()
    if memory_healthy:
        suppressed_actions.add("memory_efficiency_control_needs_work")
    if runtime_separation_lane_status == "advisory":
        suppressed_actions.add("live_runtime_separation_control_needs_work")
    if auth_lease_lane_status in {"ready", "advisory"}:
        suppressed_actions.add("auth_lease_manager_needs_work")
    if snapshot_cache_lane_status == "advisory":
        suppressed_actions.add("runtime_snapshot_cache_control_needs_work")
    if rolling_restart_lane_status == "advisory":
        suppressed_actions.add("rolling_restart_controller_blocked")
        suppressed_actions.add("rolling_restart_controller_needs_work")
    if artifact_freshness_lane_status == "advisory":
        suppressed_actions.add("artifact_freshness_slo_blocked")
        suppressed_actions.add("artifact_freshness_slo_needs_work")
    if master_infra_lane_status == "advisory":
        suppressed_actions.add("master_infrastructure_supervisor_blocked")
    if paper_soak_ready:
        suppressed_actions.update(
            {
                "daily_auto_verify_not_ok",
                "external_backlog_retry_bot_followups",
                "health_gates_stale",
                "promotion_not_ready",
                "training_quality_control_blocked",
            }
        )
    recommended_actions = [action for action in raw_recommended_actions if action not in suppressed_actions]

    adaptive_followups = _ordered_unique(
        [
            action
            for action in raw_recommended_actions
            if action in suppressed_actions
        ]
        + (
            [
                "keep the expanded bot fleet in adaptive collection mode while training, coverage, and cleanup debts clear asynchronously"
            ]
            if storage_steady and memory_healthy and active_bots >= 500
            else []
        )
    )

    hard_blockers: list[str] = []
    if not global_halt_clear:
        hard_blockers.append("global_halt_active")
    if str(storage.get("overall_status") or "") == "blocked" and not storage_steady:
        hard_blockers.append("ingestion_storage_control_blocked")
    if _safe_int(((split_brain.get("summary") or {}).get("unresolved_conflicts", 0))) > 0:
        hard_blockers.append("storage_split_brain_conflicts")
    if bool(((governor.get("sql_primary_db") or {}).get("route_drift", False))):
        hard_blockers.append("sql_primary_route_drift")
    if str(training_runtime.get("overall_status") or "") == "blocked" and not bool(training_runtime.get("snapshot_ready", False)):
        hard_blockers.append("training_runtime_snapshot_blocked")
    if str(supportability_control.get("overall_status") or "") == "blocked" and active_bots < 500:
        hard_blockers.append("supportability_control_blocked")
    if storage_tier_lane_status == "blocked":
        hard_blockers.append("storage_tier_policy_blocked")
    if str(provider_mesh.get("overall_status") or "") == "blocked":
        hard_blockers.append("provider_mesh_blocked")
    service_upgrade_lanes = service_control_plane.get("upgrade_lanes") if isinstance(service_control_plane.get("upgrade_lanes"), dict) else {}
    service_blocked_lanes = []
    for lane_name, lane_payload in service_upgrade_lanes.items():
        if not isinstance(lane_payload, dict):
            continue
        lane_status = str(lane_payload.get("status") or "")
        if lane_name == "runtime_separation" and runtime_separation_lane_status == "advisory":
            continue
        if lane_name == "operator_cockpit_contract":
            continue
        if lane_status == "blocked":
            service_blocked_lanes.append(lane_name)
    if str(service_control_plane.get("overall_status") or "") == "blocked" and service_blocked_lanes:
        hard_blockers.append("service_control_plane_blocked")
    for name, status in (
        ("live_runtime_separation_control", runtime_separation_lane_status),
        ("rolling_restart_controller", rolling_restart_lane_status),
        ("auth_lease_manager", auth_lease_lane_status),
        ("blackstart_recovery", _payload_status(blackstart_recovery)),
        ("sleeve_isolation_guard", _payload_status(sleeve_isolation)),
        ("artifact_freshness_slo", artifact_freshness_lane_status),
        ("runtime_snapshot_cache_control", snapshot_cache_lane_status),
        ("remote_alert_control", _payload_status(remote_alert)),
        ("storage_quota_guard", _payload_status(storage_quota)),
        ("chaos_drill_coordinator", _payload_status(chaos_drills)),
    ):
        if paper_soak_ready and name == "chaos_drill_coordinator" and status == "blocked":
            continue
        if status == "blocked":
            hard_blockers.append(f"{name}_blocked")
    if master_infra_lane_status == "blocked":
        hard_blockers.append("master_infrastructure_supervisor_blocked")

    live_collection_ready = storage_steady and memory_healthy and global_halt_clear and process_ready and not hard_blockers
    overall_status = "ready" if live_collection_ready else "degraded"
    adaptive_status = "stable_expansion" if live_collection_ready else "needs_attention"
    training_domain_status = "blocked" if str(training_quality.get("overall_status") or "") == "blocked" else _payload_status(training, "missing")
    if paper_soak_ready and training_domain_status == "blocked":
        training_domain_status = "managed_paper_soak"
    if "promotion_not_ready" in recommended_actions and training_domain_status == "ready":
        training_domain_status = "gated"
    readiness_domains = {
        "live_collection": {
            "status": "ready" if live_collection_ready else "degraded",
            "summary": f"storage_steady={int(storage_steady)} memory_healthy={int(memory_healthy)} halt_clear={int(global_halt_clear)}",
        },
        "training_and_promotion": {
            "status": training_domain_status,
            "summary": f"training_quality={str(training_quality.get('overall_status') or 'missing')} promotion_gated={int('promotion_not_ready' in raw_recommended_actions)}",
        },
        "expansion_scaling": {
            "status": "ready" if storage_steady and memory_healthy and active_bots >= 500 else "degraded",
            "summary": f"active_bots={active_bots} data_collection_bots={data_collection_bots} pressure_level={pressure_level}",
        },
        "storage_backpressure": {
            "status": "ready" if storage_steady else _payload_status(storage),
            "summary": (
                f"pressure_index={_safe_float(storage.get('pressure_index'), 0.0):.3f} "
                f"pending_lines={_safe_int(((storage.get('backpressure') or {}).get('total_pending_lines', 0)))}"
            ),
        },
        "operator_autonomy": {
            "status": "ready" if master_infra_lane_status in {"ready", "advisory"} and process_ready else "degraded",
            "summary": f"master_infra={master_infra_lane_status} process_lane_ownership={process_lane_status}",
        },
    }
    adaptive_posture = {
        "overall_status": adaptive_status,
        "paper_soak_ready": paper_soak_ready,
        "live_collection_ready": live_collection_ready,
        "storage_steady": storage_steady,
        "memory_healthy": memory_healthy,
        "global_halt_clear": global_halt_clear,
        "process_lane_ownership_ready": process_ready,
        "hard_blockers": hard_blockers,
        "suppressed_advisories": _ordered_unique(sorted(suppressed_actions)),
        "adaptive_followups": adaptive_followups,
        "total_bots": total_bots,
        "active_bots": active_bots,
        "data_collection_active_bots": data_collection_bots,
        "sleeve_profile_count": sleeve_profile_count,
        "pressure_level": pressure_level,
        "global_halt": global_halt_summary,
    }
    recommended_actions = _ordered_unique(
        recommended_actions
        + [action for action in adaptive_followups if action not in suppressed_actions]
    )[:14]
    managed_proof_debt = _ordered_unique(
        [
            name
            for name, status in {
                "training_report": _payload_status(training, ""),
                "training_quality_control": _payload_status(training_quality, ""),
                "coverage_seeding": _payload_status(coverage_seed, ""),
                "regime_control_plane": _payload_status(regime_control, ""),
                "bot_quality_autopilot": _payload_status(bot_quality_autopilot, ""),
                "infrastructure_autofix_bot": _payload_status(infrastructure_autofix, ""),
                "roster_resilience_planner": _payload_status(roster_resilience, ""),
                "chaos_drill_coordinator": _payload_status(chaos_drills, ""),
                "master_infrastructure_supervisor": _payload_status(master_infra, ""),
                "system_self_model": _payload_status(system_self_model, ""),
            }.items()
            if _paper_soak_managed_status(status, paper_soak_ready=paper_soak_ready) == "managed_paper_soak"
        ]
    )

    upgrade_lanes = {
        "storage_split": {
            "status": storage_tier_lane_status,
            "summary": (
                f"hot_path_over_budget_bytes={int(((storage_tier.get('pressure') or {}).get('hot_path_over_budget_bytes', 0) or 0))}"
                if storage_tier
                else ""
            ),
        },
        "training_runtime": {
            "status": str(training_runtime.get("overall_status") or "missing"),
            "summary": (
                f"snapshot_ready={int(bool(training_runtime.get('snapshot_ready', False)))} "
                f"precompute_targets={len(training_runtime.get('precompute_targets') or [])}"
                if training_runtime
                else ""
            ),
        },
        "coverage_seeding": {
            "status": str(coverage_seed.get("overall_status") or ("ready" if coverage_seed else "missing")),
            "summary": (
                f"coverage_shortfall_bots={int(coverage_seed.get('coverage_shortfall_bots', 0) or 0)} "
                f"seed_queue={len(coverage_seed.get('seed_queue') or [])}"
                if coverage_seed
                else ""
            ),
        },
        "regime_engine": {
            "status": str(regime_control.get("overall_status") or "missing"),
            "summary": (
                f"{str(regime_control.get('regime_state') or '')} {str(regime_control.get('stance_label') or '')}".strip()
                if regime_control
                else ""
            ),
        },
        "lifecycle_teaching": {
            "status": str(supportability_control.get("overall_status") or "missing"),
            "summary": (
                f"supportability={float(((supportability_control.get('supportability') or {}).get('active_supportability_score', 0.0) or 0.0)):.3f} "
                f"students_without_teachers={int(((supportability_control.get('teacher_student') or {}).get('students_without_teachers', 0) or 0))}"
                if supportability_control
                else ""
            ),
        },
        "roster_expansion": {
            "status": str(roster_expansion.get("overall_status") or "missing"),
            "summary": (
                f"registered_slots={int(((roster_expansion.get('summary') or {}).get('registered_slot_count', 0) or 0))} "
                f"missing_slots={int(((roster_expansion.get('summary') or {}).get('missing_slot_count', 0) or 0))}"
                if roster_expansion
                else ""
            ),
        },
        "teacher_quality": {
            "status": str(teacher_quality.get("overall_status") or "missing"),
            "summary": (
                f"qualified_teachers={int(((teacher_quality.get('summary') or {}).get('qualified_teacher_count', 0) or 0))} "
                f"elite_teachers={int(((teacher_quality.get('summary') or {}).get('elite_teacher_count', 0) or 0))}"
                if teacher_quality
                else ""
            ),
        },
        "bot_quality_autopilot": {
            "status": str(bot_quality_autopilot.get("overall_status") or "missing"),
            "summary": (
                f"quality_queue={len(bot_quality_autopilot.get('quality_upgrade_queue') or [])}"
                if bot_quality_autopilot
                else ""
            ),
        },
        "execution_realism": {
            "status": str(paper_calibration.get("overall_status") or ("ready" if paper_calibration else "missing")),
            "summary": (
                f"mae_bps={float(((paper_calibration.get('metrics') or {}).get('mae_bps', 0.0) or 0.0)):.3f}"
                if paper_calibration
                else ""
            ),
        },
        "operator_cockpit": {
            "status": overall_status,
            "summary": "unified control plane",
        },
        "production_readiness": {
            "status": str(production_readiness.get("overall_status") or "missing"),
            "summary": (
                f"domains={int(production_readiness.get('domain_count', 0) or 0)} "
                f"blocked={int(production_readiness.get('blocked_domain_count', 0) or 0)} "
                f"live_allowed={int(bool(production_readiness.get('live_runtime_promotion_allowed', False)))}"
                if production_readiness
                else ""
            ),
        },
        "production_soak": {
            "status": str(production_soak.get("overall_status") or "missing"),
            "summary": (
                f"controls={int(production_soak.get('control_count', 0) or 0)} "
                f"blocked={int(production_soak.get('blocked_control_count', 0) or 0)}"
                if production_soak
                else ""
            ),
        },
        "master_infrastructure_supervisor": {
            "status": master_infra_lane_status,
            "summary": (
                f"posture={str(((master_infra.get('platform_posture') or {}).get('operating_posture') or 'unknown'))} "
                f"operational_cleanliness={float(((master_infra.get('maturity_scores') or {}).get('operational_cleanliness', 0.0) or 0.0)):.2f}"
                if master_infra
                else ""
            ),
        },
    }
    for key in (
        "control_plane",
        "provider_mesh",
        "execution_gateway",
        "retrain_pipeline",
        "event_history",
        "runtime_separation",
        "operator_cockpit_contract",
    ):
        if isinstance(service_upgrade_lanes.get(key), dict):
            row = service_upgrade_lanes.get(key) or {}
            lane_status = str(row.get("status") or "missing")
            if key == "runtime_separation" and runtime_separation_lane_status == "advisory":
                lane_status = "advisory"
            if key == "operator_cockpit_contract" and overall_status == "ready":
                lane_status = "ready"
            upgrade_lanes[key] = {
                "status": lane_status,
                "summary": str(row.get("summary") or ""),
            }
    long_run_lanes = {
        "live_runtime_separation": {
            "status": runtime_separation_lane_status,
            "summary": (
                f"contention_score={int(((live_runtime_separation.get('shared_host_pressure') or {}).get('contention_score', 0) or 0))}"
                if live_runtime_separation
                else ""
            ),
        },
        "rolling_restart": {
            "status": rolling_restart_lane_status,
            "summary": (
                f"restart_due={int(bool(rolling_restart.get('restart_due', False)))} "
                f"scope={str(rolling_restart.get('recommended_scope') or '')}"
                if rolling_restart
                else ""
            ),
        },
        "auth_lease": {
            "status": auth_lease_lane_status,
            "summary": (
                f"lease_state={str(auth_lease.get('lease_state') or '')} "
                f"expires_in_seconds={float(((auth_lease.get('lease_budget') or {}).get('expires_in_seconds', 0.0) or 0.0)):.1f}"
                if auth_lease
                else ""
            ),
        },
        "blackstart_recovery": {
            "status": str(blackstart_recovery.get("overall_status") or "missing"),
            "summary": f"stages={len(blackstart_recovery.get('stages') or [])}" if blackstart_recovery else "",
        },
        "sleeve_isolation": {
            "status": str(sleeve_isolation.get("overall_status") or "missing"),
            "summary": (
                f"isolated_lanes={int(((sleeve_isolation.get('sleeve_matrix') or {}).get('isolated_lane_count', 0) or 0))}"
                if sleeve_isolation
                else ""
            ),
        },
        "artifact_freshness_slo": {
            "status": artifact_freshness_lane_status,
            "summary": (
                f"stale_required={int(((artifact_freshness.get('sla_summary') or {}).get('stale_required', 0) or 0))}"
                if artifact_freshness
                else ""
            ),
        },
        "runtime_snapshot_cache": {
            "status": snapshot_cache_lane_status,
            "summary": (
                f"snapshot_ready={int(bool(((snapshot_cache.get('cache_health') or {}).get('snapshot_ready', False))))}"
                if snapshot_cache
                else ""
            ),
        },
        "remote_alert_control": {
            "status": str(remote_alert.get("overall_status") or "missing"),
            "summary": (
                f"unacked_critical={int(((remote_alert.get('critical_backlog') or {}).get('unacked_count', 0) or 0))}"
                if remote_alert
                else ""
            ),
        },
        "storage_quota_guard": {
            "status": str(storage_quota.get("overall_status") or "missing"),
            "summary": (
                f"hard_breaches={int(((storage_quota.get('quota_summary') or {}).get('hard_breaches', 0) or 0))}"
                if storage_quota
                else ""
            ),
        },
        "release_freeze": {
            "status": str(release_freeze.get("overall_status") or "missing"),
            "summary": (
                f"active={int(bool(((release_freeze.get('window') or {}).get('active', False))))}"
                if release_freeze
                else ""
            ),
        },
        "infrastructure_autofix": {
            "status": str(infrastructure_autofix.get("overall_status") or "missing"),
            "summary": (
                f"repair_plan={int(infrastructure_autofix.get('applyable_repair_count', 0) or 0)}"
                if infrastructure_autofix
                else ""
            ),
        },
        "roster_resilience": {
            "status": str(roster_resilience.get("overall_status") or "missing"),
            "summary": (
                f"bench_depth={int(((roster_resilience.get('bench') or {}).get('bench_depth', 0) or 0))}"
                if roster_resilience
                else ""
            ),
        },
        "chaos_drills": {
            "status": str(chaos_drills.get("overall_status") or "missing"),
            "summary": f"overdue={len(chaos_drills.get('overdue_drills') or [])}" if chaos_drills else "",
        },
    }

    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "schema_version": 5,
        "ok": overall_status == "ready",
        "overall_status": overall_status,
        "recommended_actions": recommended_actions,
        "adaptive_posture": adaptive_posture,
        "readiness_domains": readiness_domains,
        "upgrade_lanes": upgrade_lanes,
        "long_run_lanes": long_run_lanes,
        "managed_proof_debt": managed_proof_debt,
        "surfaces": {
            "runtime_gate_dashboard": {"status": str(runtime.get("overall", {}).get("status") or "")},
            "platform_control_plane": {"status": str((platform.get("institutional_readiness") or {}).get("overall_status") or "")},
            "provider_mesh": {"status": str(provider_mesh.get("overall_status") or "missing") if provider_mesh else "missing"},
            "service_control_plane": {"status": str(service_control_plane.get("overall_status") or "missing") if service_control_plane else "missing"},
            "training_report": {"status": _paper_soak_managed_status(str(training.get("overall_status") or ""), paper_soak_ready=paper_soak_ready)},
            "training_quality_control": {"status": _paper_soak_managed_status(str(training_quality.get("overall_status") or ""), paper_soak_ready=paper_soak_ready)},
            "ingestion_storage_control": {"status": str(storage.get("overall_status") or "")},
            "ingestion_storage_governor": {"status": str(governor.get("profile") or "missing") if governor else "missing"},
            "storage_tier_policy": {"status": storage_tier_lane_status if storage_tier else "missing"},
            "training_runtime_control": {"status": str(training_runtime.get("overall_status") or "missing") if training_runtime else "missing"},
            "external_backlog_drain": {"status": str(backlog_drain.get("overall_status") or "")},
            "external_backlog_retry_bot": {"status": backlog_retry_lane_status if backlog_retry_bot else "missing"},
            "ingestion_priority_queue": {"status": "ready" if queue else "missing"},
            "storage_resilience_control": {"status": str(resilience.get("overall_status") or "")},
            "storage_split_brain_reconciler": {"status": "needs_review" if int(((split_brain.get("summary") or {}).get("unresolved_conflicts", 0) or 0) > 0) else "ready"},
            "training_requalification_lane": {"status": "ready" if requalification else "missing"},
            "walk_forward_coverage_seed": {"status": _paper_soak_managed_status(str(coverage_seed.get("overall_status") or "missing") if coverage_seed else "missing", paper_soak_ready=paper_soak_ready)},
            "regime_control_plane": {"status": _paper_soak_managed_status(str(regime_control.get("overall_status") or "missing") if regime_control else "missing", paper_soak_ready=paper_soak_ready)},
            "supportability_control": {"status": str(supportability_control.get("overall_status") or "missing") if supportability_control else "missing"},
            "teacher_quality_guard": {"status": str(teacher_quality.get("overall_status") or "missing") if teacher_quality else "missing"},
            "bot_quality_autopilot": {"status": _paper_soak_managed_status(str(bot_quality_autopilot.get("overall_status") or "missing") if bot_quality_autopilot else "missing", paper_soak_ready=paper_soak_ready)},
            "infrastructure_autofix_bot": {"status": _paper_soak_managed_status(str(infrastructure_autofix.get("overall_status") or "missing") if infrastructure_autofix else "missing", paper_soak_ready=paper_soak_ready)},
            "live_runtime_separation_control": {"status": runtime_separation_lane_status if live_runtime_separation else "missing"},
            "rolling_restart_controller": {"status": rolling_restart_lane_status if rolling_restart else "missing"},
            "auth_lease_manager": {"status": auth_lease_lane_status if auth_lease else "missing"},
            "blackstart_recovery": {"status": str(blackstart_recovery.get("overall_status") or "missing") if blackstart_recovery else "missing"},
            "sleeve_isolation_guard": {"status": str(sleeve_isolation.get("overall_status") or "missing") if sleeve_isolation else "missing"},
            "artifact_freshness_slo": {"status": artifact_freshness_lane_status if artifact_freshness else "missing"},
            "runtime_snapshot_cache_control": {"status": snapshot_cache_lane_status if snapshot_cache else "missing"},
            "remote_alert_control": {"status": str(remote_alert.get("overall_status") or "missing") if remote_alert else "missing"},
            "storage_quota_guard": {"status": str(storage_quota.get("overall_status") or "missing") if storage_quota else "missing"},
            "release_freeze_guard": {"status": str(release_freeze.get("overall_status") or "missing") if release_freeze else "missing"},
            "roster_expansion_slots": {"status": str(roster_expansion.get("overall_status") or "missing") if roster_expansion else "missing"},
            "roster_resilience_planner": {"status": _paper_soak_managed_status(str(roster_resilience.get("overall_status") or "missing") if roster_resilience else "missing", paper_soak_ready=paper_soak_ready)},
            "chaos_drill_coordinator": {"status": _paper_soak_managed_status(str(chaos_drills.get("overall_status") or "missing") if chaos_drills else "missing", paper_soak_ready=paper_soak_ready)},
            "calibration_abstention_control": {"status": str(calibration.get("overall_status") or "")},
            "paper_execution_calibration": {"status": str(paper_calibration.get("overall_status") or "missing") if paper_calibration else "missing"},
            "daily_verify_auto_remediation_bot": {"status": _paper_soak_managed_status(str(remediation.get("overall_status") or ""), paper_soak_ready=paper_soak_ready)},
            "memory_efficiency_control": {"status": str(memory_efficiency.get("overall_status") or "missing") if memory_efficiency else "missing"},
            "production_readiness_control": {"status": str(production_readiness.get("overall_status") or "missing") if production_readiness else "missing"},
            "production_soak_enhancement": {"status": str(production_soak.get("overall_status") or "missing") if production_soak else "missing"},
            "system_self_model": {"status": _paper_soak_managed_status(str(system_self_model.get("overall_status") or "missing") if system_self_model else "missing", paper_soak_ready=paper_soak_ready)},
            "global_killswitch": {"status": "ready" if global_halt_clear else "blocked"},
            "master_infrastructure_supervisor": {"status": master_infra_lane_status if master_infra else "missing"},
            "process_lane_ownership": {
                "status": process_lane_status if master_infra else "missing"
            },
        },
        "maturity_scores": master_infra.get("maturity_scores") if isinstance(master_infra.get("maturity_scores"), dict) else {},
        "hardening_scorecard": master_infra.get("hardening_scorecard") if isinstance(master_infra.get("hardening_scorecard"), dict) else {},
    }
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Publish a single operator cockpit across runtime, storage, training, and remediation surfaces.")
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--out-file", default=str(DEFAULT_OUT_PATH))
    parser.add_argument("--markdown-out", default=str(DEFAULT_MARKDOWN_PATH))
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    payload = build_payload(Path(args.project_root).resolve())
    out_path = Path(args.out_file).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
    markdown_path = Path(args.markdown_out).expanduser()
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(_render_markdown(payload), encoding="utf-8")
    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        print(
            "operator_cockpit "
            f"overall_status={payload.get('overall_status', '')} "
            f"recommended_actions={len(payload.get('recommended_actions') or [])}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
