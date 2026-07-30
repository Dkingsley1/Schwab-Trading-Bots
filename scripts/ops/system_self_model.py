#!/usr/bin/env python3
from __future__ import annotations

import argparse
import errno
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_PATH = PROJECT_ROOT / "governance" / "health" / "system_self_model_latest.json"
DEFAULT_MARKDOWN_PATH = PROJECT_ROOT / "exports" / "reports" / "operator" / "system_self_model_latest.md"
DEFAULT_BRIEF_PATH = PROJECT_ROOT / "exports" / "reports" / "operator" / "system_self_brief_latest.md"
DEFAULT_DEPENDENCY_MEMORY_PATH = PROJECT_ROOT / "governance" / "health" / "system_dependency_memory_latest.json"
DEFAULT_FAILURE_MEMORY_PATH = PROJECT_ROOT / "governance" / "health" / "system_failure_memory_latest.json"
DEFAULT_REGISTRY_DIFF_PATH = PROJECT_ROOT / "governance" / "health" / "system_registry_diff_latest.json"
DEFAULT_UPGRADE_PLAN_PATH = PROJECT_ROOT / "governance" / "health" / "system_upgrade_optimizer_latest.json"
SELF_MODEL_VERSION = "system_self_model_v2"
FALLBACK_ROOT_NAMES = {"data", "exports", "governance", "logs"}


STATUS_ORDER = {
    "missing": 0,
    "ok": 1,
    "ready": 1,
    "idle": 1,
    "applied": 1,
    "baseline": 1,
    "steady_state": 1,
    "watch": 2,
    "advisory": 2,
    "thin": 2,
    "applied_with_followups": 2,
    "waiting_for_writer": 2,
    "handoff_requested": 2,
    "needs_work": 3,
    "degraded": 4,
    "stalled": 4,
    "apply_failed": 4,
    "blocked": 5,
    "critical": 6,
}
GUARDED_PAPER_MANAGED_SURFACES = {
    "whole_system_intelligence": "whole_system_self_model_feedback_debt_managed_by_runtime_dashboard",
    "system_signal_bus": "signal_bus_self_model_feedback_debt_managed_by_runtime_dashboard",
    "data_plane_recovery": "data_plane_coverage_stage_deferred_while_guarded_paper_soak_is_green",
    "master_infra": "self_auditing_infra_debt_deferred_while_hot_path_is_green",
    "training_quality": "training_quality_recovery_deferred_while_paper_execution_is_clean",
    "bot_quality": "bot_quality_retrain_queue_deferred_while_training_budget_is_closed",
}
GUARDED_PAPER_OPTIONAL_SURFACES = {
    "codex_operator_bridge": "optional_codex_handoff_surface_not_required_for_unattended_soak",
    "quant_model_control": "optional_quant_model_expansion_surface_not_required_for_unattended_soak",
    "capital_growth_intelligence": "optional_live_money_growth_surface_not_required_for_guarded_paper_soak",
    "capital_growth_awareness": "optional_live_money_growth_surface_not_required_for_guarded_paper_soak",
    "capital_rotation_control": "optional_live_money_rotation_surface_not_required_for_guarded_paper_soak",
}
GUARDED_PAPER_READY_STATUSES = {"ok", "ready", "armed", "guarded_ready"}
GUARDED_PAPER_SOFTENABLE_STATUSES = {"blocked", "degraded", "needs_work", "missing"}
GUARDED_PAPER_DATA_PLANE_STATES = {
    "managed_coverage_stage_deferred",
    "recovering_under_guard",
    "awaiting_coverage_cycles",
    "ready",
}
GUARDED_PAPER_MASTER_INFRA_DEBTS = {
    "governance_artifact_freshness",
    "operator_cockpit_readiness",
    "self_auditing_infra_bots",
}
GUARDED_PAPER_OPTIONAL_STALE_SURFACES = {
    "backpressure_super_drainer",
    "backpressure_super_drainer_memory",
    "mlx_runtime",
    "mlx_library",
    "mlx_intelligence_router",
    "library_utilization_router",
}


def _load_json(path: Path) -> dict[str, Any]:
    candidates = [path]
    try:
        rel_path = path.relative_to(PROJECT_ROOT)
    except Exception:
        rel_path = None
    if rel_path is not None and rel_path.parts and rel_path.parts[0] in {"data", "exports", "governance", "logs"}:
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


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")


def _local_fallback_path(path: Path) -> Path | None:
    try:
        rel_path = path.relative_to(PROJECT_ROOT)
    except Exception:
        return None
    if not rel_path.parts or rel_path.parts[0] not in FALLBACK_ROOT_NAMES:
        return None
    return PROJECT_ROOT / "local_fallback_storage" / rel_path


def _write_text_with_local_fallback(path: Path, text: str) -> dict[str, str]:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        return {"storage_mode": "primary", "path": str(path)}
    except OSError as exc:
        if exc.errno not in {errno.ENOSPC, getattr(errno, "EDQUOT", errno.ENOSPC)}:
            raise
        fallback = _local_fallback_path(path)
        if fallback is None:
            raise
        fallback.parent.mkdir(parents=True, exist_ok=True)
        fallback.write_text(text, encoding="utf-8")
        return {
            "storage_mode": "local_fallback",
            "path": str(fallback),
            "primary_path": str(path),
            "fallback_reason": errno.errorcode.get(exc.errno or 0, str(exc.errno)),
        }


def _json_sha256(payload: Any) -> str:
    try:
        encoded = json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8")
    except Exception:
        encoded = str(payload).encode("utf-8", errors="replace")
    return hashlib.sha256(encoded).hexdigest()


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


def _payload_timestamp(payload: dict[str, Any], path: Path, now: datetime) -> tuple[str, float | None]:
    for key in ("timestamp_utc", "updated_at_utc", "updated_at", "created_at", "generated_at_utc"):
        parsed = _parse_iso(payload.get(key))
        if parsed is not None:
            return parsed.isoformat(), round(max((now - parsed).total_seconds() / 60.0, 0.0), 3)
    try:
        mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
    except Exception:
        return "", None
    return mtime.isoformat(), round(max((now - mtime).total_seconds() / 60.0, 0.0), 3)


def _ordered_unique(items: list[Any]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for raw in items:
        item = str(raw or "").strip()
        if not item or item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def _safe_int(raw: Any, default: int = 0) -> int:
    try:
        return int(float(raw))
    except Exception:
        return default


def _safe_float(raw: Any, default: float = 0.0) -> float:
    try:
        return float(raw)
    except Exception:
        return default


def _status(payload: dict[str, Any], default: str = "missing") -> str:
    if not payload:
        return default
    explicit_raw = payload.get("overall_status")
    if explicit_raw is None:
        explicit_raw = payload.get("status")
    if isinstance(explicit_raw, str):
        explicit = explicit_raw.strip()
    elif explicit_raw is None:
        explicit = ""
    else:
        explicit = ""
    if explicit:
        if explicit.strip().lower().endswith("_ready"):
            return "ready"
        return explicit
    if "ok" in payload:
        return "ready" if bool(payload.get("ok", False)) else "blocked"
    return default


def _guarded_paper_management_context(health_root: Path) -> dict[str, Any]:
    dashboard = _load_json(health_root / "runtime_gate_dashboard_latest.json")
    overall = dashboard.get("overall") if isinstance(dashboard.get("overall"), dict) else dashboard
    if not isinstance(overall, dict):
        overall = {}
    context = overall.get("soak_management_context") if isinstance(overall.get("soak_management_context"), dict) else {}
    dashboard_status = str(overall.get("status") or dashboard.get("overall_status") or dashboard.get("status") or "").strip().lower()
    health_fast_status = str(context.get("health_fast_status") or "").strip().lower()
    paper_stage = str(context.get("paper_stage") or "").strip().lower()
    enabled = bool(
        overall.get("ok", False)
        and dashboard_status in {"ok", "ready"}
        and bool(context.get("soak_ready", False))
        and bool(context.get("paper_guard_clean", False))
        and paper_stage in {"armed", "ready", "paper_armed"}
        and health_fast_status in {"ready", "ok"}
    )
    return {
        "enabled": enabled,
        "managed_by": "runtime_gate_dashboard",
        "dashboard_status": dashboard_status,
        "soak_status": str(context.get("soak_status") or "").strip().lower(),
        "soak_grade": str(context.get("soak_grade") or ""),
        "paper_stage": paper_stage,
        "health_fast_status": health_fast_status,
        "raw_attention": _ordered_unique(overall.get("raw_attention") if isinstance(overall.get("raw_attention"), list) else []),
        "forensic_attention": _ordered_unique(
            overall.get("forensic_attention")
            if isinstance(overall.get("forensic_attention"), list)
            else overall.get("raw_attention")
            if isinstance(overall.get("raw_attention"), list)
            else []
        ),
    }


def _master_infra_guarded_paper_debt(payload: dict[str, Any]) -> bool:
    metrics = payload.get("metrics") if isinstance(payload.get("metrics"), dict) else {}
    if _safe_int(metrics.get("blocked_check_count"), 0) > 0 or _safe_int(metrics.get("hard_failed_attempt_count"), 0) > 0:
        return False
    checks = payload.get("checks") if isinstance(payload.get("checks"), list) else []
    non_ready = {
        str(row.get("name") or "").strip()
        for row in checks
        if isinstance(row, dict) and str(row.get("status") or "").strip().lower() not in GUARDED_PAPER_READY_STATUSES
    }
    return bool(non_ready) and non_ready <= GUARDED_PAPER_MASTER_INFRA_DEBTS


def _data_plane_guarded_paper_debt(payload: dict[str, Any]) -> bool:
    state = str(payload.get("runtime_clearance_state") or payload.get("recovery_state") or "").strip().lower()
    if state in GUARDED_PAPER_DATA_PLANE_STATES:
        return True
    blockers = payload.get("blockers") if isinstance(payload.get("blockers"), list) else []
    return not blockers and str(payload.get("recovery_state") or "").strip().lower() == "recovering_under_guard"


def _normalize_guarded_paper_surface(
    name: str,
    status: str,
    payload: dict[str, Any],
    context: dict[str, Any],
) -> tuple[str, dict[str, Any]]:
    raw_status = str(status or "missing").strip().lower()
    if name == "runtime_gate_dashboard" and raw_status == "ok":
        return "ready", {"raw_status": raw_status, "status_normalized_reason": "runtime_dashboard_ok_is_ready"}
    if not bool(context.get("enabled", False)) or raw_status not in GUARDED_PAPER_SOFTENABLE_STATUSES:
        return status, {}

    reason = ""
    if name in GUARDED_PAPER_OPTIONAL_SURFACES:
        reason = GUARDED_PAPER_OPTIONAL_SURFACES[name]
    elif name == "master_infra":
        if not _master_infra_guarded_paper_debt(payload):
            return status, {}
        reason = GUARDED_PAPER_MANAGED_SURFACES[name]
    elif name == "data_plane_recovery":
        if not _data_plane_guarded_paper_debt(payload):
            return status, {}
        reason = GUARDED_PAPER_MANAGED_SURFACES[name]
    elif name in GUARDED_PAPER_MANAGED_SURFACES:
        reason = GUARDED_PAPER_MANAGED_SURFACES[name]
    if not reason:
        return status, {}

    return (
        "advisory",
        {
            "raw_status": raw_status,
            "guarded_paper_advisory_only": True,
            "managed_by": str(context.get("managed_by") or "runtime_gate_dashboard"),
            "managed_control_state": reason,
            "soak_grade": str(context.get("soak_grade") or ""),
            "paper_stage": str(context.get("paper_stage") or ""),
            "health_fast_status": str(context.get("health_fast_status") or ""),
        },
    )


def _worst_status(statuses: list[str]) -> str:
    values = [str(status or "missing").strip() for status in statuses if str(status or "").strip()]
    if not values:
        return "missing"
    return max(values, key=lambda item: STATUS_ORDER.get(item, 3))


def _registry_rows(registry: dict[str, Any]) -> list[dict[str, Any]]:
    rows = registry.get("sub_bots") if isinstance(registry.get("sub_bots"), list) else []
    if not rows and isinstance(registry.get("bots"), list):
        rows = registry.get("bots") or []
    return [row for row in rows if isinstance(row, dict)]


def _registry_identity(registry: dict[str, Any]) -> dict[str, Any]:
    rows = _registry_rows(registry)
    summary = registry.get("summary") if isinstance(registry.get("summary"), dict) else {}
    total = len(rows) or _safe_int(summary.get("total_bots"), 0)
    active = sum(1 for row in rows if bool(row.get("active", False))) or _safe_int(summary.get("active_bots"), 0)
    data_collection = sum(1 for row in rows if bool(row.get("data_collection_active", False))) or _safe_int(
        summary.get("data_collection_active_bots"),
        0,
    )
    training_excluded = sum(1 for row in rows if bool(row.get("training_excluded", False))) or _safe_int(
        summary.get("training_excluded_bots"),
        0,
    )
    lifecycle_counts: dict[str, int] = {}
    sleeve_profiles: set[str] = set()
    capability_packs: set[str] = set()
    for row in rows:
        lifecycle = str(row.get("lifecycle_state") or "unknown").strip().lower()
        lifecycle_counts[lifecycle] = lifecycle_counts.get(lifecycle, 0) + 1
        sleeve = str(row.get("sleeve_profile") or row.get("slot_kind") or "").strip()
        if sleeve:
            sleeve_profiles.add(sleeve)
        pack = str(row.get("capability_pack_slug") or row.get("capability_pack_version") or "").strip()
        if pack:
            capability_packs.add(pack)
    return {
        "total_bots": total,
        "active_bots": active,
        "data_collection_active_bots": data_collection,
        "training_excluded_bots": training_excluded,
        "collection_only_ratio": round(data_collection / max(active, 1), 4),
        "sleeve_profile_count": len(sleeve_profiles) or _safe_int(summary.get("sleeve_profile_count"), 0),
        "capability_pack_count": len(capability_packs),
        "lifecycle_counts": lifecycle_counts,
        "summary_source": "registry_rows" if rows else "registry_summary",
    }


def _surface_matrix(health_root: Path, project_root: Path, *, now: datetime | None = None) -> dict[str, dict[str, Any]]:
    current = now or datetime.now(timezone.utc)
    guarded_paper_context = _guarded_paper_management_context(health_root)
    paths = {
        "operator_cockpit": health_root / "operator_cockpit_latest.json",
        "memory_efficiency": health_root / "memory_efficiency_control_latest.json",
        "runtime_throttle": health_root / "runtime_throttle_control_latest.json",
        "ingestion_storage": health_root / "ingestion_storage_control_latest.json",
        "backpressure_drainer_fleet": health_root / "backpressure_drainer_fleet_latest.json",
        "backpressure_super_drainer": health_root / "backpressure_super_drainer_latest.json",
        "backpressure_super_drainer_memory": health_root / "backpressure_super_drainer_memory_latest.json",
        "writer_cycle_coordinator": health_root / "writer_cycle_coordinator_latest.json",
        "writer_process_intelligence": health_root / "writer_process_intelligence_latest.json",
        "whole_system_intelligence": health_root / "whole_system_intelligence_latest.json",
        "system_signal_bus": health_root / "system_signal_bus_latest.json",
        "system_brain": health_root / "system_brain_latest.json",
        "system_process_contracts": health_root / "system_process_contracts_latest.json",
        "system_self_intelligence": health_root / "system_self_intelligence_latest.json",
        "codex_handoff": health_root / "codex_handoff_latest.json",
        "codex_operator_bridge": health_root / "codex_operator_bridge_latest.json",
        "storage_backpressure_autopilot": health_root / "storage_backpressure_autopilot_latest.json",
        "mlx_runtime": health_root / "mlx_runtime_audit_latest.json",
        "mlx_library": health_root / "mlx_library_upgrade_latest.json",
        "mlx_intelligence_router": health_root / "mlx_intelligence_router_latest.json",
        "library_utilization_router": health_root / "library_utilization_router_latest.json",
        "quant_model_control": health_root / "quant_model_control_latest.json",
        "global_halt": health_root / "global_killswitch_latest.json",
        "process_watchdog": health_root / "process_watchdog_latest.json",
        "auth_lease_manager": health_root / "auth_lease_manager_latest.json",
        "data_plane_recovery": health_root / "data_plane_recovery_controller_latest.json",
        "live_runtime_separation": health_root / "live_runtime_separation_control_latest.json",
        "use_mode_compliance": health_root / "use_mode_compliance_guard_latest.json",
        "master_infra": health_root / "master_infrastructure_supervisor_latest.json",
        "artifact_freshness": health_root / "artifact_freshness_slo_latest.json",
        "training_quality": health_root / "training_quality_control_latest.json",
        "bot_quality": health_root / "bot_quality_autopilot_latest.json",
        "capital_growth_intelligence": health_root / "capital_growth_intelligence_latest.json",
        "capital_growth_awareness": health_root / "capital_growth_awareness_bridge_latest.json",
        "capital_rotation_control": health_root / "capital_rotation_control_latest.json",
        "schwab_indicator_intelligence": health_root / "schwab_indicator_intelligence_latest.json",
        "system_expansion_execution": health_root / "system_expansion_execution_layer_latest.json",
        "provider_mesh": health_root / "provider_mesh_latest.json",
        "core_materialization": health_root / "core_bot_materialization_guard_latest.json",
        "runtime_gate_dashboard": health_root / "runtime_gate_dashboard_latest.json",
        "storage_resilience": health_root / "storage_resilience_control_latest.json",
        "incident_auto_halt": project_root / "governance" / "alerts" / "incident_auto_halt_latest.json",
    }
    matrix: dict[str, dict[str, Any]] = {}
    for name, path in paths.items():
        payload = _load_json(path)
        status = _status(payload)
        if name == "runtime_gate_dashboard" and isinstance(payload.get("overall"), dict):
            status = str((payload.get("overall") or {}).get("status") or status)
        if name == "incident_auto_halt" and payload and status == "missing":
            status = "ready"
        if name == "global_halt" and payload:
            status = "blocked" if bool(payload.get("halt", False)) else "ready"
        if name == "process_watchdog" and payload:
            alerts = payload.get("alerts") if isinstance(payload.get("alerts"), list) else []
            rows = payload.get("status") if isinstance(payload.get("status"), list) else []
            watched_rows = [row for row in rows if isinstance(row, dict)]
            any_down = any(not bool(row.get("process_live", row.get("running", 0))) for row in watched_rows)
            status = "degraded" if alerts or any_down else "ready"
        raw_status = status
        status, status_metadata = _normalize_guarded_paper_surface(name, status, payload, guarded_paper_context)
        if name == "runtime_gate_dashboard" and bool(guarded_paper_context.get("enabled", False)):
            status_metadata.update(
                {
                    "guarded_paper_context_enabled": True,
                    "soak_grade": str(guarded_paper_context.get("soak_grade") or ""),
                    "paper_stage": str(guarded_paper_context.get("paper_stage") or ""),
                    "health_fast_status": str(guarded_paper_context.get("health_fast_status") or ""),
                }
            )
        timestamp, age_minutes = _payload_timestamp(payload, path, current) if payload else ("", None)
        matrix[name] = {
            "status": status,
            "raw_status": str(status_metadata.get("raw_status") or raw_status),
            "path": str(path),
            "loaded": bool(payload),
            "timestamp_utc": timestamp,
            "age_minutes": age_minutes,
            "payload_sha256": _json_sha256(payload) if payload else "",
            "payload_hash_short": _json_sha256(payload)[:12] if payload else "",
            **{key: value for key, value in status_metadata.items() if key != "raw_status"},
        }
    return matrix


def _resource_awareness(memory: dict[str, Any], throttle: dict[str, Any], storage: dict[str, Any]) -> dict[str, Any]:
    cotenant = memory.get("cotenant_awareness") if isinstance(memory.get("cotenant_awareness"), dict) else {}
    memory_snapshot = memory.get("memory_snapshot") if isinstance(memory.get("memory_snapshot"), dict) else {}
    storage_snapshot = memory.get("storage_snapshot") if isinstance(memory.get("storage_snapshot"), dict) else {}
    throttle_memory = str(throttle.get("memory_pressure_level") or "normal")
    throttle_profile = str(throttle.get("throttle_profile") or "")
    storage_status = _status(storage)
    status = "ready"
    if _status(memory) == "blocked" or throttle_memory == "high" or storage_status == "blocked":
        status = "blocked"
    elif _status(memory) in {"needs_work", "degraded"} or throttle_memory == "elevated" or storage_status in {"needs_work", "degraded"}:
        status = "degraded"
    elif str(cotenant.get("mode") or "") in {"managed_cotenant", "guarded_cotenant"} or throttle_profile in {"soft_cap", "sustain"}:
        status = "advisory"
    return {
        "status": status,
        "memory_guard_status": _status(memory),
        "runtime_throttle_status": _status(throttle),
        "storage_status": storage_status,
        "memory_pressure_state": str(memory_snapshot.get("memory_pressure_state") or ""),
        "memory_pressure_kind": str(memory_snapshot.get("memory_pressure_kind") or ""),
        "swap_used_gb": _safe_float(memory_snapshot.get("swap_used_gb"), 0.0),
        "storage_pressure_index": _safe_float(storage_snapshot.get("pressure_index"), _safe_float(storage.get("pressure_index"), 0.0)),
        "recommended_profile": str(memory.get("recommended_profile") or ""),
        "cotenant_awareness": cotenant,
        "runtime_throttle_profile": throttle_profile,
    }


def _host_pressure_intelligence(
    memory: dict[str, Any],
    throttle: dict[str, Any],
    mlx_router: dict[str, Any],
    library_router: dict[str, Any],
) -> dict[str, Any]:
    memory_snapshot = memory.get("memory_snapshot") if isinstance(memory.get("memory_snapshot"), dict) else {}
    cpu_snapshot = memory.get("cpu_snapshot") if isinstance(memory.get("cpu_snapshot"), dict) else {}
    cotenant = memory.get("cotenant_awareness") if isinstance(memory.get("cotenant_awareness"), dict) else {}
    mlx_caps = mlx_router.get("runtime_caps") if isinstance(mlx_router.get("runtime_caps"), dict) else {}
    library_caps = library_router.get("runtime_caps") if isinstance(library_router.get("runtime_caps"), dict) else {}

    memory_level = str(throttle.get("memory_pressure_level") or mlx_caps.get("memory_pressure_level") or "").strip().lower()
    if not memory_level:
        memory_state = str(memory_snapshot.get("memory_pressure_state") or "").strip().lower()
        memory_level = "high" if memory_state in {"red", "critical"} else "elevated" if memory_state in {"yellow", "orange"} else "normal"

    host_saturation_score = _safe_float(
        throttle.get("host_saturation_score"),
        _safe_float(mlx_caps.get("host_saturation_score"), _safe_float(cpu_snapshot.get("host_saturation_score"), 0.0)),
    )
    cpu_level = str(throttle.get("cpu_pressure_level") or mlx_caps.get("cpu_pressure_level") or cpu_snapshot.get("cpu_pressure_level") or "").strip().lower()
    if not cpu_level:
        cpu_level = "high" if host_saturation_score >= 85.0 else "elevated" if host_saturation_score >= 65.0 else "watch" if host_saturation_score >= 45.0 else "normal"

    throttle_profile = str(throttle.get("throttle_profile") or mlx_caps.get("throttle_profile") or library_caps.get("throttle_profile") or "observe").strip().lower()
    open_apps = cotenant.get("open_apps") if isinstance(cotenant.get("open_apps"), list) else []
    open_app_count = _safe_int(cotenant.get("open_app_count"), len(open_apps))
    cotenant_active = bool(cotenant.get("active", False) or cotenant.get("mode") in {"managed_cotenant", "guarded_cotenant", "pressure_aware_cotenant"})

    posture = "max_throughput"
    status = "ready"
    if memory_level == "high" or cpu_level == "high" or host_saturation_score >= 85.0:
        status = "blocked"
        posture = "protect_live"
    elif memory_level == "elevated" or cpu_level == "elevated" or host_saturation_score >= 65.0 or throttle_profile == "protect_live":
        status = "degraded"
        posture = "protect_live" if throttle_profile == "protect_live" else "sustain"
    elif cpu_level == "watch" or host_saturation_score >= 45.0 or throttle_profile in {"soft_cap", "sustain"} or cotenant_active:
        status = "advisory"
        posture = "foreground_safe"

    return {
        "status": status,
        "memory_pressure_level": memory_level,
        "memory_pressure_state": str(memory_snapshot.get("memory_pressure_state") or ""),
        "memory_free_pct": _safe_float(memory_snapshot.get("memory_free_pct"), 0.0),
        "swap_used_gb": _safe_float(memory_snapshot.get("swap_used_gb"), 0.0),
        "compressed_store_gb": _safe_float(memory_snapshot.get("compressed_store_gb"), _safe_float(memory_snapshot.get("compressor_gb"), 0.0)),
        "cpu_pressure_level": cpu_level,
        "host_saturation_score": round(float(host_saturation_score), 3),
        "throttle_profile": throttle_profile,
        "recommended_intelligence_posture": posture,
        "mlx_runtime_profile": str(mlx_caps.get("profile") or ""),
        "mlx_max_concurrent_jobs": _safe_int(mlx_caps.get("max_concurrent_mlx_jobs"), 0),
        "mlx_heavy_vlm_enabled": bool(mlx_caps.get("heavy_vlm_enabled", False)),
        "library_runtime_profile": str(library_caps.get("profile") or ""),
        "library_max_report_render_jobs": _safe_int(library_caps.get("max_report_render_jobs"), 0),
        "cotenant_active": cotenant_active,
        "open_app_count": open_app_count,
        "open_apps": [str(app) for app in open_apps[:12]],
        "co_running_level": str(cotenant.get("co_running_level") or ""),
        "live_data_priority": "protect_live_collection_and_paper_trade_before_heavy_training_or_reporting",
        "control_contract": "cpu_memory_pressure_states_feed_intelligence_routing_mlx_caps_drainers_and_training_cadence",
    }


def _mlx_intelligence_awareness(router: dict[str, Any]) -> dict[str, Any]:
    coverage = router.get("library_coverage") if isinstance(router.get("library_coverage"), dict) else {}
    route_coverage = router.get("route_coverage") if isinstance(router.get("route_coverage"), dict) else {}
    caps = router.get("runtime_caps") if isinstance(router.get("runtime_caps"), dict) else {}
    matrix = router.get("library_utilization_matrix") if isinstance(router.get("library_utilization_matrix"), dict) else {}
    status = _status(router)
    return {
        "status": status,
        "library_coverage_ratio": _safe_float(coverage.get("coverage_ratio"), 0.0),
        "route_coverage_ratio": _safe_float(route_coverage.get("route_coverage_ratio"), 0.0),
        "mapped_library_ratio": _safe_float(matrix.get("mapped_library_ratio"), 0.0),
        "missing_package_count": _safe_int(coverage.get("missing_count"), 0),
        "blocked_lane_count": _safe_int(route_coverage.get("blocked_lane_count"), 0),
        "runtime_profile": str(caps.get("profile") or ""),
        "max_concurrent_mlx_jobs": _safe_int(caps.get("max_concurrent_mlx_jobs"), 0),
        "compile_mode": str(caps.get("compile_mode") or ""),
        "heavy_vlm_enabled": bool(caps.get("heavy_vlm_enabled", False)),
        "cpu_pressure_level": str(caps.get("cpu_pressure_level") or ""),
        "memory_pressure_level": str(caps.get("memory_pressure_level") or ""),
        "host_saturation_score": _safe_float(caps.get("host_saturation_score"), 0.0),
        "host_pressure_state": str(caps.get("host_pressure_state") or ""),
        "utilization_contract": str(((router.get("control_contract") or {}).get("safe_utilization_goal")) or ""),
    }


def _library_utilization_awareness(router: dict[str, Any]) -> dict[str, Any]:
    coverage = router.get("coverage") if isinstance(router.get("coverage"), dict) else {}
    caps = router.get("runtime_caps") if isinstance(router.get("runtime_caps"), dict) else {}
    matrix = router.get("library_utilization_matrix") if isinstance(router.get("library_utilization_matrix"), dict) else {}
    contract = router.get("control_contract") if isinstance(router.get("control_contract"), dict) else {}
    return {
        "status": _status(router),
        "managed_non_mlx_package_count": _safe_int(coverage.get("managed_non_mlx_package_count"), 0),
        "locked_non_mlx_package_count": _safe_int(coverage.get("locked_non_mlx_package_count"), 0),
        "mapped_package_ratio": _safe_float(matrix.get("mapped_package_ratio"), _safe_float(coverage.get("coverage_ratio"), 0.0)),
        "locked_runtime_ok_ratio": _safe_float(coverage.get("locked_runtime_ok_ratio"), 0.0),
        "missing_runtime_count": _safe_int(coverage.get("missing_runtime_count"), 0),
        "version_mismatch_count": _safe_int(coverage.get("version_mismatch_count"), 0),
        "runtime_profile": str(caps.get("profile") or ""),
        "default_ml_backend": str(contract.get("default_ml_backend") or ""),
        "portable_ml_policy": str(contract.get("portable_ml_policy") or ""),
        "utilization_contract": str(contract.get("safe_utilization_goal") or ""),
    }


def _drainer_intelligence_awareness(
    fleet: dict[str, Any],
    super_drainer: dict[str, Any],
    coordinator: dict[str, Any],
    storage_autopilot: dict[str, Any],
) -> dict[str, Any]:
    fleet_active = fleet.get("active_drainer") if isinstance(fleet.get("active_drainer"), dict) else {}
    super_summary = super_drainer.get("summary") if isinstance(super_drainer.get("summary"), dict) else {}
    super_guardrails = super_drainer.get("guardrails") if isinstance(super_drainer.get("guardrails"), dict) else {}
    super_settings = super_drainer.get("settings") if isinstance(super_drainer.get("settings"), dict) else {}
    intelligence_layer = (
        super_drainer.get("drainer_intelligence_layer")
        if isinstance(super_drainer.get("drainer_intelligence_layer"), dict)
        else {}
    )
    intelligence_decision = (
        intelligence_layer.get("decision_packet")
        if isinstance(intelligence_layer.get("decision_packet"), dict)
        else {}
    )
    coordinator_summary = coordinator.get("summary") if isinstance(coordinator.get("summary"), dict) else {}
    coordinator_writer = coordinator.get("writer_state_after_wait") if isinstance(coordinator.get("writer_state_after_wait"), dict) else {}
    autopilot_metrics = storage_autopilot.get("metrics") if isinstance(storage_autopilot.get("metrics"), dict) else {}

    single_writer_guard = bool(super_guardrails.get("single_writer_only", False)) and not bool(
        super_guardrails.get("starts_parallel_sql_writers", True)
    )
    super_status = _status(super_drainer)
    intelligence_status = _status(intelligence_layer)
    fleet_status = _status(fleet)
    coordinator_status = _status(coordinator)
    active_drainer = str(super_drainer.get("active_drainer") or fleet_active.get("name") or "")
    target_met = bool(super_drainer.get("target_met_final", False) or super_drainer.get("target_met_initially", False))
    waves_run = _safe_int(super_summary.get("waves_run"), 0)
    final_pending = _safe_int(super_summary.get("final_pending_lines"), 0)
    progress_waves = _safe_int(super_summary.get("progress_waves"), 0)
    writer_active = bool(coordinator_writer.get("active", False) or coordinator_summary.get("writer_active_after_wait", False))

    status = "ready"
    if super_drainer and not single_writer_guard:
        status = "blocked"
    elif intelligence_layer and intelligence_status in {"blocked", "critical", "degraded"} and not target_met:
        status = "degraded"
    elif super_status in {"apply_failed", "stalled", "blocked", "critical"}:
        status = "degraded"
    elif coordinator_status in {"apply_failed", "timed_out_waiting_for_writer", "blocked", "critical"}:
        status = "degraded"
    elif fleet_status == "blocked" and not target_met:
        status = "degraded"
    elif writer_active and not bool(coordinator_summary.get("writer_progress_observed", False)):
        status = "advisory"
    elif target_met or final_pending <= _safe_int(super_settings.get("target_pending_lines"), 5000):
        status = "ready"
    elif waves_run and progress_waves:
        status = "advisory"

    return {
        "status": status,
        "fleet_status": fleet_status,
        "super_drainer_status": super_status,
        "intelligence_layer_status": intelligence_status,
        "writer_cycle_status": coordinator_status,
        "storage_autopilot_status": _status(storage_autopilot),
        "active_drainer": active_drainer,
        "ready_drainer_count": _safe_int(fleet.get("ready_drainer_count"), 0),
        "ready_drainer_names": list(super_drainer.get("ready_drainer_names") or []),
        "target_met": target_met,
        "target_pending_lines": _safe_int(super_settings.get("target_pending_lines"), 5000),
        "final_pending_lines": final_pending,
        "planned_wave_count": _safe_int(super_settings.get("planned_wave_count"), 0),
        "waves_run": waves_run,
        "progress_waves": progress_waves,
        "stop_reason": str(super_drainer.get("stop_reason") or super_summary.get("stop_reason") or ""),
        "intelligence_action": str(intelligence_decision.get("action") or ""),
        "intelligence_confidence": _safe_float(intelligence_decision.get("confidence"), 0.0),
        "intelligence_risk_flags": list(intelligence_decision.get("risk_flags") or []),
        "intelligence_next_ready_drainer": str(intelligence_decision.get("next_ready_drainer") or ""),
        "single_writer_guard": single_writer_guard,
        "writer_active": writer_active,
        "writer_progress_observed": bool(coordinator_summary.get("writer_progress_observed", False)),
        "backpressure_actionable": bool(autopilot_metrics.get("backpressure_actionable", False)),
        "assigned_infrabots": list(super_drainer.get("assigned_infrabots") or []),
        "grandmaster_context_packet": super_drainer.get("grandmaster_context_packet") if isinstance(super_drainer.get("grandmaster_context_packet"), dict) else {},
        "control_contract": "drainer_stack_is_part_of_self_model_resource_awareness_and_uses_single_writer_wave_coordination",
    }


def _writer_process_awareness(
    writer_intelligence: dict[str, Any],
    coordinator: dict[str, Any],
    process_watchdog: dict[str, Any],
    process_fanout: dict[str, Any],
) -> dict[str, Any]:
    decision = writer_intelligence.get("decision_packet") if isinstance(writer_intelligence.get("decision_packet"), dict) else {}
    writer_health = writer_intelligence.get("writer_health") if isinstance(writer_intelligence.get("writer_health"), dict) else {}
    topology = writer_intelligence.get("process_topology") if isinstance(writer_intelligence.get("process_topology"), dict) else {}
    safety = writer_intelligence.get("safety_envelope") if isinstance(writer_intelligence.get("safety_envelope"), dict) else {}
    coordinator_summary = coordinator.get("summary") if isinstance(coordinator.get("summary"), dict) else {}
    status = _status(writer_intelligence)
    if status == "missing":
        status = "degraded" if coordinator or process_watchdog else "missing"
    if bool(topology.get("duplicate_sql_writer_processes", False)):
        status = "blocked"
    elif str(writer_health.get("state") or "") == "stalled":
        status = "degraded"
    elif str(writer_health.get("state") or "") == "stale_progress":
        status = "advisory"

    return {
        "status": status,
        "intelligence_layer_status": _status(writer_intelligence),
        "writer_cycle_status": _status(coordinator),
        "process_watchdog_status": _status(process_watchdog),
        "process_fanout_status": _status(process_fanout),
        "action": str(decision.get("action") or ""),
        "confidence": _safe_float(decision.get("confidence"), 0.0),
        "writer_state": str(writer_health.get("state") or ""),
        "writer_active": bool(writer_health.get("active", coordinator_summary.get("writer_active_after_wait", False))),
        "writer_progress_age_minutes": _safe_float(writer_health.get("progress_age_minutes"), 0.0),
        "expanded_writer_lane_count": _safe_int(decision.get("expanded_writer_lane_count"), 0),
        "hot_lane_count": _safe_int(decision.get("hot_lane_count"), 0),
        "warm_lane_count": _safe_int(decision.get("warm_lane_count"), 0),
        "cold_lane_count": _safe_int(decision.get("cold_lane_count"), 0),
        "risk_flags": list(decision.get("risk_flags") or []),
        "single_writer_guard": bool(safety.get("single_writer_only", False))
        and not bool(safety.get("starts_parallel_sql_writers", True)),
        "max_parallel_sql_writers": _safe_int(safety.get("max_parallel_sql_writers"), 1),
        "process_trim_before_expansion": bool(safety.get("process_trim_before_expansion", False)),
        "writer_recovery_required": bool(safety.get("writer_recovery_required", False)),
        "playbook": writer_intelligence.get("process_playbook") if isinstance(writer_intelligence.get("process_playbook"), list) else [],
        "control_contract": "writer_process_intelligence_expands_shard_lanes_and_process_diagnostics_while_preserving_one_sql_writer",
    }


def _whole_system_intelligence_awareness(whole_system: dict[str, Any]) -> dict[str, Any]:
    signal_bus = whole_system.get("system_signal_bus") if isinstance(whole_system.get("system_signal_bus"), dict) else {}
    system_brain = whole_system.get("system_brain") if isinstance(whole_system.get("system_brain"), dict) else {}
    process_contracts = (
        whole_system.get("system_process_contracts")
        if isinstance(whole_system.get("system_process_contracts"), dict)
        else {}
    )
    self_intelligence = (
        whole_system.get("system_self_intelligence")
        if isinstance(whole_system.get("system_self_intelligence"), dict)
        else {}
    )
    codex_handoff = whole_system.get("codex_handoff") if isinstance(whole_system.get("codex_handoff"), dict) else {}
    signal_summary = signal_bus.get("summary") if isinstance(signal_bus.get("summary"), dict) else {}
    decision = system_brain.get("decision_packet") if isinstance(system_brain.get("decision_packet"), dict) else {}
    attention = codex_handoff.get("attention_packet") if isinstance(codex_handoff.get("attention_packet"), dict) else {}
    self_reflex = self_intelligence.get("reflex") if isinstance(self_intelligence.get("reflex"), dict) else {}
    self_uncertainty = self_intelligence.get("uncertainty") if isinstance(self_intelligence.get("uncertainty"), dict) else {}
    self_causal = self_intelligence.get("causal_diagnosis") if isinstance(self_intelligence.get("causal_diagnosis"), dict) else {}
    self_effect = self_intelligence.get("action_effectiveness") if isinstance(self_intelligence.get("action_effectiveness"), dict) else {}
    self_routing = self_intelligence.get("integration_routing") if isinstance(self_intelligence.get("integration_routing"), dict) else {}
    status = _status(whole_system)
    if status == "missing":
        status = "missing"
    elif str(system_brain.get("overall_status") or "") == "blocked":
        status = "blocked"
    elif str(system_brain.get("overall_status") or "") == "degraded":
        status = "degraded"
    elif str(system_brain.get("overall_status") or "") == "advisory":
        status = "advisory"
    return {
        "status": status,
        "signal_bus_status": _status(signal_bus),
        "system_brain_status": _status(system_brain),
        "process_contract_status": _status(process_contracts),
        "self_intelligence_status": _status(self_intelligence),
        "codex_handoff_status": _status(codex_handoff),
        "signal_count": _safe_int(signal_summary.get("signal_count"), 0),
        "loaded_signal_count": _safe_int(signal_summary.get("loaded_signal_count"), 0),
        "top_risk": str(decision.get("top_risk") or signal_summary.get("top_risk") or ""),
        "action": str(decision.get("action") or ""),
        "operating_mode": str(decision.get("operating_mode") or ""),
        "confidence": _safe_float(decision.get("confidence"), 0.0),
        "safe_next_command": decision.get("safe_next_command") if isinstance(decision.get("safe_next_command"), list) else [],
        "do_not_do": list(decision.get("do_not_do") or []),
        "risk_flags": list(decision.get("risk_flags") or []),
        "contract_count": _safe_int(process_contracts.get("contract_count"), 0),
        "blocked_contract_count": _safe_int(process_contracts.get("blocked_contract_count"), 0),
        "self_reflex_action": str(self_reflex.get("action") or ""),
        "self_uncertainty_level": str(self_uncertainty.get("level") or ""),
        "self_uncertainty_score": _safe_int(self_uncertainty.get("score"), 0),
        "self_causal_root": str(self_causal.get("primary_root_cause") or ""),
        "self_causal_confidence": _safe_float(self_causal.get("confidence"), 0.0),
        "self_action_effect_verdict": str(self_effect.get("verdict") or ""),
        "self_integration_route": str(self_routing.get("route_mode") or ""),
        "self_integration_owner": str(self_routing.get("primary_owner") or ""),
        "codex_needs": list(attention.get("needs_codex") or []),
        "codex_handoff_channel": "artifact_handoff",
        "proactive_codex_delivery": bool(
            ((codex_handoff.get("communication_contract") or {}).get("proactive_delivery_to_codex"))
        )
        if isinstance(codex_handoff.get("communication_contract"), dict)
        else False,
        "control_contract": "whole_system_intelligence_normalizes_signals_selects_next_safe_infrastructure_action_enforces_process_contracts_reads_self_causal_effect_routing_and_writes_codex_handoff",
    }


def _system_self_intelligence_awareness(self_intelligence: dict[str, Any]) -> dict[str, Any]:
    trend = self_intelligence.get("trend") if isinstance(self_intelligence.get("trend"), dict) else {}
    uncertainty = self_intelligence.get("uncertainty") if isinstance(self_intelligence.get("uncertainty"), dict) else {}
    memory = self_intelligence.get("learning_memory") if isinstance(self_intelligence.get("learning_memory"), dict) else {}
    action_effect = self_intelligence.get("action_effectiveness") if isinstance(self_intelligence.get("action_effectiveness"), dict) else {}
    causal = self_intelligence.get("causal_diagnosis") if isinstance(self_intelligence.get("causal_diagnosis"), dict) else {}
    routing = self_intelligence.get("integration_routing") if isinstance(self_intelligence.get("integration_routing"), dict) else {}
    reflex = self_intelligence.get("reflex") if isinstance(self_intelligence.get("reflex"), dict) else {}
    return {
        "status": _status(self_intelligence),
        "trajectory": str(trend.get("trajectory") or ""),
        "pending_lines_delta": _safe_int(trend.get("pending_lines_delta"), 0),
        "pressure_index_delta": _safe_float(trend.get("pressure_index_delta"), 0.0),
        "uncertainty_level": str(uncertainty.get("level") or ""),
        "uncertainty_score": _safe_int(uncertainty.get("score"), 0),
        "missing_signal_count": len(list(uncertainty.get("missing_signals") or [])),
        "stale_signal_count": len(list(uncertainty.get("stale_signals") or [])),
        "conflict_count": len(list(uncertainty.get("conflicting_signals") or [])),
        "contract_violation_count": len(list(uncertainty.get("contract_violations") or [])),
        "same_action_repeat_count": _safe_int(memory.get("same_action_repeat_count"), 0),
        "action_effect_verdict": str(action_effect.get("verdict") or ""),
        "same_action_run_length": _safe_int(action_effect.get("same_action_run_length"), 0),
        "causal_root": str(causal.get("primary_root_cause") or ""),
        "causal_confidence": _safe_float(causal.get("confidence"), 0.0),
        "integration_route_mode": str(routing.get("route_mode") or ""),
        "primary_owner": str(routing.get("primary_owner") or ""),
        "capability_gap_count": len(list(self_intelligence.get("capability_gaps") or [])),
        "reflex_action": str(reflex.get("action") or ""),
        "reflex_blocks_brain_action": bool(reflex.get("blocks_brain_action_until_refreshed", False)),
        "self_questions": list(self_intelligence.get("self_questions") or []),
        "control_contract": "system_self_intelligence_compares_prior_runs_tracks_uncertainty_scores_action_effects_diagnoses_causes_routes_consumers_and_can_request_pre_action_refreshes_before_the_brain_acts",
    }


def _codex_operator_bridge_awareness(bridge: dict[str, Any]) -> dict[str, Any]:
    attention = bridge.get("attention_packet") if isinstance(bridge.get("attention_packet"), dict) else {}
    sections = bridge.get("sections") if isinstance(bridge.get("sections"), dict) else {}
    paper = sections.get("paper_trading") if isinstance(sections.get("paper_trading"), dict) else {}
    training = sections.get("training") if isinstance(sections.get("training"), dict) else {}
    writer = sections.get("writer") if isinstance(sections.get("writer"), dict) else {}
    memory = sections.get("memory") if isinstance(sections.get("memory"), dict) else {}
    livefeed = sections.get("livefeed") if isinstance(sections.get("livefeed"), dict) else {}
    day = paper.get("day") if isinstance(paper.get("day"), dict) else {}
    return {
        "status": _status(bridge),
        "needs_codex": [str(item) for item in list(attention.get("needs_codex") or [])],
        "needs_codex_count": len(list(attention.get("needs_codex") or [])),
        "active_blocker_count": len(list(attention.get("active_blockers") or [])),
        "safe_next_command_count": len(list(attention.get("safe_next_commands") or [])),
        "paper_day_utc": str(day.get("day_utc") or ""),
        "paper_day_executions": _safe_int(day.get("executions"), 0),
        "paper_day_net_pnl": _safe_float(day.get("ending_net_pnl_total"), 0.0),
        "paper_day_change": _safe_float(day.get("change_vs_previous_day"), 0.0),
        "training_launch_allowed": bool(training.get("launch_allowed", False)),
        "training_recommended_batch_size": _safe_int(training.get("recommended_batch_size"), 0),
        "training_launch_blockers": [str(item) for item in list(training.get("launch_blockers") or [])],
        "writer_active": bool(writer.get("active", False)),
        "writer_completed_shards": _safe_int(writer.get("completed_shard_count"), 0),
        "writer_planned_shards": _safe_int(writer.get("planned_shard_count"), 0),
        "memory_classification": str(memory.get("classification") or ""),
        "memory_safe_for_training": bool(memory.get("safe_for_training", False)),
        "livefeed_alive": bool(livefeed.get("alive", False)),
        "communication_contract": attention.get("communication_contract")
        if isinstance(attention.get("communication_contract"), dict)
        else {
            "delivery_channel": "artifact_handoff",
            "proactive_delivery_to_codex": False,
        },
        "control_contract": "codex_operator_bridge_packages_trade_state_training_gates_writer_memory_livefeed_notifications_safe_commands_and_guardrails_for_fast_codex_handoffs",
    }


def _bot_awareness(identity: dict[str, Any], core_materialization: dict[str, Any]) -> dict[str, Any]:
    materialization_summary = core_materialization.get("summary") if isinstance(core_materialization.get("summary"), dict) else {}
    missing_modules = _safe_int(materialization_summary.get("missing_core_module_count"), 0)
    duplicate_versions = _safe_int(materialization_summary.get("duplicate_core_version_count"), 0)
    status = "ready"
    if missing_modules or duplicate_versions:
        status = "degraded"
    if _safe_int(identity.get("active_bots"), 0) <= 0:
        status = "blocked"
    return {
        "status": status,
        **identity,
        "missing_core_module_count": missing_modules,
        "duplicate_core_version_count": duplicate_versions,
        "materialization_status": _status(core_materialization),
    }


def _failure_memory(global_halt: dict[str, Any], incident: dict[str, Any], cockpit: dict[str, Any]) -> dict[str, Any]:
    adaptive = cockpit.get("adaptive_posture") if isinstance(cockpit.get("adaptive_posture"), dict) else {}
    hard_blockers = adaptive.get("hard_blockers") if isinstance(adaptive.get("hard_blockers"), list) else []
    global_halt_active = bool(global_halt.get("halt", False))
    incident_status = _status(incident, "ready" if incident else "missing")
    status = "ready"
    if global_halt_active:
        status = "blocked"
    elif hard_blockers:
        status = "degraded"
    elif incident_status == "missing":
        status = "advisory"
    return {
        "status": status,
        "global_halt_active": global_halt_active,
        "global_halt_action": str(global_halt.get("action") or "none"),
        "global_halt_reasons": global_halt.get("reasons") if isinstance(global_halt.get("reasons"), list) else [],
        "hard_blockers": hard_blockers,
        "incident_status": incident_status,
        "latest_incident_event": str(incident.get("event") or incident.get("status") or ""),
        "memory_contract": "capture_halts_tripwires_backpressure_feed_cuts_and_guard_blocks_as_replayable_causes",
    }


def _halt_recovery_intelligence(
    global_halt: dict[str, Any],
    process_watchdog: dict[str, Any],
    auth_lease: dict[str, Any],
    data_plane: dict[str, Any],
    live_runtime: dict[str, Any],
    storage: dict[str, Any],
) -> dict[str, Any]:
    clear_blockers = global_halt.get("clear_blockers") if isinstance(global_halt.get("clear_blockers"), list) else []
    degraded_clear_blockers = global_halt.get("degraded_clear_blockers") if isinstance(global_halt.get("degraded_clear_blockers"), list) else []
    halt_payload = global_halt.get("global_halt_payload") if isinstance(global_halt.get("global_halt_payload"), dict) else {}
    halt_details = halt_payload.get("details") if isinstance(halt_payload.get("details"), dict) else {}
    halt_active = bool(global_halt.get("halt", False))
    clear_ready = bool(global_halt.get("clear_ready", False) or (halt_active and not clear_blockers))
    halt_reason = str(halt_payload.get("reason") or ",".join(str(x) for x in (global_halt.get("reasons") or []) if str(x).strip()) or global_halt.get("action") or "none")

    auth_status = _status(auth_lease, "missing")
    lease_state = str(auth_lease.get("lease_state") or "")
    lease_budget = auth_lease.get("lease_budget") if isinstance(auth_lease.get("lease_budget"), dict) else {}
    expires_in = _safe_float(lease_budget.get("expires_in_seconds"), 0.0)
    min_lease = _safe_float(lease_budget.get("min_lease_seconds"), 1200.0)
    broker_state = auth_lease.get("broker_state") if isinstance(auth_lease.get("broker_state"), dict) else {}
    auth_reason = str(broker_state.get("auth_reason") or "")
    fallback_ladder = auth_lease.get("fallback_ladder") if isinstance(auth_lease.get("fallback_ladder"), list) else []
    auth_ok = bool(broker_state.get("auth_ok", False) or broker_state.get("broker_ready", False))
    broker_operable = bool(broker_state.get("broker_operable", False))
    auth_refresh_needed = bool(
        auth_status in {"blocked", "critical", "degraded", "needs_work"}
        or lease_state in {"warning", "critical", "expired"}
        or (expires_in and min_lease and expires_in < min_lease)
        or "softguard" in halt_reason
        or "account" in halt_reason
    )
    operator_auth_required = bool(
        auth_refresh_needed
        and (
            lease_state in {"critical", "expired"}
            or auth_status in {"blocked", "critical"}
            or "auth_succeeded_but_token_not_ready" in auth_reason
            or (expires_in and expires_in < _safe_float(lease_budget.get("critical_lease_seconds"), 600.0))
        )
    )

    process_rows = process_watchdog.get("status") if isinstance(process_watchdog.get("status"), list) else []
    watched_names = {"all_sleeves", "coinbase_loop", "coinbase_futures_loop"}
    down_targets = [
        str(row.get("name"))
        for row in process_rows
        if str(row.get("name")) in watched_names and not bool(row.get("process_live", False))
    ]
    paused_by_global_halt = any(bool(row.get("global_halt_active", False)) for row in process_rows if str(row.get("name")) in watched_names)
    restarted_targets = [
        str(row.get("name"))
        for row in process_rows
        if str(row.get("name")) in watched_names and row.get("restarted_pid")
    ]

    data_status = _status(data_plane, "missing")
    data_recovery = str(data_plane.get("recovery_state") or "")
    global_metrics = global_halt.get("metrics") if isinstance(global_halt.get("metrics"), dict) else {}
    runtime_clearance = str(data_plane.get("runtime_clearance_state") or global_metrics.get("runtime_clearance_state") or "")
    storage_backpressure = storage.get("backpressure") if isinstance(storage.get("backpressure"), dict) else {}
    queue_depth = _safe_int(data_plane.get("queue_depth"), _safe_int(storage_backpressure.get("total_pending_lines", storage.get("total_pending_lines")), 0))
    live_plane = live_runtime.get("live_plane") if isinstance(live_runtime.get("live_plane"), dict) else {}
    all_sleeves_running = any(
        str(row.get("name")) == "all_sleeves" and bool(row.get("process_live", row.get("running", 0)))
        for row in process_rows
    )
    live_lane_running = bool(live_plane.get("live_lane_running", False) or all_sleeves_running)

    needs = []
    if operator_auth_required:
        needs.append("operator_interactive_schwab_auth_refresh")
    elif auth_refresh_needed:
        needs.append("refresh_or_confirm_broker_auth_lease")
    if clear_blockers:
        needs.append("clear_hard_halt_blockers")
    if data_status in {"blocked", "critical", "degraded", "needs_work"} or runtime_clearance:
        needs.append("let_data_plane_recovery_and_runtime_clearance_settle")
    if halt_active and down_targets:
        needs.append("clear_halt_before_relaunching_live_sleeves")
    elif (not halt_active) and down_targets:
        needs.append("relaunch_and_verify_live_sleeves")

    recovery_sequence: list[list[str]] = [["./scripts/ops/opsctl.sh", "global-halt-refresh", "--json"]]
    if auth_refresh_needed:
        recovery_sequence.append(["./scripts/ops/opsctl.sh", "token-refresh", "--json"])
    if operator_auth_required:
        recovery_sequence.append(["./scripts/ops/opsctl.sh", "token-refresh-interactive", "--force", "--json"])
    if halt_active and clear_ready and not clear_blockers:
        recovery_sequence.append(["./scripts/ops/opsctl.sh", "global-halt-auto-clear", "--json"])
    if (not halt_active) or clear_ready:
        recovery_sequence.append(["./scripts/ops/opsctl.sh", "livefeed-refresh"])
    if queue_depth or data_status in {"blocked", "critical", "degraded", "needs_work"}:
        recovery_sequence.append(["./scripts/ops/opsctl.sh", "backpressure-drainers", "--apply", "--ttl-seconds", "900", "--json"])
    recovery_sequence.extend([
        ["./scripts/ops/opsctl.sh", "health-fast", "--json"],
        ["./scripts/ops/opsctl.sh", "system-self-model", "--json"],
    ])

    if operator_auth_required:
        next_safe_command = ["./scripts/ops/opsctl.sh", "token-refresh-interactive", "--force", "--json"]
    elif halt_active and auth_refresh_needed:
        next_safe_command = ["./scripts/ops/opsctl.sh", "token-refresh", "--json"]
    elif halt_active and clear_ready and not clear_blockers:
        next_safe_command = ["./scripts/ops/opsctl.sh", "global-halt-auto-clear", "--json"]
    elif halt_active:
        next_safe_command = ["./scripts/ops/opsctl.sh", "global-halt-refresh", "--json"]
    elif down_targets:
        next_safe_command = ["./scripts/ops/opsctl.sh", "livefeed-refresh"]
    elif queue_depth or data_status in {"blocked", "critical", "degraded", "needs_work"}:
        next_safe_command = ["./scripts/ops/opsctl.sh", "backpressure-drainers", "--apply", "--ttl-seconds", "900", "--json"]
    else:
        next_safe_command = ["./scripts/ops/opsctl.sh", "health-fast", "--json"]

    status = "ready"
    if halt_active or operator_auth_required:
        status = "blocked"
    elif down_targets or data_status in {"blocked", "critical", "degraded", "needs_work"} or runtime_clearance or auth_refresh_needed:
        status = "advisory"

    return {
        "status": status,
        "halt_active": halt_active,
        "clear_ready": clear_ready,
        "halt_reason": halt_reason,
        "halt_source": str(halt_payload.get("source") or ""),
        "halt_details": halt_details,
        "clear_blockers": clear_blockers,
        "degraded_clear_blockers": degraded_clear_blockers,
        "auth_status": auth_status,
        "lease_state": lease_state,
        "lease_expires_in_seconds": expires_in,
        "auth_ok": auth_ok,
        "broker_operable": broker_operable,
        "auth_reason": auth_reason,
        "auth_fallback_ladder": [str(item) for item in fallback_ladder],
        "auth_refresh_needed": auth_refresh_needed,
        "operator_auth_required": operator_auth_required,
        "data_plane_status": data_status,
        "data_recovery_state": data_recovery,
        "runtime_clearance_state": runtime_clearance,
        "queue_depth": queue_depth,
        "live_lane_running": live_lane_running,
        "paused_by_global_halt": paused_by_global_halt,
        "down_targets": down_targets,
        "restarted_targets": restarted_targets,
        "needs": needs,
        "next_safe_command": next_safe_command,
        "recovery_sequence": recovery_sequence,
        "post_clear_verifiers": [
            ["./scripts/ops/opsctl.sh", "health-fast", "--json"],
            ["./scripts/ops/opsctl.sh", "global-halt-status", "--json"],
        ],
        "control_contract": "active_halts_are_converted_into_auth_data_plane_clearance_relaunch_and_verification_steps_before_the_grandmaster_resumes_live_sleeves",
    }


def _dependency_edges() -> list[dict[str, str]]:
    return [
        {"from": "resource_guard", "to": "memory_efficiency", "reason": "memory and co-tenant context"},
        {"from": "memory_efficiency", "to": "runtime_throttle", "reason": "host profile and pressure caps"},
        {"from": "runtime_throttle", "to": "host_pressure_intelligence", "reason": "CPU, memory, swap, and open-app pressure state"},
        {"from": "memory_efficiency", "to": "host_pressure_intelligence", "reason": "memory pressure, compression, swap, and co-tenant awareness"},
        {"from": "host_pressure_intelligence", "to": "mlx_intelligence_router", "reason": "caps MLX jobs from CPU and unified-memory state"},
        {"from": "host_pressure_intelligence", "to": "library_utilization_router", "reason": "caps non-MLX support lanes from CPU and foreground app state"},
        {"from": "host_pressure_intelligence", "to": "backpressure_super_drainer", "reason": "drain waves respect CPU, memory, and foreground app pressure"},
        {"from": "runtime_throttle", "to": "mlx_runtime", "reason": "shared CPU/GPU memory and MLX batch pressure"},
        {"from": "mlx_runtime", "to": "mlx_intelligence_router", "reason": "MLX package and runtime readiness"},
        {"from": "mlx_library", "to": "mlx_intelligence_router", "reason": "pinned MLX library bundle coverage"},
        {"from": "mlx_intelligence_router", "to": "quant_model_control", "reason": "MLX workload routing and runtime caps"},
        {"from": "runtime_throttle", "to": "library_utilization_router", "reason": "non-MLX library worker caps and backend defaults"},
        {"from": "library_utilization_router", "to": "operator_cockpit", "reason": "library lane coverage and runtime support posture"},
        {"from": "ingestion_storage", "to": "operator_cockpit", "reason": "backpressure readiness"},
        {"from": "ingestion_storage", "to": "backpressure_drainer_fleet", "reason": "queue pressure and lane scoring"},
        {"from": "backpressure_drainer_fleet", "to": "backpressure_super_drainer", "reason": "focused lane candidate selection"},
        {"from": "backpressure_super_drainer", "to": "writer_cycle_coordinator", "reason": "bounded wave execution through one SQL writer"},
        {"from": "writer_process_intelligence", "to": "writer_cycle_coordinator", "reason": "writer health, process topology, and shard-lane expansion advice"},
        {"from": "process_fanout_guard", "to": "writer_process_intelligence", "reason": "writer expansion waits when host process fanout is over budget"},
        {"from": "writer_cycle_coordinator", "to": "ingestion_storage", "reason": "post-wave storage refresh and drain progress"},
        {"from": "backpressure_super_drainer", "to": "system_self_model", "reason": "drainer state vector for platform awareness"},
        {"from": "memory_efficiency", "to": "system_signal_bus", "reason": "resource signal normalized for whole-system decisions"},
        {"from": "runtime_throttle", "to": "system_signal_bus", "reason": "host pressure signal normalized for whole-system decisions"},
        {"from": "ingestion_storage", "to": "system_signal_bus", "reason": "storage and backpressure signal normalized for whole-system decisions"},
        {"from": "writer_process_intelligence", "to": "system_signal_bus", "reason": "writer state feeds the whole-system signal bus"},
        {"from": "drainer_intelligence", "to": "system_signal_bus", "reason": "drainer action feeds the whole-system signal bus"},
        {"from": "system_signal_bus", "to": "system_brain", "reason": "ranked signals drive the next safe infrastructure action"},
        {"from": "system_process_contracts", "to": "system_brain", "reason": "authority boundaries and concurrency limits constrain system decisions"},
        {"from": "system_brain", "to": "system_self_intelligence", "reason": "self-intelligence evaluates repeated actions, action effects, uncertainty, and trend before action"},
        {"from": "system_signal_bus", "to": "system_self_intelligence", "reason": "self-intelligence compares normalized signals against prior runs, causal diagnosis, and memory"},
        {"from": "system_self_intelligence", "to": "system_brain", "reason": "pre-action reflexes, action-effect verdicts, and causal routes can request refreshes before the brain action is trusted"},
        {"from": "system_brain", "to": "codex_handoff", "reason": "safe next action and do-not-do rules become a Codex attention packet"},
        {"from": "system_self_intelligence", "to": "codex_handoff", "reason": "uncertainty, causal root, action effect, route owner, and self-questions sharpen the Codex attention packet"},
        {"from": "whole_system_intelligence", "to": "system_self_model", "reason": "whole-system brain becomes a first-class self-model awareness domain"},
        {"from": "capital_growth_intelligence", "to": "capital_growth_awareness", "reason": "money-tree policy normalized into role-specific awareness packets"},
        {"from": "capital_growth_awareness", "to": "grand_master", "reason": "portfolio-level money-growth arbitration and live-money block state"},
        {"from": "capital_growth_awareness", "to": "masters", "reason": "per-sleeve growth, repair, and quarantine rules"},
        {"from": "capital_growth_awareness", "to": "sub_bots", "reason": "evidence, label, precision, and disconfirmation collection rules"},
        {"from": "capital_growth_awareness", "to": "master_infra", "reason": "storage, training, fill, attribution, and position-ledger freshness enforcement"},
        {"from": "capital_growth_awareness", "to": "system_self_model", "reason": "money-tree awareness becomes part of the shared self-model bus"},
        {"from": "use_mode_compliance", "to": "system_self_model", "reason": "personal, commercial, customer, marketing, and live authority boundaries become first-class awareness"},
        {"from": "use_mode_compliance", "to": "live_canary_readiness_contract", "reason": "live-money canary must pass use-mode and commercial-boundary evidence before promotion"},
        {"from": "schwab_indicator_intelligence", "to": "system_expansion_execution", "reason": "Schwab study and strategy catalog feeds the indicator-to-feature bridge lane"},
        {"from": "capital_rotation_control", "to": "system_expansion_execution", "reason": "paper-only sleeve rotation pressure feeds capital simulator v2"},
        {"from": "system_architecture_contract_graph", "to": "system_expansion_execution", "reason": "blocked, degraded, and stale nodes feed self-healing and stale-surface expansion lanes"},
        {"from": "runtime_throttle", "to": "system_expansion_execution", "reason": "runtime pressure feeds predictive stability, collector utility, and sleeve safe modes"},
        {"from": "system_expansion_execution", "to": "system_self_model", "reason": "12-lane expansion execution becomes a first-class self-model awareness surface"},
        {"from": "global_halt", "to": "operator_cockpit", "reason": "live collection clearance"},
        {"from": "master_infra", "to": "operator_cockpit", "reason": "process lane ownership"},
        {"from": "system_self_model", "to": "grand_master", "reason": "compressed self-state packet"},
    ]


def _dependency_awareness(surface_matrix: dict[str, dict[str, Any]], cockpit: dict[str, Any]) -> dict[str, Any]:
    hardening = cockpit.get("hardening_scorecard") if isinstance(cockpit.get("hardening_scorecard"), dict) else {}
    blocked_surfaces = sorted(
        name for name, row in surface_matrix.items() if str(row.get("status") or "") == "blocked"
    )
    degraded_surfaces = sorted(
        name for name, row in surface_matrix.items() if str(row.get("status") or "") in {"degraded", "needs_work"}
    )
    edges = _dependency_edges()
    status = "ready"
    if blocked_surfaces:
        status = "degraded"
    if not bool(hardening.get("process_ownership_canonical", True)):
        status = "degraded"
    return {
        "status": status,
        "blocked_surfaces": blocked_surfaces,
        "degraded_surfaces": degraded_surfaces,
        "process_ownership_canonical": bool(hardening.get("process_ownership_canonical", False)),
        "edge_count": len(edges),
        "edges": edges,
    }


def _growth_awareness(identity: dict[str, Any], memory: dict[str, Any], cockpit: dict[str, Any]) -> dict[str, Any]:
    expansion = memory.get("expansion_session") if isinstance(memory.get("expansion_session"), dict) else {}
    adaptive = cockpit.get("adaptive_posture") if isinstance(cockpit.get("adaptive_posture"), dict) else {}
    pressure_level = str(expansion.get("pressure_level") or adaptive.get("pressure_level") or "normal")
    active_bots = _safe_int(identity.get("active_bots"), _safe_int(adaptive.get("active_bots"), 0))
    collection_bots = _safe_int(identity.get("data_collection_active_bots"), _safe_int(adaptive.get("data_collection_active_bots"), 0))
    status = "ready"
    if pressure_level == "massive" and collection_bots >= 700:
        status = "advisory"
    return {
        "status": status,
        "pressure_level": pressure_level,
        "active_bots": active_bots,
        "data_collection_active_bots": collection_bots,
        "sleeve_profile_count": _safe_int(identity.get("sleeve_profile_count"), _safe_int(expansion.get("sleeve_profile_count"), 0)),
        "growth_contract": "new_expansions_must_land_as_collection_only_with_rollups_throttles_and_materialized_core_files",
    }


def _use_mode_compliance_awareness(use_mode: dict[str, Any]) -> dict[str, Any]:
    guard_status = _status(use_mode)
    if not use_mode:
        return {
            "status": "advisory",
            "use_mode": "personal",
            "guard_status": "missing",
            "personal_grade": "unknown",
            "perfect_personal_use_ready": False,
            "personal_live_money_ready": False,
            "commercial_use_intent_detected": False,
            "commercial_clearance_status": "missing",
            "commercial_blocker_count": 0,
            "commercial_blockers": [],
            "live_execution_authority": False,
            "customer_funds_allowed": False,
            "customer_order_execution_allowed": False,
            "raw_profitability_is_not_live_money_proof": True,
            "needs": ["refresh_use_mode_compliance_guard"],
            "next_safe_command": ["./scripts/ops/opsctl.sh", "use-mode-compliance", "--json"],
            "control_contract": "commercial_customer_facing_and_personal_use_boundaries_must_be_explicit_before_live_or_public_use",
        }
    commercial = use_mode.get("commercial_use") if isinstance(use_mode.get("commercial_use"), dict) else {}
    personal = use_mode.get("personal_use") if isinstance(use_mode.get("personal_use"), dict) else {}
    authority = use_mode.get("authority_boundaries") if isinstance(use_mode.get("authority_boundaries"), dict) else {}
    commercial_blockers = [str(item) for item in commercial.get("blockers", []) if str(item).strip()] if isinstance(commercial.get("blockers"), list) else []
    awareness_status = "ready"
    if guard_status == "blocked" or commercial_blockers or bool(authority.get("live_execution_authority", False)):
        awareness_status = "blocked"
    elif guard_status in {"needs_work", "degraded", "warning"} or not bool(personal.get("perfect_personal_use_ready", False)):
        awareness_status = "advisory"
    return {
        "status": awareness_status,
        "use_mode": str(use_mode.get("use_mode") or "personal"),
        "guard_status": guard_status or "missing",
        "personal_grade": str(personal.get("grade") or "unknown"),
        "perfect_personal_use_ready": bool(personal.get("perfect_personal_use_ready", False)),
        "personal_live_money_ready": bool(personal.get("personal_live_money_ready", False)),
        "commercial_use_intent_detected": bool(commercial.get("commercial_use_intent_detected", False)),
        "commercial_clearance_status": str(commercial.get("commercial_clearance_status") or ""),
        "commercial_blocker_count": len(commercial_blockers),
        "commercial_blockers": commercial_blockers,
        "live_execution_authority": bool(authority.get("live_execution_authority", False)),
        "customer_funds_allowed": bool(authority.get("customer_funds_allowed", False)),
        "customer_order_execution_allowed": bool(authority.get("customer_order_execution_allowed", False)),
        "raw_profitability_is_not_live_money_proof": bool(authority.get("raw_profitability_is_not_live_money_proof", True)),
        "needs": [
            *([] if bool(personal.get("perfect_personal_use_ready", False)) else ["resolve_personal_use_posture_blockers"]),
            *([] if not commercial_blockers else ["clear_commercial_boundary_blockers_before_public_or_customer_use"]),
        ],
        "next_safe_command": ["./scripts/ops/opsctl.sh", "use-mode-compliance", "--json"],
        "control_contract": "personal_a_plus_is_guarded_paper_data_collection_readiness_commercial_or_customer_use_requires_explicit_review_evidence",
    }


def _surface_status(surface_matrix: dict[str, dict[str, Any]], name: str) -> str:
    aliases = {
        "resource_guard": "memory_efficiency",
        "host_pressure_intelligence": "runtime_throttle",
        "grand_master": "operator_cockpit",
        "system_self_model": "system_self_model",
        "drainer_intelligence": "backpressure_super_drainer",
        "system_signal_bus": "system_signal_bus",
        "system_brain": "system_brain",
        "system_process_contracts": "system_process_contracts",
        "system_self_intelligence": "system_self_intelligence",
        "codex_handoff": "codex_handoff",
        "whole_system_intelligence": "whole_system_intelligence",
    }
    key = aliases.get(name, name)
    if key == "system_self_model":
        return "ready"
    row = surface_matrix.get(key) if isinstance(surface_matrix.get(key), dict) else {}
    return str(row.get("status") or "missing")


def _dependency_memory(
    surface_matrix: dict[str, dict[str, Any]],
    previous: dict[str, Any],
    *,
    now: datetime,
) -> dict[str, Any]:
    previous_last_good = previous.get("last_good_snapshots") if isinstance(previous.get("last_good_snapshots"), dict) else {}
    last_good: dict[str, dict[str, Any]] = {
        str(name): row for name, row in previous_last_good.items() if isinstance(row, dict)
    }
    stale_sources: list[dict[str, Any]] = []
    managed_stale_sources: list[dict[str, Any]] = []
    ready_like = {"ready", "ok", "watch", "advisory", "thin", "steady_state", "applied_with_followups", "handoff_requested"}
    dashboard_row = surface_matrix.get("runtime_gate_dashboard") if isinstance(surface_matrix.get("runtime_gate_dashboard"), dict) else {}
    guarded_paper_context_enabled = bool(dashboard_row.get("guarded_paper_context_enabled", False))

    for name, row in surface_matrix.items():
        status = str(row.get("status") or "missing")
        payload_hash = str(row.get("payload_sha256") or "")
        loaded = bool(row.get("loaded", False))
        age_minutes = row.get("age_minutes")
        if loaded and payload_hash and status in ready_like:
            last_good[name] = {
                "status": status,
                "payload_sha256": payload_hash,
                "payload_hash_short": str(row.get("payload_hash_short") or payload_hash[:12]),
                "timestamp_utc": str(row.get("timestamp_utc") or now.isoformat()),
            }
        if isinstance(age_minutes, (int, float)):
            stale_limit = 90.0 if name in {"global_halt", "memory_efficiency", "runtime_throttle"} else 360.0
            if float(age_minutes) > stale_limit:
                row_payload = {
                    "surface": name,
                    "age_minutes": round(float(age_minutes), 3),
                    "stale_limit_minutes": stale_limit,
                    "status": status,
                }
                if guarded_paper_context_enabled and name in GUARDED_PAPER_OPTIONAL_STALE_SURFACES and status in ready_like:
                    row_payload.update(
                        {
                            "managed_by": "runtime_gate_dashboard",
                            "managed_control_state": "optional_support_artifact_refresh_deferred_while_guarded_paper_soak_is_green",
                            "guarded_paper_advisory_only": True,
                        }
                    )
                    managed_stale_sources.append(row_payload)
                else:
                    stale_sources.append(row_payload)

    edge_health: list[dict[str, Any]] = []
    for edge in _dependency_edges():
        source = str(edge.get("from") or "")
        target = str(edge.get("to") or "")
        source_status = _surface_status(surface_matrix, source)
        target_status = _surface_status(surface_matrix, target)
        edge_status = _worst_status([source_status, target_status])
        edge_health.append(
            {
                **edge,
                "source_status": source_status,
                "target_status": target_status,
                "edge_status": edge_status,
            }
        )

    blocked_edges = [edge for edge in edge_health if str(edge.get("edge_status") or "") in {"blocked", "critical"}]
    degraded_edges = [
        edge
        for edge in edge_health
        if str(edge.get("edge_status") or "") in {"degraded", "needs_work", "missing"}
    ]
    status = "ready"
    if blocked_edges:
        status = "blocked"
    elif degraded_edges or stale_sources:
        status = "degraded"

    return {
        "timestamp_utc": now.isoformat(),
        "schema_version": 1,
        "overall_status": status,
        "edge_count": len(edge_health),
        "blocked_edge_count": len(blocked_edges),
        "degraded_edge_count": len(degraded_edges),
        "stale_source_count": len(stale_sources),
        "managed_stale_source_count": len(managed_stale_sources),
        "stale_sources": stale_sources,
        "managed_stale_sources": managed_stale_sources,
        "edge_health": edge_health,
        "last_good_snapshot_count": len(last_good),
        "last_good_snapshots": last_good,
        "surface_hashes": {
            name: str(row.get("payload_sha256") or "")
            for name, row in surface_matrix.items()
            if str(row.get("payload_sha256") or "")
        },
        "memory_contract": "surface_edges_last_good_hashes_stale_source_age_and_dependency_health",
    }


def _event_key(event: dict[str, Any]) -> str:
    stable = {
        key: value
        for key, value in event.items()
        if key not in {"timestamp_utc", "first_seen_utc", "last_seen_utc", "seen_count"}
    }
    return _json_sha256(stable)


def _failure_memory_index(
    *,
    global_halt: dict[str, Any],
    incident: dict[str, Any],
    cockpit: dict[str, Any],
    storage: dict[str, Any],
    throttle: dict[str, Any],
    tripwire: dict[str, Any],
    previous: dict[str, Any],
    now: datetime,
) -> dict[str, Any]:
    events: list[dict[str, Any]] = []
    halt_active = bool(global_halt.get("halt", False))
    events.append(
        {
            "timestamp_utc": now.isoformat(),
            "event_type": "global_halt",
            "source": "global_killswitch",
            "severity": "blocked" if halt_active else "ready",
            "state": "active" if halt_active else "clear",
            "reason": ",".join(str(item) for item in (global_halt.get("reasons") or []) if str(item).strip()) or str(global_halt.get("action") or "none"),
        }
    )

    if incident:
        incident_event = str(incident.get("event") or incident.get("status") or "state_update")
        failed_checks = incident.get("failed_checks") if isinstance(incident.get("failed_checks"), list) else []
        events.append(
            {
                "timestamp_utc": str(incident.get("timestamp_utc") or now.isoformat()),
                "event_type": "incident_auto_halt",
                "source": "incident_auto_halt",
                "severity": "blocked" if bool(incident.get("halt", False)) else ("degraded" if failed_checks else "ready"),
                "state": incident_event,
                "reason": ",".join(str(item) for item in failed_checks if str(item).strip()) or "none",
            }
        )

    adaptive = cockpit.get("adaptive_posture") if isinstance(cockpit.get("adaptive_posture"), dict) else {}
    for blocker in adaptive.get("hard_blockers") if isinstance(adaptive.get("hard_blockers"), list) else []:
        events.append(
            {
                "timestamp_utc": now.isoformat(),
                "event_type": "hard_blocker",
                "source": "operator_cockpit",
                "severity": "degraded",
                "state": "active",
                "reason": str(blocker),
            }
        )

    backpressure = storage.get("backpressure") if isinstance(storage.get("backpressure"), dict) else {}
    pending_lines = _safe_int(backpressure.get("total_pending_lines"), _safe_int(backpressure.get("core_pending_lines"), 0))
    pressure_index = _safe_float(storage.get("pressure_index"), 0.0)
    if pending_lines or pressure_index > 0:
        events.append(
            {
                "timestamp_utc": str(storage.get("timestamp_utc") or now.isoformat()),
                "event_type": "backpressure",
                "source": "ingestion_storage",
                "severity": "degraded" if pressure_index >= 0.5 else "advisory",
                "state": str(storage.get("severity") or storage.get("overall_status") or "observed"),
                "reason": f"pending_lines={pending_lines} pressure_index={pressure_index:.3f}",
            }
        )

    throttle_profile = str(throttle.get("throttle_profile") or "")
    if throttle_profile and throttle_profile != "observe":
        events.append(
            {
                "timestamp_utc": str(throttle.get("timestamp_utc") or now.isoformat()),
                "event_type": "runtime_throttle",
                "source": "runtime_throttle",
                "severity": str(throttle.get("overall_status") or "advisory"),
                "state": throttle_profile,
                "reason": f"host_saturation_score={_safe_float(throttle.get('host_saturation_score'), 0.0):.2f}",
            }
        )

    if bool(tripwire.get("active", False)):
        incidents = tripwire.get("active_incidents") if isinstance(tripwire.get("active_incidents"), list) else []
        targets = ",".join(str(row.get("target") or "") for row in incidents if isinstance(row, dict) and str(row.get("target") or "").strip())
        events.append(
            {
                "timestamp_utc": str(tripwire.get("timestamp_utc") or now.isoformat()),
                "event_type": "tripwire",
                "source": "shadow_watchdog",
                "severity": "blocked",
                "state": "active",
                "reason": targets or "active_tripwire",
            }
        )

    previous_events = previous.get("recent_events") if isinstance(previous.get("recent_events"), list) else []
    merged: dict[str, dict[str, Any]] = {}
    for raw in previous_events:
        if not isinstance(raw, dict):
            continue
        key = str(raw.get("event_key") or _event_key(raw))
        merged[key] = dict(raw, event_key=key)
    current_keys: list[str] = []
    for event in events:
        key = _event_key(event)
        current_keys.append(key)
        existing = merged.get(key)
        if existing:
            existing["last_seen_utc"] = event.get("timestamp_utc") or now.isoformat()
            existing["seen_count"] = _safe_int(existing.get("seen_count"), 1) + 1
            existing["severity"] = event.get("severity", existing.get("severity"))
            merged[key] = existing
        else:
            merged[key] = {
                **event,
                "event_key": key,
                "first_seen_utc": event.get("timestamp_utc") or now.isoformat(),
                "last_seen_utc": event.get("timestamp_utc") or now.isoformat(),
                "seen_count": 1,
            }
    recent_events = sorted(merged.values(), key=lambda row: str(row.get("last_seen_utc") or ""))[-120:]
    active_risk_events = [
        row
        for row in events
        if str(row.get("severity") or "") in {"blocked", "critical", "degraded", "needs_work"}
    ]
    status = "ready"
    if any(str(row.get("severity") or "") in {"blocked", "critical"} for row in active_risk_events):
        status = "blocked"
    elif active_risk_events:
        status = "degraded"
    return {
        "timestamp_utc": now.isoformat(),
        "schema_version": 1,
        "overall_status": status,
        "current_event_count": len(events),
        "active_risk_event_count": len(active_risk_events),
        "recent_event_count": len(recent_events),
        "current_event_keys": current_keys,
        "recent_events": recent_events,
        "active_risk_events": active_risk_events,
        "memory_contract": "global_halt_incident_tripwire_backpressure_margin_and_runtime_pressure_as_replayable_causes",
    }


def _registry_projection(registry: dict[str, Any]) -> dict[str, dict[str, Any]]:
    projection: dict[str, dict[str, Any]] = {}
    for row in _registry_rows(registry):
        bot_id = str(row.get("bot_id") or row.get("name") or "").strip()
        if not bot_id:
            continue
        summary = {
            "active": bool(row.get("active", False)),
            "lifecycle_state": str(row.get("lifecycle_state") or ""),
            "data_collection_active": bool(row.get("data_collection_active", False)),
            "training_excluded": bool(row.get("training_excluded", False)),
            "sleeve_profile": str(row.get("sleeve_profile") or ""),
            "slot_kind": str(row.get("slot_kind") or ""),
            "tier": str(row.get("tier") or row.get("bot_tier") or ""),
            "capability_pack_slug": str(row.get("capability_pack_slug") or ""),
            "system_self_awareness_version": str(row.get("system_self_awareness_version") or ""),
        }
        projection[bot_id] = {"fingerprint": _json_sha256(summary), "summary": summary}
    return projection


def _registry_diff_memory(registry: dict[str, Any], previous: dict[str, Any], *, now: datetime) -> dict[str, Any]:
    current = _registry_projection(registry)
    previous_map = previous.get("bot_fingerprints") if isinstance(previous.get("bot_fingerprints"), dict) else {}
    added = sorted(bot_id for bot_id in current if bot_id not in previous_map)
    removed = sorted(bot_id for bot_id in previous_map if bot_id not in current)
    changed = sorted(
        bot_id
        for bot_id, row in current.items()
        if bot_id in previous_map
        and str((previous_map.get(bot_id) or {}).get("fingerprint") or "") != str(row.get("fingerprint") or "")
    )
    fingerprint = _json_sha256({bot_id: row.get("fingerprint") for bot_id, row in sorted(current.items())})
    previous_fingerprint = str(previous.get("registry_fingerprint") or "")
    if not previous_map:
        status = "baseline"
    elif added or removed or changed:
        status = "changed"
    else:
        status = "ready"
    return {
        "timestamp_utc": now.isoformat(),
        "schema_version": 1,
        "overall_status": "ready",
        "diff_status": status,
        "registry_fingerprint": fingerprint,
        "previous_registry_fingerprint": previous_fingerprint,
        "fingerprint_changed": bool(previous_fingerprint and previous_fingerprint != fingerprint),
        "current_bot_count": len(current),
        "previous_bot_count": len(previous_map),
        "added_count": len(added),
        "removed_count": len(removed),
        "changed_count": len(changed),
        "added_bot_ids": added[:80],
        "removed_bot_ids": removed[:80],
        "changed_bot_ids": changed[:80],
        "bot_fingerprints": current,
        "memory_contract": "bot_roster_diff_between_expansions_with_stable_bot_fingerprints",
    }


def _compact_registry_diff_memory(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in payload.items()
        if key not in {"bot_fingerprints"}
    }


def _self_reporting_awareness(cockpit: dict[str, Any], surface_matrix: dict[str, dict[str, Any]]) -> dict[str, Any]:
    loaded_count = sum(1 for row in surface_matrix.values() if bool(row.get("loaded", False)))
    recommended_actions = cockpit.get("recommended_actions") if isinstance(cockpit.get("recommended_actions"), list) else []
    status = "ready" if loaded_count >= 6 else "degraded"
    return {
        "status": status,
        "surface_count": len(surface_matrix),
        "loaded_surface_count": loaded_count,
        "recommended_action_count": len(recommended_actions),
        "reporting_contract": "explain_current_state_why_it_downshifted_what_changed_and_what_to_fix_next",
    }


def _opsctl_self_model_refresh_wired(project_root: Path = PROJECT_ROOT) -> bool:
    opsctl_path = project_root / "scripts" / "ops" / "opsctl.sh"
    try:
        text = opsctl_path.read_text(encoding="utf-8")
    except OSError:
        return False
    required_markers = [
        "refresh_system_self_model_quietly",
        "run_then_refresh_self_model",
        "system_self_awareness_expansion.py",
        "memory_efficiency_control.py",
        "runtime_throttle_control.py",
        "global_risk_killswitch.py",
        "core_bot_materialization_guard.py",
    ]
    return all(marker in text for marker in required_markers)


def _runtime_throttle_cotenant_wired(project_root: Path = PROJECT_ROOT) -> bool:
    throttle_path = project_root / "scripts" / "ops" / "runtime_throttle_control.py"
    try:
        text = throttle_path.read_text(encoding="utf-8")
    except OSError:
        return False
    required_markers = [
        "cotenant_awareness_contract",
        "_cotenant_awareness_contract",
        "_apply_cotenant_profile_guard",
    ]
    return all(marker in text for marker in required_markers)


def _host_pressure_intelligence_wired(project_root: Path = PROJECT_ROOT) -> bool:
    self_path = project_root / "scripts" / "ops" / "system_self_model.py"
    router_path = project_root / "scripts" / "ops" / "mlx_intelligence_router.py"
    try:
        self_text = self_path.read_text(encoding="utf-8")
        router_text = router_path.read_text(encoding="utf-8")
    except OSError:
        return False
    required_markers = [
        "_host_pressure_intelligence",
        "cpu_pressure_level",
        "host_pressure_state",
        "host_saturation_score",
        "100_percent_library_coverage_with_cpu_memory_aware_caps",
    ]
    combined = f"{self_text}\n{router_text}"
    return all(marker in combined for marker in required_markers)


def _mlx_intelligence_router_wired(project_root: Path = PROJECT_ROOT) -> bool:
    router_path = project_root / "scripts" / "ops" / "mlx_intelligence_router.py"
    opsctl_path = project_root / "scripts" / "ops" / "opsctl.sh"
    try:
        router_text = router_path.read_text(encoding="utf-8")
        opsctl_text = opsctl_path.read_text(encoding="utf-8")
    except OSError:
        return False
    required_markers = [
        "LANE_SPECS",
        "library_utilization_matrix",
        "recommended_runtime_env",
        "mlx-intelligence-router",
    ]
    return all(marker in f"{router_text}\n{opsctl_text}" for marker in required_markers)


def _library_utilization_router_wired(project_root: Path = PROJECT_ROOT) -> bool:
    router_path = project_root / "scripts" / "ops" / "library_utilization_router.py"
    opsctl_path = project_root / "scripts" / "ops" / "opsctl.sh"
    try:
        router_text = router_path.read_text(encoding="utf-8")
        opsctl_text = opsctl_path.read_text(encoding="utf-8")
    except OSError:
        return False
    required_markers = [
        "LANE_SPECS",
        "library_utilization_matrix",
        "LIBRARY_DEFAULT_ML_BACKEND",
        "library-utilization-router",
    ]
    return all(marker in f"{router_text}\n{opsctl_text}" for marker in required_markers)


def _drainer_intelligence_wired(project_root: Path = PROJECT_ROOT) -> bool:
    super_path = project_root / "scripts" / "ops" / "backpressure_super_drainer.py"
    opsctl_path = project_root / "scripts" / "ops" / "opsctl.sh"
    try:
        super_text = super_path.read_text(encoding="utf-8")
        opsctl_text = opsctl_path.read_text(encoding="utf-8")
    except OSError:
        return False
    required_markers = [
        "self_intelligence_contract",
        "drainer_strategy",
        "grandmaster_context_packet",
        "backpressure-super-drainer",
        "run_then_refresh_self_model",
        "writer_cycle_coordinator.py",
    ]
    return all(marker in f"{super_text}\n{opsctl_text}" for marker in required_markers)


def _writer_process_intelligence_wired(project_root: Path = PROJECT_ROOT) -> bool:
    writer_path = project_root / "scripts" / "ops" / "writer_process_intelligence.py"
    coordinator_path = project_root / "scripts" / "ops" / "writer_cycle_coordinator.py"
    opsctl_path = project_root / "scripts" / "ops" / "opsctl.sh"
    shard_path = project_root / "scripts" / "ops" / "sql_link_shard_manager.py"
    try:
        writer_text = writer_path.read_text(encoding="utf-8")
        coordinator_text = coordinator_path.read_text(encoding="utf-8")
        opsctl_text = opsctl_path.read_text(encoding="utf-8")
        shard_text = shard_path.read_text(encoding="utf-8")
    except OSError:
        return False
    required_markers = [
        "writer_expansion_contract",
        "writer_process_intelligence",
        "writer-process-intelligence",
        "single_writer_only",
        "writer_progress",
        "admission_evidence",
    ]
    return all(marker in f"{writer_text}\n{coordinator_text}\n{opsctl_text}\n{shard_text}" for marker in required_markers)


def _whole_system_intelligence_wired(project_root: Path = PROJECT_ROOT) -> bool:
    coordinator_path = project_root / "scripts" / "ops" / "system_intelligence_coordinator.py"
    opsctl_path = project_root / "scripts" / "ops" / "opsctl.sh"
    try:
        coordinator_text = coordinator_path.read_text(encoding="utf-8")
        opsctl_text = opsctl_path.read_text(encoding="utf-8")
    except OSError:
        return False
    required_markers = [
        "system_signal_bus",
        "system_brain",
        "system_process_contracts",
        "system_self_intelligence",
        "self_intelligence_contract",
        "causal_diagnosis",
        "action_effectiveness",
        "integration_routing",
        "codex_handoff",
        "global_safety_contract",
        "system-intelligence",
    ]
    return all(marker in f"{coordinator_text}\n{opsctl_text}" for marker in required_markers)


def _implementation_flags(project_root: Path = PROJECT_ROOT) -> dict[str, bool]:
    return {
        "self_model_cadence": _opsctl_self_model_refresh_wired(project_root),
        "dependency_graph": True,
        "failure_memory": True,
        "halt_recovery_intelligence": True,
        "resource_awareness": _runtime_throttle_cotenant_wired(project_root),
        "host_pressure_intelligence": _host_pressure_intelligence_wired(project_root),
        "bot_awareness": True,
        "self_reporting": True,
        "mlx_compute_brain": _mlx_intelligence_router_wired(project_root),
        "library_utilization_brain": _library_utilization_router_wired(project_root),
        "drainer_intelligence": _drainer_intelligence_wired(project_root),
        "writer_process_intelligence": _writer_process_intelligence_wired(project_root),
        "whole_system_intelligence": _whole_system_intelligence_wired(project_root),
        "system_self_intelligence": _whole_system_intelligence_wired(project_root),
    }


def _optimization_plan(
    domains: dict[str, dict[str, Any]],
    surface_matrix: dict[str, dict[str, Any]],
    *,
    implementation_flags: dict[str, bool],
) -> list[dict[str, Any]]:
    plan = [
        {
            "rank": 1,
            "lane": "self_model_cadence",
            "priority": "critical",
            "upgrade": "refresh system-self-model after expansion, memory-efficiency, runtime-throttle, global-halt, and materialization commands",
            "benefit": "keeps the Grand Master and cockpit from reasoning from stale self-state",
            "implemented": bool(implementation_flags.get("self_model_cadence", False)),
        },
        {
            "rank": 2,
            "lane": "dependency_graph",
            "priority": "high",
            "upgrade": "persist dependency edges with stale-source age and last-good snapshot hashes",
            "benefit": "makes failures explainable as upstream/downstream chains instead of isolated red statuses",
            "implemented": bool(implementation_flags.get("dependency_graph", False)),
        },
        {
            "rank": 3,
            "lane": "failure_memory",
            "priority": "high",
            "upgrade": "join global halt, tripwire, backpressure, feed-cut, and margin-guard events into one incident memory table",
            "benefit": "lets the system compare current pressure to previous recoveries before hard-halting",
            "implemented": bool(implementation_flags.get("failure_memory", False)),
        },
        {
            "rank": 4,
            "lane": "halt_recovery_intelligence",
            "priority": "critical",
            "upgrade": "convert active global halts into a safe precheck, clearance, relaunch, and verification plan",
            "benefit": "lets the intelligence layer know why the halt happened, what it needs, and which bounded command should run next",
            "implemented": bool(implementation_flags.get("halt_recovery_intelligence", False)),
        },
        {
            "rank": 5,
            "lane": "resource_awareness",
            "priority": "high",
            "upgrade": "teach runtime-throttle to consume cotenant_awareness mode directly instead of inferring from status alone",
            "benefit": "keeps foreground apps smooth while preserving live collection and paper execution",
            "implemented": bool(implementation_flags.get("resource_awareness", False)),
        },
        {
            "rank": 6,
            "lane": "host_pressure_intelligence",
            "priority": "critical",
            "upgrade": "join CPU pressure, memory pressure, swap, host saturation, and open-app co-tenancy into one intelligence routing state",
            "benefit": "lets the platform brain downshift MLX, reporting, training, and drainer work before foreground apps or live collection feel pressure",
            "implemented": bool(implementation_flags.get("host_pressure_intelligence", False)),
        },
        {
            "rank": 7,
            "lane": "bot_awareness",
            "priority": "medium",
            "upgrade": "add a registry diff memory that records what changed between bot expansions",
            "benefit": "makes bot growth auditable and easier to explain in reports",
            "implemented": bool(implementation_flags.get("bot_awareness", False)),
        },
        {
            "rank": 8,
            "lane": "self_reporting",
            "priority": "medium",
            "upgrade": "generate a daily natural-language self-brief with posture, changes, blockers, and safe next commands",
            "benefit": "gives you a quick morning readout without opening every health file",
            "implemented": bool(implementation_flags.get("self_reporting", False)),
        },
        {
            "rank": 9,
            "lane": "mlx_compute_brain",
            "priority": "high",
            "upgrade": "route MLX language, embedding, graph, audio, VLM, SNN, data, rough-path, and quant workloads through one capped intelligence router",
            "benefit": "uses the installed MLX library stack broadly without letting CPU or shared-memory work starve collectors or foreground apps",
            "implemented": bool(implementation_flags.get("mlx_compute_brain", False)),
        },
        {
            "rank": 10,
            "lane": "library_utilization_brain",
            "priority": "high",
            "upgrade": "route non-MLX libraries through owner lanes while keeping MLX as the default live intelligence backend",
            "benefit": "turns the rest of the dependency stack into governed support lanes instead of idle or competing backends",
            "implemented": bool(implementation_flags.get("library_utilization_brain", False)),
        },
        {
            "rank": 11,
            "lane": "drainer_intelligence",
            "priority": "critical",
            "upgrade": "make the drainer fleet, super-drainer, writer coordinator, and storage autopilot part of the self-model state vector",
            "benefit": "lets the platform brain reason about backlog pressure, active drain lanes, wave progress, and single-writer safety before halts or expansions",
            "implemented": bool(implementation_flags.get("drainer_intelligence", False)),
        },
        {
            "rank": 12,
            "lane": "writer_process_intelligence",
            "priority": "critical",
            "upgrade": "give the SQL writer layer its own health, process-topology, shard-lane, and recovery decision packet",
            "benefit": "expands writer throughput with targeted shard lanes while preserving the single-writer lock and process fanout guardrails",
            "implemented": bool(implementation_flags.get("writer_process_intelligence", False)),
        },
        {
            "rank": 13,
            "lane": "whole_system_intelligence",
            "priority": "critical",
            "upgrade": "join signal bus, system brain, process contracts, and Codex handoff into one whole-system intelligence coordinator",
            "benefit": "lets the platform select one safe next infrastructure move and hand Codex a concise attention packet",
            "implemented": bool(implementation_flags.get("whole_system_intelligence", False)),
        },
        {
            "rank": 14,
            "lane": "system_self_intelligence",
            "priority": "critical",
            "upgrade": "add trend memory, action-effect scoring, causal diagnosis, integration routing, contract checks, and pre-action reflexes to the whole-system brain",
            "benefit": "keeps the brain from acting on stale or contradictory signals, teaches it when repeated actions are not clearing pressure, and routes the next move to the right consumer",
            "implemented": bool(implementation_flags.get("system_self_intelligence", False)),
        },
    ]
    degraded = [
        name
        for name, row in domains.items()
        if str(row.get("status") or "") in {"advisory", "needs_work", "degraded", "blocked"}
    ]
    blocked_surfaces = [
        name for name, row in surface_matrix.items() if str(row.get("status") or "") == "blocked"
    ]
    for item in plan:
        implemented = bool(item.get("implemented", False))
        item["triggered_by_current_state"] = (item["lane"] in degraded or bool(blocked_surfaces)) and not implemented
    return plan


def _advanced_upgrade_backlog(
    domains: dict[str, dict[str, Any]],
    dependency_memory: dict[str, Any],
    failure_index: dict[str, Any],
    registry_diff: dict[str, Any],
    surface_matrix: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    blocked_or_degraded = [
        name
        for name, row in surface_matrix.items()
        if str(row.get("status") or "") in {"blocked", "degraded", "needs_work"}
    ]
    active_bots = _safe_int((domains.get("bot_awareness") or {}).get("active_bots"), 0)
    active_risk_events = _safe_int(failure_index.get("active_risk_event_count"), 0)
    stale_sources = _safe_int(dependency_memory.get("stale_source_count"), 0)
    mlx_router_status = str((surface_matrix.get("mlx_intelligence_router") or {}).get("status") or "missing")
    library_router_status = str((surface_matrix.get("library_utilization_router") or {}).get("status") or "missing")
    drainer = domains.get("drainer_intelligence") if isinstance(domains.get("drainer_intelligence"), dict) else {}
    drainer_status = str(drainer.get("status") or "missing")
    drainer_target_met = bool(drainer.get("target_met", False))
    host_pressure = domains.get("host_pressure_intelligence") if isinstance(domains.get("host_pressure_intelligence"), dict) else {}
    host_pressure_status = str(host_pressure.get("status") or "missing")
    return [
        {
            "rank": 1,
            "lane": "predictive_stability",
            "upgrade": "learn pressure trajectories from memory, runtime throttle, MLX, storage, and halt history before pressure trips a global halt",
            "triggered": bool(active_bots >= 700 or active_risk_events),
            "benefit": "moves the system from reactive downshifts to preemptive smoothing",
        },
        {
            "rank": 2,
            "lane": "mlx_compute_brain",
            "upgrade": "route MLX model, simulation, and quant-pricing workloads through the same cotenant-aware throttle and shared-memory budget",
            "triggered": bool(
                mlx_router_status not in {"ready", "advisory"}
                or (active_bots >= 700 and (domains.get("mlx_intelligence_awareness") or {}).get("status") in {"missing", "blocked", "degraded"})
            ),
            "benefit": "keeps MLX fast without letting GPU/shared-memory work starve collectors, SQL writers, or foreground apps",
        },
        {
            "rank": 3,
            "lane": "library_utilization_brain",
            "upgrade": "route every non-MLX locked/runtime package into support, storage, reporting, canary, ingestion, or runtime lanes while preserving MLX as default",
            "triggered": bool(library_router_status not in {"ready", "advisory"}),
            "benefit": "keeps the dependency stack useful and governed without letting non-MLX model backends compete with MLX during live collection",
        },
        {
            "rank": 4,
            "lane": "drainer_self_intelligence",
            "upgrade": "feed super-drainer strategy, memory, active lane, target clearance, and writer safety into the self-model and Grand Master packet",
            "triggered": bool(drainer_status not in {"ready", "advisory"} or not drainer_target_met),
            "benefit": "lets the platform choose drain, wait, throttle, or expand based on queue physiology instead of raw backlog files",
        },
        {
            "rank": 5,
            "lane": "host_pressure_reflex_layer",
            "upgrade": "feed CPU, memory, swap, host saturation, and co-running apps into MLX caps, library caps, drainer waves, and training cadence",
            "triggered": bool(host_pressure_status in {"advisory", "degraded", "blocked"}),
            "benefit": "keeps live data and paper trading smooth while still using the intelligence layer aggressively when the Mac is clear",
        },
        {
            "rank": 6,
            "lane": "self_healing_router",
            "upgrade": "map each blocked surface to its safest recovery command, required prechecks, and post-refresh verifier",
            "triggered": bool(blocked_or_degraded),
            "benefit": "turns cockpit red rows into bounded recovery playbooks instead of manual hunting",
        },
        {
            "rank": 7,
            "lane": "collector_utility_budget",
            "upgrade": "score each collector by freshness value, storage cost, downstream use, and overlap so low-value collectors thin first",
            "triggered": bool(active_bots >= 700),
            "benefit": "keeps data breadth high while reducing CPU, storage, and writer pressure",
        },
        {
            "rank": 8,
            "lane": "hot_path_storage_budget",
            "upgrade": "assign per-surface hot/warm/cold storage budgets and degrade report/explanation writes before trading-path writes",
            "triggered": bool("storage_tier_policy" in blocked_or_degraded or "artifact_freshness" in blocked_or_degraded),
            "benefit": "protects paper/live collection when reports, artifacts, or explainers grow too fast",
        },
        {
            "rank": 9,
            "lane": "stale_surface_autofix",
            "upgrade": "auto-refresh stale required surfaces, compare last-good hashes, then suppress stale-only blockers when the dependency chain is otherwise healthy",
            "triggered": bool(stale_sources),
            "benefit": "prevents stale dashboards from causing unnecessary halt pressure",
        },
        {
            "rank": 10,
            "lane": "grandmaster_safe_mode",
            "upgrade": "feed a compressed self-state packet into Grand Master routing so it can choose observe, sample, buffer, or pause per sleeve",
            "triggered": bool(active_bots >= 700),
            "benefit": "lets the brain downshift specific sleeves instead of using blunt global controls",
        },
        {
            "rank": 11,
            "lane": "registry_growth_governance",
            "upgrade": "require every new bot wave to emit expected storage, CPU, labels, training horizon, teacher lineage, and rollback metadata",
            "triggered": bool(registry_diff.get("fingerprint_changed") or registry_diff.get("diff_status") == "baseline"),
            "benefit": "keeps future expansion clean and auditable",
        },
        {
            "rank": 12,
            "lane": "self_brief_learning",
            "upgrade": "turn daily self-briefs into a rolling operator memory with what changed, what helped, and what failed",
            "triggered": True,
            "benefit": "makes the platform better at explaining itself over time",
        },
    ]


def _upgrade_optimizer_payload(
    payload: dict[str, Any],
    advanced_backlog: list[dict[str, Any]],
    *,
    now: datetime,
) -> dict[str, Any]:
    implemented = [
        row
        for row in payload.get("upgrades_and_optimizations", [])
        if isinstance(row, dict) and bool(row.get("implemented", False))
    ]
    triggered = [
        row
        for row in payload.get("upgrades_and_optimizations", [])
        if isinstance(row, dict) and bool(row.get("triggered_by_current_state", False))
    ]
    advanced_triggered = [row for row in advanced_backlog if bool(row.get("triggered", False))]
    return {
        "timestamp_utc": now.isoformat(),
        "schema_version": 1,
        "overall_status": "ready",
        "implemented_upgrade_count": len(implemented),
        "triggered_upgrade_count": len(triggered),
        "advanced_triggered_count": len(advanced_triggered),
        "implemented_lanes": [str(row.get("lane") or "") for row in implemented],
        "active_upgrade_lanes": [str(row.get("lane") or "") for row in triggered],
        "next_generation_backlog": advanced_backlog,
        "top_next_actions": [str(row.get("upgrade") or "") for row in advanced_triggered[:4]],
        "optimizer_contract": "rank_next_safe_platform_brain_stabilization_and_optimization_work",
    }


def _render_markdown(payload: dict[str, Any]) -> str:
    identity = payload.get("identity") if isinstance(payload.get("identity"), dict) else {}
    domains = payload.get("awareness_domains") if isinstance(payload.get("awareness_domains"), dict) else {}
    lines = [
        "# System Self Model",
        "",
        f"- Timestamp UTC: `{payload.get('timestamp_utc', '')}`",
        f"- Overall Status: `{payload.get('overall_status', '')}`",
        f"- Total Bots: `{identity.get('total_bots', '')}`",
        f"- Active Bots: `{identity.get('active_bots', '')}`",
        f"- Collection Bots: `{identity.get('data_collection_active_bots', '')}`",
        "",
        "## Awareness Domains",
        "",
    ]
    for name, row in domains.items():
        if not isinstance(row, dict):
            continue
        lines.append(f"- `{name}`: `{row.get('status', '')}`")
    lines.extend(["", "## Top Optimizations", ""])
    for row in payload.get("upgrades_and_optimizations") or []:
        if not isinstance(row, dict):
            continue
        implemented = " (implemented)" if row.get("implemented") else ""
        lines.append(f"- `{row.get('lane', '')}`{implemented}: {row.get('upgrade', '')}")
    lines.extend(["", "## Self Summary", "", str(payload.get("self_summary") or "")])
    return "\n".join(lines) + "\n"


def _render_self_brief(payload: dict[str, Any]) -> str:
    identity = payload.get("identity") if isinstance(payload.get("identity"), dict) else {}
    domains = payload.get("awareness_domains") if isinstance(payload.get("awareness_domains"), dict) else {}
    surface_matrix = payload.get("surface_matrix") if isinstance(payload.get("surface_matrix"), dict) else {}
    dependency_memory = payload.get("dependency_memory") if isinstance(payload.get("dependency_memory"), dict) else {}
    failure_index = payload.get("failure_memory_index") if isinstance(payload.get("failure_memory_index"), dict) else {}
    registry_diff = payload.get("registry_diff_memory") if isinstance(payload.get("registry_diff_memory"), dict) else {}
    optimizer = payload.get("upgrade_optimizer") if isinstance(payload.get("upgrade_optimizer"), dict) else {}

    blocked = [
        name
        for name, row in surface_matrix.items()
        if isinstance(row, dict) and str(row.get("status") or "") == "blocked"
    ]
    degraded = [
        name
        for name, row in surface_matrix.items()
        if isinstance(row, dict) and str(row.get("status") or "") in {"degraded", "needs_work"}
    ]
    top_actions = optimizer.get("top_next_actions") if isinstance(optimizer.get("top_next_actions"), list) else []
    failure_awareness = domains.get("failure_memory") if isinstance(domains.get("failure_memory"), dict) else {}
    global_halt_active = bool(failure_awareness.get("global_halt_active", False))
    lines = [
        "# System Self Brief",
        "",
        f"Generated UTC: `{payload.get('timestamp_utc', '')}`",
        "",
        "## Posture",
        "",
        (
            f"The platform brain sees `{identity.get('active_bots', '')}` active bots and "
            f"`{identity.get('data_collection_active_bots', '')}` collection-active bots across "
            f"`{identity.get('sleeve_profile_count', '')}` sleeve profiles. "
            f"Overall self-model status is `{payload.get('overall_status', '')}`."
        ),
        "",
        "## What Is Stable",
        "",
        f"- Memory/resource guard: `{((domains.get('resource_awareness') or {}).get('memory_guard_status') or '')}`",
        f"- Host pressure intelligence: `{((domains.get('host_pressure_intelligence') or {}).get('status') or '')}` cpu `{((domains.get('host_pressure_intelligence') or {}).get('cpu_pressure_level') or '')}` memory `{((domains.get('host_pressure_intelligence') or {}).get('memory_pressure_level') or '')}`",
        f"- MLX intelligence router: `{((domains.get('mlx_intelligence_awareness') or {}).get('status') or '')}`",
        f"- Non-MLX library router: `{((domains.get('library_utilization_awareness') or {}).get('status') or '')}`",
        f"- Drainer intelligence: `{((domains.get('drainer_intelligence') or {}).get('status') or '')}` active lane `{((domains.get('drainer_intelligence') or {}).get('active_drainer') or 'none')}`",
        f"- Whole-system brain: `{((domains.get('whole_system_intelligence') or {}).get('status') or '')}` action `{((domains.get('whole_system_intelligence') or {}).get('action') or 'none')}`",
        f"- Self-intelligence: `{((domains.get('system_self_intelligence') or {}).get('status') or '')}` reflex `{((domains.get('system_self_intelligence') or {}).get('reflex_action') or 'none')}` uncertainty `{((domains.get('system_self_intelligence') or {}).get('uncertainty_level') or '')}` root `{((domains.get('system_self_intelligence') or {}).get('causal_root') or 'none')}` effect `{((domains.get('system_self_intelligence') or {}).get('action_effect_verdict') or 'none')}` route `{((domains.get('system_self_intelligence') or {}).get('integration_route_mode') or 'none')}`",
        f"- Codex operator bridge: `{((domains.get('codex_operator_bridge') or {}).get('status') or '')}` needs `{((domains.get('codex_operator_bridge') or {}).get('needs_codex_count') or 0)}` paper day PnL `{((domains.get('codex_operator_bridge') or {}).get('paper_day_net_pnl') or 0.0)}` training batch `{((domains.get('codex_operator_bridge') or {}).get('training_recommended_batch_size') or 0)}`",
        f"- Core materialization: `{((domains.get('bot_awareness') or {}).get('materialization_status') or '')}`",
        f"- Global halt active: `{global_halt_active}`",
        f"- Registry diff memory: `{registry_diff.get('diff_status', '')}`",
        "",
        "## What Needs Attention",
        "",
        f"- Blocked surfaces: `{', '.join(blocked) if blocked else 'none'}`",
        f"- Degraded surfaces: `{', '.join(degraded) if degraded else 'none'}`",
        f"- Dependency memory status: `{dependency_memory.get('overall_status', '')}` with `{dependency_memory.get('stale_source_count', 0)}` stale watched sources",
        f"- Failure memory status: `{failure_index.get('overall_status', '')}` with `{failure_index.get('active_risk_event_count', 0)}` active risk events",
        "",
        "## Next Optimizations",
        "",
    ]
    if top_actions:
        for action in top_actions[:6]:
            lines.append(f"- {action}")
    else:
        lines.append("- No triggered next-generation optimization right now.")
    lines.extend(
        [
            "",
            "## Control Note",
            "",
            "This is an operational self-model: it observes, explains, remembers, and optimizes platform state without making consciousness claims.",
        ]
    )
    return "\n".join(lines) + "\n"


def _public_payload(payload: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in payload.items() if not str(key).startswith("_")}


def build_payload(project_root: Path = PROJECT_ROOT) -> dict[str, Any]:
    now = datetime.now(timezone.utc)
    health_root = project_root / "governance" / "health"
    registry = _load_json(project_root / "master_bot_registry.json")
    cockpit = _load_json(health_root / "operator_cockpit_latest.json")
    memory = _load_json(health_root / "memory_efficiency_control_latest.json")
    throttle = _load_json(health_root / "runtime_throttle_control_latest.json")
    storage = _load_json(health_root / "ingestion_storage_control_latest.json")
    drainer_fleet = _load_json(health_root / "backpressure_drainer_fleet_latest.json")
    super_drainer = _load_json(health_root / "backpressure_super_drainer_latest.json")
    writer_cycle = _load_json(health_root / "writer_cycle_coordinator_latest.json")
    writer_process = _load_json(health_root / "writer_process_intelligence_latest.json")
    whole_system = _load_json(health_root / "whole_system_intelligence_latest.json")
    system_self_intelligence = _load_json(health_root / "system_self_intelligence_latest.json")
    codex_operator_bridge = _load_json(health_root / "codex_operator_bridge_latest.json")
    storage_autopilot = _load_json(health_root / "storage_backpressure_autopilot_latest.json")
    mlx_router = _load_json(health_root / "mlx_intelligence_router_latest.json")
    library_router = _load_json(health_root / "library_utilization_router_latest.json")
    global_halt = _load_json(health_root / "global_killswitch_latest.json")
    process_watchdog = _load_json(health_root / "process_watchdog_latest.json")
    process_fanout = _load_json(health_root / "process_fanout_guard_latest.json")
    auth_lease = _load_json(health_root / "auth_lease_manager_latest.json")
    data_plane = _load_json(health_root / "data_plane_recovery_controller_latest.json")
    live_runtime = _load_json(health_root / "live_runtime_separation_control_latest.json")
    use_mode = _load_json(health_root / "use_mode_compliance_guard_latest.json")
    incident = _load_json(project_root / "governance" / "alerts" / "incident_auto_halt_latest.json")
    core_materialization = _load_json(health_root / "core_bot_materialization_guard_latest.json")
    tripwire = _load_json(health_root / "shadow_watchdog_tripwire_latest.json")
    previous_dependency_memory = _load_json(health_root / "system_dependency_memory_latest.json")
    previous_failure_memory = _load_json(health_root / "system_failure_memory_latest.json")
    previous_registry_diff = _load_json(health_root / "system_registry_diff_latest.json")

    identity = _registry_identity(registry)
    surface_matrix = _surface_matrix(health_root, project_root, now=now)
    dependency_memory = _dependency_memory(surface_matrix, previous_dependency_memory, now=now)
    failure_index = _failure_memory_index(
        global_halt=global_halt,
        incident=incident,
        cockpit=cockpit,
        storage=storage,
        throttle=throttle,
        tripwire=tripwire,
        previous=previous_failure_memory,
        now=now,
    )
    registry_diff_full = _registry_diff_memory(registry, previous_registry_diff, now=now)
    registry_diff = _compact_registry_diff_memory(registry_diff_full)
    domains = {
        "resource_awareness": _resource_awareness(memory, throttle, storage),
        "host_pressure_intelligence": _host_pressure_intelligence(memory, throttle, mlx_router, library_router),
        "mlx_intelligence_awareness": _mlx_intelligence_awareness(mlx_router),
        "library_utilization_awareness": _library_utilization_awareness(library_router),
        "drainer_intelligence": _drainer_intelligence_awareness(drainer_fleet, super_drainer, writer_cycle, storage_autopilot),
        "writer_process_intelligence": _writer_process_awareness(writer_process, writer_cycle, process_watchdog, process_fanout),
        "whole_system_intelligence": _whole_system_intelligence_awareness(whole_system),
        "system_self_intelligence": _system_self_intelligence_awareness(system_self_intelligence),
        "codex_operator_bridge": _codex_operator_bridge_awareness(codex_operator_bridge),
        "bot_awareness": _bot_awareness(identity, core_materialization),
        "failure_memory": _failure_memory(global_halt, incident, cockpit),
        "halt_recovery_intelligence": _halt_recovery_intelligence(global_halt, process_watchdog, auth_lease, data_plane, live_runtime, storage),
        "dependency_awareness": _dependency_awareness(surface_matrix, cockpit),
        "growth_awareness": _growth_awareness(identity, memory, cockpit),
        "use_mode_compliance": _use_mode_compliance_awareness(use_mode),
        "self_reporting": _self_reporting_awareness(cockpit, surface_matrix),
    }
    domain_statuses = [str(row.get("status") or "missing") for row in domains.values()]
    worst = _worst_status(domain_statuses)
    overall_status = "ready" if worst in {"ready", "advisory"} else "degraded"
    if worst == "blocked":
        overall_status = "blocked"

    implementation_flags = _implementation_flags(project_root)
    optimization_plan = _optimization_plan(domains, surface_matrix, implementation_flags=implementation_flags)
    blocked_or_degraded = [
        name
        for name, row in surface_matrix.items()
        if str(row.get("status") or "") in {"blocked", "degraded", "needs_work"}
    ]
    self_summary = (
        f"System self-model sees {identity['active_bots']} active bots, "
        f"{identity['data_collection_active_bots']} collection-active bots, "
        f"resource mode {domains['resource_awareness']['status']}, "
        f"host pressure mode {domains['host_pressure_intelligence']['status']} "
        f"(cpu={domains['host_pressure_intelligence']['cpu_pressure_level']} "
        f"memory={domains['host_pressure_intelligence']['memory_pressure_level']}), "
        f"MLX intelligence mode {domains['mlx_intelligence_awareness']['status']}, "
        f"library utilization mode {domains['library_utilization_awareness']['status']}, "
        f"drainer intelligence mode {domains['drainer_intelligence']['status']} "
        f"with active lane {domains['drainer_intelligence']['active_drainer'] or 'none'}, "
        f"writer process mode {domains['writer_process_intelligence']['status']} "
        f"action {domains['writer_process_intelligence']['action'] or 'none'}, "
        f"whole-system brain mode {domains['whole_system_intelligence']['status']} "
        f"action {domains['whole_system_intelligence']['action'] or 'none'}, "
        f"self-intelligence mode {domains['system_self_intelligence']['status']} "
        f"reflex {domains['system_self_intelligence']['reflex_action'] or 'none'} "
        f"root {domains['system_self_intelligence']['causal_root'] or 'none'} "
        f"effect {domains['system_self_intelligence']['action_effect_verdict'] or 'none'} "
        f"route {domains['system_self_intelligence']['integration_route_mode'] or 'none'}, "
        f"Codex bridge mode {domains['codex_operator_bridge']['status']} "
        f"needs={domains['codex_operator_bridge']['needs_codex_count']} "
        f"trade_day_pnl={domains['codex_operator_bridge']['paper_day_net_pnl']:.2f}, "
        f"halt recovery mode {domains['halt_recovery_intelligence']['status']} "
        f"next={ ' '.join(domains['halt_recovery_intelligence']['next_safe_command']) if domains['halt_recovery_intelligence'].get('next_safe_command') else 'none' }, "
        f"growth pressure {domains['growth_awareness']['pressure_level']}, "
        f"use-mode boundary {domains['use_mode_compliance']['status']} "
        f"mode={domains['use_mode_compliance']['use_mode']} "
        f"personal_grade={domains['use_mode_compliance']['personal_grade']}, "
        f"and {len(blocked_or_degraded)} blocked/degraded watched surfaces."
    )
    advanced_backlog = _advanced_upgrade_backlog(domains, dependency_memory, failure_index, registry_diff, surface_matrix)
    payload = {
        "timestamp_utc": now.isoformat(),
        "schema_version": 1,
        "self_model_version": SELF_MODEL_VERSION,
        "ok": overall_status == "ready",
        "overall_status": overall_status,
        "identity": identity,
        "awareness_domains": domains,
        "surface_matrix": surface_matrix,
        "dependency_memory": dependency_memory,
        "failure_memory_index": failure_index,
        "registry_diff_memory": registry_diff,
        "upgrades_and_optimizations": optimization_plan,
        "advanced_upgrade_backlog": advanced_backlog,
        "self_summary": self_summary,
        "control_contract": {
            "observes_itself": True,
            "explains_itself": True,
            "optimizes_itself": True,
            "consciousness_claim": "none_operational_self_model_only",
            "grandmaster_packet": "compressed_self_state_for_safe_routing_throttles_growth_and_reporting",
            "platform_brain_mode": "big_platform_brain_operational_control_plane",
            "memory_surfaces": [
                "dependency_memory",
                "failure_memory_index",
                "halt_recovery_intelligence",
                "registry_diff_memory",
                "upgrade_optimizer",
                "self_brief",
                "host_pressure_intelligence",
                "mlx_intelligence_router",
                "library_utilization_router",
                "drainer_intelligence",
                "writer_process_intelligence",
                "whole_system_intelligence",
                "system_signal_bus",
                "system_brain",
                "system_process_contracts",
                "system_self_intelligence",
                "system_self_intelligence_memory",
                "codex_handoff",
                "codex_operator_bridge",
                "backpressure_super_drainer_memory",
                "capital_growth_intelligence",
                "capital_growth_awareness",
                "capital_rotation_control",
                "schwab_indicator_intelligence",
                "system_expansion_execution",
                "use_mode_compliance",
            ],
        },
        "source_files": {
            "registry": str(project_root / "master_bot_registry.json"),
            "operator_cockpit": str(health_root / "operator_cockpit_latest.json"),
            "memory_efficiency": str(health_root / "memory_efficiency_control_latest.json"),
            "runtime_throttle": str(health_root / "runtime_throttle_control_latest.json"),
            "ingestion_storage": str(health_root / "ingestion_storage_control_latest.json"),
            "backpressure_drainer_fleet": str(health_root / "backpressure_drainer_fleet_latest.json"),
            "backpressure_super_drainer": str(health_root / "backpressure_super_drainer_latest.json"),
            "backpressure_super_drainer_memory": str(health_root / "backpressure_super_drainer_memory_latest.json"),
            "writer_cycle_coordinator": str(health_root / "writer_cycle_coordinator_latest.json"),
            "writer_process_intelligence": str(health_root / "writer_process_intelligence_latest.json"),
            "whole_system_intelligence": str(health_root / "whole_system_intelligence_latest.json"),
            "system_signal_bus": str(health_root / "system_signal_bus_latest.json"),
            "system_brain": str(health_root / "system_brain_latest.json"),
            "system_process_contracts": str(health_root / "system_process_contracts_latest.json"),
            "system_self_intelligence": str(health_root / "system_self_intelligence_latest.json"),
            "system_self_intelligence_memory": str(project_root / "governance" / "system_intelligence" / "self_intelligence_memory.jsonl"),
            "codex_handoff": str(health_root / "codex_handoff_latest.json"),
            "codex_operator_bridge": str(health_root / "codex_operator_bridge_latest.json"),
            "storage_backpressure_autopilot": str(health_root / "storage_backpressure_autopilot_latest.json"),
            "mlx_runtime": str(health_root / "mlx_runtime_audit_latest.json"),
            "mlx_library": str(health_root / "mlx_library_upgrade_latest.json"),
            "mlx_intelligence_router": str(health_root / "mlx_intelligence_router_latest.json"),
            "library_utilization_router": str(health_root / "library_utilization_router_latest.json"),
            "quant_model_control": str(health_root / "quant_model_control_latest.json"),
            "capital_rotation_control": str(health_root / "capital_rotation_control_latest.json"),
            "schwab_indicator_intelligence": str(health_root / "schwab_indicator_intelligence_latest.json"),
            "system_expansion_execution": str(health_root / "system_expansion_execution_layer_latest.json"),
            "global_halt": str(health_root / "global_killswitch_latest.json"),
            "process_watchdog": str(health_root / "process_watchdog_latest.json"),
            "process_fanout_guard": str(health_root / "process_fanout_guard_latest.json"),
            "auth_lease_manager": str(health_root / "auth_lease_manager_latest.json"),
            "data_plane_recovery": str(health_root / "data_plane_recovery_controller_latest.json"),
            "live_runtime_separation": str(health_root / "live_runtime_separation_control_latest.json"),
            "use_mode_compliance": str(health_root / "use_mode_compliance_guard_latest.json"),
        },
        "_registry_diff_memory_full": registry_diff_full,
    }
    payload["upgrade_optimizer"] = _upgrade_optimizer_payload(payload, advanced_backlog, now=now)
    return payload


def write_outputs(
    payload: dict[str, Any],
    *,
    out_path: Path,
    markdown_path: Path,
    brief_path: Path | None = None,
    dependency_memory_path: Path | None = None,
    failure_memory_path: Path | None = None,
    registry_diff_path: Path | None = None,
    upgrade_plan_path: Path | None = None,
) -> None:
    public_payload = _public_payload(payload)
    _write_json(out_path, public_payload)
    report_outputs: dict[str, dict[str, str]] = {}
    report_outputs["markdown"] = _write_text_with_local_fallback(markdown_path, _render_markdown(public_payload))
    if brief_path is not None:
        report_outputs["brief"] = _write_text_with_local_fallback(brief_path, _render_self_brief(public_payload))
    if any(result.get("storage_mode") == "local_fallback" for result in report_outputs.values()):
        public_payload["report_outputs"] = report_outputs
        _write_json(out_path, public_payload)
    if dependency_memory_path is not None:
        _write_json(dependency_memory_path, payload.get("dependency_memory") if isinstance(payload.get("dependency_memory"), dict) else {})
    if failure_memory_path is not None:
        _write_json(failure_memory_path, payload.get("failure_memory_index") if isinstance(payload.get("failure_memory_index"), dict) else {})
    if registry_diff_path is not None:
        full_registry_diff = payload.get("_registry_diff_memory_full")
        _write_json(registry_diff_path, full_registry_diff if isinstance(full_registry_diff, dict) else payload.get("registry_diff_memory") if isinstance(payload.get("registry_diff_memory"), dict) else {})
    if upgrade_plan_path is not None:
        _write_json(upgrade_plan_path, payload.get("upgrade_optimizer") if isinstance(payload.get("upgrade_optimizer"), dict) else {})


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the operational self-model for the trading-bot platform.")
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--out-file", default=str(DEFAULT_OUT_PATH))
    parser.add_argument("--markdown-file", default=str(DEFAULT_MARKDOWN_PATH))
    parser.add_argument("--brief-file", default=str(DEFAULT_BRIEF_PATH))
    parser.add_argument("--dependency-memory-file", default=str(DEFAULT_DEPENDENCY_MEMORY_PATH))
    parser.add_argument("--failure-memory-file", default=str(DEFAULT_FAILURE_MEMORY_PATH))
    parser.add_argument("--registry-diff-file", default=str(DEFAULT_REGISTRY_DIFF_PATH))
    parser.add_argument("--upgrade-plan-file", default=str(DEFAULT_UPGRADE_PLAN_PATH))
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    payload = build_payload(project_root)
    write_outputs(
        payload,
        out_path=Path(args.out_file).expanduser(),
        markdown_path=Path(args.markdown_file).expanduser(),
        brief_path=Path(args.brief_file).expanduser(),
        dependency_memory_path=Path(args.dependency_memory_file).expanduser(),
        failure_memory_path=Path(args.failure_memory_file).expanduser(),
        registry_diff_path=Path(args.registry_diff_file).expanduser(),
        upgrade_plan_path=Path(args.upgrade_plan_file).expanduser(),
    )
    public_payload = _public_payload(payload)
    if args.json:
        print(json.dumps(public_payload, ensure_ascii=True))
    else:
        print(
            "system_self_model "
            f"status={public_payload['overall_status']} "
            f"active_bots={public_payload['identity']['active_bots']} "
            f"collection_bots={public_payload['identity']['data_collection_active_bots']}"
        )
    return 0 if public_payload.get("overall_status") in {"ready", "degraded"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
