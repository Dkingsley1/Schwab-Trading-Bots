#!/usr/bin/env python3
from __future__ import annotations

import argparse
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


STATUS_ORDER = {
    "missing": 0,
    "ready": 1,
    "advisory": 2,
    "thin": 2,
    "needs_work": 3,
    "degraded": 4,
    "blocked": 5,
    "critical": 6,
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
    explicit = str(payload.get("overall_status") or payload.get("status") or "").strip()
    if explicit:
        return explicit
    if "ok" in payload:
        return "ready" if bool(payload.get("ok", False)) else "blocked"
    return default


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
    paths = {
        "operator_cockpit": health_root / "operator_cockpit_latest.json",
        "memory_efficiency": health_root / "memory_efficiency_control_latest.json",
        "runtime_throttle": health_root / "runtime_throttle_control_latest.json",
        "ingestion_storage": health_root / "ingestion_storage_control_latest.json",
        "mlx_runtime": health_root / "mlx_runtime_audit_latest.json",
        "mlx_library": health_root / "mlx_library_upgrade_latest.json",
        "mlx_intelligence_router": health_root / "mlx_intelligence_router_latest.json",
        "library_utilization_router": health_root / "library_utilization_router_latest.json",
        "quant_model_control": health_root / "quant_model_control_latest.json",
        "global_halt": health_root / "global_killswitch_latest.json",
        "master_infra": health_root / "master_infrastructure_supervisor_latest.json",
        "artifact_freshness": health_root / "artifact_freshness_slo_latest.json",
        "training_quality": health_root / "training_quality_control_latest.json",
        "bot_quality": health_root / "bot_quality_autopilot_latest.json",
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
        timestamp, age_minutes = _payload_timestamp(payload, path, current) if payload else ("", None)
        matrix[name] = {
            "status": status,
            "path": str(path),
            "loaded": bool(payload),
            "timestamp_utc": timestamp,
            "age_minutes": age_minutes,
            "payload_sha256": _json_sha256(payload) if payload else "",
            "payload_hash_short": _json_sha256(payload)[:12] if payload else "",
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


def _dependency_edges() -> list[dict[str, str]]:
    return [
        {"from": "resource_guard", "to": "memory_efficiency", "reason": "memory and co-tenant context"},
        {"from": "memory_efficiency", "to": "runtime_throttle", "reason": "host profile and pressure caps"},
        {"from": "runtime_throttle", "to": "mlx_runtime", "reason": "shared CPU/GPU memory and MLX batch pressure"},
        {"from": "mlx_runtime", "to": "mlx_intelligence_router", "reason": "MLX package and runtime readiness"},
        {"from": "mlx_library", "to": "mlx_intelligence_router", "reason": "pinned MLX library bundle coverage"},
        {"from": "mlx_intelligence_router", "to": "quant_model_control", "reason": "MLX workload routing and runtime caps"},
        {"from": "runtime_throttle", "to": "library_utilization_router", "reason": "non-MLX library worker caps and backend defaults"},
        {"from": "library_utilization_router", "to": "operator_cockpit", "reason": "library lane coverage and runtime support posture"},
        {"from": "ingestion_storage", "to": "operator_cockpit", "reason": "backpressure readiness"},
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


def _surface_status(surface_matrix: dict[str, dict[str, Any]], name: str) -> str:
    aliases = {
        "resource_guard": "memory_efficiency",
        "grand_master": "operator_cockpit",
        "system_self_model": "system_self_model",
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
    ready_like = {"ready", "advisory", "thin", "steady_state", "applied_with_followups"}

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
                stale_sources.append(
                    {
                        "surface": name,
                        "age_minutes": round(float(age_minutes), 3),
                        "stale_limit_minutes": stale_limit,
                        "status": status,
                    }
                )

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
        "stale_sources": stale_sources,
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


def _implementation_flags(project_root: Path = PROJECT_ROOT) -> dict[str, bool]:
    return {
        "self_model_cadence": _opsctl_self_model_refresh_wired(project_root),
        "dependency_graph": True,
        "failure_memory": True,
        "resource_awareness": _runtime_throttle_cotenant_wired(project_root),
        "bot_awareness": True,
        "self_reporting": True,
        "mlx_compute_brain": _mlx_intelligence_router_wired(project_root),
        "library_utilization_brain": _library_utilization_router_wired(project_root),
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
            "lane": "resource_awareness",
            "priority": "high",
            "upgrade": "teach runtime-throttle to consume cotenant_awareness mode directly instead of inferring from status alone",
            "benefit": "keeps foreground apps smooth while preserving live collection and paper execution",
            "implemented": bool(implementation_flags.get("resource_awareness", False)),
        },
        {
            "rank": 5,
            "lane": "bot_awareness",
            "priority": "medium",
            "upgrade": "add a registry diff memory that records what changed between bot expansions",
            "benefit": "makes bot growth auditable and easier to explain in reports",
            "implemented": bool(implementation_flags.get("bot_awareness", False)),
        },
        {
            "rank": 6,
            "lane": "self_reporting",
            "priority": "medium",
            "upgrade": "generate a daily natural-language self-brief with posture, changes, blockers, and safe next commands",
            "benefit": "gives you a quick morning readout without opening every health file",
            "implemented": bool(implementation_flags.get("self_reporting", False)),
        },
        {
            "rank": 7,
            "lane": "mlx_compute_brain",
            "priority": "high",
            "upgrade": "route MLX language, embedding, graph, audio, VLM, SNN, data, rough-path, and quant workloads through one capped intelligence router",
            "benefit": "uses the installed MLX library stack broadly without letting shared-memory jobs starve collectors or foreground apps",
            "implemented": bool(implementation_flags.get("mlx_compute_brain", False)),
        },
        {
            "rank": 8,
            "lane": "library_utilization_brain",
            "priority": "high",
            "upgrade": "route non-MLX libraries through owner lanes while keeping MLX as the default live intelligence backend",
            "benefit": "turns the rest of the dependency stack into governed support lanes instead of idle or competing backends",
            "implemented": bool(implementation_flags.get("library_utilization_brain", False)),
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
            "lane": "self_healing_router",
            "upgrade": "map each blocked surface to its safest recovery command, required prechecks, and post-refresh verifier",
            "triggered": bool(blocked_or_degraded),
            "benefit": "turns cockpit red rows into bounded recovery playbooks instead of manual hunting",
        },
        {
            "rank": 5,
            "lane": "collector_utility_budget",
            "upgrade": "score each collector by freshness value, storage cost, downstream use, and overlap so low-value collectors thin first",
            "triggered": bool(active_bots >= 700),
            "benefit": "keeps data breadth high while reducing CPU, storage, and writer pressure",
        },
        {
            "rank": 6,
            "lane": "hot_path_storage_budget",
            "upgrade": "assign per-surface hot/warm/cold storage budgets and degrade report/explanation writes before trading-path writes",
            "triggered": bool("storage_tier_policy" in blocked_or_degraded or "artifact_freshness" in blocked_or_degraded),
            "benefit": "protects paper/live collection when reports, artifacts, or explainers grow too fast",
        },
        {
            "rank": 7,
            "lane": "stale_surface_autofix",
            "upgrade": "auto-refresh stale required surfaces, compare last-good hashes, then suppress stale-only blockers when the dependency chain is otherwise healthy",
            "triggered": bool(stale_sources),
            "benefit": "prevents stale dashboards from causing unnecessary halt pressure",
        },
        {
            "rank": 8,
            "lane": "grandmaster_safe_mode",
            "upgrade": "feed a compressed self-state packet into Grand Master routing so it can choose observe, sample, buffer, or pause per sleeve",
            "triggered": bool(active_bots >= 700),
            "benefit": "lets the brain downshift specific sleeves instead of using blunt global controls",
        },
        {
            "rank": 9,
            "lane": "registry_growth_governance",
            "upgrade": "require every new bot wave to emit expected storage, CPU, labels, training horizon, teacher lineage, and rollback metadata",
            "triggered": bool(registry_diff.get("fingerprint_changed") or registry_diff.get("diff_status") == "baseline"),
            "benefit": "keeps future expansion clean and auditable",
        },
        {
            "rank": 10,
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
        f"- MLX intelligence router: `{((domains.get('mlx_intelligence_awareness') or {}).get('status') or '')}`",
        f"- Non-MLX library router: `{((domains.get('library_utilization_awareness') or {}).get('status') or '')}`",
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
    mlx_router = _load_json(health_root / "mlx_intelligence_router_latest.json")
    library_router = _load_json(health_root / "library_utilization_router_latest.json")
    global_halt = _load_json(health_root / "global_killswitch_latest.json")
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
        "mlx_intelligence_awareness": _mlx_intelligence_awareness(mlx_router),
        "library_utilization_awareness": _library_utilization_awareness(library_router),
        "bot_awareness": _bot_awareness(identity, core_materialization),
        "failure_memory": _failure_memory(global_halt, incident, cockpit),
        "dependency_awareness": _dependency_awareness(surface_matrix, cockpit),
        "growth_awareness": _growth_awareness(identity, memory, cockpit),
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
        f"MLX intelligence mode {domains['mlx_intelligence_awareness']['status']}, "
        f"library utilization mode {domains['library_utilization_awareness']['status']}, "
        f"growth pressure {domains['growth_awareness']['pressure_level']}, "
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
                "registry_diff_memory",
                "upgrade_optimizer",
                "self_brief",
                "mlx_intelligence_router",
                "library_utilization_router",
            ],
        },
        "source_files": {
            "registry": str(project_root / "master_bot_registry.json"),
            "operator_cockpit": str(health_root / "operator_cockpit_latest.json"),
            "memory_efficiency": str(health_root / "memory_efficiency_control_latest.json"),
            "runtime_throttle": str(health_root / "runtime_throttle_control_latest.json"),
            "ingestion_storage": str(health_root / "ingestion_storage_control_latest.json"),
            "mlx_runtime": str(health_root / "mlx_runtime_audit_latest.json"),
            "mlx_library": str(health_root / "mlx_library_upgrade_latest.json"),
            "mlx_intelligence_router": str(health_root / "mlx_intelligence_router_latest.json"),
            "library_utilization_router": str(health_root / "library_utilization_router_latest.json"),
            "quant_model_control": str(health_root / "quant_model_control_latest.json"),
            "global_halt": str(health_root / "global_killswitch_latest.json"),
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
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(_render_markdown(public_payload), encoding="utf-8")
    if brief_path is not None:
        brief_path.parent.mkdir(parents=True, exist_ok=True)
        brief_path.write_text(_render_self_brief(public_payload), encoding="utf-8")
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
