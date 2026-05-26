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


DEFAULT_OUT_PATH = PROJECT_ROOT / "governance" / "health" / "mlx_intelligence_router_latest.json"
DEFAULT_EXTERNAL_CONTEXT_PATH = PROJECT_ROOT / "exports" / "external_context" / "mlx_intelligence_router_latest.json"
DEFAULT_MARKDOWN_PATH = PROJECT_ROOT / "exports" / "reports" / "operator" / "mlx_intelligence_router_latest.md"
DEFAULT_OVERRIDE_PATH = PROJECT_ROOT / "config" / ".env.mlx_intelligence_router_override"

REQUIRED_PACKAGES = (
    "mlx",
    "mlx-metal",
    "mlx-lm",
    "mlx-data",
    "mlx-graphs",
    "mlx-cluster",
    "mlx-snn",
    "mlx-vision",
    "mlx-vlm",
    "mlx-whisper",
    "mlx-audio",
    "mlx-embeddings",
    "mlx-embedding-models",
    "esig",
    "roughpy",
    "pyrecombine",
    "parakeet-mlx",
)

LANE_SPECS: tuple[dict[str, Any], ...] = (
    {
        "lane": "tensor_quant_core",
        "workload_family": "differentiable_pricing_simulation_and_training",
        "primary_libraries": ["mlx", "mlx-metal"],
        "optional_libraries": ["mlx-lm"],
        "library_hooks": ["mlx.compile", "mx.grad", "mlx.nn", "mlx.optimizers"],
        "targets": ["pricing_grad", "gpu_mc_sim", "kalman_parallel", "quant_model_control"],
        "priority": "protected_when_scheduled",
    },
    {
        "lane": "language_reasoning",
        "workload_family": "research_reasoning_strategy_narration_and_operator_briefs",
        "primary_libraries": ["mlx-lm"],
        "optional_libraries": ["mlx-embeddings"],
        "library_hooks": ["mlx_lm"],
        "targets": ["system_self_brief", "research_pipeline", "reporting_layer"],
        "priority": "throttle_first",
    },
    {
        "lane": "embedding_memory",
        "workload_family": "memory_retrieval_duplicate_alpha_detection_and_similarity_search",
        "primary_libraries": ["mlx-embeddings", "mlx-embedding-models"],
        "optional_libraries": ["mlx-lm"],
        "library_hooks": ["mlx_embeddings"],
        "targets": ["bot_similarity", "research_memory", "alpha_overlap"],
        "priority": "throttle_first",
    },
    {
        "lane": "graph_intelligence",
        "workload_family": "sleeve_dependency_graphs_cross_asset_spillovers_and_bot_lineage",
        "primary_libraries": ["mlx-graphs"],
        "optional_libraries": ["mlx-cluster"],
        "library_hooks": ["mlx_graphs"],
        "targets": ["dependency_memory", "sleeve_masters", "spillover_graphs"],
        "priority": "throttle_first",
    },
    {
        "lane": "audio_event_intelligence",
        "workload_family": "speech_transcription_macro_audio_and_event_capture",
        "primary_libraries": ["mlx-whisper"],
        "optional_libraries": ["mlx-audio", "parakeet-mlx"],
        "library_hooks": ["mlx_whisper", "mlx_audio"],
        "targets": ["macro_media_ingest", "cspan_fed_events", "live_macro_auto_watch"],
        "priority": "protected_if_live_event",
    },
    {
        "lane": "vision_vlm_intelligence",
        "workload_family": "chart_report_screenshot_and_multimodal_context_analysis",
        "primary_libraries": ["mlx-vision", "mlx-vlm"],
        "optional_libraries": ["mlx-embeddings"],
        "library_hooks": ["mlx_vision", "mlx_vlm"],
        "targets": ["report_quality", "framework_maps", "visual_anomaly_review"],
        "priority": "off_hours_preferred",
    },
    {
        "lane": "spiking_event_intelligence",
        "workload_family": "event_burst_spike_encoding_and_microstructure_reaction_research",
        "primary_libraries": ["mlx-snn"],
        "optional_libraries": ["mlx-data"],
        "library_hooks": ["mlxsnn"],
        "targets": ["microstructure", "event_burst_detection", "tail_event_replay"],
        "priority": "research_only",
    },
    {
        "lane": "data_pipeline",
        "workload_family": "batched_feature_loading_preprocessing_and_training_sample_feeds",
        "primary_libraries": ["mlx-data"],
        "optional_libraries": ["pyrecombine"],
        "library_hooks": ["mlx.data"],
        "targets": ["training_sample_quota", "feature_store", "collector_rollups"],
        "priority": "protected_when_training",
    },
    {
        "lane": "rough_path_signature",
        "workload_family": "signature_transforms_rough_volatility_and_path_dependent_features",
        "primary_libraries": ["esig", "roughpy"],
        "optional_libraries": ["mlx"],
        "library_hooks": ["esig", "roughpy"],
        "targets": ["signature_hawkes_generators", "rough_volatility", "path_dependent_vol"],
        "priority": "research_only",
    },
)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except Exception:
        return int(default)


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def _norm_package(name: str) -> str:
    return str(name or "").strip().lower().replace("_", "-")


def _status(payload: dict[str, Any], default: str = "missing") -> str:
    if not payload:
        return default
    explicit = str(payload.get("overall_status") or payload.get("status") or "").strip()
    if explicit:
        return explicit
    if "ok" in payload:
        return "ready" if bool(payload.get("ok", False)) else "blocked"
    return default


def _package_statuses(mlx_runtime: dict[str, Any], mlx_library: dict[str, Any]) -> dict[str, str]:
    statuses: dict[str, str] = {}
    rows = mlx_runtime.get("package_rows") if isinstance(mlx_runtime.get("package_rows"), list) else []
    for row in rows:
        if not isinstance(row, dict):
            continue
        package = _norm_package(str(row.get("package") or ""))
        if package:
            statuses[package] = str(row.get("status") or "missing_runtime")
    if not statuses:
        packages = mlx_library.get("packages") if isinstance(mlx_library.get("packages"), list) else []
        for row in packages:
            if not isinstance(row, dict):
                continue
            package = _norm_package(str(row.get("package") or ""))
            if package:
                statuses[package] = "ok"
    return statuses


def _available_packages(statuses: dict[str, str]) -> set[str]:
    return {package for package, status in statuses.items() if status in {"ok", "missing_lock"}}


def _coverage(statuses: dict[str, str]) -> dict[str, Any]:
    available = _available_packages(statuses)
    missing = [package for package in REQUIRED_PACKAGES if package not in available]
    covered = [package for package in REQUIRED_PACKAGES if package in available]
    return {
        "required_count": len(REQUIRED_PACKAGES),
        "covered_count": len(covered),
        "missing_count": len(missing),
        "coverage_ratio": round(len(covered) / max(len(REQUIRED_PACKAGES), 1), 4),
        "covered_packages": covered,
        "missing_packages": missing,
        "package_statuses": {package: statuses.get(package, "missing_runtime") for package in REQUIRED_PACKAGES},
    }


def _readiness_repair_plan(
    coverage: dict[str, Any],
    mlx_runtime: dict[str, Any],
    mlx_library: dict[str, Any],
    caps: dict[str, Any],
) -> dict[str, Any]:
    missing = [str(item) for item in (coverage.get("missing_packages") or [])]
    runtime_status = _status(mlx_runtime)
    library_status = _status(mlx_library)
    pcore_active = bool(caps.get("p_core_allocation_aware", False))
    coverage_ratio = _safe_float(coverage.get("coverage_ratio"), 0.0)
    if not missing and runtime_status not in {"missing", "blocked"}:
        status = "ready"
    elif runtime_status == "missing" or coverage_ratio <= 0.0:
        status = "audit_required"
    elif missing:
        status = "package_repair_required"
    else:
        status = "watch"
    commands = []
    if status in {"audit_required", "watch"}:
        commands.append(["./scripts/ops/opsctl.sh", "mlx-audit", "--json"])
    if missing or library_status == "blocked":
        commands.append(["./scripts/ops/opsctl.sh", "mlx-library-upgrade", "--json"])
        commands.append(["./scripts/ops/opsctl.sh", "mlx-intelligence-router", "--apply", "--json"])
    return {
        "status": status,
        "runtime_status": runtime_status,
        "library_status": library_status,
        "coverage_ratio": round(coverage_ratio, 4),
        "missing_count": len(missing),
        "missing_packages": missing,
        "pcore_safe_to_repair_now": not pcore_active,
        "repair_window_policy": "defer_package_installs_until_backlog_green_or_operator_approved" if pcore_active else "repair_now_if_needed",
        "recommended_commands": commands,
        "next_action": "run MLX audit first; package coverage is empty or stale"
        if status == "audit_required"
        else "repair missing MLX packages after backlog pressure cools"
        if missing and pcore_active
        else "repair missing MLX packages"
        if missing
        else "MLX readiness is clean",
    }


def _p_core_allocation_contract(throttle: dict[str, Any], storage: dict[str, Any]) -> dict[str, Any]:
    runtime_feedback = throttle.get("p_core_runtime_feedback") if isinstance(throttle.get("p_core_runtime_feedback"), dict) else {}
    backlog_relief = storage.get("backlog_relief_contract") if isinstance(storage.get("backlog_relief_contract"), dict) else {}
    storage_contract = (
        backlog_relief.get("p_core_backlog_allocation_contract")
        if isinstance(backlog_relief.get("p_core_backlog_allocation_contract"), dict)
        else {}
    )
    source = "runtime_throttle_control" if runtime_feedback else "ingestion_storage_control" if storage_contract else "missing"
    raw = runtime_feedback or storage_contract
    burst = raw.get("p_core_burst_intelligence") if isinstance(raw.get("p_core_burst_intelligence"), dict) else {}
    training_gate = raw.get("training_pcore_gate") if isinstance(raw.get("training_pcore_gate"), dict) else {}
    control_env = raw.get("control_env") if isinstance(raw.get("control_env"), dict) else {}
    mode = str(burst.get("mode") or control_env.get("BACKLOG_PCORE_BURST_MODE") or "").strip()
    workers = _safe_int(raw.get("preprocess_worker_budget"), _safe_int(burst.get("selected_workers"), 0))
    if workers <= 0:
        workers = _safe_int(control_env.get("BACKLOG_PCORE_PREPROCESS_WORKERS"), 0)
    memory_optimizer = (
        mode.startswith("memory_relief")
        or _truthy(control_env.get("BACKLOG_MEMORY_PRESSURE_CORE_OPTIMIZER"))
        or _truthy(raw.get("memory_pressure_core_optimizer"))
    )
    training_allowed = bool(training_gate.get("small_targeted_training_allowed_now", training_gate.get("allowed_now", True)))
    return {
        "active": _truthy(raw.get("active", False)),
        "source": source,
        "policy": str((raw.get("policy") or "p_core_preprocess_single_sql_writer") if raw else ""),
        "mode": mode,
        "preprocess_worker_budget": int(max(workers, 0)),
        "max_budget": _safe_int(burst.get("max_budget"), _safe_int(raw.get("p_core_count"), 0)),
        "memory_optimizer_active": bool(memory_optimizer),
        "training_gate_blocked": bool(training_gate and not training_allowed),
        "training_gate": training_gate,
        "headroom_policy": str(raw.get("headroom_policy") or raw.get("reserve_policy") or "reserve_foreground_first"),
        "reason": str(burst.get("reason") or control_env.get("BACKLOG_PCORE_BURST_REASON") or ""),
    }


def _runtime_caps(
    memory: dict[str, Any],
    throttle: dict[str, Any],
    mlx_runtime: dict[str, Any],
    p_core: dict[str, Any],
) -> dict[str, Any]:
    memory_snapshot = memory.get("memory_snapshot") if isinstance(memory.get("memory_snapshot"), dict) else {}
    cpu_snapshot = memory.get("cpu_snapshot") if isinstance(memory.get("cpu_snapshot"), dict) else {}
    cotenant = memory.get("cotenant_awareness") if isinstance(memory.get("cotenant_awareness"), dict) else {}
    throttle_profile = str(throttle.get("throttle_profile") or "observe")
    memory_level = str(throttle.get("memory_pressure_level") or "").strip().lower()
    if not memory_level:
        state = str(memory_snapshot.get("memory_pressure_state") or "").strip().lower()
        memory_level = "high" if state in {"red", "critical"} else "elevated" if state in {"yellow", "orange"} else "normal"
    host_saturation_score = _safe_float(throttle.get("host_saturation_score"), 0.0)
    cpu_level = str(throttle.get("cpu_pressure_level") or cpu_snapshot.get("cpu_pressure_level") or "").strip().lower()
    if not cpu_level:
        cpu_level = "high" if host_saturation_score >= 85.0 else "elevated" if host_saturation_score >= 65.0 else "watch" if host_saturation_score >= 45.0 else "normal"
    compile_available = bool(((mlx_runtime.get("runtime") or {}).get("compile_available", False)))
    compile_smoke_ok = bool(((mlx_runtime.get("runtime") or {}).get("compile_smoke_ok", False)))
    metal_available = bool(((mlx_runtime.get("runtime") or {}).get("metal_available", False)))
    cotenant_active = bool(cotenant.get("active", False) or cotenant.get("mode") in {"managed_cotenant", "guarded_cotenant", "pressure_aware_cotenant"})
    p_core_active = bool(p_core.get("active", False))
    p_core_mode = str(p_core.get("mode") or "").strip()
    p_core_workers = _safe_int(p_core.get("preprocess_worker_budget"), 0)
    p_core_memory_optimizer = bool(p_core.get("memory_optimizer_active", False))
    p_core_training_blocked = bool(p_core.get("training_gate_blocked", False))
    pressure_profile = "max_throughput"
    max_jobs = 3
    tensor_batch_cap = 64
    embedding_batch_cap = 128
    graph_node_cap = 12000
    audio_minutes_cap = 45
    heavy_vlm_enabled = True
    if throttle_profile == "protect_live" or memory_level == "high" or cpu_level == "high" or host_saturation_score >= 85.0:
        pressure_profile = "protect_live"
        max_jobs = 1
        tensor_batch_cap = 16
        embedding_batch_cap = 32
        graph_node_cap = 3000
        audio_minutes_cap = 12
        heavy_vlm_enabled = False
    elif throttle_profile == "sustain" or memory_level == "elevated" or cpu_level == "elevated" or host_saturation_score >= 65.0:
        pressure_profile = "sustain"
        max_jobs = 1
        tensor_batch_cap = 32
        embedding_batch_cap = 64
        graph_node_cap = 6000
        audio_minutes_cap = 20
        heavy_vlm_enabled = False
    elif throttle_profile == "soft_cap" or cotenant_active or cpu_level == "watch" or host_saturation_score >= 45.0:
        pressure_profile = "foreground_safe"
        max_jobs = 2
        tensor_batch_cap = 48
        embedding_batch_cap = 96
        graph_node_cap = 9000
        audio_minutes_cap = 30
        heavy_vlm_enabled = True
    p_core_coordination_policy = "not_active"
    if p_core_active:
        p_core_coordination_policy = "yield_to_backlog_p_core_contract"
        max_jobs = min(max_jobs, 1)
        heavy_vlm_enabled = False
        if p_core_mode == "memory_relief_2" or (p_core_memory_optimizer and p_core_workers <= 2):
            tensor_batch_cap = min(tensor_batch_cap, 8)
            embedding_batch_cap = min(embedding_batch_cap, 16)
            graph_node_cap = min(graph_node_cap, 1500)
            audio_minutes_cap = min(audio_minutes_cap, 8)
            p_core_coordination_policy = "memory_relief_yields_mlx_to_backlog_and_foreground_apps"
        elif p_core_mode == "memory_relief_3" or p_core_memory_optimizer:
            tensor_batch_cap = min(tensor_batch_cap, 12)
            embedding_batch_cap = min(embedding_batch_cap, 24)
            graph_node_cap = min(graph_node_cap, 2000)
            audio_minutes_cap = min(audio_minutes_cap, 10)
            p_core_coordination_policy = "memory_relief_yields_mlx_to_backlog_and_foreground_apps"
        elif p_core_mode in {"foreground_protect", "burst_6", "burst_7"} or p_core_workers >= 6:
            tensor_batch_cap = min(tensor_batch_cap, 16)
            embedding_batch_cap = min(embedding_batch_cap, 32)
            graph_node_cap = min(graph_node_cap, 3000)
            audio_minutes_cap = min(audio_minutes_cap, 12)
            p_core_coordination_policy = "backlog_burst_owns_p_cores_mlx_runs_light"
        elif p_core_mode == "daily_driver_5" or p_core_workers >= 5:
            tensor_batch_cap = min(tensor_batch_cap, 32)
            embedding_batch_cap = min(embedding_batch_cap, 64)
            graph_node_cap = min(graph_node_cap, 6000)
            audio_minutes_cap = min(audio_minutes_cap, 20)
            p_core_coordination_policy = "daily_backlog_driver_keeps_mlx_single_job"
    compile_mode = "canary_first" if compile_available and compile_smoke_ok and metal_available else "off"
    if p_core_active and (
        p_core_memory_optimizer
        or p_core_training_blocked
        or p_core_mode in {"foreground_protect", "burst_6", "burst_7", "memory_relief_2", "memory_relief_3"}
    ):
        compile_mode = "off"
    return {
        "profile": pressure_profile,
        "throttle_profile": throttle_profile,
        "memory_pressure_level": memory_level,
        "cpu_pressure_level": cpu_level,
        "host_saturation_score": round(host_saturation_score, 3),
        "host_pressure_state": "backlog_p_core_reserved"
        if p_core_active
        else ("constrained" if pressure_profile in {"protect_live", "sustain"} else ("foreground_safe" if pressure_profile == "foreground_safe" else "clear")),
        "cotenant_active": cotenant_active,
        "open_app_count": _safe_int(cotenant.get("open_app_count"), 0),
        "co_running_level": str(cotenant.get("co_running_level") or ""),
        "max_concurrent_mlx_jobs": max_jobs,
        "tensor_batch_cap": tensor_batch_cap,
        "embedding_batch_cap": embedding_batch_cap,
        "graph_node_cap": graph_node_cap,
        "audio_minutes_per_job_cap": audio_minutes_cap,
        "heavy_vlm_enabled": heavy_vlm_enabled,
        "compile_mode": compile_mode,
        "compile_available": compile_available,
        "compile_smoke_ok": compile_smoke_ok,
        "metal_available": metal_available,
        "p_core_allocation_aware": p_core_active,
        "p_core_contract_source": str(p_core.get("source") or "missing"),
        "p_core_allocation_mode": p_core_mode,
        "p_core_preprocess_workers": p_core_workers,
        "p_core_max_budget": _safe_int(p_core.get("max_budget"), 0),
        "p_core_memory_optimizer_active": p_core_memory_optimizer,
        "p_core_training_gate_blocked": p_core_training_blocked,
        "p_core_coordination_policy": p_core_coordination_policy,
        "mlx_cpu_affinity_library_available": False,
        "mlx_core_spread_role": "gpu_tensor_compute_only_cpu_spread_is_managed_by_os_adapter_and_autonomic_governor",
        "p_core_reason": str(p_core.get("reason") or ""),
        "policy": "maximize_mlx_library_coverage_while_yielding_to_cpu_memory_and_backlog_p_core_contracts",
    }


def _mlx_reopen_controller(caps: dict[str, Any], coverage: dict[str, Any], p_core: dict[str, Any]) -> dict[str, Any]:
    profile = str(caps.get("profile") or "").strip()
    compile_mode = str(caps.get("compile_mode") or "off").strip()
    missing_count = _safe_int(coverage.get("missing_count"), 0)
    max_jobs = _safe_int(caps.get("max_concurrent_mlx_jobs"), 0)
    p_core_active = bool(caps.get("p_core_allocation_aware", False) or p_core.get("active", False))
    memory_optimizer = bool(caps.get("p_core_memory_optimizer_active", False) or p_core.get("memory_optimizer_active", False))
    if missing_count > 0:
        mode = "closed_package_repair_required"
        allowed = False
        next_stage = "repair_packages"
        reason = "MLX package coverage is incomplete"
    elif profile in {"protect_live", "sustain"}:
        mode = "closed_runtime_pressure"
        allowed = False
        next_stage = "wait_for_foreground_safe"
        reason = "runtime or memory pressure is elevated"
    elif p_core_active and memory_optimizer:
        mode = "single_light_job_memory_guard"
        allowed = max_jobs >= 1
        next_stage = "canary_first_after_memory_clear"
        reason = "backlog P-core memory guard owns CPU headroom; MLX may run only light single jobs"
    elif p_core_active:
        mode = "single_light_job_yielding_to_pcore"
        allowed = max_jobs >= 1
        next_stage = "canary_first_after_backlog_idle"
        reason = "backlog P-core contract is active, so MLX yields heavy work"
    elif compile_mode == "canary_first":
        mode = "canary_first_ready"
        allowed = True
        next_stage = "bounded_scale_after_canary_success"
        reason = "compile, Metal, package coverage, and host caps are clear"
    else:
        mode = "single_light_job_no_compile"
        allowed = max_jobs >= 1
        next_stage = "enable_compile_after_smoke_clear"
        reason = "MLX runtime is available but compile canary is not clear"
    return {
        "enabled": True,
        "mode": mode,
        "allowed": bool(allowed),
        "max_concurrent_jobs": int(max(max_jobs, 0)),
        "compile_mode": compile_mode,
        "next_stage": next_stage,
        "p_core_active": p_core_active,
        "memory_optimizer_active": memory_optimizer,
        "requires_runtime_profile": "foreground_safe_or_clear",
        "recommended_command": ["./scripts/ops/opsctl.sh", "mlx-intelligence-router", "--apply", "--json"],
        "reason": reason,
        "policy": "reopen_mlx_as_canary_first_after_runtime_pcore_and_memory_pressure_clear",
    }


def _lane_routes(statuses: dict[str, str], caps: dict[str, Any]) -> list[dict[str, Any]]:
    available = _available_packages(statuses)
    routes: list[dict[str, Any]] = []
    for spec in LANE_SPECS:
        primary = [_norm_package(item) for item in spec.get("primary_libraries", [])]
        optional = [_norm_package(item) for item in spec.get("optional_libraries", [])]
        missing_primary = [item for item in primary if item not in available]
        optional_available = [item for item in optional if item in available]
        status = "ready" if not missing_primary else "blocked"
        if status == "ready" and spec.get("lane") == "vision_vlm_intelligence" and not bool(caps.get("heavy_vlm_enabled", True)):
            status = "advisory"
        routes.append(
            {
                "lane": spec["lane"],
                "status": status,
                "workload_family": spec["workload_family"],
                "primary_libraries": primary,
                "optional_libraries": optional,
                "missing_primary_libraries": missing_primary,
                "optional_libraries_available": optional_available,
                "library_hooks": spec.get("library_hooks", []),
                "targets": spec.get("targets", []),
                "priority": spec.get("priority", "throttle_first"),
                "runtime_profile": caps.get("profile"),
            }
        )
    return routes


def _route_coverage(routes: list[dict[str, Any]]) -> dict[str, Any]:
    ready = [row for row in routes if str(row.get("status") or "") in {"ready", "advisory"}]
    blocked = [row for row in routes if str(row.get("status") or "") == "blocked"]
    return {
        "lane_count": len(routes),
        "ready_or_advisory_lane_count": len(ready),
        "blocked_lane_count": len(blocked),
        "route_coverage_ratio": round(len(ready) / max(len(routes), 1), 4),
        "blocked_lanes": [str(row.get("lane") or "") for row in blocked],
    }


def _library_utilization_matrix(routes: list[dict[str, Any]]) -> dict[str, Any]:
    mapped: dict[str, list[str]] = {package: [] for package in REQUIRED_PACKAGES}
    for route in routes:
        lane = str(route.get("lane") or "")
        for package in list(route.get("primary_libraries") or []) + list(route.get("optional_libraries") or []):
            normalized = _norm_package(package)
            if normalized in mapped:
                mapped[normalized].append(lane)
    unmapped = [package for package, lanes in mapped.items() if not lanes]
    return {
        "library_count": len(mapped),
        "mapped_library_count": len(mapped) - len(unmapped),
        "mapped_library_ratio": round((len(mapped) - len(unmapped)) / max(len(mapped), 1), 4),
        "unmapped_libraries": unmapped,
        "library_to_lanes": {package: sorted(set(lanes)) for package, lanes in mapped.items()},
        "utilization_goal": "100_percent_library_coverage_in_control_plane_not_100_percent_hardware_saturation",
    }


def _recommended_env(caps: dict[str, Any]) -> dict[str, str]:
    reopen = caps.get("mlx_reopen_controller") if isinstance(caps.get("mlx_reopen_controller"), dict) else {}
    return {
        "MLX_INTELLIGENCE_ROUTER_ENABLED": "1",
        "MLX_INTELLIGENCE_PROFILE": str(caps.get("profile") or "foreground_safe"),
        "MLX_INTELLIGENCE_MAX_CONCURRENT_JOBS": str(_safe_int(caps.get("max_concurrent_mlx_jobs"), 1)),
        "MLX_INTELLIGENCE_TENSOR_BATCH_CAP": str(_safe_int(caps.get("tensor_batch_cap"), 32)),
        "MLX_INTELLIGENCE_EMBED_BATCH_CAP": str(_safe_int(caps.get("embedding_batch_cap"), 64)),
        "MLX_INTELLIGENCE_GRAPH_NODE_CAP": str(_safe_int(caps.get("graph_node_cap"), 6000)),
        "MLX_INTELLIGENCE_AUDIO_MINUTES_CAP": str(_safe_int(caps.get("audio_minutes_per_job_cap"), 20)),
        "MLX_INTELLIGENCE_HEAVY_VLM_ENABLED": "1" if bool(caps.get("heavy_vlm_enabled", False)) else "0",
        "MLX_INTELLIGENCE_COMPILE_MODE": str(caps.get("compile_mode") or "off"),
        "MLX_INTELLIGENCE_SHARED_MEMORY_POLICY": "foreground_safe_unified_memory",
        "MLX_INTELLIGENCE_PCORE_AWARE": "1" if bool(caps.get("p_core_allocation_aware", False)) else "0",
        "MLX_INTELLIGENCE_PCORE_MODE": str(caps.get("p_core_allocation_mode") or ""),
        "MLX_INTELLIGENCE_PCORE_PREPROCESS_WORKERS": str(_safe_int(caps.get("p_core_preprocess_workers"), 0)),
        "MLX_INTELLIGENCE_PCORE_MEMORY_OPTIMIZER": "1" if bool(caps.get("p_core_memory_optimizer_active", False)) else "0",
        "MLX_INTELLIGENCE_PCORE_COORDINATION_POLICY": str(caps.get("p_core_coordination_policy") or "not_active"),
        "MLX_INTELLIGENCE_BACKLOG_HEADROOM_POLICY": "yield_to_backlog_p_core_workers_when_active",
        "MLX_INTELLIGENCE_REOPEN_MODE": str(reopen.get("mode") or "unknown"),
        "MLX_INTELLIGENCE_REOPEN_ALLOWED": "1" if bool(reopen.get("allowed", False)) else "0",
        "MLX_INTELLIGENCE_REOPEN_NEXT_STAGE": str(reopen.get("next_stage") or ""),
    }


def _write_env_override(path: Path, env: dict[str, str]) -> bool:
    lines = ["# Auto-managed by scripts/ops/mlx_intelligence_router.py"]
    for key, value in sorted(env.items()):
        safe_value = str(value).replace("'", "'\"'\"'")
        lines.append(f"{key}='{safe_value}'")
    content = "\n".join(lines) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    current = path.read_text(encoding="utf-8") if path.exists() else ""
    if current == content:
        return False
    path.write_text(content, encoding="utf-8")
    return True


def _recommended_actions(
    coverage: dict[str, Any],
    route_coverage: dict[str, Any],
    caps: dict[str, Any],
) -> list[str]:
    reopen = caps.get("mlx_reopen_controller") if isinstance(caps.get("mlx_reopen_controller"), dict) else {}
    return ordered_unique(
        [
            "route every MLX-capable intelligence job through mlx-intelligence-router before expanding the library set",
            "MLX can reopen through the canary-first lane"
            if str(reopen.get("mode") or "") == "canary_first_ready"
            else "keep MLX in the reopen controller's light/capped mode until runtime and P-core pressure clear",
            "keep mlx.compile canary-first until runtime-throttle and memory-efficiency both stay green"
            if str(caps.get("compile_mode") or "") == "canary_first"
            else "keep mlx.compile disabled for heavy jobs until the compile smoke and Metal checks are green",
            "treat 100 percent utilization as library coverage, not hardware saturation",
            "let the backlog P-core allocation own preprocessing and keep MLX to one light job until backlog turns green"
            if bool(caps.get("p_core_allocation_aware", False))
            else "",
            "thin VLM, long audio, graph, and simulation jobs while CPU or memory pressure is elevated"
            if not bool(caps.get("heavy_vlm_enabled", True)) or bool(caps.get("cotenant_active", False)) or str(caps.get("cpu_pressure_level") or "") in {"watch", "elevated", "high"}
            else "",
            "install or repair missing MLX packages before enabling the blocked lanes"
            if _safe_int(coverage.get("missing_count"), 0) or _safe_int(route_coverage.get("blocked_lane_count"), 0)
            else "",
            "./scripts/ops/opsctl.sh runtime-throttle --apply --json",
        ]
    )


def build_payload(project_root: Path = PROJECT_ROOT) -> dict[str, Any]:
    health_root = project_root / "governance" / "health"
    mlx_runtime = load_json(health_root / "mlx_runtime_audit_latest.json")
    mlx_library = load_json(health_root / "mlx_library_upgrade_latest.json")
    memory = load_json(health_root / "memory_efficiency_control_latest.json")
    throttle = load_json(health_root / "runtime_throttle_control_latest.json")
    storage = load_json(health_root / "ingestion_storage_control_latest.json")
    quant = load_json(health_root / "quant_model_control_latest.json")
    statuses = _package_statuses(mlx_runtime, mlx_library)
    coverage = _coverage(statuses)
    p_core_contract = _p_core_allocation_contract(throttle, storage)
    caps = _runtime_caps(memory, throttle, mlx_runtime, p_core_contract)
    mlx_reopen = _mlx_reopen_controller(caps, coverage, p_core_contract)
    caps["mlx_reopen_controller"] = mlx_reopen
    readiness_repair = _readiness_repair_plan(coverage, mlx_runtime, mlx_library, caps)
    routes = _lane_routes(statuses, caps)
    route_coverage = _route_coverage(routes)
    library_matrix = _library_utilization_matrix(routes)
    env = _recommended_env(caps)
    missing_count = _safe_int(coverage.get("missing_count"), 0)
    blocked_lane_count = _safe_int(route_coverage.get("blocked_lane_count"), 0)
    runtime_verified = bool(mlx_runtime.get("ok")) and missing_count == 0
    runtime_status = _status(mlx_runtime)
    library_status = _status(mlx_library)
    status = "ready"
    if runtime_status == "blocked" or missing_count:
        status = "blocked"
    elif library_status == "blocked" and not runtime_verified:
        status = "blocked"
    elif blocked_lane_count:
        status = "degraded"
    elif str(caps.get("profile") or "") in {"foreground_safe", "sustain", "protect_live"} or bool(caps.get("p_core_allocation_aware", False)):
        status = "advisory"
    return {
        "timestamp_utc": iso_now(),
        "schema_version": 1,
        "ok": status in {"ready", "advisory"},
        "overall_status": status,
        "library_coverage": coverage,
        "readiness_repair_plan": readiness_repair,
        "route_coverage": route_coverage,
        "runtime_caps": caps,
        "mlx_reopen_controller": mlx_reopen,
        "recommended_runtime_env": env,
        "workload_routes": routes,
        "library_utilization_matrix": library_matrix,
        "quant_model_status": _status(quant),
        "control_contract": {
            "uses_all_available_mlx_libraries": bool(library_matrix.get("mapped_library_ratio") == 1.0 and missing_count == 0),
            "hardware_saturation_goal": "no",
            "safe_utilization_goal": "100_percent_library_coverage_with_cpu_memory_aware_caps",
            "p_core_allocation_aware": bool(caps.get("p_core_allocation_aware", False)),
            "p_core_allocation_policy": str(caps.get("p_core_coordination_policy") or "not_active"),
            "p_core_contract_source": str(caps.get("p_core_contract_source") or "missing"),
            "mlx_cpu_affinity_library_available": False,
            "cpu_spread_owner": "os_adapter_layer_and_autonomic_resource_governor",
            "live_path_policy": "feature_enrichment_and_risk_context_only",
            "training_path_policy": "off_hours_or_runtime_throttle_cleared",
            "paper_path_policy": "respect_paper_trade_lock_and_runtime_caps",
            "mlx_reopen_mode": str(mlx_reopen.get("mode") or ""),
            "mlx_reopen_allowed": bool(mlx_reopen.get("allowed", False)),
        },
        "recommended_actions": ordered_unique([str(readiness_repair.get("next_action") or "")] + _recommended_actions(coverage, route_coverage, caps)),
        "artifact_paths": {
            "json": str(DEFAULT_OUT_PATH),
            "external_context": str(DEFAULT_EXTERNAL_CONTEXT_PATH),
            "markdown": str(DEFAULT_MARKDOWN_PATH),
            "env_override": str(DEFAULT_OVERRIDE_PATH),
        },
    }


def render_markdown(payload: dict[str, Any]) -> str:
    coverage = payload.get("library_coverage") if isinstance(payload.get("library_coverage"), dict) else {}
    route_coverage = payload.get("route_coverage") if isinstance(payload.get("route_coverage"), dict) else {}
    repair = payload.get("readiness_repair_plan") if isinstance(payload.get("readiness_repair_plan"), dict) else {}
    caps = payload.get("runtime_caps") if isinstance(payload.get("runtime_caps"), dict) else {}
    lines = [
        "# MLX Intelligence Router",
        "",
        f"Generated UTC: `{payload.get('timestamp_utc', '')}`",
        f"Overall status: `{payload.get('overall_status', '')}`",
        "",
        "## Coverage",
        "",
        f"- MLX package coverage: `{coverage.get('coverage_ratio', 0.0)}`",
        f"- Route coverage: `{route_coverage.get('route_coverage_ratio', 0.0)}`",
        f"- Missing packages: `{', '.join(coverage.get('missing_packages') or []) or 'none'}`",
        f"- Readiness repair: `{repair.get('status', '')}`",
        "",
        "## Runtime Caps",
        "",
        f"- Profile: `{caps.get('profile', '')}`",
        f"- Max concurrent MLX jobs: `{caps.get('max_concurrent_mlx_jobs', '')}`",
        f"- Tensor batch cap: `{caps.get('tensor_batch_cap', '')}`",
        f"- Embedding batch cap: `{caps.get('embedding_batch_cap', '')}`",
        f"- Graph node cap: `{caps.get('graph_node_cap', '')}`",
        f"- Heavy VLM enabled: `{caps.get('heavy_vlm_enabled', '')}`",
        f"- Compile mode: `{caps.get('compile_mode', '')}`",
        f"- P-core aware: `{caps.get('p_core_allocation_aware', '')}`",
        f"- P-core mode: `{caps.get('p_core_allocation_mode', '')}`",
        f"- P-core preprocess workers: `{caps.get('p_core_preprocess_workers', '')}`",
        f"- P-core policy: `{caps.get('p_core_coordination_policy', '')}`",
        "",
        "## Workload Routes",
        "",
    ]
    for route in payload.get("workload_routes") or []:
        if not isinstance(route, dict):
            continue
        lines.append(
            f"- `{route.get('lane', '')}`: `{route.get('status', '')}` via "
            f"`{', '.join(route.get('primary_libraries') or [])}`"
        )
    lines.extend(["", "## Recommended Actions", ""])
    for action in payload.get("recommended_actions") or []:
        lines.append(f"- {action}")
    return "\n".join(lines) + "\n"


def write_outputs(
    payload: dict[str, Any],
    *,
    out_path: Path = DEFAULT_OUT_PATH,
    external_context_path: Path = DEFAULT_EXTERNAL_CONTEXT_PATH,
    markdown_path: Path = DEFAULT_MARKDOWN_PATH,
    override_path: Path = DEFAULT_OVERRIDE_PATH,
    apply: bool = False,
) -> dict[str, Any]:
    apply_result = {"applied": False, "override_path": str(override_path), "override_changed": False}
    if apply:
        env = payload.get("recommended_runtime_env") if isinstance(payload.get("recommended_runtime_env"), dict) else {}
        apply_result = {
            "applied": True,
            "override_path": str(override_path),
            "override_changed": _write_env_override(override_path, {str(k): str(v) for k, v in env.items()}),
            "env_override_count": len(env),
        }
        payload["apply_result"] = apply_result
    else:
        payload["apply_result"] = apply_result
    write_payload(out_path, payload)
    write_payload(external_context_path, payload)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(render_markdown(payload), encoding="utf-8")
    return apply_result


def main() -> int:
    parser = argparse.ArgumentParser(description="Route MLX libraries into safe intelligence workloads with runtime caps.")
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--out-file", default=str(DEFAULT_OUT_PATH))
    parser.add_argument("--external-context-file", default=str(DEFAULT_EXTERNAL_CONTEXT_PATH))
    parser.add_argument("--markdown-file", default=str(DEFAULT_MARKDOWN_PATH))
    parser.add_argument("--override-file", default=str(DEFAULT_OVERRIDE_PATH))
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    payload = build_payload(project_root)
    write_outputs(
        payload,
        out_path=Path(args.out_file).expanduser(),
        external_context_path=Path(args.external_context_file).expanduser(),
        markdown_path=Path(args.markdown_file).expanduser(),
        override_path=Path(args.override_file).expanduser(),
        apply=args.apply,
    )
    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        coverage = payload.get("library_coverage") if isinstance(payload.get("library_coverage"), dict) else {}
        print(
            "mlx_intelligence_router "
            f"status={payload.get('overall_status', '')} "
            f"library_coverage={float(coverage.get('coverage_ratio', 0.0) or 0.0):.3f}"
        )
    return 0 if payload.get("overall_status") in {"ready", "advisory", "degraded"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
