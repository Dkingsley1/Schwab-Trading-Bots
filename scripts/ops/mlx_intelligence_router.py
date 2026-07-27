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
COMPATIBILITY_EXCLUDED_STATUS = "compatibility_excluded"
LANE_PRIORITY_BASE_SCORE = {
    "protected_when_scheduled": 86,
    "protected_if_live_event": 84,
    "protected_when_training": 78,
    "throttle_first": 64,
    "off_hours_preferred": 52,
    "research_only": 40,
}
LANE_CLASS_BASE_MEMORY_MB = {
    "tensor": 768,
    "language": 896,
    "embedding": 320,
    "graph": 640,
    "audio": 512,
    "vlm": 1536,
    "event_research": 256,
    "data": 448,
    "signature_research": 256,
    "general": 384,
}
LANE_CLASS_BASE_COOLDOWN_SECONDS = {
    "tensor": 45,
    "language": 75,
    "embedding": 15,
    "graph": 120,
    "audio": 30,
    "vlm": 180,
    "event_research": 120,
    "data": 90,
    "signature_research": 90,
    "general": 60,
}
LANE_CLASS_TOKEN_COST = {
    "tensor": 6,
    "language": 5,
    "embedding": 2,
    "graph": 5,
    "audio": 3,
    "vlm": 8,
    "event_research": 3,
    "data": 4,
    "signature_research": 3,
    "general": 4,
}

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
    return {
        package
        for package, status in statuses.items()
        if status in {"ok", "missing_lock", "runtime_ahead_of_lock"}
    }


def _coverage(statuses: dict[str, str]) -> dict[str, Any]:
    available = _available_packages(statuses)
    excluded = [package for package in REQUIRED_PACKAGES if statuses.get(package) == COMPATIBILITY_EXCLUDED_STATUS]
    active_required = [package for package in REQUIRED_PACKAGES if package not in set(excluded)]
    missing = [package for package in active_required if package not in available]
    covered = [package for package in active_required if package in available]
    return {
        "required_count": len(active_required),
        "original_required_count": len(REQUIRED_PACKAGES),
        "covered_count": len(covered),
        "missing_count": len(missing),
        "coverage_ratio": round(len(covered) / max(len(active_required), 1), 4),
        "covered_packages": covered,
        "missing_packages": missing,
        "compatibility_excluded_count": len(excluded),
        "compatibility_excluded_packages": excluded,
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
        "compatibility_excluded_count": _safe_int(coverage.get("compatibility_excluded_count"), 0),
        "compatibility_excluded_packages": list(coverage.get("compatibility_excluded_packages") or []),
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
    memory_free_default = 90.0 if memory_level in {"", "normal", "green"} else 0.0
    memory_free_pct = _safe_float(memory_snapshot.get("memory_free_pct"), memory_free_default)
    swap_used_gb = _safe_float(memory_snapshot.get("swap_used_gb"), 0.0)
    compressed_store_gb = _safe_float(memory_snapshot.get("compressed_store_gb"), 0.0)
    compressor_gb = _safe_float(memory_snapshot.get("compressor_gb"), 0.0)
    compressed_pressure_gb = max(compressed_store_gb, compressor_gb)
    if memory_level in {"high", "critical"} or swap_used_gb >= 6.0 or compressed_pressure_gb >= 18.0 or memory_free_pct < 18.0:
        mlx_memory_tier = "pressure"
    elif memory_level == "elevated" or swap_used_gb >= 3.0 or compressed_pressure_gb >= 10.0 or memory_free_pct < 35.0:
        mlx_memory_tier = "guarded"
    elif memory_free_pct >= 75.0 and swap_used_gb <= 1.0 and compressed_pressure_gb <= 6.0:
        mlx_memory_tier = "deep_green"
    else:
        mlx_memory_tier = "green"
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
    if mlx_memory_tier == "deep_green" and pressure_profile == "max_throughput":
        max_jobs = 4
        tensor_batch_cap = 96
        embedding_batch_cap = 192
        graph_node_cap = 16000
        audio_minutes_cap = 60
    elif mlx_memory_tier == "guarded":
        pressure_profile = "sustain" if pressure_profile == "max_throughput" else pressure_profile
        max_jobs = min(max_jobs, 1)
        tensor_batch_cap = min(tensor_batch_cap, 24)
        embedding_batch_cap = min(embedding_batch_cap, 48)
        graph_node_cap = min(graph_node_cap, 4500)
        audio_minutes_cap = min(audio_minutes_cap, 16)
        heavy_vlm_enabled = False
    elif mlx_memory_tier == "pressure":
        pressure_profile = "protect_live"
        max_jobs = 1
        tensor_batch_cap = min(tensor_batch_cap, 8)
        embedding_batch_cap = min(embedding_batch_cap, 16)
        graph_node_cap = min(graph_node_cap, 1500)
        audio_minutes_cap = min(audio_minutes_cap, 6)
        heavy_vlm_enabled = False
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
    compile_mode = "direct_stable" if compile_available and compile_smoke_ok and metal_available else "off"
    if p_core_active and (
        p_core_memory_optimizer
        or p_core_training_blocked
        or p_core_mode in {"foreground_protect", "burst_6", "burst_7", "memory_relief_2", "memory_relief_3"}
    ):
        compile_mode = "off"
    if mlx_memory_tier in {"guarded", "pressure"}:
        compile_mode = "off"
    return {
        "profile": pressure_profile,
        "throttle_profile": throttle_profile,
        "memory_pressure_level": memory_level,
        "mlx_memory_tier": mlx_memory_tier,
        "memory_free_pct": round(memory_free_pct, 3),
        "swap_used_gb": round(swap_used_gb, 3),
        "compressed_store_gb": round(compressed_store_gb, 3),
        "compressed_pressure_gb": round(compressed_pressure_gb, 3),
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
        next_stage = "direct_stable_after_memory_clear"
        reason = "backlog P-core memory guard owns CPU headroom; MLX may run only light single jobs"
    elif p_core_active:
        mode = "single_light_job_yielding_to_pcore"
        allowed = max_jobs >= 1
        next_stage = "direct_stable_after_backlog_idle"
        reason = "backlog P-core contract is active, so MLX yields heavy work"
    elif compile_mode == "direct_stable":
        mode = "direct_stable_ready"
        allowed = True
        next_stage = "bounded_scale_after_smoke_success"
        reason = "compile, Metal, package coverage, and host caps are clear"
    else:
        mode = "single_light_job_no_compile"
        allowed = max_jobs >= 1
        next_stage = "enable_compile_after_smoke_clear"
        reason = "MLX runtime is available but compile smoke is not clear"
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
        "policy": "reopen_mlx_direct_stable_after_runtime_pcore_and_memory_pressure_clear",
    }


def _lane_class(lane: str) -> str:
    return {
        "tensor_quant_core": "tensor",
        "language_reasoning": "language",
        "embedding_memory": "embedding",
        "graph_intelligence": "graph",
        "audio_event_intelligence": "audio",
        "vision_vlm_intelligence": "vlm",
        "spiking_event_intelligence": "event_research",
        "data_pipeline": "data",
        "rough_path_signature": "signature_research",
    }.get(lane, "general")


def _mlx_pressure_penalty(caps: dict[str, Any]) -> float:
    profile = str(caps.get("profile") or "foreground_safe")
    memory_tier = str(caps.get("mlx_memory_tier") or "green")
    host_saturation = _safe_float(caps.get("host_saturation_score"), 0.0)
    p_core_workers = _safe_int(caps.get("p_core_preprocess_workers"), 0)
    penalty = {
        "max_throughput": 0.0,
        "foreground_safe": 8.0,
        "sustain": 22.0,
        "protect_live": 42.0,
    }.get(profile, 10.0)
    penalty += {
        "deep_green": 0.0,
        "green": 5.0,
        "guarded": 24.0,
        "pressure": 48.0,
    }.get(memory_tier, 8.0)
    penalty += min(max(host_saturation - 45.0, 0.0) * 0.35, 24.0)
    if bool(caps.get("p_core_allocation_aware", False)):
        penalty += min(max(p_core_workers, 1) * 3.0, 24.0)
    return round(min(max(penalty, 0.0), 100.0), 3)


def _lane_memory_budget_mb(lane_class: str, allowed: bool, caps: dict[str, Any]) -> int:
    if not allowed:
        return 0
    profile = str(caps.get("profile") or "foreground_safe")
    memory_tier = str(caps.get("mlx_memory_tier") or "green")
    base = LANE_CLASS_BASE_MEMORY_MB.get(lane_class, LANE_CLASS_BASE_MEMORY_MB["general"])
    tier_multiplier = {
        "deep_green": 1.45,
        "green": 1.0,
        "guarded": 0.45,
        "pressure": 0.18,
    }.get(memory_tier, 0.75)
    profile_multiplier = {
        "max_throughput": 1.35,
        "foreground_safe": 1.0,
        "sustain": 0.55,
        "protect_live": 0.25,
    }.get(profile, 0.8)
    pcore_multiplier = 0.50 if bool(caps.get("p_core_allocation_aware", False)) else 1.0
    budget = int(round(base * tier_multiplier * profile_multiplier * pcore_multiplier))
    return max(96, budget)


def _lane_cooldown_seconds(lane_class: str, allowed: bool, caps: dict[str, Any]) -> int:
    base = LANE_CLASS_BASE_COOLDOWN_SECONDS.get(lane_class, LANE_CLASS_BASE_COOLDOWN_SECONDS["general"])
    profile = str(caps.get("profile") or "foreground_safe")
    memory_tier = str(caps.get("mlx_memory_tier") or "green")
    if profile == "max_throughput":
        base = max(base // 3, 5)
    elif profile == "foreground_safe":
        base = max(base // 2, 10)
    elif profile == "sustain":
        base += 45
    elif profile == "protect_live":
        base += 180
    if memory_tier == "guarded":
        base += 90
    elif memory_tier == "pressure":
        base += 300
    if bool(caps.get("p_core_allocation_aware", False)):
        base += min(max(_safe_int(caps.get("p_core_preprocess_workers"), 1), 1) * 12, 120)
    if not allowed:
        base = max(base, 300)
    return int(base)


def _lane_queue_tier(
    lane_class: str,
    priority: str,
    status: str,
    allowed: bool,
    compile_allowed: bool,
    caps: dict[str, Any],
) -> str:
    if status == "blocked":
        return "repair"
    if status == "excluded":
        return "compatibility_hold"
    if not allowed:
        return "cold_hold"
    if bool(caps.get("p_core_allocation_aware", False)):
        return "single_flight"
    if compile_allowed or priority in {"protected_when_scheduled", "protected_if_live_event"}:
        return "hot"
    if lane_class in {"embedding", "audio"}:
        return "warm"
    if priority in {"off_hours_preferred", "research_only"}:
        return "cold"
    return "warm"


def _lane_deadline_class(priority: str) -> str:
    return {
        "protected_if_live_event": "live_event",
        "protected_when_scheduled": "scheduled_compute",
        "protected_when_training": "training_window",
        "off_hours_preferred": "off_hours",
        "research_only": "research_backlog",
    }.get(priority, "best_effort")


def _lane_execution_window(priority: str, allowed: bool, caps: dict[str, Any]) -> str:
    if not allowed:
        return "hold_until_caps_reopen"
    if bool(caps.get("p_core_allocation_aware", False)):
        return "backlog_idle_or_single_flight"
    if priority == "protected_if_live_event":
        return "live_event_window"
    if priority in {"off_hours_preferred", "research_only"}:
        return "off_hours_or_deep_green"
    if str(caps.get("profile") or "") == "max_throughput":
        return "deep_green_parallel_window"
    return "foreground_safe_window"


def _lane_runtime_profile(lane: str, status: str, priority: str, caps: dict[str, Any]) -> dict[str, Any]:
    profile = str(caps.get("profile") or "foreground_safe")
    memory_tier = str(caps.get("mlx_memory_tier") or "green")
    compile_mode = str(caps.get("compile_mode") or "off")
    max_jobs = max(_safe_int(caps.get("max_concurrent_mlx_jobs"), 1), 0)
    lane_class = _lane_class(lane)
    base_batch = {
        "tensor": _safe_int(caps.get("tensor_batch_cap"), 32),
        "language": max(_safe_int(caps.get("tensor_batch_cap"), 32) // 2, 4),
        "embedding": _safe_int(caps.get("embedding_batch_cap"), 64),
        "graph": _safe_int(caps.get("graph_node_cap"), 6000),
        "audio": _safe_int(caps.get("audio_minutes_per_job_cap"), 20),
        "vlm": max(_safe_int(caps.get("tensor_batch_cap"), 32) // 4, 1),
        "event_research": max(_safe_int(caps.get("tensor_batch_cap"), 32) // 2, 4),
        "data": _safe_int(caps.get("embedding_batch_cap"), 64),
        "signature_research": max(_safe_int(caps.get("tensor_batch_cap"), 32) // 2, 4),
    }.get(lane_class, _safe_int(caps.get("tensor_batch_cap"), 32))
    if status == "blocked":
        allowed = False
        run_mode = "blocked_missing_primary_library"
    elif status == "excluded":
        allowed = False
        run_mode = "excluded_by_python314_compatibility"
    elif profile in {"protect_live", "sustain"} or memory_tier in {"guarded", "pressure"}:
        allowed = lane_class in {"embedding", "audio"} and max_jobs >= 1 and memory_tier != "pressure"
        run_mode = "micro_batch_only" if allowed else "paused_until_pressure_clears"
    elif lane_class == "vlm" and not bool(caps.get("heavy_vlm_enabled", False)):
        allowed = False
        run_mode = "paused_until_heavy_vlm_reopens"
    elif bool(caps.get("p_core_allocation_aware", False)):
        allowed = lane_class in {"tensor", "embedding", "audio"} and max_jobs >= 1
        run_mode = "single_light_job_yielding_to_pcore" if allowed else "paused_for_pcore_backlog_contract"
    else:
        allowed = max_jobs >= 1
        run_mode = "bounded_direct_stable" if compile_mode == "direct_stable" else "bounded_eager"
    precision = "fp16"
    if lane_class in {"language", "vlm"} and memory_tier in {"guarded", "pressure"}:
        precision = "int4_or_fp16_quantized"
    elif lane_class in {"embedding", "audio"} and profile in {"protect_live", "sustain"}:
        precision = "fp16_micro_batch"
    compile_allowed = bool(allowed and compile_mode == "direct_stable" and lane_class in {"tensor", "embedding", "signature_research"})
    pressure_penalty = _mlx_pressure_penalty(caps)
    priority_score = LANE_PRIORITY_BASE_SCORE.get(priority, 50)
    class_bonus = {
        "tensor": 5,
        "embedding": 6,
        "audio": 4,
        "language": 2,
        "signature_research": 1,
        "graph": 0,
        "data": 0,
        "vlm": -4,
        "event_research": -6,
    }.get(lane_class, 0)
    if not allowed:
        scheduler_score = 0
    else:
        scheduler_score = int(max(1, min(100, round(priority_score + class_bonus - pressure_penalty))))
    token_cost = LANE_CLASS_TOKEN_COST.get(lane_class, LANE_CLASS_TOKEN_COST["general"])
    if profile in {"sustain", "protect_live"} or memory_tier in {"guarded", "pressure"}:
        token_cost = max(1, token_cost - 1)
    if bool(caps.get("p_core_allocation_aware", False)):
        token_cost = max(1, token_cost - 1)
    queue_tier = _lane_queue_tier(lane_class, priority, status, allowed, compile_allowed, caps)
    memory_budget_mb = _lane_memory_budget_mb(lane_class, allowed, caps)
    cooldown_seconds = _lane_cooldown_seconds(lane_class, allowed, caps)
    if status == "blocked":
        admission_policy = "repair_primary_library_before_admission"
    elif status == "excluded":
        admission_policy = "hold_for_python_compatibility"
    elif not allowed:
        admission_policy = "hold_until_runtime_caps_reopen"
    elif queue_tier == "single_flight":
        admission_policy = "single_flight_admit_one_after_backlog_poll"
    elif profile in {"sustain", "protect_live"} or memory_tier in {"guarded", "pressure"}:
        admission_policy = "micro_batch_admission_with_cooldown"
    elif compile_allowed:
        admission_policy = "compiled_priority_admission"
    else:
        admission_policy = "bounded_eager_admission"
    return {
        "lane_class": lane_class,
        "allowed_now": bool(allowed),
        "run_mode": run_mode,
        "max_concurrent_jobs": 1 if allowed and bool(caps.get("p_core_allocation_aware", False)) else max_jobs if allowed else 0,
        "batch_or_size_cap": int(max(base_batch, 0)) if allowed else 0,
        "precision_policy": precision,
        "compile_allowed": compile_allowed,
        "memory_tier": memory_tier,
        "pressure_profile": profile,
        "scheduler_score": scheduler_score,
        "queue_tier": queue_tier,
        "token_cost": int(token_cost),
        "memory_budget_mb": memory_budget_mb,
        "cooldown_seconds": cooldown_seconds,
        "admission_policy": admission_policy,
        "deadline_class": _lane_deadline_class(priority),
        "preferred_execution_window": _lane_execution_window(priority, bool(allowed), caps),
        "spill_policy": "spill_to_cpu_or_defer" if memory_tier in {"guarded", "pressure"} else "keep_on_mlx_when_admitted",
        "prewarm_policy": "lazy_prewarm_only" if profile in {"foreground_safe", "sustain", "protect_live"} else "prewarm_when_idle",
        "cache_policy": "reuse_model_weights_avoid_duplicate_loads",
        "routing_reason": (
            "allowed_by_priority_queue"
            if allowed
            else "paused_by_runtime_pressure_or_missing_compatibility"
        ),
    }


def _lane_routes(statuses: dict[str, str], caps: dict[str, Any]) -> list[dict[str, Any]]:
    available = _available_packages(statuses)
    routes: list[dict[str, Any]] = []
    for spec in LANE_SPECS:
        primary = [_norm_package(item) for item in spec.get("primary_libraries", [])]
        optional = [_norm_package(item) for item in spec.get("optional_libraries", [])]
        excluded_primary = [item for item in primary if statuses.get(item) == COMPATIBILITY_EXCLUDED_STATUS]
        excluded_optional = [item for item in optional if statuses.get(item) == COMPATIBILITY_EXCLUDED_STATUS]
        missing_primary = [item for item in primary if item not in available and item not in excluded_primary]
        optional_available = [item for item in optional if item in available]
        status = "ready" if not missing_primary else "blocked"
        if status == "ready" and excluded_primary and len(excluded_primary) == len(primary):
            status = "excluded"
        if status == "ready" and spec.get("lane") == "vision_vlm_intelligence" and not bool(caps.get("heavy_vlm_enabled", True)):
            status = "advisory"
        runtime_profile = _lane_runtime_profile(
            str(spec["lane"]),
            status,
            str(spec.get("priority") or "throttle_first"),
            caps,
        )
        routes.append(
            {
                "lane": spec["lane"],
                "status": status,
                "workload_family": spec["workload_family"],
                "primary_libraries": primary,
                "optional_libraries": optional,
                "missing_primary_libraries": missing_primary,
                "compatibility_excluded_primary_libraries": excluded_primary,
                "compatibility_excluded_optional_libraries": excluded_optional,
                "optional_libraries_available": optional_available,
                "library_hooks": spec.get("library_hooks", []),
                "targets": spec.get("targets", []),
                "priority": spec.get("priority", "throttle_first"),
                "runtime_profile": caps.get("profile"),
                "optimization_profile": runtime_profile,
            }
        )
    return routes


def _route_coverage(routes: list[dict[str, Any]]) -> dict[str, Any]:
    ready = [row for row in routes if str(row.get("status") or "") in {"ready", "advisory"}]
    blocked = [row for row in routes if str(row.get("status") or "") == "blocked"]
    excluded = [row for row in routes if str(row.get("status") or "") == "excluded"]
    supported_lane_count = max(len(routes) - len(excluded), 0)
    return {
        "lane_count": len(routes),
        "supported_lane_count": supported_lane_count,
        "ready_or_advisory_lane_count": len(ready),
        "blocked_lane_count": len(blocked),
        "excluded_lane_count": len(excluded),
        "route_coverage_ratio": round(len(ready) / max(supported_lane_count, 1), 4),
        "blocked_lanes": [str(row.get("lane") or "") for row in blocked],
        "excluded_lanes": [str(row.get("lane") or "") for row in excluded],
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


def _lane_optimization_summary(routes: list[dict[str, Any]], caps: dict[str, Any]) -> dict[str, Any]:
    profiles = [
        row.get("optimization_profile")
        for row in routes
        if isinstance(row, dict) and isinstance(row.get("optimization_profile"), dict)
    ]
    allowed_pairs = [
        (str(route.get("lane") or ""), profile)
        for route in routes
        if isinstance(route, dict)
        and isinstance(route.get("optimization_profile"), dict)
        and bool((route.get("optimization_profile") or {}).get("allowed_now", False))
        for profile in [route.get("optimization_profile") or {}]
    ]
    allowed_pairs.sort(
        key=lambda item: (
            -_safe_int(item[1].get("scheduler_score"), 0),
            _safe_int(item[1].get("cooldown_seconds"), 9999),
            item[0],
        )
    )
    allowed = [profile for _, profile in allowed_pairs]
    compile_allowed = [row for row in profiles if bool(row.get("compile_allowed", False))]
    queue_tiers = {
        tier: sum(1 for row in profiles if str(row.get("queue_tier") or "") == tier)
        for tier in sorted({str(row.get("queue_tier") or "") for row in profiles if str(row.get("queue_tier") or "")})
    }
    allowed_lane_order = [lane for lane, _ in allowed_pairs]
    total_memory_budget_mb = sum(_safe_int(row.get("memory_budget_mb"), 0) for row in allowed)
    min_cooldown_seconds = min([_safe_int(row.get("cooldown_seconds"), 0) for row in allowed] or [0])
    profile = str(caps.get("profile") or "")
    memory_tier = str(caps.get("mlx_memory_tier") or "")
    if memory_tier == "pressure" or profile == "protect_live":
        scheduler_mode = "protective_hold"
    elif bool(caps.get("p_core_allocation_aware", False)):
        scheduler_mode = "pcore_yield_single_flight"
    elif profile == "sustain" or memory_tier == "guarded":
        scheduler_mode = "micro_batch_priority"
    elif str(caps.get("compile_mode") or "") == "direct_stable" and _safe_int(caps.get("max_concurrent_mlx_jobs"), 0) >= 3:
        scheduler_mode = "parallel_direct_stable"
    elif str(caps.get("compile_mode") or "") == "direct_stable":
        scheduler_mode = "bounded_direct_stable"
    else:
        scheduler_mode = "bounded_eager"
    token_budget = {
        "protective_hold": 0,
        "pcore_yield_single_flight": 3,
        "micro_batch_priority": 5,
        "parallel_direct_stable": 18,
        "bounded_direct_stable": 10,
        "bounded_eager": 7,
    }.get(scheduler_mode, 5)
    if str(caps.get("mlx_memory_tier") or "") == "guarded":
        token_budget = min(token_budget, 4)
    elif str(caps.get("mlx_memory_tier") or "") == "pressure":
        token_budget = 0
    spent_tokens = 0
    admitted_with_tokens: list[str] = []
    deferred_for_tokens: list[str] = []
    for lane, row in allowed_pairs:
        cost = max(_safe_int(row.get("token_cost"), 1), 1)
        if spent_tokens + cost <= token_budget:
            spent_tokens += cost
            admitted_with_tokens.append(lane)
        else:
            deferred_for_tokens.append(lane)
    memory_tier_multiplier = {
        "deep_green": 2.0,
        "green": 1.25,
        "guarded": 0.65,
        "pressure": 0.0,
    }.get(memory_tier, 1.0)
    cache_budget_mb = int(round(total_memory_budget_mb * memory_tier_multiplier))
    if bool(caps.get("p_core_allocation_aware", False)):
        cache_budget_mb = min(cache_budget_mb, 384)
    max_warm_lanes = min(len(admitted_with_tokens), 3 if scheduler_mode == "parallel_direct_stable" else 2 if scheduler_mode in {"bounded_direct_stable", "bounded_eager"} else 1)
    warm_lane_set = admitted_with_tokens[:max_warm_lanes]
    return {
        "profile_count": len(profiles),
        "allowed_lane_count": len(allowed),
        "paused_lane_count": len(profiles) - len(allowed),
        "compile_allowed_lane_count": len(compile_allowed),
        "scheduler_mode": scheduler_mode,
        "recommended_queue_order": allowed_lane_order,
        "total_memory_budget_mb": int(total_memory_budget_mb),
        "min_lane_cooldown_seconds": int(min_cooldown_seconds),
        "admission_token_budget": int(token_budget),
        "admission_token_spend": int(spent_tokens),
        "token_admitted_lanes": admitted_with_tokens,
        "token_deferred_lanes": deferred_for_tokens,
        "model_cache_budget_mb": int(cache_budget_mb),
        "max_warm_lanes": int(max_warm_lanes),
        "warm_lane_set": warm_lane_set,
        "queue_tier_counts": queue_tiers,
        "run_mode_counts": {
            mode: sum(1 for row in profiles if str(row.get("run_mode") or "") == mode)
            for mode in sorted({str(row.get("run_mode") or "") for row in profiles if str(row.get("run_mode") or "")})
        },
        "allowed_lane_classes": sorted({str(row.get("lane_class") or "") for row in allowed if str(row.get("lane_class") or "")}),
        "admission_policy": "score_ordered_single_flight" if scheduler_mode == "pcore_yield_single_flight" else "score_ordered_bounded_queue",
        "optimization_goal": "route_every_mlx_lane_with_explicit_memory_cpu_compile_and_batch_caps",
    }


def _adaptive_reopen_controller(
    caps: dict[str, Any],
    lane_optimization: dict[str, Any],
    previous_payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    previous_payload = previous_payload if isinstance(previous_payload, dict) else {}
    previous = (
        previous_payload.get("adaptive_reopen_controller")
        if isinstance(previous_payload.get("adaptive_reopen_controller"), dict)
        else {}
    )
    profile = str(caps.get("profile") or "")
    memory_tier = str(caps.get("mlx_memory_tier") or "")
    compile_mode = str(caps.get("compile_mode") or "off")
    scheduler_mode = str(lane_optimization.get("scheduler_mode") or "")
    pcore_active = bool(caps.get("p_core_allocation_aware", False))
    pressure_score = _mlx_pressure_penalty(caps)
    max_jobs = _safe_int(caps.get("max_concurrent_mlx_jobs"), 0)
    compile_ready = compile_mode == "direct_stable"
    clear_now = bool(
        memory_tier in {"deep_green", "green"}
        and profile in {"max_throughput", "foreground_safe"}
        and compile_ready
        and not pcore_active
        and pressure_score <= 18.0
    )
    pressure_now = bool(
        memory_tier in {"guarded", "pressure"}
        or profile in {"sustain", "protect_live"}
        or pcore_active
        or pressure_score >= 38.0
    )
    stable_green_windows = _safe_int(previous.get("stable_green_windows"), 0) + 1 if clear_now else 0
    pressure_windows = _safe_int(previous.get("pressure_windows"), 0) + 1 if pressure_now else 0
    clean_windows_required = 2
    if bool(previous.get("stage_changed", False)):
        clean_windows_required = 3
    if pcore_active or pressure_windows >= 2:
        clean_windows_required = 3
    if memory_tier == "deep_green" and profile == "max_throughput" and pressure_score <= 8.0:
        clean_windows_required = 2
    if memory_tier == "pressure" or profile == "protect_live":
        reopen_stage = "pressure_hold"
        reopen_allowed = False
        next_review_seconds = 120
    elif pcore_active:
        reopen_stage = "single_flight_pcore_hold"
        reopen_allowed = bool(_safe_int(lane_optimization.get("allowed_lane_count"), 0) > 0)
        next_review_seconds = max(_safe_int(lane_optimization.get("min_lane_cooldown_seconds"), 60), 60)
    elif profile == "sustain" or memory_tier == "guarded":
        reopen_stage = "micro_batch_watch"
        reopen_allowed = bool(_safe_int(lane_optimization.get("allowed_lane_count"), 0) > 0)
        next_review_seconds = max(_safe_int(lane_optimization.get("min_lane_cooldown_seconds"), 45), 45)
    elif clear_now and stable_green_windows >= clean_windows_required and scheduler_mode == "parallel_direct_stable" and max_jobs >= 3:
        reopen_stage = "parallel_direct_stable"
        reopen_allowed = True
        next_review_seconds = 30
    elif clear_now and stable_green_windows >= clean_windows_required and compile_ready:
        reopen_stage = "bounded_direct_stable"
        reopen_allowed = True
        next_review_seconds = 30
    elif clear_now:
        reopen_stage = "warming_direct_stable"
        reopen_allowed = False
        next_review_seconds = 30
    else:
        reopen_stage = "bounded_watch"
        reopen_allowed = bool(_safe_int(lane_optimization.get("allowed_lane_count"), 0) > 0)
        next_review_seconds = 60
    previous_stage = str(previous.get("reopen_stage") or "")
    stage_changed = bool(previous_stage and previous_stage != reopen_stage)
    return {
        "enabled": True,
        "pressure_score": round(float(pressure_score), 3),
        "reopen_stage": reopen_stage,
        "reopen_allowed": bool(reopen_allowed),
        "stable_green_windows": int(stable_green_windows),
        "pressure_windows": int(pressure_windows),
        "clean_windows_required": int(clean_windows_required),
        "stage_changed": stage_changed,
        "previous_reopen_stage": previous_stage,
        "next_review_seconds": int(next_review_seconds),
        "token_budget": _safe_int(lane_optimization.get("admission_token_budget"), 0),
        "token_spend": _safe_int(lane_optimization.get("admission_token_spend"), 0),
        "model_cache_budget_mb": _safe_int(lane_optimization.get("model_cache_budget_mb"), 0),
        "max_warm_lanes": _safe_int(lane_optimization.get("max_warm_lanes"), 0),
        "warm_lane_set": list(lane_optimization.get("warm_lane_set") or []),
        "policy": "hysteresis_gates_mlx_reopen_stage_after_stable_clean_windows",
    }


def _recommended_env(
    caps: dict[str, Any],
    lane_optimization: dict[str, Any] | None = None,
    adaptive_reopen: dict[str, Any] | None = None,
) -> dict[str, str]:
    reopen = caps.get("mlx_reopen_controller") if isinstance(caps.get("mlx_reopen_controller"), dict) else {}
    lane_optimization = lane_optimization if isinstance(lane_optimization, dict) else {}
    adaptive_reopen = adaptive_reopen if isinstance(adaptive_reopen, dict) else {}
    queue_order = [str(item) for item in (lane_optimization.get("recommended_queue_order") or []) if str(item)]
    allowed_classes = [str(item) for item in (lane_optimization.get("allowed_lane_classes") or []) if str(item)]
    return {
        "MLX_INTELLIGENCE_ROUTER_ENABLED": "1",
        "MLX_INTELLIGENCE_PROFILE": str(caps.get("profile") or "foreground_safe"),
        "MLX_INTELLIGENCE_MEMORY_TIER": str(caps.get("mlx_memory_tier") or "green"),
        "MLX_INTELLIGENCE_MEMORY_FREE_PCT": str(round(_safe_float(caps.get("memory_free_pct"), 0.0), 3)),
        "MLX_INTELLIGENCE_SWAP_USED_GB": str(round(_safe_float(caps.get("swap_used_gb"), 0.0), 3)),
        "MLX_INTELLIGENCE_COMPRESSED_PRESSURE_GB": str(round(_safe_float(caps.get("compressed_pressure_gb"), 0.0), 3)),
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
        "MLX_INTELLIGENCE_PREWARM_POLICY": "lazy_prewarm_only"
        if str(caps.get("profile") or "") in {"foreground_safe", "sustain", "protect_live"}
        else "prewarm_when_idle",
        "MLX_INTELLIGENCE_CACHE_POLICY": "reuse_model_weights_avoid_duplicate_loads",
        "MLX_INTELLIGENCE_SCHEDULER_MODE": str(lane_optimization.get("scheduler_mode") or "bounded_eager"),
        "MLX_INTELLIGENCE_ALLOWED_LANES": ",".join(queue_order),
        "MLX_INTELLIGENCE_ALLOWED_LANE_CLASSES": ",".join(allowed_classes),
        "MLX_INTELLIGENCE_TOTAL_MEMORY_BUDGET_MB": str(_safe_int(lane_optimization.get("total_memory_budget_mb"), 0)),
        "MLX_INTELLIGENCE_MIN_COOLDOWN_SECONDS": str(_safe_int(lane_optimization.get("min_lane_cooldown_seconds"), 0)),
        "MLX_INTELLIGENCE_ADMISSION_POLICY": str(lane_optimization.get("admission_policy") or "score_ordered_bounded_queue"),
        "MLX_INTELLIGENCE_QUEUE_POLICY": "score_ordered_pressure_aware_lane_queue",
        "MLX_INTELLIGENCE_MODEL_LOAD_POLICY": "single_shared_weight_cache_no_duplicate_lane_loads",
        "MLX_INTELLIGENCE_PRESSURE_SCORE": str(round(_safe_float(adaptive_reopen.get("pressure_score"), 0.0), 3)),
        "MLX_INTELLIGENCE_REOPEN_STAGE": str(adaptive_reopen.get("reopen_stage") or "bounded_watch"),
        "MLX_INTELLIGENCE_HYSTERESIS_ENABLED": "1" if bool(adaptive_reopen.get("enabled", False)) else "0",
        "MLX_INTELLIGENCE_REOPEN_ALLOWED_BY_HYSTERESIS": "1" if bool(adaptive_reopen.get("reopen_allowed", False)) else "0",
        "MLX_INTELLIGENCE_STABLE_GREEN_WINDOWS": str(_safe_int(adaptive_reopen.get("stable_green_windows"), 0)),
        "MLX_INTELLIGENCE_PRESSURE_WINDOWS": str(_safe_int(adaptive_reopen.get("pressure_windows"), 0)),
        "MLX_INTELLIGENCE_CLEAN_WINDOWS_REQUIRED": str(_safe_int(adaptive_reopen.get("clean_windows_required"), 0)),
        "MLX_INTELLIGENCE_NEXT_REVIEW_SECONDS": str(_safe_int(adaptive_reopen.get("next_review_seconds"), 60)),
        "MLX_INTELLIGENCE_TOKEN_BUDGET": str(_safe_int(adaptive_reopen.get("token_budget"), 0)),
        "MLX_INTELLIGENCE_TOKEN_SPEND": str(_safe_int(adaptive_reopen.get("token_spend"), 0)),
        "MLX_INTELLIGENCE_MODEL_CACHE_BUDGET_MB": str(_safe_int(adaptive_reopen.get("model_cache_budget_mb"), 0)),
        "MLX_INTELLIGENCE_MAX_WARM_LANES": str(_safe_int(adaptive_reopen.get("max_warm_lanes"), 0)),
        "MLX_INTELLIGENCE_WARM_LANES": ",".join(str(item) for item in (adaptive_reopen.get("warm_lane_set") or []) if str(item)),
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
    lane_optimization: dict[str, Any] | None = None,
    adaptive_reopen: dict[str, Any] | None = None,
) -> list[str]:
    reopen = caps.get("mlx_reopen_controller") if isinstance(caps.get("mlx_reopen_controller"), dict) else {}
    memory_tier = str(caps.get("mlx_memory_tier") or "")
    lane_optimization = lane_optimization if isinstance(lane_optimization, dict) else {}
    adaptive_reopen = adaptive_reopen if isinstance(adaptive_reopen, dict) else {}
    return ordered_unique(
        [
            "route every MLX-capable intelligence job through mlx-intelligence-router before expanding the library set",
            f"run MLX scheduler in {lane_optimization.get('scheduler_mode')} mode with score-ordered lane admission"
            if lane_optimization.get("scheduler_mode")
            else "",
            f"hold MLX reopen at {adaptive_reopen.get('reopen_stage')} until hysteresis confirms the next clean window"
            if adaptive_reopen.get("reopen_stage") and not bool(adaptive_reopen.get("reopen_allowed", False))
            else f"allow MLX reopen stage {adaptive_reopen.get('reopen_stage')} under token budget {adaptive_reopen.get('token_budget')}"
            if adaptive_reopen.get("reopen_stage")
            else "",
            "use deep-green unified-memory tier for bounded MLX prewarm and larger tensor/embedding batches"
            if memory_tier == "deep_green" and not bool(caps.get("p_core_allocation_aware", False))
            else "keep MLX micro-batched until unified-memory pressure returns to green"
            if memory_tier in {"guarded", "pressure"}
            else "",
            "MLX can reopen through the direct-stable lane"
            if str(reopen.get("mode") or "") == "direct_stable_ready"
            else "keep MLX in the reopen controller's light/capped mode until runtime and P-core pressure clear",
            "keep mlx.compile direct-stable but bounded until runtime-throttle and memory-efficiency both stay green"
            if str(caps.get("compile_mode") or "") == "direct_stable"
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
            "track Python 3.14 MLX compatibility exclusions without downgrading the active MLX runtime"
            if _safe_int(coverage.get("compatibility_excluded_count"), 0) or _safe_int(route_coverage.get("excluded_lane_count"), 0)
            else "",
            "./scripts/ops/opsctl.sh runtime-throttle --apply --json",
        ]
    )


def build_payload(project_root: Path = PROJECT_ROOT) -> dict[str, Any]:
    health_root = project_root / "governance" / "health"
    previous_payload = load_json(health_root / "mlx_intelligence_router_latest.json")
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
    lane_optimization = _lane_optimization_summary(routes, caps)
    adaptive_reopen = _adaptive_reopen_controller(caps, lane_optimization, previous_payload)
    env = _recommended_env(caps, lane_optimization, adaptive_reopen)
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
        "lane_optimization_summary": lane_optimization,
        "adaptive_reopen_controller": adaptive_reopen,
        "quant_model_status": _status(quant),
        "control_contract": {
            "uses_all_available_mlx_libraries": bool(library_matrix.get("mapped_library_ratio") == 1.0 and missing_count == 0),
            "compatibility_excluded_packages": list(coverage.get("compatibility_excluded_packages") or []),
            "compatibility_excluded_lanes": list(route_coverage.get("excluded_lanes") or []),
            "hardware_saturation_goal": "no",
            "safe_utilization_goal": "100_percent_library_coverage_with_cpu_memory_aware_caps",
            "p_core_allocation_aware": bool(caps.get("p_core_allocation_aware", False)),
            "p_core_allocation_policy": str(caps.get("p_core_coordination_policy") or "not_active"),
            "p_core_contract_source": str(caps.get("p_core_contract_source") or "missing"),
            "mlx_cpu_affinity_library_available": False,
            "cpu_spread_owner": "os_adapter_layer_and_autonomic_resource_governor",
            "unified_memory_tier": str(caps.get("mlx_memory_tier") or ""),
            "unified_memory_policy": "adaptive_caps_from_memory_free_swap_and_compressed_pressure",
            "lane_optimization_profiled": bool(lane_optimization.get("profile_count") == len(routes)),
            "allowed_lane_count": _safe_int(lane_optimization.get("allowed_lane_count"), 0),
            "mlx_scheduler_mode": str(lane_optimization.get("scheduler_mode") or ""),
            "mlx_scheduler_policy": "score_ordered_pressure_aware_lane_queue",
            "mlx_recommended_queue_order": list(lane_optimization.get("recommended_queue_order") or []),
            "mlx_total_memory_budget_mb": _safe_int(lane_optimization.get("total_memory_budget_mb"), 0),
            "mlx_min_lane_cooldown_seconds": _safe_int(lane_optimization.get("min_lane_cooldown_seconds"), 0),
            "mlx_admission_token_budget": _safe_int(lane_optimization.get("admission_token_budget"), 0),
            "mlx_admission_token_spend": _safe_int(lane_optimization.get("admission_token_spend"), 0),
            "mlx_model_cache_budget_mb": _safe_int(lane_optimization.get("model_cache_budget_mb"), 0),
            "mlx_reopen_stage": str(adaptive_reopen.get("reopen_stage") or ""),
            "mlx_reopen_hysteresis_policy": str(adaptive_reopen.get("policy") or ""),
            "mlx_reopen_clean_windows_required": _safe_int(adaptive_reopen.get("clean_windows_required"), 0),
            "live_path_policy": "feature_enrichment_and_risk_context_only",
            "training_path_policy": "off_hours_or_runtime_throttle_cleared",
            "paper_path_policy": "respect_paper_trade_lock_and_runtime_caps",
            "mlx_reopen_mode": str(mlx_reopen.get("mode") or ""),
            "mlx_reopen_allowed": bool(mlx_reopen.get("allowed", False)),
        },
        "recommended_actions": ordered_unique(
            [str(readiness_repair.get("next_action") or "")]
            + _recommended_actions(coverage, route_coverage, caps, lane_optimization, adaptive_reopen)
        ),
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
    lane_summary = payload.get("lane_optimization_summary") if isinstance(payload.get("lane_optimization_summary"), dict) else {}
    adaptive = payload.get("adaptive_reopen_controller") if isinstance(payload.get("adaptive_reopen_controller"), dict) else {}
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
        f"- Compatibility-excluded packages: `{', '.join(coverage.get('compatibility_excluded_packages') or []) or 'none'}`",
        f"- Compatibility-excluded lanes: `{', '.join(route_coverage.get('excluded_lanes') or []) or 'none'}`",
        f"- Readiness repair: `{repair.get('status', '')}`",
        "",
        "## Runtime Caps",
        "",
        f"- Profile: `{caps.get('profile', '')}`",
        f"- Unified-memory tier: `{caps.get('mlx_memory_tier', '')}`",
        f"- Memory free percent: `{caps.get('memory_free_pct', '')}`",
        f"- Swap used GB: `{caps.get('swap_used_gb', '')}`",
        f"- Compressed pressure GB: `{caps.get('compressed_pressure_gb', '')}`",
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
        f"- Scheduler mode: `{lane_summary.get('scheduler_mode', '')}`",
        f"- Reopen stage: `{adaptive.get('reopen_stage', '')}`",
        f"- Pressure score: `{adaptive.get('pressure_score', '')}`",
        f"- Stable green windows: `{adaptive.get('stable_green_windows', '')}` / `{adaptive.get('clean_windows_required', '')}`",
        f"- Queue order: `{', '.join(lane_summary.get('recommended_queue_order') or []) or 'none'}`",
        f"- Total MLX memory budget MB: `{lane_summary.get('total_memory_budget_mb', '')}`",
        f"- Admission token budget: `{lane_summary.get('admission_token_budget', '')}`",
        f"- Model cache budget MB: `{lane_summary.get('model_cache_budget_mb', '')}`",
        f"- Min lane cooldown seconds: `{lane_summary.get('min_lane_cooldown_seconds', '')}`",
        f"- Allowed optimized lanes: `{lane_summary.get('allowed_lane_count', '')}`",
        f"- Paused optimized lanes: `{lane_summary.get('paused_lane_count', '')}`",
        f"- Compile-allowed lanes: `{lane_summary.get('compile_allowed_lane_count', '')}`",
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
            f" | `{(route.get('optimization_profile') or {}).get('run_mode', '')}`"
            f" | score `{(route.get('optimization_profile') or {}).get('scheduler_score', '')}`"
            f" | tier `{(route.get('optimization_profile') or {}).get('queue_tier', '')}`"
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
