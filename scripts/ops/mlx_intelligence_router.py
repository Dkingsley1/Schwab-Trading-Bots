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


def _runtime_caps(memory: dict[str, Any], throttle: dict[str, Any], mlx_runtime: dict[str, Any]) -> dict[str, Any]:
    memory_snapshot = memory.get("memory_snapshot") if isinstance(memory.get("memory_snapshot"), dict) else {}
    cotenant = memory.get("cotenant_awareness") if isinstance(memory.get("cotenant_awareness"), dict) else {}
    throttle_profile = str(throttle.get("throttle_profile") or "observe")
    memory_level = str(throttle.get("memory_pressure_level") or "").strip().lower()
    if not memory_level:
        state = str(memory_snapshot.get("memory_pressure_state") or "").strip().lower()
        memory_level = "high" if state in {"red", "critical"} else "elevated" if state in {"yellow", "orange"} else "normal"
    compile_available = bool(((mlx_runtime.get("runtime") or {}).get("compile_available", False)))
    compile_smoke_ok = bool(((mlx_runtime.get("runtime") or {}).get("compile_smoke_ok", False)))
    metal_available = bool(((mlx_runtime.get("runtime") or {}).get("metal_available", False)))
    cotenant_active = bool(cotenant.get("active", False) or cotenant.get("mode") in {"managed_cotenant", "guarded_cotenant"})
    pressure_profile = "max_throughput"
    max_jobs = 3
    tensor_batch_cap = 64
    embedding_batch_cap = 128
    graph_node_cap = 12000
    audio_minutes_cap = 45
    heavy_vlm_enabled = True
    if throttle_profile == "protect_live" or memory_level == "high":
        pressure_profile = "protect_live"
        max_jobs = 1
        tensor_batch_cap = 16
        embedding_batch_cap = 32
        graph_node_cap = 3000
        audio_minutes_cap = 12
        heavy_vlm_enabled = False
    elif throttle_profile == "sustain" or memory_level == "elevated":
        pressure_profile = "sustain"
        max_jobs = 1
        tensor_batch_cap = 32
        embedding_batch_cap = 64
        graph_node_cap = 6000
        audio_minutes_cap = 20
        heavy_vlm_enabled = False
    elif throttle_profile == "soft_cap" or cotenant_active:
        pressure_profile = "foreground_safe"
        max_jobs = 2
        tensor_batch_cap = 48
        embedding_batch_cap = 96
        graph_node_cap = 9000
        audio_minutes_cap = 30
        heavy_vlm_enabled = True
    compile_mode = "canary_first" if compile_available and compile_smoke_ok and metal_available else "off"
    return {
        "profile": pressure_profile,
        "throttle_profile": throttle_profile,
        "memory_pressure_level": memory_level,
        "cotenant_active": cotenant_active,
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
        "policy": "maximize_library_coverage_without_maxing_shared_memory",
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
    return ordered_unique(
        [
            "route every MLX-capable intelligence job through mlx-intelligence-router before expanding the library set",
            "keep mlx.compile canary-first until runtime-throttle and memory-efficiency both stay green"
            if str(caps.get("compile_mode") or "") == "canary_first"
            else "keep mlx.compile disabled for heavy jobs until the compile smoke and Metal checks are green",
            "treat 100 percent utilization as library coverage, not hardware saturation",
            "thin VLM and long audio jobs while foreground apps or memory pressure are active"
            if not bool(caps.get("heavy_vlm_enabled", True)) or bool(caps.get("cotenant_active", False))
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
    quant = load_json(health_root / "quant_model_control_latest.json")
    statuses = _package_statuses(mlx_runtime, mlx_library)
    coverage = _coverage(statuses)
    caps = _runtime_caps(memory, throttle, mlx_runtime)
    routes = _lane_routes(statuses, caps)
    route_coverage = _route_coverage(routes)
    library_matrix = _library_utilization_matrix(routes)
    env = _recommended_env(caps)
    missing_count = _safe_int(coverage.get("missing_count"), 0)
    blocked_lane_count = _safe_int(route_coverage.get("blocked_lane_count"), 0)
    status = "ready"
    if _status(mlx_runtime) == "blocked" or _status(mlx_library) == "blocked" or missing_count:
        status = "blocked"
    elif blocked_lane_count:
        status = "degraded"
    elif str(caps.get("profile") or "") in {"foreground_safe", "sustain", "protect_live"}:
        status = "advisory"
    return {
        "timestamp_utc": iso_now(),
        "schema_version": 1,
        "ok": status in {"ready", "advisory"},
        "overall_status": status,
        "library_coverage": coverage,
        "route_coverage": route_coverage,
        "runtime_caps": caps,
        "recommended_runtime_env": env,
        "workload_routes": routes,
        "library_utilization_matrix": library_matrix,
        "quant_model_status": _status(quant),
        "control_contract": {
            "uses_all_available_mlx_libraries": bool(library_matrix.get("mapped_library_ratio") == 1.0 and missing_count == 0),
            "hardware_saturation_goal": "no",
            "safe_utilization_goal": "100_percent_library_coverage_with_memory_aware_caps",
            "live_path_policy": "feature_enrichment_and_risk_context_only",
            "training_path_policy": "off_hours_or_runtime_throttle_cleared",
            "paper_path_policy": "respect_paper_trade_lock_and_runtime_caps",
        },
        "recommended_actions": _recommended_actions(coverage, route_coverage, caps),
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
