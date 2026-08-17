#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT_PATH = PROJECT_ROOT / "governance" / "health" / "library_efficiency_deepening_latest.json"
REPORT_PATH = PROJECT_ROOT / "governance" / "library_efficiency_deepening" / "library_efficiency_deepening_latest.md"
OVERRIDE_PATH = PROJECT_ROOT / "config" / ".env.library_efficiency_deepening_override"


EFFICIENCY_LAYERS: list[dict[str, Any]] = [
    {
        "layer": 1,
        "slug": "library_routing_brain",
        "display_name": "Library Routing Brain",
        "objective": "Choose the cheapest correct backend before work starts instead of defaulting to generic Python.",
        "required_packages": ["mlx", "polars", "duckdb", "pyarrow", "numba", "orjson", "msgspec", "xxhash"],
        "optional_packages": ["uvloop", "apsw", "adbc-driver-sqlite"],
        "backend_routes": ["mlx_tensor", "polars_dataframe", "duckdb_sql", "arrow_columnar", "numba_cpu_hot_loop"],
        "efficiency_outputs": ["backend_route_vote", "skip_reason", "cache_preference", "runtime_cap_hint"],
    },
    {
        "layer": 2,
        "slug": "columnar_data_plane",
        "display_name": "Columnar Data Plane",
        "objective": "Prefer Polars, Arrow, and DuckDB for repeated ingestion, joins, scans, and feature-table reads.",
        "required_packages": ["polars", "duckdb", "pyarrow", "adbc-driver-sqlite", "apsw", "zstandard"],
        "optional_packages": ["bottleneck", "numexpr"],
        "backend_routes": ["polars_lazy_scan", "duckdb_query", "arrow_ipc", "sqlite_adbc_bridge"],
        "efficiency_outputs": ["columnar_scan_plan", "jsonl_rescan_avoidance", "feature_table_cache_hit"],
    },
    {
        "layer": 3,
        "slug": "mlx_inference_lane",
        "display_name": "MLX Inference Lane",
        "objective": "Keep tensor inference, embeddings, and distilled model checks in MLX with guarded batch sizes.",
        "required_packages": ["mlx", "mlx-metal", "mlx-lm", "mlx-embeddings", "safetensors"],
        "optional_packages": ["mlx-vlm", "mlx-vision", "mlx-whisper", "mlx-audio"],
        "backend_routes": ["mlx_tensor_inference", "mlx_embedding_memory", "mlx_compile_when_safe"],
        "efficiency_outputs": ["mlx_batch_cap", "compile_mode_hint", "single_job_guard", "model_cache_key"],
    },
    {
        "layer": 4,
        "slug": "incremental_feature_store",
        "display_name": "Incremental Feature Store",
        "objective": "Update only changed windows and shard fingerprints instead of rebuilding full behavior datasets.",
        "required_packages": ["polars", "duckdb", "pyarrow", "xxhash", "zstandard", "msgpack"],
        "optional_packages": ["orjson", "apsw"],
        "backend_routes": ["feature_delta_scan", "content_hash_cache", "duckdb_materialized_view"],
        "efficiency_outputs": ["feature_delta_manifest", "freshness_window", "cache_invalidation_reason"],
    },
    {
        "layer": 5,
        "slug": "quant_pricing_kernel_layer",
        "display_name": "Quant Pricing Kernel Layer",
        "objective": "Route options, Greeks, vol surfaces, and covered-call roll math through dedicated pricing kernels.",
        "required_packages": ["quantlib", "py-vollib", "py-vollib-vectorized", "lets-be-rational", "scipy", "mlx"],
        "optional_packages": ["sympy", "quantstats", "empyrical-reloaded"],
        "backend_routes": ["quantlib_pricer", "vollib_vectorized_greeks", "mlx_gradient_check"],
        "efficiency_outputs": ["pricing_kernel_vote", "greek_cache_key", "covered_call_roll_math_packet"],
    },
    {
        "layer": 6,
        "slug": "econometrics_regime_layer",
        "display_name": "Econometrics And Regime Layer",
        "objective": "Use statistical libraries for regime, volatility, stationarity, spread, and drawdown diagnostics.",
        "required_packages": ["statsmodels", "arch", "scipy", "numpy", "pandas", "numba"],
        "optional_packages": ["patsy", "ta"],
        "backend_routes": ["arch_volatility", "statsmodels_regression", "scipy_stationarity", "numba_fast_metric"],
        "efficiency_outputs": ["regime_state_packet", "volatility_forecast_cache", "spread_stationarity_score"],
    },
    {
        "layer": 7,
        "slug": "tabular_alpha_court",
        "display_name": "Tabular Alpha Court",
        "objective": "Use tabular ML and search libraries for meta-labeling, ranking, feature importance, and parameter selection.",
        "required_packages": ["scikit-learn", "xgboost", "optuna", "numpy", "polars", "joblib"],
        "optional_packages": ["threadpoolctl", "scipy"],
        "backend_routes": ["sklearn_meta_label", "xgboost_ranker", "optuna_search", "joblib_cached_fit"],
        "efficiency_outputs": ["meta_label_score", "feature_importance_packet", "search_budget_stop_reason"],
    },
    {
        "layer": 8,
        "slug": "graph_cross_impact_layer",
        "display_name": "Graph Cross-Impact Layer",
        "objective": "Represent sleeves, accounts, symbols, factors, liquidity, and shared trades as graphs for crowding control.",
        "required_packages": ["networkx", "polars", "duckdb", "pyarrow"],
        "optional_packages": ["mlx-graphs", "mlx-cluster"],
        "backend_routes": ["networkx_exposure_graph", "duckdb_edge_store", "polars_adjacency_scan"],
        "efficiency_outputs": ["cross_impact_graph", "crowding_component", "duplicate_exposure_cluster"],
    },
    {
        "layer": 9,
        "slug": "path_signature_layer",
        "display_name": "Path Signature Layer",
        "objective": "Compress market path shapes into reusable fingerprints for replay matching and regime similarity.",
        "required_packages": ["roughpy", "esig", "pyrecombine", "numpy"],
        "optional_packages": ["mlx", "scipy"],
        "backend_routes": ["roughpy_signature", "esig_signature", "path_similarity_cache"],
        "efficiency_outputs": ["path_signature_hash", "regime_similarity_score", "event_shape_match"],
    },
    {
        "layer": 10,
        "slug": "benchmark_cost_governor",
        "display_name": "Benchmark And Cost Governor",
        "objective": "Make every route prove latency, memory, disk, cache, and accuracy benefit before it gets more work.",
        "required_packages": ["pyinstrument", "line-profiler", "memory-profiler", "psutil", "prometheus-client", "orjson"],
        "optional_packages": ["rich", "structlog", "sentry-sdk"],
        "backend_routes": ["latency_probe", "memory_probe", "cache_hit_meter", "route_cost_ledger"],
        "efficiency_outputs": ["efficiency_scorecard", "route_cost_delta", "library_keep_or_retire_vote"],
    },
]


def _norm_package(name: Any) -> str:
    return str(name or "").strip().lower().replace("_", "-")


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


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


def _status(payload: dict[str, Any], default: str = "missing") -> str:
    if not payload:
        return default
    return str(payload.get("overall_status") or payload.get("status") or default).strip().lower()


def _parse_lock(path: Path) -> dict[str, str]:
    versions: dict[str, str] = {}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return versions
    for raw in lines:
        line = raw.strip()
        if not line or line.startswith("#") or "==" not in line:
            continue
        package, version = line.split("==", 1)
        normalized = _norm_package(package)
        if normalized:
            versions[normalized] = version.strip()
    return versions


def _gate_state(project_root: Path) -> dict[str, Any]:
    health_root = project_root / "governance" / "health"
    health_fast = _load_json(health_root / "health_fast_latest.json")
    paper_400 = _load_json(health_root / "paper_400_ramp_latest.json")
    quality = _load_json(health_root / "promotion_quality_gate_latest.json")
    readiness = _load_json(project_root / "governance" / "walk_forward" / "promotion_readiness_latest.json")
    packet = _load_json(project_root / "governance" / "champion_challenger" / "promotion_packet_latest.json")
    retrain = _load_json(health_root / "retrain_launch_latest.json")

    storage = health_fast.get("storage") if isinstance(health_fast.get("storage"), dict) else {}
    runtime = health_fast.get("runtime_pressure") if isinstance(health_fast.get("runtime_pressure"), dict) else {}
    global_halt = health_fast.get("global_halt") if isinstance(health_fast.get("global_halt"), dict) else {}
    quality_failed = quality.get("failed_checks") if isinstance(quality.get("failed_checks"), list) else []
    readiness_blockers = readiness.get("blocking_reasons") if isinstance(readiness.get("blocking_reasons"), list) else []
    paper_blockers = paper_400.get("blockers") if isinstance(paper_400.get("blockers"), list) else []

    global_halt_clear = bool(global_halt) and not bool(global_halt.get("halt")) and not global_halt.get("clear_blockers")
    storage_green = str(storage.get("severity") or "").strip().lower() in {"stable", "ready", "green", ""}
    runtime_green = _status(runtime, "ready") in {"ready", "ok", "green", ""}
    paper_ready = bool(paper_400.get("ok")) and _status(paper_400, "ready") not in {"blocked", "critical", "failed"}
    promotion_quality_ready = (
        bool(quality.get("ok"))
        and not quality_failed
        and not readiness_blockers
        and bool(packet.get("ok", False))
        and bool(packet.get("ready_for_committee", False))
    )

    blockers: list[str] = []
    if not global_halt_clear:
        blockers.append("global_halt_not_clear")
    if not storage_green:
        blockers.append("storage_not_green")
    if not runtime_green:
        blockers.append("runtime_not_green")
    if not paper_ready:
        blockers.extend(f"paper_400:{item}" for item in paper_blockers)
        if not paper_blockers:
            blockers.append("paper_400_not_ready")
    if not promotion_quality_ready:
        blockers.extend(f"promotion_quality:{item}" for item in quality_failed)
        blockers.extend(f"promotion_readiness:{item}" for item in readiness_blockers)
        if not quality_failed and not readiness_blockers:
            blockers.append("promotion_quality_not_ready")
    if str(retrain.get("state") or "").strip().lower() == "running":
        blockers.append("large_training_batch_running_control_plane_only")

    return {
        "global_halt_clear": global_halt_clear,
        "storage_green": storage_green,
        "runtime_green": runtime_green,
        "paper_400_ready": paper_ready,
        "promotion_quality_ready": promotion_quality_ready,
        "training_batch_active": str(retrain.get("state") or "").strip().lower() == "running",
        "training_batch_pid": retrain.get("pid"),
        "blockers": list(dict.fromkeys(str(item) for item in blockers if str(item).strip())),
    }


def _layer_payload(layer: dict[str, Any], lock_versions: dict[str, str], gates: dict[str, Any]) -> dict[str, Any]:
    required = [_norm_package(item) for item in layer.get("required_packages") or []]
    optional = [_norm_package(item) for item in layer.get("optional_packages") or []]
    required_present = [package for package in required if package in lock_versions]
    required_missing = [package for package in required if package not in lock_versions]
    optional_present = [package for package in optional if package in lock_versions]
    optional_missing = [package for package in optional if package not in lock_versions]
    coverage = round(len(required_present) / max(len(required), 1), 4)
    ready = not required_missing
    paper_blockers = list(gates.get("blockers") or [])
    live_blockers = list(dict.fromkeys([*paper_blockers, "human_live_authority_required", "broker_live_execution_gate_required"]))
    return {
        **layer,
        "required_packages": required,
        "optional_packages": optional,
        "required_packages_present": required_present,
        "required_packages_missing": required_missing,
        "optional_packages_present": optional_present,
        "optional_packages_missing": optional_missing,
        "required_package_coverage": coverage,
        "coverage_status": "ready" if ready else "missing_required_packages",
        "backend_family_scope": ["mlx", "non_mlx"],
        "mode_scope": ["paper", "live"],
        "paper_contract": {
            "enabled": True,
            "mode": "paper_rehearsal_advisory",
            "decision_support_enabled": ready,
            "paper_execution_authority_enabled": False,
            "uses_same_feature_contract_as_live": True,
            "activation_blockers": paper_blockers,
        },
        "live_contract": {
            "enabled": True,
            "mode": "live_advisory_parity",
            "decision_support_enabled": ready,
            "live_execution_authority_enabled": False,
            "uses_same_feature_contract_as_paper": True,
            "activation_blockers": live_blockers,
        },
        "safety_policy": "advisory_and_rehearsal_only_until_runtime_promotion_and_live_authority_gates_clear",
    }


def _recommended_env(payload: dict[str, Any]) -> dict[str, str]:
    return {
        "LIBRARY_EFFICIENCY_DEEPENING_ENABLED": "1",
        "LIBRARY_EFFICIENCY_LAYER_COUNT": str(_safe_int(payload.get("layer_count"), 10)),
        "LIBRARY_EFFICIENCY_BACKEND_SCOPE": "mlx,non_mlx",
        "LIBRARY_EFFICIENCY_MODE_SCOPE": "paper,live",
        "LIBRARY_PAPER_REHEARSAL_ENABLED": "1",
        "LIBRARY_LIVE_ADVISORY_PARITY_ENABLED": "1",
        "LIBRARY_LIVE_EXECUTION_AUTHORITY_ENABLED": "0",
        "LIBRARY_PAPER_EXECUTION_AUTHORITY_ENABLED": "0",
        "LIBRARY_COLUMNAR_ENGINE_PRIORITY": "polars,duckdb,pyarrow",
        "LIBRARY_INCREMENTAL_FEATURE_CACHE_ENABLED": "1",
        "LIBRARY_MLX_INFERENCE_BATCH_GUARDED": "1",
        "LIBRARY_QUANT_PRICING_KERNEL_ENABLED": "1",
        "LIBRARY_GRAPH_CROSS_IMPACT_ENABLED": "1",
        "LIBRARY_PATH_SIGNATURE_REPLAY_MATCHING_ENABLED": "1",
        "LIBRARY_BENCHMARK_COST_GOVERNOR_ENABLED": "1",
    }


def _write_env_override(path: Path, env: dict[str, str]) -> bool:
    lines = ["# Auto-managed by scripts/ops/library_efficiency_deepening.py"]
    for key, value in sorted(env.items()):
        safe = str(value).replace("'", "'\"'\"'")
        lines.append(f"{key}='{safe}'")
    content = "\n".join(lines) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    current = path.read_text(encoding="utf-8") if path.exists() else ""
    if current == content:
        return False
    path.write_text(content, encoding="utf-8")
    return True


def build_payload(project_root: Path = PROJECT_ROOT, *, lock_path: Path | None = None) -> dict[str, Any]:
    health_root = project_root / "governance" / "health"
    lock_versions = _parse_lock(lock_path or project_root / "config" / "requirements.lock.txt")
    gates = _gate_state(project_root)
    layers = [_layer_payload(layer, lock_versions, gates) for layer in EFFICIENCY_LAYERS]
    missing_required = sorted({package for layer in layers for package in list(layer.get("required_packages_missing") or [])})
    all_ready = not missing_required and len(layers) == 10
    library_router = _load_json(health_root / "library_utilization_router_latest.json")
    mlx_router = _load_json(health_root / "mlx_intelligence_router_latest.json")
    deep_quant = _load_json(health_root / "deep_quant_layer_upgrade_latest.json")
    router_statuses = {
        "library_utilization_router": _status(library_router),
        "mlx_intelligence_router": _status(mlx_router),
        "deep_quant_layer_upgrade": _status(deep_quant),
    }
    layer_coverage = round(
        sum(_safe_float(layer.get("required_package_coverage"), 0.0) for layer in layers) / max(len(layers), 1),
        4,
    )
    router_ready_count = sum(
        1
        for status in router_statuses.values()
        if status
        in {
            "ready",
            "advisory",
            "paper_activation_ready",
            "deep_quant_layers_installed_collection_only_activation_blocked",
        }
    )
    efficiency_score = round(0.70 * layer_coverage + 0.30 * (router_ready_count / max(len(router_statuses), 1)), 4)
    status = "ready" if all_ready else "degraded_missing_required_packages"
    if gates.get("blockers") and all_ready:
        status = "library_efficiency_layers_installed_dual_mode_activation_blocked"
    payload: dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "schema_version": 1,
        "ok": len(layers) == 10,
        "overall_status": status,
        "layer_count": len(layers),
        "backend_family_scope": ["mlx", "non_mlx"],
        "mode_scope": ["paper", "live"],
        "paper_contract_count": len(layers),
        "live_contract_count": len(layers),
        "required_package_coverage": layer_coverage,
        "efficiency_score": efficiency_score,
        "missing_required_packages": missing_required,
        "router_statuses": router_statuses,
        "gate_state": gates,
        "paper_mode": {
            "rehearsal_enabled": True,
            "decision_support_enabled": all_ready,
            "paper_execution_authority_enabled": False,
            "uses_live_parity_contract": True,
        },
        "live_mode": {
            "advisory_enabled": True,
            "decision_support_enabled": all_ready,
            "live_execution_authority_enabled": False,
            "uses_paper_parity_contract": True,
        },
        "layers": layers,
        "recommended_actions": [
            "run library-utilization-router --apply after runtime pressure is green to refresh non-MLX caps",
            "run mlx-intelligence-router --apply after MLX package and runtime audits are clean",
            "prefer Polars/DuckDB/Arrow feature scans over raw JSONL rescans",
            "use identical feature and route contracts for paper rehearsal and live advisory parity",
            "keep paper/live execution authority disabled for this layer until promotion and broker live gates clear",
            "promote routes only when benchmark-cost governor shows latency, memory, disk, cache, and accuracy improvement",
        ],
        "artifacts": {
            "json": str(OUT_PATH),
            "report": str(REPORT_PATH),
            "env_override": str(OVERRIDE_PATH),
        },
    }
    payload["recommended_runtime_env"] = _recommended_env(payload)
    return payload


def render_report(payload: dict[str, Any]) -> str:
    lines = [
        "# Library Efficiency Deepening",
        "",
        f"- timestamp_utc: `{payload.get('timestamp_utc')}`",
        f"- status: `{payload.get('overall_status')}`",
        f"- layers: `{payload.get('layer_count')}`",
        f"- backend_scope: `{', '.join(payload.get('backend_family_scope') or [])}`",
        f"- mode_scope: `{', '.join(payload.get('mode_scope') or [])}`",
        f"- efficiency_score: `{payload.get('efficiency_score')}`",
        f"- required_package_coverage: `{payload.get('required_package_coverage')}`",
        f"- missing_required_packages: `{', '.join(payload.get('missing_required_packages') or []) or 'none'}`",
        "",
        "## Paper And Live Contracts",
        "",
        f"- paper_rehearsal_enabled: `{(payload.get('paper_mode') or {}).get('rehearsal_enabled')}`",
        f"- paper_execution_authority_enabled: `{(payload.get('paper_mode') or {}).get('paper_execution_authority_enabled')}`",
        f"- live_advisory_enabled: `{(payload.get('live_mode') or {}).get('advisory_enabled')}`",
        f"- live_execution_authority_enabled: `{(payload.get('live_mode') or {}).get('live_execution_authority_enabled')}`",
        "",
        "## Layers",
        "",
    ]
    for layer in payload.get("layers") or []:
        if not isinstance(layer, dict):
            continue
        lines.extend(
            [
                f"### {layer.get('layer')}. {layer.get('display_name')}",
                "",
                f"- slug: `{layer.get('slug')}`",
                f"- coverage: `{layer.get('required_package_coverage')}`",
                f"- routes: {', '.join(layer.get('backend_routes') or [])}",
                f"- outputs: {', '.join(layer.get('efficiency_outputs') or [])}",
                "",
            ]
        )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Install 10 dual-mode library efficiency deepening layers.")
    parser.add_argument("--json", action="store_true", help="Print the full JSON payload.")
    parser.add_argument("--apply", action="store_true", help="Write the runtime env override.")
    parser.add_argument("--no-write", action="store_true", help="Build without writing artifacts.")
    args = parser.parse_args()

    payload = build_payload()
    if args.apply:
        payload["apply_result"] = {
            "applied": True,
            "override_path": str(OVERRIDE_PATH),
            "override_changed": _write_env_override(OVERRIDE_PATH, {str(k): str(v) for k, v in payload["recommended_runtime_env"].items()}),
        }
    else:
        payload["apply_result"] = {"applied": False, "override_path": str(OVERRIDE_PATH), "override_changed": False}
    if not args.no_write:
        _write_json(OUT_PATH, payload)
        _write_text(REPORT_PATH, render_report(payload))
    if args.json:
        print(json.dumps(payload, ensure_ascii=True, indent=2))
    else:
        print(
            "library_efficiency_deepening "
            f"status={payload.get('overall_status')} "
            f"layers={payload.get('layer_count')} "
            f"score={payload.get('efficiency_score')} "
            f"paper_exec={int(bool((payload.get('paper_mode') or {}).get('paper_execution_authority_enabled')))} "
            f"live_exec={int(bool((payload.get('live_mode') or {}).get('live_execution_authority_enabled')))}"
        )
    return 0 if payload.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
