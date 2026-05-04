#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from importlib import metadata
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from scripts.ops.long_runtime_common import iso_now, load_json, ordered_unique, write_payload
else:
    from .long_runtime_common import PROJECT_ROOT, iso_now, load_json, ordered_unique, write_payload


DEFAULT_LOCK = PROJECT_ROOT / "config" / "requirements.lock.txt"
DEFAULT_OUT_PATH = PROJECT_ROOT / "governance" / "health" / "library_utilization_router_latest.json"
DEFAULT_EXTERNAL_CONTEXT_PATH = PROJECT_ROOT / "exports" / "external_context" / "library_utilization_router_latest.json"
DEFAULT_MARKDOWN_PATH = PROJECT_ROOT / "exports" / "reports" / "operator" / "library_utilization_router_latest.md"
DEFAULT_OVERRIDE_PATH = PROJECT_ROOT / "config" / ".env.library_utilization_router_override"

MLX_ROUTED_PACKAGES = {
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
    "parakeet-mlx",
    "esig",
    "roughpy",
    "pyrecombine",
}

LANE_SPECS: dict[str, dict[str, Any]] = {
    "broker_market_data": {
        "workload_family": "broker_api_market_data_and_remote_context_collection",
        "priority": "protected_if_live",
        "target_surfaces": ["schwab_auth", "yfinance", "macro_context", "market_context"],
    },
    "async_networking": {
        "workload_family": "async_http_websocket_dns_and_rate_limited_ingestion",
        "priority": "protected_when_collecting",
        "target_surfaces": ["livefeed_refresh", "macro_context_sync", "provider_mesh"],
    },
    "storage_sql": {
        "workload_family": "sqlite_duckdb_arrow_adbc_sqlalchemy_and_backpressure_writers",
        "priority": "protected_when_draining",
        "target_surfaces": ["sql_link_writer", "duckdb_analytics", "backpressure_drainers"],
    },
    "dataframe_feature_engine": {
        "workload_family": "dataframe_vectorized_features_technical_indicators_and_time_series",
        "priority": "protected_when_training",
        "target_surfaces": ["feature_store", "training_samples", "strategy_features"],
    },
    "quant_derivatives_risk": {
        "workload_family": "options_pricing_quantlib_vollib_risk_metrics_and_symbolic_math",
        "priority": "research_only_or_guarded_paper",
        "target_surfaces": ["quant_model_control", "options_greeks", "risk_service"],
    },
    "statistical_ml": {
        "workload_family": "classical_ml_optimization_stats_regime_filters_and_boosted_models",
        "priority": "off_hours_preferred",
        "target_surfaces": ["training_quality", "model_lifecycle", "regime_control"],
    },
    "portable_ml_replay": {
        "workload_family": "pytorch_onnx_transformers_replay_canaries_and_model_interop",
        "priority": "disabled_or_canary_during_live",
        "target_surfaces": ["pytorch_replay_canary", "onnx_audit", "portable_brain"],
    },
    "nlp_tokenization_research": {
        "workload_family": "tokenization_datasets_huggingface_and_text_research_inputs",
        "priority": "throttle_first",
        "target_surfaces": ["research_pipeline", "sentiment_agents", "macro_transcripts"],
    },
    "visualization_reporting": {
        "workload_family": "plots_reports_pdf_markdown_terminal_tables_and_visual_quality",
        "priority": "throttle_first",
        "target_surfaces": ["report_quality", "paper_performance", "project_timeline"],
    },
    "web_api_ui": {
        "workload_family": "local_api_cockpit_phone_feed_and_operator_web_surfaces",
        "priority": "operator_visible",
        "target_surfaces": ["operator_cockpit", "phone_feed", "local_api"],
    },
    "observability_ops": {
        "workload_family": "metrics_profiling_logging_progress_and_runtime_diagnostics",
        "priority": "throttle_first",
        "target_surfaces": ["runtime_throttle", "memory_efficiency", "cost_telemetry"],
    },
    "security_auth_config": {
        "workload_family": "auth_secrets_crypto_settings_validation_and_config_files",
        "priority": "protected_if_auth",
        "target_surfaces": ["schwab_auth_supervisor", "secret_scan", "runtime_env"],
    },
    "serialization_compression": {
        "workload_family": "json_msgpack_arrow_safetensors_flatbuffers_and_compression",
        "priority": "protected_when_writing",
        "target_surfaces": ["jsonl_buffers", "artifact_store", "external_context"],
    },
    "testing_dev_tooling": {
        "workload_family": "tests_formatting_packaging_and_developer_feedback",
        "priority": "off_hours_or_manual",
        "target_surfaces": ["regression_guard", "commands_hygiene", "codex_project_guard"],
    },
    "system_runtime_primitives": {
        "workload_family": "python_runtime_dependency_glue_typing_dates_paths_and_low_level_support",
        "priority": "always_available",
        "target_surfaces": ["runtime_python", "supportability_control", "opsctl"],
    },
    "audio_media_non_mlx": {
        "workload_family": "audio_file_io_resampling_waveforms_and_media_support_outside_mlx",
        "priority": "protected_if_live_event",
        "target_surfaces": ["macro_media_ingest", "live_macro_auto_watch"],
    },
    "runtime_support_misc": {
        "workload_family": "dependency_support_packages_that_exist_to_make_primary_lanes_work",
        "priority": "always_available",
        "target_surfaces": ["runtime_dependency_profiles", "ops_coordinator"],
    },
}

PACKAGE_LANE_OVERRIDES: dict[str, str] = {
    "apscheduler": "observability_ops",
    "authlib": "security_auth_config",
    "bottleneck": "dataframe_feature_engine",
    "flask": "web_api_ui",
    "jinja2": "visualization_reporting",
    "mako": "storage_sql",
    "markupsafe": "visualization_reporting",
    "pyyaml": "security_auth_config",
    "pygments": "visualization_reporting",
    "quantlib": "quant_derivatives_risk",
    "sqlalchemy": "storage_sql",
    "werkzeug": "web_api_ui",
    "adbc-driver-manager": "storage_sql",
    "adbc-driver-sqlite": "storage_sql",
    "alembic": "storage_sql",
    "apsw": "storage_sql",
    "duckdb": "storage_sql",
    "duckdb-engine": "storage_sql",
    "peewee": "storage_sql",
    "redis": "storage_sql",
    "pyarrow": "storage_sql",
    "polars": "dataframe_feature_engine",
    "polars-runtime-32": "dataframe_feature_engine",
    "pandas": "dataframe_feature_engine",
    "pandas-stubs": "dataframe_feature_engine",
    "pandas-ta": "dataframe_feature_engine",
    "numpy": "dataframe_feature_engine",
    "numexpr": "dataframe_feature_engine",
    "ta": "dataframe_feature_engine",
    "arch": "statistical_ml",
    "empyrical-reloaded": "quant_derivatives_risk",
    "lets-be-rational": "quant_derivatives_risk",
    "py-lets-be-rational": "quant_derivatives_risk",
    "py-vollib": "quant_derivatives_risk",
    "py-vollib-vectorized": "quant_derivatives_risk",
    "quantstats": "quant_derivatives_risk",
    "scipy": "statistical_ml",
    "scikit-learn": "statistical_ml",
    "statsmodels": "statistical_ml",
    "sympy": "quant_derivatives_risk",
    "xgboost": "statistical_ml",
    "optuna": "statistical_ml",
    "numba": "statistical_ml",
    "llvmlite": "statistical_ml",
    "torch": "portable_ml_replay",
    "onnx": "portable_ml_replay",
    "onnxruntime": "portable_ml_replay",
    "transformers": "nlp_tokenization_research",
    "datasets": "nlp_tokenization_research",
    "huggingface-hub": "nlp_tokenization_research",
    "hf-xet": "nlp_tokenization_research",
    "safetensors": "serialization_compression",
    "sentencepiece": "nlp_tokenization_research",
    "tiktoken": "nlp_tokenization_research",
    "tokenizers": "nlp_tokenization_research",
    "regex": "nlp_tokenization_research",
    "aiohttp": "async_networking",
    "aiodns": "async_networking",
    "aiofiles": "async_networking",
    "aiohappyeyeballs": "async_networking",
    "aiosignal": "async_networking",
    "anyio": "async_networking",
    "curl-cffi": "broker_market_data",
    "httpcore": "async_networking",
    "httpx": "async_networking",
    "requests": "broker_market_data",
    "urllib3": "async_networking",
    "websockets": "async_networking",
    "uvloop": "async_networking",
    "watchfiles": "observability_ops",
    "yarl": "async_networking",
    "multidict": "async_networking",
    "frozenlist": "async_networking",
    "pycares": "async_networking",
    "schwab-py": "broker_market_data",
    "yfinance": "broker_market_data",
    "beautifulsoup4": "broker_market_data",
    "soupsieve": "broker_market_data",
    "certifi": "async_networking",
    "charset-normalizer": "async_networking",
    "idna": "async_networking",
    "matplotlib": "visualization_reporting",
    "seaborn": "visualization_reporting",
    "plotly": "visualization_reporting",
    "pillow": "visualization_reporting",
    "opencv-python": "visualization_reporting",
    "fonttools": "visualization_reporting",
    "contourpy": "visualization_reporting",
    "cycler": "visualization_reporting",
    "kiwisolver": "visualization_reporting",
    "pyparsing": "visualization_reporting",
    "rich": "visualization_reporting",
    "markdown-it-py": "visualization_reporting",
    "mdurl": "visualization_reporting",
    "tabulate": "visualization_reporting",
    "fastapi": "web_api_ui",
    "starlette": "web_api_ui",
    "uvicorn": "web_api_ui",
    "click": "web_api_ui",
    "typer": "web_api_ui",
    "shellingham": "web_api_ui",
    "blinker": "web_api_ui",
    "itsdangerous": "web_api_ui",
    "orjson": "serialization_compression",
    "ujson": "serialization_compression",
    "simplejson": "serialization_compression",
    "msgspec": "serialization_compression",
    "msgpack": "serialization_compression",
    "flatbuffers": "serialization_compression",
    "protobuf": "serialization_compression",
    "zstandard": "serialization_compression",
    "xxhash": "serialization_compression",
    "cryptography": "security_auth_config",
    "python-dotenv": "security_auth_config",
    "pydantic": "security_auth_config",
    "pydantic-core": "security_auth_config",
    "pydantic-settings": "security_auth_config",
    "annotated-types": "security_auth_config",
    "typing-inspection": "system_runtime_primitives",
    "typing-extensions": "system_runtime_primitives",
    "psutil": "observability_ops",
    "prometheus-client": "observability_ops",
    "sentry-sdk": "observability_ops",
    "structlog": "observability_ops",
    "loguru": "observability_ops",
    "colorlog": "observability_ops",
    "memory-profiler": "observability_ops",
    "line-profiler": "observability_ops",
    "py-spy": "observability_ops",
    "pyinstrument": "observability_ops",
    "tqdm": "observability_ops",
    "pytest": "testing_dev_tooling",
    "pluggy": "testing_dev_tooling",
    "iniconfig": "testing_dev_tooling",
    "autopep8": "testing_dev_tooling",
    "pycodestyle": "testing_dev_tooling",
    "setuptools": "testing_dev_tooling",
    "wheel": "testing_dev_tooling",
    "packaging": "system_runtime_primitives",
    "platformdirs": "system_runtime_primitives",
    "python-dateutil": "system_runtime_primitives",
    "pytz": "system_runtime_primitives",
    "tzlocal": "system_runtime_primitives",
    "six": "system_runtime_primitives",
    "attrs": "system_runtime_primitives",
    "frozendict": "system_runtime_primitives",
    "fsspec": "system_runtime_primitives",
    "filelock": "system_runtime_primitives",
    "importlib-resources": "system_runtime_primitives",
    "more-itertools": "system_runtime_primitives",
    "multiprocess": "system_runtime_primitives",
    "multitasking": "system_runtime_primitives",
    "dill": "system_runtime_primitives",
    "joblib": "system_runtime_primitives",
    "threadpoolctl": "system_runtime_primitives",
    "tenacity": "system_runtime_primitives",
    "ml-dtypes": "system_runtime_primitives",
    "mpmath": "system_runtime_primitives",
    "cffi": "system_runtime_primitives",
    "pycparser": "system_runtime_primitives",
    "propcache": "system_runtime_primitives",
    "h11": "async_networking",
    "miniaudio": "audio_media_non_mlx",
    "audioread": "audio_media_non_mlx",
    "librosa": "audio_media_non_mlx",
    "sounddevice": "audio_media_non_mlx",
    "soundfile": "audio_media_non_mlx",
    "soxr": "audio_media_non_mlx",
    "pyloudnorm": "audio_media_non_mlx",
}


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


def _parse_version_lines(lines: list[str]) -> dict[str, str]:
    versions: dict[str, str] = {}
    for raw in lines:
        line = raw.strip()
        if not line or line.startswith("#") or "==" not in line:
            continue
        package, version = line.split("==", 1)
        package_name = _norm_package(package)
        if package_name:
            versions[package_name] = version.strip()
    return versions


def _load_lock_versions(lock_file: Path = DEFAULT_LOCK) -> dict[str, str]:
    try:
        return _parse_version_lines(lock_file.read_text(encoding="utf-8").splitlines())
    except OSError:
        return {}


def _load_installed_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for dist in metadata.distributions():
        name = _norm_package(dist.metadata.get("Name", ""))
        if name:
            versions[name] = str(dist.version)
    return versions


def _is_mlx_routed(package: str) -> bool:
    normalized = _norm_package(package)
    return normalized in MLX_ROUTED_PACKAGES or normalized.startswith("mlx-")


def _package_status(package: str, lock_versions: dict[str, str], installed_versions: dict[str, str]) -> str:
    locked = lock_versions.get(package)
    installed = installed_versions.get(package)
    if locked and installed:
        return "ok" if locked == installed else "version_mismatch"
    if locked:
        return "missing_runtime"
    if installed:
        return "runtime_only"
    return "missing"


def _infer_lane(package: str) -> str:
    normalized = _norm_package(package)
    if normalized in PACKAGE_LANE_OVERRIDES:
        return PACKAGE_LANE_OVERRIDES[normalized]
    if any(token in normalized for token in ("sql", "duck", "arrow", "adbc", "sqlite")):
        return "storage_sql"
    if any(token in normalized for token in ("aio", "http", "websocket", "curl", "request", "url")):
        return "async_networking"
    if any(token in normalized for token in ("pandas", "numpy", "polars", "bottle", "dataframe")):
        return "dataframe_feature_engine"
    if any(token in normalized for token in ("quant", "vollib", "rational", "sympy", "mpmath")):
        return "quant_derivatives_risk"
    if any(token in normalized for token in ("sklearn", "learn", "stats", "xgboost", "optuna", "numba")):
        return "statistical_ml"
    if any(token in normalized for token in ("torch", "onnx", "transformer", "token", "dataset", "huggingface")):
        return "portable_ml_replay"
    if any(token in normalized for token in ("plot", "matplotlib", "pillow", "opencv", "rich", "markdown", "font")):
        return "visualization_reporting"
    if any(token in normalized for token in ("fastapi", "flask", "uvicorn", "werkzeug", "starlette")):
        return "web_api_ui"
    if any(token in normalized for token in ("crypto", "auth", "dotenv", "pydantic", "yaml")):
        return "security_auth_config"
    if any(token in normalized for token in ("json", "msg", "flat", "proto", "zstandard", "safetensor")):
        return "serialization_compression"
    if any(token in normalized for token in ("profile", "sentry", "prometheus", "psutil", "log", "tqdm")):
        return "observability_ops"
    if any(token in normalized for token in ("pytest", "pep", "pluggy", "setuptools", "wheel")):
        return "testing_dev_tooling"
    if any(token in normalized for token in ("audio", "sound", "soxr", "loudnorm")):
        return "audio_media_non_mlx"
    return "runtime_support_misc"


def _package_inventory(lock_versions: dict[str, str], installed_versions: dict[str, str]) -> list[dict[str, Any]]:
    packages = sorted((set(lock_versions) | set(installed_versions)) - {pkg for pkg in set(lock_versions) | set(installed_versions) if _is_mlx_routed(pkg)})
    rows: list[dict[str, Any]] = []
    for package in packages:
        lane = _infer_lane(package)
        rows.append(
            {
                "package": package,
                "lane": lane,
                "locked_version": lock_versions.get(package),
                "installed_version": installed_versions.get(package),
                "status": _package_status(package, lock_versions, installed_versions),
                "source": "locked" if package in lock_versions else "runtime_only",
            }
        )
    return rows


def _lane_routes(package_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_lane: dict[str, list[dict[str, Any]]] = {lane: [] for lane in LANE_SPECS}
    for row in package_rows:
        lane = str(row.get("lane") or "runtime_support_misc")
        by_lane.setdefault(lane, []).append(row)
    routes: list[dict[str, Any]] = []
    for lane, spec in LANE_SPECS.items():
        rows = by_lane.get(lane, [])
        statuses = {str(row.get("status") or "") for row in rows}
        if "missing_runtime" in statuses:
            status = "blocked"
        elif "version_mismatch" in statuses:
            status = "degraded"
        elif any(str(row.get("source") or "") == "runtime_only" for row in rows):
            status = "advisory"
        else:
            status = "ready" if rows else "thin"
        routes.append(
            {
                "lane": lane,
                "status": status,
                "workload_family": spec.get("workload_family"),
                "priority": spec.get("priority"),
                "target_surfaces": spec.get("target_surfaces", []),
                "package_count": len(rows),
                "locked_package_count": sum(1 for row in rows if str(row.get("source") or "") == "locked"),
                "runtime_only_package_count": sum(1 for row in rows if str(row.get("source") or "") == "runtime_only"),
                "packages": [str(row.get("package") or "") for row in rows],
                "blocked_packages": [str(row.get("package") or "") for row in rows if str(row.get("status") or "") == "missing_runtime"],
                "version_mismatch_packages": [str(row.get("package") or "") for row in rows if str(row.get("status") or "") == "version_mismatch"],
            }
        )
    return routes


def _coverage(package_rows: list[dict[str, Any]]) -> dict[str, Any]:
    locked_rows = [row for row in package_rows if str(row.get("source") or "") == "locked"]
    managed_rows = package_rows
    missing = [row for row in locked_rows if str(row.get("status") or "") == "missing_runtime"]
    mismatched = [row for row in locked_rows if str(row.get("status") or "") == "version_mismatch"]
    runtime_only = [row for row in managed_rows if str(row.get("source") or "") == "runtime_only"]
    mapped = [row for row in managed_rows if str(row.get("lane") or "")]
    return {
        "locked_non_mlx_package_count": len(locked_rows),
        "managed_non_mlx_package_count": len(managed_rows),
        "mapped_package_count": len(mapped),
        "runtime_only_package_count": len(runtime_only),
        "missing_runtime_count": len(missing),
        "version_mismatch_count": len(mismatched),
        "coverage_ratio": round(len(mapped) / max(len(managed_rows), 1), 4),
        "locked_runtime_ok_ratio": round((len(locked_rows) - len(missing) - len(mismatched)) / max(len(locked_rows), 1), 4),
        "missing_runtime_packages": [str(row.get("package") or "") for row in missing],
        "version_mismatch_packages": [str(row.get("package") or "") for row in mismatched],
        "runtime_only_packages": [str(row.get("package") or "") for row in runtime_only],
    }


def _runtime_caps(memory: dict[str, Any], throttle: dict[str, Any], mlx_router: dict[str, Any]) -> dict[str, Any]:
    throttle_profile = str(throttle.get("throttle_profile") or "observe")
    memory_level = str(throttle.get("memory_pressure_level") or "").strip().lower()
    if not memory_level:
        snapshot = memory.get("memory_snapshot") if isinstance(memory.get("memory_snapshot"), dict) else {}
        state = str(snapshot.get("memory_pressure_state") or "").strip().lower()
        memory_level = "high" if state in {"red", "critical"} else "elevated" if state in {"yellow", "orange"} else "normal"
    mlx_caps = mlx_router.get("runtime_caps") if isinstance(mlx_router.get("runtime_caps"), dict) else {}
    mlx_jobs = _safe_int(mlx_caps.get("max_concurrent_mlx_jobs"), 2)
    profile = "max_library_coverage"
    async_concurrency = 12
    sql_writer_workers = 3
    dataframe_workers = 4
    model_replay_jobs = 1
    report_render_jobs = 2
    if throttle_profile == "protect_live" or memory_level == "high":
        profile = "protect_live"
        async_concurrency = 4
        sql_writer_workers = 1
        dataframe_workers = 1
        model_replay_jobs = 0
        report_render_jobs = 1
    elif throttle_profile == "sustain" or memory_level == "elevated":
        profile = "sustain"
        async_concurrency = 6
        sql_writer_workers = 1
        dataframe_workers = 2
        model_replay_jobs = 0
        report_render_jobs = 1
    elif throttle_profile == "soft_cap" or mlx_jobs <= 2:
        profile = "foreground_safe"
        async_concurrency = 8
        sql_writer_workers = 2
        dataframe_workers = 2
        model_replay_jobs = 0
        report_render_jobs = 1
    return {
        "profile": profile,
        "throttle_profile": throttle_profile,
        "memory_pressure_level": memory_level,
        "max_async_request_concurrency": async_concurrency,
        "max_sql_writer_workers": sql_writer_workers,
        "max_dataframe_workers": dataframe_workers,
        "max_portable_model_replay_jobs": model_replay_jobs,
        "max_report_render_jobs": report_render_jobs,
        "respect_mlx_job_cap": mlx_jobs,
        "policy": "100_percent_library_lane_coverage_with_runtime_caps",
    }


def _recommended_env(caps: dict[str, Any]) -> dict[str, str]:
    return {
        "LIBRARY_UTILIZATION_ROUTER_ENABLED": "1",
        "LIBRARY_UTILIZATION_PROFILE": str(caps.get("profile") or "foreground_safe"),
        "LIBRARY_ASYNC_REQUEST_CONCURRENCY_CAP": str(_safe_int(caps.get("max_async_request_concurrency"), 8)),
        "LIBRARY_SQL_WRITER_WORKER_CAP": str(_safe_int(caps.get("max_sql_writer_workers"), 1)),
        "LIBRARY_DATAFRAME_WORKER_CAP": str(_safe_int(caps.get("max_dataframe_workers"), 2)),
        "LIBRARY_PORTABLE_MODEL_REPLAY_JOBS": str(_safe_int(caps.get("max_portable_model_replay_jobs"), 0)),
        "LIBRARY_REPORT_RENDER_JOBS": str(_safe_int(caps.get("max_report_render_jobs"), 1)),
        "LIBRARY_RESPECT_MLX_JOB_CAP": str(_safe_int(caps.get("respect_mlx_job_cap"), 2)),
        "LIBRARY_UTILIZATION_GOAL": "100_percent_lane_coverage_not_100_percent_hardware_load",
        "LIBRARY_DEFAULT_ML_BACKEND": "mlx",
        "PRIMARY_ML_RUNTIME_BACKEND": "mlx",
        "PORTABLE_MODEL_REPLAY_POLICY": "canary_or_off_hours_only",
    }


def _library_utilization_matrix(package_rows: list[dict[str, Any]], routes: list[dict[str, Any]]) -> dict[str, Any]:
    package_to_lane = {str(row.get("package") or ""): str(row.get("lane") or "") for row in package_rows}
    lane_to_packages = {
        str(route.get("lane") or ""): list(route.get("packages") or [])
        for route in routes
    }
    unmapped = [package for package, lane in package_to_lane.items() if not lane]
    return {
        "package_count": len(package_to_lane),
        "mapped_package_count": len(package_to_lane) - len(unmapped),
        "mapped_package_ratio": round((len(package_to_lane) - len(unmapped)) / max(len(package_to_lane), 1), 4),
        "unmapped_packages": unmapped,
        "package_to_lane": package_to_lane,
        "lane_to_packages": lane_to_packages,
        "utilization_goal": "100_percent_non_mlx_library_coverage_in_control_plane_not_hardware_saturation",
    }


def _recommended_actions(coverage: dict[str, Any], caps: dict[str, Any]) -> list[str]:
    return ordered_unique(
        [
            "route every non-MLX runtime package through library-utilization-router before adding more dependency weight",
            "treat 100 percent library utilization as lane ownership and coverage, not CPU or memory saturation",
            "keep PyTorch and ONNX in replay/canary lanes during live MLX collection"
            if _safe_int(caps.get("max_portable_model_replay_jobs"), 0) == 0
            else "",
            "repair missing locked packages before relying on their owner lane"
            if _safe_int(coverage.get("missing_runtime_count"), 0)
            else "",
            "align version mismatches between the lock and runtime before broad retrains"
            if _safe_int(coverage.get("version_mismatch_count"), 0)
            else "",
            "./scripts/ops/opsctl.sh runtime-throttle --apply --json",
        ]
    )


def _write_env_override(path: Path, env: dict[str, str]) -> bool:
    lines = ["# Auto-managed by scripts/ops/library_utilization_router.py"]
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


def build_payload(
    project_root: Path = PROJECT_ROOT,
    *,
    lock_file: Path | None = None,
    installed_versions: dict[str, str] | None = None,
) -> dict[str, Any]:
    health_root = project_root / "governance" / "health"
    lock_versions = _load_lock_versions(lock_file or project_root / "config" / "requirements.lock.txt")
    installed = {
        _norm_package(name): str(version)
        for name, version in (installed_versions if isinstance(installed_versions, dict) else _load_installed_versions()).items()
        if _norm_package(name)
    }
    rows = _package_inventory(lock_versions, installed)
    routes = _lane_routes(rows)
    coverage = _coverage(rows)
    matrix = _library_utilization_matrix(rows, routes)
    runtime_caps = _runtime_caps(
        load_json(health_root / "memory_efficiency_control_latest.json"),
        load_json(health_root / "runtime_throttle_control_latest.json"),
        load_json(health_root / "mlx_intelligence_router_latest.json"),
    )
    env = _recommended_env(runtime_caps)
    status = "ready"
    if _safe_int(coverage.get("missing_runtime_count"), 0):
        status = "blocked"
    elif _safe_int(coverage.get("version_mismatch_count"), 0):
        status = "degraded"
    elif _safe_int(coverage.get("runtime_only_package_count"), 0) or str(runtime_caps.get("profile") or "") != "max_library_coverage":
        status = "advisory"
    return {
        "timestamp_utc": iso_now(),
        "schema_version": 1,
        "ok": status in {"ready", "advisory"},
        "overall_status": status,
        "coverage": coverage,
        "runtime_caps": runtime_caps,
        "recommended_runtime_env": env,
        "workload_routes": routes,
        "library_utilization_matrix": matrix,
        "package_inventory": rows,
        "control_contract": {
            "uses_all_managed_non_mlx_libraries": bool(matrix.get("mapped_package_ratio") == 1.0 and _safe_int(coverage.get("missing_runtime_count"), 0) == 0),
            "hardware_saturation_goal": "no",
            "safe_utilization_goal": "100_percent_non_mlx_library_lane_coverage_with_runtime_caps",
            "default_ml_backend": "mlx",
            "mlx_boundary": "mlx_specific_packages_are_owned_by_mlx_intelligence_router",
            "portable_ml_policy": "pytorch_onnx_transformers_stay_canary_or_off_hours_when_live_collection_is_active",
        },
        "recommended_actions": _recommended_actions(coverage, runtime_caps),
        "artifact_paths": {
            "json": str(DEFAULT_OUT_PATH),
            "external_context": str(DEFAULT_EXTERNAL_CONTEXT_PATH),
            "markdown": str(DEFAULT_MARKDOWN_PATH),
            "env_override": str(DEFAULT_OVERRIDE_PATH),
        },
    }


def render_markdown(payload: dict[str, Any]) -> str:
    coverage = payload.get("coverage") if isinstance(payload.get("coverage"), dict) else {}
    caps = payload.get("runtime_caps") if isinstance(payload.get("runtime_caps"), dict) else {}
    lines = [
        "# Library Utilization Router",
        "",
        f"Generated UTC: `{payload.get('timestamp_utc', '')}`",
        f"Overall status: `{payload.get('overall_status', '')}`",
        "",
        "## Coverage",
        "",
        f"- Managed non-MLX packages: `{coverage.get('managed_non_mlx_package_count', 0)}`",
        f"- Locked non-MLX packages: `{coverage.get('locked_non_mlx_package_count', 0)}`",
        f"- Mapped package coverage: `{coverage.get('coverage_ratio', 0.0)}`",
        f"- Locked runtime OK ratio: `{coverage.get('locked_runtime_ok_ratio', 0.0)}`",
        f"- Missing locked packages: `{', '.join(coverage.get('missing_runtime_packages') or []) or 'none'}`",
        "",
        "## Runtime Caps",
        "",
        f"- Profile: `{caps.get('profile', '')}`",
        f"- Async request concurrency cap: `{caps.get('max_async_request_concurrency', '')}`",
        f"- SQL writer worker cap: `{caps.get('max_sql_writer_workers', '')}`",
        f"- Dataframe worker cap: `{caps.get('max_dataframe_workers', '')}`",
        f"- Portable model replay jobs: `{caps.get('max_portable_model_replay_jobs', '')}`",
        f"- Report render jobs: `{caps.get('max_report_render_jobs', '')}`",
        "",
        "## Workload Routes",
        "",
    ]
    for route in payload.get("workload_routes") or []:
        if not isinstance(route, dict):
            continue
        lines.append(
            f"- `{route.get('lane', '')}`: `{route.get('status', '')}`, "
            f"`{route.get('package_count', 0)}` packages"
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
    write_payload(out_path, payload)
    write_payload(external_context_path, payload)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(render_markdown(payload), encoding="utf-8")
    return apply_result


def main() -> int:
    parser = argparse.ArgumentParser(description="Map non-MLX runtime libraries to workload lanes with safe utilization caps.")
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--lock-file", default=str(DEFAULT_LOCK))
    parser.add_argument("--out-file", default=str(DEFAULT_OUT_PATH))
    parser.add_argument("--external-context-file", default=str(DEFAULT_EXTERNAL_CONTEXT_PATH))
    parser.add_argument("--markdown-file", default=str(DEFAULT_MARKDOWN_PATH))
    parser.add_argument("--override-file", default=str(DEFAULT_OVERRIDE_PATH))
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    payload = build_payload(project_root, lock_file=Path(args.lock_file).expanduser())
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
        coverage = payload.get("coverage") if isinstance(payload.get("coverage"), dict) else {}
        print(
            "library_utilization_router "
            f"status={payload.get('overall_status', '')} "
            f"coverage={float(coverage.get('coverage_ratio', 0.0) or 0.0):.3f}"
        )
    return 0 if payload.get("overall_status") in {"ready", "advisory", "degraded"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
