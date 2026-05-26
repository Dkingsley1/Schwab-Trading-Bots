#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from scripts.ops.long_runtime_common import PROJECT_ROOT, iso_now, load_json, write_payload
else:
    from .long_runtime_common import PROJECT_ROOT, iso_now, load_json, write_payload


DEFAULT_OUT_PATH = PROJECT_ROOT / "governance" / "health" / "pressure_relief_control_latest.json"
DEFAULT_OVERRIDE_PATH = PROJECT_ROOT / "config" / ".env.pressure_relief_override"
SUPPORT_RENICE_PATTERNS: tuple[str, ...] = (
    "yt-dlp",
    "ffmpeg",
    "scripts/ops/live_feed_tail.sh",
    "scripts/link_jsonl_to_sql.py",
    "scripts/ops/backpressure_slo_bot.py",
    "scripts/ops/bot_quality_autopilot.py",
    "scripts/ops/command_validity_bot.py",
    "scripts/ops/commands_hygiene_bot.py",
    "scripts/ops/coverage_gap_closer.py",
    "scripts/ops/creative_cotenant_guard.py",
    "scripts/ops/data_source_divergence_bot.py",
    "scripts/ops/ingestion_storage_governor.py",
    "scripts/ops/process_watchdog.py",
    "scripts/ops/report_quality_guard.py",
    "scripts/ops/runtime_gate_dashboard.py",
    "scripts/ops/storage_quota_guard.py",
    "scripts/ops/sql_link_shard_manager.py",
    "scripts/ops/writer_cycle_coordinator.py",
)

PRESSURE_RELIEF_ITEMS: tuple[dict[str, Any], ...] = (
    {"id": "fast_health_read_only", "title": "Fast read-only health", "keys": ["OPS_HEALTH_FAST_ENABLED", "OPS_HEALTH_NO_REPORT_REFRESH"]},
    {"id": "quiet_window_maintenance", "title": "Quiet-window maintenance and reports", "keys": ["MAINTENANCE_SLOT_QUIET_WINDOWS_ENABLED", "MAINTENANCE_SLOT_DEFER_OUTSIDE_QUIET_WINDOW"]},
    {"id": "heavy_feed_ttl", "title": "Auto-expiring heavy feed tails", "keys": ["LIVE_FEED_HEAVY_TTL_ENABLED", "LIVE_FEED_HEAVY_TTL_SECONDS"]},
    {"id": "adaptive_sql_writer", "title": "Adaptive SQL writer drain cadence", "keys": ["SQL_LINK_SERVICE_ADAPTIVE_WRITER_ENABLED", "SQL_LINK_SERVICE_INTERVAL_SECONDS"]},
    {"id": "foreground_app_governor", "title": "Foreground app pressure governor", "keys": ["FOREGROUND_APP_PRESSURE_GOVERNOR_ENABLED", "RUNTIME_COTENANT_AWARE"]},
    {"id": "macro_capture_nice", "title": "Low-priority macro/media capture", "keys": ["MACRO_CAPTURE_NICE_LEVEL", "MACRO_CAPTURE_BACKGROUND_POLICY"]},
    {"id": "calm_mode_controller", "title": "Calm mode while pressure clears", "keys": ["CALM_MODE_ENABLED", "CALM_MODE_MIN_SECONDS"]},
    {"id": "mlx_lazy_caps", "title": "MLX lazy-load and concurrency caps", "keys": ["MLX_LAZY_IMPORTS", "MLX_INTELLIGENCE_MAX_CONCURRENT_JOBS"]},
    {"id": "quant_research_caps", "title": "Quant research path caps", "keys": ["QUANT_MODEL_RESEARCH_ONLY", "QUANT_MODEL_MAX_WORKERS"]},
    {"id": "report_render_cap", "title": "Single report render lane", "keys": ["REPORT_RENDER_MAX_JOBS", "LIBRARY_REPORT_RENDER_JOBS"]},
    {"id": "command_audit_cadence", "title": "Command/report audit cadence stretch", "keys": ["COMMAND_VALIDITY_MIN_INTERVAL_SECONDS", "COMMANDS_HYGIENE_MIN_INTERVAL_SECONDS"]},
    {"id": "data_source_cadence", "title": "Data-source divergence cadence stretch", "keys": ["DATA_SOURCE_DIVERGENCE_REFRESH_INTERVAL_SECONDS"]},
    {"id": "feed_file_cap", "title": "Heavy feed file and line caps", "keys": ["LIVE_FEED_HEAVY_MAX_FOLLOW_FILES", "LIVE_FEED_MAX_LINE_CHARS"]},
    {"id": "stale_lock_cleanup", "title": "Stale lock cleanup without report fanout", "keys": ["LOCK_WATCHDOG_LIGHTWEIGHT_MODE", "OPS_WATCHDOG_REFRESH_REPORTS"]},
    {"id": "watchdog_refresh_backoff", "title": "Watchdog/report refresh backoff", "keys": ["OPS_WATCHDOG_REFRESH_MAX_AGE_SECONDS", "OPS_WATCHDOG_LAUNCHD_INTERVAL_SECONDS"]},
    {"id": "paper_control_backoff", "title": "Paper control rescan backoff", "keys": ["PAPER_RUNTIME_CONTROL_REFRESH_SECONDS", "PAPER_RUNTIME_CONTROL_MAX_ROWS"]},
    {"id": "broker_context_cache_cap", "title": "Broker news/options context cache caps", "keys": ["SCHWAB_NEWS_CACHE_MAX_SYMBOLS", "SCHWAB_OPTIONS_CHAIN_CACHE_MAX_SYMBOLS"]},
    {"id": "snapshot_worker_cap", "title": "Snapshot worker cap", "keys": ["COINBASE_SNAPSHOT_MAX_WORKERS", "ASYNC_PIPELINE_WORKERS"]},
    {"id": "sqlite_temp_relief", "title": "SQLite temp/cache pressure relief", "keys": ["SQLITE_TEMP_STORE_MODE", "SQLITE_MMAP_SIZE_MB"]},
    {"id": "off_hours_heavy_only", "title": "Heavy support jobs prefer off-hours", "keys": ["OPS_HEAVY_SUPPORT_OFF_HOURS_ONLY", "MAINTENANCE_SLOT_ALLOW_DURING_MACRO_EVENT"]},
    {"id": "health_artifact_coalescer", "title": "Health artifact write coalescer", "keys": ["HEALTH_ARTIFACT_COALESCE_ENABLED", "HEALTH_ARTIFACT_MIN_WRITE_SECONDS"]},
    {"id": "report_refresh_debouncer", "title": "Report refresh debounce", "keys": ["REPORT_REFRESH_DEBOUNCE_ENABLED", "REPORT_REFRESH_DEBOUNCE_SECONDS"]},
    {"id": "collector_duty_cycle", "title": "Collector duty-cycle smoothing", "keys": ["BOT_COLLECTION_DUTY_CYCLE_ENABLED", "BOT_COLLECTION_DUTY_CYCLE_MAX_ACTIVE_RATIO"]},
    {"id": "paper_queue_jitter", "title": "Paper queue jitter smoothing", "keys": ["PAPER_TRADE_EVENT_QUEUE_JITTER_ENABLED", "PAPER_TRADE_EVENT_QUEUE_JITTER_SECONDS"]},
    {"id": "provider_failure_damper", "title": "Provider failure dampening", "keys": ["PROVIDER_FAILURE_DAMPER_ENABLED", "PROVIDER_FAILURE_DAMPER_HALF_LIFE_SECONDS"]},
    {"id": "training_research_circuit_breaker", "title": "Training/research pressure circuit breaker", "keys": ["TRAINING_RESEARCH_CIRCUIT_BREAKER_ENABLED", "TRAINING_RESEARCH_PAUSE_ON_PRESSURE"]},
    {"id": "cold_start_thin_sample", "title": "Cold-start thin sampling", "keys": ["COLD_START_COLLECTOR_THIN_SAMPLE_ENABLED", "COLD_START_COLLECTOR_SAMPLE_RATE"]},
    {"id": "launchd_jitter_stagger", "title": "Launchd jitter and stagger", "keys": ["OPS_LAUNCHD_STAGGER_ENABLED", "OPS_LAUNCHD_JITTER_SECONDS"]},
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


def _now_hour_local() -> int:
    return datetime.now().hour


def _read_health(project_root: Path, name: str) -> dict[str, Any]:
    payload = load_json(project_root / "governance" / "health" / name)
    return payload if isinstance(payload, dict) else {}


def _pressure_tier(
    *,
    runtime: dict[str, Any],
    memory: dict[str, Any],
    swap: dict[str, Any],
    global_halt: dict[str, Any],
) -> str:
    score = _safe_float(runtime.get("host_saturation_score"), 0.0)
    compute = str(runtime.get("compute_pressure_level") or "").lower()
    mem = str(runtime.get("memory_pressure_level") or "").lower()
    swap_payload = swap.get("swap_pressure") if isinstance(swap.get("swap_pressure"), dict) else {}
    swap_tier = str(swap_payload.get("tier") or swap_payload.get("raw_tier") or "").lower()
    cotenant = memory.get("co_running_session") if isinstance(memory.get("co_running_session"), dict) else {}
    cotenant_level = str(cotenant.get("level") or "").lower()
    halt = bool(global_halt.get("halt", False))
    if halt or compute == "high" or mem == "high" or swap_tier in {"survival", "pause_research", "constrained"} or score >= 82.0:
        return "deep_relief"
    if compute == "elevated" or mem == "elevated" or score >= 56.0 or cotenant_level in {"heavy_competition", "heavy"}:
        return "guarded_relief"
    if score >= 28.0 or cotenant_level in {"interactive", "developer"}:
        return "calm"
    return "observe"


def _sql_writer_coordination(backpressure_fleet: dict[str, Any]) -> dict[str, Any]:
    active_drainer = backpressure_fleet.get("active_drainer") if isinstance(backpressure_fleet.get("active_drainer"), dict) else {}
    concentration = active_drainer.get("concentration") if isinstance(active_drainer.get("concentration"), dict) else {}
    request = backpressure_fleet.get("service_request") if isinstance(backpressure_fleet.get("service_request"), dict) else {}
    env = request.get("env_overrides") if isinstance(request.get("env_overrides"), dict) else {}
    total_pending = _safe_int(concentration.get("total_pending_lines"), 0)
    top1_share = _safe_float(concentration.get("top1_share"), 0.0)
    top3_share = _safe_float(concentration.get("top3_share"), 0.0)
    concentrated = bool(concentration.get("concentrated", False)) or str(env.get("SQL_LINK_SERVICE_CONCENTRATED_CORE_DRAIN") or "").strip() == "1"
    if not concentrated and total_pending >= 5000 and (top1_share >= 0.45 or top3_share >= 0.75):
        concentrated = True
    return {
        "source": "backpressure_drainer_fleet" if backpressure_fleet else "none",
        "active_drainer": str(active_drainer.get("name") or ""),
        "concentrated_core_drain": concentrated,
        "total_pending_lines": total_pending,
        "top1_share": round(top1_share, 6),
        "top3_share": round(top3_share, 6),
    }


def _concentrated_sql_drain_overrides() -> dict[str, str]:
    return {
        "SQL_LINK_SERVICE_INTERVAL_SECONDS": "12",
        "SQL_LINK_SERVICE_HOT_MIN_INTERVAL_SECONDS": "30",
        "SQL_LINK_SERVICE_QUEUE_MIN_INTERVAL_SECONDS": "180",
        "SQL_LINK_SERVICE_HOT_BATCH_SIZE": "240000",
        "SQL_LINK_SERVICE_QUEUE_BATCH_SIZE": "180000",
        "SQL_LINK_SERVICE_CONCENTRATED_CORE_DRAIN": "1",
        "SQL_LINK_SERVICE_SHARD_LINK_TIMEOUT_SECONDS": "420",
        "SQL_LINK_SERVICE_MERGE_MAX_SECONDS_PER_CYCLE": "60",
        "SQL_LINK_SERVICE_SHARD_TRADING_STATE_CHECKPOINT_LINES": "1000",
        "SQL_LINK_SERVICE_SHARD_AGGRESSIVE_TRADING_STATE_CHECKPOINT_LINES": "1000",
        "SQL_LINK_SERVICE_SHARD_AGGRESSIVE_TRADING_MAX_LINES_PER_FILE": "12000",
    }


def _env_for_tier(tier: str) -> dict[str, str]:
    common = {
        "PRESSURE_RELIEF_ENABLED": "1",
        "PRESSURE_RELIEF_TIER": tier,
        "OPS_HEALTH_FAST_ENABLED": "1",
        "OPS_HEALTH_NO_REPORT_REFRESH": "1",
        "FOREGROUND_APP_PRESSURE_GOVERNOR_ENABLED": "1",
        "RUNTIME_COTENANT_AWARE": "1",
        "CALM_MODE_ENABLED": "1",
        "CALM_MODE_MIN_SECONDS": "900",
        "LIVE_FEED_HEAVY_TTL_ENABLED": "1",
        "LIVE_FEED_HEAVY_TTL_SECONDS": "900",
        "LIVE_FEED_HEAVY_MAX_FOLLOW_FILES": "24",
        "LIVE_FEED_HEAVY_DEFAULT_LINES": "80",
        "LIVE_FEED_HEAVY_PRESSURE_LINES": "60",
        "LIVE_FEED_MAX_LINE_CHARS": "1100",
        "LIVE_FEED_DECISION_FILE_MODE_PRESSURE": "latest_only",
        "MAINTENANCE_SLOT_QUIET_WINDOWS_ENABLED": "1",
        "MAINTENANCE_SLOT_DEFER_OUTSIDE_QUIET_WINDOW": "1",
        "MAINTENANCE_SLOT_QUIET_LOCAL_START_HOUR": "21",
        "MAINTENANCE_SLOT_QUIET_LOCAL_END_HOUR": "6",
        "MAINTENANCE_SLOT_MAX_LOAD_RATIO": "0.72",
        "MAINTENANCE_SLOT_MAX_FIVE_MIN_LOAD_RATIO": "0.62",
        "MAINTENANCE_SLOT_DEFER_WHILE_SQL_LINK_ACTIVE": "1",
        "MAINTENANCE_SLOT_ALLOW_DURING_MACRO_EVENT": "0",
        "OPS_HEAVY_SUPPORT_OFF_HOURS_ONLY": "1",
        "HEALTH_ARTIFACT_COALESCE_ENABLED": "1",
        "HEALTH_ARTIFACT_MIN_WRITE_SECONDS": "20",
        "REPORT_REFRESH_DEBOUNCE_ENABLED": "1",
        "REPORT_REFRESH_DEBOUNCE_SECONDS": "900",
        "BOT_COLLECTION_DUTY_CYCLE_ENABLED": "1",
        "BOT_COLLECTION_DUTY_CYCLE_MAX_ACTIVE_RATIO": "0.92",
        "PAPER_TRADE_EVENT_QUEUE_JITTER_ENABLED": "1",
        "PAPER_TRADE_EVENT_QUEUE_JITTER_SECONDS": "7",
        "PROVIDER_FAILURE_DAMPER_ENABLED": "1",
        "PROVIDER_FAILURE_DAMPER_HALF_LIFE_SECONDS": "900",
        "TRAINING_RESEARCH_CIRCUIT_BREAKER_ENABLED": "1",
        "TRAINING_RESEARCH_PAUSE_ON_PRESSURE": "1",
        "COLD_START_COLLECTOR_THIN_SAMPLE_ENABLED": "1",
        "COLD_START_COLLECTOR_SAMPLE_RATE": "0.50",
        "OPS_LAUNCHD_STAGGER_ENABLED": "1",
        "OPS_LAUNCHD_JITTER_SECONDS": "12",
        "OPS_WATCHDOG_REFRESH_REPORTS": "0",
        "LOCK_WATCHDOG_LIGHTWEIGHT_MODE": "1",
        "OPS_SUPPORT_JOB_NICE": "12",
        "OPS_SUPPORT_JOBS_BACKGROUND_POLICY": "1",
        "SQL_LINK_SERVICE_ADAPTIVE_WRITER_ENABLED": "1",
        "SQL_LINK_SERVICE_JSON_FILE_SYNC_MIN_INTERVAL_SECONDS": "900",
        "SQL_LINK_SERVICE_WAL_CHECKPOINT_TRIGGER_GROWTH_GB": "2.0",
        "SQLITE_TEMP_STORE_MODE": "FILE",
        "SQLITE_CACHE_SIZE_KB": "6144",
        "SQLITE_MMAP_SIZE_MB": "48",
        "BOT_OPS_SQLITE_TEMP_STORE_MODE": "FILE",
        "BOT_OPS_SQLITE_CACHE_SIZE_KB": "2048",
        "BOT_OPS_SQLITE_MMAP_SIZE_MB": "8",
        "MACRO_CAPTURE_NICE_LEVEL": "12",
        "MACRO_CAPTURE_BACKGROUND_POLICY": "1",
        "YTDLP_NICE_LEVEL": "12",
        "FFMPEG_NICE_LEVEL": "12",
        "MLX_LAZY_IMPORTS": "1",
        "MLX_INTELLIGENCE_PROFILE": "foreground_safe",
        "MLX_INTELLIGENCE_MAX_CONCURRENT_JOBS": "2",
        "PRIMARY_ML_RUNTIME_BACKEND": "mlx",
        "LIBRARY_DEFAULT_ML_BACKEND": "mlx",
        "QUANT_MODEL_RESEARCH_ONLY": "1",
        "QUANT_MODEL_MAX_WORKERS": "1",
        "QUANT_MODEL_MLX_COMPILE_ENABLED": "0",
        "REPORT_RENDER_MAX_JOBS": "1",
        "LIBRARY_REPORT_RENDER_JOBS": "1",
        "COMMAND_VALIDITY_MIN_INTERVAL_SECONDS": "3600",
        "COMMANDS_HYGIENE_MIN_INTERVAL_SECONDS": "3600",
        "DATA_SOURCE_DIVERGENCE_REFRESH_INTERVAL_SECONDS": "1200",
        "PAPER_RUNTIME_CONTROL_REFRESH_SECONDS": "300",
        "PAPER_RUNTIME_CONTROL_MAX_ROWS": "10000",
        "SCHWAB_NEWS_CACHE_MAX_SYMBOLS": "24",
        "SCHWAB_OPTIONS_CHAIN_CACHE_MAX_SYMBOLS": "24",
        "SCHWAB_QUOTE_HTTP_403_429_COUNT_AS_ANOMALY": "0",
        "SCHWAB_QUOTE_HTTP_403_429_COOLDOWN_SECONDS": "180",
        "SCHWAB_FUTURES_CONTEXT_SYMBOLS": "SPY,QQQ,GLD",
        "COINBASE_SNAPSHOT_MAX_WORKERS": "1",
        "ASYNC_PIPELINE_WORKERS": "2",
    }
    if tier == "observe":
        common.update(
            {
                "PRESSURE_RELIEF_ACTIVE": "0",
                "MAINTENANCE_SLOT_DEFER_OUTSIDE_QUIET_WINDOW": "0",
                "LIVE_FEED_HEAVY_TTL_SECONDS": "1800",
                "SQL_LINK_SERVICE_INTERVAL_SECONDS": "12",
                "SQL_LINK_SERVICE_HOT_MIN_INTERVAL_SECONDS": "30",
                "SQL_LINK_SERVICE_QUEUE_MIN_INTERVAL_SECONDS": "180",
                "SQL_LINK_SERVICE_HOT_BATCH_SIZE": "240000",
                "SQL_LINK_SERVICE_QUEUE_BATCH_SIZE": "180000",
                "SCHWAB_FUTURES_WATCH_INTERVAL_SECONDS": "12",
                "BOT_COLLECTION_DUTY_CYCLE_MAX_ACTIVE_RATIO": "1.00",
                "COLD_START_COLLECTOR_SAMPLE_RATE": "0.80",
            }
        )
    elif tier == "calm":
        common.update(
            {
                "PRESSURE_RELIEF_ACTIVE": "1",
                "SQL_LINK_SERVICE_INTERVAL_SECONDS": "30",
                "SQL_LINK_SERVICE_HOT_MIN_INTERVAL_SECONDS": "90",
                "SQL_LINK_SERVICE_QUEUE_MIN_INTERVAL_SECONDS": "360",
                "SQL_LINK_SERVICE_HOT_BATCH_SIZE": "160000",
                "SQL_LINK_SERVICE_QUEUE_BATCH_SIZE": "110000",
                "OPS_WATCHDOG_LAUNCHD_INTERVAL_SECONDS": "180",
                "OPS_WATCHDOG_REFRESH_MAX_AGE_SECONDS": "2400",
                "SCHWAB_FUTURES_WATCH_INTERVAL_SECONDS": "20",
                "BOT_COLLECTION_DUTY_CYCLE_MAX_ACTIVE_RATIO": "0.95",
                "COLD_START_COLLECTOR_SAMPLE_RATE": "0.60",
                "OPS_LAUNCHD_JITTER_SECONDS": "18",
            }
        )
    elif tier == "guarded_relief":
        common.update(
            {
                "PRESSURE_RELIEF_ACTIVE": "1",
                "SQL_LINK_SERVICE_INTERVAL_SECONDS": "75",
                "SQL_LINK_SERVICE_HOT_MIN_INTERVAL_SECONDS": "240",
                "SQL_LINK_SERVICE_QUEUE_MIN_INTERVAL_SECONDS": "900",
                "SQL_LINK_SERVICE_HOT_BATCH_SIZE": "80000",
                "SQL_LINK_SERVICE_QUEUE_BATCH_SIZE": "55000",
                "OPS_WATCHDOG_LAUNCHD_INTERVAL_SECONDS": "240",
                "OPS_WATCHDOG_REFRESH_MAX_AGE_SECONDS": "3600",
                "LIVE_FEED_HEAVY_MAX_FOLLOW_FILES": "18",
                "LIVE_FEED_HEAVY_TTL_SECONDS": "720",
                "DATA_SOURCE_DIVERGENCE_REFRESH_INTERVAL_SECONDS": "1800",
                "PAPER_RUNTIME_CONTROL_REFRESH_SECONDS": "360",
                "SCHWAB_FUTURES_WATCH_INTERVAL_SECONDS": "30",
                "BOT_COLLECTION_DUTY_CYCLE_MAX_ACTIVE_RATIO": "0.82",
                "COLD_START_COLLECTOR_SAMPLE_RATE": "0.35",
                "PAPER_TRADE_EVENT_QUEUE_JITTER_SECONDS": "11",
                "HEALTH_ARTIFACT_MIN_WRITE_SECONDS": "35",
                "REPORT_REFRESH_DEBOUNCE_SECONDS": "1500",
                "OPS_LAUNCHD_JITTER_SECONDS": "30",
            }
        )
    else:
        common.update(
            {
                "PRESSURE_RELIEF_ACTIVE": "1",
                "SQL_LINK_SERVICE_INTERVAL_SECONDS": "150",
                "SQL_LINK_SERVICE_HOT_MIN_INTERVAL_SECONDS": "600",
                "SQL_LINK_SERVICE_QUEUE_MIN_INTERVAL_SECONDS": "1800",
                "SQL_LINK_SERVICE_HOT_BATCH_SIZE": "40000",
                "SQL_LINK_SERVICE_QUEUE_BATCH_SIZE": "25000",
                "OPS_WATCHDOG_LAUNCHD_INTERVAL_SECONDS": "360",
                "OPS_WATCHDOG_REFRESH_MAX_AGE_SECONDS": "5400",
                "LIVE_FEED_HEAVY_MAX_FOLLOW_FILES": "12",
                "LIVE_FEED_HEAVY_TTL_SECONDS": "480",
                "LIVE_FEED_HEAVY_DEFAULT_LINES": "50",
                "LIVE_FEED_HEAVY_PRESSURE_LINES": "40",
                "DATA_SOURCE_DIVERGENCE_REFRESH_INTERVAL_SECONDS": "3600",
                "PAPER_RUNTIME_CONTROL_REFRESH_SECONDS": "600",
                "PAPER_RUNTIME_CONTROL_MAX_ROWS": "6000",
                "MLX_INTELLIGENCE_MAX_CONCURRENT_JOBS": "1",
                "SCHWAB_FUTURES_WATCH_INTERVAL_SECONDS": "45",
                "BOT_COLLECTION_DUTY_CYCLE_MAX_ACTIVE_RATIO": "0.68",
                "COLD_START_COLLECTOR_SAMPLE_RATE": "0.18",
                "PAPER_TRADE_EVENT_QUEUE_JITTER_SECONDS": "17",
                "HEALTH_ARTIFACT_MIN_WRITE_SECONDS": "60",
                "REPORT_REFRESH_DEBOUNCE_SECONDS": "2400",
                "OPS_LAUNCHD_JITTER_SECONDS": "45",
            }
        )
    return common


def _support_hot_stabilization_overrides(runtime: dict[str, Any], tier: str, env: dict[str, str]) -> dict[str, Any]:
    attribution = runtime.get("host_pressure_attribution") if isinstance(runtime.get("host_pressure_attribution"), dict) else {}
    support_hot = bool(attribution.get("support_jobs_hot", False))
    system_hot = bool(attribution.get("system_cotenant_hot", False))
    external_hot = bool(attribution.get("external_pressure_dominant", False))
    host_saturation = _safe_float(runtime.get("host_saturation_score"), 0.0)
    active = bool(support_hot or system_hot or external_hot or host_saturation >= 56.0)
    if not active:
        return {
            "active": False,
            "reason": "support and macOS cotenant pressure are not hot",
            "env_overrides": {},
            "policy": "only tighten support maintenance when host pressure attribution asks for it",
        }
    collector_cap = 0.35 if system_hot or external_hot else 0.45 if support_hot else 0.55
    if tier == "deep_relief":
        collector_cap = min(collector_cap, 0.22)
    current_ratio = _safe_float(env.get("BOT_COLLECTION_DUTY_CYCLE_MAX_ACTIVE_RATIO"), 1.0)
    support_nice = "16" if system_hot or external_hot else "14"
    overrides = {
        "OPS_SUPPORT_MAINTENANCE_STABILIZER_ACTIVE": "1",
        "OPS_SUPPORT_MAINTENANCE_FREEZE": "1" if system_hot or external_hot else "0",
        "OPS_SUPPORT_MAINTENANCE_COOLDOWN_SECONDS": "1800" if system_hot or external_hot else "900",
        "OPS_SUPPORT_JOB_NICE": support_nice,
        "BOT_COLLECTION_DUTY_CYCLE_MAX_ACTIVE_RATIO": f"{min(current_ratio, collector_cap):.2f}",
        "COLD_START_COLLECTOR_SAMPLE_RATE": "0.12" if tier == "deep_relief" else "0.25",
        "REPORT_REFRESH_DEBOUNCE_SECONDS": "2400",
        "HEALTH_ARTIFACT_MIN_WRITE_SECONDS": "60",
        "COMMAND_VALIDITY_MIN_INTERVAL_SECONDS": "5400",
        "COMMANDS_HYGIENE_MIN_INTERVAL_SECONDS": "5400",
        "DATA_SOURCE_DIVERGENCE_REFRESH_INTERVAL_SECONDS": "3600",
        "PAPER_RUNTIME_CONTROL_REFRESH_SECONDS": "600",
        "OPS_HEAVY_SUPPORT_OFF_HOURS_ONLY": "1",
        "TRAINING_RESEARCH_PAUSE_ON_PRESSURE": "1",
    }
    reasons = []
    if system_hot:
        reasons.append("macos_system_cotenant_hot")
    if external_hot:
        reasons.append("external_or_foreground_cotenant_hot")
    if support_hot:
        reasons.append("support_maintenance_hot")
    if host_saturation >= 56.0:
        reasons.append("host_saturation_guarded")
    return {
        "active": True,
        "reason": ",".join(reasons) or "host_pressure_guarded",
        "host_saturation_score": round(host_saturation, 3),
        "support_jobs_hot": support_hot,
        "system_cotenant_hot": system_hot,
        "external_pressure_dominant": external_hot,
        "env_overrides": overrides,
        "policy": "freeze_or_stretch_support_maintenance_before_widening_pcores_collectors_or_training",
    }


def _write_env_override(path: Path, env: dict[str, str]) -> bool:
    def assignment(name: str, value: str) -> str:
        return f"{name}={shlex.quote(str(value))}"

    lines = [
        "# Auto-managed by scripts/ops/pressure_relief_control.py",
    ]
    for key in sorted(env):
        lines.append(assignment(key, env[key]))
    content = "\n".join(lines) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    current = path.read_text(encoding="utf-8") if path.exists() else ""
    if current == content:
        return False
    path.write_text(content, encoding="utf-8")
    return True


def _process_exists(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except Exception:
        return False


def _renice_matching_processes(env: dict[str, str], *, apply: bool) -> dict[str, Any]:
    patterns = SUPPORT_RENICE_PATTERNS
    try:
        completed = subprocess.run(["ps", "-axo", "pid,command"], check=False, capture_output=True, text=True, timeout=5)
    except Exception as exc:
        return {"attempted": 0, "changed": 0, "error": str(exc)}
    matches: list[dict[str, Any]] = []
    for raw_line in (completed.stdout or "").splitlines():
        parts = raw_line.strip().split(None, 1)
        if len(parts) != 2:
            continue
        pid = _safe_int(parts[0], 0)
        command = parts[1]
        if pid <= 0 or not any(pattern in command for pattern in patterns):
            continue
        nice_level = env.get("MACRO_CAPTURE_NICE_LEVEL", "12") if ("yt-dlp" in command or "ffmpeg" in command) else env.get("OPS_SUPPORT_JOB_NICE", "12")
        row = {"pid": pid, "nice": nice_level, "command_excerpt": command[:220], "changed": False}
        if apply and _process_exists(pid):
            rc = subprocess.run(["renice", "-n", str(nice_level), "-p", str(pid)], check=False, capture_output=True, text=True, timeout=5).returncode
            row["changed"] = rc == 0
        matches.append(row)
    return {"attempted": len(matches), "changed": sum(1 for row in matches if row.get("changed")), "matches": matches[:12]}


def build_payload(project_root: Path = PROJECT_ROOT) -> dict[str, Any]:
    runtime = _read_health(project_root, "runtime_throttle_control_latest.json")
    memory = _read_health(project_root, "memory_efficiency_control_latest.json")
    swap = _read_health(project_root, "swap_pressure_governor_latest.json")
    global_halt = _read_health(project_root, "global_halt_auto_clear_latest.json")
    if not global_halt:
        global_halt = _read_health(project_root, "global_killswitch_latest.json")
    ingestion = _read_health(project_root, "ingestion_storage_control_latest.json")
    backpressure_fleet = _read_health(project_root, "backpressure_drainer_fleet_latest.json")
    feed_marker = _read_health(project_root, "live_feed_heavy_view_latest.json")
    tier = _pressure_tier(runtime=runtime, memory=memory, swap=swap, global_halt=global_halt)
    sql_writer_coordination = _sql_writer_coordination(backpressure_fleet)
    env = _env_for_tier(tier)
    support_stabilization = _support_hot_stabilization_overrides(runtime, tier, env)
    env.update({str(key): str(value) for key, value in (support_stabilization.get("env_overrides") or {}).items()})
    if bool(sql_writer_coordination.get("concentrated_core_drain", False)):
        env.update(_concentrated_sql_drain_overrides())
    hour = _now_hour_local()
    quiet_start = _safe_int(env.get("MAINTENANCE_SLOT_QUIET_LOCAL_START_HOUR"), 21)
    quiet_end = _safe_int(env.get("MAINTENANCE_SLOT_QUIET_LOCAL_END_HOUR"), 6)
    in_quiet_window = hour >= quiet_start or hour < quiet_end if quiet_start > quiet_end else quiet_start <= hour < quiet_end
    feed_pid = _safe_int(feed_marker.get("pid"), 0)
    feed_ttl = _safe_int(env.get("LIVE_FEED_HEAVY_TTL_SECONDS"), 0)
    feed_started = _safe_float(feed_marker.get("started_epoch"), 0.0)
    feed_age = max(time.time() - feed_started, 0.0) if feed_started > 0 else None
    return {
        "timestamp_utc": iso_now(),
        "schema_version": 1,
        "ok": tier in {"observe", "calm", "guarded_relief"},
        "overall_status": "ready" if tier in {"observe", "calm"} else ("degraded" if tier == "guarded_relief" else "blocked"),
        "tier": tier,
        "active": env.get("PRESSURE_RELIEF_ACTIVE") == "1",
        "host_saturation_score": _safe_float(runtime.get("host_saturation_score"), 0.0),
        "compute_pressure_level": str(runtime.get("compute_pressure_level") or "unknown"),
        "memory_pressure_level": str(runtime.get("memory_pressure_level") or "unknown"),
        "swap_pressure": swap.get("swap_pressure") if isinstance(swap.get("swap_pressure"), dict) else {},
        "storage_pressure": {
            "severity": ingestion.get("severity"),
            "pressure_index": _safe_float(ingestion.get("pressure_index"), 0.0),
            "backpressure": ingestion.get("backpressure") if isinstance(ingestion.get("backpressure"), dict) else {},
            "sql_writer_coordination": sql_writer_coordination,
        },
        "support_maintenance_stabilization": support_stabilization,
        "quiet_window": {
            "enabled": env.get("MAINTENANCE_SLOT_QUIET_WINDOWS_ENABLED") == "1",
            "local_hour": hour,
            "start_hour": quiet_start,
            "end_hour": quiet_end,
            "in_window": bool(in_quiet_window),
        },
        "heavy_feed_ttl": {
            "enabled": env.get("LIVE_FEED_HEAVY_TTL_ENABLED") == "1",
            "ttl_seconds": feed_ttl,
            "active_marker": bool(feed_marker.get("active", False)),
            "pid": feed_pid,
            "pid_live": _process_exists(feed_pid),
            "age_seconds": round(feed_age, 3) if feed_age is not None else None,
            "expired": bool(feed_age is not None and feed_ttl > 0 and feed_age > feed_ttl),
        },
        "pressure_relief_items": [
            {**item, "enabled": all(key in env for key in item["keys"])}
            for item in PRESSURE_RELIEF_ITEMS
        ],
        "env_override_count": len(env),
        "recommended_env_overrides": env,
        "override_path": str(DEFAULT_OVERRIDE_PATH),
        "infrastructure_bots": [
            "pressure_relief_governor",
            "foreground_app_pressure_infrabot",
            "maintenance_quiet_window_infrabot",
            "heavy_feed_ttl_infrabot",
            "adaptive_sql_pressure_infrabot",
            "health_artifact_coalescer_infrabot",
            "collector_duty_cycle_infrabot",
            "paper_queue_jitter_infrabot",
            "provider_failure_damper_infrabot",
            "training_research_circuit_breaker_infrabot",
        ],
        "recommended_commands": [
            ["./scripts/ops/opsctl.sh", "pressure-relief", "--apply", "--json"],
            ["./scripts/ops/opsctl.sh", "health-fast", "--json"],
            ["./scripts/ops/opsctl.sh", "runtime-throttle", "--apply", "--json"],
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Pressure relief controller for foreground-safe bot operation.")
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--out-file", default=str(DEFAULT_OUT_PATH))
    parser.add_argument("--override-file", default=str(DEFAULT_OVERRIDE_PATH))
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    payload = build_payload(project_root)
    if args.apply:
        override_path = Path(args.override_file).expanduser()
        env = payload.get("recommended_env_overrides") if isinstance(payload.get("recommended_env_overrides"), dict) else {}
        payload["apply_result"] = {
            "applied": True,
            "override_path": str(override_path),
            "override_changed": _write_env_override(override_path, {str(k): str(v) for k, v in env.items()}),
            "process_nice": _renice_matching_processes({str(k): str(v) for k, v in env.items()}, apply=True),
        }
    else:
        payload["apply_result"] = {"applied": False}
    write_payload(Path(args.out_file).expanduser(), payload)
    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        print(
            "pressure_relief "
            f"overall_status={payload.get('overall_status')} "
            f"tier={payload.get('tier')} "
            f"active={int(bool(payload.get('active')))} "
            f"items={len(payload.get('pressure_relief_items') or [])}"
        )
    return 0 if payload.get("overall_status") in {"ready", "degraded"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
