#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shlex
import signal
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from scripts import resource_guard as resource_src
    from scripts.ops import memory_efficiency_control as memory_src
    from scripts.ops import runtime_throttle_control as runtime_src
    from scripts.ops.long_runtime_common import load_json, parse_iso_utc, write_payload
else:
    from .. import resource_guard as resource_src
    from . import memory_efficiency_control as memory_src
    from . import runtime_throttle_control as runtime_src
    from .long_runtime_common import PROJECT_ROOT, load_json, parse_iso_utc, write_payload


DEFAULT_OUT_PATH = PROJECT_ROOT / "governance" / "health" / "swap_pressure_governor_latest.json"
DEFAULT_STATE_PATH = PROJECT_ROOT / "governance" / "health" / "swap_pressure_governor_state.json"
DEFAULT_OVERRIDE_PATH = PROJECT_ROOT / "config" / ".env.swap_pressure_override"
DEFAULT_MEMORY_OVERRIDE_PATH = PROJECT_ROOT / "config" / ".env.memory_efficiency_override"
DEFAULT_RUNTIME_OVERRIDE_PATH = PROJECT_ROOT / "config" / ".env.runtime_resource_guard_override"
DEFAULT_REGISTRY_PATH = PROJECT_ROOT / "master_bot_registry.json"

HEAVY_RESEARCH_PATTERNS = [
    "scripts/run_all_sleeves.py",
    "scripts/run_parallel_shadows.py",
    "scripts/run_parallel_aggressive_modes.py",
    "scripts/run_dividend_shadow.py",
    "scripts/run_bond_shadow.py",
    "scripts/run_fx_shadow.py",
    "scripts/run_specialized_sleeve_shadow.py",
    "scripts/run_shadow_training_loop.py",
    "scripts/retrain_orchestrator.py",
    "scripts/retrain_lane_scheduler.py",
    "scripts/weekly_retrain.py",
    "scripts/retrain_daily_small_batch.sh",
    "scripts/retrain_weekly_full_sweep.sh",
    "scripts/ops/run_full_retrain_overnight_once.sh",
    "scripts/ops/quant_model_control.py",
    "scripts/quant_models/gpu_mc_sim.py",
    "scripts/quant_models/pricing_grad.py",
    "scripts/quant_models/kalman_parallel.py",
    "scripts/ops/mlx_runtime_audit.py",
    "scripts/ops/mlx_library_upgrade.py",
    "scripts/data_source_divergence_bot.py",
    "scripts/build_one_numbers_report.py",
    "scripts/ops/source_verification_report.py",
    "scripts/ops/strategy_inventory_report.py",
    "scripts/sql_hot_retention.py",
    "scripts/sql_queue_retention.py",
    "scripts/data_retention_policy.py",
    "project_timeline_report.py",
    "report-bundle-pdf-open",
]

RESTART_CANDIDATE_MARKERS = [
    ("PyCharm", "PyCharm.app"),
    ("Google Chrome", "Google Chrome"),
    ("Chrome", "Chrome Helper"),
    ("Codex", "Codex Helper"),
    ("Codex", "/Applications/Codex.app"),
    ("Cursor", "Cursor"),
    ("Code", "Code Helper"),
    ("Safari", "Safari"),
    ("Arc", "Arc"),
    ("UTM", "UTM"),
]

SWAP_TIER_ORDER = {
    "normal": 0,
    "calm": 1,
    "constrained": 2,
    "pause_research": 3,
    "survival": 4,
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


def _thresholds() -> dict[str, float]:
    return {
        "calm_swap_gb": float(os.getenv("SWAP_PRESSURE_CALM_GB", "10")),
        "constrained_swap_gb": float(os.getenv("SWAP_PRESSURE_CONSTRAINED_GB", "14")),
        "pause_research_swap_gb": float(os.getenv("SWAP_PRESSURE_PAUSE_RESEARCH_GB", "18")),
        "survival_swap_gb": float(os.getenv("SWAP_PRESSURE_SURVIVAL_GB", "22")),
        "resume_swap_gb": float(os.getenv("SWAP_PRESSURE_RESUME_GB", "8")),
        "restart_candidate_rss_mb": float(os.getenv("SWAP_PRESSURE_RESTART_CANDIDATE_RSS_MB", "512")),
        "reboot_recommend_swap_gb": float(os.getenv("SWAP_PRESSURE_REBOOT_RECOMMEND_GB", "18")),
        "reboot_recommend_persistent_minutes": float(os.getenv("SWAP_PRESSURE_REBOOT_RECOMMEND_MINUTES", "30")),
        "stale_swap_relief_enabled": float(os.getenv("SWAP_PRESSURE_STALE_SWAP_RELIEF_ENABLED", "0")),
        "stale_swap_relief_free_pct": float(os.getenv("SWAP_PRESSURE_STALE_SWAP_RELIEF_FREE_PCT", "85")),
        "stale_swap_relief_compressor_gb": float(os.getenv("SWAP_PRESSURE_STALE_SWAP_RELIEF_COMPRESSOR_GB", "1.0")),
    }


def _swap_tier(swap_used_gb: float, thresholds: dict[str, float]) -> str:
    if swap_used_gb >= thresholds["survival_swap_gb"]:
        return "survival"
    if swap_used_gb >= thresholds["pause_research_swap_gb"]:
        return "pause_research"
    if swap_used_gb >= thresholds["constrained_swap_gb"]:
        return "constrained"
    if swap_used_gb >= thresholds["calm_swap_gb"]:
        return "calm"
    return "normal"


def _tier_status(tier: str) -> str:
    if tier == "survival":
        return "blocked"
    if tier in {"constrained", "pause_research"}:
        return "degraded"
    if tier == "calm":
        return "needs_work"
    return "ready"


def _pressure_active(tier: str) -> bool:
    return SWAP_TIER_ORDER.get(tier, 0) >= SWAP_TIER_ORDER["calm"]


def _research_pause_active(tier: str) -> bool:
    return SWAP_TIER_ORDER.get(tier, 0) >= SWAP_TIER_ORDER["pause_research"]


def _stale_swap_allocation_relief(
    resource_snapshot: dict[str, Any],
    *,
    raw_tier: str,
    thresholds: dict[str, float],
) -> dict[str, Any]:
    memory_kind = str(resource_snapshot.get("memory_pressure_kind") or "").strip().lower()
    free_pct = _safe_float(resource_snapshot.get("memory_free_pct"), 0.0)
    compressor_gb = _safe_float(resource_snapshot.get("compressor_gb"), 0.0)
    throttled_pages = _safe_float(resource_snapshot.get("pages_throttled"), 0.0)
    enabled = bool(thresholds.get("stale_swap_relief_enabled", 0.0) >= 1.0)
    active = bool(
        enabled
        and SWAP_TIER_ORDER.get(raw_tier, 0) >= SWAP_TIER_ORDER["pause_research"]
        and memory_kind in {"swap_only", "swap_only_with_headroom", "none", ""}
        and free_pct >= thresholds["stale_swap_relief_free_pct"]
        and compressor_gb <= thresholds["stale_swap_relief_compressor_gb"]
        and throttled_pages <= 0
    )
    return {
        "active": active,
        "enabled": enabled,
        "raw_tier": raw_tier,
        "relieved_tier": "calm" if active else raw_tier,
        "memory_free_pct": round(free_pct, 3),
        "compressor_gb": round(compressor_gb, 3),
        "pages_throttled": round(throttled_pages, 3),
        "reason": (
            "swap_file_allocated_but_memory_headroom_is_healthy"
            if active
            else "not_applicable"
        ),
    }


def _refresh_resource_guard(project_root: Path) -> dict[str, Any]:
    snapshot = resource_src.build_snapshot(project_root)
    memory_state, memory_reasons, memory_thresholds = resource_src._memory_pressure_state(snapshot)
    snapshot.update(
        {
            "resource_guard_profile": "swap_pressure_governor",
            "resource_guard_ok": True,
            "resource_guard_reasons": [],
            "memory_pressure_state": memory_state,
            "memory_pressure_reasons": memory_reasons,
            "memory_pressure_kind": resource_src._memory_pressure_kind(snapshot, memory_state, memory_reasons),
            "memory_pressure_thresholds": memory_thresholds,
        }
    )
    write_payload(project_root / "governance" / "health" / "resource_guard_latest.json", snapshot)
    return snapshot


def _swap_env_overrides(tier: str, swap_used_gb: float) -> dict[str, str]:
    base = {
        "SWAP_PRESSURE_GOVERNOR_ACTIVE": "1" if _pressure_active(tier) else "0",
        "SWAP_PRESSURE_TIER": tier,
        "SWAP_PRESSURE_SWAP_USED_GB": f"{swap_used_gb:.3f}",
        "BOT_MLX_OPTIONAL": "1",
        "MLX_LAZY_IMPORTS": "1",
        "QUANT_MODEL_LAZY_LIBRARY_IMPORTS": "1",
    }
    if tier == "normal":
        base.update(
            {
                "SWAP_PRESSURE_HEAVY_RESEARCH_PAUSED": "0",
                "TRAINING_RUNTIME_PAUSED_FOR_SWAP": "0",
                "AUTO_RETRAIN_PAUSED_FOR_SWAP": "0",
                "QUANT_RESEARCH_PAUSED_FOR_SWAP": "0",
                "MLX_RESEARCH_PAUSED_FOR_SWAP": "0",
                "REPORT_BUILD_PAUSED_FOR_SWAP": "0",
                "RETENTION_MAINTENANCE_PAUSED_FOR_SWAP": "0",
            }
        )
        return base

    base.update(
        {
            "SQLITE_TEMP_STORE_MODE": "FILE",
            "BOT_OPS_SQLITE_TEMP_STORE_MODE": "FILE",
            "BOT_MLX_OPTIONAL": "1",
            "MLX_METAL_JIT": "0",
            "QUANT_MODEL_MLX_COMPILE_ENABLED": "0",
            "QUANT_MODEL_RESEARCH_ONLY": "1",
            "QUANT_MODEL_MAX_WORKERS": "1",
            "AUTO_RETRAIN_SWAP_SOFT_MAX_GB": "6",
            "AUTO_RETRAIN_SWAP_IGNORE_IF_FREE_PCT_AT_LEAST": "92",
            "MEMORY_THROTTLE_SWAP_SOFT_MAX_GB": "6",
            "MEMORY_THROTTLE_SWAP_IGNORE_IF_FREE_PCT_AT_LEAST": "92",
        }
    )
    if tier == "calm":
        base.update(
            {
                "COINBASE_SNAPSHOT_MAX_WORKERS": "1",
                "ASYNC_PIPELINE_WORKERS": "2",
                "RUNTIME_FEATURE_CACHE_MAX_ENTRIES": "48",
                "RUNTIME_SLOW_BOT_CACHE_MAX_SYMBOLS": "16",
                "SCHWAB_NEWS_CACHE_MAX_SYMBOLS": "24",
                "SCHWAB_OPTIONS_CHAIN_CACHE_MAX_SYMBOLS": "24",
                "RUNTIME_TRAIN_BATCH_SIZE_CAP": "32",
                "RUNTIME_TRAIN_MAX_SAMPLES": "6000",
                "ONE_NUMBERS_REFRESH_INTERVAL_SECONDS": "1200",
                "INGESTION_BACKPRESSURE_REFRESH_INTERVAL_SECONDS": "420",
                "TOP_BOT_PAPER_TRADING_TOP_N": "2",
                "TOP_BOT_PAPER_TRADING_OPTIONS_TOP_N": "0",
            }
        )
    elif tier == "constrained":
        base.update(
            {
                "SQL_LINK_SERVICE_INTERVAL_SECONDS": "180",
                "SQL_LINK_SERVICE_JSON_FILE_SYNC_MIN_INTERVAL_SECONDS": "900",
                "SQL_LINK_SERVICE_HOT_MIN_INTERVAL_SECONDS": "720",
                "SQL_LINK_SERVICE_QUEUE_MIN_INTERVAL_SECONDS": "2400",
                "SQL_LINK_SERVICE_HOT_BATCH_SIZE": "30000",
                "SQL_LINK_SERVICE_QUEUE_BATCH_SIZE": "20000",
                "COINBASE_SNAPSHOT_MAX_WORKERS": "1",
                "COINBASE_CACHE_MAX_ENTRIES": "64",
                "ASYNC_PIPELINE_WORKERS": "1",
                "RUNTIME_FEATURE_CACHE_MAX_ENTRIES": "32",
                "RUNTIME_SLOW_BOT_CACHE_MAX_SYMBOLS": "8",
                "SCHWAB_NEWS_CACHE_MAX_SYMBOLS": "16",
                "SCHWAB_OPTIONS_CHAIN_CACHE_MAX_SYMBOLS": "16",
                "RUNTIME_TRAIN_BATCH_SIZE_CAP": "24",
                "RUNTIME_TRAIN_MAX_SAMPLES": "4000",
                "TRADE_BEHAVIOR_BATCH_SIZE": "384",
                "ONE_NUMBERS_REFRESH_INTERVAL_SECONDS": "1800",
                "INGESTION_BACKPRESSURE_REFRESH_INTERVAL_SECONDS": "720",
                "DATA_SOURCE_DIVERGENCE_REFRESH_INTERVAL_SECONDS": "1200",
                "TOP_BOT_PAPER_TRADING_TOP_N": "1",
                "TOP_BOT_PAPER_TRADING_OPTIONS_TOP_N": "0",
            }
        )
    elif tier == "pause_research":
        base.update(
            {
                "SQL_LINK_SERVICE_INTERVAL_SECONDS": "240",
                "SQL_LINK_SERVICE_JSON_FILE_SYNC_MIN_INTERVAL_SECONDS": "1200",
                "SQL_LINK_SERVICE_AUTO_HOT_RETENTION": "0",
                "SQL_LINK_SERVICE_AUTO_QUEUE_RETENTION": "0",
                "SQL_LINK_SERVICE_AUTO_LOCAL_FALLBACK_PRUNE": "0",
                "COINBASE_SNAPSHOT_MAX_WORKERS": "1",
                "COINBASE_CACHE_MAX_ENTRIES": "48",
                "ASYNC_PIPELINE_WORKERS": "1",
                "RUNTIME_FEATURE_CACHE_MAX_ENTRIES": "24",
                "RUNTIME_SLOW_BOT_CACHE_MAX_SYMBOLS": "6",
                "SCHWAB_NEWS_CACHE_MAX_SYMBOLS": "12",
                "SCHWAB_OPTIONS_CHAIN_CACHE_MAX_SYMBOLS": "12",
                "RUNTIME_TRAIN_BATCH_SIZE_CAP": "16",
                "RUNTIME_TRAIN_MAX_SAMPLES": "2500",
                "TRADE_BEHAVIOR_BATCH_SIZE": "256",
                "ONE_NUMBERS_REFRESH_INTERVAL_SECONDS": "2400",
                "INGESTION_BACKPRESSURE_REFRESH_INTERVAL_SECONDS": "900",
                "DATA_SOURCE_DIVERGENCE_REFRESH_INTERVAL_SECONDS": "1800",
                "TOP_BOT_PAPER_TRADING_TOP_N": "1",
                "TOP_BOT_PAPER_TRADING_OPTIONS_TOP_N": "0",
            }
        )
    else:
        base.update(
            {
                "SQL_LINK_SERVICE_INTERVAL_SECONDS": "300",
                "SQL_LINK_SERVICE_JSON_FILE_SYNC_MIN_INTERVAL_SECONDS": "1800",
                "SQL_LINK_SERVICE_AUTO_HOT_RETENTION": "0",
                "SQL_LINK_SERVICE_AUTO_QUEUE_RETENTION": "0",
                "SQL_LINK_SERVICE_AUTO_LOCAL_FALLBACK_PRUNE": "0",
                "COINBASE_SNAPSHOT_MAX_WORKERS": "1",
                "COINBASE_CACHE_MAX_ENTRIES": "32",
                "ASYNC_PIPELINE_WORKERS": "1",
                "RUNTIME_FEATURE_CACHE_MAX_ENTRIES": "16",
                "RUNTIME_SLOW_BOT_CACHE_MAX_SYMBOLS": "4",
                "SCHWAB_NEWS_CACHE_MAX_SYMBOLS": "8",
                "SCHWAB_OPTIONS_CHAIN_CACHE_MAX_SYMBOLS": "8",
                "RUNTIME_TRAIN_BATCH_SIZE_CAP": "8",
                "RUNTIME_TRAIN_MAX_SAMPLES": "1000",
                "TRADE_BEHAVIOR_BATCH_SIZE": "128",
                "ONE_NUMBERS_REFRESH_INTERVAL_SECONDS": "3600",
                "INGESTION_BACKPRESSURE_REFRESH_INTERVAL_SECONDS": "1200",
                "DATA_SOURCE_DIVERGENCE_REFRESH_INTERVAL_SECONDS": "2400",
                "DATA_COLLECTION_RESOURCE_GUARD_MODE": "swap_survival",
                "DATA_COLLECTION_RESOURCE_SAMPLE_RATE": "0.10",
                "DATA_COLLECTION_RESOURCE_CAPTURE_MODE": "thin_sample",
                "TOP_BOT_PAPER_TRADING_TOP_N": "1",
                "TOP_BOT_PAPER_TRADING_OPTIONS_TOP_N": "0",
            }
        )

    pause = "1" if _research_pause_active(tier) else "0"
    base.update(
        {
            "SWAP_PRESSURE_HEAVY_RESEARCH_PAUSED": pause,
            "TRAINING_RUNTIME_PAUSED_FOR_SWAP": pause,
            "AUTO_RETRAIN_PAUSED_FOR_SWAP": pause,
            "QUANT_RESEARCH_PAUSED_FOR_SWAP": pause,
            "MLX_RESEARCH_PAUSED_FOR_SWAP": pause,
            "REPORT_BUILD_PAUSED_FOR_SWAP": pause,
            "RETENTION_MAINTENANCE_PAUSED_FOR_SWAP": pause,
        }
    )
    return base


def _write_override(path: Path, overrides: dict[str, str]) -> bool:
    def assignment(name: str, value: str) -> str:
        return f"{name}={shlex.quote(str(value))}"

    lines = ["# Auto-managed by scripts/ops/swap_pressure_governor.py"]
    for key, value in sorted(overrides.items()):
        lines.append(assignment(key, value))
    content = "\n".join(lines) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    current = path.read_text(encoding="utf-8") if path.exists() else ""
    if current == content:
        return False
    path.write_text(content, encoding="utf-8")
    return True


def _pgrep_matching_pids(pattern: str) -> list[int]:
    try:
        completed = subprocess.run(["pgrep", "-f", pattern], check=False, capture_output=True, text=True, timeout=5)
    except Exception:
        return []
    if completed.returncode not in {0, 1}:
        return []
    out: list[int] = []
    self_pid = os.getpid()
    for raw_line in (completed.stdout or "").splitlines():
        pid = _safe_int(raw_line.strip(), 0)
        if pid > 0 and pid != self_pid and pid not in out:
            out.append(pid)
    return out


def _command_for_pid(pid: int) -> str:
    try:
        completed = subprocess.run(["ps", "-p", str(pid), "-o", "command="], check=False, capture_output=True, text=True, timeout=3)
    except Exception:
        return ""
    return (completed.stdout or "").strip()


def _pause_heavy_research(tier: str, *, apply: bool, patterns: list[str] | None = None) -> dict[str, Any]:
    active = _research_pause_active(tier)
    selected_patterns = patterns or HEAVY_RESEARCH_PATTERNS
    matches: list[dict[str, Any]] = []
    terminated: list[dict[str, Any]] = []
    seen: set[int] = set()
    if active:
        for pattern in selected_patterns:
            for pid in _pgrep_matching_pids(pattern):
                if pid in seen:
                    continue
                seen.add(pid)
                command = _command_for_pid(pid)
                matches.append({"pid": pid, "pattern": pattern, "command": command[:500]})
                if apply:
                    try:
                        os.kill(pid, signal.SIGTERM)
                        terminated.append({"pid": pid, "pattern": pattern, "ok": True})
                    except Exception as exc:
                        terminated.append({"pid": pid, "pattern": pattern, "ok": False, "error": str(exc)})
    return {
        "active": active,
        "apply": bool(apply),
        "action": "terminate_optional_heavy_jobs" if active and apply else ("observe" if active else "none"),
        "patterns": selected_patterns,
        "match_count": len(matches),
        "matches": matches,
        "terminated_count": sum(1 for row in terminated if bool(row.get("ok", False))),
        "terminated": terminated,
    }


def _parse_process_rows() -> list[dict[str, Any]]:
    try:
        completed = subprocess.run(
            ["ps", "-axo", "pid,pcpu,pmem,rss,command"],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except Exception:
        return []
    rows: list[dict[str, Any]] = []
    for raw_line in (completed.stdout or "").splitlines():
        line = raw_line.strip()
        if not line or line.lower().startswith("pid "):
            continue
        parts = line.split(None, 4)
        if len(parts) < 5:
            continue
        pid, cpu, mem, rss, command = parts
        rows.append(
            {
                "pid": _safe_int(pid, 0),
                "cpu_percent": round(_safe_float(cpu), 3),
                "mem_percent": round(_safe_float(mem), 3),
                "rss_mb": round(_safe_float(rss) / 1024.0, 3),
                "command": command,
            }
        )
    return rows


def _restart_advisory(tier: str, thresholds: dict[str, float]) -> dict[str, Any]:
    candidates: list[dict[str, Any]] = []
    threshold_mb = thresholds["restart_candidate_rss_mb"]
    if SWAP_TIER_ORDER.get(tier, 0) < SWAP_TIER_ORDER["constrained"]:
        return {
            "active": False,
            "safe_action": "none",
            "candidate_threshold_mb": threshold_mb,
            "candidates": [],
        }
    for row in _parse_process_rows():
        command = str(row.get("command") or "")
        lowered = command.lower()
        app_name = ""
        for label, marker in RESTART_CANDIDATE_MARKERS:
            if marker.lower() in lowered:
                app_name = label
                break
        if not app_name or _safe_float(row.get("rss_mb"), 0.0) < threshold_mb:
            continue
        candidates.append(
            {
                "app": app_name,
                "pid": row.get("pid"),
                "rss_mb": row.get("rss_mb"),
                "cpu_percent": row.get("cpu_percent"),
                "command": command[:500],
                "safe_action": "operator_restart_when_convenient",
            }
        )
    candidates.sort(key=lambda item: _safe_float(item.get("rss_mb"), 0.0), reverse=True)
    return {
        "active": bool(candidates),
        "safe_action": "notify_only_no_force_quit",
        "candidate_threshold_mb": threshold_mb,
        "candidates": candidates[:8],
    }


def _state_contract(
    state: dict[str, Any],
    *,
    now: datetime,
    tier: str,
    swap_used_gb: float,
    thresholds: dict[str, float],
) -> dict[str, Any]:
    previous_tier = str(state.get("current_tier") or "normal").strip().lower()
    pressure_started_at = parse_iso_utc(state.get("pressure_started_utc"))
    if _pressure_active(tier):
        if pressure_started_at is None or not _pressure_active(previous_tier):
            pressure_started_at = now
    else:
        pressure_started_at = None
    persistent_minutes = 0.0
    if pressure_started_at is not None:
        persistent_minutes = max((now - pressure_started_at).total_seconds() / 60.0, 0.0)
    return {
        "previous_tier": previous_tier,
        "current_tier": tier,
        "pressure_started_utc": pressure_started_at.isoformat() if pressure_started_at else "",
        "persistent_pressure_minutes": round(persistent_minutes, 3),
        "recovered_below_resume": swap_used_gb < thresholds["resume_swap_gb"],
    }


def _reboot_advisory(
    *,
    tier: str,
    swap_used_gb: float,
    expansion_session: dict[str, Any],
    state_contract: dict[str, Any],
    thresholds: dict[str, float],
) -> dict[str, Any]:
    expansion_level = str(expansion_session.get("pressure_level") or "normal").strip().lower()
    persistent_minutes = _safe_float(state_contract.get("persistent_pressure_minutes"), 0.0)
    active = bool(
        expansion_level == "massive"
        and swap_used_gb >= thresholds["reboot_recommend_swap_gb"]
        and (
            persistent_minutes >= thresholds["reboot_recommend_persistent_minutes"]
            or SWAP_TIER_ORDER.get(tier, 0) >= SWAP_TIER_ORDER["survival"]
        )
    )
    return {
        "active": active,
        "safe_action": "operator_reboot_when_done_not_automatic",
        "reason": (
            "massive_expansion_persistent_swap_pressure"
            if active
            else "not_needed"
        ),
        "persistent_pressure_minutes": persistent_minutes,
        "threshold_minutes": thresholds["reboot_recommend_persistent_minutes"],
    }


def _notification(
    *,
    now: datetime,
    tier: str,
    previous_tier: str,
    swap_used_gb: float,
    restart_advisory: dict[str, Any],
    reboot_advisory: dict[str, Any],
) -> dict[str, Any]:
    if bool(reboot_advisory.get("active", False)):
        return {
            "timestamp_utc": now.isoformat(),
            "event": "swap_pressure_reboot_recommended",
            "severity": "warn",
            "current_tier": tier,
            "previous_tier": previous_tier,
            "swap_used_gb": round(swap_used_gb, 3),
            "message": f"Swap pressure is still high at {swap_used_gb:.1f} GB after a huge expansion; reboot when your work is saved.",
        }
    if bool(restart_advisory.get("active", False)):
        candidates = restart_advisory.get("candidates") if isinstance(restart_advisory.get("candidates"), list) else []
        app = str((candidates[0] or {}).get("app") or "large app") if candidates else "large app"
        return {
            "timestamp_utc": now.isoformat(),
            "event": "swap_pressure_restart_advisory",
            "severity": "warn",
            "current_tier": tier,
            "previous_tier": previous_tier,
            "swap_used_gb": round(swap_used_gb, 3),
            "message": f"Swap pressure is {tier} at {swap_used_gb:.1f} GB; restart {app} when convenient to release memory.",
        }
    if tier != previous_tier and _pressure_active(tier):
        return {
            "timestamp_utc": now.isoformat(),
            "event": "swap_pressure_downshifted",
            "severity": "warn" if tier in {"pause_research", "survival"} else "info",
            "current_tier": tier,
            "previous_tier": previous_tier,
            "swap_used_gb": round(swap_used_gb, 3),
            "message": f"Swap pressure moved to {tier}; bot stack downshifted automatically.",
        }
    if tier == "normal" and _pressure_active(previous_tier):
        return {
            "timestamp_utc": now.isoformat(),
            "event": "swap_pressure_cleared",
            "severity": "info",
            "current_tier": tier,
            "previous_tier": previous_tier,
            "swap_used_gb": round(swap_used_gb, 3),
            "message": "Swap pressure cleared; bot stack can gradually resume normal budgets.",
        }
    return {}


def build_payload(
    project_root: Path = PROJECT_ROOT,
    *,
    apply: bool = False,
    state_path: Path = DEFAULT_STATE_PATH,
    override_path: Path = DEFAULT_OVERRIDE_PATH,
    memory_override_path: Path = DEFAULT_MEMORY_OVERRIDE_PATH,
    runtime_override_path: Path = DEFAULT_RUNTIME_OVERRIDE_PATH,
    registry_path: Path = DEFAULT_REGISTRY_PATH,
    now: datetime | None = None,
) -> dict[str, Any]:
    current = now or datetime.now(timezone.utc)
    thresholds = _thresholds()
    resource_snapshot = _refresh_resource_guard(project_root)
    memory_payload = memory_src.build_payload(project_root, action="status", override_path=memory_override_path, changed=False)
    runtime_payload = runtime_src.build_payload(project_root)
    swap_used_gb = _safe_float(resource_snapshot.get("swap_used_gb"), 0.0)
    raw_tier = _swap_tier(swap_used_gb, thresholds)
    stale_swap_relief = _stale_swap_allocation_relief(
        resource_snapshot,
        raw_tier=raw_tier,
        thresholds=thresholds,
    )
    tier = str(stale_swap_relief.get("relieved_tier") or raw_tier)
    state = load_json(state_path)
    state_contract = _state_contract(state, now=current, tier=tier, swap_used_gb=swap_used_gb, thresholds=thresholds)
    restart_advisory = _restart_advisory(tier, thresholds)
    reboot_advisory = _reboot_advisory(
        tier=tier,
        swap_used_gb=swap_used_gb,
        expansion_session=memory_payload.get("expansion_session") if isinstance(memory_payload.get("expansion_session"), dict) else {},
        state_contract=state_contract,
        thresholds=thresholds,
    )
    overrides = _swap_env_overrides(tier, swap_used_gb)
    heavy_pause = _pause_heavy_research(tier, apply=apply)

    apply_result: dict[str, Any] = {
        "applied": False,
        "swap_override_changed": False,
        "memory_override_changed": False,
        "runtime_apply": {},
    }
    if apply:
        apply_result["applied"] = True
        apply_result["swap_override_changed"] = _write_override(override_path, overrides)
        apply_result["memory_override_changed"] = memory_src._write_override(
            memory_override_path,
            str(memory_payload.get("recommended_profile") or "air_safe"),
            memory_payload.get("recommended_env_overrides") if isinstance(memory_payload.get("recommended_env_overrides"), dict) else {},
        )
        apply_result["runtime_apply"] = runtime_src.apply_runtime_guard(
            project_root,
            runtime_payload,
            override_path=runtime_override_path,
            registry_path=registry_path,
            max_renice_processes=4,
        )

    notification = _notification(
        now=current,
        tier=tier,
        previous_tier=str(state_contract.get("previous_tier") or "normal"),
        swap_used_gb=swap_used_gb,
        restart_advisory=restart_advisory,
        reboot_advisory=reboot_advisory,
    )
    state_payload = {
        "timestamp_utc": current.isoformat(),
        "current_tier": tier,
        "previous_tier": state_contract.get("previous_tier"),
        "swap_used_gb": round(swap_used_gb, 3),
        "pressure_started_utc": state_contract.get("pressure_started_utc"),
        "persistent_pressure_minutes": state_contract.get("persistent_pressure_minutes"),
        "last_notification": notification,
    }
    if apply:
        write_payload(state_path, state_payload)

    payload = {
        "timestamp_utc": current.isoformat(),
        "schema_version": 1,
        "ok": tier in {"normal", "calm"},
        "overall_status": _tier_status(tier),
        "apply_mode": bool(apply),
        "swap_pressure": {
            "tier": tier,
            "raw_tier": raw_tier,
            "previous_tier": state_contract.get("previous_tier"),
            "swap_used_gb": round(swap_used_gb, 3),
            "memory_pressure_state": str(resource_snapshot.get("memory_pressure_state") or ""),
            "memory_pressure_kind": str(resource_snapshot.get("memory_pressure_kind") or ""),
            "stale_swap_allocation_relief": stale_swap_relief,
            "thresholds": thresholds,
            "persistent_pressure_minutes": state_contract.get("persistent_pressure_minutes"),
        },
        "applied_actions": {
            "strict_memory_profile": bool(apply),
            "runtime_throttle_apply": bool(apply),
            "heavy_research_pause": heavy_pause,
            "restart_big_apps": "notify_only_no_force_quit",
            "reboot": "notify_only_no_automatic_reboot",
        },
        "env_overrides": overrides,
        "memory_efficiency": {
            "overall_status": memory_payload.get("overall_status"),
            "recommended_profile": memory_payload.get("recommended_profile"),
            "reasons": memory_payload.get("reasons"),
            "override_path": str(memory_override_path),
        },
        "runtime_throttle": {
            "overall_status": runtime_payload.get("overall_status"),
            "throttle_profile": runtime_payload.get("throttle_profile"),
            "host_saturation_score": runtime_payload.get("host_saturation_score"),
        },
        "restart_advisory": restart_advisory,
        "reboot_advisory": reboot_advisory,
        "notification": notification,
        "apply_result": apply_result,
        "controller_contract": {
            "mode": "applied" if apply else "advisory",
            "safe_while_live": True,
            "scope": [
                "stricter_memory_profile",
                "heavy_research_pause",
                "cache_fanout_downshift",
                "mlx_lazy_load_guard",
                "restart_candidate_advisory",
                "swap_pressure_guard",
                "reboot_needed_advisory",
            ],
            "unsafe_actions_blocked_by_default": ["force_quit_foreground_apps", "automatic_reboot"],
        },
        "infrastructure_bots": [
            "swap_pressure_governor",
            "swap_restart_reboot_advisory_guard",
            "swap_pressure_regression_guard",
        ],
        "source_files": {
            "resource_guard": str(project_root / "governance" / "health" / "resource_guard_latest.json"),
            "memory_efficiency": str(project_root / "governance" / "health" / "memory_efficiency_control_latest.json"),
            "runtime_throttle": str(project_root / "governance" / "health" / "runtime_throttle_control_latest.json"),
            "swap_state": str(state_path),
            "swap_override": str(override_path),
        },
    }
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Govern macOS swap pressure by downshifting bot fanout, pausing research, and notifying safe restart/reboot advisories.")
    parser.add_argument("action", nargs="?", choices=("status", "apply"), default="status")
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--out-file", default=str(DEFAULT_OUT_PATH))
    parser.add_argument("--state-file", default=str(DEFAULT_STATE_PATH))
    parser.add_argument("--override-file", default=str(DEFAULT_OVERRIDE_PATH))
    parser.add_argument("--memory-override-file", default=str(DEFAULT_MEMORY_OVERRIDE_PATH))
    parser.add_argument("--runtime-override-file", default=str(DEFAULT_RUNTIME_OVERRIDE_PATH))
    parser.add_argument("--registry", default=str(DEFAULT_REGISTRY_PATH))
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    payload = build_payload(
        project_root,
        apply=args.action == "apply",
        state_path=Path(args.state_file).expanduser(),
        override_path=Path(args.override_file).expanduser(),
        memory_override_path=Path(args.memory_override_file).expanduser(),
        runtime_override_path=Path(args.runtime_override_file).expanduser(),
        registry_path=Path(args.registry).expanduser(),
    )
    out_path = Path(args.out_file).expanduser()
    write_payload(out_path, payload)
    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        swap = payload.get("swap_pressure") if isinstance(payload.get("swap_pressure"), dict) else {}
        print(
            "swap_pressure_governor "
            f"status={payload.get('overall_status')} "
            f"tier={swap.get('tier')} "
            f"swap_used_gb={float(swap.get('swap_used_gb', 0.0) or 0.0):.3f}"
        )
    return 0 if payload.get("overall_status") in {"ready", "needs_work", "degraded"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
