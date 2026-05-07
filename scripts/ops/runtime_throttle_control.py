#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from scripts.ops.long_runtime_common import PROJECT_ROOT, iso_now, load_json, ordered_unique, status_rank, write_payload
else:
    from .long_runtime_common import PROJECT_ROOT, iso_now, load_json, ordered_unique, status_rank, write_payload


DEFAULT_OUT_PATH = PROJECT_ROOT / "governance" / "health" / "runtime_throttle_control_latest.json"
DEFAULT_OVERRIDE_PATH = PROJECT_ROOT / "config" / ".env.runtime_resource_guard_override"
DEFAULT_REGISTRY_PATH = PROJECT_ROOT / "master_bot_registry.json"
DEFAULT_BACKPRESSURE_DRAINER_PATH = PROJECT_ROOT / "governance" / "health" / "backpressure_drainer_fleet_latest.json"
TOP_PROCESS_COUNT = 12
APPLY_CPU_THRESHOLD = 12.0
RESEARCH_TRAINING_CPU_THRESHOLD = 25.0
SIMULATED_RESEARCH_TRAINING_CPU_THRESHOLD = 10.0
FULL_FORCE_PAPER_BOT_FLOOR = 650
FULL_FORCE_PAPER_CAPACITY_TARGET = 700


PROCESS_RULES: tuple[tuple[str, str, str, bool], ...] = (
    ("scripts/run_execution_lane.py", "live_execution", "protected", False),
    ("scripts/run_all_sleeves.py", "live_execution", "protected", False),
    ("scripts/run_parallel_shadows.py", "live_execution", "protected", False),
    ("scripts/run_dividend_shadow.py", "live_execution", "protected", False),
    ("scripts/run_bond_shadow.py", "live_execution", "protected", False),
    ("scripts/run_fx_shadow.py", "live_execution", "protected", False),
    ("scripts/run_shadow_training_loop.py", "research_training", "protected", False),
    ("scripts/weekly_retrain.py", "research_training", "protected", False),
    ("scripts/retrain_daily_small_batch.sh", "research_training", "protected", False),
    ("scripts/ops/live_macro_auto_watch.py", "macro_capture", "protected_if_live", False),
    ("scripts/ops/live_macro_media_ingest.py", "macro_capture", "protected_if_live", False),
    ("yt-dlp", "macro_capture", "protected_if_live", False),
    ("ffmpeg", "macro_capture", "protected_if_live", False),
    ("scripts/ops/schwab_auth_supervisor.py", "live_execution", "protected_if_live", False),
    ("report-bundle-pdf-open", "support_maintenance", "throttle_first", True),
    ("scripts/build_one_numbers_report.py", "support_maintenance", "throttle_first", True),
    ("scripts/paper_performance_report.py", "support_maintenance", "throttle_first", True),
    ("paper_performance_report.py", "support_maintenance", "throttle_first", True),
    ("scripts/snapshot_coverage_sentinel.py", "support_maintenance", "throttle_first", True),
    ("scripts/collector_contracts.py", "support_maintenance", "throttle_first", True),
    ("scripts/data_source_divergence_bot.py", "support_maintenance", "throttle_first", True),
    ("scripts/collect_market_crypto_correlation_context.py", "support_maintenance", "throttle_first", True),
    ("scripts/collect_market_correlation_context.py", "support_maintenance", "throttle_first", True),
    ("scripts/collect_crypto_market_context.py", "support_maintenance", "throttle_first", True),
    ("project_timeline_report.py", "support_maintenance", "throttle_first", True),
    ("scripts/ops/live_feed_tail.sh", "operator_observability", "operator_visible", False),
    ("live_feed source=", "operator_observability", "operator_visible", False),
    ("tail -n 80 -F", "operator_observability", "operator_visible", False),
    ("tail -n 120 -F", "operator_observability", "operator_visible", False),
    ("failover_hot_standby.py", "support_maintenance", "throttle_first", True),
    ("sql_queue_retention.py", "support_maintenance", "throttle_first", True),
    ("sql_hot_retention.py", "support_maintenance", "throttle_first", True),
    ("sqlite_performance_maintenance.py", "support_maintenance", "throttle_first", True),
    ("data_retention_policy.py", "support_maintenance", "throttle_first", True),
    ("scripts/ops/artifact_freshness_slo.py", "support_maintenance", "throttle_first", True),
    ("scripts/ops/backpressure_slo_bot.py", "support_maintenance", "throttle_first", True),
    ("scripts/ops/backpressure_drainer_fleet.py", "support_maintenance", "throttle_first", True),
    ("scripts/ops/bot_quality_autopilot.py", "support_maintenance", "throttle_first", True),
    ("scripts/ops/command_validity_bot.py", "support_maintenance", "throttle_first", True),
    ("scripts/ops/commands_hygiene_bot.py", "support_maintenance", "throttle_first", True),
    ("scripts/ops/coverage_gap_closer.py", "support_maintenance", "throttle_first", True),
    ("scripts/ops/daily_verify_auto_remediation_bot.py", "support_maintenance", "throttle_first", True),
    ("scripts/ops/external_backlog_drain.py", "support_maintenance", "throttle_first", True),
    ("scripts/ops/ingestion_storage_governor.py", "support_maintenance", "throttle_first", True),
    ("scripts/ops/infrastructure_autofix_bot.py", "support_maintenance", "throttle_first", True),
    ("scripts/ops/master_infrastructure_supervisor.py", "support_maintenance", "throttle_first", True),
    ("scripts/ops/process_watchdog.py", "support_maintenance", "throttle_first", True),
    ("scripts/ops/retention_debt_sheriff.py", "support_maintenance", "throttle_first", True),
    ("scripts/ops/report_quality_guard.py", "support_maintenance", "throttle_first", True),
    ("scripts/ops/runtime_gate_dashboard.py", "support_maintenance", "throttle_first", True),
    ("scripts/ops/storage_reconnect_infrabot.py", "support_maintenance", "throttle_first", True),
    ("scripts/ops/mlx_runtime_audit.py", "support_maintenance", "throttle_first", True),
    ("scripts/ops/mlx_intelligence_router.py", "support_maintenance", "throttle_first", True),
    ("scripts/ops/library_utilization_router.py", "support_maintenance", "throttle_first", True),
    ("scripts/ops/storage_maintenance_lane.py", "support_maintenance", "throttle_first", True),
    ("scripts/ops/storage_backpressure_autopilot.py", "support_maintenance", "throttle_first", True),
    ("scripts/ops/storage_quota_guard.py", "support_maintenance", "throttle_first", True),
    ("scripts/ops/storage_resilience_control.py", "support_maintenance", "throttle_first", True),
    ("scripts/ops/storage_split_brain_reconciler.py", "support_maintenance", "throttle_first", True),
    ("scripts/ops/sql_link_shard_manager.py", "support_maintenance", "throttle_first", True),
    ("scripts/ops/sql_link_writer_service.py", "support_maintenance", "throttle_first", True),
    ("scripts/ops/writer_cycle_coordinator.py", "support_maintenance", "throttle_first", True),
    ("scripts/link_jsonl_to_sql.py", "support_maintenance", "throttle_first", True),
    ("Google Chrome Helper --headless", "support_maintenance", "throttle_first", True),
    ("Google Chrome", "interactive_cotenant", "external_cotenant", False),
    ("Codex", "interactive_cotenant", "external_cotenant", False),
    ("/Applications/PyCharm", "interactive_cotenant", "external_cotenant", False),
    ("PyCharm.app", "interactive_cotenant", "external_cotenant", False),
    ("com.jetbrains.pycharm", "interactive_cotenant", "external_cotenant", False),
    ("WindowServer", "interactive_cotenant", "external_cotenant", False),
    ("Code Helper", "interactive_cotenant", "external_cotenant", False),
)


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except Exception:
        return int(default)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _run_capture(command: list[str]) -> str:
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except Exception:
        return ""
    return completed.stdout or ""


def _run_apply_command(command: list[str]) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=8,
        )
        return {
            "command": command,
            "returncode": int(completed.returncode),
            "ok": completed.returncode == 0,
            "stdout": (completed.stdout or "").strip()[:500],
            "stderr": (completed.stderr or "").strip()[:500],
        }
    except Exception as exc:
        return {
            "command": command,
            "returncode": None,
            "ok": False,
            "stdout": "",
            "stderr": str(exc)[:500],
        }


def _parse_vm_stat(text: str) -> dict[str, int]:
    metrics: dict[str, int] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        digits = "".join(ch for ch in value if ch.isdigit())
        if not digits:
            continue
        normalized = key.strip().lower().replace(" ", "_")
        metrics[normalized] = int(digits)
    return metrics


def _parse_thermal_snapshot(text: str) -> dict[str, Any]:
    normalized = text.lower()
    thermal_warning = "no thermal warning level has been recorded" not in normalized and "thermal warning" in normalized
    performance_warning = "no performance warning level has been recorded" not in normalized and "performance warning" in normalized
    cpu_power_warning = "no cpu power status has been recorded" not in normalized and "cpu power status" in normalized
    return {
        "thermal_warning_active": thermal_warning,
        "performance_warning_active": performance_warning,
        "cpu_power_warning_active": cpu_power_warning,
        "raw_excerpt": [line.strip() for line in text.splitlines() if line.strip()][:6],
    }


def _classify_process(command: str) -> dict[str, Any]:
    lowered = command.lower()
    for needle, category, priority, throttle_candidate in PROCESS_RULES:
        if needle.lower() in lowered:
            return {
                "category": category,
                "priority_tier": priority,
                "throttle_candidate": throttle_candidate,
            }
    return {
        "category": "unclassified",
        "priority_tier": "observe",
        "throttle_candidate": False,
    }


def _parse_process_rows(text: str, *, limit: int = TOP_PROCESS_COUNT) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.lower().startswith("pid "):
            continue
        parts = line.split(None, 4)
        if len(parts) < 5:
            continue
        pid, cpu_percent, mem_percent, elapsed, command = parts
        classification = _classify_process(command)
        rows.append(
            {
                "pid": _safe_int(pid, 0),
                "cpu_percent": round(_safe_float(cpu_percent, 0.0), 3),
                "mem_percent": round(_safe_float(mem_percent, 0.0), 3),
                "elapsed": elapsed,
                "command": command,
                "category": classification["category"],
                "priority_tier": classification["priority_tier"],
                "throttle_candidate": bool(classification["throttle_candidate"]),
            }
        )
    rows.sort(key=lambda row: float(row.get("cpu_percent", 0.0) or 0.0), reverse=True)
    return rows[: max(int(limit), 1)]


def collect_runtime_snapshot(*, max_processes: int = TOP_PROCESS_COUNT) -> dict[str, Any]:
    cpu_count = max(os.cpu_count() or 1, 1)
    try:
        load_1m, load_5m, load_15m = os.getloadavg()
    except Exception:
        load_1m = load_5m = load_15m = 0.0

    thermal_text = _run_capture(["pmset", "-g", "therm"])
    vm_stat_text = _run_capture(["vm_stat"])
    ps_text = _run_capture(["ps", "-axo", "pid,pcpu,pmem,etime,command"])
    process_rows = _parse_process_rows(ps_text, limit=max_processes)

    category_cpu: dict[str, float] = {}
    category_counts: dict[str, int] = {}
    for row in process_rows:
        category = str(row.get("category") or "unclassified")
        category_cpu[category] = round(category_cpu.get(category, 0.0) + _safe_float(row.get("cpu_percent"), 0.0), 3)
        category_counts[category] = category_counts.get(category, 0) + 1

    return {
        "cpu_count": cpu_count,
        "load_averages": {
            "one_minute": round(float(load_1m), 3),
            "five_minutes": round(float(load_5m), 3),
            "fifteen_minutes": round(float(load_15m), 3),
        },
        "thermal": _parse_thermal_snapshot(thermal_text),
        "vm_stat": _parse_vm_stat(vm_stat_text),
        "top_processes": process_rows,
        "category_cpu": category_cpu,
        "category_counts": category_counts,
    }


def _memory_pressure_level(resource_guard: dict[str, Any], memory_efficiency: dict[str, Any]) -> str:
    pressure_state = str(resource_guard.get("memory_pressure_state") or "").strip().lower()
    pressure_kind = str(resource_guard.get("memory_pressure_kind") or "").strip().lower()
    efficiency_status = str(memory_efficiency.get("overall_status") or "").strip().lower()
    efficiency_snapshot = memory_efficiency.get("memory_snapshot") if isinstance(memory_efficiency.get("memory_snapshot"), dict) else {}
    cotenant = memory_efficiency.get("cotenant_awareness") if isinstance(memory_efficiency.get("cotenant_awareness"), dict) else {}
    efficiency_state = str(efficiency_snapshot.get("memory_pressure_state") or "").strip().lower()
    efficiency_kind = str(efficiency_snapshot.get("memory_pressure_kind") or "").strip().lower()
    swap_used_gb = _safe_float(resource_guard.get("swap_used_gb"), 0.0)
    efficiency_swap_used_gb = _safe_float(efficiency_snapshot.get("swap_used_gb"), 0.0)
    reasons = [str(item).strip().lower() for item in memory_efficiency.get("reasons", []) if str(item).strip()]
    memory_reasons = [
        item
        for item in reasons
        if any(marker in item for marker in ("memory", "swap", "compress", "throttled"))
    ]
    memory_clear_by_efficiency = bool(
        efficiency_state in {"green", "none", "normal", "clear"}
        and efficiency_kind in {"", "none", "normal", "green", "clear"}
        and bool(cotenant.get("memory_pressure_clear", False))
        and not memory_reasons
    )
    if (
        pressure_state in {"red", "critical"}
        or pressure_kind == "throttled"
        or (efficiency_status == "blocked" and not memory_clear_by_efficiency)
        or swap_used_gb >= 20.0
        or efficiency_swap_used_gb >= 20.0
    ):
        return "high"
    if (
        pressure_state in {"yellow", "warn"}
        or (efficiency_status in {"degraded", "needs_work"} and not memory_clear_by_efficiency)
        or swap_used_gb >= 8.0
        or efficiency_swap_used_gb >= 8.0
    ):
        return "elevated"
    return "normal"


def _cotenant_awareness_contract(memory_efficiency: dict[str, Any]) -> dict[str, Any]:
    cotenant = memory_efficiency.get("cotenant_awareness") if isinstance(memory_efficiency.get("cotenant_awareness"), dict) else {}
    mode = str(cotenant.get("mode") or "").strip().lower()
    open_apps = cotenant.get("open_apps") if isinstance(cotenant.get("open_apps"), list) else []
    classes = cotenant.get("co_running_classes") if isinstance(cotenant.get("co_running_classes"), list) else []
    active = bool(cotenant.get("active", False) or mode in {"managed_cotenant", "guarded_cotenant"})
    creative_level = str(cotenant.get("creative_level") or "none").strip().lower()
    co_running_level = str(cotenant.get("co_running_level") or "").strip().lower()
    memory_clear = bool(cotenant.get("memory_pressure_clear", False))
    storage_clear = bool(cotenant.get("storage_pressure_clear", False))
    if active and creative_level not in {"", "none", "idle"}:
        guard_mode = "creative_cotenant_guarded"
        recommended_profile_cap = "soft_cap"
    elif active and co_running_level in {"interactive", "heavy", "developer"}:
        guard_mode = "interactive_cotenant_managed"
        recommended_profile_cap = "soft_cap"
    elif active:
        guard_mode = "background_cotenant_managed"
        recommended_profile_cap = "observe"
    else:
        guard_mode = "inactive"
        recommended_profile_cap = "observe"
    return {
        "active": active,
        "mode": mode or ("managed_cotenant" if active else "inactive"),
        "guard_mode": guard_mode,
        "recommended_profile_cap": recommended_profile_cap,
        "open_app_count": _safe_int(cotenant.get("open_app_count"), len(open_apps)),
        "open_apps": [str(item) for item in open_apps if str(item).strip()][:12],
        "co_running_classes": [str(item) for item in classes if str(item).strip()][:12],
        "co_running_level": co_running_level,
        "creative_level": creative_level or "none",
        "memory_pressure_clear": memory_clear,
        "storage_pressure_clear": storage_clear,
        "policy": "consume_memory_efficiency_cotenant_awareness_before_runtime_profile_selection",
    }


def _apply_cotenant_profile_guard(
    throttle_profile: str,
    *,
    cotenant_contract: dict[str, Any],
    compute_pressure_level: str,
    memory_pressure_level: str,
    saturation_score: float,
) -> tuple[str, dict[str, Any]]:
    profile = str(throttle_profile or "observe")
    active = bool(cotenant_contract.get("active", False))
    if not active:
        return profile, {**cotenant_contract, "profile_adjusted": False, "adjustment_reason": "cotenant_inactive"}
    if memory_pressure_level == "high" or compute_pressure_level == "high" or profile == "protect_live":
        return profile, {**cotenant_contract, "profile_adjusted": False, "adjustment_reason": "host_pressure_takes_priority"}

    cap = str(cotenant_contract.get("recommended_profile_cap") or "observe")
    adjusted = False
    reason = "cotenant_observed_no_profile_change"
    if cap == "soft_cap" and profile == "observe":
        profile = "soft_cap"
        adjusted = True
        reason = "foreground_cotenant_soft_cap"
    elif cap == "soft_cap" and profile == "sustain" and saturation_score < 56.0 and memory_pressure_level == "normal":
        profile = "soft_cap"
        adjusted = True
        reason = "cotenant_clear_memory_downshifted_from_sustain"
    return profile, {**cotenant_contract, "profile_adjusted": adjusted, "adjustment_reason": reason}


def _mlx_intelligence_contract(router: dict[str, Any]) -> dict[str, Any]:
    status = str(router.get("overall_status") or router.get("status") or "").strip().lower()
    caps = router.get("runtime_caps") if isinstance(router.get("runtime_caps"), dict) else {}
    coverage = router.get("library_coverage") if isinstance(router.get("library_coverage"), dict) else {}
    route_coverage = router.get("route_coverage") if isinstance(router.get("route_coverage"), dict) else {}
    env = router.get("recommended_runtime_env") if isinstance(router.get("recommended_runtime_env"), dict) else {}
    active = bool(status in {"ready", "advisory", "degraded"} and caps and coverage)
    return {
        "active": active,
        "status": status or "missing",
        "profile": str(caps.get("profile") or "foreground_safe"),
        "max_concurrent_mlx_jobs": _safe_int(caps.get("max_concurrent_mlx_jobs"), 1),
        "tensor_batch_cap": _safe_int(caps.get("tensor_batch_cap"), 32),
        "embedding_batch_cap": _safe_int(caps.get("embedding_batch_cap"), 64),
        "graph_node_cap": _safe_int(caps.get("graph_node_cap"), 6000),
        "audio_minutes_per_job_cap": _safe_int(caps.get("audio_minutes_per_job_cap"), 20),
        "heavy_vlm_enabled": bool(caps.get("heavy_vlm_enabled", False)),
        "compile_mode": str(caps.get("compile_mode") or "off"),
        "library_coverage_ratio": _safe_float(coverage.get("coverage_ratio"), 0.0),
        "route_coverage_ratio": _safe_float(route_coverage.get("route_coverage_ratio"), 0.0),
        "recommended_runtime_env": {str(key): str(value) for key, value in env.items()},
        "policy": "consume_mlx_intelligence_router_caps_before_running_heavy_mlx_jobs",
    }


def _library_utilization_contract(router: dict[str, Any]) -> dict[str, Any]:
    status = str(router.get("overall_status") or router.get("status") or "").strip().lower()
    caps = router.get("runtime_caps") if isinstance(router.get("runtime_caps"), dict) else {}
    coverage = router.get("coverage") if isinstance(router.get("coverage"), dict) else {}
    env = router.get("recommended_runtime_env") if isinstance(router.get("recommended_runtime_env"), dict) else {}
    active = bool(status in {"ready", "advisory", "degraded"} and caps and coverage)
    return {
        "active": active,
        "status": status or "missing",
        "profile": str(caps.get("profile") or "foreground_safe"),
        "coverage_ratio": _safe_float(coverage.get("coverage_ratio"), 0.0),
        "locked_runtime_ok_ratio": _safe_float(coverage.get("locked_runtime_ok_ratio"), 0.0),
        "managed_non_mlx_package_count": _safe_int(coverage.get("managed_non_mlx_package_count"), 0),
        "max_async_request_concurrency": _safe_int(caps.get("max_async_request_concurrency"), 8),
        "max_sql_writer_workers": _safe_int(caps.get("max_sql_writer_workers"), 1),
        "max_dataframe_workers": _safe_int(caps.get("max_dataframe_workers"), 2),
        "max_portable_model_replay_jobs": _safe_int(caps.get("max_portable_model_replay_jobs"), 0),
        "max_report_render_jobs": _safe_int(caps.get("max_report_render_jobs"), 1),
        "default_ml_backend": str(env.get("LIBRARY_DEFAULT_ML_BACKEND") or env.get("PRIMARY_ML_RUNTIME_BACKEND") or "mlx"),
        "recommended_runtime_env": {str(key): str(value) for key, value in env.items()},
        "policy": "consume_non_mlx_library_router_caps_while_keeping_mlx_default",
    }


def _compute_pressure_level(load_ratio_one: float, load_ratio_fifteen: float) -> str:
    if load_ratio_one >= 1.25 or load_ratio_fifteen >= 1.0:
        return "high"
    if load_ratio_one >= 0.8 or load_ratio_fifteen >= 0.65:
        return "elevated"
    return "normal"


def _host_saturation_score(
    *,
    load_ratio_one: float,
    load_ratio_fifteen: float,
    support_cpu: float,
    interactive_cpu: float,
    memory_pressure_level: str,
    thermal_warning_active: bool,
    performance_warning_active: bool,
    live_read_only: bool,
) -> float:
    score = min(load_ratio_one * 35.0, 40.0)
    score += min(load_ratio_fifteen * 18.0, 22.0)
    if support_cpu >= 60.0:
        score += min((support_cpu - 40.0) * 0.22, 16.0)
    if interactive_cpu >= 80.0:
        score += min((interactive_cpu - 60.0) * 0.18, 12.0)
    if memory_pressure_level == "elevated":
        score += 12.0
    elif memory_pressure_level == "high":
        score += 24.0
    if thermal_warning_active:
        score += 20.0
    if performance_warning_active:
        score += 16.0
    if live_read_only:
        score += 8.0
    return round(max(0.0, min(score, 100.0)), 2)


def _choose_throttle_profile(
    *,
    saturation_score: float,
    compute_pressure_level: str,
    memory_pressure_level: str,
    thermal_warning_active: bool,
    performance_warning_active: bool,
    live_read_only: bool,
) -> str:
    if thermal_warning_active or performance_warning_active:
        return "protect_live"
    if memory_pressure_level == "high" and live_read_only:
        return "protect_live"
    if saturation_score >= 82.0:
        return "protect_live"
    if saturation_score >= 56.0 or compute_pressure_level == "high" or memory_pressure_level == "high":
        return "sustain"
    if saturation_score >= 28.0 or compute_pressure_level == "elevated" or memory_pressure_level == "elevated":
        return "soft_cap"
    return "observe"


def _overall_status(profile: str) -> str:
    if profile == "protect_live":
        return "blocked"
    if profile in {"sustain", "soft_cap"}:
        return "degraded"
    return "ready"


def _registry_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows = payload.get("sub_bots") if isinstance(payload.get("sub_bots"), list) else []
    return [row for row in rows if isinstance(row, dict)]


def _registry_capacity_counts(project_root: Path, *, registry_path: Path = DEFAULT_REGISTRY_PATH) -> dict[str, Any]:
    path = registry_path if registry_path.is_absolute() else project_root / registry_path
    registry = load_json(path)
    rows = _registry_rows(registry)
    active_rows = [row for row in rows if bool(row.get("active", False))]
    data_collection_rows = [
        row
        for row in active_rows
        if str(row.get("lifecycle_state") or "").strip().lower() == "data_collection_only"
    ]
    options_rows = [
        row
        for row in active_rows
        if "option" in " ".join(str(row.get(key) or "").lower() for key in ("bot_id", "sleeve_profile", "slot_kind"))
    ]
    intraday_rows = [
        row
        for row in active_rows
        if "intraday" in " ".join(str(row.get(key) or "").lower() for key in ("bot_id", "sleeve_profile", "slot_kind"))
    ]
    paper_tagged_rows = [
        row
        for row in active_rows
        if bool(row.get("paper_execution_allowed", False))
        or bool(row.get("paper_live_data_enabled", False))
        or bool(row.get("paper_trading_enabled", False))
        or bool(row.get("paper_trade_enabled", False))
    ]
    return {
        "registry_path": str(path),
        "registered_bot_count": len(rows),
        "active_bot_count": len(active_rows),
        "data_collection_only_count": len(data_collection_rows),
        "options_active_count": len(options_rows),
        "intraday_active_count": len(intraday_rows),
        "paper_tagged_count": len(paper_tagged_rows),
    }


def _paper_capacity_contract(
    counts: dict[str, Any],
    *,
    throttle_profile: str,
    memory_pressure_level: str,
    compute_pressure_level: str,
    storage_pressure_index: float,
    storage_total_pending_lines: int,
) -> dict[str, Any]:
    active_count = _safe_int(counts.get("active_bot_count"), 0)
    full_force_required = active_count >= FULL_FORCE_PAPER_BOT_FLOOR
    pressure_limited = bool(
        throttle_profile == "protect_live"
        or memory_pressure_level == "high"
        or compute_pressure_level == "high"
        or storage_pressure_index >= 1.0
    )
    if not full_force_required:
        mode = "standard_paper_runtime"
    elif pressure_limited:
        mode = "full_force_guarded"
    else:
        mode = "full_force_buffered"
    return {
        "target_bot_floor": FULL_FORCE_PAPER_CAPACITY_TARGET,
        "full_force_stabilization_required": full_force_required,
        "mode": mode,
        "ready_for_700_bot_paper": bool(full_force_required and not pressure_limited),
        "pressure_limited": pressure_limited,
        "active_bot_count": active_count,
        "registered_bot_count": _safe_int(counts.get("registered_bot_count"), 0),
        "data_collection_only_count": _safe_int(counts.get("data_collection_only_count"), 0),
        "options_active_count": _safe_int(counts.get("options_active_count"), 0),
        "intraday_active_count": _safe_int(counts.get("intraday_active_count"), 0),
        "paper_tagged_count": _safe_int(counts.get("paper_tagged_count"), 0),
        "storage_total_pending_lines": int(storage_total_pending_lines),
        "runtime_policy": {
            "paper_execution": "buffered_jsonl_batching",
            "control_refresh_seconds": 240 if full_force_required else 180,
            "jsonl_buffer_max_items": 240 if full_force_required else 80,
            "jsonl_buffer_max_age_seconds": 1.25 if full_force_required else 2.5,
            "broker_snapshot_lock_wait_seconds": 0.75 if full_force_required else 1.25,
            "live_execution_blocked": True,
        },
    }


def _collector_guard_policy(throttle_profile: str, memory_pressure_level: str, compute_pressure_level: str) -> dict[str, Any]:
    if throttle_profile == "protect_live" or memory_pressure_level == "high" or compute_pressure_level == "high":
        return {
            "compute_guard_mode": "protect_live",
            "capture_mode": "thin_sample",
            "sample_rate": 0.15,
            "freshness_slo_minimum_seconds": 1800,
            "max_daily_mb": 35,
            "reason": "host_saturated_or_memory_pressure",
        }
    if throttle_profile == "sustain":
        return {
            "compute_guard_mode": "sustain",
            "capture_mode": "sampled",
            "sample_rate": 0.3,
            "freshness_slo_minimum_seconds": 900,
            "max_daily_mb": 60,
            "reason": "host_under_sustained_pressure",
        }
    if throttle_profile == "soft_cap" or memory_pressure_level == "elevated" or compute_pressure_level == "elevated":
        return {
            "compute_guard_mode": "soft_cap",
            "capture_mode": "sampled",
            "sample_rate": 0.5,
            "freshness_slo_minimum_seconds": 600,
            "max_daily_mb": 90,
            "reason": "host_pressure_soft_cap",
        }
    return {
        "compute_guard_mode": "observe",
        "capture_mode": "full",
        "sample_rate": 1.0,
        "freshness_slo_minimum_seconds": 60,
        "max_daily_mb": 150,
        "reason": "host_pressure_normal",
    }


def _drain_friendly_sql_overrides(*, concentrated_core: bool = False) -> dict[str, str]:
    overrides = {
        "SQL_LINK_SERVICE_INTERVAL_SECONDS": "12",
        "SQL_LINK_SERVICE_HOT_MIN_INTERVAL_SECONDS": "30",
        "SQL_LINK_SERVICE_QUEUE_MIN_INTERVAL_SECONDS": "180",
        "SQL_LINK_SERVICE_HOT_BATCH_SIZE": "240000",
        "SQL_LINK_SERVICE_QUEUE_BATCH_SIZE": "180000",
        "SQL_LINK_SERVICE_MERGE_MAX_SECONDS_PER_CYCLE": "25",
    }
    if concentrated_core:
        overrides.update(
            {
                "SQL_LINK_SERVICE_CONCENTRATED_CORE_DRAIN": "1",
                "SQL_LINK_SERVICE_SHARD_LINK_TIMEOUT_SECONDS": "420",
                "SQL_LINK_SERVICE_MERGE_MAX_SECONDS_PER_CYCLE": "60",
                "SQL_LINK_SERVICE_SHARD_TRADING_STATE_CHECKPOINT_LINES": "1000",
                "SQL_LINK_SERVICE_SHARD_AGGRESSIVE_TRADING_STATE_CHECKPOINT_LINES": "1000",
                "SQL_LINK_SERVICE_SHARD_AGGRESSIVE_TRADING_MAX_LINES_PER_FILE": "12000",
            }
        )
    return overrides


def _sql_writer_coordination(backpressure_fleet: dict[str, Any], storage_backpressure: dict[str, Any]) -> dict[str, Any]:
    active_drainer = backpressure_fleet.get("active_drainer") if isinstance(backpressure_fleet.get("active_drainer"), dict) else {}
    concentration = active_drainer.get("concentration") if isinstance(active_drainer.get("concentration"), dict) else {}
    request = backpressure_fleet.get("service_request") if isinstance(backpressure_fleet.get("service_request"), dict) else {}
    env = request.get("env_overrides") if isinstance(request.get("env_overrides"), dict) else {}
    total_pending = _safe_int(concentration.get("total_pending_lines"), _safe_int(storage_backpressure.get("total_pending_lines"), 0))
    top1_share = _safe_float(concentration.get("top1_share"), 0.0)
    top3_share = _safe_float(concentration.get("top3_share"), 0.0)
    concentrated = bool(concentration.get("concentrated", False)) or str(env.get("SQL_LINK_SERVICE_CONCENTRATED_CORE_DRAIN") or "").strip() == "1"
    if not concentrated and total_pending >= 5000 and (top1_share >= 0.45 or top3_share >= 0.75):
        concentrated = True
    drain_overrides = _drain_friendly_sql_overrides(concentrated_core=concentrated)
    return {
        "source": "backpressure_drainer_fleet" if backpressure_fleet else "storage_backpressure",
        "active_drainer": str(active_drainer.get("name") or ""),
        "concentrated_core_drain": concentrated,
        "total_pending_lines": total_pending,
        "top1_share": round(top1_share, 6),
        "top3_share": round(top3_share, 6),
        "recommended_merge_max_seconds_per_cycle": _safe_int(drain_overrides.get("SQL_LINK_SERVICE_MERGE_MAX_SECONDS_PER_CYCLE"), 25),
        "recommended_shard_link_timeout_seconds": _safe_int(drain_overrides.get("SQL_LINK_SERVICE_SHARD_LINK_TIMEOUT_SECONDS"), 0),
        "recommended_aggressive_trading_max_lines_per_file": _safe_int(
            drain_overrides.get("SQL_LINK_SERVICE_SHARD_AGGRESSIVE_TRADING_MAX_LINES_PER_FILE"),
            0,
        ),
    }


def _runtime_env_overrides(
    throttle_profile: str,
    memory_pressure_level: str,
    compute_pressure_level: str,
    *,
    storage_drain_active: bool = False,
    paper_capacity_contract: dict[str, Any] | None = None,
    cotenant_contract: dict[str, Any] | None = None,
    mlx_contract: dict[str, Any] | None = None,
    library_contract: dict[str, Any] | None = None,
    sql_writer_coordination: dict[str, Any] | None = None,
) -> dict[str, str]:
    paper_capacity_contract = paper_capacity_contract if isinstance(paper_capacity_contract, dict) else {}
    cotenant_contract = cotenant_contract if isinstance(cotenant_contract, dict) else {}
    mlx_contract = mlx_contract if isinstance(mlx_contract, dict) else {}
    library_contract = library_contract if isinstance(library_contract, dict) else {}
    sql_writer_coordination = sql_writer_coordination if isinstance(sql_writer_coordination, dict) else {}
    concentrated_core_drain = bool(sql_writer_coordination.get("concentrated_core_drain", False))
    full_force_paper = bool(paper_capacity_contract.get("full_force_stabilization_required", False))

    def _with_full_force_paper(overrides: dict[str, str]) -> dict[str, str]:
        if not full_force_paper:
            return overrides
        overrides.update(
            {
                "PAPER_FULL_FORCE_STABILITY_MODE": str(paper_capacity_contract.get("mode") or "full_force_buffered"),
                "PAPER_RUNTIME_CONTROL_REFRESH_SECONDS": "240",
                "PAPER_RUNTIME_CONTROL_MAX_ROWS": "12000",
                "JSONL_BUFFER_ENABLED": "1",
                "JSONL_BUFFER_MAX_ITEMS": "240",
                "JSONL_BUFFER_MAX_AGE_SECONDS": "1.25",
                "JSONL_MESSAGE_ID_DEDUP_WINDOW_SECONDS": "1800",
                "BROKER_TRUTH_SHARED_SNAPSHOT_LOCK_WAIT_SECONDS": "0.75",
                "BROKER_TRUTH_SHARED_SNAPSHOT_SKIP_WRITE_ON_LOCK_TIMEOUT": "1",
                "EXECUTION_LANE_QUEUE_ENQUEUE_RETRIES": "4",
                "EXECUTION_LANE_QUEUE_ENQUEUE_SLEEP_SECONDS": "0.15",
            }
        )
        return overrides

    def _with_cotenant_awareness(overrides: dict[str, str]) -> dict[str, str]:
        if not bool(cotenant_contract.get("active", False)):
            return overrides
        overrides.update(
            {
                "RUNTIME_COTENANT_AWARE": "1",
                "RUNTIME_COTENANT_MODE": str(cotenant_contract.get("mode") or "managed_cotenant"),
                "RUNTIME_COTENANT_GUARD_MODE": str(cotenant_contract.get("guard_mode") or "interactive_cotenant_managed"),
                "RUNTIME_COTENANT_OPEN_APP_COUNT": str(_safe_int(cotenant_contract.get("open_app_count"), 0)),
                "RUNTIME_COTENANT_PROFILE_CAP": str(cotenant_contract.get("recommended_profile_cap") or "observe"),
            }
        )
        return overrides

    def _with_mlx_intelligence(overrides: dict[str, str]) -> dict[str, str]:
        if not bool(mlx_contract.get("active", False)):
            return overrides
        env = mlx_contract.get("recommended_runtime_env") if isinstance(mlx_contract.get("recommended_runtime_env"), dict) else {}
        if env:
            overrides.update({str(key): str(value) for key, value in env.items()})
            return overrides
        overrides.update(
            {
                "MLX_INTELLIGENCE_ROUTER_ENABLED": "1",
                "MLX_INTELLIGENCE_PROFILE": str(mlx_contract.get("profile") or "foreground_safe"),
                "MLX_INTELLIGENCE_MAX_CONCURRENT_JOBS": str(_safe_int(mlx_contract.get("max_concurrent_mlx_jobs"), 1)),
                "MLX_INTELLIGENCE_TENSOR_BATCH_CAP": str(_safe_int(mlx_contract.get("tensor_batch_cap"), 32)),
                "MLX_INTELLIGENCE_EMBED_BATCH_CAP": str(_safe_int(mlx_contract.get("embedding_batch_cap"), 64)),
                "MLX_INTELLIGENCE_GRAPH_NODE_CAP": str(_safe_int(mlx_contract.get("graph_node_cap"), 6000)),
                "MLX_INTELLIGENCE_AUDIO_MINUTES_CAP": str(_safe_int(mlx_contract.get("audio_minutes_per_job_cap"), 20)),
                "MLX_INTELLIGENCE_HEAVY_VLM_ENABLED": "1" if bool(mlx_contract.get("heavy_vlm_enabled", False)) else "0",
                "MLX_INTELLIGENCE_COMPILE_MODE": str(mlx_contract.get("compile_mode") or "off"),
                "MLX_INTELLIGENCE_SHARED_MEMORY_POLICY": "foreground_safe_unified_memory",
            }
        )
        return overrides

    def _with_library_utilization(overrides: dict[str, str]) -> dict[str, str]:
        if not bool(library_contract.get("active", False)):
            return overrides
        env = library_contract.get("recommended_runtime_env") if isinstance(library_contract.get("recommended_runtime_env"), dict) else {}
        if env:
            overrides.update({str(key): str(value) for key, value in env.items()})
            return overrides
        overrides.update(
            {
                "LIBRARY_UTILIZATION_ROUTER_ENABLED": "1",
                "LIBRARY_UTILIZATION_PROFILE": str(library_contract.get("profile") or "foreground_safe"),
                "LIBRARY_ASYNC_REQUEST_CONCURRENCY_CAP": str(_safe_int(library_contract.get("max_async_request_concurrency"), 8)),
                "LIBRARY_SQL_WRITER_WORKER_CAP": str(_safe_int(library_contract.get("max_sql_writer_workers"), 1)),
                "LIBRARY_DATAFRAME_WORKER_CAP": str(_safe_int(library_contract.get("max_dataframe_workers"), 2)),
                "LIBRARY_PORTABLE_MODEL_REPLAY_JOBS": str(_safe_int(library_contract.get("max_portable_model_replay_jobs"), 0)),
                "LIBRARY_REPORT_RENDER_JOBS": str(_safe_int(library_contract.get("max_report_render_jobs"), 1)),
                "LIBRARY_DEFAULT_ML_BACKEND": "mlx",
                "PRIMARY_ML_RUNTIME_BACKEND": "mlx",
                "PORTABLE_MODEL_REPLAY_POLICY": "canary_or_off_hours_only",
            }
        )
        return overrides

    if throttle_profile == "protect_live" or memory_pressure_level == "high" or compute_pressure_level == "high":
        overrides = {
            "BOT_RUNTIME_RESOURCE_GUARD_PROFILE": "protect_live",
            "SQL_LINK_SERVICE_INTERVAL_SECONDS": "180",
            "SQL_LINK_SERVICE_HOT_MIN_INTERVAL_SECONDS": "720",
            "SQL_LINK_SERVICE_QUEUE_MIN_INTERVAL_SECONDS": "2400",
            "SQL_LINK_SERVICE_HOT_BATCH_SIZE": "30000",
            "SQL_LINK_SERVICE_QUEUE_BATCH_SIZE": "20000",
            "ONE_NUMBERS_REFRESH_INTERVAL_SECONDS": "1200",
            "INGESTION_BACKPRESSURE_REFRESH_INTERVAL_SECONDS": "420",
            "ASYNC_PIPELINE_WORKERS": "2",
            "COINBASE_SNAPSHOT_MAX_WORKERS": "1",
            "RUNTIME_FEATURE_CACHE_MAX_ENTRIES": "48",
            "RUNTIME_TRAIN_BATCH_SIZE_CAP": "32",
            "RUNTIME_TRAIN_MAX_SAMPLES": "6000",
            "RUNTIME_RESEARCH_TRAINING_THROTTLE_ENABLED": "1",
            "RUNTIME_RESEARCH_TRAINING_CPU_THRESHOLD": "25",
            "RUNTIME_SIMULATED_TRAINING_CPU_THRESHOLD": "10",
            "SHADOW_LOOP_PRESSURE_INTERVAL_FLOOR_ENABLED": "1",
            "SHADOW_LOOP_PROTECT_LIVE_EXTRA_INTERVAL_SECONDS": "30",
            "SHADOW_LOOP_QUEUE_BACKPRESSURE_EXTRA_INTERVAL_SECONDS": "15",
            "SHADOW_LOOP_HIGH_COMPUTE_EXTRA_INTERVAL_SECONDS": "20",
            "SHADOW_LOOP_MAX_DYNAMIC_EXTRA_INTERVAL_SECONDS": "75",
            "ADAPTIVE_INTERVAL_MAX_SECONDS": "90",
            "MEMORY_THROTTLE_STEP_UP_SECONDS": "10",
            "DATA_COLLECTION_RESOURCE_GUARD_MODE": "protect_live",
            "DATA_COLLECTION_RESOURCE_SAMPLE_RATE": "0.15",
            "DATA_COLLECTION_RESOURCE_CAPTURE_MODE": "thin_sample",
            "OPS_SUPPORT_JOB_NICE": "15",
            "OPS_SUPPORT_JOBS_BACKGROUND_POLICY": "1",
            "OPS_SUPPORT_HEAVY_COLLECTOR_COOLDOWN_SECONDS": "900",
            "OPS_SUPPORT_HEAVY_COLLECTOR_MAX_CPU_PERCENT": "35",
        }
        if storage_drain_active:
            overrides.update(_drain_friendly_sql_overrides(concentrated_core=concentrated_core_drain))
        return _with_library_utilization(_with_mlx_intelligence(_with_cotenant_awareness(_with_full_force_paper(overrides))))
    if throttle_profile == "sustain":
        overrides = {
            "BOT_RUNTIME_RESOURCE_GUARD_PROFILE": "sustain",
            "SQL_LINK_SERVICE_INTERVAL_SECONDS": "120",
            "SQL_LINK_SERVICE_HOT_MIN_INTERVAL_SECONDS": "480",
            "SQL_LINK_SERVICE_QUEUE_MIN_INTERVAL_SECONDS": "1800",
            "SQL_LINK_SERVICE_HOT_BATCH_SIZE": "50000",
            "SQL_LINK_SERVICE_QUEUE_BATCH_SIZE": "30000",
            "ONE_NUMBERS_REFRESH_INTERVAL_SECONDS": "900",
            "INGESTION_BACKPRESSURE_REFRESH_INTERVAL_SECONDS": "300",
            "ASYNC_PIPELINE_WORKERS": "2",
            "COINBASE_SNAPSHOT_MAX_WORKERS": "2",
            "RUNTIME_FEATURE_CACHE_MAX_ENTRIES": "64",
            "RUNTIME_TRAIN_BATCH_SIZE_CAP": "48",
            "RUNTIME_TRAIN_MAX_SAMPLES": "8000",
            "RUNTIME_RESEARCH_TRAINING_THROTTLE_ENABLED": "1",
            "RUNTIME_RESEARCH_TRAINING_CPU_THRESHOLD": "35",
            "RUNTIME_SIMULATED_TRAINING_CPU_THRESHOLD": "15",
            "SHADOW_LOOP_PRESSURE_INTERVAL_FLOOR_ENABLED": "1",
            "SHADOW_LOOP_PROTECT_LIVE_EXTRA_INTERVAL_SECONDS": "15",
            "SHADOW_LOOP_QUEUE_BACKPRESSURE_EXTRA_INTERVAL_SECONDS": "10",
            "SHADOW_LOOP_HIGH_COMPUTE_EXTRA_INTERVAL_SECONDS": "10",
            "SHADOW_LOOP_MAX_DYNAMIC_EXTRA_INTERVAL_SECONDS": "45",
            "ADAPTIVE_INTERVAL_MAX_SECONDS": "60",
            "DATA_COLLECTION_RESOURCE_GUARD_MODE": "sustain",
            "DATA_COLLECTION_RESOURCE_SAMPLE_RATE": "0.30",
            "DATA_COLLECTION_RESOURCE_CAPTURE_MODE": "sampled",
            "OPS_SUPPORT_JOB_NICE": "10",
            "OPS_SUPPORT_JOBS_BACKGROUND_POLICY": "1",
            "OPS_SUPPORT_HEAVY_COLLECTOR_COOLDOWN_SECONDS": "600",
            "OPS_SUPPORT_HEAVY_COLLECTOR_MAX_CPU_PERCENT": "45",
        }
        if storage_drain_active:
            overrides.update(_drain_friendly_sql_overrides(concentrated_core=concentrated_core_drain))
        return _with_library_utilization(_with_mlx_intelligence(_with_cotenant_awareness(_with_full_force_paper(overrides))))
    overrides = {
        "BOT_RUNTIME_RESOURCE_GUARD_PROFILE": throttle_profile,
        "DATA_COLLECTION_RESOURCE_GUARD_MODE": throttle_profile,
        "DATA_COLLECTION_RESOURCE_SAMPLE_RATE": "0.50" if throttle_profile == "soft_cap" else "1.0",
        "DATA_COLLECTION_RESOURCE_CAPTURE_MODE": "sampled" if throttle_profile == "soft_cap" else "full",
        "OPS_SUPPORT_JOB_NICE": "5" if throttle_profile == "soft_cap" else "0",
        "OPS_SUPPORT_JOBS_BACKGROUND_POLICY": "0",
        "OPS_SUPPORT_HEAVY_COLLECTOR_COOLDOWN_SECONDS": "300" if throttle_profile == "soft_cap" else "0",
        "OPS_SUPPORT_HEAVY_COLLECTOR_MAX_CPU_PERCENT": "60" if throttle_profile == "soft_cap" else "0",
    }
    if storage_drain_active:
        overrides.update(_drain_friendly_sql_overrides())
    return _with_library_utilization(_with_mlx_intelligence(_with_cotenant_awareness(_with_full_force_paper(overrides))))


def _write_env_override(path: Path, overrides: dict[str, str], *, profile: str) -> bool:
    def assignment(name: str, value: str) -> str:
        return f"{name}={shlex.quote(str(value))}"

    lines = [
        "# Auto-managed by scripts/ops/runtime_throttle_control.py",
        assignment("BOT_RUNTIME_RESOURCE_GUARD_PROFILE", profile),
    ]
    for key, value in sorted(overrides.items()):
        if key == "BOT_RUNTIME_RESOURCE_GUARD_PROFILE":
            continue
        lines.append(assignment(key, value))
    content = "\n".join(lines) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    current = path.read_text(encoding="utf-8") if path.exists() else ""
    if current == content:
        return False
    path.write_text(content, encoding="utf-8")
    return True


def _is_simulated_shadow_training(row: dict[str, Any]) -> bool:
    command = str(row.get("command") or "")
    return "scripts/run_shadow_training_loop.py" in command and "--simulate" in command


def _research_training_pressure_candidates(
    top_processes: list[dict[str, Any]],
    *,
    profile: str,
    compute_pressure_level: str,
    memory_pressure_level: str,
) -> list[dict[str, Any]]:
    if os.getenv("RUNTIME_RESEARCH_TRAINING_THROTTLE_ENABLED", "1").strip() == "0":
        return []
    pressure_active = bool(
        str(profile or "") == "protect_live"
        or str(compute_pressure_level or "") == "high"
        or str(memory_pressure_level or "") == "high"
    )
    if not pressure_active:
        return []
    out: list[dict[str, Any]] = []
    for row in top_processes:
        if str(row.get("category") or "") != "research_training":
            continue
        cpu = _safe_float(row.get("cpu_percent"), 0.0)
        simulated = _is_simulated_shadow_training(row)
        threshold = SIMULATED_RESEARCH_TRAINING_CPU_THRESHOLD if simulated else RESEARCH_TRAINING_CPU_THRESHOLD
        if cpu < threshold:
            continue
        out.append(
            {
                **row,
                "throttle_candidate": True,
                "priority_tier": "throttle_first_when_protect_live" if simulated else "research_downshift_when_protect_live",
                "throttle_reason": "simulated_training_loop_under_host_pressure" if simulated else "research_training_loop_under_host_pressure",
            }
        )
    return out[:4]


def _apply_process_throttle(candidates: list[dict[str, Any]], *, max_processes: int) -> dict[str, Any]:
    attempted: list[dict[str, Any]] = []
    eligible = [
        row
        for row in candidates
        if _safe_int(row.get("pid"), 0) > 0 and _safe_float(row.get("cpu_percent"), 0.0) >= APPLY_CPU_THRESHOLD
    ][: max(int(max_processes), 0)]
    for row in eligible:
        pid = _safe_int(row.get("pid"), 0)
        try:
            os.kill(pid, 0)
        except Exception as exc:
            attempted.append({"pid": pid, "ok": False, "skipped": True, "reason": f"process_not_available:{exc}"})
            continue
        process_actions = {
            "pid": pid,
            "cpu_percent": row.get("cpu_percent"),
            "command_excerpt": str(row.get("command") or "")[:220],
            "renice": _run_apply_command(["renice", "-n", "15", "-p", str(pid)]),
            "taskpolicy": _run_apply_command(["taskpolicy", "-b", "-p", str(pid)]),
        }
        process_actions["ok"] = bool(
            (process_actions["renice"].get("ok") if isinstance(process_actions.get("renice"), dict) else False)
            or (process_actions["taskpolicy"].get("ok") if isinstance(process_actions.get("taskpolicy"), dict) else False)
        )
        attempted.append(process_actions)
    return {
        "attempted_count": len(attempted),
        "successful_count": sum(1 for row in attempted if bool(row.get("ok", False))),
        "processes": attempted,
    }


def _apply_registry_collector_guard(project_root: Path, payload: dict[str, Any], *, registry_path: Path = DEFAULT_REGISTRY_PATH) -> dict[str, Any]:
    path = registry_path if registry_path.is_absolute() else project_root / registry_path
    registry = load_json(path)
    if not registry:
        return {"applied": False, "changed_count": 0, "registry_path": str(path), "error": "registry_not_found_or_empty"}

    policy = _collector_guard_policy(
        str(payload.get("throttle_profile") or "observe"),
        str(payload.get("memory_pressure_level") or "normal"),
        str(payload.get("compute_pressure_level") or "normal"),
    )
    rows = _registry_rows(registry)
    paper_capacity_contract = payload.get("paper_capacity_contract") if isinstance(payload.get("paper_capacity_contract"), dict) else {}
    full_force_paper = bool(paper_capacity_contract.get("full_force_stabilization_required", False))
    changed_count = 0
    paper_changed_count = 0
    for row in rows:
        lifecycle = str(row.get("lifecycle_state") or "").strip().lower()
        active = bool(row.get("active", False))
        if not active:
            continue
        updates: dict[str, Any] = {}
        if lifecycle == "data_collection_only":
            base_slo = _safe_int(row.get("freshness_slo_seconds"), 900)
            contract = row.get("capability_pack_contract") if isinstance(row.get("capability_pack_contract"), dict) else {}
            retention = contract.get("storage_retention_rule") if isinstance(contract.get("storage_retention_rule"), dict) else {}
            contract_sample_rate = _safe_float(retention.get("sample_rate"), 0.0)
            contract_max_daily_mb = _safe_float(retention.get("max_daily_mb_per_bot"), 0.0)
            effective_sample_rate = _safe_float(policy["sample_rate"], 1.0)
            if 0.0 < contract_sample_rate < effective_sample_rate:
                effective_sample_rate = contract_sample_rate
            effective_max_daily_mb = _safe_float(policy["max_daily_mb"], 150.0)
            if 0.0 < contract_max_daily_mb < effective_max_daily_mb:
                effective_max_daily_mb = contract_max_daily_mb
            updates.update(
                {
                    "data_collection_compute_guard_mode": policy["compute_guard_mode"],
                    "data_collection_resource_guard_reason": policy["reason"],
                    "data_collection_capture_mode": policy["capture_mode"],
                    "data_collection_sample_rate": effective_sample_rate,
                    "data_collection_max_daily_mb": effective_max_daily_mb,
                    "freshness_slo_seconds": max(base_slo, _safe_int(policy["freshness_slo_minimum_seconds"], base_slo)),
                }
            )
        if full_force_paper:
            updates.update(
                {
                    "paper_runtime_stability_mode": str(paper_capacity_contract.get("mode") or "full_force_buffered"),
                    "paper_execution_queue_policy": "buffered_jsonl_batching",
                    "paper_runtime_capacity_floor": FULL_FORCE_PAPER_CAPACITY_TARGET,
                    "paper_trade_lock_required": True,
                    "paper_runtime_control_refresh_seconds": int(
                        ((paper_capacity_contract.get("runtime_policy") or {}).get("control_refresh_seconds") or 240)
                    ),
                }
            )
        row_changed = False
        for key, value in updates.items():
            if row.get(key) != value:
                row[key] = value
                row_changed = True
        if row_changed:
            changed_count += 1
            if full_force_paper:
                paper_changed_count += 1
    if changed_count:
        registry["updated_at_utc"] = iso_now()
        path.write_text(json.dumps(registry, ensure_ascii=True, indent=2), encoding="utf-8")
    return {
        "applied": bool(changed_count),
        "changed_count": changed_count,
        "paper_runtime_changed_count": paper_changed_count,
        "collector_count": sum(1 for row in rows if bool(row.get("active", False)) and str(row.get("lifecycle_state") or "").strip().lower() == "data_collection_only"),
        "full_force_paper_stabilization": full_force_paper,
        "policy": policy,
        "registry_path": str(path),
    }


def apply_runtime_guard(
    project_root: Path,
    payload: dict[str, Any],
    *,
    override_path: Path = DEFAULT_OVERRIDE_PATH,
    registry_path: Path = DEFAULT_REGISTRY_PATH,
    max_renice_processes: int = 4,
) -> dict[str, Any]:
    profile = str(payload.get("throttle_profile") or "observe")
    runtime_snapshot = payload.get("runtime_snapshot") if isinstance(payload.get("runtime_snapshot"), dict) else {}
    storage_pressure = runtime_snapshot.get("storage_pressure") if isinstance(runtime_snapshot.get("storage_pressure"), dict) else {}
    storage_stabilization = payload.get("storage_stabilization") if isinstance(payload.get("storage_stabilization"), dict) else {}
    storage_drain_active = bool(storage_stabilization.get("drain_friendly_sql_required", False))
    env_overrides = _runtime_env_overrides(
        profile,
        str(payload.get("memory_pressure_level") or "normal"),
        str(payload.get("compute_pressure_level") or "normal"),
        storage_drain_active=storage_drain_active,
        paper_capacity_contract=payload.get("paper_capacity_contract") if isinstance(payload.get("paper_capacity_contract"), dict) else {},
        cotenant_contract=payload.get("cotenant_awareness_contract") if isinstance(payload.get("cotenant_awareness_contract"), dict) else {},
        mlx_contract=payload.get("mlx_intelligence_contract") if isinstance(payload.get("mlx_intelligence_contract"), dict) else {},
        library_contract=payload.get("library_utilization_contract") if isinstance(payload.get("library_utilization_contract"), dict) else {},
        sql_writer_coordination=(storage_stabilization.get("sql_writer_coordination") if isinstance(storage_stabilization.get("sql_writer_coordination"), dict) else {}),
    )
    if _safe_float(storage_pressure.get("pressure_index"), 0.0) >= 1.0:
        env_overrides = {
            key: value
            for key, value in env_overrides.items()
            if not key.startswith("SQL_LINK_SERVICE_")
        }
    support_candidates = payload.get("support_trim_candidates") if isinstance(payload.get("support_trim_candidates"), list) else []
    research_candidates = payload.get("research_training_trim_candidates") if isinstance(payload.get("research_training_trim_candidates"), list) else []
    throttle_candidates = list(support_candidates) + list(research_candidates)
    return {
        "applied": True,
        "override_path": str(override_path),
        "override_changed": _write_env_override(override_path, env_overrides, profile=profile),
        "env_override_count": len(env_overrides),
        "storage_drain_active": storage_drain_active,
        "drain_friendly_sql_overrides": _drain_friendly_sql_overrides(concentrated_core=bool((storage_stabilization.get("sql_writer_coordination") or {}).get("concentrated_core_drain", False))) if storage_drain_active else {},
        "process_throttle": _apply_process_throttle(throttle_candidates, max_processes=max_renice_processes),
        "collector_guard": _apply_registry_collector_guard(project_root, payload, registry_path=registry_path),
    }


def _domain_rows(runtime_snapshot: dict[str, Any]) -> dict[str, dict[str, Any]]:
    category_cpu = runtime_snapshot.get("category_cpu") if isinstance(runtime_snapshot.get("category_cpu"), dict) else {}
    category_counts = runtime_snapshot.get("category_counts") if isinstance(runtime_snapshot.get("category_counts"), dict) else {}

    def _row(category: str, *, protected: bool, throttle_candidate: bool) -> dict[str, Any]:
        return {
            "cpu_percent": round(_safe_float(category_cpu.get(category), 0.0), 3),
            "process_count": _safe_int(category_counts.get(category), 0),
            "protected": protected,
            "throttle_candidate": throttle_candidate,
        }

    return {
        "live_execution": _row("live_execution", protected=True, throttle_candidate=False),
        "research_training": _row("research_training", protected=True, throttle_candidate=False),
        "macro_capture": _row("macro_capture", protected=True, throttle_candidate=False),
        "support_maintenance": _row("support_maintenance", protected=False, throttle_candidate=True),
        "interactive_cotenant": _row("interactive_cotenant", protected=False, throttle_candidate=False),
        "operator_observability": _row("operator_observability", protected=True, throttle_candidate=False),
        "unclassified": _row("unclassified", protected=False, throttle_candidate=False),
    }


def build_payload(project_root: Path = PROJECT_ROOT, *, runtime_snapshot: dict[str, Any] | None = None) -> dict[str, Any]:
    health_root = project_root / "governance" / "health"
    resource_guard = load_json(health_root / "resource_guard_latest.json")
    memory_efficiency = load_json(health_root / "memory_efficiency_control_latest.json")
    live_runtime = load_json(health_root / "live_runtime_separation_control_latest.json")
    storage_control = load_json(health_root / "ingestion_storage_control_latest.json")
    apple_profile = load_json(health_root / "apple_silicon_profile_latest.json")
    portable_brain = load_json(health_root / "portable_brain_contract_latest.json")
    mlx_router = load_json(health_root / "mlx_intelligence_router_latest.json")
    library_router = load_json(health_root / "library_utilization_router_latest.json")
    backpressure_fleet = load_json(health_root / "backpressure_drainer_fleet_latest.json")

    snapshot = runtime_snapshot if isinstance(runtime_snapshot, dict) else collect_runtime_snapshot()
    cpu_count = max(_safe_int(snapshot.get("cpu_count"), os.cpu_count() or 1), 1)
    load_averages = snapshot.get("load_averages") if isinstance(snapshot.get("load_averages"), dict) else {}
    load_one = _safe_float(load_averages.get("one_minute"), 0.0)
    load_five = _safe_float(load_averages.get("five_minutes"), 0.0)
    load_fifteen = _safe_float(load_averages.get("fifteen_minutes"), 0.0)
    load_ratio_one = round(load_one / float(cpu_count), 4)
    load_ratio_five = round(load_five / float(cpu_count), 4)
    load_ratio_fifteen = round(load_fifteen / float(cpu_count), 4)

    thermal = snapshot.get("thermal") if isinstance(snapshot.get("thermal"), dict) else {}
    thermal_warning_active = bool(thermal.get("thermal_warning_active", False))
    performance_warning_active = bool(thermal.get("performance_warning_active", False))
    domains = _domain_rows(snapshot)
    support_cpu = _safe_float(((domains.get("support_maintenance") or {}).get("cpu_percent")), 0.0)
    interactive_cpu = _safe_float(((domains.get("interactive_cotenant") or {}).get("cpu_percent")), 0.0)
    live_read_only = bool(((live_runtime.get("release_contract") or {}).get("live_lane_should_be_read_only", False)))
    memory_pressure_level = _memory_pressure_level(resource_guard, memory_efficiency)
    cotenant_contract = _cotenant_awareness_contract(memory_efficiency)
    mlx_intelligence_contract = _mlx_intelligence_contract(mlx_router)
    library_utilization_contract = _library_utilization_contract(library_router)
    compute_pressure_level = _compute_pressure_level(load_ratio_one, load_ratio_fifteen)
    saturation_score = _host_saturation_score(
        load_ratio_one=load_ratio_one,
        load_ratio_fifteen=load_ratio_fifteen,
        support_cpu=support_cpu,
        interactive_cpu=interactive_cpu,
        memory_pressure_level=memory_pressure_level,
        thermal_warning_active=thermal_warning_active,
        performance_warning_active=performance_warning_active,
        live_read_only=live_read_only,
    )
    throttle_profile = _choose_throttle_profile(
        saturation_score=saturation_score,
        compute_pressure_level=compute_pressure_level,
        memory_pressure_level=memory_pressure_level,
        thermal_warning_active=thermal_warning_active,
        performance_warning_active=performance_warning_active,
        live_read_only=live_read_only,
    )
    throttle_profile, cotenant_contract = _apply_cotenant_profile_guard(
        throttle_profile,
        cotenant_contract=cotenant_contract,
        compute_pressure_level=compute_pressure_level,
        memory_pressure_level=memory_pressure_level,
        saturation_score=saturation_score,
    )
    storage_pressure_index = _safe_float(storage_control.get("pressure_index"), 0.0)
    storage_severity = str(storage_control.get("severity") or "").strip().lower()
    storage_backpressure = storage_control.get("backpressure") if isinstance(storage_control.get("backpressure"), dict) else {}
    storage_core_pending_lines = _safe_int(storage_backpressure.get("core_pending_lines"), 0)
    storage_total_pending_lines = _safe_int(storage_backpressure.get("total_pending_lines"), storage_core_pending_lines)
    storage_backlog_drain_status = str(((storage_control.get("storage") or {}).get("backlog_drain_status")) or "").strip().lower()
    storage_recommended_mode = str(storage_control.get("recommended_operating_mode") or "").strip().lower()
    storage_drain_active = bool(
        storage_backlog_drain_status in {"drain_active", "handoff_requested"}
        or storage_recommended_mode == "maintenance_drain_window"
        or storage_total_pending_lines > 0
    )
    sql_writer_coordination = _sql_writer_coordination(backpressure_fleet, storage_backpressure)
    if storage_pressure_index >= 1.0 or (
        storage_severity in {"high", "critical", "blocked"}
        and storage_core_pending_lines >= 15000
    ):
        throttle_profile = "protect_live"
    elif storage_pressure_index >= 0.5 and throttle_profile not in {"protect_live", "sustain"}:
        throttle_profile = "sustain"
    overall_status = _overall_status(throttle_profile)
    if (
        overall_status == "degraded"
        and bool(cotenant_contract.get("profile_adjusted", False))
        and memory_pressure_level == "normal"
        and compute_pressure_level == "normal"
        and storage_pressure_index < 0.5
    ):
        overall_status = "advisory"
    registry_counts = _registry_capacity_counts(project_root)
    paper_capacity_contract = _paper_capacity_contract(
        registry_counts,
        throttle_profile=throttle_profile,
        memory_pressure_level=memory_pressure_level,
        compute_pressure_level=compute_pressure_level,
        storage_pressure_index=storage_pressure_index,
        storage_total_pending_lines=storage_total_pending_lines,
    )

    top_processes = snapshot.get("top_processes") if isinstance(snapshot.get("top_processes"), list) else []
    protected_processes = [
        row for row in top_processes if str(row.get("priority_tier") or "") in {"protected", "protected_if_live"}
    ][:5]
    support_trim_candidates = [
        row for row in top_processes if bool(row.get("throttle_candidate", False))
    ][:5]
    research_training_trim_candidates = _research_training_pressure_candidates(
        top_processes,
        profile=throttle_profile,
        compute_pressure_level=compute_pressure_level,
        memory_pressure_level=memory_pressure_level,
    )
    upgrade_recommended = bool(overall_status in {"degraded", "blocked"} or support_trim_candidates or research_training_trim_candidates)

    host_contract = {}
    if isinstance(portable_brain.get("host_contract"), dict):
        host_contract = portable_brain.get("host_contract") or {}
    elif isinstance(apple_profile.get("hardware"), dict):
        host_contract = apple_profile.get("hardware") or {}

    recommended_actions = ordered_unique(
        [
            "keep live execution, paper execution, and the active macro capture lanes protected while the host is saturated"
            if protected_processes
            else "",
            "shift retention, timeline, report, and SQL maintenance jobs into off-hours throttle windows before touching the live lanes"
            if support_trim_candidates
            else "",
            "downshift simulated shadow training loops while protect-live pressure is active; keep live and paper collectors protected"
            if research_training_trim_candidates
            else "",
            "treat Chrome, Codex, PyCharm, and other foreground apps as cotenants and downshift background support work instead of bouncing the stack"
            if interactive_cpu >= 60.0
            else "",
            "use memory-efficiency cotenant awareness to keep MLX, SQL, report, and collector jobs inside a foreground-app-safe profile"
            if bool(cotenant_contract.get("active", False))
            else "",
            "route MLX language, embedding, graph, audio, VLM, SNN, data, and quant jobs through mlx-intelligence-router caps"
            if bool(mlx_intelligence_contract.get("active", False))
            else "",
            "keep MLX as the default backend while routing non-MLX libraries into support, storage, reporting, canary, and ingestion lanes"
            if bool(library_utilization_contract.get("active", False))
            else "",
            "./scripts/ops/opsctl.sh memory-efficiency apply --json"
            if memory_pressure_level in {"elevated", "high"} and status_rank(str(memory_efficiency.get("overall_status") or "")) >= status_rank("degraded")
            else "",
            "keep the live runtime on read-only release posture until the host saturation score drops back into the soft-cap band"
            if live_read_only and overall_status in {"degraded", "blocked"}
            else "",
            "force the collector floor into protect-live sampling while storage pressure is high"
            if storage_pressure_index >= 1.0 or storage_severity in {"high", "critical", "blocked"}
            else "",
            "keep SQL writer intervals drain-friendly while support jobs are throttled"
            if storage_drain_active and overall_status in {"degraded", "blocked"}
            else "",
            "run 700-bot paper trading through buffered JSONL, slower paper-control rescans, and the persistent paper-trade lock"
            if bool(paper_capacity_contract.get("full_force_stabilization_required", False))
            else "",
            "upgrade this throttling bot alongside autonomy, memory-efficiency, and partner API surfaces so the same policy contract governs every infrabot"
            if upgrade_recommended
            else "",
        ]
    )

    return {
        "timestamp_utc": iso_now(),
        "schema_version": 1,
        "ok": overall_status == "ready",
        "overall_status": overall_status,
        "throttle_profile": throttle_profile,
        "host_saturation_score": saturation_score,
        "compute_pressure_level": compute_pressure_level,
        "memory_pressure_level": memory_pressure_level,
        "runtime_snapshot": {
            "cpu_count": cpu_count,
            "load_averages": {
                "one_minute": round(load_one, 3),
                "five_minutes": round(load_five, 3),
                "fifteen_minutes": round(load_fifteen, 3),
            },
            "load_ratios": {
                "one_minute": load_ratio_one,
                "five_minutes": load_ratio_five,
                "fifteen_minutes": load_ratio_fifteen,
            },
            "vm_pages_throttled": _safe_int(((snapshot.get("vm_stat") or {}).get("pages_throttled")), 0),
            "thermal": thermal,
            "storage_pressure": {
                "severity": storage_severity,
                "pressure_index": round(storage_pressure_index, 3),
                "core_pending_lines": storage_core_pending_lines,
            },
        },
        "host_contract": {
            "chip": str(host_contract.get("chip") or host_contract.get("model") or ""),
            "memory_architecture": str(host_contract.get("memory_architecture") or ""),
            "shared_cpu_gpu_memory_pool": bool(host_contract.get("shared_cpu_gpu_memory_pool", False)),
            "applied_tier": str(apple_profile.get("applied_tier") or ""),
            "memory_efficiency_profile": str(memory_efficiency.get("recommended_profile") or memory_efficiency.get("current_profile") or ""),
        },
        "release_contract": {
            "live_lane_should_be_read_only": live_read_only,
            "promotions_should_wait_for_cold_lane": bool(((live_runtime.get("release_contract") or {}).get("promotions_should_wait_for_cold_lane", False))),
            "shared_host_training_resume_allowed": bool(
                ((live_runtime.get("release_contract") or {}).get("shared_host_training_resume_allowed", False))
            ),
        },
        "storage_stabilization": {
            "drain_friendly_sql_required": storage_drain_active,
            "backlog_drain_status": storage_backlog_drain_status,
            "recommended_operating_mode": storage_recommended_mode,
            "total_pending_lines": storage_total_pending_lines,
            "core_pending_lines": storage_core_pending_lines,
            "sql_writer_coordination": sql_writer_coordination,
            "policy": "keep_sql_writer_responsive_while_throttling_support_jobs" if storage_drain_active else "normal_runtime_throttle",
        },
        "paper_capacity_contract": paper_capacity_contract,
        "cotenant_awareness_contract": cotenant_contract,
        "mlx_intelligence_contract": mlx_intelligence_contract,
        "library_utilization_contract": library_utilization_contract,
        "throttle_domains": domains,
        "protected_workloads": {
            "categories": [name for name, row in domains.items() if bool(row.get("protected", False)) and _safe_float(row.get("cpu_percent"), 0.0) > 0.0],
            "top_processes": protected_processes,
        },
        "support_trim_candidates": support_trim_candidates,
        "research_training_trim_candidates": research_training_trim_candidates,
        "top_processes": top_processes,
        "controller_contract": {
            "mode": "apply_capable",
            "safe_while_live": True,
            "future_auto_apply_ready": True,
            "apply_surfaces": [
                "support_process_niceness",
                "macos_background_task_policy",
                "runtime_env_resource_override",
                "data_collection_compute_guard",
                "mlx_intelligence_runtime_caps",
                "non_mlx_library_runtime_caps",
            ],
            "priority_tiers": ["protected", "protected_if_live", "operator_visible", "throttle_first", "external_cotenant"],
        },
        "upgrade_track": {
            "family": "infrabots",
            "upgradeable": True,
            "current_generation": "advisory_control_plane",
            "co_managed_with": [
                "memory_efficiency_control",
                "live_runtime_separation_control",
                "autonomy_control_plane",
                "supportability_control",
                "mlx_intelligence_router",
                "library_utilization_router",
            ],
            "future_upgrade_paths": [
                "launchd quiet-hours for support jobs",
                "priority-tier niceness rules for maintenance workloads",
                "memory overlay auto-apply when pressure persists",
                "partner API exposure for licensed tenants",
            ],
        },
        "recommended_actions": recommended_actions,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Publish a throttle-aware infrastructure control plane that protects live workloads before trimming support jobs.")
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--out-file", default=str(DEFAULT_OUT_PATH))
    parser.add_argument("--override-file", default=str(DEFAULT_OVERRIDE_PATH))
    parser.add_argument("--registry", default=str(DEFAULT_REGISTRY_PATH))
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--max-renice-processes", type=int, default=4)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    payload = build_payload(project_root)
    payload["apply_result"] = {
        "applied": False,
        "override_path": str(Path(args.override_file).expanduser()),
        "registry_path": str(Path(args.registry).expanduser()),
    }
    if args.apply:
        payload["apply_result"] = apply_runtime_guard(
            project_root,
            payload,
            override_path=Path(args.override_file).expanduser(),
            registry_path=Path(args.registry).expanduser(),
            max_renice_processes=args.max_renice_processes,
        )
        payload["controller_contract"]["mode"] = "applied"
    out_path = Path(args.out_file).expanduser()
    write_payload(out_path, payload)
    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        print(
            "runtime_throttle_control "
            f"overall_status={payload.get('overall_status', '')} "
            f"throttle_profile={payload.get('throttle_profile', '')} "
            f"host_saturation_score={float(payload.get('host_saturation_score', 0.0) or 0.0):.2f}"
        )
    return 0 if payload.get("overall_status") in {"ready", "degraded"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
