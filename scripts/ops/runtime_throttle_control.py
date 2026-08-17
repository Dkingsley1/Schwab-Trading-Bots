#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import signal
import shlex
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from scripts.ops.long_runtime_common import PROJECT_ROOT, iso_now, load_json, ordered_unique, status_rank, write_payload
else:
    from .long_runtime_common import PROJECT_ROOT, iso_now, load_json, ordered_unique, status_rank, write_payload

from core.runtime_maintenance import maintenance_hold_snapshot


DEFAULT_OUT_PATH = PROJECT_ROOT / "governance" / "health" / "runtime_throttle_control_latest.json"
DEFAULT_OVERRIDE_PATH = PROJECT_ROOT / "config" / ".env.runtime_resource_guard_override"
DEFAULT_REGISTRY_PATH = PROJECT_ROOT / "master_bot_registry.json"
SOURCE_REGISTRY_PATH = PROJECT_ROOT / "master_bot_registry.json"
DEFAULT_CANDIDATE_REGISTRY_PATH = PROJECT_ROOT / "governance" / "health" / "runtime_throttle_registry_candidate_latest.json"
DEFAULT_SOURCE_WRITE_GUARD_PATH = PROJECT_ROOT / "governance" / "health" / "runtime_throttle_source_write_guard_latest.json"
DEFAULT_BACKPRESSURE_DRAINER_PATH = PROJECT_ROOT / "governance" / "health" / "backpressure_drainer_fleet_latest.json"
DEFAULT_RESEARCH_PAUSE_STATE_PATH = PROJECT_ROOT / "governance" / "health" / "runtime_research_pause_state.json"
DEFAULT_SUPPORT_PAUSE_STATE_PATH = PROJECT_ROOT / "governance" / "health" / "runtime_support_pause_state.json"
TOP_PROCESS_COUNT = 12
APPLY_CPU_THRESHOLD = 12.0
RESEARCH_TRAINING_CPU_THRESHOLD = 25.0
SIMULATED_RESEARCH_TRAINING_CPU_THRESHOLD = 10.0
PAPER_EXECUTION_PRESSURE_PAUSE_CPU_THRESHOLD = 60.0
FULL_FORCE_PAPER_CAPACITY_LIMIT_CPU_THRESHOLD = 240.0
FULL_FORCE_PAPER_BOT_OWNED_CPU_THRESHOLD = 340.0
FULL_FORCE_PAPER_BOUNDED_RESEARCH_CPU_THRESHOLD = 80.0
FULL_FORCE_PAPER_SAMPLING_HYSTERESIS_RATIO = 1.05
FULL_FORCE_PAPER_HYSTERESIS_MAX_HOST_SATURATION = 50.0
FULL_FORCE_PAPER_ELEVATED_MAX_HOST_SATURATION = 62.0
FULL_FORCE_PAPER_HYSTERESIS_MAX_WRITER_CPU = 110.0
BOUNDED_WRITER_SUPPORT_CPU_THRESHOLD = 90.0
BOUNDED_WRITER_SUPPORT_SAMPLING_HYSTERESIS_RATIO = 1.05
BOUNDED_WRITER_SUPPORT_HYSTERESIS_MAX_HOST_SATURATION = 50.0
BOUNDED_WRITER_SUPPORT_HYSTERESIS_MAX_WRITER_CPU = 110.0
FULL_FORCE_PAPER_BOT_FLOOR = 650
FULL_FORCE_PAPER_CAPACITY_TARGET = 700
PRESSURE_ONLY_PAPER_RAMP_BLOCKERS = {
    "runtime_capacity_not_ready_for_400_paper",
    "runtime_pressure_not_ready_for_400_paper",
    "runtime_capacity_not_ready_for_paper",
    "runtime_pressure_not_ready_for_paper",
}
OVERLAY_RAW_LIVE_MAX_CORE_LINES = 10_000
OVERLAY_RAW_LIVE_MAX_TOTAL_LINES = 15_000
OVERLAY_RAW_LIVE_MAX_AGE_SECONDS = 15 * 60
OVERLAY_RUNTIME_MAX_TOTAL_LINES = 12_000
SUPPORT_PAUSE_EXEMPT_MARKERS: tuple[str, ...] = (
    "scripts/resource_guard.py",
)


PROCESS_RULES: tuple[tuple[str, str, str, bool], ...] = (
    ("scripts/run_execution_lane.py --mode paper", "paper_execution", "paper_gate_controlled", True),
    ("scripts/run_execution_lane.py", "live_execution", "protected", False),
    ("scripts/run_all_sleeves.py", "live_execution", "protected", False),
    ("scripts/run_parallel_shadows.py", "paper_execution", "paper_shadow_downshift", True),
    ("scripts/run_dividend_shadow.py", "paper_execution", "paper_shadow_downshift", True),
    ("scripts/run_bond_shadow.py", "paper_execution", "paper_shadow_downshift", True),
    ("scripts/run_fx_shadow.py", "paper_execution", "paper_shadow_downshift", True),
    ("scripts/run_shadow_training_loop.py --broker coinbase", "paper_execution", "paper_crypto_feed", True),
    ("scripts/strategy_research_lane.py", "research_training", "research_downshift", False),
    ("scripts/run_shadow_training_loop.py", "research_training", "research_downshift", False),
    ("scripts/weekly_retrain.py", "research_training", "protected", False),
    ("scripts/retrain_daily_small_batch.sh", "research_training", "protected", False),
    ("scripts/ops/training_requalification_lane.py", "research_training", "research_downshift", False),
    ("scripts/link_jsonl_to_sql.py", "storage_writer", "backlog_writer", False),
    ("scripts/ops/sql_link_shard_manager.py", "storage_writer", "backlog_writer", False),
    ("scripts/ops/sql_link_writer_service.py", "storage_writer", "backlog_writer", False),
    ("scripts/ops/writer_cycle_coordinator.py", "storage_writer", "backlog_writer", False),
    ("scripts/ops/live_macro_auto_watch.py", "macro_capture", "protected_if_live", False),
    ("scripts/ops/live_macro_media_ingest.py", "macro_capture", "protected_if_live", False),
    ("yt-dlp", "macro_capture", "protected_if_live", False),
    ("ffmpeg", "macro_capture", "protected_if_live", False),
    ("scripts/ops/schwab_auth_supervisor.py", "live_execution", "protected_if_live", False),
    ("scripts/canary_rollout_guard.py", "support_maintenance", "throttle_first", True),
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
    ("scripts/collect_fx_market_context.py", "support_maintenance", "throttle_first", True),
    ("scripts/ingestion_backpressure_guard.py", "support_maintenance", "throttle_first", True),
    ("project_timeline_report.py", "support_maintenance", "throttle_first", True),
    ("scripts/ops/live_feed_tail.sh", "operator_observability", "operator_visible", False),
    ("scripts/ops/opsctl.sh runtime-throttle", "operator_observability", "operator_visible", False),
    ("scripts/ops/runtime_throttle_control.py", "operator_observability", "operator_visible", False),
    ("scripts/ops/pressure_relief_control.py", "operator_observability", "operator_visible", False),
    ("scripts/ops/system_intelligence_coordinator.py", "operator_observability", "operator_visible", False),
    ("scripts/ops/system_needs_intelligence.py", "operator_observability", "operator_visible", False),
    ("scripts/ops/paper_profitability_control.py", "operator_observability", "operator_visible", False),
    ("scripts/ops/quant_strategy_storage_backlog_accommodation.py", "operator_observability", "operator_visible", False),
    ("live_feed source=", "operator_observability", "operator_visible", False),
    ("tail -c ", "operator_observability", "operator_visible", False),
    ("tail -n 80 -F", "operator_observability", "operator_visible", False),
    ("tail -n 120 -F", "operator_observability", "operator_visible", False),
    ("awk -v max", "operator_observability", "operator_visible", False),
    (" -m pytest", "operator_observability", "operator_visible", False),
    ("Activity Monitor.app", "operator_observability", "operator_visible", False),
    ("asitop", "operator_observability", "operator_visible", False),
    ("btop", "operator_observability", "operator_visible", False),
    ("failover_hot_standby.py", "support_maintenance", "throttle_first", True),
    ("sql_queue_retention.py", "support_maintenance", "throttle_first", True),
    ("sql_hot_retention.py", "support_maintenance", "throttle_first", True),
    ("sqlite_performance_maintenance.py", "support_maintenance", "throttle_first", True),
    ("scripts/ops/stale_artifact_reaper_bot.py", "support_maintenance", "throttle_first", True),
    ("data_retention_policy.py", "support_maintenance", "throttle_first", True),
    ("scripts/ops/artifact_freshness_slo.py", "support_maintenance", "throttle_first", True),
    ("scripts/ops/backpressure_slo_bot.py", "support_maintenance", "throttle_first", True),
    ("scripts/ops/backpressure_drainer_fleet.py", "support_maintenance", "throttle_first", True),
    ("scripts/ops/bot_quality_autopilot.py", "support_maintenance", "throttle_first", True),
    ("scripts/ops/command_validity_bot.py", "support_maintenance", "throttle_first", True),
    ("scripts/ops/commands_hygiene_bot.py", "support_maintenance", "throttle_first", True),
    ("scripts/ops/coverage_gap_closer.py", "support_maintenance", "throttle_first", True),
    ("scripts/ops/creative_cotenant_guard.py", "support_maintenance", "throttle_first", True),
    ("scripts/ops/daily_verify_auto_remediation_bot.py", "support_maintenance", "throttle_first", True),
    ("scripts/build_runtime_training_snapshot.py", "support_maintenance", "throttle_first", True),
    ("scripts/collect_schwab_education_context.py", "support_maintenance", "throttle_first", True),
    ("scripts/ops/external_backlog_drain.py", "support_maintenance", "throttle_first", True),
    ("scripts/ops/governance_telemetry_compactor.py", "support_maintenance", "throttle_first", True),
    ("scripts/ops/governance_lifecycle_compactor.py", "support_maintenance", "throttle_first", True),
    ("scripts/ops/decision_log_compactor.py", "support_maintenance", "throttle_first", True),
    ("scripts/ops/raw_training_compaction_intelligence.py", "support_maintenance", "throttle_first", True),
    ("scripts/ops/retention_intelligence_v2.py", "support_maintenance", "throttle_first", True),
    ("scripts/ops/ingestion_storage_governor.py", "support_maintenance", "throttle_first", True),
    ("scripts/ops/infrastructure_autofix_bot.py", "support_maintenance", "throttle_first", True),
    ("scripts/ops/master_infrastructure_supervisor.py", "support_maintenance", "throttle_first", True),
    ("scripts/ops/premarket_token_guard.py", "support_maintenance", "throttle_first", True),
    ("scripts/ops/guard_intelligence_layer.py", "support_maintenance", "throttle_first", True),
    ("scripts/ops/process_watchdog.py", "support_maintenance", "throttle_first", True),
    ("scripts/ops/retention_debt_sheriff.py", "support_maintenance", "throttle_first", True),
    ("scripts/ops/report_quality_guard.py", "support_maintenance", "throttle_first", True),
    ("scripts/ops/runtime_gate_dashboard.py", "support_maintenance", "throttle_first", True),
    ("scripts/ops/swap_pressure_governor.py", "support_maintenance", "throttle_first", True),
    ("scripts/ops/storage_reconnect_infrabot.py", "support_maintenance", "throttle_first", True),
    ("scripts/ops/storage_failback_sync.py", "support_maintenance", "throttle_first", True),
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
    ("scripts/collect_market_crypto_correlation_context.py", "support_maintenance", "throttle_first", True),
    ("scripts/collect_dividend_drip_state.py", "support_maintenance", "throttle_first", True),
    ("scripts/collect_fx_market_context.py", "support_maintenance", "throttle_first", True),
    ("scripts/ingestion_backpressure_guard.py", "support_maintenance", "throttle_first", True),
    ("scripts/link_jsonl_to_sql.py", "support_maintenance", "throttle_first", True),
    ("scripts/resource_guard.py", "support_maintenance", "control_plane_sensor", False),
    ("Google Chrome Helper --headless", "support_maintenance", "throttle_first", True),
    ("Google Chrome", "interactive_cotenant", "external_cotenant", False),
    ("Codex", "interactive_cotenant", "external_cotenant", False),
    ("/Applications/PyCharm", "interactive_cotenant", "external_cotenant", False),
    ("PyCharm.app", "interactive_cotenant", "external_cotenant", False),
    ("com.jetbrains.pycharm", "interactive_cotenant", "external_cotenant", False),
    ("WindowServer", "interactive_cotenant", "external_cotenant", False),
    ("Code Helper", "interactive_cotenant", "external_cotenant", False),
    ("spotlightknowledged", "system_cotenant", "external_system", False),
    ("mds_stores", "system_cotenant", "external_system", False),
    ("mdworker", "system_cotenant", "external_system", False),
    ("suggestd", "system_cotenant", "external_system", False),
    ("knowledgeconstructiond", "system_cotenant", "external_system", False),
    ("photoanalysisd", "system_cotenant", "external_system", False),
    ("mediaanalysisd", "system_cotenant", "external_system", False),
    ("backupd", "system_cotenant", "external_system", False),
    ("fileproviderd", "system_cotenant", "external_system", False),
    ("sysmond", "system_cotenant", "external_system", False),
    ("kernel_task", "system_cotenant", "external_system", False),
    ("powerd", "system_cotenant", "external_system", False),
    ("logd", "system_cotenant", "external_system", False),
    ("diagnosticd", "system_cotenant", "external_system", False),
    ("corespotlightd", "system_cotenant", "external_system", False),
    ("CoreSpotlight.framework", "system_cotenant", "external_system", False),
    ("coreaudiod", "system_cotenant", "external_system", False),
    ("usbaudiod", "system_cotenant", "external_system", False),
    ("runningboardd", "system_cotenant", "external_system", False),
    ("cfprefsd", "system_cotenant", "external_system", False),
    ("syspolicyd", "system_cotenant", "external_system", False),
    ("pmset -g log", "system_cotenant", "external_system", False),
    ("/usr/bin/pmset -g log", "system_cotenant", "external_system", False),
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


def _env_flag(name: str, default: str = "0") -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "on"}


def _research_throttle_target_nice() -> int:
    override = os.getenv("RUNTIME_THROTTLE_RESEARCH_NICE", "").strip()
    if override:
        return min(max(_safe_int(override, 15), 0), 20)
    if _env_flag("BOT_CPU_EFFICIENCY_SATURATION_GUARD", "0"):
        return min(max(_safe_int(os.getenv("SLEEVE_NICE_SPECIALIZED", "8"), 8), 0), 20)
    return 15


def _runtime_throttle_uses_background_taskpolicy() -> bool:
    override = os.getenv("RUNTIME_THROTTLE_USE_TASKPOLICY_BACKGROUND", "").strip()
    if override:
        return override.lower() in {"1", "true", "yes", "on"}
    return not _env_flag("BOT_CPU_EFFICIENCY_SATURATION_GUARD", "0")


def _renice_delta_for_target(current_nice: int, target_nice: int) -> int:
    return max(min(int(target_nice), 20) - max(int(current_nice), 0), 0)


def _support_throttle_target_nice(env_overrides: dict[str, str] | None = None) -> int:
    env = env_overrides if isinstance(env_overrides, dict) else {}
    raw = str(env.get("OPS_SUPPORT_JOB_NICE") or os.getenv("OPS_SUPPORT_JOB_NICE") or "15")
    return min(max(_safe_int(raw, 15), 0), 20)


def _paper_execution_target_nice(env_overrides: dict[str, str] | None = None) -> int:
    env = env_overrides if isinstance(env_overrides, dict) else {}
    raw = str(
        env.get("PAPER_EXECUTION_RUNTIME_NICE")
        or env.get("PAPER_SHADOW_RUNTIME_NICE")
        or os.getenv("PAPER_EXECUTION_RUNTIME_NICE")
        or os.getenv("PAPER_SHADOW_RUNTIME_NICE")
        or "12"
    )
    return min(max(_safe_int(raw, 12), 0), 20)


def _target_nice_for_candidate(row: dict[str, Any], env_overrides: dict[str, str] | None = None) -> int:
    category = str(row.get("category") or "")
    priority_tier = str(row.get("priority_tier") or "")
    if category == "support_maintenance" or priority_tier == "throttle_first":
        return _support_throttle_target_nice(env_overrides)
    if category == "storage_writer":
        env = env_overrides if isinstance(env_overrides, dict) else {}
        raw = str(env.get("SQL_LINK_WRITER_NICE") or os.getenv("SQL_LINK_WRITER_NICE") or "")
        if raw:
            return min(max(_safe_int(raw, 15), 0), 20)
        return _support_throttle_target_nice(env_overrides)
    if category == "paper_execution":
        return _paper_execution_target_nice(env_overrides)
    if category == "research_training":
        env = env_overrides if isinstance(env_overrides, dict) else {}
        raw = str(env.get("RUNTIME_THROTTLE_RESEARCH_NICE") or env.get("RUNTIME_RESEARCH_TRAINING_NICE") or "")
        if raw:
            return min(max(_safe_int(raw, 15), 0), 20)
    return _research_throttle_target_nice()


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
    if "yt-dlp" in lowered and "--dump-single-json" in lowered:
        return {
            "category": "support_maintenance",
            "priority_tier": "throttle_first",
            "throttle_candidate": True,
        }
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
        parts = line.split(None, 5)
        if len(parts) < 6:
            continue
        pid, nice, cpu_percent, mem_percent, elapsed, command = parts
        classification = _classify_process(command)
        rows.append(
            {
                "pid": _safe_int(pid, 0),
                "nice": _safe_int(nice, 0),
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


def _parse_process_cpu_time_seconds(value: str) -> float | None:
    raw = str(value or "").strip()
    if not raw:
        return None
    days = 0.0
    if "-" in raw:
        day_text, raw = raw.split("-", 1)
        try:
            days = float(day_text)
        except ValueError:
            return None
    parts = raw.split(":")
    try:
        if len(parts) == 3:
            hours, minutes, seconds = float(parts[0]), float(parts[1]), float(parts[2])
        elif len(parts) == 2:
            hours, minutes, seconds = 0.0, float(parts[0]), float(parts[1])
        elif len(parts) == 1:
            hours, minutes, seconds = 0.0, 0.0, float(parts[0])
        else:
            return None
    except ValueError:
        return None
    return days * 86400.0 + hours * 3600.0 + minutes * 60.0 + seconds


def _parse_process_cpu_times(text: str) -> dict[int, float]:
    values: dict[int, float] = {}
    for raw_line in text.splitlines():
        parts = raw_line.strip().split(None, 1)
        if len(parts) != 2:
            continue
        pid = _safe_int(parts[0], 0)
        cpu_seconds = _parse_process_cpu_time_seconds(parts[1])
        if pid > 0 and cpu_seconds is not None:
            values[pid] = cpu_seconds
    return values


def _apply_current_process_cpu_sample(
    rows: list[dict[str, Any]],
    *,
    before_text: str,
    after_text: str,
    sample_seconds: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    return _apply_process_cpu_sample_windows(
        rows,
        samples=[(before_text, after_text, sample_seconds)],
    )


def _apply_process_cpu_sample_windows(
    rows: list[dict[str, Any]],
    *,
    samples: list[tuple[str, str, float]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    parsed_samples = [
        (
            _parse_process_cpu_times(before_text),
            _parse_process_cpu_times(after_text),
            max(float(sample_seconds), 0.001),
        )
        for before_text, after_text, sample_seconds in samples
    ]
    sampled_count = 0
    transient_burst_count = 0
    updated: list[dict[str, Any]] = []
    for source_row in rows:
        row = dict(source_row)
        pid = _safe_int(row.get("pid"), 0)
        cpu_windows = [
            max((after[pid] - before[pid]) / interval * 100.0, 0.0)
            for before, after, interval in parsed_samples
            if pid in before and pid in after and after[pid] >= before[pid]
        ]
        if cpu_windows:
            sampled_cpu = float(statistics.median(cpu_windows))
            peak_cpu = max(cpu_windows)
            row["ps_cpu_percent"] = round(_safe_float(row.get("cpu_percent"), 0.0), 3)
            row["cpu_percent"] = round(sampled_cpu, 3)
            row["cpu_sample_source"] = (
                "cpu_time_delta" if len(cpu_windows) == 1 else "cpu_time_delta_window_median"
            )
            row["cpu_sample_window_count"] = len(cpu_windows)
            row["cpu_sample_peak_percent"] = round(peak_cpu, 3)
            row["cpu_sample_window_percentages"] = [round(value, 3) for value in cpu_windows]
            if len(cpu_windows) > 1 and peak_cpu >= 35.0 and sampled_cpu < 35.0:
                transient_burst_count += 1
            sampled_count += 1
        else:
            row["cpu_sample_source"] = "ps_pcpu_fallback"
        updated.append(row)
    updated.sort(key=lambda row: float(row.get("cpu_percent", 0.0) or 0.0), reverse=True)
    total_sample_seconds = sum(interval for _before, _after, interval in parsed_samples)
    return updated, {
        "active": sampled_count > 0,
        "sample_seconds": round(total_sample_seconds, 3),
        "sample_window_count": len(parsed_samples),
        "sample_window_seconds": [round(interval, 3) for _before, _after, interval in parsed_samples],
        "sampled_process_count": sampled_count,
        "fallback_process_count": max(len(updated) - sampled_count, 0),
        "transient_burst_process_count": transient_burst_count,
        "policy": "use the median of bounded process CPU-time windows so sleeping lanes and isolated bursts are not treated as sustained pressure",
    }


def collect_runtime_snapshot(*, max_processes: int = TOP_PROCESS_COUNT) -> dict[str, Any]:
    cpu_count = max(os.cpu_count() or 1, 1)
    try:
        load_1m, load_5m, load_15m = os.getloadavg()
    except Exception:
        load_1m = load_5m = load_15m = 0.0

    thermal_text = _run_capture(["pmset", "-g", "therm"])
    vm_stat_text = _run_capture(["vm_stat"])
    ps_text = _run_capture(["ps", "-axo", "pid,ni,pcpu,pmem,etime,command"])
    self_pid = os.getpid()
    parsed_process_rows = _parse_process_rows(ps_text, limit=max(len(ps_text.splitlines()), max_processes + 4))
    process_cpu_sample_seconds = min(
        max(_safe_float(os.getenv("RUNTIME_PROCESS_CPU_SAMPLE_SECONDS"), 0.25), 0.1),
        1.0,
    )
    process_cpu_sample_windows = min(
        max(_safe_int(os.getenv("RUNTIME_PROCESS_CPU_SAMPLE_WINDOWS"), 3), 1),
        5,
    )
    cpu_before_text = _run_capture(["ps", "-axo", "pid=,time="])
    cpu_before = _parse_process_cpu_times(cpu_before_text)
    process_cpu_sampling = {
        "active": False,
        "sample_seconds": round(process_cpu_sample_seconds * process_cpu_sample_windows, 3),
        "sample_window_count": process_cpu_sample_windows,
        "sample_window_seconds": [round(process_cpu_sample_seconds, 3)] * process_cpu_sample_windows,
        "sampled_process_count": 0,
        "fallback_process_count": len(parsed_process_rows),
        "transient_burst_process_count": 0,
        "policy": "use the median of bounded process CPU-time windows so sleeping lanes and isolated bursts are not treated as sustained pressure",
    }
    if cpu_before:
        cpu_samples: list[tuple[str, str, float]] = []
        for _ in range(process_cpu_sample_windows):
            sample_started = time.monotonic()
            time.sleep(process_cpu_sample_seconds)
            cpu_after_text = _run_capture(["ps", "-axo", "pid=,time="])
            actual_sample_seconds = max(time.monotonic() - sample_started, process_cpu_sample_seconds)
            cpu_samples.append((cpu_before_text, cpu_after_text, actual_sample_seconds))
            cpu_before_text = cpu_after_text
        parsed_process_rows, process_cpu_sampling = _apply_process_cpu_sample_windows(
            parsed_process_rows,
            samples=cpu_samples,
        )
    sampled_process_rows = parsed_process_rows[: max(int(max_processes) + 4, 1)]
    self_process_rows = [row for row in sampled_process_rows if _safe_int(row.get("pid"), 0) == self_pid]
    process_rows = [row for row in sampled_process_rows if _safe_int(row.get("pid"), 0) != self_pid][
        : max(int(max_processes), 1)
    ]

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
        "process_cpu_sampling": process_cpu_sampling,
        "self_observation_exclusion": {
            "active": True,
            "pid": self_pid,
            "excluded_count": len(self_process_rows),
            "policy": "do not classify the throttle controller's own short refresh burst as sustained runtime pressure",
        },
    }


def _memory_pressure_level(resource_guard: dict[str, Any], memory_efficiency: dict[str, Any]) -> str:
    pressure_state = str(resource_guard.get("memory_pressure_state") or "").strip().lower()
    pressure_kind = str(resource_guard.get("memory_pressure_kind") or "").strip().lower()
    efficiency_status = str(memory_efficiency.get("overall_status") or "").strip().lower()
    efficiency_snapshot = memory_efficiency.get("memory_snapshot") if isinstance(memory_efficiency.get("memory_snapshot"), dict) else {}
    cotenant = memory_efficiency.get("cotenant_awareness") if isinstance(memory_efficiency.get("cotenant_awareness"), dict) else {}
    compression_relief = (
        memory_efficiency.get("compressed_memory_relief_contract")
        if isinstance(memory_efficiency.get("compressed_memory_relief_contract"), dict)
        else {}
    )
    memory_truth = (
        memory_efficiency.get("memory_truth_reconciliation")
        if isinstance(memory_efficiency.get("memory_truth_reconciliation"), dict)
        else {}
    )
    efficiency_state = str(efficiency_snapshot.get("memory_pressure_state") or "").strip().lower()
    efficiency_kind = str(efficiency_snapshot.get("memory_pressure_kind") or "").strip().lower()
    raw_swap_used_gb = _safe_float(resource_guard.get("swap_used_gb"), 0.0)
    efficiency_swap_used_gb = _safe_float(efficiency_snapshot.get("swap_used_gb"), 0.0)
    efficiency_free_pct = _safe_float(efficiency_snapshot.get("memory_free_pct"), 0.0)
    efficiency_compressor_gb = _safe_float(efficiency_snapshot.get("compressor_gb"), 0.0)
    reasons = [str(item).strip().lower() for item in memory_efficiency.get("reasons", []) if str(item).strip()]
    memory_reasons = [
        item
        for item in reasons
        if any(marker in item for marker in ("memory", "swap", "compress", "throttled"))
    ]
    allocation_only_compression = bool(
        efficiency_state in {"green", "none", "normal", "clear"}
        and efficiency_kind in {"", "none", "normal", "green", "clear"}
        and efficiency_swap_used_gb < 3.0
        and (efficiency_free_pct <= 0.0 or efficiency_free_pct >= 50.0)
        and 0.0 < efficiency_compressor_gb < 14.0
    )
    managed_compression_relief = bool(compression_relief.get("managed", False))
    stale_swap_reconciled = bool(memory_truth.get("stale_swap_relief", False)) or managed_compression_relief
    swap_used_gb = (
        efficiency_swap_used_gb
        if stale_swap_reconciled and efficiency_swap_used_gb + 0.5 < raw_swap_used_gb
        else raw_swap_used_gb
    )
    effective_memory_reasons = [
        reason
        for reason in memory_reasons
        if not (
            reason in {"compressed_memory_high", "compressed_memory_critical"}
            and (allocation_only_compression or managed_compression_relief)
        )
    ]
    memory_clear_by_efficiency = bool(
        efficiency_state in {"green", "none", "normal", "clear"}
        and efficiency_kind in {"", "none", "normal", "green", "clear"}
        and bool(cotenant.get("memory_pressure_clear", False))
        and not effective_memory_reasons
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


def _cotenant_awareness_contract(memory_efficiency: dict[str, Any], computer_task: dict[str, Any] | None = None) -> dict[str, Any]:
    cotenant = memory_efficiency.get("cotenant_awareness") if isinstance(memory_efficiency.get("cotenant_awareness"), dict) else {}
    task = computer_task if isinstance(computer_task, dict) else {}
    session = task.get("session_context") if isinstance(task.get("session_context"), dict) else {}
    infrabot = (
        session.get("process_context_infrabot")
        if isinstance(session.get("process_context_infrabot"), dict)
        else task.get("stale_process_context_infrabot")
        if isinstance(task.get("stale_process_context_infrabot"), dict)
        else {}
    )
    ignore_stale_memory_apps = bool(infrabot.get("ignored_memory_efficiency_app_context", False))
    mode = str(cotenant.get("mode") or "").strip().lower()
    memory_open_apps = cotenant.get("open_apps") if isinstance(cotenant.get("open_apps"), list) else []
    memory_classes = cotenant.get("co_running_classes") if isinstance(cotenant.get("co_running_classes"), list) else []
    session_open_apps = session.get("open_apps") if isinstance(session.get("open_apps"), list) else []
    session_classes = session.get("co_running_classes") if isinstance(session.get("co_running_classes"), list) else []
    open_apps = session_open_apps if ignore_stale_memory_apps else memory_open_apps
    classes = session_classes if ignore_stale_memory_apps else memory_classes
    active = bool(
        (session.get("cotenant_active", False) or session.get("creative_active", False))
        if ignore_stale_memory_apps
        else (cotenant.get("active", False) or mode in {"managed_cotenant", "guarded_cotenant"})
    )
    creative_level = str((session.get("creative_level") if ignore_stale_memory_apps else cotenant.get("creative_level")) or "none").strip().lower()
    co_running_level = str((session.get("co_running_level") if ignore_stale_memory_apps else cotenant.get("co_running_level")) or "").strip().lower()
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
        "open_app_count": len([item for item in open_apps if str(item).strip()])
        if ignore_stale_memory_apps
        else _safe_int(cotenant.get("open_app_count"), len(open_apps)),
        "open_apps": [str(item) for item in open_apps if str(item).strip()][:12],
        "co_running_classes": [str(item) for item in classes if str(item).strip()][:12],
        "co_running_level": co_running_level,
        "creative_level": creative_level or "none",
        "memory_pressure_clear": memory_clear,
        "storage_pressure_clear": storage_clear,
        "stale_memory_efficiency_context_ignored": ignore_stale_memory_apps,
        "memory_efficiency_open_apps": [str(item) for item in memory_open_apps if str(item).strip()][:12],
        "context_source": "computer_task_intelligence" if ignore_stale_memory_apps else "memory_efficiency_control",
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
    active = bool(status in {"ready", "advisory", "degraded", "blocked"} and caps)
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
        "p_core_allocation_aware": bool(caps.get("p_core_allocation_aware", False)),
        "p_core_allocation_mode": str(caps.get("p_core_allocation_mode") or ""),
        "p_core_preprocess_workers": _safe_int(caps.get("p_core_preprocess_workers"), 0),
        "p_core_memory_optimizer_active": bool(caps.get("p_core_memory_optimizer_active", False)),
        "p_core_coordination_policy": str(caps.get("p_core_coordination_policy") or "not_active"),
        "library_coverage_ratio": _safe_float(coverage.get("coverage_ratio"), 0.0),
        "route_coverage_ratio": _safe_float(route_coverage.get("route_coverage_ratio"), 0.0),
        "recommended_runtime_env": {str(key): str(value) for key, value in env.items()},
        "policy": "consume_mlx_intelligence_router_caps_before_running_heavy_mlx_jobs_even_when_optional_mlx_lanes_are_blocked",
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


def _effective_storage_raw_live(backpressure: dict[str, Any]) -> tuple[dict[str, Any], str]:
    payload = backpressure if isinstance(backpressure, dict) else {}
    effective = payload.get("effective_raw_live") if isinstance(payload.get("effective_raw_live"), dict) else {}
    effective_source = str(payload.get("effective_raw_live_source") or effective.get("source") or "")
    estimate = effective.get("raw_live_estimate") if isinstance(effective.get("raw_live_estimate"), dict) else {}
    managed_pressure_view = bool(
        payload.get("managed_support_overlay_backlog", False)
        or payload.get("overlay_pressure_clear", False)
        or (
            isinstance(payload.get("managed_tiny_hot_tail"), dict)
            and bool(payload.get("managed_tiny_hot_tail", {}).get("active", False))
        )
    )
    effective_clear = bool(
        effective
        and _safe_int(effective.get("core_pending_lines"), 0) <= OVERLAY_RAW_LIVE_MAX_CORE_LINES
        and _safe_int(effective.get("total_pending_lines"), 0) <= OVERLAY_RAW_LIVE_MAX_TOTAL_LINES
        and _safe_float(effective.get("oldest_pending_age_seconds"), 0.0) <= OVERLAY_RAW_LIVE_MAX_AGE_SECONDS
    )
    if effective_clear:
        source = str(effective_source or effective.get("source") or "effective_raw_live")
        return {**effective, "source": source}, source
    if estimate and effective_source == "sql_ingestion_overlay_pressure":
        source = "effective_raw_live.raw_live_estimate"
        return {**estimate, "source": source, "reconciled_from_raw_live": True}, source
    if managed_pressure_view and "pressure_total_pending_lines" in payload:
        source = str(effective_source or "managed_storage_pressure_view")
        return {
            "core_pending_lines": _safe_int(payload.get("pressure_core_pending_lines"), payload.get("core_pending_lines")),
            "total_pending_lines": _safe_int(payload.get("pressure_total_pending_lines"), payload.get("total_pending_lines")),
            "oldest_pending_age_seconds": _safe_float(
                payload.get("pressure_oldest_pending_age_seconds"),
                payload.get("oldest_pending_age_seconds"),
            ),
            "source": source,
            "reconciled_from_raw_live": True,
        }, source
    raw = effective or (payload.get("raw_live") if isinstance(payload.get("raw_live"), dict) else {})
    source = str(effective_source or ("effective_raw_live" if effective else "raw_live"))
    return raw, source


def _storage_overlay_relief_contract(
    storage_backpressure: dict[str, Any],
    *,
    storage_severity: str = "",
    storage_pressure_index: float = 0.0,
    sql_ingestion_overlay: dict[str, Any] | None = None,
) -> dict[str, Any]:
    backpressure = storage_backpressure if isinstance(storage_backpressure, dict) else {}
    sql_overlay = sql_ingestion_overlay if isinstance(sql_ingestion_overlay, dict) else {}
    raw_live, raw_live_source = _effective_storage_raw_live(backpressure)
    raw_core = _safe_int(raw_live.get("core_pending_lines"), 0)
    raw_total = _safe_int(raw_live.get("total_pending_lines"), 0)
    raw_oldest = _safe_float(raw_live.get("oldest_pending_age_seconds"), 0.0)
    managed_pressure_view = bool(
        backpressure.get("managed_support_overlay_backlog", False)
        or backpressure.get("overlay_pressure_clear", False)
        or (
            isinstance(backpressure.get("managed_tiny_hot_tail"), dict)
            and bool(backpressure.get("managed_tiny_hot_tail", {}).get("active", False))
        )
    )
    raw_overlay_total = _safe_int(backpressure.get("total_pending_lines"), 0)
    overlay_total = (
        _safe_int(backpressure.get("pressure_total_pending_lines"), raw_overlay_total)
        if managed_pressure_view
        else raw_overlay_total
    )
    overlay_adjusted = bool(backpressure.get("overlay_adjusted", False))
    direct_overlay_total = _safe_int(sql_overlay.get("total_pending_lines"), 0)
    direct_sql_overlay_clear = bool(
        sql_overlay.get("active", False)
        and direct_overlay_total == 0
        and _safe_int(sql_overlay.get("stale_pending_lines"), 0) == 0
        and _safe_int(sql_overlay.get("files_with_pending"), 0) == 0
        and not (sql_overlay.get("top_pending_files") if isinstance(sql_overlay.get("top_pending_files"), list) else [])
        and _safe_int(sql_overlay.get("fresh_source_count"), 0) > 0
        and _safe_int(sql_overlay.get("explicit_empty_source_count"), 0) > 0
        and _safe_float(sql_overlay.get("oldest_pending_age_seconds"), 0.0) <= 120.0
    )
    if direct_sql_overlay_clear:
        raw_core = 0
        raw_total = 0
        raw_oldest = 0.0
        overlay_total = 0
        overlay_adjusted = True
        raw_live_source = "fresh_empty_sql_ingestion_overlay"
    raw_live_clear = bool(
        (raw_live or direct_sql_overlay_clear)
        and raw_core <= OVERLAY_RAW_LIVE_MAX_CORE_LINES
        and raw_total <= OVERLAY_RAW_LIVE_MAX_TOTAL_LINES
        and raw_oldest <= OVERLAY_RAW_LIVE_MAX_AGE_SECONDS
    )
    bounded_raw_live_relief = bool(
        raw_live_clear
        and str(storage_severity or "").strip().lower() not in {"high", "critical", "blocked"}
        and _safe_float(storage_pressure_index) < 1.0
    )
    active = bool((overlay_adjusted and raw_live_clear) or bounded_raw_live_relief)
    bounded = bool(
        active
        and overlay_total <= OVERLAY_RUNTIME_MAX_TOTAL_LINES
    )
    return {
        "active": active,
        "bounded": bounded,
        "overlay_adjusted": overlay_adjusted,
        "bounded_raw_live_relief": bounded_raw_live_relief,
        "direct_sql_overlay_clear": direct_sql_overlay_clear,
        "overlay_total_pending_lines": overlay_total,
        "raw_overlay_total_pending_lines": raw_overlay_total,
        "managed_pressure_view": managed_pressure_view,
        "raw_live_clear": raw_live_clear,
        "raw_live": {
            "core_pending_lines": raw_core,
            "total_pending_lines": raw_total,
            "oldest_pending_age_seconds": round(raw_oldest, 3),
            "max_core_pending_lines": OVERLAY_RAW_LIVE_MAX_CORE_LINES,
            "max_total_pending_lines": OVERLAY_RAW_LIVE_MAX_TOTAL_LINES,
            "max_oldest_pending_age_seconds": OVERLAY_RAW_LIVE_MAX_AGE_SECONDS,
            "source": raw_live_source,
            "reconciled_from_raw_live": bool(raw_live.get("reconciled_from_raw_live", False)),
        },
        "storage_severity": "stable" if direct_sql_overlay_clear else str(storage_severity or ""),
        "storage_pressure_index": 0.0 if direct_sql_overlay_clear else round(_safe_float(storage_pressure_index), 3),
        "raw_storage_pressure_index": round(_safe_float(storage_pressure_index), 3),
        "max_overlay_total_pending_lines": OVERLAY_RUNTIME_MAX_TOTAL_LINES,
        "policy": "bounded raw-live or SQL-overlay relief prevents protect mode while pressure_index<1; live-money gates still consume strict storage evidence",
    }


def _soft_cap_low_pressure_advisory(
    *,
    overall_status: str,
    throttle_profile: str,
    saturation_score: float,
    compute_pressure_level: str,
    memory_pressure_level: str,
    storage_pressure_index: float,
    storage_fresh_overflow: bool,
    thermal_warning_active: bool,
    performance_warning_active: bool,
    host_pressure_attribution: dict[str, Any],
    live_read_only: bool = False,
    storage_severity: str = "",
    storage_core_pending_lines: int = 0,
    storage_total_pending_lines: int = 0,
    storage_pending_threshold: int = 15000,
    storage_oldest_pending_age_seconds: float = 0.0,
    storage_oldest_age_threshold_seconds: float = 240.0,
    storage_overlay_relief: dict[str, Any] | None = None,
    paper_execution_policy: dict[str, Any] | None = None,
    full_force_paper_required: bool = False,
) -> dict[str, Any]:
    interactive_cpu = _safe_float(host_pressure_attribution.get("foreground_app_cpu_percent"), 0.0)
    system_cpu = _safe_float(host_pressure_attribution.get("macos_system_cpu_percent"), 0.0)
    operator_cpu = _safe_float(host_pressure_attribution.get("operator_observability_cpu_percent"), 0.0)
    protected_cpu = _safe_float(host_pressure_attribution.get("protected_live_or_macro_cpu_percent"), 0.0)
    bot_owned_cpu = _safe_float(host_pressure_attribution.get("bot_owned_cpu_percent"), 0.0)
    bot_owned_non_operator_cpu = max(0.0, bot_owned_cpu - operator_cpu)
    support_cpu = _safe_float(host_pressure_attribution.get("throttle_candidate_support_cpu_percent"), 0.0)
    storage_writer_cpu = _safe_float(host_pressure_attribution.get("storage_writer_cpu_percent"), 0.0)
    paper_cpu = _safe_float(host_pressure_attribution.get("paper_execution_cpu_percent"), 0.0)
    research_cpu = _safe_float(host_pressure_attribution.get("research_training_cpu_percent"), 0.0)
    dominant_bucket = str(host_pressure_attribution.get("dominant_bucket") or "").strip().lower()
    protected_work_hot = bool(host_pressure_attribution.get("protected_work_hot", False))
    system_hot = bool(host_pressure_attribution.get("system_cotenant_hot", False))
    system_secondary_to_bot_owned = bool(host_pressure_attribution.get("system_secondary_to_bot_owned", False))
    operator_hot = bool(host_pressure_attribution.get("operator_observability_hot", False))
    operator_dominant = bool(host_pressure_attribution.get("operator_observability_pressure_dominant", False))
    storage_overlay_relief = storage_overlay_relief if isinstance(storage_overlay_relief, dict) else {}
    paper_execution_policy = paper_execution_policy if isinstance(paper_execution_policy, dict) else {}
    paper_execution_allowed = bool(paper_execution_policy.get("paper_execution_allowed", False))
    paper_execution_paused = bool(paper_execution_policy.get("pause_paper_execution", False))
    paper_ramp_pressure_recovery_probe = bool(paper_execution_policy.get("pressure_recovery_probe", False))
    paper_ramp_armed = bool(
        (paper_execution_policy.get("armed", False) and paper_execution_policy.get("ok", False))
        or paper_ramp_pressure_recovery_probe
    )
    overlay_runtime_relief_active = bool(storage_overlay_relief.get("bounded", False))
    foreground_only = bool(
        host_pressure_attribution.get("external_pressure_dominant", False)
        and dominant_bucket == "foreground_apps"
        and not system_hot
        and not bool(host_pressure_attribution.get("support_jobs_hot", False))
        and not protected_work_hot
    )
    foreground_system_guarded = bool(
        throttle_profile in {"soft_cap", "sustain"}
        and host_pressure_attribution.get("external_pressure_dominant", False)
        and dominant_bucket == "foreground_apps"
        and system_hot
        and interactive_cpu >= max(system_cpu, 35.0)
        and not bool(host_pressure_attribution.get("support_jobs_hot", False))
        and not protected_work_hot
        and saturation_score < 52.0
        and compute_pressure_level in {"normal", "elevated"}
    )
    classic_low_pressure = bool(
        throttle_profile == "soft_cap"
        and saturation_score < 50.0
        and compute_pressure_level == "normal"
        and interactive_cpu < 60.0
    )
    foreground_guarded = bool(
        throttle_profile in {"soft_cap", "sustain"}
        and foreground_only
        and saturation_score < 75.0
        and compute_pressure_level in {"normal", "elevated"}
        and interactive_cpu < 140.0
    )
    support_low_priority_guarded = bool(
        throttle_profile in {"soft_cap", "sustain"}
        and bool(host_pressure_attribution.get("support_jobs_hot", False))
        and bool(host_pressure_attribution.get("support_hot_low_priority", False))
        and (not system_hot or system_secondary_to_bot_owned)
        and (not protected_work_hot or protected_cpu < 100.0)
        and saturation_score < 90.0
        and compute_pressure_level in {"normal", "elevated", "high"}
    )
    research_low_priority_guarded = bool(
        throttle_profile in {"soft_cap", "sustain"}
        and bool(host_pressure_attribution.get("research_training_hot", False))
        and bool(host_pressure_attribution.get("research_hot_low_priority", False))
        and not bool(host_pressure_attribution.get("paper_execution_hot", False))
        and (not protected_work_hot or protected_cpu < 75.0)
        and saturation_score < 90.0
        and compute_pressure_level in {"normal", "elevated", "high"}
    )
    operator_observability_guarded = bool(
        throttle_profile in {"soft_cap", "sustain"}
        and bool(host_pressure_attribution.get("operator_observability_hot", False))
        and not bool(host_pressure_attribution.get("support_jobs_hot", False))
        and not bool(host_pressure_attribution.get("system_cotenant_hot", False))
        and not protected_work_hot
        and saturation_score < 68.0
        and compute_pressure_level in {"normal", "elevated"}
    )
    protected_work_guarded = bool(
        throttle_profile in {"soft_cap", "sustain"}
        and protected_work_hot
        and not bool(host_pressure_attribution.get("support_jobs_hot", False))
        and not bool(host_pressure_attribution.get("system_cotenant_hot", False))
        and saturation_score < 50.0
        and protected_cpu < 75.0
        and compute_pressure_level in {"normal", "elevated"}
    )
    external_cotenant_guarded = bool(
        throttle_profile in {"soft_cap", "sustain"}
        and bool(host_pressure_attribution.get("external_pressure_dominant", False))
        and not bool(host_pressure_attribution.get("bot_owned_pressure_dominant", False))
        and not bool(host_pressure_attribution.get("support_jobs_hot", False))
        and not bool(host_pressure_attribution.get("paper_execution_hot", False))
        and not bool(host_pressure_attribution.get("research_training_hot", False))
        and not bool(host_pressure_attribution.get("storage_writer_hot", False))
        and not protected_work_hot
        and (not operator_hot or operator_cpu < 80.0 or not operator_dominant)
        and saturation_score < 75.0
        and compute_pressure_level in {"normal", "elevated"}
    )
    external_high_compute_guarded = bool(
        throttle_profile == "sustain"
        and compute_pressure_level == "high"
        and bool(host_pressure_attribution.get("external_pressure_dominant", False))
        and not bool(host_pressure_attribution.get("bot_owned_pressure_dominant", False))
        and bot_owned_cpu < 60.0
        and not bool(host_pressure_attribution.get("support_jobs_hot", False))
        and not bool(host_pressure_attribution.get("paper_execution_hot", False))
        and not bool(host_pressure_attribution.get("research_training_hot", False))
        and not bool(host_pressure_attribution.get("storage_writer_hot", False))
        and not protected_work_hot
        and (not operator_hot or operator_cpu < 80.0 or not operator_dominant)
        and saturation_score < 75.0
    )
    bounded_storage_overlay_guarded = bool(
        (external_high_compute_guarded or external_cotenant_guarded or foreground_guarded or support_low_priority_guarded)
        and (
            overlay_runtime_relief_active
            or (
                str(storage_severity or "").strip().lower() not in {"high", "critical", "blocked"}
                and not storage_fresh_overflow
                and storage_pressure_index >= 0.5
                and storage_pressure_index < 0.85
                and int(storage_core_pending_lines) < max(int(storage_pending_threshold), 1)
                and int(storage_total_pending_lines) <= max(int(storage_pending_threshold * 1.25), int(storage_pending_threshold) + 1)
                and float(storage_oldest_pending_age_seconds) <= max(float(storage_oldest_age_threshold_seconds), 1.0)
            )
        )
    )
    storage_ready_for_runtime_advisory = bool(
        storage_pressure_index < 0.5
        or storage_fresh_overflow
        or bounded_storage_overlay_guarded
        or overlay_runtime_relief_active
    )
    paper_ramp_memory_guarded = bool(
        memory_pressure_level == "normal"
        or (
            memory_pressure_level == "elevated"
            and bool(live_read_only)
            and paper_execution_allowed
            and not paper_execution_paused
            and paper_ramp_armed
            and storage_ready_for_runtime_advisory
        )
    )
    operator_observability_high_compute_guarded = bool(
        throttle_profile == "sustain"
        and compute_pressure_level == "high"
        and memory_pressure_level == "normal"
        and storage_ready_for_runtime_advisory
        and bool(live_read_only)
        and paper_execution_allowed
        and not paper_execution_paused
        and paper_ramp_armed
        and operator_hot
        and operator_cpu <= 100.0
        and bot_owned_non_operator_cpu < 20.0
        and protected_cpu < 20.0
        and saturation_score < 75.0
        and not bool(host_pressure_attribution.get("support_jobs_hot", False))
        and not bool(host_pressure_attribution.get("paper_execution_hot", False))
        and not bool(host_pressure_attribution.get("research_training_hot", False))
        and not bool(host_pressure_attribution.get("storage_writer_hot", False))
        and not bool(host_pressure_attribution.get("system_cotenant_hot", False))
        and not protected_work_hot
        and not thermal_warning_active
        and not performance_warning_active
    )
    support_throttle_pending_guarded = bool(
        overall_status == "degraded"
        and throttle_profile in {"soft_cap", "sustain"}
        and bool(host_pressure_attribution.get("support_jobs_hot", False))
        and not bool(host_pressure_attribution.get("support_hot_low_priority", False))
        and bool(host_pressure_attribution.get("support_trim_required", False))
        and memory_pressure_level == "normal"
        and storage_ready_for_runtime_advisory
        and support_cpu <= 160.0
        and saturation_score < 90.0
        and not bool(host_pressure_attribution.get("paper_execution_hot", False))
        and not bool(host_pressure_attribution.get("research_training_hot", False))
        and not protected_work_hot
        and protected_cpu < 20.0
        and not thermal_warning_active
        and not performance_warning_active
    )
    plain_storage_clear_guarded_ready = bool(
        str(storage_severity or "").strip().lower() not in {"high", "critical", "blocked"}
        and storage_pressure_index < 0.5
        and int(storage_core_pending_lines) < max(int(storage_pending_threshold), 1)
        and int(storage_total_pending_lines) < max(int(storage_pending_threshold), 1)
        and float(storage_oldest_pending_age_seconds) <= max(float(storage_oldest_age_threshold_seconds), 1.0)
    )
    plain_foreground_live_read_only_guarded_ready = bool(
        plain_storage_clear_guarded_ready
        and bool(live_read_only)
        and foreground_guarded
        and not system_hot
    )
    plain_external_live_read_only_guarded_ready = bool(
        plain_storage_clear_guarded_ready
        and bool(live_read_only)
        and external_cotenant_guarded
    )
    storage_writer_cooling_guarded_ready = bool(
        overall_status == "degraded"
        and throttle_profile in {"soft_cap", "sustain"}
        and compute_pressure_level in {"normal", "elevated", "high"}
        and memory_pressure_level == "normal"
        and storage_ready_for_runtime_advisory
        and bool(host_pressure_attribution.get("storage_writer_hot", False))
        and storage_writer_cpu <= 110.0
        and bot_owned_cpu <= 150.0
        and protected_cpu < 20.0
        and operator_cpu < 30.0
        and saturation_score < 75.0
        and not bool(host_pressure_attribution.get("support_jobs_hot", False))
        and not bool(host_pressure_attribution.get("paper_execution_hot", False))
        and not bool(host_pressure_attribution.get("research_training_hot", False))
        and not protected_work_hot
        and not thermal_warning_active
        and not performance_warning_active
    )
    storage_writer_cooling_guarded_advisory = bool(
        overall_status == "degraded"
        and throttle_profile in {"soft_cap", "sustain"}
        and compute_pressure_level in {"normal", "elevated"}
        and memory_pressure_level in {"normal", "elevated"}
        and storage_ready_for_runtime_advisory
        and plain_storage_clear_guarded_ready
        and bool(live_read_only)
        and bool(host_pressure_attribution.get("storage_writer_hot", False))
        and storage_writer_cpu <= 110.0
        and bot_owned_cpu <= 170.0
        and protected_cpu < 20.0
        and operator_cpu < 45.0
        and saturation_score < 75.0
        and not bool(host_pressure_attribution.get("support_jobs_hot", False))
        and not bool(host_pressure_attribution.get("paper_execution_hot", False))
        and not bool(host_pressure_attribution.get("research_training_hot", False))
        and not protected_work_hot
        and not thermal_warning_active
        and not performance_warning_active
    )
    storage_writer_burst_complete_guarded_ready = bool(
        overall_status == "degraded"
        and throttle_profile in {"soft_cap", "sustain"}
        and compute_pressure_level == "normal"
        and memory_pressure_level == "normal"
        and storage_ready_for_runtime_advisory
        and plain_storage_clear_guarded_ready
        and bool(live_read_only)
        and bool(host_pressure_attribution.get("storage_writer_hot", False))
        and storage_writer_cpu <= 135.0
        and bot_owned_cpu <= 180.0
        and protected_cpu < 20.0
        and operator_cpu < 35.0
        and saturation_score < 50.0
        and not bool(host_pressure_attribution.get("support_jobs_hot", False))
        and not bool(host_pressure_attribution.get("paper_execution_hot", False))
        and not bool(host_pressure_attribution.get("research_training_hot", False))
        and not protected_work_hot
        and not thermal_warning_active
        and not performance_warning_active
    )
    support_throttle_pending_guarded_ready = bool(
        support_throttle_pending_guarded
        and bool(live_read_only)
        and plain_storage_clear_guarded_ready
        and support_cpu <= 80.0
        and saturation_score < 68.0
        and compute_pressure_level in {"normal", "elevated"}
    )
    support_low_priority_guarded_ready = bool(
        support_low_priority_guarded
        and bool(live_read_only)
        and (plain_storage_clear_guarded_ready or overlay_runtime_relief_active)
        and support_cpu <= 160.0
        and saturation_score < 75.0
        and compute_pressure_level in {"normal", "elevated"}
        and not bool(host_pressure_attribution.get("paper_execution_hot", False))
        and not bool(host_pressure_attribution.get("research_training_hot", False))
        and not bool(host_pressure_attribution.get("storage_writer_hot", False))
        and not protected_work_hot
    )
    bounded_writer_support_base_cpu_bounded = bool(
        support_cpu <= BOUNDED_WRITER_SUPPORT_CPU_THRESHOLD
    )
    bounded_writer_support_sampling_hysteresis_guarded = bool(
        not bounded_writer_support_base_cpu_bounded
        and bool(host_pressure_attribution.get("support_hot_low_priority", False))
        and compute_pressure_level == "normal"
        and saturation_score < BOUNDED_WRITER_SUPPORT_HYSTERESIS_MAX_HOST_SATURATION
        and storage_writer_cpu <= BOUNDED_WRITER_SUPPORT_HYSTERESIS_MAX_WRITER_CPU
        and support_cpu
        <= BOUNDED_WRITER_SUPPORT_CPU_THRESHOLD * BOUNDED_WRITER_SUPPORT_SAMPLING_HYSTERESIS_RATIO
    )
    bounded_writer_support_cpu_capacity_guarded = bool(
        bounded_writer_support_base_cpu_bounded or bounded_writer_support_sampling_hysteresis_guarded
    )
    bounded_writer_with_support_guarded_ready = bool(
        overall_status == "degraded"
        and throttle_profile in {"soft_cap", "sustain"}
        and memory_pressure_level == "normal"
        and storage_ready_for_runtime_advisory
        and bool(live_read_only)
        and bool(host_pressure_attribution.get("storage_writer_hot", False))
        and bool(host_pressure_attribution.get("support_jobs_hot", False))
        and (support_throttle_pending_guarded_ready or support_low_priority_guarded)
        and storage_writer_cpu <= 110.0
        and bounded_writer_support_cpu_capacity_guarded
        and bot_owned_cpu <= 220.0
        and protected_cpu < 20.0
        and operator_cpu < 35.0
        and saturation_score < 72.0
        and not bool(host_pressure_attribution.get("paper_execution_hot", False))
        and not bool(host_pressure_attribution.get("research_training_hot", False))
        and not protected_work_hot
        and not thermal_warning_active
        and not performance_warning_active
    )
    bounded_writer_with_paper_shadow_guarded_ready = bool(
        overall_status == "degraded"
        and throttle_profile in {"soft_cap", "sustain"}
        and compute_pressure_level in {"normal", "elevated"}
        and memory_pressure_level == "normal"
        and storage_ready_for_runtime_advisory
        and bool(live_read_only)
        and bool(host_pressure_attribution.get("storage_writer_hot", False))
        and (
            (
                bool(host_pressure_attribution.get("paper_execution_hot", False))
                and bool(host_pressure_attribution.get("paper_hot_low_priority", False))
            )
            or (
                paper_ramp_pressure_recovery_probe
                and not bool(host_pressure_attribution.get("paper_execution_hot", False))
            )
        )
        and storage_writer_cpu <= 110.0
        and paper_cpu <= 100.0
        and bot_owned_cpu <= 240.0
        and protected_cpu < 20.0
        and operator_cpu < 35.0
        and saturation_score < 72.0
        and not bool(host_pressure_attribution.get("support_jobs_hot", False))
        and not bool(host_pressure_attribution.get("research_training_hot", False))
        and not protected_work_hot
        and not thermal_warning_active
        and not performance_warning_active
    )
    full_force_paper_base_cpu_bounded = bool(
        paper_cpu <= FULL_FORCE_PAPER_CAPACITY_LIMIT_CPU_THRESHOLD
        and bot_owned_cpu <= FULL_FORCE_PAPER_BOT_OWNED_CPU_THRESHOLD
    )
    full_force_paper_sampling_hysteresis_guarded = bool(
        not full_force_paper_base_cpu_bounded
        and compute_pressure_level == "normal"
        and saturation_score < FULL_FORCE_PAPER_HYSTERESIS_MAX_HOST_SATURATION
        and storage_writer_cpu <= FULL_FORCE_PAPER_HYSTERESIS_MAX_WRITER_CPU
        and paper_cpu
        <= FULL_FORCE_PAPER_CAPACITY_LIMIT_CPU_THRESHOLD * FULL_FORCE_PAPER_SAMPLING_HYSTERESIS_RATIO
        and bot_owned_cpu
        <= FULL_FORCE_PAPER_BOT_OWNED_CPU_THRESHOLD * FULL_FORCE_PAPER_SAMPLING_HYSTERESIS_RATIO
    )
    full_force_paper_cpu_capacity_guarded = bool(
        full_force_paper_base_cpu_bounded or full_force_paper_sampling_hysteresis_guarded
    )
    full_force_paper_ramp_guarded_ready = bool(
        overall_status == "degraded"
        and bool(full_force_paper_required)
        and throttle_profile in {"soft_cap", "sustain"}
        and compute_pressure_level in {"normal", "elevated", "high"}
        and memory_pressure_level == "normal"
        and storage_ready_for_runtime_advisory
        and bool(live_read_only)
        and paper_execution_allowed
        and not paper_execution_paused
        and paper_ramp_armed
        and (
            compute_pressure_level == "high"
            or bool(host_pressure_attribution.get("storage_writer_hot", False))
            or (
                compute_pressure_level == "elevated"
                and saturation_score < FULL_FORCE_PAPER_ELEVATED_MAX_HOST_SATURATION
            )
            or (compute_pressure_level == "normal" and saturation_score < 50.0)
        )
        and (
            not bool(host_pressure_attribution.get("storage_writer_hot", False))
            or storage_writer_cpu <= 190.0
        )
        and bool(host_pressure_attribution.get("paper_execution_hot", False))
        and bool(host_pressure_attribution.get("paper_hot_low_priority", False))
        and storage_writer_cpu <= 190.0
        and full_force_paper_cpu_capacity_guarded
        and protected_cpu < 20.0
        and operator_cpu < 45.0
        and saturation_score < 75.0
        and (
            not bool(host_pressure_attribution.get("support_jobs_hot", False))
            or (
                support_cpu <= 80.0
                and not bool(host_pressure_attribution.get("support_pressure_dominant", False))
            )
        )
        and (
            not bool(host_pressure_attribution.get("research_training_hot", False))
            or (
                research_cpu <= FULL_FORCE_PAPER_BOUNDED_RESEARCH_CPU_THRESHOLD
                and bool(host_pressure_attribution.get("research_hot_low_priority", False))
            )
        )
        and not protected_work_hot
        and not thermal_warning_active
        and not performance_warning_active
    )
    paper_lane_low_priority_guarded = bool(
        overall_status == "degraded"
        and throttle_profile in {"soft_cap", "sustain"}
        and compute_pressure_level in {"normal", "elevated"}
        and paper_ramp_memory_guarded
        and storage_ready_for_runtime_advisory
        and bool(live_read_only)
        and paper_execution_allowed
        and not paper_execution_paused
        and paper_ramp_armed
        and bool(host_pressure_attribution.get("paper_execution_hot", False))
        and bool(host_pressure_attribution.get("paper_hot_low_priority", False))
        and paper_cpu <= 125.0
        and bot_owned_cpu <= 220.0
        and protected_cpu < 20.0
        and operator_cpu < 35.0
        and saturation_score < 68.0
        and not bool(host_pressure_attribution.get("support_jobs_hot", False))
        and not bool(host_pressure_attribution.get("research_training_hot", False))
        and not bool(host_pressure_attribution.get("storage_writer_hot", False))
        and not protected_work_hot
        and not thermal_warning_active
        and not performance_warning_active
    )
    full_force_paper_research_mix_guarded_advisory = bool(
        overall_status == "degraded"
        and throttle_profile in {"soft_cap", "sustain"}
        and compute_pressure_level in {"normal", "elevated"}
        and memory_pressure_level == "normal"
        and storage_ready_for_runtime_advisory
        and bool(live_read_only)
        and paper_execution_allowed
        and not paper_execution_paused
        and paper_ramp_armed
        and bool(host_pressure_attribution.get("paper_execution_hot", False))
        and bool(host_pressure_attribution.get("paper_hot_low_priority", False))
        and bool(host_pressure_attribution.get("research_training_hot", False))
        and bool(host_pressure_attribution.get("research_hot_low_priority", False))
        and paper_cpu <= 125.0
        and research_cpu <= 220.0
        and bot_owned_cpu <= 340.0
        and support_cpu <= 80.0
        and protected_cpu < 20.0
        and operator_cpu < 45.0
        and saturation_score < 75.0
        and not bool(host_pressure_attribution.get("support_jobs_hot", False))
        and not bool(host_pressure_attribution.get("storage_writer_hot", False))
        and not protected_work_hot
        and not thermal_warning_active
        and not performance_warning_active
    )
    bounded_bot_owned_runtime_guarded_ready = bool(
        overall_status == "degraded"
        and throttle_profile in {"soft_cap", "sustain"}
        and compute_pressure_level in {"normal", "elevated"}
        and memory_pressure_level == "normal"
        and storage_ready_for_runtime_advisory
        and bool(live_read_only)
        and bool(host_pressure_attribution.get("storage_writer_hot", False))
        and not bool(host_pressure_attribution.get("support_jobs_hot", False))
        and not bool(host_pressure_attribution.get("paper_execution_hot", False))
        and not bool(host_pressure_attribution.get("research_training_hot", False))
        and not protected_work_hot
        and (paper_cpu > 0.0 or research_cpu > 0.0)
        and storage_writer_cpu <= 110.0
        and paper_cpu <= 60.0
        and research_cpu <= 60.0
        and support_cpu <= 20.0
        and protected_cpu < 20.0
        and operator_cpu < 35.0
        and bot_owned_cpu <= 220.0
        and saturation_score < 50.0
        and not thermal_warning_active
        and not performance_warning_active
    )
    bounded_writer_support_protected_guarded_ready = bool(
        overall_status == "degraded"
        and throttle_profile in {"soft_cap", "sustain"}
        and compute_pressure_level in {"normal", "elevated"}
        and memory_pressure_level == "normal"
        and storage_ready_for_runtime_advisory
        and bool(live_read_only)
        and bool(host_pressure_attribution.get("storage_writer_hot", False))
        and bool(host_pressure_attribution.get("support_jobs_hot", False))
        and protected_work_hot
        and (support_low_priority_guarded or support_throttle_pending_guarded_ready or support_low_priority_guarded_ready)
        and storage_writer_cpu <= 110.0
        and support_cpu <= 90.0
        and protected_cpu <= 75.0
        and bot_owned_cpu <= 280.0
        and operator_cpu < 35.0
        and saturation_score < 62.0
        and not bool(host_pressure_attribution.get("paper_execution_hot", False))
        and not bool(host_pressure_attribution.get("research_training_hot", False))
        and not thermal_warning_active
        and not performance_warning_active
    )
    bounded_protected_lane_guarded_ready = bool(
        overall_status == "degraded"
        and throttle_profile in {"soft_cap", "sustain"}
        and compute_pressure_level in {"normal", "elevated"}
        and memory_pressure_level == "normal"
        and storage_ready_for_runtime_advisory
        and bool(live_read_only)
        and protected_work_hot
        and protected_cpu <= 75.0
        and bot_owned_cpu <= max(95.0, protected_cpu + 25.0)
        and operator_cpu < 30.0
        and saturation_score < 62.0
        and not bool(host_pressure_attribution.get("support_jobs_hot", False))
        and not bool(host_pressure_attribution.get("paper_execution_hot", False))
        and not bool(host_pressure_attribution.get("research_training_hot", False))
        and not bool(host_pressure_attribution.get("storage_writer_hot", False))
        and not thermal_warning_active
        and not performance_warning_active
    )
    runtime_ready_guarded = bool(
        storage_writer_cooling_guarded_ready
        or storage_writer_burst_complete_guarded_ready
        or bounded_writer_with_support_guarded_ready
        or bounded_writer_with_paper_shadow_guarded_ready
        or full_force_paper_ramp_guarded_ready
        or bounded_bot_owned_runtime_guarded_ready
        or bounded_writer_support_protected_guarded_ready
        or support_throttle_pending_guarded_ready
        or support_low_priority_guarded_ready
        or bounded_protected_lane_guarded_ready
        or (
            overall_status == "degraded"
            and (classic_low_pressure or foreground_guarded or foreground_system_guarded or external_cotenant_guarded)
            and memory_pressure_level == "normal"
            and compute_pressure_level in {"normal", "elevated"}
            and storage_ready_for_runtime_advisory
            and (
                plain_foreground_live_read_only_guarded_ready
                or plain_external_live_read_only_guarded_ready
                or bounded_storage_overlay_guarded
            )
            and not thermal_warning_active
            and not performance_warning_active
            and bool(host_pressure_attribution.get("external_pressure_dominant", False))
            and not bool(host_pressure_attribution.get("bot_owned_pressure_dominant", False))
            and not bool(host_pressure_attribution.get("support_jobs_hot", False))
            and not bool(host_pressure_attribution.get("paper_execution_hot", False))
            and not bool(host_pressure_attribution.get("research_training_hot", False))
            and not bool(host_pressure_attribution.get("storage_writer_hot", False))
            and not protected_work_hot
            and bot_owned_non_operator_cpu < 20.0
            and protected_cpu < 20.0
            and operator_cpu < 30.0
            and saturation_score < 62.0
        )
    )
    active = bool(
        overall_status == "degraded"
        and (
            classic_low_pressure
            or foreground_guarded
            or foreground_system_guarded
            or support_low_priority_guarded
            or support_throttle_pending_guarded
            or research_low_priority_guarded
            or operator_observability_guarded
            or operator_observability_high_compute_guarded
            or protected_work_guarded
            or external_cotenant_guarded
            or external_high_compute_guarded
            or paper_lane_low_priority_guarded
            or full_force_paper_research_mix_guarded_advisory
            or storage_writer_cooling_guarded_advisory
            or runtime_ready_guarded
            or storage_writer_cooling_guarded_ready
            or storage_writer_burst_complete_guarded_ready
            or support_throttle_pending_guarded_ready
            or support_low_priority_guarded_ready
            or bounded_writer_with_support_guarded_ready
            or bounded_writer_with_paper_shadow_guarded_ready
            or full_force_paper_ramp_guarded_ready
            or bounded_bot_owned_runtime_guarded_ready
            or bounded_writer_support_protected_guarded_ready
            or bounded_protected_lane_guarded_ready
        )
        and (
            memory_pressure_level == "normal"
            or (
                memory_pressure_level == "elevated"
                and paper_lane_low_priority_guarded
            )
            or (
                memory_pressure_level == "elevated"
                and storage_writer_cooling_guarded_advisory
            )
        )
        and storage_ready_for_runtime_advisory
        and not thermal_warning_active
        and not performance_warning_active
        and (
            not system_hot
            or foreground_system_guarded
            or support_low_priority_guarded
            or support_throttle_pending_guarded
            or research_low_priority_guarded
            or operator_observability_high_compute_guarded
            or paper_lane_low_priority_guarded
            or full_force_paper_research_mix_guarded_advisory
            or storage_writer_cooling_guarded_advisory
            or external_cotenant_guarded
            or external_high_compute_guarded
            or storage_writer_cooling_guarded_ready
            or storage_writer_burst_complete_guarded_ready
            or support_throttle_pending_guarded_ready
            or support_low_priority_guarded_ready
            or bounded_writer_with_support_guarded_ready
            or bounded_writer_with_paper_shadow_guarded_ready
            or full_force_paper_ramp_guarded_ready
            or bounded_bot_owned_runtime_guarded_ready
            or bounded_writer_support_protected_guarded_ready
            or bounded_protected_lane_guarded_ready
        )
        and (
            not protected_work_hot
            or protected_work_guarded
            or bounded_protected_lane_guarded_ready
            or bounded_writer_support_protected_guarded_ready
        )
    )
    reason = "soft_cap_still_requires_degraded_posture"
    if active and runtime_ready_guarded and bounded_storage_overlay_guarded and external_cotenant_guarded:
        reason = "external_cotenant_with_bounded_storage_overlay_is_guarded_runtime_ready"
    elif active and runtime_ready_guarded and foreground_guarded:
        reason = "foreground_cotenant_pressure_is_guarded_runtime_ready"
    elif active and runtime_ready_guarded and plain_external_live_read_only_guarded_ready:
        reason = "external_cotenant_pressure_with_clean_storage_is_guarded_runtime_ready"
    elif active and runtime_ready_guarded:
        if bounded_writer_with_support_guarded_ready:
            if bounded_writer_support_sampling_hysteresis_guarded:
                reason = "bounded_writer_and_niced_support_sampling_hysteresis_is_guarded_runtime_ready"
            elif support_low_priority_guarded:
                reason = "bounded_writer_and_niced_support_is_guarded_runtime_ready"
            else:
                reason = "bounded_writer_and_support_throttle_pending_is_guarded_runtime_ready"
        elif full_force_paper_ramp_guarded_ready:
            if paper_ramp_pressure_recovery_probe:
                reason = "paper_ramp_pressure_only_cycle_recovery_is_guarded_runtime_ready"
            elif full_force_paper_sampling_hysteresis_guarded:
                reason = "full_force_paper_sampling_hysteresis_is_guarded_runtime_ready"
            elif bool(host_pressure_attribution.get("storage_writer_hot", False)):
                reason = "full_force_paper_ramp_writer_pressure_is_guarded_runtime_ready"
            else:
                reason = "full_force_paper_ramp_pressure_is_guarded_runtime_ready"
        elif bounded_writer_with_paper_shadow_guarded_ready:
            reason = "bounded_writer_and_low_priority_paper_shadow_is_guarded_runtime_ready"
        elif bounded_bot_owned_runtime_guarded_ready:
            reason = "bounded_bot_owned_writer_paper_research_is_guarded_runtime_ready"
        elif bounded_writer_support_protected_guarded_ready:
            reason = "bounded_writer_support_and_read_only_protected_lane_is_guarded_runtime_ready"
        elif storage_writer_cooling_guarded_ready:
            reason = "single_bounded_storage_writer_after_green_backpressure_is_guarded_runtime_ready"
        elif storage_writer_burst_complete_guarded_ready:
            reason = "bounded_storage_writer_burst_after_clear_backpressure_is_guarded_runtime_ready"
        elif support_throttle_pending_guarded_ready:
            reason = "support_throttle_pending_after_green_backpressure_is_guarded_runtime_ready"
        elif support_low_priority_guarded_ready:
            reason = "niced_support_pressure_after_green_backpressure_is_guarded_runtime_ready"
        elif bounded_protected_lane_guarded_ready:
            reason = "bounded_read_only_protected_lane_after_green_backpressure_is_guarded_runtime_ready"
        else:
            reason = "runtime_pressure_is_guarded_ready"
    elif active and foreground_system_guarded:
        reason = "foreground_and_macos_system_mix_is_guarded_advisory"
    elif active and support_low_priority_guarded and system_secondary_to_bot_owned:
        reason = "niced_support_maintenance_with_secondary_system_pressure_is_guarded_advisory"
    elif active and support_low_priority_guarded:
        reason = "support_pressure_is_already_niced_and_guarded_advisory"
    elif active and support_throttle_pending_guarded:
        reason = "support_pressure_is_throttle_pending_guarded_advisory"
    elif active and research_low_priority_guarded:
        reason = "research_training_pressure_is_already_niced_and_guarded_advisory"
    elif active and paper_lane_low_priority_guarded:
        reason = "low_priority_paper_execution_pressure_is_guarded_advisory"
    elif active and full_force_paper_research_mix_guarded_advisory:
        reason = "full_force_paper_and_research_pressure_is_soak_guarded_advisory"
    elif active and storage_writer_cooling_guarded_advisory:
        reason = "bounded_storage_writer_after_green_backpressure_is_guarded_advisory"
    elif active and protected_work_guarded:
        reason = "protected_live_or_macro_work_is_guarded_advisory"
    elif active and operator_observability_high_compute_guarded:
        reason = "operator_observability_high_compute_pressure_is_capacity_limited_advisory_not_bot_runtime_degradation"
    elif active and operator_observability_guarded:
        reason = "operator_observability_pressure_is_guarded_advisory"
    elif active and bounded_storage_overlay_guarded and external_cotenant_guarded:
        reason = "external_cotenant_with_bounded_storage_overlay_is_advisory_not_bot_runtime_degradation"
    elif active and bounded_storage_overlay_guarded and foreground_guarded:
        reason = "foreground_cotenant_with_bounded_storage_overlay_is_advisory_not_bot_runtime_degradation"
    elif active and foreground_guarded:
        reason = "foreground_cotenant_pressure_is_guarded_advisory_not_bot_runtime_degradation"
    elif active and external_high_compute_guarded and bounded_storage_overlay_guarded:
        reason = "external_high_compute_with_bounded_storage_overlay_is_capacity_limited_advisory"
    elif active and external_high_compute_guarded:
        reason = "external_high_compute_pressure_is_capacity_limited_advisory_not_bot_runtime_degradation"
    elif active and external_cotenant_guarded:
        reason = "external_cotenant_pressure_is_guarded_advisory_not_bot_runtime_degradation"
    elif active:
        reason = "soft_cap_metrics_are_calm_external_or_foreground_activity_is_attribution_only"
    return {
        "active": active,
        "from_status": overall_status,
        "to_status": "ready" if (active and runtime_ready_guarded) else ("advisory" if active else overall_status),
        "reason": reason,
        "thresholds": {
            "max_advisory_host_saturation_score": 50.0,
            "max_guarded_ready_host_saturation_score": 62.0,
            "max_guarded_storage_writer_host_saturation_score": 75.0,
            "max_guarded_ready_bot_owned_cpu_percent": 20.0,
            "max_guarded_ready_protected_cpu_percent": 20.0,
            "max_guarded_ready_operator_cpu_percent": 30.0,
            "max_guarded_ready_full_force_operator_cpu_percent": 45.0,
            "max_guarded_ready_protected_lane_cpu_percent": 75.0,
            "max_guarded_ready_bot_owned_with_protected_lane_cpu_percent": 95.0,
            "max_guarded_ready_bounded_bot_owned_cpu_percent": 220.0,
            "max_guarded_ready_bounded_writer_support_cpu_percent": BOUNDED_WRITER_SUPPORT_CPU_THRESHOLD,
            "max_guarded_ready_bounded_writer_support_hysteresis_cpu_percent": (
                BOUNDED_WRITER_SUPPORT_CPU_THRESHOLD * BOUNDED_WRITER_SUPPORT_SAMPLING_HYSTERESIS_RATIO
            ),
            "max_guarded_ready_bounded_writer_support_hysteresis_host_saturation_score": (
                BOUNDED_WRITER_SUPPORT_HYSTERESIS_MAX_HOST_SATURATION
            ),
            "max_guarded_ready_bounded_paper_cpu_percent": 60.0,
            "max_guarded_ready_full_force_paper_cpu_percent": FULL_FORCE_PAPER_CAPACITY_LIMIT_CPU_THRESHOLD,
            "max_guarded_ready_full_force_paper_hysteresis_cpu_percent": (
                FULL_FORCE_PAPER_CAPACITY_LIMIT_CPU_THRESHOLD * FULL_FORCE_PAPER_SAMPLING_HYSTERESIS_RATIO
            ),
            "max_guarded_ready_full_force_hysteresis_storage_writer_cpu_percent": (
                FULL_FORCE_PAPER_HYSTERESIS_MAX_WRITER_CPU
            ),
            "max_guarded_ready_full_force_hysteresis_host_saturation_score": (
                FULL_FORCE_PAPER_HYSTERESIS_MAX_HOST_SATURATION
            ),
            "max_guarded_ready_full_force_elevated_host_saturation_score": (
                FULL_FORCE_PAPER_ELEVATED_MAX_HOST_SATURATION
            ),
            "max_guarded_ready_full_force_storage_writer_cpu_percent": 190.0,
            "max_guarded_ready_full_force_bounded_support_cpu_percent": 80.0,
            "max_guarded_ready_writer_burst_complete_cpu_percent": 135.0,
            "max_guarded_ready_full_force_bot_owned_cpu_percent": FULL_FORCE_PAPER_BOT_OWNED_CPU_THRESHOLD,
            "max_guarded_ready_full_force_bot_owned_hysteresis_cpu_percent": (
                FULL_FORCE_PAPER_BOT_OWNED_CPU_THRESHOLD * FULL_FORCE_PAPER_SAMPLING_HYSTERESIS_RATIO
            ),
            "max_guarded_ready_full_force_host_saturation_score": 75.0,
            "max_guarded_ready_bounded_research_cpu_percent": FULL_FORCE_PAPER_BOUNDED_RESEARCH_CPU_THRESHOLD,
            "max_guarded_foreground_host_saturation_score": 62.0,
            "max_guarded_niced_support_host_saturation_score": 68.0,
            "max_guarded_niced_support_ready_host_saturation_score": 75.0,
            "max_guarded_niced_support_ready_cpu_percent": 160.0,
            "max_guarded_operator_observability_host_saturation_score": 68.0,
            "max_guarded_operator_observability_high_compute_cpu_percent": 100.0,
            "max_guarded_external_cotenant_host_saturation_score": 75.0,
            "max_guarded_external_high_compute_host_saturation_score": 75.0,
            "max_guarded_external_high_compute_bot_owned_cpu_percent": 60.0,
            "max_guarded_protected_work_host_saturation_score": 50.0,
            "max_guarded_protected_work_cpu_percent": 75.0,
            "max_advisory_storage_pressure_index": 0.5,
            "max_guarded_storage_overlay_pressure_index": 0.85,
            "max_advisory_foreground_cpu_percent": 60.0,
            "max_guarded_foreground_cpu_percent": 140.0,
        },
        "measurements": {
            "host_saturation_score": round(float(saturation_score), 3),
            "compute_pressure_level": compute_pressure_level,
            "memory_pressure_level": memory_pressure_level,
            "storage_pressure_index": round(float(storage_pressure_index), 3),
            "storage_severity": str(storage_severity or ""),
            "storage_core_pending_lines": int(storage_core_pending_lines),
            "storage_total_pending_lines": int(storage_total_pending_lines),
            "storage_pending_threshold": int(storage_pending_threshold),
            "storage_oldest_pending_age_seconds": round(float(storage_oldest_pending_age_seconds), 3),
            "storage_oldest_age_threshold_seconds": round(float(storage_oldest_age_threshold_seconds), 3),
            "storage_fresh_overflow": bool(storage_fresh_overflow),
            "bounded_storage_overlay_guarded": bounded_storage_overlay_guarded,
            "overlay_runtime_relief_active": overlay_runtime_relief_active,
            "storage_overlay_relief": storage_overlay_relief,
            "plain_storage_clear_guarded_ready": plain_storage_clear_guarded_ready,
            "plain_foreground_live_read_only_guarded_ready": plain_foreground_live_read_only_guarded_ready,
            "plain_external_live_read_only_guarded_ready": plain_external_live_read_only_guarded_ready,
            "storage_ready_for_runtime_advisory": storage_ready_for_runtime_advisory,
            "runtime_ready_guarded": runtime_ready_guarded,
            "storage_writer_cooling_guarded_ready": storage_writer_cooling_guarded_ready,
            "storage_writer_cooling_guarded_advisory": storage_writer_cooling_guarded_advisory,
            "storage_writer_burst_complete_guarded_ready": storage_writer_burst_complete_guarded_ready,
            "support_throttle_pending_guarded": support_throttle_pending_guarded,
            "support_throttle_pending_guarded_ready": support_throttle_pending_guarded_ready,
            "support_low_priority_guarded_ready": support_low_priority_guarded_ready,
            "bounded_writer_with_support_guarded_ready": bounded_writer_with_support_guarded_ready,
            "bounded_writer_support_base_cpu_bounded": bounded_writer_support_base_cpu_bounded,
            "bounded_writer_support_sampling_hysteresis_guarded": (
                bounded_writer_support_sampling_hysteresis_guarded
            ),
            "bounded_writer_support_cpu_capacity_guarded": bounded_writer_support_cpu_capacity_guarded,
            "bounded_writer_with_paper_shadow_guarded_ready": bounded_writer_with_paper_shadow_guarded_ready,
            "full_force_paper_ramp_guarded_ready": full_force_paper_ramp_guarded_ready,
            "full_force_paper_base_cpu_bounded": full_force_paper_base_cpu_bounded,
            "full_force_paper_sampling_hysteresis_guarded": full_force_paper_sampling_hysteresis_guarded,
            "full_force_paper_cpu_capacity_guarded": full_force_paper_cpu_capacity_guarded,
            "full_force_paper_required": bool(full_force_paper_required),
            "paper_ramp_pressure_recovery_probe": paper_ramp_pressure_recovery_probe,
            "paper_lane_low_priority_guarded": paper_lane_low_priority_guarded,
            "full_force_paper_research_mix_guarded_advisory": full_force_paper_research_mix_guarded_advisory,
            "paper_ramp_memory_guarded": paper_ramp_memory_guarded,
            "bounded_bot_owned_runtime_guarded_ready": bounded_bot_owned_runtime_guarded_ready,
            "bounded_writer_support_protected_guarded_ready": bounded_writer_support_protected_guarded_ready,
            "bounded_protected_lane_guarded_ready": bounded_protected_lane_guarded_ready,
            "live_read_only": bool(live_read_only),
            "foreground_app_cpu_percent": round(float(interactive_cpu), 3),
            "macos_system_cpu_percent": round(float(system_cpu), 3),
            "foreground_only": foreground_only,
            "foreground_system_guarded": foreground_system_guarded,
            "foreground_guarded": foreground_guarded,
            "external_cotenant_guarded": external_cotenant_guarded,
            "external_high_compute_guarded": external_high_compute_guarded,
            "system_secondary_to_bot_owned": system_secondary_to_bot_owned,
            "support_low_priority_guarded": support_low_priority_guarded,
            "throttle_candidate_support_cpu_percent": round(float(support_cpu), 3),
            "support_hot_low_priority": bool(host_pressure_attribution.get("support_hot_low_priority", False)),
            "research_low_priority_guarded": research_low_priority_guarded,
            "research_training_hot": bool(host_pressure_attribution.get("research_training_hot", False)),
            "research_hot_low_priority": bool(host_pressure_attribution.get("research_hot_low_priority", False)),
            "research_training_cpu_percent": round(float(research_cpu), 3),
            "operator_observability_guarded": operator_observability_guarded,
            "operator_observability_high_compute_guarded": operator_observability_high_compute_guarded,
            "operator_observability_hot": bool(host_pressure_attribution.get("operator_observability_hot", False)),
            "operator_observability_cpu_percent": round(float(operator_cpu), 3),
            "protected_work_guarded": protected_work_guarded,
            "protected_work_hot": protected_work_hot,
            "protected_live_or_macro_cpu_percent": round(float(protected_cpu), 3),
            "bot_owned_cpu_percent": round(float(bot_owned_cpu), 3),
            "bot_owned_non_operator_cpu_percent": round(float(bot_owned_non_operator_cpu), 3),
            "storage_writer_cpu_percent": round(float(storage_writer_cpu), 3),
            "external_pressure_dominant": bool(host_pressure_attribution.get("external_pressure_dominant", False)),
            "bot_owned_pressure_dominant": bool(host_pressure_attribution.get("bot_owned_pressure_dominant", False)),
            "support_jobs_hot": bool(host_pressure_attribution.get("support_jobs_hot", False)),
            "paper_execution_hot": bool(host_pressure_attribution.get("paper_execution_hot", False)),
            "paper_hot_low_priority": bool(host_pressure_attribution.get("paper_hot_low_priority", False)),
            "paper_execution_cpu_percent": round(float(paper_cpu), 3),
            "paper_execution_allowed": paper_execution_allowed,
            "paper_execution_paused": paper_execution_paused,
            "paper_ramp_armed": paper_ramp_armed,
            "storage_writer_hot": bool(host_pressure_attribution.get("storage_writer_hot", False)),
        },
        "policy": "do_not_block_runtime_health_on_bounded_external_or_storage_overlay_pressure",
    }


def _registry_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows = payload.get("sub_bots") if isinstance(payload.get("sub_bots"), list) else []
    return [row for row in rows if isinstance(row, dict)]


def _registry_capacity_counts(project_root: Path, *, registry_path: Path | None = None) -> dict[str, Any]:
    effective_registry_path = registry_path if registry_path is not None else Path("master_bot_registry.json")
    path = effective_registry_path if effective_registry_path.is_absolute() else project_root / effective_registry_path
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
    storage_backpressure: dict[str, Any] | None = None,
    storage_severity: str = "",
    advisory_reclassification: dict[str, Any] | None = None,
) -> dict[str, Any]:
    active_count = _safe_int(counts.get("active_bot_count"), 0)
    full_force_required = active_count >= FULL_FORCE_PAPER_BOT_FLOOR
    advisory_reclassification = advisory_reclassification if isinstance(advisory_reclassification, dict) else {}
    advisory_measurements = (
        advisory_reclassification.get("measurements")
        if isinstance(advisory_reclassification.get("measurements"), dict)
        else {}
    )
    attribution_capacity_advisory = bool(
        advisory_reclassification.get("active", False)
        and str(advisory_reclassification.get("reason") or "")
        in {
            "external_high_compute_pressure_is_capacity_limited_advisory_not_bot_runtime_degradation",
            "external_high_compute_with_bounded_storage_overlay_is_capacity_limited_advisory",
            "operator_observability_high_compute_pressure_is_capacity_limited_advisory_not_bot_runtime_degradation",
            "single_bounded_storage_writer_after_green_backpressure_is_guarded_runtime_ready",
            "bounded_writer_and_support_throttle_pending_is_guarded_runtime_ready",
            "bounded_writer_and_low_priority_paper_shadow_is_guarded_runtime_ready",
            "full_force_paper_ramp_writer_pressure_is_guarded_runtime_ready",
            "full_force_paper_ramp_pressure_is_guarded_runtime_ready",
            "bounded_bot_owned_writer_paper_research_is_guarded_runtime_ready",
            "support_throttle_pending_after_green_backpressure_is_guarded_runtime_ready",
            "niced_support_pressure_after_green_backpressure_is_guarded_runtime_ready",
            "bounded_read_only_protected_lane_after_green_backpressure_is_guarded_runtime_ready",
            "support_pressure_is_already_niced_and_guarded_advisory",
            "support_pressure_is_throttle_pending_guarded_advisory",
            "research_training_pressure_is_already_niced_and_guarded_advisory",
            "full_force_paper_and_research_pressure_is_soak_guarded_advisory",
        }
        and (
            bool(advisory_measurements.get("external_high_compute_guarded", False))
            or bool(advisory_measurements.get("storage_writer_cooling_guarded_ready", False))
            or bool(advisory_measurements.get("bounded_writer_with_support_guarded_ready", False))
            or bool(advisory_measurements.get("bounded_bot_owned_runtime_guarded_ready", False))
            or bool(advisory_measurements.get("bounded_writer_with_paper_shadow_guarded_ready", False))
            or bool(advisory_measurements.get("full_force_paper_ramp_guarded_ready", False))
            or bool(advisory_measurements.get("support_throttle_pending_guarded_ready", False))
            or bool(advisory_measurements.get("support_low_priority_guarded_ready", False))
            or bool(advisory_measurements.get("bounded_protected_lane_guarded_ready", False))
            or bool(advisory_measurements.get("support_low_priority_guarded", False))
            or bool(advisory_measurements.get("support_throttle_pending_guarded", False))
            or bool(advisory_measurements.get("research_low_priority_guarded", False))
            or bool(advisory_measurements.get("full_force_paper_research_mix_guarded_advisory", False))
            or bool(advisory_measurements.get("operator_observability_high_compute_guarded", False))
        )
        and bool(advisory_measurements.get("storage_ready_for_runtime_advisory", False))
    )
    storage_backpressure = storage_backpressure if isinstance(storage_backpressure, dict) else {}
    raw_live, raw_live_source = _effective_storage_raw_live(storage_backpressure)
    raw_core = _safe_int(raw_live.get("core_pending_lines"), 0)
    raw_total = _safe_int(raw_live.get("total_pending_lines"), 0)
    raw_oldest = _safe_float(raw_live.get("oldest_pending_age_seconds"), 0.0)
    overlay_adjusted = bool(storage_backpressure.get("overlay_adjusted", False))
    raw_live_clear = bool(
        raw_live
        and raw_core <= OVERLAY_RAW_LIVE_MAX_CORE_LINES
        and raw_total <= OVERLAY_RAW_LIVE_MAX_TOTAL_LINES
        and raw_oldest <= OVERLAY_RAW_LIVE_MAX_AGE_SECONDS
    )
    overlay_capacity_relief = bool(overlay_adjusted and raw_live_clear)
    compute_pressure_limited = bool(compute_pressure_level == "high" and not attribution_capacity_advisory)
    storage_pressure_limited = bool(
        (storage_pressure_index >= 1.0 or str(storage_severity or "").strip().lower() in {"high", "critical", "blocked"})
        and not overlay_capacity_relief
    )
    pressure_limited = bool(
        throttle_profile == "protect_live"
        or memory_pressure_level == "high"
        or compute_pressure_limited
        or storage_pressure_limited
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
        "attribution_capacity_advisory": attribution_capacity_advisory,
        "compute_pressure_limited": compute_pressure_limited,
        "storage_pressure_limited": storage_pressure_limited,
        "storage_overlay_capacity_relief": {
            "active": overlay_capacity_relief,
            "overlay_adjusted": overlay_adjusted,
            "raw_live_clear": raw_live_clear,
            "storage_severity": str(storage_severity or ""),
            "storage_pressure_index": round(float(storage_pressure_index), 3),
            "raw_live": {
                "core_pending_lines": raw_core,
                "total_pending_lines": raw_total,
                "oldest_pending_age_seconds": round(raw_oldest, 3),
                "max_core_pending_lines": OVERLAY_RAW_LIVE_MAX_CORE_LINES,
                "max_total_pending_lines": OVERLAY_RAW_LIVE_MAX_TOTAL_LINES,
                "max_oldest_pending_age_seconds": OVERLAY_RAW_LIVE_MAX_AGE_SECONDS,
                "source": raw_live_source,
                "reconciled_from_raw_live": bool(raw_live.get("reconciled_from_raw_live", False)),
            },
            "policy": "do_not_block paper capacity on SQL-overlay-only pressure when raw live backlog is cool",
        },
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


def _paper_ramp_execution_policy(paper_ramp: dict[str, Any]) -> dict[str, Any]:
    if not paper_ramp:
        return {
            "artifact_present": False,
            "paper_execution_allowed": True,
            "pause_paper_execution": False,
            "reason": "paper_ramp_artifact_missing",
            "stage": "missing",
            "armed": False,
            "ok": False,
            "blockers": [],
        }
    stage = str(paper_ramp.get("stage") or paper_ramp.get("overall_status") or "").strip().lower()
    blockers = [str(item) for item in paper_ramp.get("blockers", []) if str(item).strip()] if isinstance(paper_ramp.get("blockers"), list) else []
    armed = bool(paper_ramp.get("armed", False))
    ok = bool(paper_ramp.get("ok", False))
    blocked = bool(stage in {"blocked", "protect_live", "halted"} or blockers)
    allowed = bool(ok and armed and not blocked)
    reason = "paper_ramp_armed_and_clean" if allowed else "paper_ramp_not_armed_or_blocked"
    if blocked:
        reason = "paper_ramp_blocked"
    elif not armed:
        reason = "paper_ramp_not_armed"
    elif not ok:
        reason = "paper_ramp_not_ok"
    return {
        "artifact_present": True,
        "paper_execution_allowed": allowed,
        "pause_paper_execution": not allowed,
        "reason": reason,
        "stage": stage or "unknown",
        "armed": armed,
        "ok": ok,
        "blockers": blockers,
    }


def _paper_execution_pressure_pause_policy(
    paper_execution_policy: dict[str, Any],
    host_pressure_attribution: dict[str, Any],
    *,
    throttle_profile: str,
    compute_pressure_level: str,
    memory_pressure_level: str,
    saturation_score: float = 0.0,
    live_read_only: bool = False,
    storage_ready_for_runtime_advisory: bool = False,
    full_force_paper_required: bool = False,
) -> dict[str, Any]:
    policy = dict(paper_execution_policy) if isinstance(paper_execution_policy, dict) else {}
    if bool(policy.get("pause_paper_execution", False)):
        raw_blockers = policy.get("blockers") if isinstance(policy.get("blockers"), list) else []
        blockers = {
            str(item or "").strip()
            for item in raw_blockers
            if str(item or "").strip()
        }
        pressure_only_ramp_block = bool(
            full_force_paper_required
            and bool(policy.get("artifact_present", False))
            and blockers
            and blockers.issubset(PRESSURE_ONLY_PAPER_RAMP_BLOCKERS)
            and bool(live_read_only)
            and bool(storage_ready_for_runtime_advisory)
            and str(memory_pressure_level or "") == "normal"
        )
        if pressure_only_ramp_block:
            policy.update(
                {
                    "paper_execution_allowed": True,
                    "pause_paper_execution": False,
                    "pressure_pause_active": False,
                    "pressure_pause_bypassed": True,
                    "pressure_pause_bypass_reason": "full_force_paper_ramp_pressure_only_blocker",
                    "pressure_recovery_probe": True,
                    "pressure_recovery_source_stage": str(policy.get("stage") or "blocked"),
                    "reason": "paper_ramp_pressure_only_blocker_bypassed_for_full_force_soak",
                }
            )
            return policy
        policy.setdefault("pressure_pause_active", False)
        return policy

    paper_cpu = _safe_float(host_pressure_attribution.get("paper_execution_cpu_percent"), 0.0)
    storage_writer_cpu = _safe_float(host_pressure_attribution.get("storage_writer_cpu_percent"), 0.0)
    bot_owned_cpu = _safe_float(host_pressure_attribution.get("bot_owned_cpu_percent"), 0.0)
    support_cpu = _safe_float(host_pressure_attribution.get("throttle_candidate_support_cpu_percent"), 0.0)
    research_cpu = _safe_float(host_pressure_attribution.get("research_training_cpu_percent"), 0.0)
    support_bounded_for_soak = bool(
        not bool(host_pressure_attribution.get("support_jobs_hot", False))
        or (
            support_cpu <= 80.0
            and not bool(host_pressure_attribution.get("support_pressure_dominant", False))
        )
    )
    storage_writer_bounded_for_soak = bool(
        not bool(host_pressure_attribution.get("storage_writer_hot", False))
        or storage_writer_cpu <= 190.0
    )
    research_bounded_for_soak = bool(
        not bool(host_pressure_attribution.get("research_training_hot", False))
        or (
            research_cpu <= FULL_FORCE_PAPER_BOUNDED_RESEARCH_CPU_THRESHOLD
            and bool(host_pressure_attribution.get("research_hot_low_priority", False))
        )
    )
    bounded_full_force_soak = bool(
        full_force_paper_required
        and bool(live_read_only)
        and bool(storage_ready_for_runtime_advisory)
        and bool(policy.get("paper_execution_allowed", False))
        and bool(policy.get("armed", False))
        and bool(policy.get("ok", False))
        and str(throttle_profile or "") in {"soft_cap", "sustain"}
        and str(compute_pressure_level or "") == "high"
        and str(memory_pressure_level or "") == "normal"
        and bool(host_pressure_attribution.get("paper_execution_hot", False))
        and bool(host_pressure_attribution.get("paper_hot_low_priority", False))
        and storage_writer_bounded_for_soak
        and support_bounded_for_soak
        and research_bounded_for_soak
        and not bool(host_pressure_attribution.get("protected_work_hot", False))
        and paper_cpu <= 125.0
        and storage_writer_cpu <= 190.0
        and bot_owned_cpu <= 340.0
    )
    if bounded_full_force_soak:
        policy.setdefault("pressure_pause_active", False)
        policy["pressure_pause_bypassed"] = True
        policy["pressure_pause_bypass_reason"] = "full_force_paper_ramp_bounded_low_priority_soak"
        return policy

    capacity_limit_threshold = _safe_float(
        os.getenv("FULL_FORCE_PAPER_CAPACITY_LIMIT_CPU_PERCENT"),
        FULL_FORCE_PAPER_CAPACITY_LIMIT_CPU_THRESHOLD,
    )
    capacity_limited_full_force_soak = bool(
        full_force_paper_required
        and bool(live_read_only)
        and bool(storage_ready_for_runtime_advisory)
        and bool(policy.get("paper_execution_allowed", False))
        and bool(policy.get("armed", False))
        and bool(policy.get("ok", False))
        and str(throttle_profile or "") in {"soft_cap", "sustain"}
        and str(compute_pressure_level or "") in {"elevated", "high"}
        and str(memory_pressure_level or "") == "normal"
        and bool(host_pressure_attribution.get("paper_execution_hot", False))
        and bool(host_pressure_attribution.get("paper_hot_low_priority", False))
        and storage_writer_bounded_for_soak
        and support_bounded_for_soak
        and research_bounded_for_soak
        and not bool(host_pressure_attribution.get("protected_work_hot", False))
        and paper_cpu <= capacity_limit_threshold
        and storage_writer_cpu <= 190.0
        and bot_owned_cpu <= 340.0
    )
    if capacity_limited_full_force_soak:
        policy.update(
            {
                "paper_execution_allowed": True,
                "pause_paper_execution": False,
                "pressure_pause_active": False,
                "pressure_pause_bypassed": True,
                "pressure_pause_bypass_reason": "full_force_paper_ramp_capacity_limited_low_priority_soak",
                "capacity_limited_paper_execution": True,
                "capacity_limit_reason": "paper_execution_cpu_pressure_capacity_limited_for_full_force_soak",
                "capacity_limit_cpu_threshold": round(capacity_limit_threshold, 3),
                "reason": "paper_ramp_armed_capacity_limited_for_full_force_soak",
            }
        )
        return policy

    elevated_compute_full_force_soak = bool(
        full_force_paper_required
        and bool(live_read_only)
        and bool(storage_ready_for_runtime_advisory)
        and bool(policy.get("paper_execution_allowed", False))
        and bool(policy.get("armed", False))
        and bool(policy.get("ok", False))
        and str(throttle_profile or "") in {"soft_cap", "sustain"}
        and str(compute_pressure_level or "") == "elevated"
        and str(memory_pressure_level or "") == "normal"
        and bool(host_pressure_attribution.get("paper_execution_hot", False))
        and bool(host_pressure_attribution.get("paper_hot_low_priority", False))
        and not bool(host_pressure_attribution.get("protected_work_hot", False))
        and storage_writer_bounded_for_soak
        and paper_cpu <= capacity_limit_threshold
    )
    if elevated_compute_full_force_soak:
        policy.update(
            {
                "paper_execution_allowed": True,
                "pause_paper_execution": False,
                "pressure_pause_active": False,
                "pressure_pause_bypassed": True,
                "pressure_pause_bypass_reason": "full_force_paper_ramp_elevated_compute_downshift_without_restart",
                "capacity_limited_paper_execution": True,
                "capacity_limit_reason": "paper_execution_cpu_pressure_downshifted_for_full_force_soak",
                "capacity_limit_cpu_threshold": round(capacity_limit_threshold, 3),
                "reason": "paper_ramp_armed_downshifted_for_elevated_compute_full_force_soak",
            }
        )
        return policy

    hot_paper_processes = host_pressure_attribution.get("hot_paper_processes")
    hot_paper_processes = hot_paper_processes if isinstance(hot_paper_processes, list) else []
    supervised_live_soak_only = bool(
        hot_paper_processes
        and all(_is_live_soak_shadow_loop(row) for row in hot_paper_processes)
    )
    supervised_full_force_downshift = bool(
        full_force_paper_required
        and bool(live_read_only)
        and bool(storage_ready_for_runtime_advisory)
        and bool(policy.get("paper_execution_allowed", False))
        and bool(policy.get("armed", False))
        and bool(policy.get("ok", False))
        and str(throttle_profile or "") in {"soft_cap", "sustain"}
        and str(compute_pressure_level or "") in {"elevated", "high"}
        and str(memory_pressure_level or "") == "normal"
        and float(saturation_score) < FULL_FORCE_PAPER_ELEVATED_MAX_HOST_SATURATION
        and bool(host_pressure_attribution.get("paper_execution_hot", False))
        and supervised_live_soak_only
        and not bool(host_pressure_attribution.get("protected_work_hot", False))
    )
    if supervised_full_force_downshift:
        policy.update(
            {
                "paper_execution_allowed": True,
                "pause_paper_execution": False,
                "pressure_pause_active": False,
                "pressure_pause_bypassed": True,
                "pressure_pause_bypass_reason": "supervised_full_force_paper_soak_downshift_without_restart",
                "capacity_limited_paper_execution": True,
                "capacity_limit_reason": "supervised_paper_workers_reniced_under_bounded_host_pressure",
                "capacity_limit_host_saturation_threshold": FULL_FORCE_PAPER_ELEVATED_MAX_HOST_SATURATION,
                "reason": "supervised_full_force_paper_soak_downshifted_without_restart",
            }
        )
        return policy

    pause_threshold = _safe_float(
        os.getenv("PAPER_EXECUTION_PRESSURE_PAUSE_CPU_PERCENT"),
        PAPER_EXECUTION_PRESSURE_PAUSE_CPU_THRESHOLD,
    )
    paper_dominant = bool(
        host_pressure_attribution.get("paper_execution_pressure_dominant", False)
        or (
            bool(host_pressure_attribution.get("bot_owned_pressure_dominant", False))
            and paper_cpu >= pause_threshold
        )
    )
    pressure_context = bool(
        str(throttle_profile or "") == "protect_live"
        or str(memory_pressure_level or "") == "high"
        or str(compute_pressure_level or "") == "high"
        or (
            str(compute_pressure_level or "") == "elevated"
            and paper_dominant
            and paper_cpu >= pause_threshold
        )
    )
    if not (pressure_context and paper_dominant and paper_cpu >= pause_threshold):
        policy.setdefault("pressure_pause_active", False)
        return policy

    policy.update(
        {
            "paper_execution_allowed": False,
            "pause_paper_execution": True,
            "pressure_pause_active": True,
            "pressure_pause_reason": "paper_execution_cpu_pressure",
            "reason": "paper_execution_cpu_pressure",
        }
    )
    return policy


def _paper_trade_lock_active(project_root: Path) -> bool:
    lock_path = project_root / "governance" / "health" / "PAPER_TRADE_LOCK.flag"
    try:
        raw = lock_path.read_text(encoding="utf-8")
    except Exception:
        return False
    if "live_data_paper_trade_only" in raw:
        return True
    return "enabled_at_utc=" in raw


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


def _drain_friendly_sql_overrides(
    *,
    concentrated_core: bool = False,
    writer_worker_budget: int | None = None,
    max_writer_lanes: int | None = None,
) -> dict[str, str]:
    configured_lane_cap = _safe_int(os.getenv("SQL_LINK_SERVICE_MAX_SHARD_WRITER_LANES"), 0)
    explicit_worker_budget = writer_worker_budget is not None
    explicit_max_writer_lanes = max_writer_lanes is not None
    selected_worker_budget = max(
        _safe_int(writer_worker_budget, 0)
        if explicit_worker_budget
        else max(
            _safe_int(os.getenv("AUTONOMIC_PCORE_PREPROCESS_WORKERS"), 0),
            _safe_int(os.getenv("BACKLOG_PCORE_PREPROCESS_WORKERS"), 0),
            _safe_int(os.getenv("SQL_LINK_SERVICE_PREPROCESS_WORKERS"), 1),
        ),
        1,
    )
    if configured_lane_cap > 0 and not explicit_worker_budget:
        selected_worker_budget = max(1, min(selected_worker_budget, configured_lane_cap))
    selected_lane_cap = max(
        selected_worker_budget,
        _safe_int(max_writer_lanes, 0)
        if explicit_max_writer_lanes
        else min(_safe_int(os.getenv("BOT_PERFORMANCE_CORE_TARGET"), 8), 8),
    )
    if configured_lane_cap > 0 and not explicit_max_writer_lanes:
        selected_lane_cap = max(1, min(selected_lane_cap, configured_lane_cap))
    worker_budget = str(selected_worker_budget)
    selected_lane_cap_text = str(max(1, min(selected_lane_cap, 8)))
    warm_lane_cap_text = str(max(1, min(2, _safe_int(selected_lane_cap_text, 1))))
    overrides = {
        "SQL_LINK_SERVICE_INTERVAL_SECONDS": "12",
        "SQL_LINK_SERVICE_HOT_MIN_INTERVAL_SECONDS": "30",
        "SQL_LINK_SERVICE_QUEUE_MIN_INTERVAL_SECONDS": "180",
        "SQL_LINK_SERVICE_HOT_BATCH_SIZE": "240000",
        "SQL_LINK_SERVICE_QUEUE_BATCH_SIZE": "180000",
        "SQL_LINK_SERVICE_MERGE_MAX_SECONDS_PER_CYCLE": "25",
        "SQL_LINK_SERVICE_PROGRESS_HEARTBEAT_SECONDS": "20",
        "SQL_LINK_SERVICE_SMART_SHARD_PARALLELISM": "1",
        "SQL_LINK_SERVICE_SENTINEL_SHARD_LANE_CAP": "1",
        "SQL_LINK_SERVICE_HOT_SHARD_LANE_CAP": selected_lane_cap_text,
        "SQL_LINK_SERVICE_WARM_SHARD_LANE_CAP": warm_lane_cap_text,
        "SQL_LINK_SERVICE_COLD_SHARD_LANE_CAP": "1",
        "SQL_LINK_SERVICE_SKIP_FRESH_IDLE_SHARDS": "1",
        "SQL_LINK_SERVICE_IDLE_SHARD_MAX_AGE_SECONDS": "120",
        "SQL_LINK_SERVICE_SKIP_IDLE_SENTINELS": "0",
        "SQL_LINK_SERVICE_SHARD_GOVERNANCE_MAX_FILES": "10",
        "SQL_LINK_SERVICE_SHARD_GOVERNANCE_MAX_BYTES_PER_FILE": str(128 * 1024 * 1024),
        "SQL_LINK_SERVICE_SHARD_GOVERNANCE_SQLITE_BATCH_MAX_BYTES": str(32 * 1024 * 1024),
        "SQL_LINK_SERVICE_SHARD_GOVERNANCE_TIMEOUT_SECONDS": "240",
        "INGEST_HOST_LOAD_SOFT_CAP": "8.0",
        "INGEST_HOST_LOAD_SLEEP_SECONDS": "0.25",
        "INGEST_FLUSH_SLEEP_SECONDS": "0.02",
        "INGEST_FILE_SLEEP_SECONDS": "0.10",
        "BACKLOG_PCORE_PREPROCESS_WORKERS": worker_budget,
        "SQL_LINK_SERVICE_PREPROCESS_WORKERS": worker_budget,
        "SQL_LINK_SERVICE_SHARD_WRITER_LANES": worker_budget,
        "SQL_LINK_SERVICE_MAX_SHARD_WRITER_LANES": selected_lane_cap_text,
        "SQL_LINK_CHILD_WRITER_CPU_POLICY": "performance_core_primary",
        "SQL_LINK_WRITER_BACKGROUND_POLICY": str(max(_safe_int(os.getenv("SQL_LINK_WRITER_BACKGROUND_POLICY"), 0), 0)),
        "SQL_LINK_WRITER_NICE": str(max(_safe_int(os.getenv("SQL_LINK_WRITER_NICE"), 0), 0)),
        "BOT_CPU_ALLOCATION_POLICY": "performance_core_primary",
        "BOT_CPU_QOS_POLICY": "performance_core_primary_no_background_writer",
    }
    if concentrated_core:
        overrides.update(
            {
                "SQL_LINK_SERVICE_CONCENTRATED_CORE_DRAIN": "1",
                "SQL_LINK_SERVICE_SHARD_LINK_TIMEOUT_SECONDS": "420",
                "SQL_LINK_SERVICE_MERGE_MAX_SECONDS_PER_CYCLE": "90",
                "SQL_LINK_SERVICE_SHARD_TRADING_STATE_CHECKPOINT_LINES": "1000",
                "SQL_LINK_SERVICE_SHARD_AGGRESSIVE_TRADING_STATE_CHECKPOINT_LINES": "1000",
                "SQL_LINK_SERVICE_SHARD_AGGRESSIVE_TRADING_MAX_LINES_PER_FILE": "12000",
            }
        )
    return overrides


def _storage_drain_requires_acceleration(
    storage_pressure: dict[str, Any],
    sql_writer_coordination: dict[str, Any],
) -> bool:
    total_pending = max(
        _safe_int(storage_pressure.get("total_pending_lines"), 0),
        _safe_int(sql_writer_coordination.get("total_pending_lines"), 0),
    )
    oldest_age = _safe_float(storage_pressure.get("oldest_pending_age_seconds"), 0.0)
    pressure_index = _safe_float(storage_pressure.get("pressure_index"), 0.0)
    return bool(
        bool(sql_writer_coordination.get("concentrated_core_drain", False))
        or pressure_index >= 0.5
        or total_pending >= 5000
        or oldest_age >= 240.0
    )


def _idle_sql_writer_cooling_overrides(throttle_profile: str) -> dict[str, str]:
    profile = str(throttle_profile or "observe").strip().lower()
    if profile == "protect_live":
        interval = "180"
        hot_min = "720"
        queue_min = "2400"
        writer_nice = "20"
    elif profile == "sustain":
        interval = "120"
        hot_min = "480"
        queue_min = "1800"
        writer_nice = "18"
    elif profile == "soft_cap":
        interval = "90"
        hot_min = "240"
        queue_min = "900"
        writer_nice = "12"
    else:
        interval = "60"
        hot_min = "180"
        queue_min = "600"
        writer_nice = "8"
    return {
        "SQL_LINK_SERVICE_HOST_COOLING_ACTIVE": "1",
        "SQL_LINK_SERVICE_IDLE_BACKLOG_COOLDOWN": "1",
        "SQL_LINK_SERVICE_INTERVAL_SECONDS": interval,
        "SQL_LINK_SERVICE_HOT_MIN_INTERVAL_SECONDS": hot_min,
        "SQL_LINK_SERVICE_QUEUE_MIN_INTERVAL_SECONDS": queue_min,
        "SQL_LINK_SERVICE_HOT_BATCH_SIZE": "30000",
        "SQL_LINK_SERVICE_QUEUE_BATCH_SIZE": "20000",
        "SQL_LINK_SERVICE_MERGE_MAX_SECONDS_PER_CYCLE": "20",
        "SQL_LINK_SERVICE_PROGRESS_HEARTBEAT_SECONDS": "20",
        "SQL_LINK_SERVICE_SMART_SHARD_PARALLELISM": "1",
        "SQL_LINK_SERVICE_SENTINEL_SHARD_LANE_CAP": "1",
        "SQL_LINK_SERVICE_HOT_SHARD_LANE_CAP": "1",
        "SQL_LINK_SERVICE_WARM_SHARD_LANE_CAP": "1",
        "SQL_LINK_SERVICE_COLD_SHARD_LANE_CAP": "1",
        "SQL_LINK_SERVICE_SKIP_FRESH_IDLE_SHARDS": "1",
        "SQL_LINK_SERVICE_IDLE_SHARD_MAX_AGE_SECONDS": "900",
        "SQL_LINK_SERVICE_SKIP_IDLE_SENTINELS": "1",
        "SQL_LINK_SERVICE_SHARD_GOVERNANCE_MAX_FILES": "4",
        "SQL_LINK_SERVICE_SHARD_GOVERNANCE_MAX_BYTES_PER_FILE": str(96 * 1024 * 1024),
        "SQL_LINK_SERVICE_SHARD_GOVERNANCE_SQLITE_BATCH_MAX_BYTES": str(8 * 1024 * 1024),
        "SQL_LINK_SERVICE_SHARD_GOVERNANCE_TIMEOUT_SECONDS": "120",
        "INGEST_HOST_LOAD_SOFT_CAP": "6.0",
        "INGEST_HOST_LOAD_SLEEP_SECONDS": "0.75" if profile == "protect_live" else "0.50",
        "INGEST_FLUSH_SLEEP_SECONDS": "0.10" if profile == "protect_live" else "0.05",
        "INGEST_FILE_SLEEP_SECONDS": "0.50" if profile == "protect_live" else "0.25",
        "BACKLOG_PCORE_PREPROCESS_WORKERS": "1",
        "SQL_LINK_SERVICE_PREPROCESS_WORKERS": "1",
        "SQL_LINK_SERVICE_SHARD_WRITER_LANES": "1",
        "SQL_LINK_SERVICE_MAX_SHARD_WRITER_LANES": "1",
        "SQL_LINK_CHILD_WRITER_CPU_POLICY": "foreground_safe_idle_backlog",
        "SQL_LINK_WRITER_BACKGROUND_POLICY": "1",
        "SQL_LINK_WRITER_NICE": writer_nice,
        "BOT_CPU_ALLOCATION_POLICY": "foreground_first_idle_storage_writer",
        "BOT_CPU_QOS_POLICY": "foreground_first_idle_storage_writer",
    }


def _sql_overrides_for_runtime_pressure(
    throttle_profile: str,
    *,
    storage_drain_active: bool,
    storage_pressure: dict[str, Any],
    sql_writer_coordination: dict[str, Any],
    writer_worker_budget: int | None = None,
    max_writer_lanes: int | None = None,
) -> dict[str, str]:
    if not storage_drain_active:
        if str(throttle_profile or "").strip().lower() in {"protect_live", "sustain", "soft_cap"}:
            return _idle_sql_writer_cooling_overrides(throttle_profile)
        return {}
    concentrated_core = bool(sql_writer_coordination.get("concentrated_core_drain", False))
    if not _storage_drain_requires_acceleration(
        storage_pressure,
        sql_writer_coordination,
    ):
        coordination_pending = max(
            _safe_int(sql_writer_coordination.get("total_pending_lines"), 0),
            _safe_int(sql_writer_coordination.get("core_pending_lines"), 0),
        )
        clean_backlog = bool(
            _safe_int(storage_pressure.get("total_pending_lines"), 0) <= 0
            and _safe_int(storage_pressure.get("core_pending_lines"), 0) <= 0
            and coordination_pending <= 25
            and _safe_float(storage_pressure.get("pressure_index"), 0.0) <= 0.05
            and _safe_float(storage_pressure.get("oldest_pending_age_seconds"), 0.0) <= 30.0
        )
        if str(throttle_profile or "").strip().lower() == "protect_live":
            return {
                "SQL_LINK_SERVICE_HOST_COOLING_ACTIVE": "1",
                "BACKLOG_PCORE_PREPROCESS_WORKERS": "1",
                "SQL_LINK_SERVICE_PREPROCESS_WORKERS": "1",
                "SQL_LINK_SERVICE_SHARD_WRITER_LANES": "1",
                "SQL_LINK_SERVICE_MAX_SHARD_WRITER_LANES": "1",
                "SQL_LINK_SERVICE_PROGRESS_HEARTBEAT_SECONDS": "20",
                "SQL_LINK_SERVICE_SMART_SHARD_PARALLELISM": "1",
                "SQL_LINK_SERVICE_SENTINEL_SHARD_LANE_CAP": "1",
                "SQL_LINK_SERVICE_HOT_SHARD_LANE_CAP": "1",
                "SQL_LINK_SERVICE_WARM_SHARD_LANE_CAP": "1",
                "SQL_LINK_SERVICE_COLD_SHARD_LANE_CAP": "1",
                "SQL_LINK_SERVICE_SHARD_GOVERNANCE_MAX_FILES": "8",
                "SQL_LINK_SERVICE_SHARD_GOVERNANCE_MAX_BYTES_PER_FILE": str(128 * 1024 * 1024),
                "SQL_LINK_SERVICE_SHARD_GOVERNANCE_SQLITE_BATCH_MAX_BYTES": str(12 * 1024 * 1024),
                "SQL_LINK_SERVICE_SHARD_GOVERNANCE_TIMEOUT_SECONDS": "180",
                "INGEST_HOST_LOAD_SOFT_CAP": "6.0",
                "INGEST_HOST_LOAD_SLEEP_SECONDS": "0.50",
                "INGEST_FLUSH_SLEEP_SECONDS": "0.05",
                "INGEST_FILE_SLEEP_SECONDS": "0.25",
            }
        if clean_backlog and str(throttle_profile or "").strip().lower() in {"protect_live", "sustain", "soft_cap"}:
            return _idle_sql_writer_cooling_overrides(throttle_profile)
        overrides = _drain_friendly_sql_overrides(
            concentrated_core=concentrated_core,
            writer_worker_budget=1,
            max_writer_lanes=1,
        )
        overrides["SQL_LINK_SERVICE_HOST_COOLING_ACTIVE"] = "1"
        return overrides
    return _drain_friendly_sql_overrides(
        concentrated_core=concentrated_core,
        writer_worker_budget=writer_worker_budget,
        max_writer_lanes=max_writer_lanes,
    )


def _context_collector_pressure_overrides(mode: str) -> dict[str, str]:
    selected = str(mode or "normal").strip().lower()
    if selected == "protect_live":
        return {
            "CONTEXT_COLLECTOR_PRESSURE_GOVERNOR_ENABLED": "1",
            "MARKET_CONTEXT_COLLECTOR_MAX_CONCURRENT": "1",
            "MARKET_CONTEXT_COLLECTOR_MIN_INTERVAL_SECONDS": "900",
            "FX_MARKET_CONTEXT_PRESSURE_MODE": "protect",
            "FX_MARKET_CONTEXT_MIN_INTERVAL_SECONDS": "900",
            "FX_MARKET_CONTEXT_TIMEOUT_CAP_SECONDS": "8",
            "FX_TWELVE_DATA_MAX_PAIRS_PER_RUN": "1",
            "FX_TWELVE_DATA_OUTPUTSIZE": "12",
            "FX_MARKET_CONTEXT_ALPHA_VANTAGE_ENABLED": "0",
        }
    if selected == "sustain":
        return {
            "CONTEXT_COLLECTOR_PRESSURE_GOVERNOR_ENABLED": "1",
            "MARKET_CONTEXT_COLLECTOR_MAX_CONCURRENT": "1",
            "MARKET_CONTEXT_COLLECTOR_MIN_INTERVAL_SECONDS": "600",
            "FX_MARKET_CONTEXT_PRESSURE_MODE": "guarded",
            "FX_MARKET_CONTEXT_MIN_INTERVAL_SECONDS": "600",
            "FX_MARKET_CONTEXT_TIMEOUT_CAP_SECONDS": "10",
            "FX_TWELVE_DATA_MAX_PAIRS_PER_RUN": "1",
            "FX_TWELVE_DATA_OUTPUTSIZE": "24",
            "FX_MARKET_CONTEXT_ALPHA_VANTAGE_ENABLED": "0",
        }
    if selected == "soft_cap":
        return {
            "CONTEXT_COLLECTOR_PRESSURE_GOVERNOR_ENABLED": "1",
            "MARKET_CONTEXT_COLLECTOR_MAX_CONCURRENT": "1",
            "MARKET_CONTEXT_COLLECTOR_MIN_INTERVAL_SECONDS": "300",
            "FX_MARKET_CONTEXT_PRESSURE_MODE": "calm",
            "FX_MARKET_CONTEXT_MIN_INTERVAL_SECONDS": "300",
            "FX_MARKET_CONTEXT_TIMEOUT_CAP_SECONDS": "12",
            "FX_TWELVE_DATA_MAX_PAIRS_PER_RUN": "2",
            "FX_TWELVE_DATA_OUTPUTSIZE": "36",
            "FX_MARKET_CONTEXT_ALPHA_VANTAGE_ENABLED": "0",
        }
    return {
        "CONTEXT_COLLECTOR_PRESSURE_GOVERNOR_ENABLED": "0",
        "MARKET_CONTEXT_COLLECTOR_MAX_CONCURRENT": "2",
        "FX_MARKET_CONTEXT_PRESSURE_MODE": "off",
        "FX_MARKET_CONTEXT_MIN_INTERVAL_SECONDS": "0",
    }


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
    storage_pressure: dict[str, Any] | None = None,
    paper_capacity_contract: dict[str, Any] | None = None,
    cotenant_contract: dict[str, Any] | None = None,
    mlx_contract: dict[str, Any] | None = None,
    library_contract: dict[str, Any] | None = None,
    sql_writer_coordination: dict[str, Any] | None = None,
    paper_execution_policy: dict[str, Any] | None = None,
    writer_worker_budget: int | None = None,
    max_writer_lanes: int | None = None,
) -> dict[str, str]:
    paper_capacity_contract = paper_capacity_contract if isinstance(paper_capacity_contract, dict) else {}
    cotenant_contract = cotenant_contract if isinstance(cotenant_contract, dict) else {}
    mlx_contract = mlx_contract if isinstance(mlx_contract, dict) else {}
    library_contract = library_contract if isinstance(library_contract, dict) else {}
    sql_writer_coordination = sql_writer_coordination if isinstance(sql_writer_coordination, dict) else {}
    paper_execution_policy = paper_execution_policy if isinstance(paper_execution_policy, dict) else {}
    storage_pressure = storage_pressure if isinstance(storage_pressure, dict) else {}
    full_force_paper = bool(paper_capacity_contract.get("full_force_stabilization_required", False))
    pause_paper_execution = bool(paper_execution_policy.get("pause_paper_execution", False))
    capacity_limited_paper_execution = bool(paper_execution_policy.get("capacity_limited_paper_execution", False))
    runtime_sql_overrides = _sql_overrides_for_runtime_pressure(
        throttle_profile,
        storage_drain_active=storage_drain_active,
        storage_pressure=storage_pressure,
        sql_writer_coordination=sql_writer_coordination,
        writer_worker_budget=writer_worker_budget,
        max_writer_lanes=max_writer_lanes,
    )

    def _with_full_force_paper(overrides: dict[str, str]) -> dict[str, str]:
        if pause_paper_execution:
            overrides.update(
                {
                    "PAPER_CRYPTO_FEED_RUNTIME_PAUSED_FOR_PRESSURE": "1"
                    if bool(paper_execution_policy.get("pressure_pause_active", False))
                    else "0",
                    "PAPER_EXECUTION_RUNTIME_PAUSED_FOR_PRESSURE": "1",
                    "PAPER_EXECUTION_QUEUE_CONSUMER_ENABLED": "0",
                    "PAPER_RECONCILIATION_HEARTBEAT_WHEN_PAUSED": "1",
                    "PAPER_400_RAMP_BLOCKED_RUNTIME_PAUSE": "1",
                    "INLINE_PAPER_EXECUTION_ENABLED": "0",
                    "PAPER_RUNTIME_CONTROL_REFRESH_SECONDS": "300",
                    "EXECUTION_LANE_BATCH_LIMIT": "25",
                    "EXECUTION_LANE_POLL_SECONDS": "10",
                }
            )
        elif bool(paper_execution_policy.get("artifact_present", False)):
            overrides.update(
                {
                    "PAPER_CRYPTO_FEED_RUNTIME_PAUSED_FOR_PRESSURE": "0",
                    "PAPER_EXECUTION_RUNTIME_PAUSED_FOR_PRESSURE": "0",
                    "PAPER_EXECUTION_QUEUE_CONSUMER_ENABLED": "1",
                    "PAPER_RECONCILIATION_HEARTBEAT_WHEN_PAUSED": "1",
                    "PAPER_400_RAMP_BLOCKED_RUNTIME_PAUSE": "0",
                    "INLINE_PAPER_EXECUTION_ENABLED": "1",
                }
            )
        if not full_force_paper:
            return overrides
        paper_cooling_required = bool(
            capacity_limited_paper_execution
            or compute_pressure_level in {"elevated", "high"}
            or str(throttle_profile or "").strip().lower() in {"soft_cap", "sustain", "protect_live"}
        )
        overrides.update(
            {
                "PAPER_FULL_FORCE_STABILITY_MODE": str(paper_capacity_contract.get("mode") or "full_force_buffered"),
                "PAPER_EXECUTION_RUNTIME_NICE": "20" if paper_cooling_required else "12",
                "EXECUTION_LANE_BATCH_LIMIT": "25" if paper_cooling_required else "100",
                "EXECUTION_LANE_BATCH_SLEEP_SECONDS": "2.0" if paper_cooling_required else "0.0",
                "EXECUTION_LANE_BACKLOG_SLEEP_SECONDS": "5.0" if paper_cooling_required else "0.0",
                "EXECUTION_LANE_HOST_LOAD_SOFT_CAP": "6.0" if paper_cooling_required else "0.0",
                "EXECUTION_LANE_HOST_LOAD_SLEEP_SECONDS": "5.0" if paper_cooling_required else "0.0",
                "EXECUTION_LANE_MESSAGE_SLEEP_SECONDS": "0.04" if paper_cooling_required else "0.0",
                "EXECUTION_LANE_PAPER_MAX_INTENT_AGE_SECONDS": "900",
                "EXECUTION_LANE_LIVE_MAX_INTENT_AGE_SECONDS": "60",
                "EXECUTION_LANE_POLL_SECONDS": "5.0" if paper_cooling_required else "2.0",
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
                "MLX_INTELLIGENCE_PCORE_AWARE": "1" if bool(mlx_contract.get("p_core_allocation_aware", False)) else "0",
                "MLX_INTELLIGENCE_PCORE_MODE": str(mlx_contract.get("p_core_allocation_mode") or ""),
                "MLX_INTELLIGENCE_PCORE_PREPROCESS_WORKERS": str(_safe_int(mlx_contract.get("p_core_preprocess_workers"), 0)),
                "MLX_INTELLIGENCE_PCORE_MEMORY_OPTIMIZER": "1" if bool(mlx_contract.get("p_core_memory_optimizer_active", False)) else "0",
                "MLX_INTELLIGENCE_PCORE_COORDINATION_POLICY": str(mlx_contract.get("p_core_coordination_policy") or "not_active"),
                "MLX_INTELLIGENCE_BACKLOG_HEADROOM_POLICY": "yield_to_backlog_p_core_workers_when_active",
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
            "INGESTION_BACKPRESSURE_REFRESH_INTERVAL_SECONDS": "120",
            "ASYNC_PIPELINE_WORKERS": "2",
            "COINBASE_SNAPSHOT_MAX_WORKERS": "1",
            "RUNTIME_FEATURE_CACHE_MAX_ENTRIES": "48",
            "RUNTIME_TRAIN_BATCH_SIZE_CAP": "32",
            "RUNTIME_TRAIN_MAX_SAMPLES": "6000",
            "RUNTIME_RESEARCH_TRAINING_THROTTLE_ENABLED": "1",
            "RUNTIME_RESEARCH_TRAINING_CPU_THRESHOLD": "25",
            "RUNTIME_SIMULATED_TRAINING_CPU_THRESHOLD": "10",
            "RUNTIME_THROTTLE_RESEARCH_NICE": "20",
            "RUNTIME_RESEARCH_TRAINING_NICE": "20",
            "RUNTIME_SATURATION_GOVERNOR_V2": "1",
            "RUNTIME_SATURATION_BAND": "protect",
            "TRAINING_RUNTIME_GOVERNOR_MODE": "paused_for_host_headroom",
            "TRAINING_RUNTIME_MAX_PARALLEL": "0",
            "TRAINING_RUNTIME_BATCH10_ALLOWED": "0",
            "TRAINING_RUNTIME_BATCH20_ALLOWED": "0",
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
            "OPS_HEAVY_SUPPORT_OFF_HOURS_ONLY": "1",
            "OPS_SUPPORT_MAINTENANCE_FREEZE": "1",
            "OPS_SUPPORT_MAINTENANCE_STABILIZER_ACTIVE": "1",
            "OPS_SUPPORT_JOB_NICE": "20",
            "YTDLP_SUPPORT_NICE": "20",
            "MACRO_YTDLP_SUPPORT_NICE": "20",
            "OPS_SUPPORT_JOBS_BACKGROUND_POLICY": "1",
            "OPS_SUPPORT_HEAVY_COLLECTOR_COOLDOWN_SECONDS": "1800",
            "OPS_SUPPORT_HEAVY_COLLECTOR_MAX_CPU_PERCENT": "25",
            "SUPPORT_MAINTENANCE_CONCURRENCY": "1",
            "TRAINING_RUNTIME_PAUSED_FOR_HOST_HEADROOM": "1",
            "SHADOW_LOOP_RUNTIME_PAUSE_SLEEP_SECONDS": "75",
        }
        overrides.update(_context_collector_pressure_overrides("protect_live"))
        overrides.update(runtime_sql_overrides)
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
            "INGESTION_BACKPRESSURE_REFRESH_INTERVAL_SECONDS": "120",
            "ASYNC_PIPELINE_WORKERS": "2",
            "COINBASE_SNAPSHOT_MAX_WORKERS": "2",
            "RUNTIME_FEATURE_CACHE_MAX_ENTRIES": "64",
            "RUNTIME_TRAIN_BATCH_SIZE_CAP": "48",
            "RUNTIME_TRAIN_MAX_SAMPLES": "8000",
            "RUNTIME_RESEARCH_TRAINING_THROTTLE_ENABLED": "1",
            "RUNTIME_RESEARCH_TRAINING_CPU_THRESHOLD": "35",
            "RUNTIME_SIMULATED_TRAINING_CPU_THRESHOLD": "15",
            "RUNTIME_THROTTLE_RESEARCH_NICE": "18",
            "RUNTIME_RESEARCH_TRAINING_NICE": "18",
            "RUNTIME_SATURATION_GOVERNOR_V2": "1",
            "RUNTIME_SATURATION_BAND": "guarded",
            "TRAINING_RUNTIME_GOVERNOR_MODE": "paused_for_host_headroom",
            "TRAINING_RUNTIME_MAX_PARALLEL": "0",
            "TRAINING_RUNTIME_BATCH10_ALLOWED": "0",
            "TRAINING_RUNTIME_BATCH20_ALLOWED": "0",
            "SHADOW_LOOP_PRESSURE_INTERVAL_FLOOR_ENABLED": "1",
            "SHADOW_LOOP_PROTECT_LIVE_EXTRA_INTERVAL_SECONDS": "15",
            "SHADOW_LOOP_QUEUE_BACKPRESSURE_EXTRA_INTERVAL_SECONDS": "10",
            "SHADOW_LOOP_HIGH_COMPUTE_EXTRA_INTERVAL_SECONDS": "10",
            "SHADOW_LOOP_SUSTAIN_EXTRA_INTERVAL_SECONDS": "20",
            "SHADOW_LOOP_SOFT_CAP_EXTRA_INTERVAL_SECONDS": "8",
            "SHADOW_LOOP_MAX_DYNAMIC_EXTRA_INTERVAL_SECONDS": "45",
            "ADAPTIVE_INTERVAL_MAX_SECONDS": "60",
            "DATA_COLLECTION_RESOURCE_GUARD_MODE": "sustain",
            "DATA_COLLECTION_RESOURCE_SAMPLE_RATE": "0.30",
            "DATA_COLLECTION_RESOURCE_CAPTURE_MODE": "sampled",
            "OPS_HEAVY_SUPPORT_OFF_HOURS_ONLY": "1",
            "OPS_SUPPORT_MAINTENANCE_FREEZE": "1",
            "OPS_SUPPORT_MAINTENANCE_STABILIZER_ACTIVE": "1",
            "OPS_SUPPORT_JOB_NICE": "20",
            "YTDLP_SUPPORT_NICE": "20",
            "MACRO_YTDLP_SUPPORT_NICE": "20",
            "OPS_SUPPORT_JOBS_BACKGROUND_POLICY": "1",
            "OPS_SUPPORT_HEAVY_COLLECTOR_COOLDOWN_SECONDS": "1200",
            "OPS_SUPPORT_HEAVY_COLLECTOR_MAX_CPU_PERCENT": "30",
            "SUPPORT_MAINTENANCE_CONCURRENCY": "1",
            "TRAINING_RUNTIME_PAUSED_FOR_HOST_HEADROOM": "1",
            "SHADOW_LOOP_RUNTIME_PAUSE_SLEEP_SECONDS": "60",
        }
        overrides.update(_context_collector_pressure_overrides("sustain"))
        overrides.update(runtime_sql_overrides)
        return _with_library_utilization(_with_mlx_intelligence(_with_cotenant_awareness(_with_full_force_paper(overrides))))
    support_nice = "12" if throttle_profile in {"soft_cap", "observe"} else "0"
    support_pause_sleep = "30" if throttle_profile == "soft_cap" else "15" if throttle_profile == "observe" else "0"
    overrides = {
        "BOT_RUNTIME_RESOURCE_GUARD_PROFILE": throttle_profile,
        "DATA_COLLECTION_RESOURCE_GUARD_MODE": throttle_profile,
        "DATA_COLLECTION_RESOURCE_SAMPLE_RATE": "0.50" if throttle_profile == "soft_cap" else "1.0",
        "DATA_COLLECTION_RESOURCE_CAPTURE_MODE": "sampled" if throttle_profile == "soft_cap" else "full",
        "RUNTIME_SATURATION_GOVERNOR_V2": "1",
        "RUNTIME_SATURATION_BAND": "advisory" if throttle_profile == "soft_cap" else "normal",
        "RUNTIME_THROTTLE_RESEARCH_NICE": "15" if throttle_profile == "soft_cap" else "8",
        "RUNTIME_RESEARCH_TRAINING_NICE": "15" if throttle_profile == "soft_cap" else "8",
        "TRAINING_RUNTIME_GOVERNOR_MODE": "micro_canary_only" if throttle_profile == "soft_cap" else "small_batch_allowed",
        "TRAINING_RUNTIME_MAX_PARALLEL": "1" if throttle_profile == "soft_cap" else "2",
        "TRAINING_RUNTIME_BATCH10_ALLOWED": "0" if throttle_profile == "soft_cap" else "1",
        "TRAINING_RUNTIME_BATCH20_ALLOWED": "0",
        "OPS_SUPPORT_JOB_NICE": support_nice,
        "YTDLP_SUPPORT_NICE": support_nice,
        "MACRO_YTDLP_SUPPORT_NICE": support_nice,
        "OPS_SUPPORT_JOBS_BACKGROUND_POLICY": "0",
        "OPS_SUPPORT_HEAVY_COLLECTOR_COOLDOWN_SECONDS": "300" if throttle_profile == "soft_cap" else "0",
        "OPS_SUPPORT_HEAVY_COLLECTOR_MAX_CPU_PERCENT": "60" if throttle_profile == "soft_cap" else "0",
        "TRAINING_RUNTIME_PAUSED_FOR_HOST_HEADROOM": "0" if throttle_profile == "soft_cap" else "0",
        "SHADOW_LOOP_RUNTIME_PAUSE_SLEEP_SECONDS": support_pause_sleep,
        "INGESTION_BACKPRESSURE_REFRESH_INTERVAL_SECONDS": "120",
    }
    overrides.update(_context_collector_pressure_overrides(throttle_profile))
    overrides.update(runtime_sql_overrides)
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
    command = str(row.get("command") or row.get("command_excerpt") or "")
    return "scripts/run_shadow_training_loop.py" in command and "--simulate" in command


def _is_live_soak_shadow_loop(row: dict[str, Any]) -> bool:
    command = str(row.get("command") or row.get("command_excerpt") or "").lower()
    if "scripts/run_shadow_training_loop.py" not in command:
        return False
    if "--simulate" in command:
        return False
    if "--broker schwab" in command or "--broker coinbase" in command:
        return True
    return "--max-iterations 0" in command or "--max-iterations=0" in command


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
        or str(profile or "") == "sustain"
        or str(compute_pressure_level or "") in {"high", "elevated"}
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
        if str(profile or "") == "sustain":
            threshold = min(threshold, APPLY_CPU_THRESHOLD)
        if cpu < threshold:
            continue
        pause_exempt = _is_live_soak_shadow_loop(row)
        out.append(
            {
                **row,
                "throttle_candidate": True,
                "priority_tier": "throttle_first_when_protect_live" if simulated else "research_downshift_when_protect_live",
                "throttle_reason": "simulated_training_loop_under_host_pressure" if simulated else "research_training_loop_under_host_pressure",
                "pause_exempt": pause_exempt,
                "pause_exempt_reason": "live_soak_shadow_loop_downshift_only" if pause_exempt else "",
            }
        )
    return out[:4]


def _paper_execution_pressure_candidates(
    top_processes: list[dict[str, Any]],
    *,
    paper_execution_policy: dict[str, Any],
    profile: str,
    compute_pressure_level: str,
    memory_pressure_level: str,
) -> list[dict[str, Any]]:
    pause_requested = bool(paper_execution_policy.get("pause_paper_execution", False))
    pressure_active = bool(
        pause_requested
        or str(profile or "") == "protect_live"
        or str(compute_pressure_level or "") == "high"
        or str(memory_pressure_level or "") == "high"
    )
    if not pressure_active:
        return []
    terminate_for_pressure = bool(
        pause_requested
        and paper_execution_policy.get("pressure_pause_active", False)
    )
    out: list[dict[str, Any]] = []
    for row in top_processes:
        if str(row.get("category") or "") != "paper_execution":
            continue
        if _safe_float(row.get("cpu_percent"), 0.0) < APPLY_CPU_THRESHOLD:
            continue
        live_soak_continuity_exempt = bool(
            _is_live_soak_shadow_loop(row)
            and bool(paper_execution_policy.get("armed", False))
            and bool(paper_execution_policy.get("ok", False))
            and str(paper_execution_policy.get("pressure_pause_reason") or "") == "paper_execution_cpu_pressure"
        )
        out.append(
            {
                **row,
                "throttle_candidate": True,
                "priority_tier": "paper_execution_pause_when_gate_blocked" if pause_requested else "paper_execution_downshift_under_host_pressure",
                "throttle_reason": str(paper_execution_policy.get("reason") or "paper_execution_under_host_pressure"),
                "terminate_when_apply": bool(terminate_for_pressure and not live_soak_continuity_exempt),
                "continuity_exempt": live_soak_continuity_exempt,
                "continuity_exempt_reason": "supervised_live_soak_worker_downshift_only" if live_soak_continuity_exempt else "",
            }
        )
    return out[:2]


def _apply_paper_execution_pause(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    attempted: list[dict[str, Any]] = []
    eligible = [
        row
        for row in candidates
        if bool(row.get("terminate_when_apply", False)) and _safe_int(row.get("pid"), 0) > 0
    ]
    for row in eligible:
        pid = _safe_int(row.get("pid"), 0)
        try:
            os.kill(pid, 0)
            os.kill(pid, signal.SIGTERM)
            attempted.append(
                {
                    "pid": pid,
                    "ok": True,
                    "signal": "SIGTERM",
                    "reason": str(row.get("throttle_reason") or "paper_execution_pause"),
                    "command_excerpt": str(row.get("command") or "")[:220],
                }
            )
        except Exception as exc:
            attempted.append(
                {
                    "pid": pid,
                    "ok": False,
                    "signal": "SIGTERM",
                    "reason": f"paper_execution_pause_failed:{exc}",
                    "command_excerpt": str(row.get("command") or "")[:220],
                }
            )
    return {
        "attempted_count": len(attempted),
        "successful_count": sum(1 for row in attempted if bool(row.get("ok", False))),
        "processes": attempted,
    }


def _research_training_pause_requested(payload: dict[str, Any]) -> tuple[bool, str]:
    governor = (
        payload.get("runtime_saturation_governor_v2")
        if isinstance(payload.get("runtime_saturation_governor_v2"), dict)
        else {}
    )
    training_policy = (
        governor.get("training_policy")
        if isinstance(governor.get("training_policy"), dict)
        else {}
    )
    if bool(training_policy.get("training_paused", False)):
        return True, str(training_policy.get("reason") or "runtime_training_paused")
    mac_fluidity = (
        payload.get("mac_fluidity_contract")
        if isinstance(payload.get("mac_fluidity_contract"), dict)
        else {}
    )
    if bool(mac_fluidity.get("research_pause_recommended", False)):
        return True, "mac_fluidity_research_pause"
    profile = str(payload.get("throttle_profile") or "").strip().lower()
    compute = str(payload.get("compute_pressure_level") or "").strip().lower()
    memory = str(payload.get("memory_pressure_level") or "").strip().lower()
    if profile in {"protect_live", "sustain"} or compute == "high" or memory == "high":
        return True, "runtime_host_headroom"
    return False, "runtime_training_ready"


def _apply_research_training_pause(
    project_root: Path,
    candidates: list[dict[str, Any]],
    payload: dict[str, Any],
    *,
    state_path: Path | None = None,
) -> dict[str, Any]:
    state_path = state_path or (project_root / "governance" / "health" / DEFAULT_RESEARCH_PAUSE_STATE_PATH.name)
    pause_requested, reason = _research_training_pause_requested(payload)
    state = load_json(state_path)
    paused_rows = state.get("paused_processes") if isinstance(state.get("paused_processes"), list) else []
    attempted: list[dict[str, Any]] = []
    resumed: list[dict[str, Any]] = []

    if pause_requested:
        eligible = [
            row
            for row in candidates
            if str(row.get("category") or "") == "research_training"
            and _safe_int(row.get("pid"), 0) > 0
            and _safe_float(row.get("cpu_percent"), 0.0) >= APPLY_CPU_THRESHOLD
            and not bool(row.get("pause_exempt", False))
            and not _is_live_soak_shadow_loop(row)
        ]
        eligible_pids = {_safe_int(row.get("pid"), 0) for row in eligible if _safe_int(row.get("pid"), 0) > 0}
        live_paused = {}
        for row in paused_rows:
            pid = _safe_int(row.get("pid"), 0)
            if pid <= 0:
                continue
            if bool(row.get("pause_exempt", False)) or _is_live_soak_shadow_loop(row) or pid not in eligible_pids:
                try:
                    os.kill(pid, 0)
                    os.kill(pid, signal.SIGCONT)
                    resumed.append(
                        {
                            "pid": pid,
                            "ok": True,
                            "signal": "SIGCONT",
                            "reason": "live_soak_shadow_loop_downshift_only",
                            "command_excerpt": str(row.get("command_excerpt") or row.get("command") or "")[:220],
                        }
                    )
                except Exception as exc:
                    resumed.append(
                        {
                            "pid": pid,
                            "ok": False,
                            "signal": "SIGCONT",
                            "reason": f"research_resume_failed:{exc}",
                            "command_excerpt": str(row.get("command_excerpt") or row.get("command") or "")[:220],
                        }
                    )
                continue
            live_paused[pid] = row
        pause_limit = max(1, _safe_int(os.getenv("RUNTIME_RESEARCH_TRAINING_PAUSE_LIMIT", "8"), 8))
        for row in eligible[:pause_limit]:
            pid = _safe_int(row.get("pid"), 0)
            try:
                os.kill(pid, 0)
                os.kill(pid, signal.SIGSTOP)
                record = {
                    "pid": pid,
                    "ok": True,
                    "signal": "SIGSTOP",
                    "reason": reason,
                    "cpu_percent": round(_safe_float(row.get("cpu_percent"), 0.0), 3),
                    "command_excerpt": str(row.get("command") or "")[:220],
                    "paused_at_utc": iso_now(),
                }
                attempted.append(record)
            except Exception as exc:
                attempted.append(
                    {
                        "pid": pid,
                        "ok": False,
                        "signal": "SIGSTOP",
                        "reason": f"research_pause_failed:{exc}",
                        "command_excerpt": str(row.get("command") or "")[:220],
                    }
                )
        for row in attempted:
            if bool(row.get("ok", False)):
                live_paused[_safe_int(row.get("pid"), 0)] = row
        state_path.parent.mkdir(parents=True, exist_ok=True)
        state_path.write_text(
            json.dumps(
                {
                    "timestamp_utc": iso_now(),
                    "pause_requested": True,
                    "reason": reason,
                    "paused_processes": list(live_paused.values()),
                },
                ensure_ascii=True,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    else:
        for row in paused_rows:
            pid = _safe_int(row.get("pid"), 0)
            if pid <= 0:
                continue
            try:
                os.kill(pid, 0)
                os.kill(pid, signal.SIGCONT)
                resumed.append(
                    {
                        "pid": pid,
                        "ok": True,
                        "signal": "SIGCONT",
                        "reason": reason,
                        "command_excerpt": str(row.get("command_excerpt") or "")[:220],
                    }
                )
            except Exception as exc:
                resumed.append(
                    {
                        "pid": pid,
                        "ok": False,
                        "signal": "SIGCONT",
                        "reason": f"research_resume_failed:{exc}",
                        "command_excerpt": str(row.get("command_excerpt") or "")[:220],
                    }
                )
        if paused_rows or state_path.exists():
            state_path.parent.mkdir(parents=True, exist_ok=True)
            state_path.write_text(
                json.dumps(
                    {
                        "timestamp_utc": iso_now(),
                        "pause_requested": False,
                        "reason": reason,
                        "paused_processes": [],
                    },
                    ensure_ascii=True,
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )

    return {
        "pause_requested": pause_requested,
        "reason": reason,
        "attempted_count": len(attempted),
        "successful_count": sum(1 for row in attempted if bool(row.get("ok", False))),
        "resumed_count": len(resumed),
        "resume_successful_count": sum(1 for row in resumed if bool(row.get("ok", False))),
        "processes": attempted,
        "resumed_processes": resumed,
        "state_path": str(state_path),
    }


def _support_maintenance_pause_requested(payload: dict[str, Any]) -> tuple[bool, str]:
    mac_fluidity = (
        payload.get("mac_fluidity_contract")
        if isinstance(payload.get("mac_fluidity_contract"), dict)
        else {}
    )
    if bool(mac_fluidity.get("support_pause_recommended", False)):
        return True, "mac_fluidity_support_pause"
    return False, "support_maintenance_ready"


def _support_pause_exempt_for_storage_recovery(
    row: dict[str, Any],
    payload: dict[str, Any],
    *,
    project_root: Path | None = None,
) -> bool:
    command = str(row.get("command") or "")
    maintenance_recovery_markers = {
        "scripts/sql_queue_retention.py",
        "scripts/ops/deep_cold_storage_layer.py",
        "scripts/ops/governance_telemetry_compactor.py",
        "scripts/ops/storage_switch_orchestrator.py",
    }
    if project_root is not None and any(marker in command for marker in maintenance_recovery_markers):
        hold = maintenance_hold_snapshot(project_root)
        if bool(hold.get("active", False)):
            return True

    storage_recovery_markers = {
        "scripts/ops/storage_backpressure_autopilot.py",
        "scripts/ops/ingestion_storage_governor.py",
        "scripts/ops/sql_link_shard_manager.py",
        "scripts/ops/sql_link_writer_service.py",
        "scripts/link_jsonl_to_sql.py",
    }
    if not any(marker in command for marker in storage_recovery_markers):
        return False
    runtime_snapshot = payload.get("runtime_snapshot") if isinstance(payload.get("runtime_snapshot"), dict) else {}
    storage_pressure = runtime_snapshot.get("storage_pressure") if isinstance(runtime_snapshot.get("storage_pressure"), dict) else {}
    storage_stabilization = payload.get("storage_stabilization") if isinstance(payload.get("storage_stabilization"), dict) else {}
    return bool(
        bool(storage_stabilization.get("drain_friendly_sql_required", False))
        or _safe_float(storage_pressure.get("pressure_index"), 0.0) >= 0.20
        or _safe_int(storage_pressure.get("total_pending_lines"), 0) > 0
    )


def _apply_support_maintenance_pause(
    project_root: Path,
    candidates: list[dict[str, Any]],
    payload: dict[str, Any],
    *,
    state_path: Path | None = None,
) -> dict[str, Any]:
    state_path = state_path or (project_root / "governance" / "health" / DEFAULT_SUPPORT_PAUSE_STATE_PATH.name)
    pause_requested, reason = _support_maintenance_pause_requested(payload)
    state = load_json(state_path)
    paused_rows = state.get("paused_processes") if isinstance(state.get("paused_processes"), list) else []
    attempted: list[dict[str, Any]] = []
    resumed: list[dict[str, Any]] = []

    if pause_requested:
        eligible = [
            row
            for row in candidates
            if str(row.get("category") or "") == "support_maintenance"
            and _safe_int(row.get("pid"), 0) > 0
            and _safe_float(row.get("cpu_percent"), 0.0) >= APPLY_CPU_THRESHOLD
            and not any(
                marker in str(row.get("command") or row.get("command_excerpt") or "").lower()
                for marker in SUPPORT_PAUSE_EXEMPT_MARKERS
            )
            and not _support_pause_exempt_for_storage_recovery(row, payload, project_root=project_root)
        ]
        pause_limit = max(1, _safe_int(os.getenv("RUNTIME_SUPPORT_MAINTENANCE_PAUSE_LIMIT", "2"), 2))
        for row in eligible[:pause_limit]:
            pid = _safe_int(row.get("pid"), 0)
            try:
                os.kill(pid, 0)
                os.kill(pid, signal.SIGSTOP)
                record = {
                    "pid": pid,
                    "ok": True,
                    "signal": "SIGSTOP",
                    "reason": reason,
                    "cpu_percent": round(_safe_float(row.get("cpu_percent"), 0.0), 3),
                    "command_excerpt": str(row.get("command") or "")[:220],
                    "paused_at_utc": iso_now(),
                }
                attempted.append(record)
            except Exception as exc:
                attempted.append(
                    {
                        "pid": pid,
                        "ok": False,
                        "signal": "SIGSTOP",
                        "reason": f"support_pause_failed:{exc}",
                        "command_excerpt": str(row.get("command") or "")[:220],
                    }
                )
        live_paused = {
            _safe_int(row.get("pid"), 0): row
            for row in paused_rows
            if _safe_int(row.get("pid"), 0) > 0
        }
        for row in attempted:
            if bool(row.get("ok", False)):
                live_paused[_safe_int(row.get("pid"), 0)] = row
        state_path.parent.mkdir(parents=True, exist_ok=True)
        state_path.write_text(
            json.dumps(
                {
                    "timestamp_utc": iso_now(),
                    "pause_requested": True,
                    "reason": reason,
                    "paused_processes": list(live_paused.values()),
                },
                ensure_ascii=True,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    else:
        for row in paused_rows:
            pid = _safe_int(row.get("pid"), 0)
            if pid <= 0:
                continue
            try:
                os.kill(pid, 0)
                os.kill(pid, signal.SIGCONT)
                resumed.append(
                    {
                        "pid": pid,
                        "ok": True,
                        "signal": "SIGCONT",
                        "reason": reason,
                        "command_excerpt": str(row.get("command_excerpt") or "")[:220],
                    }
                )
            except Exception as exc:
                resumed.append(
                    {
                        "pid": pid,
                        "ok": False,
                        "signal": "SIGCONT",
                        "reason": f"support_resume_failed:{exc}",
                        "command_excerpt": str(row.get("command_excerpt") or "")[:220],
                    }
                )
        if paused_rows or state_path.exists():
            state_path.parent.mkdir(parents=True, exist_ok=True)
            state_path.write_text(
                json.dumps(
                    {
                        "timestamp_utc": iso_now(),
                        "pause_requested": False,
                        "reason": reason,
                        "paused_processes": [],
                    },
                    ensure_ascii=True,
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )

    return {
        "pause_requested": pause_requested,
        "reason": reason,
        "attempted_count": len(attempted),
        "successful_count": sum(1 for row in attempted if bool(row.get("ok", False))),
        "resumed_count": len(resumed),
        "resume_successful_count": sum(1 for row in resumed if bool(row.get("ok", False))),
        "processes": attempted,
        "resumed_processes": resumed,
        "state_path": str(state_path),
    }


def _apply_process_throttle(
    candidates: list[dict[str, Any]],
    *,
    max_processes: int,
    env_overrides: dict[str, str] | None = None,
) -> dict[str, Any]:
    attempted: list[dict[str, Any]] = []
    use_background_taskpolicy = _runtime_throttle_uses_background_taskpolicy()
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
        current_nice = _safe_int(row.get("nice"), 0)
        target_nice = _target_nice_for_candidate(row, env_overrides)
        if "nice" in row and current_nice >= target_nice:
            renice_delta = 0
            renice_result = {
                "command": [],
                "returncode": 0,
                "ok": True,
                "skipped": True,
                "reason": "current_nice_at_or_above_target",
            }
        else:
            renice_delta = _renice_delta_for_target(current_nice, target_nice)
            renice_result = _run_apply_command(["renice", "-n", str(renice_delta), "-p", str(pid)])
        if use_background_taskpolicy:
            taskpolicy_result = _run_apply_command(["taskpolicy", "-b", "-p", str(pid)])
        else:
            taskpolicy_result = {
                "command": [],
                "returncode": 0,
                "ok": True,
                "skipped": True,
                "reason": "performance_core_efficiency_guard_active",
            }
        process_actions = {
            "pid": pid,
            "cpu_percent": row.get("cpu_percent"),
            "current_nice": current_nice if "nice" in row else None,
            "target_nice": target_nice,
            "renice_delta": renice_delta,
            "command_excerpt": str(row.get("command") or "")[:220],
            "renice": renice_result,
            "taskpolicy": taskpolicy_result,
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


def _storage_backlog_clean_for_writer_cooling(
    storage_stabilization: dict[str, Any],
    storage_pressure: dict[str, Any],
) -> bool:
    sql_writer_coordination = (
        storage_stabilization.get("sql_writer_coordination")
        if isinstance(storage_stabilization.get("sql_writer_coordination"), dict)
        else {}
    )
    if bool(storage_stabilization.get("drain_friendly_sql_required", False)):
        return False
    if bool(sql_writer_coordination.get("concentrated_core_drain", False)):
        return False
    total_pending = max(
        _safe_int(storage_stabilization.get("total_pending_lines"), 0),
        _safe_int(storage_pressure.get("total_pending_lines"), 0),
    )
    core_pending = max(
        _safe_int(storage_stabilization.get("core_pending_lines"), 0),
        _safe_int(storage_pressure.get("core_pending_lines"), 0),
    )
    coordination_pending = max(
        _safe_int(sql_writer_coordination.get("total_pending_lines"), 0),
        _safe_int(sql_writer_coordination.get("core_pending_lines"), 0),
    )
    pressure_index = _safe_float(storage_pressure.get("pressure_index"), 0.0)
    oldest_age = _safe_float(storage_pressure.get("oldest_pending_age_seconds"), 0.0)
    return bool(
        total_pending <= 0
        and core_pending <= 0
        and coordination_pending <= 25
        and pressure_index <= 0.05
        and oldest_age <= 30.0
    )


def _storage_writer_fluidity_cooling_allowed(
    *,
    storage_stabilization: dict[str, Any],
    storage_pressure: dict[str, Any],
    sql_writer_fluidity_contract: dict[str, Any],
) -> bool:
    contract = sql_writer_fluidity_contract if isinstance(sql_writer_fluidity_contract, dict) else {}
    measurements = contract.get("measurements") if isinstance(contract.get("measurements"), dict) else {}
    sql_writer_coordination = (
        storage_stabilization.get("sql_writer_coordination")
        if isinstance(storage_stabilization.get("sql_writer_coordination"), dict)
        else {}
    )
    pressure_index = _safe_float(storage_pressure.get("pressure_index"), 0.0)
    oldest_age = _safe_float(storage_pressure.get("oldest_pending_age_seconds"), 0.0)
    current_cap = _safe_int(measurements.get("current_sql_lane_cap"), 0)
    recommended_cap = _safe_int(measurements.get("recommended_sql_lane_cap"), 0)
    writer_cpu = _safe_float(measurements.get("storage_writer_cpu_percent"), 0.0)
    coordination_pending = max(
        _safe_int(sql_writer_coordination.get("total_pending_lines"), 0),
        _safe_int(sql_writer_coordination.get("core_pending_lines"), 0),
    )
    memory_level = str(measurements.get("memory_pressure_level") or "").strip().lower()
    reason = str(contract.get("reason") or "").strip()
    return bool(
        bool(contract.get("active", False))
        and reason == "storage_writer_heat_is_reducing_runtime_fluidity"
        and memory_level in {"", "normal"}
        and pressure_index <= 0.05
        and oldest_age <= 30.0
        and writer_cpu >= 85.0
        and current_cap > recommended_cap >= 1
        and coordination_pending <= 2000
    )


def _storage_writer_cooling_candidates(
    top_processes: list[dict[str, Any]],
    *,
    storage_stabilization: dict[str, Any],
    storage_pressure: dict[str, Any],
    sql_writer_fluidity_contract: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    clean_backlog_cooling = _storage_backlog_clean_for_writer_cooling(storage_stabilization, storage_pressure)
    fluidity_lane_cap_cooling = _storage_writer_fluidity_cooling_allowed(
        storage_stabilization=storage_stabilization,
        storage_pressure=storage_pressure,
        sql_writer_fluidity_contract=sql_writer_fluidity_contract or {},
    )
    if not clean_backlog_cooling and not fluidity_lane_cap_cooling:
        return []
    out: list[dict[str, Any]] = []
    for row in top_processes:
        if str(row.get("category") or "") != "storage_writer":
            continue
        if _safe_float(row.get("cpu_percent"), 0.0) < APPLY_CPU_THRESHOLD:
            continue
        command = str(row.get("command") or "")
        if "link_jsonl_to_sql.py" not in command:
            continue
        next_row = dict(row)
        next_row["throttle_candidate"] = True
        next_row["throttle_reason"] = (
            "clean_backlog_writer_cooling" if clean_backlog_cooling else "fluidity_lane_cap_writer_cooling"
        )
        next_row["terminate_when_apply"] = True
        out.append(next_row)
    out.sort(key=lambda item: _safe_float(item.get("cpu_percent"), 0.0), reverse=True)
    return out


def _apply_storage_writer_cooling(
    candidates: list[dict[str, Any]],
    *,
    max_processes: int,
) -> dict[str, Any]:
    attempted: list[dict[str, Any]] = []
    eligible = [
        row
        for row in candidates
        if _safe_int(row.get("pid"), 0) > 0 and bool(row.get("terminate_when_apply", False))
    ][: max(int(max_processes), 0)]
    for row in eligible:
        pid = _safe_int(row.get("pid"), 0)
        command_excerpt = str(row.get("command") or "")[:220]
        try:
            os.kill(pid, 0)
        except Exception as exc:
            attempted.append(
                {
                    "pid": pid,
                    "ok": False,
                    "skipped": True,
                    "reason": f"process_not_available:{exc}",
                    "command_excerpt": command_excerpt,
                }
            )
            continue
        try:
            reason = str(row.get("throttle_reason") or "clean_backlog_writer_cooling")
            os.kill(pid, signal.SIGTERM)
            attempted.append(
                {
                    "pid": pid,
                    "ok": True,
                    "signal": "SIGTERM",
                    "reason": reason,
                    "cpu_percent": row.get("cpu_percent"),
                    "command_excerpt": command_excerpt,
                }
            )
        except Exception as exc:
            attempted.append(
                {
                    "pid": pid,
                    "ok": False,
                    "reason": f"sigterm_failed:{exc}",
                    "cpu_percent": row.get("cpu_percent"),
                    "command_excerpt": command_excerpt,
                }
            )
    return {
        "cooling_requested": bool(eligible),
        "reason": str(eligible[0].get("throttle_reason") or "clean_backlog_writer_cooling") if eligible else "",
        "attempted_count": len(attempted),
        "successful_count": sum(1 for row in attempted if bool(row.get("ok", False))),
        "processes": attempted,
    }


def _canonical_registry_write_blocked(registry_out: Path, allow_source_registry_write: bool) -> bool:
    if allow_source_registry_write:
        return False
    try:
        return registry_out.resolve() == SOURCE_REGISTRY_PATH.resolve()
    except Exception:
        return False


def _apply_registry_collector_guard(
    project_root: Path,
    payload: dict[str, Any],
    *,
    registry_path: Path = DEFAULT_REGISTRY_PATH,
    candidate_registry_path: Path = DEFAULT_CANDIDATE_REGISTRY_PATH,
    source_write_guard_path: Path = DEFAULT_SOURCE_WRITE_GUARD_PATH,
    allow_source_registry_write: bool = False,
) -> dict[str, Any]:
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
    source_write_blocked = bool(changed_count) and _canonical_registry_write_blocked(path, allow_source_registry_write)
    candidate_out = candidate_registry_path if candidate_registry_path.is_absolute() else project_root / candidate_registry_path
    guard_out = source_write_guard_path if source_write_guard_path.is_absolute() else project_root / source_write_guard_path
    if changed_count:
        registry["updated_at_utc"] = iso_now()
        if source_write_blocked:
            candidate_out.parent.mkdir(parents=True, exist_ok=True)
            guard_out.parent.mkdir(parents=True, exist_ok=True)
            candidate_out.write_text(json.dumps(registry, ensure_ascii=True, indent=2), encoding="utf-8")
            write_payload(
                guard_out,
                {
                    "timestamp_utc": iso_now(),
                    "ok": True,
                    "overall_status": "ready",
                    "source_write_blocked": True,
                    "source_path": str(path),
                    "candidate_path": str(candidate_out),
                    "reason": "canonical_registry_requires_explicit_source_write",
                    "allow_env": "RUNTIME_THROTTLE_ALLOW_SOURCE_REGISTRY_WRITE=1",
                    "allow_cli": "scripts/ops/runtime_throttle_control.py --apply --allow-source-registry-write",
                },
            )
        else:
            path.write_text(json.dumps(registry, ensure_ascii=True, indent=2), encoding="utf-8")
    return {
        "applied": bool(changed_count),
        "changed_count": changed_count,
        "paper_runtime_changed_count": paper_changed_count,
        "collector_count": sum(1 for row in rows if bool(row.get("active", False)) and str(row.get("lifecycle_state") or "").strip().lower() == "data_collection_only"),
        "full_force_paper_stabilization": full_force_paper,
        "registry_source_write_blocked": source_write_blocked,
        "registry_source_written": bool(changed_count) and not source_write_blocked,
        "candidate_registry_path": str(candidate_out) if source_write_blocked else "",
        "source_write_guard_path": str(guard_out) if source_write_blocked else "",
        "source_registry_write_requires_explicit_operator_intent": True,
        "policy": policy,
        "registry_path": str(path),
    }


def apply_runtime_guard(
    project_root: Path,
    payload: dict[str, Any],
    *,
    override_path: Path = DEFAULT_OVERRIDE_PATH,
    registry_path: Path = DEFAULT_REGISTRY_PATH,
    candidate_registry_path: Path = DEFAULT_CANDIDATE_REGISTRY_PATH,
    source_write_guard_path: Path = DEFAULT_SOURCE_WRITE_GUARD_PATH,
    allow_source_registry_write: bool = False,
    max_renice_processes: int = 4,
) -> dict[str, Any]:
    profile = str(payload.get("throttle_profile") or "observe")
    runtime_snapshot = payload.get("runtime_snapshot") if isinstance(payload.get("runtime_snapshot"), dict) else {}
    storage_pressure = runtime_snapshot.get("storage_pressure") if isinstance(runtime_snapshot.get("storage_pressure"), dict) else {}
    storage_stabilization = payload.get("storage_stabilization") if isinstance(payload.get("storage_stabilization"), dict) else {}
    storage_drain_active = bool(storage_stabilization.get("drain_friendly_sql_required", False))
    p_core_feedback = payload.get("p_core_runtime_feedback") if isinstance(payload.get("p_core_runtime_feedback"), dict) else {}
    selected_writer_budget = max(
        _safe_int(p_core_feedback.get("shard_link_writer_lanes"), 0),
        _safe_int(p_core_feedback.get("preprocess_worker_budget"), 0),
    )
    writer_worker_budget = selected_writer_budget if selected_writer_budget > 0 else None
    max_writer_lanes = writer_worker_budget
    env_overrides = _runtime_env_overrides(
        profile,
        str(payload.get("memory_pressure_level") or "normal"),
        str(payload.get("compute_pressure_level") or "normal"),
        storage_drain_active=storage_drain_active,
        storage_pressure=storage_pressure,
        paper_capacity_contract=payload.get("paper_capacity_contract") if isinstance(payload.get("paper_capacity_contract"), dict) else {},
        cotenant_contract=payload.get("cotenant_awareness_contract") if isinstance(payload.get("cotenant_awareness_contract"), dict) else {},
        mlx_contract=payload.get("mlx_intelligence_contract") if isinstance(payload.get("mlx_intelligence_contract"), dict) else {},
        library_contract=payload.get("library_utilization_contract") if isinstance(payload.get("library_utilization_contract"), dict) else {},
        sql_writer_coordination=(storage_stabilization.get("sql_writer_coordination") if isinstance(storage_stabilization.get("sql_writer_coordination"), dict) else {}),
        paper_execution_policy=payload.get("paper_execution_policy") if isinstance(payload.get("paper_execution_policy"), dict) else {},
        writer_worker_budget=writer_worker_budget,
        max_writer_lanes=max_writer_lanes,
    )
    if _safe_float(storage_pressure.get("pressure_index"), 0.0) >= 1.0:
        safe_sql_pressure_keys = {
            "SQL_LINK_SERVICE_HOST_COOLING_ACTIVE",
            "SQL_LINK_SERVICE_PREPROCESS_WORKERS",
            "SQL_LINK_SERVICE_SHARD_WRITER_LANES",
            "SQL_LINK_SERVICE_MAX_SHARD_WRITER_LANES",
            "SQL_LINK_SERVICE_SMART_SHARD_PARALLELISM",
            "SQL_LINK_SERVICE_SENTINEL_SHARD_LANE_CAP",
            "SQL_LINK_SERVICE_HOT_SHARD_LANE_CAP",
            "SQL_LINK_SERVICE_WARM_SHARD_LANE_CAP",
            "SQL_LINK_SERVICE_COLD_SHARD_LANE_CAP",
        }
        env_overrides = {
            key: value
            for key, value in env_overrides.items()
            if not key.startswith("SQL_LINK_SERVICE_") or key in safe_sql_pressure_keys
        }
    mac_fluidity_contract = payload.get("mac_fluidity_contract") if isinstance(payload.get("mac_fluidity_contract"), dict) else {}
    mac_fluidity_env = (
        mac_fluidity_contract.get("env_overrides")
        if isinstance(mac_fluidity_contract.get("env_overrides"), dict)
        else {}
    )
    env_overrides.update({str(key): str(value) for key, value in mac_fluidity_env.items()})
    support_pause_recommended = bool(mac_fluidity_contract.get("support_pause_recommended", False))
    if support_pause_recommended:
        env_overrides["OPS_SUPPORT_MAINTENANCE_FREEZE"] = "1"
        env_overrides["MAC_FLUIDITY_SUPPORT_PAUSE"] = "1"
    else:
        # Protect/sustain profiles may throttle support jobs, but a full freeze is
        # reserved for an explicit Mac-fluidity support-pause decision.
        env_overrides["OPS_SUPPORT_MAINTENANCE_FREEZE"] = "0"
        env_overrides["MAC_FLUIDITY_SUPPORT_PAUSE"] = "0"
    support_candidates = payload.get("support_trim_candidates") if isinstance(payload.get("support_trim_candidates"), list) else []
    research_candidates = payload.get("research_training_trim_candidates") if isinstance(payload.get("research_training_trim_candidates"), list) else []
    paper_candidates = payload.get("paper_execution_pause_candidates") if isinstance(payload.get("paper_execution_pause_candidates"), list) else []
    top_processes = payload.get("top_processes") if isinstance(payload.get("top_processes"), list) else []
    active_sql_overrides = _sql_overrides_for_runtime_pressure(
        profile,
        storage_drain_active=storage_drain_active,
        storage_pressure=storage_pressure,
        sql_writer_coordination=(
            storage_stabilization.get("sql_writer_coordination")
            if isinstance(storage_stabilization.get("sql_writer_coordination"), dict)
            else {}
        ),
        writer_worker_budget=writer_worker_budget,
        max_writer_lanes=max_writer_lanes,
    )
    sql_writer_fluidity_contract = (
        payload.get("sql_writer_fluidity_contract")
        if isinstance(payload.get("sql_writer_fluidity_contract"), dict)
        else {}
    )
    if not sql_writer_fluidity_contract:
        sql_writer_fluidity_contract = _sql_writer_fluidity_contract(
            throttle_profile=profile,
            compute_pressure_level=str(payload.get("compute_pressure_level") or "normal"),
            memory_pressure_level=str(payload.get("memory_pressure_level") or "normal"),
            saturation_score=_safe_float(payload.get("host_saturation_score"), 0.0),
            storage_drain_active=storage_drain_active,
            storage_pressure_index=_safe_float(storage_pressure.get("pressure_index"), 0.0),
            storage_total_pending_lines=_safe_int(storage_pressure.get("total_pending_lines"), 0),
            storage_pending_threshold=_safe_int(storage_pressure.get("pending_lines_threshold"), 15000),
            storage_oldest_pending_age_seconds=_safe_float(storage_pressure.get("oldest_pending_age_seconds"), 0.0),
            host_pressure_attribution=(
                payload.get("host_pressure_attribution")
                if isinstance(payload.get("host_pressure_attribution"), dict)
                else {}
            ),
            mac_fluidity_contract=mac_fluidity_contract,
            current_sql_overrides=active_sql_overrides,
        )
    storage_writer_cooling_candidates = _storage_writer_cooling_candidates(
        top_processes,
        storage_stabilization=storage_stabilization,
        storage_pressure=storage_pressure,
        sql_writer_fluidity_contract=sql_writer_fluidity_contract,
    )
    throttle_candidates = (
        list(support_candidates)
        + list(research_candidates)
        + list(paper_candidates)
        + list(storage_writer_cooling_candidates)
    )
    sql_writer_fluidity_env = (
        sql_writer_fluidity_contract.get("env_overrides")
        if isinstance(sql_writer_fluidity_contract.get("env_overrides"), dict)
        else {}
    )
    if sql_writer_fluidity_env:
        env_overrides.update({str(key): str(value) for key, value in sql_writer_fluidity_env.items()})
        active_sql_overrides.update({str(key): str(value) for key, value in sql_writer_fluidity_env.items()})
    return {
        "applied": True,
        "override_path": str(override_path),
        "override_changed": _write_env_override(override_path, env_overrides, profile=profile),
        "env_override_count": len(env_overrides),
        "mac_fluidity_contract": {
            "overall_status": mac_fluidity_contract.get("overall_status", ""),
            "fluidity_band": mac_fluidity_contract.get("fluidity_band", ""),
            "fluidity_score": mac_fluidity_contract.get("fluidity_score", 0.0),
            "env_override_count": len(mac_fluidity_env),
        },
        "storage_drain_active": storage_drain_active,
        "drain_friendly_sql_overrides": active_sql_overrides,
        "sql_writer_fluidity_contract": {
            "active": bool(sql_writer_fluidity_contract.get("active", False)),
            "overall_status": sql_writer_fluidity_contract.get("overall_status", ""),
            "tier": sql_writer_fluidity_contract.get("tier", ""),
            "reason": sql_writer_fluidity_contract.get("reason", ""),
            "env_override_count": len(sql_writer_fluidity_env),
            "measurements": (
                sql_writer_fluidity_contract.get("measurements")
                if isinstance(sql_writer_fluidity_contract.get("measurements"), dict)
                else {}
            ),
        },
        "process_throttle": _apply_process_throttle(
            throttle_candidates,
            max_processes=max_renice_processes,
            env_overrides=env_overrides,
        ),
        "storage_writer_cooling": _apply_storage_writer_cooling(
            storage_writer_cooling_candidates,
            max_processes=max_renice_processes,
        ),
        "support_maintenance_pause": _apply_support_maintenance_pause(project_root, support_candidates, payload),
        "research_training_pause": _apply_research_training_pause(project_root, research_candidates, payload),
        "paper_execution_pause": _apply_paper_execution_pause(paper_candidates),
        "collector_guard": _apply_registry_collector_guard(
            project_root,
            payload,
            registry_path=registry_path,
            candidate_registry_path=candidate_registry_path,
            source_write_guard_path=source_write_guard_path,
            allow_source_registry_write=allow_source_registry_write,
        ),
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
        "paper_execution": _row("paper_execution", protected=False, throttle_candidate=True),
        "research_training": _row("research_training", protected=False, throttle_candidate=True),
        "macro_capture": _row("macro_capture", protected=True, throttle_candidate=False),
        "support_maintenance": _row("support_maintenance", protected=False, throttle_candidate=True),
        "storage_writer": _row("storage_writer", protected=False, throttle_candidate=False),
        "interactive_cotenant": _row("interactive_cotenant", protected=False, throttle_candidate=False),
        "system_cotenant": _row("system_cotenant", protected=False, throttle_candidate=False),
        "operator_observability": _row("operator_observability", protected=False, throttle_candidate=False),
        "unclassified": _row("unclassified", protected=False, throttle_candidate=False),
    }


def _host_pressure_attribution(domains: dict[str, dict[str, Any]], top_processes: list[dict[str, Any]]) -> dict[str, Any]:
    def cpu(category: str) -> float:
        return _safe_float((domains.get(category) or {}).get("cpu_percent"), 0.0)

    operator_cpu = round(cpu("operator_observability"), 3)
    research_cpu = round(cpu("research_training"), 3)
    protected_cpu = round(cpu("live_execution") + cpu("macro_capture"), 3)
    paper_cpu = round(cpu("paper_execution"), 3)
    throttle_candidate_cpu = round(cpu("support_maintenance"), 3)
    storage_writer_cpu = round(cpu("storage_writer"), 3)
    system_cpu = round(cpu("system_cotenant"), 3)
    interactive_cpu = round(cpu("interactive_cotenant"), 3)
    unclassified_cpu = round(cpu("unclassified"), 3)
    bot_owned_cpu = round(protected_cpu + research_cpu + paper_cpu + throttle_candidate_cpu + storage_writer_cpu + operator_cpu, 3)
    external_cpu = round(system_cpu + interactive_cpu + unclassified_cpu, 3)
    buckets = {
        "bot_owned": bot_owned_cpu,
        "protected_live_or_macro": protected_cpu,
        "research_training": research_cpu,
        "paper_execution": paper_cpu,
        "throttle_candidate_support": throttle_candidate_cpu,
        "storage_writer": storage_writer_cpu,
        "operator_observability": operator_cpu,
        "macos_system": system_cpu,
        "foreground_apps": interactive_cpu,
        "unknown": unclassified_cpu,
    }
    dominant_bucket = max(buckets.items(), key=lambda item: item[1])[0] if buckets else "unknown"
    external_dominant = external_cpu > bot_owned_cpu and external_cpu >= 35.0
    system_hot = system_cpu >= 35.0
    support_hot = throttle_candidate_cpu >= 35.0
    paper_hot = paper_cpu >= 35.0
    research_hot = research_cpu >= 35.0
    storage_writer_hot = storage_writer_cpu >= 35.0
    operator_observability_hot = operator_cpu >= 35.0
    protected_hot = protected_cpu >= 50.0
    bot_owned_pressure_dominant = bool(bot_owned_cpu >= external_cpu and bot_owned_cpu >= 35.0)
    support_pressure_dominant = bool(
        throttle_candidate_cpu >= 35.0
        and throttle_candidate_cpu >= max(system_cpu, interactive_cpu, unclassified_cpu, operator_cpu, protected_cpu, paper_cpu, research_cpu, storage_writer_cpu)
    )
    paper_execution_pressure_dominant = bool(
        paper_cpu >= 35.0
        and paper_cpu >= max(system_cpu, interactive_cpu, unclassified_cpu, throttle_candidate_cpu, operator_cpu, protected_cpu, research_cpu, storage_writer_cpu)
    )
    research_pressure_dominant = bool(
        research_cpu >= 35.0
        and research_cpu >= max(system_cpu, interactive_cpu, unclassified_cpu, throttle_candidate_cpu, operator_cpu, protected_cpu, paper_cpu, storage_writer_cpu)
    )
    operator_observability_pressure_dominant = bool(
        operator_cpu >= 35.0
        and operator_cpu >= max(system_cpu, interactive_cpu, unclassified_cpu, throttle_candidate_cpu, protected_cpu)
    )
    macos_system_pressure_dominant = bool(
        system_cpu >= 35.0
        and system_cpu >= max(throttle_candidate_cpu, operator_cpu, protected_cpu, interactive_cpu, unclassified_cpu)
    )
    protected_pressure_dominant = bool(
        protected_cpu >= 50.0
        and protected_cpu >= max(system_cpu, interactive_cpu, unclassified_cpu, throttle_candidate_cpu, operator_cpu, paper_cpu, research_cpu, storage_writer_cpu)
    )
    system_secondary_to_bot_owned = bool(system_hot and bot_owned_cpu >= max(system_cpu * 1.2, system_cpu + 40.0))
    support_trim_required = bool(
        support_hot
        and (
            support_pressure_dominant
            or dominant_bucket in {"bot_owned", "throttle_candidate_support"}
            or throttle_candidate_cpu >= system_cpu
        )
    )
    def priority_evidence_processes(category: str, aggregate_hot: bool) -> tuple[list[dict[str, Any]], str]:
        category_rows = [row for row in top_processes if str(row.get("category") or "") == category]
        individually_hot = [row for row in category_rows if _safe_float(row.get("cpu_percent"), 0.0) >= 20.0]
        if individually_hot or not aggregate_hot:
            return individually_hot, "individually_hot"
        distributed = [row for row in category_rows if _safe_float(row.get("cpu_percent"), 0.0) >= 5.0]
        return distributed, "distributed_aggregate_hot" if distributed else "missing"

    hot_support_processes, support_priority_evidence_mode = priority_evidence_processes(
        "support_maintenance", support_hot
    )
    hot_research_processes, research_priority_evidence_mode = priority_evidence_processes(
        "research_training", research_hot
    )
    hot_paper_processes, paper_priority_evidence_mode = priority_evidence_processes(
        "paper_execution", paper_hot
    )
    support_hot_low_priority = bool(
        support_hot
        and hot_support_processes
        and all(_safe_int(row.get("nice"), 0) >= 12 for row in hot_support_processes)
    )
    research_hot_low_priority = bool(
        research_hot
        and hot_research_processes
        and all(_safe_int(row.get("nice"), 0) >= 12 for row in hot_research_processes)
    )
    paper_hot_low_priority = bool(
        paper_hot
        and hot_paper_processes
        and all(_safe_int(row.get("nice"), 0) >= 12 for row in hot_paper_processes)
    )
    hot_external_processes = [
        {
            "pid": _safe_int(row.get("pid"), 0),
            "cpu_percent": _safe_float(row.get("cpu_percent"), 0.0),
            "category": str(row.get("category") or ""),
            "command_excerpt": str(row.get("command") or "")[:220],
        }
        for row in top_processes
        if str(row.get("category") or "") in {"system_cotenant", "interactive_cotenant", "unclassified"}
        and _safe_float(row.get("cpu_percent"), 0.0) >= 20.0
    ][:5]
    recommended_actions = ordered_unique(
        [
            "attribute the current host pressure to macOS/user co-tenants before widening bot workers"
            if external_dominant
            else "",
            "let Spotlight, indexing, backup, and suggestion services cool before launching wide training or extra collectors"
            if system_hot and not system_secondary_to_bot_owned
            else "",
            "trim support maintenance before touching live sleeves because support jobs are the hottest bot-owned pressure"
            if support_trim_required
            else "",
            "pause paper execution consumers while the paper ramp gate is blocked or host pressure is hot"
            if paper_hot
            else "",
            "downshift heavy research loops before treating them like protected live work"
            if research_hot
            else "",
            "let the SQL storage writer keep priority while backlog pressure is critical"
            if storage_writer_hot
            else "",
            "downshift heavy livefeed/operator views before widening collectors or training"
            if operator_observability_pressure_dominant
            else "",
            "keep live/paper and macro-capture lanes protected, but avoid adding new protected work while they dominate CPU"
            if protected_hot
            else "",
        ]
    )
    return {
        "bot_owned_cpu_percent": bot_owned_cpu,
        "protected_live_or_macro_cpu_percent": protected_cpu,
        "research_training_cpu_percent": research_cpu,
        "paper_execution_cpu_percent": paper_cpu,
        "operator_observability_cpu_percent": operator_cpu,
        "throttle_candidate_support_cpu_percent": throttle_candidate_cpu,
        "storage_writer_cpu_percent": storage_writer_cpu,
        "external_cpu_percent": external_cpu,
        "macos_system_cpu_percent": system_cpu,
        "foreground_app_cpu_percent": interactive_cpu,
        "unknown_cpu_percent": unclassified_cpu,
        "dominant_bucket": dominant_bucket,
        "external_pressure_dominant": external_dominant,
        "bot_owned_pressure_dominant": bot_owned_pressure_dominant,
        "support_pressure_dominant": support_pressure_dominant,
        "paper_execution_pressure_dominant": paper_execution_pressure_dominant,
        "research_pressure_dominant": research_pressure_dominant,
        "operator_observability_pressure_dominant": operator_observability_pressure_dominant,
        "macos_system_pressure_dominant": macos_system_pressure_dominant,
        "protected_pressure_dominant": protected_pressure_dominant,
        "system_secondary_to_bot_owned": system_secondary_to_bot_owned,
        "support_trim_required": support_trim_required,
        "system_cotenant_hot": system_hot,
        "support_jobs_hot": support_hot,
        "paper_execution_hot": paper_hot,
        "research_training_hot": research_hot,
        "storage_writer_hot": storage_writer_hot,
        "support_hot_low_priority": support_hot_low_priority,
        "research_hot_low_priority": research_hot_low_priority,
        "paper_hot_low_priority": paper_hot_low_priority,
        "low_priority_evidence_mode": {
            "support_maintenance": support_priority_evidence_mode,
            "research_training": research_priority_evidence_mode,
            "paper_execution": paper_priority_evidence_mode,
        },
        "operator_observability_hot": operator_observability_hot,
        "hot_support_processes": [
            {
                "pid": _safe_int(row.get("pid"), 0),
                "nice": _safe_int(row.get("nice"), 0),
                "cpu_percent": _safe_float(row.get("cpu_percent"), 0.0),
                "command_excerpt": str(row.get("command") or "")[:220],
            }
            for row in hot_support_processes[:5]
        ],
        "hot_research_processes": [
            {
                "pid": _safe_int(row.get("pid"), 0),
                "nice": _safe_int(row.get("nice"), 0),
                "cpu_percent": _safe_float(row.get("cpu_percent"), 0.0),
                "command_excerpt": str(row.get("command") or "")[:220],
            }
            for row in hot_research_processes[:5]
        ],
        "hot_paper_processes": [
            {
                "pid": _safe_int(row.get("pid"), 0),
                "nice": _safe_int(row.get("nice"), 0),
                "cpu_percent": _safe_float(row.get("cpu_percent"), 0.0),
                "command_excerpt": str(row.get("command") or "")[:220],
            }
            for row in hot_paper_processes[:5]
        ],
        "protected_work_hot": protected_hot,
        "hot_external_processes": hot_external_processes,
        "recommended_actions": recommended_actions,
        "policy": "attribute_host_pressure_before_widening_backlog_training_or_collector_work",
    }


def _runtime_saturation_governor_v2(
    *,
    saturation_score: float,
    throttle_profile: str,
    compute_pressure_level: str,
    memory_pressure_level: str,
    storage_total_pending_lines: int,
    storage_oldest_pending_age_seconds: float,
    support_trim_candidates: list[dict[str, Any]],
    research_training_trim_candidates: list[dict[str, Any]],
    paper_execution_policy: dict[str, Any] | None = None,
    paper_execution_pause_candidates: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    paper_execution_policy = paper_execution_policy if isinstance(paper_execution_policy, dict) else {}
    paper_execution_pause_candidates = paper_execution_pause_candidates if isinstance(paper_execution_pause_candidates, list) else []
    score = _safe_float(saturation_score, 0.0)
    if score >= 85.0 or throttle_profile == "protect_live":
        band = "protect"
    elif score >= 75.0:
        band = "saturated"
    elif score >= 60.0:
        band = "guarded"
    elif score >= 45.0:
        band = "advisory"
    else:
        band = "normal"

    paper_execution_paused = bool(paper_execution_policy.get("pause_paper_execution", False))
    bounded_compute_micro_canary = bool(
        band in {"normal", "advisory", "guarded"}
        and score <= 65.0
        and compute_pressure_level == "high"
        and memory_pressure_level == "normal"
        and int(storage_total_pending_lines) <= 1500
        and float(storage_oldest_pending_age_seconds) <= 120.0
        and not support_trim_candidates
        and not research_training_trim_candidates
        and not paper_execution_paused
        and not paper_execution_pause_candidates
    )
    training_paused = bool(
        band in {"guarded", "saturated", "protect"}
        or (compute_pressure_level == "high" and not bounded_compute_micro_canary)
        or memory_pressure_level in {"elevated", "high"}
    )
    if bounded_compute_micro_canary:
        max_parallel_trainings = 1
        training_mode = "micro_canary_only"
    elif training_paused:
        max_parallel_trainings = 0
        training_mode = "paused_for_host_headroom"
    elif band == "advisory":
        max_parallel_trainings = 1
        training_mode = "micro_canary_only"
    else:
        max_parallel_trainings = 2
        training_mode = "small_batch_allowed"

    collector_policy = _collector_guard_policy(throttle_profile, memory_pressure_level, compute_pressure_level)
    support_policy = "off_hours_or_niced_only" if band in {"guarded", "saturated", "protect"} else "bounded_inline_ok"
    paper_policy = "protect_paper_and_live_data_lanes"
    paper_execution_allowed = bool(paper_execution_policy.get("paper_execution_allowed", True))
    if band == "normal" and not paper_execution_paused:
        paper_policy = "normal_paper_live_data_priority"
    if paper_execution_paused:
        paper_policy = "pause_paper_execution_consumers_until_paper_gate_clears"

    return {
        "active": True,
        "mode": "runtime_saturation_governor_v2",
        "host_saturation_score": round(float(score), 3),
        "saturation_band": band,
        "thresholds": {
            "advisory_at": 45.0,
            "pause_training_at": 60.0,
            "support_off_hours_at": 60.0,
            "saturated_at": 75.0,
            "protect_at": 85.0,
        },
        "training_policy": {
            "mode": training_mode,
            "training_paused": training_paused,
            "max_parallel_trainings": int(max_parallel_trainings),
            "batch10_allowed": bool(not training_paused and band == "normal"),
            "batch20_allowed": False,
            "micro_canary_allowed": bool(not training_paused and band in {"normal", "advisory"}),
            "reason": (
                "bounded_compute_pressure_micro_canary"
                if bounded_compute_micro_canary
                else "host_saturation_or_memory_pressure"
                if training_paused
                else "host_headroom_available"
            ),
        },
        "collector_policy": {
            "mode": collector_policy.get("compute_guard_mode"),
            "capture_mode": collector_policy.get("capture_mode"),
            "sample_rate": collector_policy.get("sample_rate"),
            "freshness_slo_minimum_seconds": collector_policy.get("freshness_slo_minimum_seconds"),
            "max_daily_mb": collector_policy.get("max_daily_mb"),
        },
        "support_policy": {
            "mode": support_policy,
            "support_trim_candidate_count": len(support_trim_candidates),
            "research_training_trim_candidate_count": len(research_training_trim_candidates),
        },
        "paper_live_data_policy": {
            "mode": paper_policy,
            "protect_live_execution_read_only": True,
            "protect_paper_execution_queue": True,
            "paper_execution_allowed": paper_execution_allowed,
            "paper_execution_consumer_paused": paper_execution_paused,
            "paper_execution_pause_candidate_count": len(paper_execution_pause_candidates),
            "paper_execution_pause_reason": str(paper_execution_policy.get("reason") or ""),
            "do_not_restart_healthy_loops_for_pressure_only": True,
        },
        "backlog_policy": {
            "storage_total_pending_lines": int(storage_total_pending_lines),
            "storage_oldest_pending_age_seconds": round(float(storage_oldest_pending_age_seconds), 3),
            "writer_can_continue": True,
            "do_not_widen_collectors_while_training_paused": training_paused,
        },
        "recommended_commands": [
            ["./scripts/ops/opsctl.sh", "runtime-throttle", "--apply", "--json"],
            ["./scripts/ops/opsctl.sh", "writer-cycle-coordinator", "--json"],
            ["./scripts/ops/opsctl.sh", "training-runtime-control", "--json"],
        ],
        "policy": "host_saturation_is_the_primary_traffic_light_for_training_collectors_support_jobs_and_paper_live_data",
    }


def _mac_fluidity_contract(
    *,
    overall_status: str,
    throttle_profile: str,
    saturation_score: float,
    compute_pressure_level: str,
    memory_pressure_level: str,
    storage_pressure_index: float,
    storage_total_pending_lines: int,
    storage_pending_threshold: int,
    storage_oldest_pending_age_seconds: float,
    storage_oldest_age_threshold_seconds: float,
    host_pressure_attribution: dict[str, Any],
    cotenant_contract: dict[str, Any],
    runtime_saturation_governor: dict[str, Any],
    storage_overlay_relief: dict[str, Any] | None = None,
) -> dict[str, Any]:
    foreground_cpu = _safe_float(host_pressure_attribution.get("foreground_app_cpu_percent"), 0.0)
    system_cpu = _safe_float(host_pressure_attribution.get("macos_system_cpu_percent"), 0.0)
    support_cpu = _safe_float(host_pressure_attribution.get("throttle_candidate_support_cpu_percent"), 0.0)
    research_cpu = _safe_float(host_pressure_attribution.get("research_training_cpu_percent"), 0.0)
    paper_cpu = _safe_float(host_pressure_attribution.get("paper_execution_cpu_percent"), 0.0)
    protected_cpu = _safe_float(host_pressure_attribution.get("protected_live_or_macro_cpu_percent"), 0.0)
    storage_writer_cpu = _safe_float(host_pressure_attribution.get("storage_writer_cpu_percent"), 0.0)
    operator_cpu = _safe_float(host_pressure_attribution.get("operator_observability_cpu_percent"), 0.0)
    saturation_band = str(runtime_saturation_governor.get("saturation_band") or "normal").strip().lower()
    training_policy = (
        runtime_saturation_governor.get("training_policy")
        if isinstance(runtime_saturation_governor.get("training_policy"), dict)
        else {}
    )
    runtime_micro_canary_allowed = bool(
        training_policy.get("micro_canary_allowed", False)
        and _safe_int(training_policy.get("max_parallel_trainings"), 0) >= 1
    )
    storage_overlay_relief = storage_overlay_relief if isinstance(storage_overlay_relief, dict) else {}
    overlay_fluidity_managed = bool(storage_overlay_relief.get("bounded", False))
    storage_clear = bool(
        (
            storage_pressure_index < 0.35
            and int(storage_total_pending_lines) < max(int(storage_pending_threshold), 1)
            and float(storage_oldest_pending_age_seconds) <= max(float(storage_oldest_age_threshold_seconds), 1.0)
        )
        or overlay_fluidity_managed
    )
    bounded_writer_fluidity_managed = bool(
        storage_clear
        and memory_pressure_level == "normal"
        and bool(host_pressure_attribution.get("storage_writer_hot", False))
        and storage_writer_cpu <= 110.0
        and support_cpu < 20.0
        and research_cpu < 20.0
        and _safe_float(host_pressure_attribution.get("paper_execution_cpu_percent"), 0.0) < 20.0
        and _safe_float(host_pressure_attribution.get("protected_live_or_macro_cpu_percent"), 0.0) < 20.0
        and operator_cpu < 35.0
        and _safe_float(saturation_score, 0.0) < 75.0
    )
    foreground_active = bool(
        foreground_cpu >= 20.0
        or bool(cotenant_contract.get("active", False))
        or _safe_int(cotenant_contract.get("open_app_count"), 0) > 0
    )
    score = 100.0
    score -= max(0.0, _safe_float(saturation_score, 0.0) - 35.0) * 0.35
    if throttle_profile in {"soft_cap", "sustain"}:
        score -= 2.0
    if compute_pressure_level == "elevated":
        score -= 4.0
    elif compute_pressure_level == "high":
        score -= 16.0
    if memory_pressure_level == "elevated":
        score -= 8.0
    elif memory_pressure_level == "high":
        score -= 24.0
    if not storage_clear:
        score -= 8.0
    if foreground_cpu >= 90.0:
        score -= 5.0
    if system_cpu >= 90.0:
        score -= 4.0
    if support_cpu >= 60.0:
        score -= 4.0
    if research_cpu >= 60.0:
        score -= 5.0
    score = round(max(0.0, min(score, 100.0)), 2)

    if overall_status == "blocked" or throttle_profile == "protect_live" or memory_pressure_level == "high":
        band = "protect"
    elif bounded_writer_fluidity_managed:
        band = "guarded_smooth"
        score = max(score, 86.0)
    elif saturation_band in {"saturated", "protect"} or compute_pressure_level == "high" or score < 75.0:
        band = "strained"
    elif throttle_profile in {"soft_cap", "sustain"} or saturation_band in {"advisory", "guarded"} or foreground_active:
        band = "guarded_smooth"
    elif score >= 96.0:
        band = "silky"
    else:
        band = "comfortable"

    status = "ready" if score >= 90.0 and band not in {"strained", "protect"} else ("watch" if score >= 75.0 else "needs_work")
    mode_overrides: dict[str, str]
    if band == "protect":
        mode_overrides = {
            "DATA_COLLECTION_RESOURCE_SAMPLE_RATE": "0.15",
            "DATA_COLLECTION_RESOURCE_CAPTURE_MODE": "thin_sample",
            "OPS_SUPPORT_JOB_NICE": "20",
            "YTDLP_SUPPORT_NICE": "20",
            "MACRO_YTDLP_SUPPORT_NICE": "20",
            "OPS_SUPPORT_JOBS_BACKGROUND_POLICY": "1",
            "SUPPORT_MAINTENANCE_CONCURRENCY": "1",
            "TRAINING_RUNTIME_GOVERNOR_MODE": "paused_for_host_headroom",
            "TRAINING_RUNTIME_MAX_PARALLEL": "0",
            "TRAINING_RUNTIME_PAUSED_FOR_HOST_HEADROOM": "1",
            "SHADOW_LOOP_RUNTIME_PAUSE_SLEEP_SECONDS": "75",
            "ONE_NUMBERS_REFRESH_INTERVAL_SECONDS": "1200",
            "INGESTION_BACKPRESSURE_REFRESH_INTERVAL_SECONDS": "120",
        }
    elif band == "strained":
        mode_overrides = {
            "DATA_COLLECTION_RESOURCE_SAMPLE_RATE": "0.25",
            "DATA_COLLECTION_RESOURCE_CAPTURE_MODE": "sampled",
            "OPS_SUPPORT_JOB_NICE": "20",
            "YTDLP_SUPPORT_NICE": "20",
            "MACRO_YTDLP_SUPPORT_NICE": "20",
            "OPS_SUPPORT_JOBS_BACKGROUND_POLICY": "1",
            "SUPPORT_MAINTENANCE_CONCURRENCY": "1",
            "TRAINING_RUNTIME_GOVERNOR_MODE": "paused_for_host_headroom",
            "TRAINING_RUNTIME_MAX_PARALLEL": "0",
            "TRAINING_RUNTIME_PAUSED_FOR_HOST_HEADROOM": "1",
            "SHADOW_LOOP_RUNTIME_PAUSE_SLEEP_SECONDS": "60",
            "ONE_NUMBERS_REFRESH_INTERVAL_SECONDS": "900",
            "INGESTION_BACKPRESSURE_REFRESH_INTERVAL_SECONDS": "120",
        }
        bounded_micro_canary = bool(
            runtime_micro_canary_allowed
            and storage_clear
            and memory_pressure_level == "normal"
            and _safe_float(saturation_score, 0.0) <= 65.0
            and support_cpu < 20.0
            and research_cpu < 20.0
            and paper_cpu < 20.0
            and protected_cpu < 20.0
            and operator_cpu < 35.0
        )
        if bounded_micro_canary:
            mode_overrides.update(
                {
                    "TRAINING_RUNTIME_GOVERNOR_MODE": "micro_canary_only",
                    "TRAINING_RUNTIME_MAX_PARALLEL": "1",
                    "TRAINING_RUNTIME_PAUSED_FOR_HOST_HEADROOM": "0",
                    "MAC_FLUIDITY_BOUNDED_CANARY": "1",
                }
            )
    elif band == "guarded_smooth":
        mode_overrides = {
            "DATA_COLLECTION_RESOURCE_SAMPLE_RATE": "0.30",
            "DATA_COLLECTION_RESOURCE_CAPTURE_MODE": "sampled",
            "OPS_SUPPORT_JOB_NICE": "20",
            "YTDLP_SUPPORT_NICE": "20",
            "MACRO_YTDLP_SUPPORT_NICE": "20",
            "OPS_SUPPORT_JOBS_BACKGROUND_POLICY": "1",
            "SUPPORT_MAINTENANCE_CONCURRENCY": "1",
            "TRAINING_RUNTIME_GOVERNOR_MODE": "micro_canary_only",
            "TRAINING_RUNTIME_MAX_PARALLEL": "1",
            "TRAINING_RUNTIME_PAUSED_FOR_HOST_HEADROOM": "0",
            "SHADOW_LOOP_RUNTIME_PAUSE_SLEEP_SECONDS": "60",
            "ONE_NUMBERS_REFRESH_INTERVAL_SECONDS": "600",
            "INGESTION_BACKPRESSURE_REFRESH_INTERVAL_SECONDS": "120",
        }
    elif band == "comfortable":
        mode_overrides = {
            "DATA_COLLECTION_RESOURCE_SAMPLE_RATE": "0.65",
            "DATA_COLLECTION_RESOURCE_CAPTURE_MODE": "sampled",
            "OPS_SUPPORT_JOB_NICE": "12",
            "YTDLP_SUPPORT_NICE": "12",
            "MACRO_YTDLP_SUPPORT_NICE": "12",
            "OPS_SUPPORT_JOBS_BACKGROUND_POLICY": "0",
            "TRAINING_RUNTIME_GOVERNOR_MODE": "small_batch_allowed",
            "TRAINING_RUNTIME_MAX_PARALLEL": "2",
            "SHADOW_LOOP_RUNTIME_PAUSE_SLEEP_SECONDS": "15",
            "ONE_NUMBERS_REFRESH_INTERVAL_SECONDS": "300",
            "INGESTION_BACKPRESSURE_REFRESH_INTERVAL_SECONDS": "120",
        }
    else:
        mode_overrides = {
            "DATA_COLLECTION_RESOURCE_SAMPLE_RATE": "1.0",
            "DATA_COLLECTION_RESOURCE_CAPTURE_MODE": "full",
            "OPS_SUPPORT_JOB_NICE": "12",
            "YTDLP_SUPPORT_NICE": "12",
            "MACRO_YTDLP_SUPPORT_NICE": "12",
            "OPS_SUPPORT_JOBS_BACKGROUND_POLICY": "0",
            "TRAINING_RUNTIME_GOVERNOR_MODE": "small_batch_allowed",
            "TRAINING_RUNTIME_MAX_PARALLEL": "2",
            "SHADOW_LOOP_RUNTIME_PAUSE_SLEEP_SECONDS": "0",
            "ONE_NUMBERS_REFRESH_INTERVAL_SECONDS": "180",
            "INGESTION_BACKPRESSURE_REFRESH_INTERVAL_SECONDS": "60",
        }
    research_writer_contention = bool(research_cpu >= 60.0 and storage_writer_cpu >= 50.0 and storage_clear)
    research_pause_recommended = bool(research_cpu >= 60.0 and (score < 94.0 or research_writer_contention))
    if research_pause_recommended:
        mode_overrides.update(
            {
                "TRAINING_RUNTIME_GOVERNOR_MODE": "paused_for_mac_fluidity",
                "TRAINING_RUNTIME_MAX_PARALLEL": "0",
                "TRAINING_RUNTIME_PAUSED_FOR_HOST_HEADROOM": "1",
                "MAC_FLUIDITY_RESEARCH_PAUSE": "1",
                "MAC_FLUIDITY_RESEARCH_WRITER_CONTENTION": "1" if research_writer_contention else "0",
                "RUNTIME_RESEARCH_TRAINING_PAUSE_LIMIT": "8",
            }
        )
    support_pause_recommended = bool(support_cpu >= 50.0 and score < 90.0 and storage_clear)
    if support_pause_recommended:
        mode_overrides.update(
            {
                "OPS_SUPPORT_MAINTENANCE_FREEZE": "1",
                "SUPPORT_MAINTENANCE_CONCURRENCY": "0",
                "MAC_FLUIDITY_SUPPORT_PAUSE": "1",
            }
        )
    env_overrides = {
        "MAC_FLUIDITY_CONTRACT_ENABLED": "1",
        "MAC_FLUIDITY_BAND": band,
        "MAC_FLUIDITY_STATUS": status,
        "MAC_FLUIDITY_SCORE": f"{score:.2f}",
        "MAC_FOREGROUND_FIRST": "1",
        "MAC_UI_RESPONSIVENESS_PRIORITY": "1",
        "BOT_MAC_FLUIDITY_MODE": band,
        "RUNTIME_COTENANT_AWARE": "1" if foreground_active else "0",
        **mode_overrides,
    }
    return {
        "active": True,
        "overall_status": status,
        "fluidity_band": band,
        "fluidity_score": score,
        "foreground_first": True,
        "foreground_active": foreground_active,
        "research_pause_recommended": research_pause_recommended,
        "research_writer_contention": research_writer_contention,
        "support_pause_recommended": support_pause_recommended,
        "storage_clear_for_fluidity": storage_clear,
        "measurements": {
            "host_saturation_score": round(_safe_float(saturation_score), 3),
            "saturation_band": saturation_band,
            "throttle_profile": throttle_profile,
            "compute_pressure_level": compute_pressure_level,
            "memory_pressure_level": memory_pressure_level,
            "foreground_app_cpu_percent": round(foreground_cpu, 3),
            "macos_system_cpu_percent": round(system_cpu, 3),
            "support_cpu_percent": round(support_cpu, 3),
            "research_training_cpu_percent": round(research_cpu, 3),
            "storage_writer_cpu_percent": round(storage_writer_cpu, 3),
            "operator_observability_cpu_percent": round(operator_cpu, 3),
            "storage_pressure_index": round(float(storage_pressure_index), 3),
            "storage_total_pending_lines": int(storage_total_pending_lines),
            "storage_pending_threshold": int(storage_pending_threshold),
            "bounded_writer_fluidity_managed": bounded_writer_fluidity_managed,
            "overlay_fluidity_managed": overlay_fluidity_managed,
            "storage_overlay_relief": storage_overlay_relief,
        },
        "env_overrides": env_overrides,
        "policy": "foreground_first_mac_fluidity_without_starving_single_sql_writer_or_opening_live_execution",
    }


def _sql_writer_fluidity_contract(
    *,
    throttle_profile: str,
    compute_pressure_level: str,
    memory_pressure_level: str,
    saturation_score: float,
    storage_drain_active: bool,
    storage_pressure_index: float,
    storage_total_pending_lines: int,
    storage_pending_threshold: int,
    storage_oldest_pending_age_seconds: float,
    host_pressure_attribution: dict[str, Any],
    mac_fluidity_contract: dict[str, Any],
    current_sql_overrides: dict[str, str] | None = None,
) -> dict[str, Any]:
    host_pressure_attribution = host_pressure_attribution if isinstance(host_pressure_attribution, dict) else {}
    mac_fluidity_contract = mac_fluidity_contract if isinstance(mac_fluidity_contract, dict) else {}
    current_sql_overrides = current_sql_overrides if isinstance(current_sql_overrides, dict) else {}
    storage_writer_cpu = _safe_float(host_pressure_attribution.get("storage_writer_cpu_percent"), 0.0)
    research_cpu = _safe_float(host_pressure_attribution.get("research_training_cpu_percent"), 0.0)
    support_cpu = _safe_float(host_pressure_attribution.get("throttle_candidate_support_cpu_percent"), 0.0)
    operator_cpu = _safe_float(host_pressure_attribution.get("operator_observability_cpu_percent"), 0.0)
    fluidity_band = str(mac_fluidity_contract.get("fluidity_band") or "").strip().lower()
    fluidity_status = str(mac_fluidity_contract.get("overall_status") or "").strip().lower()
    fluidity_score = _safe_float(mac_fluidity_contract.get("fluidity_score"), 100.0)
    saturation = _safe_float(saturation_score, 0.0)
    backlog_hot = bool(
        int(storage_total_pending_lines) >= max(int(storage_pending_threshold), 1)
        or float(storage_oldest_pending_age_seconds) >= 240.0
    )
    idle_backlog_cooling = bool(
        not storage_drain_active
        and int(storage_total_pending_lines) <= 0
        and float(storage_pressure_index) <= 0.05
        and float(storage_oldest_pending_age_seconds) <= 30.0
    )
    writer_hot = bool(storage_writer_cpu >= 85.0 or bool(host_pressure_attribution.get("storage_writer_hot", False)))
    fluidity_strained = bool(
        fluidity_band in {"strained", "protect"}
        or fluidity_status in {"needs_work", "degraded", "blocked"}
        or fluidity_score < 80.0
        or saturation >= 60.0
        or compute_pressure_level == "high"
        or memory_pressure_level in {"elevated", "high"}
    )
    active = bool(
        writer_hot
        and (
            fluidity_strained
            or storage_writer_cpu >= 150.0
            or (storage_drain_active and backlog_hot)
            or support_cpu >= 50.0
            or research_cpu >= 50.0
            or operator_cpu >= 70.0
        )
    )
    if not active:
        return {
            "active": False,
            "overall_status": "ready",
            "tier": "observe",
            "reason": "storage_writer_cpu_within_fluidity_budget",
            "measurements": {
                "storage_writer_cpu_percent": round(storage_writer_cpu, 3),
                "host_saturation_score": round(saturation, 3),
                "fluidity_band": fluidity_band or "unknown",
                "fluidity_status": fluidity_status or "unknown",
                "fluidity_score": round(fluidity_score, 2),
            },
            "env_overrides": {},
            "policy": "cap_sql_writer_fanout_only_when_writer_heat_hurts_runtime_fluidity",
        }

    if (
        memory_pressure_level == "high"
        or fluidity_band == "protect"
        or throttle_profile == "protect_live"
        or storage_writer_cpu >= 350.0
        or saturation >= 75.0
    ):
        tier = "protect"
        base_cap = 1
        interval = "150"
        hot_min = "600"
        queue_min = "1800"
        hot_batch = "40000"
        queue_batch = "25000"
        merge_max = "20"
        governance_batch_max_bytes = str(8 * 1024 * 1024)
        ingest_host_load_sleep = "0.75"
        ingest_flush_sleep = "0.10"
        ingest_file_sleep = "0.50"
    elif (
        fluidity_band == "strained"
        or fluidity_status == "needs_work"
        or compute_pressure_level == "high"
        or storage_writer_cpu >= 150.0
        or saturation >= 60.0
    ):
        tier = "guarded_relief"
        base_cap = 2
        interval = "75"
        hot_min = "240"
        queue_min = "900"
        hot_batch = "80000"
        queue_batch = "55000"
        merge_max = "30"
        governance_batch_max_bytes = str(12 * 1024 * 1024)
        ingest_host_load_sleep = "0.50"
        ingest_flush_sleep = "0.05"
        ingest_file_sleep = "0.25"
    else:
        tier = "calm"
        base_cap = 3
        interval = "30"
        hot_min = "90"
        queue_min = "360"
        hot_batch = "160000"
        queue_batch = "110000"
        merge_max = "45"
        governance_batch_max_bytes = str(16 * 1024 * 1024)
        ingest_host_load_sleep = "0.25"
        ingest_flush_sleep = "0.02"
        ingest_file_sleep = "0.10"

    governance_timeout = {"protect": "120", "guarded_relief": "180"}.get(tier, "240")
    governance_max_files = {"protect": "6", "guarded_relief": "8"}.get(tier, "10")

    current_cap = max(
        _safe_int(current_sql_overrides.get("SQL_LINK_SERVICE_PREPROCESS_WORKERS"), 0),
        _safe_int(current_sql_overrides.get("SQL_LINK_SERVICE_SHARD_WRITER_LANES"), 0),
        _safe_int(current_sql_overrides.get("SQL_LINK_SERVICE_MAX_SHARD_WRITER_LANES"), 0),
    )
    if current_cap > 0:
        lane_cap = max(1, min(base_cap, current_cap))
    else:
        lane_cap = base_cap
    if tier == "protect" and backlog_hot and memory_pressure_level == "normal" and fluidity_band != "protect":
        lane_cap = max(lane_cap, 2)
    lane_cap = max(1, min(lane_cap, 3))
    warm_lane_cap = max(1, min(2, lane_cap))

    env_overrides = {
        "SQL_LINK_SERVICE_FLUIDITY_GOVERNOR_ACTIVE": "1",
        "SQL_LINK_SERVICE_FLUIDITY_GOVERNOR_TIER": tier,
        "SQL_LINK_SERVICE_HOST_COOLING_ACTIVE": "1",
        "SQL_LINK_SERVICE_ADAPTIVE_WRITER_ENABLED": "1",
        "SQL_LINK_SERVICE_INTERVAL_SECONDS": interval,
        "SQL_LINK_SERVICE_HOT_MIN_INTERVAL_SECONDS": hot_min,
        "SQL_LINK_SERVICE_QUEUE_MIN_INTERVAL_SECONDS": queue_min,
        "SQL_LINK_SERVICE_HOT_BATCH_SIZE": hot_batch,
        "SQL_LINK_SERVICE_QUEUE_BATCH_SIZE": queue_batch,
        "SQL_LINK_SERVICE_PREPROCESS_WORKERS": str(lane_cap),
        "BACKLOG_PCORE_PREPROCESS_WORKERS": str(lane_cap),
        "SQL_LINK_SERVICE_SHARD_WRITER_LANES": str(lane_cap),
        "SQL_LINK_SERVICE_MAX_SHARD_WRITER_LANES": str(lane_cap),
        "SQL_LINK_SERVICE_MERGE_MAX_SECONDS_PER_CYCLE": merge_max,
        "SQL_LINK_SERVICE_PROGRESS_HEARTBEAT_SECONDS": "20",
        "SQL_LINK_SERVICE_SMART_SHARD_PARALLELISM": "1",
        "SQL_LINK_SERVICE_SENTINEL_SHARD_LANE_CAP": "1",
        "SQL_LINK_SERVICE_HOT_SHARD_LANE_CAP": str(lane_cap),
        "SQL_LINK_SERVICE_WARM_SHARD_LANE_CAP": str(warm_lane_cap),
        "SQL_LINK_SERVICE_COLD_SHARD_LANE_CAP": "1",
        "SQL_LINK_SERVICE_SKIP_FRESH_IDLE_SHARDS": "1",
        "SQL_LINK_SERVICE_IDLE_SHARD_MAX_AGE_SECONDS": "180",
        "SQL_LINK_SERVICE_SHARD_GOVERNANCE_MAX_FILES": governance_max_files,
        "SQL_LINK_SERVICE_SHARD_GOVERNANCE_MAX_BYTES_PER_FILE": str(128 * 1024 * 1024),
        "SQL_LINK_SERVICE_SHARD_GOVERNANCE_SQLITE_BATCH_MAX_BYTES": governance_batch_max_bytes,
        "SQL_LINK_SERVICE_SHARD_GOVERNANCE_TIMEOUT_SECONDS": governance_timeout,
        "INGEST_HOST_LOAD_SOFT_CAP": "6.0",
        "INGEST_HOST_LOAD_SLEEP_SECONDS": ingest_host_load_sleep,
        "INGEST_FLUSH_SLEEP_SECONDS": ingest_flush_sleep,
        "INGEST_FILE_SLEEP_SECONDS": ingest_file_sleep,
        "SQL_LINK_SERVICE_FLUIDITY_PRESERVE_SINGLE_WRITER": "1",
        "SQL_LINK_WRITER_BACKGROUND_POLICY": "0",
        "SQL_LINK_WRITER_NICE": "0",
    }
    if idle_backlog_cooling:
        env_overrides.update(_idle_sql_writer_cooling_overrides(throttle_profile))
        env_overrides["SQL_LINK_SERVICE_FLUIDITY_GOVERNOR_ACTIVE"] = "1"
        env_overrides["SQL_LINK_SERVICE_FLUIDITY_GOVERNOR_TIER"] = f"{tier}_idle_backlog_cooling"
        env_overrides["SQL_LINK_SERVICE_FLUIDITY_PRESERVE_SINGLE_WRITER"] = "0"
    return {
        "active": True,
        "overall_status": "guarded",
        "tier": tier,
        "reason": "storage_writer_heat_after_clean_backlog_is_being_retired"
        if idle_backlog_cooling
        else "storage_writer_heat_is_reducing_runtime_fluidity",
        "measurements": {
            "storage_writer_cpu_percent": round(storage_writer_cpu, 3),
            "host_saturation_score": round(saturation, 3),
            "compute_pressure_level": compute_pressure_level,
            "memory_pressure_level": memory_pressure_level,
            "fluidity_band": fluidity_band or "unknown",
            "fluidity_status": fluidity_status or "unknown",
            "fluidity_score": round(fluidity_score, 2),
            "storage_drain_active": bool(storage_drain_active),
            "storage_pressure_index": round(float(storage_pressure_index), 3),
            "storage_total_pending_lines": int(storage_total_pending_lines),
            "storage_pending_threshold": int(storage_pending_threshold),
            "storage_oldest_pending_age_seconds": round(float(storage_oldest_pending_age_seconds), 3),
            "idle_backlog_cooling": idle_backlog_cooling,
            "research_training_cpu_percent": round(research_cpu, 3),
            "support_cpu_percent": round(support_cpu, 3),
            "operator_observability_cpu_percent": round(operator_cpu, 3),
            "current_sql_lane_cap": int(current_cap),
            "recommended_sql_lane_cap": int(lane_cap),
        },
        "env_overrides": env_overrides,
        "stop_when": "clean-backlog SQL child workers are retired and storage writer CPU is below 85%."
        if idle_backlog_cooling
        else "storage writer CPU is below 85%, Mac fluidity is watch/ready, and host saturation is below the guarded band.",
        "policy": "self_heal_writer_heat_by_capping_sql_fanout_before_pausing_live_or_paper_lanes",
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
    computer_task = load_json(health_root / "computer_task_intelligence_latest.json")
    paper_ramp = load_json(health_root / "paper_400_ramp_latest.json") or load_json(health_root / "paper_400_ramp_control_latest.json")

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
    paper_execution_policy = _paper_ramp_execution_policy(paper_ramp)
    paper_trade_lock_active = _paper_trade_lock_active(project_root)
    release_live_read_only = bool(((live_runtime.get("release_contract") or {}).get("live_lane_should_be_read_only", False)))
    live_read_only = bool(
        release_live_read_only
        or (paper_trade_lock_active and bool(paper_execution_policy.get("paper_execution_allowed", False)))
    )
    memory_pressure_level = _memory_pressure_level(resource_guard, memory_efficiency)
    cotenant_contract = _cotenant_awareness_contract(memory_efficiency, computer_task)
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
    storage_sql_overlay = (
        storage_control.get("sql_ingestion_pending_overlay")
        if isinstance(storage_control.get("sql_ingestion_pending_overlay"), dict)
        else {}
    )
    raw_storage_core_pending_lines = _safe_int(storage_backpressure.get("core_pending_lines"), 0)
    raw_storage_total_pending_lines = _safe_int(storage_backpressure.get("total_pending_lines"), raw_storage_core_pending_lines)
    raw_storage_oldest_pending_age_seconds = _safe_float(storage_backpressure.get("oldest_pending_age_seconds"), 0.0)
    storage_managed_pressure_view = bool(
        storage_backpressure.get("managed_support_overlay_backlog", False)
        or storage_backpressure.get("overlay_pressure_clear", False)
        or (
            isinstance(storage_backpressure.get("managed_tiny_hot_tail"), dict)
            and bool(storage_backpressure.get("managed_tiny_hot_tail", {}).get("active", False))
        )
    )
    storage_core_pending_lines = (
        _safe_int(storage_backpressure.get("pressure_core_pending_lines"), raw_storage_core_pending_lines)
        if storage_managed_pressure_view
        else raw_storage_core_pending_lines
    )
    storage_total_pending_lines = (
        _safe_int(storage_backpressure.get("pressure_total_pending_lines"), raw_storage_total_pending_lines)
        if storage_managed_pressure_view
        else raw_storage_total_pending_lines
    )
    storage_pending_threshold = _safe_int(storage_backpressure.get("pending_lines_threshold"), 15000)
    storage_oldest_pending_age_seconds = (
        _safe_float(storage_backpressure.get("pressure_oldest_pending_age_seconds"), raw_storage_oldest_pending_age_seconds)
        if storage_managed_pressure_view
        else raw_storage_oldest_pending_age_seconds
    )
    storage_oldest_age_threshold_seconds = _safe_float(storage_backpressure.get("oldest_age_threshold_seconds"), 240.0)
    storage_fresh_overflow = bool(
        storage_pressure_index < 0.75
        and storage_total_pending_lines <= max(int(storage_pending_threshold * 1.25), storage_pending_threshold + 1)
        and storage_oldest_pending_age_seconds <= max(storage_oldest_age_threshold_seconds, 1.0)
    )
    storage_overlay_relief = _storage_overlay_relief_contract(
        storage_backpressure,
        storage_severity=storage_severity,
        storage_pressure_index=storage_pressure_index,
        sql_ingestion_overlay=storage_sql_overlay,
    )
    storage_overlay_capacity_relief = bool(storage_overlay_relief.get("active", False))
    if storage_overlay_capacity_relief:
        storage_pressure_index = min(storage_pressure_index, _safe_float(storage_overlay_relief.get("storage_pressure_index"), storage_pressure_index))
        if bool(storage_overlay_relief.get("direct_sql_overlay_clear", False)):
            storage_core_pending_lines = 0
            storage_total_pending_lines = 0
            storage_oldest_pending_age_seconds = 0.0
            storage_severity = "stable"
    storage_backlog_drain_status = str(((storage_control.get("storage") or {}).get("backlog_drain_status")) or "").strip().lower()
    storage_recommended_mode = str(storage_control.get("recommended_operating_mode") or "").strip().lower()
    storage_drain_active = bool(
        storage_backlog_drain_status in {"drain_active", "handoff_requested"}
        or storage_recommended_mode == "maintenance_drain_window"
        or storage_total_pending_lines > 0
    )
    sql_writer_coordination = _sql_writer_coordination(backpressure_fleet, storage_backpressure)
    if not storage_overlay_capacity_relief and (storage_pressure_index >= 1.0 or (
        storage_severity in {"high", "critical", "blocked"}
        and storage_core_pending_lines >= 15000
    )):
        throttle_profile = "protect_live"
    elif storage_pressure_index >= 0.5 and not storage_fresh_overflow and throttle_profile not in {"protect_live", "sustain"}:
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

    top_processes = snapshot.get("top_processes") if isinstance(snapshot.get("top_processes"), list) else []
    host_pressure_attribution = _host_pressure_attribution(domains, top_processes)
    pause_policy_storage_ready = bool(
        storage_pressure_index < 0.5
        or storage_fresh_overflow
        or storage_overlay_capacity_relief
    )
    paper_execution_policy = _paper_execution_pressure_pause_policy(
        paper_execution_policy,
        host_pressure_attribution,
        throttle_profile=throttle_profile,
        compute_pressure_level=compute_pressure_level,
        memory_pressure_level=memory_pressure_level,
        saturation_score=saturation_score,
        live_read_only=live_read_only,
        storage_ready_for_runtime_advisory=pause_policy_storage_ready,
        full_force_paper_required=_safe_int(registry_counts.get("active_bot_count"), 0) >= FULL_FORCE_PAPER_BOT_FLOOR,
    )
    nice_distribution: dict[str, int] = {}
    for row in top_processes:
        nice_key = str(_safe_int(row.get("nice"), 0))
        nice_distribution[nice_key] = nice_distribution.get(nice_key, 0) + 1
    backlog_relief = storage_control.get("backlog_relief_contract") if isinstance(storage_control.get("backlog_relief_contract"), dict) else {}
    p_core_backlog_contract = (
        backlog_relief.get("p_core_backlog_allocation_contract")
        if isinstance(backlog_relief.get("p_core_backlog_allocation_contract"), dict)
        else {}
    )
    if not p_core_backlog_contract and isinstance(backpressure_fleet.get("service_request"), dict):
        request_contract = backpressure_fleet["service_request"].get("p_core_backlog_allocation_contract")
        p_core_backlog_contract = request_contract if isinstance(request_contract, dict) else {}
    p_core_active_raw = p_core_backlog_contract.get("active", False)
    p_core_preprocess_budget = _safe_int(p_core_backlog_contract.get("preprocess_worker_budget"), 0)
    p_core_writer_lanes = _safe_int(p_core_backlog_contract.get("shard_link_writer_lanes"), p_core_preprocess_budget)
    p_core_burst_intelligence = (
        p_core_backlog_contract.get("p_core_burst_intelligence")
        if isinstance(p_core_backlog_contract.get("p_core_burst_intelligence"), dict)
        else {}
    )
    operator_pcore_force_open = bool(
        _env_flag("BACKLOG_PCORE_ALWAYS_ACTIVE", "0")
        or _safe_int(os.getenv("BACKLOG_PCORE_PREPROCESS_WORKERS_OVERRIDE"), 0) > 0
        or str(p_core_burst_intelligence.get("mode") or "").strip().lower() == "operator_override"
    )
    configured_writer_lane_cap = _safe_int(os.getenv("SQL_LINK_SERVICE_MAX_SHARD_WRITER_LANES"), 0)
    configured_smooth_cap_applied = False
    configured_smooth_cap_ignored_for_operator_override = False
    if configured_writer_lane_cap > 0 and not operator_pcore_force_open:
        if p_core_preprocess_budget > configured_writer_lane_cap:
            p_core_preprocess_budget = configured_writer_lane_cap
            configured_smooth_cap_applied = True
        if p_core_writer_lanes > configured_writer_lane_cap:
            p_core_writer_lanes = configured_writer_lane_cap
            configured_smooth_cap_applied = True
    elif configured_writer_lane_cap > 0 and operator_pcore_force_open:
        configured_smooth_cap_ignored_for_operator_override = bool(
            p_core_preprocess_budget > configured_writer_lane_cap
            or p_core_writer_lanes > configured_writer_lane_cap
        )
    p_core_runtime_feedback = {
        "active": p_core_active_raw is True or str(p_core_active_raw).strip().lower() in {"1", "true", "yes", "on"},
        "policy": str(p_core_backlog_contract.get("policy") or ""),
        "preprocess_worker_budget": p_core_preprocess_budget,
        "shard_link_writer_lanes": p_core_writer_lanes,
        "configured_max_shard_writer_lanes": configured_writer_lane_cap,
        "configured_smooth_cap_applied": configured_smooth_cap_applied,
        "configured_smooth_cap_ignored_for_operator_override": configured_smooth_cap_ignored_for_operator_override,
        "primary_merge_writer_count": _safe_int(p_core_backlog_contract.get("primary_merge_writer_count"), 1),
        "writer_lane_policy": str(p_core_backlog_contract.get("writer_lane_policy") or ""),
        "p_core_burst_intelligence": p_core_burst_intelligence,
        "single_writer_only": True,
        "avoid_background_taskpolicy": not _runtime_throttle_uses_background_taskpolicy(),
        "research_nice_target": _research_throttle_target_nice(),
        "top_process_nice_distribution": nice_distribution,
        "training_pcore_gate": (
            p_core_backlog_contract.get("training_pcore_gate")
            if isinstance(p_core_backlog_contract.get("training_pcore_gate"), dict)
            else {}
        ),
        "headroom_policy": "reserve_foreground_first_then_run_bounded_p_core_drain_work",
    }
    selected_writer_budget = max(p_core_writer_lanes, p_core_preprocess_budget)
    writer_worker_budget = selected_writer_budget if selected_writer_budget > 0 else None
    max_writer_lanes = writer_worker_budget
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
    paper_execution_pause_candidates = _paper_execution_pressure_candidates(
        top_processes,
        paper_execution_policy=paper_execution_policy,
        profile=throttle_profile,
        compute_pressure_level=compute_pressure_level,
        memory_pressure_level=memory_pressure_level,
    )
    paper_lane_guarded_for_autonomic_relief = bool(
        not bool(host_pressure_attribution.get("paper_execution_hot", False))
        or (
            bool(paper_execution_policy.get("paper_execution_allowed", False))
            and not bool(paper_execution_policy.get("pause_paper_execution", False))
            and bool(paper_execution_policy.get("armed", False))
            and bool(paper_execution_policy.get("ok", False))
            and bool(host_pressure_attribution.get("paper_hot_low_priority", False))
            and _safe_float(host_pressure_attribution.get("paper_execution_cpu_percent"), 0.0) <= 125.0
        )
    )
    autonomic_background_pressure_guarded = bool(
        throttle_profile == "protect_live"
        and overall_status == "blocked"
        and bool(live_read_only)
        and memory_pressure_level == "normal"
        and not thermal_warning_active
        and not performance_warning_active
        and (storage_pressure_index < 0.5 or storage_fresh_overflow or storage_overlay_capacity_relief)
        and _safe_float(host_pressure_attribution.get("protected_live_or_macro_cpu_percent"), 0.0) < 20.0
        and paper_lane_guarded_for_autonomic_relief
        and (
            bool(host_pressure_attribution.get("support_jobs_hot", False))
            or bool(host_pressure_attribution.get("research_training_hot", False))
            or bool(host_pressure_attribution.get("storage_writer_hot", False))
        )
        and not bool(host_pressure_attribution.get("protected_pressure_dominant", False))
    )
    protect_live_autonomic_reclassification = {
        "active": autonomic_background_pressure_guarded,
        "from_profile": "protect_live" if autonomic_background_pressure_guarded else throttle_profile,
        "to_profile": "sustain" if autonomic_background_pressure_guarded else throttle_profile,
        "reason": "stoppable_background_pressure_is_guarded_before_live_or_paper_degradation"
        if autonomic_background_pressure_guarded
        else "",
        "paper_lane_guarded_for_autonomic_relief": paper_lane_guarded_for_autonomic_relief,
        "policy": "reserve protect_live for thermal, memory, storage, or real execution-lane danger",
    }
    if autonomic_background_pressure_guarded:
        throttle_profile = "sustain"
        overall_status = "degraded"
    soft_cap_advisory_reclassification = _soft_cap_low_pressure_advisory(
        overall_status=overall_status,
        throttle_profile=throttle_profile,
        saturation_score=saturation_score,
        compute_pressure_level=compute_pressure_level,
        memory_pressure_level=memory_pressure_level,
        storage_pressure_index=storage_pressure_index,
        storage_fresh_overflow=storage_fresh_overflow,
        thermal_warning_active=thermal_warning_active,
        performance_warning_active=performance_warning_active,
        host_pressure_attribution=host_pressure_attribution,
        live_read_only=live_read_only,
        storage_severity=storage_severity,
        storage_core_pending_lines=storage_core_pending_lines,
        storage_total_pending_lines=storage_total_pending_lines,
        storage_pending_threshold=storage_pending_threshold,
        storage_oldest_pending_age_seconds=storage_oldest_pending_age_seconds,
        storage_oldest_age_threshold_seconds=storage_oldest_age_threshold_seconds,
        storage_overlay_relief=storage_overlay_relief,
        paper_execution_policy=paper_execution_policy,
        full_force_paper_required=_safe_int(registry_counts.get("active_bot_count"), 0) >= FULL_FORCE_PAPER_BOT_FLOOR,
    )
    if bool(soft_cap_advisory_reclassification.get("active", False)):
        overall_status = str(soft_cap_advisory_reclassification.get("to_status") or "advisory")
    paper_capacity_contract = _paper_capacity_contract(
        registry_counts,
        throttle_profile=throttle_profile,
        memory_pressure_level=memory_pressure_level,
        compute_pressure_level=compute_pressure_level,
        storage_pressure_index=storage_pressure_index,
        storage_total_pending_lines=storage_total_pending_lines,
        storage_backpressure=storage_backpressure,
        storage_severity=storage_severity,
        advisory_reclassification=soft_cap_advisory_reclassification,
    )
    upgrade_recommended = bool(overall_status in {"degraded", "blocked"} or support_trim_candidates or research_training_trim_candidates)
    runtime_saturation_governor_v2 = _runtime_saturation_governor_v2(
        saturation_score=saturation_score,
        throttle_profile=throttle_profile,
        compute_pressure_level=compute_pressure_level,
        memory_pressure_level=memory_pressure_level,
        storage_total_pending_lines=storage_total_pending_lines,
        storage_oldest_pending_age_seconds=storage_oldest_pending_age_seconds,
        support_trim_candidates=support_trim_candidates,
        research_training_trim_candidates=research_training_trim_candidates,
        paper_execution_policy=paper_execution_policy,
        paper_execution_pause_candidates=paper_execution_pause_candidates,
    )
    mac_fluidity_contract = _mac_fluidity_contract(
        overall_status=overall_status,
        throttle_profile=throttle_profile,
        saturation_score=saturation_score,
        compute_pressure_level=compute_pressure_level,
        memory_pressure_level=memory_pressure_level,
        storage_pressure_index=storage_pressure_index,
        storage_total_pending_lines=storage_total_pending_lines,
        storage_pending_threshold=storage_pending_threshold,
        storage_oldest_pending_age_seconds=storage_oldest_pending_age_seconds,
        storage_oldest_age_threshold_seconds=storage_oldest_age_threshold_seconds,
        host_pressure_attribution=host_pressure_attribution,
        cotenant_contract=cotenant_contract,
        runtime_saturation_governor=runtime_saturation_governor_v2,
        storage_overlay_relief=storage_overlay_relief,
    )
    runtime_storage_pressure = {
        "pressure_index": storage_pressure_index,
        "total_pending_lines": storage_total_pending_lines,
        "raw_total_pending_lines": raw_storage_total_pending_lines,
        "core_pending_lines": storage_core_pending_lines,
        "raw_core_pending_lines": raw_storage_core_pending_lines,
        "oldest_pending_age_seconds": storage_oldest_pending_age_seconds,
        "raw_oldest_pending_age_seconds": raw_storage_oldest_pending_age_seconds,
        "managed_pressure_view": storage_managed_pressure_view,
    }
    base_sql_overrides_for_fluidity = _sql_overrides_for_runtime_pressure(
        throttle_profile,
        storage_drain_active=storage_drain_active,
        storage_pressure=runtime_storage_pressure,
        sql_writer_coordination=sql_writer_coordination,
        writer_worker_budget=writer_worker_budget,
        max_writer_lanes=max_writer_lanes,
    )
    sql_writer_fluidity_contract = _sql_writer_fluidity_contract(
        throttle_profile=throttle_profile,
        compute_pressure_level=compute_pressure_level,
        memory_pressure_level=memory_pressure_level,
        saturation_score=saturation_score,
        storage_drain_active=storage_drain_active,
        storage_pressure_index=storage_pressure_index,
        storage_total_pending_lines=storage_total_pending_lines,
        storage_pending_threshold=storage_pending_threshold,
        storage_oldest_pending_age_seconds=storage_oldest_pending_age_seconds,
        host_pressure_attribution=host_pressure_attribution,
        mac_fluidity_contract=mac_fluidity_contract,
        current_sql_overrides=base_sql_overrides_for_fluidity,
    )
    upgrade_recommended = bool(upgrade_recommended or bool(sql_writer_fluidity_contract.get("active", False)))

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
            "downshift heavy shadow training loops while runtime pressure is active; keep live and data collectors protected"
            if research_training_trim_candidates
            else "",
            "pause the standalone paper execution consumer until paper_400_ramp is armed and clean"
            if paper_execution_pause_candidates
            else "",
            "pause training launches and cap collectors while host saturation is in the runtime-saturation governor guarded band"
            if str(runtime_saturation_governor_v2.get("saturation_band") or "") in {"guarded", "saturated", "protect"}
            else "",
            "treat Chrome, Codex, PyCharm, and other foreground apps as cotenants and downshift background support work instead of bouncing the stack"
            if interactive_cpu >= 60.0
            else "",
            "use memory-efficiency cotenant awareness to keep MLX, SQL, report, and collector jobs inside a foreground-app-safe profile"
            if bool(cotenant_contract.get("active", False))
            else "",
            "keep Mac fluidity in foreground-first mode while preserving the single SQL writer and guarded paper lane"
            if str(mac_fluidity_contract.get("fluidity_band") or "") in {"guarded_smooth", "strained", "protect"}
            else "",
            "cap SQL writer lane width through the fluidity governor until writer CPU and host saturation drop back under guarded thresholds"
            if bool(sql_writer_fluidity_contract.get("active", False))
            else "",
            *(
                host_pressure_attribution.get("recommended_actions")
                if isinstance(host_pressure_attribution.get("recommended_actions"), list)
                else []
            ),
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
            "keep collectors sampled while bounded SQL-overlay cleanup continues"
            if storage_overlay_capacity_relief
            else "force the collector floor into protect-live sampling while storage pressure is high"
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
        "ok": overall_status in {"ready", "advisory"},
        "overall_status": overall_status,
        "throttle_profile": throttle_profile,
        "host_saturation_score": saturation_score,
        "compute_pressure_level": compute_pressure_level,
        "memory_pressure_level": memory_pressure_level,
        "protect_live_autonomic_reclassification": protect_live_autonomic_reclassification,
        "soft_cap_advisory_reclassification": soft_cap_advisory_reclassification,
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
            "process_cpu_sampling": (
                snapshot.get("process_cpu_sampling")
                if isinstance(snapshot.get("process_cpu_sampling"), dict)
                else {}
            ),
            "storage_pressure": {
                "severity": storage_severity,
                "pressure_index": round(storage_pressure_index, 3),
                "core_pending_lines": storage_core_pending_lines,
                "total_pending_lines": storage_total_pending_lines,
                "raw_core_pending_lines": raw_storage_core_pending_lines,
                "raw_total_pending_lines": raw_storage_total_pending_lines,
                "oldest_pending_age_seconds": round(storage_oldest_pending_age_seconds, 3),
                "raw_oldest_pending_age_seconds": round(raw_storage_oldest_pending_age_seconds, 3),
                "fresh_overflow": storage_fresh_overflow,
                "managed_pressure_view": storage_managed_pressure_view,
                "overlay_capacity_relief": storage_overlay_capacity_relief,
                "overlay_relief_contract": storage_overlay_relief,
            },
            "top_process_nice_distribution": nice_distribution,
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
            "release_live_lane_should_be_read_only": release_live_read_only,
            "paper_trade_lock_active": paper_trade_lock_active,
            "effective_live_read_only_reason": (
                "release_contract"
                if release_live_read_only
                else "paper_trade_lock"
                if paper_trade_lock_active and bool(paper_execution_policy.get("paper_execution_allowed", False))
                else ""
            ),
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
        "p_core_runtime_feedback": p_core_runtime_feedback,
        "runtime_saturation_governor_v2": runtime_saturation_governor_v2,
        "mac_fluidity_contract": mac_fluidity_contract,
        "sql_writer_fluidity_contract": sql_writer_fluidity_contract,
        "paper_capacity_contract": paper_capacity_contract,
        "paper_execution_policy": paper_execution_policy,
        "cotenant_awareness_contract": cotenant_contract,
        "host_pressure_attribution": host_pressure_attribution,
        "mlx_intelligence_contract": mlx_intelligence_contract,
        "library_utilization_contract": library_utilization_contract,
        "throttle_domains": domains,
        "protected_workloads": {
            "categories": [name for name, row in domains.items() if bool(row.get("protected", False)) and _safe_float(row.get("cpu_percent"), 0.0) > 0.0],
            "top_processes": protected_processes,
        },
        "support_trim_candidates": support_trim_candidates,
        "research_training_trim_candidates": research_training_trim_candidates,
        "paper_execution_pause_candidates": paper_execution_pause_candidates,
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
                "runtime_saturation_governor_v2",
                "sql_writer_fluidity_governor",
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
    parser.add_argument("--candidate-registry", default=str(DEFAULT_CANDIDATE_REGISTRY_PATH))
    parser.add_argument("--source-write-guard", default=str(DEFAULT_SOURCE_WRITE_GUARD_PATH))
    parser.add_argument(
        "--allow-source-registry-write",
        action="store_true",
        default=os.getenv("RUNTIME_THROTTLE_ALLOW_SOURCE_REGISTRY_WRITE", "0").strip() == "1",
        help="Allow this intentional operator command to update the tracked master_bot_registry.json source file.",
    )
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
        apply_result = apply_runtime_guard(
            project_root,
            payload,
            override_path=Path(args.override_file).expanduser(),
            registry_path=Path(args.registry).expanduser(),
            candidate_registry_path=Path(args.candidate_registry).expanduser(),
            source_write_guard_path=Path(args.source_write_guard).expanduser(),
            allow_source_registry_write=args.allow_source_registry_write,
            max_renice_processes=args.max_renice_processes,
        )
        payload = build_payload(project_root)
        payload["apply_result"] = apply_result
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
    return 0 if payload.get("overall_status") in {"ready", "advisory", "degraded"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
