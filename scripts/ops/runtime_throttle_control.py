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
TOP_PROCESS_COUNT = 12
APPLY_CPU_THRESHOLD = 12.0


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
    ("report-bundle-pdf-open", "support_maintenance", "throttle_first", True),
    ("scripts/build_one_numbers_report.py", "support_maintenance", "throttle_first", True),
    ("scripts/collect_market_crypto_correlation_context.py", "support_maintenance", "throttle_first", True),
    ("scripts/collect_market_correlation_context.py", "support_maintenance", "throttle_first", True),
    ("scripts/collect_crypto_market_context.py", "support_maintenance", "throttle_first", True),
    ("project_timeline_report.py", "support_maintenance", "throttle_first", True),
    ("scripts/ops/live_feed_tail.sh", "operator_observability", "operator_visible", False),
    ("live_feed source=", "operator_observability", "operator_visible", False),
    ("tail -n 80 -F", "operator_observability", "operator_visible", False),
    ("tail -n 120 -F", "operator_observability", "operator_visible", False),
    ("sql_queue_retention.py", "support_maintenance", "throttle_first", True),
    ("data_retention_policy.py", "support_maintenance", "throttle_first", True),
    ("scripts/ops/storage_maintenance_lane.py", "support_maintenance", "throttle_first", True),
    ("scripts/ops/sql_link_shard_manager.py", "support_maintenance", "throttle_first", True),
    ("scripts/ops/sql_link_writer_service.py", "support_maintenance", "throttle_first", True),
    ("scripts/link_jsonl_to_sql.py", "support_maintenance", "throttle_first", True),
    ("Google Chrome Helper --headless", "support_maintenance", "throttle_first", True),
    ("Google Chrome", "interactive_cotenant", "external_cotenant", False),
    ("Codex", "interactive_cotenant", "external_cotenant", False),
    ("PyCharm", "interactive_cotenant", "external_cotenant", False),
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
    swap_used_gb = _safe_float(resource_guard.get("swap_used_gb"), 0.0)
    if pressure_state in {"red", "critical"} or pressure_kind == "throttled" or efficiency_status == "blocked" or swap_used_gb >= 20.0:
        return "high"
    if pressure_state in {"yellow", "warn"} or efficiency_status in {"degraded", "needs_work"} or swap_used_gb >= 8.0:
        return "elevated"
    return "normal"


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


def _runtime_env_overrides(throttle_profile: str, memory_pressure_level: str, compute_pressure_level: str) -> dict[str, str]:
    if throttle_profile == "protect_live" or memory_pressure_level == "high" or compute_pressure_level == "high":
        return {
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
            "DATA_COLLECTION_RESOURCE_GUARD_MODE": "protect_live",
            "DATA_COLLECTION_RESOURCE_SAMPLE_RATE": "0.15",
            "DATA_COLLECTION_RESOURCE_CAPTURE_MODE": "thin_sample",
            "OPS_SUPPORT_JOB_NICE": "15",
            "OPS_SUPPORT_JOBS_BACKGROUND_POLICY": "1",
        }
    if throttle_profile == "sustain":
        return {
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
            "DATA_COLLECTION_RESOURCE_GUARD_MODE": "sustain",
            "DATA_COLLECTION_RESOURCE_SAMPLE_RATE": "0.30",
            "DATA_COLLECTION_RESOURCE_CAPTURE_MODE": "sampled",
            "OPS_SUPPORT_JOB_NICE": "10",
            "OPS_SUPPORT_JOBS_BACKGROUND_POLICY": "1",
        }
    return {
        "BOT_RUNTIME_RESOURCE_GUARD_PROFILE": throttle_profile,
        "DATA_COLLECTION_RESOURCE_GUARD_MODE": throttle_profile,
        "DATA_COLLECTION_RESOURCE_SAMPLE_RATE": "0.50" if throttle_profile == "soft_cap" else "1.0",
        "DATA_COLLECTION_RESOURCE_CAPTURE_MODE": "sampled" if throttle_profile == "soft_cap" else "full",
        "OPS_SUPPORT_JOB_NICE": "5" if throttle_profile == "soft_cap" else "0",
        "OPS_SUPPORT_JOBS_BACKGROUND_POLICY": "0",
    }


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
    changed_count = 0
    for row in rows:
        lifecycle = str(row.get("lifecycle_state") or "").strip().lower()
        if not bool(row.get("active", False)) or lifecycle != "data_collection_only":
            continue
        base_slo = _safe_int(row.get("freshness_slo_seconds"), 900)
        updates = {
            "data_collection_compute_guard_mode": policy["compute_guard_mode"],
            "data_collection_resource_guard_reason": policy["reason"],
            "data_collection_capture_mode": policy["capture_mode"],
            "data_collection_sample_rate": policy["sample_rate"],
            "data_collection_max_daily_mb": policy["max_daily_mb"],
            "freshness_slo_seconds": max(base_slo, _safe_int(policy["freshness_slo_minimum_seconds"], base_slo)),
        }
        row_changed = False
        for key, value in updates.items():
            if row.get(key) != value:
                row[key] = value
                row_changed = True
        if row_changed:
            changed_count += 1
    if changed_count:
        registry["updated_at_utc"] = iso_now()
        path.write_text(json.dumps(registry, ensure_ascii=True, indent=2), encoding="utf-8")
    return {
        "applied": bool(changed_count),
        "changed_count": changed_count,
        "collector_count": sum(1 for row in rows if bool(row.get("active", False)) and str(row.get("lifecycle_state") or "").strip().lower() == "data_collection_only"),
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
    env_overrides = _runtime_env_overrides(
        profile,
        str(payload.get("memory_pressure_level") or "normal"),
        str(payload.get("compute_pressure_level") or "normal"),
    )
    support_candidates = payload.get("support_trim_candidates") if isinstance(payload.get("support_trim_candidates"), list) else []
    throttle_candidates = list(support_candidates)
    if profile == "protect_live":
        top_processes = payload.get("top_processes") if isinstance(payload.get("top_processes"), list) else []
        throttle_candidates.extend(
            row
            for row in top_processes
            if str(row.get("category") or "") == "research_training"
            and _safe_float(row.get("cpu_percent"), 0.0) >= 50.0
        )
    return {
        "applied": True,
        "override_path": str(override_path),
        "override_changed": _write_env_override(override_path, env_overrides, profile=profile),
        "env_override_count": len(env_overrides),
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
    apple_profile = load_json(health_root / "apple_silicon_profile_latest.json")
    portable_brain = load_json(health_root / "portable_brain_contract_latest.json")

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
    overall_status = _overall_status(throttle_profile)

    top_processes = snapshot.get("top_processes") if isinstance(snapshot.get("top_processes"), list) else []
    protected_processes = [
        row for row in top_processes if str(row.get("priority_tier") or "") in {"protected", "protected_if_live"}
    ][:5]
    support_trim_candidates = [
        row for row in top_processes if bool(row.get("throttle_candidate", False))
    ][:5]
    upgrade_recommended = bool(overall_status in {"degraded", "blocked"} or support_trim_candidates)

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
            "treat Chrome, Codex, PyCharm, and other foreground apps as cotenants and downshift background support work instead of bouncing the stack"
            if interactive_cpu >= 60.0
            else "",
            "./scripts/ops/opsctl.sh memory-efficiency apply --json"
            if memory_pressure_level in {"elevated", "high"} and status_rank(str(memory_efficiency.get("overall_status") or "")) >= status_rank("degraded")
            else "",
            "keep the live runtime on read-only release posture until the host saturation score drops back into the soft-cap band"
            if live_read_only and overall_status in {"degraded", "blocked"}
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
        "throttle_domains": domains,
        "protected_workloads": {
            "categories": [name for name, row in domains.items() if bool(row.get("protected", False)) and _safe_float(row.get("cpu_percent"), 0.0) > 0.0],
            "top_processes": protected_processes,
        },
        "support_trim_candidates": support_trim_candidates,
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
