#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shlex
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
    from scripts.ops.long_runtime_common import PROJECT_ROOT, iso_now, load_json, ordered_unique, write_payload
else:
    from .. import resource_guard as resource_src
    from .long_runtime_common import PROJECT_ROOT, iso_now, load_json, ordered_unique, write_payload


DEFAULT_OUT_PATH = PROJECT_ROOT / "governance" / "health" / "computer_task_intelligence_latest.json"
DEFAULT_OVERRIDE_PATH = PROJECT_ROOT / "config" / ".env.computer_task_override"

BACKGROUND_PROCESS_CLASSES: tuple[dict[str, Any], ...] = (
    {
        "class_id": "single_writer",
        "target_nice": 12,
        "patterns": ("scripts/ops/sql_link_shard_manager.py", "scripts/link_jsonl_to_sql.py"),
    },
    {
        "class_id": "market_runtime",
        "target_nice": 14,
        "patterns": ("scripts/run_shadow_training_loop.py", "scripts/ops/live_feed_tail.sh"),
    },
    {
        "class_id": "drainer_accelerator",
        "target_nice": 12,
        "patterns": ("scripts/ops/backpressure_super_drainer.py", "scripts/ops/backpressure_drainer_fleet.py"),
    },
    {
        "class_id": "macro_media_capture",
        "target_nice": 14,
        "patterns": ("scripts/ops/live_macro_auto_watch.py", "scripts/ops/live_macro_media_ingest.py", "yt-dlp", "ffmpeg"),
    },
    {
        "class_id": "research_training",
        "target_nice": 16,
        "patterns": ("scripts/ops/quant_model_control.py", "scripts/quant_models/", "weekly_retrain.py", "retrain_orchestrator.py"),
    },
)


def _safe_int(raw: Any, default: int = 0) -> int:
    try:
        return int(float(raw))
    except Exception:
        return int(default)


def _safe_float(raw: Any, default: float = 0.0) -> float:
    try:
        return float(raw)
    except Exception:
        return float(default)


def _as_dict(raw: Any) -> dict[str, Any]:
    return raw if isinstance(raw, dict) else {}


def _as_list(raw: Any) -> list[Any]:
    return raw if isinstance(raw, list) else []


def _grade(score: float) -> str:
    if score >= 90.0:
        return "A"
    if score >= 75.0:
        return "B"
    if score >= 60.0:
        return "C"
    if score >= 45.0:
        return "D"
    return "F"


def _pressure_score(value: float, *, a: float, b: float, c: float, d: float) -> float:
    value = max(float(value), 0.0)
    if value <= a:
        return 96.0
    if value <= b:
        return 84.0
    if value <= c:
        return 68.0
    if value <= d:
        return 52.0
    return 35.0


def _is_off_hours(now: datetime | None = None) -> bool:
    current = now or datetime.now().astimezone()
    if current.weekday() >= 5:
        return True
    minutes = current.hour * 60 + current.minute
    return bool(minutes >= 20 * 60 or minutes < 7 * 60)


def _parse_timestamp(raw: Any) -> datetime | None:
    text = str(raw or "").strip().replace("Z", "+00:00")
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text)
    except Exception:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _payload_age_seconds(payload: dict[str, Any], *, now: datetime) -> float | None:
    parsed = _parse_timestamp(payload.get("timestamp_utc"))
    if parsed is None:
        return None
    return max((now - parsed).total_seconds(), 0.0)


def _memory_context_has_apps(memory_efficiency: dict[str, Any]) -> bool:
    creative = _as_dict(memory_efficiency.get("creative_session"))
    cotenant = _as_dict(memory_efficiency.get("cotenant_awareness"))
    co_running = _as_dict(memory_efficiency.get("co_running_session"))
    creative_kind = str(creative.get("kind") or "").strip().lower()
    creative_level = str(creative.get("level") or "").strip().lower()
    if bool(creative.get("active", False)) or creative_kind not in {"", "none"} or creative_level not in {"", "none"}:
        return True
    if _as_list(creative.get("apps")) or _as_list(cotenant.get("open_apps")) or _as_list(co_running.get("apps")):
        return True
    if _as_list(cotenant.get("co_running_classes")) or _as_list(cotenant.get("classes")) or _as_list(co_running.get("classes")):
        return True
    return _safe_float(cotenant.get("co_running_cpu_sum"), 0.0) > 1.0 or _safe_float(co_running.get("cpu_sum"), 0.0) > 1.0


def _resource_context_is_clear(resource_guard: dict[str, Any]) -> bool:
    creative_kind = str(resource_guard.get("creative_session_kind") or "").strip().lower()
    creative_level = str(resource_guard.get("creative_session_level") or "").strip().lower()
    co_running_level = str(resource_guard.get("co_running_session_level") or "").strip().lower()
    return bool(
        creative_kind in {"", "none"}
        and creative_level in {"", "none"}
        and co_running_level in {"", "none"}
        and not _as_list(resource_guard.get("creative_apps"))
        and not _as_list(resource_guard.get("co_running_apps"))
        and not _as_list(resource_guard.get("co_running_classes"))
        and _safe_float(resource_guard.get("co_running_cpu_sum"), 0.0) <= 1.0
    )


def _stale_process_context_infrabot(memory_efficiency: dict[str, Any], resource_guard: dict[str, Any]) -> dict[str, Any]:
    now = datetime.now(timezone.utc)
    max_age = max(_safe_float(os.getenv("COMPUTER_TASK_CONTEXT_MAX_AGE_SECONDS"), 180.0), 30.0)
    memory_age = _payload_age_seconds(memory_efficiency, now=now)
    resource_age = _payload_age_seconds(resource_guard, now=now)
    memory_has_apps = _memory_context_has_apps(memory_efficiency)
    resource_clear = _resource_context_is_clear(resource_guard)
    memory_ts = _parse_timestamp(memory_efficiency.get("timestamp_utc"))
    resource_ts = _parse_timestamp(resource_guard.get("timestamp_utc"))
    resource_newer = bool(memory_ts is not None and resource_ts is not None and resource_ts >= memory_ts)
    stale_memory = bool(memory_age is not None and memory_age > max_age)
    contradicted_by_live_probe = bool(memory_has_apps and resource_clear and resource_newer)
    ignored = bool(memory_has_apps and resource_clear and (stale_memory or contradicted_by_live_probe))
    reasons = ordered_unique(
        [
            "memory_efficiency_context_stale" if stale_memory and memory_has_apps else "",
            "fresh_resource_guard_clears_stale_app_context" if contradicted_by_live_probe else "",
            "strict_bundle_process_scan_active",
        ]
    )
    return {
        "active": True,
        "status": "cleared_stale_context" if ignored else "monitoring",
        "ignored_memory_efficiency_app_context": ignored,
        "max_context_age_seconds": round(float(max_age), 3),
        "memory_efficiency_age_seconds": round(float(memory_age), 3) if memory_age is not None else None,
        "resource_guard_age_seconds": round(float(resource_age), 3) if resource_age is not None else None,
        "memory_context_has_apps": memory_has_apps,
        "resource_context_is_clear": resource_clear,
        "resource_guard_newer_than_memory_efficiency": resource_newer,
        "reasons": reasons,
        "contract": {
            "does_not_close_user_apps": True,
            "does_not_kill_user_processes": True,
            "clears_context_by_trusting_fresh_resource_guard": True,
            "strict_app_detection": "real_app_bundle_processes_only",
            "do_not_touch_volumes": ["/Volumes/VIDEO"],
        },
    }


def _session_context(memory_efficiency: dict[str, Any], resource_guard: dict[str, Any]) -> dict[str, Any]:
    infrabot = _stale_process_context_infrabot(memory_efficiency, resource_guard)
    if bool(infrabot.get("ignored_memory_efficiency_app_context", False)):
        creative: dict[str, Any] = {}
        cotenant: dict[str, Any] = {}
        co_running: dict[str, Any] = {}
    else:
        creative = _as_dict(memory_efficiency.get("creative_session"))
        cotenant = _as_dict(memory_efficiency.get("cotenant_awareness"))
        co_running = _as_dict(memory_efficiency.get("co_running_session"))

    creative_kind = str(
        creative.get("kind")
        or resource_guard.get("creative_session_kind")
        or "none"
    ).strip().lower()
    creative_level = str(
        creative.get("level")
        or resource_guard.get("creative_session_level")
        or "none"
    ).strip().lower()
    creative_apps = [
        str(item)
        for item in (
            _as_list(creative.get("apps"))
            or _as_list(resource_guard.get("creative_apps"))
        )
        if str(item).strip()
    ]

    open_apps = ordered_unique(
        [
            *[str(item) for item in _as_list(cotenant.get("open_apps"))],
            *[str(item) for item in _as_list(co_running.get("apps"))],
            *creative_apps,
        ]
    )
    classes = ordered_unique(
        [
            *[str(item) for item in _as_list(cotenant.get("co_running_classes"))],
            *[str(item) for item in _as_list(cotenant.get("classes"))],
            *[str(item) for item in _as_list(co_running.get("classes"))],
        ]
    )
    co_running_level = str(
        cotenant.get("co_running_level")
        or co_running.get("level")
        or "none"
    ).strip().lower()
    cpu_sum = max(
        _safe_float(cotenant.get("co_running_cpu_sum"), 0.0),
        _safe_float(co_running.get("cpu_sum"), 0.0),
        _safe_float(resource_guard.get("co_running_cpu_sum"), 0.0),
    )
    creative_active = bool(creative.get("active", False)) or creative_level not in {"", "none"}
    cotenant_active = bool(cotenant.get("active", False) or co_running.get("active", False) or classes)
    return {
        "creative_kind": creative_kind,
        "creative_level": creative_level,
        "creative_apps": creative_apps,
        "creative_active": creative_active,
        "cotenant_active": cotenant_active,
        "co_running_level": co_running_level,
        "co_running_classes": classes,
        "open_apps": open_apps[:16],
        "co_running_cpu_sum": round(cpu_sum, 3),
        "process_context_infrabot": infrabot,
    }


def _task_profile(session: dict[str, Any], storage: dict[str, Any], drainer: dict[str, Any]) -> dict[str, Any]:
    kind = str(session.get("creative_kind") or "none")
    classes = {str(item).lower() for item in _as_list(session.get("co_running_classes"))}
    open_apps = {str(item).lower() for item in _as_list(session.get("open_apps"))}
    bp = _as_dict(storage.get("backpressure"))
    scorecard = _as_dict(drainer.get("backlog_section_scorecard"))
    total_pending = _safe_int(bp.get("total_pending_lines"), 0)
    core_pending = _safe_int(bp.get("core_pending_lines"), 0)
    threshold = max(_safe_int(bp.get("pending_lines_threshold"), 15000), 1)
    backlog_grade = str(scorecard.get("overall_grade") or "").upper()

    active_tasks: list[str] = []
    if kind.startswith("logic_pro"):
        active_tasks.append("audio_production")
    if kind.startswith("final_cut"):
        active_tasks.append("video_editing")
    if kind.startswith("music_playback") or any(app in {"music", "itunes"} for app in open_apps):
        active_tasks.append("music_playback")
    if "virtualization" in classes:
        active_tasks.append("virtualization")
    if "developer" in classes or any(app in {"pycharm", "cursor", "code", "xcode", "codex"} for app in open_apps):
        active_tasks.append("developer_work")
    if "browser" in classes or any(app in {"safari", "chrome", "google chrome", "arc", "firefox"} for app in open_apps):
        active_tasks.append("browser_work")
    if total_pending > threshold or core_pending > 15000 or backlog_grade not in {"", "A"}:
        active_tasks.append("backlog_drain")
    active_tasks = ordered_unique(active_tasks)

    priority = [
        "audio_production",
        "video_editing",
        "music_playback",
        "virtualization",
        "developer_work",
        "browser_work",
        "backlog_drain",
    ]
    primary = next((task for task in priority if task in active_tasks), "market_collection")
    if not active_tasks and _is_off_hours() and backlog_grade in {"", "A", "B"} and total_pending <= threshold:
        primary = "overnight_research"
        active_tasks.append("overnight_research")
    return {
        "primary_task": primary,
        "active_tasks": active_tasks,
        "backlog_grade": backlog_grade,
        "core_pending_lines": core_pending,
        "total_pending_lines": total_pending,
        "pending_lines_threshold": threshold,
    }


def _budget_for_task(primary_task: str) -> dict[str, Any]:
    budgets = {
        "audio_production": ("daily_driver", 35, "0.25", "1", "1", False, False, False, "15", "5"),
        "video_editing": ("daily_driver", 40, "0.30", "1", "1", False, False, False, "15", "5"),
        "music_playback": ("daily_driver", 55, "0.40", "2", "1", False, False, False, "25", "6"),
        "virtualization": ("daily_driver", 50, "0.35", "1", "1", False, False, False, "20", "6"),
        "developer_work": ("daily_driver", 55, "0.45", "2", "1", False, False, False, "30", "8"),
        "browser_work": ("daily_driver", 60, "0.55", "2", "1", False, False, True, "40", "10"),
        "backlog_drain": ("trading_focus", 70, "0.68", "3", "1", False, False, True, "60", "12"),
        "overnight_research": ("overnight_heavy", 85, "0.90", "4", "2", True, True, True, "80", "18"),
        "market_collection": ("trading_focus", 75, "0.68", "3", "1", False, False, True, "60", "12"),
    }
    mode, max_host, ratio, async_workers, quant_workers, training, heavy, reports, feed_lines, feed_files = budgets.get(primary_task, budgets["market_collection"])
    return {
        "requested_operator_mode": mode,
        "max_host_saturation_for_normal_use": max_host,
        "collector_intake_ratio": ratio,
        "async_pipeline_workers": async_workers,
        "coinbase_snapshot_workers": "1" if mode == "daily_driver" else "2",
        "quant_model_workers": quant_workers,
        "training_allowed": training,
        "heavy_collectors_allowed": heavy,
        "report_refresh_allowed": reports,
        "live_feed_lines": feed_lines,
        "live_feed_follow_files": feed_files,
    }


def _section(section_id: str, label: str, score: float, issue: str, action: str, evidence: list[str]) -> dict[str, Any]:
    clean = round(max(min(float(score), 100.0), 0.0), 1)
    return {
        "section_id": section_id,
        "label": label,
        "grade": _grade(clean),
        "score": clean,
        "primary_issue": issue,
        "recommended_next_action": action,
        "evidence": ordered_unique(evidence),
    }


def _scorecard(
    *,
    task: dict[str, Any],
    budget: dict[str, Any],
    session: dict[str, Any],
    runtime: dict[str, Any],
    memory_efficiency: dict[str, Any],
    resource_guard: dict[str, Any],
    storage: dict[str, Any],
    drainer: dict[str, Any],
    mode_switchboard: dict[str, Any],
    process_watchdog: dict[str, Any],
) -> dict[str, Any]:
    host = _safe_float(runtime.get("host_saturation_score"), 0.0)
    max_host = _safe_float(budget.get("max_host_saturation_for_normal_use"), 60.0)
    foreground_score = _pressure_score(
        host,
        a=max(max_host - 10.0, 25.0),
        b=max_host,
        c=max_host + 15.0,
        d=max_host + 25.0,
    )
    if str(session.get("co_running_level") or "") == "heavy_competition":
        foreground_score = min(foreground_score, 84.0)

    mem_level = str(runtime.get("memory_pressure_level") or resource_guard.get("memory_pressure_state") or "").lower()
    mem_state = str(resource_guard.get("memory_pressure_state") or _as_dict(memory_efficiency.get("memory_snapshot")).get("memory_pressure_state") or "").lower()
    memory_score = 96.0
    if mem_level in {"critical", "high", "red"} or mem_state in {"critical", "red"}:
        memory_score = 45.0
    elif mem_level in {"elevated", "yellow", "orange"} or mem_state in {"yellow", "orange"}:
        memory_score = 75.0

    bp = _as_dict(storage.get("backpressure"))
    line_estimation = _as_dict(_as_dict(bp.get("raw_live")).get("line_estimation")) or _as_dict(bp.get("line_estimation"))
    core = _safe_int(bp.get("core_pending_lines"), 0)
    total = _safe_int(bp.get("total_pending_lines"), 0)
    oldest = _safe_float(bp.get("oldest_pending_age_seconds"), 0.0)
    sparse = _safe_int(line_estimation.get("sparse_large_line_pending_lines"), 0)
    sparse_bytes = _safe_int(line_estimation.get("sparse_large_line_pending_bytes"), 0)
    backlog_score = 96.0
    if not (core <= 5000 and total <= 10000 and sparse <= 250 and oldest <= 3600):
        if core <= 15000 and sparse <= 700:
            backlog_score = 84.0
        elif core <= 30000 and sparse <= 1200:
            backlog_score = 68.0
        elif core <= 40000:
            backlog_score = 52.0
        else:
            backlog_score = 35.0

    selected_mode = str(_as_dict(mode_switchboard.get("operator_mode")).get("selected_mode") or "").lower()
    requested_mode = str(budget.get("requested_operator_mode") or "").lower()
    task_fit_score = 96.0 if selected_mode in {"", requested_mode} else 62.0
    if bool(session.get("creative_active") or session.get("cotenant_active")) and requested_mode != "daily_driver":
        task_fit_score = min(task_fit_score, 55.0)

    rows = [row for row in _as_list(process_watchdog.get("status")) if isinstance(row, dict)]
    down = [row for row in rows if _safe_int(row.get("running"), 1) <= 0 and str(row.get("action") or "") not in {"suppressed", "none"}]
    alert_count = len(_as_list(process_watchdog.get("alerts")))
    restart_score = 96.0 if not down and alert_count == 0 else 68.0 if alert_count <= 2 else 45.0

    sections = [
        _section(
            "foreground_responsiveness",
            "Foreground responsiveness",
            foreground_score,
            "foreground apps need lower host saturation" if foreground_score < 90 else "foreground responsiveness is inside budget",
            "keep the requested operator mode and worker caps active" if foreground_score < 90 else "maintain current caps",
            [f"host_saturation_score={host}", f"max_host_saturation_for_task={max_host}", f"co_running_level={session.get('co_running_level', '')}"],
        ),
        _section(
            "memory_headroom",
            "Memory headroom",
            memory_score,
            "memory pressure would affect normal Mac use" if memory_score < 90 else "memory pressure is clear",
            "keep training and heavy research paused until memory stays green" if memory_score < 90 else "memory headroom is acceptable",
            [f"memory_pressure_level={mem_level}", f"memory_pressure_state={mem_state}", f"memory_efficiency_status={memory_efficiency.get('overall_status', '')}"],
        ),
        _section(
            "backlog_interference",
            "Backlog interference",
            backlog_score,
            "backlog can still compete with normal use" if backlog_score < 90 else "backlog is unlikely to disturb normal use",
            "keep concentrated core drain active without widening or training" if backlog_score < 90 else "allow normal task-aware mode switching",
            [f"core_pending_lines={core}", f"total_pending_lines={total}", f"sparse_pending_lines={sparse}", f"sparse_pending_bytes={sparse_bytes}", f"oldest_pending_age_seconds={round(oldest, 3)}"],
        ),
        _section(
            "task_fit_contract",
            "Task fit contract",
            task_fit_score,
            "operator mode does not match the current computer task" if task_fit_score < 90 else "operator mode matches the current computer task",
            "apply computer-task intelligence and mode-switchboard together" if task_fit_score < 90 else "keep current task contract",
            [f"primary_task={task.get('primary_task', '')}", f"requested_operator_mode={requested_mode}", f"selected_operator_mode={selected_mode}"],
        ),
        _section(
            "restart_safety",
            "Restart safety",
            restart_score,
            "watchdog/process alerts can disrupt normal use" if restart_score < 90 else "restart posture is quiet",
            "leave heavy sleeve restarts suppressed while daily-driver or backlog guard is active" if restart_score < 90 else "no restart action needed",
            [f"alert_count={alert_count}", f"down_process_count={len(down)}"],
        ),
    ]
    weights = {
        "foreground_responsiveness": 0.28,
        "memory_headroom": 0.20,
        "backlog_interference": 0.25,
        "task_fit_contract": 0.17,
        "restart_safety": 0.10,
    }
    total_score = sum(row["score"] * weights[row["section_id"]] for row in sections)
    worst = sorted(sections, key=lambda row: row["score"])[:3]
    return {
        "overall_score": round(total_score, 1),
        "overall_grade": _grade(total_score),
        "sections": sections,
        "worst_sections": worst,
    }


def _a_grade_need(row: dict[str, Any]) -> dict[str, Any]:
    section_id = str(row.get("section_id") or "")
    criteria_by_section = {
        "foreground_responsiveness": [
            "host_saturation_score <= task max minus 10",
            "co_running_level is not heavy_competition",
            "foreground app contract remains active while user apps are open",
        ],
        "memory_headroom": [
            "memory_pressure_level remains normal",
            "memory_pressure_state remains green or none",
            "training/heavy research stay paused while creative apps are active",
        ],
        "backlog_interference": [
            "core_pending_lines <= 5000",
            "total_pending_lines <= 10000",
            "sparse_pending_lines <= 250",
            "oldest_pending_age_seconds <= 3600",
        ],
        "task_fit_contract": [
            "computer task requested_operator_mode matches mode-switchboard selected_mode",
            "daily_driver is active for foreground apps",
            "overnight_heavy is only active during clean off-hours",
        ],
        "restart_safety": [
            "process_watchdog alert_count == 0",
            "no required sleeve restart storm is active",
            "heavy sleeve restarts remain suppressed during daily_driver/backlog guard",
        ],
    }
    return {
        "section_id": section_id,
        "current_grade": str(row.get("grade") or ""),
        "current_score": _safe_float(row.get("score"), 0.0),
        "target_grade": "A",
        "target_score": 90.0,
        "score_gap": round(max(90.0 - _safe_float(row.get("score"), 0.0), 0.0), 1),
        "what_it_needs": str(row.get("recommended_next_action") or ""),
        "a_grade_exit_criteria": criteria_by_section.get(section_id, ["section_score >= 90"]),
        "evidence": [str(item) for item in _as_list(row.get("evidence")) if str(item).strip()],
    }


def _section_score(scorecard: dict[str, Any], section_id: str, default: float = 100.0) -> float:
    for row in _as_list(scorecard.get("sections")):
        if isinstance(row, dict) and str(row.get("section_id") or "") == section_id:
            return _safe_float(row.get("score"), default)
    return float(default)


def _tighten_ratio(raw: Any, cap: float) -> str:
    ratio = min(max(_safe_float(raw, cap), 0.05), float(cap))
    return f"{ratio:.2f}"


def _score_gap(scorecard: dict[str, Any], section_id: str) -> float:
    return max(90.0 - _section_score(scorecard, section_id), 0.0)


def _preemption_level(friction_index: float) -> str:
    if friction_index >= 65.0:
        return "relief"
    if friction_index >= 45.0:
        return "deep_protect"
    if friction_index >= 25.0:
        return "protect"
    if friction_index >= 10.0:
        return "coordinate"
    return "observe"


def _protected_classes(task: dict[str, Any], session: dict[str, Any]) -> list[str]:
    active = {str(item).lower() for item in _as_list(task.get("active_tasks"))}
    classes: list[str] = []
    if "audio_production" in active:
        classes.append("audio_production")
    if "video_editing" in active:
        classes.append("video_editing")
    if "music_playback" in active:
        classes.append("music_playback")
    if "virtualization" in active:
        classes.append("virtualization")
    if "developer_work" in active:
        classes.append("developer_work")
    if "browser_work" in active:
        classes.append("browser_work")
    if bool(session.get("creative_active", False)):
        classes.append("creative_session")
    return ordered_unique(classes)


def _computer_unison_contract(
    *,
    task: dict[str, Any],
    session: dict[str, Any],
    scorecard: dict[str, Any],
    runtime: dict[str, Any],
    storage: dict[str, Any],
    process_policy: dict[str, Any],
    env: dict[str, str],
) -> dict[str, Any]:
    foreground_gap = _score_gap(scorecard, "foreground_responsiveness")
    backlog_gap = _score_gap(scorecard, "backlog_interference")
    memory_gap = _score_gap(scorecard, "memory_headroom")
    restart_gap = _score_gap(scorecard, "restart_safety")
    host = _safe_float(runtime.get("host_saturation_score"), 0.0)
    compute = str(runtime.get("compute_pressure_level") or "").strip().lower()
    pressure_bonus = 12.0 if compute in {"high", "critical"} else 6.0 if compute == "elevated" or host >= 65.0 else 0.0
    friction_index = min(
        100.0,
        (foreground_gap * 0.36) + (backlog_gap * 0.27) + (memory_gap * 0.18) + (restart_gap * 0.09) + pressure_bonus,
    )
    level = _preemption_level(friction_index)
    primary = str(task.get("primary_task") or "market_collection")
    protected = _protected_classes(task, session)
    if primary == "overnight_research" and level in {"observe", "coordinate"}:
        intent = "use_idle_headroom"
    elif protected and level in {"coordinate", "protect", "deep_protect", "relief"}:
        intent = "yield_to_foreground"
    elif backlog_gap > 0:
        intent = "background_drain_only"
    else:
        intent = "balanced_collection"

    writer_merge_seconds = "30" if level in {"deep_protect", "relief"} else "45" if level == "protect" else "60"
    shard_timeout_seconds = "300" if level in {"deep_protect", "relief"} else "420"
    env_overlay = {
        "COMPUTER_UNISON_CONTRACT_ACTIVE": "1",
        "COMPUTER_RESOURCE_INTENT": intent,
        "COMPUTER_FRICTION_INDEX": str(int(round(friction_index))),
        "COMPUTER_PREEMPTION_LEVEL": level,
        "COMPUTER_PROTECTED_TASKS": ",".join(protected),
        "COMPUTER_DO_NOT_TOUCH_VOLUMES": "/Volumes/VIDEO",
        "MACOS_NORMAL_USE_FIRST": "1",
        "SQL_LINK_SERVICE_MERGE_MAX_SECONDS_PER_CYCLE": writer_merge_seconds,
        "SQL_LINK_SERVICE_SHARD_LINK_TIMEOUT_SECONDS": shard_timeout_seconds,
        "OPS_SUPPORT_JOB_NICE": "14" if level in {"protect", "deep_protect", "relief"} else "12",
        "SUPPORT_TELEMETRY_SHED_ACTIVE": "1" if backlog_gap > 0 or level in {"deep_protect", "relief"} else "0",
    }
    return {
        "active": True,
        "resource_intent": intent,
        "preemption_level": level,
        "friction_index": round(friction_index, 1),
        "protected_task_classes": protected,
        "protected_open_apps": [str(item) for item in _as_list(session.get("open_apps")) if str(item).strip()][:12],
        "host_saturation_score": round(host, 3),
        "compute_pressure_level": compute,
        "collector_intake_ratio": env.get("BOT_COLLECTION_DUTY_CYCLE_MAX_ACTIVE_RATIO", ""),
        "async_pipeline_workers": env.get("ASYNC_PIPELINE_WORKERS", ""),
        "control_env_overlay": env_overlay,
        "computer_needs": ordered_unique(
            [
                "host_saturation_score <= task max minus 10" if foreground_gap > 0 else "",
                "co-running foreground apps stay responsive" if protected else "",
                "single writer keeps progressing without stale shard-linking windows" if backlog_gap > 0 else "",
                "core_pending_lines <= 5000 and total_pending_lines <= 10000" if backlog_gap > 0 else "",
                "memory_pressure_level remains normal" if memory_gap > 0 else "",
            ]
        ),
        "preemption_ladder": [
            {"level": "observe", "action": "measure foreground, backlog, memory, and restart posture"},
            {"level": "coordinate", "action": "publish task-aware caps and keep optional work bounded"},
            {"level": "protect", "action": "renice bot loops, cap intake, pause training and heavy collectors"},
            {"level": "deep_protect", "action": "shorten future writer cycles, shed support telemetry, keep only core drain moving"},
            {"level": "relief", "action": "hold expansion/research and require writer handoff before more backlog work"},
        ],
        "routing_contract": {
            "mode_switchboard_reads": ["control_env_overlay", "resource_intent", "preemption_level"],
            "system_intelligence_reads": ["friction_index", "computer_needs", "protected_task_classes"],
            "process_coordination_reads": ["preemption_level", "protected_task_classes"],
        },
        "safety_contract": {
            "does_not_close_user_apps": True,
            "does_not_touch_video_volume": True,
            "do_not_touch_volumes": ["/Volumes/VIDEO"],
            "does_not_start_parallel_sql_writers": True,
            "renice_only_for_running_processes": bool(_as_dict(process_policy.get("contract")).get("renice_only", True)),
        },
    }


def _env_overrides(task: dict[str, Any], budget: dict[str, Any], scorecard: dict[str, Any]) -> dict[str, str]:
    training_allowed = bool(budget.get("training_allowed", False)) and str(scorecard.get("overall_grade") or "") in {"A", "B"}
    heavy_allowed = bool(budget.get("heavy_collectors_allowed", False)) and str(scorecard.get("overall_grade") or "") in {"A", "B"}
    reports_allowed = bool(budget.get("report_refresh_allowed", False)) and str(scorecard.get("overall_grade") or "") in {"A", "B"}
    primary_task = str(task.get("primary_task") or "market_collection")
    active_tasks = {str(item).lower() for item in _as_list(task.get("active_tasks"))}
    foreground_score = _section_score(scorecard, "foreground_responsiveness")
    backlog_score = _section_score(scorecard, "backlog_interference")
    hardening_active = bool(foreground_score < 75.0 or backlog_score < 75.0)
    ratio_cap = 0.35 if primary_task in {"audio_production", "video_editing", "music_playback", "virtualization"} else 0.40
    ratio = _tighten_ratio(budget.get("collector_intake_ratio"), ratio_cap) if hardening_active else str(budget.get("collector_intake_ratio") or "0.45")
    async_workers = "1" if hardening_active and primary_task in {"audio_production", "video_editing", "music_playback", "virtualization"} else str(budget.get("async_pipeline_workers") or "2")
    feed_lines = "20" if hardening_active and primary_task in {"audio_production", "video_editing", "music_playback"} else str(budget.get("live_feed_lines") or "30")
    feed_files = "5" if hardening_active and primary_task in {"audio_production", "video_editing", "music_playback"} else str(budget.get("live_feed_follow_files") or "8")
    creative_protected = primary_task in {"audio_production", "video_editing", "virtualization"}
    foreground_daily_driver = bool(active_tasks & {"music_playback", "developer_work", "browser_work"})
    performance_drain = primary_task in {"backlog_drain", "market_collection", "overnight_research"}
    if creative_protected:
        scheduler_intent = "foreground_creative_protect"
        writer_nice = "8"
        writer_background = "1"
        baseline_nice = "8"
        aggressive_nice = "8"
        specialized_nice = "12"
    elif foreground_daily_driver and not performance_drain:
        scheduler_intent = "daily_driver_no_background_writer"
        writer_nice = "3"
        writer_background = "0"
        baseline_nice = "4"
        aggressive_nice = "4"
        specialized_nice = "8"
    else:
        scheduler_intent = "performance_core_backlog_drain"
        writer_nice = "0"
        writer_background = "0"
        baseline_nice = "0"
        aggressive_nice = "0"
        specialized_nice = "6"
    return {
        "COMPUTER_TASK_INTELLIGENCE_ACTIVE": "1",
        "COMPUTER_TASK_PROFILE": primary_task,
        "COMPUTER_NORMAL_USE_TARGET_GRADE": "A",
        "COMPUTER_NORMAL_USE_CURRENT_GRADE": str(scorecard.get("overall_grade") or ""),
        "COMPUTER_NORMAL_USE_GOVERNOR_ACTIVE": "1",
        "COMPUTER_TASK_FOREGROUND_HARDENING_ACTIVE": "1" if hardening_active else "0",
        "COMPUTER_TASK_BACKGROUND_POLICY": "1",
        "COMPUTER_TASK_BACKGROUND_NICE_LEVEL": "14" if hardening_active else "10",
        "COMPUTER_TASK_MAX_RENICE_PROCESSES": "12" if hardening_active else "6",
        "BOT_CPU_EFFICIENCY_SATURATION_GUARD": "1",
        "BOT_CPU_SCHEDULER_INTENT": scheduler_intent,
        "SQL_LINK_WRITER_NICE": writer_nice,
        "SQL_LINK_WRITER_BACKGROUND_POLICY": writer_background,
        "OPS_SQL_WRITER_NICE": writer_nice,
        "OPS_SQL_WRITER_BACKGROUND_POLICY": writer_background,
        "SLEEVE_NICE_BASELINE": baseline_nice,
        "SLEEVE_NICE_AGGRESSIVE": aggressive_nice,
        "SLEEVE_NICE_SPECIALIZED": specialized_nice,
        "SLEEVE_NICE_DIVIDEND": "10" if creative_protected else "8",
        "SLEEVE_NICE_DIVIDEND_CAPTURE": "10" if creative_protected else "8",
        "SLEEVE_NICE_BOND": "10" if creative_protected else "8",
        "SLEEVE_NICE_FX": "10" if creative_protected else "8",
        "SYSTEM_OPERATOR_MODE_REQUESTED": str(budget.get("requested_operator_mode") or "daily_driver"),
        "BOT_COLLECTION_DUTY_CYCLE_ENABLED": "1",
        "BOT_COLLECTION_DUTY_CYCLE_MAX_ACTIVE_RATIO": ratio,
        "ASYNC_PIPELINE_WORKERS": async_workers,
        "COINBASE_SNAPSHOT_MAX_WORKERS": str(budget.get("coinbase_snapshot_workers") or "1"),
        "QUANT_MODEL_MAX_WORKERS": str(budget.get("quant_model_workers") or "1"),
        "TRAINING_RUNTIME_PAUSED_FOR_COMPUTER_TASK": "0" if training_allowed else "1",
        "SHADOW_RESEARCH_PAUSED_FOR_COMPUTER_TASK": "0" if training_allowed else "1",
        "HEAVY_COLLECTORS_PAUSED_FOR_COMPUTER_TASK": "0" if heavy_allowed else "1",
        "REPORT_REFRESH_PAUSED_FOR_COMPUTER_TASK": "0" if reports_allowed else "1",
        "ROSTER_EXPANSION_ALLOWED": "1" if training_allowed and heavy_allowed else "0",
        "LIVE_FEED_HEAVY_DEFAULT_LINES": feed_lines,
        "LIVE_FEED_HEAVY_MAX_FOLLOW_FILES": feed_files,
        "OPS_HEAVY_SUPPORT_OFF_HOURS_ONLY": "1",
    }


def _write_override(path: Path, env: dict[str, str]) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# Auto-managed by scripts/ops/computer_task_intelligence.py"]
    for key in sorted(env):
        lines.append(f"{key}={shlex.quote(str(env[key]))}")
    text = "\n".join(lines) + "\n"
    old = path.read_text(encoding="utf-8") if path.exists() else ""
    if old == text:
        return False
    path.write_text(text, encoding="utf-8")
    return True


def _process_target_nice(process_class: dict[str, Any], primary_task: str, backlog_score: float) -> int:
    class_id = str(process_class.get("class_id") or "")
    if primary_task in {"audio_production", "video_editing", "virtualization"}:
        protected = {
            "single_writer": 8,
            "drainer_accelerator": 10,
            "market_runtime": 12,
            "macro_media_capture": 14,
            "research_training": 16,
        }
        return protected.get(class_id, _safe_int(process_class.get("target_nice"), 12))
    if primary_task in {"music_playback", "developer_work", "browser_work"}:
        daily_driver = {
            "single_writer": 3,
            "drainer_accelerator": 4,
            "market_runtime": 8,
            "macro_media_capture": 14,
            "research_training": 16,
        }
        return daily_driver.get(class_id, _safe_int(process_class.get("target_nice"), 12))
    if primary_task in {"backlog_drain", "market_collection", "overnight_research"} or backlog_score < 90.0:
        performance_drain = {
            "single_writer": 0,
            "drainer_accelerator": 0,
            "market_runtime": 4,
            "macro_media_capture": 12,
            "research_training": 14,
        }
        return performance_drain.get(class_id, _safe_int(process_class.get("target_nice"), 12))
    return _safe_int(process_class.get("target_nice"), 12)


def _renice_delta_for_target(current_nice: int, target_nice: int) -> int:
    return max(min(int(target_nice), 20) - max(int(current_nice), 0), 0)


def _process_coordination_policy(task: dict[str, Any], scorecard: dict[str, Any], env: dict[str, str]) -> dict[str, Any]:
    foreground_score = _section_score(scorecard, "foreground_responsiveness")
    backlog_score = _section_score(scorecard, "backlog_interference")
    primary_task = str(task.get("primary_task") or "market_collection")
    hardening_active = env.get("COMPUTER_TASK_FOREGROUND_HARDENING_ACTIVE") == "1"
    active = bool(
        hardening_active
        or primary_task in {"audio_production", "video_editing", "music_playback", "virtualization", "developer_work"}
        or foreground_score < 90.0
        or backlog_score < 90.0
    )
    max_processes = _safe_int(env.get("COMPUTER_TASK_MAX_RENICE_PROCESSES"), 8)
    return {
        "active": active,
        "primary_task": primary_task,
        "reason": "foreground_or_backlog_pressure" if active else "normal_use_clear",
        "foreground_score": round(foreground_score, 1),
        "backlog_score": round(backlog_score, 1),
        "max_processes": max(max_processes, 1),
        "process_classes": [
            {
                "class_id": str(item.get("class_id") or ""),
                "target_nice": _process_target_nice(item, primary_task, backlog_score),
                "patterns": [str(pattern) for pattern in _as_list(list(item.get("patterns") or ()))],
            }
            for item in BACKGROUND_PROCESS_CLASSES
        ],
        "contract": {
            "terminates_processes": False,
            "renice_only": True,
            "protects_single_writer": True,
            "priority_lift_requires_restart": True,
            "touches_video_volume": False,
        },
    }


def _parse_process_table(stdout: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in stdout.splitlines():
        raw = line.strip()
        if not raw or raw.lower().startswith("pid "):
            continue
        parts = raw.split(None, 4)
        if len(parts) < 5:
            continue
        rows.append(
            {
                "pid": _safe_int(parts[0], -1),
                "nice": _safe_int(parts[1], 0),
                "cpu_pct": _safe_float(parts[2], 0.0),
                "mem_pct": _safe_float(parts[3], 0.0),
                "command": parts[4],
            }
        )
    return [row for row in rows if _safe_int(row.get("pid"), -1) > 0]


def _matching_processes(policy: dict[str, Any]) -> tuple[list[dict[str, Any]], str]:
    try:
        result = subprocess.run(
            ["ps", "-axo", "pid,ni,pcpu,pmem,command"],
            check=False,
            capture_output=True,
            text=True,
            timeout=8,
        )
    except Exception as exc:
        return [], f"process_table_error:{type(exc).__name__}:{exc}"
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "").strip()
        return [], f"process_table_unavailable:{detail[:180]}"

    current_pid = os.getpid()
    matches: list[dict[str, Any]] = []
    for row in _parse_process_table(result.stdout):
        pid = _safe_int(row.get("pid"), -1)
        command = str(row.get("command") or "")
        if pid == current_pid or "computer_task_intelligence.py" in command:
            continue
        for process_class in _as_list(policy.get("process_classes")):
            if not isinstance(process_class, dict):
                continue
            patterns = [str(pattern) for pattern in _as_list(process_class.get("patterns")) if str(pattern).strip()]
            if any(pattern in command for pattern in patterns):
                target_nice = _safe_int(process_class.get("target_nice"), 12)
                current_nice = _safe_int(row.get("nice"), 0)
                matches.append(
                    {
                        "pid": pid,
                        "class_id": str(process_class.get("class_id") or ""),
                        "current_nice": current_nice,
                        "target_nice": target_nice,
                        "needs_renice": current_nice < target_nice,
                        "needs_priority_lift_restart": current_nice > target_nice,
                        "cpu_pct": round(_safe_float(row.get("cpu_pct"), 0.0), 2),
                        "mem_pct": round(_safe_float(row.get("mem_pct"), 0.0), 2),
                        "command": command[:220],
                    }
                )
                break
    matches.sort(key=lambda row: (_safe_float(row.get("cpu_pct"), 0.0), bool(row.get("needs_renice"))), reverse=True)
    return matches, ""


def _coordinate_background_processes(policy: dict[str, Any], *, apply: bool) -> dict[str, Any]:
    if not bool(policy.get("active", False)):
        return {
            "active": False,
            "applied": False,
            "renice_attempted": 0,
            "renice_succeeded": 0,
            "matched_process_count": 0,
            "actions": [],
            "errors": [],
        }
    matches, error = _matching_processes(policy)
    errors = [error] if error else []
    max_processes = max(_safe_int(policy.get("max_processes"), 8), 1)
    candidates = [row for row in matches if bool(row.get("needs_renice"))][:max_processes]
    actions: list[dict[str, Any]] = []
    if apply:
        for row in candidates:
            pid = _safe_int(row.get("pid"), -1)
            target_nice = _safe_int(row.get("target_nice"), 12)
            current_nice = _safe_int(row.get("current_nice"), 0)
            renice_delta = _renice_delta_for_target(current_nice, target_nice)
            action = {
                "pid": pid,
                "class_id": str(row.get("class_id") or ""),
                "from_nice": current_nice,
                "to_nice": target_nice,
                "renice_delta": renice_delta,
                "applied": False,
                "command": str(row.get("command") or "")[:160],
            }
            try:
                result = subprocess.run(
                    ["renice", "-n", str(renice_delta), "-p", str(pid)],
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
            except Exception as exc:
                action["error"] = f"{type(exc).__name__}:{exc}"
            else:
                action["applied"] = result.returncode == 0
                if result.returncode != 0:
                    action["error"] = (result.stderr or result.stdout or "").strip()[:180]
            actions.append(action)

    return {
        "active": True,
        "applied": bool(apply),
        "renice_attempted": len(actions) if apply else 0,
        "renice_succeeded": sum(1 for row in actions if bool(row.get("applied"))),
        "matched_process_count": len(matches),
        "candidate_process_count": len(candidates),
        "priority_lift_restart_required_count": sum(1 for row in matches if bool(row.get("needs_priority_lift_restart", False))),
        "matched_processes": matches[:max_processes],
        "actions": actions,
        "errors": errors,
        "contract": policy.get("contract", {}),
    }


def build_payload(project_root: Path = PROJECT_ROOT, *, refresh_computer: bool = True) -> dict[str, Any]:
    health = project_root / "governance" / "health"
    resource_guard = resource_src.build_snapshot(project_root) if refresh_computer else load_json(health / "resource_guard_latest.json")
    memory_efficiency = load_json(health / "memory_efficiency_control_latest.json")
    runtime = load_json(health / "runtime_throttle_control_latest.json")
    storage = load_json(health / "ingestion_storage_control_latest.json")
    drainer = load_json(health / "drainer_intelligence_layer_latest.json")
    mode_switchboard = load_json(health / "mode_switchboard_mission_control_latest.json")
    process_watchdog = load_json(health / "process_watchdog_latest.json")

    session = _session_context(memory_efficiency, resource_guard)
    task = _task_profile(session, storage, drainer)
    budget = _budget_for_task(str(task.get("primary_task") or "market_collection"))
    scorecard = _scorecard(
        task=task,
        budget=budget,
        session=session,
        runtime=runtime,
        memory_efficiency=memory_efficiency,
        resource_guard=resource_guard,
        storage=storage,
        drainer=drainer,
        mode_switchboard=mode_switchboard,
        process_watchdog=process_watchdog,
    )
    needs = [_a_grade_need(row) for row in _as_list(scorecard.get("sections")) if str(row.get("grade") or "") != "A"]
    env = _env_overrides(task, budget, scorecard)
    stale_context_infrabot = _as_dict(session.get("process_context_infrabot"))
    env.update(
        {
            "STALE_PROCESS_CONTEXT_INFRABOT_ACTIVE": "1",
            "STALE_PROCESS_CONTEXT_CLEARED": "1"
            if bool(stale_context_infrabot.get("ignored_memory_efficiency_app_context", False))
            else "0",
            "STALE_PROCESS_CONTEXT_MAX_AGE_SECONDS": str(int(_safe_float(stale_context_infrabot.get("max_context_age_seconds"), 180.0))),
            "PROCESS_CONTEXT_SOURCE": "fresh_resource_guard"
            if bool(stale_context_infrabot.get("ignored_memory_efficiency_app_context", False))
            else "memory_efficiency_plus_resource_guard",
        }
    )
    process_policy = _process_coordination_policy(task, scorecard, env)
    unison = _computer_unison_contract(
        task=task,
        session=session,
        scorecard=scorecard,
        runtime=runtime,
        storage=storage,
        process_policy=process_policy,
        env=env,
    )
    env = {**env, **_as_dict(unison.get("control_env_overlay"))}
    overall_grade = str(scorecard.get("overall_grade") or "F")
    status = "ready" if overall_grade == "A" else "advisory" if overall_grade in {"B", "C"} else "blocked"
    return {
        "timestamp_utc": iso_now(),
        "schema_version": 1,
        "ok": overall_grade in {"A", "B"},
        "overall_status": status,
        "computer_probe": {
            "live_probe_used": bool(refresh_computer),
            "resource_guard_source": "live_resource_guard_build_snapshot" if refresh_computer else "resource_guard_latest_artifact",
        },
        "task_profile": task,
        "session_context": session,
        "normal_use_budget": budget,
        "normal_use_scorecard": scorecard,
        "a_grade_lift_contract": {
            "target_grade": "A",
            "target_score": 90.0,
            "needs": needs,
            "blocking_sections": [str(row.get("section_id") or "") for row in needs],
        },
        "recommended_env_overrides": env,
        "process_coordination": process_policy,
        "stale_process_context_infrabot": stale_context_infrabot,
        "computer_unison_contract": unison,
        "recommended_actions": ordered_unique(
            [
                "apply computer-task intelligence before mode-switchboard so the operator mode is task-aware",
                f"computer unison intent: {unison.get('resource_intent')} at {unison.get('preemption_level')} preemption",
                "stale process-context infrabot cleared old foreground app context"
                if bool(stale_context_infrabot.get("ignored_memory_efficiency_app_context", False))
                else "",
                "keep daily-driver active while foreground apps or heavy cotenants are present"
                if str(budget.get("requested_operator_mode") or "") == "daily_driver"
                else "",
                "renice bot background loops instead of stopping user apps while foreground pressure is elevated"
                if bool(process_policy.get("active", False))
                else "",
                "hold training and expansion until computer normal-use grade reaches A/B and backlog is below target"
                if env.get("TRAINING_RUNTIME_PAUSED_FOR_COMPUTER_TASK") == "1"
                else "",
                "continue focused single-writer drain until backlog_interference reaches A"
                if any(str(row.get("section_id") or "") == "backlog_interference" for row in needs)
                else "",
            ]
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Infer the Mac's current task and publish task-aware normal-use controls.")
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--out-file", default=str(DEFAULT_OUT_PATH))
    parser.add_argument("--override-file", default=str(DEFAULT_OVERRIDE_PATH))
    parser.add_argument("--skip-refresh", action="store_true")
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    payload = build_payload(project_root, refresh_computer=not args.skip_refresh)
    if args.apply:
        override_path = Path(args.override_file).expanduser()
        if not override_path.is_absolute():
            override_path = project_root / override_path
        changed = _write_override(override_path, payload.get("recommended_env_overrides", {}))
        process_result = _coordinate_background_processes(_as_dict(payload.get("process_coordination")), apply=True)
        payload["apply_result"] = {
            "override_path": str(override_path),
            "changed": bool(changed),
            "loaded_by": "scripts/ops/load_runtime_env.sh",
            "process_coordination": process_result,
        }

    out_path = Path(args.out_file).expanduser()
    if not out_path.is_absolute():
        out_path = project_root / out_path
    write_payload(out_path, payload)

    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        print(
            "computer_task_intelligence "
            f"overall_status={payload.get('overall_status', '')} "
            f"task={_as_dict(payload.get('task_profile')).get('primary_task', '')} "
            f"grade={_as_dict(payload.get('normal_use_scorecard')).get('overall_grade', '')}"
        )
    return 0 if payload.get("overall_status") in {"ready", "advisory"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
