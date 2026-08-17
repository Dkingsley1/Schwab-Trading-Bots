#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shlex
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from scripts.ops.long_runtime_common import PROJECT_ROOT, iso_now, load_json, ordered_unique, write_payload
else:
    from .long_runtime_common import PROJECT_ROOT, iso_now, load_json, ordered_unique, write_payload


DEFAULT_OUT_PATH = PROJECT_ROOT / "governance" / "health" / "mode_switchboard_mission_control_latest.json"
DEFAULT_OVERRIDE_PATH = PROJECT_ROOT / "config" / ".env.operator_mode_override"


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


def _process_names(process_watchdog: dict[str, Any]) -> list[str]:
    rows = process_watchdog.get("status") if isinstance(process_watchdog.get("status"), list) else []
    return [str((row or {}).get("name") or "").strip().lower() for row in rows if isinstance(row, dict)]


def _mode_row(name: str, *, active: bool, reason: str, ready: bool) -> dict[str, Any]:
    state = "active" if active else ("ready" if ready else "idle")
    return {
        "mode": name,
        "state": state,
        "active": bool(active),
        "ready": bool(ready),
        "reason": reason,
    }


def _backpressure(storage: dict[str, Any]) -> dict[str, Any]:
    return storage.get("backpressure") if isinstance(storage.get("backpressure"), dict) else {}


def _as_dict(raw: Any) -> dict[str, Any]:
    return raw if isinstance(raw, dict) else {}


def _memory_efficiency_session(memory_efficiency: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    creative = memory_efficiency.get("creative_session") if isinstance(memory_efficiency.get("creative_session"), dict) else {}
    cotenant = memory_efficiency.get("cotenant_awareness") if isinstance(memory_efficiency.get("cotenant_awareness"), dict) else {}
    return creative, cotenant


def _requested_mode() -> str:
    return str(os.getenv("SYSTEM_OPERATOR_MODE_REQUESTED") or "").strip().lower().replace("-", "_")


def _is_off_hours(now: datetime | None = None) -> bool:
    current = now or datetime.now().astimezone()
    if current.weekday() >= 5:
        return True
    local_minutes = current.hour * 60 + current.minute
    return bool(local_minutes >= 20 * 60 or local_minutes < 7 * 60)


def _operator_mode_contract(
    *,
    runtime_throttle: dict[str, Any],
    memory_efficiency: dict[str, Any],
    storage: dict[str, Any],
    drainer_intelligence: dict[str, Any],
    computer_task: dict[str, Any] | None = None,
) -> dict[str, Any]:
    creative, cotenant = _memory_efficiency_session(memory_efficiency)
    bp = _backpressure(storage)
    host_saturation = _safe_float(runtime_throttle.get("host_saturation_score"), 0.0)
    compute_pressure = str(runtime_throttle.get("compute_pressure_level") or "").strip().lower()
    memory_pressure = str(runtime_throttle.get("memory_pressure_level") or "").strip().lower()
    total_pending = _safe_int(bp.get("total_pending_lines"), _safe_int(storage.get("pending_lines_total"), 0))
    core_pending = _safe_int(bp.get("core_pending_lines"), 0)
    pending_threshold = max(_safe_int(bp.get("pending_lines_threshold"), 15000), 1)
    backlog_grade = str(((drainer_intelligence.get("backlog_section_scorecard") or {}).get("overall_grade") or "")).strip().upper()
    creative_active = bool(creative.get("active", False)) or str(creative.get("level") or "").strip().lower() not in {"", "none"}
    cotenant_active = bool(cotenant.get("active", False))
    co_running_level = str(cotenant.get("co_running_level") or "").strip().lower()
    open_apps = cotenant.get("open_apps") if isinstance(cotenant.get("open_apps"), list) else []
    storage_blocked = str(storage.get("overall_status") or "").strip().lower() in {"blocked", "critical"} or total_pending > pending_threshold
    hard_pressure = bool(host_saturation >= 80.0 or compute_pressure in {"high", "critical"} or memory_pressure in {"high", "critical"})
    off_hours = _is_off_hours()
    requested = _requested_mode()
    computer_task = _as_dict(computer_task)
    computer_budget = _as_dict(computer_task.get("normal_use_budget"))
    computer_unison = _as_dict(computer_task.get("computer_unison_contract"))
    computer_env = {**_as_dict(computer_unison.get("control_env_overlay")), **_as_dict(computer_task.get("recommended_env_overrides"))}
    computer_profile = str(_as_dict(computer_task.get("task_profile")).get("primary_task") or "").strip().lower()
    computer_resource_intent = str(computer_unison.get("resource_intent") or computer_env.get("COMPUTER_RESOURCE_INTENT") or "").strip().lower()
    computer_preemption_level = str(computer_unison.get("preemption_level") or computer_env.get("COMPUTER_PREEMPTION_LEVEL") or "").strip().lower()
    computer_protective = bool(
        computer_resource_intent in {"yield_to_foreground", "background_drain_only"}
        or computer_preemption_level in {"protect", "deep_protect", "relief"}
    )
    computer_requested_mode = str(computer_budget.get("requested_operator_mode") or "").strip().lower()
    if not requested and computer_requested_mode in {"daily_driver", "trading_focus", "overnight_heavy"}:
        requested = computer_requested_mode

    if requested in {"daily_driver", "trading_focus", "overnight_heavy"}:
        selected = requested
        reason = "operator_requested_mode"
    elif hard_pressure or creative_active or cotenant_active or storage_blocked or computer_protective:
        selected = "daily_driver"
        reason = "computer_unison_requires_daily_driver" if computer_protective else "foreground_or_pressure_requires_daily_driver"
    elif off_hours and host_saturation < 60.0 and total_pending <= pending_threshold and backlog_grade in {"A", "B", ""}:
        selected = "overnight_heavy"
        reason = "off_hours_clean_envelope"
    else:
        selected = "trading_focus"
        reason = "market_collection_with_available_headroom"

    if selected == "overnight_heavy" and (creative_active or hard_pressure or storage_blocked):
        selected = "daily_driver"
        reason = "overnight_heavy_denied_by_foreground_or_pressure"
    if selected == "trading_focus" and hard_pressure:
        selected = "daily_driver"
        reason = "trading_focus_denied_by_hard_pressure"

    budget_by_mode = {
        "daily_driver": {
            "max_host_saturation": 65,
            "collector_intake_ratio": "0.45",
            "coinbase_snapshot_workers": "1",
            "async_pipeline_workers": "2",
            "quant_model_workers": "1",
            "training_allowed": False,
            "heavy_collectors_allowed": False,
            "report_refresh_allowed": False,
            "drain_wave_mode": "micro_capped",
        },
        "trading_focus": {
            "max_host_saturation": 75,
            "collector_intake_ratio": "0.68",
            "coinbase_snapshot_workers": "2",
            "async_pipeline_workers": "3",
            "quant_model_workers": "1",
            "training_allowed": False,
            "heavy_collectors_allowed": False,
            "report_refresh_allowed": True,
            "drain_wave_mode": "bounded_single_writer",
        },
        "overnight_heavy": {
            "max_host_saturation": 85,
            "collector_intake_ratio": "0.90",
            "coinbase_snapshot_workers": "3",
            "async_pipeline_workers": "4",
            "quant_model_workers": "2",
            "training_allowed": True,
            "heavy_collectors_allowed": True,
            "report_refresh_allowed": True,
            "drain_wave_mode": "bounded_single_writer_plus_followups",
        },
    }
    budget = budget_by_mode[selected]
    if computer_requested_mode == selected:
        budget = {
            **budget,
            "collector_intake_ratio": str(computer_env.get("BOT_COLLECTION_DUTY_CYCLE_MAX_ACTIVE_RATIO") or computer_budget.get("collector_intake_ratio") or budget["collector_intake_ratio"]),
            "coinbase_snapshot_workers": str(computer_env.get("COINBASE_SNAPSHOT_MAX_WORKERS") or computer_budget.get("coinbase_snapshot_workers") or budget["coinbase_snapshot_workers"]),
            "async_pipeline_workers": str(computer_env.get("ASYNC_PIPELINE_WORKERS") or computer_budget.get("async_pipeline_workers") or budget["async_pipeline_workers"]),
            "quant_model_workers": str(computer_env.get("QUANT_MODEL_MAX_WORKERS") or computer_budget.get("quant_model_workers") or budget["quant_model_workers"]),
            "training_allowed": bool(computer_budget.get("training_allowed", budget["training_allowed"])),
            "heavy_collectors_allowed": bool(computer_budget.get("heavy_collectors_allowed", budget["heavy_collectors_allowed"])),
            "report_refresh_allowed": bool(computer_budget.get("report_refresh_allowed", budget["report_refresh_allowed"])),
        }
    backlog_intake_active = bool(storage_blocked or total_pending > pending_threshold or core_pending > 15000)
    if backlog_intake_active:
        backlog_ratio_cap = 0.20 if hard_pressure or total_pending >= pending_threshold * 2 or core_pending >= pending_threshold else 0.30
        budget = {
            **budget,
            "collector_intake_ratio": f"{min(max(_safe_float(budget.get('collector_intake_ratio'), backlog_ratio_cap), 0.05), backlog_ratio_cap):.2f}",
            "coinbase_snapshot_workers": "1",
            "async_pipeline_workers": "1" if hard_pressure else str(min(_safe_int(budget.get("async_pipeline_workers"), 2), 2)),
            "heavy_collectors_allowed": False,
            "report_refresh_allowed": False if hard_pressure or total_pending >= pending_threshold * 2 else bool(budget["report_refresh_allowed"]),
        }
    foreground_guard_active = bool(creative_active or cotenant_active)
    no_training = bool(
        not budget["training_allowed"]
        or host_saturation >= 70.0
        or backlog_intake_active
        or foreground_guard_active
    )
    no_heavy_collectors = bool(
        not budget["heavy_collectors_allowed"]
        or host_saturation >= 80.0
        or foreground_guard_active
    )
    expansion_allowed = bool(
        selected == "overnight_heavy"
        and not no_training
        and not backlog_intake_active
        and host_saturation < 65.0
    )
    env = {
        "SYSTEM_OPERATOR_MODE": selected,
        "SYSTEM_OPERATOR_MODE_REASON": reason,
        "DAILY_DRIVER_MODE_ACTIVE": "1" if selected == "daily_driver" else "0",
        "TRADING_FOCUS_MODE_ACTIVE": "1" if selected == "trading_focus" else "0",
        "OVERNIGHT_HEAVY_MODE_ACTIVE": "1" if selected == "overnight_heavy" else "0",
        "FOREGROUND_APP_GOVERNOR_ACTIVE": "1" if foreground_guard_active else "0",
        "BACKLOG_INTAKE_GOVERNOR_ACTIVE": "1" if backlog_intake_active else "0",
        "BACKLOG_INTAKE_MAX_PENDING_LINES": str(pending_threshold),
        "BOT_COLLECTION_DUTY_CYCLE_ENABLED": "1",
        "BOT_COLLECTION_DUTY_CYCLE_MAX_ACTIVE_RATIO": str(budget["collector_intake_ratio"]),
        "TRAINING_RUNTIME_PAUSED_FOR_FOREGROUND": "1" if foreground_guard_active else "0",
        "TRAINING_RUNTIME_PAUSED_FOR_BACKLOG": "1" if backlog_intake_active else "0",
        "TRAINING_RUNTIME_PAUSED_BY_OPERATOR_MODE": "1" if no_training else "0",
        "SHADOW_RESEARCH_PAUSED_BY_OPERATOR_MODE": "1" if no_training else "0",
        "HEAVY_COLLECTORS_PAUSED_BY_OPERATOR_MODE": "1" if no_heavy_collectors else "0",
        "REPORT_REFRESH_PAUSED_BY_OPERATOR_MODE": "0" if budget["report_refresh_allowed"] and not foreground_guard_active else "1",
        "ROSTER_EXPANSION_ALLOWED": "1" if expansion_allowed else "0",
        "COINBASE_SNAPSHOT_MAX_WORKERS": str(budget["coinbase_snapshot_workers"]),
        "ASYNC_PIPELINE_WORKERS": str(budget["async_pipeline_workers"]),
        "QUANT_MODEL_MAX_WORKERS": str(budget["quant_model_workers"]),
        "REPORT_RENDER_MAX_JOBS": "1",
        "LIBRARY_REPORT_RENDER_JOBS": "1",
        "SQL_LINK_SERVICE_CONCENTRATED_CORE_DRAIN": "1" if backlog_intake_active else "0",
        "SQL_LINK_SERVICE_MERGE_MAX_SECONDS_PER_CYCLE": "45" if selected == "daily_driver" else "60",
        "SQL_LINK_SERVICE_SHARD_LINK_TIMEOUT_SECONDS": "420",
        "LOG_SUB_BOT_DECISIONS": "0" if selected == "daily_driver" and backlog_intake_active else "1",
        "LOG_MASTER_VARIANT_DECISIONS": "0" if selected == "daily_driver" and backlog_intake_active else "1",
        "OPS_HEAVY_SUPPORT_OFF_HOURS_ONLY": "1",
    }
    if selected == "daily_driver":
        env.update(
            {
                "LIVE_FEED_HEAVY_DEFAULT_LINES": str(computer_env.get("LIVE_FEED_HEAVY_DEFAULT_LINES") or computer_budget.get("live_feed_lines") or "30"),
                "LIVE_FEED_HEAVY_MAX_FOLLOW_FILES": str(computer_env.get("LIVE_FEED_HEAVY_MAX_FOLLOW_FILES") or computer_budget.get("live_feed_follow_files") or "8"),
                "SCHWAB_NEWS_CACHE_MAX_SYMBOLS": "24",
                "SCHWAB_OPTIONS_CHAIN_CACHE_MAX_SYMBOLS": "24",
            }
        )
    if computer_profile:
        env["COMPUTER_TASK_PROFILE"] = computer_profile
        env["COMPUTER_TASK_INTELLIGENCE_ACTIVE"] = "1"
    for key in (
        "COMPUTER_NORMAL_USE_TARGET_GRADE",
        "COMPUTER_NORMAL_USE_CURRENT_GRADE",
        "COMPUTER_NORMAL_USE_GOVERNOR_ACTIVE",
        "COMPUTER_TASK_FOREGROUND_HARDENING_ACTIVE",
        "COMPUTER_TASK_BACKGROUND_POLICY",
        "COMPUTER_TASK_BACKGROUND_NICE_LEVEL",
        "COMPUTER_TASK_MAX_RENICE_PROCESSES",
        "COMPUTER_UNISON_CONTRACT_ACTIVE",
        "COMPUTER_RESOURCE_INTENT",
        "COMPUTER_FRICTION_INDEX",
        "COMPUTER_PREEMPTION_LEVEL",
        "COMPUTER_PROTECTED_TASKS",
        "COMPUTER_DO_NOT_TOUCH_VOLUMES",
        "MACOS_NORMAL_USE_FIRST",
        "SUPPORT_TELEMETRY_SHED_ACTIVE",
        "OPS_SUPPORT_JOB_NICE",
        "SQL_LINK_SERVICE_MERGE_MAX_SECONDS_PER_CYCLE",
        "SQL_LINK_SERVICE_SHARD_LINK_TIMEOUT_SECONDS",
    ):
        if key in computer_env:
            env[key] = str(computer_env[key])
    return {
        "selected_mode": selected,
        "reason": reason,
        "requested_mode": requested,
        "off_hours": off_hours,
        "host_saturation_score": round(host_saturation, 3),
        "compute_pressure_level": compute_pressure,
        "memory_pressure_level": memory_pressure,
        "total_pending_lines": int(total_pending),
        "core_pending_lines": int(core_pending),
        "pending_lines_threshold": int(pending_threshold),
        "backlog_grade": backlog_grade,
        "foreground_guard_active": foreground_guard_active,
        "creative_active": creative_active,
        "creative_kind": str(creative.get("kind") or "none"),
        "cotenant_active": cotenant_active,
        "co_running_level": co_running_level,
        "open_apps": [str(item) for item in open_apps if str(item).strip()][:12],
        "computer_task_profile": computer_profile,
        "computer_task_budget_used": bool(computer_requested_mode == selected),
        "computer_resource_intent": computer_resource_intent,
        "computer_preemption_level": computer_preemption_level,
        "computer_unison_used": bool(computer_unison or computer_env.get("COMPUTER_UNISON_CONTRACT_ACTIVE") == "1"),
        "backlog_intake_governor": {
            "active": backlog_intake_active,
            "reason": "pending_above_threshold_or_core_above_c_floor" if backlog_intake_active else "inside_budget",
            "freeze_expansion": not expansion_allowed,
            "throttle_deferred_lanes": backlog_intake_active,
            "shed_support_telemetry": backlog_intake_active,
            "collector_intake_ratio": str(budget["collector_intake_ratio"]),
        },
        "hard_budgets": {
            "no_training_above_host_saturation": 70,
            "no_heavy_collectors_above_host_saturation": 80,
            "no_expansion_when_backlog_over_target": True,
            "foreground_apps_pause_optional_research": True,
            "training_allowed_now": not no_training,
            "heavy_collectors_allowed_now": not no_heavy_collectors,
            "expansion_allowed_now": expansion_allowed,
        },
        "six_point_taming_contract": [
            {
                "control": "daily_driver_mode",
                "active": selected == "daily_driver",
                "status": "protecting_foreground_and_backlog" if selected == "daily_driver" else "standby",
            },
            {
                "control": "trading_focus_mode",
                "active": selected == "trading_focus",
                "status": "market_collection_with_bounded_workers" if selected == "trading_focus" else "standby",
            },
            {
                "control": "overnight_heavy_mode",
                "active": selected == "overnight_heavy",
                "status": "allowed" if selected == "overnight_heavy" else "denied_until_clean_off_hours_envelope",
            },
            {
                "control": "backlog_intake_governor",
                "active": backlog_intake_active,
                "status": "throttling_new_intake" if backlog_intake_active else "inside_budget",
            },
            {
                "control": "foreground_app_contract",
                "active": foreground_guard_active,
                "status": "optional_heavy_work_paused" if foreground_guard_active else "clear",
            },
            {
                "control": "hard_budgets",
                "active": True,
                "status": "enforced",
            },
        ],
        "mode_budget": budget,
        "env_overrides": env,
    }


def _write_override(path: Path, env: dict[str, str]) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# Auto-managed by scripts/ops/mode_switchboard_mission_control.py"]
    for key in sorted(env):
        value = str(env[key])
        lines.append(f"{key}={shlex.quote(value)}")
    text = "\n".join(lines) + "\n"
    old = path.read_text(encoding="utf-8") if path.exists() else ""
    if old == text:
        return False
    path.write_text(text, encoding="utf-8")
    return True


def build_payload(project_root: Path = PROJECT_ROOT) -> dict[str, Any]:
    health_root = project_root / "governance" / "health"
    live_readiness = load_json(health_root / "live_readiness_smoke_latest.json")
    runtime = load_json(health_root / "live_runtime_separation_control_latest.json")
    runtime_throttle = load_json(health_root / "runtime_throttle_control_latest.json")
    memory_efficiency = load_json(health_root / "memory_efficiency_control_latest.json")
    storage = load_json(health_root / "ingestion_storage_control_latest.json")
    drainer_intelligence = load_json(health_root / "drainer_intelligence_layer_latest.json")
    computer_task = load_json(health_root / "computer_task_intelligence_latest.json")
    portable = load_json(health_root / "portable_brain_contract_latest.json")
    access_mode = load_json(health_root / "runtime_access_mode_latest.json")
    process_watchdog = load_json(health_root / "process_watchdog_latest.json")

    names = _process_names(process_watchdog)
    shadow_active = any(token in name for name in names for token in ("shadow", "watchdog", "all_sleeves"))
    paper_active = bool(live_readiness.get("paper_lane_fresh", False)) or any("paper" in name for name in names)
    live_active = bool(live_readiness.get("live_lane_running", False))

    shadow_ready = shadow_active or bool(names)
    paper_ready = paper_active or bool(live_readiness.get("paper_lane_fresh", False))
    live_ready = bool(live_readiness.get("broker_ready", False)) and bool(live_readiness.get("session_ready", False))

    modes = [
        _mode_row("shadow", active=shadow_active, ready=shadow_ready, reason="process_watchdog_shadow_lane" if shadow_active else "shadow_lane_not_detected"),
        _mode_row("paper", active=paper_active, ready=paper_ready, reason="paper_lane_fresh_or_running" if paper_active else "paper_lane_not_fresh"),
        _mode_row("live", active=live_active, ready=live_ready, reason="broker_and_session_ready" if live_ready else "live_lane_gated"),
    ]

    overall_status = "ready"
    if not shadow_ready or not paper_ready:
        overall_status = "degraded"
    if str(runtime.get("overall_status") or "").strip().lower() == "blocked":
        overall_status = "blocked"

    live_contract = runtime.get("release_contract") if isinstance(runtime.get("release_contract"), dict) else {}
    host_contract = portable.get("host_contract") if isinstance(portable.get("host_contract"), dict) else {}
    recommended_actions = ordered_unique(
        [
            "keep live in read-only posture while the switchboard still sees runtime separation contention" if bool(live_contract.get("live_lane_should_be_read_only", False)) else "",
            "refresh the paper lane before claiming three-mode continuity" if not paper_ready else "",
            "bring the shadow watchdog back up before treating the switchboard as fully available" if not shadow_ready else "",
        ]
    )
    operator_mode = _operator_mode_contract(
        runtime_throttle=runtime_throttle,
        memory_efficiency=memory_efficiency,
        storage=storage,
        drainer_intelligence=drainer_intelligence,
        computer_task=computer_task,
    )
    recommended_actions = ordered_unique(
        [
            *recommended_actions,
            "daily-driver mode active: cap workers, pause optional research, and protect foreground apps"
            if operator_mode["selected_mode"] == "daily_driver"
            else "",
            "backlog intake governor active: reduce production until the single writer visibly drains core"
            if bool(operator_mode.get("backlog_intake_governor", {}).get("active", False))
            else "",
            "overnight-heavy is denied until host saturation, foreground apps, and backlog are below budget"
            if operator_mode["selected_mode"] != "overnight_heavy"
            else "",
        ]
    )

    return {
        "timestamp_utc": iso_now(),
        "schema_version": 1,
        "ok": overall_status == "ready",
        "overall_status": overall_status,
        "modes": modes,
        "operator_mode": operator_mode,
        "mode_counts": {
            "active": sum(1 for row in modes if bool(row.get("active"))),
            "ready": sum(1 for row in modes if bool(row.get("ready"))),
        },
        "control_surface": {
            "runtime_access_mode": str(access_mode.get("mode") or ""),
            "host_profile": str(host_contract.get("host_profile") or ""),
            "clearance_state": str(((runtime.get("clearance_plan") or {}).get("clearance_state") or "")),
            "live_lane_should_be_read_only": bool(live_contract.get("live_lane_should_be_read_only", False)),
            "shared_host_training_resume_allowed": bool(live_contract.get("shared_host_training_resume_allowed", False)),
        },
        "recommended_actions": recommended_actions,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Publish the three-mode switchboard mission-control snapshot.")
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--out-file", default=str(DEFAULT_OUT_PATH))
    parser.add_argument("--override-file", default=str(DEFAULT_OVERRIDE_PATH))
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    payload = build_payload(project_root)
    changed = False
    if args.apply:
        override_path = Path(args.override_file).expanduser()
        if not override_path.is_absolute():
            override_path = project_root / override_path
        changed = _write_override(override_path, payload.get("operator_mode", {}).get("env_overrides", {}))
        payload["apply_result"] = {
            "override_path": str(override_path),
            "changed": changed,
            "loaded_by": "scripts/ops/load_runtime_env.sh",
        }
    out_path = Path(args.out_file).expanduser()
    write_payload(out_path, payload)
    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        print(
            "mode_switchboard_mission_control "
            f"overall_status={payload.get('overall_status', '')} "
            f"active_modes={int(((payload.get('mode_counts') or {}).get('active', 0) or 0))} "
            f"operator_mode={((payload.get('operator_mode') or {}).get('selected_mode') or '')}"
        )
    return 0 if payload.get("overall_status") in {"ready", "degraded"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
