#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from scripts import resource_guard as resource_src
    from scripts.ops import memory_efficiency_control as memory_src
    from scripts.ops.long_runtime_common import iso_now, load_json, status_rank, write_payload
else:
    from .. import resource_guard as resource_src
    from . import memory_efficiency_control as memory_src
    from .long_runtime_common import PROJECT_ROOT, iso_now, load_json, status_rank, write_payload


DEFAULT_OUT_PATH = PROJECT_ROOT / "governance" / "health" / "creative_cotenant_guard_latest.json"
DEFAULT_STATE_PATH = PROJECT_ROOT / "governance" / "health" / "creative_cotenant_guard_state.json"
DEFAULT_PAUSE_PATH = PROJECT_ROOT / "governance" / "health" / "creative_heavy_research_pause_latest.json"
DEFAULT_RESOURCE_GUARD_PATH = PROJECT_ROOT / "governance" / "health" / "resource_guard_latest.json"
HEAVY_RESEARCH_PATTERNS = [
    "scripts/run_all_sleeves.py",
    "scripts/run_parallel_shadows.py",
    "scripts/run_parallel_aggressive_modes.py",
    "scripts/run_shadow_training_loop.py",
    "scripts/run_dividend_shadow.py",
    "scripts/run_dividend_capture_shadow.py",
    "scripts/run_bond_shadow.py",
    "scripts/run_fx_shadow.py",
    "scripts/collect_market_crypto_correlation_context.py",
    "scripts/collect_market_micro_context.py",
    "scripts/ops/bounded_market_micro_sync.py",
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
    "scripts/sql_hot_retention.py",
    "scripts/sql_queue_retention.py",
]
PROTECTED_MANUAL_TRAINING_PROFILES = {
    "coverage_micro_canary",
    "coverage_small_canary",
    "coverage_canary",
    "coverage_batch10_canary",
    "coverage_batch20_canary",
    "coverage_batch30_canary",
}


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except Exception:
        return int(default)


def _merge_status(*statuses: str) -> str:
    ranked = [str(status or "ready") for status in statuses if str(status or "").strip()]
    if not ranked:
        return "ready"
    return max(ranked, key=status_rank)


def _parse_utc(raw: Any) -> datetime | None:
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


def _creative_mode_label(kind: str) -> str:
    labels = {
        "logic_pro": "Logic Pro",
        "logic_pro_hot": "Logic Pro hot",
        "final_cut_pro": "Final Cut Pro",
        "final_cut_pro_hot": "Final Cut Pro hot",
        "music_playback": "Music/iTunes playback",
        "music_playback_hot": "Music/iTunes playback hot",
        "dual_pro": "Logic Pro + Final Cut Pro",
        "cooldown": "creative cooldown",
        "none": "normal",
    }
    return labels.get(str(kind or "").strip().lower(), str(kind or "creative").replace("_", " "))


def _creative_level(snapshot: dict[str, Any]) -> str:
    return str(snapshot.get("creative_session_level") or "none").strip().lower() or "none"


def _creative_kind(snapshot: dict[str, Any]) -> str:
    return str(snapshot.get("creative_session_kind") or _creative_level(snapshot)).strip().lower() or "none"


def _creative_active(snapshot: dict[str, Any]) -> bool:
    return bool(snapshot.get("creative_apps_active", False)) or _creative_level(snapshot) in {"active", "hot", "dual_pro"}


def _notification_for_transition(previous_kind: str, current_kind: str, snapshot: dict[str, Any], now: datetime) -> dict[str, Any]:
    previous = str(previous_kind or "none").strip().lower()
    current = str(current_kind or "none").strip().lower()
    if previous == current:
        return {}
    apps = snapshot.get("creative_apps") if isinstance(snapshot.get("creative_apps"), list) else []
    app_text = ", ".join(str(app) for app in apps if str(app).strip())
    if current == "none":
        event = "creative_mode_cleared"
        message = "Creative mode cleared; bot stack can ramp normally after guard checks."
    elif current == "cooldown":
        event = "creative_mode_cooldown"
        remaining = int(float(snapshot.get("creative_cooldown_remaining_seconds", 0.0) or 0.0))
        message = f"Creative cooldown active for {remaining}s; bot stack stays calm before ramp-up."
    else:
        event = "creative_mode_active"
        app_suffix = f": {app_text}" if app_text else ""
        message = f"Creative mode active: {_creative_mode_label(current)}{app_suffix}. Bot stack downshifted."
    return {
        "timestamp_utc": now.isoformat(),
        "event": event,
        "severity": "info",
        "previous_state": previous,
        "current_state": current,
        "message": message,
    }


def _refresh_resource_guard_snapshot(
    project_root: Path,
    *,
    apply: bool,
    state_path: Path,
    cooldown_seconds: int,
    now: datetime,
) -> dict[str, Any]:
    snapshot = resource_src.build_snapshot(project_root)
    memory_state, memory_state_reasons, memory_thresholds = resource_src._memory_pressure_state(snapshot)
    snapshot.update(
        {
            "resource_guard_profile": "creative_cotenant_guard",
            "resource_guard_ok": True,
            "resource_guard_reasons": [],
            "memory_pressure_state": memory_state,
            "memory_pressure_reasons": memory_state_reasons,
            "memory_pressure_kind": resource_src._memory_pressure_kind(snapshot, memory_state, memory_state_reasons),
            "memory_pressure_thresholds": memory_thresholds,
        }
    )

    state = load_json(state_path)
    previous_kind = str(state.get("current_state") or "none").strip().lower()
    active = _creative_active(snapshot)
    current_kind = _creative_kind(snapshot)
    cooldown_until = _parse_utc(state.get("cooldown_until_utc"))
    if active:
        cooldown_until = now + timedelta(seconds=max(int(cooldown_seconds), 0))
    elif cooldown_until is not None and cooldown_until > now:
        current_kind = "cooldown"
        snapshot["creative_session_level"] = "cooldown"
        snapshot["creative_session_kind"] = "cooldown"
        snapshot["creative_cooldown_active"] = True
        snapshot["creative_cooldown_until_utc"] = cooldown_until.isoformat()
        snapshot["creative_cooldown_remaining_seconds"] = round((cooldown_until - now).total_seconds(), 3)
    else:
        cooldown_until = None
        current_kind = "none"
        snapshot["creative_cooldown_active"] = False
        snapshot["creative_cooldown_remaining_seconds"] = 0.0

    notification = _notification_for_transition(previous_kind, current_kind, snapshot, now) if apply else {}
    snapshot["creative_guard_state"] = {
        "previous_state": previous_kind,
        "current_state": current_kind,
        "cooldown_seconds": max(int(cooldown_seconds), 0),
        "cooldown_until_utc": cooldown_until.isoformat() if cooldown_until else "",
        "notification": notification,
    }
    write_payload(project_root / "governance" / "health" / "resource_guard_latest.json", snapshot)
    if apply:
        write_payload(
            state_path,
            {
                "timestamp_utc": now.isoformat(),
                "previous_state": previous_kind,
                "current_state": current_kind,
                "cooldown_until_utc": cooldown_until.isoformat() if cooldown_until else "",
                "last_active_apps": snapshot.get("creative_apps") if isinstance(snapshot.get("creative_apps"), list) else [],
                "last_notification": notification,
            },
        )
    return snapshot


def _process_name_running(name: str) -> bool:
    try:
        completed = subprocess.run(
            ["pgrep", "-x", name],
            check=False,
            capture_output=True,
            text=True,
            timeout=1,
        )
    except Exception:
        return False
    return completed.returncode == 0


def _refresh_lightweight_creative_snapshot(
    *,
    apply: bool,
    state_path: Path,
    cooldown_seconds: int,
    now: datetime,
) -> dict[str, Any]:
    running_apps: list[str] = []
    logic_active = _process_name_running("Logic Pro")
    final_cut_active = _process_name_running("Final Cut Pro")
    music_active = _process_name_running("Music") or _process_name_running("iTunes")

    if logic_active:
        running_apps.append("Logic Pro")
    if final_cut_active:
        running_apps.append("Final Cut Pro")
    if music_active:
        running_apps.append("Music")

    if logic_active and final_cut_active:
        current_kind = "dual_pro"
        current_level = "dual_pro"
    elif logic_active:
        current_kind = "logic_pro"
        current_level = "active"
    elif final_cut_active:
        current_kind = "final_cut_pro"
        current_level = "active"
    elif music_active:
        current_kind = "music_playback"
        current_level = "active"
    else:
        current_kind = "none"
        current_level = "none"

    state = load_json(state_path)
    previous_kind = str(state.get("current_state") or "none").strip().lower()
    cooldown_until = _parse_utc(state.get("cooldown_until_utc"))
    if running_apps:
        cooldown_until = now + timedelta(seconds=max(int(cooldown_seconds), 0))
    elif cooldown_until is not None and cooldown_until > now:
        current_kind = "cooldown"
        current_level = "cooldown"
    else:
        cooldown_until = None

    remaining = round((cooldown_until - now).total_seconds(), 3) if cooldown_until else 0.0
    snapshot = {
        "timestamp_utc": now.isoformat(),
        "resource_guard_profile": "creative_cotenant_guard_lightweight",
        "resource_guard_ok": True,
        "resource_guard_reasons": [],
        "creative_apps": running_apps,
        "creative_apps_active": bool(running_apps),
        "creative_session_level": current_level,
        "creative_session_kind": current_kind,
        "creative_cooldown_active": bool(current_kind == "cooldown"),
        "creative_cooldown_until_utc": cooldown_until.isoformat() if cooldown_until else "",
        "creative_cooldown_remaining_seconds": remaining,
        "memory_pressure_state": "unknown",
        "memory_pressure_reasons": [],
        "memory_pressure_kind": "unknown",
        "lightweight": True,
    }
    notification = _notification_for_transition(previous_kind, current_kind, snapshot, now) if apply else {}
    snapshot["creative_guard_state"] = {
        "previous_state": previous_kind,
        "current_state": current_kind,
        "cooldown_seconds": max(int(cooldown_seconds), 0),
        "cooldown_until_utc": cooldown_until.isoformat() if cooldown_until else "",
        "notification": notification,
    }
    if apply:
        write_payload(
            state_path,
            {
                "timestamp_utc": now.isoformat(),
                "previous_state": previous_kind,
                "current_state": current_kind,
                "cooldown_until_utc": cooldown_until.isoformat() if cooldown_until else "",
                "last_active_apps": running_apps,
                "last_notification": notification,
            },
        )
    return snapshot


def _paper_execution_lane_pattern(project_root: Path) -> str:
    return f"{project_root / 'scripts' / 'run_execution_lane.py'} --mode paper"


def _pgrep_matching_pids(pattern: str) -> list[int]:
    try:
        completed = subprocess.run(
            ["pgrep", "-f", pattern],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except Exception:
        return []

    if completed.returncode not in {0, 1}:
        return []

    out: list[int] = []
    seen: set[int] = set()
    for raw_line in (completed.stdout or "").splitlines():
        pid = _safe_int(raw_line.strip(), 0)
        if pid <= 0 or pid in seen:
            continue
        seen.add(pid)
        out.append(pid)
    return sorted(out)


def _terminate_pids(pids: list[int]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for pid in pids:
        row = {"pid": int(pid), "ok": False}
        try:
            os.kill(int(pid), signal.SIGTERM)
        except ProcessLookupError:
            row["error"] = "missing"
        except PermissionError as exc:
            row["error"] = f"permission:{exc}"
        else:
            row["ok"] = True
        rows.append(row)
    return rows


def _command_for_pid(pid: int) -> str:
    try:
        completed = subprocess.run(
            ["ps", "-p", str(int(pid)), "-o", "command="],
            check=False,
            capture_output=True,
            text=True,
            timeout=3,
        )
    except Exception:
        return ""
    if completed.returncode != 0:
        return ""
    return " ".join((completed.stdout or "").split())


def _is_protected_manual_training(command: str) -> bool:
    text = " ".join(str(command or "").split())
    if "scripts/weekly_retrain.py" not in text:
        return False
    if "--include-bot-ids" not in text or "--skip-master-update" not in text:
        return False
    if "--force-all-targets" in text or "full_overnight" in text:
        return False
    return any(f"--retrain-profile {profile}" in text for profile in PROTECTED_MANUAL_TRAINING_PROFILES)


def _matching_heavy_research_processes() -> list[dict[str, Any]]:
    current_pid = os.getpid()
    rows: list[dict[str, Any]] = []
    seen: set[int] = set()
    for pattern in HEAVY_RESEARCH_PATTERNS:
        try:
            completed = subprocess.run(
                ["pgrep", "-af", pattern],
                check=False,
                capture_output=True,
                text=True,
                timeout=5,
            )
        except Exception:
            continue
        if completed.returncode not in {0, 1}:
            continue
        for raw_line in (completed.stdout or "").splitlines():
            raw = raw_line.strip()
            if not raw:
                continue
            first, _, command = raw.partition(" ")
            pid = _safe_int(first, 0)
            if pid <= 0 or pid == current_pid or pid in seen:
                continue
            if not command:
                command = _command_for_pid(pid)
            if "scripts/shadow_watchdog.py" in command:
                continue
            if _is_protected_manual_training(command):
                continue
            seen.add(pid)
            rows.append({"pid": pid, "pattern": pattern, "command": command})
    return sorted(rows, key=lambda row: int(row.get("pid", 0) or 0))


def _pause_heavy_research(*, apply: bool, active: bool, terminate_processes: bool | None = None) -> dict[str, Any]:
    matches = _matching_heavy_research_processes()
    initial_matches = list(matches)
    terminated: list[dict[str, Any]] = []
    should_terminate = bool(active if terminate_processes is None else terminate_processes)
    if apply and active and should_terminate and matches:
        for _ in range(5):
            if not matches:
                break
            terminated.extend(_terminate_pids([int(row["pid"]) for row in matches]))
            time.sleep(0.6)
            matches = _matching_heavy_research_processes()
    return {
        "active": bool(active),
        "apply": bool(apply),
        "terminate_processes": bool(should_terminate),
        "action": (
            "sigterm_optional_heavy_research"
            if active and should_terminate
            else "soft_pause_optional_heavy_research"
            if active
            else "observe"
        ),
        "patterns": HEAVY_RESEARCH_PATTERNS,
        "match_count": len(initial_matches),
        "matches": initial_matches[:25],
        "remaining_match_count": len(matches),
        "remaining_matches": matches[:25],
        "terminated": terminated,
        "terminated_count": sum(1 for row in terminated if bool(row.get("ok", False))),
    }


def _pause_contract(project_root: Path, *, snapshot: dict[str, Any], pause_result: dict[str, Any], now: datetime) -> dict[str, Any]:
    level = _creative_level(snapshot)
    kind = _creative_kind(snapshot)
    active = bool(pause_result.get("active", False))
    payload = {
        "timestamp_utc": now.isoformat(),
        "schema_version": 1,
        "active": active,
        "creative_session_level": level,
        "creative_session_kind": kind,
        "creative_apps": snapshot.get("creative_apps") if isinstance(snapshot.get("creative_apps"), list) else [],
        "cooldown_active": bool(snapshot.get("creative_cooldown_active", False)),
        "cooldown_until_utc": str(snapshot.get("creative_cooldown_until_utc") or ""),
        "cooldown_remaining_seconds": float(snapshot.get("creative_cooldown_remaining_seconds", 0.0) or 0.0),
        "hard_pause": pause_result,
        "env_contract": {
            "CREATIVE_MODE_ACTIVE": "1" if active else "0",
            "CREATIVE_MODE_STATE": kind,
            "CREATIVE_MODE_LEVEL": level,
            "CREATIVE_HEAVY_RESEARCH_PAUSED": "1" if active else "0",
            "TRAINING_RUNTIME_PAUSED_FOR_CREATIVE": "1" if active else "0",
            "AUTO_RETRAIN_PAUSED_FOR_CREATIVE": "1" if active else "0",
            "QUANT_RESEARCH_PAUSED_FOR_CREATIVE": "1" if active else "0",
            "MLX_RESEARCH_PAUSED_FOR_CREATIVE": "1" if active else "0",
            "REPORT_BUILD_PAUSED_FOR_CREATIVE": "1" if active else "0",
            "RETENTION_MAINTENANCE_PAUSED_FOR_CREATIVE": "1" if active else "0",
        },
    }
    write_payload(project_root / "governance" / "health" / "creative_heavy_research_pause_latest.json", payload)
    return payload


def _paper_lane_snapshot(project_root: Path, *, apply: bool) -> dict[str, Any]:
    pattern = _paper_execution_lane_pattern(project_root)
    before = _pgrep_matching_pids(pattern)
    keep_pid = before[0] if before else None
    extras = before[1:] if len(before) > 1 else []
    terminated: list[dict[str, Any]] = []

    after = before
    if apply and extras:
        terminated = _terminate_pids(extras)
        after = _pgrep_matching_pids(pattern)

    return {
        "pattern": pattern,
        "count_before": len(before),
        "count_after": len(after),
        "keep_pid": (after[0] if after else keep_pid),
        "extra_pids": extras,
        "terminated": terminated,
    }


def _memory_efficiency_snapshot(project_root: Path, *, override_path: Path, apply: bool) -> dict[str, Any]:
    action = "apply" if apply else "status"
    payload = memory_src.build_payload(project_root, action=action, override_path=override_path, changed=False)
    if apply:
        changed = memory_src._write_override(
            override_path,
            str(payload.get("recommended_profile") or "air_safe"),
            payload.get("recommended_env_overrides") if isinstance(payload.get("recommended_env_overrides"), dict) else {},
        )
        payload = memory_src.build_payload(project_root, action=action, override_path=override_path, changed=changed)
    return {
        "overall_status": str(payload.get("overall_status") or "ready"),
        "recommended_profile": str(payload.get("recommended_profile") or ""),
        "changed": bool(payload.get("changed", False)),
        "reasons": payload.get("reasons") if isinstance(payload.get("reasons"), list) else [],
        "creative_session": payload.get("creative_session") if isinstance(payload.get("creative_session"), dict) else {},
        "co_running_session": payload.get("co_running_session") if isinstance(payload.get("co_running_session"), dict) else {},
        "override_path": str(override_path),
        "override_exists": bool(override_path.exists()),
        "recommended_env_overrides": payload.get("recommended_env_overrides") if isinstance(payload.get("recommended_env_overrides"), dict) else {},
    }


def build_payload(
    project_root: Path = PROJECT_ROOT,
    *,
    apply: bool,
    override_path: Path,
    state_path: Path = DEFAULT_STATE_PATH,
    cooldown_seconds: int = 600,
    lightweight: bool = False,
) -> dict[str, Any]:
    now = datetime.now(timezone.utc)
    if lightweight:
        resource_snapshot = _refresh_lightweight_creative_snapshot(
            apply=apply,
            state_path=state_path,
            cooldown_seconds=cooldown_seconds,
            now=now,
        )
    else:
        resource_snapshot = _refresh_resource_guard_snapshot(
            project_root,
            apply=apply,
            state_path=state_path,
            cooldown_seconds=cooldown_seconds,
            now=now,
        )
    load1, load5, load15 = os.getloadavg()
    runtime_throttle = load_json(project_root / "governance" / "health" / "runtime_throttle_control_latest.json")
    if lightweight:
        previous = load_json(DEFAULT_OUT_PATH)
        previous_memory = previous.get("memory_efficiency") if isinstance(previous.get("memory_efficiency"), dict) else {}
        previous_paper = previous.get("paper_execution_lane") if isinstance(previous.get("paper_execution_lane"), dict) else {}
        level = _creative_level(resource_snapshot)
        kind = _creative_kind(resource_snapshot)
        creative_active = bool(
            _creative_active(resource_snapshot)
            or bool(resource_snapshot.get("creative_cooldown_active", False))
            or level == "cooldown"
            or kind == "cooldown"
        )
        heavy_pause = {
            "active": creative_active,
            "apply": False,
            "terminate_processes": False,
            "action": "lightweight_pause_contract_refresh" if creative_active else "observe",
            "patterns": [],
            "match_count": 0,
            "matches": [],
            "remaining_match_count": 0,
            "remaining_matches": [],
            "terminated": [],
            "terminated_count": 0,
            "lightweight": True,
        }
        pause_contract = _pause_contract(project_root, snapshot=resource_snapshot, pause_result=heavy_pause, now=now)
        load1_per_core = float(load1) / max(os.cpu_count() or 1, 1)
        return {
            "timestamp_utc": iso_now(),
            "schema_version": 1,
            "ok": True,
            "overall_status": "ready",
            "apply_mode": bool(apply),
            "lightweight": True,
            "host_load": {
                "cpu_count": max(os.cpu_count() or 1, 1),
                "load1": round(float(load1), 3),
                "load5": round(float(load5), 3),
                "load15": round(float(load15), 3),
                "load1_per_core": round(load1_per_core, 3),
            },
            "memory_efficiency": previous_memory,
            "creative_mode": {
                "active": creative_active,
                "level": level,
                "kind": kind,
                "apps": resource_snapshot.get("creative_apps") if isinstance(resource_snapshot.get("creative_apps"), list) else [],
                "cooldown_active": bool(resource_snapshot.get("creative_cooldown_active", False)),
                "cooldown_remaining_seconds": float(resource_snapshot.get("creative_cooldown_remaining_seconds", 0.0) or 0.0),
                "audio_regression_guard_active": False,
                "audio_regression_guard_reason": "",
            },
            "heavy_research_pause": heavy_pause,
            "pause_contract": pause_contract,
            "notification": (
                (resource_snapshot.get("creative_guard_state") or {}).get("notification")
                if isinstance(resource_snapshot.get("creative_guard_state"), dict)
                else {}
            ),
            "paper_execution_lane": previous_paper,
            "runtime_throttle": {
                "overall_status": str(runtime_throttle.get("overall_status") or ""),
                "throttle_profile": str(runtime_throttle.get("throttle_profile") or ""),
                "host_saturation_score": float(runtime_throttle.get("host_saturation_score") or 0.0),
            },
            "actions": ["lightweight_pause_contract_refreshed"],
            "controller_contract": {
                "mode": "assistive",
                "safe_while_live": True,
                "lightweight_refresh": True,
            },
            "source_files": {
                "resource_guard": str(DEFAULT_RESOURCE_GUARD_PATH),
                "memory_efficiency_control": str(project_root / "governance" / "health" / "memory_efficiency_control_latest.json"),
                "runtime_throttle_control": str(project_root / "governance" / "health" / "runtime_throttle_control_latest.json"),
                "creative_guard_state": str(state_path),
                "heavy_research_pause": str(DEFAULT_PAUSE_PATH),
            },
        }
    memory_efficiency = _memory_efficiency_snapshot(project_root, override_path=override_path, apply=apply)
    paper_lane = _paper_lane_snapshot(project_root, apply=apply)
    load1_per_core = float(load1) / max(os.cpu_count() or 1, 1)
    host_saturation_score = float(runtime_throttle.get("host_saturation_score") or 0.0)
    creative_session = memory_efficiency.get("creative_session") if isinstance(memory_efficiency.get("creative_session"), dict) else {}
    creative_active = bool(creative_session.get("active", False)) or str(creative_session.get("level") or "") == "cooldown"
    creative_kind = str(creative_session.get("kind") or "").strip().lower()
    creative_level = str(creative_session.get("level") or "").strip().lower()
    soft_audio_playback = creative_kind == "music_playback" and creative_level == "active"
    audio_regression_guard = bool(
        soft_audio_playback
        and (
            load1_per_core >= 0.55
            or host_saturation_score >= 55.0
            or str(runtime_throttle.get("overall_status") or "").strip().lower() in {"degraded", "blocked", "critical"}
        )
    )
    heavy_pause = _pause_heavy_research(
        apply=apply,
        active=creative_active,
        terminate_processes=bool(creative_active and (not soft_audio_playback or audio_regression_guard)),
    )
    pause_contract = _pause_contract(project_root, snapshot=resource_snapshot, pause_result=heavy_pause, now=now)

    paper_status = "ready"
    if int(paper_lane.get("count_after", 0) or 0) <= 0:
        paper_status = "needs_work"
    elif int(paper_lane.get("count_after", 0) or 0) > 1:
        paper_status = "blocked"

    overall_status = _merge_status(
        str(memory_efficiency.get("overall_status") or "ready"),
        paper_status,
    )

    actions: list[str] = []
    if bool(memory_efficiency.get("changed", False)):
        actions.append("memory_efficiency_override_updated")
    if int(paper_lane.get("count_before", 0) or 0) > 1:
        actions.append("paper_execution_lane_deduped" if apply else "paper_execution_lane_duplicates_detected")
    if int(paper_lane.get("count_after", 0) or 0) <= 0:
        actions.append("paper_execution_lane_missing")
    if bool(heavy_pause.get("active", False)):
        actions.append("heavy_research_pause_active")
    if int(heavy_pause.get("terminated_count", 0) or 0) > 0:
        actions.append("heavy_research_processes_paused")
    if audio_regression_guard:
        actions.append("music_audio_regression_guard_active")
    notification = ((resource_snapshot.get("creative_guard_state") or {}).get("notification") if isinstance(resource_snapshot.get("creative_guard_state"), dict) else {})
    if isinstance(notification, dict) and notification:
        actions.append(str(notification.get("event") or "creative_notification"))

    return {
        "timestamp_utc": iso_now(),
        "schema_version": 1,
        "ok": overall_status in {"ready", "needs_work"},
        "overall_status": overall_status,
        "apply_mode": bool(apply),
        "host_load": {
            "cpu_count": max(os.cpu_count() or 1, 1),
            "load1": round(float(load1), 3),
            "load5": round(float(load5), 3),
            "load15": round(float(load15), 3),
            "load1_per_core": round(load1_per_core, 3),
        },
        "memory_efficiency": memory_efficiency,
        "creative_mode": {
            "active": creative_active,
            "level": str(creative_session.get("level") or "none"),
            "kind": str(creative_session.get("kind") or "none"),
            "apps": creative_session.get("apps") if isinstance(creative_session.get("apps"), list) else [],
            "cooldown_active": bool(creative_session.get("cooldown_active", False)),
            "cooldown_remaining_seconds": float(creative_session.get("cooldown_remaining_seconds", 0.0) or 0.0),
            "audio_regression_guard_active": audio_regression_guard,
            "audio_regression_guard_reason": "host_saturation_or_load_above_music_safe_envelope" if audio_regression_guard else "",
        },
        "heavy_research_pause": heavy_pause,
        "pause_contract": pause_contract,
        "notification": notification if isinstance(notification, dict) else {},
        "paper_execution_lane": paper_lane,
        "runtime_throttle": {
            "overall_status": str(runtime_throttle.get("overall_status") or ""),
            "throttle_profile": str(runtime_throttle.get("throttle_profile") or ""),
            "host_saturation_score": float(runtime_throttle.get("host_saturation_score") or 0.0),
        },
        "actions": actions,
        "controller_contract": {
            "mode": "assistive",
            "safe_while_live": True,
            "scope": [
                "instant_creative_detection",
                "app_specific_creative_profiles",
                "post_app_cooldown",
                "heavy_research_pause",
                "memory_efficiency_override",
                "paper_execution_lane_dedupe",
                "creative_mode_notification",
                "audio_playback_protection",
                "audio_regression_guard",
            ],
        },
        "upgrade_track": {
            "family": "infrabots",
            "upgradeable": True,
            "current_generation": "creative_cotenant_guard_v2",
            "future_upgrade_paths": [
                "launchd-aware duplicate lane cleanup with parent-process affinity",
                "automatic opsctl reload when memory profile flips materially",
                "per-app adaptive budgets based on Logic audio buffer size and Final Cut export state",
                "audio playback first-run protection for Music and legacy iTunes",
            ],
        },
        "source_files": {
            "resource_guard": str(DEFAULT_RESOURCE_GUARD_PATH),
            "memory_efficiency_control": str(project_root / "governance" / "health" / "memory_efficiency_control_latest.json"),
            "runtime_throttle_control": str(project_root / "governance" / "health" / "runtime_throttle_control_latest.json"),
            "creative_guard_state": str(state_path),
            "heavy_research_pause": str(DEFAULT_PAUSE_PATH),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Guard creative co-tenant sessions by keeping memory-efficiency overrides fresh and collapsing duplicate paper execution lanes."
    )
    parser.add_argument("action", choices=("status", "apply"))
    parser.add_argument("--override-file", default=str(memory_src.DEFAULT_OVERRIDE))
    parser.add_argument("--state-file", default=str(DEFAULT_STATE_PATH))
    parser.add_argument("--cooldown-seconds", type=int, default=int(os.getenv("CREATIVE_COTENANT_COOLDOWN_SECONDS", "600")))
    parser.add_argument("--out-file", default=str(DEFAULT_OUT_PATH))
    parser.add_argument("--lightweight", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    override_path = Path(args.override_file).expanduser()
    payload = build_payload(
        PROJECT_ROOT,
        apply=args.action == "apply",
        override_path=override_path,
        state_path=Path(args.state_file).expanduser(),
        cooldown_seconds=int(args.cooldown_seconds),
        lightweight=bool(args.lightweight),
    )
    out_path = Path(args.out_file).expanduser()
    write_payload(out_path, payload)

    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        print(
            "creative_cotenant_guard "
            f"status={payload['overall_status']} "
            f"profile={payload.get('memory_efficiency', {}).get('recommended_profile', '')} "
            f"paper_count={payload.get('paper_execution_lane', {}).get('count_after', 0)}"
        )
    return 0 if bool(payload.get("ok", False)) else 2


if __name__ == "__main__":
    raise SystemExit(main())
