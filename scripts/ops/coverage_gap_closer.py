#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, time as dt_time, timezone
from pathlib import Path
from typing import Any

try:
    from zoneinfo import ZoneInfo
except Exception:
    ZoneInfo = None


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_PATH = PROJECT_ROOT / "governance" / "walk_forward" / "coverage_gap_closer_latest.json"
DEFAULT_QUEUE_PATH = PROJECT_ROOT / "governance" / "walk_forward" / "coverage_gap_closer_queue.jsonl"
LOCAL_TZ = ZoneInfo("America/New_York") if ZoneInfo is not None else timezone.utc
OFF_HOURS_START = dt_time(16, 15)
OFF_HOURS_END = dt_time(9, 20)
PREFLIGHT_REPAIR_ACTIONS = {
    "rebuild_model_artifact",
    "recover_training_log",
    "refresh_training_diagnostics",
    "targeted_retrain",
}


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


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


def _ordered_unique(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for raw in items:
        text = str(raw or "").strip().lower()
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out


def _needs_coverage_preflight_repair(row: dict[str, Any]) -> bool:
    actions = {str(raw or "").strip() for raw in list(row.get("actions") or []) if str(raw or "").strip()}
    return bool(actions.intersection(PREFLIGHT_REPAIR_ACTIONS))


def _off_hours_window(now_utc: datetime | None = None) -> dict[str, Any]:
    current = (now_utc or datetime.now(timezone.utc)).astimezone(LOCAL_TZ)
    local_clock = current.timetz().replace(tzinfo=None)
    is_weekend = current.weekday() >= 5
    active = bool(is_weekend or local_clock >= OFF_HOURS_START or local_clock < OFF_HOURS_END)
    return {
        "active": active,
        "is_weekend": is_weekend,
        "timezone": "America/New_York",
        "local_time": current.isoformat(),
        "window_start_local": OFF_HOURS_START.strftime("%H:%M"),
        "window_end_local": OFF_HOURS_END.strftime("%H:%M"),
        "label": "off_hours" if active else "market_hours",
    }


def _autopilot_contract(
    project_root: Path,
    *,
    active_stage: list[dict[str, Any]],
    backup_candidates: list[dict[str, Any]],
    in_flight: list[dict[str, Any]],
    readiness: dict[str, Any],
    retrain_profile: str,
) -> dict[str, Any]:
    health_root = project_root / "governance" / "health"
    training_runtime = _load_json(health_root / "training_runtime_control_latest.json")
    resource_guard = _load_json(health_root / "resource_guard_latest.json")
    runtime_separation = _load_json(health_root / "live_runtime_separation_control_latest.json")
    off_hours = _off_hours_window()

    shortfall = _safe_int(readiness.get("coverage_shortfall_bots"), 0)
    stage_count = len(active_stage)
    backup_count = len(backup_candidates)
    inflight_retrain_count = len(in_flight)
    repair_required_count = sum(1 for row in active_stage if bool(row.get("needs_runtime_input_repair", False)))
    preflight_repair_required_count = sum(1 for row in active_stage if _needs_coverage_preflight_repair(row))
    remaining_run_budget = sum(max(_safe_int(row.get("runs_remaining"), 0), 0) for row in active_stage)
    snapshot_ready = bool(training_runtime.get("snapshot_ready", False))
    training_runtime_blocked = str(training_runtime.get("overall_status") or "").strip().lower() == "blocked"
    coverage_repair_ready = bool(training_runtime.get("coverage_repair_ready", False))
    swap_used_gb = _safe_float(resource_guard.get("swap_used_gb"), 0.0)
    swap_pressure_elevated = swap_used_gb >= 8.0
    runtime_separation_status = str(runtime_separation.get("overall_status") or "").strip().lower()
    live_lane_should_be_read_only = bool(
        ((runtime_separation.get("release_contract") or {}).get("live_lane_should_be_read_only", False))
    )
    cold_lane_refresh = ((runtime_separation.get("clearance_plan") or {}).get("cold_lane_refresh") or {})
    cold_lane_ready = str(cold_lane_refresh.get("overall_status") or "").strip().lower() in {"ready", "degraded"}
    auto_launch_preconditions_met = (
        shortfall > 0
        and stage_count > 0
        and repair_required_count <= 0
        and preflight_repair_required_count <= 0
        and inflight_retrain_count <= 0
        and snapshot_ready
        and (coverage_repair_ready or not training_runtime_blocked)
        and cold_lane_ready
        and live_lane_should_be_read_only
    )
    auto_launch_off_hours_ready = bool(auto_launch_preconditions_met and not swap_pressure_elevated and bool(off_hours.get("active", False)))
    auto_launch_pending = bool(auto_launch_preconditions_met and not swap_pressure_elevated and not bool(off_hours.get("active", False)))

    gating_signals = {
        "coverage_shortfall_present": shortfall > 0,
        "snapshot_ready": snapshot_ready,
        "training_runtime_blocked": training_runtime_blocked,
        "coverage_repair_ready": coverage_repair_ready,
        "swap_pressure_elevated": swap_pressure_elevated,
        "live_runtime_blocked": runtime_separation_status == "blocked",
        "inflight_retrain_present": inflight_retrain_count > 0,
        "staged_candidates_present": stage_count > 0,
        "off_hours_active": bool(off_hours.get("active", False)),
        "cold_lane_ready": cold_lane_ready,
    }
    blocking_reasons = _ordered_unique(
        [
            "coverage_cleared" if shortfall <= 0 else "",
            "awaiting_candidates" if shortfall > 0 and stage_count <= 0 else "",
            "runtime_input_repair_required" if shortfall > 0 and stage_count > 0 and repair_required_count > 0 else "",
            "coverage_preflight_repair_required" if shortfall > 0 and stage_count > 0 and preflight_repair_required_count > 0 else "",
            "waiting_for_idle" if inflight_retrain_count > 0 else "",
            "training_runtime_blocked" if training_runtime_blocked and not coverage_repair_ready else "",
            "swap_pressure_elevated" if swap_pressure_elevated else "",
            "live_runtime_blocked" if runtime_separation_status == "blocked" else "",
            "snapshot_not_ready" if shortfall > 0 and not snapshot_ready else "",
        ]
    )
    off_hours_preferred = shortfall > 0 and (
        training_runtime_blocked or runtime_separation_status == "blocked" or swap_pressure_elevated or live_lane_should_be_read_only
    )
    can_apply_stage = shortfall > 0 and stage_count > 0
    can_launch_now = (
        shortfall > 0
        and stage_count > 0
        and repair_required_count <= 0
        and preflight_repair_required_count <= 0
        and inflight_retrain_count <= 0
        and snapshot_ready
        and (coverage_repair_ready or not training_runtime_blocked)
        and not swap_pressure_elevated
        and (runtime_separation_status != "blocked" or auto_launch_off_hours_ready)
    )

    launch_state = "ready_to_launch"
    overall_status = "ready"
    next_action = "launch staged coverage cycles under the lighter coverage canary profile"
    if shortfall <= 0:
        launch_state = "cleared"
        next_action = "coverage debt is cleared; keep the seed queue warm and let promotion gating proceed"
    elif stage_count <= 0:
        launch_state = "awaiting_candidates"
        overall_status = "blocked"
        next_action = "refresh the coverage seed queue and stage the next non-infrastructure candidates"
    elif repair_required_count > 0:
        launch_state = "runtime_input_repair_required"
        overall_status = "degraded"
        next_action = "repair runtime inputs for staged candidates before launching coverage cycles"
    elif preflight_repair_required_count > 0:
        launch_state = "coverage_preflight_repair_required"
        overall_status = "degraded"
        next_action = "repair model/log/diagnostic preflight gaps for staged candidates before launching coverage cycles"
    elif inflight_retrain_count > 0:
        launch_state = "waiting_for_idle"
        overall_status = "degraded"
        next_action = "wait for the active retrain to finish, then relaunch the staged coverage pass"
    elif auto_launch_off_hours_ready:
        launch_state = "auto_launch_off_hours_ready"
        overall_status = "ready"
        next_action = "the off-hours window is open and the cold lane is clear enough to launch staged coverage cycles without reopening the live lane"
    elif auto_launch_pending:
        launch_state = "armed_for_off_hours_auto_launch"
        overall_status = "degraded"
        next_action = "leave the staged candidates armed so the next off-hours window can launch coverage cycles without reopening the live lane"
    elif training_runtime_blocked or runtime_separation_status == "blocked" or swap_pressure_elevated:
        launch_state = "stage_only_off_hours"
        overall_status = "degraded"
        next_action = "keep the candidates staged now, but launch coverage cycles in the cold lane once shared-host pressure drops"
    elif not snapshot_ready:
        launch_state = "snapshot_repair_required"
        overall_status = "degraded"
        next_action = "refresh the shared training snapshot before launching more walk-forward coverage cycles"

    return {
        "overall_status": overall_status,
        "launch_state": launch_state,
        "stage_candidate_count": stage_count,
        "backup_candidate_count": backup_count,
        "repair_required_count": repair_required_count,
        "preflight_repair_required_count": preflight_repair_required_count,
        "remaining_run_budget": remaining_run_budget,
        "inflight_retrain_count": inflight_retrain_count,
        "snapshot_ready": snapshot_ready,
        "off_hours_preferred": off_hours_preferred,
        "off_hours_window": off_hours,
        "hold_live_lane_read_only": live_lane_should_be_read_only,
        "cold_lane_ready": cold_lane_ready,
        "can_apply_stage": can_apply_stage,
        "can_launch_now": can_launch_now,
        "can_auto_launch_off_hours": auto_launch_off_hours_ready,
        "auto_launch_pending": auto_launch_pending,
        "launch_mode": ("off_hours_cold_lane" if auto_launch_off_hours_ready else "manual"),
        "next_action": next_action,
        "blocking_reasons": blocking_reasons,
        "gating_signals": gating_signals,
        "launch_contract": {
            "auto_launch_preconditions_met": bool(auto_launch_preconditions_met),
            "auto_launch_pending": auto_launch_pending,
            "window_active": bool(off_hours.get("active", False)),
            "window_label": str(off_hours.get("label") or ""),
            "window_start_local": str(off_hours.get("window_start_local") or ""),
            "window_end_local": str(off_hours.get("window_end_local") or ""),
            "launch_guard": ("off_hours_only" if off_hours_preferred else "runtime_clear"),
            "coverage_repair_ready": coverage_repair_ready,
            "preflight_repair_required_count": preflight_repair_required_count,
        },
        "recommended_commands": {
            "stage_only": [
                "./scripts/ops/opsctl.sh",
                "coverage-gap-closer",
                "--apply-stage",
                "--json",
            ],
            "launch_off_hours": [
                "./scripts/ops/opsctl.sh",
                "coverage-gap-closer",
                "--apply-stage",
                "--launch",
                "--retrain-profile",
                str(retrain_profile or "coverage_canary"),
                "--json",
            ],
            "auto_launch_off_hours": [
                "./scripts/ops/opsctl.sh",
                "coverage-gap-closer",
                "--apply-stage",
                "--auto-launch-off-hours",
                "--retrain-profile",
                str(retrain_profile or "coverage_canary"),
                "--json",
            ],
        },
    }


def _load_registry_payload(path: Path) -> dict[str, Any]:
    payload = _load_json(path)
    if not isinstance(payload.get("sub_bots"), list):
        payload["sub_bots"] = []
    return payload


def _registry_rows(path: Path) -> dict[str, dict[str, Any]]:
    payload = _load_registry_payload(path)
    out: dict[str, dict[str, Any]] = {}
    for row in payload.get("sub_bots") or []:
        if not isinstance(row, dict):
            continue
        bot_id = str(row.get("bot_id") or "").strip().lower()
        if bot_id:
            out[bot_id] = row
    return out


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def _run_json(cmd: list[str], *, cwd: Path, timeout_sec: int, env: dict[str, str] | None = None) -> dict[str, Any]:
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(cwd),
            env=env,
            capture_output=True,
            text=True,
            check=False,
            timeout=max(int(timeout_sec), 1),
        )
        stdout = proc.stdout or ""
        stderr = proc.stderr or ""
        rc = int(proc.returncode)
        timed_out = False
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout.decode("utf-8", errors="ignore") if isinstance(exc.stdout, bytes) else str(exc.stdout or "")
        stderr = exc.stderr.decode("utf-8", errors="ignore") if isinstance(exc.stderr, bytes) else str(exc.stderr or "")
        rc = 124
        timed_out = True
    payload = {}
    for raw in reversed([line.strip() for line in stdout.splitlines() if line.strip()]):
        try:
            parsed = json.loads(raw)
        except Exception:
            continue
        if isinstance(parsed, dict):
            payload = parsed
            break
    return {
        "cmd": list(cmd),
        "rc": rc,
        "timed_out": timed_out,
        "stdout_tail": "\n".join(stdout.splitlines()[-12:]),
        "stderr_tail": "\n".join(stderr.splitlines()[-12:]),
        "payload": payload,
    }


def _active_retrain_processes(project_root: Path) -> list[dict[str, Any]]:
    try:
        proc = subprocess.run(
            ["ps", "-axo", "pid,command"],
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
    except Exception:
        return []
    matches: list[dict[str, Any]] = []
    current_pid = os.getpid()
    for raw in (proc.stdout or "").splitlines():
        line = raw.strip()
        if not line:
            continue
        parts = line.split(None, 1)
        if len(parts) != 2:
            continue
        try:
            pid = int(parts[0])
        except Exception:
            continue
        cmd = parts[1]
        if pid == current_pid:
            continue
        if "scripts/weekly_retrain.py" not in cmd:
            continue
        if "coverage_gap_closer.py" in cmd:
            continue
        if str(project_root) not in cmd:
            if project_root.name != "schwab_trading_bot" or "schwab_trading_bot" not in cmd:
                continue
        matches.append({"pid": pid, "command": cmd})
    return matches


def _refresh_artifacts(project_root: Path, *, python_bin: Path, timeout_sec: int) -> list[dict[str, Any]]:
    steps = [
        [str(python_bin), str(project_root / "scripts" / "ops" / "training_requalification_lane.py"), "--write-queue", "--json"],
        [str(python_bin), str(project_root / "scripts" / "ops" / "walk_forward_coverage_seed.py"), "--write-queue", "--json"],
        [str(python_bin), str(project_root / "scripts" / "walk_forward_validate.py")],
        [str(python_bin), str(project_root / "scripts" / "walk_forward_promotion_gate.py")],
        [str(python_bin), str(project_root / "scripts" / "lane_promotion_gate.py"), "--json"],
        [str(python_bin), str(project_root / "scripts" / "promotion_readiness_summary.py"), "--json"],
        [str(python_bin), str(project_root / "scripts" / "ops" / "training_quality_control.py"), "--json"],
    ]
    return [_run_json(cmd, cwd=project_root, timeout_sec=timeout_sec) for cmd in steps]


def _load_walk_forward_runs(project_root: Path) -> dict[str, int]:
    payload = _load_json(project_root / "governance" / "walk_forward" / "walk_forward_latest.json")
    rows = payload.get("bots") if isinstance(payload.get("bots"), dict) else {}
    out: dict[str, int] = {}
    for bot_id, row in rows.items():
        text = str(bot_id or "").strip().lower()
        if text and isinstance(row, dict):
            out[text] = _safe_int(row.get("runs"), 0)
    return out


def _diagnostic_status_rank(project_root: Path, bot_id: str) -> int:
    text = str(bot_id or "").strip()
    if not text:
        return 6
    payload = _load_json(project_root / "governance" / "training_diagnostics" / f"{text}_latest.json")
    status = str(payload.get("status") or "").strip().lower()
    sample_count = _safe_int(payload.get("sample_count"), 0)
    if status == "passed":
        return 0
    if status == "deferred_sample_starved" and sample_count > 0:
        return 1
    if sample_count > 0 and status not in {"failed"}:
        return 2
    if status == "failed" and sample_count > 0:
        return 3
    if status == "deferred_sample_starved":
        return 4
    if status == "failed":
        return 5
    return 6


def _candidate_pool(project_root: Path, *, candidate_limit: int, stage_count: int) -> dict[str, Any]:
    coverage_seed = _load_json(project_root / "governance" / "walk_forward" / "coverage_seed_latest.json")
    readiness = _load_json(project_root / "governance" / "walk_forward" / "promotion_readiness_latest.json")
    rows = coverage_seed.get("seed_queue") if isinstance(coverage_seed.get("seed_queue"), list) else []
    ranked = [row for row in rows if isinstance(row, dict)]

    def _coverage_lane_priority(row: dict[str, Any]) -> int:
        role = str(row.get("bot_role") or "").strip().lower()
        queue_bucket = str(row.get("queue_bucket") or "").strip().lower()
        if role == "signal_sub_bot" or queue_bucket == "signal":
            return 0
        if queue_bucket == "general":
            return 1
        if role == "options_sub_bot" or queue_bucket == "options":
            return 2
        if role == "futures_sub_bot":
            return 3
        if role == "infrastructure_sub_bot" or queue_bucket == "infrastructure":
            return 9
        return 4

    ranked = sorted(
        ranked,
        key=lambda row: (
            0 if bool(row.get("strong_seed_candidate", False)) else 1,
            _safe_int(row.get("needs_runtime_input_repair"), 0),
            _coverage_lane_priority(row),
            _diagnostic_status_rank(project_root, str(row.get("bot_id") or "")),
            -_safe_int(row.get("current_runs"), 0),
            -_safe_float(row.get("priority"), 0.0),
            str(row.get("bot_id") or ""),
        ),
    )
    ranked = ranked[: max(int(candidate_limit), int(stage_count), 1)]
    active_stage: list[dict[str, Any]] = []
    deferred_stage: list[dict[str, Any]] = []
    backups: list[dict[str, Any]] = []
    for row in ranked:
        role = str(row.get("bot_role") or "")
        strong_seed_candidate = bool(row.get("strong_seed_candidate", False))
        if role == "infrastructure_sub_bot" and not strong_seed_candidate:
            continue
        actions = row.get("actions") if isinstance(row.get("actions"), list) else []
        coverage_stage_ready = (
            (not actions)
            or ("seed_walk_forward_coverage" in actions)
            or (
                strong_seed_candidate
                and "recover_training_log" not in actions
                and "refresh_training_diagnostics" not in actions
            )
            or (
                "rebuild_model_artifact" in actions
                and "recover_training_log" not in actions
                and "refresh_training_diagnostics" not in actions
            )
        )
        if coverage_stage_ready:
            bucket = active_stage if len(active_stage) < max(int(stage_count), 1) else deferred_stage
            bucket.append(row)
        else:
            backups.append(row)
    if not active_stage and backups:
        promoted_backups: list[dict[str, Any]] = []
        remaining_backups: list[dict[str, Any]] = []
        for row in backups:
            role = str(row.get("bot_role") or "").strip().lower()
            if role == "infrastructure_sub_bot" and not bool(row.get("strong_seed_candidate", False)):
                remaining_backups.append(row)
                continue
            if len(promoted_backups) < max(int(stage_count), 1):
                promoted = dict(row)
                promoted["coverage_stage_kind"] = (
                    "runtime_input_repair_required"
                    if bool(promoted.get("needs_runtime_input_repair", False))
                    else "coverage_preflight_repair_required"
                )
                promoted_backups.append(promoted)
            else:
                remaining_backups.append(row)
        if promoted_backups:
            active_stage = promoted_backups
            backups = remaining_backups
    min_considered_bots = max(
        _safe_int((readiness.get("thresholds") or {}).get("min_considered_bots"), 4),
        1,
    )
    return {
        "coverage_seed": coverage_seed,
        "promotion_readiness": readiness,
        "min_considered_bots": min_considered_bots,
        "active_stage": active_stage[: max(int(stage_count), min_considered_bots, 1)],
        "backup_candidates": deferred_stage + backups,
    }


def stage_registry_candidates(
    project_root: Path,
    *,
    candidate_rows: list[dict[str, Any]],
    clear_others: bool = True,
) -> dict[str, Any]:
    registry_path = project_root / "master_bot_registry.json"
    payload = _load_registry_payload(registry_path)
    rows = payload.get("sub_bots") if isinstance(payload.get("sub_bots"), list) else []
    candidate_map = {
        str(row.get("bot_id") or "").strip().lower(): row
        for row in candidate_rows
        if isinstance(row, dict) and str(row.get("bot_id") or "").strip()
    }
    changed = False
    backup_path = ""
    staged: list[str] = []
    cleared: list[str] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        bot_id = str(row.get("bot_id") or "").strip().lower()
        if not bot_id:
            continue
        should_stage = bot_id in candidate_map
        was_staged = bool(row.get("coverage_candidate_active", False))
        if should_stage:
            candidate = candidate_map[bot_id]
            updates = {
                "coverage_candidate_active": True,
                "coverage_stage": "promotion_queue",
                "coverage_candidate_reason": "coverage_gap_closer",
                "coverage_candidate_priority": round(_safe_float(candidate.get("priority"), 0.0), 6),
                "coverage_candidate_started_utc": _iso_now(),
            }
            for key, value in updates.items():
                if row.get(key) != value:
                    row[key] = value
                    changed = True
            staged.append(bot_id)
        elif clear_others and was_staged:
            for key in (
                "coverage_candidate_active",
                "coverage_stage",
                "coverage_candidate_reason",
                "coverage_candidate_priority",
                "coverage_candidate_started_utc",
            ):
                if key in row:
                    row.pop(key, None)
                    changed = True
            cleared.append(bot_id)
    if changed and registry_path.exists():
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        backup = project_root / "governance" / "lifecycle" / f"master_bot_registry.coverage_gap_stage_backup_{stamp}.json"
        backup.parent.mkdir(parents=True, exist_ok=True)
        backup.write_text(registry_path.read_text(encoding="utf-8"), encoding="utf-8")
        backup_path = str(backup)
        _write_json(registry_path, payload)
    return {
        "registry_path": str(registry_path),
        "registry_updated": bool(changed),
        "registry_backup_path": backup_path,
        "staged_bot_ids": staged,
        "cleared_bot_ids": cleared,
    }


def _build_payload(
    project_root: Path,
    *,
    candidate_limit: int,
    stage_count: int,
    retrain_profile: str,
    active_stage_candidates: list[dict[str, Any]] | None = None,
    backup_candidates: list[dict[str, Any]] | None = None,
    cycle_records: list[dict[str, Any]] | None = None,
    stage_result: dict[str, Any] | None = None,
    refresh_attempts: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    pool = _candidate_pool(project_root, candidate_limit=candidate_limit, stage_count=stage_count)
    active_stage = list(active_stage_candidates) if active_stage_candidates is not None else list(pool.get("active_stage") or [])
    backups = list(backup_candidates) if backup_candidates is not None else list(pool.get("backup_candidates") or [])
    readiness = pool.get("promotion_readiness") if isinstance(pool.get("promotion_readiness"), dict) else {}
    in_flight = _active_retrain_processes(project_root)
    autopilot_contract = _autopilot_contract(
        project_root,
        active_stage=active_stage,
        backup_candidates=backups,
        in_flight=in_flight,
        readiness=readiness,
        retrain_profile=retrain_profile,
    )
    max_cycles = max((_safe_int(row.get("runs_remaining"), 0) for row in active_stage), default=0)
    overall_status = "needs_cycles"
    if _safe_int(readiness.get("coverage_shortfall_bots"), 0) <= 0:
        overall_status = "cleared"
    elif in_flight:
        overall_status = "waiting_for_idle"
    payload = {
        "timestamp_utc": _iso_now(),
        "schema_version": 1,
        "ok": _safe_int(readiness.get("coverage_shortfall_bots"), 0) <= 0,
        "overall_status": overall_status,
        "coverage_shortfall_bots": _safe_int(readiness.get("coverage_shortfall_bots"), 0),
        "considered_bots": _safe_int(readiness.get("considered_bots"), 0),
        "min_considered_bots": _safe_int(pool.get("min_considered_bots"), 4),
        "staged_candidate_count": len(active_stage),
        "backup_candidate_count": len(backups),
        "active_stage_candidates": active_stage,
        "backup_candidates": backups,
        "recommended_cycle_budget": int(max_cycles),
        "inflight_retrain_processes": in_flight,
        "autopilot_contract": autopilot_contract,
        "recommended_command": [
            str(project_root / "scripts" / "ops" / "opsctl.sh"),
            "retrain-force-targeted",
            "--include-bot-ids",
            ",".join(_ordered_unique([str(row.get("bot_id") or "") for row in active_stage])),
            "--retrain-profile",
            str(retrain_profile or "coverage_canary"),
            "--skip-master-update",
        ],
        "recommended_actions": _ordered_unique(
            [
                "stage the top non-infrastructure coverage candidates so promotion gating can count them without forcing them live",
                "prefer the lighter signal candidates before retrying heavier options or dividend candidates",
                "run coverage cycles under the cheaper coverage_canary retrain profile so runtime inputs fail fast instead of inflating into a full canary retrain",
                "let the current retrain finish before launching the coverage pass to avoid memory contention",
                "keep cycling targeted retrains until each staged candidate reaches the promotion run floor",
                "refresh walk-forward and promotion artifacts after every cycle so stalled candidates can be swapped out quickly",
            ]
        ),
    }
    if cycle_records is not None:
        payload["cycle_records"] = cycle_records
    if stage_result is not None:
        payload["stage_result"] = stage_result
    if refresh_attempts is not None:
        payload["refresh_attempts"] = refresh_attempts
    return payload


def _wait_for_retrain_idle(project_root: Path, *, timeout_sec: int, poll_sec: int) -> tuple[bool, list[dict[str, Any]]]:
    deadline = time.time() + max(int(timeout_sec), 1)
    while time.time() < deadline:
        active = _active_retrain_processes(project_root)
        if not active:
            return True, []
        time.sleep(max(int(poll_sec), 1))
    return False, _active_retrain_processes(project_root)


def _rotate_staged_candidates(
    *,
    staged_rows: list[dict[str, Any]],
    backup_rows: list[dict[str, Any]],
    stalled_counts: dict[str, int],
    stall_limit: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    updated_stage = list(staged_rows)
    remaining_backups = list(backup_rows)
    swapped_out: list[str] = []
    for index, row in list(enumerate(updated_stage)):
        bot_id = str(row.get("bot_id") or "").strip().lower()
        if not bot_id:
            continue
        if stalled_counts.get(bot_id, 0) < max(int(stall_limit), 1):
            continue
        if not remaining_backups:
            continue
        replacement = remaining_backups.pop(0)
        swapped_out.append(bot_id)
        updated_stage[index] = replacement
        stalled_counts.pop(bot_id, None)
    return updated_stage, remaining_backups, swapped_out


def _timed_out_batch_bot_ids(retrain_attempt: dict[str, Any], batch_bot_ids: list[str]) -> list[str]:
    batch = _ordered_unique(batch_bot_ids)
    if not batch:
        return []
    if bool(retrain_attempt.get("timed_out", False)) or int(retrain_attempt.get("rc", 0) or 0) == 124:
        return batch

    payload = retrain_attempt.get("payload") if isinstance(retrain_attempt.get("payload"), dict) else {}
    matched: list[str] = []
    for key in ("failure_details", "target_outcomes"):
        rows = payload.get(key) if isinstance(payload.get(key), list) else []
        for row in rows:
            if not isinstance(row, dict):
                continue
            bot_id = str(row.get("bot_id") or "").strip().lower()
            reason_text = " ".join(
                str(row.get(field) or "")
                for field in ("reason", "stdout_tail", "stderr_tail")
            ).lower()
            if bot_id in batch and (int(row.get("rc", 0) or 0) == 124 or "timeout" in reason_text or "exit=124" in reason_text):
                matched.append(bot_id)
    if matched:
        return _ordered_unique(matched)

    tail_text = "\n".join(
        [
            str(retrain_attempt.get("stdout_tail") or ""),
            str(retrain_attempt.get("stderr_tail") or ""),
        ]
    ).lower()
    if "timeout" not in tail_text and "exit=124" not in tail_text:
        return []
    if len(batch) == 1:
        return batch

    for bot_id in batch:
        if bot_id in tail_text or f"{bot_id}.py" in tail_text:
            matched.append(bot_id)
    return _ordered_unique(matched) if matched else batch


def run_gap_closer(
    project_root: Path = PROJECT_ROOT,
    *,
    candidate_limit: int,
    stage_count: int,
    max_cycles: int,
    retrain_timeout_sec: int,
    refresh_timeout_sec: int,
    wait_for_idle_timeout_sec: int,
    poll_sec: int,
    stall_limit: int,
    retrain_profile: str,
    apply_stage: bool,
    launch: bool,
    auto_launch_off_hours: bool,
    clear_other_candidates: bool,
    out_path: Path,
    queue_out_path: Path,
    skip_refresh: bool = False,
) -> dict[str, Any]:
    python_bin = Path(sys.executable)
    refresh_attempts = [] if skip_refresh else _refresh_artifacts(project_root, python_bin=python_bin, timeout_sec=refresh_timeout_sec)
    pool = _candidate_pool(project_root, candidate_limit=candidate_limit, stage_count=stage_count)
    staged_rows = list(pool.get("active_stage") or [])
    backup_rows = list(pool.get("backup_candidates") or [])
    stage_result: dict[str, Any] = {}
    if apply_stage:
        stage_result = stage_registry_candidates(
            project_root,
            candidate_rows=staged_rows,
            clear_others=clear_other_candidates,
        )
    initial_autopilot = _autopilot_contract(
        project_root,
        active_stage=staged_rows,
        backup_candidates=backup_rows,
        in_flight=_active_retrain_processes(project_root),
        readiness=pool.get("promotion_readiness") if isinstance(pool.get("promotion_readiness"), dict) else {},
        retrain_profile=retrain_profile,
    )
    effective_launch = bool(launch or (auto_launch_off_hours and bool(initial_autopilot.get("can_auto_launch_off_hours", False))))
    queue_rows = [
        {
            "cycle_index": cycle_index + 1,
            "bot_id": str(row.get("bot_id") or "").strip().lower(),
            "current_runs": _safe_int(row.get("current_runs"), 0),
            "runs_remaining": _safe_int(row.get("runs_remaining"), 0),
        }
        for row in staged_rows
        for cycle_index in range(max(_safe_int(row.get("runs_remaining"), 0), 1))
        if str(row.get("bot_id") or "").strip()
    ]
    _write_jsonl(queue_out_path, queue_rows)

    cycle_records: list[dict[str, Any]] = []
    stalled_counts: dict[str, int] = {}
    launch_decision = {
        "requested_launch": bool(launch),
        "auto_launch_off_hours_requested": bool(auto_launch_off_hours),
        "effective_launch": effective_launch,
        "launch_mode": str(initial_autopilot.get("launch_mode") or ""),
        "launch_state": str(initial_autopilot.get("launch_state") or ""),
        "auto_launch_pending": bool(((initial_autopilot.get("launch_contract") or {}).get("auto_launch_pending", False))),
        "auto_launch_window_active": bool(((initial_autopilot.get("launch_contract") or {}).get("window_active", False))),
    }
    if effective_launch and staged_rows:
        for cycle_index in range(max(int(max_cycles), 1)):
            readiness = _load_json(project_root / "governance" / "walk_forward" / "promotion_readiness_latest.json")
            if _safe_int(readiness.get("coverage_shortfall_bots"), 0) <= 0:
                break
            ready, blocking = _wait_for_retrain_idle(
                project_root,
                timeout_sec=wait_for_idle_timeout_sec,
                poll_sec=poll_sec,
            )
            if not ready:
                cycle_records.append(
                    {
                        "cycle_index": cycle_index + 1,
                        "status": "waiting_for_idle_timed_out",
                        "blocking_retrain_processes": blocking,
                    }
                )
                break
            prior_runs = _load_walk_forward_runs(project_root)
            batch = [
                str(row.get("bot_id") or "").strip().lower()
                for row in staged_rows
                if str(row.get("bot_id") or "").strip() and _safe_int(row.get("runs_remaining"), 0) > 0
            ]
            batch = batch[: max(int(stage_count), 1)]
            if not batch:
                cycle_records.append({"cycle_index": cycle_index + 1, "status": "no_batch_candidates"})
                break
            retrain_cmd = [
                str(project_root / "scripts" / "ops" / "opsctl.sh"),
                "retrain-force-targeted",
                "--include-bot-ids",
                ",".join(batch),
                "--retrain-profile",
                str(retrain_profile or "coverage_canary"),
                "--skip-master-update",
            ]
            retrain_attempt = _run_json(
                retrain_cmd,
                cwd=project_root,
                timeout_sec=retrain_timeout_sec,
                env={
                    **os.environ,
                    "RETRAIN_TRIGGER_SOURCE": "coverage_gap_closer",
                    "RETRAIN_TRIGGER_LABEL": "coverage_gap_closer",
                    "RETRAIN_TRIGGER_CONTEXT": f"coverage_gap_cycle:{cycle_index + 1}",
                    "RETRAIN_TRIGGER_PROFILE": str(retrain_profile or "coverage_canary"),
                },
            )
            refresh_attempts.extend(_refresh_artifacts(project_root, python_bin=python_bin, timeout_sec=refresh_timeout_sec))
            latest_runs = _load_walk_forward_runs(project_root)
            deltas: dict[str, int] = {}
            for bot_id in batch:
                delta = max(_safe_int(latest_runs.get(bot_id), 0) - _safe_int(prior_runs.get(bot_id), 0), 0)
                deltas[bot_id] = delta
                stalled_counts[bot_id] = 0 if delta > 0 else _safe_int(stalled_counts.get(bot_id), 0) + 1
            timeout_like_bot_ids = _timed_out_batch_bot_ids(retrain_attempt, batch)
            for bot_id in timeout_like_bot_ids:
                stalled_counts[bot_id] = max(_safe_int(stalled_counts.get(bot_id), 0), max(int(stall_limit), 1))
            for row in staged_rows:
                bot_id = str(row.get("bot_id") or "").strip().lower()
                current_runs = _safe_int(latest_runs.get(bot_id), _safe_int(row.get("current_runs"), 0))
                row["current_runs"] = int(current_runs)
                row["runs_remaining"] = max(
                    _safe_int((_load_json(project_root / "governance" / "walk_forward" / "promotion_readiness_latest.json").get("thresholds") or {}).get("min_runs_per_bot"), 12)
                    - current_runs,
                    0,
                )
            staged_rows, backup_rows, swapped_out = _rotate_staged_candidates(
                staged_rows=staged_rows,
                backup_rows=backup_rows,
                stalled_counts=stalled_counts,
                stall_limit=stall_limit,
            )
            if swapped_out and apply_stage:
                stage_result = stage_registry_candidates(
                    project_root,
                    candidate_rows=staged_rows,
                    clear_others=clear_other_candidates,
                )
            cycle_records.append(
                {
                    "cycle_index": cycle_index + 1,
                    "status": "completed" if _safe_int(retrain_attempt.get("rc"), 0) == 0 else "retrain_failed",
                    "batch_bot_ids": batch,
                    "run_deltas": deltas,
                    "timeout_like_bot_ids": timeout_like_bot_ids,
                    "swapped_out_bot_ids": swapped_out,
                    "retrain_attempt": retrain_attempt,
                }
            )
            if _safe_int(_load_json(project_root / "governance" / "walk_forward" / "promotion_readiness_latest.json").get("coverage_shortfall_bots"), 0) <= 0:
                break

    payload = _build_payload(
        project_root,
        candidate_limit=candidate_limit,
        stage_count=stage_count,
        retrain_profile=retrain_profile,
        active_stage_candidates=staged_rows,
        backup_candidates=backup_rows,
        cycle_records=cycle_records,
        stage_result=stage_result,
        refresh_attempts=refresh_attempts,
    )
    payload["launch_decision"] = launch_decision
    _write_json(out_path, payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Stage and cycle the next promotion candidates until the walk-forward coverage gap clears.")
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--out-file", default=str(DEFAULT_OUT_PATH))
    parser.add_argument("--queue-out", default=str(DEFAULT_QUEUE_PATH))
    parser.add_argument("--candidate-limit", type=int, default=8)
    parser.add_argument("--stage-count", type=int, default=4)
    parser.add_argument("--max-cycles", type=int, default=12)
    parser.add_argument("--retrain-timeout-sec", type=int, default=14400)
    parser.add_argument("--refresh-timeout-sec", type=int, default=1800)
    parser.add_argument("--wait-for-idle-timeout-sec", type=int, default=28800)
    parser.add_argument("--poll-sec", type=int, default=30)
    parser.add_argument("--stall-limit", type=int, default=2)
    parser.add_argument("--retrain-profile", default="coverage_canary")
    parser.add_argument("--apply-stage", action="store_true")
    parser.add_argument("--launch", action="store_true")
    parser.add_argument("--auto-launch-off-hours", action="store_true")
    parser.add_argument("--skip-refresh", action="store_true")
    parser.add_argument("--no-clear-other-candidates", dest="clear_other_candidates", action="store_false")
    parser.add_argument("--json", action="store_true")
    parser.set_defaults(clear_other_candidates=True)
    args = parser.parse_args()

    payload = run_gap_closer(
        Path(args.project_root).resolve(),
        candidate_limit=max(int(args.candidate_limit), 1),
        stage_count=max(int(args.stage_count), 1),
        max_cycles=max(int(args.max_cycles), 1),
        retrain_timeout_sec=max(int(args.retrain_timeout_sec), 1),
        refresh_timeout_sec=max(int(args.refresh_timeout_sec), 1),
        wait_for_idle_timeout_sec=max(int(args.wait_for_idle_timeout_sec), 1),
        poll_sec=max(int(args.poll_sec), 1),
        stall_limit=max(int(args.stall_limit), 1),
        retrain_profile=str(args.retrain_profile or "coverage_canary").strip() or "coverage_canary",
        apply_stage=bool(args.apply_stage),
        launch=bool(args.launch),
        auto_launch_off_hours=bool(args.auto_launch_off_hours),
        clear_other_candidates=bool(args.clear_other_candidates),
        skip_refresh=bool(args.skip_refresh),
        out_path=Path(args.out_file).expanduser(),
        queue_out_path=Path(args.queue_out).expanduser(),
    )
    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        print(
            "coverage_gap_closer "
            f"overall_status={str(payload.get('overall_status') or '')} "
            f"coverage_shortfall_bots={int(payload.get('coverage_shortfall_bots', 0) or 0)}"
        )
    return 0 if bool(payload.get("ok", False)) else 2


if __name__ == "__main__":
    raise SystemExit(main())
