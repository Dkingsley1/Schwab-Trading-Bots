#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from core.runtime_python import resolve_runtime_python
    from scripts.ops.long_runtime_common import load_json, ordered_unique, parse_iso_utc, payload_age_minutes, write_payload
else:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    from core.runtime_python import resolve_runtime_python
    from .long_runtime_common import load_json, ordered_unique, parse_iso_utc, payload_age_minutes, write_payload


PY = resolve_runtime_python(PROJECT_ROOT)
DEFAULT_OUT_PATH = PROJECT_ROOT / "governance" / "health" / "soak_self_healing_control_latest.json"
DEFAULT_STATE_PATH = PROJECT_ROOT / "governance" / "health" / "soak_self_healing_state.json"

MANAGED_DAILY_FAILURES = {
    "feature_store_manifest",
    "promotion_packet_builder",
    "promotion_quality_gate",
    "retrain_schema_compatibility_guard",
    "snapshot_coverage_sentinel",
}
MANAGED_SOAK_CONTROLS = {
    "live_plane_ready_cold_lane_refresh_deferred",
    "daily_mobile_operator_coverage_active_without_zero_touch_remote_pager",
}
STORAGE_SOAK_BLOCKERS = {
    "storage_margin_not_30_day_ready",
    "storage_retention_contract_not_ready",
}
INGESTION_SOAK_BLOCKERS = {"ingestion_soak_contract_not_ready"}
HARD_RUNTIME_STEP_NAMES = {
    "session_ready",
    "process_watchdog",
    "schwab_auth_supervisor",
    "livefeed_refresh_guard",
    "runtime_paper_regression_guard",
    "memory_efficiency",
    "storage_resilience",
    "notification_ladder",
    "nightly_resilience",
}
RUNTIME_CONTINUITY_REFRESH_GUARDS = {
    "runtime_ready_advisory_reclassification_contract",
    "runtime_guarded_ready_lane_contract",
    "paper_runtime_capacity_blocker_contract",
    "soak_paper_eligible_lane_open_contract",
    "production_grade_paper_live_authority_contract",
    "soak_30_day_continuity_contract",
}
MANAGED_LIVE_MONEY_LOCK_REASONS = {
    "target_window_not_complete",
    "live_execution_operator_release_required",
}
SAFE_ENV = {
    "MARKET_DATA_ONLY": "1",
    "ALLOW_ORDER_EXECUTION": "0",
    "BOT_LIVE_MONEY_LOCKED_DURING_SOAK": "1",
    "BOT_UNATTENDED_SOAK_ACTIVE": "1",
    "BOT_ALLOW_VIDEO_COLD_ARCHIVE": "1",
    "BOT_VIDEO_COLD_ARCHIVE_ROOT": "/Volumes/VIDEO/schwab_trading_bot_cold",
}
DEFAULT_VIDEO_COLD_ARCHIVE_ROOT = Path("/Volumes/VIDEO/schwab_trading_bot_cold")


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _iso_now() -> str:
    return _utc_now().isoformat()


def _as_dict(raw: Any) -> dict[str, Any]:
    return raw if isinstance(raw, dict) else {}


def _as_list(raw: Any) -> list[Any]:
    return raw if isinstance(raw, list) else []


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


def _path_under(path: Path, root: Path) -> bool:
    raw = str(path.expanduser())
    base = str(root.expanduser())
    return bool(raw == base or raw.startswith(f"{base}/"))


def _configure_cold_archive_env(env: dict[str, str], *, apply: bool) -> dict[str, Any]:
    video_root = Path(env.get("BOT_VIDEO_COLD_ARCHIVE_ROOT", str(DEFAULT_VIDEO_COLD_ARCHIVE_ROOT))).expanduser()
    env["BOT_VIDEO_COLD_ARCHIVE_ROOT"] = str(video_root)
    configured = str(env.get("BOT_SECOND_COLD_ROOT") or "").strip()
    auto_selected = False
    if configured:
        target = Path(configured).expanduser()
    elif video_root.parent.exists():
        target = video_root
        env["BOT_SECOND_COLD_ROOT"] = str(target)
        auto_selected = True
    else:
        return {
            "configured": False,
            "path": "",
            "auto_selected": False,
            "created": False,
            "approved_video_cold_archive": False,
        }

    approved_video = _path_under(target, video_root)
    if approved_video:
        env.setdefault("BOT_ALLOW_VIDEO_COLD_ARCHIVE", "1")
    created = False
    create_error = ""
    if apply and approved_video and target.parent.exists():
        try:
            target.mkdir(parents=True, exist_ok=True)
            created = True
        except Exception as exc:
            create_error = str(exc)
    return {
        "configured": True,
        "path": str(target),
        "auto_selected": auto_selected,
        "created": created,
        "create_error": create_error,
        "approved_video_cold_archive": approved_video
        and str(env.get("BOT_ALLOW_VIDEO_COLD_ARCHIVE") or "").strip().lower() in {"1", "true", "yes", "y", "on"},
        "scope": "cold_archive_subtree_only" if approved_video else "explicit_non_video_cold_target",
    }


def _parse_json_output(text: str) -> dict[str, Any]:
    for raw in reversed([line.strip() for line in str(text or "").splitlines() if line.strip()]):
        try:
            payload = json.loads(raw)
        except Exception:
            continue
        if isinstance(payload, dict):
            return payload
    return {}


def _status(payload: dict[str, Any]) -> str:
    return str(payload.get("overall_status") or payload.get("status") or payload.get("state") or "").strip().lower()


def _payload_ok(payload: dict[str, Any], rc: int) -> bool:
    status = _status(payload)
    if bool(payload.get("ok", False)):
        return True
    if rc == 0 and payload.get("ok") is not False and status not in {"blocked", "critical", "degraded"}:
        return True
    return False


def _command_label(cmd: list[str]) -> str:
    return " ".join(str(item) for item in cmd[:4])


def _run_command(
    cmd: list[str],
    *,
    project_root: Path,
    timeout_sec: int,
    env: dict[str, str],
) -> dict[str, Any]:
    started = _utc_now()
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(project_root),
            env=env,
            capture_output=True,
            text=True,
            timeout=max(int(timeout_sec), 1),
            check=False,
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
    parsed = _parse_json_output(stdout)
    return {
        "command": cmd,
        "rc": rc,
        "timed_out": timed_out,
        "duration_seconds": round((_utc_now() - started).total_seconds(), 3),
        "parsed": parsed,
        "ok": bool(_payload_ok(parsed, rc)) and not timed_out,
        "stdout_tail": "\n".join(stdout.splitlines()[-8:])[-1600:],
        "stderr_tail": "\n".join(stderr.splitlines()[-8:])[-1200:],
    }


def _load_state(project_root: Path, state_path: Path | None = None) -> dict[str, Any]:
    path = state_path or project_root / "governance" / "health" / "soak_self_healing_state.json"
    state = load_json(path if path.is_absolute() else project_root / path)
    return {
        "schema_version": 1,
        "timestamp_utc": str(state.get("timestamp_utc") or ""),
        "steps": _as_dict(state.get("steps")),
    }


def _write_state(project_root: Path, state: dict[str, Any], state_path: Path | None = None) -> None:
    path = state_path or project_root / "governance" / "health" / "soak_self_healing_state.json"
    state["timestamp_utc"] = _iso_now()
    write_payload(path if path.is_absolute() else project_root / path, state)


def _cooldown_active(state: dict[str, Any], step_name: str, now: datetime | None = None) -> dict[str, Any]:
    step_state = _as_dict(_as_dict(state.get("steps")).get(step_name))
    until = parse_iso_utc(step_state.get("cooldown_until_utc"))
    current = now or _utc_now()
    if until is None or until <= current:
        return {"active": False}
    return {
        "active": True,
        "cooldown_until_utc": until.isoformat(),
        "last_rc": step_state.get("last_rc"),
        "last_status": str(step_state.get("last_status") or ""),
        "reason": str(step_state.get("cooldown_reason") or ""),
    }


def _update_step_state(state: dict[str, Any], step_name: str, row: dict[str, Any], *, cooldown_seconds: int = 0) -> None:
    steps = _as_dict(state.setdefault("steps", {}))
    parsed = _as_dict(row.get("parsed"))
    ok = bool(row.get("ok", False))
    until = _utc_now() + timedelta(seconds=max(int(cooldown_seconds), 0)) if (not ok and cooldown_seconds > 0) else None
    steps[step_name] = {
        "last_seen_utc": _iso_now(),
        "last_rc": row.get("rc"),
        "last_ok": ok,
        "last_status": str(parsed.get("overall_status") or parsed.get("status") or ""),
        "failure_count": 0 if ok else _safe_int(_as_dict(steps.get(step_name)).get("failure_count"), 0) + 1,
        "cooldown_until_utc": until.isoformat() if until else "",
        "cooldown_reason": "bounded_self_heal_backoff" if until else "",
    }
    state["steps"] = steps


def _run_step(
    steps: list[dict[str, Any]],
    *,
    name: str,
    cmd: list[str],
    project_root: Path,
    timeout_sec: int,
    env: dict[str, str],
    state: dict[str, Any],
    cooldown_seconds: int = 0,
    respect_cooldowns: bool = True,
) -> dict[str, Any]:
    cooldown = _cooldown_active(state, name)
    if respect_cooldowns and bool(cooldown.get("active")):
        row = {
            "name": name,
            "command": cmd,
            "executed": False,
            "ok": True,
            "skipped_reason": "self_healing_cooldown_active",
            "cooldown": cooldown,
        }
        steps.append(row)
        return row
    result = _run_command(cmd, project_root=project_root, timeout_sec=timeout_sec, env=env)
    row = {"name": name, "executed": True, **result}
    _update_step_state(state, name, row, cooldown_seconds=cooldown_seconds)
    steps.append(row)
    return row


def _daily_failures(payload: dict[str, Any]) -> list[str]:
    return ordered_unique([str(item or "").strip() for item in _as_list(payload.get("failed_checks"))])


def _managed_daily_failures(payload: dict[str, Any]) -> list[str]:
    return [item for item in _daily_failures(payload) if item in MANAGED_DAILY_FAILURES]


def _repairable_daily_failures(payload: dict[str, Any]) -> list[str]:
    return [item for item in _daily_failures(payload) if item not in MANAGED_DAILY_FAILURES]


def _should_run_daily_verify(project_root: Path, daily_payload: dict[str, Any], *, max_age_minutes: float, force: bool) -> bool:
    if force or not daily_payload:
        return True
    if _repairable_daily_failures(daily_payload):
        return True
    age = payload_age_minutes(daily_payload, project_root / "governance" / "health" / "daily_auto_verify_latest.json")
    return bool(age is None or age > float(max_age_minutes))


def _soak_blockers(payload: dict[str, Any]) -> list[str]:
    return ordered_unique([str(item or "").strip() for item in _as_list(payload.get("blockers"))])


def _storage_blockers(payload: dict[str, Any]) -> list[str]:
    return [item for item in _soak_blockers(payload) if item in STORAGE_SOAK_BLOCKERS]


def _ingestion_blockers(payload: dict[str, Any]) -> list[str]:
    return [item for item in _soak_blockers(payload) if item in INGESTION_SOAK_BLOCKERS]


def _memory_efficiency_soft_guard(row: dict[str, Any]) -> bool:
    parsed = _as_dict(row.get("parsed"))
    status = _status(parsed)
    if status not in {"advisory", "needs_work", "degraded", "watch"}:
        return False
    memory = _as_dict(parsed.get("memory_snapshot"))
    cotenant = _as_dict(parsed.get("cotenant_awareness"))
    pressure_state = str(memory.get("memory_pressure_state") or "").strip().lower()
    pressure_kind = str(memory.get("memory_pressure_kind") or "").strip().lower()
    free_pct = _safe_float(memory.get("memory_free_pct"), 0.0)
    swap_gb = _safe_float(memory.get("swap_used_gb"), 0.0)
    memory_clear = bool(
        pressure_state in {"green", "normal", ""}
        and pressure_kind in {"normal", "green", "none", ""}
        and free_pct >= 25.0
        and swap_gb <= 8.0
        and bool(cotenant.get("memory_pressure_clear", True))
    )
    reasons = {str(item or "") for item in _as_list(parsed.get("reasons"))}
    advisory_reasons = {
        "compressed_memory_high",
        "co_running_light_competition",
        "storage_pressure_high",
        "creative_session_music_playback",
        "creative_session_active",
    }
    return bool(memory_clear and (not reasons or reasons.issubset(advisory_reasons)))


def _local_disk_headroom_recovery_contract(memory_payload: dict[str, Any]) -> dict[str, Any]:
    contract = _as_dict(memory_payload.get("local_disk_headroom_contract"))
    memory = _as_dict(memory_payload.get("memory_snapshot"))
    reasons = {str(item or "") for item in _as_list(memory_payload.get("reasons"))}
    free_raw = contract.get("local_disk_free_gb", memory.get("local_disk_free_gb"))
    free_known = free_raw is not None
    free_gb = _safe_float(free_raw, 0.0)
    warning_gb = max(_safe_float(contract.get("warning_free_gb"), 32.0), 1.0)
    critical_gb = max(_safe_float(contract.get("critical_free_gb"), 8.0), 0.5)
    active = bool(
        contract.get("active", False)
        or "local_disk_swap_temp_headroom_low" in reasons
        or (free_known and free_gb < warning_gb)
    )
    critical = bool(
        active
        and (
            str(contract.get("severity") or "").strip().lower() == "critical"
            or (free_known and free_gb < critical_gb)
        )
    )
    return {
        "active": active,
        "critical": critical,
        "severity": "critical" if critical else ("warning" if active else "clear"),
        "local_disk_free_gb": round(free_gb, 3) if free_known else None,
        "warning_free_gb": round(warning_gb, 3),
        "critical_free_gb": round(critical_gb, 3),
        "policy": "recover startup-disk headroom before restoring normal fanout because macOS swap and temp files share that capacity",
    }


def _hard_step_failed(row: dict[str, Any]) -> bool:
    if bool(row.get("ok", False)):
        return False
    if str(row.get("name") or "") == "memory_efficiency" and _memory_efficiency_soft_guard(row):
        return False
    if str(row.get("name") or "") == "runtime_paper_regression_guard":
        parsed = _as_dict(row.get("parsed"))
        status = _status(parsed)
        hard_count = _safe_int(parsed.get("hard_failed_guard_count"), 0)
        return bool(hard_count > 0 or status in {"blocked", "critical"})
    return True


def _latest_hard_failures(steps: list[dict[str, Any]]) -> list[str]:
    latest: dict[str, dict[str, Any]] = {}
    for row in steps:
        name = str(row.get("name") or "")
        if name in HARD_RUNTIME_STEP_NAMES and row.get("executed") is not False:
            latest[name] = row
    return [name for name, row in latest.items() if _hard_step_failed(row)]


def _latest_step_payload(steps: list[dict[str, Any]], step_name: str) -> dict[str, Any]:
    for row in reversed(steps):
        if str(row.get("name") or "") != step_name:
            continue
        parsed = row.get("parsed")
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _stale_profitability_control_from_runtime_guard(payload: dict[str, Any]) -> bool:
    for row in _as_list(payload.get("regression_guards")):
        if not isinstance(row, dict) or str(row.get("name") or "") != "soak_hot_artifact_freshness_contract":
            continue
        actual = _as_dict(row.get("actual"))
        for stale in _as_list(actual.get("stale_artifacts")):
            if isinstance(stale, dict) and str(stale.get("name") or "") == "paper_runtime_profitability_controls":
                return True
    profitability_repair_blockers = {
        "profitability_controls_not_enforced",
        "controlled_profitability_posture_not_ready",
        "raw_profitability_improvement_contract_not_ready",
        "raw_profitability_grade_cosmetic_upgrade",
    }
    for row in _as_list(payload.get("regression_guards")):
        if not isinstance(row, dict):
            continue
        if str(row.get("name") or "") not in {
            "production_grade_paper_live_authority_contract",
            "soak_30_day_continuity_contract",
        }:
            continue
        blockers = {str(item) for item in _as_list(_as_dict(row.get("actual")).get("blockers"))}
        if blockers.intersection(profitability_repair_blockers):
            return True
    return False


def _failed_runtime_guard_names(payload: dict[str, Any]) -> list[str]:
    return ordered_unique([str(item or "").strip() for item in _as_list(payload.get("failed_guards")) if str(item or "").strip()])


def _runtime_continuity_refresh_needed(payload: dict[str, Any]) -> bool:
    return bool(RUNTIME_CONTINUITY_REFRESH_GUARDS.intersection(set(_failed_runtime_guard_names(payload))))


def _live_money_hard_blockers(payload: dict[str, Any]) -> list[str]:
    if not payload:
        return ["live_money_readiness_missing"]
    blocking = ordered_unique(
        [
            str(item or "").strip()
            for item in _as_list(payload.get("blocking_reasons"))
            if str(item or "").strip()
        ]
    )
    hard = [item for item in blocking if item not in MANAGED_LIVE_MONEY_LOCK_REASONS]
    summary = _as_dict(payload.get("grade_summary"))
    for section_id in _as_list(summary.get("below_floor_sections")):
        if str(section_id or "").strip():
            hard.append(f"{section_id}_below_floor")
    for section_id in _as_list(summary.get("not_ready_sections")):
        if str(section_id or "").strip():
            hard.append(f"{section_id}_not_ready")
    return ordered_unique(hard)


def _promotion_packet_idle_seed_ready(payload: dict[str, Any]) -> bool:
    scope = _as_dict(payload.get("promotion_scope"))
    gates = _as_dict(payload.get("gate_results"))
    replayability = _as_dict(payload.get("replayability_contract"))
    return bool(
        not bool(payload.get("ok", False))
        and not bool(scope.get("target_count", 0) or scope.get("trained_bot_ids") or scope.get("failure_count", 0))
        and bool(payload.get("committee_packet_seed_ready", False))
        and bool(replayability.get("hash_bundle_complete", False))
        and bool(replayability.get("exact_replay_ready", False))
        and gates
        and all(bool(value) for value in gates.values())
    )


def _cmd(*parts: str | Path) -> list[str]:
    return [str(part) for part in parts]


def _fast_refresh_steps(project_root: Path, *, py: Path, apply: bool) -> list[tuple[str, list[str], int]]:
    livefeed_cmd = _cmd(py, project_root / "scripts" / "ops" / "livefeed_refresh_guard.py")
    if apply:
        livefeed_cmd.append("--apply")
    livefeed_cmd.append("--json")
    return [
        ("session_ready", _cmd(py, project_root / "scripts" / "session_ready_check.py", "--json"), 60),
        ("process_watchdog", _cmd(py, project_root / "scripts" / "ops" / "process_watchdog.py", "--json"), 90),
        ("livefeed_refresh_guard", livefeed_cmd, 90),
        ("runtime_paper_regression_guard", _cmd(py, project_root / "scripts" / "ops" / "runtime_paper_regression_guard.py", "--json"), 60),
        ("memory_efficiency", _cmd(py, project_root / "scripts" / "ops" / "memory_efficiency_control.py", "status", "--json"), 60),
        ("ingestion_storage", _cmd(py, project_root / "scripts" / "ops" / "ingestion_storage_control.py", "--json"), 120),
        ("storage_resilience", _cmd(py, project_root / "scripts" / "ops" / "storage_resilience_control.py", "--fast", "--json"), 120),
        ("notification_ladder", _cmd(py, project_root / "scripts" / "ops" / "notification_escalation_ladder.py", "--json"), 60),
        ("nightly_resilience", _cmd(py, project_root / "scripts" / "nightly_resilience_check.py", "--json"), 60),
    ]


def build_payload(
    project_root: Path = PROJECT_ROOT,
    *,
    apply: bool = False,
    target_days: float = 30.0,
    daily_max_age_minutes: float = 360.0,
    force_daily_verify: bool = False,
    step_timeout_sec: int = 120,
    storage_cooldown_minutes: float = 60.0,
    storage_cleanup_max_delete_gb: float = 16.0,
    storage_target_free_gb: float = 125.0,
    ingestion_repair_cooldown_minutes: float = 20.0,
    include_adaptive_governor: bool = True,
    max_adaptive_repairs: int = 2,
    respect_cooldowns: bool = True,
) -> dict[str, Any]:
    project_root = Path(project_root).resolve()
    py = resolve_runtime_python(project_root)
    opsctl = project_root / "scripts" / "ops" / "opsctl.sh"
    health_root = project_root / "governance" / "health"
    out_path = health_root / "soak_self_healing_control_latest.json"
    state_path = health_root / "soak_self_healing_state.json"
    env = os.environ.copy()
    env.update(SAFE_ENV)
    env.setdefault("BOT_RUNTIME_PROFILE", "live")
    env.setdefault("PYTHONUNBUFFERED", "1")
    cold_archive_env = _configure_cold_archive_env(env, apply=apply)
    state = _load_state(project_root, state_path)
    steps: list[dict[str, Any]] = []

    for name, cmd, timeout in _fast_refresh_steps(project_root, py=py, apply=apply):
        _run_step(
            steps,
            name=name,
            cmd=cmd,
            project_root=project_root,
            timeout_sec=min(max(int(step_timeout_sec), 30), int(timeout)),
            env=env,
            state=state,
            cooldown_seconds=300 if name in {"livefeed_refresh_guard", "nightly_resilience"} else 0,
            respect_cooldowns=respect_cooldowns,
        )

    memory_payload = _latest_step_payload(steps, "memory_efficiency")
    local_disk_recovery_initial = _local_disk_headroom_recovery_contract(memory_payload)
    local_disk_recovery_payloads: dict[str, dict[str, Any]] = {}
    if apply and bool(local_disk_recovery_initial.get("active", False)):
        route_row = _run_step(
            steps,
            name="local_disk_external_route_reconcile",
            cmd=_cmd(opsctl, "storage-transition-coordinator", "--transition-mode", "external", "--apply", "--json"),
            project_root=project_root,
            timeout_sec=max(int(step_timeout_sec), 180),
            env=env,
            state=state,
            cooldown_seconds=int(max(float(ingestion_repair_cooldown_minutes), 1.0) * 60),
            respect_cooldowns=respect_cooldowns,
        )
        local_disk_recovery_payloads["external_route_reconcile"] = _as_dict(route_row.get("parsed"))
        queue_row = _run_step(
            steps,
            name="local_disk_acknowledged_queue_retention",
            cmd=_cmd(
                py,
                project_root / "scripts" / "sql_queue_retention.py",
                "--acked-hours",
                "1",
                "--batch-size",
                "50000",
                "--max-rows",
                "1000000",
                "--json",
            ),
            project_root=project_root,
            timeout_sec=max(int(step_timeout_sec), 300),
            env=env,
            state=state,
            cooldown_seconds=int(max(float(storage_cooldown_minutes), 1.0) * 60),
            respect_cooldowns=respect_cooldowns,
        )
        local_disk_recovery_payloads["acknowledged_queue_retention"] = _as_dict(queue_row.get("parsed"))
        if bool(local_disk_recovery_initial.get("critical", False)):
            compactor_row = _run_step(
                steps,
                name="local_disk_governance_telemetry_compaction",
                cmd=_cmd(
                    opsctl,
                    "governance-telemetry-compactor",
                    "--apply",
                    "--channels",
                    "all",
                    "--target-free-gb",
                    "64",
                    "--min-file-mb",
                    "256",
                    "--include-current-day",
                    "--json",
                ),
                project_root=project_root,
                timeout_sec=max(int(step_timeout_sec), 300),
                env=env,
                state=state,
                cooldown_seconds=int(max(float(storage_cooldown_minutes), 1.0) * 60),
                respect_cooldowns=respect_cooldowns,
            )
            local_disk_recovery_payloads["governance_telemetry_compaction"] = _as_dict(compactor_row.get("parsed"))
            if str(env.get("BOT_SECOND_COLD_ROOT") or "").strip():
                deep_cold_row = _run_step(
                    steps,
                    name="local_disk_resumable_deep_cold_offload",
                    cmd=_cmd(
                        opsctl,
                        "deep-cold-storage-layer",
                        "--apply",
                        "--adaptive",
                        "--move-to-second-cold",
                        "--planning-horizon-days",
                        str(round(float(target_days), 3)),
                        "--json",
                    ),
                    project_root=project_root,
                    timeout_sec=max(int(step_timeout_sec), 600),
                    env=env,
                    state=state,
                    cooldown_seconds=int(max(float(storage_cooldown_minutes), 1.0) * 60),
                    respect_cooldowns=respect_cooldowns,
                )
                local_disk_recovery_payloads["resumable_deep_cold_offload"] = _as_dict(deep_cold_row.get("parsed"))
        clearance_row = _run_step(
            steps,
            name="local_disk_storage_pressure_clearance",
            cmd=_cmd(
                opsctl,
                "storage-pressure-clearance",
                "--apply",
                "--force-clear-stale-gate",
                "--checkpoint-mode",
                "passive",
                "--json",
            ),
            project_root=project_root,
            timeout_sec=max(int(step_timeout_sec), 300),
            env=env,
            state=state,
            cooldown_seconds=int(max(float(storage_cooldown_minutes), 1.0) * 60),
            respect_cooldowns=respect_cooldowns,
        )
        local_disk_recovery_payloads["storage_pressure_clearance"] = _as_dict(clearance_row.get("parsed"))
        _run_step(
            steps,
            name="local_disk_resource_guard_recheck",
            cmd=_cmd(
                py,
                project_root / "scripts" / "resource_guard.py",
                "--project-root",
                project_root,
                "--profile",
                "collection",
                "--json",
            ),
            project_root=project_root,
            timeout_sec=min(max(int(step_timeout_sec), 30), 90),
            env=env,
            state=state,
            cooldown_seconds=0,
            respect_cooldowns=False,
        )
        memory_recheck_row = _run_step(
            steps,
            name="memory_efficiency",
            cmd=_cmd(py, project_root / "scripts" / "ops" / "memory_efficiency_control.py", "status", "--json"),
            project_root=project_root,
            timeout_sec=min(max(int(step_timeout_sec), 30), 90),
            env=env,
            state=state,
            cooldown_seconds=0,
            respect_cooldowns=False,
        )
        memory_payload = _as_dict(memory_recheck_row.get("parsed"))
    local_disk_recovery_final = _local_disk_headroom_recovery_contract(memory_payload)

    if "process_watchdog" in _latest_hard_failures(steps):
        _run_step(
            steps,
            name="process_watchdog",
            cmd=_cmd(py, project_root / "scripts" / "ops" / "process_watchdog.py", "--json"),
            project_root=project_root,
            timeout_sec=min(max(int(step_timeout_sec), 30), 90),
            env=env,
            state=state,
            cooldown_seconds=0,
            respect_cooldowns=False,
        )
    if "nightly_resilience" in _latest_hard_failures(steps):
        _run_step(
            steps,
            name="nightly_resilience",
            cmd=_cmd(py, project_root / "scripts" / "nightly_resilience_check.py", "--json"),
            project_root=project_root,
            timeout_sec=min(max(int(step_timeout_sec), 30), 90),
            env=env,
            state=state,
            cooldown_seconds=0,
            respect_cooldowns=False,
        )

    runtime_paper_payload = _latest_step_payload(steps, "runtime_paper_regression_guard")
    runtime_continuity_refresh_payloads: dict[str, Any] = {}
    if apply and _runtime_continuity_refresh_needed(runtime_paper_payload):
        auth_supervisor_refresh = _run_step(
            steps,
            name="schwab_auth_supervisor",
            cmd=_cmd(opsctl, "schwab-auth-supervisor", "--apply", "--json"),
            project_root=project_root,
            timeout_sec=max(int(step_timeout_sec), 120),
            env=env,
            state=state,
            cooldown_seconds=300,
            respect_cooldowns=respect_cooldowns,
        )
        runtime_continuity_refresh_payloads["schwab_auth_supervisor"] = _as_dict(auth_supervisor_refresh.get("parsed"))
        halt_refresh = _run_step(
            steps,
            name="global_halt_refresh",
            cmd=_cmd(opsctl, "global-halt-refresh", "--json"),
            project_root=project_root,
            timeout_sec=min(max(int(step_timeout_sec), 30), 90),
            env=env,
            state=state,
            cooldown_seconds=120,
            respect_cooldowns=respect_cooldowns,
        )
        runtime_continuity_refresh_payloads["global_halt_refresh"] = _as_dict(halt_refresh.get("parsed"))
        runtime_throttle_refresh = _run_step(
            steps,
            name="runtime_throttle_continuity_refresh",
            cmd=_cmd(opsctl, "runtime-throttle", "--apply", "--max-renice-processes", "8", "--json"),
            project_root=project_root,
            timeout_sec=max(int(step_timeout_sec), 120),
            env=env,
            state=state,
            cooldown_seconds=300,
            respect_cooldowns=respect_cooldowns,
        )
        runtime_continuity_refresh_payloads["runtime_throttle"] = _as_dict(runtime_throttle_refresh.get("parsed"))
        paper_ramp_refresh = _run_step(
            steps,
            name="paper_ramp_continuity_refresh",
            cmd=_cmd(opsctl, "paper-400-ramp", "--apply", "--json"),
            project_root=project_root,
            timeout_sec=max(int(step_timeout_sec), 120),
            env=env,
            state=state,
            cooldown_seconds=300,
            respect_cooldowns=respect_cooldowns,
        )
        runtime_continuity_refresh_payloads["paper_400_ramp"] = _as_dict(paper_ramp_refresh.get("parsed"))
        runtime_paper_recheck = _run_step(
            steps,
            name="runtime_paper_regression_guard",
            cmd=_cmd(py, project_root / "scripts" / "ops" / "runtime_paper_regression_guard.py", "--json"),
            project_root=project_root,
            timeout_sec=min(max(int(step_timeout_sec), 30), 60),
            env=env,
            state=state,
            cooldown_seconds=0,
            respect_cooldowns=False,
        )
        runtime_paper_payload = _as_dict(runtime_paper_recheck.get("parsed"))

    profitability_refresh_payload: dict[str, Any] = {}
    if apply and _stale_profitability_control_from_runtime_guard(runtime_paper_payload):
        profitability_refresh = _run_step(
            steps,
            name="paper_profitability_control_refresh",
            cmd=_cmd(opsctl, "paper-profitability-control", "--apply", "--json"),
            project_root=project_root,
            timeout_sec=max(int(step_timeout_sec), 120),
            env=env,
            state=state,
            cooldown_seconds=600,
            respect_cooldowns=respect_cooldowns,
        )
        profitability_refresh_payload = _as_dict(profitability_refresh.get("parsed"))
        runtime_paper_recheck = _run_step(
            steps,
            name="runtime_paper_regression_guard",
            cmd=_cmd(py, project_root / "scripts" / "ops" / "runtime_paper_regression_guard.py", "--json"),
            project_root=project_root,
            timeout_sec=min(max(int(step_timeout_sec), 30), 60),
            env=env,
            state=state,
            cooldown_seconds=0,
            respect_cooldowns=False,
        )
        runtime_paper_payload = _as_dict(runtime_paper_recheck.get("parsed"))

    daily_payload = load_json(health_root / "daily_auto_verify_latest.json")
    daily_ran = False
    if _should_run_daily_verify(project_root, daily_payload, max_age_minutes=daily_max_age_minutes, force=force_daily_verify):
        daily_row = _run_step(
            steps,
            name="daily_auto_verify",
            cmd=_cmd(py, project_root / "scripts" / "daily_auto_verify.py", "--json"),
            project_root=project_root,
            timeout_sec=max(int(step_timeout_sec), 240),
            env=env,
            state=state,
            cooldown_seconds=1800,
            respect_cooldowns=respect_cooldowns,
        )
        daily_ran = bool(daily_row.get("executed"))
        daily_payload = _as_dict(daily_row.get("parsed")) or load_json(health_root / "daily_auto_verify_latest.json")

    remediation_payload: dict[str, Any] = {}
    if apply and _repairable_daily_failures(daily_payload):
        remediation_row = _run_step(
            steps,
            name="daily_verify_auto_remediation",
            cmd=_cmd(py, project_root / "scripts" / "ops" / "daily_verify_auto_remediation_bot.py", "--apply", "--json"),
            project_root=project_root,
            timeout_sec=max(int(step_timeout_sec), 120),
            env=env,
            state=state,
            cooldown_seconds=900,
            respect_cooldowns=respect_cooldowns,
        )
        remediation_payload = _as_dict(remediation_row.get("parsed"))
        rerun_row = _run_step(
            steps,
            name="daily_auto_verify_recheck",
            cmd=_cmd(py, project_root / "scripts" / "daily_auto_verify.py", "--json"),
            project_root=project_root,
            timeout_sec=max(int(step_timeout_sec), 240),
            env=env,
            state=state,
            cooldown_seconds=1800,
            respect_cooldowns=False,
        )
        daily_payload = _as_dict(rerun_row.get("parsed")) or load_json(health_root / "daily_auto_verify_latest.json")

    production_refresh_payloads: dict[str, Any] = {}
    source_row = _run_step(
        steps,
        name="source_verification_production_refresh",
        cmd=_cmd(opsctl, "source-verification", "--json"),
        project_root=project_root,
        timeout_sec=min(max(int(step_timeout_sec), 30), 90),
        env=env,
        state=state,
        cooldown_seconds=0,
        respect_cooldowns=False,
    )
    production_refresh_payloads["source_verification"] = _as_dict(source_row.get("parsed"))
    paper_profitability_cmd = _cmd(opsctl, "paper-profitability-control")
    if apply:
        paper_profitability_cmd.append("--apply")
    paper_profitability_cmd.append("--json")
    paper_profitability_row = _run_step(
        steps,
        name="paper_profitability_production_refresh",
        cmd=paper_profitability_cmd,
        project_root=project_root,
        timeout_sec=max(int(step_timeout_sec), 120),
        env=env,
        state=state,
        cooldown_seconds=600 if apply else 0,
        respect_cooldowns=respect_cooldowns if apply else False,
    )
    production_refresh_payloads["paper_profitability"] = _as_dict(paper_profitability_row.get("parsed"))
    if apply:
        stale_drain_row = _run_step(
            steps,
            name="paper_execution_stale_prefix_drain",
            cmd=_cmd(
                py,
                project_root / "scripts" / "run_execution_lane.py",
                "--mode",
                "paper",
                "--drain-stale-only",
                "--limit",
                "100000",
                "--stale-drain-passes",
                "25",
            ),
            project_root=project_root,
            timeout_sec=min(max(int(step_timeout_sec), 60), 180),
            env=env,
            state=state,
            cooldown_seconds=0,
            respect_cooldowns=False,
        )
        production_refresh_payloads["paper_execution_stale_prefix_drain"] = _as_dict(stale_drain_row.get("parsed")) or load_json(health_root / "execution_lane_paper_latest.json")
    paper_replay_row = _run_step(
        steps,
        name="paper_replay_drill_production_refresh",
        cmd=_cmd(py, project_root / "scripts" / "paper_replay_drill.py", "--hours", "24", "--json"),
        project_root=project_root,
        timeout_sec=max(int(step_timeout_sec), 120),
        env=env,
        state=state,
        cooldown_seconds=0,
        respect_cooldowns=False,
    )
    production_refresh_payloads["paper_replay_drill"] = _as_dict(paper_replay_row.get("parsed")) or load_json(health_root / "paper_replay_drill_latest.json")
    paper_truth_row = _run_step(
        steps,
        name="paper_execution_truth_production_refresh",
        cmd=_cmd(opsctl, "paper-execution-truth", "--json"),
        project_root=project_root,
        timeout_sec=max(int(step_timeout_sec), 120),
        env=env,
        state=state,
        cooldown_seconds=0,
        respect_cooldowns=False,
    )
    production_refresh_payloads["paper_execution_truth"] = _as_dict(paper_truth_row.get("parsed"))
    schema_row = _run_step(
        steps,
        name="retrain_schema_compatibility_production_refresh",
        cmd=_cmd(py, project_root / "scripts" / "retrain_schema_compatibility_guard.py", "--json"),
        project_root=project_root,
        timeout_sec=min(max(int(step_timeout_sec), 30), 90),
        env=env,
        state=state,
        cooldown_seconds=0,
        respect_cooldowns=False,
    )
    production_refresh_payloads["retrain_schema_compatibility"] = _as_dict(schema_row.get("parsed"))
    promotion_packet_row = _run_step(
        steps,
        name="promotion_packet_production_refresh",
        cmd=_cmd(py, project_root / "scripts" / "promotion_packet_builder.py", "--json"),
        project_root=project_root,
        timeout_sec=max(int(step_timeout_sec), 120),
        env=env,
        state=state,
        cooldown_seconds=0,
        respect_cooldowns=False,
    )
    production_refresh_payloads["promotion_packet"] = _as_dict(promotion_packet_row.get("parsed"))

    promotion_row = _run_step(
        steps,
        name="promotion_quality_gate",
        cmd=_cmd(py, project_root / "scripts" / "promotion_quality_gate.py", "--json"),
        project_root=project_root,
        timeout_sec=min(max(int(step_timeout_sec), 30), 90),
        env=env,
        state=state,
        cooldown_seconds=0,
        respect_cooldowns=False,
    )
    promotion_payload = _as_dict(promotion_row.get("parsed"))

    soak_row = _run_step(
        steps,
        name="unattended_soak_readiness",
        cmd=_cmd(py, project_root / "scripts" / "ops" / "unattended_soak_readiness.py", "--target-days", str(round(float(target_days), 3)), "--json"),
        project_root=project_root,
        timeout_sec=min(max(int(step_timeout_sec), 30), 90),
        env=env,
        state=state,
        cooldown_seconds=0,
        respect_cooldowns=False,
    )
    soak_payload = _as_dict(soak_row.get("parsed")) or load_json(health_root / "unattended_soak_readiness_latest.json")

    storage_retention_payload: dict[str, Any] = {}
    storage_recovery_payloads: dict[str, Any] = {}
    if apply and _storage_blockers(soak_payload):
        raw_compaction_row = _run_step(
            steps,
            name="storage_raw_training_compaction",
            cmd=_cmd(
                opsctl,
                "raw-training-compaction",
                "--apply",
                "--max-files",
                "24",
                "--max-gb",
                str(round(max(min(float(storage_cleanup_max_delete_gb), 12.0), 4.0), 3)),
                "--jumbo-gb",
                "12.0",
                "--min-age-hours",
                "12",
                "--write-history",
                "--json",
            ),
            project_root=project_root,
            timeout_sec=max(int(step_timeout_sec), 240),
            env=env,
            state=state,
            cooldown_seconds=int(max(float(storage_cooldown_minutes), 1.0) * 60),
            respect_cooldowns=respect_cooldowns,
        )
        storage_recovery_payloads["raw_training_compaction"] = _as_dict(raw_compaction_row.get("parsed"))
        retention_row = _run_step(
            steps,
            name="storage_retention_unison",
            cmd=_cmd(
                opsctl,
                "storage-retention-unison",
                "--apply",
                "--soak-days",
                str(round(float(target_days), 3)),
                "--cleanup-max-delete-gb",
                str(round(float(storage_cleanup_max_delete_gb), 3)),
                "--telemetry-max-gb",
                "16",
                "--decision-max-gb",
                "8",
                "--target-free-gb",
                str(round(float(storage_target_free_gb), 3)),
                "--json",
            ),
            project_root=project_root,
            timeout_sec=max(int(step_timeout_sec), 180),
            env=env,
            state=state,
            cooldown_seconds=int(max(float(storage_cooldown_minutes), 1.0) * 60),
            respect_cooldowns=respect_cooldowns,
        )
        storage_retention_payload = _as_dict(retention_row.get("parsed"))
        cold_archive_root = str(env.get("BOT_SECOND_COLD_ROOT") or "").strip()
        if cold_archive_root:
            offload_row = _run_step(
                steps,
                name="storage_manifest_backed_cold_offload",
                cmd=_cmd(
                    opsctl,
                    "manifest-backed-offload",
                    "--apply",
                    "--target-root",
                    cold_archive_root,
                    "--max-files",
                    "16",
                    "--max-gb",
                    str(round(max(min(float(storage_cleanup_max_delete_gb), 16.0), 4.0), 3)),
                    "--release-source-after-verify",
                    "--json",
                ),
                project_root=project_root,
                timeout_sec=max(int(step_timeout_sec), 300),
                env=env,
                state=state,
                cooldown_seconds=int(max(float(storage_cooldown_minutes), 1.0) * 60),
                respect_cooldowns=respect_cooldowns,
            )
            storage_recovery_payloads["manifest_backed_cold_offload"] = _as_dict(offload_row.get("parsed"))
            retention_recheck_row = _run_step(
                steps,
                name="storage_retention_unison_after_cold_offload",
                cmd=_cmd(
                    opsctl,
                    "storage-retention-unison",
                    "--apply",
                    "--soak-days",
                    str(round(float(target_days), 3)),
                    "--cleanup-max-delete-gb",
                    str(round(float(storage_cleanup_max_delete_gb), 3)),
                    "--telemetry-max-gb",
                    "16",
                    "--decision-max-gb",
                    "8",
                    "--target-free-gb",
                    str(round(float(storage_target_free_gb), 3)),
                    "--json",
                ),
                project_root=project_root,
                timeout_sec=max(int(step_timeout_sec), 240),
                env=env,
                state=state,
                cooldown_seconds=0,
                respect_cooldowns=False,
            )
            storage_recovery_payloads["retention_after_cold_offload"] = _as_dict(retention_recheck_row.get("parsed"))
        soak_recheck = _run_step(
            steps,
            name="unattended_soak_recheck",
            cmd=_cmd(py, project_root / "scripts" / "ops" / "unattended_soak_readiness.py", "--target-days", str(round(float(target_days), 3)), "--json"),
            project_root=project_root,
            timeout_sec=min(max(int(step_timeout_sec), 30), 90),
            env=env,
            state=state,
            cooldown_seconds=0,
            respect_cooldowns=False,
        )
        soak_payload = _as_dict(soak_recheck.get("parsed")) or load_json(health_root / "unattended_soak_readiness_latest.json")

    ingestion_repair_payloads: dict[str, dict[str, Any]] = {}
    if apply and _ingestion_blockers(soak_payload):
        route_row = _run_step(
            steps,
            name="storage_route_reconcile",
            cmd=_cmd(opsctl, "storage-transition-coordinator", "--transition-mode", "external", "--apply", "--json"),
            project_root=project_root,
            timeout_sec=max(int(step_timeout_sec), 180),
            env=env,
            state=state,
            cooldown_seconds=int(max(float(ingestion_repair_cooldown_minutes), 1.0) * 60),
            respect_cooldowns=respect_cooldowns,
        )
        ingestion_repair_payloads["storage_route_reconcile"] = _as_dict(route_row.get("parsed"))
        backpressure_row = _run_step(
            steps,
            name="storage_backpressure_autopilot",
            cmd=_cmd(
                opsctl,
                "storage-backpressure-autopilot",
                "--apply",
                "--quick-bounded",
                "--poll-seconds",
                "0",
                "--wait-timeout-seconds",
                "45",
                "--command-timeout-seconds",
                str(max(int(step_timeout_sec), 120)),
                "--json",
            ),
            project_root=project_root,
            timeout_sec=max(int(step_timeout_sec), 180),
            env=env,
            state=state,
            cooldown_seconds=int(max(float(ingestion_repair_cooldown_minutes), 1.0) * 60),
            respect_cooldowns=respect_cooldowns,
        )
        ingestion_repair_payloads["storage_backpressure_autopilot"] = _as_dict(backpressure_row.get("parsed"))
        _run_step(
            steps,
            name="ingestion_storage_recheck",
            cmd=_cmd(py, project_root / "scripts" / "ops" / "ingestion_storage_control.py", "--json"),
            project_root=project_root,
            timeout_sec=max(int(step_timeout_sec), 120),
            env=env,
            state=state,
            cooldown_seconds=0,
            respect_cooldowns=False,
        )
        soak_recheck = _run_step(
            steps,
            name="unattended_soak_recheck",
            cmd=_cmd(py, project_root / "scripts" / "ops" / "unattended_soak_readiness.py", "--target-days", str(round(float(target_days), 3)), "--json"),
            project_root=project_root,
            timeout_sec=min(max(int(step_timeout_sec), 30), 90),
            env=env,
            state=state,
            cooldown_seconds=0,
            respect_cooldowns=False,
        )
        soak_payload = _as_dict(soak_recheck.get("parsed")) or load_json(health_root / "unattended_soak_readiness_latest.json")

    adaptive_payload: dict[str, Any] = {}
    if apply and include_adaptive_governor:
        adaptive_row = _run_step(
            steps,
            name="adaptive_infrabot_governor",
            cmd=_cmd(
                opsctl,
                "infrabot-adaptive-governor",
                "--apply",
                "--execute-safe-repairs",
                "--max-actions",
                str(max(int(max_adaptive_repairs), 1)),
                "--max-execute-actions",
                str(max(int(max_adaptive_repairs), 1)),
                "--command-timeout-seconds",
                str(max(int(step_timeout_sec), 120)),
                "--json",
            ),
            project_root=project_root,
            timeout_sec=max(int(step_timeout_sec) * max(int(max_adaptive_repairs), 1), 180),
            env=env,
            state=state,
            cooldown_seconds=900,
        respect_cooldowns=respect_cooldowns,
        )
        adaptive_payload = _as_dict(adaptive_row.get("parsed"))

    live_money_row = _run_step(
        steps,
        name="live_money_readiness_production_recheck",
        cmd=_cmd(opsctl, "live-money-readiness", "--json"),
        project_root=project_root,
        timeout_sec=min(max(int(step_timeout_sec), 30), 90),
        env=env,
        state=state,
        cooldown_seconds=0,
        respect_cooldowns=False,
    )
    live_money_payload = _as_dict(live_money_row.get("parsed"))
    production_hard_blockers = _live_money_hard_blockers(live_money_payload)
    managed_live_money_locks = [
        item
        for item in _as_list(live_money_payload.get("blocking_reasons"))
        if str(item or "").strip() in MANAGED_LIVE_MONEY_LOCK_REASONS
    ]

    _write_state(project_root, state, state_path)

    managed_daily = _managed_daily_failures(daily_payload)
    repairable_daily = _repairable_daily_failures(daily_payload)
    soak_blockers = _soak_blockers(soak_payload)
    managed_controls = ordered_unique([str(item or "") for item in _as_list(soak_payload.get("managed_controls"))])
    scored_soak_blockers = [item for item in soak_blockers if item not in STORAGE_SOAK_BLOCKERS]
    core_failures = _latest_hard_failures(steps)
    storage_gap = _safe_float(_as_dict(_as_dict(soak_payload.get("sections")).get("storage")).get("available_margin_gb"), 0.0)
    storage_free = _safe_float(_as_dict(_as_dict(soak_payload.get("sections")).get("storage")).get("current_external_free_gb"), 0.0)
    storage_required = _safe_float(_as_dict(_as_dict(soak_payload.get("sections")).get("storage")).get("required_external_free_gb"), 0.0)

    operator_followups: list[str] = []
    if _storage_blockers(soak_payload):
        operator_followups.append("add_or_free_external_storage_capacity_for_30_day_soak")
    if managed_daily:
        operator_followups.append("keep_live_money_and_promotion_locked_until_promotion_quality_gate_clears")
    if repairable_daily:
        operator_followups.append("inspect_unresolved_daily_verify_repairs")
    if _ingestion_blockers(soak_payload):
        operator_followups.append("let_bounded_ingestion_storage_repairs_continue_until_soak_contract_clears")
    if scored_soak_blockers:
        operator_followups.append("inspect_unattended_soak_non_storage_blockers")
    if production_hard_blockers:
        operator_followups.append("inspect_production_hard_blocker_cascade")
    if bool(local_disk_recovery_final.get("active", False)):
        operator_followups.append("local_disk_swap_temp_headroom_recovery_still_required")

    if core_failures or production_hard_blockers:
        overall_status = "blocked"
    elif repairable_daily:
        overall_status = "needs_attention"
    elif bool(soak_payload.get("safe_to_leave_unattended", False)):
        overall_status = "ready"
    elif _storage_blockers(soak_payload):
        overall_status = "guarded_storage_capacity"
    elif _ingestion_blockers(soak_payload):
        overall_status = "guarded_ingestion_repair"
    elif managed_daily or MANAGED_SOAK_CONTROLS.intersection(set(managed_controls)):
        overall_status = "managed_evidence_lock"
    elif soak_blockers:
        overall_status = "guarded"
    else:
        overall_status = "ready"

    payload = {
        "timestamp_utc": _iso_now(),
        "schema_version": 1,
        "ok": not core_failures and not repairable_daily and not production_hard_blockers,
        "overall_status": overall_status,
        "apply": bool(apply),
        "target_days": float(target_days),
        "live_execution_authority": False,
        "safety_contract": {
            "market_data_only": True,
            "allow_order_execution": False,
            "live_money_locked_during_soak": True,
            "bounded_storage_retention_only": True,
            "approved_video_cold_archive_subtree_only": True,
            "destructive_manual_delete_allowed": False,
            "promotion_gate_autounlock_allowed": False,
            "startup_disk_swap_temp_reserve_required": True,
        },
        "runtime_ok": not core_failures,
        "production_hard_blockers_clear": not production_hard_blockers,
        "safe_to_leave_unattended": bool(soak_payload.get("safe_to_leave_unattended", False)),
        "unattended_soak_status": str(soak_payload.get("overall_status") or ""),
        "unattended_soak_grade": str(soak_payload.get("overall_grade") or ""),
        "soak_blockers": soak_blockers,
        "storage": {
            "free_gb": round(storage_free, 3),
            "required_free_gb": round(storage_required, 3),
            "available_margin_gb": round(storage_gap, 3),
            "blockers": _storage_blockers(soak_payload),
            "retention_attempted": bool(storage_retention_payload),
            "cold_archive": cold_archive_env,
            "recovery": {
                "raw_compaction_attempted": bool(storage_recovery_payloads.get("raw_training_compaction")),
                "raw_gb_cleared": _safe_float(
                    _as_dict(storage_recovery_payloads.get("raw_training_compaction")).get("raw_gb_cleared")
                    or _as_dict(_as_dict(storage_recovery_payloads.get("raw_training_compaction")).get("raw_summary")).get(
                        "raw_gb_cleared"
                    ),
                    0.0,
                ),
                "manifest_cold_offload_attempted": bool(storage_recovery_payloads.get("manifest_backed_cold_offload")),
                "manifest_released_gb": _safe_float(
                    _as_dict(
                        _as_dict(storage_recovery_payloads.get("manifest_backed_cold_offload")).get("apply_result")
                    ).get("released_gb"),
                    0.0,
                ),
                "manifest_offload_status": str(
                    _as_dict(storage_recovery_payloads.get("manifest_backed_cold_offload")).get("overall_status") or ""
                ),
                "retention_recheck_status": str(
                    _as_dict(storage_recovery_payloads.get("retention_after_cold_offload")).get("overall_status") or ""
                ),
            },
        },
        "application_memory_protection": {
            "incident_class": "startup_disk_exhaustion_can_starve_swap_and_temp_files",
            "recovery_attempted": bool(local_disk_recovery_payloads),
            "initial": local_disk_recovery_initial,
            "final": local_disk_recovery_final,
            "external_route_reconcile_status": str(
                _as_dict(local_disk_recovery_payloads.get("external_route_reconcile")).get("overall_status") or ""
            ),
            "acknowledged_queue_rows_deleted": _safe_int(
                _as_dict(local_disk_recovery_payloads.get("acknowledged_queue_retention")).get("deleted_acked_rows"),
                0,
            ),
            "telemetry_compaction_status": str(
                _as_dict(local_disk_recovery_payloads.get("governance_telemetry_compaction")).get("overall_status") or ""
            ),
            "deep_cold_offload_status": str(
                _as_dict(local_disk_recovery_payloads.get("resumable_deep_cold_offload")).get("overall_status") or ""
            ),
            "storage_pressure_clearance_status": str(
                _as_dict(local_disk_recovery_payloads.get("storage_pressure_clearance")).get("overall_status") or ""
            ),
            "automatic_recovery_order": [
                "external_storage_route_reconcile",
                "acknowledged_queue_retention",
                "critical_only_governance_telemetry_compaction",
                "critical_only_resumable_verified_deep_cold_offload",
                "bounded_storage_pressure_clearance",
                "resource_and_memory_guard_recheck",
            ],
        },
        "ingestion_soak_repair": {
            "attempted": bool(ingestion_repair_payloads),
            "blockers": _ingestion_blockers(soak_payload),
            "route_reconcile_status": str(
                _as_dict(ingestion_repair_payloads.get("storage_route_reconcile")).get("overall_status") or ""
            ),
            "backpressure_autopilot_status": str(
                _as_dict(ingestion_repair_payloads.get("storage_backpressure_autopilot")).get("overall_status") or ""
            ),
        },
        "daily_verify": {
            "ran": bool(daily_ran),
            "ok": bool(daily_payload.get("ok", False)),
            "failed_checks": _daily_failures(daily_payload),
            "managed_failed_checks": managed_daily,
            "repairable_failed_checks": repairable_daily,
            "remediation": {
                "attempted": bool(remediation_payload),
                "overall_status": str(remediation_payload.get("overall_status") or ""),
                "resolved_checks": _as_list(remediation_payload.get("resolved_checks")),
                "unresolved_checks": _as_list(remediation_payload.get("unresolved_checks")),
            },
        },
        "promotion_quality": {
            "ok": bool(promotion_payload.get("ok", False)),
            "failed_checks": _as_list(promotion_payload.get("failed_checks")),
            "managed_as_evidence_lock": bool(managed_daily),
        },
        "production_hardening": {
            "active": True,
            "mode": "dependency_ordered_hard_blocker_cascade_v1",
            "paper_only": True,
            "live_execution_allowed": False,
            "ready": not production_hard_blockers,
            "hard_blockers": production_hard_blockers,
            "managed_live_money_locks": managed_live_money_locks,
            "source_verification_ready": bool(production_refresh_payloads.get("source_verification", {}).get("ok", False)),
            "paper_replay_ok": bool(production_refresh_payloads.get("paper_replay_drill", {}).get("ok", False)),
            "paper_replay_failed_checks": _as_list(
                production_refresh_payloads.get("paper_replay_drill", {}).get("failed_checks")
            ),
            "paper_replay_rows": _safe_int(production_refresh_payloads.get("paper_replay_drill", {}).get("rows"), 0),
            "paper_execution_result_activity_status": str(
                _as_dict(production_refresh_payloads.get("paper_execution_stale_prefix_drain")).get("result_activity_status")
                or _as_dict(
                    _as_dict(production_refresh_payloads.get("paper_execution_stale_prefix_drain")).get("execution_result_evidence")
                ).get("activity_status")
                or ""
            ),
            "paper_execution_truth_grade": str(production_refresh_payloads.get("paper_execution_truth", {}).get("grade") or ""),
            "paper_profitability_display_grade": str(
                production_refresh_payloads.get("paper_profitability", {}).get("profitability_display_grade") or ""
            ),
            "raw_profitability_grade": str(
                production_refresh_payloads.get("paper_profitability", {}).get("raw_profitability_grade") or ""
            ),
            "schema_compatibility_ok": bool(
                production_refresh_payloads.get("retrain_schema_compatibility", {}).get("ok", False)
            ),
            "promotion_packet_idle_seed_ready": _promotion_packet_idle_seed_ready(
                _as_dict(production_refresh_payloads.get("promotion_packet"))
            ),
            "promotion_quality_ok": bool(promotion_payload.get("ok", False)),
            "live_money_ready_required_section_count": _safe_int(
                _as_dict(live_money_payload.get("grade_summary")).get("ready_required_section_count"),
                0,
            ),
            "live_money_required_section_count": _safe_int(
                _as_dict(live_money_payload.get("grade_summary")).get("required_section_count"),
                0,
            ),
            "live_money_blocking_reasons": _as_list(live_money_payload.get("blocking_reasons")),
            "live_money_grade_summary": _as_dict(live_money_payload.get("grade_summary")),
            "refresh_order": [
                "source_verification",
                "paper_profitability",
                "paper_execution_stale_prefix_drain",
                "paper_replay_drill",
                "paper_execution_truth",
                "retrain_schema_compatibility",
                "promotion_packet",
                "promotion_quality_gate",
                "unattended_soak_readiness",
                "bounded_repairs_if_needed",
                "live_money_readiness",
            ],
        },
        "self_healing": {
            "state_path": str(state_path),
            "respect_cooldowns": bool(respect_cooldowns),
            "steps_executed": len([row for row in steps if row.get("executed") is not False]),
            "steps_skipped": len([row for row in steps if row.get("executed") is False]),
            "core_failures": core_failures,
            "operator_followups": ordered_unique(operator_followups),
        },
        "profitability_control_refresh": {
            "attempted": bool(profitability_refresh_payload),
            "overall_status": str(profitability_refresh_payload.get("overall_status") or ""),
            "raw_profitability_grade": str(profitability_refresh_payload.get("raw_profitability_grade") or ""),
            "controlled_profitability_grade": str(profitability_refresh_payload.get("controlled_profitability_grade") or ""),
        },
        "runtime_continuity_refresh": {
            "attempted": bool(runtime_continuity_refresh_payloads),
            "schwab_auth_status": str(
                _as_dict(runtime_continuity_refresh_payloads.get("schwab_auth_supervisor")).get("overall_status") or ""
            ),
            "global_halt_status": str(
                _as_dict(runtime_continuity_refresh_payloads.get("global_halt_refresh")).get("overall_status")
                or _as_dict(runtime_continuity_refresh_payloads.get("global_halt_refresh")).get("status")
                or ""
            ),
            "runtime_throttle_status": str(
                _as_dict(runtime_continuity_refresh_payloads.get("runtime_throttle")).get("overall_status") or ""
            ),
            "paper_ramp_stage": str(_as_dict(runtime_continuity_refresh_payloads.get("paper_400_ramp")).get("stage") or ""),
            "runtime_guard_after_refresh": str(runtime_paper_payload.get("overall_status") or ""),
            "failed_guards_after_refresh": _failed_runtime_guard_names(runtime_paper_payload),
        },
        "adaptive_governor": {
            "attempted": bool(adaptive_payload),
            "overall_status": str(adaptive_payload.get("overall_status") or ""),
        },
        "steps": steps,
        "recommended_actions": ordered_unique(
            operator_followups
            + [
                "leave live execution disabled during the soak",
                "let soak-self-heal run on launchd; use the artifact for daily mobile review",
            ]
        ),
    }
    write_payload(out_path, payload)
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Bounded self-healing control loop for the unattended soak.")
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--apply", action="store_true", help="Apply safe mapped repairs and bounded retention relief.")
    parser.add_argument("--target-days", type=float, default=30.0)
    parser.add_argument("--daily-max-age-minutes", type=float, default=360.0)
    parser.add_argument("--force-daily-verify", action="store_true")
    parser.add_argument("--step-timeout-sec", type=int, default=120)
    parser.add_argument("--storage-cooldown-minutes", type=float, default=60.0)
    parser.add_argument("--storage-cleanup-max-delete-gb", type=float, default=16.0)
    parser.add_argument("--storage-target-free-gb", type=float, default=125.0)
    parser.add_argument("--ingestion-repair-cooldown-minutes", type=float, default=20.0)
    parser.add_argument("--include-adaptive-governor", dest="include_adaptive_governor", action="store_true", default=True)
    parser.add_argument("--skip-adaptive-governor", dest="include_adaptive_governor", action="store_false")
    parser.add_argument("--max-adaptive-repairs", type=int, default=2)
    parser.add_argument("--no-cooldowns", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    payload = build_payload(
        Path(args.project_root),
        apply=bool(args.apply),
        target_days=float(args.target_days),
        daily_max_age_minutes=float(args.daily_max_age_minutes),
        force_daily_verify=bool(args.force_daily_verify),
        step_timeout_sec=int(args.step_timeout_sec),
        storage_cooldown_minutes=float(args.storage_cooldown_minutes),
        storage_cleanup_max_delete_gb=float(args.storage_cleanup_max_delete_gb),
        storage_target_free_gb=float(args.storage_target_free_gb),
        ingestion_repair_cooldown_minutes=float(args.ingestion_repair_cooldown_minutes),
        include_adaptive_governor=bool(args.include_adaptive_governor),
        max_adaptive_repairs=int(args.max_adaptive_repairs),
        respect_cooldowns=not bool(args.no_cooldowns),
    )
    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        print(
            "soak_self_healing_control "
            f"status={payload.get('overall_status')} "
            f"runtime_ok={int(bool(payload.get('runtime_ok')))} "
            f"unattended_ready={int(bool(payload.get('safe_to_leave_unattended')))} "
            f"followups={len(_as_list(_as_dict(payload.get('self_healing')).get('operator_followups')))}"
        )
    return 0 if bool(payload.get("ok", False)) else 2


if __name__ == "__main__":
    raise SystemExit(main())
