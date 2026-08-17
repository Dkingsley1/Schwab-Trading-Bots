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
    from scripts.ops import one_numbers_regression_guard
    from scripts.ops.long_runtime_common import iso_now, load_json, ordered_unique, payload_age_minutes, write_payload
else:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    from . import one_numbers_regression_guard
    from .long_runtime_common import iso_now, load_json, ordered_unique, payload_age_minutes, write_payload


DEFAULT_OUT_PATH = PROJECT_ROOT / "governance" / "health" / "master_infrastructure_supervisor_latest.json"
REPAIR_CALL_STACK_ENV = "INFRA_REPAIR_CALL_STACK"
READY_STATUSES = {"ready", "ok", "stable", "applied", "applied_with_followups", "cleared"}
DEGRADED_STATUSES = {
    "active",
    "already_running",
    "busy",
    "degraded",
    "drain_active",
    "needs_work",
    "pending",
    "recovering",
    "recovering_under_guard",
    "running",
    "thin",
    "warn",
    "warning",
}


def _repair_call_stack() -> list[str]:
    return [
        item.strip()
        for item in str(os.getenv(REPAIR_CALL_STACK_ENV, "") or "").split(",")
        if item.strip()
    ]


def _child_env(component: str) -> dict[str, str]:
    env = os.environ.copy()
    stack = _repair_call_stack()
    name = str(component or "").strip()
    if name and name not in stack:
        stack.append(name)
    env[REPAIR_CALL_STACK_ENV] = ",".join(stack)
    return env
BLOCKED_STATUSES = {"blocked", "critical", "failed", "apply_failed", "missing", "unknown"}
LAUNCHD_JOB_SPECS = (
    ("com.dankingsley.ops.command_validity", "scripts/ops/run_command_validity_launchd.sh"),
    ("com.dankingsley.ops.process_fanout_guard", "scripts/install_process_fanout_guard_launchd.sh"),
    ("com.dankingsley.ops.system_drift_guard", "scripts/ops/run_system_drift_guard_launchd.sh"),
    ("com.dankingsley.ops.system_drift_autopilot", "scripts/ops/run_system_drift_autopilot_launchd.sh"),
    ("com.dankingsley.ops.infrastructure_autofix", "scripts/ops/run_infrastructure_autofix_launchd.sh"),
    ("com.dankingsley.ops.master_infrastructure_supervisor", "scripts/ops/run_master_infrastructure_supervisor_launchd.sh"),
    ("com.dankingsley.ops.one_numbers_regression_guard", "scripts/ops/run_one_numbers_regression_guard_launchd.sh"),
    ("com.dankingsley.ops.storage_backpressure_autopilot", "scripts/ops/run_storage_backpressure_autopilot_launchd.sh"),
    ("com.dankingsley.ops.storage_pressure_clearance", "scripts/ops/run_storage_pressure_clearance_launchd.sh"),
    ("com.dankingsley.ops.chrome_headless_guard", "scripts/ops/run_chrome_headless_guard_launchd.sh"),
)
ENVELOPE_LANES = (
    ("1", "historical_truth_layer", "one_numbers_original_coverage"),
    ("2", "master_infrastructure_supervisor_v2", "launchd_job_health"),
    ("3", "autonomous_recovery_drills", "autonomous_recovery_drills"),
    ("4", "command_surface_as_tests", "command_docs_vs_opsctl_routes"),
    ("5", "operator_cockpit", "operator_cockpit_readiness"),
    ("6", "cold_lane_research_factory", "cold_lane_research_factory"),
    ("7", "point_in_time_replay", "point_in_time_replay"),
    ("8", "self_auditing_infra_bots", "self_auditing_infra_bots"),
)
LANE_OWNER_LIMITS = {
    "schwab_all_sleeves": 1,
    "coinbase_shadow": 1,
    "coinbase_futures_shadow": 1,
    "fx_shadow": 1,
    "schwab_futures_shadow": 1,
    "dividend_shadow": 1,
    "dividend_capture_shadow": 1,
    "bond_shadow": 1,
    "parallel_shadows_simulate": 1,
}


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


def _status(raw: Any, *, missing: str = "blocked") -> str:
    text = str(raw or "").strip().lower()
    if not text:
        return missing
    if text in READY_STATUSES:
        return "ready"
    if text in DEGRADED_STATUSES:
        return "degraded"
    if text in BLOCKED_STATUSES:
        return "blocked"
    return "degraded"


def _artifact_status(payload: dict[str, Any], *, missing: str = "blocked") -> str:
    if not payload:
        return missing
    nested_overall = payload.get("overall") if isinstance(payload.get("overall"), dict) else {}
    if nested_overall and not (payload.get("overall_status") or payload.get("status")):
        nested_status = nested_overall.get("overall_status") or nested_overall.get("status")
        if nested_overall.get("ok") is True:
            normalized = _status(nested_status or "ready", missing=missing)
            return "blocked" if normalized == "blocked" else "ready"
        if nested_overall.get("ok") is False:
            return _status(nested_status or "degraded", missing=missing)
        if nested_status:
            return _status(nested_status, missing=missing)
    if payload.get("ok") is True:
        normalized = _status(payload.get("overall_status") or payload.get("status") or "ready", missing=missing)
        return "blocked" if normalized == "blocked" else "ready"
    if payload.get("ok") is False and not payload.get("overall_status"):
        return "blocked"
    return _status(payload.get("overall_status") or payload.get("status"), missing=missing)


def _as_dict(raw: Any) -> dict[str, Any]:
    return raw if isinstance(raw, dict) else {}


def _guarded_paper_strict_clear(project_root: Path) -> bool:
    health_fast = load_json(_health_path(project_root, "health_fast_latest.json"))
    operational = _as_dict(health_fast.get("operational_readiness"))
    guarded_paper = _as_dict(operational.get("guarded_paper"))
    live_execution = _as_dict(operational.get("live_execution"))
    guarded_ready = bool(guarded_paper.get("ok", False)) and str(guarded_paper.get("status") or "").strip().lower() in {
        "ready",
        "armed",
        "guarded_ready",
    }
    live_locked = str(live_execution.get("status") or "").strip().lower() in {
        "blocked_read_only",
        "locked",
        "read_only",
        "disabled",
    }
    operational_health_ready = bool(
        health_fast.get("strict_all_clear", False)
        or (
            bool(health_fast.get("ok", False))
            and str(health_fast.get("overall_status") or "").strip().lower() in {"ready", "guarded_ready"}
        )
    )
    return bool(operational_health_ready and guarded_ready and live_locked)


def _command_key(cmd: list[str]) -> str:
    return "\0".join(str(part) for part in cmd)


def _unique_commands(commands: list[list[str]]) -> list[list[str]]:
    out: list[list[str]] = []
    seen: set[str] = set()
    for raw in commands:
        cmd = [str(part) for part in list(raw or []) if str(part).strip()]
        if not cmd:
            continue
        key = _command_key(cmd)
        if key in seen:
            continue
        seen.add(key)
        out.append(cmd)
    return out


def _clamp(raw: float, low: float, high: float) -> float:
    return max(low, min(high, raw))


def _check(
    name: str,
    *,
    status: str,
    summary: str,
    family: str,
    evidence: dict[str, Any] | None = None,
    repair_commands: list[list[str]] | None = None,
) -> dict[str, Any]:
    normalized = _status(status)
    return {
        "name": name,
        "family": family,
        "status": normalized,
        "ok": normalized == "ready",
        "summary": summary,
        "evidence": evidence or {},
        "repair_commands": _unique_commands(repair_commands or []),
    }


def _has_timed_out(payload: Any) -> bool:
    if isinstance(payload, dict):
        for key, value in payload.items():
            if key in {"timed_out", "timeout", "timed_out_before_finish"} and value is True:
                return True
            if _has_timed_out(value):
                return True
    if isinstance(payload, list):
        return any(_has_timed_out(item) for item in payload)
    return False


def _blocked_surface_names(payload: dict[str, Any]) -> set[str]:
    rows = payload.get("surfaces") if isinstance(payload.get("surfaces"), list) else []
    return {
        str(row.get("name") or "")
        for row in rows
        if isinstance(row, dict) and str(row.get("status") or "").strip().lower() == "blocked"
    }


def _attempt_has_active_recovery(project_root: Path, attempt: dict[str, Any]) -> bool:
    cmd_text = " ".join(str(part) for part in list(attempt.get("cmd") or []))
    recovery_artifacts = [
        (
            "storage_backpressure_autopilot",
            "governance/health/storage_backpressure_autopilot_latest.json",
            ("storage_backpressure_autopilot.py", "storage-backpressure-autopilot"),
        ),
        (
            "storage_pressure_clearance",
            "governance/health/storage_pressure_clearance_latest.json",
            ("storage_pressure_clearance_bot.py", "storage-pressure-clearance"),
        ),
        (
            "system_drift_autopilot",
            "governance/health/system_drift_autopilot_latest.json",
            ("system_drift_autopilot.py", "system-drift-autopilot"),
        ),
    ]
    for _name, raw_path, markers in recovery_artifacts:
        if not any(marker in cmd_text for marker in markers):
            continue
        _path, payload = _load_artifact(project_root, raw_path)
        status_text = str(payload.get("overall_status") or payload.get("status") or "").strip().lower()
        if payload.get("ok") is True and (payload.get("busy") is True or status_text in DEGRADED_STATUSES | READY_STATUSES):
            return True
        if status_text in {"already_running", "busy", "running", "active", "recovering", "recovering_under_guard"}:
            return True
    return False


def _bounded_drift_timeout_attempts(attempts: list[dict[str, Any]]) -> bool:
    failed = [
        row
        for row in attempts
        if isinstance(row, dict) and (bool(row.get("timed_out", False)) or _safe_int(row.get("rc"), 1) not in {0, 2})
    ]
    if not failed:
        return False
    for row in failed:
        if _safe_int(row.get("rc"), 1) != 124:
            return False
        timeout_sec = _safe_int(row.get("timeout_sec"), 0)
        if timeout_sec <= 0 or timeout_sec > 120:
            return False
    return True


def _bounded_drift_safe_repairs(payload: dict[str, Any]) -> bool:
    attempts = payload.get("attempts") if isinstance(payload.get("attempts"), list) else []
    if not attempts:
        return False
    if any(
        isinstance(row, dict) and (bool(row.get("timed_out", False)) or _safe_int(row.get("rc"), 1) not in {0, 2})
        for row in attempts
    ):
        return False
    final_guard = _as_dict(payload.get("final_guard"))
    return _safe_int(final_guard.get("blocked_surface_count"), 0) <= 3


def _storage_clearance_active_recovery(payload: dict[str, Any]) -> bool:
    metrics = payload.get("metrics") if isinstance(payload.get("metrics"), dict) else {}
    return bool(metrics.get("active_storage_pressure", False) or metrics.get("autopilot_active", False))


def _artifact_path_candidates(project_root: Path, raw_path: str | Path) -> list[Path]:
    path = Path(raw_path)
    if path.is_absolute():
        return [path]

    candidates = [project_root / path]
    parts = path.parts
    if parts and parts[0] in {"data", "decisions", "decision_explanations", "exports", "governance", "logs", "models"}:
        candidates.append(project_root / "local_fallback_storage" / path)
        external_root = Path(os.getenv("BOT_LOGS_EXTERNAL_PROJECT_ROOT", "/Volumes/BOT_LOGS/schwab_trading_bot")).expanduser()
        candidates.append(external_root / path)

    out: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        out.append(candidate)
    return out


def _select_freshest_json(project_root: Path, raw_path: str | Path) -> tuple[Path, dict[str, Any]]:
    best_path: Path | None = None
    best_payload: dict[str, Any] = {}
    best_mtime = -1.0
    fallback = _artifact_path_candidates(project_root, raw_path)[0]
    for candidate in _artifact_path_candidates(project_root, raw_path):
        payload = load_json(candidate)
        if not payload:
            continue
        try:
            mtime = candidate.stat().st_mtime
        except Exception:
            mtime = 0.0
        if best_path is None or mtime >= best_mtime:
            best_path = candidate
            best_payload = payload
            best_mtime = mtime
    return best_path or fallback, best_payload


def _health_path(project_root: Path, name: str) -> Path:
    path, _payload = _select_freshest_json(project_root, Path("governance") / "health" / name)
    return path


def _load_artifact(project_root: Path, raw_path: str) -> tuple[Path, dict[str, Any]]:
    return _select_freshest_json(project_root, raw_path)


def _artifact_group_check(
    project_root: Path,
    *,
    name: str,
    family: str,
    specs: list[tuple[str, str]],
    repair_commands: list[list[str]],
    missing_status: str = "degraded",
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    status = "ready"
    for label, raw_path in specs:
        path, payload = _load_artifact(project_root, raw_path)
        artifact_status = _artifact_status(payload, missing=missing_status)
        if _has_timed_out(payload) and not (payload.get("ok") is True and artifact_status == "ready"):
            artifact_status = "blocked"
        if artifact_status == "blocked":
            status = "blocked"
        elif artifact_status == "degraded" and status != "blocked":
            status = "degraded"
        rows.append(
            {
                "name": label,
                "path": str(path),
                "present": bool(payload),
                "status": artifact_status,
                "overall_status": payload.get("overall_status") if payload else "",
                "ok": payload.get("ok") if payload else None,
            }
        )
    summary = ", ".join(f"{row['name']}={row['status']}" for row in rows)
    return _check(
        name,
        family=family,
        status=status,
        summary=summary,
        evidence={"artifacts": rows},
        repair_commands=repair_commands,
    )


def _launchctl_loaded(label: str) -> bool | None:
    try:
        proc = subprocess.run(
            ["launchctl", "print", f"gui/{os.getuid()}/{label}"],
            capture_output=True,
            text=True,
            check=False,
            timeout=2,
        )
    except FileNotFoundError:
        return None
    except Exception:
        return False
    return proc.returncode == 0


def _ps_rows(project_root: Path) -> list[dict[str, Any]]:
    try:
        proc = subprocess.run(
            ["ps", "-ax", "-o", "pid=,ppid=,command="],
            capture_output=True,
            text=True,
            check=False,
            timeout=4,
        )
    except Exception:
        return []
    rows: list[dict[str, Any]] = []
    root_text = str(project_root)
    for raw_line in (proc.stdout or "").splitlines():
        line = raw_line.strip()
        if not line or root_text not in line:
            continue
        parts = line.split(None, 2)
        if len(parts) != 3:
            continue
        try:
            pid = int(parts[0])
            ppid = int(parts[1])
        except Exception:
            continue
        rows.append({"pid": pid, "ppid": ppid, "command": parts[2]})
    return rows


def _command_tokens(command: str) -> list[str]:
    try:
        return shlex.split(command)
    except ValueError:
        return str(command or "").split()


def _has_script_token(tokens: list[str], script: str) -> bool:
    normalized = script.strip("/")
    return any(token.strip("'\"").endswith(normalized) for token in tokens)


def _arg_value(tokens: list[str], flag: str) -> str:
    for idx, token in enumerate(tokens):
        if token == flag and idx + 1 < len(tokens):
            return tokens[idx + 1]
        prefix = f"{flag}="
        if token.startswith(prefix):
            return token[len(prefix) :]
    return ""


def _lane_from_process(command: str) -> str:
    tokens = _command_tokens(command)
    if not tokens:
        return ""
    if _has_script_token(tokens, "scripts/shadow_watchdog.py"):
        return ""
    if _has_script_token(tokens, "scripts/run_all_sleeves.py"):
        return "schwab_all_sleeves"
    if _has_script_token(tokens, "scripts/run_parallel_shadows.py") and "--simulate" in tokens:
        return "parallel_shadows_simulate"
    if _has_script_token(tokens, "scripts/run_dividend_capture_shadow.py"):
        return "dividend_capture_shadow"
    if _has_script_token(tokens, "scripts/run_dividend_shadow.py"):
        return "dividend_shadow"
    if _has_script_token(tokens, "scripts/run_bond_shadow.py"):
        return "bond_shadow"
    if _has_script_token(tokens, "scripts/run_fx_shadow.py"):
        return "fx_shadow"
    if _has_script_token(tokens, "scripts/run_shadow_training_loop.py"):
        broker = _arg_value(tokens, "--broker").strip().lower()
        profile = _arg_value(tokens, "--profile").strip().lower()
        if broker == "coinbase" and profile == "crypto_futures":
            return "coinbase_futures_shadow"
        if broker == "coinbase":
            return "coinbase_shadow"
        if broker == "schwab" and profile == "schwab_futures":
            return "schwab_futures_shadow"
    return ""


def _process_lane_ownership_check(project_root: Path) -> dict[str, Any]:
    classified_rows: list[dict[str, Any]] = []
    pid_lanes: dict[int, str] = {}
    for row in _ps_rows(project_root):
        lane = _lane_from_process(str(row.get("command") or ""))
        if not lane:
            continue
        classified = dict(row)
        classified["lane"] = lane
        classified_rows.append(classified)
        pid_lanes[_safe_int(row.get("pid"), 0)] = lane

    lanes: dict[str, list[dict[str, Any]]] = {}
    ignored_embedded_children: list[dict[str, Any]] = []
    for row in classified_rows:
        lane = str(row.get("lane") or "")
        parent_lane = pid_lanes.get(_safe_int(row.get("ppid"), 0), "")
        if lane == "dividend_shadow" and parent_lane == "dividend_capture_shadow":
            ignored_embedded_children.append(
                {
                    "pid": row.get("pid"),
                    "ppid": row.get("ppid"),
                    "lane": lane,
                    "parent_lane": parent_lane,
                    "reason": "dividend_capture_embedded_child",
                }
            )
            continue
        lanes.setdefault(lane, []).append(row)
    lane_rows: list[dict[str, Any]] = []
    duplicate_lanes: list[str] = []
    excess_processes = 0
    for lane, limit in LANE_OWNER_LIMITS.items():
        owners = lanes.get(lane, [])
        owner_count = len(owners)
        excess = max(owner_count - limit, 0)
        if excess:
            duplicate_lanes.append(lane)
            excess_processes += excess
        lane_rows.append(
            {
                "lane": lane,
                "owner_limit": limit,
                "owner_count": owner_count,
                "excess_process_count": excess,
                "pids": [row.get("pid") for row in owners],
            }
        )
    status = "ready"
    if duplicate_lanes:
        status = "degraded"
    if excess_processes >= 8:
        status = "blocked"
    summary = (
        f"duplicate_lanes={len(duplicate_lanes)} excess_processes={excess_processes}"
        if duplicate_lanes
        else "lane ownership is canonical"
    )
    return _check(
        "process_lane_ownership",
        family="runtime_surface",
        status=status,
        summary=summary,
        evidence={
            "lanes": lane_rows,
            "duplicate_lanes": duplicate_lanes,
            "excess_process_count": excess_processes,
            "ignored_embedded_children": ignored_embedded_children,
        },
        repair_commands=[
            ["./scripts/ops/opsctl.sh", "livefeed-refresh"],
            ["./scripts/ops/opsctl.sh", "start", "--force-restart"],
        ]
        if duplicate_lanes
        else [],
    )


def _one_numbers_check(project_root: Path) -> dict[str, Any]:
    try:
        guard_payload = one_numbers_regression_guard.build_payload(project_root)
    except Exception as exc:
        return _check(
            "one_numbers_original_coverage",
            family="analytics_surface",
            status="blocked",
            summary=f"One Numbers guard could not run: {exc}",
            repair_commands=[["./scripts/ops/opsctl.sh", "one-numbers-regression-guard", "--json"]],
        )
    weaknesses = guard_payload.get("weaknesses") if isinstance(guard_payload.get("weaknesses"), list) else []
    weakness_names = {str(row.get("name") or "") for row in weaknesses if isinstance(row, dict)}
    status = _artifact_status(guard_payload, missing="blocked")
    if weakness_names.intersection({"summary_missing", "latest_csv_alias_missing", "latest_metrics_alias_missing"}):
        status = "blocked"
    elif weakness_names:
        status = "degraded"
    contract = guard_payload.get("original_coverage_contract") if isinstance(guard_payload.get("original_coverage_contract"), dict) else {}
    repair_plan = guard_payload.get("repair_plan") if isinstance(guard_payload.get("repair_plan"), dict) else {}
    repair_commands: list[list[str]] = [["./scripts/ops/opsctl.sh", "one-numbers-regression-guard", "--apply", "--json"]]
    for cmd in repair_plan.get("backfill_commands") or []:
        if isinstance(cmd, list):
            repair_commands.append([str(part) for part in cmd])
    summary = "One Numbers original coverage is pinned and rollup source days are represented"
    if weakness_names:
        summary = ", ".join(sorted(weakness_names))
    return _check(
        "one_numbers_original_coverage",
        family="analytics_surface",
        status=status,
        summary=summary,
        evidence={
            "requested_day": guard_payload.get("requested_day"),
            "resolved_day": guard_payload.get("resolved_day"),
            "history_days_available": guard_payload.get("history_days_available"),
            "expected_start_day": contract.get("expected_start_day"),
            "expected_start_source": contract.get("expected_start_source"),
            "earliest_history_day": contract.get("earliest_history_day"),
            "earliest_source_day": contract.get("earliest_source_day"),
            "source_days_missing_from_history_count": contract.get("source_days_missing_from_history_count"),
            "weaknesses": sorted(weakness_names),
        },
        repair_commands=repair_commands,
    )


def _sql_ingestion_check(project_root: Path) -> dict[str, Any]:
    _path, payload = _load_artifact(project_root, "governance/health/ingestion_storage_control_latest.json")
    if not payload:
        return _check(
            "sql_ingestion_lag_and_backlog",
            family="storage_surface",
            status="blocked",
            summary="ingestion_storage_control_latest.json is missing",
            repair_commands=[["./scripts/ops/opsctl.sh", "ingestion-storage-control", "--json"]],
        )
    backpressure = payload.get("backpressure") if isinstance(payload.get("backpressure"), dict) else {}
    storage = payload.get("storage") if isinstance(payload.get("storage"), dict) else {}
    steady_state = payload.get("steady_state") if isinstance(payload.get("steady_state"), dict) else {}
    target_status = steady_state.get("target_status") if isinstance(steady_state.get("target_status"), dict) else {}
    steady_state_ready = bool(target_status.get("steady_state_ready", False))
    backlog_truth = payload.get("backlog_truth") if isinstance(payload.get("backlog_truth"), dict) else {}
    raw_live_truth = backlog_truth.get("raw_live") if isinstance(backlog_truth.get("raw_live"), dict) else {}
    raw_live_expansion = (
        payload.get("raw_live_expansion_contract")
        if isinstance(payload.get("raw_live_expansion_contract"), dict)
        else {}
    )
    soak_contract = (
        payload.get("continuous_run_soak_contract")
        if isinstance(payload.get("continuous_run_soak_contract"), dict)
        else {}
    )
    soak_contract_inputs = (
        soak_contract.get("inputs")
        if isinstance(soak_contract.get("inputs"), dict)
        else {}
    )
    soak_contract_blockers = soak_contract.get("blockers") if isinstance(soak_contract.get("blockers"), list) else []
    bounded_soak_backlog_ready = bool(
        str(payload.get("overall_status") or "").strip().lower() == "ready"
        and str(payload.get("severity") or "").strip().lower() in {"stable", "low"}
        and str(payload.get("recovery_state") or "").strip().lower() in {"steady_state", "stabilized_recovery", ""}
        and bool(soak_contract.get("active", False))
        and bool(soak_contract.get("soak_ready", False))
        and not soak_contract_blockers
        and bool(
            soak_contract_inputs.get("bounded_sparse_reserve_soak_watch", False)
            or "bounded_sparse_and_raw_reserve_backlog_allowed_for_soak"
            in {str(item) for item in soak_contract.get("non_blocking_conditions", []) if str(item).strip()}
        )
    )
    raw_live_soak_backlog_ready = bool(
        str(payload.get("overall_status") or "").strip().lower() == "ready"
        and str(payload.get("severity") or "").strip().lower() in {"stable", "low"}
        and str(payload.get("recovery_state") or "").strip().lower() in {"steady_state", "stabilized_recovery", ""}
        and str(raw_live_truth.get("grade") or "").strip().upper() in {"A", "A+"}
        and bool(raw_live_expansion.get("expansion_ready", False))
        and not bool(raw_live_expansion.get("hard_block", False))
        and _safe_int(raw_live_truth.get("core_pending_lines"), 0) <= 5000
        and _safe_int(raw_live_truth.get("total_pending_lines"), 0) <= 15000
        and _safe_float(raw_live_truth.get("oldest_pending_age_seconds"), 0.0) <= 900.0
    )
    bounded_soak_backlog_ready = bool(bounded_soak_backlog_ready or raw_live_soak_backlog_ready)
    status = _artifact_status(payload)
    pending_lines = _safe_int(backpressure.get("total_pending_lines"), 0)
    drain_status = str(storage.get("backlog_drain_status") or "").strip()
    severity = str(payload.get("severity") or "").strip().lower()
    recovery_state = str(payload.get("recovery_state") or "").strip()
    if pending_lines > 0 and status == "ready" and not steady_state_ready and not bounded_soak_backlog_ready:
        status = "degraded"
    if severity == "critical" and recovery_state not in {"stabilized_recovery", "recovering_under_guard"}:
        status = "blocked"
    summary = f"pending_lines={pending_lines} drain_status={drain_status or 'unknown'} storage_status={payload.get('overall_status') or 'unknown'}"
    if bounded_soak_backlog_ready:
        summary += " bounded_soak_backlog=ready"
    return _check(
        "sql_ingestion_lag_and_backlog",
        family="storage_surface",
        status=status,
        summary=summary,
        evidence={
            "overall_status": payload.get("overall_status"),
            "severity": payload.get("severity"),
            "recovery_state": recovery_state,
            "pending_lines": pending_lines,
            "estimated_total_drain_minutes": _safe_float(backpressure.get("estimated_total_drain_minutes"), 0.0),
            "backlog_drain_status": drain_status,
            "bounded_soak_backlog_ready": bounded_soak_backlog_ready,
            "raw_live_soak_backlog_ready": raw_live_soak_backlog_ready,
            "raw_live_grade": raw_live_truth.get("grade"),
            "raw_live_core_pending_lines": _safe_int(raw_live_truth.get("core_pending_lines"), 0),
            "raw_live_total_pending_lines": _safe_int(raw_live_truth.get("total_pending_lines"), 0),
            "raw_live_oldest_pending_age_seconds": _safe_float(raw_live_truth.get("oldest_pending_age_seconds"), 0.0),
            "soak_contract_status": soak_contract.get("status"),
            "soak_contract_grade": soak_contract.get("grade"),
        },
        repair_commands=[
            ["./scripts/ops/opsctl.sh", "storage-pressure-clearance", "--apply", "--force-clear-stale-gate", "--json"],
            ["./scripts/ops/opsctl.sh", "storage-backpressure-autopilot", "--apply", "--json"],
        ],
    )


def _storage_route_check(project_root: Path) -> dict[str, Any]:
    _path, payload = _load_artifact(project_root, "governance/health/storage_route_status_latest.json")
    if not payload:
        return _check(
            "external_drive_route_health",
            family="storage_surface",
            status="blocked",
            summary="storage_route_status_latest.json is missing",
            repair_commands=[["./scripts/ops/opsctl.sh", "storage-resilience", "--json"]],
        )
    route_verification = payload.get("route_verification") if isinstance(payload.get("route_verification"), dict) else {}
    verification_state = str(route_verification.get("verification_state") or "").strip()
    conflicts = _safe_int(payload.get("split_brain_conflicts"), 0)
    resilience = load_json(_health_path(project_root, "storage_resilience_control_latest.json"))
    reconciler = load_json(_health_path(project_root, "storage_split_brain_reconciler_latest.json"))
    reconciler_summary = _as_dict(reconciler.get("summary"))
    unresolved_resilience = _safe_int(resilience.get("unresolved_split_brain_conflicts"), conflicts)
    unresolved_reconciler = _safe_int(reconciler_summary.get("unresolved_conflicts"), conflicts)
    mount_guard = load_json(_health_path(project_root, "storage_mount_guard_latest.json"))
    mode = str(payload.get("certified_mode") or payload.get("mode") or "").strip().lower()
    intentional_local_hot_route = bool(
        mode.startswith("local_fallback")
        and verification_state == "active_local_ready"
        and not bool(mount_guard.get("external_required_for_hot_path", True))
        and bool(mount_guard.get("hot_storage_available", False))
        and conflicts == 0
        and unresolved_resilience == 0
        and unresolved_reconciler == 0
        and _guarded_paper_strict_clear(project_root)
    )
    reconciled_legacy_split_brain = bool(
        conflicts > 0
        and verification_state in {"ready", "verified", "curated_ready"}
        and _artifact_status(resilience, missing="degraded") == "ready"
        and unresolved_resilience == 0
        and unresolved_reconciler == 0
        and _guarded_paper_strict_clear(project_root)
    )
    status = "ready" if (
        verification_state in {"ready", "verified", "curated_ready"} and conflicts == 0
    ) or intentional_local_hot_route else "degraded"
    if verification_state in {"blocked", "missing_external_copy"} or conflicts > 0:
        status = "blocked"
    if reconciled_legacy_split_brain:
        status = "degraded"
    summary = f"mode={payload.get('mode') or 'unknown'} verification_state={verification_state or 'unknown'} split_brain_conflicts={conflicts}"
    if reconciled_legacy_split_brain:
        summary += " reconciled_legacy_split_brain=1"
    return _check(
        "external_drive_route_health",
        family="storage_surface",
        status=status,
        summary=summary,
        evidence={
            "mode": payload.get("mode"),
            "certified_mode": payload.get("certified_mode"),
            "active_root": payload.get("active_root"),
            "verification_state": verification_state,
            "split_brain_conflicts": conflicts,
            "unresolved_split_brain_conflicts": unresolved_resilience,
            "reconciler_unresolved_conflicts": unresolved_reconciler,
            "reconciled_legacy_split_brain": reconciled_legacy_split_brain,
            "intentional_local_hot_route": intentional_local_hot_route,
            "external_required_for_hot_path": bool(mount_guard.get("external_required_for_hot_path", True)),
            "hot_storage_available": bool(mount_guard.get("hot_storage_available", False)),
        },
        repair_commands=[["./scripts/ops/opsctl.sh", "storage-resilience", "--json"]],
    )


def _stateful_storage_regression_check(project_root: Path) -> dict[str, Any]:
    path, payload = _load_artifact(project_root, "governance/health/stateful_storage_regression_guard_latest.json")
    if not payload:
        return _check(
            "stateful_storage_regression",
            family="storage_surface",
            status="degraded",
            summary="stateful_storage_regression_guard_latest.json is missing",
            repair_commands=[["./scripts/ops/opsctl.sh", "stateful-storage-regression-guard", "--apply", "--json"]],
        )
    status = _artifact_status(payload, missing="degraded")
    metrics = payload.get("metrics") if isinstance(payload.get("metrics"), dict) else {}
    checks = payload.get("checks") if isinstance(payload.get("checks"), list) else []
    blocked = _safe_int(metrics.get("blocked_check_count"), 0)
    degraded = _safe_int(metrics.get("degraded_check_count"), 0)
    if blocked:
        status = "blocked"
    elif degraded and status == "ready":
        status = "degraded"
    summary = (
        f"local_stateful_gb={_safe_float(metrics.get('local_stateful_gb'), 0.0):.3f} "
        f"blocked={blocked} degraded={degraded}"
    )
    return _check(
        "stateful_storage_regression",
        family="storage_surface",
        status=status,
        summary=summary,
        evidence={"path": str(path), "metrics": metrics, "checks": checks},
        repair_commands=[["./scripts/ops/opsctl.sh", "stateful-storage-regression-guard", "--apply", "--json"]],
    )


def _report_browser_jobs_check(project_root: Path) -> dict[str, Any]:
    specs = [
        ("chrome_headless_guard", "governance/health/chrome_headless_guard_latest.json", ["./scripts/ops/opsctl.sh", "chrome-headless-guard", "--apply", "--json"]),
        ("report_pdf_bundle", "governance/health/report_pdf_bundle_latest.json", ["./scripts/ops/opsctl.sh", "report-pdfs", "--json"]),
        ("system_summary_autopilot", "governance/health/system_summary_autopilot_latest.json", ["./scripts/ops/opsctl.sh", "system-summary-autopilot", "--json"]),
    ]
    rows: list[dict[str, Any]] = []
    repair_commands: list[list[str]] = []
    status = "ready"
    for name, raw_path, command in specs:
        path, payload = _load_artifact(project_root, raw_path)
        artifact_status = _artifact_status(payload, missing="degraded")
        if not payload:
            artifact_status = "degraded"
        if _has_timed_out(payload) and not (payload.get("ok") is True and artifact_status == "ready"):
            artifact_status = "blocked"
        if artifact_status == "blocked":
            status = "blocked"
        elif artifact_status == "degraded" and status != "blocked":
            status = "degraded"
        rows.append({"name": name, "status": artifact_status, "path": str(path), "present": bool(payload)})
        if artifact_status != "ready":
            repair_commands.append(command)
    summary = ", ".join(f"{row['name']}={row['status']}" for row in rows)
    return _check(
        "stuck_report_pdf_browser_jobs",
        family="reporting_surface",
        status=status,
        summary=summary,
        evidence={"artifacts": rows},
        repair_commands=repair_commands,
    )


def _governance_freshness_check(project_root: Path) -> dict[str, Any]:
    _path, payload = _load_artifact(project_root, "governance/health/system_drift_guard_latest.json")
    if not payload:
        return _check(
            "governance_artifact_freshness",
            family="governance_surface",
            status="blocked",
            summary="system_drift_guard_latest.json is missing",
            repair_commands=[["./scripts/ops/opsctl.sh", "system-drift-guard", "--json"]],
        )
    metrics = payload.get("metrics") if isinstance(payload.get("metrics"), dict) else {}
    blocked = _safe_int(metrics.get("blocked_surface_count"), 0)
    degraded = _safe_int(metrics.get("degraded_surface_count"), 0)
    stale = _safe_int(metrics.get("stale_surface_count"), 0)
    missing = _safe_int(metrics.get("missing_surface_count"), 0)
    surfaces = payload.get("surfaces") if isinstance(payload.get("surfaces"), list) else []
    managed_stale = sum(
        1
        for row in surfaces
        if isinstance(row, dict) and bool(row.get("stale", False)) and bool(row.get("managed_stale", False))
    )
    unmanaged_stale = max(stale - managed_stale, 0)
    status = _artifact_status(payload)
    blocked_names = _blocked_surface_names(payload)
    self_referential_blocked = bool(blocked_names) and blocked_names <= {"master_infrastructure_supervisor"}
    degraded_names = {
        str(row.get("name") or "").strip()
        for row in surfaces
        if isinstance(row, dict)
        and str(row.get("status") or "").strip().lower() in {"degraded", "warn", "warning", "needs_work"}
        and str(row.get("name") or "").strip()
    }
    self_referential_degraded = bool(
        degraded > 0
        and degraded == len(degraded_names)
        and degraded_names <= {"master_infrastructure_supervisor"}
        and blocked == 0
        and missing == 0
        and unmanaged_stale == 0
        and _guarded_paper_strict_clear(project_root)
    )
    if (blocked and not self_referential_blocked) or missing:
        status = "blocked"
    elif (degraded and not self_referential_degraded) or unmanaged_stale or self_referential_blocked:
        status = "degraded"
    elif self_referential_degraded:
        status = "ready"
    summary = (
        f"blocked={blocked} degraded={degraded} stale={stale} "
        f"managed_stale={managed_stale} missing={missing}"
    )
    return _check(
        "governance_artifact_freshness",
        family="governance_surface",
        status=status,
        summary=summary,
        evidence={
            "metrics": metrics,
            "managed_stale_surface_count": managed_stale,
            "unmanaged_stale_surface_count": unmanaged_stale,
            "self_referential_degraded_reconciled": self_referential_degraded,
            "degraded_surface_names": sorted(degraded_names),
        },
        repair_commands=[["./scripts/ops/opsctl.sh", "system-drift-autopilot", "--apply", "--json"]],
    )


def _command_surface_check(project_root: Path) -> dict[str, Any]:
    command_validity_path, command_validity = _load_artifact(project_root, "governance/health/command_validity_latest.json")
    commands_hygiene_path, commands_hygiene = _load_artifact(project_root, "governance/health/commands_hygiene_latest.json")
    validity_metrics = command_validity.get("metrics") if isinstance(command_validity.get("metrics"), dict) else {}
    hygiene_metrics = commands_hygiene.get("metrics") if isinstance(commands_hygiene.get("metrics"), dict) else {}
    blocked_entries = _safe_int(validity_metrics.get("blocked_entry_count"), 0)
    smoke_failures = _safe_int(validity_metrics.get("smoke_failure_count"), 0)
    runtime_smoke_failures = _safe_int(validity_metrics.get("runtime_smoke_failure_count"), 0)
    contract_probe_failures = _safe_int(validity_metrics.get("contract_dispatch_smoke_failure_count"), 0)
    contract_hash_mismatches = _safe_int(validity_metrics.get("contract_hash_mismatch_count"), 0)
    unprobed_operator_gated = _safe_int(validity_metrics.get("unprobed_operator_gated_count"), 0)
    commands_changed = bool(commands_hygiene.get("commands_changed", False))
    runbook_changed = bool(commands_hygiene.get("runbook_changed", False))
    status = "ready"
    if (
        not command_validity
        or not commands_hygiene
        or blocked_entries
        or smoke_failures
        or runtime_smoke_failures
        or contract_probe_failures
        or contract_hash_mismatches
    ):
        status = "blocked"
    elif commands_changed or runbook_changed or unprobed_operator_gated or _artifact_status(commands_hygiene, missing="degraded") != "ready":
        status = "degraded"
    summary = (
        f"blocked_entries={blocked_entries} smoke_failures={smoke_failures} "
        f"runtime_smoke_failures={runtime_smoke_failures} contract_probe_failures={contract_probe_failures} "
        f"contract_hash_mismatches={contract_hash_mismatches} unprobed_operator_gated={unprobed_operator_gated} "
        f"commands_changed={int(commands_changed)} runbook_changed={int(runbook_changed)}"
    )
    return _check(
        "command_docs_vs_opsctl_routes",
        family="command_surface",
        status=status,
        summary=summary,
        evidence={
            "command_validity_status": command_validity.get("overall_status"),
            "commands_hygiene_status": commands_hygiene.get("overall_status"),
            "command_validity_path": str(command_validity_path),
            "commands_hygiene_path": str(commands_hygiene_path),
            "validity_metrics": validity_metrics,
            "hygiene_metrics": hygiene_metrics,
        },
        repair_commands=[
            ["./scripts/ops/opsctl.sh", "commands-hygiene", "--apply", "--json"],
            ["./scripts/ops/opsctl.sh", "command-validity", "--apply", "--json"],
        ],
    )


def _child_bot_outcomes_check(project_root: Path) -> dict[str, Any]:
    _path, payload = _load_artifact(project_root, "governance/health/infrastructure_autofix_bot_latest.json")
    if not payload:
        return _check(
            "child_repair_bot_outcomes",
            family="infrastructure_surface",
            status="blocked",
            summary="infrastructure_autofix_bot_latest.json is missing",
            repair_commands=[["./scripts/ops/opsctl.sh", "infrastructure-autofix", "--json"]],
        )
    repair_plan = payload.get("repair_plan") if isinstance(payload.get("repair_plan"), list) else []
    attempts = payload.get("attempts") if isinstance(payload.get("attempts"), list) else []
    operator_followups = payload.get("operator_followups") if isinstance(payload.get("operator_followups"), list) else []
    failed_attempts_all = [
        row
        for row in attempts
        if isinstance(row, dict) and (bool(row.get("timed_out", False)) or _safe_int(row.get("rc"), 1) not in {0, 2})
    ]
    mitigated_attempts = [row for row in failed_attempts_all if _attempt_has_active_recovery(project_root, row)]
    failed_attempts = [row for row in failed_attempts_all if row not in mitigated_attempts]
    timed_out = any(bool(row.get("timed_out", False)) for row in failed_attempts if isinstance(row, dict))
    status = _artifact_status(payload)
    if operator_followups or timed_out or failed_attempts:
        status = "blocked"
    elif mitigated_attempts:
        status = "degraded"
    elif repair_plan or status != "ready":
        status = "degraded"
    paper_soak_advisory_only = bool(
        status == "degraded"
        and _guarded_paper_strict_clear(project_root)
        and not operator_followups
        and not failed_attempts
        and not timed_out
    )
    if paper_soak_advisory_only:
        status = "ready"
    summary = (
        f"repair_plan={len(repair_plan)} attempts={len(attempts)} "
        f"operator_followups={len(operator_followups)} failed_attempts={len(failed_attempts)} "
        f"mitigated_active_recovery_attempts={len(mitigated_attempts)}"
    )
    return _check(
        "child_repair_bot_outcomes",
        family="infrastructure_surface",
        status=status,
        summary=summary,
        evidence={
            "overall_status": payload.get("overall_status"),
            "repair_plan_count": len(repair_plan),
            "attempt_count": len(attempts),
            "operator_followups": operator_followups,
            "failed_attempt_count": len(failed_attempts),
            "mitigated_active_recovery_attempt_count": len(mitigated_attempts),
            "paper_soak_advisory_only": paper_soak_advisory_only,
        },
        repair_commands=[["./scripts/ops/opsctl.sh", "infrastructure-autofix", "--apply", "--json"]],
    )


def _launchd_job_health_check(project_root: Path) -> dict[str, Any]:
    installer = project_root / "scripts" / "ops" / "install_ops_automation_launchd.sh"
    if not installer.exists():
        return _check(
            "launchd_job_health",
            family="infrastructure_surface",
            status="ready",
            summary="ops automation installer is not present in this workspace fixture",
        )
    agents_dir = Path.home() / "Library" / "LaunchAgents"
    rows: list[dict[str, Any]] = []
    status = "ready"
    for label, run_script_rel in LAUNCHD_JOB_SPECS:
        run_script = project_root / run_script_rel
        plist_path = agents_dir / f"{label}.plist"
        loaded = _launchctl_loaded(label) if plist_path.exists() else False
        row_status = "ready"
        if not run_script.exists():
            row_status = "blocked"
        elif not plist_path.exists() or loaded is False:
            row_status = "degraded"
        if row_status == "blocked":
            status = "blocked"
        elif row_status == "degraded" and status != "blocked":
            status = "degraded"
        rows.append(
            {
                "label": label,
                "run_script": str(run_script),
                "run_script_exists": run_script.exists(),
                "plist_path": str(plist_path),
                "plist_exists": plist_path.exists(),
                "loaded": loaded,
                "status": row_status,
            }
        )
    summary = f"jobs={len(rows)} degraded={sum(1 for row in rows if row['status'] == 'degraded')} blocked={sum(1 for row in rows if row['status'] == 'blocked')}"
    return _check(
        "launchd_job_health",
        family="infrastructure_surface",
        status=status,
        summary=summary,
        evidence={"jobs": rows, "installer": str(installer)},
        repair_commands=[["./scripts/ops/install_ops_automation_launchd.sh"]],
    )


def _autonomous_recovery_drills_check(project_root: Path) -> dict[str, Any]:
    return _artifact_group_check(
        project_root,
        name="autonomous_recovery_drills",
        family="resilience_surface",
        specs=[
            ("storage_disaster_recovery", "governance/health/storage_disaster_recovery_latest.json"),
            ("chaos_drill_coordinator", "governance/health/chaos_drill_coordinator_latest.json"),
            ("storage_resilience", "governance/health/storage_resilience_control_latest.json"),
        ],
        repair_commands=[
            ["./scripts/ops/opsctl.sh", "storage-disaster-recovery", "--json"],
            ["./scripts/ops/opsctl.sh", "chaos-drills", "--json"],
            ["./scripts/ops/opsctl.sh", "storage-resilience", "--json"],
        ],
    )


def _operator_cockpit_check(project_root: Path) -> dict[str, Any]:
    return _artifact_group_check(
        project_root,
        name="operator_cockpit_readiness",
        family="operator_surface",
        specs=[
            ("operator_cockpit", "governance/health/operator_cockpit_latest.json"),
            ("runtime_gate_dashboard", "governance/health/runtime_gate_dashboard_latest.json"),
            ("platform_control_plane", "governance/health/platform_control_plane_latest.json"),
        ],
        repair_commands=[
            ["./scripts/ops/opsctl.sh", "operator-cockpit", "--json"],
            ["./scripts/ops/opsctl.sh", "dashboard-refresh", "--json"],
            ["./scripts/ops/opsctl.sh", "platform-control-plane", "--json"],
        ],
    )


def _cold_lane_research_factory_check(project_root: Path) -> dict[str, Any]:
    check = _artifact_group_check(
        project_root,
        name="cold_lane_research_factory",
        family="research_surface",
        specs=[
            ("cold_lane_refresh", "governance/health/cold_lane_refresh_latest.json"),
            ("coverage_gap_closer", "governance/walk_forward/coverage_gap_closer_latest.json"),
            ("immutable_experiment_ledger", "governance/experiments/immutable_experiment_ledger_latest.json"),
            ("promotion_autopilot", "governance/champion_challenger/promotion_autopilot_packet_latest.json"),
        ],
        repair_commands=[
            ["./scripts/ops/opsctl.sh", "cold-lane-refresh", "--json"],
            ["./scripts/ops/opsctl.sh", "coverage-gap-closer", "--json"],
            ["./scripts/ops/opsctl.sh", "experiment-ledger", "--event-type", "control_plane_probe", "--name", "cold_lane_factory_probe"],
            ["./scripts/ops/opsctl.sh", "promotion-autopilot", "--json"],
        ],
    )
    if check["status"] == "degraded" and _guarded_paper_strict_clear(project_root):
        artifacts = check.get("evidence", {}).get("artifacts") if isinstance(check.get("evidence"), dict) else []
        degraded_names = [
            str(row.get("name") or "")
            for row in artifacts
            if isinstance(row, dict) and str(row.get("status") or "") == "degraded"
        ]
        if degraded_names and set(degraded_names).issubset(
            {"coverage_gap_closer", "immutable_experiment_ledger", "promotion_autopilot"}
        ):
            check["status"] = "ready"
            check["ok"] = True
            check["summary"] += ", paper_soak_advisory_only=true"
            check["evidence"]["paper_soak_advisory_only"] = True
            check["evidence"]["managed_degraded_artifacts"] = degraded_names
    return check


def _point_in_time_replay_check(project_root: Path) -> dict[str, Any]:
    check = _artifact_group_check(
        project_root,
        name="point_in_time_replay",
        family="replay_surface",
        specs=[
            ("point_in_time_event_store", "governance/health/point_in_time_event_store_latest.json"),
            ("replay_hash_registry", "governance/health/replay_hash_registry_guard_latest.json"),
            ("golden_replay_regression", "governance/health/golden_replay_regression_latest.json"),
            ("replay_end_to_end", "governance/health/replay_end_to_end_latest.json"),
        ],
        repair_commands=[
            ["./scripts/ops/opsctl.sh", "point-in-time-event-store", "--json"],
            ["./scripts/ops/opsctl.sh", "replay-hash-registry", "--json"],
            ["./scripts/ops/opsctl.sh", "golden-replay-regression", "--json"],
        ],
    )
    point_store = load_json(_health_path(project_root, "point_in_time_event_store_latest.json"))
    event_count = _safe_int(point_store.get("event_count"), 0)
    if point_store and event_count <= 0 and check["status"] == "ready":
        check["status"] = "degraded"
        check["ok"] = False
        check["summary"] += ", point_in_time_event_store_event_count=0"
    check["evidence"]["point_in_time_event_count"] = event_count
    return check


def _backlog_organizer_paper_soak_advisory(payload: dict[str, Any]) -> bool:
    summary = _as_dict(payload.get("summary"))
    if not bool(summary.get("guarded_paper_soak_green", False)):
        return False
    lanes = [row for row in payload.get("lanes") or [] if isinstance(row, dict)]
    hard_blocked = [
        row
        for row in lanes
        if str(row.get("status") or "").strip().lower() in {"blocked", "critical", "failed"}
    ]
    managed_hard_lane_ids = {"admission_contracts", "promotion_training_quality"}
    if any(str(row.get("lane_id") or "").strip() not in managed_hard_lane_ids for row in hard_blocked):
        return False
    operational_lane_ids = {"runtime_pressure", "health_visibility", "auth_runtime_separation", "admission_contracts"}
    operational_rows = {
        str(row.get("lane_id") or "").strip(): str(row.get("status") or "").strip().lower()
        for row in lanes
        if str(row.get("lane_id") or "").strip() in operational_lane_ids
    }
    required_runtime_lanes = {"runtime_pressure", "auth_runtime_separation"}
    if not required_runtime_lanes <= set(operational_rows):
        return False
    if any(operational_rows[lane_id] not in {"ready", "advisory"} for lane_id in required_runtime_lanes):
        return False
    if "health_visibility" in operational_rows and operational_rows["health_visibility"] not in {"ready", "advisory"}:
        return False
    return operational_rows.get("admission_contracts", "ready") in {"ready", "advisory", "blocked"}


def _self_auditing_infra_bots_check(project_root: Path) -> dict[str, Any]:
    expected = [
        ("one_numbers_regression_guard", "governance/health/one_numbers_regression_guard_latest.json"),
        ("infrastructure_autofix", "governance/health/infrastructure_autofix_bot_latest.json"),
        ("system_drift_autopilot", "governance/health/system_drift_autopilot_latest.json"),
        ("storage_backpressure_autopilot", "governance/health/storage_backpressure_autopilot_latest.json"),
        ("storage_pressure_clearance", "governance/health/storage_pressure_clearance_latest.json"),
        ("stateful_storage_regression_guard", "governance/health/stateful_storage_regression_guard_latest.json"),
        ("schwab_auth_supervisor", "governance/health/schwab_auth_supervisor_latest.json"),
        ("command_validity", "governance/health/command_validity_latest.json"),
        ("chrome_headless_guard", "governance/health/chrome_headless_guard_latest.json"),
        ("backlog_organizer", "governance/health/backlog_organizer_latest.json"),
    ]
    rows: list[dict[str, Any]] = []
    status = "ready"
    guarded_paper_clear = _guarded_paper_strict_clear(project_root)
    paper_soak_advisory_bots = {
        "infrastructure_autofix",
        "system_drift_autopilot",
        "storage_backpressure_autopilot",
        "storage_pressure_clearance",
        "backlog_organizer",
    }
    for label, raw_path in expected:
        path, payload = _load_artifact(project_root, raw_path)
        artifact_age_minutes = payload_age_minutes(payload, path) if payload else None
        artifact_status = _artifact_status(payload, missing="degraded")
        initial_artifact_status = artifact_status
        managed_bounded_drift_repair = False
        if label == "command_validity" and payload:
            metrics = payload.get("metrics") if isinstance(payload.get("metrics"), dict) else {}
            if (
                _safe_int(metrics.get("blocked_entry_count"), 0) == 0
                and _safe_int(metrics.get("smoke_failure_count"), 0) == 0
                and _safe_int(metrics.get("runtime_smoke_failure_count"), 0) == 0
                and _safe_int(metrics.get("contract_dispatch_smoke_failure_count"), 0) == 0
                and _safe_int(metrics.get("contract_hash_mismatch_count"), 0) == 0
            ):
                artifact_status = "ready"
        has_status = bool(str(payload.get("overall_status") or payload.get("status") or "").strip()) if payload else False
        has_timestamp = bool(str(payload.get("timestamp_utc") or payload.get("updated_at_utc") or "").strip()) if payload else False
        repair_plan = payload.get("repair_plan") if isinstance(payload.get("repair_plan"), list) else []
        attempts = payload.get("attempts") if isinstance(payload.get("attempts"), list) else []
        operator_followups = payload.get("operator_followups") if isinstance(payload.get("operator_followups"), list) else []
        failed_attempts = [
            row
            for row in attempts
            if isinstance(row, dict) and (bool(row.get("timed_out", False)) or _safe_int(row.get("rc"), 1) not in {0, 2})
        ]
        mitigated_attempts = [row for row in failed_attempts if _attempt_has_active_recovery(project_root, row)]
        unmitigated_failed_attempts = [row for row in failed_attempts if row not in mitigated_attempts]
        advisory_followups = bool(operator_followups) and {
            str(item)
            for item in operator_followups
            if str(item).strip()
        } <= {"infrastructure_autofix", "master_infrastructure_supervisor"}
        if label == "infrastructure_autofix" and artifact_status == "blocked" and mitigated_attempts and not unmitigated_failed_attempts and not operator_followups:
            artifact_status = "degraded"
        if label == "storage_pressure_clearance" and artifact_status == "blocked" and _storage_clearance_active_recovery(payload):
            artifact_status = "degraded"
            unmitigated_failed_attempts = []
        if label == "system_drift_autopilot" and artifact_status == "blocked" and not unmitigated_failed_attempts and advisory_followups:
            artifact_status = "degraded"
        if (
            label == "system_drift_autopilot"
            and artifact_status == "blocked"
            and _bounded_drift_timeout_attempts(attempts)
            and not operator_followups
        ):
            artifact_status = "degraded"
            unmitigated_failed_attempts = []
        if (
            label == "system_drift_autopilot"
            and artifact_status == "blocked"
            and _bounded_drift_safe_repairs(payload)
            and _guarded_paper_strict_clear(project_root)
            and not operator_followups
        ):
            artifact_status = "degraded"
            unmitigated_failed_attempts = []
            managed_bounded_drift_repair = True
        managed_blocking_backlog = bool(
            label == "backlog_organizer"
            and artifact_status == "blocked"
            and _backlog_organizer_paper_soak_advisory(payload)
        )
        if managed_blocking_backlog:
            artifact_status = "degraded"
            unmitigated_failed_attempts = []
        no_action_with_plan = bool(repair_plan and not attempts and str(payload.get("apply") or payload.get("apply_requested") or "").lower() in {"true", "1"})
        row_status = artifact_status
        if payload and (not has_status or not has_timestamp):
            row_status = "blocked"
        elif no_action_with_plan:
            row_status = "degraded"
        if row_status == "blocked":
            status = "blocked"
        elif row_status == "degraded" and status != "blocked":
            status = "degraded"
        paper_soak_advisory_only = bool(
            guarded_paper_clear
            and row_status == "degraded"
            and (initial_artifact_status != "blocked" or managed_blocking_backlog or managed_bounded_drift_repair)
            and label in paper_soak_advisory_bots
            and not unmitigated_failed_attempts
        )
        authoritative_recovery = False
        authoritative_recovery_source = ""
        if guarded_paper_clear and isinstance(artifact_age_minutes, (int, float)) and artifact_age_minutes > 30:
            if label == "system_drift_autopilot":
                current_guard = load_json(_health_path(project_root, "system_drift_guard_latest.json"))
                guard_metrics = _as_dict(current_guard.get("metrics"))
                authoritative_recovery = bool(
                    _artifact_status(current_guard, missing="degraded") == "ready"
                    and _safe_int(guard_metrics.get("blocked_surface_count"), 0) == 0
                    and _safe_int(guard_metrics.get("degraded_surface_count"), 0) == 0
                )
                authoritative_recovery_source = "system_drift_guard"
            elif label in {"storage_backpressure_autopilot", "storage_pressure_clearance"}:
                current_storage = load_json(_health_path(project_root, "ingestion_storage_control_latest.json"))
                storage_backpressure = _as_dict(current_storage.get("backpressure"))
                authoritative_recovery = bool(
                    _artifact_status(current_storage, missing="degraded") == "ready"
                    and str(current_storage.get("severity") or "stable").strip().lower() == "stable"
                    and _safe_float(current_storage.get("pressure_index"), 1.0) < 0.5
                    and _safe_int(storage_backpressure.get("total_pending_lines"), 0) <= 15000
                )
                authoritative_recovery_source = "ingestion_storage_control"
        stale_snapshot_superseded = bool(authoritative_recovery and not unmitigated_failed_attempts)
        if stale_snapshot_superseded:
            paper_soak_advisory_only = True
        if paper_soak_advisory_only:
            row_status = "advisory"
        rows.append(
            {
                "name": label,
                "path": str(path),
                "present": bool(payload),
                "status": row_status,
                "has_status": has_status,
                "has_timestamp": has_timestamp,
                "repair_plan_count": len(repair_plan),
                "attempt_count": len(attempts),
                "failed_attempt_count": len(unmitigated_failed_attempts),
                "mitigated_active_recovery_attempt_count": len(mitigated_attempts),
                "paper_soak_advisory_only": paper_soak_advisory_only,
                "artifact_age_minutes": artifact_age_minutes,
                "stale_snapshot_superseded": stale_snapshot_superseded,
                "authoritative_recovery_source": authoritative_recovery_source if stale_snapshot_superseded else "",
            }
        )
    # Rows can be downgraded to advisory after their source status is read, for
    # example when a stale storage-repair snapshot is superseded by current
    # healthy storage evidence. Derive the aggregate from the finalized rows so
    # an earlier blocked value cannot survive after every blocker is reconciled.
    blocked_rows = [row for row in rows if row.get("status") == "blocked"]
    degraded_rows = [row for row in rows if row.get("status") == "degraded"]
    if blocked_rows:
        status = "blocked"
    elif degraded_rows:
        status = "degraded"
    else:
        status = "ready"
    summary = f"bots={len(rows)} degraded={sum(1 for row in rows if row['status'] == 'degraded')} blocked={sum(1 for row in rows if row['status'] == 'blocked')}"
    return _check(
        "self_auditing_infra_bots",
        family="infrastructure_surface",
        status=status,
        summary=summary,
        evidence={"bots": rows},
        repair_commands=[
            ["./scripts/ops/opsctl.sh", "infrastructure-autofix", "--json"],
            ["./scripts/ops/opsctl.sh", "system-drift-autopilot", "--json"],
        ],
    )


def _schwab_auth_supervisor_check(project_root: Path) -> dict[str, Any]:
    payload = load_json(_health_path(project_root, "schwab_auth_supervisor_latest.json"))
    if not payload:
        return _check(
            "schwab_auth_supervisor",
            family="broker_surface",
            status="degraded",
            summary="schwab_auth_supervisor_latest.json is missing",
            repair_commands=[["./scripts/ops/opsctl.sh", "schwab-auth-supervisor", "--json"]],
        )
    status = _artifact_status(payload, missing="degraded")
    findings = payload.get("findings") if isinstance(payload.get("findings"), list) else []
    token = payload.get("token") if isinstance(payload.get("token"), dict) else {}
    callback = payload.get("callback") if isinstance(payload.get("callback"), dict) else {}
    auth_processes = payload.get("auth_processes") if isinstance(payload.get("auth_processes"), list) else []
    stale_count = sum(1 for row in auth_processes if isinstance(row, dict) and row.get("stale") is True)
    if not bool(token.get("ready", False)):
        status = "blocked"
    elif stale_count and status == "ready":
        status = "degraded"
    summary = (
        f"token_ready={int(bool(token.get('ready', False)))} "
        f"expires_in_seconds={_safe_float(token.get('expires_in_seconds'), 0.0):.1f} "
        f"stale_auth_helpers={stale_count} callback_port_in_use={int(bool(callback.get('port_in_use', False)))} "
        f"findings={','.join(str(item) for item in findings[:4])}"
    )
    return _check(
        "schwab_auth_supervisor",
        family="broker_surface",
        status=status,
        summary=summary,
        evidence={
            "overall_status": payload.get("overall_status"),
            "findings": findings,
            "token": token,
            "callback": callback,
            "auth_process_count": len(auth_processes),
            "stale_auth_process_count": stale_count,
            "recent_auth_signals": payload.get("recent_auth_signals") if isinstance(payload.get("recent_auth_signals"), dict) else {},
        },
        repair_commands=[["./scripts/ops/opsctl.sh", "schwab-auth-supervisor", "--apply", "--json"]],
    )


def _coinbase_api_health_check(project_root: Path) -> dict[str, Any]:
    payload = load_json(_health_path(project_root, "coinbase_api_health_latest.json"))
    if not payload:
        return _check(
            "coinbase_api_health",
            family="broker_surface",
            status="degraded",
            summary="coinbase_api_health_latest.json is missing",
            repair_commands=[["./scripts/ops/opsctl.sh", "coinbase-api-health", "--json"]],
        )
    status = _artifact_status(payload, missing="degraded")
    public_market_data = payload.get("public_market_data") if isinstance(payload.get("public_market_data"), dict) else {}
    if not bool(public_market_data.get("ok", False)):
        status = "blocked"
    summary = f"public_market_data_ok={int(bool(public_market_data.get('ok', False)))} symbol={public_market_data.get('symbol') or ''}"
    return _check(
        "coinbase_api_health",
        family="broker_surface",
        status=status,
        summary=summary,
        evidence={
            "overall_status": payload.get("overall_status"),
            "public_market_data": public_market_data,
            "credentials": payload.get("credentials") if isinstance(payload.get("credentials"), dict) else {},
        },
        repair_commands=[["./scripts/ops/opsctl.sh", "coinbase-api-health", "--json"]],
    )


def _maturity_scores(checks: list[dict[str, Any]], envelope_lanes: list[dict[str, Any]]) -> dict[str, Any]:
    blocked_count = sum(1 for row in checks if row.get("status") == "blocked")
    degraded_count = sum(1 for row in checks if row.get("status") == "degraded")
    ready_count = sum(1 for row in checks if row.get("status") == "ready")
    check_count = max(len(checks), 1)
    ready_ratio = ready_count / check_count
    envelope_ready_ratio = sum(1 for row in envelope_lanes if row.get("status") == "ready") / max(len(envelope_lanes), 1)
    process = next((row for row in checks if row.get("name") == "process_lane_ownership"), {})
    process_excess = _safe_int(((process.get("evidence") or {}).get("excess_process_count")), 0) if isinstance(process.get("evidence"), dict) else 0
    command = next((row for row in checks if row.get("name") == "command_docs_vs_opsctl_routes"), {})
    command_ready = 1.0 if command.get("status") == "ready" else 0.0
    one_numbers = next((row for row in checks if row.get("name") == "one_numbers_original_coverage"), {})
    one_numbers_ready = 1.0 if one_numbers.get("status") == "ready" else 0.0
    storage = next((row for row in checks if row.get("name") == "external_drive_route_health"), {})
    storage_ready = 1.0 if storage.get("status") == "ready" else 0.0
    operational = 9.0 - (blocked_count * 0.38) - (degraded_count * 0.16) - min(process_excess * 0.10, 1.2)
    infra = 8.5 + (ready_ratio * 1.0) - (blocked_count * 0.25) - (degraded_count * 0.10)
    feature = 8.4 + (envelope_ready_ratio * 0.6) + (command_ready * 0.2)
    data = 7.8 + (one_numbers_ready * 0.6) + (storage_ready * 0.4) - min(process_excess * 0.04, 0.5)
    autonomy = 6.0 + (ready_ratio * 2.0) + (envelope_ready_ratio * 1.0) - (blocked_count * 0.25)
    scores = {
        "feature_sophistication": round(_clamp(feature, 1.0, 9.4), 2),
        "data_collection_breadth": round(_clamp(data, 1.0, 9.2), 2),
        "infrastructure_control_plane": round(_clamp(infra, 1.0, 9.2), 2),
        "operational_cleanliness": round(_clamp(operational, 1.0, 9.0), 2),
        "unattended_autonomy": round(_clamp(autonomy, 1.0, 8.8), 2),
    }
    scores["target_state"] = {
        "operational_cleanliness_target": 8.0,
        "nine_level_target": 9.0,
        "ready_ratio": round(ready_ratio, 3),
        "envelope_ready_ratio": round(envelope_ready_ratio, 3),
        "blocked_count": blocked_count,
        "degraded_count": degraded_count,
        "process_excess_count": process_excess,
    }
    return scores


def _run_json(cmd: list[str], *, cwd: Path, timeout_sec: int) -> dict[str, Any]:
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(cwd),
            capture_output=True,
            text=True,
            check=False,
            timeout=max(int(timeout_sec), 1),
            env=_child_env("master_infrastructure_supervisor"),
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
    payload: dict[str, Any] = {}
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
        "stdout_tail": "\n".join(stdout.splitlines()[-10:]),
        "stderr_tail": "\n".join(stderr.splitlines()[-10:]),
        "payload": payload,
    }


def build_payload(
    project_root: Path = PROJECT_ROOT,
    *,
    apply: bool = False,
    timeout_sec: int = 900,
) -> dict[str, Any]:
    checks = [
        _one_numbers_check(project_root),
        _sql_ingestion_check(project_root),
        _storage_route_check(project_root),
        _stateful_storage_regression_check(project_root),
        _process_lane_ownership_check(project_root),
        _report_browser_jobs_check(project_root),
        _launchd_job_health_check(project_root),
        _autonomous_recovery_drills_check(project_root),
        _governance_freshness_check(project_root),
        _command_surface_check(project_root),
        _operator_cockpit_check(project_root),
        _cold_lane_research_factory_check(project_root),
        _point_in_time_replay_check(project_root),
        _child_bot_outcomes_check(project_root),
        _self_auditing_infra_bots_check(project_root),
        _schwab_auth_supervisor_check(project_root),
        _coinbase_api_health_check(project_root),
    ]
    blocked = [row for row in checks if row.get("status") == "blocked"]
    degraded = [row for row in checks if row.get("status") == "degraded"]
    overall_status = "blocked" if blocked else ("degraded" if degraded else "ready")
    repair_commands = _unique_commands(
        [
            cmd
            for row in checks
            if row.get("status") != "ready"
            for cmd in list(row.get("repair_commands") or [])
            if isinstance(cmd, list)
        ]
    )

    attempts: list[dict[str, Any]] = []
    if apply:
        for cmd in repair_commands:
            attempts.append(_run_json(cmd, cwd=project_root, timeout_sec=timeout_sec))

    hard_failed_attempts = [
        row
        for row in attempts
        if bool(row.get("timed_out", False)) or _safe_int(row.get("rc"), 1) not in {0, 2}
    ]
    degraded_attempts = [row for row in attempts if _safe_int(row.get("rc"), 1) == 2 and not bool(row.get("timed_out", False))]
    if hard_failed_attempts:
        overall_status = "blocked"
    elif degraded_attempts and overall_status == "ready":
        overall_status = "degraded"
    envelope_lanes = [
        {
            "number": number,
            "name": lane_name,
            "check": check_name,
            "status": next((str(row.get("status") or "") for row in checks if row.get("name") == check_name), "missing"),
        }
        for number, lane_name, check_name in ENVELOPE_LANES
    ]
    blocked_lanes = [row["name"] for row in envelope_lanes if row.get("status") == "blocked"]
    degraded_lanes = [row["name"] for row in envelope_lanes if row.get("status") == "degraded"]
    ready_lanes = [row["name"] for row in envelope_lanes if row.get("status") == "ready"]
    operating_posture = "coherent"
    if blocked_lanes:
        operating_posture = "recovery"
    elif degraded_lanes:
        operating_posture = "guarded_collection"
    maturity_scores = _maturity_scores(checks, envelope_lanes)

    return {
        "timestamp_utc": iso_now(),
        "schema_version": 1,
        "ok": overall_status == "ready",
        "overall_status": overall_status,
        "apply": bool(apply),
        "checks": checks,
        "repair_plan": [{"name": f"repair_{idx + 1}", "cmd": cmd} for idx, cmd in enumerate(repair_commands)],
        "attempts": [
            {
                "cmd": list(row.get("cmd") or []),
                "rc": _safe_int(row.get("rc"), 1),
                "timed_out": bool(row.get("timed_out", False)),
            }
            for row in attempts
        ],
        "metrics": {
            "check_count": len(checks),
            "blocked_check_count": len(blocked),
            "degraded_check_count": len(degraded),
            "repair_command_count": len(repair_commands),
            "hard_failed_attempt_count": len(hard_failed_attempts),
            "degraded_attempt_count": len(degraded_attempts),
        },
        "envelope_lanes": envelope_lanes,
        "platform_posture": {
            "operating_posture": operating_posture,
            "ready_lanes": ready_lanes,
            "degraded_lanes": degraded_lanes,
            "blocked_lanes": blocked_lanes,
            "collection_bias": "protect_live_collection_and_drain_backlog" if operating_posture == "recovery" else "normal",
        },
        "maturity_scores": maturity_scores,
        "hardening_scorecard": {
            "truth_layer_ready": next((row.get("status") for row in checks if row.get("name") == "one_numbers_original_coverage"), "missing") == "ready",
            "storage_route_certified": next((row.get("status") for row in checks if row.get("name") == "external_drive_route_health"), "missing") == "ready",
            "process_ownership_canonical": next((row.get("status") for row in checks if row.get("name") == "process_lane_ownership"), "missing") == "ready",
            "command_surface_clean": next((row.get("status") for row in checks if row.get("name") == "command_docs_vs_opsctl_routes"), "missing") == "ready",
            "self_auditing_bots_current": next((row.get("status") for row in checks if row.get("name") == "self_auditing_infra_bots"), "missing") == "ready",
            "launchd_jobs_installed": next((row.get("status") for row in checks if row.get("name") == "launchd_job_health"), "missing") == "ready",
        },
        "regression_control_map": [
            {
                "surface": "commands_and_runbook",
                "guard": "command_validity_bot",
                "autofix": "commands_hygiene_bot",
                "status": next((str(row.get("status") or "") for row in checks if row.get("name") == "command_docs_vs_opsctl_routes"), "missing"),
            },
            {
                "surface": "one_numbers_original_coverage",
                "guard": "one_numbers_regression_guard",
                "autofix": "system_drift_autopilot",
                "status": next((str(row.get("status") or "") for row in checks if row.get("name") == "one_numbers_original_coverage"), "missing"),
            },
            {
                "surface": "storage_and_backpressure",
                "guard": "ingestion_storage_control",
                "autofix": "storage_pressure_clearance_bot",
                "status": next((str(row.get("status") or "") for row in checks if row.get("name") == "sql_ingestion_lag_and_backlog"), "missing"),
            },
            {
                "surface": "stateful_storage_routes",
                "guard": "stateful_storage_regression_guard",
                "autofix": "infrastructure_autofix_bot",
                "status": next((str(row.get("status") or "") for row in checks if row.get("name") == "stateful_storage_regression"), "missing"),
            },
            {
                "surface": "process_lane_ownership",
                "guard": "master_infrastructure_supervisor",
                "autofix": "livefeed-refresh/start --force-restart",
                "status": next((str(row.get("status") or "") for row in checks if row.get("name") == "process_lane_ownership"), "missing"),
            },
            {
                "surface": "governance_artifact_freshness",
                "guard": "system_drift_guard",
                "autofix": "system_drift_autopilot",
                "status": next((str(row.get("status") or "") for row in checks if row.get("name") == "governance_artifact_freshness"), "missing"),
            },
            {
                "surface": "schwab_auth",
                "guard": "schwab_auth_supervisor",
                "autofix": "schwab_auth_supervisor",
                "status": next((str(row.get("status") or "") for row in checks if row.get("name") == "schwab_auth_supervisor"), "missing"),
            },
            {
                "surface": "child_bot_outcomes",
                "guard": "master_infrastructure_supervisor",
                "autofix": "infrastructure_autofix_bot",
                "status": next((str(row.get("status") or "") for row in checks if row.get("name") == "child_repair_bot_outcomes"), "missing"),
            },
        ],
        "next_capability_paths": [
            {
                "name": "regression_budget_controller",
                "purpose": "Give each surface retry budgets, cooldowns, and escalation rules so noisy repairs do not starve collection.",
            },
            {
                "name": "historical_data_ledger",
                "purpose": "Track first-seen day, raw source days, rollup days, and external/local copy provenance as one audit trail.",
            },
            {
                "name": "recovery_progress_slo",
                "purpose": "Grade active recovery by backlog delta and child-bot freshness instead of treating every old timeout as current failure.",
            },
            {
                "name": "operator_mission_control",
                "purpose": "Condense halt, throttle, collection, training, and incident state into one actionable cockpit.",
            },
        ],
        "infra_dependency_graph": {
            "master_infrastructure_supervisor": [
                "one_numbers_regression_guard",
                "ingestion_storage_control",
                "storage_pressure_clearance_bot",
                "storage_route_status",
                "stateful_storage_regression_guard",
                "launchd_job_health",
                "autonomous_recovery_drills",
                "chrome_headless_guard",
                "report_pdf_bundle",
                "system_summary_autopilot",
                "system_drift_guard",
                "commands_hygiene_bot",
                "command_validity_bot",
                "operator_cockpit",
                "cold_lane_research_factory",
                "point_in_time_replay",
                "infrastructure_autofix_bot",
                "schwab_auth_supervisor",
                "coinbase_api_health",
            ],
            "infrastructure_autofix_bot": [
                "commands_hygiene_bot",
                "command_validity_bot",
                "system_drift_autopilot",
                "storage_pressure_clearance_bot",
                "storage_backpressure_autopilot",
                "stateful_storage_regression_guard",
                "daily_verify_auto_remediation_bot",
            ],
        },
        "operator_followups": ordered_unique(
            [
                "pin the One Numbers original start day in config/one_numbers_start_day.txt or ONE_NUMBERS_ORIGINAL_START_DAY"
                if any(row.get("name") == "one_numbers_original_coverage" and "one_numbers_original_start_unpinned" in str(row.get("summary") or "") for row in checks)
                else "",
                "review child infrastructure bot followups because at least one child repair path cannot complete automatically"
                if any(row.get("name") == "child_repair_bot_outcomes" and row.get("status") == "blocked" for row in checks)
                else "",
            ]
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Parent supervisor for infrastructure bot coherence and One Numbers historical coverage.")
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--out-file", default=str(DEFAULT_OUT_PATH))
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--timeout-sec", type=int, default=900)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    payload = build_payload(project_root, apply=bool(args.apply), timeout_sec=int(args.timeout_sec))
    write_payload(Path(args.out_file).expanduser(), payload)
    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        print(
            "master_infrastructure_supervisor "
            f"overall_status={payload.get('overall_status', '')} "
            f"blocked={int((payload.get('metrics') or {}).get('blocked_check_count', 0) or 0)} "
            f"degraded={int((payload.get('metrics') or {}).get('degraded_check_count', 0) or 0)}"
        )
    return 0 if payload.get("overall_status") in {"ready", "degraded"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
