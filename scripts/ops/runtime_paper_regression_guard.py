#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from scripts.ops.long_runtime_common import iso_now, load_json, ordered_unique, payload_age_minutes, write_payload
else:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    from .long_runtime_common import iso_now, load_json, ordered_unique, payload_age_minutes, write_payload


DEFAULT_OUT_PATH = PROJECT_ROOT / "governance" / "health" / "runtime_paper_regression_guard_latest.json"
DEFAULT_RUNTIME_PATH = PROJECT_ROOT / "governance" / "health" / "runtime_throttle_control_latest.json"
DEFAULT_PAPER_PATH = PROJECT_ROOT / "governance" / "health" / "paper_400_ramp_latest.json"
DEFAULT_OVERRIDE_PATH = PROJECT_ROOT / "config" / ".env.runtime_resource_guard_override"
DEFAULT_PROCESS_PATH = PROJECT_ROOT / "governance" / "health" / "process_watchdog_latest.json"
DEFAULT_RUNTIME_PROFITABILITY_PATH = PROJECT_ROOT / "governance" / "health" / "paper_runtime_profitability_controls_latest.json"
DEFAULT_AUTH_LEASE_PATH = PROJECT_ROOT / "governance" / "health" / "auth_lease_manager_latest.json"
DEFAULT_SCHWAB_AUTH_PATH = PROJECT_ROOT / "governance" / "health" / "schwab_auth_supervisor_latest.json"
DEFAULT_BROKER_PATH = PROJECT_ROOT / "governance" / "health" / "broker_readiness_latest.json"
DEFAULT_SESSION_PATH = PROJECT_ROOT / "governance" / "health" / "session_ready_latest.json"

CAPACITY_BLOCKER = "runtime_capacity_not_ready_for_400_paper"
READY_LIKE_STATUSES = {"ready", "ok", "advisory"}
PRESSURE_PROFILES = {"soft_cap", "sustain", "guarded", "protect"}
HARD_SEVERITIES = {"critical", "high"}
TRUTHY_VALUES = {"1", "true", "yes", "on", "ready", "ok", "armed"}
FALSEY_VALUES = {"0", "false", "no", "off", "disabled", "blocked"}
HARD_PAPER_BLOCKERS = {
    "global_halt_or_clear_blocker_active",
    "ingestion_or_backpressure_above_paper_400_gate",
    "memory_pressure_above_paper_400_gate",
    "auth_not_ready",
    "broker_not_ready",
    "session_not_ready",
    "market_data_not_ready",
}
PAPER_EXPANSION_ONLY_BLOCKERS = {
    "ingestion_or_backpressure_above_paper_400_gate",
    "memory_pressure_above_paper_400_gate",
}
LIVE_AUTHORITY_KEYS = [
    "TOP_BOT_ENABLE_LIVE_EXECUTION",
    "EXECUTION_LANE_LIVE_ENABLED",
    "RUN_ALL_SLEEVES_WITH_LIVE_EXECUTOR",
]
HOT_ARTIFACT_MAX_AGE_MINUTES = {
    "runtime_throttle_control": 30.0,
    "paper_400_ramp": 30.0,
    "process_watchdog": 30.0,
    "paper_runtime_profitability_controls": 120.0,
}
SUPPORT_OVERRIDE_KEYS = [
    "YTDLP_SUPPORT_NICE",
    "MACRO_YTDLP_SUPPORT_NICE",
    "TRAINING_RUNTIME_PAUSED_FOR_HOST_HEADROOM",
    "SHADOW_LOOP_RUNTIME_PAUSE_SLEEP_SECONDS",
]
SUPPORT_OVERRIDE_ALIASES = {
    "YTDLP_SUPPORT_NICE": ["YTDLP_SUPPORT_NICE", "OPS_SUPPORT_JOB_NICE"],
    "MACRO_YTDLP_SUPPORT_NICE": ["MACRO_YTDLP_SUPPORT_NICE", "OPS_SUPPORT_JOB_NICE"],
    "TRAINING_RUNTIME_PAUSED_FOR_HOST_HEADROOM": [
        "TRAINING_RUNTIME_PAUSED_FOR_HOST_HEADROOM",
        "RUNTIME_RESEARCH_TRAINING_NICE",
        "RUNTIME_THROTTLE_RESEARCH_NICE",
    ],
    "SHADOW_LOOP_RUNTIME_PAUSE_SLEEP_SECONDS": ["SHADOW_LOOP_RUNTIME_PAUSE_SLEEP_SECONDS"],
}
PAPER_PAUSE_OVERRIDE_KEYS = {
    "PAPER_EXECUTION_RUNTIME_PAUSED_FOR_PRESSURE": "1",
    "PAPER_EXECUTION_QUEUE_CONSUMER_ENABLED": "0",
    "INLINE_PAPER_EXECUTION_ENABLED": "0",
}
GUARD_BOTS = [
    "runtime_ready_contract_regression_guard_bot",
    "paper_ramp_runtime_blocker_guard_bot",
    "paper_execution_pause_guard_bot",
    "support_spawn_niceness_guard_bot",
    "soak_paper_lane_regression_guard_bot",
]


def _as_dict(raw: Any) -> dict[str, Any]:
    return raw if isinstance(raw, dict) else {}


def _as_list(raw: Any) -> list[Any]:
    return raw if isinstance(raw, list) else []


def _safe_float(raw: Any, default: float = 0.0) -> float:
    try:
        return float(raw)
    except Exception:
        return float(default)


def _bool(raw: Any) -> bool:
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, (int, float)):
        return raw != 0
    return str(raw or "").strip().lower() in TRUTHY_VALUES


def _falsey(raw: Any) -> bool:
    if isinstance(raw, bool):
        return not raw
    if isinstance(raw, (int, float)):
        return raw == 0
    return str(raw or "").strip().lower() in FALSEY_VALUES


def _lower(raw: Any) -> str:
    return str(raw or "").strip().lower()


def _parse_env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    try:
        text = path.read_text(encoding="utf-8")
    except Exception:
        return values
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            values[key] = value
    return values


def _guard_row(
    name: str,
    ok: bool,
    *,
    severity: str,
    expected: str,
    actual: Any,
    evidence: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "name": name,
        "ok": bool(ok),
        "severity": severity,
        "status": "ready" if ok else ("blocked" if severity in HARD_SEVERITIES else "degraded"),
        "expected": expected,
        "actual": actual,
        "evidence": ordered_unique(evidence or []),
    }


def _paper_is_blocked(paper: dict[str, Any]) -> bool:
    if not paper:
        return False
    return (
        _lower(paper.get("stage")) == "blocked"
        or not _bool(paper.get("ok", False))
        or bool(_as_list(paper.get("blockers")))
    )


def _artifact_snapshot(name: str, path: Path, payload: dict[str, Any], max_age_minutes: float) -> dict[str, Any]:
    age = payload_age_minutes(payload, path)
    present = bool(payload)
    stale = bool(not present or age is None or float(age) > float(max_age_minutes))
    return {
        "name": name,
        "path": str(path),
        "present": present,
        "age_minutes": round(float(age), 3) if age is not None else None,
        "max_age_minutes": round(float(max_age_minutes), 3),
        "stale": stale,
    }


def _paper_lane_eligible(runtime: dict[str, Any], paper: dict[str, Any]) -> bool:
    paper_policy = _as_dict(runtime.get("paper_execution_policy"))
    live_policy = _as_dict(_as_dict(runtime.get("runtime_saturation_governor_v2")).get("paper_live_data_policy"))
    blockers = _as_list(paper.get("blockers")) if paper else []
    paper_armed_clean = bool(paper and _bool(paper.get("armed")) and _bool(paper.get("ok")) and not blockers)
    runtime_armed_clean = bool(
        _bool(paper_policy.get("armed"))
        and _bool(paper_policy.get("ok"))
        and _lower(paper_policy.get("stage")) == "armed"
        and not _as_list(paper_policy.get("blockers"))
    )
    return bool(
        paper_armed_clean
        or runtime_armed_clean
        or _bool(paper_policy.get("paper_execution_allowed"))
        or _bool(live_policy.get("paper_execution_allowed"))
    )


def _paper_and_gate_blockers(paper: dict[str, Any]) -> list[str]:
    paper_blockers = [str(item) for item in _as_list(paper.get("blockers"))]
    paper_gate = _as_dict(_as_dict(paper.get("gates")).get("runtime")) if paper else {}
    paper_gate_blockers = [str(item) for item in _as_list(paper_gate.get("blockers"))]
    return ordered_unique(paper_blockers + paper_gate_blockers)


def _paper_execution_open(runtime: dict[str, Any], env_values: dict[str, str]) -> bool:
    paper_policy = _as_dict(runtime.get("paper_execution_policy"))
    live_policy = _as_dict(_as_dict(runtime.get("runtime_saturation_governor_v2")).get("paper_live_data_policy"))
    return bool(
        _bool(paper_policy.get("paper_execution_allowed", False))
        and not _bool(paper_policy.get("pause_paper_execution", False))
        and _bool(live_policy.get("paper_execution_allowed", False))
        and not _bool(live_policy.get("paper_execution_consumer_paused", False))
        and not _paper_env_pause_keys(env_values)
    )


def _expansion_only_blockers(blockers: list[str]) -> list[str]:
    return [item for item in blockers if item in PAPER_EXPANSION_ONLY_BLOCKERS]


def _fail_closed_blockers(blockers: list[str]) -> list[str]:
    return [item for item in blockers if item in HARD_PAPER_BLOCKERS and item not in PAPER_EXPANSION_ONLY_BLOCKERS]


def _capacity_limited_paper_gate_safe(paper_gate: dict[str, Any]) -> bool:
    if not paper_gate:
        return False
    return bool(
        _bool(paper_gate.get("capacity_limited_armed", False))
        and _bool(paper_gate.get("runtime_capacity_ready", False))
        and _bool(paper_gate.get("paper_execution_clean", False))
        and _bool(paper_gate.get("live_execution_locked", False))
        and _bool(paper_gate.get("active_bot_capacity_ready", True))
        and _bool(paper_gate.get("paper_roster_ready", True))
        and not _as_list(paper_gate.get("blockers"))
    )


def _source_ready(payload: dict[str, Any], *, ready_keys: tuple[str, ...] = ()) -> bool:
    if not payload:
        return False
    if _bool(payload.get("ok", False)):
        return True
    if _bool(payload.get("ready", False)):
        return True
    if any(_bool(payload.get(key, False)) for key in ready_keys):
        return True
    return _lower(payload.get("overall_status") or payload.get("status")) in READY_LIKE_STATUSES


def _auth_stack_ready(auth_lease: dict[str, Any], schwab_auth: dict[str, Any]) -> bool:
    lease_budget = _as_dict(auth_lease.get("lease_budget"))
    lease_expires = _safe_float(auth_lease.get("expires_in_seconds"), _safe_float(lease_budget.get("expires_in_seconds"), 0.0))
    lease_ready = bool(
        _source_ready(auth_lease)
        and _lower(auth_lease.get("lease_state")) in {"", "healthy", "ready", "ok"}
        and lease_expires >= _safe_float(lease_budget.get("critical_lease_seconds"), 600.0)
    )
    schwab_ready = bool(
        _source_ready(schwab_auth, ready_keys=("token_ready",))
        and not _bool(schwab_auth.get("refresh_needed", False))
    )
    return bool(lease_ready and schwab_ready)


def _broker_ready(broker: dict[str, Any]) -> bool:
    return bool(
        _source_ready(broker, ready_keys=("ready_for_open", "broker_ready", "market_data_ready"))
        and broker.get("auth_ok", True) is not False
        and broker.get("network_ok", True) is not False
    )


def _paper_soak_auth_ready(auth_lease: dict[str, Any], schwab_auth: dict[str, Any], broker: dict[str, Any], env_values: dict[str, str]) -> bool:
    live_authority = _live_execution_authority_enabled(env_values)
    if _bool(live_authority.get("enabled", False)):
        return False
    if "MARKET_DATA_ONLY" in env_values and _falsey(env_values.get("MARKET_DATA_ONLY")):
        return False

    broker_state = _as_dict(auth_lease.get("broker_state"))
    lease_budget = _as_dict(auth_lease.get("lease_budget"))
    token = _as_dict(schwab_auth.get("token"))
    broker_preflight = _as_dict(broker.get("preflight_checks"))
    lease_expires = max(
        _safe_float(auth_lease.get("expires_in_seconds"), 0.0),
        _safe_float(lease_budget.get("expires_in_seconds"), 0.0),
        _safe_float(token.get("expires_in_seconds"), 0.0),
        _safe_float(broker.get("token_expires_in_seconds"), 0.0),
    )
    ready_floor = max(
        _safe_float(schwab_auth.get("min_ready_expires_seconds"), 0.0),
        _safe_float(token.get("min_ready_expires_seconds"), 0.0),
        _safe_float(_as_dict(schwab_auth.get("regression_contract")).get("schwab_token_ready_floor_seconds"), 0.0),
        900.0,
    )
    critical_floor = max(_safe_float(lease_budget.get("critical_lease_seconds"), 0.0), 600.0)
    token_ready = bool(
        _bool(schwab_auth.get("token_ready", False))
        or _bool(token.get("ready", False))
        or _bool(broker_preflight.get("token_ready_for_open", False))
        or _bool(broker.get("ready_for_open", False))
    )
    readiness_refresh_needed = bool(
        _bool(schwab_auth.get("readiness_refresh_needed", False))
        or _bool(token.get("readiness_refresh_needed", False))
        or _bool(broker_preflight.get("readiness_refresh_needed_after", False))
        or lease_expires < ready_floor
    )
    network_ok = bool(
        broker.get("network_ok", True) is not False
        and broker_state.get("network_ok", True) is not False
    )
    broker_operable = bool(_bool(broker.get("ready_for_open", False)) or _bool(broker_state.get("broker_operable", False)))
    configured_for_refresh = bool(
        broker_state.get("configured_for_refresh", True) is not False
        and (
            token_ready
            or bool(token)
            or _bool(broker_preflight.get("token_exists", False))
        )
    )
    return bool(
        token_ready
        and not readiness_refresh_needed
        and network_ok
        and broker_operable
        and configured_for_refresh
        and lease_expires >= critical_floor
    )


def _session_ready(session: dict[str, Any]) -> bool:
    checks = [row for row in _as_list(session.get("checks")) if isinstance(row, dict)]
    failed_checks = [str(row.get("name") or "") for row in checks if row.get("ok") is False]
    return bool(_source_ready(session, ready_keys=("ready",)) and not failed_checks)


def _paper_trade_lock_active(project_root: Path, env_values: dict[str, str]) -> bool:
    if _bool(env_values.get("PAPER_TRADE_LOCK", False)):
        return True
    raw_path = str(env_values.get("PAPER_TRADE_LOCK_PATH") or "").strip()
    lock_path = Path(raw_path) if raw_path else project_root / "governance" / "health" / "PAPER_TRADE_LOCK.flag"
    if not lock_path.is_absolute():
        lock_path = project_root / lock_path
    return lock_path.exists()


def _live_execution_authority_enabled(env_values: dict[str, str]) -> dict[str, Any]:
    live_flags = {
        key: env_values.get(key)
        for key in LIVE_AUTHORITY_KEYS
        if key in env_values and _bool(env_values.get(key, False))
    }
    order_env_present = "ALLOW_ORDER_EXECUTION" in env_values or "MARKET_DATA_ONLY" in env_values
    order_authority = bool(
        order_env_present
        and _bool(env_values.get("ALLOW_ORDER_EXECUTION", False))
        and _falsey(env_values.get("MARKET_DATA_ONLY", "1"))
    )
    return {
        "enabled": bool(live_flags or order_authority),
        "live_flags": live_flags,
        "order_authority": order_authority,
        "allow_order_execution": env_values.get("ALLOW_ORDER_EXECUTION", "<missing>"),
        "market_data_only": env_values.get("MARKET_DATA_ONLY", "<missing>"),
    }


def _all_sleeves_runtime_state(process: dict[str, Any]) -> dict[str, Any]:
    rows = _as_list(process.get("status"))
    row = next((item for item in rows if isinstance(item, dict) and str(item.get("name") or "") == "all_sleeves"), {})
    launcher = _as_dict(row.get("launcher_artifact_health"))
    child_fanout = _as_dict(row.get("child_fanout"))
    child_fanout_ok = row.get("child_fanout_ok")
    if child_fanout_ok is None:
        child_fanout_ok = child_fanout.get("ok")
    launcher_live = row.get("launcher_live")
    if launcher_live is None:
        launcher_live = launcher.get("launcher_live")
    heartbeat_ok = row.get("heartbeat_ok")
    process_live = row.get("process_live")
    running = _safe_float(row.get("running"), 0.0)
    ok = bool(
        row
        and running > 0
        and _bool(process_live)
        and _bool(heartbeat_ok)
        and _bool(launcher_live)
        and _bool(child_fanout_ok)
    )
    return {
        "present": bool(row),
        "ok": ok,
        "running": int(running),
        "process_live": _bool(process_live),
        "heartbeat_ok": _bool(heartbeat_ok),
        "launcher_live": _bool(launcher_live),
        "child_fanout_ok": _bool(child_fanout_ok),
        "child_process_count": _safe_float(child_fanout.get("child_process_count"), _safe_float(row.get("child_count"), 0.0)),
        "launcher_phase": str(launcher.get("phase") or row.get("launcher_phase") or ""),
        "launcher_running_job_count": _safe_float(launcher.get("running_job_count"), _safe_float(row.get("launcher_running"), 0.0)),
        "launcher_expected_job_count": _safe_float(launcher.get("expected_job_count"), _safe_float(row.get("launcher_expected"), 0.0)),
    }


def _runtime_ready_contract_guard(runtime: dict[str, Any]) -> dict[str, Any]:
    if not runtime:
        return _guard_row(
            "runtime_artifact_present",
            False,
            severity="medium",
            expected="runtime_throttle_control_latest.json is available",
            actual="missing",
            evidence=["cannot evaluate runtime ready/advisory contract without runtime artifact"],
        )

    status = _lower(runtime.get("overall_status"))
    profile = _lower(runtime.get("throttle_profile"))
    compute = _lower(runtime.get("compute_pressure_level"))
    memory = _lower(runtime.get("memory_pressure_level"))
    soft_cap = _as_dict(runtime.get("soft_cap_advisory_reclassification"))
    soft_active = _bool(soft_cap.get("active", False))
    to_status = _lower(soft_cap.get("to_status"))
    under_pressure = profile in PRESSURE_PROFILES or compute not in {"", "normal"} or memory not in {"", "normal"}
    already_reclassified_low_pressure_advisory = bool(
        status == "advisory"
        and _bool(runtime.get("ok", False))
        and profile in PRESSURE_PROFILES
        and compute in {"", "normal"}
        and memory in {"", "normal"}
    )

    if status in READY_LIKE_STATUSES and under_pressure:
        ok = (soft_active and (not to_status or to_status == status)) or already_reclassified_low_pressure_advisory
    else:
        ok = True

    return _guard_row(
        "runtime_ready_advisory_reclassification_contract",
        ok,
        severity="high",
        expected="ready/advisory runtime under pressure has an active reclassification whose to_status matches overall_status",
        actual={
            "overall_status": status,
            "throttle_profile": profile,
            "compute_pressure_level": compute,
            "memory_pressure_level": memory,
            "soft_cap_active": soft_active,
            "soft_cap_to_status": to_status,
            "already_reclassified_low_pressure_advisory": already_reclassified_low_pressure_advisory,
        },
        evidence=[
            f"reason={soft_cap.get('reason', '')}",
            "runtime_artifact_already_advisory_ok_with_normal_pressure"
            if already_reclassified_low_pressure_advisory
            else "",
            "normal runtime pressure does not require soft-cap reclassification" if not under_pressure else "",
        ],
    )


def _runtime_guarded_ready_lane_guard(runtime: dict[str, Any]) -> dict[str, Any]:
    if not runtime:
        return _guard_row(
            "runtime_guarded_ready_lane_contract",
            False,
            severity="medium",
            expected="runtime artifact is available before guarded-ready lane checks",
            actual="missing",
        )

    status = _lower(runtime.get("overall_status"))
    soft_cap = _as_dict(runtime.get("soft_cap_advisory_reclassification"))
    reason = str(soft_cap.get("reason") or "")
    to_status = _lower(soft_cap.get("to_status"))
    guarded_ready = status == "ready" and _bool(soft_cap.get("active", False)) and (
        to_status == "ready" or "guarded_runtime_ready" in reason
    )
    if not guarded_ready:
        return _guard_row(
            "runtime_guarded_ready_lane_contract",
            True,
            severity="high",
            expected="non-guarded-ready runtime states skip guarded-ready hot-lane checks",
            actual={"overall_status": status, "reason": reason},
        )

    measurements = _as_dict(soft_cap.get("measurements"))
    thresholds = _as_dict(soft_cap.get("thresholds"))
    storage_writer_ready = bool(
        _bool(measurements.get("storage_writer_cooling_guarded_ready", False))
        or reason == "single_bounded_storage_writer_after_green_backpressure_is_guarded_runtime_ready"
    )
    full_force_paper_ready = bool(
        _bool(measurements.get("full_force_paper_ramp_guarded_ready", False))
        or reason
        in {
            "full_force_paper_ramp_writer_pressure_is_guarded_runtime_ready",
            "full_force_paper_ramp_pressure_is_guarded_runtime_ready",
        }
    )
    protected_lane_ready = bool(
        _bool(measurements.get("bounded_protected_lane_guarded_ready", False))
        or reason == "bounded_read_only_protected_lane_after_green_backpressure_is_guarded_runtime_ready"
    )
    hot_flags = [
        "support_jobs_hot",
        "paper_execution_hot",
        "research_training_hot",
        "storage_writer_hot",
        "bot_owned_pressure_dominant",
    ]
    allowed_hot_flags: set[str] = set()
    if storage_writer_ready:
        allowed_hot_flags.update({"storage_writer_hot", "bot_owned_pressure_dominant"})
    if full_force_paper_ready:
        allowed_hot_flags.update({"paper_execution_hot", "storage_writer_hot", "bot_owned_pressure_dominant"})
        if _safe_float(measurements.get("throttle_candidate_support_cpu_percent"), 0.0) <= 80.0:
            allowed_hot_flags.add("support_jobs_hot")
    if protected_lane_ready:
        allowed_hot_flags.add("bot_owned_pressure_dominant")
    hot_lanes = [
        name
        for name in hot_flags
        if _bool(measurements.get(name, False)) and name not in allowed_hot_flags
    ]
    bot_limit = _safe_float(thresholds.get("max_guarded_ready_bot_owned_cpu_percent"), 20.0)
    protected_limit = _safe_float(thresholds.get("max_guarded_ready_protected_cpu_percent"), 20.0)
    operator_limit = _safe_float(thresholds.get("max_guarded_ready_operator_cpu_percent"), 30.0)
    if storage_writer_ready:
        bot_limit = max(bot_limit, 150.0)
    if full_force_paper_ready:
        bot_limit = max(
            bot_limit,
            _safe_float(thresholds.get("max_guarded_ready_full_force_bot_owned_cpu_percent"), 340.0),
        )
        operator_limit = max(
            operator_limit,
            _safe_float(thresholds.get("max_guarded_ready_full_force_operator_cpu_percent"), 45.0),
        )
    if protected_lane_ready:
        bot_limit = max(
            bot_limit,
            _safe_float(thresholds.get("max_guarded_ready_bot_owned_with_protected_lane_cpu_percent"), 95.0),
        )
        protected_limit = max(
            protected_limit,
            _safe_float(thresholds.get("max_guarded_ready_protected_lane_cpu_percent"), 75.0),
        )
    bot_owned_raw = _safe_float(measurements.get("bot_owned_cpu_percent"), 0.0)
    bot_owned = _safe_float(measurements.get("bot_owned_non_operator_cpu_percent"), bot_owned_raw)
    protected = _safe_float(measurements.get("protected_live_or_macro_cpu_percent"), 0.0)
    operator = _safe_float(measurements.get("operator_observability_cpu_percent"), 0.0)
    memory = _lower(measurements.get("memory_pressure_level") or runtime.get("memory_pressure_level"))
    storage_ready = _bool(measurements.get("storage_ready_for_runtime_advisory", True))
    ok = (
        bool(measurements)
        and _bool(measurements.get("runtime_ready_guarded", False))
        and not hot_lanes
        and memory == "normal"
        and storage_ready
        and bot_owned < bot_limit
        and protected < protected_limit
        and operator <= operator_limit
    )
    return _guard_row(
        "runtime_guarded_ready_lane_contract",
        ok,
        severity="high",
        expected="guarded-ready runtime has normal memory, low bot-owned/protected/operator pressure, and no hot bot-owned lanes",
        actual={
            "runtime_ready_guarded": _bool(measurements.get("runtime_ready_guarded", False)),
            "memory_pressure_level": memory,
            "hot_lanes": hot_lanes,
            "storage_writer_cooling_guarded_ready": storage_writer_ready,
            "bounded_protected_lane_guarded_ready": protected_lane_ready,
            "storage_ready_for_runtime_advisory": storage_ready,
            "bot_owned_cpu_percent": bot_owned_raw,
            "bot_owned_non_operator_cpu_percent": bot_owned,
            "bot_owned_limit": bot_limit,
            "protected_live_or_macro_cpu_percent": protected,
            "protected_limit": protected_limit,
            "operator_observability_cpu_percent": operator,
            "operator_limit": operator_limit,
        },
        evidence=[f"reason={reason}"],
    )


def _paper_runtime_capacity_blocker_guard(paper: dict[str, Any]) -> dict[str, Any]:
    if not paper:
        return _guard_row(
            "paper_runtime_capacity_blocker_contract",
            False,
            severity="medium",
            expected="paper_400_ramp_latest.json is available",
            actual="missing",
            evidence=["cannot evaluate paper ramp runtime blocker contract without paper ramp artifact"],
        )
    blockers = [str(item) for item in _as_list(paper.get("blockers"))]
    runtime_gate = _as_dict(_as_dict(paper.get("gates")).get("runtime"))
    runtime_gate_blockers = [str(item) for item in _as_list(runtime_gate.get("blockers"))]
    runtime_pressure_ready = _bool(runtime_gate.get("runtime_pressure_ready"))
    runtime_capacity_ready = _bool(runtime_gate.get("runtime_capacity_ready"))
    capacity_blocker_present = CAPACITY_BLOCKER in set(blockers + runtime_gate_blockers)
    ok = True
    if runtime_pressure_ready and runtime_capacity_ready:
        ok = not capacity_blocker_present
    return _guard_row(
        "paper_runtime_capacity_blocker_contract",
        ok,
        severity="high",
        expected="runtime_capacity_not_ready_for_400_paper is absent when runtime pressure and capacity gates are ready",
        actual={
            "runtime_pressure_ready": runtime_pressure_ready,
            "runtime_capacity_ready": runtime_capacity_ready,
            "paper_blockers": blockers,
            "runtime_gate_blockers": runtime_gate_blockers,
        },
        evidence=[
            f"stage={paper.get('stage', '')}",
            f"runtime_gate_status={runtime_gate.get('status', '')}",
        ],
    )


def _paper_arm_blocker_guard(paper: dict[str, Any]) -> dict[str, Any]:
    if not paper:
        return _guard_row(
            "paper_armed_blocker_contract",
            False,
            severity="medium",
            expected="paper ramp artifact is available before armed/blocker checks",
            actual="missing",
        )
    blockers = [str(item) for item in _as_list(paper.get("blockers"))]
    armed = _bool(paper.get("armed", False))
    paper_ok = _bool(paper.get("ok", False))
    stage = _lower(paper.get("stage"))
    ok = (not armed) or (paper_ok and stage == "armed" and not blockers)
    return _guard_row(
        "paper_armed_blocker_contract",
        ok,
        severity="critical",
        expected="paper ramp can only be armed when ok=true, stage=armed, and blockers are empty",
        actual={"armed": armed, "ok": paper_ok, "stage": stage, "blockers": blockers},
        evidence=["blocked paper ramps must keep live/paper widening disabled"],
    )


def _paper_execution_pause_guard(runtime: dict[str, Any], paper: dict[str, Any], env_values: dict[str, str]) -> dict[str, Any]:
    paper_blocked = _paper_is_blocked(paper)
    if not paper_blocked:
        return _guard_row(
            "blocked_paper_execution_pause_contract",
            True,
            severity="high",
            expected="paper execution pause is only required while paper ramp is blocked",
            actual={"paper_blocked": False, "stage": paper.get("stage") if paper else ""},
        )
    if not runtime:
        return _guard_row(
            "blocked_paper_execution_pause_contract",
            False,
            severity="high",
            expected="runtime artifact confirms paper execution consumers are paused while paper ramp is blocked",
            actual="runtime_artifact_missing",
        )

    paper_policy = _as_dict(runtime.get("paper_execution_policy"))
    live_policy = _as_dict(_as_dict(runtime.get("runtime_saturation_governor_v2")).get("paper_live_data_policy"))
    all_blockers = _paper_and_gate_blockers(paper)
    expansion_blockers = _expansion_only_blockers(all_blockers)
    fail_closed_blockers = _fail_closed_blockers(all_blockers)
    existing_paper_open = _paper_execution_open(runtime, env_values)
    if expansion_blockers and not fail_closed_blockers and existing_paper_open:
        return _guard_row(
            "blocked_paper_execution_pause_contract",
            True,
            severity="high",
            expected="400-paper expansion blockers may pause widening while existing paper execution remains open",
            actual={
                "paper_blocked": paper_blocked,
                "paper_policy_pause_paper_execution": _bool(paper_policy.get("pause_paper_execution")),
                "live_policy_consumer_paused": _bool(live_policy.get("paper_execution_consumer_paused")),
                "expansion_only_blockers": expansion_blockers,
                "fail_closed_blockers": fail_closed_blockers,
                "existing_paper_execution_open": True,
            },
            evidence=[
                f"paper_stage={paper.get('stage', '')}",
                f"paper_blockers={','.join(all_blockers)}",
                "existing_paper_soak_must_fail_open_when_only_expansion_is_paused",
            ],
        )
    policy_paused = _bool(paper_policy.get("pause_paper_execution")) and _bool(live_policy.get("paper_execution_consumer_paused"))
    env_mismatches = {
        key: env_values.get(key, "<missing>")
        for key, expected in PAPER_PAUSE_OVERRIDE_KEYS.items()
        if env_values.get(key) != expected
    }
    ok = policy_paused and not env_mismatches
    return _guard_row(
        "blocked_paper_execution_pause_contract",
        ok,
        severity="high",
        expected="blocked paper ramp sets pause_paper_execution=true, consumer_paused=true, and queue/inline override gates off",
        actual={
            "paper_blocked": paper_blocked,
            "paper_policy_pause_paper_execution": _bool(paper_policy.get("pause_paper_execution")),
            "live_policy_consumer_paused": _bool(live_policy.get("paper_execution_consumer_paused")),
            "env_mismatches": env_mismatches,
        },
        evidence=[
            f"paper_stage={paper.get('stage', '')}",
            f"paper_blockers={','.join(str(item) for item in _as_list(paper.get('blockers')))}",
        ],
    )


def _support_override_guard(runtime: dict[str, Any], env_values: dict[str, str], override_path: Path) -> dict[str, Any]:
    if not runtime:
        return _guard_row(
            "runtime_override_support_spawn_contract",
            False,
            severity="medium",
            expected="runtime artifact is available before support override checks",
            actual="missing",
        )
    profile = _lower(runtime.get("throttle_profile"))
    governor = _as_dict(runtime.get("runtime_saturation_governor_v2"))
    needs_overrides = profile in PRESSURE_PROFILES or _bool(governor.get("active", False))
    if not needs_overrides:
        return _guard_row(
            "runtime_override_support_spawn_contract",
            True,
            severity="medium",
            expected="normal runtime profile does not require sustain/protect support overrides",
            actual={"throttle_profile": profile, "override_path": str(override_path)},
        )

    resolved_keys: dict[str, str] = {}
    for key in SUPPORT_OVERRIDE_KEYS:
        aliases = SUPPORT_OVERRIDE_ALIASES.get(key, [key])
        present = [alias for alias in aliases if alias in env_values]
        matched = ""
        if key in {"YTDLP_SUPPORT_NICE", "MACRO_YTDLP_SUPPORT_NICE"}:
            matched = next((alias for alias in present if _safe_float(env_values.get(alias), -1.0) >= 10.0), "")
        if not matched:
            matched = present[0] if present else ""
        if matched:
            resolved_keys[key] = matched
    missing = [key for key in SUPPORT_OVERRIDE_KEYS if key not in resolved_keys]
    nice_values = {
        key: _safe_float(env_values.get(alias), -1.0)
        for key, alias in resolved_keys.items()
        if key in {"YTDLP_SUPPORT_NICE", "MACRO_YTDLP_SUPPORT_NICE", "TRAINING_RUNTIME_PAUSED_FOR_HOST_HEADROOM"}
        and "NICE" in alias
    }
    weak_nice = [key for key, value in nice_values.items() if value < 10.0]
    ok = not missing and not weak_nice
    return _guard_row(
        "runtime_override_support_spawn_contract",
        ok,
        severity="medium",
        expected="sustain/protect runtime override carries support niceness and shadow-loop pause keys",
        actual={
            "throttle_profile": profile,
            "override_path": str(override_path),
            "missing_keys": missing,
            "resolved_keys": resolved_keys,
            "weak_nice_keys": weak_nice,
        },
        evidence=["support yt-dlp and training loops must inherit runtime backoff without blocking core writers"],
    )


def _hot_artifact_freshness_guard(artifact_snapshots: list[dict[str, Any]]) -> dict[str, Any]:
    stale = [row for row in artifact_snapshots if bool(row.get("stale", False))]
    return _guard_row(
        "soak_hot_artifact_freshness_contract",
        not stale,
        severity="medium",
        expected="hot paper-runtime artifacts stay fresh enough that stale surfaces cannot silently pause paper trading",
        actual={
            "stale_artifact_count": len(stale),
            "stale_artifacts": stale,
            "artifacts": artifact_snapshots,
        },
        evidence=[
            "degraded freshness should trigger refresh/self-heal, not silently disable eligible paper execution",
        ],
    )


def _paper_eligible_lane_open_guard(
    runtime: dict[str, Any],
    paper: dict[str, Any],
    process: dict[str, Any],
    artifact_snapshots: list[dict[str, Any]],
) -> dict[str, Any]:
    eligible = _paper_lane_eligible(runtime, paper)
    if not eligible:
        return _guard_row(
            "soak_paper_eligible_lane_open_contract",
            True,
            severity="critical",
            expected="paper lane open check only applies after the ramp or runtime policy says paper execution is eligible",
            actual={"paper_lane_eligible": False, "paper_stage": paper.get("stage") if paper else "missing"},
        )

    paper_policy = _as_dict(runtime.get("paper_execution_policy"))
    live_policy = _as_dict(_as_dict(runtime.get("runtime_saturation_governor_v2")).get("paper_live_data_policy"))
    all_sleeves = _all_sleeves_runtime_state(process)
    stale = [row for row in artifact_snapshots if bool(row.get("stale", False))]
    lane_blockers: list[str] = []
    if not _bool(paper_policy.get("paper_execution_allowed")):
        lane_blockers.append("runtime_policy_disallows_paper_execution")
    if _bool(paper_policy.get("pause_paper_execution")):
        lane_blockers.append("runtime_policy_pauses_paper_execution")
    if not _bool(live_policy.get("paper_execution_allowed")):
        lane_blockers.append("saturation_governor_disallows_paper_execution")
    if _bool(live_policy.get("paper_execution_consumer_paused")):
        lane_blockers.append("paper_execution_consumer_paused")
    if not bool(all_sleeves.get("ok", False)):
        lane_blockers.append("all_sleeves_runtime_not_effectively_live")
    if stale and lane_blockers:
        lane_blockers.append("stale_artifact_blocking_eligible_paper_lane")

    return _guard_row(
        "soak_paper_eligible_lane_open_contract",
        not lane_blockers,
        severity="critical",
        expected="when paper is armed/eligible, paper execution remains allowed, consumers stay unpaused, and all_sleeves fanout stays live",
        actual={
            "paper_lane_eligible": eligible,
            "lane_blockers": ordered_unique(lane_blockers),
            "paper_policy": paper_policy,
            "live_policy": live_policy,
            "all_sleeves": all_sleeves,
            "stale_artifacts": stale,
        },
        evidence=[
            "30_day_soak_invariant=eligible_paper_lanes_must_not_stop_for_stale_or_missing_hot_artifacts",
        ],
    )


def _paper_env_pause_keys(env_values: dict[str, str]) -> dict[str, str]:
    return {
        key: env_values.get(key, "<missing>")
        for key, paused in {
            "PAPER_EXECUTION_RUNTIME_PAUSED_FOR_PRESSURE": {"1", "true", "yes", "on"},
            "PAPER_EXECUTION_QUEUE_CONSUMER_ENABLED": {"0", "false", "off"},
            "INLINE_PAPER_EXECUTION_ENABLED": {"0", "false", "off"},
        }.items()
        if str(env_values.get(key, "")).strip().lower() in paused
    }


def _production_grade_authority_guard(
    project_root: Path,
    runtime: dict[str, Any],
    paper: dict[str, Any],
    process: dict[str, Any],
    runtime_profitability: dict[str, Any],
    auth_lease: dict[str, Any],
    schwab_auth: dict[str, Any],
    broker: dict[str, Any],
    session: dict[str, Any],
    env_values: dict[str, str],
    artifact_snapshots: list[dict[str, Any]],
) -> dict[str, Any]:
    eligible = _paper_lane_eligible(runtime, paper)
    paper_policy = _as_dict(runtime.get("paper_execution_policy"))
    live_policy = _as_dict(_as_dict(runtime.get("runtime_saturation_governor_v2")).get("paper_live_data_policy"))
    all_sleeves = _all_sleeves_runtime_state(process)
    all_paper_blockers = _paper_and_gate_blockers(paper)
    hard_blockers = ordered_unique([item for item in all_paper_blockers if item in HARD_PAPER_BLOCKERS])
    expansion_blockers = _expansion_only_blockers(all_paper_blockers)
    fail_closed_blockers = _fail_closed_blockers(all_paper_blockers)
    env_pause_keys = _paper_env_pause_keys(env_values)
    stale_artifacts = [row for row in artifact_snapshots if bool(row.get("stale", False))]
    paper_lock_active = _paper_trade_lock_active(project_root, env_values)
    live_authority = _live_execution_authority_enabled(env_values)
    strict_auth_ready = _auth_stack_ready(auth_lease, schwab_auth)
    strict_broker_ready = _broker_ready(broker)
    paper_auth_ready = bool(strict_auth_ready or _paper_soak_auth_ready(auth_lease, schwab_auth, broker, env_values))
    paper_broker_ready = bool(strict_broker_ready or (_bool(broker.get("ready_for_open", False)) and broker.get("network_ok", True) is not False))
    session_ready = _session_ready(session)
    raw_grade = str(runtime_profitability.get("raw_profitability_grade") or "").strip().upper()
    controlled_grade = str(runtime_profitability.get("controlled_profitability_grade") or "").strip().upper()
    raw_recovery = _as_dict(runtime_profitability.get("raw_profitability_a_recovery_contract"))
    raw_gap = _safe_float(_as_dict(raw_recovery.get("gap_to_raw_a")).get("net_pnl_gap"), 0.0)
    runtime_enforcement = _as_dict(raw_recovery.get("runtime_enforcement"))
    raw_improvement = _as_dict(runtime_profitability.get("raw_profitability_improvement_contract"))
    raw_improvement_enforcement = _as_dict(raw_improvement.get("runtime_enforcement"))
    raw_improvement_required = bool(
        runtime_profitability
        and (
            _bool(raw_recovery.get("active", False))
            or raw_gap > 0.0
            or (raw_grade and raw_grade not in {"A", "A+"})
        )
    )
    raw_improvement_ready = bool(
        not raw_improvement_required
        or (
            _bool(raw_improvement.get("control_ready", False))
            and _bool(raw_improvement.get("raw_grade_remains_evidence_based", False))
            and _bool(raw_improvement_enforcement.get("block_new_entries_on_weak_profiles", False))
            and _bool(raw_improvement_enforcement.get("keep_sells_and_reduce_only_paths_open", False))
            and _bool(raw_improvement_enforcement.get("raise_clean_profile_buy_gate_while_raw_below_a", False))
            and _bool(raw_improvement_enforcement.get("require_position_telemetry_on_paper_fills", False))
            and _bool(raw_improvement_enforcement.get("feed_loss_causes_to_training", False))
            and _bool(raw_improvement_enforcement.get("require_three_profitable_refreshes_before_reentry", False))
            and _bool(raw_improvement_enforcement.get("track_raw_gap_burn_down", False))
        )
    )
    raw_grade_cosmetic = bool(
        raw_grade in {"A", "A+"}
        and _bool(raw_recovery.get("active", False))
        and raw_gap > 0.0
    )
    controlled_profitability_enforced = bool(
        controlled_grade in {"A", "A+"}
        or (
            _bool(raw_recovery.get("active", False))
            and _bool(runtime_enforcement.get("block_new_entries_on_weak_profiles", False))
            and _bool(runtime_enforcement.get("keep_sells_and_reduce_only_paths_open", False))
        )
    )
    controlled_profitability_enforced = bool(controlled_profitability_enforced and raw_improvement_ready)
    paper_open = _paper_execution_open(runtime, env_values)
    hard_blocker_fail_closed = bool(
        not fail_closed_blockers
        or (
            _paper_is_blocked(paper)
            and (
                _bool(paper_policy.get("pause_paper_execution", False))
                or _bool(live_policy.get("paper_execution_consumer_paused", False))
                or bool(env_pause_keys)
            )
        )
    )
    paper_fail_open_when_safe = bool(
        not eligible
        or fail_closed_blockers
        or (
            paper_open
            and bool(all_sleeves.get("ok", False))
            and paper_auth_ready
            and paper_broker_ready
            and session_ready
            and paper_lock_active
        )
    )
    blockers: list[str] = []
    if _bool(live_authority.get("enabled", False)):
        blockers.append("live_execution_authority_enabled")
    if not paper_lock_active:
        blockers.append("paper_trade_lock_not_active")
    if not hard_blocker_fail_closed:
        blockers.append("hard_safety_blocker_without_paper_fail_closed")
    if eligible and not hard_blockers and not paper_fail_open_when_safe:
        blockers.append("eligible_paper_not_open_under_soft_degradation")
    if eligible and not paper_auth_ready:
        blockers.append("auth_stack_not_ready")
    if eligible and not paper_broker_ready:
        blockers.append("broker_not_ready")
    if eligible and not session_ready:
        blockers.append("session_not_ready")
    if stale_artifacts and not paper_open and not hard_blockers:
        blockers.append("stale_artifact_has_control_path_authority")
    if runtime_profitability and not controlled_profitability_enforced:
        blockers.append("profitability_controls_not_enforced")
    if raw_improvement_required and not raw_improvement_ready:
        blockers.append("raw_profitability_improvement_contract_not_ready")
    if raw_grade_cosmetic:
        blockers.append("raw_profitability_grade_cosmetic_upgrade")

    return _guard_row(
        "production_grade_paper_live_authority_contract",
        not blockers,
        severity="critical",
        expected="live execution fails closed, eligible paper fails open under soft degradation, hard blockers fail paper closed, stale artifacts refresh only, and raw profitability remains evidence-based",
        actual={
            "paper_lane_eligible": eligible,
            "blockers": ordered_unique(blockers),
            "hard_paper_blockers": hard_blockers,
            "expansion_only_blockers": expansion_blockers,
            "fail_closed_blockers": fail_closed_blockers,
            "paper_open": paper_open,
            "hard_blocker_fail_closed": hard_blocker_fail_closed,
            "paper_fail_open_when_safe": paper_fail_open_when_safe,
            "paper_lock_active": paper_lock_active,
            "live_authority": live_authority,
            "auth_ready": paper_auth_ready,
            "broker_ready": paper_broker_ready,
            "strict_auth_ready": strict_auth_ready,
            "strict_broker_ready": strict_broker_ready,
            "paper_soak_auth_grace": bool(paper_auth_ready and not strict_auth_ready),
            "session_ready": session_ready,
            "all_sleeves": all_sleeves,
            "paper_policy": paper_policy,
            "live_policy": live_policy,
            "env_pause_keys": env_pause_keys,
            "stale_artifacts": stale_artifacts,
            "raw_profitability_grade": raw_grade,
            "controlled_profitability_grade": controlled_grade,
            "raw_profitability_a_recovery_active": _bool(raw_recovery.get("active", False)),
            "raw_profitability_gap_to_a_net_pnl": raw_gap,
            "raw_profitability_improvement_required": raw_improvement_required,
            "raw_profitability_improvement_ready": raw_improvement_ready,
            "raw_profitability_improvement_contract": {
                "active": _bool(raw_improvement.get("active", False)),
                "control_ready": _bool(raw_improvement.get("control_ready", False)),
                "position_telemetry_evidence_gap_active": _bool(
                    _as_dict(raw_improvement.get("position_telemetry_contract")).get("evidence_gap_active", False)
                ),
                "burn_down_active": _bool(_as_dict(raw_improvement.get("burn_down_contract")).get("active", False)),
            },
            "controlled_profitability_enforced": controlled_profitability_enforced,
        },
        evidence=[
            "production_grade_rule=paper_only_lanes_fail_open_when_safe_live_money_fails_closed",
            "incident_replay_cases=stale_artifact_pause,operator_observability_pressure,runtime_capacity_blocker,storage_backpressure_hard_block,raw_profitability_cosmetic_upgrade",
        ],
    )


def _soak_30_day_continuity_guard(
    runtime: dict[str, Any],
    paper: dict[str, Any],
    process: dict[str, Any],
    runtime_profitability: dict[str, Any],
    auth_lease: dict[str, Any],
    schwab_auth: dict[str, Any],
    broker: dict[str, Any],
    session: dict[str, Any],
    env_values: dict[str, str],
    artifact_snapshots: list[dict[str, Any]],
) -> dict[str, Any]:
    eligible = _paper_lane_eligible(runtime, paper)
    if not eligible:
        return _guard_row(
            "soak_30_day_continuity_contract",
            True,
            severity="critical",
            expected="continuity contract applies once paper is armed or runtime policy marks paper eligible",
            actual={"paper_lane_eligible": False, "paper_stage": paper.get("stage") if paper else "missing"},
        )

    paper_policy = _as_dict(runtime.get("paper_execution_policy"))
    live_policy = _as_dict(_as_dict(runtime.get("runtime_saturation_governor_v2")).get("paper_live_data_policy"))
    paper_gate = _as_dict(_as_dict(paper.get("gates")).get("runtime")) if paper else {}
    all_paper_blockers = _paper_and_gate_blockers(paper)
    expansion_blockers = _expansion_only_blockers(all_paper_blockers)
    fail_closed_blockers = _fail_closed_blockers(all_paper_blockers)
    runtime_capacity = _as_dict(runtime.get("paper_capacity_contract"))
    all_sleeves = _all_sleeves_runtime_state(process)
    stale_artifacts = [row for row in artifact_snapshots if bool(row.get("stale", False))]
    profitability_grade = str(runtime_profitability.get("controlled_profitability_grade") or "").strip().upper()
    raw_recovery = _as_dict(runtime_profitability.get("raw_profitability_a_recovery_contract"))
    runtime_enforcement = _as_dict(raw_recovery.get("runtime_enforcement"))
    raw_grade = str(runtime_profitability.get("raw_profitability_grade") or "").strip().upper()
    raw_gap = _safe_float(_as_dict(raw_recovery.get("gap_to_raw_a")).get("net_pnl_gap"), 0.0)
    raw_improvement = _as_dict(runtime_profitability.get("raw_profitability_improvement_contract"))
    raw_improvement_enforcement = _as_dict(raw_improvement.get("runtime_enforcement"))
    raw_improvement_required = bool(
        runtime_profitability
        and (
            _bool(raw_recovery.get("active", False))
            or raw_gap > 0.0
            or (raw_grade and raw_grade not in {"A", "A+"})
        )
    )
    raw_improvement_ready = bool(
        not raw_improvement_required
        or (
            _bool(raw_improvement.get("control_ready", False))
            and _bool(raw_improvement_enforcement.get("block_new_entries_on_weak_profiles", False))
            and _bool(raw_improvement_enforcement.get("keep_sells_and_reduce_only_paths_open", False))
            and _bool(raw_improvement_enforcement.get("raise_clean_profile_buy_gate_while_raw_below_a", False))
            and _bool(raw_improvement_enforcement.get("require_position_telemetry_on_paper_fills", False))
            and _bool(raw_improvement_enforcement.get("feed_loss_causes_to_training", False))
            and _bool(raw_improvement_enforcement.get("require_three_profitable_refreshes_before_reentry", False))
            and _bool(raw_improvement_enforcement.get("track_raw_gap_burn_down", False))
        )
    )
    env_pause_keys = _paper_env_pause_keys(env_values)
    strict_auth_ready = _auth_stack_ready(auth_lease, schwab_auth)
    strict_broker_ready = _broker_ready(broker)
    paper_auth_ready = bool(strict_auth_ready or _paper_soak_auth_ready(auth_lease, schwab_auth, broker, env_values))
    paper_broker_ready = bool(strict_broker_ready or (_bool(broker.get("ready_for_open", False)) and broker.get("network_ok", True) is not False))
    session_ready = _session_ready(session)
    capacity_limited_gate_safe = _capacity_limited_paper_gate_safe(paper_gate)
    runtime_status_ready = bool(_lower(runtime.get("overall_status")) in READY_LIKE_STATUSES or capacity_limited_gate_safe)
    strict_paper_ramp_open = bool(
        paper
        and _bool(paper.get("ok", False))
        and _bool(paper.get("armed", False))
        and _lower(paper.get("stage")) == "armed"
        and not _as_list(paper.get("blockers"))
    )
    existing_paper_execution_open = _paper_execution_open(runtime, env_values)
    expansion_pause_existing_paper_open = bool(
        expansion_blockers
        and not fail_closed_blockers
        and existing_paper_execution_open
    )
    paper_ramp_open = bool(strict_paper_ramp_open or expansion_pause_existing_paper_open)
    runtime_gate_ready = bool(
        not paper_gate
        or (
            _bool(paper_gate.get("runtime_pressure_ready", False))
            and _bool(paper_gate.get("runtime_capacity_ready", False))
            and not _as_list(paper_gate.get("blockers"))
        )
        or capacity_limited_gate_safe
    )
    capacity_ready = bool(
        not runtime_capacity
        or (
            _bool(runtime_capacity.get("ready_for_700_bot_paper", False))
            and not _bool(runtime_capacity.get("pressure_limited", False))
        )
        or capacity_limited_gate_safe
    )
    controlled_profitability_ready = bool(
        (
            profitability_grade in {"A", "A+"}
            or (
                _bool(raw_recovery.get("active", False))
                and _bool(runtime_enforcement.get("block_new_entries_on_weak_profiles", False))
                and _bool(runtime_enforcement.get("keep_sells_and_reduce_only_paths_open", False))
            )
        )
        and raw_improvement_ready
    )
    blockers: list[str] = []
    if not runtime_status_ready:
        blockers.append("runtime_not_ready_or_advisory")
    if not paper_ramp_open:
        blockers.append("paper_ramp_not_armed_clean")
    if not runtime_gate_ready:
        blockers.append("paper_ramp_runtime_gate_not_ready")
    if not capacity_ready:
        blockers.append("runtime_paper_capacity_not_ready")
    if not _bool(paper_policy.get("paper_execution_allowed", False)):
        blockers.append("paper_policy_disallows_execution")
    if _bool(paper_policy.get("pause_paper_execution", False)):
        blockers.append("paper_policy_paused")
    if not _bool(live_policy.get("paper_execution_allowed", False)):
        blockers.append("live_policy_disallows_paper_execution")
    if _bool(live_policy.get("paper_execution_consumer_paused", False)):
        blockers.append("live_policy_paused_paper_consumer")
    if env_pause_keys:
        blockers.append("runtime_override_pauses_paper_consumer")
    if not bool(all_sleeves.get("ok", False)):
        blockers.append("all_sleeves_fanout_not_live")
    if runtime_profitability and not controlled_profitability_ready:
        blockers.append("controlled_profitability_posture_not_ready")
    if not paper_auth_ready:
        blockers.append("auth_stack_not_ready")
    if not paper_broker_ready:
        blockers.append("broker_not_ready")
    if not session_ready:
        blockers.append("session_not_ready")

    return _guard_row(
        "soak_30_day_continuity_contract",
        not blockers,
        severity="critical",
        expected="eligible 30-day soak keeps runtime ready, ramp armed, paper consumers open, artifacts fresh, fanout live, and profitability controls enforced",
        actual={
            "paper_lane_eligible": eligible,
            "blockers": ordered_unique(blockers),
            "runtime_status": runtime.get("overall_status") if runtime else "missing",
            "runtime_status_ready": runtime_status_ready,
            "paper_stage": paper.get("stage") if paper else "missing",
            "paper_ramp_open": paper_ramp_open,
            "strict_paper_ramp_open": strict_paper_ramp_open,
            "expansion_pause_existing_paper_open": expansion_pause_existing_paper_open,
            "expansion_only_blockers": expansion_blockers,
            "fail_closed_blockers": fail_closed_blockers,
            "runtime_gate_ready": runtime_gate_ready,
            "capacity_ready": capacity_ready,
            "capacity_limited_paper_gate_safe": capacity_limited_gate_safe,
            "paper_policy": paper_policy,
            "live_policy": live_policy,
            "env_pause_keys": env_pause_keys,
            "all_sleeves": all_sleeves,
            "auth_ready": paper_auth_ready,
            "broker_ready": paper_broker_ready,
            "strict_auth_ready": strict_auth_ready,
            "strict_broker_ready": strict_broker_ready,
            "paper_soak_auth_grace": bool(paper_auth_ready and not strict_auth_ready),
            "session_ready": session_ready,
            "stale_artifacts": stale_artifacts,
            "controlled_profitability_grade": profitability_grade,
            "raw_profitability_a_recovery_active": _bool(raw_recovery.get("active", False)),
            "raw_profitability_improvement_required": raw_improvement_required,
            "raw_profitability_improvement_ready": raw_improvement_ready,
        },
        evidence=[
            "30_day_soak_invariant=runtime_ramp_paper_consumers_artifacts_fanout_profitability_controls_move_as_one",
        ],
    )


def build_payload(
    project_root: Path = PROJECT_ROOT,
    *,
    runtime_path: Path | None = None,
    paper_path: Path | None = None,
    override_path: Path | None = None,
    process_path: Path | None = None,
    runtime_profitability_path: Path | None = None,
    auth_lease_path: Path | None = None,
    schwab_auth_path: Path | None = None,
    broker_path: Path | None = None,
    session_path: Path | None = None,
) -> dict[str, Any]:
    runtime_path = Path(runtime_path or project_root / "governance" / "health" / "runtime_throttle_control_latest.json")
    paper_path = Path(paper_path or project_root / "governance" / "health" / "paper_400_ramp_latest.json")
    override_path = Path(override_path or project_root / "config" / ".env.runtime_resource_guard_override")
    process_path = Path(process_path or project_root / "governance" / "health" / "process_watchdog_latest.json")
    runtime_profitability_path = Path(
        runtime_profitability_path
        or project_root / "governance" / "health" / "paper_runtime_profitability_controls_latest.json"
    )
    auth_lease_path = Path(auth_lease_path or project_root / "governance" / "health" / "auth_lease_manager_latest.json")
    schwab_auth_path = Path(schwab_auth_path or project_root / "governance" / "health" / "schwab_auth_supervisor_latest.json")
    broker_path = Path(broker_path or project_root / "governance" / "health" / "broker_readiness_latest.json")
    session_path = Path(session_path or project_root / "governance" / "health" / "session_ready_latest.json")
    runtime = load_json(runtime_path)
    paper = load_json(paper_path)
    process = load_json(process_path)
    runtime_profitability = load_json(runtime_profitability_path)
    auth_lease = load_json(auth_lease_path)
    schwab_auth = load_json(schwab_auth_path)
    broker = load_json(broker_path)
    session = load_json(session_path)
    env_values = _parse_env_file(override_path)
    artifact_snapshots = [
        _artifact_snapshot("runtime_throttle_control", runtime_path, runtime, HOT_ARTIFACT_MAX_AGE_MINUTES["runtime_throttle_control"]),
        _artifact_snapshot("paper_400_ramp", paper_path, paper, HOT_ARTIFACT_MAX_AGE_MINUTES["paper_400_ramp"]),
        _artifact_snapshot("process_watchdog", process_path, process, HOT_ARTIFACT_MAX_AGE_MINUTES["process_watchdog"]),
        _artifact_snapshot(
            "paper_runtime_profitability_controls",
            runtime_profitability_path,
            runtime_profitability,
            HOT_ARTIFACT_MAX_AGE_MINUTES["paper_runtime_profitability_controls"],
        ),
    ]

    guards = [
        _runtime_ready_contract_guard(runtime),
        _runtime_guarded_ready_lane_guard(runtime),
        _paper_runtime_capacity_blocker_guard(paper),
        _paper_arm_blocker_guard(paper),
        _paper_execution_pause_guard(runtime, paper, env_values),
        _support_override_guard(runtime, env_values, override_path),
        _hot_artifact_freshness_guard(artifact_snapshots),
        _paper_eligible_lane_open_guard(runtime, paper, process, artifact_snapshots),
        _production_grade_authority_guard(
            project_root,
            runtime,
            paper,
            process,
            runtime_profitability,
            auth_lease,
            schwab_auth,
            broker,
            session,
            env_values,
            artifact_snapshots,
        ),
        _soak_30_day_continuity_guard(
            runtime,
            paper,
            process,
            runtime_profitability,
            auth_lease,
            schwab_auth,
            broker,
            session,
            env_values,
            artifact_snapshots,
        ),
    ]
    failed = [row for row in guards if not bool(row.get("ok", False))]
    hard_failed = [row for row in failed if str(row.get("severity")) in HARD_SEVERITIES]
    degraded_failed = [row for row in failed if str(row.get("severity")) not in HARD_SEVERITIES]
    overall_status = "ready"
    if hard_failed:
        overall_status = "blocked"
    elif degraded_failed:
        overall_status = "degraded"

    failed_names = [str(row.get("name")) for row in failed]
    recommended_actions = ordered_unique(
        [
            "rerun runtime-throttle --apply --json so runtime ready/advisory status matches attribution evidence"
            if any(name.startswith("runtime_") for name in failed_names)
            else "",
            "rerun paper-400-ramp --json and keep paper widening off until runtime_capacity_not_ready_for_400_paper is absent when runtime capacity is ready"
            if "paper_runtime_capacity_blocker_contract" in failed_names
            else "",
            "keep paper execution consumers disabled until the paper ramp is armed with no blockers"
            if "blocked_paper_execution_pause_contract" in failed_names or "paper_armed_blocker_contract" in failed_names
            else "",
            "let runtime-throttle refresh support niceness overrides before launching support collectors"
            if "runtime_override_support_spawn_contract" in failed_names
            else "",
            "run soak-self-heal or reapply runtime-throttle and paper-400-ramp controls when the 30-day continuity contract breaks"
            if "soak_30_day_continuity_contract" in failed_names
            else "",
            "treat production authority failures as hard control-path regressions: keep live locked, refresh auth/broker/runtime/ramp, and do not let stale artifacts pause eligible paper"
            if "production_grade_paper_live_authority_contract" in failed_names
            else "",
        ]
    )

    return {
        "timestamp_utc": iso_now(),
        "schema_version": 1,
        "ok": overall_status == "ready",
        "overall_status": overall_status,
        "runtime_status": runtime.get("overall_status") if runtime else "missing",
        "paper_stage": paper.get("stage") if paper else "missing",
        "paper_armed": _bool(paper.get("armed", False)) if paper else False,
        "paper_blocked": _paper_is_blocked(paper),
        "failed_guard_count": len(failed),
        "hard_failed_guard_count": len(hard_failed),
        "degraded_guard_count": len(degraded_failed),
        "failed_guards": failed_names,
        "regression_guards": guards,
        "assigned_infrabots": list(GUARD_BOTS),
        "active_infrabots": list(GUARD_BOTS) if failed else [],
        "source_artifacts": {
            "runtime_throttle_control": str(runtime_path),
            "paper_400_ramp": str(paper_path),
            "process_watchdog": str(process_path),
            "paper_runtime_profitability_controls": str(runtime_profitability_path),
            "auth_lease_manager": str(auth_lease_path),
            "schwab_auth_supervisor": str(schwab_auth_path),
            "broker_readiness": str(broker_path),
            "session_ready": str(session_path),
            "runtime_override": str(override_path),
            "runtime_artifact_present": bool(runtime),
            "paper_artifact_present": bool(paper),
            "process_artifact_present": bool(process),
            "runtime_profitability_artifact_present": bool(runtime_profitability),
            "auth_lease_artifact_present": bool(auth_lease),
            "schwab_auth_artifact_present": bool(schwab_auth),
            "broker_artifact_present": bool(broker),
            "session_artifact_present": bool(session),
            "runtime_override_present": override_path.exists(),
            "hot_artifact_freshness": artifact_snapshots,
        },
        "runtime_paper_contract": {
            "paper_ramp_blocked_keeps_execution_paused": True,
            "runtime_capacity_ready_must_not_emit_capacity_blocker": True,
            "guarded_runtime_ready_requires_low_bot_owned_pressure": True,
            "support_subprocesses_inherit_runtime_niceness": True,
            "eligible_paper_lanes_must_remain_open": True,
            "stale_hot_artifacts_must_refresh_without_disabling_eligible_paper": True,
            "thirty_day_soak_continuity_must_hold": True,
            "auth_broker_session_must_be_ready_for_continuity": True,
            "paper_only_lanes_fail_open_under_soft_degradation": True,
            "hard_safety_blockers_fail_paper_closed": True,
            "raw_profitability_grade_is_evidence_based": True,
            "incident_replay_regressions_required": True,
            "live_execution_authority": False,
            "paper_widening_authority": False,
        },
        "recommended_actions": recommended_actions,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Guard runtime throttle and paper ramp contracts against regression.")
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--out-file", default=str(DEFAULT_OUT_PATH))
    parser.add_argument("--runtime-file", default="")
    parser.add_argument("--paper-file", default="")
    parser.add_argument("--override-file", default="")
    parser.add_argument("--process-file", default="")
    parser.add_argument("--runtime-profitability-file", default="")
    parser.add_argument("--auth-lease-file", default="")
    parser.add_argument("--schwab-auth-file", default="")
    parser.add_argument("--broker-file", default="")
    parser.add_argument("--session-file", default="")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    payload = build_payload(
        project_root,
        runtime_path=Path(args.runtime_file).expanduser() if args.runtime_file else None,
        paper_path=Path(args.paper_file).expanduser() if args.paper_file else None,
        override_path=Path(args.override_file).expanduser() if args.override_file else None,
        process_path=Path(args.process_file).expanduser() if args.process_file else None,
        runtime_profitability_path=Path(args.runtime_profitability_file).expanduser() if args.runtime_profitability_file else None,
        auth_lease_path=Path(args.auth_lease_file).expanduser() if args.auth_lease_file else None,
        schwab_auth_path=Path(args.schwab_auth_file).expanduser() if args.schwab_auth_file else None,
        broker_path=Path(args.broker_file).expanduser() if args.broker_file else None,
        session_path=Path(args.session_file).expanduser() if args.session_file else None,
    )
    write_payload(Path(args.out_file).expanduser(), payload)
    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        print(
            "runtime_paper_regression_guard "
            f"overall_status={payload.get('overall_status')} "
            f"failed_guards={payload.get('failed_guard_count')}"
        )
    return 0 if payload.get("overall_status") in {"ready", "degraded"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
