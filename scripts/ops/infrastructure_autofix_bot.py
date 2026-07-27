#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from scripts.ops.long_runtime_common import PROJECT_ROOT, iso_now, load_json, ordered_unique, write_payload
else:
    from .long_runtime_common import PROJECT_ROOT, iso_now, load_json, ordered_unique, write_payload


DEFAULT_OUT_PATH = PROJECT_ROOT / "governance" / "health" / "infrastructure_autofix_bot_latest.json"
PYTHON_BIN = Path(sys.executable)
COMMANDS_HYGIENE_SCRIPT = Path(__file__).resolve().with_name("commands_hygiene_bot.py")
COMMAND_VALIDITY_SCRIPT = Path(__file__).resolve().with_name("command_validity_bot.py")
OPTIONS_FLOW_EXPORT_HYGIENE_SCRIPT = Path(__file__).resolve().with_name("options_flow_export_hygiene_bot.py")
OPTIONS_FLOW_EFFICIENCY_SCRIPT = Path(__file__).resolve().with_name("options_flow_efficiency_bot.py")
STORAGE_PRESSURE_CLEARANCE_SCRIPT = Path(__file__).resolve().with_name("storage_pressure_clearance_bot.py")
STORAGE_RECONNECT_INFRABOT_SCRIPT = Path(__file__).resolve().with_name("storage_reconnect_infrabot.py")
CORE_BOT_MATERIALIZATION_INFRABOT_SCRIPT = Path(__file__).resolve().with_name("core_bot_materialization_infrabot.py")
CORE_BOT_TIER_ORGANIZER_SCRIPT = Path(__file__).resolve().with_name("organize_core_bot_tiers.py")
ONE_NUMBERS_REGRESSION_GUARD_SCRIPT = Path(__file__).resolve().with_name("one_numbers_regression_guard.py")
STATEFUL_STORAGE_REGRESSION_GUARD_SCRIPT = Path(__file__).resolve().with_name("stateful_storage_regression_guard.py")
RUNTIME_PAPER_REGRESSION_GUARD_SCRIPT = Path(__file__).resolve().with_name("runtime_paper_regression_guard.py")
MASTER_INFRA_SUPERVISOR_SCRIPT = Path(__file__).resolve().with_name("master_infrastructure_supervisor.py")
STALE_SURFACE_AUTOHEALER_SCRIPT = Path(__file__).resolve().with_name("stale_surface_autohealer.py")
HOST_CAPABILITY_SCRIPT = Path(__file__).resolve().with_name("host_capability_contract.py")
HALT_TRIGGER_CONTROL_SCRIPT = Path(__file__).resolve().with_name("halt_trigger_control_plane.py")
COORDINATION_STATE_CONTROL_SCRIPT = Path(__file__).resolve().with_name("coordination_state_control.py")
WHOLE_SYSTEM_GOVERNOR_SCRIPT = Path(__file__).resolve().with_name("whole_system_governor.py")
LIBRARY_UTILIZATION_ROUTER_SCRIPT = Path(__file__).resolve().with_name("library_utilization_router.py")
MLX_INTELLIGENCE_ROUTER_SCRIPT = Path(__file__).resolve().with_name("mlx_intelligence_router.py")
LIBRARY_UPGRADE_ROUTE_CONTROL_SCRIPT = Path(__file__).resolve().with_name("library_upgrade_route_control.py")
SHADOW_WATCHDOG_SCRIPT = PROJECT_ROOT / "scripts" / "shadow_watchdog.py"
HEAVY_FEED_GUARD_SCRIPT = PROJECT_ROOT / "scripts" / "ops" / "live_feed_heavy_guarded.sh"
REQUIRED_COLLECTOR_REFRESH_NAMES = {
    "official_macro_context",
    "market_micro_context",
    "crypto_market_context",
    "fx_market_context",
}
SYSTEM_DRIFT_AUTOFIX_STEP_TIMEOUT_SECONDS = 90
REPAIR_CALL_STACK_ENV = "INFRA_REPAIR_CALL_STACK"


def _repair_call_stack() -> list[str]:
    return [
        item.strip()
        for item in str(os.getenv(REPAIR_CALL_STACK_ENV, "") or "").split(",")
        if item.strip()
    ]


def _stack_contains(component: str) -> bool:
    return str(component or "").strip() in set(_repair_call_stack())


def _child_env(component: str) -> dict[str, str]:
    env = os.environ.copy()
    stack = _repair_call_stack()
    name = str(component or "").strip()
    if name and name not in stack:
        stack.append(name)
    env[REPAIR_CALL_STACK_ENV] = ",".join(stack)
    return env


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


def _artifact_candidates(project_root: Path, raw_path: str | Path) -> list[Path]:
    path = Path(raw_path)
    if path.is_absolute():
        return [path]
    candidates = [project_root / path]
    parts = path.parts
    if parts and parts[0] in {"data", "decisions", "decision_explanations", "exports", "governance", "logs", "models"}:
        candidates.append(project_root / "local_fallback_storage" / path)
        candidates.append(Path("/Volumes/BOT_LOGS/schwab_trading_bot") / path)
    out: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        out.append(candidate)
    return out


def _load_freshest_json(project_root: Path, raw_path: str | Path) -> dict[str, Any]:
    best_payload: dict[str, Any] = {}
    best_mtime = -1.0
    for candidate in _artifact_candidates(project_root, raw_path):
        payload = load_json(candidate)
        if not payload:
            continue
        try:
            mtime = candidate.stat().st_mtime
        except Exception:
            mtime = 0.0
        if mtime >= best_mtime:
            best_payload = payload
            best_mtime = mtime
    return best_payload


def _collector_contract_row(payload: dict[str, Any], name: str) -> dict[str, Any]:
    rows = payload.get("rows") if isinstance(payload.get("rows"), list) else []
    for row in rows:
        if isinstance(row, dict) and str(row.get("name") or "").strip() == name:
            return row
    return {}


def _source_verification_row(payload: dict[str, Any], source_id: str) -> dict[str, Any]:
    rows = payload.get("sources") if isinstance(payload.get("sources"), list) else []
    for row in rows:
        if isinstance(row, dict) and str(row.get("source_id") or "").strip() == source_id:
            return row
    return {}


def _required_collector_failures(payload: dict[str, Any]) -> list[str]:
    failures = [
        str(raw or "").strip()
        for raw in (payload.get("required_failures") if isinstance(payload.get("required_failures"), list) else [])
        if str(raw or "").strip()
    ]
    rows = payload.get("rows") if isinstance(payload.get("rows"), list) else []
    for row in rows:
        if not isinstance(row, dict) or not bool(row.get("required", False)):
            continue
        name = str(row.get("name") or "").strip()
        if name and not bool(row.get("contract_ok", False)):
            failures.append(name)
    return ordered_unique(failures)


def _required_collector_refresh_command(project_root: Path, collector_name: str) -> list[str]:
    name = str(collector_name or "").strip()
    if name == "official_macro_context":
        return [str(PYTHON_BIN), str(project_root / "scripts" / "collect_official_macro_context.py"), "--json"]
    if name == "market_micro_context":
        return [
            str(PYTHON_BIN),
            str(project_root / "scripts" / "collect_market_micro_context.py"),
            "--lookback-days",
            "21",
            "--finra-lookback-days",
            "15",
            "--json",
        ]
    if name == "crypto_market_context":
        return [str(PYTHON_BIN), str(project_root / "scripts" / "collect_crypto_market_context.py"), "--json"]
    if name == "fx_market_context":
        return [str(PYTHON_BIN), str(project_root / "scripts" / "collect_fx_market_context.py"), "--json"]
    return []


def _run_json(cmd: list[str], *, cwd: Path, timeout_sec: int) -> dict[str, Any]:
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(cwd),
            capture_output=True,
            text=True,
            check=False,
            timeout=max(int(timeout_sec), 1),
            env=_child_env("infrastructure_autofix_bot"),
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
        "timeout_sec": max(int(timeout_sec), 1),
        "stdout_tail": "\n".join(stdout.splitlines()[-10:]),
        "stderr_tail": "\n".join(stderr.splitlines()[-10:]),
        "payload": payload,
    }


def _remaining_timeout_seconds(deadline_monotonic: float, *, per_command_cap: int) -> int:
    remaining = float(deadline_monotonic) - time.monotonic()
    if remaining <= 0:
        return 0
    return max(1, min(int(per_command_cap), int(math.ceil(remaining))))


def _budget_exhausted_attempt(cmd: list[str]) -> dict[str, Any]:
    return {
        "cmd": list(cmd),
        "rc": 124,
        "timed_out": True,
        "skipped": True,
        "reason": "run_timeout_budget_exhausted",
        "timeout_sec": 0,
        "stdout_tail": "",
        "stderr_tail": "",
        "payload": {},
    }


def _blocked_surface_names(payload: dict[str, Any]) -> set[str]:
    rows = payload.get("surfaces") if isinstance(payload.get("surfaces"), list) else []
    return {
        str(row.get("name") or "")
        for row in rows
        if isinstance(row, dict) and str(row.get("status") or "").strip().lower() == "blocked"
    }


def _blocked_check_names(payload: dict[str, Any]) -> set[str]:
    rows = payload.get("checks") if isinstance(payload.get("checks"), list) else []
    return {
        str(row.get("name") or "")
        for row in rows
        if isinstance(row, dict) and str(row.get("status") or "").strip().lower() == "blocked"
    }


def _system_drift_is_self_referential(payload: dict[str, Any]) -> bool:
    blocked = _blocked_surface_names(payload)
    return bool(blocked) and blocked <= {"infrastructure_autofix", "master_infrastructure_supervisor"}


def _master_infra_is_self_referential(payload: dict[str, Any]) -> bool:
    blocked = _blocked_check_names(payload)
    return bool(blocked) and blocked <= {
        "governance_artifact_freshness",
        "child_repair_bot_outcomes",
        "self_auditing_infra_bots",
    }


QUALITY_DEBT_PRIORITIES = {
    "runtime_input_coverage",
    "active_probation_isolation",
    "teacher_quality",
    "targeted_retrain",
    "walk_forward_coverage",
}
QUALITY_DEBT_ACTION_KEYS = {
    "refresh_diagnostics_bot_ids",
    "unsupported_stale_bot_ids",
    "repair_runtime_input_bot_ids",
    "runtime_input_depth_debt_bot_ids",
    "quality_probation_bot_ids",
    "targeted_retrain_bot_ids",
    "selected_targeted_retrain_bot_ids",
    "coverage_repair_bot_ids",
    "precompute_target_bot_ids",
}


def _training_quality_is_quality_debt_only(payload: dict[str, Any]) -> bool:
    status = str(payload.get("overall_status") or "").strip().lower()
    if status not in {"blocked", "degraded"}:
        return True
    recoverable_blocked = payload.get("recoverable_blocked_keys")
    if isinstance(recoverable_blocked, list) and any(str(item).strip() for item in recoverable_blocked):
        return False
    priorities = {str(item).strip() for item in payload.get("top_priorities", []) if str(item).strip()} if isinstance(payload.get("top_priorities"), list) else set()
    targeted_actions = payload.get("targeted_actions") if isinstance(payload.get("targeted_actions"), dict) else {}
    has_quality_queue = any(
        isinstance(targeted_actions.get(key), list) and bool(targeted_actions.get(key))
        for key in QUALITY_DEBT_ACTION_KEYS
    )
    return bool(has_quality_queue or (priorities and priorities <= QUALITY_DEBT_PRIORITIES))


def build_payload(
    project_root: Path = PROJECT_ROOT,
    *,
    apply: bool = False,
    timeout_sec: int = 1200,
) -> dict[str, Any]:
    daily_verify = _load_freshest_json(project_root, "governance/health/daily_auto_verify_latest.json")
    storage_control = _load_freshest_json(project_root, "governance/health/ingestion_storage_control_latest.json")
    collector_contracts = _load_freshest_json(project_root, "governance/health/collector_contracts_latest.json")
    source_verification = _load_freshest_json(project_root, "governance/health/source_verification_latest.json")
    options_flow_context = _load_freshest_json(project_root, "governance/health/options_flow_context_sync_latest.json")
    if not options_flow_context:
        options_flow_context = _load_freshest_json(project_root, "governance/health/tastytrade_context_sync_latest.json")
    schwab_education_context = _load_freshest_json(project_root, "governance/health/schwab_education_context_sync_latest.json")
    auth_lease = _load_freshest_json(project_root, "governance/health/auth_lease_manager_latest.json")
    schwab_auth_supervisor = _load_freshest_json(project_root, "governance/health/schwab_auth_supervisor_latest.json")
    blackstart = _load_freshest_json(project_root, "governance/health/blackstart_recovery_latest.json")
    freshness = _load_freshest_json(project_root, "governance/health/artifact_freshness_slo_latest.json")
    snapshot_cache = _load_freshest_json(project_root, "governance/health/runtime_snapshot_cache_control_latest.json")
    remote_alert = _load_freshest_json(project_root, "governance/health/remote_alert_control_latest.json")
    training_quality = _load_freshest_json(project_root, "governance/health/training_quality_control_latest.json")
    supportability = _load_freshest_json(project_root, "governance/health/supportability_control_latest.json")
    bot_quality = _load_freshest_json(project_root, "governance/health/bot_quality_autopilot_latest.json")
    system_drift = _load_freshest_json(project_root, "governance/health/system_drift_guard_latest.json")
    one_numbers_guard = _load_freshest_json(project_root, "governance/health/one_numbers_regression_guard_latest.json")
    stateful_storage_guard = _load_freshest_json(project_root, "governance/health/stateful_storage_regression_guard_latest.json")
    runtime_paper_guard = _load_freshest_json(project_root, "governance/health/runtime_paper_regression_guard_latest.json")
    host_capability = _load_freshest_json(project_root, "governance/health/host_capability_contract_latest.json")
    halt_trigger = _load_freshest_json(project_root, "governance/health/halt_trigger_control_plane_latest.json")
    coordination_state = _load_freshest_json(project_root, "governance/health/coordination_state_latest.json")
    whole_system_governor = _load_freshest_json(project_root, "governance/health/whole_system_governor_latest.json")
    library_utilization = _load_freshest_json(project_root, "governance/health/library_utilization_router_latest.json")
    mlx_intelligence = _load_freshest_json(project_root, "governance/health/mlx_intelligence_router_latest.json")
    library_upgrade_route = _load_freshest_json(project_root, "governance/health/library_upgrade_route_control_latest.json")
    storage_reconnect = _load_freshest_json(project_root, "governance/health/storage_reconnect_infrabot_latest.json")
    storage_reconnect_guard = _load_freshest_json(project_root, "governance/health/storage_reconnect_regression_guard_latest.json")
    core_bot_materialization = _load_freshest_json(project_root, "governance/health/core_bot_materialization_infrabot_latest.json")
    core_bot_materialization_guard = _load_freshest_json(project_root, "governance/health/core_bot_materialization_guard_latest.json")
    core_bot_tier_organizer = _load_freshest_json(project_root, "governance/health/core_bot_tier_organizer_latest.json")
    master_infra = _load_freshest_json(project_root, "governance/health/master_infrastructure_supervisor_latest.json")
    process_watchdog = _load_freshest_json(project_root, "governance/health/process_watchdog_latest.json")
    stale_surface_autohealer = _load_freshest_json(project_root, "governance/health/stale_surface_autohealer_latest.json")

    failed_checks = daily_verify.get("failed_checks") if isinstance(daily_verify.get("failed_checks"), list) else []
    repair_plan: list[dict[str, Any]] = []
    advisory_repair_plan: list[dict[str, Any]] = []
    operator_followups: list[str] = []

    def add_plan(name: str, reason: str, cmd: list[str], *, advisory: bool = False) -> None:
        target = advisory_repair_plan if advisory else repair_plan
        target.append({"name": name, "reason": reason, "cmd": cmd})

    if failed_checks:
        add_plan(
            "daily_verify_auto_remediation",
            f"failed_checks={len(failed_checks)}",
            [str(PYTHON_BIN), str(project_root / "scripts" / "ops" / "daily_verify_auto_remediation_bot.py"), "--apply", "--json"],
        )

    storage_status = str(storage_control.get("overall_status") or "")
    storage_backpressure = storage_control.get("backpressure") if isinstance(storage_control.get("backpressure"), dict) else {}
    retention_debt_gb = _safe_float(((storage_control.get("storage") or {}).get("retention_debt_gb")), 0.0)
    total_pending_lines = _safe_int(storage_backpressure.get("total_pending_lines"), 0)
    total_drain_minutes = _safe_float(storage_backpressure.get("estimated_total_drain_minutes"), 0.0)
    if storage_status in {"blocked", "degraded", "needs_work"}:
        add_plan(
            "storage_pressure_clearance",
            f"ingestion_storage_status={storage_status} retention_debt_gb={retention_debt_gb:.3f} pending_lines={total_pending_lines} total_drain_minutes={total_drain_minutes:.3f}",
            [str(PYTHON_BIN), str(STORAGE_PRESSURE_CLEARANCE_SCRIPT), "--apply", "--force-clear-stale-gate", "--json"],
        )

    stateful_storage_status = str(stateful_storage_guard.get("overall_status") or "")
    if not stateful_storage_guard or stateful_storage_status in {"blocked", "degraded", "needs_work"}:
        add_plan(
            "stateful_storage_regression_guard",
            f"stateful_storage_status={stateful_storage_status or 'missing'}",
            [str(PYTHON_BIN), str(STATEFUL_STORAGE_REGRESSION_GUARD_SCRIPT), "--apply", "--json"],
        )

    runtime_paper_status = str(runtime_paper_guard.get("overall_status") or "")
    runtime_paper_failed = _safe_int(runtime_paper_guard.get("failed_guard_count"), 0)
    if not runtime_paper_guard or runtime_paper_status in {"blocked", "degraded", "needs_work"} or runtime_paper_failed > 0:
        add_plan(
            "runtime_paper_regression_guard",
            f"runtime_paper_guard_status={runtime_paper_status or 'missing'} failed_guards={runtime_paper_failed}",
            [str(PYTHON_BIN), str(RUNTIME_PAPER_REGRESSION_GUARD_SCRIPT), "--json"],
        )

    host_capability_status = str(host_capability.get("overall_status") or "")
    if not host_capability or host_capability_status in {"blocked", "degraded", "needs_work"}:
        add_plan(
            "host_capability_contract",
            f"host_capability_status={host_capability_status or 'missing'}",
            [str(PYTHON_BIN), str(HOST_CAPABILITY_SCRIPT), "--json"],
        )

    library_status = str(library_utilization.get("overall_status") or "")
    if not library_utilization or library_status in {"blocked", "degraded", "needs_work"}:
        add_plan(
            "library_utilization_router",
            f"library_utilization_status={library_status or 'missing'}",
            [str(PYTHON_BIN), str(LIBRARY_UTILIZATION_ROUTER_SCRIPT), "--apply", "--json"],
        )

    mlx_status = str(mlx_intelligence.get("overall_status") or "")
    mlx_coverage = mlx_intelligence.get("library_coverage") if isinstance(mlx_intelligence.get("library_coverage"), dict) else {}
    if (
        not mlx_intelligence
        or mlx_status in {"blocked", "degraded", "needs_work"}
        or _safe_float(mlx_coverage.get("coverage_ratio"), 0.0) < 1.0
    ):
        add_plan(
            "mlx_intelligence_router",
            f"mlx_intelligence_status={mlx_status or 'missing'} coverage={_safe_float(mlx_coverage.get('coverage_ratio'), 0.0):.4f}",
            [str(PYTHON_BIN), str(MLX_INTELLIGENCE_ROUTER_SCRIPT), "--apply", "--json"],
        )

    upgrade_status = str(library_upgrade_route.get("overall_status") or "")
    upgrade_plan = library_upgrade_route.get("upgrade_plan") if isinstance(library_upgrade_route.get("upgrade_plan"), dict) else {}
    if (
        not library_upgrade_route
        or upgrade_status in {"blocked", "degraded", "needs_work"}
        or _safe_int(upgrade_plan.get("hard_blocker_count"), 0) > 0
    ):
        add_plan(
            "library_upgrade_route_control",
            f"library_upgrade_route_status={upgrade_status or 'missing'} hard_blockers={_safe_int(upgrade_plan.get('hard_blocker_count'), 0)}",
            [str(PYTHON_BIN), str(LIBRARY_UPGRADE_ROUTE_CONTROL_SCRIPT), "--apply", "--json"],
        )

    halt_status = str(halt_trigger.get("overall_status") or halt_trigger.get("status") or "")
    coordination_status = str(coordination_state.get("overall_status") or "")
    coordination_issue_names = [
        str(row.get("name") or row.get("source") or "")
        for row in coordination_state.get("artifact_issues", [])
        if isinstance(row, dict)
    ]
    if (
        not halt_trigger
        or halt_status in {"stale", "missing", "invalid_json"}
        or any("halt_trigger_control_plane" in name for name in coordination_issue_names)
    ):
        add_plan(
            "halt_trigger_control_plane",
            f"halt_trigger_status={halt_status or 'missing'}",
            [str(PYTHON_BIN), str(HALT_TRIGGER_CONTROL_SCRIPT), "--json"],
        )
    if any("shadow_watchdog_tripwire" in name for name in coordination_issue_names):
        add_plan(
            "shadow_watchdog_tripwire_refresh",
            "coordination_state_reports_shadow_watchdog_tripwire_stale",
            [str(PYTHON_BIN), str(SHADOW_WATCHDOG_SCRIPT), "--once", "--dry-run", "--json"],
            advisory=True,
        )
    if any("heavy_livefeed" in name or "live_feed_heavy_guarded" in name for name in coordination_issue_names):
        add_plan(
            "live_feed_heavy_guard_refresh",
            "coordination_state_reports_heavy_livefeed_guard_stale",
            [str(HEAVY_FEED_GUARD_SCRIPT), "--check-only", "--no-color"],
            advisory=True,
        )
    if not coordination_state or coordination_status in {"blocked", "degraded", "needs_work"}:
        add_plan(
            "coordination_state_control",
            f"coordination_state_status={coordination_status or 'missing'}",
            [str(PYTHON_BIN), str(COORDINATION_STATE_CONTROL_SCRIPT), "--json"],
        )

    whole_system_status = str(whole_system_governor.get("overall_status") or whole_system_governor.get("status") or "")
    if not whole_system_governor or whole_system_status in {"blocked", "degraded", "needs_work", "missing"}:
        add_plan(
            "whole_system_governor",
            f"whole_system_governor_status={whole_system_status or 'missing'}",
            [str(PYTHON_BIN), str(WHOLE_SYSTEM_GOVERNOR_SCRIPT), "--apply", "--json"],
        )

    storage_reconnect_status = str(storage_reconnect.get("overall_status") or "")
    storage_reconnect_guard_status = str(storage_reconnect_guard.get("overall_status") or "")
    if (
        not storage_reconnect
        or storage_reconnect_status in {"blocked", "degraded", "needs_work"}
        or storage_reconnect_guard_status in {"blocked", "degraded", "needs_work"}
    ):
        add_plan(
            "storage_reconnect_infrabot",
            f"reconnect_status={storage_reconnect_status or 'missing'} guard_status={storage_reconnect_guard_status or 'missing'}",
            [str(PYTHON_BIN), str(STORAGE_RECONNECT_INFRABOT_SCRIPT), "--apply", "--json"],
        )

    core_materialization_status = str(core_bot_materialization.get("overall_status") or "")
    core_materialization_guard_status = str(core_bot_materialization_guard.get("overall_status") or "")
    if (
        not core_bot_materialization
        or core_materialization_status in {"blocked", "degraded", "needs_work"}
        or core_materialization_guard_status in {"blocked", "degraded", "needs_work"}
    ):
        add_plan(
            "core_bot_materialization_infrabot",
            f"materialization_status={core_materialization_status or 'missing'} guard_status={core_materialization_guard_status or 'missing'}",
            [str(PYTHON_BIN), str(CORE_BOT_MATERIALIZATION_INFRABOT_SCRIPT), "--apply", "--json"],
        )
    core_tier_status = str(core_bot_tier_organizer.get("overall_status") or "")
    if not core_bot_tier_organizer or core_tier_status in {"blocked", "degraded", "needs_work"}:
        add_plan(
            "core_bot_tier_organizer",
            f"tier_status={core_tier_status or 'missing'}",
            [str(PYTHON_BIN), str(CORE_BOT_TIER_ORGANIZER_SCRIPT), "--json"],
        )

    one_numbers_status = str(one_numbers_guard.get("overall_status") or "")
    one_numbers_weaknesses = one_numbers_guard.get("weaknesses") if isinstance(one_numbers_guard.get("weaknesses"), list) else []
    if not one_numbers_guard or one_numbers_status in {"blocked", "degraded"} or one_numbers_weaknesses:
        add_plan(
            "one_numbers_regression_guard",
            f"one_numbers_status={one_numbers_status or 'missing'} weaknesses={len(one_numbers_weaknesses)}",
            [str(PYTHON_BIN), str(ONE_NUMBERS_REGRESSION_GUARD_SCRIPT), "--apply", "--json"],
        )

    schwab_contract = _collector_contract_row(collector_contracts, "schwab_education_context")
    for collector_name in _required_collector_failures(collector_contracts):
        if collector_name not in REQUIRED_COLLECTOR_REFRESH_NAMES:
            continue
        refresh_cmd = _required_collector_refresh_command(project_root, collector_name)
        if not refresh_cmd:
            continue
        add_plan(
            f"{collector_name}_refresh",
            "required_collector_contract_not_ok",
            list(refresh_cmd),
        )
    if (
        (schwab_contract and not bool(schwab_contract.get("contract_ok", False)))
        or (schwab_education_context and not bool(schwab_education_context.get("ok", False)))
    ):
        add_plan(
            "schwab_education_refresh",
            "schwab_education_contract_not_ok",
            [str(PYTHON_BIN), str(project_root / "scripts" / "collect_schwab_education_context.py"), "--json"],
        )

    options_flow_row = _source_verification_row(source_verification, "polygon_unusual_whales_options_context")
    options_flow_sources = options_flow_context.get("sources") if isinstance(options_flow_context.get("sources"), dict) else {}
    options_flow_export = (
        options_flow_sources.get("unusual_whales_export")
        if isinstance(options_flow_sources.get("unusual_whales_export"), dict)
        else {}
    )
    if bool(options_flow_export.get("configured", False)) and (
        not bool(options_flow_export.get("ok", False))
        or bool(options_flow_export.get("errors"))
        or int(options_flow_export.get("rejected_row_count", 0) or 0) > 0
    ):
        add_plan(
            "options_flow_export_hygiene",
            "options_flow_export_handoff_needs_attention",
            [str(PYTHON_BIN), str(OPTIONS_FLOW_EXPORT_HYGIENE_SCRIPT), "--apply", "--json"],
        )
    if (
        (options_flow_row and str(options_flow_row.get("verification_status") or "") == "single_source_unverified")
        or (options_flow_context and str(options_flow_context.get("overall_status") or "") in {"blocked", "degraded"})
        or (options_flow_context and not bool(options_flow_context.get("ok", False)))
    ):
        add_plan(
            "options_flow_efficiency",
            "options_flow_context_not_ready",
            [str(PYTHON_BIN), str(OPTIONS_FLOW_EFFICIENCY_SCRIPT), "--apply", "--json"],
        )
    if bool(options_flow_context.get("operator_action_required", False)):
        auth_issue = str(options_flow_context.get("auth_issue") or "").strip() or "options_flow_auth_issue"
        operator_followups.append(
            f"review Polygon and Unusual Whales access because the collector reported {auth_issue} and the autofix bot cannot rotate API keys or invent export feeds for you"
        )

    commands_hygiene = _run_json(
        [str(PYTHON_BIN), str(COMMANDS_HYGIENE_SCRIPT), "--project-root", str(project_root), "--json"],
        cwd=project_root,
        timeout_sec=min(int(timeout_sec), 180),
    )
    commands_hygiene_payload = commands_hygiene.get("payload") if isinstance(commands_hygiene.get("payload"), dict) else {}
    command_validity = _run_json(
        [str(PYTHON_BIN), str(COMMAND_VALIDITY_SCRIPT), "--project-root", str(project_root), "--json"],
        cwd=project_root,
        timeout_sec=min(int(timeout_sec), 180),
    )
    command_validity_payload = command_validity.get("payload") if isinstance(command_validity.get("payload"), dict) else {}
    if (
        str(commands_hygiene_payload.get("overall_status") or "") in {"degraded", "blocked"}
        or bool(commands_hygiene_payload.get("commands_changed", False))
        or bool(commands_hygiene_payload.get("runbook_changed", False))
    ):
        add_plan(
            "commands_hygiene",
            "commands_md_or_runbook_drift",
            [
                str(PYTHON_BIN),
                str(COMMANDS_HYGIENE_SCRIPT),
                "--project-root",
                str(project_root),
                "--apply",
                "--json",
            ],
        )
    if (
        str(command_validity_payload.get("overall_status") or "") == "blocked"
        or _safe_int((((command_validity_payload.get("metrics") or {}).get("blocked_entry_count"))), 0) > 0
    ):
        add_plan(
            "command_validity",
            "commands_md_contains_invalid_or_unresolved_entries",
            [
                str(PYTHON_BIN),
                str(COMMAND_VALIDITY_SCRIPT),
                "--project-root",
                str(project_root),
                "--apply",
                "--json",
            ],
        )
    drift_status = str(system_drift.get("overall_status") or "")
    core_infra_clear = (
        not failed_checks
        and storage_status not in {"blocked", "degraded", "needs_work"}
        and stateful_storage_status not in {"blocked", "degraded", "needs_work"}
        and runtime_paper_failed == 0
        and _safe_int((((command_validity_payload.get("metrics") or {}).get("blocked_entry_count"))), 0) == 0
        and bool(((snapshot_cache.get("cache_health") or {}).get("snapshot_ready", False)))
    )
    drift_is_advisory = drift_status == "degraded" or (
        drift_status == "blocked" and core_infra_clear and _system_drift_is_self_referential(system_drift)
    )
    if drift_status in {"blocked", "degraded"}:
        add_plan(
            "system_drift_autopilot",
            f"system_drift_status={drift_status}",
            [
                str(PYTHON_BIN),
                str(project_root / "scripts" / "ops" / "system_drift_autopilot.py"),
                "--apply",
                "--max-step-timeout-seconds",
                str(SYSTEM_DRIFT_AUTOFIX_STEP_TIMEOUT_SECONDS),
                "--json",
            ],
            advisory=drift_is_advisory,
        )

    master_status = str(master_infra.get("overall_status") or "")
    nested_under_master_supervisor = _stack_contains("master_infrastructure_supervisor")
    master_is_advisory = nested_under_master_supervisor or (bool(master_infra) and (
        master_status == "degraded"
        or (master_status == "blocked" and core_infra_clear and _master_infra_is_self_referential(master_infra))
    ))
    if not master_infra or master_status in {"blocked", "degraded"}:
        add_plan(
            "master_infrastructure_supervisor_refresh",
            (
                f"master_infra_status={master_status or 'missing'}"
                + (" nested_under_master_supervisor=1" if nested_under_master_supervisor else "")
            ),
            [str(PYTHON_BIN), str(MASTER_INFRA_SUPERVISOR_SCRIPT), "--json"],
            advisory=master_is_advisory,
        )
        for check in master_infra.get("checks") or []:
            if not isinstance(check, dict) or check.get("name") != "launchd_job_health":
                continue
            if check.get("status") in {"blocked", "degraded"}:
                add_plan(
                    "ops_automation_launchd_install",
                    "launchd_job_health_not_ready",
                    [str(project_root / "scripts" / "ops" / "install_ops_automation_launchd.sh")],
                )
            break

    auth_status = str(auth_lease.get("overall_status") or "")
    if auth_status in {"blocked", "degraded"}:
        add_plan(
            "schwab_auth_supervisor",
            f"auth_lease_status={auth_status}",
            [str(PYTHON_BIN), str(project_root / "scripts" / "ops" / "schwab_auth_supervisor.py"), "--apply", "--json"],
        )

    schwab_auth_status = str(schwab_auth_supervisor.get("overall_status") or "")
    if schwab_auth_status in {"blocked", "degraded"}:
        add_plan(
            "schwab_auth_supervisor",
            f"schwab_auth_supervisor_status={schwab_auth_status}",
            [str(PYTHON_BIN), str(project_root / "scripts" / "ops" / "schwab_auth_supervisor.py"), "--apply", "--json"],
        )

    if str(freshness.get("overall_status") or "") in {"blocked", "degraded"}:
        add_plan(
            "artifact_freshness_refresh",
            f"artifact_freshness_status={str(freshness.get('overall_status') or '')}",
            [str(PYTHON_BIN), str(project_root / "scripts" / "ops" / "artifact_freshness_slo.py"), "--json"],
        )

    freshness_summary = freshness.get("sla_summary") if isinstance(freshness.get("sla_summary"), dict) else {}
    process_watchdog_status = str(process_watchdog.get("overall_status") or "")
    watchdog_intelligence = (
        process_watchdog.get("watchdog_intelligence")
        if isinstance(process_watchdog.get("watchdog_intelligence"), dict)
        else {}
    )
    stale_surface_metrics = (
        stale_surface_autohealer.get("metrics")
        if isinstance(stale_surface_autohealer.get("metrics"), dict)
        else {}
    )
    stale_signal_count = (
        _safe_int(freshness_summary.get("stale_required"), 0)
        + _safe_int(freshness_summary.get("stale_optional"), 0)
        + _safe_int(watchdog_intelligence.get("active_issue_count"), 0)
        + _safe_int(stale_surface_metrics.get("planned_repair_count"), 0)
    )
    stale_surface_status = str(stale_surface_autohealer.get("overall_status") or "")
    if (
        stale_signal_count > 0
        or process_watchdog_status in {"blocked", "degraded", "critical"}
        or stale_surface_status in {"blocked", "degraded"}
    ):
        add_plan(
            "stale_surface_autohealer",
            (
                f"stale_signal_count={stale_signal_count} "
                f"process_watchdog_status={process_watchdog_status or 'missing'} "
                f"stale_surface_status={stale_surface_status or 'missing'}"
            ),
            [str(PYTHON_BIN), str(STALE_SURFACE_AUTOHEALER_SCRIPT), "--apply", "--timeout-sec", "60", "--json"],
        )

    if str(snapshot_cache.get("overall_status") or "") in {"blocked", "degraded"}:
        add_plan(
            "runtime_snapshot_refresh",
            f"snapshot_cache_status={str(snapshot_cache.get('overall_status') or '')}",
            [str(PYTHON_BIN), str(project_root / "scripts" / "build_runtime_training_snapshot.py"), "--json"],
        )
        add_plan(
            "runtime_snapshot_recheck",
            "refresh_runtime_snapshot_controls",
            [str(PYTHON_BIN), str(project_root / "scripts" / "ops" / "runtime_snapshot_cache_control.py"), "--json"],
        )

    if str(blackstart.get("overall_status") or "") in {"blocked", "degraded"}:
        add_plan(
            "restart_sanity_bundle",
            f"blackstart_status={str(blackstart.get('overall_status') or '')}",
            [str(PYTHON_BIN), str(project_root / "scripts" / "ops" / "restart_sanity_bundle.py"), "--json"],
        )
        add_plan(
            "blackstart_recheck",
            "refresh_blackstart_status",
            [str(PYTHON_BIN), str(project_root / "scripts" / "ops" / "blackstart_recovery.py"), "--json"],
        )

    if (
        str(training_quality.get("overall_status") or "") in {"blocked", "degraded"}
        or str(supportability.get("overall_status") or "") in {"blocked", "degraded"}
        or str(bot_quality.get("overall_status") or "") in {"blocked", "degraded"}
    ):
        quality_advisory = bool(
            str(supportability.get("overall_status") or "") not in {"blocked", "degraded"}
            and _training_quality_is_quality_debt_only(training_quality)
        )
        add_plan(
            "bot_quality_autopilot",
            "bot_quality_or_supportability_degraded",
            [str(PYTHON_BIN), str(project_root / "scripts" / "ops" / "bot_quality_autopilot.py"), "--apply", "--json"],
            advisory=quality_advisory,
        )

    channels = remote_alert.get("channels") if isinstance(remote_alert.get("channels"), dict) else {}
    if not bool(channels.get("any_configured", False)):
        operator_followups.append("configure at least one remote alert channel because the autofix bot cannot invent webhook or pushover credentials")
    if str(remote_alert.get("overall_status") or "") == "blocked" and _safe_int(((remote_alert.get("critical_backlog") or {}).get("unsent_count")), 0) > 0:
        operator_followups.append("review unsent critical alerts after configuring remote alert channels so the pager backlog drains cleanly")

    timeout_budget_seconds = max(int(timeout_sec), 1)
    run_deadline_monotonic = time.monotonic() + timeout_budget_seconds
    timeout_budget_exhausted = False
    attempts: list[dict[str, Any]] = []

    def run_with_budget(cmd: list[str], *, per_command_cap: int) -> None:
        nonlocal timeout_budget_exhausted
        remaining_timeout = _remaining_timeout_seconds(run_deadline_monotonic, per_command_cap=per_command_cap)
        if remaining_timeout <= 0:
            timeout_budget_exhausted = True
            attempts.append(_budget_exhausted_attempt(cmd))
            return
        attempts.append(_run_json(cmd, cwd=project_root, timeout_sec=remaining_timeout))
        if _remaining_timeout_seconds(run_deadline_monotonic, per_command_cap=1) <= 0:
            timeout_budget_exhausted = True

    if apply:
        for row in repair_plan:
            run_with_budget(list(row.get("cmd") or []), per_command_cap=timeout_budget_seconds)
        for cmd in (
            [str(PYTHON_BIN), str(project_root / "scripts" / "ops" / "runtime_gate_dashboard.py"), "--json"],
            [str(PYTHON_BIN), str(project_root / "scripts" / "ops" / "operator_cockpit.py"), "--json"],
            [str(PYTHON_BIN), str(project_root / "scripts" / "collector_contracts.py"), "--json"],
            [str(PYTHON_BIN), str(project_root / "scripts" / "ops" / "source_verification_report.py"), "--json"],
        ):
            run_with_budget(cmd, per_command_cap=min(timeout_budget_seconds, 300))

    hard_failed_attempts = [
        row
        for row in attempts
        if bool(row.get("timed_out", False)) or int(row.get("rc", 1)) not in {0, 2}
    ]
    degraded_attempts = [row for row in attempts if int(row.get("rc", 1)) == 2 and not bool(row.get("timed_out", False))]
    remaining_repair_plan = list(repair_plan)
    remaining_advisory_repair_plan = list(advisory_repair_plan)
    post_apply_recheck: dict[str, Any] = {
        "enabled": False,
        "status": "",
        "repair_count": None,
        "advisory_count": None,
    }
    if apply and repair_plan and not hard_failed_attempts and not degraded_attempts and not timeout_budget_exhausted:
        recheck_payload = build_payload(project_root, apply=False, timeout_sec=timeout_sec)
        remaining_repair_plan = list(recheck_payload.get("repair_plan") or [])
        remaining_advisory_repair_plan = list(recheck_payload.get("advisory_repair_plan") or [])
        post_apply_recheck = {
            "enabled": True,
            "status": str(recheck_payload.get("overall_status") or ""),
            "repair_count": len(remaining_repair_plan),
            "advisory_count": len(remaining_advisory_repair_plan),
        }

    overall_status = "ready"
    if operator_followups or hard_failed_attempts:
        overall_status = "blocked"
    elif remaining_repair_plan or degraded_attempts:
        overall_status = "degraded"

    recommended_actions = ordered_unique(
        [
            "install the infrastructure autofix bot on a timer so safe repairs happen before small degradations stack into outages",
            "let the bot run in apply mode for safe fixes, but keep destructive retention or credential changes operator-gated",
            "configure remote alert delivery so degraded states page you instead of silently accumulating" if operator_followups else "",
            "use the bot-quality autopilot alongside the infrastructure autofix bot so system health and bot quality improve together"
            if remaining_repair_plan or remaining_advisory_repair_plan
            else "",
        ]
    )

    return {
        "timestamp_utc": iso_now(),
        "schema_version": 1,
        "ok": overall_status == "ready",
        "overall_status": overall_status,
        "apply": bool(apply),
        "repair_plan": remaining_repair_plan,
        "advisory_repair_plan": remaining_advisory_repair_plan,
        "pre_apply_repair_plan": repair_plan if apply else [],
        "pre_apply_advisory_repair_plan": advisory_repair_plan if apply else [],
        "post_apply_recheck": post_apply_recheck,
        "attempts": [
            {
                "cmd": list(row.get("cmd") or []),
                "rc": int(row.get("rc", 1)),
                "timed_out": bool(row.get("timed_out", False)),
                "timeout_sec": _safe_int(row.get("timeout_sec"), 0),
                "skipped": bool(row.get("skipped", False)),
                "reason": str(row.get("reason") or ""),
            }
            for row in attempts
        ],
        "applyable_repair_count": len(remaining_repair_plan),
        "advisory_repair_count": len(remaining_advisory_repair_plan),
        "operator_followups": operator_followups,
        "metrics": {
            "pre_apply_repair_count": len(repair_plan),
            "pre_apply_advisory_repair_count": len(advisory_repair_plan),
            "post_apply_recheck_enabled": bool(post_apply_recheck.get("enabled", False)),
            "daily_verify_failed_checks": len(failed_checks),
            "retention_debt_gb": retention_debt_gb,
            "auth_expires_in_seconds": _safe_float(((auth_lease.get("lease_budget") or {}).get("expires_in_seconds")), 0.0),
            "snapshot_ready": bool(((snapshot_cache.get("cache_health") or {}).get("snapshot_ready", False))),
            "unsent_critical_alerts": _safe_int(((remote_alert.get("critical_backlog") or {}).get("unsent_count")), 0),
            "storage_total_pending_lines": total_pending_lines,
            "storage_total_drain_minutes": total_drain_minutes,
            "stateful_storage_local_gb": _safe_float(((stateful_storage_guard.get("metrics") or {}).get("local_stateful_gb")), 0.0),
            "storage_reconnect_repair_plan_count": _safe_int(((storage_reconnect.get("metrics") or {}).get("repair_plan_count")), 0),
            "storage_reconnect_missing_contract_count": _safe_int(((storage_reconnect_guard.get("metrics") or {}).get("missing_contract_count")), 0),
            "runtime_paper_failed_guard_count": runtime_paper_failed,
            "stale_surface_repair_count": _safe_int(stale_surface_metrics.get("planned_repair_count"), 0),
            "artifact_freshness_stale_required": _safe_int(freshness_summary.get("stale_required"), 0),
            "artifact_freshness_stale_optional": _safe_int(freshness_summary.get("stale_optional"), 0),
            "process_watchdog_active_issue_count": _safe_int(watchdog_intelligence.get("active_issue_count"), 0),
            "commands_duplicate_entries": _safe_int((((commands_hygiene_payload.get("metrics") or {}).get("duplicate_entry_count"))), 0),
            "commands_runbook_changed": bool(commands_hygiene_payload.get("runbook_changed", False)),
            "commands_blocked_entries": _safe_int((((command_validity_payload.get("metrics") or {}).get("blocked_entry_count"))), 0),
            "timeout_budget_seconds": timeout_budget_seconds,
            "timeout_budget_exhausted": bool(timeout_budget_exhausted),
        },
        "infra_bots": [
            "infrastructure_autofix_bot",
            "commands_hygiene_bot",
            "command_validity_bot",
            "system_drift_guard",
            "system_drift_autopilot",
            "daily_verify_auto_remediation_bot",
            "storage_pressure_clearance_bot",
            "storage_reconnect_infrabot",
            "storage_reconnect_regression_guard",
            "core_bot_materialization_infrabot",
            "core_bot_materialization_guard",
            "core_bot_tier_organizer",
            "storage_backpressure_autopilot",
            "stateful_storage_regression_guard",
            "runtime_paper_regression_guard",
            "host_capability_contract",
            "halt_trigger_control_plane",
            "coordination_state_control",
            "whole_system_governor",
            "library_utilization_router",
            "mlx_intelligence_router",
            "library_upgrade_route_control",
            "stale_surface_autohealer",
            "one_numbers_regression_guard",
            "master_infrastructure_supervisor",
            "premarket_token_guard",
            "bot_quality_autopilot",
            "restart_sanity_bundle",
            "writer_lock_handoff_infrabot",
            "health_truth_reconciler_infrabot",
            "provider_cross_verification_infrabot",
            "paper_feedback_repair_infrabot",
            "promotion_replay_gate_infrabot",
            "bot_data_labeling_targeter_infrabot",
            "recovery_drill_infrabot",
            "self_audit_freshness_infrabot",
            "cotenant_headroom_guard_infrabot",
            "protected_volume_boundary_infrabot",
            "training_batch_readiness_infrabot",
            "paper_execution_queue_reconciler_infrabot",
            "duplicate_alpha_compression_infrabot",
            "livefeed_mirror_continuity_infrabot",
            "auth_lease_preflight_infrabot",
            "market_explanation_evidence_infrabot",
            "runtime_paper_contract_infrabot",
        ],
        "recommended_actions": recommended_actions,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Safely orchestrate automated fixes for degraded infrastructure surfaces.")
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--out-file", default=str(DEFAULT_OUT_PATH))
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--timeout-sec", type=int, default=1200)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    payload = build_payload(Path(args.project_root).resolve(), apply=bool(args.apply), timeout_sec=int(args.timeout_sec))
    out_path = Path(args.out_file).expanduser()
    write_payload(out_path, payload)
    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        print(
            "infrastructure_autofix_bot "
            f"overall_status={payload.get('overall_status', '')} "
            f"repair_plan={int(payload.get('applyable_repair_count', 0) or 0)}"
        )
    return 0 if payload.get("overall_status") in {"ready", "degraded"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
