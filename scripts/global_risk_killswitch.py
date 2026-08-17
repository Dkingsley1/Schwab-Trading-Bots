import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.halt_flags import write_halt_flag_atomic

PY = Path(sys.executable or "python")
RECOVERABLE_HEALTH_GATES = {
    "collector_contracts",
    "ingestion_backpressure_overload",
    "priority_shard_latency",
    "priority_shard_storage",
}
THAW_SAFE_RUNTIME_STATES = {
    "ready",
    "guarded_live_read_only",
    "coverage_cycles_ready",
    "scheduled_off_hours_launch",
    "off_hours_cold_lane_launch_ready",
}


def _load(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except Exception:
        return {}


def _first_non_empty(paths: List[Path]) -> Tuple[dict, str]:
    for p in paths:
        payload = _load(p)
        if payload:
            return payload, str(p)
    return {}, ''


def _clearance_state(payload: dict) -> str:
    clearance_plan = payload.get('clearance_plan')
    if isinstance(clearance_plan, dict):
        return str(clearance_plan.get('clearance_state') or '').strip().lower()
    return ''


def _active_hard_gates(payload: dict[str, Any]) -> list[str]:
    hard_gates = payload.get("hard_gates")
    if not isinstance(hard_gates, dict):
        return []
    return sorted(str(name) for name, active in hard_gates.items() if bool(active))


def _truthy_env(name: str, default: str = "0") -> bool:
    return str(os.getenv(name, default) or "").strip().lower() in {"1", "true", "yes", "on"}


def _execution_expected() -> bool:
    return _truthy_env("ALLOW_ORDER_EXECUTION", "0") and not _truthy_env("MARKET_DATA_ONLY", "1")


def _current_backpressure_is_clear(payload: dict[str, Any]) -> bool:
    if not payload:
        return False
    pending = int(payload.get("pending_lines_total", payload.get("pending_lines", 0)) or 0)
    threshold = int(payload.get("pending_lines_threshold", 15000) or 15000)
    return (
        not bool(payload.get("overload", False))
        and not bool(payload.get("line_pressure", False))
        and not bool(payload.get("file_pressure", False))
        and not bool(payload.get("age_pressure", False))
        and pending < threshold
    )


def _effective_storage_backpressure(storage_control: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(storage_control, dict) or not storage_control:
        return {"authoritative": False}
    backpressure = storage_control.get("backpressure") if isinstance(storage_control.get("backpressure"), dict) else {}
    effective = backpressure.get("effective_raw_live") if isinstance(backpressure.get("effective_raw_live"), dict) else {}
    data_integrity = storage_control.get("data_integrity") if isinstance(storage_control.get("data_integrity"), dict) else {}
    source = str(backpressure.get("effective_raw_live_source") or effective.get("source") or "").strip()
    storage_ready = bool(
        str(storage_control.get("overall_status") or "").strip().lower() == "ready"
        and str(storage_control.get("severity") or "").strip().lower() == "stable"
    )
    overlay_clear = bool(backpressure.get("overlay_pressure_clear", False) or source == "fresh_empty_sql_ingestion_overlay")
    data_clean = bool(
        int(data_integrity.get("sql_overlay_invalid_lines", 0) or 0) <= 0
        and int(data_integrity.get("sql_overlay_oversize_payloads", 0) or 0) <= 0
        and int(data_integrity.get("sql_overlay_ops_write_failures", 0) or 0) <= 0
    )
    authoritative = bool(
        storage_ready
        and bool(backpressure.get("overlay_adjusted", False))
        and overlay_clear
        and data_clean
    )
    total = int(float(effective.get("total_pending_lines", backpressure.get("total_pending_lines", 0)) or 0))
    core = int(float(effective.get("core_pending_lines", backpressure.get("core_pending_lines", total)) or 0))
    oldest = float(effective.get("oldest_pending_age_seconds", backpressure.get("oldest_pending_age_seconds", 0.0)) or 0.0)
    return {
        "authoritative": authoritative,
        "source": source or "ingestion_storage_control_effective_raw_live",
        "core_pending_lines": core,
        "total_pending_lines": total,
        "oldest_pending_age_seconds": round(oldest, 3),
    }


def _watchdog_restart_storm_recovered(payload: dict[str, Any]) -> bool:
    storms = payload.get("restart_storms")
    if not isinstance(storms, list) or not storms:
        return True
    rows = payload.get("status")
    if not isinstance(rows, list):
        return False
    by_name = {str(row.get("name") or "").strip(): row for row in rows if isinstance(row, dict)}
    for storm in storms:
        if not isinstance(storm, dict):
            return False
        if bool(storm.get("resolved", False)):
            continue
        name = str(storm.get("name") or "").strip()
        row = by_name.get(name)
        if not isinstance(row, dict):
            return False
        running_count = int(row.get("running", 0) or 0) + int(row.get("alt_running", 0) or 0)
        process_live = bool(row.get("process_live", False))
        heartbeat_ok = bool(row.get("heartbeat_ok", row.get("heartbeat_fresh", False)))
        if running_count <= 0 or not process_live or not heartbeat_ok:
            return False
    return True


def _watchdog_restart_storm_isolation(
    payload: dict[str, Any],
    *,
    execution_expected: bool,
) -> dict[str, Any]:
    storms = payload.get("restart_storms")
    if not isinstance(storms, list) or not storms:
        return {
            "isolated_count": 0,
            "execution_blocking_count": 0,
            "isolated_targets": [],
            "execution_blocking_targets": [],
            "safe_to_clear_when_not_executing": False,
            "execution_expected": bool(execution_expected),
            "policy": "no_active_restart_storms",
        }

    isolated_targets: list[str] = []
    execution_blocking_targets: list[str] = []
    for storm in storms:
        if not isinstance(storm, dict) or bool(storm.get("resolved", False)):
            continue
        name = str(storm.get("name") or "").strip()
        if not name:
            continue
        quarantinable = bool(storm.get("quarantinable", False))
        blocks_execution_clear = bool(storm.get("blocks_execution_clear", not quarantinable))
        if quarantinable and not blocks_execution_clear and not execution_expected:
            isolated_targets.append(name)
        else:
            execution_blocking_targets.append(name)

    return {
        "isolated_count": len(isolated_targets),
        "execution_blocking_count": len(execution_blocking_targets),
        "isolated_targets": sorted(isolated_targets),
        "execution_blocking_targets": sorted(execution_blocking_targets),
        "safe_to_clear_when_not_executing": bool(isolated_targets and not execution_blocking_targets and not execution_expected),
        "execution_expected": bool(execution_expected),
        "policy": "read_only_collection_restart_storms_may_be_quarantined_only_while_order_execution_is_off",
    }


def _backpressure_pressure_ratio(payload: dict[str, Any]) -> float:
    if not payload:
        return 0.0
    pending = float(payload.get("pending_lines_total", payload.get("pending_lines", 0)) or 0.0)
    threshold = float(payload.get("pending_lines_threshold", 15000) or 15000.0)
    return max(pending, 0.0) / max(threshold, 1.0)


def _classify_hard_gates(
    hard_gate_names: list[str],
    backpressure: dict[str, Any],
    *,
    execution_expected: bool = False,
) -> tuple[list[str], list[str], list[str]]:
    critical: list[str] = []
    degraded: list[str] = []
    stale: list[str] = []
    backpressure_clear = _current_backpressure_is_clear(backpressure)
    backpressure_ratio = _backpressure_pressure_ratio(backpressure)
    severe_backpressure_ratio = float(os.getenv("GLOBAL_KILL_BACKPRESSURE_SEVERE_RATIO", "2.0"))
    live_degrade_to_halt = _truthy_env("GLOBAL_KILL_DEGRADE_TO_HALT_ON_LIVE_EXECUTION", "1")
    recoverable = {
        gate.strip()
        for gate in str(os.getenv("GLOBAL_KILL_RECOVERABLE_HEALTH_GATES", ",".join(sorted(RECOVERABLE_HEALTH_GATES)))).split(",")
        if gate.strip()
    }
    for gate in hard_gate_names:
        if gate == "ingestion_backpressure_overload" and backpressure_clear:
            stale.append(gate)
        elif gate == "ingestion_backpressure_overload" and backpressure_ratio >= severe_backpressure_ratio:
            critical.append(gate)
        elif execution_expected and live_degrade_to_halt and gate in recoverable:
            critical.append(gate)
        elif gate in recoverable:
            degraded.append(gate)
        else:
            critical.append(gate)
    return critical, degraded, stale


def _expansion_pressure_score(
    *,
    degraded_hard_gates: list[str],
    stale_hard_gates: list[str],
    degraded_clear_blockers: list[str],
    backpressure: dict[str, Any],
    quant_pressure: float = 0.0,
) -> float:
    score = min(_backpressure_pressure_ratio(backpressure), 2.0) * 0.35
    score += 0.20 * len(degraded_hard_gates)
    score += 0.08 * len(stale_hard_gates)
    score += 0.15 * len(degraded_clear_blockers)
    score += min(max(float(quant_pressure), 0.0), 1.0) * 0.18
    return round(min(score, 1.0), 4)


def _operating_mode(
    *,
    reasons: list[str],
    clear_blockers: list[str],
    degraded_hard_gates: list[str],
    stale_hard_gates: list[str],
    degraded_clear_blockers: list[str],
    execution_expected: bool,
) -> str:
    if reasons:
        return "global_halt_required"
    if clear_blockers:
        return "clear_blocked"
    if execution_expected and (degraded_hard_gates or degraded_clear_blockers):
        return "live_execution_read_only"
    if degraded_hard_gates or stale_hard_gates or degraded_clear_blockers:
        return "degraded_collection"
    return "normal"


def _halt_posture(
    *,
    halt_latched: bool,
    halt_required: bool,
    clear_ready: bool,
    clear_blockers: list[str],
    degraded_hard_gates: list[str],
    stale_hard_gates: list[str],
    degraded_clear_blockers: list[str],
) -> str:
    if halt_latched and halt_required:
        return "latched_halt_required"
    if halt_latched and clear_ready:
        return "latched_clear_ready"
    if halt_latched:
        return "latched_clear_blocked"
    if halt_required:
        return "unlatched_halt_required"
    if clear_blockers:
        return "unlatched_clear_blocked"
    if degraded_hard_gates or stale_hard_gates or degraded_clear_blockers:
        return "unlatched_degraded_collection"
    return "unlatched_clear"


def _operator_status_line(
    *,
    halt_latched: bool,
    halt_required: bool,
    clear_ready: bool,
    degraded_hard_gates: list[str],
    stale_hard_gates: list[str],
    degraded_clear_blockers: list[str],
) -> str:
    if halt_latched and halt_required:
        return "global halt is latched and still required by hard gates"
    if halt_latched and clear_ready:
        return "global halt is latched but safe-clear gates are ready"
    if halt_latched:
        return "global halt is latched and waiting on safe-clear blockers"
    if halt_required:
        return "global halt flag is clear, but halt pressure remains and would re-trigger under enforcement"
    if degraded_hard_gates or stale_hard_gates or degraded_clear_blockers:
        return "global halt flag is clear; system remains degraded/read-only while soft blockers settle"
    return "global halt is clear"


def _parse_json_output(text: str) -> dict[str, Any]:
    for raw in reversed([line.strip() for line in str(text or "").splitlines() if line.strip()]):
        try:
            payload = json.loads(raw)
        except Exception:
            continue
        if isinstance(payload, dict):
            return payload
    return {}


def _clear_blocker_steps() -> list[tuple[str, list[str]]]:
    ops_root = PROJECT_ROOT / "scripts" / "ops"
    return [
        ("process_watchdog", [str(PY), str(ops_root / "process_watchdog.py"), "--json"]),
        ("auth_lease_manager", [str(PY), str(ops_root / "auth_lease_manager.py"), "--json"]),
        ("data_plane_recovery_controller", [str(PY), str(ops_root / "data_plane_recovery_controller.py"), "--json"]),
        ("live_runtime_separation_control", [str(PY), str(ops_root / "live_runtime_separation_control.py"), "--json"]),
    ]


def _attempt_clear_blockers(*, timeout_sec: int) -> list[dict[str, Any]]:
    attempts: list[dict[str, Any]] = []
    for name, cmd in _clear_blocker_steps():
        try:
            proc = subprocess.run(
                cmd,
                cwd=str(PROJECT_ROOT),
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
        payload = _parse_json_output(stdout)
        attempts.append(
            {
                "name": name,
                "rc": rc,
                "ok": rc == 0 and not timed_out,
                "timed_out": timed_out,
                "payload": payload,
                "stdout_tail": "\n".join(stdout.splitlines()[-6:]),
                "stderr_tail": "\n".join(stderr.splitlines()[-6:]) or ("timeout" if timed_out else ""),
            }
        )
    return attempts


def main() -> int:
    parser = argparse.ArgumentParser(description='Account-level global risk kill-switch.')
    parser.add_argument('--max-blocked-rate', type=float, default=float(os.getenv('GLOBAL_KILL_BLOCKED_RATE_MAX', '0.45')))
    parser.add_argument('--max-abs-pnl-proxy', type=float, default=float(os.getenv('GLOBAL_KILL_ABS_PNL_PROXY_MAX', '0.03')))
    parser.add_argument('--max-stale-windows', type=int, default=int(os.getenv('GLOBAL_KILL_STALE_WINDOWS_MAX', '2')))
    parser.add_argument('--max-watchdog-restarts', type=int, default=int(os.getenv('GLOBAL_KILL_WATCHDOG_RESTARTS_MAX', '5')))
    parser.add_argument('--auto-clear', action='store_true')
    parser.add_argument('--clear-blockers', action='store_true', help='Refresh blocker artifacts before reevaluating safe-clear readiness.')
    parser.add_argument('--json', action='store_true', help='Emit JSON; retained for opsctl wrapper consistency.')
    parser.add_argument('--exit-zero', action='store_true', help='Return 0 after writing the diagnostic payload.')
    parser.add_argument(
        '--clear-blocker-timeout-seconds',
        type=int,
        default=int(os.getenv('GLOBAL_KILL_CLEAR_BLOCKER_STEP_TIMEOUT_SECONDS', '10')),
        help='Maximum seconds for each blocker refresh step.',
    )
    parser.add_argument('--status-only', action='store_true', help='Evaluate and report halt posture without mutating halt flags.')
    args = parser.parse_args()

    blocker_refresh_attempts: list[dict[str, Any]] = []
    if args.clear_blockers:
        blocker_refresh_attempts = _attempt_clear_blockers(timeout_sec=int(args.clear_blocker_timeout_seconds))

    one, one_src = _first_non_empty(
        [
            PROJECT_ROOT / 'governance' / 'health' / 'one_numbers_latest.json',
            PROJECT_ROOT / 'exports' / 'one_numbers' / 'one_numbers_summary.json',
            PROJECT_ROOT / 'exports' / 'one_numbers' / 'latest' / 'one_numbers_summary.json',
        ]
    )
    health_root = PROJECT_ROOT / 'governance' / 'health'
    halt_flag = health_root / "GLOBAL_TRADING_HALT.flag"
    operator_stop_flag = health_root / "OPERATOR_STOP.flag"
    health = _load(health_root / 'health_gates_latest.json')
    hard_gate_names = _active_hard_gates(health)
    backpressure = _load(health_root / "ingestion_backpressure_latest.json")
    storage_control = _load(health_root / "ingestion_storage_control_latest.json")
    effective_backpressure = _effective_storage_backpressure(storage_control)
    if bool(effective_backpressure.get("authoritative", False)):
        backpressure = {
            **backpressure,
            "overload": False,
            "line_pressure": False,
            "file_pressure": False,
            "age_pressure": False,
            "pending_lines": int(effective_backpressure.get("core_pending_lines", 0) or 0),
            "pending_lines_total": int(effective_backpressure.get("total_pending_lines", 0) or 0),
            "oldest_pending_age_seconds": float(effective_backpressure.get("oldest_pending_age_seconds", 0.0) or 0.0),
            "storage_control_override": effective_backpressure,
        }
    auth = _load(health_root / 'auth_lease_manager_latest.json')
    data_plane = _load(health_root / 'data_plane_recovery_controller_latest.json')
    watchdog = _load(health_root / 'process_watchdog_latest.json')
    runtime = _load(health_root / 'live_runtime_separation_control_latest.json')
    quant_model = _load(health_root / "quant_model_control_latest.json")
    execution_expected = _execution_expected()
    live_lane_running = bool((runtime.get("live_plane") or {}).get("live_lane_running", False)) if isinstance(runtime.get("live_plane"), dict) else False
    operator_stop_active = operator_stop_flag.exists()
    halt_latched_before = halt_flag.exists()
    global_halt_payload_before = _load(halt_flag) if halt_latched_before else {}
    operator_stop_payload = _load(operator_stop_flag) if operator_stop_flag.exists() else {}

    blocked_rate = float(one.get('combined_blocked_rate', 0.0) or 0.0)
    pnl_proxy = float(one.get('combined_pnl_proxy', one.get('crypto_pnl_proxy', 0.0) or 0.0) or 0.0)
    stale = int(one.get('decision_stale_windows_4h', one.get('decision_stale_windows', 0) or 0) or 0)
    restarts = int(one.get('watchdog_restarts', 0) or 0)

    reasons = []
    if blocked_rate > args.max_blocked_rate:
        reasons.append(f'blocked_rate>{args.max_blocked_rate}')
    if abs(pnl_proxy) > args.max_abs_pnl_proxy:
        reasons.append(f'abs_pnl_proxy>{args.max_abs_pnl_proxy}')
    if stale > args.max_stale_windows:
        reasons.append(f'stale_windows>{args.max_stale_windows}')
    if restarts > args.max_watchdog_restarts:
        reasons.append(f'watchdog_restarts>{args.max_watchdog_restarts}')
    critical_hard_gates, degraded_hard_gates, stale_hard_gates = _classify_hard_gates(
        hard_gate_names,
        backpressure,
        execution_expected=execution_expected,
    )
    if bool(health.get('hard_gate_triggered', False)) and critical_hard_gates:
        reasons.append('health_hard_gate_triggered')

    auth_state = str(auth.get('lease_state', '') or '').strip().lower()
    write_failures = int(data_plane.get('write_failure_count', 0) or 0)
    snapshot_failures = int(data_plane.get('account_snapshot_failure_count', 0) or 0)
    queue_depth = int(data_plane.get('queue_depth', 0) or 0)
    if bool(effective_backpressure.get("authoritative", False)):
        queue_depth = min(queue_depth, int(effective_backpressure.get("total_pending_lines", queue_depth) or 0))
    restart_storms = len(watchdog.get('restart_storms') or [])
    restart_storm_recovered = _watchdog_restart_storm_recovered(watchdog)
    restart_storm_isolation = _watchdog_restart_storm_isolation(
        watchdog,
        execution_expected=execution_expected,
    )
    clearance_state = _clearance_state(runtime)
    clear_blockers = []
    if operator_stop_active:
        clear_blockers.append('operator_stop_active')
    if auth_state == 'critical':
        clear_blockers.append('auth_lease_critical')
    degraded_clear_blockers = []
    if restart_storms > 0:
        if restart_storm_recovered:
            degraded_clear_blockers.append('restart_storm_recovered_waiting_settle')
        elif bool(restart_storm_isolation.get('safe_to_clear_when_not_executing', False)):
            degraded_clear_blockers.append('restart_storm_isolated_read_only_collection')
        else:
            clear_blockers.append('restart_storm_active')
    if snapshot_failures > 0:
        if execution_expected:
            clear_blockers.append('account_snapshot_recovery_pending')
        else:
            degraded_clear_blockers.append('account_snapshot_recovery_pending')
    if write_failures > 0:
        clear_blockers.append('write_path_recovery_pending')
    if queue_depth >= 10000:
        if _current_backpressure_is_clear(backpressure) and not execution_expected:
            degraded_clear_blockers.append('queue_depth_recovered_waiting_backlog_drain')
        else:
            clear_blockers.append('queue_backpressure_active')
    if clearance_state and clearance_state not in THAW_SAFE_RUNTIME_STATES:
        runtime_blocker = f'runtime_clearance={clearance_state}'
        if execution_expected:
            clear_blockers.append(runtime_blocker)
        else:
            degraded_clear_blockers.append(runtime_blocker)
    quant_features = quant_model.get("features") if isinstance(quant_model.get("features"), dict) else {}
    quant_resource_pressure = float(quant_features.get("quant_model_resource_pressure_norm", 0.0) or 0.0)
    quant_status = str(quant_model.get("overall_status") or "").strip().lower()
    if quant_resource_pressure >= 0.80 or quant_status == "degraded":
        degraded_clear_blockers.append("quant_model_resource_pressure")

    now = datetime.now(timezone.utc).isoformat()
    action = 'none'
    clear_ready = not reasons and not clear_blockers
    expansion_pressure_score = _expansion_pressure_score(
        degraded_hard_gates=degraded_hard_gates,
        stale_hard_gates=stale_hard_gates,
        degraded_clear_blockers=degraded_clear_blockers,
        backpressure=backpressure,
        quant_pressure=quant_resource_pressure,
    )
    operating_mode = _operating_mode(
        reasons=reasons,
        clear_blockers=clear_blockers,
        degraded_hard_gates=degraded_hard_gates,
        stale_hard_gates=stale_hard_gates,
        degraded_clear_blockers=degraded_clear_blockers,
        execution_expected=execution_expected,
    )

    halt_required = bool(reasons)
    if reasons and args.auto_clear:
        action = 'clear_blocked' if halt_latched_before else 'halt_required_unlatched'
    elif reasons and not args.status_only:
        halt_flag.parent.mkdir(parents=True, exist_ok=True)
        write_halt_flag_atomic(
            halt_flag,
            {'timestamp_utc': now, 'reason': 'global_risk_killswitch', 'details': reasons},
            project_root=str(PROJECT_ROOT),
            source='global_risk_killswitch',
        )
        action = 'halt_set'
    elif reasons and args.status_only:
        action = 'halt_would_set'
    elif args.auto_clear and halt_flag.exists():
        if clear_ready:
            if args.status_only:
                action = 'halt_would_clear'
            else:
                halt_flag.unlink()
                action = 'halt_cleared'
        else:
            action = 'clear_blocked'

    halt_latched_after = halt_flag.exists()
    global_halt_payload_after = _load(halt_flag) if halt_latched_after else {}
    legacy_halt_state = 'active' if halt_latched_after else 'clear_ready' if clear_ready else 'clear_blocked'
    halt_posture = _halt_posture(
        halt_latched=halt_latched_after,
        halt_required=halt_required,
        clear_ready=clear_ready,
        clear_blockers=clear_blockers,
        degraded_hard_gates=degraded_hard_gates,
        stale_hard_gates=stale_hard_gates,
        degraded_clear_blockers=degraded_clear_blockers,
    )
    status_line = _operator_status_line(
        halt_latched=halt_latched_after,
        halt_required=halt_required,
        clear_ready=clear_ready,
        degraded_hard_gates=degraded_hard_gates,
        stale_hard_gates=stale_hard_gates,
        degraded_clear_blockers=degraded_clear_blockers,
    )

    payload = {
        'timestamp_utc': now,
        'action': action,
        'halt': halt_latched_after,
        'halt_state': legacy_halt_state,
        'halt_posture': halt_posture,
        'halt_latched': halt_latched_after,
        'halt_latched_before': halt_latched_before,
        'halt_required': halt_required,
        'would_rehalt': bool(halt_required and not halt_latched_after),
        'status_line': status_line,
        'clear_ready': clear_ready,
        'clear_blockers': clear_blockers,
        'clear_blocker_refresh_attempts': blocker_refresh_attempts,
        'operator_stop': operator_stop_active,
        'operator_stop_payload': operator_stop_payload,
        'global_halt_payload': global_halt_payload_after,
        'previous_global_halt_payload': global_halt_payload_before,
        'hard_gate_names': hard_gate_names,
        'critical_hard_gate_names': critical_hard_gates,
        'degraded_hard_gate_names': degraded_hard_gates,
        'stale_hard_gate_names': stale_hard_gates,
        'degraded_clear_blockers': degraded_clear_blockers,
        'halt_pressure': {
            'required': halt_required,
            'reasons': reasons,
            'critical_hard_gates': critical_hard_gates,
            'degraded_hard_gates': degraded_hard_gates,
            'stale_hard_gates': stale_hard_gates,
        },
        'safe_clear': {
            'ready': clear_ready,
            'hard_blockers': clear_blockers,
            'degraded_blockers': degraded_clear_blockers,
            'operator_stop': operator_stop_active,
        },
        'operating_mode': operating_mode,
        'expansion_pressure_score': expansion_pressure_score,
        'sleeve_throttle_recommended': bool(degraded_hard_gates or stale_hard_gates or degraded_clear_blockers),
        'read_only_commands': [
            ['./scripts/ops/opsctl.sh', 'feed', '--source', 'all', '--include-decisions'],
            ['./scripts/ops/opsctl.sh', 'global-halt-status', '--json'],
        ],
        'control_commands': {
            'status': ['./scripts/ops/opsctl.sh', 'global-halt-status', '--json'],
            'refresh_blockers': ['./scripts/ops/opsctl.sh', 'global-halt-refresh', '--json'],
            'safe_auto_clear': ['./scripts/ops/opsctl.sh', 'global-halt-auto-clear', '--json'],
            'manual_clear_all_halts': ['./scripts/ops/opsctl.sh', 'clear-all-halts', '--json'],
            'operator_release': ['./scripts/ops/opsctl.sh', 'operator-release', '--json'],
        },
        'source_files': {
            'one_numbers': one_src,
            'health_gates': str(health_root / 'health_gates_latest.json'),
            'auth_lease_manager': str(health_root / 'auth_lease_manager_latest.json'),
            'data_plane_recovery_controller': str(health_root / 'data_plane_recovery_controller_latest.json'),
            'process_watchdog': str(health_root / 'process_watchdog_latest.json'),
            'live_runtime_separation_control': str(health_root / 'live_runtime_separation_control_latest.json'),
            'quant_model_control': str(health_root / 'quant_model_control_latest.json'),
            'operator_stop_flag': str(operator_stop_flag),
        },
        'reasons': reasons,
        'metrics': {
            'blocked_rate': blocked_rate,
            'pnl_proxy': pnl_proxy,
            'stale_windows': stale,
            'watchdog_restarts': restarts,
            'restart_storms': restart_storms,
            'restart_storm_recovered': restart_storm_recovered,
            'restart_storm_isolation': restart_storm_isolation,
            'operator_stop_active': operator_stop_active,
            'auth_state': auth_state,
            'write_failure_count': write_failures,
            'account_snapshot_failure_count': snapshot_failures,
            'queue_depth': queue_depth,
            'raw_queue_depth': int(data_plane.get('raw_queue_depth', queue_depth) or 0),
            'queue_depth_source': str(data_plane.get('queue_depth_source') or ""),
            'storage_backpressure_override': effective_backpressure,
            'runtime_clearance_state': clearance_state,
            'execution_expected': execution_expected,
            'live_lane_running': live_lane_running,
            'current_backpressure_clear': _current_backpressure_is_clear(backpressure),
            'backpressure_pressure_ratio': round(_backpressure_pressure_ratio(backpressure), 4),
            'quant_model_status': quant_status,
            'quant_model_resource_pressure': round(quant_resource_pressure, 4),
        },
        'recommended_actions': [
            action_text
            for action_text in [
                'keep GLOBAL_TRADING_HALT engaged until write-path recovery pressure is clear' if write_failures > 0 else '',
                'keep live execution read-only until account snapshot recovery pressure is clear' if snapshot_failures > 0 and execution_expected else '',
                'run expansion pressure in degraded/throttled collection mode while recoverable health gates clear' if degraded_hard_gates or stale_hard_gates or degraded_clear_blockers else '',
                'reduce sleeve fanout or collector cadence until expansion pressure score falls below 0.35' if expansion_pressure_score >= 0.35 and not reasons else '',
                'run quant-model-control and memory-efficiency before clearing if quant resource pressure is elevated' if quant_resource_pressure >= 0.80 else '',
                'allow halt clear while recovered restart storms settle; keep watching process heartbeats' if restart_storms > 0 and restart_storm_recovered else '',
                'keep isolated read-only restart storms quarantined; do not widen or enable live execution until they settle' if bool(restart_storm_isolation.get('safe_to_clear_when_not_executing', False)) else '',
                'do not clear the halt while execution-impacting restart storms are still active' if restart_storms > 0 and not restart_storm_recovered and not bool(restart_storm_isolation.get('safe_to_clear_when_not_executing', False)) else '',
                'release OPERATOR_STOP before attempting a safe global halt clear' if operator_stop_active else '',
                'refresh broker auth before clearing if auth lease is critical' if auth_state == 'critical' else '',
                'wait for runtime clearance to return to a thaw-safe state before clearing the halt' if clearance_state and clearance_state not in THAW_SAFE_RUNTIME_STATES and execution_expected else '',
            ]
            if action_text
        ],
    }
    recommended_commands: list[list[str]] = []
    def add_command(command: list[str]) -> None:
        if command not in recommended_commands:
            recommended_commands.append(command)

    if operator_stop_active:
        add_command(['./scripts/ops/opsctl.sh', 'operator-release', '--json'])
    if auth_state in {'warning', 'critical'}:
        add_command(['./scripts/ops/opsctl.sh', 'token-refresh', '--json'])
    if write_failures > 0 or queue_depth >= 10000:
        add_command(['./scripts/ops/opsctl.sh', 'external-backlog-drain', '--json'])
    if snapshot_failures > 0:
        add_command(['./scripts/ops/opsctl.sh', 'token-refresh', '--json'])
    if hard_gate_names:
        add_command(['./scripts/ops/opsctl.sh', 'ingestion-storage-control', '--json'])
        add_command(['./scripts/ops/opsctl.sh', 'collector-contracts', '--json'])
    if quant_resource_pressure >= 0.35 or quant_status in {'watch', 'degraded', 'needs_data'}:
        add_command(['./scripts/ops/opsctl.sh', 'quant-model-control', '--json'])
    if quant_resource_pressure >= 0.65:
        add_command(['./scripts/ops/opsctl.sh', 'memory-efficiency', '--apply', '--json'])
    add_command(['./scripts/ops/opsctl.sh', 'global-halt-auto-clear', '--json'])
    payload['recommended_commands'] = recommended_commands

    # Observability side effects: write latest snapshot + append event stream.
    # These are evidence artifacts and must never prevent the command from emitting JSON.
    io_errors: list[str] = []
    out = PROJECT_ROOT / 'governance' / 'health' / 'global_killswitch_latest.json'
    try:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding='utf-8')
    except (OSError, PermissionError) as e:
        io_errors.append(f"write_latest_failed:{out}:{type(e).__name__}:{e}")

    events = PROJECT_ROOT / 'governance' / 'watchdog' / 'global_killswitch_events.jsonl'
    try:
        events.parent.mkdir(parents=True, exist_ok=True)
        with events.open('a', encoding='utf-8') as f:
            f.write(json.dumps(payload, ensure_ascii=True) + '\n')
    except (OSError, PermissionError) as e:
        io_errors.append(f"append_events_failed:{events}:{type(e).__name__}:{e}")

    if io_errors:
        payload['io_errors'] = io_errors

    print(json.dumps(payload, ensure_ascii=True))
    if args.exit_zero:
        return 0
    if reasons:
        return 2
    if (args.auto_clear or args.clear_blockers) and not clear_ready:
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
