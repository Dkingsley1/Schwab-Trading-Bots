#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable

if __package__ in {None, ""}:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from scripts.ops.long_runtime_common import PROJECT_ROOT, iso_now, ordered_unique, write_payload
else:
    from .long_runtime_common import PROJECT_ROOT, iso_now, ordered_unique, write_payload


DEFAULT_OUT_PATH = PROJECT_ROOT / "governance" / "health" / "system_intense_drill_autopilot_latest.json"
DEFAULT_RESULTS_PATH = PROJECT_ROOT / "governance" / "drills" / "system_intense_drill_results_latest.json"
DEFAULT_IMPROVEMENT_PATH = PROJECT_ROOT / "governance" / "drills" / "system_intense_drill_improvement_plan_latest.json"

Runner = Callable[[list[str], Path, int], dict[str, Any]]

READY_STATUSES = {"ready", "ok", "advisory", "guarded_ready", "stable", "clear_ready"}
BAD_STATUSES = {"blocked", "critical", "failed", "fatal", "apply_failed", "missing", "needs_work", "degraded"}
BLOCKED_COMMAND_PATTERNS = ("start-live", "clear-all-halts", "operator-release", "token-refresh-interactive")


def _as_dict(raw: Any) -> dict[str, Any]:
    return raw if isinstance(raw, dict) else {}


def _as_list(raw: Any) -> list[Any]:
    if isinstance(raw, tuple):
        return list(raw)
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


def _parse_json_output(text: str) -> dict[str, Any]:
    for raw in reversed([line.strip() for line in str(text or "").splitlines() if line.strip()]):
        try:
            payload = json.loads(raw)
        except Exception:
            continue
        if isinstance(payload, dict):
            return payload
    return {}


def _tail_text(text: str, *, max_lines: int = 12, max_chars: int = 3500) -> str:
    tail = "\n".join(str(text or "").splitlines()[-max_lines:])
    if len(tail) <= max_chars:
        return tail
    return "...truncated...\n" + tail[-max_chars:]


def _run(cmd: list[str], project_root: Path, timeout_sec: int) -> dict[str, Any]:
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(project_root),
            capture_output=True,
            text=True,
            check=False,
            timeout=max(int(timeout_sec), 1),
        )
        return {
            "cmd": list(cmd),
            "rc": int(proc.returncode),
            "payload": _parse_json_output(proc.stdout or ""),
            "stdout_tail": _tail_text(proc.stdout or ""),
            "stderr_tail": _tail_text(proc.stderr or ""),
        }
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout.decode("utf-8", errors="ignore") if isinstance(exc.stdout, bytes) else str(exc.stdout or "")
        stderr = exc.stderr.decode("utf-8", errors="ignore") if isinstance(exc.stderr, bytes) else str(exc.stderr or "")
        return {
            "cmd": list(cmd),
            "rc": 124,
            "payload": _parse_json_output(stdout),
            "stdout_tail": _tail_text(stdout),
            "stderr_tail": _tail_text(stderr) or "timeout",
        }


def _py(project_root: Path) -> str:
    return str(Path(sys.executable).resolve()) if sys.executable else str(project_root / ".venv314" / "bin" / "python")


def _drill_specs(project_root: Path, *, replay_hours: int, queue_hours: int, min_replay_rows: int) -> list[dict[str, Any]]:
    python = _py(project_root)
    scripts = project_root / "scripts"
    ops = scripts / "ops"
    return [
        {
            "drill_id": "architecture_autopilot",
            "family": "architecture",
            "title": "Architecture Autopilot Drill",
            "cmd": [python, str(ops / "system_architecture_autopilot.py"), "--apply", "--json"],
            "timeout_sec": 120,
            "intensity": "high",
        },
        {
            "drill_id": "fast_health_gate",
            "family": "readiness",
            "title": "Fast Health Gate Drill",
            "cmd": [python, str(ops / "health_fast.py"), "--json"],
            "timeout_sec": 60,
            "intensity": "high",
        },
        {
            "drill_id": "runtime_pressure_gate",
            "family": "runtime",
            "title": "Runtime Pressure Drill",
            "cmd": [python, str(ops / "runtime_throttle_control.py"), "--apply", "--json"],
            "timeout_sec": 120,
            "intensity": "high",
        },
        {
            "drill_id": "execution_queue_stress",
            "family": "queue",
            "title": "Execution Queue Stress Drill",
            "cmd": [
                python,
                str(scripts / "execution_queue_stress_bot.py"),
                "--hours",
                str(max(int(queue_hours), 1)),
                "--max-queue-depth",
                "750",
                "--max-queue-breach-rate",
                "0.10",
                "--json",
            ],
            "timeout_sec": 120,
            "intensity": "high",
        },
        {
            "drill_id": "paper_replay_integrity",
            "family": "replay",
            "title": "Paper Replay Integrity Drill",
            "cmd": [
                python,
                str(scripts / "paper_replay_drill.py"),
                "--hours",
                str(max(int(replay_hours), 1)),
                "--min-rows",
                str(max(int(min_replay_rows), 1)),
                "--json",
            ],
            "timeout_sec": 240,
            "intensity": "high",
        },
        {
            "drill_id": "nightly_resilience_strict",
            "family": "resilience",
            "title": "Strict Resilience Drill",
            "cmd": [python, str(scripts / "nightly_resilience_check.py"), "--max-log-age-minutes", "10", "--json"],
            "timeout_sec": 90,
            "intensity": "high",
        },
        {
            "drill_id": "storage_resilience_fast",
            "family": "storage",
            "title": "Fast Storage Resilience Drill",
            "cmd": [python, str(ops / "storage_resilience_control.py"), "--fast", "--json"],
            "timeout_sec": 180,
            "intensity": "high",
        },
        {
            "drill_id": "chaos_drill_cadence",
            "family": "chaos",
            "title": "Chaos Drill Cadence Drill",
            "cmd": [python, str(ops / "chaos_drill_coordinator.py"), "--overdue-days", "3", "--json"],
            "timeout_sec": 90,
            "intensity": "high",
        },
        {
            "drill_id": "golden_replay_regression",
            "family": "replay",
            "title": "Golden Replay Regression Drill",
            "cmd": [python, str(scripts / "golden_replay_regression_guard.py"), "--json"],
            "timeout_sec": 180,
            "intensity": "high",
        },
    ]


def _payload_status(payload: dict[str, Any], rc: int) -> str:
    raw = str(payload.get("overall_status") or payload.get("status") or payload.get("state") or "").strip().lower()
    if not raw and "ok" in payload:
        raw = "ready" if bool(payload.get("ok", False)) else "blocked"
    if int(rc) == 124:
        return "timeout"
    if int(rc) != 0 and raw in READY_STATUSES:
        return "degraded"
    if raw:
        return raw
    return "ready" if int(rc) == 0 else "blocked"


def _drill_ready(payload: dict[str, Any], rc: int) -> bool:
    status = _payload_status(payload, rc)
    if int(rc) != 0:
        return False
    if "ok" in payload and not bool(payload.get("ok", False)):
        return False
    return status in READY_STATUSES


def _append_improvement(plan: list[dict[str, Any]], *, deficiency_id: str, reason: str, cmd: list[str], priority: int) -> None:
    plan.append(
        {
            "deficiency_id": deficiency_id,
            "reason": reason,
            "priority": int(priority),
            "cmd": [str(part) for part in cmd],
            "timeout_sec": 300,
        }
    )


def _architecture_benefit_improvements(deficiency_id: str, payload: dict[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    backlog = _as_dict(payload.get("architecture_benefit_backlog"))
    for candidate in _as_list(backlog.get("active_candidates"))[:3]:
        if not isinstance(candidate, dict):
            continue
        for command in _as_list(candidate.get("safe_commands"))[:2]:
            if isinstance(command, list) and command:
                out.append(
                    {
                        "deficiency_id": deficiency_id,
                        "reason": f"architecture_candidate:{candidate.get('candidate_id')}",
                        "priority": 20 - _safe_int(candidate.get("score"), 0),
                        "cmd": [str(part) for part in command],
                        "timeout_sec": 300,
                    }
                )
    return out


def _deficiency_from_result(result: dict[str, Any]) -> dict[str, Any] | None:
    drill_id = str(result.get("drill_id") or "")
    family = str(result.get("family") or "")
    payload = _as_dict(result.get("payload"))
    status = str(result.get("status") or "")
    rc = _safe_int(result.get("rc"), 1)
    if bool(result.get("ready", False)):
        return None

    reasons: list[str] = []
    commands: list[dict[str, Any]] = []
    severity = "high" if status in {"blocked", "critical", "timeout", "apply_failed"} or rc == 124 else "medium"
    if rc != 0:
        reasons.append(f"rc={rc}")
    if status:
        reasons.append(f"status={status}")

    if drill_id == "architecture_autopilot":
        summary = _as_dict(payload.get("architecture_benefit_summary"))
        if summary.get("top_candidate_id"):
            reasons.append(f"top_architecture={summary.get('top_candidate_id')}")
        commands.extend(_architecture_benefit_improvements(drill_id, payload))
        _append_improvement(commands, deficiency_id=drill_id, reason="refresh_architecture_plan", cmd=["./scripts/ops/opsctl.sh", "system-architecture-autopilot", "--apply", "--json"], priority=8)
    elif drill_id == "fast_health_gate":
        guarded = _as_dict(_as_dict(payload.get("operational_readiness")).get("guarded_paper"))
        blockers = {str(item) for item in _as_list(guarded.get("blockers"))}
        for blocker in _as_list(guarded.get("blockers")):
            reasons.append(str(blocker))
        if str(guarded.get("status") or "") == "blocked" or _as_list(guarded.get("blockers")):
            severity = "high"
        if "runtime_status=degraded" in blockers:
            _append_improvement(commands, deficiency_id=drill_id, reason="guarded_paper_blocked_by_runtime", cmd=["./scripts/ops/opsctl.sh", "runtime-throttle", "--apply", "--max-renice-processes", "30", "--json"], priority=1)
            _append_improvement(commands, deficiency_id=drill_id, reason="verify_runtime_paper_contract", cmd=["./scripts/ops/opsctl.sh", "runtime-paper-regression-guard", "--json"], priority=3)
        if "paper_ramp_not_armed" in blockers:
            _append_improvement(commands, deficiency_id=drill_id, reason="rearm_guarded_paper_ramp", cmd=["./scripts/ops/opsctl.sh", "paper-400-ramp", "--apply", "--json"], priority=2)
        if any(item.startswith("global_clear_blocker=queue_backpressure_active") or item == "storage_pressure_index_high" for item in blockers):
            _append_improvement(commands, deficiency_id=drill_id, reason="bounded_storage_backpressure_for_paper_gate", cmd=["./scripts/ops/opsctl.sh", "storage-backpressure-autopilot", "--apply", "--quick-bounded", "--json"], priority=2)
            _append_improvement(commands, deficiency_id=drill_id, reason="refresh_global_halt_blockers", cmd=["./scripts/ops/opsctl.sh", "global-halt-refresh", "--json"], priority=4)
    elif drill_id == "runtime_pressure_gate":
        _append_improvement(commands, deficiency_id=drill_id, reason="runtime_pressure_not_clear", cmd=["./scripts/ops/opsctl.sh", "runtime-throttle", "--apply", "--max-renice-processes", "30", "--json"], priority=1)
        _append_improvement(commands, deficiency_id=drill_id, reason="runtime_paper_regression_check", cmd=["./scripts/ops/opsctl.sh", "runtime-paper-regression-guard", "--json"], priority=3)
    elif drill_id == "execution_queue_stress":
        reasons.append(f"queue_breach_rate={payload.get('queue_breach_rate')}")
        _append_improvement(commands, deficiency_id=drill_id, reason="queue_stress_breach", cmd=["./scripts/ops/opsctl.sh", "writer-cycle-coordinator", "--apply", "--json"], priority=4)
        _append_improvement(commands, deficiency_id=drill_id, reason="bounded_storage_backpressure", cmd=["./scripts/ops/opsctl.sh", "storage-backpressure-autopilot", "--apply", "--quick-bounded", "--json"], priority=5)
    elif family == "replay":
        _append_improvement(commands, deficiency_id=drill_id, reason="refresh_replay_hash_registry", cmd=["./scripts/ops/opsctl.sh", "replay-hash-registry", "--json"], priority=6)
        _append_improvement(commands, deficiency_id=drill_id, reason="rerun_golden_replay_guard", cmd=["./scripts/ops/opsctl.sh", "golden-replay-regression", "--json"], priority=7)
    elif drill_id == "nightly_resilience_strict":
        for failed in _as_list(payload.get("failed_checks")):
            reasons.append(str(failed))
        _append_improvement(commands, deficiency_id=drill_id, reason="refresh_process_watchdog", cmd=["./scripts/ops/opsctl.sh", "process-watchdog", "--json"], priority=6)
        _append_improvement(commands, deficiency_id=drill_id, reason="refresh_chaos_drill_cadence", cmd=["./scripts/ops/opsctl.sh", "chaos-drills", "--json"], priority=8)
    elif drill_id == "storage_resilience_fast":
        reasons.append(f"resilience_score={payload.get('resilience_score')}")
        _append_improvement(commands, deficiency_id=drill_id, reason="refresh_storage_resilience", cmd=["./scripts/ops/opsctl.sh", "storage-resilience", "--fast", "--json"], priority=6)
        _append_improvement(commands, deficiency_id=drill_id, reason="bounded_storage_backpressure", cmd=["./scripts/ops/opsctl.sh", "storage-backpressure-autopilot", "--apply", "--quick-bounded", "--json"], priority=7)
    elif drill_id == "chaos_drill_cadence":
        overdue = [str(row.get("drill") or "") for row in _as_list(payload.get("overdue_drills")) if isinstance(row, dict)]
        reasons.extend([f"overdue:{item}" for item in overdue])
        _append_improvement(commands, deficiency_id=drill_id, reason="refresh_chaos_drill_cadence", cmd=["./scripts/ops/opsctl.sh", "chaos-drills", "--json"], priority=8)

    if not reasons:
        reasons.append("drill_not_ready")

    return {
        "deficiency_id": drill_id,
        "family": family,
        "severity": severity,
        "status": status,
        "reasons": ordered_unique(reasons),
        "safe_improvement_commands": commands,
    }


def _command_key(cmd: list[str]) -> str:
    return " ".join(str(part) for part in cmd)


def _safe_command(cmd: list[str]) -> bool:
    joined = _command_key(cmd)
    if any(pattern in joined for pattern in BLOCKED_COMMAND_PATTERNS):
        return False
    if not cmd:
        return False
    if cmd[0] != "./scripts/ops/opsctl.sh":
        return False
    return True


def _improvement_plan(deficiencies: list[dict[str, Any]], *, max_steps: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    seen: set[str] = set()
    for deficiency in deficiencies:
        for raw in _as_list(deficiency.get("safe_improvement_commands")):
            if not isinstance(raw, dict):
                continue
            cmd = [str(part) for part in _as_list(raw.get("cmd"))]
            key = _command_key(cmd)
            step = {
                "deficiency_id": str(raw.get("deficiency_id") or deficiency.get("deficiency_id") or ""),
                "reason": str(raw.get("reason") or ""),
                "priority": _safe_int(raw.get("priority"), 50),
                "cmd": cmd,
                "timeout_sec": _safe_int(raw.get("timeout_sec"), 300),
            }
            if key in seen:
                continue
            seen.add(key)
            if not _safe_command(cmd):
                skipped.append({**step, "skip_reason": "not_safe_for_auto_improvement"})
                continue
            rows.append(step)
    ranked = sorted(rows, key=lambda row: (_safe_int(row.get("priority"), 50), str(row.get("deficiency_id") or ""), _command_key(_as_list(row.get("cmd")))))
    return ranked[: max(int(max_steps), 1)], skipped


def _compact_payload(payload: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "overall_status",
        "ok",
        "strict_all_clear",
        "repair_step_count",
        "safe_repair_step_count",
        "attempt_count",
        "blocked_node_count",
        "degraded_node_count",
        "queue_breach_rate",
        "samples",
        "resilience_score",
        "overdue_drills",
        "architecture_benefit_summary",
    )
    return {key: payload.get(key) for key in keys if key in payload}


def _postcheck(project_root: Path, runner: Runner, timeout_sec: int) -> dict[str, Any]:
    checks = [
        ("health_fast", ["./scripts/ops/opsctl.sh", "health-fast", "--json"]),
        ("runtime_throttle", ["./scripts/ops/opsctl.sh", "runtime-throttle", "--apply", "--json"]),
        ("architecture_autopilot", ["./scripts/ops/opsctl.sh", "system-architecture-autopilot", "--apply", "--json"]),
    ]
    out: dict[str, Any] = {}
    for name, cmd in checks:
        result = runner(cmd, project_root, timeout_sec)
        payload = _as_dict(result.get("payload"))
        out[name] = {
            "rc": _safe_int(result.get("rc"), 1),
            "overall_status": str(payload.get("overall_status") or payload.get("status") or ""),
            "ok": bool(payload.get("ok", False)),
            "summary": _compact_payload(payload),
        }
    return out


def _attempt_failed(row: dict[str, Any]) -> bool:
    if _safe_int(row.get("rc"), 1) == 124:
        return True
    summary = _as_dict(row.get("payload_summary"))
    status = str(summary.get("overall_status") or summary.get("status") or "").strip().lower()
    return status in {"apply_failed", "failed", "fatal"}


def build_payload(
    project_root: Path = PROJECT_ROOT,
    *,
    apply: bool = False,
    execute_safe_improvements: bool = False,
    max_improvements: int = 8,
    drill_timeout_sec: int = 240,
    improvement_timeout_sec: int = 300,
    replay_hours: int = 48,
    queue_hours: int = 8,
    min_replay_rows: int = 50,
    runner: Runner | None = None,
    drill_specs: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    project_root = project_root.resolve()
    run_step = runner or _run
    specs = drill_specs or _drill_specs(project_root, replay_hours=replay_hours, queue_hours=queue_hours, min_replay_rows=min_replay_rows)

    drill_results: list[dict[str, Any]] = []
    for spec in specs:
        timeout_sec = min(_safe_int(spec.get("timeout_sec"), drill_timeout_sec), max(int(drill_timeout_sec), 1))
        result = run_step([str(part) for part in _as_list(spec.get("cmd"))], project_root, timeout_sec)
        payload = _as_dict(result.get("payload"))
        rc = _safe_int(result.get("rc"), 1)
        status = _payload_status(payload, rc)
        drill_results.append(
            {
                "drill_id": str(spec.get("drill_id") or ""),
                "family": str(spec.get("family") or ""),
                "title": str(spec.get("title") or ""),
                "intensity": str(spec.get("intensity") or "high"),
                "cmd": list(result.get("cmd") or []),
                "rc": rc,
                "status": status,
                "ready": _drill_ready(payload, rc),
                "timeout_sec": timeout_sec,
                "payload_summary": _compact_payload(payload),
                "payload": payload,
                "stdout_tail": str(result.get("stdout_tail") or ""),
                "stderr_tail": str(result.get("stderr_tail") or ""),
            }
        )

    deficiencies = [row for row in (_deficiency_from_result(result) for result in drill_results) if row]
    improvement_plan, skipped_improvements = _improvement_plan(deficiencies, max_steps=max_improvements)

    attempts: list[dict[str, Any]] = []
    if apply and execute_safe_improvements:
        for step in improvement_plan:
            timeout_sec = min(_safe_int(step.get("timeout_sec"), improvement_timeout_sec), max(int(improvement_timeout_sec), 1))
            result = run_step(list(step["cmd"]), project_root, timeout_sec)
            payload = _as_dict(result.get("payload"))
            step_kind = "mutating_repair" if "--apply" in {str(part) for part in _as_list(step.get("cmd"))} else "diagnostic_followup"
            attempts.append(
                {
                    "deficiency_id": str(step.get("deficiency_id") or ""),
                    "reason": str(step.get("reason") or ""),
                    "step_kind": step_kind,
                    "cmd": list(result.get("cmd") or []),
                    "rc": _safe_int(result.get("rc"), 1),
                    "timeout_sec": timeout_sec,
                    "payload_summary": _compact_payload(payload),
                    "stdout_tail": str(result.get("stdout_tail") or ""),
                    "stderr_tail": str(result.get("stderr_tail") or ""),
                }
            )

    postcheck = _postcheck(project_root, run_step, min(max(int(improvement_timeout_sec), 1), 180)) if attempts else {}
    failed_attempt_count = sum(
        1
        for row in attempts
        if str(row.get("step_kind") or "") == "mutating_repair" and _attempt_failed(row)
    )
    unresolved_mutating_count = sum(
        1
        for row in attempts
        if str(row.get("step_kind") or "") == "mutating_repair"
        and _safe_int(row.get("rc"), 1) != 0
        and not _attempt_failed(row)
    )
    unresolved_diagnostic_count = sum(
        1
        for row in attempts
        if str(row.get("step_kind") or "") == "diagnostic_followup" and _safe_int(row.get("rc"), 1) != 0
    )
    critical_deficiency_count = sum(1 for row in deficiencies if str(row.get("severity") or "") == "high")
    ready_drill_count = sum(1 for row in drill_results if bool(row.get("ready", False)))
    overall_status = "ready"
    if failed_attempt_count:
        overall_status = "blocked"
    elif critical_deficiency_count:
        overall_status = "blocked"
    elif deficiencies:
        overall_status = "degraded"

    payload = {
        "timestamp_utc": iso_now(),
        "schema_version": 1,
        "ok": overall_status == "ready",
        "overall_status": overall_status,
        "apply": bool(apply),
        "execute_safe_improvements": bool(execute_safe_improvements),
        "drill_count": len(drill_results),
        "ready_drill_count": ready_drill_count,
        "deficiency_count": len(deficiencies),
        "critical_deficiency_count": critical_deficiency_count,
        "improvement_step_count": len(improvement_plan),
        "skipped_improvement_count": len(skipped_improvements),
        "attempt_count": len(attempts),
        "failed_attempt_count": failed_attempt_count,
        "unresolved_mutating_count": unresolved_mutating_count,
        "unresolved_diagnostic_count": unresolved_diagnostic_count,
        "drill_results": drill_results,
        "deficiencies": deficiencies,
        "improvement_plan": improvement_plan,
        "skipped_improvements": skipped_improvements,
        "attempts": attempts,
        "postcheck": postcheck,
        "intense_drill_contract": {
            "generation": "system_intense_drill_autopilot_v1",
            "drills_before_improvements": True,
            "improvements_require_explicit_execute_flag": True,
            "does_not_enable_live_execution": True,
            "does_not_release_halts": True,
            "paper_runtime_storage_governance_replay_covered": True,
        },
        "recommended_commands": [
            ["./scripts/ops/opsctl.sh", "system-intense-drills", "--apply", "--json"],
            ["./scripts/ops/opsctl.sh", "system-intense-drills", "--apply", "--execute-safe-improvements", "--json"],
        ],
        "recommended_actions": ordered_unique(
            [
                "review deficiencies before executing improvements" if deficiencies and not execute_safe_improvements else "",
                "execute safe improvements only when the machine can tolerate bounded repair work" if improvement_plan and not execute_safe_improvements else "",
                "rerun drills after improvements to confirm deficiencies actually cleared" if attempts else "",
            ]
            + [f"{row['deficiency_id']}: {', '.join(_as_list(row.get('reasons'))[:3])}" for row in deficiencies[:8]]
        ),
    }

    if apply:
        results_path = DEFAULT_RESULTS_PATH if project_root == PROJECT_ROOT else project_root / "governance" / "drills" / "system_intense_drill_results_latest.json"
        improvement_path = DEFAULT_IMPROVEMENT_PATH if project_root == PROJECT_ROOT else project_root / "governance" / "drills" / "system_intense_drill_improvement_plan_latest.json"
        write_payload(results_path, {"timestamp_utc": payload["timestamp_utc"], "drill_results": drill_results, "deficiencies": deficiencies})
        write_payload(improvement_path, {"timestamp_utc": payload["timestamp_utc"], "improvement_plan": improvement_plan, "skipped_improvements": skipped_improvements, "attempts": attempts})
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Run bounded intense system drills and optionally execute safe deficiency improvements.")
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--out-file", default=str(DEFAULT_OUT_PATH))
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--execute-safe-improvements", action="store_true")
    parser.add_argument("--max-improvements", type=int, default=8)
    parser.add_argument("--drill-timeout-seconds", type=int, default=240)
    parser.add_argument("--improvement-timeout-seconds", type=int, default=300)
    parser.add_argument("--replay-hours", type=int, default=48)
    parser.add_argument("--queue-hours", type=int, default=8)
    parser.add_argument("--min-replay-rows", type=int, default=50)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    project_root = Path(args.project_root).expanduser().resolve()
    payload = build_payload(
        project_root,
        apply=bool(args.apply),
        execute_safe_improvements=bool(args.execute_safe_improvements),
        max_improvements=max(int(args.max_improvements), 1),
        drill_timeout_sec=max(int(args.drill_timeout_seconds), 1),
        improvement_timeout_sec=max(int(args.improvement_timeout_seconds), 1),
        replay_hours=max(int(args.replay_hours), 1),
        queue_hours=max(int(args.queue_hours), 1),
        min_replay_rows=max(int(args.min_replay_rows), 1),
    )
    write_payload(Path(args.out_file).expanduser(), payload)
    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        print(
            "system_intense_drill_autopilot "
            f"overall_status={payload.get('overall_status', '')} "
            f"drills={payload.get('ready_drill_count', 0)}/{payload.get('drill_count', 0)} "
            f"deficiencies={payload.get('deficiency_count', 0)} "
            f"attempts={payload.get('attempt_count', 0)}"
        )
    return 0 if payload.get("overall_status") in {"ready", "degraded"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
