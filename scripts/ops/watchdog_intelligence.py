#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from core.runtime_python import resolve_runtime_python
    from scripts.ops.long_runtime_common import load_json, payload_age_minutes, write_payload
else:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    from core.runtime_python import resolve_runtime_python
    from .long_runtime_common import load_json, payload_age_minutes, write_payload


PY = resolve_runtime_python(PROJECT_ROOT)
HEALTH_DIR = PROJECT_ROOT / "governance" / "health"
DEFAULT_OUT_PATH = HEALTH_DIR / "watchdog_intelligence_latest.json"

PROCESS_WATCHDOG_PATH = HEALTH_DIR / "process_watchdog_latest.json"
PROCESS_FANOUT_PATH = HEALTH_DIR / "process_fanout_guard_latest.json"
MAC_NOTIFICATION_STATE_PATH = HEALTH_DIR / "mac_notification_watch_state.json"
ALL_SLEEVES_LAUNCHER_PATH = HEALTH_DIR / "all_sleeves_launcher_latest.json"

SUPERVISOR_PATTERNS = {
    "process_watchdog": "scripts/ops/process_watchdog.py",
    "shadow_watchdog": "scripts/shadow_watchdog.py",
    "mac_notification_watch": "scripts/ops/mac_notification_watch.py",
    "all_sleeves_launcher": "scripts/run_all_sleeves.py",
}
SINGLETON_SUPERVISORS = {"shadow_watchdog", "mac_notification_watch", "all_sleeves_launcher"}
SUPERVISOR_MATCH_EXCLUDES = {
    "all_sleeves_launcher": {
        "scripts/shadow_watchdog.py",
        "scripts/failover_hot_standby.py",
        "scripts/ops/process_watchdog.py",
        "scripts/ops/watchdog_intelligence.py",
    },
    "shadow_watchdog": {"scripts/ops/watchdog_intelligence.py"},
    "mac_notification_watch": {"scripts/ops/watchdog_intelligence.py"},
    "process_watchdog": {"scripts/ops/watchdog_intelligence.py"},
}


def _safe_int(raw: Any, default: int = 0) -> int:
    try:
        return int(float(raw))
    except Exception:
        return int(default)


def _run_json(cmd: list[str], *, timeout_seconds: float) -> dict[str, Any]:
    started = time.time()
    try:
        completed = subprocess.run(
            cmd,
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            check=False,
            timeout=max(float(timeout_seconds), 0.1),
        )
    except subprocess.TimeoutExpired as exc:
        return {
            "cmd": cmd,
            "ok": False,
            "rc": 124,
            "elapsed_seconds": round(time.time() - started, 3),
            "error": f"timeout_after_seconds={timeout_seconds}",
            "stdout_tail": (exc.stdout or "")[-500:] if isinstance(exc.stdout, str) else "",
            "stderr_tail": (exc.stderr or "")[-500:] if isinstance(exc.stderr, str) else "",
        }
    payload: dict[str, Any] = {}
    stdout = completed.stdout or ""
    if stdout.strip():
        try:
            parsed = json.loads(stdout)
            if isinstance(parsed, dict):
                payload = parsed
        except Exception:
            payload = {}
    return {
        "cmd": cmd,
        "ok": completed.returncode == 0,
        "rc": int(completed.returncode),
        "elapsed_seconds": round(time.time() - started, 3),
        "payload": payload,
        "stdout_tail": stdout[-500:],
        "stderr_tail": (completed.stderr or "")[-500:],
    }


def _collect_supervisors(project_marker: str = str(PROJECT_ROOT)) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            ["ps", "-axo", "pid,command"],
            capture_output=True,
            text=True,
            check=False,
            timeout=5.0,
        )
    except Exception as exc:
        return {"ok": False, "error": str(exc), "counts": {}, "duplicates": []}

    rows: list[dict[str, Any]] = []
    for line in (completed.stdout or "").splitlines()[1:]:
        raw = line.strip()
        if not raw or project_marker not in raw:
            continue
        parts = raw.split(maxsplit=1)
        if len(parts) != 2:
            continue
        pid_raw, command = parts
        if "watchdog_intelligence.py" in command:
            continue
        try:
            pid = int(pid_raw)
        except Exception:
            continue
        rows.append({"pid": pid, "command": command})

    counts: dict[str, int] = {}
    matches: dict[str, list[int]] = {}
    for name, pattern in SUPERVISOR_PATTERNS.items():
        excludes = SUPERVISOR_MATCH_EXCLUDES.get(name, set())
        pids = [
            row["pid"]
            for row in rows
            if pattern in str(row["command"])
            and not any(exclude in str(row["command"]) for exclude in excludes)
        ]
        counts[name] = len(pids)
        matches[name] = pids
    duplicates = [
        {"name": name, "count": count, "pids": matches.get(name, [])}
        for name, count in counts.items()
        if name in SINGLETON_SUPERVISORS and count > 1
    ]
    return {"ok": True, "counts": counts, "matches": matches, "duplicates": duplicates}


def _fallback_contract(process_watchdog: dict[str, Any]) -> dict[str, Any]:
    status_rows = process_watchdog.get("status") if isinstance(process_watchdog.get("status"), list) else []
    alerts = process_watchdog.get("alerts") if isinstance(process_watchdog.get("alerts"), list) else []
    restart_storms = (
        process_watchdog.get("restart_storms")
        if isinstance(process_watchdog.get("restart_storms"), list)
        else []
    )
    unhealthy = [
        row
        for row in status_rows
        if isinstance(row, dict) and not bool(row.get("heartbeat_ok", False)) and not row.get("restart_skipped")
    ]
    status = "critical" if restart_storms else ("degraded" if alerts or unhealthy else "ready")
    return {
        "overall_status": status,
        "grade": "A" if status == "ready" else ("C" if status == "degraded" else "F"),
        "score": 100.0 if status == "ready" else (76.0 if status == "degraded" else 40.0),
        "target_count": len(status_rows),
        "healthy_target_count": sum(1 for row in status_rows if isinstance(row, dict) and bool(row.get("heartbeat_ok", False))),
        "active_issue_count": len(unhealthy),
        "intentional_hold_count": 0,
        "restart_storm_count": len(restart_storms),
        "alert_count": len(alerts),
        "exact_needs": [],
        "recommended_commands": [["./scripts/ops/opsctl.sh", "watchdog-intelligence", "--apply", "--json"]],
    }


def _grade_from_score(score: float) -> str:
    if score >= 94:
        return "A"
    if score >= 86:
        return "B"
    if score >= 76:
        return "C"
    if score >= 66:
        return "D"
    return "F"


def _restart_storm_isolation(contract: dict[str, Any]) -> dict[str, int | bool]:
    isolation = contract.get("restart_storm_isolation") if isinstance(contract.get("restart_storm_isolation"), dict) else {}
    isolated_count = _safe_int(isolation.get("isolated_count"), 0)
    execution_blocking_count = _safe_int(isolation.get("execution_blocking_count"), 0)
    restart_storm_count = _safe_int(contract.get("restart_storm_count"), 0)
    if not isolation and restart_storm_count > 0:
        execution_blocking_count = restart_storm_count
    return {
        "isolated_count": isolated_count,
        "execution_blocking_count": execution_blocking_count,
        "all_active_storms_isolated": bool(
            isolated_count > 0
            and execution_blocking_count <= 0
            and bool(isolation.get("all_active_storms_isolated", False))
        ),
    }


def _all_sleeves_watchdog_ready(process_watchdog: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
    status_rows = process_watchdog.get("status") if isinstance(process_watchdog.get("status"), list) else []
    for row in status_rows:
        if not isinstance(row, dict) or str(row.get("name") or "") != "all_sleeves":
            continue
        child_fanout = row.get("child_fanout") if isinstance(row.get("child_fanout"), dict) else {}
        child_count = _safe_int(child_fanout.get("child_process_count"), _safe_int(row.get("alt_running"), 0))
        parent_live = bool(row.get("launcher_live", False))
        effective_live = bool(
            row.get("effective_process_live", False)
            or row.get("process_live", False)
            or row.get("launcher_live", False)
        )
        fanout_ok = bool(row.get("child_fanout_ok", child_fanout.get("ok", True)))
        heartbeat_ok = bool(row.get("heartbeat_ok", False))
        launcher_artifact_health = (
            row.get("launcher_artifact_health")
            if isinstance(row.get("launcher_artifact_health"), dict)
            else {}
        )
        ready = bool(effective_live and fanout_ok and heartbeat_ok and child_count > 0)
        return ready, {
            "active": ready,
            "source": "process_watchdog_all_sleeves",
            "parent_live": parent_live,
            "effective_live": effective_live,
            "launcher_artifact_certified_fanout": bool(row.get("launcher_artifact_certified_fanout", False)),
            "heartbeat_ok": heartbeat_ok,
            "child_fanout_ok": fanout_ok,
            "child_process_count": child_count,
            "launcher_artifact_reason": str(launcher_artifact_health.get("reason") or ""),
            "reason": "live_process_watchdog_fanout_verified" if ready else "process_watchdog_fanout_not_ready",
        }
    return False, {
        "active": False,
        "source": "process_watchdog_all_sleeves",
        "parent_live": False,
        "heartbeat_ok": False,
        "child_fanout_ok": False,
        "child_process_count": 0,
        "reason": "all_sleeves_row_missing",
    }


def build_report(
    *,
    process_watchdog: dict[str, Any],
    fanout_guard: dict[str, Any],
    mac_notification_state: dict[str, Any],
    all_sleeves_launcher: dict[str, Any],
    supervisors: dict[str, Any],
    now: datetime | None = None,
) -> dict[str, Any]:
    current = now or datetime.now(timezone.utc)
    contract = process_watchdog.get("watchdog_intelligence")
    if not isinstance(contract, dict):
        contract = _fallback_contract(process_watchdog)

    exact_needs: list[dict[str, Any]] = []
    for need in contract.get("exact_needs", []) if isinstance(contract.get("exact_needs"), list) else []:
        if isinstance(need, dict):
            exact_needs.append(need)

    for duplicate in supervisors.get("duplicates", []) if isinstance(supervisors.get("duplicates"), list) else []:
        name = str(duplicate.get("name") or "unknown")
        exact_needs.append(
            {
                "target": name,
                "severity": "warn",
                "status": "duplicate_supervisor",
                "blocker": "singleton_supervisor_count_above_one",
                "exact_file": str(PROCESS_WATCHDOG_PATH),
                "exact_command": ["./scripts/ops/opsctl.sh", "watchdog-intelligence", "--apply", "--json"],
                "expected_impact": "prevents duplicate alerts and duplicate restart decisions",
                "risk_level": "low",
                "when_to_stop": f"stop when {name} count is 1",
                "pids": duplicate.get("pids", []),
            }
        )

    fanout_override = fanout_guard.get("override") if isinstance(fanout_guard.get("override"), dict) else {}
    fanout_active = bool(fanout_override.get("active", False) or fanout_guard.get("triggered", False))
    fanout_hold_active = bool(fanout_override.get("hold_active", False))
    launcher_status = str(all_sleeves_launcher.get("status") or all_sleeves_launcher.get("current_step") or "")
    launcher_readiness_contract = (
        all_sleeves_launcher.get("launcher_readiness_contract")
        if isinstance(all_sleeves_launcher.get("launcher_readiness_contract"), dict)
        else {}
    )
    all_sleeves_ready, launcher_reconciliation = _all_sleeves_watchdog_ready(process_watchdog)
    launcher_exact_needs = (
        launcher_readiness_contract.get("exact_needs")
        if isinstance(launcher_readiness_contract.get("exact_needs"), list)
        else []
    )
    if all_sleeves_ready and launcher_exact_needs:
        launcher_reconciliation["suppressed_launcher_need_count"] = len(launcher_exact_needs)
        launcher_reconciliation["policy"] = "prefer_live_process_watchdog_child_fanout_over_stale_launcher_wrapper_rows"
    for need in launcher_exact_needs if not all_sleeves_ready else []:
        if isinstance(need, dict) and need.get("status") != "intentional_hold":
            exact_needs.append(
                {
                    **need,
                    "source": "all_sleeves_launcher_readiness",
                }
            )

    process_age = payload_age_minutes(process_watchdog, PROCESS_WATCHDOG_PATH, now=current)
    fanout_age = payload_age_minutes(fanout_guard, PROCESS_FANOUT_PATH, now=current)
    notification_age = payload_age_minutes(mac_notification_state, MAC_NOTIFICATION_STATE_PATH, now=current)
    readiness_raw = launcher_readiness_contract.get("readiness_score", 100.0)
    try:
        launcher_readiness_score = float(readiness_raw) if readiness_raw not in {None, ""} else 100.0
    except Exception:
        launcher_readiness_score = 100.0
    if all_sleeves_ready:
        launcher_readiness_score = max(launcher_readiness_score, 94.0)
    storm_isolation = _restart_storm_isolation(contract)
    isolated_storm_count = _safe_int(storm_isolation.get("isolated_count"), 0)
    execution_blocking_storm_count = _safe_int(storm_isolation.get("execution_blocking_count"), 0)

    section_scores = {
        "target_health": float(contract.get("score", 100.0) or 100.0),
        "restart_storm_control": 100.0
        - min(float(execution_blocking_storm_count) * 30.0, 70.0)
        - min(float(isolated_storm_count) * 8.0, 24.0),
        "notification_noise": 100.0 - min(float(len(supervisors.get("duplicates", []) or [])) * 20.0, 60.0),
        "guard_coordination": 94.0 if fanout_active or fanout_hold_active else 100.0,
        "sleeve_launcher_readiness": max(min(launcher_readiness_score, 100.0), 0.0),
        "artifact_freshness": 100.0,
    }
    stale_artifacts: list[str] = []
    for name, age in {
        "process_watchdog": process_age,
        "process_fanout_guard": fanout_age,
        "mac_notification_state": notification_age,
    }.items():
        if age is not None and age > 20.0:
            stale_artifacts.append(name)
    if stale_artifacts:
        section_scores["artifact_freshness"] = max(70.0 - len(stale_artifacts) * 8.0, 40.0)
        exact_needs.append(
            {
                "target": "watchdog_artifacts",
                "severity": "warn",
                "status": "stale_artifacts",
                "blocker": ",".join(stale_artifacts),
                "exact_file": str(PROCESS_WATCHDOG_PATH),
                "exact_command": ["./scripts/ops/opsctl.sh", "watchdog-intelligence", "--apply", "--json"],
                "expected_impact": "refreshes watchdog decisions before acting on old state",
                "risk_level": "low",
                "when_to_stop": "stop when process_watchdog and fanout artifacts are under 20 minutes old",
            }
        )

    overall_score = round(sum(section_scores.values()) / max(len(section_scores), 1), 1)
    active_need_count = sum(1 for need in exact_needs if need.get("status") != "intentional_hold")
    if execution_blocking_storm_count > 0:
        overall_status = "critical"
    elif active_need_count > 0:
        overall_status = "degraded"
    else:
        overall_status = "ready"

    recommended_commands: list[list[str]] = []
    for command in contract.get("recommended_commands", []):
        if isinstance(command, list) and command and command not in recommended_commands:
            recommended_commands.append([str(part) for part in command])
    if ["./scripts/ops/opsctl.sh", "watchdog-intelligence", "--apply", "--json"] not in recommended_commands:
        recommended_commands.append(["./scripts/ops/opsctl.sh", "watchdog-intelligence", "--apply", "--json"])

    return {
        "timestamp_utc": current.isoformat(),
        "overall_status": overall_status,
        "grade": _grade_from_score(overall_score),
        "score": overall_score,
        "section_grades": {
            name: {"score": round(score, 1), "grade": _grade_from_score(float(score))}
            for name, score in section_scores.items()
        },
        "process_watchdog_contract": contract,
        "restart_storm_isolation": {
            "isolated_count": isolated_storm_count,
            "execution_blocking_count": execution_blocking_storm_count,
            "all_active_storms_isolated": bool(storm_isolation.get("all_active_storms_isolated", False)),
            "policy": "isolated_read_only_collection_restart_debt_is_advisory_not_critical",
        },
        "supervisors": supervisors,
        "fanout_guard": {
            "active": fanout_active,
            "hold_active": fanout_hold_active,
            "hold_until_utc": str(fanout_override.get("hold_until_utc") or ""),
            "startup_policy": fanout_guard.get("startup_policy") if isinstance(fanout_guard.get("startup_policy"), dict) else {},
        },
        "all_sleeves_launcher": {
            "status": launcher_status,
            "running_job_count": _safe_int(all_sleeves_launcher.get("running_job_count"), 0),
            "policy_parked_job_count": _safe_int(all_sleeves_launcher.get("policy_parked_job_count"), 0),
            "clean_exited_job_count": _safe_int(all_sleeves_launcher.get("clean_exited_job_count"), 0),
            "readiness_contract": launcher_readiness_contract,
            "process_watchdog_reconciliation": launcher_reconciliation,
        },
        "artifact_ages_minutes": {
            "process_watchdog": round(float(process_age), 3) if process_age is not None else None,
            "process_fanout_guard": round(float(fanout_age), 3) if fanout_age is not None else None,
            "mac_notification_state": round(float(notification_age), 3) if notification_age is not None else None,
        },
        "exact_needs": exact_needs,
        "recommended_commands": recommended_commands,
        "out_file": str(DEFAULT_OUT_PATH),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Unify watchdog health, notification noise, and guard coordination.")
    parser.add_argument("--apply", action="store_true", help="Refresh fanout/process watchdog artifacts before reporting.")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--timeout-seconds", type=float, default=90.0)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_PATH)
    args = parser.parse_args()

    apply_records: list[dict[str, Any]] = []
    if args.apply:
        apply_records.append(
            _run_json(
                [str(PY), str(PROJECT_ROOT / "scripts" / "ops" / "process_fanout_guard.py"), "--apply", "--json"],
                timeout_seconds=max(float(args.timeout_seconds) / 2.0, 10.0),
            )
        )
        apply_records.append(
            _run_json(
                [str(PY), str(PROJECT_ROOT / "scripts" / "ops" / "process_watchdog.py"), "--json"],
                timeout_seconds=max(float(args.timeout_seconds), 10.0),
            )
        )

    report = build_report(
        process_watchdog=load_json(PROCESS_WATCHDOG_PATH),
        fanout_guard=load_json(PROCESS_FANOUT_PATH),
        mac_notification_state=load_json(MAC_NOTIFICATION_STATE_PATH),
        all_sleeves_launcher=load_json(ALL_SLEEVES_LAUNCHER_PATH),
        supervisors=_collect_supervisors(),
    )
    report["apply_records"] = apply_records
    write_payload(args.out, report)
    report["out_file"] = str(args.out)

    if args.json:
        print(json.dumps(report, ensure_ascii=True))
    else:
        print(
            f"watchdog_intelligence status={report['overall_status']} "
            f"grade={report['grade']} score={report['score']} out={args.out}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
