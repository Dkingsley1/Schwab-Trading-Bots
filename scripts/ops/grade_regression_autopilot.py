#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Callable

if __package__ in {None, ""}:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from scripts.ops import grade_regression_guard
    from scripts.ops.long_runtime_common import PROJECT_ROOT, eastern_off_hours_window, iso_now, ordered_unique, run_bounded_process_group, write_payload
else:
    from . import grade_regression_guard
    from .long_runtime_common import PROJECT_ROOT, eastern_off_hours_window, iso_now, ordered_unique, run_bounded_process_group, write_payload


DEFAULT_OUT_PATH = PROJECT_ROOT / "governance" / "health" / "grade_regression_autopilot_latest.json"
PYTHON_BIN = Path(sys.executable)
Runner = Callable[[list[str], Path, int], dict[str, Any]]
GuardBuilder = Callable[[Path], dict[str, Any]]


def _parse_json_output(text: str) -> dict[str, Any]:
    for raw in reversed([line.strip() for line in str(text or "").splitlines() if line.strip()]):
        try:
            payload = json.loads(raw)
        except Exception:
            continue
        if isinstance(payload, dict):
            return payload
    return {}


def _run(cmd: list[str], project_root: Path, timeout_sec: int) -> dict[str, Any]:
    result = run_bounded_process_group(cmd, cwd=project_root, timeout_seconds=max(int(timeout_sec), 1))
    stdout = str(result.get("stdout") or "")
    stderr = str(result.get("stderr") or "")
    return {
        "cmd": list(cmd),
        "rc": int(result.get("rc", 1)),
        "payload": _parse_json_output(stdout),
        "stdout_tail": "\n".join(stdout.splitlines()[-12:]),
        "stderr_tail": "\n".join(stderr.splitlines()[-12:]) or ("timeout" if result.get("timed_out") else ""),
        "timeout_cleanup": result.get("timeout_cleanup") if isinstance(result.get("timeout_cleanup"), dict) else {},
    }


def _surface_policy(row: dict[str, Any], *, default_timeout_sec: int) -> dict[str, Any]:
    retry_budget = row.get("retry_budget") if isinstance(row.get("retry_budget"), dict) else {}
    try:
        timeout_sec = int(float(retry_budget.get("step_timeout_sec", default_timeout_sec)))
    except Exception:
        timeout_sec = int(default_timeout_sec)
    try:
        max_attempts = int(float(retry_budget.get("max_attempts_per_run", 1)))
    except Exception:
        max_attempts = 1
    return {
        "timeout_sec": max(timeout_sec, 1),
        "max_attempts_per_run": max(max_attempts, 1),
        "cooldown_minutes": max(int(float(retry_budget.get("cooldown_minutes", 0) or 0)), 0),
        "quiet_hours_preferred": bool(retry_budget.get("quiet_hours_preferred", row.get("quiet_hours_preferred", False))),
        "notification_contract": dict(row.get("notification_contract") or {}),
    }


def _repair_plan(project_root: Path, guard_payload: dict[str, Any], *, storage_max_cycles: int) -> list[dict[str, Any]]:
    ops_root = project_root / "scripts" / "ops"
    plan: list[dict[str, Any]] = []
    seen: set[str] = set()

    policies: dict[str, dict[str, Any]] = {}
    for row in guard_payload.get("surfaces") or []:
        if isinstance(row, dict):
            policies[str(row.get("surface") or "")] = _surface_policy(row, default_timeout_sec=180)

    def add(surface: str, reason: str, cmd: list[str], timeout_sec: int) -> None:
        key = " ".join(cmd)
        if key in seen:
            return
        seen.add(key)
        policy = policies.get(surface, _surface_policy({}, default_timeout_sec=timeout_sec))
        plan.append(
            {
                "surface": surface,
                "reason": reason,
                "cmd": list(cmd),
                "timeout_sec": int(policy.get("timeout_sec", timeout_sec) or timeout_sec),
                "max_attempts_per_run": int(policy.get("max_attempts_per_run", 1) or 1),
                "cooldown_minutes": int(policy.get("cooldown_minutes", 0) or 0),
                "quiet_hours_preferred": bool(policy.get("quiet_hours_preferred", False)),
                "notification_contract": dict(policy.get("notification_contract") or {}),
            }
        )

    for row in guard_payload.get("surfaces") or []:
        if not isinstance(row, dict):
            continue
        surface = str(row.get("surface") or "")
        state = str(row.get("state") or "")
        if state == "ready":
            continue
        if surface == "training_quality":
            add(surface, state, [str(PYTHON_BIN), str(ops_root / "training_quality_control.py"), "--json"], 180)
        elif surface == "training_lineage":
            add(surface, state, [str(PYTHON_BIN), str(ops_root / "training_lineage_manifest.py"), "--json"], 180)
            add(surface, state, [str(PYTHON_BIN), str(ops_root / "promotion_autopilot_packet.py"), "--json"], 180)
        elif surface == "storage_control":
            add(surface, state, [str(PYTHON_BIN), str(ops_root / "ingestion_storage_control.py"), "--json"], 180)
            if str(os.getenv("RUNTIME_ARTIFACT_REFRESH_ACTIVE", "")).strip().lower() not in {"1", "true", "yes", "on"}:
                add(
                    surface,
                    state,
                    [
                        str(PYTHON_BIN),
                        str(ops_root / "storage_backpressure_autopilot.py"),
                        "--apply",
                        "--max-cycles",
                        str(max(int(storage_max_cycles), 1)),
                        "--json",
                    ],
                    900,
                )
        elif surface == "security_audit":
            add(surface, state, [str(PYTHON_BIN), str(ops_root / "security_evidence_autofix.py"), "--json"], 300)
            add(surface, state, [str(PYTHON_BIN), str(project_root / "scripts" / "security_hardening_audit.py")], 180)
        elif surface == "incident_closeout":
            add(surface, state, [str(PYTHON_BIN), str(ops_root / "incident_timeline.py"), "--json"], 180)
            add(surface, state, [str(PYTHON_BIN), str(ops_root / "incident_review_packet.py"), "--json"], 180)
            add(surface, state, [str(PYTHON_BIN), str(ops_root / "incident_closeout_autopilot.py"), "--json"], 180)
        elif surface == "live_canary":
            add(surface, state, [str(PYTHON_BIN), str(ops_root / "live_canary_control.py"), "--json"], 180)
            add(surface, state, [str(PYTHON_BIN), str(project_root / "scripts" / "live_readiness_smoke.py"), "--json"], 180)
        elif surface == "autonomy_control":
            add(surface, state, [str(PYTHON_BIN), str(ops_root / "runtime_throttle_control.py"), "--json"], 180)
            add(surface, state, [str(PYTHON_BIN), str(ops_root / "autonomy_control_plane.py"), "--json"], 180)
        elif surface == "promotion_autopilot":
            add(surface, state, [str(PYTHON_BIN), str(ops_root / "promotion_autopilot_packet.py"), "--json"], 180)

    return plan


def build_payload(
    project_root: Path = PROJECT_ROOT,
    *,
    apply: bool = False,
    timeout_sec: int = 1200,
    storage_max_cycles: int = 1,
    respect_quiet_hours: bool = False,
    runner: Runner | None = None,
    guard_builder: GuardBuilder | None = None,
) -> dict[str, Any]:
    run_step = runner or _run
    build_guard = guard_builder or grade_regression_guard.build_payload

    initial_guard = build_guard(project_root)
    repair_plan = _repair_plan(project_root, initial_guard, storage_max_cycles=storage_max_cycles)
    quiet_hours = eastern_off_hours_window()
    attempts: list[dict[str, Any]] = []
    if apply:
        for step in repair_plan:
            if respect_quiet_hours and bool(step.get("quiet_hours_preferred", False)) and not bool(quiet_hours.get("active", False)):
                attempts.append(
                    {
                        "surface": step["surface"],
                        "reason": step["reason"],
                        "cmd": list(step.get("cmd") or []),
                        "rc": 75,
                        "deferred": True,
                        "defer_reason": "quiet_hours_preferred",
                        "quiet_hours_window": quiet_hours,
                        "payload_summary": {},
                        "stdout_tail": "",
                        "stderr_tail": "deferred until quiet-hours window",
                    }
                )
                continue
            result = run_step(list(step["cmd"]), project_root, int(step.get("timeout_sec", timeout_sec)))
            attempts.append(
                {
                    "surface": step["surface"],
                    "reason": step["reason"],
                    "cmd": list(result.get("cmd") or []),
                    "rc": int(result.get("rc", 1)),
                    "deferred": False,
                    "quiet_hours_preferred": bool(step.get("quiet_hours_preferred", False)),
                    "notification_contract": dict(step.get("notification_contract") or {}),
                    "payload_summary": {
                        key: (result.get("payload") or {}).get(key)
                        for key in ("overall_status", "ok", "training_quality_score", "lineage_score", "autonomy_score")
                        if isinstance(result.get("payload"), dict) and key in (result.get("payload") or {})
                    },
                    "stdout_tail": str(result.get("stdout_tail") or ""),
                    "stderr_tail": str(result.get("stderr_tail") or ""),
                }
            )

    final_guard = build_guard(project_root)
    initial_status = str(initial_guard.get("overall_status") or "")
    final_status = str(final_guard.get("overall_status") or "")
    recommended_actions = ordered_unique(
        [
            "keep the grade regression guard running on a short interval so recoverable drift is republished before it becomes a hard blocker"
            if repair_plan
            else "",
            "leave the upgraded regression autopilot in apply mode so storage, training, incident, and canary surfaces keep their repair loop"
            if apply and repair_plan
            else "",
        ]
        + [str(item or "") for item in (final_guard.get("recommended_actions") or [])]
    )

    return {
        "timestamp_utc": iso_now(),
        "schema_version": 1,
        "ok": final_status == "ready",
        "overall_status": final_status or initial_status,
        "apply": bool(apply),
        "respect_quiet_hours": bool(respect_quiet_hours),
        "quiet_hours_window": quiet_hours,
        "repair_step_count": len(repair_plan),
        "attempt_count": len(attempts),
        "deferred_attempt_count": sum(1 for attempt in attempts if bool(attempt.get("deferred", False))),
        "initial_guard": {
            "overall_status": initial_status,
            "blocked_surface_count": int(initial_guard.get("blocked_surface_count", 0) or 0),
            "degraded_surface_count": int(initial_guard.get("degraded_surface_count", 0) or 0),
        },
        "final_guard": {
            "overall_status": final_status,
            "blocked_surface_count": int(final_guard.get("blocked_surface_count", 0) or 0),
            "degraded_surface_count": int(final_guard.get("degraded_surface_count", 0) or 0),
        },
        "repair_plan": repair_plan,
        "attempts": attempts,
        "regression_autopilot_contract": {
            "generation": "coordinated_regression_prevention_v4",
            "uses_per_surface_retry_budgets": True,
            "quiet_hours_deferral_available": True,
            "tenant_notification_contract_passthrough": True,
            "healthy_cycle_is_noop": True,
            "full_graph_refresh_forbidden": True,
            "evidence_accrual_owner": "readiness_evidence_refresh:accrual",
            "heavy_surfaces": [
                str(step.get("surface") or "")
                for step in repair_plan
                if bool(step.get("quiet_hours_preferred", False))
            ],
        },
        "upgrade_track": {
            "family": "infrabots",
            "upgradeable": True,
            "current_generation": "coordinated_regression_prevention_v4",
            "co_managed_with": [
                "grade_regression_guard",
                "grade_lift_hardening",
                "readiness_evidence_refresh:accrual",
            ],
            "future_upgrade_paths": [
                "adaptive retry budgets learned from repair success by surface",
                "launchd quiet-hours that bias heavy repair steps into cold-lane windows",
                "partner API exposure for licensee-facing health regression notifications",
            ],
        },
        "recommended_actions": recommended_actions,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the upgraded regression-prevention bot that refreshes the highest-value grade surfaces before they drift backward.")
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--out-file", default=str(DEFAULT_OUT_PATH))
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--timeout-sec", type=int, default=1200)
    parser.add_argument("--storage-max-cycles", type=int, default=1)
    parser.add_argument("--respect-quiet-hours", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    payload = build_payload(
        Path(args.project_root).resolve(),
        apply=bool(args.apply),
        timeout_sec=int(args.timeout_sec),
        storage_max_cycles=int(args.storage_max_cycles),
        respect_quiet_hours=bool(args.respect_quiet_hours),
    )
    out_path = Path(args.out_file).expanduser()
    write_payload(out_path, payload)
    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        print(
            "grade_regression_autopilot "
            f"overall_status={payload.get('overall_status', '')} "
            f"repair_step_count={payload.get('repair_step_count', 0)} "
            f"attempt_count={payload.get('attempt_count', 0)}"
        )
    return 0 if payload.get("overall_status") in {"ready", "degraded"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
