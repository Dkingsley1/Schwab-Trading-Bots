#!/usr/bin/env python3
"""Bounded native fast-control pass; heavyweight repairs have separate owners."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.ops.long_runtime_common import (
    evidence_freshness,
    iso_now,
    load_json,
    parse_iso_utc,
    run_bounded_process_group,
    write_payload,
)
from scripts.ops.scheduled_lifecycle_common import lifecycle_receipt

CADENCE_SECONDS = 20
WORK_BUDGET_SECONDS = 18
FAIL_SAFE_RESERVE_SECONDS = 6
FAIL_SAFE_STEP = (
    "runtime_throttle_control",
    "scripts/ops/runtime_throttle_control.py",
    ("--apply", "--protective-hold", "--json"),
    8,
)

# Fixed dependency order. A fresh decision is not proof that workload admission passed.
FAST_STEPS = (
    (
        "resource_guard",
        "scripts/resource_guard.py",
        ("--profile", "refresh", "--json"),
        10,
    ),
    (
        "memory_efficiency_control",
        "scripts/ops/memory_efficiency_control.py",
        ("apply", "--json"),
        6,
    ),
    (
        "runtime_throttle_control",
        "scripts/ops/runtime_throttle_control.py",
        ("--apply", "--json"),
        12,
    ),
    (
        "support_maintenance_gate",
        "scripts/ops/support_maintenance_gate.py",
        ("--json",),
        3,
    ),
    (
        "memory_pressure_intelligence",
        "scripts/ops/memory_pressure_intelligence.py",
        ("--apply", "--json"),
        6,
    ),
    (
        "autonomic_resource_governor",
        "scripts/ops/autonomic_resource_governor.py",
        ("--apply", "--json"),
        6,
    ),
)
SLOW_STEPS = (
    (
        "grade_regression_guard",
        "scripts/ops/grade_regression_guard.py",
        ("--json",),
        5,
        300,
    ),
    (
        "training_runtime_control",
        "scripts/ops/training_runtime_control.py",
        ("--json",),
        8,
        300,
    ),
    (
        "whole_system_governor",
        "scripts/ops/whole_system_governor.py",
        ("--refresh", "--json"),
        8,
        900,
    ),
    (
        "paper_400_ramp",
        "scripts/ops/paper_400_ramp_control.py",
        ("--apply", "--json"),
        8,
        300,
    ),
    (
        "paper_live_data_standard",
        "scripts/ops/paper_live_data_standard.py",
        ("--apply", "--json"),
        8,
        300,
    ),
    (
        "paper_trade_lock_infrabot",
        "scripts/ops/paper_trade_lock_infrabot.py",
        ("--apply", "--json"),
        8,
        300,
    ),
)


def _run_step(root: Path, step: tuple, deadline: float) -> dict[str, Any]:
    name, script, args, limit = step[:4]
    step_started = time.monotonic()
    remaining = deadline - time.monotonic()
    if remaining < 3:
        return {
            "owner": name,
            "status": "deferred",
            "reason": "cycle_deadline",
            "attempted": False,
        }
    started = iso_now()
    env = dict(os.environ)
    env.update(
        MARKET_DATA_ONLY="1",
        ALLOW_ORDER_EXECUTION="0",
        TOP_BOT_ENABLE_LIVE_EXECUTION="0",
        EXECUTION_LANE_LIVE_ENABLED="0",
        BOT_LIVE_MONEY_LOCKED_DURING_SOAK="1",
        BOT_MLX_DISABLE="1",
        OPENBLAS_NUM_THREADS="1",
        OMP_NUM_THREADS="1",
    )
    try:
        result = run_bounded_process_group(
            [sys.executable, str(root / script), *args],
            cwd=root,
            timeout_seconds=min(limit, int(remaining) - 2),
            env=env,
            terminate_grace_seconds=0.5,
        )
    except OSError as exc:
        return {
            "owner": name,
            "status": "failed",
            "attempted": True,
            "started_utc": started,
            "completed_utc": iso_now(),
            "reason": "owner_launch_failed",
            "error_type": type(exc).__name__,
            "elapsed_seconds": round(time.monotonic() - step_started, 3),
        }
    payload = load_json(root / "governance" / "health" / f"{name}_latest.json")
    produced = parse_iso_utc(
        payload.get("timestamp_utc") or payload.get("generated_at_utc")
    )
    published = produced is not None and produced >= parse_iso_utc(started)
    protective_hold = "--protective-hold" in args
    applied = payload.get("apply_result")
    runtime_applied = (
        name == "runtime_throttle_control"
        and isinstance(applied, dict)
        and applied.get("applied") is True
    )
    observation = evidence_freshness(payload, max_age_minutes=3)
    memory_applied = (
        name == "memory_efficiency_control"
        and payload.get("action") == "apply"
        and payload.get("input_evidence_ready") is True
        and observation["fresh"]
        and isinstance(applied, dict)
        and applied.get("applied") is True
        and applied.get("override_verified") is True
        and applied.get("profile") == payload.get("recommended_profile")
    )
    # A read-only regression verdict must remain visible while repairs are held.
    regression_observed = (
        name == "grade_regression_guard"
        and payload.get("overall_status") in {"ready", "blocked", "degraded"}
        and payload.get("ok") is (payload.get("overall_status") == "ready")
        and isinstance(payload.get("surfaces"), list)
        and bool(payload["surfaces"])
        and all(
            isinstance(row, dict)
            and bool(row.get("surface"))
            and row.get("state") in {"ready", "blocked", "degraded"}
            for row in payload["surfaces"]
        )
        and observation["fresh"]
    )
    accepted_rc = result["rc"] == 0 or (
        result["rc"] == 2
        and (
            name == "resource_guard"
            or runtime_applied
            or memory_applied
            or regression_observed
        )
    )
    completed = bool(accepted_rc and not result["timed_out"] and published)
    if name == "grade_regression_guard":
        completed = bool(
            completed
            and regression_observed
            and result["rc"] == (2 if payload["overall_status"] == "blocked" else 0)
        )
    if protective_hold:
        completed = bool(
            completed
            and payload.get("protective_hold") is True
            and isinstance(applied, dict)
            and applied.get("applied") is True
        )
    return {
        "owner": name,
        "status": "complete" if completed else "failed",
        "attempted": True,
        "started_utc": started,
        "completed_utc": iso_now(),
        "elapsed_seconds": round(time.monotonic() - step_started, 3),
        "rc": result["rc"],
        "timed_out": result["timed_out"],
        "timeout_cleanup": result["timeout_cleanup"],
        "published_new_decision": published,
        "observation_evidence": observation,
        "applied_control_verified": runtime_applied or memory_applied,
        "regression_assessment_observed": regression_observed,
        "reported_status": payload.get("overall_status"),
        "reason": "" if completed else "owner_failed_or_did_not_publish",
        "protective_hold": protective_hold,
    }


def run_cycle(
    root: Path, *, seconds: float = WORK_BUDGET_SECONDS, scheduled: bool = False
) -> dict[str, Any]:
    health = root / "governance" / "health"
    health.mkdir(parents=True, exist_ok=True)
    lock_path = health / "governor_refresh.lock"
    with lock_path.open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            # Contenders must never replace the active owner's progress or final receipt.
            return {"ok": True, "overall_status": "deferred", "reason": "owner_running"}
        started = iso_now()
        cycle_started = time.monotonic()
        work_budget = min(max(seconds, 5), WORK_BUDGET_SECONDS)
        deadline = cycle_started + work_budget
        out = health / "governor_refresh_latest.json"
        previous = load_json(out)
        attempts = dict(previous.get("slow_attempts") or {})
        slow_results = dict(previous.get("slow_results") or {})
        steps: list[dict[str, Any]] = []
        payload: dict[str, Any] = {
            "timestamp_utc": started,
            "controller_timestamp_utc": started,
            "ok": False,
            "overall_status": "running",
            "steps": steps,
            "slow_attempts": attempts,
            "slow_results": slow_results,
            "cadence_seconds": CADENCE_SECONDS,
            "deadline_seconds": work_budget,
            "authority": {
                "live_orders": False,
                "training_launch": False,
                "registry_promotion": False,
            },
            "heavy_repair_owner": "production_hardening_watch",
        }
        write_payload(out, payload)
        fast_deadline = deadline - min(FAIL_SAFE_RESERVE_SECONDS, work_budget / 2)
        for step in FAST_STEPS:
            row = _run_step(root, step, fast_deadline)
            steps.append(row)
            write_payload(out, payload)
            if row["status"] != "complete":
                break
        fast_complete = len(steps) == len(FAST_STEPS) and all(
            row["status"] == "complete" for row in steps
        )
        payload["fast_control_complete"] = fast_complete
        payload["fast_control_elapsed_seconds"] = round(
            time.monotonic() - cycle_started, 3
        )
        write_payload(out, payload)
        if not fast_complete:
            # A failed sensor/controller must not leave a prior permissive override in charge.
            payload["protective_fallback"] = _run_step(root, FAIL_SAFE_STEP, deadline)
            write_payload(out, payload)
        if fast_complete:
            # Oldest-due first prevents a failing optional owner starving its peers.
            due = sorted(SLOW_STEPS, key=lambda step: str(attempts.get(step[0], "")))
            for step in due:
                name, _, _, _, interval = step
                if (
                    name.startswith("paper_")
                    and os.getenv("RUNTIME_SMOOTH_MODE_PAPER_REFRESH", "1") != "1"
                ):
                    continue
                if slow_results.get(name, {}).get("status") == "failed":
                    interval = 60
                artifact = load_json(health / f"{name}_latest.json")
                if not evidence_freshness(artifact, max_age_minutes=interval / 60)[
                    "fresh"
                ]:
                    interval = min(interval, 60)
                age = evidence_freshness(
                    {"timestamp_utc": attempts.get(name)}, max_age_minutes=interval / 60
                )
                if age["fresh"]:
                    continue
                # Optional work only starts with its full allowance plus cleanup reserve.
                if deadline - time.monotonic() < step[3] + 2:
                    row = {
                        "owner": name,
                        "status": "deferred",
                        "attempted": False,
                        "reason": "fast_cadence_budget_reserved",
                    }
                else:
                    row = _run_step(root, step, deadline)
                steps.append(row)
                if row["attempted"]:
                    attempts[name] = row["started_utc"]
                    slow_results[name] = row
                write_payload(out, payload)
                if not row["attempted"]:
                    break
        debt = [
            name
            for name, result in slow_results.items()
            if result.get("status") != "complete"
        ]
        complete = (
            fast_complete
            and not debt
            and all(row["status"] == "complete" for row in steps)
        )
        payload.update(
            timestamp_utc=iso_now(),
            controller_timestamp_utc=iso_now(),
            ok=complete,
            fast_control_complete=fast_complete,
            overall_status="complete" if complete else "degraded",
            cycle_elapsed_seconds=round(time.monotonic() - cycle_started, 3),
        )
        payload["unfinished_owners"] = debt + [
            row["owner"]
            for row in steps
            if row["status"] != "complete" and row["owner"] not in debt
        ]
        write_payload(out, payload)
        if scheduled:
            lifecycle = lifecycle_receipt(
                job_id="runtime_smooth_mode",
                scheduled=True,
                started_utc=started,
                completed_utc=parse_iso_utc(payload["timestamp_utc"]),
                schedule_interval_seconds=CADENCE_SECONDS,
                rc=0 if complete else 2,
                terminal_status=payload["overall_status"],
                ok=complete,
                failure_reason="" if complete else "governor_refresh_incomplete",
                deferred_reason="",
                artifact_present_before=True,
                artifact_present_after=True,
                deadline_seconds=work_budget,
                command=["scripts/ops/run_runtime_smooth_mode_launchd.sh"],
                source="runtime_smooth_mode_launchd",
            )
            write_payload(
                health / "runtime_smooth_mode_latest.json",
                {**payload, "job_lifecycle": lifecycle},
            )
        return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scheduled", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    payload = run_cycle(PROJECT_ROOT, scheduled=args.scheduled)
    print(
        json.dumps(payload, ensure_ascii=True)
        if args.json
        else f"governor_refresh={payload['overall_status']}"
    )
    return 0 if payload["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
