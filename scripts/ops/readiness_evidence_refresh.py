#!/usr/bin/env python3
from __future__ import annotations

import argparse
import fcntl
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

if __package__ in {None, ""}:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from scripts.ops.long_runtime_common import load_json, payload_age_minutes, run_bounded_process_group, write_payload
else:
    from .long_runtime_common import PROJECT_ROOT, load_json, payload_age_minutes, run_bounded_process_group, write_payload


DEFAULT_OUT = Path("governance/health/readiness_evidence_refresh_latest.json")
DEFAULT_LOCK = Path("governance/locks/readiness_evidence_refresh.lock")
SCHEMA_VERSION = 1
Runner = Callable[..., dict[str, Any]]


def _resolve(project_root: Path, path: Path) -> Path:
    return path.expanduser() if path.is_absolute() else project_root / path


def _step(
    name: str,
    script: str,
    artifact: str,
    *args: str,
    max_age_minutes: float = 15.0,
    allowed_returncodes: tuple[int, ...] = (0,),
    depends_on: tuple[str, ...] = (),
) -> dict[str, Any]:
    return {
        "name": name,
        "script": script,
        "artifact": artifact,
        "args": list(args),
        "max_age_minutes": float(max_age_minutes),
        "allowed_returncodes": list(allowed_returncodes),
        "depends_on": list(depends_on),
    }


def default_steps() -> list[dict[str, Any]]:
    return [
        _step(
            "independent_fill_acquisition",
            "scripts/ops/independent_fill_evidence_acquisition.py",
            "governance/health/independent_fill_evidence_acquisition_latest.json",
            "--apply",
            "--json",
            max_age_minutes=5,
        ),
        _step(
            "paper_execution_calibration",
            "scripts/paper_execution_calibration_report.py",
            "governance/health/paper_execution_calibration_latest.json",
            "--hours",
            "720",
            "--json",
            depends_on=("independent_fill_acquisition",),
        ),
        _step(
            "paper_performance",
            "scripts/paper_performance_report.py",
            "governance/health/paper_performance_latest.json",
            "--json",
        ),
        _step(
            "paper_profitability_control",
            "scripts/ops/paper_profitability_control.py",
            "governance/health/paper_profitability_control_latest.json",
            "--apply",
            "--json",
            depends_on=("paper_performance",),
        ),
        _step(
            "canary_rollout",
            "scripts/canary_rollout_guard.py",
            "governance/health/canary_rollout_latest.json",
            "--json",
        ),
        _step(
            "walk_forward_coverage_seed",
            "scripts/ops/walk_forward_coverage_seed.py",
            "governance/walk_forward/coverage_seed_latest.json",
            "--write-queue",
            "--json",
            max_age_minutes=60,
        ),
        _step(
            "promotion_candidate_advancement",
            "scripts/ops/promotion_candidate_advancement.py",
            "governance/health/promotion_candidate_advancement_latest.json",
            "--json",
            max_age_minutes=60,
            depends_on=("walk_forward_coverage_seed",),
        ),
        _step(
            "promotion_quality_gate",
            "scripts/promotion_quality_gate.py",
            "governance/health/promotion_quality_gate_latest.json",
            "--json",
            max_age_minutes=60,
            allowed_returncodes=(0, 2),
            depends_on=("promotion_candidate_advancement", "paper_execution_calibration"),
        ),
        _step(
            "unattended_soak_readiness",
            "scripts/ops/unattended_soak_readiness.py",
            "governance/health/unattended_soak_readiness_latest.json",
            "--json",
            allowed_returncodes=(0, 2),
        ),
        _step(
            "production_quality_control",
            "scripts/ops/production_quality_control.py",
            "governance/health/production_quality_control_latest.json",
            "--apply",
            "--refresh-contract",
            "--json",
            depends_on=("paper_profitability_control", "unattended_soak_readiness"),
        ),
        _step(
            "production_quality_slo",
            "scripts/ops/production_quality_slo_guard.py",
            "governance/health/production_quality_slo_guard_latest.json",
            "--apply",
            "--json",
            depends_on=("production_quality_control",),
        ),
        _step(
            "production_readiness",
            "scripts/ops/production_readiness_control.py",
            "governance/health/production_readiness_control_latest.json",
            "--apply",
            "--exit-zero",
            "--json",
            depends_on=("production_quality_slo",),
        ),
        _step(
            "live_money_readiness",
            "scripts/ops/live_money_readiness_contract.py",
            "governance/health/live_money_readiness_contract_latest.json",
            "--json",
            allowed_returncodes=(0, 2),
            depends_on=("production_readiness", "promotion_quality_gate"),
        ),
        _step(
            "production_excellence",
            "scripts/ops/production_excellence_control.py",
            "governance/health/production_excellence_control_latest.json",
            "--apply",
            "--json",
            depends_on=("live_money_readiness", "canary_rollout"),
        ),
        _step(
            "readiness_evidence_accrual",
            "scripts/ops/readiness_evidence_accrual.py",
            "governance/health/readiness_evidence_accrual_latest.json",
            "--apply",
            "--json",
            depends_on=("production_excellence",),
        ),
        _step(
            "readiness_blocker_rollup",
            "scripts/ops/readiness_blocker_rollup.py",
            "governance/health/readiness_blocker_rollup_latest.json",
            "--json",
            depends_on=("readiness_evidence_accrual", "production_quality_slo"),
        ),
    ]


def _artifact_age(path: Path, *, now: datetime) -> float | None:
    payload = load_json(path)
    return payload_age_minutes(payload, path, now=now)


def _parse_last_json(stdout: str) -> dict[str, Any]:
    for raw in reversed([line.strip() for line in str(stdout or "").splitlines() if line.strip()]):
        try:
            payload = json.loads(raw)
        except Exception:
            continue
        if isinstance(payload, dict):
            return payload
    return {}


def _acquire_lock(path: Path) -> tuple[Any | None, str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+", encoding="utf-8")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        handle.seek(0)
        owner = handle.read().strip()
        handle.close()
        return None, owner
    handle.seek(0)
    handle.truncate()
    handle.write(f"pid={os.getpid()} started={datetime.now(timezone.utc).isoformat()}\n")
    handle.flush()
    return handle, ""


def refresh(
    project_root: Path = PROJECT_ROOT,
    *,
    steps: list[dict[str, Any]] | None = None,
    force: bool = False,
    cooldown_minutes: float = 15.0,
    timeout_seconds: int = 180,
    out_path: Path = DEFAULT_OUT,
    runner: Runner = run_bounded_process_group,
    now: datetime | None = None,
) -> dict[str, Any]:
    current = now or datetime.now(timezone.utc)
    effective_out = _resolve(project_root, out_path)
    prior = load_json(effective_out)
    prior_age = payload_age_minutes(prior, effective_out, now=current)
    if not force and prior and prior_age is not None and prior_age < max(float(cooldown_minutes), 1.0):
        return {
            **prior,
            "refresh_skipped": True,
            "refresh_skip_reason": "cooldown_active",
            "refresh_query_timestamp_utc": current.isoformat(),
            "refresh_report_age_minutes": round(prior_age, 3),
            "write_latest": False,
        }

    selected_steps = steps if steps is not None else default_steps()
    results: list[dict[str, Any]] = []
    statuses: dict[str, str] = {}
    operational_failures: list[str] = []
    refreshed_names: list[str] = []
    for spec in selected_steps:
        name = str(spec.get("name") or "unnamed")
        artifact = _resolve(project_root, Path(str(spec.get("artifact") or "")))
        age_before = _artifact_age(artifact, now=current)
        dependency_refreshed = any(statuses.get(str(dep)) == "refreshed" for dep in spec.get("depends_on") or [])
        due = bool(force or dependency_refreshed or age_before is None or age_before > float(spec.get("max_age_minutes", 15.0)))
        if not due:
            statuses[name] = "fresh"
            results.append(
                {
                    "name": name,
                    "status": "fresh",
                    "artifact": str(artifact),
                    "age_minutes": round(age_before, 3) if age_before is not None else None,
                    "executed": False,
                }
            )
            continue
        command = [sys.executable, str(project_root / str(spec.get("script") or "")), *[str(arg) for arg in spec.get("args") or []]]
        result = runner(
            command,
            cwd=project_root,
            timeout_seconds=max(int(timeout_seconds), 30),
            env={
                **os.environ,
                "MARKET_DATA_ONLY": "1",
                "ALLOW_ORDER_EXECUTION": "0",
                "TOP_BOT_ENABLE_LIVE_EXECUTION": "0",
                "BOT_LIVE_MONEY_LOCKED_DURING_SOAK": "1",
            },
        )
        rc = int(result.get("rc", 125))
        allowed = {int(value) for value in spec.get("allowed_returncodes") or [0]}
        artifact_present = artifact.exists()
        parsed = _parse_last_json(str(result.get("stdout") or ""))
        operational_ok = bool(rc in allowed and artifact_present and not result.get("timed_out", False))
        status = "refreshed" if operational_ok else "failed"
        statuses[name] = status
        if operational_ok:
            refreshed_names.append(name)
        else:
            operational_failures.append(name)
        results.append(
            {
                "name": name,
                "status": status,
                "executed": True,
                "returncode": rc,
                "allowed_returncodes": sorted(allowed),
                "timed_out": bool(result.get("timed_out", False)),
                "artifact": str(artifact),
                "artifact_present": artifact_present,
                "age_minutes_before": round(age_before, 3) if age_before is not None else None,
                "published_status": str(parsed.get("overall_status") or parsed.get("status") or ""),
                "published_ok": parsed.get("ok"),
                "stdout_tail": str(result.get("stdout") or "")[-1000:],
                "stderr_tail": str(result.get("stderr") or "")[-1000:],
            }
        )
    payload = {
        "schema_version": SCHEMA_VERSION,
        "timestamp_utc": current.isoformat(),
        "overall_status": "ready" if not operational_failures else "degraded",
        "ok": not operational_failures,
        "refresh_skipped": False,
        "write_latest": True,
        "step_count": len(results),
        "refreshed_step_count": len(refreshed_names),
        "fresh_step_count": sum(1 for row in results if row["status"] == "fresh"),
        "failed_step_count": len(operational_failures),
        "refreshed_steps": refreshed_names,
        "operational_failures": operational_failures,
        "steps": results,
        "control_contract": {
            "bounded_step_timeouts": True,
            "single_writer_lock": True,
            "atomic_artifact_producers_required": True,
            "candidate_bound_evidence_only": True,
            "training_launch_authority": False,
            "live_execution_authority": False,
            "full_runtime_refresh_replacement": False,
        },
    }
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a bounded dependency-ordered refresh of live-money readiness evidence.")
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--out-file", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--lock-file", type=Path, default=DEFAULT_LOCK)
    parser.add_argument("--cooldown-minutes", type=float, default=15.0)
    parser.add_argument("--timeout-seconds", type=int, default=180)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--apply", action="store_true", help="Publish the refresh report; evidence producers publish their own bounded artifacts.")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    project_root = args.project_root.expanduser().resolve()
    lock_path = _resolve(project_root, args.lock_file)
    lock_handle, owner = _acquire_lock(lock_path)
    if lock_handle is None:
        payload = {
            "schema_version": SCHEMA_VERSION,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "overall_status": "already_running",
            "ok": True,
            "refresh_skipped": True,
            "refresh_skip_reason": "lock_busy",
            "lock_owner": owner,
            "write_latest": False,
        }
    else:
        try:
            payload = refresh(
                project_root,
                force=bool(args.force),
                cooldown_minutes=float(args.cooldown_minutes),
                timeout_seconds=int(args.timeout_seconds),
                out_path=args.out_file,
            )
            if args.apply and payload.get("write_latest", False):
                write_payload(_resolve(project_root, args.out_file), payload)
        finally:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
            lock_handle.close()
    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        print(
            "readiness_evidence_refresh "
            f"status={payload.get('overall_status')} refreshed={payload.get('refreshed_step_count', 0)} "
            f"failed={payload.get('failed_step_count', 0)} skipped={int(bool(payload.get('refresh_skipped', False)))}"
        )
    return 0 if bool(payload.get("ok", False)) else 2


if __name__ == "__main__":
    raise SystemExit(main())
