#!/usr/bin/env python3
from __future__ import annotations

import argparse
import fcntl
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable

if __package__ in {None, ""}:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from scripts.ops.long_runtime_common import iso_now, load_json, ordered_unique, write_payload
    from scripts.ops.paper_execution_truth_layer import assess_input_freshness
else:
    from .long_runtime_common import PROJECT_ROOT, iso_now, load_json, ordered_unique, write_payload
    from .paper_execution_truth_layer import assess_input_freshness


DEFAULT_OUT_PATH = PROJECT_ROOT / "governance" / "health" / "paper_truth_dependency_refresh_latest.json"
DEFAULT_LOCK_PATH = PROJECT_ROOT / "governance" / "health" / ".paper_truth_dependency_refresh.lock"
PAPER_ONLY_ENV = {
    "MARKET_DATA_ONLY": "1",
    "ALLOW_ORDER_EXECUTION": "0",
    "TOP_BOT_ENABLE_LIVE_EXECUTION": "0",
    "EXECUTION_LANE_LIVE_ENABLED": "0",
    "RUN_ALL_SLEEVES_WITH_LIVE_EXECUTOR": "0",
    "BOT_LIVE_MONEY_LOCKED_DURING_SOAK": "1",
}


RefreshRunner = Callable[..., dict[str, Any]]


def _dict(raw: Any) -> dict[str, Any]:
    return raw if isinstance(raw, dict) else {}


def _parse_payload(stdout: str) -> dict[str, Any]:
    for raw in reversed([line.strip() for line in str(stdout or "").splitlines() if line.strip()]):
        try:
            payload = json.loads(raw)
        except Exception:
            continue
        if isinstance(payload, dict):
            return payload
    return {}


def _run_step(name: str, cmd: list[str], *, cwd: Path, timeout_sec: int) -> dict[str, Any]:
    env = os.environ.copy()
    env.update(PAPER_ONLY_ENV)
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(cwd),
            env=env,
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
    return {
        "name": name,
        "cmd": cmd,
        "rc": rc,
        "timed_out": timed_out,
        "payload": _parse_payload(stdout),
        "stdout_tail": "\n".join(stdout.splitlines()[-6:]),
        "stderr_tail": "\n".join(stderr.splitlines()[-6:]),
    }


def _source_paths(project_root: Path) -> dict[str, Path]:
    health = project_root / "governance" / "health"
    return {
        "paper_performance": health / "paper_performance_latest.json",
        "calibration": health / "paper_execution_calibration_latest.json",
        "counterfactual": health / "counterfactual_replay_latest.json",
        "paper_replay": health / "paper_replay_drill_latest.json",
        "account_study": health / "account_position_study_latest.json",
        "covered_call_watch": health / "covered_call_roll_watch_latest.json",
        "execution_lab": health / "execution_lab_latest.json",
        "live_readiness": health / "live_readiness_smoke_latest.json",
        "ingestion_storage": health / "ingestion_storage_control_latest.json",
        "ingestion_backpressure": health / "ingestion_backpressure_latest.json",
        "promotion_quality": health / "promotion_quality_gate_latest.json",
        "broker_truth": health / "schwab_account_snapshot_refresh_latest.json",
        "source_verification": health / "source_verification_latest.json",
    }


def _freshness(project_root: Path) -> dict[str, Any]:
    paths = _source_paths(project_root)
    return assess_input_freshness(paths, {name: load_json(path) for name, path in paths.items()})


def _repair_groups(project_root: Path) -> list[dict[str, Any]]:
    opsctl = project_root / "scripts" / "ops" / "opsctl.sh"
    return [
        {
            "id": "auth_account_truth",
            "inputs": {"account_study", "covered_call_watch", "broker_truth"},
            "commands": [("schwab_auth_post_refresh", [str(opsctl), "schwab-auth-post-refresh", "--json"], 420)],
        },
        {
            "id": "ingestion_truth",
            "inputs": {"ingestion_storage", "ingestion_backpressure"},
            "commands": [
                (
                    "ingestion_backpressure",
                    [sys.executable, str(project_root / "scripts" / "ingestion_backpressure_guard.py"), "--json"],
                    180,
                ),
                ("ingestion_storage", [str(opsctl), "ingestion-storage-control", "--json"], 180),
            ],
        },
        {
            "id": "paper_performance_truth",
            "inputs": {"paper_performance", "calibration"},
            "commands": [
                ("paper_calibration", [str(opsctl), "paper-calibration", "--json"], 240),
                ("paper_performance", [str(opsctl), "paper-performance", "--json-only", "--json"], 240),
            ],
        },
        {
            "id": "replay_truth",
            "inputs": {"counterfactual", "paper_replay"},
            "commands": [
                (
                    "counterfactual_replay",
                    [sys.executable, str(project_root / "scripts" / "counterfactual_replay_harness.py"), "--json"],
                    240,
                ),
                (
                    "paper_replay",
                    [sys.executable, str(project_root / "scripts" / "paper_replay_drill.py"), "--hours", "24", "--json"],
                    240,
                ),
            ],
        },
        {
            "id": "execution_contract_truth",
            "inputs": {"execution_lab"},
            "commands": [("execution_lab", [str(opsctl), "execution-lab", "--json"], 180)],
        },
        {
            "id": "live_transition_truth",
            "inputs": {"live_readiness"},
            "commands": [
                (
                    "live_readiness",
                    [sys.executable, str(project_root / "scripts" / "live_readiness_smoke.py"), "--json"],
                    180,
                )
            ],
        },
        {
            "id": "source_context_truth",
            "inputs": {"source_verification"},
            "commands": [("source_verification", [str(opsctl), "source-verification", "--json"], 180)],
        },
        {
            "id": "promotion_contract_truth",
            "inputs": {"promotion_quality"},
            "commands": [
                (
                    "promotion_quality",
                    [sys.executable, str(project_root / "scripts" / "promotion_quality_gate.py"), "--json"],
                    180,
                )
            ],
        },
    ]


def build_payload(project_root: Path = PROJECT_ROOT, *, runner: RefreshRunner = _run_step) -> dict[str, Any]:
    project_root = Path(project_root).expanduser().resolve()
    before = _freshness(project_root)
    stale_before = ordered_unique(
        list(before.get("stale_operational_inputs") or [])
        + list(before.get("stale_promotion_evidence_inputs") or [])
    )
    stale_set = set(stale_before)
    steps: list[dict[str, Any]] = []
    selected_groups: list[str] = []
    for group in _repair_groups(project_root):
        if not stale_set.intersection(set(group["inputs"])):
            continue
        selected_groups.append(str(group["id"]))
        for name, cmd, timeout_sec in group["commands"]:
            step = runner(name, cmd, cwd=project_root, timeout_sec=timeout_sec)
            step["contract_ok"] = bool(not step.get("timed_out", False) and int(step.get("rc", 1)) in {0, 2})
            steps.append(step)

    truth_cmd = [sys.executable, str(project_root / "scripts" / "ops" / "paper_execution_truth_layer.py"), "--json"]
    truth_step = runner("paper_truth_verify", truth_cmd, cwd=project_root, timeout_sec=180)
    truth_step["contract_ok"] = bool(
        not truth_step.get("timed_out", False) and int(truth_step.get("rc", 1)) in {0, 2}
    )
    steps.append(truth_step)

    after = _freshness(project_root)
    stale_operational_after = list(after.get("stale_operational_inputs") or [])
    stale_promotion_after = list(after.get("stale_promotion_evidence_inputs") or [])
    truth_payload = _dict(truth_step.get("payload"))
    truth_ready = bool(
        truth_payload.get("ok", False)
        and str(truth_payload.get("overall_status") or "").strip().lower() == "ready"
    )
    operational_fresh = not stale_operational_after
    status = "ready" if operational_fresh and truth_ready else "blocked"
    failed_steps = [str(step.get("name") or "") for step in steps if not bool(step.get("contract_ok", False))]
    return {
        "timestamp_utc": iso_now(),
        "schema_version": 1,
        "ok": status == "ready",
        "overall_status": status,
        "truth_ready": truth_ready,
        "operational_conformance_score": float(truth_payload.get("score", 0.0) or 0.0),
        "raw_metric_score": float(truth_payload.get("raw_metric_score", 0.0) or 0.0),
        "promotion_ready": bool(truth_payload.get("promotion_ready", False)),
        "stale_inputs_before": stale_before,
        "stale_operational_inputs_after": stale_operational_after,
        "stale_promotion_evidence_inputs_after": stale_promotion_after,
        "selected_repair_groups": selected_groups,
        "failed_steps": failed_steps,
        "freshness_before": before,
        "freshness_after": after,
        "steps": [
            {
                "name": step.get("name"),
                "rc": int(step.get("rc", 1)),
                "timed_out": bool(step.get("timed_out", False)),
                "contract_ok": bool(step.get("contract_ok", False)),
                "stdout_tail": str(step.get("stdout_tail") or ""),
                "stderr_tail": str(step.get("stderr_tail") or ""),
            }
            for step in steps
        ],
        "safety_contract": {
            "market_data_only": True,
            "live_execution_allowed": False,
            "opens_browser": False,
            "interactive_oauth_allowed": False,
            "refreshes_only_stale_dependency_groups": True,
            "single_refresh_lock": True,
        },
        "recovery_contract": {
            "final_truth_evaluation_always_runs": True,
            "stale_operational_truth_fails_closed": True,
            "promotion_evidence_debt_remains_separate": True,
            "repair_owner": "paper_truth_dependency_recovery",
        },
    }


def _repair_in_progress_payload() -> dict[str, Any]:
    return {
        "timestamp_utc": iso_now(),
        "schema_version": 1,
        "ok": True,
        "overall_status": "repair_in_progress",
        "repair_in_progress": True,
        "recommended_actions": ["wait_for_active_paper_truth_dependency_refresh"],
        "safety_contract": {"single_refresh_lock": True, "live_execution_allowed": False},
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Refresh only stale paper-truth dependencies, then verify truth once.")
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--out-file", default=str(DEFAULT_OUT_PATH))
    parser.add_argument("--lock-file", default=str(DEFAULT_LOCK_PATH))
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    lock_path = Path(args.lock_file).expanduser()
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+", encoding="utf-8") as lock_handle:
        try:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            payload = _repair_in_progress_payload()
        else:
            payload = build_payload(Path(args.project_root))
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)

    write_payload(Path(args.out_file).expanduser(), payload)
    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        print(
            "paper_truth_dependency_refresh "
            f"status={payload.get('overall_status')} "
            f"score={float(payload.get('operational_conformance_score', 0.0) or 0.0):.2f} "
            f"stale_after={len(payload.get('stale_operational_inputs_after') or [])}"
        )
    return 0 if payload.get("overall_status") in {"ready", "repair_in_progress"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
