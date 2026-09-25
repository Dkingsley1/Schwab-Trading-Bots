#!/usr/bin/env python3
"""Run a resumable, resource-admitted cohort of isolated research fits."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import uuid

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.environ["BOT_MLX_DISABLE"] = "1"

from core.training_diagnostic_contract import diagnostic_age_hours
from scripts.ops.long_runtime_common import load_json, run_bounded_process_group, write_payload
from scripts.ops import training_dataset_evaluation as evaluation
from scripts.ops import training_dataset_preflight as preflight

SUPPORTED = (*preflight.SUPPORTED_MODULES, *preflight.EXPLICIT_RUNTIME_MODULES)
STATE_PATH = Path("governance/health/training_research_batch_latest.json")
RUNS_PATH = Path("governance/training/research_batches")
TERMINAL = {"evaluated", "data_blocked", "failed"}
THREAD_ENV = {"BOT_MLX_DISABLE": "1", "OPENBLAS_NUM_THREADS": "1",
              "OMP_NUM_THREADS": "1", "VECLIB_MAXIMUM_THREADS": "1"}


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def select_bots(bot_ids: list[str]) -> list[str]:
    selected = list(dict.fromkeys(bot_ids or SUPPORTED))
    if not selected or len(selected) > len(SUPPORTED) or any(bot not in SUPPORTED for bot in selected):
        raise ValueError("unsupported_research_cohort")
    return selected


def cohort_contract(root: Path, bot_ids: list[str]) -> dict:
    snapshot = load_json(root / "governance/health/runtime_training_snapshot_latest.json")
    registry = load_json(root / "master_bot_registry.json")
    rows = {row["bot_id"]: row for row in registry.get("sub_bots", [])}
    intake = load_json(root / "governance/training_labeling_intelligence/label_depth_training_dataset_latest.json")
    policies = {row["bot_id"]: row.get("label_quality_contract", {}).get("split_policy", {})
                for row in intake.get("work_items", [])}
    owners = ["scripts/ops/training_research_batch.py", "scripts/ops/training_dataset_preflight.py",
              "scripts/ops/training_dataset_evaluation.py", "core/runtime_training_common.py",
              "core/crypto_runtime_bot_common.py", "core/runtime_requested_bot_common.py",
              "core/training_diagnostic_contract.py", "core/indicator_bot_common.py"]
    owners.extend(f"core/{bot}.py" for bot in bot_ids)
    return {
        "bot_ids": bot_ids,
        "snapshot": {key: snapshot.get(key) for key in (
            "schema_version", "rows_sha256", "rows_path", "timestamp_utc",
            "latest_row_timestamp_utc", "lookback_days")},
        "sources": {path: hashlib.sha256((root / path).read_bytes()).hexdigest() for path in owners},
        "bots": {bot: {"active": rows.get(bot, {}).get("active"),
                       "deleted_from_rotation": rows.get(bot, {}).get("deleted_from_rotation"),
                       "label_contract": rows.get(bot, {}).get("training_label_materialization_contract"),
                       "split_policy": policies.get(bot, {})} for bot in bot_ids},
    }


def contract_digest(contract: dict) -> str:
    return hashlib.sha256(json.dumps(contract, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def snapshot_ready(contract: dict) -> bool:
    snapshot = contract["snapshot"]
    now = datetime.now(timezone.utc)
    ages = [diagnostic_age_hours({"timestamp_utc": snapshot.get(key)}, now)
            for key in ("timestamp_utc", "latest_row_timestamp_utc")]
    return snapshot.get("schema_version") == 2 and bool(snapshot.get("rows_sha256")) and all(
        age is not None and 0 <= age <= 24 for age in ages)


def resource_admission(root: Path, env: dict) -> bool:
    result = run_bounded_process_group(
        [sys.executable, str(root / "scripts/resource_guard.py"), "--profile", "refresh"],
        cwd=root, env=env, timeout_seconds=20)
    return result["rc"] == 0 and not result["timed_out"]


def run_worker(root: Path, bot_id: str, bot_ids: list[str], expected_digest: str) -> dict:
    def verify():
        current = cohort_contract(root, bot_ids)
        if contract_digest(current) != expected_digest or not snapshot_ready(current):
            raise ValueError("cohort_input_changed_or_expired")

    verify()
    prepared = preflight.build_payload(root, bot_ids=[bot_id], materialize=True)
    verify()
    data_ready = any(row.get("data_checks_passed") for row in prepared.get("results", []))
    if data_ready and not resource_admission(root, {**os.environ, **THREAD_ENV}):
        return {"bot_id": bot_id, "status": "deferred", "reason": "resource_admission_denied_before_fit",
                "contract_sha256": expected_digest}
    measured = evaluation.build_payload(root, preflight=prepared) if data_ready else None
    verify()
    status = "data_blocked" if not data_ready else "evaluated" if measured["ok"] else "failed"
    return {"bot_id": bot_id, "status": status, "contract_sha256": expected_digest,
            "preflight": prepared, "evaluation": measured, "timestamp_utc": now_iso(),
            "authority_contract": dict(evaluation.AUTHORITY)}


def publish(root: Path, state: dict) -> dict:
    state["timestamp_utc"] = now_iso()
    state["completed_bot_count"] = sum(row["status"] in TERMINAL for row in state["results"])
    state["evaluated_bot_count"] = sum(row["status"] == "evaluated" for row in state["results"])
    state["data_blocked_bot_count"] = sum(row["status"] == "data_blocked" for row in state["results"])
    state["failed_bot_count"] = sum(row["status"] == "failed" for row in state["results"])
    state["pending_bot_count"] = len(state["bot_ids"]) - state["completed_bot_count"]
    state["diagnostic_quality_passed_count"] = sum(bool(row.get("diagnostic_quality_passed")) for row in state["results"])
    write_payload(root / RUNS_PATH / state["run_id"] / "progress.json", state)
    write_payload(root / STATE_PATH, state)
    return state


def resume_receipts_valid(root: Path, state: dict) -> bool:
    if [row.get("bot_id") for row in state.get("results", [])] != state["bot_ids"]:
        return False
    for row in state["results"]:
        if row.get("status") not in {*TERMINAL, "running", "pending"} or not 0 <= row.get("attempts", -1) <= 2:
            return False
        if row["status"] in {"evaluated", "data_blocked"} and not row.get("receipt_path"):
            return False
        if row.get("receipt_path"):
            path = root / RUNS_PATH / state["run_id"] / f"{row['bot_id']}.json"
            if str(path) != row["receipt_path"] or not path.is_file():
                return False
            if hashlib.sha256(path.read_bytes()).hexdigest() != row.get("receipt_sha256"):
                return False
            receipt = load_json(path)
            if (receipt.get("bot_id") != row["bot_id"] or receipt.get("status") != row["status"]
                    or receipt.get("contract_sha256") != state["contract_sha256"]):
                return False
    return True


def execute_queue(root: Path, *, bot_ids: list[str], seconds: int, new_run: bool = False) -> dict:
    if not 150 <= seconds <= 1800:
        raise ValueError("batch_seconds_must_be_between_150_and_1800")
    selected = select_bots(bot_ids)
    contract = cohort_contract(root, selected)
    digest = contract_digest(contract)
    state = load_json(root / STATE_PATH)
    if (root / STATE_PATH).exists() and not state and not new_run:
        return {"overall_status": "blocked", "blockers": ["batch_state_invalid_start_new_run"],
                "authority_contract": dict(evaluation.AUTHORITY)}
    if state and not new_run:
        if state.get("bot_ids") != selected or state.get("contract_sha256") != digest:
            return {"overall_status": "blocked", "blockers": ["cohort_changed_start_new_run"],
                    "authority_contract": dict(evaluation.AUTHORITY)}
        # The run ID is only a local directory component, never a supplied path.
        if str(uuid.UUID(state["run_id"])) != state["run_id"]:
            raise ValueError("invalid_batch_run_id")
        if not resume_receipts_valid(root, state):
            return {"overall_status": "blocked", "blockers": ["batch_receipt_invalid_start_new_run"],
                    "authority_contract": dict(evaluation.AUTHORITY)}
        if state.get("overall_status") == "complete":
            return state
    else:
        state = {"schema_version": 1, "run_id": str(uuid.uuid4()), "started_at_utc": now_iso(),
                 "bot_ids": selected, "contract_sha256": digest, "contract": contract,
                 "results": [{"bot_id": bot, "status": "pending", "attempts": 0} for bot in selected],
                 "authority_contract": dict(evaluation.AUTHORITY),
                 "limits": {"workers": 1, "worker_seconds": 120, "invocation_seconds": seconds,
                            "maximum_worker_attempts_per_bot": 2, "samples_per_bot": 2000},
                 "overall_status": "queued", "blockers": []}
    if not snapshot_ready(contract):
        state.update(overall_status="deferred", blockers=["snapshot_missing_stale_or_unverified"])
        return publish(root, state)
    deadline = time.monotonic() + seconds
    env = {**os.environ, **THREAD_ENV}
    state.update(overall_status="running", blockers=[])
    publish(root, state)
    for row in state["results"]:
        if row["status"] in TERMINAL:
            continue
        if contract_digest(cohort_contract(root, selected)) != digest or not snapshot_ready(contract):
            state.update(overall_status="deferred", blockers=["cohort_input_changed_or_expired"])
            break
        if deadline - time.monotonic() < 145:
            state.update(overall_status="deferred", blockers=["invocation_time_budget"])
            break
        if not resource_admission(root, env):
            state.update(overall_status="deferred", blockers=["resource_admission_denied"])
            break
        if row["attempts"] >= 2:
            row.update(status="failed", reason="worker_attempt_budget_exhausted")
            publish(root, state)
            continue
        row.update(status="running", attempts=row["attempts"] + 1)
        publish(root, state)
        try:
            outcome = run_bounded_process_group(
                [sys.executable, str(root / "scripts/ops/training_research_batch.py"),
                 "--worker", row["bot_id"], "--include-bot-ids", ",".join(selected),
                 "--contract-sha256", digest], cwd=root, env=env, timeout_seconds=120)
            if outcome["timed_out"]:
                row.update(status="pending", reason="worker_deadline")
                state.update(overall_status="deferred", blockers=["worker_deadline"])
                break
            if outcome["rc"] != 0:
                raise ValueError("worker_failed")
            if contract_digest(cohort_contract(root, selected)) != digest or not snapshot_ready(contract):
                row.update(status="pending", reason="cohort_input_changed_or_expired")
                state.update(overall_status="deferred", blockers=["cohort_input_changed_or_expired"])
                break
            result = json.loads(outcome["stdout"])
            if (result.get("bot_id") != row["bot_id"] or result.get("contract_sha256") != digest
                    or result.get("status") not in {*TERMINAL, "deferred"}):
                raise ValueError("worker_receipt_mismatch")
            if result["status"] == "deferred":
                row.update(status="pending", reason=result["reason"], attempts=row["attempts"] - 1)
                state.update(overall_status="deferred", blockers=[result["reason"]])
                break
            receipt_path = root / RUNS_PATH / state["run_id"] / f"{row['bot_id']}.json"
            write_payload(receipt_path, result)
            evaluated = (result.get("evaluation") or {}).get("results", [])
            data = result.get("preflight", {})
            row.update(status=result["status"], receipt_path=str(receipt_path),
                       receipt_sha256=hashlib.sha256(receipt_path.read_bytes()).hexdigest(),
                       blockers=data.get("blockers", []) + [blocker for item in data.get("results", [])
                                                            for blocker in item.get("blockers", [])],
                       diagnostic_quality_passed=bool(evaluated and evaluated[0].get("diagnostic_quality_passed")),
                       metrics=evaluated[0].get("metrics", {}) if evaluated else {})
        except (OSError, ValueError, KeyError, TypeError) as exc:
            row.update(status="failed", reason=f"worker_failed:{type(exc).__name__}")
        publish(root, state)
    if all(row["status"] in TERMINAL for row in state["results"]):
        state.update(overall_status="complete", blockers=[], finished_at_utc=now_iso())
    return publish(root, state)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--include-bot-ids", default="")
    parser.add_argument("--seconds", type=int, default=600)
    parser.add_argument("--new-run", action="store_true")
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--worker", choices=SUPPORTED, help=argparse.SUPPRESS)
    parser.add_argument("--contract-sha256", help=argparse.SUPPRESS)
    args = parser.parse_args()
    bot_ids = select_bots([part.strip() for part in args.include_bot_ids.split(",") if part.strip()])
    if args.worker:
        if args.worker not in bot_ids:
            parser.error("worker must belong to the selected cohort")
        try:
            os.nice(10)
        except OSError:
            pass  # Host resource admission and single-thread caps remain mandatory.
        print(json.dumps(run_worker(PROJECT_ROOT, args.worker, bot_ids, args.contract_sha256)))
        return 0
    if args.status:
        payload = load_json(PROJECT_ROOT / STATE_PATH) or {"overall_status": "not_started"}
    else:
        directory = PROJECT_ROOT / RUNS_PATH
        directory.mkdir(parents=True, exist_ok=True)
        with (directory / ".lock").open("a") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                print(json.dumps({"overall_status": "deferred", "blockers": ["batch_already_running"]}))
                return 2
            payload = execute_queue(PROJECT_ROOT, bot_ids=bot_ids, seconds=args.seconds, new_run=args.new_run)
    print(json.dumps(payload, indent=2) if args.json else
          f"Research batch: {payload['overall_status']}; evaluated={payload.get('evaluated_bot_count', 0)}; pending={payload.get('pending_bot_count', 0)}")
    return 0 if args.status or payload["overall_status"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
