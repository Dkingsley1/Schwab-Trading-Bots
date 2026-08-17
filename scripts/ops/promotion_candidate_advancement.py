#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from scripts.ops import bot_needs_intelligence, training_runtime_control
    from scripts.ops.long_runtime_common import iso_now, load_json, ordered_unique, run_bounded_process_group, write_payload
else:
    from . import bot_needs_intelligence, training_runtime_control
    from .long_runtime_common import PROJECT_ROOT, iso_now, load_json, ordered_unique, run_bounded_process_group, write_payload


DEFAULT_OUT = Path("governance/health/promotion_candidate_advancement_latest.json")
DEFAULT_QUEUE = Path("governance/walk_forward/promotion_candidate_advancement_queue.json")
SCHEMA_VERSION = 1


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


def _parse_ts(raw: Any) -> datetime | None:
    text = str(raw or "").strip().replace("Z", "+00:00")
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text)
    except Exception:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _diagnostic_age_hours(row: dict[str, Any], now: datetime) -> float | None:
    direct = row.get("diagnostic_age_hours")
    if direct is not None:
        return max(_safe_float(direct), 0.0)
    timestamp = _parse_ts(row.get("diagnostic_timestamp_utc"))
    return max((now - timestamp).total_seconds() / 3600.0, 0.0) if timestamp is not None else None


def _candidate_rows(project_root: Path, *, limit: int, now: datetime) -> list[dict[str, Any]]:
    requalification = load_json(project_root / "governance" / "health" / "training_requalification_latest.json")
    walk_forward = load_json(project_root / "governance" / "walk_forward" / "walk_forward_latest.json")
    wf_rows = walk_forward.get("bots") if isinstance(walk_forward.get("bots"), dict) else {}
    source_rows = requalification.get("top_reactivation_ready") if isinstance(requalification.get("top_reactivation_ready"), list) else []
    out: list[dict[str, Any]] = []
    for rank, raw in enumerate(source_rows[: max(int(limit), 1)], start=1):
        if not isinstance(raw, dict):
            continue
        bot_id = str(raw.get("bot_id") or "").strip().lower()
        if not bot_id:
            continue
        wf = wf_rows.get(bot_id) if isinstance(wf_rows.get(bot_id), dict) else {}
        sample_count = _safe_int(raw.get("sample_count"), 0)
        diagnostic_age = _diagnostic_age_hours(raw, now)
        model_path = Path(str(raw.get("model_path") or ""))
        current_runs = max(_safe_int(raw.get("walk_forward_runs"), 0), _safe_int(wf.get("runs"), 0))
        blockers: list[str] = []
        if sample_count < 200:
            blockers.append("minimum_training_samples_pending")
        if diagnostic_age is None or diagnostic_age > bot_needs_intelligence.MAX_TRAINING_DIAGNOSTIC_AGE_HOURS:
            blockers.append("training_diagnostic_refresh_required")
        if not str(raw.get("model_path") or "").strip() or not model_path.exists():
            blockers.append("model_artifact_missing")
        if _safe_float(raw.get("quality_score"), 0.0) <= 0.0:
            blockers.append("quality_score_missing")
        if _safe_float(raw.get("test_accuracy"), 0.0) <= 0.0:
            blockers.append("test_accuracy_missing")
        stage = "walk_forward_training_ready" if not blockers else "data_or_repair_first"
        out.append(
            {
                "rank": rank,
                "bot_id": bot_id,
                "bot_role": str(raw.get("bot_role") or ""),
                "lifecycle_state": str(raw.get("lifecycle_state") or ""),
                "quality_score": round(_safe_float(raw.get("quality_score")), 6),
                "test_accuracy": round(_safe_float(raw.get("test_accuracy")), 6),
                "sample_count": sample_count,
                "diagnostic_age_hours": round(diagnostic_age, 3) if diagnostic_age is not None else None,
                "current_walk_forward_runs": current_runs,
                "target_walk_forward_runs": 12,
                "runs_remaining": max(12 - current_runs, 0),
                "stage": stage,
                "blockers": blockers,
                "next_actions": (
                    ["queue_runtime_guarded_walk_forward_training", "refresh_held_out_walk_forward_evidence"]
                    if not blockers
                    else ordered_unique(
                        (["collect_more_labeled_observations"] if "minimum_training_samples_pending" in blockers else [])
                        + (["refresh_training_diagnostics"] if "training_diagnostic_refresh_required" in blockers else [])
                        + (["repair_model_artifact"] if "model_artifact_missing" in blockers else [])
                    )
                ),
            }
        )
    return out


def _runtime_contract(project_root: Path, *, limit: int) -> dict[str, Any]:
    payload = training_runtime_control.build_payload(project_root, fresh_minutes=360, limit=max(int(limit), 1))
    contract = payload.get("training_launch_contract") if isinstance(payload.get("training_launch_contract"), dict) else {}
    return {
        "overall_status": str(payload.get("overall_status") or ""),
        "launch_allowed": bool(contract.get("launch_allowed", False)),
        "launch_blockers": [str(item) for item in contract.get("launch_blockers") or []],
        "recommended_batch_size": _safe_int(contract.get("recommended_batch_size"), 0),
        "recommended_retrain_profile": str(contract.get("recommended_retrain_profile") or ""),
        "canary_batch": [row for row in contract.get("canary_batch") or [] if isinstance(row, dict)],
    }


def build_payload(
    project_root: Path = PROJECT_ROOT,
    *,
    limit: int = 5,
    execute: bool = False,
    timeout_seconds: int = 7200,
    now: datetime | None = None,
) -> dict[str, Any]:
    current = now or datetime.now(timezone.utc)
    candidates = _candidate_rows(project_root, limit=limit, now=current)
    training_ready_ids = [row["bot_id"] for row in candidates if row["stage"] == "walk_forward_training_ready"]
    data_first_ids = [row["bot_id"] for row in candidates if row["stage"] != "walk_forward_training_ready"]
    runtime = _runtime_contract(project_root, limit=limit)
    runtime_ids = [str(row.get("bot_id") or "").strip().lower() for row in runtime.get("canary_batch") or []]
    approved_ids = [bot_id for bot_id in training_ready_ids if bot_id in set(runtime_ids)]
    approved_ids = approved_ids[: max(_safe_int(runtime.get("recommended_batch_size"), 0), 0)]
    command = [
        str(project_root / "scripts" / "ops" / "opsctl.sh"),
        "retrain-force-targeted",
        "--include-bot-ids",
        ",".join(approved_ids),
        "--retrain-profile",
        str(runtime.get("recommended_retrain_profile") or "coverage_micro_canary"),
        "--skip-master-update",
        "--refresh-held-out-walk-forward",
    ] if approved_ids else []
    execution: dict[str, Any] = {"attempted": False, "status": "publish_only" if not execute else "not_runtime_approved"}
    if execute and runtime.get("launch_allowed", False) and command:
        second_check = _runtime_contract(project_root, limit=limit)
        second_ids = {str(row.get("bot_id") or "").strip().lower() for row in second_check.get("canary_batch") or []}
        if second_check.get("launch_allowed", False) and set(approved_ids).issubset(second_ids):
            result = run_bounded_process_group(
                command,
                cwd=project_root,
                timeout_seconds=max(int(timeout_seconds), 60),
                env={**os.environ, "MARKET_DATA_ONLY": "1", "ALLOW_ORDER_EXECUTION": "0"},
                terminate_grace_seconds=10.0,
            )
            execution = {
                "attempted": True,
                "status": "completed" if int(result.get("rc", 1)) == 0 else "failed",
                "returncode": int(result.get("rc", 1)),
                "timed_out": bool(result.get("timed_out", False)),
                "stdout_tail": str(result.get("stdout") or "")[-1600:],
                "stderr_tail": str(result.get("stderr") or "")[-1600:],
            }
        else:
            execution = {"attempted": False, "status": "runtime_gate_changed_before_launch"}
    overall_status = "advanced" if execution.get("status") == "completed" else "queued" if training_ready_ids else "data_first"
    return {
        "schema_version": SCHEMA_VERSION,
        "timestamp_utc": iso_now(),
        "overall_status": overall_status,
        "ok": True,
        "candidate_limit": max(int(limit), 1),
        "staged_candidate_count": len(candidates),
        "training_ready_count": len(training_ready_ids),
        "data_or_repair_first_count": len(data_first_ids),
        "training_ready_bot_ids": training_ready_ids,
        "data_or_repair_first_bot_ids": data_first_ids,
        "runtime_approved_bot_ids": approved_ids,
        "candidates": candidates,
        "training_queue": [row for row in candidates if row["stage"] == "walk_forward_training_ready"],
        "data_first_queue": [row for row in candidates if row["stage"] != "walk_forward_training_ready"],
        "runtime_gate": runtime,
        "recommended_command": command,
        "execution": execution,
        "control_contract": {
            "top_requalification_candidates_prioritized": True,
            "sample_starved_candidates_route_to_data_first": True,
            "maximum_training_diagnostic_age_hours": bot_needs_intelligence.MAX_TRAINING_DIAGNOSTIC_AGE_HOURS,
            "diagnostic_freshness_matches_authoritative_bot_needs_selector": True,
            "training_requires_two_consecutive_runtime_gate_checks": True,
            "master_registry_updates_disabled": True,
            "held_out_walk_forward_refresh_requested": True,
            "live_execution_authority": False,
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Advance the strongest staged bots through guarded walk-forward evidence.")
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--out-file", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--queue-out", type=Path, default=DEFAULT_QUEUE)
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--timeout-seconds", type=int, default=7200)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    project_root = args.project_root.expanduser().resolve()
    payload = build_payload(
        project_root,
        limit=int(args.limit),
        execute=bool(args.execute),
        timeout_seconds=int(args.timeout_seconds),
    )
    out_path = args.out_file if args.out_file.is_absolute() else project_root / args.out_file
    queue_path = args.queue_out if args.queue_out.is_absolute() else project_root / args.queue_out
    write_payload(out_path, payload)
    write_payload(
        queue_path,
        {
            "schema_version": SCHEMA_VERSION,
            "timestamp_utc": payload["timestamp_utc"],
            "training_queue": payload["training_queue"],
            "data_first_queue": payload["data_first_queue"],
            "runtime_approved_bot_ids": payload["runtime_approved_bot_ids"],
        },
    )
    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        print(
            "promotion_candidate_advancement "
            f"status={payload['overall_status']} staged={payload['staged_candidate_count']} "
            f"training_ready={payload['training_ready_count']} data_first={payload['data_or_repair_first_count']}"
        )
    return 2 if payload.get("execution", {}).get("status") == "failed" else 0


if __name__ == "__main__":
    raise SystemExit(main())
