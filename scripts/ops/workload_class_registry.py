#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from scripts.ops.long_runtime_common import PROJECT_ROOT, iso_now, ordered_unique, write_payload
else:
    from .long_runtime_common import PROJECT_ROOT, iso_now, ordered_unique, write_payload


DEFAULT_OUT_PATH = PROJECT_ROOT / "governance" / "health" / "workload_class_registry_latest.json"


WORKLOAD_CLASSES: tuple[dict[str, Any], ...] = (
    {
        "class_id": "live_critical",
        "priority": 100,
        "objective": "Keep read-only live data, auth, risk, and protected execution controls responsive.",
        "default_cpu_policy": "protected_qos_observe",
        "default_storage_policy": "minimal_hot_path_writes",
        "may_run_during_user_work": True,
        "pause_first_under_pressure": False,
        "patterns": ["run_all_sleeves.py", "run_shadow_training_loop.py", "schwab_auth_supervisor.py", "global_risk_killswitch.py"],
    },
    {
        "class_id": "backlog_drain",
        "priority": 90,
        "objective": "Drain JSONL and queue backlog through the single-writer path with bounded P-core preprocessing.",
        "default_cpu_policy": "performance_core_primary_single_writer",
        "default_storage_policy": "sequential_sqlite_writer",
        "may_run_during_user_work": True,
        "pause_first_under_pressure": False,
        "patterns": ["writer_cycle_coordinator.py", "sql_link_shard_manager.py", "link_jsonl_to_sql.py", "backpressure_drainer_fleet.py", "backpressure_super_drainer.py"],
    },
    {
        "class_id": "collector",
        "priority": 70,
        "objective": "Collect fresh market, macro, provider, and sleeve context without outrunning the writer.",
        "default_cpu_policy": "duty_cycle_capped",
        "default_storage_policy": "pressure_aware_intake",
        "may_run_during_user_work": True,
        "pause_first_under_pressure": True,
        "patterns": ["collect_market", "market-correlation-sync", "macro-context-sync", "sec-edgar-sync", "extended-quant-sync"],
    },
    {
        "class_id": "maintenance_accelerated",
        "priority": 65,
        "objective": "Run bounded stale-artifact hashing and indexing with pressure-gated performance-core preference.",
        "default_cpu_policy": "darwin_user_initiated_qos_with_runtime_downshift",
        "default_storage_policy": "bounded_manifest_backed_retention",
        "may_run_during_user_work": True,
        "pause_first_under_pressure": True,
        "patterns": ["stale_artifact_reaper_bot.py"],
    },
    {
        "class_id": "training",
        "priority": 55,
        "objective": "Run targeted model and bot training only when memory, backlog, and user co-tenant pressure allow it.",
        "default_cpu_policy": "off_hours_or_clear_host_pcore",
        "default_storage_policy": "artifact_batched",
        "may_run_during_user_work": False,
        "pause_first_under_pressure": True,
        "patterns": ["weekly_retrain.py", "retrain", "coverage_gap_closer.py", "training_quality_control.py"],
    },
    {
        "class_id": "report",
        "priority": 45,
        "objective": "Build operator and performance reports from current artifacts without blocking core drains.",
        "default_cpu_policy": "nice_low",
        "default_storage_policy": "freshness_slo_batched",
        "may_run_during_user_work": True,
        "pause_first_under_pressure": True,
        "patterns": ["build_one_numbers_report.py", "paper_performance_report.py", "system_summary_report.py", "report_quality_guard.py"],
    },
    {
        "class_id": "maintenance",
        "priority": 40,
        "objective": "Run retention, vacuum, cleanup, quota, and guard maintenance after active writer work clears.",
        "default_cpu_policy": "idle_or_bounded",
        "default_storage_policy": "guarded_no_protected_volumes",
        "may_run_during_user_work": True,
        "pause_first_under_pressure": True,
        "patterns": ["data_retention_policy.py", "sqlite_performance_maintenance.py", "storage_quota_guard.py", "stale_artifact"],
    },
    {
        "class_id": "research",
        "priority": 30,
        "objective": "Explore new alpha, quant, and intelligence ideas without taking budget from live or backlog lanes.",
        "default_cpu_policy": "low_priority",
        "default_storage_policy": "digest_first",
        "may_run_during_user_work": False,
        "pause_first_under_pressure": True,
        "patterns": ["research_pipeline", "quant_model_control.py", "world_model", "alpha_benchmark"],
    },
    {
        "class_id": "user_coexistent",
        "priority": 110,
        "objective": "Protect normal computer use such as Logic Pro, Final Cut, Music, PyCharm, browsers, and Codex.",
        "default_cpu_policy": "user_foreground_wins",
        "default_storage_policy": "avoid_bursty_io",
        "may_run_during_user_work": True,
        "pause_first_under_pressure": False,
        "patterns": ["Logic Pro", "Final Cut Pro", "Music", "iTunes", "PyCharm", "Google Chrome", "Codex"],
    },
)


def classify_command(command: str) -> dict[str, Any]:
    lowered = str(command or "").lower()
    matches: list[dict[str, Any]] = []
    for row in WORKLOAD_CLASSES:
        patterns = row.get("patterns") if isinstance(row.get("patterns"), list) else []
        hit = [str(pattern) for pattern in patterns if str(pattern).lower() in lowered]
        if hit:
            matches.append({"class_id": row["class_id"], "priority": row["priority"], "matched_patterns": hit})
    if not matches:
        return {"class_id": "research", "confidence": 0.25, "matched_patterns": [], "reason": "default_low_priority_unknown_workload"}
    matches.sort(key=lambda item: int(item.get("priority", 0)), reverse=True)
    top = matches[0]
    return {"class_id": top["class_id"], "confidence": 0.9, "matched_patterns": top["matched_patterns"], "all_matches": matches}


def build_payload(*, classify: str = "") -> dict[str, Any]:
    class_map = {str(row["class_id"]): row for row in WORKLOAD_CLASSES}
    payload: dict[str, Any] = {
        "timestamp_utc": iso_now(),
        "schema_version": 1,
        "ok": True,
        "overall_status": "ready",
        "workload_classes": list(WORKLOAD_CLASSES),
        "class_order": [str(row["class_id"]) for row in sorted(WORKLOAD_CLASSES, key=lambda item: int(item["priority"]), reverse=True)],
        "class_contract": {
            "every_long_running_task_should_emit_workload_class": True,
            "unknown_tasks_default_to_low_priority_research": True,
            "user_coexistent_class_can_override_system_work": True,
            "single_writer_remains_exclusive_for_sqlite_writes": True,
            "protected_volumes": ["/Volumes/VIDEO"],
        },
    }
    if classify:
        result = classify_command(classify)
        result["class_spec"] = class_map.get(result["class_id"], {})
        payload["classification"] = result
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Publish the workload class registry used by resource governors and process guards.")
    parser.add_argument("--classify", default="")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--out", default=str(DEFAULT_OUT_PATH))
    args = parser.parse_args()
    payload = build_payload(classify=args.classify)
    write_payload(Path(args.out), payload)
    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        classes = ", ".join(ordered_unique([str(row["class_id"]) for row in WORKLOAD_CLASSES]))
        print(f"workload_class_registry status={payload['overall_status']} classes={classes}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
