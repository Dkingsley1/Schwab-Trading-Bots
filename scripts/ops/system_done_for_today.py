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
    from scripts.ops.long_runtime_common import PROJECT_ROOT, iso_now, load_json, write_payload
else:
    from .long_runtime_common import PROJECT_ROOT, iso_now, load_json, write_payload


DEFAULT_OUT_PATH = PROJECT_ROOT / "governance" / "health" / "system_done_for_today_latest.json"


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except Exception:
        return int(default)


def _writer_done(writer: dict[str, Any]) -> tuple[bool, str]:
    state = _as_dict(writer.get("writer_state_before") or writer.get("writer_state_after_wait"))
    if not state:
        return True, "writer_state_missing_treat_as_no_active_blocker"
    active = bool(state.get("active", False))
    current_step = str(state.get("effective_current_step") or state.get("current_step") or "")
    planned = _safe_int(state.get("planned_shard_count"), 0)
    completed = _safe_int(state.get("completed_shard_count"), 0)
    complete = current_step == "complete" or (planned > 0 and completed >= planned and not bool(state.get("child_writer_active", False)))
    if active and not complete:
        return False, f"writer_active_{completed}/{planned}_{current_step}"
    return True, f"writer_done_or_idle_{completed}/{planned}_{current_step or 'idle'}"


def build_payload(project_root: Path = PROJECT_ROOT) -> dict[str, Any]:
    health = project_root / "governance" / "health"
    health_fast = load_json(health / "health_fast_latest.json")
    runtime = load_json(health / "runtime_throttle_control_latest.json")
    writer = load_json(health / "writer_cycle_coordinator_latest.json")
    paper = load_json(health / "paper_performance_latest.json")
    profitability = load_json(health / "paper_profitability_control_latest.json")
    watchdog = load_json(health / "watchdog_intelligence_latest.json")
    bot_needs = load_json(health / "bot_needs_intelligence_latest.json")

    storage = _as_dict(health_fast.get("storage")).get("backpressure")
    storage = _as_dict(storage)
    pending = _safe_int(storage.get("total_pending_lines"), 0)
    threshold = max(_safe_int(storage.get("pending_lines_threshold"), 15000), 1)
    oldest_age = _safe_float(storage.get("oldest_pending_age_seconds"), 0.0)
    oldest_threshold = max(_safe_float(storage.get("oldest_age_threshold_seconds"), 240.0), 1.0)
    backlog_green = bool(pending <= threshold and oldest_age <= oldest_threshold)
    writer_green, writer_reason = _writer_done(writer)
    runtime_status = str(runtime.get("overall_status") or _as_dict(health_fast.get("runtime_pressure")).get("overall_status") or "")
    saturation = _safe_float(runtime.get("host_saturation_score"), _safe_float(_as_dict(health_fast.get("runtime_pressure")).get("host_saturation_score"), 0.0))
    runtime_green = runtime_status in {"ready", "advisory"} or (runtime_status == "degraded" and saturation < 68.0)
    memory_status = str(_as_dict(health_fast.get("memory")).get("overall_status") or "")
    memory_green = memory_status in {"ready", "advisory", "normal", ""}
    paper_day = _as_dict(paper.get("day"))
    paper_green = bool(paper.get("ok", False) or paper_day.get("available", False))
    watchdog_green = str(watchdog.get("overall_status") or watchdog.get("status") or "").lower() in {"ready", "advisory", ""}
    training_selector = _as_dict(bot_needs.get("training_candidate_selector"))
    zero_repair = _as_dict(bot_needs.get("zero_observation_repair_contract"))

    blockers = []
    if not backlog_green:
        blockers.append("backlog_not_green")
    if not writer_green:
        blockers.append("writer_still_active")
    if not runtime_green:
        blockers.append("runtime_pressure_not_cool")
    if not memory_green:
        blockers.append("memory_not_green")
    if not paper_green:
        blockers.append("paper_performance_not_fresh")
    if not watchdog_green:
        blockers.append("watchdog_intelligence_not_clean")
    if bool(zero_repair.get("active", False)) and _safe_int(zero_repair.get("zero_observation_count"), 0) > 0:
        blockers.append("zero_observation_collectors_need_repair")

    can_stop = not blockers
    payload = {
        "timestamp_utc": iso_now(),
        "schema_version": 1,
        "ok": can_stop,
        "overall_status": "done_for_today" if can_stop else "needs_attention",
        "can_stop_chasing": can_stop,
        "blockers": blockers,
        "checks": {
            "backlog": {
                "green": backlog_green,
                "total_pending_lines": pending,
                "pending_lines_threshold": threshold,
                "oldest_pending_age_seconds": round(float(oldest_age), 3),
                "oldest_age_threshold_seconds": round(float(oldest_threshold), 3),
            },
            "writer": {"green": writer_green, "reason": writer_reason},
            "runtime": {"green": runtime_green, "status": runtime_status, "host_saturation_score": round(float(saturation), 3)},
            "memory": {"green": memory_green, "status": memory_status},
            "paper": {
                "green": paper_green,
                "day_utc": str(paper_day.get("day_utc") or paper.get("day") or ""),
                "net_pnl_total": _safe_float(paper_day.get("ending_net_pnl_total"), 0.0),
                "profitability_status": str(profitability.get("overall_status") or ""),
                "profitability_grade": str(profitability.get("profitability_grade") or ""),
            },
            "watchdog": {"green": watchdog_green, "grade": watchdog.get("grade"), "score": watchdog.get("score")},
            "training": {
                "mode": "micro_canary_or_paused_until_runtime_clear",
                "selected_candidate_count": _safe_int(training_selector.get("selected_count"), 0),
                "zero_observation_count": _safe_int(zero_repair.get("zero_observation_count"), 0),
            },
        },
        "recommended_actions": [
            "stop expanding and let collection/paper run" if can_stop else "",
            "run writer-cycle-coordinator --json again after the active writer cycle finishes" if "writer_still_active" in blockers else "",
            "run runtime-throttle --apply --json before training" if "runtime_pressure_not_cool" in blockers else "",
            "run training-data-intake --apply for zero-observation bots" if "zero_observation_collectors_need_repair" in blockers else "",
            "refresh paper-performance and paper-profitability-control" if "paper_performance_not_fresh" in blockers else "",
        ],
        "contract": {
            "mode": "system_done_for_today_v1",
            "read_only": True,
            "live_execution_allowed": False,
            "operator_goal": "make it obvious when the system is healthy enough to stop poking it",
        },
    }
    payload["recommended_actions"] = [str(item) for item in payload["recommended_actions"] if str(item)]
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Tell the operator whether the system is done for today.")
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--out-file", default=str(DEFAULT_OUT_PATH))
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    payload = build_payload(Path(args.project_root).resolve())
    write_payload(Path(args.out_file).expanduser(), payload)
    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        print(
            "system_done_for_today "
            f"status={payload.get('overall_status')} "
            f"can_stop={int(bool(payload.get('can_stop_chasing')))} "
            f"blockers={','.join(payload.get('blockers') or []) or 'none'}"
        )
    return 0 if payload.get("ok") else 2


if __name__ == "__main__":
    raise SystemExit(main())
