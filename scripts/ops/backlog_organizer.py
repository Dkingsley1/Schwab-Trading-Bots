#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_PATH = PROJECT_ROOT / "governance" / "health" / "backlog_organizer_latest.json"
DEFAULT_STATE_PATH = PROJECT_ROOT / "governance" / "backlog" / "backlog_organizer_allocation_latest.json"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except Exception:
        return int(default)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _status(payload: dict[str, Any], default: str = "missing") -> str:
    text = str(payload.get("overall_status") or payload.get("status") or "").strip().lower()
    if not text and isinstance(payload.get("overall"), dict):
        text = str(payload["overall"].get("status") or "").strip().lower()
    return text or default


def _guarded_paper_soak_green(health_root: Path) -> bool:
    health_fast = _load_json(health_root / "health_fast_latest.json")
    operational = health_fast.get("operational_readiness") if isinstance(health_fast.get("operational_readiness"), dict) else {}
    guarded_paper = operational.get("guarded_paper") if isinstance(operational.get("guarded_paper"), dict) else {}
    live_execution = operational.get("live_execution") if isinstance(operational.get("live_execution"), dict) else {}
    soak = _load_json(health_root / "unattended_soak_readiness_latest.json")
    paper_guard = _load_json(health_root / "runtime_paper_regression_guard_latest.json")
    guarded_ready = bool(guarded_paper.get("ok", False)) and str(guarded_paper.get("status") or "").strip().lower() in {
        "ready",
        "armed",
        "guarded_ready",
    }
    live_locked = str(live_execution.get("status") or "").strip().lower() in {
        "blocked_read_only",
        "locked",
        "read_only",
        "disabled",
    }
    soak_ready = bool(soak.get("ok", False)) and str(soak.get("overall_status") or "").strip().lower() == "ready"
    paper_guard_ready = bool(paper_guard.get("ok", False)) and str(paper_guard.get("overall_status") or "").strip().lower() == "ready"
    operational_health_ready = bool(
        health_fast.get("strict_all_clear", False)
        or (
            bool(health_fast.get("ok", False))
            and str(health_fast.get("overall_status") or "").strip().lower() in {"ready", "guarded_ready"}
        )
    )
    return bool(operational_health_ready and guarded_ready and live_locked and soak_ready and paper_guard_ready)


def _bounded_storage_soak_backlog(storage: dict[str, Any], *, total_pending: int, estimated_drain_minutes: float) -> bool:
    backpressure = storage.get("backpressure") if isinstance(storage.get("backpressure"), dict) else {}
    raw_live = backpressure.get("raw_live") if isinstance(backpressure.get("raw_live"), dict) else {}
    effective_raw_live = backpressure.get("effective_raw_live") if isinstance(backpressure.get("effective_raw_live"), dict) else {}
    raw = effective_raw_live or raw_live
    core_pending = _safe_int(raw.get("core_pending_lines"), 0)
    raw_total = max(_safe_int(raw.get("total_pending_lines"), 0), total_pending)
    oldest_age = _safe_float(raw.get("oldest_pending_age_seconds"), 0.0)
    pressure_index = _safe_float(storage.get("pressure_index"), 0.0)
    contract = storage.get("continuous_run_soak_contract") if isinstance(storage.get("continuous_run_soak_contract"), dict) else {}
    soak_ready = bool(contract.get("soak_ready", False)) and not list(contract.get("blockers") or [])
    low_pressure_bounded = bool(
        _status(storage) == "ready"
        and pressure_index <= 0.50
        and total_pending <= 15_000
        and core_pending <= 10_000
        and raw_total <= 15_000
        and oldest_age <= 900.0
    )
    steady_state_bounded = bool(
        _status(storage) == "ready"
        and pressure_index <= 0.85
        and total_pending <= 5_000
        and core_pending <= 2_500
        and raw_total <= 5_000
        and oldest_age <= 300.0
    )
    return bool((low_pressure_bounded or steady_state_bounded) and (soak_ready or estimated_drain_minutes >= 0.0))


def _ok(payload: dict[str, Any]) -> bool | None:
    value = payload.get("ok")
    return value if isinstance(value, bool) else None


def _command(*parts: str) -> list[str]:
    return ["./scripts/ops/opsctl.sh", *parts]


def _lane(
    *,
    lane_id: str,
    title: str,
    weak_points: list[int],
    owner: str,
    priority: int,
    status: str,
    evidence: list[str],
    next_commands: list[list[str]],
    policy: str,
) -> dict[str, Any]:
    return {
        "lane_id": lane_id,
        "title": title,
        "weak_points": weak_points,
        "owner": owner,
        "priority": priority,
        "status": status,
        "evidence": [item for item in evidence if item],
        "next_commands": next_commands,
        "policy": policy,
    }


def _registry_summary(project_root: Path) -> dict[str, Any]:
    registry = _load_json(project_root / "master_bot_registry.json")
    rows = registry.get("sub_bots") if isinstance(registry.get("sub_bots"), list) else []
    rows = [row for row in rows if isinstance(row, dict)]
    return {
        "total_bots": len(rows),
        "active_bots": sum(1 for row in rows if bool(row.get("active"))),
        "data_collection_only_bots": sum(
            1 for row in rows if str(row.get("lifecycle_state") or "") == "data_collection_only"
        ),
        "training_excluded_bots": sum(
            1 for row in rows if bool(row.get("training_excluded")) or bool(row.get("exclude_from_training"))
        ),
    }


def _git_status_summary(project_root: Path) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            ["git", "status", "--short", "--untracked-files=all"],
            cwd=str(project_root),
            check=False,
            capture_output=True,
            text=True,
            timeout=8,
        )
    except Exception:
        completed = None
    empty = {
        "available": False,
        "tracked_change_count": 0,
        "untracked_count": 0,
        "generated_core_untracked_count": 0,
        "untracked_test_count": 0,
        "untracked_config_count": 0,
        "obvious_scratch_count": 0,
    }
    if completed is None or completed.returncode != 0:
        return empty
    tracked = 0
    untracked = 0
    generated_core = 0
    untracked_tests = 0
    untracked_config = 0
    scratch = 0
    scratch_names = {"overwrite_test.txt", "private_copy_test.txt", "private_write_test.txt"}
    sample: list[str] = []
    for raw_line in completed.stdout.splitlines():
        line = raw_line.rstrip()
        if not line:
            continue
        status = line[:2]
        path = line[3:].strip()
        sample.append(line)
        if status == "??":
            untracked += 1
            if path.startswith("core/brain_refinery_v") and path.endswith(".py"):
                generated_core += 1
            if path.startswith("tests/"):
                untracked_tests += 1
            if path.startswith("config/"):
                untracked_config += 1
            if path in scratch_names:
                scratch += 1
        else:
            tracked += 1
    return {
        "available": True,
        "tracked_change_count": tracked,
        "untracked_count": untracked,
        "generated_core_untracked_count": generated_core,
        "untracked_test_count": untracked_tests,
        "untracked_config_count": untracked_config,
        "obvious_scratch_count": scratch,
        "sample": sample[:20],
    }


def build_payload(project_root: Path = PROJECT_ROOT, *, apply: bool = False) -> dict[str, Any]:
    health_root = project_root / "governance" / "health"
    runtime = _load_json(health_root / "runtime_throttle_control_latest.json")
    memory = _load_json(health_root / "memory_efficiency_control_latest.json")
    expansion = _load_json(health_root / "expansion_capacity_planner_latest.json")
    admission = _load_json(health_root / "new_bot_admission_guard_latest.json")
    collection = _load_json(health_root / "data_collection_observation_rollup_latest.json")
    dashboard = _load_json(health_root / "runtime_gate_dashboard_latest.json")
    storage = _load_json(health_root / "ingestion_storage_control_latest.json")
    drainer_fleet = _load_json(health_root / "backpressure_drainer_fleet_latest.json")
    super_drainer = _load_json(health_root / "backpressure_super_drainer_latest.json")
    training = _load_json(health_root / "training_quality_control_latest.json")
    bot_quality = _load_json(health_root / "bot_quality_autopilot_latest.json")
    live_runtime = _load_json(health_root / "live_runtime_separation_control_latest.json")
    auth = _load_json(health_root / "auth_lease_manager_latest.json")
    fanout = _load_json(health_root / "process_fanout_guard_latest.json")
    materialization = _load_json(health_root / "core_bot_materialization_guard_latest.json")
    worktree = _git_status_summary(project_root)
    guarded_paper_soak_green = _guarded_paper_soak_green(health_root)

    pressure = expansion.get("pressure_snapshot") if isinstance(expansion.get("pressure_snapshot"), dict) else {}
    capacity = expansion.get("capacity_contract") if isinstance(expansion.get("capacity_contract"), dict) else {}
    storage_backpressure = storage.get("backpressure") if isinstance(storage.get("backpressure"), dict) else {}
    dashboard_attention = []
    if isinstance(dashboard.get("overall"), dict):
        dashboard_attention = [
            str(item)
            for item in dashboard["overall"].get("attention", [])
            if str(item or "").strip()
        ]

    admission_blocking = _safe_int(admission.get("blocking_candidate_count"), _safe_int(pressure.get("admission_blocking_candidate_count"), 0))
    admission_candidates = _safe_int(admission.get("candidate_bot_count"), _safe_int(pressure.get("admission_candidate_count"), 0))
    training_ready = _safe_int(collection.get("training_ready_count"), 0)
    collector_count = _safe_int(collection.get("collector_count"), 0)
    total_pending = max(
        _safe_int(storage_backpressure.get("total_pending_lines"), 0),
        _safe_int(storage.get("pending_lines_total"), 0),
    )
    estimated_drain_minutes = _safe_float(storage_backpressure.get("estimated_total_drain_minutes"), 0.0)
    active_drainer = drainer_fleet.get("active_drainer") if isinstance(drainer_fleet.get("active_drainer"), dict) else {}
    active_drainer_name = str(active_drainer.get("name") or drainer_fleet.get("active_drainer") or "").strip()
    ready_drainer_count = _safe_int(drainer_fleet.get("ready_drainer_count"), 0)
    fleet_self = drainer_fleet.get("self_accommodation") if isinstance(drainer_fleet.get("self_accommodation"), dict) else {}
    super_summary = super_drainer.get("summary") if isinstance(super_drainer.get("summary"), dict) else {}
    super_packet = (
        super_drainer.get("grandmaster_context_packet")
        if isinstance(super_drainer.get("grandmaster_context_packet"), dict)
        else {}
    )
    super_safe_next_action = str(
        super_packet.get("safe_next_action")
        or fleet_self.get("next_safe_action")
        or "unknown"
    )
    drainer_accommodation_status = (
        "blocked"
        if total_pending and _status(drainer_fleet) == "blocked"
        else "needs_work"
        if total_pending or ready_drainer_count
        else "ready"
    )
    bounded_storage_soak_backlog = bool(
        guarded_paper_soak_green
        and _bounded_storage_soak_backlog(
            storage,
            total_pending=total_pending,
            estimated_drain_minutes=estimated_drain_minutes,
        )
    )
    promotion_training_status = (
        "blocked"
        if _status(training) in {"blocked", "critical"} or _status(bot_quality) in {"blocked", "critical"}
        else "ready"
        if _status(training) == "ready" and _status(bot_quality) == "ready"
        else "needs_work"
    )
    if guarded_paper_soak_green and promotion_training_status != "ready":
        promotion_training_status = "advisory"
    collection_operational = (
        collection.get("operational_collection")
        if isinstance(collection.get("operational_collection"), dict)
        else {}
    )
    collection_operational_ready = bool(
        collection.get("operational_ok", False)
        and str(collection.get("operational_status") or collection_operational.get("status") or "").strip().lower()
        == "ready"
    )
    collection_status = "needs_work" if collector_count and training_ready == 0 else _status(collection)
    if guarded_paper_soak_green and collection_operational_ready and collection_status not in {"ready", "advisory"}:
        collection_status = "advisory"
    if guarded_paper_soak_green and collection_status in {"missing", "needs_work"}:
        collection_status = "advisory"
    storage_backlog_status = (
        "blocked"
        if _status(storage) in {"blocked", "critical"}
        else "needs_work"
        if total_pending or estimated_drain_minutes > 120
        else _status(storage)
    )
    if bounded_storage_soak_backlog and storage_backlog_status == "needs_work":
        storage_backlog_status = "advisory"
    if bounded_storage_soak_backlog and drainer_accommodation_status == "needs_work":
        drainer_accommodation_status = "advisory"
    auth_runtime_status = "needs_work" if _status(auth) != "ready" or _status(live_runtime) != "ready" else "ready"
    if guarded_paper_soak_green and auth_runtime_status == "needs_work" and _status(auth) == "ready":
        auth_runtime_status = "advisory"
    worktree_status = "needs_work" if worktree.get("tracked_change_count") or worktree.get("untracked_count") else "ready"
    if guarded_paper_soak_green and worktree_status == "needs_work":
        worktree_status = "advisory"

    lanes = [
        _lane(
            lane_id="runtime_pressure",
            title="Runtime Pressure Organizer",
            weak_points=[1, 2],
            owner="runtime_pressure_infrabot",
            priority=10 if _status(runtime) in {"blocked", "critical", "degraded"} else 4,
            status=_status(runtime),
            evidence=[
                f"host_saturation_score={runtime.get('host_saturation_score', pressure.get('host_saturation_score', 'unknown'))}",
                f"compute_pressure_level={runtime.get('compute_pressure_level', pressure.get('compute_pressure_level', 'unknown'))}",
                f"expansion_rollout_mode={capacity.get('rollout_mode', 'unknown')}",
            ],
            next_commands=[
                _command("pressure-relief", "--apply", "--json"),
                _command("runtime-throttle", "--apply", "--json"),
                _command("memory-efficiency", "apply", "--json"),
                _command("expansion-capacity", "--json"),
            ],
            policy="protect live and paper lanes; downshift support/research loops before allowing expansion",
        ),
        _lane(
            lane_id="admission_contracts",
            title="Admission Contract Organizer",
            weak_points=[3],
            owner="admission_backlog_infrabot",
            priority=10 if admission_blocking else 3,
            status="blocked" if admission_blocking else "ready",
            evidence=[
                f"candidate_bot_count={admission_candidates}",
                f"blocking_candidate_count={admission_blocking}",
                f"top_actions={'; '.join(str(item) for item in admission.get('top_actions', [])[:3])}",
            ],
            next_commands=[
                _command("feature-store", "--json"),
                _command("replay-hash-registry", "--json"),
                _command("new-bot-admission", "--json"),
                _command("promotion-autopilot", "--json"),
            ],
            policy="do not promote or widen bots until sample, sequence, walk-forward, feature-store, and replay contracts clear",
        ),
        _lane(
            lane_id="collection_maturity",
            title="Collection Maturity Organizer",
            weak_points=[4],
            owner="collection_maturity_infrabot",
            priority=8 if collector_count and training_ready == 0 else 4,
            status=collection_status,
            evidence=[
                f"collector_count={collector_count}",
                f"bots_with_observations={collection.get('bots_with_observations', 'unknown')}",
                f"total_observations={collection.get('total_observations', 'unknown')}",
                f"training_ready_count={training_ready}",
                "paper_soak_advisory_only=true" if guarded_paper_soak_green and collection_status == "advisory" else "",
            ],
            next_commands=[
                _command("data-collection-observation-rollup", "--apply", "--json"),
                _command("runtime-training-snapshot", "--json"),
                _command("training-runtime-control", "--json"),
            ],
            policy="keep collection-only bots excluded from training until both observation and minimum-day gates clear",
        ),
        _lane(
            lane_id="promotion_training_quality",
            title="Promotion and Training Quality Organizer",
            weak_points=[5],
            owner="promotion_quality_infrabot",
            priority=9 if _status(training) in {"blocked", "critical"} or _status(bot_quality) in {"blocked", "critical"} else 5,
            status=promotion_training_status,
            evidence=[
                f"training_quality_status={_status(training)}",
                f"bot_quality_status={_status(bot_quality)}",
                f"training_quality_score={(training.get('summary') or {}).get('training_quality_score', training.get('training_quality_score', 'unknown')) if isinstance(training.get('summary'), dict) else training.get('training_quality_score', 'unknown')}",
                "paper_soak_advisory_only=true" if guarded_paper_soak_green and promotion_training_status == "advisory" else "",
            ],
            next_commands=[
                _command("training-quality", "--json"),
                _command("bot-quality-autopilot", "--json"),
                _command("promotion-autopilot", "--json"),
                _command("promotion-quality-gate", "--json"),
            ],
            policy="promotion remains locked until quality, replay, drift, supportability, and packet gates are coherent",
        ),
        _lane(
            lane_id="health_visibility",
            title="Health Visibility Organizer",
            weak_points=[6],
            owner="health_visibility_infrabot",
            priority=8 if dashboard_attention else 4,
            status=_status(dashboard, "missing") if dashboard_attention else "ready",
            evidence=[
                f"runtime_gate_dashboard_status={_status(dashboard, 'missing')}",
                f"attention={'; '.join(dashboard_attention[:8])}",
                f"materialization_status={_status(materialization, 'missing')}",
                f"fanout_status={_status(fanout, 'missing')}",
            ],
            next_commands=[
                ["./scripts/session_ready_check.py", "--json"],
                _command("runtime-gate-dashboard", "--json"),
                _command("health-fast", "--json"),
                _command("core-bot-materialization-guard", "--json"),
            ],
            policy="refresh required health artifacts before treating the control plane as authoritative",
        ),
        _lane(
            lane_id="storage_backlog",
            title="Storage Backlog Organizer",
            weak_points=[7],
            owner="storage_backlog_infrabot",
            priority=7 if total_pending or estimated_drain_minutes > 120 else 3,
            status=storage_backlog_status,
            evidence=[
                f"storage_status={_status(storage)}",
                f"total_pending_lines={total_pending}",
                f"estimated_total_drain_minutes={estimated_drain_minutes}",
                f"pressure_index={storage.get('pressure_index', 'unknown')}",
                f"active_drainer={active_drainer_name or 'none'}",
                f"ready_drainer_count={ready_drainer_count}",
                f"super_drainer_status={_status(super_drainer, 'missing')}",
                f"super_safe_next_action={super_safe_next_action}",
                "bounded_storage_soak_backlog=true" if bounded_storage_soak_backlog else "",
            ],
            next_commands=[
                _command("external-backlog-drain", "--apply", "--json"),
                _command("backpressure-drainer-fleet", "--apply", "--json"),
                _command("backpressure-super-drainer", "--apply", "--max-waves", "1", "--target-pending-lines", "10000", "--json"),
                _command("storage-pressure-clearance", "--apply", "--json"),
                _command("ingestion-storage-control", "--json"),
            ],
            policy="drain hot-path JSONL and SQL backlog without increasing live write contention",
        ),
        _lane(
            lane_id="drainer_self_accommodation",
            title="Drainer Self Accommodation Organizer",
            weak_points=[1, 2, 7],
            owner="drainer_accommodation_infrabot",
            priority=8 if total_pending or ready_drainer_count else 3,
            status=drainer_accommodation_status,
            evidence=[
                f"fleet_status={_status(drainer_fleet, 'missing')}",
                f"fleet_active_drainer={active_drainer_name or 'none'}",
                f"fleet_ready_drainer_count={ready_drainer_count}",
                f"fleet_self_mode={fleet_self.get('mode', 'unknown')}",
                f"fleet_next_safe_action={fleet_self.get('next_safe_action', 'unknown')}",
                f"super_drainer_status={_status(super_drainer, 'missing')}",
                f"super_waves_run={super_summary.get('waves_run', 'unknown')}",
                f"super_progress_waves={super_summary.get('progress_waves', 'unknown')}",
                f"super_stop_reason={super_summary.get('stop_reason', 'unknown')}",
            ],
            next_commands=[
                _command("backpressure-drainer-fleet", "--json"),
                _command("backpressure-super-drainer", "--max-waves", "1", "--target-pending-lines", "10000", "--json"),
                _command("backpressure-super-drainer", "--apply", "--max-waves", "1", "--target-pending-lines", "10000", "--json"),
                _command("writer-cycle-coordinator", "--json"),
                _command("ingestion-storage-control", "--json"),
            ],
            policy="let drainers widen by sequencing one focused handoff at a time, parking on writer locks, market-hour guards, stale snapshots, and progress stalls",
        ),
        _lane(
            lane_id="auth_runtime_separation",
            title="Auth and Runtime Separation Organizer",
            weak_points=[1, 5, 6],
            owner="runtime_separation_infrabot",
            priority=8 if _status(auth) in {"degraded", "blocked"} or _status(live_runtime) in {"degraded", "blocked"} else 4,
            status=auth_runtime_status,
            evidence=[
                f"auth_lease_status={_status(auth)}",
                f"auth_expires_in_seconds={(auth.get('summary') or {}).get('expires_in_seconds', auth.get('expires_in_seconds', 'unknown')) if isinstance(auth.get('summary'), dict) else auth.get('expires_in_seconds', 'unknown')}",
                f"live_runtime_separation_status={_status(live_runtime)}",
                "paper_soak_advisory_only=true" if guarded_paper_soak_green and auth_runtime_status == "advisory" else "",
            ],
            next_commands=[
                _command("schwab-auth-guard", "--json"),
                _command("live-runtime-separation", "--json"),
                _command("operator-cockpit", "--json"),
            ],
            policy="keep auth leases fresh and isolate live lanes before any promotion or expansion work",
        ),
        _lane(
            lane_id="worktree_hygiene",
            title="Worktree Hygiene Organizer",
            weak_points=[6],
            owner="worktree_hygiene_infrabot",
            priority=6 if worktree.get("tracked_change_count") or worktree.get("untracked_count") else 2,
            status=worktree_status,
            evidence=[
                "tracked_change_count={}".format(worktree.get("tracked_change_count", 0)),
                "untracked_count={}".format(worktree.get("untracked_count", 0)),
                "generated_core_untracked_count={}".format(worktree.get("generated_core_untracked_count", 0)),
                "untracked_config_count={}".format(worktree.get("untracked_config_count", 0)),
                "untracked_test_count={}".format(worktree.get("untracked_test_count", 0)),
                "obvious_scratch_count={}".format(worktree.get("obvious_scratch_count", 0)),
                "paper_soak_advisory_only=true" if guarded_paper_soak_green and worktree_status == "advisory" else "",
            ],
            next_commands=[
                ["git", "status", "--short"],
                _command("core-bot-materialization-guard", "--json"),
                _command("backlog-organizer", "--apply", "--json"),
            ],
            policy="remove only obvious scratch files automatically; preserve generated expansion files and user edits until explicitly reviewed",
        ),
    ]

    lanes = sorted(lanes, key=lambda row: int(row.get("priority", 0)), reverse=True)
    blocking_lanes = [lane for lane in lanes if str(lane.get("status")) in {"blocked", "critical", "degraded", "missing"}]
    needs_work_lanes = [lane for lane in lanes if str(lane.get("status")) == "needs_work"]
    advisory_lanes = [lane for lane in lanes if str(lane.get("status")) == "advisory"]
    registry = _registry_summary(project_root)
    overall_status = "blocked" if blocking_lanes else "needs_work" if needs_work_lanes else "ready"

    payload = {
        "timestamp_utc": _utc_now(),
        "schema_version": 1,
        "ok": overall_status == "ready",
        "overall_status": overall_status,
        "apply": bool(apply),
        "summary": {
            "lane_count": len(lanes),
            "blocking_lane_count": len(blocking_lanes),
            "needs_work_lane_count": len(needs_work_lanes),
            "advisory_lane_count": len(advisory_lanes),
            "guarded_paper_soak_green": bool(guarded_paper_soak_green),
            "bounded_storage_soak_backlog": bool(bounded_storage_soak_backlog),
            **registry,
            "worktree_tracked_change_count": int(worktree.get("tracked_change_count", 0) or 0),
            "worktree_untracked_count": int(worktree.get("untracked_count", 0) or 0),
        },
        "lanes": lanes,
        "allocated_organizers": [
            {
                "organizer_id": lane["owner"],
                "lane_id": lane["lane_id"],
                "priority": lane["priority"],
                "status": lane["status"],
            }
            for lane in lanes
        ],
        "recommended_actions": [
            "run the highest-priority lane commands in order, stopping before any command that asks to promote, widen, or live-enable bots",
            "keep expansion blocked until runtime_pressure and admission_contracts lanes are no longer blocked",
            "use this artifact as the backlog index for operator cockpit and future infrastructure autofix passes",
        ],
        "source_files": {
            "artifact": str(DEFAULT_OUT_PATH),
            "state": str(DEFAULT_STATE_PATH),
        },
    }
    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Allocate current platform weak points into backlog organizer lanes.")
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--out-file", default=str(DEFAULT_OUT_PATH))
    parser.add_argument("--state-file", default=str(DEFAULT_STATE_PATH))
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    project_root = Path(args.project_root).expanduser().resolve()
    payload = build_payload(project_root, apply=bool(args.apply))
    out_path = Path(args.out_file).expanduser()
    if not out_path.is_absolute():
        out_path = project_root / out_path
    _write_json(out_path, payload)
    if args.apply:
        state_path = Path(args.state_file).expanduser()
        if not state_path.is_absolute():
            state_path = project_root / state_path
        _write_json(state_path, payload)
    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        summary = payload["summary"]
        print(
            "backlog_organizer "
            f"status={payload['overall_status']} "
            f"lanes={summary['lane_count']} "
            f"blocking={summary['blocking_lane_count']} "
            f"needs_work={summary['needs_work_lane_count']}"
        )
    return 0 if payload["overall_status"] == "ready" else 2


if __name__ == "__main__":
    raise SystemExit(main())
