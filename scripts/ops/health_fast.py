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


DEFAULT_OUT_PATH = PROJECT_ROOT / "governance" / "health" / "health_fast_latest.json"


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


def _health(project_root: Path, name: str) -> dict[str, Any]:
    payload = load_json(project_root / "governance" / "health" / name)
    return payload if isinstance(payload, dict) else {}


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _status(value: Any) -> str:
    return str(value or "").strip().lower()


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _alert_details(alerts: list[Any]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for raw in alerts:
        row = raw if isinstance(raw, dict) else {}
        alert = row.get("alert") if isinstance(row.get("alert"), dict) else {}
        nested: dict[str, Any] = {}
        stdout = str(alert.get("stdout") or "").strip()
        if stdout:
            try:
                parsed = json.loads(stdout)
                if isinstance(parsed, dict):
                    nested = parsed
            except Exception:
                nested = {}
        severity = _status(alert.get("severity") or nested.get("severity") or row.get("severity") or "warn")
        event = str(alert.get("event") or nested.get("event") or row.get("type") or "").strip()
        target = str(row.get("name") or row.get("target") or nested.get("target") or "").strip()
        rows.append(
            {
                "target": target,
                "type": str(row.get("type") or ""),
                "severity": severity,
                "event": event,
                "blocks_guarded_paper": severity in {"critical", "fatal", "blocker"},
            }
        )

    critical = [row for row in rows if bool(row.get("blocks_guarded_paper", False))]
    warning = [row for row in rows if not bool(row.get("blocks_guarded_paper", False))]
    return {
        "total_count": len(rows),
        "critical_count": len(critical),
        "warning_count": len(warning),
        "critical_targets": sorted({str(row.get("target") or "") for row in critical if str(row.get("target") or "")}),
        "warning_targets": sorted({str(row.get("target") or "") for row in warning if str(row.get("target") or "")}),
        "rows": rows,
    }


def _storage_ready(storage: dict[str, Any]) -> tuple[bool, list[str]]:
    severity = _status(storage.get("severity") or storage.get("overall_status") or "stable")
    pressure_index = _safe_float(storage.get("pressure_index"), 0.0)
    backpressure = storage.get("backpressure") if isinstance(storage.get("backpressure"), dict) else {}
    effective_raw_live = backpressure.get("effective_raw_live") if isinstance(backpressure.get("effective_raw_live"), dict) else {}
    effective_source = str(backpressure.get("effective_raw_live_source") or effective_raw_live.get("source") or "")
    use_effective = bool(backpressure.get("overlay_adjusted", False) and (backpressure.get("overlay_pressure_clear", False) or effective_source == "fresh_empty_sql_ingestion_overlay"))
    raw_live = effective_raw_live if use_effective and effective_raw_live else backpressure.get("raw_live") if isinstance(backpressure.get("raw_live"), dict) else {}
    raw_core = _safe_int(raw_live.get("core_pending_lines"), 0)
    raw_total = _safe_int(raw_live.get("total_pending_lines"), 0)
    raw_oldest = _safe_float(raw_live.get("oldest_pending_age_seconds"), 0.0)
    overlay_relief = bool(
        backpressure.get("overlay_adjusted", False)
        and raw_live
        and raw_core <= 5000
        and raw_total <= 15000
        and raw_oldest <= 15 * 60
        and _safe_int(backpressure.get("total_pending_lines"), 0) <= 12000
    )
    total_pending = _safe_int(backpressure.get("total_pending_lines"), 0)
    pending_threshold = max(_safe_int(backpressure.get("pending_lines_threshold"), 15000), 1)
    bounded_recovery = _dict(storage.get("bounded_recovery_contract"))
    route = _dict(storage.get("external_route_verification"))
    resilience = _dict(storage.get("storage_resilience"))
    integrity = _dict(storage.get("data_integrity"))
    writer_shedding = _dict(storage.get("writer_shedding"))
    efficiency = _dict(storage.get("storage_efficiency_contract"))
    storage_section = _dict(storage.get("storage"))
    efficiency_status = _status(efficiency.get("overall_status") or storage.get("storage_efficiency_status"))
    efficiency_grade = str(efficiency.get("grade") or storage_section.get("efficiency_grade") or storage.get("storage_efficiency_grade") or "").strip().upper()
    route_ready = _status(route.get("verification_state")) in {"ready", "verified", "ok"}
    resilience_ready = _status(resilience.get("overall_status")) in {"", "ready", "ok"}
    integrity_clean = all(
        _safe_int(integrity.get(key), 0) == 0
        for key in ("sql_invalid_lines", "sql_overlay_invalid_lines", "sql_overlay_oversize_payloads", "sql_overlay_ops_write_failures")
    )
    no_queue_breaches = not writer_shedding.get("hard_breaches") and not writer_shedding.get("elevated_breaches")
    bounded_drain_relief = bool(
        severity in {"stable", "low", "normal", "ready", "watch", "elevated"}
        and pressure_index <= 1.05
        and raw_core <= 5000
        and raw_total <= 10000
        and raw_oldest <= 300.0
        and total_pending < pending_threshold
        and bool(bounded_recovery.get("active_drain_progress") or bounded_recovery.get("drain_delta_signal_observed"))
        and not bool(bounded_recovery.get("hard_gate_active"))
        and not bool(bounded_recovery.get("effective_hard_gate_active"))
        and route_ready
        and resilience_ready
        and integrity_clean
        and no_queue_breaches
        and efficiency_status in {"", "ready", "ok"}
        and efficiency_grade in {"", "A", "A+"}
    )
    blockers: list[str] = []
    if severity in {"blocked", "critical", "high"} and not overlay_relief and not bounded_drain_relief:
        blockers.append(f"storage_severity={severity}")
    if pressure_index >= 0.50 and not overlay_relief and not bounded_drain_relief:
        blockers.append("storage_pressure_index_high")
    if total_pending >= pending_threshold:
        blockers.append("storage_pending_above_threshold")
    return not blockers, blockers


def _collection_rollup_advisory_ready(rollup: dict[str, Any], rollup_status: str) -> bool:
    return bool(
        rollup_status == "degraded"
        and _safe_int(rollup.get("collector_count"), 0) > 0
        and _safe_int(rollup.get("bots_with_observations"), 0) > 0
        and _safe_int(rollup.get("total_observations"), 0) > 0
        and _safe_int(rollup.get("training_ready_count"), 0) == 0
    )


def _platform_repair_contract(
    *,
    platform: dict[str, Any],
    brain_v5: dict[str, Any],
    stabilizer: dict[str, Any],
    settlement: dict[str, Any],
    architecture: dict[str, Any] | None = None,
    plumbing: dict[str, Any] | None = None,
) -> dict[str, Any]:
    sources = {
        "platform_intelligence": platform,
        "platform_brain_v5": brain_v5,
        "platform_stabilization_quality": stabilizer,
        "platform_settlement_stabilization": settlement,
    }
    if architecture:
        sources["system_architecture_hardening"] = architecture
    if plumbing:
        sources["system_plumbing_control"] = plumbing
    issues: list[dict[str, Any]] = []
    for name, payload in sources.items():
        status = _status(payload.get("overall_status"))
        if status in {"blocked", "critical", "needs_work", "degraded"}:
            command = payload.get("next_best_command", "")
            issues.append({"source": name, "overall_status": status, "next_best_command": command})
    return {
        "ok": not issues,
        "status": "ready" if not issues else "needs_work",
        "issue_count": len(issues),
        "issues": issues,
        "blocks_guarded_paper": False,
    }


def _collector_repair_contract(alert_summary: dict[str, Any], isolated_contract: dict[str, Any]) -> dict[str, Any]:
    warning_count = _safe_int(alert_summary.get("warning_count"), 0)
    critical_count = _safe_int(alert_summary.get("critical_count"), 0)
    isolated_count = _safe_int(isolated_contract.get("isolated_count"), 0)
    execution_blocking_count = _safe_int(isolated_contract.get("execution_blocking_count"), 0)
    isolated_targets = [str(item) for item in _as_list(isolated_contract.get("isolated_targets")) if str(item).strip()]
    isolated_target_set = set(isolated_targets)
    rows = [row if isinstance(row, dict) else {} for row in _as_list(alert_summary.get("rows"))]
    warning_rows = [row for row in rows if not bool(row.get("blocks_guarded_paper", False))]
    isolated_warning_events = {
        "watchdog_restart_budget_exhausted_isolated",
        "watchdog_restart_storm_isolated",
    }
    warning_rows_cover_count = bool(warning_rows) and len(warning_rows) >= warning_count
    warning_rows_are_isolated = all(
        str(row.get("target") or "").strip() in isolated_target_set
        and (
            str(row.get("event") or "").strip() in isolated_warning_events
            or str(row.get("event") or "").strip().endswith("_isolated")
        )
        for row in warning_rows
    )
    managed_isolated = bool(
        warning_count > 0
        and critical_count == 0
        and isolated_count > 0
        and execution_blocking_count == 0
        and warning_rows_cover_count
        and warning_rows_are_isolated
    )
    ready = bool(warning_count == 0 and isolated_count == 0 and critical_count == 0 and execution_blocking_count == 0)
    ok = bool(ready or managed_isolated)
    return {
        "ok": ok,
        "status": "ready" if ready else ("managed_isolated" if managed_isolated else "needs_repair"),
        "warning_alert_count": warning_count,
        "critical_alert_count": critical_count,
        "isolated_restart_storm_count": isolated_count,
        "execution_blocking_restart_storm_count": execution_blocking_count,
        "isolated_targets": isolated_targets,
        "managed_isolated": managed_isolated,
        "warning_rows_cover_count": warning_rows_cover_count,
        "warning_rows_are_isolated": warning_rows_are_isolated,
        "blocks_guarded_paper": False,
        "blocks_strict_clear": not ok,
        "policy": "read_only_isolated_collector_budget_or_restart_debt_does_not_block_strict_clear",
    }


def _all_sleeves_effective_runtime(process: dict[str, Any]) -> dict[str, Any]:
    rows = [row if isinstance(row, dict) else {} for row in _as_list(process.get("status"))]
    for row in rows:
        if str(row.get("name") or "") != "all_sleeves":
            continue
        child_fanout = row.get("child_fanout") if isinstance(row.get("child_fanout"), dict) else {}
        launcher_artifact = (
            row.get("launcher_artifact_health")
            if isinstance(row.get("launcher_artifact_health"), dict)
            else {}
        )
        child_count = _safe_int(child_fanout.get("child_process_count"), _safe_int(row.get("alt_running"), 0))
        effective_live = bool(
            row.get("effective_process_live", False)
            or row.get("process_live", False)
            or row.get("launcher_live", False)
        )
        heartbeat_ok = bool(row.get("heartbeat_ok", False))
        child_fanout_ok = bool(row.get("child_fanout_ok", child_fanout.get("ok", True)))
        ok = bool(effective_live and heartbeat_ok and child_fanout_ok and child_count > 0)
        return {
            "ok": ok,
            "status": "ready" if ok else "needs_repair",
            "effective_live": effective_live,
            "launcher_live": bool(row.get("launcher_live", False)),
            "child_process_live": bool(row.get("child_process_live", False)),
            "child_process_count": child_count,
            "child_fanout_ok": child_fanout_ok,
            "heartbeat_ok": heartbeat_ok,
            "launcher_artifact_certified_fanout": bool(row.get("launcher_artifact_certified_fanout", False)),
            "launcher_artifact_reason": str(launcher_artifact.get("reason") or ""),
            "process_live_reason": str(row.get("process_live_reason") or ""),
            "policy": "surface all-sleeves fanout reconciliation without changing guarded-paper or live-execution gates",
        }
    return {
        "ok": False,
        "status": "missing",
        "effective_live": False,
        "launcher_live": False,
        "child_process_live": False,
        "child_process_count": 0,
        "child_fanout_ok": False,
        "heartbeat_ok": False,
        "launcher_artifact_certified_fanout": False,
        "launcher_artifact_reason": "all_sleeves_row_missing",
        "process_live_reason": "",
        "policy": "surface all-sleeves fanout reconciliation without changing guarded-paper or live-execution gates",
    }


def build_payload(project_root: Path = PROJECT_ROOT) -> dict[str, Any]:
    process = _health(project_root, "process_watchdog_latest.json")
    runtime = _health(project_root, "runtime_throttle_control_latest.json")
    memory = _health(project_root, "memory_efficiency_control_latest.json")
    swap = _health(project_root, "swap_pressure_governor_latest.json")
    pressure = _health(project_root, "pressure_relief_control_latest.json")
    platform = _health(project_root, "platform_intelligence_expansion_latest.json")
    brain_v4 = _health(project_root, "platform_brain_v4_latest.json")
    brain_v5 = _health(project_root, "platform_brain_v5_latest.json")
    stabilizer = _health(project_root, "platform_stabilization_quality_latest.json")
    settlement = _health(project_root, "platform_settlement_stabilization_latest.json")
    architecture = _health(project_root, "system_architecture_hardening_latest.json")
    plumbing = _health(project_root, "system_plumbing_control_latest.json")
    rollup = _health(project_root, "data_collection_observation_rollup_latest.json")
    halt = _health(project_root, "global_halt_auto_clear_latest.json") or _health(project_root, "global_killswitch_latest.json")
    storage = _health(project_root, "ingestion_storage_control_latest.json")
    paper_ramp = _health(project_root, "paper_400_ramp_latest.json")
    schwab_futures = _health(project_root, "data_ingress_latest_schwab_futures_equities_schwab.json")
    done_for_today = _health(project_root, "system_done_for_today_latest.json")
    use_mode = _health(project_root, "use_mode_compliance_guard_latest.json")
    alerts = process.get("alerts") if isinstance(process.get("alerts"), list) else []
    alert_summary = _alert_details(alerts)
    safety = process.get("safety_pause") if isinstance(process.get("safety_pause"), dict) else {}
    swap_payload = swap.get("swap_pressure") if isinstance(swap.get("swap_pressure"), dict) else {}
    storage_ready, storage_blockers = _storage_ready(storage)
    halt_blockers = [str(item) for item in _as_list(halt.get("clear_blockers")) if str(item).strip()]
    runtime_status = _status(runtime.get("overall_status"))
    memory_status = _status(memory.get("overall_status"))
    rollup_status = _status(rollup.get("overall_status"))
    collection_advisory_ready = _collection_rollup_advisory_ready(rollup, rollup_status)
    swap_tier = _status(swap_payload.get("tier") or "normal")
    paper_ramp_blockers = [str(item) for item in _as_list(paper_ramp.get("blockers")) if str(item).strip()]
    paper_ramp_stage = _status(paper_ramp.get("stage"))
    paper_ramp_present = bool(paper_ramp)
    paper_ramp_gates = paper_ramp.get("gates") if isinstance(paper_ramp.get("gates"), dict) else {}
    paper_runtime_gate = paper_ramp_gates.get("runtime") if isinstance(paper_ramp_gates.get("runtime"), dict) else {}
    paper_memory_gate = paper_ramp_gates.get("memory") if isinstance(paper_ramp_gates.get("memory"), dict) else {}
    paper_global_halt_gate = paper_ramp.get("gates", {}).get("global_halt") if isinstance(paper_ramp.get("gates"), dict) else {}
    paper_clear_relief = (
        paper_global_halt_gate.get("clear_blocker_relief")
        if isinstance(paper_global_halt_gate, dict) and isinstance(paper_global_halt_gate.get("clear_blocker_relief"), dict)
        else {}
    )
    advisory_halt_blockers = {
        str(item)
        for item in _as_list(paper_clear_relief.get("clear_blockers"))
        if str(item).strip()
    } if bool(paper_clear_relief.get("active", False)) and bool(paper_global_halt_gate.get("ok", False)) else set()
    blocking_halt_blockers = [item for item in halt_blockers if item not in advisory_halt_blockers]
    paper_ramp_stale_global_blocker = bool(
        paper_ramp_present
        and paper_ramp_stage not in {"armed", "ready"}
        and set(paper_ramp_blockers) == {"global_halt_or_clear_blocker_active"}
        and not bool(halt.get("halt", False))
        and not blocking_halt_blockers
    )
    isolated_contract = process.get("restart_storm_isolation") if isinstance(process.get("restart_storm_isolation"), dict) else {}
    platform_repair = _platform_repair_contract(
        platform=platform,
        brain_v5=brain_v5,
        stabilizer=stabilizer,
        settlement=settlement,
        architecture=architecture,
        plumbing=plumbing,
    )
    plumbing_sections = plumbing.get("sections") if isinstance(plumbing.get("sections"), dict) else {}
    plumbing_runtime = plumbing_sections.get("runtime_memory") if isinstance(plumbing_sections.get("runtime_memory"), dict) else {}
    plumbing_runtime_memory_relief = bool(
        _status(plumbing.get("overall_status")) in {"ready", "guarded_ready", "advisory"}
        and bool(plumbing_runtime.get("ok", False))
        and bool(plumbing_runtime.get("paper_only_runtime_memory_relief", False))
    )
    paper_ramp_runtime_ready = bool(paper_runtime_gate.get("ok", False))
    paper_ramp_memory_ready = bool(paper_memory_gate.get("ok", False))
    runtime_guarded_ok = bool(
        runtime_status in {"ready", "advisory", "guarded_ready"}
        or plumbing_runtime_memory_relief
        or paper_ramp_runtime_ready
    )
    memory_guarded_ok = bool(
        memory_status in {"ready", "advisory", "guarded_ready"}
        or plumbing_runtime_memory_relief
        or paper_ramp_memory_ready
    )

    guarded_paper_blockers: list[str] = []
    if bool(safety.get("active", False)):
        guarded_paper_blockers.append("process_safety_pause_active")
    if bool(halt.get("halt", False)):
        guarded_paper_blockers.append("global_halt_active")
    guarded_paper_blockers.extend(f"global_clear_blocker={item}" for item in blocking_halt_blockers)
    if int(alert_summary["critical_count"]) > 0:
        guarded_paper_blockers.append("critical_process_alerts_active")
    if not runtime_guarded_ok:
        guarded_paper_blockers.append(f"runtime_status={runtime_status or 'missing'}")
    if not memory_guarded_ok:
        guarded_paper_blockers.append(f"memory_status={memory_status or 'missing'}")
    if rollup_status not in {"ready", ""} and not collection_advisory_ready:
        guarded_paper_blockers.append(f"collection_status={rollup_status}")
    if swap_tier not in {"normal", "calm", ""}:
        guarded_paper_blockers.append(f"swap_tier={swap_tier}")
    guarded_paper_blockers.extend(storage_blockers)
    if plumbing and _status(plumbing.get("overall_status")) in {"blocked", "critical"}:
        guarded_paper_blockers.append("system_plumbing_blocked")
    if paper_ramp_present and (paper_ramp_stage not in {"armed", "ready"} or paper_ramp_blockers) and not paper_ramp_stale_global_blocker:
        guarded_paper_blockers.append("paper_ramp_not_armed")

    guarded_paper_ready = not guarded_paper_blockers
    collector_repair = _collector_repair_contract(alert_summary, isolated_contract)
    strict_ready = bool(guarded_paper_ready and collector_repair["ok"] and platform_repair["ok"])
    overall_status = "ready" if strict_ready else ("guarded_ready" if guarded_paper_ready else "degraded")
    all_sleeves_effective_runtime = _all_sleeves_effective_runtime(process)
    return {
        "timestamp_utc": iso_now(),
        "schema_version": 1,
        "ok": guarded_paper_ready,
        "overall_status": overall_status,
        "strict_all_clear": strict_ready,
        "repair_backlog_active": not strict_ready,
        "read_only": True,
        "started_heavy_reports": False,
        "operational_readiness": {
            "guarded_paper": {
                "ok": guarded_paper_ready,
                "status": "ready" if guarded_paper_ready else "blocked",
                "blockers": guarded_paper_blockers,
                "paper_ramp_stage": paper_ramp_stage,
                "paper_ramp_blockers": paper_ramp_blockers,
                "advisory_clear_blockers": sorted(advisory_halt_blockers),
                "collection_advisory_ready": collection_advisory_ready,
                "runtime_memory_advisory_relief": plumbing_runtime_memory_relief,
                "paper_ramp_runtime_ready": paper_ramp_runtime_ready,
                "paper_ramp_memory_ready": paper_ramp_memory_ready,
                "paper_ramp_stale_global_blocker_ignored": paper_ramp_stale_global_blocker,
                "policy": "guarded paper readiness consumes paper-ramp and system-plumbing advisory contracts while live execution stays locked",
            },
            "live_execution": {
                "ok": False,
                "status": "blocked_read_only",
                "blockers": ["health_fast_is_read_only", "live_execution_requires_explicit_operator_control"],
                "policy": "never infer live readiness from fast read-only health",
            },
            "collector_repair": collector_repair,
            "platform_repair": platform_repair,
        },
        "global_halt": {
            "halt": bool(halt.get("halt", False)),
            "halt_state": halt.get("halt_state", "unknown"),
            "clear_blockers": halt.get("clear_blockers", []),
        },
        "process_watchdog": {
            "alerts": alerts,
            "alert_summary": alert_summary,
            "restart_storm_isolation": isolated_contract,
            "safety_pause": safety,
            "all_sleeves_effective_runtime": all_sleeves_effective_runtime,
            "status": process.get("status", []),
        },
        "runtime_pressure": {
            "overall_status": runtime.get("overall_status"),
            "tier": pressure.get("tier"),
            "host_saturation_score": _safe_float(runtime.get("host_saturation_score"), 0.0),
            "compute_pressure_level": runtime.get("compute_pressure_level"),
            "memory_pressure_level": runtime.get("memory_pressure_level"),
        },
        "memory": {
            "overall_status": memory.get("overall_status"),
            "recommended_profile": memory.get("recommended_profile"),
            "swap_tier": swap_payload.get("tier", "unknown"),
            "swap_used_gb": _safe_float(swap_payload.get("swap_used_gb"), 0.0),
        },
        "storage": {
            "severity": storage.get("severity"),
            "pressure_index": _safe_float(storage.get("pressure_index"), 0.0),
            "backpressure": storage.get("backpressure") if isinstance(storage.get("backpressure"), dict) else {},
        },
        "collection": {
            "overall_status": rollup.get("overall_status"),
            "collector_count": _safe_int(rollup.get("collector_count"), 0),
            "bots_with_observations": _safe_int(rollup.get("bots_with_observations"), 0),
            "effective_bots_with_observations": _safe_int(
                rollup.get("effective_bots_with_observations", rollup.get("bots_with_observations")),
                0,
            ),
            "zero_observation_count": _safe_int(rollup.get("zero_observation_count"), 0),
            "unmanaged_zero_observation_count": _safe_int(
                rollup.get("unmanaged_zero_observation_count", rollup.get("zero_observation_count")),
                0,
            ),
            "managed_zero_observation_count": _safe_int(rollup.get("managed_zero_observation_count"), 0),
            "raw_zero_observation_count": _safe_int(
                rollup.get("raw_zero_observation_count", rollup.get("zero_observation_count")),
                0,
            ),
            "total_observations": _safe_int(rollup.get("total_observations"), 0),
            "training_ready_count": _safe_int(rollup.get("training_ready_count"), 0),
            "advisory_ready": collection_advisory_ready,
        },
        "platform_intelligence": {
            "overall_status": platform.get("overall_status"),
            "expansion_count": _safe_int(platform.get("expansion_count"), 0),
            "control_count": _safe_int(platform.get("control_count"), 0),
        },
        "platform_brain_v4": {
            "overall_status": brain_v4.get("overall_status"),
            "section_count": _safe_int(brain_v4.get("section_count"), 0),
            "control_count": _safe_int(brain_v4.get("control_count"), 0),
            "next_best_command": (((brain_v4.get("sections") or {}).get("executive_meta_orchestrator") or {}).get("next_best_command") if isinstance(brain_v4.get("sections"), dict) else ""),
        },
        "platform_brain_v5": {
            "overall_status": brain_v5.get("overall_status"),
            "section_count": _safe_int(brain_v5.get("section_count"), 0),
            "control_count": _safe_int(brain_v5.get("control_count"), 0),
            "next_best_command": (((brain_v5.get("sections") or {}).get("reflex_action_router") or {}).get("next_best_command") if isinstance(brain_v5.get("sections"), dict) else ""),
        },
        "platform_stabilization_quality": {
            "overall_status": stabilizer.get("overall_status"),
            "section_count": _safe_int(stabilizer.get("section_count"), 0),
            "control_count": _safe_int(stabilizer.get("control_count"), 0),
            "next_best_command": stabilizer.get("next_best_command", ""),
            "expansion_allowed_now": (((stabilizer.get("sections") or {}).get("expansion_rehearsal_gate") or {}).get("expansion_allowed_now") if isinstance(stabilizer.get("sections"), dict) else None),
        },
        "platform_settlement_stabilization": {
            "overall_status": settlement.get("overall_status"),
            "section_count": _safe_int(settlement.get("section_count"), 0),
            "control_count": _safe_int(settlement.get("control_count"), 0),
            "next_best_command": settlement.get("next_best_command", ""),
            "queue_backpressure_active": (((settlement.get("sections") or {}).get("queue_decay_meter") or {}).get("queue_backpressure_active") if isinstance(settlement.get("sections"), dict) else None),
            "global_clear_status": (((settlement.get("sections") or {}).get("global_clear_settlement_guard") or {}).get("overall_status") if isinstance(settlement.get("sections"), dict) else None),
        },
        "system_architecture_hardening": {
            "overall_status": architecture.get("overall_status"),
            "section_count": _safe_int(architecture.get("section_count"), 0),
            "hard_section_count": _safe_int(architecture.get("hard_section_count"), 0),
            "watch_section_count": _safe_int(architecture.get("watch_section_count"), 0),
            "next_best_command": architecture.get("next_best_command", ""),
        },
        "system_plumbing_control": {
            "overall_status": plumbing.get("overall_status"),
            "plumbing_score": _safe_int(plumbing.get("plumbing_score"), 0),
            "blockers": plumbing.get("blockers", []),
            "warnings": plumbing.get("warnings", []),
            "root_cause": plumbing.get("root_cause", {}) if isinstance(plumbing.get("root_cause"), dict) else {},
            "next_best_command": plumbing.get("next_best_command", ""),
            "global_clear_relief": plumbing.get("global_clear_relief", {}) if isinstance(plumbing.get("global_clear_relief"), dict) else {},
        },
        "schwab_futures": {
            "loop_state": schwab_futures.get("loop_state"),
            "pause_gate": schwab_futures.get("pause_gate"),
            "pause_reason": schwab_futures.get("pause_reason"),
            "total_counts": schwab_futures.get("total_counts", {}),
        },
        "done_for_today": {
            "overall_status": done_for_today.get("overall_status"),
            "can_stop_chasing": bool(done_for_today.get("can_stop_chasing", False)),
            "blockers": done_for_today.get("blockers", []),
            "next_command": ["./scripts/ops/opsctl.sh", "done-for-today", "--json"],
        },
        "use_mode_compliance": {
            "overall_status": use_mode.get("overall_status") or "missing",
            "use_mode": use_mode.get("use_mode") or "personal",
            "personal_grade": _dict(use_mode.get("personal_use")).get("grade"),
            "perfect_personal_use_ready": bool(_dict(use_mode.get("personal_use")).get("perfect_personal_use_ready", False)),
            "commercial_use_intent_detected": bool(_dict(use_mode.get("commercial_use")).get("commercial_use_intent_detected", False)),
            "commercial_clearance_status": _dict(use_mode.get("commercial_use")).get("commercial_clearance_status"),
            "commercial_blockers": _dict(use_mode.get("commercial_use")).get("blockers", []),
            "live_execution_authority": bool(_dict(use_mode.get("authority_boundaries")).get("live_execution_authority", False)),
            "policy": "context_only_for_health_fast_guarded_paper; live_canary_readiness_consumes_as_hard_boundary",
            "next_command": ["./scripts/ops/opsctl.sh", "use-mode-compliance", "--json"],
        },
        "recommended_commands": [
            *([["./scripts/ops/opsctl.sh", "coinbase-api-health", "--snapshot", "--json"]] if collector_repair["status"] == "needs_repair" else []),
            *([["./scripts/ops/opsctl.sh", "pressure-relief", "--apply", "--json"]] if not platform_repair["ok"] else []),
            *([plumbing.get("root_cause", {}).get("next_command")] if isinstance(plumbing.get("root_cause"), dict) and isinstance(plumbing.get("root_cause", {}).get("next_command"), list) else []),
            *([["./scripts/ops/opsctl.sh", "use-mode-compliance", "--json"]] if not use_mode else []),
            ["./scripts/ops/opsctl.sh", "system-plumbing-control", "--json"],
            ["./scripts/ops/opsctl.sh", "global-halt-status", "--json"],
            ["./scripts/ops/opsctl.sh", "done-for-today", "--json"],
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Fast read-only health summary. Does not refresh reports.")
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--out-file", default=str(DEFAULT_OUT_PATH))
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    project_root = Path(args.project_root).resolve()
    payload = build_payload(project_root)
    write_payload(Path(args.out_file).expanduser(), payload)
    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        print(
            "health_fast "
            f"overall_status={payload.get('overall_status')} "
            f"guarded_paper={int(bool(((payload.get('operational_readiness') or {}).get('guarded_paper') or {}).get('ok')))} "
            f"halt={int(bool((payload.get('global_halt') or {}).get('halt')))} "
            f"collection={((payload.get('collection') or {}).get('overall_status') or 'unknown')} "
            f"pressure={((payload.get('runtime_pressure') or {}).get('tier') or 'unknown')}"
        )
    return 0 if bool(payload.get("ok", False)) else 2


if __name__ == "__main__":
    raise SystemExit(main())
