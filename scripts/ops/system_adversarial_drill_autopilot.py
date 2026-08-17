#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable

if __package__ in {None, ""}:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from scripts.ops.long_runtime_common import PROJECT_ROOT, iso_now, load_json, ordered_unique, payload_age_minutes, write_payload
else:
    from .long_runtime_common import PROJECT_ROOT, iso_now, load_json, ordered_unique, payload_age_minutes, write_payload


DEFAULT_OUT_PATH = PROJECT_ROOT / "governance" / "health" / "system_adversarial_drill_autopilot_latest.json"
DEFAULT_RESULTS_PATH = PROJECT_ROOT / "governance" / "drills" / "system_adversarial_drill_results_latest.json"

Runner = Callable[[list[str], Path, int], dict[str, Any]]

BLOCKED_COMMAND_PATTERNS = ("start-live", "clear-all-halts", "operator-release", "token-refresh-interactive")
READY_STATUSES = {"ready", "ok", "stable", "advisory", "guarded_ready", "clear_ready"}
DEGRADED_STATUSES = {"degraded", "needs_attention", "needs_work", "warning", "warn", "stale"}
BLOCKED_STATUSES = {"blocked", "critical", "failed", "fatal", "missing", "timeout"}


def _as_dict(raw: Any) -> dict[str, Any]:
    return raw if isinstance(raw, dict) else {}


def _as_list(raw: Any) -> list[Any]:
    if isinstance(raw, tuple):
        return list(raw)
    return raw if isinstance(raw, list) else []


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


def _parse_json_output(text: str) -> dict[str, Any]:
    for raw in reversed([line.strip() for line in str(text or "").splitlines() if line.strip()]):
        try:
            payload = json.loads(raw)
        except Exception:
            continue
        if isinstance(payload, dict):
            return payload
    return {}


def _tail_text(text: str, *, max_lines: int = 10, max_chars: int = 2400) -> str:
    tail = "\n".join(str(text or "").splitlines()[-max_lines:])
    if len(tail) <= max_chars:
        return tail
    return "...truncated...\n" + tail[-max_chars:]


def _run(cmd: list[str], project_root: Path, timeout_sec: int) -> dict[str, Any]:
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(project_root),
            capture_output=True,
            text=True,
            check=False,
            timeout=max(int(timeout_sec), 1),
        )
        return {
            "cmd": list(cmd),
            "rc": int(proc.returncode),
            "payload": _parse_json_output(proc.stdout or ""),
            "stdout_tail": _tail_text(proc.stdout or ""),
            "stderr_tail": _tail_text(proc.stderr or ""),
        }
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout.decode("utf-8", errors="ignore") if isinstance(exc.stdout, bytes) else str(exc.stdout or "")
        stderr = exc.stderr.decode("utf-8", errors="ignore") if isinstance(exc.stderr, bytes) else str(exc.stderr or "")
        return {
            "cmd": list(cmd),
            "rc": 124,
            "payload": _parse_json_output(stdout),
            "stdout_tail": _tail_text(stdout),
            "stderr_tail": _tail_text(stderr) or "timeout",
        }


def _safe_command(cmd: list[str]) -> bool:
    joined = " ".join(str(part) for part in cmd)
    return bool(cmd and cmd[0] == "./scripts/ops/opsctl.sh" and not any(pattern in joined for pattern in BLOCKED_COMMAND_PATTERNS))


def _probe_specs() -> list[dict[str, Any]]:
    return [
        {"name": "health_fast", "cmd": ["./scripts/ops/opsctl.sh", "health-fast", "--json"], "timeout_sec": 90},
        {"name": "runtime_paper_guard", "cmd": ["./scripts/ops/opsctl.sh", "runtime-paper-regression-guard", "--json"], "timeout_sec": 90},
        {"name": "incident_closeout", "cmd": ["./scripts/ops/opsctl.sh", "incident-closeout", "--json"], "timeout_sec": 90},
        {"name": "live_canary_control", "cmd": ["./scripts/ops/opsctl.sh", "live-canary-control", "--json"], "timeout_sec": 90},
        {"name": "architecture_scoreboard", "cmd": ["./scripts/ops/opsctl.sh", "architecture-upgrade-scoreboard", "--json"], "timeout_sec": 90},
        {"name": "system_drift_guard", "cmd": ["./scripts/ops/opsctl.sh", "system-drift-guard", "--json"], "timeout_sec": 120},
        {"name": "golden_replay_regression", "cmd": ["./scripts/ops/opsctl.sh", "golden-replay-regression", "--json"], "timeout_sec": 120},
        {"name": "master_infra_supervisor", "cmd": ["./scripts/ops/opsctl.sh", "master-infra-supervisor", "--json"], "timeout_sec": 180},
    ]


def _status(payload: dict[str, Any], *, default: str = "missing") -> str:
    raw = str(payload.get("overall_status") or payload.get("status") or payload.get("state") or "").strip().lower()
    if not raw and "ok" in payload:
        raw = "ready" if bool(payload.get("ok", False)) else "blocked"
    return raw or default


def _artifact_status(payload: dict[str, Any], path: Path, *, max_age_minutes: float | None = None) -> tuple[str, bool]:
    if not payload:
        return "missing", False
    status = _status(payload)
    stale = False
    if max_age_minutes is not None:
        age = payload_age_minutes(payload, path)
        stale = bool(age is not None and age > float(max_age_minutes))
    if stale and status in READY_STATUSES:
        return "stale", False
    return status, status in READY_STATUSES and not stale


def _command_surface_ready(payload: dict[str, Any], path: Path) -> tuple[str, bool, dict[str, Any]]:
    status, artifact_ready = _artifact_status(payload, path, max_age_minutes=45)
    metrics = _as_dict(payload.get("metrics"))
    failure_keys = (
        "blocked_entry_count",
        "degraded_entry_count",
        "smoke_failure_count",
        "runtime_smoke_failure_count",
        "base_runtime_smoke_failure_count",
        "contract_dispatch_smoke_failure_count",
        "commands_hygiene_failure_count",
        "contract_hash_mismatch_count",
    )
    failure_counts = {key: _safe_int(metrics.get(key), 0) for key in failure_keys}
    failure_total = sum(failure_counts.values())
    stale = status == "stale"
    ok = bool(payload.get("ok", False))
    operator_gated_only = ok and not stale and failure_total == 0
    ready = artifact_ready or operator_gated_only
    evidence = {
        "status": status,
        "ok": ok,
        "age_minutes": payload_age_minutes(payload, path),
        "operator_gated_entry_count": _safe_int(metrics.get("operator_gated_entry_count"), 0),
        "failure_total": failure_total,
        **failure_counts,
    }
    return status, ready, evidence


def _weak_point(
    *,
    weak_point_id: str,
    family: str,
    severity: str,
    status: str,
    summary: str,
    evidence: dict[str, Any],
    recommended_commands: list[list[str]],
) -> dict[str, Any]:
    return {
        "weak_point_id": weak_point_id,
        "family": family,
        "severity": severity,
        "status": status,
        "summary": summary,
        "evidence": evidence,
        "recommended_commands": [list(cmd) for cmd in recommended_commands if _safe_command([str(part) for part in cmd])],
    }


def _rank_severity(severity: str) -> int:
    return {"critical": 0, "high": 1, "medium": 2, "low": 3, "info": 4}.get(str(severity), 5)


def _evaluate(project_root: Path) -> list[dict[str, Any]]:
    health = project_root / "governance" / "health"
    champion = project_root / "governance" / "champion_challenger"

    health_fast = load_json(health / "health_fast_latest.json")
    runtime = load_json(health / "runtime_throttle_control_latest.json")
    memory = load_json(health / "memory_efficiency_control_latest.json")
    storage = load_json(health / "ingestion_storage_control_latest.json")
    storage_clearance = load_json(health / "storage_pressure_clearance_latest.json")
    runtime_paper = load_json(health / "runtime_paper_regression_guard_latest.json")
    drift = load_json(health / "system_drift_guard_latest.json")
    contract_graph = load_json(health / "system_architecture_contract_graph_latest.json")
    architecture = load_json(health / "architecture_upgrade_scoreboard_latest.json")
    incident_closeout = load_json(health / "incident_closeout_autopilot_latest.json")
    live_canary = load_json(health / "live_canary_control_latest.json")
    master_infra = load_json(health / "master_infrastructure_supervisor_latest.json")
    commands = load_json(health / "command_validity_latest.json")
    golden_replay = load_json(health / "golden_replay_regression_latest.json")
    replay_hash = load_json(health / "replay_hash_registry_guard_latest.json")
    point_in_time = load_json(health / "point_in_time_event_store_latest.json")
    promotion = load_json(champion / "promotion_autopilot_packet_latest.json")
    operator_cockpit = load_json(health / "operator_cockpit_latest.json")

    weak: list[dict[str, Any]] = []

    guarded_paper = _as_dict(_as_dict(health_fast.get("operational_readiness")).get("guarded_paper"))
    paper_blockers = [str(item) for item in _as_list(guarded_paper.get("blockers"))]
    runtime_status = _status(runtime)
    memory_status = _status(memory)
    if str(guarded_paper.get("status") or "").strip().lower() == "blocked" or paper_blockers:
        weak.append(
            _weak_point(
                weak_point_id="guarded_paper_pressure_coupling",
                family="paper_runtime",
                severity="critical" if "runtime_status=degraded" in paper_blockers else "high",
                status=str(guarded_paper.get("status") or "blocked"),
                summary=f"guarded paper is blocked by {', '.join(paper_blockers) or 'unknown pressure'}",
                evidence={"paper_blockers": paper_blockers, "runtime_status": runtime_status, "memory_status": memory_status},
                recommended_commands=[
                    ["./scripts/ops/opsctl.sh", "runtime-throttle", "--apply", "--max-renice-processes", "30", "--json"],
                    ["./scripts/ops/opsctl.sh", "memory-efficiency", "apply", "--json"],
                    ["./scripts/ops/opsctl.sh", "pressure-relief", "--apply", "--json"],
                ],
            )
        )

    runtime_measurements = _as_dict(_as_dict(runtime.get("soft_cap_advisory_reclassification")).get("measurements"))
    if not runtime_measurements:
        runtime_measurements = _as_dict(runtime.get("measurements"))
    storage_writer_cpu = _safe_float(runtime_measurements.get("storage_writer_cpu_percent"), 0.0)
    if storage_writer_cpu >= 150.0 or bool(runtime_measurements.get("storage_writer_hot", False)):
        weak.append(
            _weak_point(
                weak_point_id="sql_writer_heat",
                family="storage_runtime",
                severity="high",
                status="degraded",
                summary=f"SQL writer heat is high at {storage_writer_cpu:.1f}% CPU",
                evidence={
                    "storage_writer_cpu_percent": round(storage_writer_cpu, 2),
                    "host_saturation_score": _safe_float(runtime.get("host_saturation_score"), 0.0),
                    "throttle_profile": str(runtime.get("throttle_profile") or ""),
                },
                recommended_commands=[
                    ["./scripts/ops/opsctl.sh", "runtime-throttle", "--apply", "--max-renice-processes", "30", "--json"],
                    ["./scripts/ops/opsctl.sh", "pressure-relief", "--apply", "--json"],
                ],
            )
        )

    storage_status = _status(storage)
    storage_quality = _safe_float(storage.get("backpressure_quality_score"), 100.0)
    storage_pressure = _safe_float(storage.get("pressure_index"), 0.0)
    clearance_metrics = _as_dict(storage_clearance.get("metrics"))
    if storage_status not in READY_STATUSES or storage_quality < 88.0 or bool(clearance_metrics.get("active_storage_pressure", False)):
        weak.append(
            _weak_point(
                weak_point_id="raw_live_storage_headroom",
                family="storage_runtime",
                severity="high" if bool(clearance_metrics.get("active_storage_pressure", False)) else "medium",
                status=storage_status,
                summary=f"raw-live storage headroom is thin: pressure_index={storage_pressure:.3f} quality={storage_quality:.2f}",
                evidence={
                    "pressure_index": storage_pressure,
                    "backpressure_quality_score": storage_quality,
                    "active_storage_pressure": bool(clearance_metrics.get("active_storage_pressure", False)),
                    "core_pending_lines": _safe_int(clearance_metrics.get("core_pending_lines"), 0),
                    "total_pending_lines": _safe_int(clearance_metrics.get("total_pending_lines"), 0),
                },
                recommended_commands=[
                    ["./scripts/ops/opsctl.sh", "storage-pressure-clearance", "--apply", "--force-clear-stale-gate", "--json"],
                    ["./scripts/ops/opsctl.sh", "ingestion-storage-control", "--json"],
                ],
            )
        )

    if _status(incident_closeout) != "ready":
        critical_blockers = [
            row for row in _as_list(incident_closeout.get("blocking_surfaces"))
            if isinstance(row, dict) and str(row.get("severity") or "").strip().lower() == "critical"
        ]
        bounded = bool(incident_closeout.get("bounded_closeout_path_ready", False))
        weak.append(
            _weak_point(
                weak_point_id="bounded_incident_closeout",
                family="governance",
                severity="high" if critical_blockers else "medium",
                status=_status(incident_closeout),
                summary=(
                    "incident closeout still has critical blockers"
                    if critical_blockers
                    else "incident closeout is bounded but not archived"
                ),
                evidence={
                    "open_incident_count": _safe_int(incident_closeout.get("open_incident_count"), 0),
                    "bounded_closeout_path_ready": bounded,
                    "critical_blocker_count": len(critical_blockers),
                    "closeout_score": _safe_float(incident_closeout.get("closeout_score"), 0.0),
                },
                recommended_commands=[["./scripts/ops/opsctl.sh", "incident-closeout", "--json"], ["./scripts/ops/opsctl.sh", "incident-report", "--json"]],
            )
        )

    if not bool(live_canary.get("supervised_canary_ready", False)):
        weak.append(
            _weak_point(
                weak_point_id="live_canary_validate_only",
                family="governance",
                severity="medium" if bool(live_canary.get("preapproved_supervised_ready", False)) else "high",
                status=_status(live_canary),
                summary=f"live canary is {str(live_canary.get('recommended_mode') or 'unknown')}, not supervised",
                evidence={
                    "recommended_mode": str(live_canary.get("recommended_mode") or ""),
                    "preclearance_score": _safe_float(live_canary.get("preclearance_score"), 0.0),
                    "blocking_reasons": [str(item) for item in _as_list(live_canary.get("blocking_reasons"))],
                },
                recommended_commands=[["./scripts/ops/opsctl.sh", "live-canary-control", "--json"], ["./scripts/ops/opsctl.sh", "promotion-autopilot", "--json"]],
            )
        )

    drift_status = _status(drift)
    if drift_status in BLOCKED_STATUSES or _safe_int(_as_dict(drift.get("metrics")).get("stale_surface_count"), 0) > 0:
        metrics = _as_dict(drift.get("metrics"))
        weak.append(
            _weak_point(
                weak_point_id="artifact_drift_mesh",
                family="architecture",
                severity="high" if drift_status in BLOCKED_STATUSES else "medium",
                status=drift_status,
                summary=(
                    f"drift mesh has blocked={_safe_int(metrics.get('blocked_surface_count'), 0)} "
                    f"stale={_safe_int(metrics.get('stale_surface_count'), 0)}"
                ),
                evidence={
                    "blocked_surface_count": _safe_int(metrics.get("blocked_surface_count"), 0),
                    "degraded_surface_count": _safe_int(metrics.get("degraded_surface_count"), 0),
                    "stale_surface_count": _safe_int(metrics.get("stale_surface_count"), 0),
                },
                recommended_commands=[["./scripts/ops/opsctl.sh", "system-drift-guard", "--json"], ["./scripts/ops/opsctl.sh", "system-drift-autopilot", "--apply", "--json"]],
            )
        )

    if _status(contract_graph) in BLOCKED_STATUSES or _status(architecture) in BLOCKED_STATUSES | DEGRADED_STATUSES:
        weak.append(
            _weak_point(
                weak_point_id="architecture_contract_pressure",
                family="architecture",
                severity="high" if _status(contract_graph) in BLOCKED_STATUSES else "medium",
                status=_status(contract_graph),
                summary="architecture contract graph still carries blocked or degraded proof surfaces",
                evidence={
                    "contract_graph_status": _status(contract_graph),
                    "blocked_node_count": _safe_int(contract_graph.get("blocked_node_count"), 0),
                    "degraded_node_count": _safe_int(contract_graph.get("degraded_node_count"), 0),
                    "architecture_scoreboard_status": _status(architecture),
                    "architecture_ready_count": _safe_int(architecture.get("ready_count"), 0),
                },
                recommended_commands=[
                    ["./scripts/ops/opsctl.sh", "system-architecture-contract-graph", "--apply", "--json"],
                    ["./scripts/ops/opsctl.sh", "architecture-upgrade-scoreboard", "--json"],
                ],
            )
        )

    master_status = _status(master_infra)
    if master_status in BLOCKED_STATUSES | DEGRADED_STATUSES:
        weak.append(
            _weak_point(
                weak_point_id="self_auditing_infra_bots",
                family="infrastructure",
                severity="high" if master_status in BLOCKED_STATUSES else "medium",
                status=master_status,
                summary="self-auditing infrastructure bots still report blocked or degraded lanes",
                evidence={
                    "blocked_check_count": _safe_int(_as_dict(master_infra.get("metrics")).get("blocked_check_count"), 0),
                    "degraded_check_count": _safe_int(_as_dict(master_infra.get("metrics")).get("degraded_check_count"), 0),
                    "platform_posture": str(_as_dict(master_infra.get("platform_posture")).get("operating_posture") or ""),
                },
                recommended_commands=[["./scripts/ops/opsctl.sh", "master-infra-supervisor", "--json"], ["./scripts/ops/opsctl.sh", "infrastructure-autofix", "--json"]],
            )
        )

    command_status, command_ready, command_evidence = _command_surface_ready(commands, health / "command_validity_latest.json")
    if not command_ready:
        weak.append(
            _weak_point(
                weak_point_id="command_surface_freshness",
                family="command_surface",
                severity="medium",
                status=command_status,
                summary="command validity is stale or degraded",
                evidence=command_evidence,
                recommended_commands=[["./scripts/ops/opsctl.sh", "command-validity", "--apply", "--json"], ["./scripts/ops/opsctl.sh", "commands-hygiene", "--apply", "--json"]],
            )
        )

    replay_surfaces = {
        "golden_replay": _status(golden_replay),
        "replay_hash": _status(replay_hash),
        "point_in_time": _status(point_in_time),
    }
    if any(status not in READY_STATUSES for status in replay_surfaces.values()):
        weak.append(
            _weak_point(
                weak_point_id="replay_provenance_chain",
                family="replay",
                severity="high",
                status="degraded",
                summary="one or more replay provenance guards is not ready",
                evidence=replay_surfaces,
                recommended_commands=[["./scripts/ops/opsctl.sh", "replay-hash-registry", "--json"], ["./scripts/ops/opsctl.sh", "golden-replay-regression", "--json"]],
            )
        )

    repairable_gate_count = _safe_int(promotion.get("repairable_gate_count"), 0)
    if repairable_gate_count > 0 or not bool(promotion.get("promotion_ready", False)):
        weak.append(
            _weak_point(
                weak_point_id="promotion_packet_repairable_gates",
                family="promotion",
                severity="medium" if _safe_int(_as_dict(promotion.get("readiness_repair_contract")).get("critical_repair_gate_count"), 0) == 0 else "high",
                status=_status(promotion),
                summary=f"promotion packet has {repairable_gate_count} repairable gate(s)",
                evidence={
                    "packet_completeness_score": _safe_float(promotion.get("packet_completeness_score"), 0.0),
                    "repairable_gate_count": repairable_gate_count,
                    "blockers": [str(item) for item in _as_list(promotion.get("blockers"))],
                },
                recommended_commands=[["./scripts/ops/opsctl.sh", "promotion-autopilot", "--json"], ["./scripts/ops/opsctl.sh", "promotion-quality-gate", "--json"]],
            )
        )

    if _status(operator_cockpit) in BLOCKED_STATUSES | DEGRADED_STATUSES:
        weak.append(
            _weak_point(
                weak_point_id="operator_cockpit_readiness",
                family="operator_surface",
                severity="medium",
                status=_status(operator_cockpit),
                summary="operator cockpit is not fully ready",
                evidence={"overall_status": _status(operator_cockpit)},
                recommended_commands=[["./scripts/ops/opsctl.sh", "operator-cockpit", "--json"], ["./scripts/ops/opsctl.sh", "dashboard-refresh", "--json"]],
            )
        )

    if not bool(runtime_paper.get("ok", False)):
        weak.append(
            _weak_point(
                weak_point_id="runtime_paper_guard",
                family="paper_runtime",
                severity="high",
                status=_status(runtime_paper),
                summary="runtime paper regression guard is not ready",
                evidence={
                    "failed_guard_count": _safe_int(runtime_paper.get("failed_guard_count"), 0),
                    "hard_failed_guard_count": _safe_int(runtime_paper.get("hard_failed_guard_count"), 0),
                },
                recommended_commands=[["./scripts/ops/opsctl.sh", "runtime-paper-regression-guard", "--json"]],
            )
        )

    return sorted(weak, key=lambda row: (_rank_severity(str(row.get("severity") or "")), str(row.get("family") or ""), str(row.get("weak_point_id") or "")))


def _run_probes(project_root: Path, runner: Runner, timeout_sec: int) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for spec in _probe_specs():
        cmd = [str(part) for part in _as_list(spec.get("cmd"))]
        if not _safe_command(cmd):
            results.append({"name": str(spec.get("name") or ""), "cmd": cmd, "skipped": True, "skip_reason": "unsafe_probe"})
            continue
        result = runner(cmd, project_root, min(_safe_int(spec.get("timeout_sec"), timeout_sec), max(int(timeout_sec), 1)))
        payload = _as_dict(result.get("payload"))
        results.append(
            {
                "name": str(spec.get("name") or ""),
                "cmd": list(result.get("cmd") or []),
                "rc": _safe_int(result.get("rc"), 1),
                "overall_status": _status(payload, default=""),
                "ok": bool(payload.get("ok", False)),
                "stdout_tail": str(result.get("stdout_tail") or ""),
                "stderr_tail": str(result.get("stderr_tail") or ""),
            }
        )
    return results


def _degradation_scenario_packets(project_root: Path) -> list[dict[str, Any]]:
    health = project_root / "governance" / "health"
    source_verification = load_json(health / "source_verification_latest.json")
    broker_truth = load_json(health / "schwab_account_snapshot_refresh_latest.json")
    labeling = load_json(health / "training_labeling_intelligence_latest.json")
    paper_truth = load_json(health / "paper_execution_truth_layer_latest.json")
    source_ids = {
        str(row.get("source_id") or "")
        for row in _as_list(source_verification.get("sources"))
        if isinstance(row, dict)
    }
    broker_v2 = _as_dict(broker_truth.get("broker_truth_reconcile_v2"))
    labeling_enrichment = _as_dict(labeling.get("free_label_source_enrichment"))
    scenarios = [
        {
            "scenario_id": "empty_schwab_snapshot",
            "family": "broker_truth",
            "simulated_failure": "Schwab returns ok with an empty/unrecognized account snapshot.",
            "expected_detector": "broker_truth_reconcile_v2.account_identity plus account_snapshot_proof",
            "covered": bool(broker_v2 or broker_truth.get("account_snapshot_proof")),
            "repair_packet": {
                "action": "reject_empty_snapshot_and_refresh_connected_account_aggregate",
                "command": ["./scripts/ops/opsctl.sh", "schwab-account-snapshot-refresh", "--skip-derived", "--json"],
            },
        },
        {
            "scenario_id": "stale_options_chain",
            "family": "source_verification",
            "simulated_failure": "Options chain source goes stale or falls back to a single delayed chain.",
            "expected_detector": "source_verification options_context_mesh freshness/confidence",
            "covered": "options_context_mesh" in source_ids,
            "repair_packet": {"action": "refresh_options_context_mesh", "command": ["./scripts/ops/opsctl.sh", "options-flow-sync", "--json"]},
        },
        {
            "scenario_id": "broken_fx_context",
            "family": "source_verification",
            "simulated_failure": "FX official/proxy context stops updating.",
            "expected_detector": "source_verification fx_market_context freshness and confidence",
            "covered": "fx_market_context" in source_ids,
            "repair_packet": {"action": "refresh_fx_market_context", "command": ["./scripts/ops/opsctl.sh", "fx-market-sync", "--json"]},
        },
        {
            "scenario_id": "missing_crypto_context",
            "family": "source_verification",
            "simulated_failure": "Crypto market context loses exchange/provider coverage.",
            "expected_detector": "source_verification crypto_market_context provider score",
            "covered": "crypto_market_context" in source_ids,
            "repair_packet": {"action": "refresh_crypto_market_context", "command": ["./scripts/ops/opsctl.sh", "crypto-market-sync", "--json"]},
        },
        {
            "scenario_id": "corrupted_label_enrichment",
            "family": "training_labels",
            "simulated_failure": "Label enrichment loses source mapping, confidence, or materialization contracts.",
            "expected_detector": "training_labeling_intelligence free_label_source_enrichment counts",
            "covered": bool(labeling_enrichment.get("classification_counts") and labeling_enrichment.get("materialization_ready_context_count") is not None),
            "repair_packet": {
                "action": "regenerate_labeling_intelligence_and_audit",
                "command": ["./scripts/ops/opsctl.sh", "training-labeling-intelligence", "--apply", "--json"],
            },
        },
        {
            "scenario_id": "paper_activity_without_truth",
            "family": "paper_truth",
            "simulated_failure": "Paper decisions/executions continue while broker truth or source context is degraded.",
            "expected_detector": "paper_execution_truth_layer paper_broker_truth_reconciliation gate",
            "covered": "paper_broker_truth_reconciliation" in _as_dict(paper_truth.get("gates")),
            "repair_packet": {"action": "refresh_paper_execution_truth_layer", "command": ["./scripts/ops/opsctl.sh", "paper-execution-truth", "--json"]},
        },
    ]
    for row in scenarios:
        row["status"] = "covered" if bool(row.get("covered", False)) else "needs_guard"
        row["severity_if_uncovered"] = "high" if row["family"] in {"broker_truth", "paper_truth"} else "medium"
    return scenarios


def build_payload(
    project_root: Path = PROJECT_ROOT,
    *,
    apply: bool = False,
    run_probes: bool = False,
    probe_timeout_sec: int = 120,
    runner: Runner | None = None,
) -> dict[str, Any]:
    project_root = project_root.resolve()
    run_step = runner or _run
    probe_results = _run_probes(project_root, run_step, probe_timeout_sec) if run_probes else []
    weak_points = _evaluate(project_root)
    degradation_scenarios = _degradation_scenario_packets(project_root)
    critical_count = sum(1 for row in weak_points if str(row.get("severity") or "") == "critical")
    high_count = sum(1 for row in weak_points if str(row.get("severity") or "") == "high")
    medium_count = sum(1 for row in weak_points if str(row.get("severity") or "") == "medium")
    family_counts: dict[str, int] = {}
    for row in weak_points:
        family = str(row.get("family") or "unknown")
        family_counts[family] = family_counts.get(family, 0) + 1
    overall_status = "ready"
    if critical_count or high_count:
        overall_status = "blocked"
    elif medium_count:
        overall_status = "degraded"
    payload = {
        "timestamp_utc": iso_now(),
        "schema_version": 1,
        "ok": overall_status == "ready",
        "overall_status": overall_status,
        "apply": bool(apply),
        "run_probes": bool(run_probes),
        "probe_count": len(probe_results),
        "weak_point_count": len(weak_points),
        "critical_weak_point_count": critical_count,
        "high_weak_point_count": high_count,
        "medium_weak_point_count": medium_count,
        "family_counts": family_counts,
        "top_weak_points": weak_points[:6],
        "weak_points": weak_points,
        "probe_results": probe_results,
        "degradation_scenario_count": len(degradation_scenarios),
        "degradation_scenarios_covered": sum(1 for row in degradation_scenarios if str(row.get("status") or "") == "covered"),
        "degradation_repair_packets": degradation_scenarios,
        "adversarial_drill_contract": {
            "generation": "system_adversarial_drill_autopilot_v1",
            "different_from_intense_drill": True,
            "artifact_first": True,
            "safe_probes_only": True,
            "does_not_enable_live_execution": True,
            "does_not_launch_duplicate_storage_drains": True,
            "ranks_cross_layer_weak_points": True,
        },
        "recommended_commands": [
            ["./scripts/ops/opsctl.sh", "system-adversarial-drills", "--apply", "--run-probes", "--json"],
            ["./scripts/ops/opsctl.sh", "system-intense-drills", "--apply", "--json"],
        ],
        "recommended_actions": ordered_unique(
            [
                "clear critical paper/runtime coupling before widening any collection or architecture lanes" if critical_count else "",
                "repair high-severity architecture, storage, and infrastructure weak points before rerunning the intense drill" if high_count else "",
                "treat medium governance items as bounded follow-through rather than live execution permission" if medium_count else "",
            ]
            + [f"{row['weak_point_id']}: {row['summary']}" for row in weak_points[:8]]
        ),
    }
    if apply:
        results_path = DEFAULT_RESULTS_PATH if project_root == PROJECT_ROOT else project_root / "governance" / "drills" / "system_adversarial_drill_results_latest.json"
        write_payload(
            results_path,
            {
                "timestamp_utc": payload["timestamp_utc"],
                "weak_points": weak_points,
                "probe_results": probe_results,
                "degradation_repair_packets": degradation_scenarios,
            },
        )
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Run an artifact-first adversarial system drill that exposes cross-layer weak points.")
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--out-file", default=str(DEFAULT_OUT_PATH))
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--run-probes", action="store_true")
    parser.add_argument("--probe-timeout-seconds", type=int, default=120)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    project_root = Path(args.project_root).expanduser().resolve()
    payload = build_payload(
        project_root,
        apply=bool(args.apply),
        run_probes=bool(args.run_probes),
        probe_timeout_sec=max(int(args.probe_timeout_seconds), 1),
    )
    write_payload(Path(args.out_file).expanduser(), payload)
    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        print(
            "system_adversarial_drill_autopilot "
            f"overall_status={payload.get('overall_status', '')} "
            f"weak_points={payload.get('weak_point_count', 0)} "
            f"critical={payload.get('critical_weak_point_count', 0)}"
        )
    return 0 if payload.get("overall_status") in {"ready", "degraded"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
