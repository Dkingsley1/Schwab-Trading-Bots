#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT_PATH = PROJECT_ROOT / "governance" / "health" / "system_efficiency_frontier_latest.json"
REPORT_PATH = PROJECT_ROOT / "governance" / "system_efficiency_frontier" / "system_efficiency_frontier_latest.md"
OVERRIDE_PATH = PROJECT_ROOT / "config" / ".env.system_efficiency_frontier_override"


EFFICIENCY_DOMAINS: list[dict[str, Any]] = [
    {
        "domain": 1,
        "slug": "library_backend_routing_efficiency",
        "display_name": "Library Backend Routing Efficiency",
        "objective": "Route every expensive task to MLX, Polars, DuckDB, Arrow, QuantLib, Numba, or cached lookup before generic Python.",
        "source_artifacts": [
            "governance/health/library_efficiency_deepening_latest.json",
            "governance/health/library_utilization_router_latest.json",
            "governance/health/mlx_intelligence_router_latest.json",
        ],
        "safe_commands": [
            "./scripts/ops/opsctl.sh library-efficiency-deepening --json",
            "./scripts/ops/opsctl.sh library-utilization-router --json",
            "./scripts/ops/opsctl.sh mlx-intelligence-router --json",
        ],
        "efficiency_outputs": ["backend_route_vote", "route_cost_hint", "library_cap_contract"],
    },
    {
        "domain": 2,
        "slug": "runtime_memory_pcore_efficiency",
        "display_name": "Runtime, Memory, And P-Core Efficiency",
        "objective": "Keep runtime, memory, P-core, foreground-app, and MLX caps aligned before widening work.",
        "source_artifacts": [
            "governance/health/runtime_throttle_control_latest.json",
            "governance/health/memory_pressure_intelligence_latest.json",
            "governance/health/memory_efficiency_control_latest.json",
            "governance/health/pressure_relief_control_latest.json",
        ],
        "safe_commands": [
            "./scripts/ops/opsctl.sh runtime-throttle --json",
            "./scripts/ops/opsctl.sh memory-pressure-intelligence --json",
            "./scripts/ops/opsctl.sh pressure-relief --json",
        ],
        "efficiency_outputs": ["runtime_cap_packet", "memory_pressure_budget", "pcore_yield_policy"],
    },
    {
        "domain": 3,
        "slug": "storage_writer_backpressure_efficiency",
        "display_name": "Storage, Writer, And Backpressure Efficiency",
        "objective": "Prefer bounded writer handoff, low-churn drains, and backpressure-aware routing over new write pressure.",
        "source_artifacts": [
            "governance/health/storage_backpressure_autopilot_latest.json",
            "governance/health/writer_cycle_coordinator_latest.json",
            "governance/health/ingestion_storage_control_latest.json",
            "governance/health/system_plumbing_control_latest.json",
        ],
        "safe_commands": [
            "./scripts/ops/opsctl.sh writer-cycle-coordinator --json",
            "./scripts/ops/opsctl.sh storage-backpressure-autopilot --json",
            "./scripts/ops/opsctl.sh system-plumbing-control --json",
        ],
        "efficiency_outputs": ["bounded_writer_plan", "backpressure_stop_reason", "low_churn_route_hint"],
    },
    {
        "domain": 4,
        "slug": "feature_cache_incremental_efficiency",
        "display_name": "Feature Cache And Incremental Dataset Efficiency",
        "objective": "Reuse fresh snapshots, feature deltas, and content hashes before triggering rebuilds.",
        "source_artifacts": [
            "governance/health/runtime_snapshot_cache_control_latest.json",
            "governance/health/library_efficiency_deepening_latest.json",
            "governance/health/replay_hash_registry_guard_latest.json",
        ],
        "safe_commands": [
            "./scripts/ops/opsctl.sh runtime-snapshot-cache --json",
            "./scripts/ops/opsctl.sh library-efficiency-deepening --json",
            "./scripts/ops/opsctl.sh replay-hash-registry --json",
        ],
        "efficiency_outputs": ["snapshot_reuse_vote", "feature_delta_plan", "cache_invalidation_guard"],
    },
    {
        "domain": 5,
        "slug": "livefeed_operator_view_efficiency",
        "display_name": "Livefeed And Operator View Efficiency",
        "objective": "Keep livefeed tails useful without high fanout, heavy TTL abuse, or repeated restarts of healthy loops.",
        "source_artifacts": [
            "governance/health/livefeed_refresh_guard_latest.json",
            "governance/health/livefeed_local_latest.json",
            "governance/health/memory_efficiency_control_latest.json",
        ],
        "safe_commands": [
            "./scripts/ops/opsctl.sh livefeed-refresh-guard --json",
            "./scripts/ops/opsctl.sh health-fast --json",
        ],
        "efficiency_outputs": ["feed_tail_budget", "safe_refresh_vote", "operator_view_trim_policy"],
    },
    {
        "domain": 6,
        "slug": "training_scheduler_efficiency",
        "display_name": "Training Scheduler Efficiency",
        "objective": "Keep retrain, coverage, and drain work evidence-aware without starting another heavy batch.",
        "source_artifacts": [
            "governance/health/training_runtime_control_latest.json",
            "governance/health/training_drain_autopilot_latest.json",
            "governance/health/retrain_launch_latest.json",
            "governance/health/coverage_gap_closer_latest.json",
        ],
        "safe_commands": [
            "./scripts/ops/opsctl.sh training-runtime-control --json",
            "./scripts/ops/opsctl.sh coverage-gap-closer --json",
        ],
        "efficiency_outputs": ["training_queue_budget", "coverage_recheck_plan", "heavy_training_stop_reason"],
    },
    {
        "domain": 7,
        "slug": "report_render_efficiency",
        "display_name": "Report And Render Efficiency",
        "objective": "Use cached artifacts and bounded render jobs before rebuilding large reports or PDFs.",
        "source_artifacts": [
            "governance/health/report_quality_guard_latest.json",
            "governance/health/a_plus_operating_packet_latest.json",
            "governance/health/commands_hygiene_latest.json",
        ],
        "safe_commands": [
            "./scripts/ops/opsctl.sh report-quality-guard --json",
            "./scripts/ops/opsctl.sh a-plus-operating-packet --json",
            "./scripts/ops/opsctl.sh commands-hygiene --json",
        ],
        "efficiency_outputs": ["report_cache_plan", "render_job_cap", "stale_report_reuse_vote"],
    },
    {
        "domain": 8,
        "slug": "notification_noise_efficiency",
        "display_name": "Notification Noise Efficiency",
        "objective": "Keep iMessage and critical alerts reliable while suppressing duplicates and low-value noise.",
        "source_artifacts": [
            "governance/health/remote_alert_control_latest.json",
            "governance/health/notification_escalation_ladder_latest.json",
            "governance/health/mac_notification_watch_state.json",
        ],
        "safe_commands": [
            "./scripts/ops/opsctl.sh remote-alert-control --json",
            "./scripts/ops/opsctl.sh notification-escalation-ladder --json",
        ],
        "efficiency_outputs": ["alert_noise_budget", "duplicate_suppression_state", "imessage_reliability_hint"],
    },
    {
        "domain": 9,
        "slug": "paper_execution_truth_efficiency",
        "display_name": "Paper Execution Truth Efficiency",
        "objective": "Preserve paper/live fill realism and regression checks without granting execution authority.",
        "source_artifacts": [
            "governance/health/paper_execution_truth_layer_latest.json",
            "governance/health/runtime_paper_regression_guard_latest.json",
            "governance/health/paper_live_data_standard_latest.json",
        ],
        "safe_commands": [
            "./scripts/ops/opsctl.sh runtime-paper-regression-guard --json",
            "./scripts/ops/opsctl.sh paper-live-data-standard --json",
        ],
        "efficiency_outputs": ["fill_truth_reuse_packet", "paper_live_gap_budget", "execution_authority_stop_reason"],
    },
    {
        "domain": 10,
        "slug": "model_route_lifecycle_efficiency",
        "display_name": "Model Route Lifecycle Efficiency",
        "objective": "Mark stale, duplicate, or expensive model routes for review without automatic deletion or retirement.",
        "source_artifacts": [
            "governance/health/model_lifecycle_latest.json",
            "governance/health/sleeve_profitability_dashboard_latest.json",
            "governance/health/promotion_quality_gate_latest.json",
        ],
        "safe_commands": [
            "./scripts/ops/opsctl.sh model-lifecycle --json",
            "./scripts/ops/opsctl.sh sleeve-profitability-dashboard --json",
        ],
        "efficiency_outputs": ["route_lifecycle_review", "downrank_candidate", "manual_retirement_required"],
    },
    {
        "domain": 11,
        "slug": "disaster_replay_efficiency",
        "display_name": "Disaster Recovery And Replay Efficiency",
        "objective": "Keep replay/restore proof visible without running heavy replay during pressure or training.",
        "source_artifacts": [
            "governance/health/storage_disaster_recovery_latest.json",
            "governance/health/replay_hash_registry_guard_latest.json",
            "governance/health/golden_replay_regression_latest.json",
            "governance/health/paper_replay_drill_latest.json",
        ],
        "safe_commands": [
            "./scripts/ops/opsctl.sh storage-disaster-recovery --json",
            "./scripts/ops/opsctl.sh replay-hash-registry --json",
            "./scripts/ops/opsctl.sh golden-replay-regression --json",
        ],
        "efficiency_outputs": ["replay_proof_reuse", "restore_check_budget", "heavy_replay_stop_reason"],
    },
    {
        "domain": 12,
        "slug": "command_operator_efficiency",
        "display_name": "Command And Operator Efficiency",
        "objective": "Keep command docs, validity, next-step packets, and A+ status coherent so operator work stays short.",
        "source_artifacts": [
            "governance/health/commands_hygiene_latest.json",
            "governance/health/command_validity_latest.json",
            "governance/health/system_needs_latest.json",
            "governance/health/a_plus_operating_packet_latest.json",
        ],
        "safe_commands": [
            "./scripts/ops/opsctl.sh commands-hygiene --json",
            "./scripts/ops/opsctl.sh command-validity --json",
            "./scripts/ops/opsctl.sh system-needs --json",
        ],
        "efficiency_outputs": ["operator_next_step_packet", "command_drift_stop_reason", "a_plus_efficiency_gap"],
    },
]


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def _status(payload: dict[str, Any], default: str = "missing") -> str:
    if not payload:
        return default
    return str(payload.get("overall_status") or payload.get("status") or payload.get("state") or default).strip().lower()


def _ordered_unique(items: list[Any]) -> list[str]:
    return list(dict.fromkeys(str(item) for item in items if str(item).strip()))


def _source_status(project_root: Path, rel_path: str) -> dict[str, Any]:
    path = project_root / rel_path
    payload = _load_json(path)
    return {
        "path": rel_path,
        "exists": path.exists(),
        "status": _status(payload),
        "ok": bool(payload.get("ok", False)) if payload else False,
    }


def _gate_state(project_root: Path) -> dict[str, Any]:
    health_root = project_root / "governance" / "health"
    whole = _load_json(health_root / "whole_system_safety_frontier_latest.json")
    safety = _load_json(health_root / "safety_bounded_advancement_frontier_latest.json")
    quant_lanes = _load_json(health_root / "quant_strategy_lane_upgrades_latest.json")
    health_fast = _load_json(health_root / "health_fast_latest.json")
    retrain = _load_json(health_root / "retrain_launch_latest.json")

    lane_gates = quant_lanes.get("gate_state") if isinstance(quant_lanes.get("gate_state"), dict) else {}
    storage = health_fast.get("storage") if isinstance(health_fast.get("storage"), dict) else {}
    runtime = health_fast.get("runtime_pressure") if isinstance(health_fast.get("runtime_pressure"), dict) else {}

    runtime_green = bool(lane_gates.get("runtime_green", True)) and _status(runtime, "ready") in {"ready", "ok", "green", ""}
    storage_green = bool(lane_gates.get("storage_green", True)) and str(storage.get("severity") or "").strip().lower() in {"stable", "ready", "green", ""}
    paper_400_ready = bool(lane_gates.get("paper_400_ready", True))
    promotion_ready = bool(lane_gates.get("promotion_quality_ready", False))
    training_active = str(retrain.get("state") or "").strip().lower() == "running"

    blockers: list[Any] = []
    blockers.extend(whole.get("safety_stop_reason") if isinstance(whole.get("safety_stop_reason"), list) else [])
    blockers.extend(safety.get("safety_stop_reason") if isinstance(safety.get("safety_stop_reason"), list) else [])
    blockers.extend(lane_gates.get("promotion_quality_failed_checks") or [])
    blockers.extend(f"promotion_readiness:{item}" for item in (lane_gates.get("promotion_readiness_blockers") or []))
    if training_active:
        blockers.append("large_training_batch_running_control_plane_only")
    if not runtime_green:
        blockers.append("runtime_not_green")
    if not storage_green:
        blockers.append("storage_not_green")
    if not paper_400_ready:
        blockers.append("paper_400_not_ready")
    if not promotion_ready:
        blockers.append("promotion_quality_not_ready")
    blockers = _ordered_unique(blockers)
    return {
        "runtime_green": runtime_green,
        "storage_green": storage_green,
        "paper_400_ready": paper_400_ready,
        "promotion_quality_ready": promotion_ready,
        "training_batch_active": training_active,
        "training_batch_pid": retrain.get("pid"),
        "control_plane_allowed": bool(runtime_green and storage_green),
        "activation_allowed": bool(runtime_green and storage_green and paper_400_ready and promotion_ready and not blockers),
        "safety_guard_stop_active": bool(blockers),
        "blockers": blockers,
    }


def _domain_payload(domain: dict[str, Any], project_root: Path, gates: dict[str, Any]) -> dict[str, Any]:
    sources = [_source_status(project_root, rel_path) for rel_path in (domain.get("source_artifacts") or [])]
    present = sum(1 for source in sources if bool(source.get("exists")))
    coverage = round(present / max(len(sources), 1), 4)
    control_plane = bool(gates.get("control_plane_allowed"))
    return {
        **domain,
        "state": "applied_control_plane" if control_plane else "blocked_by_runtime_or_storage_guard",
        "control_plane_enabled": control_plane,
        "advisory_enabled": control_plane,
        "paper_rehearsal_enabled": control_plane,
        "live_advisory_enabled": control_plane,
        "efficiency_contract_enabled": control_plane,
        "paper_execution_authority_enabled": False,
        "live_execution_authority_enabled": False,
        "allocation_authority_enabled": False,
        "training_intake_authority_enabled": False,
        "new_collector_authority_enabled": False,
        "heavy_replay_authority_enabled": False,
        "destructive_cleanup_authority_enabled": False,
        "source_artifact_coverage": coverage,
        "source_statuses": sources,
        "activation_blockers": list(gates.get("blockers") or []),
        "stop_before": [
            "paper_execution_authority",
            "live_execution_authority",
            "allocation_authority",
            "training_intake_authority",
            "new_high_volume_collectors",
            "heavy_replay_or_large_training",
            "destructive_cleanup",
            "automatic_model_retirement_or_deletion",
        ],
    }


def _recommended_env(payload: dict[str, Any]) -> dict[str, str]:
    return {
        "SYSTEM_EFFICIENCY_FRONTIER_ENABLED": "1",
        "SYSTEM_EFFICIENCY_FRONTIER_DOMAIN_COUNT": str(payload.get("domain_count") or 0),
        "SYSTEM_EFFICIENCY_FRONTIER_MODE": str(payload.get("frontier_mode") or "control_plane_advisory"),
        "SYSTEM_EFFICIENCY_FRONTIER_STOP_ACTIVE": "1" if payload.get("safety_stop_active") else "0",
        "SYSTEM_EFFICIENCY_PAPER_EXECUTION_AUTHORITY": "0",
        "SYSTEM_EFFICIENCY_LIVE_EXECUTION_AUTHORITY": "0",
        "SYSTEM_EFFICIENCY_ALLOCATION_AUTHORITY": "0",
        "SYSTEM_EFFICIENCY_TRAINING_INTAKE_AUTHORITY": "0",
        "SYSTEM_EFFICIENCY_HEAVY_REPLAY_AUTHORITY": "0",
        "SYSTEM_EFFICIENCY_DESTRUCTIVE_CLEANUP_AUTHORITY": "0",
        "SYSTEM_EFFICIENCY_NEXT_SCOPE": "soak_recheck_and_low_churn_evidence_only",
    }


def _write_env_override(path: Path, env: dict[str, str]) -> bool:
    lines = ["# Auto-managed by scripts/ops/system_efficiency_frontier_push.py"]
    for key, value in sorted(env.items()):
        safe = str(value).replace("'", "'\"'\"'")
        lines.append(f"{key}='{safe}'")
    content = "\n".join(lines) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    current = path.read_text(encoding="utf-8") if path.exists() else ""
    if current == content:
        return False
    path.write_text(content, encoding="utf-8")
    return True


def build_payload(project_root: Path = PROJECT_ROOT) -> dict[str, Any]:
    gates = _gate_state(project_root)
    domains = [_domain_payload(domain, project_root, gates) for domain in EFFICIENCY_DOMAINS]
    control_plane_count = sum(1 for domain in domains if bool(domain.get("control_plane_enabled")))
    artifact_coverage = round(
        sum(float(domain.get("source_artifact_coverage") or 0.0) for domain in domains) / max(len(domains), 1),
        4,
    )
    safety_stop = bool(gates.get("blockers") or not gates.get("activation_allowed"))
    status = (
        "system_efficiency_frontier_control_plane_applied_pause_for_soak"
        if safety_stop and control_plane_count == len(domains)
        else "system_efficiency_frontier_activation_ready"
        if not safety_stop
        else "system_efficiency_frontier_blocked_before_control_plane"
    )
    payload: dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "schema_version": 1,
        "ok": len(domains) == 12,
        "overall_status": status,
        "frontier_mode": "control_plane_advisory",
        "domain_count": len(domains),
        "control_plane_domain_count": control_plane_count,
        "advisory_domain_count": sum(1 for domain in domains if bool(domain.get("advisory_enabled"))),
        "paper_rehearsal_domain_count": sum(1 for domain in domains if bool(domain.get("paper_rehearsal_enabled"))),
        "live_advisory_domain_count": sum(1 for domain in domains if bool(domain.get("live_advisory_enabled"))),
        "efficiency_contract_domain_count": sum(1 for domain in domains if bool(domain.get("efficiency_contract_enabled"))),
        "source_artifact_coverage": artifact_coverage,
        "paper_execution_authority_enabled": False,
        "live_execution_authority_enabled": False,
        "allocation_authority_enabled": False,
        "training_intake_authority_enabled": False,
        "new_collector_authority_enabled": False,
        "heavy_replay_authority_enabled": False,
        "destructive_cleanup_authority_enabled": False,
        "safety_stop_active": safety_stop,
        "pause_kind": "soak_until_training_batch_and_promotion_evidence_clear" if safety_stop else "none",
        "safety_stop_reason": list(gates.get("blockers") or []),
        "gate_state": gates,
        "domains": domains,
        "do_not_push_until_guard_clears": [
            "paper_execution_authority",
            "live_execution_authority",
            "allocation_authority",
            "training_intake_authority",
            "new_high_volume_collectors",
            "heavy_replay_or_large_training",
            "destructive_cleanup",
            "automatic_model_retirement_or_deletion",
        ]
        if safety_stop
        else [],
        "next_recheck_commands": [
            "./scripts/ops/opsctl.sh health-fast --json",
            "./scripts/ops/opsctl.sh system-efficiency-frontier --json",
            "./scripts/ops/opsctl.sh library-efficiency-deepening --json",
            "./scripts/ops/opsctl.sh whole-system-safety-frontier --json",
        ],
        "artifacts": {
            "json": str(OUT_PATH),
            "report": str(REPORT_PATH),
            "env_override": str(OVERRIDE_PATH),
        },
    }
    payload["recommended_runtime_env"] = _recommended_env(payload)
    return payload


def render_report(payload: dict[str, Any]) -> str:
    lines = [
        "# System Efficiency Frontier",
        "",
        f"- timestamp_utc: `{payload.get('timestamp_utc')}`",
        f"- status: `{payload.get('overall_status')}`",
        f"- domain_count: `{payload.get('domain_count')}`",
        f"- control_plane_domain_count: `{payload.get('control_plane_domain_count')}`",
        f"- efficiency_contract_domain_count: `{payload.get('efficiency_contract_domain_count')}`",
        f"- safety_stop_active: `{payload.get('safety_stop_active')}`",
        f"- pause_kind: `{payload.get('pause_kind')}`",
        f"- source_artifact_coverage: `{payload.get('source_artifact_coverage')}`",
        "",
        "## Stop Reason",
        "",
    ]
    reasons = payload.get("safety_stop_reason") if isinstance(payload.get("safety_stop_reason"), list) else []
    lines.extend(f"- `{reason}`" for reason in reasons) if reasons else lines.append("- none")
    lines.extend(["", "## Efficiency Domains", ""])
    for domain in payload.get("domains") or []:
        if not isinstance(domain, dict):
            continue
        lines.extend(
            [
                f"### {domain.get('domain')}. {domain.get('display_name')}",
                "",
                f"- slug: `{domain.get('slug')}`",
                f"- state: `{domain.get('state')}`",
                f"- source_artifact_coverage: `{domain.get('source_artifact_coverage')}`",
                f"- paper_execution_authority_enabled: `{domain.get('paper_execution_authority_enabled')}`",
                f"- live_execution_authority_enabled: `{domain.get('live_execution_authority_enabled')}`",
                f"- outputs: {', '.join(str(item) for item in domain.get('efficiency_outputs') or [])}",
                "",
            ]
        )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Push system efficiency until the safety guard says wait.")
    parser.add_argument("--apply", action="store_true", help="Write the guarded runtime env override.")
    parser.add_argument("--json", action="store_true", help="Print the full JSON payload.")
    parser.add_argument("--no-write", action="store_true", help="Build without writing artifacts.")
    args = parser.parse_args()

    payload = build_payload()
    if args.apply:
        payload["apply_result"] = {
            "applied": True,
            "override_path": str(OVERRIDE_PATH),
            "override_changed": _write_env_override(OVERRIDE_PATH, {str(k): str(v) for k, v in payload["recommended_runtime_env"].items()}),
        }
    else:
        payload["apply_result"] = {"applied": False, "override_path": str(OVERRIDE_PATH), "override_changed": False}
    if not args.no_write:
        _write_json(OUT_PATH, payload)
        _write_text(REPORT_PATH, render_report(payload))
    if args.json:
        print(json.dumps(payload, ensure_ascii=True, indent=2))
    else:
        print(
            "system_efficiency_frontier "
            f"status={payload.get('overall_status')} "
            f"domains={payload.get('domain_count')} "
            f"control_plane={payload.get('control_plane_domain_count')} "
            f"safety_stop={int(bool(payload.get('safety_stop_active')))}"
        )
    return 0 if payload.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
