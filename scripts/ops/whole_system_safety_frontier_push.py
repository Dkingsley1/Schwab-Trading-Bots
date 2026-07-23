#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT_PATH = PROJECT_ROOT / "governance" / "health" / "whole_system_safety_frontier_latest.json"
REPORT_PATH = PROJECT_ROOT / "governance" / "whole_system_safety_frontier" / "whole_system_safety_frontier_latest.md"
OVERRIDE_PATH = PROJECT_ROOT / "config" / ".env.whole_system_safety_frontier_override"


FRONTIER_DOMAINS: list[dict[str, Any]] = [
    {
        "domain": 1,
        "slug": "promotion_evidence_factory",
        "display_name": "Promotion Evidence Factory",
        "objective": "Turn promotion blockers into explicit evidence packets, coverage asks, and recheck commands.",
        "source_artifacts": [
            "governance/health/evidence_packet_latest.json",
            "governance/health/promotion_quality_gate_latest.json",
            "governance/champion_challenger/promotion_packet_latest.json",
            "governance/walk_forward/promotion_readiness_latest.json",
        ],
        "safe_commands": [
            "./scripts/ops/opsctl.sh evidence-packet --json",
            "./scripts/ops/opsctl.sh quant-strategy-lane-upgrades --json",
        ],
        "outputs": ["promotion_gap_packet", "evidence_recheck_plan", "walk_forward_coverage_ask"],
    },
    {
        "domain": 2,
        "slug": "paper_live_fill_truth_layer",
        "display_name": "Paper/Live Fill Truth Layer",
        "objective": "Compare paper fills against spread, slippage, latency, and counterfactual replay before trusting profit numbers.",
        "source_artifacts": [
            "governance/health/paper_execution_truth_layer_latest.json",
            "governance/health/paper_live_data_standard_latest.json",
            "governance/health/runtime_paper_regression_guard_latest.json",
        ],
        "safe_commands": [
            "./scripts/ops/opsctl.sh paper-live-data-standard --json",
            "./scripts/ops/opsctl.sh runtime-paper-regression-guard --json",
        ],
        "outputs": ["paper_live_fill_gap_packet", "slippage_realism_score", "fill_truth_stop_reason"],
    },
    {
        "domain": 3,
        "slug": "feature_cache_incremental_dataset_layer",
        "display_name": "Feature Cache And Incremental Dataset Layer",
        "objective": "Prefer delta refresh, artifact reuse, and cache invalidation over full rebuilds.",
        "source_artifacts": [
            "governance/health/runtime_snapshot_cache_control_latest.json",
            "governance/health/library_efficiency_deepening_latest.json",
            "governance/health/replay_hash_registry_guard_latest.json",
        ],
        "safe_commands": [
            "./scripts/ops/opsctl.sh library-efficiency-deepening --json",
            "./scripts/ops/opsctl.sh replay-hash-registry --json",
        ],
        "outputs": ["incremental_dataset_contract", "cache_reuse_vote", "full_rebuild_avoidance_plan"],
    },
    {
        "domain": 4,
        "slug": "storage_backpressure_autopilot_vnext",
        "display_name": "Storage/Backpressure Autopilot vNext",
        "objective": "Use bounded drain, compaction, and route planning without increasing write pressure.",
        "source_artifacts": [
            "governance/health/storage_backpressure_autopilot_latest.json",
            "governance/health/writer_cycle_coordinator_latest.json",
            "governance/health/ingestion_storage_control_latest.json",
            "governance/health/system_plumbing_control_latest.json",
        ],
        "safe_commands": [
            "./scripts/ops/opsctl.sh writer-cycle-coordinator --json",
            "./scripts/ops/opsctl.sh system-plumbing-control --json",
        ],
        "outputs": ["bounded_drain_plan", "write_pressure_stop_reason", "storage_route_contract"],
    },
    {
        "domain": 5,
        "slug": "livefeed_reliability_layer",
        "display_name": "Livefeed Reliability Layer",
        "objective": "Track feed freshness, skipped files, unreadable logs, mirror health, and refresh guard state.",
        "source_artifacts": [
            "governance/health/livefeed_local_latest.json",
            "governance/health/livefeed_refresh_guard_latest.json",
            "governance/health/process_watchdog_latest.json",
        ],
        "safe_commands": [
            "./scripts/ops/opsctl.sh livefeed-refresh-guard --json",
            "./scripts/ops/opsctl.sh process-watchdog --json",
        ],
        "outputs": ["feed_freshness_packet", "tail_continuity_score", "safe_refresh_vote"],
    },
    {
        "domain": 6,
        "slug": "account_position_intelligence",
        "display_name": "Account/Position Intelligence",
        "objective": "Keep account rules, account holdings, covered calls, and position context available to advisory logic.",
        "source_artifacts": [
            "governance/health/account_position_study_latest.json",
            "governance/health/account_policy_context_latest.json",
            "governance/health/covered_call_roll_watch_latest.json",
        ],
        "safe_commands": [
            "./scripts/ops/opsctl.sh account-position-study --json",
            "./scripts/ops/opsctl.sh covered-call-roll-watch --json",
        ],
        "outputs": ["position_context_packet", "covered_call_advisory_packet", "account_rule_guard"],
    },
    {
        "domain": 7,
        "slug": "risk_exposure_graph",
        "display_name": "Risk And Exposure Graph",
        "objective": "Map sleeve, symbol, account, factor, option, and liquidity overlap without allocation authority.",
        "source_artifacts": [
            "governance/health/library_efficiency_deepening_latest.json",
            "governance/health/deep_quant_layer_upgrade_latest.json",
            "governance/health/account_position_study_latest.json",
        ],
        "safe_commands": [
            "./scripts/ops/opsctl.sh library-efficiency-deepening --json",
            "./scripts/ops/opsctl.sh deep-quant-layer-upgrade --json",
        ],
        "outputs": ["exposure_graph_packet", "crowding_alert_context", "duplicate_exposure_hint"],
    },
    {
        "domain": 8,
        "slug": "benchmark_cost_governor",
        "display_name": "Benchmark/Cost Governor",
        "objective": "Make library routes prove speed, memory, disk, cache, and quality lift before wider use.",
        "source_artifacts": [
            "governance/health/library_efficiency_deepening_latest.json",
            "governance/health/safety_bounded_advancement_frontier_latest.json",
            "governance/health/a_plus_operating_packet_latest.json",
        ],
        "safe_commands": [
            "./scripts/ops/opsctl.sh library-efficiency-deepening --json",
            "./scripts/ops/opsctl.sh a-plus-operating-packet --json",
        ],
        "outputs": ["route_benchmark_contract", "benefit_cost_score", "scale_hold_vote"],
    },
    {
        "domain": 9,
        "slug": "model_retirement_court",
        "display_name": "Model Retirement Court",
        "objective": "Identify stale, redundant, expensive, or low-contribution routes without deleting anything automatically.",
        "source_artifacts": [
            "governance/health/model_lifecycle_latest.json",
            "governance/health/promotion_quality_gate_latest.json",
            "governance/health/sleeve_profitability_dashboard_latest.json",
        ],
        "safe_commands": [
            "./scripts/ops/opsctl.sh model-lifecycle --json",
            "./scripts/ops/opsctl.sh sleeve-profitability-dashboard --json",
        ],
        "outputs": ["retirement_candidate_packet", "keep_or_downrank_vote", "manual_review_required"],
    },
    {
        "domain": 10,
        "slug": "operator_cockpit_a_plus_scoreboard",
        "display_name": "Operator Cockpit / A+ Scoreboard",
        "objective": "Summarize what is green, blocked, stale, soaking, and next to recheck in one packet.",
        "source_artifacts": [
            "governance/health/a_plus_operating_packet_latest.json",
            "governance/health/health_fast_latest.json",
            "governance/health/safety_bounded_advancement_frontier_latest.json",
        ],
        "safe_commands": [
            "./scripts/ops/opsctl.sh a-plus-operating-packet --json",
            "./scripts/ops/opsctl.sh health-fast --json",
        ],
        "outputs": ["a_plus_frontier_scoreboard", "blocked_surface_table", "next_recheck_packet"],
    },
    {
        "domain": 11,
        "slug": "notification_reliability",
        "display_name": "Notification Reliability",
        "objective": "Track iMessage/critical alert readiness, duplicate suppression, ack state, and escalation posture.",
        "source_artifacts": [
            "governance/health/remote_alert_control_latest.json",
            "governance/health/notification_escalation_ladder_latest.json",
            "governance/health/mac_notification_watch_state.json",
        ],
        "safe_commands": [
            "./scripts/ops/opsctl.sh remote-alert-control --json",
            "./scripts/ops/opsctl.sh notification-escalation-ladder --json",
        ],
        "outputs": ["alert_ladder_packet", "duplicate_alert_suppression_state", "imessage_readiness_hint"],
    },
    {
        "domain": 12,
        "slug": "disaster_recovery_replay_drills",
        "display_name": "Disaster Recovery / Replay Drills",
        "objective": "Keep restore, replay hash, deterministic regression, and storage recovery proof visible without heavy replay.",
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
        "outputs": ["dr_replay_proof_packet", "restore_readiness_hint", "heavy_replay_stop_reason"],
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
    safety = _load_json(health_root / "safety_bounded_advancement_frontier_latest.json")
    quant_lanes = _load_json(health_root / "quant_strategy_lane_upgrades_latest.json")
    health_fast = _load_json(health_root / "health_fast_latest.json")
    retrain = _load_json(health_root / "retrain_launch_latest.json")

    safety_reasons = safety.get("safety_stop_reason") if isinstance(safety.get("safety_stop_reason"), list) else []
    lane_gates = quant_lanes.get("gate_state") if isinstance(quant_lanes.get("gate_state"), dict) else {}
    storage = health_fast.get("storage") if isinstance(health_fast.get("storage"), dict) else {}
    runtime = health_fast.get("runtime_pressure") if isinstance(health_fast.get("runtime_pressure"), dict) else {}

    runtime_green = bool(lane_gates.get("runtime_green", True)) and _status(runtime, "ready") in {"ready", "ok", "green", ""}
    storage_green = bool(lane_gates.get("storage_green", True)) and str(storage.get("severity") or "").strip().lower() in {"stable", "ready", "green", ""}
    paper_400_ready = bool(lane_gates.get("paper_400_ready", True))
    promotion_ready = bool(lane_gates.get("promotion_quality_ready", False))
    training_active = str(retrain.get("state") or "").strip().lower() == "running"

    blockers = list(safety_reasons)
    if training_active:
        blockers.append("large_training_batch_running_control_plane_only")
    blockers.extend(lane_gates.get("promotion_quality_failed_checks") or [])
    blockers.extend(f"promotion_readiness:{item}" for item in (lane_gates.get("promotion_readiness_blockers") or []))
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
        "safety_guard_stop_active": bool(safety.get("safety_stop_active", bool(blockers))),
        "control_plane_allowed": bool(runtime_green and storage_green),
        "activation_allowed": bool(runtime_green and storage_green and paper_400_ready and promotion_ready and not blockers),
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
        "paper_execution_authority_enabled": False,
        "live_execution_authority_enabled": False,
        "allocation_authority_enabled": False,
        "training_intake_authority_enabled": False,
        "new_collector_authority_enabled": False,
        "heavy_replay_authority_enabled": False,
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
            "automatic_model_retirement_or_deletion",
        ],
    }


def _recommended_env(payload: dict[str, Any]) -> dict[str, str]:
    return {
        "WHOLE_SYSTEM_SAFETY_FRONTIER_ENABLED": "1",
        "WHOLE_SYSTEM_SAFETY_FRONTIER_DOMAIN_COUNT": str(payload.get("domain_count") or 0),
        "WHOLE_SYSTEM_SAFETY_FRONTIER_MODE": str(payload.get("frontier_mode") or "control_plane_advisory"),
        "WHOLE_SYSTEM_SAFETY_FRONTIER_STOP_ACTIVE": "1" if payload.get("safety_stop_active") else "0",
        "WHOLE_SYSTEM_SAFETY_FRONTIER_PAPER_EXECUTION_AUTHORITY": "0",
        "WHOLE_SYSTEM_SAFETY_FRONTIER_LIVE_EXECUTION_AUTHORITY": "0",
        "WHOLE_SYSTEM_SAFETY_FRONTIER_ALLOCATION_AUTHORITY": "0",
        "WHOLE_SYSTEM_SAFETY_FRONTIER_TRAINING_INTAKE_AUTHORITY": "0",
        "WHOLE_SYSTEM_SAFETY_FRONTIER_HEAVY_REPLAY_AUTHORITY": "0",
        "WHOLE_SYSTEM_SAFETY_FRONTIER_NEXT_SCOPE": "soak_recheck_and_evidence_only",
    }


def _write_env_override(path: Path, env: dict[str, str]) -> bool:
    lines = ["# Auto-managed by scripts/ops/whole_system_safety_frontier_push.py"]
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
    domains = [_domain_payload(domain, project_root, gates) for domain in FRONTIER_DOMAINS]
    control_plane_count = sum(1 for domain in domains if bool(domain.get("control_plane_enabled")))
    coverage = round(
        sum(float(domain.get("source_artifact_coverage") or 0.0) for domain in domains) / max(len(domains), 1),
        4,
    )
    safety_stop = bool(gates.get("blockers") or not gates.get("activation_allowed"))
    status = (
        "whole_system_frontier_control_plane_applied_pause_for_soak"
        if safety_stop and control_plane_count == len(domains)
        else "whole_system_frontier_activation_ready"
        if not safety_stop
        else "whole_system_frontier_blocked_before_control_plane"
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
        "source_artifact_coverage": coverage,
        "paper_execution_authority_enabled": False,
        "live_execution_authority_enabled": False,
        "allocation_authority_enabled": False,
        "training_intake_authority_enabled": False,
        "new_collector_authority_enabled": False,
        "heavy_replay_authority_enabled": False,
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
            "automatic_model_retirement_or_deletion",
        ]
        if safety_stop
        else [],
        "next_recheck_commands": [
            "./scripts/ops/opsctl.sh health-fast --json",
            "./scripts/ops/opsctl.sh whole-system-safety-frontier --json",
            "./scripts/ops/opsctl.sh safety-bounded-advancement-frontier --json",
            "./scripts/ops/opsctl.sh evidence-packet --json",
            "./scripts/ops/opsctl.sh quant-strategy-lane-upgrades --json",
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
        "# Whole-System Safety Frontier",
        "",
        f"- timestamp_utc: `{payload.get('timestamp_utc')}`",
        f"- status: `{payload.get('overall_status')}`",
        f"- domain_count: `{payload.get('domain_count')}`",
        f"- control_plane_domain_count: `{payload.get('control_plane_domain_count')}`",
        f"- safety_stop_active: `{payload.get('safety_stop_active')}`",
        f"- pause_kind: `{payload.get('pause_kind')}`",
        f"- source_artifact_coverage: `{payload.get('source_artifact_coverage')}`",
        "",
        "## Stop Reason",
        "",
    ]
    reasons = payload.get("safety_stop_reason") if isinstance(payload.get("safety_stop_reason"), list) else []
    lines.extend(f"- `{reason}`" for reason in reasons) if reasons else lines.append("- none")
    lines.extend(["", "## Domains", ""])
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
                f"- outputs: {', '.join(str(item) for item in domain.get('outputs') or [])}",
                "",
            ]
        )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Push the 12-domain whole-system frontier until safety guard says wait.")
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
            "whole_system_safety_frontier "
            f"status={payload.get('overall_status')} "
            f"domains={payload.get('domain_count')} "
            f"control_plane={payload.get('control_plane_domain_count')} "
            f"safety_stop={int(bool(payload.get('safety_stop_active')))}"
        )
    return 0 if payload.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
