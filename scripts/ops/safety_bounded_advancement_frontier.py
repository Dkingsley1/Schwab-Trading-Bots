#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT_PATH = PROJECT_ROOT / "governance" / "health" / "safety_bounded_advancement_frontier_latest.json"
REPORT_PATH = PROJECT_ROOT / "governance" / "safety_bounded_advancement_frontier" / "safety_bounded_advancement_frontier_latest.md"
OVERRIDE_PATH = PROJECT_ROOT / "config" / ".env.safety_bounded_advancement_frontier_override"


FRONTIER_STAGES: list[dict[str, Any]] = [
    {
        "stage": 1,
        "slug": "route_assimilation_bridge",
        "display_name": "Route Assimilation Bridge",
        "objective": "Merge deep-quant, MLX, and non-MLX library route decisions into one advisory route packet.",
        "outputs": ["unified_route_packet", "backend_vote_trace", "route_conflict_reason"],
    },
    {
        "stage": 2,
        "slug": "artifact_freshness_dag",
        "display_name": "Artifact Freshness DAG",
        "objective": "Track which health, feature, and route artifacts are fresh enough to reuse before recomputing.",
        "outputs": ["freshness_dependency_graph", "reuse_or_refresh_vote", "stale_artifact_reason"],
    },
    {
        "stage": 3,
        "slug": "cache_ownership_contracts",
        "display_name": "Cache Ownership Contracts",
        "objective": "Assign cache owners for feature deltas, pricing kernels, route decisions, and path signatures.",
        "outputs": ["cache_owner_map", "cache_ttl_policy", "invalidation_scope"],
    },
    {
        "stage": 4,
        "slug": "benchmark_cost_ledger",
        "display_name": "Benchmark Cost Ledger",
        "objective": "Require every route to report latency, memory, disk, cache hit rate, and benefit before scale-up.",
        "outputs": ["route_cost_ledger", "benefit_cost_score", "scale_or_hold_vote"],
    },
    {
        "stage": 5,
        "slug": "paper_live_parity_witness",
        "display_name": "Paper/Live Parity Witness",
        "objective": "Prove paper rehearsal and live advisory use the same feature, pricing, and route contracts.",
        "outputs": ["paper_live_contract_diff", "parity_witness_packet", "drift_alert"],
    },
    {
        "stage": 6,
        "slug": "incremental_feature_reuse_frontier",
        "display_name": "Incremental Feature Reuse Frontier",
        "objective": "Promote incremental feature reuse wherever artifacts are fresh and point-in-time safe.",
        "outputs": ["feature_reuse_plan", "full_rebuild_avoidance", "point_in_time_guard"],
    },
    {
        "stage": 7,
        "slug": "pricing_kernel_reuse_frontier",
        "display_name": "Pricing Kernel Reuse Frontier",
        "objective": "Reuse Greeks, vol, covered-call roll, and pricing packets when inputs have not materially changed.",
        "outputs": ["pricing_cache_key", "greek_reuse_vote", "covered_call_math_reuse"],
    },
    {
        "stage": 8,
        "slug": "cross_impact_graph_frontier",
        "display_name": "Cross-Impact Graph Frontier",
        "objective": "Tie sleeve, account, symbol, and factor graphs into crowding and duplicate-exposure warnings.",
        "outputs": ["crowding_graph_packet", "duplicate_exposure_warning", "deconflict_hint"],
    },
    {
        "stage": 9,
        "slug": "route_retirement_court",
        "display_name": "Route Retirement Court",
        "objective": "Retire or down-rank expensive routes that do not improve accuracy, stability, or latency.",
        "outputs": ["route_retirement_candidate", "library_keep_vote", "fallback_route_hint"],
    },
    {
        "stage": 10,
        "slug": "soak_pause_guard",
        "display_name": "Soak/Pause Guard",
        "objective": "Stop further expansion when activation gates require evidence, walk-forward coverage, or training completion.",
        "outputs": ["frontier_stop_reason", "soak_requirement", "next_recheck_packet"],
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


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _status(payload: dict[str, Any], default: str = "missing") -> str:
    if not payload:
        return default
    return str(payload.get("overall_status") or payload.get("status") or payload.get("state") or default).strip().lower()


def _ordered_unique(items: list[Any]) -> list[str]:
    return list(dict.fromkeys(str(item) for item in items if str(item).strip()))


def _gate_blockers(project_root: Path) -> dict[str, Any]:
    health_root = project_root / "governance" / "health"
    quant_lanes = _load_json(health_root / "quant_strategy_lane_upgrades_latest.json")
    library_efficiency = _load_json(health_root / "library_efficiency_deepening_latest.json")
    deep_quant = _load_json(health_root / "deep_quant_layer_upgrade_latest.json")
    retrain = _load_json(health_root / "retrain_launch_latest.json")

    lane_gates = quant_lanes.get("gate_state") if isinstance(quant_lanes.get("gate_state"), dict) else {}
    lib_gates = library_efficiency.get("gate_state") if isinstance(library_efficiency.get("gate_state"), dict) else {}
    blockers: list[Any] = []
    blockers.extend(lane_gates.get("promotion_quality_failed_checks") or [])
    blockers.extend(f"promotion_readiness:{item}" for item in (lane_gates.get("promotion_readiness_blockers") or []))
    blockers.extend(lib_gates.get("blockers") or [])
    blockers.extend(deep_quant.get("activation_blockers") or [])
    if str(retrain.get("state") or "").strip().lower() == "running":
        blockers.append("large_training_batch_running_control_plane_only")

    runtime_green = bool(lane_gates.get("runtime_green", True)) and bool(lib_gates.get("runtime_green", True))
    storage_green = bool(lane_gates.get("storage_green", True)) and bool(lib_gates.get("storage_green", True))
    paper_400_ready = bool(lane_gates.get("paper_400_ready", True)) and bool(lib_gates.get("paper_400_ready", True))
    promotion_ready = bool(lane_gates.get("promotion_quality_ready", False)) and bool(lib_gates.get("promotion_quality_ready", False))
    blockers = _ordered_unique(blockers)
    return {
        "runtime_green": runtime_green,
        "storage_green": storage_green,
        "paper_400_ready": paper_400_ready,
        "promotion_quality_ready": promotion_ready,
        "training_batch_active": str(retrain.get("state") or "").strip().lower() == "running",
        "training_batch_pid": retrain.get("pid"),
        "blockers": blockers,
        "activation_allowed": bool(runtime_green and storage_green and paper_400_ready and promotion_ready and not blockers),
    }


def _stage_payload(stage: dict[str, Any], gates: dict[str, Any]) -> dict[str, Any]:
    control_plane_safe = bool(gates.get("runtime_green") and gates.get("storage_green"))
    activation_allowed = bool(gates.get("activation_allowed"))
    return {
        **stage,
        "state": "applied_control_plane" if control_plane_safe else "blocked_by_runtime_or_storage_guard",
        "control_plane_enabled": control_plane_safe,
        "advisory_enabled": control_plane_safe,
        "paper_rehearsal_enabled": control_plane_safe,
        "live_advisory_enabled": control_plane_safe,
        "paper_execution_authority_enabled": False,
        "live_execution_authority_enabled": False,
        "allocation_authority_enabled": False,
        "training_intake_authority_enabled": False,
        "activation_allowed": activation_allowed,
        "activation_blockers": list(gates.get("blockers") or []),
        "safety_policy": "advisory_control_plane_only_until_safety_guard_clears",
    }


def _recommended_env(payload: dict[str, Any]) -> dict[str, str]:
    return {
        "SAFETY_BOUNDED_ADVANCEMENT_FRONTIER_ENABLED": "1",
        "SAFETY_BOUNDED_ADVANCEMENT_STAGE_COUNT": str(payload.get("stage_count") or 0),
        "SAFETY_BOUNDED_ADVANCEMENT_MODE": str(payload.get("frontier_mode") or "control_plane_advisory"),
        "SAFETY_BOUNDED_ADVANCEMENT_STOP_ACTIVE": "1" if payload.get("safety_stop_active") else "0",
        "SAFETY_BOUNDED_ADVANCEMENT_PAPER_EXECUTION_AUTHORITY": "0",
        "SAFETY_BOUNDED_ADVANCEMENT_LIVE_EXECUTION_AUTHORITY": "0",
        "SAFETY_BOUNDED_ADVANCEMENT_ALLOCATION_AUTHORITY": "0",
        "SAFETY_BOUNDED_ADVANCEMENT_TRAINING_INTAKE_AUTHORITY": "0",
        "SAFETY_BOUNDED_ADVANCEMENT_NEXT_SCOPE": "soak_and_recheck_after_training_and_promotion_evidence",
    }


def _write_env_override(path: Path, env: dict[str, str]) -> bool:
    lines = ["# Auto-managed by scripts/ops/safety_bounded_advancement_frontier.py"]
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
    health_root = project_root / "governance" / "health"
    gates = _gate_blockers(project_root)
    stages = [_stage_payload(stage, gates) for stage in FRONTIER_STAGES]
    lib_eff = _load_json(health_root / "library_efficiency_deepening_latest.json")
    deep_quant = _load_json(health_root / "deep_quant_layer_upgrade_latest.json")
    quant_lanes = _load_json(health_root / "quant_strategy_lane_upgrades_latest.json")
    source_scores = [
        _safe_float(lib_eff.get("efficiency_score"), 0.0),
        _safe_float(deep_quant.get("coverage_ratio"), 0.0),
        1.0 if bool(quant_lanes.get("collection_runtime_active")) else 0.0,
    ]
    readiness_score = round(sum(source_scores) / max(len(source_scores), 1), 4)
    control_plane_count = sum(1 for stage in stages if bool(stage.get("control_plane_enabled")))
    safety_stop = bool(gates.get("blockers") or not gates.get("activation_allowed"))
    status = (
        "frontier_control_plane_applied_pause_for_soak"
        if safety_stop and control_plane_count == len(stages)
        else "frontier_activation_ready"
        if not safety_stop
        else "frontier_blocked_before_control_plane"
    )
    payload: dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "schema_version": 1,
        "ok": len(stages) == 10,
        "overall_status": status,
        "frontier_mode": "control_plane_advisory",
        "stage_count": len(stages),
        "control_plane_stage_count": control_plane_count,
        "advisory_stage_count": sum(1 for stage in stages if bool(stage.get("advisory_enabled"))),
        "paper_rehearsal_stage_count": sum(1 for stage in stages if bool(stage.get("paper_rehearsal_enabled"))),
        "live_advisory_stage_count": sum(1 for stage in stages if bool(stage.get("live_advisory_enabled"))),
        "paper_execution_authority_enabled": False,
        "live_execution_authority_enabled": False,
        "allocation_authority_enabled": False,
        "training_intake_authority_enabled": False,
        "safety_stop_active": safety_stop,
        "safety_stop_reason": list(gates.get("blockers") or []),
        "pause_kind": "soak_until_training_batch_and_promotion_evidence_clear" if safety_stop else "none",
        "readiness_score": readiness_score,
        "source_statuses": {
            "library_efficiency_deepening": _status(lib_eff),
            "deep_quant_layer_upgrade": _status(deep_quant),
            "quant_strategy_lane_upgrades": _status(quant_lanes),
        },
        "gate_state": gates,
        "stages": stages,
        "do_not_push_until_guard_clears": [
            "paper_execution_authority",
            "live_execution_authority",
            "allocation_authority",
            "training_intake_authority",
            "new_high_volume_collectors",
            "heavy_replay_or_large_training",
        ]
        if safety_stop
        else [],
        "next_recheck_commands": [
            "./scripts/ops/opsctl.sh health-fast --json",
            "./scripts/ops/opsctl.sh library-efficiency-deepening --json",
            "./scripts/ops/opsctl.sh deep-quant-layer-upgrade --json",
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
        "# Safety-Bounded Advancement Frontier",
        "",
        f"- timestamp_utc: `{payload.get('timestamp_utc')}`",
        f"- status: `{payload.get('overall_status')}`",
        f"- frontier_mode: `{payload.get('frontier_mode')}`",
        f"- stage_count: `{payload.get('stage_count')}`",
        f"- control_plane_stage_count: `{payload.get('control_plane_stage_count')}`",
        f"- safety_stop_active: `{payload.get('safety_stop_active')}`",
        f"- pause_kind: `{payload.get('pause_kind')}`",
        "",
        "## Stop Reason",
        "",
    ]
    reasons = payload.get("safety_stop_reason") if isinstance(payload.get("safety_stop_reason"), list) else []
    lines.extend(f"- `{reason}`" for reason in reasons) if reasons else lines.append("- none")
    lines.extend(["", "## Stages", ""])
    for stage in payload.get("stages") or []:
        if not isinstance(stage, dict):
            continue
        lines.extend(
            [
                f"### {stage.get('stage')}. {stage.get('display_name')}",
                "",
                f"- slug: `{stage.get('slug')}`",
                f"- state: `{stage.get('state')}`",
                f"- paper_execution_authority_enabled: `{stage.get('paper_execution_authority_enabled')}`",
                f"- live_execution_authority_enabled: `{stage.get('live_execution_authority_enabled')}`",
                f"- outputs: {', '.join(str(item) for item in stage.get('outputs') or [])}",
                "",
            ]
        )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Push advisory advancement until the safety guard requires a pause.")
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
            "safety_bounded_advancement_frontier "
            f"status={payload.get('overall_status')} "
            f"stages={payload.get('stage_count')} "
            f"control_plane={payload.get('control_plane_stage_count')} "
            f"safety_stop={int(bool(payload.get('safety_stop_active')))}"
        )
    return 0 if payload.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
