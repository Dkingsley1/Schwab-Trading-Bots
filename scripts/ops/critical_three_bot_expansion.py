#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from scripts.ops.long_runtime_common import PROJECT_ROOT, iso_now, load_json, ordered_unique, write_payload
    from scripts.ops.roster_expansion_slots import _refresh_registry_summary, _slot_registry_row
    from scripts.ops.sleeve_micro_expansion import _bot_id_set, _max_bot_version, _registry_rows
else:
    from .long_runtime_common import PROJECT_ROOT, iso_now, load_json, ordered_unique, write_payload
    from .roster_expansion_slots import _refresh_registry_summary, _slot_registry_row
    from .sleeve_micro_expansion import _bot_id_set, _max_bot_version, _registry_rows


PACK_VERSION = "critical_three_bot_expansion_v1"
PACK_SLUG = "backlog_launcher_profitability_critical_three"
DEFAULT_REGISTRY_PATH = PROJECT_ROOT / "master_bot_registry.json"
DEFAULT_OUT_PATH = PROJECT_ROOT / "governance" / "health" / "critical_three_bot_expansion_latest.json"
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "critical_three_bot_expansion_v1.json"

CRITICAL_THREE_SPECS: list[dict[str, Any]] = [
    {
        "bot_id": "brain_refinery_v1648_backlog_pressure_oracle_guard",
        "bot_role": "infrastructure_sub_bot",
        "slot_label": "Backlog Pressure Oracle Guard",
        "slot_kind": "data_plane_backpressure_oracle_guard",
        "priority": "critical",
        "sleeve_profile": "data_plane_backpressure_resilience",
        "sleeve_family": "system_governor",
        "objective": "Collect lead indicators that predict backlog, sparse JSONL pressure, writer stalls, and stale pending work before they degrade the system.",
        "target_functions": [
            "backpressure_super_drainer",
            "writer_cycle_coordinator",
            "ingestion_storage_control",
            "system_needs_intelligence",
        ],
        "preferred_regimes": ["all_weather", "runtime_pressure", "fragile_transition"],
        "bootstrap_teacher_bot_ids": [
            "brain_refinery_v110_runtime_input_resilience_guard",
            "brain_refinery_v116_drawdown_circuit_allocator",
        ],
        "data_intake_collections": [
            "backlog_lead_indicator_trace",
            "writer_wave_effectiveness_labels",
            "sparse_jsonl_pressure_signature",
            "oldest_pending_age_recovery_labels",
            "drainer_accelerator_counterfactuals",
        ],
        "rationale": "Backlog has been the chronic system constraint; this bot gives the governor better early warning and exact fix framing.",
    },
    {
        "bot_id": "brain_refinery_v1649_sleeve_launcher_recovery_commander",
        "bot_role": "infrastructure_sub_bot",
        "slot_label": "Sleeve Launcher Recovery Commander",
        "slot_kind": "halt_recovery_sleeve_launcher_recovery_commander",
        "priority": "critical",
        "sleeve_profile": "halt_recovery_stability",
        "sleeve_family": "runtime_recovery",
        "objective": "Collect launcher-down root causes, sleeve handoff failures, restart-storm signatures, and safe thaw evidence.",
        "target_functions": [
            "sleeve_isolation_guard",
            "blackstart_recovery",
            "watchdog_intelligence",
            "all_sleeves_launcher",
        ],
        "preferred_regimes": ["all_weather", "runtime_pressure", "restart_recovery"],
        "bootstrap_teacher_bot_ids": [
            "brain_refinery_v110_runtime_input_resilience_guard",
            "brain_refinery_v267_infra_teacher_execution_quality_champion",
        ],
        "data_intake_collections": [
            "sleeve_launcher_down_event_trace",
            "restart_storm_signature",
            "sleeve_handoff_lock_state",
            "safe_thaw_sequence_labels",
            "watchdog_escalation_outcome",
        ],
        "rationale": "Sleeve launcher notifications should become diagnosable events with a safe recovery playbook, not recurring mystery alerts.",
    },
    {
        "bot_id": "brain_refinery_v1650_paper_profitability_attribution_teacher",
        "bot_role": "infrastructure_sub_bot",
        "slot_label": "Paper Profitability Attribution Teacher",
        "slot_kind": "transaction_cost_paper_profitability_attribution_teacher",
        "priority": "critical",
        "sleeve_profile": "transaction_cost_slippage_intelligence",
        "sleeve_family": "execution_quality",
        "objective": "Collect sleeve-level paper PnL attribution, slippage realism, decision-quality, and counterevidence labels for teacher/student training.",
        "target_functions": [
            "paper_profitability_control",
            "execution_quality_lab",
            "teacher_student_mesh",
            "decision_intelligence",
        ],
        "preferred_regimes": ["all_weather", "event_volatility", "thin_liquidity"],
        "bootstrap_teacher_bot_ids": [
            "brain_refinery_v111_slippage_capacity_limiter",
            "brain_refinery_v80_execution_feasibility_sentinel",
        ],
        "data_intake_collections": [
            "paper_pnl_attribution_by_sleeve",
            "slippage_realism_context",
            "decision_counterevidence_trace",
            "trade_reason_outcome_join",
            "teacher_student_profitability_feedback",
        ],
        "rationale": "Profitability needs explainable feedback loops so weak bots learn why trades lose instead of only seeing pass/fail scores.",
    },
]


def _safety_updates(spec: dict[str, Any]) -> dict[str, Any]:
    return {
        "critical_three_expansion_version": PACK_VERSION,
        "critical_three_expansion_pack": PACK_SLUG,
        "critical_three_slot": True,
        "expansion_scope": "three_critical_system_quality_bots",
        "expansion_batch_size": len(CRITICAL_THREE_SPECS),
        "data_collection_compute_guard_mode": "soft_cap",
        "data_collection_storage_guard_mode": "metadata_first",
        "collection_throttle": "thin_digest",
        "max_daily_mb_per_bot": 2,
        "max_collection_events_per_minute": 1,
        "no_live_execution": True,
        "paper_trade_excluded_until_training_ready": True,
        "operator_intent": "critical_collect_only_quality_expansion",
        "criticality_reason": str(spec.get("rationale") or ""),
    }


def _planned_row(spec: dict[str, Any]) -> dict[str, Any]:
    row = _slot_registry_row(spec)
    row.update(_safety_updates(spec))
    row.update(
        {
            "trading_enabled": False,
            "paper_trading_enabled": False,
            "live_trading_enabled": False,
            "allocation_enabled": False,
            "execution_enabled": False,
            "rotation_blocked": True,
            "rotation_block_reason": "critical_three_expansion_data_collection_only",
            "training_excluded": True,
            "exclude_from_training": True,
            "training_exclusion_reason": "critical_three_collecting_observations_before_training",
            "eligible_for_master_vote": False,
            "weight": 0.0,
            "preference_score": 0.0,
        }
    )
    return row


def plan_registry_expansion(registry: dict[str, Any]) -> dict[str, Any]:
    rows = _registry_rows(registry)
    existing = _bot_id_set(rows)
    skipped = [spec for spec in CRITICAL_THREE_SPECS if str(spec.get("bot_id") or "").strip().lower() in existing]
    missing = [spec for spec in CRITICAL_THREE_SPECS if str(spec.get("bot_id") or "").strip().lower() not in existing]
    planned_rows = [_planned_row(spec) for spec in missing]
    sleeve_profiles = ordered_unique(str(spec.get("sleeve_profile") or "") for spec in CRITICAL_THREE_SPECS)
    return {
        "pack_version": PACK_VERSION,
        "pack_slug": PACK_SLUG,
        "current_total_bots": len(rows),
        "current_max_bot_version": _max_bot_version(rows),
        "planned_bot_count": len(planned_rows),
        "skipped_existing_count": len(skipped),
        "planned_total_after_apply": len(rows) + len(planned_rows),
        "planned_bot_ids": [str(row.get("bot_id") or "") for row in planned_rows],
        "skipped_existing_bot_ids": [str(spec.get("bot_id") or "") for spec in skipped],
        "sleeve_profiles": sleeve_profiles,
        "planned_rows": planned_rows,
        "safety_contract": {
            "data_collection_only": True,
            "execution_enabled": False,
            "allocation_enabled": False,
            "paper_trading_enabled": False,
            "training_excluded_until_threshold": True,
            "max_daily_mb_per_bot": 2,
            "collection_throttle": "thin_digest",
            "protected_volume_policy": "do_not_touch_/Volumes/VIDEO",
        },
    }


def build_payload(project_root: Path = PROJECT_ROOT, *, registry_path: Path = DEFAULT_REGISTRY_PATH) -> dict[str, Any]:
    registry = load_json(registry_path)
    plan = plan_registry_expansion(registry)
    planned = int(plan.get("planned_bot_count", 0) or 0)
    return {
        "timestamp_utc": iso_now(),
        "schema_version": 1,
        "ok": True,
        "overall_status": "ready" if planned == 0 else "planned",
        "mode": "dry_run",
        "registry_path": str(registry_path),
        "summary": {
            "pack_version": plan["pack_version"],
            "pack_slug": plan["pack_slug"],
            "current_total_bots": plan["current_total_bots"],
            "current_max_bot_version": plan["current_max_bot_version"],
            "planned_bot_count": plan["planned_bot_count"],
            "skipped_existing_count": plan["skipped_existing_count"],
            "planned_total_after_apply": plan["planned_total_after_apply"],
            "sleeve_profiles": plan["sleeve_profiles"],
        },
        "planned_bot_ids": plan["planned_bot_ids"],
        "skipped_existing_bot_ids": plan["skipped_existing_bot_ids"],
        "planned_bots": [
            {
                "bot_id": row.get("bot_id"),
                "bot_role": row.get("bot_role"),
                "sleeve_profile": row.get("sleeve_profile"),
                "sleeve_family": row.get("sleeve_family"),
                "slot_label": row.get("slot_label"),
                "minimum_training_observations": row.get("minimum_training_observations"),
                "minimum_data_collection_days": row.get("minimum_data_collection_days"),
                "target_functions": row.get("target_functions"),
                "data_intake_collections": row.get("data_intake_collections"),
            }
            for row in plan["planned_rows"]
        ],
        "safety_contract": plan["safety_contract"],
        "recommended_apply_command": ["./scripts/ops/opsctl.sh", "critical-three-expansion", "--apply", "--json"],
        "recommended_actions": [
            "apply the three critical bots as collect-only observers",
            "use backlog, launcher, and paper-profitability labels to improve the system's exact fix recommendations",
            "keep these bots out of training and rotation until their observation floors are met",
        ],
    }


def apply_registry(
    project_root: Path = PROJECT_ROOT,
    *,
    registry_path: Path = DEFAULT_REGISTRY_PATH,
    config_path: Path = DEFAULT_CONFIG_PATH,
) -> dict[str, Any]:
    registry = load_json(registry_path)
    rows = _registry_rows(registry)
    plan = plan_registry_expansion(registry)
    planned_rows = list(plan.get("planned_rows") or [])
    backup_path = ""
    if planned_rows:
        lifecycle_dir = project_root / "governance" / "lifecycle"
        lifecycle_dir.mkdir(parents=True, exist_ok=True)
        stamp = iso_now().replace(":", "").replace("+00:00", "Z")
        backup = lifecycle_dir / f"master_bot_registry.critical_three_expansion_backup_{stamp}.json"
        if registry_path.exists():
            shutil.copy2(registry_path, backup)
            backup_path = str(backup)
        rows.extend(planned_rows)
        registry["sub_bots"] = rows
        _refresh_registry_summary(registry)
        summary = registry.get("summary") if isinstance(registry.get("summary"), dict) else {}
        summary["critical_three_expansion_version"] = PACK_VERSION
        summary["critical_three_expansion_bot_count"] = sum(1 for row in rows if row.get("critical_three_slot"))
        summary["latest_critical_three_expansion"] = {
            "timestamp_utc": iso_now(),
            "pack_slug": PACK_SLUG,
            "added_bot_count": len(planned_rows),
            "added_bot_ids": [str(row.get("bot_id") or "") for row in planned_rows],
            "scope": "backlog_launcher_profitability",
            "execution_enabled": False,
        }
        target = max(int(summary.get("target_platform_total_bots") or 0), len(rows))
        summary["target_platform_total_bots"] = target
        summary["target_platform_total_bots_met"] = len(rows) >= target
        summary["max_bot_version"] = _max_bot_version(rows)
        registry["summary"] = summary
        registry["updated_at_utc"] = iso_now()
        registry_path.write_text(json.dumps(registry, ensure_ascii=True, indent=2), encoding="utf-8")

    config_payload = {
        "timestamp_utc": iso_now(),
        "schema_version": 1,
        "pack_version": PACK_VERSION,
        "pack_slug": PACK_SLUG,
        "bot_ids": [str(spec.get("bot_id") or "") for spec in CRITICAL_THREE_SPECS],
        "sleeve_profiles": plan.get("sleeve_profiles", []),
        "safety_contract": plan.get("safety_contract", {}),
        "applied": bool(planned_rows),
        "added_bot_count": len(planned_rows),
        "skipped_existing_bot_ids": plan.get("skipped_existing_bot_ids", []),
    }
    write_payload(config_path, config_payload)

    return {
        "applied": bool(planned_rows),
        "added_bot_count": len(planned_rows),
        "added_bot_ids": [str(row.get("bot_id") or "") for row in planned_rows],
        "skipped_existing_count": int(plan.get("skipped_existing_count", 0) or 0),
        "skipped_existing_bot_ids": list(plan.get("skipped_existing_bot_ids") or []),
        "backup_path": backup_path,
        "registry_path": str(registry_path),
        "config_path": str(config_path),
        "planned_total_after_apply": int(plan.get("planned_total_after_apply", len(rows)) or len(rows)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Stage or apply the three most important collect-only system-quality bots.")
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--registry", default=str(DEFAULT_REGISTRY_PATH))
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--out-file", default=str(DEFAULT_OUT_PATH))
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    registry_path = Path(args.registry).expanduser()
    config_path = Path(args.config).expanduser()
    out_path = Path(args.out_file).expanduser()

    apply_result = {
        "applied": False,
        "added_bot_count": 0,
        "added_bot_ids": [],
        "skipped_existing_count": 0,
        "skipped_existing_bot_ids": [],
        "backup_path": "",
        "registry_path": str(registry_path),
        "config_path": str(config_path),
    }
    if args.apply:
        apply_result = apply_registry(project_root, registry_path=registry_path, config_path=config_path)

    payload = build_payload(project_root, registry_path=registry_path)
    payload["mode"] = "applied" if args.apply else "dry_run"
    payload["apply_result"] = apply_result
    write_payload(out_path, payload)

    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        summary = payload.get("summary") if isinstance(payload.get("summary"), dict) else {}
        print(
            "critical_three_bot_expansion "
            f"mode={payload.get('mode')} "
            f"overall_status={payload.get('overall_status')} "
            f"planned_bot_count={summary.get('planned_bot_count')} "
            f"added_bot_count={apply_result.get('added_bot_count')}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
