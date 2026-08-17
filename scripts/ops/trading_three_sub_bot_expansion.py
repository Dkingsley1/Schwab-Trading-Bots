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


PACK_VERSION = "trading_three_sub_bot_expansion_v1"
PACK_SLUG = "intraday_options_crypto_trading_three"
DEFAULT_REGISTRY_PATH = PROJECT_ROOT / "master_bot_registry.json"
DEFAULT_OUT_PATH = PROJECT_ROOT / "governance" / "health" / "trading_three_sub_bot_expansion_latest.json"
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "trading_three_sub_bot_expansion_v1.json"

TRADING_THREE_SPECS: list[dict[str, Any]] = [
    {
        "bot_id": "brain_refinery_v1651_intraday_liquidity_sweep_reversal_scalper",
        "bot_role": "signal_sub_bot",
        "slot_label": "Intraday Liquidity Sweep Reversal Scalper",
        "slot_kind": "aggressive_intraday_liquidity_sweep_reversal_scalper",
        "priority": "critical",
        "sleeve_profile": "intraday_aggressive",
        "sleeve_family": "intraday",
        "objective": "Collect trade-quality evidence for liquidity sweep reversals, failed breaks, VWAP reclaims, and fast intraday exit timing.",
        "target_functions": ["market_micro_sync", "paper_profitability_control", "training_quality_control"],
        "preferred_regimes": ["open_drive_stress", "thin_liquidity", "event_volatility"],
        "bootstrap_teacher_bot_ids": [
            "brain_refinery_v45_intraday_open_close_regimes",
            "brain_refinery_v47_swing_1w_3w",
        ],
        "data_intake_collections": [
            "liquidity_sweep_reversal_labels",
            "vwap_reclaim_reject_context",
            "opening_range_failed_break_labels",
            "intraday_exit_timing_outcome",
            "paper_trade_outcome_labels",
        ],
        "rationale": "Adds a real trading-style intraday scout focused on turning red/sideways chop into clearer reversal or no-trade evidence.",
    },
    {
        "bot_id": "brain_refinery_v1652_options_gamma_pin_breakout_trader",
        "bot_role": "options_sub_bot",
        "slot_label": "Options Gamma Pin Breakout Trader",
        "slot_kind": "options_gamma_pin_breakout_trader",
        "priority": "critical",
        "sleeve_profile": "options_aggressive",
        "sleeve_family": "options",
        "objective": "Collect option-trade evidence around gamma pins, pin breaks, IV crush risk, skew, and realistic spread/slippage costs.",
        "target_functions": ["collect_options_flow_context", "paper_profitability_control", "risk_service"],
        "preferred_regimes": ["event_volatility", "open_drive_stress", "thin_liquidity"],
        "bootstrap_teacher_bot_ids": [
            "brain_refinery_v117_iv_skew_dislocation_overlay",
            "brain_refinery_v111_slippage_capacity_limiter",
        ],
        "data_intake_collections": [
            "gamma_pin_breakout_labels",
            "iv_crush_risk_context",
            "options_spread_slippage_realism",
            "skew_confirmation_context",
            "paper_trade_outcome_labels",
        ],
        "rationale": "Adds a trading sub bot that can learn when options breakouts are worth the spread and when gamma/IV conditions make them traps.",
    },
    {
        "bot_id": "brain_refinery_v1653_crypto_futures_funding_momentum_switch",
        "bot_role": "futures_sub_bot",
        "slot_label": "Crypto Futures Funding Momentum Switch",
        "slot_kind": "crypto_futures_funding_momentum_switch",
        "priority": "critical",
        "sleeve_profile": "crypto_futures",
        "sleeve_family": "crypto_futures",
        "objective": "Collect BTC/ETH trade evidence for funding flips, liquidation cascades, open-interest confirmation, and sideways chop avoidance.",
        "target_functions": ["crypto_market_sync", "decision_intelligence", "paper_profitability_control"],
        "preferred_regimes": ["event_volatility", "thin_liquidity", "risk_off_shock"],
        "bootstrap_teacher_bot_ids": [
            "brain_refinery_v1646_crypto_futures_basis_liquidation_guard",
            "brain_refinery_v10_seasonal",
        ],
        "data_intake_collections": [
            "funding_flip_momentum_labels",
            "open_interest_confirmation",
            "liquidation_cascade_detection",
            "sideways_chop_no_trade_labels",
            "paper_trade_outcome_labels",
        ],
        "rationale": "Adds a trading-style crypto futures scout so BTC/ETH decisions can separate funding-driven momentum from sideways chop.",
    },
]


def _safety_updates(spec: dict[str, Any]) -> dict[str, Any]:
    return {
        "trading_three_expansion_version": PACK_VERSION,
        "trading_three_expansion_pack": PACK_SLUG,
        "trading_three_slot": True,
        "expansion_scope": "three_trading_sub_bots_collect_only",
        "expansion_batch_size": len(TRADING_THREE_SPECS),
        "data_collection_compute_guard_mode": "soft_cap",
        "data_collection_storage_guard_mode": "metadata_first",
        "collection_throttle": "thin_digest",
        "max_daily_mb_per_bot": 2,
        "max_collection_events_per_minute": 1,
        "no_live_execution": True,
        "paper_trade_excluded_until_training_ready": True,
        "trade_style": "observer_until_promoted",
        "requires_profitability_feedback_before_training": True,
        "requires_counterevidence_before_promotion": True,
        "operator_intent": "trading_sub_bot_expansion_collect_first",
        "trading_rationale": str(spec.get("rationale") or ""),
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
            "rotation_block_reason": "trading_three_expansion_data_collection_only",
            "training_excluded": True,
            "exclude_from_training": True,
            "training_exclusion_reason": "trading_three_collecting_observations_before_training",
            "eligible_for_master_vote": False,
            "weight": 0.0,
            "preference_score": 0.0,
        }
    )
    return row


def plan_registry_expansion(registry: dict[str, Any]) -> dict[str, Any]:
    rows = _registry_rows(registry)
    existing = _bot_id_set(rows)
    skipped = [spec for spec in TRADING_THREE_SPECS if str(spec.get("bot_id") or "").strip().lower() in existing]
    missing = [spec for spec in TRADING_THREE_SPECS if str(spec.get("bot_id") or "").strip().lower() not in existing]
    planned_rows = [_planned_row(spec) for spec in missing]
    sleeve_profiles = ordered_unique(str(spec.get("sleeve_profile") or "") for spec in TRADING_THREE_SPECS)
    roles = ordered_unique(str(spec.get("bot_role") or "") for spec in TRADING_THREE_SPECS)
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
        "bot_roles": roles,
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
            "bot_roles": plan["bot_roles"],
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
        "recommended_apply_command": ["./scripts/ops/opsctl.sh", "trading-three-expansion", "--apply", "--json"],
        "recommended_actions": [
            "apply the three trading sub bots as collect-only observers",
            "let paper outcome, slippage, and counterevidence labels accumulate before training",
            "keep live execution disabled until promotion and execution gates explicitly allow it",
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
        backup = lifecycle_dir / f"master_bot_registry.trading_three_expansion_backup_{stamp}.json"
        if registry_path.exists():
            shutil.copy2(registry_path, backup)
            backup_path = str(backup)
        rows.extend(planned_rows)
        registry["sub_bots"] = rows
        _refresh_registry_summary(registry)
        summary = registry.get("summary") if isinstance(registry.get("summary"), dict) else {}
        summary["trading_three_expansion_version"] = PACK_VERSION
        summary["trading_three_expansion_bot_count"] = sum(1 for row in rows if row.get("trading_three_slot"))
        summary["latest_trading_three_expansion"] = {
            "timestamp_utc": iso_now(),
            "pack_slug": PACK_SLUG,
            "added_bot_count": len(planned_rows),
            "added_bot_ids": [str(row.get("bot_id") or "") for row in planned_rows],
            "scope": "intraday_options_crypto_trading_sub_bots",
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
        "bot_ids": [str(spec.get("bot_id") or "") for spec in TRADING_THREE_SPECS],
        "sleeve_profiles": plan.get("sleeve_profiles", []),
        "bot_roles": plan.get("bot_roles", []),
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
    parser = argparse.ArgumentParser(description="Stage or apply three collect-first trading sub bots.")
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
            "trading_three_sub_bot_expansion "
            f"mode={payload.get('mode')} "
            f"overall_status={payload.get('overall_status')} "
            f"planned_bot_count={summary.get('planned_bot_count')} "
            f"added_bot_count={apply_result.get('added_bot_count')}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
