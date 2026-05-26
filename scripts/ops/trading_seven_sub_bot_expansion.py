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


PACK_VERSION = "trading_seven_sub_bot_expansion_v1"
PACK_SLUG = "seven_collect_first_trading_sub_bots"
DEFAULT_REGISTRY_PATH = PROJECT_ROOT / "master_bot_registry.json"
DEFAULT_OUT_PATH = PROJECT_ROOT / "governance" / "health" / "trading_seven_sub_bot_expansion_latest.json"
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "trading_seven_sub_bot_expansion_v1.json"

TRADING_SEVEN_SPECS: list[dict[str, Any]] = [
    {
        "bot_id": "brain_refinery_v1654_intraday_momentum_continuation_trader",
        "bot_role": "signal_sub_bot",
        "slot_label": "Intraday Momentum Continuation Trader",
        "slot_kind": "aggressive_intraday_momentum_continuation_trader",
        "priority": "critical",
        "sleeve_profile": "intraday_aggressive",
        "sleeve_family": "intraday",
        "objective": "Collect trade-quality evidence for opening-drive continuation, volume participation, VWAP holds, and fast momentum exits.",
        "target_functions": ["market_micro_sync", "paper_profitability_control", "training_quality_control"],
        "preferred_regimes": ["open_drive_stress", "risk_on_trend", "event_volatility"],
        "bootstrap_teacher_bot_ids": ["brain_refinery_v45_intraday_open_close_regimes", "brain_refinery_v1651_intraday_liquidity_sweep_reversal_scalper"],
        "data_intake_collections": [
            "opening_drive_continuation_labels",
            "vwap_hold_momentum_context",
            "volume_participation_confirmation",
            "intraday_exit_timing_outcome",
            "paper_trade_outcome_labels",
        ],
        "rationale": "Adds a trading scout for true intraday follow-through instead of treating every fast move like a reversal candidate.",
    },
    {
        "bot_id": "brain_refinery_v1655_intraday_mean_reversion_chop_filter_trader",
        "bot_role": "signal_sub_bot",
        "slot_label": "Intraday Mean Reversion Chop Filter Trader",
        "slot_kind": "aggressive_intraday_mean_reversion_chop_filter_trader",
        "priority": "critical",
        "sleeve_profile": "intraday_aggressive",
        "sleeve_family": "intraday",
        "objective": "Collect chop, failed-continuation, VWAP snapback, and no-trade labels for intraday mean-reversion setups.",
        "target_functions": ["market_micro_sync", "paper_profitability_control", "calibration_abstention"],
        "preferred_regimes": ["thin_liquidity", "mixed_transition", "sideways_chop"],
        "bootstrap_teacher_bot_ids": ["brain_refinery_v13_choppy", "brain_refinery_v1644_intraday_open_drive_false_break_filter"],
        "data_intake_collections": [
            "sideways_chop_no_trade_labels",
            "vwap_snapback_context",
            "failed_continuation_labels",
            "range_reversion_quality",
            "paper_trade_outcome_labels",
        ],
        "rationale": "Gives the system a trading-style no-trade/chop specialist so it does not force action in sideways intraday markets.",
    },
    {
        "bot_id": "brain_refinery_v1656_swing_breakout_quality_trader",
        "bot_role": "signal_sub_bot",
        "slot_label": "Swing Breakout Quality Trader",
        "slot_kind": "swing_breakout_quality_trader",
        "priority": "high",
        "sleeve_profile": "swing_aggressive",
        "sleeve_family": "swing",
        "objective": "Collect multi-day breakout, pullback durability, relative strength, and failed-breakout labels for swing trades.",
        "target_functions": ["walk_forward_coverage_seed", "paper_profitability_control", "regime_control_plane"],
        "preferred_regimes": ["risk_on_trend", "slow_rotation", "mixed_transition"],
        "bootstrap_teacher_bot_ids": ["brain_refinery_v47_swing_1w_3w", "brain_refinery_v1645_swing_multi_day_breakout_quality_filter"],
        "data_intake_collections": [
            "multi_day_breakout_quality",
            "relative_strength_confirmation",
            "pullback_durability_context",
            "failed_breakout_trap_labels",
            "paper_trade_outcome_labels",
        ],
        "rationale": "Adds a true swing trading sub bot focused on breakouts that can survive more than one session.",
    },
    {
        "bot_id": "brain_refinery_v1657_options_vol_crush_reversal_trader",
        "bot_role": "options_sub_bot",
        "slot_label": "Options Vol Crush Reversal Trader",
        "slot_kind": "options_vol_crush_reversal_trader",
        "priority": "critical",
        "sleeve_profile": "options_aggressive",
        "sleeve_family": "options",
        "objective": "Collect post-event IV crush, skew reset, gamma reversal, and spread-cost labels for options reversal trades.",
        "target_functions": ["collect_options_flow_context", "paper_profitability_control", "risk_service"],
        "preferred_regimes": ["event_volatility", "open_drive_stress", "mixed_transition"],
        "bootstrap_teacher_bot_ids": ["brain_refinery_v122_vol_crush_event_iv_reset_overlay", "brain_refinery_v1652_options_gamma_pin_breakout_trader"],
        "data_intake_collections": [
            "post_event_iv_crush_labels",
            "skew_reset_reversal_context",
            "gamma_reversal_quality",
            "options_spread_slippage_realism",
            "paper_trade_outcome_labels",
        ],
        "rationale": "Adds an options trading scout for the event-aftershock zone where price direction and IV crush can fight each other.",
    },
    {
        "bot_id": "brain_refinery_v1658_crypto_spot_momentum_chop_switch",
        "bot_role": "signal_sub_bot",
        "slot_label": "Crypto Spot Momentum Chop Switch",
        "slot_kind": "crypto_spot_momentum_chop_switch",
        "priority": "high",
        "sleeve_profile": "crypto_spot",
        "sleeve_family": "crypto",
        "objective": "Collect BTC/ETH spot momentum, sideways chop, volume confirmation, and exchange microstructure labels.",
        "target_functions": ["crypto_market_sync", "decision_intelligence", "paper_profitability_control"],
        "preferred_regimes": ["event_volatility", "sideways_chop", "risk_on_trend"],
        "bootstrap_teacher_bot_ids": ["brain_refinery_v10_seasonal", "brain_refinery_v1653_crypto_futures_funding_momentum_switch"],
        "data_intake_collections": [
            "crypto_spot_momentum_labels",
            "sideways_chop_no_trade_labels",
            "spot_volume_confirmation",
            "exchange_microstructure_context",
            "paper_trade_outcome_labels",
        ],
        "rationale": "Adds a crypto spot trading scout so futures funding signals are checked against spot participation and chop quality.",
    },
    {
        "bot_id": "brain_refinery_v1659_futures_macro_followthrough_trader",
        "bot_role": "futures_sub_bot",
        "slot_label": "Futures Macro Follow-Through Trader",
        "slot_kind": "futures_macro_followthrough_trader",
        "priority": "high",
        "sleeve_profile": "futures_macro",
        "sleeve_family": "futures",
        "objective": "Collect ES/NQ/rates/commodities follow-through, failed macro impulse, and overnight-to-cash-session handoff labels.",
        "target_functions": ["macro_bulletin", "regime_control_plane", "paper_profitability_control"],
        "preferred_regimes": ["event_volatility", "risk_off_trend", "inflation_shock"],
        "bootstrap_teacher_bot_ids": ["brain_refinery_v90_macro_fomc_tone_liquidity", "brain_refinery_v104_futures_event_followthrough"],
        "data_intake_collections": [
            "futures_macro_followthrough_labels",
            "overnight_cash_session_handoff",
            "failed_macro_impulse_labels",
            "rates_equity_duration_context",
            "paper_trade_outcome_labels",
        ],
        "rationale": "Adds a futures trading scout to separate real macro follow-through from overnight moves that fail during cash trading.",
    },
    {
        "bot_id": "brain_refinery_v1660_fx_rates_cross_asset_confirmation_trader",
        "bot_role": "signal_sub_bot",
        "slot_label": "FX Rates Cross-Asset Confirmation Trader",
        "slot_kind": "fx_rates_cross_asset_confirmation_trader",
        "priority": "high",
        "sleeve_profile": "fx",
        "sleeve_family": "fx",
        "objective": "Collect USD, rates, gold, equity-duration, and carry-unwind confirmation labels for cross-asset trading decisions.",
        "target_functions": ["fx_market_sync", "source_verification", "regime_control_plane", "paper_profitability_control"],
        "preferred_regimes": ["inflation_shock", "risk_off_shock", "mixed_transition"],
        "bootstrap_teacher_bot_ids": ["brain_refinery_v1643_fx_usd_funding_carry_stress_guard", "brain_refinery_v92_macro_rates_curve_regime"],
        "data_intake_collections": [
            "fx_rates_confirmation_labels",
            "usd_funding_stress_context",
            "carry_unwind_detection",
            "gold_rates_equity_confirmation",
            "paper_trade_outcome_labels",
        ],
        "rationale": "Adds a trading scout that asks whether FX and rates agree before equity, futures, or crypto signals get more confidence.",
    },
]


def _safety_updates(spec: dict[str, Any]) -> dict[str, Any]:
    return {
        "trading_seven_expansion_version": PACK_VERSION,
        "trading_seven_expansion_pack": PACK_SLUG,
        "trading_seven_slot": True,
        "expansion_scope": "seven_trading_sub_bots_collect_only",
        "expansion_batch_size": len(TRADING_SEVEN_SPECS),
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
        "operator_intent": "seven_trading_sub_bot_expansion_collect_first",
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
            "rotation_block_reason": "trading_seven_expansion_data_collection_only",
            "training_excluded": True,
            "exclude_from_training": True,
            "training_exclusion_reason": "trading_seven_collecting_observations_before_training",
            "eligible_for_master_vote": False,
            "weight": 0.0,
            "preference_score": 0.0,
        }
    )
    return row


def plan_registry_expansion(registry: dict[str, Any]) -> dict[str, Any]:
    rows = _registry_rows(registry)
    existing = _bot_id_set(rows)
    skipped = [spec for spec in TRADING_SEVEN_SPECS if str(spec.get("bot_id") or "").strip().lower() in existing]
    missing = [spec for spec in TRADING_SEVEN_SPECS if str(spec.get("bot_id") or "").strip().lower() not in existing]
    planned_rows = [_planned_row(spec) for spec in missing]
    sleeve_profiles = ordered_unique(str(spec.get("sleeve_profile") or "") for spec in TRADING_SEVEN_SPECS)
    roles = ordered_unique(str(spec.get("bot_role") or "") for spec in TRADING_SEVEN_SPECS)
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
        "recommended_apply_command": ["./scripts/ops/opsctl.sh", "trading-seven-expansion", "--apply", "--json"],
        "recommended_actions": [
            "apply the seven trading sub bots as collect-only observers",
            "let paper outcome, slippage, chop, follow-through, and counterevidence labels accumulate before training",
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
        backup = lifecycle_dir / f"master_bot_registry.trading_seven_expansion_backup_{stamp}.json"
        if registry_path.exists():
            shutil.copy2(registry_path, backup)
            backup_path = str(backup)
        rows.extend(planned_rows)
        registry["sub_bots"] = rows
        _refresh_registry_summary(registry)
        summary = registry.get("summary") if isinstance(registry.get("summary"), dict) else {}
        summary["trading_seven_expansion_version"] = PACK_VERSION
        summary["trading_seven_expansion_bot_count"] = sum(1 for row in rows if row.get("trading_seven_slot"))
        summary["latest_trading_seven_expansion"] = {
            "timestamp_utc": iso_now(),
            "pack_slug": PACK_SLUG,
            "added_bot_count": len(planned_rows),
            "added_bot_ids": [str(row.get("bot_id") or "") for row in planned_rows],
            "scope": "seven_collect_first_trading_sub_bots",
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
        "bot_ids": [str(spec.get("bot_id") or "") for spec in TRADING_SEVEN_SPECS],
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
    parser = argparse.ArgumentParser(description="Stage or apply seven collect-first trading sub bots.")
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
            "trading_seven_sub_bot_expansion "
            f"mode={payload.get('mode')} "
            f"overall_status={payload.get('overall_status')} "
            f"planned_bot_count={summary.get('planned_bot_count')} "
            f"added_bot_count={apply_result.get('added_bot_count')}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
