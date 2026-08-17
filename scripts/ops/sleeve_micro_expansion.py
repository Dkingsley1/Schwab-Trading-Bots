#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
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
else:
    from .long_runtime_common import PROJECT_ROOT, iso_now, load_json, ordered_unique, write_payload
    from .roster_expansion_slots import _refresh_registry_summary, _slot_registry_row


PACK_VERSION = "sleeve_micro_expansion_v1"
PACK_SLUG = "ten_across_major_sleeves"
DEFAULT_REGISTRY_PATH = PROJECT_ROOT / "master_bot_registry.json"
DEFAULT_OUT_PATH = PROJECT_ROOT / "governance" / "health" / "sleeve_micro_expansion_latest.json"
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "sleeve_micro_expansion_v1.json"

SLEEVE_MICRO_SPECS: list[dict[str, Any]] = [
    {
        "bot_id": "brain_refinery_v1638_default_evidence_disagreement_arbiter",
        "bot_role": "infrastructure_sub_bot",
        "slot_label": "Default Evidence Disagreement Arbiter",
        "slot_kind": "default_evidence_disagreement_arbiter",
        "priority": "high",
        "sleeve_profile": "default",
        "sleeve_family": "default",
        "objective": "Collect cross-source disagreement, confirmation-bias, and paper-outcome evidence for the default sleeve.",
        "target_functions": ["decision_intelligence", "paper_profitability_control", "source_verification"],
        "preferred_regimes": ["all_weather", "mixed_transition", "fragile_transition"],
        "bootstrap_teacher_bot_ids": ["brain_refinery_v10_seasonal", "brain_refinery_v50"],
        "data_intake_collections": [
            "ensemble_disagreement_surface",
            "confirmation_bias_counterevidence",
            "paper_trade_outcome_labels",
        ],
        "rationale": "The default sleeve needs a neutral observer that asks what evidence disagrees before confidence rises.",
    },
    {
        "bot_id": "brain_refinery_v1639_aggressive_breakout_confirmation_scout",
        "bot_role": "signal_sub_bot",
        "slot_label": "Aggressive Breakout Confirmation Scout",
        "slot_kind": "aggressive_breakout_confirmation_scout",
        "priority": "high",
        "sleeve_profile": "aggressive",
        "sleeve_family": "aggressive",
        "objective": "Collect breakout follow-through, failed-breakout, and volume confirmation labels for aggressive decisions.",
        "target_functions": ["training_quality_control", "paper_profitability_control", "regime_control_plane"],
        "preferred_regimes": ["risk_on_trend", "event_volatility", "thin_liquidity"],
        "bootstrap_teacher_bot_ids": ["brain_refinery_v10_seasonal", "brain_refinery_v59"],
        "data_intake_collections": [
            "breakout_follow_through_labels",
            "failed_breakout_trap_labels",
            "volume_participation_confirmation",
        ],
        "rationale": "Aggressive sleeves should get faster without becoming gullible around noisy breakouts.",
    },
    {
        "bot_id": "brain_refinery_v1640_conservative_drawdown_stability_guard",
        "bot_role": "infrastructure_sub_bot",
        "slot_label": "Conservative Drawdown Stability Guard",
        "slot_kind": "conservative_drawdown_stability_guard",
        "priority": "high",
        "sleeve_profile": "conservative",
        "sleeve_family": "conservative",
        "objective": "Collect conservative-sleeve drawdown, false-safety, and hedge-quality evidence before promotion gates.",
        "target_functions": ["risk_service", "paper_profitability_control", "portfolio_allocator"],
        "preferred_regimes": ["risk_off_shock", "risk_off_trend", "fragile_transition"],
        "bootstrap_teacher_bot_ids": ["brain_refinery_v31_defensive_rotation", "brain_refinery_v99_defensive_dividend_concentration"],
        "data_intake_collections": [
            "drawdown_stability_labels",
            "hedge_quality_context",
            "false_safety_regime_detection",
        ],
        "rationale": "The conservative sleeve needs to prove it is reducing risk, not just taking fewer trades.",
    },
    {
        "bot_id": "brain_refinery_v1641_dividend_quality_cashflow_regime_guard",
        "bot_role": "signal_sub_bot",
        "slot_label": "Dividend Quality Cashflow Regime Guard",
        "slot_kind": "dividend_quality_cashflow_regime_guard",
        "priority": "medium",
        "sleeve_profile": "dividend_income",
        "sleeve_family": "dividend",
        "objective": "Collect dividend safety, cashflow quality, rate sensitivity, and ex-dividend trap labels.",
        "target_functions": ["dividend_drip_sync", "portfolio_allocator", "risk_service"],
        "preferred_regimes": ["slow_rotation", "inflation_shock", "risk_off_trend"],
        "bootstrap_teacher_bot_ids": ["brain_refinery_v99_defensive_dividend_concentration", "brain_refinery_v31_defensive_rotation"],
        "data_intake_collections": [
            "dividend_safety_score",
            "cashflow_quality_context",
            "ex_dividend_trap_labels",
            "rate_sensitivity_income_context",
        ],
        "rationale": "Dividend decisions should distinguish quality income from yield traps and rate-sensitive weakness.",
    },
    {
        "bot_id": "brain_refinery_v1642_bond_duration_credit_spread_router",
        "bot_role": "signal_sub_bot",
        "slot_label": "Bond Duration Credit Spread Router",
        "slot_kind": "bond_duration_credit_spread_router",
        "priority": "medium",
        "sleeve_profile": "bond",
        "sleeve_family": "bond",
        "objective": "Collect duration, credit-spread, curve, and equity-duration spillover labels for fixed-income context.",
        "target_functions": ["macro_bulletin", "regime_control_plane", "portfolio_allocator"],
        "preferred_regimes": ["inflation_shock", "risk_off_trend", "fragile_transition"],
        "bootstrap_teacher_bot_ids": ["brain_refinery_v92_macro_rates_curve_regime", "brain_refinery_v90_macro_fomc_tone_liquidity"],
        "data_intake_collections": [
            "duration_shock_context",
            "credit_spread_waterfall",
            "curve_steepener_flattener_labels",
        ],
        "rationale": "Bond context is a core explanation layer for equity multiples, defensive rotation, and risk-off moves.",
    },
    {
        "bot_id": "brain_refinery_v1643_fx_usd_funding_carry_stress_guard",
        "bot_role": "signal_sub_bot",
        "slot_label": "FX USD Funding Carry Stress Guard",
        "slot_kind": "fx_usd_funding_carry_stress_guard",
        "priority": "medium",
        "sleeve_profile": "fx",
        "sleeve_family": "fx",
        "objective": "Collect USD funding, carry unwind, proxy agreement, and cross-asset FX stress evidence.",
        "target_functions": ["fx_market_sync", "source_verification", "regime_control_plane"],
        "preferred_regimes": ["risk_off_shock", "inflation_shock", "mixed_transition"],
        "bootstrap_teacher_bot_ids": ["brain_refinery_v90_macro_fomc_tone_liquidity", "brain_refinery_v92_macro_rates_curve_regime"],
        "data_intake_collections": [
            "fx_proxy_agreement_labels",
            "usd_funding_stress_context",
            "carry_unwind_detection",
        ],
        "rationale": "FX needs cleaner proxy agreement and funding-stress context before it can explain market moves confidently.",
    },
    {
        "bot_id": "brain_refinery_v1644_intraday_open_drive_false_break_filter",
        "bot_role": "signal_sub_bot",
        "slot_label": "Intraday Open Drive False Break Filter",
        "slot_kind": "aggressive_intraday_open_drive_false_break_filter",
        "priority": "high",
        "sleeve_profile": "intraday_aggressive",
        "sleeve_family": "intraday",
        "objective": "Collect opening-drive, false-break, liquidity fade, and continuation labels for intraday aggressive trades.",
        "target_functions": ["market_micro_sync", "paper_profitability_control", "training_quality_control"],
        "preferred_regimes": ["open_drive_stress", "event_volatility", "thin_liquidity"],
        "bootstrap_teacher_bot_ids": ["brain_refinery_v45_intraday_open_close_regimes", "brain_refinery_v43_intraday_ultrafast_proxy"],
        "data_intake_collections": [
            "open_drive_continuation_labels",
            "false_break_reversal_labels",
            "liquidity_fade_context",
        ],
        "rationale": "Intraday aggression needs a specialist that knows when the opening move is real and when it is bait.",
    },
    {
        "bot_id": "brain_refinery_v1645_swing_multi_day_breakout_quality_filter",
        "bot_role": "signal_sub_bot",
        "slot_label": "Swing Multi-Day Breakout Quality Filter",
        "slot_kind": "swing_multi_day_breakout_quality_filter",
        "priority": "medium",
        "sleeve_profile": "swing_aggressive",
        "sleeve_family": "swing",
        "objective": "Collect multi-day breakout quality, pullback durability, and failed-continuation labels for swing decisions.",
        "target_functions": ["walk_forward_coverage_seed", "paper_profitability_control", "regime_control_plane"],
        "preferred_regimes": ["risk_on_trend", "mixed_transition", "slow_rotation"],
        "bootstrap_teacher_bot_ids": ["brain_refinery_v10_seasonal", "brain_refinery_v50"],
        "data_intake_collections": [
            "multi_day_breakout_quality",
            "pullback_durability_context",
            "failed_continuation_labels",
        ],
        "rationale": "Swing aggressive signals need confirmation that their edge survives beyond the first impulse.",
    },
    {
        "bot_id": "brain_refinery_v1646_crypto_futures_basis_liquidation_guard",
        "bot_role": "futures_sub_bot",
        "slot_label": "Crypto Futures Basis Liquidation Guard",
        "slot_kind": "crypto_futures_basis_liquidation_guard",
        "priority": "high",
        "sleeve_profile": "crypto_futures",
        "sleeve_family": "crypto_futures",
        "objective": "Collect BTC/ETH basis, funding, liquidation cascade, and open-interest confirmation evidence.",
        "target_functions": ["crypto_market_sync", "decision_intelligence", "paper_profitability_control"],
        "preferred_regimes": ["event_volatility", "thin_liquidity", "risk_off_shock"],
        "bootstrap_teacher_bot_ids": ["brain_refinery_v10_seasonal", "brain_refinery_v59"],
        "data_intake_collections": [
            "crypto_futures_basis_context",
            "funding_rate_stress_labels",
            "liquidation_cascade_detection",
            "open_interest_confirmation",
        ],
        "rationale": "Crypto futures needs direct evidence for sideways chop, liquidation risk, and funding/basis disagreement.",
    },
    {
        "bot_id": "brain_refinery_v1647_options_on_futures_skew_convexity_guard",
        "bot_role": "options_sub_bot",
        "slot_label": "Options On Futures Skew Convexity Guard",
        "slot_kind": "options_on_futures_skew_convexity_guard",
        "priority": "medium",
        "sleeve_profile": "options_on_futures",
        "sleeve_family": "options_on_futures",
        "objective": "Collect futures-options skew, convexity, term-structure, and event-window risk labels.",
        "target_functions": ["collect_options_flow_context", "risk_service", "paper_profitability_control"],
        "preferred_regimes": ["event_volatility", "risk_off_shock", "inflation_shock"],
        "bootstrap_teacher_bot_ids": ["brain_refinery_v99_defensive_dividend_concentration", "brain_refinery_v31_defensive_rotation"],
        "data_intake_collections": [
            "futures_options_skew_context",
            "convexity_cost_surface",
            "event_window_margin_context",
        ],
        "rationale": "Options-on-futures needs its own convexity observer instead of borrowing equity-options assumptions.",
    },
]


def _registry_rows(registry: dict[str, Any]) -> list[dict[str, Any]]:
    rows = registry.get("sub_bots")
    if isinstance(rows, list):
        return [row for row in rows if isinstance(row, dict)]
    rows = registry.get("bots")
    if isinstance(rows, list):
        return [row for row in rows if isinstance(row, dict)]
    return []


def _bot_id_set(rows: list[dict[str, Any]]) -> set[str]:
    return {str(row.get("bot_id") or "").strip().lower() for row in rows if str(row.get("bot_id") or "").strip()}


def _max_bot_version(rows: list[dict[str, Any]]) -> int:
    versions: list[int] = []
    for row in rows:
        match = re.search(r"_v(\d+)", str(row.get("bot_id") or ""))
        if match:
            versions.append(int(match.group(1)))
    return max(versions) if versions else 0


def _safety_updates(spec: dict[str, Any]) -> dict[str, Any]:
    return {
        "sleeve_micro_expansion_version": PACK_VERSION,
        "sleeve_micro_expansion_pack": PACK_SLUG,
        "micro_expansion_slot": True,
        "expansion_scope": "ten_total_across_major_sleeves",
        "expansion_batch_size": len(SLEEVE_MICRO_SPECS),
        "data_collection_compute_guard_mode": "soft_cap",
        "data_collection_storage_guard_mode": "metadata_first",
        "collection_throttle": "thin_digest",
        "max_daily_mb_per_bot": 2,
        "max_collection_events_per_minute": 1,
        "no_live_execution": True,
        "paper_trade_excluded_until_training_ready": True,
        "confirmation_bias_guard_enabled": True,
        "counterevidence_required_before_promotion": True,
        "profitability_feedback_required_before_promotion": True,
        "operator_intent": "small_safe_expansion_after_backlog_green",
        "expansion_notes": str(spec.get("rationale") or ""),
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
            "rotation_block_reason": "sleeve_micro_expansion_data_collection_only",
            "training_excluded": True,
            "exclude_from_training": True,
            "training_exclusion_reason": "micro_expansion_collecting_observations_before_training",
            "eligible_for_master_vote": False,
            "weight": 0.0,
            "preference_score": 0.0,
        }
    )
    return row


def plan_registry_expansion(registry: dict[str, Any]) -> dict[str, Any]:
    rows = _registry_rows(registry)
    existing = _bot_id_set(rows)
    skipped = [spec for spec in SLEEVE_MICRO_SPECS if str(spec.get("bot_id") or "").strip().lower() in existing]
    missing = [spec for spec in SLEEVE_MICRO_SPECS if str(spec.get("bot_id") or "").strip().lower() not in existing]
    planned_rows = [_planned_row(spec) for spec in missing]
    sleeve_profiles = ordered_unique(str(spec.get("sleeve_profile") or "") for spec in SLEEVE_MICRO_SPECS)
    sleeve_families = ordered_unique(str(spec.get("sleeve_family") or "") for spec in SLEEVE_MICRO_SPECS)
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
        "sleeve_profile_count": len(sleeve_profiles),
        "sleeve_profiles": sleeve_profiles,
        "sleeve_family_count": len(sleeve_families),
        "sleeve_families": sleeve_families,
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
    status = "ready" if planned == 0 else "planned"
    return {
        "timestamp_utc": iso_now(),
        "schema_version": 1,
        "ok": True,
        "overall_status": status,
        "mode": "dry_run",
        "registry_path": str(registry_path),
        "summary": {
            key: plan[key]
            for key in (
                "pack_version",
                "pack_slug",
                "current_total_bots",
                "current_max_bot_version",
                "planned_bot_count",
                "skipped_existing_count",
                "planned_total_after_apply",
                "sleeve_profile_count",
                "sleeve_profiles",
                "sleeve_family_count",
                "sleeve_families",
            )
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
        "recommended_apply_command": [
            "./scripts/ops/opsctl.sh",
            "sleeve-micro-expansion",
            "--apply",
            "--json",
        ],
        "recommended_actions": [
            "apply the 10-bot micro expansion only as data collection observers",
            "keep training blocked until each new bot reaches its minimum observation and age floor",
            "use these bots to enrich sleeve-specific counterevidence, paper outcome, and profitability labels",
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
        backup = lifecycle_dir / f"master_bot_registry.sleeve_micro_expansion_backup_{stamp}.json"
        if registry_path.exists():
            shutil.copy2(registry_path, backup)
            backup_path = str(backup)
        rows.extend(planned_rows)
        registry["sub_bots"] = rows
        _refresh_registry_summary(registry)
        summary = registry.get("summary") if isinstance(registry.get("summary"), dict) else {}
        summary["sleeve_micro_expansion_version"] = PACK_VERSION
        summary["sleeve_micro_expansion_bot_count"] = sum(1 for row in rows if row.get("micro_expansion_slot"))
        summary["latest_sleeve_micro_expansion"] = {
            "timestamp_utc": iso_now(),
            "pack_slug": PACK_SLUG,
            "added_bot_count": len(planned_rows),
            "added_bot_ids": [str(row.get("bot_id") or "") for row in planned_rows],
            "scope": "ten_total_across_major_sleeves",
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
        "bot_ids": [str(spec.get("bot_id") or "") for spec in SLEEVE_MICRO_SPECS],
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
    parser = argparse.ArgumentParser(description="Stage or apply a bounded 10-bot, collect-only expansion across major sleeve families.")
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
            "sleeve_micro_expansion "
            f"mode={payload.get('mode')} "
            f"overall_status={payload.get('overall_status')} "
            f"planned_bot_count={summary.get('planned_bot_count')} "
            f"added_bot_count={apply_result.get('added_bot_count')}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
