#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REGISTRY_PATH = PROJECT_ROOT / "master_bot_registry.json"
DEFAULT_DIAGNOSTICS_DIR = PROJECT_ROOT / "governance" / "training_diagnostics"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "governance" / "health" / "training_label_audit_latest.json"
DEFAULT_MAX_DIAGNOSTIC_AGE_HOURS = 72.0
DEFAULT_COLLECTION_TRAINING_THRESHOLD = 250

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from scripts.ops.training_labeling_intelligence import FREE_LABEL_CONTEXT_SOURCE_MAP
except Exception:
    FREE_LABEL_CONTEXT_SOURCE_MAP = {}


def _load_json(path: Path) -> dict[str, Any]:
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return obj if isinstance(obj, dict) else {}


def _registry_rows(path: Path) -> list[dict[str, Any]]:
    payload = _load_json(path)
    rows = payload.get("sub_bots") if isinstance(payload.get("sub_bots"), list) else []
    return [row for row in rows if isinstance(row, dict)]


def _bot_diagnostic(diag_dir: Path, bot_id: str) -> dict[str, Any]:
    if not bot_id:
        return {}
    return _load_json(diag_dir / f"{bot_id}_latest.json")


def _float(raw: Any, default: float = 0.0) -> float:
    try:
        return float(raw)
    except Exception:
        return float(default)


def _int(raw: Any, default: int = 0) -> int:
    try:
        return int(float(raw))
    except Exception:
        return int(default)


def _free_source_candidates_for_contexts(contexts: list[Any]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for raw in contexts:
        context = str(raw or "").strip()
        if not context:
            continue
        mapped = FREE_LABEL_CONTEXT_SOURCE_MAP.get(context)
        if mapped:
            out[context] = list(mapped)
    return out


def _label_contract_for_row(registry_row: dict[str, Any]) -> dict[str, Any]:
    for explicit in (
        registry_row.get("label_contract"),
        registry_row.get("training_label_contract"),
        registry_row.get("universal_label_contract"),
    ):
        if not isinstance(explicit, dict) or not explicit:
            continue
        label_family = str(explicit.get("label_family") or explicit.get("family") or "").strip()
        primary = str(explicit.get("primary_horizon") or explicit.get("primary_label_horizon") or "").strip()
        if label_family and primary:
            return {
                "label_family": label_family,
                "primary_horizon": primary,
                "aux_horizons": list(explicit.get("aux_horizons") or explicit.get("aux_label_horizons") or []),
                "required_context": list(explicit.get("required_context") or explicit.get("required_label_context") or []),
                "contract_version": str(explicit.get("contract_version") or registry_row.get("data_label_contract_version") or ""),
                "source": "registry",
            }
    bot_id = str(registry_row.get("bot_id") or "").strip().lower()
    slot_kind = str(registry_row.get("slot_kind") or "").strip().lower()
    role = str(registry_row.get("bot_role") or "").strip().lower()
    sleeve = str(registry_row.get("sleeve_profile") or registry_row.get("sleeve_family") or "").strip().lower()
    haystack = " ".join([bot_id, slot_kind, role, sleeve])
    contracts = [
        (
            ("monte_carlo", "quasi_monte_carlo", "latin_hypercube", "antithetic", "finite_difference", "fft_pricing", "trinomial", "heston", "merton", "quant_pricing"),
            "quant_pricing_research",
            "model_price_error_surface_stability",
            ["pricing_model_dispersion", "realized_vs_model_error", "tail_scenario_price_error", "variance_reduction_efficiency"],
            ["listed_option_surface", "realized_vol", "rates_context", "model_price_sensitivity_grid", "variance_reduction_diagnostics"],
        ),
        (
            ("kalman", "particle", "ornstein", "ou_mean", "state_space"),
            "state_space_filter_research",
            "hidden_state_signal_followthrough",
            ["filter_confidence", "regime_transition", "noise_adjusted_return"],
            ["runtime_feature_history", "market_micro_features", "macro_context", "state_filter_diagnostics"],
        ),
        (
            ("cvar", "copula", "tail_dependency", "tail_cluster", "scenario_loss"),
            "tail_dependency_research",
            "tail_loss_exceedance_5d",
            ["cvar_breach", "copula_tail_dependence", "correlation_break"],
            ["cross_sleeve_correlation_matrix", "scenario_stress_ladder", "tail_risk_surface"],
        ),
        (
            ("kelly", "genetic", "optimization", "overfit", "allocation_constraint"),
            "optimization_research",
            "walk_forward_parameter_stability",
            ["overfit_penalty", "allocation_stability", "post_cost_edge"],
            ["walk_forward_requalification", "optimization_search_trace", "execution_cost_context"],
        ),
        (
            ("sentiment", "nlp", "filings_language", "narrative"),
            "sentiment_nlp_research",
            "source_weighted_event_followthrough",
            ["source_confidence", "entity_relevance", "narrative_crowding"],
            ["news_source_consensus", "sec_filing_context", "macro_event_bulletins"],
        ),
        (
            ("day_trading", "same_session", "flatten"),
            "same_session",
            "same_session_close",
            ["1m", "5m", "15m", "60m", "flatten_before_close"],
            ["one_minute_bars", "vwap", "relative_volume", "no_overnight_hold"],
        ),
        (
            ("aggressive_intraday", "ultrafast", "opening_range", "vwap"),
            "intraday_fast",
            "5m_30m_forward_return",
            ["1m", "5m", "15m", "60m", "halt_reopen_liquidity"],
            ["one_minute_bars", "vwap", "spread_quality", "relative_volume"],
        ),
        (
            ("swing", "multi_day"),
            "multi_day",
            "2d_5d_forward_return",
            ["1d", "5d", "10d", "stop_followthrough"],
            ["daily_bars", "sector_context", "macro_context", "overnight_gap"],
        ),
        (
            ("conservative", "capital_preservation", "cash_parking"),
            "risk_adjusted_preservation",
            "drawdown_avoidance_5d",
            ["vol_adjusted_return", "max_drawdown", "cash_parking"],
            ["volatility_budget", "credit_stress", "liquidity_state", "defensive_sector_context"],
        ),
        (
            ("dividend", "income", "drip", "payout"),
            "income_total_return",
            "20d_total_return_income",
            ["payout_safety", "dividend_cut_risk", "ex_dividend_window"],
            ["ex_dividend_calendar", "payout_metrics", "rate_context", "total_return_bars"],
        ),
        (
            ("option", "gamma", "iv_", "0dte"),
            "options_surface",
            "iv_realized_1d_5d",
            ["gamma", "skew", "spread_quality", "event_vol_reset"],
            ["options_chain", "iv_surface", "open_interest", "bid_ask_spread"],
        ),
        (
            ("future", "/es", "curve", "basis"),
            "futures_event_session",
            "session_event_followthrough",
            ["basis", "curve", "overnight_gap", "macro_event_window"],
            ["futures_bars", "session_calendar", "basis_context", "macro_calendar"],
        ),
        (
            ("market_neutral", "pairs", "stat_arb"),
            "spread_convergence",
            "spread_zscore_reversion_3d",
            ["beta_neutral_residual", "correlation_break", "pair_drawdown"],
            ["pair_universe", "correlation_matrix", "factor_residuals", "borrow_liquidity"],
        ),
        (
            ("sector_master", "sector_rotation"),
            "sector_rotation_master",
            "sector_relative_strength_3d_10d",
            ["sector_breadth", "cross_sector_risk_budget"],
            ["sector_etf_bars", "breadth_context", "macro_context", "correlation_clusters"],
        ),
        (
            ("volatility_regime", "term_structure", "vix"),
            "volatility_regime",
            "term_structure_realized_vol_followthrough",
            ["iv_realized_spread", "vol_of_vol", "tail_hedge_state"],
            ["vix_term_structure", "options_surface", "realized_vol", "macro_event_window"],
        ),
        (
            ("position_lifecycle", "trim_add_hold", "tax_aware"),
            "position_management",
            "trim_add_hold_outcome",
            ["edge_decay", "risk_budget", "tax_lot_holding_period"],
            ["position_history", "tax_lots", "risk_budget", "signal_decay"],
        ),
        (
            ("source_rank", "credibility", "guidance_language", "teacher_champion", "correlation_risk", "execution_guard"),
            "infrastructure_guard",
            "guard_prevents_bad_runtime_action",
            ["false_positive_guard", "incident_prevention", "data_quality_delta"],
            ["source_scores", "incident_log", "runtime_health", "decision_context"],
        ),
    ]
    for tokens, family, primary, aux, context in contracts:
        if any(token in haystack for token in tokens):
            return {
                "label_family": family,
                "primary_horizon": primary,
                "aux_horizons": aux,
                "required_context": context,
            }
    return {
        "label_family": "generic_directional",
        "primary_horizon": "1d_forward_return",
        "aux_horizons": ["5d_forward_return", "risk_adjusted_return"],
        "required_context": ["price_bars", "volume", "market_context"],
    }


def _diagnostic_label_contract(runtime_meta: dict[str, Any]) -> dict[str, Any]:
    for key in ("label_contract", "training_label_contract"):
        raw = runtime_meta.get(key)
        if isinstance(raw, dict):
            return raw
    label_audit = runtime_meta.get("label_audit") if isinstance(runtime_meta.get("label_audit"), dict) else {}
    raw = label_audit.get("label_contract")
    return raw if isinstance(raw, dict) else {}


def _label_contract_complete(expected: dict[str, Any], observed: dict[str, Any]) -> bool:
    if not observed:
        return False
    expected_family = str(expected.get("label_family") or "").strip().lower()
    observed_family = str(observed.get("label_family") or observed.get("family") or "").strip().lower()
    observed_primary = str(observed.get("primary_horizon") or observed.get("primary_label_horizon") or "").strip()
    if expected_family and observed_family and expected_family != observed_family:
        return False
    return bool(observed_family and observed_primary)


def _recommendation(row: dict[str, Any]) -> str:
    lifecycle_state = str(row.get("lifecycle_state") or "").strip().lower()
    if not bool(row.get("diagnostic_present", False)):
        if lifecycle_state == "data_collection_only":
            return "create_collect_only_diagnostics"
        return "refresh_training_diagnostics"
    if not bool(row.get("diagnostic_fresh", True)):
        return "refresh_training_diagnostics"
    sample_count = _int(row.get("sample_count"))
    sequence_count = _int(row.get("sequence_count"))
    skipped_filtered = _int(row.get("skipped_filtered"))
    skipped_low_confidence = _int(row.get("skipped_low_confidence"))
    skipped_labels = _int(row.get("skipped_labels"))
    positive_rate = _float(row.get("positive_rate"))
    acted_coverage = _float(row.get("acted_coverage"), -1.0)
    acted_accuracy = _float(row.get("acted_accuracy"), -1.0)
    accuracy_lift = _float(row.get("accuracy_lift_over_majority"), 0.0)
    long_precision = _float(row.get("long_precision"), 0.0)
    short_precision = _float(row.get("short_precision"), 0.0)
    label_balance = _float(row.get("label_balance_score"), 0.0)
    label_depth_status = str(row.get("label_depth_status") or "").strip().lower()
    if (
        lifecycle_state in {"data_collection_only", "paper_live_data"}
        and sample_count < DEFAULT_COLLECTION_TRAINING_THRESHOLD
        and label_depth_status in {"materialize_label_depth", "collect_and_materialize_label_depth"}
    ):
        return "materialize_label_depth"
    if (
        lifecycle_state in {"data_collection_only", "paper_live_data"}
        and sample_count < DEFAULT_COLLECTION_TRAINING_THRESHOLD
        and label_depth_status == "label_depth_ready_for_real_diagnostic_refresh"
    ):
        return "refresh_training_diagnostics"
    if lifecycle_state == "data_collection_only" and sample_count < DEFAULT_COLLECTION_TRAINING_THRESHOLD:
        return "keep_collecting_until_threshold"
    if sample_count == 0 and sequence_count == 0:
        return "fix_shared_runtime_input"
    if sample_count == 0 and skipped_filtered > max(skipped_low_confidence, skipped_labels):
        return "relax_sample_filter"
    if sample_count == 0 and skipped_low_confidence > max(skipped_filtered, skipped_labels):
        return "relax_confidence_gate"
    if sample_count == 0 and skipped_labels > 0:
        return "rebalance_label_builder"
    if label_balance < 0.18 or positive_rate <= 0.03 or positive_rate >= 0.97:
        return "rebalance_label_builder"
    if acted_coverage >= 0.50 and accuracy_lift < 0.0:
        return "tighten_abstention_thresholds"
    if 0.0 <= acted_coverage <= 0.02 and sample_count > 0:
        return "loosen_abstention_thresholds"
    if acted_accuracy >= 0.0 and acted_accuracy < 0.53 and accuracy_lift < 0.0:
        return "tighten_or_relabel_for_quality"
    if long_precision > 0.0 and short_precision > 0.0 and abs(long_precision - short_precision) >= 0.18:
        return "use_side_specific_thresholds"
    if bool(row.get("label_upgrade_needed", False)):
        return "upgrade_label_contract"
    return "monitor"


def _audit_row(registry_row: dict[str, Any], diag_dir: Path, *, max_diagnostic_age_hours: float) -> dict[str, Any]:
    bot_id = str(registry_row.get("bot_id") or "").strip().lower()
    diag_path = diag_dir / f"{bot_id}_latest.json" if bot_id else Path()
    diag = _bot_diagnostic(diag_dir, bot_id)
    metrics = diag.get("metrics") if isinstance(diag.get("metrics"), dict) else {}
    runtime_meta = diag.get("runtime_meta") if isinstance(diag.get("runtime_meta"), dict) else {}
    label_audit = runtime_meta.get("label_audit") if isinstance(runtime_meta.get("label_audit"), dict) else {}
    label_depth_contract = (
        runtime_meta.get("label_depth_contract")
        if isinstance(runtime_meta.get("label_depth_contract"), dict)
        else {}
    )
    label_contract = _label_contract_for_row(registry_row)
    observed_contract = _diagnostic_label_contract(runtime_meta)
    if (
        not observed_contract
        and str(label_contract.get("source") or "").strip().lower() == "registry"
        and _label_contract_complete(label_contract, label_contract)
    ):
        observed_contract = dict(label_contract)
        observed_contract["source"] = "registry_fallback_for_legacy_diagnostic"
    contract_complete = _label_contract_complete(label_contract, observed_contract)
    diagnostic_age_hours = None
    if diag_path and diag_path.exists():
        try:
            modified = datetime.fromtimestamp(diag_path.stat().st_mtime, tz=timezone.utc)
            diagnostic_age_hours = max((datetime.now(timezone.utc) - modified).total_seconds() / 3600.0, 0.0)
        except Exception:
            diagnostic_age_hours = None
    sample_count = _int(diag.get("sample_count", runtime_meta.get("sample_count", 0)))
    skipped_filtered = _int(diag.get("skipped_filtered", runtime_meta.get("skipped_filtered", 0)))
    skipped_low_confidence = _int(diag.get("skipped_low_confidence", runtime_meta.get("skipped_low_confidence", 0)))
    skipped_labels = _int(diag.get("skipped_labels", runtime_meta.get("skipped_labels", 0)))
    attempted = max(sample_count + skipped_filtered + skipped_low_confidence + skipped_labels, 0)
    required_context = list(label_contract.get("required_context") or [])
    free_source_candidates = _free_source_candidates_for_contexts(required_context)
    out = {
        "bot_id": bot_id,
        "bot_role": str(registry_row.get("bot_role") or ""),
        "slot_kind": str(registry_row.get("slot_kind") or ""),
        "lifecycle_state": str(registry_row.get("lifecycle_state") or ""),
        "training_excluded": bool(registry_row.get("training_excluded", False)),
        "data_collection_active": bool(registry_row.get("data_collection_active", False)),
        "active": bool(registry_row.get("active", False)),
        "status": str(diag.get("status") or "missing_diagnostic"),
        "diagnostic_present": bool(diag_path and diag_path.exists()),
        "diagnostic_age_hours": round(float(diagnostic_age_hours), 3) if diagnostic_age_hours is not None else None,
        "diagnostic_fresh": bool(
            diagnostic_age_hours is not None and float(diagnostic_age_hours) <= max(float(max_diagnostic_age_hours), 0.0)
        ),
        "sample_count": sample_count,
        "eligible_sequences": _int(diag.get("eligible_sequences", runtime_meta.get("eligible_sequences", 0))),
        "sequence_count": _int(diag.get("sequence_count", runtime_meta.get("sequence_count", 0))),
        "observation_count": _int(diag.get("observation_count", runtime_meta.get("observation_count", 0))),
        "positive_rate": _float(diag.get("positive_rate", runtime_meta.get("positive_rate", 0.0))),
        "acted_coverage": _float(metrics.get("acted_coverage"), -1.0),
        "acted_accuracy": _float(metrics.get("acted_accuracy"), -1.0),
        "accuracy_lift_over_majority": _float(metrics.get("accuracy_lift_over_majority"), 0.0),
        "long_precision": _float(metrics.get("long_precision"), 0.0),
        "short_precision": _float(metrics.get("short_precision"), 0.0),
        "label_balance_score": _float(metrics.get("label_balance_score"), 0.0),
        "precision_balance_score": _float(metrics.get("precision_balance_score"), 0.0),
        "long_acted_count": _int(metrics.get("long_acted_count", 0)),
        "short_acted_count": _int(metrics.get("short_acted_count", 0)),
        "skipped_filtered": skipped_filtered,
        "skipped_low_confidence": skipped_low_confidence,
        "skipped_labels": skipped_labels,
        "acceptance_rate": round((sample_count / attempted), 6) if attempted > 0 else 0.0,
        "attempted_candidate_count": attempted,
        "label_audit": label_audit,
        "label_depth_status": str(diag.get("label_depth_status") or label_depth_contract.get("status") or ""),
        "label_depth_contract": label_depth_contract,
        "estimated_usable_sample_capacity": _int(label_depth_contract.get("estimated_usable_sample_capacity", 0)),
        "usable_sample_gap": _int(label_depth_contract.get("usable_sample_gap", 0)),
        "label_depth_next_action": str(label_depth_contract.get("next_action") or ""),
        "label_contract": label_contract,
        "observed_label_contract": observed_contract,
        "label_family": str(label_contract.get("label_family") or ""),
        "primary_label_horizon": str(label_contract.get("primary_horizon") or ""),
        "aux_label_horizons": list(label_contract.get("aux_horizons") or []),
        "required_label_context": required_context,
        "free_source_context_candidates": free_source_candidates,
        "free_source_context_count": len(free_source_candidates),
        "label_contract_complete": contract_complete,
        "label_upgrade_needed": bool(diag_path and diag_path.exists()) and not contract_complete,
        "label_upgrade_reason": "" if contract_complete else "diagnostic_missing_expected_label_contract",
        "diagnostics_path": str(diag_path) if bot_id else "",
    }
    out["recommendation"] = _recommendation(out)
    return out


def build_label_audit_payload(
    *,
    registry_path: Path,
    diagnostics_dir: Path,
    max_diagnostic_age_hours: float = DEFAULT_MAX_DIAGNOSTIC_AGE_HOURS,
) -> dict[str, Any]:
    rows = [
        _audit_row(row, diagnostics_dir, max_diagnostic_age_hours=max_diagnostic_age_hours)
        for row in _registry_rows(registry_path)
    ]
    active_rows = [row for row in rows if bool(row.get("active"))]
    recommendation_counts = Counter(str(row.get("recommendation") or "") for row in active_rows)
    source_context_counts = Counter(
        context
        for row in active_rows
        for context in (row.get("free_source_context_candidates") or {}).keys()
    )
    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "registry_path": str(registry_path),
        "diagnostics_dir": str(diagnostics_dir),
        "active_rows": len(active_rows),
        "recommendation_counts": dict(sorted(recommendation_counts.items())),
        "free_source_context_counts": dict(sorted(source_context_counts.items())),
        "free_source_context_active_bot_count": sum(
            1 for row in active_rows if bool(row.get("free_source_context_candidates"))
        ),
        "active_zero_sample": [row for row in active_rows if _int(row.get("sample_count")) == 0][:25],
        "active_overacting": [
            row for row in active_rows
            if _float(row.get("acted_coverage"), -1.0) >= 0.5
        ][:25],
        "active_underacting": [
            row for row in active_rows
            if 0.0 <= _float(row.get("acted_coverage"), -1.0) <= 0.02 and _int(row.get("sample_count")) > 0
        ][:25],
        "active_unbalanced_labels": [
            row for row in active_rows
            if _float(row.get("label_balance_score"), 1.0) < 0.18 or _float(row.get("positive_rate"), 0.5) <= 0.03 or _float(row.get("positive_rate"), 0.5) >= 0.97
        ][:25],
        "active_label_contract_upgrades": [
            row for row in active_rows if bool(row.get("label_upgrade_needed", False))
        ][:25],
        "top_actions": [],
    }
    top_actions: list[str] = []
    for name in [
        "create_collect_only_diagnostics",
        "upgrade_label_contract",
        "materialize_label_depth",
        "keep_collecting_until_threshold",
        "refresh_training_diagnostics",
        "fix_shared_runtime_input",
        "relax_sample_filter",
        "relax_confidence_gate",
        "rebalance_label_builder",
        "tighten_abstention_thresholds",
        "use_side_specific_thresholds",
    ]:
        if recommendation_counts.get(name, 0) > 0:
            top_actions.append(name)
    payload["top_actions"] = top_actions
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit runtime label quality and abstention behavior across the registry.")
    parser.add_argument("--registry-path", default=str(DEFAULT_REGISTRY_PATH))
    parser.add_argument("--diagnostics-dir", default=str(DEFAULT_DIAGNOSTICS_DIR))
    parser.add_argument("--output-path", default=str(DEFAULT_OUTPUT_PATH))
    parser.add_argument("--max-diagnostic-age-hours", type=float, default=DEFAULT_MAX_DIAGNOSTIC_AGE_HOURS)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    payload = build_label_audit_payload(
        registry_path=Path(args.registry_path).expanduser(),
        diagnostics_dir=Path(args.diagnostics_dir).expanduser(),
        max_diagnostic_age_hours=float(args.max_diagnostic_age_hours),
    )
    output_path = Path(args.output_path).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        print(
            "training_label_audit "
            f"active_rows={int(payload['active_rows'])} "
            f"zero_sample={len(payload['active_zero_sample'])} "
            f"overacting={len(payload['active_overacting'])} "
            f"underacting={len(payload['active_underacting'])}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
