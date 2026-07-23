#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, deque
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from scripts.ops.long_runtime_common import PROJECT_ROOT, iso_now, load_json, ordered_unique, write_payload
else:
    from .long_runtime_common import PROJECT_ROOT, iso_now, load_json, ordered_unique, write_payload


DEFAULT_OUT_PATH = PROJECT_ROOT / "governance" / "health" / "market_cycle_state_latest.json"
DEFAULT_HISTORY_PATH = PROJECT_ROOT / "governance" / "market_cycle" / "market_cycle_state_history.jsonl"

PROFILE_FILES = {
    "bond": "bond_equities_schwab",
    "conservative": "conservative_equities_schwab",
    "default": "default_crypto_schwab",
    "dividend": "dividend_equities_schwab",
    "aggressive": "aggressive_equities_schwab",
    "crypto_futures": "crypto_futures_crypto_schwab",
    "intraday_aggressive": "intraday_aggressive_equities_schwab",
    "long_term_core_etf": "long_term_core_etf_equities_schwab",
    "long_term_dividend": "long_term_dividend_equities_schwab",
    "schwab_futures": "schwab_futures_equities_schwab",
    "swing_aggressive": "swing_aggressive_equities_schwab",
}

AGGRESSIVE_PROFILES = {"aggressive", "intraday_aggressive", "swing_aggressive", "crypto_futures", "schwab_futures"}
DEFENSIVE_PROFILES = {"bond", "conservative", "dividend", "long_term_core_etf", "long_term_dividend"}

RISK_ON_FEATURES = (
    "flow_risk_on_norm",
    "breadth_risk_on_norm",
    "market_crypto_risk_on_crypto_alignment_norm",
    "fx_risk_on_alignment_norm",
    "bond_credit_risk_on_norm",
    "cot_risk_on_norm",
    "cot_equity_risk_on_norm",
)
RISK_OFF_FEATURES = (
    "flow_risk_off_norm",
    "breadth_risk_off_norm",
    "bond_credit_risk_off_norm",
    "live_macro_risk_off_pressure_norm",
    "cot_bond_risk_off_norm",
)
DEFENSIVE_FEATURES = (
    "flow_defensive_rotation_norm",
    "defensive_rotation_norm",
    "bond_credit_risk_off_norm",
)
TREND_FEATURES = (
    "core_cross_asset_confirmation_norm",
    "lead_lag_confirmation_norm",
    "flow_conviction_norm",
    "allocation_confidence_norm",
)
STRESS_FEATURES = (
    "flow_stress_norm",
    "market_regime_stress_norm",
    "live_macro_risk_off_pressure_norm",
    "breadth_risk_off_norm",
)
EDGE_FEATURES = (
    "allocation_trade_edge_norm",
    "expected_edge_norm",
    "trade_edge_norm",
)
CONFIDENCE_FEATURES = (
    "allocation_confidence_norm",
    "core_cross_asset_confirmation_norm",
    "lead_lag_confirmation_norm",
)


def _as_dict(raw: Any) -> dict[str, Any]:
    return raw if isinstance(raw, dict) else {}


def _safe_float(raw: Any, default: float = 0.0) -> float:
    try:
        return float(raw)
    except Exception:
        return float(default)


def _safe_int(raw: Any, default: int = 0) -> int:
    try:
        return int(float(raw))
    except Exception:
        return int(default)


def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(float(lo), min(float(hi), float(value)))


def _parse_jsonl_lines(lines: list[bytes], *, limit: int) -> list[dict[str, Any]]:
    rows: deque[dict[str, Any]] = deque(maxlen=max(int(limit), 1))
    for line in lines:
        raw = line.strip()
        if not raw:
            continue
        try:
            payload = json.loads(raw.decode("utf-8"))
        except Exception:
            continue
        if isinstance(payload, dict):
            rows.append(payload)
    return list(rows)


def _tail_jsonl(path: Path, *, limit: int) -> list[dict[str, Any]]:
    try:
        size = path.stat().st_size
    except Exception:
        return []
    if size <= 0:
        return []
    for window in (8 * 1024 * 1024, 32 * 1024 * 1024):
        try:
            offset = max(size - window, 0)
            with path.open("rb") as handle:
                handle.seek(offset)
                data = handle.read(min(size, window))
            lines = data.splitlines()
            if offset > 0 and lines:
                lines = lines[1:]
            rows = _parse_jsonl_lines(lines, limit=limit)
            if rows or offset == 0:
                return rows
        except Exception:
            return []
    return []


def _available_decision_days(project_root: Path) -> list[str]:
    days: set[str] = set()
    for channel in sorted(set(PROFILE_FILES.values())):
        root = project_root / "governance" / "channels" / "decision" / channel
        try:
            paths = list(root.glob("decision_*.jsonl"))
        except Exception:
            paths = []
        for path in paths:
            day = path.name.removeprefix("decision_").removesuffix(".jsonl")
            if len(day) != 8 or not day.isdigit():
                continue
            try:
                if path.stat().st_size <= 0:
                    continue
            except Exception:
                continue
            days.add(day)
    return sorted(days)


def _decision_day(project_root: Path) -> str:
    days = _available_decision_days(project_root)
    if days:
        return days[-1]
    paper = load_json(project_root / "governance" / "health" / "sleeve_profitability_dashboard_latest.json")
    day = str(paper.get("paper_day") or "").strip()
    return day if day else ""


def _nested_sources(row: dict[str, Any]) -> list[dict[str, Any]]:
    sources = [row]
    for key in (
        "flow_awareness_features",
        "lead_lag_features",
        "allocation_confidence",
        "market_context_features",
        "external_context",
        "feature_vector",
        "features",
    ):
        nested = row.get(key)
        if isinstance(nested, dict):
            sources.append(nested)
    return sources


def _feature(row: dict[str, Any], names: tuple[str, ...] | list[str], *, default: float | None = None) -> float | None:
    values: list[float] = []
    for source in _nested_sources(row):
        for name in names:
            if name in source:
                values.append(_clamp(_safe_float(source.get(name), 0.0)))
    if not values:
        return default
    return sum(values) / max(len(values), 1)


def _action(row: dict[str, Any]) -> str:
    for path in (
        ("master_action",),
        ("master_dispatch", "action"),
        ("master_intent_action",),
        ("execution_guard", "action"),
        ("options_margin_guard", "action"),
        ("futures_margin_guard", "action"),
        ("action",),
    ):
        cursor: Any = row
        for key in path:
            cursor = cursor.get(key) if isinstance(cursor, dict) else None
        text = str(cursor or "").strip().upper()
        if text:
            return text
    return "HOLD"


def _mean(values: list[float], default: float = 0.0) -> float:
    return sum(values) / max(len(values), 1) if values else float(default)


def _summarize_profile(project_root: Path, *, profile: str, day: str, limit: int) -> dict[str, Any]:
    channel = PROFILE_FILES.get(profile, f"{profile}_equities_schwab")
    path = project_root / "governance" / "channels" / "decision" / channel / f"decision_{day}.jsonl" if day else project_root / "__missing__"
    rows = _tail_jsonl(path, limit=limit) if day else []
    actions = Counter(_action(row) for row in rows)
    risk_on = [_feature(row, RISK_ON_FEATURES) for row in rows]
    risk_off = [_feature(row, RISK_OFF_FEATURES) for row in rows]
    defensive = [_feature(row, DEFENSIVE_FEATURES) for row in rows]
    trend = [_feature(row, TREND_FEATURES) for row in rows]
    stress = [_feature(row, STRESS_FEATURES) for row in rows]
    edge = [_feature(row, EDGE_FEATURES) for row in rows]
    confidence = [_feature(row, CONFIDENCE_FEATURES) for row in rows]
    n = max(len(rows), 1)
    return {
        "profile": profile,
        "path": str(path),
        "exists": bool(day and path.exists()),
        "rows_sampled": len(rows),
        "action_counts": dict(sorted(actions.items())),
        "buy_ratio": round(actions.get("BUY", 0) / n, 6),
        "sell_ratio": round(actions.get("SELL", 0) / n, 6),
        "hold_ratio": round(actions.get("HOLD", 0) / n, 6),
        "avg_risk_on_norm": round(_mean([v for v in risk_on if v is not None]), 6),
        "avg_risk_off_norm": round(_mean([v for v in risk_off if v is not None]), 6),
        "avg_defensive_rotation_norm": round(_mean([v for v in defensive if v is not None]), 6),
        "avg_trend_confirmation_norm": round(_mean([v for v in trend if v is not None]), 6),
        "avg_stress_norm": round(_mean([v for v in stress if v is not None]), 6),
        "avg_edge_norm": round(_mean([v for v in edge if v is not None]), 6),
        "avg_confidence_norm": round(_mean([v for v in confidence if v is not None]), 6),
    }


def _weighted_profile_mean(profiles: dict[str, dict[str, Any]], key: str) -> float:
    total_weight = 0
    total = 0.0
    for row in profiles.values():
        weight = _safe_int(row.get("rows_sampled"), 0)
        if weight <= 0:
            continue
        total += _safe_float(row.get(key), 0.0) * weight
        total_weight += weight
    return total / max(total_weight, 1)


def _action_ratio(profiles: dict[str, dict[str, Any]], action: str) -> float:
    total = 0
    count = 0
    for row in profiles.values():
        rows = _safe_int(row.get("rows_sampled"), 0)
        actions = _as_dict(row.get("action_counts"))
        total += rows
        count += _safe_int(actions.get(action.upper()), 0)
    return count / max(total, 1)


def _regime_family(regime: str) -> str:
    text = str(regime or "").strip().lower()
    if "risk_on" in text or "expansion" in text or "rebound" in text:
        return "risk_on"
    if "risk_off" in text or "contraction" in text or "shock" in text:
        return "risk_off"
    if "late" in text or "fragile" in text or "high_vol" in text:
        return "fragile"
    if "range" in text or "chop" in text or "transition" in text or "mixed" in text:
        return "transition"
    return "unknown"


def _cycle_scores(signals: dict[str, float]) -> dict[str, float]:
    risk_on = signals["risk_on_norm"]
    risk_off = signals["risk_off_norm"]
    defensive = signals["defensive_rotation_norm"]
    trend = signals["trend_confirmation_norm"]
    stress = signals["stress_norm"]
    edge = signals["edge_norm"]
    confidence = signals["confidence_norm"]
    buy_ratio = signals["buy_ratio"]
    sell_ratio = signals["sell_ratio"]
    hold_ratio = signals["hold_ratio"]
    return {
        "expansion": round(_clamp((0.42 * risk_on) + (0.20 * trend) + (0.18 * confidence) + (0.12 * edge) + (0.08 * (1.0 - risk_off))), 6),
        "late_cycle_distribution": round(_clamp((0.26 * risk_on) + (0.24 * risk_off) + (0.20 * defensive) + (0.20 * stress) + (0.10 * (1.0 - confidence))), 6),
        "contraction": round(_clamp((0.40 * risk_off) + (0.20 * defensive) + (0.20 * stress) + (0.10 * (1.0 - trend)) + (0.10 * sell_ratio)), 6),
        "rebound": round(_clamp((0.35 * risk_on) + (0.22 * trend) + (0.18 * (1.0 - stress)) + (0.15 * confidence) + (0.10 * buy_ratio)), 6),
        "defensive_high_vol_chop": round(_clamp((0.34 * stress) + (0.26 * hold_ratio) + (0.20 * defensive) + (0.12 * (1.0 - trend)) + (0.08 * risk_off)), 6),
        "rangebound_chop": round(_clamp((0.35 * (1.0 - abs(risk_on - risk_off))) + (0.30 * hold_ratio) + (0.20 * (1.0 - trend)) + (0.15 * (1.0 - edge))), 6),
    }


def _classify_cycle(scores: dict[str, float], signals: dict[str, float]) -> tuple[str, str, list[str]]:
    risk_on = signals["risk_on_norm"]
    risk_off = signals["risk_off_norm"]
    defensive = signals["defensive_rotation_norm"]
    trend = signals["trend_confirmation_norm"]
    stress = signals["stress_norm"]
    hold_ratio = signals["hold_ratio"]
    reasons: list[str] = []

    if stress >= 0.72 and risk_off >= 0.55:
        return "contraction", "risk_off_shock", ["stress_and_risk_off_are_both_elevated"]
    if stress >= 0.70 and hold_ratio >= 0.65:
        return "defensive_high_vol_chop", "high_vol_chop", ["high_stress_with_most_sleeves_holding"]
    if risk_on >= risk_off + 0.16 and trend >= 0.45:
        return "expansion", "risk_on_trend", ["risk_on_leads_risk_off_with_confirmation"]
    if risk_off >= risk_on + 0.12:
        return "contraction", "risk_off_trend", ["risk_off_leads_risk_on"]
    if risk_on > risk_off and stress <= 0.45 and trend >= 0.45:
        return "rebound", "recovery_rerisk", ["risk_on_is_recovering_while_stress_is_contained"]
    if risk_on >= 0.45 and risk_off >= 0.40 and (defensive >= 0.35 or stress >= 0.45):
        return "late_cycle_distribution", "fragile_late_cycle", ["risk_on_and_risk_off_are_both_present"]
    if hold_ratio >= 0.65 or trend < 0.35:
        return "rangebound_chop", "rangebound_chop", ["hold_ratio_or_low_confirmation_points_to_chop"]

    selected = max(scores.items(), key=lambda item: item[1])[0]
    if selected == "expansion":
        return selected, "risk_on_trend", ["expansion_score_is_highest"]
    if selected == "contraction":
        return selected, "risk_off_trend", ["contraction_score_is_highest"]
    if selected == "rebound":
        return selected, "recovery_rerisk", ["rebound_score_is_highest"]
    if selected == "late_cycle_distribution":
        return selected, "fragile_late_cycle", ["late_cycle_score_is_highest"]
    if selected == "defensive_high_vol_chop":
        return selected, "high_vol_chop", ["defensive_high_vol_chop_score_is_highest"]
    return selected, "rangebound_chop", ["rangebound_score_is_highest"]


def _sleeve_bias(cycle_phase: str) -> dict[str, Any]:
    table = {
        "expansion": {
            "aggressive": 0.18,
            "core": 0.12,
            "dividend": 0.02,
            "bond": -0.06,
            "cash": -0.08,
        },
        "rebound": {
            "aggressive": 0.10,
            "core": 0.10,
            "dividend": 0.04,
            "bond": -0.02,
            "cash": -0.04,
        },
        "late_cycle_distribution": {
            "aggressive": -0.10,
            "core": 0.02,
            "dividend": 0.12,
            "bond": 0.08,
            "cash": 0.04,
        },
        "contraction": {
            "aggressive": -0.22,
            "core": -0.10,
            "dividend": 0.12,
            "bond": 0.16,
            "cash": 0.10,
        },
        "rangebound_chop": {
            "aggressive": -0.04,
            "core": 0.02,
            "dividend": 0.04,
            "bond": 0.02,
            "cash": 0.02,
        },
        "defensive_high_vol_chop": {
            "aggressive": -0.12,
            "core": -0.04,
            "dividend": 0.10,
            "bond": 0.10,
            "cash": 0.08,
        },
    }
    bias = table.get(cycle_phase, table["rangebound_chop"])
    return {
        "mode": "shadow_sleeve_bias_only",
        "paper_or_live_budget_changes_allowed": False,
        "bias_delta": bias,
        "policy": "cycle bias is advisory until separately validated by replay and promotion gates",
    }


def _build_crosscheck(regime_control: dict[str, Any], market_regime: str) -> dict[str, Any]:
    existing = str(regime_control.get("regime_state") or "").strip()
    existing_family = _regime_family(existing)
    engine_family = _regime_family(market_regime)
    agreement = bool(existing_family != "unknown" and existing_family == engine_family)
    return {
        "existing_regime_state": existing or "missing",
        "existing_regime_family": existing_family,
        "engine_regime_family": engine_family,
        "agreement": agreement,
        "stance_label": str(regime_control.get("stance_label") or ""),
        "stance_score": _safe_float(regime_control.get("stance_score"), 0.0),
        "source_status": str(regime_control.get("overall_status") or "missing"),
    }


def build_payload(
    project_root: Path = PROJECT_ROOT,
    *,
    sample_limit: int = 1200,
    min_rows: int = 25,
) -> dict[str, Any]:
    health = project_root / "governance" / "health"
    day = _decision_day(project_root)
    profiles = {
        profile: _summarize_profile(project_root, profile=profile, day=day, limit=sample_limit)
        for profile in PROFILE_FILES
    }
    regime_control = load_json(health / "regime_control_plane_latest.json")
    market_posture = load_json(health / "market_posture_control_latest.json")
    derived_state = load_json(health / "derived_state_latest.json")

    rows_sampled = sum(_safe_int(row.get("rows_sampled"), 0) for row in profiles.values())
    regime_scores = _as_dict(regime_control.get("scores"))
    regime_shock = _clamp(_safe_float(regime_scores.get("shock_score"), 0.0))
    regime_risk = _clamp(_safe_float(regime_scores.get("risk_norm"), _safe_float(derived_state.get("risk_score"), 0.0) / 100.0))
    cross_asset_confidence = _clamp(_safe_float(regime_scores.get("cross_asset_confidence"), 0.0))
    signals = {
        "risk_on_norm": _clamp(_weighted_profile_mean(profiles, "avg_risk_on_norm")),
        "risk_off_norm": _clamp(_weighted_profile_mean(profiles, "avg_risk_off_norm")),
        "defensive_rotation_norm": _clamp(_weighted_profile_mean(profiles, "avg_defensive_rotation_norm")),
        "trend_confirmation_norm": _clamp(_weighted_profile_mean(profiles, "avg_trend_confirmation_norm")),
        "stress_norm": _clamp(max(_weighted_profile_mean(profiles, "avg_stress_norm"), regime_shock, regime_risk)),
        "edge_norm": _clamp(_weighted_profile_mean(profiles, "avg_edge_norm")),
        "confidence_norm": _clamp(max(_weighted_profile_mean(profiles, "avg_confidence_norm"), cross_asset_confidence)),
        "buy_ratio": _clamp(_action_ratio(profiles, "BUY")),
        "sell_ratio": _clamp(_action_ratio(profiles, "SELL")),
        "hold_ratio": _clamp(_action_ratio(profiles, "HOLD")),
    }
    scores = _cycle_scores(signals)
    cycle_phase, market_regime, reasons = _classify_cycle(scores, signals)
    crosscheck = _build_crosscheck(regime_control, market_regime)
    signal_separation = abs(signals["risk_on_norm"] - signals["risk_off_norm"])
    row_coverage = _clamp(rows_sampled / max(float(min_rows), 1.0))
    agreement_bonus = 0.10 if crosscheck["agreement"] else 0.0
    confidence = _clamp(
        0.28
        + (0.25 * row_coverage)
        + (0.18 * signal_separation)
        + (0.16 * signals["trend_confirmation_norm"])
        + (0.13 * signals["confidence_norm"])
        + agreement_bonus
    )
    if rows_sampled < min_rows:
        confidence = min(confidence, 0.58)

    if rows_sampled >= min_rows:
        overall_status = "ready"
    elif rows_sampled > 0 or regime_control:
        overall_status = "thin"
    else:
        overall_status = "degraded"

    recommended_actions = ordered_unique(
        [
            "keep this engine shadow-only until replay shows cycle labels improve drawdown or false-positive filtering",
            "backfill more decision rows before trusting cycle confidence" if rows_sampled < min_rows else "",
            "compare cycle labels against paper PnL by sleeve before feeding this into promotion gates",
            "investigate disagreement between regime control and cycle extraction" if regime_control and not crosscheck["agreement"] else "",
            "refresh market posture and regime control so cycle extraction has current cross-asset context" if not regime_control or not market_posture else "",
        ]
    )

    return {
        "timestamp_utc": iso_now(),
        "schema_version": 1,
        "engine_id": "market_cycle_extraction_shadow_v1",
        "ok": overall_status in {"ready", "thin"},
        "overall_status": overall_status,
        "mode": "shadow_observation_only",
        "decision_day": day,
        "cycle_phase": cycle_phase,
        "market_regime": market_regime,
        "confidence": round(float(confidence), 6),
        "classification_reasons": reasons,
        "cycle_scores": scores,
        "aggregate_signals": {key: round(float(value), 6) for key, value in signals.items()},
        "data_depth": {
            "rows_sampled": rows_sampled,
            "min_rows_for_ready": int(min_rows),
            "profile_count": len(profiles),
            "profiles_with_rows": sum(1 for row in profiles.values() if _safe_int(row.get("rows_sampled"), 0) > 0),
            "sample_limit_per_profile": int(sample_limit),
        },
        "profile_summaries": profiles,
        "existing_regime_crosscheck": crosscheck,
        "market_posture_context": {
            "overall_status": str(market_posture.get("overall_status") or "missing"),
            "posture_state": str(market_posture.get("posture_state") or ""),
        },
        "shadow_feature_packet": {
            "feature_prefix": "market_cycle_",
            "features": {
                "market_cycle_confidence_norm": round(float(confidence), 6),
                "market_cycle_risk_on_norm": round(float(signals["risk_on_norm"]), 6),
                "market_cycle_risk_off_norm": round(float(signals["risk_off_norm"]), 6),
                "market_cycle_stress_norm": round(float(signals["stress_norm"]), 6),
                "market_cycle_trend_confirmation_norm": round(float(signals["trend_confirmation_norm"]), 6),
            },
            "consumer_status": "available_for_shadow_replay_only",
        },
        "sleeve_bias": _sleeve_bias(cycle_phase),
        "shadow_contract": {
            "shadow_only": True,
            "live_execution_allowed": False,
            "paper_execution_changes_allowed": False,
            "order_routing_allowed": False,
            "writes_runtime_env_overrides": False,
            "writes_trade_intents": False,
            "safe_consumers": [
                "replay_analysis",
                "promotion_packet_evidence",
                "paper_shadow_feature_enrichment",
                "operator_dashboard",
            ],
            "blocked_consumers": [
                "live_order_router",
                "paper_order_router",
                "position_sizer",
                "pre_trade_approval",
            ],
            "graduation_requirements": [
                "30+ days of shadow history",
                "counterfactual replay shows lower drawdown or fewer false positives",
                "promotion gate explicitly approves non-shadow consumption",
                "operator separately enables any runtime consumer",
            ],
        },
        "recommended_actions": recommended_actions,
        "source_files": {
            "regime_control": str(health / "regime_control_plane_latest.json"),
            "market_posture": str(health / "market_posture_control_latest.json"),
            "derived_state": str(health / "derived_state_latest.json"),
            "decision_channels_root": str(project_root / "governance" / "channels" / "decision"),
        },
    }


def append_history(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "timestamp_utc": payload.get("timestamp_utc"),
        "decision_day": payload.get("decision_day"),
        "overall_status": payload.get("overall_status"),
        "cycle_phase": payload.get("cycle_phase"),
        "market_regime": payload.get("market_regime"),
        "confidence": payload.get("confidence"),
        "rows_sampled": _as_dict(payload.get("data_depth")).get("rows_sampled"),
        "aggregate_signals": payload.get("aggregate_signals"),
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Extract market-cycle state as a shadow-only feature lane.")
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--sample-limit", type=int, default=1200)
    parser.add_argument("--min-rows", type=int, default=25)
    parser.add_argument("--out-file", type=Path, default=DEFAULT_OUT_PATH)
    parser.add_argument("--history-file", type=Path, default=DEFAULT_HISTORY_PATH)
    parser.add_argument("--no-history", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    project_root = Path(args.project_root).resolve()
    payload = build_payload(
        project_root,
        sample_limit=max(int(args.sample_limit), 1),
        min_rows=max(int(args.min_rows), 1),
    )
    out_path = args.out_file if args.out_file.is_absolute() else project_root / args.out_file
    write_payload(out_path, payload)
    if not args.no_history:
        history_path = args.history_file if args.history_file.is_absolute() else project_root / args.history_file
        append_history(history_path, payload)
    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        print(
            "market_cycle_extraction "
            f"status={payload.get('overall_status')} "
            f"phase={payload.get('cycle_phase')} "
            f"regime={payload.get('market_regime')} "
            f"confidence={payload.get('confidence')}"
        )
    return 0 if payload.get("ok") else 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
