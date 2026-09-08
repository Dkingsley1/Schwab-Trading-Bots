#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

if __package__ in {None, ""}:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from scripts.ops.long_runtime_common import (
        iso_now,
        load_json,
        ordered_unique,
        parse_iso_utc,
        write_payload,
    )
else:
    from .long_runtime_common import (
        PROJECT_ROOT,
        iso_now,
        load_json,
        ordered_unique,
        parse_iso_utc,
        write_payload,
    )


DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "market_pattern_observation_v1.json"
DEFAULT_PAPER_COLLECTION_PATH = (
    PROJECT_ROOT / "config" / "paper_evidence_collection_controls_v1.json"
)
DEFAULT_OUT_PATH = (
    PROJECT_ROOT / "governance" / "health" / "market_pattern_feedback_latest.json"
)
DEFAULT_PLATFORM_OUT_PATH = (
    PROJECT_ROOT
    / "governance"
    / "platform_intelligence"
    / "market_pattern_feedback_latest.json"
)
DEFAULT_HISTORY_PATH = (
    PROJECT_ROOT / "governance" / "market_patterns" / "feedback_history.jsonl"
)

TRADING_SLEEVES = (
    "equity_core",
    "intraday_aggressive",
    "day_trading",
    "swing_aggressive",
    "dividend_income",
    "dividend_capture",
    "bond_rates",
    "fx_macro",
    "crypto_spot",
    "crypto_futures",
    "schwab_futures",
    "volatility",
    "pairs_correlation",
    "stat_arb_market_neutral",
    "earnings_event",
    "commodity_inflation",
    "international_macro",
    "market_making_liquidity",
    "short_bias_hedge",
    "single_name_options_event",
    "rates_credit_macro",
    "cash_rotation_tactical",
    "conservative",
    "futures_index_intraday",
    "futures_rates_curve",
    "futures_commodity_macro",
    "crypto_futures_basis",
    "futures_event_reaction",
    "options_on_futures_aggressive",
)

FALLBACK_SOURCE_ARTIFACTS = {
    "regime_control": "governance/health/regime_control_plane_latest.json",
    "market_cycle": "governance/health/market_cycle_state_latest.json",
    "market_move": "governance/health/market_move_explainer_latest.json",
    "decision_context_mesh": "governance/health/decision_context_mesh_latest.json",
    "market_micro": "governance/health/market_micro_sync_latest.json",
    "official_macro": "governance/health/official_macro_context_sync_latest.json",
    "cross_asset_breadth": "governance/health/cross_asset_breadth_context_latest.json",
    "tape_liquidity": "governance/health/tape_liquidity_context_latest.json",
    "options_flow": "governance/health/options_flow_context_sync_latest.json",
    "fx_market": "governance/health/fx_market_context_sync_latest.json",
    "crypto_market": "governance/health/crypto_market_context_sync_latest.json",
    "crypto_correlation": "governance/health/market_crypto_correlation_sync_latest.json",
    "sleeve_profitability": "governance/health/sleeve_profitability_dashboard_latest.json",
    "paper_profitability": "governance/health/paper_profitability_control_latest.json",
    "independent_fills": "governance/health/independent_fill_evidence_acquisition_latest.json",
    "bot_profitability": "governance/health/bot_profitability_scalability_latest.json",
}


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except Exception:
        return float(default)
    return parsed if math.isfinite(parsed) else float(default)


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except Exception:
        return int(default)


def _clamp(value: float, minimum: float = 0.0, maximum: float = 1.0) -> float:
    return max(float(minimum), min(float(value), float(maximum)))


def _status(payload: Mapping[str, Any]) -> str:
    return (
        str(
            payload.get("overall_status")
            or payload.get("status")
            or payload.get("state")
            or ("ready" if payload.get("ok") is True else "")
        )
        .strip()
        .lower()
    )


def _source_age_seconds(payload: Mapping[str, Any], path: Path) -> float | None:
    parsed = parse_iso_utc(
        payload.get("timestamp_utc")
        or payload.get("updated_at_utc")
        or payload.get("generated_at_utc")
    )
    if parsed is None:
        try:
            parsed = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
        except OSError:
            return None
    return max((datetime.now(timezone.utc) - parsed).total_seconds(), 0.0)


def _resolve(project_root: Path, raw: Any) -> Path:
    path = Path(str(raw or "")).expanduser()
    return path if path.is_absolute() else project_root / path


def _load_sources(
    project_root: Path,
    config: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    source_artifacts = (
        _as_dict(config.get("source_artifacts")) or FALLBACK_SOURCE_ARTIFACTS
    )
    sources: dict[str, dict[str, Any]] = {}
    for source_id, raw_path in sorted(source_artifacts.items()):
        path = _resolve(project_root, raw_path)
        payload = load_json(path)
        age_seconds = _source_age_seconds(payload, path) if payload else None
        status = _status(payload) if payload else "missing"
        sources[str(source_id)] = {
            "source_id": str(source_id),
            "path": str(path),
            "present": bool(payload),
            "status": status,
            "ok": bool(payload.get("ok", False)) if payload else False,
            "age_seconds": round(age_seconds, 3) if age_seconds is not None else None,
            "payload": payload,
        }
    return sources


def _source_snapshot(sources: Mapping[str, dict[str, Any]]) -> dict[str, Any]:
    rows = []
    ready_count = 0
    context_count = 0
    for source_id, row in sorted(sources.items()):
        present = bool(row.get("present", False))
        status = str(row.get("status") or "missing")
        context_usable = present and status not in {"blocked", "critical", "missing"}
        ready = present and status in {
            "ready",
            "ok",
            "collecting",
            "waiting_for_source",
        }
        if ready:
            ready_count += 1
        if context_usable:
            context_count += 1
        rows.append(
            {
                "source_id": source_id,
                "present": present,
                "status": status,
                "context_usable": context_usable,
                "ready": ready,
                "age_seconds": row.get("age_seconds"),
                "path": row.get("path"),
            }
        )
    return {
        "source_count": len(rows),
        "ready_source_count": ready_count,
        "context_usable_source_count": context_count,
        "rows": rows,
    }


def _pattern(
    pattern_id: str,
    *,
    label: str,
    strength: float,
    direction: str,
    evidence: Mapping[str, Any],
    confidence: float | None = None,
) -> dict[str, Any]:
    return {
        "pattern_id": pattern_id,
        "label": label,
        "strength": round(_clamp(strength), 6),
        "confidence": round(
            _clamp(confidence if confidence is not None else strength), 6
        ),
        "direction": direction,
        "evidence": dict(evidence),
        "profitability_claim": False,
    }


def _detect_patterns(sources: Mapping[str, dict[str, Any]]) -> list[dict[str, Any]]:
    regime = _as_dict(_as_dict(sources.get("regime_control")).get("payload"))
    cycle = _as_dict(_as_dict(sources.get("market_cycle")).get("payload"))
    move = _as_dict(_as_dict(sources.get("market_move")).get("payload"))
    aggregate = _as_dict(cycle.get("aggregate_signals"))
    source_snapshot = _source_snapshot(sources)

    regime_state = str(regime.get("regime_state") or "unknown").strip().lower()
    stance_label = str(regime.get("stance_label") or "unknown").strip().lower()
    stance_score = _safe_float(regime.get("stance_score"), 0.0)
    cycle_phase = str(cycle.get("cycle_phase") or "unknown").strip().lower()
    market_regime = (
        str(cycle.get("market_regime") or regime_state or "unknown").strip().lower()
    )
    cycle_confidence = _safe_float(cycle.get("confidence"), 0.0)
    stress = _safe_float(aggregate.get("stress_norm"), 0.0)
    risk_off = _safe_float(aggregate.get("risk_off_norm"), 0.0)
    risk_on = _safe_float(aggregate.get("risk_on_norm"), 0.0)
    defensive = _safe_float(aggregate.get("defensive_rotation_norm"), 0.0)
    trend = _safe_float(aggregate.get("trend_confirmation_norm"), 0.0)
    hold_ratio = _safe_float(aggregate.get("hold_ratio"), 0.0)
    buy_ratio = _safe_float(aggregate.get("buy_ratio"), 0.0)
    sell_ratio = _safe_float(aggregate.get("sell_ratio"), 0.0)

    patterns: list[dict[str, Any]] = []
    if "high_vol" in market_regime or "high_vol" in cycle_phase or stress >= 0.65:
        patterns.append(
            _pattern(
                "defensive_high_vol_chop",
                label="Defensive high-volatility chop",
                strength=max(stress, cycle_confidence, defensive, risk_off),
                direction="defensive_or_mean_reversion",
                confidence=max(cycle_confidence, 0.5),
                evidence={
                    "cycle_phase": cycle_phase,
                    "market_regime": market_regime,
                    "stress_norm": round(stress, 6),
                    "defensive_rotation_norm": round(defensive, 6),
                    "risk_off_norm": round(risk_off, 6),
                },
            )
        )
    if (
        "transition" in regime_state
        or "transition" in cycle_phase
        or regime_state
        in {
            "mixed_transition",
            "fragile_transition",
            "rangebound_transition",
        }
    ):
        patterns.append(
            _pattern(
                "mixed_transition",
                label="Mixed or transitioning regime",
                strength=max(abs(stance_score), cycle_confidence * 0.8, 0.45),
                direction="segment_by_regime",
                confidence=max(cycle_confidence, 0.45),
                evidence={
                    "regime_state": regime_state,
                    "stance_label": stance_label,
                    "stance_score": round(stance_score, 6),
                    "cycle_phase": cycle_phase,
                },
            )
        )
    if hold_ratio >= 0.75:
        patterns.append(
            _pattern(
                "system_hold_consensus",
                label="System-wide hold consensus",
                strength=hold_ratio,
                direction="hesitation_or_no_edge",
                confidence=max(cycle_confidence, 0.45),
                evidence={
                    "hold_ratio": round(hold_ratio, 6),
                    "buy_ratio": round(buy_ratio, 6),
                    "sell_ratio": round(sell_ratio, 6),
                },
            )
        )
    if risk_on >= 0.55 and trend >= 0.45 and risk_on > risk_off:
        patterns.append(
            _pattern(
                "risk_on_trend",
                label="Risk-on trend attempt",
                strength=max(risk_on, trend),
                direction="trend_following",
                confidence=max(cycle_confidence, 0.45),
                evidence={
                    "risk_on_norm": round(risk_on, 6),
                    "trend_confirmation_norm": round(trend, 6),
                    "risk_off_norm": round(risk_off, 6),
                },
            )
        )
    if risk_off >= 0.35 or defensive >= 0.35 or stance_score < -0.25:
        patterns.append(
            _pattern(
                "risk_off_pressure",
                label="Risk-off or defensive pressure",
                strength=max(risk_off, defensive, abs(min(stance_score, 0.0))),
                direction="defensive",
                confidence=max(cycle_confidence, 0.45),
                evidence={
                    "risk_off_norm": round(risk_off, 6),
                    "defensive_rotation_norm": round(defensive, 6),
                    "stance_score": round(stance_score, 6),
                },
            )
        )

    if source_snapshot["context_usable_source_count"] >= 4:
        patterns.append(
            _pattern(
                "source_context_available",
                label="Cross-source market context available",
                strength=source_snapshot["context_usable_source_count"]
                / max(source_snapshot["source_count"], 1),
                direction="feature_backfill",
                confidence=source_snapshot["ready_source_count"]
                / max(source_snapshot["source_count"], 1),
                evidence={
                    "ready_source_count": source_snapshot["ready_source_count"],
                    "context_usable_source_count": source_snapshot[
                        "context_usable_source_count"
                    ],
                    "source_count": source_snapshot["source_count"],
                },
            )
        )

    move_drivers = [
        row for row in _as_list(move.get("ranked_drivers")) if isinstance(row, dict)
    ]
    if move_drivers:
        top = move_drivers[0]
        patterns.append(
            _pattern(
                "symbol_driver_detected",
                label=str(move.get("primary_readout") or "Symbol driver detected"),
                strength=_safe_float(top.get("strength"), 0.5),
                direction=str(top.get("direction") or "symbol_specific"),
                confidence=_safe_float(move.get("primary_confidence"), 0.5),
                evidence={
                    "symbol": str(move.get("symbol") or ""),
                    "driver": str(top.get("driver") or ""),
                    "source": "market_move_explainer",
                },
            )
        )
    elif _as_list(move.get("unknowns")):
        patterns.append(
            _pattern(
                "symbol_specific_evidence_gap",
                label="Symbol-specific driver evidence is thin",
                strength=0.65,
                direction="collect_symbol_attribution",
                confidence=_safe_float(move.get("primary_confidence"), 0.45),
                evidence={
                    "symbol": str(move.get("symbol") or ""),
                    "unknowns": [str(item) for item in _as_list(move.get("unknowns"))],
                },
            )
        )

    return sorted(
        patterns,
        key=lambda row: (
            -float(row.get("strength", 0.0)),
            -float(row.get("confidence", 0.0)),
            str(row.get("pattern_id") or ""),
        ),
    )


def _expand_sleeves(items: list[Any]) -> set[str]:
    out: set[str] = set()
    for raw in items:
        item = str(raw or "").strip()
        if not item:
            continue
        if item == "all_trading_sleeves":
            out.update(TRADING_SLEEVES)
        elif item == "macro":
            out.update(
                {
                    "bond_rates",
                    "fx_macro",
                    "rates_credit_macro",
                    "commodity_inflation",
                    "international_macro",
                    "futures_rates_curve",
                    "futures_commodity_macro",
                }
            )
        else:
            out.add(item)
    return out


def _paper_collection_profile(
    paper_controls: Mapping[str, Any],
    sleeve: str,
) -> dict[str, Any]:
    defaults = _as_dict(paper_controls.get("defaults"))
    profiles = _as_dict(paper_controls.get("profiles"))
    raw = profiles.get(sleeve)
    if not isinstance(raw, dict):
        raw = profiles.get("default")
    merged = dict(defaults)
    if isinstance(raw, dict):
        merged.update(raw)
    return {
        "configured": bool(paper_controls),
        "enabled": bool(merged.get("enabled", True)) if paper_controls else False,
        "minimum_model_score_edge_over_threshold": _safe_float(
            merged.get("minimum_model_score_edge_over_threshold"), 0.02
        ),
        "max_entries_per_symbol_day": _safe_int(
            merged.get("max_entries_per_symbol_day"), 0
        ),
        "new_entry_cooldown_seconds": _safe_float(
            merged.get("new_entry_cooldown_seconds"), 0.0
        ),
        "paper_only": bool(paper_controls.get("paper_only", False)),
        "live_execution_allowed": bool(
            paper_controls.get("live_execution_allowed", True)
        ),
    }


def _sleeve_feedback(
    patterns: list[dict[str, Any]],
    config: Mapping[str, Any],
    paper_controls: Mapping[str, Any],
) -> list[dict[str, Any]]:
    responses = _as_dict(config.get("pattern_response"))
    scores = {
        sleeve: {
            "boost": 0.0,
            "caution": 0.0,
            "context": 0.0,
            "patterns": [],
            "context_patterns": [],
            "focus": [],
        }
        for sleeve in TRADING_SLEEVES
    }
    for pattern in patterns:
        pattern_id = str(pattern.get("pattern_id") or "")
        response = _as_dict(responses.get(pattern_id))
        strength = _safe_float(pattern.get("strength"), 0.0)
        boost = _expand_sleeves(_as_list(response.get("boost")))
        downshift = _expand_sleeves(_as_list(response.get("downshift")))
        context = _expand_sleeves(
            _as_list(response.get("context"))
            or _as_list(response.get("annotate"))
            or _as_list(response.get("observe"))
        )
        focus = [
            str(item)
            for item in _as_list(response.get("collection_focus"))
            if str(item)
        ]
        for sleeve in boost:
            if sleeve in scores:
                scores[sleeve]["boost"] += strength
                scores[sleeve]["patterns"].append(pattern_id)
                scores[sleeve]["focus"].extend(focus)
        for sleeve in downshift:
            if sleeve in scores:
                scores[sleeve]["caution"] += strength
                scores[sleeve]["patterns"].append(pattern_id)
                scores[sleeve]["focus"].extend(focus)
        for sleeve in context:
            if sleeve in scores:
                scores[sleeve]["context"] += strength
                scores[sleeve]["context_patterns"].append(pattern_id)
                scores[sleeve]["focus"].extend(focus)

    rows: list[dict[str, Any]] = []
    for sleeve, row in scores.items():
        boost = float(row["boost"])
        caution = float(row["caution"])
        context = float(row["context"])
        if boost <= 0.0 and caution <= 0.0 and context <= 0.0:
            continue
        if caution >= 0.65 and caution > boost:
            posture = "downshift_or_context_first"
        elif boost >= 0.65 and boost >= caution:
            posture = "prioritize_bounded_paper_sampling"
        elif boost > caution:
            posture = "normal_paper_collection"
        elif caution <= 0.0 and context > 0.0:
            posture = "context_only"
        else:
            posture = "cautious_paper_collection"
        all_patterns = ordered_unique(
            [str(item) for item in row["patterns"]]
            + [str(item) for item in row["context_patterns"]]
        )
        rows.append(
            {
                "sleeve": sleeve,
                "paper_sampling_posture": posture,
                "boost_score": round(_clamp(boost), 6),
                "caution_score": round(_clamp(caution), 6),
                "context_score": round(_clamp(context), 6),
                "pattern_ids": all_patterns,
                "context_pattern_ids": ordered_unique(
                    [str(item) for item in row["context_patterns"]]
                ),
                "collection_focus": ordered_unique(
                    [str(item) for item in row["focus"]]
                ),
                "paper_collection_profile": _paper_collection_profile(
                    paper_controls, sleeve
                ),
                "capital_allocation_allowed": False,
                "live_execution_allowed": False,
                "profitability_claim_allowed": False,
            }
        )
    return sorted(
        rows,
        key=lambda item: (
            item["paper_sampling_posture"] != "prioritize_bounded_paper_sampling",
            -float(item.get("boost_score", 0.0)),
            float(item.get("caution_score", 0.0)),
            item["paper_sampling_posture"] != "context_only",
            -float(item.get("context_score", 0.0)),
            str(item.get("sleeve") or ""),
        ),
    )


def _profitability_evidence_gaps(
    sources: Mapping[str, dict[str, Any]],
) -> dict[str, Any]:
    paper = _as_dict(_as_dict(sources.get("paper_profitability")).get("payload"))
    contract = _as_dict(paper.get("paper_debt_recovery_contract"))
    proof = _as_dict(contract.get("candidate_proof"))
    independent = _as_dict(_as_dict(sources.get("independent_fills")).get("payload"))
    bot = _as_dict(_as_dict(sources.get("bot_profitability")).get("payload"))
    recovery = _as_dict(
        _as_dict(bot.get("profitability_diagnosis")).get("paper_balance_recovery_plan")
    )
    independent_count = _safe_int(
        independent.get("candidate_eligible_ledger_records"), 0
    )
    return {
        "paper_recovery_state": str(contract.get("state") or ""),
        "remaining_debt_amount": _safe_float(
            contract.get("remaining_debt_amount"), 0.0
        ),
        "candidate_post_cost_sample_count": _safe_int(
            proof.get("sample_count"),
            _safe_int(recovery.get("sample_count"), 0),
        ),
        "candidate_post_cost_minimum_samples": _safe_int(
            proof.get("minimum_samples"),
            _safe_int(recovery.get("min_post_cost_samples"), 30),
        ),
        "candidate_observed_days": _safe_int(
            proof.get("observed_days"),
            _safe_int(recovery.get("observed_days"), 0),
        ),
        "candidate_minimum_observed_days": _safe_int(
            proof.get("minimum_observed_days"),
            _safe_int(recovery.get("min_observed_days"), 3),
        ),
        "candidate_independent_fill_records": independent_count,
        "independent_fill_materialization_ready": bool(
            _as_dict(independent.get("trade_log_materialization")).get("ok", False)
        ),
        "positive_post_cost_lcb": bool(
            proof.get("positive_post_cost_lower_confidence_bound_95", False)
        ),
        "promotion_blockers": [
            str(item)
            for item in _as_list(contract.get("promotion_blockers"))
            if str(item)
        ],
    }


def _recommended_actions(
    patterns: list[dict[str, Any]],
    sleeve_feedback: list[dict[str, Any]],
    evidence_gaps: Mapping[str, Any],
) -> list[str]:
    pattern_ids = {str(row.get("pattern_id") or "") for row in patterns}
    prioritized = [
        str(row.get("sleeve"))
        for row in sleeve_feedback
        if row.get("paper_sampling_posture") == "prioritize_bounded_paper_sampling"
    ][:6]
    return ordered_unique(
        [
            (
                "prioritize bounded paper samples for " + ",".join(prioritized)
                if prioritized
                else ""
            ),
            (
                "record symbol-level driver features with each paper intent"
                if "symbol_specific_evidence_gap" in pattern_ids
                else ""
            ),
            (
                "segment every new paper sample by regime, cycle phase, sleeve, symbol, and post-cost outcome"
                if patterns
                else ""
            ),
            (
                "keep live execution disabled until profitability and production pillars clear"
                if patterns
                else ""
            ),
            (
                "continue independent fill collection now that materialization is ready"
                if bool(
                    evidence_gaps.get("independent_fill_materialization_ready", False)
                )
                else "repair independent-fill materialization before relying on fill evidence"
            ),
        ]
    )


def build_payload(
    project_root: Path = PROJECT_ROOT,
    *,
    config_path: Path = DEFAULT_CONFIG_PATH,
    paper_collection_path: Path = DEFAULT_PAPER_COLLECTION_PATH,
) -> dict[str, Any]:
    project_root = project_root.resolve()
    config_path = (
        config_path if config_path.is_absolute() else project_root / config_path
    )
    paper_collection_path = (
        paper_collection_path
        if paper_collection_path.is_absolute()
        else project_root / paper_collection_path
    )
    config = load_json(config_path)
    paper_controls = load_json(paper_collection_path)
    sources = _load_sources(project_root, config)
    source_snapshot = _source_snapshot(sources)
    patterns = _detect_patterns(sources)
    sleeve_feedback = _sleeve_feedback(patterns, config, paper_controls)
    evidence_gaps = _profitability_evidence_gaps(sources)
    ready_patterns = [
        row for row in patterns if _safe_float(row.get("strength"), 0.0) >= 0.6
    ]
    if ready_patterns and source_snapshot["context_usable_source_count"] >= 4:
        status = "ready"
    elif patterns or source_snapshot["context_usable_source_count"] > 0:
        status = "thin"
    else:
        status = "degraded"
    dimensions = [
        dict(row)
        for row in _as_list(config.get("observable_dimensions"))
        if isinstance(row, dict)
    ]
    return {
        "timestamp_utc": iso_now(),
        "schema_version": 1,
        "ok": status in {"ready", "thin"},
        "overall_status": status,
        "policy_id": str(config.get("policy_id") or "market_pattern_observation_v1"),
        "paper_only": True,
        "live_execution_allowed": False,
        "profitability_claim_allowed": False,
        "dominant_patterns": ready_patterns[:5],
        "patterns": patterns,
        "pattern_count": len(patterns),
        "observable_market_dimensions": dimensions,
        "observable_dimension_count": len(dimensions),
        "source_snapshot": {
            key: value for key, value in source_snapshot.items() if key != "rows"
        },
        "source_rows": source_snapshot["rows"],
        "sleeve_feedback": sleeve_feedback,
        "sleeve_feedback_count": len(sleeve_feedback),
        "profitability_evidence_gaps": evidence_gaps,
        "platform_feedback_contract": {
            "can_route_paper_collection_priority": True,
            "can_write_context_artifact": True,
            "can_change_live_execution": False,
            "can_claim_profitability": False,
            "can_force_trade": False,
            "can_increase_size_after_loss": False,
            "can_relax_promotion_gates": False,
        },
        "recommended_actions": _recommended_actions(
            patterns, sleeve_feedback, evidence_gaps
        ),
        "source_files": {
            "config": str(config_path),
            "paper_evidence_collection_controls": str(paper_collection_path),
            "health_output": str(DEFAULT_OUT_PATH),
            "platform_output": str(DEFAULT_PLATFORM_OUT_PATH),
        },
    }


def append_history(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "timestamp_utc": payload.get("timestamp_utc"),
        "overall_status": payload.get("overall_status"),
        "pattern_count": payload.get("pattern_count"),
        "dominant_pattern_ids": [
            str(item.get("pattern_id") or "")
            for item in _as_list(payload.get("dominant_patterns"))
            if isinstance(item, dict)
        ],
        "top_sleeves": [
            str(item.get("sleeve") or "")
            for item in _as_list(payload.get("sleeve_feedback"))[:8]
            if isinstance(item, dict)
        ],
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build market-pattern feedback for platform and sleeve paper collection."
    )
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--config-file", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument(
        "--paper-collection-file",
        type=Path,
        default=DEFAULT_PAPER_COLLECTION_PATH,
    )
    parser.add_argument("--out-file", type=Path, default=DEFAULT_OUT_PATH)
    parser.add_argument(
        "--platform-out-file", type=Path, default=DEFAULT_PLATFORM_OUT_PATH
    )
    parser.add_argument("--history-file", type=Path, default=DEFAULT_HISTORY_PATH)
    parser.add_argument("--no-history", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    project_root = Path(args.project_root).expanduser().resolve()
    payload = build_payload(
        project_root,
        config_path=args.config_file,
        paper_collection_path=args.paper_collection_file,
    )
    out_path = (
        args.out_file if args.out_file.is_absolute() else project_root / args.out_file
    )
    platform_out = (
        args.platform_out_file
        if args.platform_out_file.is_absolute()
        else project_root / args.platform_out_file
    )
    write_payload(out_path, payload)
    write_payload(platform_out, payload)
    if not args.no_history:
        history = (
            args.history_file
            if args.history_file.is_absolute()
            else project_root / args.history_file
        )
        append_history(history, payload)
    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        top = ",".join(
            str(row.get("pattern_id") or "")
            for row in _as_list(payload.get("dominant_patterns"))[:4]
            if isinstance(row, dict)
        )
        sleeves = ",".join(
            str(row.get("sleeve") or "")
            for row in _as_list(payload.get("sleeve_feedback"))[:6]
            if isinstance(row, dict)
        )
        print(
            "market_pattern_feedback "
            f"status={payload.get('overall_status')} "
            f"patterns={payload.get('pattern_count')} "
            f"dominant={top or 'none'} "
            f"top_sleeves={sleeves or 'none'}"
        )
    return 0 if payload.get("ok") else 2


if __name__ == "__main__":
    raise SystemExit(main())
