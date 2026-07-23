#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import date
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from scripts.ops.long_runtime_common import iso_now, load_json, ordered_unique, payload_age_minutes, us_equity_market_holiday, write_payload
else:
    from .long_runtime_common import PROJECT_ROOT, iso_now, load_json, ordered_unique, payload_age_minutes, us_equity_market_holiday, write_payload


DEFAULT_OUT_PATH = PROJECT_ROOT / "governance" / "health" / "trading_desk_upgrade_control_latest.json"
DEFAULT_PLATFORM_OS_OUT_PATH = PROJECT_ROOT / "governance" / "platform_os" / "trading_desk_upgrade_control_latest.json"

SOURCE_FILES = {
    "paper_execution_truth_layer": "governance/health/paper_execution_truth_layer_latest.json",
    "promotion_quality": "governance/health/promotion_quality_gate_latest.json",
    "live_readiness": "governance/health/live_readiness_smoke_latest.json",
    "paper_live_data_standard": "governance/health/paper_live_data_standard_latest.json",
    "live_canary": "governance/health/live_canary_control_latest.json",
    "paper_performance": "governance/health/paper_performance_latest.json",
    "account_position": "governance/health/account_position_study_latest.json",
    "covered_call_roll_watch": "governance/health/covered_call_roll_watch_latest.json",
    "execution_lab": "governance/health/execution_lab_latest.json",
    "counterfactual_replay": "governance/health/counterfactual_replay_latest.json",
    "paper_replay_drill": "governance/health/paper_replay_drill_latest.json",
    "strategy_attribution": "governance/health/strategy_attribution_latest.json",
    "operator_cockpit": "governance/health/operator_cockpit_latest.json",
    "a_plus_operating_packet": "governance/health/a_plus_operating_packet_latest.json",
    "ingestion_storage": "governance/health/ingestion_storage_control_latest.json",
    "storage_retention": "governance/health/storage_retention_unison_latest.json",
    "bot_quality_autopilot": "governance/health/bot_quality_autopilot_latest.json",
    "infrastructure_autofix": "governance/health/infrastructure_autofix_bot_latest.json",
    "system_self_model": "governance/health/system_self_model_latest.json",
    "whole_system_intelligence": "governance/health/whole_system_intelligence_latest.json",
}

LANE_LABELS = {
    "execution_truth_recorder": "Execution Truth Recorder",
    "paper_live_acceptance_harness": "Paper-to-Live Acceptance Harness",
    "supervised_live_canary_mode": "Supervised Live Canary Mode",
    "advisory_capital_allocator": "Advisory Paper Capital Allocator",
    "account_position_intelligence": "Account + Position Intelligence",
    "market_condition_replay": "Market-Condition Replay",
    "decision_quality_attribution": "Decision Quality Attribution",
    "operator_cockpit": "Operator Cockpit",
    "storage_ingestion_discipline": "Storage + Ingestion Discipline",
    "autonomous_improvement_loop": "Autonomous Improvement Loop",
}


def _as_dict(raw: Any) -> dict[str, Any]:
    return raw if isinstance(raw, dict) else {}


def _as_list(raw: Any) -> list[Any]:
    return raw if isinstance(raw, list) else []


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


def _clamp(value: float, low: float = 0.0, high: float = 100.0) -> float:
    return max(float(low), min(float(value), float(high)))


def _status(payload: dict[str, Any]) -> str:
    if not payload:
        return "missing"
    raw = str(payload.get("overall_status") or payload.get("status") or payload.get("state") or "").strip().lower()
    if raw:
        return raw
    if bool(payload.get("ok", False)):
        return "ready"
    return "unknown"


def _artifact_ok(payload: dict[str, Any], *, allow_watch: bool = False) -> bool:
    if not payload:
        return False
    if "ok" in payload:
        return bool(payload.get("ok", False))
    status = _status(payload)
    ready = {"ready", "ok", "running", "active", "clear_ready", "advisory"}
    if allow_watch:
        ready.add("watch")
        ready.add("degraded")
    return status in ready


def _grade(score: float) -> str:
    score = _clamp(score)
    if score >= 97.0:
        return "A+"
    if score >= 93.0:
        return "A"
    if score >= 90.0:
        return "A-"
    if score >= 87.0:
        return "B+"
    if score >= 83.0:
        return "B"
    if score >= 75.0:
        return "C"
    if score >= 65.0:
        return "D"
    return "F"


def _non_trading_day(day_text: Any) -> bool:
    text = str(day_text or "").strip()
    if len(text) != 8 or not text.isdigit():
        return False
    try:
        day = date(int(text[:4]), int(text[4:6]), int(text[6:8]))
    except Exception:
        return False
    return bool(day.weekday() >= 5 or us_equity_market_holiday(day))


def _source_path(project_root: Path, name: str) -> Path:
    return project_root / SOURCE_FILES[name]


def _source_evidence(project_root: Path, sources: dict[str, dict[str, Any]], name: str) -> dict[str, Any]:
    path = _source_path(project_root, name)
    payload = sources.get(name, {})
    age = payload_age_minutes(payload, path)
    return {
        "name": name,
        "path": str(path),
        "present": bool(payload),
        "status": _status(payload),
        "age_minutes": round(float(age), 3) if age is not None else None,
    }


def _load_sources(project_root: Path) -> dict[str, dict[str, Any]]:
    return {name: load_json(_source_path(project_root, name)) for name in SOURCE_FILES}


def _lane(
    lane_id: str,
    *,
    score: float,
    blockers: list[str] | None = None,
    warnings: list[str] | None = None,
    evidence: dict[str, Any] | None = None,
    source_names: list[str] | None = None,
    sources: dict[str, dict[str, Any]],
    project_root: Path,
    recommended_actions: list[str] | None = None,
) -> dict[str, Any]:
    blockers = ordered_unique(blockers or [])
    warnings = ordered_unique(warnings or [])
    score = _clamp(score)
    if blockers:
        score = min(score, 74.0)
    status = "ready" if score >= 90.0 and not blockers else ("watch" if not blockers else "blocked")
    return {
        "id": lane_id,
        "label": LANE_LABELS[lane_id],
        "ok": not blockers,
        "status": status,
        "score": round(score, 6),
        "grade": _grade(score),
        "a_plus": bool(score >= 97.0 and not blockers and not warnings),
        "blockers": blockers,
        "warnings": warnings,
        "evidence": evidence or {},
        "sources": [_source_evidence(project_root, sources, name) for name in (source_names or [])],
        "recommended_actions": ordered_unique(recommended_actions or []),
    }


def _execution_truth_lane(project_root: Path, sources: dict[str, dict[str, Any]]) -> dict[str, Any]:
    truth = sources["paper_execution_truth_layer"]
    gates = _as_dict(truth.get("gates"))
    live_transition = _as_dict(gates.get("live_execution_transition_parity"))
    throttle = _as_dict(gates.get("auto_throttle_overtrading"))
    blockers = list(map(str, _as_list(truth.get("failed_checks"))))
    warnings = list(map(str, _as_list(truth.get("warnings"))))
    if not bool(truth.get("ok", False)):
        blockers.append("paper_execution_truth_layer_not_ok")
    if live_transition and _status(live_transition) not in {"ready", "ok"}:
        blockers.append("live_transition_parity_not_ready")
    if throttle and _status(throttle) not in {"ready", "ok"}:
        warnings.append("overtrading_throttle_not_ready")
    score = _safe_float(truth.get("score"), 0.0)
    return _lane(
        "execution_truth_recorder",
        score=score,
        blockers=blockers,
        warnings=warnings,
        evidence={
            "truth_layer_status": truth.get("overall_status"),
            "truth_layer_grade": truth.get("grade"),
            "a_plus_ready": bool(truth.get("a_plus_ready", False)),
            "raw_metric_score": truth.get("raw_metric_score"),
            "live_transition_status": live_transition.get("status"),
            "throttle_actions": _as_list(throttle.get("throttle_actions")),
        },
        source_names=["paper_execution_truth_layer"],
        sources=sources,
        project_root=project_root,
        recommended_actions=["refresh paper execution truth layer"] if not truth else [],
    )


def _acceptance_lane(project_root: Path, sources: dict[str, dict[str, Any]]) -> dict[str, Any]:
    truth = sources["paper_execution_truth_layer"]
    promotion = sources["promotion_quality"]
    live = sources["live_readiness"]
    standard = sources["paper_live_data_standard"]
    blockers: list[str] = []
    warnings: list[str] = []
    score = 100.0
    if not bool(truth.get("a_plus_ready", False)):
        blockers.append("truth_layer_not_a_plus_ready")
        score -= 30.0
    if not _artifact_ok(promotion):
        blockers.append("promotion_quality_not_ok")
        score -= 25.0
    if not _artifact_ok(live):
        blockers.append("live_readiness_not_ok")
        score -= 25.0
    if _as_list(live.get("hard_blocks")):
        blockers.append("live_readiness_hard_blocks_present")
        score -= 15.0
    if not standard:
        warnings.append("paper_live_data_standard_missing")
        score -= 5.0
    elif not _artifact_ok(standard, allow_watch=True):
        warnings.append("paper_live_data_standard_not_ready")
        score -= 8.0
    return _lane(
        "paper_live_acceptance_harness",
        score=score,
        blockers=blockers,
        warnings=warnings,
        evidence={
            "acceptance_ready": bool(not blockers),
            "truth_layer_a_plus": bool(truth.get("a_plus_ready", False)),
            "promotion_quality_ok": bool(promotion.get("ok", False)),
            "live_readiness_status": live.get("overall_status"),
            "live_submit_path_enabled": bool(live.get("submit_path_enabled", False)),
            "paper_live_data_standard_present": bool(standard),
        },
        source_names=["paper_execution_truth_layer", "promotion_quality", "live_readiness", "paper_live_data_standard"],
        sources=sources,
        project_root=project_root,
        recommended_actions=[
            "keep live submit disabled until acceptance remains clean across a rolling window",
            "refresh promotion quality after truth-layer changes",
        ],
    )


def _canary_lane(project_root: Path, sources: dict[str, dict[str, Any]]) -> dict[str, Any]:
    canary = sources["live_canary"]
    blockers = list(map(str, _as_list(canary.get("blocking_reasons"))))
    hard_blockers = [
        reason
        for reason in blockers
        if reason in {"broker_not_ready", "session_not_ready", "storage_not_ready", "storage_not_external"}
    ]
    warnings = [reason for reason in blockers if reason not in hard_blockers]
    score = _safe_float(canary.get("preclearance_score"), 0.0 if not canary else 75.0)
    if bool(canary.get("supervised_canary_ready", False)):
        score = 100.0
        warnings = []
    elif bool(canary.get("preapproved_supervised_ready", False)):
        score = max(score, 92.0)
    elif bool(canary.get("staged_preclearance_ready", False)):
        score = max(score, 85.0)
    return _lane(
        "supervised_live_canary_mode",
        score=score,
        blockers=hard_blockers,
        warnings=warnings,
        evidence={
            "recommended_mode": canary.get("recommended_mode"),
            "supervised_canary_ready": bool(canary.get("supervised_canary_ready", False)),
            "staged_preclearance_ready": bool(canary.get("staged_preclearance_ready", False)),
            "preapproved_supervised_ready": bool(canary.get("preapproved_supervised_ready", False)),
            "target_canary_weight": canary.get("target_canary_weight"),
            "applied_canary_weight": canary.get("applied_canary_weight"),
            "canary_weight_ok": bool(canary.get("canary_weight_ok", False)),
        },
        source_names=["live_canary", "live_readiness"],
        sources=sources,
        project_root=project_root,
        recommended_actions=["run canary only as supervised, tiny-weight, explicitly approved mode"],
    )


def _capital_plan(truth: dict[str, Any]) -> dict[str, Any]:
    rows = [row for row in _as_list(truth.get("sleeve_scorecards")) if isinstance(row, dict)]
    candidates: list[dict[str, Any]] = []
    for row in rows:
        profile = str(row.get("profile") or "default").strip().lower() or "default"
        score = _safe_float(row.get("execution_realism_score"), 0.0)
        pnl = _safe_float(row.get("ending_net_pnl_total"), 0.0)
        change = _safe_float(row.get("change_vs_previous_day"), 0.0)
        execs = _safe_int(row.get("executions"), 0)
        status = _status(row)
        quality_value = max(score - 60.0, 0.0)
        pnl_bonus = max(min(math.tanh(max(pnl, 0.0) / 250.0) * 25.0, 25.0), 0.0)
        change_bonus = max(min(math.tanh(max(change, 0.0) / 125.0) * 12.0, 12.0), 0.0)
        negative_drag = 20.0 if pnl < 0.0 else 0.0
        value = max(quality_value + pnl_bonus + change_bonus - negative_drag, 0.0)
        if status == "watch":
            value *= 0.7
        if score < 65.0:
            value = 0.0
        candidates.append(
            {
                "profile": profile,
                "advisory_score": round(_clamp(value), 6),
                "execution_realism_score": round(score, 6),
                "ending_net_pnl_total": round(pnl, 6),
                "change_vs_previous_day": round(change, 6),
                "executions": execs,
                "status": status or "unknown",
                "paper_only": True,
            }
        )
    total = sum(_safe_float(row.get("advisory_score"), 0.0) for row in candidates)
    max_weight = 0.25
    allocated = 0.0
    ranked: list[dict[str, Any]] = []
    for row in sorted(candidates, key=lambda item: _safe_float(item.get("advisory_score"), 0.0), reverse=True):
        raw_weight = (_safe_float(row.get("advisory_score"), 0.0) / total) if total > 0.0 else 0.0
        weight = min(raw_weight, max_weight)
        allocated += weight
        ranked.append({**row, "advisory_paper_weight": round(weight, 6)})
    return {
        "enabled_for_live": False,
        "advisory_only": True,
        "max_sleeve_weight": max_weight,
        "reserved_weight": round(max(1.0 - allocated, 0.0), 6),
        "ranked_sleeves": ranked,
    }


def _capital_lane(project_root: Path, sources: dict[str, dict[str, Any]], plan: dict[str, Any]) -> dict[str, Any]:
    ranked = _as_list(plan.get("ranked_sleeves"))
    nonzero = [row for row in ranked if _safe_float(_as_dict(row).get("advisory_paper_weight"), 0.0) > 0.0]
    blockers: list[str] = []
    warnings: list[str] = []
    if not ranked:
        blockers.append("no_sleeve_scorecards_available")
    if len(nonzero) < 2:
        warnings.append("fewer_than_two_sleeves_eligible_for_advisory_weight")
    top_score = _safe_float(_as_dict(nonzero[0] if nonzero else {}).get("advisory_score"), 0.0)
    score = min(100.0, 70.0 + min(len(nonzero), 4) * 5.0 + min(top_score, 10.0))
    return _lane(
        "advisory_capital_allocator",
        score=score,
        blockers=blockers,
        warnings=warnings,
        evidence={
            "advisory_only": True,
            "live_allocation_enabled": False,
            "eligible_sleeve_count": len(nonzero),
            "top_sleeves": ranked[:5],
            "reserved_weight": plan.get("reserved_weight"),
        },
        source_names=["paper_execution_truth_layer", "paper_performance"],
        sources=sources,
        project_root=project_root,
        recommended_actions=["use advisory paper weights to starve weak sleeves and study strong sleeves"],
    )


def _account_lane(project_root: Path, sources: dict[str, dict[str, Any]]) -> dict[str, Any]:
    study = sources["account_position"]
    roll = sources["covered_call_roll_watch"]
    account_count = _safe_int(study.get("account_count"), 0)
    position_count = _safe_int(study.get("position_count"), 0)
    cc_count = max(
        _safe_int(_as_dict(study.get("covered_call_roll_watch")).get("covered_call_count"), 0),
        _safe_int(roll.get("covered_call_count"), 0),
    )
    alert_count = max(
        _safe_int(_as_dict(study.get("covered_call_roll_watch")).get("alert_count"), 0),
        _safe_int(roll.get("alert_count"), 0),
    )
    blockers: list[str] = []
    warnings: list[str] = []
    score = 55.0
    if bool(study.get("ok", False)):
        score += 15.0
    if account_count > 0:
        score += 10.0
    else:
        blockers.append("no_accounts_visible")
    if position_count > 0:
        score += 10.0
    else:
        blockers.append("no_positions_visible")
    if cc_count > 0:
        score += 10.0
    else:
        warnings.append("no_covered_calls_detected")
    if alert_count > 0:
        warnings.append("covered_call_alerts_present")
        score -= min(alert_count * 5.0, 20.0)
    return _lane(
        "account_position_intelligence",
        score=score,
        blockers=blockers,
        warnings=warnings,
        evidence={
            "account_count": account_count,
            "position_count": position_count,
            "covered_call_count": cc_count,
            "covered_call_alert_count": alert_count,
            "roll_watch_status": roll.get("overall_status") or _as_dict(study.get("covered_call_roll_watch")).get("overall_status"),
        },
        source_names=["account_position", "covered_call_roll_watch"],
        sources=sources,
        project_root=project_root,
        recommended_actions=["keep all account snapshots and covered-call roll windows refreshed"],
    )


def _replay_lane(project_root: Path, sources: dict[str, dict[str, Any]]) -> dict[str, Any]:
    truth = sources["paper_execution_truth_layer"]
    execution_lab = sources["execution_lab"]
    counterfactual = sources["counterfactual_replay"]
    paper_replay = sources["paper_replay_drill"]
    gates = _as_dict(truth.get("gates"))
    replay_gate = _as_dict(gates.get("decision_replay_harness"))
    stress_gate = _as_dict(gates.get("market_regime_stress_mode"))
    blockers: list[str] = []
    warnings: list[str] = []
    if replay_gate and _status(replay_gate) not in {"ready", "ok"}:
        blockers.append("decision_replay_harness_not_ready")
    if stress_gate and _status(stress_gate) not in {"ready", "ok"}:
        warnings.append("stress_mode_not_ready")
    if not _artifact_ok(execution_lab):
        blockers.append("execution_lab_not_ok")
    if not _artifact_ok(counterfactual):
        blockers.append("counterfactual_replay_not_ok")
    if paper_replay and not _artifact_ok(paper_replay, allow_watch=True):
        warnings.append("paper_replay_drill_not_ok")
    scores = [
        _safe_float(replay_gate.get("score"), 0.0),
        _safe_float(stress_gate.get("score"), 0.0),
        100.0 if _artifact_ok(execution_lab) else 0.0,
        100.0 if _artifact_ok(counterfactual) else 0.0,
    ]
    score = sum(scores) / max(len(scores), 1)
    return _lane(
        "market_condition_replay",
        score=score,
        blockers=blockers,
        warnings=warnings,
        evidence={
            "decision_replay_status": replay_gate.get("status"),
            "stress_status": stress_gate.get("status"),
            "worst_slippage_bps": stress_gate.get("worst_slippage_bps"),
            "counterfactual_candidate_count": len(_as_list(counterfactual.get("top_candidates"))),
            "execution_lab_scenario_count": execution_lab.get("scenario_count"),
        },
        source_names=["execution_lab", "counterfactual_replay", "paper_replay_drill", "paper_execution_truth_layer"],
        sources=sources,
        project_root=project_root,
        recommended_actions=["keep crisis/stress replay in the promotion gate before any live scale"],
    )


def _attribution_lane(project_root: Path, sources: dict[str, dict[str, Any]]) -> dict[str, Any]:
    attribution = sources["strategy_attribution"]
    performance = sources["paper_performance"]
    truth = sources["paper_execution_truth_layer"]
    haircuts = _as_dict(truth.get("paper_pnl_haircut_ledger"))
    aggregates = _as_dict(attribution.get("aggregates"))
    row_count = _safe_int(aggregates.get("row_count"), _safe_int(attribution.get("row_count"), 0))
    file_count = _safe_int(attribution.get("file_count"), 0)
    market_closed_no_rows = bool(row_count == 0 and file_count == 0 and _non_trading_day(attribution.get("day")))
    blockers: list[str] = []
    warnings: list[str] = []
    score = 60.0
    if row_count > 0:
        score += 20.0
    elif market_closed_no_rows:
        score += 15.0
    else:
        warnings.append("strategy_attribution_has_no_rows")
    if performance:
        score += 10.0
    else:
        warnings.append("paper_performance_missing")
    if haircuts:
        score += 10.0
    else:
        warnings.append("paper_pnl_haircut_ledger_missing")
    return _lane(
        "decision_quality_attribution",
        score=score,
        blockers=blockers,
        warnings=warnings,
        evidence={
            "strategy_attribution_rows": row_count,
            "strategy_attribution_file_count": file_count,
            "attribution_coverage_state": "market_closed_no_rows" if market_closed_no_rows else ("active_rows" if row_count > 0 else "missing_rows"),
            "total_pnl_proxy": aggregates.get("total_pnl_proxy", attribution.get("total_pnl_proxy")),
            "raw_week_pnl": haircuts.get("raw_week_pnl"),
            "realism_adjusted_week_pnl": haircuts.get("realism_adjusted_week_pnl"),
        },
        source_names=["strategy_attribution", "paper_performance", "paper_execution_truth_layer"],
        sources=sources,
        project_root=project_root,
        recommended_actions=["split P/L by edge, market beta, timing, sleeve, data quality, and execution drag"],
    )


def _cockpit_lane(project_root: Path, sources: dict[str, dict[str, Any]]) -> dict[str, Any]:
    cockpit = sources["operator_cockpit"]
    a_plus = sources["a_plus_operating_packet"]
    blockers: list[str] = []
    warnings: list[str] = []
    score = 50.0
    if cockpit:
        score += 25.0
    else:
        warnings.append("operator_cockpit_missing")
    if _artifact_ok(cockpit, allow_watch=True):
        score += 10.0
    elif cockpit and not bool(a_plus.get("a_plus_ready", False)):
        warnings.append("operator_cockpit_not_ready")
    if a_plus:
        score += 10.0
    else:
        warnings.append("a_plus_operating_packet_missing")
    if bool(a_plus.get("a_plus_ready", False)):
        score += 5.0
    return _lane(
        "operator_cockpit",
        score=score,
        blockers=blockers,
        warnings=warnings,
        evidence={
            "operator_cockpit_status": _status(cockpit),
            "a_plus_ready": bool(a_plus.get("a_plus_ready", False)),
            "a_plus_grade": a_plus.get("overall_grade"),
        },
        source_names=["operator_cockpit", "a_plus_operating_packet"],
        sources=sources,
        project_root=project_root,
        recommended_actions=["surface this 1-10 packet inside the operator cockpit attention queue"],
    )


def _storage_lane(project_root: Path, sources: dict[str, dict[str, Any]]) -> dict[str, Any]:
    storage = sources["ingestion_storage"]
    retention = sources["storage_retention"]
    backpressure = _as_dict(storage.get("backpressure"))
    total_pending = _safe_int(backpressure.get("total_pending_lines"), 0)
    threshold = max(_safe_int(backpressure.get("pending_lines_threshold"), 15000), 1)
    oldest = _safe_float(backpressure.get("oldest_pending_age_seconds"), 0.0)
    blockers: list[str] = []
    warnings: list[str] = []
    score = 100.0
    if storage and not _artifact_ok(storage, allow_watch=True):
        blockers.append("ingestion_storage_not_ok")
        score -= 35.0
    if total_pending > threshold:
        blockers.append("pending_lines_above_threshold")
        score -= 30.0
    else:
        score -= min((total_pending / threshold) * 12.0, 12.0)
    if oldest > 240.0:
        warnings.append("oldest_pending_age_above_240s")
        score -= 10.0
    if retention:
        if _status(retention) in {"blocked", "critical", "failed"}:
            blockers.append("storage_retention_blocked")
            score -= 20.0
    else:
        warnings.append("storage_retention_unison_missing")
        score -= 5.0
    return _lane(
        "storage_ingestion_discipline",
        score=score,
        blockers=blockers,
        warnings=warnings,
        evidence={
            "total_pending_lines": total_pending,
            "pending_lines_threshold": threshold,
            "oldest_pending_age_seconds": round(oldest, 3),
            "retention_status": _status(retention),
            "storage_status": _status(storage),
        },
        source_names=["ingestion_storage", "storage_retention"],
        sources=sources,
        project_root=project_root,
        recommended_actions=["compact raw capture into feature summaries before storage pressure returns"],
    )


def _autonomous_lane(project_root: Path, sources: dict[str, dict[str, Any]]) -> dict[str, Any]:
    bot_quality = sources["bot_quality_autopilot"]
    autofix = sources["infrastructure_autofix"]
    self_model = sources["system_self_model"]
    whole = sources["whole_system_intelligence"]
    present = [name for name in ("bot_quality_autopilot", "infrastructure_autofix", "system_self_model", "whole_system_intelligence") if sources[name]]
    blockers: list[str] = []
    warnings: list[str] = []
    if not present:
        blockers.append("no_autonomous_improvement_artifacts_present")
    score = 55.0 + len(present) * 10.0
    for name, payload in (
        ("bot_quality_autopilot", bot_quality),
        ("infrastructure_autofix", autofix),
        ("system_self_model", self_model),
        ("whole_system_intelligence", whole),
    ):
        status = _status(payload)
        if payload and status in {"blocked", "critical", "failed", "error"}:
            warnings.append(f"{name}_not_ready")
            score -= 8.0
    return _lane(
        "autonomous_improvement_loop",
        score=score,
        blockers=blockers,
        warnings=warnings,
        evidence={
            "present_artifacts": present,
            "bot_quality_status": _status(bot_quality),
            "infrastructure_autofix_status": _status(autofix),
            "system_self_model_status": _status(self_model),
            "whole_system_intelligence_status": _status(whole),
            "requires_tests_before_accepting_changes": True,
        },
        source_names=["bot_quality_autopilot", "infrastructure_autofix", "system_self_model", "whole_system_intelligence"],
        sources=sources,
        project_root=project_root,
        recommended_actions=["let the system propose improvements, but require tests and operator review before applying them"],
    )


def build_payload(project_root: Path = PROJECT_ROOT) -> dict[str, Any]:
    project_root = Path(project_root).resolve()
    sources = _load_sources(project_root)
    capital_plan = _capital_plan(sources["paper_execution_truth_layer"])
    lanes = [
        _execution_truth_lane(project_root, sources),
        _acceptance_lane(project_root, sources),
        _canary_lane(project_root, sources),
        _capital_lane(project_root, sources, capital_plan),
        _account_lane(project_root, sources),
        _replay_lane(project_root, sources),
        _attribution_lane(project_root, sources),
        _cockpit_lane(project_root, sources),
        _storage_lane(project_root, sources),
        _autonomous_lane(project_root, sources),
    ]
    lane_scores = [_safe_float(row.get("score"), 0.0) for row in lanes]
    overall_score = sum(lane_scores) / max(len(lane_scores), 1)
    blockers = ordered_unique(
        f"{row['id']}:{blocker}"
        for row in lanes
        for blocker in _as_list(row.get("blockers"))
    )
    warnings = ordered_unique(
        f"{row['id']}:{warning}"
        for row in lanes
        for warning in _as_list(row.get("warnings"))
    )
    a_plus_lane_count = sum(1 for row in lanes if bool(row.get("a_plus", False)))
    ready_lane_count = sum(1 for row in lanes if str(row.get("status")) == "ready")
    overall_status = "ready" if not blockers and overall_score >= 90.0 else ("watch" if not blockers else "blocked")
    acceptance = next(row for row in lanes if row["id"] == "paper_live_acceptance_harness")
    canary = next(row for row in lanes if row["id"] == "supervised_live_canary_mode")
    safe_next_actions = ordered_unique(
        action
        for row in lanes
        for action in _as_list(row.get("recommended_actions"))
        if action
    )
    return {
        "timestamp_utc": iso_now(),
        "schema_version": 1,
        "ok": not blockers,
        "overall_status": overall_status,
        "overall_score": round(_clamp(overall_score), 6),
        "overall_grade": _grade(overall_score),
        "a_plus_ready": bool(not blockers and a_plus_lane_count == len(lanes)),
        "a_plus_lane_count": a_plus_lane_count,
        "ready_lane_count": ready_lane_count,
        "lane_count": len(lanes),
        "blocker_count": len(blockers),
        "warning_count": len(warnings),
        "blockers": blockers,
        "warnings": warnings,
        "authority_boundary": {
            "live_trading_enabled_by_this_artifact": False,
            "real_capital_allocation_enabled_by_this_artifact": False,
            "advisory_paper_allocation_only": True,
            "requires_operator_approval_for_live_canary": True,
        },
        "paper_to_live_acceptance": {
            "ready": bool(acceptance.get("ok", False)),
            "status": acceptance.get("status"),
            "blockers": acceptance.get("blockers", []),
            "live_canary_mode": _as_dict(canary.get("evidence")).get("recommended_mode"),
        },
        "advisory_capital_plan": capital_plan,
        "lanes": lanes,
        "safe_next_actions": safe_next_actions[:30],
        "source_files": {name: str(_source_path(project_root, name)) for name in SOURCE_FILES},
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build the 1-10 trading desk upgrade control packet.")
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--out-file", default=str(DEFAULT_OUT_PATH))
    parser.add_argument("--platform-os-out-file", default=str(DEFAULT_PLATFORM_OS_OUT_PATH))
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    payload = build_payload(Path(args.project_root))
    write_payload(Path(args.out_file), payload)
    if str(args.platform_os_out_file or "").strip():
        write_payload(Path(args.platform_os_out_file), payload)
    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        print(
            "trading_desk_upgrade_control "
            f"status={payload.get('overall_status')} "
            f"score={float(payload.get('overall_score', 0.0) or 0.0):.2f} "
            f"grade={payload.get('overall_grade')} "
            f"blockers={payload.get('blocker_count')}"
        )
    return 0 if payload.get("ok") else 2


if __name__ == "__main__":
    raise SystemExit(main())
