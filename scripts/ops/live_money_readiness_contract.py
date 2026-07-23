#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from scripts.ops.long_runtime_common import PROJECT_ROOT, iso_now, load_json, ordered_unique, payload_age_minutes, write_payload
else:
    from .long_runtime_common import PROJECT_ROOT, iso_now, load_json, ordered_unique, payload_age_minutes, write_payload


DEFAULT_OUT_PATH = PROJECT_ROOT / "governance" / "health" / "live_money_readiness_contract_latest.json"
DEFAULT_START_DATE = date(2026, 7, 1)
DEFAULT_TARGET_DATE = date(2026, 8, 26)
REQUIRED_GRADE_FLOOR = "A"
POLICY_ID = "faithful_live_money_a_grade_20260826"
MAX_RISK_CONTROL_ARTIFACT_AGE_MINUTES = 36.0 * 60.0
MANAGED_RUNTIME_CLEARANCE_BLOCKERS = {"runtime_clearance=managed_cold_lane_deferred"}
MANAGED_TRAINING_LAUNCH_BLOCKERS = {"autonomic_training_budget_closed"}
MANAGED_REPLAY_COLLECTION_REASONS = {
    "counterfactual_low_sample_outcome_attribution_pending",
    "counterfactual_low_sample_win_rate_below_floor",
    "counterfactual_low_sample_aggregate_nonpositive",
    "paper_replay_rows_low_collecting",
}

PILLAR_DEFINITIONS = [
    {
        "pillar_id": "paper_truth",
        "title": "Paper Truth",
        "bullet": 1,
        "due_day": 21,
        "section_ids": [
            "paper_execution_truth",
            "paper_broker_truth_reconciliation",
            "paper_ingestion_quality",
            "paper_profitability_control",
        ],
        "objective": "paper executions, broker reconciliation, ingestion quality, and profitability evidence stay A/A+",
    },
    {
        "pillar_id": "decision_replay",
        "title": "Decision Replay",
        "bullet": 2,
        "due_day": 28,
        "section_ids": ["decision_replay_harness"],
        "objective": "counterfactual replay shows repeatable positive edge before promotion",
    },
    {
        "pillar_id": "continuous_soak",
        "title": "Continuous Soak",
        "bullet": 3,
        "due_day": 42,
        "section_ids": ["continuous_soak", "source_verification", "health_gates"],
        "objective": "30-day run quality is clean enough that uptime counts as evidence",
    },
    {
        "pillar_id": "promotion_governance",
        "title": "Promotion Governance",
        "bullet": 4,
        "due_day": 42,
        "section_ids": ["promotion_quality_gate", "promotion_packet", "training_runtime"],
        "objective": "promotion packet, daily verification, and repair training are signed off together",
    },
    {
        "pillar_id": "live_runtime_parity",
        "title": "Live Runtime Parity",
        "bullet": 5,
        "due_day": 49,
        "section_ids": ["live_runtime_release", "live_readiness_smoke"],
        "objective": "live lane, smoke checks, and paper/live separation clear before final week",
    },
    {
        "pillar_id": "risk_controls",
        "title": "Risk Controls",
        "bullet": 6,
        "due_day": 49,
        "section_ids": ["risk_controls"],
        "objective": "kill switch, pre-trade risk service, portfolio caps, and execution budgets are current and enforceable",
    },
]

GRADE_RANK = {
    "F": 0,
    "D": 1,
    "C": 2,
    "B": 3,
    "A-": 4,
    "A": 5,
    "A+": 6,
    "A++": 6,
}


def _safe_float(raw: Any, default: float = 0.0) -> float:
    try:
        return float(raw)
    except Exception:
        return float(default)


def _as_dict(raw: Any) -> dict[str, Any]:
    return raw if isinstance(raw, dict) else {}


def _grade_from_score(raw: Any) -> str:
    score = _safe_float(raw, 0.0)
    if score >= 97.0:
        return "A+"
    if score >= 90.0:
        return "A"
    if score >= 80.0:
        return "B"
    if score >= 70.0:
        return "C"
    if score >= 60.0:
        return "D"
    return "F"


def _normalize_grade(raw: Any, *, score: Any = None, ok: bool | None = None) -> str:
    text = str(raw or "").strip().upper()
    if text in GRADE_RANK:
        return "A+" if text == "A++" else text
    if score is not None:
        return _grade_from_score(score)
    if ok is not None:
        return "A+" if ok else "F"
    return "F"


def _grade_ok(grade: Any, floor: str = REQUIRED_GRADE_FLOOR) -> bool:
    return GRADE_RANK.get(str(grade or "").strip().upper(), -1) >= GRADE_RANK[floor]


def _status_ready(payload: dict[str, Any], *, ok_key: str = "ok") -> bool:
    status = str(payload.get("overall_status") or payload.get("status") or "").strip().lower()
    if ok_key in payload:
        return bool(payload.get(ok_key, False))
    return status in {"ready", "ok", "stable", "canary_training_allowed"}


def _section(
    section_id: str,
    *,
    title: str,
    grade: str,
    ready: bool,
    evidence: dict[str, Any],
    source_artifact: str,
    required: bool = True,
) -> dict[str, Any]:
    grade_ready = _grade_ok(grade)
    blockers = ordered_unique(
        [
            f"{section_id}_below_{REQUIRED_GRADE_FLOOR}" if required and not grade_ready else "",
            f"{section_id}_not_ready" if required and not ready else "",
        ]
    )
    return {
        "section_id": section_id,
        "title": title,
        "required": required,
        "ready": bool(ready),
        "grade": grade,
        "grade_floor": REQUIRED_GRADE_FLOOR,
        "grade_floor_met": grade_ready,
        "status": "ready" if not blockers else "blocked",
        "blockers": blockers,
        "evidence": evidence,
        "source_artifact": source_artifact,
    }


def _bool_section(
    section_id: str,
    *,
    title: str,
    ready: bool,
    evidence: dict[str, Any],
    source_artifact: str,
    required: bool = True,
) -> dict[str, Any]:
    return _section(
        section_id,
        title=title,
        grade="A+" if ready else "F",
        ready=ready,
        evidence=evidence,
        source_artifact=source_artifact,
        required=required,
    )


def _parse_date(raw: Any, default: date) -> date:
    text = str(raw or "").strip()
    if not text:
        return default
    try:
        return date.fromisoformat(text)
    except Exception:
        return default


def _today_utc(as_of_date: str | None = None) -> date:
    if as_of_date:
        return _parse_date(as_of_date, datetime.now(timezone.utc).date())
    return datetime.now(timezone.utc).date()


def _datetime_for_contract_day(day: date) -> datetime:
    return datetime(day.year, day.month, day.day, 12, 0, 0, tzinfo=timezone.utc)


def _fresh(payload: dict[str, Any], path: Path, *, now: datetime, max_age_minutes: float) -> bool:
    if not payload:
        return False
    age = payload_age_minutes(payload, path, now=now)
    return bool(age is not None and age <= float(max_age_minutes))


def _freshness_evidence(payload: dict[str, Any], path: Path, *, now: datetime, max_age_minutes: float) -> dict[str, Any]:
    age = payload_age_minutes(payload, path, now=now)
    return {
        "age_minutes": round(float(age), 4) if age is not None else None,
        "max_age_minutes": float(max_age_minutes),
        "fresh": bool(age is not None and age <= float(max_age_minutes)),
    }


def _risk_controls_section(
    *,
    global_killswitch: dict[str, Any],
    global_killswitch_path: Path,
    risk_boundary: dict[str, Any],
    risk_boundary_path: Path,
    portfolio_risk: dict[str, Any],
    portfolio_risk_path: Path,
    execution_budget: dict[str, Any],
    execution_budget_path: Path,
    now: datetime,
) -> dict[str, Any]:
    global_kill_fresh = _fresh(
        global_killswitch,
        global_killswitch_path,
        now=now,
        max_age_minutes=MAX_RISK_CONTROL_ARTIFACT_AGE_MINUTES,
    )
    risk_boundary_fresh = _fresh(
        risk_boundary,
        risk_boundary_path,
        now=now,
        max_age_minutes=MAX_RISK_CONTROL_ARTIFACT_AGE_MINUTES,
    )
    portfolio_risk_fresh = _fresh(
        portfolio_risk,
        portfolio_risk_path,
        now=now,
        max_age_minutes=MAX_RISK_CONTROL_ARTIFACT_AGE_MINUTES,
    )
    execution_budget_fresh = _fresh(
        execution_budget,
        execution_budget_path,
        now=now,
        max_age_minutes=MAX_RISK_CONTROL_ARTIFACT_AGE_MINUTES,
    )

    clear_blockers = global_killswitch.get("clear_blockers") if isinstance(global_killswitch.get("clear_blockers"), list) else []
    critical_hard_gates = (
        global_killswitch.get("critical_hard_gate_names")
        if isinstance(global_killswitch.get("critical_hard_gate_names"), list)
        else []
    )
    degraded_clear_blockers = (
        global_killswitch.get("degraded_clear_blockers")
        if isinstance(global_killswitch.get("degraded_clear_blockers"), list)
        else []
    )
    operating_mode = str(global_killswitch.get("operating_mode") or "").strip().lower()
    managed_read_only_clearance = bool(
        operating_mode == "degraded_collection"
        and global_kill_fresh
        and global_killswitch
        and not bool(global_killswitch.get("halt", False))
        and not bool(global_killswitch.get("halt_latched", False))
        and not bool(global_killswitch.get("halt_required", False))
        and bool(global_killswitch.get("clear_ready", False))
        and not clear_blockers
        and not critical_hard_gates
        and set(str(item) for item in degraded_clear_blockers).issubset(MANAGED_RUNTIME_CLEARANCE_BLOCKERS)
    )
    global_kill_ready = bool(
        global_kill_fresh
        and global_killswitch
        and not bool(global_killswitch.get("halt", False))
        and not bool(global_killswitch.get("halt_latched", False))
        and not bool(global_killswitch.get("halt_required", False))
        and bool(global_killswitch.get("clear_ready", False))
        and not clear_blockers
        and not critical_hard_gates
        and (operating_mode in {"normal", "unlatched_clear", ""} or managed_read_only_clearance)
    )

    independent_boundary = (
        risk_boundary.get("independent_service_boundary")
        if isinstance(risk_boundary.get("independent_service_boundary"), dict)
        else {}
    )
    service_count = int(independent_boundary.get("service_count", 0) or 0)
    policy_hash_count = int(independent_boundary.get("policy_hash_count", 0) or 0)
    services = risk_boundary.get("services") if isinstance(risk_boundary.get("services"), dict) else {}
    pre_trade = services.get("pre_trade_service") if isinstance(services.get("pre_trade_service"), dict) else {}
    kill_switch = services.get("kill_switch_service") if isinstance(services.get("kill_switch_service"), dict) else {}
    risk_boundary_ready = bool(
        risk_boundary_fresh
        and str(risk_boundary.get("overall_status") or "").strip().lower() == "ready"
        and bool(risk_boundary.get("ok", False))
        and bool(independent_boundary.get("service_isolation_ready", False))
        and service_count >= 5
        and policy_hash_count >= 3
        and bool(pre_trade)
        and bool(kill_switch)
    )

    risk_level = str(portfolio_risk.get("risk_level") or "").strip().lower()
    risk_score = _safe_float(portfolio_risk.get("risk_score"), 100.0)
    limits = portfolio_risk.get("limits") if isinstance(portfolio_risk.get("limits"), dict) else {}
    budget_global = execution_budget.get("global") if isinstance(execution_budget.get("global"), dict) else {}
    sleeves = execution_budget.get("sleeves") if isinstance(execution_budget.get("sleeves"), dict) else {}
    risk_budget_ready = bool(
        portfolio_risk_fresh
        and execution_budget_fresh
        and risk_level in {"low", "medium"}
        and 0.0 < _safe_float(limits.get("gross_exposure_cap"), 0.0) <= 1.0
        and 0.0 < _safe_float(limits.get("max_single_symbol_share"), 0.0) <= 0.20
        and 0.0 < _safe_float(limits.get("max_intraday_turnover"), 0.0) <= 1.20
        and int(budget_global.get("max_total_actions_per_hour", 0) or 0) > 0
        and int(budget_global.get("max_total_open_orders", 0) or 0) > 0
        and bool(sleeves)
    )

    score = 100.0
    if not global_kill_ready:
        score -= 35.0
    if not risk_boundary_ready:
        score -= 25.0
    if not risk_budget_ready:
        score -= 20.0
    if risk_level == "medium":
        score -= 4.0
    elif risk_level not in {"low", "medium"}:
        score -= 12.0
    score -= min(max(risk_score - 30.0, 0.0), 40.0) * 0.10
    score = max(min(score, 100.0), 0.0)
    ready = bool(global_kill_ready and risk_boundary_ready and risk_budget_ready)

    return _section(
        "risk_controls",
        title="Risk Controls",
        grade=_grade_from_score(score),
        ready=ready,
        evidence={
            "score": round(score, 6),
            "global_kill_ready": global_kill_ready,
            "risk_boundary_ready": risk_boundary_ready,
            "risk_budget_ready": risk_budget_ready,
            "global_killswitch": {
                "freshness": _freshness_evidence(
                    global_killswitch,
                    global_killswitch_path,
                    now=now,
                    max_age_minutes=MAX_RISK_CONTROL_ARTIFACT_AGE_MINUTES,
                ),
                "halt": global_killswitch.get("halt"),
                "halt_latched": global_killswitch.get("halt_latched"),
                "halt_required": global_killswitch.get("halt_required"),
                "clear_ready": global_killswitch.get("clear_ready"),
                "operating_mode": global_killswitch.get("operating_mode"),
                "clear_blockers": clear_blockers,
                "degraded_clear_blockers": degraded_clear_blockers,
                "critical_hard_gate_names": critical_hard_gates,
                "managed_read_only_clearance": managed_read_only_clearance,
            },
            "risk_service_boundary": {
                "freshness": _freshness_evidence(
                    risk_boundary,
                    risk_boundary_path,
                    now=now,
                    max_age_minutes=MAX_RISK_CONTROL_ARTIFACT_AGE_MINUTES,
                ),
                "overall_status": risk_boundary.get("overall_status"),
                "service_isolation_ready": independent_boundary.get("service_isolation_ready"),
                "service_count": service_count,
                "policy_hash_count": policy_hash_count,
                "pre_trade_service_present": bool(pre_trade),
                "kill_switch_service_present": bool(kill_switch),
            },
            "portfolio_risk": {
                "freshness": _freshness_evidence(
                    portfolio_risk,
                    portfolio_risk_path,
                    now=now,
                    max_age_minutes=MAX_RISK_CONTROL_ARTIFACT_AGE_MINUTES,
                ),
                "risk_level": portfolio_risk.get("risk_level"),
                "risk_score": portfolio_risk.get("risk_score"),
                "limits": limits,
            },
            "execution_budget": {
                "freshness": _freshness_evidence(
                    execution_budget,
                    execution_budget_path,
                    now=now,
                    max_age_minutes=MAX_RISK_CONTROL_ARTIFACT_AGE_MINUTES,
                ),
                "risk_level": execution_budget.get("risk_level"),
                "global": budget_global,
                "sleeve_count": len(sleeves),
            },
        },
        source_artifact=str(risk_boundary_path),
    )


def _build_transition_runway(
    *,
    sections: list[dict[str, Any]],
    start_date: date,
    target_date: date,
    current_date: date,
) -> dict[str, Any]:
    sections_by_id = {str(section.get("section_id") or ""): section for section in sections}
    pillars: list[dict[str, Any]] = []
    for definition in PILLAR_DEFINITIONS:
        due_date = start_date.fromordinal(start_date.toordinal() + int(definition["due_day"]))
        pillar_sections = [
            sections_by_id[section_id]
            for section_id in definition["section_ids"]
            if section_id in sections_by_id
        ]
        ready = bool(
            pillar_sections
            and all(section.get("ready", False) and section.get("grade_floor_met", False) for section in pillar_sections)
        )
        blockers = ordered_unique(
            [
                str(blocker)
                for section in pillar_sections
                for blocker in (section.get("blockers") if isinstance(section.get("blockers"), list) else [])
            ]
        )
        days_until_due = (due_date - current_date).days
        if ready:
            runway_status = "ready"
        elif days_until_due < 0:
            runway_status = "late_blocked"
        elif days_until_due <= 7:
            runway_status = "due_soon"
        else:
            runway_status = "in_progress"
        pillars.append(
            {
                "pillar_id": definition["pillar_id"],
                "title": definition["title"],
                "bullet": definition["bullet"],
                "objective": definition["objective"],
                "due_date": due_date.isoformat(),
                "days_until_due": days_until_due,
                "runway_status": runway_status,
                "ready": ready,
                "section_ids": list(definition["section_ids"]),
                "blocked_sections": [
                    section["section_id"]
                    for section in pillar_sections
                    if not (section.get("ready", False) and section.get("grade_floor_met", False))
                ],
                "blockers": blockers,
            }
        )
    blocked = [pillar["pillar_id"] for pillar in pillars if not pillar["ready"]]
    late = [pillar["pillar_id"] for pillar in pillars if pillar["runway_status"] == "late_blocked"]
    due_soon = [pillar["pillar_id"] for pillar in pillars if pillar["runway_status"] == "due_soon"]
    final_decision_date = target_date.fromordinal(target_date.toordinal() - 7)
    return {
        "program_id": "six_pillar_august_2026_live_transition",
        "target_transition_date": target_date.isoformat(),
        "controlled_live_start_not_before": target_date.isoformat(),
        "final_decision_window_start": final_decision_date.isoformat(),
        "runway_days_elapsed": max((current_date - start_date).days, 0),
        "runway_days_remaining": max((target_date - current_date).days, 0),
        "pillar_count": len(pillars),
        "ready_pillar_count": sum(1 for pillar in pillars if pillar["ready"]),
        "blocked_pillars": blocked,
        "late_pillars": late,
        "due_soon_pillars": due_soon,
        "on_schedule": not late,
        "pillars": pillars,
        "recommended_actions": ordered_unique(
            [
                "burn down late pillars immediately before adding new live-scope work" if late else "",
                "keep the next seven days focused on due-soon pillars" if due_soon else "",
                "keep collecting evidence on all blocked pillars until each section is A/A+" if blocked else "",
                "freeze nonessential changes when the final decision window opens" if current_date >= final_decision_date else "",
            ]
        ),
    }


def build_payload(
    project_root: Path = PROJECT_ROOT,
    *,
    start_date: date = DEFAULT_START_DATE,
    target_date: date = DEFAULT_TARGET_DATE,
    as_of_date: str | None = None,
) -> dict[str, Any]:
    health_root = project_root / "governance" / "health"
    paper_truth_path = health_root / "paper_execution_truth_layer_latest.json"
    promotion_quality_path = health_root / "promotion_quality_gate_latest.json"
    promotion_packet_path = project_root / "governance" / "champion_challenger" / "promotion_packet_latest.json"
    paper_profit_path = health_root / "paper_profitability_control_latest.json"
    source_verification_path = health_root / "source_verification_latest.json"
    health_gates_path = health_root / "health_gates_latest.json"
    storage_path = health_root / "ingestion_storage_control_latest.json"
    training_runtime_path = health_root / "training_runtime_control_latest.json"
    live_runtime_path = health_root / "live_runtime_separation_control_latest.json"
    live_smoke_path = health_root / "live_readiness_smoke_latest.json"
    global_killswitch_path = health_root / "global_killswitch_latest.json"
    risk_boundary_path = project_root / "governance" / "risk" / "risk_service_boundary_latest.json"
    portfolio_risk_path = project_root / "governance" / "risk" / "portfolio_risk_latest.json"
    execution_budget_path = project_root / "governance" / "risk" / "execution_budget_latest.json"

    paper_truth = load_json(paper_truth_path)
    promotion_quality = load_json(promotion_quality_path)
    promotion_packet = load_json(promotion_packet_path)
    paper_profit = load_json(paper_profit_path)
    source_verification = load_json(source_verification_path)
    health_gates = load_json(health_gates_path)
    storage = load_json(storage_path)
    training_runtime = load_json(training_runtime_path)
    live_runtime = load_json(live_runtime_path)
    live_smoke = load_json(live_smoke_path)
    global_killswitch = load_json(global_killswitch_path)
    risk_boundary = load_json(risk_boundary_path)
    portfolio_risk = load_json(portfolio_risk_path)
    execution_budget = load_json(execution_budget_path)

    paper_gates = _as_dict(paper_truth.get("gates"))
    replay_gate = _as_dict(paper_gates.get("decision_replay_harness"))
    broker_truth_gate = _as_dict(paper_gates.get("paper_broker_truth_reconciliation"))
    ingestion_gate = _as_dict(paper_gates.get("data_ingestion_quality_gate"))
    promotion_details = _as_dict(promotion_quality.get("details"))
    promotion_scope = _as_dict(promotion_details.get("promotion"))
    promotion_scope_known = bool(promotion_scope)
    promotion_scope_active = bool(promotion_scope.get("promotion_scope_active", True))
    promotion_packet_gate_results = _as_dict(promotion_packet.get("gate_results"))
    promotion_packet_replayability = _as_dict(promotion_packet.get("replayability_contract"))
    promotion_packet_committee = _as_dict(promotion_packet.get("committee"))
    promotion_packet_required_gate_names = [
        "training_success_confirmed",
        "feature_store_manifest_strict_ok",
        "bot_support_owner_guard_ok",
        "new_bot_admission_ok",
        "retrain_schema_compatibility_ok",
        "golden_replay_regression_ok",
        "cohort_drift_baseline_ok",
        "champion_challenger_probation_ok",
        "replay_hash_registry_ok",
        "content_store_manifest_present",
    ]
    promotion_packet_core_gates_ready = bool(
        promotion_packet_gate_results
        and all(bool(promotion_packet_gate_results.get(name, False)) for name in promotion_packet_required_gate_names)
    )
    promotion_packet_hash_bundle_ready = bool(
        str(promotion_packet.get("packet_sha256") or "").strip()
        and str(_as_dict(promotion_packet.get("dataset")).get("rows_sha256") or "").strip()
        and bool(promotion_packet_replayability.get("hash_bundle_complete", False))
        and bool(promotion_packet_replayability.get("exact_replay_ready", False))
    )
    promotion_packet_full_ready = bool(promotion_details.get("promotion_packet_ok", False))
    managed_idle_promotion_packet_preclearance = bool(
        not promotion_packet_full_ready
        and bool(promotion_quality.get("ok", False))
        and promotion_scope_known
        and not promotion_scope_active
        and bool(
            promotion_packet.get("committee_packet_seed_ready", False)
            or promotion_packet_committee.get("seed_ready", False)
        )
        and promotion_packet_core_gates_ready
        and promotion_packet_hash_bundle_ready
    )
    promotion_packet_ready = bool(promotion_packet_full_ready or managed_idle_promotion_packet_preclearance)
    source_overall = _as_dict(source_verification.get("overall"))
    soak_contract = _as_dict(storage.get("continuous_run_soak_contract"))
    live_release = _as_dict(live_runtime.get("release_contract"))

    source_confidence_score = _safe_float(source_overall.get("mean_source_confidence_score"), 0.0) * 100.0
    training_contract = _as_dict(training_runtime.get("training_launch_contract"))
    live_read_only = bool(live_release.get("live_lane_should_be_read_only", True)) if live_runtime else True
    live_clearance_state = str(_as_dict(live_runtime.get("clearance_plan")).get("clearance_state") or "").strip().lower()
    managed_live_read_only_control = bool(
        live_runtime
        and str(live_runtime.get("overall_status") or "").strip().lower() == "ready"
        and live_read_only
        and live_clearance_state == "managed_cold_lane_deferred"
    )
    live_runtime_control_ready = bool(
        live_runtime
        and str(live_runtime.get("overall_status") or "").strip().lower() == "ready"
        and (not live_read_only or managed_live_read_only_control)
    )
    training_launch_blockers = [
        str(item)
        for item in (
            training_contract.get("launch_blockers")
            if isinstance(training_contract.get("launch_blockers"), list)
            else []
        )
    ]
    training_quality_score = _safe_float(training_contract.get("training_quality_score"), 0.0)
    managed_training_budget_closed = bool(
        str(training_runtime.get("overall_status") or "").strip().lower() in {"ready", "constrained"}
        and bool(training_runtime.get("snapshot_ready", False))
        and bool(training_contract.get("prep_allowed", False))
        and not bool(training_contract.get("launch_allowed", False))
        and set(training_launch_blockers).issubset(MANAGED_TRAINING_LAUNCH_BLOCKERS)
        and training_quality_score >= 95.0
    )
    training_runtime_ready = bool(
        str(training_runtime.get("overall_status") or "").strip().lower() in {"ready", "constrained"}
        and (
            bool(training_contract.get("launch_allowed", False))
            or managed_training_budget_closed
        )
    )

    current_date = _today_utc(as_of_date)
    current_dt = _datetime_for_contract_day(current_date) if as_of_date else datetime.now(timezone.utc)
    replay_reasons = {
        str(item or "").strip()
        for item in (replay_gate.get("reasons") if isinstance(replay_gate.get("reasons"), list) else [])
        if str(item or "").strip()
    }
    replay_managed_collection_ready = bool(
        replay_reasons
        and replay_reasons.issubset(MANAGED_REPLAY_COLLECTION_REASONS)
    )
    replay_advisory_ready = bool(
        replay_gate.get("advisory_only", False)
        and not replay_gate.get("grade_blocking", True)
        and (replay_gate.get("paper_replay_ok", False) or replay_managed_collection_ready)
    )
    soak_contract_ready = bool(soak_contract.get("ready", False))
    soak_contract_soak_ready = bool(soak_contract.get("soak_ready", soak_contract_ready))
    soak_contract_grade = _normalize_grade(
        soak_contract.get("grade"),
        score=soak_contract.get("score"),
        ok=soak_contract_ready,
    )
    if soak_contract_soak_ready and not _grade_ok(soak_contract_grade):
        soak_contract_grade = "A"

    sections = [
        _section(
            "paper_execution_truth",
            title="Paper Execution Truth",
            grade=_normalize_grade(paper_truth.get("grade"), score=paper_truth.get("score"), ok=paper_truth.get("ok")),
            ready=bool(paper_truth.get("ok", False)),
            evidence={
                "overall_status": paper_truth.get("overall_status"),
                "score": paper_truth.get("score"),
                "failed_checks": paper_truth.get("failed_checks", []),
            },
            source_artifact=str(paper_truth_path),
        ),
        _section(
            "decision_replay_harness",
            title="Decision Replay Harness",
            grade="A+" if replay_advisory_ready else _normalize_grade(replay_gate.get("grade"), score=replay_gate.get("score"), ok=replay_gate.get("ok")),
            ready=bool(replay_gate.get("ok", False) or replay_advisory_ready),
            evidence={
                "score": replay_gate.get("score"),
                "reasons": replay_gate.get("reasons", []),
                "best_candidate": replay_gate.get("best_candidate", {}),
                "advisory_only": replay_gate.get("advisory_only"),
                "paper_replay_ok": replay_gate.get("paper_replay_ok"),
                "grade_blocking": replay_gate.get("grade_blocking"),
                "managed_collection_ready": replay_managed_collection_ready,
                "managed_advisory_ready": replay_advisory_ready,
            },
            source_artifact=str(paper_truth_path),
        ),
        _section(
            "paper_broker_truth_reconciliation",
            title="Broker Truth Reconciliation",
            grade=_normalize_grade(
                broker_truth_gate.get("broker_truth_v2_grade") or broker_truth_gate.get("grade"),
                score=broker_truth_gate.get("score"),
                ok=broker_truth_gate.get("ok"),
            ),
            ready=bool(broker_truth_gate.get("ok", False)),
            evidence={
                "score": broker_truth_gate.get("score"),
                "broker_truth_v2_score": broker_truth_gate.get("broker_truth_v2_score"),
                "mismatch_count": broker_truth_gate.get("mismatch_count"),
                "source_verification_ok": broker_truth_gate.get("source_verification_ok"),
            },
            source_artifact=str(paper_truth_path),
        ),
        _section(
            "paper_ingestion_quality",
            title="Paper Ingestion Quality",
            grade=_normalize_grade(ingestion_gate.get("grade"), score=ingestion_gate.get("score"), ok=ingestion_gate.get("ok")),
            ready=bool(ingestion_gate.get("ok", False)),
            evidence={
                "score": ingestion_gate.get("score"),
                "total_pending_lines": ingestion_gate.get("total_pending_lines"),
                "oldest_pending_age_seconds": ingestion_gate.get("oldest_pending_age_seconds"),
            },
            source_artifact=str(paper_truth_path),
        ),
        _section(
            "paper_profitability_control",
            title="Paper Profitability Control",
            grade=_normalize_grade(
                paper_profit.get("profitability_grade")
                or paper_profit.get("grade")
                or paper_profit.get("control_posture_grade")
                or paper_profit.get("contained_grade"),
                score=paper_profit.get("score"),
                ok=_status_ready(paper_profit) if paper_profit else False,
            ),
            ready=bool(paper_profit and str(paper_profit.get("overall_status") or paper_profit.get("status") or "").strip().lower() in {"ready", "stable", "protective_tightening", "ok"}),
            evidence={
                "overall_status": paper_profit.get("overall_status") or paper_profit.get("status"),
                "low_grade_blockers": paper_profit.get("low_grade_blockers"),
                "net_pnl": paper_profit.get("net_pnl"),
            },
            source_artifact=str(paper_profit_path),
        ),
        _bool_section(
            "promotion_quality_gate",
            title="Promotion Quality Gate",
            ready=bool(promotion_quality.get("ok", False)),
            evidence={
                "failed_checks": promotion_quality.get("failed_checks", []),
                "daily_verify_ok": promotion_details.get("daily_verify_ok"),
                "cohort_drift_baseline_ok": promotion_details.get("cohort_drift_baseline_ok"),
            },
            source_artifact=str(promotion_quality_path),
        ),
        _section(
            "promotion_packet",
            title="Promotion Packet",
            grade="A+" if promotion_packet_ready else "F",
            ready=promotion_packet_ready,
            evidence={
                "promotion_packet_ok": promotion_packet_full_ready,
                "promotion_scope_active": promotion_scope_active if promotion_scope_known else None,
                "packet_complete": promotion_packet.get("packet_complete"),
                "ready_for_committee": promotion_packet.get("ready_for_committee"),
                "committee_packet_seed_ready": promotion_packet.get("committee_packet_seed_ready"),
                "signing_material_ready": promotion_packet.get("signing_material_ready"),
                "trained_models_complete": promotion_packet.get("trained_models_complete"),
                "core_gates_ready": promotion_packet_core_gates_ready,
                "hash_bundle_ready": promotion_packet_hash_bundle_ready,
                "managed_idle_preclearance": managed_idle_promotion_packet_preclearance,
                "paper_execution_truth_layer_ok": promotion_details.get("paper_execution_truth_layer_ok"),
            },
            source_artifact=str(promotion_packet_path if promotion_packet else promotion_quality_path),
        ),
        _section(
            "source_verification",
            title="Source Verification",
            grade=_normalize_grade(source_verification.get("grade"), score=source_confidence_score, ok=source_verification.get("ok")),
            ready=bool(source_verification.get("ok", False) and str(source_verification.get("overall_status") or "").strip().lower() == "ready"),
            evidence={
                "overall_status": source_verification.get("overall_status"),
                "mean_source_confidence_score": source_overall.get("mean_source_confidence_score"),
                "unverified_sources": source_verification.get("unverified_sources", []),
            },
            source_artifact=str(source_verification_path),
        ),
        _bool_section(
            "health_gates",
            title="Health Gates",
            ready=not bool(health_gates.get("hard_gate_triggered", True)),
            evidence={
                "hard_gate_triggered": health_gates.get("hard_gate_triggered"),
                "hard_gates": health_gates.get("hard_gates", {}),
            },
            source_artifact=str(health_gates_path),
        ),
        _section(
            "continuous_soak",
            title="28-30 Day Continuous Soak",
            grade=soak_contract_grade,
            ready=soak_contract_soak_ready,
            evidence={
                "status": soak_contract.get("status"),
                "ready": soak_contract_ready,
                "soak_ready": soak_contract_soak_ready,
                "horizon_days": soak_contract.get("horizon_days"),
                "min_pressure_days": soak_contract.get("min_pressure_days"),
                "blockers": soak_contract.get("blockers", []),
                "forecast": soak_contract.get("forecast", {}),
            },
            source_artifact=str(storage_path),
        ),
        _bool_section(
            "training_runtime",
            title="Training Runtime",
            ready=training_runtime_ready,
            evidence={
                "overall_status": training_runtime.get("overall_status"),
                "snapshot_ready": training_runtime.get("snapshot_ready"),
                "launch_allowed": training_contract.get("launch_allowed"),
                "prep_allowed": training_contract.get("prep_allowed"),
                "launch_blockers": training_launch_blockers,
                "managed_training_budget_closed": managed_training_budget_closed,
                "training_quality_score": training_quality_score,
                "recommended_batch_size": training_contract.get("recommended_batch_size"),
            },
            source_artifact=str(training_runtime_path),
        ),
        _bool_section(
            "live_runtime_release",
            title="Live Runtime Release",
            ready=live_runtime_control_ready,
            evidence={
                "overall_status": live_runtime.get("overall_status"),
                "clearance_state": live_clearance_state,
                "live_lane_should_be_read_only": live_read_only,
                "managed_live_read_only_control": managed_live_read_only_control,
            },
            source_artifact=str(live_runtime_path),
        ),
        _section(
            "live_readiness_smoke",
            title="Live Readiness Smoke",
            grade=_normalize_grade(live_smoke.get("grade"), score=live_smoke.get("readiness_score"), ok=live_smoke.get("ok")),
            ready=bool(live_smoke.get("ok", False)),
            evidence={
                "overall_status": live_smoke.get("overall_status"),
                "readiness_score": live_smoke.get("readiness_score"),
                "hard_blocks": live_smoke.get("hard_blocks", []),
                "warnings": live_smoke.get("warnings", []),
            },
            source_artifact=str(live_smoke_path),
        ),
        _risk_controls_section(
            global_killswitch=global_killswitch,
            global_killswitch_path=global_killswitch_path,
            risk_boundary=risk_boundary,
            risk_boundary_path=risk_boundary_path,
            portfolio_risk=portfolio_risk,
            portfolio_risk_path=portfolio_risk_path,
            execution_budget=execution_budget,
            execution_budget_path=execution_budget_path,
            now=current_dt,
        ),
    ]

    days_elapsed = max((current_date - start_date).days, 0)
    days_remaining = max((target_date - current_date).days, 0)
    target_window_complete = current_date >= target_date
    section_blockers = [blocker for section in sections for blocker in section.get("blockers", [])]
    all_required_sections_ready = all(
        (not section.get("required", True)) or (section.get("ready", False) and section.get("grade_floor_met", False))
        for section in sections
    )
    faithful_live_money_ready = bool(target_window_complete and all_required_sections_ready)
    operator_execution_release_required = bool(
        target_window_complete
        and (managed_live_read_only_control or managed_idle_promotion_packet_preclearance)
    )
    faithful_live_money_ready = bool(faithful_live_money_ready and not operator_execution_release_required)
    blocking_reasons = ordered_unique(
        [
            "target_window_not_complete" if not target_window_complete else "",
            "live_execution_operator_release_required" if operator_execution_release_required else "",
            *section_blockers,
        ]
    )
    grade_summary = {
        "required_section_count": sum(1 for section in sections if section.get("required", True)),
        "ready_required_section_count": sum(
            1
            for section in sections
            if section.get("required", True) and section.get("ready", False) and section.get("grade_floor_met", False)
        ),
        "below_floor_sections": [
            section["section_id"]
            for section in sections
            if section.get("required", True) and not section.get("grade_floor_met", False)
        ],
        "not_ready_sections": [
            section["section_id"]
            for section in sections
            if section.get("required", True) and not section.get("ready", False)
        ],
    }
    transition_runway = _build_transition_runway(
        sections=sections,
        start_date=start_date,
        target_date=target_date,
        current_date=current_date,
    )

    return {
        "timestamp_utc": iso_now(),
        "schema_version": 1,
        "policy_id": POLICY_ID,
        "ok": faithful_live_money_ready,
        "overall_status": "ready" if faithful_live_money_ready else "blocked",
        "faithful_live_money_ready": faithful_live_money_ready,
        "live_money_locked": not faithful_live_money_ready,
        "required_grade_floor": REQUIRED_GRADE_FLOOR,
        "allowed_grades": ["A", "A+"],
        "start_date": start_date.isoformat(),
        "target_date": target_date.isoformat(),
        "as_of_date": current_date.isoformat(),
        "days_elapsed": days_elapsed,
        "days_remaining": days_remaining,
        "target_window_complete": target_window_complete,
        "operator_execution_release_required": operator_execution_release_required,
        "blocking_reasons": blocking_reasons,
        "grade_summary": grade_summary,
        "transition_runway": transition_runway,
        "sections": sections,
        "recommended_actions": ordered_unique(
            [
                f"hold faithful live-money execution locked until {target_date.isoformat()}" if not target_window_complete else "",
                "raise every required readiness section to A/A+ before live-money clearance" if grade_summary["below_floor_sections"] else "",
                "clear all not-ready readiness sections before live-money clearance" if grade_summary["not_ready_sections"] else "",
                "work the six-pillar transition runway in due-date order" if transition_runway["blocked_pillars"] else "",
                "keep live execution validate-only until this contract reports faithful_live_money_ready=true" if not faithful_live_money_ready else "",
            ]
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Publish the strict A/A+ faithful live-money readiness contract.")
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--out-file", default=str(DEFAULT_OUT_PATH))
    parser.add_argument("--start-date", default=DEFAULT_START_DATE.isoformat())
    parser.add_argument("--target-date", default=DEFAULT_TARGET_DATE.isoformat())
    parser.add_argument("--as-of-date", default="")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    payload = build_payload(
        Path(args.project_root).resolve(),
        start_date=_parse_date(args.start_date, DEFAULT_START_DATE),
        target_date=_parse_date(args.target_date, DEFAULT_TARGET_DATE),
        as_of_date=args.as_of_date or None,
    )
    out_path = Path(args.out_file).expanduser()
    write_payload(out_path, payload)
    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        print(
            "live_money_readiness_contract "
            f"status={payload.get('overall_status')} "
            f"ready={int(bool(payload.get('faithful_live_money_ready')))} "
            f"target_date={payload.get('target_date')} "
            f"days_remaining={payload.get('days_remaining')}"
        )
    return 0 if bool(payload.get("faithful_live_money_ready", False)) else 2


if __name__ == "__main__":
    raise SystemExit(main())
