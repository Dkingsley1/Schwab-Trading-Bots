#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from scripts.ops.long_runtime_common import PROJECT_ROOT, iso_now, load_json, ordered_unique, write_payload
    from scripts.ops import production_flow_smoke, source_mutation_guard
else:
    from .long_runtime_common import PROJECT_ROOT, iso_now, load_json, ordered_unique, write_payload
    from . import production_flow_smoke, source_mutation_guard


DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "use_mode_compliance_policy_v1.json"
DEFAULT_OUT_PATH = PROJECT_ROOT / "governance" / "health" / "use_mode_compliance_guard_latest.json"
TRUE_VALUES = {"1", "true", "yes", "on", "enabled"}
READY_STATUSES = {"", "ready", "ok", "armed", "guarded_ready", "stable"}
BAD_STATUSES = {"blocked", "critical", "degraded", "failed", "missing", "needs_work", "stale", "warning"}
GRADE_RANK = {"F": 0, "D": 1, "C": 2, "B": 3, "A-": 4, "A": 5, "A+": 6, "A++": 6}


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


def _status(raw: Any) -> str:
    return str(raw or "").strip().lower()


def _grade(raw: Any) -> str:
    text = str(raw or "").strip().upper()
    return "A+" if text == "A++" else text


def _grade_at_least(raw: Any, floor: str) -> bool:
    return GRADE_RANK.get(_grade(raw), -1) >= GRADE_RANK.get(_grade(floor), 99)


def _payload_ready(payload: dict[str, Any], *, allow_guarded_recovery: bool = False) -> bool:
    if not payload:
        return False
    status = _status(payload.get("overall_status") or payload.get("status"))
    ok = payload.get("ok")
    if ok is not None:
        return bool(ok) and status not in BAD_STATUSES
    if allow_guarded_recovery and status in {"ready", "guarded_ready", "stable", "advisory", "degraded"}:
        return True
    return status in READY_STATUSES


def _env_bool(env: dict[str, str], name: str, default: bool = False) -> bool:
    value = str(env.get(name, "")).strip().lower()
    if not value:
        return default
    return value in TRUE_VALUES


def _health(project_root: Path, name: str) -> dict[str, Any]:
    payload = load_json(project_root / "governance" / "health" / name)
    return payload if isinstance(payload, dict) else {}


def _computed_control_payload(name: str, project_root: Path) -> dict[str, Any]:
    try:
        if name == "source_mutation_guard":
            return source_mutation_guard.build_payload(project_root)
        if name == "production_flow_smoke":
            return production_flow_smoke.build_payload(project_root)
    except Exception as exc:
        return {"overall_status": "blocked", "ok": False, "error": f"{type(exc).__name__}: {exc}"}
    return {}


def _managed_deferred_backlog_relief(health_fast: dict[str, Any], storage: dict[str, Any]) -> dict[str, Any]:
    guarded = _as_dict(_as_dict(health_fast.get("operational_readiness")).get("guarded_paper"))
    relief = _as_dict(guarded.get("storage_relief_contract"))
    if relief:
        active = bool(relief.get("active", False))
        return {
            "managed": active,
            "active": active,
            "status": relief.get("status") or ("managed_deferred_backlog_waiting_for_off_hours" if active else "inactive"),
            "core_pending_lines": _safe_int(relief.get("core_pending_lines"), 0),
            "support_pending_lines": _safe_int(relief.get("support_pending_lines"), 0),
            "deferred_pending_lines": _safe_int(relief.get("deferred_pending_lines"), 0),
            "total_pending_lines": _safe_int(relief.get("total_pending_lines"), 0),
            "backlog_drain_status": relief.get("backlog_drain_status"),
            "policy": "personal operator-grade paper soak can continue with deferred backlog managed for off-hours; live-money readiness remains blocked by raw backlog evidence",
        }
    backpressure = _as_dict(storage.get("backpressure"))
    storage_section = _as_dict(storage.get("storage"))
    backlog_status = _status(storage_section.get("backlog_drain_status") or storage.get("backlog_drain_status"))
    active = bool(
        _safe_int(backpressure.get("core_pending_lines"), 0) <= 5000
        and _safe_int(backpressure.get("support_pending_lines"), 0) <= 12000
        and _safe_int(backpressure.get("deferred_pending_lines"), 0) > 0
        and backlog_status in {"waiting_for_off_hours", "off_hours_scheduled", "market_hours_guard", "handoff_requested"}
    )
    return {
        "managed": active,
        "active": active,
        "status": "managed_deferred_backlog_waiting_for_off_hours" if active else "inactive",
        "core_pending_lines": _safe_int(backpressure.get("core_pending_lines"), 0),
        "support_pending_lines": _safe_int(backpressure.get("support_pending_lines"), 0),
        "deferred_pending_lines": _safe_int(backpressure.get("deferred_pending_lines"), 0),
        "total_pending_lines": _safe_int(backpressure.get("total_pending_lines"), 0),
        "backlog_drain_status": backlog_status,
        "policy": "personal operator-grade paper soak can continue with deferred backlog managed for off-hours; live-money readiness remains blocked by raw backlog evidence",
    }


def _criterion(
    criterion_id: str,
    title: str,
    ready: bool,
    blockers: list[str],
    evidence: dict[str, Any],
    *,
    required: bool = True,
) -> dict[str, Any]:
    clean_blockers = ordered_unique(str(item or "").strip() for item in blockers if str(item or "").strip())
    effective_ready = bool(ready and not clean_blockers)
    return {
        "criterion_id": criterion_id,
        "title": title,
        "required": bool(required),
        "ready": effective_ready,
        "status": "ready" if effective_ready else "blocked",
        "blockers": clean_blockers,
        "evidence": evidence,
    }


def _commercial_flag_snapshot(config: dict[str, Any], env: dict[str, str]) -> dict[str, bool]:
    env_contract = _as_dict(config.get("environment_contract"))
    env_names = [str(item) for item in _as_list(env_contract.get("commercial_trigger_envs")) if str(item).strip()]
    return {name: _env_bool(env, name, False) for name in env_names}


def _approval_snapshot(config: dict[str, Any], env: dict[str, str]) -> dict[str, bool]:
    approval_envs = _as_dict(_as_dict(config.get("environment_contract")).get("approval_envs"))
    return {key: _env_bool(env, str(name), False) for key, name in approval_envs.items() if str(key).strip() and str(name).strip()}


def _commercial_boundary(
    *,
    config: dict[str, Any],
    env: dict[str, str],
    use_mode: str,
    flags: dict[str, bool],
    approvals: dict[str, bool],
) -> dict[str, Any]:
    triggers = []
    review_blockers: list[str] = []
    hard_blockers: list[str] = []
    active_flag_names = sorted(name for name, enabled in flags.items() if enabled)
    commercial_intent = bool(use_mode != "personal" or active_flag_names)
    if use_mode not in {"personal", "commercial_software", "investment_advice", "broker_dealer_or_customer_execution", "commodity_advice_or_pool"}:
        review_blockers.append(f"unknown_use_mode={use_mode}")

    if commercial_intent:
        if not approvals.get("commercial_legal_review", False):
            review_blockers.append("commercial_legal_review_not_approved")
        if not approvals.get("commercial_compliance_review", False):
            review_blockers.append("commercial_compliance_review_not_approved")

    for raw_trigger in _as_list(config.get("commercial_boundary_triggers")):
        trigger = _as_dict(raw_trigger)
        env_names = [str(item) for item in _as_list(trigger.get("envs")) if str(item).strip()]
        active_envs = [name for name in env_names if flags.get(name, False)]
        if not active_envs:
            continue
        review_key = str(trigger.get("review_required") or "").strip()
        if review_key and not approvals.get(review_key, False):
            review_blockers.append(f"{review_key}_not_approved")
        if bool(trigger.get("hard_block", False)):
            hard_blockers.append(f"{trigger.get('trigger_id')}_hard_block")
        triggers.append(
            {
                "trigger_id": str(trigger.get("trigger_id") or ""),
                "active_envs": active_envs,
                "review_required": review_key,
                "review_approved": bool(approvals.get(review_key, False)) if review_key else False,
                "hard_block": bool(trigger.get("hard_block", False)),
                "summary": str(trigger.get("summary") or ""),
            }
        )

    if flags.get("CUSTOMER_FUNDS_ENABLED", False) or flags.get("CUSTODY_ENABLED", False):
        hard_blockers.append("customer_funds_or_custody_not_allowed_without_registered_reviewed_program")
    if flags.get("CUSTOMER_ORDER_EXECUTION_ENABLED", False) or flags.get("CUSTOMER_ACCOUNTS_ENABLED", False) or flags.get("COPY_TRADING_ENABLED", False):
        review_blockers.append("broker_dealer_customer_execution_review_required")

    blockers = ordered_unique([*hard_blockers, *review_blockers])
    clearance_status = "not_requested_personal_mode"
    ready_for_commercial_use = False
    if commercial_intent and blockers:
        clearance_status = "blocked_requires_compliance_review"
    elif commercial_intent:
        clearance_status = "review_evidence_present_operator_still_must_validate_before_public_use"
        ready_for_commercial_use = True

    return {
        "commercial_use_intent_detected": commercial_intent,
        "ready_for_commercial_use": ready_for_commercial_use,
        "commercial_clearance_status": clearance_status,
        "active_flags": active_flag_names,
        "active_triggers": triggers,
        "hard_blockers": ordered_unique(hard_blockers),
        "review_blockers": ordered_unique(review_blockers),
        "blockers": blockers,
        "approvals": approvals,
        "policy": "commercial_or_customer_facing_use_is_blocked_until_review_evidence_is_explicitly_present",
    }


def _operator_grade_personal_autonomy(
    project_root: Path,
    *,
    personal: dict[str, Any],
    commercial: dict[str, Any],
) -> dict[str, Any]:
    a_plus_packet = _health(project_root, "a_plus_operating_packet_latest.json")
    unattended_soak = _health(project_root, "unattended_soak_readiness_latest.json")
    autonomy = _health(project_root, "autonomy_control_plane_latest.json")
    storage_dr = _health(project_root, "storage_disaster_recovery_latest.json")
    blackstart = _health(project_root, "blackstart_recovery_latest.json")
    data_plane = _health(project_root, "data_plane_recovery_controller_latest.json")
    live_canary = _health(project_root, "live_canary_readiness_contract_latest.json")
    commercial_readiness = _health(project_root, "commercial_readiness_control_latest.json")
    security = _health(project_root, "security_audit_latest.json")
    secret = _health(project_root, "secret_scan_latest.json")
    redaction = _health(project_root, "telemetry_redaction_canary_latest.json")
    source_guard = _computed_control_payload("source_mutation_guard", project_root)
    production_flow = _computed_control_payload("production_flow_smoke", project_root)

    live_milestones = [row for row in _as_list(live_canary.get("live_money_canary_milestones")) if isinstance(row, dict)]
    m11 = next((row for row in live_milestones if str(row.get("milestone_id") or "") == "m11_use_mode_and_commercial_boundary"), {})
    data_plane_state = _status(data_plane.get("recovery_state"))
    data_plane_managed = bool(
        data_plane
        and _status(data_plane.get("overall_status") or data_plane.get("status")) not in {"blocked", "critical", "failed"}
        and data_plane_state not in {"blocked_backpressure", "blocked", "critical"}
    )
    commercial_mode_clean = bool(
        commercial_readiness
        and _status(commercial_readiness.get("overall_status") or commercial_readiness.get("status")) not in BAD_STATUSES
        and not bool(commercial_readiness.get("commercial_intent", False))
        and not bool(commercial_readiness.get("commercial_release_blocked", False))
        and not _as_list(commercial_readiness.get("blockers"))
    )
    secret_findings = _safe_int(secret.get("findings_count", _as_dict(secret.get("summary")).get("findings_count", 0)), 0)
    secret_clean = bool(secret and secret_findings == 0 and _status(secret.get("overall_status") or secret.get("status")) not in BAD_STATUSES)

    criteria = [
        _criterion(
            "base_personal_a_plus_ready",
            "Base Personal A+ Ready",
            bool(personal.get("perfect_personal_use_ready", False) and _grade_at_least(personal.get("grade"), "A+")),
            ["base_personal_a_plus_not_ready" if not bool(personal.get("perfect_personal_use_ready", False)) else ""],
            {"personal_grade": personal.get("grade"), "perfect_personal_use_ready": bool(personal.get("perfect_personal_use_ready", False))},
        ),
        _criterion(
            "a_plus_operating_packet_all_lanes",
            "A+ Operating Packet All Lanes",
            bool(a_plus_packet and a_plus_packet.get("a_plus_ready", False) and _safe_int(a_plus_packet.get("non_a_plus_lane_count"), 0) == 0),
            [
                "a_plus_operating_packet_missing" if not a_plus_packet else "",
                "a_plus_operating_packet_not_ready" if a_plus_packet and not bool(a_plus_packet.get("a_plus_ready", False)) else "",
                f"non_a_plus_lanes={_safe_int(a_plus_packet.get('non_a_plus_lane_count'), 0)}" if a_plus_packet and _safe_int(a_plus_packet.get("non_a_plus_lane_count"), 0) > 0 else "",
            ],
            {
                "overall_status": a_plus_packet.get("overall_status") or "missing",
                "overall_score": _safe_float(a_plus_packet.get("overall_score"), 0.0),
                "a_plus_ready": bool(a_plus_packet.get("a_plus_ready", False)),
                "a_plus_lane_count": _safe_int(a_plus_packet.get("a_plus_lane_count"), 0),
                "lane_count": _safe_int(a_plus_packet.get("lane_count"), 0),
                "non_a_plus_lane_count": _safe_int(a_plus_packet.get("non_a_plus_lane_count"), 0),
            },
        ),
        _criterion(
            "unattended_soak_green",
            "Unattended Soak Green",
            bool(unattended_soak and _payload_ready(unattended_soak) and unattended_soak.get("safe_to_leave_unattended", False) and _grade_at_least(unattended_soak.get("overall_grade") or unattended_soak.get("grade"), "A")),
            [
                "unattended_soak_missing" if not unattended_soak else "",
                "unattended_soak_not_ready" if unattended_soak and not _payload_ready(unattended_soak) else "",
                "safe_to_leave_unattended_not_true" if unattended_soak and not bool(unattended_soak.get("safe_to_leave_unattended", False)) else "",
            ],
            {
                "overall_status": unattended_soak.get("overall_status") or "missing",
                "overall_grade": unattended_soak.get("overall_grade") or unattended_soak.get("grade"),
                "safe_to_leave_unattended": bool(unattended_soak.get("safe_to_leave_unattended", False)),
                "blockers": _as_list(unattended_soak.get("blockers")),
            },
        ),
        _criterion(
            "source_mutation_guard_clean",
            "Source Mutation Guard Clean",
            bool(source_guard.get("ok", False)),
            [
                "source_mutation_guard_not_clean" if source_guard and not bool(source_guard.get("ok", False)) else "",
                "source_mutation_guard_missing" if not source_guard else "",
            ],
            {
                "overall_status": source_guard.get("overall_status") or "missing",
                "dirty_count": _safe_int(source_guard.get("dirty_count"), 0),
                "dirty_entries": _as_list(source_guard.get("dirty_entries"))[:12],
                "error": source_guard.get("error"),
            },
        ),
        _criterion(
            "production_flow_smoke_ready",
            "Production Flow Smoke Ready",
            bool(production_flow.get("ok", False)),
            [
                "production_flow_smoke_not_ready" if production_flow and not bool(production_flow.get("ok", False)) else "",
                "production_flow_smoke_missing" if not production_flow else "",
            ],
            {
                "overall_status": production_flow.get("overall_status") or "missing",
                "failed_checks": _as_list(production_flow.get("failed_checks")),
                "error": production_flow.get("error"),
            },
        ),
        _criterion(
            "autonomy_recovery_score",
            "Autonomy Recovery Score",
            bool(autonomy and _payload_ready(autonomy, allow_guarded_recovery=True) and _safe_float(autonomy.get("autonomy_score"), 0.0) >= 95.0),
            [
                "autonomy_control_plane_missing" if not autonomy else "",
                "autonomy_control_plane_not_ready" if autonomy and not _payload_ready(autonomy, allow_guarded_recovery=True) else "",
                "autonomy_score_below_operator_floor" if autonomy and _safe_float(autonomy.get("autonomy_score"), 0.0) < 95.0 else "",
            ],
            {
                "overall_status": autonomy.get("overall_status") or "missing",
                "autonomy_score": _safe_float(autonomy.get("autonomy_score"), 0.0),
            },
        ),
        _criterion(
            "disaster_recovery_blackstart_ready",
            "Disaster Recovery And Blackstart Ready",
            bool(_payload_ready(storage_dr) and _payload_ready(blackstart)),
            [
                "storage_disaster_recovery_not_ready" if not _payload_ready(storage_dr) else "",
                "blackstart_recovery_not_ready" if not _payload_ready(blackstart) else "",
            ],
            {
                "storage_disaster_recovery_status": storage_dr.get("overall_status") or storage_dr.get("status") or "missing",
                "blackstart_recovery_status": blackstart.get("overall_status") or blackstart.get("status") or "missing",
            },
        ),
        _criterion(
            "data_plane_recovery_managed",
            "Data Plane Recovery Managed",
            data_plane_managed,
            [
                "data_plane_recovery_missing" if not data_plane else "",
                "data_plane_recovery_not_managed" if data_plane and not data_plane_managed else "",
            ],
            {
                "overall_status": data_plane.get("overall_status") or data_plane.get("status") or "missing",
                "recovery_state": data_plane.get("recovery_state"),
                "write_failure_count": _safe_int(data_plane.get("write_failure_count"), 0),
                "queue_depth": _safe_int(data_plane.get("queue_depth"), 0),
            },
        ),
        _criterion(
            "live_money_boundaries_locked",
            "Live Money Boundaries Locked",
            bool(live_canary and not bool(live_canary.get("live_canary_money_ready", False)) and not bool(_as_dict(live_canary.get("authority_boundaries")).get("live_execution_authority", False)) and (not m11 or bool(m11.get("ready", False)))),
            [
                "live_canary_readiness_contract_missing" if not live_canary else "",
                "live_canary_money_ready_should_not_be_true_for_personal_operator_grade" if live_canary and bool(live_canary.get("live_canary_money_ready", False)) else "",
                "live_execution_authority_leaked" if bool(_as_dict(live_canary.get("authority_boundaries")).get("live_execution_authority", False)) else "",
                "use_mode_commercial_boundary_milestone_not_ready" if m11 and not bool(m11.get("ready", False)) else "",
            ],
            {
                "overall_status": live_canary.get("overall_status") or "missing",
                "live_canary_money_ready": bool(live_canary.get("live_canary_money_ready", False)),
                "blocked_milestones": _as_list(live_canary.get("blocked_milestones")),
                "m11_ready": bool(m11.get("ready", False)) if m11 else None,
            },
        ),
        _criterion(
            "commercial_personal_boundary_clean",
            "Commercial Personal Boundary Clean",
            bool(not commercial.get("commercial_use_intent_detected", False) and commercial_mode_clean),
            [
                "commercial_use_intent_detected" if commercial.get("commercial_use_intent_detected", False) else "",
                "commercial_readiness_not_personal_clean" if not commercial_mode_clean else "",
            ],
            {
                "use_mode_commercial_intent": bool(commercial.get("commercial_use_intent_detected", False)),
                "commercial_readiness_status": commercial_readiness.get("overall_status") or "missing",
                "commercial_product_mode": commercial_readiness.get("commercial_product_mode") or "personal_only",
                "commercial_readiness_grade": commercial_readiness.get("grade"),
            },
        ),
        _criterion(
            "security_privacy_runtime_clean",
            "Security And Privacy Runtime Clean",
            bool(_payload_ready(security) and secret_clean and _payload_ready(redaction)),
            [
                "security_audit_not_ready" if not _payload_ready(security) else "",
                "secret_scan_not_clean" if not secret_clean else "",
                "telemetry_redaction_not_ready" if not _payload_ready(redaction) else "",
            ],
            {
                "security_audit_status": security.get("overall_status") or security.get("status") or "missing",
                "secret_scan_findings": secret_findings if secret else None,
                "telemetry_redaction_status": redaction.get("overall_status") or redaction.get("status") or "missing",
            },
        ),
    ]

    ready_count = sum(1 for row in criteria if bool(row.get("ready", False)))
    score = round((ready_count / max(len(criteria), 1)) * 100.0, 2)
    ready = bool(ready_count == len(criteria))
    tier = "operator_grade_personal_autonomy" if ready else (
        "near_operator_grade_personal_autonomy" if score >= 90.0 else (
            "a_plus_personal_production" if bool(personal.get("perfect_personal_use_ready", False)) else "personal_needs_work"
        )
    )
    blockers = ordered_unique(
        f"{row['criterion_id']}:{blocker}"
        for row in criteria
        if not bool(row.get("ready", False))
        for blocker in _as_list(row.get("blockers"))
    )
    return {
        "ready": ready,
        "tier": tier,
        "score": score,
        "strength_grade": "A+ / operator-grade" if ready else "A+ / operator-grade-pending",
        "next_after_production": "operator_grade_personal_autonomy",
        "criterion_count": len(criteria),
        "ready_criterion_count": ready_count,
        "blockers": blockers,
        "criteria": criteria,
        "policy": "operator_grade_personal_autonomy_is_the_next_personal_use_bar_after_production_grade_and_never_grants_live_execution",
    }


def _personal_use_posture(project_root: Path, flags: dict[str, bool]) -> dict[str, Any]:
    health_fast = _health(project_root, "health_fast_latest.json")
    dashboard = _health(project_root, "runtime_gate_dashboard_latest.json")
    broker = _health(project_root, "broker_readiness_latest.json")
    auth_lease = _health(project_root, "auth_lease_manager_latest.json")
    schwab_auth = _health(project_root, "schwab_auth_supervisor_latest.json")
    process = _health(project_root, "process_watchdog_latest.json")
    paper_runtime = _health(project_root, "runtime_paper_regression_guard_latest.json")
    paper_truth = _health(project_root, "paper_execution_truth_layer_latest.json")
    paper_ramp = _health(project_root, "paper_400_ramp_latest.json")
    rollup = _health(project_root, "data_collection_observation_rollup_latest.json")
    storage = _health(project_root, "ingestion_storage_control_latest.json")
    live_sep = _health(project_root, "live_runtime_separation_control_latest.json")
    paper_profit = _health(project_root, "paper_profitability_control_latest.json")
    runtime_profit = _health(project_root, "paper_runtime_profitability_controls_latest.json")

    health_status = _status(health_fast.get("overall_status"))
    operational = _as_dict(health_fast.get("operational_readiness"))
    guarded_paper = _as_dict(operational.get("guarded_paper"))
    live_execution = _as_dict(operational.get("live_execution"))
    all_sleeves = _as_dict(_as_dict(health_fast.get("process_watchdog")).get("all_sleeves_effective_runtime"))
    if not all_sleeves:
        rows = [row for row in _as_list(process.get("status")) if isinstance(row, dict)]
        all_sleeves_row = next((row for row in rows if str(row.get("name") or "") == "all_sleeves"), {})
        all_sleeves = {
            "ok": bool(all_sleeves_row.get("process_live", all_sleeves_row.get("running", False)) or all_sleeves_row.get("effective_process_live", False)),
            "child_process_count": _safe_int(_as_dict(all_sleeves_row.get("child_fanout")).get("child_process_count"), _safe_int(all_sleeves_row.get("alt_running"), 0)),
            "status": _status(all_sleeves_row.get("status")) or "ready",
        }

    paper_runtime_status = _status(paper_runtime.get("overall_status") or paper_runtime.get("status"))
    paper_truth_status = _status(paper_truth.get("overall_status") or paper_truth.get("status"))
    paper_ramp_status = _status(paper_ramp.get("overall_status") or paper_ramp.get("stage") or paper_ramp.get("status"))
    paper_failed = [
        *[str(item) for item in _as_list(paper_runtime.get("failed_checks")) if str(item).strip()],
        *[str(item) for item in _as_list(paper_truth.get("failed_checks")) if str(item).strip()],
        *[str(item) for item in _as_list(paper_ramp.get("blockers")) if str(item).strip()],
    ]
    rollup_status = _status(rollup.get("overall_status") or rollup.get("status"))
    total_observations = _safe_int(rollup.get("total_observations"), _safe_int(_as_dict(health_fast.get("collection")).get("total_observations"), 0))
    bots_with_observations = _safe_int(
        rollup.get("effective_bots_with_observations", rollup.get("bots_with_observations")),
        _safe_int(_as_dict(health_fast.get("collection")).get("effective_bots_with_observations"), 0),
    )
    storage_status = _status(storage.get("overall_status") or storage.get("status") or storage.get("severity"))
    storage_pressure = _safe_float(storage.get("pressure_index"), _safe_float(_as_dict(health_fast.get("storage")).get("pressure_index"), 0.0))
    storage_relief = _managed_deferred_backlog_relief(health_fast, storage)
    auth_status = _status(auth_lease.get("overall_status") or auth_lease.get("status"))
    lease_state = _status(auth_lease.get("lease_state"))
    schwab_status = _status(schwab_auth.get("overall_status") or schwab_auth.get("status"))
    broker_ready = bool(broker.get("ready_for_open", broker.get("broker_ready", False)))
    broker_auth_ok = bool(broker.get("auth_ok", True))
    broker_network_ok = bool(broker.get("network_ok", True))
    live_sep_status = _status(live_sep.get("overall_status") or live_sep.get("status"))
    live_sep_read_only = bool(
        live_sep.get("live_orders_blocked", False)
        or live_sep.get("market_data_only", False)
        or _status(live_sep.get("live_execution_state")) == "blocked_read_only"
        or _status(live_execution.get("status")) == "blocked_read_only"
    )
    controlled_grade = _grade(
        runtime_profit.get("controlled_profitability_grade")
        or runtime_profit.get("grade")
        or paper_profit.get("controlled_profitability_grade")
    )
    raw_grade = _grade(
        paper_profit.get("raw_profitability_grade")
        or runtime_profit.get("raw_profitability_grade")
        or paper_profit.get("financial_profitability_grade")
        or paper_profit.get("grade")
    )
    raw_status = _status(paper_profit.get("overall_status") or runtime_profit.get("overall_status") or paper_profit.get("status") or runtime_profit.get("status"))

    commercial_flags_clear = not any(flags.values())
    criteria = [
        _criterion(
            "health_fast_strict_clear",
            "Fast Health Strict Clear",
            bool(health_fast and health_status == "ready" and health_fast.get("strict_all_clear", False)),
            [
                "health_fast_missing" if not health_fast else "",
                f"health_fast_status={health_status or 'unknown'}" if health_fast and health_status != "ready" else "",
                "strict_all_clear_not_true" if health_fast and not bool(health_fast.get("strict_all_clear", False)) else "",
            ],
            {"health_fast_status": health_status or "missing", "strict_all_clear": bool(health_fast.get("strict_all_clear", False))},
        ),
        _criterion(
            "guarded_paper_ready",
            "Guarded Paper Ready",
            bool(guarded_paper and guarded_paper.get("ok", False) and _status(guarded_paper.get("status")) == "ready"),
            [
                "guarded_paper_missing" if not guarded_paper else "",
                "guarded_paper_not_ready" if guarded_paper and _status(guarded_paper.get("status")) != "ready" else "",
                *[f"guarded_paper_blocker={item}" for item in _as_list(guarded_paper.get("blockers"))],
            ],
            {"status": guarded_paper.get("status"), "ok": bool(guarded_paper.get("ok", False)), "blockers": _as_list(guarded_paper.get("blockers"))},
        ),
        _criterion(
            "auth_and_broker_ready",
            "Auth And Broker Ready",
            bool(
                auth_lease
                and auth_status in READY_STATUSES
                and lease_state not in {"critical", "expired", "blocked", "warning"}
                and (not schwab_auth or schwab_status not in BAD_STATUSES)
                and (not broker or (broker_ready and broker_auth_ok and broker_network_ok))
            ),
            [
                "auth_lease_missing" if not auth_lease else "",
                f"auth_status={auth_status}" if auth_lease and auth_status not in READY_STATUSES else "",
                f"lease_state={lease_state}" if lease_state in {"critical", "expired", "blocked", "warning"} else "",
                "broker_not_ready" if broker and not broker_ready else "",
                "broker_auth_or_network_not_ok" if broker and (not broker_auth_ok or not broker_network_ok) else "",
                f"schwab_auth_status={schwab_status}" if schwab_auth and schwab_status in BAD_STATUSES else "",
            ],
            {
                "auth_status": auth_status or "missing",
                "lease_state": lease_state,
                "broker_ready": broker_ready if broker else None,
                "broker_auth_ok": broker_auth_ok if broker else None,
                "broker_network_ok": broker_network_ok if broker else None,
                "schwab_auth_status": schwab_status or "missing",
            },
        ),
        _criterion(
            "all_sleeves_effective_runtime",
            "All Sleeves Effective Runtime",
            bool(all_sleeves and all_sleeves.get("ok", False) and _safe_int(all_sleeves.get("child_process_count"), 0) > 0),
            [
                "all_sleeves_effective_runtime_missing" if not all_sleeves else "",
                "all_sleeves_not_effective" if all_sleeves and not bool(all_sleeves.get("ok", False)) else "",
                "no_sleeve_child_processes" if all_sleeves and _safe_int(all_sleeves.get("child_process_count"), 0) <= 0 else "",
            ],
            {
                "status": all_sleeves.get("status"),
                "ok": bool(all_sleeves.get("ok", False)),
                "child_process_count": _safe_int(all_sleeves.get("child_process_count"), 0),
            },
        ),
        _criterion(
            "paper_execution_continuity",
            "Paper Execution Continuity",
            bool(
                (paper_runtime or paper_truth or paper_ramp)
                and paper_runtime_status not in BAD_STATUSES
                and paper_truth_status not in BAD_STATUSES
                and paper_ramp_status not in BAD_STATUSES
                and not paper_failed
            ),
            [
                "paper_execution_artifacts_missing" if not (paper_runtime or paper_truth or paper_ramp) else "",
                f"runtime_paper_status={paper_runtime_status}" if paper_runtime_status in BAD_STATUSES else "",
                f"paper_truth_status={paper_truth_status}" if paper_truth_status in BAD_STATUSES else "",
                f"paper_ramp_status={paper_ramp_status}" if paper_ramp_status in BAD_STATUSES else "",
                *[f"paper_execution_issue={item}" for item in paper_failed],
            ],
            {
                "runtime_paper_status": paper_runtime_status or "missing",
                "paper_truth_status": paper_truth_status or "missing",
                "paper_ramp_status": paper_ramp_status or "missing",
                "issue_count": len(paper_failed),
            },
        ),
        _criterion(
            "data_collection_nonzero",
            "Data Collection Nonzero",
            bool((rollup or health_fast) and rollup_status not in {"blocked", "critical", "failed"} and total_observations > 0 and bots_with_observations > 0),
            [
                "data_collection_rollup_missing" if not rollup and not health_fast else "",
                f"collection_status={rollup_status}" if rollup_status in {"blocked", "critical", "failed"} else "",
                "zero_total_observations" if total_observations <= 0 else "",
                "zero_bots_with_observations" if bots_with_observations <= 0 else "",
            ],
            {
                "rollup_status": rollup_status or _status(_as_dict(health_fast.get("collection")).get("overall_status")) or "missing",
                "bots_with_observations": bots_with_observations,
                "total_observations": total_observations,
            },
        ),
        _criterion(
            "storage_pressure_clean",
            "Storage Pressure Clean",
            bool(
                (storage or health_fast)
                and (
                    (storage_status not in {"blocked", "critical", "failed"} and storage_pressure <= 0.2)
                    or bool(storage_relief.get("managed", False))
                )
            ),
            [
                "storage_artifact_missing" if not storage and not health_fast else "",
                f"storage_status={storage_status}" if storage_status in {"blocked", "critical", "failed"} and not bool(storage_relief.get("managed", False)) else "",
                "storage_pressure_above_personal_use_floor" if storage_pressure > 0.2 and not bool(storage_relief.get("managed", False)) else "",
            ],
            {
                "storage_status": storage_status or "missing",
                "pressure_index": storage_pressure,
                "managed_deferred_backlog_relief": storage_relief,
            },
        ),
        _criterion(
            "live_execution_read_only",
            "Live Execution Read Only",
            bool(live_sep_read_only),
            ["live_execution_not_confirmed_read_only" if not live_sep_read_only else ""],
            {
                "live_runtime_separation_status": live_sep_status or "missing",
                "health_fast_live_execution_status": live_execution.get("status"),
                "read_only_confirmed": live_sep_read_only,
            },
        ),
        _criterion(
            "no_customer_or_commercial_flags",
            "No Customer Or Commercial Flags",
            commercial_flags_clear,
            [f"commercial_flag_active={name}" for name, enabled in sorted(flags.items()) if enabled],
            {"active_flags": sorted(name for name, enabled in flags.items() if enabled)},
        ),
        _criterion(
            "profitability_evidence_labeled",
            "Profitability Evidence Labeled",
            bool((paper_profit or runtime_profit) and (controlled_grade or raw_grade or raw_status) and not (raw_grade == "D" and not controlled_grade)),
            [
                "profitability_artifact_missing" if not (paper_profit or runtime_profit) else "",
                "raw_d_without_controlled_profitability_context" if raw_grade == "D" and not controlled_grade else "",
            ],
            {
                "controlled_profitability_grade": controlled_grade or "unknown",
                "raw_profitability_grade": raw_grade or "unknown",
                "raw_profitability_live_money_ready": _grade_at_least(raw_grade, "A"),
                "raw_profitability_policy": "raw_profitability_is_live_canary_evidence_not_a_guarded_paper_runtime_blocker",
            },
        ),
    ]

    required = [row for row in criteria if bool(row.get("required", True))]
    ready_required = [row for row in required if bool(row.get("ready", False))]
    required_ratio = len(ready_required) / max(len(required), 1)
    grade = "A+"
    if required_ratio < 1.0:
        grade = "A" if required_ratio >= 0.9 else "B" if required_ratio >= 0.8 else "C" if required_ratio >= 0.7 else "D"
    perfect_ready = bool(required_ratio == 1.0)
    personal = {
        "perfect_personal_use_ready": perfect_ready,
        "personal_soak_ready": perfect_ready,
        "personal_live_money_ready": False,
        "grade": grade,
        "required_criterion_count": len(required),
        "ready_required_criterion_count": len(ready_required),
        "required_ready_ratio": round(required_ratio, 4),
        "criteria": criteria,
        "blockers": ordered_unique(
            f"{row['criterion_id']}:{blocker}"
            for row in required
            if not bool(row.get("ready", False))
            for blocker in _as_list(row.get("blockers"))
        ),
        "dashboard_context": {
            "dashboard_status": _status(_as_dict(dashboard.get("overall")).get("status") or dashboard.get("overall_status") or dashboard.get("status")) or "missing",
            "dashboard_ok": bool(dashboard.get("ok", _as_dict(dashboard.get("overall")).get("ok", False))),
        },
        "policy": "personal_use_perfection_means_clean_guarded_paper_data_collection_and_read_only_boundaries_not_live_money_or_commercial_clearance",
    }
    return personal


def build_payload(
    project_root: Path = PROJECT_ROOT,
    *,
    config_path: Path = DEFAULT_CONFIG_PATH,
    env: dict[str, str] | None = None,
) -> dict[str, Any]:
    config = load_json(config_path)
    if not config:
        config = load_json(DEFAULT_CONFIG_PATH)
    if not config:
        config = {
            "schema_version": 1,
            "policy_id": "use_mode_compliance_policy_v1",
            "default_use_mode": "personal",
            "environment_contract": {"commercial_trigger_envs": []},
            "commercial_boundary_triggers": [],
            "authority_boundaries": {"does_not_enable_live_execution": True},
        }
    runtime_env = dict(os.environ if env is None else env)
    use_mode_env = str(_as_dict(config.get("environment_contract")).get("use_mode_env") or "SYSTEM_USE_MODE")
    use_mode = str(runtime_env.get(use_mode_env) or config.get("default_use_mode") or "personal").strip().lower()
    flags = _commercial_flag_snapshot(config, runtime_env)
    approvals = _approval_snapshot(config, runtime_env)
    commercial = _commercial_boundary(config=config, env=runtime_env, use_mode=use_mode, flags=flags, approvals=approvals)
    personal = _personal_use_posture(project_root, flags)
    personal["operator_grade_personal_autonomy"] = _operator_grade_personal_autonomy(
        project_root,
        personal=personal,
        commercial=commercial,
    )
    active_commercial_block = bool(commercial["commercial_use_intent_detected"] and commercial["blockers"])
    overall_status = "blocked" if active_commercial_block else ("ready" if personal["perfect_personal_use_ready"] else "needs_work")
    authority_boundaries = _as_dict(config.get("authority_boundaries"))
    return {
        "timestamp_utc": iso_now(),
        "schema_version": 1,
        "policy_id": str(config.get("policy_id") or "use_mode_compliance_policy_v1"),
        "source": "use_mode_compliance_guard",
        "ok": overall_status in {"ready", "needs_work"},
        "overall_status": overall_status,
        "use_mode": use_mode,
        "personal_use": personal,
        "commercial_use": commercial,
        "authority_boundaries": {
            **authority_boundaries,
            "does_not_enable_live_execution": True,
            "live_execution_authority": False,
            "customer_funds_allowed": False,
            "customer_accounts_allowed": False,
            "customer_order_execution_allowed": False,
            "custody_allowed": False,
            "copy_trading_allowed": False,
            "raw_profitability_is_not_live_money_proof": True,
        },
        "blockers": ordered_unique(
            [
                *[f"personal_use:{item}" for item in _as_list(personal.get("blockers"))],
                *[f"commercial_use:{item}" for item in _as_list(commercial.get("blockers"))],
            ]
        ),
        "regulatory_source_references": _as_list(config.get("regulatory_source_references")),
        "recommended_actions": ordered_unique(
            [
                "keep live orders disabled; this guard never grants live execution authority",
                "treat personal A+ as guarded paper/data-collection readiness only",
                "resolve personal-use blockers before calling the system unattended" if not personal["perfect_personal_use_ready"] else "",
                "clear operator-grade personal autonomy blockers before treating personal use as beyond-production unattended"
                if not _as_dict(personal.get("operator_grade_personal_autonomy")).get("ready", False)
                else "",
                "complete commercial legal and compliance review before marketing, paid signals, customer accounts, custody, copy trading, or customer order execution"
                if commercial["commercial_use_intent_detected"]
                else "",
                "keep raw profitability separated from controlled profitability until raw grade clears the live-canary floor",
            ]
        ),
        "artifact_paths": {"json": str(DEFAULT_OUT_PATH), "config": str(config_path)},
        "control_contract": {
            "personal_use_default": True,
            "commercial_or_customer_facing_use_requires_explicit_review_evidence": True,
            "commercial_block_does_not_toggle_live_orders": True,
            "live_money_canary_must_consume_this_guard": True,
            "not_legal_advice": True,
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate personal-use readiness and commercial-use compliance boundaries.")
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--out-file", type=Path, default=DEFAULT_OUT_PATH)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    payload = build_payload(args.project_root.resolve(), config_path=args.config)
    write_payload(args.out_file, payload)
    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        print(
            "use_mode_compliance_guard "
            f"status={payload.get('overall_status')} "
            f"use_mode={payload.get('use_mode')} "
            f"personal_grade={_as_dict(payload.get('personal_use')).get('grade')} "
            f"commercial_intent={int(bool(_as_dict(payload.get('commercial_use')).get('commercial_use_intent_detected')))} "
            f"blockers={len(_as_list(payload.get('blockers')))}"
        )
    return 0 if payload.get("overall_status") in {"ready", "needs_work"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
