#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from scripts.ops.long_runtime_common import PROJECT_ROOT, iso_now, load_json, ordered_unique, payload_age_minutes, write_payload
    from scripts.ops import commercial_readiness_control, production_flow_smoke, production_readiness_control, source_mutation_guard, use_mode_compliance_guard
else:
    from .long_runtime_common import PROJECT_ROOT, iso_now, load_json, ordered_unique, payload_age_minutes, write_payload
    from . import commercial_readiness_control, production_flow_smoke, production_readiness_control, source_mutation_guard, use_mode_compliance_guard


DEFAULT_OUT_PATH = PROJECT_ROOT / "governance" / "health" / "live_canary_readiness_contract_latest.json"
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "live_canary_readiness_contract.json"
POLICY_ID = "production_hardening_live_canary_bar_v1"
BAD_STATUSES = {"blocked", "critical", "degraded", "failed", "missing", "needs_work", "stale", "warning"}
GRADE_RANK = {"F": 0, "D": 1, "C": 2, "B": 3, "A-": 4, "A": 5, "A+": 6, "A++": 6}
DEFAULT_CANARY_MILESTONES: tuple[dict[str, Any], ...] = (
    {
        "milestone_id": "m01_continuous_soak_no_hard_blockers",
        "title": "30-Day Soak With No Hard Blockers",
        "owner": "unattended_soak_readiness",
        "required": True,
        "description": "Guarded paper stays strict-clear for the full soak window before live-money canary consideration.",
    },
    {
        "milestone_id": "m02_live_like_paper_execution",
        "title": "Live-Like Paper Execution Evidence",
        "owner": "paper_execution_truth_layer",
        "required": True,
        "description": "Paper PnL is backed by execution truth, continuity, slippage, latency, spread, fees, and rejected-order handling.",
    },
    {
        "milestone_id": "m03_pre_trade_risk_controls",
        "title": "Pre-Trade Risk Controls Proven",
        "owner": "risk_service_boundary",
        "required": True,
        "description": "Kill switch, pre-trade approval, risk budgets, and portfolio caps are fresh and enforceable.",
    },
    {
        "milestone_id": "m04_autonomous_recovery_without_operator",
        "title": "Autonomous Recovery Without Operator",
        "owner": "process_watchdog",
        "required": True,
        "description": "Auth, storage, providers, and sleeve fanout can repair, quarantine, or degrade without manual babysitting.",
    },
    {
        "milestone_id": "m05_no_fake_green_dashboard_semantics",
        "title": "No Hidden Fake-Green Dashboard State",
        "owner": "health_fast",
        "required": True,
        "description": "Dashboards separate ready, managed advisory, true blocker, and live-money blocker states.",
    },
    {
        "milestone_id": "m06_explained_loss_attribution",
        "title": "Explained Loss Attribution",
        "owner": "paper_profitability_control",
        "required": True,
        "description": "Losses must trace to expected strategy behavior, not stale data, duplicate exposure, sizing, or fill modeling bugs.",
    },
    {
        "milestone_id": "m07_broker_order_reconciliation",
        "title": "Broker And Order Reconciliation",
        "owner": "broker_truth_reconciliation",
        "required": True,
        "description": "Every intended, paper, rejected, canceled, filled, and position state reconciles with no mystery exposure.",
    },
    {
        "milestone_id": "m08_microscopic_canary_plan",
        "title": "Microscopic Canary Plan",
        "owner": "live_canary_control",
        "required": True,
        "description": "The first live canary is capped at a tiny weight with rollback, no leverage, and supervised release.",
    },
    {
        "milestone_id": "m09_explainable_trade_permission",
        "title": "Explainable Trade Permission",
        "owner": "live_canary_readiness_contract",
        "required": True,
        "description": "The system can explain why a sleeve is allowed to trade before any live-money canary is considered.",
    },
    {
        "milestone_id": "m10_live_money_production_bar",
        "title": "Live-Money Production Bar",
        "owner": "production_readiness_control",
        "required": True,
        "description": "External supervision, immutable evidence, security, recovery, release, SLO, and read-only firewall controls are all production-ready.",
    },
    {
        "milestone_id": "m11_use_mode_and_commercial_boundary",
        "title": "Use-Mode And Commercial Boundary",
        "owner": "use_mode_compliance_guard",
        "required": True,
        "description": "Personal use, commercial use, customer-facing use, marketing, advice, custody, and broker/customer execution boundaries are explicit before live-money canary consideration.",
    },
)


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


def _gate(gate_id: str, title: str, ready: bool, blockers: list[str], evidence: dict[str, Any], *, owner: str) -> dict[str, Any]:
    clean_blockers = ordered_unique(str(item or "").strip() for item in blockers if str(item or "").strip())
    return {
        "gate_id": gate_id,
        "title": title,
        "ready": bool(ready and not clean_blockers),
        "status": "ready" if ready and not clean_blockers else "blocked",
        "blockers": clean_blockers,
        "owner": owner,
        "evidence": evidence,
    }


def _milestone(
    definition: dict[str, Any],
    *,
    ready: bool,
    blockers: list[str],
    evidence: dict[str, Any],
) -> dict[str, Any]:
    clean_blockers = ordered_unique(str(item or "").strip() for item in blockers if str(item or "").strip())
    required = bool(definition.get("required", True))
    return {
        "milestone_id": str(definition.get("milestone_id") or ""),
        "title": str(definition.get("title") or ""),
        "required": required,
        "ready": bool(ready and not clean_blockers),
        "status": "ready" if ready and not clean_blockers else "blocked",
        "blockers": clean_blockers,
        "owner": str(definition.get("owner") or ""),
        "description": str(definition.get("description") or ""),
        "evidence": evidence,
    }


def _contract_section(contract: dict[str, Any], section_id: str) -> dict[str, Any]:
    for row in _as_list(contract.get("sections")):
        section = _as_dict(row)
        if str(section.get("section_id") or "") == section_id:
            return section
    return {}


def _section_grade_ready(contract: dict[str, Any], section_id: str, *, floor: str = "A") -> bool:
    section = _contract_section(contract, section_id)
    return bool(section and section.get("ready", False) and _grade_at_least(section.get("grade"), floor))


def _build_live_money_canary_milestones(
    *,
    config: dict[str, Any],
    gates: list[dict[str, Any]],
    sustained: dict[str, Any],
    health_fast: dict[str, Any],
    paper_truth: dict[str, Any],
    live_money_contract: dict[str, Any],
    live_canary_control: dict[str, Any],
    production_readiness: dict[str, Any],
    use_mode_compliance: dict[str, Any],
    commercial_readiness: dict[str, Any],
) -> list[dict[str, Any]]:
    definitions = _as_list(config.get("live_money_canary_milestones")) or [dict(row) for row in DEFAULT_CANARY_MILESTONES]
    by_id = {str(gate.get("gate_id") or ""): gate for gate in gates}
    min_soak_hours = _safe_float(config.get("live_money_canary_min_soak_hours"), 720.0)
    max_initial_weight = _safe_float(config.get("max_initial_live_canary_weight"), 0.04)
    health_status = _status(health_fast.get("overall_status"))
    strict_clear = bool(health_fast.get("strict_all_clear", False))
    guarded_paper = _as_dict(_as_dict(health_fast.get("operational_readiness")).get("guarded_paper"))
    platform_layers = [
        "platform_intelligence",
        "platform_brain_v4",
        "platform_brain_v5",
        "platform_stabilization_quality",
        "system_architecture_hardening",
    ]
    platform_ready = bool(
        health_fast
        and all(_status(_as_dict(health_fast.get(name)).get("overall_status")) == "ready" for name in platform_layers)
    )
    watchdog = _as_dict(_as_dict(health_fast.get("process_watchdog")).get("all_sleeves_effective_runtime"))
    collector_repair = _as_dict(_as_dict(health_fast.get("operational_readiness")).get("collector_repair"))
    platform_repair = _as_dict(_as_dict(health_fast.get("operational_readiness")).get("platform_repair"))
    broker_truth = _as_dict(_as_dict(paper_truth.get("gates")).get("paper_broker_truth_reconciliation"))
    live_money_risk_ready = _section_grade_ready(live_money_contract, "risk_controls")
    target_weight = _safe_float(live_canary_control.get("target_canary_weight"), 0.0)
    applied_weight = _safe_float(live_canary_control.get("applied_canary_weight"), target_weight)
    canary_weight = applied_weight if applied_weight > 0 else target_weight
    production_bar_ready = bool(production_readiness.get("live_money_production_bar_ready", False))
    canary_consideration_ready = bool(production_readiness.get("live_money_canary_consideration_ready", False))
    production_domains_blocked = _safe_int(production_readiness.get("blocked_domain_count"), 0)
    use_commercial = _as_dict(use_mode_compliance.get("commercial_use"))
    use_authority = _as_dict(use_mode_compliance.get("authority_boundaries"))
    commercial_authority = _as_dict(commercial_readiness.get("authority_boundaries"))
    commercial_blockers = [str(item) for item in _as_list(use_commercial.get("blockers")) if str(item).strip()]
    commercial_intent = bool(use_commercial.get("commercial_use_intent_detected", False))
    use_guard_status = _status(use_mode_compliance.get("overall_status") or use_mode_compliance.get("status"))
    commercial_guard_status = _status(commercial_readiness.get("overall_status") or commercial_readiness.get("status"))
    commercial_framework_blockers = [str(item) for item in _as_list(commercial_readiness.get("blockers")) if str(item).strip()]
    commercial_framework_intent = bool(commercial_readiness.get("commercial_intent", False))
    commercial_framework_blocked = bool(
        commercial_readiness
        and (
            commercial_guard_status in BAD_STATUSES
            or bool(commercial_readiness.get("commercial_release_blocked", False))
            or commercial_framework_blockers
            or bool(commercial_authority.get("live_execution_authority", False))
        )
    )
    use_boundary_ready = bool(
        use_mode_compliance
        and use_guard_status not in BAD_STATUSES
        and not commercial_blockers
        and not bool(use_authority.get("live_execution_authority", False))
        and bool(use_authority.get("does_not_enable_live_execution", True))
        and (not commercial_readiness or not commercial_framework_blocked)
    )

    prior_milestones: dict[str, bool] = {}
    milestones: list[dict[str, Any]] = []
    for raw_definition in definitions:
        definition = _as_dict(raw_definition)
        milestone_id = str(definition.get("milestone_id") or "")
        if milestone_id == "m01_continuous_soak_no_hard_blockers":
            ready = bool(
                health_fast
                and health_status == "ready"
                and strict_clear
                and _status(guarded_paper.get("status")) == "ready"
                and _safe_float(sustained.get("sustained_ready_hours"), 0.0) >= min_soak_hours
            )
            blockers = [
                "health_fast_missing" if not health_fast else "",
                "strict_all_clear_not_true" if health_fast and not strict_clear else "",
                "guarded_paper_not_ready" if guarded_paper and _status(guarded_paper.get("status")) != "ready" else "",
                f"continuous_soak_below_{int(min_soak_hours)}h"
                if _safe_float(sustained.get("sustained_ready_hours"), 0.0) < min_soak_hours
                else "",
            ]
            evidence = {
                "health_status": health_status or "unknown",
                "strict_all_clear": strict_clear,
                "guarded_paper_status": guarded_paper.get("status"),
                "required_soak_hours": min_soak_hours,
                "sustained_ready_hours": sustained.get("sustained_ready_hours"),
                "continuous_all_gates_ready_since_utc": sustained.get("continuous_all_gates_ready_since_utc"),
            }
        elif milestone_id == "m02_live_like_paper_execution":
            raw_ready = bool(by_id.get("raw_profitability_posture", {}).get("ready", False))
            paper_ready = bool(by_id.get("sleeve_paper_trading_continuity", {}).get("ready", False))
            ready = bool(raw_ready and paper_ready)
            blockers = [
                "raw_profitability_posture_not_ready" if not raw_ready else "",
                "sleeve_paper_trading_continuity_not_ready" if not paper_ready else "",
            ]
            evidence = {
                "raw_profitability_posture": by_id.get("raw_profitability_posture", {}),
                "sleeve_paper_trading_continuity": by_id.get("sleeve_paper_trading_continuity", {}),
            }
        elif milestone_id == "m03_pre_trade_risk_controls":
            ready = bool(live_money_risk_ready)
            blockers = ["live_money_risk_controls_not_A_ready" if not ready else ""]
            evidence = {
                "live_money_contract_present": bool(live_money_contract),
                "risk_controls": _contract_section(live_money_contract, "risk_controls"),
            }
        elif milestone_id == "m04_autonomous_recovery_without_operator":
            auth_ready = bool(by_id.get("auth_token_continuity", {}).get("ready", False))
            storage_ready = bool(by_id.get("storage_pressure_clean", {}).get("ready", False))
            process_ready = bool(
                health_fast
                and watchdog.get("ok", False)
                and _status(collector_repair.get("status")) == "ready"
                and _status(platform_repair.get("status")) == "ready"
            )
            ready = bool(auth_ready and storage_ready and process_ready)
            blockers = [
                "auth_token_continuity_not_ready" if not auth_ready else "",
                "storage_pressure_clean_not_ready" if not storage_ready else "",
                "process_or_repair_plane_not_ready" if not process_ready else "",
            ]
            evidence = {
                "auth_token_continuity": by_id.get("auth_token_continuity", {}),
                "storage_pressure_clean": by_id.get("storage_pressure_clean", {}),
                "all_sleeves_effective_runtime": watchdog,
                "collector_repair": collector_repair,
                "platform_repair": platform_repair,
            }
        elif milestone_id == "m05_no_fake_green_dashboard_semantics":
            source_ready = bool(by_id.get("runtime_source_mutation_guard", {}).get("ready", False))
            ci_ready = bool(by_id.get("ci_production_guardrails", {}).get("ready", False))
            freshness_ready = bool(by_id.get("promotion_paper_gate_freshness", {}).get("ready", False))
            ready = bool(health_fast and strict_clear and platform_ready and source_ready and ci_ready and freshness_ready)
            blockers = [
                "health_fast_missing" if not health_fast else "",
                "strict_all_clear_not_true" if health_fast and not strict_clear else "",
                "platform_layers_not_ready" if health_fast and not platform_ready else "",
                "runtime_source_mutation_guard_not_ready" if not source_ready else "",
                "ci_production_guardrails_not_ready" if not ci_ready else "",
                "promotion_paper_gate_freshness_not_ready" if not freshness_ready else "",
            ]
            evidence = {
                "strict_all_clear": strict_clear,
                "platform_ready": platform_ready,
                "runtime_source_mutation_guard": by_id.get("runtime_source_mutation_guard", {}),
                "ci_production_guardrails": by_id.get("ci_production_guardrails", {}),
                "promotion_paper_gate_freshness": by_id.get("promotion_paper_gate_freshness", {}),
            }
        elif milestone_id == "m06_explained_loss_attribution":
            raw_ready = bool(by_id.get("raw_profitability_posture", {}).get("ready", False))
            paper_ready = bool(by_id.get("sleeve_paper_trading_continuity", {}).get("ready", False))
            ready = bool(raw_ready and paper_ready)
            blockers = [
                "raw_profitability_not_A_ready" if not raw_ready else "",
                "paper_truth_continuity_not_ready" if not paper_ready else "",
            ]
            evidence = {
                "raw_profitability_posture": by_id.get("raw_profitability_posture", {}),
                "paper_truth_failed_checks": _as_list(paper_truth.get("failed_checks")),
            }
        elif milestone_id == "m07_broker_order_reconciliation":
            broker_truth_ready = bool(broker_truth.get("ok", False)) if broker_truth else bool(by_id.get("sleeve_paper_trading_continuity", {}).get("ready", False))
            ready = bool(by_id.get("sleeve_paper_trading_continuity", {}).get("ready", False) and broker_truth_ready)
            blockers = [
                "sleeve_paper_trading_continuity_not_ready" if not by_id.get("sleeve_paper_trading_continuity", {}).get("ready", False) else "",
                "paper_broker_truth_reconciliation_not_ready" if not broker_truth_ready else "",
            ]
            evidence = {
                "paper_broker_truth_reconciliation": broker_truth,
                "sleeve_paper_trading_continuity": by_id.get("sleeve_paper_trading_continuity", {}),
            }
        elif milestone_id == "m08_microscopic_canary_plan":
            ready = bool(live_canary_control and canary_weight > 0.0 and canary_weight <= max_initial_weight)
            blockers = [
                "live_canary_control_missing" if not live_canary_control else "",
                "canary_weight_not_positive" if live_canary_control and canary_weight <= 0.0 else "",
                f"initial_canary_weight_above_{max_initial_weight:.4f}" if canary_weight > max_initial_weight else "",
            ]
            evidence = {
                "recommended_mode": live_canary_control.get("recommended_mode"),
                "target_canary_weight": target_weight,
                "applied_canary_weight": applied_weight,
                "effective_canary_weight": canary_weight,
                "max_initial_live_canary_weight": max_initial_weight,
                "canary_weight_ok": bool(canary_weight > 0.0 and canary_weight <= max_initial_weight),
            }
        elif milestone_id == "m09_explainable_trade_permission":
            previous_ready = all(prior_milestones.values()) if prior_milestones else False
            gates_ready = all(bool(gate.get("ready", False)) for gate in gates)
            ready = bool(previous_ready and gates_ready and canary_consideration_ready)
            blockers = [
                "prior_live_money_canary_milestones_not_ready" if not previous_ready else "",
                "hard_live_canary_gates_not_ready" if not gates_ready else "",
                "live_money_production_bar_not_ready" if not canary_consideration_ready else "",
            ]
            evidence = {
                "prior_milestones_ready": previous_ready,
                "all_hard_gates_ready": gates_ready,
                "ready_gate_count": sum(1 for gate in gates if gate.get("ready", False)),
                "gate_count": len(gates),
                "live_money_production_bar_ready": production_bar_ready,
                "live_money_canary_consideration_ready": canary_consideration_ready,
                "production_readiness_blocked_domain_count": production_domains_blocked,
            }
        elif milestone_id == "m10_live_money_production_bar":
            ready = bool(production_bar_ready and canary_consideration_ready)
            blockers = [
                "production_readiness_control_missing" if not production_readiness else "",
                "live_money_production_bar_not_ready" if production_readiness and not production_bar_ready else "",
                "live_money_canary_consideration_not_ready" if production_readiness and not canary_consideration_ready else "",
                "production_readiness_domains_blocked" if production_domains_blocked > 0 else "",
            ]
            evidence = {
                "production_readiness_status": production_readiness.get("overall_status"),
                "live_money_production_bar_ready": production_bar_ready,
                "live_money_canary_consideration_ready": canary_consideration_ready,
                "blocked_domain_count": production_domains_blocked,
                "domain_count": production_readiness.get("domain_count"),
                "ready_domain_count": production_readiness.get("ready_domain_count"),
                "blockers": _as_list(production_readiness.get("blockers")),
            }
        elif milestone_id == "m11_use_mode_and_commercial_boundary":
            ready = use_boundary_ready
            blockers = [
                "use_mode_compliance_guard_missing" if not use_mode_compliance else "",
                f"use_mode_compliance_status={use_guard_status}" if use_mode_compliance and use_guard_status in BAD_STATUSES else "",
                "commercial_boundary_blockers_present" if commercial_blockers else "",
                f"commercial_readiness_status={commercial_guard_status}" if commercial_readiness and commercial_guard_status in BAD_STATUSES else "",
                "commercial_readiness_blockers_present" if commercial_framework_blockers else "",
                "commercial_release_blocked" if commercial_readiness and bool(commercial_readiness.get("commercial_release_blocked", False)) else "",
                "use_mode_guard_granted_live_execution_authority" if bool(use_authority.get("live_execution_authority", False)) else "",
                "commercial_readiness_granted_live_execution_authority" if bool(commercial_authority.get("live_execution_authority", False)) else "",
                "use_mode_guard_does_not_confirm_read_only_authority" if not bool(use_authority.get("does_not_enable_live_execution", True)) else "",
            ]
            evidence = {
                "use_mode": use_mode_compliance.get("use_mode"),
                "use_mode_status": use_mode_compliance.get("overall_status"),
                "commercial_use_intent_detected": commercial_intent,
                "commercial_clearance_status": use_commercial.get("commercial_clearance_status"),
                "commercial_blockers": commercial_blockers,
                "personal_use_grade": _as_dict(use_mode_compliance.get("personal_use")).get("grade"),
                "perfect_personal_use_ready": bool(_as_dict(use_mode_compliance.get("personal_use")).get("perfect_personal_use_ready", False)),
                "authority_boundaries": use_authority,
                "commercial_readiness_status": commercial_readiness.get("overall_status"),
                "commercial_product_mode": commercial_readiness.get("commercial_product_mode"),
                "commercial_framework_intent": commercial_framework_intent,
                "commercial_release_ready": bool(commercial_readiness.get("commercial_release_ready", False)),
                "commercial_release_blocked": bool(commercial_readiness.get("commercial_release_blocked", False)),
                "commercial_readiness_blockers": commercial_framework_blockers,
                "commercial_authority_boundaries": commercial_authority,
            }
        else:
            ready = False
            blockers = [f"unknown_milestone_id={milestone_id or 'missing'}"]
            evidence = {}

        row = _milestone(definition, ready=ready, blockers=blockers, evidence=evidence)
        milestones.append(row)
        if milestone_id:
            prior_milestones[milestone_id] = bool(row.get("ready", False))
    return milestones


def _fresh_gate(
    project_root: Path,
    artifact_name: str,
    *,
    max_age_hours: float,
    now: datetime,
) -> dict[str, Any]:
    artifact_path = Path(artifact_name)
    if artifact_path.is_absolute():
        path = artifact_path
    elif len(artifact_path.parts) > 1:
        path = project_root / artifact_path
    else:
        path = project_root / "governance" / "health" / artifact_path
    payload = load_json(path)
    age_minutes = payload_age_minutes(payload, path, now=now) if payload else None
    status = _status(payload.get("overall_status") or payload.get("status"))
    ok_value = payload.get("ok") if "ok" in payload else payload.get("ready")
    ready = bool(
        payload
        and age_minutes is not None
        and age_minutes <= float(max_age_hours) * 60.0
        and (bool(ok_value) if ok_value is not None else status not in BAD_STATUSES)
        and status not in BAD_STATUSES
    )
    return {
        "artifact": artifact_name,
        "path": str(path),
        "present": bool(payload),
        "status": status or "unknown",
        "ok": ok_value,
        "age_minutes": round(float(age_minutes), 4) if age_minutes is not None else None,
        "max_age_minutes": float(max_age_hours) * 60.0,
        "fresh": bool(age_minutes is not None and age_minutes <= float(max_age_hours) * 60.0),
        "ready": ready,
    }


def _sustained_state(
    *,
    previous: dict[str, Any],
    all_gates_ready: bool,
    now: datetime,
    sustained_window_hours: float,
) -> dict[str, Any]:
    previous_since = str(previous.get("continuous_all_gates_ready_since_utc") or "").strip()
    since_dt: datetime | None = None
    if previous_since:
        try:
            since_dt = datetime.fromisoformat(previous_since.replace("Z", "+00:00")).astimezone(timezone.utc)
        except Exception:
            since_dt = None
    if not all_gates_ready:
        return {
            "continuous_all_gates_ready_since_utc": "",
            "sustained_window_hours": sustained_window_hours,
            "sustained_ready_hours": 0.0,
            "sustained_window_met": False,
        }
    if since_dt is None:
        since_dt = now
    ready_hours = max((now - since_dt).total_seconds() / 3600.0, 0.0)
    return {
        "continuous_all_gates_ready_since_utc": since_dt.isoformat(),
        "sustained_window_hours": sustained_window_hours,
        "sustained_ready_hours": round(ready_hours, 4),
        "sustained_window_met": ready_hours >= sustained_window_hours,
    }


def build_payload(
    project_root: Path = PROJECT_ROOT,
    *,
    config_path: Path = DEFAULT_CONFIG_PATH,
    out_path: Path = DEFAULT_OUT_PATH,
) -> dict[str, Any]:
    now = datetime.now(timezone.utc)
    health = project_root / "governance" / "health"
    config = load_json(config_path)
    if not config:
        config = load_json(DEFAULT_CONFIG_PATH)

    sustained_window_hours = _safe_float(config.get("sustained_window_hours"), 168.0)
    raw_live_floor = str(config.get("raw_profitability_live_canary_grade_floor") or "A")
    raw_hard_floor = str(config.get("raw_profitability_hard_block_floor") or "C")
    min_auth_expires = _safe_float(config.get("auth_min_expires_in_seconds"), 1800.0)
    storage_max_pressure = _safe_float(config.get("storage_max_pressure_index"), 0.2)
    storage_max_pending = _safe_int(config.get("storage_max_total_pending_lines"), 15000)
    freshness_hours = _as_dict(config.get("gate_freshness_hours"))

    paper_profit = load_json(health / "paper_profitability_control_latest.json")
    paper_runtime_profit = load_json(health / "paper_runtime_profitability_controls_latest.json")
    paper_truth = load_json(health / "paper_execution_truth_layer_latest.json")
    paper_runtime = load_json(health / "runtime_paper_regression_guard_latest.json")
    paper_ramp = load_json(health / "paper_400_ramp_latest.json")
    broker = load_json(health / "broker_readiness_latest.json")
    schwab_auth = load_json(health / "schwab_auth_supervisor_latest.json")
    auth_lease = load_json(health / "auth_lease_manager_latest.json")
    storage = load_json(health / "ingestion_storage_control_latest.json")
    health_fast = load_json(health / "health_fast_latest.json")
    live_money_contract = load_json(health / "live_money_readiness_contract_latest.json")
    live_canary_control = load_json(health / "live_canary_control_latest.json")
    production_readiness = load_json(health / "production_readiness_control_latest.json")
    use_mode_compliance = load_json(health / "use_mode_compliance_guard_latest.json")
    commercial_readiness = load_json(health / "commercial_readiness_control_latest.json")

    raw_grade = _grade(
        paper_profit.get("raw_profitability_grade")
        or paper_runtime_profit.get("raw_profitability_grade")
        or paper_profit.get("financial_profitability_grade")
        or paper_profit.get("grade")
    )
    raw_gate = _gate(
        "raw_profitability_posture",
        "Raw Profitability Posture",
        _grade_at_least(raw_grade, raw_live_floor),
        [
            f"raw_profitability_grade_below_{raw_live_floor}" if not _grade_at_least(raw_grade, raw_live_floor) else "",
            f"raw_profitability_hard_block_below_{raw_hard_floor}" if not _grade_at_least(raw_grade, raw_hard_floor) else "",
            "raw_profitability_artifact_missing" if not (paper_profit or paper_runtime_profit) else "",
        ],
        {
            "raw_profitability_grade": raw_grade or "unknown",
            "live_canary_grade_floor": raw_live_floor,
            "hard_block_floor": raw_hard_floor,
            "paper_profitability_status": paper_profit.get("overall_status") or paper_profit.get("status"),
            "runtime_profitability_status": paper_runtime_profit.get("overall_status") or paper_runtime_profit.get("status"),
        },
        owner="paper_profitability_control",
    )

    ramp_blockers = [str(item) for item in _as_list(paper_ramp.get("blockers")) if str(item or "").strip()]
    truth_failed = [str(item) for item in _as_list(paper_truth.get("failed_checks")) if str(item or "").strip()]
    runtime_failed = [str(item) for item in _as_list(paper_runtime.get("failed_checks")) if str(item or "").strip()]
    dropout_terms = ("dropout", "paper_trading_inactive", "paper_not_active", "sleeve_not_paper", "missing_paper")
    unexplained_dropout_markers = [
        item
        for item in [*ramp_blockers, *truth_failed, *runtime_failed]
        if any(term in item.lower() for term in dropout_terms)
    ]
    paper_gate = _gate(
        "sleeve_paper_trading_continuity",
        "Sleeve Paper-Trading Continuity",
        bool(
            paper_truth
            and paper_runtime
            and paper_ramp
            and _status(paper_truth.get("overall_status") or paper_truth.get("status")) not in BAD_STATUSES
            and _status(paper_runtime.get("overall_status") or paper_runtime.get("status")) not in BAD_STATUSES
            and _status(paper_ramp.get("overall_status") or paper_ramp.get("status") or paper_ramp.get("stage")) not in BAD_STATUSES
            and not ramp_blockers
            and not truth_failed
            and not runtime_failed
            and not unexplained_dropout_markers
        ),
        [
            "paper_execution_truth_missing" if not paper_truth else "",
            "runtime_paper_regression_guard_missing" if not paper_runtime else "",
            "paper_400_ramp_missing" if not paper_ramp else "",
            "paper_ramp_blockers_present" if ramp_blockers else "",
            "paper_truth_failed_checks_present" if truth_failed else "",
            "runtime_paper_failed_checks_present" if runtime_failed else "",
            "unexplained_paper_trading_dropout_marker_present" if unexplained_dropout_markers else "",
        ],
        {
            "paper_truth_status": paper_truth.get("overall_status") or paper_truth.get("status"),
            "runtime_paper_status": paper_runtime.get("overall_status") or paper_runtime.get("status"),
            "paper_ramp_status": paper_ramp.get("overall_status") or paper_ramp.get("status") or paper_ramp.get("stage"),
            "paper_ramp_blockers": ramp_blockers,
            "paper_truth_failed_checks": truth_failed,
            "runtime_paper_failed_checks": runtime_failed,
            "dropout_markers": unexplained_dropout_markers,
        },
        owner="runtime_paper_regression_guard",
    )

    lease_budget = _as_dict(auth_lease.get("lease_budget"))
    token = _as_dict(schwab_auth.get("token"))
    expires_in = max(
        _safe_float(auth_lease.get("expires_in_seconds"), 0.0),
        _safe_float(lease_budget.get("expires_in_seconds"), 0.0),
        _safe_float(token.get("expires_in_seconds"), 0.0),
        _safe_float(broker.get("token_expires_in_seconds"), 0.0),
    )
    auth_gate = _gate(
        "auth_token_continuity",
        "Auth And Token Continuity",
        bool(
            broker
            and schwab_auth
            and auth_lease
            and bool(broker.get("ready_for_open", broker.get("broker_ready", False)))
            and bool(broker.get("auth_ok", True))
            and bool(broker.get("network_ok", True))
            and _status(schwab_auth.get("overall_status") or schwab_auth.get("status")) not in BAD_STATUSES
            and _status(auth_lease.get("overall_status") or auth_lease.get("status")) not in BAD_STATUSES
            and _status(auth_lease.get("lease_state")) not in {"critical", "expired", "blocked", "warning"}
            and expires_in >= min_auth_expires
        ),
        [
            "broker_readiness_missing" if not broker else "",
            "schwab_auth_supervisor_missing" if not schwab_auth else "",
            "auth_lease_manager_missing" if not auth_lease else "",
            "broker_not_ready" if broker and not bool(broker.get("ready_for_open", broker.get("broker_ready", False))) else "",
            "broker_auth_not_ok" if broker and not bool(broker.get("auth_ok", True)) else "",
            "broker_network_not_ok" if broker and not bool(broker.get("network_ok", True)) else "",
            f"auth_expires_below_{int(min_auth_expires)}s" if expires_in < min_auth_expires else "",
            "auth_status_not_ready" if schwab_auth and _status(schwab_auth.get("overall_status") or schwab_auth.get("status")) in BAD_STATUSES else "",
            "lease_status_not_ready" if auth_lease and _status(auth_lease.get("overall_status") or auth_lease.get("status")) in BAD_STATUSES else "",
        ],
        {
            "broker_status": broker.get("overall_status") or broker.get("status"),
            "broker_ready": broker.get("ready_for_open", broker.get("broker_ready")),
            "broker_auth_ok": broker.get("auth_ok"),
            "broker_network_ok": broker.get("network_ok"),
            "schwab_auth_status": schwab_auth.get("overall_status") or schwab_auth.get("status"),
            "auth_lease_status": auth_lease.get("overall_status") or auth_lease.get("status"),
            "lease_state": auth_lease.get("lease_state"),
            "expires_in_seconds": expires_in,
            "min_expires_in_seconds": min_auth_expires,
        },
        owner="auth_lease_manager",
    )

    source_guard = source_mutation_guard.build_payload(project_root)
    source_gate = _gate(
        "runtime_source_mutation_guard",
        "Runtime Source Mutation Guard",
        bool(source_guard.get("ok", False)),
        [
            "runtime_source_mutation_guard_not_clean" if not source_guard.get("ok", False) else "",
            "protected_source_dirty" if _safe_int(source_guard.get("dirty_count"), 0) > 0 else "",
        ],
        {
            "source_mutation_guard_status": source_guard.get("overall_status"),
            "dirty_count": source_guard.get("dirty_count"),
            "dirty_entries": source_guard.get("dirty_entries", []),
            "error": source_guard.get("error", ""),
        },
        owner="source_mutation_guard",
    )

    production_smoke = production_flow_smoke.build_payload(project_root)
    if not production_readiness:
        production_readiness = production_readiness_control.build_payload(project_root)
    if not use_mode_compliance:
        use_mode_compliance = use_mode_compliance_guard.build_payload(project_root)
    if not commercial_readiness:
        commercial_readiness = commercial_readiness_control.build_payload(project_root)
    ci_gate = _gate(
        "ci_production_guardrails",
        "CI And Production Guardrails",
        bool(production_smoke.get("ok", False)),
        [
            "production_flow_smoke_failed" if not production_smoke.get("ok", False) else "",
            *[f"failed_check={item}" for item in _as_list(production_smoke.get("failed_checks"))],
        ],
        {
            "production_flow_smoke_status": production_smoke.get("overall_status"),
            "failed_checks": production_smoke.get("failed_checks", []),
        },
        owner="production_flow_smoke",
    )

    storage_status = _status(storage.get("overall_status") or storage.get("status"))
    pressure_index = _safe_float(storage.get("pressure_index"), _safe_float(storage.get("pressure"), 0.0))
    backpressure = _as_dict(storage.get("backpressure"))
    effective = _as_dict(backpressure.get("effective_raw_live"))
    total_pending = _safe_int(effective.get("total_pending_lines"), _safe_int(backpressure.get("total_pending_lines"), 0))
    storage_gate = _gate(
        "storage_pressure_clean",
        "Storage Pressure Clean",
        bool(
            storage
            and storage_status not in BAD_STATUSES
            and pressure_index <= storage_max_pressure
            and total_pending <= storage_max_pending
        ),
        [
            "ingestion_storage_control_missing" if not storage else "",
            "storage_status_not_ready" if storage and storage_status in BAD_STATUSES else "",
            "storage_pressure_index_too_high" if pressure_index > storage_max_pressure else "",
            "storage_total_pending_lines_too_high" if total_pending > storage_max_pending else "",
        ],
        {
            "overall_status": storage.get("overall_status") or storage.get("status"),
            "severity": storage.get("severity"),
            "pressure_index": pressure_index,
            "max_pressure_index": storage_max_pressure,
            "total_pending_lines": total_pending,
            "max_total_pending_lines": storage_max_pending,
        },
        owner="ingestion_storage_control",
    )

    freshness_specs = {
        "health_gates": ("health_gates_latest.json", _safe_float(freshness_hours.get("health_gates"), 2.0)),
        "promotion_quality_gate": ("promotion_quality_gate_latest.json", _safe_float(freshness_hours.get("promotion_quality_gate"), 24.0)),
        "promotion_readiness": ("governance/walk_forward/promotion_readiness_latest.json", _safe_float(freshness_hours.get("promotion_readiness"), 24.0)),
        "paper_performance": ("paper_performance_latest.json", _safe_float(freshness_hours.get("paper_performance"), 12.0)),
        "runtime_paper_regression_guard": (
            "runtime_paper_regression_guard_latest.json",
            _safe_float(freshness_hours.get("runtime_paper_regression_guard"), 12.0),
        ),
    }
    freshness_rows = {
        name: _fresh_gate(project_root, artifact, max_age_hours=max_age, now=now)
        for name, (artifact, max_age) in freshness_specs.items()
    }
    stale_or_blocked = [name for name, row in freshness_rows.items() if not row["ready"]]
    freshness_gate = _gate(
        "promotion_paper_gate_freshness",
        "Promotion And Paper Gate Freshness",
        not stale_or_blocked,
        [f"{name}_not_ready_or_stale" for name in stale_or_blocked],
        {"artifacts": freshness_rows},
        owner="promotion_gate_snapshot_policy",
    )

    gates = [raw_gate, paper_gate, auth_gate, source_gate, ci_gate, storage_gate, freshness_gate]
    all_gates_ready = all(gate["ready"] for gate in gates)
    previous = load_json(out_path)
    sustained = _sustained_state(
        previous=previous,
        all_gates_ready=all_gates_ready,
        now=now,
        sustained_window_hours=sustained_window_hours,
    )
    milestones = _build_live_money_canary_milestones(
        config=config,
        gates=gates,
        sustained=sustained,
        health_fast=health_fast,
        paper_truth=paper_truth,
        live_money_contract=live_money_contract,
        live_canary_control=live_canary_control,
        production_readiness=production_readiness,
        use_mode_compliance=use_mode_compliance,
        commercial_readiness=commercial_readiness,
    )
    require_milestones = bool(config.get("require_live_money_canary_milestones", True))
    required_milestones_ready = all(
        milestone.get("ready", False)
        for milestone in milestones
        if bool(milestone.get("required", True))
    )
    live_canary_ready = bool(
        all_gates_ready
        and sustained["sustained_window_met"]
        and (required_milestones_ready or not require_milestones)
    )
    blockers = ordered_unique(
        [
            *[f"{gate['gate_id']}_blocked" for gate in gates if not gate["ready"]],
            "sustained_window_not_met" if all_gates_ready and not sustained["sustained_window_met"] else "",
            "live_money_canary_milestones_not_ready"
            if require_milestones and not required_milestones_ready
            else "",
            *[
                f"{milestone['milestone_id']}_blocked"
                for milestone in milestones
                if bool(milestone.get("required", True)) and not bool(milestone.get("ready", False))
            ],
        ]
    )
    return {
        "schema_version": 1,
        "timestamp_utc": iso_now(),
        "policy_id": str(config.get("policy_id") or POLICY_ID),
        "source": "live_canary_readiness_contract",
        "ok": live_canary_ready,
        "overall_status": "ready" if live_canary_ready else "blocked",
        "live_canary_money_ready": live_canary_ready,
        "live_money_canary_blocked": not live_canary_ready,
        "infrastructure_message": str(config.get("infrastructure_message") or ""),
        "readiness_bar": [
            "no raw D-grade posture",
            "no unexplained sleeve paper-trading dropouts",
            "no auth/token surprises",
            "no source mutation from runtime",
            "clean CI",
            "clean storage pressure",
            "clean promotion/paper gate freshness",
            "clean live-money production bar",
            "clear use-mode and commercial boundary",
            "clear seven-section commercial readiness framework",
            "all gates sustained before live canary money",
        ],
        "milestone_bar": [str(item.get("title") or item.get("milestone_id")) for item in milestones],
        "gate_count": len(gates),
        "ready_gate_count": sum(1 for gate in gates if gate["ready"]),
        "milestone_count": len(milestones),
        "ready_milestone_count": sum(1 for milestone in milestones if milestone.get("ready", False)),
        "required_milestone_count": sum(1 for milestone in milestones if bool(milestone.get("required", True))),
        "ready_required_milestone_count": sum(
            1
            for milestone in milestones
            if bool(milestone.get("required", True)) and bool(milestone.get("ready", False))
        ),
        "required_live_money_canary_milestones_ready": required_milestones_ready,
        "require_live_money_canary_milestones": require_milestones,
        "blocked_milestones": [
            milestone["milestone_id"]
            for milestone in milestones
            if bool(milestone.get("required", True)) and not bool(milestone.get("ready", False))
        ],
        "blockers": blockers,
        "sustained_window": sustained,
        "gates": gates,
        "live_money_canary_milestones": milestones,
        "infrastructure_bot_contract": {
            "target_bots": [
                "infrabot_adaptive_governor",
                "infrastructure_autofix_bot",
                "master_infrastructure_supervisor",
                "runtime_gate_dashboard",
                "system_signal_bus",
                "system_self_model",
                "production_readiness_control",
                "use_mode_compliance_guard",
                "commercial_readiness_control",
            ],
            "live_execution_authority": False,
            "must_keep_live_orders_disabled_until_ready": True,
            "repair_bias": "production_hardening_soak_before_live_canary",
        },
        "recommended_actions": ordered_unique(
            [
                "keep live orders disabled until live_canary_money_ready=true",
                "route blocked gates to the owning infrastructure bot",
                "refresh use-mode-compliance before any commercial/customer-facing or live-money promotion discussion",
                "refresh commercial-readiness before any public, customer-facing, or paid product release discussion",
                "refresh this contract after paper/auth/storage/promotion guard repairs",
                "treat blocked live-money canary milestones as pre-canary work, not runtime noise"
                if require_milestones and not required_milestones_ready
                else "",
                "do not treat controlled A+ posture as raw-profitability proof while raw grade is below A",
            ]
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Publish the hard infrastructure bar required before live-canary money.")
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_PATH)
    parser.add_argument("--apply", action="store_true", help="Write the live canary readiness contract health artifact.")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    payload = build_payload(args.project_root, config_path=args.config, out_path=args.out)
    if args.apply:
        write_payload(args.out, payload)
        payload["out_path"] = str(args.out)
    if args.json:
        print(json.dumps(payload, ensure_ascii=True, indent=2))
    else:
        print(
            "live_canary_readiness_contract "
            f"status={payload['overall_status']} "
            f"ready={int(bool(payload['live_canary_money_ready']))} "
            f"ready_gates={payload['ready_gate_count']}/{payload['gate_count']}"
        )
    return 0 if payload["overall_status"] in {"ready", "blocked"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
