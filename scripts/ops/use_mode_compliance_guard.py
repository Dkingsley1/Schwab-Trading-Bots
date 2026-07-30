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
else:
    from .long_runtime_common import PROJECT_ROOT, iso_now, load_json, ordered_unique, write_payload


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


def _env_bool(env: dict[str, str], name: str, default: bool = False) -> bool:
    value = str(env.get(name, "")).strip().lower()
    if not value:
        return default
    return value in TRUE_VALUES


def _health(project_root: Path, name: str) -> dict[str, Any]:
    payload = load_json(project_root / "governance" / "health" / name)
    return payload if isinstance(payload, dict) else {}


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
            bool((storage or health_fast) and storage_status not in {"blocked", "critical", "failed"} and storage_pressure <= 0.2),
            [
                "storage_artifact_missing" if not storage and not health_fast else "",
                f"storage_status={storage_status}" if storage_status in {"blocked", "critical", "failed"} else "",
                "storage_pressure_above_personal_use_floor" if storage_pressure > 0.2 else "",
            ],
            {"storage_status": storage_status or "missing", "pressure_index": storage_pressure},
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
    return {
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
