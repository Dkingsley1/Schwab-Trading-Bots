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
    from scripts.ops import production_flow_smoke, source_mutation_guard
else:
    from .long_runtime_common import PROJECT_ROOT, iso_now, load_json, ordered_unique, payload_age_minutes, write_payload
    from . import production_flow_smoke, source_mutation_guard


DEFAULT_OUT_PATH = PROJECT_ROOT / "governance" / "health" / "live_canary_readiness_contract_latest.json"
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "live_canary_readiness_contract.json"
POLICY_ID = "production_hardening_live_canary_bar_v1"
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


def _fresh_gate(
    project_root: Path,
    artifact_name: str,
    *,
    max_age_hours: float,
    now: datetime,
) -> dict[str, Any]:
    path = project_root / "governance" / "health" / artifact_name
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
        "promotion_readiness": ("promotion_readiness_latest.json", _safe_float(freshness_hours.get("promotion_readiness"), 24.0)),
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
    live_canary_ready = bool(all_gates_ready and sustained["sustained_window_met"])
    blockers = ordered_unique(
        [
            *[f"{gate['gate_id']}_blocked" for gate in gates if not gate["ready"]],
            "sustained_window_not_met" if all_gates_ready and not sustained["sustained_window_met"] else "",
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
            "all gates sustained before live canary money",
        ],
        "gate_count": len(gates),
        "ready_gate_count": sum(1 for gate in gates if gate["ready"]),
        "blockers": blockers,
        "sustained_window": sustained,
        "gates": gates,
        "infrastructure_bot_contract": {
            "target_bots": [
                "infrabot_adaptive_governor",
                "infrastructure_autofix_bot",
                "master_infrastructure_supervisor",
                "runtime_gate_dashboard",
                "system_signal_bus",
                "system_self_model",
            ],
            "live_execution_authority": False,
            "must_keep_live_orders_disabled_until_ready": True,
            "repair_bias": "production_hardening_soak_before_live_canary",
        },
        "recommended_actions": ordered_unique(
            [
                "keep live orders disabled until live_canary_money_ready=true",
                "route blocked gates to the owning infrastructure bot",
                "refresh this contract after paper/auth/storage/promotion guard repairs",
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
