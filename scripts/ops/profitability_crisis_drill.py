#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.accountability import safe_write_json_atomic
from core.execution_simulator import simulate_execution
from core.profitability_hardening import evaluate_profitability_entry

DEFAULT_POLICY_PATH = Path("config/profitability_crisis_drill_v1.json")
DEFAULT_CANDIDATE_PATH = Path("governance/runtime/production_candidate_state.json")
DEFAULT_OUT_PATH = Path("governance/research/profitability_crisis_drill_latest.json")
REJECTED_EXECUTION_STATUSES = {
    "crossed_or_locked_quote_rejected",
    "stale_quote_rejected",
    "rejected",
}


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _sha256_file(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return ""


def _resolve_path(project_root: Path, raw_path: str | Path) -> Path:
    path = Path(raw_path).expanduser()
    return path if path.is_absolute() else project_root / path


def _float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _grade(score: float) -> str:
    if score >= 99.5:
        return "A+"
    if score >= 93.0:
        return "A"
    if score >= 87.0:
        return "B+"
    if score >= 80.0:
        return "B"
    if score >= 70.0:
        return "C"
    if score >= 60.0:
        return "D"
    return "F"


def _candidate_binding(project_root: Path) -> tuple[dict[str, Any], Path, str]:
    path = project_root / DEFAULT_CANDIDATE_PATH
    payload = _load_json(path)
    binding = {
        "candidate_id": str(payload.get("candidate_id") or ""),
        "candidate_generation": int(_float(payload.get("generation"), 0.0)),
        "accepted_at_utc": str(payload.get("accepted_at_utc") or ""),
        "accepted_git_head": str(payload.get("accepted_git_head") or ""),
        "live_execution_authority": bool(
            payload.get("live_execution_authority", False)
        ),
    }
    binding["valid"] = bool(
        binding["candidate_id"]
        and binding["candidate_generation"] > 0
        and binding["accepted_at_utc"]
    )
    return binding, path, _sha256_file(path)


def _official_source_ready(source: Mapping[str, Any]) -> tuple[bool, list[str]]:
    references = source.get("references")
    rows = references if isinstance(references, list) else []
    failures: list[str] = []
    if not rows:
        failures.append("official_references_missing")
    for index, raw in enumerate(rows):
        row = raw if isinstance(raw, Mapping) else {}
        if not str(row.get("publisher") or "").strip():
            failures.append(f"reference_{index}_publisher_missing")
        if not str(row.get("title") or "").strip():
            failures.append(f"reference_{index}_title_missing")
        if not str(row.get("url") or "").startswith("https://"):
            failures.append(f"reference_{index}_official_https_url_missing")
        if row.get("official_authoritative_source") is not True:
            failures.append(f"reference_{index}_official_source_flag_missing")
    return not failures, failures


def _execution_result(
    *,
    action: str,
    phase: Mapping[str, Any],
    policy: Mapping[str, Any],
    profile: str,
    symbol: str,
) -> dict[str, Any]:
    execution = policy.get("execution_contract")
    execution = execution if isinstance(execution, Mapping) else {}
    depth = max(_float(phase.get("same_side_depth"), 0.0), 0.0)
    result = simulate_execution(
        action=action,
        last_price=max(_float(execution.get("reference_price"), 100.0), 0.01),
        return_1m=_float(phase.get("diagnostic_forward_return_bps"), 0.0) / 10000.0,
        spread_bps=max(_float(phase.get("spread_bps"), 0.0), 0.0),
        volatility_1m=max(_float(phase.get("volatility_1m"), 0.0), 0.0),
        latency_ms=max(_float(phase.get("latency_ms"), 0.0), 0.0),
        bid_size=depth,
        ask_size=depth,
        order_size=max(_float(phase.get("order_size"), 1.0), 0.01),
        broker=str(execution.get("broker") or "schwab"),
        market_kind=str(execution.get("market_kind") or "equities"),
        asset_class=str(execution.get("asset_class") or "equities"),
        symbol=symbol,
        sleeve=profile,
        session=str(phase.get("session") or "regular"),
        order_type=str(execution.get("order_type") or "limit"),
        quote_age_ms=max(_float(phase.get("quote_age_ms"), 0.0), 0.0),
    )
    return {
        "action": action,
        "net_counterfactual_return_bps": round(result.adjusted_return_1m * 10000.0, 6),
        "total_cost_bps": round(result.total_cost_bps, 6),
        "expected_fill_price": round(result.expected_fill_price, 6),
        "effective_fill_ratio": round(result.effective_fill_ratio, 6),
        "paper_execution_status": result.paper_execution_status,
        "paper_execution_score": round(result.paper_execution_score, 6),
        "fill_quality_bucket": result.fill_quality_bucket,
        "cancel_probability": round(result.cancel_probability, 6),
        "requote_probability": round(result.requote_probability, 6),
        "reject_probability": round(result.reject_probability, 6),
        "stale_quote_probability": round(result.stale_quote_probability, 6),
        "queue_fill_probability": round(result.queue_fill_probability, 6),
        "market_impact_bps": round(result.market_impact_bps, 6),
    }


def _entry_gate(
    *,
    profile: str,
    phase: Mapping[str, Any],
    buy_execution: Mapping[str, Any],
) -> dict[str, Any]:
    features = {
        "profitability_strict_evidence_required": True,
        "market_micro_tradeability_score_norm": phase.get("tradeability_norm"),
        "execution_fitness_norm": phase.get("execution_fitness_norm"),
        "news_source_quality_norm": phase.get("source_quality_norm"),
        "core_cross_asset_confirmation_norm": phase.get(
            "cross_asset_confirmation_norm"
        ),
        "core_portfolio_overlap_pressure_norm": phase.get(
            "portfolio_overlap_pressure_norm"
        ),
        "cross_bot_conflict_norm": phase.get("cross_bot_conflict_norm"),
        "spread_bps": phase.get("spread_bps"),
        "quote_age_ms": phase.get("quote_age_ms"),
        "liquidity_quality_norm": phase.get("liquidity_quality_norm"),
        "session_quality_norm": phase.get("session_quality_norm"),
        "session": phase.get("session") or "regular",
        "predicted_edge_lower_confidence_bound_bps": phase.get(
            "predicted_edge_lcb_bps"
        ),
        "round_trip_cost_bps": 2.0 * _float(buy_execution.get("total_cost_bps"), 0.0),
        "minimum_edge_cost_multiple": 1.5,
    }
    return evaluate_profitability_entry(profile=profile, features=features)


def _phase_required_fields_present(phase: Mapping[str, Any]) -> tuple[bool, list[str]]:
    required = (
        "phase",
        "severity_norm",
        "diagnostic_forward_return_bps",
        "spread_bps",
        "volatility_1m",
        "latency_ms",
        "same_side_depth",
        "order_size",
        "quote_age_ms",
        "tradeability_norm",
        "execution_fitness_norm",
        "liquidity_quality_norm",
        "session_quality_norm",
        "source_quality_norm",
        "cross_asset_confirmation_norm",
        "portfolio_overlap_pressure_norm",
        "cross_bot_conflict_norm",
        "predicted_edge_lcb_bps",
        "expected_new_long_entry",
    )
    missing = [key for key in required if phase.get(key) in {None, ""}]
    return not missing, missing


def _run_scenario(
    scenario: Mapping[str, Any],
    *,
    policy: Mapping[str, Any],
    scenario_path: Path,
) -> dict[str, Any]:
    drill = scenario.get("profitability_drill")
    drill = drill if isinstance(drill, Mapping) else {}
    phases = drill.get("phases")
    phases = phases if isinstance(phases, list) else []
    severity = policy.get("severity_contract")
    severity = severity if isinstance(severity, Mapping) else {}
    severe_floor = _float(severity.get("severe_phase_floor_norm"), 0.70)
    recovery_phase = str(drill.get("recovery_phase") or "")
    profile = str(drill.get("profile") or "default")
    symbol = str(drill.get("symbol") or "SPY")
    source = scenario.get("source")
    source = source if isinstance(source, Mapping) else {}
    source_ready, source_failures = _official_source_ready(source)

    phase_rows: list[dict[str, Any]] = []
    for raw_phase in phases:
        phase = raw_phase if isinstance(raw_phase, Mapping) else {}
        phase_ready, missing_fields = _phase_required_fields_present(phase)
        buy = _execution_result(
            action="BUY", phase=phase, policy=policy, profile=profile, symbol=symbol
        )
        short = _execution_result(
            action="SELL_SHORT",
            phase=phase,
            policy=policy,
            profile=profile,
            symbol=symbol,
        )
        reduce_exit = _execution_result(
            action="SELL", phase=phase, policy=policy, profile=profile, symbol=symbol
        )
        hold = {
            "action": "HOLD",
            "net_counterfactual_return_bps": 0.0,
            "total_cost_bps": 0.0,
            "effective_fill_ratio": 1.0,
            "paper_execution_status": "no_trade",
        }
        entry = _entry_gate(profile=profile, phase=phase, buy_execution=buy)
        actions = {"BUY": buy, "HOLD": hold, "SELL_SHORT": short}
        best_action = max(
            actions,
            key=lambda action: _float(
                actions[action].get("net_counterfactual_return_bps")
            ),
        )
        phase_name = str(phase.get("phase") or "unknown")
        is_severe = _float(phase.get("severity_norm"), 0.0) >= severe_floor
        is_recovery = bool(recovery_phase and phase_name == recovery_phase)
        expected_allowed = (
            str(phase.get("expected_new_long_entry") or "").lower() == "allowed"
        )
        entry_allowed = bool(entry.get("allowed", False))
        reduce_only_available = bool(
            _float(reduce_exit.get("effective_fill_ratio"), 0.0) > 0.0
            and str(reduce_exit.get("paper_execution_status") or "")
            not in REJECTED_EXECUTION_STATUSES
        )
        phase_rows.append(
            {
                "phase": phase_name,
                "severity_norm": round(_float(phase.get("severity_norm"), 0.0), 6),
                "diagnostic_forward_return_bps": round(
                    _float(phase.get("diagnostic_forward_return_bps"), 0.0), 6
                ),
                "phase_contract_complete": phase_ready,
                "missing_phase_fields": missing_fields,
                "is_severe_phase": is_severe,
                "is_recovery_phase": is_recovery,
                "new_long_entry": {
                    "allowed": entry_allowed,
                    "expected_allowed": expected_allowed,
                    "expectation_matched": entry_allowed == expected_allowed,
                    "blockers": list(entry.get("blockers") or []),
                    "risk_multiplier_norm": entry.get("risk_multiplier_norm"),
                    "regime_fit_norm": entry.get("regime_fit_norm"),
                    "entry_economics": dict(entry.get("entry_economics") or {}),
                    "execution_plan": dict(entry.get("execution_plan") or {}),
                },
                "new_exposure_counterfactuals": actions,
                "best_cost_adjusted_action": best_action,
                "best_cost_adjusted_return_bps": actions[best_action][
                    "net_counterfactual_return_bps"
                ],
                "existing_long_reduce_only_exit": {
                    **reduce_exit,
                    "available": reduce_only_available,
                    "may_cross_flat": False,
                    "creates_new_short_exposure": False,
                },
                "severe_new_long_entry_blocked": bool(is_severe and not entry_allowed),
                "recovery_reentry_opportunity": bool(
                    is_recovery
                    and entry_allowed
                    and best_action == "BUY"
                    and _float(buy.get("net_counterfactual_return_bps"), 0.0) > 0.0
                ),
            }
        )

    severe_rows = [row for row in phase_rows if row["is_severe_phase"]]
    recovery_rows = [row for row in phase_rows if row["is_recovery_phase"]]
    scenario_checks = {
        "scenario_receipt_present": bool(_sha256_file(scenario_path)),
        "official_sources_complete": source_ready,
        "phase_contracts_complete": bool(phase_rows)
        and all(row["phase_contract_complete"] for row in phase_rows),
        "expected_entry_postures_match": bool(phase_rows)
        and all(row["new_long_entry"]["expectation_matched"] for row in phase_rows),
        "severe_new_long_entries_blocked": bool(severe_rows)
        and all(row["severe_new_long_entry_blocked"] for row in severe_rows),
        "reduce_only_exit_paths_available": bool(phase_rows)
        and all(
            row["existing_long_reduce_only_exit"]["available"] for row in phase_rows
        ),
        "recovery_reentry_opportunity_present": bool(recovery_rows)
        and all(row["recovery_reentry_opportunity"] for row in recovery_rows),
        "cost_adjusted_action_comparison_complete": bool(phase_rows)
        and all(
            set(row["new_exposure_counterfactuals"]) == {"BUY", "HOLD", "SELL_SHORT"}
            for row in phase_rows
        ),
    }
    return {
        "scenario_id": str(scenario.get("scenario_id") or ""),
        "aliases": list(scenario.get("aliases") or []),
        "scenario_path": str(scenario_path),
        "scenario_receipt_sha256": _sha256_file(scenario_path),
        "source": dict(source),
        "source_failures": source_failures,
        "profile": profile,
        "symbol": symbol,
        "parameter_notice": str(drill.get("parameter_notice") or ""),
        "phase_count": len(phase_rows),
        "severe_phase_count": len(severe_rows),
        "recovery_phase_count": len(recovery_rows),
        "checks": scenario_checks,
        "ok": all(scenario_checks.values()),
        "phases": phase_rows,
    }


def _requested_scenarios(
    rows: Sequence[tuple[Path, dict[str, Any]]], requested: Sequence[str] | None
) -> tuple[list[tuple[Path, dict[str, Any]]], list[str]]:
    wanted = [
        str(value or "").strip()
        for value in (requested or [])
        if str(value or "").strip()
    ]
    if not wanted or "all" in wanted:
        return list(rows), []
    selected: list[tuple[Path, dict[str, Any]]] = []
    unresolved: list[str] = []
    for name in wanted:
        match = next(
            (
                row
                for row in rows
                if name
                in {
                    str(row[1].get("scenario_id") or ""),
                    *[str(alias) for alias in row[1].get("aliases") or []],
                }
            ),
            None,
        )
        if match is None:
            unresolved.append(name)
        elif match not in selected:
            selected.append(match)
    return selected, unresolved


def build_payload(
    *,
    project_root: Path = PROJECT_ROOT,
    policy_path: str | Path = DEFAULT_POLICY_PATH,
    requested_scenarios: Sequence[str] | None = None,
    generated_at_utc: str | None = None,
) -> dict[str, Any]:
    root = Path(project_root).expanduser().resolve()
    resolved_policy_path = _resolve_path(root, policy_path)
    policy = _load_json(resolved_policy_path)
    binding, candidate_path, candidate_sha_before = _candidate_binding(root)
    configured_rows: list[tuple[Path, dict[str, Any]]] = []
    for raw_path in policy.get("scenario_paths") or []:
        scenario_path = _resolve_path(root, str(raw_path))
        configured_rows.append((scenario_path, _load_json(scenario_path)))
    selected_rows, unresolved = _requested_scenarios(
        configured_rows, requested_scenarios
    )
    scenario_results = [
        _run_scenario(scenario, policy=policy, scenario_path=path)
        for path, scenario in selected_rows
    ]
    authority = policy.get("authority_contract")
    authority = authority if isinstance(authority, Mapping) else {}
    authority_locked = bool(authority) and not any(
        bool(value) for value in authority.values()
    )
    candidate_sha_after = _sha256_file(candidate_path)
    candidate_unchanged = bool(
        candidate_sha_before
        and candidate_sha_after
        and candidate_sha_before == candidate_sha_after
    )
    all_mode = not requested_scenarios or "all" in (requested_scenarios or [])
    minimum_count = (
        int(_float(policy.get("minimum_scenario_count"), 3.0))
        if all_mode
        else len(selected_rows)
    )
    checks = {
        "policy_receipt_present": bool(_sha256_file(resolved_policy_path)),
        "candidate_binding_valid": bool(binding.get("valid")),
        "candidate_state_unchanged": candidate_unchanged,
        "authority_locked": authority_locked,
        "scenario_selection_resolved": not unresolved,
        "scenario_count_floor_met": len(scenario_results) >= max(minimum_count, 1),
        "scenario_receipts_complete": bool(scenario_results)
        and all(row["checks"]["scenario_receipt_present"] for row in scenario_results),
        "official_sources_complete": bool(scenario_results)
        and all(row["checks"]["official_sources_complete"] for row in scenario_results),
        "phase_contracts_complete": bool(scenario_results)
        and all(row["checks"]["phase_contracts_complete"] for row in scenario_results),
        "expected_entry_postures_match": bool(scenario_results)
        and all(
            row["checks"]["expected_entry_postures_match"] for row in scenario_results
        ),
        "severe_new_long_entries_blocked": bool(scenario_results)
        and all(
            row["checks"]["severe_new_long_entries_blocked"] for row in scenario_results
        ),
        "reduce_only_exit_paths_available": bool(scenario_results)
        and all(
            row["checks"]["reduce_only_exit_paths_available"]
            for row in scenario_results
        ),
        "recovery_reentry_opportunities_present": bool(scenario_results)
        and all(
            row["checks"]["recovery_reentry_opportunity_present"]
            for row in scenario_results
        ),
        "cost_adjusted_action_comparisons_complete": bool(scenario_results)
        and all(
            row["checks"]["cost_adjusted_action_comparison_complete"]
            for row in scenario_results
        ),
    }
    passed = sum(1 for value in checks.values() if value)
    score = round(100.0 * passed / max(len(checks), 1), 4)
    phase_rows = [phase for row in scenario_results for phase in row["phases"]]
    best_actions = Counter(str(row["best_cost_adjusted_action"]) for row in phase_rows)
    degraded_fills = sum(
        1
        for row in phase_rows
        if str(row["new_exposure_counterfactuals"]["BUY"].get("fill_quality_bucket"))
        in {"fair", "poor"}
        or _float(
            row["new_exposure_counterfactuals"]["BUY"].get("effective_fill_ratio"), 1.0
        )
        < 0.75
    )
    return {
        "timestamp_utc": generated_at_utc or datetime.now(timezone.utc).isoformat(),
        "schema_version": 1,
        "policy_id": str(policy.get("policy_id") or ""),
        "operating_mode": str(policy.get("operating_mode") or ""),
        "status": "ready" if all(checks.values()) else "degraded",
        "ok": all(checks.values()),
        "control_grade": _grade(score),
        "control_score": score,
        "candidate_binding": binding,
        "candidate_mutation_guard": {
            "candidate_path": str(candidate_path),
            "sha256_before": candidate_sha_before,
            "sha256_after": candidate_sha_after,
            "unchanged": candidate_unchanged,
        },
        "selection": {
            "requested": list(requested_scenarios or ["all"]),
            "unresolved": unresolved,
            "scenario_count": len(scenario_results),
        },
        "checks": checks,
        "failed_checks": [
            name for name, passed_check in checks.items() if not passed_check
        ],
        "diagnostic_summary": {
            "phase_count": len(phase_rows),
            "severe_phase_count": sum(
                1 for row in phase_rows if row["is_severe_phase"]
            ),
            "severe_new_long_blocks": sum(
                1 for row in phase_rows if row["severe_new_long_entry_blocked"]
            ),
            "recovery_reentry_opportunities": sum(
                1 for row in phase_rows if row["recovery_reentry_opportunity"]
            ),
            "degraded_fill_phase_count": degraded_fills,
            "best_cost_adjusted_action_counts": dict(sorted(best_actions.items())),
        },
        "scenario_results": scenario_results,
        "evidence_classification": {
            "diagnostic_only": True,
            "historical_event_anchors_official": True,
            "stress_parameters_are_historical_ticks": False,
            "promotion_evidence": False,
            "organic_candidate_profitability_evidence": False,
            "live_release_evidence": False,
            "profitability_proof": False,
        },
        "resource_contract": {
            "persistent_processes_started": 0,
            "network_requests": 0,
            "broker_requests": 0,
            "paper_orders_submitted": 0,
            "live_orders_submitted": 0,
            "candidate_mutations": 0,
            "runtime_control_writes": 0,
            "work_units": len(phase_rows),
        },
        "authority_contract": dict(authority),
        "next_actions": [
            "Preserve severe-phase new-exposure abstention and reduce-only exit priority.",
            "Use candidate-forward paper outcomes to calibrate recovery re-entry; this drill cannot tune thresholds by itself.",
            "Investigate any phase where cost-adjusted action ranking, fill degradation, or expected entry posture regresses.",
        ],
        "receipts": {
            "policy_path": str(resolved_policy_path),
            "policy_sha256": _sha256_file(resolved_policy_path),
            "scenario_sha256": {
                row["scenario_id"]: row["scenario_receipt_sha256"]
                for row in scenario_results
            },
        },
    }


def publish_payload(
    *,
    project_root: Path,
    payload: Mapping[str, Any],
    out_path: str | Path = DEFAULT_OUT_PATH,
    source: str = "profitability_crisis_drill",
) -> bool:
    return safe_write_json_atomic(
        str(_resolve_path(project_root, out_path)),
        dict(payload),
        project_root=str(project_root),
        source=source,
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run deterministic paper-only profitability drills across the 2008, 2020, "
            "and 2023 financial-collapse scenarios."
        )
    )
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--policy", default=str(DEFAULT_POLICY_PATH))
    parser.add_argument("--scenario", action="append", default=[])
    parser.add_argument("--out-file", default=str(DEFAULT_OUT_PATH))
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    root = Path(args.project_root).expanduser().resolve()
    payload = build_payload(
        project_root=root,
        policy_path=args.policy,
        requested_scenarios=args.scenario or None,
    )
    written = publish_payload(
        project_root=root,
        payload=payload,
        out_path=args.out_file,
    )
    if not written:
        payload["ok"] = False
        payload["status"] = "degraded"
        payload.setdefault("failed_checks", []).append("artifact_write_failed")
    if args.json:
        print(json.dumps(payload, ensure_ascii=True, sort_keys=True))
    else:
        summary = payload.get("diagnostic_summary") or {}
        print(
            "profitability_crisis_drill "
            f"status={payload.get('status')} grade={payload.get('control_grade')} "
            f"scenarios={payload.get('selection', {}).get('scenario_count', 0)} "
            f"phases={summary.get('phase_count', 0)} "
            f"severe_blocks={summary.get('severe_new_long_blocks', 0)}/"
            f"{summary.get('severe_phase_count', 0)} "
            f"recovery_reentries={summary.get('recovery_reentry_opportunities', 0)}"
        )
    return 0 if payload.get("ok") else 2


if __name__ == "__main__":
    raise SystemExit(main())
