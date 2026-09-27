#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.accountability import safe_write_json_atomic
from core.paper_behavior_interventions import (
    DEFAULT_POLICY_PATH,
    build_runtime_overlay_proposal,
    canonical_sha256,
    evaluate_paper_behavior_interventions,
    file_sha256,
    load_policy,
)

DEFAULT_CANDIDATE_PATH = Path("governance/runtime/production_candidate_state.json")
DEFAULT_OUT_PATH = Path(
    "governance/research/paper_behavior_intervention_drill_latest.json"
)


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _float(value: Any, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return float(default)
    return parsed if math.isfinite(parsed) else float(default)


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _resolve(project_root: Path, raw_path: str | Path) -> Path:
    path = Path(raw_path).expanduser()
    return path if path.is_absolute() else project_root / path


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


def _utc_datetime(value: str | None) -> datetime:
    raw = str(value or "").strip().replace("Z", "+00:00")
    if not raw:
        return datetime.now(timezone.utc)
    try:
        parsed = datetime.fromisoformat(raw)
    except ValueError:
        return datetime.now(timezone.utc)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _candidate_binding(project_root: Path) -> tuple[dict[str, Any], Path, str]:
    path = project_root / DEFAULT_CANDIDATE_PATH
    payload = _load_json(path)
    receipt = file_sha256(path)
    binding = {
        "candidate_id": str(payload.get("candidate_id") or ""),
        "candidate_generation": int(_float(payload.get("generation"), 0.0)),
        "accepted_at_utc": str(payload.get("accepted_at_utc") or ""),
        "accepted_git_head": str(payload.get("accepted_git_head") or ""),
        "candidate_state_sha256": receipt,
        "live_execution_authority": bool(
            payload.get("live_execution_authority", False)
        ),
    }
    binding["valid"] = bool(
        binding["candidate_id"]
        and binding["candidate_generation"] > 0
        and binding["accepted_at_utc"]
        and receipt
    )
    return binding, path, receipt


def _diagnostic_net_bps(
    *,
    action: str,
    size_multiplier: float,
    forward_return_bps: float,
    round_trip_cost_bps: float,
) -> float:
    side = str(action or "HOLD").strip().upper()
    if side == "HOLD":
        return 0.0
    unit_net = _float(forward_return_bps) - max(_float(round_trip_cost_bps), 0.0)
    if side == "SELL":
        return round(unit_net, 8)
    return round(max(_float(size_multiplier), 0.0) * unit_net, 8)


def _run_case(
    *,
    case: Mapping[str, Any],
    baseline_features: Mapping[str, Any],
    runtime_proposal: Mapping[str, Any],
    evaluation_at_utc: datetime,
) -> dict[str, Any]:
    features = dict(baseline_features)
    features.update(_mapping(case.get("feature_overrides")))
    requested_action = str(case.get("requested_action") or "HOLD").upper()
    result = evaluate_paper_behavior_interventions(
        action=requested_action,
        features=features,
        runtime_contract=runtime_proposal,
        profile=str(case.get("profile") or "swing_aggressive"),
        strategy=str(case.get("strategy") or "drill_challenger"),
        now_utc=evaluation_at_utc,
    )
    final_action = str(result.get("action") or "HOLD").upper()
    final_multiplier = _float(result.get("entry_size_multiplier_norm"), 1.0)
    forward_bps = _float(case.get("diagnostic_forward_return_bps"), 0.0)
    cost_bps = max(_float(case.get("diagnostic_round_trip_cost_bps"), 0.0), 0.0)
    baseline_net = _diagnostic_net_bps(
        action=requested_action,
        size_multiplier=1.0,
        forward_return_bps=forward_bps,
        round_trip_cost_bps=cost_bps,
    )
    challenger_net = _diagnostic_net_bps(
        action=final_action,
        size_multiplier=final_multiplier,
        forward_return_bps=forward_bps,
        round_trip_cost_bps=cost_bps,
    )
    expected_action = str(case.get("expected_action") or requested_action).upper()
    expected_min = _float(case.get("expected_min_size_multiplier_norm"), 0.0)
    expected_max = _float(case.get("expected_max_size_multiplier_norm"), 1.0)
    expected_trigger = str(case.get("expected_trigger") or "")
    triggered = [str(value) for value in result.get("triggered_interventions", [])]
    no_origination = requested_action != "HOLD" or final_action == "HOLD"
    no_reversal = not (
        (requested_action == "BUY" and final_action == "SELL")
        or (requested_action == "SELL" and final_action == "BUY")
    )
    exit_preserved = requested_action != "SELL" or final_action == "SELL"
    hold_preserved = requested_action != "HOLD" or final_action == "HOLD"
    checks = {
        "expected_action": final_action == expected_action,
        "size_floor": final_multiplier + 1e-9 >= expected_min,
        "size_ceiling": final_multiplier <= expected_max + 1e-9,
        "entry_never_enlarged": final_multiplier <= 1.0 + 1e-9,
        "expected_intervention_triggered": (
            not expected_trigger or expected_trigger in triggered
        ),
        "diagnostic_net_result_non_worsening": challenger_net + 1e-9 >= baseline_net,
        "action_not_originated": no_origination,
        "action_not_reversed": no_reversal,
        "sell_exit_preserved": exit_preserved,
        "hold_preserved": hold_preserved,
    }
    return {
        "case_id": str(case.get("case_id") or ""),
        "ok": all(checks.values()),
        "checks": checks,
        "failed_checks": [name for name, passed in checks.items() if not passed],
        "requested_action": requested_action,
        "challenger_action": final_action,
        "challenger_size_multiplier_norm": round(final_multiplier, 8),
        "triggered_interventions": triggered,
        "reasons": list(result.get("reasons") or []),
        "diagnostic_economics": {
            "forward_return_bps": round(forward_bps, 8),
            "round_trip_cost_bps": round(cost_bps, 8),
            "baseline_net_bps": baseline_net,
            "challenger_net_bps": challenger_net,
            "diagnostic_net_improvement_bps": round(challenger_net - baseline_net, 8),
        },
        "feature_overrides": _mapping(case.get("feature_overrides")),
        "intervention_observations": list(result.get("observations") or []),
    }


def _selected_scenarios(
    scenarios: Sequence[Mapping[str, Any]], requested: Sequence[str] | None
) -> tuple[list[dict[str, Any]], list[str], bool]:
    rows = [dict(row) for row in scenarios if isinstance(row, Mapping)]
    if not requested:
        return rows, [], True
    requested_ids = [str(value).strip() for value in requested if str(value).strip()]
    if any(value.lower() == "all" for value in requested_ids):
        return rows, [], True
    by_id = {str(row.get("scenario_id") or ""): row for row in rows}
    selected = [by_id[value] for value in requested_ids if value in by_id]
    unresolved = [value for value in requested_ids if value not in by_id]
    return selected, unresolved, False


def build_payload(
    *,
    project_root: Path = PROJECT_ROOT,
    policy_path: str | Path = DEFAULT_POLICY_PATH,
    requested_scenarios: Sequence[str] | None = None,
    generated_at_utc: str | None = None,
) -> dict[str, Any]:
    resolved_policy_path = _resolve(project_root, policy_path)
    policy = load_policy(project_root=project_root, policy_path=resolved_policy_path)
    policy_sha = file_sha256(resolved_policy_path)
    candidate_binding, candidate_path, candidate_sha_before = _candidate_binding(
        project_root
    )
    generated_at = _utc_datetime(generated_at_utc)
    runtime_proposal = build_runtime_overlay_proposal(
        policy=policy,
        candidate_binding=candidate_binding,
        policy_sha256=policy_sha,
        generated_at_utc=generated_at,
    )
    scenarios = (
        policy.get("scenarios") if isinstance(policy.get("scenarios"), list) else []
    )
    selected, unresolved, all_mode = _selected_scenarios(scenarios, requested_scenarios)
    baseline_features = _mapping(policy.get("healthy_feature_baseline"))
    scenario_results: list[dict[str, Any]] = []
    for scenario in selected:
        case_rows = (
            scenario.get("cases") if isinstance(scenario.get("cases"), list) else []
        )
        case_results = [
            _run_case(
                case=case,
                baseline_features=baseline_features,
                runtime_proposal=runtime_proposal,
                evaluation_at_utc=generated_at,
            )
            for case in case_rows
            if isinstance(case, Mapping)
        ]
        checks = {
            "cases_present": bool(case_results),
            "all_cases_pass": bool(case_results)
            and all(bool(row.get("ok", False)) for row in case_results),
            "named_intervention_present": str(scenario.get("intervention_id") or "")
            in _mapping(policy.get("interventions")),
        }
        scenario_results.append(
            {
                "scenario_id": str(scenario.get("scenario_id") or ""),
                "intervention_id": str(scenario.get("intervention_id") or ""),
                "ok": all(checks.values()),
                "checks": checks,
                "failed_checks": [
                    name for name, passed in checks.items() if not passed
                ],
                "case_count": len(case_results),
                "cases": case_results,
            }
        )

    candidate_sha_after = file_sha256(candidate_path)
    case_results = [
        case
        for scenario in scenario_results
        for case in scenario.get("cases", [])
        if isinstance(case, dict)
    ]
    configured_count = len([row for row in scenarios if isinstance(row, Mapping)])
    passed_scenarios = sum(bool(row.get("ok", False)) for row in scenario_results)
    authority = _mapping(policy.get("authority_contract"))
    checks = {
        "policy_loaded": bool(policy),
        "policy_schema_supported": int(_float(policy.get("schema_version"), 0)) == 1,
        "policy_receipt_present": bool(policy_sha),
        "candidate_binding_valid": bool(candidate_binding.get("valid", False)),
        "candidate_state_unchanged": bool(candidate_sha_before)
        and candidate_sha_before == candidate_sha_after,
        "selection_resolved": not unresolved,
        "scenario_selection_present": bool(scenario_results),
        "all_selected_scenarios_pass": bool(scenario_results)
        and passed_scenarios == len(scenario_results),
        "full_pack_has_fourteen_scenarios": (not all_mode) or configured_count == 14,
        "full_pack_executed": (not all_mode)
        or len(scenario_results) == configured_count,
        "proposal_is_paper_only": runtime_proposal.get("paper_only") is True,
        "proposal_cannot_enable_live": runtime_proposal.get("live_execution_allowed")
        is False,
        "drill_cannot_write_runtime_control": authority.get("can_write_runtime_control")
        is False,
        "drill_cannot_submit_orders": authority.get("can_submit_paper_orders") is False
        and authority.get("can_submit_live_orders") is False,
        "drill_cannot_mutate_or_promote_candidate": authority.get(
            "can_mutate_candidate"
        )
        is False
        and authority.get("can_promote") is False,
        "all_cases_non_worsening": bool(case_results)
        and all(
            bool(
                _mapping(case.get("checks")).get(
                    "diagnostic_net_result_non_worsening", False
                )
            )
            for case in case_results
        ),
        "all_cases_monotonic_safe": bool(case_results)
        and all(
            all(
                bool(_mapping(case.get("checks")).get(key, False))
                for key in (
                    "entry_never_enlarged",
                    "action_not_originated",
                    "action_not_reversed",
                    "sell_exit_preserved",
                    "hold_preserved",
                )
            )
            for case in case_results
        ),
    }
    passed_checks = sum(bool(value) for value in checks.values())
    control_score = round(100.0 * passed_checks / max(len(checks), 1), 4)
    admission_eligible = bool(
        all_mode
        and configured_count == 14
        and len(scenario_results) == configured_count
        and passed_scenarios == configured_count
        and all(checks.values())
    )
    proposal_sha = canonical_sha256(runtime_proposal)
    payload = {
        "timestamp_utc": generated_at.isoformat(),
        "schema_version": 1,
        "status": "ready" if all(checks.values()) else "degraded",
        "ok": all(checks.values()),
        "control_grade": _grade(control_score),
        "control_score": control_score,
        "admission_eligible": admission_eligible,
        "checks": checks,
        "failed_checks": [name for name, passed in checks.items() if not passed],
        "selection": {
            "requested": list(requested_scenarios or ["all"]),
            "all_mode": all_mode,
            "unresolved": unresolved,
        },
        "scenario_summary": {
            "configured_scenario_count": configured_count,
            "executed_scenario_count": len(scenario_results),
            "passed_scenario_count": passed_scenarios,
            "failed_scenario_count": len(scenario_results) - passed_scenarios,
            "case_count": len(case_results),
            "blocked_case_count": sum(
                str(case.get("challenger_action") or "") == "HOLD"
                and str(case.get("requested_action") or "") == "BUY"
                for case in case_results
            ),
            "throttled_case_count": sum(
                str(case.get("challenger_action") or "") == "BUY"
                and _float(case.get("challenger_size_multiplier_norm"), 1.0) < 1.0
                for case in case_results
            ),
            "preserved_case_count": sum(
                str(case.get("challenger_action") or "")
                == str(case.get("requested_action") or "")
                and abs(_float(case.get("challenger_size_multiplier_norm"), 1.0) - 1.0)
                <= 1e-9
                for case in case_results
            ),
            "diagnostic_net_improvement_bps": round(
                sum(
                    _float(
                        _mapping(case.get("diagnostic_economics")).get(
                            "diagnostic_net_improvement_bps"
                        ),
                        0.0,
                    )
                    for case in case_results
                ),
                8,
            ),
        },
        "scenario_results": scenario_results,
        "runtime_overlay_proposal": runtime_proposal,
        "runtime_overlay_proposal_sha256": proposal_sha,
        "candidate_binding": candidate_binding,
        "candidate_state_receipt": {
            "path": str(candidate_path),
            "sha256_before": candidate_sha_before,
            "sha256_after": candidate_sha_after,
            "unchanged": candidate_sha_before == candidate_sha_after,
        },
        "authority_contract": authority,
        "resource_contract": {
            "persistent_processes_started": 0,
            "network_requests": 0,
            "broker_requests": 0,
            "paper_orders_submitted": 0,
            "live_orders_submitted": 0,
            "candidate_mutations": 0,
            "runtime_control_writes": 0,
            "work_units": len(case_results),
        },
        "evidence_classification": {
            "diagnostic_champion_challenger_evidence": True,
            "organic_candidate_profitability_evidence": False,
            "promotion_evidence": False,
            "live_release_evidence": False,
            "profitability_guarantee": False,
            "paper_behavior_change_requires_single_writer_admission": True,
        },
        "receipts": {
            "policy_path": str(resolved_policy_path),
            "policy_sha256": policy_sha,
            "candidate_state_path": str(candidate_path),
            "candidate_state_sha256": candidate_sha_after,
            "runtime_overlay_proposal_sha256": proposal_sha,
        },
        "next_actions": [
            "Admit only the complete A+ proposal through paper-profitability-control --apply.",
            "Collect intervention-tagged candidate-forward post-cost paper outcomes before widening or removing any guard.",
            "Keep the behavior overlay paper-only; live execution remains unchanged and locked.",
        ],
    }
    return payload


def publish_payload(
    *,
    project_root: Path,
    payload: Mapping[str, Any],
    out_path: str | Path = DEFAULT_OUT_PATH,
    source: str = "paper_behavior_intervention_drill",
) -> bool:
    return safe_write_json_atomic(
        str(_resolve(project_root, out_path)),
        dict(payload),
        project_root=str(project_root),
        source=source,
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run candidate-bound paper champion-challenger drills that may propose "
            "monotonic risk-reducing behavior changes."
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
        payload["admission_eligible"] = False
        payload.setdefault("failed_checks", []).append("artifact_write_failed")
    if args.json:
        print(json.dumps(payload, ensure_ascii=True, sort_keys=True))
    else:
        summary = _mapping(payload.get("scenario_summary"))
        print(
            "paper_behavior_intervention_drill "
            f"status={payload.get('status')} grade={payload.get('control_grade')} "
            f"scenarios={summary.get('passed_scenario_count', 0)}/"
            f"{summary.get('executed_scenario_count', 0)} "
            f"blocked={summary.get('blocked_case_count', 0)} "
            f"throttled={summary.get('throttled_case_count', 0)} "
            f"admission_eligible={int(bool(payload.get('admission_eligible', False)))}"
        )
    return 0 if payload.get("ok") else 2


if __name__ == "__main__":
    raise SystemExit(main())
