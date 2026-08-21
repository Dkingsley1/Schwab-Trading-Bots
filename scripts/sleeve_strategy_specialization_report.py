#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_POLICY_PATH = PROJECT_ROOT / "config" / "sleeve_strategy_contracts_v1.json"
DEFAULT_PERFORMANCE_PATH = PROJECT_ROOT / "governance" / "health" / "paper_performance_latest.json"
DEFAULT_OUT_PATH = PROJECT_ROOT / "governance" / "research" / "sleeve_strategy_specialization_latest.json"
DEFAULT_CONTRACTS_OUT_PATH = PROJECT_ROOT / "governance" / "research" / "sleeve_strategy_contracts_latest.json"
DEFAULT_LIBRARY_OUT_PATH = PROJECT_ROOT / "governance" / "research" / "sleeve_strategy_library_latest.json"
DEFAULT_FAMILIES_OUT_PATH = PROJECT_ROOT / "governance" / "research" / "sleeve_strategy_families_latest.json"

if str(PROJECT_ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(PROJECT_ROOT))

from core.accountability import safe_write_json_atomic  # noqa: E402
from core.sleeve_strategy_specialization import (  # noqa: E402
    FORBIDDEN_AUTHORITY,
    materialize_strategy_contracts,
    materialize_strategy_library,
    strategy_regime_assessment,
    validate_policy,
)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _canonical_hash(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _display_name(value: Any) -> str:
    return " ".join(
        part.capitalize()
        for part in str(value or "").strip().replace("-", "_").split("_")
        if part
    )


def _parse_timestamp(value: Any) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _regime_context(
    project_root: Path,
    policy: Mapping[str, Any],
    generated_at: datetime,
) -> dict[str, Any]:
    regime_policy = policy.get("regime_adaptation")
    settings = dict(regime_policy) if isinstance(regime_policy, Mapping) else {}
    source_path = Path(
        str(settings.get("source_path") or "governance/health/regime_control_plane_latest.json")
    )
    if not source_path.is_absolute():
        source_path = project_root / source_path
    payload = _read_json(source_path)
    timestamp = _parse_timestamp(payload.get("timestamp_utc"))
    age_seconds = (
        max((generated_at - timestamp).total_seconds(), 0.0)
        if timestamp is not None
        else None
    )
    maximum_age = max(int(settings.get("maximum_source_age_seconds") or 3600), 1)
    fresh = bool(timestamp is not None and age_seconds is not None and age_seconds <= maximum_age)
    source_status = str(payload.get("overall_status") or "missing").strip().lower()
    current_regime = str(payload.get("regime_state") or "").strip().lower()
    return {
        "current_regime": current_regime or "unknown",
        "stance_label": str(payload.get("stance_label") or "unknown"),
        "stance_score": payload.get("stance_score"),
        "source_status": source_status,
        "source_timestamp_utc": str(payload.get("timestamp_utc") or ""),
        "source_age_seconds": round(age_seconds, 3) if age_seconds is not None else None,
        "source_fresh": fresh,
        "activation_ready": fresh and source_status == "ready" and bool(current_regime),
        "source_path": str(source_path),
        "low_confidence_behavior": str(settings.get("low_confidence_behavior") or ""),
        "authority": "ranking_research_admission_and_evidence_segmentation_only",
    }


def _quality_assessment(
    contract: Mapping[str, Any],
    evidence: Mapping[str, Any],
    lifecycle: str,
    policy: Mapping[str, Any],
) -> dict[str, Any]:
    objective = str(contract.get("objective_class") or "")
    tier = str(contract.get("library_tier") or "")
    if tier == "cold_research":
        return {
            "verdict": "cold_untested",
            "grade": "COLD",
            "quality_score": None,
            "evidence_maturity_percent": 0.0,
            "reason": "cold hypothesis has not been admitted to candidate-bound evaluation",
        }
    if objective == "control_only":
        return {
            "verdict": "control_only",
            "grade": "N/A",
            "quality_score": None,
            "evidence_maturity_percent": 100.0,
            "reason": "control effectiveness is evaluated without trading-profit claims",
        }
    if objective in {"hedge_utility", "capital_preservation"}:
        return {
            "verdict": "objective_evidence_pending",
            "grade": "NE",
            "quality_score": None,
            "evidence_maturity_percent": 0.0,
            "reason": "portfolio contribution evidence is required instead of standalone profit",
        }
    thresholds = policy.get("candidate_binding")
    settings = dict(thresholds) if isinstance(thresholds, Mapping) else {}
    probation_samples = max(int(settings.get("minimum_probation_samples") or 30), 1)
    validation_samples = max(
        int(settings.get("minimum_validation_samples") or 100), probation_samples
    )
    minimum_days = max(int(settings.get("minimum_independent_days") or 7), 1)
    minimum_symbols = max(int(settings.get("minimum_independent_symbols") or 3), 1)
    sample_count = max(int(evidence.get("sample_count") or 0), 0)
    day_count = max(int(evidence.get("independent_day_count") or 0), 0)
    symbol_count = max(int(evidence.get("independent_symbol_count") or 0), 0)
    maturity = 100.0 * min(
        sample_count / validation_samples,
        day_count / minimum_days,
        symbol_count / minimum_symbols,
        1.0,
    )
    expectancy = evidence.get("post_cost_expectancy")
    row = dict(expectancy) if isinstance(expectancy, Mapping) else {}
    mean_return = float(row.get("mean_post_cost_return_bps") or 0.0)
    lower_bound = float(
        row.get("lower_confidence_bound_95_post_cost_return_bps") or 0.0
    )
    robust_positive = bool(
        row.get("positive_clustered_lower_confidence_bound_95", False)
    )
    if lifecycle == "validated_candidate" and robust_positive:
        verdict, grade, score = "validated_good", "A+", 95.0
        reason = "positive clustered lower confidence bound after costs"
    elif lifecycle == "retirement_review":
        verdict, grade, score = "retirement_candidate", "F", 10.0
        reason = "mature negative candidate-forward expectancy and lower bound"
    elif lifecycle == "demotion_review":
        verdict, grade, score = "weak", "D", 25.0
        reason = "negative candidate-forward post-cost mean"
    elif sample_count < probation_samples:
        verdict, grade, score = "insufficient_evidence", "NE", None
        reason = f"samples below probation threshold: {sample_count}/{probation_samples}"
    elif mean_return > 0.0:
        verdict, grade, score = "promising_unconfirmed", "B", 72.0
        reason = "positive mean but robust independent confidence is pending"
    else:
        verdict, grade, score = "mixed_watch", "C", 50.0
        reason = "evidence is mature enough to inspect but not decisively positive or negative"
    return {
        "verdict": verdict,
        "grade": grade,
        "quality_score": score,
        "evidence_maturity_percent": round(maturity, 3),
        "sample_count": sample_count,
        "independent_day_count": day_count,
        "independent_symbol_count": symbol_count,
        "mean_post_cost_return_bps": mean_return if row else None,
        "lower_confidence_bound_95_post_cost_return_bps": lower_bound if row else None,
        "reason": reason,
        "policy": "unknown and insufficient evidence are never labeled bad",
    }


def _candidate_binding(performance: Mapping[str, Any]) -> dict[str, Any]:
    window = performance.get("profitability_evidence_window")
    source = dict(window) if isinstance(window, Mapping) else {}
    candidate_id = str(source.get("candidate_id") or "").strip()
    mismatch_count = max(
        int(source.get("candidate_binding_mismatch_rows_excluded") or 0), 0
    )
    reasons: list[str] = []
    if not candidate_id:
        reasons.append("candidate_id_missing")
    if not bool(source.get("candidate_binding_required", False)):
        reasons.append("candidate_binding_not_required")
    if not bool(source.get("candidate_filter_active", False)):
        reasons.append("candidate_filter_inactive")
    if mismatch_count:
        reasons.append("candidate_binding_mismatch_rows_present")
    if not str(source.get("candidate_cutoff_utc") or "").strip():
        reasons.append("candidate_cutoff_missing")
    if not str(source.get("evidence_through_utc") or "").strip():
        reasons.append("evidence_watermark_missing")
    return {
        "candidate_id": candidate_id,
        "generation": int(source.get("candidate_generation") or 0),
        "candidate_cutoff_utc": str(source.get("candidate_cutoff_utc") or ""),
        "evidence_through_utc": str(source.get("evidence_through_utc") or ""),
        "bound": not reasons,
        "mismatch_rows_excluded": mismatch_count,
        "reasons": reasons,
        "policy": "candidate-forward evidence only; historical and cross-candidate pooling are forbidden",
    }


def _evidence_rows(performance: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    rows = performance.get("strategy_latest")
    if not isinstance(rows, list):
        return {}
    return {
        str(row.get("strategy_id") or "").strip(): dict(row)
        for row in rows
        if isinstance(row, Mapping) and str(row.get("strategy_id") or "").strip()
    }


def _lifecycle(
    contract: Mapping[str, Any],
    evidence: Mapping[str, Any],
    binding: Mapping[str, Any],
    policy: Mapping[str, Any],
) -> tuple[str, list[str], bool]:
    objective = str(contract.get("objective_class") or "")
    if objective == "control_only":
        return "control_only", ["trading_profit_not_applicable"], True
    if not bool(binding.get("bound", False)):
        return "parked_candidate", ["candidate_binding_pending"], False
    if not evidence:
        return "parked_candidate", ["no_identity_bound_post_cost_samples"], False

    binding_policy = policy.get("candidate_binding")
    thresholds = dict(binding_policy) if isinstance(binding_policy, Mapping) else {}
    sample_count = max(int(evidence.get("sample_count") or 0), 0)
    day_count = max(int(evidence.get("independent_day_count") or 0), 0)
    symbol_count = max(int(evidence.get("independent_symbol_count") or 0), 0)
    probation_samples = max(int(thresholds.get("minimum_probation_samples") or 30), 1)
    validation_samples = max(int(thresholds.get("minimum_validation_samples") or 100), probation_samples)
    minimum_days = max(int(thresholds.get("minimum_independent_days") or 7), 1)
    minimum_symbols = max(int(thresholds.get("minimum_independent_symbols") or 3), 1)
    expectancy = evidence.get("post_cost_expectancy")
    expectancy_row = dict(expectancy) if isinstance(expectancy, Mapping) else {}

    if sample_count < probation_samples:
        return "parked_candidate", [f"samples_below_probation:{sample_count}/{probation_samples}"], False
    if objective in {"hedge_utility", "capital_preservation"}:
        return (
            "probation",
            [f"objective_specific_portfolio_contribution_evidence_pending:{objective}"],
            False,
        )
    if sample_count < validation_samples or day_count < minimum_days or symbol_count < minimum_symbols:
        return (
            "probation",
            [
                f"samples:{sample_count}/{validation_samples}",
                f"independent_days:{day_count}/{minimum_days}",
                f"independent_symbols:{symbol_count}/{minimum_symbols}",
            ],
            False,
        )
    if bool(expectancy_row.get("positive_clustered_lower_confidence_bound_95", False)):
        return "validated_candidate", [], True
    mean_return = float(expectancy_row.get("mean_post_cost_return_bps") or 0.0)
    if mean_return < 0.0:
        lower_bound = float(
            expectancy_row.get("lower_confidence_bound_95_post_cost_return_bps")
            or 0.0
        )
        if sample_count >= (2 * validation_samples) and lower_bound < 0.0:
            return (
                "retirement_review",
                ["mature_negative_candidate_forward_post_cost_expectancy"],
                False,
            )
        return "demotion_review", ["nonpositive_candidate_forward_post_cost_expectancy"], False
    return "watch", ["positive_mean_but_robust_confidence_pending"], False


def build_payload(
    project_root: Path,
    *,
    policy_path: Path,
    performance_path: Path,
    generated_at_utc: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    policy = _read_json(policy_path)
    validate_policy(policy)
    performance = _read_json(performance_path)
    contracts = materialize_strategy_contracts(
        policy=policy,
        project_root=project_root,
    )
    evidence_by_strategy = _evidence_rows(performance)
    binding = _candidate_binding(performance)
    generated_at = generated_at_utc or datetime.now(timezone.utc).isoformat()
    generated_datetime = _parse_timestamp(generated_at) or datetime.now(timezone.utc)
    regime_context = _regime_context(project_root, policy, generated_datetime)
    rows: list[dict[str, Any]] = []
    lifecycle_counts: dict[str, int] = {}
    quality_counts: dict[str, int] = {}
    for strategy_id, contract in contracts.items():
        evidence = evidence_by_strategy.get(strategy_id, {})
        lifecycle, blockers, objective_ready = _lifecycle(
            contract, evidence, binding, policy
        )
        lifecycle_counts[lifecycle] = lifecycle_counts.get(lifecycle, 0) + 1
        quality = _quality_assessment(contract, evidence, lifecycle, policy)
        verdict = str(quality.get("verdict") or "insufficient_evidence")
        quality_counts[verdict] = quality_counts.get(verdict, 0) + 1
        regime = strategy_regime_assessment(
            contract,
            regime_context.get("current_regime"),
            policy=policy,
            source_ready=bool(regime_context.get("source_fresh", False)),
            source_status=str(regime_context.get("source_status") or "missing"),
        )
        rows.append(
            {
                "strategy_id": strategy_id,
                "strategy_name": str(contract.get("strategy_name") or ""),
                "sleeve_id": str(contract.get("sleeve_id") or ""),
                "source_kind": str(contract.get("source_kind") or ""),
                "library_tier": str(contract.get("library_tier") or ""),
                "activation_state": str(contract.get("activation_state") or ""),
                "objective_class": str(contract.get("objective_class") or ""),
                "primary_metric": str(
                    (contract.get("objective_scorecard") or {}).get("primary_metric")
                    if isinstance(contract.get("objective_scorecard"), Mapping)
                    else ""
                ),
                "contract_complete": bool(contract.get("contract_complete", False)),
                "contract_receipt_sha256": str(
                    contract.get("contract_receipt_sha256") or ""
                ),
                "lifecycle": lifecycle,
                "objective_evidence_ready": objective_ready,
                "lifecycle_blockers": blockers,
                "sample_count": int(evidence.get("sample_count") or 0),
                "independent_day_count": int(
                    evidence.get("independent_day_count") or 0
                ),
                "independent_symbol_count": int(
                    evidence.get("independent_symbol_count") or 0
                ),
                "post_cost_expectancy": dict(
                    evidence.get("post_cost_expectancy")
                    if isinstance(evidence.get("post_cost_expectancy"), Mapping)
                    else {}
                ),
                "strategy_definition": dict(
                    contract.get("strategy_definition")
                    if isinstance(contract.get("strategy_definition"), Mapping)
                    else {}
                ),
                "regime_assessment": regime,
                "quality_assessment": quality,
                "authority": "evidence_only_no_decision_sizing_allocation_promotion_or_live_authority",
            }
        )

    authority_violations = [
        strategy_id
        for strategy_id, contract in contracts.items()
        if any(
            bool((contract.get("authority") or {}).get(key, False))
            for key in FORBIDDEN_AUTHORITY
        )
    ]
    incomplete = [
        strategy_id
        for strategy_id, contract in contracts.items()
        if not bool(contract.get("contract_complete", False))
    ]
    catalog_count = sum(
        1 for contract in contracts.values() if contract.get("source_kind") == "catalog"
    )
    addition_count = sum(
        1
        for contract in contracts.values()
        if contract.get("source_kind") == "curated_addition"
    )
    payload = {
        "timestamp_utc": generated_at,
        "schema_version": 1,
        "ok": not incomplete and not authority_violations,
        "status": "ready" if not incomplete and not authority_violations else "blocked",
        "policy_id": str(policy.get("policy_id") or ""),
        "candidate_binding": binding,
        "current_regime": regime_context,
        "contract_coverage": {
            "grade": "A+" if not incomplete and not authority_violations else "F",
            "sleeve_count": len({row["sleeve_id"] for row in rows}),
            "strategy_count": len(rows),
            "catalog_strategy_count": catalog_count,
            "curated_addition_count": addition_count,
            "complete_contract_count": len(rows) - len(incomplete),
            "incomplete_contract_count": len(incomplete),
            "authority_violation_count": len(authority_violations),
        },
        "lifecycle_counts": dict(sorted(lifecycle_counts.items())),
        "quality_summary": {
            "verdict_counts": dict(sorted(quality_counts.items())),
            "validated_good_count": quality_counts.get("validated_good", 0),
            "promising_unconfirmed_count": quality_counts.get(
                "promising_unconfirmed", 0
            ),
            "weak_count": quality_counts.get("weak", 0),
            "retirement_candidate_count": quality_counts.get(
                "retirement_candidate", 0
            ),
            "insufficient_evidence_count": quality_counts.get(
                "insufficient_evidence", 0
            ),
            "policy": "good and bad verdicts require candidate-bound objective-aware evidence; unknown is not bad",
        },
        "strategy_rows": rows,
        "limitations": [
            "complete contracts improve attribution and research discipline but do not manufacture profitable evidence",
            "broad master decisions remain ensemble_champion identities until an explicit named strategy is selected",
            "hedge and capital-preservation sleeves require portfolio-contribution evidence rather than standalone-profit grading",
            "control-only sleeves never receive trading-profit objectives",
        ],
        "authority_contract": dict(policy.get("authority") or {}),
        "action": (
            "continue candidate-bound paper collection and objective-specific attribution"
            if not binding.get("bound", False)
            else "review lifecycle blockers; do not promote automatically"
        ),
    }
    contract_payload = {
        "timestamp_utc": generated_at,
        "schema_version": 1,
        "policy_id": str(policy.get("policy_id") or ""),
        "policy_sha256": _canonical_hash(policy),
        "contract_count": len(contracts),
        "contracts_sha256": _canonical_hash(contracts),
        "contracts": contracts,
        "authority_contract": dict(policy.get("authority") or {}),
    }
    return payload, contract_payload


def build_library_payload(
    project_root: Path,
    *,
    policy_path: Path,
    hot_strategy_rows: list[dict[str, Any]],
    generated_at_utc: str | None = None,
) -> dict[str, Any]:
    policy = _read_json(policy_path)
    validate_policy(policy)
    generated_at = generated_at_utc or datetime.now(timezone.utc).isoformat()
    generated_datetime = _parse_timestamp(generated_at) or datetime.now(timezone.utc)
    regime_context = _regime_context(project_root, policy, generated_datetime)
    library = materialize_strategy_library(policy=policy, project_root=project_root)
    hot_quality = {
        str(row.get("strategy_id") or ""): dict(row.get("quality_assessment") or {})
        for row in hot_strategy_rows
        if isinstance(row, Mapping)
    }
    rows: list[dict[str, Any]] = []
    sleeve_summary: dict[str, dict[str, Any]] = {}
    tier_counts: dict[str, int] = {}
    verdict_counts: dict[str, int] = {}
    relevance_counts: dict[str, int] = {}
    cold_activation_eligible_count = 0
    authority_violations = 0
    incomplete = 0
    for strategy_id, contract in library.items():
        tier = str(contract.get("library_tier") or "")
        definition = dict(contract.get("strategy_definition") or {})
        quality = hot_quality.get(strategy_id)
        if quality is None:
            quality = _quality_assessment(contract, {}, "cold_research", policy)
        regime = strategy_regime_assessment(
            contract,
            regime_context.get("current_regime"),
            policy=policy,
            source_ready=bool(regime_context.get("source_fresh", False)),
            source_status=str(regime_context.get("source_status") or "missing"),
        )
        verdict = str(quality.get("verdict") or "cold_untested")
        relevance = str(regime.get("relevance") or "unknown")
        tier_counts[tier] = tier_counts.get(tier, 0) + 1
        verdict_counts[verdict] = verdict_counts.get(verdict, 0) + 1
        relevance_counts[relevance] = relevance_counts.get(relevance, 0) + 1
        if not bool(contract.get("contract_complete", False)):
            incomplete += 1
        if any(
            bool((contract.get("authority") or {}).get(key, False))
            for key in FORBIDDEN_AUTHORITY
        ):
            authority_violations += 1
        sleeve_id = str(contract.get("sleeve_id") or "")
        sleeve = sleeve_summary.setdefault(
            sleeve_id,
            {
                "sleeve_id": sleeve_id,
                "strategy_count": 0,
                "hot_count": 0,
                "cold_count": 0,
                "regime_aligned_count": 0,
                "regime_guarded_count": 0,
                "cold_activation_eligible_count": 0,
                "verdict_counts": {},
            },
        )
        sleeve["strategy_count"] += 1
        if tier == "cold_research":
            sleeve["cold_count"] += 1
        else:
            sleeve["hot_count"] += 1
        if relevance == "aligned":
            sleeve["regime_aligned_count"] += 1
        if relevance == "guarded":
            sleeve["regime_guarded_count"] += 1
        if bool(regime.get("cold_activation_eligible", False)):
            sleeve["cold_activation_eligible_count"] += 1
            cold_activation_eligible_count += 1
        sleeve_verdicts = sleeve["verdict_counts"]
        sleeve_verdicts[verdict] = sleeve_verdicts.get(verdict, 0) + 1
        rows.append(
            {
                "strategy_id": strategy_id,
                "strategy_name": str(contract.get("strategy_name") or ""),
                "display_name": str(contract.get("display_name") or ""),
                "sleeve_id": sleeve_id,
                "objective_class": str(contract.get("objective_class") or ""),
                "source_kind": str(contract.get("source_kind") or ""),
                "library_tier": tier,
                "activation_state": str(contract.get("activation_state") or ""),
                "signal_family": str(definition.get("signal_family") or "general"),
                "archetype": str(definition.get("archetype") or ""),
                "conditioning_overlay": str(definition.get("conditioning_overlay") or ""),
                "plain_language_summary": str(definition.get("plain_language_summary") or ""),
                "ideal_regimes": list(definition.get("ideal_regimes") or []),
                "hostile_regimes": list(definition.get("hostile_regimes") or []),
                "expected_failure_modes": list(definition.get("expected_failure_modes") or []),
                "regime_assessment": regime,
                "quality_assessment": quality,
                "contract_receipt_sha256": str(contract.get("contract_receipt_sha256") or ""),
                "authority": "research_catalog_only_no_decision_sizing_allocation_promotion_or_live_authority",
            }
        )
    sleeve_counts = [row["strategy_count"] for row in sleeve_summary.values()]
    return {
        "timestamp_utc": generated_at,
        "schema_version": 1,
        "ok": incomplete == 0 and authority_violations == 0 and len(rows) == 12000,
        "status": "ready" if incomplete == 0 and authority_violations == 0 and len(rows) == 12000 else "blocked",
        "policy_id": str(policy.get("policy_id") or ""),
        "library_contract": {
            "target_total_strategies": 12000,
            "strategy_count": len(rows),
            "sleeve_count": len(sleeve_summary),
            "minimum_strategies_per_sleeve": min(sleeve_counts) if sleeve_counts else 0,
            "maximum_strategies_per_sleeve": max(sleeve_counts) if sleeve_counts else 0,
            "hot_strategy_count": len(rows) - tier_counts.get("cold_research", 0),
            "cold_strategy_count": tier_counts.get("cold_research", 0),
            "complete_contract_count": len(rows) - incomplete,
            "incomplete_contract_count": incomplete,
            "authority_violation_count": authority_violations,
            "runtime_materialization": "hot_catalog_only",
            "full_library_materialization": "report_and_research_on_demand",
        },
        "current_regime": regime_context,
        "tier_counts": dict(sorted(tier_counts.items())),
        "quality_summary": {
            "verdict_counts": dict(sorted(verdict_counts.items())),
            "policy": "cold and insufficient evidence are unknown, not bad",
        },
        "regime_relevance_counts": dict(sorted(relevance_counts.items())),
        "regime_activation_summary": {
            "cold_activation_eligible_count": cold_activation_eligible_count,
            "cold_activation_blocked_count": tier_counts.get("cold_research", 0) - cold_activation_eligible_count,
            "activation_ready": bool(regime_context.get("activation_ready", False)),
            "policy": "cold admission requires a fresh ready regime and still needs explicit candidate-bound governance approval",
        },
        "sleeves": [
            {**row, "verdict_counts": dict(sorted(row["verdict_counts"].items()))}
            for _, row in sorted(sleeve_summary.items())
        ],
        "strategies": rows,
        "authority_contract": dict(policy.get("authority") or {}),
        "interpretation": (
            "The full library is a bounded hypothesis inventory. Only hot contracts are "
            "available to runtime ranking, and cold admission requires a fresh ready regime, "
            "candidate-bound evidence, capacity, and explicit governance approval."
        ),
    }


def _parent_contract(contract: Mapping[str, Any], *, failure_modes: list[str]) -> dict[str, Any]:
    definition = dict(contract.get("strategy_definition") or {})
    return {
        "economic_thesis": str(contract.get("economic_thesis") or ""),
        "edge_hypothesis": str(definition.get("edge_hypothesis") or ""),
        "signal_family": str(definition.get("signal_family") or "general"),
        "taxonomy_groups": list(definition.get("taxonomy_groups") or []),
        "required_inputs": list(definition.get("required_inputs") or []),
        "ideal_regimes": list(definition.get("ideal_regimes") or []),
        "hostile_regimes": list(definition.get("hostile_regimes") or []),
        "expected_failure_modes": failure_modes,
        "evaluation_question": str(definition.get("evaluation_question") or ""),
        "objective_scorecard": dict(contract.get("objective_scorecard") or {}),
        "label_definition": str(contract.get("label_definition") or ""),
        "decision_horizon": str(contract.get("decision_horizon") or ""),
        "holding_horizon": str(contract.get("holding_horizon") or ""),
        "entry_rule": str(contract.get("entry_rule") or ""),
        "add_rule": str(contract.get("add_rule") or ""),
        "trim_rule": str(contract.get("trim_rule") or ""),
        "exit_rule": str(contract.get("exit_rule") or ""),
        "time_stop": str(contract.get("time_stop") or ""),
        "benchmark": str(contract.get("benchmark") or ""),
        "benchmark_rule": str(definition.get("benchmark_rule") or ""),
        "cost_model": str(contract.get("cost_model") or ""),
        "capacity_method": str(contract.get("capacity_method") or ""),
        "risk_budget": str(contract.get("risk_budget") or ""),
        "shorting_policy": str(contract.get("shorting_policy") or ""),
        "evidence_policy": str(contract.get("evidence_policy") or ""),
        "lifecycle_policy": str(contract.get("lifecycle_policy") or ""),
    }


def _shared_failure_modes(contracts: list[Mapping[str, Any]]) -> list[str]:
    rows = [
        set(
            str(item)
            for item in (dict(contract.get("strategy_definition") or {}).get("expected_failure_modes") or [])
        )
        for contract in contracts
    ]
    if not rows:
        return []
    return sorted(set.intersection(*rows))


def build_family_payload(
    project_root: Path,
    *,
    policy_path: Path,
    hot_strategy_rows: list[dict[str, Any]],
    library_rows: list[dict[str, Any]],
    generated_at_utc: str | None = None,
    library_contracts: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Consolidate cold variants for inspection without changing runtime identities."""
    policy = _read_json(policy_path)
    validate_policy(policy)
    generated_at = generated_at_utc or datetime.now(timezone.utc).isoformat()
    contracts = (
        dict(library_contracts)
        if isinstance(library_contracts, Mapping)
        else materialize_strategy_library(policy=policy, project_root=project_root)
    )
    row_by_id = {
        str(row.get("strategy_id") or ""): row
        for row in library_rows
        if isinstance(row, Mapping) and str(row.get("strategy_id") or "")
    }
    hot_row_by_id = {
        str(row.get("strategy_id") or ""): row
        for row in hot_strategy_rows
        if isinstance(row, Mapping) and str(row.get("strategy_id") or "")
    }
    overlay_policy = dict((policy.get("strategy_library") or {}).get("conditioning_overlays") or {})
    configured_conditions = [
        {
            "condition_id": str(name),
            "confirmation": str((definition or {}).get("confirmation") or ""),
            "failure_mode": str((definition or {}).get("failure_mode") or ""),
        }
        for name, definition in sorted(overlay_policy.items())
    ]
    configured_condition_ids = [row["condition_id"] for row in configured_conditions]

    cold_groups: dict[tuple[str, str, str], list[Mapping[str, Any]]] = {}
    hot_contracts: list[Mapping[str, Any]] = []
    for contract in contracts.values():
        if str(contract.get("library_tier") or "") == "cold_research":
            definition = dict(contract.get("strategy_definition") or {})
            key = (
                str(contract.get("sleeve_id") or ""),
                str(contract.get("objective_class") or ""),
                str(definition.get("archetype") or ""),
            )
            cold_groups.setdefault(key, []).append(contract)
        else:
            hot_contracts.append(contract)

    families: list[dict[str, Any]] = []
    objective_counts: dict[str, int] = {}
    condition_materialized_counts = {name: 0 for name in configured_condition_ids}

    def child_receipt(contract: Mapping[str, Any]) -> dict[str, Any]:
        strategy_id = str(contract.get("strategy_id") or "")
        definition = dict(contract.get("strategy_definition") or {})
        report_row = row_by_id.get(strategy_id) or hot_row_by_id.get(strategy_id) or {}
        quality = dict(report_row.get("quality_assessment") or {})
        regime = dict(report_row.get("regime_assessment") or {})
        return {
            "strategy_id": strategy_id,
            "strategy_name": str(contract.get("strategy_name") or ""),
            "display_name": str(contract.get("display_name") or ""),
            "library_tier": str(contract.get("library_tier") or ""),
            "activation_state": str(contract.get("activation_state") or ""),
            "conditioning_overlay": str(definition.get("conditioning_overlay") or "native_strategy_logic"),
            "confirmation_requirement": str(definition.get("confirmation_requirement") or ""),
            "overlay_failure_modes": sorted(
                set(str(item) for item in (definition.get("expected_failure_modes") or []))
                - set(_shared_failure_modes([contract]))
            ),
            "contract_receipt_sha256": str(contract.get("contract_receipt_sha256") or ""),
            "evidence": {
                "quality_verdict": str(quality.get("verdict") or "unknown"),
                "quality_grade": str(quality.get("grade") or "unknown"),
                "evidence_maturity_percent": quality.get("evidence_maturity_percent"),
                "sample_count": quality.get("sample_count"),
                "independent_day_count": quality.get("independent_day_count"),
                "independent_symbol_count": quality.get("independent_symbol_count"),
                "regime_relevance": str(regime.get("relevance") or "unknown"),
                "cold_activation_eligible": bool(regime.get("cold_activation_eligible", False)),
                "source_artifact": "governance/research/sleeve_strategy_library_latest.json",
                "lookup_key": strategy_id,
            },
            "evidence_is_variant_specific": True,
        }

    for contract in sorted(hot_contracts, key=lambda row: str(row.get("strategy_id") or "")):
        strategy_id = str(contract.get("strategy_id") or "")
        definition = dict(contract.get("strategy_definition") or {})
        child = child_receipt(contract)
        quality = dict(child.get("evidence") or {})
        families.append(
            {
                "family_id": strategy_id,
                "family_kind": "native_hot_identity",
                "family_name": str(contract.get("display_name") or contract.get("strategy_name") or ""),
                "sleeve_id": str(contract.get("sleeve_id") or ""),
                "objective_class": str(contract.get("objective_class") or ""),
                "archetype": str(definition.get("archetype") or contract.get("strategy_name") or ""),
                "library_tier": str(contract.get("library_tier") or ""),
                "parent_contract": _parent_contract(
                    contract,
                    failure_modes=list(definition.get("expected_failure_modes") or []),
                ),
                "supported_conditions": ["native_strategy_logic"],
                "materialized_conditions": ["native_strategy_logic"],
                "unmaterialized_conditions": [],
                "variant_count": 1,
                "child_variants": [child],
                "family_evidence": {
                    "verdict_counts": {
                        str(quality.get("quality_verdict") or "unknown"): 1,
                    },
                    "parent_verdict": "native_identity_uses_its_own_child_evidence",
                    "evidence_pooling_allowed": False,
                },
                "runtime_identity_preserved": True,
                "authority": "read_only_catalog_no_runtime_or_execution_authority",
            }
        )

    for (sleeve_id, objective_class, archetype), group in sorted(cold_groups.items()):
        ordered = sorted(group, key=lambda row: str(row.get("strategy_id") or ""))
        first = ordered[0]
        shared_failures = _shared_failure_modes(ordered)
        children = []
        verdict_counts: dict[str, int] = {}
        materialized: list[str] = []
        for contract in ordered:
            child = child_receipt(contract)
            definition = dict(contract.get("strategy_definition") or {})
            all_failures = set(str(item) for item in (definition.get("expected_failure_modes") or []))
            child["overlay_failure_modes"] = sorted(all_failures - set(shared_failures))
            overlay = str(child.get("conditioning_overlay") or "")
            if overlay:
                materialized.append(overlay)
                if overlay in condition_materialized_counts:
                    condition_materialized_counts[overlay] += 1
            verdict = str((child.get("evidence") or {}).get("quality_verdict") or "unknown")
            verdict_counts[verdict] = verdict_counts.get(verdict, 0) + 1
            children.append(child)
        materialized = sorted(set(materialized))
        objective_counts[objective_class] = objective_counts.get(objective_class, 0) + 1
        family_id = f"family::{sleeve_id}::{archetype}::v1"
        families.append(
            {
                "family_id": family_id,
                "family_kind": "cold_generated_parent",
                "family_name": f"{_display_name(archetype)} Research Family",
                "sleeve_id": sleeve_id,
                "objective_class": objective_class,
                "archetype": archetype,
                "library_tier": "cold_research",
                "parent_contract": _parent_contract(first, failure_modes=shared_failures),
                "supported_conditions": configured_condition_ids,
                "materialized_conditions": materialized,
                "unmaterialized_conditions": sorted(set(configured_condition_ids) - set(materialized)),
                "variant_count": len(children),
                "child_variants": children,
                "family_evidence": {
                    "verdict_counts": dict(sorted(verdict_counts.items())),
                    "parent_verdict": "not_scored_at_parent_level",
                    "evidence_pooling_allowed": False,
                    "policy": "each child variant must earn candidate-bound evidence independently",
                },
                "runtime_identity_preserved": True,
                "authority": "read_only_catalog_no_runtime_or_execution_authority",
            }
        )

    families.sort(key=lambda row: str(row.get("family_id") or ""))
    lineage_ids = [
        str(child.get("strategy_id") or "")
        for family in families
        for child in (family.get("child_variants") or [])
        if isinstance(child, Mapping)
    ]
    unique_ids = set(lineage_ids)
    hot_count = len(hot_contracts)
    cold_parent_count = len(cold_groups)
    cold_variant_count = sum(len(group) for group in cold_groups.values())
    canonical_count = len(families)
    all_conditions_supported = all(
        set(family.get("supported_conditions") or []) == set(configured_condition_ids)
        for family in families
        if family.get("family_kind") == "cold_generated_parent"
    )
    lineage_complete = len(lineage_ids) == len(unique_ids) == len(contracts) == 12000
    exact_consolidation = (
        hot_count == 879
        and cold_variant_count == 11121
        and cold_parent_count == 1110
        and canonical_count == 1989
    )
    return {
        "timestamp_utc": generated_at,
        "schema_version": 1,
        "ok": lineage_complete and exact_consolidation and all_conditions_supported,
        "status": "ready" if lineage_complete and exact_consolidation and all_conditions_supported else "blocked",
        "policy_id": str(policy.get("policy_id") or ""),
        "consolidation_contract": {
            "conceptual_strategy_count": len(contracts),
            "canonical_record_count": canonical_count,
            "native_hot_family_count": hot_count,
            "cold_parent_family_count": cold_parent_count,
            "cold_child_variant_count": cold_variant_count,
            "lineage_covered_strategy_count": len(unique_ids),
            "lineage_missing_count": max(len(contracts) - len(unique_ids), 0),
            "lineage_duplicate_count": max(len(lineage_ids) - len(unique_ids), 0),
            "runtime_identity_change_count": 0,
            "evidence_pooling_allowed": False,
            "runtime_authority": False,
            "paper_or_live_behavior_changed": False,
        },
        "condition_coverage": {
            "configured_condition_count": len(configured_conditions),
            "configured_conditions": configured_conditions,
            "materialized_parent_counts": dict(sorted(condition_materialized_counts.items())),
            "all_cold_parent_families_support_all_conditions": all_conditions_supported,
            "supported_only_conditions_are_not_materialized_strategies": True,
        },
        "cold_parent_family_counts_by_objective": dict(sorted(objective_counts.items())),
        "sections": {
            "trading_family_count": sum(1 for row in families if row.get("objective_class") != "control_only"),
            "control_family_count": sum(1 for row in families if row.get("objective_class") == "control_only"),
        },
        "families": families,
        "authority_contract": {
            "read_only_primary_human_view": True,
            "can_change_runtime_strategy_id": False,
            "can_pool_variant_evidence": False,
            "can_activate_cold_strategy": False,
            "can_create_or_modify_order": False,
            "can_promote_candidate": False,
            "can_submit_live_order": False,
        },
        "interpretation": (
            "The catalog presents 12,000 preserved strategy identities as 1,989 canonical records. "
            "Cold variants remain separately evidenced child receipts under 1,110 parent families; "
            "the 879 native hot identities and all paper/live behavior remain unchanged."
        ),
    }

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Materialize and score candidate-bound sleeve strategy contracts."
    )
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--policy", default=str(DEFAULT_POLICY_PATH))
    parser.add_argument("--performance", default=str(DEFAULT_PERFORMANCE_PATH))
    parser.add_argument("--out", default=str(DEFAULT_OUT_PATH))
    parser.add_argument("--contracts-out", default=str(DEFAULT_CONTRACTS_OUT_PATH))
    parser.add_argument("--library-out", default=str(DEFAULT_LIBRARY_OUT_PATH))
    parser.add_argument("--families-out", default=str(DEFAULT_FAMILIES_OUT_PATH))
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    payload, contract_payload = build_payload(
        project_root,
        policy_path=Path(args.policy).resolve(),
        performance_path=Path(args.performance).resolve(),
    )
    library_payload = build_library_payload(
        project_root,
        policy_path=Path(args.policy).resolve(),
        hot_strategy_rows=list(payload.get("strategy_rows") or []),
        generated_at_utc=str(payload.get("timestamp_utc") or ""),
    )
    family_payload = build_family_payload(
        project_root,
        policy_path=Path(args.policy).resolve(),
        hot_strategy_rows=list(payload.get("strategy_rows") or []),
        library_rows=list(library_payload.get("strategies") or []),
        generated_at_utc=str(payload.get("timestamp_utc") or ""),
    )
    payload["strategy_library"] = {
        **dict(library_payload.get("library_contract") or {}),
        "artifact_path": str(Path(args.library_out).resolve()),
        "status": str(library_payload.get("status") or "missing"),
    }
    payload["strategy_families"] = {
        **dict(family_payload.get("consolidation_contract") or {}),
        "artifact_path": str(Path(args.families_out).resolve()),
        "status": str(family_payload.get("status") or "missing"),
    }
    safe_write_json_atomic(
        str(Path(args.out).resolve()),
        payload,
        project_root=str(project_root),
        source="sleeve_strategy_specialization_report",
    )
    safe_write_json_atomic(
        str(Path(args.contracts_out).resolve()),
        contract_payload,
        project_root=str(project_root),
        source="sleeve_strategy_specialization_report.contracts",
    )
    safe_write_json_atomic(
        str(Path(args.library_out).resolve()),
        library_payload,
        project_root=str(project_root),
        source="sleeve_strategy_specialization_report.library",
    )
    safe_write_json_atomic(
        str(Path(args.families_out).resolve()),
        family_payload,
        project_root=str(project_root),
        source="sleeve_strategy_specialization_report.families",
    )
    if args.json:
        print(json.dumps(payload, ensure_ascii=True, sort_keys=True))
    else:
        coverage = payload["contract_coverage"]
        print(
            "[strategy-specialization] "
            f"status={payload['status']} grade={coverage['grade']} "
            f"sleeves={coverage['sleeve_count']} "
            f"strategies={coverage['strategy_count']} "
            f"library={payload['strategy_library'].get('strategy_count', 0)}/12000 "
            f"families={payload['strategy_families'].get('canonical_record_count', 0)}/1989 "
            f"candidate_bound={payload['candidate_binding']['bound']} "
            f"validated={payload['lifecycle_counts'].get('validated_candidate', 0)} "
            f"regime={payload['current_regime'].get('current_regime', 'unknown')}"
        )
    return 0 if payload.get("ok", False) and library_payload.get("ok", False) and family_payload.get("ok", False) else 2


if __name__ == "__main__":
    raise SystemExit(main())
