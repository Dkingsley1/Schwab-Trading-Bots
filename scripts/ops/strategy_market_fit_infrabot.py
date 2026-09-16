#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

if __package__ in {None, ""}:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from scripts.ops.long_runtime_common import load_json, write_payload
else:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    from .long_runtime_common import load_json, write_payload


DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "strategy_market_fit_infrabot_v1.json"
DEFAULT_OUT_PATH = (
    PROJECT_ROOT / "governance" / "health" / "strategy_market_fit_infrabot_latest.json"
)
DEFAULT_COHORT_OUT_PATH = (
    PROJECT_ROOT
    / "governance"
    / "research"
    / "strategy_shadow_challenger_cohort_latest.json"
)


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _number(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _utc(value: Any) -> datetime | None:
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


def _hash(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _file_receipt(path: Path) -> dict[str, Any]:
    try:
        stat = path.stat()
    except OSError:
        return {
            "path": str(path),
            "present": False,
            "size_bytes": 0,
            "mtime_ns": 0,
        }
    return {
        "path": str(path),
        "present": True,
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def _resolve(root: Path, value: Any, fallback: str) -> Path:
    path = Path(str(value or fallback)).expanduser()
    return path if path.is_absolute() else root / path


def _age_seconds(payload: Mapping[str, Any], now: datetime) -> float | None:
    timestamp = _utc(
        payload.get("timestamp_utc")
        or payload.get("generated_at_utc")
        or payload.get("updated_at_utc")
    )
    return max((now - timestamp).total_seconds(), 0.0) if timestamp else None


def _source_signature(
    paths: Mapping[str, Path],
    *,
    policy_id: str,
    candidate_hint: str,
    freshness_state: Mapping[str, Any],
) -> tuple[str, dict[str, Any]]:
    receipts = {name: _file_receipt(path) for name, path in sorted(paths.items())}
    signature = _hash(
        {
            "policy_id": policy_id,
            "candidate_hint": candidate_hint,
            "freshness_state": dict(freshness_state),
            "receipts": receipts,
        }
    )
    return signature, receipts


def _validate_authority(policy: Mapping[str, Any]) -> dict[str, bool]:
    authority = _as_dict(policy.get("authority"))
    if not authority:
        raise ValueError("strategy market-fit infrabot requires an authority contract")
    enabled = sorted(key for key, value in authority.items() if bool(value))
    if enabled:
        raise ValueError(f"forbidden strategy market-fit authority: {enabled}")
    return {str(key): False for key in authority}


def _candidate_binding(
    alpha_inputs: Mapping[str, Any],
    alpha_generation: Mapping[str, Any],
    paper_performance: Mapping[str, Any],
) -> dict[str, Any]:
    alpha_binding = _as_dict(alpha_inputs.get("candidate_binding"))
    generation_binding = _as_dict(alpha_generation.get("candidate_binding"))
    performance_window = _as_dict(
        paper_performance.get("profitability_evidence_window")
    )
    identities = {
        "alpha_inputs": str(
            alpha_inputs.get("candidate_id") or alpha_binding.get("candidate_id") or ""
        ).strip(),
        "alpha_generation": str(generation_binding.get("candidate_id") or "").strip(),
        "paper_performance": str(performance_window.get("candidate_id") or "").strip(),
    }
    present = sorted({value for value in identities.values() if value})
    blockers: list[str] = []
    if not present:
        blockers.append("candidate_identity_missing")
    if len(present) > 1:
        blockers.append("candidate_identity_mismatch")
    if alpha_inputs and not bool(alpha_binding.get("bound", False)):
        blockers.append("alpha_input_candidate_binding_not_ready")
    return {
        "candidate_id": present[0] if len(present) == 1 else "",
        "identities": identities,
        "bound": not blockers,
        "blockers": blockers,
        "historical_relabeling_allowed": False,
        "cross_candidate_pooling_allowed": False,
    }


def _regime_context(
    payload: Mapping[str, Any], *, now: datetime, maximum_age_seconds: int
) -> dict[str, Any]:
    age = _age_seconds(payload, now)
    status = str(payload.get("overall_status") or payload.get("status") or "missing")
    status = status.strip().lower()
    regime = str(payload.get("regime_state") or payload.get("current_regime") or "")
    regime = regime.strip().lower()
    fresh = bool(regime and age is not None and age <= max(int(maximum_age_seconds), 1))
    trusted = bool(fresh and status == "ready")
    return {
        "current_regime": regime or "unknown",
        "stance_label": str(payload.get("stance_label") or "unknown"),
        "stance_score": _number(payload.get("stance_score")),
        "source_status": status,
        "source_age_seconds": round(age, 3) if age is not None else None,
        "fresh": fresh,
        "trusted_for_shadow_admission": trusted,
        "screening_allowed": bool(regime and fresh),
        "policy": "thin or stale regime evidence may rank hypotheses provisionally but may not admit a cold strategy",
    }


def _sleeve_forecast_evidence(
    alpha_inputs: Mapping[str, Any], *, minimum_observations: int
) -> dict[str, dict[str, Any]]:
    routing = _as_dict(
        _as_dict(alpha_inputs.get("diagnostics")).get(
            "regime_and_decay_research_routing"
        )
    )
    grouped: dict[str, list[tuple[int, float]]] = defaultdict(list)
    for raw in _as_list(routing.get("routes")):
        row = _as_dict(raw)
        count = max(int(row.get("observation_count") or 0), 0)
        rank_ic = _number(row.get("rank_ic"))
        sleeve = str(row.get("sleeve") or "").strip()
        if sleeve and rank_ic is not None and count >= minimum_observations:
            grouped[sleeve].append((count, rank_ic))
    result: dict[str, dict[str, Any]] = {}
    for sleeve, values in grouped.items():
        total = sum(count for count, _ in values)
        weighted_ic = sum(count * value for count, value in values) / max(total, 1)
        result[sleeve] = {
            "available": True,
            "observation_count": total,
            "weighted_rank_ic": round(weighted_ic, 8),
            "score": round(min(max((weighted_ic + 1.0) / 2.0, 0.0), 1.0), 8),
            "evidence_class": "candidate_bound_counterfactual_forecast",
            "counts_as_trade_profit": False,
        }
    return result


def _derived_relevance(
    row: Mapping[str, Any],
    *,
    regime: str,
    strategy_policy: Mapping[str, Any],
) -> tuple[str, str]:
    existing = _as_dict(row.get("regime_assessment"))
    existing_regime = str(existing.get("current_regime") or "").strip().lower()
    existing_relevance = str(existing.get("relevance") or "").strip().lower()
    if existing_regime == regime and existing_relevance:
        return existing_relevance, "library_exact_regime_assessment"

    adaptation = _as_dict(strategy_policy.get("regime_adaptation"))
    affinity = _as_dict(_as_dict(adaptation.get("regime_affinity")).get(regime))
    favored = {str(value) for value in _as_list(affinity.get("favored_taxonomy"))}
    guarded = {str(value) for value in _as_list(affinity.get("guarded_taxonomy"))}
    groups = {str(row.get("signal_family") or "general")}
    overlay = str(row.get("conditioning_overlay") or "")
    if overlay in {"stress_tested", "tail_risk_aware"}:
        groups.add("risk_control")
    if str(row.get("objective_class") or "") == "control_only":
        return "aligned", "control_only_operational_relevance"
    if groups & favored:
        return "aligned", "compact_contract_taxonomy_matches_current_regime"
    if groups & guarded:
        return "guarded", "compact_contract_taxonomy_guarded_in_current_regime"
    return "neutral", "compact_contract_has_no_direct_regime_affinity"


def _score_strategy(
    row: Mapping[str, Any],
    *,
    regime: str,
    strategy_policy: Mapping[str, Any],
    policy: Mapping[str, Any],
    sleeve_evidence: Mapping[str, Mapping[str, Any]],
    source_integrity_score: float,
    capacity_clear: bool,
    candidate_bound: bool,
    current_conditions_trusted: bool,
) -> dict[str, Any]:
    scoring = _as_dict(policy.get("scoring"))
    weights = _as_dict(scoring.get("weights"))
    regime_scores = _as_dict(scoring.get("regime_scores"))
    quality_scores = _as_dict(scoring.get("quality_scores"))
    relevance, relevance_source = _derived_relevance(
        row, regime=regime, strategy_policy=strategy_policy
    )
    quality = _as_dict(row.get("quality_assessment"))
    verdict = str(quality.get("verdict") or "cold_untested")
    regime_score = float(regime_scores.get(relevance, 0.0) or 0.0)
    quality_score = float(quality_scores.get(verdict, 0.0) or 0.0)
    sleeve_id = str(row.get("sleeve_id") or "")
    evidence = dict(sleeve_evidence.get(sleeve_id) or {})
    evidence_score = float(evidence.get("score") or 0.0) if evidence else 0.0
    contract_score = 1.0 if str(row.get("contract_receipt_sha256") or "") else 0.0
    components = {
        "regime_fit": regime_score,
        "candidate_forecast_evidence": evidence_score,
        "quality_maturity": quality_score,
        "source_integrity": source_integrity_score,
        "contract_integrity": contract_score,
    }
    score = sum(
        float(weights.get(name, 0.0) or 0.0) * value
        for name, value in components.items()
    )
    score = min(max(score, 0.0), 1.0)
    if score >= 0.75:
        band = "high_research_priority"
    elif score >= 0.55:
        band = "observe"
    elif score >= 0.35:
        band = "low_research_priority"
    else:
        band = "parked"
    expectancy = _as_dict(row.get("post_cost_expectancy"))
    lower_bound = _number(quality.get("lower_confidence_bound_95_post_cost_return_bps"))
    robust_positive = bool(
        expectancy.get("positive_clustered_lower_confidence_bound_95", False)
        or (
            verdict == "validated_good"
            and lower_bound is not None
            and lower_bound > 0.0
        )
    )
    proven_working = bool(
        verdict == "validated_good"
        and robust_positive
        and capacity_clear
        and candidate_bound
        and current_conditions_trusted
    )
    archetype = str(row.get("archetype") or row.get("strategy_name") or "")
    return {
        "strategy_id": str(row.get("strategy_id") or ""),
        "strategy_name": str(row.get("strategy_name") or ""),
        "sleeve_id": sleeve_id,
        "family_id": f"family::{sleeve_id}::{archetype}::v1",
        "archetype": archetype,
        "signal_family": str(row.get("signal_family") or "general"),
        "objective_class": str(row.get("objective_class") or ""),
        "library_tier": str(row.get("library_tier") or ""),
        "activation_state": str(row.get("activation_state") or ""),
        "conditioning_overlay": str(row.get("conditioning_overlay") or ""),
        "regime_relevance": relevance,
        "regime_relevance_source": relevance_source,
        "quality_verdict": verdict,
        "post_cost_lower_confidence_bound_bps": lower_bound,
        "market_fit_score": round(score, 8),
        "market_fit_band": band,
        "score_components": {key: round(value, 8) for key, value in components.items()},
        "candidate_forecast_evidence": evidence,
        "proven_working_now": proven_working,
        "profitability_claim_allowed": proven_working,
        "contract_receipt_sha256": str(row.get("contract_receipt_sha256") or ""),
        "authority": "research_priority_only_no_action_sizing_promotion_or_execution_authority",
    }


def _batch_receipts(
    rows: Sequence[Mapping[str, Any]], batch_size: int
) -> list[dict[str, Any]]:
    receipts: list[dict[str, Any]] = []
    size = max(int(batch_size), 1)
    for start in range(0, len(rows), size):
        batch = rows[start : start + size]
        ids = [str(row.get("strategy_id") or "") for row in batch]
        scores = [float(row.get("market_fit_score") or 0.0) for row in batch]
        receipts.append(
            {
                "batch": len(receipts) + 1,
                "start_index": start,
                "end_index_exclusive": start + len(batch),
                "strategy_count": len(batch),
                "strategy_ids_sha256": _hash(ids),
                "maximum_market_fit_score": round(max(scores), 8) if scores else None,
                "minimum_market_fit_score": round(min(scores), 8) if scores else None,
            }
        )
    return receipts


def _ranked_subset(
    rows: Sequence[dict[str, Any]], *, limit: int
) -> list[dict[str, Any]]:
    trading_rows = [
        row for row in rows if str(row.get("objective_class") or "") != "control_only"
    ]
    return sorted(
        trading_rows,
        key=lambda row: (-float(row["market_fit_score"]), str(row["strategy_id"])),
    )[: max(int(limit), 0)]


def _best_grouped(
    rows: Sequence[dict[str, Any]], *, key: str, limit: int
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if str(row.get("objective_class") or "") != "control_only":
            grouped[str(row.get(key) or "unknown")].append(row)
    selected: list[dict[str, Any]] = []
    for group_id, group_rows in sorted(grouped.items()):
        ordered = _ranked_subset(group_rows, limit=limit)
        for rank, row in enumerate(ordered, start=1):
            selected.append({**row, "group_id": group_id, "group_rank": rank})
    return selected


def _infrabot_rows(
    *,
    contract_ready: bool,
    regime_trusted: bool,
    cohort_ready: bool,
    proven_count: int,
    drift_detected: bool,
) -> list[dict[str, Any]]:
    return [
        {
            "infrabot_id": "strategy_market_fit_scanner_infrabot",
            "status": "ready" if contract_ready else "blocked",
            "responsibility": "Evaluate every catalog strategy against the current regime and candidate-bound research evidence in bounded batches.",
        },
        {
            "infrabot_id": "challenger_cohort_curator_infrabot",
            "status": "shadow_ready" if cohort_ready and regime_trusted else "queued",
            "responsibility": "Maintain exactly five existing strategy contracts as shadow challengers without changing active strategy selection.",
        },
        {
            "infrabot_id": "strategy_evidence_guard_infrabot",
            "status": "clear" if proven_count else "holding",
            "responsibility": "Prevent market-fit rankings from being represented as profitability or execution approval.",
        },
        {
            "infrabot_id": "strategy_market_drift_sentinel_infrabot",
            "status": "changed" if drift_detected else "watching",
            "responsibility": "Detect regime and top-ranking turnover while preserving the prior receipt for comparison.",
        },
    ]


def _cohort_payload(
    payload: Mapping[str, Any], *, source_artifact: Path = DEFAULT_OUT_PATH
) -> dict[str, Any]:
    cohort = _as_dict(payload.get("challenger_cohort"))
    return {
        "timestamp_utc": str(payload.get("timestamp_utc") or ""),
        "schema_version": 1,
        "policy_id": str(payload.get("policy_id") or ""),
        "overall_status": str(cohort.get("status") or "blocked"),
        "candidate_binding": dict(payload.get("candidate_binding") or {}),
        "current_regime": dict(payload.get("current_regime") or {}),
        "cohort_mode": str(cohort.get("mode") or ""),
        "slot_count": int(cohort.get("slot_count") or 0),
        "maximum_slots": int(cohort.get("maximum_slots") or 0),
        "strategies": list(cohort.get("strategies") or []),
        "authority_contract": dict(payload.get("authority_contract") or {}),
        "runtime_contract": {
            "changes_selected_strategy": False,
            "changes_action": False,
            "changes_quantity": False,
            "counterfactual_research_only": True,
            "paper_order_authority": False,
            "live_order_authority": False,
        },
        "source_artifact": str(source_artifact),
        "source_receipt_sha256": str(payload.get("report_receipt_sha256") or ""),
    }


def build_payload(
    project_root: Path = PROJECT_ROOT,
    *,
    config_path: Path | None = None,
    generated_at_utc: str | None = None,
    previous_path: Path | None = None,
    use_cache: bool = True,
) -> dict[str, Any]:
    root = project_root.expanduser().resolve()
    policy_path = (config_path or DEFAULT_CONFIG_PATH).expanduser().resolve()
    policy = load_json(policy_path)
    authority = _validate_authority(policy)
    now = _utc(generated_at_utc) or datetime.now(timezone.utc)
    sources = _as_dict(policy.get("sources"))
    paths = {
        "config": policy_path,
        "strategy_library": _resolve(
            root,
            sources.get("strategy_library_path"),
            "governance/research/sleeve_strategy_library_latest.json",
        ),
        "strategy_policy": _resolve(
            root,
            sources.get("strategy_policy_path"),
            "config/sleeve_strategy_contracts_v1.json",
        ),
        "regime": _resolve(
            root,
            sources.get("regime_path"),
            "governance/health/regime_control_plane_latest.json",
        ),
        "alpha_inputs": _resolve(
            root,
            sources.get("alpha_inputs_path"),
            "governance/research/alpha_concept_inputs_latest.json",
        ),
        "alpha_generation": _resolve(
            root,
            sources.get("alpha_generation_path"),
            "governance/health/alpha_generation_control_latest.json",
        ),
        "paper_performance": _resolve(
            root,
            sources.get("paper_performance_path"),
            "governance/health/paper_performance_latest.json",
        ),
    }
    alpha_inputs = load_json(paths["alpha_inputs"])
    alpha_generation = load_json(paths["alpha_generation"])
    paper_performance = load_json(paths["paper_performance"])
    regime_payload = load_json(paths["regime"])
    binding = _candidate_binding(alpha_inputs, alpha_generation, paper_performance)
    regime_preview = _regime_context(
        regime_payload,
        now=now,
        maximum_age_seconds=int(sources.get("maximum_regime_age_seconds") or 3600),
    )
    alpha_age_preview = _age_seconds(alpha_inputs, now)
    alpha_fresh_preview = bool(
        alpha_age_preview is not None
        and alpha_age_preview
        <= max(int(sources.get("maximum_alpha_input_age_seconds") or 7200), 1)
    )
    try:
        library_mtime_age = max(
            now.timestamp() - paths["strategy_library"].stat().st_mtime, 0.0
        )
    except OSError:
        library_mtime_age = None
    library_mtime_fresh = bool(
        library_mtime_age is not None
        and library_mtime_age
        <= max(int(sources.get("maximum_library_age_seconds") or 86400), 1)
    )
    freshness_state = {
        "library_present": paths["strategy_library"].is_file(),
        "library_mtime_fresh": library_mtime_fresh,
        "regime_fresh": bool(regime_preview.get("fresh")),
        "regime_trusted": bool(regime_preview.get("trusted_for_shadow_admission")),
        "alpha_inputs_fresh": alpha_fresh_preview,
    }
    source_signature, source_receipts = _source_signature(
        paths,
        policy_id=str(policy.get("policy_id") or ""),
        candidate_hint=str(binding.get("candidate_id") or ""),
        freshness_state=freshness_state,
    )
    prior_path = previous_path or (
        root / "governance" / "health" / "strategy_market_fit_infrabot_latest.json"
    )
    prior = load_json(prior_path) if use_cache else {}
    catalog_policy = _as_dict(policy.get("catalog_contract"))
    if (
        use_cache
        and bool(catalog_policy.get("cache_when_source_signature_unchanged", True))
        and prior.get("source_signature") == source_signature
        and prior.get("policy_id") == policy.get("policy_id")
        and int(prior.get("evaluated_strategy_count") or 0)
        == int(catalog_policy.get("expected_strategy_count") or 12000)
    ):
        payload = dict(prior)
        payload["timestamp_utc"] = now.isoformat()
        payload["evaluation_mode"] = "cache_hit"
        payload["candidate_binding"] = binding
        payload["current_regime"] = regime_preview
        catalog = _as_dict(payload.get("catalog_contract"))
        catalog["library_age_seconds"] = (
            round(library_mtime_age, 3) if library_mtime_age is not None else None
        )
        catalog["library_fresh"] = library_mtime_fresh
        payload["catalog_contract"] = catalog
        forecast = _as_dict(payload.get("candidate_forecast_evidence"))
        forecast["fresh"] = alpha_fresh_preview
        forecast["age_seconds"] = (
            round(alpha_age_preview, 3) if alpha_age_preview is not None else None
        )
        payload["candidate_forecast_evidence"] = forecast
        payload["cache"] = {
            "hit": True,
            "source_signature_unchanged": True,
            "full_scan_timestamp_utc": str(
                prior.get("full_scan_timestamp_utc") or prior.get("timestamp_utc") or ""
            ),
        }
        payload["source_receipts"] = source_receipts
        payload["report_receipt_sha256"] = _hash(
            {
                "source_signature": source_signature,
                "full_scan_receipt": prior.get("full_scan_receipt_sha256"),
                "timestamp_utc": now.isoformat(),
            }
        )
        return payload

    library = load_json(paths["strategy_library"])
    strategy_policy = load_json(paths["strategy_policy"])
    strategies = [_as_dict(row) for row in _as_list(library.get("strategies"))]
    expected_count = max(int(catalog_policy.get("expected_strategy_count") or 12000), 1)
    strategy_ids = [str(row.get("strategy_id") or "").strip() for row in strategies]
    duplicate_ids = sorted(
        strategy_id
        for strategy_id, count in Counter(strategy_ids).items()
        if strategy_id and count > 1
    )
    missing_ids = sum(not strategy_id for strategy_id in strategy_ids)
    missing_receipts = sum(
        not str(row.get("contract_receipt_sha256") or "") for row in strategies
    )
    library_age = _age_seconds(library, now)
    library_fresh = bool(
        library_age is not None
        and library_age
        <= max(int(sources.get("maximum_library_age_seconds") or 86400), 1)
    )
    regime = regime_preview
    alpha_age = _age_seconds(alpha_inputs, now)
    alpha_fresh = bool(
        alpha_age is not None
        and alpha_age
        <= max(int(sources.get("maximum_alpha_input_age_seconds") or 7200), 1)
    )
    source_integrity_score = (
        1.0
        if library_fresh and regime.get("trusted_for_shadow_admission") and alpha_fresh
        else (
            0.6
            if library_fresh and regime.get("fresh")
            else 0.2 if library_fresh else 0.0
        )
    )
    minimum_observations = max(
        int(
            _as_dict(policy.get("scoring")).get("minimum_sleeve_regime_observations")
            or 20
        ),
        1,
    )
    sleeve_evidence = _sleeve_forecast_evidence(
        alpha_inputs, minimum_observations=minimum_observations
    )
    capacity_truth = _as_dict(
        _as_dict(alpha_inputs.get("diagnostics")).get("capacity_truth")
    )
    capacity_clear = bool(
        capacity_truth
        and not capacity_truth.get("capacity_remains_blocked_without_direct_adv", True)
    )
    scored = [
        _score_strategy(
            row,
            regime=str(regime.get("current_regime") or "unknown"),
            strategy_policy=strategy_policy,
            policy=policy,
            sleeve_evidence=sleeve_evidence,
            source_integrity_score=source_integrity_score,
            capacity_clear=capacity_clear,
            candidate_bound=bool(binding.get("bound")),
            current_conditions_trusted=bool(regime.get("trusted_for_shadow_admission")),
        )
        for row in strategies
    ]
    scored.sort(key=lambda row: str(row.get("strategy_id") or ""))
    full_ranked = _ranked_subset(scored, limit=len(scored))
    top_limit = max(int(catalog_policy.get("top_strategy_limit") or 100), 1)
    family_limit = max(int(catalog_policy.get("top_family_limit") or 100), 1)
    per_sleeve_limit = max(int(catalog_policy.get("top_per_sleeve_limit") or 3), 1)
    family_winners = _best_grouped(scored, key="family_id", limit=1)
    family_winners = _ranked_subset(family_winners, limit=family_limit)
    sleeve_winners = _best_grouped(scored, key="sleeve_id", limit=per_sleeve_limit)
    fit_counts = Counter(str(row.get("market_fit_band") or "unknown") for row in scored)
    relevance_counts = Counter(
        str(row.get("regime_relevance") or "unknown") for row in scored
    )
    proven_count = sum(bool(row.get("proven_working_now")) for row in scored)
    cohort_policy = _as_dict(policy.get("challenger_cohort"))
    configured_cohort = [
        _as_dict(row) for row in _as_list(cohort_policy.get("strategies"))
    ]
    maximum_slots = max(int(cohort_policy.get("maximum_slots") or 5), 1)
    cohort_ids = [str(row.get("strategy_id") or "") for row in configured_cohort]
    by_id = {str(row.get("strategy_id") or ""): row for row in scored}
    cohort_rows: list[dict[str, Any]] = []
    missing_cohort: list[str] = []
    for configured in configured_cohort:
        strategy_id = str(configured.get("strategy_id") or "")
        row = by_id.get(strategy_id)
        if row is None:
            missing_cohort.append(strategy_id)
            continue
        if not binding.get("bound"):
            state = "queued_candidate_binding"
        elif not regime.get("trusted_for_shadow_admission"):
            state = "queued_regime_source_not_ready"
        else:
            state = "shadow_observe"
        cohort_rows.append(
            {
                **row,
                "cohort_role": str(configured.get("role") or "challenger"),
                "cohort_state": state,
                "changes_active_strategy": False,
                "paper_order_authority": False,
                "live_order_authority": False,
            }
        )
    cohort_families = {
        str(row.get("signal_family") or "unknown") for row in cohort_rows
    }
    cohort_complete = bool(
        len(cohort_rows) == len(configured_cohort)
        and len(cohort_rows) <= maximum_slots
        and not missing_cohort
        and len(set(cohort_ids)) == len(cohort_ids)
        and (
            not bool(cohort_policy.get("require_distinct_signal_families", True))
            or len(cohort_families) == len(cohort_rows)
        )
    )
    contract_blockers: list[str] = []
    if len(strategies) != expected_count:
        contract_blockers.append(
            f"strategy_count_mismatch:{len(strategies)}/{expected_count}"
        )
    if duplicate_ids:
        contract_blockers.append("duplicate_strategy_ids")
    if missing_ids:
        contract_blockers.append("missing_strategy_ids")
    if missing_receipts:
        contract_blockers.append("missing_contract_receipts")
    if not cohort_complete:
        contract_blockers.append("challenger_cohort_contract_incomplete")
    contract_ready = not contract_blockers
    previous_top_ids = {
        str(row.get("strategy_id") or "")
        for row in _as_list(prior.get("top_strategy_rankings"))
    }
    current_top_ids = {
        str(row.get("strategy_id") or "") for row in full_ranked[:top_limit]
    }
    union = previous_top_ids | current_top_ids
    ranking_turnover = (
        1.0 - len(previous_top_ids & current_top_ids) / len(union)
        if previous_top_ids and union
        else 0.0
    )
    prior_regime = str(
        _as_dict(prior.get("current_regime")).get("current_regime") or ""
    )
    drift_detected = bool(
        (prior_regime and prior_regime != regime.get("current_regime"))
        or ranking_turnover > 0.25
    )
    status = (
        "blocked"
        if not contract_ready
        else (
            "ready"
            if regime.get("trusted_for_shadow_admission") and binding.get("bound")
            else "guarded"
        )
    )
    batch_receipts = _batch_receipts(
        scored, int(catalog_policy.get("batch_size") or 500)
    )
    full_scan_receipt = _hash(
        {
            "candidate_id": binding.get("candidate_id"),
            "current_regime": regime.get("current_regime"),
            "strategy_ids": strategy_ids,
            "ranking": [
                (row["strategy_id"], row["market_fit_score"]) for row in full_ranked
            ],
        }
    )
    payload = {
        "timestamp_utc": now.isoformat(),
        "full_scan_timestamp_utc": now.isoformat(),
        "schema_version": 1,
        "policy_id": str(policy.get("policy_id") or ""),
        "ok": contract_ready,
        "overall_status": status,
        "evaluation_mode": "full_scan",
        "source_signature": source_signature,
        "source_receipts": source_receipts,
        "candidate_binding": binding,
        "current_regime": regime,
        "catalog_contract": {
            "expected_strategy_count": expected_count,
            "input_strategy_count": len(strategies),
            "evaluated_strategy_count": len(scored),
            "unique_strategy_count": len(set(strategy_ids))
            - (1 if "" in strategy_ids else 0),
            "missing_strategy_id_count": missing_ids,
            "duplicate_strategy_id_count": len(duplicate_ids),
            "missing_contract_receipt_count": missing_receipts,
            "library_age_seconds": (
                round(library_age, 3) if library_age is not None else None
            ),
            "library_fresh": library_fresh,
            "all_strategies_checked": len(scored) == expected_count and contract_ready,
            "batch_count": len(batch_receipts),
            "batch_size": int(catalog_policy.get("batch_size") or 500),
            "strategy_ids_sha256": _hash(sorted(strategy_ids)),
            "blockers": contract_blockers,
        },
        "evaluated_strategy_count": len(scored),
        "proven_working_strategy_count": proven_count,
        "market_fit_counts": dict(sorted(fit_counts.items())),
        "regime_relevance_counts": dict(sorted(relevance_counts.items())),
        "candidate_forecast_evidence": {
            "fresh": alpha_fresh,
            "age_seconds": round(alpha_age, 3) if alpha_age is not None else None,
            "sleeve_count": len(sleeve_evidence),
            "evidence_class": "candidate_bound_counterfactual_forecast",
            "counts_as_trade_profit": False,
        },
        "capacity_gate": {
            "clear": capacity_clear,
            "direct_daily_dollar_volume_required": True,
            "normalized_relative_volume_substitute_allowed": False,
        },
        "expansion_freeze": {
            "active": bool(
                _as_dict(alpha_generation.get("strategy_expansion_freeze")).get(
                    "active", True
                )
            ),
            "cohort_compatible": True,
            "reason": "the cohort observes five existing contracts and creates no offspring or runtime activation",
        },
        "challenger_cohort": {
            "status": "ready" if cohort_complete else "blocked",
            "mode": str(cohort_policy.get("mode") or ""),
            "slot_count": len(cohort_rows),
            "maximum_slots": maximum_slots,
            "missing_strategy_ids": missing_cohort,
            "distinct_signal_family_count": len(cohort_families),
            "contract_complete": cohort_complete,
            "strategies": cohort_rows,
            "paper_order_authority": False,
            "live_order_authority": False,
        },
        "top_strategy_rankings": full_ranked[:top_limit],
        "top_family_rankings": family_winners,
        "top_sleeve_rankings": sleeve_winners,
        "batch_receipts": batch_receipts,
        "drift": {
            "detected": drift_detected,
            "previous_regime": prior_regime or "unknown",
            "current_regime": str(regime.get("current_regime") or "unknown"),
            "top_ranking_turnover": round(ranking_turnover, 8),
            "changes_runtime_behavior": False,
        },
        "assigned_infrabots": _infrabot_rows(
            contract_ready=contract_ready,
            regime_trusted=bool(regime.get("trusted_for_shadow_admission")),
            cohort_ready=cohort_complete,
            proven_count=proven_count,
            drift_detected=drift_detected,
        ),
        "authority_contract": authority,
        "proof_contract": dict(policy.get("proof_contract") or {}),
        "soak_contract": {
            "candidate_id_unchanged": True,
            "observational_metadata_only": True,
            "changes_runtime_decisions": False,
            "requires_soak_reset": False,
            "may_claim_profitability": False,
        },
        "cache": {
            "hit": False,
            "source_signature_unchanged": False,
        },
        "full_scan_receipt_sha256": full_scan_receipt,
        "recommended_actions": [
            "keep the five existing challengers in shadow-only observation until the regime source is fresh and ready",
            "treat market-fit scores as research priority, never expected return or proof of profitability",
            "continue G123 candidate-bound paper collection until strategy-level post-cost evidence clears existing gates",
            "retain the strategy expansion freeze and admit no new offspring from this scan",
        ],
    }
    payload["report_receipt_sha256"] = _hash(payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate the complete strategy catalog against current market conditions."
    )
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--out-file", default=str(DEFAULT_OUT_PATH))
    parser.add_argument("--cohort-out-file", default=str(DEFAULT_COHORT_OUT_PATH))
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    root = Path(args.project_root).expanduser().resolve()
    out_path = Path(args.out_file).expanduser().resolve()
    payload = build_payload(
        root,
        config_path=Path(args.config).expanduser().resolve(),
        previous_path=out_path,
        use_cache=not bool(args.force),
    )
    write_payload(out_path, payload)
    write_payload(
        Path(args.cohort_out_file).expanduser().resolve(),
        _cohort_payload(payload, source_artifact=out_path),
    )
    if args.json:
        print(json.dumps(payload, ensure_ascii=True, sort_keys=True))
    else:
        cohort = _as_dict(payload.get("challenger_cohort"))
        catalog = _as_dict(payload.get("catalog_contract"))
        print(
            "strategy_market_fit_infrabot "
            f"status={payload.get('overall_status')} "
            f"mode={payload.get('evaluation_mode')} "
            f"checked={payload.get('evaluated_strategy_count')}/"
            f"{catalog.get('expected_strategy_count')} "
            f"cohort={cohort.get('slot_count')}/{cohort.get('maximum_slots')} "
            f"regime={_as_dict(payload.get('current_regime')).get('current_regime')} "
            f"proven={payload.get('proven_working_strategy_count')}"
        )
    return 0 if payload.get("overall_status") in {"ready", "guarded"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
