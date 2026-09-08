#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_POLICY_PATH = PROJECT_ROOT / "config" / "quantitative_challengers_v1.json"
DEFAULT_PERFORMANCE_PATH = PROJECT_ROOT / "governance" / "health" / "paper_performance_latest.json"
DEFAULT_OUT_PATH = PROJECT_ROOT / "governance" / "research" / "quantitative_challenger_latest.json"

if str(PROJECT_ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(PROJECT_ROOT))

from core.quantitative_challengers import (  # noqa: E402
    block_bootstrap_model_selection,
    cost_aware_expert_aggregation,
    cpcv_triple_barrier_diagnostic,
    drawdown_constrained_kelly,
    entropy_pooling_downside_view,
    least_squares_optimal_stopping,
    probabilistic_sharpe_bayesian_utility,
    sequential_sign_sprt,
)


METHOD_IDS = (
    "always_valid_sequential_inference",
    "spa_reality_check",
    "probabilistic_sharpe_bayesian_utility",
    "drawdown_constrained_kelly",
    "entropy_pooling",
    "optimal_stopping",
    "cpcv_triple_barrier",
    "online_expert_aggregation",
)


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
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


def _file_hash(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return ""


def _utc(value: Any) -> datetime | None:
    raw = str(value or "").strip().replace("Z", "+00:00")
    if not raw:
        return None
    try:
        parsed = datetime.fromisoformat(raw)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _method(policy: Mapping[str, Any], method_id: str) -> dict[str, Any]:
    methods = policy.get("methods") if isinstance(policy.get("methods"), dict) else {}
    raw = methods.get(method_id) if isinstance(methods, dict) else {}
    return dict(raw) if isinstance(raw, dict) else {}


def _validate_policy(policy: Mapping[str, Any]) -> None:
    methods = policy.get("methods") if isinstance(policy.get("methods"), dict) else {}
    if set(methods) != set(METHOD_IDS):
        raise ValueError("quantitative challenger policy must define exactly eight methods")
    if not all(bool((methods.get(method_id) or {}).get("enabled", False)) for method_id in METHOD_IDS):
        raise ValueError("all quantitative challenger methods must be explicitly enabled")
    authority = policy.get("authority") if isinstance(policy.get("authority"), dict) else {}
    if not authority or any(bool(value) for value in authority.values()):
        raise ValueError("quantitative challenger policy requests forbidden authority")
    binding = policy.get("candidate_binding") if isinstance(policy.get("candidate_binding"), dict) else {}
    required_binding = (
        "required",
        "require_candidate_filter_active",
        "require_zero_mismatch_rows",
        "require_post_cutoff_watermark",
    )
    if not all(bool(binding.get(key, False)) for key in required_binding):
        raise ValueError("quantitative challenger policy may not weaken candidate binding")
    if bool(binding.get("allow_historical_fallback", True)) or bool(
        binding.get("allow_cross_candidate_pooling", True)
    ):
        raise ValueError("quantitative challenger policy may not borrow historical candidates")


def _candidate_binding(performance: Mapping[str, Any]) -> dict[str, Any]:
    window = (
        performance.get("profitability_evidence_window")
        if isinstance(performance.get("profitability_evidence_window"), dict)
        else {}
    )
    candidate_id = str(window.get("candidate_id") or "").strip()
    cutoff = _utc(window.get("candidate_cutoff_utc"))
    watermark = _utc(window.get("evidence_through_utc"))
    mismatch_count = max(int(window.get("candidate_binding_mismatch_rows_excluded") or 0), 0)
    reasons: list[str] = []
    if not candidate_id:
        reasons.append("candidate_id_missing")
    if not bool(window.get("candidate_binding_required", False)):
        reasons.append("candidate_binding_not_required")
    if not bool(window.get("candidate_filter_active", False)):
        reasons.append("candidate_filter_inactive")
    if mismatch_count:
        reasons.append("candidate_binding_mismatch_rows_present")
    if cutoff is None:
        reasons.append("candidate_cutoff_missing")
    if watermark is None:
        reasons.append("evidence_watermark_missing")
    elif cutoff is not None and watermark < cutoff:
        reasons.append("evidence_watermark_precedes_candidate")
    return {
        "candidate_id": candidate_id,
        "generation": int(window.get("candidate_generation") or 0),
        "cutoff_utc": cutoff.isoformat() if cutoff else "",
        "evidence_through_utc": watermark.isoformat() if watermark else "",
        "required": bool(window.get("candidate_binding_required", False)),
        "bound": not reasons,
        "mismatch_rows_excluded": mismatch_count,
        "reasons": reasons,
        "policy": "candidate-forward post-cost evidence only; historical and cross-candidate fallback are forbidden",
    }


def _profile_series(
    performance: Mapping[str, Any],
) -> tuple[dict[str, list[float]], dict[str, list[str]]]:
    raw = (
        performance.get("candidate_post_cost_daily_series")
        if isinstance(performance.get("candidate_post_cost_daily_series"), dict)
        else {}
    )
    values_by_profile: dict[str, list[float]] = {}
    days_by_profile: dict[str, list[str]] = {}
    for profile, rows in sorted(raw.items()):
        if not isinstance(rows, list):
            continue
        parsed: list[tuple[str, float]] = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            day = str(row.get("day_utc") or "").strip()
            try:
                value = float(row.get("post_cost_return_bps_total"))
            except (TypeError, ValueError):
                continue
            if day:
                parsed.append((day, value))
        parsed.sort()
        if parsed:
            key = str(profile).strip().lower()
            values_by_profile[key] = [value for _day, value in parsed]
            days_by_profile[key] = [day for day, _value in parsed]
    return values_by_profile, days_by_profile


def _aligned_series(
    values: Mapping[str, list[float]],
    days: Mapping[str, list[str]],
) -> tuple[dict[str, list[float]], list[str]]:
    if len(values) < 2:
        return {}, []
    by_profile = {
        profile: dict(zip(days.get(profile, []), profile_values))
        for profile, profile_values in values.items()
    }
    common = sorted(set.intersection(*(set(rows) for rows in by_profile.values())))
    return {
        profile: [rows[day] for day in common]
        for profile, rows in sorted(by_profile.items())
    }, common


def _per_profile_methods(values: list[float], policy: Mapping[str, Any]) -> dict[str, Any]:
    sequential = _method(policy, "always_valid_sequential_inference")
    probabilistic = _method(policy, "probabilistic_sharpe_bayesian_utility")
    kelly = _method(policy, "drawdown_constrained_kelly")
    entropy = _method(policy, "entropy_pooling")
    stopping = _method(policy, "optimal_stopping")
    cpcv = _method(policy, "cpcv_triple_barrier")
    return {
        "always_valid_sequential_inference": sequential_sign_sprt(
            values,
            null_win_probability=float(sequential.get("null_win_probability", 0.5)),
            alternative_win_probability=float(sequential.get("alternative_win_probability", 0.57)),
            alpha=float(sequential.get("alpha", 0.05)),
            beta=float(sequential.get("beta", 0.2)),
            minimum_observations=int(sequential.get("minimum_observations", 20)),
            hurdle_bps=float(sequential.get("hurdle_bps", 0.0)),
        ),
        "probabilistic_sharpe_bayesian_utility": probabilistic_sharpe_bayesian_utility(
            values,
            annualization_periods=int(probabilistic.get("annualization_periods", 252)),
            reference_sharpe=float(probabilistic.get("reference_sharpe", 0.0)),
            minimum_observations=int(probabilistic.get("minimum_observations", 20)),
            posterior_draws=int(probabilistic.get("posterior_draws", 1000)),
            posterior_probability_floor=float(probabilistic.get("posterior_probability_floor", 0.95)),
            prior_strength=float(probabilistic.get("prior_strength", 5.0)),
            prior_scale_bps=float(probabilistic.get("prior_scale_bps", 25.0)),
            risk_aversion=float(probabilistic.get("risk_aversion", 3.0)),
            seed=int(probabilistic.get("seed", 1733)),
        ),
        "drawdown_constrained_kelly": drawdown_constrained_kelly(
            values,
            minimum_observations=int(kelly.get("minimum_observations", 20)),
            max_fraction=float(kelly.get("maximum_fraction", 0.25)),
            drawdown_limit=float(kelly.get("drawdown_limit", 0.1)),
            grid_steps=int(kelly.get("grid_steps", 50)),
        ),
        "entropy_pooling": entropy_pooling_downside_view(
            values,
            minimum_observations=int(entropy.get("minimum_observations", 20)),
            tail_quantile=float(entropy.get("tail_quantile", 0.25)),
            target_tail_probability=float(entropy.get("target_tail_probability", 0.35)),
        ),
        "optimal_stopping": least_squares_optimal_stopping(
            values,
            horizon=int(stopping.get("horizon", 5)),
            minimum_paths=int(stopping.get("minimum_independent_paths", 12)),
            training_fraction=float(stopping.get("training_fraction", 0.7)),
        ),
        "cpcv_triple_barrier": cpcv_triple_barrier_diagnostic(
            values,
            minimum_observations=int(cpcv.get("minimum_observations", 30)),
            upper_barrier_bps=float(cpcv.get("upper_barrier_bps", 25.0)),
            lower_barrier_bps=float(cpcv.get("lower_barrier_bps", 25.0)),
            horizon=int(cpcv.get("horizon", 5)),
            group_count=int(cpcv.get("group_count", 6)),
            test_group_count=int(cpcv.get("test_group_count", 2)),
            embargo_observations=int(cpcv.get("embargo_observations", 1)),
        ),
    }


def build_payload(
    project_root: Path = PROJECT_ROOT,
    *,
    policy_path: Path | None = None,
    performance_path: Path | None = None,
    generated_at_utc: str | None = None,
) -> dict[str, Any]:
    policy_source = policy_path or project_root / "config" / "quantitative_challengers_v1.json"
    performance_source = performance_path or project_root / "governance" / "health" / "paper_performance_latest.json"
    policy = _load_json(policy_source)
    _validate_policy(policy)
    performance = _load_json(performance_source)
    binding = _candidate_binding(performance)
    values_by_profile, days_by_profile = _profile_series(performance)
    known_profiles = {
        str(row.get("profile") or "").strip().lower()
        for row in (performance.get("sleeve_latest") or [])
        if isinstance(row, dict) and str(row.get("profile") or "").strip()
    }
    known_profiles.update(values_by_profile)
    profiles: dict[str, Any] = {}
    for profile in sorted(known_profiles):
        values = values_by_profile.get(profile, []) if binding["bound"] else []
        methods = _per_profile_methods(values, policy)
        available_count = sum(bool(row.get("available", False)) for row in methods.values())
        support_count = sum(bool(row.get("passes", False)) for row in methods.values())
        profiles[profile] = {
            "profile": profile,
            "candidate_period_count": len(values),
            "first_day_utc": days_by_profile.get(profile, [""])[0] if days_by_profile.get(profile) else "",
            "last_day_utc": days_by_profile.get(profile, [""])[-1] if days_by_profile.get(profile) else "",
            "available_method_count": available_count,
            "supported_method_count": support_count,
            "method_count": len(methods),
            "methods": methods,
        }

    aligned, common_days = _aligned_series(values_by_profile, days_by_profile)
    if not binding["bound"]:
        aligned = {}
        common_days = []
    spa = _method(policy, "spa_reality_check")
    online = _method(policy, "online_expert_aggregation")
    cross_profile = {
        "spa_reality_check": block_bootstrap_model_selection(
            aligned,
            replications=int(spa.get("replications", 500)),
            block_length=int(spa.get("block_length", 3)),
            alpha=float(spa.get("alpha", 0.05)),
            minimum_periods=int(spa.get("minimum_periods", 12)),
            seed=int(spa.get("seed", 1729)),
        ),
        "online_expert_aggregation": cost_aware_expert_aggregation(
            aligned,
            minimum_periods=int(online.get("minimum_periods", 12)),
            learning_rate=float(online.get("learning_rate", 1.0)),
            return_scale_bps=float(online.get("return_scale_bps", 100.0)),
            transaction_cost_bps=float(online.get("transaction_cost_bps", 2.0)),
            maximum_weight=float(online.get("maximum_weight", 0.35)),
        ),
    }
    method_available = {
        method_id: (
            bool(cross_profile[method_id].get("available", False))
            if method_id in cross_profile
            else any(
                bool((row.get("methods") or {}).get(method_id, {}).get("available", False))
                for row in profiles.values()
            )
        )
        for method_id in METHOD_IDS
    }
    method_supported = {
        method_id: (
            bool(cross_profile[method_id].get("passes", False))
            if method_id in cross_profile
            else any(
                bool((row.get("methods") or {}).get(method_id, {}).get("passes", False))
                for row in profiles.values()
            )
        )
        for method_id in METHOD_IDS
    }
    evidence_ready_count = sum(method_available.values())
    timestamp = _utc(generated_at_utc) or datetime.now(timezone.utc)
    receipt_material = {
        "candidate_binding": binding,
        "policy_sha256": _file_hash(policy_source),
        "performance_sha256": _file_hash(performance_source),
        "profiles": profiles,
        "cross_profile": cross_profile,
    }
    report_receipt = _canonical_hash(receipt_material)
    cross_profile_available_count = sum(
        bool(method.get("available", False)) for method in cross_profile.values()
    )
    cross_profile_supported_count = sum(
        bool(method.get("passes", False)) for method in cross_profile.values()
    )
    decision_metadata_by_profile = {
        profile: {
            "policy_id": str(policy.get("policy_id") or ""),
            "candidate_id": str(binding.get("candidate_id") or ""),
            "candidate_bound": bool(binding.get("bound", False)),
            "status": (
                "binding_invalid"
                if not binding["bound"]
                else "evidence_available"
                if row["available_method_count"] + cross_profile_available_count
                else "collecting"
            ),
            "available_method_count": (
                row["available_method_count"] + cross_profile_available_count
            ),
            "supported_method_count": (
                row["supported_method_count"] + cross_profile_supported_count
            ),
            "method_count": len(METHOD_IDS),
            "profile_method_count": row["method_count"],
            "profile_available_method_count": row["available_method_count"],
            "profile_supported_method_count": row["supported_method_count"],
            "cross_profile_available_method_count": cross_profile_available_count,
            "cross_profile_supported_method_count": cross_profile_supported_count,
            "method_statuses": {
                method_id: str(
                    (
                        cross_profile.get(method_id)
                        or row["methods"].get(method_id)
                        or {}
                    ).get("status")
                    or ""
                )
                for method_id in METHOD_IDS
            },
            "report_receipt_sha256": report_receipt,
            "authority": "read_only_metadata_no_decision_authority",
        }
        for profile, row in profiles.items()
    }
    structurally_ready = bool(binding["bound"] and len(method_available) == 8)
    return {
        "timestamp_utc": timestamp.isoformat(),
        "schema_version": 1,
        "policy_id": str(policy.get("policy_id") or ""),
        "ok": structurally_ready,
        "overall_status": (
            "ready" if structurally_ready and evidence_ready_count == 8 else "collecting" if structurally_ready else "blocked"
        ),
        "operating_mode": str(policy.get("operating_mode") or ""),
        "candidate_binding": binding,
        "authority_contract": dict(policy.get("authority") or {}),
        "resource_contract": dict(policy.get("resource_contract") or {}),
        "graduation_contract": dict(policy.get("graduation_contract") or {}),
        "concept_count": 8,
        "implemented_concept_count": 8,
        "evidence_ready_concept_count": evidence_ready_count,
        "supported_concept_count": sum(method_supported.values()),
        "method_availability": method_available,
        "method_support": method_supported,
        "profile_count": len(profiles),
        "candidate_profile_series_count": len(values_by_profile),
        "aligned_profile_count": len(aligned),
        "aligned_common_period_count": len(common_days),
        "aligned_common_days_utc": common_days,
        "profiles": profiles,
        "cross_profile_methods": cross_profile,
        "decision_metadata_by_profile": decision_metadata_by_profile,
        "report_receipt_sha256": report_receipt,
        "source_receipts": {
            "policy_path": str(policy_source),
            "policy_sha256": _file_hash(policy_source),
            "paper_performance_path": str(performance_source),
            "paper_performance_sha256": _file_hash(performance_source),
            "paper_performance_timestamp_utc": str(performance.get("timestamp_utc") or ""),
        },
        "interpretation": {
            "collecting": "the method is implemented but the frozen candidate has not earned its required post-cost sample floor",
            "supported": "the challenger passed its configured diagnostic; this grants no active authority",
            "not_supported": "the challenger had enough evidence but did not pass its configured diagnostic",
            "profitability_guaranteed": False,
        },
        "blockers": list(binding.get("reasons") or []),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build the candidate-bound paper-only quantitative challenger report."
    )
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--policy", default="config/quantitative_challengers_v1.json")
    parser.add_argument("--performance", default="governance/health/paper_performance_latest.json")
    parser.add_argument("--out-file", default="governance/research/quantitative_challenger_latest.json")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    project_root = Path(args.project_root).expanduser().resolve()
    policy_path = Path(args.policy).expanduser()
    if not policy_path.is_absolute():
        policy_path = project_root / policy_path
    performance_path = Path(args.performance).expanduser()
    if not performance_path.is_absolute():
        performance_path = project_root / performance_path
    out_path = Path(args.out_file).expanduser()
    if not out_path.is_absolute():
        out_path = project_root / out_path
    payload = build_payload(
        project_root,
        policy_path=policy_path,
        performance_path=performance_path,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        print(
            "quantitative_challengers "
            f"status={payload['overall_status']} "
            f"candidate={payload['candidate_binding']['candidate_id'] or 'missing'} "
            f"implemented={payload['implemented_concept_count']}/8 "
            f"evidence_ready={payload['evidence_ready_concept_count']}/8 "
            f"supported={payload['supported_concept_count']}/8"
        )
    return 0 if bool(payload.get("ok", False)) else 2


if __name__ == "__main__":
    raise SystemExit(main())
