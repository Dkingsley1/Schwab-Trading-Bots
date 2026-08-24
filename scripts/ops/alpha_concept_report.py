#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np

if __package__ in {None, ""}:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from core.alpha_concept_engine import MEASUREMENT_FUNCTIONS
    from scripts.ops.long_runtime_common import load_json, write_payload
else:
    from core.alpha_concept_engine import MEASUREMENT_FUNCTIONS
    from .long_runtime_common import PROJECT_ROOT, load_json, write_payload


DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "alpha_concept_registry_v1.json"
DEFAULT_OUT_PATH = (
    PROJECT_ROOT / "governance" / "research" / "alpha_concept_report_latest.json"
)
DEFAULT_MARKDOWN_PATH = (
    PROJECT_ROOT / "exports" / "reports" / "operator" / "alpha_concept_report_latest.md"
)
ENGINE_IDS = (
    "information_coefficient_term_structure",
    "effective_breadth_transfer_coefficient",
    "hierarchical_bayesian_skill",
    "subsample_stability_selection",
    "economic_alpha_decomposition",
    "factor_neutral_residualization",
    "cross_fitted_causal_transportability",
    "capacity_impact_surface",
    "execution_alpha_attribution",
    "point_in_time_security_master_audit",
    "split_conformal_residual_calibration",
    "sequential_change_point_stability",
    "residual_redundancy_graph",
    "regime_conditional_robustness",
    "cost_stress_survival",
    "active_learning_value_of_information",
)


def _as_dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _as_list(value: Any) -> list[Any]:
    return list(value) if isinstance(value, (list, tuple)) else []


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


def _file_hash(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return ""


def _canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            ensure_ascii=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
    ).hexdigest()


def _grade(score: float, *, complete: bool = False) -> str:
    if complete and score >= 100.0:
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


def _validate_policy(policy: Mapping[str, Any]) -> None:
    engines = _as_dict(policy.get("measurement_engines"))
    if tuple(engines) != ENGINE_IDS:
        raise ValueError(
            "alpha concept policy must define the canonical measurement engines in order"
        )
    if set(MEASUREMENT_FUNCTIONS) != set(ENGINE_IDS):
        raise ValueError("alpha concept engine registry does not match the policy")
    authority = _as_dict(policy.get("authority"))
    if not authority or any(bool(value) for value in authority.values()):
        raise ValueError("alpha concept policy requests forbidden authority")
    scope = _as_dict(policy.get("scope_contract"))
    if bool(scope.get("claims_to_enumerate_every_possible_alpha_concept", True)):
        raise ValueError("alpha concept registry may not claim a universal enumeration")
    if not bool(scope.get("finite_architecture_specific_ontology", False)):
        raise ValueError(
            "alpha concept registry scope must be finite and architecture specific"
        )
    binding = _as_dict(policy.get("candidate_binding"))
    for key in (
        "required",
        "require_candidate_filter_active",
        "require_zero_mismatch_rows",
        "require_post_cutoff_watermark",
    ):
        if not bool(binding.get(key, False)):
            raise ValueError(f"candidate binding may not weaken {key}")
    if bool(binding.get("allow_historical_fallback", True)) or bool(
        binding.get("allow_cross_candidate_pooling", True)
    ):
        raise ValueError(
            "historical or cross-candidate evidence borrowing is forbidden"
        )


def _candidate_binding(
    candidate: Mapping[str, Any], performance: Mapping[str, Any]
) -> dict[str, Any]:
    window = _as_dict(performance.get("profitability_evidence_window"))
    state_id = str(candidate.get("candidate_id") or "").strip()
    performance_id = str(window.get("candidate_id") or "").strip()
    cutoff = _utc(window.get("candidate_cutoff_utc"))
    watermark = _utc(window.get("evidence_through_utc"))
    mismatch_count = max(
        int(window.get("candidate_binding_mismatch_rows_excluded") or 0), 0
    )
    blockers: list[str] = []
    if not state_id:
        blockers.append("candidate_state_id_missing")
    if not performance_id:
        blockers.append("paper_performance_candidate_id_missing")
    if state_id and performance_id and state_id != performance_id:
        blockers.append("candidate_identity_mismatch")
    if not bool(window.get("candidate_binding_required", False)):
        blockers.append("candidate_binding_not_required")
    if not bool(window.get("candidate_filter_active", False)):
        blockers.append("candidate_filter_inactive")
    if mismatch_count:
        blockers.append("candidate_binding_mismatch_rows_present")
    if cutoff is None:
        blockers.append("candidate_cutoff_missing")
    if watermark is None:
        blockers.append("evidence_watermark_missing")
    elif cutoff is not None and watermark < cutoff:
        blockers.append("evidence_watermark_precedes_candidate")
    return {
        "candidate_id": state_id or performance_id,
        "candidate_generation": int(
            candidate.get("generation") or window.get("candidate_generation") or 0
        ),
        "candidate_cutoff_utc": cutoff.isoformat() if cutoff else "",
        "evidence_through_utc": watermark.isoformat() if watermark else "",
        "bound": not blockers,
        "mismatch_rows_excluded": mismatch_count,
        "blockers": blockers,
        "historical_fallback_used": False,
        "cross_candidate_pooling_used": False,
    }


def _measurement_input_binding(
    payload: Mapping[str, Any], candidate_binding: Mapping[str, Any]
) -> dict[str, Any]:
    expected = str(candidate_binding.get("candidate_id") or "")
    candidate = str(
        payload.get("candidate_id")
        or _as_dict(payload.get("candidate_binding")).get("candidate_id")
        or ""
    ).strip()
    timestamp = _utc(
        payload.get("timestamp_utc")
        or _as_dict(payload.get("candidate_binding")).get("evidence_through_utc")
    )
    cutoff = _utc(candidate_binding.get("candidate_cutoff_utc"))
    blockers: list[str] = []
    if payload and not candidate:
        blockers.append("measurement_input_candidate_id_missing")
    if candidate and candidate != expected:
        blockers.append("measurement_input_candidate_mismatch")
    if payload and timestamp is None:
        blockers.append("measurement_input_timestamp_missing")
    elif timestamp is not None and cutoff is not None and timestamp < cutoff:
        blockers.append("measurement_input_precedes_candidate_cutoff")
    return {
        "present": bool(payload),
        "candidate_id": candidate,
        "timestamp_utc": timestamp.isoformat() if timestamp else "",
        "bound": bool(payload)
        and not blockers
        and bool(candidate_binding.get("bound")),
        "blockers": blockers,
    }


def _candidate_returns_by_profile(
    performance: Mapping[str, Any],
) -> dict[str, list[float]]:
    raw = _as_dict(performance.get("candidate_post_cost_daily_series"))
    result: dict[str, list[float]] = {}
    for profile, rows in sorted(raw.items()):
        values: list[float] = []
        for row in _as_list(rows):
            if not isinstance(row, Mapping):
                continue
            try:
                value = float(row.get("post_cost_return_bps_total"))
            except (TypeError, ValueError):
                continue
            if value == value and abs(value) != float("inf"):
                values.append(value)
        if values:
            result[str(profile)] = values
    return result


def _empty_inputs(engine_id: str) -> dict[str, Any]:
    return {
        "information_coefficient_term_structure": {"observations": []},
        "effective_breadth_transfer_coefficient": {
            "forecast_matrix": [],
            "realized_matrix": [],
        },
        "hierarchical_bayesian_skill": {"returns_by_group": {}},
        "subsample_stability_selection": {"features": {}, "outcomes": []},
        "economic_alpha_decomposition": {
            "gross_active_returns": [],
            "factors": {},
            "execution_costs": [],
        },
        "factor_neutral_residualization": {
            "target_returns": [],
            "factors": {},
        },
        "cross_fitted_causal_transportability": {
            "outcomes": [],
            "treatments": [],
            "controls": {},
            "environments": [],
        },
        "capacity_impact_surface": {
            "expected_gross_alpha_bps": None,
            "half_spread_bps": None,
            "fees_bps": None,
            "baseline_slippage_bps": None,
            "daily_dollar_volume": None,
            "volatility_bps": None,
            "impact_coefficient": None,
            "notionals": [],
        },
        "execution_alpha_attribution": {"fills": []},
        "point_in_time_security_master_audit": {
            "security_records": [],
            "observations": [],
        },
        "split_conformal_residual_calibration": {
            "predictions": [],
            "outcomes": [],
        },
        "sequential_change_point_stability": {"values_by_group": {}},
        "residual_redundancy_graph": {"returns_by_group": {}},
        "regime_conditional_robustness": {"outcomes": [], "regimes": []},
        "cost_stress_survival": {
            "expected_gross_alpha_bps": None,
            "base_cost_bps": None,
        },
        "active_learning_value_of_information": {"collection_gaps": []},
    }[engine_id]


def _engine_parameters(spec: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in spec.items()
        if key not in {"function", "concept_ids"}
    }


def _sample_count(result: Mapping[str, Any]) -> int:
    for key in (
        "observation_count",
        "period_count",
        "fill_count",
        "record_count",
        "gap_count",
        "group_count",
        "scenario_count",
        "regime_count",
    ):
        try:
            return int(result.get(key) or 0)
        except (TypeError, ValueError):
            continue
    return 0


def _collection_gap(
    engine_id: str, spec: Mapping[str, Any], result: Mapping[str, Any]
) -> dict[str, Any]:
    minimum = 1
    for key, value in spec.items():
        if str(key).startswith("minimum_") and "ratio" not in str(key):
            try:
                minimum = max(minimum, int(float(value)))
            except (TypeError, ValueError):
                continue
    observed = _sample_count(result)
    coverage = min(observed / minimum, 1.0) if minimum else 0.0
    importance = (
        1.0
        if engine_id
        in {
            "information_coefficient_term_structure",
            "factor_neutral_residualization",
            "capacity_impact_surface",
            "execution_alpha_attribution",
        }
        else 0.85
    )
    return {
        "gap_id": engine_id,
        "route": "candidate_bound_alpha_measurement_input",
        "uncertainty": 1.0 if not bool(result.get("available", False)) else 0.35,
        "economic_relevance": importance,
        "expected_uncertainty_reduction": 0.9,
        "novelty": 0.8,
        "coverage_deficit": 1.0 - coverage,
        "observation_cost": 1.0,
        "resource_pressure": 0.2,
        "minimum_observations": minimum,
        "observed_observations": observed,
    }


def _catalog(
    policy: Mapping[str, Any], root: Path
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    defaults = _as_dict(policy.get("family_defaults"))
    overrides = _as_dict(policy.get("operational_overrides"))
    conditional = _as_dict(policy.get("conditional_concepts"))
    families = _as_dict(policy.get("concept_families"))
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    duplicate_ids: list[str] = []
    malformed = 0
    for family, concepts in families.items():
        family_default = _as_dict(defaults.get(family))
        for raw in _as_list(concepts):
            if not isinstance(raw, (list, tuple)) or len(raw) != 2:
                malformed += 1
                continue
            concept_id = str(raw[0]).strip()
            title = str(raw[1]).strip()
            if not concept_id or not title:
                malformed += 1
                continue
            if concept_id in seen:
                duplicate_ids.append(concept_id)
                continue
            seen.add(concept_id)
            override = _as_dict(overrides.get(concept_id))
            owner = str(override.get("owner") or family_default.get("owner") or "")
            owner_path = root / owner if owner else Path()
            implementation_class = str(
                override.get("class")
                or (
                    "conditional_external"
                    if concept_id in conditional
                    else "registered_research_route"
                )
            )
            rows.append(
                {
                    "concept_id": concept_id,
                    "title": title,
                    "family": str(family),
                    "route": str(family_default.get("route") or ""),
                    "owner": owner,
                    "owner_present": bool(owner and owner_path.is_file()),
                    "implementation_class": implementation_class,
                    "conditional_reason": str(conditional.get(concept_id) or ""),
                }
            )
    summary = {
        "family_count": len(families),
        "concept_count": len(rows),
        "duplicate_concept_ids": sorted(set(duplicate_ids)),
        "malformed_concept_count": malformed,
        "owner_present_count": sum(bool(row["owner_present"]) for row in rows),
        "implementation_class_counts": {
            class_name: sum(row["implementation_class"] == class_name for row in rows)
            for class_name in sorted({str(row["implementation_class"]) for row in rows})
        },
    }
    return rows, summary


def build_payload(
    project_root: Path = PROJECT_ROOT,
    *,
    config_path: Path | None = None,
    generated_at_utc: str | None = None,
) -> dict[str, Any]:
    root = project_root.expanduser().resolve()
    policy_path = config_path or root / "config" / "alpha_concept_registry_v1.json"
    policy = load_json(policy_path)
    _validate_policy(policy)
    binding_policy = _as_dict(policy.get("candidate_binding"))
    candidate_path = root / str(
        binding_policy.get("candidate_state_path")
        or "governance/runtime/production_candidate_state.json"
    )
    performance_path = root / str(
        binding_policy.get("paper_performance_path")
        or "governance/health/paper_performance_latest.json"
    )
    inputs_path = root / str(
        binding_policy.get("measurement_inputs_path")
        or "governance/research/alpha_concept_inputs_latest.json"
    )
    candidate = load_json(candidate_path)
    performance = load_json(performance_path)
    measurement_inputs = load_json(inputs_path)
    binding = _candidate_binding(candidate, performance)
    input_binding = _measurement_input_binding(measurement_inputs, binding)
    raw_measurements = _as_dict(measurement_inputs.get("measurements"))
    returns_by_profile = (
        _candidate_returns_by_profile(performance) if bool(binding.get("bound")) else {}
    )
    policy_engines = _as_dict(policy.get("measurement_engines"))
    measurements: dict[str, dict[str, Any]] = {}
    for engine_id in ENGINE_IDS[:-1]:
        spec = _as_dict(policy_engines.get(engine_id))
        inputs = _empty_inputs(engine_id)
        source = "missing_candidate_measurement_input"
        if engine_id == "hierarchical_bayesian_skill" and returns_by_profile:
            inputs = {"returns_by_group": returns_by_profile}
            source = "paper_performance_candidate_post_cost_daily_series"
        elif bool(input_binding.get("bound")):
            supplied = _as_dict(raw_measurements.get(engine_id))
            supplied_inputs = _as_dict(supplied.get("inputs")) or supplied
            if supplied_inputs:
                inputs = supplied_inputs
                source = "candidate_bound_alpha_concept_inputs"
        if engine_id == "sequential_change_point_stability" and returns_by_profile:
            inputs = {"values_by_group": returns_by_profile}
            source = "paper_performance_candidate_post_cost_daily_series"
        elif engine_id == "residual_redundancy_graph" and returns_by_profile:
            inputs = {"returns_by_group": returns_by_profile}
            source = "paper_performance_candidate_post_cost_daily_series"
        try:
            result = MEASUREMENT_FUNCTIONS[engine_id](
                **inputs, **_engine_parameters(spec)
            )
        except (TypeError, ValueError, np.linalg.LinAlgError) as exc:
            result = {
                "method": str(spec.get("function") or engine_id),
                "status": "insufficient_evidence",
                "available": False,
                "passes": False,
                "blockers": [f"invalid_measurement_input:{type(exc).__name__}"],
            }
        result = dict(result)
        result["engine_id"] = engine_id
        result["concept_ids"] = list(spec.get("concept_ids") or [])
        result["input_source"] = source
        result["candidate_bound"] = bool(binding.get("bound")) and (
            source == "paper_performance_candidate_post_cost_daily_series"
            or bool(input_binding.get("bound"))
        )
        measurements[engine_id] = result

    gaps = [
        _collection_gap(engine_id, _as_dict(policy_engines.get(engine_id)), result)
        for engine_id, result in measurements.items()
        if not bool(result.get("available", False))
    ]
    active_spec = _as_dict(policy_engines.get("active_learning_value_of_information"))
    active_result = MEASUREMENT_FUNCTIONS["active_learning_value_of_information"](
        collection_gaps=gaps,
        **_engine_parameters(active_spec),
    )
    active_result["engine_id"] = "active_learning_value_of_information"
    active_result["concept_ids"] = list(active_spec.get("concept_ids") or [])
    active_result["input_source"] = "derived_from_candidate_measurement_deficits"
    active_result["candidate_bound"] = bool(binding.get("bound"))
    measurements["active_learning_value_of_information"] = active_result

    catalog, catalog_summary = _catalog(policy, root)
    implemented_engine_count = sum(
        engine_id in MEASUREMENT_FUNCTIONS
        and (root / "core" / "alpha_concept_engine.py").is_file()
        for engine_id in ENGINE_IDS
    )
    economic_engine_ids = set(ENGINE_IDS) - {"active_learning_value_of_information"}
    evidence_ready_count = sum(
        bool(measurements[engine_id].get("available", False))
        for engine_id in economic_engine_ids
    )
    economic_support_count = sum(
        bool(measurements[engine_id].get("passes", False))
        for engine_id in economic_engine_ids
    )
    implementation_score = 100.0 * implemented_engine_count / len(ENGINE_IDS)
    routing_score = (
        100.0
        * int(catalog_summary["owner_present_count"])
        / max(int(catalog_summary["concept_count"]), 1)
    )
    evidence_score = 100.0 * evidence_ready_count / len(economic_engine_ids)
    economic_score = 100.0 * economic_support_count / len(economic_engine_ids)
    structural_ready = bool(
        implemented_engine_count == len(ENGINE_IDS)
        and not catalog_summary["duplicate_concept_ids"]
        and int(catalog_summary["malformed_concept_count"]) == 0
        and int(catalog_summary["owner_present_count"])
        == int(catalog_summary["concept_count"])
    )
    ok = bool(structural_ready and binding.get("bound"))
    timestamp = _utc(generated_at_utc) or datetime.now(timezone.utc)
    receipt_material = {
        "candidate_binding": binding,
        "policy_sha256": _file_hash(policy_path),
        "performance_sha256": _file_hash(performance_path),
        "inputs_sha256": _file_hash(inputs_path),
        "measurements": measurements,
        "catalog_summary": catalog_summary,
    }
    receipt = _canonical_hash(receipt_material)
    blockers = list(binding.get("blockers") or [])
    if not structural_ready:
        blockers.append("alpha_concept_structural_contract_incomplete")
    return {
        "timestamp_utc": timestamp.isoformat(),
        "schema_version": 1,
        "policy_id": str(policy.get("policy_id") or ""),
        "ok": ok,
        "overall_status": (
            "blocked"
            if not ok
            else (
                "evidence_ready"
                if evidence_ready_count == len(economic_engine_ids)
                else "collecting_candidate_evidence"
            )
        ),
        "operating_mode": str(policy.get("operating_mode") or ""),
        "candidate_binding": binding,
        "measurement_input_binding": input_binding,
        "authority_contract": dict(policy.get("authority") or {}),
        "scope_contract": dict(policy.get("scope_contract") or {}),
        "resource_contract": dict(policy.get("resource_contract") or {}),
        "grades": {
            "implementation_grade": _grade(
                implementation_score,
                complete=implemented_engine_count == len(ENGINE_IDS),
            ),
            "implementation_score": round(implementation_score, 4),
            "catalog_routing_grade": _grade(
                routing_score,
                complete=int(catalog_summary["owner_present_count"])
                == int(catalog_summary["concept_count"]),
            ),
            "catalog_routing_score": round(routing_score, 4),
            "candidate_evidence_grade": _grade(
                evidence_score,
                complete=evidence_ready_count == len(economic_engine_ids),
            ),
            "candidate_evidence_score": round(evidence_score, 4),
            "economic_support_grade": _grade(
                economic_score,
                complete=economic_support_count == len(economic_engine_ids),
            ),
            "economic_support_score": round(economic_score, 4),
        },
        "measurement_engine_count": len(ENGINE_IDS),
        "implemented_measurement_engine_count": implemented_engine_count,
        "evidence_ready_measurement_engine_count": evidence_ready_count,
        "candidate_evidence_measurement_engine_count": len(economic_engine_ids),
        "advisory_measurement_engine_count": 1,
        "advisory_measurement_engine_ready_count": int(
            bool(active_result.get("available", False))
        ),
        "economically_supported_measurement_engine_count": economic_support_count,
        "economic_measurement_engine_count": len(economic_engine_ids),
        "measurements": measurements,
        "catalog_summary": catalog_summary,
        "concept_catalog": catalog,
        "collection_priorities": list(
            active_result.get("ranked_collection_gaps") or []
        ),
        "reference_basis": list(policy.get("reference_basis") or []),
        "measurement_input_template": {
            "candidate_id": str(binding.get("candidate_id") or ""),
            "timestamp_utc": "ISO-8601 timestamp at or after candidate cutoff",
            "measurements": {
                engine_id: {"inputs": _empty_inputs(engine_id)}
                for engine_id in ENGINE_IDS[:-1]
            },
        },
        "source_receipts": {
            "policy_path": str(policy_path),
            "policy_sha256": _file_hash(policy_path),
            "candidate_path": str(candidate_path),
            "candidate_sha256": _file_hash(candidate_path),
            "paper_performance_path": str(performance_path),
            "paper_performance_sha256": _file_hash(performance_path),
            "measurement_inputs_path": str(inputs_path),
            "measurement_inputs_sha256": _file_hash(inputs_path),
        },
        "report_receipt_sha256": receipt,
        "blockers": sorted(set(blockers)),
        "interpretation": {
            "implementation_a_plus": "all declared estimators exist and are structurally routed; it is not economic evidence",
            "catalog_a_plus": "all finite architecture-specific concepts have a declared local owner; it is not universal coverage or proof",
            "candidate_evidence": "only identity-matched post-cutoff inputs can make an estimator available",
            "economic_support": "only a passing estimator with mature candidate evidence counts; active-learning priority never counts as alpha support",
            "profitability_guaranteed": False,
            "live_execution_authority": False,
        },
    }


def render_markdown(payload: Mapping[str, Any]) -> str:
    grades = _as_dict(payload.get("grades"))
    binding = _as_dict(payload.get("candidate_binding"))
    lines = [
        "# Alpha Concept Measurement Report",
        "",
        f"Generated: `{payload.get('timestamp_utc', '')}`",
        f"Candidate: `{binding.get('candidate_id') or 'missing'}` (G{binding.get('candidate_generation', 0)})",
        f"Status: **{payload.get('overall_status', 'unknown')}**",
        "",
        "## Separate Truths",
        "",
        f"- Implementation: **{grades.get('implementation_grade', '')}** ({grades.get('implementation_score', 0)}%)",
        f"- Catalog routing: **{grades.get('catalog_routing_grade', '')}** ({grades.get('catalog_routing_score', 0)}%)",
        f"- Candidate evidence: **{grades.get('candidate_evidence_grade', '')}** ({grades.get('candidate_evidence_score', 0)}%)",
        f"- Economic support: **{grades.get('economic_support_grade', '')}** ({grades.get('economic_support_score', 0)}%)",
        "- Profitability guaranteed: **no**",
        "- Live execution authority: **no**",
        "",
        f"## {payload.get('measurement_engine_count', 0)} Measurements",
        "",
        "| Measurement | Evidence | Result | Source |",
        "|---|---:|---|---|",
    ]
    measurements = _as_dict(payload.get("measurements"))
    for engine_id in ENGINE_IDS:
        row = _as_dict(measurements.get(engine_id))
        lines.append(
            f"| `{engine_id}` | {'ready' if row.get('available') else 'collecting'} | {row.get('status', 'missing')} | `{row.get('input_source', '')}` |"
        )
    catalog = _as_dict(payload.get("catalog_summary"))
    lines.extend(
        [
            "",
            "## Ontology",
            "",
            f"- Families: **{catalog.get('family_count', 0)}**",
            f"- Canonical concepts: **{catalog.get('concept_count', 0)}**",
            f"- Owners present: **{catalog.get('owner_present_count', 0)}/{catalog.get('concept_count', 0)}**",
            "- Scope: finite and architecture-specific; it does not claim every possible market idea has been enumerated.",
            "",
            "## Collection Priorities",
            "",
        ]
    )
    priorities = _as_list(payload.get("collection_priorities"))
    if priorities:
        for row in priorities[:11]:
            if isinstance(row, Mapping):
                lines.append(
                    f"- {row.get('priority_rank', '')}. `{row.get('gap_id', '')}`: score `{row.get('priority_score_0_100', 0)}`, remaining floor `{row.get('remaining_observations', 0)}`"
                )
    else:
        lines.append("- No unresolved measurement gaps were ranked.")
    lines.extend(
        [
            "",
            "## Guardrails",
            "",
            "This laboratory is read-only. It cannot create trades, alter sizing or allocation, write labels, promote a candidate, rewrite soak history, submit paper or live orders, or guarantee profitability.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build the candidate-bound alpha concept measurement report."
    )
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--config", default="config/alpha_concept_registry_v1.json")
    parser.add_argument(
        "--out-file", default="governance/research/alpha_concept_report_latest.json"
    )
    parser.add_argument(
        "--markdown-file",
        default="exports/reports/operator/alpha_concept_report_latest.md",
    )
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    root = Path(args.project_root).expanduser().resolve()
    config_path = Path(args.config).expanduser()
    out_path = Path(args.out_file).expanduser()
    markdown_path = Path(args.markdown_file).expanduser()
    if not config_path.is_absolute():
        config_path = root / config_path
    if not out_path.is_absolute():
        out_path = root / out_path
    if not markdown_path.is_absolute():
        markdown_path = root / markdown_path
    payload = build_payload(root, config_path=config_path)
    write_payload(out_path, payload)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(render_markdown(payload), encoding="utf-8")
    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        grades = _as_dict(payload.get("grades"))
        print(
            "alpha_concepts "
            f"status={payload.get('overall_status', '')} "
            f"candidate={_as_dict(payload.get('candidate_binding')).get('candidate_id') or 'missing'} "
            f"implementation={grades.get('implementation_grade', '')} "
            f"evidence={payload.get('evidence_ready_measurement_engine_count', 0)}/{payload.get('candidate_evidence_measurement_engine_count', 10)} "
            f"economic={payload.get('economically_supported_measurement_engine_count', 0)}/{payload.get('economic_measurement_engine_count', 10)}"
        )
    return 0 if bool(payload.get("ok", False)) else 2


if __name__ == "__main__":
    raise SystemExit(main())
