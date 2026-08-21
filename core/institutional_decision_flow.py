from __future__ import annotations

import hashlib
import json
import math
from copy import deepcopy
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from core.sleeve_strategy_specialization import strategy_specialization_guard_reasons


PROJECT_ROOT = Path(__file__).resolve().parents[1]
POLICY_PATH = PROJECT_ROOT / "config" / "institutional_decision_flow_v1.json"
DIRECTIONAL_ACTIONS = frozenset({"BUY", "SELL"})
COMPONENT_NAMES = frozenset(
    {
        "data_integrity",
        "signal_conviction",
        "ensemble_consensus",
        "regime_alignment",
        "net_edge_quality",
        "execution_quality",
        "portfolio_fit",
        "risk_headroom",
        "evidence_maturity",
        "long_term_alignment",
    }
)
NON_EXECUTION_LIFECYCLES = frozenset(
    {
        "collect_only",
        "data_collection_only",
        "research_only",
        "training_excluded",
        "training_only",
    }
)
NON_MONOTONIC_CONTROL_FLAGS = frozenset(
    {
        "can_create_intent",
        "can_reverse_intent",
        "can_increase_quantity",
        "can_bypass_existing_hold",
        "can_submit_live_order",
        "can_grant_promotion",
    }
)
STRATEGY_DEFINITION_FIELDS = (
    "decision_horizon",
    "portfolio_role",
    "primary_edge",
    "entry_style",
    "exit_style",
    "sizing_method",
    "regime_dependency",
    "cost_model",
    "uncertainty_method",
    "capacity_method",
    "validation_method",
    "shorting_policy",
    "allowed_position_transitions",
)
QUANTITATIVE_EVIDENCE_AXES = frozenset(
    {
        "selection_bias_control",
        "independent_samples",
        "uncertainty_calibration",
        "signal_decay_fit",
        "payoff_asymmetry",
        "capacity_headroom",
        "crowding_residual",
        "tail_survival",
        "regime_stability",
    }
)
STAGE_COMPONENTS = {
    "02_data_qualification": ("data_integrity",),
    "03_signal_formation": ("signal_conviction",),
    "04_consensus_and_regime": ("ensemble_consensus", "regime_alignment"),
    "05_post_cost_edge": ("net_edge_quality",),
    "06_execution_feasibility": ("execution_quality",),
    "07_portfolio_fit": ("portfolio_fit", "long_term_alignment"),
    "08_non_bypassable_risk": ("risk_headroom",),
    "09_shadow_priority": ("evidence_maturity",),
}


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _number(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float(default)
    return result if math.isfinite(result) else float(default)


def _clamp01(value: Any) -> float:
    return min(max(_number(value), 0.0), 1.0)


def _mean(values: Sequence[float], default: float = 0.0) -> float:
    return sum(values) / len(values) if values else float(default)


def _first_number(
    sources: Sequence[Mapping[str, Any]],
    keys: Sequence[str],
) -> tuple[float | None, str]:
    for key in keys:
        for source in sources:
            if key not in source or source.get(key) in {None, ""}:
                continue
            value = _number(source.get(key), math.nan)
            if math.isfinite(value):
                return value, key
    return None, ""


def _canonical_hash(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _resolved_decision_playbook(
    *,
    profile: str,
    family_id: str,
    family: Mapping[str, Any],
    strategy_definition: Mapping[str, Any],
    policy: Mapping[str, Any],
    profile_override: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind the sleeve's economic definition to one explicit stage playbook."""

    weights = _mapping(family.get("component_weights"))
    stage_priority = [
        {
            "stage_id": stage_id,
            "weight_norm": round(
                sum(_number(weights.get(component)) for component in components),
                6,
            ),
            "components": list(components),
        }
        for stage_id, components in STAGE_COMPONENTS.items()
    ]
    stage_priority.sort(
        key=lambda item: (-_number(item.get("weight_norm")), str(item.get("stage_id")))
    )
    paper_control = _mapping(policy.get("active_paper_control"))
    live_control = _mapping(policy.get("active_live_control"))
    material = {
        "contract_version": "resolved_sleeve_decision_playbook_v1",
        "profile": profile,
        "policy_family_id": family_id,
        "objective": str(family.get("objective") or ""),
        "decision_horizon": str(strategy_definition.get("decision_horizon") or ""),
        "portfolio_role": str(strategy_definition.get("portfolio_role") or ""),
        "primary_edge": str(strategy_definition.get("primary_edge") or ""),
        "entry_contract": str(strategy_definition.get("entry_style") or ""),
        "exit_contract": str(strategy_definition.get("exit_style") or ""),
        "sizing_contract": str(strategy_definition.get("sizing_method") or ""),
        "regime_contract": str(strategy_definition.get("regime_dependency") or ""),
        "cost_contract": str(strategy_definition.get("cost_model") or ""),
        "uncertainty_contract": str(strategy_definition.get("uncertainty_method") or ""),
        "capacity_contract": str(strategy_definition.get("capacity_method") or ""),
        "validation_contract": str(strategy_definition.get("validation_method") or ""),
        "shorting_contract": str(strategy_definition.get("shorting_policy") or ""),
        "allowed_position_transitions": sorted(
            str(value)
            for value in (strategy_definition.get("allowed_position_transitions") or [])
            if str(value).strip()
        ),
        "family_evidence_focus": [
            str(value) for value in (family.get("evidence_focus") or [])
        ],
        "required_quantitative_evidence": sorted(
            str(value)
            for value in (
                strategy_definition.get("required_quantitative_evidence") or []
            )
            if str(value).strip()
        ),
        "stage_sequence": [
            str(_mapping(stage).get("stage_id") or "")
            for stage in (policy.get("stages") or [])
            if str(_mapping(stage).get("stage_id") or "").strip()
        ],
        "stage_priority": stage_priority,
        "paper_required_stages": [
            str(value) for value in (paper_control.get("required_pass_stages") or [])
        ],
        "live_required_stages": [
            str(value) for value in (live_control.get("required_pass_stages") or [])
        ],
        "profile_override_fields": sorted(str(key) for key in profile_override),
        "paper_live_same_thesis": bool(
            _mapping(policy.get("mode_parity_contract")).get(
                "paper_and_live_use_same_resolved_policy", False
            )
        ),
    }
    return {**material, "playbook_sha256": _canonical_hash(material)}


def _utc_text(value: Any) -> str:
    raw = str(value or "").strip().replace("Z", "+00:00")
    if not raw:
        return ""
    try:
        parsed = datetime.fromisoformat(raw)
    except ValueError:
        return ""
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc).isoformat()


def _utc_datetime(value: Any) -> datetime | None:
    normalized = _utc_text(value)
    if not normalized:
        return None
    try:
        return datetime.fromisoformat(normalized)
    except ValueError:
        return None


def _artifact_matches_candidate(
    payload: Mapping[str, Any],
    *,
    candidate_id: str,
    candidate_cutoff_utc: datetime,
) -> bool:
    binding = _mapping(payload.get("candidate_binding"))
    artifact_candidate_id = str(
        binding.get("candidate_id")
        or _mapping(payload.get("profitability_evidence_window")).get("candidate_id")
        or ""
    ).strip()
    generated_at = _utc_datetime(payload.get("timestamp_utc"))
    return bool(
        candidate_id
        and artifact_candidate_id == candidate_id
        and bool(binding.get("bound", False))
        and generated_at is not None
        and generated_at >= candidate_cutoff_utc
    )


def build_candidate_bound_quantitative_evidence(
    profile: str,
    *,
    paper_performance: Mapping[str, Any],
    multiple_testing: Mapping[str, Any] | None = None,
    independent_validator: Mapping[str, Any] | None = None,
    decay_monitor: Mapping[str, Any] | None = None,
    quantitative_challengers: Mapping[str, Any] | None = None,
    expected_candidate_id: str = "",
) -> dict[str, Any]:
    """Translate verified profitability artifacts into direct decision evidence.

    Missing or candidate-mismatched artifacts intentionally produce no axis score.
    That keeps paper collection available while future live execution fails closed.
    """

    profile_name = str(profile or "default").strip().lower() or "default"
    window = _mapping(paper_performance.get("profitability_evidence_window"))
    candidate_id = str(window.get("candidate_id") or "").strip()
    candidate_cutoff = _utc_datetime(window.get("candidate_cutoff_utc"))
    evidence_through = _utc_datetime(window.get("evidence_through_utc"))
    mismatch_count = max(
        int(_number(window.get("candidate_binding_mismatch_rows_excluded"), 0)),
        0,
    )
    binding_reasons: list[str] = []
    if not candidate_id:
        binding_reasons.append("candidate_id_missing")
    if expected_candidate_id and candidate_id != str(expected_candidate_id).strip():
        binding_reasons.append("candidate_id_mismatch")
    if not bool(window.get("candidate_filter_active", False)):
        binding_reasons.append("candidate_filter_inactive")
    if not bool(window.get("candidate_binding_required", False)):
        binding_reasons.append("candidate_binding_not_required")
    if candidate_cutoff is None:
        binding_reasons.append("candidate_cutoff_missing")
    if evidence_through is None:
        binding_reasons.append("evidence_watermark_missing")
    elif candidate_cutoff is not None and evidence_through < candidate_cutoff:
        binding_reasons.append("evidence_watermark_precedes_candidate")
    if mismatch_count:
        binding_reasons.append("candidate_binding_mismatch_rows_present")

    packet: dict[str, Any] = {}
    source_receipts: dict[str, Any] = {}
    if binding_reasons or candidate_cutoff is None:
        packet["_bridge"] = {
            "status": "unbound",
            "profile": profile_name,
            "candidate_id": candidate_id,
            "reasons": binding_reasons,
            "direct_axes": [],
        }
        return packet

    sleeve = next(
        (
            _mapping(raw)
            for raw in (paper_performance.get("sleeve_latest") or [])
            if isinstance(raw, Mapping)
            and str(raw.get("profile") or "").strip().lower() == profile_name
        ),
        {},
    )
    expectancy = _mapping(sleeve.get("post_cost_expectancy"))
    robust = _mapping(expectancy.get("robust_statistics"))
    if bool(robust.get("available", False)):
        thresholds = _mapping(robust.get("thresholds"))
        target = max(_number(thresholds.get("minimum_effective_samples"), 20.0), 1.0)
        effective_samples = max(_number(robust.get("effective_sample_size"), 0.0), 0.0)
        packet["independent_samples"] = round(
            min(effective_samples / target, 1.0),
            8,
        )
        packet["effective_sample_size"] = round(effective_samples, 6)
        packet["effective_sample_target"] = round(target, 6)
        source_receipts["independent_samples"] = {
            "artifact": "paper_performance_latest.json",
            "source": "sleeve_latest.post_cost_expectancy.robust_statistics.effective_sample_size",
            "effective_sample_size": round(effective_samples, 6),
            "target": round(target, 6),
        }

    payoff = _mapping(expectancy.get("payoff_asymmetry"))
    if bool(payoff.get("available", False)):
        ratio = max(_number(payoff.get("average_win_to_average_loss_ratio"), 0.0), 0.0)
        packet["payoff_asymmetry"] = round(ratio / (1.0 + ratio), 8)
        source_receipts["payoff_asymmetry"] = {
            "artifact": "paper_performance_latest.json",
            "source": "sleeve_latest.post_cost_expectancy.payoff_asymmetry",
            "average_win_to_average_loss_ratio": round(ratio, 8),
            "positive_sample_count": int(_number(payoff.get("positive_sample_count"), 0)),
            "negative_sample_count": int(_number(payoff.get("negative_sample_count"), 0)),
        }

    multiple_payload = _mapping(multiple_testing)
    if _artifact_matches_candidate(
        multiple_payload,
        candidate_id=candidate_id,
        candidate_cutoff_utc=candidate_cutoff,
    ):
        correction = _mapping(multiple_payload.get("actual_statistical_correction"))
        correction_row = next(
            (
                _mapping(raw)
                for raw in (correction.get("rows") or [])
                if isinstance(raw, Mapping)
                and str(raw.get("hypothesis_id") or "").strip().lower()
                == profile_name
            ),
            {},
        )
        pbo = _mapping(multiple_payload.get("probability_of_backtest_overfitting"))
        dsr_by_sleeve = _mapping(
            multiple_payload.get("deflated_sharpe_available_by_sleeve")
        )
        dsr = _mapping(dsr_by_sleeve.get(profile_name)) or _mapping(
            robust.get("deflated_sharpe")
        )
        if (
            correction_row.get("q_value") is not None
            and bool(pbo.get("available", False))
            and pbo.get("pbo") is not None
            and bool(dsr.get("available", False))
            and dsr.get("probability") is not None
        ):
            q_resilience = 1.0 - _clamp01(correction_row.get("q_value"))
            pbo_resilience = 1.0 - _clamp01(pbo.get("pbo"))
            dsr_probability = _clamp01(dsr.get("probability"))
            packet["selection_bias_control"] = round(
                min(q_resilience, pbo_resilience, dsr_probability),
                8,
            )
            source_receipts["selection_bias_control"] = {
                "artifact": "multiple_testing_guard_latest.json",
                "source": "minimum_fdr_pbo_and_deflated_sharpe_resilience",
                "q_value": round(_clamp01(correction_row.get("q_value")), 8),
                "pbo": round(_clamp01(pbo.get("pbo")), 8),
                "deflated_sharpe_probability": round(dsr_probability, 8),
            }

    validator_payload = _mapping(independent_validator)
    if _artifact_matches_candidate(
        validator_payload,
        candidate_id=candidate_id,
        candidate_cutoff_utc=candidate_cutoff,
    ):
        risk = _mapping(validator_payload.get("risk_of_ruin"))
        if bool(risk.get("available", False)):
            ruin = _clamp01(risk.get("ruin_probability"))
            drawdown = _clamp01(risk.get("drawdown_breach_probability"))
            packet["tail_survival"] = round(1.0 - max(ruin, drawdown), 8)
            source_receipts["tail_survival"] = {
                "artifact": "profitability_independent_validator_latest.json",
                "source": "risk_of_ruin_and_drawdown_block_bootstrap",
                "ruin_probability": round(ruin, 8),
                "drawdown_breach_probability": round(drawdown, 8),
                "independent_day_count": int(_number(risk.get("day_count"), 0)),
            }

    decay_payload = _mapping(decay_monitor)
    if _artifact_matches_candidate(
        decay_payload,
        candidate_id=candidate_id,
        candidate_cutoff_utc=candidate_cutoff,
    ):
        edge_contract = _mapping(decay_payload.get("edge_decay_contract"))
        decay_row = next(
            (
                _mapping(raw)
                for raw in (edge_contract.get("profiles") or [])
                if isinstance(raw, Mapping)
                and str(raw.get("profile") or "").strip().lower() == profile_name
            ),
            {},
        )
        if decay_row:
            decline = _clamp01(decay_row.get("mean_decay_fraction"))
            score = 0.0 if bool(decay_row.get("decayed", False)) else 1.0 - decline
            packet["signal_decay_fit"] = round(_clamp01(score), 8)
            source_receipts["signal_decay_fit"] = {
                "artifact": "decay_monitor_latest.json",
                "source": "candidate_forward_profile_daily_post_cost_decay",
                "history_days": int(_number(decay_row.get("history_days"), 0)),
                "mean_decay_fraction": round(decline, 8),
                "decayed": bool(decay_row.get("decayed", False)),
            }

    challenger_payload = _mapping(quantitative_challengers)
    challenger_bound = _artifact_matches_candidate(
        challenger_payload,
        candidate_id=candidate_id,
        candidate_cutoff_utc=candidate_cutoff,
    )
    if challenger_bound:
        metadata_by_profile = _mapping(
            challenger_payload.get("decision_metadata_by_profile")
        )
        profile_metadata = _mapping(metadata_by_profile.get(profile_name))
        if profile_metadata:
            packet["_challengers"] = {
                **profile_metadata,
                "report_status": str(
                    challenger_payload.get("overall_status") or ""
                ),
                "implemented_concept_count": int(
                    _number(
                        challenger_payload.get("implemented_concept_count"),
                        0,
                    )
                ),
                "evidence_ready_concept_count": int(
                    _number(
                        challenger_payload.get("evidence_ready_concept_count"),
                        0,
                    )
                ),
                "supported_concept_count": int(
                    _number(
                        challenger_payload.get("supported_concept_count"),
                        0,
                    )
                ),
                "authority_contract": _mapping(
                    challenger_payload.get("authority_contract")
                ),
                "changes_active_evidence_axes": False,
                "changes_decision_utility": False,
                "changes_order_quantity": False,
            }

    direct_axes = sorted(set(packet).intersection(QUANTITATIVE_EVIDENCE_AXES))
    receipt_material = {
        "candidate_id": candidate_id,
        "profile": profile_name,
        "sources": source_receipts,
        "challenger_report_receipt_sha256": str(
            _mapping(packet.get("_challengers")).get(
                "report_receipt_sha256", ""
            )
        ),
    }
    packet["_bridge"] = {
        "status": "bound",
        "profile": profile_name,
        "candidate_id": candidate_id,
        "candidate_cutoff_utc": candidate_cutoff.isoformat(),
        "evidence_through_utc": evidence_through.isoformat()
        if evidence_through is not None
        else "",
        "direct_axes": direct_axes,
        "sources": source_receipts,
        "challenger_report_bound": bool(packet.get("_challengers")),
        "receipt_sha256": _canonical_hash(receipt_material),
    }
    return packet


def _validate_component_weights(weights: Mapping[str, Any], *, label: str) -> None:
    if set(weights) != COMPONENT_NAMES or not math.isclose(
        sum(_number(value) for value in weights.values()),
        1.0,
        rel_tol=1e-9,
        abs_tol=1e-9,
    ):
        raise ValueError(f"{label} component weights must define every component and sum to 1.0")


def _validate_monotonic_control(
    control: Mapping[str, Any],
    *,
    target_mode: str,
) -> None:
    if not bool(control.get("enabled", False)):
        return
    if str(control.get("target_mode") or "").strip().lower() != target_mode:
        raise ValueError(f"active decision-flow control must target {target_mode} mode")
    if any(bool(control.get(key, False)) for key in NON_MONOTONIC_CONTROL_FLAGS):
        raise ValueError(f"active {target_mode} control requests non-monotonic authority")
    for key, raw in control.items():
        if not str(key).endswith("_quantity_multiplier"):
            continue
        value = _number(raw, 0.0)
        if not 0.0 <= value <= 1.0:
            raise ValueError(
                f"active {target_mode} control multiplier must be in [0, 1]: {key}"
            )


def load_policy(path: Path | str = POLICY_PATH) -> dict[str, Any]:
    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("decision-flow policy must be a JSON object")
    stage_ids = [
        str(_mapping(stage).get("stage_id") or "")
        for stage in (payload.get("stages") or [])
    ]
    expected_stage_ids = [
        "01_observation",
        "02_data_qualification",
        "03_signal_formation",
        "04_consensus_and_regime",
        "05_post_cost_edge",
        "06_execution_feasibility",
        "07_portfolio_fit",
        "08_non_bypassable_risk",
        "09_shadow_priority",
        "10_outcome_learning",
    ]
    if stage_ids != expected_stage_ids:
        raise ValueError("decision-flow stages must retain the ordered ten-stage contract")
    weights = _mapping(payload.get("component_weights"))
    _validate_component_weights(weights, label="decision-flow")
    authority = _mapping(payload.get("authority"))
    forbidden_true = {
        "change_active_action",
        "change_position_size",
        "submit_paper_order",
        "submit_live_order",
        "mutate_registry",
        "accept_candidate_change",
        "grant_promotion",
    }
    if any(bool(authority.get(key, False)) for key in forbidden_true):
        raise ValueError("shadow decision-flow policy requests forbidden authority")
    _validate_monotonic_control(
        _mapping(payload.get("active_paper_control")),
        target_mode="paper",
    )
    live_control = _mapping(payload.get("active_live_control"))
    _validate_monotonic_control(live_control, target_mode="live")
    for control_name in ("active_paper_control", "active_live_control"):
        required_stages = {
            str(value)
            for value in (
                _mapping(payload.get(control_name)).get("required_pass_stages") or []
            )
        }
        if not required_stages.issubset(set(stage_ids)):
            raise ValueError(f"{control_name} references an unknown decision stage")
    if bool(live_control.get("enabled", False)):
        strict_live_flags = (
            "require_current_policy_receipt",
            "require_qualified_candidate",
            "require_execution_eligible_sleeve",
            "require_complete_strategy_definition",
            "require_quantitative_evidence_ready",
            "require_action_semantics_ready",
        )
        if not all(bool(live_control.get(key, False)) for key in strict_live_flags):
            raise ValueError("active live control must retain strict receipt and qualification gates")
        required_live_stages = {
            "02_data_qualification",
            "03_signal_formation",
            "04_consensus_and_regime",
            "05_post_cost_edge",
            "06_execution_feasibility",
            "07_portfolio_fit",
            "08_non_bypassable_risk",
            "09_shadow_priority",
        }
        if not required_live_stages.issubset(
            {str(value) for value in (live_control.get("required_pass_stages") or [])}
        ):
            raise ValueError("active live control may not omit a qualification stage")
        if str(live_control.get("missing_edge_action") or "").lower() != "veto" or str(
            live_control.get("point_estimate_action") or ""
        ).lower() != "veto":
            raise ValueError("active live control must veto incomplete edge evidence")
    parity = _mapping(payload.get("mode_parity_contract"))
    if bool(_mapping(payload.get("active_live_control")).get("enabled", False)) and not bool(
        parity.get("paper_and_live_use_same_resolved_policy", False)
    ):
        raise ValueError("live decision-flow control requires paper/live resolved-policy parity")

    families = _mapping(payload.get("sleeve_policy_families"))
    if not families:
        raise ValueError("decision-flow policy must define sleeve policy families")
    for family_id, raw_family in families.items():
        family = _mapping(raw_family)
        _validate_component_weights(
            _mapping(family.get("component_weights")),
            label=f"sleeve family {family_id}",
        )
        for key, value in _mapping(family.get("qualification_floor_overrides")).items():
            numeric = _number(value, math.nan)
            if not math.isfinite(numeric) or numeric < 0.0:
                raise ValueError(f"invalid sleeve qualification floor: {family_id}.{key}")
    family_ids = set(families)

    strategy_contract = _mapping(payload.get("strategy_definition_contract"))
    required_strategy_fields = {
        str(value) for value in (strategy_contract.get("required_fields") or [])
    }
    if required_strategy_fields != set(STRATEGY_DEFINITION_FIELDS):
        raise ValueError("strategy definition contract must declare every required field")
    strategy_definitions = _mapping(payload.get("strategy_definitions"))
    if set(strategy_definitions) != family_ids:
        raise ValueError("strategy definitions must cover every sleeve policy family exactly")

    quantitative_contract = _mapping(payload.get("quantitative_evidence_contract"))
    quantitative_axes = _mapping(quantitative_contract.get("axes"))
    if set(quantitative_axes) != QUANTITATIVE_EVIDENCE_AXES:
        raise ValueError("quantitative evidence contract must define every supported axis")
    for axis, raw_axis_contract in quantitative_axes.items():
        axis_contract = _mapping(raw_axis_contract)
        floor = _number(axis_contract.get("floor"), math.nan)
        if not math.isfinite(floor) or not 0.0 <= floor <= 1.0:
            raise ValueError(f"invalid quantitative evidence floor: {axis}")

    for family_id, raw_definition in strategy_definitions.items():
        definition = _mapping(raw_definition)
        missing = [
            field
            for field in STRATEGY_DEFINITION_FIELDS
            if field not in definition
            or (
                field != "allowed_position_transitions"
                and not str(definition.get(field) or "").strip()
            )
        ]
        transitions = definition.get("allowed_position_transitions")
        if not isinstance(transitions, list) or not transitions:
            missing.append("allowed_position_transitions")
        if missing:
            raise ValueError(
                f"incomplete strategy definition for {family_id}: {','.join(sorted(set(missing)))}"
            )
        required_axes = {
            str(value)
            for value in (definition.get("required_quantitative_evidence") or [])
            if str(value).strip()
        }
        if not required_axes or not required_axes.issubset(QUANTITATIVE_EVIDENCE_AXES):
            raise ValueError(f"invalid quantitative evidence requirements for {family_id}")

    allowed_override_fields = set(STRATEGY_DEFINITION_FIELDS) | {
        "required_quantitative_evidence"
    }
    for profile, raw_override in _mapping(
        payload.get("profile_strategy_overrides")
    ).items():
        override = _mapping(raw_override)
        unknown = set(override) - allowed_override_fields
        if unknown:
            raise ValueError(
                f"unknown strategy override fields for {profile}: {','.join(sorted(unknown))}"
            )
        if not override:
            raise ValueError(f"empty strategy override for {profile}")

    for profile, family_id in _mapping(payload.get("profile_policy_map")).items():
        if str(family_id) not in family_ids:
            raise ValueError(f"unknown sleeve policy family for profile {profile}: {family_id}")
    for rule in payload.get("profile_policy_rules") or []:
        if not isinstance(rule, Mapping):
            raise ValueError("profile policy rules must be objects")
        family_id = str(rule.get("policy_family_id") or "")
        if family_id not in family_ids:
            raise ValueError(f"unknown sleeve policy family in rule: {family_id}")
    return payload


def _token_match(value: str, tokens: Sequence[Any]) -> bool:
    return any(str(token or "").strip().lower() in value for token in tokens if str(token or "").strip())


def resolve_sleeve_policy(
    profile: str,
    policy: Mapping[str, Any],
    *,
    domain: str = "",
    lifecycle_state: str = "",
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Resolve one deterministic policy for a sleeve and its paper/live lifecycle."""

    profile_name = str(profile or "default").strip().lower() or "default"
    domain_name = str(domain or "").strip().lower()
    lifecycle = str(lifecycle_state or "").strip().lower()
    families = _mapping(policy.get("sleeve_policy_families"))
    profile_map = {
        str(key).strip().lower(): str(value).strip()
        for key, value in _mapping(policy.get("profile_policy_map")).items()
    }
    family_id = ""
    match_source = ""
    matched_rule_index: int | None = None
    if domain_name:
        for index, raw_rule in enumerate(policy.get("profile_policy_rules") or []):
            rule = _mapping(raw_rule)
            if _token_match(domain_name, rule.get("domain_tokens_any") or []):
                family_id = str(rule.get("policy_family_id") or "")
                match_source = "domain_rule"
                matched_rule_index = index
                break
    if not family_id:
        family_id = profile_map.get(profile_name, "")
        match_source = "exact_profile" if family_id else ""
    if not family_id:
        for index, raw_rule in enumerate(policy.get("profile_policy_rules") or []):
            rule = _mapping(raw_rule)
            profile_match = _token_match(profile_name, rule.get("profile_tokens_any") or [])
            if profile_match:
                family_id = str(rule.get("policy_family_id") or "")
                match_source = "profile_rule"
                matched_rule_index = index
                break
    if not family_id:
        family_id = "balanced_directional"
        match_source = "default_fallback"
    family = _mapping(families.get(family_id))
    if not family:
        raise ValueError(f"resolved sleeve policy family is missing: {family_id}")

    family_definition = deepcopy(
        _mapping(_mapping(policy.get("strategy_definitions")).get(family_id))
    )
    profile_override = deepcopy(
        _mapping(_mapping(policy.get("profile_strategy_overrides")).get(profile_name))
    )
    strategy_definition = {**family_definition, **profile_override}
    strategy_definition_complete = bool(
        all(
            field in strategy_definition
            and (
                bool(strategy_definition.get(field))
                if field == "allowed_position_transitions"
                else bool(str(strategy_definition.get(field) or "").strip())
            )
            for field in STRATEGY_DEFINITION_FIELDS
        )
    )
    required_quantitative_evidence = sorted(
        {
            str(value)
            for value in (
                strategy_definition.get("required_quantitative_evidence") or []
            )
            if str(value).strip()
        }
    )
    strategy_definition_sha256 = _canonical_hash(strategy_definition)
    strategy_variant_id = (
        f"{family_id}::{profile_name}::{strategy_definition_sha256[:16]}"
    )
    decision_playbook = _resolved_decision_playbook(
        profile=profile_name,
        family_id=family_id,
        family=family,
        strategy_definition=strategy_definition,
        policy=policy,
        profile_override=profile_override,
    )

    resolved = dict(policy)
    resolved["component_weights"] = deepcopy(_mapping(family.get("component_weights")))
    resolved["qualification_floors"] = {
        **deepcopy(_mapping(policy.get("qualification_floors"))),
        **deepcopy(_mapping(family.get("qualification_floor_overrides"))),
    }
    resolved["market_quality"] = {
        **deepcopy(_mapping(policy.get("market_quality"))),
        **deepcopy(_mapping(family.get("market_quality_overrides"))),
    }
    resolved["strategy_definition"] = deepcopy(strategy_definition)
    resolved["decision_playbook"] = deepcopy(decision_playbook)
    resolved["required_quantitative_evidence"] = list(
        required_quantitative_evidence
    )
    lifecycle_eligible = lifecycle not in NON_EXECUTION_LIFECYCLES
    execution_eligible = bool(family.get("execution_eligible_default", False)) and lifecycle_eligible
    policy_material = {
        "schema_version": int(_number(policy.get("schema_version"), 0)),
        "base_policy_id": str(policy.get("policy_id") or ""),
        "policy_family_id": family_id,
        "profile": profile_name,
        "domain": domain_name,
        "lifecycle_state": lifecycle,
        "execution_eligible": execution_eligible,
        "objective": str(family.get("objective") or ""),
        "evidence_focus": [str(item) for item in (family.get("evidence_focus") or [])],
        "strategy_variant_id": strategy_variant_id,
        "strategy_definition": strategy_definition,
        "strategy_definition_sha256": strategy_definition_sha256,
        "strategy_definition_complete": strategy_definition_complete,
        "decision_playbook": decision_playbook,
        "required_quantitative_evidence": required_quantitative_evidence,
        "component_weights": resolved["component_weights"],
        "qualification_floors": resolved["qualification_floors"],
        "market_quality": resolved["market_quality"],
        "active_paper_control": _mapping(resolved.get("active_paper_control")),
        "active_live_control": _mapping(resolved.get("active_live_control")),
        "mode_parity_contract": _mapping(resolved.get("mode_parity_contract")),
        "strategy_definition_contract": _mapping(
            resolved.get("strategy_definition_contract")
        ),
        "quantitative_evidence_contract": _mapping(
            resolved.get("quantitative_evidence_contract")
        ),
    }
    resolved_sha256 = _canonical_hash(policy_material)
    receipt = {
        "schema_version": 3,
        "base_policy_id": str(policy.get("policy_id") or ""),
        "policy_family_id": family_id,
        "resolved_policy_id": (
            f"{str(policy.get('policy_id') or 'decision_flow')}::{family_id}::{resolved_sha256[:16]}"
        ),
        "resolved_policy_sha256": resolved_sha256,
        "profile": profile_name,
        "domain": domain_name,
        "lifecycle_state": lifecycle,
        "execution_eligible": execution_eligible,
        "match_source": match_source,
        "matched_rule_index": matched_rule_index,
        "objective": str(family.get("objective") or ""),
        "evidence_focus": [str(item) for item in (family.get("evidence_focus") or [])],
        "strategy_variant_id": strategy_variant_id,
        "strategy_definition_sha256": strategy_definition_sha256,
        "strategy_definition_complete": strategy_definition_complete,
        "decision_playbook_version": str(
            decision_playbook.get("contract_version") or ""
        ),
        "decision_playbook_sha256": str(
            decision_playbook.get("playbook_sha256") or ""
        ),
        "decision_stage_priority": [
            str(item.get("stage_id") or "")
            for item in decision_playbook.get("stage_priority", [])
        ],
        "decision_horizon": str(strategy_definition.get("decision_horizon") or ""),
        "portfolio_role": str(strategy_definition.get("portfolio_role") or ""),
        "primary_edge": str(strategy_definition.get("primary_edge") or ""),
        "required_quantitative_evidence": required_quantitative_evidence,
        "paper_live_policy_parity": bool(
            _mapping(policy.get("mode_parity_contract")).get(
                "paper_and_live_use_same_resolved_policy", False
            )
        ),
    }
    resolved["resolved_sleeve_policy"] = deepcopy(receipt)
    resolved["resolved_sleeve_family"] = deepcopy(family)
    return resolved, receipt


def _policy_for_evaluation(
    row: Mapping[str, Any],
    policy: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    profile = str(row.get("shadow_profile") or row.get("profile") or "default")
    domain = str(row.get("shadow_domain") or row.get("domain") or "")
    lifecycle = str(row.get("lifecycle_state") or "")
    existing = _mapping(policy.get("resolved_sleeve_policy"))
    if (
        existing
        and str(existing.get("profile") or "").strip().lower()
        == str(profile or "default").strip().lower()
        and str(existing.get("domain") or "").strip().lower()
        == str(domain or "").strip().lower()
        and str(existing.get("lifecycle_state") or "").strip().lower()
        == str(lifecycle or "").strip().lower()
    ):
        return dict(policy), deepcopy(existing)
    return resolve_sleeve_policy(
        profile,
        policy,
        domain=domain,
        lifecycle_state=lifecycle,
    )


def _stage_passed(evaluation: Mapping[str, Any], stage_id: str) -> bool:
    for stage in evaluation.get("stages") or []:
        if not isinstance(stage, Mapping):
            continue
        if str(stage.get("stage_id") or "") == stage_id:
            return bool(stage.get("passed", False))
    return False


def _normalized_action(value: Any) -> str:
    action = str(value or "HOLD").strip().upper()
    return action if action in {*DIRECTIONAL_ACTIONS, "HOLD"} else "HOLD"


def _bounded_ingestion_route(value: Any) -> dict[str, Any]:
    route = _mapping(value)
    material = {
        "contract_version": "decision_ingestion_route_v1",
        "status": str(route.get("status") or "unavailable"),
        "route_state": str(route.get("route_state") or "unavailable"),
        "runtime_profile": str(route.get("runtime_profile") or ""),
        "requested_runtime_profile": str(
            route.get("requested_runtime_profile") or ""
        ),
        "fallback_profile_used": bool(route.get("fallback_profile_used", False)),
        "decision_policy_family_id": str(
            route.get("decision_policy_family_id") or ""
        ),
        "ingestion_lane": str(route.get("ingestion_lane") or ""),
        "cadence": str(route.get("cadence") or ""),
        "profile_id": str(route.get("profile_id") or ""),
        "average_route_score_norm": round(
            _number(route.get("average_route_score")), 6
        ),
        "paper_required_coverage_norm": round(
            _number(route.get("paper_required_capability_coverage_ratio")), 6
        ),
        "live_required_coverage_norm": round(
            _number(route.get("live_required_capability_coverage_ratio")), 6
        ),
        "independent_failover_coverage_norm": round(
            _number(route.get("independent_failover_coverage_ratio")), 6
        ),
        "selected_producer_count": max(
            int(_number(route.get("selected_producer_count"))), 0
        ),
        "paper_decision_data_ready": bool(
            route.get("paper_decision_data_ready", False)
        ),
        "live_decision_data_ready": bool(
            route.get("live_decision_data_ready", False)
        ),
        "missing_paper_required_capability_ids": [
            str(item)
            for item in (
                route.get("missing_paper_required_capability_ids") or []
            )[:8]
            if str(item)
        ],
        "missing_live_required_capability_ids": [
            str(item)
            for item in (route.get("missing_required_capability_ids") or [])[:8]
            if str(item)
        ],
        "artifact_age_minutes": (
            round(_number(route.get("artifact_age_minutes")), 3)
            if route.get("artifact_age_minutes") is not None
            else None
        ),
        "artifact_fresh": bool(route.get("artifact_fresh", False)),
        "receipt_valid": bool(route.get("receipt_valid", False)),
        "routing_receipt_sha256": str(
            route.get("routing_receipt_sha256") or ""
        ),
        "route_receipt_sha256": str(route.get("route_receipt_sha256") or ""),
        "route_summary_receipt_sha256": str(
            route.get("route_summary_receipt_sha256") or ""
        ),
        "authority_contract": {
            "paper_execution_authority": False,
            "live_execution_authority": False,
            "automatic_promotion_authority": False,
        },
    }
    return {
        **material,
        "decision_route_receipt_sha256": _canonical_hash(material),
    }


def build_decision_operator_summary(
    evaluation: Mapping[str, Any],
    control: Mapping[str, Any],
) -> dict[str, Any]:
    """Return one bounded operator view shared by persisted decisions and live feed."""

    receipt = _mapping(evaluation.get("policy_receipt"))
    trace = _mapping(evaluation.get("decision_trace"))
    blocking = _mapping(trace.get("blocking"))
    progress = _mapping(trace.get("stage_progress"))
    action_semantics = _mapping(evaluation.get("action_semantics"))
    diagnostics = _mapping(evaluation.get("diagnostics"))
    quantitative = _mapping(evaluation.get("quantitative_evidence"))
    specialization = _mapping(evaluation.get("strategy_specialization"))
    ingestion_route = _mapping(evaluation.get("ingestion_route"))
    reasons = list(
        dict.fromkeys(
            [
                *[str(value) for value in (control.get("reasons") or []) if str(value)],
                *[str(value) for value in (trace.get("reason_codes") or []) if str(value)],
            ]
        )
    )[:8]
    material = {
        "contract_version": "decision_operator_summary_v1",
        "timestamp_utc": str(evaluation.get("timestamp_utc") or ""),
        "evaluation_id": str(evaluation.get("evaluation_id") or ""),
        "profile": str(evaluation.get("profile") or receipt.get("profile") or ""),
        "domain": str(evaluation.get("domain") or receipt.get("domain") or ""),
        "lane": str(evaluation.get("lane") or ""),
        "symbol": str(evaluation.get("symbol") or ""),
        "policy_family_id": str(receipt.get("policy_family_id") or ""),
        "resolved_policy_id": str(receipt.get("resolved_policy_id") or ""),
        "decision_playbook_sha256": str(
            receipt.get("decision_playbook_sha256") or ""
        ),
        "strategy_variant_id": str(receipt.get("strategy_variant_id") or ""),
        "selected_strategy_id": str(
            specialization.get("selected_strategy_id") or ""
        ),
        "objective": str(receipt.get("objective") or ""),
        "decision_horizon": str(receipt.get("decision_horizon") or ""),
        "portfolio_role": str(receipt.get("portfolio_role") or ""),
        "decision_state": str(
            trace.get("decision_state")
            or evaluation.get("classification")
            or "unknown"
        ),
        "classification": str(evaluation.get("classification") or ""),
        "control_mode": str(control.get("target_mode") or ""),
        "control_outcome": str(control.get("disposition") or ""),
        "intent_action": str(evaluation.get("intent_action") or "HOLD"),
        "evaluated_action": str(evaluation.get("final_action") or "HOLD"),
        "output_action": str(
            control.get("output_action") or evaluation.get("final_action") or "HOLD"
        ),
        "input_quantity": round(_number(control.get("input_quantity")), 6),
        "output_quantity": round(_number(control.get("output_quantity")), 6),
        "quantity_multiplier": round(_number(control.get("quantity_multiplier")), 6),
        "position_transition": str(action_semantics.get("semantic") or "unknown"),
        "position_truth_available": bool(
            action_semantics.get("position_truth_available", False)
        ),
        "current_stage": str(trace.get("current_stage") or ""),
        "next_stage": str(trace.get("next_stage") or ""),
        "blocking_stage": str(
            control.get("blocking_stage")
            or blocking.get("stage_id")
            or evaluation.get("first_failed_stage")
            or ""
        ),
        "blocking_reason_code": str(blocking.get("reason_code") or ""),
        "blocking_reason": str(blocking.get("reason") or ""),
        "stage_progress": progress,
        "regime_state": str(trace.get("regime_state") or "unknown"),
        "edge_state": str(trace.get("edge_state") or "unknown"),
        "edge_margin_bps": diagnostics.get("edge_margin_bps"),
        "quantitative_direct_coverage_norm": round(
            _number(quantitative.get("direct_coverage_norm")), 6
        ),
        "ingestion_route_status": str(
            ingestion_route.get("status") or "unavailable"
        ),
        "ingestion_route_state": str(
            ingestion_route.get("route_state") or "unavailable"
        ),
        "ingestion_route_profile_id": str(
            ingestion_route.get("profile_id") or ""
        ),
        "ingestion_route_quality_norm": round(
            _number(ingestion_route.get("average_route_score_norm")), 6
        ),
        "ingestion_paper_coverage_norm": round(
            _number(ingestion_route.get("paper_required_coverage_norm")), 6
        ),
        "ingestion_live_coverage_norm": round(
            _number(ingestion_route.get("live_required_coverage_norm")), 6
        ),
        "ingestion_selected_producer_count": max(
            int(_number(ingestion_route.get("selected_producer_count"))), 0
        ),
        "ingestion_route_receipt_valid": bool(
            ingestion_route.get("receipt_valid", False)
        ),
        "ingestion_route_receipt_sha256": str(
            ingestion_route.get("route_receipt_sha256") or ""
        ),
        "ingestion_route_summary_receipt_sha256": str(
            ingestion_route.get("decision_route_receipt_sha256") or ""
        ),
        "paper_quality_gate_state": str(
            _mapping(trace.get("mode_quality_gates")).get("paper") or "unknown"
        ),
        "live_quality_gate_state": str(
            _mapping(trace.get("mode_quality_gates")).get("live") or "unknown"
        ),
        "reason_codes": reasons,
        "evidence_actions": [
            str(value)
            for value in (evaluation.get("evidence_actions") or [])[:3]
            if str(value)
        ],
        "execution_eligible_sleeve": bool(receipt.get("execution_eligible", False)),
        "live_execution_authority": False,
    }
    return {**material, "summary_sha256": _canonical_hash(material)}


def apply_decision_flow_control(
    *,
    target_mode: str,
    current_action: str,
    quantity: float,
    evaluation: Mapping[str, Any],
    policy: Mapping[str, Any],
) -> tuple[str, float, dict[str, Any]]:
    """Apply a mode-specific monotonic cap to an existing authorized intent."""

    action = _normalized_action(current_action)
    input_quantity = max(_number(quantity), 0.0)
    mode = str(target_mode or "").strip().lower()
    receipt_hint = _mapping(evaluation.get("policy_receipt"))
    resolved_policy, current_receipt = _policy_for_evaluation(
        {
            "shadow_profile": str(
                evaluation.get("profile") or receipt_hint.get("profile") or "default"
            ),
            "shadow_domain": str(
                evaluation.get("domain") or receipt_hint.get("domain") or ""
            ),
            "lifecycle_state": str(
                evaluation.get("lifecycle_state")
                or receipt_hint.get("lifecycle_state")
                or ""
            ),
        },
        policy,
    )
    control_key = "active_paper_control" if mode == "paper" else "active_live_control"
    control = _mapping(resolved_policy.get(control_key)) if mode in {"paper", "live"} else {}
    reasons: list[str] = []
    disposition = "disabled_passthrough"
    blocking_stage = ""
    multiplier = 1.0

    enabled = bool(control.get("enabled", False))
    authorized_mode = mode in {"paper", "live"} and str(
        control.get("target_mode") or ""
    ).lower() == mode
    intent_action = str(evaluation.get("intent_action") or "HOLD").strip().upper()

    if not enabled:
        reasons.append(f"active_{mode or 'unknown'}_control_disabled")
    elif not authorized_mode:
        disposition = "unauthorized_mode_passthrough"
        reasons.append("decision_flow_has_no_nonpaper_authority")
    elif action not in DIRECTIONAL_ACTIONS or input_quantity <= 0.0:
        disposition = "existing_hold_passthrough"
        multiplier = 0.0 if action == "HOLD" else 1.0
        reasons.append("no_existing_directional_paper_order")
    elif intent_action != action:
        disposition = "veto_direction_mismatch"
        blocking_stage = "03_signal_formation"
        multiplier = 0.0
        reasons.append("active_action_does_not_match_original_intent")
    else:
        execution_eligible = bool(current_receipt.get("execution_eligible", False))
        if mode == "live" and bool(
            control.get("require_execution_eligible_sleeve", True)
        ) and not execution_eligible:
            disposition = "veto_execution_ineligible_sleeve"
            blocking_stage = "08_non_bypassable_risk"
            multiplier = 0.0
            reasons.append("resolved_sleeve_or_lifecycle_is_not_execution_eligible")
        if (
            multiplier > 0.0
            and mode == "live"
            and bool(control.get("require_complete_strategy_definition", True))
            and not bool(current_receipt.get("strategy_definition_complete", False))
        ):
            disposition = "veto_incomplete_strategy_definition"
            blocking_stage = "09_shadow_priority"
            multiplier = 0.0
            reasons.append("live_requires_complete_strategy_definition")
        quantitative_evidence = _mapping(evaluation.get("quantitative_evidence"))
        if (
            multiplier > 0.0
            and mode == "live"
            and bool(control.get("require_quantitative_evidence_ready", True))
            and not bool(quantitative_evidence.get("live_ready", False))
        ):
            disposition = "veto_quantitative_evidence_not_ready"
            blocking_stage = "09_shadow_priority"
            multiplier = 0.0
            reasons.extend(
                [
                    "live_requires_direct_passing_quantitative_evidence",
                    *[
                        f"quantitative_evidence:{reason}"
                        for reason in quantitative_evidence.get("live_blockers", [])
                    ],
                ]
            )
        action_semantics = _mapping(evaluation.get("action_semantics"))
        if (
            multiplier > 0.0
            and mode == "live"
            and bool(control.get("require_action_semantics_ready", True))
            and not bool(action_semantics.get("ready", False))
        ):
            disposition = "veto_action_semantics_not_ready"
            blocking_stage = "08_non_bypassable_risk"
            multiplier = 0.0
            reasons.extend(
                [
                    "live_requires_position_aware_action_semantics",
                    *[
                        f"action_semantics:{reason}"
                        for reason in action_semantics.get("reasons", [])
                    ],
                ]
            )
        required_stages = [
            str(value)
            for value in (control.get("required_pass_stages") or [])
            if str(value).strip()
        ]
        failed_required = "" if multiplier == 0.0 else next(
            (
                stage_id
                for stage_id in required_stages
                if not _stage_passed(evaluation, stage_id)
            ),
            "",
        )
        if multiplier == 0.0:
            pass
        elif failed_required:
            disposition = "veto_required_stage"
            blocking_stage = failed_required
            multiplier = 0.0
            reasons.append(f"required_stage_failed:{failed_required}")
        else:
            diagnostics = _mapping(evaluation.get("diagnostics"))
            edge_kind = str(diagnostics.get("edge_evidence_kind") or "missing")
            edge_proven = bool(diagnostics.get("edge_proven", False))
            if mode == "live" and (
                edge_kind != "lower_confidence_bound" or not edge_proven
            ):
                disposition = "veto_live_edge_not_proven"
                blocking_stage = "05_post_cost_edge"
                multiplier = 0.0
                reasons.append("live_requires_positive_post_cost_edge_lcb")
            elif edge_kind == "lower_confidence_bound" and not edge_proven:
                disposition = "veto_nonpositive_post_cost_edge"
                blocking_stage = "05_post_cost_edge"
                multiplier = 0.0
                reasons.append("explicit_post_cost_edge_lcb_did_not_clear_costs")
            elif mode == "paper" and edge_kind == "missing":
                disposition = "bounded_evidence_probe"
                blocking_stage = "05_post_cost_edge"
                multiplier = _clamp01(control.get("missing_edge_max_quantity_multiplier", 0.10))
                reasons.append("post_cost_edge_missing_probe_capped")
            elif mode == "paper" and edge_kind == "point_estimate":
                disposition = "bounded_point_estimate_probe"
                blocking_stage = "05_post_cost_edge"
                multiplier = _clamp01(control.get("point_estimate_max_quantity_multiplier", 0.15))
                reasons.append("post_cost_edge_lcb_missing_probe_capped")
            else:
                evidence_maturity = _number(
                    _mapping(evaluation.get("components")).get("evidence_maturity"),
                    0.0,
                )
                evidence_floor = _number(
                    _mapping(resolved_policy.get("qualification_floors")).get(
                        "evidence_maturity"
                    ),
                    0.50,
                )
                if mode == "live" and bool(
                    control.get("require_qualified_candidate", True)
                ) and not bool(evaluation.get("qualified_shadow_candidate", False)):
                    disposition = "veto_live_candidate_not_qualified"
                    blocking_stage = str(
                        evaluation.get("first_failed_stage") or "09_shadow_priority"
                    )
                    multiplier = 0.0
                    reasons.append("live_requires_fully_qualified_sleeve_candidate")
                elif evidence_maturity < evidence_floor:
                    disposition = "immature_evidence_downsize"
                    blocking_stage = "09_shadow_priority"
                    multiplier = (
                        0.0
                        if mode == "live"
                        else _clamp01(
                            control.get(
                                "immature_evidence_max_quantity_multiplier", 0.25
                            )
                        )
                    )
                    reasons.append("post_cost_sample_evidence_below_maturity_floor")
                elif not bool(evaluation.get("qualified_shadow_candidate", False)):
                    disposition = "near_miss_downsize"
                    blocking_stage = "09_shadow_priority"
                    multiplier = (
                        0.0
                        if mode == "live"
                        else _clamp01(
                            control.get("near_miss_max_quantity_multiplier", 0.50)
                        )
                    )
                    reasons.append("decision_utility_below_full_qualification")
                else:
                    disposition = "qualified_passthrough"
                    multiplier = _clamp01(
                        control.get("qualified_max_quantity_multiplier", 1.0)
                    )
                    reasons.append(f"all_active_{mode}_quality_requirements_passed")

        if (
            mode == "paper"
            and multiplier > 0.0
            and bool(quantitative_evidence.get("explicit_adverse", False))
        ):
            adverse_cap = _clamp01(
                control.get(
                    "adverse_quantitative_evidence_max_quantity_multiplier",
                    0.25,
                )
            )
            if adverse_cap < multiplier:
                multiplier = adverse_cap
                disposition = "adverse_quantitative_evidence_downsize"
                blocking_stage = "09_shadow_priority"
                reasons.extend(
                    [
                        "direct_quantitative_evidence_below_family_floor",
                        *[
                            f"quantitative_evidence_below_floor:{axis}"
                            for axis in quantitative_evidence.get(
                                "failed_required_axes", []
                            )
                        ],
                    ]
                )

    output_quantity = min(input_quantity, max(input_quantity * multiplier, 0.0))
    output_quantity = round(output_quantity, 6)
    output_action = action
    if authorized_mode and action in DIRECTIONAL_ACTIONS and output_quantity <= 0.0:
        output_action = "HOLD"

    metadata = {
        "policy_id": str(resolved_policy.get("policy_id") or ""),
        "resolved_policy_id": str(current_receipt.get("resolved_policy_id") or ""),
        "policy_family_id": str(current_receipt.get("policy_family_id") or ""),
        "policy_receipt": current_receipt,
        "evaluation_id": str(evaluation.get("evaluation_id") or ""),
        "evaluation_sha256": _canonical_hash(dict(evaluation)),
        "target_mode": mode,
        "enabled": enabled,
        "authorized_mode": authorized_mode,
        "disposition": disposition,
        "blocking_stage": blocking_stage,
        "classification": str(evaluation.get("classification") or ""),
        "decision_quality_utility_norm": round(
            _number(evaluation.get("decision_quality_utility_norm")), 6
        ),
        "input_action": action,
        "output_action": output_action,
        "input_quantity": round(input_quantity, 6),
        "output_quantity": output_quantity,
        "quantity_multiplier": round(
            output_quantity / input_quantity if input_quantity > 0.0 else 0.0,
            6,
        ),
        "action_vetoed": bool(action in DIRECTIONAL_ACTIONS and output_action == "HOLD"),
        "quantity_reduced": bool(output_quantity + 1e-12 < input_quantity),
        "reasons": reasons,
        "authority_contract": {
            "can_create_intent": False,
            "can_reverse_intent": False,
            "can_increase_quantity": False,
            "can_bypass_existing_hold": False,
            "paper_veto_or_downsize_only": mode == "paper",
            "live_veto_or_downsize_only": mode == "live",
            "live_execution_authority": False,
            "promotion_authority": False,
        },
    }
    metadata["operator_summary"] = build_decision_operator_summary(
        evaluation,
        metadata,
    )
    return output_action, output_quantity, metadata


def apply_paper_decision_flow_control(
    *,
    target_mode: str,
    current_action: str,
    quantity: float,
    evaluation: Mapping[str, Any],
    policy: Mapping[str, Any],
) -> tuple[str, float, dict[str, Any]]:
    """Compatibility wrapper retaining paper-only mutation authority."""

    if str(target_mode or "").strip().lower() == "paper":
        return apply_decision_flow_control(
            target_mode="paper",
            current_action=current_action,
            quantity=quantity,
            evaluation=evaluation,
            policy=policy,
        )
    action = _normalized_action(current_action)
    input_quantity = max(_number(quantity), 0.0)
    receipt = _mapping(evaluation.get("policy_receipt"))
    metadata = {
        "policy_id": str(policy.get("policy_id") or ""),
        "resolved_policy_id": str(receipt.get("resolved_policy_id") or ""),
        "policy_family_id": str(receipt.get("policy_family_id") or ""),
        "policy_receipt": receipt,
        "evaluation_id": str(evaluation.get("evaluation_id") or ""),
        "evaluation_sha256": _canonical_hash(dict(evaluation)),
        "target_mode": str(target_mode or "").strip().lower(),
        "enabled": bool(_mapping(policy.get("active_paper_control")).get("enabled", False)),
        "authorized_mode": False,
        "disposition": "unauthorized_mode_passthrough",
        "blocking_stage": "",
        "classification": str(evaluation.get("classification") or ""),
        "decision_quality_utility_norm": round(
            _number(evaluation.get("decision_quality_utility_norm")), 6
        ),
        "input_action": action,
        "output_action": action,
        "input_quantity": round(input_quantity, 6),
        "output_quantity": round(input_quantity, 6),
        "quantity_multiplier": 1.0 if input_quantity > 0.0 else 0.0,
        "action_vetoed": False,
        "quantity_reduced": False,
        "reasons": ["decision_flow_has_no_nonpaper_authority"],
        "authority_contract": {
            "can_create_intent": False,
            "can_reverse_intent": False,
            "can_increase_quantity": False,
            "can_bypass_existing_hold": False,
            "paper_veto_or_downsize_only": True,
            "live_veto_or_downsize_only": False,
            "live_execution_authority": False,
            "promotion_authority": False,
        },
    }
    metadata["operator_summary"] = build_decision_operator_summary(
        evaluation,
        metadata,
    )
    return action, input_quantity, metadata


def _component_inputs(row: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        "market": _mapping(row.get("market")),
        "micro": _mapping(row.get("market_micro_features")),
        "quality": _mapping(row.get("data_quality_features")),
        "freshness": _mapping(row.get("feature_freshness")),
        "allocation": _mapping(row.get("allocation_confidence")),
        "meta": _mapping(row.get("grand_master_meta")),
        "execution_guard": _mapping(row.get("execution_guard")),
        "execution_sim": _mapping(row.get("execution_sim")),
        "portfolio": _mapping(row.get("portfolio")),
        "portfolio_risk": _mapping(row.get("portfolio_risk_engine")),
        "turnover": _mapping(row.get("long_term_turnover_policy")),
        "circuits": _mapping(row.get("circuit_breakers")),
        "quant": _mapping(row.get("quant_model_features")),
        "dividend": _mapping(row.get("dividend_features")),
        "broker_truth": _mapping(row.get("broker_truth_reconcile")),
    }


def _score_components(
    row: Mapping[str, Any],
    policy: Mapping[str, Any],
) -> tuple[dict[str, float], dict[str, Any]]:
    inputs = _component_inputs(row)
    market = inputs["market"]
    micro = inputs["micro"]
    quality = inputs["quality"]
    freshness = inputs["freshness"]
    allocation = inputs["allocation"]
    meta = inputs["meta"]
    execution_guard = inputs["execution_guard"]
    execution_sim = inputs["execution_sim"]
    portfolio = inputs["portfolio"]
    portfolio_risk = inputs["portfolio_risk"]
    turnover = inputs["turnover"]
    circuits = inputs["circuits"]
    quant = inputs["quant"]
    dividend = inputs["dividend"]
    broker_truth = inputs["broker_truth"]

    market_quality = _mapping(policy.get("market_quality"))
    spread_ceiling = max(_number(market_quality.get("spread_ceiling_bps"), 35.0), 1.0)
    latency_ceiling = max(_number(market_quality.get("latency_ceiling_ms"), 1500.0), 1.0)
    source_quality = _clamp01(row.get("source_quality_score", 0.5))
    quote_agreement = _clamp01(quality.get("data_quality_quote_agreement_norm", 0.5))
    missing_ratio = _clamp01(quality.get("data_quality_missing_feature_ratio_norm", 0.5))
    latency_ms = max(
        _number(
            market.get("market_data_latency_ms"),
            execution_sim.get("latency_ms", latency_ceiling),
        ),
        0.0,
    )
    freshness_ok = bool(freshness.get("ok", False))
    data_integrity = _clamp01(
        _mean(
            [
                source_quality,
                quote_agreement,
                1.0 - missing_ratio,
                1.0 if freshness_ok else 0.0,
                1.0 - min(latency_ms / latency_ceiling, 1.0),
            ]
        )
    )

    intent_score = _clamp01(
        row.get("master_intent_score", row.get("master_score", 0.5))
    )
    signal_conviction = _clamp01(abs(intent_score - 0.5) * 2.0)

    intent_action = str(
        row.get("master_intent_action") or row.get("master_action") or "HOLD"
    ).upper()
    intent_sign = 1.0 if intent_action == "BUY" else (-1.0 if intent_action == "SELL" else 0.0)

    def directional_support(value: Any) -> float:
        signed = min(max(_number(value), -1.0), 1.0)
        return _clamp01(0.5 + 0.5 * intent_sign * signed) if intent_sign else 0.5

    specialist_consensus = directional_support(meta.get("specialist_consensus", 0.0))
    sleeve_consensus = directional_support(
        meta.get("sleeve_master_consensus", meta.get("sleeve_consensus", 0.0))
    )
    directional_alignment = directional_support(meta.get("directional_alignment", 0.0))
    disagreement = _clamp01(meta.get("master_disagreement", 0.5))
    ensemble_consensus = _clamp01(
        0.30 * specialist_consensus
        + 0.30 * sleeve_consensus
        + 0.20 * directional_alignment
        + 0.20 * (1.0 - disagreement)
    )

    regime_values: list[float] = []
    for source, key in (
        (meta, "quant_strategy_fit"),
        (micro, "market_micro_trend_persistence_norm"),
        (micro, "market_micro_post_event_drift_norm"),
        (quant, "quant_strategy_portfolio_fit_norm"),
    ):
        if source.get(key) not in {None, ""}:
            regime_values.append(_clamp01(source.get(key)))
    if micro.get("market_micro_reversal_risk_norm") not in {None, ""}:
        regime_values.append(1.0 - _clamp01(micro.get("market_micro_reversal_risk_norm")))
    regime_alignment = _clamp01(_mean(regime_values, default=0.5))

    search_sources = [row, market, micro, allocation, meta, quant, execution_sim]
    edge_lcb_bps, edge_lcb_key = _first_number(
        search_sources,
        (
            "predicted_edge_lower_confidence_bound_bps",
            "expected_edge_lower_confidence_bound_bps",
            "edge_lower_confidence_bound_bps",
            "post_cost_lower_confidence_bound_bps",
        ),
    )
    edge_point_bps, edge_point_key = _first_number(
        search_sources,
        ("predicted_edge_bps", "expected_alpha_bps", "expected_edge_bps"),
    )
    edge_bps = edge_lcb_bps if edge_lcb_bps is not None else edge_point_bps
    edge_source = edge_lcb_key or edge_point_key
    edge_evidence_kind = "lower_confidence_bound" if edge_lcb_key else (
        "point_estimate" if edge_point_key else "missing"
    )
    explicit_cost_bps, cost_key = _first_number(
        search_sources,
        ("round_trip_cost_bps", "expected_round_trip_cost_bps"),
    )
    spread_bps = max(_number(market.get("spread_bps"), 0.0), 0.0)
    slippage_bps = max(_number(execution_sim.get("slippage_bps"), 0.0), 0.0)
    impact_bps = max(_number(execution_sim.get("impact_bps"), 0.0), 0.0)
    fee_bps = max(_number(execution_sim.get("fee_bps"), 0.0), 0.0)
    round_trip_cost_bps = (
        max(_number(explicit_cost_bps), 0.0)
        if explicit_cost_bps is not None
        else 2.0 * spread_bps + 2.0 * slippage_bps + impact_bps + fee_bps
    )
    floors = _mapping(policy.get("qualification_floors"))
    edge_cost_multiple = min(
        max(_number(floors.get("minimum_edge_cost_multiple"), 1.5), 1.0),
        5.0,
    )
    required_edge_bps = round_trip_cost_bps * edge_cost_multiple
    edge_margin_bps = (
        _number(edge_bps) - required_edge_bps if edge_bps is not None else None
    )
    edge_scale = max(required_edge_bps, 5.0)
    net_edge_quality = (
        _clamp01(0.5 + _number(edge_margin_bps) / (2.0 * edge_scale))
        if edge_margin_bps is not None
        else 0.0
    )
    edge_proven = bool(
        edge_lcb_bps is not None and _number(edge_lcb_bps) > required_edge_bps
    )

    tradeability = _clamp01(micro.get("market_micro_tradeability_score_norm", 0.5))
    spread_quality = 1.0 - min(spread_bps / spread_ceiling, 1.0)
    latency_quality = 1.0 - min(latency_ms / latency_ceiling, 1.0)
    route_cost_quality = 1.0 - min(
        (slippage_bps + impact_bps + fee_bps) / spread_ceiling,
        1.0,
    )
    execution_guard_ok = bool(execution_guard.get("ok", False))
    execution_quality = _clamp01(
        _mean(
            [
                tradeability,
                spread_quality,
                latency_quality,
                route_cost_quality,
                1.0 if execution_guard_ok else 0.0,
            ]
        )
    )

    conflict = _clamp01(
        allocation.get(
            "allocation_conflict_norm",
            allocation.get("cross_bot_conflict_norm", meta.get("directional_conflict", 0.5)),
        )
    )
    overlap = _clamp01(
        allocation.get(
            "portfolio_overlap_pressure_norm",
            allocation.get("core_portfolio_overlap_pressure_norm", 0.5),
        )
    )
    allocation_confidence = _clamp01(allocation.get("allocation_confidence_norm", 0.0))
    portfolio_risk_clear = not bool(portfolio_risk.get("blocked", False))
    lane_budget = _clamp01(portfolio.get("lane_budget_mult", 0.5))
    portfolio_fit = _clamp01(
        _mean(
            [
                allocation_confidence,
                1.0 - conflict,
                1.0 - overlap,
                1.0 if portfolio_risk_clear else 0.0,
                lane_budget,
            ]
        )
    )

    circuit_keys = (
        "kill_switch_active",
        "vol_shock_pause_active",
        "liquidity_pause_active",
        "symbol_circuit_active",
        "lane_kill_switch_active",
    )
    circuit_clear = not any(bool(circuits.get(key, False)) for key in circuit_keys)
    broker_truth_ok = bool(broker_truth.get("ok", True))
    guard_categories = {
        str(value).strip().lower()
        for value in (row.get("decision_guard_categories") or [])
        if str(value).strip()
    }
    risk_category_clear = not bool(
        guard_categories.intersection({"risk", "portfolio", "broker_truth", "account"})
    )
    turnover_clear = not bool(turnover.get("blocked", False))
    risk_headroom = _clamp01(
        _mean(
            [
                1.0 if circuit_clear else 0.0,
                1.0 if broker_truth_ok else 0.0,
                1.0 if portfolio_risk_clear else 0.0,
                1.0 if risk_category_clear else 0.0,
                1.0 if turnover_clear else 0.0,
            ]
        )
    )
    if not risk_category_clear:
        risk_headroom = min(risk_headroom, 0.25)

    post_cost_samples, sample_key = _first_number(
        search_sources,
        ("post_cost_samples", "candidate_post_cost_samples"),
    )
    post_cost_lcb, post_cost_lcb_key = _first_number(
        search_sources,
        ("post_cost_lower_confidence_bound", "candidate_post_cost_lower_confidence_bound"),
    )
    data_confidence = _clamp01(meta.get("quant_data_confidence", 0.0))
    sample_score = min(max(_number(post_cost_samples), 0.0) / 100.0, 1.0)
    lcb_score = 1.0 if post_cost_lcb is not None and post_cost_lcb > 0.0 else 0.0
    evidence_maturity = _clamp01(
        0.60 * sample_score + 0.25 * lcb_score + 0.15 * data_confidence
    )

    profile = str(row.get("shadow_profile") or row.get("profile") or "default").lower()
    receipt = _mapping(policy.get("resolved_sleeve_policy"))
    policy_family_id = str(receipt.get("policy_family_id") or "")
    if policy_family_id == "long_horizon_income" or any(
        token in profile for token in ("long_term", "dividend", "compound", "bond")
    ):
        long_term_values = [1.0 - overlap, 1.0 if turnover_clear else 0.0]
        for key in (
            "dividend_quality_score_norm",
            "dividend_compounding_quality_norm",
            "dividend_growth_persistence_norm",
        ):
            if dividend.get(key) not in {None, ""}:
                long_term_values.append(_clamp01(dividend.get(key)))
        long_term_alignment = _clamp01(_mean(long_term_values, default=0.5))
    else:
        long_term_alignment = _clamp01(
            _mean([1.0 - overlap, 1.0 - conflict, risk_headroom], default=0.5)
        )

    components = {
        "data_integrity": data_integrity,
        "signal_conviction": signal_conviction,
        "ensemble_consensus": ensemble_consensus,
        "regime_alignment": regime_alignment,
        "net_edge_quality": net_edge_quality,
        "execution_quality": execution_quality,
        "portfolio_fit": portfolio_fit,
        "risk_headroom": risk_headroom,
        "evidence_maturity": evidence_maturity,
        "long_term_alignment": long_term_alignment,
    }
    diagnostics = {
        "intent_score": round(intent_score, 8),
        "freshness_ok": freshness_ok,
        "source_quality_norm": round(source_quality, 6),
        "quote_agreement_norm": round(quote_agreement, 6),
        "missing_feature_ratio_norm": round(missing_ratio, 6),
        "spread_bps": round(spread_bps, 6),
        "slippage_bps": round(slippage_bps, 6),
        "impact_bps": round(impact_bps, 6),
        "fee_bps": round(fee_bps, 6),
        "latency_ms": round(latency_ms, 3),
        "execution_guard_ok": execution_guard_ok,
        "portfolio_risk_clear": portfolio_risk_clear,
        "circuit_clear": circuit_clear,
        "broker_truth_ok": broker_truth_ok,
        "risk_category_clear": risk_category_clear,
        "turnover_clear": turnover_clear,
        "conflict_pressure_norm": round(conflict, 6),
        "overlap_pressure_norm": round(overlap, 6),
        "edge_evidence_kind": edge_evidence_kind,
        "edge_source": edge_source,
        "edge_bps": round(_number(edge_bps), 6) if edge_bps is not None else None,
        "round_trip_cost_source": cost_key or "modeled_from_spread_slippage_impact_fees",
        "round_trip_cost_bps": round(round_trip_cost_bps, 6),
        "required_edge_bps": round(required_edge_bps, 6),
        "edge_margin_bps": round(_number(edge_margin_bps), 6) if edge_margin_bps is not None else None,
        "edge_proven": edge_proven,
        "post_cost_samples": max(int(_number(post_cost_samples)), 0) if sample_key else 0,
        "post_cost_lower_confidence_bound": (
            round(_number(post_cost_lcb), 8) if post_cost_lcb_key else None
        ),
        "market_impact_curve_available": bool(
            row.get("market_impact_curve") or market.get("market_impact_curve")
        ),
        "policy_family_id": policy_family_id,
    }
    return components, diagnostics


def _quantitative_evidence_assessment(
    row: Mapping[str, Any],
    components: Mapping[str, float],
    diagnostics: Mapping[str, Any],
    policy: Mapping[str, Any],
) -> dict[str, Any]:
    packet = _mapping(row.get("quantitative_evidence"))
    inputs = _component_inputs(row)
    direct_sources = [
        packet,
        row,
        inputs["quant"],
        inputs["meta"],
        inputs["market"],
        inputs["micro"],
        inputs["allocation"],
        inputs["execution_sim"],
        inputs["portfolio"],
        inputs["portfolio_risk"],
    ]
    direct_aliases: dict[str, tuple[str, ...]] = {
        "selection_bias_control": (
            "selection_bias_control",
            "selection_bias_control_quality_norm",
            "deflated_sharpe_probability_norm",
            "deflated_sharpe_probability",
            "multiple_testing_control_quality_norm",
            "backtest_overfitting_resilience_norm",
            "probability_of_backtest_overfitting",
        ),
        "independent_samples": (
            "independent_samples",
            "independent_sample_quality_norm",
            "effective_sample_coverage_norm",
            "clustered_effective_sample_quality_norm",
            "effective_sample_size",
            "effective_post_cost_samples",
        ),
        "uncertainty_calibration": (
            "uncertainty_calibration",
            "uncertainty_calibration_quality_norm",
            "conformal_coverage_quality_norm",
            "probability_calibration_quality_norm",
        ),
        "signal_decay_fit": (
            "signal_decay_fit",
            "signal_decay_fit_norm",
            "alpha_decay_fit_norm",
            "half_life_stability_norm",
        ),
        "payoff_asymmetry": (
            "payoff_asymmetry",
            "payoff_asymmetry_quality_norm",
            "mfe_mae_asymmetry_norm",
            "upside_downside_capture_quality_norm",
        ),
        "capacity_headroom": (
            "capacity_headroom",
            "capacity_headroom_norm",
            "market_impact_capacity_quality_norm",
            "participation_capacity_quality_norm",
        ),
        "crowding_residual": (
            "crowding_residual",
            "crowding_residual_quality_norm",
            "residual_alpha_quality_norm",
            "factor_neutrality_quality_norm",
        ),
        "tail_survival": (
            "tail_survival",
            "tail_survival_quality_norm",
            "risk_of_ruin_resilience_norm",
            "expected_shortfall_survival_norm",
            "risk_of_ruin",
        ),
        "regime_stability": (
            "regime_stability",
            "regime_stability_norm",
            "changepoint_stability_norm",
            "cross_regime_stability_norm",
        ),
    }
    proxy_aliases: dict[str, tuple[str, ...]] = {
        "selection_bias_control": (
            "quant_replication_crisis_shield_norm",
            "quant_strategy_selection_confidence_norm",
        ),
        "uncertainty_calibration": (
            "quant_data_confidence",
            "quant_model_data_confidence_norm",
            "quant_kalman_filter_confidence_norm",
        ),
        "signal_decay_fit": (
            "market_micro_trend_persistence_norm",
            "market_micro_post_event_drift_norm",
        ),
        "payoff_asymmetry": (
            "profitability_evidence_quality_norm",
            "quant_strategy_risk_adjusted_conviction_norm",
        ),
        "regime_stability": (
            "quant_strategy_fit",
            "quant_strategy_portfolio_fit_norm",
            "quant_regime_switch_filter_confidence_norm",
        ),
    }
    evidence_contract = _mapping(policy.get("quantitative_evidence_contract"))
    axis_contracts = _mapping(evidence_contract.get("axes"))
    required_axes = [
        str(value)
        for value in (policy.get("required_quantitative_evidence") or [])
        if str(value).strip()
    ]
    rows: dict[str, dict[str, Any]] = {}
    for axis in sorted(QUANTITATIVE_EVIDENCE_AXES):
        contract = _mapping(axis_contracts.get(axis))
        floor = _clamp01(contract.get("floor", 0.0))
        score, source = _first_number(direct_sources, direct_aliases[axis])
        evidence_kind = "direct" if source else "missing"
        if source == "probability_of_backtest_overfitting":
            score = 1.0 - _clamp01(score)
        elif source == "risk_of_ruin":
            score = 1.0 - _clamp01(score)
        elif source in {"effective_sample_size", "effective_post_cost_samples"}:
            target = max(_number(packet.get("effective_sample_target"), 100.0), 1.0)
            score = min(max(_number(score), 0.0) / target, 1.0)
        if score is None:
            proxy_sources = [
                packet,
                row,
                inputs["quant"],
                inputs["meta"],
                inputs["micro"],
                inputs["allocation"],
            ]
            score, source = _first_number(
                proxy_sources,
                proxy_aliases.get(axis, ()),
            )
            evidence_kind = "proxy" if source else "missing"
        if score is None and axis == "independent_samples":
            score = _clamp01(components.get("evidence_maturity", 0.0))
            source = "component:evidence_maturity"
            evidence_kind = "proxy"
        elif score is None and axis == "capacity_headroom":
            score = _clamp01(
                _mean(
                    [
                        components.get("execution_quality", 0.0),
                        _number(inputs["portfolio"].get("lane_budget_mult"), 0.0),
                        1.0
                        if diagnostics.get("market_impact_curve_available", False)
                        else 0.0,
                    ]
                )
            )
            source = "proxy:execution_lane_budget_and_impact_presence"
            evidence_kind = "proxy"
        elif score is None and axis == "crowding_residual":
            score = _clamp01(
                1.0
                - _mean(
                    [
                        _clamp01(diagnostics.get("conflict_pressure_norm", 0.5)),
                        _clamp01(diagnostics.get("overlap_pressure_norm", 0.5)),
                    ]
                )
            )
            source = "proxy:portfolio_conflict_and_overlap"
            evidence_kind = "proxy"
        elif score is None and axis == "tail_survival":
            score = _clamp01(components.get("risk_headroom", 0.0))
            source = "proxy:risk_headroom"
            evidence_kind = "proxy"
        elif score is None and axis == "regime_stability":
            score = _clamp01(components.get("regime_alignment", 0.0))
            source = "proxy:regime_alignment"
            evidence_kind = "proxy"

        available = score is not None
        normalized_score = _clamp01(score) if available else None
        passed = bool(available and normalized_score is not None and normalized_score >= floor)
        rows[axis] = {
            "required": axis in required_axes,
            "available": available,
            "direct_measurement": evidence_kind == "direct",
            "evidence_kind": evidence_kind,
            "source": source,
            "score_norm": round(normalized_score, 6)
            if normalized_score is not None
            else None,
            "floor_norm": round(floor, 6),
            "passed": passed,
            "critical": bool(contract.get("critical", False)),
        }

    proxy_counts_for_live = bool(
        evidence_contract.get("live_proxy_evidence_counts_as_available", False)
    )
    missing_required: list[str] = []
    proxy_only_required: list[str] = []
    failed_required: list[str] = []
    for axis in required_axes:
        axis_row = rows.get(axis, {})
        if not bool(axis_row.get("available", False)):
            missing_required.append(axis)
        elif not bool(axis_row.get("direct_measurement", False)):
            proxy_only_required.append(axis)
        elif not bool(axis_row.get("passed", False)):
            failed_required.append(axis)
    direct_required_count = sum(
        1 for axis in required_axes if rows.get(axis, {}).get("direct_measurement")
    )
    direct_failed = [
        axis
        for axis in required_axes
        if rows.get(axis, {}).get("direct_measurement")
        and not rows.get(axis, {}).get("passed")
    ]
    critical_adverse = [
        axis for axis in direct_failed if rows.get(axis, {}).get("critical")
    ]
    live_missing = list(missing_required)
    if not proxy_counts_for_live:
        live_missing.extend(proxy_only_required)
    live_ready = bool(not live_missing and not failed_required)
    available_scores = [
        _number(rows[axis].get("score_norm"))
        for axis in required_axes
        if rows.get(axis, {}).get("score_norm") is not None
    ]
    return {
        "contract_version": "quantitative_evidence_v1",
        "required_axes": required_axes,
        "axes": rows,
        "required_axis_count": len(required_axes),
        "direct_required_axis_count": direct_required_count,
        "direct_coverage_norm": round(
            direct_required_count / max(len(required_axes), 1), 6
        ),
        "available_score_mean_norm": round(_mean(available_scores), 6)
        if available_scores
        else None,
        "missing_required_axes": sorted(missing_required),
        "proxy_only_required_axes": sorted(proxy_only_required),
        "failed_required_axes": sorted(failed_required),
        "explicit_adverse": bool(direct_failed),
        "critical_adverse_axes": sorted(critical_adverse),
        "paper_collection_allowed": True,
        "live_ready": live_ready,
        "live_blockers": [
            *[f"missing_or_proxy_only:{axis}" for axis in sorted(set(live_missing))],
            *[f"below_floor:{axis}" for axis in sorted(failed_required)],
        ],
    }


def _action_semantics(
    row: Mapping[str, Any],
    policy: Mapping[str, Any],
) -> dict[str, Any]:
    definition = _mapping(policy.get("strategy_definition"))
    allowed = {
        str(value)
        for value in (definition.get("allowed_position_transitions") or [])
        if str(value).strip()
    }
    position_context = _mapping(row.get("position_context"))
    broker_truth = _mapping(row.get("broker_truth_reconcile"))
    symbol = str(row.get("symbol") or "").strip().upper()
    positions = _mapping(broker_truth.get("positions"))
    truth_available = bool(position_context.get("truth_available", False))
    current_quantity: float | None = None
    if position_context.get("current_quantity") not in {None, ""}:
        current_quantity = _number(position_context.get("current_quantity"), 0.0)
        truth_available = bool(position_context.get("truth_available", True))
    elif "positions" in broker_truth and bool(broker_truth.get("ok", False)):
        current_quantity = _number(positions.get(symbol), 0.0)
        truth_available = True

    action = _normalized_action(
        row.get("action") or row.get("master_action") or "HOLD"
    )
    if action == "HOLD":
        semantic = "abstain"
    elif not truth_available or current_quantity is None:
        semantic = "position_transition_unknown"
    elif action == "BUY" and current_quantity < 0.0:
        semantic = "cover_short"
    elif action == "BUY" and current_quantity > 0.0:
        semantic = "add_long"
    elif action == "BUY":
        semantic = "enter_long"
    elif current_quantity > 0.0:
        semantic = "reduce_or_exit_long"
    elif current_quantity < 0.0:
        semantic = "add_short"
    else:
        semantic = "enter_short"

    shorting_policy = str(definition.get("shorting_policy") or "").strip().lower()
    reasons: list[str] = []
    if action in DIRECTIONAL_ACTIONS and not truth_available:
        reasons.append("account_position_truth_required")
    if semantic not in allowed:
        reasons.append(f"position_transition_not_allowed:{semantic}")
    short_entry = semantic in {"enter_short", "add_short"}
    if short_entry and (
        "long_only" in shorting_policy or "disabled" in shorting_policy
    ):
        reasons.append("short_entry_forbidden_by_strategy_definition")
    if short_entry and "paired" in shorting_policy and not bool(
        position_context.get("linked_leg_truth_ready", False)
    ):
        reasons.append("paired_short_requires_linked_leg_truth")
    if short_entry and "defined_risk" in shorting_policy and not bool(
        position_context.get("defined_risk_structure_ready", False)
    ):
        reasons.append("short_option_exposure_requires_defined_risk_structure")
    if short_entry and not any(
        token in shorting_policy for token in ("long_only", "disabled", "defined_risk")
    ) and not bool(position_context.get("short_permission_confirmed", False)):
        reasons.append("account_short_permission_not_confirmed")
    return {
        "contract_version": "position_aware_action_semantics_v1",
        "action": action,
        "semantic": semantic,
        "current_quantity": round(current_quantity, 6)
        if current_quantity is not None
        else None,
        "position_truth_available": truth_available,
        "shorting_policy": shorting_policy,
        "allowed_position_transitions": sorted(allowed),
        "ready": not reasons,
        "reasons": reasons,
    }


def _stage_rows(
    *,
    row: Mapping[str, Any],
    components: Mapping[str, float],
    diagnostics: Mapping[str, Any],
    policy: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], str]:
    floors = _mapping(policy.get("qualification_floors"))
    intent_action = str(row.get("master_intent_action") or row.get("master_action") or "HOLD").upper()
    timestamp_ok = bool(_utc_text(row.get("timestamp_utc")))
    observation_ok = bool(timestamp_ok and str(row.get("symbol") or "").strip())
    paper_required = {
        str(value)
        for value in (
            _mapping(policy.get("active_paper_control")).get("required_pass_stages")
            or []
        )
    }
    live_required = {
        str(value)
        for value in (
            _mapping(policy.get("active_live_control")).get("required_pass_stages")
            or []
        )
    }
    checks = [
        {
            "stage_id": "01_observation",
            "passed": observation_ok,
            "reason_code": "observation_identity_missing",
            "reason": "missing timestamp or symbol identity",
            "evidence": {
                "timestamp_bound": timestamp_ok,
                "symbol_bound": bool(str(row.get("symbol") or "").strip()),
            },
        },
        {
            "stage_id": "02_data_qualification",
            "passed": bool(
                components["data_integrity"] >= _number(floors.get("data_integrity"), 0.65)
                and diagnostics.get("freshness_ok", False)
            ),
            "reason_code": "data_quality_or_freshness_below_floor",
            "reason": "fresh or complete market evidence is below policy floor",
            "evidence": {
                "score_norm": round(components["data_integrity"], 6),
                "floor_norm": round(_number(floors.get("data_integrity"), 0.65), 6),
                "freshness_ok": bool(diagnostics.get("freshness_ok", False)),
            },
        },
        {
            "stage_id": "03_signal_formation",
            "passed": bool(
                intent_action in DIRECTIONAL_ACTIONS
                and components["signal_conviction"] >= _number(floors.get("signal_conviction"), 0.12)
            ),
            "reason_code": (
                "intentional_no_edge_hold"
                if intent_action not in DIRECTIONAL_ACTIONS
                else "signal_conviction_below_floor"
            ),
            "reason": "no directional edge cleared the signal threshold",
            "evidence": {
                "intent_action": intent_action,
                "score_norm": round(components["signal_conviction"], 6),
                "floor_norm": round(_number(floors.get("signal_conviction"), 0.12), 6),
            },
        },
        {
            "stage_id": "04_consensus_and_regime",
            "passed": bool(
                components["ensemble_consensus"]
                >= _number(floors.get("ensemble_consensus"), 0.55)
                and components["regime_alignment"]
                >= _number(floors.get("regime_alignment"), 0.45)
            ),
            "reason_code": "consensus_or_regime_below_floor",
            "reason": "independent consensus or regime alignment is insufficient",
            "evidence": {
                "consensus_norm": round(components["ensemble_consensus"], 6),
                "consensus_floor_norm": round(
                    _number(floors.get("ensemble_consensus"), 0.55), 6
                ),
                "regime_alignment_norm": round(components["regime_alignment"], 6),
                "regime_floor_norm": round(
                    _number(floors.get("regime_alignment"), 0.45), 6
                ),
            },
        },
        {
            "stage_id": "05_post_cost_edge",
            "passed": bool(
                diagnostics.get("edge_proven", False)
                and components["net_edge_quality"]
                >= _number(floors.get("net_edge_quality"), 0.0)
            ),
            "reason_code": "post_cost_edge_lcb_not_proven",
            "reason": "post-cost lower-confidence-bound edge is missing or nonpositive",
            "evidence": {
                "evidence_kind": str(diagnostics.get("edge_evidence_kind") or "missing"),
                "edge_bps": diagnostics.get("edge_bps"),
                "required_edge_bps": diagnostics.get("required_edge_bps"),
                "edge_margin_bps": diagnostics.get("edge_margin_bps"),
            },
        },
        {
            "stage_id": "06_execution_feasibility",
            "passed": bool(
                components["execution_quality"]
                >= _number(floors.get("execution_quality"), 0.60)
                and diagnostics.get("execution_guard_ok", False)
            ),
            "reason_code": "execution_quality_or_route_guard_failed",
            "reason": "spread, latency, impact, depth, or route evidence is insufficient",
            "evidence": {
                "score_norm": round(components["execution_quality"], 6),
                "floor_norm": round(_number(floors.get("execution_quality"), 0.60), 6),
                "execution_guard_ok": bool(diagnostics.get("execution_guard_ok", False)),
                "spread_bps": diagnostics.get("spread_bps"),
                "latency_ms": diagnostics.get("latency_ms"),
            },
        },
        {
            "stage_id": "07_portfolio_fit",
            "passed": bool(
                components["portfolio_fit"] >= _number(floors.get("portfolio_fit"), 0.50)
                and diagnostics.get("portfolio_risk_clear", False)
                and components["long_term_alignment"]
                >= _number(floors.get("long_term_alignment"), 0.0)
            ),
            "reason_code": "portfolio_or_mandate_fit_below_floor",
            "reason": "portfolio budget, overlap, concentration, or turnover fit is insufficient",
            "evidence": {
                "portfolio_fit_norm": round(components["portfolio_fit"], 6),
                "portfolio_floor_norm": round(_number(floors.get("portfolio_fit"), 0.50), 6),
                "long_term_alignment_norm": round(components["long_term_alignment"], 6),
                "long_term_floor_norm": round(
                    _number(floors.get("long_term_alignment"), 0.0), 6
                ),
                "portfolio_risk_clear": bool(
                    diagnostics.get("portfolio_risk_clear", False)
                ),
            },
        },
        {
            "stage_id": "08_non_bypassable_risk",
            "passed": bool(
                components["risk_headroom"]
                >= _number(floors.get("risk_headroom"), 0.80)
            ),
            "reason_code": "non_bypassable_risk_control_active",
            "reason": "a non-bypassable broker, risk, circuit, or account control is active",
            "evidence": {
                "score_norm": round(components["risk_headroom"], 6),
                "floor_norm": round(_number(floors.get("risk_headroom"), 0.80), 6),
                "broker_truth_ok": bool(diagnostics.get("broker_truth_ok", False)),
                "circuit_clear": bool(diagnostics.get("circuit_clear", False)),
                "risk_category_clear": bool(
                    diagnostics.get("risk_category_clear", False)
                ),
            },
        },
    ]
    stages: list[dict[str, Any]] = []
    reached = True
    first_failed = ""
    for check in checks:
        stage_id = str(check["stage_id"])
        passed = bool(check["passed"])
        stage_reached = reached
        if stage_reached and not passed and not first_failed:
            first_failed = stage_id
        stages.append(
            {
                "stage_id": stage_id,
                "reached": stage_reached,
                "passed": passed,
                "outcome": (
                    "pass" if stage_reached and passed else (
                        "block" if stage_reached else "not_reached"
                    )
                ),
                "reason_code": (
                    "passed"
                    if stage_reached and passed
                    else (
                        str(check["reason_code"])
                        if stage_reached
                        else "earlier_stage_blocked"
                    )
                ),
                "reason": (
                    "passed"
                    if stage_reached and passed
                    else (
                        str(check["reason"])
                        if stage_reached
                        else "not reached because an earlier stage blocked"
                    )
                ),
                "assessment_reason": str(check["reason"]),
                "required_for_paper": stage_id in paper_required,
                "required_for_live": stage_id in live_required,
                "evidence": deepcopy(_mapping(check.get("evidence"))),
            }
        )
        reached = bool(reached and passed)
    stages.append(
        {
            "stage_id": "09_shadow_priority",
            "reached": reached,
            "passed": False,
            "outcome": "pending" if reached else "not_reached",
            "reason_code": (
                "utility_and_maturity_pending" if reached else "earlier_stage_blocked"
            ),
            "reason": "evaluated below after utility and evidence-maturity scoring",
            "required_for_paper": "09_shadow_priority" in paper_required,
            "required_for_live": "09_shadow_priority" in live_required,
            "evidence": {},
        }
    )
    stages.append(
        {
            "stage_id": "10_outcome_learning",
            "reached": False,
            "passed": False,
            "outcome": "pending_outcome",
            "reason_code": "future_outcome_join_pending",
            "reason": "pending future outcome join",
            "required_for_paper": False,
            "required_for_live": False,
            "evidence": {},
        }
    )
    return stages, first_failed


def _sleeve_check_rows(
    components: Mapping[str, float],
    policy: Mapping[str, Any],
) -> list[dict[str, Any]]:
    family = _mapping(policy.get("resolved_sleeve_family"))
    overrides = _mapping(family.get("qualification_floor_overrides"))
    rows: list[dict[str, Any]] = []
    for component in sorted(COMPONENT_NAMES):
        if component not in overrides:
            continue
        actual = _number(components.get(component), 0.0)
        floor = _number(overrides.get(component), 0.0)
        rows.append(
            {
                "check_id": f"sleeve_component_floor:{component}",
                "passed": actual >= floor,
                "actual": round(actual, 6),
                "floor": round(floor, 6),
            }
        )
    return rows


def _evidence_actions(first_failed_stage: str, diagnostics: Mapping[str, Any]) -> list[str]:
    actions = {
        "01_observation": "repair decision identity and point-in-time lineage",
        "02_data_qualification": "refresh or repair the missing source and feature evidence",
        "03_signal_formation": "retain the HOLD and label the near-boundary observation for calibration",
        "04_consensus_and_regime": "collect independent cluster votes and regime-specific outcomes",
        "05_post_cost_edge": "calibrate edge in basis points with lower confidence bounds and realistic costs",
        "06_execution_feasibility": "improve spread, depth, latency, slippage, impact, or route evidence",
        "07_portfolio_fit": "improve overlap netting, capital budgets, and concentration-aware allocation",
        "08_non_bypassable_risk": "retain the veto and repair its underlying broker, account, or risk evidence",
        "09_shadow_priority": "accumulate candidate-bound out-of-sample post-cost outcomes",
    }
    result = [actions.get(first_failed_stage, "continue collecting point-in-time decision evidence")]
    if diagnostics.get("edge_evidence_kind") != "lower_confidence_bound":
        result.append("materialize an explicit post-cost edge lower confidence bound")
    if int(diagnostics.get("post_cost_samples", 0) or 0) < 30:
        result.append("accumulate at least 30 identity-bound post-cost samples before qualification")
    if not diagnostics.get("market_impact_curve_available", False):
        result.append("collect size-dependent market-impact evidence before making capital-scale claims")
    return list(dict.fromkeys(result))[:4]


def _build_decision_trace(
    *,
    classification: str,
    qualified: bool,
    protected_hold: bool,
    intent_action: str,
    final_action: str,
    stages: Sequence[Mapping[str, Any]],
    components: Mapping[str, float],
    diagnostics: Mapping[str, Any],
    policy: Mapping[str, Any],
    policy_receipt: Mapping[str, Any],
    action_semantics: Mapping[str, Any],
    quantitative_evidence: Mapping[str, Any],
    evidence_actions: Sequence[str],
    active_guard_reasons: Sequence[Any],
    ingestion_route: Mapping[str, Any],
) -> dict[str, Any]:
    stage_by_id = {
        str(stage.get("stage_id") or ""): dict(stage)
        for stage in stages
        if isinstance(stage, Mapping)
    }
    playbook = _mapping(policy.get("decision_playbook"))
    paper_required = [
        str(value) for value in (playbook.get("paper_required_stages") or [])
    ]
    live_required = [
        str(value) for value in (playbook.get("live_required_stages") or [])
    ]

    def progress(required: Sequence[str]) -> dict[str, Any]:
        passed = [
            stage_id
            for stage_id in required
            if bool(_mapping(stage_by_id.get(stage_id)).get("passed", False))
        ]
        failed = [stage_id for stage_id in required if stage_id not in passed]
        return {
            "passed": len(passed),
            "required": len(required),
            "passed_stage_ids": passed,
            "failed_stage_ids": failed,
            "complete": not failed,
        }

    blocking = next(
        (
            dict(stage)
            for stage in stages
            if isinstance(stage, Mapping) and stage.get("outcome") == "block"
        ),
        {},
    )
    paper_progress = progress(paper_required)
    live_progress = progress(live_required)
    if intent_action not in DIRECTIONAL_ACTIONS:
        paper_gate = "intentional_no_trade"
    elif not paper_progress["complete"]:
        paper_gate = "required_stage_blocked"
    elif qualified:
        paper_gate = "fully_qualified"
    else:
        paper_gate = "bounded_evidence_path"

    if not bool(policy_receipt.get("execution_eligible", False)):
        live_gate = "execution_ineligible_sleeve"
    elif intent_action not in DIRECTIONAL_ACTIONS:
        live_gate = "intentional_no_trade"
    elif not live_progress["complete"]:
        live_gate = "required_stage_blocked"
    elif not bool(quantitative_evidence.get("live_ready", False)):
        live_gate = "quantitative_evidence_blocked"
    elif not bool(action_semantics.get("ready", False)):
        live_gate = "position_transition_blocked"
    elif not qualified:
        live_gate = "candidate_not_qualified"
    else:
        live_gate = "decision_quality_qualified"

    regime_floor = _number(
        _mapping(policy.get("qualification_floors")).get("regime_alignment"),
        0.45,
    )
    regime_state = (
        "aligned"
        if _number(components.get("regime_alignment")) >= regime_floor
        else "below_family_floor"
    )
    edge_kind = str(diagnostics.get("edge_evidence_kind") or "missing")
    if bool(diagnostics.get("edge_proven", False)):
        edge_state = "positive_post_cost_lcb"
    elif edge_kind == "lower_confidence_bound":
        edge_state = "lcb_did_not_clear_costs"
    elif edge_kind == "point_estimate":
        edge_state = "point_estimate_only"
    else:
        edge_state = "missing"

    reason_codes = [classification]
    if blocking:
        reason_codes.append(str(blocking.get("reason_code") or "stage_blocked"))
    reason_codes.extend(str(value) for value in active_guard_reasons if str(value))
    if not bool(action_semantics.get("ready", False)):
        reason_codes.extend(
            f"action_semantics:{value}"
            for value in (action_semantics.get("reasons") or [])
            if str(value)
        )
    material = {
        "contract_version": "sleeve_decision_trace_v1",
        "decision_state": classification,
        "qualified_shadow_candidate": qualified,
        "protected_hold": protected_hold,
        "intent_action": intent_action,
        "final_action": final_action,
        "current_stage": str(
            blocking.get("stage_id")
            or ("10_outcome_learning" if qualified else "09_shadow_priority")
        ),
        "next_stage": str(
            blocking.get("stage_id")
            or ("10_outcome_learning" if qualified else "09_shadow_priority")
        ),
        "blocking": {
            "stage_id": str(blocking.get("stage_id") or ""),
            "reason_code": str(blocking.get("reason_code") or ""),
            "reason": str(blocking.get("reason") or ""),
        },
        "stage_progress": {
            "paper": paper_progress,
            "live": live_progress,
        },
        "mode_quality_gates": {
            "paper": paper_gate,
            "live": live_gate,
        },
        "regime_state": regime_state,
        "regime_alignment_norm": round(
            _number(components.get("regime_alignment")), 6
        ),
        "regime_floor_norm": round(regime_floor, 6),
        "edge_state": edge_state,
        "data_route": {
            "status": str(ingestion_route.get("status") or "unavailable"),
            "route_state": str(
                ingestion_route.get("route_state") or "unavailable"
            ),
            "profile_id": str(ingestion_route.get("profile_id") or ""),
            "quality_norm": round(
                _number(ingestion_route.get("average_route_score_norm")), 6
            ),
            "paper_coverage_norm": round(
                _number(ingestion_route.get("paper_required_coverage_norm")), 6
            ),
            "live_coverage_norm": round(
                _number(ingestion_route.get("live_required_coverage_norm")), 6
            ),
            "paper_ready": bool(
                ingestion_route.get("paper_decision_data_ready", False)
            ),
            "live_ready": bool(
                ingestion_route.get("live_decision_data_ready", False)
            ),
            "receipt_valid": bool(ingestion_route.get("receipt_valid", False)),
            "receipt_sha256": str(
                ingestion_route.get("decision_route_receipt_sha256") or ""
            ),
        },
        "position_transition": str(action_semantics.get("semantic") or "unknown"),
        "reason_codes": list(dict.fromkeys(reason_codes))[:12],
        "next_evidence_actions": [str(value) for value in evidence_actions[:3]],
        "top_family_stage_priorities": [
            str(value)
            for value in (policy_receipt.get("decision_stage_priority") or [])[:3]
        ],
        "decision_playbook_sha256": str(
            policy_receipt.get("decision_playbook_sha256") or ""
        ),
        "live_execution_authority": False,
    }
    return {**material, "trace_sha256": _canonical_hash(material)}


def evaluate_decision(
    row: Mapping[str, Any],
    policy: Mapping[str, Any],
) -> dict[str, Any]:
    values = dict(row)
    ingestion_route = _bounded_ingestion_route(values.get("ingestion_route"))
    resolved_policy, policy_receipt = _policy_for_evaluation(values, policy)
    components, diagnostics = _score_components(values, resolved_policy)
    strategy_definition = deepcopy(
        _mapping(resolved_policy.get("strategy_definition"))
    )
    quantitative_evidence = _quantitative_evidence_assessment(
        values,
        components,
        diagnostics,
        resolved_policy,
    )
    action_semantics = _action_semantics(values, resolved_policy)
    weights = _mapping(resolved_policy.get("component_weights"))
    floors = _mapping(resolved_policy.get("qualification_floors"))
    raw_utility = sum(
        _number(weights.get(name)) * score for name, score in components.items()
    )
    uncertainty_penalty = 0.0
    if diagnostics.get("edge_evidence_kind") == "missing":
        uncertainty_penalty += 0.08
    elif diagnostics.get("edge_evidence_kind") != "lower_confidence_bound":
        uncertainty_penalty += 0.04
    if int(diagnostics.get("post_cost_samples", 0) or 0) < 30:
        uncertainty_penalty += 0.05
    uncertainty_penalty += 0.05 * _clamp01(diagnostics.get("conflict_pressure_norm", 0.0))
    if quantitative_evidence.get("explicit_adverse", False):
        uncertainty_penalty += 0.05
    decision_utility = _clamp01(raw_utility - uncertainty_penalty)

    stages, first_failed = _stage_rows(
        row=values,
        components=components,
        diagnostics=diagnostics,
        policy=resolved_policy,
    )
    sleeve_checks = _sleeve_check_rows(components, resolved_policy)
    intent_action = str(
        values.get("master_intent_action") or values.get("master_action") or "HOLD"
    ).upper()
    final_action = str(values.get("action") or values.get("master_action") or "HOLD").upper()
    protected_hold = bool(intent_action in DIRECTIONAL_ACTIONS and final_action != intent_action)
    qualified = bool(
        not first_failed
        and components["evidence_maturity"]
        >= _number(floors.get("evidence_maturity"), 0.50)
        and decision_utility >= _number(floors.get("qualified_utility"), 0.72)
        and not bool(quantitative_evidence.get("explicit_adverse", False))
    )
    stages[8] = {
        "stage_id": "09_shadow_priority",
        "reached": not bool(first_failed),
        "passed": qualified,
        "outcome": (
            "pass" if qualified else (
                "block" if not first_failed else "not_reached"
            )
        ),
        "reason_code": (
            "passed"
            if qualified
            else (
                "utility_or_evidence_maturity_below_floor"
                if not first_failed
                else "earlier_stage_blocked"
            )
        ),
        "reason": (
            "qualified shadow opportunity"
            if qualified
            else "utility or evidence maturity is below qualification floor"
        ),
        "required_for_paper": "09_shadow_priority"
        in {
            str(value)
            for value in (
                _mapping(resolved_policy.get("active_paper_control")).get(
                    "required_pass_stages"
                )
                or []
            )
        },
        "required_for_live": "09_shadow_priority"
        in {
            str(value)
            for value in (
                _mapping(resolved_policy.get("active_live_control")).get(
                    "required_pass_stages"
                )
                or []
            )
        },
        "evidence": {
            "decision_utility_norm": round(decision_utility, 6),
            "qualified_utility_floor_norm": round(
                _number(floors.get("qualified_utility"), 0.72), 6
            ),
            "evidence_maturity_norm": round(
                components["evidence_maturity"], 6
            ),
            "evidence_maturity_floor_norm": round(
                _number(floors.get("evidence_maturity"), 0.50), 6
            ),
            "explicit_adverse_quantitative_evidence": bool(
                quantitative_evidence.get("explicit_adverse", False)
            ),
        },
    }
    if not first_failed and not qualified:
        first_failed = "09_shadow_priority"

    if protected_hold:
        classification = "protected_hold"
    elif intent_action not in DIRECTIONAL_ACTIONS:
        classification = "no_edge_hold"
    elif qualified:
        classification = "qualified_shadow_candidate"
    elif first_failed == "02_data_qualification":
        classification = "data_evidence_blocked"
    elif first_failed == "05_post_cost_edge":
        classification = "economic_edge_unproven"
    elif decision_utility >= _number(floors.get("near_miss_utility"), 0.55):
        classification = "watchlist_near_miss"
    else:
        classification = "shadow_candidate_rejected"

    evidence_actions = list(
        dict.fromkeys(
            [
                *_evidence_actions(first_failed, diagnostics),
                *[
                    f"materialize direct quantitative evidence for {axis}"
                    for axis in quantitative_evidence.get(
                        "missing_required_axes", []
                    )
                ],
                *[
                    f"replace proxy evidence with direct measurement for {axis}"
                    for axis in quantitative_evidence.get(
                        "proxy_only_required_axes", []
                    )
                ],
            ]
        )
    )[:8]
    decision_trace = _build_decision_trace(
        classification=classification,
        qualified=qualified,
        protected_hold=protected_hold,
        intent_action=intent_action,
        final_action=final_action,
        stages=stages,
        components=components,
        diagnostics=diagnostics,
        policy=resolved_policy,
        policy_receipt=policy_receipt,
        action_semantics=action_semantics,
        quantitative_evidence=quantitative_evidence,
        evidence_actions=evidence_actions,
        active_guard_reasons=values.get("decision_guard_reasons") or [],
        ingestion_route=ingestion_route,
    )

    identity = {
        "timestamp_utc": _utc_text(values.get("timestamp_utc")),
        "message_id": str(values.get("message_id") or ""),
        "run_id": str(values.get("run_id") or ""),
        "snapshot_id": str(values.get("snapshot_id") or ""),
        "broker": str(values.get("broker") or values.get("source_broker") or ""),
        "profile": str(values.get("shadow_profile") or values.get("profile") or "default"),
        "domain": str(values.get("shadow_domain") or values.get("domain") or ""),
        "lifecycle_state": str(values.get("lifecycle_state") or ""),
        "lane": str(values.get("routing_lane") or _mapping(values.get("portfolio")).get("runtime_lane") or "unclassified"),
        "symbol": str(values.get("symbol") or ""),
        "intent_action": intent_action,
        "final_action": final_action,
        "intent_score": round(_number(values.get("master_intent_score"), 0.5), 8),
        "resolved_policy_id": str(policy_receipt.get("resolved_policy_id") or ""),
        "resolved_policy_sha256": str(
            policy_receipt.get("resolved_policy_sha256") or ""
        ),
        "strategy_definition_sha256": str(
            policy_receipt.get("strategy_definition_sha256") or ""
        ),
        "decision_playbook_sha256": str(
            policy_receipt.get("decision_playbook_sha256") or ""
        ),
        "strategy_specialization_id": str(
            _mapping(values.get("strategy_specialization")).get(
                "selected_strategy_id"
            )
            or ""
        ),
        "strategy_specialization_receipt_sha256": str(
            _mapping(values.get("strategy_specialization")).get(
                "contract_receipt_sha256"
            )
            or ""
        ),
        "ingestion_route_receipt_sha256": str(
            ingestion_route.get("decision_route_receipt_sha256") or ""
        ),
    }
    evaluation_id = _canonical_hash(identity)
    return {
        "evaluation_id": evaluation_id,
        **identity,
        "active_decision_disposition": str(values.get("decision_disposition") or ""),
        "active_blocking_stage": str(values.get("decision_blocking_stage") or ""),
        "active_guard_categories": [
            str(item) for item in (values.get("decision_guard_categories") or [])
        ],
        "active_guard_reasons": [
            str(item) for item in (values.get("decision_guard_reasons") or [])
        ],
        "classification": classification,
        "qualified_shadow_candidate": qualified,
        "protected_hold": protected_hold,
        "policy_receipt": policy_receipt,
        "strategy_definition": {
            **strategy_definition,
            "strategy_variant_id": str(
                policy_receipt.get("strategy_variant_id") or ""
            ),
            "strategy_definition_sha256": str(
                policy_receipt.get("strategy_definition_sha256") or ""
            ),
            "complete": bool(
                policy_receipt.get("strategy_definition_complete", False)
            ),
        },
        "decision_playbook": deepcopy(
            _mapping(resolved_policy.get("decision_playbook"))
        ),
        "decision_trace": decision_trace,
        "ingestion_route": ingestion_route,
        "strategy_specialization": deepcopy(
            _mapping(values.get("strategy_specialization"))
        ),
        "action_semantics": action_semantics,
        "quantitative_evidence": quantitative_evidence,
        "research_challengers": deepcopy(
            _mapping(
                _mapping(values.get("quantitative_evidence")).get(
                    "_challengers"
                )
            )
        ),
        "sleeve_policy": {
            "policy_family_id": str(policy_receipt.get("policy_family_id") or ""),
            "objective": str(policy_receipt.get("objective") or ""),
            "evidence_focus": list(policy_receipt.get("evidence_focus") or []),
            "execution_eligible": bool(policy_receipt.get("execution_eligible", False)),
            "paper_live_policy_parity": bool(
                policy_receipt.get("paper_live_policy_parity", False)
            ),
            "decision_horizon": str(
                policy_receipt.get("decision_horizon") or ""
            ),
            "portfolio_role": str(policy_receipt.get("portfolio_role") or ""),
            "primary_edge": str(policy_receipt.get("primary_edge") or ""),
            "checks": sleeve_checks,
        },
        "decision_quality_utility_norm": round(decision_utility, 6),
        "decision_quality_utility_score": round(decision_utility * 100.0, 3),
        "raw_weighted_utility_norm": round(raw_utility, 6),
        "uncertainty_penalty_norm": round(uncertainty_penalty, 6),
        "components": {key: round(value, 6) for key, value in components.items()},
        "diagnostics": diagnostics,
        "stages": stages,
        "first_failed_stage": first_failed,
        "evidence_actions": evidence_actions,
        "capital_scale_evidence": {
            "status": "unproven" if not (
                diagnostics.get("market_impact_curve_available", False)
                and diagnostics.get("edge_proven", False)
                and int(diagnostics.get("post_cost_samples", 0) or 0) >= 30
            ) else "bounded_evidence_available",
            "market_impact_curve_available": bool(
                diagnostics.get("market_impact_curve_available", False)
            ),
            "positive_post_cost_lcb_available": bool(
                diagnostics.get("post_cost_lower_confidence_bound") is not None
                and _number(diagnostics.get("post_cost_lower_confidence_bound")) > 0.0
            ),
            "post_cost_samples": int(diagnostics.get("post_cost_samples", 0) or 0),
            "max_deployable_capital_inferred": False,
        },
        "authority": {
            "changes_active_action": False,
            "changes_position_size": False,
            "paper_order_authority": False,
            "live_order_authority": False,
            "promotion_authority": False,
        },
    }


def _receipt_mismatch_reasons(
    supplied: Mapping[str, Any],
    expected: Mapping[str, Any],
    *,
    prefix: str,
) -> list[str]:
    reasons: list[str] = []
    for key in (
        "base_policy_id",
        "policy_family_id",
        "resolved_policy_id",
        "resolved_policy_sha256",
        "strategy_variant_id",
        "strategy_definition_sha256",
        "strategy_definition_complete",
        "decision_playbook_version",
        "decision_playbook_sha256",
        "decision_stage_priority",
        "required_quantitative_evidence",
        "profile",
        "domain",
        "lifecycle_state",
        "execution_eligible",
        "paper_live_policy_parity",
    ):
        if supplied.get(key) != expected.get(key):
            reasons.append(f"{prefix}_{key}_mismatch")
    return reasons


def evaluate_execution_policy_guard(
    *,
    intent: Mapping[str, Any],
    target_mode: str,
    policy: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Revalidate the sleeve policy receipt before paper or live execution."""

    active_policy = dict(policy) if isinstance(policy, Mapping) else load_policy()
    payload = dict(intent)
    metadata = _mapping(payload.get("metadata"))
    flow = _mapping(metadata.get("institutional_decision_flow"))
    evaluation = _mapping(flow.get("evaluation"))
    paper_control = _mapping(flow.get("control"))
    supplied_receipt = _mapping(
        flow.get("policy_receipt")
        or evaluation.get("policy_receipt")
        or paper_control.get("policy_receipt")
    )
    mode = str(target_mode or "").strip().lower()
    strategy = str(payload.get("strategy") or "").strip().lower()
    required = bool(
        mode == "live"
        or flow
        or str(metadata.get("layer") or "").strip().lower() == "grand_master"
        or strategy == "grand_master_bot"
    )
    action = _normalized_action(payload.get("action"))
    quantity = max(_number(payload.get("quantity")), 0.0)
    profile = str(
        metadata.get("source_profile")
        or supplied_receipt.get("profile")
        or evaluation.get("profile")
        or "default"
    )
    domain = str(
        metadata.get("shadow_domain")
        or supplied_receipt.get("domain")
        or evaluation.get("domain")
        or ""
    )
    lifecycle = str(
        metadata.get("lifecycle_state")
        or supplied_receipt.get("lifecycle_state")
        or evaluation.get("lifecycle_state")
        or ""
    )
    resolved_policy, expected_receipt = resolve_sleeve_policy(
        profile,
        active_policy,
        domain=domain,
        lifecycle_state=lifecycle,
    )
    reasons: list[str] = []
    specialization_reasons = strategy_specialization_guard_reasons(
        metadata,
        require_candidate=mode == "live",
        require_regime_alignment=mode == "live",
    )
    if mode == "live":
        reasons.extend(
            f"strategy_specialization:{reason}"
            for reason in specialization_reasons
        )
    if not required and not flow:
        return {
            "allow_execute": True,
            "required": False,
            "target_mode": mode,
            "status": "legacy_paper_passthrough",
            "action": action,
            "quantity": round(quantity, 6),
            "reasons": [],
            "policy_receipt": expected_receipt,
            "control": {},
        }
    if not flow:
        reasons.append("decision_flow_metadata_missing")
    if not evaluation:
        reasons.append("decision_flow_evaluation_missing")
    if not paper_control:
        reasons.append("decision_flow_paper_control_missing")
    if not supplied_receipt:
        reasons.append("decision_flow_policy_receipt_missing")
    else:
        reasons.extend(
            _receipt_mismatch_reasons(
                supplied_receipt,
                expected_receipt,
                prefix="decision_flow_receipt",
            )
        )
    evaluation_receipt = _mapping(evaluation.get("policy_receipt"))
    if evaluation_receipt:
        reasons.extend(
            _receipt_mismatch_reasons(
                evaluation_receipt,
                expected_receipt,
                prefix="decision_flow_evaluation_receipt",
            )
        )
    elif evaluation:
        reasons.append("decision_flow_evaluation_receipt_missing")
    paper_receipt = _mapping(paper_control.get("policy_receipt"))
    if paper_receipt:
        reasons.extend(
            _receipt_mismatch_reasons(
                paper_receipt,
                expected_receipt,
                prefix="decision_flow_paper_control_receipt",
            )
        )
    elif paper_control:
        reasons.append("decision_flow_paper_control_receipt_missing")
    expected_evaluation_sha256 = _canonical_hash(dict(evaluation)) if evaluation else ""
    if paper_control and str(paper_control.get("evaluation_sha256") or "") != expected_evaluation_sha256:
        reasons.append("decision_flow_evaluation_digest_mismatch")
    if evaluation and str(evaluation.get("evaluation_id") or "") != str(
        paper_control.get("evaluation_id") or ""
    ):
        reasons.append("decision_flow_evaluation_id_mismatch")
    if action not in DIRECTIONAL_ACTIONS or quantity <= 0.0:
        reasons.append("decision_flow_non_directional_or_zero_intent")
    if not bool(expected_receipt.get("execution_eligible", False)):
        reasons.append("decision_flow_sleeve_not_execution_eligible")

    output_action = action
    output_quantity = quantity
    applied_control: dict[str, Any] = {}
    if mode == "paper":
        if str(paper_control.get("target_mode") or "").strip().lower() != "paper":
            reasons.append("decision_flow_paper_control_mode_mismatch")
        if not bool(paper_control.get("authorized_mode", False)):
            reasons.append("decision_flow_paper_control_not_authorized")
        if _normalized_action(paper_control.get("output_action")) != action:
            reasons.append("decision_flow_paper_output_action_mismatch")
        authorized_quantity = max(_number(paper_control.get("output_quantity")), 0.0)
        if quantity > authorized_quantity + 1e-9:
            reasons.append("decision_flow_paper_quantity_exceeds_authorized_output")
        applied_control = paper_control
    elif mode == "live" and not reasons:
        output_action, output_quantity, applied_control = apply_decision_flow_control(
            target_mode="live",
            current_action=action,
            quantity=quantity,
            evaluation=evaluation,
            policy=resolved_policy,
        )
        if not bool(applied_control.get("enabled", False)) or not bool(
            applied_control.get("authorized_mode", False)
        ):
            reasons.append("decision_flow_live_control_not_authorized")
        if output_action not in DIRECTIONAL_ACTIONS or output_quantity <= 0.0:
            reasons.extend(
                str(reason)
                for reason in (applied_control.get("reasons") or [])
                if str(reason)
            )
            reasons.append("decision_flow_live_quality_gate_blocked")
        live_control = _mapping(resolved_policy.get("active_live_control"))
        max_age_seconds = max(_number(live_control.get("max_evaluation_age_seconds"), 900.0), 1.0)
        evaluation_ts = _utc_text(evaluation.get("timestamp_utc"))
        if not evaluation_ts:
            reasons.append("decision_flow_live_evaluation_timestamp_missing")
        else:
            evaluation_dt = datetime.fromisoformat(evaluation_ts)
            age_seconds = max(
                (datetime.now(timezone.utc) - evaluation_dt).total_seconds(),
                0.0,
            )
            if age_seconds > max_age_seconds:
                reasons.append("decision_flow_live_evaluation_stale")
    elif mode not in {"paper", "live"}:
        reasons.append("decision_flow_unknown_target_mode")

    reasons = list(dict.fromkeys(reasons))
    allow_execute = bool(
        not reasons
        and output_action in DIRECTIONAL_ACTIONS
        and output_quantity > 0.0
    )
    return {
        "allow_execute": allow_execute,
        "required": required,
        "target_mode": mode,
        "status": "ready" if allow_execute else "blocked",
        "action": output_action if allow_execute else "HOLD",
        "quantity": round(output_quantity if allow_execute else 0.0, 6),
        "input_action": action,
        "input_quantity": round(quantity, 6),
        "quantity_reduced": bool(output_quantity + 1e-12 < quantity),
        "reasons": reasons,
        "policy_receipt": expected_receipt,
        "control": applied_control,
        "strategy_specialization_reasons": specialization_reasons,
    }


def build_report(
    rows: Iterable[Mapping[str, Any]],
    policy: Mapping[str, Any],
    *,
    generated_at_utc: str | None = None,
    max_ranked_rows: int = 250,
) -> dict[str, Any]:
    evaluated = [evaluate_decision(row, policy) for row in rows if isinstance(row, Mapping)]
    latest_by_identity: dict[tuple[str, str, str], dict[str, Any]] = {}
    for item in evaluated:
        key = (item["profile"], item["lane"], item["symbol"])
        current = latest_by_identity.get(key)
        if current is None or item["timestamp_utc"] >= current["timestamp_utc"]:
            latest_by_identity[key] = item
    current_rows = sorted(
        latest_by_identity.values(),
        key=lambda item: (
            -_number(item.get("decision_quality_utility_norm")),
            item.get("profile", ""),
            item.get("lane", ""),
            item.get("symbol", ""),
        ),
    )
    for rank, item in enumerate(current_rows, start=1):
        item["global_shadow_rank"] = rank

    classification_counts = Counter(item["classification"] for item in current_rows)
    family_counts = Counter(
        str(_mapping(item.get("policy_receipt")).get("policy_family_id") or "missing")
        for item in current_rows
    )
    horizon_counts = Counter(
        str(_mapping(item.get("strategy_definition")).get("decision_horizon") or "missing")
        for item in current_rows
    )
    portfolio_role_counts = Counter(
        str(_mapping(item.get("strategy_definition")).get("portfolio_role") or "missing")
        for item in current_rows
    )
    quantitative_gap_counts = Counter(
        axis
        for item in current_rows
        for axis in (
            list(
                _mapping(item.get("quantitative_evidence")).get(
                    "missing_required_axes", []
                )
            )
            + list(
                _mapping(item.get("quantitative_evidence")).get(
                    "proxy_only_required_axes", []
                )
            )
            + list(
                _mapping(item.get("quantitative_evidence")).get(
                    "failed_required_axes", []
                )
            )
        )
    )
    failure_counts = Counter(item["first_failed_stage"] or "none" for item in current_rows)
    disposition_counts = Counter(
        item["active_decision_disposition"] or "missing" for item in current_rows
    )
    guard_category_counts = Counter(
        category
        for item in current_rows
        for category in item.get("active_guard_categories", [])
    )
    guard_reason_counts = Counter(
        reason
        for item in current_rows
        for reason in item.get("active_guard_reasons", [])
    )
    stage_ids = [str(row.get("stage_id")) for row in policy.get("stages", []) if isinstance(row, Mapping)]
    funnel: list[dict[str, Any]] = []
    previous_passed = len(current_rows)
    for stage_id in stage_ids:
        reached = sum(
            1
            for item in current_rows
            for stage in item["stages"]
            if stage["stage_id"] == stage_id and stage["reached"]
        )
        passed = sum(
            1
            for item in current_rows
            for stage in item["stages"]
            if stage["stage_id"] == stage_id and stage["reached"] and stage["passed"]
        )
        funnel.append(
            {
                "stage_id": stage_id,
                "reached": reached,
                "passed": passed,
                "conversion_from_previous_passed": round(
                    passed / max(previous_passed, 1), 6
                ),
            }
        )
        previous_passed = passed

    lane_rankings: dict[str, list[dict[str, Any]]] = {}
    for item in current_rows:
        key = f"{item['profile']}::{item['lane']}"
        lane_rankings.setdefault(key, [])
        if len(lane_rankings[key]) < 10:
            lane_rankings[key].append(
                {
                    "rank": len(lane_rankings[key]) + 1,
                    "evaluation_id": item["evaluation_id"],
                    "symbol": item["symbol"],
                    "classification": item["classification"],
                    "policy_family_id": str(
                        _mapping(item.get("policy_receipt")).get(
                            "policy_family_id"
                        )
                        or ""
                    ),
                    "strategy_variant_id": str(
                        _mapping(item.get("policy_receipt")).get(
                            "strategy_variant_id"
                        )
                        or ""
                    ),
                    "decision_horizon": str(
                        _mapping(item.get("strategy_definition")).get(
                            "decision_horizon"
                        )
                        or ""
                    ),
                    "intent_action": item["intent_action"],
                    "utility_score": item["decision_quality_utility_score"],
                    "first_failed_stage": item["first_failed_stage"],
                }
            )

    directional = sum(1 for item in current_rows if item["intent_action"] in DIRECTIONAL_ACTIONS)
    protected = sum(1 for item in current_rows if item["protected_hold"])
    qualified = sum(1 for item in current_rows if item["qualified_shadow_candidate"])
    explicit_edge = sum(
        1 for item in current_rows if item["diagnostics"].get("edge_proven", False)
    )
    capacity_evidence = sum(
        1
        for item in current_rows
        if item["diagnostics"].get("market_impact_curve_available", False)
    )
    post_cost_evidence = sum(
        1
        for item in current_rows
        if int(item["diagnostics"].get("post_cost_samples", 0) or 0) >= 30
        and item["diagnostics"].get("post_cost_lower_confidence_bound") is not None
    )
    complete_strategy_definitions = sum(
        1
        for item in current_rows
        if bool(_mapping(item.get("strategy_definition")).get("complete", False))
    )
    position_aware_actions = sum(
        1
        for item in current_rows
        if bool(_mapping(item.get("action_semantics")).get("ready", False))
    )
    live_quantitative_evidence_ready = sum(
        1
        for item in current_rows
        if bool(_mapping(item.get("quantitative_evidence")).get("live_ready", False))
    )
    direct_coverage_values = [
        _clamp01(
            _mapping(item.get("quantitative_evidence")).get(
                "direct_coverage_norm", 0.0
            )
        )
        for item in current_rows
    ]
    timestamp = generated_at_utc or datetime.now(timezone.utc).isoformat()
    report_id = _canonical_hash(
        {
            "policy_id": policy.get("policy_id"),
            "evaluation_ids": sorted(item["evaluation_id"] for item in current_rows),
        }
    )
    count = len(current_rows)
    return {
        "timestamp_utc": _utc_text(timestamp),
        "schema_version": 2,
        "policy_id": str(policy.get("policy_id") or ""),
        "report_id": report_id,
        "ok": bool(current_rows),
        "overall_status": "ready" if current_rows else "no_evidence",
        "operating_mode": "hierarchical_strategy_resolved_read_only_shadow_evaluation",
        "input_contract": {
            "raw_rows_evaluated": len(evaluated),
            "latest_profile_lane_symbol_rows": count,
            "point_in_time_only": True,
            "future_outcomes_used_for_current_ranking": False,
        },
        "authority_contract": _mapping(policy.get("authority")),
        "mode_parity_contract": _mapping(policy.get("mode_parity_contract")),
        "soak_contract": {
            **_mapping(policy.get("soak_contract")),
            "active_action_change_count": 0,
            "position_size_change_count": 0,
            "order_submission_count": 0,
            "candidate_mutation_count": 0,
        },
        "decision_efficiency": {
            "directional_intent_count": directional,
            "directional_intent_rate": round(directional / max(count, 1), 6),
            "protected_hold_count": protected,
            "protected_hold_rate_of_directional": round(protected / max(directional, 1), 6),
            "qualified_shadow_candidate_count": qualified,
            "qualified_rate_of_directional": round(qualified / max(directional, 1), 6),
            "no_edge_hold_count": int(classification_counts.get("no_edge_hold", 0)),
            "no_edge_hold_rate": round(
                int(classification_counts.get("no_edge_hold", 0)) / max(count, 1), 6
            ),
            "classification_counts": dict(sorted(classification_counts.items())),
            "policy_family_counts": dict(sorted(family_counts.items())),
            "decision_horizon_counts": dict(sorted(horizon_counts.items())),
            "portfolio_role_counts": dict(sorted(portfolio_role_counts.items())),
            "first_failed_stage_counts": dict(sorted(failure_counts.items())),
            "active_disposition_counts": dict(sorted(disposition_counts.items())),
            "active_guard_category_counts": dict(sorted(guard_category_counts.items())),
            "active_guard_reason_counts": dict(sorted(guard_reason_counts.items())),
        },
        "strategy_definition_coverage": {
            "complete_count": complete_strategy_definitions,
            "total_count": count,
            "complete_rate": round(
                complete_strategy_definitions / max(count, 1), 6
            ),
            "position_aware_action_ready_count": position_aware_actions,
            "position_aware_action_ready_rate": round(
                position_aware_actions / max(count, 1), 6
            ),
        },
        "quantitative_evidence_readiness": {
            "live_ready_count": live_quantitative_evidence_ready,
            "total_count": count,
            "live_ready_rate": round(
                live_quantitative_evidence_ready / max(count, 1), 6
            ),
            "mean_direct_coverage_norm": round(
                _mean(direct_coverage_values), 6
            ),
            "gap_counts": dict(sorted(quantitative_gap_counts.items())),
            "missing_is_not_assumed_passing": True,
            "proxy_evidence_is_not_live_proof": True,
        },
        "funnel": funnel,
        "capital_scale_readiness": {
            "status": "unproven" if not (
                count > 0
                and explicit_edge == count
                and capacity_evidence == count
                and post_cost_evidence == count
            ) else "bounded_evidence_available",
            "explicit_positive_post_cost_edge_rate": round(explicit_edge / max(count, 1), 6),
            "market_impact_curve_coverage_rate": round(capacity_evidence / max(count, 1), 6),
            "candidate_post_cost_evidence_rate": round(post_cost_evidence / max(count, 1), 6),
            "max_deployable_capital_inferred": False,
            "profitability_guaranteed": False,
            "policy": _mapping(policy.get("capital_scale_contract")).get("policy", ""),
        },
        "rankings": {
            "global": current_rows[: max(int(max_ranked_rows), 1)],
            "by_profile_lane": lane_rankings,
        },
        "interpretation": {
            "no_edge_hold": "the active strategy intentionally found no directional edge",
            "protected_hold": "a directional intent was stopped by an active safety or execution control",
            "economic_edge_unproven": "a directional intent lacks a positive post-cost lower confidence bound",
            "qualified_shadow_candidate": "all shadow stages passed; this still grants no order authority",
        },
    }
