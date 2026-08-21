from __future__ import annotations

import hashlib
import json
import math
import re
from copy import deepcopy
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, Sequence

from core.strategy_validity import default_validity_contract


PROJECT_ROOT = Path(__file__).resolve().parents[1]
POLICY_PATH = PROJECT_ROOT / "config" / "sleeve_strategy_contracts_v1.json"
REQUIRED_CONTRACT_FIELDS = (
    "strategy_id",
    "strategy_name",
    "sleeve_id",
    "objective_class",
    "economic_thesis",
    "universe",
    "decision_horizon",
    "holding_horizon",
    "entry_rule",
    "add_rule",
    "trim_rule",
    "exit_rule",
    "time_stop",
    "label_definition",
    "allowed_regimes",
    "blocked_regimes",
    "cost_model",
    "capacity_method",
    "benchmark",
    "risk_budget",
    "shorting_policy",
    "conflict_policy",
    "evidence_policy",
    "validity_contract",
    "lifecycle_policy",
    "runtime_authority",
    "library_tier",
    "activation_state",
    "strategy_definition",
)
FORBIDDEN_AUTHORITY = (
    "can_create_intent",
    "can_reverse_intent",
    "can_increase_quantity",
    "can_allocate_capital",
    "can_change_labels",
    "can_grant_promotion",
    "can_submit_live_order",
)
MASTER_STRATEGIES = frozenset(
    {
        "grand_master_bot",
        "grand_master",
        "master_bot",
        "master_futures_bot",
        "master_equity_bot",
        "master_options_bot",
        "master_crypto_bot",
        "default",
    }
)
_MATERIALIZATION_CACHE: dict[str, dict[str, dict[str, Any]]] = {}
_LIBRARY_CACHE: dict[str, dict[str, dict[str, Any]]] = {}
_SLEEVE_INDEX_CACHE: dict[tuple[int, str], tuple[dict[str, Any], ...]] = {}
_REGIME_SOURCE_CACHE: dict[str, tuple[int, int, dict[str, Any]]] = {}

_PLAYBOOK_TEMPLATES: dict[str, dict[str, list[str] | str]] = {
    "trend": {
        "ideal_regimes": ["persistent_direction", "broad_confirmation", "adequate_liquidity"],
        "hostile_regimes": ["choppy_range", "crowded_reversal", "gap_without_confirmation"],
        "required_inputs": ["point_in_time_returns", "trend_strength", "breadth", "volume_and_liquidity"],
        "failure_modes": ["false_breakout", "late_entry", "trend_crowding", "regime_reversal"],
    },
    "mean_reversion": {
        "ideal_regimes": ["stable_relationship", "bounded_range", "temporary_dislocation"],
        "hostile_regimes": ["structural_break", "persistent_trend", "liquidity_shock"],
        "required_inputs": ["point_in_time_spread", "normalization_window", "relationship_stability", "cost_and_borrow"],
        "failure_modes": ["falling_knife", "relationship_break", "slow_convergence", "crowded_exit"],
    },
    "carry_value": {
        "ideal_regimes": ["stable_funding", "orderly_curve", "adequate_term_premium"],
        "hostile_regimes": ["funding_shock", "curve_inversion_break", "forced_deleveraging"],
        "required_inputs": ["carry_or_yield", "roll_down", "funding_and_borrow", "valuation_and_risk"],
        "failure_modes": ["carry_crash", "value_trap", "funding_reversal", "hidden_duration"],
    },
    "event": {
        "ideal_regimes": ["verified_event_window", "measurable_surprise", "qualified_liquidity"],
        "hostile_regimes": ["source_disagreement", "leakage_or_stale_event", "unbounded_gap_risk"],
        "required_inputs": ["event_timestamp", "point_in_time_consensus", "surprise_measure", "implied_move_and_liquidity"],
        "failure_modes": ["already_priced_event", "whipsaw", "source_latency", "gap_beyond_risk_budget"],
    },
    "volatility": {
        "ideal_regimes": ["observable_surface", "hedgeable_underlier", "priced_volatility_dislocation"],
        "hostile_regimes": ["surface_staleness", "gap_risk", "unhedgeable_liquidity"],
        "required_inputs": ["implied_volatility_surface", "realized_volatility", "greeks", "premium_and_hedging_cost"],
        "failure_modes": ["volatility_regime_jump", "greek_instability", "hedging_cost_overrun", "tail_loss"],
    },
    "liquidity_execution": {
        "ideal_regimes": ["observable_depth", "bounded_toxicity", "stable_venue_state"],
        "hostile_regimes": ["quote_fade", "halt_or_reopen", "latency_spike", "thin_book"],
        "required_inputs": ["quotes_and_depth", "trade_flow", "spread", "latency_and_fill_quality"],
        "failure_modes": ["adverse_selection", "queue_decay", "impact_underestimate", "inventory_trap"],
    },
    "risk_control": {
        "ideal_regimes": ["observable_risk_state", "verified_control_inputs", "recoverable_operation"],
        "hostile_regimes": ["missing_telemetry", "correlated_failure", "control_loop_instability"],
        "required_inputs": ["fresh_health_evidence", "risk_limits", "incident_history", "recovery_receipts"],
        "failure_modes": ["false_positive_halt", "missed_failure", "restart_loop", "stale_clearance"],
    },
    "general": {
        "ideal_regimes": ["contract_supported", "source_verified", "cost_qualified"],
        "hostile_regimes": ["contract_unsupported", "source_unverified", "risk_veto"],
        "required_inputs": ["point_in_time_features", "source_quality", "cost_estimate", "risk_state"],
        "failure_modes": ["overfit", "feature_decay", "cost_misspecification", "regime_shift"],
    },
}


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _normalize(value: Any) -> str:
    text = re.sub(r"[^a-z0-9]+", "_", str(value or "").strip().lower())
    return text.strip("_")


def _number(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float(default)
    return result if math.isfinite(result) else float(default)


def _canonical_hash(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return {}
    return payload if isinstance(payload, dict) else {}


@lru_cache(maxsize=8)
def _read_json_cached(path_text: str, mtime_ns: int) -> dict[str, Any]:
    del mtime_ns
    return _read_json(Path(path_text))


@lru_cache(maxsize=4)
def _load_policy_cached(path_text: str, mtime_ns: int) -> dict[str, Any]:
    del mtime_ns
    return _read_json(Path(path_text))


def load_policy(path: Path | None = None) -> dict[str, Any]:
    policy_path = (path or POLICY_PATH).resolve()
    try:
        mtime_ns = policy_path.stat().st_mtime_ns
    except OSError:
        mtime_ns = 0
    policy = deepcopy(_load_policy_cached(str(policy_path), mtime_ns))
    validate_policy(policy)
    return policy


def validate_policy(policy: Mapping[str, Any]) -> None:
    if int(policy.get("schema_version") or 0) != 1:
        raise ValueError("strategy specialization policy schema must be version 1")
    authority = _mapping(policy.get("authority"))
    if set(FORBIDDEN_AUTHORITY) - set(authority):
        raise ValueError("strategy specialization authority contract is incomplete")
    if any(bool(authority.get(key, False)) for key in FORBIDDEN_AUTHORITY):
        raise ValueError("strategy specialization policy requests forbidden authority")
    binding = _mapping(policy.get("candidate_binding"))
    if not bool(binding.get("required", False)):
        raise ValueError("candidate binding may not be disabled")
    if bool(binding.get("allow_historical_fallback", True)) or bool(
        binding.get("allow_cross_candidate_pooling", True)
    ):
        raise ValueError("candidate binding may not borrow historical evidence")
    objective_classes = _mapping(policy.get("objective_classes"))
    sleeves = _mapping(policy.get("sleeves"))
    if not sleeves:
        raise ValueError("strategy specialization policy defines no sleeves")
    missing_objectives = sorted(
        {
            str(_mapping(row).get("objective_class") or "")
            for row in sleeves.values()
        }
        - set(objective_classes)
    )
    if missing_objectives:
        raise ValueError(f"unknown objective classes: {','.join(missing_objectives)}")
    infrastructure = _mapping(sleeves.get("infrastructure_risk"))
    if str(infrastructure.get("objective_class") or "") != "control_only":
        raise ValueError("infrastructure_risk must remain control_only")
    library = _mapping(policy.get("strategy_library"))
    if not bool(library.get("enabled", False)):
        raise ValueError("strategy library expansion may not be disabled")
    target_total = int(library.get("target_total_strategies") or 0)
    minimum_per_sleeve = int(library.get("minimum_strategies_per_sleeve") or 0)
    if target_total != 12000:
        raise ValueError("strategy library target must remain exactly 12000")
    if minimum_per_sleeve < 100:
        raise ValueError("every sleeve must retain at least 100 strategy hypotheses")
    overlays = _mapping(library.get("conditioning_overlays"))
    archetypes = _mapping(library.get("objective_archetypes"))
    if len(overlays) < 10:
        raise ValueError("strategy library requires at least ten conditioning overlays")
    missing_archetypes = sorted(set(objective_classes) - set(archetypes))
    if missing_archetypes:
        raise ValueError(
            f"strategy library lacks objective archetypes: {','.join(missing_archetypes)}"
        )
    if any(
        not isinstance(archetypes.get(objective), list)
        or len(archetypes.get(objective) or []) < 10
        for objective in objective_classes
    ):
        raise ValueError("each objective class requires at least ten archetypes")
    if int(library.get("max_concurrent_generated_experiments_per_sleeve") or 0) > int(
        library.get("max_generated_hot_slots_per_sleeve") or 0
    ):
        raise ValueError("concurrent generated experiments exceed the hot-slot limit")
    quality = _mapping(policy.get("quality_assessment"))
    if not bool(quality.get("unknown_is_not_bad", False)):
        raise ValueError("insufficient evidence may not be labeled bad")
    regime = _mapping(policy.get("regime_adaptation"))
    if not bool(regime.get("enabled", False)):
        raise ValueError("strategy regime adaptation may not be disabled")
    if bool(
        regime.get("may_mutate_action_quantity_risk_limits_or_live_authority", True)
    ):
        raise ValueError("regime adaptation requests forbidden execution authority")
    if not str(regime.get("source_path") or "").strip():
        raise ValueError("regime adaptation source path is required")
    if not _mapping(regime.get("regime_affinity")):
        raise ValueError("regime adaptation requires explicit affinity mappings")


def canonical_profile(
    profile: Any,
    policy: Mapping[str, Any] | None = None,
    *,
    broker: Any = "",
    domain: Any = "",
) -> str:
    active_policy = policy if isinstance(policy, Mapping) else load_policy()
    normalized = _normalize(profile) or "default"
    broker_name = _normalize(broker)
    domain_name = _normalize(domain)
    if normalized == "default" and (
        broker_name == "coinbase" or domain_name == "crypto"
    ):
        normalized = "crypto_spot"
    aliases = _mapping(active_policy.get("profile_aliases"))
    return _normalize(aliases.get(normalized) or normalized)


def _manifest_path(policy: Mapping[str, Any], project_root: Path) -> Path:
    source = str(policy.get("source_manifest") or "config/sleeve_strategy_expansion.json")
    path = Path(source)
    return path if path.is_absolute() else project_root / path


def _taxonomy_groups(strategy_name: str, policy: Mapping[str, Any]) -> list[str]:
    normalized = _normalize(strategy_name)
    groups: list[str] = []
    for group, tokens in sorted(_mapping(policy.get("taxonomy")).items()):
        if not isinstance(tokens, Sequence) or isinstance(tokens, (str, bytes)):
            continue
        if any(_normalize(token) in normalized for token in tokens if _normalize(token)):
            groups.append(_normalize(group))
    return groups or ["general"]


def _taxonomy_rules(groups: Sequence[str], holding_horizon: str) -> dict[str, str]:
    group_set = set(groups)
    if "mean_reversion" in group_set:
        entry = "enter_only_after_measured_dislocation_and_relationship_stability_confirmation"
        exit_rule = "exit_on_convergence_relationship_break_stop_or_time_stop"
    elif "event" in group_set:
        entry = "enter_only_inside_the_defined_event_window_after_source_and_implied_move_qualification"
        exit_rule = "exit_on_event_window_expiry_thesis_break_target_or_gap_risk_veto"
    elif "carry_value" in group_set:
        entry = "enter_only_when_expected_carry_or_value_exceeds_cost_funding_and_risk_hurdles"
        exit_rule = "exit_when_carry_compresses_value_closes_funding_reverses_or_risk_is_invalidated"
    elif "trend" in group_set:
        entry = "enter_only_after_point_in_time_trend_confirmation_with_cost_and_regime_support"
        exit_rule = "exit_on_trend_break_trailing_risk_limit_target_or_time_stop"
    elif "liquidity_execution" in group_set:
        entry = "enter_only_when_quote_depth_toxicity_latency_and_expected_shortfall_are_qualified"
        exit_rule = "exit_on_inventory_limit_toxicity_shift_liquidity_loss_or_session_boundary"
    else:
        entry = "enter_only_after_source_quality_signal_regime_cost_and_risk_qualification"
        exit_rule = "exit_on_thesis_break_stop_target_or_regime_invalidation"
    return {
        "entry_rule": entry,
        "exit_rule": exit_rule,
        "time_stop": f"expire_at_{_normalize(holding_horizon) or 'strategy_horizon'}_without_requalification",
    }


def _display_name(value: Any) -> str:
    return " ".join(part.capitalize() for part in _normalize(value).split("_") if part)


def _strategy_definition(
    *,
    sleeve_id: str,
    strategy_name: str,
    objective_class: str,
    groups: Sequence[str],
    objective: Mapping[str, Any],
    source_kind: str,
    library_tier: str,
    archetype: str = "",
    overlay: str = "",
    overlay_definition: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    primary_group = next(
        (group for group in groups if group in _PLAYBOOK_TEMPLATES),
        "general",
    )
    playbook = _PLAYBOOK_TEMPLATES[primary_group]
    display = _display_name(strategy_name)
    sleeve_display = _display_name(sleeve_id)
    archetype_name = _normalize(archetype) or _normalize(strategy_name)
    overlay_name = _normalize(overlay) or "native_strategy_logic"
    overlay_row = _mapping(overlay_definition)
    confirmation = str(
        overlay_row.get("confirmation")
        or "strategy_signal_plus_source_cost_regime_and_risk_qualification"
    )
    overlay_failure = str(
        overlay_row.get("failure_mode") or "strategy_specific_model_error"
    )
    failures = list(playbook.get("failure_modes") or [])
    if overlay_failure not in failures:
        failures.append(overlay_failure)
    return {
        "display_name": display,
        "plain_language_summary": (
            f"Tests {display.lower()} inside the {sleeve_display} sleeve using "
            f"{_display_name(overlay_name).lower()} confirmation."
        ),
        "edge_hypothesis": (
            f"The {archetype_name} relationship may improve the "
            f"{objective_class} objective after costs when its confirmation and regime contract hold."
        ),
        "signal_family": primary_group,
        "taxonomy_groups": list(groups),
        "confirmation_requirement": confirmation,
        "ideal_regimes": list(playbook.get("ideal_regimes") or []),
        "hostile_regimes": list(playbook.get("hostile_regimes") or []),
        "required_inputs": list(playbook.get("required_inputs") or []),
        "expected_failure_modes": failures,
        "evaluation_question": (
            f"Does {display} improve {str(objective.get('primary_metric') or 'the sleeve objective')} "
            "on candidate-forward point-in-time evidence after realistic costs and objective-specific risk?"
        ),
        "benchmark_rule": str(objective.get("benchmark_rule") or ""),
        "source_kind": source_kind,
        "library_tier": library_tier,
        "archetype": archetype_name,
        "conditioning_overlay": overlay_name,
        "unknown_evidence_is_bad": False,
    }


def _derived_objective_class(sleeve_id: str) -> str:
    name = _normalize(sleeve_id)
    rules = (
        ("control_only", ("infrastructure", "data_plane", "backpressure", "orchestration", "adversarial", "security", "architecture", "alpha_research_os", "data_ingestion", "data_plumbing", "data_confidence", "market_data", "governance", "evidence_court", "execution_safety", "model_risk", "provider_adapter", "runtime_capacity", "halt_recovery", "event_intelligence", "gpu_quant_acceleration", "privacy", "formal_backend", "signal_governance", "xva_counterparty", "uncertainty_robust_control")),
        ("capital_preservation", ("cash", "capital_preservation", "portfolio_construction", "position_lifecycle", "collateral_margin_liquidity")),
        ("hedge_utility", ("hedge", "hedging", "tail_risk", "tail_dependency", "black_swan", "risk_parity")),
        ("volatility_relative_value", ("volatility", "variance", "option", "gamma", "vanna", "volga", "greek", "swaption", "barrier", "lookback", "dispersion")),
        ("execution_alpha", ("market_making", "order_flow", "microstructure", "high_frequency", "low_latency", "execution_quality", "transaction_cost", "liquidity_regime")),
        ("event_alpha", ("earnings", "event_reaction", "event_driven", "catalyst")),
        ("basis_relative_value", ("basis", "pricing_model", "synthetic_cdo", "cdo_squared", "cdo_cubed", "structured_product")),
        ("market_neutral_relative_value", ("statistical_arbitrage", "stat_arb", "pairs", "relative_value", "cross_asset")),
        ("digital_asset_alpha", ("crypto", "digital_asset")),
        ("income_total_return", ("dividend", "income")),
        ("macro_carry_relative_value", ("macro", "rates", "bond", "credit", "commodity", "futures", "fx", "international", "sovereign", "inflation", "repo_securities", "securitized_product")),
    )
    for objective, tokens in rules:
        if any(token in name for token in tokens):
            return objective
    return "directional_alpha"


def _derived_sleeve_definition(sleeve_id: str, policy: Mapping[str, Any]) -> dict[str, Any]:
    derived = deepcopy(_mapping(policy.get("derived_sleeve_policy")))
    derived.pop("purpose", None)
    objective_class = _derived_objective_class(sleeve_id)
    derived["objective_class"] = objective_class
    if objective_class == "control_only":
        derived.update(
            {
                "economic_thesis": "This operational sleeve prevents unsafe or invalid system behavior and has no trading-profit objective.",
                "universe": "system_controls",
                "decision_horizon": "continuous",
                "holding_horizon": "not_applicable",
                "entry_rule": "not_applicable_no_trade_authority",
                "add_rule": "not_applicable_no_trade_authority",
                "trim_rule": "not_applicable_no_trade_authority",
                "exit_rule": "not_applicable_no_trade_authority",
                "time_stop": "not_applicable",
                "benchmark": "control_slo",
                "risk_budget": "zero_trading_risk",
                "shorting_policy": "forbidden",
            }
        )
    return derived


def _contract_complete(contract: Mapping[str, Any]) -> bool:
    for field in REQUIRED_CONTRACT_FIELDS:
        value = contract.get(field)
        if value in (None, "", [], {}):
            return False
    return True


def _build_contract(
    *,
    sleeve_id: str,
    strategy_name: str,
    source_kind: str,
    policy: Mapping[str, Any],
    source_manifest_sha256: str,
    synthetic_description: str = "",
    library_tier: str = "hot_catalog",
    activation_state: str = "admitted_hot",
    archetype: str = "",
    overlay: str = "",
    overlay_definition: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    defaults = _mapping(policy.get("contract_defaults"))
    sleeve = _mapping(_mapping(policy.get("sleeves")).get(sleeve_id))
    if not sleeve:
        sleeve = _derived_sleeve_definition(sleeve_id, policy)
    objective_class = str(sleeve.get("objective_class") or "directional_alpha")
    objective = _mapping(_mapping(policy.get("objective_classes")).get(objective_class))
    groups = _taxonomy_groups(strategy_name, policy)
    rules = _taxonomy_rules(groups, str(sleeve.get("holding_horizon") or defaults.get("holding_horizon") or ""))
    normalized_strategy = _normalize(strategy_name) or "unknown"
    strategy_id = f"sleeve::{sleeve_id}::{normalized_strategy}::v1"
    economic_thesis = str(synthetic_description or sleeve.get("economic_thesis") or defaults.get("economic_thesis") or "")
    definition = _strategy_definition(
        sleeve_id=sleeve_id,
        strategy_name=normalized_strategy,
        objective_class=objective_class,
        groups=groups,
        objective=objective,
        source_kind=source_kind,
        library_tier=library_tier,
        archetype=archetype,
        overlay=overlay,
        overlay_definition=overlay_definition,
    )
    ideal_regimes = list(definition.get("ideal_regimes") or ["contract_supported"])
    hostile_regimes = list(definition.get("hostile_regimes") or [])
    blocked_regimes = list(defaults.get("blocked_regimes") or [])
    blocked_regimes.extend(
        value for value in hostile_regimes if value not in blocked_regimes
    )
    contract: dict[str, Any] = {
        **deepcopy(defaults),
        **deepcopy(sleeve),
        **rules,
        "policy_id": str(policy.get("policy_id") or ""),
        "strategy_id": strategy_id,
        "strategy_name": normalized_strategy,
        "sleeve_id": sleeve_id,
        "source_kind": source_kind,
        "display_name": str(definition.get("display_name") or normalized_strategy),
        "library_tier": library_tier,
        "activation_state": activation_state,
        "economic_thesis": economic_thesis,
        "taxonomy_groups": groups,
        "strategy_definition": definition,
        "allowed_regimes": ideal_regimes,
        "blocked_regimes": blocked_regimes,
        "objective_scorecard": deepcopy(objective),
        "label_definition": (
            "candidate_forward_post_cost_outcome_at_"
            f"{_normalize(sleeve.get('holding_horizon') or defaults.get('holding_horizon')) or 'contract_horizon'}"
            "_with_point_in_time_features"
        ),
        "authority": deepcopy(_mapping(policy.get("authority"))),
        "candidate_binding_policy": deepcopy(_mapping(policy.get("candidate_binding"))),
        "quality_verdict_policy": {
            "unknown_is_not_bad": bool(
                _mapping(policy.get("quality_assessment")).get("unknown_is_not_bad", False)
            ),
            "good_requires": str(
                _mapping(policy.get("quality_assessment")).get("good_requires") or ""
            ),
            "bad_requires": str(
                _mapping(policy.get("quality_assessment")).get("bad_requires") or ""
            ),
        },
        "validity_contract": default_validity_contract(),
        "regime_adaptation_policy": {
            "enabled": bool(
                _mapping(policy.get("regime_adaptation")).get("enabled", False)
            ),
            "source_path": str(
                _mapping(policy.get("regime_adaptation")).get("source_path") or ""
            ),
            "fresh_source_required_for_cold_activation": bool(
                _mapping(policy.get("regime_adaptation")).get(
                    "fresh_source_required_for_cold_activation", True
                )
            ),
            "may_mutate_action_quantity_risk_limits_or_live_authority": bool(
                _mapping(policy.get("regime_adaptation")).get(
                    "may_mutate_action_quantity_risk_limits_or_live_authority", False
                )
            ),
        },
        "source_manifest_sha256": source_manifest_sha256,
    }
    contract["contract_complete"] = _contract_complete(contract)
    receipt_payload = {key: value for key, value in contract.items() if key != "contract_receipt_sha256"}
    contract["contract_receipt_sha256"] = _canonical_hash(receipt_payload)
    return contract


def materialize_strategy_contracts(
    *,
    policy: Mapping[str, Any] | None = None,
    manifest: Mapping[str, Any] | None = None,
    project_root: Path = PROJECT_ROOT,
) -> dict[str, dict[str, Any]]:
    active_policy = deepcopy(dict(policy)) if isinstance(policy, Mapping) else load_policy()
    validate_policy(active_policy)
    manifest_path = _manifest_path(active_policy, project_root)
    if isinstance(manifest, Mapping):
        active_manifest = deepcopy(dict(manifest))
    else:
        try:
            manifest_mtime_ns = manifest_path.stat().st_mtime_ns
        except OSError:
            manifest_mtime_ns = 0
        active_manifest = _read_json_cached(
            str(manifest_path.resolve()), manifest_mtime_ns
        )
    source_hash = _canonical_hash(active_manifest)
    cache_key = _canonical_hash(
        {
            "policy": active_policy,
            "manifest_sha256": source_hash,
            "project_root": str(project_root.resolve()),
        }
    )
    cached = _MATERIALIZATION_CACHE.get(cache_key)
    if cached is not None:
        return cached
    allowed_states = {str(value) for value in (active_policy.get("included_runtime_states") or [])}
    included = {_normalize(value) for value in (active_policy.get("included_sleeves") or [])}
    additions = _mapping(active_policy.get("strategy_additions"))
    contracts: dict[str, dict[str, Any]] = {}
    for raw_sleeve in active_manifest.get("sleeves") or []:
        if not isinstance(raw_sleeve, Mapping):
            continue
        sleeve_id = _normalize(raw_sleeve.get("name"))
        runtime_status = str(raw_sleeve.get("runtime_status") or "")
        if runtime_status not in allowed_states and sleeve_id not in included:
            continue
        manifest_names = [_normalize(value) for value in (raw_sleeve.get("strategies") or [])]
        addition_names = [_normalize(value) for value in (additions.get(sleeve_id) or [])]
        seen: set[str] = set()
        for strategy_name in manifest_names + addition_names:
            if not strategy_name or strategy_name in seen:
                continue
            seen.add(strategy_name)
            source_kind = "catalog" if strategy_name in manifest_names else "curated_addition"
            contract = _build_contract(
                sleeve_id=sleeve_id,
                strategy_name=strategy_name,
                source_kind=source_kind,
                policy=active_policy,
                source_manifest_sha256=source_hash,
            )
            contracts[contract["strategy_id"]] = contract
    result = dict(sorted(contracts.items()))
    if len(_MATERIALIZATION_CACHE) >= 4:
        oldest = next(iter(_MATERIALIZATION_CACHE))
        evicted = _MATERIALIZATION_CACHE.pop(oldest, None)
        stale_index_keys = [
            key for key in _SLEEVE_INDEX_CACHE if key[0] == id(evicted)
        ]
        for key in stale_index_keys:
            _SLEEVE_INDEX_CACHE.pop(key, None)
    _MATERIALIZATION_CACHE[cache_key] = result
    return result


def materialize_strategy_library(
    *,
    policy: Mapping[str, Any] | None = None,
    manifest: Mapping[str, Any] | None = None,
    project_root: Path = PROJECT_ROOT,
) -> dict[str, dict[str, Any]]:
    """Materialize the full cold research library without widening runtime fan-out."""
    active_policy = deepcopy(dict(policy)) if isinstance(policy, Mapping) else load_policy()
    validate_policy(active_policy)
    manifest_path = _manifest_path(active_policy, project_root)
    if isinstance(manifest, Mapping):
        active_manifest = deepcopy(dict(manifest))
    else:
        try:
            manifest_mtime_ns = manifest_path.stat().st_mtime_ns
        except OSError:
            manifest_mtime_ns = 0
        active_manifest = _read_json_cached(
            str(manifest_path.resolve()), manifest_mtime_ns
        )
    source_hash = _canonical_hash(active_manifest)
    library_policy = _mapping(active_policy.get("strategy_library"))
    cache_key = _canonical_hash(
        {
            "policy": active_policy,
            "manifest_sha256": source_hash,
            "project_root": str(project_root.resolve()),
            "mode": "full_strategy_library",
        }
    )
    cached = _LIBRARY_CACHE.get(cache_key)
    if cached is not None:
        return cached

    hot_contracts = materialize_strategy_contracts(
        policy=active_policy,
        manifest=active_manifest,
        project_root=project_root,
    )
    allowed_states = {
        str(value) for value in (active_policy.get("included_runtime_states") or [])
    }
    included = {
        _normalize(value) for value in (active_policy.get("included_sleeves") or [])
    }
    sleeve_ids = sorted(
        {
            _normalize(row.get("name"))
            for row in (active_manifest.get("sleeves") or [])
            if isinstance(row, Mapping)
            and _normalize(row.get("name"))
            and (
                str(row.get("runtime_status") or "") in allowed_states
                or _normalize(row.get("name")) in included
            )
        }
    )
    target_total = int(library_policy.get("target_total_strategies") or 0)
    minimum_per_sleeve = int(
        library_policy.get("minimum_strategies_per_sleeve") or 0
    )
    if not sleeve_ids or target_total < len(sleeve_ids) * minimum_per_sleeve:
        raise ValueError("strategy library target cannot satisfy the sleeve minimum")
    base_target, remainder = divmod(target_total, len(sleeve_ids))
    overlays = _mapping(library_policy.get("conditioning_overlays"))
    archetypes_by_objective = _mapping(library_policy.get("objective_archetypes"))
    library = {key: deepcopy(value) for key, value in hot_contracts.items()}

    for sleeve_index, sleeve_id in enumerate(sleeve_ids):
        sleeve_target = base_target + (1 if sleeve_index < remainder else 0)
        existing = {
            str(row.get("strategy_name") or "")
            for row in library.values()
            if row.get("sleeve_id") == sleeve_id
        }
        sleeve_definition = _mapping(_mapping(active_policy.get("sleeves")).get(sleeve_id))
        if not sleeve_definition:
            sleeve_definition = _derived_sleeve_definition(sleeve_id, active_policy)
        objective_class = str(
            sleeve_definition.get("objective_class") or "directional_alpha"
        )
        archetypes = [
            _normalize(value)
            for value in (archetypes_by_objective.get(objective_class) or [])
            if _normalize(value)
        ]
        candidates: list[tuple[str, str, Mapping[str, Any]]] = []
        for overlay_name, overlay_definition in sorted(overlays.items()):
            for archetype in archetypes:
                candidates.append(
                    (
                        f"research_{archetype}__{_normalize(overlay_name)}",
                        archetype,
                        _mapping(overlay_definition),
                    )
                )
        for strategy_name, archetype, overlay_definition in candidates:
            if len(existing) >= sleeve_target:
                break
            if strategy_name in existing:
                continue
            overlay_name = strategy_name.rsplit("__", 1)[-1]
            contract = _build_contract(
                sleeve_id=sleeve_id,
                strategy_name=strategy_name,
                source_kind="generated_research_hypothesis",
                policy=active_policy,
                source_manifest_sha256=source_hash,
                library_tier="cold_research",
                activation_state="cold_untested",
                archetype=archetype,
                overlay=overlay_name,
                overlay_definition=overlay_definition,
            )
            library[contract["strategy_id"]] = contract
            existing.add(strategy_name)
        if len(existing) != sleeve_target:
            raise ValueError(
                f"unable to materialize target strategy count for {sleeve_id}: "
                f"{len(existing)}/{sleeve_target}"
            )

    result = dict(sorted(library.items()))
    if len(result) != target_total:
        raise ValueError(
            f"strategy library count mismatch: {len(result)}/{target_total}"
        )
    if len(_LIBRARY_CACHE) >= 2:
        _LIBRARY_CACHE.pop(next(iter(_LIBRARY_CACHE)), None)
    _LIBRARY_CACHE[cache_key] = result
    return result


def _synthetic_contract(
    sleeve_id: str,
    strategy_name: str,
    *,
    policy: Mapping[str, Any],
    source_kind: str,
    description: str,
) -> dict[str, Any]:
    return _build_contract(
        sleeve_id=sleeve_id,
        strategy_name=strategy_name,
        source_kind=source_kind,
        policy=policy,
        source_manifest_sha256="runtime_synthetic",
        synthetic_description=description,
        library_tier="runtime_identity",
        activation_state="runtime_observed",
    )


def resolve_strategy_contract(
    profile: Any,
    raw_strategy: Any,
    *,
    policy: Mapping[str, Any] | None = None,
    manifest: Mapping[str, Any] | None = None,
    project_root: Path = PROJECT_ROOT,
    contracts: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    active_policy = deepcopy(dict(policy)) if isinstance(policy, Mapping) else load_policy()
    sleeve_id = canonical_profile(profile, active_policy)
    active_contracts = (
        contracts
        if isinstance(contracts, Mapping)
        else materialize_strategy_contracts(
            policy=active_policy,
            manifest=manifest,
            project_root=project_root,
        )
    )
    normalized_strategy = _normalize(raw_strategy) or "default"
    exact_id = f"sleeve::{sleeve_id}::{normalized_strategy}::v1"
    if exact_id in active_contracts:
        return deepcopy(dict(active_contracts[exact_id]))
    synthetic = _mapping(active_policy.get("synthetic_runtime_strategies"))
    if normalized_strategy == "paper_portfolio_consensus":
        name = "portfolio_consensus"
        source_kind = "synthetic_portfolio_consensus"
    elif normalized_strategy in MASTER_STRATEGIES or normalized_strategy.startswith("master_"):
        name = "ensemble_champion"
        source_kind = "synthetic_ensemble"
    else:
        name = f"runtime_challenger_{normalized_strategy}"
        source_kind = "runtime_challenger"
    description = str(
        synthetic.get(name)
        or synthetic.get("runtime_challenger")
        or "Runtime strategy retained under an explicit non-catalog identity."
    )
    return _synthetic_contract(
        sleeve_id,
        name,
        policy=active_policy,
        source_kind=source_kind,
        description=description,
    )


def _flatten_numeric_features(value: Any, prefix: str = "", depth: int = 0) -> dict[str, float]:
    if depth > 3:
        return {}
    if isinstance(value, Mapping):
        result: dict[str, float] = {}
        for key, child in list(value.items())[:128]:
            child_prefix = f"{prefix}_{_normalize(key)}" if prefix else _normalize(key)
            result.update(_flatten_numeric_features(child, child_prefix, depth + 1))
        return result
    if isinstance(value, bool):
        return {prefix: 1.0 if value else 0.0} if prefix else {}
    number = _number(value, math.nan)
    return {prefix: number} if prefix and math.isfinite(number) else {}


def _feature_factor_scores(features: Mapping[str, Any]) -> tuple[dict[str, float], float]:
    numeric = _flatten_numeric_features(features)
    factor_tokens = {
        "trend": ("momentum", "trend", "relative_strength", "slope", "breakout"),
        "mean_reversion": ("zscore", "deviation", "rsi", "reversion", "spread"),
        "carry_value": ("carry", "yield", "basis", "value", "funding", "roll"),
        "event": ("event", "surprise", "earnings", "news", "macro"),
        "volatility": ("volatility", "implied", "realized", "skew", "gamma", "variance"),
        "liquidity_execution": ("spread", "depth", "volume", "liquidity", "imbalance", "vwap"),
        "risk_control": ("drawdown", "risk", "stress", "quality", "source", "fresh"),
    }
    scores: dict[str, float] = {}
    matched_keys: set[str] = set()
    for factor, tokens in factor_tokens.items():
        values = [
            abs(value)
            for key, value in numeric.items()
            if any(token in key for token in tokens)
        ]
        matched_keys.update(
            key for key in numeric if any(token in key for token in tokens)
        )
        if not values:
            scores[factor] = 0.0
            continue
        bounded = [min(value / (1.0 + value), 1.0) for value in values[:16]]
        scores[factor] = sum(bounded) / len(bounded)
    coverage = min(len(matched_keys) / max(len(factor_tokens), 1), 1.0)
    return scores, coverage


def extract_current_regime(features: Mapping[str, Any] | None) -> str:
    queue: list[Any] = [_mapping(features)]
    preferred_keys = {
        "current_regime",
        "current_live_regime",
        "market_regime",
        "regime_state",
    }
    while queue:
        value = queue.pop(0)
        if not isinstance(value, Mapping):
            continue
        normalized_items = {_normalize(key): child for key, child in value.items()}
        for key in preferred_keys:
            regime = _normalize(normalized_items.get(key))
            if regime:
                return regime
        queue.extend(
            child for child in value.values() if isinstance(child, Mapping)
        )
    return ""


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


def resolve_runtime_regime_context(
    features: Mapping[str, Any] | None,
    *,
    policy: Mapping[str, Any] | None = None,
    project_root: Path = PROJECT_ROOT,
    now_utc: datetime | None = None,
) -> dict[str, Any]:
    embedded_regime = extract_current_regime(features)
    if embedded_regime:
        return {
            "current_regime": embedded_regime,
            "source_ready": True,
            "source_status": "runtime_point_in_time_features",
            "source_path": "decision_features",
            "source_age_seconds": 0.0,
            "fresh": True,
        }

    active_policy = policy if isinstance(policy, Mapping) else load_policy()
    regime_policy = _mapping(active_policy.get("regime_adaptation"))
    relative_path = str(regime_policy.get("source_path") or "").strip()
    max_age_seconds = max(
        _number(regime_policy.get("maximum_source_age_seconds"), 0.0),
        0.0,
    )
    if not relative_path:
        return {
            "current_regime": "",
            "source_ready": False,
            "source_status": "missing_source_path",
            "source_path": "",
            "source_age_seconds": None,
            "fresh": False,
        }

    source_path = Path(relative_path)
    if not source_path.is_absolute():
        source_path = project_root / source_path
    try:
        stat = source_path.stat()
        cache_key = str(source_path.resolve())
        cached = _REGIME_SOURCE_CACHE.get(cache_key)
        signature = (int(stat.st_mtime_ns), int(stat.st_size))
        if cached is not None and cached[:2] == signature:
            payload = cached[2]
        else:
            loaded = json.loads(source_path.read_text(encoding="utf-8"))
            payload = dict(loaded) if isinstance(loaded, Mapping) else {}
            _REGIME_SOURCE_CACHE[cache_key] = (*signature, payload)
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return {
            "current_regime": "",
            "source_ready": False,
            "source_status": "unreadable",
            "source_path": str(source_path),
            "source_age_seconds": None,
            "fresh": False,
        }

    regime = extract_current_regime(payload)
    timestamp = _parse_timestamp(
        payload.get("timestamp_utc")
        or payload.get("generated_at_utc")
        or payload.get("updated_at_utc")
    )
    observed_now = now_utc or datetime.now(timezone.utc)
    if observed_now.tzinfo is None:
        observed_now = observed_now.replace(tzinfo=timezone.utc)
    age_seconds = (
        max((observed_now.astimezone(timezone.utc) - timestamp).total_seconds(), 0.0)
        if timestamp is not None
        else None
    )
    fresh = bool(
        timestamp is not None
        and age_seconds is not None
        and max_age_seconds > 0.0
        and age_seconds <= max_age_seconds
    )
    return {
        "current_regime": regime,
        "source_ready": bool(regime and fresh),
        "source_status": str(
            payload.get("overall_status") or payload.get("status") or "unknown"
        ).strip().lower(),
        "source_path": str(source_path),
        "source_timestamp_utc": timestamp.isoformat() if timestamp else "",
        "source_age_seconds": round(age_seconds, 3) if age_seconds is not None else None,
        "fresh": fresh,
    }


def strategy_regime_assessment(
    contract: Mapping[str, Any],
    current_regime: Any,
    *,
    policy: Mapping[str, Any] | None = None,
    source_ready: bool = True,
    source_status: str = "ready",
) -> dict[str, Any]:
    active_policy = policy if isinstance(policy, Mapping) else load_policy()
    regime_policy = _mapping(active_policy.get("regime_adaptation"))
    regime = _normalize(current_regime)
    objective = str(contract.get("objective_class") or "")
    source_kind = str(contract.get("source_kind") or "")
    library_tier = str(contract.get("library_tier") or "")
    normalized_source_status = _normalize(source_status)
    execution_alignment_ready = bool(
        source_ready
        and normalized_source_status in {"ready", "runtime_point_in_time_features"}
    )
    groups = {_normalize(value) for value in (contract.get("taxonomy_groups") or [])}
    if not bool(regime_policy.get("enabled", False)):
        return {
            "current_regime": regime or "unknown",
            "relevance": "disabled",
            "cold_activation_eligible": False,
            "source_ready": False,
            "execution_alignment_ready": False,
            "reason": "regime_adaptation_disabled",
        }
    if not regime or not source_ready:
        return {
            "current_regime": regime or "unknown",
            "relevance": "unknown",
            "cold_activation_eligible": False,
            "source_ready": bool(source_ready),
            "source_status": str(source_status or "unknown"),
            "execution_alignment_ready": False,
            "reason": "fresh_confident_regime_evidence_required",
        }
    affinity = _mapping(_mapping(regime_policy.get("regime_affinity")).get(regime))
    if not affinity:
        return {
            "current_regime": regime,
            "relevance": "unknown",
            "cold_activation_eligible": False,
            "source_ready": True,
            "source_status": str(source_status or "ready"),
            "execution_alignment_ready": False,
            "reason": "regime_not_in_policy",
        }
    favored = {_normalize(value) for value in (affinity.get("favored_taxonomy") or [])}
    guarded = {_normalize(value) for value in (affinity.get("guarded_taxonomy") or [])}
    favored_matches = sorted(groups & favored)
    guarded_matches = sorted(groups & guarded)
    if source_kind in {"synthetic_ensemble", "synthetic_portfolio_consensus"}:
        relevance = "aligned"
        reason = "upstream_ensemble_receipt_is_revalidated_under_current_regime"
    elif objective == "control_only":
        relevance = "aligned"
        reason = "control_only_relevance_is_operational_not_profit_seeking"
    elif favored_matches:
        relevance = "aligned"
        reason = "strategy_taxonomy_matches_current_regime"
    elif guarded_matches:
        relevance = "guarded"
        reason = "strategy_taxonomy_is_hostile_in_current_regime"
    else:
        relevance = "neutral"
        reason = "no_direct_regime_affinity_match"
    return {
        "current_regime": regime,
        "relevance": relevance,
        "cold_activation_eligible": (
            library_tier == "cold_research"
            and relevance == "aligned"
            and str(source_status or "").lower() == "ready"
        ),
        "source_ready": True,
        "source_status": str(source_status or "ready"),
        "execution_alignment_ready": execution_alignment_ready,
        "favored_matches": favored_matches,
        "guarded_matches": guarded_matches,
        "reason": reason,
        "authority": "ranking_and_research_admission_only_no_intent_or_sizing_authority",
    }


def rank_counterfactual_strategies(
    profile: Any,
    features: Mapping[str, Any] | None,
    *,
    policy: Mapping[str, Any] | None = None,
    manifest: Mapping[str, Any] | None = None,
    project_root: Path = PROJECT_ROOT,
    limit: int = 3,
    contracts: Mapping[str, Mapping[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    active_policy = deepcopy(dict(policy)) if isinstance(policy, Mapping) else load_policy()
    sleeve_id = canonical_profile(profile, active_policy)
    active_contracts = (
        contracts
        if isinstance(contracts, Mapping)
        else materialize_strategy_contracts(
            policy=active_policy,
            manifest=manifest,
            project_root=project_root,
        )
    )
    factor_scores, coverage = _feature_factor_scores(_mapping(features))
    regime_context = resolve_runtime_regime_context(
        features,
        policy=active_policy,
        project_root=project_root,
    )
    current_regime = str(regime_context.get("current_regime") or "")
    rows: list[dict[str, Any]] = []
    index_key = (id(active_contracts), sleeve_id)
    sleeve_contracts = _SLEEVE_INDEX_CACHE.get(index_key)
    if sleeve_contracts is None:
        sleeve_contracts = tuple(
            dict(contract)
            for contract in active_contracts.values()
            if contract.get("sleeve_id") == sleeve_id
        )
        _SLEEVE_INDEX_CACHE[index_key] = sleeve_contracts
    for contract in sleeve_contracts:
        groups = list(contract.get("taxonomy_groups") or ["general"])
        regime_assessment = strategy_regime_assessment(
            contract,
            current_regime,
            policy=active_policy,
            source_ready=bool(regime_context.get("source_ready", False)),
            source_status=str(regime_context.get("source_status") or "missing"),
        )
        matched = [group for group in groups if factor_scores.get(group, 0.0) > 0.0]
        signal_score = max((factor_scores.get(group, 0.0) for group in groups), default=0.0)
        tie_break = int(str(contract.get("contract_receipt_sha256") or "0")[:8], 16) / 0xFFFFFFFF
        relevance_adjustment = {
            "aligned": 0.1,
            "neutral": 0.0,
            "unknown": -0.05,
            "guarded": -0.2,
        }.get(str(regime_assessment.get("relevance") or "unknown"), -0.05)
        score = (
            0.35
            + 0.45 * signal_score
            + 0.15 * coverage
            + 0.05 * tie_break
            + relevance_adjustment
        )
        rows.append(
            {
                "strategy_id": str(contract.get("strategy_id") or ""),
                "strategy_name": str(contract.get("strategy_name") or ""),
                "counterfactual_score": round(min(max(score, 0.0), 1.0), 8),
                "feature_coverage_ratio": round(coverage, 8),
                "matched_factors": matched,
                "regime_assessment": regime_assessment,
                "contract_receipt_sha256": str(contract.get("contract_receipt_sha256") or ""),
                "authority": "read_only_counterfactual_no_action_or_sizing_authority",
            }
        )
    rows.sort(key=lambda row: (-float(row["counterfactual_score"]), str(row["strategy_id"])))
    return rows[: max(int(limit), 0)]


def _candidate_id(metadata: Mapping[str, Any]) -> str:
    binding = _mapping(metadata.get("candidate_binding"))
    flow = _mapping(metadata.get("institutional_decision_flow"))
    return str(
        metadata.get("production_candidate_id")
        or metadata.get("candidate_id")
        or binding.get("candidate_id")
        or _mapping(flow.get("candidate_binding")).get("candidate_id")
        or ""
    ).strip()


def attach_strategy_specialization(
    metadata: Mapping[str, Any] | None,
    *,
    profile: Any,
    raw_strategy: Any,
    features: Mapping[str, Any] | None,
    action: Any,
    quantity: Any,
    policy: Mapping[str, Any] | None = None,
    manifest: Mapping[str, Any] | None = None,
    project_root: Path = PROJECT_ROOT,
) -> dict[str, Any]:
    md = deepcopy(dict(metadata)) if isinstance(metadata, Mapping) else {}
    active_policy = deepcopy(dict(policy)) if isinstance(policy, Mapping) else load_policy()
    resolved_profile = canonical_profile(
        profile or md.get("source_profile") or md.get("profile") or "default",
        active_policy,
        broker=md.get("source_broker") or md.get("broker"),
        domain=md.get("shadow_domain") or md.get("domain"),
    )
    contracts = materialize_strategy_contracts(
        policy=active_policy,
        manifest=manifest,
        project_root=project_root,
    )
    contract = resolve_strategy_contract(
        resolved_profile,
        raw_strategy,
        policy=active_policy,
        manifest=manifest,
        project_root=project_root,
        contracts=contracts,
    )
    ranking = rank_counterfactual_strategies(
        resolved_profile,
        features,
        policy=active_policy,
        manifest=manifest,
        project_root=project_root,
        limit=3,
        contracts=contracts,
    )
    regime_context = resolve_runtime_regime_context(
        features,
        policy=active_policy,
        project_root=project_root,
    )
    current_regime = str(regime_context.get("current_regime") or "")
    regime_assessment = strategy_regime_assessment(
        contract,
        current_regime,
        policy=active_policy,
        source_ready=bool(regime_context.get("source_ready", False)),
        source_status=str(regime_context.get("source_status") or "missing"),
    )
    regime_assessment["source_path"] = str(regime_context.get("source_path") or "")
    regime_assessment["source_timestamp_utc"] = str(
        regime_context.get("source_timestamp_utc") or ""
    )
    regime_assessment["source_age_seconds"] = regime_context.get("source_age_seconds")
    regime_assessment["fresh"] = bool(regime_context.get("fresh", False))
    receipt = {
        "policy_id": str(active_policy.get("policy_id") or ""),
        "strategy_id": str(contract.get("strategy_id") or ""),
        "sleeve_id": str(contract.get("sleeve_id") or ""),
        "source_kind": str(contract.get("source_kind") or ""),
        "objective_class": str(contract.get("objective_class") or ""),
        "contract_complete": bool(contract.get("contract_complete", False)),
        "contract_receipt_sha256": str(contract.get("contract_receipt_sha256") or ""),
        "candidate_id": _candidate_id(md),
        "authority": deepcopy(_mapping(active_policy.get("authority"))),
    }
    md["strategy_specialization"] = {
        "schema_version": 1,
        "policy_id": str(active_policy.get("policy_id") or ""),
        "profile": resolved_profile,
        "raw_strategy": _normalize(raw_strategy) or "default",
        "selected_strategy_id": str(contract.get("strategy_id") or ""),
        "selected_strategy_name": str(contract.get("strategy_name") or ""),
        "source_kind": str(contract.get("source_kind") or ""),
        "objective_class": str(contract.get("objective_class") or ""),
        "objective_scorecard": deepcopy(_mapping(contract.get("objective_scorecard"))),
        "strategy_definition": deepcopy(_mapping(contract.get("strategy_definition"))),
        "library_tier": str(contract.get("library_tier") or ""),
        "activation_state": str(contract.get("activation_state") or ""),
        "regime_assessment": regime_assessment,
        "contract_complete": bool(contract.get("contract_complete", False)),
        "contract_receipt_sha256": str(contract.get("contract_receipt_sha256") or ""),
        "contract_receipt": receipt,
        "counterfactual_ranking": ranking,
        "candidate_binding": {
            "candidate_id": receipt["candidate_id"],
            "required": True,
            "historical_fallback_allowed": False,
            "cross_candidate_pooling_allowed": False,
        },
        "action_observed": str(action or "").strip().upper(),
        "quantity_observed": round(max(_number(quantity), 0.0), 8),
        "action_or_quantity_mutated": False,
        "authority": deepcopy(_mapping(active_policy.get("authority"))),
    }
    return md


def strategy_specialization_guard_reasons(
    metadata: Mapping[str, Any] | None,
    *,
    require_candidate: bool = False,
    require_regime_alignment: bool = False,
) -> list[str]:
    md = _mapping(metadata)
    specialization = _mapping(md.get("strategy_specialization"))
    receipt = _mapping(specialization.get("contract_receipt"))
    reasons: list[str] = []
    if not specialization:
        return ["strategy_specialization_missing"]
    if not bool(specialization.get("contract_complete", False)):
        reasons.append("strategy_contract_incomplete")
    if str(specialization.get("selected_strategy_id") or "") != str(
        receipt.get("strategy_id") or ""
    ):
        reasons.append("strategy_contract_identity_mismatch")
    if str(specialization.get("contract_receipt_sha256") or "") != str(
        receipt.get("contract_receipt_sha256") or ""
    ):
        reasons.append("strategy_contract_receipt_mismatch")
    authority = _mapping(specialization.get("authority"))
    if any(bool(authority.get(key, False)) for key in FORBIDDEN_AUTHORITY):
        reasons.append("strategy_specialization_forbidden_authority")
    if bool(specialization.get("action_or_quantity_mutated", True)):
        reasons.append("strategy_specialization_mutated_intent")
    if require_candidate and not str(receipt.get("candidate_id") or "").strip():
        reasons.append("strategy_contract_candidate_binding_missing")
    if require_regime_alignment:
        regime = _mapping(specialization.get("regime_assessment"))
        if str(regime.get("relevance") or "") != "aligned":
            reasons.append("strategy_current_regime_alignment_missing")
        if not bool(regime.get("execution_alignment_ready", False)):
            reasons.append("strategy_current_regime_source_not_execution_ready")
    return reasons


def strategy_id_from_metadata(metadata: Mapping[str, Any] | None, fallback: Any = "unknown") -> str:
    specialization = _mapping(_mapping(metadata).get("strategy_specialization"))
    return str(
        specialization.get("selected_strategy_id")
        or _mapping(_mapping(metadata).get("strategy_contract")).get("strategy_id")
        or fallback
        or "unknown"
    ).strip().lower()
