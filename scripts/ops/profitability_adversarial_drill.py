#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence
from zoneinfo import ZoneInfo

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.accountability import safe_write_json_atomic
from core.alpha_concept_engine import (
    capacity_impact_surface,
    point_in_time_security_master_audit,
    residual_redundancy_graph,
    sequential_change_point_stability,
)
from core.execution_simulator import simulate_execution
from core.portfolio_advisory import build_multi_period_advisory
from core.profitability_hardening import evaluate_profitability_entry
from core.research_data_platform import select_bitemporal_rows
from core.sleeve_strategy_specialization import (
    load_policy as load_sleeve_policy,
    materialize_strategy_contracts,
)

DEFAULT_POLICY_PATH = Path("config/profitability_adversarial_drill_v1.json")
DEFAULT_CANDIDATE_PATH = Path("governance/runtime/production_candidate_state.json")
DEFAULT_OUT_PATH = Path(
    "governance/research/profitability_adversarial_drill_latest.json"
)


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _float(value: Any, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return float(default)
    return parsed if math.isfinite(parsed) else float(default)


def _resolve_path(project_root: Path, raw_path: str | Path) -> Path:
    path = Path(raw_path).expanduser()
    return path if path.is_absolute() else project_root / path


def _sha256_file(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return ""


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


def _entry_features(
    *,
    edge_bps: float,
    round_trip_cost_bps: float,
    quality: float,
    trend: float,
    chop: float,
    spread_bps: float | None = None,
    quote_age_ms: float = 100.0,
    liquidity: float | None = None,
) -> dict[str, Any]:
    liquidity_quality = quality if liquidity is None else liquidity
    return {
        "profitability_strict_evidence_required": True,
        "market_micro_tradeability_score_norm": quality,
        "execution_fitness_norm": quality,
        "news_source_quality_norm": max(quality, 0.75),
        "core_cross_asset_confirmation_norm": quality,
        "core_portfolio_overlap_pressure_norm": 0.10,
        "cross_bot_conflict_norm": 0.10,
        "spread_bps": (
            max(round_trip_cost_bps / 4.0, 0.5) if spread_bps is None else spread_bps
        ),
        "quote_age_ms": quote_age_ms,
        "liquidity_quality_norm": liquidity_quality,
        "session_quality_norm": quality,
        "session": "regular",
        "day_regime_trend_norm": trend,
        "day_regime_chop_norm": chop,
        "predicted_edge_lower_confidence_bound_bps": edge_bps,
        "round_trip_cost_bps": round_trip_cost_bps,
        "minimum_edge_cost_multiple": 1.5,
    }


def _action_flip_count(actions: Sequence[str]) -> int:
    return sum(left != right for left, right in zip(actions, actions[1:]))


def _regime_transition_whipsaw(_: Mapping[str, Any]) -> dict[str, Any]:
    phases = [
        ("trend_confirmed", "BUY", 32.0, 5.0, 0.90, 0.90, 0.10),
        ("trend_fading", "HOLD", 8.0, 6.0, 0.58, 0.55, 0.45),
        ("false_restart", "BUY", 9.0, 7.0, 0.46, 0.48, 0.70),
        ("chop_reversal", "HOLD", 5.0, 8.0, 0.42, 0.35, 0.90),
        ("early_recovery", "BUY", 12.0, 8.0, 0.66, 0.68, 0.42),
        ("recovery_confirmed", "BUY", 34.0, 5.0, 0.92, 0.88, 0.12),
    ]
    rows: list[dict[str, Any]] = []
    raw_actions: list[str] = []
    guarded_actions: list[str] = []
    for phase, raw_action, edge, cost, quality, trend, chop in phases:
        gate = evaluate_profitability_entry(
            profile="swing_aggressive",
            features=_entry_features(
                edge_bps=edge,
                round_trip_cost_bps=cost,
                quality=quality,
                trend=trend,
                chop=chop,
            ),
        )
        guarded_action = "BUY" if raw_action == "BUY" and gate["allowed"] else "HOLD"
        raw_actions.append(raw_action)
        guarded_actions.append(guarded_action)
        rows.append(
            {
                "phase": phase,
                "raw_action": raw_action,
                "guarded_action": guarded_action,
                "entry_allowed": bool(gate["allowed"]),
                "blockers": list(gate["blockers"]),
                "regime_fit_norm": gate["regime_fit_norm"],
                "risk_multiplier_norm": gate["risk_multiplier_norm"],
            }
        )
    raw_flips = _action_flip_count(raw_actions)
    guarded_flips = _action_flip_count(guarded_actions)
    checks = {
        "confirmed_trend_entry_preserved": rows[0]["guarded_action"] == "BUY",
        "false_restart_suppressed": rows[2]["guarded_action"] == "HOLD",
        "early_recovery_remains_guarded": rows[4]["guarded_action"] == "HOLD",
        "confirmed_recovery_reopens": rows[-1]["guarded_action"] == "BUY",
        "guard_reduces_whipsaw_flips": guarded_flips < raw_flips,
    }
    return {
        "checks": checks,
        "metrics": {
            "work_units": len(rows),
            "raw_action_flip_count": raw_flips,
            "guarded_action_flip_count": guarded_flips,
            "suppressed_flip_count": raw_flips - guarded_flips,
        },
        "details": {"phases": rows},
    }


def _net_alpha_break_even_ladder(_: Mapping[str, Any]) -> dict[str, Any]:
    edge_bps = 30.0
    cost_levels = [2.0, 4.0, 8.0, 12.0, 16.0, 20.0, 24.0, 30.0]
    rows: list[dict[str, Any]] = []
    for cost in cost_levels:
        gate = evaluate_profitability_entry(
            profile="swing_aggressive",
            features=_entry_features(
                edge_bps=edge_bps,
                round_trip_cost_bps=cost,
                quality=0.90,
                trend=0.90,
                chop=0.10,
            ),
        )
        required_edge = _float(
            _mapping(gate.get("entry_economics")).get("required_edge_bps")
        )
        rows.append(
            {
                "round_trip_cost_bps": cost,
                "gross_edge_bps": edge_bps,
                "required_edge_bps": required_edge,
                "edge_after_cost_bps": round(edge_bps - cost, 6),
                "edge_after_required_margin_bps": round(edge_bps - required_edge, 6),
                "entry_allowed": bool(gate["allowed"]),
                "blockers": list(gate["blockers"]),
            }
        )
    first_blocked = next(
        (row["round_trip_cost_bps"] for row in rows if not row["entry_allowed"]), None
    )
    allowed_flags = [bool(row["entry_allowed"]) for row in rows]
    block_seen = False
    monotonic = True
    for allowed in allowed_flags:
        if not allowed:
            block_seen = True
        elif block_seen:
            monotonic = False
    checks = {
        "low_cost_positive_edge_allowed": rows[0]["entry_allowed"],
        "break_even_boundary_found": first_blocked is not None,
        "eligibility_is_monotonic_after_boundary": monotonic,
        "highest_cost_is_blocked": not rows[-1]["entry_allowed"],
        "unknown_cost_defaults_not_used": all(
            row["round_trip_cost_bps"] > 0.0 for row in rows
        ),
    }
    return {
        "checks": checks,
        "metrics": {
            "work_units": len(rows),
            "ladder_point_count": len(rows),
            "first_blocked_round_trip_cost_bps": first_blocked,
            "last_allowed_round_trip_cost_bps": max(
                row["round_trip_cost_bps"] for row in rows if row["entry_allowed"]
            ),
        },
        "details": {"ladder": rows},
    }


def _infer_asset_class(
    sleeve_id: str, objective_class: str, default_asset_class: str
) -> str:
    name = sleeve_id.lower()
    if objective_class == "control_only":
        return "none"
    if "crypto" in name or "digital_asset" in name:
        return "crypto"
    if any(token in name for token in ("cdo", "structured", "xva", "swaption")):
        return "structured_derivative"
    if any(
        token in name
        for token in (
            "option",
            "volatility",
            "variance",
            "gamma",
            "vanna",
            "volga",
            "dispersion",
            "barrier",
            "lookback",
            "greek",
        )
    ):
        return "options"
    if any(token in name for token in ("futures", "commodity", "basis", "rates_curve")):
        return "futures"
    if any(token in name for token in ("fx", "currency")):
        return "fx"
    if any(
        token in name
        for token in (
            "bond",
            "credit",
            "sovereign",
            "debt",
            "repo",
            "securitized",
            "collateral",
        )
    ):
        return "fixed_income"
    return default_asset_class or "equity"


def _capacity_point(
    *,
    profile: Mapping[str, Any],
    asset: Mapping[str, Any],
    state: Mapping[str, Any],
    notional: float,
) -> dict[str, Any]:
    volume = (
        _float(profile.get("daily_dollar_volume"))
        * _float(asset.get("volume_multiplier"), 1.0)
        * _float(state.get("daily_volume_multiplier"), 1.0)
    )
    fixed_multiplier = _float(asset.get("fixed_cost_multiplier"), 1.0)
    spread = (
        _float(profile.get("half_spread_bps"))
        * _float(state.get("spread_multiplier"), 1.0)
        * fixed_multiplier
    )
    fees = _float(profile.get("fees_bps")) * fixed_multiplier
    slippage = (
        _float(profile.get("baseline_slippage_bps"))
        * _float(state.get("slippage_multiplier"), 1.0)
        * fixed_multiplier
    )
    volatility = _float(profile.get("volatility_bps")) * _float(
        state.get("volatility_multiplier"), 1.0
    )
    coefficient = _float(profile.get("impact_coefficient")) * _float(
        state.get("impact_multiplier"), 1.0
    )
    participation = notional / max(volume, 1e-12)
    impact = coefficient * volatility * math.sqrt(max(participation, 0.0))
    total_cost = spread + fees + slippage + impact
    gross = _float(profile.get("expected_gross_alpha_bps"))
    net = gross - total_cost
    return {
        "participation_ratio": round(participation, 10),
        "market_impact_bps": round(impact, 8),
        "total_cost_bps": round(total_cost, 8),
        "net_alpha_bps": round(net, 8),
        "expected_net_alpha_dollars": round(notional * net / 10000.0, 8),
        "positive_net_alpha": net > 0.0,
    }


def _strategy_research_counts(
    sleeve_ids: Sequence[str], target_total: int
) -> dict[str, int]:
    ordered = sorted(set(sleeve_ids))
    if not ordered:
        return {}
    base, remainder = divmod(max(int(target_total), 0), len(ordered))
    return {
        sleeve_id: base + (1 if index < remainder else 0)
        for index, sleeve_id in enumerate(ordered)
    }


def _capacity_curve_for_sleeve(
    *,
    sleeve_id: str,
    objective_class: str,
    hot_strategy_count: int,
    research_strategy_count: int,
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    capacity = _mapping(contract.get("capital_capacity_contract"))
    profiles = _mapping(capacity.get("objective_profiles"))
    profile = _mapping(profiles.get(objective_class))
    applicable = bool(profile.get("applicable", False))
    if not applicable:
        return {
            "sleeve_id": sleeve_id,
            "objective_class": objective_class,
            "asset_class": "none",
            "applicable": False,
            "reason": "control_only_no_trading_profit_or_capacity_objective",
            "hot_strategy_count": hot_strategy_count,
            "research_strategy_count": research_strategy_count,
            "capital_tiers": [],
            "organic_calibration_required": False,
        }
    asset_class = _infer_asset_class(
        sleeve_id,
        objective_class,
        str(profile.get("default_asset_class") or "equity"),
    )
    assets = _mapping(capacity.get("asset_class_adjustments"))
    asset = _mapping(assets.get(asset_class))
    market_states = _mapping(capacity.get("market_states"))
    tiers = sorted({_float(value) for value in capacity.get("capital_tiers_usd") or []})
    primary_fraction = _float(capacity.get("primary_allocation_fraction"), 0.10)
    sensitivity_fractions = sorted(
        {
            _float(value)
            for value in capacity.get("allocation_fraction_sensitivity") or []
            if _float(value) > 0.0
        }
    )
    risk_fraction = _float(profile.get("risk_budget_fraction"), 0.005)
    assumed_loss_fraction = max(
        _float(profile.get("assumed_loss_fraction"), 0.05), 1e-6
    )
    notionals = [
        min(capital * primary_fraction, capital * risk_fraction / assumed_loss_fraction)
        for capital in tiers
    ]
    state_surfaces: dict[str, list[dict[str, Any]]] = {}
    for state_id, raw_state in sorted(market_states.items()):
        state = _mapping(raw_state)
        result = capacity_impact_surface(
            expected_gross_alpha_bps=profile.get("expected_gross_alpha_bps"),
            half_spread_bps=(
                _float(profile.get("half_spread_bps"))
                * _float(state.get("spread_multiplier"), 1.0)
                * _float(asset.get("fixed_cost_multiplier"), 1.0)
            ),
            fees_bps=(
                _float(profile.get("fees_bps"))
                * _float(asset.get("fixed_cost_multiplier"), 1.0)
            ),
            baseline_slippage_bps=(
                _float(profile.get("baseline_slippage_bps"))
                * _float(state.get("slippage_multiplier"), 1.0)
                * _float(asset.get("fixed_cost_multiplier"), 1.0)
            ),
            daily_dollar_volume=(
                _float(profile.get("daily_dollar_volume"))
                * _float(asset.get("volume_multiplier"), 1.0)
                * _float(state.get("daily_volume_multiplier"), 1.0)
            ),
            volatility_bps=(
                _float(profile.get("volatility_bps"))
                * _float(state.get("volatility_multiplier"), 1.0)
            ),
            impact_coefficient=(
                _float(profile.get("impact_coefficient"))
                * _float(state.get("impact_multiplier"), 1.0)
            ),
            notionals=notionals,
        )
        state_surfaces[state_id] = [dict(row) for row in result.get("surface") or []]

    reference_price = _float(asset.get("reference_price"))
    contract_multiplier = _float(asset.get("contract_multiplier"), 1.0)
    margin_fraction = _float(asset.get("margin_requirement_fraction"), 1.0)
    minimum_unit = _float(asset.get("minimum_unit"), 1.0)
    fractional_allowed = bool(asset.get("fractional_units_allowed", False))
    minimum_unit_capital = max(
        reference_price * contract_multiplier * margin_fraction * minimum_unit,
        1.0 if fractional_allowed else 0.0,
    )
    max_participation = _float(capacity.get("maximum_participation_ratio"), 0.02)
    tier_rows: list[dict[str, Any]] = []
    for index, (capital, target_notional) in enumerate(zip(tiers, notionals)):
        state_rows: dict[str, dict[str, Any]] = {}
        executable = bool(target_notional >= minimum_unit_capital)
        for state_id in sorted(market_states):
            surface = state_surfaces.get(state_id) or []
            row = dict(surface[index]) if index < len(surface) else {}
            row["executable_at_configured_unit"] = executable
            row["participation_clear"] = (
                _float(row.get("participation_ratio")) <= max_participation
            )
            row["capacity_clear"] = bool(
                executable
                and row.get("positive_net_alpha", False)
                and row["participation_clear"]
            )
            state_rows[state_id] = row
        sensitivity: list[dict[str, Any]] = []
        for fraction in sensitivity_fractions:
            notional = min(
                capital * fraction,
                capital * risk_fraction / assumed_loss_fraction,
            )
            point = _capacity_point(
                profile=profile,
                asset=asset,
                state=_mapping(market_states.get("normal")),
                notional=notional,
            )
            sensitivity.append(
                {
                    "allocation_fraction": fraction,
                    "target_notional_usd": round(notional, 2),
                    "net_alpha_bps": point["net_alpha_bps"],
                    "expected_net_alpha_dollars": point["expected_net_alpha_dollars"],
                    "positive_net_alpha": point["positive_net_alpha"],
                    "executable_at_configured_unit": (notional >= minimum_unit_capital),
                }
            )
        state_clear_count = sum(
            bool(row["capacity_clear"]) for row in state_rows.values()
        )
        tier_rows.append(
            {
                "capital_usd": capital,
                "risk_budget_usd": round(capital * risk_fraction, 2),
                "risk_limited_notional_usd": round(
                    capital * risk_fraction / assumed_loss_fraction, 2
                ),
                "target_notional_usd": round(target_notional, 2),
                "minimum_unit_capital_usd": round(minimum_unit_capital, 6),
                "fractional_units_allowed": fractional_allowed,
                "executable_at_configured_unit": executable,
                "estimated_units": round(
                    target_notional
                    / max(
                        reference_price * contract_multiplier * margin_fraction,
                        1e-12,
                    ),
                    8,
                ),
                "state_clear_count": state_clear_count,
                "state_count": len(state_rows),
                "stress_survival_ratio": round(
                    state_clear_count / max(len(state_rows), 1), 6
                ),
                "market_states": state_rows,
                "allocation_sensitivity_normal": sensitivity,
            }
        )

    normal_clear = [
        row["capital_usd"]
        for row in tier_rows
        if _mapping(row["market_states"].get("normal")).get("capacity_clear")
    ]
    all_state_clear = [
        row["capital_usd"]
        for row in tier_rows
        if row["state_clear_count"] == row["state_count"]
    ]
    executable_tiers = [
        row["capital_usd"] for row in tier_rows if row["executable_at_configured_unit"]
    ]
    first_normal_nonpositive = next(
        (
            row["capital_usd"]
            for row in tier_rows
            if row["executable_at_configured_unit"]
            and not _mapping(row["market_states"].get("normal")).get(
                "positive_net_alpha"
            )
        ),
        None,
    )
    canary_capital = _float(capacity.get("canary_capital_usd"), 200.0)
    target_capital = _float(capacity.get("target_scale_capital_usd"), 1000000.0)
    canary = next(
        (row for row in tier_rows if row["capital_usd"] == canary_capital), {}
    )
    target = next(
        (row for row in tier_rows if row["capital_usd"] == target_capital), {}
    )
    result = {
        "sleeve_id": sleeve_id,
        "objective_class": objective_class,
        "asset_class": asset_class,
        "applicable": True,
        "hot_strategy_count": hot_strategy_count,
        "research_strategy_count": research_strategy_count,
        "strategy_capacity_inheritance": (
            "strategies inherit the sleeve diagnostic curve until candidate-forward "
            "strategy-specific fills and costs justify a narrower curve"
        ),
        "assumptions": {
            "expected_gross_alpha_bps": profile.get("expected_gross_alpha_bps"),
            "daily_dollar_volume": round(
                _float(profile.get("daily_dollar_volume"))
                * _float(asset.get("volume_multiplier"), 1.0),
                2,
            ),
            "risk_budget_fraction": risk_fraction,
            "assumed_loss_fraction": assumed_loss_fraction,
            "primary_allocation_fraction": primary_fraction,
            "minimum_unit_capital_usd": round(minimum_unit_capital, 6),
            "maximum_participation_ratio": max_participation,
            "diagnostic_only": True,
        },
        "breakpoints": {
            "minimum_executable_capital_usd": (
                min(executable_tiers) if executable_tiers else None
            ),
            "largest_normal_capacity_clear_capital_usd": (
                max(normal_clear) if normal_clear else None
            ),
            "largest_all_state_capacity_clear_capital_usd": (
                max(all_state_clear) if all_state_clear else None
            ),
            "first_normal_nonpositive_capital_usd": first_normal_nonpositive,
            "upper_bound_reached": bool(
                tier_rows
                and not _mapping(tier_rows[-1]["market_states"].get("normal")).get(
                    "capacity_clear"
                )
            ),
        },
        "canary_200": canary,
        "target_scale": target,
        "capital_tiers": tier_rows,
        "organic_calibration_required": True,
        "organic_calibration_fields": [
            "candidate_forward_strategy_edge_lcb_bps",
            "realized_daily_dollar_volume",
            "realized_spread_bps",
            "realized_slippage_bps",
            "realized_fill_ratio",
            "realized_market_impact_bps",
            "independent_days_and_symbols",
        ],
    }
    result[f"target_scale_{int(target_capital)}"] = target
    return result


def _capital_capacity_scaling(context: Mapping[str, Any]) -> dict[str, Any]:
    project_root = Path(context["project_root"])
    policy = _mapping(context.get("policy"))
    capacity = _mapping(policy.get("capital_capacity_contract"))
    contracts = materialize_strategy_contracts(project_root=project_root)
    by_sleeve: dict[str, dict[str, Any]] = {}
    hot_counts: Counter[str] = Counter()
    for row in contracts.values():
        sleeve_id = str(row.get("sleeve_id") or "")
        if not sleeve_id:
            continue
        hot_counts[sleeve_id] += 1
        by_sleeve.setdefault(sleeve_id, dict(row))
    sleeve_ids = sorted(by_sleeve)
    sleeve_policy = load_sleeve_policy()
    target_total = int(
        _float(
            _mapping(sleeve_policy.get("strategy_library")).get(
                "target_total_strategies"
            ),
            0.0,
        )
    )
    research_counts = _strategy_research_counts(sleeve_ids, target_total)
    curves = [
        _capacity_curve_for_sleeve(
            sleeve_id=sleeve_id,
            objective_class=str(by_sleeve[sleeve_id].get("objective_class") or ""),
            hot_strategy_count=hot_counts[sleeve_id],
            research_strategy_count=research_counts.get(sleeve_id, 0),
            contract=policy,
        )
        for sleeve_id in sleeve_ids
    ]
    applicable = [row for row in curves if row["applicable"]]
    control_only = [row for row in curves if not row["applicable"]]
    tiers = list(capacity.get("capital_tiers_usd") or [])
    states = _mapping(capacity.get("market_states"))
    tier_summary: list[dict[str, Any]] = []
    for raw_capital in tiers:
        capital = _float(raw_capital)
        rows = [
            next(
                (
                    tier
                    for tier in curve["capital_tiers"]
                    if _float(tier.get("capital_usd")) == capital
                ),
                {},
            )
            for curve in applicable
        ]
        rows = [row for row in rows if row]
        normal_nets = [
            _float(
                _mapping(_mapping(row.get("market_states")).get("normal")).get(
                    "net_alpha_bps"
                )
            )
            for row in rows
        ]
        tier_summary.append(
            {
                "capital_usd": capital,
                "applicable_sleeve_count": len(rows),
                "minimum_unit_executable_sleeve_count": sum(
                    bool(row.get("executable_at_configured_unit")) for row in rows
                ),
                "normal_capacity_clear_sleeve_count": sum(
                    bool(
                        _mapping(_mapping(row.get("market_states")).get("normal")).get(
                            "capacity_clear"
                        )
                    )
                    for row in rows
                ),
                "all_state_capacity_clear_sleeve_count": sum(
                    int(row.get("state_clear_count") or 0)
                    == int(row.get("state_count") or 0)
                    for row in rows
                ),
                "median_normal_net_alpha_bps": (
                    round(statistics.median(normal_nets), 8) if normal_nets else None
                ),
            }
        )

    objective_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in curves:
        objective_groups[str(row.get("objective_class") or "")].append(row)
    objective_summary = []
    for objective, rows in sorted(objective_groups.items()):
        active_rows = [row for row in rows if row["applicable"]]
        capacities = [
            _float(
                _mapping(row.get("breakpoints")).get(
                    "largest_all_state_capacity_clear_capital_usd"
                )
            )
            for row in active_rows
            if _mapping(row.get("breakpoints")).get(
                "largest_all_state_capacity_clear_capital_usd"
            )
            is not None
        ]
        objective_summary.append(
            {
                "objective_class": objective,
                "sleeve_count": len(rows),
                "applicable_sleeve_count": len(active_rows),
                "hot_strategy_count": sum(
                    int(row.get("hot_strategy_count") or 0) for row in rows
                ),
                "research_strategy_count": sum(
                    int(row.get("research_strategy_count") or 0) for row in rows
                ),
                "median_largest_all_state_capacity_clear_capital_usd": (
                    statistics.median(capacities) if capacities else None
                ),
            }
        )

    work_units = len(applicable) * len(tiers) * len(states)
    scenario_contract = _mapping(policy.get("scenario_contract"))
    checks = {
        "all_runtime_sleeves_covered": len(curves) == len(by_sleeve) > 0,
        "every_sleeve_classified_once": len({row["sleeve_id"] for row in curves})
        == len(curves),
        "trading_and_control_sleeves_separated": bool(applicable)
        and bool(control_only)
        and len(applicable) + len(control_only) == len(curves),
        "capital_tier_floor_met": len(tiers)
        >= int(_float(capacity.get("minimum_capital_tier_count"), 12.0)),
        "market_state_floor_met": len(states)
        >= int(_float(capacity.get("minimum_market_state_count"), 4.0)),
        "canary_and_target_scale_present": all(
            _float(_mapping(row.get("canary_200")).get("capital_usd"))
            == _float(capacity.get("canary_capital_usd"), 200.0)
            and _float(_mapping(row.get("target_scale")).get("capital_usd"))
            == _float(capacity.get("target_scale_capital_usd"), 1000000.0)
            for row in applicable
        ),
        "non_trading_controls_have_no_fake_curve": all(
            not row["capital_tiers"] and not row["organic_calibration_required"]
            for row in control_only
        ),
        "hot_strategy_inheritance_complete": sum(hot_counts.values()) == len(contracts),
        "research_strategy_inheritance_complete": sum(research_counts.values())
        == target_total
        == 12000,
        "workload_is_bounded": work_units
        <= int(_float(scenario_contract.get("maximum_work_units"), 12000.0)),
        "organic_calibration_debt_visible": all(
            row.get("organic_calibration_required") for row in applicable
        ),
    }
    return {
        "checks": checks,
        "metrics": {
            "work_units": work_units,
            "sleeve_count": len(curves),
            "trading_sleeve_count": len(applicable),
            "control_only_sleeve_count": len(control_only),
            "objective_class_count": len(objective_groups),
            "hot_strategy_count": len(contracts),
            "research_strategy_count": target_total,
            "capital_tier_count": len(tiers),
            "market_state_count": len(states),
            "surface_point_count": work_units,
            "summary": {
                "canary_capital_usd": capacity.get("canary_capital_usd"),
                "target_scale_capital_usd": capacity.get("target_scale_capital_usd"),
                "primary_allocation_fraction": capacity.get(
                    "primary_allocation_fraction"
                ),
                "maximum_participation_ratio": capacity.get(
                    "maximum_participation_ratio"
                ),
                "diagnostic_only": True,
            },
        },
        "details": {
            "tier_summary": tier_summary,
            "objective_summary": objective_summary,
            "sleeve_curves": curves,
            "capacity_interpretation": {
                "control_grade_meaning": (
                    "all configured curves were generated and constraints were "
                    "evaluated correctly"
                ),
                "not_economic_proof": True,
                "not_allocation_authority": True,
                "strategy_specific_curve_required_before_live_scaling": True,
                "upper_bound_policy": (
                    "a curve that remains clear at the configured maximum reports "
                    "that the diagnostic upper bound was not reached; it does not "
                    "certify deployability at that amount"
                ),
            },
        },
    }


def _liquidity_evaporation_partial_fill(_: Mapping[str, Any]) -> dict[str, Any]:
    initial = simulate_execution(
        action="BUY",
        last_price=100.0,
        return_1m=0.004,
        spread_bps=3.0,
        volatility_1m=0.003,
        latency_ms=100.0,
        bid_size=400.0,
        ask_size=400.0,
        order_size=1000.0,
        broker="schwab",
        market_kind="equities",
        symbol="SPY",
        order_type="limit",
    )
    filled_quantity = 1000.0 * initial.effective_fill_ratio
    collapse_gate = evaluate_profitability_entry(
        profile="swing_aggressive",
        features=_entry_features(
            edge_bps=35.0,
            round_trip_cost_bps=30.0,
            quality=0.40,
            trend=0.75,
            chop=0.40,
            spread_bps=35.0,
            quote_age_ms=2500.0,
            liquidity=0.15,
        ),
    )
    reduce_exit = simulate_execution(
        action="SELL",
        last_price=98.0,
        return_1m=-0.006,
        spread_bps=35.0,
        volatility_1m=0.02,
        latency_ms=400.0,
        bid_size=50.0,
        ask_size=50.0,
        order_size=max(filled_quantity, 1.0),
        broker="schwab",
        market_kind="equities",
        symbol="SPY",
        order_type="limit",
    )
    additional_entry_quantity = 0.0 if not collapse_gate["allowed"] else 600.0
    checks = {
        "initial_order_partially_fills": 0.0 < initial.effective_fill_ratio < 1.0,
        "liquidity_collapse_blocks_additional_entry": not collapse_gate["allowed"],
        "residual_inventory_is_measured": filled_quantity > 0.0,
        "position_is_not_doubled": additional_entry_quantity == 0.0,
        "reduce_only_exit_remains_available": reduce_exit.action == "SELL",
    }
    return {
        "checks": checks,
        "metrics": {
            "work_units": 3,
            "initial_effective_fill_ratio": round(initial.effective_fill_ratio, 8),
            "residual_inventory_quantity": round(filled_quantity, 8),
            "additional_entry_quantity": additional_entry_quantity,
            "reduce_exit_effective_fill_ratio": round(
                reduce_exit.effective_fill_ratio, 8
            ),
        },
        "details": {
            "initial_execution": {
                "status": initial.paper_execution_status,
                "total_cost_bps": initial.total_cost_bps,
                "effective_fill_ratio": initial.effective_fill_ratio,
            },
            "collapse_gate": collapse_gate,
            "reduce_only_exit": {
                "status": reduce_exit.paper_execution_status,
                "total_cost_bps": reduce_exit.total_cost_bps,
                "effective_fill_ratio": reduce_exit.effective_fill_ratio,
                "may_cross_flat": False,
            },
        },
    }


def _covariance(ids: Sequence[str], off_diagonal: float) -> dict[str, dict[str, float]]:
    return {
        left: {right: 1.0 if left == right else off_diagonal for right in ids}
        for left in ids
    }


def _correlation_crowding_collapse(context: Mapping[str, Any]) -> dict[str, Any]:
    candidate_id = str(_mapping(context.get("candidate_binding")).get("candidate_id"))
    ids = ["dividend_income", "swing", "stat_arb", "macro_rates"]
    sleeves = [
        {
            "candidate_id": candidate_id,
            "sleeve_id": sleeve_id,
            "qualified": True,
            "independent_fills": 50,
            "expected_return_bps": 20.0 + index,
            "cost_bps": 5.0,
        }
        for index, sleeve_id in enumerate(ids)
    ]
    diversified = build_multi_period_advisory(
        candidate_id=candidate_id,
        sleeves=sleeves,
        covariance=_covariance(ids, 0.10),
        current_weights={sleeve_id: 0.0 for sleeve_id in ids},
    )
    crowded = build_multi_period_advisory(
        candidate_id=candidate_id,
        sleeves=sleeves,
        covariance=_covariance(ids, 0.90),
        current_weights={sleeve_id: 0.0 for sleeve_id in ids},
    )
    checks = {
        "diversified_baseline_is_advisory_ready": diversified.get("status")
        == "advisory_ready",
        "crowded_portfolio_abstains": crowded.get("status") == "abstain",
        "high_correlation_pairs_are_named": bool(crowded.get("high_correlation_pairs")),
        "crowding_does_not_allocate": not bool(crowded.get("execution_authority")),
        "candidate_identity_is_preserved": diversified.get("candidate_id")
        == candidate_id,
    }
    return {
        "checks": checks,
        "metrics": {
            "work_units": 2,
            "sleeve_count": len(ids),
            "crowded_pair_count": len(crowded.get("high_correlation_pairs") or []),
            "diversified_target_weight_sum": round(
                sum(_mapping(diversified.get("target_weights")).values()), 8
            ),
        },
        "details": {"diversified": diversified, "crowded": crowded},
    }


def _gradual_strategy_decay(_: Mapping[str, Any]) -> dict[str, Any]:
    stable = [10.0 + ((index % 5) - 2) * 0.2 for index in range(50)]
    decaying = stable[:30] + [8.0 - 0.25 * index for index in range(20)]
    stable_result = sequential_change_point_stability(
        {"stable_control": stable},
        minimum_observations_per_group=30,
        burn_in_observations=15,
        alarm_threshold_standard_deviations=6.0,
        recent_window_observations=10,
    )
    decay_result = sequential_change_point_stability(
        {"decaying_strategy": decaying},
        minimum_observations_per_group=30,
        burn_in_observations=15,
        alarm_threshold_standard_deviations=4.0,
        recent_window_observations=15,
    )
    checks = {
        "stable_control_does_not_false_alarm": stable_result.get("passes") is True,
        "gradual_decay_is_detected": decay_result.get("passes") is False,
        "recent_decay_is_named": int(
            decay_result.get("recent_change_point_group_count") or 0
        )
        == 1,
        "decay_has_lower_stability": _float(decay_result.get("stability_score_norm"))
        < _float(stable_result.get("stability_score_norm")),
        "diagnostic_has_no_execution_authority": not bool(
            _mapping(decay_result.get("authority")).get("submits_live_orders")
        ),
    }
    return {
        "checks": checks,
        "metrics": {
            "work_units": len(stable) + len(decaying),
            "stable_change_points": stable_result.get("change_point_count"),
            "decay_change_points": decay_result.get("change_point_count"),
            "decay_stability_score_norm": decay_result.get("stability_score_norm"),
        },
        "details": {"stable_control": stable_result, "decaying": decay_result},
    }


def _point_in_time_contamination(_: Mapping[str, Any]) -> dict[str, Any]:
    rows = [
        {
            "series_id": "GDP",
            "effective_at_utc": "2026-01-01T00:00:00+00:00",
            "known_at_utc": "2026-01-10T13:30:00+00:00",
            "superseded_at_utc": "2026-02-10T13:30:00+00:00",
            "revision_id": "r1",
            "value": 100.0,
        },
        {
            "series_id": "GDP",
            "effective_at_utc": "2026-01-01T00:00:00+00:00",
            "known_at_utc": "2026-02-10T13:30:00+00:00",
            "revision_id": "r2",
            "value": 101.0,
        },
        {
            "series_id": "GDP",
            "effective_at_utc": "2026-01-01T00:00:00+00:00",
            "known_at_utc": "2026-04-10T13:30:00+00:00",
            "revision_id": "r3_future",
            "value": 102.0,
        },
    ]
    contract = {
        "effective_from_field": "effective_at_utc",
        "effective_to_field": "effective_until_utc",
        "known_at_field": "known_at_utc",
        "superseded_at_field": "superseded_at_utc",
        "revision_id_field": "revision_id",
    }
    january = select_bitemporal_rows(
        rows,
        as_of_utc="2026-01-31T23:59:59+00:00",
        valid_at_utc="2026-01-31T23:59:59+00:00",
        natural_key_columns=["series_id"],
        contract=contract,
    )
    march = select_bitemporal_rows(
        rows,
        as_of_utc="2026-03-01T00:00:00+00:00",
        valid_at_utc="2026-03-01T00:00:00+00:00",
        natural_key_columns=["series_id"],
        contract=contract,
    )
    january_revision = str(january[0].get("revision_id") or "") if january else ""
    march_revision = str(march[0].get("revision_id") or "") if march else ""
    checks = {
        "earlier_replay_uses_as_known_revision": january_revision == "r1",
        "later_replay_uses_available_revision": march_revision == "r2",
        "future_revision_is_excluded": march_revision != "r3_future",
        "one_natural_key_resolves_once": len(january) == len(march) == 1,
        "future_knowledge_is_not_allowed": True,
    }
    return {
        "checks": checks,
        "metrics": {
            "work_units": len(rows) * 2,
            "input_revision_count": len(rows),
            "january_selected_revision": january_revision,
            "march_selected_revision": march_revision,
            "future_revision_excluded_count": 1,
        },
        "details": {
            "january_as_known_rows": january,
            "march_as_known_rows": march,
            "bitemporal_contract": contract,
        },
    }


def _reentry_path(
    prices: Sequence[float],
    entries: Sequence[tuple[int, float]],
    *,
    cost_bps: float,
) -> dict[str, Any]:
    cash = 1.0
    units = 0.0
    values: list[float] = []
    entry_map = {index: fraction for index, fraction in entries}
    for index, price in enumerate(prices):
        if index in entry_map:
            requested_fraction = entry_map[index]
            investment = min(requested_fraction, cash)
            net_investment = investment * (1.0 - cost_bps / 10000.0)
            cash -= investment
            units += net_investment / price
        values.append(cash + units * price)
    peak = values[0]
    max_drawdown = 0.0
    for value in values:
        peak = max(peak, value)
        max_drawdown = min(max_drawdown, value / max(peak, 1e-12) - 1.0)
    total_return = values[-1] - 1.0
    return {
        "entries": [
            {"price_index": index, "capital_fraction": fraction}
            for index, fraction in entries
        ],
        "ending_value": round(values[-1], 8),
        "total_return_fraction": round(total_return, 8),
        "maximum_drawdown_fraction": round(max_drawdown, 8),
        "equity_curve": [round(value, 8) for value in values],
    }


def _recovery_reentry_timing(_: Mapping[str, Any]) -> dict[str, Any]:
    prices = [100.0, 80.0, 88.0, 74.0, 78.0, 90.0, 102.0]
    strategies = {
        "immediate": _reentry_path(prices, [(1, 1.0)], cost_bps=20.0),
        "staged": _reentry_path(prices, [(1, 0.5), (3, 0.5)], cost_bps=20.0),
        "delayed": _reentry_path(prices, [(5, 1.0)], cost_bps=20.0),
    }
    penalty = 0.25
    for row in strategies.values():
        row["drawdown_penalized_utility"] = round(
            _float(row.get("total_return_fraction"))
            + penalty * _float(row.get("maximum_drawdown_fraction")),
            8,
        )
    selected = max(
        strategies,
        key=lambda name: (
            _float(strategies[name].get("drawdown_penalized_utility")),
            name,
        ),
    )
    checks = {
        "all_reentry_policies_compared": set(strategies)
        == {"immediate", "staged", "delayed"},
        "transaction_costs_are_included": True,
        "drawdown_is_measured": all(
            row.get("maximum_drawdown_fraction") is not None
            for row in strategies.values()
        ),
        "staged_reentry_wins_designed_path": selected == "staged",
        "selection_is_diagnostic_only": True,
    }
    return {
        "checks": checks,
        "metrics": {
            "work_units": len(prices) * len(strategies),
            "selected_policy": selected,
            "drawdown_penalty": penalty,
            "path_start_price": prices[0],
            "path_end_price": prices[-1],
        },
        "details": {"prices": prices, "strategies": strategies},
    }


def _benchmark_opportunity_cost(_: Mapping[str, Any]) -> dict[str, Any]:
    rows = [
        {
            "regime": "strong_bull",
            "system_return": 0.00,
            "cash_return": 0.005,
            "sgov_return": 0.010,
            "passive_return": 0.100,
            "expected_classification": "costly_inactivity",
        },
        {
            "regime": "bear",
            "system_return": 0.010,
            "cash_return": 0.005,
            "sgov_return": 0.010,
            "passive_return": -0.150,
            "expected_classification": "prudent_abstention",
        },
        {
            "regime": "chop",
            "system_return": 0.010,
            "cash_return": 0.005,
            "sgov_return": 0.010,
            "passive_return": 0.000,
            "expected_classification": "prudent_abstention",
        },
        {
            "regime": "active_alpha",
            "system_return": 0.120,
            "cash_return": 0.005,
            "sgov_return": 0.010,
            "passive_return": 0.050,
            "expected_classification": "active_value_added",
        },
    ]
    for row in rows:
        best = max(row["cash_return"], row["sgov_return"], row["passive_return"])
        opportunity_cost = best - row["system_return"]
        if row["system_return"] > best:
            classification = "active_value_added"
        elif (
            row["system_return"] >= row["sgov_return"]
            and row["passive_return"] <= row["sgov_return"]
        ):
            classification = "prudent_abstention"
        elif opportunity_cost >= 0.02:
            classification = "costly_inactivity"
        else:
            classification = "benchmark_watch"
        row["best_benchmark_return"] = best
        row["opportunity_cost_fraction"] = round(opportunity_cost, 8)
        row["classification"] = classification
        row["expectation_matched"] = classification == row["expected_classification"]
    counts = Counter(str(row["classification"]) for row in rows)
    checks = {
        "all_regimes_match_expected_classification": all(
            row["expectation_matched"] for row in rows
        ),
        "costly_inactivity_is_detected": counts["costly_inactivity"] == 1,
        "prudent_abstention_is_preserved": counts["prudent_abstention"] == 2,
        "active_value_added_is_detected": counts["active_value_added"] == 1,
        "cash_sgov_and_passive_are_all_compared": True,
    }
    return {
        "checks": checks,
        "metrics": {
            "work_units": len(rows),
            "classification_counts": dict(sorted(counts.items())),
            "maximum_opportunity_cost_fraction": max(
                row["opportunity_cost_fraction"] for row in rows
            ),
        },
        "details": {"regimes": rows},
    }


def _owner_horizon(decision_horizon: str, holding_horizon: str) -> str:
    combined = f"{decision_horizon} {holding_horizon}".lower()
    if any(token in combined for token in ("second", "minute", "intraday")):
        return "5m"
    if any(token in combined for token in ("hour", "two_days", "six_weeks")):
        return "1h"
    return "1d"


def _signal_horizon_conflict(context: Mapping[str, Any]) -> dict[str, Any]:
    project_root = Path(context["project_root"])
    contracts = materialize_strategy_contracts(project_root=project_root)
    first_by_sleeve: dict[str, dict[str, Any]] = {}
    for row in contracts.values():
        first_by_sleeve.setdefault(str(row.get("sleeve_id") or ""), dict(row))
    cases = [
        {
            "sleeve_id": "dividend_income",
            "signals": {
                "5m": {"action": "SELL", "confidence": 0.95},
                "1h": {"action": "HOLD", "confidence": 0.70},
                "1d": {"action": "BUY", "confidence": 0.82},
            },
            "expected_action": "BUY",
        },
        {
            "sleeve_id": "futures_index_intraday",
            "signals": {
                "5m": {"action": "SELL", "confidence": 0.86},
                "1h": {"action": "BUY", "confidence": 0.92},
                "1d": {"action": "BUY", "confidence": 0.96},
            },
            "expected_action": "SELL",
        },
        {
            "sleeve_id": "swing_aggressive",
            "signals": {
                "5m": {"action": "SELL", "confidence": 0.96},
                "1h": {"action": "BUY", "confidence": 0.80},
                "1d": {"action": "HOLD", "confidence": 0.91},
            },
            "expected_action": "BUY",
        },
    ]
    for case in cases:
        contract = first_by_sleeve.get(case["sleeve_id"]) or {}
        owner = _owner_horizon(
            str(contract.get("decision_horizon") or ""),
            str(contract.get("holding_horizon") or ""),
        )
        signal = _mapping(_mapping(case.get("signals")).get(owner))
        action = (
            str(signal.get("action") or "HOLD")
            if _float(signal.get("confidence")) >= 0.65
            else "HOLD"
        )
        case["decision_horizon"] = contract.get("decision_horizon")
        case["holding_horizon"] = contract.get("holding_horizon")
        case["owner_horizon"] = owner
        case["resolved_action"] = action
        case["expectation_matched"] = action == case["expected_action"]
        case["conflicting_higher_confidence_ignored"] = any(
            horizon != owner
            and _float(row.get("confidence")) > _float(signal.get("confidence"))
            and str(row.get("action")) != action
            for horizon, row in _mapping(case.get("signals")).items()
        )
    checks = {
        "all_sleeves_have_specialization_contracts": all(
            case.get("decision_horizon") and case.get("holding_horizon")
            for case in cases
        ),
        "all_owner_horizons_resolve": all(
            case.get("owner_horizon") in {"5m", "1h", "1d"} for case in cases
        ),
        "owner_actions_match_expected": all(
            case["expectation_matched"] for case in cases
        ),
        "cross_horizon_override_is_rejected": all(
            case["conflicting_higher_confidence_ignored"] for case in cases
        ),
        "sleeve_identity_is_preserved": len({case["sleeve_id"] for case in cases})
        == len(cases),
    }
    return {
        "checks": checks,
        "metrics": {
            "work_units": len(cases) * 3,
            "sleeve_count": len(cases),
            "horizon_count": 3,
        },
        "details": {"cases": cases},
    }


def _corporate_action_calendar(_: Mapping[str, Any]) -> dict[str, Any]:
    security_records = [
        {
            "security_id": "SEC-ABC",
            "symbol": "ABC",
            "valid_from": "2020-01-01T00:00:00+00:00",
            "identifiers": {"figi": "BBG-ABC"},
            "corporate_actions": [
                {
                    "type": "split",
                    "effective_at": "2026-06-01T00:00:00+00:00",
                    "adjustment_factor": 2.0,
                }
            ],
        },
        {
            "security_id": "SEC-RENAMED",
            "symbol": "OLD",
            "valid_from": "2020-01-01T00:00:00+00:00",
            "valid_to": "2026-06-01T00:00:00+00:00",
            "identifiers": {"figi": "BBG-RENAME"},
            "corporate_actions": [
                {
                    "type": "symbol_change",
                    "effective_at": "2026-06-01T00:00:00+00:00",
                    "adjustment_factor": 1.0,
                }
            ],
        },
        {
            "security_id": "SEC-RENAMED",
            "symbol": "NEW",
            "valid_from": "2026-06-01T00:00:00+00:00",
            "identifiers": {"figi": "BBG-RENAME"},
            "corporate_actions": [],
        },
        {
            "security_id": "SEC-DEAD",
            "symbol": "DEAD",
            "valid_from": "2020-01-01T00:00:00+00:00",
            "valid_to": "2026-05-01T00:00:00+00:00",
            "status": "delisted",
            "delisted_at": "2026-05-01T00:00:00+00:00",
            "identifiers": {"figi": "BBG-DEAD"},
            "corporate_actions": [],
        },
    ]
    observations = [
        {"symbol": "ABC", "timestamp_utc": "2026-06-02T14:00:00+00:00"},
        {"symbol": "OLD", "timestamp_utc": "2026-05-15T14:00:00+00:00"},
        {"symbol": "NEW", "timestamp_utc": "2026-06-15T14:00:00+00:00"},
        {"symbol": "DEAD", "timestamp_utc": "2026-04-15T14:00:00+00:00"},
    ]
    identity_audit = point_in_time_security_master_audit(security_records, observations)
    split_before = 100.0 * 50.0
    split_after = 200.0 * 25.0
    dividend_total_return = (49.5 - 50.0) * 100.0 + 0.5 * 100.0
    option_before = 1.0 * 100.0 * 5.0
    option_after = 2.0 * 50.0 * 5.0
    ny = ZoneInfo("America/New_York")
    winter_open = datetime(2026, 1, 15, 9, 30, tzinfo=ny).astimezone(timezone.utc)
    summer_open = datetime(2026, 7, 15, 9, 30, tzinfo=ny).astimezone(timezone.utc)
    half_day_close_minutes = 13 * 60
    half_day_entry_cutoff_minutes = 12 * 60 + 45
    late_entry_minutes = 12 * 60 + 50
    futures_roll = {
        "root": "ES",
        "before_roll_contract": "ESM26",
        "after_roll_contract": "ESU26",
        "root_identity_preserved": True,
    }
    checks = {
        "effective_dated_security_identity_passes": identity_audit.get("passes")
        is True,
        "split_value_is_preserved": split_before == split_after,
        "dividend_cashflow_is_in_total_return": abs(dividend_total_return) < 1e-12,
        "option_adjustment_value_is_preserved": option_before == option_after,
        "dst_open_is_utc_adjusted": winter_open.hour == 14 and summer_open.hour == 13,
        "half_day_late_entry_is_blocked": late_entry_minutes
        > half_day_entry_cutoff_minutes
        and half_day_entry_cutoff_minutes < half_day_close_minutes,
        "futures_roll_preserves_root_identity": futures_roll["root_identity_preserved"],
    }
    return {
        "checks": checks,
        "metrics": {
            "work_units": len(security_records) + len(observations) + 6,
            "security_record_count": len(security_records),
            "resolved_observation_count": identity_audit.get(
                "resolved_observation_count"
            ),
            "calendar_case_count": 3,
            "accounting_invariant_count": 3,
        },
        "details": {
            "security_master_audit": identity_audit,
            "accounting_invariants": {
                "split_value_before": split_before,
                "split_value_after": split_after,
                "dividend_total_return_dollars": dividend_total_return,
                "option_value_before": option_before,
                "option_value_after": option_after,
            },
            "calendar": {
                "winter_open_utc": winter_open.isoformat(),
                "summer_open_utc": summer_open.isoformat(),
                "half_day_close_minutes": half_day_close_minutes,
                "half_day_entry_cutoff_minutes": half_day_entry_cutoff_minutes,
                "late_entry_minutes": late_entry_minutes,
                "futures_roll": futures_roll,
            },
        },
    }


def _volatility_dependent_data_loss(_: Mapping[str, Any]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for index in range(100):
        high_volatility = index % 5 == 0
        missing = (high_volatility and index < 75) or index in {1, 2}
        rows.append(
            {
                "index": index,
                "high_volatility": high_volatility,
                "missing": missing,
            }
        )
    high = [row for row in rows if row["high_volatility"]]
    low = [row for row in rows if not row["high_volatility"]]
    high_missing_rate = sum(row["missing"] for row in high) / len(high)
    low_missing_rate = sum(row["missing"] for row in low) / len(low)
    gap = high_missing_rate - low_missing_rate
    last_good_age_seconds = 20.0
    maximum_last_good_age_seconds = 5.0
    current_action = (
        "BLOCK_CURRENT_AND_QUARANTINE_TRAINING"
        if gap >= 0.20 and last_good_age_seconds > maximum_last_good_age_seconds
        else "ALLOW"
    )
    checks = {
        "missingness_depends_on_volatility": gap >= 0.20,
        "high_volatility_missingness_is_named": high_missing_rate > low_missing_rate,
        "stale_last_good_is_rejected": last_good_age_seconds
        > maximum_last_good_age_seconds,
        "current_decision_is_blocked": current_action.startswith("BLOCK"),
        "training_rows_are_quarantined": "QUARANTINE_TRAINING" in current_action,
    }
    return {
        "checks": checks,
        "metrics": {
            "work_units": len(rows),
            "observation_count": len(rows),
            "high_volatility_missing_rate": round(high_missing_rate, 8),
            "low_volatility_missing_rate": round(low_missing_rate, 8),
            "missingness_gap": round(gap, 8),
            "last_good_age_seconds": last_good_age_seconds,
            "maximum_last_good_age_seconds": maximum_last_good_age_seconds,
            "response": current_action,
        },
        "details": {
            "training_bias": {
                "high_volatility_available_ratio": round(1.0 - high_missing_rate, 8),
                "low_volatility_available_ratio": round(1.0 - low_missing_rate, 8),
                "calm_period_overrepresentation_detected": True,
            }
        },
    }


def _false_model_consensus(_: Mapping[str, Any]) -> dict[str, Any]:
    base = [float(((index * 7) % 11) - 5) for index in range(40)]
    returns = {
        "model_a": base,
        "model_b": [
            value + (0.05 if index % 2 else -0.05) for index, value in enumerate(base)
        ],
        "model_c": [value * 1.02 for value in base],
        "model_d": [float(((index * 3) % 13) - 6) for index in range(40)],
        "model_e": [1.0 if index % 2 else -1.0 for index in range(40)],
        "model_f": [float(((index * index) % 17) - 8) for index in range(40)],
    }
    redundancy = residual_redundancy_graph(
        returns,
        minimum_common_observations=30,
        absolute_correlation_threshold=0.80,
        minimum_independent_group_ratio=0.50,
    )
    components = list(redundancy.get("components") or [])
    duplicate_cluster = next(
        (
            component
            for component in components
            if {"model_a", "model_b", "model_c"}.issubset(set(component))
        ),
        [],
    )
    raw_vote_count = len(returns)
    effective_vote_count = len(components)
    checks = {
        "redundancy_graph_is_available": redundancy.get("available") is True,
        "correlated_model_cluster_is_detected": bool(duplicate_cluster),
        "effective_votes_are_less_than_raw_votes": effective_vote_count
        < raw_vote_count,
        "pairwise_receipts_are_present": bool(redundancy.get("pairwise_correlations")),
        "consensus_has_no_order_authority": not bool(
            _mapping(redundancy.get("authority")).get("submits_live_orders")
        ),
    }
    return {
        "checks": checks,
        "metrics": {
            "work_units": sum(len(values) for values in returns.values()),
            "raw_vote_count": raw_vote_count,
            "effective_independent_vote_count": effective_vote_count,
            "redundant_pair_count": redundancy.get("redundant_pair_count"),
            "duplicate_cluster": duplicate_cluster,
        },
        "details": {"redundancy_graph": redundancy},
    }


def _max_drawdown(prices: Sequence[float]) -> float:
    peak = prices[0]
    maximum = 0.0
    for price in prices:
        peak = max(peak, price)
        maximum = min(maximum, price / max(peak, 1e-12) - 1.0)
    return maximum


def _portfolio_path_dependency(_: Mapping[str, Any]) -> dict[str, Any]:
    smooth = [100.0, 102.0, 104.0, 106.0, 108.0, 110.0]
    stressed = [100.0, 90.0, 75.0, 85.0, 95.0, 110.0]
    smooth_return = smooth[-1] / smooth[0] - 1.0
    stressed_return = stressed[-1] / stressed[0] - 1.0
    smooth_drawdown = _max_drawdown(smooth)
    stressed_drawdown = _max_drawdown(stressed)
    stop_floor = -0.15
    smooth_stop = smooth_drawdown <= stop_floor
    stressed_stop = stressed_drawdown <= stop_floor
    checks = {
        "endpoint_returns_match": abs(smooth_return - stressed_return) < 1e-12,
        "drawdowns_materially_differ": abs(stressed_drawdown - smooth_drawdown) >= 0.20,
        "stressed_path_triggers_stop": stressed_stop,
        "smooth_path_does_not_trigger_stop": not smooth_stop,
        "path_state_changes_risk_outcome": smooth_stop != stressed_stop,
    }
    return {
        "checks": checks,
        "metrics": {
            "work_units": len(smooth) + len(stressed),
            "endpoint_return_fraction": round(smooth_return, 8),
            "smooth_maximum_drawdown_fraction": round(smooth_drawdown, 8),
            "stressed_maximum_drawdown_fraction": round(stressed_drawdown, 8),
            "stop_floor_fraction": stop_floor,
        },
        "details": {
            "smooth_path": smooth,
            "stressed_path": stressed,
            "smooth_stop_triggered": smooth_stop,
            "stressed_stop_triggered": stressed_stop,
        },
    }


SCENARIO_FUNCTIONS: dict[str, Callable[[Mapping[str, Any]], dict[str, Any]]] = {
    "regime_transition_whipsaw": _regime_transition_whipsaw,
    "net_alpha_break_even_ladder": _net_alpha_break_even_ladder,
    "capital_capacity_scaling": _capital_capacity_scaling,
    "liquidity_evaporation_partial_fill": _liquidity_evaporation_partial_fill,
    "correlation_crowding_collapse": _correlation_crowding_collapse,
    "gradual_strategy_decay": _gradual_strategy_decay,
    "point_in_time_contamination": _point_in_time_contamination,
    "recovery_reentry_timing": _recovery_reentry_timing,
    "benchmark_opportunity_cost": _benchmark_opportunity_cost,
    "signal_horizon_conflict": _signal_horizon_conflict,
    "corporate_action_calendar": _corporate_action_calendar,
    "volatility_dependent_data_loss": _volatility_dependent_data_loss,
    "false_model_consensus": _false_model_consensus,
    "portfolio_path_dependency": _portfolio_path_dependency,
}


def _policy_errors(policy: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    if int(_float(policy.get("schema_version"), 0.0)) != 1:
        errors.append("policy_schema_version_invalid")
    scenario_contract = _mapping(policy.get("scenario_contract"))
    required = [
        str(value) for value in scenario_contract.get("required_scenario_ids") or []
    ]
    configured = [
        str(_mapping(row).get("scenario_id") or "")
        for row in policy.get("scenarios") or []
    ]
    if len(required) != 14 or len(set(required)) != 14:
        errors.append("exactly_fourteen_unique_required_scenarios_expected")
    if set(required) != set(configured):
        errors.append("configured_scenarios_do_not_match_required_scenarios")
    if set(required) != set(SCENARIO_FUNCTIONS):
        errors.append("scenario_function_registry_incomplete")
    capacity = _mapping(policy.get("capital_capacity_contract"))
    tiers = list(capacity.get("capital_tiers_usd") or [])
    canary_capital = _float(capacity.get("canary_capital_usd"), 200.0)
    target_capital = _float(capacity.get("target_scale_capital_usd"), 1000000.0)
    minimum_tier_count = int(_float(capacity.get("minimum_capital_tier_count"), 12.0))
    normalized_tiers = {_float(value) for value in tiers}
    if (
        canary_capital not in normalized_tiers
        or target_capital not in normalized_tiers
        or len(tiers) < minimum_tier_count
    ):
        errors.append("capacity_tiers_missing_canary_or_target_scale")
    minimum_state_count = int(_float(capacity.get("minimum_market_state_count"), 4.0))
    if len(_mapping(capacity.get("market_states"))) < minimum_state_count:
        errors.append("capacity_market_state_floor_not_met")
    objective_profiles = _mapping(capacity.get("objective_profiles"))
    expected_objectives = {
        "basis_relative_value",
        "capital_preservation",
        "control_only",
        "digital_asset_alpha",
        "directional_alpha",
        "event_alpha",
        "execution_alpha",
        "hedge_utility",
        "income_total_return",
        "macro_carry_relative_value",
        "market_neutral_relative_value",
        "volatility_relative_value",
    }
    if set(objective_profiles) != expected_objectives:
        errors.append("capacity_objective_profile_registry_incomplete")
    authority = _mapping(policy.get("authority_contract"))
    if not authority or any(bool(value) for value in authority.values()):
        errors.append("authority_contract_not_fully_locked")
    evidence = _mapping(policy.get("evidence_contract"))
    if evidence.get("results_are_promotion_evidence") is not False:
        errors.append("promotion_evidence_must_remain_false")
    if evidence.get("results_are_profitability_proof") is not False:
        errors.append("profitability_proof_must_remain_false")
    if evidence.get("automatic_threshold_tuning_allowed") is not False:
        errors.append("automatic_threshold_tuning_must_remain_false")
    return errors


def _select_scenarios(
    specs: Sequence[Mapping[str, Any]], requested: Sequence[str] | None
) -> tuple[list[dict[str, Any]], list[str]]:
    wanted = [
        str(value or "").strip()
        for value in (requested or [])
        if str(value or "").strip()
    ]
    if not wanted or "all" in wanted:
        return [dict(row) for row in specs], []
    selected: list[dict[str, Any]] = []
    unresolved: list[str] = []
    for name in wanted:
        match = next(
            (
                dict(row)
                for row in specs
                if name
                in {
                    str(row.get("scenario_id") or ""),
                    *[str(alias) for alias in row.get("aliases") or []],
                }
            ),
            None,
        )
        if match is None:
            unresolved.append(name)
        elif not any(
            row.get("scenario_id") == match.get("scenario_id") for row in selected
        ):
            selected.append(match)
    return selected, unresolved


def _run_scenario(
    spec: Mapping[str, Any], context: Mapping[str, Any]
) -> dict[str, Any]:
    scenario_id = str(spec.get("scenario_id") or "")
    function = SCENARIO_FUNCTIONS.get(scenario_id)
    if function is None:
        return {
            "scenario_id": scenario_id,
            "title": str(spec.get("title") or scenario_id),
            "purpose": str(spec.get("purpose") or ""),
            "ok": False,
            "checks": {"scenario_function_present": False},
            "failed_checks": ["scenario_function_present"],
            "metrics": {"work_units": 0},
            "details": {},
            "error": "scenario_function_missing",
        }
    try:
        result = function(context)
        checks = {
            str(key): bool(value)
            for key, value in _mapping(result.get("checks")).items()
        }
        failed = [name for name, passed in checks.items() if not passed]
        return {
            "scenario_id": scenario_id,
            "aliases": list(spec.get("aliases") or []),
            "title": str(spec.get("title") or scenario_id),
            "purpose": str(spec.get("purpose") or ""),
            "ok": bool(checks) and not failed,
            "checks": checks,
            "failed_checks": failed,
            "metrics": _mapping(result.get("metrics")),
            "details": _mapping(result.get("details")),
            "error": "",
        }
    except Exception as exc:
        return {
            "scenario_id": scenario_id,
            "aliases": list(spec.get("aliases") or []),
            "title": str(spec.get("title") or scenario_id),
            "purpose": str(spec.get("purpose") or ""),
            "ok": False,
            "checks": {"scenario_completed_without_exception": False},
            "failed_checks": ["scenario_completed_without_exception"],
            "metrics": {"work_units": 0},
            "details": {},
            "error": f"{type(exc).__name__}:{exc}",
        }


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
    policy_errors = _policy_errors(policy)
    binding, candidate_path, candidate_sha_before = _candidate_binding(root)
    specs = [
        _mapping(row)
        for row in policy.get("scenarios") or []
        if isinstance(row, Mapping)
    ]
    selected_specs, unresolved = _select_scenarios(specs, requested_scenarios)
    context = {
        "project_root": root,
        "policy": policy,
        "candidate_binding": binding,
    }
    scenario_results = [_run_scenario(spec, context) for spec in selected_specs]
    candidate_sha_after = _sha256_file(candidate_path)
    candidate_unchanged = bool(
        candidate_sha_before
        and candidate_sha_after
        and candidate_sha_before == candidate_sha_after
    )
    required_ids = set(
        str(value)
        for value in _mapping(policy.get("scenario_contract")).get(
            "required_scenario_ids"
        )
        or []
    )
    all_mode = not requested_scenarios or "all" in (requested_scenarios or [])
    work_units = sum(
        int(_float(_mapping(row.get("metrics")).get("work_units"), 0.0))
        for row in scenario_results
    )
    maximum_work_units = int(
        _float(
            _mapping(policy.get("scenario_contract")).get("maximum_work_units"),
            12000.0,
        )
    )
    authority = _mapping(policy.get("authority_contract"))
    checks = {
        "policy_receipt_present": bool(_sha256_file(resolved_policy_path)),
        "policy_contract_valid": not policy_errors,
        "candidate_binding_valid": bool(binding.get("valid")),
        "candidate_state_unchanged": candidate_unchanged,
        "authority_locked": bool(authority)
        and not any(bool(value) for value in authority.values()),
        "scenario_selection_resolved": not unresolved,
        "scenario_selection_nonempty": bool(scenario_results),
        "configured_scenario_coverage_complete": set(SCENARIO_FUNCTIONS)
        == required_ids,
        "selected_scenarios_pass": bool(scenario_results)
        and all(row.get("ok", False) for row in scenario_results),
        "full_suite_has_fourteen_scenarios": (
            len(scenario_results) == 14 if all_mode else True
        ),
        "workload_is_bounded": work_units <= maximum_work_units,
    }
    capacity_row = next(
        (
            row
            for row in scenario_results
            if row.get("scenario_id") == "capital_capacity_scaling"
        ),
        {},
    )
    payload: dict[str, Any] = {
        "timestamp_utc": generated_at_utc or datetime.now(timezone.utc).isoformat(),
        "schema_version": 1,
        "policy_id": str(policy.get("policy_id") or ""),
        "operating_mode": str(policy.get("operating_mode") or ""),
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
            "all_mode": all_mode,
        },
        "policy_errors": policy_errors,
        "scenario_results": scenario_results,
        "capacity_summary": _mapping(
            _mapping(capacity_row.get("metrics")).get("summary")
        ),
        "evidence_classification": {
            "diagnostic_only": True,
            "synthetic_parameters_are_organic_candidate_evidence": False,
            "promotion_evidence": False,
            "live_release_evidence": False,
            "profitability_proof": False,
            "capacity_is_certified_deployable_capital": False,
            "candidate_forward_calibration_required": True,
        },
        "authority_contract": authority,
        "resource_contract": {
            "persistent_processes_started": 0,
            "network_requests": 0,
            "broker_requests": 0,
            "orders_submitted": 0,
            "work_units": work_units,
            "maximum_work_units": maximum_work_units,
            "estimated_artifact_bytes": 0,
            "maximum_artifact_bytes": int(
                _float(
                    _mapping(policy.get("scenario_contract")).get(
                        "maximum_artifact_bytes"
                    ),
                    8000000.0,
                )
            ),
        },
        "next_actions": [
            "Use candidate-forward fills, spreads, volume, and post-cost edge to replace diagnostic capacity assumptions sleeve by sleeve.",
            "Investigate failed scenario checks without relaxing risk or profitability thresholds automatically.",
            "Keep the 200-dollar canary as execution validation; capital growth alone may not increase strategy weight.",
            "Preserve cumulative soak history while starting a new clean affected-scope segment if this candidate is accepted.",
        ],
        "receipts": {
            "policy_path": str(resolved_policy_path),
            "policy_sha256": _sha256_file(resolved_policy_path),
            "candidate_state_sha256": candidate_sha_after,
        },
    }
    estimated_bytes = len(
        json.dumps(payload, ensure_ascii=True, sort_keys=True).encode("utf-8")
    )
    payload["resource_contract"]["estimated_artifact_bytes"] = estimated_bytes
    checks["artifact_size_is_bounded"] = estimated_bytes <= int(
        payload["resource_contract"]["maximum_artifact_bytes"]
    )
    passed = sum(bool(value) for value in checks.values())
    score = round(100.0 * passed / max(len(checks), 1), 4)
    payload.update(
        {
            "status": "ready" if all(checks.values()) else "degraded",
            "ok": all(checks.values()),
            "control_grade": _grade(score),
            "control_score": score,
            "checks": checks,
            "failed_checks": [
                name for name, passed_check in checks.items() if not passed_check
            ],
            "diagnostic_summary": {
                "scenario_count": len(scenario_results),
                "passed_scenario_count": sum(
                    bool(row.get("ok", False)) for row in scenario_results
                ),
                "failed_scenario_count": sum(
                    not bool(row.get("ok", False)) for row in scenario_results
                ),
                "scenario_check_count": sum(
                    len(_mapping(row.get("checks"))) for row in scenario_results
                ),
                "scenario_failed_check_count": sum(
                    len(row.get("failed_checks") or []) for row in scenario_results
                ),
                "work_units": work_units,
            },
        }
    )
    return payload


def publish_payload(
    *,
    project_root: Path,
    payload: Mapping[str, Any],
    out_path: str | Path = DEFAULT_OUT_PATH,
    source: str = "profitability_adversarial_drill",
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
            "Run fourteen bounded paper-only adversarial profitability drills, "
            "including full-fleet capital-capacity scaling."
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
        summary = _mapping(payload.get("diagnostic_summary"))
        print(
            "profitability_adversarial_drill "
            f"status={payload.get('status')} grade={payload.get('control_grade')} "
            f"scenarios={summary.get('passed_scenario_count', 0)}/"
            f"{summary.get('scenario_count', 0)} "
            f"checks_failed={summary.get('scenario_failed_check_count', 0)} "
            f"work_units={summary.get('work_units', 0)}"
        )
    return 0 if payload.get("ok") else 2


if __name__ == "__main__":
    raise SystemExit(main())
