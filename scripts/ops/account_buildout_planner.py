#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from scripts.ops.long_runtime_common import load_json, write_payload
else:
    from .long_runtime_common import PROJECT_ROOT, load_json, write_payload


DEFAULT_STUDY_PATH = PROJECT_ROOT / "governance" / "health" / "account_position_study_latest.json"
DEFAULT_OPPORTUNITY_PATH = PROJECT_ROOT / "governance" / "health" / "position_opportunity_watch_latest.json"
DEFAULT_ROUND_TRIP_PATH = PROJECT_ROOT / "governance" / "health" / "position_round_trip_watch_latest.json"
DEFAULT_ALLOCATOR_PATH = PROJECT_ROOT / "governance" / "allocator" / "portfolio_allocator_service_latest.json"
DEFAULT_RISK_PATH = PROJECT_ROOT / "governance" / "risk" / "portfolio_risk_latest.json"
DEFAULT_POLICY_PATH = PROJECT_ROOT / "config" / "account_buildout_policy_v1.json"
DEFAULT_OUT_PATH = PROJECT_ROOT / "governance" / "health" / "account_buildout_plan_latest.json"


def _dict(raw: Any) -> dict[str, Any]:
    return raw if isinstance(raw, dict) else {}


def _safe_float(raw: Any, default: float = 0.0) -> float:
    try:
        value = float(raw)
    except Exception:
        return float(default)
    return value if math.isfinite(value) else float(default)


def _clamp_fraction(raw: Any, default: float) -> float:
    return max(min(_safe_float(raw, default), 1.0), 0.0)


def _parse_timestamp(raw: Any) -> datetime | None:
    text = str(raw or "").strip()
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except Exception:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _freshness_row(name: str, payload: dict[str, Any], *, now: datetime, max_age_seconds: float) -> dict[str, Any]:
    timestamp = _parse_timestamp(payload.get("timestamp_utc"))
    age = max((now - timestamp).total_seconds(), 0.0) if timestamp is not None else None
    fresh = age is not None and age <= max(float(max_age_seconds), 0.0)
    return {
        "name": name,
        "timestamp_utc": payload.get("timestamp_utc"),
        "age_seconds": round(age, 3) if age is not None else None,
        "max_age_seconds": float(max_age_seconds),
        "fresh": bool(fresh),
    }


def _policy_for_account(policy: dict[str, Any], account_label: str) -> dict[str, Any]:
    scoped = dict(policy)
    overrides = _dict(policy.get("account_overrides"))
    override = _dict(overrides.get(account_label))
    scoped.update(override)
    scoped["freshness"] = dict(_dict(policy.get("freshness")))
    if isinstance(override.get("freshness"), dict):
        scoped["freshness"].update(_dict(override.get("freshness")))
    return scoped


def _price_map(study: dict[str, Any], opportunities: dict[str, Any]) -> dict[str, float]:
    prices: dict[str, float] = {}
    for row in study.get("positions") or []:
        if not isinstance(row, dict):
            continue
        symbol = str(row.get("underlying") or row.get("symbol") or "").strip().upper()
        qty = abs(_safe_float(row.get("quantity"), 0.0))
        value = abs(_safe_float(row.get("market_value"), 0.0))
        price = value / qty if qty > 0.0 and value > 0.0 else _safe_float(row.get("average_price"), 0.0)
        if symbol and price > 0.0:
            prices[symbol] = price
    for row in study.get("underlyings") or []:
        if not isinstance(row, dict):
            continue
        symbol = str(row.get("underlying") or "").strip().upper()
        context = _dict(row.get("chart_context"))
        price = _safe_float(_dict(context.get("market")).get("last_price"), 0.0)
        if symbol and price > 0.0:
            prices[symbol] = price
    for row in opportunities.get("observations") or []:
        if not isinstance(row, dict):
            continue
        symbol = str(row.get("underlying") or "").strip().upper()
        price = _safe_float(_dict(row.get("market")).get("last_price"), 0.0)
        if symbol and price > 0.0:
            prices[symbol] = price
    return prices


def _allocator_signals(allocator: dict[str, Any], fallback_prices: dict[str, float]) -> dict[str, dict[str, Any]]:
    aggregated: dict[str, dict[str, Any]] = {}
    rows = allocator.get("approved_intents") if isinstance(allocator.get("approved_intents"), list) else []
    for row in rows:
        if not isinstance(row, dict):
            continue
        symbol = str(row.get("symbol") or "").strip().upper()
        approved_qty = max(_safe_float(row.get("approved_qty"), 0.0), 0.0)
        price = _safe_float(row.get("price"), fallback_prices.get(symbol, 0.0))
        if not symbol or approved_qty <= 0.0 or price <= 0.0:
            continue
        side = str(row.get("side") or "BUY").strip().upper()
        direction = -1.0 if side.startswith("SELL") else 1.0
        item = aggregated.setdefault(
            symbol,
            {
                "symbol": symbol,
                "net_strength": 0.0,
                "buy_strength": 0.0,
                "sell_strength": 0.0,
                "reference_price": price,
                "sleeves": [],
                "intent_count": 0,
            },
        )
        strength = approved_qty * price
        item["net_strength"] = float(item["net_strength"]) + direction * strength
        key = "sell_strength" if direction < 0.0 else "buy_strength"
        item[key] = float(item[key]) + strength
        item["reference_price"] = price
        sleeve = str(row.get("sleeve") or "").strip()
        if sleeve and sleeve not in item["sleeves"]:
            item["sleeves"].append(sleeve)
        item["intent_count"] = int(item["intent_count"]) + 1
    return {
        symbol: {
            **row,
            "net_strength": round(_safe_float(row.get("net_strength"), 0.0), 6),
            "buy_strength": round(_safe_float(row.get("buy_strength"), 0.0), 6),
            "sell_strength": round(_safe_float(row.get("sell_strength"), 0.0), 6),
            "reference_price": round(_safe_float(row.get("reference_price"), 0.0), 6),
            "sleeves": sorted(row.get("sleeves") or []),
        }
        for symbol, row in aggregated.items()
        if abs(_safe_float(row.get("net_strength"), 0.0)) > 1e-9
    }


def _weighted_fill(total: float, weights: dict[str, float], caps: dict[str, float]) -> dict[str, float]:
    remaining = max(float(total), 0.0)
    active = {
        symbol: max(float(weight), 0.0)
        for symbol, weight in weights.items()
        if float(weight) > 0.0 and float(caps.get(symbol, 0.0)) > 0.0
    }
    allocations = {symbol: 0.0 for symbol in active}
    while remaining > 1e-8 and active:
        denominator = sum(active.values())
        if denominator <= 0.0:
            break
        capped: list[str] = []
        provisional = {symbol: remaining * weight / denominator for symbol, weight in active.items()}
        for symbol, amount in provisional.items():
            headroom = max(float(caps.get(symbol, 0.0)) - allocations[symbol], 0.0)
            if amount >= headroom - 1e-9:
                allocations[symbol] += headroom
                remaining -= headroom
                capped.append(symbol)
        if capped:
            for symbol in capped:
                active.pop(symbol, None)
            continue
        for symbol, amount in provisional.items():
            allocations[symbol] += amount
        remaining = 0.0
    return {symbol: round(amount, 6) for symbol, amount in allocations.items() if amount > 1e-8}


def _account_position_state(study: dict[str, Any], account_label: str) -> dict[str, Any]:
    equity_positions: dict[str, dict[str, float]] = {}
    exposure_by_underlying: dict[str, float] = {}
    gross = 0.0
    net = 0.0
    for row in study.get("positions") or []:
        if not isinstance(row, dict) or str(row.get("account_label") or "") != account_label:
            continue
        market_value = _safe_float(row.get("market_value"), 0.0)
        gross += abs(market_value)
        net += market_value
        underlying = str(row.get("underlying") or row.get("symbol") or "").strip().upper()
        if underlying:
            exposure_by_underlying[underlying] = exposure_by_underlying.get(underlying, 0.0) + abs(market_value)
        if str(row.get("asset_type") or "").strip().upper() != "EQUITY":
            continue
        symbol = str(row.get("symbol") or "").strip().upper()
        if not symbol:
            continue
        item = equity_positions.setdefault(symbol, {"quantity": 0.0, "market_value": 0.0, "price": 0.0})
        item["quantity"] += _safe_float(row.get("quantity"), 0.0)
        item["market_value"] += market_value
        qty = abs(_safe_float(row.get("quantity"), 0.0))
        if qty > 0.0 and abs(market_value) > 0.0:
            item["price"] = abs(market_value) / qty
    return {
        "gross_market_value": gross,
        "net_market_value": net,
        "exposure_by_underlying": exposure_by_underlying,
        "equity_positions": equity_positions,
    }


def _stage_summary(notional: float, quantity: float, *, equity: float, policy: dict[str, Any]) -> dict[str, Any]:
    stage_cap = max(equity * _clamp_fraction(policy.get("max_stage_account_fraction"), 0.02), 0.01)
    count = max(int(math.ceil(abs(notional) / stage_cap)), 1)
    return {
        "stage_count": count,
        "max_stage_notional": round(stage_cap, 4),
        "average_stage_notional": round(abs(notional) / count, 4),
        "average_stage_quantity": round(abs(quantity) / count, 6),
        "expanded_rows_emitted": False,
    }


def _quantized_notional(notional: float, price: float, *, fractional: bool) -> tuple[float, float]:
    if notional <= 0.0 or price <= 0.0:
        return 0.0, 0.0
    quantity = notional / price
    if not fractional:
        quantity = float(math.floor(quantity + 1e-12))
    quantity = round(max(quantity, 0.0), 6)
    return quantity, round(quantity * price, 4)


def _opportunity_index(opportunities: dict[str, Any]) -> dict[str, dict[str, Any]]:
    rows = opportunities.get("observations") if isinstance(opportunities.get("observations"), list) else []
    return {
        str(row.get("underlying") or "").strip().upper(): row
        for row in rows
        if isinstance(row, dict) and str(row.get("underlying") or "").strip()
    }


def _review_rows(opportunities: dict[str, Any]) -> list[dict[str, Any]]:
    reviews: list[dict[str, Any]] = []
    rows = opportunities.get("candidates") if isinstance(opportunities.get("candidates"), list) else []
    for row in rows:
        if not isinstance(row, dict) or str(row.get("position_action") or "").upper() != "ROLL_REVIEW":
            continue
        reviews.append(
            {
                "symbol": str(row.get("underlying") or "").strip().upper(),
                "accounts": sorted(str(item) for item in (row.get("accounts") or []) if str(item)),
                "action": "ROLL_REVIEW",
                "reason": str(row.get("reason") or "covered_call_review"),
                "quantity": None,
                "notional": None,
                "manual_review_only": True,
                "execution_allowed": False,
            }
        )
    return reviews


def _round_trip_review_rows(round_trips: dict[str, Any]) -> list[dict[str, Any]]:
    reviews: list[dict[str, Any]] = []
    for row in round_trips.get("observations") or []:
        if not isinstance(row, dict):
            continue
        action = str(row.get("action") or "").strip().upper()
        if action not in {"PAPER_EXIT_CANDIDATE", "PAPER_REENTRY_CANDIDATE"}:
            continue
        signal_key = "exit_signal" if action == "PAPER_EXIT_CANDIDATE" else "reentry_signal"
        reviews.append(
            {
                "account_label": str(row.get("account_label") or "").strip(),
                "symbol": str(row.get("symbol") or "").strip().upper(),
                "action": action,
                "reason": ",".join(str(value) for value in (row.get("reasons") or []) if str(value)),
                "score": _safe_float(_dict(row.get(signal_key)).get("score"), 0.0),
                "zone": _dict(_dict(row.get("zones")).get("exit" if signal_key == "exit_signal" else "reentry")),
                "quantity": None,
                "notional": None,
                "manual_review_only": True,
                "paper_candidate_only": True,
                "execution_allowed": False,
            }
        )
    return reviews


def evaluate(
    *,
    study: dict[str, Any],
    opportunities: dict[str, Any],
    round_trips: dict[str, Any] | None = None,
    allocator: dict[str, Any],
    risk: dict[str, Any],
    policy: dict[str, Any] | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    current = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    active_policy = dict(policy or {})
    round_trip_payload = dict(round_trips or {})
    freshness_policy = _dict(active_policy.get("freshness"))
    freshness = {
        "account_study": _freshness_row(
            "account_study",
            study,
            now=current,
            max_age_seconds=_safe_float(freshness_policy.get("account_study_max_age_seconds"), 300.0),
        ),
        "position_opportunity": _freshness_row(
            "position_opportunity",
            opportunities,
            now=current,
            max_age_seconds=_safe_float(freshness_policy.get("position_opportunity_max_age_seconds"), 900.0),
        ),
        "portfolio_allocator": _freshness_row(
            "portfolio_allocator",
            allocator,
            now=current,
            max_age_seconds=_safe_float(freshness_policy.get("portfolio_allocator_max_age_seconds"), 900.0),
        ),
        "portfolio_risk": _freshness_row(
            "portfolio_risk",
            risk,
            now=current,
            max_age_seconds=_safe_float(freshness_policy.get("portfolio_risk_max_age_seconds"), 3600.0),
        ),
        "position_round_trip": _freshness_row(
            "position_round_trip",
            round_trip_payload,
            now=current,
            max_age_seconds=_safe_float(freshness_policy.get("position_round_trip_max_age_seconds"), 900.0),
        ),
    }
    hard_blockers: list[str] = []
    holds: list[str] = []
    accounts = [row for row in (study.get("accounts") or []) if isinstance(row, dict)]
    if not bool(study.get("ok", False)):
        hard_blockers.append("account_position_study_not_ok")
    if not freshness["account_study"]["fresh"]:
        hard_blockers.append("account_position_study_stale_or_missing_timestamp")
    if not accounts:
        hard_blockers.append("no_redacted_account_summaries")
    planning_fresh = all(
        freshness[name]["fresh"]
        for name in ("position_opportunity", "portfolio_allocator", "portfolio_risk")
    )
    if not planning_fresh:
        holds.extend(
            f"{name}_stale_or_missing_timestamp"
            for name in ("position_opportunity", "portfolio_allocator", "portfolio_risk")
            if not freshness[name]["fresh"]
        )

    policy_enabled = bool(active_policy.get("enabled", True))
    if not policy_enabled:
        holds.append("account_buildout_policy_disabled")

    allocator_summary = _dict(allocator.get("summary"))
    risk_limits = _dict(risk.get("limits"))
    allocator_gross = _clamp_fraction(allocator_summary.get("gross_budget"), 0.0)
    risk_gross = _clamp_fraction(risk_limits.get("gross_exposure_cap"), 1.0)
    policy_gross = _clamp_fraction(active_policy.get("max_deployable_equity_fraction"), 1.0)
    effective_gross = min(allocator_gross, risk_gross, policy_gross)
    risk_symbol_cap = _clamp_fraction(risk_limits.get("max_single_symbol_share"), 1.0)
    policy_symbol_cap = _clamp_fraction(active_policy.get("max_single_symbol_fraction"), 0.15)
    base_symbol_cap = min(risk_symbol_cap, policy_symbol_cap)
    raw_symbol_budgets = _dict(risk_limits.get("symbol_budgets"))
    symbol_budgets = {str(key).strip().upper(): _clamp_fraction(value, base_symbol_cap) for key, value in raw_symbol_budgets.items()}

    prices = _price_map(study, opportunities)
    signals = _allocator_signals(allocator, prices)
    allocator_input_freshness = _dict(allocator.get("input_freshness"))
    risk_input_freshness = _dict(risk.get("input_freshness"))
    allocator_sources_ready = bool(allocator_input_freshness.get("sources_ready", True))
    risk_sources_ready = bool(risk_input_freshness.get("sources_ready", True))
    allocator_status = str(allocator.get("overall_status") or "").strip().lower()
    risk_status = str(risk.get("overall_status") or "").strip().lower()
    allocator_contract_ready = allocator_status in {"", "ready"}
    risk_contract_ready = risk_status in {"", "ready"}
    signal_contract_ready = bool(
        not signals
        or (allocator_sources_ready and risk_sources_ready and allocator_contract_ready and risk_contract_ready)
    )
    if signals and not allocator_sources_ready:
        holds.append("portfolio_allocator_upstream_sources_stale")
    if signals and not risk_sources_ready:
        holds.append("portfolio_risk_upstream_sources_stale")
    if signals and not allocator_contract_ready:
        holds.append(f"portfolio_allocator_contract_{allocator_status or 'not_ready'}")
    if signals and not risk_contract_ready:
        holds.append(f"portfolio_risk_contract_{risk_status or 'not_ready'}")
    opportunities_by_symbol = _opportunity_index(opportunities)
    actions: list[dict[str, Any]] = []
    account_plans: list[dict[str, Any]] = []
    skipped_signals: list[dict[str, Any]] = []

    if allocator_gross <= 0.0:
        holds.append("allocator_gross_budget_zero")
    if not signals:
        holds.append("no_allocator_approved_directional_signals")

    can_plan = bool(
        not hard_blockers
        and policy_enabled
        and planning_fresh
        and signal_contract_ready
        and effective_gross > 0.0
        and signals
    )
    for account in accounts:
        label = str(account.get("account_label") or "").strip()
        scoped_policy = _policy_for_account(active_policy, label)
        state = _account_position_state(study, label)
        equity = max(_safe_float(account.get("liquidation_value"), account.get("equity", 0.0)), 0.0)
        cash = max(_safe_float(account.get("cash_balance"), 0.0), 0.0)
        available = max(_safe_float(account.get("available_funds"), 0.0), 0.0)
        flags = _dict(account.get("flags"))
        reserve = equity * _clamp_fraction(scoped_policy.get("cash_reserve_fraction"), 0.1)
        funding_base = max(cash, available) if bool(scoped_policy.get("allow_margin_expansion", False)) else cash
        cash_capacity = max(funding_base - reserve, 0.0)
        gross_cap_notional = equity * effective_gross
        gross_headroom = max(gross_cap_notional - _safe_float(state.get("gross_market_value"), 0.0), 0.0)
        buy_pool = min(gross_headroom, cash_capacity)
        account_holds: list[str] = []
        if equity <= 0.0:
            account_holds.append("nonpositive_account_liquidation_value")
        if bool(flags.get("closing_only", False)):
            account_holds.append("account_closing_only")
        if bool(flags.get("in_margin_call", False)):
            account_holds.append("account_in_margin_call")
        if gross_headroom <= 0.0:
            account_holds.append("existing_exposure_at_or_above_effective_gross_cap")
        if cash_capacity <= 0.0:
            account_holds.append("cash_reserve_or_cash_balance_leaves_no_addition_capacity")

        account_action_start = len(actions)
        additions_allowed = can_plan and equity > 0.0 and not flags.get("closing_only") and not flags.get("in_margin_call")
        buy_weights = {
            symbol: _safe_float(signal.get("net_strength"), 0.0)
            for symbol, signal in signals.items()
            if _safe_float(signal.get("net_strength"), 0.0) > 0.0
        }
        exposure_by_symbol = _dict(state.get("exposure_by_underlying"))
        buy_caps = {
            symbol: max(
                equity * min(base_symbol_cap, symbol_budgets.get(symbol, base_symbol_cap))
                - _safe_float(exposure_by_symbol.get(symbol), 0.0),
                0.0,
            )
            for symbol in buy_weights
        }
        buy_allocations = _weighted_fill(buy_pool, buy_weights, buy_caps) if additions_allowed else {}
        equity_positions = _dict(state.get("equity_positions"))
        for symbol, planned_notional in buy_allocations.items():
            signal = signals[symbol]
            current_position = _dict(equity_positions.get(symbol))
            price = _safe_float(signal.get("reference_price"), prices.get(symbol, current_position.get("price", 0.0)))
            quantity, actual_notional = _quantized_notional(
                planned_notional,
                price,
                fractional=bool(scoped_policy.get("fractional_equities", True)),
            )
            minimum = max(_safe_float(scoped_policy.get("minimum_order_notional"), 5.0), 0.0)
            if actual_notional < minimum or quantity <= 0.0:
                skipped_signals.append({"account_label": label, "symbol": symbol, "reason": "below_minimum_notional"})
                continue
            current_qty = _safe_float(current_position.get("quantity"), 0.0)
            current_notional = _safe_float(exposure_by_symbol.get(symbol), 0.0)
            context = _dict(opportunities_by_symbol.get(symbol))
            actions.append(
                {
                    "account_label": label,
                    "symbol": symbol,
                    "asset_type": "EQUITY",
                    "action": "ADD_OR_COVER",
                    "current_quantity": round(current_qty, 6),
                    "current_symbol_exposure": round(current_notional, 4),
                    "proposed_quantity_change": quantity,
                    "proposed_notional_change": actual_notional,
                    "target_quantity": round(current_qty + quantity, 6),
                    "target_symbol_exposure": round(current_notional + actual_notional, 4),
                    "reference_price": round(price, 6),
                    "post_plan_symbol_fraction": round((current_notional + actual_notional) / equity, 6),
                    "allocator_sleeves": list(signal.get("sleeves") or []),
                    "allocator_net_strength": signal.get("net_strength"),
                    "position_observer_action": context.get("position_action"),
                    "staging": _stage_summary(actual_notional, quantity, equity=equity, policy=scoped_policy),
                    "advisory_only": True,
                    "paper_plan_only": True,
                    "execution_allowed": False,
                }
            )

        sell_weights: dict[str, float] = {}
        sell_caps: dict[str, float] = {}
        for symbol, signal in signals.items():
            net_strength = _safe_float(signal.get("net_strength"), 0.0)
            current_position = _dict(equity_positions.get(symbol))
            current_qty = _safe_float(current_position.get("quantity"), 0.0)
            current_long_notional = max(_safe_float(current_position.get("market_value"), 0.0), 0.0)
            if net_strength >= 0.0:
                continue
            if current_qty <= 0.0 or current_long_notional <= 0.0:
                skipped_signals.append({"account_label": label, "symbol": symbol, "reason": "sell_cannot_open_new_short"})
                continue
            sell_weights[symbol] = abs(net_strength)
            sell_caps[symbol] = current_long_notional
        reduction_pool = min(
            sum(sell_caps.values()),
            equity * _clamp_fraction(scoped_policy.get("max_reduction_account_fraction_per_cycle"), 0.1),
        )
        sell_allocations = _weighted_fill(reduction_pool, sell_weights, sell_caps) if can_plan and equity > 0.0 else {}
        for symbol, planned_notional in sell_allocations.items():
            signal = signals[symbol]
            current_position = _dict(equity_positions.get(symbol))
            current_qty = max(_safe_float(current_position.get("quantity"), 0.0), 0.0)
            price = _safe_float(signal.get("reference_price"), prices.get(symbol, current_position.get("price", 0.0)))
            quantity, actual_notional = _quantized_notional(
                planned_notional,
                price,
                fractional=bool(scoped_policy.get("fractional_equities", True)),
            )
            quantity = min(quantity, current_qty)
            actual_notional = round(quantity * price, 4)
            minimum = max(_safe_float(scoped_policy.get("minimum_order_notional"), 5.0), 0.0)
            if actual_notional < minimum or quantity <= 0.0:
                skipped_signals.append({"account_label": label, "symbol": symbol, "reason": "below_minimum_reduction_notional"})
                continue
            actions.append(
                {
                    "account_label": label,
                    "symbol": symbol,
                    "asset_type": "EQUITY",
                    "action": "REDUCE_LONG",
                    "current_quantity": round(current_qty, 6),
                    "current_symbol_exposure": round(_safe_float(exposure_by_symbol.get(symbol), 0.0), 4),
                    "proposed_quantity_change": round(-quantity, 6),
                    "proposed_notional_change": round(-actual_notional, 4),
                    "target_quantity": round(current_qty - quantity, 6),
                    "reference_price": round(price, 6),
                    "allocator_sleeves": list(signal.get("sleeves") or []),
                    "allocator_net_strength": signal.get("net_strength"),
                    "staging": _stage_summary(actual_notional, quantity, equity=equity, policy=scoped_policy),
                    "advisory_only": True,
                    "paper_plan_only": True,
                    "execution_allowed": False,
                }
            )

        action_count = len(actions) - account_action_start
        if action_count:
            account_state = "plan_ready"
        elif account_holds:
            account_state = "observe_only_constrained"
        else:
            account_state = "observe_only"
        account_plans.append(
            {
                "account_label": label,
                "operator_account_label": account.get("operator_account_label"),
                "account_type": account.get("account_type"),
                "liquidation_value": round(equity, 4),
                "cash_balance": round(cash, 4),
                "current_gross_position_market_value": round(_safe_float(state.get("gross_market_value"), 0.0), 4),
                "effective_gross_cap_notional": round(gross_cap_notional, 4),
                "gross_headroom": round(gross_headroom, 4),
                "cash_reserve": round(reserve, 4),
                "addition_funding_capacity": round(cash_capacity, 4),
                "addition_plan_pool": round(buy_pool, 4),
                "action_count": action_count,
                "plan_state": account_state,
                "holds": account_holds,
            }
        )

    reviews = _review_rows(opportunities)
    round_trip_reviews: list[dict[str, Any]] = []
    if round_trip_payload:
        if freshness["position_round_trip"]["fresh"]:
            round_trip_reviews = _round_trip_review_rows(round_trip_payload)
            reviews.extend(round_trip_reviews)
        else:
            holds.append("position_round_trip_stale_review_suppressed")
    if hard_blockers:
        overall_status = "blocked"
        plan_state = "blocked_account_truth"
    elif not policy_enabled:
        overall_status = "ready"
        plan_state = "disabled_by_policy"
    elif not planning_fresh:
        overall_status = "ready"
        plan_state = "held_by_freshness"
    elif not signal_contract_ready:
        overall_status = "ready"
        plan_state = "held_by_upstream_contract"
    elif actions:
        overall_status = "ready"
        plan_state = "plan_ready"
    elif reviews:
        overall_status = "ready"
        plan_state = "review_only"
    else:
        overall_status = "ready"
        plan_state = "observe_only"

    return {
        "timestamp_utc": current.isoformat(),
        "schema_version": 1,
        "ok": overall_status != "blocked",
        "overall_status": overall_status,
        "mode": "advisory_paper_only",
        "plan_state": plan_state,
        "buildout_ready": bool(can_plan),
        "plan_generated": bool(actions),
        "account_count": len(accounts),
        "action_count": len(actions),
        "review_count": len(reviews),
        "round_trip_review_count": len(round_trip_reviews),
        "approved_signal_count": len(signals),
        "effective_limits": {
            "allocator_gross_budget_fraction": round(allocator_gross, 6),
            "portfolio_risk_gross_cap_fraction": round(risk_gross, 6),
            "policy_gross_cap_fraction": round(policy_gross, 6),
            "effective_gross_cap_fraction": round(effective_gross, 6),
            "effective_default_symbol_cap_fraction": round(base_symbol_cap, 6),
            "allow_margin_expansion": bool(active_policy.get("allow_margin_expansion", False)),
            "allow_new_short_positions": False,
        },
        "freshness": freshness,
        "upstream_contracts": {
            "portfolio_allocator_status": allocator_status or "unspecified",
            "portfolio_allocator_sources_ready": allocator_sources_ready,
            "portfolio_risk_status": risk_status or "unspecified",
            "portfolio_risk_sources_ready": risk_sources_ready,
            "signal_contract_ready": signal_contract_ready,
            "enforced_when_approved_signals_exist": True,
        },
        "hard_blockers": sorted(set(hard_blockers)),
        "holds": sorted(set(holds)),
        "accounts": account_plans,
        "actions": actions,
        "reviews": reviews,
        "skipped_signals": skipped_signals,
        "allocator_signals": [signals[key] for key in sorted(signals)],
        "safety_contract": {
            "advisory_only": True,
            "paper_plan_only": True,
            "does_not_publish_execution_intents": True,
            "does_not_call_broker_order_endpoints": True,
            "live_execution_allowed": False,
            "automatic_option_sizing": False,
            "new_short_positions_allowed": False,
            "existing_positions_are_starting_state": True,
            "standard_allocator_and_risk_service_required": True,
            "required_route": str(
                active_policy.get("required_route")
                or "portfolio_allocator_to_risk_service_to_standard_paper_execution_gateway"
            ),
        },
        "regression_contract": {
            "account_size_agnostic_fractional_sizing": True,
            "arbitrary_existing_position_count_supported": True,
            "empty_accounts_supported": True,
            "existing_exposure_consumes_headroom": True,
            "gross_and_symbol_caps_are_hard_planning_limits": True,
            "stale_inputs_emit_no_action_plan": True,
            "fresh_wrapper_timestamps_cannot_launder_stale_upstream_sources": True,
            "zero_allocator_budget_is_valid_observe_only_state": True,
            "sell_signals_cannot_create_uncovered_short_positions": True,
            "roll_reviews_never_receive_automatic_quantities": True,
            "round_trip_reviews_never_receive_automatic_quantities": True,
            "stages_are_compressed_not_expanded": True,
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build account-size-agnostic advisory portfolio buildout plans.")
    parser.add_argument("--study-file", default=str(DEFAULT_STUDY_PATH))
    parser.add_argument("--opportunity-file", default=str(DEFAULT_OPPORTUNITY_PATH))
    parser.add_argument("--round-trip-file", default=str(DEFAULT_ROUND_TRIP_PATH))
    parser.add_argument("--allocator-file", default=str(DEFAULT_ALLOCATOR_PATH))
    parser.add_argument("--risk-file", default=str(DEFAULT_RISK_PATH))
    parser.add_argument("--policy-file", default=str(DEFAULT_POLICY_PATH))
    parser.add_argument("--out-file", default=str(DEFAULT_OUT_PATH))
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    payload = evaluate(
        study=load_json(Path(args.study_file).expanduser()),
        opportunities=load_json(Path(args.opportunity_file).expanduser()),
        round_trips=load_json(Path(args.round_trip_file).expanduser()),
        allocator=load_json(Path(args.allocator_file).expanduser()),
        risk=load_json(Path(args.risk_file).expanduser()),
        policy=load_json(Path(args.policy_file).expanduser()),
    )
    payload["source_files"] = {
        "account_study": str(Path(args.study_file).expanduser()),
        "position_opportunity": str(Path(args.opportunity_file).expanduser()),
        "position_round_trip": str(Path(args.round_trip_file).expanduser()),
        "portfolio_allocator": str(Path(args.allocator_file).expanduser()),
        "portfolio_risk": str(Path(args.risk_file).expanduser()),
        "policy": str(Path(args.policy_file).expanduser()),
    }
    write_payload(Path(args.out_file).expanduser(), payload)
    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        print(
            "account_buildout_planner "
            f"status={payload.get('overall_status')} "
            f"plan_state={payload.get('plan_state')} "
            f"accounts={payload.get('account_count', 0)} "
            f"actions={payload.get('action_count', 0)} "
            f"reviews={payload.get('review_count', 0)}"
        )
    return 0 if payload.get("ok") else 2


if __name__ == "__main__":
    raise SystemExit(main())
