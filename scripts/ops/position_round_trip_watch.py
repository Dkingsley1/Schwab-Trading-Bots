#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import statistics
import sys
from copy import deepcopy
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from core.provider_access_guard import (
        activate_provider_cooldown,
        mark_provider_recovered,
        provider_access_status,
        provider_http_status_code,
        provider_request_slot,
    )
    from scripts.brokers.schwab.common import build_schwab_trader, resp_json
    from scripts.ops.long_runtime_common import load_json, write_payload
    from scripts.ops.trading_tax_estimator import _account_tax_treatments
else:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    from core.provider_access_guard import (
        activate_provider_cooldown,
        mark_provider_recovered,
        provider_access_status,
        provider_http_status_code,
        provider_request_slot,
    )
    from scripts.brokers.schwab.common import build_schwab_trader, resp_json
    from .long_runtime_common import load_json, write_payload
    from .trading_tax_estimator import _account_tax_treatments


DEFAULT_STUDY_PATH = PROJECT_ROOT / "governance" / "health" / "account_position_study_latest.json"
DEFAULT_CHART_PATH = PROJECT_ROOT / "governance" / "health" / "held_position_chart_cache_latest.json"
DEFAULT_RISK_PATH = PROJECT_ROOT / "governance" / "risk" / "portfolio_risk_latest.json"
DEFAULT_TAX_ESTIMATE_PATH = PROJECT_ROOT / "governance" / "health" / "trading_tax_estimate_latest.json"
DEFAULT_TAX_PROFILE_PATH = PROJECT_ROOT / "config" / "trading_tax_profile.json"
DEFAULT_DIVIDEND_PATH = PROJECT_ROOT / "governance" / "health" / "held_position_dividend_calendar_latest.json"
DEFAULT_POLICY_PATH = PROJECT_ROOT / "config" / "position_round_trip_policy_v1.json"
DEFAULT_STATE_PATH = PROJECT_ROOT / "governance" / "state" / "position_round_trip_paper_state_latest.json"
DEFAULT_EVENTS_PATH = PROJECT_ROOT / "governance" / "events" / "position_round_trip_paper_events.jsonl"
DEFAULT_OUT_PATH = PROJECT_ROOT / "governance" / "health" / "position_round_trip_watch_latest.json"


def _dict(raw: Any) -> dict[str, Any]:
    return raw if isinstance(raw, dict) else {}


def _list(raw: Any) -> list[Any]:
    return raw if isinstance(raw, list) else []


def _number(raw: Any) -> float | None:
    if raw in {None, ""}:
        return None
    try:
        value = float(raw)
    except Exception:
        return None
    return value if math.isfinite(value) else None


def _float(raw: Any, default: float = 0.0) -> float:
    value = _number(raw)
    return float(value if value is not None else default)


def _clamp(raw: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(min(float(raw), high), low)


def _utc(now: datetime | None = None) -> datetime:
    value = now or datetime.now(timezone.utc)
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _parse_datetime(raw: Any) -> datetime | None:
    text = str(raw or "").strip()
    if not text:
        return None
    try:
        value = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except Exception:
        return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _parse_date(raw: Any) -> date | None:
    parsed = _parse_datetime(raw)
    if parsed is not None:
        return parsed.date()
    try:
        return date.fromisoformat(str(raw or "")[:10])
    except Exception:
        return None


def _business_days_after(start: date, days: int) -> date:
    cursor = start
    remaining = max(int(days), 0)
    while remaining:
        cursor += timedelta(days=1)
        if cursor.weekday() < 5:
            remaining -= 1
    return cursor


def _sma(values: list[float], window: int) -> float | None:
    if len(values) < max(window, 1):
        return None
    return sum(values[-window:]) / float(window)


def _rsi(values: list[float], window: int = 14) -> float | None:
    if len(values) <= window:
        return None
    changes = [values[index] - values[index - 1] for index in range(len(values) - window, len(values))]
    gains = sum(max(change, 0.0) for change in changes) / float(window)
    losses = sum(max(-change, 0.0) for change in changes) / float(window)
    if losses <= 1e-12:
        return 100.0 if gains > 0.0 else 50.0
    relative = gains / losses
    return 100.0 - (100.0 / (1.0 + relative))


def _atr(rows: list[dict[str, Any]], window: int = 14) -> float | None:
    if len(rows) <= window:
        return None
    ranges: list[float] = []
    for index in range(len(rows) - window, len(rows)):
        row = rows[index]
        previous_close = _number(rows[index - 1].get("close"))
        high = _number(row.get("high"))
        low = _number(row.get("low"))
        if high is None or low is None or previous_close is None:
            continue
        ranges.append(max(high - low, abs(high - previous_close), abs(low - previous_close)))
    return sum(ranges) / len(ranges) if ranges else None


def _chart_metrics(chart: dict[str, Any], *, policy: dict[str, Any], now: datetime) -> dict[str, Any]:
    daily = [_dict(row) for row in _list(chart.get("daily_bars"))]
    intraday = [_dict(row) for row in _list(chart.get("intraday_bars"))]
    fetched = _parse_datetime(chart.get("fetched_at_utc") or chart.get("timestamp_utc"))
    age = max((now - fetched).total_seconds(), 0.0) if fetched is not None else None
    max_age = _float(policy.get("chart_cache_max_age_seconds"), 900.0)
    daily_closes = [value for row in daily if (value := _number(row.get("close"))) is not None and value > 0.0]
    intraday_closes = [value for row in intraday if (value := _number(row.get("close"))) is not None and value > 0.0]
    last = intraday_closes[-1] if intraday_closes else (daily_closes[-1] if daily_closes else None)
    sma20 = _sma(daily_closes, 20)
    sma50 = _sma(daily_closes, 50)
    rsi14 = _rsi(daily_closes, 14)
    atr14 = _atr(daily, 14)
    recent = daily[-20:]
    lows = [value for row in recent if (value := _number(row.get("low"))) is not None and value > 0.0]
    highs = [value for row in recent if (value := _number(row.get("high"))) is not None and value > 0.0]
    support = min(lows) if lows else None
    resistance = max(highs) if highs else None
    intraday_momentum = None
    if len(intraday_closes) >= 6 and intraday_closes[-6] > 0.0:
        intraday_momentum = intraday_closes[-1] / intraday_closes[-6] - 1.0
    vwap_numerator = 0.0
    vwap_volume = 0.0
    for row in intraday:
        close = _number(row.get("close"))
        volume = _number(row.get("volume"))
        if close is None or volume is None or volume <= 0.0:
            continue
        vwap_numerator += close * volume
        vwap_volume += volume
    vwap = vwap_numerator / vwap_volume if vwap_volume > 0.0 else None
    minimum_daily = int(_float(policy.get("minimum_daily_bars"), 50))
    minimum_intraday = int(_float(policy.get("minimum_intraday_bars"), 24))
    ready = (
        age is not None
        and age <= max_age
        and len(daily_closes) >= minimum_daily
        and len(intraday_closes) >= minimum_intraday
        and last is not None
    )
    return {
        "ready": ready,
        "fetched_at_utc": chart.get("fetched_at_utc") or chart.get("timestamp_utc"),
        "age_seconds": round(age, 3) if age is not None else None,
        "max_age_seconds": max_age,
        "daily_bar_count": len(daily_closes),
        "intraday_bar_count": len(intraday_closes),
        "last_price": round(last, 6) if last is not None else None,
        "sma20": round(sma20, 6) if sma20 is not None else None,
        "sma50": round(sma50, 6) if sma50 is not None else None,
        "rsi14": round(rsi14, 4) if rsi14 is not None else None,
        "atr14": round(atr14, 6) if atr14 is not None else None,
        "support_20d": round(support, 6) if support is not None else None,
        "resistance_20d": round(resistance, 6) if resistance is not None else None,
        "intraday_vwap": round(vwap, 6) if vwap is not None else None,
        "intraday_momentum_25m": round(intraday_momentum, 8) if intraday_momentum is not None else None,
    }


def _position_inventory(study: dict[str, Any], policy: dict[str, Any]) -> list[dict[str, Any]]:
    multiplier = _float(policy.get("covered_call_contract_multiplier"), 100.0)
    groups: dict[tuple[str, str], dict[str, Any]] = {}
    for raw in _list(study.get("positions")):
        row = _dict(raw)
        account = str(row.get("account_label") or "").strip()
        underlying = str(row.get("underlying") or row.get("symbol") or "").strip().upper()
        if not account or not underlying:
            continue
        item = groups.setdefault(
            (account, underlying),
            {
                "account_label": account,
                "symbol": underlying,
                "equity_quantity": 0.0,
                "equity_market_value": 0.0,
                "average_price": 0.0,
                "short_call_contracts": 0.0,
            },
        )
        asset_type = str(row.get("asset_type") or "").strip().upper()
        if asset_type == "EQUITY" and str(row.get("symbol") or "").strip().upper() == underlying:
            quantity = _float(row.get("quantity"), 0.0)
            item["equity_quantity"] += quantity
            item["equity_market_value"] += _float(row.get("market_value"), 0.0)
            if quantity > 0.0 and _float(row.get("average_price"), 0.0) > 0.0:
                item["average_price"] = _float(row.get("average_price"), 0.0)
        elif asset_type == "OPTION" and _float(row.get("quantity"), 0.0) < 0.0:
            item["short_call_contracts"] += abs(_float(row.get("quantity"), 0.0))
    output: list[dict[str, Any]] = []
    for item in groups.values():
        quantity = max(_float(item.get("equity_quantity"), 0.0), 0.0)
        reserved = min(quantity, _float(item.get("short_call_contracts"), 0.0) * multiplier)
        output.append(
            {
                **item,
                "covered_call_reserved_quantity": round(reserved, 6),
                "unencumbered_quantity": round(max(quantity - reserved, 0.0), 6),
            }
        )
    return sorted(output, key=lambda row: (str(row["account_label"]), str(row["symbol"])))


def _symbol_exposure(study: dict[str, Any]) -> dict[str, float]:
    exposure: dict[str, float] = {}
    for row in _list(study.get("positions")):
        item = _dict(row)
        underlying = str(item.get("underlying") or item.get("symbol") or "").strip().upper()
        if underlying:
            exposure[underlying] = exposure.get(underlying, 0.0) + abs(_float(item.get("market_value"), 0.0))
    return exposure


def _model_stance(study: dict[str, Any], symbol: str) -> dict[str, Any]:
    for row in _list(study.get("underlyings")):
        item = _dict(row)
        if str(item.get("underlying") or "").strip().upper() != symbol:
            continue
        context = _dict(item.get("chart_context"))
        return _dict(context.get("stance"))
    return {}


def _exit_score(
    metrics: dict[str, Any],
    *,
    average_price: float,
    concentration_share: float,
    model_stance: dict[str, Any],
    policy: dict[str, Any],
) -> dict[str, Any]:
    price = _float(metrics.get("last_price"), 0.0)
    rsi = _float(metrics.get("rsi14"), 50.0)
    sma20 = _float(metrics.get("sma20"), price)
    resistance = _float(metrics.get("resistance_20d"), price)
    momentum = _float(metrics.get("intraday_momentum_25m"), 0.0)
    vwap = _float(metrics.get("intraday_vwap"), price)
    overbought = _float(policy.get("overbought_rsi"), 68.0)
    components = {
        "rsi_extension": 0.20 * _clamp((rsi - 55.0) / max(overbought - 55.0, 1.0)),
        "near_resistance": 0.16 * _clamp(1.0 - max(resistance - price, 0.0) / max(price * 0.03, 1e-9)),
        "sma20_extension": 0.14 * _clamp((price / max(sma20, 1e-9) - 1.0) / 0.06),
        "intraday_rollover": 0.14 * _clamp(-momentum / 0.01),
        "below_vwap": 0.08 if price < vwap else 0.0,
        "unrealized_gain": 0.12 * _clamp((price / max(average_price, 1e-9) - 1.0) / 0.25) if average_price > 0.0 else 0.0,
        "concentration": 0.08 * _clamp(concentration_share / 0.30),
        "fresh_model_sell": 0.16 if str(model_stance.get("master_action") or "").upper() == "SELL" else 0.0,
    }
    return {
        "score": round(_clamp(sum(components.values())), 6),
        "components": {key: round(value, 6) for key, value in components.items()},
    }


def _reentry_score(
    metrics: dict[str, Any],
    *,
    exit_price: float,
    minimum_discount: float,
    policy: dict[str, Any],
) -> dict[str, Any]:
    price = _float(metrics.get("last_price"), 0.0)
    rsi = _float(metrics.get("rsi14"), 50.0)
    support = _float(metrics.get("support_20d"), price)
    momentum = _float(metrics.get("intraday_momentum_25m"), 0.0)
    vwap = _float(metrics.get("intraday_vwap"), price)
    sma20 = _float(metrics.get("sma20"), price)
    sma50 = _float(metrics.get("sma50"), price)
    discount = 1.0 - price / max(exit_price, 1e-9)
    rsi_ceiling = _float(policy.get("reentry_rsi_ceiling"), 48.0)
    components = {
        "discount_after_friction": 0.28 * _clamp(discount / max(minimum_discount, 1e-6)),
        "rsi_reset": 0.20 * _clamp((rsi_ceiling + 12.0 - rsi) / 20.0),
        "near_support": 0.16 * _clamp(1.0 - abs(price - support) / max(price * 0.04, 1e-9)),
        "positive_intraday_reversal": 0.14 * _clamp(momentum / 0.01),
        "above_vwap": 0.10 if price >= vwap else 0.0,
        "trend_preserved": 0.12 if sma20 >= sma50 else 0.0,
    }
    return {
        "score": round(_clamp(sum(components.values())), 6),
        "discount_fraction": round(discount, 8),
        "components": {key: round(value, 6) for key, value in components.items()},
    }


def _tax_friction(
    account_label: str,
    *,
    treatments: dict[str, str],
    tax_estimate: dict[str, Any],
    policy: dict[str, Any],
) -> dict[str, Any]:
    treatment = str(treatments.get(account_label) or "unknown")
    if treatment == "tax_advantaged":
        return {"tax_treatment": treatment, "tax_friction_bps": 0.0, "evidence_status": "verified_tax_advantaged"}
    upper = _number(_dict(tax_estimate.get("federal")).get("estimate_upper_usd"))
    buckets = _dict(_dict(tax_estimate.get("taxable_activity")).get("upper_scenario_buckets_usd"))
    positive = sum(max(_float(buckets.get(key), 0.0), 0.0) for key in (
        "ordinary_investment_income",
        "short_term_capital",
        "long_term_capital",
        "qualified_dividends",
    ))
    if treatment == "taxable" and upper is not None and positive > 0.0:
        rate = _clamp(upper / positive, 0.0, 0.45)
        return {
            "tax_treatment": treatment,
            "tax_friction_bps": round(rate * 10000.0, 3),
            "evidence_status": "derived_from_current_tax_estimate",
        }
    return {
        "tax_treatment": treatment,
        "tax_friction_bps": _float(policy.get("unknown_tax_friction_bps"), 300.0),
        "evidence_status": "policy_hurdle_tax_evidence_unknown",
    }


def _dividend_guard(dividend_calendar: dict[str, Any], symbol: str, *, policy: dict[str, Any], now: datetime) -> dict[str, Any]:
    nearest: date | None = None
    for row in _list(dividend_calendar.get("events")):
        item = _dict(row)
        if str(item.get("symbol") or "").strip().upper() != symbol:
            continue
        ex_date = _parse_date(item.get("ex_date") or item.get("ex_dividend_date"))
        if ex_date is not None and (nearest is None or abs((ex_date - now.date()).days) < abs((nearest - now.date()).days)):
            nearest = ex_date
    if nearest is None:
        return {"evidence_status": "unavailable", "blocked": False, "nearest_ex_date": None}
    delta = (nearest - now.date()).days
    before = int(_float(policy.get("dividend_blackout_days_before_ex_date"), 3))
    after = int(_float(policy.get("dividend_blackout_days_after_ex_date"), 1))
    return {
        "evidence_status": "available",
        "blocked": -after <= delta <= before,
        "nearest_ex_date": nearest.isoformat(),
        "days_to_ex_date": delta,
    }


def _read_events(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                row = json.loads(line)
            except Exception:
                continue
            if isinstance(row, dict):
                rows.append(row)
    return rows


def _apply_paper_events(
    state: dict[str, Any],
    events: list[dict[str, Any]],
    *,
    policy: dict[str, Any],
    treatments: dict[str, str],
    tax_estimate: dict[str, Any],
) -> dict[str, Any]:
    updated = deepcopy(state) if state else {"schema_version": 1, "positions": {}, "completed_round_trips": []}
    positions = dict(_dict(updated.get("positions")))
    processed = set(str(value) for value in _list(updated.get("processed_event_ids")) if str(value))
    completed = list(_list(updated.get("completed_round_trips")))
    for index, raw in enumerate(events):
        row = _dict(raw)
        metadata = _dict(row.get("metadata"))
        event_id = str(row.get("event_id") or row.get("fill_id") or f"event_{index}")
        if event_id in processed:
            continue
        leg = str(row.get("round_trip_leg") or metadata.get("round_trip_leg") or "").strip().lower()
        account = str(row.get("account_label") or metadata.get("account_label") or "").strip()
        symbol = str(row.get("symbol") or metadata.get("symbol") or "").strip().upper()
        if leg not in {"exit", "reentry", "reentry_failed"} or not account or not symbol:
            continue
        key = f"{account}|{symbol}"
        current = dict(_dict(positions.get(key)))
        timestamp = _parse_datetime(row.get("timestamp_utc") or row.get("filled_at_utc")) or datetime.now(timezone.utc)
        price = _float(row.get("fill_price") or metadata.get("fill_price"), 0.0)
        quantity = abs(_float(row.get("quantity") or metadata.get("quantity"), 0.0))
        round_trip_id = str(row.get("position_round_trip_id") or metadata.get("position_round_trip_id") or event_id)
        if leg == "exit" and price > 0.0 and quantity > 0.0:
            friction = _tax_friction(
                account,
                treatments=treatments,
                tax_estimate=tax_estimate,
                policy=policy,
            )
            settlement = _business_days_after(
                timestamp.date(),
                int(_float(policy.get("settlement_business_days"), 1)),
            )
            current.update(
                {
                    "phase": "paper_exited_waiting_settlement",
                    "position_round_trip_id": round_trip_id,
                    "exit_fill_price": price,
                    "exit_quantity": quantity,
                    "exit_filled_at_utc": timestamp.isoformat(),
                    "settlement_date": settlement.isoformat(),
                    "tax_friction_bps_at_exit": _float(friction.get("tax_friction_bps"), 0.0),
                    "tax_friction_evidence_status_at_exit": friction.get("evidence_status"),
                    "failed_reentry_count": int(current.get("failed_reentry_count", 0) or 0),
                }
            )
        elif leg == "reentry" and price > 0.0 and quantity > 0.0 and _float(current.get("exit_fill_price"), 0.0) > 0.0:
            exit_price = _float(current.get("exit_fill_price"), 0.0)
            costs = _float(policy.get("round_trip_cost_bps"), 24.0)
            tax_hurdle = _float(
                current.get("tax_friction_bps_at_exit"),
                _float(policy.get("unknown_tax_friction_bps"), 300.0),
            )
            edge_bps = ((exit_price - price) / exit_price) * 10000.0 - costs - tax_hurdle
            completed.append(
                {
                    "position_round_trip_id": current.get("position_round_trip_id") or round_trip_id,
                    "account_label": account,
                    "symbol": symbol,
                    "exit_fill_price": exit_price,
                    "reentry_fill_price": price,
                    "quantity": min(quantity, _float(current.get("exit_quantity"), quantity)),
                    "completed_at_utc": timestamp.isoformat(),
                    "post_cost_edge_bps": round(edge_bps, 6),
                    "trading_cost_bps": costs,
                    "tax_hurdle_bps": tax_hurdle,
                    "edge_is_after_modeled_tax_hurdle": True,
                    "successful": edge_bps > 0.0,
                }
            )
            current = {"phase": "held", "failed_reentry_count": 0, "last_completed_at_utc": timestamp.isoformat()}
        elif leg == "reentry_failed":
            failures = int(current.get("failed_reentry_count", 0) or 0) + 1
            current["failed_reentry_count"] = failures
            current["last_reentry_failure_at_utc"] = timestamp.isoformat()
            if failures >= int(_float(policy.get("maximum_failed_reentries"), 2)):
                current["phase"] = "paper_reentry_failed_cooldown"
                current["cooldown_until_utc"] = (
                    timestamp + timedelta(hours=_float(policy.get("failed_reentry_cooldown_hours"), 72.0))
                ).isoformat()
        positions[key] = current
        processed.add(event_id)
    updated["positions"] = positions
    updated["completed_round_trips"] = completed[-1000:]
    updated["processed_event_ids"] = sorted(processed)[-5000:]
    return updated


def _paper_proof(state: dict[str, Any], policy: dict[str, Any]) -> dict[str, Any]:
    completed = [_dict(row) for row in _list(state.get("completed_round_trips"))]
    edges = [_float(row.get("post_cost_edge_bps"), 0.0) for row in completed]
    successes = sum(1 for row in completed if bool(row.get("successful", False)))
    rate = successes / len(completed) if completed else 0.0
    median_edge = statistics.median(edges) if edges else 0.0
    minimum_count = int(_float(policy.get("minimum_completed_paper_round_trips"), 50))
    minimum_rate = _float(policy.get("minimum_successful_paper_round_trip_rate"), 0.55)
    minimum_edge = _float(policy.get("minimum_median_post_cost_edge_bps"), 10.0)
    ready = len(completed) >= minimum_count and rate >= minimum_rate and median_edge >= minimum_edge
    return {
        "completed_round_trips": len(completed),
        "successful_round_trips": successes,
        "success_rate": round(rate, 6),
        "median_post_cost_edge_bps": round(median_edge, 6),
        "minimum_completed_round_trips": minimum_count,
        "minimum_success_rate": minimum_rate,
        "minimum_median_post_cost_edge_bps": minimum_edge,
        "paper_promotion_evidence_ready": ready,
        "live_execution_allowed": False,
    }


def evaluate(
    study: dict[str, Any],
    charts: dict[str, Any],
    *,
    risk: dict[str, Any],
    tax_estimate: dict[str, Any],
    tax_profile: dict[str, Any],
    dividend_calendar: dict[str, Any],
    state: dict[str, Any],
    paper_events: list[dict[str, Any]],
    policy: dict[str, Any],
    now: datetime | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    timestamp = _utc(now)
    treatments = _account_tax_treatments(tax_profile, {})
    updated_state = _apply_paper_events(
        state,
        paper_events,
        policy=policy,
        treatments=treatments,
        tax_estimate=tax_estimate,
    )
    positions_state = _dict(updated_state.get("positions"))
    inventory = _position_inventory(study, policy)
    chart_rows = _dict(charts.get("symbols"))
    exposures = _symbol_exposure(study)
    portfolio = _dict(study.get("portfolio_summary"))
    portfolio_value = max(
        _float(portfolio.get("liquidation_value"), 0.0),
        _float(portfolio.get("equity"), 0.0),
        1.0,
    )
    max_symbol_share = _float(_dict(risk.get("limits")).get("max_single_symbol_share"), 0.15)
    observations: list[dict[str, Any]] = []
    candidate_count = 0

    for row in inventory:
        account = str(row["account_label"])
        symbol = str(row["symbol"])
        key = f"{account}|{symbol}"
        lifecycle = dict(_dict(positions_state.get(key)))
        phase = str(lifecycle.get("phase") or "held")
        metrics = _chart_metrics(_dict(chart_rows.get(symbol)), policy=policy, now=timestamp)
        price = _float(metrics.get("last_price"), 0.0)
        average = _float(row.get("average_price"), 0.0)
        free_quantity = _float(row.get("unencumbered_quantity"), 0.0)
        concentration_share = exposures.get(symbol, 0.0) / portfolio_value
        friction = _tax_friction(
            account,
            treatments=treatments,
            tax_estimate=tax_estimate,
            policy=policy,
        )
        minimum_discount = max(
            _float(policy.get("minimum_reentry_discount_fraction"), 0.015),
            (
                _float(policy.get("round_trip_cost_bps"), 24.0)
                + _float(friction.get("tax_friction_bps"), 0.0)
            )
            / 10000.0,
        )
        dividend_guard = _dividend_guard(dividend_calendar, symbol, policy=policy, now=timestamp)
        model_stance = _model_stance(study, symbol)
        exit_signal = _exit_score(
            metrics,
            average_price=average,
            concentration_share=concentration_share,
            model_stance=model_stance,
            policy=policy,
        ) if metrics.get("ready") else {"score": 0.0, "components": {}}
        reentry_signal = None
        action = "HOLD"
        state_name = "observation"
        reasons: list[str] = []
        hard_holds: list[str] = []
        target_exit_zone = None
        target_reentry_zone = None

        max_symbol_value = portfolio_value * max_symbol_share
        exited_value = free_quantity * price
        remaining_symbol_value = max(exposures.get(symbol, 0.0) - exited_value, 0.0)
        reentry_headroom_value = max(max_symbol_value - remaining_symbol_value, 0.0)
        reentry_quantity_cap = reentry_headroom_value / price if price > 0.0 else 0.0
        planned_quantity_cap = min(free_quantity, reentry_quantity_cap)

        if not metrics.get("ready"):
            hard_holds.append("chart_evidence_missing_or_stale")
        if free_quantity <= 0.0:
            hard_holds.append("all_equity_reserved_for_short_call_coverage")
        if dividend_guard.get("blocked"):
            hard_holds.append("dividend_ex_date_blackout")

        cooldown_until = _parse_datetime(lifecycle.get("cooldown_until_utc"))
        if phase == "paper_reentry_failed_cooldown" and cooldown_until is not None:
            if timestamp < cooldown_until:
                hard_holds.append("failed_reentry_cooldown_active")
            else:
                phase = "held"
                lifecycle = {"phase": "held", "failed_reentry_count": 0}

        if phase.startswith("paper_exited"):
            exit_price = _float(lifecycle.get("exit_fill_price"), 0.0)
            exit_time = _parse_datetime(lifecycle.get("exit_filled_at_utc"))
            settlement = _parse_date(lifecycle.get("settlement_date"))
            wash_end = None
            if exit_price > 0.0 and average > 0.0 and exit_price < average and friction.get("tax_treatment") != "tax_advantaged":
                wash_end = (exit_time.date() + timedelta(days=int(_float(policy.get("wash_sale_window_days"), 30)))) if exit_time else None
            earliest = settlement
            if wash_end is not None and (earliest is None or wash_end > earliest):
                earliest = wash_end
            lifecycle["earliest_reentry_date"] = earliest.isoformat() if earliest else None
            if earliest is not None and timestamp.date() < earliest:
                action = "WAIT_REENTRY"
                state_name = "paper_lifecycle_hold"
                reasons.append("settlement_or_wash_sale_window_active")
            elif exit_price <= 0.0:
                hard_holds.append("exit_fill_evidence_missing")
            elif price > exit_price * (1.0 + _float(policy.get("maximum_reentry_chase_fraction"), 0.02)):
                action = "NO_CHASE"
                state_name = "paper_lifecycle_hold"
                reasons.append("price_above_no_chase_ceiling")
            else:
                reentry_signal = _reentry_score(
                    metrics,
                    exit_price=exit_price,
                    minimum_discount=minimum_discount,
                    policy=policy,
                ) if metrics.get("ready") else {"score": 0.0, "components": {}}
                target_reentry_zone = {
                    "upper_price": round(exit_price * (1.0 - minimum_discount), 4),
                    "support_reference": metrics.get("support_20d"),
                    "minimum_discount_fraction": round(minimum_discount, 8),
                }
                if planned_quantity_cap <= 0.0:
                    hard_holds.append("portfolio_concentration_blocks_reentry")
                elif _float(reentry_signal.get("score"), 0.0) >= _float(policy.get("reentry_score_threshold"), 0.70):
                    action = "PAPER_REENTRY_CANDIDATE"
                    state_name = "paper_candidate"
                    candidate_count += 1
                    reasons.append("reentry_score_and_discount_satisfied")
                else:
                    action = "WAIT_REENTRY"
                    state_name = "paper_lifecycle_watch"
                    reasons.append("reentry_conditions_not_yet_satisfied")

            if exit_time is not None and (timestamp - exit_time).days > int(_float(policy.get("maximum_reentry_wait_days"), 20)):
                if not lifecycle.get("timeout_recorded_at_utc"):
                    failures = int(lifecycle.get("failed_reentry_count", 0) or 0) + 1
                    lifecycle["failed_reentry_count"] = failures
                    lifecycle["timeout_recorded_at_utc"] = timestamp.isoformat()
                    if failures >= int(_float(policy.get("maximum_failed_reentries"), 2)):
                        lifecycle["phase"] = "paper_reentry_failed_cooldown"
                        lifecycle["cooldown_until_utc"] = (
                            timestamp + timedelta(hours=_float(policy.get("failed_reentry_cooldown_hours"), 72.0))
                        ).isoformat()
                hard_holds.append("maximum_reentry_wait_exceeded")
        else:
            target_exit_zone = {
                "resistance_reference": metrics.get("resistance_20d"),
                "last_price": metrics.get("last_price"),
                "score_threshold": _float(policy.get("exit_score_threshold"), 0.68),
            }
            if planned_quantity_cap <= 0.0 and free_quantity > 0.0:
                hard_holds.append("reentry_not_feasible_under_portfolio_concentration_limit")
            if not hard_holds and _float(exit_signal.get("score"), 0.0) >= _float(policy.get("exit_score_threshold"), 0.68):
                action = "PAPER_EXIT_CANDIDATE"
                state_name = "paper_candidate"
                candidate_count += 1
                reasons.append("exit_score_satisfied")
            elif not reasons:
                reasons.append("exit_conditions_not_satisfied")

        if hard_holds:
            action = "HOLD"
            state_name = "blocked"
        lifecycle["phase"] = phase if lifecycle.get("phase") is None else lifecycle.get("phase")
        lifecycle["last_evaluated_at_utc"] = timestamp.isoformat()
        positions_state[key] = lifecycle
        expected_gain = max((price - average) * free_quantity, 0.0) if price > 0.0 and average > 0.0 else 0.0
        tax_evidence_status = str(friction.get("evidence_status") or "")
        estimated_tax_reserve = (
            expected_gain * (_float(friction.get("tax_friction_bps"), 0.0) / 10000.0)
            if tax_evidence_status in {"derived_from_current_tax_estimate", "verified_tax_advantaged"}
            else None
        )
        tax_uncertainty_hurdle = exited_value * (_float(friction.get("tax_friction_bps"), 0.0) / 10000.0)
        observations.append(
            {
                "account_label": account,
                "symbol": symbol,
                "state": state_name,
                "action": action,
                "reasons": reasons,
                "hard_holds": sorted(set(hard_holds)),
                "lifecycle": lifecycle,
                "position": {
                    "equity_quantity": round(_float(row.get("equity_quantity"), 0.0), 6),
                    "covered_call_reserved_quantity": round(_float(row.get("covered_call_reserved_quantity"), 0.0), 6),
                    "maximum_unencumbered_quantity": round(free_quantity, 6),
                    "average_price": round(average, 6),
                },
                "chart": metrics,
                "exit_signal": exit_signal,
                "reentry_signal": reentry_signal,
                "zones": {"exit": target_exit_zone, "reentry": target_reentry_zone},
                "tax_and_cost": {
                    **friction,
                    "round_trip_cost_bps": _float(policy.get("round_trip_cost_bps"), 24.0),
                    "minimum_required_reentry_discount_fraction": round(minimum_discount, 8),
                    "estimated_exit_tax_reserve_usd": (
                        round(estimated_tax_reserve, 2) if estimated_tax_reserve is not None else None
                    ),
                    "policy_tax_uncertainty_hurdle_usd": round(tax_uncertainty_hurdle, 2),
                    "reserve_is_not_amount_owed": True,
                    "verified_tax_estimate_required_before_live": True,
                },
                "dividend_guard": dividend_guard,
                "portfolio_guard": {
                    "current_symbol_share": round(concentration_share, 8),
                    "maximum_symbol_share": max_symbol_share,
                    "maximum_reentry_quantity_under_current_limit": round(max(reentry_quantity_cap, 0.0), 6),
                    "planned_round_trip_quantity_cap": round(max(planned_quantity_cap, 0.0), 6),
                },
                "execution_contract": {
                    "direct_intent_publish_allowed": False,
                    "quantity_recommendation": None,
                    "paper_candidate_only": True,
                    "required_route": "portfolio_allocator_to_risk_service_to_standard_paper_gateway",
                    "live_execution_allowed": False,
                },
            }
        )

    updated_state["positions"] = positions_state
    updated_state["timestamp_utc"] = timestamp.isoformat()
    proof = _paper_proof(updated_state, policy)
    stale_count = sum(1 for row in observations if not _dict(row.get("chart")).get("ready"))
    blocked_count = sum(1 for row in observations if row.get("state") == "blocked")
    payload = {
        "timestamp_utc": timestamp.isoformat(),
        "schema_version": 1,
        "ok": bool(study.get("ok", False)),
        "overall_status": "ready" if bool(study.get("ok", False)) and stale_count == 0 else "needs_evidence",
        "mode": "guarded_stateful_advisory_paper_only",
        "position_count": len(observations),
        "candidate_count": candidate_count,
        "blocked_count": blocked_count,
        "stale_chart_count": stale_count,
        "observations": observations,
        "paper_proof": proof,
        "safety_contract": {
            "live_execution_allowed": False,
            "direct_order_or_intent_publish_allowed": False,
            "covered_call_shares_reserved_before_exit": True,
            "tax_and_wash_sale_friction_applied": True,
            "settlement_enforced": True,
            "concentration_checked_before_exit_and_reentry": True,
            "failed_reentry_cooldown_enforced": True,
            "paper_proof_required": True,
        },
        "evidence_gaps": {
            "tax_profile_status": tax_estimate.get("status"),
            "dividend_calendar_available": bool(_list(dividend_calendar.get("events"))),
            "paper_promotion_evidence_ready": proof["paper_promotion_evidence_ready"],
        },
    }
    return payload, updated_state


def _normalize_candles(payload: Any) -> list[dict[str, Any]]:
    rows = _list(_dict(payload).get("candles"))
    output: list[dict[str, Any]] = []
    for row in rows:
        item = _dict(row)
        close = _number(item.get("close"))
        if close is None or close <= 0.0:
            continue
        output.append(
            {
                "datetime": item.get("datetime"),
                "open": _number(item.get("open")),
                "high": _number(item.get("high")),
                "low": _number(item.get("low")),
                "close": close,
                "volume": _number(item.get("volume")),
            }
        )
    return output


def refresh_chart_cache(
    symbols: list[str],
    existing: dict[str, Any],
    *,
    policy: dict[str, Any],
    now: datetime | None = None,
) -> dict[str, Any]:
    timestamp = _utc(now)
    rows = dict(_dict(existing.get("symbols")))
    minimum_interval = _float(policy.get("chart_refresh_min_interval_seconds"), 240.0)
    due: list[str] = []
    for symbol in symbols[: int(_float(policy.get("maximum_symbols_per_refresh"), 30))]:
        fetched = _parse_datetime(_dict(rows.get(symbol)).get("fetched_at_utc"))
        age = (timestamp - fetched).total_seconds() if fetched is not None else float("inf")
        if age >= minimum_interval:
            due.append(symbol)
    if not due:
        return {**existing, "timestamp_utc": timestamp.isoformat(), "refresh_status": "cache_fresh"}

    access = provider_access_status(PROJECT_ROOT, "schwab")
    if access.get("active"):
        return {
            **existing,
            "timestamp_utc": timestamp.isoformat(),
            "refresh_status": "provider_cooldown_active_previous_cache_preserved",
            "refresh_errors": [{"symbol": symbol, "error": "schwab_provider_cooldown_active"} for symbol in due],
        }

    old_env = {
        "ALLOW_ORDER_EXECUTION": os.environ.get("ALLOW_ORDER_EXECUTION"),
        "MARKET_DATA_ONLY": os.environ.get("MARKET_DATA_ONLY"),
        "SCHWAB_AUTH_INTERACTIVE": os.environ.get("SCHWAB_AUTH_INTERACTIVE"),
    }
    os.environ["ALLOW_ORDER_EXECUTION"] = "0"
    os.environ["MARKET_DATA_ONLY"] = "1"
    os.environ["SCHWAB_AUTH_INTERACTIVE"] = "0"
    errors: list[dict[str, Any]] = []
    try:
        trader = build_schwab_trader(PROJECT_ROOT, mode="shadow")
        with open(os.devnull, "w", encoding="utf-8") as devnull:
            with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
                client = trader.authenticate()
        for symbol in due:
            try:
                with provider_request_slot(PROJECT_ROOT, "schwab", symbol, slot_count=2, wait_seconds=20.0):
                    daily_response = client.get_price_history_every_day(
                        symbol,
                        start_datetime=timestamp - timedelta(days=260),
                        end_datetime=timestamp,
                        need_extended_hours_data=False,
                        need_previous_close=True,
                    )
                    intraday_response = client.get_price_history_every_five_minutes(
                        symbol,
                        start_datetime=timestamp - timedelta(days=9),
                        end_datetime=timestamp,
                        need_extended_hours_data=True,
                        need_previous_close=True,
                    )
                    daily_payload = resp_json(daily_response)
                    intraday_payload = resp_json(intraday_response)
                daily = _normalize_candles(daily_payload)
                intraday = _normalize_candles(intraday_payload)
                if len(daily) < int(_float(policy.get("minimum_daily_bars"), 50)):
                    raise RuntimeError(f"insufficient_daily_bars:{len(daily)}")
                if len(intraday) < int(_float(policy.get("minimum_intraday_bars"), 24)):
                    raise RuntimeError(f"insufficient_intraday_bars:{len(intraday)}")
                rows[symbol] = {
                    "symbol": symbol,
                    "fetched_at_utc": timestamp.isoformat(),
                    "provider": "schwab",
                    "daily_bars": daily[-int(_float(policy.get("maximum_daily_bars_cached"), 260)):],
                    "intraday_bars": intraday[-int(_float(policy.get("maximum_intraday_bars_cached"), 1200)):],
                }
                mark_provider_recovered(PROJECT_ROOT, "schwab", evidence=f"held_position_chart_refresh:{symbol}", force=True)
            except Exception as exc:
                message = f"{type(exc).__name__}:{exc}"
                status_code = provider_http_status_code(message)
                if status_code in {401, 403, 429}:
                    activate_provider_cooldown(
                        PROJECT_ROOT,
                        "schwab",
                        status_code=status_code,
                        reason=message,
                        symbol=symbol,
                        profile="position_round_trip_watch",
                        domain="held_position_charts",
                    )
                errors.append({"symbol": symbol, "error": message})
    except Exception as exc:
        errors.extend({"symbol": symbol, "error": f"{type(exc).__name__}:{exc}"} for symbol in due)
    finally:
        for key, value in old_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
    return {
        "timestamp_utc": timestamp.isoformat(),
        "schema_version": 1,
        "provider": "schwab",
        "symbols": rows,
        "refresh_status": "ready" if not errors else "partial_previous_good_cache_preserved",
        "refresh_errors": errors,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate guarded paper-only exits and re-entries for held positions.")
    parser.add_argument("--study-file", default=str(DEFAULT_STUDY_PATH))
    parser.add_argument("--chart-file", default=str(DEFAULT_CHART_PATH))
    parser.add_argument("--risk-file", default=str(DEFAULT_RISK_PATH))
    parser.add_argument("--tax-estimate-file", default=str(DEFAULT_TAX_ESTIMATE_PATH))
    parser.add_argument("--tax-profile-file", default=str(DEFAULT_TAX_PROFILE_PATH))
    parser.add_argument("--dividend-file", default=str(DEFAULT_DIVIDEND_PATH))
    parser.add_argument("--policy-file", default=str(DEFAULT_POLICY_PATH))
    parser.add_argument("--state-file", default=str(DEFAULT_STATE_PATH))
    parser.add_argument("--events-file", default=str(DEFAULT_EVENTS_PATH))
    parser.add_argument("--out-file", default=str(DEFAULT_OUT_PATH))
    parser.add_argument("--refresh-market-data", action="store_true")
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    study = load_json(Path(args.study_file))
    policy = load_json(Path(args.policy_file))
    chart_path = Path(args.chart_file)
    charts = load_json(chart_path)
    symbols = sorted({
        str(row.get("underlying") or row.get("symbol") or "").strip().upper()
        for row in _list(study.get("positions"))
        if str(row.get("asset_type") or "").strip().upper() == "EQUITY"
    })
    if args.refresh_market_data:
        charts = refresh_chart_cache(symbols, charts, policy=policy)
        write_payload(chart_path, charts)
    payload, updated_state = evaluate(
        study,
        charts,
        risk=load_json(Path(args.risk_file)),
        tax_estimate=load_json(Path(args.tax_estimate_file)),
        tax_profile=load_json(Path(args.tax_profile_file)),
        dividend_calendar=load_json(Path(args.dividend_file)),
        state=load_json(Path(args.state_file)),
        paper_events=_read_events(Path(args.events_file)),
        policy=policy,
    )
    write_payload(Path(args.state_file), updated_state)
    write_payload(Path(args.out_file), payload)
    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        print(
            "position_round_trip_watch "
            f"status={payload.get('overall_status')} "
            f"positions={payload.get('position_count', 0)} "
            f"candidates={payload.get('candidate_count', 0)} "
            f"blocked={payload.get('blocked_count', 0)}"
        )
    return 0 if bool(payload.get("ok", False)) else 2


if __name__ == "__main__":
    raise SystemExit(main())
