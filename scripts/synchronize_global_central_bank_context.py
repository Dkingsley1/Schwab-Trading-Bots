#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gzip
import json
import math
import os
import sys
import tempfile
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.central_bank_liquidity import assess_central_bank_liquidity_context
from core.global_central_bank_context import (
    CENTRAL_BANK_CROSS_SOURCE_FEATURE_KEYS,
    assess_central_bank_cross_source_context,
    assess_global_central_bank_context,
)


EXTERNAL_ROOT = PROJECT_ROOT / "exports" / "external_context"
LATEST_PATH = EXTERNAL_ROOT / "central_bank_cross_source_latest.json"
HEALTH_PATH = PROJECT_ROOT / "governance" / "health" / "central_bank_cross_source_sync_latest.json"
HISTORY_ROOT = PROJECT_ROOT / "data" / "external_context" / "global_central_bank_history"
SOURCE_MAX_AGE_HOURS = {
    "global_central_bank_context": 48.0,
    "fx_market_context": 24.0,
    "public_policy_context": 48.0,
    "official_macro_context": 24.0,
    "macro_cross_asset": 24.0,
    "central_bank_liquidity": 24.0,
}
PAIR_BY_CURRENCY = {
    "EUR": ("EURUSD", 1.0),
    "JPY": ("USDJPY", -1.0),
    "GBP": ("GBPUSD", 1.0),
    "CHF": ("USDCHF", -1.0),
    "CAD": ("USDCAD", -1.0),
    "AUD": ("AUDUSD", 1.0),
}
CURRENCY_SYMBOLS = {
    "USD": ["UUP", "SPY", "QQQ", "IWM", "DIA", "TLT", "IEF", "LQD", "HYG"],
    "EUR": ["FXE", "EURUSD"],
    "JPY": ["FXY", "USDJPY"],
    "GBP": ["FXB", "GBPUSD"],
    "CHF": ["FXF", "USDCHF"],
    "CAD": ["FXC", "USDCAD"],
    "AUD": ["FXA", "AUDUSD"],
    "NZD": ["NZDUSD"],
    "CNY": ["CYB", "USDCNY", "USDCNH"],
    "INR": ["INDA", "USDINR"],
    "BRL": ["EWZ", "USDBRL"],
    "MXN": ["EWW", "USDMXN"],
    "KRW": ["EWY", "USDKRW"],
    "NOK": ["ENOR", "USDNOK"],
    "SEK": ["EWD", "USDSEK"],
    "SGD": ["EWS", "USDSGD"],
    "HKD": ["EWH", "USDHKD"],
    "IDR": ["EIDO", "USDIDR"],
    "ZAR": ["EZA", "USDZAR"],
    "TRY": ["TUR", "USDTRY"],
    "SAR": ["KSA", "USDSAR"],
    "RUB": ["USDRUB"],
    "ILS": ["EIS", "USDILS"],
    "MYR": ["EWM", "USDMYR"],
    "THB": ["THD", "USDTHB"],
    "PHP": ["EPHE", "USDPHP"],
    "PLN": ["EPOL", "USDPLN"],
    "DKK": ["EDEN", "USDDKK"],
    "CZK": ["USDCZK"],
    "CLP": ["ECH", "USDCLP"],
    "AED": ["UAE", "USDAED"],
    "ARS": ["ARGT", "USDARS"],
}
BANK_ALIASES = {
    "federal_reserve": ("federal reserve", "fomc", " fed "),
    "european_central_bank": ("european central bank", "ecb"),
    "peoples_bank_of_china": ("people's bank of china", "peoples bank of china", "pboc"),
    "bank_of_japan": ("bank of japan", "boj"),
    "bank_of_england": ("bank of england", "boe"),
    "bank_of_canada": ("bank of canada", "boc"),
    "swiss_national_bank": ("swiss national bank", "snb"),
    "reserve_bank_of_australia": ("reserve bank of australia", "rba"),
    "reserve_bank_of_new_zealand": ("reserve bank of new zealand", "rbnz"),
    "reserve_bank_of_india": ("reserve bank of india", "rbi"),
    "central_bank_of_brazil": ("central bank of brazil", "copom"),
    "bank_of_mexico": ("bank of mexico", "banxico"),
    "bank_of_korea": ("bank of korea", "bok"),
    "norges_bank": ("norges bank",),
    "sveriges_riksbank": ("sveriges riksbank", "riksbank"),
    "monetary_authority_of_singapore": ("monetary authority of singapore", "mas"),
    "hong_kong_monetary_authority": ("hong kong monetary authority", "hkma"),
    "bank_indonesia": ("bank indonesia",),
    "south_african_reserve_bank": ("south african reserve bank", "sarb"),
    "central_bank_of_turkiye": ("central bank of the republic of turkiye", "central bank of turkiye", "cbrt"),
    "saudi_central_bank": ("saudi central bank", "sama"),
    "bank_of_russia": ("bank of russia", "central bank of russia"),
    "bank_of_israel": ("bank of israel",),
    "bank_negara_malaysia": ("bank negara malaysia", "bnm"),
    "bank_of_thailand": ("bank of thailand", "bot"),
    "bangko_sentral_ng_pilipinas": ("bangko sentral ng pilipinas", "bsp"),
    "national_bank_of_poland": ("national bank of poland", "narodowy bank polski", "nbp"),
    "danmarks_nationalbank": ("danmarks nationalbank",),
    "czech_national_bank": ("czech national bank", "cnb"),
    "central_bank_of_chile": ("central bank of chile", "banco central de chile"),
    "central_bank_of_the_uae": ("central bank of the uae", "central bank of the united arab emirates", "cbuae"),
    "central_bank_of_argentina": ("central bank of argentina", "banco central de la republica argentina", "bcra"),
}


def _clamp01(value: float) -> float:
    return max(0.0, min(float(value), 1.0))


def _safe_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _signed_norm(value: float, scale: float) -> float:
    return _clamp01(0.5 + float(value) / (2.0 * max(abs(float(scale)), 1e-9)))


def _parse_ts(value: Any) -> datetime | None:
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


def _parse_observation_time(value: Any) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    if len(text) == 4 and text.isdigit():
        return datetime(int(text), 1, 1, tzinfo=timezone.utc)
    if len(text) == 7 and text[4:6] == "-Q" and text[-1] in "1234":
        month = int(text[-1]) * 3
        return datetime(int(text[:4]), month, 1, tzinfo=timezone.utc)
    parsed = _parse_ts(text)
    if parsed is not None:
        return parsed
    try:
        parsed_date = date.fromisoformat(text[:10])
    except ValueError:
        return None
    return datetime.combine(parsed_date, datetime.min.time(), tzinfo=timezone.utc)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _source_state(source_id: str, payload: Mapping[str, Any], now: datetime) -> dict[str, Any]:
    timestamp = _parse_ts(payload.get("timestamp_utc"))
    age_hours = (now - timestamp).total_seconds() / 3600.0 if timestamp is not None else None
    maximum_age = float(SOURCE_MAX_AGE_HOURS[source_id])
    future = bool(age_hours is not None and age_hours < -0.1)
    fresh = bool(age_hours is not None and not future and age_hours <= maximum_age)
    freshness = _clamp01(1.0 - max(float(age_hours or 0.0), 0.0) / maximum_age) if fresh else 0.0
    return {
        "source_id": source_id,
        "timestamp_utc": timestamp.isoformat() if timestamp is not None else None,
        "age_hours": round(age_hours, 6) if age_hours is not None else None,
        "maximum_age_hours": maximum_age,
        "fresh": fresh,
        "timestamp_in_future": future,
        "freshness_norm": freshness,
    }


def _lineage(
    *,
    source_state: Mapping[str, Any],
    dimension: str,
    field_path: str,
    observation_time: Any,
    source_url: str,
    confidence: float,
    as_of: datetime,
    publisher_url: str = "",
) -> dict[str, Any]:
    observation = _parse_observation_time(observation_time)
    observation_in_future = bool(observation is not None and observation > as_of)
    observation_time_valid = observation is not None
    return {
        "dimension": dimension,
        "source_id": source_state.get("source_id"),
        "artifact_timestamp_utc": source_state.get("timestamp_utc"),
        "observation_time": str(observation_time or ""),
        "field_path": field_path,
        "source_url": source_url,
        "publisher_url": publisher_url,
        "source_confidence_norm": _clamp01(confidence),
        "freshness_norm": float(source_state.get("freshness_norm", 0.0) or 0.0),
        "observation_time_valid": observation_time_valid,
        "observation_in_future": observation_in_future,
        "point_in_time": bool(
            source_state.get("fresh", False)
            and not source_state.get("timestamp_in_future", False)
            and observation_time_valid
            and not observation_in_future
        ),
    }


def _world_bank_fields(public_policy: Mapping[str, Any], country: str, *, as_of: date) -> tuple[dict[str, Any], list[str]]:
    sources = public_policy.get("sources") if isinstance(public_policy.get("sources"), Mapping) else {}
    world_bank = sources.get("world_bank_indicators") if isinstance(sources.get("world_bank_indicators"), Mapping) else {}
    indicators = world_bank.get("indicators") if isinstance(world_bank.get("indicators"), Mapping) else {}
    values: dict[str, Any] = {}
    future_fields: list[str] = []
    for feature_name, raw in indicators.items():
        if not isinstance(raw, Mapping):
            continue
        country_values = raw.get("values") if isinstance(raw.get("values"), Mapping) else {}
        row = country_values.get(country) if isinstance(country_values.get(country), Mapping) else {}
        value = _safe_float(row.get("value"))
        period = str(row.get("date") or "")
        if value is None:
            continue
        if period[:4].isdigit() and int(period[:4]) > as_of.year:
            future_fields.append(str(feature_name))
            continue
        values[str(feature_name)] = {
            "value": value,
            "period": period,
            "country_name": row.get("country_name"),
            "cached": bool(row.get("cached", False)),
        }
    return values, future_fields


def _currency_context(fx: Mapping[str, Any], currency: str) -> dict[str, Any]:
    derived = fx.get("derived") if isinstance(fx.get("derived"), Mapping) else {}
    reference = derived.get("currency_reference_rates") if isinstance(derived.get("currency_reference_rates"), Mapping) else {}
    reference_changes = (
        derived.get("currency_reference_changes")
        if isinstance(derived.get("currency_reference_changes"), Mapping)
        else {}
    )
    row = reference.get(currency) if isinstance(reference.get(currency), Mapping) else {}
    rate = _safe_float(row.get("units_per_eur"))
    raw_change = _safe_float(reference_changes.get(currency))
    if rate is not None:
        return {
            "currency": currency,
            "reference": "units_per_eur",
            "reference_rate": rate,
            "reference_date": row.get("date"),
            "currency_strength_change": -raw_change if raw_change is not None else None,
            "raw_reference_rate_change": raw_change,
            "source": "ecb_reference_rates",
        }

    pair_values = derived.get("pair_values") if isinstance(derived.get("pair_values"), Mapping) else {}
    pair_changes = derived.get("pair_changes") if isinstance(derived.get("pair_changes"), Mapping) else {}
    pair, sign = PAIR_BY_CURRENCY.get(currency, ("", 0.0))
    value = _safe_float(pair_values.get(pair)) if pair else None
    change = _safe_float(pair_changes.get(pair)) if pair else None
    if value is None:
        return {}
    return {
        "currency": currency,
        "reference": pair,
        "reference_rate": value,
        "reference_date": fx.get("timestamp_utc"),
        "currency_strength_change": change * sign if change is not None else None,
        "raw_reference_rate_change": change,
        "source": "canonical_fx_pair",
    }


def _event_rows_for_bank(official_macro: Mapping[str, Any], bank_id: str, *, as_of: datetime) -> list[dict[str, Any]]:
    derived = official_macro.get("derived") if isinstance(official_macro.get("derived"), Mapping) else {}
    rows = []
    for key in ("calendar_rows", "news_rows"):
        source_rows = derived.get(key) if isinstance(derived.get(key), list) else []
        for raw in source_rows:
            if not isinstance(raw, Mapping):
                continue
            text = " ".join(
                str(raw.get(field) or "") for field in ("title", "headline", "summary", "event", "source")
            ).lower()
            aliases = BANK_ALIASES.get(bank_id, ())
            if not aliases or not any(alias in f" {text} " for alias in aliases):
                continue
            event_ts = _parse_ts(raw.get("datetime") or raw.get("published") or raw.get("date"))
            if event_ts is not None and event_ts > as_of:
                continue
            rows.append(dict(raw))
    return rows[:20]


def _mean(values: Iterable[float]) -> float | None:
    clean = [float(value) for value in values if math.isfinite(float(value))]
    return sum(clean) / len(clean) if clean else None


def _weighted_mean(rows: Iterable[tuple[float, float]]) -> float | None:
    clean = [(float(value), max(float(weight), 0.0)) for value, weight in rows if math.isfinite(float(value))]
    weight = sum(item[1] for item in clean)
    return sum(value * item_weight for value, item_weight in clean) / weight if clean and weight > 0.0 else None


def _bank_feature_map(bank: Mapping[str, Any]) -> dict[str, float]:
    derived = bank.get("derived_features") if isinstance(bank.get("derived_features"), Mapping) else {}
    usable = bank.get("usable_features") if isinstance(bank.get("usable_features"), Mapping) else {}
    mapping = {
        "policy_fx_confirmation_norm": "central_bank_policy_fx_confirmation_norm",
        "policy_inflation_alignment_norm": "central_bank_policy_inflation_alignment_norm",
        "external_balance_alignment_norm": "central_bank_policy_external_balance_alignment_norm",
        "policy_liquidity_alignment_norm": "central_bank_policy_liquidity_alignment_norm",
        "cross_asset_confirmation_norm": "central_bank_policy_cross_asset_confirmation_norm",
        "policy_divergence_signal_norm": "central_bank_policy_divergence_signal_norm",
        "spillover_risk_norm": "central_bank_policy_spillover_risk_norm",
    }
    out = {
        "central_bank_sync_available_norm": 1.0 if bool(bank.get("synchronized_ready", False)) else 0.0,
        "central_bank_sync_coverage_norm": 1.0 if bool(bank.get("synchronized_ready", False)) else 0.0,
        "central_bank_sync_point_in_time_norm": _clamp01(float(bank.get("lineage_point_in_time_ratio", 0.0) or 0.0)),
        "central_bank_sync_lineage_coverage_norm": _clamp01(float(bank.get("lineage_point_in_time_ratio", 0.0) or 0.0)),
        "central_bank_sync_fx_coverage_norm": 1.0 if "fx_context" in usable else 0.0,
        "central_bank_sync_macro_coverage_norm": 1.0 if "sovereign_macro" in usable else 0.0,
        "central_bank_sync_liquidity_coverage_norm": 1.0 if "global_usd_liquidity" in usable else 0.0,
        "central_bank_sync_conflict_free_norm": 0.0 if list(bank.get("hard_conflicts") or []) else 1.0,
        "central_bank_sync_freshness_norm": _clamp01(float(bank.get("cross_source_confidence_norm", 0.0) or 0.0)),
    }
    for source_key, output_key in mapping.items():
        value = _safe_float(derived.get(source_key))
        if value is not None:
            out[output_key] = _clamp01(value)
    return out


def build_central_bank_cross_source_context(
    *,
    global_central_banks: Mapping[str, Any],
    fx_market: Mapping[str, Any],
    public_policy: Mapping[str, Any],
    official_macro: Mapping[str, Any],
    macro_cross_asset: Mapping[str, Any],
    central_bank_liquidity: Mapping[str, Any],
    now: datetime,
) -> dict[str, Any]:
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    now = now.astimezone(timezone.utc)
    as_of = now.date()
    source_payloads = {
        "global_central_bank_context": global_central_banks,
        "fx_market_context": fx_market,
        "public_policy_context": public_policy,
        "official_macro_context": official_macro,
        "macro_cross_asset": macro_cross_asset,
        "central_bank_liquidity": central_bank_liquidity,
    }
    source_states = {source_id: _source_state(source_id, payload, now) for source_id, payload in source_payloads.items()}
    global_assessment = assess_global_central_bank_context(global_central_banks, now_utc=now)
    liquidity_assessment = assess_central_bank_liquidity_context(central_bank_liquidity, now_utc=now)
    raw_banks = global_central_banks.get("banks") if isinstance(global_central_banks.get("banks"), Mapping) else {}
    raw_global_features = (
        global_central_banks.get("global_features")
        if isinstance(global_central_banks.get("global_features"), Mapping)
        else {}
    )
    macro_cross = macro_cross_asset.get("cross_asset") if isinstance(macro_cross_asset.get("cross_asset"), Mapping) else {}
    liquidity_features = (
        central_bank_liquidity.get("global_features")
        if isinstance(central_bank_liquidity.get("global_features"), Mapping)
        else {}
    )
    fx_derived = fx_market.get("derived") if isinstance(fx_market.get("derived"), Mapping) else {}
    canonical_fx = fx_derived.get("canonical_reconciliation") if isinstance(fx_derived.get("canonical_reconciliation"), Mapping) else {}

    synchronized: dict[str, dict[str, Any]] = {}
    future_exclusions: dict[str, list[str]] = {}
    hard_conflicts: list[str] = []
    soft_conflicts: list[str] = []
    weighted_ready = 0.0
    weighted_sync = 0.0
    policy_fx_scores: list[tuple[float, float]] = []
    inflation_scores: list[tuple[float, float]] = []
    external_balance_scores: list[tuple[float, float]] = []
    liquidity_scores: list[tuple[float, float]] = []
    cross_asset_scores: list[tuple[float, float]] = []
    lineage_ratios: list[tuple[float, float]] = []
    freshness_scores: list[float] = []
    fx_join_weight = 0.0
    macro_join_weight = 0.0
    liquidity_join_weight = 0.0

    usd_rate = None
    fed_row = raw_banks.get("federal_reserve") if isinstance(raw_banks.get("federal_reserve"), Mapping) else {}
    if isinstance(fed_row.get("policy_rate"), Mapping):
        usd_rate = _safe_float(fed_row["policy_rate"].get("rate_percent"))

    for bank_id, raw in raw_banks.items():
        if not isinstance(raw, Mapping):
            continue
        weight = max(float(raw.get("weight", 0.0) or 0.0), 0.0)
        if bool(raw.get("ready", False)):
            weighted_ready += weight
        links: list[dict[str, Any]] = []
        usable: dict[str, Any] = {}
        bank_hard_conflicts: list[str] = []
        bank_soft_conflicts: list[str] = []
        policy = raw.get("policy_rate") if isinstance(raw.get("policy_rate"), Mapping) else {}
        assets = raw.get("balance_sheet") if isinstance(raw.get("balance_sheet"), Mapping) else {}
        policy_rate = _safe_float(policy.get("rate_percent"))
        policy_impulse = _safe_float(policy.get("change_bps_90d"))
        if policy_rate is not None and bool(policy.get("fresh", False)):
            usable.update(
                {
                    "policy_rate_percent": policy_rate,
                    "policy_change_bps_30d": _safe_float(policy.get("change_bps_30d")),
                    "policy_change_bps_90d": policy_impulse,
                    "days_since_policy_change": policy.get("days_since_last_change"),
                }
            )
            links.append(
                _lineage(
                    source_state=source_states["global_central_bank_context"],
                    dimension="policy_rate",
                    field_path=f"banks.{bank_id}.policy_rate",
                    observation_time=policy.get("observation_date"),
                    source_url="https://data.bis.org/topics/CBPOL",
                    publisher_url=str(raw.get("official_policy_url") or ""),
                    confidence=0.99,
                    as_of=now,
                )
            )
        asset_impulse = _safe_float(assets.get("quarter_over_quarter_change_pct"))
        if _safe_float(assets.get("total_assets_usd_billions")) is not None and bool(assets.get("fresh", False)):
            usable.update(
                {
                    "central_bank_assets_usd_billions": _safe_float(assets.get("total_assets_usd_billions")),
                    "central_bank_assets_qoq_pct": asset_impulse,
                    "central_bank_assets_yoy_pct": _safe_float(assets.get("year_over_year_change_pct")),
                }
            )
            links.append(
                _lineage(
                    source_state=source_states["global_central_bank_context"],
                    dimension="balance_sheet",
                    field_path=f"banks.{bank_id}.balance_sheet",
                    observation_time=assets.get("observation_date"),
                    source_url="https://data.bis.org/topics/CBTA",
                    publisher_url=str(raw.get("official_policy_url") or ""),
                    confidence=0.97,
                    as_of=now,
                )
            )

        currency = str(raw.get("currency") or "").upper()
        currency_context = _currency_context(fx_market, currency) if source_states["fx_market_context"]["fresh"] else {}
        currency_strength = _safe_float(currency_context.get("currency_strength_change"))
        if currency_context:
            usable["fx_context"] = currency_context
            fx_join_weight += weight
            links.append(
                _lineage(
                    source_state=source_states["fx_market_context"],
                    dimension="fx_transmission",
                    field_path=f"derived.currency_reference_rates.{currency}",
                    observation_time=currency_context.get("reference_date"),
                    source_url="https://www.ecb.europa.eu/stats/policy_and_exchange_rates/euro_reference_exchange_rates/html/index.en.html",
                    confidence=0.95,
                    as_of=now,
                )
            )
        pair, _ = PAIR_BY_CURRENCY.get(currency, ("", 0.0))
        reconciliation = canonical_fx.get(pair) if pair and isinstance(canonical_fx.get(pair), Mapping) else {}
        severity = str(reconciliation.get("divergence_severity") or "none").lower()
        if severity in {"high", "critical"}:
            conflict = f"{bank_id}:fx_provider_divergence:{pair}"
            hard_conflicts.append(conflict)
            bank_hard_conflicts.append(conflict)
        elif severity not in {"", "none"}:
            conflict = f"{bank_id}:fx_provider_divergence:{pair}:{severity}"
            soft_conflicts.append(conflict)
            bank_soft_conflicts.append(conflict)

        country = str(raw.get("world_bank_area_code") or "").upper()
        macro_fields, future_fields = _world_bank_fields(public_policy, country, as_of=as_of)
        if future_fields:
            future_exclusions[bank_id] = future_fields
        if macro_fields and source_states["public_policy_context"]["fresh"]:
            usable["sovereign_macro"] = macro_fields
            macro_join_weight += weight
            for feature_name, row in macro_fields.items():
                links.append(
                    _lineage(
                        source_state=source_states["public_policy_context"],
                        dimension="sovereign_macro",
                        field_path=f"sources.world_bank_indicators.indicators.{feature_name}.values.{country}",
                        observation_time=row.get("period"),
                        source_url="https://api.worldbank.org/v2",
                        confidence=0.9 if bool(row.get("cached", False)) else 0.96,
                        as_of=now,
                    )
                )

        events = _event_rows_for_bank(official_macro, str(bank_id), as_of=now)
        if events and source_states["official_macro_context"]["fresh"]:
            usable["official_events"] = events
            links.append(
                _lineage(
                    source_state=source_states["official_macro_context"],
                    dimension="official_communications",
                    field_path="derived.calendar_rows|derived.news_rows",
                    observation_time=events[0].get("datetime") or events[0].get("published") or events[0].get("date"),
                    source_url=str(events[0].get("url") or events[0].get("link") or raw.get("official_policy_url") or ""),
                    publisher_url=str(raw.get("official_policy_url") or ""),
                    confidence=0.97,
                    as_of=now,
                )
            )

        if bool(liquidity_assessment.get("ready", False)) and source_states["central_bank_liquidity"]["fresh"]:
            usable["global_usd_liquidity"] = {
                key: liquidity_features.get(key)
                for key in (
                    "fed_net_liquidity_impulse_norm",
                    "fed_liquidity_tightening_norm",
                    "fed_funding_stress_norm",
                    "fed_central_bank_swap_usage_norm",
                )
                if key in liquidity_features
            }
            liquidity_join_weight += weight
            links.append(
                _lineage(
                    source_state=source_states["central_bank_liquidity"],
                    dimension="global_usd_liquidity",
                    field_path="global_features",
                    observation_time=central_bank_liquidity.get("timestamp_utc"),
                    source_url="https://fred.stlouisfed.org/",
                    confidence=0.98,
                    as_of=now,
                )
            )

        if macro_cross and source_states["macro_cross_asset"]["fresh"]:
            usable["cross_asset_context"] = {
                key: value for key, value in macro_cross.items() if _safe_float(value) is not None
            }
            links.append(
                _lineage(
                    source_state=source_states["macro_cross_asset"],
                    dimension="cross_asset_confirmation",
                    field_path="cross_asset",
                    observation_time=macro_cross_asset.get("timestamp_utc"),
                    source_url="https://fred.stlouisfed.org/",
                    confidence=0.96,
                    as_of=now,
                )
            )

        derived_features: dict[str, float] = {}
        policy_fx_confirmation = None
        if policy_impulse is not None and currency_strength is not None:
            if abs(policy_impulse) <= 0.5 or abs(currency_strength) <= 1e-6:
                policy_fx_confirmation = 0.5
            else:
                policy_fx_confirmation = 1.0 if policy_impulse * currency_strength > 0.0 else 0.0
            policy_fx_scores.append((policy_fx_confirmation, weight))
            derived_features["policy_fx_confirmation_norm"] = policy_fx_confirmation

        inflation_row = macro_fields.get("inflation_cpi_annual_pct") if isinstance(macro_fields.get("inflation_cpi_annual_pct"), Mapping) else {}
        inflation = _safe_float(inflation_row.get("value"))
        inflation_alignment = _signed_norm(policy_rate - inflation, 15.0) if policy_rate is not None and inflation is not None else None
        if inflation is not None and policy_rate is not None:
            inflation_scores.append((inflation_alignment, weight))
            derived_features["policy_inflation_alignment_norm"] = inflation_alignment

        current_account_row = macro_fields.get("current_account_pct_gdp") if isinstance(macro_fields.get("current_account_pct_gdp"), Mapping) else {}
        current_account = _safe_float(current_account_row.get("value"))
        external_balance_alignment = _signed_norm(current_account, 15.0) if current_account is not None else None
        if current_account is not None:
            external_balance_scores.append((external_balance_alignment, weight))
            derived_features["external_balance_alignment_norm"] = external_balance_alignment

        liquidity_impulse = _safe_float(liquidity_features.get("fed_net_liquidity_impulse_norm"))
        policy_liquidity_alignment = None
        if policy_impulse is not None and liquidity_impulse is not None and abs(policy_impulse) > 0.5:
            policy_easing = policy_impulse < 0.0
            liquidity_easing = liquidity_impulse > 0.5
            policy_liquidity_alignment = 1.0 if policy_easing == liquidity_easing else 0.0
            liquidity_scores.append((policy_liquidity_alignment, weight))
            derived_features["policy_liquidity_alignment_norm"] = policy_liquidity_alignment

        vix = _safe_float(macro_cross.get("vix"))
        high_yield = _safe_float(macro_cross.get("high_yield_oas_bps"))
        risk_stress = None
        stress_inputs = [value for value in (vix / 60.0 if vix is not None else None, high_yield / 10.0 if high_yield is not None else None) if value is not None]
        if stress_inputs:
            risk_stress = _clamp01(sum(stress_inputs) / len(stress_inputs))
        if policy_fx_confirmation is not None and risk_stress is not None:
            cross_asset_confirmation = _clamp01(0.5 + (policy_fx_confirmation - 0.5) * (1.0 - risk_stress))
            cross_asset_scores.append((cross_asset_confirmation, weight))
            derived_features["cross_asset_confirmation_norm"] = cross_asset_confirmation
        if policy_rate is not None and usd_rate is not None:
            rate_divergence = abs(policy_rate - usd_rate)
            policy_divergence_signal = _clamp01(rate_divergence / 10.0)
            derived_features["policy_divergence_signal_norm"] = policy_divergence_signal
            if risk_stress is not None:
                derived_features["spillover_risk_norm"] = _clamp01(
                    policy_divergence_signal * (0.5 + 0.5 * risk_stress)
                )

        future_link_dimensions = [
            str(link.get("dimension") or "unknown")
            for link in links
            if bool(link.get("observation_in_future", False))
        ]
        if future_link_dimensions:
            existing_future = future_exclusions.setdefault(str(bank_id), [])
            existing_future.extend(dimension for dimension in future_link_dimensions if dimension not in existing_future)
        point_in_time_links = [link for link in links if bool(link.get("point_in_time", False))]
        cross_source_links = [
            link
            for link in point_in_time_links
            if str(link.get("source_id") or "") != "global_central_bank_context"
        ]
        lineage_ratio = len(point_in_time_links) / max(len(links), 1)
        lineage_ratios.append((lineage_ratio, weight))
        confidence_values = [
            float(link.get("source_confidence_norm", 0.0) or 0.0) * float(link.get("freshness_norm", 0.0) or 0.0)
            for link in point_in_time_links
        ]
        join_confidence = _mean(confidence_values) or 0.0
        cross_source_confidence = _mean(
            float(link.get("source_confidence_norm", 0.0) or 0.0)
            * float(link.get("freshness_norm", 0.0) or 0.0)
            for link in cross_source_links
        ) or 0.0
        freshness_scores.extend(float(link.get("freshness_norm", 0.0) or 0.0) for link in point_in_time_links)
        synchronized_ready = bool(
            raw.get("ready", False)
            and cross_source_links
            and cross_source_confidence > 0.0
            and not bank_hard_conflicts
        )
        if synchronized_ready:
            weighted_sync += weight
        synchronized[str(bank_id)] = {
            "bank_id": bank_id,
            "name": raw.get("name"),
            "tier": raw.get("tier"),
            "weight": weight,
            "currency": currency,
            "region": raw.get("region"),
            "groups": list(raw.get("groups") or []),
            "bot_domains": list(raw.get("bot_domains") or []),
            "official_policy_url": raw.get("official_policy_url"),
            "usable_features": usable,
            "lineage": links,
            "lineage_point_in_time_ratio": lineage_ratio,
            "cross_source_link_count": len(cross_source_links),
            "joined_source_ids": sorted({str(link.get("source_id") or "") for link in cross_source_links}),
            "join_confidence_norm": join_confidence,
            "cross_source_confidence_norm": cross_source_confidence,
            "hard_conflicts": bank_hard_conflicts,
            "soft_conflicts": bank_soft_conflicts,
            "derived_features": derived_features,
            "synchronized_ready": synchronized_ready,
        }

    synchronized_rows = [row for row in synchronized.values() if bool(row.get("synchronized_ready", False))]
    total_sync_weight = max(weighted_ready, 1e-9)
    sync_ratio = weighted_sync / total_sync_weight
    fx_ratio = fx_join_weight / total_sync_weight
    macro_ratio = macro_join_weight / total_sync_weight
    liquidity_ratio = liquidity_join_weight / total_sync_weight
    lineage_coverage = _weighted_mean(lineage_ratios) or 0.0
    freshness_norm = _mean(freshness_scores) or 0.0
    conflict_free = 0.0 if hard_conflicts else _clamp01(1.0 - len(soft_conflicts) / max(len(synchronized), 1))
    policy_divergence = _safe_float(raw_global_features.get("global_central_bank_policy_divergence_norm")) or 0.0
    vix = _safe_float(macro_cross.get("vix")) or 0.0
    high_yield = _safe_float(macro_cross.get("high_yield_oas_bps")) or 0.0
    global_risk_stress = _clamp01(0.5 * (vix / 60.0) + 0.5 * (high_yield / 10.0))
    global_features = {
        "central_bank_sync_available_norm": _clamp01(min(sync_ratio, conflict_free)),
        "central_bank_sync_coverage_norm": _clamp01(sync_ratio),
        "central_bank_sync_point_in_time_norm": 1.0 if not future_exclusions else _clamp01(1.0 - len(future_exclusions) / max(len(synchronized), 1)),
        "central_bank_sync_lineage_coverage_norm": _clamp01(lineage_coverage),
        "central_bank_sync_fx_coverage_norm": _clamp01(fx_ratio),
        "central_bank_sync_macro_coverage_norm": _clamp01(macro_ratio),
        "central_bank_sync_liquidity_coverage_norm": _clamp01(liquidity_ratio),
        "central_bank_sync_conflict_free_norm": conflict_free,
        "central_bank_sync_freshness_norm": _clamp01(freshness_norm),
        "central_bank_policy_fx_confirmation_norm": _clamp01(_weighted_mean(policy_fx_scores) or 0.5),
        "central_bank_policy_inflation_alignment_norm": _clamp01(_weighted_mean(inflation_scores) or 0.5),
        "central_bank_policy_external_balance_alignment_norm": _clamp01(_weighted_mean(external_balance_scores) or 0.5),
        "central_bank_policy_liquidity_alignment_norm": _clamp01(_weighted_mean(liquidity_scores) or 0.5),
        "central_bank_policy_cross_asset_confirmation_norm": _clamp01(_weighted_mean(cross_asset_scores) or 0.5),
        "central_bank_policy_divergence_signal_norm": _clamp01(policy_divergence * sync_ratio),
        "central_bank_policy_spillover_risk_norm": _clamp01(policy_divergence * (0.5 + 0.5 * global_risk_stress)),
    }
    missing_features = [key for key in CENTRAL_BANK_CROSS_SOURCE_FEATURE_KEYS if key not in global_features]
    if missing_features:
        raise ValueError(f"central_bank_sync_feature_schema_incomplete:{','.join(missing_features)}")

    domain_routes: dict[str, list[str]] = {}
    symbol_routes: dict[str, list[str]] = {}
    symbol_features: dict[str, dict[str, float]] = {}
    for bank_id, row in synchronized.items():
        if not bool(row.get("synchronized_ready", False)):
            continue
        for domain in row.get("bot_domains") or []:
            domain_routes.setdefault(str(domain), []).append(bank_id)
        currency = str(row.get("currency") or "")
        for symbol in CURRENCY_SYMBOLS.get(currency, []):
            symbol_routes.setdefault(symbol, []).append(bank_id)
            scoped = dict(global_features)
            scoped.update(_bank_feature_map(row))
            symbol_features[symbol] = scoped
    for route in (domain_routes, symbol_routes):
        for key, bank_ids in route.items():
            route[key] = sorted(
                set(bank_ids),
                key=lambda bank_id: (
                    int(synchronized[bank_id].get("tier", 99) or 99),
                    -float(synchronized[bank_id].get("weight", 0.0) or 0.0),
                    bank_id,
                ),
            )

    coverage = {
        "registry_bank_count": len(raw_banks),
        "raw_ready_bank_count": sum(1 for row in raw_banks.values() if isinstance(row, Mapping) and bool(row.get("ready", False))),
        "synchronized_ready_bank_count": len(synchronized_rows),
        "distinct_cross_source_link_count": sum(int(row.get("cross_source_link_count", 0) or 0) for row in synchronized.values()),
        "banks_without_distinct_cross_source": sorted(
            bank_id
            for bank_id, row in synchronized.items()
            if bool((raw_banks.get(bank_id) or {}).get("ready", False))
            and int(row.get("cross_source_link_count", 0) or 0) == 0
        ),
        "synchronized_bank_coverage_ratio": sync_ratio,
        "fx_join_coverage_ratio": fx_ratio,
        "macro_join_coverage_ratio": macro_ratio,
        "liquidity_join_coverage_ratio": liquidity_ratio,
        "lineage_coverage_ratio": lineage_coverage,
        "hard_conflict_count": len(hard_conflicts),
        "soft_conflict_count": len(soft_conflicts),
        "hard_conflicts": hard_conflicts,
        "soft_conflicts": soft_conflicts,
        "future_observations_excluded": future_exclusions,
        "future_observation_selected": False,
    }
    return {
        "schema_version": 1,
        "timestamp_utc": now.isoformat(),
        "provider": "central_bank_point_in_time_cross_source_router",
        "contract": {
            "minimum_sync_coverage_ratio": 0.6,
            "minimum_distinct_cross_source_count": 1,
            "live_execution_authority": False,
            "automatic_promotion_authority": False,
        },
        "methodology": {
            "point_in_time_join": True,
            "distinct_cross_source_required": True,
            "join_keys": ["bis_area_code", "world_bank_area_code", "currency", "observation_time"],
            "future_dated_records_excluded": True,
            "stale_sources_excluded": True,
            "missing_values_are_neutralized": False,
            "raw_missing_values_are_omitted": True,
            "aggregate_neutral_values_require_coverage_companions": True,
            "source_conflicts_are_fail_visible": True,
            "sparse_merge_prevents_valid_values_from_being_overwritten": True,
        },
        "source_states": source_states,
        "source_contracts": {
            "global_central_bank_context": global_assessment,
            "central_bank_liquidity": liquidity_assessment,
        },
        "coverage": coverage,
        "banks": synchronized,
        "routing": {
            "domain_to_bank_ids": domain_routes,
            "symbol_to_bank_ids": symbol_routes,
            "selection_policy": "freshest_point_in_time_then_tier_then_governance_weight",
            "fallback_policy": "omit_missing_dimension_and_surface_coverage; never substitute_zero",
        },
        "global_features": global_features,
        "symbol_features": symbol_features,
        "derived": {
            "global_features": global_features,
            "symbol_features": symbol_features,
        },
    }


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, raw_temp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temp_path = Path(raw_temp)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(dict(payload), handle, ensure_ascii=True, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
    finally:
        temp_path.unlink(missing_ok=True)


def _write_history(payload: Mapping[str, Any]) -> Path:
    HISTORY_ROOT.mkdir(parents=True, exist_ok=True)
    timestamp = str(payload.get("timestamp_utc") or "").replace(":", "").replace("-", "").replace("+00:00", "Z")
    path = HISTORY_ROOT / f"central_bank_cross_source_{timestamp}.json.gz"
    fd, raw_temp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(fd)
    temp_path = Path(raw_temp)
    try:
        with gzip.open(temp_path, "wt", encoding="utf-8") as handle:
            json.dump(dict(payload), handle, ensure_ascii=True, separators=(",", ":"))
            handle.write("\n")
        os.replace(temp_path, path)
    finally:
        temp_path.unlink(missing_ok=True)
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description="Synchronize global central-bank facts with FX, macro, liquidity, and official-event sources.")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--test-only", action="store_true")
    args = parser.parse_args()
    now = datetime.now(timezone.utc)
    payload = build_central_bank_cross_source_context(
        global_central_banks=_read_json(EXTERNAL_ROOT / "global_central_bank_context_latest.json"),
        fx_market=_read_json(EXTERNAL_ROOT / "fx_market_context_latest.json"),
        public_policy=_read_json(EXTERNAL_ROOT / "public_policy_context_latest.json"),
        official_macro=_read_json(EXTERNAL_ROOT / "official_macro_context_latest.json"),
        macro_cross_asset=_read_json(EXTERNAL_ROOT / "macro_cross_asset_latest.json"),
        central_bank_liquidity=_read_json(EXTERNAL_ROOT / "central_bank_liquidity_latest.json"),
        now=now,
    )
    assessment = assess_central_bank_cross_source_context(payload, now_utc=now)
    status = {
        "timestamp_utc": now.isoformat(),
        "ok": bool(assessment.get("ready", False)),
        "status": "ready" if bool(assessment.get("ready", False)) else "degraded",
        "coverage": payload.get("coverage"),
        "source_states": payload.get("source_states"),
        "assessment": assessment,
    }
    if not args.test_only:
        _atomic_write_json(LATEST_PATH, payload)
        status["history_path"] = str(_write_history(payload))
        _atomic_write_json(HEALTH_PATH, status)
    if args.json:
        print(json.dumps(status, ensure_ascii=True))
    else:
        coverage = payload.get("coverage") or {}
        print(
            "central_bank_cross_source "
            f"ok={int(bool(status['ok']))} "
            f"banks={coverage.get('synchronized_ready_bank_count', 0)}/{coverage.get('raw_ready_bank_count', 0)} "
            f"conflicts={coverage.get('hard_conflict_count', 0)}"
        )
    return 0 if bool(status["ok"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
