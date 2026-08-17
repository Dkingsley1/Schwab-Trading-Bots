#!/usr/bin/env python3
from __future__ import annotations

import argparse
import calendar
import gzip
import json
import math
import os
import statistics
import sys
import tempfile
import xml.etree.ElementTree as ET
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.collector_transport import fetch_text
from core.global_central_bank_context import (
    GLOBAL_CENTRAL_BANK_FEATURE_KEYS,
    assess_global_central_bank_context,
    load_global_central_bank_registry,
)


USER_AGENT = "schwab-trading-bot/1.0"
BIS_POLICY_API_ROOT = "https://stats.bis.org/api/v2/data/dataflow/BIS/WS_CBPOL/1.0"
BIS_ASSETS_API_ROOT = "https://stats.bis.org/api/v2/data/dataflow/BIS/WS_CBTA/1.0"
BIS_POLICY_TOPIC_URL = "https://data.bis.org/topics/CBPOL"
BIS_ASSETS_TOPIC_URL = "https://data.bis.org/topics/CBTA"
LATEST_PATH = PROJECT_ROOT / "exports" / "external_context" / "global_central_bank_context_latest.json"
HEALTH_PATH = PROJECT_ROOT / "governance" / "health" / "global_central_bank_context_sync_latest.json"
HISTORY_ROOT = PROJECT_ROOT / "data" / "external_context" / "global_central_bank_history"


def _clamp01(value: float) -> float:
    return max(0.0, min(float(value), 1.0))


def _safe_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _signed_norm(value: float, scale: float) -> float:
    width = max(abs(float(scale)), 1e-9)
    return _clamp01(0.5 + (float(value) / (2.0 * width)))


def _weighted_mean(rows: Iterable[tuple[float, float]]) -> float | None:
    clean = [(float(value), max(float(weight), 0.0)) for value, weight in rows if math.isfinite(float(value))]
    total_weight = sum(weight for _, weight in clean)
    if not clean or total_weight <= 0.0:
        return None
    return sum(value * weight for value, weight in clean) / total_weight


def _local_name(tag: str) -> str:
    return str(tag or "").rsplit("}", 1)[-1]


def _period_end(raw: Any) -> date | None:
    text = str(raw or "").strip()
    if not text:
        return None
    try:
        return date.fromisoformat(text)
    except ValueError:
        pass
    if len(text) == 7 and text[4:6] == "-Q" and text[-1] in "1234":
        year = int(text[:4])
        month = int(text[-1]) * 3
        return date(year, month, calendar.monthrange(year, month)[1])
    if len(text) == 7 and text[4] == "-":
        try:
            year = int(text[:4])
            month = int(text[5:7])
            return date(year, month, calendar.monthrange(year, month)[1])
        except (TypeError, ValueError):
            return None
    if len(text) == 4 and text.isdigit():
        return date(int(text), 12, 31)
    return None


def _parse_sdmx_series(xml_text: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    root = ET.fromstring(str(xml_text or ""))
    extracted = ""
    reporting_end = ""
    for node in root.iter():
        name = _local_name(node.tag)
        if name == "Extracted" and not extracted:
            extracted = str(node.text or "").strip()
        elif name == "ReportingEnd" and not reporting_end:
            reporting_end = str(node.text or "").strip()

    series_rows: list[dict[str, Any]] = []
    for node in root.iter():
        if _local_name(node.tag) != "Series":
            continue
        observations: list[dict[str, Any]] = []
        for child in node:
            if _local_name(child.tag) != "Obs":
                continue
            value = _safe_float(child.attrib.get("OBS_VALUE"))
            period = str(child.attrib.get("TIME_PERIOD") or "").strip()
            period_date = _period_end(period)
            if value is None or period_date is None:
                continue
            observations.append(
                {
                    "period": period,
                    "date": period_date,
                    "value": value,
                    "status": str(child.attrib.get("OBS_STATUS") or ""),
                    "confidence": str(child.attrib.get("OBS_CONF") or ""),
                }
            )
        observations.sort(key=lambda row: row["date"])
        series_rows.append({"attributes": dict(node.attrib), "observations": observations})
    return series_rows, {"extracted": extracted, "reporting_end": reporting_end}


def _split_as_of(
    observations: list[dict[str, Any]],
    *,
    as_of_date: date,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    eligible = [row for row in observations if isinstance(row.get("date"), date) and row["date"] <= as_of_date]
    future = [row for row in observations if isinstance(row.get("date"), date) and row["date"] > as_of_date]
    return eligible, future


def _value_on_or_before(observations: list[dict[str, Any]], cutoff: date) -> dict[str, Any] | None:
    for row in reversed(observations):
        if row["date"] <= cutoff:
            return row
    return None


def _policy_record(
    series: Mapping[str, Any],
    *,
    as_of_date: date,
    max_age_days: int,
) -> tuple[dict[str, Any], list[str]]:
    observations, future = _split_as_of(list(series.get("observations") or []), as_of_date=as_of_date)
    if not observations:
        return {}, [str(row.get("period") or "") for row in future]
    latest = observations[-1]
    latest_value = float(latest["value"])
    previous_distinct = None
    change_start = latest["date"]
    for index in range(len(observations) - 2, -1, -1):
        row = observations[index]
        if math.isclose(float(row["value"]), latest_value, rel_tol=0.0, abs_tol=1e-12):
            change_start = row["date"]
            continue
        previous_distinct = row
        break
    if previous_distinct is not None:
        following_index = observations.index(previous_distinct) + 1
        if following_index < len(observations):
            change_start = observations[following_index]["date"]

    changes: dict[str, float | None] = {}
    for days in (30, 90, 365):
        prior = _value_on_or_before(observations, as_of_date - timedelta(days=days))
        changes[f"change_bps_{days}d"] = (
            round((latest_value - float(prior["value"])) * 100.0, 8) if prior is not None else None
        )
    age_days = max((as_of_date - latest["date"]).days, 0)
    previous_value = float(previous_distinct["value"]) if previous_distinct is not None else None
    attrs = dict(series.get("attributes") or {})
    return (
        {
            "rate_percent": latest_value,
            "observation_date": latest["date"].isoformat(),
            "observation_age_days": age_days,
            "fresh": age_days <= max(int(max_age_days), 1),
            "previous_distinct_rate_percent": previous_value,
            "last_change_bps": round((latest_value - previous_value) * 100.0, 8) if previous_value is not None else None,
            "last_change_effective_date": change_start.isoformat(),
            "days_since_last_change": max((as_of_date - change_start).days, 0),
            **changes,
            "source_name": str(attrs.get("SOURCE_REF") or ""),
            "compilation": str(attrs.get("COMPILATION") or ""),
            "title": str(attrs.get("TITLE") or ""),
            "frequency": str(attrs.get("FREQ") or ""),
        },
        [str(row.get("period") or "") for row in future],
    )


def _asset_record(
    series: Mapping[str, Any],
    *,
    as_of_date: date,
    max_age_days: int,
) -> tuple[dict[str, Any], list[str]]:
    observations, future = _split_as_of(list(series.get("observations") or []), as_of_date=as_of_date)
    if not observations:
        return {}, [str(row.get("period") or "") for row in future]
    latest = observations[-1]
    previous = observations[-2] if len(observations) >= 2 else None
    year_prior = observations[-5] if len(observations) >= 5 else None
    latest_value = float(latest["value"])

    def pct_change(prior: Mapping[str, Any] | None) -> float | None:
        if prior is None:
            return None
        prior_value = float(prior["value"])
        if abs(prior_value) <= 1e-12:
            return None
        return round((latest_value / prior_value - 1.0) * 100.0, 8)

    attrs = dict(series.get("attributes") or {})
    age_days = max((as_of_date - latest["date"]).days, 0)
    return (
        {
            "total_assets_usd_billions": latest_value,
            "observation_period": str(latest.get("period") or ""),
            "observation_date": latest["date"].isoformat(),
            "observation_age_days": age_days,
            "fresh": age_days <= max(int(max_age_days), 1),
            "quarter_over_quarter_change_pct": pct_change(previous),
            "year_over_year_change_pct": pct_change(year_prior),
            "compiling_organizations": str(attrs.get("COMPILING_ORG") or ""),
            "compilation_method": str(attrs.get("METHOD_REF") or attrs.get("COMP_METHOD") or ""),
            "title": str(attrs.get("TITLE") or ""),
            "frequency": str(attrs.get("FREQ") or ""),
            "unit_multiplier": str(attrs.get("UNIT_MULT") or ""),
        },
        [str(row.get("period") or "") for row in future],
    )


def _series_by_area(series_rows: Iterable[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for row in series_rows:
        attrs = row.get("attributes") if isinstance(row.get("attributes"), Mapping) else {}
        area = str(attrs.get("REF_AREA") or "").strip().upper()
        if area and row.get("observations"):
            out[area] = dict(row)
    return out


def _breadth(
    banks: Iterable[Mapping[str, Any]],
    *,
    field: str,
    predicate: Any,
) -> float:
    rows: list[tuple[float, float]] = []
    for bank in banks:
        policy = bank.get("policy_rate") if isinstance(bank.get("policy_rate"), Mapping) else {}
        value = _safe_float(policy.get(field))
        if value is None:
            continue
        rows.append((1.0 if predicate(value) else 0.0, float(bank.get("weight", 0.0) or 0.0)))
    return _clamp01(_weighted_mean(rows) or 0.0)


def _build_global_features(banks: Mapping[str, Mapping[str, Any]], coverage: Mapping[str, Any]) -> dict[str, float]:
    bank_rows = list(banks.values())
    policy_rows = [row for row in bank_rows if bool((row.get("policy_rate") or {}).get("fresh", False))]
    asset_rows = [row for row in bank_rows if bool((row.get("balance_sheet") or {}).get("fresh", False))]

    easing_30 = _breadth(policy_rows, field="change_bps_30d", predicate=lambda value: value < -0.5)
    tightening_30 = _breadth(policy_rows, field="change_bps_30d", predicate=lambda value: value > 0.5)
    hold_30 = _breadth(policy_rows, field="change_bps_30d", predicate=lambda value: abs(value) <= 0.5)
    policy_30 = _weighted_mean(
        (
            float((row.get("policy_rate") or {}).get("change_bps_30d") or 0.0),
            float(row.get("weight", 0.0) or 0.0),
        )
        for row in policy_rows
    ) or 0.0
    policy_90 = _weighted_mean(
        (
            float((row.get("policy_rate") or {}).get("change_bps_90d") or 0.0),
            float(row.get("weight", 0.0) or 0.0),
        )
        for row in policy_rows
    ) or 0.0
    intensity_30 = _weighted_mean(
        (
            abs(float((row.get("policy_rate") or {}).get("change_bps_30d") or 0.0)),
            float(row.get("weight", 0.0) or 0.0),
        )
        for row in policy_rows
    ) or 0.0
    intensity_90 = _weighted_mean(
        (
            abs(float((row.get("policy_rate") or {}).get("change_bps_90d") or 0.0)),
            float(row.get("weight", 0.0) or 0.0),
        )
        for row in policy_rows
    ) or 0.0
    policy_values = [float((row.get("policy_rate") or {}).get("rate_percent")) for row in policy_rows]
    policy_divergence = statistics.pstdev(policy_values) if len(policy_values) >= 2 else 0.0

    asset_expansion = _clamp01(
        _weighted_mean(
            (
                1.0 if float((row.get("balance_sheet") or {}).get("quarter_over_quarter_change_pct") or 0.0) > 0.05 else 0.0,
                float(row.get("weight", 0.0) or 0.0),
            )
            for row in asset_rows
        )
        or 0.0
    )
    asset_contraction = _clamp01(
        _weighted_mean(
            (
                1.0 if float((row.get("balance_sheet") or {}).get("quarter_over_quarter_change_pct") or 0.0) < -0.05 else 0.0,
                float(row.get("weight", 0.0) or 0.0),
            )
            for row in asset_rows
        )
        or 0.0
    )
    asset_impulse = _weighted_mean(
        (
            float((row.get("balance_sheet") or {}).get("quarter_over_quarter_change_pct") or 0.0),
            float(row.get("weight", 0.0) or 0.0),
        )
        for row in asset_rows
    ) or 0.0

    def grouped_breadth(group: str, *, easing: bool) -> float:
        rows = [row for row in policy_rows if group in set(row.get("groups") or [])]
        return _breadth(
            rows,
            field="change_bps_90d",
            predicate=(lambda value: value < -0.5) if easing else (lambda value: value > 0.5),
        )

    policy_by_area = {
        str(row.get("bis_area_code") or ""): float((row.get("policy_rate") or {}).get("rate_percent"))
        for row in policy_rows
        if _safe_float((row.get("policy_rate") or {}).get("rate_percent")) is not None
    }
    usd_rate = policy_by_area.get("US")
    foreign_mean = _weighted_mean(
        (
            float((row.get("policy_rate") or {}).get("rate_percent")),
            float(row.get("weight", 0.0) or 0.0),
        )
        for row in policy_rows
        if str(row.get("bis_area_code") or "") != "US"
    )

    def usd_spread(area: str) -> float:
        if usd_rate is None or area not in policy_by_area:
            return 0.5
        return _signed_norm(usd_rate - policy_by_area[area], 10.0)

    fx_framework_rows = [
        row
        for row in bank_rows
        if any(token in str(row.get("policy_framework") or "") for token in ("exchange_rate", "peg", "currency_board"))
    ]
    fx_framework_ready = sum(1 for row in fx_framework_rows if bool(row.get("ready", False)))
    important_ratio = float(coverage.get("important_bank_coverage_ratio", 0.0) or 0.0)
    tier1_ratio = float(coverage.get("tier_1_coverage_ratio", 0.0) or 0.0)
    return {
        "global_central_bank_context_available_norm": _clamp01(min(important_ratio, tier1_ratio)),
        "global_central_bank_important_coverage_norm": _clamp01(important_ratio),
        "global_central_bank_tier1_coverage_norm": _clamp01(tier1_ratio),
        "global_central_bank_policy_rate_coverage_norm": _clamp01(float(coverage.get("policy_rate_coverage_ratio", 0.0) or 0.0)),
        "global_central_bank_balance_sheet_coverage_norm": _clamp01(float(coverage.get("balance_sheet_coverage_ratio", 0.0) or 0.0)),
        "global_central_bank_policy_easing_breadth_norm": easing_30,
        "global_central_bank_policy_tightening_breadth_norm": tightening_30,
        "global_central_bank_policy_hold_breadth_norm": hold_30,
        "global_central_bank_policy_impulse_30d_norm": _signed_norm(policy_30, 200.0),
        "global_central_bank_policy_impulse_90d_norm": _signed_norm(policy_90, 300.0),
        "global_central_bank_policy_divergence_norm": _clamp01(policy_divergence / 10.0),
        "global_central_bank_policy_change_intensity_30d_norm": _clamp01(intensity_30 / 200.0),
        "global_central_bank_policy_change_intensity_90d_norm": _clamp01(intensity_90 / 300.0),
        "global_central_bank_synchronized_easing_norm": _clamp01((easing_30 - 0.3) / 0.7),
        "global_central_bank_synchronized_tightening_norm": _clamp01((tightening_30 - 0.3) / 0.7),
        "global_central_bank_balance_sheet_expansion_breadth_norm": asset_expansion,
        "global_central_bank_balance_sheet_contraction_breadth_norm": asset_contraction,
        "global_central_bank_balance_sheet_impulse_norm": _signed_norm(asset_impulse, 10.0),
        "global_central_bank_usd_rate_advantage_norm": (
            _signed_norm(usd_rate - foreign_mean, 10.0)
            if usd_rate is not None and foreign_mean is not None
            else 0.5
        ),
        "global_central_bank_g5_easing_breadth_norm": grouped_breadth("g5", easing=True),
        "global_central_bank_g5_tightening_breadth_norm": grouped_breadth("g5", easing=False),
        "global_central_bank_em_easing_breadth_norm": grouped_breadth("emerging", easing=True),
        "global_central_bank_em_tightening_breadth_norm": grouped_breadth("emerging", easing=False),
        "global_central_bank_fx_framework_coverage_norm": _clamp01(
            fx_framework_ready / max(len(fx_framework_rows), 1)
        ),
        "global_central_bank_usd_eur_policy_spread_norm": usd_spread("XM"),
        "global_central_bank_usd_jpy_policy_spread_norm": usd_spread("JP"),
        "global_central_bank_usd_gbp_policy_spread_norm": usd_spread("GB"),
        "global_central_bank_usd_cny_policy_spread_norm": usd_spread("CN"),
    }


def build_global_central_bank_context(
    *,
    policy_xml: str,
    assets_xml: str,
    registry: Mapping[str, Any],
    as_of_date: date,
    collected_at: datetime,
    policy_source: Mapping[str, Any] | None = None,
    assets_source: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    thresholds = registry.get("coverage_thresholds") if isinstance(registry.get("coverage_thresholds"), Mapping) else {}
    policy_max_age = int(thresholds.get("policy_rate_max_age_days", 45) or 45)
    assets_max_age = int(thresholds.get("balance_sheet_max_age_days", 240) or 240)
    policy_series, policy_meta = _parse_sdmx_series(policy_xml)
    asset_series, asset_meta = _parse_sdmx_series(assets_xml)
    policy_by_area = _series_by_area(policy_series)
    assets_by_area = _series_by_area(asset_series)

    banks: dict[str, dict[str, Any]] = {}
    future_excluded: dict[str, dict[str, list[str]]] = {"policy_rates": {}, "balance_sheets": {}}
    for raw in registry.get("banks") if isinstance(registry.get("banks"), list) else []:
        if not isinstance(raw, Mapping):
            continue
        bank_id = str(raw.get("bank_id") or "").strip()
        area = str(raw.get("bis_area_code") or "").strip().upper()
        if not bank_id or not area:
            continue
        policy_record: dict[str, Any] = {}
        asset_record: dict[str, Any] = {}
        if area in policy_by_area:
            policy_record, policy_future = _policy_record(
                policy_by_area[area],
                as_of_date=as_of_date,
                max_age_days=policy_max_age,
            )
            if policy_future:
                future_excluded["policy_rates"][area] = policy_future
        if area in assets_by_area:
            asset_record, asset_future = _asset_record(
                assets_by_area[area],
                as_of_date=as_of_date,
                max_age_days=assets_max_age,
            )
            if asset_future:
                future_excluded["balance_sheets"][area] = asset_future

        policy_required = bool(raw.get("policy_rate_required", False))
        assets_required = bool(raw.get("balance_sheet_required", False))
        policy_ok = bool(policy_record.get("fresh", False))
        assets_ok = bool(asset_record.get("fresh", False))
        observed_count = int(bool(policy_record)) + int(bool(asset_record))
        missing_dimensions: list[str] = []
        if policy_required and not policy_ok:
            missing_dimensions.append("policy_rate")
        if assets_required and not assets_ok:
            missing_dimensions.append("balance_sheet")
        ready = observed_count > 0 and not missing_dimensions
        banks[bank_id] = {
            **dict(raw),
            "policy_rate": policy_record,
            "balance_sheet": asset_record,
            "observed_dimension_count": observed_count,
            "missing_required_dimensions": missing_dimensions,
            "ready": ready,
        }

    bank_rows = list(banks.values())
    tier1_rows = [row for row in bank_rows if int(row.get("tier", 99) or 99) == 1]
    ready_rows = [row for row in bank_rows if bool(row.get("ready", False))]
    tier1_ready = [row for row in tier1_rows if bool(row.get("ready", False))]
    total_weight = sum(max(float(row.get("weight", 0.0) or 0.0), 0.0) for row in bank_rows)
    ready_weight = sum(max(float(row.get("weight", 0.0) or 0.0), 0.0) for row in ready_rows)
    policy_required_rows = [row for row in bank_rows if bool(row.get("policy_rate_required", False))]
    asset_required_rows = [row for row in bank_rows if bool(row.get("balance_sheet_required", False))]
    source_failures = []
    if not policy_series:
        source_failures.append("bis_policy_rates")
    if not asset_series:
        source_failures.append("bis_total_assets")
    coverage = {
        "as_of_date": as_of_date.isoformat(),
        "registry_bank_count": len(bank_rows),
        "ready_bank_count": len(ready_rows),
        "tier_1_bank_count": len(tier1_rows),
        "tier_1_ready_count": len(tier1_ready),
        "tier_1_coverage_ratio": len(tier1_ready) / max(len(tier1_rows), 1),
        "important_bank_coverage_ratio": ready_weight / max(total_weight, 1e-9),
        "policy_rate_coverage_ratio": sum(
            1 for row in policy_required_rows if bool((row.get("policy_rate") or {}).get("fresh", False))
        )
        / max(len(policy_required_rows), 1),
        "balance_sheet_coverage_ratio": sum(
            1 for row in asset_required_rows if bool((row.get("balance_sheet") or {}).get("fresh", False))
        )
        / max(len(asset_required_rows), 1),
        "raw_policy_area_count": len(policy_by_area),
        "raw_balance_sheet_area_count": len(assets_by_area),
        "registered_policy_area_count": len({str(row.get("bis_area_code") or "") for row in bank_rows} & set(policy_by_area)),
        "registered_balance_sheet_area_count": len({str(row.get("bis_area_code") or "") for row in bank_rows} & set(assets_by_area)),
        "unregistered_policy_areas": sorted(set(policy_by_area) - {str(row.get("bis_area_code") or "") for row in bank_rows}),
        "unregistered_balance_sheet_areas": sorted(set(assets_by_area) - {str(row.get("bis_area_code") or "") for row in bank_rows}),
        "future_observations_excluded": future_excluded,
        "future_observation_selected": False,
        "source_failures": source_failures,
    }
    global_features = _build_global_features(banks, coverage)
    missing_feature_keys = [key for key in GLOBAL_CENTRAL_BANK_FEATURE_KEYS if key not in global_features]
    if missing_feature_keys:
        raise ValueError(f"global_central_bank_feature_schema_incomplete:{','.join(missing_feature_keys)}")

    timestamp = collected_at.astimezone(timezone.utc).isoformat()
    return {
        "schema_version": 1,
        "timestamp_utc": timestamp,
        "as_of_date": as_of_date.isoformat(),
        "provider": "bis_member_central_bank_mesh",
        "contract": {
            "tier_1_minimum_ratio": float(thresholds.get("tier_1_minimum_ratio", 0.8) or 0.8),
            "important_bank_minimum_ratio": float(thresholds.get("important_bank_minimum_ratio", 0.85) or 0.85),
            "policy_rate_max_age_days": policy_max_age,
            "balance_sheet_max_age_days": assets_max_age,
            "live_execution_authority": False,
            "automatic_promotion_authority": False,
        },
        "methodology": {
            "point_in_time_only": True,
            "future_dated_observations_excluded": True,
            "revision_policy": "append_only_collection_snapshots_preserve_values_known_at_collection_time",
            "policy_rate_source": "BIS WS_CBPOL; daily data reported directly by member central banks",
            "balance_sheet_source": "BIS WS_CBTA; national central-bank and official balance-sheet sources",
            "policy_frameworks_are_not_forced_into_a_single_rate": True,
            "missing_values_are_not_zero_filled": True,
        },
        "sources": {
            "bis_policy_rates": {
                "ok": bool(policy_series),
                "url": str((policy_source or {}).get("url") or BIS_POLICY_TOPIC_URL),
                "api_dataset": "BIS,WS_CBPOL,1.0",
                "series_count": len(policy_series),
                "extracted": policy_meta.get("extracted"),
                "reporting_end": policy_meta.get("reporting_end"),
                "transport": dict(policy_source or {}),
            },
            "bis_total_assets": {
                "ok": bool(asset_series),
                "url": str((assets_source or {}).get("url") or BIS_ASSETS_TOPIC_URL),
                "api_dataset": "BIS,WS_CBTA,1.0",
                "series_count": len(asset_series),
                "extracted": asset_meta.get("extracted"),
                "reporting_end": asset_meta.get("reporting_end"),
                "transport": dict(assets_source or {}),
            },
        },
        "coverage": coverage,
        "banks": banks,
        "global_features": global_features,
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
    history_path = HISTORY_ROOT / f"global_central_bank_context_{timestamp}.json.gz"
    fd, raw_temp = tempfile.mkstemp(prefix=f".{history_path.name}.", suffix=".tmp", dir=history_path.parent)
    os.close(fd)
    temp_path = Path(raw_temp)
    try:
        with gzip.open(temp_path, "wt", encoding="utf-8") as handle:
            json.dump(dict(payload), handle, ensure_ascii=True, separators=(",", ":"))
            handle.write("\n")
        os.replace(temp_path, history_path)
    finally:
        temp_path.unlink(missing_ok=True)
    return history_path


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _transport_summary(result: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "url": result.get("url"),
        "status_code": result.get("status_code"),
        "fetched_utc": result.get("fetched_utc"),
        "attempt_count": result.get("attempt_count"),
        "duration_ms": result.get("duration_ms"),
        "payload_sha256": result.get("payload_sha256"),
        "size_bytes": result.get("size_bytes"),
        "source_confidence_norm": result.get("source_confidence_norm"),
        "schema_confidence_norm": result.get("schema_confidence_norm"),
    }


def _fetch_bis(url: str, *, source_name: str, timeout: float) -> dict[str, Any]:
    return fetch_text(
        url,
        user_agent=USER_AGENT,
        timeout=timeout,
        accept="application/vnd.sdmx.structurespecificdata+xml;version=2.1",
        retries=2,
        collector_key="global_central_bank_context",
        source_name=source_name,
        entity_key=url,
        project_root=PROJECT_ROOT,
        source_confidence_norm=0.99,
        schema_confidence_norm=0.97,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Collect point-in-time policy-rate and balance-sheet context for major central banks.")
    parser.add_argument("--timeout-seconds", type=float, default=20.0)
    parser.add_argument("--as-of-date", default="")
    parser.add_argument("--registry", type=Path, default=None)
    parser.add_argument("--test-only", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    now = datetime.now(timezone.utc)
    try:
        as_of_date = date.fromisoformat(str(args.as_of_date or now.date().isoformat()))
    except ValueError:
        raise SystemExit("invalid --as-of-date; expected YYYY-MM-DD")
    registry = load_global_central_bank_registry(args.registry)
    if not registry:
        raise SystemExit("global central-bank registry missing or invalid")

    policy_start = (as_of_date - timedelta(days=420)).isoformat()
    assets_start = f"{as_of_date.year - 4}-Q1"
    policy_url = f"{BIS_POLICY_API_ROOT}/D.?startPeriod={policy_start}&endPeriod={as_of_date.isoformat()}"
    assets_url = f"{BIS_ASSETS_API_ROOT}/Q..B.USD._Z.B?startPeriod={assets_start}&endPeriod={as_of_date.isoformat()}"
    policy_result = _fetch_bis(policy_url, source_name="bis_policy_rates", timeout=float(args.timeout_seconds))
    assets_result = _fetch_bis(assets_url, source_name="bis_total_assets", timeout=float(args.timeout_seconds))

    if not bool(policy_result.get("ok", False)) or not bool(assets_result.get("ok", False)):
        cached = _read_json(LATEST_PATH)
        cached_assessment = assess_global_central_bank_context(cached, now_utc=now, max_age_hours=72.0)
        if bool(cached_assessment.get("ready", False)):
            status = {
                "timestamp_utc": now.isoformat(),
                "ok": True,
                "status": "ready_cached",
                "using_cached_snapshot": True,
                "cached_snapshot_timestamp_utc": cached.get("timestamp_utc"),
                "cached_snapshot_age_hours": cached_assessment.get("age_hours"),
                "sources": {
                    "bis_policy_rates": {"ok": bool(policy_result.get("ok", False)), "error": policy_result.get("error")},
                    "bis_total_assets": {"ok": bool(assets_result.get("ok", False)), "error": assets_result.get("error")},
                },
                "assessment": cached_assessment,
            }
            if not args.test_only:
                _atomic_write_json(HEALTH_PATH, status)
            print(json.dumps(status, ensure_ascii=True) if args.json else "global_central_bank_context ready_cached=1")
            return 0

        status = {
            "timestamp_utc": now.isoformat(),
            "ok": False,
            "status": "degraded",
            "using_cached_snapshot": False,
            "sources": {
                "bis_policy_rates": {"ok": bool(policy_result.get("ok", False)), "error": policy_result.get("error")},
                "bis_total_assets": {"ok": bool(assets_result.get("ok", False)), "error": assets_result.get("error")},
            },
            "assessment": cached_assessment,
        }
        if not args.test_only:
            _atomic_write_json(HEALTH_PATH, status)
        print(json.dumps(status, ensure_ascii=True) if args.json else "global_central_bank_context ok=0")
        return 1

    payload = build_global_central_bank_context(
        policy_xml=str(policy_result.get("text") or ""),
        assets_xml=str(assets_result.get("text") or ""),
        registry=registry,
        as_of_date=as_of_date,
        collected_at=now,
        policy_source=_transport_summary(policy_result),
        assets_source=_transport_summary(assets_result),
    )
    assessment = assess_global_central_bank_context(payload, now_utc=now)
    status = {
        "timestamp_utc": now.isoformat(),
        "ok": bool(assessment.get("ready", False)),
        "status": "ready" if bool(assessment.get("ready", False)) else "degraded",
        "using_cached_snapshot": False,
        "provider": payload.get("provider"),
        "coverage": payload.get("coverage"),
        "sources": {
            key: {
                "ok": bool(value.get("ok", False)),
                "series_count": value.get("series_count"),
                "reporting_end": value.get("reporting_end"),
            }
            for key, value in (payload.get("sources") or {}).items()
            if isinstance(value, Mapping)
        },
        "assessment": assessment,
    }
    history_path = None
    if not args.test_only:
        _atomic_write_json(LATEST_PATH, payload)
        history_path = _write_history(payload)
        status["history_path"] = str(history_path)
        _atomic_write_json(HEALTH_PATH, status)

    if args.json:
        print(json.dumps(status, ensure_ascii=True))
    else:
        coverage = payload.get("coverage") or {}
        print(
            "global_central_bank_context "
            f"ok={int(bool(status['ok']))} "
            f"banks={coverage.get('ready_bank_count', 0)}/{coverage.get('registry_bank_count', 0)} "
            f"tier1={coverage.get('tier_1_ready_count', 0)}/{coverage.get('tier_1_bank_count', 0)}"
        )
    return 0 if bool(status["ok"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
