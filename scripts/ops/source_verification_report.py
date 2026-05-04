#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPORTS_DIR = PROJECT_ROOT / "exports" / "reports"
HEALTH_DIR = PROJECT_ROOT / "governance" / "health"

STATUS_CROSS_VERIFIED = "cross_verified"
STATUS_SINGLE_VERIFIED = "single_source_verified"
STATUS_SINGLE_UNVERIFIED = "single_source_unverified"


def _read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _parse_ts(raw: Any) -> datetime | None:
    text = str(raw or "").strip()
    if not text:
        return None
    try:
        dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except Exception:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _age_hours(ts: datetime | None, now: datetime) -> float | None:
    if ts is None:
        return None
    return max((now - ts).total_seconds() / 3600.0, 0.0)


def _is_fresh(ts: datetime | None, now: datetime, max_age_hours: float) -> bool:
    age = _age_hours(ts, now)
    if age is None:
        return False
    return age <= max(max_age_hours, 0.25)


def _round_age(age: float | None) -> float | None:
    if age is None:
        return None
    return round(age, 3)


def _ok_count(mapping: dict[str, Any]) -> tuple[int, int]:
    total = 0
    ok = 0
    for value in mapping.values():
        if not isinstance(value, dict):
            continue
        if not bool(value.get("contract_participates", True)):
            continue
        total += 1
        if bool(value.get("ok", False)):
            ok += 1
    return ok, total


def _minimum_ok_sources(total: int, *, floor: int = 1, tolerate_failures: int = 1, min_ratio: float = 0.75) -> int:
    if int(total) <= 0:
        return 0
    ratio_target = int(round(float(total) * float(min_ratio)))
    tolerated_target = int(total) - max(int(tolerate_failures), 0)
    return min(int(total), max(int(floor), ratio_target, tolerated_target))


def _row(
    *,
    source_id: str,
    title: str,
    category: str,
    verification_status: str,
    verification_mode: str,
    artifact_path: Path,
    artifact_timestamp: datetime | None,
    age_hours: float | None,
    fresh: bool,
    ok: bool,
    notes: list[str],
    evidence: dict[str, Any],
) -> dict[str, Any]:
    return {
        "source_id": source_id,
        "title": title,
        "category": category,
        "verification_status": verification_status,
        "verification_mode": verification_mode,
        "artifact_path": str(artifact_path),
        "artifact_timestamp_utc": artifact_timestamp.isoformat() if artifact_timestamp is not None else None,
        "age_hours": _round_age(age_hours),
        "fresh": bool(fresh),
        "ok": bool(ok),
        "notes": [str(item) for item in notes if str(item).strip()],
        "evidence": evidence,
    }


def _market_quote_row(health_dir: Path, now: datetime) -> dict[str, Any]:
    path = health_dir / "data_source_divergence_latest.json"
    payload = _read_json(path)
    ts = _parse_ts(payload.get("timestamp_utc"))
    fresh = _is_fresh(ts, now, 12.0)
    notes: list[str] = []
    cross_profile = payload.get("cross_profile") if isinstance(payload.get("cross_profile"), dict) else {}
    offenders = cross_profile.get("offenders") if isinstance(cross_profile.get("offenders"), list) else []
    if cross_profile and not bool(cross_profile.get("ok", True)):
        notes.append(f"cross_profile_residual_offenders={len(offenders)}")
    if not fresh:
        notes.append("stale_artifact")
    status = STATUS_CROSS_VERIFIED if bool(payload.get("ok", False)) and fresh else STATUS_SINGLE_UNVERIFIED
    return _row(
        source_id="market_quote_profiles",
        title="Market Quote Profiles",
        category="market_data",
        verification_status=status,
        verification_mode="cross_profile_divergence",
        artifact_path=path,
        artifact_timestamp=ts,
        age_hours=_age_hours(ts, now),
        fresh=fresh,
        ok=bool(payload.get("ok", False)),
        notes=notes,
        evidence={
            "compared_buckets": int(payload.get("compared_buckets", 0) or 0),
            "worst_relative_spread": float(payload.get("worst_relative_spread", 0.0) or 0.0),
            "max_relative_spread": float(payload.get("max_relative_spread", 0.0) or 0.0),
            "cross_profile_ok": bool(cross_profile.get("ok", False)) if cross_profile else None,
            "cross_profile_offenders": offenders[:5],
        },
    )


def _options_flow_row(health_dir: Path, now: datetime) -> dict[str, Any]:
    path = health_dir / "options_flow_context_sync_latest.json"
    if not path.exists():
        path = health_dir / "tastytrade_context_sync_latest.json"
    payload = _read_json(path)
    ts = _parse_ts(payload.get("timestamp_utc"))
    fresh = _is_fresh(ts, now, 12.0)
    notes: list[str] = []
    sources = payload.get("sources") if isinstance(payload.get("sources"), dict) else {}
    polygon = sources.get("polygon") if isinstance(sources.get("polygon"), dict) else {}
    unusual_whales_api = sources.get("unusual_whales_api") if isinstance(sources.get("unusual_whales_api"), dict) else {}
    unusual_whales_export = sources.get("unusual_whales_export") if isinstance(sources.get("unusual_whales_export"), dict) else {}
    coverage = payload.get("coverage") if isinstance(payload.get("coverage"), dict) else {}
    polygon_ok = bool(polygon.get("ok", False))
    unusual_whales_ok = bool(unusual_whales_api.get("ok", False) or unusual_whales_export.get("ok", False))
    unusual_whales_expected = bool(
        unusual_whales_api.get("expected", False)
        or unusual_whales_export.get("expected", False)
        or unusual_whales_export.get("configured", False)
    )
    polygon_backbone_ok = bool(coverage.get("polygon_backbone_ok", False) or int(payload.get("symbols_with_chain", 0) or 0) > 0)
    context_profile = str(payload.get("context_profile") or coverage.get("context_profile") or "").strip()
    if not context_profile:
        context_profile = (
            "multi_provider_full"
            if polygon_backbone_ok and unusual_whales_ok
            else "polygon_primary_only"
            if polygon_backbone_ok and not unusual_whales_expected
            else "polygon_backbone_only"
            if polygon_backbone_ok
            else "unusual_whales_overlay_only"
            if unusual_whales_ok
            else "unavailable"
        )
    overall_status = str(payload.get("overall_status") or ("ready" if payload.get("ok", False) else "blocked")).strip()
    if not polygon_ok:
        for err in list(polygon.get("errors") or [])[:3]:
            text = str(err or "").strip()
            if text:
                notes.append(text)
    if unusual_whales_expected and (not unusual_whales_ok) and bool(payload.get("operator_action_required", False)):
        notes.append(str(payload.get("auth_issue") or "options_flow_source_unavailable"))
    if context_profile and context_profile not in {"multi_provider_full", "polygon_primary_only"}:
        notes.append(f"context_profile={context_profile}")
    if overall_status and overall_status != "ready":
        notes.append(f"overall_status={overall_status}")
    if not fresh:
        notes.append("stale_artifact")
    if bool(payload.get("ok", False)) and fresh and context_profile == "multi_provider_full" and overall_status == "ready":
        status = STATUS_CROSS_VERIFIED
    elif bool(payload.get("ok", False)) and fresh and polygon_backbone_ok and overall_status == "ready":
        status = STATUS_SINGLE_VERIFIED
    else:
        status = STATUS_SINGLE_UNVERIFIED
    return _row(
        source_id="polygon_unusual_whales_options_context",
        title="Polygon + Optional Overlay Options Context",
        category="derivatives_data",
        verification_status=status,
        verification_mode="multi_provider_options_flow",
        artifact_path=path,
        artifact_timestamp=ts,
        age_hours=_age_hours(ts, now),
        fresh=fresh,
        ok=bool(payload.get("ok", False)),
        notes=notes,
        evidence={
            "symbols_requested": int(payload.get("symbols_requested", 0) or 0),
            "symbols_with_chain": int(payload.get("symbols_with_chain", 0) or 0),
            "symbols_with_metrics": int(payload.get("symbols_with_metrics", 0) or 0),
            "polygon_ok": polygon_ok,
            "polygon_backbone_ok": polygon_backbone_ok,
            "unusual_whales_api_ok": bool(unusual_whales_api.get("ok", False)),
            "unusual_whales_export_ok": bool(unusual_whales_export.get("ok", False)),
            "unusual_whales_expected": unusual_whales_expected,
            "context_profile": context_profile,
            "overall_status": overall_status,
            "coverage_score": payload.get("coverage_score"),
        },
    )


def _macro_crosscheck_row(health_dir: Path, now: datetime) -> dict[str, Any]:
    path = health_dir / "macro_crosscheck_latest.json"
    payload = _read_json(path)
    ts = _parse_ts(payload.get("timestamp_utc"))
    fresh = _is_fresh(ts, now, 24.0)
    notes: list[str] = []
    if isinstance(payload.get("notes"), list):
        notes.extend(str(item) for item in payload.get("notes", []) if str(item).strip())
    if not fresh:
        notes.append("stale_artifact")
    status = STATUS_CROSS_VERIFIED if bool(payload.get("ok", False)) and fresh else STATUS_SINGLE_UNVERIFIED
    return _row(
        source_id="macro_crossstack",
        title="Macro Cross-Stack",
        category="macro_data",
        verification_status=status,
        verification_mode="cross_artifact_overlap",
        artifact_path=path,
        artifact_timestamp=ts,
        age_hours=_age_hours(ts, now),
        fresh=fresh,
        ok=bool(payload.get("ok", False)),
        notes=notes,
        evidence={
            "passed_checks": int(payload.get("passed_checks", 0) or 0),
            "total_checks": int(payload.get("total_checks", 0) or 0),
            "checks": {
                key: bool(value.get("ok", False))
                for key, value in (payload.get("checks") or {}).items()
                if isinstance(value, dict)
            },
        },
    )


def _crypto_market_row(health_dir: Path, now: datetime) -> dict[str, Any]:
    path = health_dir / "crypto_market_context_sync_latest.json"
    payload = _read_json(path)
    ts = _parse_ts(payload.get("timestamp_utc"))
    fresh = _is_fresh(ts, now, 24.0)
    notes: list[str] = []
    compared_assets = int(payload.get("compared_assets", 0) or 0)
    ok_sources = int(payload.get("ok_source_count", 0) or 0)
    total_sources = int(payload.get("source_count", 0) or 0)
    if compared_assets <= 0:
        notes.append("no_cross_provider_overlap")
    if ok_sources < total_sources:
        notes.append(f"partial_sources={ok_sources}/{total_sources}")
    warning_count = int(payload.get("warning_count", 0) or 0)
    if warning_count > 0:
        notes.append(f"source_warnings={warning_count}")
    if not fresh:
        notes.append("stale_artifact")
    status = (
        STATUS_CROSS_VERIFIED
        if bool(payload.get("ok", False))
        and compared_assets >= 3
        and ok_sources >= _minimum_ok_sources(total_sources, floor=5, tolerate_failures=2, min_ratio=0.70)
        and fresh
        else STATUS_SINGLE_UNVERIFIED
    )
    return _row(
        source_id="crypto_market_context",
        title="Crypto Market Context",
        category="crypto_data",
        verification_status=status,
        verification_mode="multi_provider_price_overlap",
        artifact_path=path,
        artifact_timestamp=ts,
        age_hours=_age_hours(ts, now),
        fresh=fresh,
        ok=bool(payload.get("ok", False)),
        notes=notes,
        evidence={
            "tracked_symbols": int(payload.get("tracked_symbols", 0) or 0),
            "tracked_assets": int(payload.get("tracked_assets", 0) or 0),
            "ok_sources": ok_sources,
            "total_sources": total_sources,
            "compared_assets": compared_assets,
            "warning_count": warning_count,
            "sources": {
                key: bool(value.get("ok", False))
                for key, value in (payload.get("sources") or {}).items()
                if isinstance(value, dict)
            },
        },
    )


def _fx_market_row(health_dir: Path, now: datetime) -> dict[str, Any]:
    path = health_dir / "fx_market_context_sync_latest.json"
    payload = _read_json(path)
    ts = _parse_ts(payload.get("timestamp_utc"))
    fresh = _is_fresh(ts, now, 24.0)
    notes: list[str] = []
    ok_sources = int(payload.get("ok_source_count", 0) or 0)
    total_sources = int(payload.get("source_count", 0) or 0)
    proxy_agreement_norm = float(payload.get("proxy_agreement_norm", 0.0) or 0.0)
    if ok_sources < total_sources:
        notes.append(f"partial_sources={ok_sources}/{total_sources}")
    if proxy_agreement_norm < 0.34:
        notes.append("proxy_agreement_low")
    if not fresh:
        notes.append("stale_artifact")
    status = (
        STATUS_CROSS_VERIFIED
        if bool(payload.get("ok", False))
        and ok_sources >= 2
        and int(payload.get("official_pairs", 0) or 0) >= 3
        and int(payload.get("proxy_symbols_observed", 0) or 0) >= 3
        and proxy_agreement_norm > 0.0
        and fresh
        else STATUS_SINGLE_UNVERIFIED
    )
    return _row(
        source_id="fx_market_context",
        title="FX Market Context",
        category="cross_asset_data",
        verification_status=status,
        verification_mode="official_rates_plus_market_proxies",
        artifact_path=path,
        artifact_timestamp=ts,
        age_hours=_age_hours(ts, now),
        fresh=fresh,
        ok=bool(payload.get("ok", False)),
        notes=notes,
        evidence={
            "ok_sources": ok_sources,
            "total_sources": total_sources,
            "official_pairs": int(payload.get("official_pairs", 0) or 0),
            "proxy_symbols_observed": int(payload.get("proxy_symbols_observed", 0) or 0),
            "proxy_agreement_norm": proxy_agreement_norm,
            "direct_forex_execution_supported": bool(payload.get("direct_forex_execution_supported", False)),
            "direct_forex_execution_reason": str(payload.get("direct_forex_execution_reason") or ""),
        },
    )


def _external_feeds_row(project_root: Path, now: datetime) -> dict[str, Any]:
    path = project_root / "exports" / "external_feeds" / "latest_status.json"
    payload = _read_json(path)
    ts = _parse_ts(payload.get("timestamp_utc"))
    fresh = _is_fresh(ts, now, 24.0)
    ok_count, total_count = _ok_count(payload)
    notes: list[str] = []
    fred = payload.get("fred") if isinstance(payload.get("fred"), dict) else {}
    warnings = fred.get("warnings") if isinstance(fred.get("warnings"), list) else []
    fred_ok = bool(fred.get("ok"))
    if warnings and not fred_ok:
        notes.append(f"fred_warnings={len(warnings)}")
    if not fresh:
        notes.append("stale_artifact")
    status = STATUS_SINGLE_VERIFIED if ok_count == total_count and total_count > 0 and fresh else STATUS_SINGLE_UNVERIFIED
    return _row(
        source_id="public_macro_feeds",
        title="Public Macro Feeds",
        category="macro_data",
        verification_status=status,
        verification_mode="single_source_health",
        artifact_path=path,
        artifact_timestamp=ts,
        age_hours=_age_hours(ts, now),
        fresh=fresh,
        ok=ok_count == total_count and total_count > 0,
        notes=notes,
        evidence={
            "ok_sources": ok_count,
            "total_sources": total_count,
            "sources": {key: bool(value.get("ok", False)) for key, value in payload.items() if isinstance(value, dict) and "ok" in value},
        },
    )


def _official_macro_row(health_dir: Path, now: datetime) -> dict[str, Any]:
    path = health_dir / "official_macro_context_sync_latest.json"
    payload = _read_json(path)
    ts = _parse_ts(payload.get("timestamp_utc"))
    fresh = _is_fresh(ts, now, 24.0)
    sources = payload.get("sources") if isinstance(payload.get("sources"), dict) else {}
    ok_count, total_count = _ok_count(sources)
    notes: list[str] = []
    auxiliary_degraded = sorted(
        key
        for key, value in sources.items()
        if isinstance(value, dict)
        and not bool(value.get("contract_participates", True))
        and not bool(value.get("ok", False))
    )
    if not fresh:
        notes.append("stale_artifact")
    min_ok_sources = _minimum_ok_sources(total_count, floor=4, tolerate_failures=1, min_ratio=0.80)
    if total_count > 0 and ok_count < total_count:
        notes.append(f"partial_sources={ok_count}/{total_count}")
    if auxiliary_degraded:
        notes.append(f"auxiliary_source_degraded={','.join(auxiliary_degraded)}")
    status = (
        STATUS_SINGLE_VERIFIED
        if bool(payload.get("ok", False)) and ok_count >= min_ok_sources and total_count > 0 and fresh
        else STATUS_SINGLE_UNVERIFIED
    )
    return _row(
        source_id="official_macro_context",
        title="Official Macro Context",
        category="macro_data",
        verification_status=status,
        verification_mode="single_source_health",
        artifact_path=path,
        artifact_timestamp=ts,
        age_hours=_age_hours(ts, now),
        fresh=fresh,
        ok=bool(payload.get("ok", False)),
        notes=notes,
        evidence={
            "ok_sources": ok_count,
            "total_sources": total_count,
            "min_ok_sources_required": min_ok_sources,
            "sources": {key: bool(value.get("ok", False)) for key, value in sources.items() if isinstance(value, dict)},
        },
    )


def _schwab_education_row(health_dir: Path, now: datetime) -> dict[str, Any]:
    path = health_dir / "schwab_education_context_sync_latest.json"
    payload = _read_json(path)
    ts = _parse_ts(payload.get("timestamp_utc"))
    fresh = _is_fresh(ts, now, 36.0)
    ok_count = int(payload.get("ok_source_count", 0) or 0)
    total_count = int(payload.get("source_count", 0) or 0)
    min_ok_sources_required = int(payload.get("min_ok_sources_required", total_count) or 0)
    notes: list[str] = []
    if not fresh:
        notes.append("stale_artifact")
    if int(payload.get("item_count", 0) or 0) <= 0:
        notes.append("no_items_collected")
    status = (
        STATUS_SINGLE_VERIFIED
        if bool(payload.get("ok", False)) and ok_count >= min_ok_sources_required and total_count > 0 and fresh
        else STATUS_SINGLE_UNVERIFIED
    )
    return _row(
        source_id="schwab_education_context",
        title="Schwab Education Context",
        category="education_media",
        verification_status=status,
        verification_mode="single_source_health",
        artifact_path=path,
        artifact_timestamp=ts,
        age_hours=_age_hours(ts, now),
        fresh=fresh,
        ok=bool(payload.get("ok", False)),
        notes=notes,
        evidence={
            "ok_sources": ok_count,
            "total_sources": total_count,
            "min_ok_sources_required": min_ok_sources_required,
            "item_count": int(payload.get("item_count", 0) or 0),
            "page_item_count": int(payload.get("page_item_count", 0) or 0),
            "channel_item_count": int(payload.get("channel_item_count", 0) or 0),
        },
    )


def _market_micro_row(health_dir: Path, now: datetime) -> dict[str, Any]:
    path = health_dir / "market_micro_sync_latest.json"
    payload = _read_json(path)
    ts = _parse_ts(payload.get("timestamp_utc"))
    fresh = _is_fresh(ts, now, 24.0)
    sources = payload.get("sources") if isinstance(payload.get("sources"), dict) else {}
    ok_count, total_count = _ok_count(sources)
    notes: list[str] = []
    auxiliary_degraded = sorted(
        key
        for key, value in sources.items()
        if isinstance(value, dict)
        and not bool(value.get("contract_participates", True))
        and not bool(value.get("ok", False))
    )
    critical_sources = {
        "local_micro": bool((sources.get("local_micro") or {}).get("ok", False)) if isinstance(sources, dict) else False,
        "finra_short_volume": bool((sources.get("finra_short_volume") or {}).get("ok", False)) if isinstance(sources, dict) else False,
    }
    if not fresh:
        notes.append("stale_artifact")
    min_ok_sources = _minimum_ok_sources(total_count, floor=3, tolerate_failures=1, min_ratio=0.75)
    if total_count > 0 and ok_count < total_count:
        notes.append(f"partial_sources={ok_count}/{total_count}")
    if auxiliary_degraded:
        notes.append(f"auxiliary_source_degraded={','.join(auxiliary_degraded)}")
    status = (
        STATUS_SINGLE_VERIFIED
        if bool(payload.get("ok", False))
        and total_count > 0
        and ok_count >= min_ok_sources
        and all(critical_sources.values())
        and fresh
        else STATUS_SINGLE_UNVERIFIED
    )
    return _row(
        source_id="market_micro_context",
        title="Market Micro Context",
        category="market_structure",
        verification_status=status,
        verification_mode="single_source_health",
        artifact_path=path,
        artifact_timestamp=ts,
        age_hours=_age_hours(ts, now),
        fresh=fresh,
        ok=bool(payload.get("ok", False)),
        notes=notes,
        evidence={
            "ok_sources": ok_count,
            "total_sources": total_count,
            "min_ok_sources_required": min_ok_sources,
            "critical_sources": critical_sources,
            "local_micro_symbol_count": int(((sources.get("local_micro") or {}).get("symbol_count", 0)) or 0),
            "finra_symbol_count": int(((sources.get("finra_short_volume") or {}).get("symbol_count", 0)) or 0),
        },
    )


def _sec_edgar_row(health_dir: Path, now: datetime) -> dict[str, Any]:
    path = health_dir / "sec_edgar_sync_latest.json"
    payload = _read_json(path)
    ts = _parse_ts(payload.get("timestamp_utc"))
    fresh = _is_fresh(ts, now, 36.0)
    notes: list[str] = []
    if not fresh:
        notes.append("stale_artifact")
    status = (
        STATUS_SINGLE_VERIFIED
        if bool(payload.get("ok", False))
        and bool(payload.get("ticker_map_ok", False))
        and int(payload.get("error_count", 0) or 0) == 0
        and fresh
        else STATUS_SINGLE_UNVERIFIED
    )
    return _row(
        source_id="sec_edgar_context",
        title="SEC EDGAR Context",
        category="fundamental_data",
        verification_status=status,
        verification_mode="single_source_health",
        artifact_path=path,
        artifact_timestamp=ts,
        age_hours=_age_hours(ts, now),
        fresh=fresh,
        ok=bool(payload.get("ok", False)),
        notes=notes,
        evidence={
            "requested_symbols": int(payload.get("requested_symbols", 0) or 0),
            "resolved_symbols": int(payload.get("resolved_symbols", 0) or 0),
            "tracked_symbols": int(payload.get("tracked_symbols", 0) or 0),
            "ticker_map_ok": bool(payload.get("ticker_map_ok", False)),
        },
    )


def _extended_quant_row(health_dir: Path, now: datetime) -> dict[str, Any]:
    path = health_dir / "extended_quant_context_sync_latest.json"
    payload = _read_json(path)
    ts = _parse_ts(payload.get("timestamp_utc"))
    fresh = _is_fresh(ts, now, 48.0)
    sources = payload.get("sources") if isinstance(payload.get("sources"), dict) else {}
    ok_count, total_count = _ok_count(sources)
    notes: list[str] = []
    nyfed = sources.get("nyfed_sofr") if isinstance(sources.get("nyfed_sofr"), dict) else {}
    if nyfed.get("averages_error"):
        notes.append("nyfed_partial_averages_fallback")
    if not fresh:
        notes.append("stale_artifact")
    status = STATUS_SINGLE_VERIFIED if bool(payload.get("ok", False)) and ok_count == total_count and total_count > 0 and fresh else STATUS_SINGLE_UNVERIFIED
    return _row(
        source_id="extended_quant_context",
        title="Extended Quant Context",
        category="cross_asset_data",
        verification_status=status,
        verification_mode="single_source_health",
        artifact_path=path,
        artifact_timestamp=ts,
        age_hours=_age_hours(ts, now),
        fresh=fresh,
        ok=bool(payload.get("ok", False)),
        notes=notes,
        evidence={
            "tracked_symbols": int(payload.get("tracked_symbols", 0) or 0),
            "ok_sources": ok_count,
            "total_sources": total_count,
            "sources": {key: bool(value.get("ok", False)) for key, value in sources.items() if isinstance(value, dict)},
        },
    )


def _fed_2026_stress_scenario_row(project_root: Path, now: datetime) -> dict[str, Any]:
    scenario_path = project_root / "config" / "stress_scenarios" / "fed_2026_supervisory_severely_adverse.json"
    plumbing_path = project_root / "config" / "stress_scenarios" / "fed_2026_source_plumbing.json"
    modules_path = project_root / "config" / "stress_scenarios" / "fed_2026_stress_modules.json"
    scenario = _read_json(scenario_path)
    plumbing = _read_json(plumbing_path)
    modules_payload = _read_json(modules_path)
    source = scenario.get("source") if isinstance(scenario.get("source"), dict) else {}
    ts = _parse_ts(source.get("retrieved_date"))
    domestic = scenario.get("domestic_variables") if isinstance(scenario.get("domestic_variables"), dict) else {}
    international = scenario.get("international_variables") if isinstance(scenario.get("international_variables"), dict) else {}
    anchors = scenario.get("key_stress_anchors") if isinstance(scenario.get("key_stress_anchors"), dict) else {}
    series_map = plumbing.get("series_map") if isinstance(plumbing.get("series_map"), dict) else {}
    stress_module_map = plumbing.get("stress_module_map") if isinstance(plumbing.get("stress_module_map"), dict) else {}
    internal_feature_keys = plumbing.get("internal_feature_keys") if isinstance(plumbing.get("internal_feature_keys"), list) else []
    proxy_symbols = plumbing.get("market_proxy_symbols") if isinstance(plumbing.get("market_proxy_symbols"), dict) else {}
    governance_targets = plumbing.get("governance_targets") if isinstance(plumbing.get("governance_targets"), list) else []
    stress_modules = modules_payload.get("stress_modules") if isinstance(modules_payload.get("stress_modules"), list) else []
    stress_module_ids = [str(item.get("module_id") or "") for item in stress_modules if isinstance(item, dict)]
    expected_module_ids = {
        "fed_2026_equity_crash_volatility_spike",
        "fed_2026_corporate_credit_spread_blowout",
        "fed_2026_housing_price_shock",
        "fed_2026_commercial_real_estate_shock",
        "fed_2026_unemployment_recession_shock",
        "fed_2026_global_recession_deflation_shock",
        "fed_2026_commodity_inflation_shock",
        "fed_2026_treasury_yield_shock",
        "fed_2026_us_dollar_stress",
        "fed_2026_counterparty_default_contagion_shock",
    }
    notes: list[str] = []
    if not scenario_path.exists():
        notes.append("scenario_artifact_missing")
    if not plumbing_path.exists():
        notes.append("source_plumbing_missing")
    if not modules_path.exists():
        notes.append("stress_modules_missing")
    if scenario.get("scenario_id") != "fed_2026_supervisory_severely_adverse":
        notes.append("scenario_id_mismatch")
    if modules_payload.get("scenario_id") not in {None, "fed_2026_supervisory_severely_adverse"}:
        notes.append("stress_module_scenario_id_mismatch")
    if "federalreserve.gov" not in str(source.get("url") or ""):
        notes.append("official_fed_url_missing")
    module_source = modules_payload.get("source") if isinstance(modules_payload.get("source"), dict) else {}
    if modules_payload and "federalreserve.gov" not in str(module_source.get("url") or ""):
        notes.append("stress_module_official_fed_url_missing")
    if not domestic.get("columns") or not domestic.get("rows"):
        notes.append("domestic_variables_missing")
    if not international.get("columns") or not international.get("rows"):
        notes.append("international_variables_missing")
    if not anchors:
        notes.append("key_stress_anchors_missing")
    if not series_map.get("domestic_variables") or not series_map.get("international_variables"):
        notes.append("series_map_incomplete")
    if len(stress_modules) < 10:
        notes.append(f"stress_module_count_low={len(stress_modules)}")
    missing_modules = sorted(expected_module_ids.difference(stress_module_ids))
    if missing_modules:
        notes.append(f"stress_modules_missing_ids={','.join(missing_modules[:5])}")
    if len(stress_module_map) < 10:
        notes.append("stress_module_map_incomplete")
    for module in stress_modules:
        if not isinstance(module, dict):
            continue
        module_id = str(module.get("module_id") or "")
        if module_id and module_id not in stress_module_map:
            notes.append(f"stress_module_not_plumbed={module_id}")
            break
        if not module.get("primary_series") or not module.get("internal_feature_keys"):
            notes.append(f"stress_module_contract_incomplete={module_id}")
            break
    usage_policy = modules_payload.get("usage_policy") if isinstance(modules_payload.get("usage_policy"), dict) else {}
    if modules_payload and bool(usage_policy.get("direct_execution_allowed", True)):
        notes.append("stress_modules_direct_execution_not_blocked")
    if not internal_feature_keys:
        notes.append("internal_feature_keys_missing")
    if not proxy_symbols:
        notes.append("market_proxy_symbols_missing")
    for required_target in ("source_verification", "point_in_time_event_store", "replay_hash_registry"):
        if required_target not in governance_targets:
            notes.append(f"governance_target_missing={required_target}")
    fresh = ts is not None and (_age_hours(ts, now) or 0.0) <= (365.0 * 3.0 * 24.0)
    if not fresh:
        notes.append("stale_or_missing_retrieved_date")
    ok = not notes or notes == ["stale_or_missing_retrieved_date"]
    status = STATUS_SINGLE_VERIFIED if ok and fresh else STATUS_SINGLE_UNVERIFIED
    return _row(
        source_id="fed_2026_supervisory_stress_scenario",
        title="Fed 2026 Supervisory Stress Scenario",
        category="macro_stress_scenario",
        verification_status=status,
        verification_mode="official_static_scenario_plus_internal_source_plumbing",
        artifact_path=scenario_path,
        artifact_timestamp=ts,
        age_hours=_age_hours(ts, now),
        fresh=fresh,
        ok=ok and fresh,
        notes=notes,
        evidence={
            "scenario_path": str(scenario_path),
            "plumbing_path": str(plumbing_path),
            "stress_modules_path": str(modules_path),
            "domestic_column_count": len(domestic.get("columns") or []),
            "domestic_row_count": len(domestic.get("rows") or []),
            "international_column_count": len(international.get("columns") or []),
            "international_row_count": len(international.get("rows") or []),
            "anchor_count": len(anchors),
            "stress_module_count": len(stress_modules),
            "stress_module_ids": stress_module_ids,
            "stress_module_map_count": len(stress_module_map),
            "series_map_sections": sorted(series_map.keys()),
            "internal_feature_count": len(internal_feature_keys),
            "proxy_symbol_groups": sorted(proxy_symbols.keys()),
            "governance_targets": governance_targets,
        },
    )


def build_source_verification_payload(project_root: Path = PROJECT_ROOT) -> dict[str, Any]:
    now = datetime.now(timezone.utc)
    health_dir = project_root / "governance" / "health"
    rows = [
        _market_quote_row(health_dir, now),
        _options_flow_row(health_dir, now),
        _macro_crosscheck_row(health_dir, now),
        _crypto_market_row(health_dir, now),
        _fx_market_row(health_dir, now),
        _external_feeds_row(project_root, now),
        _official_macro_row(health_dir, now),
        _schwab_education_row(health_dir, now),
        _market_micro_row(health_dir, now),
        _sec_edgar_row(health_dir, now),
        _extended_quant_row(health_dir, now),
        _fed_2026_stress_scenario_row(project_root, now),
    ]

    counts = {
        STATUS_CROSS_VERIFIED: sum(1 for row in rows if row["verification_status"] == STATUS_CROSS_VERIFIED),
        STATUS_SINGLE_VERIFIED: sum(1 for row in rows if row["verification_status"] == STATUS_SINGLE_VERIFIED),
        STATUS_SINGLE_UNVERIFIED: sum(1 for row in rows if row["verification_status"] == STATUS_SINGLE_UNVERIFIED),
    }
    unverified = [row["source_id"] for row in rows if row["verification_status"] == STATUS_SINGLE_UNVERIFIED]
    warnings = [row["source_id"] for row in rows if row["notes"]]
    return {
        "timestamp_utc": now.isoformat(),
        "overall": {
            "all_cross_verified": counts[STATUS_CROSS_VERIFIED] == len(rows),
            "all_verified": counts[STATUS_SINGLE_UNVERIFIED] == 0,
            "counts": counts,
            "total_sources": len(rows),
            "unverified_sources": unverified,
            "sources_with_notes": warnings,
        },
        "sources": rows,
    }


def _render_markdown(payload: dict[str, Any]) -> str:
    overall = payload.get("overall") if isinstance(payload.get("overall"), dict) else {}
    counts = overall.get("counts") if isinstance(overall.get("counts"), dict) else {}
    lines = [
        f"# Source Verification Report ({payload.get('timestamp_utc', '')})",
        f"- all_verified: {bool(overall.get('all_verified', False))}",
        f"- all_cross_verified: {bool(overall.get('all_cross_verified', False))}",
        f"- cross_verified: {int(counts.get(STATUS_CROSS_VERIFIED, 0) or 0)}",
        f"- single_source_verified: {int(counts.get(STATUS_SINGLE_VERIFIED, 0) or 0)}",
        f"- single_source_unverified: {int(counts.get(STATUS_SINGLE_UNVERIFIED, 0) or 0)}",
        "",
        "| Source | Status | Mode | Fresh | Age (h) | Notes |",
        "| --- | --- | --- | --- | ---: | --- |",
    ]
    for row in payload.get("sources", []):
        notes = ", ".join(row.get("notes", [])) if isinstance(row.get("notes"), list) and row.get("notes") else "none"
        age = row.get("age_hours")
        age_text = "" if age is None else str(age)
        lines.append(
            "| {title} | {status} | {mode} | {fresh} | {age} | {notes} |".format(
                title=str(row.get("title", "")),
                status=str(row.get("verification_status", "")),
                mode=str(row.get("verification_mode", "")),
                fresh=str(bool(row.get("fresh", False))).lower(),
                age=age_text,
                notes=notes,
            )
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize source verification and cross-check coverage.")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    payload = build_source_verification_payload(PROJECT_ROOT)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    HEALTH_DIR.mkdir(parents=True, exist_ok=True)

    day = datetime.now(timezone.utc).strftime("%Y%m%d")
    out_json = REPORTS_DIR / f"source_verification_{day}.json"
    out_md = REPORTS_DIR / f"source_verification_{day}.md"
    latest_json = HEALTH_DIR / "source_verification_latest.json"
    latest_md = REPORTS_DIR / "source_verification_latest.md"

    rendered_md = _render_markdown(payload)
    json_text = json.dumps(payload, ensure_ascii=True, indent=2) + "\n"

    out_json.write_text(json_text, encoding="utf-8")
    out_md.write_text(rendered_md, encoding="utf-8")
    latest_json.write_text(json_text, encoding="utf-8")
    latest_md.write_text(rendered_md, encoding="utf-8")

    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        overall = payload.get("overall", {})
        print(
            "source_verification all_verified={all_verified} cross_verified={cross_verified} single_verified={single_verified} unverified={unverified}".format(
                all_verified=str(bool(overall.get("all_verified", False))).lower(),
                cross_verified=int((((overall.get("counts") or {}).get(STATUS_CROSS_VERIFIED, 0)) or 0)),
                single_verified=int((((overall.get("counts") or {}).get(STATUS_SINGLE_VERIFIED, 0)) or 0)),
                unverified=int((((overall.get("counts") or {}).get(STATUS_SINGLE_UNVERIFIED, 0)) or 0)),
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
