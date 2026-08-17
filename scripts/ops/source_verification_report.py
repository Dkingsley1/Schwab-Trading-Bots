#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from zoneinfo import ZoneInfo
except Exception:  # pragma: no cover - zoneinfo is available on normal Python 3.9+ runtimes.
    ZoneInfo = None

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.central_bank_liquidity import (
    CENTRAL_BANK_LIQUIDITY_FEATURE_KEYS,
    assess_central_bank_liquidity_context,
)
from core.global_central_bank_context import (
    CENTRAL_BANK_CROSS_SOURCE_FEATURE_KEYS,
    GLOBAL_CENTRAL_BANK_FEATURE_KEYS,
    assess_central_bank_cross_source_context,
    assess_global_central_bank_context,
)
from core.decision_context_mesh import (
    DECISION_CONTEXT_MESH_FEATURE_KEYS,
    assess_decision_context_mesh,
)

REPORTS_DIR = PROJECT_ROOT / "exports" / "reports"
HEALTH_DIR = PROJECT_ROOT / "governance" / "health"

STATUS_CROSS_VERIFIED = "cross_verified"
STATUS_SINGLE_VERIFIED = "single_source_verified"
STATUS_SINGLE_UNVERIFIED = "single_source_unverified"
OPTIONS_CONTEXT_SOURCE_ID = "options_context_mesh"
OPTIONS_CONTEXT_LEGACY_SOURCE_ID = "polygon_unusual_whales_options_context"
SOURCE_CRITICALITY = {
    "market_quote_profiles": "decision_critical",
    "macro_crossstack": "decision_critical",
    "crypto_market_context": "decision_critical",
    "free_equity_reference_context": "decision_critical",
    "fx_market_context": "decision_critical",
    "public_macro_feeds": "decision_critical",
    "official_macro_context": "decision_critical",
    "central_bank_liquidity_context": "decision_critical",
    "global_central_bank_context": "decision_context",
    "central_bank_cross_source_context": "decision_context",
    "decision_context_mesh": "decision_context",
    "market_micro_context": "decision_critical",
    "sec_edgar_context": "decision_context",
    OPTIONS_CONTEXT_SOURCE_ID: "decision_context",
    "schwab_symbol_news": "decision_context",
    "ticker_news_context": "decision_context",
    "extended_quant_context": "decision_context",
    "public_policy_context": "decision_context",
    "schwab_education_context": "optional_enrichment",
    "fed_2026_supervisory_stress_scenario": "optional_enrichment",
}


def _read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


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


def _ordered_unique(items: list[Any]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for raw in items:
        text = str(raw or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out


def _minimum_ok_sources(total: int, *, floor: int = 1, tolerate_failures: int = 1, min_ratio: float = 0.75) -> int:
    if int(total) <= 0:
        return 0
    ratio_target = int(round(float(total) * float(min_ratio)))
    tolerated_target = int(total) - max(int(tolerate_failures), 0)
    return min(int(total), max(int(floor), ratio_target, tolerated_target))


def _safe_float(raw: Any, default: float = 0.0) -> float:
    try:
        return float(raw)
    except Exception:
        return float(default)


def _source_confidence_components(
    *,
    verification_status: str,
    fresh: bool,
    ok: bool,
    notes: list[str],
    evidence: dict[str, Any],
) -> dict[str, float]:
    status_score = (
        1.0
        if verification_status == STATUS_CROSS_VERIFIED
        else 0.84
        if verification_status == STATUS_SINGLE_VERIFIED
        else 0.30
    )
    freshness_score = 1.0 if fresh else 0.20
    health_score = 1.0 if ok else 0.20
    provider_total = max(
        int(_safe_float(evidence.get("total_sources"), 0.0) or 0),
        int(_safe_float(evidence.get("source_count"), 0.0) or 0),
        1 if _safe_float(evidence.get("symbols_requested"), 0.0) > 0 else 0,
    )
    provider_ok = max(
        int(_safe_float(evidence.get("ok_sources"), 0.0) or 0),
        int(_safe_float(evidence.get("effective_ok_sources"), 0.0) or 0),
        1 if _safe_float(evidence.get("symbols_with_chain"), 0.0) > 0 else 0,
    )
    provider_score = min(provider_ok / max(provider_total, 1), 1.0) if provider_total > 0 else (1.0 if ok else 0.0)
    schema_score = 0.72
    if evidence.get("cross_profile_ok") is True or evidence.get("options_backbone_ok") is True:
        schema_score = 0.88
    if evidence.get("ticker_map_ok") is True:
        schema_score = 0.92
    notes_penalty = min(len([item for item in notes if str(item).strip()]) * 0.035, 0.18)
    return {
        "status_score": round(status_score, 6),
        "freshness_score": round(freshness_score, 6),
        "health_score": round(health_score, 6),
        "provider_score": round(provider_score, 6),
        "schema_score": round(schema_score, 6),
        "notes_penalty": round(notes_penalty, 6),
    }


def _source_confidence_score(components: dict[str, float]) -> float:
    score = (
        0.30 * _safe_float(components.get("status_score"), 0.0)
        + 0.22 * _safe_float(components.get("freshness_score"), 0.0)
        + 0.18 * _safe_float(components.get("health_score"), 0.0)
        + 0.18 * _safe_float(components.get("provider_score"), 0.0)
        + 0.12 * _safe_float(components.get("schema_score"), 0.0)
        - _safe_float(components.get("notes_penalty"), 0.0)
    )
    return round(max(0.0, min(score, 1.0)), 6)


def _grade(score: float, *, complete: bool = False) -> str:
    if complete:
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


def _market_closed_for_local_micro(now: datetime) -> bool:
    if ZoneInfo is not None:
        local = now.astimezone(ZoneInfo("America/New_York"))
    else:
        local = now.astimezone(timezone.utc)
    if local.weekday() >= 5:
        return True
    minutes = local.hour * 60 + local.minute
    return minutes < (9 * 60 + 30) or minutes > (16 * 60)


def _market_holiday_pause_observed(health_dir: Path, now: datetime) -> bool:
    for path in sorted(health_dir.glob("data_ingress_latest_*_equities_schwab.json"))[:200]:
        payload = _read_json(path)
        if str(payload.get("pause_reason") or "").strip().lower() != "holiday":
            continue
        if str(payload.get("loop_state") or "").strip().lower() != "paused_session_gate":
            continue
        ts = _parse_ts(payload.get("timestamp_utc"))
        if _is_fresh(ts, now, 12.0):
            return True
    return False


def _row_has_actionable_notes(row: dict[str, Any]) -> bool:
    notes = [str(item or "").strip() for item in row.get("notes") or [] if str(item or "").strip()]
    if not notes:
        return False
    verification_status = str(row.get("verification_status") or "")
    if verification_status == STATUS_SINGLE_UNVERIFIED:
        return True
    evidence = row.get("evidence") if isinstance(row.get("evidence"), dict) else {}
    accepted_note_tokens: set[str] = set()
    if verification_status == STATUS_CROSS_VERIFIED:
        accepted_note_tokens.update({"cross_verified_partial_sources"})
        accepted_note_tokens.update({"cross_verified_source_warnings"})
    if bool(evidence.get("market_closed_local_micro_fallback")):
        accepted_note_tokens.update({"local_micro_absent_market_closed"})
    if bool(evidence.get("external_micro_reference_verified_fallback")):
        accepted_note_tokens.update({"local_micro_absent_external_reference_verified"})
    if bool(evidence.get("official_rate_only_holiday_fallback")):
        accepted_note_tokens.update({"market_proxy_absent_market_closed"})
    if bool(evidence.get("official_plus_twelvedata_verified_fallback")):
        accepted_note_tokens.update({"market_proxy_absent_direct_fx_unavailable"})
    if bool(evidence.get("official_reference_rates_only_direct_fx_unavailable")):
        accepted_note_tokens.update({"official_reference_rates_only_direct_fx_unavailable"})
    if bool(evidence.get("optional_unconfigured")):
        accepted_note_tokens.update({"optional_options_flow_credentials_not_configured"})
    if bool(evidence.get("free_options_chain_ok")) and bool(evidence.get("options_backbone_ok")):
        accepted_note_tokens.update({"polygon_api_key_missing"})
    if bool(evidence.get("official_macro_context_verified_partial_public_feeds")):
        accepted_note_tokens.update({"official_macro_context_verified_partial_public_feeds"})
    if bool(evidence.get("world_bank_partial_verified")):
        accepted_note_tokens.update({"world_bank_indicators_partial"})
    for note in notes:
        if note in accepted_note_tokens:
            continue
        if note.startswith("partial_sources=") and accepted_note_tokens:
            continue
        if note.startswith("source_warnings=") and "cross_verified_source_warnings" in accepted_note_tokens:
            continue
        if note.startswith("fred_warnings=") and bool(evidence.get("official_macro_context_verified_partial_public_feeds")):
            continue
        if note.startswith("world_bank_indicators_partial=") and bool(evidence.get("world_bank_partial_verified")):
            continue
        return True
    return False


def _refresh_command_for_source(project_root: Path, source_id: str) -> list[str]:
    opsctl = str(project_root / "scripts" / "ops" / "opsctl.sh")
    mapping: dict[str, list[str]] = {
        "market_quote_profiles": [str(project_root / ".venv314" / "bin" / "python"), str(project_root / "scripts" / "data_source_divergence_bot.py"), "--json"],
        OPTIONS_CONTEXT_SOURCE_ID: [opsctl, "options-flow-sync", "--json"],
        OPTIONS_CONTEXT_LEGACY_SOURCE_ID: [opsctl, "options-flow-sync", "--json"],
        "macro_crossstack": [opsctl, "macro-crosscheck", "--json"],
        "crypto_market_context": [opsctl, "crypto-market-sync", "--json"],
        "free_equity_reference_context": [
            opsctl,
            "free-equity-reference-sync",
            "--max-symbols",
            "40",
            "--timeout",
            "2.5",
            "--max-runtime-seconds",
            "45",
            "--json",
        ],
        "fx_market_context": [opsctl, "fx-market-sync", "--json"],
        "public_macro_feeds": [opsctl, "macro-context-sync", "--json"],
        "official_macro_context": [opsctl, "macro-context-sync", "--json"],
        "central_bank_liquidity_context": [opsctl, "macro-context-sync", "--json"],
        "global_central_bank_context": [opsctl, "global-central-bank-sync", "--json"],
        "central_bank_cross_source_context": [opsctl, "central-bank-context-sync", "--json"],
        "decision_context_mesh": [opsctl, "decision-context-sync", "--json"],
        "schwab_education_context": [opsctl, "schwab-education-sync", "--json"],
        "schwab_symbol_news": [opsctl, "schwab-symbol-news-sync", "--max-runtime-seconds", "180", "--json"],
        "ticker_news_context": [opsctl, "ticker-news-sync", "--max-runtime-seconds", "240", "--json"],
        "market_micro_context": [opsctl, "market-micro-sync", "--json"],
        "sec_edgar_context": [opsctl, "sec-edgar-sync", "--json"],
        "extended_quant_context": [opsctl, "extended-quant-sync", "--json"],
        "public_policy_context": [opsctl, "public-policy-sync", "--json"],
        "fed_2026_supervisory_stress_scenario": [opsctl, "source-verification", "--json"],
    }
    return mapping.get(str(source_id), [opsctl, "source-verification", "--json"])


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
    aliases: list[str] | None = None,
) -> dict[str, Any]:
    clean_notes = [str(item) for item in notes if str(item).strip()]
    confidence_components = _source_confidence_components(
        verification_status=verification_status,
        fresh=fresh,
        ok=ok,
        notes=clean_notes,
        evidence=evidence,
    )
    return {
        "source_id": source_id,
        "title": title,
        "category": category,
        "criticality": SOURCE_CRITICALITY.get(source_id, "decision_context"),
        "verification_status": verification_status,
        "verification_mode": verification_mode,
        "artifact_path": str(artifact_path),
        "artifact_timestamp_utc": artifact_timestamp.isoformat() if artifact_timestamp is not None else None,
        "age_hours": _round_age(age_hours),
        "fresh": bool(fresh),
        "ok": bool(ok),
        "notes": clean_notes,
        "evidence": evidence,
        "aliases": _ordered_unique(list(aliases or [])),
        "source_confidence_score": _source_confidence_score(confidence_components),
        "confidence_components": confidence_components,
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
    yahoo_options_chain = sources.get("yahoo_options_chain") if isinstance(sources.get("yahoo_options_chain"), dict) else {}
    cboe_delayed_options = sources.get("cboe_delayed_options") if isinstance(sources.get("cboe_delayed_options"), dict) else {}
    coverage = payload.get("coverage") if isinstance(payload.get("coverage"), dict) else {}
    polygon_ok = bool(polygon.get("ok", False))
    free_options_chain_ok = bool(
        coverage.get("free_options_chain_ok", False)
        or yahoo_options_chain.get("ok", False)
        or cboe_delayed_options.get("ok", False)
    )
    unusual_whales_ok = bool(unusual_whales_api.get("ok", False) or unusual_whales_export.get("ok", False))
    unusual_whales_expected = bool(
        unusual_whales_api.get("expected", False)
        or unusual_whales_export.get("expected", False)
        or unusual_whales_export.get("configured", False)
    )
    symbols_with_chain = int(payload.get("symbols_with_chain", 0) or 0)
    symbols_with_polygon_chain = int(payload.get("symbols_with_polygon_chain", coverage.get("symbols_with_polygon_chain", 0)) or 0)
    if symbols_with_polygon_chain <= 0 and polygon_ok and not free_options_chain_ok:
        symbols_with_polygon_chain = symbols_with_chain
    polygon_backbone_ok = bool(coverage.get("polygon_backbone_ok", False) or (polygon_ok and symbols_with_polygon_chain > 0))
    options_backbone_ok = bool(polygon_backbone_ok or free_options_chain_ok)
    context_profile = str(payload.get("context_profile") or coverage.get("context_profile") or "").strip()
    if not context_profile:
        context_profile = (
            "multi_provider_full"
            if polygon_backbone_ok and unusual_whales_ok
            else "polygon_primary_only"
            if polygon_backbone_ok and not unusual_whales_expected
            else "polygon_backbone_only"
            if polygon_backbone_ok
            else "free_options_chain_plus_overlay"
            if free_options_chain_ok and unusual_whales_ok
            else "free_options_chain_only"
            if free_options_chain_ok
            else "unusual_whales_overlay_only"
            if unusual_whales_ok
            else "unavailable"
        )
    overall_status = str(payload.get("overall_status") or ("ready" if payload.get("ok", False) else "blocked")).strip()
    auth_issue = str(payload.get("auth_issue") or "").strip()
    optional_unconfigured = bool(
        auth_issue == "options_flow_credentials_missing"
        and not polygon_ok
        and not options_backbone_ok
        and not unusual_whales_ok
    )
    if optional_unconfigured:
        notes.append("optional_options_flow_credentials_not_configured")
    else:
        if not polygon_ok and not free_options_chain_ok:
            for err in list(polygon.get("errors") or [])[:3]:
                text = str(err or "").strip()
                if text:
                    notes.append(text)
        if unusual_whales_expected and (not unusual_whales_ok) and bool(payload.get("operator_action_required", False)):
            notes.append(auth_issue or "options_flow_source_unavailable")
        if context_profile and context_profile not in {"multi_provider_full", "polygon_primary_only", "free_options_chain_plus_overlay", "free_options_chain_only"}:
            notes.append(f"context_profile={context_profile}")
        if overall_status and overall_status != "ready":
            notes.append(f"overall_status={overall_status}")
    if not fresh:
        notes.append("stale_artifact")
    if bool(payload.get("ok", False)) and fresh and context_profile == "multi_provider_full" and overall_status == "ready":
        status = STATUS_CROSS_VERIFIED
    elif bool(payload.get("ok", False)) and fresh and options_backbone_ok and overall_status == "ready":
        status = STATUS_SINGLE_VERIFIED
    elif optional_unconfigured and fresh:
        status = STATUS_SINGLE_VERIFIED
    else:
        status = STATUS_SINGLE_UNVERIFIED
    effective_ok = bool(payload.get("ok", False)) or (optional_unconfigured and fresh)
    return _row(
        source_id=OPTIONS_CONTEXT_SOURCE_ID,
        title="Options Context Mesh",
        category="derivatives_data",
        verification_status=status,
        verification_mode="multi_provider_options_flow_with_free_chain_fallback",
        artifact_path=path,
        artifact_timestamp=ts,
        age_hours=_age_hours(ts, now),
        fresh=fresh,
        ok=effective_ok,
        notes=notes,
        evidence={
            "symbols_requested": int(payload.get("symbols_requested", 0) or 0),
            "symbols_with_chain": symbols_with_chain,
            "symbols_with_metrics": int(payload.get("symbols_with_metrics", 0) or 0),
            "symbols_with_polygon_chain": symbols_with_polygon_chain,
            "symbols_with_free_options": int(payload.get("symbols_with_free_options", 0) or 0),
            "polygon_ok": polygon_ok,
            "polygon_backbone_ok": polygon_backbone_ok,
            "free_options_chain_ok": free_options_chain_ok,
            "options_backbone_ok": options_backbone_ok,
            "unusual_whales_api_ok": bool(unusual_whales_api.get("ok", False)),
            "unusual_whales_export_ok": bool(unusual_whales_export.get("ok", False)),
            "yahoo_options_chain_ok": bool(yahoo_options_chain.get("ok", False)),
            "cboe_delayed_options_ok": bool(cboe_delayed_options.get("ok", False)),
            "unusual_whales_expected": unusual_whales_expected,
            "context_profile": context_profile,
            "overall_status": overall_status,
            "auth_issue": auth_issue,
            "optional_unconfigured": optional_unconfigured,
            "coverage_score": payload.get("coverage_score"),
        },
        aliases=[OPTIONS_CONTEXT_LEGACY_SOURCE_ID],
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


def _free_equity_reference_row(health_dir: Path, now: datetime) -> dict[str, Any]:
    path = health_dir / "free_equity_reference_context_latest.json"
    payload = _read_json(path)
    ts = _parse_ts(payload.get("timestamp_utc"))
    fresh = _is_fresh(ts, now, 12.0)
    sources = payload.get("sources") if isinstance(payload.get("sources"), dict) else {}
    ok_count, total_count = _ok_count(sources)
    symbols_with_reference = int(payload.get("symbols_with_reference", 0) or 0)
    requested_symbols = int(payload.get("requested_symbol_count", 0) or 0)
    notes: list[str] = []
    if total_count > 0 and ok_count < total_count:
        notes.append(f"partial_sources={ok_count}/{total_count}")
    if symbols_with_reference <= 0:
        notes.append("no_equity_reference_symbols")
    if not fresh:
        notes.append("stale_artifact")
    status = (
        STATUS_SINGLE_VERIFIED
        if bool(payload.get("ok", False))
        and fresh
        and ok_count >= 1
        and symbols_with_reference > 0
        else STATUS_SINGLE_UNVERIFIED
    )
    return _row(
        source_id="free_equity_reference_context",
        title="Free Equity Reference Context",
        category="equity_market_data",
        verification_status=status,
        verification_mode="free_public_quote_reference_mesh",
        artifact_path=path,
        artifact_timestamp=ts,
        age_hours=_age_hours(ts, now),
        fresh=fresh,
        ok=bool(payload.get("ok", False)) and fresh,
        notes=notes,
        evidence={
            "requested_symbols": requested_symbols,
            "symbols_with_reference": symbols_with_reference,
            "ok_sources": ok_count,
            "total_sources": total_count,
            "sources": {key: bool(value.get("ok", False)) for key, value in sources.items() if isinstance(value, dict)},
        },
    )


def _fx_market_row(health_dir: Path, now: datetime) -> dict[str, Any]:
    path = health_dir / "fx_market_context_sync_latest.json"
    payload = _read_json(path)
    ts = _parse_ts(payload.get("timestamp_utc"))
    fresh = _is_fresh(ts, now, 24.0)
    notes: list[str] = []
    sources = payload.get("sources") if isinstance(payload.get("sources"), dict) else {}
    ok_sources = int(payload.get("ok_source_count", 0) or 0)
    total_sources = int(payload.get("source_count", 0) or 0)
    official_pairs = int(payload.get("official_pairs", 0) or 0)
    proxy_symbols_observed = int(payload.get("proxy_symbols_observed", 0) or 0)
    proxy_agreement_norm = float(payload.get("proxy_agreement_norm", 0.0) or 0.0)
    if sources:
        source_ok_count, source_total_count = _ok_count(sources)
        if total_sources <= 0 and source_total_count > 0:
            total_sources = source_total_count
        if ok_sources <= 0 and source_ok_count > 0:
            ok_sources = source_ok_count
    ecb = sources.get("ecb") if isinstance(sources.get("ecb"), dict) else {}
    fed_h10 = sources.get("fed_h10") if isinstance(sources.get("fed_h10"), dict) else {}
    if official_pairs <= 0:
        official_pairs = max(official_pairs, int(fed_h10.get("pair_count", 0) or 0))
    twelve_data = sources.get("twelve_data") if isinstance(sources.get("twelve_data"), dict) else {}
    twelve_data_ok = bool(twelve_data.get("ok", False)) and int(twelve_data.get("pairs_ok", 0) or 0) > 0
    direct_forex_execution_supported = bool(payload.get("direct_forex_execution_supported", False))
    official_rate_only_holiday_fallback = bool(
        bool(payload.get("ok", False))
        and ok_sources >= 3
        and official_pairs >= 3
        and proxy_symbols_observed <= 0
        and (_market_closed_for_local_micro(now) or _market_holiday_pause_observed(health_dir, now))
        and fresh
    )
    official_plus_twelvedata_verified_fallback = bool(
        bool(payload.get("ok", False))
        and ok_sources >= 4
        and official_pairs >= 3
        and twelve_data_ok
        and not direct_forex_execution_supported
        and proxy_symbols_observed <= 0
        and fresh
    )
    official_reference_rates_only_direct_fx_unavailable = bool(
        bool(payload.get("ok", False))
        and ok_sources >= 2
        and official_pairs >= 3
        and bool(ecb.get("ok", False))
        and bool(fed_h10.get("ok", False))
        and not direct_forex_execution_supported
        and proxy_symbols_observed <= 0
        and fresh
    )
    if ok_sources < total_sources:
        notes.append(f"partial_sources={ok_sources}/{total_sources}")
    if official_rate_only_holiday_fallback:
        notes.append("market_proxy_absent_market_closed")
    elif official_plus_twelvedata_verified_fallback:
        notes.append("market_proxy_absent_direct_fx_unavailable")
    elif official_reference_rates_only_direct_fx_unavailable:
        notes.append("official_reference_rates_only_direct_fx_unavailable")
    elif proxy_agreement_norm < 0.34:
        notes.append("proxy_agreement_low")
    if not fresh:
        notes.append("stale_artifact")
    if (
        bool(payload.get("ok", False))
        and ok_sources >= 2
        and official_pairs >= 3
        and proxy_symbols_observed >= 3
        and proxy_agreement_norm > 0.0
        and fresh
    ):
        status = STATUS_CROSS_VERIFIED
    elif (
        official_rate_only_holiday_fallback
        or official_plus_twelvedata_verified_fallback
        or official_reference_rates_only_direct_fx_unavailable
    ):
        status = STATUS_SINGLE_VERIFIED
    else:
        status = STATUS_SINGLE_UNVERIFIED
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
            "official_pairs": official_pairs,
            "proxy_symbols_observed": proxy_symbols_observed,
            "proxy_agreement_norm": proxy_agreement_norm,
            "official_rate_only_holiday_fallback": official_rate_only_holiday_fallback,
            "official_plus_twelvedata_verified_fallback": official_plus_twelvedata_verified_fallback,
            "official_reference_rates_only_direct_fx_unavailable": official_reference_rates_only_direct_fx_unavailable,
            "twelve_data_ok": twelve_data_ok,
            "direct_forex_execution_supported": direct_forex_execution_supported,
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
    official_macro_path = project_root / "governance" / "health" / "official_macro_context_sync_latest.json"
    official_macro = _read_json(official_macro_path)
    official_macro_ts = _parse_ts(official_macro.get("timestamp_utc"))
    official_macro_fresh = _is_fresh(official_macro_ts, now, 24.0)
    official_sources = official_macro.get("sources") if isinstance(official_macro.get("sources"), dict) else {}
    official_ok_count, official_total_count = _ok_count(official_sources)
    official_min_ok = _minimum_ok_sources(official_total_count, floor=4, tolerate_failures=1, min_ratio=0.80)
    official_macro_verified = bool(
        bool(official_macro.get("ok", False))
        and official_macro_fresh
        and official_total_count > 0
        and official_ok_count >= official_min_ok
    )
    public_feeds_fully_ok = bool(ok_count == total_count and total_count > 0)
    official_macro_context_verified_partial_public_feeds = bool(
        fresh and ok_count > 0 and total_count > 0 and not public_feeds_fully_ok and official_macro_verified
    )
    fred = payload.get("fred") if isinstance(payload.get("fred"), dict) else {}
    warnings = fred.get("warnings") if isinstance(fred.get("warnings"), list) else []
    fred_ok = bool(fred.get("ok"))
    if ok_count < total_count:
        notes.append(f"partial_sources={ok_count}/{total_count}")
    if warnings and not fred_ok:
        notes.append(f"fred_warnings={len(warnings)}")
    if official_macro_context_verified_partial_public_feeds:
        notes.append("official_macro_context_verified_partial_public_feeds")
    if not fresh:
        notes.append("stale_artifact")
    verified = bool((public_feeds_fully_ok or official_macro_context_verified_partial_public_feeds) and fresh)
    effective_ok_count = ok_count
    effective_total_count = total_count
    if official_macro_context_verified_partial_public_feeds:
        effective_ok_count = max(ok_count, official_ok_count, official_min_ok)
        effective_total_count = max(total_count, effective_ok_count)
    status = STATUS_SINGLE_VERIFIED if verified else STATUS_SINGLE_UNVERIFIED
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
        ok=verified,
        notes=notes,
        evidence={
            "ok_sources": ok_count,
            "total_sources": total_count,
            "raw_public_ok_sources": ok_count,
            "raw_public_total_sources": total_count,
            "effective_ok_sources": effective_ok_count,
            "effective_total_sources": effective_total_count,
            "official_macro_context_verified_partial_public_feeds": official_macro_context_verified_partial_public_feeds,
            "official_macro_ok_sources": official_ok_count,
            "official_macro_total_sources": official_total_count,
            "official_macro_min_ok_sources_required": official_min_ok,
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
    central_source = sources.get("central_bank_liquidity") if isinstance(sources.get("central_bank_liquidity"), dict) else {}
    central_liquidity_ok = bool(central_source.get("ok", False))
    if not fresh:
        notes.append("stale_artifact")
    min_ok_sources = _minimum_ok_sources(total_count, floor=4, tolerate_failures=1, min_ratio=0.80)
    if total_count > 0 and ok_count < total_count:
        notes.append(f"partial_sources={ok_count}/{total_count}")
    if auxiliary_degraded:
        notes.append(f"auxiliary_source_degraded={','.join(auxiliary_degraded)}")
    if not central_liquidity_ok:
        notes.append("central_bank_liquidity_contract_not_ready")
    status = (
        STATUS_SINGLE_VERIFIED
        if bool(payload.get("ok", False)) and central_liquidity_ok and ok_count >= min_ok_sources and total_count > 0 and fresh
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
            "central_bank_liquidity_ok": central_liquidity_ok,
            "sources": {key: bool(value.get("ok", False)) for key, value in sources.items() if isinstance(value, dict)},
        },
    )


def _central_bank_liquidity_row(project_root: Path, now: datetime) -> dict[str, Any]:
    path = project_root / "exports" / "external_context" / "central_bank_liquidity_latest.json"
    payload = _read_json(path)
    ts = _parse_ts(payload.get("timestamp_utc"))
    fresh = _is_fresh(ts, now, 24.0)
    coverage = payload.get("coverage") if isinstance(payload.get("coverage"), dict) else {}
    missing = coverage.get("missing_required_series") if isinstance(coverage.get("missing_required_series"), list) else []
    stale = coverage.get("stale_required_series") if isinstance(coverage.get("stale_required_series"), list) else []
    unusable = coverage.get("unusable_required_series") if isinstance(coverage.get("unusable_required_series"), list) else []
    future_selected = bool(coverage.get("future_observation_selected", False))
    coverage_ratio = float(coverage.get("required_coverage_ratio", 0.0) or 0.0)
    features = payload.get("global_features") if isinstance(payload.get("global_features"), dict) else {}
    assessment = assess_central_bank_liquidity_context(payload, now_utc=now)
    verified = bool(fresh and assessment.get("ready", False))
    notes: list[str] = []
    if not fresh:
        notes.append("stale_artifact")
    if missing:
        notes.append(f"missing_required_series={','.join(str(item) for item in missing)}")
    if stale:
        notes.append(f"stale_required_series={','.join(str(item) for item in stale)}")
    if future_selected:
        notes.append("future_observation_selected")
    if coverage_ratio < 1.0:
        notes.append(f"coverage_ratio={coverage_ratio:.6f}")
    for reason in assessment.get("reasons", []):
        if str(reason) not in notes:
            notes.append(str(reason))
    return _row(
        source_id="central_bank_liquidity_context",
        title="Central Bank And Fed Liquidity Context",
        category="macro_data",
        verification_status=STATUS_SINGLE_VERIFIED if verified else STATUS_SINGLE_UNVERIFIED,
        verification_mode="official_series_coverage_and_freshness",
        artifact_path=path,
        artifact_timestamp=ts,
        age_hours=_age_hours(ts, now),
        fresh=fresh,
        ok=verified,
        notes=notes,
        evidence={
            "required_coverage_ratio": coverage_ratio,
            "required_series": coverage.get("required_series", []),
            "available_series": coverage.get("available_series", []),
            "missing_required_series": missing,
            "stale_required_series": stale,
            "unusable_required_series": unusable,
            "as_of_date": coverage.get("as_of_date"),
            "latest_observation_dates": coverage.get("latest_observation_dates", {}),
            "latest_observation_age_days": coverage.get("latest_observation_age_days", {}),
            "future_observations_excluded": coverage.get("future_observations_excluded", {}),
            "future_observation_selected": future_selected,
            "feature_count": len(features),
            "required_feature_count": len(CENTRAL_BANK_LIQUIDITY_FEATURE_KEYS),
            "consumer_contract": assessment,
            "methodology": payload.get("methodology", {}),
        },
    )


def _global_central_bank_row(project_root: Path, now: datetime) -> dict[str, Any]:
    path = project_root / "exports" / "external_context" / "global_central_bank_context_latest.json"
    payload = _read_json(path)
    ts = _parse_ts(payload.get("timestamp_utc"))
    fresh = _is_fresh(ts, now, 48.0)
    coverage = payload.get("coverage") if isinstance(payload.get("coverage"), dict) else {}
    features = payload.get("global_features") if isinstance(payload.get("global_features"), dict) else {}
    assessment = assess_global_central_bank_context(payload, now_utc=now)
    verified = bool(fresh and assessment.get("ready", False))
    notes = [str(reason) for reason in assessment.get("reasons", [])]
    if not fresh and "stale_artifact" not in notes:
        notes.append("stale_artifact")
    return _row(
        source_id="global_central_bank_context",
        title="Global Central Bank Policy And Balance Sheet Context",
        category="macro_data",
        verification_status=STATUS_SINGLE_VERIFIED if verified else STATUS_SINGLE_UNVERIFIED,
        verification_mode="bis_member_reported_point_in_time_coverage",
        artifact_path=path,
        artifact_timestamp=ts,
        age_hours=_age_hours(ts, now),
        fresh=fresh,
        ok=verified,
        notes=notes,
        evidence={
            "registry_bank_count": coverage.get("registry_bank_count", 0),
            "ready_bank_count": coverage.get("ready_bank_count", 0),
            "tier_1_coverage_ratio": coverage.get("tier_1_coverage_ratio", 0.0),
            "important_bank_coverage_ratio": coverage.get("important_bank_coverage_ratio", 0.0),
            "policy_rate_coverage_ratio": coverage.get("policy_rate_coverage_ratio", 0.0),
            "balance_sheet_coverage_ratio": coverage.get("balance_sheet_coverage_ratio", 0.0),
            "raw_policy_area_count": coverage.get("raw_policy_area_count", 0),
            "raw_balance_sheet_area_count": coverage.get("raw_balance_sheet_area_count", 0),
            "future_observations_excluded": coverage.get("future_observations_excluded", {}),
            "future_observation_selected": coverage.get("future_observation_selected", False),
            "feature_count": len(features),
            "required_feature_count": len(GLOBAL_CENTRAL_BANK_FEATURE_KEYS),
            "consumer_contract": assessment,
            "methodology": payload.get("methodology", {}),
        },
    )


def _central_bank_cross_source_row(project_root: Path, now: datetime) -> dict[str, Any]:
    path = project_root / "exports" / "external_context" / "central_bank_cross_source_latest.json"
    payload = _read_json(path)
    ts = _parse_ts(payload.get("timestamp_utc"))
    fresh = _is_fresh(ts, now, 24.0)
    coverage = payload.get("coverage") if isinstance(payload.get("coverage"), dict) else {}
    features = payload.get("global_features") if isinstance(payload.get("global_features"), dict) else {}
    assessment = assess_central_bank_cross_source_context(payload, now_utc=now)
    verified = bool(fresh and assessment.get("ready", False))
    notes = [str(reason) for reason in assessment.get("reasons", [])]
    if not fresh and "stale_artifact" not in notes:
        notes.append("stale_artifact")
    if int(coverage.get("soft_conflict_count", 0) or 0) > 0:
        notes.append(f"soft_source_conflicts={int(coverage.get('soft_conflict_count', 0) or 0)}")
    return _row(
        source_id="central_bank_cross_source_context",
        title="Central Bank Point-In-Time Cross-Source Router",
        category="macro_data",
        verification_status=STATUS_SINGLE_VERIFIED if verified else STATUS_SINGLE_UNVERIFIED,
        verification_mode="point_in_time_lineage_and_conflict_contract",
        artifact_path=path,
        artifact_timestamp=ts,
        age_hours=_age_hours(ts, now),
        fresh=fresh,
        ok=verified,
        notes=notes,
        evidence={
            "synchronized_ready_bank_count": coverage.get("synchronized_ready_bank_count", 0),
            "distinct_cross_source_link_count": coverage.get("distinct_cross_source_link_count", 0),
            "banks_without_distinct_cross_source": coverage.get("banks_without_distinct_cross_source", []),
            "synchronized_bank_coverage_ratio": coverage.get("synchronized_bank_coverage_ratio", 0.0),
            "fx_join_coverage_ratio": coverage.get("fx_join_coverage_ratio", 0.0),
            "macro_join_coverage_ratio": coverage.get("macro_join_coverage_ratio", 0.0),
            "liquidity_join_coverage_ratio": coverage.get("liquidity_join_coverage_ratio", 0.0),
            "lineage_coverage_ratio": coverage.get("lineage_coverage_ratio", 0.0),
            "hard_conflict_count": coverage.get("hard_conflict_count", 0),
            "soft_conflict_count": coverage.get("soft_conflict_count", 0),
            "future_observations_excluded": coverage.get("future_observations_excluded", {}),
            "feature_count": len(features),
            "required_feature_count": len(CENTRAL_BANK_CROSS_SOURCE_FEATURE_KEYS),
            "consumer_contract": assessment,
            "routing": payload.get("routing", {}),
            "methodology": payload.get("methodology", {}),
        },
    )


def _decision_context_mesh_row(project_root: Path, now: datetime) -> dict[str, Any]:
    path = project_root / "exports" / "external_context" / "decision_context_mesh_latest.json"
    payload = _read_json(path)
    ts = _parse_ts(payload.get("timestamp_utc"))
    fresh = _is_fresh(ts, now, 24.0)
    assessment = assess_decision_context_mesh(payload, now_utc=now)
    coverage = payload.get("coverage") if isinstance(payload.get("coverage"), dict) else {}
    grade_summary = payload.get("grade_summary") if isinstance(payload.get("grade_summary"), dict) else {}
    features = ((payload.get("derived") or {}).get("global_features") or {}) if isinstance(payload.get("derived"), dict) else {}
    sources = payload.get("sources") if isinstance(payload.get("sources"), dict) else {}
    healthy_sources = sum(1 for row in sources.values() if isinstance(row, dict) and row.get("ok") is True)
    source_count = len(sources)
    distinct_families = {
        str(row.get("source_family") or "")
        for row in sources.values()
        if isinstance(row, dict) and row.get("ok") is True and str(row.get("source_family") or "")
    }
    verified = bool(fresh and assessment.get("ready", False) and healthy_sources >= max(source_count - 1, 1))
    notes = [str(reason) for reason in assessment.get("reasons", [])]
    if not fresh and "stale_artifact" not in notes:
        notes.append("stale_artifact")
    return _row(
        source_id="decision_context_mesh",
        title="Twelve-Plane Macro And Micro Decision Context Mesh",
        category="cross_source_decision_context",
        verification_status=(
            STATUS_CROSS_VERIFIED if verified and len(distinct_families) >= 2 else STATUS_SINGLE_UNVERIFIED
        ),
        verification_mode="point_in_time_multi_source_lineage_and_routing_contract",
        artifact_path=path,
        artifact_timestamp=ts,
        age_hours=_age_hours(ts, now),
        fresh=fresh,
        ok=verified,
        notes=notes,
        evidence={
            "macro_percentage": grade_summary.get("macro_percentage", 0.0),
            "macro_grade": grade_summary.get("macro_grade", "F"),
            "micro_percentage": grade_summary.get("micro_percentage", 0.0),
            "micro_grade": grade_summary.get("micro_grade", "F"),
            "combined_percentage": grade_summary.get("combined_percentage", 0.0),
            "plane_count": coverage.get("observed_plane_count", 0),
            "ready_plane_count": coverage.get("ready_plane_count", 0),
            "signal_coverage_ratio": coverage.get("signal_coverage_ratio", 0.0),
            "healthy_source_count": healthy_sources,
            "ok_sources": healthy_sources,
            "total_sources": source_count,
            "distinct_source_family_count": len(distinct_families),
            "cross_profile_ok": bool(verified and len(distinct_families) >= 2),
            "feature_count": len(features),
            "required_feature_count": len(DECISION_CONTEXT_MESH_FEATURE_KEYS),
            "future_observations_excluded": coverage.get("future_observations_excluded", {}),
            "consumer_contract": assessment,
            "authority_contract": payload.get("contract", {}),
            "methodology": payload.get("methodology", {}),
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


def _schwab_symbol_news_row(health_dir: Path, now: datetime) -> dict[str, Any]:
    path = health_dir / "schwab_symbol_news_latest.json"
    payload = _read_json(path)
    ts = _parse_ts(payload.get("timestamp_utc"))
    fresh = _is_fresh(ts, now, 12.0)
    overall_status = str(payload.get("overall_status") or "").strip()
    auth_required = bool(payload.get("auth_required", True))
    auth_ok = bool(payload.get("auth_ok", False))
    auth_ready = bool(auth_ok or not auth_required)
    requested = int(payload.get("requested_symbol_count", 0) or 0)
    attempted = int(payload.get("attempted_symbol_count", 0) or 0)
    with_news = int(payload.get("symbols_with_news", 0) or 0)
    total_items = int(payload.get("total_news_items", 0) or 0)
    coverage_ratio = float(payload.get("coverage_ratio", 0.0) or 0.0)
    method_counts = payload.get("method_counts") if isinstance(payload.get("method_counts"), dict) else {}
    fallback_active = bool(payload.get("fallback_active", False))
    fallback_source_contract = (
        payload.get("fallback_source_contract")
        if isinstance(payload.get("fallback_source_contract"), dict)
        else {}
    )
    fallback_source_fresh = bool(not fallback_active or fallback_source_contract.get("fresh", False))
    no_endpoint = str(overall_status) == "degraded_no_broker_news_endpoint" or (
        attempted > 0 and int(method_counts.get("none", 0) or 0) >= attempted
    )
    notes: list[str] = []
    if not fresh:
        notes.append("stale_artifact")
    if not auth_ready:
        notes.append("schwab_auth_blocked")
    if no_endpoint and not fallback_active:
        notes.append("no_callable_broker_news_endpoint")
    if requested > 0 and attempted < requested:
        notes.append(f"partial_symbol_attempts={attempted}/{requested}")
    if total_items <= 0 and auth_ok and not no_endpoint:
        notes.append("no_symbol_news_items")
    if fallback_active and not fallback_source_fresh:
        notes.append("stale_public_schwab_fallback_source")

    ok = bool(
        payload.get("ok", False)
        and fresh
        and auth_ready
        and attempted > 0
        and (not no_endpoint or fallback_active)
        and fallback_source_fresh
    )
    status = STATUS_SINGLE_VERIFIED if ok else STATUS_SINGLE_UNVERIFIED
    return _row(
        source_id="schwab_symbol_news",
        title="Schwab Symbol News",
        category="broker_native_news",
        verification_status=status,
        verification_mode="broker_native_symbol_news_health",
        artifact_path=path,
        artifact_timestamp=ts,
        age_hours=_age_hours(ts, now),
        fresh=fresh,
        ok=ok,
        notes=notes,
        evidence={
            "overall_status": overall_status,
            "auth_ok": auth_ok,
            "auth_required": auth_required,
            "auth_ready": auth_ready,
            "requested_symbols": requested,
            "attempted_symbols": attempted,
            "symbols_with_news": with_news,
            "total_news_items": total_items,
            "coverage_ratio": coverage_ratio,
            "method_counts": method_counts,
            "source_counts": payload.get("source_counts") if isinstance(payload.get("source_counts"), dict) else {},
            "broker_native_news_endpoint_available": bool(payload.get("broker_native_news_endpoint_available", False)),
            "fallback_active": fallback_active,
            "fallback_mode": str(payload.get("fallback_mode") or ""),
            "fallback_symbol_count": int(payload.get("fallback_symbol_count", 0) or 0),
            "fallback_source_contract": fallback_source_contract,
        },
    )


def _ticker_news_context_row(health_dir: Path, now: datetime) -> dict[str, Any]:
    path = health_dir / "ticker_news_context_latest.json"
    payload = _read_json(path)
    ts = _parse_ts(payload.get("timestamp_utc"))
    fresh = _is_fresh(ts, now, 12.0)
    requested = int(payload.get("requested_symbol_count", 0) or 0)
    with_news = int(payload.get("symbols_with_news", 0) or 0)
    total_items = int(payload.get("total_news_items", 0) or 0)
    ok_sources = int(payload.get("ok_source_count", 0) or 0)
    total_sources = int(payload.get("source_count", 0) or 0)
    coverage_ratio = float(payload.get("coverage_ratio", 0.0) or 0.0)
    notes: list[str] = []
    if not fresh:
        notes.append("stale_artifact")
    if ok_sources <= 0:
        notes.append("no_news_sources_ready")
    elif total_sources > 0 and ok_sources < max(1, min(total_sources, 2)):
        notes.append(f"partial_sources={ok_sources}/{total_sources}")
    if requested > 0 and with_news <= 0:
        notes.append("no_symbol_news_coverage")
    elif requested > 0 and coverage_ratio < 0.02:
        notes.append(f"low_symbol_news_coverage={coverage_ratio:.3f}")
    ok = bool(payload.get("ok", False)) and fresh and ok_sources > 0 and total_items > 0
    status = STATUS_SINGLE_VERIFIED if ok else STATUS_SINGLE_UNVERIFIED
    return _row(
        source_id="ticker_news_context",
        title="Ticker News Context Mesh",
        category="news_data",
        verification_status=status,
        verification_mode="multi_source_ticker_news_health",
        artifact_path=path,
        artifact_timestamp=ts,
        age_hours=_age_hours(ts, now),
        fresh=fresh,
        ok=ok,
        notes=notes,
        evidence={
            "requested_symbols": requested,
            "symbols_with_news": with_news,
            "coverage_ratio": coverage_ratio,
            "total_news_items": total_items,
            "ok_sources": ok_sources,
            "total_sources": total_sources,
            "source_counts": payload.get("source_counts") if isinstance(payload.get("source_counts"), dict) else {},
            "sources": {
                key: bool(value.get("ok", False))
                for key, value in (payload.get("sources") or {}).items()
                if isinstance(value, dict)
            },
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
    local_micro_ok = bool((sources.get("local_micro") or {}).get("ok", False)) if isinstance(sources, dict) else False
    finra_ok = bool((sources.get("finra_short_volume") or {}).get("ok", False)) if isinstance(sources, dict) else False
    nasdaq_halts_ok = bool((sources.get("nasdaq_trade_halts") or {}).get("ok", False)) if isinstance(sources, dict) else False
    treasury_ok = bool((sources.get("treasury_auctions") or {}).get("ok", False)) if isinstance(sources, dict) else False
    critical_sources = {
        "local_micro": local_micro_ok,
        "external_micro_reference": bool(finra_ok or nasdaq_halts_ok),
        "treasury_auction_context": treasury_ok,
        "finra_short_volume": finra_ok,
        "nasdaq_trade_halts": nasdaq_halts_ok,
    }
    if not fresh:
        notes.append("stale_artifact")
    min_ok_sources = _minimum_ok_sources(total_count, floor=3, tolerate_failures=1, min_ratio=0.75)
    holiday_pause_observed = _market_holiday_pause_observed(health_dir, now)
    market_closed_local_micro_fallback = bool(
        not local_micro_ok
        and (_market_closed_for_local_micro(now) or holiday_pause_observed)
        and bool(finra_ok or nasdaq_halts_ok)
        and treasury_ok
        and fresh
    )
    finra_symbol_count = int(((sources.get("finra_short_volume") or {}).get("symbol_count", 0)) or 0)
    external_micro_reference_verified_fallback = bool(
        not local_micro_ok
        and finra_ok
        and treasury_ok
        and finra_symbol_count >= 50
        and fresh
    )
    effective_ok_count = ok_count + (
        1 if (market_closed_local_micro_fallback or external_micro_reference_verified_fallback) else 0
    )
    if total_count > 0 and ok_count < total_count:
        notes.append(f"partial_sources={ok_count}/{total_count}")
    if market_closed_local_micro_fallback:
        notes.append("local_micro_absent_market_closed")
    elif external_micro_reference_verified_fallback:
        notes.append("local_micro_absent_external_reference_verified")
    if auxiliary_degraded:
        notes.append(f"auxiliary_source_degraded={','.join(auxiliary_degraded)}")
    status = (
        STATUS_SINGLE_VERIFIED
        if bool(payload.get("ok", False))
        and total_count > 0
        and effective_ok_count >= min_ok_sources
        and (
            local_micro_ok
            or market_closed_local_micro_fallback
            or external_micro_reference_verified_fallback
        )
        and bool(finra_ok or nasdaq_halts_ok)
        and treasury_ok
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
            "effective_ok_sources": effective_ok_count,
            "total_sources": total_count,
            "min_ok_sources_required": min_ok_sources,
            "critical_sources": critical_sources,
            "market_closed_local_micro_fallback": market_closed_local_micro_fallback,
            "external_micro_reference_verified_fallback": external_micro_reference_verified_fallback,
            "holiday_pause_observed": holiday_pause_observed,
            "local_micro_symbol_count": int(((sources.get("local_micro") or {}).get("symbol_count", 0)) or 0),
            "finra_symbol_count": finra_symbol_count,
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


def _public_policy_context_row(health_dir: Path, now: datetime) -> dict[str, Any]:
    path = health_dir / "public_policy_context_sync_latest.json"
    payload = _read_json(path)
    ts = _parse_ts(payload.get("timestamp_utc"))
    fresh = _is_fresh(ts, now, 72.0)
    sources = payload.get("sources") if isinstance(payload.get("sources"), dict) else {}
    ok_count, total_count = _ok_count(sources)
    features = payload.get("features") if isinstance(payload.get("features"), dict) else {}
    treasury_debt = sources.get("treasury_debt_to_penny") if isinstance(sources.get("treasury_debt_to_penny"), dict) else {}
    treasury_rates = (
        sources.get("treasury_avg_interest_rates")
        if isinstance(sources.get("treasury_avg_interest_rates"), dict)
        else {}
    )
    world_bank = sources.get("world_bank_indicators") if isinstance(sources.get("world_bank_indicators"), dict) else {}
    treasury_debt_ok = bool(treasury_debt.get("ok", False))
    treasury_rates_ok = bool(treasury_rates.get("ok", False))
    world_bank_ok = bool(world_bank.get("ok", False))
    world_bank_indicator_count = int(world_bank.get("indicator_count", 0) or 0)
    world_bank_success_count = int(world_bank.get("indicator_success_count", 0) or 0)
    world_bank_value_count = int(world_bank.get("value_count", 0) or 0)
    min_world_bank_success = max(1, min(world_bank_indicator_count, 4)) if world_bank_indicator_count > 0 else 0
    world_bank_partial_verified = bool(
        not world_bank_ok
        and world_bank_indicator_count > 0
        and world_bank_success_count >= min_world_bank_success
        and world_bank_value_count >= 8
    )
    world_bank_effective_ok = bool(world_bank_ok or world_bank_partial_verified)
    effective_ok_count = ok_count + (1 if world_bank_partial_verified else 0)
    notes: list[str] = []
    if ok_count < total_count:
        notes.append(f"partial_sources={ok_count}/{total_count}")
    if not treasury_debt_ok:
        notes.append("treasury_debt_to_penny_unavailable")
    if not treasury_rates_ok:
        notes.append("treasury_avg_interest_rates_unavailable")
    if not world_bank_ok and world_bank_partial_verified:
        notes.append(f"world_bank_indicators_partial={world_bank_success_count}/{world_bank_indicator_count}")
    elif not world_bank_ok:
        notes.append("world_bank_indicators_unavailable")
    if world_bank_value_count < 8:
        notes.append("world_bank_value_coverage_low")
    if not features:
        notes.append("derived_features_missing")
    if not fresh:
        notes.append("stale_artifact")
    min_ok_sources = _minimum_ok_sources(total_count, floor=2, tolerate_failures=1, min_ratio=0.67)
    status = (
        STATUS_SINGLE_VERIFIED
        if (bool(payload.get("ok", False)) or (treasury_debt_ok and treasury_rates_ok and world_bank_effective_ok and bool(features)))
        and fresh
        and total_count >= 2
        and effective_ok_count >= min_ok_sources
        and treasury_debt_ok
        and treasury_rates_ok
        and world_bank_effective_ok
        else STATUS_SINGLE_UNVERIFIED
    )
    effective_ok = status == STATUS_SINGLE_VERIFIED
    return _row(
        source_id="public_policy_context",
        title="Public Policy / Sovereign Liquidity Context",
        category="macro_liquidity_data",
        verification_status=status,
        verification_mode="official_free_public_api_health",
        artifact_path=path,
        artifact_timestamp=ts,
        age_hours=_age_hours(ts, now),
        fresh=fresh,
        ok=effective_ok,
        notes=notes,
        evidence={
            "context_profile": str(payload.get("context_profile") or ""),
            "ok_sources": ok_count,
            "effective_ok_sources": effective_ok_count,
            "total_sources": total_count,
            "min_ok_sources_required": min_ok_sources,
            "countries": payload.get("countries") if isinstance(payload.get("countries"), list) else [],
            "treasury_debt_record_date": str(treasury_debt.get("record_date") or ""),
            "treasury_avg_interest_record_date": str(treasury_rates.get("record_date") or ""),
            "world_bank_lastupdated": str(world_bank.get("lastupdated") or ""),
            "world_bank_indicator_success_count": world_bank_success_count,
            "world_bank_indicator_count": world_bank_indicator_count,
            "world_bank_value_count": world_bank_value_count,
            "world_bank_partial_verified": world_bank_partial_verified,
            "us_public_debt_to_worldbank_gdp_proxy": features.get("us_public_debt_to_worldbank_gdp_proxy"),
            "treasury_avg_interest_rate_pct": features.get("treasury_avg_interest_rate_pct"),
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
        _free_equity_reference_row(health_dir, now),
        _fx_market_row(health_dir, now),
        _external_feeds_row(project_root, now),
        _official_macro_row(health_dir, now),
        _central_bank_liquidity_row(project_root, now),
        _global_central_bank_row(project_root, now),
        _central_bank_cross_source_row(project_root, now),
        _decision_context_mesh_row(project_root, now),
        _schwab_education_row(health_dir, now),
        _schwab_symbol_news_row(health_dir, now),
        _ticker_news_context_row(health_dir, now),
        _market_micro_row(health_dir, now),
        _sec_edgar_row(health_dir, now),
        _extended_quant_row(health_dir, now),
        _public_policy_context_row(health_dir, now),
        _fed_2026_stress_scenario_row(project_root, now),
    ]

    counts = {
        STATUS_CROSS_VERIFIED: sum(1 for row in rows if row["verification_status"] == STATUS_CROSS_VERIFIED),
        STATUS_SINGLE_VERIFIED: sum(1 for row in rows if row["verification_status"] == STATUS_SINGLE_VERIFIED),
        STATUS_SINGLE_UNVERIFIED: sum(1 for row in rows if row["verification_status"] == STATUS_SINGLE_UNVERIFIED),
    }
    confidence_scores = [float(row.get("source_confidence_score", 0.0) or 0.0) for row in rows]
    low_confidence_sources = [
        str(row.get("source_id") or "")
        for row in rows
        if float(row.get("source_confidence_score", 0.0) or 0.0) < 0.70
    ]
    unverified = [row["source_id"] for row in rows if row["verification_status"] == STATUS_SINGLE_UNVERIFIED]
    warnings = [row["source_id"] for row in rows if _row_has_actionable_notes(row)]
    stale = [row["source_id"] for row in rows if not bool(row.get("fresh", False))]
    degraded = _ordered_unique(unverified + warnings)
    all_verified = counts[STATUS_SINGLE_UNVERIFIED] == 0
    all_cross_verified = counts[STATUS_CROSS_VERIFIED] == len(rows)
    decision_critical_rows = [row for row in rows if row.get("criticality") == "decision_critical"]
    decision_context_rows = [row for row in rows if row.get("criticality") == "decision_context"]
    optional_enrichment_rows = [row for row in rows if row.get("criticality") == "optional_enrichment"]

    def _runtime_source_ready(row: dict[str, Any]) -> bool:
        return bool(
            row.get("verification_status") != STATUS_SINGLE_UNVERIFIED
            and row.get("fresh", False)
            and row.get("ok", False)
            and _safe_float(row.get("source_confidence_score"), 0.0) >= 0.70
        )

    decision_critical_blockers = [
        str(row.get("source_id") or "") for row in decision_critical_rows if not _runtime_source_ready(row)
    ]
    decision_context_debt = [
        str(row.get("source_id") or "") for row in decision_context_rows if not _runtime_source_ready(row)
    ]
    optional_enrichment_debt = [
        str(row.get("source_id") or "") for row in optional_enrichment_rows if not _runtime_source_ready(row)
    ]
    decision_critical_sources_ready = bool(decision_critical_rows and not decision_critical_blockers)
    overall_status = "ready" if all_verified else "degraded"
    row_scores = [
        100.0
        * (
            (0.35 if row.get("verification_status") != STATUS_SINGLE_UNVERIFIED else 0.0)
            + (0.20 if row.get("fresh", False) else 0.0)
            + (0.20 if row.get("ok", False) else 0.0)
            + 0.25 * float(row.get("source_confidence_score", 0.0) or 0.0)
        )
        for row in rows
    ]
    evidence_score = round(sum(row_scores) / max(len(row_scores), 1), 3)
    evidence_complete = bool(
        all_verified
        and not stale
        and all(bool(row.get("ok", False)) for row in rows)
        and min(confidence_scores or [0.0]) >= 0.70
        and (sum(confidence_scores) / max(len(confidence_scores), 1)) >= 0.90
    )
    evidence_grade = _grade(evidence_score, complete=evidence_complete)
    control_checks = {
        "point_in_time_freshness_slos": True,
        "per_source_confidence_components": True,
        "decision_criticality_classification": True,
        "decision_critical_runtime_contract": True,
        "bounded_adaptive_refresh_batches": True,
        "persistent_exponential_retry_state": True,
        "bounded_quarantine_with_starvation_override": True,
        "atomic_report_and_state_replacement": True,
        "downstream_contract_reconciliation": True,
    }
    refresh_commands: list[list[str]] = []
    for source_id in degraded:
        command = _refresh_command_for_source(project_root, source_id)
        if command not in refresh_commands:
            refresh_commands.append(command)
    if refresh_commands and [str(project_root / "scripts" / "ops" / "opsctl.sh"), "source-verification", "--json"] not in refresh_commands:
        refresh_commands.append([str(project_root / "scripts" / "ops" / "opsctl.sh"), "source-verification", "--json"])
    return {
        "timestamp_utc": now.isoformat(),
        "schema_version": 2,
        "ok": all_verified,
        "overall_status": overall_status,
        "grade": evidence_grade,
        "source_evidence_grade": evidence_grade,
        "source_evidence_score": evidence_score,
        "source_evidence_a_plus_earned": evidence_complete,
        "source_control_grade": "A+" if all(control_checks.values()) else "F",
        "source_control_score": 100.0 * sum(1 for value in control_checks.values() if value) / len(control_checks),
        "overall": {
            "all_cross_verified": all_cross_verified,
            "all_verified": all_verified,
            "counts": counts,
            "total_sources": len(rows),
            "unverified_sources": unverified,
            "sources_with_notes": warnings,
            "stale_sources": stale,
            "mean_source_confidence_score": round(sum(confidence_scores) / max(len(confidence_scores), 1), 6),
            "min_source_confidence_score": round(min(confidence_scores or [0.0]), 6),
            "low_confidence_sources": low_confidence_sources,
        },
        "source_runtime_contract": {
            "decision_critical_sources_ready": decision_critical_sources_ready,
            "decision_critical_source_count": len(decision_critical_rows),
            "decision_critical_blocker_count": len(decision_critical_blockers),
            "decision_critical_blockers": decision_critical_blockers,
            "decision_context_debt": decision_context_debt,
            "optional_enrichment_debt": optional_enrichment_debt,
            "minimum_confidence_score": 0.70,
            "policy": "paper_runtime_requires_fresh_healthy_verified_decision_critical_sources; context_and_optional_debt_remains_visible_without_execution_authority",
        },
        "source_confidence_summary": {
            "mean_score": round(sum(confidence_scores) / max(len(confidence_scores), 1), 6),
            "min_score": round(min(confidence_scores or [0.0]), 6),
            "low_confidence_source_count": len(low_confidence_sources),
            "low_confidence_sources": low_confidence_sources,
            "policy": "training_and_paper_truth_downweight_contexts_when_source_confidence_is_thin",
        },
        "unverified_sources": unverified,
        "stale_artifacts": stale,
        "degraded_artifacts": degraded,
        "recommended_refresh_commands": refresh_commands,
        "recommended_actions": [
            "refresh degraded source artifacts with the recommended commands, then rerun source-verification",
            "keep required market context lanes fresh before using market-move explanations for confidence claims",
        ]
        if degraded
        else ["source verification is clean; keep scheduled collectors current"],
        "autorefresh_contract": {
            "enabled": True,
            "apply_command": [str(project_root / "scripts" / "ops" / "opsctl.sh"), "source-verification-refresh", "--apply", "--json"],
            "preview_command": [str(project_root / "scripts" / "ops" / "opsctl.sh"), "source-verification-refresh", "--json"],
            "policy": "refresh_only_degraded_or_stale_source_artifacts_then_rerun_source_verification",
            "persistent_retry_state": "governance/runtime/source_verification_retry_state.json",
            "exponential_backoff": True,
            "bounded_quarantine": True,
            "starvation_protection": True,
            "atomic_report_replacement": True,
        },
        "source_reliability_contract": {
            "control_checks": control_checks,
            "control_grade_measures_hardening_not_current_provider_health": True,
            "evidence_A_plus_requires_every_source_verified_fresh_healthy_and_confident": True,
            "provider_failure_never_inherits_an_A_plus_label": True,
            "optional_enrichment_failure_cannot_authorize_a_trade": True,
        },
        "sources": rows,
    }


def _render_markdown(payload: dict[str, Any]) -> str:
    overall = payload.get("overall") if isinstance(payload.get("overall"), dict) else {}
    counts = overall.get("counts") if isinstance(overall.get("counts"), dict) else {}
    lines = [
        f"# Source Verification Report ({payload.get('timestamp_utc', '')})",
        f"- all_verified: {bool(overall.get('all_verified', False))}",
        f"- decision_critical_sources_ready: {bool(((payload.get('source_runtime_contract') or {}).get('decision_critical_sources_ready', False)))}",
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

    _atomic_write_text(out_json, json_text)
    _atomic_write_text(out_md, rendered_md)
    _atomic_write_text(latest_json, json_text)
    _atomic_write_text(latest_md, rendered_md)

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
