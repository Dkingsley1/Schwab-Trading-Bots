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
        total += 1
        if bool(value.get("ok", False)):
            ok += 1
    return ok, total


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


def _tastytrade_row(health_dir: Path, now: datetime) -> dict[str, Any]:
    path = health_dir / "tastytrade_context_sync_latest.json"
    payload = _read_json(path)
    ts = _parse_ts(payload.get("timestamp_utc"))
    fresh = _is_fresh(ts, now, 12.0)
    notes: list[str] = []
    sandbox = bool(payload.get("sandbox", False))
    if int(payload.get("alignment_reference_only", 0) or 0) > 0 and not sandbox:
        notes.append("reference_only_alignment")
    if int(payload.get("symbols_with_metrics", 0) or 0) == 0 and not sandbox:
        notes.append("sandbox_metrics_unavailable")
    if not fresh:
        notes.append("stale_artifact")
    status = (
        STATUS_CROSS_VERIFIED
        if bool(payload.get("ok", False))
        and bool(payload.get("alignment_ok", False))
        and (int(payload.get("alignment_compared", 0) or 0) > 0 or int(payload.get("alignment_reference_only", 0) or 0) > 0)
        and fresh
        else STATUS_SINGLE_UNVERIFIED
    )
    return _row(
        source_id="tastytrade_options_context",
        title="Tastytrade Options Context",
        category="derivatives_data",
        verification_status=status,
        verification_mode="schwab_alignment_guard",
        artifact_path=path,
        artifact_timestamp=ts,
        age_hours=_age_hours(ts, now),
        fresh=fresh,
        ok=bool(payload.get("ok", False)),
        notes=notes,
        evidence={
            "sandbox": sandbox,
            "symbols_requested": int(payload.get("symbols_requested", 0) or 0),
            "symbols_with_chain": int(payload.get("symbols_with_chain", 0) or 0),
            "alignment_compared": int(payload.get("alignment_compared", 0) or 0),
            "alignment_reference_only": int(payload.get("alignment_reference_only", 0) or 0),
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
    fresh = _is_fresh(ts, now, 12.0)
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
        if bool(payload.get("ok", False)) and compared_assets > 0 and ok_sources >= 5 and fresh
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
    if not fresh:
        notes.append("stale_artifact")
    status = STATUS_SINGLE_VERIFIED if bool(payload.get("ok", False)) and ok_count == total_count and total_count > 0 and fresh else STATUS_SINGLE_UNVERIFIED
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
            "sources": {key: bool(value.get("ok", False)) for key, value in sources.items() if isinstance(value, dict)},
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
    if not fresh:
        notes.append("stale_artifact")
    status = STATUS_SINGLE_VERIFIED if bool(payload.get("ok", False)) and ok_count == total_count and total_count > 0 and fresh else STATUS_SINGLE_UNVERIFIED
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


def build_source_verification_payload(project_root: Path = PROJECT_ROOT) -> dict[str, Any]:
    now = datetime.now(timezone.utc)
    health_dir = project_root / "governance" / "health"
    rows = [
        _market_quote_row(health_dir, now),
        _tastytrade_row(health_dir, now),
        _macro_crosscheck_row(health_dir, now),
        _crypto_market_row(health_dir, now),
        _fx_market_row(health_dir, now),
        _external_feeds_row(project_root, now),
        _official_macro_row(health_dir, now),
        _market_micro_row(health_dir, now),
        _sec_edgar_row(health_dir, now),
        _extended_quant_row(health_dir, now),
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
