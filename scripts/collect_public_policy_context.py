#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


USER_AGENT_DEFAULT = "Daniel Kingsley dan_kingsley@aol.com"
TREASURY_DEBT_TO_PENNY_URL = (
    "https://api.fiscaldata.treasury.gov/services/api/fiscal_service/v2/accounting/od/"
    "debt_to_penny?sort=-record_date&page%5Bsize%5D=40"
)
TREASURY_AVG_INTEREST_URL = (
    "https://api.fiscaldata.treasury.gov/services/api/fiscal_service/v2/accounting/od/"
    "avg_interest_rates?sort=-record_date&page%5Bsize%5D=30"
)
WORLD_BANK_API_ROOT = "https://api.worldbank.org/v2"
DEFAULT_COUNTRIES = ["USA", "CHN", "JPN", "DEU", "GBR", "IND", "BRA", "CAN", "MEX", "KOR"]
WORLD_BANK_INDICATORS = {
    "NY.GDP.MKTP.CD": "gdp_current_usd",
    "FP.CPI.TOTL.ZG": "inflation_cpi_annual_pct",
    "BN.CAB.XOKA.GD.ZS": "current_account_pct_gdp",
    "GC.DOD.TOTL.GD.ZS": "central_government_debt_pct_gdp",
    "FR.INR.RINR": "real_interest_rate_pct",
}
SOURCE_CONTRACTS = {
    "treasury_debt_to_penny": {
        "publisher": "U.S. Treasury FiscalData",
        "api_key_required": False,
        "update_cadence": "daily_business_day",
        "source_confidence_norm": 0.99,
        "schema_confidence_norm": 0.98,
        "contract_participates": True,
    },
    "treasury_avg_interest_rates": {
        "publisher": "U.S. Treasury FiscalData",
        "api_key_required": False,
        "update_cadence": "monthly",
        "source_confidence_norm": 0.99,
        "schema_confidence_norm": 0.97,
        "contract_participates": True,
    },
    "world_bank_indicators": {
        "publisher": "World Bank Indicators API",
        "api_key_required": False,
        "update_cadence": "annual_or_periodic",
        "source_confidence_norm": 0.96,
        "schema_confidence_norm": 0.95,
        "contract_participates": True,
    },
}


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), ensure_ascii=True, indent=2) + "\n", encoding="utf-8")


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        out = float(str(value).replace(",", "").strip())
    except Exception:
        return None
    if not math.isfinite(out):
        return None
    return out


def _clamp01(value: float) -> float:
    return max(0.0, min(float(value), 1.0))


def _mean(values: Iterable[float | None]) -> float | None:
    clean = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    if not clean:
        return None
    return sum(clean) / float(len(clean))


def _http_json(url: str, *, user_agent: str, timeout: float) -> Any:
    req = Request(
        url=url,
        headers={
            "User-Agent": user_agent,
            "Accept": "application/json,*/*",
        },
    )
    with urlopen(req, timeout=max(float(timeout), 1.0)) as resp:
        return json.loads(resp.read().decode("utf-8", "replace"))


def _safe_http_json(url: str, *, user_agent: str, timeout: float) -> tuple[Any | None, str | None]:
    try:
        return _http_json(url, user_agent=user_agent, timeout=timeout), None
    except (HTTPError, URLError, TimeoutError, OSError, ValueError, json.JSONDecodeError) as exc:
        return None, str(exc)


def _contract(name: str) -> dict[str, Any]:
    return dict(SOURCE_CONTRACTS.get(name, {}))


def _source_status(name: str, *, ok: bool, url: str, error: str | None = None, **extra: Any) -> dict[str, Any]:
    status = {
        **_contract(name),
        "ok": bool(ok),
        "url": url,
    }
    if error:
        status["error"] = str(error)
    status.update(extra)
    return status


def _iter_treasury_rate_cache_paths() -> Iterable[Path]:
    yield PROJECT_ROOT / "governance" / "health" / "source_verification_latest.json"
    yield PROJECT_ROOT / "governance" / "health" / "source_verification_autorefresh_latest.json"
    yield PROJECT_ROOT / "governance" / "health" / "training_labeling_intelligence_latest.json"
    reports = PROJECT_ROOT / "exports" / "reports"
    if reports.exists():
        yield from sorted(reports.glob("source_verification_*.json"), key=lambda item: item.stat().st_mtime, reverse=True)


def _find_public_policy_evidence(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    if isinstance(payload.get("source_verification"), Mapping):
        nested = _find_public_policy_evidence(payload["source_verification"])
        if nested:
            return nested
    if isinstance(payload.get("free_label_source_enrichment"), Mapping):
        nested = _find_public_policy_evidence(payload["free_label_source_enrichment"])
        if nested:
            return nested
    sources = payload.get("sources")
    if isinstance(sources, list):
        for source in sources:
            if not isinstance(source, Mapping):
                continue
            if source.get("source_id") == "public_policy_context":
                evidence = source.get("evidence")
                return evidence if isinstance(evidence, Mapping) else source
    if isinstance(sources, Mapping) and isinstance(sources.get("treasury_avg_interest_rates"), Mapping):
        return payload
    return {}


def _cached_treasury_avg_interest() -> dict[str, Any] | None:
    for path in _iter_treasury_rate_cache_paths():
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        if not isinstance(payload, Mapping):
            continue
        evidence = _find_public_policy_evidence(payload)
        if not evidence:
            continue
        source = evidence.get("treasury_avg_interest_rates") if isinstance(evidence.get("treasury_avg_interest_rates"), Mapping) else {}
        features = evidence.get("features") if isinstance(evidence.get("features"), Mapping) else {}
        rate = _safe_float(source.get("avg_interest_rate_pct"))
        if rate is None:
            rate = _safe_float(features.get("treasury_avg_interest_rate_pct"))
        if rate is None:
            rate = _safe_float(evidence.get("treasury_avg_interest_rate_pct"))
        if rate is None:
            continue
        return _source_status(
            "treasury_avg_interest_rates",
            ok=True,
            url=TREASURY_AVG_INTEREST_URL,
            record_date=str(
                source.get("record_date")
                or evidence.get("treasury_avg_interest_record_date")
                or "cached_monthly_reference"
            ),
            row_count=int(source.get("row_count", 0) or 0),
            avg_interest_rate_pct=rate,
            rates_by_security=source.get("rates_by_security") if isinstance(source.get("rates_by_security"), dict) else {},
            cached_fallback=True,
            cache_path=str(path),
            cache_reason="live_treasury_avg_interest_rates_unavailable",
        )
    return None


def _fetch_treasury_debt(*, user_agent: str, timeout: float) -> dict[str, Any]:
    payload, error = _safe_http_json(TREASURY_DEBT_TO_PENNY_URL, user_agent=user_agent, timeout=timeout)
    rows = payload.get("data") if isinstance(payload, dict) and isinstance(payload.get("data"), list) else []
    latest = rows[0] if rows and isinstance(rows[0], dict) else {}
    prior = rows[1] if len(rows) > 1 and isinstance(rows[1], dict) else {}
    oldest = rows[-1] if rows and isinstance(rows[-1], dict) else {}
    total = _safe_float(latest.get("tot_pub_debt_out_amt"))
    held_public = _safe_float(latest.get("debt_held_public_amt"))
    intragov = _safe_float(latest.get("intragov_hold_amt"))
    prior_total = _safe_float(prior.get("tot_pub_debt_out_amt"))
    oldest_total = _safe_float(oldest.get("tot_pub_debt_out_amt"))
    ok = bool(latest) and total is not None and held_public is not None and intragov is not None
    return _source_status(
        "treasury_debt_to_penny",
        ok=ok,
        url=TREASURY_DEBT_TO_PENNY_URL,
        error=error,
        record_date=str(latest.get("record_date") or ""),
        row_count=len(rows),
        total_public_debt_usd=total,
        debt_held_public_usd=held_public,
        intragov_holdings_usd=intragov,
        daily_change_usd=(total - prior_total) if total is not None and prior_total is not None else None,
        sample_window_change_usd=(total - oldest_total) if total is not None and oldest_total is not None else None,
        sample_window_days=max(len(rows) - 1, 0),
    )


def _fetch_treasury_avg_interest(*, user_agent: str, timeout: float) -> dict[str, Any]:
    payload, error = _safe_http_json(TREASURY_AVG_INTEREST_URL, user_agent=user_agent, timeout=timeout)
    rows = payload.get("data") if isinstance(payload, dict) and isinstance(payload.get("data"), list) else []
    latest_date = str((rows[0] or {}).get("record_date") or "") if rows and isinstance(rows[0], dict) else ""
    latest_rows = [row for row in rows if isinstance(row, dict) and str(row.get("record_date") or "") == latest_date]
    rates_by_security: dict[str, float] = {}
    for row in latest_rows:
        security = str(row.get("security_desc") or row.get("security_type_desc") or "").strip()
        rate = _safe_float(row.get("avg_interest_rate_amt"))
        if security and rate is not None:
            rates_by_security[security] = rate
    avg_rate = _mean(rates_by_security.values())
    ok = bool(latest_rows) and avg_rate is not None
    if not ok:
        cached = _cached_treasury_avg_interest()
        if cached:
            cached["live_error"] = error
            return cached
    return _source_status(
        "treasury_avg_interest_rates",
        ok=ok,
        url=TREASURY_AVG_INTEREST_URL,
        error=error,
        record_date=latest_date,
        row_count=len(latest_rows),
        avg_interest_rate_pct=avg_rate,
        rates_by_security=rates_by_security,
    )


def _world_bank_url(countries: list[str], indicator_id: str) -> str:
    country_path = quote(";".join(countries), safe=";")
    indicator_path = quote(str(indicator_id), safe="")
    return (
        f"{WORLD_BANK_API_ROOT}/country/{country_path}/indicator/{indicator_path}"
        "?format=json&per_page=80&MRV=1"
    )


def _parse_world_bank_rows(payload: Any) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if not isinstance(payload, list) or len(payload) < 2:
        return {}, []
    meta = payload[0] if isinstance(payload[0], dict) else {}
    rows = payload[1] if isinstance(payload[1], list) else []
    return meta, [row for row in rows if isinstance(row, dict)]


def _fetch_world_bank_indicators(*, countries: list[str], user_agent: str, timeout: float) -> dict[str, Any]:
    indicators: dict[str, Any] = {}
    urls: dict[str, str] = {}
    errors: dict[str, str] = {}
    value_count = 0
    row_count = 0
    latest_update = ""
    for indicator_id, feature_name in WORLD_BANK_INDICATORS.items():
        url = _world_bank_url(countries, indicator_id)
        urls[indicator_id] = url
        payload, error = _safe_http_json(url, user_agent=user_agent, timeout=timeout)
        if error:
            errors[indicator_id] = str(error)
            continue
        meta, rows = _parse_world_bank_rows(payload)
        row_count += len(rows)
        if str(meta.get("lastupdated") or "") > latest_update:
            latest_update = str(meta.get("lastupdated") or "")
        values: dict[str, Any] = {}
        indicator_name = ""
        for row in rows:
            value = _safe_float(row.get("value"))
            country = str(row.get("countryiso3code") or "").strip().upper()
            if not country:
                country_info = row.get("country") if isinstance(row.get("country"), dict) else {}
                country = str(country_info.get("id") or "").strip().upper()
            if not country:
                continue
            indicator_info = row.get("indicator") if isinstance(row.get("indicator"), dict) else {}
            indicator_name = indicator_name or str(indicator_info.get("value") or "")
            values[country] = {
                "date": str(row.get("date") or ""),
                "value": value,
                "country_name": str((row.get("country") or {}).get("value") or "") if isinstance(row.get("country"), dict) else "",
            }
            if value is not None:
                value_count += 1
        indicators[feature_name] = {
            "indicator_id": indicator_id,
            "indicator_name": indicator_name,
            "row_count": len(rows),
            "value_count": sum(1 for item in values.values() if item.get("value") is not None),
            "values": values,
            "lastupdated": str(meta.get("lastupdated") or ""),
        }
    indicator_success_count = sum(1 for item in indicators.values() if int(item.get("row_count", 0) or 0) > 0)
    ok = indicator_success_count >= 3 and value_count >= max(8, len(countries) * 3)
    return _source_status(
        "world_bank_indicators",
        ok=ok,
        url=WORLD_BANK_API_ROOT,
        errors=errors,
        countries=countries,
        indicator_count=len(WORLD_BANK_INDICATORS),
        indicator_success_count=indicator_success_count,
        row_count=row_count,
        value_count=value_count,
        lastupdated=latest_update,
        urls=urls,
        indicators=indicators,
    )


def _wb_value(world_bank: Mapping[str, Any], feature_name: str, country: str) -> float | None:
    indicators = world_bank.get("indicators") if isinstance(world_bank.get("indicators"), dict) else {}
    feature = indicators.get(feature_name) if isinstance(indicators.get(feature_name), dict) else {}
    values = feature.get("values") if isinstance(feature.get("values"), dict) else {}
    item = values.get(country.upper()) if isinstance(values.get(country.upper()), dict) else {}
    return _safe_float(item.get("value"))


def _wb_values(world_bank: Mapping[str, Any], feature_name: str) -> list[float]:
    indicators = world_bank.get("indicators") if isinstance(world_bank.get("indicators"), dict) else {}
    feature = indicators.get(feature_name) if isinstance(indicators.get(feature_name), dict) else {}
    values = feature.get("values") if isinstance(feature.get("values"), dict) else {}
    out: list[float] = []
    for item in values.values():
        if not isinstance(item, dict):
            continue
        value = _safe_float(item.get("value"))
        if value is not None:
            out.append(value)
    return out


def _build_features(sources: Mapping[str, Any]) -> dict[str, Any]:
    treasury_debt = sources.get("treasury_debt_to_penny") if isinstance(sources.get("treasury_debt_to_penny"), dict) else {}
    treasury_rates = (
        sources.get("treasury_avg_interest_rates")
        if isinstance(sources.get("treasury_avg_interest_rates"), dict)
        else {}
    )
    world_bank = sources.get("world_bank_indicators") if isinstance(sources.get("world_bank_indicators"), dict) else {}
    total_debt = _safe_float(treasury_debt.get("total_public_debt_usd"))
    debt_daily_change = _safe_float(treasury_debt.get("daily_change_usd"))
    debt_window_change = _safe_float(treasury_debt.get("sample_window_change_usd"))
    avg_rate = _safe_float(treasury_rates.get("avg_interest_rate_pct"))
    us_gdp = _wb_value(world_bank, "gdp_current_usd", "USA")
    us_world_bank_debt_pct = _wb_value(world_bank, "central_government_debt_pct_gdp", "USA")
    inflation_values = _wb_values(world_bank, "inflation_cpi_annual_pct")
    current_account_values = _wb_values(world_bank, "current_account_pct_gdp")
    world_gdp_values = _wb_values(world_bank, "gdp_current_usd")
    debt_to_gdp = (total_debt / us_gdp) if total_debt is not None and us_gdp and us_gdp > 0 else None
    current_account_abs_mean = _mean(abs(value) for value in current_account_values)
    return {
        "us_total_public_debt_usd": total_debt,
        "us_debt_held_public_usd": _safe_float(treasury_debt.get("debt_held_public_usd")),
        "us_intragov_holdings_usd": _safe_float(treasury_debt.get("intragov_holdings_usd")),
        "us_public_debt_daily_change_usd": debt_daily_change,
        "us_public_debt_sample_window_change_usd": debt_window_change,
        "us_public_debt_to_worldbank_gdp_proxy": debt_to_gdp,
        "us_public_debt_to_worldbank_gdp_norm": _clamp01((debt_to_gdp or 0.0) / 1.5) if debt_to_gdp is not None else None,
        "us_worldbank_central_government_debt_pct_gdp": us_world_bank_debt_pct,
        "treasury_avg_interest_rate_pct": avg_rate,
        "treasury_avg_interest_rate_norm": _clamp01((avg_rate or 0.0) / 8.0) if avg_rate is not None else None,
        "world_bank_top_country_count": len(world_bank.get("countries") or []),
        "world_bank_indicator_count": int(world_bank.get("indicator_count", 0) or 0),
        "world_bank_value_count": int(world_bank.get("value_count", 0) or 0),
        "world_bank_top_gdp_sum_usd": sum(world_gdp_values) if world_gdp_values else None,
        "world_bank_inflation_mean_pct": _mean(inflation_values),
        "world_bank_current_account_abs_mean_pct_gdp": current_account_abs_mean,
        "world_bank_current_account_imbalance_norm": _clamp01((current_account_abs_mean or 0.0) / 8.0)
        if current_account_abs_mean is not None
        else None,
    }


def _parse_csv(raw: str) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for token in str(raw or "").replace("\n", ",").split(","):
        item = token.strip().upper()
        if not item or item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def collect_public_policy_context(*, countries: list[str], user_agent: str, timeout: float) -> tuple[dict[str, Any], dict[str, Any]]:
    now = _utc_now_iso()
    sources = {
        "treasury_debt_to_penny": _fetch_treasury_debt(user_agent=user_agent, timeout=timeout),
        "treasury_avg_interest_rates": _fetch_treasury_avg_interest(user_agent=user_agent, timeout=timeout),
        "world_bank_indicators": _fetch_world_bank_indicators(countries=countries, user_agent=user_agent, timeout=timeout),
    }
    source_count = sum(1 for source in sources.values() if bool(source.get("contract_participates", True)))
    ok_source_count = sum(
        1 for source in sources.values() if bool(source.get("contract_participates", True)) and bool(source.get("ok", False))
    )
    required_ok = bool(sources["treasury_debt_to_penny"].get("ok", False)) and bool(
        sources["world_bank_indicators"].get("ok", False)
    )
    ok = required_ok and ok_source_count >= 2
    features = _build_features(sources)
    status = {
        "timestamp_utc": now,
        "provider": "public_policy_context",
        "ok": ok,
        "overall_status": "ready" if ok else "degraded",
        "context_profile": "official_free_public_policy_liquidity",
        "source_count": source_count,
        "ok_source_count": ok_source_count,
        "required_sources_ok": required_ok,
        "countries": countries,
        "source_contracts": SOURCE_CONTRACTS,
        "sources": sources,
        "features": features,
    }
    payload = {
        "timestamp_utc": now,
        "provider": "public_policy_context",
        "status": status["overall_status"],
        "source_contracts": SOURCE_CONTRACTS,
        "sources": sources,
        "features": features,
    }
    return payload, status


def main() -> int:
    parser = argparse.ArgumentParser(description="Collect free official public-policy and sovereign-liquidity context.")
    parser.add_argument("--countries", default=",".join(DEFAULT_COUNTRIES))
    parser.add_argument("--timeout", type=float, default=float(os.getenv("PUBLIC_POLICY_CONTEXT_TIMEOUT_SECONDS", "12")))
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    countries = _parse_csv(args.countries) or list(DEFAULT_COUNTRIES)
    payload, status = collect_public_policy_context(
        countries=countries[:40],
        user_agent=str(os.getenv("PUBLIC_POLICY_CONTEXT_USER_AGENT") or USER_AGENT_DEFAULT).strip() or USER_AGENT_DEFAULT,
        timeout=float(args.timeout),
    )
    _write_json(PROJECT_ROOT / "exports" / "external_context" / "public_policy_context_latest.json", payload)
    _write_json(PROJECT_ROOT / "governance" / "health" / "public_policy_context_sync_latest.json", status)

    if args.json:
        print(json.dumps(status, ensure_ascii=True))
    else:
        print(
            "public_policy_context ok={ok} ok_sources={ok_sources}/{total_sources} countries={countries} profile={profile}".format(
                ok=status["ok"],
                ok_sources=status["ok_source_count"],
                total_sources=status["source_count"],
                countries=len(countries[:40]),
                profile=status["context_profile"],
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
