#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import sys
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping
from urllib.parse import urlencode


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.collector_transport import fetch_json, fetch_text
from core.economic_source_registry import load_economic_source_registry, validate_economic_source_registry
from scripts.ops.long_runtime_common import write_payload


LATEST_PATH = PROJECT_ROOT / "exports" / "external_context" / "public_financial_context_latest.json"
HEALTH_PATH = PROJECT_ROOT / "governance" / "health" / "public_financial_context_sync_latest.json"
ROUTING_CONFIG_PATH = PROJECT_ROOT / "config" / "public_financial_context_routing_v1.json"
USER_AGENT_DEFAULT = "Daniel Kingsley dan_kingsley@aol.com"
OFR_FSI_URL = "https://www.financialresearch.gov/financial-stress-index/data/fsi.csv"
FDIC_FAILURES_URL = "https://api.fdic.gov/banks/failures"
FEDERAL_REGISTER_URL = "https://www.federalregister.gov/api/v1/documents.json"
ECB_ESTR_URL = (
    "https://data-api.ecb.europa.eu/service/data/EST/"
    "B.EU000A2X2A25.WT+TT+R25+R75?lastNObservations=10&detail=full"
)
NYFED_PD_SERIESBREAKS_URL = "https://markets.newyorkfed.org/api/pd/list/seriesbreaks.json"
NYFED_PD_LATEST_URL = "https://markets.newyorkfed.org/api/pd/latest/{seriesbreak}.json"
FDIC_FINANCIALS_URL = "https://api.fdic.gov/banks/financials"
SEC_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SEC_COMPANYFACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"

DEFAULT_SYMBOLS = (
    "AAPL",
    "MSFT",
    "NVDA",
    "AMZN",
    "GOOGL",
    "META",
    "JPM",
    "XOM",
    "CVX",
    "JNJ",
    "PG",
    "ABBV",
)
FINANCIAL_AGENCIES = (
    "federal-reserve-system",
    "securities-and-exchange-commission",
    "commodity-futures-trading-commission",
    "federal-deposit-insurance-corporation",
    "comptroller-of-the-currency",
    "treasury-department",
)
SOURCE_CONTRACTS = {
    "ofr_financial_stress_index": {
        "publisher": "U.S. Office of Financial Research",
        "url": OFR_FSI_URL,
        "cadence": "daily_with_two_business_day_lag",
        "maximum_observation_age_days": 10,
        "source_confidence_norm": 0.99,
        "schema_confidence_norm": 0.98,
    },
    "fdic_bank_failures": {
        "publisher": "Federal Deposit Insurance Corporation",
        "url": FDIC_FAILURES_URL,
        "cadence": "event_driven",
        "maximum_observation_age_days": 7,
        "source_confidence_norm": 0.99,
        "schema_confidence_norm": 0.97,
    },
    "federal_register_financial_rules": {
        "publisher": "Federal Register",
        "url": FEDERAL_REGISTER_URL,
        "cadence": "business_daily",
        "maximum_observation_age_days": 7,
        "source_confidence_norm": 0.98,
        "schema_confidence_norm": 0.96,
        "legal_truth_authority": False,
    },
    "ecb_estr": {
        "publisher": "European Central Bank",
        "url": ECB_ESTR_URL,
        "cadence": "business_daily",
        "maximum_observation_age_days": 10,
        "source_confidence_norm": 0.99,
        "schema_confidence_norm": 0.98,
    },
    "sec_companyfacts": {
        "publisher": "U.S. Securities and Exchange Commission",
        "url": "https://data.sec.gov/api/xbrl/companyfacts/",
        "cadence": "filing_driven",
        "maximum_observation_age_days": 550,
        "source_confidence_norm": 0.99,
        "schema_confidence_norm": 0.96,
    },
    "nyfed_primary_dealer_statistics": {
        "publisher": "Federal Reserve Bank of New York",
        "url": NYFED_PD_SERIESBREAKS_URL,
        "cadence": "weekly_with_reporting_lag",
        "maximum_observation_age_days": 21,
        "source_confidence_norm": 0.99,
        "schema_confidence_norm": 0.98,
        "api_key_required": False,
    },
    "fdic_bank_financials": {
        "publisher": "Federal Deposit Insurance Corporation",
        "url": FDIC_FINANCIALS_URL,
        "cadence": "quarterly",
        "maximum_observation_age_days": 190,
        "source_confidence_norm": 0.99,
        "schema_confidence_norm": 0.98,
        "api_key_required": False,
    },
}
CAPABILITY_SOURCE_REQUIREMENTS = {
    "financial_statements": ("sec_companyfacts",),
    "quality_ratios": ("sec_companyfacts",),
    "operating_profitability": ("sec_companyfacts",),
    "cash_flow_quality": ("sec_companyfacts",),
    "leverage_state": ("sec_companyfacts",),
    "distress_indicators": ("sec_companyfacts",),
    "funding_stress": ("ofr_financial_stress_index", "ecb_estr"),
    "default_risk": ("ofr_financial_stress_index", "fdic_bank_failures"),
    "rates_credit_regime": ("ofr_financial_stress_index", "ecb_estr"),
    "risk_on_off_state": ("ofr_financial_stress_index",),
    "volatility_regime": ("ofr_financial_stress_index",),
    "regulatory_policy": ("federal_register_financial_rules",),
    "policy_news": ("federal_register_financial_rules",),
    "dealer_balance_sheet": ("nyfed_primary_dealer_statistics",),
    "repo_conditions": ("nyfed_primary_dealer_statistics",),
    "collateral_settlement_stress": ("nyfed_primary_dealer_statistics",),
    "bank_credit_conditions": ("fdic_bank_financials",),
}
CAPABILITY_FEATURE_REQUIREMENTS = {
    "dealer_balance_sheet": ("nyfed_dealer_treasury_inventory_pressure_norm",),
    "repo_conditions": ("nyfed_dealer_repo_imbalance_norm",),
    "collateral_settlement_stress": ("nyfed_dealer_financing_fails_pressure_norm",),
    "bank_credit_conditions": ("fdic_bank_noncurrent_loan_pressure_norm",),
}
BASELINE_SOURCE_IDS = (
    "ofr_financial_stress_index",
    "fdic_bank_failures",
    "federal_register_financial_rules",
    "ecb_estr",
    "sec_companyfacts",
)
SUPPLEMENTAL_SOURCE_IDS = (
    "nyfed_primary_dealer_statistics",
    "fdic_bank_financials",
)
NYFED_PRIMARY_DEALER_KEYS = {
    "treasury_position": "PDPOSGST-TOT",
    "corporate_position": "PDPOSCS-TOT",
    "treasury_repo": "PDSORA-UTSETTOT",
    "treasury_reverse_repo": "PDSIRRA-UTSETTOT",
    "treasury_fails_to_deliver": "PDFTD-USTET",
    "treasury_fails_to_receive": "PDFTR-USTET",
}
FACT_CONCEPTS = {
    "assets": ("Assets",),
    "assets_current": ("AssetsCurrent",),
    "liabilities": ("Liabilities",),
    "liabilities_current": ("LiabilitiesCurrent",),
    "equity": ("StockholdersEquity", "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"),
    "revenue": ("RevenueFromContractWithCustomerExcludingAssessedTax", "SalesRevenueNet", "Revenues"),
    "operating_income": ("OperatingIncomeLoss",),
    "net_income": ("NetIncomeLoss", "ProfitLoss"),
    "operating_cash_flow": ("NetCashProvidedByUsedInOperatingActivities",),
    "capital_expenditure": ("PaymentsToAcquirePropertyPlantAndEquipment", "PaymentsForAdditionsToPropertyPlantAndEquipment"),
    "cash": ("CashAndCashEquivalentsAtCarryingValue", "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents"),
    "long_term_debt": ("LongTermDebtAndFinanceLeaseObligationsCurrent", "LongTermDebtCurrent", "LongTermDebtNoncurrent", "LongTermDebt"),
}
PUBLIC_FINANCIAL_FEATURE_KEYS = (
    "public_financial_context_available_norm",
    "public_financial_source_coverage_norm",
    "public_financial_supplemental_coverage_norm",
    "ofr_financial_stress_norm",
    "ofr_credit_stress_norm",
    "ofr_funding_stress_norm",
    "ofr_safe_asset_stress_norm",
    "ofr_volatility_stress_norm",
    "ofr_equity_valuation_stress_norm",
    "fdic_failure_12m_norm",
    "fdic_failure_assets_12m_norm",
    "federal_register_financial_activity_norm",
    "federal_register_high_impact_norm",
    "ecb_estr_funding_pressure_norm",
    "ecb_estr_rate_norm",
    "ecb_estr_change_5d_norm",
    "companyfacts_coverage_norm",
    "companyfacts_operating_profitability_norm",
    "companyfacts_cash_flow_quality_norm",
    "companyfacts_distress_norm",
    "nyfed_dealer_repo_imbalance_norm",
    "nyfed_dealer_financing_fails_pressure_norm",
    "nyfed_dealer_treasury_inventory_pressure_norm",
    "nyfed_dealer_corporate_inventory_pressure_norm",
    "fdic_bank_noncurrent_loan_pressure_norm",
    "fdic_bank_deposit_funding_norm",
    "fdic_bank_lending_intensity_norm",
    "fdic_bank_asset_growth_yoy_norm",
)
PUBLIC_FINANCIAL_SYMBOL_FEATURE_KEYS = (
    "fundamental_financial_statement_coverage_norm",
    "fundamental_quality_norm",
    "fundamental_operating_profitability_norm",
    "fundamental_cash_flow_quality_norm",
    "fundamental_leverage_norm",
    "fundamental_distress_norm",
    "fundamental_current_ratio_norm",
    "fundamental_cash_to_assets_norm",
    "fundamental_free_cash_flow_positive_norm",
)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _canonical_hash(value: Any) -> str:
    raw = json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _load_routing_taxonomy(path: Path = ROUTING_CONFIG_PATH) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _economic_registry_validation() -> tuple[dict[str, Any], dict[str, Any]]:
    try:
        registry = load_economic_source_registry()
        validation = validate_economic_source_registry(registry, project_root=PROJECT_ROOT)
    except Exception as exc:
        return {}, {
            "ok": False,
            "errors": [f"registry_load_failed:{type(exc).__name__}:{str(exc)[:300]}"],
            "warnings": [],
        }
    registered = {
        str(row.get("source_id") or "")
        for row in registry.get("sources", [])
        if isinstance(row, Mapping) and str(row.get("source_id") or "")
    }
    missing = sorted(set(SOURCE_CONTRACTS) - registered)
    registered_for_producer = {
        str(row.get("source_id") or "")
        for row in registry.get("sources", [])
        if isinstance(row, Mapping)
        and str(row.get("producer_id") or "") == "public_financial_context"
        and str(row.get("source_id") or "")
    }
    unimplemented = sorted(registered_for_producer - set(SOURCE_CONTRACTS))
    validation = dict(validation)
    validation["public_financial_missing_source_ids"] = missing
    validation["public_financial_source_ids_registered"] = not missing
    validation["public_financial_unimplemented_source_ids"] = unimplemented
    validation["public_financial_registered_sources_implemented"] = not unimplemented
    validation["ok"] = bool(validation.get("ok") and not missing and not unimplemented)
    if missing:
        validation.setdefault("errors", []).append(
            f"public_financial_sources_missing_from_registry:{','.join(missing)}"
        )
    if unimplemented:
        validation.setdefault("errors", []).append(
            f"public_financial_registry_sources_without_adapters:{','.join(unimplemented)}"
        )
    return registry, validation


def _apply_routing_taxonomy(
    derived: Mapping[str, Any],
    taxonomy: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    routes = taxonomy.get("feature_routes") if isinstance(taxonomy.get("feature_routes"), Mapping) else {}
    global_features = derived.get("global_features") if isinstance(derived.get("global_features"), Mapping) else {}
    symbol_features = derived.get("symbol_features") if isinstance(derived.get("symbol_features"), Mapping) else {}
    allowed_global = {key for key, row in routes.items() if isinstance(row, Mapping) and row.get("scope") == "global"}
    allowed_symbol = {key for key, row in routes.items() if isinstance(row, Mapping) and row.get("scope") == "symbol"}
    unclassified_global = sorted(set(global_features) - allowed_global)
    unclassified_symbol = sorted(
        {
            str(key)
            for row in symbol_features.values()
            if isinstance(row, Mapping)
            for key in row
            if str(key) not in allowed_symbol
        }
    )
    classified_global = {key: value for key, value in global_features.items() if key in allowed_global}
    classified_symbols = {
        symbol: {key: value for key, value in row.items() if key in allowed_symbol}
        for symbol, row in symbol_features.items()
        if isinstance(row, Mapping)
    }
    classified_symbols = {symbol: row for symbol, row in classified_symbols.items() if row}
    classified_lineage = {
        key: value
        for key, value in (derived.get("feature_lineage") or {}).items()
        if key in classified_global
    } if isinstance(derived.get("feature_lineage"), Mapping) else {}
    validation = {
        "ok": bool(taxonomy and not unclassified_global and not unclassified_symbol),
        "unclassified_global_feature_keys": unclassified_global,
        "unclassified_symbol_feature_keys": unclassified_symbol,
        "classified_global_feature_count": len(classified_global),
        "classified_symbol_feature_count": len(allowed_symbol),
        "unclassified_feature_policy": "quarantine_from_bot_context",
    }
    return {
        "global_features": classified_global,
        "symbol_features": classified_symbols,
        "feature_lineage": classified_lineage,
    }, validation


def _safe_float(value: Any) -> float | None:
    try:
        result = float(str(value).replace(",", "").strip())
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _clamp01(value: float) -> float:
    return max(0.0, min(float(value), 1.0))


def _signed_norm(value: float, scale: float) -> float:
    return _clamp01(0.5 + 0.5 * math.tanh(float(value) / max(abs(float(scale)), 1e-12)))


def _mean(values: Iterable[float | None]) -> float | None:
    clean = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return sum(clean) / len(clean) if clean else None


def _parse_date(value: Any) -> date | None:
    text = str(value or "").strip()
    if not text:
        return None
    for fmt in ("%Y-%m-%d", "%m/%d/%Y", "%Y%m%d"):
        try:
            return datetime.strptime(text[:10], fmt).date()
        except ValueError:
            continue
    return None


def _parse_timestamp(value: Any) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        parsed_date = _parse_date(text)
        return datetime.combine(parsed_date, datetime.min.time(), tzinfo=timezone.utc) if parsed_date else None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _receipt(result: Mapping[str, Any]) -> dict[str, Any]:
    keys = (
        "ok",
        "url",
        "status_code",
        "fetched_utc",
        "attempt_count",
        "duration_ms",
        "payload_sha256",
        "size_bytes",
        "source_confidence_norm",
        "schema_confidence_norm",
        "transport_contract_version",
        "transport_receipt_sha256",
        "error_class",
        "error",
    )
    return {key: result.get(key) for key in keys if result.get(key) not in (None, "")}


def _transport_text(
    source_id: str,
    url: str,
    *,
    user_agent: str,
    timeout: float,
    accept: str,
    capability_ids: Iterable[str],
) -> dict[str, Any]:
    contract = SOURCE_CONTRACTS[source_id]
    return fetch_text(
        url=url,
        user_agent=user_agent,
        timeout=timeout,
        accept=accept,
        retries=1,
        collector_key="public_financial_context",
        source_name=source_id,
        entity_key=source_id,
        project_root=PROJECT_ROOT,
        source_confidence_norm=float(contract["source_confidence_norm"]),
        schema_confidence_norm=float(contract["schema_confidence_norm"]),
        route_id=f"public_financial_context:{source_id}",
        capability_ids=tuple(capability_ids),
        max_response_bytes=16 * 1024 * 1024,
    )


def _transport_json(
    source_id: str,
    url: str,
    *,
    user_agent: str,
    timeout: float,
    entity_key: str,
    capability_ids: Iterable[str],
) -> dict[str, Any]:
    contract = SOURCE_CONTRACTS[source_id]
    return fetch_json(
        url=url,
        user_agent=user_agent,
        timeout=timeout,
        retries=1,
        collector_key="public_financial_context",
        source_name=source_id,
        entity_key=entity_key,
        project_root=PROJECT_ROOT,
        source_confidence_norm=float(contract["source_confidence_norm"]),
        schema_confidence_norm=float(contract["schema_confidence_norm"]),
        route_id=f"public_financial_context:{source_id}",
        capability_ids=tuple(capability_ids),
        max_response_bytes=16 * 1024 * 1024,
    )


def _source_row(source_id: str, *, ok: bool, transport: Mapping[str, Any], **values: Any) -> dict[str, Any]:
    contract = SOURCE_CONTRACTS[source_id]
    return {
        **contract,
        "source_id": source_id,
        "ok": bool(ok),
        "contract_participates": source_id in BASELINE_SOURCE_IDS,
        "supplemental": source_id in SUPPLEMENTAL_SOURCE_IDS,
        "transport": _receipt(transport),
        "terms": {
            "api_key_required": bool(contract.get("api_key_required", False)),
            "redistribution_allowed": False,
            "commercial_deployment_terms_review_required": True,
            "paper_and_research_context_only": True,
        },
        **values,
    }


def _isolated_source(source_id: str, fetcher: Any) -> dict[str, Any]:
    try:
        row = fetcher()
        if not isinstance(row, Mapping):
            raise TypeError("source adapter returned a non-mapping payload")
        return dict(row)
    except Exception as exc:
        return _source_row(
            source_id,
            ok=False,
            transport={},
            observation_time=None,
            isolated_failure=True,
            error_class=type(exc).__name__,
            error=str(exc)[:500],
        )


def _stress_norm(value: Any) -> float | None:
    parsed = _safe_float(value)
    return _clamp01(0.5 + parsed / 8.0) if parsed is not None else None


def _parse_ofr_csv(raw: str, *, as_of: date) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in csv.DictReader(io.StringIO(raw or "")):
        observed = _parse_date(row.get("Date"))
        if observed is None or observed > as_of:
            continue
        parsed = {
            "date": observed.isoformat(),
            "financial_stress": _safe_float(row.get("OFR FSI")),
            "credit": _safe_float(row.get("Credit")),
            "equity_valuation": _safe_float(row.get("Equity valuation")),
            "safe_assets": _safe_float(row.get("Safe assets")),
            "funding": _safe_float(row.get("Funding")),
            "volatility": _safe_float(row.get("Volatility")),
            "united_states": _safe_float(row.get("United States")),
            "other_advanced_economies": _safe_float(row.get("Other advanced economies")),
            "emerging_markets": _safe_float(row.get("Emerging markets")),
        }
        if parsed["financial_stress"] is not None:
            rows.append(parsed)
    return sorted(rows, key=lambda item: str(item["date"]))


def _fetch_ofr(*, now: datetime, user_agent: str, timeout: float) -> dict[str, Any]:
    transport = _transport_text(
        "ofr_financial_stress_index",
        OFR_FSI_URL,
        user_agent=user_agent,
        timeout=timeout,
        accept="text/csv,*/*",
        capability_ids=("funding_stress", "default_risk", "rates_credit_regime", "risk_on_off_state", "volatility_regime"),
    )
    rows = _parse_ofr_csv(str(transport.get("text") or ""), as_of=now.date()) if transport.get("ok") else []
    latest = rows[-1] if rows else {}
    observed = _parse_date(latest.get("date"))
    age_days = (now.date() - observed).days if observed else None
    ok = bool(rows and age_days is not None and age_days <= 10)
    return _source_row(
        "ofr_financial_stress_index",
        ok=ok,
        transport=transport,
        observation_time=f"{observed.isoformat()}T00:00:00+00:00" if observed else None,
        observation_age_days=age_days,
        latest=latest,
        recent_rows=rows[-20:],
        future_rows_rejected=True,
    )


def _fdic_url(now: datetime) -> str:
    params = {
        "filters": f'FAILYR:["{now.year - 2}" TO "{now.year}"]',
        "fields": "NAME,CERT,FAILDATE,FAILYR,CITYST,PSTALP,RESTYPE1,QBFDEP,QBFASSET,COST",
        "sort_by": "FAILDATE",
        "sort_order": "DESC",
        "limit": "100",
        "format": "json",
    }
    return f"{FDIC_FAILURES_URL}?{urlencode(params)}"


def _fetch_fdic(*, now: datetime, user_agent: str, timeout: float) -> dict[str, Any]:
    url = _fdic_url(now)
    transport = _transport_json(
        "fdic_bank_failures",
        url,
        user_agent=user_agent,
        timeout=timeout,
        entity_key="us_bank_failures_recent",
        capability_ids=("default_risk", "rates_credit_regime"),
    )
    payload = transport.get("json") if isinstance(transport.get("json"), Mapping) else {}
    parsed_rows: list[dict[str, Any]] = []
    for wrapper in payload.get("data", []) if isinstance(payload.get("data"), list) else []:
        row = wrapper.get("data") if isinstance(wrapper, Mapping) and isinstance(wrapper.get("data"), Mapping) else {}
        failed = _parse_date(row.get("FAILDATE"))
        if failed is None or failed > now.date():
            continue
        parsed_rows.append(
            {
                "name": str(row.get("NAME") or ""),
                "cert": str(row.get("CERT") or ""),
                "failure_date": failed.isoformat(),
                "location": str(row.get("CITYST") or ""),
                "resolution_type": str(row.get("RESTYPE1") or ""),
                "assets_thousands_usd": _safe_float(row.get("QBFASSET")),
                "deposits_thousands_usd": _safe_float(row.get("QBFDEP")),
                "estimated_cost_thousands_usd": _safe_float(row.get("COST")),
            }
        )
    cutoff = now.date() - timedelta(days=365)
    recent = [row for row in parsed_rows if (_parse_date(row.get("failure_date")) or date.min) >= cutoff]
    assets = sum(float(row.get("assets_thousands_usd") or 0.0) for row in recent)
    latest_date = _parse_date(parsed_rows[0].get("failure_date")) if parsed_rows else None
    return _source_row(
        "fdic_bank_failures",
        ok=bool(transport.get("ok") and isinstance(payload.get("data"), list)),
        transport=transport,
        observation_time=now.isoformat(),
        latest_failure_date=latest_date.isoformat() if latest_date else None,
        failures_12m=len(recent),
        failed_assets_12m_thousands_usd=assets,
        recent_failures=recent[:30],
        query_total=int(((payload.get("meta") or {}).get("total", 0) or 0)) if isinstance(payload.get("meta"), Mapping) else 0,
        future_rows_rejected=True,
    )


def _federal_register_url(now: datetime) -> str:
    params: list[tuple[str, str]] = [
        ("per_page", "100"),
        ("order", "newest"),
        ("conditions[publication_date][gte]", (now.date() - timedelta(days=30)).isoformat()),
    ]
    params.extend(("conditions[agencies][]", agency) for agency in FINANCIAL_AGENCIES)
    return f"{FEDERAL_REGISTER_URL}?{urlencode(params)}"


def _fetch_federal_register(*, now: datetime, user_agent: str, timeout: float) -> dict[str, Any]:
    url = _federal_register_url(now)
    transport = _transport_json(
        "federal_register_financial_rules",
        url,
        user_agent=user_agent,
        timeout=timeout,
        entity_key="financial_agency_documents_30d",
        capability_ids=("regulatory_policy", "policy_news"),
    )
    payload = transport.get("json") if isinstance(transport.get("json"), Mapping) else {}
    rows: list[dict[str, Any]] = []
    for row in payload.get("results", []) if isinstance(payload.get("results"), list) else []:
        if not isinstance(row, Mapping):
            continue
        published = _parse_date(row.get("publication_date"))
        if published is None or published > now.date():
            continue
        agencies = [
            str(item.get("name") or "")
            for item in row.get("agencies", [])
            if isinstance(item, Mapping) and str(item.get("name") or "")
        ]
        rows.append(
            {
                "document_number": str(row.get("document_number") or ""),
                "publication_date": published.isoformat(),
                "type": str(row.get("type") or ""),
                "title": str(row.get("title") or ""),
                "abstract": str(row.get("abstract") or "")[:1200],
                "agencies": agencies,
                "html_url": str(row.get("html_url") or ""),
            }
        )
    high_impact_types = {"Rule", "Proposed Rule", "Presidential Document"}
    recent_cutoff = now.date() - timedelta(days=7)
    recent = [row for row in rows if (_parse_date(row.get("publication_date")) or date.min) >= recent_cutoff]
    high_impact = [row for row in recent if row.get("type") in high_impact_types]
    latest = max((_parse_date(row.get("publication_date")) for row in rows), default=None)
    return _source_row(
        "federal_register_financial_rules",
        ok=bool(transport.get("ok") and isinstance(payload.get("results"), list)),
        transport=transport,
        observation_time=f"{latest.isoformat()}T00:00:00+00:00" if latest else now.isoformat(),
        document_count_30d=len(rows),
        document_count_7d=len(recent),
        high_impact_count_7d=len(high_impact),
        documents=rows[:60],
        future_rows_rejected=True,
        legal_truth_authority=False,
    )


def _parse_ecb_csv(raw: str, *, as_of: date) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    for row in csv.DictReader(io.StringIO(raw or "")):
        observed = _parse_date(row.get("Time period or range"))
        value = _safe_float(row.get("Observation value"))
        key = str(row.get("Key") or "")
        if observed is None or observed > as_of or value is None:
            continue
        data_type = key.rsplit(".", 1)[-1]
        out.setdefault(data_type, []).append({"date": observed.isoformat(), "value": value})
    for rows in out.values():
        rows.sort(key=lambda item: str(item["date"]))
    return out


def _fetch_ecb(*, now: datetime, user_agent: str, timeout: float) -> dict[str, Any]:
    transport = _transport_text(
        "ecb_estr",
        ECB_ESTR_URL,
        user_agent=user_agent,
        timeout=timeout,
        accept="application/vnd.ecb.data+csv;version=1.0.0",
        capability_ids=("funding_stress", "rates_credit_regime"),
    )
    series = _parse_ecb_csv(str(transport.get("text") or ""), as_of=now.date()) if transport.get("ok") else {}
    rate_rows = series.get("WT", [])
    latest = rate_rows[-1] if rate_rows else {}
    observed = _parse_date(latest.get("date"))
    age_days = (now.date() - observed).days if observed else None
    latest_rate = _safe_float(latest.get("value"))
    prior_rate = _safe_float(rate_rows[-6].get("value")) if len(rate_rows) >= 6 else None
    r25 = _safe_float((series.get("R25") or [{}])[-1].get("value"))
    r75 = _safe_float((series.get("R75") or [{}])[-1].get("value"))
    volume_rows = series.get("TT", [])
    latest_volume = _safe_float(volume_rows[-1].get("value")) if volume_rows else None
    mean_volume = _mean(_safe_float(row.get("value")) for row in volume_rows)
    return _source_row(
        "ecb_estr",
        ok=bool(rate_rows and age_days is not None and age_days <= 10),
        transport=transport,
        observation_time=f"{observed.isoformat()}T00:00:00+00:00" if observed else None,
        observation_age_days=age_days,
        rate_pct=latest_rate,
        rate_change_5d_pct=(latest_rate - prior_rate) if latest_rate is not None and prior_rate is not None else None,
        interquartile_range_pct=(r75 - r25) if r75 is not None and r25 is not None else None,
        total_volume_eur_millions=latest_volume,
        ten_observation_mean_volume_eur_millions=mean_volume,
        series=series,
        future_rows_rejected=True,
    )


def _select_nyfed_seriesbreak(payload: Any, *, as_of: date) -> dict[str, Any]:
    pd_payload = payload.get("pd") if isinstance(payload, Mapping) and isinstance(payload.get("pd"), Mapping) else {}
    rows = pd_payload.get("seriesbreaks") if isinstance(pd_payload.get("seriesbreaks"), list) else []
    candidates: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        start = _parse_date(row.get("startdate"))
        end = _parse_date(row.get("enddate"))
        seriesbreak = str(row.get("seriesbreak") or "").strip()
        if not seriesbreak or start is None or end is None or start > as_of or end < as_of:
            continue
        candidates.append(
            {
                "seriesbreak": seriesbreak,
                "label": str(row.get("label") or ""),
                "start_date": start.isoformat(),
                "end_date": end.isoformat(),
            }
        )
    return max(candidates, key=lambda row: str(row["start_date"]), default={})


def _parse_nyfed_primary_dealer(payload: Any, *, as_of: date) -> dict[str, Any]:
    pd_payload = payload.get("pd") if isinstance(payload, Mapping) and isinstance(payload.get("pd"), Mapping) else {}
    rows = pd_payload.get("timeseries") if isinstance(pd_payload.get("timeseries"), list) else []
    wanted = {value: key for key, value in NYFED_PRIMARY_DEALER_KEYS.items()}
    selected: dict[str, dict[str, Any]] = {}
    future_rejected = 0
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        key_id = str(row.get("keyid") or "").strip()
        name = wanted.get(key_id)
        if not name:
            continue
        observed = _parse_date(row.get("asofdate"))
        if observed is None or observed > as_of:
            future_rejected += 1
            continue
        value = _safe_float(row.get("value"))
        if value is None:
            continue
        prior = selected.get(name)
        if prior is None or str(prior.get("date") or "") < observed.isoformat():
            selected[name] = {
                "series_id": key_id,
                "date": observed.isoformat(),
                "value_native_units": value,
            }
    observation_dates = sorted({str(row["date"]) for row in selected.values()})
    return {
        "observation_date": observation_dates[-1] if observation_dates else None,
        "observation_dates": observation_dates,
        "values": {name: row["value_native_units"] for name, row in selected.items()},
        "series": selected,
        "selected_series_count": len(selected),
        "future_rows_rejected_count": future_rejected,
    }


def _fetch_nyfed_primary_dealer(*, now: datetime, user_agent: str, timeout: float) -> dict[str, Any]:
    discovery = _transport_json(
        "nyfed_primary_dealer_statistics",
        NYFED_PD_SERIESBREAKS_URL,
        user_agent=user_agent,
        timeout=timeout,
        entity_key="primary_dealer_seriesbreaks",
        capability_ids=("dealer_balance_sheet", "repo_conditions", "collateral_settlement_stress"),
    )
    seriesbreak = _select_nyfed_seriesbreak(discovery.get("json"), as_of=now.date()) if discovery.get("ok") else {}
    seriesbreak_id = str(seriesbreak.get("seriesbreak") or "")
    if not seriesbreak_id:
        return _source_row(
            "nyfed_primary_dealer_statistics",
            ok=False,
            transport=discovery,
            observation_time=None,
            discovery_transport=_receipt(discovery),
            error="active_seriesbreak_not_found",
            future_rows_rejected=True,
        )
    latest_url = NYFED_PD_LATEST_URL.format(seriesbreak=seriesbreak_id)
    transport = _transport_json(
        "nyfed_primary_dealer_statistics",
        latest_url,
        user_agent=user_agent,
        timeout=timeout,
        entity_key=f"primary_dealer_latest:{seriesbreak_id}",
        capability_ids=("dealer_balance_sheet", "repo_conditions", "collateral_settlement_stress"),
    )
    parsed = _parse_nyfed_primary_dealer(transport.get("json"), as_of=now.date()) if transport.get("ok") else {}
    observed = _parse_date(parsed.get("observation_date"))
    age_days = (now.date() - observed).days if observed else None
    values = parsed.get("values") if isinstance(parsed.get("values"), Mapping) else {}
    repo = _safe_float(values.get("treasury_repo"))
    reverse_repo = _safe_float(values.get("treasury_reverse_repo"))
    fails_deliver = _safe_float(values.get("treasury_fails_to_deliver"))
    fails_receive = _safe_float(values.get("treasury_fails_to_receive"))
    treasury_position = _safe_float(values.get("treasury_position"))
    corporate_position = _safe_float(values.get("corporate_position"))
    financing_base = (repo + reverse_repo) if repo is not None and reverse_repo is not None else None
    ok = bool(
        transport.get("ok")
        and int(parsed.get("selected_series_count", 0) or 0) >= 5
        and age_days is not None
        and age_days <= int(SOURCE_CONTRACTS["nyfed_primary_dealer_statistics"]["maximum_observation_age_days"])
    )
    return _source_row(
        "nyfed_primary_dealer_statistics",
        ok=ok,
        transport=transport,
        observation_time=f"{observed.isoformat()}T00:00:00+00:00" if observed else None,
        observation_age_days=age_days,
        active_seriesbreak=seriesbreak,
        discovery_transport=_receipt(discovery),
        selected_series_count=int(parsed.get("selected_series_count", 0) or 0),
        expected_series_count=len(NYFED_PRIMARY_DEALER_KEYS),
        values_native_units=dict(values),
        series=parsed.get("series", {}),
        repo_reverse_repo_total_native_units=financing_base,
        repo_imbalance_ratio=((repo - reverse_repo) / financing_base) if financing_base and repo is not None and reverse_repo is not None else None,
        financing_fails_ratio=((fails_deliver + fails_receive) / financing_base) if financing_base and fails_deliver is not None and fails_receive is not None else None,
        treasury_inventory_ratio=(abs(treasury_position) / financing_base) if financing_base and treasury_position is not None else None,
        corporate_inventory_ratio=(abs(corporate_position) / financing_base) if financing_base and corporate_position is not None else None,
        future_rows_rejected=True,
        future_rows_rejected_count=int(parsed.get("future_rows_rejected_count", 0) or 0),
    )


def _fdic_financials_url() -> str:
    params = {
        "fields": "REPDTE",
        "agg_by": "REPDTE",
        "agg_sum_fields": "ASSET,DEP,LNLSNET,NCLNLS,NETINC",
        "agg_limit": "8",
        "sort_by": "REPDTE",
        "sort_order": "DESC",
        "limit": "1",
        "format": "json",
    }
    return f"{FDIC_FINANCIALS_URL}?{urlencode(params)}"


def _parse_fdic_financials(payload: Any, *, as_of: date) -> dict[str, Any]:
    rows = payload.get("data") if isinstance(payload, Mapping) and isinstance(payload.get("data"), list) else []
    parsed: list[dict[str, Any]] = []
    future_rejected = 0
    for wrapper in rows:
        row = wrapper.get("data") if isinstance(wrapper, Mapping) and isinstance(wrapper.get("data"), Mapping) else {}
        observed = _parse_date(row.get("REPDTE"))
        if observed is None:
            continue
        if observed > as_of:
            future_rejected += 1
            continue
        parsed.append(
            {
                "report_date": observed.isoformat(),
                "institution_count": int(_safe_float(row.get("count")) or 0),
                "assets_thousands_usd": _safe_float(row.get("sum_ASSET")),
                "deposits_thousands_usd": _safe_float(row.get("sum_DEP")),
                "loans_net_thousands_usd": _safe_float(row.get("sum_LNLSNET")),
                "noncurrent_loans_thousands_usd": _safe_float(row.get("sum_NCLNLS")),
                "net_income_thousands_usd": _safe_float(row.get("sum_NETINC")),
            }
        )
    parsed.sort(key=lambda row: str(row["report_date"]), reverse=True)
    latest = parsed[0] if parsed else {}
    year_ago = parsed[4] if len(parsed) >= 5 else {}
    assets = _safe_float(latest.get("assets_thousands_usd"))
    deposits = _safe_float(latest.get("deposits_thousands_usd"))
    loans = _safe_float(latest.get("loans_net_thousands_usd"))
    noncurrent = _safe_float(latest.get("noncurrent_loans_thousands_usd"))
    prior_assets = _safe_float(year_ago.get("assets_thousands_usd"))
    return {
        "latest": latest,
        "recent_quarters": parsed,
        "report_date": latest.get("report_date"),
        "deposit_to_asset_ratio": deposits / assets if assets and deposits is not None else None,
        "loan_to_asset_ratio": loans / assets if assets and loans is not None else None,
        "noncurrent_loan_ratio": noncurrent / loans if loans and noncurrent is not None else None,
        "asset_growth_yoy_ratio": (assets / prior_assets - 1.0) if assets and prior_assets else None,
        "future_rows_rejected_count": future_rejected,
    }


def _fetch_fdic_financials(*, now: datetime, user_agent: str, timeout: float) -> dict[str, Any]:
    url = _fdic_financials_url()
    transport = _transport_json(
        "fdic_bank_financials",
        url,
        user_agent=user_agent,
        timeout=timeout,
        entity_key="us_bank_financials_aggregate",
        capability_ids=("bank_credit_conditions",),
    )
    payload = transport.get("json") if isinstance(transport.get("json"), Mapping) else {}
    parsed = _parse_fdic_financials(payload, as_of=now.date()) if transport.get("ok") else {}
    observed = _parse_date(parsed.get("report_date"))
    age_days = (now.date() - observed).days if observed else None
    ok = bool(
        transport.get("ok")
        and len(parsed.get("recent_quarters", [])) >= 5
        and _safe_float(parsed.get("noncurrent_loan_ratio")) is not None
        and age_days is not None
        and age_days <= int(SOURCE_CONTRACTS["fdic_bank_financials"]["maximum_observation_age_days"])
    )
    return _source_row(
        "fdic_bank_financials",
        ok=ok,
        transport=transport,
        observation_time=f"{observed.isoformat()}T00:00:00+00:00" if observed else None,
        observation_age_days=age_days,
        latest=parsed.get("latest", {}),
        recent_quarters=parsed.get("recent_quarters", []),
        deposit_to_asset_ratio=parsed.get("deposit_to_asset_ratio"),
        loan_to_asset_ratio=parsed.get("loan_to_asset_ratio"),
        noncurrent_loan_ratio=parsed.get("noncurrent_loan_ratio"),
        asset_growth_yoy_ratio=parsed.get("asset_growth_yoy_ratio"),
        future_rows_rejected=True,
        future_rows_rejected_count=int(parsed.get("future_rows_rejected_count", 0) or 0),
        aggregation_contract="server_side_quarterly_aggregate_no_institution_bulk_download",
    )


def _ticker_map(payload: Any) -> dict[str, str]:
    out: dict[str, str] = {}
    if not isinstance(payload, Mapping):
        return out
    for row in payload.values():
        if not isinstance(row, Mapping):
            continue
        symbol = str(row.get("ticker") or "").strip().upper().replace(".", "-")
        cik = str(row.get("cik_str") or "").strip()
        if symbol and cik.isdigit():
            out[symbol] = cik.zfill(10)
    return out


def _fact_rows(payload: Mapping[str, Any], concepts: Iterable[str]) -> list[dict[str, Any]]:
    facts = payload.get("facts") if isinstance(payload.get("facts"), Mapping) else {}
    us_gaap = facts.get("us-gaap") if isinstance(facts.get("us-gaap"), Mapping) else {}
    for concept in concepts:
        fact = us_gaap.get(concept) if isinstance(us_gaap.get(concept), Mapping) else {}
        units = fact.get("units") if isinstance(fact.get("units"), Mapping) else {}
        rows: list[dict[str, Any]] = []
        for unit in ("USD", "shares", "USD/shares", "pure"):
            if isinstance(units.get(unit), list):
                rows.extend(
                    dict(row, unit=unit)
                    for row in units[unit]
                    if isinstance(row, Mapping)
                )
        if rows:
            return [dict(row, concept=concept) for row in rows]
    return []


def _latest_fact(payload: Mapping[str, Any], concepts: Iterable[str], *, as_of: date) -> dict[str, Any]:
    candidates: list[dict[str, Any]] = []
    for row in _fact_rows(payload, concepts):
        filed = _parse_date(row.get("filed"))
        ended = _parse_date(row.get("end"))
        value = _safe_float(row.get("val"))
        if filed is None or ended is None or filed > as_of or ended > as_of or value is None:
            continue
        if str(row.get("form") or "") not in {"10-K", "10-Q", "20-F", "40-F", "6-K"}:
            continue
        candidates.append(
            {
                "concept": str(row.get("concept") or ""),
                "value": value,
                "unit": str(row.get("unit") or "USD"),
                "start": str(row.get("start") or ""),
                "end": ended.isoformat(),
                "filed": filed.isoformat(),
                "form": str(row.get("form") or ""),
                "accession_number": str(row.get("accn") or ""),
            }
        )
    return max(candidates, key=lambda row: (str(row["end"]), str(row["filed"]), str(row["accession_number"])), default={})


def _ratio(numerator: Mapping[str, Any], denominator: Mapping[str, Any], *, maximum_end_gap_days: int = 120) -> float | None:
    num = _safe_float(numerator.get("value"))
    den = _safe_float(denominator.get("value"))
    num_end = _parse_date(numerator.get("end"))
    den_end = _parse_date(denominator.get("end"))
    if num is None or den is None or abs(den) < 1e-12 or num_end is None or den_end is None:
        return None
    if abs((num_end - den_end).days) > maximum_end_gap_days:
        return None
    return num / den


def _companyfacts_symbol_row(symbol: str, cik: str, payload: Mapping[str, Any], *, as_of: date) -> dict[str, Any]:
    facts = {name: _latest_fact(payload, concepts, as_of=as_of) for name, concepts in FACT_CONCEPTS.items()}
    assets = facts["assets"]
    liabilities = facts["liabilities"]
    equity = facts["equity"]
    revenue = facts["revenue"]
    operating_income = facts["operating_income"]
    net_income = facts["net_income"]
    operating_cash = facts["operating_cash_flow"]
    capex = facts["capital_expenditure"]
    current_ratio = _ratio(facts["assets_current"], facts["liabilities_current"])
    liabilities_to_assets = _ratio(liabilities, assets)
    equity_to_assets = _ratio(equity, assets)
    operating_margin = _ratio(operating_income, revenue)
    net_margin = _ratio(net_income, revenue)
    cash_to_assets = _ratio(facts["cash"], assets)
    cash_flow_to_income = _ratio(operating_cash, net_income)
    operating_cash_value = _safe_float(operating_cash.get("value"))
    capex_value = _safe_float(capex.get("value"))
    free_cash_flow = operating_cash_value - abs(capex_value) if operating_cash_value is not None and capex_value is not None else None
    core_statement_count = sum(1 for name in ("assets", "liabilities", "equity", "revenue", "net_income", "operating_cash_flow") if facts[name])
    statement_coverage = _clamp01(core_statement_count / 6.0)
    profitability = _mean(
        (
            _clamp01(0.5 + operating_margin / 0.6) if operating_margin is not None else None,
            _clamp01(0.5 + net_margin / 0.4) if net_margin is not None else None,
        )
    )
    cash_flow_quality = _mean(
        (
            _clamp01(0.5 + (cash_flow_to_income or 0.0) / 4.0) if cash_flow_to_income is not None else None,
            1.0 if free_cash_flow is not None and free_cash_flow > 0.0 else (0.0 if free_cash_flow is not None else None),
            1.0 if operating_cash_value is not None and operating_cash_value > 0.0 else (0.0 if operating_cash_value is not None else None),
        )
    )
    quality = _mean(
        (
            _clamp01((current_ratio or 0.0) / 2.5) if current_ratio is not None else None,
            _clamp01((equity_to_assets or 0.0) / 0.75) if equity_to_assets is not None else None,
            profitability,
            cash_flow_quality,
        )
    )
    leverage = _clamp01(liabilities_to_assets) if liabilities_to_assets is not None else None
    distress = _mean(
        (
            leverage,
            0.0 if operating_cash_value is not None and operating_cash_value > 0.0 else (1.0 if operating_cash_value is not None else None),
            0.0 if _safe_float(net_income.get("value")) is not None and float(net_income["value"]) > 0.0 else (1.0 if net_income else None),
            (1.0 - quality) if quality is not None else None,
        )
    )
    filed_dates = [_parse_date(row.get("filed")) for row in facts.values() if row]
    ended_dates = [_parse_date(row.get("end")) for row in facts.values() if row]
    latest_filed = max((value for value in filed_dates if value is not None), default=None)
    latest_end = max((value for value in ended_dates if value is not None), default=None)
    features = {
        "fundamental_financial_statement_coverage_norm": statement_coverage,
        "fundamental_quality_norm": quality,
        "fundamental_operating_profitability_norm": profitability,
        "fundamental_cash_flow_quality_norm": cash_flow_quality,
        "fundamental_leverage_norm": leverage,
        "fundamental_distress_norm": distress,
        "fundamental_current_ratio_norm": _clamp01((current_ratio or 0.0) / 3.0) if current_ratio is not None else None,
        "fundamental_cash_to_assets_norm": _clamp01((cash_to_assets or 0.0) / 0.5) if cash_to_assets is not None else None,
        "fundamental_free_cash_flow_positive_norm": 1.0 if free_cash_flow is not None and free_cash_flow > 0.0 else (0.0 if free_cash_flow is not None else None),
    }
    return {
        "symbol": symbol,
        "cik": cik,
        "entity_name": str(payload.get("entityName") or ""),
        "latest_period_end": latest_end.isoformat() if latest_end else None,
        "latest_filed_date": latest_filed.isoformat() if latest_filed else None,
        "facts": facts,
        "ratios": {
            "current_ratio": current_ratio,
            "liabilities_to_assets": liabilities_to_assets,
            "equity_to_assets": equity_to_assets,
            "operating_margin": operating_margin,
            "net_margin": net_margin,
            "cash_to_assets": cash_to_assets,
            "cash_flow_to_income": cash_flow_to_income,
            "free_cash_flow_usd": free_cash_flow,
        },
        "features": {key: round(value, 8) for key, value in features.items() if value is not None},
    }


def _previous_company_rows() -> dict[str, dict[str, Any]]:
    try:
        payload = json.loads(LATEST_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}
    rows = ((payload.get("sources") or {}).get("sec_companyfacts") or {}).get("symbol_rows", [])
    return {
        str(row.get("symbol") or "").upper(): dict(row)
        for row in rows
        if isinstance(row, Mapping) and str(row.get("symbol") or "")
    } if isinstance(rows, list) else {}


def _fetch_companyfacts(
    *,
    now: datetime,
    symbols: list[str],
    user_agent: str,
    timeout: float,
    pause_seconds: float,
) -> dict[str, Any]:
    ticker_transport = _transport_json(
        "sec_companyfacts",
        SEC_TICKERS_URL,
        user_agent=user_agent,
        timeout=timeout,
        entity_key="sec_ticker_map",
        capability_ids=("financial_statements",),
    )
    ticker_by_symbol = _ticker_map(ticker_transport.get("json")) if ticker_transport.get("ok") else {}
    previous = _previous_company_rows()
    symbol_rows: list[dict[str, Any]] = []
    fetch_receipts: dict[str, Any] = {}
    live_count = 0
    cache_count = 0
    for symbol in symbols:
        cik = ticker_by_symbol.get(symbol)
        if not cik:
            continue
        transport = _transport_json(
            "sec_companyfacts",
            SEC_COMPANYFACTS_URL.format(cik=cik),
            user_agent=user_agent,
            timeout=timeout,
            entity_key=symbol,
            capability_ids=("financial_statements", "quality_ratios", "operating_profitability", "cash_flow_quality", "leverage_state", "distress_indicators"),
        )
        fetch_receipts[symbol] = _receipt(transport)
        payload = transport.get("json") if isinstance(transport.get("json"), Mapping) else {}
        row = _companyfacts_symbol_row(symbol, cik, payload, as_of=now.date()) if payload else {}
        if row and float((row.get("features") or {}).get("fundamental_financial_statement_coverage_norm", 0.0) or 0.0) > 0.0:
            row["cached_fallback"] = False
            symbol_rows.append(row)
            live_count += 1
        else:
            cached = previous.get(symbol, {})
            filed = _parse_date(cached.get("latest_filed_date"))
            if cached and filed and (now.date() - filed).days <= 550:
                cached["cached_fallback"] = True
                cached["cache_reason"] = "live_companyfacts_unavailable"
                symbol_rows.append(cached)
                cache_count += 1
        if pause_seconds > 0.0:
            time.sleep(max(float(pause_seconds), 0.0))
    coverage_values = [
        _safe_float((row.get("features") or {}).get("fundamental_financial_statement_coverage_norm"))
        for row in symbol_rows
    ]
    coverage = len(symbol_rows) / max(len(symbols), 1)
    average_statement_coverage = _mean(coverage_values) or 0.0
    ok = bool(ticker_by_symbol and coverage >= 0.5 and average_statement_coverage >= 0.5)
    latest_filed = max((_parse_date(row.get("latest_filed_date")) for row in symbol_rows), default=None)
    return _source_row(
        "sec_companyfacts",
        ok=ok,
        transport=ticker_transport,
        observation_time=f"{latest_filed.isoformat()}T00:00:00+00:00" if latest_filed else None,
        configured_symbols=symbols,
        tracked_symbol_count=len(symbol_rows),
        live_symbol_count=live_count,
        cached_symbol_count=cache_count,
        symbol_coverage_norm=round(coverage, 8),
        average_statement_coverage_norm=round(average_statement_coverage, 8),
        symbol_rows=symbol_rows,
        fetch_receipts=fetch_receipts,
        future_rows_rejected=True,
    )


def _build_derived(sources: Mapping[str, Any]) -> dict[str, Any]:
    ofr = sources.get("ofr_financial_stress_index") if isinstance(sources.get("ofr_financial_stress_index"), Mapping) else {}
    fdic = sources.get("fdic_bank_failures") if isinstance(sources.get("fdic_bank_failures"), Mapping) else {}
    federal_register = sources.get("federal_register_financial_rules") if isinstance(sources.get("federal_register_financial_rules"), Mapping) else {}
    ecb = sources.get("ecb_estr") if isinstance(sources.get("ecb_estr"), Mapping) else {}
    company = sources.get("sec_companyfacts") if isinstance(sources.get("sec_companyfacts"), Mapping) else {}
    nyfed_dealer = sources.get("nyfed_primary_dealer_statistics") if isinstance(sources.get("nyfed_primary_dealer_statistics"), Mapping) else {}
    fdic_financials = sources.get("fdic_bank_financials") if isinstance(sources.get("fdic_bank_financials"), Mapping) else {}
    ofr_latest = ofr.get("latest") if isinstance(ofr.get("latest"), Mapping) else {}
    estr_rate = _safe_float(ecb.get("rate_pct"))
    estr_change = _safe_float(ecb.get("rate_change_5d_pct"))
    estr_iqr = _safe_float(ecb.get("interquartile_range_pct"))
    estr_volume = _safe_float(ecb.get("total_volume_eur_millions"))
    estr_mean_volume = _safe_float(ecb.get("ten_observation_mean_volume_eur_millions"))
    volume_drop = _clamp01(1.0 - estr_volume / estr_mean_volume) if estr_volume is not None and estr_mean_volume and estr_mean_volume > 0.0 else None
    funding_pressure = _mean(
        (
            _clamp01(abs(estr_change or 0.0) / 0.25) if estr_change is not None else None,
            _clamp01(abs(estr_iqr or 0.0) / 0.25) if estr_iqr is not None else None,
            volume_drop,
        )
    )
    company_rows = company.get("symbol_rows") if isinstance(company.get("symbol_rows"), list) else []
    symbol_features = {
        str(row.get("symbol") or "").upper(): dict(row.get("features") or {})
        for row in company_rows
        if isinstance(row, Mapping) and str(row.get("symbol") or "") and isinstance(row.get("features"), Mapping)
    }
    profitability = _mean(_safe_float(row.get("fundamental_operating_profitability_norm")) for row in symbol_features.values())
    cash_flow_quality = _mean(_safe_float(row.get("fundamental_cash_flow_quality_norm")) for row in symbol_features.values())
    distress = _mean(_safe_float(row.get("fundamental_distress_norm")) for row in symbol_features.values())
    baseline_ok_count = sum(
        1 for source_id in BASELINE_SOURCE_IDS
        if isinstance(sources.get(source_id), Mapping) and sources[source_id].get("ok") is True
    )
    supplemental_ok_count = sum(
        1 for source_id in SUPPLEMENTAL_SOURCE_IDS
        if isinstance(sources.get(source_id), Mapping) and sources[source_id].get("ok") is True
    )
    repo_imbalance = _safe_float(nyfed_dealer.get("repo_imbalance_ratio"))
    financing_fails = _safe_float(nyfed_dealer.get("financing_fails_ratio"))
    treasury_inventory = _safe_float(nyfed_dealer.get("treasury_inventory_ratio"))
    corporate_inventory = _safe_float(nyfed_dealer.get("corporate_inventory_ratio"))
    noncurrent_loans = _safe_float(fdic_financials.get("noncurrent_loan_ratio"))
    deposit_funding = _safe_float(fdic_financials.get("deposit_to_asset_ratio"))
    lending_intensity = _safe_float(fdic_financials.get("loan_to_asset_ratio"))
    asset_growth_yoy = _safe_float(fdic_financials.get("asset_growth_yoy_ratio"))
    global_features = {
        "public_financial_context_available_norm": 1.0 if baseline_ok_count else 0.0,
        "public_financial_source_coverage_norm": _clamp01(baseline_ok_count / max(len(BASELINE_SOURCE_IDS), 1)),
        "public_financial_supplemental_coverage_norm": _clamp01(supplemental_ok_count / max(len(SUPPLEMENTAL_SOURCE_IDS), 1)),
        "ofr_financial_stress_norm": _stress_norm(ofr_latest.get("financial_stress")),
        "ofr_credit_stress_norm": _stress_norm(ofr_latest.get("credit")),
        "ofr_funding_stress_norm": _stress_norm(ofr_latest.get("funding")),
        "ofr_safe_asset_stress_norm": _stress_norm(ofr_latest.get("safe_assets")),
        "ofr_volatility_stress_norm": _stress_norm(ofr_latest.get("volatility")),
        "ofr_equity_valuation_stress_norm": _stress_norm(ofr_latest.get("equity_valuation")),
        "fdic_failure_12m_norm": _clamp01(float(fdic.get("failures_12m", 0) or 0) / 8.0) if fdic.get("ok") else None,
        "fdic_failure_assets_12m_norm": _clamp01(math.log1p(float(fdic.get("failed_assets_12m_thousands_usd", 0.0) or 0.0)) / math.log1p(500_000_000.0)) if fdic.get("ok") else None,
        "federal_register_financial_activity_norm": _clamp01(float(federal_register.get("document_count_7d", 0) or 0) / 75.0) if federal_register.get("ok") else None,
        "federal_register_high_impact_norm": _clamp01(float(federal_register.get("high_impact_count_7d", 0) or 0) / 12.0) if federal_register.get("ok") else None,
        "ecb_estr_funding_pressure_norm": funding_pressure,
        "ecb_estr_rate_norm": _clamp01((estr_rate or 0.0) / 8.0) if estr_rate is not None else None,
        "ecb_estr_change_5d_norm": _clamp01(0.5 + (estr_change or 0.0) / 0.5) if estr_change is not None else None,
        "companyfacts_coverage_norm": _safe_float(company.get("symbol_coverage_norm")),
        "companyfacts_operating_profitability_norm": profitability,
        "companyfacts_cash_flow_quality_norm": cash_flow_quality,
        "companyfacts_distress_norm": distress,
        "nyfed_dealer_repo_imbalance_norm": _signed_norm(repo_imbalance, 0.25) if nyfed_dealer.get("ok") and repo_imbalance is not None else None,
        "nyfed_dealer_financing_fails_pressure_norm": _clamp01(financing_fails / 0.10) if nyfed_dealer.get("ok") and financing_fails is not None else None,
        "nyfed_dealer_treasury_inventory_pressure_norm": _clamp01(treasury_inventory / 0.20) if nyfed_dealer.get("ok") and treasury_inventory is not None else None,
        "nyfed_dealer_corporate_inventory_pressure_norm": _clamp01(corporate_inventory / 0.05) if nyfed_dealer.get("ok") and corporate_inventory is not None else None,
        "fdic_bank_noncurrent_loan_pressure_norm": _clamp01(noncurrent_loans / 0.05) if fdic_financials.get("ok") and noncurrent_loans is not None else None,
        "fdic_bank_deposit_funding_norm": _clamp01(deposit_funding / 0.90) if fdic_financials.get("ok") and deposit_funding is not None else None,
        "fdic_bank_lending_intensity_norm": _clamp01(lending_intensity / 0.80) if fdic_financials.get("ok") and lending_intensity is not None else None,
        "fdic_bank_asset_growth_yoy_norm": _signed_norm(asset_growth_yoy, 0.10) if fdic_financials.get("ok") and asset_growth_yoy is not None else None,
    }
    global_features = {key: round(value, 8) for key, value in global_features.items() if value is not None}
    lineage: dict[str, list[dict[str, Any]]] = {}
    source_by_prefix = {
        "ofr_": "ofr_financial_stress_index",
        "fdic_bank_": "fdic_bank_financials",
        "fdic_": "fdic_bank_failures",
        "federal_register_": "federal_register_financial_rules",
        "ecb_": "ecb_estr",
        "companyfacts_": "sec_companyfacts",
        "nyfed_dealer_": "nyfed_primary_dealer_statistics",
    }
    for feature_key in global_features:
        source_id = next((value for prefix, value in source_by_prefix.items() if feature_key.startswith(prefix)), "public_financial_context")
        source = sources.get(source_id) if isinstance(sources.get(source_id), Mapping) else {}
        lineage[feature_key] = [{
            "source_id": source_id,
            "publisher": source.get("publisher", "derived public financial context"),
            "observation_time": source.get("observation_time"),
            "transport_receipt_sha256": ((source.get("transport") or {}).get("transport_receipt_sha256") if isinstance(source.get("transport"), Mapping) else None),
            "point_in_time_valid": True,
        }]
    return {
        "global_features": global_features,
        "symbol_features": symbol_features,
        "feature_lineage": lineage,
    }


def _capability_readiness(sources: Mapping[str, Any], derived: Mapping[str, Any]) -> dict[str, bool]:
    global_features = derived.get("global_features") if isinstance(derived.get("global_features"), Mapping) else {}
    company = sources.get("sec_companyfacts") if isinstance(sources.get("sec_companyfacts"), Mapping) else {}
    company_ready = bool(
        company.get("ok")
        and _safe_float(company.get("symbol_coverage_norm")) is not None
        and float(company.get("symbol_coverage_norm")) >= 0.5
        and float(company.get("average_statement_coverage_norm", 0.0) or 0.0) >= 0.5
    )
    readiness: dict[str, bool] = {}
    for capability_id, source_ids in CAPABILITY_SOURCE_REQUIREMENTS.items():
        source_ready = all(bool((sources.get(source_id) or {}).get("ok", False)) for source_id in source_ids)
        if capability_id in {"financial_statements", "quality_ratios", "operating_profitability", "cash_flow_quality", "leverage_state", "distress_indicators"}:
            source_ready = source_ready and company_ready
        required_features = CAPABILITY_FEATURE_REQUIREMENTS.get(capability_id, ())
        feature_ready = all(feature_key in global_features for feature_key in required_features)
        readiness[capability_id] = bool(source_ready and feature_ready)
    readiness["funding_stress"] = readiness["funding_stress"] and "ofr_funding_stress_norm" in global_features and "ecb_estr_funding_pressure_norm" in global_features
    readiness["default_risk"] = readiness["default_risk"] and "ofr_credit_stress_norm" in global_features and "fdic_failure_12m_norm" in global_features
    return readiness


def _health_source_rows(sources: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for source_id in SOURCE_CONTRACTS:
        source = sources.get(source_id) if isinstance(sources.get(source_id), Mapping) else {}
        rows[source_id] = {
            "ok": source.get("ok") is True,
            "contract_participates": source_id in BASELINE_SOURCE_IDS,
            "supplemental": source_id in SUPPLEMENTAL_SOURCE_IDS,
            "publisher": source.get("publisher") or SOURCE_CONTRACTS[source_id].get("publisher"),
            "observation_time": source.get("observation_time"),
            "observation_age_days": source.get("observation_age_days"),
            "isolated_failure": bool(source.get("isolated_failure", False)),
            "error": source.get("error"),
        }
    return rows


def collect_public_financial_context(
    *,
    symbols: list[str],
    user_agent: str,
    timeout: float,
    pause_seconds: float,
    now_utc: datetime | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    now = now_utc or _utc_now()
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    now = now.astimezone(timezone.utc)
    sources = {
        "ofr_financial_stress_index": _isolated_source(
            "ofr_financial_stress_index",
            lambda: _fetch_ofr(now=now, user_agent=user_agent, timeout=timeout),
        ),
        "fdic_bank_failures": _isolated_source(
            "fdic_bank_failures",
            lambda: _fetch_fdic(now=now, user_agent=user_agent, timeout=timeout),
        ),
        "federal_register_financial_rules": _isolated_source(
            "federal_register_financial_rules",
            lambda: _fetch_federal_register(now=now, user_agent=user_agent, timeout=timeout),
        ),
        "ecb_estr": _isolated_source(
            "ecb_estr",
            lambda: _fetch_ecb(now=now, user_agent=user_agent, timeout=timeout),
        ),
        "sec_companyfacts": _isolated_source(
            "sec_companyfacts",
            lambda: _fetch_companyfacts(
                now=now,
                symbols=symbols,
                user_agent=user_agent,
                timeout=timeout,
                pause_seconds=pause_seconds,
            ),
        ),
        "nyfed_primary_dealer_statistics": _isolated_source(
            "nyfed_primary_dealer_statistics",
            lambda: _fetch_nyfed_primary_dealer(now=now, user_agent=user_agent, timeout=timeout),
        ),
        "fdic_bank_financials": _isolated_source(
            "fdic_bank_financials",
            lambda: _fetch_fdic_financials(now=now, user_agent=user_agent, timeout=timeout),
        ),
    }
    taxonomy = _load_routing_taxonomy()
    economic_registry, economic_registry_validation = _economic_registry_validation()
    derived, taxonomy_validation = _apply_routing_taxonomy(_build_derived(sources), taxonomy)
    capability_readiness = _capability_readiness(sources, derived)
    ok_source_count = sum(1 for source in sources.values() if source.get("ok") is True)
    baseline_ok_source_count = sum(1 for source_id in BASELINE_SOURCE_IDS if sources[source_id].get("ok") is True)
    supplemental_ok_source_count = sum(1 for source_id in SUPPLEMENTAL_SOURCE_IDS if sources[source_id].get("ok") is True)
    ready_capability_count = sum(1 for ready in capability_readiness.values() if ready)
    ok = bool(
        baseline_ok_source_count >= 2
        and ready_capability_count >= 1
        and taxonomy_validation["ok"]
        and economic_registry_validation.get("ok")
    )
    status = {
        "timestamp_utc": now.isoformat(),
        "provider": "public_financial_context",
        "ok": ok,
        "overall_status": "ready" if ok and baseline_ok_source_count >= 4 else ("degraded" if ok else "failed"),
        "source_count": len(sources),
        "ok_source_count": ok_source_count,
        "baseline_source_count": len(BASELINE_SOURCE_IDS),
        "baseline_ok_source_count": baseline_ok_source_count,
        "supplemental_source_count": len(SUPPLEMENTAL_SOURCE_IDS),
        "supplemental_ok_source_count": supplemental_ok_source_count,
        "supplemental_source_failures_change_baseline_status": False,
        "sources": _health_source_rows(sources),
        "capability_count": len(capability_readiness),
        "ready_capability_count": ready_capability_count,
        "capability_readiness": capability_readiness,
        "taxonomy_validation": taxonomy_validation,
        "economic_source_registry_validation": economic_registry_validation,
        "optional_failure_is_soak_blocking": False,
        "paper_execution_authority": False,
        "live_execution_authority": False,
        "automatic_promotion_authority": False,
    }
    payload = {
        "timestamp_utc": now.isoformat(),
        "schema_version": 1,
        "provider": "public_financial_context",
        "status": status,
        "source_contracts": SOURCE_CONTRACTS,
        "sources": sources,
        "capability_readiness": capability_readiness,
        "derived": derived,
        "evidence_taxonomy": taxonomy,
        "economic_source_registry_id": economic_registry.get("registry_id"),
        "economic_source_registry_receipt_sha256": economic_registry_validation.get("registry_sha256"),
        "economic_source_registry_validation": economic_registry_validation,
        "taxonomy_validation": taxonomy_validation,
        "taxonomy_receipt_sha256": _canonical_hash(taxonomy),
        "routing_contract": {
            "paper_and_training_context_only": True,
            "missing_source_dimensions_are_omitted_not_zero_filled": True,
            "future_observations_rejected": True,
            "source_count_is_not_alpha": True,
            "symbol_features_require_symbol_level_coverage": True,
            "unclassified_feature_policy": "quarantine_from_bot_context",
            "paper_execution_authority": False,
            "live_execution_authority": False,
            "automatic_promotion_authority": False,
        },
    }
    payload["snapshot_receipt_sha256"] = _canonical_hash(
        {
            "timestamp_utc": payload["timestamp_utc"],
            "source_receipts": {
                source_id: ((source.get("transport") or {}).get("transport_receipt_sha256") if isinstance(source.get("transport"), Mapping) else None)
                for source_id, source in sources.items()
            },
            "capability_readiness": capability_readiness,
            "global_features": derived.get("global_features", {}),
            "symbol_features": derived.get("symbol_features", {}),
        }
    )
    status["snapshot_receipt_sha256"] = payload["snapshot_receipt_sha256"]
    return payload, status


def _parse_symbols(raw: str) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for token in str(raw or "").replace("\n", ",").split(","):
        symbol = token.strip().upper().replace(".", "-")
        if not symbol or symbol in seen or "/" in symbol or symbol.endswith("-USD"):
            continue
        seen.add(symbol)
        out.append(symbol)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Collect bounded official public financial-system and issuer context.")
    parser.add_argument("--symbols", default=str(os.getenv("PUBLIC_FINANCIAL_SYMBOLS") or ",".join(DEFAULT_SYMBOLS)))
    parser.add_argument("--max-symbols", type=int, default=int(os.getenv("PUBLIC_FINANCIAL_MAX_SYMBOLS", "12") or 12))
    parser.add_argument("--timeout", type=float, default=float(os.getenv("PUBLIC_FINANCIAL_TIMEOUT_SECONDS", "15") or 15))
    parser.add_argument("--pause-seconds", type=float, default=float(os.getenv("PUBLIC_FINANCIAL_SEC_PAUSE_SECONDS", "0.15") or 0.15))
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    symbols = (_parse_symbols(args.symbols) or list(DEFAULT_SYMBOLS))[: max(int(args.max_symbols), 1)]
    payload, status = collect_public_financial_context(
        symbols=symbols,
        user_agent=str(os.getenv("PUBLIC_FINANCIAL_USER_AGENT") or USER_AGENT_DEFAULT).strip() or USER_AGENT_DEFAULT,
        timeout=max(float(args.timeout), 1.0),
        pause_seconds=max(float(args.pause_seconds), 0.0),
    )
    write_payload(LATEST_PATH, payload)
    write_payload(HEALTH_PATH, status)
    if args.json:
        print(json.dumps(status, ensure_ascii=True))
    else:
        print(
            "public_financial_context status={status} sources={ready}/{total} capabilities={caps}/{cap_total}".format(
                status=status["overall_status"],
                ready=status["ok_source_count"],
                total=status["source_count"],
                caps=status["ready_capability_count"],
                cap_total=status["capability_count"],
            )
        )
    return 0 if status["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
