#!/usr/bin/env python3
import argparse
from http.client import RemoteDisconnected
import json
import math
import os
import xml.etree.ElementTree as ET
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Optional
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse
from urllib.request import Request, urlopen


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FRED_SERIES_ALIASES = {
    "GOLDAMGBD228NLBM": ["GOLDPMGBD228NLBM"],
}
CENTRAL_BANK_LIQUIDITY_SERIES = {
    "WALCL": "fed_total_assets",
    "WRESBAL": "fed_reserve_balances",
    "RRPONTSYD": "fed_overnight_reverse_repo",
    "RPONTSYD": "fed_overnight_repo",
    "WTREGEN": "treasury_general_account",
    "SWPT": "central_bank_liquidity_swaps",
    "SOFR": "secured_overnight_financing_rate",
    "EFFR": "effective_federal_funds_rate",
    "OBFR": "overnight_bank_funding_rate",
    "IORB": "interest_on_reserve_balances",
    "DFEDTARL": "federal_funds_target_lower",
    "DFEDTARU": "federal_funds_target_upper",
    "NFCI": "national_financial_conditions_index",
    "ANFCI": "adjusted_national_financial_conditions_index",
    "STLFSI4": "st_louis_financial_stress_index",
    "TREAST": "fed_treasury_securities_held",
    "WSHOMCB": "fed_mortgage_backed_securities_held",
    "BOGMBASE": "monetary_base",
    "M2SL": "m2_money_stock",
}
DEFAULT_FRED_SERIES_IDS = (
    "GDP,UNRATE,CPIAUCSL,DGS2,DGS5,DGS10,DGS30,DFII10,VIXCLS,DCOILWTICO,"
    "DTWEXBGS,BAMLH0A0HYM2," + ",".join(CENTRAL_BANK_LIQUIDITY_SERIES)
)
DEFAULT_CENTRAL_BANK_REQUIRED_SERIES = (
    "WALCL,WRESBAL,RRPONTSYD,RPONTSYD,WTREGEN,SWPT,SOFR,EFFR,IORB,NFCI,ANFCI,STLFSI4"
)
CENTRAL_BANK_SERIES_MAX_AGE_DAYS = {
    "WALCL": 10,
    "WRESBAL": 10,
    "RRPONTSYD": 7,
    "RPONTSYD": 7,
    "WTREGEN": 10,
    "SWPT": 10,
    "SOFR": 7,
    "EFFR": 7,
    "OBFR": 7,
    "IORB": 7,
    "DFEDTARL": 7,
    "DFEDTARU": 7,
    "NFCI": 12,
    "ANFCI": 12,
    "STLFSI4": 12,
    "TREAST": 10,
    "WSHOMCB": 10,
    "BOGMBASE": 75,
    "M2SL": 75,
}
PLACEHOLDER_API_KEYS = {
    "your_real_key",
    "your_api_key",
    "your_key",
    "replace_me",
    "changeme",
    "missing",
    "none",
    "null",
}


def _usable_api_key(value: Any) -> str:
    key = str(value or "").strip()
    normalized = key.lower().replace("-", "_").replace(" ", "_")
    if not key or normalized in PLACEHOLDER_API_KEYS or normalized.startswith("your_"):
        return ""
    return key


def _load_env_file(path: Path, *, override: bool = False) -> None:
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and (override or key not in os.environ):
            os.environ[key] = value


def _bootstrap_env() -> None:
    for path, override in [
        (PROJECT_ROOT / ".env", False),
        (PROJECT_ROOT / ".env.live", True),
        (PROJECT_ROOT / "config" / ".env", True),
        (PROJECT_ROOT / "config" / ".env.live", True),
        (PROJECT_ROOT / ".env.secrets.local", True),
        (PROJECT_ROOT / ".env.live.secrets.local", True),
        (PROJECT_ROOT / "config" / ".env.secrets.local", True),
        (PROJECT_ROOT / "config" / ".env.live.secrets.local", True),
    ]:
        _load_env_file(path, override=override)


def _sanitize_url(url: str) -> str:
    parsed = urlparse(url)
    query = []
    for k, v in parse_qsl(parsed.query, keep_blank_values=True):
        if k.lower() in {"key", "api_key", "registrationkey", "userid"}:
            query.append((k, "REDACTED"))
        else:
            query.append((k, v))
    return urlunparse((parsed.scheme, parsed.netloc, parsed.path, parsed.params, urlencode(query), parsed.fragment))


def _http_json(url: str, *, method: str = "GET", body: Optional[dict] = None, timeout: int = 25) -> Any:
    data = None
    headers = {"User-Agent": "schwab-trading-bot/1.0"}
    if body is not None:
        data = json.dumps(body).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = Request(url=url, data=data, method=method, headers=headers)
    with urlopen(req, timeout=timeout) as resp:
        payload = resp.read().decode("utf-8")
    return json.loads(payload)


def _http_text(url: str, *, timeout: int = 25) -> str:
    req = Request(url=url, method="GET", headers={"User-Agent": "schwab-trading-bot/1.0"})
    with urlopen(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8", errors="replace")


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")


def _to_float(value: Any) -> Optional[float]:
    try:
        num = float(value)
    except Exception:
        return None
    if not math.isfinite(num):
        return None
    return num


def _latest_numeric(rows: Any) -> Optional[float]:
    if not isinstance(rows, list):
        return None
    for row in rows:
        if not isinstance(row, dict):
            continue
        value = _to_float(row.get("value"))
        if value is not None:
            return value
    return None


def _parse_observation_date(value: Any) -> Optional[date]:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return date.fromisoformat(text[:10])
    except ValueError:
        return None


def _payload_as_of_date(payload: dict[str, Any]) -> date:
    raw = str(payload.get("timestamp_utc") or "").strip()
    if raw:
        try:
            parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
            return parsed.astimezone(timezone.utc).date()
        except ValueError:
            pass
    return datetime.now(timezone.utc).date()


def _numeric_observations(payload: Any, *, as_of_date: Optional[date] = None) -> list[tuple[str, float]]:
    rows = payload.get("observations") if isinstance(payload, dict) else []
    out: list[tuple[str, float]] = []
    for row in rows if isinstance(rows, list) else []:
        if not isinstance(row, dict):
            continue
        observation_date = _parse_observation_date(row.get("date"))
        if as_of_date is not None and (observation_date is None or observation_date > as_of_date):
            continue
        value = _to_float(row.get("value"))
        if value is not None:
            out.append((observation_date.isoformat() if observation_date is not None else "", value))
    return sorted(out, key=lambda item: item[0], reverse=True)


def _future_observation_dates(payload: Any, *, as_of_date: date) -> list[str]:
    rows = payload.get("observations") if isinstance(payload, dict) else []
    return sorted(
        {
            observation_date.isoformat()
            for row in rows if isinstance(rows, list) and isinstance(row, dict)
            if (observation_date := _parse_observation_date(row.get("date"))) is not None
            and observation_date > as_of_date
        }
    )


def _signed_norm(value: Any, scale: float) -> float:
    numeric = _to_float(value)
    if numeric is None:
        return 0.5
    return max(0.0, min(1.0, 0.5 + 0.5 * math.tanh(numeric / max(abs(float(scale)), 1e-9))))


def _level_norm(value: Any, scale: float) -> float:
    numeric = _to_float(value)
    if numeric is None:
        return 0.0
    return max(0.0, min(1.0, numeric / max(abs(float(scale)), 1e-9)))


def _derive_central_bank_liquidity_context(fred_payload: dict[str, Any]) -> dict[str, Any]:
    responses = fred_payload.get("responses") if isinstance(fred_payload.get("responses"), dict) else {}
    as_of_date = _payload_as_of_date(fred_payload)
    points = {
        series_id: _numeric_observations(responses.get(series_id), as_of_date=as_of_date)
        for series_id in CENTRAL_BANK_LIQUIDITY_SERIES
    }
    future_observations_excluded = {
        series_id: dates
        for series_id in CENTRAL_BANK_LIQUIDITY_SERIES
        if (dates := _future_observation_dates(responses.get(series_id), as_of_date=as_of_date))
    }

    def value(series_id: str, index: int = 0) -> Optional[float]:
        rows = points.get(series_id) or []
        return rows[index][1] if len(rows) > index else None

    def delta(series_id: str, index: int = 1) -> Optional[float]:
        current = value(series_id, 0)
        prior = value(series_id, index)
        return current - prior if current is not None and prior is not None else None

    total_assets = value("WALCL")
    reserves = value("WRESBAL")
    rrp_billions = value("RRPONTSYD")
    repo_billions = value("RPONTSYD")
    tga = value("WTREGEN")
    swaps = value("SWPT")
    rrp_millions = rrp_billions * 1000.0 if rrp_billions is not None else None
    repo_millions = repo_billions * 1000.0 if repo_billions is not None else None
    net_liquidity = (
        total_assets - tga - rrp_millions
        if total_assets is not None and tga is not None and rrp_millions is not None
        else None
    )
    prior_assets = value("WALCL", 1)
    prior_tga = value("WTREGEN", 1)
    prior_rrp = value("RRPONTSYD", 5)
    prior_net_liquidity = (
        prior_assets - prior_tga - (prior_rrp * 1000.0)
        if prior_assets is not None and prior_tga is not None and prior_rrp is not None
        else None
    )
    net_liquidity_change = (
        net_liquidity - prior_net_liquidity
        if net_liquidity is not None and prior_net_liquidity is not None
        else None
    )

    sofr = value("SOFR")
    effr = value("EFFR")
    obfr = value("OBFR")
    iorb = value("IORB")
    target_lower = value("DFEDTARL")
    target_upper = value("DFEDTARU")
    sofr_effr_bps = (sofr - effr) * 100.0 if sofr is not None and effr is not None else None
    effr_iorb_bps = (effr - iorb) * 100.0 if effr is not None and iorb is not None else None
    corridor_width_bps = (
        (target_upper - target_lower) * 100.0
        if target_upper is not None and target_lower is not None
        else None
    )

    nfci = value("NFCI")
    anfci = value("ANFCI")
    stlfsi = value("STLFSI4")
    funding_stress_inputs = [
        numeric
        for numeric in (
            (sofr_effr_bps / 10.0) if sofr_effr_bps is not None else None,
            nfci,
            anfci,
            (stlfsi / 2.0) if stlfsi is not None else None,
        )
        if numeric is not None
    ]
    funding_stress_raw = sum(funding_stress_inputs) / len(funding_stress_inputs) if funding_stress_inputs else None
    liquidity_impulse_norm = _signed_norm(net_liquidity_change, 100000.0)
    funding_stress_norm = _signed_norm(funding_stress_raw, 1.0)
    available = sorted(series_id for series_id, rows in points.items() if rows)
    required = [
        token.strip().upper()
        for token in os.getenv(
            "FRED_CENTRAL_BANK_REQUIRED_SERIES_IDS",
            DEFAULT_CENTRAL_BANK_REQUIRED_SERIES,
        ).split(",")
        if token.strip()
    ]
    latest_dates = {series_id: rows[0][0] for series_id, rows in points.items() if rows and rows[0][0]}
    latest_age_days = {
        series_id: (as_of_date - observation_date).days
        for series_id, raw_date in latest_dates.items()
        if (observation_date := _parse_observation_date(raw_date)) is not None
    }
    missing_required = sorted(series_id for series_id in required if series_id not in available)
    stale_required = sorted(
        series_id
        for series_id in required
        if series_id in latest_age_days
        and latest_age_days[series_id] > CENTRAL_BANK_SERIES_MAX_AGE_DAYS.get(series_id, 14)
    )
    unusable_required = sorted(set(missing_required).union(stale_required))
    coverage_ratio = float(len(required) - len(unusable_required)) / float(max(len(required), 1))
    availability_ratio = float(len(required) - len(missing_required)) / float(max(len(required), 1))
    fresh_series = sorted(
        series_id
        for series_id in available
        if latest_age_days.get(series_id, CENTRAL_BANK_SERIES_MAX_AGE_DAYS.get(series_id, 14) + 1)
        <= CENTRAL_BANK_SERIES_MAX_AGE_DAYS.get(series_id, 14)
    )
    global_features = {
        "central_bank_liquidity_available_norm": coverage_ratio,
        "central_bank_liquidity_source_coverage_norm": coverage_ratio,
        "fed_total_assets_level_norm": _level_norm(total_assets, 10000000.0),
        "fed_total_assets_impulse_norm": _signed_norm(delta("WALCL"), 100000.0),
        "fed_reserve_balances_level_norm": _level_norm(reserves, 5000000.0),
        "fed_reserve_balances_impulse_norm": _signed_norm(delta("WRESBAL"), 100000.0),
        "fed_rrp_drain_level_norm": _level_norm(rrp_billions, 2500.0),
        "fed_rrp_drain_impulse_norm": _signed_norm(delta("RRPONTSYD", 5), 100.0),
        "fed_repo_injection_level_norm": _level_norm(repo_billions, 500.0),
        "fed_tga_drain_level_norm": _level_norm(tga, 2000000.0),
        "fed_tga_drain_impulse_norm": _signed_norm(delta("WTREGEN"), 100000.0),
        "fed_net_liquidity_impulse_norm": liquidity_impulse_norm,
        "fed_liquidity_expansion_norm": liquidity_impulse_norm,
        "fed_liquidity_tightening_norm": 1.0 - liquidity_impulse_norm,
        "fed_central_bank_swap_usage_norm": _level_norm(swaps, 500000.0),
        "fed_sofr_level_norm": _level_norm(sofr, 10.0),
        "fed_effr_level_norm": _level_norm(effr, 10.0),
        "fed_iorb_level_norm": _level_norm(iorb, 10.0),
        "fed_sofr_effr_spread_norm": _signed_norm(sofr_effr_bps, 10.0),
        "fed_effr_iorb_spread_norm": _signed_norm(effr_iorb_bps, 10.0),
        "fed_policy_corridor_width_norm": _level_norm(corridor_width_bps, 100.0),
        "fed_funding_stress_norm": funding_stress_norm,
        "fed_financial_conditions_tightness_norm": _signed_norm(nfci, 1.0),
        "fed_adjusted_financial_conditions_tightness_norm": _signed_norm(anfci, 1.0),
        "fed_financial_stress_norm": _signed_norm(stlfsi, 2.0),
    }
    if net_liquidity_change is None:
        global_features["fed_net_liquidity_impulse_norm"] = 0.5
        global_features["fed_liquidity_expansion_norm"] = 0.5
        global_features["fed_liquidity_tightening_norm"] = 0.5
    return {
        "schema_version": 1,
        "timestamp_utc": fred_payload.get("timestamp_utc"),
        "provider": "fred_official_sources",
        "methodology": {
            "net_liquidity_proxy": "Fed total assets minus Treasury General Account minus overnight reverse repo usage",
            "classification": "heuristic_market_liquidity_proxy_not_official_accounting_identity",
            "unit_normalization": "H41 million-dollar series; New York Fed repo/reverse-repo billion-dollar series converted to millions",
            "point_in_time_only": True,
        },
        "coverage": {
            "as_of_date": as_of_date.isoformat(),
            "required_series": required,
            "available_series": available,
            "fresh_series": fresh_series,
            "missing_required_series": missing_required,
            "stale_required_series": stale_required,
            "unusable_required_series": unusable_required,
            "required_availability_ratio": availability_ratio,
            "required_coverage_ratio": coverage_ratio,
            "latest_observation_dates": latest_dates,
            "latest_observation_age_days": latest_age_days,
            "max_age_days_by_series": CENTRAL_BANK_SERIES_MAX_AGE_DAYS,
            "future_observations_excluded": future_observations_excluded,
            "future_observation_selected": False,
        },
        "balance_sheet": {
            "fed_total_assets_millions": total_assets,
            "fed_reserve_balances_millions": reserves,
            "treasury_general_account_millions": tga,
            "overnight_reverse_repo_millions": rrp_millions,
            "overnight_repo_millions": repo_millions,
            "central_bank_liquidity_swaps_millions": swaps,
            "fed_treasury_securities_held_millions": value("TREAST"),
            "fed_mbs_held_millions": value("WSHOMCB"),
            "net_liquidity_proxy_millions": net_liquidity,
            "net_liquidity_proxy_change_millions": net_liquidity_change,
        },
        "funding_rates": {
            "sofr_percent": sofr,
            "effr_percent": effr,
            "obfr_percent": obfr,
            "iorb_percent": iorb,
            "target_lower_percent": target_lower,
            "target_upper_percent": target_upper,
            "sofr_minus_effr_bps": sofr_effr_bps,
            "effr_minus_iorb_bps": effr_iorb_bps,
            "policy_corridor_width_bps": corridor_width_bps,
        },
        "financial_conditions": {
            "nfci": nfci,
            "adjusted_nfci": anfci,
            "st_louis_financial_stress": stlfsi,
        },
        "money_stock": {
            "monetary_base_millions": value("BOGMBASE"),
            "m2_billions": value("M2SL"),
        },
        "regime": {
            "liquidity": "expanding" if liquidity_impulse_norm >= 0.6 else ("tightening" if liquidity_impulse_norm <= 0.4 else "neutral"),
            "funding": "stressed" if funding_stress_norm >= 0.65 else ("easy" if funding_stress_norm <= 0.35 else "normal"),
        },
        "global_features": global_features,
    }


def _fred_csv_to_payload(text: str, *, series_id: str, limit: int) -> dict[str, Any]:
    observations: list[dict[str, str]] = []
    for raw in reversed(str(text or "").splitlines()):
        line = raw.strip()
        if not line or line.lower().startswith("observation_date"):
            continue
        if "," not in line:
            continue
        date, value = line.split(",", 1)
        value = value.strip()
        if _to_float(value) is None:
            continue
        observations.append({"date": date.strip(), "value": value})
        if len(observations) >= max(int(limit), 1):
            break
    return {
        "realtime_start": "",
        "realtime_end": "",
        "observation_start": "",
        "observation_end": "",
        "units": "",
        "output_type": 1,
        "file_type": "csv_public_graph_fallback",
        "order_by": "observation_date",
        "sort_order": "desc",
        "count": len(observations),
        "offset": 0,
        "limit": int(limit),
        "series_id": series_id,
        "observations": observations,
    }


def _bea_rss_payload(text: str) -> dict[str, Any]:
    try:
        root = ET.fromstring(text)
    except Exception:
        return {"items": []}
    items = []
    for item in root.findall(".//item"):
        title = (item.findtext("title") or "").strip()
        link = (item.findtext("link") or "").strip()
        pub_date = (item.findtext("pubDate") or "").strip()
        if title or link:
            items.append({"title": title, "link": link, "published": pub_date})
    return {"items": items}


def _is_static_census_dataset(dataset: str) -> bool:
    head = str(dataset or "").split("/", 1)[0]
    return head.isdigit()


def _cached_static_census_payload(
    census_root: Path,
    *,
    census_dataset: str,
    census_get: str,
    census_for: str,
) -> dict[str, Any] | None:
    if not _is_static_census_dataset(census_dataset):
        return None
    path = census_root / "latest.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    request = payload.get("request") if isinstance(payload.get("request"), dict) else {}
    if request.get("dataset") != census_dataset:
        return None
    if request.get("get") != census_get or request.get("for") != census_for:
        return None
    response = payload.get("response")
    if not isinstance(response, list) or len(response) < 2:
        return None
    return payload


def _merge_mapping(base: Any, overlay: Any) -> dict:
    out = dict(base) if isinstance(base, dict) else {}
    if isinstance(overlay, dict):
        for key, value in overlay.items():
            if isinstance(value, dict) and isinstance(out.get(key), dict):
                nested = dict(out[key])
                nested.update(value)
                out[key] = nested
            else:
                out[key] = value
    return out


def _derive_fred_macro_context(fred_payload: dict[str, Any]) -> dict[str, Any]:
    responses = fred_payload.get("responses") if isinstance(fred_payload.get("responses"), dict) else {}
    latest = {
        series_id: _latest_numeric((payload or {}).get("observations"))
        for series_id, payload in responses.items()
        if isinstance(payload, dict)
    }

    treasury_yields = {
        "2y": latest.get("DGS2"),
        "5y": latest.get("DGS5"),
        "10y": latest.get("DGS10"),
        "30y": latest.get("DGS30"),
        "real_10y": latest.get("DFII10"),
    }
    treasury_yields = {k: float(v) for k, v in treasury_yields.items() if v is not None}

    gold_fix = latest.get("GOLDAMGBD228NLBM")
    if gold_fix is None:
        gold_fix = latest.get("GOLDPMGBD228NLBM")

    cross_asset = {
        "vix": latest.get("VIXCLS"),
        "dollar_index_broad": latest.get("DTWEXBGS"),
        "gold_fix": gold_fix,
        "wti_spot": latest.get("DCOILWTICO"),
        "high_yield_oas_bps": latest.get("BAMLH0A0HYM2"),
    }
    cross_asset = {k: float(v) for k, v in cross_asset.items() if v is not None}

    bond_reference_overlay = {
        "timestamp_utc": fred_payload.get("timestamp_utc"),
        "provider": "fred",
        "treasury_yields": treasury_yields,
    }
    if "high_yield_oas_bps" in cross_asset:
        bond_reference_overlay["credit_spread_bps"] = float(cross_asset["high_yield_oas_bps"])

    central_bank_liquidity = _derive_central_bank_liquidity_context(fred_payload)

    return {
        "timestamp_utc": fred_payload.get("timestamp_utc"),
        "provider": "fred",
        "treasury_yields": treasury_yields,
        "cross_asset": cross_asset,
        "bond_reference_overlay": bond_reference_overlay,
        "central_bank_liquidity": central_bank_liquidity,
        "global_features": central_bank_liquidity.get("global_features", {}),
    }


def collect(args: argparse.Namespace) -> int:
    _bootstrap_env()

    now = datetime.now(timezone.utc)
    stamp = now.strftime("%Y%m%d_%H%M%S")
    year_now = now.year
    start_year = str(max(year_now - 1, 2000))
    end_year = str(year_now)

    bls_series = [s.strip() for s in (os.getenv("BLS_SERIES_IDS", "CUUR0000SA0,LNS14000000")).split(",") if s.strip()]
    bls_key = _usable_api_key(os.getenv("BLS_API_KEY", ""))

    census_key = _usable_api_key(os.getenv("CENSUS_API_KEY", ""))
    census_dataset = os.getenv("CENSUS_DATASET", "2023/acs/acs5")
    census_get = os.getenv("CENSUS_GET_VARS", "NAME,B01001_001E")
    census_for = os.getenv("CENSUS_FOR", "us:1")

    fred_key = _usable_api_key(os.getenv("FRED_API_KEY", ""))
    fred_series = [
        s.strip()
        for s in (
            os.getenv(
                "FRED_SERIES_IDS",
                DEFAULT_FRED_SERIES_IDS,
            )
        ).split(",")
        if s.strip()
    ]
    fred_required = {
        s.strip().upper()
        for s in (os.getenv("FRED_REQUIRED_SERIES_IDS", "GDP,UNRATE,CPIAUCSL")).split(",")
        if s.strip()
    }
    fred_limit = max(int(os.getenv("FRED_LIMIT", "64")), 1)

    bea_key = _usable_api_key(os.getenv("BEA_API_KEY", ""))

    out_root = PROJECT_ROOT / "exports" / "external_feeds"
    bls_root = out_root / "bls"
    census_root = out_root / "census"
    fred_root = out_root / "fred"
    bea_root = out_root / "bea"

    status = {
        "timestamp_utc": now.isoformat(),
        "bls": {"ok": False, "error": None, "url": None, "series_count": len(bls_series)},
        "census": {"ok": False, "error": None, "url": None, "dataset": census_dataset},
        "fred": {"ok": False, "error": None, "url": None, "series_count": len(fred_series), "limit": fred_limit},
        "bea": {"ok": False, "error": None, "url": None, "dataset_count": 0},
    }

    # BLS
    bls_url = "https://api.bls.gov/publicAPI/v2/timeseries/data/"
    bls_body = {"seriesid": bls_series, "startyear": start_year, "endyear": end_year}
    if bls_key:
        bls_body["registrationkey"] = bls_key
    status["bls"]["url"] = bls_url
    try:
        bls_resp = _http_json(bls_url, method="POST", body=bls_body)
        bls_ok = isinstance(bls_resp, dict) and str(bls_resp.get("status", "")).upper() == "REQUEST_SUCCEEDED"
        status["bls"]["ok"] = bool(bls_ok)
        if not bls_ok:
            status["bls"]["error"] = str((bls_resp or {}).get("message") if isinstance(bls_resp, dict) else "request_failed")
        if not args.test_only:
            bls_payload = {
                "timestamp_utc": now.isoformat(),
                "request": {"seriesid": bls_series, "startyear": start_year, "endyear": end_year, "key_used": bool(bls_key)},
                "response": bls_resp,
            }
            _write_json(bls_root / f"bls_{stamp}.json", bls_payload)
            _write_json(bls_root / "latest.json", bls_payload)
    except (HTTPError, URLError, TimeoutError, ValueError, RemoteDisconnected, OSError) as exc:
        status["bls"]["error"] = str(exc)

    # Census
    census_base_params = {"get": census_get, "for": census_for}
    census_urls = []
    if census_key:
        census_urls.append(
            (
                "api_key",
                f"https://api.census.gov/data/{census_dataset}?"
                + urlencode({**census_base_params, "key": census_key}),
            )
        )
    census_urls.append(("public_no_key", f"https://api.census.gov/data/{census_dataset}?" + urlencode(census_base_params)))
    status["census"]["url"] = _sanitize_url(census_urls[0][1])
    census_errors: list[str] = []
    census_payload: dict[str, Any] | None = None
    for mode, census_url in census_urls:
        try:
            census_resp = _http_json(census_url, method="GET")
            census_ok = isinstance(census_resp, list) and len(census_resp) >= 2
            if not census_ok:
                raise ValueError("unexpected_response_shape")
            status["census"]["ok"] = True
            status["census"]["fallback"] = mode if mode != "api_key" else None
            census_payload = {
                "timestamp_utc": now.isoformat(),
                "request": {
                    "dataset": census_dataset,
                    "get": census_get,
                    "for": census_for,
                    "url": _sanitize_url(census_url),
                    "mode": mode,
                },
                "response": census_resp,
            }
            break
        except (HTTPError, URLError, TimeoutError, ValueError, RemoteDisconnected, OSError) as exc:
            census_errors.append(f"{mode}:{exc}")
    if not status["census"]["ok"]:
        live_error = "; ".join(census_errors)
        cached_payload = _cached_static_census_payload(
            census_root,
            census_dataset=census_dataset,
            census_get=census_get,
            census_for=census_for,
        )
        if cached_payload is not None:
            status["census"]["ok"] = True
            status["census"]["fallback"] = "cached_static_snapshot"
            status["census"]["cache_timestamp_utc"] = cached_payload.get("timestamp_utc")
            status["census"]["live_error"] = live_error
        else:
            status["census"]["error"] = live_error
    elif not args.test_only and census_payload is not None:
        _write_json(census_root / f"census_{stamp}.json", census_payload)
        _write_json(census_root / "latest.json", census_payload)

    # FRED
    fred_base_url = "https://api.stlouisfed.org/fred/series/observations"
    fred_csv_base_url = "https://fred.stlouisfed.org/graph/fredgraph.csv"
    if fred_series:
        sample_url = (
            fred_base_url
            + "?"
            + urlencode({"series_id": fred_series[0], "api_key": fred_key or "missing", "file_type": "json", "sort_order": "desc", "limit": fred_limit})
            if fred_key
            else fred_csv_base_url + "?" + urlencode({"id": fred_series[0]})
        )
        status["fred"]["url"] = _sanitize_url(sample_url)

    fred_collected: dict[str, Any] = {}
    fred_errors: list[str] = []
    fred_warnings: list[str] = []
    fred_aliases_used: dict[str, str] = {}
    fred_fallbacks_used: dict[str, str] = {}
    for series_id in fred_series:
        candidate_ids = [series_id, *FRED_SERIES_ALIASES.get(series_id, [])]
        candidate_errors: list[str] = []
        for candidate_id in candidate_ids:
            if fred_key:
                fred_url = fred_base_url + "?" + urlencode(
                    {"series_id": candidate_id, "api_key": fred_key, "file_type": "json", "sort_order": "desc", "limit": fred_limit}
                )
                try:
                    resp = _http_json(fred_url, method="GET")
                    if not isinstance(resp, dict):
                        raise ValueError(f"unexpected_response_shape series_id={candidate_id}")
                    if candidate_id != series_id:
                        resp = dict(resp)
                        resp["series_id_requested"] = series_id
                        resp["series_id_resolved"] = candidate_id
                        fred_aliases_used[series_id] = candidate_id
                    fred_collected[series_id] = resp
                    break
                except (HTTPError, URLError, TimeoutError, ValueError, RemoteDisconnected, OSError) as exc:
                    candidate_errors.append(f"{candidate_id}:{exc}")
            if series_id in fred_collected:
                break
            csv_url = fred_csv_base_url + "?" + urlencode({"id": candidate_id})
            try:
                csv_text = _http_text(csv_url)
                resp = _fred_csv_to_payload(csv_text, series_id=candidate_id, limit=fred_limit)
                if not resp.get("observations"):
                    raise ValueError(f"empty_public_csv series_id={candidate_id}")
                if candidate_id != series_id:
                    resp = dict(resp)
                    resp["series_id_requested"] = series_id
                    resp["series_id_resolved"] = candidate_id
                    fred_aliases_used[series_id] = candidate_id
                fred_collected[series_id] = resp
                fred_fallbacks_used[series_id] = "public_csv"
                break
            except (HTTPError, URLError, TimeoutError, ValueError, RemoteDisconnected, OSError) as exc:
                candidate_errors.append(f"{candidate_id}:public_csv:{exc}")
        if series_id not in fred_collected:
            message = f"series_id={series_id} error={' | '.join(candidate_errors)}"
            if series_id in fred_required:
                fred_errors.append(message)
            else:
                fred_warnings.append(message)

    fred_payload = {
        "timestamp_utc": now.isoformat(),
        "request": {"series_ids": fred_series, "limit": fred_limit},
        "responses": fred_collected,
    }
    macro_context = _derive_fred_macro_context(fred_payload)
    central_context = (
        macro_context.get("central_bank_liquidity")
        if isinstance(macro_context.get("central_bank_liquidity"), dict)
        else {}
    )
    central_context_coverage = (
        central_context.get("coverage")
        if isinstance(central_context.get("coverage"), dict)
        else {}
    )
    central_required = {
        token.strip().upper()
        for token in os.getenv(
            "FRED_CENTRAL_BANK_REQUIRED_SERIES_IDS",
            DEFAULT_CENTRAL_BANK_REQUIRED_SERIES,
        ).split(",")
        if token.strip()
    }
    central_available = central_required.intersection(
        str(item).upper() for item in central_context_coverage.get("available_series", [])
    )
    central_missing = sorted(str(item) for item in central_context_coverage.get("missing_required_series", []))
    central_stale = sorted(str(item) for item in central_context_coverage.get("stale_required_series", []))
    central_unusable = sorted(str(item) for item in central_context_coverage.get("unusable_required_series", []))
    central_coverage = float(central_context_coverage.get("required_coverage_ratio", 0.0) or 0.0)
    central_min_coverage = max(
        0.0,
        min(1.0, float(os.getenv("FRED_CENTRAL_BANK_MIN_REQUIRED_COVERAGE", "1.0"))),
    )
    central_ok = bool(central_coverage >= central_min_coverage)
    status["fred"]["central_bank_liquidity"] = {
        "ok": central_ok,
        "required_series": sorted(central_required),
        "available_series": sorted(central_available),
        "missing_required_series": central_missing,
        "stale_required_series": central_stale,
        "unusable_required_series": central_unusable,
        "required_coverage_ratio": round(central_coverage, 6),
        "minimum_required_coverage_ratio": central_min_coverage,
        "as_of_date": central_context_coverage.get("as_of_date"),
        "latest_observation_dates": central_context_coverage.get("latest_observation_dates", {}),
        "latest_observation_age_days": central_context_coverage.get("latest_observation_age_days", {}),
        "future_observations_excluded": central_context_coverage.get("future_observations_excluded", {}),
        "future_observation_selected": bool(central_context_coverage.get("future_observation_selected", False)),
        "fail_visible": True,
    }
    if not central_ok:
        fred_errors.append(
            "central_bank_liquidity_coverage_below_contract:"
            f"coverage={central_coverage:.6f}:unusable={','.join(central_unusable)}"
        )
    fred_ok = bool(all(series_id in fred_collected for series_id in fred_required) and central_ok)
    status["fred"]["ok"] = fred_ok
    if fred_aliases_used:
        status["fred"]["aliases_used"] = fred_aliases_used
    if fred_fallbacks_used:
        status["fred"]["fallbacks_used"] = fred_fallbacks_used
    if fred_errors:
        status["fred"]["error"] = "; ".join(fred_errors)
    if fred_warnings:
        status["fred"]["warnings"] = fred_warnings

    if not args.test_only:
        _write_json(fred_root / f"fred_{stamp}.json", fred_payload)
        _write_json(fred_root / "latest.json", fred_payload)

        external_context_root = PROJECT_ROOT / "exports" / "external_context"
        if macro_context:
            _write_json(external_context_root / "macro_cross_asset_latest.json", macro_context)
            if isinstance(central_context, dict):
                _write_json(external_context_root / "central_bank_liquidity_latest.json", central_context)
            existing_bond_reference_path = external_context_root / "bond_reference_latest.json"
            existing_bond_reference: dict[str, Any] = {}
            if existing_bond_reference_path.exists():
                try:
                    existing_bond_reference = json.loads(existing_bond_reference_path.read_text(encoding="utf-8"))
                except Exception:
                    existing_bond_reference = {}
            merged_bond_reference = _merge_mapping(existing_bond_reference, macro_context.get("bond_reference_overlay"))
            _write_json(existing_bond_reference_path, merged_bond_reference)

    # BEA (dataset list metadata pull)
    if not bea_key:
        status["bea"]["error"] = "BEA_API_KEY missing in environment (.env/.env.live/.env.live.secrets.local)"
    else:
        bea_base_url = "https://apps.bea.gov/api/data"
        bea_url = bea_base_url + "?" + urlencode(
            {"UserID": bea_key, "method": "GETDATASETLIST", "ResultFormat": "JSON"}
        )
        status["bea"]["url"] = _sanitize_url(bea_url)
        try:
            bea_resp = _http_json(bea_url, method="GET")
            bea_api = bea_resp.get("BEAAPI", {}) if isinstance(bea_resp, dict) else {}
            datasets = ((bea_api.get("Results") or {}).get("Dataset") or []) if isinstance(bea_api, dict) else []
            bea_ok = isinstance(datasets, list) and len(datasets) > 0
            status["bea"]["ok"] = bool(bea_ok)
            status["bea"]["dataset_count"] = len(datasets) if isinstance(datasets, list) else 0
            if not bea_ok:
                status["bea"]["error"] = "unexpected_response_shape"
            if not args.test_only:
                bea_payload = {
                    "timestamp_utc": now.isoformat(),
                    "request": {"method": "GETDATASETLIST", "url": _sanitize_url(bea_url)},
                    "response": bea_resp,
                }
                _write_json(bea_root / f"bea_{stamp}.json", bea_payload)
                _write_json(bea_root / "latest.json", bea_payload)
        except (HTTPError, URLError, TimeoutError, ValueError, RemoteDisconnected, OSError) as exc:
            status["bea"]["error"] = str(exc)
    if not status["bea"]["ok"]:
        rss_url = "https://apps.bea.gov/rss/rss.xml"
        try:
            rss_text = _http_text(rss_url)
            rss_payload = _bea_rss_payload(rss_text)
            items = rss_payload.get("items") if isinstance(rss_payload.get("items"), list) else []
            if not items:
                raise ValueError("empty_bea_rss")
            status["bea"]["ok"] = True
            status["bea"]["fallback"] = "rss"
            status["bea"]["dataset_count"] = len(items)
            status["bea"]["url"] = rss_url
            status["bea"]["error"] = None
            if not args.test_only:
                bea_payload = {
                    "timestamp_utc": now.isoformat(),
                    "request": {"method": "RSS", "url": rss_url},
                    "response": rss_payload,
                }
                _write_json(bea_root / f"bea_{stamp}.json", bea_payload)
                _write_json(bea_root / "latest.json", bea_payload)
        except (HTTPError, URLError, TimeoutError, ValueError, RemoteDisconnected, OSError) as exc:
            if status["bea"]["error"]:
                status["bea"]["error"] = f"{status['bea']['error']}; rss_fallback:{exc}"
            else:
                status["bea"]["error"] = f"rss_fallback:{exc}"

    _write_json(out_root / "latest_status.json", status)

    print(
        f"bls_ok={status['bls']['ok']} census_ok={status['census']['ok']} "
        f"fred_ok={status['fred']['ok']} bea_ok={status['bea']['ok']}"
    )
    print(f"status_file={out_root / 'latest_status.json'}")
    if not args.test_only:
        print(f"bls_latest={bls_root / 'latest.json'}")
        print(f"census_latest={census_root / 'latest.json'}")
        print(f"fred_latest={fred_root / 'latest.json'}")
        print(f"bea_latest={bea_root / 'latest.json'}")

    return 0 if all(bool(status[name]["ok"]) for name in ("bls", "census", "fred", "bea")) else 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Collect BLS + Census + FRED + BEA snapshots for ingestion.")
    parser.add_argument("--test-only", action="store_true", help="Connectivity check only; do not write provider snapshots.")
    args = parser.parse_args()
    return collect(args)


if __name__ == "__main__":
    raise SystemExit(main())
