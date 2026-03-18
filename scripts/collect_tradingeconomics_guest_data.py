#!/usr/bin/env python3
import argparse
import json
import math
import os
import statistics
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qsl, quote, urlencode, urlparse, urlunparse
from urllib.request import Request, urlopen

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from core.derivatives_features import summarize_calendar_payload
    from core.market_context_features import summarize_structured_news_items
except Exception:
    from core.derivatives_features import summarize_calendar_payload
    from core.market_context_features import summarize_structured_news_items

DEFAULT_COUNTRIES = "United States,Euro Area,China,Japan,United Kingdom,Canada"
DEFAULT_MARKET_SYMBOLS = (
    "spy:us,qqq:us,iwm:us,dia:us,tlt:us,ief:us,shy:us,tip:us,hyg:us,lqd:us,"
    "coin:us,mstr:us,tsla:us,nvda:us,btcusd:cur,ethusd:cur,gld:com"
)
DEFAULT_AUTH = "guest:guest"


def _load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def _bootstrap_env() -> None:
    _load_env_file(PROJECT_ROOT / ".env")
    _load_env_file(PROJECT_ROOT / "config" / ".env")
    _load_env_file(PROJECT_ROOT / "config" / ".env.live")


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        num = float(value)
        if math.isfinite(num):
            return num
    except Exception:
        pass
    return default


def _try_float(value: Any) -> Optional[float]:
    try:
        num = float(value)
        if math.isfinite(num):
            return num
    except Exception:
        pass
    return None


def _sanitize_url(url: str) -> str:
    parsed = urlparse(url)
    query = []
    for k, v in parse_qsl(parsed.query, keep_blank_values=True):
        if k.lower() in {"c", "client", "secret", "api_key", "key", "userid", "user"}:
            query.append((k, "REDACTED"))
        else:
            query.append((k, v))
    return urlunparse((parsed.scheme, parsed.netloc, parsed.path, parsed.params, urlencode(query), parsed.fragment))


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")


def _append_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=True) + "\n")


def _http_json(url: str, *, timeout: float = 25.0) -> Any:
    req = Request(
        url=url,
        headers={
            "Accept": "application/json",
            "User-Agent": "schwab-trading-bot/1.0",
        },
    )
    with urlopen(req, timeout=max(float(timeout), 1.0)) as resp:
        payload = resp.read().decode("utf-8", "replace")
    return json.loads(payload)


def _row_get_ci(row: Dict[str, Any], *keys: str) -> Any:
    if not isinstance(row, dict):
        return None
    lower_map = {str(k).lower(): v for k, v in row.items()}
    for key in keys:
        if key.lower() in lower_map:
            return lower_map[key.lower()]
    return None


def _rows_from_payload(payload: Any) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if isinstance(payload, list):
        rows.extend([row for row in payload if isinstance(row, dict)])
        return rows
    if not isinstance(payload, dict):
        return rows
    for key, value in payload.items():
        if str(key).lower() in {
            "data",
            "results",
            "calendar",
            "events",
            "items",
            "news",
            "markets",
            "earnings",
            "dividends",
            "forecasts",
            "forecast",
        } and isinstance(value, list):
            rows.extend([row for row in value if isinstance(row, dict)])
    if rows:
        return rows
    if any(isinstance(v, (str, int, float, bool)) for v in payload.values()):
        return [payload]
    return rows


def _item_text(row: Dict[str, Any]) -> str:
    chunks: List[str] = []
    for key in (
        "country",
        "category",
        "title",
        "headline",
        "name",
        "event",
        "symbol",
        "ticker",
        "description",
        "topic",
        "source",
    ):
        value = _row_get_ci(row, key)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            chunks.append(text.lower())
    return " ".join(chunks)


def _country_text(row: Dict[str, Any]) -> str:
    country = _row_get_ci(row, "country")
    return str(country or "").strip().lower()


def _value_from_row(row: Dict[str, Any]) -> Optional[float]:
    for key in (
        "latestvalue",
        "value",
        "actual",
        "close",
        "price",
        "last",
        "lastprice",
        "forecast",
        "yearend",
    ):
        value = _try_float(_row_get_ci(row, key))
        if value is not None:
            return value
    return None


def _fractional_pct(value: Any) -> Optional[float]:
    num = _try_float(value)
    if num is None:
        return None
    if abs(num) > 1.0:
        return num / 100.0
    return num


def _row_sort_key(row: Dict[str, Any]) -> Tuple[int, str]:
    date_raw = _row_get_ci(
        row,
        "datetime",
        "date",
        "dateutc",
        "latestvaluedate",
        "publisheddate",
        "published",
        "createdat",
    )
    if date_raw is None:
        return (1, "")
    return (0, str(date_raw))


def _best_matching_row(
    rows: Sequence[Dict[str, Any]],
    queries: Sequence[str],
    *,
    country: str = "",
) -> Optional[Dict[str, Any]]:
    country_token = str(country or "").strip().lower()
    best_score = -1
    best_row: Optional[Dict[str, Any]] = None
    for row in sorted(rows, key=_row_sort_key, reverse=True):
        text = _item_text(row)
        if not text:
            continue
        row_country = _country_text(row)
        if country_token and row_country and country_token not in row_country and row_country not in country_token:
            continue
        score = 0
        for query in queries:
            q = str(query or "").strip().lower()
            if not q:
                continue
            if q in text:
                score += max(2, len(q.split()))
                continue
            tokens = [tok for tok in q.split() if tok]
            if tokens and all(tok in text for tok in tokens):
                score += len(tokens)
        if score > best_score:
            best_score = score
            best_row = row
    return best_row if best_score > 0 else None


def _percent_to_ratio(value: Optional[float], row: Optional[Dict[str, Any]]) -> Optional[float]:
    if value is None:
        return None
    unit = str(_row_get_ci(row or {}, "unit") or "").strip().lower()
    if "%" in unit or "percent" in unit or "percentage" in unit:
        return value / 100.0
    if abs(value) >= 0.05:
        return value / 100.0
    return value


def _derive_macro_backfill(
    indicator_rows: Sequence[Dict[str, Any]],
    forecast_rows: Sequence[Dict[str, Any]],
    *,
    country: str,
) -> Dict[str, Any]:
    indicator_rows = list(indicator_rows or [])
    forecast_rows = list(forecast_rows or [])

    def _pick(queries: Sequence[str]) -> Tuple[Optional[float], Optional[Dict[str, Any]], str]:
        row = _best_matching_row(indicator_rows, queries, country=country)
        if row is not None:
            return _value_from_row(row), row, "indicators"
        row = _best_matching_row(forecast_rows, queries, country=country)
        if row is not None:
            return _value_from_row(row), row, "forecast"
        return None, None, ""

    unrate_value, unrate_row, unrate_source = _pick(("unemployment rate", "jobless rate"))
    infl_value, infl_row, infl_source = _pick(
        (
            "inflation rate mom",
            "inflation rate m/m",
            "consumer price index mom",
            "cpi mom",
            "core inflation rate mom",
            "inflation rate",
        )
    )
    gdp_value, gdp_row, gdp_source = _pick(
        (
            "gdp growth rate",
            "gdp growth",
            "gross domestic product growth rate",
            "gdp qoq",
        )
    )

    inflation_ratio = _percent_to_ratio(infl_value, infl_row)
    gdp_ratio = _percent_to_ratio(gdp_value, gdp_row)

    return {
        "country": country,
        "unemployment_rate_latest": unrate_value,
        "inflation_mom_ratio": inflation_ratio,
        "gdp_qoq_ratio": gdp_ratio,
        "unemployment_source": unrate_source,
        "inflation_source": infl_source,
        "gdp_source": gdp_source,
        "unemployment_title": str(_row_get_ci(unrate_row or {}, "title", "category") or ""),
        "inflation_title": str(_row_get_ci(infl_row or {}, "title", "category") or ""),
        "gdp_title": str(_row_get_ci(gdp_row or {}, "title", "category") or ""),
    }


def _derive_market_breadth(market_rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    pct_moves: List[float] = []
    weights: List[float] = []
    for row in market_rows:
        pct = None
        for key in ("changepercent", "percentchange", "pctchange", "change_pct", "changepercentage"):
            pct = _fractional_pct(_row_get_ci(row, key))
            if pct is not None:
                break
        if pct is None:
            change_value = _fractional_pct(_row_get_ci(row, "change"))
            close_value = _try_float(_row_get_ci(row, "close", "price", "last", "lastprice"))
            if change_value is not None and close_value and abs(close_value) > 1e-9 and abs(change_value) <= abs(close_value) * 0.75:
                pct = change_value / abs(close_value)
        if pct is None or not math.isfinite(pct):
            continue
        weight = _to_float(_row_get_ci(row, "volume", "turnover", "marketcap"), 0.0)
        pct_moves.append(pct)
        weights.append(weight if weight > 0.0 else (1.0 + abs(pct)))

    if not pct_moves:
        return {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "source": "tradingeconomics_guest",
            "row_count": 0,
        }

    advancers = float(sum(1 for pct in pct_moves if pct > 0.0))
    decliners = float(sum(1 for pct in pct_moves if pct < 0.0))
    up_volume = float(sum(w for pct, w in zip(pct_moves, weights) if pct > 0.0))
    down_volume = float(sum(w for pct, w in zip(pct_moves, weights) if pct < 0.0))
    new_highs = float(sum(1 for pct in pct_moves if pct >= 0.015))
    new_lows = float(sum(1 for pct in pct_moves if pct <= -0.015))
    sector_dispersion = float(statistics.pstdev(pct_moves)) if len(pct_moves) > 1 else 0.0

    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "source": "tradingeconomics_guest",
        "row_count": len(pct_moves),
        "advancers": advancers,
        "decliners": decliners,
        "up_volume": up_volume,
        "down_volume": down_volume,
        "new_highs": new_highs,
        "new_lows": new_lows,
        "sector_dispersion": sector_dispersion,
    }


def _derive_bond_reference(
    indicator_rows: Sequence[Dict[str, Any]],
    market_rows: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    combined_rows = list(indicator_rows or []) + list(market_rows or [])
    yield_queries = {
        "2y": ("2-year note yield", "2-year bond yield", "2 year yield", "2y yield"),
        "5y": ("5-year note yield", "5-year bond yield", "5 year yield", "5y yield"),
        "10y": ("10-year bond yield", "10-year note yield", "10 year yield", "10y yield"),
        "30y": ("30-year bond yield", "30-year treasury bond", "30 year yield", "30y yield"),
        "real_10y": ("real interest rate", "10-year tips", "10 year tips", "10-year real yield"),
    }
    treasury_yields: Dict[str, float] = {}
    for key, queries in yield_queries.items():
        row = _best_matching_row(combined_rows, queries, country="united states")
        value = _value_from_row(row or {})
        if value is None or value <= 0.0:
            continue
        treasury_yields[key] = value

    if not treasury_yields:
        return {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "source": "tradingeconomics_guest",
            "symbols": {},
        }

    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "source": "tradingeconomics_guest",
        "treasury_yields": treasury_yields,
        "symbols": {},
    }


def _endpoint_path(base: str, *, auth: str, params: Optional[Dict[str, Any]] = None) -> str:
    query = {"c": auth, "f": "json"}
    if params:
        query.update(params)
    return f"https://api.tradingeconomics.com{base}?{urlencode(query)}"


def _fetch_dataset(url: str, *, timeout: float) -> Tuple[Any, List[Dict[str, Any]], Optional[str]]:
    try:
        payload = _http_json(url, timeout=timeout)
    except HTTPError as exc:
        return None, [], f"HTTPError:{exc.code}:{getattr(exc, 'reason', exc)}"
    except URLError as exc:
        return None, [], f"URLError:{getattr(exc, 'reason', exc)}"
    except Exception as exc:
        return None, [], f"{type(exc).__name__}:{exc}"
    return payload, _rows_from_payload(payload), None


def _fetch_market_symbols(
    symbols: Sequence[str],
    *,
    auth: str,
    timeout: float,
) -> Tuple[List[Dict[str, Any]], List[str]]:
    rows: List[Dict[str, Any]] = []
    errors: List[str] = []
    chunk_size = 6
    clean_symbols = [str(symbol).strip() for symbol in symbols if str(symbol).strip()]
    for idx in range(0, len(clean_symbols), chunk_size):
        chunk = clean_symbols[idx : idx + chunk_size]
        url = _endpoint_path(
            f"/markets/symbol/{quote(','.join(chunk), safe=',:')}",
            auth=auth,
        )
        payload, chunk_rows, error = _fetch_dataset(url, timeout=timeout)
        if error:
            errors.append(f"chunk={','.join(chunk)} error={error}")
            continue
        if isinstance(payload, dict) and not chunk_rows:
            errors.append(f"chunk={','.join(chunk)} error=unexpected_response_shape")
            continue
        rows.extend(chunk_rows)
    return rows, errors


def _dataset_event_rows(
    *,
    now_iso: str,
    dataset: str,
    rows: Sequence[Dict[str, Any]],
    country: str,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for idx, row in enumerate(rows):
        out.append(
            {
                "timestamp_utc": now_iso,
                "provider": "tradingeconomics_guest",
                "dataset": dataset,
                "country": country,
                "row_index": idx,
                "row_key": str(
                    _row_get_ci(row, "calendarid", "eventid", "symbol", "ticker", "historicaldatasymbol", "title", "headline")
                    or idx
                ),
                "row": row,
            }
        )
    return out


def collect(args: argparse.Namespace) -> int:
    _bootstrap_env()

    now = datetime.now(timezone.utc)
    now_iso = now.isoformat()
    stamp = now.strftime("%Y%m%d_%H%M%S")
    day_stamp = now.strftime("%Y%m%d")
    countries = [token.strip() for token in str(args.countries or DEFAULT_COUNTRIES).split(",") if token.strip()]
    primary_country = countries[0] if countries else "United States"
    market_symbols = [token.strip() for token in str(args.market_symbols or DEFAULT_MARKET_SYMBOLS).split(",") if token.strip()]
    auth = str(args.auth or os.getenv("TRADINGECONOMICS_AUTH") or DEFAULT_AUTH).strip() or DEFAULT_AUTH
    auth_mode = "guest" if auth == DEFAULT_AUTH else "custom"
    start_date = now.date().isoformat()
    end_date = (now + timedelta(days=max(int(args.lookahead_days), 1))).date().isoformat()

    external_root = PROJECT_ROOT / "exports" / "external_feeds" / "tradingeconomics"
    health_root = PROJECT_ROOT / "governance" / "health"
    events_root = PROJECT_ROOT / "governance" / "events"
    data_rows_path = PROJECT_ROOT / "data" / "tradingeconomics" / f"tradingeconomics_guest_rows_{day_stamp}.jsonl"
    summary_jsonl_path = events_root / f"tradingeconomics_guest_sync_{day_stamp}.jsonl"
    external_context_root = PROJECT_ROOT / "data" / "external_context"

    dataset_status: Dict[str, Dict[str, Any]] = {}
    export_payloads: Dict[str, Dict[str, Any]] = {}
    row_archives: Dict[str, List[Dict[str, Any]]] = {}

    dataset_specs = [
        (
            "indicators",
            _endpoint_path(f"/country/{quote(','.join(countries), safe=',')}", auth=auth),
        ),
        (
            "forecast",
            _endpoint_path(f"/forecast/country/{quote(','.join(countries), safe=',')}", auth=auth),
        ),
        (
            "calendar",
            _endpoint_path("/calendar", auth=auth),
        ),
        (
            "calendar_country",
            _endpoint_path(f"/calendar/country/{quote(primary_country, safe='')}", auth=auth),
        ),
        (
            "news",
            _endpoint_path("/news", auth=auth, params={"limit": max(int(args.news_limit), 1)}),
        ),
        (
            "earnings",
            _endpoint_path("/earnings-revenues", auth=auth, params={"d1": start_date, "d2": end_date}),
        ),
        (
            "dividends",
            _endpoint_path("/dividends", auth=auth, params={"d1": start_date, "d2": end_date}),
        ),
    ]

    for name, url in dataset_specs:
        payload, rows, error = _fetch_dataset(url, timeout=float(args.timeout_seconds))
        ok = error is None and (rows or isinstance(payload, dict))
        dataset_status[name] = {
            "ok": bool(ok),
            "error": error,
            "url": _sanitize_url(url),
            "rows_count": len(rows),
        }
        if payload is not None:
            export_payloads[name] = {
                "timestamp_utc": now_iso,
                "dataset": name,
                "request_url": _sanitize_url(url),
                "response": payload,
            }
            row_archives[name] = rows
            if not args.test_only:
                dataset_dir = external_root / name
                _write_json(dataset_dir / f"{name}_{stamp}.json", export_payloads[name])
                _write_json(dataset_dir / "latest.json", export_payloads[name])

    market_url = _sanitize_url(
        _endpoint_path(f"/markets/symbol/{quote(','.join(market_symbols), safe=',:')}", auth=auth)
    )
    market_rows, market_errors = _fetch_market_symbols(market_symbols, auth=auth, timeout=float(args.timeout_seconds))
    market_ok = len(market_rows) > 0
    dataset_status["markets"] = {
        "ok": market_ok,
        "error": "; ".join(market_errors) if market_errors else None,
        "url": market_url,
        "rows_count": len(market_rows),
    }
    export_payloads["markets"] = {
        "timestamp_utc": now_iso,
        "dataset": "markets",
        "request_url": market_url,
        "response": market_rows,
    }
    row_archives["markets"] = market_rows
    if not args.test_only:
        dataset_dir = external_root / "markets"
        _write_json(dataset_dir / f"markets_{stamp}.json", export_payloads["markets"])
        _write_json(dataset_dir / "latest.json", export_payloads["markets"])

    indicator_rows = row_archives.get("indicators", [])
    forecast_rows = row_archives.get("forecast", [])
    calendar_rows = list(row_archives.get("calendar_country", [])) or list(row_archives.get("calendar", []))
    earnings_rows = row_archives.get("earnings", [])
    dividends_rows = row_archives.get("dividends", [])
    news_rows = row_archives.get("news", [])

    combined_calendar_rows = list(calendar_rows) + list(earnings_rows) + list(dividends_rows)
    calendar_features = summarize_calendar_payload(combined_calendar_rows, now_ts=now.timestamp()) if combined_calendar_rows else {}
    news_features = summarize_structured_news_items(news_rows, symbol="SPY", now_ts=now.timestamp(), max_items=80) if news_rows else {}
    macro_backfill = _derive_macro_backfill(indicator_rows, forecast_rows, country=primary_country)
    market_breadth = _derive_market_breadth(market_rows)
    bond_reference = _derive_bond_reference(indicator_rows, market_rows)

    derived = {
        "calendar_features": calendar_features,
        "news_features": news_features,
        "macro_backfill": macro_backfill,
        "market_breadth": market_breadth,
        "bond_reference": bond_reference,
        "calendar_rows": combined_calendar_rows,
    }

    datasets_ok_count = sum(1 for node in dataset_status.values() if bool(node.get("ok")))
    status = {
        "timestamp_utc": now_iso,
        "provider": "tradingeconomics_guest",
        "auth_mode": auth_mode,
        "countries": countries,
        "datasets_ok_count": datasets_ok_count,
        "datasets_total_count": len(dataset_status),
        "ok": datasets_ok_count > 0,
        "datasets": dataset_status,
    }

    latest_payload = {
        "timestamp_utc": now_iso,
        "provider": "tradingeconomics_guest",
        "auth_mode": auth_mode,
        "countries": countries,
        "status": status,
        "datasets": {
            name: {
                "rows_count": len(rows),
                "rows": rows,
            }
            for name, rows in row_archives.items()
        },
        "derived": derived,
    }

    if not args.test_only:
        _write_json(external_root / f"tradingeconomics_{stamp}.json", latest_payload)
        _write_json(external_root / "latest.json", latest_payload)
        _write_json(health_root / "tradingeconomics_guest_sync_latest.json", status)
        _write_json(
            external_context_root / "tradingeconomics_latest.json",
            {
                "timestamp_utc": now_iso,
                "provider": "tradingeconomics_guest",
                "auth_mode": auth_mode,
                "countries": countries,
                "derived": derived,
                "status": status,
            },
        )
        _write_json(external_context_root / "market_breadth_latest.json", market_breadth)
        _write_json(external_context_root / "bond_reference_latest.json", bond_reference)

        _append_jsonl(
            summary_jsonl_path,
            [
                {
                    "timestamp_utc": now_iso,
                    "provider": "tradingeconomics_guest",
                    "event_type": "sync_summary",
                    "status": status,
                }
            ],
        )

        data_rows: List[Dict[str, Any]] = []
        for dataset_name, rows in row_archives.items():
            data_rows.extend(
                _dataset_event_rows(
                    now_iso=now_iso,
                    dataset=dataset_name,
                    rows=rows,
                    country=primary_country,
                )
            )
        if data_rows:
            _append_jsonl(data_rows_path, data_rows)

    if args.json:
        print(json.dumps(status, ensure_ascii=True))
    else:
        print(
            f"tradingeconomics_guest ok={status['ok']} datasets_ok={datasets_ok_count}/{len(dataset_status)} "
            f"auth_mode={auth_mode}"
        )
        print(f"status_file={health_root / 'tradingeconomics_guest_sync_latest.json'}")
        if not args.test_only:
            print(f"latest_file={external_root / 'latest.json'}")
            print(f"sql_rows={data_rows_path}")
            print(f"market_breadth_latest={external_context_root / 'market_breadth_latest.json'}")
            print(f"bond_reference_latest={external_context_root / 'bond_reference_latest.json'}")

    return 0 if status["ok"] else 1


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Collect the free guest-mode TradingEconomics datasets that are available without an API key."
    )
    parser.add_argument("--auth", default="", help="TradingEconomics auth token. Defaults to TRADINGECONOMICS_AUTH or guest:guest.")
    parser.add_argument("--countries", default=DEFAULT_COUNTRIES, help="Comma-separated country list for indicator/forecast pulls.")
    parser.add_argument("--market-symbols", default=DEFAULT_MARKET_SYMBOLS, help="Comma-separated TradingEconomics market symbols.")
    parser.add_argument("--lookahead-days", type=int, default=45, help="Days ahead for earnings/dividend calendar pulls.")
    parser.add_argument("--news-limit", type=int, default=40, help="Latest news rows to request.")
    parser.add_argument("--timeout-seconds", type=float, default=25.0, help="Per-request timeout in seconds.")
    parser.add_argument("--test-only", action="store_true", help="Connectivity check only; do not write files.")
    parser.add_argument("--json", action="store_true", help="Print final status as JSON.")
    args = parser.parse_args()
    return collect(args)


if __name__ == "__main__":
    raise SystemExit(main())
