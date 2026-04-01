#!/usr/bin/env python3
import argparse
from http.client import RemoteDisconnected
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse
from urllib.request import Request, urlopen


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FRED_SERIES_ALIASES = {
    "GOLDAMGBD228NLBM": ["GOLDPMGBD228NLBM"],
}


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

    return {
        "timestamp_utc": fred_payload.get("timestamp_utc"),
        "provider": "fred",
        "treasury_yields": treasury_yields,
        "cross_asset": cross_asset,
        "bond_reference_overlay": bond_reference_overlay,
    }


def collect(args: argparse.Namespace) -> int:
    _bootstrap_env()

    now = datetime.now(timezone.utc)
    stamp = now.strftime("%Y%m%d_%H%M%S")
    year_now = now.year
    start_year = str(max(year_now - 1, 2000))
    end_year = str(year_now)

    bls_series = [s.strip() for s in (os.getenv("BLS_SERIES_IDS", "CUUR0000SA0,LNS14000000")).split(",") if s.strip()]
    bls_key = os.getenv("BLS_API_KEY", "").strip()

    census_key = os.getenv("CENSUS_API_KEY", "").strip()
    if not census_key:
        print("CENSUS_API_KEY missing in environment (.env/.env.live)")
        return 2
    census_dataset = os.getenv("CENSUS_DATASET", "2023/acs/acs5")
    census_get = os.getenv("CENSUS_GET_VARS", "NAME,B01001_001E")
    census_for = os.getenv("CENSUS_FOR", "us:1")

    fred_key = os.getenv("FRED_API_KEY", "").strip()
    fred_series = [
        s.strip()
        for s in (
            os.getenv(
                "FRED_SERIES_IDS",
                "GDP,UNRATE,CPIAUCSL,DGS2,DGS5,DGS10,DGS30,DFII10,VIXCLS,DCOILWTICO,DTWEXBGS,BAMLH0A0HYM2",
            )
        ).split(",")
        if s.strip()
    ]
    fred_required = {
        s.strip().upper()
        for s in (os.getenv("FRED_REQUIRED_SERIES_IDS", "GDP,UNRATE,CPIAUCSL")).split(",")
        if s.strip()
    }
    fred_limit = max(int(os.getenv("FRED_LIMIT", "5")), 1)

    bea_key = os.getenv("BEA_API_KEY", "").strip()

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
    census_url = f"https://api.census.gov/data/{census_dataset}?get={census_get}&for={census_for}&key={census_key}"
    status["census"]["url"] = _sanitize_url(census_url)
    try:
        census_resp = _http_json(census_url, method="GET")
        census_ok = isinstance(census_resp, list) and len(census_resp) >= 2
        status["census"]["ok"] = bool(census_ok)
        if not census_ok:
            status["census"]["error"] = "unexpected_response_shape"
        if not args.test_only:
            census_payload = {
                "timestamp_utc": now.isoformat(),
                "request": {"dataset": census_dataset, "get": census_get, "for": census_for, "url": _sanitize_url(census_url)},
                "response": census_resp,
            }
            _write_json(census_root / f"census_{stamp}.json", census_payload)
            _write_json(census_root / "latest.json", census_payload)
    except (HTTPError, URLError, TimeoutError, ValueError, RemoteDisconnected, OSError) as exc:
        status["census"]["error"] = str(exc)

    # FRED
    if not fred_key:
        status["fred"]["error"] = "FRED_API_KEY missing in environment (.env/.env.live)"
    else:
        fred_base_url = "https://api.stlouisfed.org/fred/series/observations"
        if fred_series:
            sample_url = fred_base_url + "?" + urlencode(
                {"series_id": fred_series[0], "api_key": fred_key, "file_type": "json", "sort_order": "desc", "limit": fred_limit}
            )
            status["fred"]["url"] = _sanitize_url(sample_url)

        fred_collected: dict[str, Any] = {}
        fred_errors: list[str] = []
        fred_warnings: list[str] = []
        fred_aliases_used: dict[str, str] = {}
        for series_id in fred_series:
            candidate_ids = [series_id, *FRED_SERIES_ALIASES.get(series_id, [])]
            candidate_errors: list[str] = []
            for candidate_id in candidate_ids:
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
            if series_id not in fred_collected:
                message = f"series_id={series_id} error={' | '.join(candidate_errors)}"
                if series_id in fred_required:
                    fred_errors.append(message)
                else:
                    fred_warnings.append(message)

        fred_ok = all(series_id in fred_collected for series_id in fred_required)
        status["fred"]["ok"] = fred_ok
        if fred_aliases_used:
            status["fred"]["aliases_used"] = fred_aliases_used
        if fred_errors:
            status["fred"]["error"] = "; ".join(fred_errors)
        if fred_warnings:
            status["fred"]["warnings"] = fred_warnings

        if not args.test_only:
            fred_payload = {
                "timestamp_utc": now.isoformat(),
                "request": {"series_ids": fred_series, "limit": fred_limit},
                "responses": fred_collected,
            }
            _write_json(fred_root / f"fred_{stamp}.json", fred_payload)
            _write_json(fred_root / "latest.json", fred_payload)

            external_context_root = PROJECT_ROOT / "exports" / "external_context"
            macro_context = _derive_fred_macro_context(fred_payload)
            if macro_context:
                _write_json(external_context_root / "macro_cross_asset_latest.json", macro_context)
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
        status["bea"]["error"] = "BEA_API_KEY missing in environment (.env/.env.live)"
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
