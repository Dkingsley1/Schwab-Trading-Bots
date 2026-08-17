#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from core.coinbase_market_data import CoinbaseMarketDataClient, MarketDataAPIError
    from scripts.ops.long_runtime_common import iso_now, write_payload
else:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    from core.coinbase_market_data import CoinbaseMarketDataClient, MarketDataAPIError
    from .long_runtime_common import iso_now, write_payload


DEFAULT_OUT_PATH = PROJECT_ROOT / "governance" / "health" / "coinbase_api_health_latest.json"


def _credential_state() -> dict[str, Any]:
    key = str(os.getenv("COINBASE_API_KEY") or "").strip()
    secret = str(os.getenv("COINBASE_API_SECRET") or "").strip()
    redirect = str(os.getenv("COINBASE_REDIRECT") or "").strip()
    return {
        "api_key_present": bool(key),
        "api_secret_present": bool(secret),
        "redirect_present": bool(redirect),
        "auth_credentials_complete": bool(key and secret),
        "note": "Coinbase market-data checks use public endpoints; secrets are never printed.",
    }


def _error_dict(exc: BaseException) -> dict[str, Any]:
    if isinstance(exc, MarketDataAPIError):
        return {
            "provider": exc.provider,
            "path": exc.path,
            "symbol": exc.symbol,
            "status_code": exc.status_code,
            "reason": exc.reason,
            "attempts": exc.attempts,
        }
    return {"reason": f"{type(exc).__name__}:{exc}"}


def build_payload(
    project_root: Path = PROJECT_ROOT,
    *,
    symbol: str = "BTC-USD",
    timeout_sec: float = 8.0,
    snapshot: bool = False,
) -> dict[str, Any]:
    _ = project_root
    client = CoinbaseMarketDataClient(timeout_seconds=max(float(timeout_sec), 1.0))
    product_id = client.normalize_symbol(symbol)
    started = time.monotonic()
    product: dict[str, Any] = {}
    ticker: dict[str, Any] = {}
    snapshot_payload: dict[str, Any] = {}
    errors: list[dict[str, Any]] = []
    try:
        try:
            product = client.get_product(product_id)
        except BaseException as exc:
            errors.append({"step": "get_product", **_error_dict(exc)})
        try:
            ticker = client.get_ticker(product_id)
        except BaseException as exc:
            errors.append({"step": "get_ticker", **_error_dict(exc)})
        if snapshot:
            try:
                snapshot_payload = client.market_snapshot(product_id)
            except BaseException as exc:
                errors.append({"step": "market_snapshot", **_error_dict(exc)})
    finally:
        client.close()

    latency_ms = round((time.monotonic() - started) * 1000.0, 2)
    product_ok = bool(product.get("id") or product.get("product_id") or product.get("base_currency"))
    ticker_ok = bool(ticker.get("price") or ticker.get("bid") or ticker.get("ask"))
    snapshot_ok = bool(snapshot_payload.get("last_price", 0.0)) if snapshot else None
    public_ok = bool(product_ok and ticker_ok and (snapshot_ok is not False))
    require_auth_creds = str(os.getenv("COINBASE_REQUIRE_AUTH_CREDS", "0")).strip() == "1"
    credentials = _credential_state()
    overall_status = "ready"
    if not public_ok:
        overall_status = "blocked"
    elif require_auth_creds and not bool(credentials["auth_credentials_complete"]):
        overall_status = "degraded"
    recommended_actions = [
        "check local DNS/network access to api.exchange.coinbase.com" if not public_ok else "",
        "set COINBASE_API_KEY and COINBASE_API_SECRET if you intentionally require authenticated Coinbase checks"
        if require_auth_creds and not bool(credentials["auth_credentials_complete"])
        else "",
    ]

    return {
        "timestamp_utc": iso_now(),
        "schema_version": 1,
        "ok": overall_status == "ready",
        "overall_status": overall_status,
        "public_market_data": {
            "ok": public_ok,
            "symbol": product_id,
            "base_url": client.base_url,
            "latency_ms": latency_ms,
            "product_ok": product_ok,
            "ticker_ok": ticker_ok,
            "snapshot_requested": bool(snapshot),
            "snapshot_ok": snapshot_ok,
            "price_present": bool(ticker.get("price") or snapshot_payload.get("last_price")),
            "product_status": str(product.get("status") or "").strip(),
        },
        "credentials": credentials,
        "errors": errors,
        "recommended_actions": [action for action in recommended_actions if action],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Check Coinbase public market-data API health without printing secrets.")
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--out-file", default=str(DEFAULT_OUT_PATH))
    parser.add_argument("--symbol", default=os.getenv("COINBASE_HEALTH_SYMBOL", "BTC-USD"))
    parser.add_argument("--timeout-sec", type=float, default=float(os.getenv("COINBASE_HEALTH_TIMEOUT_SECONDS", "8") or 8))
    parser.add_argument("--snapshot", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    payload = build_payload(project_root, symbol=str(args.symbol), timeout_sec=float(args.timeout_sec), snapshot=bool(args.snapshot))
    write_payload(Path(args.out_file).expanduser(), payload)
    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        print(
            "coinbase_api_health "
            f"overall_status={payload.get('overall_status', '')} "
            f"public_ok={int(bool((payload.get('public_market_data') or {}).get('ok', False)))}"
        )
    return 0 if payload.get("overall_status") in {"ready", "degraded"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
