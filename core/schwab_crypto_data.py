"""Read-only Schwab crypto futures/ETP context, never a spot or order feed."""

from __future__ import annotations

import http.client
import json
import math
import os
import re
import stat
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlencode

HOST = "api.schwabapi.com"
QUOTE_PATH = "/marketdata/v1/quotes"
MAX_BYTES = 512 * 1024
MAX_QUOTE_AGE = 120
# Explicit product identities, not ticker-name guesses. ETHE is reported as CEF.
INSTRUMENTS = (
    ("BTC", "/BTC", "future"),
    ("BTC", "/MBT", "future"),
    ("BTC", "IBIT", "etp"),
    ("BTC", "FBTC", "etp"),
    ("ETH", "/ETH", "future"),
    ("ETH", "/MET", "future"),
    ("ETH", "ETHA", "etp"),
    ("ETH", "ETHE", "etp"),
)
FEATURE_KEYS = tuple(
    f"crypto_schwab_{kind}_{suffix}_norm"
    for kind in ("future", "etp")
    for suffix in ("available", "return", "spread")
)


class SchwabContextError(ValueError):
    """Fixed non-secret reason codes only."""


def number(value):
    if isinstance(value, bool):
        return None
    try:
        value = float(value)
        return value if math.isfinite(value) and abs(value) <= 1e18 else None
    except (TypeError, ValueError, OverflowError):
        return None


def iso(epoch):
    return datetime.fromtimestamp(epoch, timezone.utc).isoformat()


def read_access_token(path: Path, now: float) -> tuple[str, bool]:
    """Read the auth owner's current generation without refreshing or writing it."""
    try:
        if any(parent.is_symlink() for parent in (path, *path.parents)):
            raise SchwabContextError("token_path_unsafe")
        with os.fdopen(
            os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK), "rb"
        ) as handle:
            info = os.fstat(handle.fileno())
            if (
                not stat.S_ISREG(info.st_mode)
                or info.st_uid != os.getuid()
                or info.st_mode & 0o022
            ):
                raise SchwabContextError("token_path_unsafe")
            data = handle.read(65537)
        if len(data) > 65536:
            raise SchwabContextError("token_file_invalid")
        payload = json.loads(data)
        token = payload.get("token") if isinstance(payload, dict) else None
        if not isinstance(token, dict):
            raise SchwabContextError("token_file_invalid")
        expiry = number(token.get("expires_at"))
        if expiry is None or expiry <= now + 15:
            raise SchwabContextError("auth_refresh_required")
        access = token.get("access_token")
        if not isinstance(access, str) or not re.fullmatch(r"[!-~]{1,8192}", access):
            raise SchwabContextError("token_file_invalid")
        return access, bool(info.st_mode & 0o077)
    except SchwabContextError:
        raise
    except (OSError, ValueError, UnicodeError, RecursionError):
        raise SchwabContextError("token_unavailable") from None


def fetch_quotes(symbols: list[str], *, access_token: str, timeout: float) -> dict:
    allowed = {item[1] for item in INSTRUMENTS}
    if not symbols or len(symbols) > 8 or not set(symbols) <= allowed:
        raise SchwabContextError("unsupported_instruments")
    budget = number(timeout)
    if budget is None or budget <= 0:
        raise SchwabContextError("request_deadline_exceeded")
    deadline = time.monotonic() + min(budget, 8.0)
    connection = http.client.HTTPSConnection(HOST, timeout=min(budget, 8.0))

    def remaining():
        left = deadline - time.monotonic()
        if left <= 0:
            raise SchwabContextError("request_deadline_exceeded")
        if connection.sock is not None:
            connection.sock.settimeout(left)

    try:
        connection.connect()
        remaining()
        connection.request(
            "GET",
            QUOTE_PATH
            + "?"
            + urlencode({"symbols": ",".join(symbols), "fields": "quote,reference"}),
            headers={
                "Authorization": "Bearer " + access_token,
                "Accept": "application/json",
                "Accept-Encoding": "identity",
                "User-Agent": "SchwabTradingPlatform/crypto-context",
            },
        )
        remaining()
        response = connection.getresponse()
        if response.status != 200:
            code = (
                "auth_refresh_required"
                if response.status == 401
                else f"http_{response.status}"
            )
            raise SchwabContextError(code)
        if response.getheader("Content-Encoding", "identity").lower() != "identity":
            raise SchwabContextError("unsupported_response_encoding")
        chunks, size = [], 0
        while True:
            remaining()
            chunk = response.read1(min(65536, MAX_BYTES + 1 - size))
            size += len(chunk)
            if size > MAX_BYTES:
                raise SchwabContextError("response_too_large")
            if not chunk:
                break
            chunks.append(chunk)
        remaining()
        payload = json.loads(b"".join(chunks))
        if not isinstance(payload, dict):
            raise SchwabContextError("invalid_quote_payload")
        return payload
    except SchwabContextError:
        raise
    except (
        OSError,
        http.client.HTTPException,
        ValueError,
        UnicodeError,
        RecursionError,
    ):
        raise SchwabContextError("quote_request_failed") from None
    finally:
        connection.close()


def normalize_quote(payload: dict, instrument: tuple, now: float) -> dict:
    asset, requested, kind = instrument
    result = {
        "provider": "schwab",
        "asset": asset,
        "requested_symbol": requested,
        "instrument_type": kind,
        "data_role": "context_only",
        "currency": "USD",
        "currency_source": "instrument_registry",
        "observed_at": iso(now),
        "price_unit": (
            "USD_per_underlying_unit" if kind == "future" else "USD_per_share"
        ),
        "volume_unit": "contracts" if kind == "future" else "shares",
        "source_symbol": None,
        "quote_timestamp_utc": None,
        "quote_age_seconds": None,
        "realtime": False,
        "usable": False,
        "quality": "missing_quote",
        "spot_price_eligible": False,
    }
    rows = [
        row
        for row in payload.values()
        if isinstance(row, dict)
        and (
            row.get("symbol") == requested
            or (
                kind == "future"
                and isinstance(row.get("reference"), dict)
                and row["reference"].get("product") == requested
            )
        )
    ]
    if len(rows) != 1:
        result["quality"] = "ambiguous_quote" if rows else "missing_quote"
        return result
    row = rows[0]
    reference = row.get("reference")
    quote = row.get("quote")
    if not isinstance(reference, dict) or not isinstance(quote, dict):
        result["quality"] = "invalid_quote_schema"
        return result
    symbol = row.get("symbol")
    if kind == "future":
        expiry = number(reference.get("futureExpirationDate"))
        valid = (
            row.get("assetMainType") == "FUTURE"
            and isinstance(symbol, str)
            and re.fullmatch(re.escape(requested) + r"[FGHJKMNQUVXZ]\d{2}", symbol)
            and reference.get("product") == requested
            and reference.get("futureIsActive") is True
            and expiry is not None
            and now * 1000 < expiry < (now + 366 * 86400) * 1000
        )
    else:
        valid = (
            row.get("assetMainType") == "EQUITY"
            and row.get("assetSubType") in ("ETF", "CEF")
            and symbol == requested
        )
    if not valid:
        result["quality"] = "instrument_identity_mismatch"
        return result
    result.update(source_symbol=symbol, realtime=row.get("realtime") is True)
    if kind == "future":
        result.update(
            contract_expiration_utc=iso(expiry / 1000),
            contract_multiplier=number(reference.get("futureMultiplier")),
        )
    stamp = number(quote.get("quoteTime"))
    # Epoch milliseconds are required. Never substitute fetch time or last close.
    age = now - stamp / 1000 if stamp is not None else None
    if stamp is not None and 946684800000 <= stamp <= (now + 366 * 86400) * 1000:
        result.update(
            quote_timestamp_utc=iso(stamp / 1000), quote_age_seconds=round(age, 3)
        )
    if age is None or not 0 <= age <= MAX_QUOTE_AGE:
        result["quality"] = (
            "quote_time_invalid" if age is None or age < 0 else "stale_quote"
        )
        return result
    result.update(
        quote_timestamp_utc=iso(stamp / 1000), quote_age_seconds=round(age, 3)
    )
    if not result["realtime"]:
        result["quality"] = "delayed_or_unverified"
        return result
    if quote.get("securityStatus") != "Normal" or (
        kind == "future" and quote.get("quotedInSession") is not True
    ):
        result["quality"] = "market_not_normal"
        return result
    bid, ask, mark = (
        number(quote.get(key)) for key in ("bidPrice", "askPrice", "mark")
    )
    if any(value is None or value <= 0 for value in (bid, ask, mark)) or bid > ask:
        result["quality"] = "invalid_price_or_spread"
        return result
    for key in ("bidTime", "askTime"):
        side_stamp = number(quote.get(key))
        if side_stamp is None or not 0 <= now - side_stamp / 1000 <= MAX_QUOTE_AGE:
            result["quality"] = "stale_or_invalid_book"
            return result
        result["bid_timestamp_utc" if key == "bidTime" else "ask_timestamp_utc"] = iso(
            side_stamp / 1000
        )
    spread = (ask - bid) / ((ask + bid) / 2) * 10000
    if spread > 500 or not bid * 0.95 <= mark <= ask * 1.05:
        result["quality"] = "implausible_price_or_spread"
        return result
    close = number(quote.get("closePrice"))
    change = (mark / close - 1) * 100 if close is not None and close > 0 else None
    result.update(
        quality="fresh",
        usable=True,
        mark=mark,
        bid=bid,
        ask=ask,
        spread_bps=round(spread, 6),
        return_pct=change,
        return_baseline="previous_settlement" if kind == "future" else "previous_close",
        total_volume=number(quote.get("totalVolume")),
        open_interest=number(quote.get("openInterest")) if kind == "future" else None,
    )
    return result


def derive_features(usable: list[dict]) -> dict:
    features = {}
    for asset in sorted({row["asset"] for row in usable}):
        values = features.setdefault(asset, {})
        for kind in ("future", "etp"):
            group = [
                row
                for row in usable
                if row["asset"] == asset and row["instrument_type"] == kind
            ]
            if not group:
                continue
            prefix = f"crypto_schwab_{kind}"
            values[prefix + "_available_norm"] = 1.0
            changes = [
                row["return_pct"] for row in group if row["return_pct"] is not None
            ]
            if changes:
                values[prefix + "_return_norm"] = max(
                    0.0, min(1.0, 0.5 + sum(changes) / len(changes) / 20)
                )
            values[prefix + "_spread_norm"] = min(
                1.0, sum(row["spread_bps"] for row in group) / len(group) / 100
            )
    return features


def current_features(snapshot: dict, symbol: str, now: float | None = None) -> dict:
    """Recompute from still-fresh instrument receipts, not cached aggregate values."""
    if snapshot.get("provider") != "crypto_market_context" or symbol not in {
        "BTC-USD",
        "ETH-USD",
    }:
        return {}
    now = time.time() if now is None else now

    def current(stamp, max_age):
        try:
            measured = datetime.fromisoformat(str(stamp).replace("Z", "+00:00"))
            return (
                measured.tzinfo is not None
                and 0 <= now - measured.timestamp() <= max_age
            )
        except (ValueError, TypeError, OverflowError, OSError):
            return False

    if not current(snapshot.get("timestamp_utc"), 300):
        return {}
    sources = snapshot.get("sources")
    if (
        not isinstance(sources, dict)
        or sources.get("schwab_data_role") != "context_only"
    ):
        return {}
    rows = sources.get("schwab_instruments")
    if not isinstance(rows, list) or len(rows) > 8:
        return {}
    usable, seen = [], set()
    for row in rows:
        if not isinstance(row, dict):
            continue
        identity = (
            row.get("asset"),
            row.get("requested_symbol"),
            row.get("instrument_type"),
        )
        if any(not isinstance(x, str) for x in identity) or identity not in INSTRUMENTS:
            continue
        if identity in seen:
            return {}
        seen.add(identity)
        if (
            row.get("provider") != "schwab"
            or row.get("data_role") != "context_only"
            or row.get("usable") is not True
            or row.get("realtime") is not True
            or row.get("spot_price_eligible") is not False
            or row.get("currency") != "USD"
            or not current(row.get("quote_timestamp_utc"), MAX_QUOTE_AGE)
            or not current(row.get("bid_timestamp_utc"), MAX_QUOTE_AGE)
            or not current(row.get("ask_timestamp_utc"), MAX_QUOTE_AGE)
            or not current(row.get("observed_at"), MAX_QUOTE_AGE)
        ):
            continue
        spread, change = number(row.get("spread_bps")), number(row.get("return_pct"))
        if spread is None or not 0 <= spread <= 500:
            continue
        usable.append({**row, "spread_bps": spread, "return_pct": change})
    return derive_features(usable).get(symbol.split("-")[0], {})


def collect_context(
    assets: list[str], *, token_path: Path, timeout: float
) -> tuple[dict, dict]:
    requested = [item for item in INSTRUMENTS if item[0] in assets]
    status = {
        "schema_version": 1,
        "ok": False,
        "optional": True,
        "state": "unavailable",
        "provider": "schwab",
        "linked_provider": "coinbase",
        "data_role": "context_only",
        "spot_feed_verified": False,
        "live_execution_allowed": False,
        "transfers_allowed": False,
        "requested_instruments": len(requested),
        "usable_instruments": 0,
        "resolved_assets": 0,
        "max_quote_age_seconds": MAX_QUOTE_AGE,
        "error": None,
        "instruments": [],
        "warnings": [],
    }
    if not requested:
        status["state"] = "no_supported_assets"
        return {}, status
    try:
        token, permission_warning = read_access_token(token_path, time.time())
        if permission_warning:
            status["warnings"].append("auth_owner_token_not_owner_only")
        payload = fetch_quotes(
            [item[1] for item in requested], access_token=token, timeout=timeout
        )
    except SchwabContextError as exc:
        status.update(state=str(exc), error=str(exc))
        return {}, status
    now = time.time()
    rows = [normalize_quote(payload, item, now) for item in requested]
    usable = [row for row in rows if row["usable"]]
    features = derive_features(usable)
    status.update(
        ok=bool(usable),
        instruments=rows,
        usable_instruments=len(usable),
        resolved_assets=len(features),
        timestamp_utc=iso(now),
        state=(
            "ready"
            if len(usable) == len(requested)
            else "partial" if usable else "no_usable_quotes"
        ),
    )
    return features, status
