"""Provider observations and point-in-time context, never trading authority."""

from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import stat

from core.storage_router import inspect_storage_path
from core.decision_price_evidence import digest


def provider_quote_features(payload, symbol, *, now):
    node = payload.get(symbol, {}) if isinstance(payload, dict) else {}
    node = node if isinstance(node, dict) and node.get("symbol") == symbol else {}
    quote = node.get("quote", {})
    quote = quote if isinstance(quote, dict) else {}
    result = {"provider_quote_realtime_norm": float(node.get("realtime") is True)}
    for target, source in (
        ("provider_quote_ts_utc", "quoteTime"),
        ("provider_bid_ts_utc", "bidTime"),
        ("provider_ask_ts_utc", "askTime"),
    ):
        value = quote.get(source)
        try:
            seconds = float(value) / 1000
            valid = (
                not isinstance(value, bool)
                and math.isfinite(seconds)
                and 0 < seconds <= now.timestamp()
            )
        except (TypeError, ValueError, OverflowError):
            valid = False
        if valid:
            result[target] = seconds
    value = quote.get("lastPrice")
    try:
        price = float(value)
        if not isinstance(value, bool) and math.isfinite(price) and price > 0:
            result["provider_quote_last_price"] = price
    except (TypeError, ValueError, OverflowError):
        pass
    return result


def candle_context_receipt(root, symbol, *, now=None):
    """Bind existing bounded context without extra provider calls or file writes."""
    if symbol != "SCHD":
        return {}
    now = now or datetime.now(timezone.utc)
    path = Path(root) / "governance/rehearsals/schd/market_latest.json"
    try:
        route = inspect_storage_path(path, boundary_root=root, allow_external=False)
        if route.get("status") != "present" or route.get("symlinks"):
            return {"state": "unavailable"}
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        with os.fdopen(fd, "rb") as handle:
            info = os.fstat(handle.fileno())
            if not stat.S_ISREG(info.st_mode) or info.st_size > 6 * 1024 * 1024:
                return {"state": "unavailable"}
            raw = handle.read(6 * 1024 * 1024 + 1)
        if len(raw) > 6 * 1024 * 1024:
            return {"state": "unavailable"}
        market = json.loads(raw)
        source = market["source"]
        captured = datetime.fromisoformat(source["fetch_started_at_utc"])
        if captured.tzinfo is None or not 0 <= (now - captured).total_seconds() <= 300:
            return {"state": "stale_or_future"}
        if source.get("symbol") != "SCHD" or source.get("provider") != "schwab":
            return {"state": "wrong_source"}
        return {
            "state": "observed_context_not_claimed_model_input",
            "capture_sha256": digest(market),
            "fetch_started_at_utc": captured.isoformat(),
            "observed_at_utc": now.isoformat(),
        }
    except (OSError, ValueError, KeyError, TypeError):
        return {"state": "unavailable"}
