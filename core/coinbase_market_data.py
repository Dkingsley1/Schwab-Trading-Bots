import json
import os
import socket
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from core.derivatives_features import (
    default_futures_features,
    summarize_futures_quote_features,
    summarize_order_book,
)


class MarketDataAPIError(RuntimeError):
    def __init__(
        self,
        *,
        provider: str,
        path: str,
        symbol: str = "",
        reason: str = "",
        status_code: int = 0,
        attempts: int = 1,
    ):
        self.provider = provider
        self.path = path
        self.symbol = symbol
        self.reason = reason
        self.status_code = int(status_code or 0)
        self.attempts = max(int(attempts), 1)
        detail = f"provider={provider} path={path}"
        if symbol:
            detail += f" symbol={symbol}"
        if self.status_code > 0:
            detail += f" status_code={self.status_code}"
        if reason:
            detail += f" reason={reason}"
        detail += f" attempts={self.attempts}"
        super().__init__(detail)


class CoinbaseMarketDataClient:
    """Public Coinbase market-data client (no trading endpoints)."""

    def __init__(self, timeout_seconds: float = 8.0):
        self.base_url = "https://api.exchange.coinbase.com"
        self.timeout_seconds = float(timeout_seconds)
        self.max_retries = max(int(os.getenv("COINBASE_HTTP_RETRIES", "2")), 0)
        self.retry_backoff_seconds = max(float(os.getenv("COINBASE_HTTP_BACKOFF_SECONDS", "0.35")), 0.0)

    @staticmethod
    def normalize_symbol(symbol: str) -> str:
        s = (symbol or "").strip().upper()
        if not s:
            return s
        if "-" in s:
            return s
        if s.endswith("USD") and len(s) > 3:
            return f"{s[:-3]}-USD"
        return f"{s}-USD"

    def _sleep_before_retry(self, attempt: int) -> None:
        if self.retry_backoff_seconds <= 0.0:
            return
        delay = self.retry_backoff_seconds * (2 ** max(attempt, 0))
        time.sleep(min(delay, 3.0))

    def _get_json(self, path: str, params: Optional[Dict[str, Any]] = None, *, symbol: str = "") -> Any:
        query = ""
        if params:
            query = "?" + urllib.parse.urlencode(params)
        url = f"{self.base_url}{path}{query}"

        attempts = self.max_retries + 1
        last_exc: Optional[BaseException] = None
        for attempt in range(attempts):
            req = urllib.request.Request(url, headers={"User-Agent": "schwab-trading-bot/coinbase-data"})
            try:
                with urllib.request.urlopen(req, timeout=self.timeout_seconds) as resp:
                    status_code = int(getattr(resp, "status", 0) or resp.getcode() or 0)
                    body = resp.read().decode("utf-8")

                if status_code >= 400:
                    raise MarketDataAPIError(
                        provider="coinbase",
                        path=path,
                        symbol=symbol,
                        status_code=status_code,
                        reason="non_2xx_status",
                        attempts=attempt + 1,
                    )

                try:
                    return json.loads(body)
                except json.JSONDecodeError as exc:
                    raise MarketDataAPIError(
                        provider="coinbase",
                        path=path,
                        symbol=symbol,
                        reason=f"invalid_json:{exc}",
                        attempts=attempt + 1,
                    ) from exc
            except MarketDataAPIError as exc:
                last_exc = exc
            except urllib.error.HTTPError as exc:
                last_exc = MarketDataAPIError(
                    provider="coinbase",
                    path=path,
                    symbol=symbol,
                    status_code=int(getattr(exc, "code", 0) or 0),
                    reason=f"http_error:{exc}",
                    attempts=attempt + 1,
                )
            except urllib.error.URLError as exc:
                last_exc = MarketDataAPIError(
                    provider="coinbase",
                    path=path,
                    symbol=symbol,
                    reason=f"url_error:{getattr(exc, 'reason', exc)}",
                    attempts=attempt + 1,
                )
            except socket.timeout as exc:
                last_exc = MarketDataAPIError(
                    provider="coinbase",
                    path=path,
                    symbol=symbol,
                    reason=f"timeout:{exc}",
                    attempts=attempt + 1,
                )
            except TimeoutError as exc:
                last_exc = MarketDataAPIError(
                    provider="coinbase",
                    path=path,
                    symbol=symbol,
                    reason=f"timeout:{exc}",
                    attempts=attempt + 1,
                )
            except Exception as exc:
                last_exc = MarketDataAPIError(
                    provider="coinbase",
                    path=path,
                    symbol=symbol,
                    reason=f"unexpected:{type(exc).__name__}:{exc}",
                    attempts=attempt + 1,
                )

            if attempt < (attempts - 1):
                self._sleep_before_retry(attempt)

        if isinstance(last_exc, MarketDataAPIError):
            raise last_exc
        raise MarketDataAPIError(provider="coinbase", path=path, symbol=symbol, reason="unknown_error", attempts=attempts)

    def get_ticker(self, symbol: str) -> Dict[str, Any]:
        product_id = self.normalize_symbol(symbol)
        out = self._get_json(f"/products/{product_id}/ticker", symbol=product_id)
        return out if isinstance(out, dict) else {}

    def get_product(self, symbol: str) -> Dict[str, Any]:
        product_id = self.normalize_symbol(symbol)
        out = self._get_json(f"/products/{product_id}", symbol=product_id)
        return out if isinstance(out, dict) else {}

    def get_book(self, symbol: str, level: int = 2) -> Dict[str, Any]:
        product_id = self.normalize_symbol(symbol)
        out = self._get_json(f"/products/{product_id}/book", params={"level": int(level)}, symbol=product_id)
        return out if isinstance(out, dict) else {}

    def get_candles(self, symbol: str, minutes: int = 60, granularity: int = 60) -> List[List[float]]:
        product_id = self.normalize_symbol(symbol)
        end_dt = datetime.now(timezone.utc)
        start_dt = end_dt - timedelta(minutes=max(int(minutes), 1))
        out = self._get_json(
            f"/products/{product_id}/candles",
            params={
                "start": start_dt.isoformat().replace("+00:00", "Z"),
                "end": end_dt.isoformat().replace("+00:00", "Z"),
                "granularity": int(granularity),
            },
            symbol=product_id,
        )
        if not isinstance(out, list):
            return []
        out = sorted(out, key=lambda row: float(row[0]) if isinstance(row, list) and row else 0.0)
        return out

    def market_snapshot(self, symbol: str) -> Dict[str, float]:
        product_id = self.normalize_symbol(symbol)
        try:
            ticker = self.get_ticker(product_id)
            candles = self.get_candles(product_id, minutes=60, granularity=60)
        except MarketDataAPIError:
            raise
        except Exception as exc:
            raise MarketDataAPIError(
                provider="coinbase",
                path="market_snapshot",
                symbol=product_id,
                reason=f"unexpected:{type(exc).__name__}:{exc}",
                attempts=1,
            ) from exc

        product: Dict[str, Any] = {}
        book: Dict[str, Any] = {}
        try:
            product = self.get_product(product_id)
        except Exception:
            product = {}
        try:
            book = self.get_book(product_id, level=int(os.getenv("COINBASE_BOOK_LEVEL", "2")))
        except Exception:
            book = {}

        closes: List[float] = []
        highs: List[float] = []
        lows: List[float] = []
        for c in candles:
            if not isinstance(c, list) or len(c) < 5:
                continue
            lows.append(float(c[1]))
            highs.append(float(c[2]))
            closes.append(float(c[4]))

        last_price = 0.0
        try:
            last_price = float(ticker.get("price") or 0.0)
        except Exception:
            last_price = 0.0
        if last_price <= 0.0 and closes:
            last_price = closes[-1]

        prev_close = closes[0] if len(closes) >= 2 else last_price
        if prev_close <= 0.0:
            prev_close = max(last_price, 1.0)

        pct_from_close = (last_price - prev_close) / max(prev_close, 1e-8)

        rets: List[float] = []
        for i in range(1, len(closes)):
            p0 = closes[i - 1]
            p1 = closes[i]
            if p0 > 0 and p1 > 0:
                rets.append((p1 - p0) / p0)

        tail = rets[-30:]
        vol_30m = (sum(r * r for r in tail) / max(len(tail), 1)) ** 0.5

        mom_5m = 0.0
        if len(closes) >= 6 and closes[-6] > 0:
            mom_5m = (closes[-1] - closes[-6]) / closes[-6]

        day_high = max(highs) if highs else max(last_price, prev_close)
        day_low = min(lows) if lows else min(last_price, prev_close)
        range_pos = 0.5
        if day_high > day_low:
            range_pos = (last_price - day_low) / (day_high - day_low)

        futures_quote = summarize_futures_quote_features(
            {"ticker": ticker, "product": product},
            last_price=float(last_price),
        )
        book_features = summarize_order_book(
            book,
            last_price=float(last_price),
            top_n=max(int(os.getenv("COINBASE_BOOK_TOP_N", "5")), 1),
        )
        futures = default_futures_features()
        futures.update(futures_quote)
        for key, value in book_features.items():
            if key in futures:
                futures[key] = float(value)

        out = {
            "last_price": float(last_price),
            "prev_close": float(prev_close),
            "pct_from_close": float(pct_from_close),
            "vol_30m": float(vol_30m),
            "mom_5m": float(mom_5m),
            "range_pos": float(range_pos),
            "spread_bps": float(futures.get("futures_spread_bps", 0.0)),
            "bid_size": float(futures.get("futures_bid_size", 0.0)),
            "ask_size": float(futures.get("futures_ask_size", 0.0)),
            "snapshot_ts_utc": time.time(),
        }
        out.update(futures)
        return out
