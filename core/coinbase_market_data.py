import asyncio
import atexit
import json
import os
import socket
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

from core.derivatives_features import (
    default_futures_features,
    summarize_futures_quote_features,
    summarize_order_book,
)

try:
    from websockets.asyncio.client import connect as websocket_connect
except Exception:  # pragma: no cover - compatibility fallback
    try:
        import websockets as _websockets

        websocket_connect = getattr(_websockets, "connect", None)
    except Exception:  # pragma: no cover - optional dependency in runtime env
        websocket_connect = None


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


def _parse_csv_symbols(raw: str) -> List[str]:
    out: List[str] = []
    seen: set[str] = set()
    for token in str(raw or "").split(","):
        symbol = CoinbaseMarketDataClient.normalize_symbol(token)
        if not symbol or symbol in seen:
            continue
        seen.add(symbol)
        out.append(symbol)
    return out


class _CoinbaseWebsocketCache:
    """Best-effort public websocket cache for fast Coinbase symbols."""

    def __init__(self) -> None:
        self.enabled = (websocket_connect is not None) and (os.getenv("COINBASE_WEBSOCKET_ENABLED", "1").strip() == "1")
        self.url = os.getenv("COINBASE_WEBSOCKET_URL", "wss://ws-feed.exchange.coinbase.com").strip() or "wss://ws-feed.exchange.coinbase.com"
        self.depth_levels = max(int(os.getenv("COINBASE_WEBSOCKET_BOOK_DEPTH", "10")), 1)
        self.stale_seconds = max(float(os.getenv("COINBASE_WEBSOCKET_STALE_SECONDS", "8")), 0.5)
        self.reconnect_seconds = max(float(os.getenv("COINBASE_WEBSOCKET_RECONNECT_SECONDS", "2.0")), 0.25)
        self._state_lock = threading.Lock()
        self._state: Dict[str, Dict[str, Any]] = {}
        self._symbols: List[str] = []
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def close(self) -> None:
        self._stop.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.5)
        self._thread = None

    def set_symbols(self, symbols: List[str]) -> None:
        if not self.enabled:
            return
        requested = [CoinbaseMarketDataClient.normalize_symbol(s) for s in symbols if str(s or "").strip()]
        explicit = _parse_csv_symbols(os.getenv("COINBASE_WEBSOCKET_SYMBOLS", ""))
        if explicit:
            allow = set(requested)
            if allow:
                requested = [s for s in explicit if s in allow]
            else:
                requested = explicit
        deduped: List[str] = []
        seen: set[str] = set()
        for symbol in requested:
            if not symbol or symbol in seen:
                continue
            seen.add(symbol)
            deduped.append(symbol)

        with self._state_lock:
            if deduped == self._symbols:
                return
            self._symbols = deduped

        self.close()
        if not deduped:
            return

        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run,
            name="coinbase-websocket-cache",
            daemon=True,
        )
        self._thread.start()

    def latest(self, symbol: str) -> Optional[Dict[str, Any]]:
        if not self.enabled:
            return None
        product_id = CoinbaseMarketDataClient.normalize_symbol(symbol)
        with self._state_lock:
            row = dict(self._state.get(product_id, {}))
        if not row:
            return None
        freshest = max(float(row.get("ticker_ts", 0.0) or 0.0), float(row.get("book_ts", 0.0) or 0.0))
        if freshest <= 0.0 or (time.time() - freshest) > self.stale_seconds:
            return None

        bids = row.get("bids") if isinstance(row.get("bids"), dict) else {}
        asks = row.get("asks") if isinstance(row.get("asks"), dict) else {}
        best_bid = self._best_level(bids, reverse=True)
        best_ask = self._best_level(asks, reverse=False)
        ticker = dict(row.get("ticker", {})) if isinstance(row.get("ticker"), dict) else {}

        if best_bid:
            ticker.setdefault("bid", f"{best_bid[0]:.8f}")
            ticker.setdefault("bidSize", f"{best_bid[1]:.8f}")
        if best_ask:
            ticker.setdefault("ask", f"{best_ask[0]:.8f}")
            ticker.setdefault("askSize", f"{best_ask[1]:.8f}")

        return {
            "ticker": ticker,
            "book": {
                "bids": [[f"{price:.8f}", f"{size:.8f}"] for price, size in self._ordered_levels(bids, reverse=True)],
                "asks": [[f"{price:.8f}", f"{size:.8f}"] for price, size in self._ordered_levels(asks, reverse=False)],
            },
            "snapshot_ts_utc": freshest,
            "source": "websocket",
        }

    def _run(self) -> None:
        asyncio.run(self._run_async())

    async def _run_async(self) -> None:
        if websocket_connect is None:
            return

        while not self._stop.is_set():
            with self._state_lock:
                symbols = list(self._symbols)
            if not symbols:
                return
            try:
                async with websocket_connect(
                    self.url,
                    ping_interval=20,
                    ping_timeout=20,
                    close_timeout=2,
                    max_size=None,
                ) as ws:
                    await ws.send(
                        json.dumps(
                            {
                                "type": "subscribe",
                                "product_ids": symbols,
                                "channels": ["ticker", "level2_batch", "heartbeat"],
                            }
                        )
                    )
                    while not self._stop.is_set():
                        raw = await asyncio.wait_for(ws.recv(), timeout=10.0)
                        payload = json.loads(raw)
                        if isinstance(payload, dict):
                            self._handle_message(payload)
            except Exception:
                if self._stop.is_set():
                    break
                await asyncio.sleep(self.reconnect_seconds)

    def _handle_message(self, payload: Dict[str, Any]) -> None:
        msg_type = str(payload.get("type") or "").strip().lower()
        product_id = CoinbaseMarketDataClient.normalize_symbol(payload.get("product_id", ""))
        if not product_id:
            return
        now_ts = time.time()
        with self._state_lock:
            row = self._state.setdefault(product_id, {"ticker": {}, "bids": {}, "asks": {}, "ticker_ts": 0.0, "book_ts": 0.0})
            if msg_type == "ticker":
                row["ticker"] = {
                    "price": str(payload.get("price") or ""),
                    "bid": str(payload.get("best_bid") or payload.get("bid") or ""),
                    "ask": str(payload.get("best_ask") or payload.get("ask") or ""),
                    "bidSize": str(payload.get("best_bid_size") or payload.get("bid_size") or ""),
                    "askSize": str(payload.get("best_ask_size") or payload.get("ask_size") or ""),
                }
                row["ticker_ts"] = now_ts
                return

            if msg_type == "snapshot":
                row["bids"] = self._levels_from_rows(payload.get("bids"), reverse=True)
                row["asks"] = self._levels_from_rows(payload.get("asks"), reverse=False)
                row["book_ts"] = now_ts
                return

            if msg_type == "l2update":
                self._apply_changes(row.get("bids"), row.get("asks"), payload.get("changes"))
                row["bids"] = self._trim_levels(row.get("bids"), reverse=True)
                row["asks"] = self._trim_levels(row.get("asks"), reverse=False)
                row["book_ts"] = now_ts

    def _levels_from_rows(self, rows: Any, *, reverse: bool) -> Dict[float, float]:
        out: Dict[float, float] = {}
        if not isinstance(rows, list):
            return out
        for row in rows:
            if not isinstance(row, list) or len(row) < 2:
                continue
            try:
                price = float(row[0] or 0.0)
                size = float(row[1] or 0.0)
            except Exception:
                continue
            if price > 0.0 and size > 0.0:
                out[price] = size
        return self._trim_levels(out, reverse=reverse)

    def _apply_changes(self, bids: Any, asks: Any, changes: Any) -> None:
        if not isinstance(changes, list):
            return
        bid_levels = bids if isinstance(bids, dict) else {}
        ask_levels = asks if isinstance(asks, dict) else {}
        for change in changes:
            if not isinstance(change, list) or len(change) < 3:
                continue
            side = str(change[0] or "").strip().lower()
            try:
                price = float(change[1] or 0.0)
                size = float(change[2] or 0.0)
            except Exception:
                continue
            if price <= 0.0:
                continue
            levels = bid_levels if side == "buy" else ask_levels
            if size <= 0.0:
                levels.pop(price, None)
            else:
                levels[price] = size

    def _trim_levels(self, levels: Any, *, reverse: bool) -> Dict[float, float]:
        if not isinstance(levels, dict):
            return {}
        ordered = sorted(
            ((float(price), float(size)) for price, size in levels.items() if float(price) > 0.0 and float(size) > 0.0),
            key=lambda item: item[0],
            reverse=reverse,
        )
        return {price: size for price, size in ordered[: self.depth_levels]}

    def _ordered_levels(self, levels: Any, *, reverse: bool) -> List[Tuple[float, float]]:
        if not isinstance(levels, dict):
            return []
        return sorted(
            ((float(price), float(size)) for price, size in levels.items() if float(price) > 0.0 and float(size) > 0.0),
            key=lambda item: item[0],
            reverse=reverse,
        )[: self.depth_levels]

    def _best_level(self, levels: Any, *, reverse: bool) -> Optional[Tuple[float, float]]:
        ordered = self._ordered_levels(levels, reverse=reverse)
        return ordered[0] if ordered else None


class CoinbaseMarketDataClient:
    """Public Coinbase market-data client (no trading endpoints)."""

    def __init__(self, timeout_seconds: float = 8.0):
        self.base_url = "https://api.exchange.coinbase.com"
        self.timeout_seconds = float(timeout_seconds)
        self.max_retries = max(int(os.getenv("COINBASE_HTTP_RETRIES", "2")), 0)
        self.retry_backoff_seconds = max(float(os.getenv("COINBASE_HTTP_BACKOFF_SECONDS", "0.35")), 0.0)
        self.product_cache_seconds = max(float(os.getenv("COINBASE_PRODUCT_CACHE_SECONDS", "900")), 0.0)
        self.candle_cache_seconds = max(float(os.getenv("COINBASE_CANDLE_CACHE_SECONDS", "12")), 0.0)
        self.snapshot_max_workers = max(int(os.getenv("COINBASE_SNAPSHOT_MAX_WORKERS", "4")), 2)
        self._cache_lock = threading.Lock()
        self._cache: Dict[str, Tuple[float, Any]] = {}
        self._ws_cache = _CoinbaseWebsocketCache()
        self._executor = ThreadPoolExecutor(
            max_workers=self.snapshot_max_workers,
            thread_name_prefix="coinbase-market-data",
        )
        atexit.register(self.close)

    def close(self) -> None:
        self._ws_cache.close()
        try:
            self._executor.shutdown(wait=False, cancel_futures=True)
        except Exception:
            return

    def set_live_symbols(self, symbols: List[str]) -> None:
        self._ws_cache.set_symbols(symbols)

    def _cache_get(self, key: str) -> Any:
        with self._cache_lock:
            row = self._cache.get(key)
            if not row:
                return None
            expires_at, value = row
            if time.time() >= float(expires_at):
                self._cache.pop(key, None)
                return None
            return value

    def _cache_set(self, key: str, value: Any, ttl_seconds: float) -> Any:
        if ttl_seconds <= 0.0:
            return value
        with self._cache_lock:
            self._cache[key] = (time.time() + float(ttl_seconds), value)
        return value

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
        cache_key = f"product:{product_id}"
        out = self._cache_get(cache_key)
        if out is None:
            out = self._get_json(f"/products/{product_id}", symbol=product_id)
            self._cache_set(cache_key, out, self.product_cache_seconds)
        return out if isinstance(out, dict) else {}

    def get_book(self, symbol: str, level: int = 2) -> Dict[str, Any]:
        product_id = self.normalize_symbol(symbol)
        out = self._get_json(f"/products/{product_id}/book", params={"level": int(level)}, symbol=product_id)
        return out if isinstance(out, dict) else {}

    def get_candles(self, symbol: str, minutes: int = 60, granularity: int = 60) -> List[List[float]]:
        product_id = self.normalize_symbol(symbol)
        cache_key = f"candles:{product_id}:{max(int(minutes), 1)}:{max(int(granularity), 1)}"
        cached = self._cache_get(cache_key)
        if isinstance(cached, list):
            return cached
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
        self._cache_set(cache_key, out, self.candle_cache_seconds)
        return out

    def market_snapshot(self, symbol: str) -> Dict[str, float]:
        product_id = self.normalize_symbol(symbol)
        websocket_snapshot = self._ws_cache.latest(product_id) or {}
        ws_ticker = websocket_snapshot.get("ticker") if isinstance(websocket_snapshot.get("ticker"), dict) else {}
        ws_book = websocket_snapshot.get("book") if isinstance(websocket_snapshot.get("book"), dict) else {}
        use_ws_ticker = bool(ws_ticker)
        use_ws_book = bool(ws_book)
        try:
            book_level = int(os.getenv("COINBASE_BOOK_LEVEL", "2"))
            futures: Dict[str, Any] = {
                "candles": self._executor.submit(self.get_candles, product_id, 60, 60),
                "product": self._executor.submit(self.get_product, product_id),
            }
            if not use_ws_ticker:
                futures["ticker"] = self._executor.submit(self.get_ticker, product_id)
            if not use_ws_book:
                futures["book"] = self._executor.submit(self.get_book, product_id, book_level)
            errors: Dict[str, Exception] = {}
            results: Dict[str, Any] = {}
            for key, future in futures.items():
                try:
                    results[key] = future.result()
                except Exception as exc:
                    errors[key] = exc

            ticker = ws_ticker if use_ws_ticker else (results.get("ticker") if isinstance(results.get("ticker"), dict) else {})
            candles = results.get("candles") if isinstance(results.get("candles"), list) else []
            if ("ticker" in errors) or ("candles" in errors):
                err = errors.get("ticker") or errors.get("candles")
                if isinstance(err, MarketDataAPIError):
                    raise err
                raise RuntimeError(f"snapshot_fetch_failed:{type(err).__name__}:{err}") from err
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

        product = results.get("product") if isinstance(results.get("product"), dict) else {}
        book = ws_book if use_ws_book else (results.get("book") if isinstance(results.get("book"), dict) else {})

        closes: List[float] = []
        highs: List[float] = []
        lows: List[float] = []
        volumes: List[float] = []
        vwap_numerator = 0.0
        vwap_denominator = 0.0
        for c in candles:
            if not isinstance(c, list) or len(c) < 5:
                continue
            lows.append(float(c[1]))
            highs.append(float(c[2]))
            closes.append(float(c[4]))
            volume = float(c[5]) if len(c) >= 6 else 0.0
            volumes.append(max(volume, 0.0))
            typical_price = (float(c[1]) + float(c[2]) + float(c[4])) / 3.0
            if volume > 0.0 and typical_price > 0.0:
                vwap_numerator += typical_price * volume
                vwap_denominator += volume

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
        tail_5 = rets[-5:]
        volatility_1m = (sum(r * r for r in tail_5) / max(len(tail_5), 1)) ** 0.5
        return_1m = rets[-1] if rets else 0.0

        mom_5m = 0.0
        if len(closes) >= 6 and closes[-6] > 0:
            mom_5m = (closes[-1] - closes[-6]) / closes[-6]
        mom_15m = 0.0
        if len(closes) >= 16 and closes[-16] > 0:
            mom_15m = (closes[-1] - closes[-16]) / closes[-16]

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

        vwap_60m = (vwap_numerator / vwap_denominator) if vwap_denominator > 0.0 else 0.0
        if vwap_60m > 0.0 and last_price > 0.0:
            vwap_bias = (last_price - vwap_60m) / vwap_60m
            futures["futures_vwap_bias_norm"] = min(max((vwap_bias + 0.05) / 0.10, 0.0), 1.0)

        volume_30m = sum(v for v in volumes[-30:] if v > 0.0)
        volume_zscore = 0.0
        vol_tail = [v for v in volumes[-30:] if v >= 0.0]
        if len(vol_tail) >= 5:
            vol_mean = sum(vol_tail) / max(len(vol_tail), 1)
            vol_var = sum((v - vol_mean) ** 2 for v in vol_tail) / max(len(vol_tail) - 1, 1)
            vol_std = vol_var ** 0.5
            if vol_std > 0.0:
                volume_zscore = (vol_tail[-1] - vol_mean) / vol_std
        futures["futures_session_volume_profile_norm"] = min(max((volume_zscore + 2.0) / 4.0, 0.0), 1.0)
        queue_depth = max(float(futures.get("futures_bid_size", 0.0)), 0.0) + max(float(futures.get("futures_ask_size", 0.0)), 0.0)

        out = {
            "last_price": float(last_price),
            "prev_close": float(prev_close),
            "pct_from_close": float(pct_from_close),
            "vol_30m": float(vol_30m),
            "volatility_1m": float(volatility_1m),
            "return_1m": float(return_1m),
            "mom_5m": float(mom_5m),
            "mom_15m": float(mom_15m),
            "range_pos": float(range_pos),
            "vwap_60m": float(vwap_60m),
            "volume_30m": float(volume_30m),
            "volume_zscore": float(volume_zscore),
            "spread_bps": float(futures.get("futures_spread_bps", 0.0)),
            "bid_size": float(futures.get("futures_bid_size", 0.0)),
            "ask_size": float(futures.get("futures_ask_size", 0.0)),
            "queue_depth": float(queue_depth),
            "queue_depth_norm": float(futures.get("futures_depth_ratio_norm", 0.0)),
            "snapshot_transport": "websocket" if (use_ws_ticker and use_ws_book) else ("hybrid" if (use_ws_ticker or use_ws_book) else "rest"),
            "snapshot_ts_utc": time.time(),
        }
        out.update(futures)
        return out
