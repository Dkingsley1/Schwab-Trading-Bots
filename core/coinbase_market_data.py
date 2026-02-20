import json
import time
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional


class CoinbaseMarketDataClient:
    """Public Coinbase market-data client (no trading endpoints)."""

    def __init__(self, timeout_seconds: float = 8.0):
        self.base_url = "https://api.exchange.coinbase.com"
        self.timeout_seconds = float(timeout_seconds)

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

    def _get_json(self, path: str, params: Optional[Dict[str, Any]] = None) -> Any:
        query = ""
        if params:
            query = "?" + urllib.parse.urlencode(params)
        url = f"{self.base_url}{path}{query}"
        req = urllib.request.Request(url, headers={"User-Agent": "schwab-trading-bot/coinbase-data"})
        with urllib.request.urlopen(req, timeout=self.timeout_seconds) as resp:
            data = resp.read()
        return json.loads(data.decode("utf-8"))

    def get_ticker(self, symbol: str) -> Dict[str, Any]:
        product_id = self.normalize_symbol(symbol)
        return self._get_json(f"/products/{product_id}/ticker")

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
        )
        if not isinstance(out, list):
            return []
        # Coinbase returns newest->oldest; sort ascending by time.
        out = sorted(out, key=lambda row: float(row[0]) if isinstance(row, list) and row else 0.0)
        return out

    def market_snapshot(self, symbol: str) -> Dict[str, float]:
        ticker = self.get_ticker(symbol)
        candles = self.get_candles(symbol, minutes=60, granularity=60)

        closes: List[float] = []
        highs: List[float] = []
        lows: List[float] = []
        for c in candles:
            if not isinstance(c, list) or len(c) < 5:
                continue
            # [time, low, high, open, close, volume]
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

        return {
            "last_price": float(last_price),
            "prev_close": float(prev_close),
            "pct_from_close": float(pct_from_close),
            "vol_30m": float(vol_30m),
            "mom_5m": float(mom_5m),
            "range_pos": float(range_pos),
        }
