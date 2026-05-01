import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.ops import coinbase_api_health as src


class FakeCoinbaseClient:
    base_url = "https://api.exchange.coinbase.com"

    def __init__(self, timeout_seconds: float):
        self.timeout_seconds = timeout_seconds

    def normalize_symbol(self, symbol: str) -> str:
        return str(symbol or "").upper()

    def get_product(self, product_id: str) -> dict:
        return {"id": product_id, "status": "online", "base_currency": product_id.split("-", 1)[0]}

    def get_ticker(self, product_id: str) -> dict:
        return {"product_id": product_id, "price": "65000.00", "bid": "64999.00", "ask": "65001.00"}

    def market_snapshot(self, product_id: str) -> dict:
        return {"symbol": product_id, "last_price": 65000.0}

    def close(self) -> None:
        pass


class FailingCoinbaseClient(FakeCoinbaseClient):
    def get_product(self, product_id: str) -> dict:
        raise src.MarketDataAPIError(
            provider="coinbase",
            path="/products/BTC-USD",
            symbol=product_id,
            reason="network_unreachable",
            status_code=0,
            attempts=1,
        )


def test_coinbase_api_health_ready_without_printing_secrets(monkeypatch) -> None:
    monkeypatch.setattr(src, "CoinbaseMarketDataClient", FakeCoinbaseClient)
    monkeypatch.setenv("COINBASE_API_KEY", "key-secret-value")
    monkeypatch.setenv("COINBASE_API_SECRET", "secret-value")

    payload = src.build_payload(symbol="btc-usd", timeout_sec=2, snapshot=True)
    rendered = json.dumps(payload, ensure_ascii=True)

    assert payload["overall_status"] == "ready"
    assert payload["public_market_data"]["ok"] is True
    assert payload["credentials"]["api_key_present"] is True
    assert payload["credentials"]["api_secret_present"] is True
    assert "key-secret-value" not in rendered
    assert "secret-value" not in rendered


def test_coinbase_api_health_blocks_when_public_market_data_fails(monkeypatch) -> None:
    monkeypatch.setattr(src, "CoinbaseMarketDataClient", FailingCoinbaseClient)

    payload = src.build_payload(symbol="BTC-USD", timeout_sec=2)

    assert payload["overall_status"] == "blocked"
    assert payload["public_market_data"]["ok"] is False
    assert payload["errors"][0]["step"] == "get_product"
    assert payload["recommended_actions"] == ["check local DNS/network access to api.exchange.coinbase.com"]


def test_coinbase_api_health_degrades_when_auth_is_required_but_missing(monkeypatch) -> None:
    monkeypatch.setattr(src, "CoinbaseMarketDataClient", FakeCoinbaseClient)
    monkeypatch.setenv("COINBASE_REQUIRE_AUTH_CREDS", "1")
    monkeypatch.delenv("COINBASE_API_KEY", raising=False)
    monkeypatch.delenv("COINBASE_API_SECRET", raising=False)

    payload = src.build_payload(symbol="BTC-USD", timeout_sec=2)

    assert payload["overall_status"] == "degraded"
    assert payload["public_market_data"]["ok"] is True
    assert payload["credentials"]["auth_credentials_complete"] is False
