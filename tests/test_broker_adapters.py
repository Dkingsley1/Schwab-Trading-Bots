import pytest

from core.base_trader import BaseTrader
from core.brokers import (
    BrokerRuntimeConfig,
    available_broker_names,
    available_broker_names_for_role,
    build_broker_adapter,
)
from tests.broker_contract import assert_broker_adapter_contract


def test_build_broker_adapter_rejects_unknown_broker():
    with pytest.raises(ValueError):
        build_broker_adapter("not-a-real-broker")


def test_all_registered_adapters_satisfy_contract():
    for broker_name in available_broker_names():
        assert_broker_adapter_contract(build_broker_adapter(broker_name))


def test_base_trader_from_env_loads_schwab_adapter(monkeypatch):
    monkeypatch.setenv("DATA_BROKER", "schwab")
    monkeypatch.setenv("SCHWAB_API_KEY", "live_key")
    monkeypatch.setenv("SCHWAB_SECRET", "live_secret")
    monkeypatch.setenv("SCHWAB_REDIRECT", "https://127.0.0.1:9999/callback")

    trader = BaseTrader.from_env(mode="shadow")

    assert "schwab" in available_broker_names()
    assert trader.broker_name == "schwab"
    assert trader.broker_display_name == "Schwab"
    assert trader.api_key == "live_key"
    assert trader.app_secret == "live_secret"
    assert trader.callback_url == "https://127.0.0.1:9999/callback"
    assert trader.credentials_are_placeholder() is False
    assert trader.shadow_domain == "equities"


def test_runtime_config_resolves_role_specific_brokers(monkeypatch):
    monkeypatch.setenv("DATA_BROKER", "schwab")
    monkeypatch.setenv("MARKET_DATA_PROVIDER", "coinbase")
    monkeypatch.setenv("PAPER_EXECUTION_BROKER", "mock")
    monkeypatch.setenv("LIVE_EXECUTION_BROKER", "schwab")
    monkeypatch.setenv("AUTH_BROKER", "schwab")

    runtime = BrokerRuntimeConfig.from_env()

    assert runtime.market_data_provider_name == "coinbase"
    assert runtime.paper_execution_broker_name == "mock"
    assert runtime.execution_broker_name == "schwab"
    assert runtime.broker_for_role("market_data") == "coinbase"
    assert runtime.broker_for_role("paper") == "mock"
    assert "mock" in available_broker_names_for_role("paper")
    assert "coinbase" in available_broker_names_for_role("market_data")


def test_capability_roles_cover_universal_news_and_calendar_surfaces():
    assert "schwab" in available_broker_names_for_role("news")
    assert "schwab" in available_broker_names_for_role("calendar")
    assert "coinbase" not in available_broker_names_for_role("news")


def test_schwab_adapter_exposes_universal_news_and_calendar_candidates():
    adapter = build_broker_adapter("schwab")

    news_candidates = adapter.news_candidates(symbol="SPY", limit=12)
    calendar_candidates = adapter.calendar_candidates(symbol="SPY", days_ahead=5)

    assert any(method_name == "get_news" for method_name, _args, _kwargs in news_candidates)
    assert any(method_name == "get_market_calendar" for method_name, _args, _kwargs in calendar_candidates)


def test_base_trader_from_env_can_target_coinbase_without_changing_data_broker_env(monkeypatch):
    monkeypatch.setenv("DATA_BROKER", "schwab")
    monkeypatch.setenv("COINBASE_API_KEY", "coinbase_key")
    monkeypatch.setenv("COINBASE_API_SECRET", "coinbase_secret")

    trader = BaseTrader.from_env(mode="shadow", broker="coinbase")

    assert "coinbase" in available_broker_names()
    assert trader.broker_name == "coinbase"
    assert trader.broker_display_name == "Coinbase"
    assert trader.api_key == "coinbase_key"
    assert trader.app_secret == "coinbase_secret"
    assert trader.credentials_are_placeholder() is False
    assert trader.shadow_domain == "crypto"
    assert trader.mode_label == "shadow_crypto"


def test_base_trader_role_selection_and_mock_adapter_flow(monkeypatch):
    monkeypatch.setenv("DATA_BROKER", "schwab")
    monkeypatch.setenv("MARKET_DATA_PROVIDER", "coinbase")
    monkeypatch.setenv("PAPER_EXECUTION_BROKER", "mock")

    trader = BaseTrader.from_env(mode="paper", role="paper")
    trader.client = trader.authenticate()

    assert trader.broker_name == "mock"
    assert trader.credentials_are_placeholder() is False

    accounts = trader.fetch_connected_account_rows()
    assert accounts[0]["account_hash"] == "mock-account"

    order_spec = trader._build_live_single_order_spec(
        symbol="MOCK",
        action="BUY",
        quantity=1.0,
        limit_price=0.0,
        asset_type="EQUITY",
    )
    placed = trader._live_place_order(
        symbol="MOCK",
        action="BUY",
        quantity=1.0,
        order_spec=order_spec,
    )
    assert placed["ok"] is True
    assert placed["order_result"]["ok"] is True
    assert placed["order_request"]["account_reference"] == "mock-account"
