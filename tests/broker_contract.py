from __future__ import annotations

from core.brokers import BrokerAdapter, BrokerCapabilities, BrokerCredentials


def assert_broker_adapter_contract(adapter: BrokerAdapter) -> None:
    assert isinstance(adapter.name, str)
    assert adapter.name.strip()
    assert isinstance(adapter.display_name, str)
    assert adapter.display_name.strip()
    assert isinstance(adapter.capabilities, BrokerCapabilities)

    credentials = adapter.load_credentials_from_env()
    assert isinstance(credentials, BrokerCredentials)

    for method_name, args, kwargs in adapter.account_numbers_candidates():
        assert isinstance(method_name, str)
        assert isinstance(args, tuple)
        assert isinstance(kwargs, dict)

    for method_name, args, kwargs in adapter.accounts_snapshot_candidates(
        account_reference="acct-1",
        allow_global_fallback=True,
    ):
        assert isinstance(method_name, str)
        assert isinstance(args, tuple)
        assert isinstance(kwargs, dict)

    for method_name, args, kwargs in adapter.place_order_candidates(
        account_reference="acct-1",
        order_spec={"orderType": "MARKET"},
    ):
        assert isinstance(method_name, str)
        assert isinstance(args, tuple)
        assert isinstance(kwargs, dict)

    for method_name, args, kwargs in adapter.news_candidates(
        symbol="SPY",
        limit=12,
    ):
        assert isinstance(method_name, str)
        assert isinstance(args, tuple)
        assert isinstance(kwargs, dict)

    for method_name, args, kwargs in adapter.calendar_candidates(
        symbol="SPY",
        days_ahead=5,
    ):
        assert isinstance(method_name, str)
        assert isinstance(args, tuple)
        assert isinstance(kwargs, dict)

    quote_snapshot = adapter.parse_quote_snapshot(
        "SPY",
        {"SPY": {"bidPrice": 1.0, "askPrice": 1.1, "lastPrice": 1.05, "mark": 1.04}},
    )
    assert quote_snapshot.symbol == "SPY"
    assert isinstance(adapter.capabilities.supports_news_context, bool)
    assert isinstance(adapter.capabilities.supports_calendar_context, bool)
