from __future__ import annotations

import itertools
from typing import Any, Dict, List

from core.brokers.base import BrokerAdapter, BrokerCallSpec
from core.brokers.models import BrokerAuthRequest, BrokerCapabilities, BrokerCredentials


class _MockResponse:
    def __init__(self, status_code: int, payload: Any) -> None:
        self.status_code = int(status_code)
        self._payload = payload
        self.headers: Dict[str, str] = {}

    def json(self) -> Any:
        if isinstance(self._payload, dict):
            return dict(self._payload)
        if isinstance(self._payload, list):
            return list(self._payload)
        return self._payload


class MockBrokerClient:
    _id_counter = itertools.count(1)

    def __init__(self) -> None:
        self._positions = [{"instrument": {"symbol": "MOCK", "assetType": "EQUITY"}, "netQuantity": 0.0}]
        self._orders: Dict[str, Dict[str, Any]] = {}

    def get_account_numbers(self) -> _MockResponse:
        return _MockResponse(200, [{"accountNumber": "000000001", "hashValue": "mock-account"}])

    def get_account(self, *args, **kwargs) -> _MockResponse:
        _ = (args, kwargs)
        return _MockResponse(
            200,
            {
                "securitiesAccount": {
                    "positions": list(self._positions),
                    "orderStrategies": list(self._orders.values()),
                }
            },
        )

    def get_accounts(self, *args, **kwargs) -> _MockResponse:
        _ = (args, kwargs)
        return self.get_account()

    def get_quote(self, symbol: str) -> _MockResponse:
        sym = str(symbol or "").strip().upper() or "MOCK"
        payload = {
            sym: {
                "symbol": sym,
                "bidPrice": 99.9,
                "askPrice": 100.1,
                "lastPrice": 100.0,
                "mark": 100.0,
                "closePrice": 99.5,
            }
        }
        return _MockResponse(200, payload)

    def place_order(self, *args, **kwargs) -> _MockResponse:
        order_spec = args[-1] if args else kwargs.get("order_spec") or {}
        order_id = f"mock-order-{next(self._id_counter)}"
        payload = {
            "orderId": order_id,
            "status": "ACCEPTED",
            "orderLegCollection": list(order_spec.get("orderLegCollection", [])),
        }
        self._orders[order_id] = dict(payload)
        return _MockResponse(201, payload)

    def replace_order(self, *args, **kwargs) -> _MockResponse:
        order_id = str(args[-2] if len(args) >= 2 else kwargs.get("order_id") or "").strip()
        order_spec = args[-1] if args else kwargs.get("order_spec") or {}
        payload = {
            "orderId": order_id or f"mock-order-{next(self._id_counter)}",
            "status": "REPLACED",
            "orderLegCollection": list(order_spec.get("orderLegCollection", [])),
        }
        self._orders[payload["orderId"]] = dict(payload)
        return _MockResponse(200, payload)

    def cancel_order(self, *args, **kwargs) -> _MockResponse:
        order_id = str(args[-1] if args else kwargs.get("order_id") or "").strip()
        payload = {"orderId": order_id, "status": "CANCELED"}
        self._orders[order_id] = dict(payload)
        return _MockResponse(200, payload)

    def get_order(self, *args, **kwargs) -> _MockResponse:
        order_id = str(args[-1] if args else kwargs.get("order_id") or "").strip()
        payload = dict(self._orders.get(order_id) or {"orderId": order_id, "status": "FILLED"})
        return _MockResponse(200, payload)


class MockBrokerAdapter(BrokerAdapter):
    name = "mock"
    display_name = "Mock Broker"
    capabilities = BrokerCapabilities(
        requires_auth=False,
        supports_market_data=True,
        supports_live_execution=True,
        supports_paper_execution=True,
        supports_account_discovery=True,
        supports_account_snapshot=True,
        supports_positions=True,
        supports_order_place=True,
        supports_order_replace=True,
        supports_order_cancel=True,
        supports_order_fetch=True,
        supports_options=False,
        supports_futures=False,
    )

    api_key_env_var = "MOCK_BROKER_API_KEY"
    app_secret_env_var = "MOCK_BROKER_API_SECRET"
    callback_url_env_var = "MOCK_BROKER_CALLBACK_URL"
    placeholder_api_key = ""
    placeholder_app_secret = ""
    placeholder_callback_url = ""

    def is_placeholder_credentials(self, credentials: BrokerCredentials) -> bool:
        _ = credentials
        return False

    def authenticate(self, auth_request: BrokerAuthRequest) -> Any:
        _ = auth_request
        return MockBrokerClient()

    def account_numbers_candidates(self) -> List[BrokerCallSpec]:
        return [("get_account_numbers", tuple(), {})]

    def accounts_snapshot_candidates(self, *, account_reference: str, allow_global_fallback: bool) -> List[BrokerCallSpec]:
        _ = (account_reference, allow_global_fallback)
        return [("get_account", tuple(), {}), ("get_accounts", tuple(), {})]

    def place_order_candidates(self, *, account_reference: str, order_spec: Dict[str, Any]) -> List[BrokerCallSpec]:
        _ = account_reference
        return [("place_order", (order_spec,), {})]

    def replace_order_candidates(self, *, account_reference: str, order_id: str, order_spec: Dict[str, Any]) -> List[BrokerCallSpec]:
        _ = account_reference
        return [("replace_order", (order_id, order_spec), {})]

    def cancel_order_candidates(self, *, account_reference: str, order_id: str) -> List[BrokerCallSpec]:
        _ = account_reference
        return [("cancel_order", (order_id,), {})]

    def fetch_order_candidates(self, *, account_reference: str, order_id: str) -> List[BrokerCallSpec]:
        _ = account_reference
        return [("get_order", (order_id,), {})]

    def quote_candidates(self, *, symbol: str) -> List[BrokerCallSpec]:
        return [("get_quote", (str(symbol or "").strip().upper(),), {})]
