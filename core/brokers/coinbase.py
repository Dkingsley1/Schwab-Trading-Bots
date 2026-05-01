from __future__ import annotations

import os

from core.coinbase_market_data import CoinbaseMarketDataClient
from core.brokers.base import BrokerAdapter
from core.brokers.models import BrokerAuthRequest, BrokerCapabilities


class CoinbaseBrokerAdapter(BrokerAdapter):
    name = "coinbase"
    display_name = "Coinbase"
    capabilities = BrokerCapabilities(
        requires_auth=False,
        supports_market_data=True,
        supports_live_execution=False,
        supports_paper_execution=False,
        supports_account_discovery=False,
        supports_account_snapshot=False,
        supports_positions=False,
        supports_order_place=False,
        supports_order_replace=False,
        supports_order_cancel=False,
        supports_order_fetch=False,
        supports_options=False,
        supports_futures=True,
    )

    api_key_env_var = "COINBASE_API_KEY"
    app_secret_env_var = "COINBASE_API_SECRET"
    callback_url_env_var = "COINBASE_REDIRECT"
    placeholder_callback_url = ""

    def authenticate(self, auth_request: BrokerAuthRequest):
        _ = auth_request
        return CoinbaseMarketDataClient(
            timeout_seconds=max(float(os.getenv("COINBASE_TIMEOUT_SECONDS", "8") or 8), 1.0)
        )
