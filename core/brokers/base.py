from __future__ import annotations

import os
import re
from typing import Any, Dict, List, Optional, Tuple

from core.brokers.models import (
    BrokerAuthRequest,
    BrokerCapabilities,
    BrokerConnectedAccount,
    BrokerCredentials,
    BrokerQuoteSnapshot,
)

BrokerCallSpec = Tuple[str, Tuple[Any, ...], Dict[str, Any]]


class BrokerAdapter:
    name = "generic"
    display_name = "Generic Broker"
    capabilities = BrokerCapabilities()

    api_key_env_var = ""
    app_secret_env_var = ""
    callback_url_env_var = ""
    requested_browser_env_var = ""
    interactive_env_var = ""
    callback_timeout_env_var = ""
    max_token_age_env_var = ""
    account_reference_env_var = ""
    account_reference_auto_discover_env_var = ""
    options_chain_strike_count_env_var = ""

    placeholder_api_key = "YOUR_KEY_HERE"
    placeholder_app_secret = "YOUR_SECRET_HERE"
    placeholder_callback_url = "https://127.0.0.1:8182"

    def load_credentials_from_env(self) -> BrokerCredentials:
        return BrokerCredentials(
            api_key=(os.getenv(self.api_key_env_var, self.placeholder_api_key).strip() if self.api_key_env_var else self.placeholder_api_key),
            app_secret=(os.getenv(self.app_secret_env_var, self.placeholder_app_secret).strip() if self.app_secret_env_var else self.placeholder_app_secret),
            callback_url=(os.getenv(self.callback_url_env_var, self.placeholder_callback_url).strip() if self.callback_url_env_var else self.placeholder_callback_url),
        )

    def is_placeholder_credentials(self, credentials: BrokerCredentials) -> bool:
        if not self.capabilities.requires_auth:
            return False
        return (
            str(credentials.api_key or "").strip() in {"", self.placeholder_api_key}
            or str(credentials.app_secret or "").strip() in {"", self.placeholder_app_secret}
        )

    def authenticate(self, auth_request: BrokerAuthRequest) -> Any:
        raise RuntimeError(f"{self.name}_broker_auth_not_implemented")

    def account_numbers_candidates(self) -> List[BrokerCallSpec]:
        return []

    def accounts_snapshot_candidates(self, *, account_reference: str, allow_global_fallback: bool) -> List[BrokerCallSpec]:
        return []

    def place_order_candidates(self, *, account_reference: str, order_spec: Dict[str, Any]) -> List[BrokerCallSpec]:
        return []

    def replace_order_candidates(self, *, account_reference: str, order_id: str, order_spec: Dict[str, Any]) -> List[BrokerCallSpec]:
        return []

    def cancel_order_candidates(self, *, account_reference: str, order_id: str) -> List[BrokerCallSpec]:
        return []

    def fetch_order_candidates(self, *, account_reference: str, order_id: str) -> List[BrokerCallSpec]:
        return []

    def quote_candidates(self, *, symbol: str) -> List[BrokerCallSpec]:
        return []

    def option_chain_candidates(self, *, symbol: str, strike_count: int) -> List[BrokerCallSpec]:
        return []

    def news_candidates(self, *, symbol: str, limit: int) -> List[BrokerCallSpec]:
        return []

    def calendar_candidates(self, *, symbol: str, days_ahead: int) -> List[BrokerCallSpec]:
        return []

    def position_candidates(self, *, account_reference: str) -> List[BrokerCallSpec]:
        return self.accounts_snapshot_candidates(
            account_reference=account_reference,
            allow_global_fallback=True,
        )

    def extract_connected_accounts(self, payload: Any) -> List[BrokerConnectedAccount]:
        rows: List[BrokerConnectedAccount] = []
        if not isinstance(payload, list):
            return rows
        for item in payload:
            if not isinstance(item, dict):
                continue
            account_reference = str(item.get("hashValue") or item.get("account_hash") or "").strip()
            if not account_reference:
                continue
            rows.append(
                BrokerConnectedAccount(
                    account_number=str(item.get("accountNumber") or item.get("account_number") or "").strip(),
                    account_reference=account_reference,
                )
            )
        return rows

    def extract_account_rows(self, payload: Any) -> List[Dict[str, str]]:
        return [row.to_dict() for row in self.extract_connected_accounts(payload)]

    def extract_quote_payload(self, raw: Any, symbol: str) -> Dict[str, Any]:
        if not isinstance(raw, dict):
            return {}
        sym = str(symbol or "").strip().upper()
        if sym in raw and isinstance(raw[sym], dict):
            return raw[sym]

        normalized = re.sub(r"[^A-Z0-9]", "", sym)
        for key, value in raw.items():
            if not isinstance(key, str) or not isinstance(value, dict):
                continue
            if key.upper() == sym:
                return value
            if re.sub(r"[^A-Z0-9]", "", key.upper()) == normalized:
                return value

        dict_children = [value for value in raw.values() if isinstance(value, dict)]
        if len(dict_children) == 1:
            return dict_children[0]
        return {}

    def parse_quote_snapshot(self, symbol: str, raw: Any) -> BrokerQuoteSnapshot:
        payload = raw if isinstance(raw, dict) else {}
        quote_payload = self.extract_quote_payload(payload, symbol)

        def _as_float(value: Any) -> float:
            try:
                return float(value)
            except Exception:
                return 0.0

        def _field(keys: Tuple[str, ...]) -> float:
            containers: List[Dict[str, Any]] = []
            if isinstance(quote_payload, dict):
                containers.append(quote_payload)
                for nested in ("quote", "regular", "reference", "extended", "fundamental"):
                    child = quote_payload.get(nested)
                    if isinstance(child, dict):
                        containers.append(child)
            for container in containers:
                for key in keys:
                    if key in container and container.get(key) is not None:
                        value = _as_float(container.get(key))
                        if value > 0.0:
                            return value
            return 0.0

        return BrokerQuoteSnapshot(
            symbol=str(symbol or "").strip().upper(),
            raw_payload=dict(payload),
            quote_payload=dict(quote_payload),
            bid_price=_field(("bidPrice", "bid", "bestBid", "bidPriceInDouble")),
            ask_price=_field(("askPrice", "ask", "bestAsk", "askPriceInDouble")),
            last_price=_field(("lastPrice", "regularMarketLastPrice", "price", "closePrice")),
            mark_price=_field(("mark", "markPrice")),
        )
