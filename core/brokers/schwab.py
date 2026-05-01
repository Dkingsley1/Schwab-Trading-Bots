from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from schwab.auth import easy_client

from core.brokers.base import BrokerAdapter, BrokerCallSpec
from core.brokers.models import BrokerAuthRequest, BrokerCapabilities


class SchwabBrokerAdapter(BrokerAdapter):
    name = "schwab"
    display_name = "Schwab"
    capabilities = BrokerCapabilities(
        requires_auth=True,
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
        supports_options=True,
        supports_futures=True,
        supports_exotic_derivatives_direct=False,
        supports_structured_products_direct=False,
        supports_news_context=True,
        supports_calendar_context=True,
    )

    api_key_env_var = "SCHWAB_API_KEY"
    app_secret_env_var = "SCHWAB_SECRET"
    callback_url_env_var = "SCHWAB_REDIRECT"
    requested_browser_env_var = "SCHWAB_AUTH_REQUESTED_BROWSER"
    interactive_env_var = "SCHWAB_AUTH_INTERACTIVE"
    callback_timeout_env_var = "SCHWAB_AUTH_CALLBACK_TIMEOUT_SECONDS"
    max_token_age_env_var = "SCHWAB_MAX_TOKEN_AGE_SECONDS"
    account_reference_env_var = "SCHWAB_ACCOUNT_HASH"
    account_reference_auto_discover_env_var = "SCHWAB_ACCOUNT_HASH_AUTO_DISCOVER"
    options_chain_strike_count_env_var = "SCHWAB_OPTIONS_CHAIN_STRIKE_COUNT"

    def authenticate(self, auth_request: BrokerAuthRequest) -> Any:
        return easy_client(
            api_key=auth_request.credentials.api_key,
            app_secret=auth_request.credentials.app_secret,
            callback_url=auth_request.credentials.callback_url,
            token_path=auth_request.token_path,
            max_token_age=auth_request.max_token_age,
            callback_timeout=auth_request.callback_timeout,
            interactive=auth_request.interactive,
            requested_browser=auth_request.requested_browser,
        )

    def account_numbers_candidates(self) -> List[BrokerCallSpec]:
        return [("get_account_numbers", tuple(), {})]

    def accounts_snapshot_candidates(self, *, account_reference: str, allow_global_fallback: bool) -> List[BrokerCallSpec]:
        candidates: List[BrokerCallSpec] = []
        account_reference_value = str(account_reference or "").strip()
        if account_reference_value:
            candidates.append(("get_account", (account_reference_value,), {}))
        if (not candidates) or bool(allow_global_fallback):
            candidates.append(("get_accounts", tuple(), {}))
            candidates.append(("get_account", tuple(), {}))
        return candidates

    def place_order_candidates(self, *, account_reference: str, order_spec: Dict[str, Any]) -> List[BrokerCallSpec]:
        candidates: List[BrokerCallSpec] = []
        account_reference_value = str(account_reference or "").strip()
        if account_reference_value:
            candidates.append(("place_order", (account_reference_value, order_spec), {}))
        candidates.append(("place_order", (order_spec,), {}))
        return candidates

    def replace_order_candidates(self, *, account_reference: str, order_id: str, order_spec: Dict[str, Any]) -> List[BrokerCallSpec]:
        candidates: List[BrokerCallSpec] = []
        account_reference_value = str(account_reference or "").strip()
        order_id_value = str(order_id or "").strip()
        if account_reference_value:
            candidates.append(("replace_order", (account_reference_value, order_id_value, order_spec), {}))
        candidates.append(("replace_order", (order_id_value, order_spec), {}))
        return candidates

    def cancel_order_candidates(self, *, account_reference: str, order_id: str) -> List[BrokerCallSpec]:
        candidates: List[BrokerCallSpec] = []
        account_reference_value = str(account_reference or "").strip()
        order_id_value = str(order_id or "").strip()
        if account_reference_value:
            candidates.append(("cancel_order", (account_reference_value, order_id_value), {}))
        candidates.append(("cancel_order", (order_id_value,), {}))
        return candidates

    def fetch_order_candidates(self, *, account_reference: str, order_id: str) -> List[BrokerCallSpec]:
        candidates: List[BrokerCallSpec] = []
        account_reference_value = str(account_reference or "").strip()
        order_id_value = str(order_id or "").strip()
        if account_reference_value:
            candidates.append(("get_order", (account_reference_value, order_id_value), {}))
        candidates.append(("get_order", (order_id_value,), {}))
        return candidates

    def quote_candidates(self, *, symbol: str) -> List[BrokerCallSpec]:
        symbol_value = str(symbol or "").strip().upper()
        return [
            ("get_quote", (symbol_value,), {}),
            ("quote", (symbol_value,), {}),
            ("get_quotes", ((symbol_value,),), {}),
            ("get_quotes", ([symbol_value],), {}),
            ("quotes", ((symbol_value,),), {}),
            ("quotes", ([symbol_value],), {}),
        ]

    def option_chain_candidates(self, *, symbol: str, strike_count: int) -> List[BrokerCallSpec]:
        symbol_value = str(symbol or "").strip().upper()
        arg_sets = [
            ((symbol_value,), {"strike_count": strike_count, "include_quotes": True}),
            ((symbol_value,), {"strike_count": strike_count}),
            ((symbol_value,), {"include_quotes": True}),
            ((symbol_value,), {}),
            ((), {"symbol": symbol_value, "strike_count": strike_count, "include_quotes": True}),
            ((), {"symbol": symbol_value, "strike_count": strike_count}),
            ((), {"symbol": symbol_value, "include_quotes": True}),
            ((), {"symbol": symbol_value}),
        ]
        out: List[BrokerCallSpec] = []
        for method_name in (
            "get_option_chain",
            "get_options_chain",
            "option_chain",
            "options_chain",
            "get_option_chain_for_symbol",
        ):
            for args, kwargs in arg_sets:
                out.append((method_name, args, dict(kwargs)))
        return out

    def news_candidates(self, *, symbol: str, limit: int) -> List[BrokerCallSpec]:
        symbol_value = str(symbol or "").strip().upper()
        arg_sets = [
            ((symbol_value,), {"limit": limit}),
            ((symbol_value,), {}),
            ((), {"symbol": symbol_value, "limit": limit}),
            ((), {"symbol": symbol_value}),
            ((), {"symbols": symbol_value, "limit": limit}),
            ((), {"symbols": symbol_value}),
        ]
        out: List[BrokerCallSpec] = []
        for method_name in (
            "get_news",
            "get_news_headlines",
            "get_news_for_symbol",
            "get_news_headlines_for_symbol",
            "search_news",
        ):
            for args, kwargs in arg_sets:
                out.append((method_name, args, dict(kwargs)))
        return out

    def calendar_candidates(self, *, symbol: str, days_ahead: int) -> List[BrokerCallSpec]:
        symbol_value = str(symbol or "").strip().upper()
        now_utc = datetime.now(timezone.utc)
        end_utc = now_utc + timedelta(days=max(int(days_ahead), 1))
        arg_sets = [
            ((), {"symbol": symbol_value, "start_datetime": now_utc, "end_datetime": end_utc}),
            ((), {"symbols": symbol_value, "start_datetime": now_utc, "end_datetime": end_utc}),
            ((), {"symbol": symbol_value}),
            ((), {"symbols": symbol_value}),
            ((), {"start_datetime": now_utc, "end_datetime": end_utc}),
            ((symbol_value,), {}),
            ((), {}),
        ]
        out: List[BrokerCallSpec] = []
        for method_name in (
            "get_market_calendar",
            "get_calendar",
            "get_events",
            "get_market_events",
            "get_economic_calendar",
        ):
            for args, kwargs in arg_sets:
                out.append((method_name, args, dict(kwargs)))
        return out
