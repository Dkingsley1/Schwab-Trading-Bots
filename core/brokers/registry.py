from __future__ import annotations

from typing import Dict, Tuple, Type

from core.brokers.base import BrokerAdapter
from core.brokers.coinbase import CoinbaseBrokerAdapter
from core.brokers.mock import MockBrokerAdapter
from core.brokers.schwab import SchwabBrokerAdapter


_ADAPTERS: Dict[str, Type[BrokerAdapter]] = {
    "coinbase": CoinbaseBrokerAdapter,
    "mock": MockBrokerAdapter,
    "schwab": SchwabBrokerAdapter,
}

_ROLE_CAPABILITY_MAP = {
    "market": "supports_market_data",
    "market_data": "supports_market_data",
    "market-data": "supports_market_data",
    "data": "supports_market_data",
    "execution": "supports_live_execution",
    "live": "supports_live_execution",
    "live_execution": "supports_live_execution",
    "paper": "supports_paper_execution",
    "paper_execution": "supports_paper_execution",
    "accounts": "supports_account_snapshot",
    "account_snapshot": "supports_account_snapshot",
    "positions": "supports_positions",
    "options": "supports_options",
    "futures": "supports_futures",
    "news": "supports_news_context",
    "news_context": "supports_news_context",
    "calendar": "supports_calendar_context",
    "calendar_context": "supports_calendar_context",
    "auth": "requires_auth",
    "authentication": "requires_auth",
}


def normalize_broker_name(name: str) -> str:
    return str(name or "").strip().lower() or "schwab"


def available_broker_names() -> Tuple[str, ...]:
    return tuple(sorted(_ADAPTERS.keys()))


def available_broker_names_for_role(role: str = "default") -> Tuple[str, ...]:
    normalized_role = str(role or "default").strip().lower()
    capability_name = _ROLE_CAPABILITY_MAP.get(normalized_role)
    if not capability_name:
        return available_broker_names()

    matched = []
    for name, adapter_cls in sorted(_ADAPTERS.items()):
        capabilities = getattr(adapter_cls, "capabilities", None)
        if bool(getattr(capabilities, capability_name, False)):
            matched.append(name)
    return tuple(matched)


def build_broker_adapter(name: str) -> BrokerAdapter:
    broker_name = normalize_broker_name(name)
    adapter_cls = _ADAPTERS.get(broker_name)
    if adapter_cls is None:
        supported = ",".join(sorted(_ADAPTERS.keys()))
        raise ValueError(f"unsupported_broker_adapter:{broker_name}:supported={supported}")
    return adapter_cls()
