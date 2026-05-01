from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class BrokerCredentials:
    api_key: str = ""
    app_secret: str = ""
    callback_url: str = ""


@dataclass(frozen=True)
class BrokerCapabilities:
    requires_auth: bool = True
    supports_market_data: bool = True
    supports_live_execution: bool = True
    supports_paper_execution: bool = True
    supports_account_discovery: bool = True
    supports_account_snapshot: bool = True
    supports_positions: bool = True
    supports_order_place: bool = True
    supports_order_replace: bool = True
    supports_order_cancel: bool = True
    supports_order_fetch: bool = True
    supports_options: bool = True
    supports_futures: bool = True
    supports_exotic_derivatives_direct: bool = False
    supports_structured_products_direct: bool = False
    supports_news_context: bool = False
    supports_calendar_context: bool = False


@dataclass(frozen=True)
class BrokerAuthRequest:
    credentials: BrokerCredentials
    token_path: str
    max_token_age: Optional[float]
    callback_timeout: float
    interactive: bool
    requested_browser: Optional[str]


@dataclass(frozen=True)
class BrokerConnectedAccount:
    account_number: str = ""
    account_reference: str = ""

    def to_dict(self) -> Dict[str, str]:
        return {
            "account_number": str(self.account_number or "").strip(),
            "account_hash": str(self.account_reference or "").strip(),
        }


@dataclass(frozen=True)
class BrokerQuoteSnapshot:
    symbol: str
    raw_payload: Dict[str, Any] = field(default_factory=dict)
    quote_payload: Dict[str, Any] = field(default_factory=dict)
    bid_price: float = 0.0
    ask_price: float = 0.0
    last_price: float = 0.0
    mark_price: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": str(self.symbol or "").strip().upper(),
            "raw_payload": dict(self.raw_payload),
            "quote_payload": dict(self.quote_payload),
            "bid_price": float(self.bid_price),
            "ask_price": float(self.ask_price),
            "last_price": float(self.last_price),
            "mark_price": float(self.mark_price),
        }


@dataclass(frozen=True)
class BrokerOrderRequest:
    symbol: str
    action: str
    quantity: float
    order_spec: Dict[str, Any]
    account_reference: str = ""
    asset_type: str = "EQUITY"
    limit_price: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": str(self.symbol or "").strip().upper(),
            "action": str(self.action or "").strip().upper(),
            "quantity": float(self.quantity),
            "order_spec": dict(self.order_spec),
            "account_reference": str(self.account_reference or "").strip(),
            "asset_type": str(self.asset_type or "EQUITY").strip().upper(),
            "limit_price": float(self.limit_price),
        }


@dataclass(frozen=True)
class BrokerOrderResult:
    ok: bool
    order_id: str = ""
    status_code: int = 0
    attempts_made: int = 1
    max_attempts: int = 1
    error: str = ""
    payload: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ok": bool(self.ok),
            "order_id": str(self.order_id or "").strip(),
            "status_code": int(self.status_code),
            "attempts_made": int(self.attempts_made),
            "max_attempts": int(self.max_attempts),
            "error": str(self.error or "").strip(),
            "payload": dict(self.payload),
        }
