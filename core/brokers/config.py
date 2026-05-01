from __future__ import annotations

import os
from dataclasses import dataclass

from core.brokers.registry import normalize_broker_name


@dataclass(frozen=True)
class BrokerRuntimeConfig:
    default_broker_name: str
    market_data_provider_name: str
    execution_broker_name: str
    paper_execution_broker_name: str
    auth_broker_name: str

    @classmethod
    def from_env(cls, *, default_broker: str = "schwab") -> "BrokerRuntimeConfig":
        default_name = normalize_broker_name(default_broker or os.getenv("DATA_BROKER", "schwab"))
        market_data_provider = normalize_broker_name(
            os.getenv(
                "MARKET_DATA_PROVIDER",
                os.getenv("SHADOW_MARKET_DATA_PROVIDER", default_name),
            )
        )
        execution_broker = normalize_broker_name(
            os.getenv(
                "LIVE_EXECUTION_BROKER",
                os.getenv("EXECUTION_BROKER", default_name),
            )
        )
        paper_execution_broker = normalize_broker_name(
            os.getenv("PAPER_EXECUTION_BROKER", execution_broker)
        )
        auth_broker = normalize_broker_name(
            os.getenv("AUTH_BROKER", execution_broker or default_name)
        )
        return cls(
            default_broker_name=default_name,
            market_data_provider_name=market_data_provider,
            execution_broker_name=execution_broker,
            paper_execution_broker_name=paper_execution_broker,
            auth_broker_name=auth_broker,
        )

    def broker_for_role(self, role: str = "default") -> str:
        normalized_role = str(role or "default").strip().lower()
        if normalized_role in {"market", "market_data", "market-data", "data"}:
            return self.market_data_provider_name
        if normalized_role in {"execution", "live", "live_execution"}:
            return self.execution_broker_name
        if normalized_role in {"paper", "paper_execution"}:
            return self.paper_execution_broker_name
        if normalized_role in {"auth", "authentication"}:
            return self.auth_broker_name
        return self.default_broker_name

    def to_dict(self) -> dict[str, str]:
        return {
            "default_broker_name": self.default_broker_name,
            "market_data_provider_name": self.market_data_provider_name,
            "execution_broker_name": self.execution_broker_name,
            "paper_execution_broker_name": self.paper_execution_broker_name,
            "auth_broker_name": self.auth_broker_name,
        }
