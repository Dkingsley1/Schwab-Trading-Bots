from __future__ import annotations

import json
from pathlib import Path

from core.brokers.shared_rate_limiter import (
    acquire_broker_rate_limit,
    broker_operation_pool,
)


def _policy(root: Path) -> None:
    path = root / "config" / "broker_capability_contracts_v1.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "brokers": {
                    "schwab": {
                        "rate_limit_pools": {
                            "market_data": 2,
                            "accounts": 2,
                            "orders": 1,
                        }
                    }
                }
            }
        ),
        encoding="utf-8",
    )


def test_operation_pool_classification() -> None:
    assert broker_operation_pool("get_quote") == "market_data"
    assert broker_operation_pool("get_accounts_snapshot") == "accounts"
    assert broker_operation_pool("place_order") == "orders"


def test_shared_rate_limit_blocks_cross_process_budget_exhaustion(
    tmp_path: Path,
) -> None:
    _policy(tmp_path)
    database = tmp_path / "limits.sqlite3"
    env = {"BROKER_SHARED_RATE_LIMIT_ENABLED": "1"}

    first = acquire_broker_rate_limit(
        project_root=tmp_path,
        broker="schwab",
        operation="place_order",
        env=env,
        now_epoch=1000.0,
        database_path=database,
    )
    second = acquire_broker_rate_limit(
        project_root=tmp_path,
        broker="schwab",
        operation="cancel_order",
        env=env,
        now_epoch=1001.0,
        database_path=database,
    )

    assert first["allowed"] is True
    assert first["remaining"] == 0
    assert second["allowed"] is False
    assert second["retry_after_seconds"] > 0.0


def test_shared_rate_limit_resets_on_next_window(tmp_path: Path) -> None:
    _policy(tmp_path)
    database = tmp_path / "limits.sqlite3"
    env = {"BROKER_SHARED_RATE_LIMIT_ENABLED": "1"}
    acquire_broker_rate_limit(
        project_root=tmp_path,
        broker="schwab",
        operation="place_order",
        env=env,
        now_epoch=1000.0,
        database_path=database,
    )

    next_window = acquire_broker_rate_limit(
        project_root=tmp_path,
        broker="schwab",
        operation="place_order",
        env=env,
        now_epoch=1080.0,
        database_path=database,
    )

    assert next_window["allowed"] is True
    assert next_window["used"] == 1
