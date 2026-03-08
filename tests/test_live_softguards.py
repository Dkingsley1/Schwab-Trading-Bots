import os
from pathlib import Path

from core.base_trader import BaseTrader


def _mk_trader(mode: str = "shadow") -> BaseTrader:
    return BaseTrader("dummy_key", "dummy_secret", "https://127.0.0.1:8182", mode=mode)


def test_extract_all_positions_from_payload_reads_nested_accounts():
    trader = _mk_trader("shadow")
    payload = {
        "accounts": [
            {
                "securitiesAccount": {
                    "positions": [
                        {
                            "instrument": {"symbol": "AAPL", "assetType": "EQUITY"},
                            "longQuantity": 5,
                            "shortQuantity": 0,
                        },
                        {
                            "instrument": {"symbol": "MSFT", "assetType": "EQUITY"},
                            "netQuantity": -2,
                        },
                    ]
                }
            }
        ]
    }

    rows = trader._extract_all_positions_from_payload(payload)
    by_symbol = {r["symbol"]: r for r in rows}

    assert "AAPL" in by_symbol
    assert "MSFT" in by_symbol
    assert by_symbol["AAPL"]["quantity"] == 5.0
    assert by_symbol["MSFT"]["quantity"] == -2.0


def test_extract_open_order_ids_filters_open_statuses():
    trader = _mk_trader("shadow")
    payload = {
        "securitiesAccount": {
            "orderStrategies": [
                {"orderId": "123", "status": "WORKING"},
                {"orderId": "124", "status": "FILLED"},
            ]
        }
    }

    ids = trader._extract_open_order_ids_from_payload(payload)
    assert "123" in ids
    assert "124" not in ids


def test_modify_live_order_blocked_by_operator_stop_env(monkeypatch):
    monkeypatch.setenv("OPERATOR_STOP", "1")
    trader = _mk_trader("live")

    out = trader.modify_live_order(
        order_id="abc123",
        symbol="AAPL",
        action="BUY",
        quantity=1.0,
    )

    assert out.get("ok") is False
    assert out.get("error") == "operator_stop"


def test_operator_stop_flag_path_triggers_softguard(monkeypatch, tmp_path: Path):
    flag = tmp_path / "OPERATOR_STOP.flag"
    flag.write_text("{}", encoding="utf-8")
    monkeypatch.delenv("OPERATOR_STOP", raising=False)
    monkeypatch.setenv("OPERATOR_STOP_FLAG_PATH", str(flag))

    trader = _mk_trader("live")
    assert trader._operator_stop_enabled() is True



class _DummyResponse:
    def __init__(self, status_code: int, payload: dict | None = None):
        self.status_code = int(status_code)
        self._payload = dict(payload or {})
        self.headers = {}

    def json(self):
        return dict(self._payload)


class _FlakyPlaceOrderClient:
    def __init__(self):
        self.calls = 0

    def place_order(self, *args, **kwargs):
        _ = (args, kwargs)
        self.calls += 1
        if self.calls < 3:
            raise RuntimeError("timeout")
        return _DummyResponse(201, {"orderId": "retry-success"})


class _UnauthorizedPlaceOrderClient:
    def __init__(self):
        self.calls = 0

    def place_order(self, *args, **kwargs):
        _ = (args, kwargs)
        self.calls += 1
        return _DummyResponse(401, {"error": "invalid_client"})


def test_live_place_order_retries_transient_failure(monkeypatch):
    monkeypatch.setenv("LIVE_API_RETRY_ATTEMPTS", "4")
    monkeypatch.setenv("LIVE_API_RETRY_BACKOFF_SECONDS", "0")
    monkeypatch.setenv("LIVE_API_RETRY_JITTER_SECONDS", "0")
    trader = _mk_trader("live")
    trader.client = _FlakyPlaceOrderClient()

    order_spec = trader._build_live_order_spec(
        symbol="AAPL",
        action="BUY",
        quantity=1.0,
        limit_price=0.0,
        asset_type="EQUITY",
    )
    out = trader._live_place_order(symbol="AAPL", action="BUY", quantity=1.0, order_spec=order_spec)

    assert out.get("ok") is True
    assert out.get("order_id") == "retry-success"
    assert out.get("attempts_made") == 3
    assert trader.client.calls == 3


def test_live_place_order_does_not_retry_non_retryable_http(monkeypatch):
    monkeypatch.setenv("LIVE_API_RETRY_ATTEMPTS", "4")
    monkeypatch.setenv("LIVE_API_RETRY_BACKOFF_SECONDS", "0")
    monkeypatch.setenv("LIVE_API_RETRY_JITTER_SECONDS", "0")
    trader = _mk_trader("live")
    trader.client = _UnauthorizedPlaceOrderClient()

    order_spec = trader._build_live_order_spec(
        symbol="AAPL",
        action="BUY",
        quantity=1.0,
        limit_price=0.0,
        asset_type="EQUITY",
    )
    out = trader._live_place_order(symbol="AAPL", action="BUY", quantity=1.0, order_spec=order_spec)

    assert out.get("ok") is False
    assert "http_status_401" in str(out.get("error", ""))
    assert out.get("attempts_made") == 1
    assert trader.client.calls == 1


def test_paper_execute_uses_guard_and_fill_modeling(monkeypatch):
    monkeypatch.setenv("ALLOW_ORDER_EXECUTION", "1")
    monkeypatch.setenv("MARKET_DATA_ONLY", "0")

    trader = _mk_trader("paper")
    trader.execution_enabled = True
    trader.market_data_only = False

    out = trader.execute_decision(
        symbol="AAPL",
        action="BUY",
        quantity=1.0,
        model_score=0.62,
        threshold=0.55,
        features={"last_price": 100.0, "volatility_1m": 0.01},
        gates={"model_gate": True},
        reasons=["unit_test"],
        strategy="paper_guard_test",
        metadata={"ask_price": 100.05, "bot_id": "test_bot"},
    )

    assert out.get("status") == "PAPER_EXECUTED"
    assert "paper_fill_model" in out
    assert "order_lifecycle_reconcile" in out
    assert out["order_lifecycle_reconcile"].get("ok") is True


def test_paper_execute_can_block_on_guard(monkeypatch):
    monkeypatch.setenv("ALLOW_ORDER_EXECUTION", "1")
    monkeypatch.setenv("MARKET_DATA_ONLY", "0")

    trader = _mk_trader("paper")
    trader.execution_enabled = True
    trader.market_data_only = False

    out = trader.execute_decision(
        symbol="AAPL",
        action="BUY",
        quantity=1000.0,
        model_score=0.62,
        threshold=0.55,
        features={"last_price": 100.0},
        gates={"model_gate": True},
        reasons=["unit_test"],
        strategy="paper_guard_block_test",
        metadata={"ask_price": 100.05, "bot_id": "test_bot"},
    )

    assert out.get("status") == "PAPER_GUARD_BLOCKED"
    assert out.get("live_guard_decision", {}).get("gate") in {"position_limit", "order_notional_limit"}



def test_pretrade_reconcile_allows_manual_adjustment_and_syncs_local(monkeypatch):
    monkeypatch.setenv("LIVE_PRETRADE_RECONCILE_BLOCK_ON_MISMATCH", "1")
    monkeypatch.setenv("LIVE_PRETRADE_RECONCILE_SYNC_LOCAL", "0")
    monkeypatch.setenv("LIVE_MANUAL_TRADE_AWARE_ENABLED", "1")
    monkeypatch.setenv("LIVE_MANUAL_TRADE_QTY_TOLERANCE", "2.0")
    monkeypatch.setenv("LIVE_MANUAL_TRADE_AUTO_SYNC_LOCAL", "1")

    trader = _mk_trader("live")
    trader.live_guard.set_local_position(symbol="AAPL", quantity=0.0, avg_price=0.0)

    def _fake_fetch(*, symbol: str):
        return {"ok": True, "symbol": symbol.upper(), "broker_qty": 1.0}

    trader._live_fetch_broker_position = _fake_fetch
    out = trader._pre_trade_reconcile_before_order(symbol="AAPL")

    assert out.get("ok") is True
    assert out.get("reason") == "manual_adjustment_detected"
    details = out.get("details", {})
    assert details.get("manual_adjustment_detected") is True
    assert details.get("synced_local_position") is True
    assert float(details.get("local_qty_after_sync", 0.0)) == 1.0


def test_pretrade_reconcile_blocks_true_mismatch_when_manual_awareness_disabled(monkeypatch):
    monkeypatch.setenv("LIVE_PRETRADE_RECONCILE_BLOCK_ON_MISMATCH", "1")
    monkeypatch.setenv("LIVE_MANUAL_TRADE_AWARE_ENABLED", "0")
    monkeypatch.setenv("LIVE_MANUAL_TRADE_QTY_TOLERANCE", "2.0")

    trader = _mk_trader("live")
    trader.live_guard.set_local_position(symbol="AAPL", quantity=0.0, avg_price=0.0)

    def _fake_fetch(*, symbol: str):
        return {"ok": True, "symbol": symbol.upper(), "broker_qty": 1.0}

    trader._live_fetch_broker_position = _fake_fetch
    out = trader._pre_trade_reconcile_before_order(symbol="AAPL")

    assert out.get("ok") is False
    assert out.get("reason") == "position_mismatch"
