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


def test_discover_live_account_hash_populates_hash_from_account_numbers(monkeypatch):
    monkeypatch.delenv("SCHWAB_ACCOUNT_HASH", raising=False)
    trader = _mk_trader("live")
    trader.client = _AccountNumbersClient()

    discovered = trader._discover_live_account_hash(force=True)

    assert discovered == "hash-123"
    assert trader.live_account_hash == "hash-123"
    assert trader.client.get_account_numbers_calls == 1


def test_live_fetch_accounts_payload_prefers_account_hash_endpoint_when_discovered(monkeypatch):
    monkeypatch.delenv("SCHWAB_ACCOUNT_HASH", raising=False)
    monkeypatch.delenv("LIVE_ACCOUNTS_SNAPSHOT_ALLOW_GLOBAL_FALLBACK", raising=False)
    trader = _mk_trader("live")
    trader.client = _AccountNumbersClient()

    out = trader._live_fetch_accounts_payload()

    assert out["ok"] is True
    assert trader.live_account_hash == "hash-123"
    assert trader.client.get_account_numbers_calls == 1
    assert trader.client.get_account_calls == [("hash-123",)]
    assert trader.client.get_accounts_calls == 0



class _DummyResponse:
    def __init__(self, status_code: int, payload: dict | None = None):
        self.status_code = int(status_code)
        self._payload = dict(payload or {})
        self.headers = {}

    def json(self):
        return dict(self._payload)


class _DummyListResponse:
    def __init__(self, status_code: int, payload: list | None = None):
        self.status_code = int(status_code)
        self._payload = list(payload or [])
        self.headers = {}

    def json(self):
        return list(self._payload)


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


class _AccountNumbersClient:
    def __init__(self):
        self.get_account_numbers_calls = 0
        self.get_account_calls = []
        self.get_accounts_calls = 0

    def get_account_numbers(self):
        self.get_account_numbers_calls += 1
        return _DummyListResponse(
            200,
            [
                {
                    "accountNumber": "123456789",
                    "hashValue": "hash-123",
                }
            ],
        )

    def get_account(self, *args, **kwargs):
        _ = kwargs
        self.get_account_calls.append(args)
        return _DummyResponse(200, {"securitiesAccount": {"positions": []}})

    def get_accounts(self, *args, **kwargs):
        _ = (args, kwargs)
        self.get_accounts_calls += 1
        raise AssertionError("global get_accounts fallback should not be used when account hash is discovered")


def _sample_option_chain_payload():
    return {
        "callExpDateMap": {
            "2026-04-17:28": {
                "100.0": [
                    {
                        "symbol": "AAPL_041726C100",
                        "putCall": "CALL",
                        "strikePrice": 100.0,
                        "daysToExpiration": 28,
                        "bidPrice": 3.40,
                        "askPrice": 3.60,
                        "mark": 3.50,
                    }
                ],
                "105.0": [
                    {
                        "symbol": "AAPL_041726C105",
                        "putCall": "CALL",
                        "strikePrice": 105.0,
                        "daysToExpiration": 28,
                        "bidPrice": 1.15,
                        "askPrice": 1.30,
                        "mark": 1.22,
                    }
                ],
            },
            "2026-05-15:56": {
                "100.0": [
                    {
                        "symbol": "AAPL_051526C100",
                        "putCall": "CALL",
                        "strikePrice": 100.0,
                        "daysToExpiration": 56,
                        "bidPrice": 4.80,
                        "askPrice": 5.05,
                        "mark": 4.92,
                    }
                ],
                "105.0": [
                    {
                        "symbol": "AAPL_051526C105",
                        "putCall": "CALL",
                        "strikePrice": 105.0,
                        "daysToExpiration": 56,
                        "bidPrice": 2.45,
                        "askPrice": 2.70,
                        "mark": 2.57,
                    }
                ],
            }
        }
    }


class _OptionsPlaceOrderClient:
    def __init__(self):
        self.placed_specs = []

    def get_option_chain(self, *args, **kwargs):
        _ = (args, kwargs)
        return _sample_option_chain_payload()

    def place_order(self, *args, **kwargs):
        order_spec = args[-1] if args else kwargs.get("order_spec")
        self.placed_specs.append(order_spec)
        return _DummyResponse(201, {})


class _FuturesPlaceOrderClient:
    def __init__(self):
        self.placed_specs = []

    def get_quote(self, symbol):
        mapping = {
            "/ES": {
                "/ES": {
                    "symbol": "/ES",
                    "futureActiveSymbol": "/ESM26",
                    "lastPrice": 5300.0,
                }
            },
            "/ESM26": {
                "/ESM26": {
                    "symbol": "/ESM26",
                    "lastPrice": 5300.0,
                    "expirationDate": "2026-06-19T00:00:00+00:00",
                }
            },
            "/ESU26": {
                "/ESU26": {
                    "symbol": "/ESU26",
                    "lastPrice": 5312.0,
                    "expirationDate": "2026-09-18T00:00:00+00:00",
                }
            },
        }
        payload = mapping.get(str(symbol).upper(), {})
        return _DummyResponse(200, payload)

    def place_order(self, *args, **kwargs):
        order_spec = args[-1] if args else kwargs.get("order_spec")
        self.placed_specs.append(order_spec)
        return _DummyResponse(201, {})


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


def test_build_live_order_spec_supports_multi_leg_options_plan():
    trader = _mk_trader("live")
    trader.client = _OptionsPlaceOrderClient()

    spec = trader._build_live_order_spec(
        symbol="AAPL",
        action="BUY_TO_OPEN",
        quantity=1.0,
        limit_price=0.0,
        asset_type="EQUITY",
        metadata={
            "options_plan": {
                "options_style": "BULL_CALL_DEBIT_SPREAD",
                "strategy_family": "debit_spread",
                "contracts": 1,
                "legs": [
                    {"side": "BUY_TO_OPEN", "type": "CALL", "strike": 100.0, "expiry_days": 28, "quantity": 1},
                    {"side": "SELL_TO_OPEN", "type": "CALL", "strike": 105.0, "expiry_days": 28, "quantity": 1},
                ],
            }
        },
    )

    assert spec.get("orderType") == "NET_DEBIT"
    assert spec.get("complexOrderStrategyType") == "VERTICAL"
    assert len(spec.get("orderLegCollection", [])) == 2
    assert spec["orderLegCollection"][0]["instrument"]["assetType"] == "OPTION"
    assert spec["orderLegCollection"][0]["instrument"]["symbol"] == "AAPL_041726C100"
    assert spec["orderLegCollection"][1]["instrument"]["symbol"] == "AAPL_041726C105"


def test_live_execute_uses_options_plan_order_spec(monkeypatch):
    monkeypatch.setenv("ALLOW_ORDER_EXECUTION", "1")
    monkeypatch.setenv("MARKET_DATA_ONLY", "0")
    monkeypatch.setenv("LIVE_PRETRADE_RECONCILE_REQUIRED", "0")

    trader = _mk_trader("live")
    trader.execution_enabled = True
    trader.market_data_only = False
    trader.client = _OptionsPlaceOrderClient()

    out = trader.execute_decision(
        symbol="AAPL",
        action="BUY_TO_OPEN",
        quantity=1.0,
        model_score=0.71,
        threshold=0.55,
        features={"last_price": 100.0},
        gates={"model_gate": True},
        reasons=["unit_test"],
        strategy="options_live_exec_test",
        metadata={
            "bot_id": "test_bot",
            "options_plan": {
                "options_style": "BULL_CALL_DEBIT_SPREAD",
                "strategy_family": "debit_spread",
                "contracts": 1,
                "legs": [
                    {"side": "BUY_TO_OPEN", "type": "CALL", "strike": 100.0, "expiry_days": 28, "quantity": 1},
                    {"side": "SELL_TO_OPEN", "type": "CALL", "strike": 105.0, "expiry_days": 28, "quantity": 1},
                ],
            },
        },
    )

    assert out.get("status") == "LIVE_ORDER_ACK_NO_ID"
    assert trader.client.placed_specs
    placed = trader.client.placed_specs[-1]
    assert placed.get("complexOrderStrategyType") == "VERTICAL"
    assert placed["orderLegCollection"][0]["instrument"]["assetType"] == "OPTION"


def test_build_live_order_spec_supports_options_roll_plan():
    trader = _mk_trader("live")
    trader.client = _OptionsPlaceOrderClient()

    spec = trader._build_live_order_spec(
        symbol="AAPL",
        action="ROLL",
        quantity=1.0,
        limit_price=0.0,
        asset_type="EQUITY",
        metadata={
            "options_plan": {
                "options_style": "BULL_CALL_DEBIT_SPREAD",
                "strategy_family": "debit_spread",
                "contracts": 1,
                "dte_days": 28,
                "roll_target_dte_days": 56,
                "legs": [
                    {"side": "BUY_TO_OPEN", "type": "CALL", "strike": 100.0, "expiry_days": 28, "quantity": 1},
                    {"side": "SELL_TO_OPEN", "type": "CALL", "strike": 105.0, "expiry_days": 28, "quantity": 1},
                ],
            }
        },
    )

    assert spec.get("complexOrderStrategyType") == "VERTICAL_ROLL"
    assert len(spec.get("orderLegCollection", [])) == 4
    assert spec["orderLegCollection"][0]["instruction"] == "SELL_TO_CLOSE"
    assert spec["orderLegCollection"][1]["instruction"] == "BUY_TO_CLOSE"
    assert spec["orderLegCollection"][2]["instruction"] == "BUY_TO_OPEN"
    assert spec["orderLegCollection"][3]["instruction"] == "SELL_TO_OPEN"
    assert spec["orderLegCollection"][2]["instrument"]["symbol"] == "AAPL_051526C100"
    assert spec["orderLegCollection"][3]["instrument"]["symbol"] == "AAPL_051526C105"


def test_live_execute_uses_futures_plan_order_spec(monkeypatch):
    monkeypatch.setenv("ALLOW_ORDER_EXECUTION", "1")
    monkeypatch.setenv("MARKET_DATA_ONLY", "0")
    monkeypatch.setenv("LIVE_PRETRADE_RECONCILE_REQUIRED", "0")

    trader = _mk_trader("live")
    trader.execution_enabled = True
    trader.market_data_only = False
    trader.client = _FuturesPlaceOrderClient()

    out = trader.execute_decision(
        symbol="/ES",
        action="BUY",
        quantity=1.0,
        model_score=0.66,
        threshold=0.55,
        features={"last_price": 5000.0},
        gates={"model_gate": True},
        reasons=["unit_test"],
        strategy="futures_live_exec_guard_test",
        metadata={
            "bot_id": "test_bot",
            "futures_plan": {
                "futures_style": "FUTURES_BASIS_CARRY_CALENDAR",
                "strategy_family": "calendar",
                "contracts": 1,
                "legs": [
                    {"side": "BUY", "contract": "M1", "quantity": 1, "month_offset": 0},
                    {"side": "SELL", "contract": "M2", "quantity": 1, "month_offset": 1},
                ],
            },
        },
    )

    assert out.get("status") == "LIVE_ORDER_ACK_NO_ID"
    placed = trader.client.placed_specs[-1]
    assert len(placed.get("orderLegCollection", [])) == 2
    assert placed["orderLegCollection"][0]["instrument"]["assetType"] == "FUTURE"
    assert placed["orderLegCollection"][0]["instrument"]["symbol"] == "/ESM26"
    assert placed["orderLegCollection"][1]["instrument"]["symbol"] == "/ESU26"


def test_live_execute_uses_futures_roll_legs(monkeypatch):
    monkeypatch.setenv("ALLOW_ORDER_EXECUTION", "1")
    monkeypatch.setenv("MARKET_DATA_ONLY", "0")
    monkeypatch.setenv("LIVE_PRETRADE_RECONCILE_REQUIRED", "0")

    trader = _mk_trader("live")
    trader.execution_enabled = True
    trader.market_data_only = False
    trader.client = _FuturesPlaceOrderClient()

    out = trader.execute_decision(
        symbol="/ES",
        action="ROLL",
        quantity=1.0,
        model_score=0.66,
        threshold=0.55,
        features={"last_price": 5300.0},
        gates={"model_gate": True},
        reasons=["unit_test"],
        strategy="futures_roll_live_exec_test",
        metadata={
            "bot_id": "test_bot",
            "futures_plan": {
                "futures_style": "FUTURES_TERM_STRUCTURE_ROLL_ROTATION",
                "strategy_family": "calendar_spread",
                "contracts": 1,
                "front_month": "M2",
                "legs": [
                    {"side": "BUY", "contract": "M2", "quantity": 1, "month_offset": 1},
                    {"side": "SELL", "contract": "M1", "quantity": 1, "month_offset": 0},
                ],
                "roll_legs": [
                    {"side": "SELL", "contract": "M1", "quantity": 1, "month_offset": 0},
                    {"side": "BUY", "contract": "M2", "quantity": 1, "month_offset": 1},
                ],
            },
        },
    )

    assert out.get("status") == "LIVE_ORDER_ACK_NO_ID"
    placed = trader.client.placed_specs[-1]
    assert placed["orderLegCollection"][0]["instruction"] == "SELL"
    assert placed["orderLegCollection"][1]["instruction"] == "BUY"
    assert placed["orderLegCollection"][0]["instrument"]["symbol"] == "/ESM26"
    assert placed["orderLegCollection"][1]["instrument"]["symbol"] == "/ESU26"


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
