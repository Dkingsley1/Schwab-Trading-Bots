import json
from pathlib import Path
from types import SimpleNamespace

import core.base_trader as base_src
import core.decision_logger as decision_logger_src
from core.base_trader import BaseTrader


def _mk_trader(mode: str = "shadow") -> BaseTrader:
    return BaseTrader("dummy_key", "dummy_secret", "https://127.0.0.1:8182", mode=mode)


def _reset_base_storage_override_cache() -> None:
    base_src._DYNAMIC_STORAGE_OVERRIDE_CACHE.clear()
    base_src._DYNAMIC_STORAGE_OVERRIDE_CACHE.update(
        {
            "checked_at_monotonic": 0.0,
            "fingerprint": (),
            "values": {},
        }
    )


def _reset_paper_profitability_guard_cache() -> None:
    base_src._PAPER_PROFITABILITY_GUARD_CACHE.clear()
    base_src._PAPER_PROFITABILITY_GUARD_CACHE.update(
        {
            "checked_at_monotonic": 0.0,
            "fingerprint": (),
            "payload": {},
        }
    )


def _allow_production_order_firewall(monkeypatch) -> None:
    monkeypatch.setattr(
        base_src,
        "production_order_firewall_check",
        lambda **_kwargs: SimpleNamespace(ok=True, gate="production_order_firewall", reason="ok", details={}),
    )


def _write_paper_profitability_control(tmp_path: Path, *, weak_profile: str = "default") -> None:
    health = tmp_path / "governance" / "health"
    health.mkdir(parents=True, exist_ok=True)
    payload = {
        "timestamp_utc": "2026-07-13T17:36:52+00:00",
        "controlled_profitability_grade": "A+",
        "raw_profitability_grade": "D",
        "raw_profitability_a_recovery_contract": {
            "active": True,
            "current_raw_profitability_grade": "D",
            "weak_profiles": [weak_profile],
            "runtime_enforcement": {
                "block_new_entries_on_weak_profiles": True,
            },
        },
        "raw_profitability_improvement_contract": {
            "active": True,
            "runtime_enforcement": {
                "block_new_entries_on_weak_profiles": True,
            },
            "weak_sleeve_zero_entry_contract": {
                "profiles": [
                    {
                        "profile": weak_profile,
                        "block_new_entries": True,
                    }
                ],
            },
        },
    }
    (health / "paper_runtime_profitability_controls_latest.json").write_text(
        json.dumps(payload),
        encoding="utf-8",
    )
    _reset_paper_profitability_guard_cache()


def test_execute_decision_can_skip_explanations_via_storage_override(tmp_path: Path, monkeypatch) -> None:
    pressure = tmp_path / "config" / ".env.storage_pressure_override"
    pressure.parent.mkdir(parents=True, exist_ok=True)
    pressure.write_text("LOG_DECISION_EXPLANATIONS=0\n", encoding="utf-8")
    _reset_base_storage_override_cache()

    writes: list[str] = []

    def _capture(path: str, payload: dict, **kwargs) -> bool:
        _ = (payload, kwargs)
        writes.append(path)
        return True

    monkeypatch.setattr(base_src, "safe_append_channel_event", _capture)
    monkeypatch.setattr(decision_logger_src, "safe_append_channel_event", _capture)

    trader = _mk_trader("shadow")
    trader.project_root = str(tmp_path)
    trader.set_mode("shadow")

    out = trader.execute_decision(
        symbol="SPY",
        action="BUY",
        quantity=1.0,
        model_score=0.64,
        threshold=0.55,
        features={"last_price": 100.0},
        gates={"market_data_ok": True, "risk_limit_ok": True},
        reasons=["score_above_threshold"],
        strategy="grand_master_bot",
        metadata={"snapshot_id": "snap-1"},
    )

    assert out["status"] == "DATA_ONLY_BLOCKED"
    assert len(writes) == 1
    assert "trade_decisions_" in writes[0]


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
    monkeypatch.delenv("LIVE_ACCOUNTS_SNAPSHOT_AGGREGATE_CONNECTED", raising=False)
    trader = _mk_trader("live")
    trader.client = _AccountNumbersClient()

    out = trader._live_fetch_accounts_payload()

    assert out["ok"] is True
    assert trader.live_account_hash == "hash-123"
    assert trader.client.get_account_numbers_calls == 1
    assert trader.client.get_account_calls == [("hash-123",)]
    assert str(trader.client.get_account_kwargs[0]["fields"]) == "Fields.POSITIONS"
    assert trader.client.get_accounts_calls == 0


def test_live_fetch_accounts_payload_can_aggregate_connected_accounts(monkeypatch):
    monkeypatch.delenv("SCHWAB_ACCOUNT_HASH", raising=False)
    monkeypatch.setenv("LIVE_ACCOUNTS_SNAPSHOT_AGGREGATE_CONNECTED", "1")
    trader = _mk_trader("live")
    trader.client = _MultiAccountNumbersClient()

    out = trader._live_fetch_accounts_payload()

    assert out["ok"] is True
    assert out["account_snapshot_mode"] == "connected_account_aggregate"
    assert out["account_count"] == 2
    assert trader.client.get_account_numbers_calls == 1
    assert trader.client.get_account_calls == [("hash-111",), ("hash-222",)]
    payload = out["payload"]
    assert payload["account_snapshot_mode"] == "connected_account_aggregate"
    assert len(payload["accounts"]) == 2
    assert payload["accounts"][0]["_broker_account"]["account_number_tail"] == "1111"
    rows = trader._extract_all_positions_from_payload(payload)
    by_symbol = {r["symbol"]: r for r in rows}
    assert by_symbol["AAPL"]["quantity"] == 5.0
    assert by_symbol["MSFT"]["quantity"] == 3.0



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


class _UnauthorizedAccountSnapshotClient:
    def __init__(self):
        self.get_account_calls = 0

    def get_account(self, *args, **kwargs):
        _ = (args, kwargs)
        self.get_account_calls += 1
        return _DummyResponse(401, {"error": "invalid_client"})


class _AccountNumbersClient:
    def __init__(self):
        self.get_account_numbers_calls = 0
        self.get_account_calls = []
        self.get_account_kwargs = []
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
        self.get_account_calls.append(args)
        self.get_account_kwargs.append(kwargs)
        return _DummyResponse(200, {"securitiesAccount": {"positions": []}})

    def get_accounts(self, *args, **kwargs):
        _ = (args, kwargs)
        self.get_accounts_calls += 1
        raise AssertionError("global get_accounts fallback should not be used when account hash is discovered")


class _MultiAccountNumbersClient:
    def __init__(self):
        self.get_account_numbers_calls = 0
        self.get_account_calls = []
        self.get_account_kwargs = []

    def get_account_numbers(self):
        self.get_account_numbers_calls += 1
        return _DummyListResponse(
            200,
            [
                {"accountNumber": "111111111", "hashValue": "hash-111"},
                {"accountNumber": "222222222", "hashValue": "hash-222"},
            ],
        )

    def get_account(self, *args, **kwargs):
        self.get_account_calls.append(args)
        self.get_account_kwargs.append(kwargs)
        symbol = "AAPL" if args and args[0] == "hash-111" else "MSFT"
        qty = 5 if symbol == "AAPL" else 3
        return _DummyResponse(
            200,
            {
                "securitiesAccount": {
                    "positions": [
                        {
                            "instrument": {"symbol": symbol, "assetType": "EQUITY"},
                            "longQuantity": qty,
                            "shortQuantity": 0,
                        }
                    ]
                }
            },
        )


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


def test_live_place_order_retries_transient_failure(monkeypatch, tmp_path: Path):
    _allow_production_order_firewall(monkeypatch)
    monkeypatch.setenv("LIVE_API_RETRY_ATTEMPTS", "4")
    monkeypatch.setenv("LIVE_API_RETRY_BACKOFF_SECONDS", "0")
    monkeypatch.setenv("LIVE_API_RETRY_JITTER_SECONDS", "0")
    trader = _mk_trader("live")
    trader.project_root = str(tmp_path)
    trader.client = _FlakyPlaceOrderClient()

    order_spec = trader._build_live_order_spec(
        symbol="AAPL",
        action="BUY",
        quantity=1.0,
        limit_price=0.0,
        asset_type="EQUITY",
    )
    out = trader._live_place_order(
        symbol="AAPL",
        action="BUY",
        quantity=1.0,
        order_spec=order_spec,
        intent_id="decision-retry-success",
    )

    assert out.get("ok") is True
    assert out.get("order_id") == "retry-success"
    assert out.get("attempts_made") == 3
    assert trader.client.calls == 3


def test_live_place_order_does_not_retry_non_retryable_http(monkeypatch, tmp_path: Path):
    _allow_production_order_firewall(monkeypatch)
    monkeypatch.setenv("LIVE_API_RETRY_ATTEMPTS", "4")
    monkeypatch.setenv("LIVE_API_RETRY_BACKOFF_SECONDS", "0")
    monkeypatch.setenv("LIVE_API_RETRY_JITTER_SECONDS", "0")
    trader = _mk_trader("live")
    trader.project_root = str(tmp_path)
    trader.client = _UnauthorizedPlaceOrderClient()

    order_spec = trader._build_live_order_spec(
        symbol="AAPL",
        action="BUY",
        quantity=1.0,
        limit_price=0.0,
        asset_type="EQUITY",
    )
    out = trader._live_place_order(
        symbol="AAPL",
        action="BUY",
        quantity=1.0,
        order_spec=order_spec,
        intent_id="decision-unauthorized",
    )

    assert out.get("ok") is False
    assert "http_status_401" in str(out.get("error", ""))
    assert out.get("attempts_made") == 1
    assert trader.client.calls == 1
    assert out["durable_order_intent"]["state"] == "rejected"


def test_unknown_broker_submit_halts_and_blocks_unrelated_intents(monkeypatch, tmp_path: Path):
    _allow_production_order_firewall(monkeypatch)
    trader = _mk_trader("live")
    trader.project_root = str(tmp_path)
    trader.global_halt_flag_path = str(tmp_path / "governance" / "health" / "GLOBAL_TRADING_HALT.flag")
    trader.client = _OptionsPlaceOrderClient()
    order_spec = trader._build_live_order_spec(
        symbol="AAPL",
        action="BUY",
        quantity=1.0,
        limit_price=10.0,
        asset_type="EQUITY",
    )

    first = trader._live_place_order(
        symbol="AAPL",
        action="BUY",
        quantity=1.0,
        order_spec=order_spec,
        intent_id="decision-unknown-1",
        reference_price=10.0,
    )
    second = trader._live_place_order(
        symbol="AAPL",
        action="BUY",
        quantity=1.0,
        order_spec=order_spec,
        intent_id="decision-unknown-2",
        reference_price=10.0,
    )

    assert first["ok"] is False
    assert first["error"] == "broker_submit_outcome_unknown"
    assert first["auto_halt"]["ok"] is True
    assert Path(trader.global_halt_flag_path).exists()
    assert second["ok"] is False
    assert second["error"] == "unresolved_broker_operation_requires_reconciliation"
    assert len(trader.client.placed_specs) == 1


def test_account_snapshot_api_circuit_is_debounced_before_global_halt(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("ALLOW_ORDER_EXECUTION", "1")
    monkeypatch.setenv("MARKET_DATA_ONLY", "0")
    monkeypatch.setenv("LIVE_API_FAIL_LIMIT", "1")
    monkeypatch.setenv("LIVE_API_COOLDOWN_SECONDS", "120")
    monkeypatch.setenv("LIVE_API_RETRY_ATTEMPTS", "1")
    monkeypatch.setenv("LIVE_ACCOUNTS_SNAPSHOT_SOFT_FAIL_GRACE", "1")
    monkeypatch.setenv("LIVE_ACCOUNTS_SNAPSHOT_HALT_MIN_FAILURES", "3")
    monkeypatch.setenv("LIVE_SOFTGUARD_AUTO_HALT_ON_API_CIRCUIT", "1")
    monkeypatch.setenv("SCHWAB_ACCOUNT_HASH_AUTO_DISCOVER", "0")

    trader = _mk_trader("live")
    trader.project_root = str(tmp_path)
    trader.global_halt_flag_path = str(tmp_path / "governance" / "health" / "GLOBAL_TRADING_HALT.flag")
    trader.live_account_hash = "hash-123"
    trader.client = _UnauthorizedAccountSnapshotClient()

    first = trader._live_fetch_accounts_payload()
    second = trader._live_fetch_accounts_payload()

    assert first.get("soft_failure") is True
    assert second.get("circuit_opened") is True
    assert not Path(trader.global_halt_flag_path).exists()

    third = trader._live_fetch_accounts_payload()

    assert third.get("error") == "api_circuit_open"
    assert Path(trader.global_halt_flag_path).exists()


def test_account_snapshot_api_circuit_suppresses_global_halt_in_collection_mode(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("ALLOW_ORDER_EXECUTION", "0")
    monkeypatch.setenv("MARKET_DATA_ONLY", "1")
    monkeypatch.setenv("LIVE_API_FAIL_LIMIT", "1")
    monkeypatch.setenv("LIVE_API_COOLDOWN_SECONDS", "120")
    monkeypatch.setenv("LIVE_API_RETRY_ATTEMPTS", "1")
    monkeypatch.setenv("LIVE_ACCOUNTS_SNAPSHOT_SOFT_FAIL_GRACE", "1")
    monkeypatch.setenv("LIVE_ACCOUNTS_SNAPSHOT_HALT_MIN_FAILURES", "2")
    monkeypatch.setenv("LIVE_SOFTGUARD_AUTO_HALT_ON_API_CIRCUIT", "1")
    monkeypatch.setenv("SCHWAB_ACCOUNT_HASH_AUTO_DISCOVER", "0")

    trader = _mk_trader("paper")
    trader.project_root = str(tmp_path)
    trader.global_halt_flag_path = str(tmp_path / "governance" / "health" / "GLOBAL_TRADING_HALT.flag")
    trader.live_account_hash = ""
    trader.client = _UnauthorizedAccountSnapshotClient()

    first = trader._live_fetch_accounts_payload()
    second = trader._live_fetch_accounts_payload()
    third = trader._live_fetch_accounts_payload()

    assert first.get("soft_failure") is True
    assert second.get("circuit_opened") is True
    assert third.get("error") == "api_circuit_open"
    assert not Path(trader.global_halt_flag_path).exists()


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


def test_live_execute_uses_options_plan_order_spec(monkeypatch, tmp_path: Path):
    _allow_production_order_firewall(monkeypatch)
    monkeypatch.setenv("ALLOW_ORDER_EXECUTION", "1")
    monkeypatch.setenv("MARKET_DATA_ONLY", "0")
    monkeypatch.setenv("LIVE_PRETRADE_RECONCILE_REQUIRED", "0")

    trader = _mk_trader("live")
    trader.project_root = str(tmp_path)
    trader.global_halt_flag_path = str(tmp_path / "GLOBAL_TRADING_HALT.flag")
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

    assert out.get("status") == "LIVE_ORDER_OUTCOME_UNKNOWN"
    assert out["live_order"]["broker_submission_may_have_succeeded"] is True
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


def test_live_execute_uses_futures_plan_order_spec(monkeypatch, tmp_path: Path):
    _allow_production_order_firewall(monkeypatch)
    monkeypatch.setenv("ALLOW_ORDER_EXECUTION", "1")
    monkeypatch.setenv("MARKET_DATA_ONLY", "0")
    monkeypatch.setenv("LIVE_PRETRADE_RECONCILE_REQUIRED", "0")

    trader = _mk_trader("live")
    trader.project_root = str(tmp_path)
    trader.global_halt_flag_path = str(tmp_path / "GLOBAL_TRADING_HALT.flag")
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

    assert out.get("status") == "LIVE_ORDER_OUTCOME_UNKNOWN"
    placed = trader.client.placed_specs[-1]
    assert len(placed.get("orderLegCollection", [])) == 2
    assert placed["orderLegCollection"][0]["instrument"]["assetType"] == "FUTURE"
    assert placed["orderLegCollection"][0]["instrument"]["symbol"] == "/ESM26"
    assert placed["orderLegCollection"][1]["instrument"]["symbol"] == "/ESU26"


def test_live_execute_uses_futures_roll_legs(monkeypatch, tmp_path: Path):
    _allow_production_order_firewall(monkeypatch)
    monkeypatch.setenv("ALLOW_ORDER_EXECUTION", "1")
    monkeypatch.setenv("MARKET_DATA_ONLY", "0")
    monkeypatch.setenv("LIVE_PRETRADE_RECONCILE_REQUIRED", "0")

    trader = _mk_trader("live")
    trader.project_root = str(tmp_path)
    trader.global_halt_flag_path = str(tmp_path / "GLOBAL_TRADING_HALT.flag")
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

    assert out.get("status") == "LIVE_ORDER_OUTCOME_UNKNOWN"
    placed = trader.client.placed_specs[-1]
    assert placed["orderLegCollection"][0]["instruction"] == "SELL"
    assert placed["orderLegCollection"][1]["instruction"] == "BUY"
    assert placed["orderLegCollection"][0]["instrument"]["symbol"] == "/ESM26"
    assert placed["orderLegCollection"][1]["instrument"]["symbol"] == "/ESU26"


def test_paper_execute_uses_guard_and_fill_modeling(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("ALLOW_ORDER_EXECUTION", "1")
    monkeypatch.setenv("MARKET_DATA_ONLY", "0")
    monkeypatch.setenv("PAPER_EXECUTION_USE_EXPECTED_FILL_PRICE", "1")

    trader = _mk_trader("paper")
    trader.project_root = str(tmp_path)
    trader.set_mode("paper")
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
    paper = out["paper_order"]
    assert paper["paper_fill_source"] == "expected_fill_model"
    assert paper["fill_price"] == paper["expected_fill_price"]
    assert abs(paper["realized_slippage_bps"] - paper["expected_slippage_bps"]) < 1e-6


def test_paper_profitability_guard_blocks_weak_profile_new_buy_entries(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("ALLOW_ORDER_EXECUTION", "1")
    monkeypatch.setenv("MARKET_DATA_ONLY", "0")
    _write_paper_profitability_control(tmp_path, weak_profile="default")

    trader = _mk_trader("paper")
    trader.project_root = str(tmp_path)
    trader.set_mode("paper")
    trader.execution_enabled = True
    trader.market_data_only = False

    out = trader.execute_decision(
        symbol="PEPE-USD",
        action="BUY",
        quantity=100.0,
        model_score=0.62,
        threshold=0.55,
        features={"last_price": 0.00001, "tradeability_score": 0.0},
        gates={"model_gate": True, "market_data_ok": True},
        reasons=["unit_test"],
        strategy="paper_profitability_guard_test",
        metadata={"source_profile": "default", "shadow_domain": "crypto"},
    )

    assert out.get("status") == "PAPER_PROFITABILITY_GUARD_BLOCKED"
    decision = out.get("live_guard_decision", {})
    assert decision.get("gate") == "paper_profitability_weak_profile_new_entry"
    assert decision.get("reason") == "paper_profitability_weak_profile_new_entry_block"
    assert decision.get("details", {}).get("source_profile") == "default"
    paper_log = Path(trader.paper_log_path)
    assert not paper_log.exists() or paper_log.read_text(encoding="utf-8").strip() == ""
    event_files = sorted((tmp_path / "governance" / "events").glob("paper_execution_guard_*.jsonl"))
    assert event_files
    assert "paper_profitability_weak_profile_new_entry_block" in event_files[-1].read_text(encoding="utf-8")


def test_paper_profitability_guard_allows_weak_profile_sell_reduction(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("ALLOW_ORDER_EXECUTION", "1")
    monkeypatch.setenv("MARKET_DATA_ONLY", "0")
    _write_paper_profitability_control(tmp_path, weak_profile="default")

    trader = _mk_trader("paper")
    trader.project_root = str(tmp_path)
    trader.set_mode("paper")
    trader.execution_enabled = True
    trader.market_data_only = False
    seeded_position = {"qty": 100.0, "avg_price": 0.00001, "mark_price": 0.00001}
    trader._paper_positions["PEPE-USD"] = dict(seeded_position)
    trader._paper_profile_positions.setdefault("default", {})["PEPE-USD"] = dict(seeded_position)
    trader._paper_strategy_positions.setdefault("paper_profitability_guard_test", {})["PEPE-USD"] = dict(
        seeded_position
    )

    out = trader.execute_decision(
        symbol="PEPE-USD",
        action="SELL",
        quantity=100.0,
        model_score=0.62,
        threshold=0.55,
        features={"last_price": 0.00001, "tradeability_score": 0.0},
        gates={"model_gate": True, "market_data_ok": True},
        reasons=["unit_test"],
        strategy="paper_profitability_guard_test",
        metadata={"source_profile": "default", "shadow_domain": "crypto"},
    )

    assert out.get("status") == "PAPER_EXECUTED"
    assert Path(trader.paper_log_path).exists()
    assert out["paper_order"]["position_qty"] == 0.0


def test_paper_profitability_guard_enforces_declared_clean_sleeve_evidence(tmp_path: Path) -> None:
    health = tmp_path / "governance" / "health"
    health.mkdir(parents=True, exist_ok=True)
    payload = {
        "raw_profitability_a_recovery_contract": {
            "active": True,
            "weak_profiles": ["weak"],
            "runtime_enforcement": {"block_new_entries_on_weak_profiles": True},
        },
        "raw_profitability_improvement_contract": {
            "active": True,
            "runtime_enforcement": {"block_new_entries_on_weak_profiles": True},
            "clean_sleeve_strict_buy_gate_contract": {
                "active": True,
                "enforced": True,
                "min_quality_gate_norm": 0.72,
                "min_tradeability_norm": 0.58,
                "min_execution_fitness_norm": 0.58,
                "min_cross_asset_confirmation_norm": 0.56,
                "max_overlap_pressure_norm": 0.58,
                "min_independent_evidence_channels": 4,
                "block_when_spread_regime_unknown": True,
            },
        },
    }
    (health / "paper_runtime_profitability_controls_latest.json").write_text(json.dumps(payload), encoding="utf-8")
    _reset_paper_profitability_guard_cache()
    trader = _mk_trader("paper")
    trader.project_root = str(tmp_path)

    blocked, reason, details = trader._paper_profitability_new_entry_blocked(
        symbol="AAPL",
        action="BUY",
        quantity=1.0,
        metadata={"source_profile": "clean", "session": "regular"},
        features={"last_price": 100.0},
        strategy="alpha",
    )
    allowed, allowed_reason, _allowed_details = trader._paper_profitability_new_entry_blocked(
        symbol="AAPL",
        action="BUY",
        quantity=1.0,
        metadata={"source_profile": "clean", "session": "regular"},
        features={
            "last_price": 100.0,
            "source_quality_norm": 0.90,
            "tradeability_norm": 0.90,
            "execution_fitness_norm": 0.90,
            "cross_asset_confirmation_norm": 0.90,
            "overlap_pressure_norm": 0.10,
            "spread_bps": 5.0,
            "event_catalyst_confirmation_norm": 0.90,
            "portfolio_conflict_clearance_norm": 0.90,
            "session_quality_norm": 0.90,
        },
        strategy="alpha",
    )

    assert blocked is True
    assert reason == "paper_profitability_clean_profile_evidence_block"
    assert "source_quality_unknown" in details["failures"]
    assert allowed is False
    assert allowed_reason == "clean_profile_evidence_gate_passed"


def test_paper_profitability_guard_enforces_profile_strategy_quarantine(tmp_path: Path) -> None:
    health = tmp_path / "governance" / "health"
    health.mkdir(parents=True, exist_ok=True)
    payload = {
        "raw_profitability_a_recovery_contract": {
            "active": True,
            "weak_profiles": [],
            "runtime_enforcement": {"block_new_entries_on_weak_profiles": True},
        },
        "raw_profitability_improvement_contract": {
            "active": True,
            "runtime_enforcement": {"block_new_entries_on_weak_profiles": True},
            "losing_strategy_pair_quarantine_contract": {
                "active": True,
                "pairs": [
                    {
                        "profile": "clean",
                        "strategy": "alpha",
                        "protected": True,
                        "new_entry_cap": 0,
                    }
                ],
            },
        },
    }
    (health / "paper_runtime_profitability_controls_latest.json").write_text(json.dumps(payload), encoding="utf-8")
    _reset_paper_profitability_guard_cache()
    trader = _mk_trader("paper")
    trader.project_root = str(tmp_path)

    blocked, reason, details = trader._paper_profitability_new_entry_blocked(
        symbol="AAPL",
        action="BUY",
        quantity=1.0,
        metadata={"source_profile": "clean"},
        features={"last_price": 100.0},
        strategy="alpha",
    )

    assert blocked is True
    assert reason == "paper_profitability_strategy_pair_quarantine_block"
    assert details["strategy"] == "alpha"


def test_paper_book_state_survives_restart_and_preserves_reduction_semantics(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("ALLOW_ORDER_EXECUTION", "1")
    monkeypatch.setenv("MARKET_DATA_ONLY", "0")
    monkeypatch.setenv("SHADOW_PROFILE", "restart_persistence")
    monkeypatch.setenv("SHADOW_DOMAIN", "equities")

    first = _mk_trader("paper")
    first.project_root = str(tmp_path)
    first.set_mode("paper")
    first.execution_enabled = True
    first.market_data_only = False
    opened = first.execute_decision(
        symbol="AAPL",
        action="BUY",
        quantity=2.0,
        model_score=0.62,
        threshold=0.55,
        features={"last_price": 100.0, "tradeability_score": 1.0},
        gates={"model_gate": True, "market_data_ok": True},
        reasons=["unit_test"],
        strategy="paper_restart_state_test",
        metadata={"source_profile": "restart_persistence", "shadow_domain": "equities"},
    )
    assert opened["status"] == "PAPER_EXECUTED"
    state_path = Path(first._paper_state_path)
    state_payload = json.loads(state_path.read_text(encoding="utf-8"))
    first_book_id = state_payload["paper_book_id"]

    restarted = _mk_trader("paper")
    restarted.project_root = str(tmp_path)
    restarted.set_mode("paper")
    restarted.execution_enabled = True
    restarted.market_data_only = False

    assert restarted._paper_book_id == first_book_id
    assert restarted._paper_profile_positions["restart_persistence"]["AAPL"]["qty"] == 2.0

    closed = restarted.execute_decision(
        symbol="AAPL",
        action="SELL",
        quantity=2.0,
        model_score=0.62,
        threshold=0.55,
        features={"last_price": 101.0, "tradeability_score": 1.0},
        gates={"model_gate": True, "market_data_ok": True},
        reasons=["unit_test"],
        strategy="paper_restart_state_test",
        metadata={"source_profile": "restart_persistence", "shadow_domain": "equities"},
    )

    assert closed["status"] == "PAPER_EXECUTED"
    assert closed["paper_order"]["paper_book_id"] == first_book_id
    assert closed["paper_order"]["paper_profile_net_pnl_delta"] > 0.0
    assert restarted._paper_profile_positions["restart_persistence"]["AAPL"]["qty"] == 0.0


def test_paper_execute_can_block_on_guard(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("ALLOW_ORDER_EXECUTION", "1")
    monkeypatch.setenv("MARKET_DATA_ONLY", "0")

    trader = _mk_trader("paper")
    trader.project_root = str(tmp_path)
    trader.set_mode("paper")
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
