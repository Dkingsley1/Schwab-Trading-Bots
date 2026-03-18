from pathlib import Path
import sys
import threading

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.coinbase_market_data import CoinbaseMarketDataClient
from core.execution_simulator import simulate_execution
import scripts.run_shadow_training_loop as loop


def test_coinbase_market_snapshot_fetches_sources_in_parallel(monkeypatch) -> None:
    monkeypatch.setenv("COINBASE_SNAPSHOT_MAX_WORKERS", "4")
    client = CoinbaseMarketDataClient(timeout_seconds=0.1)
    barrier = threading.Barrier(4, timeout=2.0)
    called: list[str] = []

    def _wait(name: str, payload):
        called.append(name)
        barrier.wait(timeout=2.0)
        return payload

    monkeypatch.setattr(client, "get_ticker", lambda symbol: _wait("ticker", {"price": "101.0"}))
    monkeypatch.setattr(
        client,
        "get_candles",
        lambda symbol, minutes=60, granularity=60: _wait(
            "candles",
            [[1710000000.0, 99.0, 101.0, 100.0, 100.5, 50.0], [1710000060.0, 100.0, 102.0, 100.5, 101.0, 60.0]],
        ),
    )
    monkeypatch.setattr(client, "get_product", lambda symbol: _wait("product", {}))
    monkeypatch.setattr(client, "get_book", lambda symbol, level=2: _wait("book", {"bids": [], "asks": []}))

    snapshot = client.market_snapshot("BTC-USD")
    client.close()

    assert snapshot["last_price"] == 101.0
    assert set(called) == {"ticker", "candles", "product", "book"}


def test_coinbase_client_caches_product_and_candles(monkeypatch) -> None:
    monkeypatch.setenv("COINBASE_PRODUCT_CACHE_SECONDS", "900")
    monkeypatch.setenv("COINBASE_CANDLE_CACHE_SECONDS", "30")
    client = CoinbaseMarketDataClient(timeout_seconds=0.1)
    calls: list[str] = []

    def _fake_get_json(path: str, params=None, *, symbol: str = ""):
        calls.append(path)
        if path.endswith("/candles"):
            return [[1710000000.0, 99.0, 101.0, 100.0, 100.5, 25.0]]
        return {"product_id": symbol or "BTC-USD"}

    monkeypatch.setattr(client, "_get_json", _fake_get_json)

    client.get_product("BTC-USD")
    client.get_product("BTC-USD")
    client.get_candles("BTC-USD", minutes=60, granularity=60)
    client.get_candles("BTC-USD", minutes=60, granularity=60)
    client.close()

    assert calls.count("/products/BTC-USD") == 1
    assert calls.count("/products/BTC-USD/candles") == 1


def test_coinbase_market_snapshot_adds_execution_relevant_features(monkeypatch) -> None:
    monkeypatch.setenv("COINBASE_SNAPSHOT_MAX_WORKERS", "4")
    client = CoinbaseMarketDataClient(timeout_seconds=0.1)
    candles = []
    base_ts = 1710000000
    for idx in range(20):
        close = 100.0 + idx
        candles.append([base_ts + (idx * 60), close - 1.0, close + 1.0, close - 0.5, close, 100.0 + (idx * 10.0)])

    monkeypatch.setattr(
        client,
        "get_ticker",
        lambda symbol: {"price": "120.0", "bid": "119.9", "ask": "120.1", "bidSize": "14", "askSize": "10"},
    )
    monkeypatch.setattr(client, "get_candles", lambda symbol, minutes=60, granularity=60: candles)
    monkeypatch.setattr(client, "get_product", lambda symbol: {})
    monkeypatch.setattr(
        client,
        "get_book",
        lambda symbol, level=2: {
            "bids": [["119.9", "14"], ["119.8", "12"]],
            "asks": [["120.1", "10"], ["120.2", "9"]],
        },
    )

    snapshot = client.market_snapshot("BTC-USD")
    client.close()

    assert snapshot["return_1m"] > 0.0
    assert snapshot["mom_15m"] > 0.0
    assert snapshot["volatility_1m"] > 0.0
    assert snapshot["volume_30m"] > 0.0
    assert snapshot["queue_depth_norm"] > 0.0
    assert snapshot["futures_vwap_bias_norm"] > 0.0


def test_coinbase_market_snapshot_prefers_fresh_websocket_data(monkeypatch) -> None:
    client = CoinbaseMarketDataClient(timeout_seconds=0.1)
    monkeypatch.setattr(
        client._ws_cache,
        "latest",
        lambda symbol: {
            "ticker": {
                "price": "101.5",
                "bid": "101.4",
                "ask": "101.6",
                "bidSize": "7",
                "askSize": "9",
            },
            "book": {
                "bids": [["101.4", "7"], ["101.3", "6"]],
                "asks": [["101.6", "9"], ["101.7", "5"]],
            },
        },
    )
    monkeypatch.setattr(client, "get_ticker", lambda symbol: (_ for _ in ()).throw(AssertionError("ticker REST should not be used")))
    monkeypatch.setattr(client, "get_book", lambda symbol, level=2: (_ for _ in ()).throw(AssertionError("book REST should not be used")))
    monkeypatch.setattr(
        client,
        "get_candles",
        lambda symbol, minutes=60, granularity=60: [[1710000000.0, 100.0, 102.0, 100.5, 101.0, 25.0], [1710000060.0, 101.0, 103.0, 101.5, 101.5, 30.0]],
    )
    monkeypatch.setattr(client, "get_product", lambda symbol: {})

    snapshot = client.market_snapshot("BTC-USD")
    client.close()

    assert snapshot["last_price"] == 101.5
    assert snapshot["snapshot_transport"] == "websocket"
    assert snapshot["bid_size"] > 0.0
    assert snapshot["ask_size"] > 0.0


def test_coinbase_symbol_policy_uses_fast_slow_and_websocket_env(monkeypatch) -> None:
    monkeypatch.setenv("COINBASE_WATCH_SYMBOLS_FAST", "BTC-USD,ETH-USD")
    monkeypatch.setenv("COINBASE_WATCH_SYMBOLS_SLOW", "DOGE-USD")
    monkeypatch.setenv("COINBASE_WEBSOCKET_SYMBOLS", "BTC-USD")
    monkeypatch.setenv("COINBASE_CORE_EVERY_N_ITERS", "1")
    monkeypatch.setenv("COINBASE_FAST_EVERY_N_ITERS", "1")
    monkeypatch.setenv("COINBASE_SLOW_EVERY_N_ITERS", "3")

    policy = loop._coinbase_symbol_policy("default", ["BTC-USD", "ETH-USD", "DOGE-USD", "SOL-USD"])

    assert policy["fast_symbols"] == ["BTC-USD", "ETH-USD"]
    assert policy["slow_symbols"] == ["DOGE-USD"]
    assert policy["websocket_symbols"] == ["BTC-USD"]
    assert policy["slow_every_n"] == 3


def test_schwab_symbol_policy_uses_fast_slow_env(monkeypatch) -> None:
    monkeypatch.setenv("SCHWAB_WATCH_SYMBOLS_FAST", "SPY,QQQ")
    monkeypatch.setenv("SCHWAB_WATCH_SYMBOLS_SLOW", "TLT,GLD")
    monkeypatch.setenv("SCHWAB_CORE_EVERY_N_ITERS", "1")
    monkeypatch.setenv("SCHWAB_FAST_EVERY_N_ITERS", "1")
    monkeypatch.setenv("SCHWAB_SLOW_EVERY_N_ITERS", "4")

    policy = loop._schwab_symbol_policy("default", ["SPY", "QQQ", "TLT", "GLD", "NVDA"])

    assert policy["fast_symbols"] == ["SPY", "QQQ"]
    assert policy["slow_symbols"] == ["TLT", "GLD"]
    assert policy["websocket_symbols"] == []
    assert policy["slow_every_n"] == 4


def test_simulate_execution_applies_market_specific_scale_and_fee(monkeypatch) -> None:
    monkeypatch.setenv("EXEC_SIM_SLIPPAGE_SCALE_CRYPTO", "0.5")
    monkeypatch.setenv("EXEC_SIM_BASE_FEE_BPS_CRYPTO", "1.8")

    result = simulate_execution(
        action="BUY",
        last_price=100.0,
        return_1m=0.002,
        spread_bps=10.0,
        volatility_1m=0.01,
        latency_ms=200.0,
        bid_size=20.0,
        ask_size=20.0,
        order_size=2.0,
        broker="coinbase",
        market_kind="crypto",
        symbol="BTC-USD",
    )

    assert result.fee_bps == 1.8
    assert result.slippage_bps > result.fee_bps
    assert result.expected_fill_price > 100.0


def test_jsonl_write_buffer_dedupes_recent_message_ids_per_path() -> None:
    path = "/tmp/test_api_calls_runtime.jsonl"
    rows = [
        {"message_id": "dup-1", "timestamp_utc": "2026-03-12T12:00:00+00:00", "symbol": "BTC-USD", "endpoint": "coinbase.market_snapshot", "status": "ok"},
        {"message_id": "dup-1", "timestamp_utc": "2026-03-12T12:00:00+00:00", "symbol": "BTC-USD", "endpoint": "coinbase.market_snapshot", "status": "ok"},
        {"message_id": "dup-2", "timestamp_utc": "2026-03-12T12:00:01+00:00", "symbol": "ETH-USD", "endpoint": "coinbase.market_snapshot", "status": "ok"},
    ]

    loop.JsonlWriteBuffer._recent_message_ids.pop(str(Path(path).resolve()), None)
    deduped = loop.JsonlWriteBuffer._dedupe_rows(path, rows)
    loop.JsonlWriteBuffer._remember_written_rows(path, deduped)
    deduped_again = loop.JsonlWriteBuffer._dedupe_rows(path, rows)

    assert [row["message_id"] for row in deduped] == ["dup-1", "dup-2"]
    assert deduped_again == []


def test_execution_guard_blocks_slow_adverse_coinbase_trade() -> None:
    action, score, reasons, meta = loop._apply_execution_guard(
        action="BUY",
        score=0.71,
        threshold=0.58,
        reasons=["base_buy"],
        features={
            "spread_bps": 8.0,
            "futures_depth_ratio_norm": 0.35,
            "futures_order_book_imbalance": -0.80,
            "market_data_latency_ms": 2200.0,
        },
        symbol_is_futures=True,
        broker="coinbase",
    )

    assert action == "HOLD"
    assert meta["market_kind"] == "crypto"
    assert (meta["latency_ok"] is False) or (meta["imbalance_ok"] is False)
    assert any("execution_guard_block" in reason for reason in reasons)


def test_build_options_plan_emits_put_calendar_spread_for_bearish_term_structure() -> None:
    decision = loop._build_options_plan(
        symbol="SPY",
        mkt={
            "last_price": 100.0,
            "vol_30m": 0.008,
            "range_pos": 0.38,
            "options_iv_atm_norm": 0.42,
            "options_iv_skew_norm": 0.54,
            "options_iv_term_structure_norm": 0.72,
            "options_put_call_oi_ratio_norm": 0.66,
            "options_negative_bias_norm": 0.63,
            "options_roll_yield_norm": 0.18,
            "options_vwap_bias_norm": 0.42,
            "options_vol_expectation_norm": 0.34,
            "options_spread_bps_norm": 0.18,
            "calendar_event_proximity_norm": 0.20,
            "calendar_high_impact_24h_norm": 0.12,
            "calendar_options_expiry_week_norm": 0.35,
            "options_near_expiry_days": 21.0,
            "options_far_expiry_days": 56.0,
        },
        master_action="SELL",
        master_score=0.36,
        master_vote=-0.42,
        covered_call_shares=0,
    )

    assert decision["plan"]["options_style"] == "PUT_CALENDAR_SPREAD"
    assert decision["plan"]["strategy_family"] == "calendar"
    assert len(decision["plan"]["legs"]) == 2


def test_build_options_plan_emits_iron_butterfly_for_neutral_high_vol_setup(monkeypatch) -> None:
    monkeypatch.setenv("OPTIONS_WHEEL_ENABLED", "0")
    decision = loop._build_options_plan(
        symbol="QQQ",
        mkt={
            "last_price": 100.0,
            "vol_30m": 0.022,
            "range_pos": 0.50,
            "options_iv_atm_norm": 0.64,
            "options_iv_skew_norm": 0.53,
            "options_iv_term_structure_norm": 0.55,
            "options_put_call_oi_ratio_norm": 0.51,
            "options_negative_bias_norm": 0.49,
            "options_roll_yield_norm": 0.24,
            "options_vwap_bias_norm": 0.51,
            "options_vol_expectation_norm": 0.68,
            "options_spread_bps_norm": 0.22,
            "calendar_event_proximity_norm": 0.18,
            "calendar_high_impact_24h_norm": 0.10,
            "calendar_options_expiry_week_norm": 0.30,
            "options_near_expiry_days": 18.0,
            "options_far_expiry_days": 45.0,
        },
        master_action="HOLD",
        master_score=0.55,
        master_vote=0.02,
        covered_call_shares=0,
    )

    assert decision["plan"]["options_style"] == "IRON_BUTTERFLY"
    assert decision["plan"]["strategy_family"] == "neutral_income"
    assert len(decision["plan"]["legs"]) == 4


def test_build_futures_plan_emits_funding_mean_revert_style() -> None:
    decision = loop._build_futures_plan(
        symbol="BTC-USD",
        mkt={
            "last_price": 100.0,
            "vol_30m": 0.010,
            "range_pos": 0.48,
            "mom_5m": -0.0002,
            "futures_spread_bps_norm": 0.22,
            "futures_depth_ratio_norm": 0.40,
            "futures_order_book_imbalance_norm": 0.66,
            "futures_basis_bps_norm": 0.54,
            "futures_term_structure_norm": 0.52,
            "futures_roll_yield_norm": 0.50,
            "futures_vwap_bias_norm": 0.50,
            "futures_negative_bias_norm": 0.48,
            "futures_funding_rate_norm": 0.72,
            "calendar_event_proximity_norm": 0.10,
            "calendar_high_impact_24h_norm": 0.12,
        },
        master_action="HOLD",
        master_score=0.54,
        master_vote=0.08,
        symbol_is_futures=True,
    )

    assert decision["plan"]["futures_style"] == "FUTURES_FUNDING_MEAN_REVERT"
    assert decision["plan"]["strategy_family"] == "mean_revert"
    assert decision["action"] == "SELL"


def test_build_futures_plan_emits_orderbook_breakout_style() -> None:
    decision = loop._build_futures_plan(
        symbol="ETH-USD",
        mkt={
            "last_price": 100.0,
            "vol_30m": 0.014,
            "range_pos": 0.63,
            "mom_5m": 0.0021,
            "futures_spread_bps_norm": 0.20,
            "futures_depth_ratio_norm": 0.55,
            "futures_order_book_imbalance_norm": 0.78,
            "futures_basis_bps_norm": 0.53,
            "futures_term_structure_norm": 0.51,
            "futures_roll_yield_norm": 0.54,
            "futures_vwap_bias_norm": 0.55,
            "futures_negative_bias_norm": 0.47,
            "futures_funding_rate_norm": 0.56,
            "calendar_event_proximity_norm": 0.08,
            "calendar_high_impact_24h_norm": 0.10,
        },
        master_action="BUY",
        master_score=0.66,
        master_vote=0.34,
        symbol_is_futures=True,
    )

    assert decision["plan"]["futures_style"] == "FUTURES_ORDERBOOK_IMBALANCE_BREAKOUT"
    assert decision["plan"]["strategy_family"] == "directional"
    assert decision["action"] == "BUY"
