import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.execution_simulator import simulate_execution
import scripts.execution_lab as lab


def test_execution_simulator_emits_extended_microstructure_fields() -> None:
    result = simulate_execution(
        action="BUY",
        last_price=100.0,
        return_1m=0.002,
        spread_bps=14.0,
        volatility_1m=0.01,
        latency_ms=240.0,
        bid_size=100.0,
        ask_size=90.0,
        order_size=50.0,
        broker="schwab",
        market_kind="equities",
        symbol="AAPL",
        session="open",
        order_type="market",
        live_fill_slippage_bps=1.0,
    )

    assert result.queue_priority_score >= 0.0
    assert result.requote_probability >= 0.0
    assert result.session_penalty_bps >= 0.0
    assert result.latency_bucket in {"fast", "watch", "slow"}
    assert result.paper_execution_score > 0.0
    assert result.effective_fill_ratio > 0.0
    assert result.reject_probability >= 0.0
    assert result.market_impact_bps >= result.impact_bps


def test_execution_simulator_models_option_stale_quote_rejects() -> None:
    result = simulate_execution(
        action="SELL_TO_OPEN",
        last_price=4.0,
        return_1m=0.0,
        spread_bps=60.0,
        volatility_1m=0.02,
        latency_ms=500.0,
        bid_size=5.0,
        ask_size=5.0,
        order_size=10.0,
        broker="schwab",
        market_kind="options",
        symbol="NVDA_covered_call",
        session="regular",
        order_type="limit",
        asset_class="options",
        sleeve="covered_call",
        quote_age_ms=6000.0,
        open_interest=0.0,
    )

    assert result.asset_class == "options"
    assert result.paper_execution_status == "stale_quote_rejected"
    assert result.effective_fill_ratio == 0.0
    assert result.option_liquidity_penalty_bps > 0.0
    assert result.stale_quote_probability == 1.0


def test_execution_simulator_starts_market_fill_at_observed_touch() -> None:
    result = simulate_execution(
        action="BUY",
        last_price=100.0,
        return_1m=0.001,
        spread_bps=2.0,
        volatility_1m=0.001,
        bid_price=99.95,
        ask_price=100.05,
        bid_size=5000.0,
        ask_size=5000.0,
        order_size=1.0,
        broker="schwab",
        market_kind="equities",
        symbol="SCHD",
    )

    assert result.quote_source_mode == "observed_bid_ask"
    assert result.touch_price == 100.05
    assert result.expected_fill_price > result.touch_price
    assert 9.9 <= result.quoted_spread_bps <= 10.1
    assert result.beyond_touch_cost_bps > 0.0
    assert result.total_cost_bps == result.slippage_bps


def test_execution_simulator_rejects_crossed_or_locked_quote() -> None:
    result = simulate_execution(
        action="SELL",
        last_price=100.0,
        return_1m=0.0,
        spread_bps=8.0,
        volatility_1m=0.001,
        bid_price=100.10,
        ask_price=100.00,
        bid_size=1000.0,
        ask_size=1000.0,
        order_size=1.0,
        broker="schwab",
        market_kind="equities",
        symbol="SCHD",
    )

    assert result.quote_crossed_or_locked is True
    assert result.quote_source_mode == "crossed_or_locked_bid_ask"
    assert result.paper_execution_status == "crossed_or_locked_quote_rejected"
    assert result.reject_probability == 1.0
    assert result.effective_fill_ratio == 0.0


def test_execution_lab_builds_scenario_grid() -> None:
    payload = lab.build_payload()
    assert payload["scenario_count"] >= 4
    assert payload["capabilities"]["queue_priority_modeling"] is True
    assert payload["capabilities"]["reject_cancel_stale_quote_modeling"] is True
    assert payload["capabilities"]["realistic_option_fills"] is True
    assert any(row["asset_class"] == "options" for row in payload["rows"])
