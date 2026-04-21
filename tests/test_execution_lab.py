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


def test_execution_lab_builds_scenario_grid() -> None:
    payload = lab.build_payload()
    assert payload["scenario_count"] >= 4
    assert payload["capabilities"]["queue_priority_modeling"] is True
