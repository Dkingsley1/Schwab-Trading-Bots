from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any, Callable

from core.execution_simulator import simulate_execution
from core.live_order_ledger import LiveOrderLedger


def _scenario(name: str, run: Callable[[], dict[str, Any]]) -> dict[str, Any]:
    try:
        result = run()
        return {"scenario": name, "ok": bool(result.get("ok", False)), **result}
    except Exception as exc:
        return {
            "scenario": name,
            "ok": False,
            "error": f"{type(exc).__name__}:{exc}",
        }


def _reserve(ledger: LiveOrderLedger, intent_id: str, quantity: float = 10.0) -> None:
    result = ledger.reserve(
        intent_id=intent_id,
        payload={"symbol": "SPY", "action": "BUY", "quantity": quantity},
        requested_quantity=quantity,
    )
    if not result.get("reserved", False):
        raise RuntimeError(str(result.get("reason") or "reservation_failed"))


def _normal(path: Path) -> dict[str, Any]:
    ledger = LiveOrderLedger(path)
    _reserve(ledger, "normal")
    ledger.mark_submitting("normal")
    ledger.mark_submit_result(
        intent_id="normal", acknowledged=True, broker_order_id="broker-normal"
    )
    ledger.record_broker_update(
        broker_order_id="broker-normal", broker_status="WORKING"
    )
    final = ledger.record_broker_update(
        broker_order_id="broker-normal",
        broker_status="FILLED",
        filled_quantity=10.0,
        average_fill_price=500.0,
    )
    return {"ok": final.get("state") == "filled" and ledger.verify_integrity()["ok"]}


def _latency() -> dict[str, Any]:
    baseline = simulate_execution(
        action="BUY",
        last_price=500.0,
        return_1m=0.001,
        spread_bps=2.0,
        volatility_1m=0.002,
        latency_ms=50.0,
        bid_size=1000.0,
        ask_size=1000.0,
        order_size=10.0,
        broker="schwab",
        market_kind="equities",
        symbol="SPY",
    )
    stressed = simulate_execution(
        action="BUY",
        last_price=500.0,
        return_1m=0.001,
        spread_bps=2.0,
        volatility_1m=0.002,
        latency_ms=5000.0,
        bid_size=1000.0,
        ask_size=1000.0,
        order_size=10.0,
        broker="schwab",
        market_kind="equities",
        symbol="SPY",
    )
    return {
        "ok": stressed.latency_ms > baseline.latency_ms
        and stressed.slippage_bps >= baseline.slippage_bps,
        "baseline_slippage_bps": baseline.slippage_bps,
        "stressed_slippage_bps": stressed.slippage_bps,
    }


def _submit_disconnect(path: Path) -> dict[str, Any]:
    ledger = LiveOrderLedger(path)
    _reserve(ledger, "disconnect")
    ledger.mark_submitting("disconnect")
    state = ledger.mark_submit_result(
        intent_id="disconnect", acknowledged=False, error="simulated_connection_drop"
    )
    duplicate = ledger.reserve(
        intent_id="disconnect",
        payload={"symbol": "SPY", "action": "BUY", "quantity": 10.0},
        requested_quantity=10.0,
    )
    return {
        "ok": state.get("state") == "submit_unknown"
        and duplicate.get("duplicate") is True
        and len(ledger.unresolved()) == 1
    }


def _duplicate(path: Path) -> dict[str, Any]:
    ledger = LiveOrderLedger(path)
    _reserve(ledger, "duplicate")
    result = ledger.reserve(
        intent_id="duplicate",
        payload={"symbol": "SPY", "action": "BUY", "quantity": 10.0},
        requested_quantity=10.0,
    )
    return {"ok": result.get("duplicate") is True and result.get("reserved") is False}


def _partial_fill(path: Path) -> dict[str, Any]:
    ledger = LiveOrderLedger(path)
    _reserve(ledger, "partial")
    ledger.mark_submitting("partial")
    ledger.mark_submit_result(
        intent_id="partial", acknowledged=True, broker_order_id="broker-partial"
    )
    first = ledger.record_broker_update(
        broker_order_id="broker-partial",
        broker_status="PARTIALLY_FILLED",
        filled_quantity=2.0,
        average_fill_price=499.9,
    )
    second = ledger.record_broker_update(
        broker_order_id="broker-partial",
        broker_status="PARTIALLY_FILLED",
        filled_quantity=7.0,
        average_fill_price=500.1,
    )
    final = ledger.record_broker_update(
        broker_order_id="broker-partial",
        broker_status="FILLED",
        filled_quantity=10.0,
        average_fill_price=500.2,
    )
    return {
        "ok": first.get("filled_quantity") == 2.0
        and second.get("filled_quantity") == 7.0
        and final.get("state") == "filled"
    }


def order_allowed_during_halt(*, global_halt: bool, risk_reducing_exit: bool) -> bool:
    return bool(not global_halt or risk_reducing_exit)


def _halt() -> dict[str, Any]:
    return {
        "ok": not order_allowed_during_halt(global_halt=True, risk_reducing_exit=False)
        and order_allowed_during_halt(global_halt=True, risk_reducing_exit=True)
    }


def gap_guard(
    *, reference_price: float, observed_price: float, maximum_gap_bps: float
) -> dict[str, Any]:
    reference = float(reference_price or 0.0)
    observed = float(observed_price or 0.0)
    if reference <= 0.0 or observed <= 0.0:
        return {
            "ok": False,
            "allow_execute": False,
            "reason": "invalid_gap_price",
            "gap_bps": None,
        }
    gap_bps = abs(observed - reference) / reference * 10000.0
    return {
        "ok": True,
        "allow_execute": gap_bps <= max(float(maximum_gap_bps), 0.0),
        "reason": "within_gap_budget"
        if gap_bps <= maximum_gap_bps
        else "gap_budget_exceeded",
        "gap_bps": gap_bps,
    }


def _gap() -> dict[str, Any]:
    result = gap_guard(
        reference_price=100.0, observed_price=108.0, maximum_gap_bps=250.0
    )
    return {
        "ok": result["ok"]
        and not result["allow_execute"]
        and result["gap_bps"] == 800.0
    }


def _cancel_fill_race(path: Path) -> dict[str, Any]:
    ledger = LiveOrderLedger(path)
    _reserve(ledger, "cancel-race")
    ledger.mark_submitting("cancel-race")
    ledger.mark_submit_result(
        intent_id="cancel-race", acknowledged=True, broker_order_id="broker-race"
    )
    ledger.record_broker_update(broker_order_id="broker-race", broker_status="WORKING")
    ledger.mark_cancel_pending("broker-race")
    final = ledger.record_broker_update(
        broker_order_id="broker-race",
        broker_status="FILLED",
        filled_quantity=10.0,
        average_fill_price=501.0,
    )
    return {"ok": final.get("state") == "filled" and not ledger.unresolved()}


def run_execution_scenarios(root: str | Path | None = None) -> dict[str, Any]:
    if root is None:
        temporary = tempfile.TemporaryDirectory(prefix="execution-scenarios-")
        base = Path(temporary.name)
    else:
        temporary = None
        base = Path(root)
        base.mkdir(parents=True, exist_ok=True)
    try:
        rows = [
            _scenario("normal_fill", lambda: _normal(base / "normal.sqlite3")),
            _scenario("latency_stress", _latency),
            _scenario(
                "submit_disconnect",
                lambda: _submit_disconnect(base / "disconnect.sqlite3"),
            ),
            _scenario(
                "duplicate_intent", lambda: _duplicate(base / "duplicate.sqlite3")
            ),
            _scenario(
                "progressive_partial_fill",
                lambda: _partial_fill(base / "partial.sqlite3"),
            ),
            _scenario("global_halt", _halt),
            _scenario("price_gap", _gap),
            _scenario(
                "cancel_fill_race",
                lambda: _cancel_fill_race(base / "cancel-race.sqlite3"),
            ),
        ]
        return {
            "ok": all(row.get("ok", False) for row in rows),
            "scenario_count": len(rows),
            "passed_count": sum(1 for row in rows if row.get("ok", False)),
            "scenarios": rows,
        }
    finally:
        if temporary is not None:
            temporary.cleanup()
