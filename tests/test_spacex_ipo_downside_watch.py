import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.ops import spacex_ipo_downside_watch as src


def test_waits_for_first_quote_without_alert() -> None:
    payload, state, alert = src.evaluate_watch(
        symbol="SPCX",
        quote={"ok": False, "error": "quote_not_found"},
        state={},
        bands=[0.05, 0.10],
    )

    assert payload["overall_status"] == "waiting_for_first_quote"
    assert payload["alert"]["triggered"] is False
    assert state["last_quote_error"] == "quote_not_found"
    assert alert is None


def test_first_quote_arms_watch_without_alert() -> None:
    payload, state, alert = src.evaluate_watch(
        symbol="SPCX",
        quote={"ok": True, "source": "test", "last_price": 100.0, "spread_bps": 20.0},
        state={},
        bands=[0.05, 0.10],
    )

    assert payload["overall_status"] == "first_quote_seen"
    assert state["first_print_price"] == 100.0
    assert state["high_price"] == 100.0
    assert alert is None


def test_drop_from_first_print_triggers_critical_alert() -> None:
    state = {"first_print_price": 100.0, "high_price": 103.0}

    payload, new_state, alert = src.evaluate_watch(
        symbol="SPCX",
        quote={"ok": True, "source": "test", "last_price": 89.0, "spread_bps": 40.0},
        state=state,
        bands=[0.05, 0.10, 0.15],
    )

    assert payload["overall_status"] == "alert"
    assert payload["metrics"]["drop_from_first_print_pct"] == 11.0
    assert alert is not None
    assert alert["event"] == "spacex_ipo_downside_watch"
    assert alert["severity"] == "critical"
    assert "from_first_print:0.100000" in new_state["alerted"]
    assert "no automatic order" in alert["message"]


def test_high_watermark_drop_triggers_when_first_print_is_intact() -> None:
    state = {"first_print_price": 100.0, "high_price": 120.0}

    payload, new_state, alert = src.evaluate_watch(
        symbol="SPCX",
        quote={"ok": True, "source": "test", "last_price": 108.0, "spread_bps": 30.0},
        state=state,
        bands=[0.05, 0.10],
    )

    assert payload["overall_status"] == "alert"
    assert payload["metrics"]["drop_from_first_print_pct"] == 0.0
    assert payload["metrics"]["drop_from_high_pct"] == 10.0
    assert alert is not None
    assert "from_high:0.100000" in new_state["alerted"]


def test_alerted_band_is_deduped() -> None:
    state = {"first_print_price": 100.0, "high_price": 100.0}
    _, state_after_alert, first_alert = src.evaluate_watch(
        symbol="SPCX",
        quote={"ok": True, "source": "test", "last_price": 94.0, "spread_bps": 20.0},
        state=state,
        bands=[0.05],
    )

    payload, _, second_alert = src.evaluate_watch(
        symbol="SPCX",
        quote={"ok": True, "source": "test", "last_price": 94.0, "spread_bps": 20.0},
        state=state_after_alert,
        bands=[0.05],
    )

    assert first_alert is not None
    assert payload["overall_status"] == "armed"
    assert second_alert is None


def test_spread_alert_can_fire_independently() -> None:
    payload, state, alert = src.evaluate_watch(
        symbol="SPCX",
        quote={"ok": True, "source": "test", "last_price": 100.0, "bid_price": 95.0, "ask_price": 105.0, "spread_bps": 952.0},
        state={"first_print_price": 100.0, "high_price": 100.0},
        bands=[0.05],
        spread_bps_alert=500.0,
    )

    assert payload["overall_status"] == "alert"
    assert alert is not None
    assert "spread_bps" in state["alerted"]
    assert "spread_bps_ge_500" in payload["alert"]["payload"]["details"]["reasons"]
