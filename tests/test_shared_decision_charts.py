"""Synthetic offline fixtures, not market evidence."""

from copy import deepcopy
from datetime import timedelta
import json

import pytest

from core import decision_candle_store as store
from core.decision_logger import DecisionLogger
from core.decision_price_evidence import timestamp
from scripts.ops import decision_chart_report as charts


def fixture(root, symbol="BTC-USD", provider="coinbase"):
    start = (
        timestamp("2026-09-19T00:00:00Z")
        if provider == "coinbase"
        else timestamp("2026-09-18T13:30:00Z")
    )
    now = start + timedelta(minutes=150)
    rows = [
        dict(
            start_utc=(start + timedelta(minutes=i * 5)).isoformat(),
            end_utc=(start + timedelta(minutes=(i + 1) * 5)).isoformat(),
            open=100,
            high=102,
            low=99,
            close=101,
            volume=5,
        )
        for i in range(30)
    ]
    capture = {
        "source": {
            "symbol": symbol,
            "provider": provider,
            "fetch_started_at_utc": now.isoformat(),
            "observed_at_utc": now.isoformat(),
            "requests": [
                {
                    "timeframe": "5m",
                    "endpoint": (
                        f"/products/{symbol}/candles"
                        if provider == "coinbase"
                        else "get_price_history_every_five_minutes"
                    ),
                    "payload_sha256": "a" * 64,
                    "closed_bars": 30,
                }
            ],
        },
        "candles": {"5m": rows},
    }
    receipt = store.publish(root, capture)
    decision = {
        "symbol": symbol,
        "decision_id": "test-decision",
        "timestamp_utc": now.isoformat(),
        "action": "HOLD",
        "decision": "BLOCK",
        "strategy": "fixture",
        "reasons": ["rsi_above_buy_threshold"],
        "features": {"rsi14": 71},
        "threshold": 30,
        "gates": {"risk": False},
        "metadata": {"candle_context": receipt},
    }
    return capture, decision, now


@pytest.mark.parametrize("action", ["BUY", "SELL", "HOLD"])
@pytest.mark.parametrize(
    "provider,symbol",
    [
        ("coinbase", "BTC-USD"),
        ("coinbase", "ETH-USD"),
        ("schwab", "O"),
        ("schwab", "NVDA"),
    ],
)
def test_all_actions_symbols_provider_calendar(tmp_path, action, provider, symbol):
    _, decision, now = fixture(tmp_path, symbol, provider)
    decision["action"] = action
    report = charts.build_decision_chart(tmp_path, decision, now=now)
    assert report["status"] == "available"
    assert report["chart_source"]["provider"] == provider
    assert report["timeframes"]["5m"]["gap_count"] == 0
    assert report["recorded_reasoning"]["indicator_features"]["rsi14"] == 71
    assert report["decision_input_eligible"] is False
    assert not any(m["kind"] == "executed" for m in report["decision_chart_markers"])
    assert len(report["decision_chart_markers"]) == (action != "HOLD")


@pytest.mark.parametrize("change", ["provider", "symbol", "time", "hash", "observed"])
def test_identity_and_point_in_time_fail_closed(tmp_path, change):
    _, decision, now = fixture(tmp_path)
    receipt = decision["metadata"]["candle_context"]
    if change == "provider":
        receipt["provider"] = "schwab"
    if change == "symbol":
        decision["symbol"] = "ETH-USD"
    if change == "hash":
        receipt["capture_sha256"] = "b" * 64
    if change == "observed":
        receipt["observed_at_utc"] = (now - timedelta(seconds=1)).isoformat()
    if change == "time":
        decision["timestamp_utc"] = (now - timedelta(seconds=1)).isoformat()
    report = charts.build_decision_chart(tmp_path, decision, now=now)
    assert report["status"] == "unavailable"


def test_small_pointer_and_capacity_preserve_history(tmp_path, monkeypatch):
    capture, decision, now = fixture(tmp_path)
    assert (
        store.context_receipt(tmp_path, "BTC-USD", now=now)["capture_sha256"]
        == decision["metadata"]["candle_context"]["capture_sha256"]
    )
    assert (
        store.context_receipt(tmp_path, "BTC-USD", now=now + timedelta(hours=1))[
            "state"
        ]
        == "unavailable"
    )
    monkeypatch.setattr(store, "MAX_FILES", 3)
    capture["candles"]["5m"][0]["volume"] = 6
    with pytest.raises(ValueError, match="capacity"):
        store.publish(tmp_path, capture)
    assert store.read_bound(tmp_path, decision)


def test_shared_logger_attaches_context_every_action(tmp_path, monkeypatch):
    seen = []
    monkeypatch.setattr(
        store,
        "context_receipt",
        lambda *a, **kw: {"state": "unavailable", "reason": "test"},
    )
    logger = DecisionLogger(str(tmp_path))
    monkeypatch.setattr(logger, "_append_jsonl", seen.append)
    for action in ("BUY", "SELL", "HOLD"):
        row = logger.log_decision(
            symbol="O",
            action=action,
            model_score=0.5,
            threshold=0.6,
            quantity=1,
            features={},
            gates={"risk": False},
            reasons=["risk"],
        )
        assert row["metadata"]["candle_context"]["reason"] == "test"
    assert len(seen) == 3


def test_report_reasoning_html_escaped_and_separated(tmp_path):
    _, decision, now = fixture(tmp_path)
    decision["reasons"] = ["<script>bad</script>"]
    report = charts.build_decision_chart(tmp_path, decision, now=now)
    page = charts._page(report, {}).decode()
    assert "<script>" not in page
    assert "Recorded Indicator Reasoning" in page
    assert "Review calculations, not claimed bot inputs" in page
    assert "71" in page


def test_coinbase_gap_is_not_filled(tmp_path):
    capture, _, now = fixture(tmp_path)
    del capture["candles"]["5m"][10]
    capture["source"]["requests"][0]["closed_bars"] = 29
    frames = charts._frames(capture, asof=now, available_at=now)
    assert frames["5m"]["gap_count"] == 1
    assert "candle_gaps" in frames["5m"]["issues"]


def test_real_png_coinbase_label_and_pixels(tmp_path):
    from PIL import Image, ImageStat

    _, decision, now = fixture(tmp_path)
    receipt = charts.publish_decision_chart(tmp_path, decision, now=now)
    assert receipt["status"] == "available"
    saved = json.loads((tmp_path / receipt["sidecar_path"]).read_text())
    assert saved["chart_source"]["provider"] == "coinbase"
    png = next((tmp_path / charts.DIRECTORY).glob("*.png"))
    with Image.open(png) as image:
        assert min(image.size) > 500
        assert max(ImageStat.Stat(image.convert("RGB")).stddev) > 10
