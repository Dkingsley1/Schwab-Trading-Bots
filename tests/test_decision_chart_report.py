"""Offline contract fixtures only; fabricated prices are never runtime evidence."""

from copy import deepcopy
from datetime import timedelta
import json
import gzip
from pathlib import Path

from PIL import Image, ImageStat
import pytest

from core.decision_price_evidence import digest, timestamp
from core.schd_capture_store import DIRECTORY as CAPTURE_DIRECTORY, preserve_capture
from scripts.ops import decision_chart_report as charts
from scripts.ops.schd_candle_report import render_charts


@pytest.fixture
def evidence(tmp_path):
    start = timestamp("2026-09-23T13:30:00+00:00")
    bars = [
        dict(
            start_utc=(start + timedelta(minutes=i)).isoformat(),
            end_utc=(start + timedelta(minutes=i + 1)).isoformat(),
            open=32,
            high=33,
            low=31,
            close=32.5,
            volume=1000,
        )
        for i in range(30)
    ]
    captured = start + timedelta(minutes=30)
    capture = {
        "source": {
            "provider": "schwab",
            "symbol": "SCHD",
            "fetch_started_at_utc": captured.isoformat(),
            "requests": [
                {
                    "timeframe": "1m",
                    "endpoint": "get_price_history_every_minute",
                    "payload_sha256": "a" * 64,
                    "closed_bars": len(bars),
                }
            ],
        },
        "candles": {"1m": bars},
    }
    identity = preserve_capture(tmp_path, capture)
    decision = {
        "decision_id": "original-decision",
        "timestamp_utc": captured.isoformat(),
        "symbol": "SCHD",
        "action": "BUY",
        "decision": "EXECUTE",
        "quantity": 1,
        "metadata": {
            "schd_candle_context": {
                "state": "observed_context_not_claimed_model_input",
                "capture_sha256": identity,
                "fetch_started_at_utc": captured.isoformat(),
                "observed_at_utc": captured.isoformat(),
            }
        },
    }
    order = {
        "orderId": 123,
        "status": "FILLED",
        "filledQuantity": 1,
        "price": 99,
        "orderLegCollection": [
            {
                "legId": 1,
                "quantity": 1,
                "instruction": "BUY",
                "instrument": {"symbol": "SCHD", "assetType": "EQUITY"},
            }
        ],
        "orderActivityCollection": [
            {
                "activityId": 456,
                "activityType": "EXECUTION",
                "executionType": "FILL",
                "executionLegs": [
                    {
                        "legId": 1,
                        "quantity": 1,
                        "price": 32.7,
                        "time": (captured + timedelta(seconds=30)).isoformat(),
                    }
                ],
            }
        ],
    }
    receipt = {
        "owner": "supervised_broker_test.reconcile_order",
        "provider": "schwab",
        "mode": "live",
        "reconciled": True,
        "decision_id": decision["decision_id"],
        "broker_order_id": "123",
        "broker_order_sha256": digest(order),
        "broker_order": order,
        "observed_at_utc": (captured + timedelta(minutes=1)).isoformat(),
    }
    return tmp_path, decision, capture, receipt, captured + timedelta(minutes=10)


def test_original_context_and_no_execution_inferred(evidence):
    root, decision, capture, _, now = evidence
    original = deepcopy(decision)
    report = charts.build_decision_chart(root, decision, now=now)
    assert report["status"] == "available"
    assert report["as_of_utc"] == decision["timestamp_utc"]
    assert decision == original
    assert report["timeframes"]["1m"]["chart_candles"] == capture["candles"]["1m"]
    assert report["timeframes"]["5m"]["status"] == "unavailable"
    assert report["execution_status"] == "unavailable"
    assert [m["kind"] for m in report["decision_chart_markers"]] == ["proposed"]
    assert not report["live_execution_authority"] and not report["order_authority"]


def test_reconciliation_retains_execution_legs_and_report_reads_without_writer(
    evidence,
):
    from core.live_order_ledger import LiveOrderLedger
    from core.supervised_broker_test import reconcile_order
    from datetime import datetime, timezone

    root, decision, _, receipt, _ = evidence
    order = deepcopy(receipt["broker_order"])
    order.update(orderType="LIMIT", session="NORMAL", duration="DAY")
    packet = {"decision": decision}
    payload = {
        "symbol": "SCHD",
        "action": "BUY",
        "order_spec": order,
        "bot_handoff": {"packet": packet, "packet_sha256": digest(packet)},
    }
    ledger = LiveOrderLedger(root / "governance/runtime/live_order_ledger.sqlite3")
    ledger.reserve(intent_id="test", payload=payload, requested_quantity=1)
    ledger.mark_submitting("test")
    ledger.mark_submit_result(
        intent_id="test", acknowledged=True, broker_order_id="123"
    )
    result = reconcile_order(ledger, ledger.get("test"), order)
    assert result["state"] == "filled"
    before = ledger.get("test")
    receipts = charts.read_ledger_executions(root, decision["decision_id"])
    assert len(receipts) == 1
    assert receipts[0]["broker_order"] == order
    markers, issues = charts.execution_markers(
        decision, receipts, now=datetime.now(timezone.utc)
    )
    assert not issues
    assert markers[0]["price"] == 32.7
    assert (
        markers[0]["timestamp_utc"]
        == order["orderActivityCollection"][0]["executionLegs"][0]["time"]
    )
    assert charts.read_ledger_executions(root, "another-decision") == []
    assert ledger.get("test") == before
    with ledger._connect() as connection:
        connection.execute(
            "UPDATE order_events SET details_json = replace(details_json, '32.7', '32.8') WHERE to_state='filled'"
        )
    with pytest.raises(ValueError, match="event_hash_mismatch"):
        charts.read_ledger_executions(root, decision["decision_id"])


def test_chart_capture_without_bound_native_decision_does_not_infer_one(evidence):
    from core.supervised_broker_test import chart_execution_receipt

    _, _, _, receipt, now = evidence
    assert chart_execution_receipt({}, receipt["broker_order"], observed_at=now) is None


@pytest.mark.parametrize(
    "change",
    ["binding", "missing", "provider", "future", "unclosed", "receipt", "symbol"],
)
def test_fail_closed_capture_contract(evidence, change):
    root, decision, capture, _, now = evidence
    context = decision["metadata"]["schd_candle_context"]
    if change == "binding":
        context.clear()
    elif change == "missing":
        context["capture_sha256"] = "f" * 64
    elif change == "provider":
        capture["source"]["provider"] = "synthetic"
    elif change == "future":
        capture["source"]["fetch_started_at_utc"] = now.isoformat()
    elif change == "unclosed":
        capture["candles"]["1m"][-1]["end_utc"] = now.isoformat()
    elif change == "receipt":
        capture["source"]["requests"] = []
    else:
        decision["symbol"] = "O"
    if change in {"provider", "future", "unclosed", "receipt"}:
        context["capture_sha256"] = preserve_capture(root, capture)
    report = charts.build_decision_chart(root, decision, now=now)
    assert report["status"] == "unavailable"
    assert not any(f.get("chart_candles") for f in report["timeframes"].values())


def test_fills_use_actual_prices_and_deduplicate(evidence):
    root, decision, _, receipt, now = evidence
    report = charts.build_decision_chart(
        root, decision, executions=[receipt, receipt], now=now
    )
    fills = [m for m in report["decision_chart_markers"] if m["kind"] == "executed"]
    assert len(fills) == 1 and fills[0]["price"] == 32.7
    assert (
        fills[0]["timestamp_utc"]
        == receipt["broker_order"]["orderActivityCollection"][0]["executionLegs"][0][
            "time"
        ]
    )


@pytest.mark.parametrize(
    "change",
    [
        "paper",
        "owner",
        "decision",
        "order",
        "symbol",
        "side",
        "time",
        "quantity",
        "price",
        "missing_legs",
        "leg_id",
        "duplicate",
        "hash",
    ],
)
def test_execution_rejection(evidence, change):
    root, decision, _, receipt, now = evidence
    order = receipt["broker_order"]
    activity = order["orderActivityCollection"][0]
    leg = activity["executionLegs"][0]
    if change == "paper":
        receipt["mode"] = "paper"
    elif change == "owner":
        receipt["owner"] = "simulator"
    elif change == "decision":
        receipt["decision_id"] = "someone-else"
    elif change == "order":
        receipt["broker_order_id"] = "321"
    elif change == "symbol":
        order["orderLegCollection"][0]["instrument"]["symbol"] = "O"
    elif change == "side":
        order["orderLegCollection"][0]["instruction"] = "SELL"
    elif change == "time":
        leg["time"] = (now + timedelta(days=1)).isoformat()
    elif change == "quantity":
        leg["quantity"] = 0.5
    elif change == "price":
        leg["price"] = 0
    elif change == "missing_legs":
        activity["executionLegs"] = []
    elif change == "leg_id":
        leg["legId"] = 2
    elif change == "duplicate":
        activity["executionLegs"].append(dict(leg))
    if change != "hash":
        receipt["broker_order_sha256"] = digest(order)
    else:
        receipt["broker_order_sha256"] = "0" * 64
    report = charts.build_decision_chart(root, decision, executions=[receipt], now=now)
    assert report["execution_status"] == "unavailable"
    assert "broker_execution_evidence_invalid_or_unbound" in report["issues"]


def test_review_is_separate_and_fills_map_to_candle_time(
    evidence, monkeypatch, tmp_path
):
    root, decision, capture, receipt, now = evidence
    captured = timestamp(capture["source"]["fetch_started_at_utc"])
    capture["candles"]["1m"].append(
        dict(
            capture["candles"]["1m"][-1],
            start_utc=captured.isoformat(),
            end_utc=(captured + timedelta(minutes=1)).isoformat(),
        )
    )
    capture["source"]["fetch_started_at_utc"] = (
        captured + timedelta(minutes=1)
    ).isoformat()
    capture["source"]["requests"][0]["closed_bars"] += 1
    identity = preserve_capture(root, capture)
    review = charts.build_execution_review(
        root, decision, capture_sha256=identity, executions=[receipt], now=now
    )
    original = charts.build_decision_chart(
        root, decision, executions=[receipt], now=now
    )
    assert len(original["timeframes"]["1m"]["chart_candles"]) == 30
    assert len(review["timeframes"]["1m"]["chart_candles"]) == 31
    assert review["original_decision_timestamp_utc"] == decision["timestamp_utc"]
    assert review["decision_input_eligible"] is False
    from matplotlib.axes import Axes

    actual, scatter = [], Axes.scatter

    def spy(self, x, y, **kwargs):
        actual.append((x, y, kwargs))
        return scatter(self, x, y, **kwargs)

    monkeypatch.setattr(Axes, "scatter", spy)
    paths = render_charts(
        review, directory=tmp_path, prefix="fixture-review", checked_path=lambda p: p
    )
    fills = [(x, y) for x, y, options in actual if options["label"] == "EXECUTED BUY"]
    assert fills == [(30.0, 32.7)]
    assert any(options["facecolors"] == "none" for _, _, options in actual)
    with Image.open(paths["1m"]) as image:
        assert min(ImageStat.Stat(image.convert("RGB")).stddev) > 15
    actual.clear()
    render_charts(
        original,
        directory=tmp_path,
        prefix="fixture-original",
        checked_path=lambda p: p,
    )
    assert not any(options["label"] == "EXECUTED BUY" for _, _, options in actual)


def test_partial_sell_fill_and_unfilled_cancel(evidence):
    root, decision, _, receipt, now = evidence
    decision["action"] = "SELL"
    order = receipt["broker_order"]
    order["orderLegCollection"][0]["instruction"] = "SELL"
    order["status"] = "CANCELED"
    order["filledQuantity"] = 0.25
    order["orderActivityCollection"][0]["executionLegs"][0]["quantity"] = 0.25
    receipt["broker_order_sha256"] = digest(order)
    report = charts.build_decision_chart(root, decision, executions=[receipt], now=now)
    assert report["decision_chart_markers"][-1]["quantity"] == 0.25
    assert report["decision_chart_markers"][-1]["action"] == "SELL"
    order["filledQuantity"] = 0
    order["orderActivityCollection"] = []
    receipt["broker_order_sha256"] = digest(order)
    assert (
        charts.build_decision_chart(root, decision, executions=[receipt], now=now)[
            "execution_status"
        ]
        == "unavailable"
    )


def test_dedup_bounded_cache_and_links(evidence, monkeypatch):
    root, decision, _, receipt, now = evidence
    first = charts.publish_decision_chart(root, decision, executions=[receipt], now=now)
    assert first["report_path"] and first["status"] == "available"
    directory = root / charts.DIRECTORY
    before = {p.name: p.stat().st_mtime_ns for p in directory.iterdir()}
    second = charts.publish_decision_chart(
        root, decision, executions=[receipt], now=now + timedelta(minutes=1)
    )
    assert first == second
    assert before == {p.name: p.stat().st_mtime_ns for p in directory.iterdir()}
    page = (root / first["report_path"]).read_text()
    assert ".png" in page and "data:image" not in page
    assert sum(p.stat().st_size for p in directory.iterdir()) < charts.MAX_REPORT_BYTES
    monkeypatch.setattr(charts, "MAX_FILES", len(before) + 1)
    changed = dict(decision, decision_id="different-decision")
    assert charts.publish_decision_chart(root, changed, now=now)["report_path"] is None


def test_protected_output_route_and_missing_capture_report(evidence, tmp_path):
    root, decision, _, _, now = evidence
    decision["metadata"] = {}
    receipt = charts.publish_decision_chart(root, decision, now=now)
    assert receipt["status"] == "unavailable" and receipt["report_path"]
    report = json.loads(
        (root / receipt["report_path"]).with_suffix(".json").read_text()
    )
    assert not report["timeframes"]
    other = tmp_path / "other"
    other.mkdir()
    (other / "governance").symlink_to(root / "governance", target_is_directory=True)
    assert (
        charts.publish_decision_chart(other, decision, now=now)["report_path"] is None
    )


def test_conflicting_execution_snapshots_and_corrupt_cache(evidence):
    root, decision, _, receipt, now = evidence
    changed = deepcopy(receipt)
    changed["broker_order"]["orderActivityCollection"][0]["executionLegs"][0][
        "price"
    ] = 32.8
    changed["broker_order_sha256"] = digest(changed["broker_order"])
    report = charts.build_decision_chart(
        root, decision, executions=[receipt, changed], now=now
    )
    assert report["execution_status"] == "unavailable"
    assert "conflicting_broker_execution_evidence" in report["issues"]
    link = charts.publish_decision_chart(root, decision, now=now)
    image = next((root / charts.DIRECTORY).glob("*.png"))
    image.write_bytes(b"broken test asset")
    result = charts.publish_decision_chart(root, decision, now=now)
    assert result["report_path"] is None
    assert "decision_chart_cache_corrupt_or_incomplete" in result["issues"]


def test_cli_publishes_real_linked_sidecar_from_exact_original(
    evidence, monkeypatch, capsys
):
    root, decision, _, _, now = evidence
    path = root / "original.jsonl"
    path.write_text(json.dumps(decision) + "\n")
    publish = charts.publish_decision_chart
    monkeypatch.setattr(
        charts, "publish_decision_chart", lambda *a, **k: publish(*a, now=now, **k)
    )
    assert (
        charts.main(
            [
                "--root",
                str(root),
                "--decision-log",
                path.name,
                "--decision-id",
                decision["decision_id"],
            ]
        )
        == 0
    )
    receipt = json.loads(capsys.readouterr().out)
    assert (root / receipt["report_path"]).is_file()
    sidecar = json.loads((root / receipt["sidecar_path"]).read_text())
    assert sidecar["original_decision_timestamp_utc"] == decision["timestamp_utc"]
    assert sidecar["decision_id"] == decision["decision_id"]
    assert any(name.endswith(".png") for name in sidecar["chart_asset_sha256"])
    assert (
        charts.main(
            [
                "--root",
                str(root),
                "--decision-log",
                path.name,
                "--decision-id",
                "missing",
            ]
        )
        == 2
    )
    assert json.loads(capsys.readouterr().out)["report_path"] is None


def test_original_log_scan_gzip_bounds_and_ambiguity(evidence, monkeypatch):
    root, decision, _, _, _ = evidence
    path = root / "original.jsonl.gz"
    raw = (json.dumps(decision) + "\n").encode()
    with gzip.open(path, "wb") as stream:
        stream.write(raw)
    assert (
        charts.read_original_decision(root, path, decision_id=decision["decision_id"])
        == decision
    )
    monkeypatch.setattr(charts, "MAX_LOG_BYTES", 10)
    with pytest.raises(ValueError, match="budget"):
        charts.read_original_decision(root, path, decision_id=decision["decision_id"])
    monkeypatch.setattr(charts, "MAX_LOG_BYTES", 10000)
    plain = root / "original.jsonl"
    plain.write_bytes(raw + json.dumps(dict(decision, action="SELL")).encode())
    with pytest.raises(ValueError, match="ambiguous"):
        charts.read_original_decision(root, plain, decision_id=decision["decision_id"])
