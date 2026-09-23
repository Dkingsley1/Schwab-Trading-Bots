from copy import deepcopy
from datetime import timedelta
from decimal import Decimal
import json

import pytest

from core.decision_price_evidence import (
    aggregate_bars,
    build_evidence,
    describe_candle,
    timestamp,
    validate_bars,
)
from core.schd_decision_rehearsal import (
    AUTHORITY,
    advance,
    initial_state,
    render_markdown,
    validate_state,
)
from scripts.ops.schd_decision_rehearsal import (
    DIRECTORY,
    demo,
    read_json,
    run,
    synthetic_packet,
)


@pytest.fixture(scope="module")
def source():
    return synthetic_packet()


@pytest.fixture
def packet(source):
    return deepcopy(source)


def start(packet):
    return initial_state(
        candidate_id=packet["candidate_id"],
        evidence_kind=packet["evidence_kind"],
        implementation_sha256="source",
    )


def step(state, packet, seconds=0, now=None):
    now = now or timestamp("2026-09-23T15:05:01+00:00") + timedelta(seconds=seconds)
    return advance(state, packet, now=now, implementation_sha256="source")


def fill(packet, *, seconds=1, snapshot="new-fill", price=None):
    quote = dict(
        packet["quote"],
        timestamp_utc=(
            timestamp(packet["decision"]["timestamp_utc"]) + timedelta(seconds=seconds)
        ).isoformat(),
        snapshot_id=snapshot,
    )
    if price is not None:
        quote.update(bid=price, ask=price + 0.01, last=price)
    return {
        "candidate_id": packet["candidate_id"],
        "evidence_kind": packet["evidence_kind"],
        "fill_quote": quote,
    }


def test_complete_detailed_packet(source):
    report = build_evidence(source, now=timestamp(source["decision"]["timestamp_utc"]))
    assert report["blockers"] == []
    assert report["decision_status"] == "SIMULATE_BUY"
    assert report["timeframes"]["1m"]["status"] == "missing"
    for name in ("5m", "15m", "1h", "1d", "1M", "180d", "1Y"):
        frame = report["timeframes"][name]
        assert frame["status"] == "complete", (name, frame)
        assert frame["recent_candles"][0]["upper_wick_usd"] >= 0
    assert report["timeframes"]["1Y"]["recent_candles"][-1]["period"] == "2025"
    assert report["timeframes"]["1M"]["recent_candles"][-1]["period"] == "2026-08"
    assert report["timeframes"]["180d"]["window_end_date"] == "2026-09-22"
    assert report["timeframes"]["180d"]["recent_candles"][0]["trading_sessions"] < 180
    assert report["bot_record"]["reasons"] == source["decision"]["reasons"]
    assert "not proof" in report["attribution"]
    assert report["corporate_action_context"]["independently_verified"] is False


@pytest.mark.parametrize(
    "mutation,reason",
    [
        (lambda p: p["quote"].update(ask=p["quote"]["bid"]), "quote_unusable"),
        (lambda p: p["quote"].update(bid=float("nan")), "quote_unusable"),
        (lambda p: p["quote"].update(bid=True), "quote_unusable"),
        (
            lambda p: p["quote"].update(snapshot_id="unbound"),
            "decision_quote_snapshot_mismatch",
        ),
        (lambda p: p["quote"].update(ask=35), "spread_above"),
        (lambda p: p["decision"].update(gates={}), "bot_gates"),
        (lambda p: p["decision"].update(gates={"risk": "yes"}), "bot_gates"),
        (lambda p: p["decision"].update(reasons=[]), "recorded_bot_reasons_missing"),
        (lambda p: p["decision"].update(action="HOLD"), "bot_did_not_request"),
        (lambda p: p["decision"].update(decision="BLOCK"), "bot_did_not_request"),
        (lambda p: p.update(symbol="O"), "schd_only"),
        (lambda p: p.update(price_basis="unknown"), "price_basis_required"),
        (lambda p: p["candles"]["5m"].pop(-10), "5m_evidence_incomplete"),
        (lambda p: p["candles"]["1d"].pop(10), "1Y_evidence_incomplete"),
        (lambda p: p["candles"]["1d"].pop(-25), "180d_evidence_incomplete"),
        (lambda p: p["candles"]["5m"][-1].update(volume=-1), "5m_evidence_incomplete"),
    ],
)
def test_bad_or_missing_evidence_does_not_create_order(packet, mutation, reason):
    mutation(packet)
    # Nonfinite inputs are invalid JSON and must fail serialization as well.
    if (
        isinstance(packet["quote"].get("bid"), float)
        and packet["quote"]["bid"] != packet["quote"]["bid"]
    ):
        with pytest.raises(ValueError):
            step(start(packet), packet)
        return
    state = step(start(packet), packet)
    assert state["phase"] == "waiting_entry"
    assert any(reason in item for item in state["last_report"]["blockers"])
    assert not state["pending"] and not state["orders"]


@pytest.mark.parametrize("seconds", [-1, 121])
def test_decision_future_or_stale(packet, seconds):
    state = step(start(packet), packet, seconds=seconds)
    assert "decision_stale_or_future" in state["last_report"]["blockers"]


@pytest.mark.parametrize("seconds", [-31, 1])
def test_quote_future_or_stale(packet, seconds):
    packet["quote"]["timestamp_utc"] = (
        timestamp(packet["decision"]["timestamp_utc"]) + timedelta(seconds=seconds)
    ).isoformat()
    state = step(start(packet), packet)
    assert state["phase"] == "waiting_entry"
    assert any("quote_stale_or_future" in s for s in state["last_report"]["blockers"])


def test_candle_math_and_unclosed_rejection(packet):
    row = dict(packet["candles"]["5m"][-1], open=10, high=14, low=8, close=12)
    details = describe_candle(row)
    assert details["body_usd"] == 2
    assert details["upper_wick_usd"] == 2
    assert details["lower_wick_usd"] == 2
    assert details["close_position_in_candle"] == pytest.approx(2 / 3)
    with pytest.raises(ValueError, match="unclosed"):
        validate_bars([row], minutes=5, asof=timestamp(row["start_utc"]))


def test_aggregation_does_not_bridge_missing_bars(packet):
    rows = packet["candles"]["5m"][:12]
    full, _ = aggregate_bars(rows, 15)
    incomplete, skipped = aggregate_bars(rows[:1] + rows[2:], 15)
    assert len(incomplete) == len(full) - 1
    assert skipped == 1


def test_dst_early_close_and_weekend():
    from core.decision_price_evidence import latest_closed_end, session_bounds

    before = session_bounds(timestamp("2026-03-06T17:00:00+00:00"))
    after = session_bounds(timestamp("2026-03-09T17:00:00+00:00"))
    assert before[0].hour == 14 and after[0].hour == 13
    early = session_bounds(timestamp("2026-11-27T17:00:00+00:00"))
    assert early[1].hour == 18
    assert session_bounds(timestamp("2026-09-26T17:00:00+00:00")) is None
    assert (
        latest_closed_end(timestamp("2026-09-26T17:00:00+00:00"), None)
        .date()
        .isoformat()
        == "2026-09-25"
    )


def test_round_trip_reconciles_without_double_counting_fees():
    state = demo(source_digest="source")
    assert state["phase"] == "complete"
    assert [o["side"] for o in state["orders"]] == ["BUY", "SELL"]
    buy, sell = (Decimal(o["modeled_fill_price_usd"]) for o in state["orders"])
    assert Decimal(state["virtual_cash_usd"]) == Decimal("100") - buy + sell
    assert state["reconciliation"]["cash_delta_equals_modeled_pnl"]
    assert state["virtual_shares"] == 0
    assert all(state[k] is False for k in AUTHORITY)
    assert state["orders"][0]["execution_model"]["effective_fill_ratio"] < 1
    assert (
        state["orders"][0]["fill_status"]
        == "conditional_assumed_full_share_not_observed"
    )
    text = render_markdown(state)
    for phrase in (
        "SIMULATION ONLY",
        "Recorded Bot Rationale",
        "Opposing Or Neutral Context",
        "180d",
        "Corporate Actions",
        "Upper Wick",
        "source_sha256",
    ):
        assert phrase in text


def test_same_quote_or_early_quote_cannot_fill(packet):
    state = step(start(packet), packet)
    same = fill(packet, snapshot=packet["quote"]["snapshot_id"])
    assert step(state, same, seconds=1)["phase"] == "pending_buy"
    early = fill(packet, seconds=0)
    assert step(state, early, seconds=1)["phase"] == "pending_buy"
    state = step(state, fill(packet), seconds=1)
    assert state["phase"] == "holding"
    assert len(state["orders"]) == 1


def test_expiry_no_retry_or_reentry(packet):
    pending = step(start(packet), packet)
    cancelled = step(pending, fill(packet), seconds=60)
    assert cancelled["phase"] == "cancelled"
    assert step(cancelled, packet, seconds=61) == cancelled


@pytest.mark.parametrize(
    "outcome,phase",
    [
        ("no_fill", "cancelled"),
        ("rejected", "cancelled"),
        ("partial", "reconciliation_required"),
    ],
)
def test_unfilled_scenarios_fail_closed(packet, outcome, phase):
    state = step(start(packet), packet)
    request = dict(fill(packet), scenario_outcome=outcome)
    state = step(state, request, seconds=1)
    assert state["phase"] == phase
    assert state["orders"] == []
    assert step(state, packet, seconds=2) == state


def test_loss_is_not_hidden_and_terminal_cannot_repeat(packet):
    state = step(step(start(packet), packet), fill(packet), seconds=1)
    sell = deepcopy(packet)
    sell["decision"].update(
        action="SELL",
        decision_id="exit-loss",
        timestamp_utc="2026-09-23T15:05:03+00:00",
    )
    sell["quote"].update(timestamp_utc=sell["decision"]["timestamp_utc"])
    state = step(state, sell, seconds=2)
    state = step(state, fill(sell, price=32), seconds=3)
    assert state["phase"] == "complete"
    assert Decimal(state["reconciliation"]["modeled_net_pnl_usd"]) < 0
    assert not state["reconciliation"]["sold_above_modeled_buy"]
    assert step(state, packet, seconds=4) == state


def test_candidate_kind_source_and_ledger_tampering_rejected(packet):
    state = start(packet)
    for key in ("candidate_id", "evidence_kind"):
        bad = dict(packet, **{key: "other"})
        with pytest.raises(ValueError, match="changed"):
            step(state, bad)
    with pytest.raises(ValueError, match="changed"):
        advance(
            state,
            packet,
            now=timestamp(packet["decision"]["timestamp_utc"]),
            implementation_sha256="changed",
        )
    state["virtual_shares"] = 1
    with pytest.raises(ValueError, match="ledger"):
        validate_state(state)


def test_cli_demo_is_separate_and_status_is_read_only(tmp_path):
    status = run("status", root=tmp_path, source_digest="source")
    assert status["phase"] == "waiting_entry"
    assert not (tmp_path / DIRECTORY).exists()
    result = run("demo", root=tmp_path, source_digest="source")
    assert result["phase"] == "complete"
    assert (tmp_path / DIRECTORY / "synthetic_latest.json").is_file()
    assert not (tmp_path / DIRECTORY / "recorded_state.json").exists()
    assert run("status", root=tmp_path, source_digest="source")["orders"] == []


def test_cli_restart_preserves_one_entry(tmp_path, packet):
    packet["evidence_kind"] = "recorded"
    path = tmp_path / "packet.json"
    path.write_text(json.dumps(packet))
    now = timestamp(packet["decision"]["timestamp_utc"])
    pending = run(
        "evaluate", root=tmp_path, source_digest="source", input_path=path, now=now
    )
    assert pending["phase"] == "pending_buy"
    path.write_text(json.dumps(fill(packet)))
    filled = run(
        "evaluate",
        root=tmp_path,
        source_digest="source",
        input_path=path,
        now=now + timedelta(seconds=1),
    )
    assert filled["phase"] == "holding"
    path.write_text(json.dumps(packet))
    repeated = run(
        "evaluate",
        root=tmp_path,
        source_digest="source",
        input_path=path,
        now=now + timedelta(seconds=2),
    )
    assert len(repeated["orders"]) == 1
    assert repeated["phase"] == "holding"


def test_cli_rejects_corrupt_state_and_protected_route(tmp_path, monkeypatch):
    path = tmp_path / DIRECTORY
    path.mkdir(parents=True)
    (path / "recorded_state.json").write_text("{bad")
    with pytest.raises(ValueError):
        run("status", root=tmp_path, source_digest="source")
    (path / "recorded_state.json").unlink()
    (path / "recorded_state.json").symlink_to("/Volumes/VIDEO/do-not-read.json")
    from pathlib import Path

    original = Path.lstat

    def guard(self, *args, **kwargs):
        assert not str(self).startswith("/Volumes/VIDEO")
        return original(self, *args, **kwargs)

    monkeypatch.setattr(Path, "lstat", guard)
    with pytest.raises(ValueError, match="unsafe"):
        run("status", root=tmp_path, source_digest="source")


def test_cli_bounds_and_power_off(tmp_path):
    path = tmp_path / "input.json"
    path.write_text('{"value": NaN}')
    with pytest.raises(ValueError):
        read_json(path)
    flag = tmp_path / "governance/health/SYSTEM_POWER_OFF.flag"
    flag.parent.mkdir(parents=True)
    flag.touch()
    with pytest.raises(ValueError, match="system_power_off"):
        run("demo", root=tmp_path, source_digest="source")
