from copy import deepcopy
from datetime import timedelta
import json
from pathlib import Path

import pytest

from core.decision_price_evidence import build_evidence, digest, timestamp
from scripts.ops.schd_candle_report import normalize_schwab_quote
from scripts.ops.schd_decision_rehearsal import DIRECTORY, run, synthetic_packet
from scripts.ops.schd_native_decision import (
    SOURCE_MODE,
    SOURCE_STRATEGY,
    advance_native,
    native_packet,
    read_latest,
)


@pytest.fixture(scope="module")
def fixture_data():
    base = synthetic_packet()
    now = timestamp(base["decision"]["timestamp_utc"])
    row = deepcopy(base["decision"])
    row.update(
        strategy=SOURCE_STRATEGY,
        production_candidate_id="candidate-test",
        source_broker="schwab",
        source_provider="schwab",
        source_quality_label="broker_native",
        source_quality_score=0.95,
        schema_valid=True,
    )
    row["metadata"].update(
        mode="shadow",
        layer="grand_master",
        source_profile="dividend",
        production_candidate_id="candidate-test",
        production_candidate_receipt_sha256="receipt-test",
    )
    row["candidate_binding"] = {
        "candidate_bound": True,
        "expected_candidate_id": "candidate-test",
        "observed_candidate_id": "candidate-test",
        "candidate_scope_cutoff_utc": "2026-09-01T00:00:00+00:00",
    }
    row["features"].update(
        last_price=33.0,
        bid_price=32.99,
        ask_price=33.0,
        bid_size=10000.0,
        ask_size=10000.0,
        snapshot_ts_utc=now.timestamp(),
        provider_quote_ts_utc=now.timestamp(),
        provider_quote_realtime_norm=1.0,
        provider_quote_last_price=33.0,
    )
    candidate = {"candidate_id": "candidate-test", "overall_sha256": "receipt-test"}
    market = {
        "candles": deepcopy(base["candles"]),
        "source": {
            "provider": "schwab",
            "symbol": "SCHD",
            "fetch_started_at_utc": (now - timedelta(seconds=1)).isoformat(),
            "price_adjustment_basis": "split_adjusted_dividends_unadjusted",
        },
    }
    row["metadata"]["schd_candle_context"] = {
        "state": "observed_context_not_claimed_model_input",
        "capture_sha256": digest(market),
        "observed_at_utc": now.isoformat(),
    }
    return row, candidate, market, now


def write_rows(root, rows, *, day="20260923", suffix=b""):
    path = root / "decisions" / SOURCE_MODE / f"trade_decisions_{day}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(
        b"".join(json.dumps(row).encode() + b"\n" for row in rows) + suffix
    )
    return path


def test_compressed_decision_is_read_without_admission_bypass(tmp_path, fixture_data):
    import gzip
    row, _, _, now = deepcopy(fixture_data)
    path = write_rows(tmp_path, [row], day=now.strftime("%Y%m%d"))
    archived = Path(str(path) + ".gz")
    archived.write_bytes(gzip.compress(path.read_bytes()))
    path.unlink()
    selected, receipt, scan = read_latest(tmp_path, now=now)
    assert selected["decision_id"] == row["decision_id"]
    assert receipt["offset_basis"] == "decompressed_bytes"
    assert not scan["issues"]
    selected, _, scan = read_latest(tmp_path, now=now, max_bytes=20)
    assert selected is None
    assert "native_compressed_scan_budget_exhausted" in scan["issues"]


def packet_from(root, rows, candidate, market, now):
    write_rows(root, rows)
    selected, receipt, scan = read_latest(root, now=now)
    return native_packet(selected, receipt, scan, market, candidate, now=now)


def test_real_reader_pins_role_and_preserves_hold(tmp_path, fixture_data):
    row, candidate, market, now = deepcopy(fixture_data)
    hold = deepcopy(row)
    hold.update(
        action="HOLD",
        decision_id="real-hold-fixture",
        timestamp_utc=(now + timedelta(seconds=1)).isoformat(),
        reasons=["defensive_buy_requires_risk_off", "risk_off_below_floor"],
    )
    unrelated = dict(hold, strategy="other_bot", action="BUY")
    packet = packet_from(
        tmp_path, [row, hold, unrelated], candidate, market, now + timedelta(seconds=2)
    )
    assert packet["decision"]["action"] == "HOLD"
    assert packet["decision"]["reasons"] == hold["reasons"]
    assert packet["decision"]["gates"] == hold["gates"]
    assert packet["native_validation"]["receipt"]["byte_offset"] > 0
    state = advance_native(
        None,
        packet,
        market,
        now=now + timedelta(seconds=2),
        source_digest="test-source",
    )
    assert state["phase"] == "waiting_entry"
    assert state["last_report"]["decision_status"] == "WAIT"
    assert state["native_connection"]["decision_id"] == "real-hold-fixture"
    assert state["orders"] == []


@pytest.mark.parametrize(
    "mutation,reason",
    [
        (
            lambda row: row["metadata"].update(source_profile="intraday"),
            "native_strategy_scope_mismatch",
        ),
        (
            lambda row: row.update(production_candidate_id="old"),
            "native_candidate_binding_mismatch",
        ),
        (
            lambda row: row["metadata"].update(
                production_candidate_receipt_sha256="old"
            ),
            "native_candidate_binding_mismatch",
        ),
        (
            lambda row: row["candidate_binding"].update(
                candidate_scope_cutoff_utc="2026-10-01T00:00:00+00:00"
            ),
            "native_decision_before_candidate_scope",
        ),
        (
            lambda row: row.update(source_quality_label="synthetic"),
            "native_source_provenance_incomplete",
        ),
        (
            lambda row: row.update(source_quality_score=0.5),
            "native_source_quality_below_floor",
        ),
        (
            lambda row: row.update(schema_valid=False),
            "native_source_provenance_incomplete",
        ),
        (
            lambda row: row["features"].pop("provider_quote_ts_utc"),
            "native_provider_quote_timestamp_missing",
        ),
        (
            lambda row: row["features"].pop("bid_price"),
            "native_recorded_quote_fields_incomplete",
        ),
    ],
)
def test_native_proof_gaps_cannot_simulate(tmp_path, fixture_data, mutation, reason):
    row, candidate, market, now = deepcopy(fixture_data)
    mutation(row)
    packet = packet_from(tmp_path, [row], candidate, market, now)
    state = advance_native(None, packet, market, now=now, source_digest="test-source")
    assert reason in state["last_report"]["blockers"]
    assert state["phase"] == "waiting_entry"


def test_reconstructed_candles_are_context_not_original_model_inputs(
    tmp_path, fixture_data
):
    row, candidate, market, now = deepcopy(fixture_data)
    market["source"]["fetch_started_at_utc"] = (now + timedelta(seconds=10)).isoformat()
    future = dict(
        market["candles"]["5m"][-1], end_utc=(now + timedelta(minutes=5)).isoformat()
    )
    market["candles"]["5m"].append(future)
    packet = packet_from(
        tmp_path, [row], candidate, market, now + timedelta(seconds=20)
    )
    assert packet["native_validation"]["post_decision_candles_excluded"] == 1
    assert future not in packet["candles"]["5m"]
    report = build_evidence(packet, now=now + timedelta(seconds=20))
    assert "native_candles_retrieved_after_decision_context_only" in report["blockers"]
    assert report["bot_record"]["reasons"] == row["reasons"]


@pytest.mark.parametrize(
    "suffix,reason",
    [
        (b'{"symbol":"SCHD"', "native_uncommitted_or_oversize_tail"),
        (b"malformed\n", "native_malformed_row"),
    ],
)
def test_malformed_tail_does_not_expose_previous_buy_as_ready(
    tmp_path, fixture_data, suffix, reason
):
    row, candidate, market, now = deepcopy(fixture_data)
    write_rows(tmp_path, [row], suffix=suffix)
    selected, receipt, scan = read_latest(tmp_path, now=now)
    packet = native_packet(selected, receipt, scan, market, candidate, now=now)
    assert reason in packet["native_validation"]["blockers"]
    assert (
        advance_native(None, packet, market, now=now, source_digest="s")["phase"]
        == "waiting_entry"
    )


def test_conflict_future_and_scan_bounds(tmp_path, fixture_data):
    row, candidate, market, now = deepcopy(fixture_data)
    conflict = dict(row, action="HOLD")
    write_rows(tmp_path, [row, conflict])
    _, _, scan = read_latest(tmp_path, now=now)
    assert "native_conflicting_decision_id" in scan["issues"]
    future = dict(
        row,
        timestamp_utc=(now + timedelta(minutes=1)).isoformat(),
        decision_id="future",
    )
    write_rows(tmp_path, [row, future])
    _, _, scan = read_latest(tmp_path, now=now)
    assert "native_invalid_decision_identity_or_time" in scan["issues"]
    _, _, scan = read_latest(tmp_path, now=now, max_bytes=80)
    assert scan["bytes_read"] <= 80
    assert scan["issues"]
    _, _, scan = read_latest(tmp_path, now=now, max_seconds=1e-12)
    assert "native_scan_budget_exhausted" in scan["issues"]


def test_older_day_and_protected_alias(tmp_path, fixture_data, monkeypatch):
    row, _, _, now = deepcopy(fixture_data)
    write_rows(tmp_path, [row], day="20260922")
    found, _, _ = read_latest(tmp_path, now=now)
    assert found["decision_id"] == row["decision_id"]
    target = tmp_path / "decisions" / SOURCE_MODE / "trade_decisions_20260923.jsonl"
    target.symlink_to("/Volumes/VIDEO/do-not-access.jsonl")
    original = Path.lstat

    def guard(self, *args, **kwargs):
        assert not str(self).startswith("/Volumes/VIDEO")
        return original(self, *args, **kwargs)

    monkeypatch.setattr(Path, "lstat", guard)
    with pytest.raises(ValueError, match="unsafe_native"):
        read_latest(tmp_path, now=now)


def test_native_reader_round_trip_fixture_reconciles_and_does_not_reenter(
    tmp_path, fixture_data
):
    row, candidate, market, now = deepcopy(fixture_data)
    packet = packet_from(tmp_path, [row], candidate, market, now)
    state = advance_native(None, packet, market, now=now, source_digest="s")
    assert state["phase"] == "pending_buy"
    market["quote"] = dict(
        packet["quote"],
        timestamp_utc=(now + timedelta(seconds=1)).isoformat(),
        snapshot_id="independent-entry",
    )
    state = advance_native(
        state, packet, market, now=now + timedelta(seconds=1), source_digest="s"
    )
    assert state["phase"] == "holding"
    sell = deepcopy(row)
    sell.update(
        action="SELL",
        decision_id="separate-exit",
        timestamp_utc=(now + timedelta(seconds=2)).isoformat(),
    )
    sell["features"].update(
        provider_quote_ts_utc=(now + timedelta(seconds=2)).timestamp(),
        bid_price=33.1,
        ask_price=33.11,
        last_price=33.1,
        provider_quote_last_price=33.1,
    )
    sell["metadata"]["schd_candle_context"]["capture_sha256"] = digest(market)
    sell["metadata"]["snapshot_id"] = "exit-source"
    exit_packet = packet_from(
        tmp_path, [row, sell], candidate, market, now + timedelta(seconds=2)
    )
    state = advance_native(
        state, exit_packet, market, now=now + timedelta(seconds=2), source_digest="s"
    )
    assert state["phase"] == "pending_sell"
    market["quote"] = dict(
        exit_packet["quote"],
        timestamp_utc=(now + timedelta(seconds=3)).isoformat(),
        snapshot_id="independent-exit",
    )
    state = advance_native(
        state, exit_packet, market, now=now + timedelta(seconds=3), source_digest="s"
    )
    assert state["phase"] == "complete"
    assert state["reconciliation"]["cash_delta_equals_modeled_pnl"]
    assert state["virtual_shares"] == 0
    assert not state["live_execution_authority"]
    state = advance_native(
        state, exit_packet, market, now=now + timedelta(seconds=4), source_digest="s"
    )
    assert len(state["orders"]) == 2


def test_newer_hold_cancels_pending_but_candidate_change_cannot_reset(
    tmp_path, fixture_data
):
    row, candidate, market, now = deepcopy(fixture_data)
    packet = packet_from(tmp_path, [row], candidate, market, now)
    state = advance_native(None, packet, market, now=now, source_digest="s")
    hold = dict(
        row,
        action="HOLD",
        decision_id="later-hold",
        timestamp_utc=(now + timedelta(seconds=1)).isoformat(),
    )
    packet = packet_from(
        tmp_path, [row, hold], candidate, market, now + timedelta(seconds=1)
    )
    cancelled = advance_native(
        state, packet, market, now=now + timedelta(seconds=1), source_digest="s"
    )
    assert cancelled["phase"] == "cancelled"
    assert cancelled["orders"] == []
    assert cancelled["last_report"]["bot_record"]["action"] == "HOLD"
    packet["candidate_id"] = "new-candidate"
    with pytest.raises(ValueError, match="review_required"):
        advance_native(state, packet, market, now=now, source_digest="s")


def test_native_cli_cache_no_network_and_read_only_source(
    tmp_path, fixture_data, monkeypatch
):
    row, candidate, market, now = deepcopy(fixture_data)
    row["action"] = "HOLD"
    path = write_rows(tmp_path, [row])
    original = path.read_bytes()
    candidate_path = tmp_path / "governance/runtime/production_candidate_state.json"
    candidate_path.parent.mkdir(parents=True)
    candidate_path.write_text(json.dumps(candidate))
    cache = tmp_path / DIRECTORY / "market_latest.json"
    cache.parent.mkdir(parents=True)
    cache.write_text(json.dumps(market))
    monkeypatch.setattr(
        "scripts.ops.schd_candle_report.fetch_bounded",
        lambda *a, **k: pytest.fail("unexpected_network"),
    )
    monkeypatch.setattr(
        "scripts.ops.schd_candle_report.render_charts", lambda *a, **k: {}
    )
    state = run("native", root=tmp_path, now=now, source_digest="s")
    assert state["native_observation"]["source_selected"]
    assert state["native_observation"]["readiness"] == "WAIT"
    assert path.read_bytes() == original
    status = run("status", root=tmp_path, now=now, source_digest="s")
    assert status["source_data_connected"] and status["native_decision_fresh"]
    assert not run(
        "status", root=tmp_path, now=now + timedelta(minutes=3), source_digest="s"
    )["native_decision_fresh"]


def test_provider_quote_uses_source_time_not_local_clock():
    payload = {
        "SCHD": {
            "symbol": "SCHD",
            "realtime": True,
            "quote": {
                "quoteTime": 1790183825000,
                "lastPrice": 33.1,
                "bidPrice": 33.1,
                "askPrice": 33.11,
                "bidSize": 100,
                "askSize": 100,
            },
        }
    }
    quote = normalize_schwab_quote(payload)
    assert timestamp(quote["timestamp_utc"]).timestamp() == 1790183825
    payload["SCHD"]["realtime"] = False
    with pytest.raises(ValueError, match="realtime"):
        normalize_schwab_quote(payload)


def test_equal_timestamp_different_ids_is_ambiguous(tmp_path, fixture_data):
    row, candidate, market, now = deepcopy(fixture_data)
    packet = packet_from(
        tmp_path,
        [row, dict(row, decision_id="ambiguous", action="HOLD")],
        candidate,
        market,
        now,
    )
    assert (
        "native_ambiguous_same_time_decisions"
        in packet["native_validation"]["blockers"]
    )


def test_failed_refresh_preserves_cache_and_source_time(
    tmp_path, fixture_data, monkeypatch
):
    row, candidate, market, now = deepcopy(fixture_data)
    row["action"] = "HOLD"
    write_rows(tmp_path, [row])
    candidate_path = tmp_path / "governance/runtime/production_candidate_state.json"
    candidate_path.parent.mkdir(parents=True)
    candidate_path.write_text(json.dumps(candidate))
    cache = tmp_path / DIRECTORY / "market_latest.json"
    cache.parent.mkdir(parents=True)
    cache.write_text(json.dumps(market))
    original = cache.read_bytes()

    def failed_capture(*args, **kwargs):
        assert kwargs["include_quote"] is True
        raise ValueError("schwab_provider_cooldown_active")

    monkeypatch.setattr("scripts.ops.schd_candle_report.fetch_bounded", failed_capture)
    monkeypatch.setattr(
        "scripts.ops.schd_candle_report.render_charts", lambda *a, **k: {}
    )
    state = run(
        "native", root=tmp_path, now=now, source_digest="s", refresh_market_data=True
    )
    assert cache.read_bytes() == original
    assert (
        state["native_observation"]["market_refresh"]["reason"]
        == "schwab_provider_cooldown_active"
    )
    assert (
        state["last_report"]["chart_source"]["fetch_started_at_utc"]
        == market["source"]["fetch_started_at_utc"]
    )


def test_missing_source_expires_pending_and_reports_disconnected(
    tmp_path, fixture_data, monkeypatch
):
    row, candidate, market, now = deepcopy(fixture_data)
    packet = packet_from(tmp_path, [row], candidate, market, now)
    state = advance_native(None, packet, market, now=now, source_digest="s")
    state_path = tmp_path / DIRECTORY / "recorded_state.json"
    state_path.parent.mkdir(parents=True)
    state_path.write_text(json.dumps(state))
    write_rows(tmp_path, [])
    monkeypatch.setattr(
        "scripts.ops.schd_candle_report.render_charts", lambda *a, **k: {}
    )
    state = run(
        "native", root=tmp_path, now=now + timedelta(seconds=61), source_digest="s"
    )
    assert state["phase"] == "cancelled"
    assert state["pending"] is None
    status = run(
        "status", root=tmp_path, now=now + timedelta(seconds=61), source_digest="s"
    )
    assert not status["source_data_connected"]
    assert not status["native_decision_fresh"]
