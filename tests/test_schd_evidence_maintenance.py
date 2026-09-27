from copy import deepcopy
from datetime import timedelta
import fcntl
import json
import os
from pathlib import Path

import pytest

from core.decision_price_evidence import digest, timestamp
from core.schd_capture_store import (
    DIRECTORY,
    decision_capture,
    latest_capture,
    preserve_capture,
    prune_captures,
    publish_capture,
)
from core.schd_market_evidence import candle_context_receipt
from scripts.ops.schd_evidence_maintenance import (
    maintain,
    maintenance_metric,
    refresh_due,
)
from scripts.ops.schd_native_decision import native_packet
from tests.test_schd_native_decision import fixture_data, write_rows


def capture_at(market, now):
    market = deepcopy(market)
    market["source"]["fetch_started_at_utc"] = now.isoformat()
    market["source"][
        "price_adjustment_basis"
    ] = "provider_as_returned_not_independently_verified"
    market["candles"]["1m"] = [
        dict(
            market["candles"]["5m"][-1],
            start_utc=(now.replace(second=0) - timedelta(minutes=1)).isoformat(),
            end_utc=now.replace(second=0).isoformat(),
        )
    ]
    return market


def configure(root):
    path = root / "config/supervised_schd_broker_test_v1.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{}")


def test_new_capture_preserves_original_decision_context(tmp_path, fixture_data):
    row, candidate, market, now = deepcopy(fixture_data)
    old = tmp_path / DIRECTORY / "market_latest.json"
    old.parent.mkdir(parents=True)
    old.write_text(json.dumps(market))
    newer = capture_at(market, now + timedelta(seconds=1))
    publish_capture(tmp_path, newer)
    original = decision_capture(tmp_path, row)
    assert digest(original) == row["metadata"]["schd_candle_context"]["capture_sha256"]
    assert latest_capture(tmp_path) == newer
    packet = native_packet(row, {}, {"issues": []}, original, candidate, now=now)
    assert (
        "native_original_candle_context_binding_missing_or_changed"
        not in packet["native_validation"]["blockers"]
    )
    assert row["action"] == fixture_data[0]["action"]


def test_missing_original_never_substitutes_new_market(tmp_path, fixture_data):
    row, _, market, now = deepcopy(fixture_data)
    publish_capture(tmp_path, capture_at(market, now + timedelta(seconds=1)))
    assert decision_capture(tmp_path, row) == {}
    row["metadata"]["schd_candle_context"]["capture_sha256"] = "../../outside"
    assert decision_capture(tmp_path, row) == latest_capture(tmp_path)


def test_capture_corruption_is_not_overwritten(tmp_path, fixture_data):
    _, _, market, _ = deepcopy(fixture_data)
    identity = preserve_capture(tmp_path, market)
    path = tmp_path / DIRECTORY / "captures" / f"{identity}.json"
    path.write_text("{}")
    with pytest.raises(ValueError, match="corruption"):
        preserve_capture(tmp_path, market)
    assert path.read_text() == "{}"


def test_retention_only_expired_verified_owned_captures(tmp_path, fixture_data):
    _, _, market, now = deepcopy(fixture_data)
    expired = capture_at(market, now - timedelta(minutes=31))
    old_hash = preserve_capture(tmp_path, expired)
    current = capture_at(market, now)
    current_hash = publish_capture(tmp_path, current)
    directory = tmp_path / DIRECTORY / "captures"
    unknown = directory / "operator-note.json"
    unknown.write_text("{}")
    corrupt = directory / ("f" * 64 + ".json")
    corrupt.write_text("{}")
    history = tmp_path / DIRECTORY / "recorded_state.json"
    history.write_text('{"audit":"preserve"}')
    report = prune_captures(tmp_path, now=now)
    assert report["deleted"] == [f"{old_hash}.json"]
    assert (directory / f"{current_hash}.json").exists()
    assert unknown.exists() and corrupt.exists() and history.exists()
    assert latest_capture(tmp_path) == current


def test_future_recent_hardlinked_and_latest_are_preserved(tmp_path, fixture_data):
    _, _, market, now = deepcopy(fixture_data)
    for delta in (1, -29, -32):
        value = capture_at(market, now + timedelta(minutes=delta))
        identity = preserve_capture(tmp_path, value)
        if delta == -32:
            os.link(
                tmp_path / DIRECTORY / "captures" / f"{identity}.json",
                tmp_path / "retain",
            )
    publish_capture(tmp_path, capture_at(market, now - timedelta(hours=2)))
    assert prune_captures(tmp_path, now=now)["deleted"] == []


def test_abandoned_complete_build_cleanup_respects_recent_writes(
    tmp_path, fixture_data
):
    _, _, market, now = deepcopy(fixture_data)
    directory = tmp_path / DIRECTORY / "captures"
    directory.mkdir(parents=True)
    old = directory / ".capture-abcdefgh"
    recent = directory / ".capture-12345678"
    for path in (old, recent):
        path.write_text(json.dumps(capture_at(market, now - timedelta(hours=2))))
    os.utime(old, (now.timestamp() - 3600,) * 2)
    os.utime(recent, (now.timestamp(),) * 2)
    assert prune_captures(tmp_path, now=now)["deleted"] == [old.name]
    assert recent.exists()


def test_cleanup_and_publication_budgets(tmp_path, fixture_data, monkeypatch):
    _, _, market, now = deepcopy(fixture_data)
    for index in range(10):
        preserve_capture(
            tmp_path, capture_at(market, now - timedelta(hours=1, seconds=index))
        )
    assert len(prune_captures(tmp_path, now=now)["deleted"]) == 8
    monkeypatch.setattr("core.schd_capture_store.MAX_TOTAL_BYTES", 1)
    with pytest.raises(ValueError, match="budget"):
        publish_capture(tmp_path, capture_at(market, now))
    assert latest_capture(tmp_path) == {}


def test_protected_capture_alias_never_followed(tmp_path, fixture_data, monkeypatch):
    row, _, _, _ = deepcopy(fixture_data)
    directory = tmp_path / DIRECTORY
    directory.mkdir(parents=True)
    (directory / "captures").symlink_to("/Volumes/VIDEO/do-not-touch")
    original = Path.lstat

    def guarded(path, *args, **kwargs):
        assert not str(path).startswith("/Volumes/VIDEO")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "lstat", guarded)
    with pytest.raises(ValueError, match="unsafe"):
        decision_capture(tmp_path, row)


def test_native_owner_refreshes_without_changing_decision_or_price_basis(
    tmp_path, fixture_data
):
    row, _, market, now = deepcopy(fixture_data)
    row["action"] = "HOLD"
    path = write_rows(tmp_path, [row])
    original = path.read_bytes()
    captured = capture_at(market, now)

    def fetch(root, **kwargs):
        assert kwargs == {"timeout_seconds": 25}
        return captured

    report = maintain(tmp_path, now=now, fetcher=fetch)
    assert report["market_refresh"]["state"] == "refreshed"
    assert report["current_closed_candles_available"]
    assert report["native_decision"]["action"] == "HOLD"
    assert report["native_decision"]["fresh"]
    assert (
        not report["live_execution_authority"]
        and not report["broker_mutation_attempted"]
    )
    assert report["market_source"]["price_adjustment_basis"].endswith(
        "not_independently_verified"
    )
    assert path.read_bytes() == original
    receipt = candle_context_receipt(tmp_path, "SCHD", now=now)
    assert receipt["capture_sha256"] == digest(captured)
    maintain(tmp_path, now=now, fetcher=lambda *a, **k: pytest.fail("redundant GETs"))


def test_failed_fetch_keeps_original_bytes_and_reports_stale_decision(
    tmp_path, fixture_data
):
    row, _, market, now = deepcopy(fixture_data)
    write_rows(tmp_path, [row])
    publish_capture(tmp_path, market)
    latest = tmp_path / DIRECTORY / "market_latest.json"
    before = latest.read_bytes()

    def failed(*args, **kwargs):
        raise ValueError("schwab_provider_cooldown_active")

    result = maintain(tmp_path, now=now + timedelta(minutes=10), fetcher=failed)
    assert not result["ok"] and not result["native_decision"]["fresh"]
    assert result["market_refresh"]["reason"] == "schwab_provider_cooldown_active"
    assert latest.read_bytes() == before


def test_off_and_writer_busy_do_not_fetch_or_replace_receipt(tmp_path, fixture_data):
    now = fixture_data[-1]
    directory = tmp_path / DIRECTORY
    directory.mkdir(parents=True)
    receipt = directory / "evidence_maintenance_latest.json"
    receipt.write_text("{}")
    with (directory / "writer.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        result = maintain(
            tmp_path, now=now, fetcher=lambda *a, **k: pytest.fail("fetch")
        )
        assert "writer_busy" in result["state"]
    off = tmp_path / "governance/health/SYSTEM_POWER_OFF.flag"
    off.parent.mkdir(parents=True)
    off.touch()
    assert maintain(tmp_path, now=now)["state"] == "system_power_off"
    assert receipt.read_text() == "{}"


def test_native_cadence_regular_session_and_hourly_retention(tmp_path, fixture_data):
    now = fixture_data[-1]
    assert not maintenance_metric(tmp_path, now=now)["refresh_due"]
    configure(tmp_path)
    assert maintenance_metric(tmp_path, now=now)["refresh_due"]
    receipt = tmp_path / DIRECTORY / "evidence_maintenance_latest.json"
    receipt.parent.mkdir(parents=True)
    receipt.write_text(json.dumps({"timestamp_utc": now.isoformat()}))
    assert not maintenance_metric(tmp_path, now=now + timedelta(seconds=59))[
        "refresh_due"
    ]
    assert maintenance_metric(tmp_path, now=now + timedelta(seconds=60))["refresh_due"]
    night = timestamp("2026-09-23T23:00:00+00:00")
    receipt.write_text(json.dumps({"timestamp_utc": night.isoformat()}))
    assert not maintenance_metric(tmp_path, now=night + timedelta(minutes=59))[
        "refresh_due"
    ]
    assert maintenance_metric(tmp_path, now=night + timedelta(hours=1))["refresh_due"]
    result = maintain(
        tmp_path, now=night, fetcher=lambda *a, **k: pytest.fail("after-hours fetch")
    )
    assert result["market_refresh"]["state"] == "outside_regular_session"


def test_fresh_wrapper_cannot_make_missing_closed_candles_current(fixture_data):
    _, _, market, now = deepcopy(fixture_data)
    market = capture_at(market, now)
    assert not refresh_due(market, now=now)
    later = now + timedelta(minutes=5)
    market["source"]["fetch_started_at_utc"] = later.isoformat()
    assert refresh_due(market, now=later)


def test_existing_scheduler_prioritizes_risk_then_read_only_context():
    from scripts.ops.adaptive_ops_recovery_policy import (
        build_control_plane_refresh_plan,
    )

    plan = build_control_plane_refresh_plan(
        {
            "control_plane_freshness": {
                "risk_service_boundary": {"refresh_due": True},
                "schd_evidence_maintenance": {"refresh_due": True},
            }
        }
    )
    assert [step["id"] for step in plan[:2]] == [
        "risk_service_boundary",
        "schd_evidence_maintenance",
    ]
    assert plan[1]["command"] == [
        "./scripts/ops/opsctl.sh",
        "schd-decision-rehearsal",
        "maintain",
        "--json",
    ]
    assert plan[1]["timeout_seconds"] == 45


def test_invalid_owned_receipt_is_refreshable_but_protected_route_is_not(
    tmp_path, fixture_data
):
    now = fixture_data[-1]
    configure(tmp_path)
    path = tmp_path / DIRECTORY / "evidence_maintenance_latest.json"
    path.parent.mkdir(parents=True)
    path.write_text("incomplete json")
    assert maintenance_metric(tmp_path, now=now)["refresh_due"]
    path.unlink()
    path.symlink_to("/Volumes/VIDEO/do-not-read")
    assert not maintenance_metric(tmp_path, now=now)["refresh_due"]


def test_context_must_have_been_fresh_at_original_decision(fixture_data):
    row, candidate, market, now = deepcopy(fixture_data)
    market["source"]["fetch_started_at_utc"] = (now - timedelta(minutes=6)).isoformat()
    row["metadata"]["schd_candle_context"]["capture_sha256"] = digest(market)
    packet = native_packet(row, {}, {"issues": []}, market, candidate, now=now)
    assert (
        "native_original_candle_context_binding_missing_or_changed"
        in packet["native_validation"]["blockers"]
    )


def test_chart_explanation_preserves_recorded_hold_and_separate_attribution(
    tmp_path, fixture_data, monkeypatch
):
    from scripts.ops.schd_decision_rehearsal import run

    row, _, market, now = deepcopy(fixture_data)
    row["action"] = "HOLD"
    row["reasons"] = ["defensive_buy_requires_risk_off"]
    write_rows(tmp_path, [row])
    monkeypatch.setattr(
        "scripts.ops.schd_candle_report.fetch_bounded", lambda *a, **k: market
    )
    monkeypatch.setattr(
        "scripts.ops.schd_candle_report.render_charts", lambda *a, **k: {}
    )
    state = run("charts", root=tmp_path, now=now, source_digest="test")
    report = state["last_report"]
    sample = report["recorded_decision_sample"]
    assert sample["record"]["action"] == "HOLD"
    assert sample["record"]["reasons"] == row["reasons"]
    assert (
        sample["chart_binding"]
        == "separate_current_context_not_original_decision_input"
    )
    assert report["bot_record"]["action"] == "WAIT"
    assert state["orders"] == [] and not state["live_execution_authority"]
    markdown = Path(state["report_path"]).read_text()
    assert "Recorded action: **HOLD**" in markdown
    assert "not proof the bot used these candles" in markdown
