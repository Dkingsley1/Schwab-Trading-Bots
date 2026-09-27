from datetime import datetime, timezone
import fcntl
import json
import os

import pytest

from scripts.ops import bitcoin_price_watch as watch
from scripts.ops import readiness_evidence_refresh as refresh

NOW = datetime(2026, 9, 23, 14, tzinfo=timezone.utc)


def candles(profile, *, now):
    spec = watch.PROFILES[profile]
    step = spec["granularity"]
    end = int(now.timestamp()) // step * step
    return [[end - (spec["bars"] - i) * step, 990+i, 1010+i, 1000+i, 1005+i, 5]
            for i in range(spec["bars"])]


def test_three_observers_share_two_windows_and_never_allocate():
    calls = []
    def fetch(name, **kwargs):
        calls.append(name)
        return candles(name, **kwargs)
    result = watch.build(now=NOW, fetcher=fetch)
    assert calls == ["day", "swing"]
    assert result["ok"] and len(result["bots"]) == 3
    assert result["cost_and_capital"]["capital_allocated_usd"] == 0
    assert not result["live_execution_authority"]
    assert not result["paper_execution_authority"]
    assert all(not row["order_requested"] for row in result["bots"])
    assert result["profiles"]["day"]["source_timestamp_utc"] == NOW.isoformat()
    assert result["profiles"]["swing"]["source_timestamp_utc"] == "2026-09-23T12:00:00+00:00"


@pytest.mark.parametrize("fault", ["stale", "gap", "duplicate", "nan"])
def test_bad_window_is_not_replaced_with_old_or_invented_signal(fault):
    def fetch(name, **kwargs):
        rows = candles(name, **kwargs)
        if name == "day":
            if fault == "stale":
                rows = [[r[0]-300, *r[1:]] for r in rows]
            elif fault == "gap":
                rows.pop(20)
            elif fault == "duplicate":
                rows[21] = rows[20]
            else:
                rows[20][4] = float("nan")
        return rows
    result = watch.build(now=NOW, fetcher=fetch)
    assert not result["ok"]
    assert [x["observation"] for x in result["bots"][:2]] == ["unavailable"] * 2
    assert result["profiles"]["swing"]["state"] == "observed"


def test_power_off_does_not_fetch_or_rewrite(tmp_path, monkeypatch):
    path = tmp_path / "governance/health/SYSTEM_POWER_OFF.flag"
    path.parent.mkdir(parents=True)
    path.touch()
    monkeypatch.setattr(watch, "build", lambda: pytest.fail("must not fetch"))
    assert watch.run(tmp_path)["reason"] == "system_power_off"
    assert not (tmp_path / watch.OUT).exists()


def test_singleton_preserves_previous_evidence(tmp_path, monkeypatch):
    lock = tmp_path / watch.LOCK
    lock.parent.mkdir(parents=True)
    monkeypatch.setattr(watch, "build", lambda: pytest.fail("must not fetch"))
    with lock.open("w") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert watch.run(tmp_path)["reason"] == "observer_busy"
    assert not (tmp_path / watch.OUT).exists()


def test_existing_cadence_owns_bounded_observer():
    assert "bitcoin_price_watch" in refresh.PROFILE_STEP_NAMES["accrual"]
    row = next(x for x in refresh.default_steps() if x["name"] == "bitcoin_price_watch")
    assert row["owner_timeout_seconds"] == 30 and row["max_age_minutes"] == 15


def test_fifo_lock_is_rejected_without_fetch(tmp_path, monkeypatch):
    lock = tmp_path / watch.LOCK
    lock.parent.mkdir(parents=True)
    os.mkfifo(lock)
    monkeypatch.setattr(watch, "build", lambda: pytest.fail("must not fetch"))
    with pytest.raises(ValueError, match="observer_lock_requires_regular_file"):
        watch.run(tmp_path)
    assert not (tmp_path / watch.OUT).exists()


@pytest.mark.parametrize("accepted", [True, False])
def test_native_decision_history_reports_writer_acceptance(tmp_path, monkeypatch, accepted):
    from core import accountability
    build = watch.build
    monkeypatch.setattr(watch, "build", lambda **kwargs: build(now=NOW, fetcher=candles))
    rows = []
    def append(path, row, **kwargs):
        rows.append(row)
        return accepted
    monkeypatch.setattr(accountability, "safe_append_channel_event", append)
    result = watch.run(tmp_path)
    assert len(rows) == 3
    assert all(r["action"] == "HOLD" and r["decision"] == "OBSERVE_ONLY" for r in rows)
    assert all(r["metadata"]["indicator_reasoning"]["rules"] for r in rows)
    assert result["decision_history"]["accepted_by_native_writer"] == (3 if accepted else 0)
    assert result["decision_history"]["durability_certified"] is False
    assert result["ok"] is accepted
