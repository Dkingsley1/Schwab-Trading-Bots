import inspect

import pytest

from scripts.ops import process_watchdog as pw


@pytest.mark.parametrize("flag", ["1", "true", "yes", "on", "TRUE"])
def test_storage_owner_pause_defers_missing_writer_restart(monkeypatch, flag):
    monkeypatch.setenv("SQL_LINK_SERVICE_PAUSED_FOR_LOCAL_STORAGE", flag)
    hold = pw._sql_writer_restart_hold({})
    assert hold["active"] is True
    assert hold["reason"] == "local_storage_reserve_pressure"
    assert hold["writer_ready"] is False


def test_maintenance_hold_applies_to_writer_without_global_halt_pause(monkeypatch):
    monkeypatch.delenv("SQL_LINK_SERVICE_PAUSED_FOR_LOCAL_STORAGE", raising=False)
    hold = pw._sql_writer_restart_hold({"runtime_maintenance_hold_active": True})
    assert hold["reason"] == "runtime_maintenance_hold"
    assert pw._sql_writer_restart_hold({"global_halt_active": True})["active"] is False


def test_cleared_owner_flag_allows_normal_writer_recovery(monkeypatch):
    monkeypatch.setenv("SQL_LINK_SERVICE_PAUSED_FOR_LOCAL_STORAGE", "0")
    assert pw._sql_writer_restart_hold({})["active"] is False


def test_held_writer_is_not_healthy_and_restart_history_is_not_forgiven():
    row = {
        "name": "sql_link_writer",
        "process_live": False,
        "running": 0,
        "heartbeat_ok": False,
        "restart_skipped": "writer_admission_hold",
        "reason": "local_storage_reserve_pressure",
        "writer_admission_hold": {"active": True},
    }
    events = [
        {"name": "sql_link_writer", "event": "restart", "ts_epoch": 990.0}
        for _ in range(6)
    ]
    kwargs = dict(
        events=events,
        restart_window_seconds=3600,
        restart_storm_threshold=3,
        settle_seconds=120,
        now_epoch=1000.0,
    )
    active, recent = pw._resolved_restart_storms(status_rows=[row], **kwargs)
    assert active == []
    assert recent[0]["resolved"] is False
    assert recent[0]["deferred_by_writer_admission"] is True
    retained, receipt = pw._forgive_resolved_restart_debt(events, recent)
    assert retained == events
    assert receipt["active"] is False
    assert pw._watchdog_need_for_row(row)["status"] == "intentional_hold"
    assert pw._row_effective_process_live(row) is False
    row.pop("restart_skipped")
    active, _ = pw._resolved_restart_storms(status_rows=[row], **kwargs)
    assert len(active) == 1


def test_writer_admission_precedes_idle_recovery_and_restart_budget():
    source = inspect.getsource(pw.main)
    assert source.index("writer_hold = _sql_writer_restart_hold") < source.index(
        "writer_idle_health = _sql_link_writer_idle_health"
    )
    assert source.index(
        'row["restart_skipped"] = "writer_admission_hold"'
    ) < source.index("if not _within_budget")
