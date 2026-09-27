import argparse
import json
from pathlib import Path
import pytest

from scripts.ops import maintenance_slot_guard as src
from scripts.ops import one_numbers_refresh_policy as risk_policy


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def test_smooth_mode_gate_blocks_nonessential_slot_under_protect_pressure(tmp_path: Path) -> None:
    runtime_path = tmp_path / "governance" / "health" / "runtime_throttle_control_latest.json"
    _write_json(
        runtime_path,
        {
            "host_saturation_score": 84.14,
            "compute_pressure_level": "high",
            "memory_pressure_level": "high",
            "mac_fluidity_contract": {
                "overall_status": "needs_work",
                "fluidity_band": "protect",
                "support_pause_recommended": True,
            },
        },
    )

    blocked, reason, snapshot = src._smooth_mode_blocked(
        "daily_auto_verify",
        max_saturation_score=68.0,
        exempt_slots=set(),
        runtime_path=runtime_path,
    )

    assert blocked is True
    assert reason == "runtime_smooth_gate:fluidity_band=protect"
    assert snapshot["host_saturation_score"] == 84.14
    assert snapshot["policy"] == "defer_nonessential_maintenance_when_runtime_smooth_mode_is_strained"


def test_smooth_mode_gate_exempts_backlog_plumbing_slots(tmp_path: Path) -> None:
    runtime_path = tmp_path / "governance" / "health" / "runtime_throttle_control_latest.json"
    _write_json(
        runtime_path,
        {
            "host_saturation_score": 99.0,
            "mac_fluidity_contract": {"overall_status": "needs_work", "fluidity_band": "protect"},
        },
    )

    blocked, reason, snapshot = src._smooth_mode_blocked(
        "storage_backpressure_autopilot",
        max_saturation_score=68.0,
        exempt_slots=src.DEFAULT_SMOOTH_GATE_EXEMPT_SLOTS,
        runtime_path=runtime_path,
    )

    assert blocked is False
    assert reason == "smooth_gate_exempt"
    assert snapshot["exempt"] is True


def test_dead_owner_maintenance_lock_is_reaped_without_waiting_for_stale_timeout(
    tmp_path: Path, monkeypatch
) -> None:
    lock_path = tmp_path / "maintenance_bundle.lock"
    lock_path.mkdir()
    _write_json(lock_path / "owner.json", {"pid": 424242})
    monkeypatch.setattr(src, "_pid_is_running", lambda pid: False)

    assert src._reap_abandoned_lock(lock_path, stale_seconds=1800.0) is True
    assert not lock_path.exists()


def test_new_lock_without_owner_gets_initialization_grace(tmp_path: Path) -> None:
    lock_path = tmp_path / "maintenance_bundle.lock"
    lock_path.mkdir()

    assert src._reap_abandoned_lock(
        lock_path,
        stale_seconds=1800.0,
        owner_grace_seconds=60.0,
    ) is False
    assert lock_path.exists()


def test_sql_writer_slot_bypasses_host_pressure_and_cooldown(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(src, "RUNTIME_ROOT", tmp_path / "runtime" / "maintenance_slots")
    monkeypatch.setattr(src, "LOCK_ROOT", tmp_path / "runtime" / "maintenance_slots" / "locks")
    monkeypatch.setattr(src, "STATE_ROOT", tmp_path / "runtime" / "maintenance_slots" / "state")
    monkeypatch.setattr(src, "HEALTH_PATH", tmp_path / "governance" / "health" / "maintenance_slot_guard_latest.json")
    monkeypatch.setattr(src, "EXTERNAL_HEALTH_PATH", tmp_path / "external" / "maintenance_slot_guard_latest.json")
    monkeypatch.setattr(src, "_host_pressure", lambda *args, **kwargs: (True, {"load_ratios": {"one_minute": 9.9}}))
    monkeypatch.setattr(src, "_cooldown_blocked", lambda *args, **kwargs: (True, "slot_cooldown_age_seconds=1<900", {}))
    monkeypatch.setattr(src, "_load_macro_status", lambda: {})
    monkeypatch.setattr(src, "_process_running", lambda needles: False)
    monkeypatch.setattr(src, "maintenance_hold_snapshot", lambda _project_root: {"active": False})

    args = argparse.Namespace(
        slot="sql_link_writer",
        max_load_ratio=0.1,
        max_five_min_load_ratio=0.1,
        max_one_min_load=0.0,
        min_interval_seconds=None,
        stale_seconds=1800.0,
        protect_macro_before_minutes=180.0,
        protect_macro_after_minutes=75.0,
        allow_during_macro_event=False,
        defer_while_sql_link_active=True,
        quiet_windows_enabled=False,
        defer_outside_quiet_window=False,
        quiet_start_hour=21,
        quiet_end_hour=6,
        smooth_gate_enabled=False,
        smooth_gate_max_saturation_score=68.0,
        smooth_gate_exempt_slots="",
        skip_exit_code=75,
        json=True,
    )

    assert src._begin(args) == 0
    assert (tmp_path / "runtime" / "maintenance_slots" / "locks" / "sql_link_writer.lock").exists()

    end_args = argparse.Namespace(slot="sql_link_writer", json=True)
    assert src._end(end_args) == 0


def test_runtime_maintenance_hold_blocks_even_sql_writer_slot(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(src, "RUNTIME_ROOT", tmp_path / "runtime" / "maintenance_slots")
    monkeypatch.setattr(src, "LOCK_ROOT", tmp_path / "runtime" / "maintenance_slots" / "locks")
    monkeypatch.setattr(src, "STATE_ROOT", tmp_path / "runtime" / "maintenance_slots" / "state")
    monkeypatch.setattr(src, "HEALTH_PATH", tmp_path / "governance" / "health" / "maintenance_slot_guard_latest.json")
    monkeypatch.setattr(src, "EXTERNAL_HEALTH_PATH", tmp_path / "external" / "maintenance_slot_guard_latest.json")
    monkeypatch.setattr(src, "_host_pressure", lambda *args, **kwargs: (False, {}))
    monkeypatch.setattr(src, "_cooldown_blocked", lambda *args, **kwargs: (False, "", {}))
    monkeypatch.setattr(src, "_load_macro_status", lambda: {})
    monkeypatch.setattr(src, "_process_running", lambda needles: False)
    monkeypatch.setattr(
        src,
        "maintenance_hold_snapshot",
        lambda _project_root: {"active": True, "reason": "sqlite_local_failover"},
    )
    args = argparse.Namespace(
        slot="sql_link_writer",
        max_load_ratio=0.85,
        max_five_min_load_ratio=0.7,
        max_one_min_load=0.0,
        min_interval_seconds=None,
        stale_seconds=1800.0,
        protect_macro_before_minutes=180.0,
        protect_macro_after_minutes=75.0,
        allow_during_macro_event=False,
        defer_while_sql_link_active=True,
        quiet_windows_enabled=False,
        defer_outside_quiet_window=False,
        quiet_start_hour=21,
        quiet_end_hour=6,
        smooth_gate_enabled=False,
        smooth_gate_max_saturation_score=68.0,
        smooth_gate_exempt_slots="",
        skip_exit_code=75,
        json=True,
    )

    assert src._begin(args) == 75
    payload = json.loads(src.HEALTH_PATH.read_text(encoding="utf-8"))
    assert "runtime_maintenance_hold" in payload["reasons"]
    assert not (src.LOCK_ROOT / "sql_link_writer.lock").exists()


@pytest.mark.parametrize("slot,admitted,hold,expected", [
    ("one_numbers_refresh", True, False, 0),
    ("one_numbers_refresh", False, False, 75),
    ("one_numbers_refresh", True, True, 75),
    ("daily_auto_verify", True, False, 75),
])
@pytest.mark.parametrize("smooth_reason", ["runtime_smooth_gate:fluidity_band=protect", "runtime_smooth_gate:fluidity_band=strained", "runtime_smooth_gate:support_pause_recommended"])
def test_bounded_risk_launch_admission_is_slot_specific_and_keeps_holds(tmp_path, monkeypatch, slot, admitted, hold, expected, smooth_reason):
    for name, path in {
        "PROJECT_ROOT": tmp_path, "RUNTIME_ROOT": tmp_path / "runtime",
        "LOCK_ROOT": tmp_path / "runtime/locks", "STATE_ROOT": tmp_path / "runtime/state",
        "HEALTH_PATH": tmp_path / "health.json", "EXTERNAL_HEALTH_PATH": tmp_path / "external.json",
        "RUNTIME_THROTTLE_HEALTH_PATH": tmp_path / "throttle.json",
    }.items():
        monkeypatch.setattr(src, name, path)
    monkeypatch.setenv("ONE_NUMBERS_BOUNDED_RISK_REFRESH", "1")
    monkeypatch.setattr(risk_policy, "bounded_refresh_admitted", lambda *args: admitted)
    monkeypatch.setattr(src, "_host_pressure", lambda *args: (True, {}))
    monkeypatch.setattr(src, "_cooldown_blocked", lambda *args: (False, "", {}))
    monkeypatch.setattr(src, "_load_macro_status", lambda: {})
    monkeypatch.setattr(src, "_process_running", lambda *args: False)
    monkeypatch.setattr(src, "maintenance_hold_snapshot", lambda *args: {"active": hold})
    monkeypatch.setattr(src, "_smooth_mode_blocked", lambda *args, **kwargs: (
        True, smooth_reason, {"support_pause_recommended": smooth_reason.endswith("support_pause_recommended")}
    ))
    if smooth_reason != "runtime_smooth_gate:fluidity_band=protect":
        expected = 75
    args = argparse.Namespace(
        slot=slot, max_load_ratio=0.72, max_five_min_load_ratio=0.62, max_one_min_load=0,
        min_interval_seconds=None, stale_seconds=1800, protect_macro_before_minutes=180,
        protect_macro_after_minutes=75, allow_during_macro_event=False, defer_while_sql_link_active=True,
        quiet_windows_enabled=False, defer_outside_quiet_window=False, quiet_start_hour=21, quiet_end_hour=6,
        smooth_gate_enabled=True, smooth_gate_max_saturation_score=68, smooth_gate_exempt_slots="", skip_exit_code=75, json=True,
    )
    assert src._begin(args) == expected
    if expected == 0:
        assert json.loads(src.HEALTH_PATH.read_text())["bounded_overdue_risk_refresh"]
        assert src._end(argparse.Namespace(slot=slot, json=True)) == 0
