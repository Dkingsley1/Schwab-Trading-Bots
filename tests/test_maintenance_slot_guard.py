import argparse
import json
from pathlib import Path

from scripts.ops import maintenance_slot_guard as src


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
