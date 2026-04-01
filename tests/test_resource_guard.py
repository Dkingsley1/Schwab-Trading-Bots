from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import resource_guard


def test_memory_pressure_state_turns_yellow_on_low_available(monkeypatch) -> None:
    monkeypatch.setenv("RESOURCE_GUARD_MEMORY_YELLOW_AVAILABLE_PCT", "50")
    snapshot = {
        "memory_available_pct": 42.0,
        "memory_free_pct": 12.0,
        "swap_used_gb": 6.0,
        "pages_throttled": 0,
    }
    state, reasons, _thresholds = resource_guard._memory_pressure_state(snapshot)
    assert state == "yellow"
    assert any("available_pct" in reason for reason in reasons)


def test_memory_pressure_state_turns_red_on_throttled_pages(monkeypatch) -> None:
    monkeypatch.setenv("RESOURCE_GUARD_MEMORY_RED_THROTTLED_PAGES", "1")
    snapshot = {
        "memory_available_pct": 60.0,
        "memory_free_pct": 20.0,
        "swap_used_gb": 2.0,
        "pages_throttled": 3,
    }
    state, reasons, _thresholds = resource_guard._memory_pressure_state(snapshot)
    assert state == "red"
    assert any("pages_throttled" in reason for reason in reasons)


def test_optional_job_blocks_on_yellow_pressure(monkeypatch) -> None:
    monkeypatch.setenv("RESOURCE_GUARD_OPTIONAL_BLOCK_ON_MEMORY_STATES", "yellow,red")
    snapshot = {
        "memory_available_pct": 45.0,
        "memory_free_pct": 9.0,
        "swap_used_gb": 13.0,
        "pages_throttled": 0,
        "load1_per_core": 0.4,
        "disk_free_gb": 120.0,
        "editing_app_cpu_sum": 0.0,
    }
    ok, reasons, details = resource_guard.evaluate_optional_job(snapshot)
    assert ok is False
    assert details["memory_pressure_state"] == "yellow"
    assert any(reason.startswith("memory_pressure_yellow") for reason in reasons)


def test_optional_job_allows_green_pressure(monkeypatch) -> None:
    monkeypatch.setenv("RESOURCE_GUARD_OPTIONAL_BLOCK_ON_MEMORY_STATES", "yellow,red")
    snapshot = {
        "memory_available_pct": 68.0,
        "memory_free_pct": 18.0,
        "swap_used_gb": 4.0,
        "pages_throttled": 0,
        "load1_per_core": 0.6,
        "disk_free_gb": 120.0,
        "editing_app_cpu_sum": 20.0,
    }
    ok, reasons, details = resource_guard.evaluate_optional_job(snapshot)
    assert ok is True
    assert details["memory_pressure_state"] == "green"
    assert reasons == []


def test_refresh_job_allows_swap_only_pressure_with_healthy_headroom(monkeypatch) -> None:
    monkeypatch.setenv("RESOURCE_GUARD_OPTIONAL_BLOCK_ON_MEMORY_STATES", "yellow,red")
    snapshot = {
        "memory_available_pct": 58.0,
        "memory_free_pct": 18.0,
        "swap_used_gb": 23.5,
        "pages_throttled": 0,
        "load1_per_core": 0.6,
        "disk_free_gb": 120.0,
        "editing_app_cpu_sum": 10.0,
    }

    ok, reasons, details = resource_guard.evaluate_refresh_job(snapshot)

    assert ok is True
    assert reasons == []
    assert details["memory_pressure_state"] == "red"
    assert details["memory_pressure_kind"] == "swap_only_with_headroom"
    assert details["refresh_relax_applied"] is True
    assert details["refresh_relax_reason"] == "swap_only_pressure_with_healthy_headroom"


def test_refresh_job_still_blocks_true_memory_pressure(monkeypatch) -> None:
    monkeypatch.setenv("RESOURCE_GUARD_OPTIONAL_BLOCK_ON_MEMORY_STATES", "yellow,red")
    snapshot = {
        "memory_available_pct": 28.0,
        "memory_free_pct": 3.0,
        "swap_used_gb": 23.5,
        "pages_throttled": 2,
        "load1_per_core": 0.6,
        "disk_free_gb": 120.0,
        "editing_app_cpu_sum": 10.0,
    }

    ok, reasons, details = resource_guard.evaluate_refresh_job(snapshot)

    assert ok is False
    assert details["memory_pressure_kind"] == "throttled"
    assert details["refresh_relax_applied"] is False
    assert any(reason.startswith("memory_pressure_red") for reason in reasons)
