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


def test_default_guard_uses_runtime_disk_but_keeps_local_floor(monkeypatch) -> None:
    monkeypatch.setenv("RESOURCE_GUARD_MIN_LOCAL_DISK_GB", "2")
    snapshot = {
        "memory_available_pct": 68.0,
        "memory_free_pct": 18.0,
        "load1_per_core": 0.6,
        "disk_free_gb": 705.0,
        "local_disk_free_gb": 4.5,
        "editing_app_cpu_sum": 20.0,
    }

    ok, reasons = resource_guard.evaluate(
        snapshot,
        max_load_per_core=1.8,
        min_disk_gb=20.0,
        min_memory_free_pct=10.0,
        max_editing_cpu=180.0,
    )

    assert ok is True
    assert reasons == []


def test_default_guard_blocks_when_local_floor_is_too_low(monkeypatch) -> None:
    monkeypatch.setenv("RESOURCE_GUARD_MIN_LOCAL_DISK_GB", "2")
    snapshot = {
        "memory_available_pct": 68.0,
        "memory_free_pct": 18.0,
        "load1_per_core": 0.6,
        "disk_free_gb": 705.0,
        "local_disk_free_gb": 1.0,
        "editing_app_cpu_sum": 20.0,
    }

    ok, reasons = resource_guard.evaluate(
        snapshot,
        max_load_per_core=1.8,
        min_disk_gb=20.0,
        min_memory_free_pct=10.0,
        max_editing_cpu=180.0,
    )

    assert ok is False
    assert reasons == ["local_disk_free_low:1.0<2.0"]


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


def test_optional_job_blocks_on_active_creative_session(monkeypatch) -> None:
    monkeypatch.setenv("RESOURCE_GUARD_OPTIONAL_BLOCK_ON_MEMORY_STATES", "yellow,red")
    snapshot = {
        "memory_available_pct": 72.0,
        "memory_free_pct": 22.0,
        "swap_used_gb": 3.0,
        "pages_throttled": 0,
        "load1_per_core": 0.5,
        "disk_free_gb": 120.0,
        "editing_app_cpu_sum": 34.0,
        "creative_apps_active": True,
        "creative_app_count": 1,
        "creative_apps": ["Final Cut Pro"],
        "creative_session_level": "active",
    }

    ok, reasons, details = resource_guard.evaluate_optional_job(snapshot)

    assert ok is False
    assert "creative_session_active" in reasons
    assert "active" in details["optional_job_thresholds"]["block_on_creative_session_levels"]


def test_refresh_job_allows_active_creative_session_when_system_has_headroom(monkeypatch) -> None:
    monkeypatch.setenv("RESOURCE_GUARD_OPTIONAL_BLOCK_ON_MEMORY_STATES", "yellow,red")
    snapshot = {
        "memory_available_pct": 71.0,
        "memory_free_pct": 20.0,
        "swap_used_gb": 3.5,
        "pages_throttled": 0,
        "load1_per_core": 0.5,
        "disk_free_gb": 120.0,
        "editing_app_cpu_sum": 28.0,
        "creative_apps_active": True,
        "creative_app_count": 1,
        "creative_apps": ["Logic Pro"],
        "creative_session_level": "active",
    }

    ok, reasons, details = resource_guard.evaluate_refresh_job(snapshot)

    assert ok is True
    assert reasons == []
    assert details["refresh_creative_override_applied"] is True
    assert details["refresh_creative_override_reason"] == "creative_session_active_allowed"


def test_music_app_counts_as_audio_playback_cotenant(monkeypatch) -> None:
    monkeypatch.setenv("RESOURCE_GUARD_CREATIVE_APP_NAMES", "Final Cut Pro,Logic Pro,Music,iTunes")
    monkeypatch.setattr(
        resource_guard,
        "_scan_named_processes",
        lambda _markers: (
            {"Music": 4.2},
            {"Music": "/System/Applications/Music.app/Contents/MacOS/Music"},
        ),
    )

    snapshot = resource_guard._creative_apps_snapshot()

    assert snapshot["creative_apps_active"] is True
    assert snapshot["creative_apps"] == ["Music"]
    assert snapshot["creative_session_level"] == "active"
    assert snapshot["creative_session_kind"] == "music_playback"
    assert snapshot["music_playback_cpu"] == 4.2


def test_named_process_scan_ignores_helper_and_path_false_positives(monkeypatch) -> None:
    class Result:
        stdout = "\n".join(
            [
                "0.0 /System/Library/PrivateFrameworks/iTunesCloud.framework/Support/itunescloudd",
                "58.2 ./Codex Computer Use.app/Contents/SharedSupport/SkyComputerUseClient.app/Contents/MacOS/SkyComputerUseClient mcp",
                "192.1 /opt/homebrew/bin/python /Users/dankingsley/PycharmProjects/schwab_trading_bot/scripts/failover_hot_standby.py",
                "0.0 /Applications/Codex.app/Contents/Frameworks/Electron Framework.framework/Helpers/chrome_crashpad_handler",
                "0.0 /System/Applications/Safari.app/Contents/Extensions/SafariWidgetExtension.appex/Contents/MacOS/SafariWidgetExtension",
                "0.0 /System/Library/PrivateFrameworks/TextInputUIMacHelper.framework/Versions/A/XPCServices/CursorUIViewService.xpc/Contents/MacOS/CursorUIViewService",
                "0.0 /System/Library/Frameworks/InputMethodKit.framework/Resources/imklaunchagent",
                "4.2 /System/Applications/Music.app/Contents/MacOS/Music",
                "12.5 /Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
                "9.0 /Applications/PyCharm.app/Contents/MacOS/pycharm",
            ]
        )

    monkeypatch.setattr(resource_guard.subprocess, "run", lambda *_args, **_kwargs: Result())

    cpu_by_app, commands = resource_guard._scan_named_processes(
        ["Music", "iTunes", "Code", "PyCharm", "Chrome", "Safari", "Cursor", "UTM"]
    )

    assert cpu_by_app == {"Music": 4.2, "Chrome": 12.5, "PyCharm": 9.0}
    assert "itunescloudd" not in str(commands)
    assert "Codex Computer Use" not in str(commands)
    assert "SafariWidgetExtension" not in str(commands)


def test_refresh_job_blocks_dual_creative_session(monkeypatch) -> None:
    monkeypatch.setenv("RESOURCE_GUARD_OPTIONAL_BLOCK_ON_MEMORY_STATES", "yellow,red")
    snapshot = {
        "memory_available_pct": 69.0,
        "memory_free_pct": 19.0,
        "swap_used_gb": 4.0,
        "pages_throttled": 0,
        "load1_per_core": 0.4,
        "disk_free_gb": 120.0,
        "editing_app_cpu_sum": 42.0,
        "creative_apps_active": True,
        "creative_app_count": 2,
        "creative_apps": ["Final Cut Pro", "Logic Pro"],
        "creative_session_level": "dual_pro",
    }

    ok, reasons, details = resource_guard.evaluate_refresh_job(snapshot)

    assert ok is False
    assert "creative_session_dual_pro" in reasons
    assert "dual_pro" in details["refresh_job_thresholds"]["block_on_creative_session_levels"]
