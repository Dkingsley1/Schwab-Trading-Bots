import fcntl
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.local_storage_reserve import GIB, local_storage_reserve_contract
from scripts.ops import local_storage_reserve_guard as guard


def _usage(*, free_gb: float, total_gb: float = 100.0) -> SimpleNamespace:
    free = int(free_gb * GIB)
    total = int(total_gb * GIB)
    return SimpleNamespace(total=total, free=free, used=total - free)


def test_live_reserve_contract_blocks_before_enospc(tmp_path: Path) -> None:
    payload = local_storage_reserve_contract(
        tmp_path,
        disk_usage_fn=lambda _path: _usage(free_gb=7.5),
    )

    assert payload["status"] == "emergency"
    assert payload["hard_block"] is True
    assert payload["pressure_active"] is True
    assert payload["control_env"]["TRAINING_RUNTIME_PAUSED_FOR_BACKLOG"] == "1"
    assert payload["control_env"]["BOT_COLLECTION_DUTY_CYCLE_MAX_ACTIVE_RATIO"] == "0.05"
    assert payload["control_env"]["BOT_STORAGE_EMERGENCY_DISK_GUARD"] == "1"
    assert payload["control_env"]["SQL_LINK_SERVICE_PAUSED_FOR_LOCAL_STORAGE"] == "1"
    assert payload["control_env"]["PAPER_EXECUTION_RUNTIME_PAUSED_FOR_LOCAL_STORAGE"] == "1"


def test_live_reserve_contract_healthy_state_drops_pressure_only_keys(tmp_path: Path) -> None:
    payload = local_storage_reserve_contract(
        tmp_path,
        disk_usage_fn=lambda _path: _usage(free_gb=80.0),
    )

    assert payload["status"] == "ready"
    assert payload["grade"] == "A+"
    assert payload["ready"] is True
    assert "TRAINING_RUNTIME_PAUSED_FOR_BACKLOG" not in payload["control_env"]
    assert "HEAVY_COLLECTORS_PAUSED_FOR_BACKLOG" not in payload["control_env"]


def test_launchd_log_guard_caps_same_inode_and_preserves_tail(tmp_path: Path) -> None:
    log_path = tmp_path / "ops_runtime_smooth_mode.out.log"
    payload = (b"x" * 8192) + b"important-tail"
    log_path.write_bytes(payload)
    inode_before = os.stat(log_path).st_ino

    report = guard.cap_launchd_logs(tmp_path, max_bytes=1024, tail_bytes=256, apply=True)

    assert report["capped_count"] == 1
    assert report["verification_failed_count"] == 0
    assert report["rows"][0]["inode_preserved"] is True
    assert report["rows"][0]["within_limit"] is True
    assert report["rows"][0]["tail_content_preserved"] is True
    assert report["bytes_reclaimed"] > 0
    assert os.stat(log_path).st_ino == inode_before
    assert log_path.stat().st_size <= 256
    assert log_path.read_bytes().endswith(b"important-tail")


def test_launchd_log_guard_recurses_into_service_log_directories(tmp_path: Path) -> None:
    nested = tmp_path / "launchd_watchdog"
    nested.mkdir()
    log_path = nested / "shadow_watchdog.out.log"
    log_path.write_bytes((b"old-output\n" * 1024) + b"newest-watchdog-state\n")

    report = guard.cap_launchd_logs(tmp_path, max_bytes=1024, tail_bytes=256, apply=True)

    assert report["file_count"] == 1
    assert report["capped_count"] == 1
    assert log_path.stat().st_size <= 256
    assert log_path.read_bytes().endswith(b"newest-watchdog-state\n")


def test_launchd_log_guard_aggregates_multiple_roots(tmp_path: Path) -> None:
    first = tmp_path / "tmp_logs"
    second = tmp_path / "library_logs"
    first.mkdir()
    second.mkdir()
    (first / "first.out.log").write_bytes(b"a" * 4096)
    (second / "second.err.log").write_bytes(b"b" * 4096)

    report = guard.cap_launchd_log_roots(
        [first, second, first],
        max_bytes=1024,
        tail_bytes=128,
        apply=True,
    )

    assert report["log_roots"] == [str(first), str(second)]
    assert report["file_count"] == 2
    assert report["capped_count"] == 2
    assert report["error_count"] == 0


def test_healthy_override_replaces_stale_pause_without_reintroducing_it(tmp_path: Path) -> None:
    override = tmp_path / ".env.local_storage_reserve_override"
    override.write_text("TRAINING_RUNTIME_PAUSED_FOR_BACKLOG=1\n", encoding="utf-8")
    contract = local_storage_reserve_contract(
        tmp_path,
        disk_usage_fn=lambda _path: _usage(free_gb=80.0),
    )

    changed = guard._write_override(override, contract["control_env"])
    content = override.read_text(encoding="utf-8")

    assert changed is True
    assert "BOT_LOCAL_STORAGE_RESERVE_STATE=ready" in content
    assert "TRAINING_RUNTIME_PAUSED_FOR_BACKLOG" not in content


def test_ingestion_disk_contract_treats_local_hard_reserve_as_emergency(
    tmp_path: Path, monkeypatch
) -> None:
    from scripts.ops import ingestion_storage_control as ingestion

    external = tmp_path / "BOT_LOGS"
    external.mkdir()
    monkeypatch.setattr(
        ingestion,
        "local_storage_reserve_contract",
        lambda _root: {
            "free_gb": 4.0,
            "target_free_gb": 64.0,
            "pressure_active": True,
            "hard_block": True,
        },
    )

    payload = ingestion._storage_plane_disk_contract(
        project_root=tmp_path,
        data_collection_storage_guard={"external_root": str(external)},
        raw_training_compaction={"scan_roots": [{"path": str(external)}]},
    )

    assert payload["local_emergency_disk_guard"] is True
    assert payload["emergency_disk_guard"] is True
    assert payload["local_reserve"]["free_gb"] == 4.0


def test_active_telemetry_route_rejects_quarantine_targets(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    target_root = Path("/Volumes/BOT_LOGS/schwab_trading_bot/quarantine/recovery")
    for relative in guard.TELEMETRY_ROUTE_PATHS:
        path = project_root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.symlink_to(target_root / relative.replace("/", "_"))

    payload = guard.telemetry_route_contract(project_root)

    assert payload["ready"] is False
    assert all(row["quarantine_backed"] is True for row in payload["rows"])


def test_proactive_recovery_preserves_paper_above_pressure_floor(tmp_path: Path) -> None:
    reserve = local_storage_reserve_contract(
        tmp_path,
        target_free_gb=125.0,
        pressure_free_gb=64.0,
        hard_free_gb=32.0,
        emergency_free_gb=16.0,
        disk_usage_fn=lambda _path: _usage(free_gb=120.0, total_gb=200.0),
    )

    request = guard._recovery_request(reserve, tmp_path)

    assert request["active"] is True
    assert request["severity"] == "proactive"
    assert request["paper_pause_required"] is False
    assert request["collection_may_continue"] is True
    assert request["warning_target_free_gb"] == 125.0
    assert request["recovery_target_free_gb"] == 135.0
    assert request["command"][-3:-1] == ["--storage-target-free-gb", "135.0"]


def test_pressure_recovery_reports_paper_pause_and_specific_reason(tmp_path: Path) -> None:
    reserve = local_storage_reserve_contract(
        tmp_path,
        target_free_gb=125.0,
        pressure_free_gb=64.0,
        hard_free_gb=32.0,
        emergency_free_gb=16.0,
        disk_usage_fn=lambda _path: _usage(free_gb=60.0, total_gb=200.0),
    )

    request = guard._recovery_request(reserve, tmp_path)

    assert request["active"] is True
    assert request["severity"] == "pressure"
    assert request["reason"] == "local_hot_storage_pressure_reserve_breached"
    assert request["paper_pause_required"] is True
    assert request["collection_may_continue"] is False


def test_history_is_event_driven_and_bounded(tmp_path: Path) -> None:
    payload = {
        "timestamp_utc": "2026-08-10T00:00:00+00:00",
        "overall_status": "ready",
        "grade": "A+",
        "hard_blockers": [],
        "warnings": [],
        "local_storage_reserve": {
            "status": "ready",
            "free_gb": 130.0,
            "target_free_gb": 125.0,
            "pressure_free_gb": 64.0,
            "hard_free_gb": 32.0,
            "emergency_free_gb": 16.0,
        },
        "cleanup_verification": {"verified": True, "file_bytes_reclaimed": 0},
        "recovery_request": {"active": False, "severity": "none", "paper_pause_required": False},
        "launchd_log_guard": {"bytes_reclaimed": 0, "capped_count": 0},
        "override_changed": False,
    }

    assert guard._history_event_required({}, payload) is True
    assert guard._history_event_required(dict(payload), payload) is False

    free_space_drift = json.loads(json.dumps(payload))
    free_space_drift["local_storage_reserve"]["free_gb"] = 129.5
    free_space_drift["override_changed"] = True
    assert guard._history_event_required(payload, free_space_drift) is False

    threshold_change = json.loads(json.dumps(payload))
    threshold_change["local_storage_reserve"]["target_free_gb"] = 126.0
    assert guard._history_event_required(payload, threshold_change) is True

    history = tmp_path / "guard_history.jsonl"
    for index in range(4):
        payload["timestamp_utc"] = f"2026-08-10T00:00:0{index}+00:00"
        result = guard._append_history(history, payload, max_lines=2)
        assert result["ok"] is True

    assert len(history.read_text(encoding="utf-8").splitlines()) == 2


def test_installer_publishes_unattended_reserve_ladder() -> None:
    installer = (PROJECT_ROOT / "scripts" / "install_local_storage_reserve_guard_launchd.sh").read_text(
        encoding="utf-8"
    )

    assert 'BOT_LOCAL_STORAGE_TARGET_FREE_GB:-125' in installer
    assert 'BOT_LOCAL_STORAGE_PRESSURE_FREE_GB:-64' in installer
    assert 'BOT_LOCAL_STORAGE_HARD_FREE_GB:-32' in installer
    assert 'BOT_LOCAL_STORAGE_EMERGENCY_FREE_GB:-16' in installer


def test_guard_defaults_do_not_fall_back_to_legacy_core_thresholds() -> None:
    assert guard.DEFAULT_TARGET_FREE_GB == 125.0
    assert guard.DEFAULT_PRESSURE_FREE_GB == 64.0
    assert guard.DEFAULT_HARD_FREE_GB == 32.0
    assert guard.DEFAULT_EMERGENCY_FREE_GB == 16.0
    assert guard.DEFAULT_RECOVERY_HEADROOM_GB == 10.0


def test_guard_skips_overlapping_invocation_without_replacing_latest(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    lock_path = tmp_path / "guard.lock"
    out_path = tmp_path / "latest.json"
    with lock_path.open("a+", encoding="utf-8") as held_lock:
        fcntl.flock(held_lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "local_storage_reserve_guard.py",
                "--project-root",
                str(tmp_path),
                "--lock-file",
                str(lock_path),
                "--out-file",
                str(out_path),
                "--json",
            ],
        )

        assert guard.main() == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["busy"] is True
    assert payload["reason"] == "non_overlapping_guard_lock_held"
    assert out_path.exists() is False
