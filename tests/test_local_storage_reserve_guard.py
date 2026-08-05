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
    assert report["bytes_reclaimed"] > 0
    assert os.stat(log_path).st_ino == inode_before
    assert log_path.stat().st_size <= 256
    assert log_path.read_bytes().endswith(b"important-tail")


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
