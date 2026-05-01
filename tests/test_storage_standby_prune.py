import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path("/Users/dankingsley/PycharmProjects/schwab_trading_bot")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


storage_standby_prune = _load_module(
    "storage_standby_prune_test",
    ROOT / "scripts" / "ops" / "storage_standby_prune.py",
)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _failback_payload(project_root: Path, *, verification_state: str = "curated_ready", active_local_count: int = 0) -> dict:
    local_root = project_root / "local_fallback_storage" / "data"
    return {
        "mode": "external",
        "certified_mode": "external_curated",
        "active_root": str(project_root / "external"),
        "sqlite_skip_report": {
            "summary": {
                "active_local_count": active_local_count,
            },
            "route_verification": {
                "verification_state": verification_state,
                "mismatches": [],
            },
            "entries": [
                {
                    "relative_path": "data/bot_channel_queue.sqlite3",
                    "classification": "warm_standby_retained",
                    "route_verification": {
                        "state": "verified",
                    },
                    "local": {
                        "path": str(local_root / "bot_channel_queue.sqlite3"),
                        "exists": True,
                        "sidecars": ["bot_channel_queue.sqlite3-wal"],
                    },
                },
                {
                    "relative_path": "data/jsonl_link.sqlite3",
                    "classification": "warm_standby_retained",
                    "route_verification": {
                        "state": "curated_standby",
                    },
                    "local": {
                        "path": str(local_root / "jsonl_link.sqlite3"),
                        "exists": True,
                        "sidecars": [],
                    },
                },
                {
                    "relative_path": "data/snapshot_context.sqlite3",
                    "classification": "warm_standby_retained",
                    "route_verification": {
                        "state": "verified",
                    },
                    "local": {
                        "path": str(local_root / "snapshot_context.sqlite3"),
                        "exists": True,
                        "sidecars": [],
                    },
                },
            ],
        },
    }


def test_build_payload_dry_run_selects_only_verified_standby(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    local_data = project_root / "local_fallback_storage" / "data"
    local_data.mkdir(parents=True, exist_ok=True)
    (local_data / "bot_channel_queue.sqlite3").write_text("queue", encoding="utf-8")
    (local_data / "bot_channel_queue.sqlite3-wal").write_text("wal", encoding="utf-8")
    (local_data / "jsonl_link.sqlite3").write_text("primary", encoding="utf-8")
    (local_data / "snapshot_context.sqlite3").write_text("snapshot", encoding="utf-8")
    _write_json(
        project_root / "governance" / "health" / "storage_switch_orchestrator_latest.json",
        {"timestamp_utc": "2026-04-22T18:00:00+00:00", "overall_status": "switched", "target_mode": "external"},
    )

    payload = storage_standby_prune.build_payload(
        project_root=project_root,
        apply=False,
        min_route_soak_hours=2.0,
        failback_payload=_failback_payload(project_root),
    )

    assert payload["overall_status"] == "dry_run"
    assert payload["summary"]["eligible_count"] == 2
    by_rel = {row["relative_path"]: row for row in payload["candidates"]}
    assert by_rel["data/bot_channel_queue.sqlite3"]["eligible"] is True
    assert by_rel["data/snapshot_context.sqlite3"]["eligible"] is True
    assert by_rel["data/jsonl_link.sqlite3"]["eligible"] is False


def test_build_payload_blocks_until_route_soak_window_passes(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    local_data = project_root / "local_fallback_storage" / "data"
    local_data.mkdir(parents=True, exist_ok=True)
    (local_data / "bot_channel_queue.sqlite3").write_text("queue", encoding="utf-8")
    _write_json(
        project_root / "governance" / "health" / "storage_switch_orchestrator_latest.json",
        {"timestamp_utc": "3026-04-22T20:00:00+00:00", "overall_status": "switched", "target_mode": "external"},
    )

    payload = storage_standby_prune.build_payload(
        project_root=project_root,
        apply=False,
        min_route_soak_hours=2.0,
        failback_payload=_failback_payload(project_root),
    )

    assert payload["overall_status"] == "deferred_route_soak"
    assert payload["summary"]["eligible_count"] == 0
    assert payload["route_soak"]["ok"] is False


def test_build_payload_apply_prunes_verified_standby_only(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    local_data = project_root / "local_fallback_storage" / "data"
    local_data.mkdir(parents=True, exist_ok=True)
    (local_data / "bot_channel_queue.sqlite3").write_text("queue", encoding="utf-8")
    (local_data / "bot_channel_queue.sqlite3-wal").write_text("wal", encoding="utf-8")
    (local_data / "jsonl_link.sqlite3").write_text("primary", encoding="utf-8")
    (local_data / "snapshot_context.sqlite3").write_text("snapshot", encoding="utf-8")
    _write_json(
        project_root / "governance" / "health" / "storage_switch_orchestrator_latest.json",
        {"timestamp_utc": "2026-04-22T18:00:00+00:00", "overall_status": "switched", "target_mode": "external"},
    )

    payload = storage_standby_prune.build_payload(
        project_root=project_root,
        apply=True,
        min_route_soak_hours=2.0,
        failback_payload=_failback_payload(project_root),
    )

    assert payload["overall_status"] == "pruned"
    assert payload["summary"]["deleted_paths_count"] == 3
    assert not (local_data / "bot_channel_queue.sqlite3").exists()
    assert not (local_data / "bot_channel_queue.sqlite3-wal").exists()
    assert not (local_data / "snapshot_context.sqlite3").exists()
    assert (local_data / "jsonl_link.sqlite3").exists()
