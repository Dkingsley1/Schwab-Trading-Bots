from __future__ import annotations

import json
from pathlib import Path

from scripts.ops import local_sql_shard_standby_prune as src


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _seed_ready_route(project_root: Path, external_root: Path) -> None:
    health = project_root / "governance" / "health"
    _write_json(
        health / "stateful_storage_regression_guard_latest.json",
        {
            "stateful_target_mode": "external",
            "checks": [
                {
                    "name": "sql_link_shards",
                    "status": "ready",
                    "target_match": True,
                }
            ],
        },
    )
    _write_json(
        health / "storage_failback_sync_latest.json",
        {
            "certified_mode": "external_curated",
            "route_verification": {
                "verification_state": "ready",
                "mismatches": [],
            },
        },
    )
    active = external_root / "data" / "sql_link_shards"
    active.mkdir(parents=True)
    data_root = project_root / "data"
    data_root.mkdir(parents=True)
    (data_root / "sql_link_shards").symlink_to(active, target_is_directory=True)


def test_local_sql_shard_standby_prune_deletes_only_mirrored_inactive_cache(
    tmp_path: Path, monkeypatch
) -> None:
    project_root = tmp_path / "project"
    external_root = tmp_path / "BOT_LOGS" / "schwab_trading_bot"
    _seed_ready_route(project_root, external_root)
    monkeypatch.setenv("LOCAL_SQL_SHARD_STANDBY_PRUNE_CHECK_OPEN_HANDLES", "0")

    local = project_root / "local_fallback_storage" / "data" / "sql_link_shards"
    local.mkdir(parents=True)
    mirrored = local / "jsonl_link_trading.sqlite3"
    sidecar = local / "jsonl_link_trading.sqlite3-wal"
    unmirrored = local / "jsonl_link_runtime.sqlite3"
    raw_evidence = local / "trade_decisions_20260909.jsonl"
    mirrored.write_bytes(b"x" * 11)
    sidecar.write_bytes(b"w" * 7)
    unmirrored.write_bytes(b"u" * 5)
    raw_evidence.write_text("{}\n", encoding="utf-8")
    (external_root / "data" / "sql_link_shards" / "jsonl_link_trading.sqlite3").write_bytes(
        b"external"
    )

    payload = src.build_payload(project_root, external_root=str(external_root), apply=True)

    assert payload["overall_status"] == "applied"
    assert payload["deleted_count"] == 2
    assert not mirrored.exists()
    assert not sidecar.exists()
    assert unmirrored.exists()
    assert raw_evidence.exists()


def test_local_sql_shard_standby_prune_blocks_when_active_route_is_local(
    tmp_path: Path, monkeypatch
) -> None:
    project_root = tmp_path / "project"
    external_root = tmp_path / "BOT_LOGS" / "schwab_trading_bot"
    monkeypatch.setenv("LOCAL_SQL_SHARD_STANDBY_PRUNE_CHECK_OPEN_HANDLES", "0")
    local = project_root / "local_fallback_storage" / "data" / "sql_link_shards"
    local.mkdir(parents=True)
    data_root = project_root / "data"
    data_root.mkdir(parents=True)
    (data_root / "sql_link_shards").symlink_to(local, target_is_directory=True)
    _write_json(
        project_root / "governance" / "health" / "stateful_storage_regression_guard_latest.json",
        {
            "stateful_target_mode": "local_fallback",
            "checks": [
                {
                    "name": "sql_link_shards",
                    "status": "ready",
                    "target_match": True,
                }
            ],
        },
    )
    _write_json(
        project_root / "governance" / "health" / "storage_failback_sync_latest.json",
        {
            "certified_mode": "local_fallback",
            "route_verification": {"verification_state": "active_local_ready", "mismatches": []},
        },
    )
    (local / "jsonl_link_trading.sqlite3").write_bytes(b"x" * 11)

    payload = src.build_payload(project_root, external_root=str(external_root), apply=True)

    assert payload["overall_status"] == "blocked"
    assert "active_route_is_local_fallback" in payload["blockers"]
    assert (local / "jsonl_link_trading.sqlite3").exists()
