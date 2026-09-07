import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.ops import backlog_pcore_accelerator as src


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def _seed_host_lane(project_root: Path) -> None:
    health = project_root / "governance" / "health"
    _write_json(
        health / "autonomic_resource_governor_latest.json",
        {
            "overall_status": "ready",
            "host_lane_budget": {
                "primary_compute_lanes": 6,
                "selected_p_core_preprocess_workers": 5,
                "p_core_allocation_contract": {"user_app_reserved_p_cores": 1},
                "p_core_widening_controller": {
                    "memory_pressure_controller": {
                        "status": "clear",
                        "max_memory_safe_workers": 5,
                    }
                },
            },
        },
    )
    _write_json(
        health / "memory_pressure_intelligence_latest.json",
        {
            "overall_status": "ready",
            "classification": {
                "status": "clear",
                "recommended_p_core_worker_cap": 5,
            },
            "observer_overhead": {"active": False},
            "reopen_gate": {"safe_to_widen_p_core_workers": True},
        },
    )


def test_pcore_storage_maintenance_allows_parallel_file_compaction_only(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.delenv("BACKLOG_PCORE_STORAGE_COMPACTION_WORKERS", raising=False)
    monkeypatch.delenv("BOT_RAW_TRAINING_COMPACTION_WORKERS", raising=False)
    project_root = tmp_path / "project"
    health = project_root / "governance" / "health"
    _seed_host_lane(project_root)
    _write_json(
        health / "writer_process_intelligence_latest.json",
        {
            "overall_status": "ready",
            "process_topology": {
                "sql_link_writer_running_count": 1,
                "raw_sql_link_writer_running_count": 1,
                "duplicate_sql_writer_processes": False,
            },
        },
    )
    _write_json(
        health / "data_collection_storage_guard_latest.json",
        {
            "overall_status": "ready",
            "guard_mode": "normal",
            "disk": {"available_gb": 152.0},
            "safe_space_recovery": {"candidate_gb": 6.3},
        },
    )
    _write_json(
        health / "raw_training_compaction_intelligence_latest.json",
        {
            "overall_status": "ready",
            "raw_summary": {
                "compression_candidate_count": 3,
                "compression_candidate_gb": 2.5,
            },
        },
    )

    payload = src.build_payload(project_root)
    contract = payload["storage_maintenance_pcore_contract"]
    raw_lane = next(
        row
        for row in contract["lanes"]
        if row["lane"] == "raw_training_file_compaction"
    )
    checkpoint_lane = next(
        row
        for row in contract["lanes"]
        if row["lane"] == "sqlite_checkpoint_and_pressure_clearance"
    )

    assert contract["status"] == "ready"
    assert contract["file_compaction_workers"] == 4
    assert raw_lane["uses_p_core"] is True
    assert raw_lane["writes_files"] is True
    assert raw_lane["writes_sqlite"] is False
    assert raw_lane["parallel_safe"] is True
    assert "--compaction-workers" in raw_lane["command"]
    assert (
        raw_lane["command"][raw_lane["command"].index("--compaction-workers") + 1]
        == "4"
    )
    assert checkpoint_lane["writes_sqlite"] is True
    assert checkpoint_lane["parallel_safe"] is False
    assert contract["parallel_sqlite_writes_allowed"] is False
    assert (
        payload["integration_contract"]["p_core_accelerators_preprocess_only"] is False
    )
    assert (
        payload["integration_contract"]["p_core_sqlite_accelerators_preprocess_only"]
        is True
    )
    assert payload["integration_contract"]["active_storage_mode"] == "unknown"
    assert payload["integration_contract"]["storage_route_rehome_required"] is False
    env_lines = src._env_lines(payload)
    assert "BOT_RAW_TRAINING_COMPACTION_WORKERS=4" in env_lines
    assert "BACKLOG_PCORE_STORAGE_SQLITE_PARALLELISM=1" in env_lines


def test_pcore_storage_maintenance_blocks_checkpoint_on_duplicate_writer(
    tmp_path: Path,
) -> None:
    project_root = tmp_path / "project"
    health = project_root / "governance" / "health"
    _seed_host_lane(project_root)
    _write_json(
        health / "writer_process_intelligence_latest.json",
        {
            "overall_status": "degraded",
            "process_topology": {
                "sql_link_writer_running_count": 2,
                "raw_sql_link_writer_running_count": 2,
                "duplicate_sql_writer_processes": True,
            },
        },
    )
    _write_json(
        health / "data_collection_storage_guard_latest.json",
        {
            "overall_status": "ready",
            "guard_mode": "normal",
            "disk": {"available_gb": 152.0},
        },
    )

    contract = src.build_payload(project_root)["storage_maintenance_pcore_contract"]

    assert contract["status"] == "blocked"
    assert contract["sqlite_checkpoint_ready"] is False
    assert "duplicate_sqlite_writer_blocks_checkpoint" in contract["blockers"]
    assert contract["sqlite_write_parallelism"] == 1


def test_pcore_storage_maintenance_surfaces_local_fallback_route_pressure(
    tmp_path: Path,
) -> None:
    project_root = tmp_path / "project"
    health = project_root / "governance" / "health"
    _seed_host_lane(project_root)
    fallback_shards = (
        project_root / "local_fallback_storage" / "data" / "sql_link_shards"
    )
    fallback_shards.mkdir(parents=True)
    (fallback_shards / "jsonl_link_trading.sqlite3").write_bytes(b"x" * 4096)
    _write_json(
        health / "writer_process_intelligence_latest.json",
        {
            "overall_status": "ready",
            "process_topology": {
                "sql_link_writer_running_count": 1,
                "raw_sql_link_writer_running_count": 1,
                "duplicate_sql_writer_processes": False,
            },
        },
    )
    _write_json(
        health / "data_collection_storage_guard_latest.json",
        {
            "overall_status": "ready",
            "guard_mode": "normal",
            "disk": {"available_gb": 152.0},
        },
    )
    _write_json(
        health / "local_storage_reserve_guard_latest.json",
        {
            "overall_status": "degraded",
            "local_storage_reserve": {
                "free_gb": 54.0,
                "pressure_active": True,
                "hard_block": False,
                "reserve_deficit_gb": 71.0,
            },
        },
    )
    _write_json(
        health / "storage_failback_sync_latest.json",
        {
            "overall_status": "ready",
            "mode": "local_fallback",
            "certified_mode": "local_fallback",
            "active_root": str(project_root / "local_fallback_storage"),
            "sqlite_skip_report": {
                "summary": {
                    "active_local_count": 3,
                    "active_external_count": 0,
                    "warm_standby_count": 0,
                    "local_bytes_total": 4096,
                },
                "route_verification": {
                    "verification_state": "active_local_ready",
                    "mismatches": [],
                },
            },
        },
    )

    payload = src.build_payload(project_root)
    contract = payload["storage_maintenance_pcore_contract"]
    failback_lane = next(
        row
        for row in contract["lanes"]
        if row["lane"] == "storage_route_failback_reconciliation"
    )
    prune_lane = next(
        row
        for row in contract["lanes"]
        if row["lane"] == "verified_local_standby_prune"
    )

    assert contract["status"] == "limited"
    assert contract["file_compaction_ready"] is True
    assert contract["sqlite_checkpoint_ready"] is True
    assert contract["local_pressure_active"] is True
    assert (
        "active_local_fallback_route_needs_verified_external_failback_before_standby_prune"
        in contract["route_limiters"]
    )
    assert contract["active_storage_route"]["mode"] == "local_fallback"
    assert contract["active_storage_route"]["route_is_local_fallback"] is True
    assert contract["active_storage_route"]["route_is_external_prune_ready"] is False
    assert (
        contract["active_storage_route"]["local_fallback_sql_link_shards"]["size_bytes"]
        == 4096
    )
    assert (
        contract["active_storage_route"]["local_fallback_sql_link_shards"]["size_kind"]
        == "complete"
    )
    assert failback_lane["parallel_safe"] is False
    assert failback_lane["writes_sqlite"] is True
    assert prune_lane["parallel_safe"] is False
    assert prune_lane["writes_sqlite"] is False
    assert payload["integration_contract"]["active_storage_mode"] == "local_fallback"
    assert payload["integration_contract"]["storage_route_rehome_required"] is True
    env_lines = src._env_lines(payload)
    assert "BACKLOG_PCORE_STORAGE_ROUTE_REHOME_REQUIRED=1" in env_lines
