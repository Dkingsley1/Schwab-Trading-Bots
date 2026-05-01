import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.ops import ingestion_storage_governor as src


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def test_ingestion_storage_governor_marks_route_drift_and_targets_routed_paths(tmp_path: Path) -> None:
    health = tmp_path / "governance" / "health"
    _write_json(health / "ingestion_storage_control_latest.json", {"overall_status": "needs_work"})
    _write_json(
        health / "ingestion_backpressure_latest.json",
        {"pending_lines": 0, "pending_lines_deferred": 362543, "pending_lines_cold": 362543},
    )
    _write_json(health / "ingestion_priority_queue_latest.json", {"lane_counts": {}})
    _write_json(health / "storage_mount_guard_latest.json", {"external_available": True, "storage_mode": "external"})
    _write_json(health / "storage_failback_sync_latest.json", {"mode": "external", "split_brain_conflicts": 0})
    _write_json(health / "storage_split_brain_reconciler_latest.json", {"summary": {"unresolved_conflicts": 0}})
    _write_json(
        health / "sql_link_service_latest.json",
        {"primary_db": str(tmp_path / "local_fallback_storage" / "data" / "jsonl_link.sqlite3")},
    )
    _write_json(health / "sql_link_service_progress_latest.json", {})
    _write_json(health / "health_gates_latest.json", {"hard_gate_triggered": False, "storage_pressure": {"retention_debt_gb": 0.0}})

    payload = src.build_payload(tmp_path, action="status")

    assert payload["profile"] == "critical_backpressure"
    assert payload["sql_primary_db"]["route_drift"] is True
    assert payload["env_overrides"]["SQL_LINK_SERVICE_PRIMARY_DB"] == str(tmp_path / "data" / "jsonl_link.sqlite3")
    assert payload["env_overrides"]["BOT_CHANNEL_QUEUE_DB"] == str(tmp_path / "data" / "bot_channel_queue.sqlite3")
    assert payload["env_overrides"]["SQL_LINK_SERVICE_WAL_CHECKPOINT_THRESHOLD_GB"] == "0.5"
    assert payload["env_overrides"]["LOG_GATE_EVALUATIONS"] == "0"
    assert payload["env_overrides"]["LOG_SUB_BOT_DECISIONS"] == "0"
    assert payload["env_overrides"]["LOG_DECISION_EXPLANATIONS"] == "0"
    assert payload["env_overrides"]["CHANNEL_LOG_PRIMARY_MODE"] == "channel"
    assert payload["env_overrides"]["LEGACY_HOT_CHANNEL_MIRROR_ENABLED"] == "0"
    assert payload["throttle_controls"]["deferred_files_budget"] == 0
    assert payload["throttle_controls"]["log_gate_evaluations"] == "0"
    assert payload["throttle_controls"]["log_shadow_pnl_attribution"] == "0"
    assert payload["queue_watermarks"]["overall_status"] == "blocked"
    assert payload["writer_shedding"]["level"] == "protect_core"
    assert payload["writer_shedding"]["freeze_cold_lanes"] is True


def test_ingestion_storage_governor_allows_small_deferred_trickle_when_core_is_low(tmp_path: Path) -> None:
    health = tmp_path / "governance" / "health"
    _write_json(
        health / "ingestion_storage_control_latest.json",
        {"overall_status": "blocked", "storage": {"retention_debt_gb": 8.4}},
    )
    _write_json(
        health / "ingestion_backpressure_latest.json",
        {"pending_lines": 4200, "pending_lines_deferred": 98543, "pending_lines_cold": 0},
    )
    _write_json(health / "ingestion_priority_queue_latest.json", {"lane_counts": {}})
    _write_json(health / "storage_mount_guard_latest.json", {"external_available": True, "storage_mode": "external"})
    _write_json(health / "storage_failback_sync_latest.json", {"mode": "external", "split_brain_conflicts": 0})
    _write_json(health / "storage_split_brain_reconciler_latest.json", {"summary": {"unresolved_conflicts": 0}})
    _write_json(
        health / "sql_link_service_latest.json",
        {"primary_db": str(tmp_path / "data" / "jsonl_link.sqlite3")},
    )
    _write_json(health / "sql_link_service_progress_latest.json", {})
    _write_json(health / "health_gates_latest.json", {"hard_gate_triggered": True, "storage_pressure": {"retention_debt_gb": 8.4}})

    payload = src.build_payload(tmp_path, action="status")

    assert payload["profile"] == "critical_backpressure"
    assert payload["throttle_controls"]["deferred_files_budget"] == 2
    assert payload["env_overrides"]["INGEST_MAX_DEFERRED_FILES"] == "2"
    assert payload["env_overrides"]["SQL_LINK_SERVICE_SHARD_EXPLANATIONS_MAX_FILES"] == "4"


def test_ingestion_storage_governor_treats_stale_stage_as_archive_debt_not_critical_pressure(tmp_path: Path) -> None:
    health = tmp_path / "governance" / "health"
    _write_json(health / "ingestion_storage_control_latest.json", {"overall_status": "ready"})
    _write_json(
        health / "ingestion_backpressure_latest.json",
        {
            "pending_lines": 270,
            "pending_lines_deferred": 115976,
            "pending_lines_cold": 115877,
            "pending_lines_stale_stage": 115877,
        },
    )
    _write_json(health / "ingestion_priority_queue_latest.json", {"lane_counts": {}})
    _write_json(health / "storage_mount_guard_latest.json", {"external_available": True, "storage_mode": "external"})
    _write_json(health / "storage_failback_sync_latest.json", {"mode": "external", "split_brain_conflicts": 0})
    _write_json(health / "storage_split_brain_reconciler_latest.json", {"summary": {"unresolved_conflicts": 0}})
    _write_json(
        health / "sql_link_service_latest.json",
        {"primary_db": str(tmp_path / "data" / "jsonl_link.sqlite3")},
    )
    _write_json(health / "sql_link_service_progress_latest.json", {})
    _write_json(
        health / "health_gates_latest.json",
        {
            "hard_gate_triggered": True,
            "hard_gates": {
                "collector_contracts": True,
                "ingestion_backpressure_overload": False,
                "priority_shard_storage": False,
                "sql_progress_stall": False,
                "sql_wal_pressure": False,
            },
            "storage_pressure": {"retention_debt_gb": 0.0},
        },
    )

    payload = src.build_payload(tmp_path, action="status")

    assert payload["profile"] == "elevated_backpressure"
    assert payload["pressure"]["stale_stage_pending_lines"] == 115877
    assert payload["env_overrides"]["LOG_GATE_EVALUATIONS"] == "0"
    assert payload["env_overrides"]["LOG_GATE_PASSES"] == "0"
    assert any("archive or reap" in action for action in payload["top_actions"])


def test_ingestion_storage_governor_treats_watchdog_support_backlog_as_non_core_pressure(tmp_path: Path) -> None:
    health = tmp_path / "governance" / "health"
    _write_json(health / "ingestion_storage_control_latest.json", {"overall_status": "ready"})
    _write_json(
        health / "ingestion_backpressure_latest.json",
        {
            "pending_lines": 1200,
            "pending_lines_deferred": 185000,
            "pending_lines_cold": 0,
            "pending_lines_support_telemetry": 178000,
            "pending_lines_stale_stage": 0,
        },
    )
    _write_json(health / "ingestion_priority_queue_latest.json", {"lane_counts": {}})
    _write_json(health / "storage_mount_guard_latest.json", {"external_available": True, "storage_mode": "external"})
    _write_json(health / "storage_failback_sync_latest.json", {"mode": "external", "split_brain_conflicts": 0})
    _write_json(health / "storage_split_brain_reconciler_latest.json", {"summary": {"unresolved_conflicts": 0}})
    _write_json(
        health / "sql_link_service_latest.json",
        {"primary_db": str(tmp_path / "data" / "jsonl_link.sqlite3")},
    )
    _write_json(health / "sql_link_service_progress_latest.json", {})
    _write_json(
        health / "health_gates_latest.json",
        {
            "hard_gate_triggered": False,
            "hard_gates": {
                "collector_contracts": False,
                "ingestion_backpressure_overload": False,
                "priority_shard_storage": False,
                "sql_progress_stall": False,
                "sql_wal_pressure": False,
            },
            "storage_pressure": {"retention_debt_gb": 0.0},
        },
    )

    payload = src.build_payload(tmp_path, action="status")

    assert payload["profile"] == "elevated_backpressure"
    assert payload["pressure"]["support_pending_lines"] == 178000
    assert payload["env_overrides"]["SQL_LINK_SERVICE_SHARD_SUPPORT_WATCHDOG_MAX_LINES_PER_FILE"] == "96000"
    assert any("support shard" in action for action in payload["top_actions"])
    assert payload["writer_shedding"]["shed_support_telemetry"] is True


def test_ingestion_storage_governor_write_override_includes_profile_and_queue_paths(tmp_path: Path) -> None:
    override = tmp_path / "config" / ".env.storage_pressure_override"

    changed = src._write_override(
        override,
        "critical_backpressure",
        {
            "SQL_LINK_SERVICE_PRIMARY_DB": str(tmp_path / "data" / "jsonl_link.sqlite3"),
            "BOT_CHANNEL_QUEUE_DB": str(tmp_path / "data" / "bot_channel_queue.sqlite3"),
        },
    )

    text = override.read_text(encoding="utf-8")
    assert changed is True
    assert "BOT_INGESTION_STORAGE_PROFILE=critical_backpressure" in text
    assert f"BOT_CHANNEL_QUEUE_DB={tmp_path / 'data' / 'bot_channel_queue.sqlite3'}" in text
