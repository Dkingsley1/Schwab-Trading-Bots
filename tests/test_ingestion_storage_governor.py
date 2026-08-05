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
    assert payload["env_overrides"]["LOG_GRAND_MASTER_DECISIONS"] == "0"
    assert payload["env_overrides"]["LOG_OPTIONS_MASTER_DECISIONS"] == "0"
    assert payload["env_overrides"]["LOG_FUTURES_MASTER_DECISIONS"] == "0"
    assert payload["env_overrides"]["LOG_DECISION_EXPLANATIONS"] == "0"
    assert payload["env_overrides"]["CHANNEL_LOG_PRIMARY_MODE"] == "channel"
    assert payload["env_overrides"]["LEGACY_HOT_CHANNEL_MIRROR_ENABLED"] == "0"
    assert payload["env_overrides"]["SQL_LINK_SERVICE_QUEUE_PRUNE_ORPHANS"] == "1"
    assert payload["env_overrides"]["SQL_LINK_SERVICE_QUEUE_MAX_DB_GB"] == "8"
    assert payload["env_overrides"]["INGEST_JOURNAL_DAILY_ENABLED"] == "0"
    assert payload["env_overrides"]["INGEST_JOURNAL_FILE_START_ENABLED"] == "0"
    assert payload["env_overrides"]["INGEST_JOURNAL_CHECKPOINT_ENABLED"] == "0"
    assert payload["env_overrides"]["INGEST_JOURNAL_ZERO_PENDING_ENABLED"] == "0"
    assert payload["env_overrides"]["RETENTION_STALE_PURGE_ENABLED"] == "1"
    assert payload["env_overrides"]["RETENTION_STALE_PURGE_LOW_VALUE_DAYS"] == "3"
    assert payload["env_overrides"]["RETENTION_STALE_PURGE_MAX_GB"] == "20"
    assert payload["throttle_controls"]["deferred_files_budget"] == 0
    assert payload["throttle_controls"]["queue_prune_orphans"] == "1"
    assert payload["throttle_controls"]["stale_purge_low_value_days"] == 3
    assert payload["throttle_controls"]["log_gate_evaluations"] == "0"
    assert payload["throttle_controls"]["log_grand_master_decisions"] == "0"
    assert payload["throttle_controls"]["log_options_master_decisions"] == "0"
    assert payload["throttle_controls"]["log_futures_master_decisions"] == "0"
    assert payload["throttle_controls"]["log_shadow_pnl_attribution"] == "0"
    assert payload["queue_watermarks"]["overall_status"] == "blocked"
    assert payload["writer_shedding"]["level"] == "protect_core"
    assert payload["writer_shedding"]["freeze_cold_lanes"] is True


def test_ingestion_storage_governor_accepts_verified_guarded_local_sqlite_route(tmp_path: Path) -> None:
    health = tmp_path / "governance" / "health"
    local_primary = tmp_path / "local_fallback_storage" / "data" / "jsonl_link.sqlite3"
    _write_json(health / "ingestion_storage_control_latest.json", {"overall_status": "ready"})
    _write_json(
        health / "ingestion_backpressure_latest.json",
        {"pending_lines": 0, "pending_lines_deferred": 0, "pending_lines_cold": 0},
    )
    _write_json(health / "ingestion_priority_queue_latest.json", {"lane_counts": {}})
    _write_json(health / "storage_mount_guard_latest.json", {"external_available": True, "storage_mode": "external"})
    _write_json(
        health / "storage_failback_sync_latest.json",
        {
            "mode": "external",
            "split_brain_conflicts": 0,
            "route_verification": {"verification_state": "ready"},
            "sqlite_skip_report": {
                "route_verification": {"verification_state": "ready"},
                "entries": [
                    {
                        "relative_path": "data/jsonl_link.sqlite3",
                        "classification": "active_local_route",
                        "active_path": str(local_primary),
                        "route_verification": {"state": "active_local_ready"},
                        "local": {"path": str(local_primary)},
                    }
                ],
            },
        },
    )
    _write_json(health / "storage_split_brain_reconciler_latest.json", {"summary": {"unresolved_conflicts": 0}})
    _write_json(health / "sql_link_service_latest.json", {"primary_db": str(local_primary)})
    _write_json(health / "sql_link_service_progress_latest.json", {})
    _write_json(health / "health_gates_latest.json", {"hard_gate_triggered": False, "storage_pressure": {"retention_debt_gb": 0.0}})

    payload = src.build_payload(tmp_path, action="status")

    assert payload["profile"] == "steady_state"
    assert payload["sql_primary_db"]["raw_route_drift"] is True
    assert payload["sql_primary_db"]["route_drift"] is False
    assert payload["sql_primary_db"]["guarded_local_sqlite_route"]["guarded"] is True
    assert not any("normalize SQL linker" in action for action in payload["top_actions"])


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
    assert payload["env_overrides"]["RETENTION_STALE_PURGE_MAX_FILES"] == "8000"


def test_ingestion_storage_governor_uses_ready_storage_control_over_stale_health_gate(tmp_path: Path) -> None:
    health = tmp_path / "governance" / "health"
    _write_json(
        health / "ingestion_storage_control_latest.json",
        {
            "overall_status": "ready",
            "severity": "stable",
            "pressure_index": 0.007,
            "backpressure": {
                "core_pending_lines": 74,
                "deferred_pending_lines": 1906,
                "cold_pending_lines": 0,
                "support_pending_lines": 0,
                "stale_stage_pending_lines": 0,
                "effective_raw_live_source": "raw_live_backpressure",
            },
            "data_integrity": {
                "sql_overlay_invalid_lines": 0,
                "sql_overlay_oversize_payloads": 0,
                "sql_overlay_ops_write_failures": 0,
            },
            "storage_efficiency_contract": {
                "active": False,
                "overall_status": "ready",
                "control_env_recommendations": {
                    "BOT_INGESTION_STORAGE_EFFICIENCY_CONTRACT_ACTIVE": "0",
                    "BOT_STORAGE_PLANE_PHASE": "steady_state",
                    "BOT_STORAGE_ALLOW_TRAINING": "1",
                    "BOT_STORAGE_ALLOW_EXPANSION": "1",
                },
            },
        },
    )
    _write_json(
        health / "ingestion_backpressure_latest.json",
        {"pending_lines": 74, "pending_lines_deferred": 1906, "pending_lines_cold": 0},
    )
    _write_json(health / "ingestion_priority_queue_latest.json", {"lane_counts": {}})
    _write_json(health / "storage_mount_guard_latest.json", {"external_available": False, "storage_mode": "local_fallback"})
    _write_json(
        health / "storage_failback_sync_latest.json",
        {"mode": "local_fallback", "split_brain_conflicts": 0, "route_verification": {"verification_state": "active_local_ready"}},
    )
    _write_json(health / "storage_split_brain_reconciler_latest.json", {"summary": {"unresolved_conflicts": 0}})
    _write_json(health / "sql_link_service_latest.json", {"primary_db": str(tmp_path / "data" / "jsonl_link.sqlite3")})
    _write_json(health / "sql_link_service_progress_latest.json", {})
    _write_json(
        health / "health_gates_latest.json",
        {
            "hard_gate_triggered": True,
            "hard_gates": {"ingestion_backpressure_overload": True},
            "ingestion_pressure": {"severe_backpressure_overload": True},
            "storage_pressure": {"retention_debt_gb": 0.0},
        },
    )

    payload = src.build_payload(tmp_path, action="status")

    assert payload["profile"] == "steady_state"
    assert payload["pressure"]["hard_gate"] is False
    assert payload["pressure"]["authoritative_storage_control"] is True
    assert payload["pressure"]["health_hard_gate_suppressed_by_storage_control"] is True
    assert payload["writer_shedding"]["level"] == "normal"
    assert payload["env_overrides"]["BOT_STORAGE_PLANE_PHASE"] == "steady_state"
    assert not any("maintenance priority" in action for action in payload["top_actions"])


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


def test_ingestion_storage_governor_sheds_support_target_pressure_when_critical(tmp_path: Path) -> None:
    health = tmp_path / "governance" / "health"
    _write_json(
        health / "ingestion_storage_control_latest.json",
        {
            "overall_status": "blocked",
            "pressure_index": 4.0,
            "severity": "critical",
            "backpressure": {"support_pending_lines": 46000},
        },
    )
    _write_json(
        health / "ingestion_backpressure_latest.json",
        {
            "pending_lines": 42000,
            "pending_lines_deferred": 0,
            "pending_lines_cold": 0,
            "pending_lines_support_telemetry": 0,
            "pending_lines_stale_stage": 0,
        },
    )
    _write_json(health / "ingestion_priority_queue_latest.json", {"lane_counts": {}})
    _write_json(health / "storage_mount_guard_latest.json", {"external_available": True, "storage_mode": "external"})
    _write_json(health / "storage_failback_sync_latest.json", {"mode": "external", "split_brain_conflicts": 0})
    _write_json(health / "storage_split_brain_reconciler_latest.json", {"summary": {"unresolved_conflicts": 0}})
    _write_json(health / "sql_link_service_latest.json", {"primary_db": str(tmp_path / "data" / "jsonl_link.sqlite3")})
    _write_json(health / "sql_link_service_progress_latest.json", {})
    _write_json(
        health / "health_gates_latest.json",
        {
            "hard_gate_triggered": True,
            "hard_gates": {"ingestion_backpressure_overload": True},
            "storage_pressure": {"retention_debt_gb": 0.0},
        },
    )

    payload = src.build_payload(tmp_path, action="status")

    assert payload["profile"] == "critical_backpressure"
    assert payload["writer_shedding"]["shed_support_telemetry"] is True
    assert payload["writer_shedding"]["support_target_pressure"] is True
    assert "support_telemetry" in payload["writer_shedding"]["target_breaches"]
    assert payload["throttle_controls"]["ingest_journal_daily_enabled"] == "0"


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


def test_ingestion_storage_governor_shell_quotes_values_with_spaces(tmp_path: Path) -> None:
    override = tmp_path / "config" / ".env.storage_pressure_override"

    changed = src._write_override(
        override,
        "critical_backpressure",
        {
            "BACKLOG_PCORE_BURST_MODE": "foreground_protect",
            "BACKLOG_PCORE_BURST_REASON": "foreground, memory, compute, or host pressure needs extra headroom",
        },
    )

    text = override.read_text(encoding="utf-8")
    assert changed is True
    assert "BACKLOG_PCORE_BURST_MODE=foreground_protect" in text
    assert "BACKLOG_PCORE_BURST_REASON='foreground, memory, compute, or host pressure needs extra headroom'" in text


def test_ingestion_storage_governor_applies_backlog_relief_contract_env(tmp_path: Path) -> None:
    health = tmp_path / "governance" / "health"
    _write_json(
        health / "ingestion_storage_control_latest.json",
        {
            "overall_status": "blocked",
            "severity": "critical",
            "pressure_index": 3.2,
            "backlog_relief_contract": {
                "active": True,
                "active_issue_ids": [
                    "single_writer_merge_speed",
                    "storage_write_latency",
                    "sparse_huge_jsonl_files",
                    "intake_outpaces_drain",
                    "raw_live_expansion_headroom",
                    "stale_old_pending_work",
                ],
                "control_env_recommendations": {
                    "RAW_LIVE_EXPANSION_GUARD_ACTIVE": "1",
                    "RAW_LIVE_EXPANSION_READY": "0",
                    "RAW_LIVE_EXPANSION_TIER": "blocked_until_raw_live_cools",
                    "RAW_LIVE_CORE_RESERVE_TARGET": "4000",
                    "RAW_LIVE_TOTAL_RESERVE_TARGET": "5500",
                    "SHADOW_LOOP_FRESH_BACKLOG_PAUSE_LINES": "4000",
                    "BOT_COLLECTION_DUTY_CYCLE_ENABLED": "1",
                    "BOT_COLLECTION_DUTY_CYCLE_MAX_ACTIVE_RATIO": "0.16",
                    "SQL_LINK_SERVICE_RAW_LIVE_PRIORITY_BOOST": "1",
                    "SQL_LINK_SERVICE_COLD_STAGE_YIELDS_TO_RAW_LIVE": "1",
                },
                "p_core_backlog_allocation_contract": {
                    "active": True,
                    "control_env": {
                        "BACKLOG_PCORE_ALLOCATION_ACTIVE": "1",
                        "BACKLOG_DRAIN_SINGLE_WRITER_ONLY": "1",
                        "SQL_LINK_SERVICE_SINGLE_WRITER_ONLY": "1",
                        "BACKLOG_PCORE_PREPROCESS_WORKERS": "4",
                        "SQL_LINK_SERVICE_PREPROCESS_WORKERS": "4",
                        "TRAINING_PCORE_ALLOWED_WHEN_BACKLOG_GREEN": "1",
                        "TRAINING_PCORE_MAX_WORKERS": "2",
                        "TRAINING_PCORE_NICE": "8",
                    },
                },
            },
        },
    )
    _write_json(
        health / "ingestion_backpressure_latest.json",
        {"pending_lines": 42000, "pending_lines_deferred": 9000, "pending_lines_cold": 0},
    )
    _write_json(health / "ingestion_priority_queue_latest.json", {"lane_counts": {}})
    _write_json(health / "storage_mount_guard_latest.json", {"external_available": True, "storage_mode": "external"})
    _write_json(health / "storage_failback_sync_latest.json", {"mode": "external", "split_brain_conflicts": 0})
    _write_json(health / "storage_split_brain_reconciler_latest.json", {"summary": {"unresolved_conflicts": 0}})
    _write_json(health / "sql_link_service_latest.json", {"primary_db": str(tmp_path / "data" / "jsonl_link.sqlite3")})
    _write_json(health / "sql_link_service_progress_latest.json", {})
    _write_json(health / "health_gates_latest.json", {"hard_gate_triggered": True, "hard_gates": {"ingestion_backpressure_overload": True}, "storage_pressure": {"retention_debt_gb": 0.0}})

    payload = src.build_payload(tmp_path, action="status")

    env = payload["env_overrides"]
    assert env["BACKLOG_RELIEF_CONTRACT_ACTIVE"] == "1"
    assert "single_writer_merge_speed" in env["BACKLOG_RELIEF_ACTIVE_ISSUES"]
    assert env["SQL_LINK_SERVICE_MERGE_MAX_SECONDS_PER_CYCLE"] == "90"
    assert env["SQLITE_CACHE_SIZE_KB"] == "32768"
    assert env["INGEST_MAX_BYTES_PER_FILE"] == str(128 * 1024 * 1024)
    assert env["BOT_COLLECTION_DUTY_CYCLE_MAX_ACTIVE_RATIO"] == "0.16"
    assert env["BOT_COLLECTION_DUTY_CYCLE_A_PLUS_PLUS_TARGET"] == "1"
    assert env["RAW_LIVE_EXPANSION_GUARD_ACTIVE"] == "1"
    assert env["RAW_LIVE_EXPANSION_TIER"] == "blocked_until_raw_live_cools"
    assert env["SQL_LINK_SERVICE_RAW_LIVE_PRIORITY_BOOST"] == "1"
    assert env["SQL_LINK_SERVICE_COLD_STAGE_YIELDS_TO_RAW_LIVE"] == "1"
    assert env["WRITER_CYCLE_MAX_CATCH_UP_WAVES"] == "3"
    assert env["BACKLOG_PCORE_ALLOCATION_ACTIVE"] == "1"
    assert env["BACKLOG_DRAIN_SINGLE_WRITER_ONLY"] == "1"
    assert env["SQL_LINK_SERVICE_PREPROCESS_WORKERS"] == "4"
    assert env["TRAINING_PCORE_ALLOWED_WHEN_BACKLOG_GREEN"] == "1"
    assert payload["throttle_controls"]["backlog_relief_contract_active"] == "1"
    assert payload["throttle_controls"]["raw_live_expansion_guard_active"] == "1"
    assert payload["throttle_controls"]["raw_live_core_reserve_target"] == 4000
    assert payload["throttle_controls"]["p_core_backlog_allocation_active"] == "1"
    assert payload["throttle_controls"]["p_core_preprocess_workers"] == 4


def test_ingestion_storage_governor_keeps_healthy_raw_live_duty_cap_authoritative(tmp_path: Path) -> None:
    health = tmp_path / "governance" / "health"
    _write_json(
        health / "ingestion_storage_control_latest.json",
        {
            "overall_status": "ready",
            "severity": "stable",
            "pressure_index": 0.2,
            "backpressure": {
                "core_pending_lines": 1800,
                "deferred_pending_lines": 0,
                "cold_pending_lines": 0,
                "support_pending_lines": 0,
                "stale_stage_pending_lines": 0,
                "effective_raw_live_source": "raw_live_backpressure",
            },
            "raw_live_expansion_contract": {
                "active": False,
                "expansion_ready": True,
                "control_env": {
                    "RAW_LIVE_EXPANSION_GUARD_ACTIVE": "0",
                    "RAW_LIVE_EXPANSION_READY": "1",
                    "RAW_LIVE_EXPANSION_TIER": "ready_for_bigger_expansion",
                    "RAW_LIVE_CORE_RESERVE_TARGET": "4000",
                    "RAW_LIVE_TOTAL_RESERVE_TARGET": "5500",
                    "SHADOW_LOOP_FRESH_BACKLOG_PAUSE_LINES": "4000",
                    "SHADOW_LOOP_FRESH_BACKLOG_INFLIGHT_RESERVE_LINES": "2000",
                    "SIGNAL_GENERATION_SUB_BOT_SAMPLE_MODULUS": "2",
                    "SHADOW_LOOP_BOOTSTRAP_BACKLOG_STAGGER_ENABLED": "1",
                    "BOT_COLLECTION_DUTY_CYCLE_ENABLED": "1",
                    "BOT_COLLECTION_DUTY_CYCLE_MAX_ACTIVE_RATIO": "0.24",
                    "SQL_LINK_SERVICE_RAW_LIVE_PRIORITY_BOOST": "0",
                    "SQL_LINK_SERVICE_RAW_LIVE_AUTO_FOCUS_ENABLED": "1",
                    "SQL_LINK_SERVICE_RAW_LIVE_PRIORITY_MIN_PENDING_LINES": "2000",
                    "SQL_LINK_SERVICE_RAW_LIVE_PRIORITY_AGED_SOURCE_SECONDS": "180.0",
                },
            },
            "backlog_relief_contract": {
                "active": False,
                "active_issue_ids": [],
                "control_env_recommendations": {
                    "BOT_COLLECTION_DUTY_CYCLE_ENABLED": "1",
                    "BOT_COLLECTION_DUTY_CYCLE_MAX_ACTIVE_RATIO": "0.35",
                },
            },
            "storage_efficiency_contract": {
                "active": False,
                "control_env_recommendations": {
                    "BOT_COLLECTION_DUTY_CYCLE_ENABLED": "1",
                    "BOT_COLLECTION_DUTY_CYCLE_MAX_ACTIVE_RATIO": "0.35",
                },
            },
        },
    )
    _write_json(health / "ingestion_backpressure_latest.json", {"pending_lines": 1800})
    _write_json(health / "ingestion_priority_queue_latest.json", {"lane_counts": {}})
    _write_json(health / "storage_mount_guard_latest.json", {"external_available": True, "storage_mode": "external"})
    _write_json(health / "storage_failback_sync_latest.json", {"mode": "external", "split_brain_conflicts": 0})
    _write_json(health / "storage_split_brain_reconciler_latest.json", {"summary": {"unresolved_conflicts": 0}})
    _write_json(health / "sql_link_service_latest.json", {"primary_db": str(tmp_path / "data" / "jsonl_link.sqlite3")})
    _write_json(health / "sql_link_service_progress_latest.json", {})
    _write_json(health / "health_gates_latest.json", {"hard_gate_triggered": False, "storage_pressure": {"retention_debt_gb": 0.0}})

    payload = src.build_payload(tmp_path, action="status")

    assert payload["profile"] == "steady_state"
    assert payload["env_overrides"]["BOT_COLLECTION_DUTY_CYCLE_MAX_ACTIVE_RATIO"] == "0.24"
    assert payload["throttle_controls"]["collection_duty_cycle_max_active_ratio"] == "0.24"
    assert payload["throttle_controls"]["raw_live_expansion_ready"] == "1"
    assert payload["env_overrides"]["SHADOW_LOOP_FRESH_BACKLOG_PAUSE_LINES"] == "4000"
    assert payload["env_overrides"]["SHADOW_LOOP_FRESH_BACKLOG_INFLIGHT_RESERVE_LINES"] == "2000"
    assert payload["env_overrides"]["SIGNAL_GENERATION_SUB_BOT_SAMPLE_MODULUS"] == "2"
    assert payload["env_overrides"]["SQL_LINK_SERVICE_RAW_LIVE_PRIORITY_AGED_SOURCE_SECONDS"] == "180.0"
    assert payload["env_overrides"]["SHADOW_LOOP_BOOTSTRAP_BACKLOG_STAGGER_ENABLED"] == "1"
    assert payload["env_overrides"]["SQL_LINK_SERVICE_RAW_LIVE_AUTO_FOCUS_ENABLED"] == "1"
    assert payload["env_overrides"]["SQL_LINK_SERVICE_RAW_LIVE_PRIORITY_MIN_PENDING_LINES"] == "2000"


def test_ingestion_storage_governor_applies_storage_efficiency_contract_env(tmp_path: Path) -> None:
    health = tmp_path / "governance" / "health"
    _write_json(
        health / "ingestion_storage_control_latest.json",
        {
            "overall_status": "blocked",
            "severity": "critical",
            "pressure_index": 3.4,
            "storage_efficiency_contract": {
                "active": True,
                "active_blockers": [
                    "duplicate_fallback_artifacts",
                    "raw_training_compaction_debt",
                    "fallback_route_reconciliation",
                ],
                "control_env_recommendations": {
                    "BOT_INGESTION_STORAGE_EFFICIENCY_CONTRACT_ACTIVE": "1",
                    "BOT_STORAGE_PLANE_PHASE": "emergency_disk_guard",
                    "BOT_STORAGE_EMERGENCY_DISK_GUARD": "1",
                    "BOT_STORAGE_EXTERNAL_FREE_GB": "1.5",
                    "BOT_STORAGE_EXTERNAL_MIN_FREE_GB": "32.0",
                    "BOT_STORAGE_ALLOW_RAW_COMPACTION_APPLY": "0",
                    "BOT_STORAGE_ALLOW_TRAINING": "0",
                    "BOT_STORAGE_ALLOW_EXPANSION": "0",
                    "BOT_INGESTION_STORAGE_MODE": "fallback_reconcile_first",
                    "BOT_DATA_CAPTURE_MODE": "manifest_only_hot_path",
                    "BOT_RAW_PAYLOAD_STORAGE_MODE": "manifest_first",
                    "BOT_FALLBACK_DUPLICATE_SUPPRESSION": "1",
                    "BOT_LOCAL_FALLBACK_RECONCILE_BEFORE_EXPAND": "1",
                    "BOT_RAW_TRAINING_MANIFEST_REFRESH_REQUIRED": "1",
                    "BOT_RAW_TRAINING_COMPACTION_REQUIRED": "1",
                    "BOT_RAW_TRAINING_COMPACTION_APPLY_ALLOWED_NOW": "0",
                    "BOT_RAW_TRAINING_WAVE_MAX_FILES": "4",
                    "BOT_RAW_TRAINING_WAVE_MAX_GB": "2.0",
                    "BOT_COLLECTION_DUTY_CYCLE_ENABLED": "1",
                    "BOT_COLLECTION_DUTY_CYCLE_MAX_ACTIVE_RATIO": "0.15",
                },
            },
        },
    )
    _write_json(
        health / "ingestion_backpressure_latest.json",
        {"pending_lines": 38000, "pending_lines_deferred": 7000, "pending_lines_cold": 0},
    )
    _write_json(health / "ingestion_priority_queue_latest.json", {"lane_counts": {}})
    _write_json(health / "storage_mount_guard_latest.json", {"external_available": True, "storage_mode": "external"})
    _write_json(health / "storage_failback_sync_latest.json", {"mode": "external", "split_brain_conflicts": 0})
    _write_json(health / "storage_split_brain_reconciler_latest.json", {"summary": {"unresolved_conflicts": 0}})
    _write_json(health / "sql_link_service_latest.json", {"primary_db": str(tmp_path / "data" / "jsonl_link.sqlite3")})
    _write_json(health / "sql_link_service_progress_latest.json", {})
    _write_json(health / "health_gates_latest.json", {"hard_gate_triggered": True, "hard_gates": {"ingestion_backpressure_overload": True}, "storage_pressure": {"retention_debt_gb": 0.0}})

    payload = src.build_payload(tmp_path, action="status")

    env = payload["env_overrides"]
    assert env["BOT_INGESTION_STORAGE_EFFICIENCY_CONTRACT_ACTIVE"] == "1"
    assert env["BOT_DATA_CAPTURE_MODE"] == "manifest_only_hot_path"
    assert env["BOT_RAW_PAYLOAD_STORAGE_MODE"] == "manifest_first"
    assert env["BOT_LOCAL_FALLBACK_RECONCILE_BEFORE_EXPAND"] == "1"
    assert env["BOT_RAW_TRAINING_MANIFEST_REFRESH_REQUIRED"] == "1"
    assert env["BOT_STORAGE_PLANE_PHASE"] == "emergency_disk_guard"
    assert payload["throttle_controls"]["storage_efficiency_contract_active"] == "1"
    assert payload["throttle_controls"]["storage_plane_phase"] == "emergency_disk_guard"
    assert payload["throttle_controls"]["storage_emergency_disk_guard"] == "1"
    assert payload["throttle_controls"]["storage_external_free_gb"] == 1.5
    assert payload["throttle_controls"]["storage_allow_raw_compaction_apply"] == "0"
    assert payload["throttle_controls"]["storage_allow_training"] == "0"
    assert payload["throttle_controls"]["storage_allow_expansion"] == "0"
    assert payload["throttle_controls"]["data_capture_mode"] == "manifest_only_hot_path"
    assert payload["throttle_controls"]["raw_payload_storage_mode"] == "manifest_first"
    assert payload["throttle_controls"]["raw_training_wave_max_files"] == 4
    assert payload["throttle_controls"]["raw_training_wave_max_gb"] == 2.0
    assert payload["storage_efficiency_contract"]["active"] is True
    assert any("storage efficiency contract" in action for action in payload["top_actions"])


def test_ingestion_storage_governor_keeps_backlog_relief_authoritative_after_storage_efficiency(tmp_path: Path) -> None:
    health = tmp_path / "governance" / "health"
    _write_json(
        health / "ingestion_storage_control_latest.json",
        {
            "overall_status": "blocked",
            "severity": "critical",
            "pressure_index": 4.2,
            "backlog_relief_contract": {
                "active": True,
                "active_issue_ids": [
                    "single_writer_merge_speed",
                    "sparse_huge_jsonl_files",
                    "intake_outpaces_drain",
                    "stale_old_pending_work",
                ],
                "p_core_backlog_allocation_contract": {
                    "active": True,
                    "control_env": {
                        "BACKLOG_PCORE_ALLOCATION_ACTIVE": "1",
                        "BACKLOG_DRAIN_SINGLE_WRITER_ONLY": "1",
                        "SQL_LINK_SERVICE_SINGLE_WRITER_ONLY": "1",
                        "BACKLOG_PCORE_PREPROCESS_WORKERS": "3",
                        "BACKLOG_PCORE_USER_APP_RESERVE_TARGET": "5",
                        "BACKLOG_PCORE_BURST_MODE": "protect_live_backlog_probe_3",
                        "SQL_LINK_SERVICE_PREPROCESS_WORKERS": "3",
                        "SQL_LINK_SERVICE_SHARD_WRITER_LANES": "3",
                        "SQL_LINK_SERVICE_MAX_SHARD_WRITER_LANES": "7",
                    },
                },
            },
            "storage_efficiency_contract": {
                "active": True,
                "active_blockers": ["raw_training_compaction_debt"],
                "control_env_recommendations": {
                    "BOT_INGESTION_STORAGE_EFFICIENCY_CONTRACT_ACTIVE": "1",
                    "BOT_STORAGE_PLANE_PHASE": "manifest_only_recovery",
                    "BOT_COLLECTION_DUTY_CYCLE_ENABLED": "1",
                    "BOT_COLLECTION_DUTY_CYCLE_MAX_ACTIVE_RATIO": "0.15",
                    "BACKLOG_PCORE_PREPROCESS_WORKERS": "4",
                    "BACKLOG_PCORE_USER_APP_RESERVE_TARGET": "0",
                    "BACKLOG_PCORE_BURST_MODE": "stale_service_request_4",
                    "SQL_LINK_SERVICE_PREPROCESS_WORKERS": "4",
                    "SQL_LINK_SERVICE_SHARD_WRITER_LANES": "4",
                    "BOT_RAW_TRAINING_COMPACTION_REQUIRED": "1",
                },
            },
        },
    )
    _write_json(
        health / "ingestion_backpressure_latest.json",
        {"pending_lines": 68000, "pending_lines_deferred": 9000, "pending_lines_cold": 0},
    )
    _write_json(health / "ingestion_priority_queue_latest.json", {"lane_counts": {}})
    _write_json(health / "storage_mount_guard_latest.json", {"external_available": True, "storage_mode": "external"})
    _write_json(health / "storage_failback_sync_latest.json", {"mode": "external", "split_brain_conflicts": 0})
    _write_json(health / "storage_split_brain_reconciler_latest.json", {"summary": {"unresolved_conflicts": 0}})
    _write_json(health / "sql_link_service_latest.json", {"primary_db": str(tmp_path / "data" / "jsonl_link.sqlite3")})
    _write_json(health / "sql_link_service_progress_latest.json", {})
    _write_json(
        health / "health_gates_latest.json",
        {"hard_gate_triggered": True, "hard_gates": {"ingestion_backpressure_overload": True}, "storage_pressure": {"retention_debt_gb": 0.0}},
    )

    payload = src.build_payload(tmp_path, action="status")

    env = payload["env_overrides"]
    assert env["BOT_INGESTION_STORAGE_EFFICIENCY_CONTRACT_ACTIVE"] == "1"
    assert env["BOT_RAW_TRAINING_COMPACTION_REQUIRED"] == "1"
    assert env["BOT_COLLECTION_DUTY_CYCLE_MAX_ACTIVE_RATIO"] == "0.20"
    assert env["BACKLOG_PCORE_PREPROCESS_WORKERS"] == "3"
    assert env["BACKLOG_PCORE_USER_APP_RESERVE_TARGET"] == "5"
    assert env["BACKLOG_PCORE_BURST_MODE"] == "protect_live_backlog_probe_3"
    assert env["SQL_LINK_SERVICE_PREPROCESS_WORKERS"] == "3"
    assert env["SQL_LINK_SERVICE_SHARD_WRITER_LANES"] == "3"
    assert payload["throttle_controls"]["p_core_preprocess_workers"] == 3
    assert payload["throttle_controls"]["collection_duty_cycle_max_active_ratio"] == "0.20"
