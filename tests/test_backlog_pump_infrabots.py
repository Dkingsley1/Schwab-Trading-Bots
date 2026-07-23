from __future__ import annotations

import json
from pathlib import Path

from scripts.ops import backlog_pump_infrabots as pumps


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_backlog_pump_infrabots_builds_ten_bot_contract(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("BACKLOG_SLEEVE_PUMP_MAX_ACTIVE_SLEEVES", "8")
    monkeypatch.setenv("SQL_LINK_SERVICE_MAX_SHARD_WRITER_LANES", "8")
    health = tmp_path / "governance" / "health"
    _write_json(
        health / "backlog_pcore_accelerator_latest.json",
        {
            "overall_status": "ready",
            "host_lane_contract": {"selected_p_core_preprocess_workers": 7},
            "storage_contract": {
                "green": False,
                "oldest_sources": [
                    {
                        "source_rel": "decisions/shadow_crypto/trade_decisions_20260601.jsonl",
                        "shard": "crypto_trading",
                        "pending_lines": 1000,
                        "oldest_pending_age_seconds": 500,
                    },
                    {
                        "source_rel": "decisions/shadow_fx_equities/trade_decisions_20260601.jsonl",
                        "shard": "trading",
                        "pending_lines": 500,
                        "oldest_pending_age_seconds": 600,
                    },
                ],
            },
            "sleeve_pump_contract": {"enabled": True, "max_active_sleeves_per_wave": 8, "selected_active_sleeve_slots": 7},
            "single_writer_tuning_contract": {
                "hot_batch_size": 360000,
                "sqlite_timeout_seconds": 600,
                "wal_checkpoint": {"enabled": True},
                "sqlite_memory": {"cache_size_kb": 65536},
            },
        },
    )
    _write_json(
        health / "writer_cycle_coordinator_latest.json",
        {
            "overall_status": "waiting_for_writer",
            "writer_state_before": {
                "active": True,
                "current_step": "merge_primary",
                "merged_rows_this_cycle": 10000,
                "cycle_age_minutes": 5,
                "progress_age_minutes": 1,
            },
        },
    )
    _write_json(
        health / "writer_process_intelligence_latest.json",
        {
            "overall_status": "advisory",
            "writer_health": {
                "active": True,
                "current_step": "merge_primary",
                "merged_rows_this_cycle": 10000,
                "cycle_age_minutes": 5,
                "progress_age_minutes": 1,
                "shard_writer_lane_contract": {"selected_shard_writer_lanes": 3, "max_shard_writer_lanes": 4},
            },
        },
    )

    payload = pumps.build_payload(tmp_path)

    assert set(payload["bots"]) == {
        "sleeve_pump_fairness_bot",
        "writer_throughput_sentinel",
        "pump_regression_guard",
        "stale_source_hunter",
        "sleeve_intake_governor",
        "wal_sqlite_steward",
        "shard_hotness_router_bot",
        "catch_up_wave_budget_bot",
        "stale_signal_arbitrator_bot",
        "writer_lane_preflight_bot",
    }
    assert payload["integration_contract"]["adds_parallel_sqlite_writers"] is False
    assert payload["integration_contract"]["added_speed_infrabot_count"] == 4
    assert payload["integration_contract"]["never_touch_protected_volumes"] == ["/Volumes/VIDEO"]
    assert payload["bots"]["pump_regression_guard"]["old_cycle_pending_new_contract"] is True
    assert payload["bots"]["writer_lane_preflight_bot"]["old_cycle_pending_new_contract"] is True
    assert payload["bots"]["sleeve_intake_governor"]["control_env"]["BOT_COLLECTION_DUTY_CYCLE_ENABLED"] == "1"
    env_lines = "\n".join(pumps._env_lines(payload))
    assert "BACKLOG_SHARD_HOTNESS_ROUTER_ENABLED=1" in env_lines
    assert "BACKLOG_CATCH_UP_WAVE_BUDGET_ENABLED=1" in env_lines
    assert "SQL_LINK_SERVICE_SINGLE_WRITER_ONLY=1" in env_lines


def test_backlog_pump_speed_helpers_target_stale_backpressure(tmp_path: Path) -> None:
    health = tmp_path / "governance" / "health"
    _write_json(
        health / "backlog_pcore_accelerator_latest.json",
        {
            "overall_status": "ready",
            "host_lane_contract": {"selected_p_core_preprocess_workers": 3},
            "storage_contract": {
                "green": True,
                "total_pending_lines": 1000,
                "oldest_pending_age_seconds": 5,
                "oldest_sources": [],
            },
            "storage_accelerator_contract": {
                "enabled": False,
                "catch_up_wave_controller": {"enabled": False, "max_waves": 1, "max_seconds_per_writer_cycle": 30},
            },
            "sleeve_pump_contract": {"enabled": True, "max_active_sleeves_per_wave": 8, "selected_active_sleeve_slots": 3},
            "single_writer_tuning_contract": {
                "hot_batch_size": 360000,
                "sqlite_timeout_seconds": 600,
                "wal_checkpoint": {"enabled": True},
                "sqlite_memory": {"cache_size_kb": 65536},
            },
        },
    )
    _write_json(
        health / "ingestion_storage_control_latest.json",
        {
            "overall_status": "ready",
            "backpressure": {
                "effective_raw_live": {
                    "core_pending_lines": 10487,
                    "total_pending_lines": 10487,
                    "oldest_pending_age_seconds": 323.132,
                    "artifact_age_seconds": 23025.538,
                    "artifact_stale_for_overlay_reconciliation": True,
                    "source": "sql_ingestion_overlay_pressure",
                },
                "raw_live": {
                    "total_pending_lines": 439,
                    "oldest_pending_age_seconds": 143.852,
                    "artifact_age_seconds": 23025.538,
                    "artifact_stale_for_overlay_reconciliation": True,
                },
                "pending_lines_threshold": 15000,
                "oldest_age_threshold_seconds": 240,
            },
            "backlog_relief_contract": {
                "active_issue_ids": [
                    "sparse_huge_jsonl_files",
                    "raw_live_expansion_headroom",
                    "stale_old_pending_work",
                ],
                "issues": [
                    {
                        "id": "stale_old_pending_work",
                        "evidence": {
                            "oldest_sources": [
                                {
                                    "source_rel": "governance/channels/decision/default_crypto_schwab/decision_20260613.jsonl",
                                    "shard": "crypto_trading",
                                    "pressure_lane": "core",
                                    "pending_lines": 8133,
                                    "oldest_pending_age_seconds": 259.505,
                                },
                                {
                                    "source_rel": "governance/channels/decision/crypto_futures_crypto_schwab/decision_20260613.jsonl",
                                    "shard": "crypto_trading",
                                    "pressure_lane": "core",
                                    "pending_lines": 2354,
                                    "oldest_pending_age_seconds": 323.132,
                                },
                            ]
                        },
                    }
                ],
                "accelerator_contract": {
                    "enabled": True,
                    "p_core_preprocess_workers": 3,
                    "max_shard_writer_lanes": 8,
                    "catch_up_wave_controller": {
                        "enabled": True,
                        "max_waves": 5,
                        "max_seconds_per_writer_cycle": 120,
                    },
                    "trigger_context": {
                        "active_issue_ids": [
                            "sparse_huge_jsonl_files",
                            "raw_live_expansion_headroom",
                            "stale_old_pending_work",
                        ]
                    },
                },
                "p_core_backlog_allocation_contract": {
                    "shard_link_writer_lanes": 3,
                    "max_shard_link_writer_lanes": 8,
                },
            },
        },
    )
    _write_json(
        health / "writer_cycle_coordinator_latest.json",
        {
            "overall_status": "applied_with_followups",
            "drain_effectiveness": {
                "status": "no_progress",
                "waves_run": 1,
                "pending_after": 10487,
                "oldest_pending_age_after_seconds": 323.132,
            },
        },
    )
    _write_json(
        health / "writer_process_intelligence_latest.json",
        {
            "overall_status": "advisory",
            "writer_health": {
                "active": True,
                "shard_writer_lane_contract": {"selected_shard_writer_lanes": 3, "max_shard_writer_lanes": 8},
            },
        },
    )

    payload = pumps.build_payload(tmp_path)

    router = payload["bots"]["shard_hotness_router_bot"]
    assert router["ranked_shards"][0]["shard"] == "crypto_trading"
    assert router["control_env"]["SQL_LINK_SERVICE_HOT_SHARD_PRIORITY"] == "crypto_trading"
    wave_budget = payload["bots"]["catch_up_wave_budget_bot"]
    assert wave_budget["recommended_wave_limit"] == 5
    assert wave_budget["status"] == "needs_work"
    stale = payload["bots"]["stale_signal_arbitrator_bot"]
    assert stale["status"] == "needs_work"
    assert "raw_live_snapshot_stale_for_overlay_reconciliation" in stale["blockers"]
