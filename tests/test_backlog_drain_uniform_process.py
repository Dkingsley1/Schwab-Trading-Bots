from __future__ import annotations

import json
from pathlib import Path

from scripts.ops import backlog_drain_uniform_process as uniform


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_uniform_process_late_override_pins_hot_overlay_without_parallel_writers(tmp_path: Path) -> None:
    health = tmp_path / "governance" / "health"
    _write_json(
        health / "ingestion_storage_control_latest.json",
        {
            "overall_status": "blocked",
            "severity": "critical",
            "pressure_index": 5.056,
            "backlog_truth": {
                "raw_live": {
                    "grade": "A+",
                    "core_pending_lines": 439,
                    "total_pending_lines": 439,
                    "oldest_pending_age_seconds": 143.852,
                },
                "sql_overlay": {
                    "grade": "F",
                    "pressure_ratio": 5.056,
                    "core_pending_lines": 10199,
                    "total_pending_lines": 10199,
                    "oldest_pending_age_seconds": 1213.506,
                    "used_for_pressure": True,
                },
                "truth_gap": {"pending_line_delta": 9760},
            },
            "backlog_relief_contract": {
                "active_issue_ids": [
                    "sparse_huge_jsonl_files",
                    "raw_live_expansion_headroom",
                    "stale_old_pending_work",
                ]
            },
            "stale_pending_locator": {
                "oldest_sources": [
                    {
                        "source_rel": "governance/channels/decision/default_crypto_schwab/decision_20260613.jsonl",
                        "shard": "crypto_trading",
                        "pressure_lane": "core",
                        "pending_lines": 7989,
                        "oldest_pending_age_seconds": 1187.186,
                    }
                ]
            },
        },
    )
    _write_json(
        health / "backlog_pcore_accelerator_latest.json",
        {
            "overall_status": "advisory",
            "host_lane_contract": {"selected_p_core_preprocess_workers": 2},
            "storage_accelerator_contract": {
                "p_core_preprocess_workers": 3,
                "max_shard_writer_lanes": 8,
                "catch_up_wave_controller": {
                    "enabled": True,
                    "max_waves": 5,
                    "max_seconds_per_writer_cycle": 120,
                },
            },
            "single_writer_tuning_contract": {"hot_batch_size": 120000},
        },
    )
    _write_json(
        health / "backlog_pump_infrabots_latest.json",
        {
            "overall_status": "advisory",
            "bots": {
                "shard_hotness_router_bot": {
                    "control_env": {"SQL_LINK_SERVICE_HOT_SHARD_PRIORITY": "crypto_trading"},
                    "focused_sources": [
                        {
                            "source_rel": "governance/channels/decision/default_crypto_schwab/decision_20260613.jsonl",
                            "shard": "crypto_trading",
                            "pressure_lane": "core",
                            "pending_lines": 7989,
                            "oldest_pending_age_seconds": 1187.186,
                        }
                    ],
                },
                "catch_up_wave_budget_bot": {
                    "control_env": {
                        "WRITER_CYCLE_MAX_CATCH_UP_WAVES": "5",
                        "SQL_LINK_SERVICE_MERGE_MAX_SECONDS_PER_CYCLE": "120",
                    }
                },
            },
        },
    )
    _write_json(health / "writer_cycle_coordinator_latest.json", {"overall_status": "waiting_for_writer"})
    _write_json(
        health / "writer_process_intelligence_latest.json",
        {
            "overall_status": "ready",
            "writer_health": {
                "active": True,
                "current_step": "shard_linking",
                "shard_writer_lane_contract": {
                    "selected_shard_writer_lanes": 2,
                    "max_shard_writer_lanes": 2,
                    "single_primary_merge_writer": True,
                },
            },
        },
    )
    _write_json(health / "memory_pressure_intelligence_latest.json", {"overall_status": "ready", "memory_status": "soft_guard"})

    payload = uniform.build_payload(tmp_path)
    contract = payload["speed_contract"]

    assert contract["canonical_pressure"]["source"] == "sql_overlay_attributed"
    assert contract["target_shards"] == ["crypto_trading"]
    assert contract["wave_limit"] == 5
    assert contract["hot_batch_size"] == 240000
    assert contract["turbo_contract"]["enabled"] is False
    assert "storage_still_critical" in contract["turbo_contract"]["blockers"]
    assert contract["lane_contract"]["live_preprocess_workers"] == 2
    assert contract["lane_contract"]["desired_preprocess_workers_when_host_clear"] == 3
    assert payload["integration_contract"]["single_sqlite_writer_only"] is True
    assert payload["integration_contract"]["adds_parallel_sqlite_writers"] is False

    env = "\n".join(uniform._env_lines(payload))
    assert "BACKLOG_DRAIN_UNIFORM_PROCESS_ENABLED=1" in env
    assert "SQL_LINK_SERVICE_HOT_SHARD_PRIORITY=crypto_trading" in env
    assert "SQL_LINK_SERVICE_HOT_BATCH_SIZE=240000" in env
    assert "SQL_LINK_SERVICE_SHARD_WRITER_LANES=2" in env


def test_uniform_process_fast_paths_completed_writer_handoff(tmp_path: Path) -> None:
    health = tmp_path / "governance" / "health"
    _write_json(
        health / "ingestion_storage_control_latest.json",
        {
            "overall_status": "blocked",
            "severity": "critical",
            "pressure_index": 2.0,
            "backlog_truth": {
                "raw_live": {
                    "core_pending_lines": 1200,
                    "total_pending_lines": 1400,
                    "oldest_pending_age_seconds": 400.0,
                }
            },
        },
    )
    _write_json(health / "backlog_pcore_accelerator_latest.json", {})
    _write_json(health / "backlog_pump_infrabots_latest.json", {})
    _write_json(
        health / "writer_cycle_coordinator_latest.json",
        {
            "overall_status": "handoff_needed",
            "writer_state_before": {
                "active": True,
                "active_source": "completed_lock_handoff_needed",
                "complete_lock_handoff_needed": True,
                "current_step": "complete",
            },
            "summary": {"completed_writer_lock_handoff_needed": True},
        },
    )
    _write_json(health / "writer_process_intelligence_latest.json", {"writer_health": {}})
    _write_json(health / "storage_backpressure_autopilot_latest.json", {})
    _write_json(health / "memory_pressure_intelligence_latest.json", {"classification": {"status": "soft_guard"}})

    payload = uniform.build_payload(tmp_path)
    handoff_step = next(row for row in payload["uniform_process_steps"] if row["step"] == "single_writer_handoff_or_wait")

    assert payload["speed_contract"]["writer_state"]["completed_lock_handoff_needed"] is True
    assert "--handoff-only" in handoff_step["command"]
    assert handoff_step["reason"] == "clear_completed_writer_lock_handoff"


def test_uniform_process_turbo_narrows_to_hot_shards_and_lifts_single_writer_batches(tmp_path: Path) -> None:
    health = tmp_path / "governance" / "health"
    _write_json(
        health / "ingestion_storage_control_latest.json",
        {
            "overall_status": "ready",
            "severity": "stable",
            "pressure_index": 0.649,
            "backlog_truth": {
                "raw_live": {
                    "grade": "A+",
                    "core_pending_lines": 439,
                    "total_pending_lines": 439,
                    "oldest_pending_age_seconds": 143.852,
                },
                "sql_overlay": {
                    "grade": "B",
                    "pressure_ratio": 0.649,
                    "core_pending_lines": 9739,
                    "total_pending_lines": 9739,
                    "oldest_pending_age_seconds": 68.399,
                    "used_for_pressure": True,
                },
                "truth_gap": {"pending_line_delta": 9300},
            },
            "backlog_relief_contract": {
                "active_issue_ids": [
                    "sparse_huge_jsonl_files",
                    "raw_live_expansion_headroom",
                ]
            },
            "stale_pending_locator": {
                "oldest_sources": [
                    {
                        "source_rel": "governance/channels/decision/default_crypto_schwab/decision_20260613.jsonl",
                        "shard": "crypto_trading",
                        "pressure_lane": "core",
                        "pending_lines": 7989,
                        "oldest_pending_age_seconds": 68.399,
                    }
                ]
            },
        },
    )
    _write_json(
        health / "backlog_pcore_accelerator_latest.json",
        {
            "overall_status": "advisory",
            "host_lane_contract": {"selected_p_core_preprocess_workers": 2, "memory_status": "soft_guard"},
            "storage_accelerator_contract": {
                "p_core_preprocess_workers": 3,
                "max_shard_writer_lanes": 8,
                "catch_up_wave_controller": {
                    "enabled": True,
                    "max_waves": 5,
                    "max_seconds_per_writer_cycle": 120,
                },
            },
            "single_writer_tuning_contract": {"hot_batch_size": 120000},
        },
    )
    _write_json(
        health / "backlog_pump_infrabots_latest.json",
        {
            "overall_status": "advisory",
            "bots": {
                "shard_hotness_router_bot": {
                    "control_env": {"SQL_LINK_SERVICE_HOT_SHARD_PRIORITY": "crypto_trading"},
                    "focused_sources": [
                        {
                            "source_rel": "governance/channels/decision/default_crypto_schwab/decision_20260613.jsonl",
                            "shard": "crypto_trading",
                            "pressure_lane": "core",
                            "pending_lines": 7989,
                            "oldest_pending_age_seconds": 68.399,
                        }
                    ],
                },
                "catch_up_wave_budget_bot": {
                    "control_env": {
                        "WRITER_CYCLE_MAX_CATCH_UP_WAVES": "5",
                        "SQL_LINK_SERVICE_MERGE_MAX_SECONDS_PER_CYCLE": "120",
                    }
                },
            },
        },
    )
    _write_json(health / "writer_cycle_coordinator_latest.json", {"overall_status": "waiting_for_writer"})
    _write_json(
        health / "writer_process_intelligence_latest.json",
        {
            "overall_status": "ready",
            "writer_health": {
                "active": True,
                "current_step": "shard_linking",
                "shard_writer_lane_contract": {
                    "selected_shard_writer_lanes": 2,
                    "max_shard_writer_lanes": 2,
                    "single_primary_merge_writer": True,
                },
            },
        },
    )
    _write_json(
        health / "memory_pressure_intelligence_latest.json",
        {
            "overall_status": "advisory",
            "classification": {"status": "soft_guard"},
            "snapshot": {
                "pressure_level": "normal",
                "pressure_kind": "normal",
                "swap_used_gb": 2.0,
                "compressed_pressure_gb": 1.0,
                "pages_throttled": 0,
            },
        },
    )

    payload = uniform.build_payload(tmp_path)
    contract = payload["speed_contract"]

    assert contract["mode"] == "turbo_plus_single_writer_catchup"
    assert contract["turbo_contract"]["enabled"] is True
    assert contract["turbo_contract"]["turbo_plus_enabled"] is True
    assert contract["turbo_contract"]["shard_scope"] == ["health_fast", "writer_progress", "crypto_trading"]
    assert contract["wave_limit"] == 6
    assert contract["max_seconds_per_cycle"] == 210
    assert contract["hot_batch_size"] == 420000
    assert contract["queue_batch_size"] == 340000
    assert contract["poll_seconds"] == 6
    assert contract["storage_autopilot_cycles"] == 2
    assert contract["single_writer_only"] is True
    assert contract["adds_parallel_sqlite_writers"] is False

    env = "\n".join(uniform._env_lines(payload))
    assert "BACKLOG_DRAIN_TURBO_ENABLED=1" in env
    assert "BACKLOG_DRAIN_TURBO_PLUS_ENABLED=1" in env
    assert "SQL_LINK_SERVICE_SHARDS=health_fast,writer_progress,crypto_trading" in env
    assert "SQL_LINK_SERVICE_HOT_BATCH_SIZE=420000" in env
    assert "SQL_LINK_SERVICE_QUEUE_BATCH_SIZE=340000" in env
    assert "SQL_LINK_SERVICE_INTERVAL_SECONDS=10" in env
    assert "SQL_LINK_SERVICE_SKIP_PROMOTION_IDLE_SHARDS=1" in env


def test_uniform_process_adaptive_convergence_advances_after_proven_drain_progress(tmp_path: Path) -> None:
    health = tmp_path / "governance" / "health"
    _write_json(
        health / "ingestion_storage_control_latest.json",
        {
            "overall_status": "ready",
            "severity": "stable",
            "pressure_index": 0.612,
            "backlog_truth": {
                "raw_live": {
                    "grade": "A+",
                    "core_pending_lines": 439,
                    "total_pending_lines": 439,
                    "oldest_pending_age_seconds": 143.852,
                },
                "sql_overlay": {
                    "grade": "A+",
                    "pressure_ratio": 0.612,
                    "core_pending_lines": 9178,
                    "total_pending_lines": 9178,
                    "oldest_pending_age_seconds": 61.164,
                    "used_for_pressure": True,
                },
                "truth_gap": {"pending_line_delta": 8739},
            },
            "backlog_relief_contract": {
                "active_issue_ids": [
                    "sparse_huge_jsonl_files",
                    "raw_live_expansion_headroom",
                ]
            },
            "stale_pending_locator": {
                "oldest_sources": [
                    {
                        "source_rel": "governance/channels/decision/default_crypto_schwab/decision_20260613.jsonl",
                        "shard": "crypto_trading",
                        "pressure_lane": "core",
                        "pending_lines": 8002,
                        "oldest_pending_age_seconds": 61.164,
                    }
                ]
            },
        },
    )
    _write_json(
        health / "backlog_pcore_accelerator_latest.json",
        {
            "overall_status": "advisory",
            "host_lane_contract": {"selected_p_core_preprocess_workers": 2, "memory_status": "soft_guard"},
            "storage_accelerator_contract": {
                "p_core_preprocess_workers": 3,
                "max_shard_writer_lanes": 8,
                "catch_up_wave_controller": {
                    "enabled": True,
                    "max_waves": 5,
                    "max_seconds_per_writer_cycle": 120,
                },
            },
            "single_writer_tuning_contract": {"hot_batch_size": 120000, "queue_batch_size": 120000},
        },
    )
    _write_json(
        health / "backlog_pump_infrabots_latest.json",
        {
            "overall_status": "advisory",
            "bots": {
                "shard_hotness_router_bot": {
                    "control_env": {"SQL_LINK_SERVICE_HOT_SHARD_PRIORITY": "crypto_trading"},
                    "focused_sources": [
                        {
                            "source_rel": "governance/channels/decision/default_crypto_schwab/decision_20260613.jsonl",
                            "shard": "crypto_trading",
                            "pressure_lane": "core",
                            "pending_lines": 8002,
                            "oldest_pending_age_seconds": 61.164,
                        }
                    ],
                },
                "catch_up_wave_budget_bot": {
                    "control_env": {
                        "WRITER_CYCLE_MAX_CATCH_UP_WAVES": "5",
                        "SQL_LINK_SERVICE_MERGE_MAX_SECONDS_PER_CYCLE": "120",
                    }
                },
            },
        },
    )
    _write_json(health / "writer_cycle_coordinator_latest.json", {"overall_status": "waiting_for_writer"})
    _write_json(
        health / "writer_process_intelligence_latest.json",
        {
            "overall_status": "ready",
            "writer_health": {
                "active": True,
                "current_step": "merge_primary",
                "progress_age_minutes": 0.1,
                "shard_writer_lane_contract": {
                    "selected_shard_writer_lanes": 2,
                    "max_shard_writer_lanes": 2,
                    "single_primary_merge_writer": True,
                },
            },
        },
    )
    _write_json(
        health / "memory_pressure_intelligence_latest.json",
        {
            "overall_status": "advisory",
            "classification": {"status": "soft_guard"},
            "snapshot": {
                "pressure_level": "normal",
                "pressure_kind": "normal",
                "swap_used_gb": 2.0,
                "compressed_pressure_gb": 1.0,
                "pages_throttled": 0,
            },
            "workload_guidance": {"p_core_preprocess_worker_cap": 4},
        },
    )
    _write_json(
        health / "storage_backpressure_autopilot_latest.json",
        {
            "overall_status": "applied",
            "ok": True,
            "attempts": [
                {"name": "backpressure_slo_bot", "status": "ok", "rc": 0, "timed_out": False},
                {"name": "backpressure_drainer_fleet", "status": "ok", "rc": 0, "timed_out": False},
                {"name": "writer_cycle_coordinator", "status": "ok", "rc": 0, "timed_out": False},
                {"name": "raw_training_manifest_refresh", "status": "deferred", "rc": 0, "timed_out": False},
            ],
            "clearance_state": {"steady_state_ready": False},
            "cycle_records": [
                {
                    "cycle_index": 1,
                    "progress": {
                        "progress_observed": True,
                        "pending_lines_reduced": 1469,
                        "before": {"total_pending_lines": 10647},
                        "after": {"total_pending_lines": 9178},
                    },
                    "clearance_before": {"total_pending_lines": 10647},
                    "clearance_after": {"total_pending_lines": 9178},
                }
            ],
            "metrics": {"storage_plane_phase": "bounded_raw_compaction"},
            "previews": {"storage_plane": {"phase": "bounded_raw_compaction"}},
        },
    )

    payload = uniform.build_payload(tmp_path)
    contract = payload["speed_contract"]
    convergence = contract["adaptive_convergence_contract"]

    assert contract["mode"] == "adaptive_convergence_single_writer_catchup"
    assert convergence["enabled"] is True
    assert convergence["progress"]["pending_lines_reduced"] == 1469
    assert convergence["progress"]["reduction_ratio"] > 0.13
    assert contract["turbo_contract"]["convergence_enabled"] is True
    assert contract["turbo_contract"]["shard_scope"] == ["health_fast", "writer_progress", "hot_path_storage", "crypto_trading"]
    assert contract["hot_batch_size"] == 480000
    assert contract["queue_batch_size"] == 380000
    assert contract["max_seconds_per_cycle"] == 240
    assert contract["poll_seconds"] == 5
    assert contract["wait_timeout_seconds"] == 210
    assert contract["storage_autopilot_cycles"] == 3
    assert contract["single_writer_only"] is True
    assert contract["adds_parallel_sqlite_writers"] is False

    env = "\n".join(uniform._env_lines(payload))
    assert "BACKLOG_DRAIN_ADAPTIVE_CONVERGENCE_ENABLED=1" in env
    assert "BACKLOG_DRAIN_ADAPTIVE_CONVERGENCE_TIER=adaptive_convergence_single_writer_catchup" in env
    assert "SQL_LINK_SERVICE_SHARDS=health_fast,writer_progress,hot_path_storage,crypto_trading" in env
    assert "SQL_LINK_SERVICE_HOT_BATCH_SIZE=480000" in env
    assert "SQL_LINK_SERVICE_QUEUE_BATCH_SIZE=380000" in env
    assert "SQL_LINK_SERVICE_INTERVAL_SECONDS=8" in env
    assert "SQL_LINK_SERVICE_IDLE_SHARD_MAX_AGE_SECONDS=45" in env


def test_uniform_process_sparse_pressure_finalizer_handles_last_pressure_breach(tmp_path: Path) -> None:
    health = tmp_path / "governance" / "health"
    sparse_env = {
        "INGEST_MAX_BYTES_PER_FILE": str(128 * 1024 * 1024),
        "SQLITE_BATCH_MAX_BYTES": str(32 * 1024 * 1024),
        "INGEST_TOP_PENDING_FILES": "24",
    }
    pcore_env = {
        "BACKLOG_PCORE_ALLOCATION_ACTIVE": "1",
        "BACKLOG_PCORE_PREPROCESS_WORKERS": "6",
        "BACKLOG_PCORE_USER_APP_RESERVE_TARGET": "2",
        "BACKLOG_PCORE_BURST_MODE": "full_p_core_budget_7_plus_primary_writer",
        "SQL_LINK_SERVICE_PREPROCESS_WORKERS": "6",
        "SQL_LINK_SERVICE_SHARD_WRITER_LANES": "6",
        "SQL_LINK_SERVICE_MAX_SHARD_WRITER_LANES": "8",
        "SQL_LINK_CHILD_WRITER_CPU_POLICY": "performance_core_primary",
        "SQL_LINK_WRITER_BACKGROUND_POLICY": "0",
        "SQL_LINK_WRITER_NICE": "0",
    }
    _write_json(
        health / "ingestion_storage_control_latest.json",
        {
            "overall_status": "ready",
            "severity": "stable",
            "pressure_index": 0.599,
            "steady_state": {
                "target_status": {
                    "pressure_index_ok": False,
                    "core_pending_lines_ok": True,
                    "estimated_total_drain_minutes_ok": True,
                    "stale_stage_pending_lines_ok": True,
                    "retention_debt_gb_ok": True,
                    "steady_state_ready": False,
                    "target_breach_count": 1,
                    "target_breaches": ["pressure_index"],
                }
            },
            "backlog_truth": {
                "raw_live": {
                    "grade": "A+",
                    "core_pending_lines": 439,
                    "total_pending_lines": 439,
                    "oldest_pending_age_seconds": 143.852,
                },
                "sql_overlay": {
                    "grade": "A+",
                    "pressure_ratio": 0.263,
                    "core_pending_lines": 3948,
                    "total_pending_lines": 3948,
                    "oldest_pending_age_seconds": 27.275,
                    "used_for_pressure": True,
                },
                "truth_gap": {"pending_line_delta": 3509},
            },
            "backpressure": {
                "core_pending_lines": 3948,
                "total_pending_lines": 3948,
                "overlay_adjusted": True,
                "overlay_pressure_clear": True,
                "pending_lines_threshold": 15000,
                "oldest_pending_age_seconds": 143.852,
                "estimated_total_drain_minutes": 15.0,
                "raw_live": {
                    "core_pending_lines": 439,
                    "total_pending_lines": 439,
                    "oldest_pending_age_seconds": 143.852,
                    "line_estimation": {
                        "sparse_large_line_active": True,
                        "sparse_large_line_files": 1,
                        "sparse_large_line_pending_lines": 349,
                        "sparse_large_line_bytes": 14736788037,
                        "sparse_large_line_pending_bytes": 323325518,
                        "sparse_large_line_policy": "multi_sample_density_then_sparse_window_floor",
                    },
                },
            },
            "storage_plane_contract": {
                "disk_contract": {
                    "external_available_gb": 68.5,
                }
            },
            "storage_efficiency_contract": {
                "metrics": {
                    "safe_space_recovery_target_free_gb": 64.0,
                }
            },
            "backlog_relief_contract": {
                "active": True,
                "active_issue_ids": ["sparse_huge_jsonl_files"],
                "control_env_recommendations": {**sparse_env, **pcore_env},
                "p_core_backlog_allocation_contract": {
                    "policy": "p_core_preprocess_single_sql_writer",
                    "sqlite_writer_count": 1,
                    "control_env": pcore_env,
                },
                "issues": [
                    {
                        "id": "sparse_huge_jsonl_files",
                        "active": True,
                        "grade": "F",
                        "pressure_ratio": 4.818,
                        "evidence": {
                            "sparse_large_line_detected": True,
                            "sparse_large_line_files": 1,
                            "sparse_large_line_pending_lines": 349,
                            "sparse_large_line_pending_bytes": 323325518,
                        },
                        "next_action": "drain sparse JSONL files by byte windows and payload-byte SQLite batch caps",
                        "control_env": sparse_env,
                    }
                ],
            },
            "stale_pending_locator": {
                "oldest_sources": [
                    {
                        "source_rel": "governance/channels/decision/default_crypto_schwab/decision_20260613.jsonl",
                        "shard": "crypto_trading",
                        "pressure_lane": "core",
                        "pending_lines": 3948,
                        "oldest_pending_age_seconds": 27.275,
                    }
                ]
            },
        },
    )
    _write_json(
        health / "backlog_pcore_accelerator_latest.json",
        {
            "overall_status": "advisory",
            "host_lane_contract": {"selected_p_core_preprocess_workers": 2, "memory_status": "soft_guard"},
            "storage_accelerator_contract": {
                "p_core_preprocess_workers": 6,
                "max_shard_writer_lanes": 8,
                "catch_up_wave_controller": {
                    "enabled": True,
                    "max_waves": 6,
                    "max_seconds_per_writer_cycle": 150,
                },
            },
            "single_writer_tuning_contract": {"hot_batch_size": 120000, "queue_batch_size": 120000},
        },
    )
    _write_json(
        health / "backlog_pump_infrabots_latest.json",
        {
            "overall_status": "advisory",
            "bots": {
                "shard_hotness_router_bot": {
                    "control_env": {"SQL_LINK_SERVICE_HOT_SHARD_PRIORITY": "crypto_trading"},
                    "focused_sources": [
                        {
                            "source_rel": "governance/channels/decision/default_crypto_schwab/decision_20260613.jsonl",
                            "shard": "crypto_trading",
                            "pressure_lane": "core",
                            "pending_lines": 3948,
                            "oldest_pending_age_seconds": 27.275,
                        }
                    ],
                },
                "catch_up_wave_budget_bot": {
                    "control_env": {
                        "WRITER_CYCLE_MAX_CATCH_UP_WAVES": "6",
                        "SQL_LINK_SERVICE_MERGE_MAX_SECONDS_PER_CYCLE": "150",
                    }
                },
            },
        },
    )
    _write_json(health / "writer_cycle_coordinator_latest.json", {"overall_status": "waiting_for_writer"})
    _write_json(
        health / "writer_process_intelligence_latest.json",
        {
            "overall_status": "ready",
            "writer_health": {
                "active": True,
                "current_step": "shard_linking",
                "progress_age_minutes": 0.1,
                "shard_writer_lane_contract": {
                    "selected_shard_writer_lanes": 3,
                    "max_shard_writer_lanes": 8,
                    "single_primary_merge_writer": True,
                },
            },
        },
    )
    _write_json(
        health / "memory_pressure_intelligence_latest.json",
        {
            "overall_status": "advisory",
            "classification": {"status": "soft_guard"},
            "snapshot": {
                "pressure_level": "normal",
                "pressure_kind": "normal",
                "swap_used_gb": 1.5,
                "compressed_pressure_gb": 1.0,
                "pages_throttled": 0,
            },
            "workload_guidance": {"p_core_preprocess_worker_cap": 6},
        },
    )
    _write_json(
        health / "storage_backpressure_autopilot_latest.json",
        {
            "overall_status": "applied",
            "ok": True,
            "attempts": [
                {"name": "backpressure_slo_bot", "status": "ok", "rc": 0, "timed_out": False},
                {"name": "writer_cycle_coordinator", "status": "ok", "rc": 0, "timed_out": False},
            ],
            "clearance_state": {"steady_state_ready": False},
            "cycle_records": [
                {
                    "cycle_index": 1,
                    "progress": {
                        "progress_observed": True,
                        "pending_lines_reduced": 246,
                        "before": {"total_pending_lines": 4400},
                        "after": {"total_pending_lines": 4154},
                    },
                }
            ],
            "metrics": {"storage_plane_phase": "deep_cold_managed_steady_state"},
            "previews": {"storage_plane": {"phase": "deep_cold_managed_steady_state"}},
        },
    )

    payload = uniform.build_payload(tmp_path)
    contract = payload["speed_contract"]
    finalizer = contract["sparse_pressure_finalizer_contract"]

    assert contract["mode"] == "sparse_pressure_finalizer_single_writer"
    assert finalizer["enabled"] is True
    assert finalizer["context"]["pending_bytes"] == 323325518
    assert finalizer["storage_reserve"]["margin_gb"] == 4.5
    assert finalizer["progress"]["enough_progress"] is True
    assert finalizer["target_breaches"] == ["pressure_index"]
    assert contract["turbo_contract"]["sparse_finalizer_enabled"] is True
    assert contract["turbo_contract"]["shard_scope"] == ["health_fast", "writer_progress", "hot_path_storage", "crypto_trading"]
    assert contract["hot_batch_size"] == 360000
    assert contract["queue_batch_size"] == 260000
    assert contract["max_seconds_per_cycle"] == 180
    assert contract["poll_seconds"] == 7
    assert contract["wait_timeout_seconds"] == 240
    assert contract["storage_autopilot_cycles"] == 2
    assert contract["single_writer_only"] is True
    assert contract["adds_parallel_sqlite_writers"] is False

    env = uniform.env_dict(payload)
    assert env["BACKLOG_DRAIN_SPARSE_PRESSURE_FINALIZER_ENABLED"] == "1"
    assert env["INGEST_MAX_BYTES_PER_FILE"] == str(128 * 1024 * 1024)
    assert env["SQLITE_BATCH_MAX_BYTES"] == str(32 * 1024 * 1024)
    assert env["INGEST_TOP_PENDING_FILES"] == "24"
    assert env["SQL_LINK_SERVICE_SPARSE_LARGE_DECISION_DRAIN"] == "1"
    assert env["SQL_LINK_SERVICE_SPARSE_LARGE_DECISION_FILE_COUNT"] == "1"
    assert env["SQL_LINK_SERVICE_SPARSE_LARGE_DECISION_SHARDS"] == "crypto_trading"
    assert env["SQL_LINK_SERVICE_PREPROCESS_WORKERS"] == "6"
    assert env["SQL_LINK_SERVICE_SHARD_WRITER_LANES"] == "6"
    assert env["SQL_LINK_SERVICE_SINGLE_WRITER_ONLY"] == "1"

    storage_path = health / "ingestion_storage_control_latest.json"
    storage_payload = json.loads(storage_path.read_text(encoding="utf-8"))
    storage_payload["storage_plane_contract"]["disk_contract"]["external_available_gb"] = 64.5
    _write_json(storage_path, storage_payload)

    held_payload = uniform.build_payload(tmp_path)
    held_contract = held_payload["speed_contract"]
    held_finalizer = held_contract["sparse_pressure_finalizer_contract"]

    assert held_contract["mode"] == "steady_uniform_drain"
    assert held_finalizer["enabled"] is False
    assert "storage_reserve_margin_too_thin" in held_finalizer["blockers"]
    assert held_contract["turbo_contract"]["enabled"] is False
    assert "sparse_finalizer_safety_hold" in held_contract["turbo_contract"]["blockers"]
