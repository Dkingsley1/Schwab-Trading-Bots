import json
from pathlib import Path

from scripts.ops import platform_settlement_stabilization as src


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def _seed_project(project_root: Path) -> None:
    _write_json(
        project_root / "master_bot_registry.json",
        {
            "sub_bots": [
                {"bot_id": "collector_one", "active": True, "data_collection_active": True, "training_excluded": True, "paper_trade_active": True},
                {"bot_id": "collector_two", "active": True, "data_collection_active": True, "training_excluded": True},
            ]
        },
    )
    _write_json(
        project_root / "governance" / "health" / "ingestion_storage_control_latest.json",
        {
            "severity": "high",
            "pressure_index": 2.0,
            "throughput": {"merged_rows_this_cycle": 12000, "throughput_rows_per_second": 80.0},
            "bounded_recovery_contract": {"active_drain_progress": True, "drain_delta_total_lines": -200},
            "backpressure": {
                "core_pending_lines": 25000,
                "deferred_pending_lines": 20,
                "cold_pending_lines": 0,
                "support_pending_lines": 5,
                "total_pending_lines": 25025,
                "pending_lines_threshold": 15000,
                "estimated_total_drain_minutes": 5.2,
            },
        },
    )
    _write_json(project_root / "governance" / "health" / "ingestion_backpressure_latest.json", {"pending_lines": 24000, "pending_lines_total": 24025})
    _write_json(
        project_root / "governance" / "health" / "process_watchdog_latest.json",
        {"status": [{"name": "sql_link_writer", "running": 1, "heartbeat_ok": True}]},
    )
    _write_json(
        project_root / "governance" / "health" / "backpressure_drainer_fleet_latest.json",
        {"writer_active": False, "writer_lock_held": False, "ready_drainer_count": 1},
    )
    _write_json(
        project_root / "governance" / "health" / "runtime_throttle_control_latest.json",
        {
            "overall_status": "blocked",
            "host_saturation_score": 84.0,
            "compute_pressure_level": "high",
            "memory_pressure_level": "elevated",
            "cotenant_awareness_contract": {"co_running_level": "heavy_competition", "open_apps": ["PyCharm", "Safari"]},
        },
    )
    _write_json(project_root / "governance" / "health" / "pressure_relief_control_latest.json", {"host_saturation_score": 70.0})
    _write_json(
        project_root / "governance" / "health" / "global_halt_auto_clear_latest.json",
        {"halt": False, "halt_state": "clear_blocked", "clear_ready": False, "clear_blockers": ["queue_backpressure_active"]},
    )
    _write_json(project_root / "governance" / "health" / "data_collection_observation_rollup_latest.json", {"collector_count": 2, "bots_with_observations": 2, "zero_observation_count": 0})
    _write_json(project_root / "governance" / "health" / "paper_400_ramp_control_latest.json", {"overall_status": "ready"})
    _write_json(
        project_root / "governance" / "health" / "external_backlog_drain_latest.json",
        {
            "overall_status": "blocked",
            "material_drain_recommended": True,
            "blocked_reasons": ["market_hours_guard"],
            "storage_mode": "external",
            "off_hours_window": {"active": False, "label": "market_hours"},
        },
    )


def test_platform_settlement_stabilization_builds_settlement_controls(tmp_path: Path) -> None:
    _seed_project(tmp_path)

    payload = src.build_payload(tmp_path)

    assert payload["section_count"] == 7
    assert set(payload["section_keys"]) == set(src.SECTION_KEYS)
    assert payload["control_count"] == 7
    assert payload["sections"]["queue_decay_meter"]["queue_backpressure_active"] is True
    assert payload["sections"]["queue_decay_meter"]["progress_observed"] is True
    assert payload["sections"]["single_writer_guard"]["sql_link_writer_running_count"] == 1
    assert payload["sections"]["global_clear_settlement_guard"]["overall_status"] == "watch"
    assert payload["recommended_env_overrides"]["PLATFORM_SETTLEMENT_STABILIZATION_ENABLED"] == "1"
    assert payload["recommended_env_overrides"]["GLOBAL_HALT_NOTIFY_ON_SELF_CLEAR"] == "1"
    assert payload["recommended_env_overrides"]["EXPANSION_APPLY_ALLOWED"] == "0"


def test_single_writer_guard_treats_launchd_wrapper_chain_as_one_writer(tmp_path: Path) -> None:
    _seed_project(tmp_path)
    _write_json(
        tmp_path / "governance" / "health" / "process_watchdog_latest.json",
        {"status": [{"name": "sql_link_writer", "running": 2, "heartbeat_ok": True}]},
    )

    payload = src.build_payload(tmp_path)
    guard = payload["sections"]["single_writer_guard"]

    assert guard["overall_status"] == "ready"
    assert guard["raw_sql_link_writer_running_count"] == 2
    assert guard["sql_link_writer_running_count"] == 1
    assert guard["wrapper_chain_only"] is True


def test_platform_settlement_stabilization_writes_artifacts(tmp_path: Path) -> None:
    _seed_project(tmp_path)

    payload = src.build_payload(tmp_path)
    written = src.write_section_artifacts(tmp_path, payload)

    assert len(written) == 8
    assert all(Path(path).exists() for path in written.values())
    assert "infrabot_assignments" in written


def test_platform_settlement_stabilization_uses_previous_memory(tmp_path: Path) -> None:
    _seed_project(tmp_path)
    _write_json(
        tmp_path / "governance" / "health" / "platform_settlement_stabilization_latest.json",
        {
            "sections": {
                "queue_decay_meter": {"metrics": {"total_pending_lines": 26000}},
                "market_hours_cadence_smoother": {"host_saturation_score": 90.0},
            }
        },
    )

    payload = src.build_payload(tmp_path)
    memory = payload["sections"]["stabilization_effectiveness_memory"]

    assert memory["previous_artifact_seen"] is True
    assert memory["queue_delta_total_lines"] == -975
    assert memory["host_saturation_delta"] == -6.0
    assert memory["improving"] is True
