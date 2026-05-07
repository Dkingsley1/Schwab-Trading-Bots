import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.ops import health_fast, pressure_relief_control


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def test_pressure_relief_enables_twenty_eight_pressure_controls(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    health_root = project_root / "governance" / "health"
    _write_json(
        health_root / "runtime_throttle_control_latest.json",
        {
            "host_saturation_score": 64.0,
            "compute_pressure_level": "elevated",
            "memory_pressure_level": "elevated",
        },
    )
    _write_json(
        health_root / "memory_efficiency_control_latest.json",
        {"co_running_session": {"level": "heavy_competition"}},
    )
    _write_json(
        health_root / "swap_pressure_governor_latest.json",
        {"swap_pressure": {"tier": "normal", "swap_used_gb": 0.5}},
    )
    _write_json(health_root / "global_halt_auto_clear_latest.json", {"halt": False})
    _write_json(health_root / "ingestion_storage_control_latest.json", {"severity": "stable", "pressure_index": 0.02})

    payload = pressure_relief_control.build_payload(project_root)
    env = payload["recommended_env_overrides"]

    assert payload["tier"] == "guarded_relief"
    assert len(payload["pressure_relief_items"]) == 28
    assert env["OPS_HEALTH_FAST_ENABLED"] == "1"
    assert env["LIVE_FEED_HEAVY_TTL_ENABLED"] == "1"
    assert env["MAINTENANCE_SLOT_DEFER_OUTSIDE_QUIET_WINDOW"] == "1"
    assert env["SQL_LINK_SERVICE_ADAPTIVE_WRITER_ENABLED"] == "1"
    assert env["MLX_LAZY_IMPORTS"] == "1"
    assert env["HEALTH_ARTIFACT_COALESCE_ENABLED"] == "1"
    assert env["BOT_COLLECTION_DUTY_CYCLE_ENABLED"] == "1"
    assert env["PAPER_TRADE_EVENT_QUEUE_JITTER_ENABLED"] == "1"
    assert env["TRAINING_RESEARCH_PAUSE_ON_PRESSURE"] == "1"


def test_pressure_relief_keeps_concentrated_sql_drain_fast_under_deep_relief(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    health_root = project_root / "governance" / "health"
    _write_json(
        health_root / "runtime_throttle_control_latest.json",
        {
            "host_saturation_score": 91.0,
            "compute_pressure_level": "high",
            "memory_pressure_level": "elevated",
        },
    )
    _write_json(health_root / "memory_efficiency_control_latest.json", {"co_running_session": {"level": "heavy_competition"}})
    _write_json(health_root / "swap_pressure_governor_latest.json", {"swap_pressure": {"tier": "normal", "swap_used_gb": 0.5}})
    _write_json(health_root / "global_halt_auto_clear_latest.json", {"halt": False})
    _write_json(health_root / "ingestion_storage_control_latest.json", {"severity": "high", "pressure_index": 2.2})
    _write_json(
        health_root / "backpressure_drainer_fleet_latest.json",
        {
            "active_drainer": {
                "name": "core_decision_drainer",
                "concentration": {
                    "total_pending_lines": 33623,
                    "top1_share": 0.544806,
                    "top3_share": 0.92871,
                    "concentrated": True,
                },
            },
            "service_request": {"env_overrides": {"SQL_LINK_SERVICE_CONCENTRATED_CORE_DRAIN": "1"}},
        },
    )

    payload = pressure_relief_control.build_payload(project_root)
    env = payload["recommended_env_overrides"]

    assert payload["tier"] == "deep_relief"
    assert payload["storage_pressure"]["sql_writer_coordination"]["concentrated_core_drain"] is True
    assert env["SQL_LINK_SERVICE_INTERVAL_SECONDS"] == "12"
    assert env["SQL_LINK_SERVICE_SHARD_LINK_TIMEOUT_SECONDS"] == "420"
    assert env["SQL_LINK_SERVICE_MERGE_MAX_SECONDS_PER_CYCLE"] == "60"
    assert env["SQL_LINK_SERVICE_SHARD_AGGRESSIVE_TRADING_MAX_LINES_PER_FILE"] == "12000"


def test_health_fast_is_read_only_and_ready_from_latest_files(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    health_root = project_root / "governance" / "health"
    _write_json(health_root / "process_watchdog_latest.json", {"alerts": [], "safety_pause": {"active": False}, "status": []})
    _write_json(health_root / "runtime_throttle_control_latest.json", {"overall_status": "ready", "host_saturation_score": 12})
    _write_json(health_root / "memory_efficiency_control_latest.json", {"overall_status": "ready", "recommended_profile": "max_throughput"})
    _write_json(health_root / "swap_pressure_governor_latest.json", {"swap_pressure": {"tier": "normal", "swap_used_gb": 0.0}})
    _write_json(health_root / "pressure_relief_control_latest.json", {"tier": "observe"})
    _write_json(health_root / "data_collection_observation_rollup_latest.json", {"overall_status": "ready", "collector_count": 2, "bots_with_observations": 2})
    _write_json(health_root / "global_halt_auto_clear_latest.json", {"halt": False, "halt_state": "clear_ready", "clear_blockers": []})
    _write_json(health_root / "ingestion_storage_control_latest.json", {"severity": "stable", "pressure_index": 0.01, "backpressure": {}})

    payload = health_fast.build_payload(project_root)

    assert payload["ok"] is True
    assert payload["read_only"] is True
    assert payload["started_heavy_reports"] is False
    assert payload["collection"]["bots_with_observations"] == 2
