import json
from pathlib import Path

from scripts.ops import feature_maturity_control as src


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def _seed_mature_project(project_root: Path) -> None:
    health = project_root / "governance" / "health"
    _write_json(
        health / "runtime_throttle_control_latest.json",
        {
            "timestamp_utc": "2026-06-19T03:00:00+00:00",
            "overall_status": "advisory",
            "host_saturation_score": 44.2,
            "paper_execution_policy": {
                "paper_execution_allowed": True,
                "pause_paper_execution": False,
                "armed": True,
                "stage": "armed",
                "reason": "paper_ramp_armed_and_clean",
            },
            "paper_capacity_contract": {
                "ready_for_700_bot_paper": True,
                "active_bot_count": 1608,
                "paper_tagged_count": 401,
                "runtime_policy": {"live_execution_blocked": True},
            },
            "release_contract": {
                "live_lane_should_be_read_only": True,
                "release_live_lane_should_be_read_only": True,
                "paper_trade_lock_active": True,
                "effective_live_read_only_reason": "release_contract",
            },
            "controller_contract": {"mode": "apply_capable", "safe_while_live": True},
            "host_pressure_attribution": {"support_trim_required": True, "support_jobs_hot": True},
        },
    )
    _write_json(
        health / "ingestion_storage_control_latest.json",
        {
            "overall_status": "ready",
            "severity": "stable",
            "pressure_index": 0.08,
            "backpressure": {
                "total_pending_lines": 722,
                "pending_lines_threshold": 15000,
                "oldest_pending_age_seconds": 20.0,
            },
        },
    )
    _write_json(
        health / "sql_link_service_progress_latest.json",
        {
            "timestamp_utc": "2026-06-19T03:00:00+00:00",
            "status": "ok",
            "current_step": "complete",
            "completed_shard_count": 17,
            "planned_shard_count": 17,
            "timed_out_shard_count": 0,
            "merged_rows_this_cycle": 3434,
            "shard_writer_lane_contract": {
                "smart_shard_parallelism": {
                    "enabled": True,
                    "enforced_single_primary_merge_writer": True,
                    "policy": "tier_capped_hot_first_parallel_child_shards_single_primary_merge",
                }
            },
        },
    )
    _write_json(
        health / "training_runtime_control_latest.json",
        {
            "timestamp_utc": "2026-06-19T03:00:00+00:00",
            "overall_status": "constrained",
            "snapshot_ready": True,
            "snapshot_age_minutes": 30.0,
            "runtime_backend_parity": {
                "parity_state": "ready",
                "runtime_matches_current": True,
                "runtime_python_version": "3.14.5",
            },
            "training_launch_contract": {
                "mode": "prep_only",
                "prep_allowed": True,
                "prep_blockers": [],
                "launch_allowed": False,
            },
            "training_quality": {"overall_status": "needs_attention", "training_quality_score": 100.0},
        },
    )
    _write_json(health / "training_quality_control_latest.json", {"overall_status": "needs_attention", "training_quality_score": 100.0})
    _write_json(health / "bot_quality_autopilot_latest.json", {"overall_status": "ready"})
    _write_json(health / "bot_needs_intelligence_latest.json", {"overall_status": "needs_action", "need_counts": {"collect_more_data": 2}})
    _write_json(health / "system_drift_guard_latest.json", {"overall_status": "ready", "recommended_commands": [["./scripts/ops/opsctl.sh", "system-drift", "--json"]]})
    _write_json(health / "system_drift_registry_latest.json", {"overall_status": "ready"})
    _write_json(health / "watchdog_intelligence_latest.json", {"overall_status": "ready", "recommended_commands": [["./scripts/ops/opsctl.sh", "watchdog-intelligence", "--json"]]})
    _write_json(health / "decision_provenance_cards_latest.json", {"overall_status": "ready", "card_count": 12})
    _write_json(health / "governance_telemetry_compactor_latest.json", {"overall_status": "ready"})
    _write_json(health / "evidence_packet_latest.json", {"overall_status": "ready"})
    _write_json(health / "live_runtime_separation_control_latest.json", {"overall_status": "ready"})
    _write_json(health / "broker_readiness_latest.json", {"ready_for_open": True})
    _write_json(health / "source_verification_latest.json", {"overall_status": "ready"})
    _write_json(health / "data_collection_observation_rollup_latest.json", {"bots_with_observations": 100, "collector_count": 100})
    _write_json(health / "point_in_time_event_store_latest.json", {"overall_status": "ready"})
    _write_json(health / "platform_stabilization_quality_latest.json", {"overall_status": "ready"})


def test_feature_maturity_control_levels_mature_project_to_target(tmp_path: Path) -> None:
    _seed_mature_project(tmp_path)

    payload = src.build_payload(tmp_path)

    assert payload["target_level"] == 4
    assert payload["features_below_target"] == 0
    assert payload["overall_status"] == "ready"
    live = next(row for row in payload["features"] if row["slug"] == "live_execution_safety")
    assert live["maturity_level"] >= 4
    assert live["safety_mode"] == "mature_means_locked_until_release"
    assert payload["recommended_env_overrides"]["ALLOW_ORDER_EXECUTION"] == "0"


def test_feature_maturity_control_surfaces_missing_drift_and_governance(tmp_path: Path) -> None:
    _seed_mature_project(tmp_path)
    for name in (
        "system_drift_guard_latest.json",
        "system_drift_registry_latest.json",
        "watchdog_intelligence_latest.json",
        "decision_provenance_cards_latest.json",
        "governance_telemetry_compactor_latest.json",
        "evidence_packet_latest.json",
    ):
        (tmp_path / "governance" / "health" / name).unlink()

    payload = src.build_payload(tmp_path)

    assert payload["overall_status"] in {"needs_work", "blocked"}
    assert "drift_anomaly_monitoring" in payload["below_target_slugs"]
    assert "decision_explainability_governance" in payload["below_target_slugs"]
    assert payload["recommended_env_overrides"]["DRIFT_MONITORING_REQUIRED"] == "1"
    assert payload["recommended_env_overrides"]["DECISION_EXPLAINABILITY_REQUIRED"] == "1"


def test_feature_maturity_control_apply_writes_safe_override(tmp_path: Path) -> None:
    _seed_mature_project(tmp_path)
    payload = src.build_payload(tmp_path)
    override_path = tmp_path / "config" / ".env.feature_maturity_control_override"

    changed = src._write_env_override(override_path, payload["recommended_env_overrides"])

    assert changed is True
    text = override_path.read_text(encoding="utf-8")
    assert "FEATURE_MATURITY_CONTROL_ENABLED=1" in text
    assert "LIVE_EXECUTION_RELEASE_GATE_REQUIRED=1" in text
    assert "ALLOW_ORDER_EXECUTION=0" in text
