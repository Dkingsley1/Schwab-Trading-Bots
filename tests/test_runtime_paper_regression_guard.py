import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from scripts.ops import infrastructure_autofix_bot as infra_src
from scripts.ops import runtime_paper_regression_guard as src


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def _write_env(path: Path, values: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(f"{key}={value}" for key, value in values.items()) + "\n", encoding="utf-8")


def _runtime_payload(*, blocked_paper: bool = True) -> dict:
    return {
        "overall_status": "ready",
        "ok": True,
        "throttle_profile": "sustain",
        "compute_pressure_level": "elevated",
        "memory_pressure_level": "normal",
        "soft_cap_advisory_reclassification": {
            "active": True,
            "from_status": "degraded",
            "to_status": "ready",
            "reason": "external_cotenant_with_bounded_storage_overlay_is_guarded_runtime_ready",
            "thresholds": {
                "max_guarded_ready_bot_owned_cpu_percent": 20.0,
                "max_guarded_ready_protected_cpu_percent": 20.0,
                "max_guarded_ready_operator_cpu_percent": 30.0,
            },
            "measurements": {
                "runtime_ready_guarded": True,
                "memory_pressure_level": "normal",
                "bot_owned_cpu_percent": 7.0,
                "protected_live_or_macro_cpu_percent": 0.0,
                "operator_observability_cpu_percent": 3.0,
                "support_jobs_hot": False,
                "paper_execution_hot": False,
                "storage_writer_hot": False,
                "bot_owned_pressure_dominant": False,
            },
        },
        "paper_execution_policy": {
            "artifact_present": True,
            "paper_execution_allowed": not blocked_paper,
            "pause_paper_execution": blocked_paper,
            "reason": "paper_ramp_blocked" if blocked_paper else "paper_ramp_armed",
            "stage": "blocked" if blocked_paper else "armed",
            "armed": not blocked_paper,
            "ok": not blocked_paper,
            "blockers": ["paper_roster_below_400_target"] if blocked_paper else [],
        },
        "runtime_saturation_governor_v2": {
            "active": True,
            "paper_live_data_policy": {
                "paper_execution_allowed": not blocked_paper,
                "paper_execution_consumer_paused": blocked_paper,
            },
        },
    }


def _paper_payload(*, blockers: list[str] | None = None, armed: bool = False) -> dict:
    blockers = blockers if blockers is not None else [
        "global_halt_or_clear_blocker_active",
        "ingestion_or_backpressure_above_paper_400_gate",
        "paper_roster_below_400_target",
    ]
    return {
        "stage": "armed" if armed else "blocked",
        "ok": bool(armed and not blockers),
        "armed": armed,
        "blockers": blockers,
        "gates": {
            "runtime": {
                "status": "blocked" if blockers else "ready",
                "blockers": ["paper_roster_below_400_target"] if "paper_roster_below_400_target" in blockers else [],
                "runtime_pressure_ready": True,
                "runtime_capacity_ready": True,
                "ready_for_700_bot_paper": True,
                "pressure_limited": False,
            }
        },
    }


def test_runtime_guard_accepts_full_force_paper_ready_hot_lane() -> None:
    runtime = _runtime_payload(blocked_paper=False)
    soft_cap = runtime["soft_cap_advisory_reclassification"]
    soft_cap["reason"] = "full_force_paper_ramp_pressure_is_guarded_runtime_ready"
    soft_cap["measurements"].update(
        {
            "full_force_paper_ramp_guarded_ready": True,
            "paper_execution_hot": True,
            "paper_hot_low_priority": True,
            "paper_execution_cpu_percent": 105.0,
            "bot_owned_cpu_percent": 124.3,
            "bot_owned_non_operator_cpu_percent": 113.8,
            "bot_owned_pressure_dominant": True,
            "operator_observability_cpu_percent": 32.2,
            "storage_ready_for_runtime_advisory": True,
        }
    )
    soft_cap["thresholds"].update(
        {
            "max_guarded_ready_full_force_bot_owned_cpu_percent": 340.0,
            "max_guarded_ready_full_force_operator_cpu_percent": 45.0,
        }
    )

    guard = src._runtime_guarded_ready_lane_guard(runtime)

    assert guard["ok"] is True
    assert guard["status"] == "ready"


def _ready_override(path: Path) -> None:
    _write_env(
        path,
        {
            "ALLOW_ORDER_EXECUTION": "0",
            "MARKET_DATA_ONLY": "1",
            "PAPER_TRADE_LOCK": "1",
            "TOP_BOT_ENABLE_LIVE_EXECUTION": "0",
            "EXECUTION_LANE_LIVE_ENABLED": "0",
            "RUN_ALL_SLEEVES_WITH_LIVE_EXECUTOR": "0",
            "PAPER_CRYPTO_FEED_RUNTIME_PAUSED_FOR_PRESSURE": "0",
            "PAPER_EXECUTION_RUNTIME_PAUSED_FOR_PRESSURE": "0",
            "PAPER_EXECUTION_QUEUE_CONSUMER_ENABLED": "1",
            "INLINE_PAPER_EXECUTION_ENABLED": "1",
            "YTDLP_SUPPORT_NICE": "20",
            "MACRO_YTDLP_SUPPORT_NICE": "20",
            "TRAINING_RUNTIME_PAUSED_FOR_HOST_HEADROOM": "1",
            "SHADOW_LOOP_RUNTIME_PAUSE_SLEEP_SECONDS": "60",
        },
    )


def _blocked_override(path: Path) -> None:
    _write_env(
        path,
        {
            "ALLOW_ORDER_EXECUTION": "0",
            "MARKET_DATA_ONLY": "1",
            "PAPER_TRADE_LOCK": "1",
            "TOP_BOT_ENABLE_LIVE_EXECUTION": "0",
            "EXECUTION_LANE_LIVE_ENABLED": "0",
            "RUN_ALL_SLEEVES_WITH_LIVE_EXECUTOR": "0",
            "PAPER_EXECUTION_RUNTIME_PAUSED_FOR_PRESSURE": "1",
            "PAPER_EXECUTION_QUEUE_CONSUMER_ENABLED": "0",
            "INLINE_PAPER_EXECUTION_ENABLED": "0",
            "YTDLP_SUPPORT_NICE": "20",
            "MACRO_YTDLP_SUPPORT_NICE": "20",
            "TRAINING_RUNTIME_PAUSED_FOR_HOST_HEADROOM": "1",
            "SHADOW_LOOP_RUNTIME_PAUSE_SLEEP_SECONDS": "60",
        },
    )


def _write_soak_lane_artifacts(project_root: Path, *, profitability_timestamp_utc: str | None = None) -> None:
    health = project_root / "governance" / "health"
    _write_json(
        health / "auth_lease_manager_latest.json",
        {
            "timestamp_utc": src.iso_now(),
            "ok": True,
            "overall_status": "ready",
            "lease_state": "healthy",
            "lease_budget": {"expires_in_seconds": 1800, "critical_lease_seconds": 600},
        },
    )
    _write_json(
        health / "schwab_auth_supervisor_latest.json",
        {
            "timestamp_utc": src.iso_now(),
            "ok": True,
            "overall_status": "ready",
            "token_ready": True,
            "refresh_needed": False,
        },
    )
    _write_json(
        health / "broker_readiness_latest.json",
        {
            "timestamp_utc": src.iso_now(),
            "ready_for_open": True,
            "auth_ok": True,
            "network_ok": True,
        },
    )
    _write_json(
        health / "session_ready_latest.json",
        {
            "timestamp_utc": src.iso_now(),
            "ok": True,
            "checks": [{"name": "global_halt_not_set", "ok": True}],
        },
    )
    (health / "PAPER_TRADE_LOCK.flag").write_text(
        "live_data_paper_trade_only\nenabled_at_utc=2026-07-12T00:00:00+00:00\n",
        encoding="utf-8",
    )
    _write_json(
        health / "process_watchdog_latest.json",
        {
            "timestamp_utc": src.iso_now(),
            "overall_status": "ready",
            "status": [
                {
                    "name": "all_sleeves",
                    "running": 1,
                    "process_live": True,
                    "heartbeat_ok": True,
                    "launcher_live": True,
                    "child_fanout_ok": True,
                    "child_fanout": {"child_process_count": 20},
                    "launcher_artifact_health": {
                        "phase": "running",
                        "running_job_count": 33,
                        "expected_job_count": 101,
                    },
                }
            ],
        },
    )
    _write_json(
        health / "paper_runtime_profitability_controls_latest.json",
        {
            "timestamp_utc": profitability_timestamp_utc or src.iso_now(),
            "raw_profitability_grade": "D",
            "controlled_profitability_grade": "A+",
            "raw_profitability_a_recovery_contract": {
                "active": True,
                "gap_to_raw_a": {"net_pnl_gap": 100.0},
                "runtime_enforcement": {
                    "block_new_entries_on_weak_profiles": True,
                    "keep_sells_and_reduce_only_paths_open": True,
                    "raise_clean_profile_buy_gate_while_raw_below_a": True,
                    "block_when_source_or_fill_unknown": True,
                },
            },
            "raw_profitability_improvement_contract": {
                "active": True,
                "control_ready": True,
                "raw_grade_remains_evidence_based": True,
                "runtime_enforcement": {
                    "block_new_entries_on_weak_profiles": True,
                    "keep_sells_and_reduce_only_paths_open": True,
                    "raise_clean_profile_buy_gate_while_raw_below_a": True,
                    "require_position_telemetry_on_paper_fills": True,
                    "feed_loss_causes_to_training": True,
                    "require_three_profitable_refreshes_before_reentry": True,
                    "track_raw_gap_burn_down": True,
                },
            },
            "global_runtime_policy": {"apply_raw_profitability_a_recovery": True},
        },
    )


def test_runtime_paper_guard_accepts_blocked_paper_when_runtime_capacity_is_ready(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    health = project_root / "governance" / "health"
    override = project_root / "config" / ".env.runtime_resource_guard_override"
    _write_json(health / "runtime_throttle_control_latest.json", _runtime_payload(blocked_paper=True))
    _write_json(health / "paper_400_ramp_latest.json", _paper_payload())
    _write_soak_lane_artifacts(project_root)
    _blocked_override(override)

    payload = src.build_payload(project_root)

    assert payload["overall_status"] == "ready"
    assert payload["failed_guards"] == []
    assert payload["paper_blocked"] is True
    assert "paper_execution_pause_guard_bot" in payload["assigned_infrabots"]


def test_runtime_paper_guard_accepts_already_advisory_low_pressure_runtime_without_metadata(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    health = project_root / "governance" / "health"
    override = project_root / "config" / ".env.runtime_resource_guard_override"
    runtime = _runtime_payload(blocked_paper=False)
    runtime["overall_status"] = "advisory"
    runtime["throttle_profile"] = "soft_cap"
    runtime["compute_pressure_level"] = "normal"
    runtime["memory_pressure_level"] = "normal"
    runtime.pop("soft_cap_advisory_reclassification", None)
    _write_json(health / "runtime_throttle_control_latest.json", runtime)
    _write_json(health / "paper_400_ramp_latest.json", _paper_payload(blockers=[], armed=True))
    _write_soak_lane_artifacts(project_root)
    _ready_override(override)

    payload = src.build_payload(project_root)

    assert payload["overall_status"] == "ready"
    assert payload["failed_guards"] == []
    guard = next(row for row in payload["regression_guards"] if row["name"] == "runtime_ready_advisory_reclassification_contract")
    assert guard["actual"]["already_reclassified_low_pressure_advisory"] is True


def test_runtime_paper_guard_accepts_single_bounded_storage_writer_ready(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    health = project_root / "governance" / "health"
    override = project_root / "config" / ".env.runtime_resource_guard_override"
    runtime = _runtime_payload(blocked_paper=False)
    advisory = runtime["soft_cap_advisory_reclassification"]
    advisory["reason"] = "single_bounded_storage_writer_after_green_backpressure_is_guarded_runtime_ready"
    advisory["measurements"].update(
        {
            "storage_ready_for_runtime_advisory": True,
            "storage_writer_cooling_guarded_ready": True,
            "storage_writer_hot": True,
            "storage_writer_cpu_percent": 95.0,
            "bot_owned_pressure_dominant": True,
            "bot_owned_cpu_percent": 121.0,
            "protected_live_or_macro_cpu_percent": 0.0,
            "operator_observability_cpu_percent": 20.0,
        }
    )
    _write_json(health / "runtime_throttle_control_latest.json", runtime)
    _write_json(health / "paper_400_ramp_latest.json", _paper_payload(blockers=[], armed=True))
    _write_soak_lane_artifacts(project_root)
    _ready_override(override)

    payload = src.build_payload(project_root)

    assert payload["overall_status"] == "ready"
    assert payload["failed_guards"] == []


def test_runtime_paper_guard_accepts_bounded_read_only_protected_lane_ready(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    health = project_root / "governance" / "health"
    override = project_root / "config" / ".env.runtime_resource_guard_override"
    runtime = _runtime_payload(blocked_paper=False)
    advisory = runtime["soft_cap_advisory_reclassification"]
    advisory["reason"] = "bounded_read_only_protected_lane_after_green_backpressure_is_guarded_runtime_ready"
    advisory["thresholds"].update(
        {
            "max_guarded_ready_protected_lane_cpu_percent": 75.0,
            "max_guarded_ready_bot_owned_with_protected_lane_cpu_percent": 95.0,
        }
    )
    advisory["measurements"].update(
        {
            "storage_ready_for_runtime_advisory": True,
            "bounded_protected_lane_guarded_ready": True,
            "live_read_only": True,
            "protected_work_hot": True,
            "bot_owned_pressure_dominant": True,
            "bot_owned_cpu_percent": 71.0,
            "protected_live_or_macro_cpu_percent": 62.0,
            "operator_observability_cpu_percent": 20.0,
        }
    )
    _write_json(health / "runtime_throttle_control_latest.json", runtime)
    _write_json(health / "paper_400_ramp_latest.json", _paper_payload(blockers=[], armed=True))
    _write_soak_lane_artifacts(project_root)
    _ready_override(override)

    payload = src.build_payload(project_root)

    assert payload["overall_status"] == "ready"
    assert payload["failed_guards"] == []


def test_runtime_paper_guard_blocks_stale_runtime_capacity_blocker(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    health = project_root / "governance" / "health"
    override = project_root / "config" / ".env.runtime_resource_guard_override"
    _write_json(health / "runtime_throttle_control_latest.json", _runtime_payload(blocked_paper=True))
    _write_json(
        health / "paper_400_ramp_latest.json",
        _paper_payload(blockers=["runtime_capacity_not_ready_for_400_paper", "paper_roster_below_400_target"]),
    )
    _write_soak_lane_artifacts(project_root)
    _blocked_override(override)

    payload = src.build_payload(project_root)

    assert payload["overall_status"] == "blocked"
    assert "paper_runtime_capacity_blocker_contract" in payload["failed_guards"]


def test_runtime_paper_guard_blocks_armed_paper_with_blockers(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    health = project_root / "governance" / "health"
    override = project_root / "config" / ".env.runtime_resource_guard_override"
    _write_json(health / "runtime_throttle_control_latest.json", _runtime_payload(blocked_paper=True))
    _write_json(health / "paper_400_ramp_latest.json", _paper_payload(armed=True))
    _write_soak_lane_artifacts(project_root)
    _blocked_override(override)

    payload = src.build_payload(project_root)

    assert payload["overall_status"] == "blocked"
    assert "paper_armed_blocker_contract" in payload["failed_guards"]


def test_runtime_paper_guard_degrades_on_missing_support_override_keys(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    health = project_root / "governance" / "health"
    override = project_root / "config" / ".env.runtime_resource_guard_override"
    _write_json(health / "runtime_throttle_control_latest.json", _runtime_payload(blocked_paper=False))
    _write_json(health / "paper_400_ramp_latest.json", _paper_payload(blockers=[], armed=True))
    _write_soak_lane_artifacts(project_root)
    _write_env(
        override,
        {
            "YTDLP_SUPPORT_NICE": "20",
            "TRAINING_RUNTIME_PAUSED_FOR_HOST_HEADROOM": "0",
        },
    )

    payload = src.build_payload(project_root)

    assert payload["overall_status"] == "degraded"
    assert "runtime_override_support_spawn_contract" in payload["failed_guards"]
    support_guard = next(row for row in payload["regression_guards"] if row["name"] == "runtime_override_support_spawn_contract")
    assert "MACRO_YTDLP_SUPPORT_NICE" in support_guard["actual"]["missing_keys"]
    assert "SHADOW_LOOP_RUNTIME_PAUSE_SLEEP_SECONDS" in support_guard["actual"]["missing_keys"]


def test_runtime_paper_guard_accepts_generic_support_override_aliases(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    health = project_root / "governance" / "health"
    override = project_root / "config" / ".env.runtime_resource_guard_override"
    _write_json(health / "runtime_throttle_control_latest.json", _runtime_payload(blocked_paper=False))
    _write_json(health / "paper_400_ramp_latest.json", _paper_payload(blockers=[], armed=True))
    _write_soak_lane_artifacts(project_root)
    _write_env(
        override,
        {
            "YTDLP_SUPPORT_NICE": "0",
            "MACRO_YTDLP_SUPPORT_NICE": "0",
            "OPS_SUPPORT_JOB_NICE": "20",
            "RUNTIME_RESEARCH_TRAINING_NICE": "15",
            "SHADOW_LOOP_RUNTIME_PAUSE_SLEEP_SECONDS": "60",
        },
    )

    payload = src.build_payload(project_root)

    assert payload["overall_status"] == "ready"
    support_guard = next(row for row in payload["regression_guards"] if row["name"] == "runtime_override_support_spawn_contract")
    assert support_guard["actual"]["missing_keys"] == []
    assert support_guard["actual"]["resolved_keys"]["YTDLP_SUPPORT_NICE"] == "OPS_SUPPORT_JOB_NICE"
    assert support_guard["actual"]["resolved_keys"]["TRAINING_RUNTIME_PAUSED_FOR_HOST_HEADROOM"] == "RUNTIME_RESEARCH_TRAINING_NICE"


def test_runtime_paper_guard_blocks_eligible_paper_when_stale_artifact_pauses_consumer(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    health = project_root / "governance" / "health"
    override = project_root / "config" / ".env.runtime_resource_guard_override"
    runtime = _runtime_payload(blocked_paper=False)
    runtime["paper_execution_policy"].update(
        {
            "paper_execution_allowed": True,
            "pause_paper_execution": True,
            "reason": "stale_runtime_control_artifact",
            "stage": "armed",
            "armed": True,
            "ok": True,
            "blockers": [],
        }
    )
    runtime["runtime_saturation_governor_v2"]["paper_live_data_policy"].update(
        {
            "paper_execution_allowed": True,
            "paper_execution_consumer_paused": True,
            "paper_execution_pause_reason": "stale_runtime_control_artifact",
        }
    )
    _write_json(health / "runtime_throttle_control_latest.json", runtime)
    _write_json(health / "paper_400_ramp_latest.json", _paper_payload(blockers=[], armed=True))
    old_ts = (datetime.now(timezone.utc) - timedelta(hours=8)).isoformat()
    _write_soak_lane_artifacts(project_root, profitability_timestamp_utc=old_ts)
    _ready_override(override)

    payload = src.build_payload(project_root)

    assert payload["overall_status"] == "blocked"
    assert "soak_paper_eligible_lane_open_contract" in payload["failed_guards"]
    lane_guard = next(row for row in payload["regression_guards"] if row["name"] == "soak_paper_eligible_lane_open_contract")
    assert "stale_artifact_blocking_eligible_paper_lane" in lane_guard["actual"]["lane_blockers"]


def test_runtime_paper_guard_degrades_on_stale_profitability_without_pausing_paper(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    health = project_root / "governance" / "health"
    override = project_root / "config" / ".env.runtime_resource_guard_override"
    _write_json(health / "runtime_throttle_control_latest.json", _runtime_payload(blocked_paper=False))
    _write_json(health / "paper_400_ramp_latest.json", _paper_payload(blockers=[], armed=True))
    old_ts = (datetime.now(timezone.utc) - timedelta(hours=8)).isoformat()
    _write_soak_lane_artifacts(project_root, profitability_timestamp_utc=old_ts)
    _ready_override(override)

    payload = src.build_payload(project_root)

    assert payload["overall_status"] == "degraded"
    assert "soak_hot_artifact_freshness_contract" in payload["failed_guards"]
    assert "soak_paper_eligible_lane_open_contract" not in payload["failed_guards"]
    production = next(row for row in payload["regression_guards"] if row["name"] == "production_grade_paper_live_authority_contract")
    assert production["ok"] is True
    assert production["actual"]["paper_open"] is True


def test_production_authority_contract_blocks_live_execution_authority(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    health = project_root / "governance" / "health"
    override = project_root / "config" / ".env.runtime_resource_guard_override"
    _write_json(health / "runtime_throttle_control_latest.json", _runtime_payload(blocked_paper=False))
    _write_json(health / "paper_400_ramp_latest.json", _paper_payload(blockers=[], armed=True))
    _write_soak_lane_artifacts(project_root)
    _ready_override(override)
    with override.open("a", encoding="utf-8") as handle:
        handle.write("EXECUTION_LANE_LIVE_ENABLED=1\n")

    payload = src.build_payload(project_root)

    assert payload["overall_status"] == "blocked"
    assert "production_grade_paper_live_authority_contract" in payload["failed_guards"]
    production = next(row for row in payload["regression_guards"] if row["name"] == "production_grade_paper_live_authority_contract")
    assert "live_execution_authority_enabled" in production["actual"]["blockers"]


def test_production_authority_contract_requires_auth_broker_session_for_eligible_paper(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    health = project_root / "governance" / "health"
    override = project_root / "config" / ".env.runtime_resource_guard_override"
    _write_json(health / "runtime_throttle_control_latest.json", _runtime_payload(blocked_paper=False))
    _write_json(health / "paper_400_ramp_latest.json", _paper_payload(blockers=[], armed=True))
    _write_soak_lane_artifacts(project_root)
    _write_json(health / "broker_readiness_latest.json", {"ready_for_open": False, "auth_ok": True, "network_ok": True})
    _ready_override(override)

    payload = src.build_payload(project_root)

    assert payload["overall_status"] == "blocked"
    assert "production_grade_paper_live_authority_contract" in payload["failed_guards"]
    production = next(row for row in payload["regression_guards"] if row["name"] == "production_grade_paper_live_authority_contract")
    assert "broker_not_ready" in production["actual"]["blockers"]
    continuity = next(row for row in payload["regression_guards"] if row["name"] == "soak_30_day_continuity_contract")
    assert "broker_not_ready" in continuity["actual"]["blockers"]


def test_paper_soak_accepts_probe_denied_when_token_and_broker_are_operable(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    health = project_root / "governance" / "health"
    override = project_root / "config" / ".env.runtime_resource_guard_override"
    _write_json(health / "runtime_throttle_control_latest.json", _runtime_payload(blocked_paper=False))
    _write_json(health / "paper_400_ramp_latest.json", _paper_payload(blockers=[], armed=True))
    _write_soak_lane_artifacts(project_root)
    _write_json(
        health / "auth_lease_manager_latest.json",
        {
            "timestamp_utc": src.iso_now(),
            "ok": False,
            "overall_status": "degraded",
            "lease_state": "warning",
            "lease_budget": {"expires_in_seconds": 1365, "critical_lease_seconds": 600},
            "broker_state": {
                "broker_ready": True,
                "broker_operable": True,
                "network_ok": True,
                "auth_ok": False,
                "auth_probe_ok": False,
                "configured_for_refresh": True,
            },
        },
    )
    _write_json(
        health / "schwab_auth_supervisor_latest.json",
        {
            "timestamp_utc": src.iso_now(),
            "ok": False,
            "overall_status": "degraded",
            "token": {
                "ready": True,
                "expires_in_seconds": 1365,
                "readiness_refresh_needed": False,
            },
            "min_ready_expires_seconds": 900,
            "refresh_needed": True,
        },
    )
    _write_json(
        health / "broker_readiness_latest.json",
        {
            "timestamp_utc": src.iso_now(),
            "ready_for_open": True,
            "auth_ok": False,
            "network_ok": True,
            "token_expires_in_seconds": 1365,
            "preflight_checks": {
                "token_exists": True,
                "token_ready_for_open": True,
                "readiness_refresh_needed_after": False,
            },
        },
    )
    _ready_override(override)

    payload = src.build_payload(project_root)

    assert payload["overall_status"] == "ready"
    production = next(row for row in payload["regression_guards"] if row["name"] == "production_grade_paper_live_authority_contract")
    continuity = next(row for row in payload["regression_guards"] if row["name"] == "soak_30_day_continuity_contract")
    assert production["actual"]["strict_auth_ready"] is False
    assert production["actual"]["paper_soak_auth_grace"] is True
    assert "auth_stack_not_ready" not in production["actual"]["blockers"]
    assert "broker_not_ready" not in production["actual"]["blockers"]
    assert "auth_stack_not_ready" not in continuity["actual"]["blockers"]


def test_production_authority_contract_blocks_raw_profitability_cosmetic_upgrade(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    health = project_root / "governance" / "health"
    override = project_root / "config" / ".env.runtime_resource_guard_override"
    _write_json(health / "runtime_throttle_control_latest.json", _runtime_payload(blocked_paper=False))
    _write_json(health / "paper_400_ramp_latest.json", _paper_payload(blockers=[], armed=True))
    _write_soak_lane_artifacts(project_root)
    _write_json(
        health / "paper_runtime_profitability_controls_latest.json",
        {
            "timestamp_utc": src.iso_now(),
            "raw_profitability_grade": "A",
            "controlled_profitability_grade": "A+",
            "raw_profitability_a_recovery_contract": {
                "active": True,
                "gap_to_raw_a": {"net_pnl_gap": 2500.0},
                "runtime_enforcement": {
                    "block_new_entries_on_weak_profiles": True,
                    "keep_sells_and_reduce_only_paths_open": True,
                },
            },
        },
    )
    _ready_override(override)

    payload = src.build_payload(project_root)

    assert payload["overall_status"] == "blocked"
    assert "production_grade_paper_live_authority_contract" in payload["failed_guards"]
    production = next(row for row in payload["regression_guards"] if row["name"] == "production_grade_paper_live_authority_contract")
    assert "raw_profitability_grade_cosmetic_upgrade" in production["actual"]["blockers"]


def test_production_authority_contract_requires_raw_improvement_contract(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    health = project_root / "governance" / "health"
    override = project_root / "config" / ".env.runtime_resource_guard_override"
    _write_json(health / "runtime_throttle_control_latest.json", _runtime_payload(blocked_paper=False))
    _write_json(health / "paper_400_ramp_latest.json", _paper_payload(blockers=[], armed=True))
    _write_soak_lane_artifacts(project_root)
    _write_json(
        health / "paper_runtime_profitability_controls_latest.json",
        {
            "timestamp_utc": src.iso_now(),
            "raw_profitability_grade": "D",
            "controlled_profitability_grade": "A+",
            "raw_profitability_a_recovery_contract": {
                "active": True,
                "gap_to_raw_a": {"net_pnl_gap": 2500.0},
                "runtime_enforcement": {
                    "block_new_entries_on_weak_profiles": True,
                    "keep_sells_and_reduce_only_paths_open": True,
                    "raise_clean_profile_buy_gate_while_raw_below_a": True,
                    "block_when_source_or_fill_unknown": True,
                },
            },
            "global_runtime_policy": {"apply_raw_profitability_a_recovery": True},
        },
    )
    _ready_override(override)

    payload = src.build_payload(project_root)

    assert payload["overall_status"] == "blocked"
    production = next(row for row in payload["regression_guards"] if row["name"] == "production_grade_paper_live_authority_contract")
    assert "raw_profitability_improvement_contract_not_ready" in production["actual"]["blockers"]


def test_production_authority_contract_accepts_hard_storage_blocker_only_when_paper_fails_closed(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    health = project_root / "governance" / "health"
    override = project_root / "config" / ".env.runtime_resource_guard_override"
    runtime = _runtime_payload(blocked_paper=True)
    _write_json(health / "runtime_throttle_control_latest.json", runtime)
    _write_json(
        health / "paper_400_ramp_latest.json",
        _paper_payload(blockers=["ingestion_or_backpressure_above_paper_400_gate"], armed=False),
    )
    _write_soak_lane_artifacts(project_root)
    _blocked_override(override)

    payload = src.build_payload(project_root)

    production = next(row for row in payload["regression_guards"] if row["name"] == "production_grade_paper_live_authority_contract")
    assert production["ok"] is True
    assert production["actual"]["hard_paper_blockers"] == ["ingestion_or_backpressure_above_paper_400_gate"]
    assert production["actual"]["hard_blocker_fail_closed"] is True


def test_paper_soak_keeps_existing_execution_open_when_only_400_expansion_is_paused(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    health = project_root / "governance" / "health"
    override = project_root / "config" / ".env.runtime_resource_guard_override"
    _write_json(health / "runtime_throttle_control_latest.json", _runtime_payload(blocked_paper=False))
    _write_json(
        health / "paper_400_ramp_latest.json",
        _paper_payload(blockers=["ingestion_or_backpressure_above_paper_400_gate"], armed=False),
    )
    _write_soak_lane_artifacts(project_root)
    _ready_override(override)

    payload = src.build_payload(project_root)

    assert payload["overall_status"] == "ready"
    assert payload["failed_guards"] == []
    pause = next(row for row in payload["regression_guards"] if row["name"] == "blocked_paper_execution_pause_contract")
    production = next(row for row in payload["regression_guards"] if row["name"] == "production_grade_paper_live_authority_contract")
    continuity = next(row for row in payload["regression_guards"] if row["name"] == "soak_30_day_continuity_contract")
    assert pause["actual"]["existing_paper_execution_open"] is True
    assert production["actual"]["expansion_only_blockers"] == ["ingestion_or_backpressure_above_paper_400_gate"]
    assert production["actual"]["fail_closed_blockers"] == []
    assert continuity["actual"]["expansion_pause_existing_paper_open"] is True


def test_paper_soak_still_fails_closed_for_global_halt_blocker(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    health = project_root / "governance" / "health"
    override = project_root / "config" / ".env.runtime_resource_guard_override"
    _write_json(health / "runtime_throttle_control_latest.json", _runtime_payload(blocked_paper=False))
    _write_json(
        health / "paper_400_ramp_latest.json",
        _paper_payload(blockers=["global_halt_or_clear_blocker_active"], armed=False),
    )
    _write_soak_lane_artifacts(project_root)
    _ready_override(override)

    payload = src.build_payload(project_root)

    assert payload["overall_status"] == "blocked"
    assert "blocked_paper_execution_pause_contract" in payload["failed_guards"]
    assert "production_grade_paper_live_authority_contract" in payload["failed_guards"]
    production = next(row for row in payload["regression_guards"] if row["name"] == "production_grade_paper_live_authority_contract")
    assert production["actual"]["fail_closed_blockers"] == ["global_halt_or_clear_blocker_active"]
    assert "hard_safety_blocker_without_paper_fail_closed" in production["actual"]["blockers"]


def test_soak_continuity_blocks_eligible_paper_when_override_pauses_queue(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    health = project_root / "governance" / "health"
    override = project_root / "config" / ".env.runtime_resource_guard_override"
    _write_json(health / "runtime_throttle_control_latest.json", _runtime_payload(blocked_paper=False))
    _write_json(health / "paper_400_ramp_latest.json", _paper_payload(blockers=[], armed=True))
    _write_soak_lane_artifacts(project_root)
    _blocked_override(override)

    payload = src.build_payload(project_root)

    assert payload["overall_status"] == "blocked"
    assert "soak_30_day_continuity_contract" in payload["failed_guards"]
    continuity = next(row for row in payload["regression_guards"] if row["name"] == "soak_30_day_continuity_contract")
    assert "runtime_override_pauses_paper_consumer" in continuity["actual"]["blockers"]


def test_soak_continuity_blocks_eligible_paper_when_runtime_is_degraded(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    health = project_root / "governance" / "health"
    override = project_root / "config" / ".env.runtime_resource_guard_override"
    runtime = _runtime_payload(blocked_paper=False)
    runtime["overall_status"] = "degraded"
    runtime["ok"] = False
    runtime["soft_cap_advisory_reclassification"].update(
        {"active": False, "to_status": "degraded", "reason": "soft_cap_still_requires_degraded_posture"}
    )
    _write_json(health / "runtime_throttle_control_latest.json", runtime)
    _write_json(health / "paper_400_ramp_latest.json", _paper_payload(blockers=[], armed=True))
    _write_soak_lane_artifacts(project_root)
    _ready_override(override)

    payload = src.build_payload(project_root)

    assert payload["overall_status"] == "blocked"
    assert "soak_30_day_continuity_contract" in payload["failed_guards"]
    continuity = next(row for row in payload["regression_guards"] if row["name"] == "soak_30_day_continuity_contract")
    assert "runtime_not_ready_or_advisory" in continuity["actual"]["blockers"]


def test_soak_continuity_accepts_capacity_limited_armed_paper_ramp(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    health = project_root / "governance" / "health"
    override = project_root / "config" / ".env.runtime_resource_guard_override"
    runtime = _runtime_payload(blocked_paper=False)
    runtime["overall_status"] = "degraded"
    runtime["paper_capacity_contract"] = {
        "ready_for_700_bot_paper": False,
        "pressure_limited": True,
        "mode": "full_force_guarded",
    }
    paper = _paper_payload(blockers=[], armed=True)
    paper["gates"]["runtime"].update(
        {
            "status": "capacity_limited_armed",
            "blockers": [],
            "runtime_pressure_ready": False,
            "runtime_capacity_ready": True,
            "ready_for_700_bot_paper": False,
            "capacity_limited_armed": True,
            "paper_execution_clean": True,
            "live_execution_locked": True,
            "pressure_limited": True,
            "active_bot_capacity_ready": True,
            "paper_roster_ready": True,
        }
    )
    _write_json(health / "runtime_throttle_control_latest.json", runtime)
    _write_json(health / "paper_400_ramp_latest.json", paper)
    _write_soak_lane_artifacts(project_root)
    _ready_override(override)

    payload = src.build_payload(project_root)

    assert "soak_30_day_continuity_contract" not in payload["failed_guards"]
    continuity = next(row for row in payload["regression_guards"] if row["name"] == "soak_30_day_continuity_contract")
    assert continuity["ok"] is True
    assert continuity["actual"]["runtime_gate_ready"] is True
    assert continuity["actual"]["runtime_status_ready"] is True
    assert continuity["actual"]["capacity_ready"] is True
    assert continuity["actual"]["capacity_limited_paper_gate_safe"] is True


def test_infrastructure_autofix_assigns_runtime_paper_regression_guard(tmp_path: Path, monkeypatch) -> None:
    project_root = tmp_path / "project"
    health = project_root / "governance" / "health"
    _write_json(
        health / "runtime_paper_regression_guard_latest.json",
        {"overall_status": "blocked", "failed_guard_count": 2},
    )
    monkeypatch.setattr(
        infra_src,
        "_run_json",
        lambda cmd, *, cwd, timeout_sec: {
            "cmd": list(cmd),
            "rc": 0,
            "timed_out": False,
            "stdout_tail": "",
            "stderr_tail": "",
            "payload": {"overall_status": "ready", "ok": True, "metrics": {}},
        },
    )

    payload = infra_src.build_payload(project_root, apply=False)

    names = [row["name"] for row in payload["repair_plan"]]
    assert "runtime_paper_regression_guard" in names
    assert "runtime_paper_regression_guard" in payload["infra_bots"]
    assert "runtime_paper_contract_infrabot" in payload["infra_bots"]
    assert payload["metrics"]["runtime_paper_failed_guard_count"] == 2
