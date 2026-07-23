import json
from datetime import date
from pathlib import Path

from scripts.ops import paper_400_ramp_control as src


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def _seed_ready_project(project_root: Path, *, bot_count: int = 700) -> None:
    health = project_root / "governance" / "health"
    _write_json(
        health / "memory_efficiency_control_latest.json",
        {
            "overall_status": "needs_work",
            "recommended_profile": "constrained",
            "memory_snapshot": {
                "memory_pressure_state": "green",
                "memory_free_pct": 52.0,
                "swap_used_gb": 0.5,
                "compressed_store_gb": 18.5,
                "compressor_gb": 9.5,
            },
        },
    )
    _write_json(
        health / "runtime_throttle_control_latest.json",
        {
            "overall_status": "degraded",
            "throttle_profile": "sustain",
            "compute_pressure_level": "elevated",
            "memory_pressure_level": "elevated",
            "paper_capacity_contract": {
                "ready_for_700_bot_paper": True,
                "pressure_limited": False,
                "active_bot_count": bot_count,
                "paper_tagged_count": bot_count,
            },
        },
    )
    _write_json(
        health / "ingestion_storage_control_latest.json",
        {
            "overall_status": "ready",
            "severity": "stable",
            "pressure_index": 0.02,
            "backpressure": {
                "core_pending_lines": 120,
                "total_pending_lines": 2500,
            },
        },
    )
    _write_json(
        health / "global_killswitch_latest.json",
        {
            "halt": False,
            "clear_ready": True,
            "clear_blockers": [],
        },
    )
    _write_json(
        project_root / "master_bot_registry.json",
        {
            "sub_bots": [
                {
                    "bot_id": f"brain_refinery_v{i}_paper_ready_bot",
                    "active": True,
                    "paper_trade_enabled": True,
                    "lifecycle_state": "active",
                }
                for i in range(bot_count)
            ]
        },
    )


def _write_profitability_weak_profiles(project_root: Path, profiles: list[str]) -> None:
    _write_json(
        project_root / "governance" / "health" / "paper_runtime_profitability_controls_latest.json",
        {
            "timestamp_utc": "2026-07-13T17:36:52+00:00",
            "raw_profitability_grade": "D",
            "controlled_profitability_grade": "A+",
            "raw_profitability_a_recovery_contract": {
                "active": True,
                "weak_profiles": profiles,
                "runtime_enforcement": {
                    "block_new_entries_on_weak_profiles": True,
                    "keep_sells_and_reduce_only_paths_open": True,
                },
            },
            "raw_profitability_improvement_contract": {
                "active": True,
                "weak_sleeve_zero_entry_contract": {
                    "profiles": [
                        {"profile": profile, "block_new_entries": True}
                        for profile in profiles
                    ],
                },
            },
        },
    )


def test_paper_400_ramp_plans_before_activation_without_high_caps(tmp_path: Path) -> None:
    _seed_ready_project(tmp_path)

    payload = src.build_payload(
        tmp_path,
        today=date(2026, 5, 4),
        registry_path=tmp_path / "master_bot_registry.json",
    )
    payload = src.apply_payload(
        tmp_path,
        payload,
        out_path=tmp_path / "governance" / "health" / "paper_400_ramp_latest.json",
        override_path=tmp_path / "config" / ".env.paper_400_ramp_override",
    )

    override_text = (tmp_path / "config" / ".env.paper_400_ramp_override").read_text(encoding="utf-8")
    assert payload["stage"] == "planned"
    assert payload["armed"] is False
    assert payload["blockers"] == ["calendar_wait_until_2026-05-11"]
    assert "PAPER_400_RAMP_ARMED=0" in override_text
    assert "PAPER_MIRROR_ALL_ACTIVE_SUB_BOTS=1" in override_text
    assert "SCHWAB_TOP_BOT_PAPER_TRADING_TOP_N" not in override_text
    assert "COINBASE_FUTURES_TOP_BOT_PAPER_TRADING_TOP_N" not in override_text


def test_paper_400_ramp_arms_after_activation_when_gates_are_clean(tmp_path: Path) -> None:
    _seed_ready_project(tmp_path)

    payload = src.build_payload(
        tmp_path,
        today=date(2026, 5, 11),
        registry_path=tmp_path / "master_bot_registry.json",
    )
    payload = src.apply_payload(
        tmp_path,
        payload,
        out_path=tmp_path / "governance" / "health" / "paper_400_ramp_latest.json",
        override_path=tmp_path / "config" / ".env.paper_400_ramp_override",
    )

    override_text = (tmp_path / "config" / ".env.paper_400_ramp_override").read_text(encoding="utf-8")
    allocation_total = sum(int(row["target"]) for row in payload["paper_allocation"]["lanes"].values())
    assert payload["stage"] == "armed"
    assert payload["armed"] is True
    assert allocation_total == 400
    assert "PAPER_400_RAMP_ARMED=1" in override_text
    assert "PAPER_MIRROR_ALL_ACTIVE_SUB_BOTS=1" in override_text
    assert "PAPER_400_RAMP_AGGREGATE_TOP_N=700" in override_text
    assert "PAPER_400_RAMP_SELECTION_POLICY=all_eligible_paper_live_data_when_mirror_all_active_enabled" in override_text
    assert "PAPER_FULL_FORCE_STABILITY_MODE=all_eligible_paper_buffered" in override_text
    assert "SCHWAB_TOP_BOT_PAPER_TRADING_TOP_N=700" in override_text
    assert "SCHWAB_OPTIONS_TOP_BOT_PAPER_TRADING_TOP_N=700" in override_text
    assert "SCHWAB_FUTURES_TOP_BOT_PAPER_TRADING_TOP_N=700" in override_text
    assert "COINBASE_TOP_BOT_PAPER_TRADING_TOP_N=50" in override_text
    assert "COINBASE_FUTURES_TOP_BOT_PAPER_TRADING_TOP_N=30" in override_text


def test_paper_400_ramp_keeps_coinbase_paper_probation_when_profiles_are_weak(tmp_path: Path) -> None:
    _seed_ready_project(tmp_path)
    _write_profitability_weak_profiles(
        tmp_path,
        ["default", "crypto_futures", "bond", "fx", "options_on_futures"],
    )

    payload = src.build_payload(
        tmp_path,
        today=date(2026, 5, 11),
        registry_path=tmp_path / "master_bot_registry.json",
    )
    payload = src.apply_payload(
        tmp_path,
        payload,
        out_path=tmp_path / "governance" / "health" / "paper_400_ramp_latest.json",
        override_path=tmp_path / "config" / ".env.paper_400_ramp_override",
    )

    override_text = (tmp_path / "config" / ".env.paper_400_ramp_override").read_text(encoding="utf-8")
    lanes = payload["paper_allocation"]["lanes"]
    assert "default" not in lanes["schwab_equities"]["profiles"].split(",")
    assert "bond" not in lanes["schwab_equities"]["profiles"].split(",")
    assert "fx" not in lanes["schwab_equities"]["profiles"].split(",")
    assert "volatility" in lanes["schwab_equities"]["profiles"].split(",")
    assert lanes["coinbase_spot"]["profiles"] == "default"
    assert lanes["coinbase_futures"]["profiles"] == "crypto_futures"
    assert lanes["coinbase_spot"]["paper_probation_active"] is True
    assert lanes["coinbase_futures"]["paper_probation_active"] is True
    assert "COINBASE_TOP_BOT_PAPER_TRADING_TOP_N=50" in override_text
    assert "COINBASE_TOP_BOT_PAPER_TRADING_PROFILES=default" in override_text
    assert "COINBASE_FUTURES_TOP_BOT_PAPER_TRADING_TOP_N=30" in override_text
    assert "COINBASE_FUTURES_TOP_BOT_PAPER_TRADING_PROFILES=crypto_futures" in override_text
    assert "COINBASE_PAPER_PROBATION_ENABLED=1" in override_text
    assert "PAPER_SOAK_SPECIALIZED_ALLOWLIST_BYPASS_FANOUT=1" in override_text
    assert "RUN_ALL_SLEEVES_WITH_SPECIALIZED_SLEEVES=1" in override_text
    assert "RUN_ALL_SLEEVES_SPECIALIZED_PROFILE_ALLOWLIST=volatility" in override_text


def test_paper_400_ramp_separates_paper_roster_from_runtime_capacity(tmp_path: Path) -> None:
    _seed_ready_project(tmp_path)
    health = tmp_path / "governance" / "health"
    _write_json(
        health / "runtime_throttle_control_latest.json",
        {
            "overall_status": "advisory",
            "throttle_profile": "soft_cap",
            "compute_pressure_level": "elevated",
            "memory_pressure_level": "normal",
            "paper_capacity_contract": {
                "ready_for_700_bot_paper": True,
                "pressure_limited": False,
                "active_bot_count": 700,
                "paper_tagged_count": 35,
            },
        },
    )
    _write_json(
        tmp_path / "master_bot_registry.json",
        {
            "sub_bots": [
                {
                    "bot_id": f"brain_refinery_v{i}_paper_roster_test_bot",
                    "active": True,
                    "paper_trade_enabled": i < 35,
                    "lifecycle_state": "active",
                }
                for i in range(700)
            ]
        },
    )

    payload = src.build_payload(
        tmp_path,
        today=date(2026, 5, 11),
        registry_path=tmp_path / "master_bot_registry.json",
    )

    assert "paper_roster_below_400_target" in payload["blockers"]
    assert "runtime_capacity_not_ready_for_400_paper" not in payload["blockers"]
    assert payload["gates"]["runtime"]["runtime_pressure_ready"] is True
    assert payload["gates"]["runtime"]["paper_roster_ready"] is False


def test_paper_400_ramp_accepts_external_compute_advisory_as_runtime_ready(tmp_path: Path) -> None:
    _seed_ready_project(tmp_path)
    health = tmp_path / "governance" / "health"
    _write_json(
        health / "runtime_throttle_control_latest.json",
        {
            "overall_status": "advisory",
            "throttle_profile": "sustain",
            "compute_pressure_level": "high",
            "memory_pressure_level": "normal",
            "soft_cap_advisory_reclassification": {
                "active": True,
                "reason": "external_high_compute_with_bounded_storage_overlay_is_capacity_limited_advisory",
                "measurements": {
                    "external_high_compute_guarded": True,
                    "bounded_storage_overlay_guarded": True,
                    "storage_ready_for_runtime_advisory": True,
                },
            },
            "paper_capacity_contract": {
                "ready_for_700_bot_paper": False,
                "pressure_limited": True,
                "active_bot_count": 700,
                "paper_tagged_count": 35,
            },
        },
    )
    _write_json(
        tmp_path / "master_bot_registry.json",
        {
            "sub_bots": [
                {
                    "bot_id": f"brain_refinery_v{i}_external_advisory_test_bot",
                    "active": True,
                    "paper_trade_enabled": i < 35,
                    "lifecycle_state": "active",
                }
                for i in range(700)
            ]
        },
    )

    payload = src.build_payload(
        tmp_path,
        today=date(2026, 5, 11),
        registry_path=tmp_path / "master_bot_registry.json",
    )

    assert "paper_roster_below_400_target" in payload["blockers"]
    assert "runtime_capacity_not_ready_for_400_paper" not in payload["blockers"]
    assert payload["gates"]["runtime"]["attribution_capacity_advisory"] is True
    assert payload["gates"]["runtime"]["compute_pressure_ready"] is True
    assert payload["gates"]["runtime"]["runtime_pressure_ready"] is True
    assert payload["gates"]["runtime"]["runtime_capacity_ready"] is True
    assert payload["gates"]["runtime"]["paper_roster_ready"] is False


def test_paper_400_ramp_preserves_clean_armed_state_under_transient_capacity_pressure(tmp_path: Path) -> None:
    _seed_ready_project(tmp_path, bot_count=700)
    health = tmp_path / "governance" / "health"
    _write_json(
        health / "runtime_throttle_control_latest.json",
        {
            "overall_status": "degraded",
            "throttle_profile": "sustain",
            "compute_pressure_level": "high",
            "memory_pressure_level": "normal",
            "paper_execution_policy": {
                "paper_execution_allowed": True,
                "pause_paper_execution": False,
                "reason": "paper_ramp_armed_and_clean",
                "stage": "armed",
                "armed": True,
                "ok": True,
                "blockers": [],
            },
            "paper_capacity_contract": {
                "ready_for_700_bot_paper": False,
                "pressure_limited": True,
                "active_bot_count": 700,
                "paper_tagged_count": 400,
                "runtime_policy": {"live_execution_blocked": True},
            },
        },
    )
    _write_json(
        tmp_path / "master_bot_registry.json",
        {
            "sub_bots": [
                {
                    "bot_id": f"brain_refinery_v{i}_capacity_limited_armed_bot",
                    "active": True,
                    "paper_trade_enabled": i < 400,
                    "lifecycle_state": "active",
                }
                for i in range(700)
            ]
        },
    )

    payload = src.build_payload(
        tmp_path,
        today=date(2026, 5, 11),
        registry_path=tmp_path / "master_bot_registry.json",
    )

    assert payload["stage"] == "armed"
    assert payload["armed"] is True
    assert "runtime_capacity_not_ready_for_400_paper" not in payload["blockers"]
    assert payload["gates"]["runtime"]["status"] == "capacity_limited_armed"
    assert payload["gates"]["runtime"]["capacity_limited_armed"] is True
    assert payload["gates"]["runtime"]["runtime_capacity_ready"] is True
    assert payload["gates"]["runtime"]["ready_for_700_bot_paper"] is False


def test_paper_400_ramp_accepts_runtime_pressure_bypass_as_capacity_limited_armed(tmp_path: Path) -> None:
    _seed_ready_project(tmp_path, bot_count=700)
    health = tmp_path / "governance" / "health"
    _write_json(
        health / "runtime_throttle_control_latest.json",
        {
            "overall_status": "degraded",
            "throttle_profile": "sustain",
            "compute_pressure_level": "high",
            "memory_pressure_level": "normal",
            "paper_execution_policy": {
                "paper_execution_allowed": True,
                "pause_paper_execution": False,
                "reason": "paper_ramp_pressure_only_blocker_bypassed_for_full_force_soak",
                "stage": "blocked",
                "armed": False,
                "ok": False,
                "blockers": ["runtime_capacity_not_ready_for_400_paper"],
                "pressure_pause_bypassed": True,
                "pressure_pause_bypass_reason": "full_force_paper_ramp_pressure_only_blocker",
            },
            "paper_capacity_contract": {
                "ready_for_700_bot_paper": False,
                "pressure_limited": True,
                "active_bot_count": 700,
                "paper_tagged_count": 400,
                "runtime_policy": {"live_execution_blocked": True},
            },
        },
    )

    payload = src.build_payload(
        tmp_path,
        today=date(2026, 5, 11),
        registry_path=tmp_path / "master_bot_registry.json",
    )

    assert payload["stage"] == "armed"
    assert payload["armed"] is True
    assert "runtime_capacity_not_ready_for_400_paper" not in payload["blockers"]
    assert payload["gates"]["runtime"]["status"] == "capacity_limited_armed"
    assert payload["gates"]["runtime"]["paper_pressure_bypass"] is True
    assert payload["gates"]["runtime"]["paper_execution_clean"] is True


def test_paper_400_ramp_allows_cool_raw_live_sql_overlay_storage_pressure(tmp_path: Path) -> None:
    _seed_ready_project(tmp_path)
    health = tmp_path / "governance" / "health"
    _write_json(
        health / "ingestion_storage_control_latest.json",
        {
            "overall_status": "blocked",
            "severity": "critical",
            "pressure_index": 11.95,
            "backpressure": {
                "core_pending_lines": 6161,
                "total_pending_lines": 6161,
                "overlay_adjusted": True,
                "raw_live": {
                    "core_pending_lines": 374,
                    "total_pending_lines": 374,
                    "oldest_pending_age_seconds": 0.0,
                },
            },
        },
    )

    payload = src.build_payload(
        tmp_path,
        today=date(2026, 5, 11),
        registry_path=tmp_path / "master_bot_registry.json",
    )

    assert "ingestion_or_backpressure_above_paper_400_gate" not in payload["blockers"]
    assert payload["gates"]["storage"]["ok"] is True
    assert payload["gates"]["storage"]["status"] == "overlay_drain_advisory"
    assert payload["gates"]["storage"]["overlay_only_relief"]["active"] is True


def test_paper_400_ramp_prefers_effective_raw_live_for_overlay_storage_relief(tmp_path: Path) -> None:
    _seed_ready_project(tmp_path)
    health = tmp_path / "governance" / "health"
    _write_json(
        health / "ingestion_storage_control_latest.json",
        {
            "overall_status": "needs_work",
            "severity": "high",
            "pressure_index": 1.894,
            "backpressure": {
                "core_pending_lines": 5980,
                "total_pending_lines": 5980,
                "overlay_adjusted": True,
                "raw_live": {
                    "core_pending_lines": 18881,
                    "total_pending_lines": 1235466,
                    "oldest_pending_age_seconds": 106.561,
                },
                "effective_raw_live": {
                    "core_pending_lines": 5980,
                    "total_pending_lines": 5980,
                    "oldest_pending_age_seconds": 454.443,
                    "source": "sql_ingestion_overlay_pressure",
                    "reconciled_from_raw_live": True,
                },
            },
        },
    )

    payload = src.build_payload(
        tmp_path,
        today=date(2026, 5, 11),
        registry_path=tmp_path / "master_bot_registry.json",
    )

    assert "ingestion_or_backpressure_above_paper_400_gate" not in payload["blockers"]
    assert payload["gates"]["storage"]["ok"] is True
    assert payload["gates"]["storage"]["status"] == "overlay_drain_advisory"
    assert payload["gates"]["storage"]["overlay_only_relief"]["active"] is True
    assert payload["gates"]["storage"]["overlay_only_relief"]["raw_live_source"] == "sql_ingestion_overlay_pressure"


def test_paper_400_ramp_uses_reconciled_raw_live_estimate_for_overlay_relief(tmp_path: Path) -> None:
    _seed_ready_project(tmp_path)
    health = tmp_path / "governance" / "health"
    _write_json(
        health / "ingestion_storage_control_latest.json",
        {
            "overall_status": "blocked",
            "severity": "critical",
            "pressure_index": 63.527,
            "backpressure": {
                "core_pending_lines": 379,
                "total_pending_lines": 114637,
                "oldest_pending_age_seconds": 15246.365,
                "overlay_adjusted": True,
                "raw_live": {
                    "core_pending_lines": 379,
                    "total_pending_lines": 380,
                    "oldest_pending_age_seconds": 417.192,
                },
                "effective_raw_live": {
                    "core_pending_lines": 379,
                    "support_pending_lines": 114257,
                    "total_pending_lines": 114637,
                    "oldest_pending_age_seconds": 15246.365,
                    "source": "sql_ingestion_overlay_pressure",
                    "reconciled_from_raw_live": True,
                    "raw_live_estimate": {
                        "core_pending_lines": 379,
                        "total_pending_lines": 380,
                        "oldest_pending_age_seconds": 417.192,
                    },
                },
            },
        },
    )

    payload = src.build_payload(
        tmp_path,
        today=date(2026, 5, 11),
        registry_path=tmp_path / "master_bot_registry.json",
    )

    assert payload["stage"] == "armed"
    assert "ingestion_or_backpressure_above_paper_400_gate" not in payload["blockers"]
    assert payload["gates"]["storage"]["status"] == "overlay_drain_advisory"
    assert payload["gates"]["storage"]["overlay_only_relief"]["raw_live_source"] == "effective_raw_live.raw_live_estimate"
    assert payload["gates"]["storage"]["overlay_only_relief"]["raw_live"]["total_pending_lines"] == 380


def test_paper_400_ramp_allows_stable_storage_pressure_hysteresis_band(tmp_path: Path) -> None:
    _seed_ready_project(tmp_path)
    health = tmp_path / "governance" / "health"
    _write_json(
        health / "ingestion_storage_control_latest.json",
        {
            "overall_status": "ready",
            "severity": "stable",
            "pressure_index": 0.334,
            "backpressure": {
                "core_pending_lines": 5156,
                "total_pending_lines": 5156,
                "overlay_adjusted": True,
                "raw_live": {
                    "core_pending_lines": 2059,
                    "total_pending_lines": 2172,
                    "oldest_pending_age_seconds": 23.5,
                },
            },
        },
    )

    payload = src.build_payload(
        tmp_path,
        today=date(2026, 5, 11),
        registry_path=tmp_path / "master_bot_registry.json",
    )

    assert payload["stage"] == "armed"
    assert "ingestion_or_backpressure_above_paper_400_gate" not in payload["blockers"]
    assert payload["gates"]["storage"]["ok"] is True
    assert payload["gates"]["storage"]["pressure_advisory"] is True
    assert payload["gates"]["storage"]["status"] == "overlay_drain_advisory"


def test_paper_400_ramp_treats_bounded_write_path_recovery_as_advisory(tmp_path: Path) -> None:
    _seed_ready_project(tmp_path)
    health = tmp_path / "governance" / "health"
    _write_json(
        health / "global_killswitch_latest.json",
        {
            "halt": False,
            "clear_ready": False,
            "clear_blockers": ["write_path_recovery_pending"],
            "metrics": {"execution_expected": False},
        },
    )
    _write_json(
        health / "data_plane_recovery_controller_latest.json",
        {
            "overall_status": "degraded",
            "recovery_state": "recovering_under_guard",
            "write_failure_count": 2,
            "account_snapshot_failure_count": 0,
            "queue_depth": 351,
        },
    )

    payload = src.build_payload(
        tmp_path,
        today=date(2026, 5, 11),
        registry_path=tmp_path / "master_bot_registry.json",
    )

    assert "global_halt_or_clear_blocker_active" not in payload["blockers"]
    assert payload["gates"]["global_halt"]["ok"] is True
    assert payload["gates"]["global_halt"]["status"] == "write_path_recovery_advisory"
    assert payload["gates"]["global_halt"]["clear_blocker_relief"]["active"] is True
    assert payload["stage"] == "armed"


def test_paper_400_ramp_prefers_auto_clear_halt_artifact_over_stale_killswitch(tmp_path: Path) -> None:
    _seed_ready_project(tmp_path)
    health = tmp_path / "governance" / "health"
    _write_json(
        health / "global_killswitch_latest.json",
        {
            "halt": False,
            "clear_ready": False,
            "clear_blockers": ["write_path_recovery_pending"],
            "metrics": {"execution_expected": False},
        },
    )
    _write_json(
        health / "global_halt_auto_clear_latest.json",
        {
            "halt": False,
            "clear_ready": False,
            "clear_blockers": ["write_path_recovery_pending"],
            "metrics": {"execution_expected": False},
        },
    )
    _write_json(
        health / "data_plane_recovery_controller_latest.json",
        {
            "overall_status": "degraded",
            "recovery_state": "recovering_under_guard",
            "write_failure_count": 5,
            "account_snapshot_failure_count": 0,
            "queue_depth": 351,
        },
    )

    payload = src.build_payload(
        tmp_path,
        today=date(2026, 5, 11),
        registry_path=tmp_path / "master_bot_registry.json",
    )

    assert payload["stage"] == "armed"
    assert payload["gates"]["global_halt"]["status"] == "write_path_recovery_advisory"


def test_paper_400_ramp_accepts_clean_unlatched_degraded_collection_halt(tmp_path: Path) -> None:
    _seed_ready_project(tmp_path)
    health = tmp_path / "governance" / "health"
    _write_json(
        health / "global_halt_auto_clear_latest.json",
        {
            "halt": False,
            "halt_latched": False,
            "halt_required": False,
            "would_rehalt": False,
            "halt_posture": "unlatched_degraded_collection",
            "clear_ready": True,
            "clear_blockers": [],
            "safe_clear": {
                "ready": True,
                "hard_blockers": [],
                "degraded_blockers": ["collector_contracts"],
            },
        },
    )

    payload = src.build_payload(
        tmp_path,
        today=date(2026, 5, 11),
        registry_path=tmp_path / "master_bot_registry.json",
    )

    assert payload["stage"] == "armed"
    assert "global_halt_or_clear_blocker_active" not in payload["blockers"]
    assert payload["gates"]["global_halt"]["ok"] is True
    assert payload["gates"]["global_halt"]["status"] == "ready"
    assert payload["gates"]["global_halt"]["halt_posture"] == "unlatched_degraded_collection"


def test_paper_400_ramp_uses_system_plumbing_write_path_relief(tmp_path: Path) -> None:
    _seed_ready_project(tmp_path)
    health = tmp_path / "governance" / "health"
    _write_json(
        health / "global_halt_auto_clear_latest.json",
        {
            "halt": False,
            "clear_ready": False,
            "clear_blockers": ["write_path_recovery_pending"],
            "metrics": {"execution_expected": False},
        },
    )
    _write_json(
        health / "data_plane_recovery_controller_latest.json",
        {
            "overall_status": "degraded",
            "recovery_state": "recovering_under_guard",
            "write_failure_count": 6,
            "raw_write_failure_count": 6,
            "account_snapshot_failure_count": 0,
            "queue_depth": 2172,
        },
    )
    _write_json(
        health / "system_plumbing_control_latest.json",
        {
            "ok": True,
            "overall_status": "ready",
            "plumbing_score": 94,
            "global_clear_relief": {
                "active": True,
                "bounded_write_recovery": True,
                "advisory_clear_blockers": ["write_path_recovery_pending"],
            },
            "paper_ramp_relief_contract": {"bounded_write_recovery": True},
        },
    )

    payload = src.build_payload(
        tmp_path,
        today=date(2026, 5, 11),
        registry_path=tmp_path / "master_bot_registry.json",
    )

    assert payload["stage"] == "armed"
    assert "global_halt_or_clear_blocker_active" not in payload["blockers"]
    relief = payload["gates"]["global_halt"]["clear_blocker_relief"]
    assert relief["active"] is True
    assert relief["bounded_write_recovery"] is True
    assert relief["plumbing_bounded_write_recovery"] is True


def test_paper_400_ramp_treats_isolated_read_only_restart_storm_as_advisory(tmp_path: Path) -> None:
    _seed_ready_project(tmp_path)
    health = tmp_path / "governance" / "health"
    _write_json(
        health / "global_killswitch_latest.json",
        {
            "halt": False,
            "clear_ready": False,
            "clear_blockers": ["restart_storm_active"],
            "metrics": {
                "execution_expected": False,
                "restart_storm_isolation": {
                    "isolated_count": 1,
                    "execution_blocking_count": 0,
                    "isolated_targets": ["all_sleeves"],
                    "execution_blocking_targets": [],
                    "safe_to_clear_when_not_executing": True,
                },
            },
        },
    )

    payload = src.build_payload(
        tmp_path,
        today=date(2026, 5, 11),
        registry_path=tmp_path / "master_bot_registry.json",
    )

    assert "global_halt_or_clear_blocker_active" not in payload["blockers"]
    assert payload["gates"]["global_halt"]["ok"] is True
    assert payload["gates"]["global_halt"]["status"] == "restart_storm_isolation_advisory"
    assert payload["gates"]["global_halt"]["clear_blocker_relief"]["isolated_restart_storm"] is True
    assert payload["stage"] == "armed"


def _seed_paper_roster_candidates(project_root: Path) -> None:
    _seed_ready_project(project_root, bot_count=700)
    _write_json(
        project_root / "master_bot_registry.json",
        {
            "sub_bots": [
                {
                    "bot_id": f"brain_refinery_v{i}_paper_roster_candidate",
                    "active": True,
                    "paper_trade_enabled": i < 35,
                    "paper_live_data_enabled": i < 35,
                    "lifecycle_state": "active" if i < 35 else "data_collection_only",
                    "quality_score": 0.7 if i >= 35 else 0.9,
                    "data_collection_observations": 1500 + i,
                    "minimum_training_observations": 1000,
                    "data_collection_training_ready": i >= 35,
                    "label_contract": {"version": "test"},
                }
                for i in range(700)
            ]
        },
    )


def test_paper_400_ramp_promotes_guarded_roster_to_candidate_by_default(tmp_path: Path, monkeypatch) -> None:
    _seed_paper_roster_candidates(tmp_path)
    monkeypatch.setattr(src, "SOURCE_REGISTRY_PATH", tmp_path / "master_bot_registry.json")
    source_before = (tmp_path / "master_bot_registry.json").read_text(encoding="utf-8")

    promotion = src.promote_paper_roster(
        tmp_path,
        tmp_path / "master_bot_registry.json",
        target=400,
        candidate_registry_path=tmp_path / "governance" / "health" / "paper_400_ramp_registry_candidate_latest.json",
        source_write_guard_path=tmp_path / "governance" / "health" / "paper_400_ramp_source_write_guard_latest.json",
    )
    registry = json.loads((tmp_path / "master_bot_registry.json").read_text(encoding="utf-8"))
    candidate_path = Path(promotion["candidate_registry_path"])
    candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
    promoted = [
        row
        for row in candidate["sub_bots"]
        if row.get("paper_standard_cohort") == src.PAPER_400_PROMOTION_COHORT
    ]

    assert promotion["overall_status"] == "applied"
    assert promotion["registry_source_write_blocked"] is True
    assert promotion["registry_source_written"] is False
    assert promotion["paper_count_after"] == 400
    assert promotion["live_execution_locked"] is True
    assert (tmp_path / "master_bot_registry.json").read_text(encoding="utf-8") == source_before
    assert not [
        row
        for row in registry["sub_bots"]
        if row.get("paper_standard_cohort") == src.PAPER_400_PROMOTION_COHORT
    ]
    assert len(promoted) == 365
    assert all(row["paper_trade_enabled"] is True for row in promoted)
    assert all(row["live_trading_enabled"] is False for row in promoted)
    assert Path(promotion["source_write_guard_path"]).exists()


def test_paper_400_ramp_promotes_guarded_roster_to_source_only_when_allowed(tmp_path: Path) -> None:
    _seed_paper_roster_candidates(tmp_path)

    promotion = src.promote_paper_roster(
        tmp_path,
        tmp_path / "master_bot_registry.json",
        target=400,
        allow_source_registry_write=True,
    )
    payload = src.build_payload(
        tmp_path,
        today=date(2026, 5, 11),
        registry_path=tmp_path / "master_bot_registry.json",
    )
    registry = json.loads((tmp_path / "master_bot_registry.json").read_text(encoding="utf-8"))
    promoted = [
        row
        for row in registry["sub_bots"]
        if row.get("paper_standard_cohort") == src.PAPER_400_PROMOTION_COHORT
    ]

    assert promotion["overall_status"] == "applied"
    assert promotion["registry_source_write_blocked"] is False
    assert promotion["registry_source_written"] is True
    assert promotion["paper_count_after"] == 400
    assert promotion["live_execution_locked"] is True
    assert len(promoted) == 365
    assert all(row["paper_trade_enabled"] is True for row in promoted)
    assert all(row["live_trading_enabled"] is False for row in promoted)
    assert payload["stage"] == "armed"
    assert payload["gates"]["runtime"]["paper_roster_ready"] is True


def test_paper_400_ramp_removes_high_caps_when_memory_blocks(tmp_path: Path) -> None:
    _seed_ready_project(tmp_path)
    _write_json(
        tmp_path / "governance" / "health" / "memory_efficiency_control_latest.json",
        {
            "overall_status": "blocked",
            "recommended_profile": "constrained",
            "memory_snapshot": {
                "memory_pressure_state": "red",
                "memory_free_pct": 8.0,
                "swap_used_gb": 13.0,
                "compressed_store_gb": 31.0,
                "compressor_gb": 17.0,
            },
        },
    )

    payload = src.build_payload(
        tmp_path,
        today=date(2026, 5, 11),
        registry_path=tmp_path / "master_bot_registry.json",
    )
    payload = src.apply_payload(
        tmp_path,
        payload,
        out_path=tmp_path / "governance" / "health" / "paper_400_ramp_latest.json",
        override_path=tmp_path / "config" / ".env.paper_400_ramp_override",
    )

    override_text = (tmp_path / "config" / ".env.paper_400_ramp_override").read_text(encoding="utf-8")
    assert payload["stage"] == "blocked"
    assert payload["armed"] is False
    assert "memory_pressure_above_paper_400_gate" in payload["blockers"]
    assert "SCHWAB_TOP_BOT_PAPER_TRADING_TOP_N" not in override_text
    assert "PAPER_400_RAMP_ARMED=0" in override_text


def test_paper_400_ramp_counts_only_explicit_paper_live_data_bots(tmp_path: Path) -> None:
    _write_json(
        tmp_path / "master_bot_registry.json",
        {
            "sub_bots": [
                {
                    "bot_id": "collector_stability_only",
                    "active": True,
                    "lifecycle_state": "data_collection_only",
                    "paper_runtime_stability_mode": "full_force_guarded",
                },
                {
                    "bot_id": "legacy_paper_live_data",
                    "active": True,
                    "lifecycle_state": "active",
                    "paper_live_data_enabled": True,
                },
                {
                    "bot_id": "legacy_paper_trade",
                    "active": True,
                    "lifecycle_state": "active",
                    "paper_trade_enabled": True,
                },
                {
                    "bot_id": "inactive_paper",
                    "active": False,
                    "lifecycle_state": "inactive",
                    "paper_live_data_enabled": True,
                },
            ]
        },
    )

    counts = src._registry_counts(tmp_path, tmp_path / "master_bot_registry.json")

    assert counts["active_bot_count"] == 3
    assert counts["paper_tagged_count"] == 2
