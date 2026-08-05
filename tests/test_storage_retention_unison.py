from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from scripts.ops import storage_retention_unison as src


def test_storage_growth_forecast_excludes_pre_containment_incident_slope() -> None:
    now = datetime.now(timezone.utc)
    epoch = now - timedelta(minutes=20)
    history = [
        {
            "timestamp_utc": (now - timedelta(hours=2)).isoformat(),
            "disk": {"external": {"free_gb": 300.0}},
        },
        {
            "timestamp_utc": (now - timedelta(minutes=10)).isoformat(),
            "disk": {"external": {"free_gb": 140.0}},
        },
    ]

    payload = src._storage_growth_forecast(
        current_external={"free_gb": 139.0},
        current_internal={"free_gb": 100.0},
        history_rows=history,
        target_free_gb=125.0,
        pressure_free_gb=64.0,
        baseline_not_before_utc=epoch,
    )

    assert payload["baseline_scope"] == "post_hot_lane_control_epoch"
    assert payload["discarded_pre_control_samples"] == 1
    assert payload["baseline"]["external_free_gb"] == 140.0
    assert payload["consumed_gb_per_day"] < 200.0


def test_video_cold_archive_override_keeps_video_root_protected(monkeypatch) -> None:
    monkeypatch.setenv("BOT_ALLOW_VIDEO_COLD_ARCHIVE", "1")
    monkeypatch.setenv("BOT_VIDEO_COLD_ARCHIVE_ROOT", "/Volumes/VIDEO/schwab_trading_bot_cold")

    assert src._is_protected_volume(Path("/Volumes/VIDEO")) is True
    assert src._is_protected_volume(Path("/Volumes/VIDEO/schwab_trading_bot_cold")) is False
    assert src._is_protected_volume(Path("/Volumes/VIDEO/schwab_trading_bot_cold/data/proof.jsonl")) is False


def test_unconfigured_protected_candidate_does_not_claim_target_points_there(monkeypatch, tmp_path: Path) -> None:
    video_candidate = tmp_path / "VIDEO" / "schwab_trading_bot_cold"
    monkeypatch.delenv("BOT_SECOND_COLD_ROOT", raising=False)
    monkeypatch.setattr(src, "DEFAULT_SECOND_COLD_CANDIDATES", (str(tmp_path / "BOT_COLD"), str(video_candidate)))
    monkeypatch.setattr(src, "_is_protected_volume", lambda path: Path(path) == video_candidate)
    monkeypatch.setattr(src, "_disk_snapshot", lambda path: {"free_gb": 0.0, "used_percent": 0.0})

    payload = src._second_cold_preflight()

    assert payload["configured_path"] == ""
    assert payload["status"] == "prewired_waiting_for_drive"
    assert payload["score"] == 96.0
    assert any(row["protected"] for row in payload["candidates"])


def test_configured_protected_second_cold_target_remains_blocked(monkeypatch, tmp_path: Path) -> None:
    video_candidate = tmp_path / "VIDEO" / "schwab_trading_bot_cold"
    monkeypatch.setenv("BOT_SECOND_COLD_ROOT", str(video_candidate))
    monkeypatch.setattr(src, "_is_protected_volume", lambda path: Path(path) == video_candidate)
    monkeypatch.setattr(src, "_disk_snapshot", lambda path: {"free_gb": 0.0, "used_percent": 0.0})

    payload = src._second_cold_preflight()

    assert payload["configured_path"] == str(video_candidate)
    assert payload["status"] == "blocked_protected_target"
    assert payload["score"] == 50.0


def test_soak_storage_controls_treat_clean_optional_collector_intake_as_safe() -> None:
    controls = src._soak_storage_controls(
        forecast={"current_external_free_gb": 240.0},
        storage_payload={
            "storage_efficiency_contract": {
                "overall_status": "ready",
                "grade": "A+",
                "metrics": {"deep_cold_ready": True},
            },
            "steady_state": {"target_status": {"steady_state_ready": True}},
            "storage": {"retention_debt_gb": 0.0, "retention_debt_target_gb": 0.25},
            "collector_intake_enforcement_audit": {
                "status": "not_required",
                "required": False,
                "mismatch_count": 0,
            },
            "external_route_verification": {"verification_state": "ready"},
            "storage_resilience": {"overall_status": "ready"},
            "backlog_relief_contract": {"active": False},
        },
        quota_payload={"overall_status": "ready"},
        hot_lane_payload={"overall_status": "active"},
        target_free_gb=125.0,
        pressure_free_gb=64.0,
        safety_buffer_gb=32.0,
    )

    assert controls["collector_intake_status"] == "not_required"
    assert controls["collector_intake_soak_safe"] is True
    assert controls["collector_intake_enforced"] is True
    assert controls["storage_governed_core_ready"] is True
    assert controls["storage_bounded_post_maintenance_ready"] is True


def test_continuous_run_contract_blocks_when_projected_margin_is_negative() -> None:
    payload = src._continuous_run_contract(
        forecast={
            "status": "forecast_ready",
            "confidence": "sustained",
            "current_external_free_gb": 100.0,
            "sustained_consumed_gb_per_day": 2.0,
            "burst_consumed_gb_per_day": 0.5,
            "days_until_pressure_free": 18.0,
        },
        horizon_days=28.0,
        pressure_free_gb=64.0,
        safety_buffer_gb=16.0,
        min_daily_growth_gb=0.5,
    )

    assert payload["status"] == "blocked"
    assert payload["ready"] is False
    assert "insufficient_projected_free_space" in payload["blockers"]
    assert "forecast_pressure_inside_horizon" in payload["blockers"]
    assert payload["control_env"]["BOT_CONTINUOUS_COLLECTION_READY"] == "0"


def test_continuous_run_contract_ready_with_sustained_margin() -> None:
    payload = src._continuous_run_contract(
        forecast={
            "status": "forecast_ready",
            "confidence": "sustained",
            "current_external_free_gb": 240.0,
            "sustained_consumed_gb_per_day": 1.0,
            "burst_consumed_gb_per_day": 0.2,
            "days_until_pressure_free": 176.0,
        },
        horizon_days=28.0,
        pressure_free_gb=64.0,
        safety_buffer_gb=32.0,
        min_daily_growth_gb=0.5,
    )

    assert payload["status"] == "ready"
    assert payload["ready"] is True
    assert payload["available_margin_gb"] == 116.0
    assert payload["control_env"]["BOT_CONTINUOUS_COLLECTION_READY"] == "1"


def test_cold_archive_spillover_clears_projection_only_storage_blocker() -> None:
    blocked = src._continuous_run_contract(
        forecast={
            "status": "target_floor_breach",
            "confidence": "sustained",
            "current_external_free_gb": 105.0,
            "sustained_consumed_gb_per_day": 0.0,
            "burst_consumed_gb_per_day": 0.0,
            "days_until_pressure_free": 82.0,
        },
        horizon_days=30.0,
        pressure_free_gb=64.0,
        safety_buffer_gb=32.0,
        min_daily_growth_gb=0.5,
    )

    payload = src._apply_cold_archive_spillover_contract(
        blocked,
        {
            "ready": True,
            "candidates": [{"ready": True, "free_gb": 786.0}],
        },
    )

    assert blocked["status"] == "blocked"
    assert payload["ready"] is True
    assert payload["blockers"] == []
    assert payload["managed_blockers"] == ["insufficient_projected_free_space", "forecast_status_target_floor_breach"]
    assert payload["cold_archive_adjusted_margin_gb"] > 0.0
    assert payload["control_env"]["BOT_COLD_ARCHIVE_SPILLOVER_READY"] == "1"


def test_cold_archive_spillover_does_not_hide_primary_pressure_risk() -> None:
    blocked = src._continuous_run_contract(
        forecast={
            "status": "forecast_ready",
            "confidence": "sustained",
            "current_external_free_gb": 70.0,
            "sustained_consumed_gb_per_day": 0.0,
            "burst_consumed_gb_per_day": 0.0,
            "days_until_pressure_free": 12.0,
        },
        horizon_days=30.0,
        pressure_free_gb=64.0,
        safety_buffer_gb=32.0,
        min_daily_growth_gb=0.5,
    )

    payload = src._apply_cold_archive_spillover_contract(
        blocked,
        {
            "ready": True,
            "candidates": [{"ready": True, "free_gb": 786.0}],
        },
    )

    assert payload["status"] == "blocked"
    assert "insufficient_projected_free_space" in payload["blockers"]


def test_cold_archive_spillover_capacity_uses_live_headroom_without_fixed_credit_cap(monkeypatch) -> None:
    monkeypatch.setenv("BOT_COLD_ARCHIVE_RESERVE_GB", "64")
    monkeypatch.delenv("BOT_COLD_ARCHIVE_SPILLOVER_MAX_CREDIT_GB", raising=False)

    capacity = src._cold_archive_spillover_capacity_gb(
        {
            "ready": True,
            "candidates": [{"ready": True, "free_gb": 468.208}],
        }
    )

    assert capacity == 404.208


def test_cold_archive_spillover_reports_capacity_shortfall_without_hiding_blocker(monkeypatch) -> None:
    monkeypatch.setenv("BOT_COLD_ARCHIVE_RESERVE_GB", "64")
    monkeypatch.delenv("BOT_COLD_ARCHIVE_SPILLOVER_MAX_CREDIT_GB", raising=False)
    continuous = {
        "status": "blocked",
        "ready": False,
        "score": 72.0,
        "grade": "D",
        "current_external_free_gb": 136.0,
        "pressure_free_gb": 64.0,
        "available_margin_gb": -500.0,
        "blockers": ["insufficient_projected_free_space", "forecast_pressure_inside_horizon"],
        "warnings": [],
        "control_env": {},
    }

    payload = src._apply_cold_archive_spillover_contract(
        continuous,
        {
            "ready": True,
            "candidates": [{"ready": True, "free_gb": 468.0}],
        },
    )

    assert payload["ready"] is False
    assert payload["cold_archive_spillover_available"] is True
    assert payload["cold_archive_spillover_ready"] is False
    assert payload["cold_archive_spillover_status"] == "insufficient_capacity_for_horizon"
    assert payload["cold_archive_spillover_capacity_gb"] == 404.0
    assert payload["cold_archive_required_spillover_gb"] == 500.0
    assert payload["cold_archive_capacity_shortfall_gb"] == 96.0
    assert payload["blockers"] == continuous["blockers"]
    assert payload["control_env"]["BOT_COLD_ARCHIVE_SPILLOVER_READY"] == "0"


def test_continuous_run_contract_uses_sustained_growth_with_burst_watch() -> None:
    payload = src._continuous_run_contract(
        forecast={
            "status": "forecast_ready",
            "confidence": "sustained",
            "current_external_free_gb": 404.0,
            "sustained_consumed_gb_per_day": 6.7,
            "burst_consumed_gb_per_day": 18.6,
            "days_until_pressure_free": 50.0,
        },
        horizon_days=28.0,
        pressure_free_gb=64.0,
        safety_buffer_gb=32.0,
        min_daily_growth_gb=0.5,
    )

    assert payload["status"] == "watch"
    assert payload["ready"] is True
    assert payload["blockers"] == []
    assert "burst_growth_above_sustained_rate" in payload["warnings"]
    assert payload["effective_daily_growth_gb"] == 6.7
    assert payload["available_margin_gb"] == 120.4


def test_continuous_run_contract_ignores_noisy_burst_when_sustained_growth_is_flat() -> None:
    payload = src._continuous_run_contract(
        forecast={
            "status": "forecast_ready",
            "confidence": "sustained",
            "current_external_free_gb": 175.0,
            "sustained_consumed_gb_per_day": 0.0,
            "burst_consumed_gb_per_day": 49.0,
            "days_until_pressure_free": None,
        },
        horizon_days=30.0,
        pressure_free_gb=64.0,
        safety_buffer_gb=32.0,
        min_daily_growth_gb=0.5,
    )

    assert payload["status"] == "ready"
    assert payload["ready"] is True
    assert payload["blockers"] == []
    assert payload["effective_daily_growth_gb"] == 0.5
    assert payload["required_external_free_gb"] == 111.0
    assert payload["available_margin_gb"] == 64.0
    assert payload["control_env"]["BOT_CONTINUOUS_COLLECTION_READY"] == "1"


def test_continuous_run_contract_does_not_hard_block_on_short_window_high_growth() -> None:
    payload = src._continuous_run_contract(
        forecast={
            "status": "near_pressure",
            "confidence": "sustained",
            "elapsed_days": 0.0214,
            "current_external_free_gb": 175.0,
            "sustained_consumed_gb_per_day": 47.5,
            "burst_consumed_gb_per_day": 39.8,
            "days_until_pressure_free": 2.34,
        },
        horizon_days=30.0,
        pressure_free_gb=64.0,
        safety_buffer_gb=32.0,
        min_daily_growth_gb=0.5,
    )

    assert payload["status"] == "watch"
    assert payload["ready"] is True
    assert payload["blockers"] == []
    assert "growth_rate_window_too_short_for_30_day_projection" in payload["warnings"]
    assert payload["effective_daily_growth_gb"] == 0.5
    assert payload["required_external_free_gb"] == 111.0
    assert payload["control_env"]["BOT_CONTINUOUS_COLLECTION_READY"] == "1"


def test_continuous_run_contract_reclassifies_safe_short_window_watch_when_controls_are_green() -> None:
    payload = src._continuous_run_contract(
        forecast={
            "status": "near_pressure",
            "confidence": "sustained",
            "elapsed_days": 0.0214,
            "current_external_free_gb": 148.0,
            "sustained_consumed_gb_per_day": 47.5,
            "burst_consumed_gb_per_day": 39.8,
            "days_until_pressure_free": 2.34,
        },
        horizon_days=30.0,
        pressure_free_gb=64.0,
        safety_buffer_gb=32.0,
        min_daily_growth_gb=0.5,
        storage_controls={
            "storage_efficiency_ready": True,
            "quota_ready": True,
            "route_verified": True,
            "resilience_ready": True,
            "steady_state_ready": True,
            "retention_debt_ok": True,
            "collector_intake_enforced": True,
            "manifest_first_storage": False,
            "raw_candidate_compaction_ok": True,
            "sparse_large_line_pending_bounded": True,
            "deep_cold_ready": True,
            "hot_lane_retention_active": True,
            "external_free_above_target": True,
        },
    )

    assert payload["status"] == "ready"
    assert payload["ready"] is True
    assert payload["short_window_warning_reclassified_ready"] is True
    assert payload["warnings"] == ["growth_rate_window_too_short_for_30_day_projection"]
    assert payload["available_margin_gb"] == 37.0
    assert payload["controlled_days_until_pressure_free"] > 30.0 * 2.0
    assert payload["control_env"]["BOT_CONTINUOUS_COLLECTION_READY"] == "1"


def test_continuous_run_contract_accounts_for_collection_duty_cycle() -> None:
    payload = src._continuous_run_contract(
        forecast={
            "status": "forecast_ready",
            "confidence": "sustained",
            "elapsed_days": 0.5774,
            "current_external_free_gb": 530.524,
            "sustained_consumed_gb_per_day": 52.8676,
            "burst_consumed_gb_per_day": 52.8676,
            "days_until_pressure_free": 8.82,
        },
        horizon_days=30.0,
        pressure_free_gb=64.0,
        safety_buffer_gb=32.0,
        min_daily_growth_gb=0.5,
        duty_cycle_max_active_ratio=0.16,
    )

    assert payload["status"] == "watch"
    assert payload["ready"] is True
    assert payload["blockers"] == []
    assert "collection_duty_cycle_controls_growth_projection" in payload["warnings"]
    assert payload["raw_effective_daily_growth_gb"] == 52.8676
    assert payload["effective_daily_growth_gb"] == 8.4588
    assert payload["controlled_days_until_pressure_free"] > 30.0
    assert payload["control_env"]["BOT_CONTINUOUS_COLLECTION_READY"] == "1"
    assert payload["control_env"]["BOT_COLLECTION_DUTY_CYCLE_MAX_ACTIVE_RATIO"] == "0.16"


def test_continuous_run_contract_uses_governed_projection_when_storage_controls_are_green() -> None:
    payload = src._continuous_run_contract(
        forecast={
            "status": "near_pressure",
            "confidence": "sustained",
            "elapsed_days": 0.278,
            "current_external_free_gb": 127.656,
            "target_free_gb": 125.0,
            "sustained_consumed_gb_per_day": 232.1926,
            "burst_consumed_gb_per_day": 232.1926,
            "days_until_pressure_free": 0.27,
        },
        horizon_days=30.0,
        pressure_free_gb=64.0,
        safety_buffer_gb=32.0,
        min_daily_growth_gb=0.5,
        duty_cycle_max_active_ratio=0.16,
        storage_controls={
            "storage_efficiency_ready": True,
            "quota_ready": True,
            "route_verified": True,
            "resilience_ready": True,
            "steady_state_ready": True,
            "retention_debt_ok": True,
            "collector_intake_enforced": True,
            "manifest_first_storage": True,
            "raw_candidate_compaction_ok": True,
            "sparse_large_line_pending_bounded": True,
            "deep_cold_ready": True,
            "hot_lane_retention_active": True,
            "external_free_above_target": True,
        },
    )

    assert payload["status"] == "watch"
    assert payload["ready"] is True
    assert payload["blockers"] == []
    assert payload["raw_effective_daily_growth_gb"] == 232.1926
    assert payload["effective_daily_growth_gb"] == 0.5
    assert payload["required_external_free_gb"] == 111.0
    assert payload["available_margin_gb"] == 16.656
    assert payload["storage_governed_projection"] is True
    assert "storage_governed_controls_override_short_slope" in payload["warnings"]
    assert payload["control_env"]["BOT_CONTINUOUS_COLLECTION_READY"] == "1"


def test_continuous_run_contract_allows_bounded_post_maintenance_slope_without_manifest_first_storage() -> None:
    payload = src._continuous_run_contract(
        forecast={
            "status": "near_pressure",
            "confidence": "sustained",
            "elapsed_days": 0.4348,
            "current_external_free_gb": 149.638,
            "target_free_gb": 125.0,
            "sustained_consumed_gb_per_day": 45.683,
            "burst_consumed_gb_per_day": 45.683,
            "days_until_pressure_free": 1.81,
        },
        horizon_days=30.0,
        pressure_free_gb=64.0,
        safety_buffer_gb=32.0,
        min_daily_growth_gb=0.5,
        duty_cycle_max_active_ratio=0.16,
        storage_controls={
            "storage_efficiency_ready": True,
            "quota_ready": True,
            "route_verified": True,
            "resilience_ready": True,
            "steady_state_ready": True,
            "retention_debt_ok": True,
            "collector_intake_enforced": True,
            "manifest_first_storage": False,
            "raw_candidate_compaction_ok": True,
            "sparse_large_line_pending_bounded": True,
            "deep_cold_ready": True,
            "hot_lane_retention_active": True,
            "external_free_above_target": True,
        },
    )

    assert payload["status"] == "watch"
    assert payload["ready"] is True
    assert payload["blockers"] == []
    assert payload["raw_effective_daily_growth_gb"] == 45.683
    assert payload["effective_daily_growth_gb"] == 0.5
    assert payload["required_external_free_gb"] == 111.0
    assert payload["available_margin_gb"] == 38.638
    assert payload["storage_governed_control_ready"] is False
    assert payload["storage_bounded_control_ready"] is True
    assert payload["storage_bounded_projection"] is True
    assert payload["storage_projection_override"] is True
    assert "bounded_storage_controls_override_short_post_maintenance_slope" in payload["warnings"]
    assert "manifest_first_storage_pending" in payload["warnings"]
    assert payload["control_env"]["BOT_CONTINUOUS_COLLECTION_READY"] == "1"


def test_continuous_run_contract_does_not_allow_bounded_projection_below_required_free_space() -> None:
    payload = src._continuous_run_contract(
        forecast={
            "status": "near_pressure",
            "confidence": "sustained",
            "elapsed_days": 0.4348,
            "current_external_free_gb": 104.0,
            "target_free_gb": 125.0,
            "sustained_consumed_gb_per_day": 45.683,
            "burst_consumed_gb_per_day": 45.683,
            "days_until_pressure_free": 0.88,
        },
        horizon_days=30.0,
        pressure_free_gb=64.0,
        safety_buffer_gb=32.0,
        min_daily_growth_gb=0.5,
        duty_cycle_max_active_ratio=0.16,
        storage_controls={
            "storage_efficiency_ready": True,
            "quota_ready": True,
            "route_verified": True,
            "resilience_ready": True,
            "steady_state_ready": True,
            "retention_debt_ok": True,
            "collector_intake_enforced": True,
            "manifest_first_storage": False,
            "raw_candidate_compaction_ok": True,
            "sparse_large_line_pending_bounded": True,
            "deep_cold_ready": True,
            "hot_lane_retention_active": True,
            "external_free_above_target": False,
        },
    )

    assert payload["status"] == "blocked"
    assert payload["ready"] is False
    assert "insufficient_projected_free_space" in payload["blockers"]
    assert "projected_below_pressure_floor" in payload["blockers"]
    assert payload["storage_bounded_projection"] is False
    assert payload["control_env"]["BOT_CONTINUOUS_COLLECTION_READY"] == "0"


def test_deep_cold_needs_data_without_candidates_is_advisory() -> None:
    step = {
        "returncode": 2,
        "overall_status": "needs_data",
        "payload": {"summary": {"candidate_count": 0, "candidate_gb": 0.0, "managed_count": 0}},
    }

    assert src._deep_cold_needs_data_is_advisory("retention_freshness_deep_cold", step) is True
    assert src._deep_cold_needs_data_is_advisory("retention_freshness_v2", step) is False


def test_storage_retention_unison_runs_hot_plane_compactors(monkeypatch, tmp_path: Path) -> None:
    external_root = tmp_path / "external" / "schwab_trading_bot"
    external_root.mkdir(parents=True)
    (tmp_path / "governance" / "health").mkdir(parents=True)
    second_cold = tmp_path / "VIDEO" / "schwab_trading_bot_cold"
    monkeypatch.setenv("BOT_ALLOW_VIDEO_COLD_ARCHIVE", "1")
    monkeypatch.setenv("BOT_VIDEO_COLD_ARCHIVE_ROOT", str(second_cold))
    forecast_path = tmp_path / "forecast.json"

    commands: list[list[str]] = []

    def fake_resolve_external_storage() -> SimpleNamespace:
        return SimpleNamespace(external_root=external_root)

    def fake_second_cold_preflight() -> dict[str, Any]:
        return {"status": "ready", "score": 100.0, "grade": "A+", "ready": True, "next_action": "ready"}

    def fake_run_json(command: list[str], *, cwd: Path, timeout_sec: int) -> dict[str, Any]:
        commands.append(list(command))
        name = command[1]
        payload: dict[str, Any]
        if name == "deep-cold-storage-layer":
            published_forecast = json.loads(forecast_path.read_text(encoding="utf-8"))
            assert published_forecast.get("timestamp_utc")
            assert "current_external_free_gb" in published_forecast
            payload = {"ok": True, "overall_status": "ready", "manifest_path": "manifest.json"}
        elif name == "cold-archive-compactor":
            payload = {
                "ok": True,
                "overall_status": "applied",
                "archive_root": str(second_cold),
                "manifest_path": str(second_cold / "cold_archive_compaction_manifest.jsonl"),
                "readme_path": str(second_cold / "COLD_ARCHIVE_README.txt"),
                "summary": {
                    "jsonl_candidate_count": 3,
                    "selected_jsonl_count": 2,
                    "gzip_finalize_candidate_count": 4,
                    "selected_gzip_finalize_count": 4,
                    "tmp_duplicate_candidate_count": 1,
                    "sqlite_inventory_count": 66,
                    "sqlite_vacuum_eligible_count": 0,
                    "successful_action_count": 3,
                    "error_count": 0,
                    "released_gb": 4.25,
                },
                "next_action": "run another bounded wave if candidates remain",
            }
        elif name == "retention-intelligence-v2":
            payload = {
                "ok": True,
                "overall_status": "ready",
                "retention_report_card": {"overall_score": 99.0, "overall_grade": "A+"},
            }
        elif name == "raw-training-compaction":
            payload = {
                "ok": True,
                "overall_status": "ready",
                "raw_summary": {
                    "raw_jsonl_count": 4,
                    "eligible_training_source_count": 2,
                    "compression_candidate_gb": 3.0,
                    "raw_gb_cleared": 1.5,
                },
                "decision_packet": {"blocked_reasons": ["raw_compaction_not_applied"]},
                "next_training_manifest": {"raw_source_queue_path": "raw.jsonl", "raw_eligible_source_queue_path": "eligible.jsonl"},
            }
        elif name == "bot-logs-cleanup-intelligence":
            payload = {"ok": True, "overall_status": "ready", "projected_free_gb": 140.0, "selected_count": 0}
        elif name == "governance-telemetry-compactor":
            if "--project-root" in command:
                payload = {
                    "ok": True,
                    "overall_status": "applied",
                    "summary": {"candidate_count": 2, "selected_gb": 5.0, "estimated_hot_reduction_gb": 4.0},
                }
            else:
                payload = {
                    "ok": True,
                    "overall_status": "applied",
                    "summary": {"candidate_count": 4, "selected_gb": 11.0, "estimated_hot_reduction_gb": 9.0},
                }
        elif name == "governance-lifecycle-compactor":
            payload = {
                "ok": True,
                "overall_status": "applied",
                "summary": {"candidate_count": 8, "selected_gb": 3.0, "estimated_reduction_gb": 2.4},
            }
        elif name == "decision-log-compactor":
            payload = {
                "ok": True,
                "overall_status": "nothing_to_do",
                "summary": {"candidate_count": 0, "selected_gb": 0.0, "estimated_reduction_gb": 0.0},
            }
        elif name == "storage-tier-policy":
            payload = {
                "ok": True,
                "overall_status": "ready",
                "pressure": {"live_hot_path_bytes": 0},
                "manifest_backed_offload_contract": {
                    "status": "planned",
                    "manifest_path": str(tmp_path / "manifest.json"),
                    "eligible_offload_files": 3,
                    "eligible_offload_gb": 7.5,
                    "compaction_only_files": 2,
                    "compaction_only_gb": 10.25,
                    "delete_requires": ["verified_cold_copy", "sha256_match", "restore_probe", "retention_gate"],
                    "never_delete_classes": ["keep_hot_critical", "stateful_sql_compaction_only"],
                    "stateful_sql_policy": "checkpoint or mirror only",
                    "next_action": "use manifest for bounded offload",
                },
                "offload_manifest_summary": {"entry_count": 9, "omitted_count": 0},
            }
        elif name == "hot-lane-retention-control":
            payload = {"ok": True, "overall_status": "ready", "overall_score": 99.0, "mode": "watch", "reasons": []}
        elif name == "creative-cotenant-guard":
            payload = {"ok": True, "overall_status": "ready", "actions": [], "creative_mode": {}, "runtime_throttle": {}}
        elif name == "storage-quota-guard":
            payload = {"ok": True, "overall_status": "ready"}
        elif name == "ingestion-storage-control":
            payload = {"ok": True, "overall_status": "ready", "storage_efficiency_contract": {"grade": "A+"}}
        else:
            payload = {"ok": True, "overall_status": "ready"}
        return {
            "command": list(command),
            "returncode": 0,
            "timed_out": False,
            "ok": True,
            "overall_status": str(payload.get("overall_status") or "ready"),
            "payload": payload,
            "stdout_tail": "",
            "stderr_tail": "",
        }

    monkeypatch.setattr(src, "resolve_external_storage", fake_resolve_external_storage)
    monkeypatch.setattr(src, "_second_cold_preflight", fake_second_cold_preflight)
    monkeypatch.setattr(src, "_run_json", fake_run_json)

    payload = src.build_payload(
        tmp_path,
        apply=True,
        pressure_free_gb=999999.0,
        cleanup_max_delete_gb=22.0,
        telemetry_max_gb=11.0,
        lifecycle_max_gb=3.0,
        decision_max_gb=5.0,
        cold_archive_max_files=6,
        cold_archive_max_gb=9.0,
        cold_archive_min_age_hours=36.0,
        cold_archive_compression_level=4,
        out_path=tmp_path / "unison.json",
        history_path=tmp_path / "history.jsonl",
        forecast_path=forecast_path,
    )

    command_names = [row[1] for row in commands]
    deep_cold_command = commands[command_names.index("deep-cold-storage-layer")]
    cold_archive_command = commands[command_names.index("cold-archive-compactor")]
    assert "governance-telemetry-compactor" in command_names
    assert "governance-lifecycle-compactor" in command_names
    assert "decision-log-compactor" in command_names
    assert "--move-to-second-cold" in deep_cold_command
    assert "--adaptive" in deep_cold_command
    assert deep_cold_command[deep_cold_command.index("--planning-horizon-days") + 1] == "30.0"
    assert str(second_cold) in deep_cold_command
    assert "--apply" in cold_archive_command
    assert cold_archive_command[cold_archive_command.index("--archive-root") + 1] == str(second_cold)
    assert cold_archive_command[cold_archive_command.index("--max-files") + 1] == "6"
    assert cold_archive_command[cold_archive_command.index("--max-raw-gb") + 1] == "9.0"
    assert cold_archive_command[cold_archive_command.index("--min-age-hours") + 1] == "36.0"
    assert cold_archive_command[cold_archive_command.index("--compression-level") + 1] == "4"
    assert "--allow-active-writer" not in cold_archive_command
    assert "--coordinate-writer-handoff" in cold_archive_command
    telemetry_command = commands[command_names.index("governance-telemetry-compactor")]
    telemetry_commands = [row for row in commands if row[1] == "governance-telemetry-compactor"]
    external_telemetry_command = [row for row in telemetry_commands if "--project-root" in row][0]
    lifecycle_command = commands[command_names.index("governance-lifecycle-compactor")]
    decision_command = commands[command_names.index("decision-log-compactor")]
    cleanup_command = commands[command_names.index("bot-logs-cleanup-intelligence")]
    assert "--apply" in telemetry_command
    assert "--apply" in external_telemetry_command
    assert str(external_root) in external_telemetry_command
    assert telemetry_command[telemetry_command.index("--target-free-gb") + 1] == "11.0"
    assert lifecycle_command[lifecycle_command.index("--target-free-gb") + 1] == "3.0"
    assert decision_command[decision_command.index("--target-free-gb") + 1] == "5.0"
    assert cleanup_command[cleanup_command.index("--max-tier") + 1] == "2"
    assert cleanup_command[cleanup_command.index("--max-delete-gb") + 1] == "22.0"
    assert payload["sections"]["hot_plane_compaction"]["status"] == "applied"
    assert payload["sections"]["hot_plane_compaction"]["evidence"]["estimated_reduction_gb"] == 15.4
    assert payload["sections"]["bot_logs_lean"]["evidence"]["effective_max_tier"] == 2
    assert payload["sections"]["manifest_backed_offload"]["status"] == "planned"
    assert payload["sections"]["manifest_backed_offload"]["evidence"]["eligible_offload_gb"] == 7.5
    assert payload["sections"]["cold_archive_compaction"]["status"] == "applied"
    assert payload["sections"]["cold_archive_compaction"]["evidence"]["released_gb"] == 4.25
    assert payload["sections"]["cold_archive_compaction"]["evidence"]["gzip_finalize_candidate_count"] == 4
    assert payload["sections"]["cold_archive_compaction"]["evidence"]["selected_gzip_finalize_count"] == 4
    assert payload["sections"]["cold_archive_compaction"]["evidence"]["sqlite_inventory_count"] == 66
    assert payload["integration_contract"]["compacts_hot_governance_telemetry"] is True
    assert payload["integration_contract"]["compacts_external_hot_governance_telemetry"] is True
    assert payload["integration_contract"]["compacts_lifecycle_registry_backups"] is True
    assert payload["integration_contract"]["compacts_old_decision_logs"] is True
    assert payload["integration_contract"]["uses_manifest_backed_offload_contract"] is True
    assert payload["integration_contract"]["has_manifest_backed_copy_verify_worker"] is True
    assert payload["integration_contract"]["stateful_sql_compaction_only"] is True
    assert payload["integration_contract"]["publishes_growth_forecast_before_deep_cold"] is True
    assert payload["integration_contract"]["compacts_cold_archive_losslessly"] is True
    assert payload["integration_contract"]["cold_archive_restore_proof_manifest"] is True
    assert payload["integration_contract"]["recovers_verified_cold_archive_gzip_orphans"] is True
    assert payload["integration_contract"]["coordinates_cold_archive_writer_handoff"] is True
    assert payload["integration_contract"]["preserves_direct_archive_readability"] is True
    assert payload["integration_contract"]["defers_cold_compaction_while_writer_active"] is True
    assert payload["control_env"]["BOT_MANIFEST_BACKED_OFFLOAD_CONTRACT_ACTIVE"] == "1"
    assert payload["control_env"]["BOT_COLD_ARCHIVE_COMPACTION_ACTIVE"] == "1"
    assert payload["recommended_commands"]["bounded_cold_archive_compaction_wave"][1] == "cold-archive-compactor"


def test_hot_plane_compaction_treats_lock_owner_as_in_progress() -> None:
    contract = src._hot_plane_compaction_contract(
        steps_by_lane={
            "governance_telemetry_compactor": {
                "returncode": 2,
                "overall_status": "busy",
                "payload": {"overall_status": "busy"},
            },
            "external_governance_telemetry_compactor": {
                "returncode": 0,
                "overall_status": "nothing_to_do",
                "payload": {"overall_status": "nothing_to_do", "summary": {}},
            },
        }
    )

    assert contract["status"] == "in_progress"
    assert contract["grade"] == "A+"
    assert contract["errors"] == []
    assert contract["busy_lanes"] == ["governance_telemetry_compactor"]


def test_storage_retention_unison_treats_foreground_advisory_as_non_hard(monkeypatch, tmp_path: Path) -> None:
    external_root = tmp_path / "external" / "schwab_trading_bot"
    external_root.mkdir(parents=True)
    (tmp_path / "governance" / "health").mkdir(parents=True)

    def fake_resolve_external_storage() -> SimpleNamespace:
        return SimpleNamespace(external_root=external_root)

    def fake_second_cold_preflight() -> dict[str, Any]:
        return {"status": "ready", "score": 100.0, "grade": "A+", "ready": True, "next_action": "ready"}

    def fake_run_json(command: list[str], *, cwd: Path, timeout_sec: int) -> dict[str, Any]:
        name = command[1]
        payload: dict[str, Any] = {"ok": True, "overall_status": "ready"}
        rc = 0
        ok = True
        if name == "retention-intelligence-v2":
            payload["retention_report_card"] = {"overall_score": 99.0, "overall_grade": "A+"}
        elif name == "raw-training-compaction":
            payload.update(
                {
                    "raw_summary": {"raw_jsonl_count": 4, "eligible_training_source_count": 2},
                    "decision_packet": {"blocked_reasons": []},
                }
            )
        elif name == "bot-logs-cleanup-intelligence":
            payload.update({"projected_free_gb": 140.0, "selected_count": 0})
        elif name == "storage-tier-policy":
            payload.update(
                {
                    "pressure": {"live_hot_path_bytes": 0},
                    "manifest_backed_offload_contract": {"status": "planned", "score": 99.0},
                    "offload_manifest_summary": {},
                }
            )
        elif name == "hot-lane-retention-control":
            payload.update({"overall_score": 99.0, "mode": "watch", "reasons": []})
        elif name == "creative-cotenant-guard":
            rc = 2
            ok = False
            payload = {
                "ok": False,
                "overall_status": "advisory",
                "actions": ["heavy_research_pause_active"],
                "creative_mode": {"active": True, "kind": "music_playback"},
                "runtime_throttle": {"overall_status": "ready"},
            }
        elif name == "ingestion-storage-control":
            payload["storage_efficiency_contract"] = {"grade": "A+"}
        return {
            "command": list(command),
            "returncode": rc,
            "timed_out": False,
            "ok": ok,
            "overall_status": str(payload.get("overall_status") or "ready"),
            "payload": payload,
            "stdout_tail": "",
            "stderr_tail": "",
        }

    monkeypatch.setattr(src, "resolve_external_storage", fake_resolve_external_storage)
    monkeypatch.setattr(src, "_second_cold_preflight", fake_second_cold_preflight)
    monkeypatch.setattr(src, "_run_json", fake_run_json)

    payload = src.build_payload(
        tmp_path,
        apply=True,
        target_free_gb=1.0,
        pressure_free_gb=1.0,
        cleanup_max_delete_gb=1.0,
        out_path=tmp_path / "unison.json",
        history_path=tmp_path / "history.jsonl",
        forecast_path=tmp_path / "forecast.json",
    )

    assert "foreground_app_protection" not in payload["command_failures"]
    assert "command_failed:foreground_app_protection" not in payload["hard_blockers"]
    assert payload["sections"]["foreground_protection"]["status"] == "advisory"
    assert payload["overall_status"] == "ready"


def test_storage_retention_unison_treats_foreground_timeout_as_advisory(monkeypatch, tmp_path: Path) -> None:
    external_root = tmp_path / "external" / "schwab_trading_bot"
    external_root.mkdir(parents=True)
    (tmp_path / "governance" / "health").mkdir(parents=True)

    def fake_resolve_external_storage() -> SimpleNamespace:
        return SimpleNamespace(external_root=external_root)

    def fake_second_cold_preflight() -> dict[str, Any]:
        return {"status": "ready", "score": 100.0, "grade": "A+", "ready": True, "next_action": "ready"}

    def fake_run_json(command: list[str], *, cwd: Path, timeout_sec: int) -> dict[str, Any]:
        name = command[1]
        payload: dict[str, Any] = {"ok": True, "overall_status": "ready"}
        rc = 0
        ok = True
        timed_out = False
        if name == "retention-intelligence-v2":
            payload["retention_report_card"] = {"overall_score": 99.0, "overall_grade": "A+"}
        elif name == "raw-training-compaction":
            payload.update(
                {
                    "raw_summary": {"raw_jsonl_count": 4, "eligible_training_source_count": 2},
                    "decision_packet": {"blocked_reasons": []},
                }
            )
        elif name == "bot-logs-cleanup-intelligence":
            payload.update({"projected_free_gb": 140.0, "selected_count": 0})
        elif name == "storage-tier-policy":
            payload.update(
                {
                    "pressure": {"live_hot_path_bytes": 0},
                    "manifest_backed_offload_contract": {"status": "planned", "score": 99.0},
                    "offload_manifest_summary": {},
                }
            )
        elif name == "hot-lane-retention-control":
            payload.update({"overall_score": 99.0, "mode": "watch", "reasons": []})
        elif name == "creative-cotenant-guard":
            rc = 124
            ok = False
            timed_out = True
            payload = {}
        elif name == "ingestion-storage-control":
            payload["storage_efficiency_contract"] = {"grade": "A+"}
        return {
            "command": list(command),
            "returncode": rc,
            "timed_out": timed_out,
            "ok": ok,
            "overall_status": str(payload.get("overall_status") or "error"),
            "payload": payload,
            "stdout_tail": "",
            "stderr_tail": "",
        }

    monkeypatch.setattr(src, "resolve_external_storage", fake_resolve_external_storage)
    monkeypatch.setattr(src, "_second_cold_preflight", fake_second_cold_preflight)
    monkeypatch.setattr(src, "_run_json", fake_run_json)

    payload = src.build_payload(
        tmp_path,
        apply=True,
        target_free_gb=1.0,
        pressure_free_gb=1.0,
        cleanup_max_delete_gb=1.0,
        out_path=tmp_path / "unison.json",
        history_path=tmp_path / "history.jsonl",
        forecast_path=tmp_path / "forecast.json",
    )

    assert "foreground_app_protection" not in payload["command_failures"]
    assert "command_failed:foreground_app_protection" not in payload["hard_blockers"]
    assert payload["sections"]["foreground_protection"]["status"] == "advisory"
    assert payload["sections"]["foreground_protection"]["evidence"]["actions"] == ["foreground_guard_timeout"]


def test_storage_retention_unison_accepts_degraded_quota_when_free_space_is_above_target(
    monkeypatch, tmp_path: Path
) -> None:
    external_root = tmp_path / "external" / "schwab_trading_bot"
    external_root.mkdir(parents=True)
    (tmp_path / "governance" / "health").mkdir(parents=True)

    def fake_resolve_external_storage() -> SimpleNamespace:
        return SimpleNamespace(external_root=external_root)

    def fake_second_cold_preflight() -> dict[str, Any]:
        return {"status": "ready", "score": 100.0, "grade": "A+", "ready": True, "next_action": "ready"}

    def fake_run_json(command: list[str], *, cwd: Path, timeout_sec: int) -> dict[str, Any]:
        name = command[1]
        payload: dict[str, Any] = {"ok": True, "overall_status": "ready"}
        if name == "retention-intelligence-v2":
            payload["retention_report_card"] = {"overall_score": 99.0, "overall_grade": "A+"}
        elif name == "raw-training-compaction":
            payload.update(
                {
                    "raw_summary": {
                        "raw_jsonl_count": 4,
                        "eligible_training_source_count": 2,
                        "compression_candidate_gb": 0.25,
                    },
                    "decision_packet": {"blocked_reasons": []},
                }
            )
        elif name == "bot-logs-cleanup-intelligence":
            payload.update({"overall_status": "ready", "projected_free_gb": 140.0, "selected_count": 0})
        elif name == "storage-tier-policy":
            payload.update(
                {
                    "pressure": {"live_hot_path_bytes": 0},
                    "manifest_backed_offload_contract": {"status": "planned", "score": 99.0},
                    "offload_manifest_summary": {},
                }
            )
        elif name == "hot-lane-retention-control":
            payload.update({"overall_score": 99.0, "mode": "watch", "reasons": []})
        elif name == "storage-quota-guard":
            payload = {
                "ok": False,
                "overall_status": "degraded",
                "quota_summary": {"external_free_below_target": False},
            }
        elif name == "ingestion-storage-control":
            payload.update(
                {
                    "storage_efficiency_contract": {
                        "overall_status": "ready",
                        "grade": "A+",
                        "raw_payload_policy": "manifest_first_compress_old_sources",
                        "metrics": {
                            "raw_compression_candidate_gb": 0.25,
                            "local_fallback_reconciliation_count": 0,
                            "sparse_large_line_pending_bytes": 0,
                            "deep_cold_ready": True,
                        },
                    },
                    "steady_state": {"target_status": {"steady_state_ready": True}},
                    "storage": {"retention_debt_gb": 0.0, "retention_debt_target_gb": 0.25},
                    "collector_intake_enforcement_audit": {"status": "enforced"},
                    "external_route_verification": {"verification_state": "ready", "coverage_ratio": 1.0},
                    "storage_resilience": {"overall_status": "ready"},
                }
            )
        return {
            "command": list(command),
            "returncode": 0,
            "timed_out": False,
            "ok": bool(payload.get("ok", True)),
            "overall_status": str(payload.get("overall_status") or "ready"),
            "payload": payload,
            "stdout_tail": "",
            "stderr_tail": "",
        }

    monkeypatch.setattr(src, "resolve_external_storage", fake_resolve_external_storage)
    monkeypatch.setattr(src, "_second_cold_preflight", fake_second_cold_preflight)
    monkeypatch.setattr(src, "_run_json", fake_run_json)

    payload = src.build_payload(
        tmp_path,
        apply=True,
        target_free_gb=1.0,
        pressure_free_gb=1.0,
        cleanup_max_delete_gb=1.0,
        out_path=tmp_path / "unison.json",
        history_path=tmp_path / "history.jsonl",
        forecast_path=tmp_path / "forecast.json",
    )

    assert payload["continuous_run_contract"]["storage_controls"]["quota_status"] == "degraded"
    assert payload["continuous_run_contract"]["storage_controls"]["quota_ready"] is True
    assert "storage_quota_not_ready" not in payload["hard_blockers"]
    assert payload["overall_status"] == "ready"
